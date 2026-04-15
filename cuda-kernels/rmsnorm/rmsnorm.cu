#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// ── Error check ───────────────────────────────────────────────────────────────
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t _e = (call);                                                \
        if (_e != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error %s:%d: %s\n",                          \
                    __FILE__, __LINE__, cudaGetErrorString(_e));                \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

// ─────────────────────────────────────────────────────────────────────────────
// RMSNorm definition:
//   rms = sqrt( mean(x^2) + eps )
//   y[i] = (x[i] / rms) * w[i]
//
// Each row of x is normalized independently.
// Input:  x [rows, hidden], w [hidden]   (FP32)
// Output: y [rows, hidden]               (FP32)
// ─────────────────────────────────────────────────────────────────────────────

// ── v1: naive — one block per row, direct global-memory accumulation ──────────
__global__ void rmsnorm_v1_naive(const float* __restrict__ x,
                                  const float* __restrict__ w,
                                  float*       __restrict__ y,
                                  int hidden, float eps) {
    int row = blockIdx.x;
    const float* xr = x + row * hidden;
    float*       yr = y + row * hidden;

    // Each thread sums a strided slice of x^2
    float sum = 0.0f;
    for (int i = threadIdx.x; i < hidden; i += blockDim.x)
        sum += xr[i] * xr[i];

    // Block-level reduction via shared memory
    extern __shared__ float smem[];
    smem[threadIdx.x] = sum;
    __syncthreads();

    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
        __syncthreads();
    }

    float rms_inv = rsqrtf(smem[0] / (float)hidden + eps);

    for (int i = threadIdx.x; i < hidden; i += blockDim.x)
        yr[i] = xr[i] * rms_inv * w[i];
}

// ── v2: warp shuffle + vectorized load (float4) ───────────────────────────────
// Uses __shfl_down_sync for the final warp reduction (no shared mem for that).
// Loads 4 floats per instruction where possible.
__global__ void rmsnorm_v2_warp(const float* __restrict__ x,
                                 const float* __restrict__ w,
                                 float*       __restrict__ y,
                                 int hidden, float eps) {
    int row = blockIdx.x;
    const float* xr = x + row * hidden;
    float*       yr = y + row * hidden;

    // ── Phase 1: compute sum(x^2) with warp shuffle ───────────────────────
    float sum = 0.0f;
    // Vectorized loop (hidden must be divisible by 4 for the fast path)
    int i = threadIdx.x * 4;
    for (; i + 3 < hidden; i += blockDim.x * 4) {
        float4 v = *reinterpret_cast<const float4*>(xr + i);
        sum += v.x*v.x + v.y*v.y + v.z*v.z + v.w*v.w;
    }
    // Scalar tail
    for (int j = threadIdx.x + (hidden / 4) * 4; j < hidden; j += blockDim.x)
        sum += xr[j] * xr[j];

    // Warp reduce
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, offset);

    // Cross-warp reduce via shared memory (one float per warp)
    extern __shared__ float smem[];
    int warp_id = threadIdx.x >> 5;
    int lane    = threadIdx.x & 31;
    if (lane == 0) smem[warp_id] = sum;
    __syncthreads();

    // First warp reduces partial sums
    int num_warps = blockDim.x >> 5;
    if (warp_id == 0) {
        sum = (lane < num_warps) ? smem[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1)
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        if (lane == 0) smem[0] = sum;
    }
    __syncthreads();

    float rms_inv = rsqrtf(smem[0] / (float)hidden + eps);

    // ── Phase 2: normalize + scale ────────────────────────────────────────
    i = threadIdx.x * 4;
    for (; i + 3 < hidden; i += blockDim.x * 4) {
        float4 vx = *reinterpret_cast<const float4*>(xr + i);
        float4 vw = *reinterpret_cast<const float4*>(w  + i);
        float4 vy;
        vy.x = vx.x * rms_inv * vw.x;
        vy.y = vx.y * rms_inv * vw.y;
        vy.z = vx.z * rms_inv * vw.z;
        vy.w = vx.w * rms_inv * vw.w;
        *reinterpret_cast<float4*>(yr + i) = vy;
    }
    for (int j = threadIdx.x + (hidden / 4) * 4; j < hidden; j += blockDim.x)
        yr[j] = xr[j] * rms_inv * w[j];
}

// ── v3: warp shuffle + float4 + register-cached x ─────────────────────────────
// Eliminates the redundant second read of x in phase 2 by caching it in
// registers between the two passes. Traffic: x (1x) + w (1x) + y (1x) = 3N
// vs v1/v2 which effectively do x (2x) + w + y = 4N (second x read hits L2
// but still consumes bandwidth). Expected speedup over v1: ~1.33x.
//
// Block size = 256. Each thread handles hidden / 256 elements via float4,
// cached in a small register array. For hidden ≤ 8192 this stays in regs.
__global__ void rmsnorm_v3_regcache(const float* __restrict__ x,
                                     const float* __restrict__ w,
                                     float*       __restrict__ y,
                                     int hidden, float eps) {
    constexpr int BLOCK = 256;
    constexpr int MAX_VEC_PER_THREAD = 8;  // supports hidden up to 256*4*8 = 8192

    int row = blockIdx.x;
    const float* xr = x + row * hidden;
    float*       yr = y + row * hidden;

    // Uniform vec_iters across all threads: vec region = [0, vec_iters*BLOCK*4).
    // Remainder in [vec_end, hidden) handled by scalar tail (executed by all
    // threads but only for indices they own). Eliminates double counting when
    // some threads would have nvec=0 while others have nvec>0.
    const int vec_iters = hidden / (BLOCK * 4);
    const int vec_end   = vec_iters * BLOCK * 4;

    float4 cache[MAX_VEC_PER_THREAD];
    float  sum = 0.0f;

    int i = threadIdx.x * 4;
    #pragma unroll
    for (int k = 0; k < MAX_VEC_PER_THREAD; k++) {
        if (k < vec_iters) {
            float4 v = *reinterpret_cast<const float4*>(xr + i);
            cache[k] = v;
            sum += v.x*v.x + v.y*v.y + v.z*v.z + v.w*v.w;
            i += BLOCK * 4;
        }
    }
    // Scalar tail for [vec_end, hidden). Re-read from global (not cached).
    for (int j = vec_end + threadIdx.x; j < hidden; j += BLOCK) {
        float v = xr[j];
        sum += v * v;
    }

    // Warp reduce
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, offset);

    extern __shared__ float smem[];
    int warp_id = threadIdx.x >> 5;
    int lane    = threadIdx.x & 31;
    if (lane == 0) smem[warp_id] = sum;
    __syncthreads();

    constexpr int NUM_WARPS = BLOCK >> 5;  // 8
    if (warp_id == 0) {
        sum = (lane < NUM_WARPS) ? smem[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1)
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        if (lane == 0) smem[0] = sum;
    }
    __syncthreads();

    float rms_inv = rsqrtf(smem[0] / (float)hidden + eps);

    // Phase 2: use cached x for the vec region, only read w and write y
    i = threadIdx.x * 4;
    #pragma unroll
    for (int k = 0; k < MAX_VEC_PER_THREAD; k++) {
        if (k < vec_iters) {
            float4 vx = cache[k];
            float4 vw = *reinterpret_cast<const float4*>(w + i);
            float4 vy;
            vy.x = vx.x * rms_inv * vw.x;
            vy.y = vx.y * rms_inv * vw.y;
            vy.z = vx.z * rms_inv * vw.z;
            vy.w = vx.w * rms_inv * vw.w;
            *reinterpret_cast<float4*>(yr + i) = vy;
            i += BLOCK * 4;
        }
    }
    // Tail: re-read x from global (not cached)
    for (int j = vec_end + threadIdx.x; j < hidden; j += BLOCK)
        yr[j] = xr[j] * rms_inv * w[j];
}

// ── CPU reference ─────────────────────────────────────────────────────────────
static void rmsnorm_cpu(const float* x, const float* w, float* y,
                         int rows, int hidden, float eps) {
    for (int r = 0; r < rows; r++) {
        const float* xr = x + r * hidden;
        float* yr = y + r * hidden;
        float sum = 0.0f;
        for (int i = 0; i < hidden; i++) sum += xr[i] * xr[i];
        float rms_inv = 1.0f / sqrtf(sum / (float)hidden + eps);
        for (int i = 0; i < hidden; i++) yr[i] = xr[i] * rms_inv * w[i];
    }
}

// ── Benchmark one kernel ──────────────────────────────────────────────────────
typedef void (*KernelFn)(const float*, const float*, float*, int, float);

static float bench(KernelFn fn, const char* name,
                   const float* d_x, const float* d_w, float* d_y,
                   int rows, int hidden, float eps,
                   int block_size, int smem_bytes) {
    dim3 grid(rows), block(block_size);

    // Warmup
    for (int i = 0; i < 3; i++)
        fn<<<grid, block, smem_bytes>>>(d_x, d_w, d_y, hidden, eps);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));

    float times[5];
    for (int r = 0; r < 5; r++) {
        CUDA_CHECK(cudaEventRecord(t0));
        fn<<<grid, block, smem_bytes>>>(d_x, d_w, d_y, hidden, eps);
        CUDA_CHECK(cudaEventRecord(t1));
        CUDA_CHECK(cudaEventSynchronize(t1));
        CUDA_CHECK(cudaEventElapsedTime(&times[r], t0, t1));
    }
    CUDA_CHECK(cudaEventDestroy(t0));
    CUDA_CHECK(cudaEventDestroy(t1));

    // Median
    for (int i = 0; i < 4; i++)
        for (int j = i+1; j < 5; j++)
            if (times[j] < times[i]) { float t = times[i]; times[i] = times[j]; times[j] = t; }
    float ms = times[2];

    // Read/write: x (read) + y (write) + w (read) = 3 * rows * hidden * 4 bytes
    float gb = 3.0f * rows * hidden * sizeof(float) / 1e9f;
    float gbs = gb / (ms / 1000.0f);

    printf("  %-18s  %8.3f ms  %7.1f GB/s\n", name, ms, gbs);
    return ms;
}

// ── Correctness check ─────────────────────────────────────────────────────────
static void check_correctness(const float* ref, const float* got,
                               int n, const char* label) {
    float max_err = 0.0f;
    for (int i = 0; i < n; i++)
        max_err = fmaxf(max_err, fabsf(ref[i] - got[i]));
    printf("  %-18s  max_abs_err = %e  %s\n",
           label, max_err, max_err < 1e-5f ? "PASS" : "FAIL !!!");
}

// ── Dump mode (invoked by test_correctness.py) ───────────────────────────────
// Reads raw binary input/weight, runs v2_warp, writes raw binary output.
// Supports --dtype fp32|fp16 (fp16: input/weight/output are __half arrays).

static void run_dump(int rows, int cols, bool use_fp16,
                     const char* in_path, const char* wt_path,
                     const char* out_path) {
    const float EPS = 1e-5f;
    // v2_warp precondition: float4 vectorized loads require cols divisible by 4
    if (cols % 4 != 0) {
        fprintf(stderr, "rmsnorm dump mode requires cols %% 4 == 0 (got %d)\n", cols);
        exit(1);
    }
    const size_t elem = (size_t)rows * cols;

    if (use_fp16) {
        using T = __half;
        size_t bytes  = elem * sizeof(T);
        size_t wbytes = (size_t)cols * sizeof(T);
        T* h_x = (T*)malloc(bytes);
        T* h_w = (T*)malloc(wbytes);
        T* h_y = (T*)malloc(bytes);

        FILE* f;
        f = fopen(in_path, "rb");
        if (!f) { fprintf(stderr, "Cannot open %s\n", in_path); exit(1); }
        if (fread(h_x, sizeof(T), elem, f) != elem) { fprintf(stderr, "Short read %s\n", in_path); exit(1); }
        fclose(f);
        f = fopen(wt_path, "rb");
        if (!f) { fprintf(stderr, "Cannot open %s\n", wt_path); exit(1); }
        if (fread(h_w, sizeof(T), (size_t)cols, f) != (size_t)cols) { fprintf(stderr, "Short read %s\n", wt_path); exit(1); }
        fclose(f);

        // v2_warp is FP32-only: convert FP16 input to FP32 on host,
        // run FP32 kernel, then convert output back to FP16.
        float* h_xf = (float*)malloc(elem * sizeof(float));
        float* h_wf = (float*)malloc((size_t)cols * sizeof(float));
        float* h_yf = (float*)malloc(elem * sizeof(float));
        for (size_t i = 0; i < elem;          i++) h_xf[i] = __half2float(h_x[i]);
        for (int    i = 0; i < cols;           i++) h_wf[i] = __half2float(h_w[i]);

        float *d_xf, *d_wf, *d_yf;
        CUDA_CHECK(cudaMalloc(&d_xf, elem * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_wf, (size_t)cols * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_yf, elem * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_xf, h_xf, elem * sizeof(float),          cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_wf, h_wf, (size_t)cols * sizeof(float),  cudaMemcpyHostToDevice));

        rmsnorm_v2_warp<<<rows, 128, 4 * sizeof(float)>>>(d_xf, d_wf, d_yf, cols, EPS);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h_yf, d_yf, elem * sizeof(float), cudaMemcpyDeviceToHost));

        // Convert fp32 result back to fp16 for output
        for (size_t i = 0; i < elem; i++) h_y[i] = __float2half(h_yf[i]);

        f = fopen(out_path, "wb");
        if (!f) { fprintf(stderr, "Cannot write %s\n", out_path); exit(1); }
        fwrite(h_y, sizeof(T), elem, f);
        fclose(f);

        CUDA_CHECK(cudaFree(d_xf)); CUDA_CHECK(cudaFree(d_wf)); CUDA_CHECK(cudaFree(d_yf));
        free(h_x); free(h_w); free(h_y); free(h_xf); free(h_wf); free(h_yf);
    } else {
        size_t bytes  = elem * sizeof(float);
        size_t wbytes = (size_t)cols * sizeof(float);
        float* h_x = (float*)malloc(bytes);
        float* h_w = (float*)malloc(wbytes);
        float* h_y = (float*)malloc(bytes);

        FILE* f;
        f = fopen(in_path, "rb");
        if (!f) { fprintf(stderr, "Cannot open %s\n", in_path); exit(1); }
        if (fread(h_x, sizeof(float), elem, f) != elem) { fprintf(stderr, "Short read %s\n", in_path); exit(1); }
        fclose(f);
        f = fopen(wt_path, "rb");
        if (!f) { fprintf(stderr, "Cannot open %s\n", wt_path); exit(1); }
        if (fread(h_w, sizeof(float), (size_t)cols, f) != (size_t)cols) { fprintf(stderr, "Short read %s\n", wt_path); exit(1); }
        fclose(f);

        float *d_x, *d_w, *d_y;
        CUDA_CHECK(cudaMalloc(&d_x, bytes));
        CUDA_CHECK(cudaMalloc(&d_w, wbytes));
        CUDA_CHECK(cudaMalloc(&d_y, bytes));
        CUDA_CHECK(cudaMemcpy(d_x, h_x, bytes,  cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_w, h_w, wbytes, cudaMemcpyHostToDevice));

        rmsnorm_v2_warp<<<rows, 128, 4 * sizeof(float)>>>(d_x, d_w, d_y, cols, EPS);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h_y, d_y, bytes, cudaMemcpyDeviceToHost));

        f = fopen(out_path, "wb");
        if (!f) { fprintf(stderr, "Cannot write %s\n", out_path); exit(1); }
        fwrite(h_y, sizeof(float), elem, f);
        fclose(f);

        CUDA_CHECK(cudaFree(d_x)); CUDA_CHECK(cudaFree(d_w)); CUDA_CHECK(cudaFree(d_y));
        free(h_x); free(h_w); free(h_y);
    }
    printf("rmsnorm dump: rows=%d cols=%d dtype=%s -> %s\n",
           rows, cols, use_fp16 ? "fp16" : "fp32", out_path);
}

// ── Main ──────────────────────────────────────────────────────────────────────
int main(int argc, char** argv) {
    // Optional: path to benchmark.csv
    const char* csv_path         = nullptr;
    const char* dump_path        = nullptr;
    const char* load_input_path  = nullptr;
    const char* load_weight_path = nullptr;
    const char* dtype            = "fp32";
    int test_rows = 1024, test_hidden = 2048;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--csv")          && i+1 < argc) csv_path          = argv[++i];
        if (!strcmp(argv[i], "--dump_output")  && i+1 < argc) dump_path         = argv[++i];
        if (!strcmp(argv[i], "--load_input")   && i+1 < argc) load_input_path   = argv[++i];
        if (!strcmp(argv[i], "--load_weight")  && i+1 < argc) load_weight_path  = argv[++i];
        if (!strcmp(argv[i], "--dtype")        && i+1 < argc) dtype             = argv[++i];
        if (!strcmp(argv[i], "--rows")         && i+1 < argc) test_rows   = atoi(argv[++i]);
        if (!strcmp(argv[i], "--cols")         && i+1 < argc) test_hidden  = atoi(argv[++i]);
        if (!strcmp(argv[i], "--hidden")       && i+1 < argc) test_hidden  = atoi(argv[++i]);
    }

    // ── Dump mode (called by test_correctness.py) ─────────────────────────────
    if (load_input_path) {
        if (!load_weight_path || !dump_path) {
            fprintf(stderr, "Dump mode requires --load_input, --load_weight, --dump_output\n");
            return 1;
        }
        run_dump(test_rows, test_hidden, strcmp(dtype, "fp16") == 0,
                 load_input_path, load_weight_path, dump_path);
        return 0;
    }

    const float EPS = 1e-5f;

    // Config sweep: (rows, hidden)
    int configs[][2] = {{256, 512}, {256, 2048}, {1024, 2048}, {1024, 4096}};
    int num_cfg = sizeof(configs) / sizeof(configs[0]);

    FILE* csv_fp = nullptr;
    if (csv_path) {
        csv_fp = fopen(csv_path, "a");
        if (!csv_fp) fprintf(stderr, "Warning: cannot open %s\n", csv_path);
    }

    printf("RMSNorm benchmark  (FP32, RTX 3080 sm_86)\n");
    printf("%-30s  %-10s  %-10s\n", "config", "time_ms", "GB/s");

    for (int c = 0; c < num_cfg; c++) {
        int rows = configs[c][0], hidden = configs[c][1];
        size_t bytes_xyw = (size_t)rows * hidden * sizeof(float);
        size_t bytes_w   = hidden * sizeof(float);

        // Allocate host
        float* h_x   = (float*)malloc(bytes_xyw);
        float* h_w   = (float*)malloc(bytes_w);
        float* h_ref = (float*)malloc(bytes_xyw);
        float* h_got = (float*)malloc(bytes_xyw);

        // Random init
        for (int i = 0; i < rows * hidden; i++) h_x[i] = (float)rand() / RAND_MAX - 0.5f;
        for (int i = 0; i < hidden;        i++) h_w[i] = (float)rand() / RAND_MAX + 0.5f;

        // CPU reference
        rmsnorm_cpu(h_x, h_w, h_ref, rows, hidden, EPS);

        // GPU alloc
        float *d_x, *d_w, *d_y;
        CUDA_CHECK(cudaMalloc(&d_x, bytes_xyw));
        CUDA_CHECK(cudaMalloc(&d_w, bytes_w));
        CUDA_CHECK(cudaMalloc(&d_y, bytes_xyw));
        CUDA_CHECK(cudaMemcpy(d_x, h_x, bytes_xyw, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_w, h_w, bytes_w,   cudaMemcpyHostToDevice));

        printf("\n[rows=%d  hidden=%d]\n", rows, hidden);

        // v1 naive: block_size=256, shared = 256 floats
        float ms_v1 = bench(rmsnorm_v1_naive, "v1_naive",
                            d_x, d_w, d_y, rows, hidden, EPS,
                            256, 256 * sizeof(float));

        // Correctness v1
        CUDA_CHECK(cudaMemcpy(h_got, d_y, bytes_xyw, cudaMemcpyDeviceToHost));
        check_correctness(h_ref, h_got, rows * hidden, "v1_naive");

        // v2_warp precondition: float4 vectorized loads require hidden divisible by 4
        if (hidden % 4 != 0) {
            fprintf(stderr, "rmsnorm_v2_warp requires hidden %% 4 == 0 (got %d)\n", hidden);
            exit(1);
        }
        // v2 warp: block_size=128 (4 warps), shared = 4 floats (one per warp)
        float ms_v2 = bench(rmsnorm_v2_warp, "v2_warp",
                            d_x, d_w, d_y, rows, hidden, EPS,
                            128, 4 * sizeof(float));

        // Correctness v2
        CUDA_CHECK(cudaMemcpy(h_got, d_y, bytes_xyw, cudaMemcpyDeviceToHost));
        check_correctness(h_ref, h_got, rows * hidden, "v2_warp");

        // v3 regcache: block_size=256 (8 warps), shared = 8 floats
        float ms_v3 = bench(rmsnorm_v3_regcache, "v3_regcache",
                            d_x, d_w, d_y, rows, hidden, EPS,
                            256, 8 * sizeof(float));

        // Correctness v3
        CUDA_CHECK(cudaMemcpy(h_got, d_y, bytes_xyw, cudaMemcpyDeviceToHost));
        check_correctness(h_ref, h_got, rows * hidden, "v3_regcache");

        printf("  speedup v2/v1: %.2fx   v3/v1: %.2fx\n",
               ms_v1 / ms_v2, ms_v1 / ms_v3);

        // Dump output for test_correctness.py (uses last config's v2 output)
        if (dump_path && c == num_cfg - 1) {
            FILE* fp = fopen(dump_path, "wb");
            if (fp) { fwrite(h_got, sizeof(float), rows * hidden, fp); fclose(fp); }
        }

        if (csv_fp) {
            float gb = 3.0f * rows * hidden * sizeof(float) / 1e9f;
            fprintf(csv_fp, "rmsnorm,v1_naive,%dx%d,256,%.3f,%.1f,FP32 GB/s\n",
                    rows, hidden, ms_v1, gb / (ms_v1 / 1000.0f));
            fprintf(csv_fp, "rmsnorm,v2_warp,%dx%d,128,%.3f,%.1f,FP32 GB/s\n",
                    rows, hidden, ms_v2, gb / (ms_v2 / 1000.0f));
            fprintf(csv_fp, "rmsnorm,v3_regcache,%dx%d,256,%.3f,%.1f,FP32 GB/s\n",
                    rows, hidden, ms_v3, gb / (ms_v3 / 1000.0f));
        }

        CUDA_CHECK(cudaFree(d_x));
        CUDA_CHECK(cudaFree(d_w));
        CUDA_CHECK(cudaFree(d_y));
        free(h_x); free(h_w); free(h_ref); free(h_got);
    }

    if (csv_fp) { fclose(csv_fp); printf("\nResults appended to %s\n", csv_path); }
    return 0;
}
