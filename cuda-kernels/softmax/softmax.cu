/*
 * softmax.cu  —  cuda-kernels / softmax
 *
 * Two CUDA softmax variants for a [B, N] FP32 matrix.
 * One block per row; block handles N elements via stride loop.
 *
 *   v1_naive  : two-pass (max reduction → sum reduction → write)
 *   v2_online : one-pass (max+sum combined via online correction)
 *               Reference: Milakov & Gimelshein, "Online normalizer calculation
 *               for softmax", 2018; also the core trick in Flash Attention.
 *
 * Build: see CMakeLists.txt (sm_86, CUDA::cudart)
 * Run:   ./softmax [path/to/benchmark.csv]
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cfloat>
#include <algorithm>
#include <cuda_runtime.h>

// ── Error checking ────────────────────────────────────────────────────────────
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t _e = (call);                                               \
        if (_e != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error %s:%d  %s\n",                         \
                    __FILE__, __LINE__, cudaGetErrorString(_e));               \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

// ── v1_naive: two-pass softmax ────────────────────────────────────────────────
//
// Pass 1: block-wide reduction to find row max  (shared memory tree)
// Pass 2: block-wide reduction to find sum(exp) (shared memory tree)
// Pass 3: write exp(x - max) / sum
//
// Shared memory: blockDim.x floats
//
__global__ void softmax_v1_naive(const float* __restrict__ in,
                                  float* __restrict__ out, int N)
{
    extern __shared__ float smem[];
    const int row = blockIdx.x;
    const float* row_in  = in  + (size_t)row * N;
    float*       row_out = out + (size_t)row * N;

    // ── Pass 1: max ───────────────────────────────────────────────────────────
    float local_max = -FLT_MAX;
    for (int i = threadIdx.x; i < N; i += blockDim.x)
        local_max = fmaxf(local_max, row_in[i]);

    smem[threadIdx.x] = local_max;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            smem[threadIdx.x] = fmaxf(smem[threadIdx.x], smem[threadIdx.x + s]);
        __syncthreads();
    }
    const float row_max = smem[0];
    __syncthreads();   // re-use smem for pass 2

    // ── Pass 2: sum(exp) ──────────────────────────────────────────────────────
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < N; i += blockDim.x)
        local_sum += expf(row_in[i] - row_max);

    smem[threadIdx.x] = local_sum;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            smem[threadIdx.x] += smem[threadIdx.x + s];
        __syncthreads();
    }
    const float row_sum = smem[0];

    // ── Pass 3: write ─────────────────────────────────────────────────────────
    for (int i = threadIdx.x; i < N; i += blockDim.x)
        row_out[i] = expf(row_in[i] - row_max) / row_sum;
}

// ── v2_online: single-pass (max + sum combined) ───────────────────────────────
//
// Each thread sweeps its chunk accumulating a local (max, sum) pair using
// the online correction:
//
//   new_sum = old_sum * exp(old_max - new_max) + exp(x - new_max)
//
// This keeps sum numerically stable as max increases—no separate max pass.
// After the sweep, a tree reduction merges (max, sum) pairs from all threads
// using the same associative operator.  One final global-memory write pass.
//
// Shared memory: 2 * blockDim.x floats  (max[] + sum[])
//
__global__ void softmax_v2_online(const float* __restrict__ in,
                                   float* __restrict__ out, int N)
{
    extern __shared__ float smem[];
    float* smem_max = smem;
    float* smem_sum = smem + blockDim.x;

    const int row = blockIdx.x;
    const float* row_in  = in  + (size_t)row * N;
    float*       row_out = out + (size_t)row * N;

    // ── Single sweep: online (max, sum) ───────────────────────────────────────
    float local_max = -FLT_MAX;
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        float x       = row_in[i];
        float new_max = fmaxf(local_max, x);
        // Rescale accumulated sum when max increases; add new term
        local_sum = local_sum * expf(local_max - new_max) + expf(x - new_max);
        local_max = new_max;
    }
    smem_max[threadIdx.x] = local_max;
    smem_sum[threadIdx.x] = local_sum;
    __syncthreads();

    // ── Tree reduction: merge (max, sum) pairs ────────────────────────────────
    // Operator is associative: (m1,s1) ⊕ (m2,s2) = (max, s1·exp(m1-max)+s2·exp(m2-max))
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            float m1 = smem_max[threadIdx.x],     m2 = smem_max[threadIdx.x + s];
            float s1 = smem_sum[threadIdx.x],     s2 = smem_sum[threadIdx.x + s];
            float new_m = fmaxf(m1, m2);
            smem_max[threadIdx.x] = new_m;
            smem_sum[threadIdx.x] = s1 * expf(m1 - new_m) + s2 * expf(m2 - new_m);
        }
        __syncthreads();
    }
    const float row_max = smem_max[0];
    const float row_sum = smem_sum[0];

    // ── Write output ──────────────────────────────────────────────────────────
    for (int i = threadIdx.x; i < N; i += blockDim.x)
        row_out[i] = expf(row_in[i] - row_max) / row_sum;
}

// ── CPU reference ─────────────────────────────────────────────────────────────
static void softmax_cpu(const float* in, float* out, int B, int N)
{
    for (int b = 0; b < B; b++) {
        const float* row_in  = in  + b * N;
        float*       row_out = out + b * N;
        float mx = -FLT_MAX;
        for (int i = 0; i < N; i++) mx = fmaxf(mx, row_in[i]);
        float s = 0.0f;
        for (int i = 0; i < N; i++) s += expf(row_in[i] - mx);
        for (int i = 0; i < N; i++) row_out[i] = expf(row_in[i] - mx) / s;
    }
}

// ── Kernel selector ───────────────────────────────────────────────────────────
enum KernelVariant { V1_NAIVE, V2_ONLINE };

static void launch(KernelVariant v, const float* d_in, float* d_out,
                   int B, int N, int block_size)
{
    dim3 grid(B), block(block_size);
    size_t smem = (v == V2_ONLINE ? 2 : 1) * block_size * sizeof(float);
    if (v == V1_NAIVE)
        softmax_v1_naive<<<grid, block, smem>>>(d_in, d_out, N);
    else
        softmax_v2_online<<<grid, block, smem>>>(d_in, d_out, N);
    CUDA_CHECK(cudaGetLastError());
}

// ── Benchmark one (variant × N) combination ───────────────────────────────────
static float bench_one(KernelVariant v, int B, int N, int block_size,
                       float* h_ref, const char* variant_name,
                       const char* csv_path)
{
    size_t bytes = (size_t)B * N * sizeof(float);

    // Host input (random)
    float* h_in = (float*)malloc(bytes);
    for (size_t i = 0; i < (size_t)B * N; i++)
        h_in[i] = (float)rand() / RAND_MAX * 4.0f - 2.0f;   // uniform [-2, 2]

    // CPU reference
    softmax_cpu(h_in, h_ref, B, N);

    // Device buffers
    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in,  bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    // Warmup
    for (int i = 0; i < 3; i++)
        launch(v, d_in, d_out, B, N, block_size);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed runs
    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));

    float times[5];
    for (int r = 0; r < 5; r++) {
        CUDA_CHECK(cudaEventRecord(t0));
        launch(v, d_in, d_out, B, N, block_size);
        CUDA_CHECK(cudaEventRecord(t1));
        CUDA_CHECK(cudaEventSynchronize(t1));
        CUDA_CHECK(cudaEventElapsedTime(&times[r], t0, t1));
    }
    std::sort(times, times + 5);
    const float median_ms = times[2];

    // Correctness: max absolute error vs CPU
    float* h_out = (float*)malloc(bytes);
    CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));
    float max_err = 0.0f;
    for (size_t i = 0; i < (size_t)B * N; i++)
        max_err = fmaxf(max_err, fabsf(h_out[i] - h_ref[i]));
    bool ok = (max_err < 1e-5f);

    // Effective bandwidth: 1 read + 1 write (minimum useful traffic)
    float gb    = 2.0f * bytes / 1e9f;
    float gbps  = gb / (median_ms / 1000.0f);

    printf("  %-14s  B=%-5d N=%-5d  block=%-4d  %7.3f ms  %6.1f GB/s  err=%.2e  [%s]\n",
           variant_name, B, N, block_size, median_ms, gbps, max_err, ok ? "OK" : "FAIL");
    if (!ok)
        fprintf(stderr, "  CORRECTNESS FAILED: max_err=%.6e\n", max_err);

    // Append to CSV
    if (csv_path) {
        FILE* fp = fopen(csv_path, "a");
        if (fp) {
            // kernel,variant,size,block_size,time_ms,throughput,notes
            fprintf(fp, "softmax,%s,%dx%d,%d,%.3f,%.1f,RTX3080 FP32 sm86\n",
                    variant_name, B, N, block_size, median_ms, gbps);
            fclose(fp);
        }
    }

    CUDA_CHECK(cudaEventDestroy(t0));
    CUDA_CHECK(cudaEventDestroy(t1));
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    free(h_in);
    free(h_out);

    return median_ms;
}

// ── Dump mode (invoked by test_correctness.py) ───────────────────────────────
// --load_input FILE --dump_output FILE --rows R --cols C --dtype fp32|fp16
static void run_dump(int rows, int cols, bool use_fp16,
                     const char* in_path, const char* out_path) {
    const int block_size = 256;
    const size_t elem    = (size_t)rows * cols;

    if (use_fp16) {
        // Run on fp32 internally (v2_online is fp32-only), convert at boundaries
        size_t bytes32 = elem * sizeof(float);
        size_t bytes16 = elem * sizeof(__half);
        __half* h_x16 = (__half*)malloc(bytes16);
        float*  h_xf  = (float*)malloc(bytes32);
        float*  h_yf  = (float*)malloc(bytes32);
        __half* h_y16 = (__half*)malloc(bytes16);

        FILE* f = fopen(in_path, "rb");
        if (!f) { fprintf(stderr, "Cannot open %s\n", in_path); exit(1); }
        if (fread(h_x16, sizeof(__half), elem, f) != elem) {
            fprintf(stderr, "Short read %s\n", in_path); exit(1);
        }
        fclose(f);

        for (size_t i = 0; i < elem; i++) h_xf[i] = __half2float(h_x16[i]);

        float *d_in, *d_out;
        CUDA_CHECK(cudaMalloc(&d_in,  bytes32));
        CUDA_CHECK(cudaMalloc(&d_out, bytes32));
        CUDA_CHECK(cudaMemcpy(d_in, h_xf, bytes32, cudaMemcpyHostToDevice));

        size_t smem = 2 * block_size * sizeof(float);
        softmax_v2_online<<<rows, block_size, smem>>>(d_in, d_out, cols);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h_yf, d_out, bytes32, cudaMemcpyDeviceToHost));

        for (size_t i = 0; i < elem; i++) h_y16[i] = __float2half(h_yf[i]);

        f = fopen(out_path, "wb");
        if (!f) { fprintf(stderr, "Cannot write %s\n", out_path); exit(1); }
        fwrite(h_y16, sizeof(__half), elem, f);
        fclose(f);

        CUDA_CHECK(cudaFree(d_in)); CUDA_CHECK(cudaFree(d_out));
        free(h_x16); free(h_xf); free(h_yf); free(h_y16);
    } else {
        size_t bytes = elem * sizeof(float);
        float* h_x = (float*)malloc(bytes);
        float* h_y = (float*)malloc(bytes);

        FILE* f = fopen(in_path, "rb");
        if (!f) { fprintf(stderr, "Cannot open %s\n", in_path); exit(1); }
        if (fread(h_x, sizeof(float), elem, f) != elem) {
            fprintf(stderr, "Short read %s\n", in_path); exit(1);
        }
        fclose(f);

        float *d_in, *d_out;
        CUDA_CHECK(cudaMalloc(&d_in,  bytes));
        CUDA_CHECK(cudaMalloc(&d_out, bytes));
        CUDA_CHECK(cudaMemcpy(d_in, h_x, bytes, cudaMemcpyHostToDevice));

        size_t smem = 2 * block_size * sizeof(float);
        softmax_v2_online<<<rows, block_size, smem>>>(d_in, d_out, cols);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h_y, d_out, bytes, cudaMemcpyDeviceToHost));

        f = fopen(out_path, "wb");
        if (!f) { fprintf(stderr, "Cannot write %s\n", out_path); exit(1); }
        fwrite(h_y, sizeof(float), elem, f);
        fclose(f);

        CUDA_CHECK(cudaFree(d_in)); CUDA_CHECK(cudaFree(d_out));
        free(h_x); free(h_y);
    }
    printf("softmax dump: rows=%d cols=%d dtype=%s -> %s\n",
           rows, cols, use_fp16 ? "fp16" : "fp32", out_path);
}

// ── main ─────────────────────────────────────────────────────────────────────
int main(int argc, char* argv[])
{
    // Parse flags; preserve legacy positional csv_path for benchmark mode
    const char* csv_path      = nullptr;
    const char* load_input    = nullptr;
    const char* dump_output   = nullptr;
    const char* dtype         = "fp32";
    int         dump_rows     = 128;
    int         dump_cols     = 512;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--load_input")  && i+1 < argc) load_input  = argv[++i];
        else if (!strcmp(argv[i], "--dump_output") && i+1 < argc) dump_output = argv[++i];
        else if (!strcmp(argv[i], "--dtype")  && i+1 < argc) dtype       = argv[++i];
        else if (!strcmp(argv[i], "--rows")   && i+1 < argc) dump_rows   = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--cols")   && i+1 < argc) dump_cols   = atoi(argv[++i]);
        else if (argv[i][0] != '-')                          csv_path    = argv[i];
    }

    // ── Dump mode ─────────────────────────────────────────────────────────────
    if (load_input) {
        if (!dump_output) {
            fprintf(stderr, "Dump mode requires --load_input and --dump_output\n");
            return 1;
        }
        run_dump(dump_rows, dump_cols, strcmp(dtype, "fp16") == 0, load_input, dump_output);
        return 0;
    }

    // Fixed parameters
    const int B          = 1024;
    const int block_size = 256;   // must be power-of-2 for tree reduction
    const int Ns[]       = {128, 512, 2048, 4096};
    const int num_N      = (int)(sizeof(Ns) / sizeof(Ns[0]));

    // Pre-allocate largest reference buffer (reused across runs)
    const int N_max = Ns[num_N - 1];
    float* h_ref = (float*)malloc((size_t)B * N_max * sizeof(float));

    printf("softmax benchmark  B=%d  block_size=%d\n", B, block_size);
    printf("%-16s %-8s %-8s %-8s %-12s %-10s %-12s\n",
           "variant", "B", "N", "block", "time_ms", "GB/s", "correctness");
    printf("%s\n", std::string(80, '-').c_str());

    srand(42);

    for (int ni = 0; ni < num_N; ni++) {
        int N = Ns[ni];
        bench_one(V1_NAIVE,  B, N, block_size, h_ref, "v1_naive",  csv_path);
        bench_one(V2_ONLINE, B, N, block_size, h_ref, "v2_online", csv_path);
        if (ni < num_N - 1) printf("\n");
    }

    free(h_ref);
    if (csv_path) printf("\nResults appended to %s\n", csv_path);
    return 0;
}
