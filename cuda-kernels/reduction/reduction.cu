// Parallel Reduction — 4 versions following Mark Harris "Optimizing Parallel
// Reduction in CUDA" (2004 GPU Gems 2 / NVIDIA SDK white paper).
//
// v1: naive interleaved addressing        → warp divergence + bank conflicts
// v2: sequential addressing               → no divergence, still has bank conflicts
// v3: first add during load               → halves thread count needed
// v4: warp-level unrolling (__shfl_down_sync) → eliminates shared mem for final warp
//
// Each kernel is benchmarked with CUDA events; throughput and speedup vs v1 are
// printed and results are appended to ../../benchmark.csv.

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cuda_runtime.h>

// ─── helpers ────────────────────────────────────────────────────────────────

#define CHECK_CUDA(call)                                                    \
    do {                                                                    \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error %s:%d — %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err));                               \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

// ─── v1: naive interleaved addressing ───────────────────────────────────────
// Thread stride doubles every step → active threads are 0,2,4… then 0,4,8…
// Causes warp divergence (threads in same warp take different branch paths)
// and non-sequential shared-memory access (bank conflicts).
__global__ void reduce_v1_naive(const float* __restrict__ g_in,
                                float* __restrict__ g_out, int n) {
    extern __shared__ float sdata[];
    unsigned tid = threadIdx.x;
    unsigned i   = blockIdx.x * blockDim.x + tid;

    sdata[tid] = (i < (unsigned)n) ? g_in[i] : 0.f;
    __syncthreads();

    // interleaved addressing: stride = 1, 2, 4, …
    for (unsigned s = 1; s < blockDim.x; s <<= 1) {
        if (tid % (2 * s) == 0)          // ← divergence: half warp idles
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) g_out[blockIdx.x] = sdata[0];
}

// ─── v2: sequential addressing ──────────────────────────────────────────────
// Stride starts at blockDim.x/2 and halves each step.
// Active threads are always the lowest-indexed → no warp divergence.
// Still has shared-memory bank conflicts when stride is odd.
__global__ void reduce_v2_seq(const float* __restrict__ g_in,
                              float* __restrict__ g_out, int n) {
    extern __shared__ float sdata[];
    unsigned tid = threadIdx.x;
    unsigned i   = blockIdx.x * blockDim.x + tid;

    sdata[tid] = (i < (unsigned)n) ? g_in[i] : 0.f;
    __syncthreads();

    for (unsigned s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) g_out[blockIdx.x] = sdata[0];
}

// ─── v3: first add during load ───────────────────────────────────────────────
// Each thread loads TWO elements and adds them before entering the reduction
// loop → we need only n/2 threads to process n elements, so we can launch
// half as many blocks (or use the same block count for 2× the data range).
// Eliminates the "idle first half of threads" problem.
__global__ void reduce_v3_fad(const float* __restrict__ g_in,
                              float* __restrict__ g_out, int n) {
    extern __shared__ float sdata[];
    unsigned tid      = threadIdx.x;
    unsigned gridSize = blockDim.x * 2 * gridDim.x;
    unsigned i        = blockIdx.x * (blockDim.x * 2) + tid;

    // load + first add
    float acc = 0.f;
    while (i < (unsigned)n) {
        acc += g_in[i];
        if (i + blockDim.x < (unsigned)n)
            acc += g_in[i + blockDim.x];
        i += gridSize;
    }
    sdata[tid] = acc;
    __syncthreads();

    for (unsigned s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) g_out[blockIdx.x] = sdata[0];
}

// ─── v4: warp-level unrolling with __shfl_down_sync ─────────────────────────
// Once the active count drops to ≤ 32 (one warp), threads are already
// implicitly synchronised by the warp.  Using __shfl_down_sync avoids
// shared memory entirely for the final 5 reduction steps.
__device__ __forceinline__ float warp_reduce(float val) {
    // full warp mask: all 32 lanes participate
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__global__ void reduce_v4_shfl(const float* __restrict__ g_in,
                               float* __restrict__ g_out, int n) {
    extern __shared__ float sdata[];
    unsigned tid      = threadIdx.x;
    unsigned gridSize = blockDim.x * 2 * gridDim.x;
    unsigned i        = blockIdx.x * (blockDim.x * 2) + tid;

    // load + first add (same as v3)
    float acc = 0.f;
    while (i < (unsigned)n) {
        acc += g_in[i];
        if (i + blockDim.x < (unsigned)n)
            acc += g_in[i + blockDim.x];
        i += gridSize;
    }
    sdata[tid] = acc;
    __syncthreads();

    // tree reduction down to 64 elements (loop stops when s==32 since 32>32 is false)
    for (unsigned s = blockDim.x >> 1; s > 32; s >>= 1) {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    // final warp: combine the remaining two 32-element groups via shuffle.
    // Requires blockDim >= 64 (guaranteed by THREADS=256 above).
    if (tid < 32) {
        // fold sdata[32..63] into sdata[0..31] before the warp reduction
        float val = sdata[tid] + sdata[tid + 32];
        val = warp_reduce(val);
        if (tid == 0) g_out[blockIdx.x] = val;
    }
}

// ─── benchmark harness ───────────────────────────────────────────────────────

struct KernelSpec {
    const char* name;
    void (*fn)(const float*, float*, int);  // wrapped below
};

// unified wrapper type so we can store function pointers cleanly
using ReduceFn = void (*)(const float* __restrict__, float* __restrict__, int);

static float run_kernel(ReduceFn kernel, const char* variant_name,
                        const float* d_in, float* d_out_partial,
                        int n, int blocks, int threads,
                        int warmup, int iters) {
    const size_t smem = threads * sizeof(float);

    // warmup
    for (int r = 0; r < warmup; ++r) {
        kernel<<<blocks, threads, smem>>>(d_in, d_out_partial, n);
        // second-level reduction on host (CPU sum of partial sums) — latency
        // dominated by kernel, so we skip the host part for timing purposes.
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    for (int r = 0; r < iters; ++r)
        kernel<<<blocks, threads, smem>>>(d_in, d_out_partial, n);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0.f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    ms /= iters;

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    return ms;
}

// CPU reference sum (Kahan compensated)
static double cpu_sum(const float* h, int n) {
    double s = 0.0, c = 0.0;
    for (int i = 0; i < n; ++i) {
        double y = (double)h[i] - c;
        double t = s + y;
        c = (t - s) - y;
        s = t;
    }
    return s;
}

// finish reduction: copy partial sums back and sum on CPU
static float finish_reduction(const float* d_partial, float* h_partial, int blocks) {
    CHECK_CUDA(cudaMemcpy(h_partial, d_partial, blocks * sizeof(float),
                         cudaMemcpyDeviceToHost));
    float s = 0.f;
    for (int i = 0; i < blocks; ++i) s += h_partial[i];
    return s;
}

int main() {
    // ── configuration ──────────────────────────────────────────────────────
    const int N         = 1 << 24;   // 16 M elements  (~64 MB)
    const int THREADS   = 256;
    const int WARMUP    = 3;
    const int ITERS     = 5;

    // Preconditions for reduce_v4_shfl: the final warp folds
    // sdata[tid] + sdata[tid+32], so THREADS must be ≥ 64. All tree
    // reductions (v1-v4) additionally assume THREADS is a power of 2.
    static_assert(THREADS >= 64,
                  "reduce_v4_shfl requires THREADS >= 64");
    static_assert((THREADS & (THREADS - 1)) == 0,
                  "THREADS must be a power of 2 (tree reduction)");

    // v3/v4: each thread loads 2 elements → half as many blocks
    const int BLOCKS_V12 = (N + THREADS - 1) / THREADS;
    const int BLOCKS_V34 = (N + THREADS * 2 - 1) / (THREADS * 2);

    // clamp to hardware max
    int max_blocks;
    CHECK_CUDA(cudaDeviceGetAttribute(&max_blocks,
                                     cudaDevAttrMaxGridDimX, 0));
    int b12 = min(BLOCKS_V12, max_blocks);
    int b34 = min(BLOCKS_V34, max_blocks);

    printf("N = %d (%.1f M), THREADS = %d, BLOCKS v1/v2 = %d, v3/v4 = %d\n",
           N, N / 1e6f, THREADS, b12, b34);

    // ── allocate ────────────────────────────────────────────────────────────
    float* h_in      = (float*)malloc(N * sizeof(float));
    float* h_partial = (float*)malloc(b12 * sizeof(float));  // b12 >= b34

    // fill with values that sum to a known integer (easy correctness check)
    for (int i = 0; i < N; ++i) h_in[i] = 1.f;
    double ref = cpu_sum(h_in, N);

    float *d_in, *d_out;
    CHECK_CUDA(cudaMalloc(&d_in,  N     * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_out, b12   * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice));

    // ── run each version ────────────────────────────────────────────────────
    const int NVER = 4;
    const char* names[NVER]  = { "v1_naive", "v2_seq", "v3_fad", "v4_shfl" };
    ReduceFn    fns[NVER]    = { reduce_v1_naive, reduce_v2_seq,
                                 reduce_v3_fad,   reduce_v4_shfl };
    int         blocks[NVER] = { b12, b12, b34, b34 };

    float times[NVER];
    bool  correct[NVER];

    for (int v = 0; v < NVER; ++v) {
        times[v] = run_kernel(fns[v], names[v], d_in, d_out,
                              N, blocks[v], THREADS, WARMUP, ITERS);

        // correctness: sum partial sums on host
        float result = finish_reduction(d_out, h_partial, blocks[v]);
        double err   = fabs((double)result - ref) / ref;
        correct[v]   = (err < 1e-4);   // relative tolerance for FP32 sum

        // bandwidth: N reads + blocks writes, treat as effective BW
        double bytes_gb = (N + blocks[v]) * sizeof(float) / 1e9;
        double gbps     = bytes_gb / (times[v] * 1e-3);

        printf("  %-12s  time=%7.3f ms  BW=%6.1f GB/s  result=%.0f  %s\n",
               names[v], times[v], gbps, (double)result,
               correct[v] ? "OK" : "WRONG");
    }

    printf("\nSpeedup vs v1:\n");
    for (int v = 0; v < NVER; ++v)
        printf("  %-12s  %.2fx\n", names[v], times[0] / times[v]);

    // ── append to benchmark.csv ─────────────────────────────────────────────
    // Format: kernel,variant,size,block_size,time_ms,throughput,notes
    FILE* csv = fopen("../../benchmark.csv", "a");
    if (!csv) csv = fopen("benchmark.csv", "a");   // fallback if run from build/
    if (csv) {
        for (int v = 0; v < NVER; ++v) {
            double bytes_gb = (N + blocks[v]) * sizeof(float) / 1e9;
            double gbps     = bytes_gb / (times[v] * 1e-3);
            fprintf(csv,
                    "reduction,%s,%d,%d,%.4f,%.2f GB/s,speedup_vs_v1=%.2fx %s\n",
                    names[v], N, THREADS, times[v], gbps,
                    times[0] / times[v], correct[v] ? "" : "WRONG");
        }
        fclose(csv);
        printf("\nResults appended to benchmark.csv\n");
    } else {
        fprintf(stderr, "Warning: could not open benchmark.csv\n");
    }

    // ── cleanup ─────────────────────────────────────────────────────────────
    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_out));
    free(h_in);
    free(h_partial);
    return 0;
}
