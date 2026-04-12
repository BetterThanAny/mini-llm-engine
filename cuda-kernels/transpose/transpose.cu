// transpose.cu — Matrix Transpose: naive vs. shared-memory tiled (tile=32)
// Benchmark: effective bandwidth (GB/s), 3 warmup + 5 measure, take median
// Hardware target: RTX 3080, sm_86

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error %s:%d  %s\n",                          \
                    __FILE__, __LINE__, cudaGetErrorString(err));               \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

static const int TILE = 32;
// +1 padding eliminates shared-memory bank conflicts on 32-wide float tiles
static const int TILE_PAD = TILE + 1;

// ── Kernels ───────────────────────────────────────────────────────────────────

// Naive: thread (r,c) reads A[r][c] (coalesced) and writes B[c][r] (strided)
__global__ void transpose_naive(const float* __restrict__ A,
                                float* __restrict__       B,
                                int rows, int cols) {
    int c = blockIdx.x * TILE + threadIdx.x;
    int r = blockIdx.y * TILE + threadIdx.y;
    if (r < rows && c < cols)
        B[c * rows + r] = A[r * cols + c];
}

// Tiled: load tile into shared memory (coalesced read), write transposed
// (coalesced write); +1 padding removes bank conflicts.
__global__ void transpose_tiled(const float* __restrict__ A,
                                float* __restrict__       B,
                                int rows, int cols) {
    __shared__ float smem[TILE][TILE_PAD];

    // Source position
    int sc = blockIdx.x * TILE + threadIdx.x;
    int sr = blockIdx.y * TILE + threadIdx.y;

    if (sr < rows && sc < cols)
        smem[threadIdx.y][threadIdx.x] = A[sr * cols + sc];

    __syncthreads();

    // Destination position — blocks are swapped so write is coalesced
    int dc = blockIdx.y * TILE + threadIdx.x;
    int dr = blockIdx.x * TILE + threadIdx.y;

    if (dr < cols && dc < rows)
        B[dr * rows + dc] = smem[threadIdx.x][threadIdx.y];
}

// ── Helpers ───────────────────────────────────────────────────────────────────

static void fill_random(float* h, int n) {
    for (int i = 0; i < n; i++) h[i] = (float)rand() / RAND_MAX;
}

static void check_correctness(const float* A, const float* B, int rows, int cols) {
    for (int r = 0; r < rows; r++)
        for (int c = 0; c < cols; c++)
            if (fabsf(A[r * cols + c] - B[c * rows + r]) > 1e-5f) {
                fprintf(stderr, "Correctness FAILED at (%d,%d): A=%f B=%f\n",
                        r, c, A[r * cols + c], B[c * rows + r]);
                exit(1);
            }
}

// Returns median kernel time in milliseconds over 5 runs (3 warmup discarded).
// kernel_id: 0 = naive, 1 = tiled
static float bench(int rows, int cols, int kernel_id,
                   const float* d_A, float* d_B) {
    dim3 block(TILE, TILE);
    dim3 grid((cols + TILE - 1) / TILE, (rows + TILE - 1) / TILE);

    auto launch = [&]() {
        if (kernel_id == 0)
            transpose_naive<<<grid, block>>>(d_A, d_B, rows, cols);
        else
            transpose_tiled<<<grid, block>>>(d_A, d_B, rows, cols);
    };

    for (int i = 0; i < 3; i++) launch();
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));

    float times[5];
    for (int r = 0; r < 5; r++) {
        CUDA_CHECK(cudaEventRecord(t0));
        launch();
        CUDA_CHECK(cudaEventRecord(t1));
        CUDA_CHECK(cudaEventSynchronize(t1));
        CUDA_CHECK(cudaEventElapsedTime(&times[r], t0, t1));
    }

    CUDA_CHECK(cudaEventDestroy(t0));
    CUDA_CHECK(cudaEventDestroy(t1));

    // Bubble-sort for median
    for (int i = 0; i < 4; i++)
        for (int j = i + 1; j < 5; j++)
            if (times[j] < times[i]) { float t = times[i]; times[i] = times[j]; times[j] = t; }

    return times[2];
}

// ── Main ──────────────────────────────────────────────────────────────────────

int main() {
    // Test square matrices of increasing size
    const int sizes[] = {1024, 2048, 4096, 8192};
    const int nsizes  = (int)(sizeof(sizes) / sizeof(sizes[0]));

    printf("%-10s  %-12s  %-12s  %-14s  %-14s  %s\n",
           "N", "naive_ms", "tiled_ms",
           "naive_GB/s", "tiled_GB/s", "speedup");
    printf("%s\n", std::string(80, '-').c_str());

    // CSV header hint (appended later)
    // kernel,variant,size,block_size,time_ms,throughput,notes

    for (int si = 0; si < nsizes; si++) {
        int N     = sizes[si];
        int rows  = N, cols = N;
        size_t bytes = (size_t)rows * cols * sizeof(float);

        float* h_A  = (float*)malloc(bytes);
        float* h_B  = (float*)malloc(bytes);
        fill_random(h_A, rows * cols);

        float *d_A, *d_B;
        CUDA_CHECK(cudaMalloc(&d_A, bytes));
        CUDA_CHECK(cudaMalloc(&d_B, bytes));
        CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));

        // Correctness check (small size only, full-size is too slow on CPU)
        if (N <= 2048) {
            bench(rows, cols, 0, d_A, d_B);     // naive
            CUDA_CHECK(cudaMemcpy(h_B, d_B, bytes, cudaMemcpyDeviceToHost));
            check_correctness(h_A, h_B, rows, cols);
            printf("  [N=%d] naive correctness OK\n", N);

            bench(rows, cols, 1, d_A, d_B);     // tiled
            CUDA_CHECK(cudaMemcpy(h_B, d_B, bytes, cudaMemcpyDeviceToHost));
            check_correctness(h_A, h_B, rows, cols);
            printf("  [N=%d] tiled correctness OK\n", N);
        }

        float naive_ms = bench(rows, cols, 0, d_A, d_B);
        float tiled_ms = bench(rows, cols, 1, d_A, d_B);

        // Effective bandwidth: 2 * N*N*4 bytes (one read + one write)
        double data_gb  = 2.0 * bytes / 1e9;
        float  naive_bw = (float)(data_gb / (naive_ms * 1e-3));
        float  tiled_bw = (float)(data_gb / (tiled_ms * 1e-3));
        float  speedup  = naive_ms / tiled_ms;

        printf("N=%-7d  naive=%6.3f ms  tiled=%6.3f ms  "
               "naive=%6.1f GB/s  tiled=%6.1f GB/s  speedup=%.2fx\n",
               N, naive_ms, tiled_ms, naive_bw, tiled_bw, speedup);

        // Print CSV rows to stdout for piping into benchmark.csv
        // format: kernel,variant,size,block_size,time_ms,throughput,notes
        printf("CSV: transpose,naive,%d,%d,%.4f,%.2f,bandwidth_GBs\n",
               N, TILE, naive_ms, naive_bw);
        printf("CSV: transpose,tiled,%d,%d,%.4f,%.2f,bandwidth_GBs\n",
               N, TILE, tiled_ms, tiled_bw);

        CUDA_CHECK(cudaFree(d_A));
        CUDA_CHECK(cudaFree(d_B));
        free(h_A);
        free(h_B);
    }

    return 0;
}
