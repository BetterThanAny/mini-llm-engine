#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

// ── Kernel ────────────────────────────────────────────────────────────────────
__global__ void vector_add(const float* __restrict__ a,
                           const float* __restrict__ b,
                           float* __restrict__ c,
                           int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

// ── Helpers ───────────────────────────────────────────────────────────────────
#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err = (call);                                          \
        if (err != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error %s:%d: %s\n",                     \
                    __FILE__, __LINE__, cudaGetErrorString(err));          \
            exit(1);                                                       \
        }                                                                  \
    } while (0)

static float bench_block_size(int n, int block_size) {
    size_t bytes = n * sizeof(float);

    float *h_a = (float*)malloc(bytes);
    float *h_b = (float*)malloc(bytes);
    float *h_c = (float*)malloc(bytes);

    for (int i = 0; i < n; i++) { h_a[i] = 1.0f; h_b[i] = 2.0f; }

    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));

    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    int grid_size = (n + block_size - 1) / block_size;

    // Warmup
    for (int i = 0; i < 3; i++)
        vector_add<<<grid_size, block_size>>>(d_a, d_b, d_c, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Measure 5 runs, take median
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    float times[5];
    for (int r = 0; r < 5; r++) {
        CUDA_CHECK(cudaEventRecord(start));
        vector_add<<<grid_size, block_size>>>(d_a, d_b, d_c, n);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&times[r], start, stop));
    }

    // Sort for median
    for (int i = 0; i < 4; i++)
        for (int j = i+1; j < 5; j++)
            if (times[j] < times[i]) { float t = times[i]; times[i] = times[j]; times[j] = t; }
    float median_ms = times[2];

    // Correctness check
    CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));
    for (int i = 0; i < n; i++) {
        if (fabsf(h_c[i] - 3.0f) > 1e-5f) {
            fprintf(stderr, "Correctness FAILED at index %d: got %f\n", i, h_c[i]);
            exit(1);
        }
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    free(h_a); free(h_b); free(h_c);

    return median_ms;
}

int main() {
    // N = 1 << 24 ≈ 16M elements, 3 arrays × 64 MB = 192 MB total memory traffic
    const int N = 1 << 24;
    const float bytes_gb = 3.0f * N * sizeof(float) / 1e9f;

    int block_sizes[] = {32, 64, 128, 256, 512};
    int num_blocks = sizeof(block_sizes) / sizeof(block_sizes[0]);

    printf("vector_add  N=%d (%.0f MB per array)\n", N, N * sizeof(float) / 1e6f);
    printf("%-12s %-12s %-12s\n", "block_size", "time_ms", "GB/s");
    printf("%-12s %-12s %-12s\n", "----------", "-------", "----");

    for (int i = 0; i < num_blocks; i++) {
        int bs = block_sizes[i];
        float ms = bench_block_size(N, bs);
        float gbs = bytes_gb / (ms / 1000.0f);
        printf("%-12d %-12.3f %-12.1f\n", bs, ms, gbs);
    }

    printf("\nCorrectness: PASSED\n");
    return 0;
}
