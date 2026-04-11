// gemm.cu — GEMM: naive CUDA vs cuBLAS sgemm
// C = alpha*A*B + beta*C, square matrices, FP32
// Benchmark: TFLOPS, 3 warmup + 5 measure, take median
// Hardware target: RTX 3080, sm_86

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error %s:%d  %s\n",                          \
                    __FILE__, __LINE__, cudaGetErrorString(err));               \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

#define CUBLAS_CHECK(call)                                                      \
    do {                                                                        \
        cublasStatus_t st = (call);                                             \
        if (st != CUBLAS_STATUS_SUCCESS) {                                      \
            fprintf(stderr, "cuBLAS error %s:%d  status=%d\n",                 \
                    __FILE__, __LINE__, (int)st);                               \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

static const int TILE = 32;  // block tile size for naive kernel

// ── Kernels ───────────────────────────────────────────────────────────────────

// Naive: one thread computes one element of C by accumulating a dot product.
// A is M×K, B is K×N, C is M×N (row-major).
__global__ void gemm_naive(const float* __restrict__ A,
                           const float* __restrict__ B,
                           float* __restrict__       C,
                           int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    float sum = 0.0f;
    for (int k = 0; k < K; k++)
        sum += A[row * K + k] * B[k * N + col];
    C[row * N + col] = sum;
}

// ── Helpers ───────────────────────────────────────────────────────────────────

static void fill_random(float* h, int n) {
    for (int i = 0; i < n; i++) h[i] = (float)rand() / RAND_MAX * 0.1f;
}

// Check C_ref vs C_test; tolerances for FP32 GEMM accumulation
static void check_correctness(const float* ref, const float* test, int n,
                               float atol = 1e-3f) {
    float max_err = 0.0f;
    int   max_idx = 0;
    for (int i = 0; i < n; i++) {
        float err = fabsf(ref[i] - test[i]);
        if (err > max_err) { max_err = err; max_idx = i; }
    }
    if (max_err > atol) {
        fprintf(stderr,
                "Correctness FAILED: max_err=%.6f at idx=%d (ref=%f test=%f)\n",
                max_err, max_idx, ref[max_idx], test[max_idx]);
        exit(1);
    }
}

// Returns median kernel time in milliseconds (3 warmup, 5 measured).
// kind: 0 = naive, 1 = cuBLAS
static float bench_naive(int M, int N, int K,
                         const float* d_A, const float* d_B, float* d_C) {
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);

    for (int i = 0; i < 3; i++)
        gemm_naive<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));
    float times[5];
    for (int r = 0; r < 5; r++) {
        CUDA_CHECK(cudaEventRecord(t0));
        gemm_naive<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
        CUDA_CHECK(cudaEventRecord(t1));
        CUDA_CHECK(cudaEventSynchronize(t1));
        CUDA_CHECK(cudaEventElapsedTime(&times[r], t0, t1));
    }
    CUDA_CHECK(cudaEventDestroy(t0));
    CUDA_CHECK(cudaEventDestroy(t1));
    for (int i = 0; i < 4; i++)
        for (int j = i + 1; j < 5; j++)
            if (times[j] < times[i]) { float t = times[i]; times[i] = times[j]; times[j] = t; }
    return times[2];
}

static float bench_cublas(cublasHandle_t handle,
                          int M, int N, int K,
                          const float* d_A, const float* d_B, float* d_C) {
    const float alpha = 1.0f, beta = 0.0f;

    // cuBLAS is column-major; compute B^T * A^T = C^T, which gives
    // the correct row-major C = A * B without explicit transposing.
    // (i.e. call cublasSgemm(N, M, K, B, N, A, K, C, N))
    auto launch = [&]() {
        CUBLAS_CHECK(cublasSgemm(handle,
                                 CUBLAS_OP_N, CUBLAS_OP_N,
                                 N, M, K,
                                 &alpha,
                                 d_B, N,   // B, ldb=N
                                 d_A, K,   // A, lda=K
                                 &beta,
                                 d_C, N)); // C, ldc=N
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
    for (int i = 0; i < 4; i++)
        for (int j = i + 1; j < 5; j++)
            if (times[j] < times[i]) { float t = times[i]; times[i] = times[j]; times[j] = t; }
    return times[2];
}

// ── Main ──────────────────────────────────────────────────────────────────────

int main() {
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    // Square matrix sizes
    const int sizes[] = {512, 1024, 2048, 4096};
    const int nsizes  = (int)(sizeof(sizes) / sizeof(sizes[0]));

    printf("%-8s  %-12s  %-12s  %-14s  %-14s  %-10s  %s\n",
           "N", "naive_ms", "cublas_ms",
           "naive_TFLOPS", "cublas_TFLOPS", "gap_ratio",
           "cublas/naive");
    printf("%s\n", std::string(90, '-').c_str());

    for (int si = 0; si < nsizes; si++) {
        int N = sizes[si];
        int M = N, K = N;
        size_t bytes_A = (size_t)M * K * sizeof(float);
        size_t bytes_B = (size_t)K * N * sizeof(float);
        size_t bytes_C = (size_t)M * N * sizeof(float);

        float* h_A   = (float*)malloc(bytes_A);
        float* h_B   = (float*)malloc(bytes_B);
        float* h_ref = (float*)malloc(bytes_C);  // cuBLAS result
        float* h_tst = (float*)malloc(bytes_C);  // naive result

        fill_random(h_A, M * K);
        fill_random(h_B, K * N);

        float *d_A, *d_B, *d_C;
        CUDA_CHECK(cudaMalloc(&d_A, bytes_A));
        CUDA_CHECK(cudaMalloc(&d_B, bytes_B));
        CUDA_CHECK(cudaMalloc(&d_C, bytes_C));
        CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes_B, cudaMemcpyHostToDevice));

        // Correctness: compare naive vs cuBLAS (small sizes only)
        if (N <= 1024) {
            // cuBLAS reference
            CUDA_CHECK(cudaMemset(d_C, 0, bytes_C));
            bench_cublas(handle, M, N, K, d_A, d_B, d_C);
            CUDA_CHECK(cudaMemcpy(h_ref, d_C, bytes_C, cudaMemcpyDeviceToHost));

            // Naive result
            CUDA_CHECK(cudaMemset(d_C, 0, bytes_C));
            bench_naive(M, N, K, d_A, d_B, d_C);
            CUDA_CHECK(cudaMemcpy(h_tst, d_C, bytes_C, cudaMemcpyDeviceToHost));

            check_correctness(h_ref, h_tst, M * N);
            printf("  [N=%d] naive correctness vs cuBLAS OK\n", N);
        }

        // Skip naive benchmark for large N (too slow for repeated timing)
        float naive_ms  = -1.0f;
        float naive_tfl = -1.0f;
        if (N <= 2048) {
            naive_ms  = bench_naive(M, N, K, d_A, d_B, d_C);
            // FLOPs for M×N×K GEMM: 2*M*N*K
            double flops = 2.0 * M * N * K;
            naive_tfl = (float)(flops / (naive_ms * 1e-3) / 1e12);
        }

        float cublas_ms  = bench_cublas(handle, M, N, K, d_A, d_B, d_C);
        double flops     = 2.0 * M * N * K;
        float cublas_tfl = (float)(flops / (cublas_ms * 1e-3) / 1e12);

        if (naive_ms > 0) {
            float ratio = naive_ms / cublas_ms;
            printf("N=%-6d  naive=%8.3f ms  cublas=%6.3f ms  "
                   "naive=%.4f TFLOPS  cublas=%.4f TFLOPS  "
                   "gap=%.1fx\n",
                   N, naive_ms, cublas_ms, naive_tfl, cublas_tfl, ratio);
            printf("CSV: gemm,naive,%d,%d,%.4f,%.4f,TFLOPS\n",
                   N, TILE, naive_ms, naive_tfl);
        } else {
            printf("N=%-6d  naive=  (skipped)     cublas=%6.3f ms  "
                   "cublas=%.4f TFLOPS\n",
                   N, cublas_ms, cublas_tfl);
        }
        printf("CSV: gemm,cublas,%d,N/A,%.4f,%.4f,TFLOPS\n",
               N, cublas_ms, cublas_tfl);

        CUDA_CHECK(cudaFree(d_A));
        CUDA_CHECK(cudaFree(d_B));
        CUDA_CHECK(cudaFree(d_C));
        free(h_A); free(h_B); free(h_ref); free(h_tst);
    }

    CUBLAS_CHECK(cublasDestroy(handle));
    return 0;
}
