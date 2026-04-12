// gemm.cu — GEMM: naive CUDA vs register-tiled CUDA vs cuBLAS sgemm
// C = alpha*A*B + beta*C, square matrices, FP32
// Benchmark: TFLOPS, 3 warmup + 5 measure, take median
// Hardware target: RTX 3080, sm_86

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string>
#include <algorithm>
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

// ── Kernel v1: Naive ─────────────────────────────────────────────────────────

// One thread computes one output element via a dot-product over K.
// A is M×K, B is K×N, C is M×N (all row-major).
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

// ── Kernel v2: Register-Tiled GEMM ───────────────────────────────────────────
//
// Tile configuration (compile-time constants):
//   BM=128, BN=128  — output tile per block (rows × cols)
//   BK=8            — K-strip depth per smem load
//   TM=8,  TN=8     — output elements per thread (register tile)
//   Threads/block:  (BM/TM) × (BN/TN) = 16 × 16 = 256
//
// Shared memory layout:
//   As[BM][BK+1] — A stored non-transposed in smem, +1 pad eliminates all
//                  bank conflicts in inner-loop reads (gcd(BK+1, 32)=gcd(9,32)=1
//                  means BK+1=9 stride cycles all 32 banks without collision).
//   Bs[BK][BN+4] — B stored row-major; +4 pad reduces but does not eliminate
//                  the inherent 4-way bank conflicts (stride TN=8, 32/gcd(8,32)=4).
//                  Acceptable: kernel is compute-bound at large N.
//
// Global → smem loading (using float4 = ldg.128):
//   A: thread tid=(ty*16+tx): a_row=tid/2 (0..127), a_col4=(tid%2)*4 (0 or 4)
//      pairs of consecutive threads load 8 contiguous floats from one A-row ✓
//   B: b_row=tid/32 (0..7), b_col4=(tid%32)*4 (0,4,…,124)
//      each warp loads a full row of the BN-tile → perfectly coalesced ✓
//
// Inner computation: register outer-product accumulation
//   for k in 0..BK-1:
//       a_reg[TM]  ← As[ty*TM + 0..TM-1][k]
//       b_reg[TN]  ← Bs[k][tx*TN + 0..TN-1]
//       C_reg[TM][TN] += outer(a_reg, b_reg)        ← TM*TN FMAs, high ILP

#define BM 128
#define BN 128
#define BK 8
#define TM 8
#define TN 8

__global__ void gemm_register_tiled(const float* __restrict__ A,
                                    const float* __restrict__ B,
                                    float* __restrict__       C,
                                    int M, int N, int K) {
    // ── Thread and block coordinates ─────────────────────────────────────────
    const int tx  = threadIdx.x;          // 0..15: output-column thread index
    const int ty  = threadIdx.y;          // 0..15: output-row thread index
    const int tid = ty * (BN / TN) + tx;  // 0..255: linear thread id within block

    const int bm = blockIdx.y * BM;       // first global row this block owns
    const int bn = blockIdx.x * BN;       // first global col this block owns

    // ── Shared memory ────────────────────────────────────────────────────────
    // As[BM][BK+1]: non-transposed A smem, +1 column pad removes bank conflicts.
    // Bs[BK][BN+4]: B smem, +4 column pad (see design notes above).
    __shared__ float As[BM][BK + 1];
    __shared__ float Bs[BK][BN + 4];

    // ── Per-thread register accumulators ─────────────────────────────────────
    float C_reg[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; i++)
        #pragma unroll
        for (int j = 0; j < TN; j++)
            C_reg[i][j] = 0.f;

    // ── Pre-compute A-tile loading indices (constant across k_tile loop) ─────
    // Each thread loads one float4 from A and one float4 from B per k-tile.
    // A: (BM * BK) / 256 = 4 floats per thread, loaded as one float4.
    //    a_row  = tid / 2         → 0..127 (each row handled by 2 threads)
    //    a_col4 = (tid % 2) * 4   → 0 or 4 (two float4 halves of the 8-wide K-strip)
    const int a_row  = tid / 2;
    const int a_col4 = (tid % 2) * 4;

    // B: (BK * BN) / 256 = 4 floats per thread, loaded as one float4.
    //    b_row  = tid / 32        → 0..7  (one K-row per 32-thread group)
    //    b_col4 = (tid % 32) * 4  → 0,4,8,…,124 (32 float4s span full BN width)
    const int b_row  = tid / (BN / 4);
    const int b_col4 = (tid % (BN / 4)) * 4;

    // ── Main K-strip loop ────────────────────────────────────────────────────
    for (int k_tile = 0; k_tile < K; k_tile += BK) {

        // ── Load A tile into As (float4, with boundary guard) ────────────────
        {
            const int gm = bm + a_row;
            const int gk = k_tile + a_col4;
            if (gm < M && gk + 3 < K) {
                // Fast path: aligned float4 load (ldg.128)
                float4 a4 = *reinterpret_cast<const float4*>(A + gm * K + gk);
                As[a_row][a_col4 + 0] = a4.x;
                As[a_row][a_col4 + 1] = a4.y;
                As[a_row][a_col4 + 2] = a4.z;
                As[a_row][a_col4 + 3] = a4.w;
            } else {
                // Boundary: scalar, zero-pad out-of-bounds
                #pragma unroll
                for (int dk = 0; dk < 4; dk++) {
                    int k = gk + dk, m = gm;
                    As[a_row][a_col4 + dk] = (m < M && k < K) ? A[m * K + k] : 0.f;
                }
            }
        }

        // ── Load B tile into Bs (float4, with boundary guard) ────────────────
        {
            const int gk = k_tile + b_row;
            const int gn = bn + b_col4;
            if (gk < K && gn + 3 < N) {
                float4 b4 = *reinterpret_cast<const float4*>(B + gk * N + gn);
                Bs[b_row][b_col4 + 0] = b4.x;
                Bs[b_row][b_col4 + 1] = b4.y;
                Bs[b_row][b_col4 + 2] = b4.z;
                Bs[b_row][b_col4 + 3] = b4.w;
            } else {
                #pragma unroll
                for (int dn = 0; dn < 4; dn++) {
                    int k = gk, n = gn + dn;
                    Bs[b_row][b_col4 + dn] = (k < K && n < N) ? B[k * N + n] : 0.f;
                }
            }
        }

        __syncthreads();

        // ── Register outer-product: accumulate BK steps ──────────────────────
        // Each k-step: load TM A-values and TN B-values into registers, then
        // compute TM×TN FMAs — high arithmetic intensity, high ILP.
        #pragma unroll
        for (int k = 0; k < BK; k++) {
            float a_reg[TM], b_reg[TN];

            // Load A column (ty's rows) — As[ty*TM+tm][k]
            // Bank: (ty*TM+tm)*(BK+1)+k = (ty*8+tm)*9+k; gcd(9,32)=1 → no conflict
            #pragma unroll
            for (int tm = 0; tm < TM; tm++)
                a_reg[tm] = As[ty * TM + tm][k];

            // Load B row (tx's cols) — Bs[k][tx*TN+tn]
            // Bank: k*(BN+4) + tx*TN+tn; 4-way conflict (acceptable, compute-bound)
            #pragma unroll
            for (int tn = 0; tn < TN; tn++)
                b_reg[tn] = Bs[k][tx * TN + tn];

            // Outer product → TM*TN = 64 FMAs per k-step
            #pragma unroll
            for (int tm = 0; tm < TM; tm++)
                #pragma unroll
                for (int tn = 0; tn < TN; tn++)
                    C_reg[tm][tn] += a_reg[tm] * b_reg[tn];
        }

        __syncthreads();
    }

    // ── Write C_reg to global C (float4 for TN=8, TN/4=2 stores per row) ────
    #pragma unroll
    for (int tm = 0; tm < TM; tm++) {
        const int gm = bm + ty * TM + tm;
        if (gm >= M) continue;

        // First float4: C_reg[tm][0..3]
        {
            const int gn = bn + tx * TN;
            if (gn + 3 < N) {
                float4 c4 = {C_reg[tm][0], C_reg[tm][1],
                             C_reg[tm][2], C_reg[tm][3]};
                *reinterpret_cast<float4*>(C + gm * N + gn) = c4;
            } else {
                for (int i = 0; i < 4 && gn + i < N; i++)
                    C[gm * N + gn + i] = C_reg[tm][i];
            }
        }
        // Second float4: C_reg[tm][4..7]
        {
            const int gn = bn + tx * TN + 4;
            if (gn + 3 < N) {
                float4 c4 = {C_reg[tm][4], C_reg[tm][5],
                             C_reg[tm][6], C_reg[tm][7]};
                *reinterpret_cast<float4*>(C + gm * N + gn) = c4;
            } else {
                for (int i = 0; i < 4 && gn + i < N; i++)
                    C[gm * N + gn + i] = C_reg[tm][4 + i];
            }
        }
    }
}

#undef BM
#undef BN
#undef BK
#undef TM
#undef TN

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

// ── Generic timed benchmark helper ───────────────────────────────────────────
// Runs fn() 3 warmup + 5 measured times; returns median kernel ms.
template <typename Fn>
static float bench(Fn fn) {
    for (int i = 0; i < 3; i++) fn();
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));
    float times[5];
    for (int r = 0; r < 5; r++) {
        CUDA_CHECK(cudaEventRecord(t0));
        fn();
        CUDA_CHECK(cudaEventRecord(t1));
        CUDA_CHECK(cudaEventSynchronize(t1));
        CUDA_CHECK(cudaEventElapsedTime(&times[r], t0, t1));
    }
    CUDA_CHECK(cudaEventDestroy(t0));
    CUDA_CHECK(cudaEventDestroy(t1));
    std::sort(times, times + 5);
    return times[2];
}

static float bench_naive(int M, int N, int K,
                         const float* d_A, const float* d_B, float* d_C) {
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    return bench([&]() {
        gemm_naive<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    });
}

static float bench_regtiled(int M, int N, int K,
                            const float* d_A, const float* d_B, float* d_C) {
    // Precondition: float4 load/store paths require M, N, K divisible by 4
    // for 16-byte alignment. Misaligned float4 access is UB.
    if (M % 4 != 0 || N % 4 != 0 || K % 4 != 0) {
        fprintf(stderr,
                "gemm_register_tiled requires M, N, K all divisible by 4 "
                "(got M=%d N=%d K=%d)\n", M, N, K);
        exit(1);
    }
    constexpr int BM = 128, BN = 128, TM = 8, TN = 8;
    dim3 block(BN / TN, BM / TM);                           // (16, 16) = 256 threads
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    return bench([&]() {
        gemm_register_tiled<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    });
}

static float bench_cublas(cublasHandle_t handle,
                          int M, int N, int K,
                          const float* d_A, const float* d_B, float* d_C) {
    const float alpha = 1.0f, beta = 0.0f;
    // cuBLAS is column-major; compute B^T * A^T = C^T, which gives
    // the correct row-major C = A * B without explicit transposing.
    return bench([&]() {
        CUBLAS_CHECK(cublasSgemm(handle,
                                 CUBLAS_OP_N, CUBLAS_OP_N,
                                 N, M, K,
                                 &alpha,
                                 d_B, N,   // B, ldb=N
                                 d_A, K,   // A, lda=K
                                 &beta,
                                 d_C, N)); // C, ldc=N
    });
}

// ── Main ──────────────────────────────────────────────────────────────────────

int main() {
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    const int sizes[] = {512, 1024, 2048, 4096};
    const int nsizes  = (int)(sizeof(sizes) / sizeof(sizes[0]));

    printf("%-6s  %-10s  %-12s  %-12s  %-12s  %-12s  %-8s  %s\n",
           "N", "naive_ms", "regtile_ms", "cublas_ms",
           "naive_TF", "regtile_TF", "cublas_TF", "regtile/cublas");
    printf("%s\n", std::string(100, '-').c_str());

    FILE* csv = fopen("../../benchmark.csv", "a");
    if (!csv) csv = fopen("benchmark.csv", "a");

    for (int si = 0; si < nsizes; si++) {
        int N = sizes[si];
        int M = N, K = N;
        size_t bytes_A = (size_t)M * K * sizeof(float);
        size_t bytes_B = (size_t)K * N * sizeof(float);
        size_t bytes_C = (size_t)M * N * sizeof(float);

        float* h_A   = (float*)malloc(bytes_A);
        float* h_B   = (float*)malloc(bytes_B);
        float* h_ref = (float*)malloc(bytes_C);
        float* h_tst = (float*)malloc(bytes_C);

        fill_random(h_A, M * K);
        fill_random(h_B, K * N);

        float *d_A, *d_B, *d_C;
        CUDA_CHECK(cudaMalloc(&d_A, bytes_A));
        CUDA_CHECK(cudaMalloc(&d_B, bytes_B));
        CUDA_CHECK(cudaMalloc(&d_C, bytes_C));
        CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes_B, cudaMemcpyHostToDevice));

        double flops = 2.0 * M * N * K;

        // ── Correctness checks (N ≤ 1024 to keep runtime reasonable) ─────────
        if (N <= 1024) {
            // cuBLAS reference
            CUDA_CHECK(cudaMemset(d_C, 0, bytes_C));
            bench_cublas(handle, M, N, K, d_A, d_B, d_C);
            CUDA_CHECK(cudaMemcpy(h_ref, d_C, bytes_C, cudaMemcpyDeviceToHost));

            // register-tiled correctness
            CUDA_CHECK(cudaMemset(d_C, 0, bytes_C));
            bench_regtiled(M, N, K, d_A, d_B, d_C);
            CUDA_CHECK(cudaMemcpy(h_tst, d_C, bytes_C, cudaMemcpyDeviceToHost));
            check_correctness(h_ref, h_tst, M * N);
            printf("  [N=%d] regtiled correctness vs cuBLAS OK\n", N);

            // naive correctness
            CUDA_CHECK(cudaMemset(d_C, 0, bytes_C));
            bench_naive(M, N, K, d_A, d_B, d_C);
            CUDA_CHECK(cudaMemcpy(h_tst, d_C, bytes_C, cudaMemcpyDeviceToHost));
            check_correctness(h_ref, h_tst, M * N);
            printf("  [N=%d] naive correctness vs cuBLAS OK\n", N);
        }

        // ── Timing ────────────────────────────────────────────────────────────
        float naive_ms   = -1.f, naive_tfl   = -1.f;
        float regtile_ms = -1.f, regtile_tfl = -1.f;

        // naive: skip for N=4096 (too slow for repeated timing)
        if (N <= 2048) {
            naive_ms  = bench_naive(M, N, K, d_A, d_B, d_C);
            naive_tfl = (float)(flops / (naive_ms * 1e-3) / 1e12);
        }

        // register-tiled: run for all sizes
        regtile_ms  = bench_regtiled(M, N, K, d_A, d_B, d_C);
        regtile_tfl = (float)(flops / (regtile_ms * 1e-3) / 1e12);

        float cublas_ms  = bench_cublas(handle, M, N, K, d_A, d_B, d_C);
        float cublas_tfl = (float)(flops / (cublas_ms * 1e-3) / 1e12);

        float rt_vs_cublas = regtile_tfl / cublas_tfl * 100.f;

        // ── Print ─────────────────────────────────────────────────────────────
        if (naive_ms > 0) {
            printf("N=%-5d  naive=%7.2f ms  regtile=%7.2f ms  cublas=%6.2f ms  "
                   "%.3f TF  %.3f TF  %.3f TF  %.1f%%  %s\n",
                   N, naive_ms, regtile_ms, cublas_ms,
                   naive_tfl, regtile_tfl, cublas_tfl, rt_vs_cublas,
                   rt_vs_cublas >= 50.f ? "EXCELLENT" :
                   rt_vs_cublas >= 30.f ? "PASS" : "BELOW_TARGET");
        } else {
            printf("N=%-5d  naive=(skip)     regtile=%7.2f ms  cublas=%6.2f ms  "
                   " —          %.3f TF  %.3f TF  %.1f%%  %s\n",
                   N, regtile_ms, cublas_ms,
                   regtile_tfl, cublas_tfl, rt_vs_cublas,
                   rt_vs_cublas >= 50.f ? "EXCELLENT" :
                   rt_vs_cublas >= 30.f ? "PASS" : "BELOW_TARGET");
        }

        // ── Append to benchmark.csv ───────────────────────────────────────────
        // Format: kernel,variant,size,block_size,time_ms,throughput,notes
        if (csv) {
            if (naive_ms > 0)
                fprintf(csv, "gemm,naive,%d,%d,%.4f,%.4f TFLOPS,\n",
                        N, TILE, naive_ms, naive_tfl);
            fprintf(csv,
                    "gemm,register_tiled,%d,BM128_BN128_BK8_TM8_TN8,%.4f,%.4f TFLOPS,"
                    "regtile_vs_cublas=%.1f%%\n",
                    N, regtile_ms, regtile_tfl, rt_vs_cublas);
            fprintf(csv, "gemm,cublas,%d,N/A,%.4f,%.4f TFLOPS,\n",
                    N, cublas_ms, cublas_tfl);
        }

        CUDA_CHECK(cudaFree(d_A));
        CUDA_CHECK(cudaFree(d_B));
        CUDA_CHECK(cudaFree(d_C));
        free(h_A); free(h_B); free(h_ref); free(h_tst);
    }

    if (csv) {
        fclose(csv);
        printf("\nResults appended to benchmark.csv\n");
    }

    CUBLAS_CHECK(cublasDestroy(handle));
    return 0;
}
