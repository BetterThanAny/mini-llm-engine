#include "ops_cuda.h"
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <mutex>

// ── Error-check helpers ───────────────────────────────────────────────────────

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

#define CUBLAS_CHECK(call) do { \
    cublasStatus_t st = (call); \
    if (st != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error at %s:%d: status=%d\n", \
                __FILE__, __LINE__, (int)st); \
        exit(1); \
    } \
} while(0)

// ── cuBLAS global handle (lazy init, single-threaded) ─────────────────────────

static cublasHandle_t g_cublas_handle = nullptr;
static std::once_flag  g_cublas_init_flag;

static void init_cublas() {
    CUBLAS_CHECK(cublasCreate(&g_cublas_handle));
    // Use tensor cores where possible; fallback to standard FP16 math
    CUBLAS_CHECK(cublasSetMathMode(g_cublas_handle, CUBLAS_DEFAULT_MATH));
}

static cublasHandle_t get_cublas_handle() {
    std::call_once(g_cublas_init_flag, init_cublas);
    return g_cublas_handle;
}

// ═════════════════════════════════════════════════════════════════════════════
// 1.  RMSNorm  — hand-written warp-shuffle FP16 kernel
//
//  Formula:  y[i] = (x[i] / rms(x)) * w[i]
//            rms  = sqrt( mean(x^2) + eps )
//
//  Port of rmsnorm_v2_warp (FP32) to FP16 I/O with FP32 accumulation:
//   - One CUDA block per row.
//   - 128 threads → 4 warps → shared memory = 4 floats (one per warp).
//   - Phase 1: vectorized half2 loads, accumulate sum(x^2) as float,
//              warp-shuffle reduce, cross-warp reduce via smem.
//   - Phase 2: normalize + scale, store back as half.
// ─────────────────────────────────────────────────────────────────────────────

__global__ void rmsnorm_v2_fp16(const __half* __restrict__ x,
                                 const __half* __restrict__ w,
                                 __half*       __restrict__ y,
                                 int hidden, float eps)
{
    int row = blockIdx.x;
    const __half* xr = x + row * hidden;
    __half*       yr = y + row * hidden;

    // ── Phase 1: compute sum(x^2) in FP32 ─────────────────────────────────
    float sum = 0.0f;

    // Fast path: load two halves at a time using __half2.
    // Requires hidden to be even (TinyLlama hidden=2048: always true).
    int i = threadIdx.x * 2;
    for (; i + 1 < hidden; i += blockDim.x * 2) {
        // Load a pair of __half values as __half2
        __half2 v = *reinterpret_cast<const __half2*>(xr + i);
        float a = __half2float(v.x);
        float b = __half2float(v.y);
        sum += a * a + b * b;
    }
    // Scalar tail (handles odd hidden sizes)
    for (int j = threadIdx.x + (hidden & ~1); j < hidden; j += blockDim.x) {
        float a = __half2float(xr[j]);
        sum += a * a;
    }

    // Intra-warp reduce
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, offset);

    // Cross-warp reduce via shared memory (one float per warp)
    extern __shared__ float smem[];   // smem[num_warps] == smem[4]
    int warp_id = threadIdx.x >> 5;
    int lane    = threadIdx.x & 31;
    if (lane == 0) smem[warp_id] = sum;
    __syncthreads();

    // First warp reduces the per-warp partial sums
    int num_warps = blockDim.x >> 5;   // = 4 for blockDim.x=128
    if (warp_id == 0) {
        sum = (lane < num_warps) ? smem[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1)
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        if (lane == 0) smem[0] = sum;
    }
    __syncthreads();

    float rms_inv = rsqrtf(smem[0] / (float)hidden + eps);

    // ── Phase 2: normalize and scale ──────────────────────────────────────
    i = threadIdx.x * 2;
    for (; i + 1 < hidden; i += blockDim.x * 2) {
        __half2 vx = *reinterpret_cast<const __half2*>(xr + i);
        __half2 vw = *reinterpret_cast<const __half2*>(w  + i);
        float x0 = __half2float(vx.x) * rms_inv * __half2float(vw.x);
        float x1 = __half2float(vx.y) * rms_inv * __half2float(vw.y);
        __half2 vy = {__float2half(x0), __float2half(x1)};
        *reinterpret_cast<__half2*>(yr + i) = vy;
    }
    // Scalar tail
    for (int j = threadIdx.x + (hidden & ~1); j < hidden; j += blockDim.x) {
        float v = __half2float(xr[j]) * rms_inv * __half2float(w[j]);
        yr[j] = __float2half(v);
    }
}

// Public wrapper
void rms_norm_cuda(const Tensor& x, const Tensor& weight, Tensor& out, float eps)
{
    // Validate
    if (x.device != Device::CUDA || weight.device != Device::CUDA ||
        out.device != Device::CUDA) {
        fprintf(stderr, "rms_norm_cuda: all tensors must be on CUDA\n");
        exit(1);
    }

    // Expect x shape [rows, hidden] or [hidden] (single row)
    int rows   = (x.ndim >= 2) ? x.shape[0] : 1;
    int hidden = x.shape[x.ndim - 1];
    // For ndim > 2, fold leading dims into rows
    for (int d = 1; d < x.ndim - 1; d++) rows *= x.shape[d];

    const __half* xp  = x.fp16();
    const __half* wp  = weight.fp16();
    __half*       yp  = out.fp16();

    // grid = rows, block = 128, smem = 4 floats (one per warp)
    dim3 grid(rows), block(128);
    size_t smem = 4 * sizeof(float);
    rmsnorm_v2_fp16<<<grid, block, smem>>>(xp, wp, yp, hidden, eps);
    CUDA_CHECK(cudaGetLastError());
}

// ═════════════════════════════════════════════════════════════════════════════
// 2.  Softmax in-place along last dim (FP16)
//
//  Algorithm: block reduction
//    - Each block handles one row.
//    - Pass 1: find max (numerically stable).
//    - Pass 2: compute exp(x - max) and sum.
//    - Pass 3: divide by sum.
//
//  For W4, this covers attention score rows (seq_len up to 2048).
//  block = 256, smem = 256 floats for the per-thread partial results.
// ─────────────────────────────────────────────────────────────────────────────

__global__ void softmax_fp16_kernel(__half* __restrict__ x, int row_len)
{
    extern __shared__ float sdata[];   // blockDim.x floats

    int row = blockIdx.x;
    __half* xr = x + row * row_len;

    // ── Pass 1: load into shared mem as float, find max ───────────────────
    float thread_max = -1e30f;
    for (int i = threadIdx.x; i < row_len; i += blockDim.x) {
        float v = __half2float(xr[i]);
        sdata[threadIdx.x] = v;   // cache for later (only valid for first tile)
        thread_max = fmaxf(thread_max, v);
    }

    // Block reduce max
    sdata[threadIdx.x] = thread_max;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            sdata[threadIdx.x] = fmaxf(sdata[threadIdx.x], sdata[threadIdx.x + s]);
        __syncthreads();
    }
    float row_max = sdata[0];
    __syncthreads();

    // ── Pass 2: exp(x - max), accumulate sum ─────────────────────────────
    float thread_sum = 0.0f;
    for (int i = threadIdx.x; i < row_len; i += blockDim.x) {
        float v = expf(__half2float(xr[i]) - row_max);
        xr[i] = __float2half(v);
        thread_sum += v;
    }

    // Block reduce sum
    sdata[threadIdx.x] = thread_sum;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    float inv_sum = 1.0f / sdata[0];
    __syncthreads();

    // ── Pass 3: normalize ─────────────────────────────────────────────────
    for (int i = threadIdx.x; i < row_len; i += blockDim.x) {
        xr[i] = __float2half(__half2float(xr[i]) * inv_sum);
    }
}

void softmax_cuda(Tensor& x)
{
    if (x.device != Device::CUDA) {
        fprintf(stderr, "softmax_cuda: tensor must be on CUDA\n");
        exit(1);
    }

    int row_len = x.shape[x.ndim - 1];
    int rows    = (int)(x.numel / row_len);

    __half* xp = x.fp16();
    int block  = 256;
    size_t smem = block * sizeof(float);
    softmax_fp16_kernel<<<rows, block, smem>>>(xp, row_len);
    CUDA_CHECK(cudaGetLastError());
}

// ═════════════════════════════════════════════════════════════════════════════
// 3.  RoPE — Rotary Position Embedding in-place (FP16)
//
//  Tensors q, k have shape [seq_len, num_heads, head_dim].
//  Each thread handles one (token, head, dim_pair) triple — two elements.
//
//  Grid:   (num_tokens, num_heads)
//  Block:  (head_dim / 2)
// ─────────────────────────────────────────────────────────────────────────────

__global__ void rope_fp16_kernel(__half* __restrict__ x,
                                  int num_heads,
                                  int head_dim,
                                  int position_offset)
{
    int tok  = blockIdx.x;
    int head = blockIdx.y;
    int j    = threadIdx.x;   // dim pair index: processes elements j, j + half_dim

    int half_dim = head_dim / 2;
    if (head >= num_heads || j >= half_dim) return;

    // Half-half RoPE (matches HuggingFace LLaMA): pair (j, j + half_dim)
    int base = tok * num_heads * head_dim + head * head_dim;
    __half* xp = x + base;

    float x0 = __half2float(xp[j]);
    float x1 = __half2float(xp[j + half_dim]);

    int pos = position_offset + tok;
    float freq  = 1.0f / powf(10000.0f, 2.0f * (float)j / (float)head_dim);
    float angle = (float)pos * freq;
    float c = cosf(angle);
    float s = sinf(angle);

    xp[j]             = __float2half(x0 * c - x1 * s);
    xp[j + half_dim]  = __float2half(x0 * s + x1 * c);
}

void rope_cuda(Tensor& q, Tensor& k, int position_offset)
{
    if (q.device != Device::CUDA || k.device != Device::CUDA) {
        fprintf(stderr, "rope_cuda: tensors must be on CUDA\n");
        exit(1);
    }
    // q: [seq_len, num_heads, head_dim]
    // k: [seq_len, num_kv_heads, head_dim]
    int seq_len    = q.shape[0];
    int num_heads  = q.shape[1];
    int head_dim   = q.shape[2];

    int num_kv_heads = k.shape[1];

    // block covers head_dim/2 pairs; grid covers (seq_len, num_heads)
    dim3 block_q(head_dim / 2);
    dim3 grid_q(seq_len, num_heads);
    rope_fp16_kernel<<<grid_q, block_q>>>(q.fp16(), num_heads, head_dim, position_offset);
    CUDA_CHECK(cudaGetLastError());

    dim3 block_k(head_dim / 2);
    dim3 grid_k(seq_len, num_kv_heads);
    rope_fp16_kernel<<<grid_k, block_k>>>(k.fp16(), num_kv_heads, head_dim, position_offset);
    CUDA_CHECK(cudaGetLastError());
}

// ═════════════════════════════════════════════════════════════════════════════
// 4.  GEMM  — cuBLAS FP16 wrapper
//
//  C = alpha * A @ B^T + beta * C
//  A: [M, K]  B: [N, K]  C: [M, N]  (all FP16, row-major)
//
//  cuBLAS is column-major.  To stay in row-major we exploit the identity:
//    row-major A @ B^T  =  col-major (B^T)^T @ A^T
//                        =  col-major B @ A^T
//  so we call: cublasGemmEx(B, A, C) with transa=CUBLAS_OP_T, transb=CUBLAS_OP_N
//  giving col-major C[N, M]  ≡  row-major C[M, N].
// ─────────────────────────────────────────────────────────────────────────────

void gemm_fp16(const Tensor& A, const Tensor& B, Tensor& C,
               float alpha, float beta)
{
    if (A.device != Device::CUDA || B.device != Device::CUDA ||
        C.device != Device::CUDA) {
        fprintf(stderr, "gemm_fp16: all tensors must be on CUDA\n");
        exit(1);
    }

    // A: [M, K],  B: [N, K],  C: [M, N]
    int M = A.shape[0];
    int K = A.shape[1];
    int N = B.shape[0];

    const __half* A_ptr = A.fp16();
    const __half* B_ptr = B.fp16();
    __half*       C_ptr = C.fp16();

    // CUBLAS_COMPUTE_32F requires alpha/beta as float*, not __half*
    float f_alpha = alpha;
    float f_beta  = beta;

    cublasHandle_t handle = get_cublas_handle();

    // Call convention (column-major interpretation):
    //   Result C[N, M] col-major  =  B[N, K] col-major  @  A[M, K]^T col-major
    //   => op(B) = no-trans (B is [K, N] in col-major with ldb=K)
    //   => op(A) = transpose  (A is [K, M] in col-major with lda=K)
    CUBLAS_CHECK(cublasGemmEx(
        handle,
        CUBLAS_OP_T,      // transa: transpose A (row-major A → col-major A^T)
        CUBLAS_OP_N,      // transb: no-transpose B
        N, M, K,          // m, n, k  (C is N×M in col-major)
        &f_alpha,
        B_ptr, CUDA_R_16F, K,   // B [N, K] row-major → [K, N] col-major, ldb=K
        A_ptr, CUDA_R_16F, K,   // A [M, K] row-major → [K, M] col-major, lda=K
        &f_beta,
        C_ptr, CUDA_R_16F, N,   // C [M, N] row-major → [N, M] col-major, ldc=N
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

// ═════════════════════════════════════════════════════════════════════════════
// 5.  Attention  — Flash Attention v1 (W6, replaces W5 three-kernel naive impl)
//
//  Reference: Dao et al. 2022, "FlashAttention: Fast and Memory-Efficient Exact
//             Attention with IO-Awareness"
//
//  Key idea: tile K and V into blocks of Bc rows; maintain running online-softmax
//  state (m, l, O) so the full scores buffer is never materialised in VRAM.
//  Memory: O(Bc·HD) shared mem per block vs O(q_len·NH·kv_len) for naive impl.
//
//  Q:   [q_len,       num_q_heads,  head_dim]  FP16
//  K:   [max_seq_len, num_kv_heads, head_dim]  FP16  (first kv_len rows valid)
//  V:   [max_seq_len, num_kv_heads, head_dim]  FP16
//  out: [q_len,       num_q_heads,  head_dim]  FP16
//
//  Supports GQA: kv_repeat = num_q_heads / num_kv_heads
// ─────────────────────────────────────────────────────────────────────────────

// KV tile width.  For TinyLlama head_dim = 64, so Bc = head_dim = block width.
// This lets thread d both compute the score for KV slot d *and* own output dim d.
static constexpr int FLASH_BC = 64;

// ─── Flash Attention kernel ───────────────────────────────────────────────────
//
// Grid:  (q_len, num_q_heads)   — one block per (query position, query head)
// Block: FLASH_BC = 64 threads  — must equal head_dim for TinyLlama
//
// Shared memory layout (each block, ~16.9 KB):
//   q_smem [HD]            __half  — query row, loaded once
//   K_smem [Bc][HD+2]      __half  — current K tile (+2 col pad: bank-conflict)
//   V_smem [Bc][HD+2]      __half  — current V tile
//   s_smem [Bc]            float   — QK scores for current tile
//
// Per-thread registers:
//   O_d  — unnormalised output accumulator for dimension d
//   m    — running max (identical across all threads after each tile)
//   l    — running normaliser Σ exp(s - m)
//
// Online softmax recurrence (Milakov & Gimelshein 2018 / FA eq. 3):
//   On each tile:
//     m_new = max(m, max(s_tile))
//     alpha = exp(m - m_new)            ← rescales old state
//     l     = alpha * l + Σ exp(s_j - m_new)
//     O_d   = alpha * O_d + Σ exp(s_j - m_new) * V[j][d]
//   At end: output[d] = O_d / l

__global__ void flash_attn_kernel(
    const __half* __restrict__ Q,    // [q_len, num_q_heads, HD]
    const __half* __restrict__ K,    // [kv_max_len, num_kv_heads, HD]
    const __half* __restrict__ V,    // [kv_max_len, num_kv_heads, HD]
    __half*       __restrict__ out,  // [q_len, num_q_heads, HD]
    int q_len,  int kv_len,
    int num_q_heads, int num_kv_heads,
    int head_dim, float scale)
{
    const int qi      = blockIdx.x;
    const int qh      = blockIdx.y;
    const int d       = threadIdx.x;              // [0, HD)
    const int kv_head = qh * num_kv_heads / num_q_heads;   // GQA mapping
    // Causal window: query qi can attend to KV positions [0 .. kv_offset+qi]
    const int kv_offset = kv_len - q_len;

    // ── Shared memory ──────────────────────────────────────────────────────────
    // Layout must match size computation in the wrapper.
    extern __shared__ char smem_raw[];
    const int HD_pad = head_dim + 2;              // padded column stride (bank safety)
    __half* q_smem = (__half*)smem_raw;
    __half* K_smem = q_smem + head_dim;           // [Bc][HD_pad]
    __half* V_smem = K_smem + FLASH_BC * HD_pad;  // [Bc][HD_pad]
    float*  s_smem = (float*)(V_smem + FLASH_BC * HD_pad);  // [Bc]

    // ── Load query row into shared mem ─────────────────────────────────────────
    q_smem[d] = Q[((size_t)qi * num_q_heads + qh) * head_dim + d];

    // ── Per-thread online-softmax state ────────────────────────────────────────
    float O_d = 0.f;
    float m   = -1e30f;
    float l   = 0.f;

    const int num_tiles = (kv_len + FLASH_BC - 1) / FLASH_BC;

    for (int t = 0; t < num_tiles; ++t) {
        const int tile_start = t * FLASH_BC;

        // ── Load K_tile and V_tile ─────────────────────────────────────────────
        // Thread d loads column d across all Bc rows (coalesced in KV dim,
        // strided by num_kv_heads*HD in global memory).
        for (int j = 0; j < FLASH_BC; ++j) {
            const int kvi = tile_start + j;
            if (kvi < kv_len) {
                const size_t off =
                    ((size_t)kvi * num_kv_heads + kv_head) * head_dim + d;
                K_smem[j * HD_pad + d] = K[off];
                V_smem[j * HD_pad + d] = V[off];
            } else {
                K_smem[j * HD_pad + d] = __float2half(0.f);
                V_smem[j * HD_pad + d] = __float2half(0.f);
            }
        }
        __syncthreads();

        // ── Compute score for KV slot d in this tile ───────────────────────────
        // Thread d owns score s_smem[d] = Q[qi,qh,:] · K[tile_start+d,:] * scale
        {
            const int kvi     = tile_start + d;
            const bool masked = (kvi > kv_offset + qi) || (kvi >= kv_len);
            if (!masked) {
                float dot = 0.f;
                const __half* k_row = K_smem + (size_t)d * HD_pad;
                for (int d2 = 0; d2 < head_dim; ++d2)
                    dot += __half2float(q_smem[d2]) * __half2float(k_row[d2]);
                s_smem[d] = dot * scale;
            } else {
                s_smem[d] = -1e30f;
            }
        }
        __syncthreads();

        // ── Online softmax update ──────────────────────────────────────────────
        // Every thread scans the full s_smem[0..Bc) (reads 64 floats from smem).
        float tile_max = -1e30f;
        for (int j = 0; j < FLASH_BC; ++j)
            tile_max = fmaxf(tile_max, s_smem[j]);

        const float m_new = fmaxf(m, tile_max);

        // Guard: if everything so far (including this tile) is fully masked,
        // m_new == -1e30f.  Skipping avoids expf(0)=1 on all-masked scores.
        // In practice this only fires if qi=0 and kv_len=0, which never happens.
        if (m_new < -1e29f) {
            __syncthreads();
            continue;
        }

        const float alpha = expf(m - m_new);  // rescale factor; → 0 on first real tile

        // Thread d accumulates its output dimension and the shared normaliser.
        float l_inc = 0.f, O_inc = 0.f;
        for (int j = 0; j < FLASH_BC; ++j) {
            const float exp_s = expf(s_smem[j] - m_new);
            l_inc += exp_s;
            // V_smem[j][d] is at index j*HD_pad + d
            O_inc += exp_s * __half2float(V_smem[(size_t)j * HD_pad + d]);
        }

        O_d = alpha * O_d + O_inc;
        l   = alpha * l   + l_inc;
        m   = m_new;

        __syncthreads();  // protect s_smem / K_smem / V_smem before next tile
    }

    // ── Normalise and write output ─────────────────────────────────────────────
    const float val = (l > 1e-10f) ? O_d / l : 0.f;
    out[((size_t)qi * num_q_heads + qh) * head_dim + d] = __float2half(val);
}

// ─── Public wrapper (same signature as W5) ────────────────────────────────────
void attention_cuda(const Tensor& Q, const Tensor& K, const Tensor& V,
                    Tensor& out, int num_q_heads, int num_kv_heads, int kv_len)
{
    if (Q.device != Device::CUDA || K.device != Device::CUDA ||
        V.device != Device::CUDA || out.device != Device::CUDA) {
        fprintf(stderr, "attention_cuda: all tensors must be on CUDA\n");
        exit(1);
    }

    const int actual_kv_len = (kv_len < 0) ? (int)K.shape[0] : kv_len;
    const int q_len          = (int)Q.shape[0];
    const int head_dim       = (int)Q.shape[2];   // Q is [q_len, num_q_heads, HD]

    if (head_dim != FLASH_BC) {
        fprintf(stderr,
            "attention_cuda: flash_attn_kernel requires head_dim == %d, got %d\n",
            FLASH_BC, head_dim);
        exit(1);
    }

    const float scale  = 1.f / sqrtf((float)head_dim);
    const int   HD_pad = head_dim + 2;

    // Shared memory: q_smem + K_smem + V_smem + s_smem
    const size_t smem_bytes =
          (size_t)head_dim            * sizeof(__half)   // q_smem
        + (size_t)FLASH_BC * HD_pad   * sizeof(__half)   // K_smem
        + (size_t)FLASH_BC * HD_pad   * sizeof(__half)   // V_smem
        + (size_t)FLASH_BC            * sizeof(float);   // s_smem

    dim3 grid(q_len, num_q_heads);
    dim3 block(head_dim);   // == FLASH_BC == 64

    flash_attn_kernel<<<grid, block, smem_bytes>>>(
        Q.fp16(), K.fp16(), V.fp16(), out.fp16(),
        q_len, actual_kv_len, num_q_heads, num_kv_heads, head_dim, scale);
    CUDA_CHECK(cudaGetLastError());
}

// ── W5 three-kernel implementation (kept for reference / correctness tests) ──
// Uncomment the #if 1 to revert for comparison.
#if 0

__global__ void attn_scores_kernel(
    const __half* __restrict__ Q, const __half* __restrict__ K,
    float* __restrict__ scores,
    int q_len, int kv_len, int num_q_heads, int num_kv_heads,
    int head_dim, float scale)
{
    int qi = blockIdx.x, qh = blockIdx.y;
    int kv_head   = qh * num_kv_heads / num_q_heads;
    int kv_offset = kv_len - q_len;
    const __half* q_row = Q + ((size_t)qi * num_q_heads + qh) * head_dim;
    for (int kvi = threadIdx.x; kvi < kv_len; kvi += blockDim.x) {
        float score;
        if (kvi > kv_offset + qi) { score = -1e30f; }
        else {
            const __half* k_row = K + ((size_t)kvi * num_kv_heads + kv_head) * head_dim;
            float dot = 0.f;
            for (int d = 0; d < head_dim; d++)
                dot += __half2float(q_row[d]) * __half2float(k_row[d]);
            score = dot * scale;
        }
        scores[(size_t)qi * num_q_heads * kv_len + (size_t)qh * kv_len + kvi] = score;
    }
}

__global__ void softmax_f32_rows_kernel(float* __restrict__ scores, int kv_len) {
    extern __shared__ float smem[];
    int row = blockIdx.x;
    float* rp = scores + (size_t)row * kv_len;
    float tmax = -1e30f;
    for (int i = threadIdx.x; i < kv_len; i += blockDim.x) tmax = fmaxf(tmax, rp[i]);
    smem[threadIdx.x] = tmax; __syncthreads();
    for (int s = blockDim.x>>1; s>0; s>>=1) { if (threadIdx.x<s) smem[threadIdx.x]=fmaxf(smem[threadIdx.x],smem[threadIdx.x+s]); __syncthreads(); }
    float row_max = smem[0]; __syncthreads();
    float tsum = 0.f;
    for (int i = threadIdx.x; i < kv_len; i += blockDim.x) { float v=expf(rp[i]-row_max); rp[i]=v; tsum+=v; }
    smem[threadIdx.x] = tsum; __syncthreads();
    for (int s = blockDim.x>>1; s>0; s>>=1) { if (threadIdx.x<s) smem[threadIdx.x]+=smem[threadIdx.x+s]; __syncthreads(); }
    float inv = 1.f/smem[0]; __syncthreads();
    for (int i = threadIdx.x; i < kv_len; i += blockDim.x) rp[i] *= inv;
}

__global__ void attn_output_kernel(
    const float* __restrict__ scores, const __half* __restrict__ V,
    __half* __restrict__ out,
    int q_len, int kv_len, int num_q_heads, int num_kv_heads, int head_dim)
{
    int qi=blockIdx.x, qh=blockIdx.y, d=threadIdx.x;
    if (d >= head_dim) return;
    int kv_head = qh * num_kv_heads / num_q_heads;
    const float* sr = scores + ((size_t)qi*num_q_heads+qh)*kv_len;
    float acc = 0.f;
    for (int kvi=0; kvi<kv_len; kvi++)
        acc += sr[kvi] * __half2float(V[((size_t)kvi*num_kv_heads+kv_head)*head_dim+d]);
    out[((size_t)qi*num_q_heads+qh)*head_dim+d] = __float2half(acc);
}

#endif  // W5 three-kernel reference

// ═════════════════════════════════════════════════════════════════════════════
// 6.  Embedding lookup  — out[t, :] = embed_table[tokens[t], :]  (FP16)
//
//  embed_table: [vocab_size, hidden]  FP16
//  d_tokens:    [seq_len]             int32 on device
//  out:         [seq_len, hidden]     FP16
//
//  Grid = (seq_len,), Block = min(hidden, 256)
//  Each block copies one full embedding row.
// ─────────────────────────────────────────────────────────────────────────────

__global__ void embed_kernel(
    const __half* __restrict__ embed_table, // [vocab_size, hidden]
    const int*    __restrict__ d_tokens,    // [seq_len]
    __half*       __restrict__ out,         // [seq_len, hidden]
    int hidden)
{
    int t = blockIdx.x;
    int tok_id = d_tokens[t];
    const __half* src = embed_table + (size_t)tok_id * hidden;
    __half*       dst = out         + (size_t)t       * hidden;
    for (int i = threadIdx.x; i < hidden; i += blockDim.x)
        dst[i] = src[i];
}

void embed_cuda(const Tensor& embed_table, const int* d_tokens,
                Tensor& out, int seq_len)
{
    if (embed_table.device != Device::CUDA || out.device != Device::CUDA) {
        fprintf(stderr, "embed_cuda: tensors must be on CUDA\n");
        exit(1);
    }
    int hidden = (int)embed_table.shape[1];
    int block  = (hidden < 256) ? hidden : 256;
    embed_kernel<<<seq_len, block>>>(
        embed_table.fp16(), d_tokens, out.fp16(), hidden);
    CUDA_CHECK(cudaGetLastError());
}

// ═════════════════════════════════════════════════════════════════════════════
// 7.  Add in-place  — a[i] += b[i]  (FP16)
//
//  Both tensors must have the same total number of elements.
//  Grid-stride loop, Block = 256.
// ─────────────────────────────────────────────────────────────────────────────

__global__ void add_inplace_kernel(__half* __restrict__ a,
                                    const __half* __restrict__ b,
                                    int n)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
             i += gridDim.x * blockDim.x) {
        a[i] = __float2half(__half2float(a[i]) + __half2float(b[i]));
    }
}

void add_inplace_cuda(Tensor& a, const Tensor& b)
{
    if (a.device != Device::CUDA || b.device != Device::CUDA) {
        fprintf(stderr, "add_inplace_cuda: tensors must be on CUDA\n");
        exit(1);
    }
    int n     = (int)a.numel;
    int block = 256;
    int grid  = (n + block - 1) / block;
    // Cap grid to avoid launching huge numbers of blocks unnecessarily
    if (grid > 65535) grid = 65535;
    add_inplace_kernel<<<grid, block>>>(a.fp16(), b.fp16(), n);
    CUDA_CHECK(cudaGetLastError());
}

// ═════════════════════════════════════════════════════════════════════════════
// 8.  SiLU activation  — element-wise (FP16)
//
//  SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
// ─────────────────────────────────────────────────────────────────────────────

__global__ void silu_kernel(const __half* __restrict__ x,
                             __half* __restrict__ out,
                             int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = __half2float(x[i]);
        out[i] = __float2half(v / (1.0f + expf(-v)));
    }
}

void silu_cuda(Tensor& x)
{
    if (x.device != Device::CUDA) {
        fprintf(stderr, "silu_cuda: tensor must be on CUDA\n");
        exit(1);
    }
    int n = (int)x.numel;
    int block = 256;
    int grid  = (n + block - 1) / block;
    silu_kernel<<<grid, block>>>(x.fp16(), x.fp16(), n);
    CUDA_CHECK(cudaGetLastError());
}

// ═════════════════════════════════════════════════════════════════════════════
// 9.  Element-wise multiply  — for SwiGLU: out = a * b  (FP16)
// ─────────────────────────────────────────────────────────────────────────────

__global__ void mul_kernel(const __half* __restrict__ a,
                            const __half* __restrict__ b,
                            __half* __restrict__ c,
                            int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        c[i] = __float2half(__half2float(a[i]) * __half2float(b[i]));
}

void mul_cuda(const Tensor& a, const Tensor& b, Tensor& out)
{
    if (a.device != Device::CUDA || b.device != Device::CUDA ||
        out.device != Device::CUDA) {
        fprintf(stderr, "mul_cuda: all tensors must be on CUDA\n");
        exit(1);
    }
    int n = (int)a.numel;
    int block = 256;
    int grid  = (n + block - 1) / block;
    mul_kernel<<<grid, block>>>(a.fp16(), b.fp16(), out.fp16(), n);
    CUDA_CHECK(cudaGetLastError());
}

// ═════════════════════════════════════════════════════════════════════════════
// 10.  INT8 weight dequantisation  — W8A16 (W7 deliverable)
//
//  Per-row symmetric quantisation:
//    scale[row]   = max_abs(W[row, :]) / 127
//    w_int8[row]  = round(W[row, :] / scale[row])  clipped to [-127, 127]
//
//  Dequant kernel:  out[row, col] = (float)w_int8[row, col] * scale[row]
//
//  Grid-stride over all elements; each thread looks up its row's scale.
//  Block = 256.
// ─────────────────────────────────────────────────────────────────────────────

__global__ void dequant_int8_fp16_kernel(
    const int8_t* __restrict__ w_int8,  // [rows, cols]  INT8
    const float*  __restrict__ scale,   // [rows]         FP32 per-row scale
    __half*       __restrict__ out,     // [rows, cols]  FP16
    int rows, int cols)
{
    const int total = rows * cols;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
             idx < total;
             idx += gridDim.x * blockDim.x)
    {
        int row = idx / cols;
        out[idx] = __float2half((float)w_int8[idx] * scale[row]);
    }
}

void dequant_int8_to_fp16(const int8_t* d_int8, const float* d_scale,
                           __half* d_out, int rows, int cols)
{
    const int total = rows * cols;
    const int block = 256;
    const int grid  = min((total + block - 1) / block, 65535);
    dequant_int8_fp16_kernel<<<grid, block>>>(d_int8, d_scale, d_out, rows, cols);
    CUDA_CHECK(cudaGetLastError());
}

// Quantise FP16 GPU weight matrix to INT8+scale (CPU-side loop, called once at load time).
// src:   [rows, cols]  FP16 on GPU   — copied to CPU for quantisation then uploaded
// d_dst: [rows, cols]  INT8 on GPU   — caller must cudaMalloc
// d_scl: [rows]        FP32 on GPU   — caller must cudaMalloc
void quantize_fp16_to_int8_gpu(const __half* d_src, int8_t* d_dst, float* d_scl,
                                 int rows, int cols)
{
    // 1. Copy FP16 weights from GPU to CPU
    const size_t n = (size_t)rows * cols;
    std::vector<__half> h_src(n);
    CUDA_CHECK(cudaMemcpy(h_src.data(), d_src, n * sizeof(__half), cudaMemcpyDeviceToHost));

    // 2. Quantise row by row on CPU
    std::vector<int8_t> h_dst(n);
    std::vector<float>  h_scl(rows);

    for (int r = 0; r < rows; ++r) {
        const __half* row_ptr = h_src.data() + (size_t)r * cols;
        float max_abs = 0.f;
        for (int c = 0; c < cols; ++c)
            max_abs = fmaxf(max_abs, fabsf(__half2float(row_ptr[c])));

        h_scl[r] = (max_abs > 0.f) ? (max_abs / 127.f) : 1.f;
        const float inv_scl = (max_abs > 0.f) ? (127.f / max_abs) : 0.f;

        int8_t* dst_row = h_dst.data() + (size_t)r * cols;
        for (int c = 0; c < cols; ++c) {
            float v = __half2float(row_ptr[c]) * inv_scl;
            dst_row[c] = (int8_t)fmaxf(-127.f, fminf(127.f, roundf(v)));
        }
    }

    // 3. Upload back to GPU
    CUDA_CHECK(cudaMemcpy(d_dst, h_dst.data(), n  * sizeof(int8_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_scl, h_scl.data(), rows * sizeof(float),  cudaMemcpyHostToDevice));
}
