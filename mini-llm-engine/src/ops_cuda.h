#pragma once
#include "tensor.h"

// All ops expect FP16 tensors on GPU unless noted.

// RMSNorm: y = x / rms(x) * weight   (hand-written kernel, W2 deliverable)
void rms_norm_cuda(const Tensor& x, const Tensor& weight, Tensor& out, float eps);

// Softmax in-place along last dim
void softmax_cuda(Tensor& x);

// RoPE: apply rotary position embedding in-place
// q, k: [seq_len, num_heads, head_dim]
void rope_cuda(Tensor& q, Tensor& k, int position_offset);

// cuBLAS GEMM wrapper: C = alpha * A @ B^T + beta * C
// A: [M, K], B: [N, K], C: [M, N]  (all FP16)
void gemm_fp16(const Tensor& A, const Tensor& B, Tensor& C,
               float alpha = 1.0f, float beta = 0.0f);

// Causal GQA attention: scores = Q @ K^T / sqrt(d), causal mask, softmax, out = scores @ V
// Q:   [q_len, num_q_heads,  head_dim]  FP16 CUDA
// K,V: [kv_len, num_kv_heads, head_dim] FP16 CUDA  (kv_len may be max_seq_len in cache)
// out: [q_len, num_q_heads,  head_dim]  FP16 CUDA
// kv_len: actual filled length (if -1, uses K.shape[0])
void attention_cuda(const Tensor& Q, const Tensor& K, const Tensor& V,
                    Tensor& out, int num_q_heads, int num_kv_heads,
                    int kv_len = -1);

// SiLU activation: x = x * sigmoid(x)
void silu_cuda(Tensor& x);

// Element-wise multiply (for SwiGLU: gate * up)
void mul_cuda(const Tensor& a, const Tensor& b, Tensor& out);

// Embedding lookup: out[t, :] = embed_table[d_tokens[t], :]
// embed_table: [vocab, hidden] FP16 CUDA
// d_tokens: int* on GPU, length = seq_len
// out: [seq_len, hidden] FP16 CUDA
void embed_cuda(const Tensor& embed_table, const int* d_tokens,
                Tensor& out, int seq_len);

// Element-wise in-place add: a += b  (both FP16 CUDA, same numel)
void add_inplace_cuda(Tensor& a, const Tensor& b);

// ── INT8 W8A16 weight quantisation (W7) ──────────────────────────────────────

// Dequantise INT8 weight matrix to FP16 in-place.
// d_int8: [rows, cols] INT8 on GPU
// d_scale:[rows]       FP32 per-row scale on GPU
// d_out:  [rows, cols] FP16 on GPU  (pre-allocated by caller)
void dequant_int8_to_fp16(const int8_t* d_int8, const float* d_scale,
                           __half* d_out, int rows, int cols);

// Quantise FP16 GPU tensor to INT8+scale, uploading result back to GPU.
// d_dst and d_scl must be pre-allocated (cudaMalloc) by the caller.
void quantize_fp16_to_int8_gpu(const __half* d_src, int8_t* d_dst, float* d_scl,
                                 int rows, int cols);

// Fused INT8 matrix-vector multiply: y = W_int8 @ x  (W8A16)
// W_int8: [N, K] INT8 on GPU, scale: [N] FP32 per-row
// x: [K] FP16 on GPU, y: [N] FP16 on GPU
// Reads INT8 weights directly without dequantising to FP16 buffer.
void gemv_int8_fp16(const int8_t* d_weight, const float* d_scale,
                    const __half* d_x, __half* d_y, int N, int K);
