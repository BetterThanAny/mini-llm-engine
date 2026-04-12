#pragma once
#include <cstddef>

// CPU operator implementations for TinyLlama-1.1B inference (FP32).
// All tensors are row-major, contiguous.

// RMSNorm: out[i] = x[i] / rms(x_row) * w[i]
// x, out: [rows, hidden]    w: [hidden]
void rms_norm_cpu(const float* x, const float* w, float* out, int rows, int hidden, float eps);

// RoPE: apply rotary position embedding in-place.
// xq: [num_tokens, num_q_heads, head_dim]
// xk: [num_tokens, num_kv_heads, head_dim]
// pos_offset: position index of first token in xq/xk
void rope_cpu(float* xq, float* xk, int num_tokens,
              int num_q_heads, int num_kv_heads, int head_dim,
              int pos_offset, float rope_theta = 10000.0f);

// Matrix multiplication: C[M,N] = A[M,K] @ B[N,K]^T
// A is row-major [M, K], B is row-major [N, K] (transposed in the product),
// C is row-major [M, N].
// This matches the convention for weight matrices stored as [out_features, in_features].
void matmul_cpu(const float* A, const float* B, float* C, int M, int N, int K);

// Causal grouped-query attention (GQA).
// q:   [q_len, num_q_heads, head_dim]
// k:   [kv_len, num_kv_heads, head_dim]   — full cached K
// v:   [kv_len, num_kv_heads, head_dim]   — full cached V
// out: [q_len, num_q_heads, head_dim]
// num_q_heads must be divisible by num_kv_heads (ratio = kv_repeat = 8 for TinyLlama)
// Applies causal masking: q[i] can only attend to k[0..kv_offset+i]
// kv_offset = kv_len - q_len  (number of cached tokens before current query)
void attention_cpu(const float* q, const float* k, const float* v, float* out,
                   int q_len, int kv_len,
                   int num_q_heads, int num_kv_heads, int head_dim);

// SiLU activation: out[i] = gate[i] * sigmoid(gate[i]) * up[i]
// (SwiGLU: fused silu(gate) * up)
// gate and up both [n], out [n] (can be same pointer as gate)
void silu_mul_cpu(const float* gate, const float* up, float* out, int n);

// Softmax in-place along a single row of length n
void softmax_inplace(float* x, int n);
