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

// Causal attention: scores = Q @ K^T / sqrt(d), mask, softmax, out = scores @ V
// Q: [seq_len, num_heads, head_dim]
// K, V: [cache_len, num_kv_heads, head_dim]
void attention_cuda(const Tensor& Q, const Tensor& K, const Tensor& V,
                    Tensor& out, int num_heads, int num_kv_heads);

// SiLU activation: x = x * sigmoid(x)
void silu_cuda(Tensor& x);

// Element-wise multiply (for SwiGLU: gate * up)
void mul_cuda(const Tensor& a, const Tensor& b, Tensor& out);
