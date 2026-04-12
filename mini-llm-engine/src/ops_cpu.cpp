#include "ops_cpu.h"

#include <cmath>
#include <algorithm>
#include <cstring>
#include <vector>

// ---------------------------------------------------------------------------
// softmax_inplace
// ---------------------------------------------------------------------------
void softmax_inplace(float* x, int n) {
    // Find max for numerical stability
    float max_val = x[0];
    for (int i = 1; i < n; ++i) {
        if (x[i] > max_val) max_val = x[i];
    }

    // Exp and accumulate
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        x[i] = std::exp(x[i] - max_val);
        sum += x[i];
    }

    // Normalize
    float inv_sum = 1.0f / sum;
    for (int i = 0; i < n; ++i) {
        x[i] *= inv_sum;
    }
}

// ---------------------------------------------------------------------------
// rms_norm_cpu
// ---------------------------------------------------------------------------
void rms_norm_cpu(const float* x, const float* w, float* out, int rows, int hidden, float eps) {
    for (int r = 0; r < rows; ++r) {
        const float* x_row = x + r * hidden;
        float*       o_row = out + r * hidden;

        // Compute mean of squares
        float sum_sq = 0.0f;
        for (int i = 0; i < hidden; ++i) {
            sum_sq += x_row[i] * x_row[i];
        }
        float rms = std::sqrt(sum_sq / static_cast<float>(hidden) + eps);
        float inv_rms = 1.0f / rms;

        // Scale by weight
        for (int i = 0; i < hidden; ++i) {
            o_row[i] = x_row[i] * inv_rms * w[i];
        }
    }
}

// ---------------------------------------------------------------------------
// rope_cpu
// ---------------------------------------------------------------------------
void rope_cpu(float* xq, float* xk, int num_tokens,
              int num_q_heads, int num_kv_heads, int head_dim,
              int pos_offset, float rope_theta) {
    // head_dim must be even for RoPE
    int half_dim = head_dim / 2;

    // Apply to query heads — half-half RoPE (matches HuggingFace LLaMA)
    // Pairs dim j with dim j + half_dim (NOT interleaved 2j, 2j+1)
    for (int tok = 0; tok < num_tokens; ++tok) {
        int pos = pos_offset + tok;
        for (int h = 0; h < num_q_heads; ++h) {
            float* head_ptr = xq + tok * num_q_heads * head_dim + h * head_dim;
            for (int j = 0; j < half_dim; ++j) {
                float freq  = 1.0f / std::pow(rope_theta, (2.0f * j) / static_cast<float>(head_dim));
                float angle = static_cast<float>(pos) * freq;
                float cos_a = std::cos(angle);
                float sin_a = std::sin(angle);

                float x0 = head_ptr[j];
                float x1 = head_ptr[j + half_dim];

                head_ptr[j]            = x0 * cos_a - x1 * sin_a;
                head_ptr[j + half_dim] = x0 * sin_a + x1 * cos_a;
            }
        }
    }

    // Apply to key heads
    for (int tok = 0; tok < num_tokens; ++tok) {
        int pos = pos_offset + tok;
        for (int h = 0; h < num_kv_heads; ++h) {
            float* head_ptr = xk + tok * num_kv_heads * head_dim + h * head_dim;
            for (int j = 0; j < half_dim; ++j) {
                float freq  = 1.0f / std::pow(rope_theta, (2.0f * j) / static_cast<float>(head_dim));
                float angle = static_cast<float>(pos) * freq;
                float cos_a = std::cos(angle);
                float sin_a = std::sin(angle);

                float x0 = head_ptr[j];
                float x1 = head_ptr[j + half_dim];

                head_ptr[j]            = x0 * cos_a - x1 * sin_a;
                head_ptr[j + half_dim] = x0 * sin_a + x1 * cos_a;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// matmul_cpu  —  C[M,N] = A[M,K] @ B[N,K]^T
// ---------------------------------------------------------------------------
void matmul_cpu(const float* A, const float* B, float* C, int M, int N, int K) {
    // Zero output
    std::memset(C, 0, sizeof(float) * M * N);

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k) {
                // A[i,k] * B[j,k]  (B stored row-major [N,K])
                acc += A[i * K + k] * B[j * K + k];
            }
            C[i * N + j] = acc;
        }
    }
}

// ---------------------------------------------------------------------------
// attention_cpu  —  causal GQA
// ---------------------------------------------------------------------------
void attention_cpu(const float* q, const float* k, const float* v, float* out,
                   int q_len, int kv_len,
                   int num_q_heads, int num_kv_heads, int head_dim) {
    // kv_repeat: how many q-heads share one kv-head
    int kv_repeat = num_q_heads / num_kv_heads;

    // kv_offset: number of cached (prefix) tokens before the current query block
    int kv_offset = kv_len - q_len;

    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    // Temporary score buffer (heap to avoid stack overflow for long sequences)
    std::vector<float> scores(kv_len);

    for (int qi = 0; qi < q_len; ++qi) {
        // The absolute position of this query token in the sequence
        int q_abs_pos = kv_offset + qi;

        for (int qh = 0; qh < num_q_heads; ++qh) {
            // GQA: map query head -> kv head
            int kvh = qh / kv_repeat;

            // Pointers into q and out for this (token, head)
            const float* q_ptr  = q   + qi  * num_q_heads * head_dim + qh  * head_dim;
            float*       o_ptr  = out + qi  * num_q_heads * head_dim + qh  * head_dim;

            // Compute dot-product scores with all kv positions [0, kv_len)
            for (int t = 0; t < kv_len; ++t) {
                const float* k_ptr = k + t * num_kv_heads * head_dim + kvh * head_dim;

                float dot = 0.0f;
                for (int d = 0; d < head_dim; ++d) {
                    dot += q_ptr[d] * k_ptr[d];
                }
                scores[t] = dot * scale;
            }

            // Causal mask: q[qi] can only attend to kv positions <= kv_offset + qi
            // Set future positions to -inf so they become 0 after softmax
            for (int t = q_abs_pos + 1; t < kv_len; ++t) {
                scores[t] = -1e9f;
            }

            // Softmax over [0, kv_len)
            softmax_inplace(scores.data(), kv_len);

            // Weighted sum of value vectors -> output
            std::memset(o_ptr, 0, sizeof(float) * head_dim);
            for (int t = 0; t < kv_len; ++t) {
                const float* v_ptr = v + t * num_kv_heads * head_dim + kvh * head_dim;
                float w = scores[t];
                for (int d = 0; d < head_dim; ++d) {
                    o_ptr[d] += w * v_ptr[d];
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// silu_mul_cpu  —  SwiGLU: out[i] = silu(gate[i]) * up[i]
// ---------------------------------------------------------------------------
void silu_mul_cpu(const float* gate, const float* up, float* out, int n) {
    for (int i = 0; i < n; ++i) {
        float g = gate[i];
        // SiLU(g) = g * sigmoid(g) = g / (1 + exp(-g))
        float silu_g = g / (1.0f + std::exp(-g));
        out[i] = silu_g * up[i];
    }
}
