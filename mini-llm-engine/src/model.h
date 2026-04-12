#pragma once
#include <string>
#include <vector>
#include <cstdint>
#include "tensor.h"
#include <cuda_runtime.h>

// Forward declaration to avoid circular include (kv_cache.h includes model.h)
struct KVCache;

// TinyLlama-1.1B architecture constants
// Source: https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/blob/main/config.json
struct LlamaConfig {
    int vocab_size   = 32000;
    int hidden_size  = 2048;
    int num_layers   = 22;
    int num_heads    = 32;
    int num_kv_heads = 4;     // GQA: 4 kv groups
    int head_dim     = 64;    // hidden_size / num_heads
    int ffn_dim      = 5632;  // intermediate_size
    float rms_eps    = 1e-5f;
    int max_seq_len  = 2048;
};

// Weights for one transformer layer (FP16 on GPU)
struct LayerWeights {
    Tensor* attn_q;   // [hidden, hidden]
    Tensor* attn_k;   // [kv_hidden, hidden]  kv_hidden = num_kv_heads * head_dim
    Tensor* attn_v;   // [kv_hidden, hidden]
    Tensor* attn_o;   // [hidden, hidden]
    Tensor* ffn_gate; // [ffn_dim, hidden]
    Tensor* ffn_up;   // [ffn_dim, hidden]
    Tensor* ffn_down; // [hidden, ffn_dim]
    Tensor* rms_attn; // [hidden]
    Tensor* rms_ffn;  // [hidden]
};

struct ModelWeights {
    Tensor* embed_tokens;  // [vocab_size, hidden_size]
    Tensor* lm_head;       // [vocab_size, hidden_size]
    Tensor* rms_final;     // [hidden_size]
    std::vector<LayerWeights> layers;
};

// Load MINILLM binary weights from file (implement in model.cpp)
ModelWeights load_weights(const std::string& path, const LlamaConfig& cfg);
void         free_weights(ModelWeights& w);

// Forward pass: processes tokens[0..seq_len-1] starting at pos_offset in sequence.
// Writes logits to logits_out[vocab_size] (only last token's logits, float32).
// Advances kv_cache.cur_len by seq_len on return.
void forward_cpu(const ModelWeights& w, const LlamaConfig& cfg,
                 KVCache& kv_cache, const int* tokens, int seq_len,
                 int pos_offset, float* logits_out);

// Upload CPU FP32 weights to GPU as FP16. Returns new ModelWeights (GPU).
// Caller must call free_weights() on the returned object when done.
ModelWeights weights_to_gpu(const ModelWeights& cpu_w);

// GPU forward pass: same interface as forward_cpu but uses GPU ops.
// Requires: w tensors on GPU (FP16), kv_cache on GPU (FP16).
void forward_gpu(const ModelWeights& w, const LlamaConfig& cfg,
                 KVCache& kv_cache, const int* tokens, int seq_len,
                 int pos_offset, float* logits_out);

// ── Batched GPU forward pass (W7) ─────────────────────────────────────────────
//
// Processes batch_size independent sequences simultaneously.
// Each sequence has the same seq_len (padded if needed by the caller).
//
// tokens:      [batch_size * seq_len]  token IDs, row-major (host)
// kv_caches:   one KVCache per batch item (each pre-allocated for max_seq_len)
// logits_out:  [batch_size * vocab_size]  FP32 output (host), last-token logits
//
// pos_offset: all sequences are at the same position (same prompt length).
//
// VRAM estimate per batch item: same as single-item (~43MB KV + activations).
// Total for batch=8: ~344MB KV cache (still fits in RTX 3080 10GB).
void forward_gpu_batched(const ModelWeights& w, const LlamaConfig& cfg,
                         std::vector<KVCache>& kv_caches,
                         const int* tokens,   // [bs * seq_len]
                         int seq_len, int batch_size,
                         int pos_offset, float* logits_out); // [bs * vocab_size]

// ── INT8 W8A16 weight-only quantisation (W7) ─────────────────────────────────
//
// All weight matrices (Q/K/V/O projections, FFN gate/up/down) are stored as
// INT8 + per-row FP32 scale.  Norm weights and embedding table stay FP16
// (1-D or used for lookup, not GEMM).
//
// VRAM: ~550 MB (vs ~1.1 GB for FP16) for TinyLlama-1.1B.

struct Int8Tensor {
    int8_t* d_data  = nullptr;   // [rows, cols] INT8 on GPU
    float*  d_scale = nullptr;   // [rows]        FP32 per-row scale on GPU
    int     rows    = 0;
    int     cols    = 0;

    void alloc(int r, int c) {
        rows = r; cols = c;
        cudaMalloc(&d_data,  (size_t)r * c * sizeof(int8_t));
        cudaMalloc(&d_scale, (size_t)r     * sizeof(float));
    }
    void free_mem() {
        if (d_data)  { cudaFree(d_data);  d_data  = nullptr; }
        if (d_scale) { cudaFree(d_scale); d_scale = nullptr; }
    }
};

struct Int8LayerWeights {
    Int8Tensor attn_q, attn_k, attn_v, attn_o;
    Int8Tensor ffn_gate, ffn_up, ffn_down;
    Tensor*    rms_attn = nullptr;   // FP16 — kept as-is
    Tensor*    rms_ffn  = nullptr;   // FP16 — kept as-is
};

struct Int8ModelWeights {
    Tensor* embed_tokens = nullptr;   // FP16 — embedding lookup, not GEMM
    Tensor* lm_head      = nullptr;   // FP16 — could quantize; kept FP16 for accuracy
    Tensor* rms_final    = nullptr;   // FP16
    std::vector<Int8LayerWeights> layers;
};

// Convert GPU FP16 weights → INT8 (quantises all GEMM weight matrices in place).
// Input must be the result of weights_to_gpu().
Int8ModelWeights weights_to_int8(const ModelWeights& gpu_fp16_w);
void             free_int8_weights(Int8ModelWeights& w);

// GPU forward pass using INT8 weight matrices (W8A16).
// Same external interface as forward_gpu; activations stay FP16.
void forward_gpu_int8(const Int8ModelWeights& w, const LlamaConfig& cfg,
                      KVCache& kv_cache, const int* tokens, int seq_len,
                      int pos_offset, float* logits_out);
