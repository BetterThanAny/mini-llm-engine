#pragma once
#include <string>
#include <vector>
#include "tensor.h"

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

// Load GGUF weights from file (stub — implement in model.cpp)
ModelWeights load_weights(const std::string& path, const LlamaConfig& cfg);
void         free_weights(ModelWeights& w);
