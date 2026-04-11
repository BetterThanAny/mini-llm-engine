#pragma once
#include "tensor.h"
#include "model.h"

// Per-layer KV cache: stores key and value tensors for all past tokens.
// Layout: [max_seq_len, num_kv_heads, head_dim] in FP16 on GPU.
struct KVCache {
    int num_layers;
    int max_seq_len;
    int num_kv_heads;
    int head_dim;
    int cur_len = 0;  // tokens filled so far

    // k_cache[l] and v_cache[l]: shape [max_seq_len, num_kv_heads, head_dim]
    std::vector<Tensor> k_cache;
    std::vector<Tensor> v_cache;

    KVCache(const LlamaConfig& cfg);
    void reset() { cur_len = 0; }
};
