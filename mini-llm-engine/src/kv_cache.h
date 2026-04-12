#pragma once
#include "tensor.h"
#include "model.h"

// Per-layer KV cache: stores key and value tensors for all past tokens.
// CPU mode: FP32.  GPU mode: FP16.
// Layout: [max_seq_len, num_kv_heads, head_dim]
struct KVCache {
    int num_layers;
    int max_seq_len;
    int num_kv_heads;
    int head_dim;
    int cur_len = 0;  // tokens filled so far
    Device device;    // CPU or CUDA

    // k_cache[l] and v_cache[l]: shape [max_seq_len, num_kv_heads, head_dim]
    std::vector<Tensor> k_cache;
    std::vector<Tensor> v_cache;

    KVCache(const LlamaConfig& cfg, Device device = Device::CPU);
    void reset() { cur_len = 0; }
};
