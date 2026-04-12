#include "kv_cache.h"
#include <cstdio>

KVCache::KVCache(const LlamaConfig& cfg, Device dev)
    : num_layers(cfg.num_layers), max_seq_len(cfg.max_seq_len),
      num_kv_heads(cfg.num_kv_heads), head_dim(cfg.head_dim),
      device(dev)
{
    int dims[3] = {cfg.max_seq_len, cfg.num_kv_heads, cfg.head_dim};
    DType dtype = (dev == Device::CUDA) ? DType::FP16 : DType::FP32;

    k_cache.reserve(cfg.num_layers);
    v_cache.reserve(cfg.num_layers);
    for (int i = 0; i < cfg.num_layers; i++) {
        k_cache.emplace_back(dims, 3, dtype, dev);
        v_cache.emplace_back(dims, 3, dtype, dev);
    }

    size_t elem_bytes = (dev == Device::CUDA) ? 2 : 4;
    size_t total_bytes = 2ULL * cfg.num_layers * cfg.max_seq_len
                         * cfg.num_kv_heads * cfg.head_dim * elem_bytes;
    printf("[KVCache] Allocated: %zu MB  (device=%s, dtype=%s, layers=%d, seq=%d, kv_heads=%d, head_dim=%d)\n",
           total_bytes / (1024 * 1024),
           (dev == Device::CUDA) ? "CUDA" : "CPU",
           (dev == Device::CUDA) ? "FP16" : "FP32",
           cfg.num_layers, cfg.max_seq_len, cfg.num_kv_heads, cfg.head_dim);
}
