"""
KV Cache for TinyLlama-1.1B inference.

Pre-allocates fixed-size buffers for keys and values across all layers.
TinyLlama uses GQA: num_kv_heads=4, so KV tensors are smaller than Q.
"""

import torch
from dataclasses import dataclass


@dataclass
class ModelConfig:
    num_layers: int = 22
    num_kv_heads: int = 4
    head_dim: int = 64
    max_seq_len: int = 2048


class KVCache:
    """
    Pre-allocated KV cache for all transformer layers.

    Layout: [num_layers, max_seq_len, num_kv_heads, head_dim]

    VRAM cost for TinyLlama FP16:
        2 (K+V) * 22 layers * 2048 seq * 4 kv_heads * 64 head_dim * 2 bytes
        = 2 * 22 * 2048 * 4 * 64 * 2 = 91,750,400 bytes ≈ 87.5 MB
    """

    def __init__(self, config: ModelConfig, device: torch.device, dtype: torch.dtype = torch.float16):
        self.config = config
        self.device = device
        self.dtype = dtype

        shape = (config.num_layers, config.max_seq_len, config.num_kv_heads, config.head_dim)
        self.k_cache = torch.zeros(shape, dtype=dtype, device=device)
        self.v_cache = torch.zeros(shape, dtype=dtype, device=device)

        # Current sequence length (how many tokens have been cached)
        self.seq_len = 0

    def update(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Store new K/V and return the full cached K/V for this layer.

        Args:
            layer_idx: transformer layer index
            k: new keys  [batch=1, seq_len, num_kv_heads, head_dim]
            v: new values [batch=1, seq_len, num_kv_heads, head_dim]

        Returns:
            k_full, v_full: [batch=1, total_seq_len, num_kv_heads, head_dim]
        """
        # k/v shape: [1, new_tokens, num_kv_heads, head_dim]
        new_tokens = k.shape[1]
        start = self.seq_len
        end = start + new_tokens

        if end > self.config.max_seq_len:
            raise ValueError(f"KV cache overflow: {end} > {self.config.max_seq_len}")

        self.k_cache[layer_idx, start:end] = k[0]
        self.v_cache[layer_idx, start:end] = v[0]

        k_full = self.k_cache[layer_idx, :end].unsqueeze(0)  # [1, end, num_kv_heads, head_dim]
        v_full = self.v_cache[layer_idx, :end].unsqueeze(0)
        return k_full, v_full

    def advance(self, n: int = 1):
        """Advance the sequence length after processing n new tokens."""
        self.seq_len += n

    def reset(self):
        """Clear the cache for a new sequence."""
        self.k_cache.zero_()
        self.v_cache.zero_()
        self.seq_len = 0

    @property
    def current_len(self) -> int:
        return self.seq_len

    def vram_mb(self) -> float:
        total_bytes = (self.k_cache.numel() + self.v_cache.numel()) * self.k_cache.element_size()
        return total_bytes / (1024 ** 2)
