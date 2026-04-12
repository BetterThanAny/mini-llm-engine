"""
TinyLlama-1.1B model definition for manual inference.

Architecture: Llama-2 style with GQA, SwiGLU, RoPE
  - vocab_size=32000, hidden_size=2048, num_layers=22
  - num_heads=32, num_kv_heads=4, head_dim=64
  - ffn_dim=5632, rms_eps=1e-5, max_seq_len=2048

Weights loaded from HuggingFace: TinyLlama/TinyLlama-1.1B-Chat-v1.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional
from kv_cache import KVCache


# ─────────────────────────────────────────
# Config
# ─────────────────────────────────────────

class TinyLlamaConfig:
    vocab_size: int = 32000
    hidden_size: int = 2048
    num_layers: int = 22
    num_heads: int = 32
    num_kv_heads: int = 4
    head_dim: int = 64        # hidden_size // num_heads
    ffn_dim: int = 5632
    rms_eps: float = 1e-5
    max_seq_len: int = 2048
    rope_theta: float = 10000.0


# ─────────────────────────────────────────
# RoPE
# ─────────────────────────────────────────

def precompute_freqs_cis(head_dim: int, max_seq_len: int, theta: float = 10000.0) -> torch.Tensor:
    """
    Precompute complex exponentials for RoPE.
    Returns: [max_seq_len, head_dim//2] complex tensor
    """
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_seq_len)
    freqs = torch.outer(t, freqs)  # [seq, head_dim//2]
    return torch.polar(torch.ones_like(freqs), freqs)  # complex64


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """
    Apply RoPE to query or key tensor.

    Args:
        x: [batch, seq_len, num_heads, head_dim]
        freqs_cis: [seq_len, head_dim//2] complex

    Returns: same shape as x
    """
    x_f = x.float()
    # Reshape to complex: [batch, seq_len, num_heads, head_dim//2]
    x_c = torch.view_as_complex(x_f.reshape(*x_f.shape[:-1], -1, 2))
    freqs = freqs_cis.unsqueeze(0).unsqueeze(2)  # [1, seq, 1, head_dim//2]
    x_rot = torch.view_as_real(x_c * freqs).flatten(-2)
    return x_rot.to(x.dtype)


# ─────────────────────────────────────────
# Modules
# ─────────────────────────────────────────

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (norm * self.weight).to(x.dtype)


class GQAttention(nn.Module):
    """
    Grouped Query Attention.
    num_heads=32 query heads share num_kv_heads=4 KV heads (ratio 8:1).
    """

    def __init__(self, cfg: TinyLlamaConfig):
        super().__init__()
        self.num_heads = cfg.num_heads
        self.num_kv_heads = cfg.num_kv_heads
        self.head_dim = cfg.head_dim
        self.kv_repeat = cfg.num_heads // cfg.num_kv_heads  # 8

        self.q_proj = nn.Linear(cfg.hidden_size, cfg.num_heads * cfg.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, cfg.num_kv_heads * cfg.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, cfg.num_kv_heads * cfg.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.num_heads * cfg.head_dim, cfg.hidden_size, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        layer_idx: int,
        kv_cache: Optional[KVCache] = None,
    ) -> torch.Tensor:
        B, S, _ = x.shape

        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(B, S, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, S, self.num_kv_heads, self.head_dim)

        # Apply RoPE to Q and K
        q = apply_rotary_emb(q, freqs_cis)
        k = apply_rotary_emb(k, freqs_cis)

        # KV Cache update
        if kv_cache is not None:
            k, v = kv_cache.update(layer_idx, k, v)
            # k/v now: [B, total_seq, num_kv_heads, head_dim]

        # Expand KV to match num_heads (GQA → repeat interleave)
        k = k.repeat_interleave(self.kv_repeat, dim=2)  # [B, total_seq, num_heads, head_dim]
        v = v.repeat_interleave(self.kv_repeat, dim=2)

        # Scaled dot-product attention
        # q: [B, S, num_heads, head_dim] → [B, num_heads, S, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scale = math.sqrt(self.head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) / scale

        # Causal mask (only needed during prefill; decode has S=1)
        if S > 1:
            total_len = k.shape[-2]
            mask = torch.full((S, total_len), float('-inf'), device=x.device, dtype=x.dtype)
            # Each query position i can attend to positions 0..kv_cache.seq_len+i
            # Simpler: standard lower-triangular causal mask over total_len
            q_offset = total_len - S
            for i in range(S):
                mask[i, q_offset + i + 1:] = float('-inf')
            attn = attn + mask.unsqueeze(0).unsqueeze(0)

        attn = F.softmax(attn.float(), dim=-1).to(q.dtype)
        out = torch.matmul(attn, v)  # [B, num_heads, S, head_dim]
        out = out.transpose(1, 2).contiguous().view(B, S, -1)
        return self.o_proj(out)


class SwiGLUFFN(nn.Module):
    """
    Feed-forward with SwiGLU: out = SiLU(gate_proj(x)) * up_proj(x), then down_proj
    """

    def __init__(self, cfg: TinyLlamaConfig):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.ffn_dim, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.ffn_dim, bias=False)
        self.down_proj = nn.Linear(cfg.ffn_dim, cfg.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    def __init__(self, cfg: TinyLlamaConfig):
        super().__init__()
        self.attn = GQAttention(cfg)
        self.ffn = SwiGLUFFN(cfg)
        self.norm1 = RMSNorm(cfg.hidden_size, cfg.rms_eps)
        self.norm2 = RMSNorm(cfg.hidden_size, cfg.rms_eps)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        layer_idx: int,
        kv_cache: Optional[KVCache] = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), freqs_cis, layer_idx, kv_cache)
        x = x + self.ffn(self.norm2(x))
        return x


# ─────────────────────────────────────────
# Full Model
# ─────────────────────────────────────────

class TinyLlama(nn.Module):
    def __init__(self, cfg: TinyLlamaConfig = None):
        super().__init__()
        if cfg is None:
            cfg = TinyLlamaConfig()
        self.cfg = cfg

        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.num_layers)])
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

        # Precompute RoPE frequencies
        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        start_pos: int = 0,
        kv_cache: Optional[KVCache] = None,
    ) -> torch.Tensor:
        """
        Args:
            input_ids: [batch, seq_len]
            start_pos: position offset (for decode step when using KV cache)
            kv_cache: KVCache instance

        Returns:
            logits: [batch, seq_len, vocab_size]
        """
        B, S = input_ids.shape
        x = self.embed_tokens(input_ids)

        freqs = self.freqs_cis[start_pos: start_pos + S]

        for i, layer in enumerate(self.layers):
            x = layer(x, freqs, layer_idx=i, kv_cache=kv_cache)

        x = self.norm(x)
        logits = self.lm_head(x)
        return logits


# ─────────────────────────────────────────
# Weight Loading from HuggingFace
# ─────────────────────────────────────────

def load_tinyllama(model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0", device: str = "cuda") -> TinyLlama:
    """
    Load TinyLlama weights from HuggingFace into our custom model.

    HF parameter name → our module mapping:
      model.embed_tokens.weight          → embed_tokens.weight
      model.layers.N.self_attn.q_proj.weight → layers.N.attn.q_proj.weight
      model.layers.N.self_attn.k_proj.weight → layers.N.attn.k_proj.weight
      model.layers.N.self_attn.v_proj.weight → layers.N.attn.v_proj.weight
      model.layers.N.self_attn.o_proj.weight → layers.N.attn.o_proj.weight
      model.layers.N.mlp.gate_proj.weight    → layers.N.ffn.gate_proj.weight
      model.layers.N.mlp.up_proj.weight      → layers.N.ffn.up_proj.weight
      model.layers.N.mlp.down_proj.weight    → layers.N.ffn.down_proj.weight
      model.layers.N.input_layernorm.weight  → layers.N.norm1.weight
      model.layers.N.post_attention_layernorm.weight → layers.N.norm2.weight
      model.norm.weight                      → norm.weight
      lm_head.weight                         → lm_head.weight
    """
    from transformers import AutoModelForCausalLM
    import re

    print(f"Loading HuggingFace model: {model_name} ...")
    hf_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    hf_sd = hf_model.state_dict()

    cfg = TinyLlamaConfig()
    model = TinyLlama(cfg)
    our_sd = model.state_dict()

    def map_key(hf_key: str) -> Optional[str]:
        k = hf_key
        k = re.sub(r'^model\.', '', k)
        k = re.sub(r'\.self_attn\.q_proj', '.attn.q_proj', k)
        k = re.sub(r'\.self_attn\.k_proj', '.attn.k_proj', k)
        k = re.sub(r'\.self_attn\.v_proj', '.attn.v_proj', k)
        k = re.sub(r'\.self_attn\.o_proj', '.attn.o_proj', k)
        k = re.sub(r'\.mlp\.gate_proj', '.ffn.gate_proj', k)
        k = re.sub(r'\.mlp\.up_proj', '.ffn.up_proj', k)
        k = re.sub(r'\.mlp\.down_proj', '.ffn.down_proj', k)
        k = re.sub(r'\.input_layernorm', '.norm1', k)
        k = re.sub(r'\.post_attention_layernorm', '.norm2', k)
        return k if k in our_sd else None

    mapped, skipped = 0, []
    new_sd = {}
    for hf_key, tensor in hf_sd.items():
        our_key = map_key(hf_key)
        if our_key is not None:
            new_sd[our_key] = tensor
            mapped += 1
        else:
            skipped.append(hf_key)

    # freqs_cis is a buffer, not a weight — keep ours
    new_sd["freqs_cis"] = our_sd["freqs_cis"]

    missing = our_sd.keys() - new_sd.keys()
    model.load_state_dict(new_sd, strict=True)

    print(f"Loaded {mapped} tensors. Skipped HF keys: {skipped}")
    if missing:
        print(f"WARNING: missing keys: {missing}")

    model = model.to(device)
    model.eval()
    del hf_model
    torch.cuda.empty_cache()
    return model
