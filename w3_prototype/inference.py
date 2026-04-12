"""
Manual prefill + decode inference loop for TinyLlama-1.1B.

No transformers.generate — hand-rolled KV cache loop.
Measures TTFT (Time To First Token) and decode tokens/s.
"""

import time
import torch
from transformers import AutoTokenizer

from model import TinyLlama, TinyLlamaConfig, load_tinyllama
from kv_cache import KVCache, ModelConfig
from sampler import sample_token


def build_kv_cache(cfg: TinyLlamaConfig, device: torch.device) -> KVCache:
    mc = ModelConfig(
        num_layers=cfg.num_layers,
        num_kv_heads=cfg.num_kv_heads,
        head_dim=cfg.head_dim,
        max_seq_len=cfg.max_seq_len,
    )
    return KVCache(mc, device=device, dtype=torch.float16)


@torch.no_grad()
def generate(
    model: TinyLlama,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 0.0,
    top_k: int = 1,
    top_p: float = 1.0,
    device: torch.device = None,
    verbose: bool = True,
) -> dict:
    """
    Full prefill + decode generation with manual KV cache.

    Returns:
        dict with keys: text, tokens, ttft_ms, decode_tps, total_ms
    """
    if device is None:
        device = next(model.parameters()).device

    cfg = model.cfg
    kv_cache = build_kv_cache(cfg, device)

    # ── Tokenize ──────────────────────────────────────────────────────────
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    prompt_len = input_ids.shape[1]

    if verbose:
        print(f"Prompt tokens: {prompt_len}")
        print(f"KV cache VRAM: {kv_cache.vram_mb():.1f} MB")

    generated_ids = []

    # ── Prefill ───────────────────────────────────────────────────────────
    prefill_start = time.perf_counter()

    logits = model(input_ids, start_pos=0, kv_cache=kv_cache)  # [1, prompt_len, vocab]
    kv_cache.advance(prompt_len)

    # Sample first token from last position
    first_token_logits = logits[0, -1, :]  # [vocab]
    next_token = sample_token(first_token_logits, temperature, top_k, top_p)
    generated_ids.append(next_token)

    prefill_end = time.perf_counter()
    ttft_ms = (prefill_end - prefill_start) * 1000

    if verbose:
        print(f"TTFT: {ttft_ms:.1f} ms")

    # ── Decode ────────────────────────────────────────────────────────────
    decode_start = time.perf_counter()

    for step in range(max_new_tokens - 1):
        # Feed single token
        cur_id = torch.tensor([[next_token]], dtype=torch.long, device=device)
        cur_pos = prompt_len + step  # position of this token in the full sequence

        logits = model(cur_id, start_pos=cur_pos, kv_cache=kv_cache)  # [1, 1, vocab]
        kv_cache.advance(1)

        next_token_logits = logits[0, -1, :]
        next_token = sample_token(next_token_logits, temperature, top_k, top_p)
        generated_ids.append(next_token)

        # Stop on EOS
        if next_token == tokenizer.eos_token_id:
            if verbose:
                print(f"EOS at step {step+1}")
            break

    decode_end = time.perf_counter()
    decode_elapsed = decode_end - decode_start
    num_decode_tokens = len(generated_ids) - 1  # excluding first token (TTFT)
    decode_tps = num_decode_tokens / decode_elapsed if decode_elapsed > 0 else 0

    total_ms = (decode_end - prefill_start) * 1000

    output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    if verbose:
        print(f"Decode: {num_decode_tokens} tokens in {decode_elapsed*1000:.1f} ms → {decode_tps:.2f} tok/s")
        print(f"Total: {total_ms:.1f} ms")
        print(f"\nGenerated:\n{output_text}")

    return {
        "text": output_text,
        "tokens": generated_ids,
        "prompt_tokens": prompt_len,
        "ttft_ms": ttft_ms,
        "decode_tps": decode_tps,
        "total_ms": total_ms,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--prompt", default="The quick brown fox jumps over the lazy dog. Explain the origin of this phrase:")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = load_tinyllama(args.model, device=str(device))

    result = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        device=device,
        verbose=True,
    )
