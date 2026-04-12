"""
HuggingFace baseline benchmark for TinyLlama-1.1B.

This is the W5 comparison baseline. Measures TTFT and decode throughput
using the stock transformers pipeline (with built-in KV cache).

Protocol:
  - prompt = fixed 128 tokens (padded/truncated)
  - generate = 128 new tokens
  - temperature=0, top_k=1, seed=42
  - warmup 3 runs (discarded), measure 5 runs, take median
  - results appended to benchmark.csv
"""

import csv
import json
import os
import statistics
import time
from datetime import datetime
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ─────────────────────────────────────────
# Config
# ─────────────────────────────────────────

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
TARGET_PROMPT_TOKENS = 128
MAX_NEW_TOKENS = 128
WARMUP_RUNS = 3
MEASURE_RUNS = 5
SEED = 42

BENCHMARK_CSV = Path(__file__).parent.parent / "benchmark.csv"

# A fixed seed prompt that will be truncated/padded to TARGET_PROMPT_TOKENS
BASE_PROMPT = (
    "Below is a detailed technical explanation of how large language models work. "
    "These models are trained on vast amounts of text data using the transformer architecture, "
    "which relies on attention mechanisms to capture long-range dependencies. "
    "The attention mechanism computes query, key, and value projections for each token, "
    "allowing the model to focus on relevant parts of the input sequence. "
    "During inference, the model generates tokens one at a time in an autoregressive fashion. "
    "This is known as the decode phase, and it is typically memory-bandwidth bound. "
    "The prefill phase processes the entire prompt in parallel and is compute-bound. "
    "Key-value caching is used to avoid recomputing attention for previously seen tokens."
)


# ─────────────────────────────────────────
# Environment info
# ─────────────────────────────────────────

def get_env_info() -> dict:
    info = {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else "N/A",
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        "gpu_vram_gb": (
            round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)
            if torch.cuda.is_available() else "N/A"
        ),
    }
    return info


# ─────────────────────────────────────────
# Benchmark
# ─────────────────────────────────────────

def build_prompt_ids(tokenizer, target_len: int) -> torch.Tensor:
    """Build a prompt tensor of exactly target_len tokens."""
    ids = tokenizer.encode(BASE_PROMPT)
    if len(ids) >= target_len:
        ids = ids[:target_len]
    else:
        # Repeat to fill
        while len(ids) < target_len:
            ids = ids + ids
        ids = ids[:target_len]
    return torch.tensor([ids])


@torch.no_grad()
def run_one(model, input_ids: torch.Tensor, device: torch.device) -> dict:
    """
    Run a single benchmark trial.
    Returns ttft_ms and total_tokens_per_sec.
    """
    torch.manual_seed(SEED)
    input_ids = input_ids.to(device)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    t0 = time.perf_counter()

    # Prefill: forward pass over prompt
    outputs = model(input_ids, use_cache=True)
    past_kv = outputs.past_key_values
    next_token = outputs.logits[0, -1, :].argmax().unsqueeze(0).unsqueeze(0)  # [1, 1]

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    t_first = time.perf_counter()
    ttft_ms = (t_first - t0) * 1000

    # Decode: generate remaining tokens one at a time
    generated = [next_token.item()]
    for _ in range(MAX_NEW_TOKENS - 1):
        out = model(next_token, past_key_values=past_kv, use_cache=True)
        past_kv = out.past_key_values
        next_token = out.logits[0, -1, :].argmax().unsqueeze(0).unsqueeze(0)
        generated.append(next_token.item())

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    t_end = time.perf_counter()
    decode_elapsed = t_end - t_first
    decode_tps = (MAX_NEW_TOKENS - 1) / decode_elapsed

    return {
        "ttft_ms": ttft_ms,
        "decode_tps": decode_tps,
        "total_ms": (t_end - t0) * 1000,
    }


def main():
    env = get_env_info()
    print("Environment:")
    for k, v in env.items():
        print(f"  {k}: {v}")
    print()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model {MODEL_NAME} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16)
    model = model.to(device)
    model.eval()
    print("Model loaded.\n")

    input_ids = build_prompt_ids(tokenizer, TARGET_PROMPT_TOKENS)
    actual_len = input_ids.shape[1]
    print(f"Prompt tokens: {actual_len} (target: {TARGET_PROMPT_TOKENS})")
    print(f"Max new tokens: {MAX_NEW_TOKENS}")
    print()

    # Warmup
    print(f"Warming up ({WARMUP_RUNS} runs) ...")
    for i in range(WARMUP_RUNS):
        run_one(model, input_ids, device)
        print(f"  warmup {i+1}/{WARMUP_RUNS} done")

    # Measurement
    print(f"\nMeasuring ({MEASURE_RUNS} runs) ...")
    results = []
    for i in range(MEASURE_RUNS):
        r = run_one(model, input_ids, device)
        results.append(r)
        print(f"  run {i+1}: TTFT={r['ttft_ms']:.1f}ms  decode={r['decode_tps']:.1f}tok/s")

    # Medians
    ttft_median = statistics.median(r["ttft_ms"] for r in results)
    tps_median = statistics.median(r["decode_tps"] for r in results)
    total_median = statistics.median(r["total_ms"] for r in results)

    print(f"\nMedian TTFT:     {ttft_median:.1f} ms")
    print(f"Median decode:   {tps_median:.1f} tok/s")
    print(f"Median total:    {total_median:.1f} ms")

    # VRAM
    if torch.cuda.is_available():
        vram_mb = torch.cuda.max_memory_allocated() / 1024**2
        print(f"Peak VRAM:       {vram_mb:.0f} MB")
    else:
        vram_mb = 0

    # Write CSV
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "engine": "hf_baseline",
        "model": MODEL_NAME,
        "prompt_tokens": actual_len,
        "gen_tokens": MAX_NEW_TOKENS,
        "temperature": 0,
        "top_k": 1,
        "seed": SEED,
        "ttft_ms": round(ttft_median, 2),
        "decode_tps": round(tps_median, 2),
        "total_ms": round(total_median, 2),
        "peak_vram_mb": round(vram_mb, 0),
        "gpu_name": env["gpu_name"],
        "cuda_version": env["cuda_version"],
        "torch_version": env["torch_version"],
        "warmup_runs": WARMUP_RUNS,
        "measure_runs": MEASURE_RUNS,
    }

    BENCHMARK_CSV.parent.mkdir(parents=True, exist_ok=True)
    write_header = not BENCHMARK_CSV.exists()
    with open(BENCHMARK_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    print(f"\nResults written to: {BENCHMARK_CSV}")


if __name__ == "__main__":
    main()
