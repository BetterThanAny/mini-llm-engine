#!/usr/bin/env bash
# llama.cpp FP16 baseline benchmark using llama-bench
# Protocol: TinyLlama-1.1B FP16 GGUF, prompt=128 tokens, generate=128 tokens
# Warmup is handled internally by llama-bench; runs=5.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

LLAMACPP_BENCH="$REPO_ROOT/llama.cpp/build/bin/llama-bench"
MODEL="$SCRIPT_DIR/../weights/tinyllama-f16.gguf"
GPU_LAYERS=999

echo "=== llama.cpp FP16 Baseline Benchmark ==="
echo "Model: $MODEL"
echo "Protocol: pp128 + tg128, ngl=$GPU_LAYERS, runs=5"
echo ""

"$LLAMACPP_BENCH" \
    -m "$MODEL" \
    -ngl "$GPU_LAYERS" \
    -p 128 \
    -n 128 \
    -r 5

echo ""
echo "=== VRAM (from llama-cli single run) ==="
"$REPO_ROOT/llama.cpp/build/bin/llama-cli" \
    -m "$MODEL" \
    -p "Hello" \
    -n 1 \
    -ngl "$GPU_LAYERS" \
    --temp 0 \
    --single-turn 2>&1 | grep 'memory_breakdown'
