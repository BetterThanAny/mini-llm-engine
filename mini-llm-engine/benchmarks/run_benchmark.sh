#!/usr/bin/env bash
# Benchmark protocol: TinyLlama-1.1B FP16, prompt=128 tokens, generate=128 tokens
# Warmup 3 runs (discarded), measure 5 runs, take median.
#
# Usage:
#   MODEL_PATH=/path/to/weights.bin \
#   TOKENIZER_PATH=/path/to/tokenizer.model \
#   [DEVICE=cuda]  \
#   bash benchmarks/run_benchmark.sh
#
# Defaults: DEVICE=cuda, OUTPUT_CSV=benchmark.csv next to this repo root.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/../build"
BINARY="$BUILD_DIR/llm_engine"

# Required paths (override via environment variables)
MODEL="${MODEL_PATH:-${SCRIPT_DIR}/../weights/tinyllama-fp32.bin}"
TOKENIZER="${TOKENIZER_PATH:-${SCRIPT_DIR}/../weights/tokenizer.model}"
DEVICE="${DEVICE:-cuda}"
OUTPUT_CSV="${OUTPUT_CSV:-${SCRIPT_DIR}/../benchmark.csv}"

if [[ ! -f "$BINARY" ]]; then
    echo "Error: $BINARY not found. Run 'cmake .. && make -j\$(nproc)' in build/ first." >&2
    exit 1
fi

if [[ ! -f "$MODEL" ]]; then
    echo "Error: model weights not found at '$MODEL'." >&2
    echo "  Set MODEL_PATH=/path/to/tinyllama-fp32.bin" >&2
    exit 1
fi

if [[ ! -f "$TOKENIZER" ]]; then
    echo "Error: tokenizer not found at '$TOKENIZER'." >&2
    echo "  Set TOKENIZER_PATH=/path/to/tokenizer.model" >&2
    exit 1
fi

echo "=== Mini-LLM-Engine Benchmark ==="
echo "Binary:     $BINARY"
echo "Model:      $MODEL"
echo "Tokenizer:  $TOKENIZER"
echo "Device:     $DEVICE"
if command -v nvidia-smi &>/dev/null; then
    echo "GPU:        $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
fi

# Fixed prompt of ~128 tokens (verified with TinyLlama tokenizer)
PROMPT="The quick brown fox jumps over the lazy dog. In the beginning, there was light, and the light was good. Scientists have long studied the mysteries of the universe, from the smallest subatomic particles to the vast cosmic structures that span billions of light-years across the observable universe."

echo ""
echo "Running: prompt=128 tokens, generate=128 tokens, temperature=0, top_k=1, seed=42"
echo "Warmup: 3 runs (discarded)  |  Measure: 5 runs (median reported)"
echo ""

"$BINARY" \
    --model      "$MODEL"      \
    --tokenizer  "$TOKENIZER"  \
    --prompt     "$PROMPT"     \
    --device     "$DEVICE"     \
    --max_new_tokens 128       \
    --temperature 0            \
    --top_k 1                  \
    --seed 42                  \
    --benchmark                \
    --warmup 3                 \
    --runs 5                   \
    --output_csv "$OUTPUT_CSV"

echo ""
echo "Results written to: $OUTPUT_CSV"
