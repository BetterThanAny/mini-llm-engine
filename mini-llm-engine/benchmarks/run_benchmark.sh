#!/usr/bin/env bash
# Benchmark protocol: TinyLlama-1.1B FP16, prompt=128 tokens, generate=128 tokens
# Warmup 3 runs (discarded), measure 5 runs, take median.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/../build"
BINARY="$BUILD_DIR/llm_engine"
MODEL="${MODEL_PATH:-model.gguf}"
OUTPUT_CSV="$SCRIPT_DIR/../benchmark.csv"

if [[ ! -f "$BINARY" ]]; then
    echo "Error: $BINARY not found. Run cmake && make first." >&2
    exit 1
fi

echo "Running benchmark: prompt=128 tokens, generate=128 tokens, seed=42"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "CUDA: $(nvcc --version | grep release | awk '{print $5}' | tr -d ',')"

# Fixed prompt of ~128 tokens
PROMPT="The quick brown fox jumps over the lazy dog. In the beginning, there was light, and the light was good. Scientists have long studied the mysteries of the universe, from the smallest subatomic particles to the vast cosmic structures that span billions of light-years across the observable universe."

"$BINARY" \
    --model "$MODEL" \
    --prompt "$PROMPT" \
    --max_new_tokens 128 \
    --temperature 0 \
    --top_k 1 \
    --benchmark \
    --warmup 3 \
    --runs 5 \
    --output_csv "$OUTPUT_CSV"

echo "Results written to $OUTPUT_CSV"
