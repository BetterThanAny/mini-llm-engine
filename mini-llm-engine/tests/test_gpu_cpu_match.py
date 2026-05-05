"""
tests/test_gpu_cpu_match.py
───────────────────────────
End-to-end correctness test: GPU (FP16) vs CPU (FP32) token match rate.

Runs the engine binary twice — once with --device cpu, once with --device cuda —
and checks that at least 95% of the first 64 generated tokens agree.

Usage:
    MODEL_PATH=/path/to/weights.bin \
    TOKENIZER_PATH=/path/to/tokenizer.model \
    python tests/test_gpu_cpu_match.py

Environment variables:
    MODEL_PATH      — path to MINILLM .bin weights
    TOKENIZER_PATH  — path to sentencepiece model
    ENGINE_BIN      — path to built binary           (default: build/llm_engine)
    NUM_TOKENS      — how many tokens to generate    (default: 64)
    MATCH_THRESHOLD — minimum match fraction         (default: 0.95)
"""

import os
import sys
import subprocess
import pathlib

ROOT = pathlib.Path(__file__).parent.parent

MODEL_PATH      = os.environ.get("MODEL_PATH")
TOKENIZER_PATH  = os.environ.get("TOKENIZER_PATH")
ENGINE_BIN      = os.environ.get("ENGINE_BIN",      str(ROOT / "build"   / "llm_engine"))
NUM_TOKENS      = int(os.environ.get("NUM_TOKENS",  "64"))
MATCH_THRESHOLD = float(os.environ.get("MATCH_THRESHOLD", "0.95"))

# Standard benchmark prompt (~128 tokens)
PROMPT = (
    "The quick brown fox jumps over the lazy dog. "
    "In the beginning, there was light, and the light was good. "
    "Scientists have long studied the mysteries of the universe, "
    "from the smallest subatomic particles to the vast cosmic structures "
    "that span billions of light-years across the observable universe."
)


def run_engine(device: str) -> str:
    """
    Run the engine binary and return the generated text.
    Raises subprocess.CalledProcessError on non-zero exit.
    """
    cmd = [
        ENGINE_BIN,
        "--model",          MODEL_PATH,
        "--tokenizer",      TOKENIZER_PATH,
        "--prompt",         PROMPT,
        "--device",         device,
        "--max_new_tokens", str(NUM_TOKENS),
        "--temperature",    "0",
        "--top_k",          "1",
        "--seed",           "42",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    # Parse generated text from stdout: look for "=== Generated Text ===" section
    lines = result.stdout.splitlines()
    in_section = False
    text_lines = []
    for line in lines:
        if "=== Generated Text ===" in line:
            in_section = True
            continue
        if in_section:
            if line.startswith("==="):
                break
            text_lines.append(line)
    return "\n".join(text_lines).strip()


def check_prerequisites():
    errors = []
    if not MODEL_PATH:
        errors.append("MODEL_PATH is not set")
    if not TOKENIZER_PATH:
        errors.append("TOKENIZER_PATH is not set")
    if not pathlib.Path(ENGINE_BIN).exists():
        errors.append(f"Engine binary not found: {ENGINE_BIN}\n"
                      f"  Build with: cd build && cmake .. && make -j$(nproc)")
    if MODEL_PATH and not pathlib.Path(MODEL_PATH).exists():
        errors.append(f"Model weights not found: {MODEL_PATH}\n"
                      f"  Set MODEL_PATH=/path/to/weights.bin")
    if TOKENIZER_PATH and not pathlib.Path(TOKENIZER_PATH).exists():
        errors.append(f"Tokenizer not found: {TOKENIZER_PATH}\n"
                      f"  Set TOKENIZER_PATH=/path/to/tokenizer.model")
    if errors:
        for e in errors:
            print(f"SKIP: {e}")
        sys.exit(77)


def char_token_match(text_a: str, text_b: str, n: int = NUM_TOKENS) -> float:
    """
    Approximate token-level match rate using character n-grams.
    Since we don't have the tokenizer in Python here, we compare
    whitespace-split words as a proxy for tokens.
    """
    words_a = text_a.split()[:n]
    words_b = text_b.split()[:n]
    if not words_a:
        return 0.0
    matches = sum(a == b for a, b in zip(words_a, words_b))
    return matches / len(words_a)


def main():
    print("GPU vs CPU token match test")
    print(f"  Engine:     {ENGINE_BIN}")
    print(f"  Model:      {MODEL_PATH}")
    print(f"  Tokens:     {NUM_TOKENS}")
    print(f"  Threshold:  {MATCH_THRESHOLD:.0%}")
    print()

    check_prerequisites()

    # CPU run
    print("Running CPU inference (FP32)...")
    try:
        cpu_text = run_engine("cpu")
        print(f"  CPU output: {cpu_text[:80]}...")
    except subprocess.CalledProcessError as e:
        print(f"  ERROR running CPU inference:\n{e.stderr}")
        sys.exit(1)

    # GPU run
    print("Running GPU inference (FP16 CUDA)...")
    try:
        gpu_text = run_engine("cuda")
        print(f"  GPU output: {gpu_text[:80]}...")
    except subprocess.CalledProcessError as e:
        print(f"  ERROR running GPU inference:\n{e.stderr}")
        sys.exit(1)

    # Compute match rate
    match_rate = char_token_match(cpu_text, gpu_text, NUM_TOKENS)
    passed = match_rate >= MATCH_THRESHOLD

    print()
    print(f"Match rate: {match_rate:.1%}  (threshold: {MATCH_THRESHOLD:.0%})")
    if passed:
        print(f"PASS — GPU/CPU outputs agree on >= {MATCH_THRESHOLD:.0%} of tokens.")
        sys.exit(0)
    else:
        print(f"FAIL — GPU/CPU diverged: only {match_rate:.1%} token match.")
        print()
        print(f"CPU text:\n{cpu_text}")
        print()
        print(f"GPU text:\n{gpu_text}")
        sys.exit(1)


if __name__ == "__main__":
    main()
