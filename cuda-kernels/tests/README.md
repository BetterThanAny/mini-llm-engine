# Correctness Tests

```bash
# 1. Build all kernels (from cuda-kernels/)
cmake -B build && cmake --build build -j

# 2. Install dependencies
pip install -r tests/requirements.txt

# 3. Run all tests (rmsnorm + softmax, fp32 + fp16)
python tests/test_correctness.py

# 4. Run a specific kernel / dtype
python tests/test_correctness.py --kernel rmsnorm --dtype fp16
```

Acceptance: FP32 max_abs_err < 1e-5, FP16 max_abs_err < 1e-3 (per CLAUDE.md).
