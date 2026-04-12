#!/usr/bin/env python3
"""
test_correctness.py — PyTorch correctness validation for RMSNorm and Softmax CUDA kernels.

Protocol
--------
1. Python generates random input (fixed seed) as raw binary.
2. Calls the compiled CUDA binary with --load_input / --dump_output flags.
3. Reads the binary output and compares against PyTorch reference.

Acceptance thresholds (from CLAUDE.md):
  FP32: max abs error < 1e-5
  FP16: max abs error < 1e-3

Usage
-----
  python test_correctness.py --kernel rmsnorm --dtype fp32
  python test_correctness.py --kernel softmax --dtype fp16
  python test_correctness.py                         # runs all four combinations
"""

import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np
import torch
import torch.nn.functional as F

# ── Thresholds ────────────────────────────────────────────────────────────────
ATOL = {"fp32": 1e-5, "fp16": 1e-3}

# ── Locate binaries ───────────────────────────────────────────────────────────

def find_binary(build_dir: str, name: str) -> str:
    candidates = [
        os.path.join(build_dir, name, name),
        os.path.join(build_dir, name),
    ]
    for path in candidates:
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    raise FileNotFoundError(
        f"Binary '{name}' not found in {build_dir}. "
        f"Run: cd cuda-kernels && cmake -B build && cmake --build build"
    )

# ── NumPy dtype helpers ───────────────────────────────────────────────────────

def np_dtype(dtype: str):
    return np.float32 if dtype == "fp32" else np.float16

def torch_dtype(dtype: str):
    return torch.float32 if dtype == "fp32" else torch.float16

# ── RMSNorm reference (PyTorch) ───────────────────────────────────────────────

def rmsnorm_ref(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """RMSNorm: y = x / RMS(x) * w,  RMS(x) = sqrt(mean(x^2) + eps)"""
    # Always compute in fp32 for a stable reference, then cast back
    xf = x.float()
    wf = weight.float()
    rms = torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (xf * rms * wf).to(x.dtype)

# ── Softmax reference (PyTorch) ───────────────────────────────────────────────

def softmax_ref(x: torch.Tensor) -> torch.Tensor:
    return F.softmax(x.float(), dim=-1).to(x.dtype)

# ── Run one test ──────────────────────────────────────────────────────────────

def run_test(kernel: str, dtype: str, rows: int, cols: int,
             build_dir: str, seed: int = 42) -> bool:
    rng   = np.random.default_rng(seed)
    npdtype = np_dtype(dtype)
    atol    = ATOL[dtype]

    with tempfile.TemporaryDirectory() as tmpdir:
        # ── Generate inputs ───────────────────────────────────────────────────
        x_np = rng.standard_normal((rows, cols)).astype(npdtype)

        in_path  = os.path.join(tmpdir, "input.bin")
        out_path = os.path.join(tmpdir, "output.bin")
        x_np.tofile(in_path)

        cmd = [
            find_binary(build_dir, kernel),
            "--rows",  str(rows),
            "--cols",  str(cols),
            "--dtype", dtype,
            "--load_input",  in_path,
            "--dump_output", out_path,
        ]

        if kernel == "rmsnorm":
            # Generate weight vector
            w_np = rng.standard_normal(cols).astype(npdtype) * 0.5 + 1.0
            wt_path = os.path.join(tmpdir, "weight.bin")
            w_np.tofile(wt_path)
            cmd += ["--load_weight", wt_path]

        # ── Run binary ────────────────────────────────────────────────────────
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        except FileNotFoundError as e:
            print(f"  [SKIP] {e}")
            return None

        if result.returncode != 0:
            print(f"  [ERROR] binary exited {result.returncode}")
            print("  stdout:", result.stdout.strip())
            print("  stderr:", result.stderr.strip())
            return False

        # ── Read output ───────────────────────────────────────────────────────
        if not os.path.isfile(out_path):
            print(f"  [ERROR] output file not created: {out_path}")
            return False

        cuda_out = np.fromfile(out_path, dtype=npdtype).reshape(rows, cols)

        # ── Compute reference ─────────────────────────────────────────────────
        x_t = torch.from_numpy(x_np)

        if kernel == "rmsnorm":
            w_t   = torch.from_numpy(w_np)
            ref_t = rmsnorm_ref(x_t, w_t)
        else:
            ref_t = softmax_ref(x_t)

        ref_np = ref_t.numpy().astype(npdtype)

        # ── Compare ───────────────────────────────────────────────────────────
        abs_err    = np.abs(cuda_out.astype(np.float32) - ref_np.astype(np.float32))
        max_err    = float(abs_err.max())
        mean_err   = float(abs_err.mean())
        passed     = max_err < atol

        status = "PASS" if passed else "FAIL"
        print(
            f"  [{status}] {kernel:<10} dtype={dtype}  "
            f"rows={rows:<5} cols={cols:<5}  "
            f"max_abs_err={max_err:.3e}  mean_abs_err={mean_err:.3e}  "
            f"(tol={atol:.0e})"
        )

        if not passed:
            # Show worst offenders
            flat = abs_err.reshape(-1)
            idx  = int(np.argmax(flat))
            r, c = divmod(idx, cols)
            print(f"         worst element: [{r},{c}]  cuda={cuda_out[r,c]:.6f}  ref={ref_np[r,c]:.6f}")

        return passed

# ── Test matrix ───────────────────────────────────────────────────────────────

TEST_CONFIGS = [
    # (rows, cols)
    (1,    128),
    (32,   512),
    (128,  2048),
    (256,  4096),
]

KERNELS = ["rmsnorm", "softmax"]
DTYPES  = ["fp32", "fp16"]

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CUDA kernel correctness validator")
    parser.add_argument("--kernel", choices=KERNELS,
                        help="Kernel to test (default: all)")
    parser.add_argument("--dtype",  choices=DTYPES,
                        help="Data type (default: all)")
    parser.add_argument("--rows",   type=int, default=None,
                        help="Override number of rows")
    parser.add_argument("--cols",   type=int, default=None,
                        help="Override number of columns")
    parser.add_argument("--build_dir", default=None,
                        help="Path to build directory (default: ../build relative to this script)")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    build_dir  = args.build_dir or os.path.join(script_dir, "..", "build")
    build_dir  = os.path.normpath(build_dir)

    kernels = [args.kernel] if args.kernel else KERNELS
    dtypes  = [args.dtype]  if args.dtype  else DTYPES
    configs = [(args.rows, args.cols)] if (args.rows and args.cols) else TEST_CONFIGS

    total  = 0
    passed = 0
    skipped = 0

    print(f"Build dir: {build_dir}\n")

    for kernel in kernels:
        for dtype in dtypes:
            print(f"=== {kernel.upper()} / {dtype.upper()} ===")
            for rows, cols in configs:
                res = run_test(kernel, dtype, rows, cols, build_dir)
                if res is None:
                    skipped += 1
                elif res:
                    passed += 1
                    total  += 1
                else:
                    total += 1
            print()

    real_total = total + skipped
    print("─" * 60)
    if skipped:
        print(f"Results: {passed}/{total} passed  ({skipped} skipped — binary not found)")
    else:
        print(f"Results: {passed}/{total} passed")

    if passed < total:
        sys.exit(1)


if __name__ == "__main__":
    main()
