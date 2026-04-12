# Week 2: LLM-Relevant CUDA Kernels

## Deliverables

1. **RMSNorm** kernel — v1 naive + v2 warp shuffle
2. **Softmax** kernel — numerically stable (online max tracking)
3. **GEMM v3** register tiling: BM=BN=128, BK=8, TM=TN=8
4. **Matrix transpose** — naive + shared-memory tiled
5. **PyTorch correctness test harness** — automated comparison against PyTorch reference

## Key Learnings

- RMSNorm: reduction across hidden dimension, warp shuffle for inter-thread communication
- Softmax numerical stability: subtract max before exp to prevent overflow
- Register tiling for GEMM: each thread computes an 8x8 output tile, reducing shared memory traffic
- Correctness testing against PyTorch with configurable tolerance (FP32 < 1e-5, FP16 < 1e-3)

## Correctness

All kernels pass against PyTorch reference:
- RMSNorm: max error < 1e-5 (FP32)
- Softmax: max error < 1e-5 (FP32)
- GEMM v3: max error < 1e-3 (FP16)

## Tag

`week2-done`
