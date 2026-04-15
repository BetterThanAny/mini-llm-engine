# Week 2: LLM-Relevant CUDA Kernels

## Deliverables

1. **RMSNorm** kernel — v1 naive + v2 warp shuffle + v3 register-cached x
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

## Performance (FP32, RTX 3080 Laptop sm_86)

| rows × hidden | v1_naive (GB/s) | v2_warp (GB/s) | v3_regcache (GB/s) | v3/v1 |
| --- | --- | --- | --- | --- |
| 256 × 512    | 118.2 | 102.4 | 128.0 | 1.08× |
| 256 × 2048   | 384.0 | 438.9 | 438.9 | 1.14× |
| 1024 × 2048  | 390.1 | 276.1 | 423.7 | 1.09× |
| 1024 × 4096  | 402.9 | 387.0 | **565.0** | **1.40×** |

Plan target: optimized ≥ 1.3× naive on at least one configuration. Hit at
hidden=4096 (LLM-relevant size for TinyLlama hidden=2048, Llama-7B hidden=4096).

### Analysis

RMSNorm is purely bandwidth-bound. v1 already achieves ~400 GB/s on large
configs — close to the RTX 3080 Laptop peak of ~384 GB/s, because the second
read of `x` hits L2. v2 (warp shuffle + float4) does not outperform v1 because:

1. Both kernels max out memory bandwidth with room only to reduce total
   traffic, not accelerate compute-bound work.
2. v2's smaller block (128 vs 256) reduces inflight memory requests per SM,
   hurting latency hiding at large hidden sizes.

**v3 solution**: cache x in registers between the two passes (sum → normalize),
eliminating the second x read. Effective traffic drops from 4N to 3N bytes,
yielding the theoretical ceiling ~1.33×. At hidden=4096 measured 1.40×, since
v3 also uses block=256 restoring occupancy.

At small hidden (≤ 2048), the register-cache win is dampened because L2
already served the second x read at near-free cost, and kernel launch
overhead grows as a fraction of total runtime.

## Tag

`week2-done`
