# Week 1: CUDA Kernel Fundamentals

## Deliverables

1. **vector_add** kernel with benchmark harness
2. **Parallel reduction** with 4 progressive optimisations:
   - v1: naive (interleaved addressing) — 105 GB/s
   - v2: sequential addressing — 153 GB/s (1.46x)
   - v3: first-add-during-load — 291 GB/s (2.77x)
   - v4: warp shuffle (`__shfl_down_sync`) — 418 GB/s (3.98x)
3. **GEMM** naive + shared-memory tiled versions

## Key Learnings

- Thread/block/grid hierarchy and occupancy considerations
- Shared memory bank conflicts and sequential vs interleaved addressing
- Warp-level primitives (`__shfl_down_sync`) eliminating shared memory for reduction
- Memory coalescing: sequential addressing achieves 1.46x over naive
- How first-add-during-load halves the number of idle threads

## Benchmark Results

See `benchmark.csv` (reduction section) for detailed timing on N=16M elements.

## Tag

`week1-done`
