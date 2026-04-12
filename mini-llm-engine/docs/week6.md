# Week 6 — Flash Attention v1

## What changed

Replaced the W5 three-kernel naive attention (`attn_scores_kernel` → `softmax_f32_rows_kernel` → `attn_output_kernel`) with a single **Flash Attention v1** kernel (`flash_attn_kernel`) in `src/ops_cuda.cu`.

The public interface `attention_cuda(...)` is unchanged — no edits to `model.cpp` or `ops_cuda.h`.

---

## Algorithm derivation

### Problem with naive attention

For a sequence of length N and H heads, the naive implementation allocates a score buffer:

```
scores: [q_len, num_q_heads, kv_len]  FP32
```

At prefill with N=128: `128 × 32 × 128 × 4B = 2 MB` — fine.
At generation with max_seq_len=2048: `1 × 32 × 2048 × 4B = 256 KB` per forward pass — also fine *for our current config*.

However, the per-layer `cudaMalloc` / `cudaFree` is expensive (synchronizes the GPU), and the approach doesn't scale to longer contexts where the score buffer becomes gigabytes.

### Online softmax (Milakov & Gimelshein 2018)

Standard softmax requires two passes: one to find the max, one to compute exp/sum.  
The **online** variant maintains a running max `m` and accumulates a rescaled sum, allowing the computation to be fused into a single pass:

```
m₀ = -∞,  l₀ = 0,  O₀ = 0

For each element sⱼ:
    m_new = max(m, sⱼ)
    α     = exp(m − m_new)        ← rescales old state to new max
    l     = α · l + exp(sⱼ − m_new)
    O     = α · O + exp(sⱼ − m_new) · vⱼ

Output = O / l
```

The key property: `O` and `l` can be accumulated over *tiles* of (K, V) without ever writing the full scores array.

### Flash Attention v1 tiling

Tile K and V into blocks of `Bc` rows.  For each tile `t`:

```
Load K_tile[Bc, HD] and V_tile[Bc, HD] into shared memory
Compute s_tile[j] = Q · K_tile[j] / sqrt(HD)  for j ∈ [0, Bc)
Apply causal mask: s[j] = −∞ if kv_pos > kv_offset + qi

tile_max = max(s_tile[0..Bc−1])
m_new    = max(m, tile_max)
α        = exp(m − m_new)

l  ← α · l + Σⱼ exp(sⱼ − m_new)
Oᵈ ← α · Oᵈ + Σⱼ exp(sⱼ − m_new) · V_tile[j][d]

m ← m_new
```

Correctness proof: the recurrence is the same as online softmax, applied tile-by-tile.  
The final normalisation `output[d] = O_d / l` is identical to standard softmax weighted sum.

### Why it's exact

FA v1 is **not** approximate — it computes the same values as standard attention, just in a different order.  
The reordering is valid because:
1. Softmax denominator `l` accumulates the same sum regardless of order (after rescaling by α).
2. The weighted sum `O` uses exactly the same exp weights as standard softmax after the final division.

Max absolute error vs W5 naive: should be < 1e-2 in FP16 (accumulation in FP32 registers).

---

## Implementation details (`src/ops_cuda.cu`, line 374)

### Thread/block mapping

| Dimension | Value | Rationale |
|---|---|---|
| `FLASH_BC` (tile size) | 64 | = `head_dim` for TinyLlama; lets thread `d` both compute score for KV slot `d` and own output dim `d` |
| Grid | `(q_len, num_q_heads)` | One block per (query position, query head) |
| Block | 64 threads | = `head_dim` = `FLASH_BC` |

### Shared memory layout (~16.9 KB per block)

```
q_smem  [64]      __half    128 B    — query row loaded once
K_smem  [64][66]  __half   8448 B    — current K tile (+2 col pad)
V_smem  [64][66]  __half   8448 B    — current V tile
s_smem  [64]      float     256 B    — QK scores for current tile
─────────────────────────────────────
Total                      17280 B  ≈ 16.9 KB  (fits in 48 KB L1 on SM86)
```

The `+2` column padding (`HD_pad = HD + 2 = 66`) avoids 2-way shared memory bank conflicts when all 64 threads simultaneously read the same row of K_smem or V_smem.

### Score computation (per tile)

Thread `d` computes `s_smem[d]` = `Q[qi,qh,:] · K_smem[d,:]` × scale:

```
dot = Σ_{d2=0..63}  q_smem[d2] * K_smem[d * HD_pad + d2]
```

This is 64 multiply-adds per thread per tile — sequential inner loop, but all 64 threads run in parallel computing different KV positions.

### Causal mask

```
kv_offset = kv_len − q_len
masked    = (tile_start + d) > (kv_offset + qi)  ||  (tile_start + d) >= kv_len
s_smem[d] = masked ? −1e30f : dot * scale
```

Works for both prefill (q_len = N, kv_offset = 0) and decode (q_len = 1, kv_offset = kv_len − 1).

### Numerical guard

When all positions so far are masked, `m_new = −1e30f`. Without a guard, `expf(−1e30f − (−1e30f)) = expf(0) = 1.0` would corrupt the accumulator.  A `continue` when `m_new < −1e29f` prevents this.

---

## Roofline analysis: W5 vs W6

### W5 (three-kernel naive)

| Step | VRAM traffic | Notes |
|---|---|---|
| Kernel 1 (Q@K^T) | Read Q+K, write scores | scores = `q_len × NH × kv_len × 4B` |
| Kernel 2 (softmax) | Read+write scores | 2× the scores buffer |
| Kernel 3 (scores@V) | Read scores+V, write out | 3× scores + V |
| **cudaMalloc/Free** | sync overhead | every layer, every token |

For decode (q_len=1, kv_len=2048): scores = `1 × 32 × 2048 × 4 = 256 KB`.  
Three kernel launches + one sync per layer × 22 layers.

### W6 (Flash Attention v1)

| Step | VRAM traffic | Notes |
|---|---|---|
| Load K+V tiles | `kv_len × KVH × HD × 2 × 2B` (K+V) | per head, per layer |
| Load Q once | `q_len × NH × HD × 2B` | loaded into smem once |
| Write output | `q_len × NH × HD × 2B` | one pass |
| **No score buffer** | 0 B | online softmax in registers |

For decode: K+V reads = `2048 × 4 × 64 × 2 × 2B = 2 MB` — same as naive.  
But: **one kernel launch**, **no cudaMalloc**, **no intermediate writes** to VRAM.

### Memory savings

| Config | W5 score buffer | W6 score buffer |
|---|---|---|
| Decode, kv_len=2048 | 256 KB | 0 |
| Prefill, seq=128 | 2 MB | 0 |
| Prefill, seq=2048 | 512 MB | 0 |

At seq_len=2048 prefill, W5 would OOM (512 MB just for scores × 22 layers in flight); W6 handles it with ~17 KB shared mem per block.

---

## Benchmark (to be filled on GPU machine)

Protocol: TinyLlama-1.1B FP16, prompt=128 tokens, generate=128 tokens,  
temperature=0, top_k=1, seed=42. Warmup 3 runs, measure 5 runs, median.

| Metric | W5 (naive) | W6 (flash) | HF baseline |
|---|---|---|---|
| TTFT (ms) | — | — | — |
| Decode (tok/s) | — | — | — |
| Peak VRAM (MB) | — | — | — |
| P95 decode latency (ms) | — | — | — |

*Fill in after running `benchmarks/run_benchmark.sh` on the RTX 3080.*

---

## Correctness test (planned)

Compare `attention_cuda` (W6) vs Python reference attention on random FP16 inputs:

```python
# tests/test_flash_attn.py
import torch, subprocess

def ref_attention(Q, K, V):
    scale = Q.shape[-1] ** -0.5
    scores = (Q @ K.transpose(-1,-2)) * scale
    # causal mask
    mask = torch.triu(torch.ones(Q.shape[-2], K.shape[-2]), diagonal=1).bool()
    scores.masked_fill_(mask, float('-inf'))
    return torch.softmax(scores, dim=-1) @ V

# Build C++ test, run, compare max abs error
# Target: max abs error < 1e-2 (FP16 accumulation budget)
```

---

## Git tag

```bash
git tag week6-done
```
