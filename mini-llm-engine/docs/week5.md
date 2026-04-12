# Week 5: GPU End-to-End Inference (FP16)

**Tag**: `week5-done`
**Hardware**: RTX 3080 (10GB VRAM, 29.8 TFLOPS FP16, 760 GB/s memory bandwidth)
**Model**: TinyLlama-1.1B, FP16, single GPU

---

## Overview

W5 moves the full forward pass from CPU (FP32) to GPU (FP16). The W4 deliverable was a working CPU inference pipeline across all 22 TinyLlama layers with GQA attention and KV cache. W5 adds:

- A complete GPU forward path (`forward_gpu`) mirroring `forward_cpu`
- Three hand-written CUDA kernels for attention (`attention_cuda`)
- Helper kernels: `embed_cuda`, `add_inplace_cuda`
- A weight conversion pipeline: CPU FP32 → GPU FP16 (`weights_to_gpu`)
- GPU-resident KV cache (FP16)
- `--device cpu|cuda` CLI flag

The CPU path is preserved unchanged. GPU path uses FP16 for all intermediate activations and weights; only the final logit projection output is copied back to CPU as FP32 for sampling.

---

## New Kernels

### attention_cuda

The most architecturally significant addition. Implements multi-head attention with grouped-query attention (GQA) over a causal prefix (prefill) or single token (decode), using three separate kernels.

**TinyLlama GQA configuration**:
- `num_q_heads = 32`, `num_kv_heads = 4`, `head_dim = 64`
- `kv_repeat = num_q_heads / num_kv_heads = 8`
- KV heads are shared: `kv_head = q_head * num_kv_heads / num_q_heads = q_head / 8`

#### Kernel 1: Q @ K^T + causal mask

```
Grid  = (q_len, num_q_heads)
Block = 128 threads
Output: scores[q_len][num_q_heads][kv_len]  (FP32)
```

Each thread block handles one `(qi, q_head)` pair and computes the dot products against all `kv_len` keys. The KV head is derived as `kv_head = q_head / kv_repeat`.

Causal masking: for query position `qi`, keys at positions `kvi > kvi_max` are masked to `-inf`:

```
kvi_max = (kv_len - q_len) + qi
```

This correctly handles both prefill (`q_len == kv_len`, lower-triangular mask) and decode (`q_len == 1`, all KV positions visible).

Scores are scaled by `1 / sqrt(head_dim) = 1 / 8.0` before masking.

Scores buffer is allocated in FP32 to avoid precision collapse during softmax; size is `q_len * num_q_heads * kv_len * 4 bytes`. Peak usage during prefill with `seq_len=2048`: `2048 * 32 * 2048 * 4 = 512MB`. In practice, W5 caps `max_seq_len=512` for development; at that size the buffer is `512 * 32 * 512 * 4 ≈ 32MB`.

#### Kernel 2: Softmax (per attention row)

```
Grid  = (q_len, num_q_heads)
Block = 256 threads (shared memory reduction)
```

Each block softmaxes one row of length `kv_len` in FP32. Uses shared memory for the two-pass reduction (max then sum). FP32 throughout to avoid NaN from `-inf` masked positions.

#### Kernel 3: Weighted V sum → output

```
Grid  = (q_len, num_q_heads)
Block = head_dim = 64 threads  (one thread per output dimension)
```

Each thread computes `sum_over_kv(scores[qi][qh][kvi] * V[kvi][kv_head][dim])`. KV head reuse: `kv_head = q_head / kv_repeat`. Output written as FP16 to the attention output buffer.

### Helper Kernels

#### embed_cuda

Embedding lookup on GPU. Reads token IDs (int32), writes FP16 embedding vectors.

```
Grid  = seq_len
Block = min(hidden_dim, 256) = 256
```

Each block copies one embedding row from the weight table to the output buffer. No arithmetic; pure memory bandwidth.

#### add_inplace_cuda

Element-wise FP16 addition for residual connections: `x += residual`.

```
Grid  = ceil(n / 256)
Block = 256
```

Used after attention and after FFN in every transformer layer (44 calls per full forward pass over 22 layers).

---

## GPU Memory Layout

**Weight memory** (FP16 on device after `weights_to_gpu`):

| Component | Tensors | Size |
|-----------|---------|------|
| Token embedding table | 1 | `32000 * 2048 * 2B = 128MB` |
| Per-layer (22 layers × 9 tensors) | 198 | ~900MB total |
| Final norm + LM head | 2 | ~128MB |
| **Total** | **201** | **~1.1GB** |

Source on CPU (FP32): ~2.2GB. Freed after conversion.

**KV cache** (FP16, GPU-resident):

```
2 (K/V) × 22 layers × 2048 max_seq_len × 4 kv_heads × 64 head_dim × 2 bytes
= 2 × 22 × 2048 × 4 × 64 × 2 = 45,088,768 bytes ≈ 43MB
```

**Activation buffers** (FP16, reused across layers): `O(hidden_dim)` per token, negligible.

**Attention scores buffer** (FP32, allocated once, reused): `q_len * num_q_heads * kv_len * 4B`
- Decode: `1 * 32 * kv_len * 4B` (tiny)
- Prefill (seq=512): `32MB`

**Total peak VRAM** (estimated): ~1.2GB weights + ~43MB KV cache + ~32MB scores + activations ≈ **~1.3GB**

---

## Performance Analysis

### Roofline Model

**Decode step** (single token, 1 Q against `kv_len` KV pairs):

For a single linear layer with weight matrix `W` of shape `[out, in]` (FP16):
- FLOPs: `2 * in * out`
- Bytes loaded (weights): `in * out * 2B` (FP16)
- Arithmetic intensity: `(2 * in * out) / (in * out * 2) = 1 FLOP/byte`

RTX 3080 peak FP16: 29.8 TFLOPS. Memory bandwidth: 760 GB/s.
Ridge point: `29800 / 760 ≈ 39 FLOP/byte`.

Decode arithmetic intensity (1 FLOP/byte) << ridge point (39 FLOP/byte). **Decode is firmly memory-bandwidth bound.**

Theoretical decode throughput ceiling:
- Model weights loaded per token: ~1.1GB (all 201 tensors, assuming no caching)
- At 760 GB/s: `1.1GB / 760 GB/s ≈ 1.45ms per token → ~690 tokens/s` (upper bound, bandwidth-only)
- Realistic efficiency ~15-25%: **100-170 tokens/s expected**

**Prefill** (full prompt, `seq_len` queries, Q@K^T is compute-bound):
- Attention FLOPs: `O(seq_len^2 * num_q_heads * head_dim)`
- At seq=128: `128^2 * 32 * 64 * 2 = 67M FLOPs` — still dominated by linear layers

### Benchmark Targets

_To be filled after GPU run. Protocol: warmup 3, measure 5, take median._

| Config | TTFT (ms) | Decode (tok/s) | Peak VRAM (MB) | P95 decode latency (ms) |
|--------|-----------|---------------|----------------|------------------------|
| HF baseline FP16 (W3) | TBD | TBD | TBD | TBD |
| CPU FP32 (W4) | TBD | TBD | N/A | TBD |
| GPU FP16 (W5) | TBD | TBD | TBD | TBD |
| llama.cpp FP16 | TBD | TBD | TBD | TBD |

Target: GPU decode > 100 tok/s for 128-token generation at prompt length 128.

Run via: `benchmarks/run_benchmark.sh`

---

## Architecture Decisions

### Separate scores buffer vs. flash attention

W5 uses a naive `O(seq_len^2)` scores buffer in global memory for correctness simplicity. Flash attention (Dao et al., 2022) fuses the three attention kernels into one tiled kernel, eliminating the scores buffer and reducing HBM traffic from `O(seq_len^2)` to `O(seq_len)` per head — critical for long sequences and prefill throughput.

Decision: defer to W6. Correctness against CPU baseline must be established first. The three-kernel split makes each stage independently debuggable (intermediate `scores` can be dumped and compared against PyTorch `einsum`). Once end-to-end token match rate ≥ 95% is confirmed, replace with flash attention v2.

### GQA head mapping

GQA reduces KV cache size by `kv_repeat = 8×` versus MHA. The mapping is:

```
kv_head_idx = q_head_idx * num_kv_heads / num_q_heads
            = q_head_idx / 8    (integer division)
```

Q heads 0-7 → KV head 0, Q heads 8-15 → KV head 1, etc. Implemented directly in all three attention kernels; no data duplication or expand-then-contract strategy.

KV cache savings: `4 kv_heads vs 32 q_heads → 8× less cache memory`. At `max_seq_len=2048`, MHA would need `22 × 2 × 2048 × 32 × 64 × 2B = 366MB`; GQA needs `43MB`.

### FP16 precision

FP16 is used for all weights and activations on GPU. Known risks:
- RMSNorm accumulation: accumulates in FP16, susceptible to overflow for large `hidden_dim`. TinyLlama `hidden=2048` stays well within FP16 range.
- Softmax: computed in FP32 to prevent `-inf` masked values from producing NaN in the final sum. The FP32 scores buffer addresses this.
- Logits: projected back to CPU as FP32 before temperature scaling and top-k sampling to preserve sampling distribution fidelity.

Correctness threshold: first 64 generated tokens must achieve ≥ 95% match against llama.cpp with `temperature=0, top_k=1`.

---

## How to Run

```bash
# Build
cd mini-llm-engine && mkdir -p build && cd build
cmake .. && make -j$(nproc)

# CPU inference (W4, unchanged)
./mini-llm-engine --model ../weights/tinyllama-1.1b-fp32.bin \
                  --tokenizer ../weights/tokenizer.model \
                  --prompt "Once upon a time" \
                  --device cpu

# GPU inference (W5, new)
./mini-llm-engine --model ../weights/tinyllama-1.1b-fp32.bin \
                  --tokenizer ../weights/tokenizer.model \
                  --prompt "Once upon a time" \
                  --device cuda

# Benchmark
cd .. && bash benchmarks/run_benchmark.sh
```

Weight loading: the binary loads FP32 weights from disk regardless of `--device`. When `--device cuda` is set, `weights_to_gpu()` converts all 201 tensors to FP16 and uploads to device. CPU FP32 buffers are freed after conversion.

---

## Next Steps (W6)

1. **Flash attention**: Replace three-kernel attention with a tiled flash attention v2 implementation. Target: eliminate the 32MB scores buffer, reduce prefill TTFT by ~2-4× for long sequences.

2. **Batch size > 1**: Extend tensor shapes from `[seq_len, hidden]` to `[batch, seq_len, hidden]`. Requires batched GEMM via cuBLAS `cublasSgemmBatched` or `cublasGemmStridedBatched`.

3. **Continuous batching**: Dynamic request scheduling so GPU is not idle between requests. Requires variable-length sequence handling and a request queue.

4. **INT8 quantization**: Weight-only INT8 (W8A16) to reduce model size from 1.1GB to ~0.55GB, increasing effective arithmetic intensity toward ridge point.

5. **Benchmark sweep**: Run full benchmark protocol across prompt lengths [32, 128, 512, 2048] and compare against llama.cpp.
