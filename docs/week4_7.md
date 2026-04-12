# Weeks 4-7: Mini-LLM-Engine — Full Inference Engine

## Overview

Built a complete TinyLlama-1.1B inference engine from scratch in C++/CUDA, targeting RTX 3080 (FP16, single GPU).

## Architecture

```
main.cpp          Entry point, arg parsing, benchmark harness
model.h/cpp       Weight loading (custom binary format), forward passes (CPU/GPU/INT8/batched)
tensor.h/cpp      Tensor class with CPU+GPU memory, FP32/FP16 support, reshape
ops_cpu.h/cpp     CPU forward pass operators (FP32)
ops_cuda.h/cu     GPU operators: RMSNorm, RoPE, Softmax, SiLU, Flash Attention, GEMM, INT8
kv_cache.h/cpp    Per-layer KV cache (CPU or GPU)
sampler.h/cpp     Temperature, top-k, top-p sampling with seed
```

## Week-by-Week Progress

### W4: Foundation
- Tensor class with device-aware memory management (CPU FP32, GPU FP16)
- Weight loading from custom binary format (converted from HuggingFace safetensors)
- CPU forward pass: embedding, RMSNorm, GEMM, RoPE, attention, SwiGLU FFN
- SentencePiece tokenizer integration
- KV cache for autoregressive decoding

### W5: GPU Forward Pass
- FP32-to-FP16 weight upload (`weights_to_gpu`)
- cuBLAS GEMM wrapper (`cublasGemmEx` with FP16 tensor cores)
- Hand-written CUDA kernels: RMSNorm, RoPE, SiLU, embedding lookup, element-wise ops
- Naive multi-kernel attention (Q@K, softmax, scores@V)
- GQA support (4 KV heads, 32 query heads)

### W6: Flash Attention
- Replaced naive 3-kernel attention with Flash Attention v1
- Tiled K/V loading with online softmax (running max + sum)
- O(Bc * HD) shared memory instead of O(q_len * kv_len) global memory
- GQA-aware: kv_repeat = num_q_heads / num_kv_heads

### W7: INT8 Quantisation + Batched Inference
- **INT8 W8A16**: per-row symmetric quantisation of all GEMM weight matrices
  - Fused INT8 GEMV for decode path (no temporary FP16 buffer)
  - Dequant + cuBLAS for prefill path
  - Weight VRAM: ~550 MB (vs ~1.1 GB FP16) — 50% reduction
- **Batched inference**: batch_size up to 8, independent KV caches per stream
- Benchmark harness: warmup/measure/median protocol, CSV output

## Performance (RTX 3080 Laptop GPU)

| Config | TTFT (ms) | Decode (tok/s) | Notes |
|--------|-----------|----------------|-------|
| FP16 | 24.2 | 91.5 | Baseline GPU |
| INT8 W8A16 | 27.4 | 86.3 | ~50% less weight VRAM |
| FP16 batch=2 | 50.1 | 95.7 (aggregate) | |
| FP16 batch=4 | 99.5 | 95.1 (aggregate) | |
| FP16 batch=8 | 198.5 | 93.0 (aggregate) | |

Prompt=128 tokens, generate=128 tokens, temperature=0, top_k=1, median of 5 runs.

### llama.cpp Baseline Comparison (RTX 3080 Laptop GPU)

| Engine | TTFT (ms) | Decode (tok/s) | Model VRAM (MB) |
|--------|-----------|----------------|-----------------|
| llama.cpp FP16 | ~14.7 | 148.1 | 2083 |
| mini-llm-engine FP16 | 24.2 | 91.5 | ~1938 |
| mini-llm-engine INT8 | 27.4 | 86.3 | ~970 |

Decode throughput is ~62% of llama.cpp — reasonable for a from-scratch engine without the extensive optimization llama.cpp has (custom quantized GEMM kernels, optimized memory layouts, etc).

## Correctness Verification

With temperature=0, top_k=1 (greedy decoding), the first 128 generated tokens are **100% identical** between mini-llm-engine and llama.cpp (using passthrough chat template). This exceeds the 95% match rate requirement.

## Hand-Written CUDA Kernels

1. **RMSNorm** — warp shuffle reduction
2. **RoPE** — rotary position embedding
3. **Flash Attention v1** — tiled with online softmax, GQA
4. **INT8 fused GEMV** — reads INT8 weights directly, warp+shared reduction
5. **INT8 dequant** — per-row scale dequantisation
6. **Softmax, SiLU, embedding, element-wise ops**

## Tag

`week7-done`
