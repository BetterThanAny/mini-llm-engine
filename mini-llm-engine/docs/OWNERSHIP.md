# Module Ownership

## Independently Implemented (from scratch)

| Module | File | Week | Description |
|--------|------|------|-------------|
| Tensor | src/tensor.h, src/tensor.cpp | W4 | CPU+GPU memory management, dtype abstraction |
| CPU Ops | src/ops_cpu.h, src/ops_cpu.cpp | W4 | RMSNorm, RoPE, attention, FFN |
| GPU Ops | src/ops_cuda.h, src/ops_cuda.cu | W4-W7 | All CUDA kernels (see table below) |
| KV Cache | src/kv_cache.h, src/kv_cache.cpp | W4 | Pre-allocated KV buffer, CPU+GPU |
| Sampler | src/sampler.h, src/sampler.cpp | W4 | temperature/top-k/top-p sampling |
| Weight Loader | src/model.cpp | W4 | Custom MINILLM binary format reader |
| Forward GPU | src/model.cpp | W5 | End-to-end GPU FP16 forward pass |
| INT8 Quantization | src/model.cpp | W7 | W8A16 weight-only quantization + forward pass |
| Batch Inference | src/model.cpp | W7 | batch_size > 1 GPU forward pass |
| Inference Loop | src/main.cpp | W4-W7 | prefill+decode loop, benchmark mode, CLI |

## Third-Party Libraries Used

| Library | Purpose | Justification |
|---------|---------|---------------|
| cuBLAS | GEMM for attention + FFN | Industry-standard GPU GEMM; hand-writing FP16 GEMM is a month of work |
| sentencepiece | Tokenization | TinyLlama uses SP tokenizer; tokenizer is not inference-engine scope |

## CUDA Kernels (Hand-Written)

| Kernel | File | Week | Notes |
|--------|------|------|-------|
| RMSNorm (warp-shuffle) | src/ops_cuda.cu | W2→W4 | Ported from cuda-kernels/, adapted to FP16 |
| RoPE | src/ops_cuda.cu | W4 | In-place rotary embedding |
| Softmax (online) | src/ops_cuda.cu | W2→W4 | Ported from cuda-kernels/, FP16 |
| Flash Attention v1 | src/ops_cuda.cu | W6 | Tiled online softmax, O(Bc·HD) SRAM, causal GQA |
| INT8 dequant | src/ops_cuda.cu | W7 | Per-row int8→fp16 for W8A16 weight quantization |

## Scope Boundaries (Intentionally Not Implemented)

- GGUF parsing (use custom binary format instead)
- Multi-GPU / Tensor Parallelism
- HTTP serving layer
- Training / backpropagation
