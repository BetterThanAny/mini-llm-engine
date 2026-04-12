# OWNERSHIP.md

Tracks which modules are independently implemented vs using third-party libraries.

## mini-llm-engine

### Hand-written CUDA kernels (from scratch)
| Module | File | Description |
|--------|------|-------------|
| RMSNorm | `ops_cuda.cu` | Warp-shuffle reduction, FP16 in/out |
| RoPE | `ops_cuda.cu` | Rotary position embedding, in-place on Q/K |
| Softmax | `ops_cuda.cu` | Numerically stable, along last dim |
| SiLU | `ops_cuda.cu` | SiLU activation for SwiGLU |
| Element-wise mul | `ops_cuda.cu` | For SwiGLU gate * up |
| Element-wise add | `ops_cuda.cu` | Residual connections |
| Embedding lookup | `ops_cuda.cu` | Token ID to FP16 hidden vector |
| Flash Attention | `ops_cuda.cu` | Flash Attention v1 with GQA, tiled online softmax |
| INT8 dequant | `ops_cuda.cu` | Per-row INT8 to FP16 dequantisation |
| INT8 fused GEMV | `ops_cuda.cu` | Fused W8A16 matrix-vector multiply (decode path) |
| INT8 quantise | `ops_cuda.cu` | FP16 to INT8 per-row quantisation |
| Tensor class | `tensor.h/cpp` | CPU/GPU memory management, FP32/FP16, reshape |
| KV Cache | `kv_cache.h/cpp` | Per-layer K/V cache, CPU or GPU |
| Model loading | `model.cpp` | Custom binary format parser, FP32 to FP16 upload |
| Forward pass (CPU) | `model.cpp` + `ops_cpu.cpp` | Full transformer forward in FP32 |
| Forward pass (GPU) | `model.cpp` | FP16 GPU forward, INT8 forward, batched forward |
| Sampler | `sampler.h/cpp` | Temperature, top-k, top-p sampling |
| Benchmark harness | `main.cpp` | Warmup/measure/median protocol, CSV output |

### Third-party libraries
| Library | Usage | Reason |
|---------|-------|--------|
| cuBLAS (`cublasGemmEx`) | GEMM (FP16 matrix multiplication) | High-performance GEMM on tensor cores |
| SentencePiece | Tokenizer (encode/decode) | Standard LLaMA tokenizer format |
| CUDA Runtime | Memory management, kernel launch | Required for GPU programming |

### Model weights
| Item | Source |
|------|--------|
| TinyLlama-1.1B FP32 weights | Converted from HuggingFace `TinyLlama-1.1B-Chat-v1.0` to custom binary format |
| TinyLlama-1.1B FP16 GGUF | Converted via llama.cpp for baseline comparison |

## cuda-kernels (standalone experiments)

All kernels are hand-written from scratch:

| Kernel | Variants |
|--------|----------|
| vector_add | Basic CUDA kernel |
| parallel reduction | v1 naive, v2 sequential addressing, v3 first-add-during-load, v4 warp shuffle |
| matrix transpose | Naive + shared-memory tiled |
| GEMM | v1 naive, v2 shared-memory tiled, v3 register tiling (BM=BN=128, BK=8, TM=TN=8) |
| RMSNorm | v1 naive, v2 warp shuffle |
| Softmax | Numerically stable with online max tracking |

All kernels include correctness tests against PyTorch reference and performance benchmarks.
