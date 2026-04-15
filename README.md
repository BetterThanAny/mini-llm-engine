# Mini-LLM-Engine

从零实现的 LLM 推理引擎,目标模型 TinyLlama-1.1B,FP16,单 GPU (RTX 3080 Laptop, 10 GB VRAM)。用 C++ / CUDA 手写核心路径,对照 llama.cpp FP16 baseline 做公平 benchmark。

同仓库内还附带一个独立 `cuda-kernels/` 子项目,收录推理引擎里用到的 CUDA 算子 (vector_add / reduction / transpose / GEMM / RMSNorm / Softmax) 的独立实现与性能测试。

## 性能 (RTX 3080 Laptop, sm_86)

Prompt = 128 tokens,生成 = 128 tokens,FP16,warmup 3 次 + 测量 5 次取中位数。

| 指标 | Mini-LLM-Engine | llama.cpp FP16 | 比率 |
| --- | --- | --- | --- |
| TTFT (ms) | 24.2 | 14.7 | — |
| 吞吐 (tokens/s) | 91.5 | 148.1 | 62% |
| 峰值显存 (MB) | 7095 | — | — |

INT8 权重量化与 batch 2/4/8 数据见 `benchmark.csv`。

## 手写 CUDA 算子性能

| 算子 | 配置 | naive | 优化 | 加速比 |
| --- | --- | --- | --- | --- |
| RMSNorm | 1024 × 4096, FP32 | 403 GB/s | 565 GB/s (register-cache x) | 1.40× |
| Softmax | 1024 × 4096, FP32 | 217 GB/s | 262 GB/s (online, warp-shuffle) | 1.21× |
| Transpose | 4096², FP32 | 125 GB/s | 351 GB/s (shared-mem tiled) | 2.81× |
| GEMM | N=4096, FP32 | — | 7.74 TFLOPS (register tiled) | 89.8% of cuBLAS |

详见 `cuda-kernels/benchmark.csv` 与 `docs/week2.md`。

## 项目结构

```
mini-llm-engine/          C++/CUDA 推理引擎主项目
├── src/                  主体实现
│   ├── model.{h,cpp}     模型定义 + 权重加载
│   ├── tensor.{h,cpp}    Tensor 抽象 (CPU + CUDA)
│   ├── ops_cpu.{h,cpp}   CPU 算子
│   ├── ops_cuda.{h,cu}   CUDA 算子 (含手写 kernel + cuBLAS 封装)
│   ├── kv_cache.{h,cpp}  KV Cache 管理
│   ├── sampler.{h,cpp}   temperature / top-k / top-p
│   └── main.cpp          CLI 入口
├── benchmarks/
│   ├── run_benchmark.sh            Mini-Engine 一键 benchmark
│   └── run_llamacpp_baseline.sh    llama.cpp 对照 benchmark
├── tests/                GPU vs CPU / FlashAttention 数值验证
└── docs/                 周报 + OWNERSHIP + 设计文档

cuda-kernels/             独立 CUDA 算子集
├── vector_add/ reduction/ transpose/ gemm/ rmsnorm/ softmax/
├── tests/                PyTorch 参考实现比对
└── benchmark.csv         所有 kernel 性能数据

w3_prototype/             Python 原型 (Transformer + KV Cache)
docs/                     全局文档 (周报、面试材料、分析笔记)
benchmark.csv             端到端推理性能数据
```

## 构建与运行

### 前置依赖

- CUDA Toolkit ≥ 12.0
- CMake ≥ 3.18
- GCC ≥ 11
- GPU: Ampere 及以上 (sm_86),默认编译参数针对 RTX 3080

### Mini-LLM-Engine

```bash
cd mini-llm-engine
mkdir -p build && cd build
cmake ..
make -j

# 跑推理
./llm_engine --weights ../weights/tinyllama-1.1b-fp16.bin --prompt "Hello"

# 跑 benchmark (需要 weights 文件)
cd ../benchmarks && bash run_benchmark.sh
```

权重文件不包含在仓库内,需自行转换 (脚本见 `mini-llm-engine/scripts/`)。

### cuda-kernels (独立算子)

```bash
cd cuda-kernels
mkdir -p build && cd build
cmake ..
make -j

# 跑单个 kernel 的 benchmark
./rmsnorm/rmsnorm
./gemm/gemm

# 与 PyTorch 做精度对比
cd ../tests && python test_correctness.py
```

## 数值正确性

| 类型 | 阈值 | 验收方式 |
| --- | --- | --- |
| CUDA kernel vs PyTorch (FP32) | max abs err < 1e-5 | `cuda-kernels/tests/test_correctness.py` |
| CUDA kernel vs PyTorch (FP16) | max abs err < 1e-3 | 同上 |
| 端到端 vs llama.cpp | 前 64 token 一致率 ≥ 95% | `mini-llm-engine/tests/` |

## 模块归属

引擎各模块的独立实现 / 第三方库归属见 [`mini-llm-engine/docs/OWNERSHIP.md`](mini-llm-engine/docs/OWNERSHIP.md)。简述:

- 独立实现: RMSNorm / Softmax / RoPE / INT8 GEMV kernel,KV Cache,Sampler,推理主循环
- 第三方库: cuBLAS (GEMM),sentencepiece (tokenizer)
- 参考实现: 模型加载逻辑参考 llama.cpp,模型结构硬编码 TinyLlama

## 硬件 / 环境

开发与测试环境:

- GPU: NVIDIA GeForce RTX 3080 Laptop (GA104, 10 GB, sm_86)
- Driver: 566.36 (WSL2 passthrough)
- CUDA: 12.0
- OS: Ubuntu 22.04 on WSL2

## 许可证

MIT
