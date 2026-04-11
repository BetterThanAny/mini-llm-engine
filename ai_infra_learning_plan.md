# AI Infra 转型学习计划（执行版）

> 背景：北大计科本科，C/C++ 主力，有编译器（SysY）和 OS（xv6）项目经验，目标转向 AI Infra 方向
>
> 总时间：8 周主线 + W0 热身 3-5 天（约 2 个月）
>
> 每日节奏：学习 2h + 编码 2-3h + 复盘 30min
>
> 产出目标：1 个可写入简历的项目 + 能应对 AI Infra 面试
>
> 硬件环境：RTX 3080 单卡（如无本地 GPU，用 AutoDL 租卡或学校 GPU 服务器）

---

## 项目边界（先定死，不可蔓延）

**必做：**
- Mini-LLM-Engine（独立实现核心推理链路），仅支持 TinyLlama-1.1B、FP16、单卡
- 手写 CUDA 算子至少 2 个（RMSNorm + Softmax，或 RMSNorm + RoPE）
- Attention + FFN + KV Cache + Sampling 跑通并有 benchmark

**不做：**
- 自己写 GGUF 解析（用现成库 / 硬编码模型结构）
- 自己写 tokenizer（用 sentencepiece）
- 多 GPU / Tensor Parallelism 编码（只需理论能讲）
- 完整 serving 系统 / HTTP server

**可选加分（W6 二选一）：**
- INT8/INT4 量化推理
- Continuous Batching

### 显存预算（RTX 3080, 10GB VRAM）

在开始编码前必须算清楚显存上限，避免中途反复 OOM：

```text
TinyLlama-1.1B FP16 显存预估：
  模型权重：1.1B × 2 bytes = ~2.2 GB
  KV Cache（每层）：2 × seq_len × num_heads × head_dim × 2 bytes
    - 22 层, 32 heads, head_dim=64
    - seq_len=512:  2 × 512 × 32 × 64 × 2 × 22 = ~90 MB
    - seq_len=2048: 2 × 2048 × 32 × 64 × 2 × 22 = ~360 MB
  激活值（推理时较小）：~200-500 MB
  CUDA 运行时开销：~500 MB

  总计（seq_len=2048, batch=1）：~3.5 GB → 安全
  总计（seq_len=2048, batch=4）：~4.6 GB → 安全
  总计（seq_len=2048, batch=16）：~8.5 GB → 接近上限，需谨慎

硬性约束：
  - max_seq_len = 2048（不超过此值）
  - batch_size 上限 = 8（留 ~2GB 安全余量）
  - 如做 Continuous Batching，总 token 数不超过 8192
```

---

## AI Infra 完整知识图谱

### 第一层：基础（学过但需要复习）

```text
C/C++ 工程能力
├─ 内存管理：new/delete, smart pointers, RAII
├─ 现代 C++：move semantics, std::optional, lambda, constexpr
└─ 构建工具：CMake, Makefile

计算机体系结构
├─ CPU：流水线、分支预测、乱序执行
├─ 缓存体系：L1/L2/L3, cache line, 局部性原理
├─ 内存：虚拟内存、TLB、NUMA
└─ 指令级并行：SIMD (SSE/AVX)

编译原理（SysY 编译器）
├─ 前端：lexer → parser → AST → 语义分析
├─ 中间表示：IR 设计、SSA 形式
├─ 优化：常量折叠、死代码消除、循环优化
└─ 后端：指令选择、寄存器分配、指令调度

操作系统（xv6 Lab）
├─ 进程/线程管理、调度算法
├─ 虚拟内存、页表、缺页中断
├─ 并发同步：mutex, spinlock, RWLock, lock-free
└─ I/O 模型：阻塞/非阻塞、epoll、io_uring
```

### 第二层：GPU 与并行计算（需要新学）

```text
GPU 硬件架构
├─ SM (Streaming Multiprocessor) 结构
├─ Warp 调度与 SIMT 执行模型
├─ 显存层级：Global / Shared / Local / Register
├─ 显存带宽 vs 计算吞吐 (Roofline Model)
└─ Tensor Core 工作原理

CUDA 编程
├─ 基础：kernel, grid, block, thread 层级
├─ 内存管理：cudaMalloc, cudaMemcpy, unified memory
├─ 同步：__syncthreads(), atomic, cooperative groups
├─ Warp 级原语：__shfl_sync, warp reduce
└─ 性能分析：Nsight Compute, nvprof, roofline

高性能优化技术
├─ Coalesced Memory Access（合并访存）
├─ Shared Memory Tiling（分块复用）
├─ Bank Conflict 避免
├─ Occupancy 调优（寄存器/shared mem 平衡）
├─ 指令级并行 (ILP) 与访存隐藏
└─ Parallel Reduction / Scan / Prefix Sum
```

### 第三层：深度学习系统知识（需要新学）

```text
深度学习基础（够用即可，不需要算法层面太深）
├─ 前向传播 / 反向传播 / 计算图
├─ 常见层：Linear, Conv2d, LayerNorm, Softmax
├─ Transformer 架构：Self-Attention, FFN, 位置编码
└─ LLM 推理流程：tokenize → prefill → decode → detok

算子实现与优化
├─ GEMM 优化（Tiling → Shared Mem → Register Block）
├─ Softmax / LayerNorm CUDA 实现
├─ FlashAttention 原理（tiling + online softmax）
├─ 算子融合（Kernel Fusion）
├─ 混合精度：FP32 / FP16 / BF16 / INT8 / INT4
└─ cuBLAS / cuDNN / CUTLASS 使用

LLM 推理系统（核心重点）
├─ KV Cache 机制与显存管理
├─ PagedAttention（vLLM 核心思想）
├─ Continuous Batching / Dynamic Batching
├─ 投机解码 (Speculative Decoding)
├─ 模型量化：GPTQ, AWQ, GGUF 格式
├─ Tensor Parallelism / Pipeline Parallelism（理论即可）
└─ 请求调度与 SLA 管理

模型编译器（进阶加分项，非必须）
├─ TVM：Relay IR → TIR → 代码生成
├─ MLIR：Dialect 体系、Pass 编写
├─ Triton：Python DSL → GPU kernel
├─ 图优化：算子融合、常量折叠、layout 变换
└─ 算子优化：auto-tuning, polyhedral model
```

### 面试中各知识点考察频率

| 知识点                                  | 面试频率  | 说明              |
| --------------------------------------- | --------- | ----------------- |
| CUDA 编程模型（grid/block/thread）      | ★★★★★ | 必问              |
| GPU 内存层级与合并访存                  | ★★★★★ | 必问              |
| GEMM 优化                              | ★★★★★ | 手写或讲思路      |
| Transformer / Attention 机制            | ★★★★★ | 必须能讲清楚      |
| KV Cache 原理                           | ★★★★★ | LLM 推理必问      |
| FlashAttention 原理                     | ★★★★  | 高频              |
| 量化（FP16/INT8）                       | ★★★★  | 高频              |
| PagedAttention / Continuous Batching    | ★★★★  | vLLM 相关必问     |
| Warp 级编程                             | ★★★    | 中频              |
| 模型编译器（TVM/MLIR）                  | ★★★    | 投编译方向才必问  |
| 分布式训练（AllReduce/TP/PP）           | ★★      | 偏训练方向        |

---

## 8 周主线 + W0 热身执行计划（可打勾版）

### 总览

| 周次 | 核心任务 | 硬验收 | 止损点 |
| ---- | -------- | ------ | ------ |
| W0 (3-5天) | SysY + xv6 回血，整理可讲稿 | 各 1 页总结，能 5 分钟讲清楚 | 超时则只保"架构 + 难点 + bug"三块 |
| W1 | CUDA 基础：vec add / reduction / GEMM v1-v2 | cuda-kernels 仓库可跑，含性能表 | GEMM v2 卡 >2 天就先用 cuBLAS 替代 |
| W2 | CUDA 进阶：RMSNorm / Softmax + Nsight 分析 | 至少 1 个 kernel 达到 optimized >= 1.3x naive | 达不到 1.3x 也收敛，保 profiling 分析 |
| W3 | Transformer 推理理解 + Python 原型（含 KV Cache）+ llama.cpp 核心链路阅读（主链路/KV Cache/调度） | Python 原型可运行 + llama.cpp 架构笔记 + baseline benchmark | 精度验证卡住就用 top-k 一致率做弱验收 |
| W4 | 引擎骨架（C++）：CPU 闭环为主，最小 GPU 路径（cuBLAS + 1 自写 kernel）为加分 | C++ 端 CPU prefill + decode 跑通 | 若 C++ 集成阻塞，先 Python + C++ 混合跑通 |
| W5 | GPU 端到端 + 按 benchmark protocol 做性能对比 | TinyLlama GPU 可稳定生成 + 完整对比表 | **Milestone Gate：不通过则 W6 取消加分项** |
| W6 | 可选加分：INT8 量化 或 Continuous Batching 二选一 | 至少一个加分点落地并有数据 | 二者都卡则放弃加分，转文档和面试材料 |
| W7 | 文档 + 可复现脚本 + Demo 录屏 + 简历项目描述 | README + run_benchmark.sh + 3-5 分钟录屏 | 不再加功能，只打磨 |
| W8 | 面试冲刺 + 投递（保留 20% 定向修复） | 10 分钟讲解稿 + 问答清单 + 投递跟踪表 | 只做 bug fix，不加新功能 |

### 每周固定交付（强制）

每周日晚必须产出以下三样，缺一不可：

1. **代码**：1 个可运行里程碑（git tag: `week0-done`, `week1-done`, ...）
2. **数据**：1 份 `benchmark.csv`（性能数据，哪怕只有一行）
3. **文档**：1 份 `docs/weekN.md`（做了什么、数据、问题、**本周最大风险 + 回退方案**、下一步）

### 必建文档（项目初始化时创建）

**`docs/correctness.md` — 数值验收标准**

```text
精度验收定义：
  - CUDA kernel vs PyTorch 参考实现：max absolute error < 1e-5 (FP32), < 1e-3 (FP16)
  - 端到端生成验证（主验收）：相同 prompt + temperature=0 + top_k=1 条件下，
    前 64 个生成 token 一致率 >= 95%
  - 严格验收（可选）：前 32 个生成 token 完全一致
  - 弱验收（止损用）：top-5 token 一致率 >= 95%

每次修改 kernel 后必须跑 tests/test_correctness.py 验证
```

**`docs/benchmark_protocol.md` — 性能测量规范**

```text
公平基准协议：
  - 模型：TinyLlama-1.1B, FP16
  - Prompt 长度：固定 128 tokens
  - 生成长度：固定 128 tokens
  - 随机性控制：固定 random seed = 42，temperature=0，top_k=1
  - Warmup：3 次（丢弃结果）
  - 测量：重复 5 次，取中位数
  - 环境记录：GPU 型号、驱动版本、CUDA 版本、GPU 温度
  - 功耗/频率记录：power limit、SM/memory clocks（nvidia-smi），并保证无其他高负载任务
  - llama.cpp 基线：相同 prompt、相同 FP16 配置、相同硬件
  - 指标：TTFT (ms), tokens/s, 显存峰值 (MB), P95 decode latency (ms)

所有 benchmark 数据用 benchmarks/run_benchmark.sh 一键复现
```

---

### W0：基础回血（3-5 天）

> 目标：快速唤醒 SysY 和 xv6 的记忆，能在面试中 5 分钟讲清楚每个项目。

#### Day 1-2：复习 SysY 编译器

- [ ] 重新读自己的代码，画出编译流水线图：`源码 → Lexer → Parser → AST → Koopa IR → RISC-V`
- [ ] 整理三块核心内容：
  - **架构**：IR 是怎么设计的？为什么这样设计？
  - **难点**：寄存器分配用的什么算法？函数调用怎么处理？栈帧布局？
  - **Bug**：遇到的最难的 bug 是什么？怎么定位和解决的？
- [ ] 产出：`docs/sysy_summary.md`（1 页，面试前快速过一遍）

#### Day 3-4：复习 xv6 OS Lab

- [ ] 整理三块核心内容：
  - **架构**：系统调用完整链路（用户态 → ecall → 内核态 → 返回）
  - **难点**：页表结构（三级页表、PTE 格式）、spinlock vs sleeplock
  - **Bug**：并发或内存相关的 bug 怎么调的？
- [ ] 产出：`docs/xv6_summary.md`（1 页）

#### Day 5：C++ 现代特性速览

- [ ] 重点复习：`unique_ptr / shared_ptr`、move semantics、`std::optional`、lambda、`constexpr`
- [ ] 不需要深入模板元编程，能读懂 llama.cpp 级别代码即可

**止损**：如果超时，只保每个项目的"架构 + 难点 + bug"三块，够面试讲就行。

**验收**：
- [ ] 两份项目总结文档完成
- [ ] 能对着文档 5 分钟讲清楚每个项目

**参考资源：**
- 自己的 SysY 和 xv6 代码仓库
- [C++ Core Guidelines](https://isocpp.github.io/CppCoreGuidelines/)

---

### W1：CUDA 基础（7 天）

> 目标：掌握 CUDA 编程模型，实现 GEMM v1-v2，建立 cuda-kernels 仓库。

#### Day 1：GPU 架构总览

- [ ] 学习内容：
  - GPU vs CPU 架构差异（多核 vs 众核）
  - SM 结构：CUDA cores, Tensor Cores, warp scheduler
  - SIMT 执行模型：warp（32 threads）是最小调度单位
  - 显存层级：Register > Shared Memory > L1/L2 Cache > Global Memory
- [ ] 产出：手画一张 GPU 架构图，标注各层存储的大小和延迟
- 推荐资源：
  - [CUDA C++ Programming Guide — Chapter 1-2](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)

#### Day 2：CUDA Hello World

- [ ] 学习内容：
  - CUDA 编程模型：kernel 函数、`<<<grid, block>>>` 启动
  - thread/block/grid 的索引计算
  - `cudaMalloc`, `cudaMemcpy`, `cudaFree`
  - 编译：`nvcc` 基本用法
- [ ] 代码任务：
  - vector_add：两个数组逐元素相加
  - 实验不同 block size (32, 64, 128, 256, 512) 对性能的影响
- [ ] 验收：kernel 正确执行，理解 threadIdx/blockIdx/blockDim 的关系

#### Day 3：Parallel Reduction

- [ ] 学习内容：
  - 朴素 reduction 的 warp divergence 问题
  - 优化：sequential addressing → first-add-during-load → warp-level unrolling
  - `__syncthreads()` 的作用与限制
- [ ] 代码任务：
  - 实现 sum reduction kernel，逐步优化 4 个版本
  - 对比每个版本的性能（用 CUDA event 计时）
- 推荐资源：
  - Mark Harris: Optimizing Parallel Reduction in CUDA（搜索即可找到 PDF）

#### Day 4：Shared Memory 与 Tiling

- [ ] 学习内容：
  - Shared memory 的物理位置（on-chip，与 L1 共享）
  - Bank conflict 原理与避免策略
  - Tiling 模式：把 global memory 数据分块搬到 shared memory
- [ ] 代码任务：
  - 矩阵转置：naive → shared memory tiled（消除 bank conflict）
  - 对比两个版本的显存带宽利用率
- [ ] 验收：理解 coalesced access 和 bank conflict

#### Day 5：GEMM v1 — 朴素实现

- [ ] 学习内容：
  - 矩阵乘法 C = A x B 的并行化思路
  - 每个 thread 计算 C 的一个元素
  - 分析瓶颈：global memory 带宽受限
- [ ] 代码任务：
  - 实现 naive GEMM kernel
  - 与 cuBLAS 对比性能，记录差距（记入 benchmark.csv）

#### Day 6-7：GEMM v2 — Shared Memory Tiling

- [ ] 学习内容：
  - 分块矩阵乘法（tiled GEMM）
  - 每个 block 负责 C 的一个 tile
  - 循环加载 A 和 B 的 tile 到 shared memory
- [ ] 代码任务：
  - 实现 tiled GEMM（tile size = 16x16 或 32x32）
  - 对比 naive 版本的加速比
- [ ] 验收：tiled GEMM 比 naive 快 2-6x（首次实现的现实预期，5x+ 为优秀）

**止损**：GEMM v2 卡住超过 2 天，先跳过，后续引擎项目中用 cuBLAS 替代。GEMM 优化思路面试能讲清楚即可。

**周验收**：
- [ ] `cuda-kernels/` 仓库创建，包含 vector_add / reduction / transpose / gemm_naive / gemm_tiled
- [ ] `benchmark.csv` 记录每个 kernel 的性能数据
- [ ] `docs/week1.md` 完成

---

### W2：CUDA 进阶与算子实现（7 天）

> 目标：实现 RMSNorm + Softmax CUDA kernel（这两个会直接用到引擎项目中），掌握 Nsight 性能分析。

#### Day 8：GEMM v3 — Register Tiling（可选，视 W1 进度）

- [ ] 每个 thread 计算多个输出元素（增大 ILP），`float4` 向量化访存
- [ ] 性能目标：达到 cuBLAS 的 30-50% 为达标，50%+ 为优秀
- 推荐资源：[CUTLASS GEMM 层级分解图](https://github.com/NVIDIA/cutlass)
- **如果 W1 GEMM v2 未完成，此天用于补完 v2**

#### Day 9：Softmax CUDA 实现（必做）

- [ ] 学习内容：
  - Softmax 的数值稳定性问题（减最大值）
  - Online Softmax 算法（一次遍历完成 max + sum）
  - Warp-level reduction 用 `__shfl_xor_sync`
- [ ] 代码任务：
  - 实现 safe softmax kernel（two-pass: find max → compute exp/sum）
  - 实现 online softmax kernel（one-pass）
  - 对比两种实现的性能
- [ ] 验收：数值与 PyTorch `F.softmax()` 对比误差 < 1e-5

#### Day 10：RMSNorm CUDA 实现（必做）

- [ ] 学习内容：
  - RMSNorm 公式（LLaMA 使用）：`y = x / sqrt(mean(x^2) + eps) * gamma`
  - Welford's online algorithm
  - LayerNorm 与 RMSNorm 区别
- [ ] 代码任务：
  - 实现 RMSNorm kernel
  - 验证数值精度（与 PyTorch 对比）
- [ ] 验收：精度正确，性能记入 benchmark.csv

#### Day 11：Warp 级编程

- [ ] 学习内容：
  - Warp shuffle：`__shfl_sync`, `__shfl_down_sync`, `__shfl_xor_sync`
  - Warp-level reduction 和 scan
- [ ] 代码任务：
  - 用 warp shuffle 重写 reduction（不使用 shared memory）
  - 把 warp shuffle 技术应用到 Softmax/RMSNorm kernel 中

#### Day 12：混合精度基础

- [ ] 学习内容：
  - FP32 / FP16 / BF16 的表示范围与精度
  - 为什么推理用 FP16/BF16（带宽减半，Tensor Core 加速）
  - `__half` 类型和 CUDA 半精度运算
  - INT8 量化基本概念（对称/非对称，per-tensor/per-channel）
- [ ] 代码任务：
  - 把 GEMM 改为 FP16 版本，对比精度和速度
  - 简单的 INT8 量化实验：float → int8 → dequant → 对比误差

#### Day 13-14：Nsight 性能分析 + 阶段总结

- [ ] 学习内容：
  - Nsight Compute：分析 kernel 的 compute / memory 利用率
  - Roofline Model：判断 kernel 是 compute-bound 还是 memory-bound
- [ ] 代码任务：
  - 用 Nsight 分析写过的所有 kernel，找出瓶颈
  - 整理 CUDA 优化笔记
  - 对 RMSNorm 和 Softmax 做至少一轮基于 profiling 的优化
- [ ] 产出：所有 kernel 的 profiling 分析记录

**止损**：优化后达不到 naive 的 1.3x 也收敛，保留 profiling 分析报告即可。面试能讲"我做了什么分析、瓶颈在哪、为什么"比数字本身重要。

**周验收**：
- [ ] RMSNorm + Softmax kernel 实现并通过精度验证
- [ ] 至少 1 个 kernel 有 Nsight profiling 分析
- [ ] `benchmark.csv` 更新
- [ ] `docs/week2.md` 完成

---

### W3：Transformer 与 LLM 推理理解（7 天）

> 目标：理解 Transformer 和 LLM 推理全流程，用 Python 写出原型，阅读 llama.cpp 核心链路为下周 C++ 实现做准备。

#### Day 15：Transformer 架构

- [ ] 学习内容：
  - Self-Attention：`Attention(Q,K,V) = softmax(QK^T / sqrt(d)) V`
  - Multi-Head Attention、FFN、Residual + LayerNorm
  - 位置编码：sinusoidal / RoPE
- [ ] 产出：能在白板上画出 Transformer 一层的完整数据流
- 推荐资源：
  - Jay Alammar: The Illustrated Transformer
  - Attention Is All You Need 论文 Section 3

#### Day 16：LLM 推理流程 + KV Cache

- [ ] 学习内容：
  - Prefill（处理 prompt，compute-bound）vs Decode（逐 token 生成，memory-bound）
  - KV Cache：缓存每层的 K、V，避免重复计算
  - 显存占用：`2 x num_layers x seq_len x hidden_dim x dtype_size`
  - MQA / GQA 对 KV Cache 的影响
- [ ] 代码任务：
  - 用 PyTorch 手写简化版 Transformer decoder（不用 nn.Transformer）
  - 实现 KV Cache，对比有无 cache 的计算量
- [ ] 验收：Python 原型能正确做 autoregressive 生成

#### Day 17：FlashAttention + PagedAttention + 量化（理论）

- [ ] 学习内容：
  - FlashAttention：tiling + online softmax，O(N) 显存
  - PagedAttention：KV Cache 分页管理（类比 OS 页表）
  - Continuous Batching 概念
  - 量化概念：PTQ (GPTQ, AWQ)、GGUF 格式（理论了解即可，动手实验移到 W6）
- [ ] 代码任务：
  - 用 Python 实现 simplified FlashAttention（理解算法逻辑）
  - 用 Python 实现简化版 Paged KV Cache Manager（allocate / free）
- 推荐资源：
  - FlashAttention 论文：Dao et al., 2022
  - vLLM 论文：Kwon et al., 2023

#### Day 18-21：阅读 llama.cpp 源码（聚焦三块）

- [ ] 目标：为 W4 的 C++ 实现建立参考，只读三块核心逻辑，其余跳过
- [ ] **聚焦范围（不要发散）**：
  1. **推理主链路**：从 `main()` 跟踪 `load model → tokenize → prefill → decode loop → sample → detokenize`
  2. **KV Cache 管理**：分配、更新、内存布局
  3. **CUDA kernel dispatch**：哪些算子走 GPU、如何调度
- [ ] 重点文件：
  - `llama.cpp` / `llama.h`：模型加载、推理主流程
  - `ggml.c`：底层张量运算库
  - `ggml-cuda.cu`：CUDA 算子实现（只看 RMSNorm/Attention/GEMM 相关部分）
- [ ] 代码任务：
  - 自己编译 llama.cpp，跑通 TinyLlama 1.1B
  - 用不同量化级别（Q4_0, Q8_0, F16）对比速度和质量
  - **这个 benchmark 数据就是你后续引擎的对比基线**
- [ ] 产出：`docs/llamacpp_analysis.md`（架构图 + 三块核心模块分析）
- [ ] **不需要读的**：训练相关代码、server 代码、非 CUDA backend、GGUF 解析细节

**止损**：精度验证卡住就用 top-k 一致率做弱验收（生成的 top-5 token 一致即可）。llama.cpp 源码严格控制在三块范围内，超出范围的留到 W5 有需要时再看。

**周验收**：
- [ ] Python Transformer 原型可运行，KV Cache 实现正确
- [ ] llama.cpp 编译通过，TinyLlama 跑通，baseline benchmark 数据记录
- [ ] `docs/llamacpp_analysis.md` 完成
- [ ] `benchmark.csv` 更新（llama.cpp baseline 数据）
- [ ] `docs/week3.md` 完成

---

### W4：Mini-LLM-Engine 骨架（7 天）

> 目标：搭建 C++ 引擎骨架，完成 **CPU 闭环**为首要目标，最小 GPU 路径（cuBLAS + 1 个自写 kernel）为加分。

#### Day 22-23：项目骨架 + 模型加载

- [ ] 搭建 CMake 项目结构：

```text
mini-llm-engine/
├── CMakeLists.txt
├── src/
│   ├── main.cpp           # 入口
│   ├── model.h/cpp        # 模型定义与加载
│   ├── tensor.h/cpp       # Tensor 类（CPU + CUDA 内存管理）
│   ├── ops_cpu.h/cpp      # CPU 算子实现
│   ├── ops_cuda.h/cu      # CUDA 算子实现
│   ├── kv_cache.h/cpp     # KV Cache 管理
│   └── sampler.h/cpp      # Sampling 策略
├── third_party/
│   └── sentencepiece/     # tokenizer
├── tests/                 # 精度验证脚本
├── benchmarks/            # 性能测试
├── docs/
│   └── OWNERSHIP.md       # 模块归属说明
└── README.md
```

- [ ] 模型加载：用现成库加载权重（或参考 llama.cpp 的加载逻辑，硬编码 TinyLlama 结构）
- [ ] Tokenizer：集成 sentencepiece
- [ ] 验收：能加载模型权重到内存，能 tokenize/detokenize

#### Day 24-25：CPU Forward Pass

- [ ] 实现 CPU 版本的所有算子：RMSNorm, RoPE, Attention, FFN (SiLU + Linear)
- [ ] 实现 KV Cache 的分配与更新
- [ ] 实现 Sampling（temperature, top-k, top-p）
- [ ] 验收：CPU 版本能生成文本，输出与 llama.cpp 对比合理

#### Day 26-28：最小 GPU 路径（加分，非必须）

- [ ] 把 W2 写的 RMSNorm kernel 集成进来（1 个自写 kernel 即可证明能力）
- [ ] GEMM 用 cuBLAS（不自己写，降低集成风险）
- [ ] KV Cache 搬到 GPU 显存
- [ ] 验收：GPU 路径能跑通至少 1 层的 forward pass
- [ ] **注意**：完整 GPU 路径留到 W5，本周只需证明 C++ ↔ CUDA 集成链路通了

**止损**：若 C++ 集成阻塞（CMake、链接问题等），先用 Python + ctypes/pybind11 调用 CUDA kernel 跑通闭环，再逐步迁移到纯 C++。

**周验收**：
- [ ] mini-llm-engine 仓库创建，项目结构完整
- [ ] OWNERSHIP.md 初版完成（标注每个模块：独立实现 / 第三方库 / 参考实现）
- [ ] 至少 CPU 闭环跑通（GPU 闭环为加分）
- [ ] `benchmark.csv` 更新
- [ ] `docs/week4.md` 完成

---

### W5：GPU 推理 + 性能优化（7 天）

> 目标：GPU 路径完全跑通，生成文本正确，产出完整 benchmark 报告。
>
> **里程碑门禁（Milestone Gate）**：W5 结束时必须达到"可稳定生成 + 可重复 benchmark"。如果达不到，W6 直接取消加分项，全力转稳定性与文档。

#### Day 29-31：GPU 端到端调通

- [ ] 必须集成 2 个自写 kernel（RMSNorm + Softmax 或 RMSNorm + RoPE）；其余算子可回退到库实现
- [ ] GEMM 统一走 cuBLAS
- [ ] 确保 TinyLlama 在 GPU 上完整推理
- [ ] 精度验证：按 `docs/correctness.md` 标准，与 llama.cpp FP16 输出对比

#### Day 32-33：性能测量与优化

- [ ] **严格按 `docs/benchmark_protocol.md` 执行**：
  - 固定 prompt = 128 tokens, 生成 = 128 tokens
  - Warmup 3 次，测量 5 次取中位数
  - 记录 GPU 型号、驱动版本、CUDA 版本
- [ ] 测量指标：
  - **TTFT** (Time To First Token)：首 token 延迟
  - **tokens/s**：decode 阶段吞吐
  - **显存占用**：峰值 GPU memory
  - **P95 延迟**：多次运行的延迟分布
- [ ] 用 Nsight 找瓶颈，尝试优化 1-2 个热点 kernel
- [ ] 与 llama.cpp 对比（llama.cpp 也按相同 protocol 测量）

#### Day 34-35：Benchmark 报告

- [ ] 整理完整对比表：

```text
| 指标           | Mini-LLM-Engine (FP16) | llama.cpp (FP16) | 比率   |
| -------------- | ---------------------- | ----------------- | ------ |
| TTFT (ms)      | XXX                    | XXX               | XX%    |
| tokens/s       | XXX                    | XXX               | XX%    |
| 显存峰值 (MB)  | XXX                    | XXX               | XX%    |
| P95 延迟 (ms)  | XXX                    | XXX               | XX%    |
```

- [ ] 如果差距大（<30% of llama.cpp），给出瓶颈归因和改进计划
- [ ] 产出：`benchmarks/report.md`

**止损**：若端到端不稳定，先锁定 batch=1、固定 prompt 长度保闭环。性能差距大不可怕，关键是能解释"瓶颈在哪、为什么、怎么改"。

**周验收**：
- [ ] TinyLlama GPU 推理稳定运行
- [ ] 完整 benchmark 数据（至少 TTFT + tokens/s + 显存）
- [ ] `benchmarks/report.md` 完成
- [ ] `benchmark.csv` 更新
- [ ] `docs/week5.md` 完成

---

### W6：可选加分项（7 天）

> 目标：二选一做深，为简历增加一个技术亮点。
>
> **前置条件**：W5 milestone gate 通过（可稳定生成 + 可重复 benchmark）。未通过则本周全力修稳定性和文档。

#### 选项 A：INT8/INT4 量化推理

- [ ] Day 1：量化理论实验（从 W3 移来的动手部分）
  - Python 实现 per-channel INT8 量化和反量化
  - 在小模型上对比量化前后输出差异
- [ ] Day 2-3：实现 INT8 dequantize CUDA kernel
- [ ] Day 4-5：量化 GEMM（weight 以 INT8 存储，运行时反量化为 FP16 计算）
- [ ] Day 6-7：对比量化前后的精度、速度、显存
- [ ] 验收：量化模型能正常推理，显存明显减少

#### 选项 B：Continuous Batching

- [ ] 实现请求队列和 iteration-level 调度器
- [ ] batch 中每个请求独立管理 KV Cache 和 seq_len
- [ ] 对比 batch=1 串行 vs continuous batching 的吞吐
- [ ] 验收：多请求下吞吐量高于串行

**止损**：两个都卡则果断放弃加分项，转入 W7 的文档和面试材料。一个跑通的核心引擎 > 一堆半成品功能。

**周验收**：
- [ ] 加分项落地并有数据（或明确放弃并记录原因）
- [ ] `benchmark.csv` 更新
- [ ] `docs/week6.md` 完成

---

### W7：文档打磨 + 面试材料（7 天）

> 目标：项目可交付、可展示、可讲解。本周开始不再加任何功能。

#### Day 43-44：README + 架构文档

- [ ] README.md 完善：
  - 项目简介（一句话说清楚这是什么）
  - 架构图（Mermaid 或手画）
  - 快速启动（git clone → cmake → run）
  - 性能数据表
  - 技术亮点
- [ ] 画图：
  - 系统架构图（模块关系）
  - 推理流程时序图（prefill → decode → sample）
  - KV Cache 内存布局图

#### Day 45：OWNERSHIP.md 定稿

- [ ] 明确标注每个模块的归属：

```text
| 模块           | 归属                  | 说明                              |
| -------------- | --------------------- | --------------------------------- |
| RMSNorm kernel | 独立实现              | 手写 CUDA，含 warp shuffle 优化   |
| Softmax kernel | 独立实现              | online softmax 算法               |
| GEMM           | cuBLAS                | 调用 cuBLAS API                   |
| RoPE kernel    | 独立实现              | CUDA 实现                         |
| Tokenizer      | sentencepiece (库)    | 集成调用                          |
| 模型加载       | 参考 llama.cpp 简化   | 硬编码 LLaMA 结构                 |
| KV Cache       | 独立设计              | 动态分配 + GPU 端管理             |
| Sampling       | 独立实现              | temperature / top-k / top-p       |
```

#### Day 46：简历项目描述

- [ ] 写 3 条 bullet points（含量化数据）
- [ ] 确保每条都能展开讲 2-3 分钟

#### Day 47-48：可复现实验脚本 + Demo 录屏

- [ ] 编写 `benchmarks/run_benchmark.sh`：一键复现所有 benchmark 数据
  - 自动 warmup → 测量 → 输出 CSV
  - 自动对比 llama.cpp 基线（如已安装）
- [ ] 编写 `tests/run_correctness.sh`：一键跑精度验证
- [ ] **录制 3-5 分钟 Demo 视频**：
  - 展示：编译 → 加载模型 → 输入 prompt → 生成输出 → 显示性能数据
  - 面试时比纯文档有说服力得多（可以放 GitHub README 或面试时播放）
- [ ] KV Cache 结构图 + 设计决策说明
- [ ] 手写 kernel 的优化前后对比（含 Nsight 截图或数据）
- [ ] 与 llama.cpp 的对比分析（不是竞争，是"基线验证"）

**周验收**：
- [ ] README.md 完整，新人看 README 能在 10 分钟内跑起项目
- [ ] `benchmarks/run_benchmark.sh` 和 `tests/run_correctness.sh` 可一键执行
- [ ] 3-5 分钟 Demo 录屏完成
- [ ] OWNERSHIP.md 定稿
- [ ] 简历项目描述写好
- [ ] 关键设计文档完成
- [ ] `docs/week7.md` 完成

---

### W8：面试冲刺（7 天）

> 目标：能在面试中自信地讲项目、答八股。本周 80% 时间用于面试准备，**保留 20% 用于模拟面试暴露问题后的定向修复**（仅限 bug fix 和数据补充，不加新功能）。

#### Day 49-50：CUDA 高频面试题

- [ ] GPU 架构（SM, warp, 内存层级）
- [ ] GEMM 优化完整思路（naive → tiling → register blocking）
- [ ] Coalesced access, bank conflict, occupancy
- [ ] 手写 reduction / softmax kernel（纸上或白板）

#### Day 51-52：LLM 推理系统面试题

- [ ] Transformer 前向过程（手画数据流）
- [ ] KV Cache 原理、显存分析、MQA/GQA
- [ ] FlashAttention 为什么快（tiling + online softmax）
- [ ] PagedAttention 与 OS 虚拟内存类比（用 xv6 经验讲）
- [ ] Continuous Batching vs Static Batching
- [ ] 量化方法对比（FP16/INT8/INT4，对精度和速度的影响）
- [ ] Tensor Parallelism 理论（Column/Row parallel + AllReduce）

#### Day 53-54：项目讲解准备

- [ ] 三个项目各准备 5-10 分钟讲解：
  - **SysY 编译器**：架构、IR 设计、寄存器分配、难点
  - **xv6 OS Lab**：页表、系统调用、并发、难点
  - **Mini-LLM-Engine**：架构、手写 kernel、KV Cache 设计、性能数据
- [ ] 准备常见追问的回答：
  - "为什么自己写而不用现成框架？"
  - "性能和 llama.cpp 差多少？瓶颈在哪？"
  - "如果要支持更大模型怎么做？"
  - "KV Cache 显存不够怎么办？"

#### Day 55-56：模拟面试 + 投递

- [ ] 对着镜子或找同学模拟面试
- [ ] 自我介绍（1 分钟版 + 3 分钟版）
- [ ] 建立投递跟踪表（公司、岗位、状态、面试时间）
- [ ] 开始投递

**周验收**：
- [ ] 10 分钟项目讲解稿定稿
- [ ] 高频面试问答清单（含自己的回答要点）
- [ ] 投递跟踪表建立，至少投出 5 家
- [ ] `docs/week8.md` 完成

---

## 面试防追问策略

### 核心原则：主叙事始终是"我自己独立设计实现的引擎"

1. **OWNERSHIP.md**：明确"哪些模块独立实现、哪些用了第三方库"，面试前过一遍
2. **关键模块保留设计文档**：KV Cache 结构图、kernel 优化前后对比数据
3. **对比 llama.cpp 只用于"基线验证"**，不要说"我参考/模仿了 llama.cpp"
4. **准备好差距解释**：如果性能只有 llama.cpp 的 30%，要能说清"瓶颈在 XX，改进方向是 YY"

### 常见追问模板

| 追问 | 回答要点 |
| ---- | -------- |
| 这个项目哪些是你写的？ | 翻 OWNERSHIP.md，逐模块说明 |
| 为什么不直接用 llama.cpp？ | 学习目的 + 深入理解推理链路 + 手写 kernel 练习 |
| 性能差多少？为什么？ | 给出数据 + 瓶颈归因（如"GEMM 用了 cuBLAS 但 kernel launch 开销大"） |
| 如果要支持 70B 模型？ | Tensor Parallelism + 量化 + PagedAttention（理论储备） |
| FlashAttention 你实现了吗？ | 理解原理 + 有 Python 原型实现，CUDA 版未做（诚实） |

---

## 核心学习资源汇总

### CUDA 编程

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [CUTLASS](https://github.com/NVIDIA/cutlass)
- Mark Harris: Optimizing Parallel Reduction in CUDA（搜索可得 PDF）

### 深度学习系统

- Jay Alammar: The Illustrated Transformer
- Attention Is All You Need 论文
- FlashAttention 论文：Dao et al., 2022
- vLLM 论文：Kwon et al., 2023
- Orca 论文：Yu et al., 2022

### 推理引擎源码

- [llama.cpp](https://github.com/ggerganov/llama.cpp)
- [vLLM](https://github.com/vllm-project/vllm)
- [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)

### 模型编译器（进阶，非必须）

- [TVM](https://tvm.apache.org/docs/)
- [MLIR](https://mlir.llvm.org/docs/)
- [Triton](https://triton-lang.org/main/index.html)

### 性能分析工具

- [Nsight Compute](https://developer.nvidia.com/nsight-compute)
- [Nsight Systems](https://developer.nvidia.com/nsight-systems)
