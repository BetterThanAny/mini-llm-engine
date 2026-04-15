# 面试项目讲解稿 (10 分钟版)

> 目标:AI Infra 岗位面试。三个项目按"工程深度递进"顺序讲:SysY 编译器 → xv6 OS Lab → Mini-LLM-Engine。
> 实际面试按面试官兴趣点展开,本稿为准备用的"完整版"。

---

## 0. 自我介绍

### 1 分钟版

你好,我叫 [姓名],[学校] [专业] [年级]。我的技术背景偏系统方向,本科做过三个有代表性的项目:一个 SysY 到 RISC-V 的编译器,一个基于 xv6 的 OS lab,以及最近花 7 周从零实现的 LLM 推理引擎 Mini-LLM-Engine。Mini-LLM-Engine 用 C++/CUDA 跑 TinyLlama-1.1B,在 RTX 3080 上做到 91.5 tok/s,约为 llama.cpp 基线 (148 tok/s) 的 62%。我对 AI Infra、尤其是推理引擎和 CUDA kernel 优化有强烈兴趣,希望能加入贵司相关团队。

### 3 分钟版

你好,我叫 [姓名],[学校] [专业] [年级]。我的主线是**系统软件 + GPU 编程**。

三个代表项目:

第一是 **SysY 编译器**,本科编译原理课做的完整 toy compiler,前端手写词法/语法分析,中端设计了 SSA 风格的 IR,后端做寄存器分配并生成 RISC-V 汇编。项目让我对数据流分析、IR 设计、指令选择有了第一手经验。

第二是 **xv6 OS Lab**,MIT 6.S081 风格,完成了页表、系统调用、copy-on-write、多线程、文件系统等实验,对虚拟内存、进程、并发原语有较深的机制级理解。这部分对后来理解 PagedAttention 帮助很大。

第三是最近 7 周做的 **Mini-LLM-Engine**,从零用 C++/CUDA 实现 TinyLlama-1.1B 的 FP16 推理,包括手写 RMSNorm/RoPE/Softmax/Flash Attention v1/INT8 GEMV 五个以上 CUDA kernel,设计了 KV Cache、Sampler、W8A16 量化以及 batch 推理。RTX 3080 上 91.5 tok/s,约为 llama.cpp 基线的 62%,差距主要来自 GEMM kernel launch overhead 和未实现 tensor core 融合。

我希望投 AI Infra、推理引擎或 CUDA kernel 工程师岗位。

---

## 1. SysY 编译器 (5-10 min)

### 一句话总览
一个 SysY (C 子集) 到 [目标架构: 填 RISC-V / ARM / x86] 的完整编译器,前中后端独立实现。

### 架构
- **前端**:[手写递归下降 / flex+bison],词法 + 语法 + 语义分析,输出 AST
- **中端 IR**:[IR 形式: 填 三地址码 / SSA / LLVM-like],做了 [优化 pass: 填 常量折叠/死代码消除/GVN/循环优化]
- **后端**:指令选择 + 寄存器分配 + 汇编发射

### 关键设计点

**IR 设计**:`[填入你的 IR 设计:是否 SSA / 基本块组织方式 / 指令粒度]`。选这种 IR 的理由:`[填]`。

**寄存器分配**:`[填入算法:线性扫描 / 图着色 / Chaitin]`。难点在 `[spill 处理 / 预着色寄存器 / live range splitting]`。最终效果:`[跑分数据或稳定性表现]`。

**优化 pass 顺序**:`[填]`。

### 难点 & 解决

1. **[难点 1: 填,如 SSA 构造的 dominance frontier 算法]** — 当时卡在 `[具体点]`,后来通过 `[如何解决]`。
2. **[难点 2: 填,如函数调用的 caller-saved / callee-saved 约定]**。
3. **[难点 3: 填,如数组 + 多维下标的寻址计算]**。

### 一句话收尾
这个项目让我对"从源码到机器码"的链路有机制级理解,后来读 llama.cpp / cuBLAS / Triton 生成的 PTX 时不陌生。

---

## 2. xv6 OS Lab (5-10 min)

### 一句话总览
在 MIT xv6 教学 OS 基础上完成 [Lab 数量: 填] 个实验,覆盖页表、陷入、系统调用、COW、多线程调度、文件系统。

### 做了哪些 Lab
- **Page tables**:`[具体做的事,如实现 vmprint、kvmmap 独立内核页表]`
- **Traps / Syscall**:`[如添加 sigalarm 系统调用、backtrace]`
- **Copy-on-Write fork**:延迟复制物理页,通过引用计数管理释放
- **Multithreading / Lock**:`[如 uthread 用户线程切换、buffer cache 分桶锁减少竞争]`
- **File System**:`[如大文件二级间接块、符号链接]`

### 关键设计点

**虚拟内存**:RISC-V Sv39 三级页表,内核/用户各自 satp,进程切换刷 TLB。`[你实现的具体点: 填]`。

**并发**:spinlock + sleeplock 两类,spinlock 关中断防死锁。`[你修复的并发 bug 或分桶锁细节: 填]`。

**COW fork**:`[讲一下 refcount 在 kfree / usertrap 缺页处理时的一致性维护]`。

### 难点 & 解决
1. **[难点 1: 填,如 COW 下 fork 的 page fault 处理与 race condition]**
2. **[难点 2: 填,如文件系统 log layer 崩溃恢复]**
3. **[难点 3: 填,如 kernel stack 与 trapframe 切换]**

### 与 AI Infra 的关联
- **PagedAttention 与 OS 分页的类比**:vLLM 的 block table ≈ 页表,KV block ≈ 物理页,block size ≈ page size。理解 xv6 页表后,PagedAttention 的 non-contiguous KV 组织就是"直接把 OS 虚存搬到 KV 管理"。
- **多线程调度**:理解上下文切换,迁移到理解 continuous batching 的调度模型不吃力。

---

## 3. Mini-LLM-Engine (10 min)

### 一句话总览
从零用 C++/CUDA 实现的 TinyLlama-1.1B FP16 推理引擎,单卡 RTX 3080。7 周从 naive CPU forward 做到 Flash Attention + INT8 + batch 推理,对标 llama.cpp 做基线验证。

### 架构
```
main.cpp  (CLI + benchmark harness)
   │
   ├── Tokenizer (sentencepiece, 三方库)
   ├── Model (weight loading + forward_gpu / forward_gpu_int8 / forward_gpu_batched)
   ├── Tensor (CPU+GPU 内存管理, FP32/FP16)
   ├── ops_cuda.cu  ← 手写 CUDA kernel (见下)
   ├── cuBLAS (GEMM,三方库)
   ├── KV Cache (预分配 FP16 GPU 缓冲)
   └── Sampler (temperature / top-k / top-p)
```

### 手写的 CUDA Kernel (5 个以上)

| Kernel | 要点 |
|---|---|
| RMSNorm | warp-shuffle reduction 算平方和,单 block 处理一行 |
| RoPE | in-place 对 Q/K 做 2D 旋转,theta = 10000^(-2i/d_head) |
| Softmax | online max + 归一化,数值稳定 |
| Flash Attention v1 | tiled online softmax,~16.9 KB shared memory,causal + GQA,无 O(N²) 中间 buffer |
| INT8 dequant + fused GEMV | W8A16,per-row scale;decode 路径用融合 GEMV kernel 免去 dequant 中间写回 |

### KV Cache 设计
- 预分配 `[num_layers, 2, max_seq_len, num_kv_heads, head_dim]` 的 FP16 GPU 缓冲
- 对 TinyLlama: `2 × 22 × 4 × 2048 × 64 × 2B = 44 MB` (GQA,G=4 而非 32,8× 节省)
- RoPE 在写入 cache 之前对 K 做旋转,cache 里存的就是"位置已编码"的 K,后续无需重算
- batch 推理下每条 sequence 一个独立 cache,batch=8 时总 344 MB,仍远低于 3080 的 10 GB

### 三个演进阶段的性能数据 (RTX 3080, prompt=128, gen=128, median of 5)

| 版本 | TTFT (ms) | Decode (tok/s) | VRAM (MB) |
|---|---|---|---|
| CPU FP32 | ~3470 | ~1.4 | N/A |
| GPU FP16 (Flash Attn) | 24.2 | **91.5** | 7095 |
| GPU INT8 W8A16 | 27.4 | 86.3 | 7590 |
| GPU FP16 batch=8 | 198.5 (aggregate) | 93.0 (aggregate) | 55911 |
| **llama.cpp FP16 基线** | 14.7 | **148.1** | N/A |

相对 llama.cpp 达到 **91.5 / 148.1 ≈ 62%**。

### 差距归因 (诚实)
1. **cuBLAS GEMM 的 launch overhead**:decode 每 step 要 launch 7 个 GEMM,我没做 graph capture 或 kernel fusion;llama.cpp 有 CUDA Graph
2. **未融合的 kernel**:RMSNorm + add 可以融成一个;Q/K/V 的 3 个 projection 可以 concat 成一个大 GEMM
3. **未使用 tensor core 的专用路径**:cuBLAS 已经用了 tensor core,但我没做 CUTLASS 级的 auto-tuning
4. **INT8 反而更慢**:fused GEMV 的 kernel launch 开销在 decode 步中占比过高 (91.5 → 86.3 tok/s);如果走 prefill (batch GEMM) 会快,但 decode 是单行 GEMV,带宽收益没有完全体现

### 正确性验证
- Flash Attention v1: Python 参考实现对比,max abs err < 1e-2 (FP16)
- GPU vs CPU:同 prompt、temperature=0、top_k=1,首 64 token 匹配率 ≥ 95%
- 对比 llama.cpp:同样条件生成,基线验证

### 难点 & 解决

1. **Flash Attention 的 online softmax 与 causal mask**:
   - 难点:分块处理时 rescale 因子 `exp(m_old - m_new)` 要正确传递到累加的 O 矩阵
   - 解决:FP32 accumulation in softmax inner loop,避免 FP16 下 softmax 溢出/NaN
   - 写了纯 Python 参考 + diff 调试

2. **INT8 W8A16 的 fused GEMV kernel**:
   - 难点:dequant 和 GEMV 分开会写回整个 dequantized 权重到 DRAM,带宽浪费
   - 解决:一个 kernel 里每个 warp 负责一行,边读 int8 边乘 scale 边累加到 x 上

3. **WSL2 下 CUDA 设备检测**:[修复 commit 3720c61 记录]

### 扩展性思考
- **70B 模型怎么办?**:需要 Tensor Parallelism (Column/Row parallel + AllReduce),KV cache 跨卡切分;我的 OWNERSHIP 里明确写了 TP 是"有理论储备,未实现"
- **KV Cache OOM?**:PagedAttention + 量化 (KV int8)。PagedAttention 的类比我能用 xv6 页表讲清
- **吞吐进一步提升?**:Continuous batching (Orca) + prefix caching

### 一句话收尾
这个项目的主叙事是"我独立设计实现的推理引擎"。cuBLAS 和 sentencepiece 是工业标准,自己写 FP16 GEMM 或 tokenizer 不是"学 AI Infra"该花的时间。OWNERSHIP.md 里每一行我都能讲清楚。

---

## 常见追问准备 (面试官可能打断)

| 追问 | 预设答案 |
|---|---|
| 为什么不直接用 llama.cpp? | 学习目的:深入理解推理链路 + 手写 kernel 练习,llama.cpp 是基线验证不是模板 |
| 你和 llama.cpp 差多少?为什么? | 62%,瓶颈:GEMM launch overhead + 未做 CUDA Graph + kernel 未融合 |
| 手写过几个 kernel?最得意的? | 5+ 个,最得意 Flash Attn v1 (online softmax + tiling),最有性价比 INT8 fused GEMV |
| FlashAttention v2/v3 实现了吗? | v1 实现了,v2 的 warp-level split-Q 没做 (诚实) |
| KV cache 为什么是 44 MB 不是 352 MB? | GQA,num_kv_heads=4 不是 32,8× 节省 |
| 为什么 INT8 反而慢? | decode 是 GEMV,kernel launch 占比高;prefill 下 INT8 应该快 |
| 如果让你优化一周,你先做什么? | CUDA Graph + Q/K/V fused projection GEMM;预期再提 15-25% |
| 怎么保证正确性? | Flash Attn 有 Python ref,端到端 GPU vs CPU 首 64 token ≥ 95%,再对 llama.cpp |

---

## 节奏控制

- 前两个项目各 2-3 分钟,按面试官兴趣深入
- Mini-LLM-Engine 主战场:架构 1 min → kernel 2 min → 性能 2 min → 归因 + 扩展 2 min → 难点 2 min
- 总时长可压缩到 7-8 min,留出追问时间
