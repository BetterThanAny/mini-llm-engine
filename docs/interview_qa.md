# AI Infra 高频面试问答清单

> 按 W8 day 49-52 的主题组织:CUDA / LLM 推理 / 项目追问。
> 每题:问题 + 回答要点 (bullet,不写长段落)。

---

## 一、CUDA 基础 & GPU 架构

### Q1. 讲一下 GPU 的层级结构
- GPU → 多个 SM (Streaming Multiprocessor)
- SM 内含多个 warp scheduler,每 warp = 32 线程
- 线程层级:grid → block → warp → thread
- 内存层级:register (最快) → shared memory (block 内共享,~100KB) → L1/L2 cache → global memory (最慢,~760 GB/s on 3080)

### Q2. Warp 是什么?为什么是 32?
- 硬件调度的最小单位,一条指令在 32 个 lane 同时发射 (SIMT)
- Warp divergence:if-else 让 warp 内线程走不同分支时串行化,浪费算力
- 编程时尽量让 warp 内线程走同一分支

### Q3. Coalesced memory access
- 一个 warp 的 32 线程访问连续、对齐的 128 字节 → 单次 memory transaction
- 非 coalesced (stride 访问) → 多次 transaction,带宽降几倍
- 写 kernel 时 thread.x 对应访问的最内维度

### Q4. Bank conflict
- Shared memory 分 32 个 bank,每 bank 4 字节
- 同一 warp 的不同线程访问同一 bank 的不同地址 → 串行化
- 常见缓解:padding (如 `shared[32][33]` 避免列访问冲突)

### Q5. Occupancy
- 定义:活跃 warp 数 / 硬件最大 warp 数
- 限制因素:register 用量、shared memory 用量、block size
- 高 occupancy 未必高性能 (高 register 用量的 kernel 低 occupancy 反而快)
- Nsight Compute 看 achieved occupancy

### Q6. CUDA 内存层级的 latency / bandwidth 数量级
- Register:0 cycle
- Shared memory:~30 cycle
- L2 cache:~200 cycle
- Global memory:~500 cycle,760 GB/s 带宽 (3080)
- 优化核心:把数据从 global 搬到 shared 一次,block 内复用

### Q7. CUDA stream 和 event
- Stream:一串串行执行的 kernel/memcpy;不同 stream 间可并发
- Event:同步点,可测时间或跨 stream 同步
- 配合 `cudaMemcpyAsync` + multi-stream 可以 overlap compute 和 H2D/D2H

### Q8. CUDA Graph
- 把多个 kernel launch 录成 graph,一次性 replay
- 省掉 launch overhead,适合 decode 这种 per-step 小 kernel 串
- 我的引擎没做;这是和 llama.cpp 差距的来源之一

---

## 二、CUDA kernel 优化

### Q9. GEMM 优化从 naive 到 register tiling
- **v1 naive**:每线程算一个 C[i,j],全部读 global
- **v2 shared memory tiling**:block 载入 A/B 的 tile 到 shared,block 内复用
- **v3 register tiling**:每线程算 TM×TN 个 C 元素,A/B 从 shared 载入到 register
- **v4 double buffering**:预取下一 tile 与当前计算 overlap
- **v5 tensor core**:`wmma` API,FP16 `mma.sync`,16×16×16 块

### Q10. 手写 parallel reduction (写伪代码或白板)
- v1 naive:`if (tid % (2*stride) == 0) shared[tid] += shared[tid+stride]` — warp divergence 严重
- v2 sequential addressing:`if (tid < stride) shared[tid] += shared[tid+stride]` — 消除 divergence
- v3 first-add-during-load:load 时就做一次加法,block 数减半
- v4 warp shuffle:最后 32 元素用 `__shfl_down_sync` 代替 shared memory
- 我的 reduction 实现:v4 在 16M 元素上达 418 GB/s,vs v1 的 105 GB/s,3.98× 提速

### Q11. Softmax kernel 数值稳定
- 朴素 `exp(x) / sum(exp(x))` 对大 x 溢出
- 稳定版:`exp(x - max) / sum(exp(x - max))`,需先 pass 求 max
- Online softmax:一 pass 同时维护 `(max, sum)`,每遇更大 max 就 rescale sum
- `sum_new = sum_old * exp(max_old - max_new) + exp(x - max_new)`

### Q12. RMSNorm vs LayerNorm
- LayerNorm:减均值、除标准差,两次 reduction
- RMSNorm:`x / sqrt(mean(x^2)) * weight`,只一次 reduction,少算均值
- LLaMA 系列用 RMSNorm,速度略快,精度等效
- 我的实现:warp shuffle reduce 平方和,block 处理一行

### Q13. Flash Attention 为什么快?
- 朴素 attention 要写回 [N,N] 的 attention matrix 到 HBM,再读回来做 softmax
- Flash Attn:tile Q/K/V 到 shared memory,online softmax 分块累加 O,**从不写回 [N,N]**
- IO-aware:算力已过剩,瓶颈是 HBM 带宽;减少 HBM 读写 = 加速
- 复杂度仍 O(N²) 但常数小

### Q14. FlashAttention v1 vs v2 vs v3
- v1:外层 K/V tile,内层 Q tile;有 Q 的重复加载
- v2:外层 Q tile,减少 HBM 读;warp-level 并行优化
- v3:Hopper 架构 (H100) 的异步 copy + FP8 支持

### Q15. 怎么 profile CUDA kernel?
- Nsight Systems:timeline 粒度,看 kernel launch 间隔、memcpy overlap
- Nsight Compute:kernel 内部,看 occupancy、memory throughput、warp stall reason
- 常看指标:SM utilization、DRAM throughput、L2 hit rate

---

## 三、LLM 推理系统

### Q16. 画 Transformer 前向数据流
- Input tokens → embedding `[B, S, d_model]`
- N 层 decoder block:
  - RMSNorm → Q/K/V projection → RoPE → attention → O projection → residual
  - RMSNorm → gate/up projection → SiLU(gate)*up → down projection → residual
- 最后 RMSNorm → LM head `[B, S, vocab_size]` → sampling

### Q17. Prefill vs Decode 特征对比
- **Prefill**:输入整个 prompt,QK^T 是 matrix × matrix,arithmetic intensity ∝ S,compute-bound
- **Decode**:每次 1 token,weight matrix × vector,arithmetic intensity ≈ 1 FLOP/byte,memory-bound
- 3080 roofline ridge point ≈ 156 FLOP/byte,decode 远低于

### Q18. KV Cache 原理 & 显存公式
- 避免每步重算所有历史 K/V:O(t²) → O(t)
- 公式:`2 × L × H_kv × S × d_head × dtype_bytes`
- TinyLlama:`2 × 22 × 4 × 2048 × 64 × 2 = 44 MB`
- 每 token ≈ 22 KB

### Q19. MHA / MQA / GQA 对比
- MHA:H 个 Q head,H 个 KV head (原始 Transformer)
- MQA:H 个 Q,1 个 KV (PaLM/Falcon);KV 省 H×
- GQA:H 个 Q,G 个 KV (LLaMA-2/TinyLlama);KV 省 H/G×
- TinyLlama H=32, G=4,KV 省 8×,质量接近 MHA

### Q20. FlashAttention 为什么减少 HBM 读写就加速?
- 对现代 GPU:compute 算力远超 HBM 带宽 (3080: 29.8 TF / 760 GB)
- Attention 朴素实现是 memory-bound 的 (写回 [N,N] 再读)
- 减少 HBM 读 = 直接提速
- 即使 FLOPs 不变也能快 2-4×

### Q21. PagedAttention 原理
- 类比 OS 虚拟内存:KV cache 按 block 切分,block table 维护逻辑→物理映射
- 每 block 如 16 token,non-contiguous 分配
- 解决问题:连续分配下的内碎片 (预留 max_seq_len 但用不到)、不同请求间 block 共享 (prefix caching)
- 我用 xv6 页表类比讲

### Q22. Continuous Batching vs Static Batching
- Static batching:一 batch 中所有请求必须同 step 同 length,等最长的
- Continuous batching (Orca):每 step 动态加入新请求 / 移除完成请求
- GPU 利用率从 ~20% 提到 ~70%+
- 代价:调度复杂、KV cache 管理需 PagedAttention 配合

### Q23. 量化方法对比
- FP16:基线,2 byte/weight
- INT8 (W8A16):weight 量化为 int8,activation 保持 FP16;1 byte/weight + scale
  - 对称 per-row:`scale[r] = max_abs(W[r]) / 127`
- INT4 / GPTQ / AWQ:权重 int4,groupwise scale;准确度要配 calibration
- FP8 (H100):E4M3/E5M2,硬件原生,训练推理都能用

### Q24. Weight-only vs Weight-Activation 量化
- W8A16 / W4A16:只量化权重,decode 带宽减半,计算仍 FP16
  - 好处:精度损失小,decode 是 memory-bound 直接受益
- W8A8:权重 + 激活都 int8,利用 INT tensor core
  - 好处:prefill/训练受益多;坏处:activation outlier 难处理 (SmoothQuant)

### Q25. Tensor Parallelism 原理
- Column parallel:`Y = X W`,切 W 的列,每卡出一段 Y,无需通信
- Row parallel:`Y = X W`,切 W 的行,每卡出部分和,AllReduce 累加
- Transformer 典型组合:QKV projection (column) → attention → O projection (row) → AllReduce
- FFN:W_gate/W_up (column) → activation → W_down (row) → AllReduce
- 每层 2 次 AllReduce,通信量 = `batch × seq × d_model × 2`

### Q26. Pipeline Parallelism vs Tensor Parallelism
- PP:按层切,卡间传 activation;bubble 问题,需 1F1B / interleaved 调度
- TP:层内切,卡间 AllReduce;延迟敏感,要 NVLink
- 70B+ 一般 TP (卡内) + PP (跨节点) 混合

### Q27. Speculative decoding
- 小模型 (draft) 生成 k 个 token,大模型并行验证
- 命中率高则一次出多个 token,decode 的 memory-bound 瓶颈被摊薄
- 变体:Medusa (多头并行 draft)、EAGLE

### Q28. vLLM 的核心贡献
- PagedAttention:KV cache 按 block 管理
- Continuous batching
- Prefix caching:相同 prefix 的请求共享 KV block
- 开源社区事实标准

### Q29. SGLang / TensorRT-LLM 特点
- SGLang:RadixAttention (前缀树式 KV 共享),适合 agent / multi-turn
- TensorRT-LLM:NV 官方,CUDA kernel 高度融合,in-flight batching,闭源性能天花板

### Q30. Chunked prefill
- 长 prompt 的 prefill 一次算会阻塞 decode 请求
- 切成 chunk,每 step 混入少量 prefill token + 多个 decode 请求
- 提升 TTFT 波动和整体吞吐

---

## 四、项目追问

### Q31. 你这项目哪些是自己写的?
- 翻 OWNERSHIP.md 逐模块说:Tensor、KV Cache、Sampler、Weight Loader、CPU forward、GPU forward、INT8 forward、batch forward、benchmark harness、5 个以上 CUDA kernel (RMSNorm/RoPE/Softmax/Flash Attn v1/INT8 dequant/INT8 fused GEMV)
- 三方:cuBLAS (FP16 GEMM) + sentencepiece (tokenizer)

### Q32. 为什么不直接用 llama.cpp / vLLM?
- 学习目的:从零理解推理全链路
- 手写 kernel 的练习价值在"从 naive 到 tiled"的迭代过程
- llama.cpp 是基线验证,不是模板

### Q33. 为什么 GEMM 用 cuBLAS 不自己写?
- 手写 FP16 GEMM (tensor core + double buffering + split-K) 是几周工作,ROI 低
- 项目重点是"推理引擎架构" + "kernel 多样性",不是 GEMM 专项
- CUTLASS-level GEMM 是单独课题

### Q34. 性能和 llama.cpp 差多少?瓶颈在哪?
- 91.5 vs 148.1 tok/s,约 62%
- 瓶颈:
  1. 未做 CUDA Graph:每 step 7 次 GEMM launch + kernel 间隙
  2. 未做 kernel fusion:RMSNorm+add、Q/K/V projection 都可融
  3. 未做 tensor core 专用路径调优 (cuBLAS 默认 heuristic 未必最优)

### Q35. 优化一周,你先做什么?
- CUDA Graph capture decode loop (预期 +15%)
- Q/K/V fused projection (三个 [d,d] GEMM → 一个 [d, 3d] GEMM,+5-10%)
- FP16 accumulation 改为 BF16 或探索 FP8 (3080 不支持,只是理论)

### Q36. KV Cache 显存不够怎么办?
- 方法 1:PagedAttention,按 block 分配消除碎片
- 方法 2:KV 量化 (KV int8,H2O/StreamingLLM 等)
- 方法 3:窗口注意力 / sink token
- 方法 4:卸载到 CPU (FlexGen 思路)

### Q37. 扩展到 70B 怎么办?
- 单卡装不下,必须 TP
- 典型:TP=8 + FP16,或 TP=4 + INT4
- KV cache 也切,每卡只存 H_kv/TP 个 head
- 需要 NCCL AllReduce,架构上 layer norm/residual 在 TP rank 内即可

### Q38. INT8 反而更慢,不正常吧?
- 91.5 → 86.3 tok/s 确实反直觉
- 分析:decode 是 GEMV,fused GEMV kernel launch overhead + 单行工作量不足以 amortize
- Prefill (batch GEMM) 下 INT8 会快;我当前 benchmark 全是短 prompt,prefill 占比小
- Roofline 理论预期 1.8×,实测 0.94×,差距在 kernel 实现

### Q39. Flash Attention 你自己实现的?v 几?
- v1,带 causal mask + GQA
- online softmax 算法理解到位,FP32 accumulation 避免 FP16 softmax NaN
- v2 没做 (warp-level split-Q 未实现),v3 不具备硬件

### Q40. batch_size 大了但每流 tok/s 骤降?
- 实测:bs=1 → 100.1/stream,bs=8 → 11.9/stream
- 原因:我的 batch 是"循环每 item 跑 forward_gpu",没有 batched GEMM
- 生产级应改 `cublasGemmStridedBatched`,每 item 并行走 attention heads,消除 serialize
- 架构上 `forward_gpu_batched` API 已预留,未来升级不需改调用方

### Q41. 正确性怎么验证?
- 单 kernel:Python 参考 (PyTorch) 对比,FP16 max abs err < 1e-2,FP32 < 1e-5
- 端到端:GPU vs CPU 同 prompt + temp=0 + top_k=1,首 64 token ≥ 95% 匹配
- 外部基线:llama.cpp 同配置对照

### Q42. 如果面试要你白板写 RMSNorm kernel?
- `__shared__ float s_sum;`
- 每线程累加部分平方和
- block 内 warp-shuffle reduction `__shfl_down_sync`
- thread 0 写 shared,__syncthreads
- 广播 `rsqrt(mean+eps)` 到所有线程
- 每线程乘 weight 写回

### Q43. 如果面试要你白板写 online softmax?
```
m = -inf, l = 0
for each x_i:
    m_new = max(m, x_i)
    l = l * exp(m - m_new) + exp(x_i - m_new)
    m = m_new
output: exp(x_i - m) / l
```

### Q44. 做这项目最大的收获?
- 从 roofline 视角看 kernel:不是 FLOPs 少就快,是 bytes moved 少才快
- 推理引擎的瓶颈图谱:prefill 是 compute-bound (GEMM) + decode 是 memory-bound (KV + weight)
- 系统设计的"自底向上 vs 自顶向下"平衡:先让 CPU 跑通,再逐层换 GPU kernel

### Q45. 最大的坑 / bug?
- 参考 commit 94319db:cuBLAS alpha/beta 搞错导致 GPU 输出不对;RoPE 的 theta 索引;BOS token 处理
- 参考 commit 3720c61:WSL2 下 CUDA device detection
- 教训:先对 CPU,再对 GPU;layer-by-layer 比对中间 tensor

---

## 五、行为题 / 开放题

### Q46. 你觉得 AI Infra 未来 3 年方向?
- 推理侧:长上下文 (1M+) 的高效架构,KV 压缩/稀疏/共享
- 训练侧:FP8 主流化、MoE infra 成熟
- 硬件:国产芯片软件栈,CUDA 替代 (TPU/NPU/GPU 国产)
- 算子编译器:Triton / TVM / MLIR 的融合

### Q47. 你为什么想做 AI Infra 而不是算法?
- 系统背景 (编译器 + OS) 更匹配 infra
- 喜欢确定性的性能数据 (roofline / tok/s) 多于调参
- [填入个人理由]

### Q48. 你对加班 / 晋升有什么看法?
- [填入个人立场]

---

## 附:快速检索表 (自测用)

| 主题 | 关键公式 / 数字 |
|---|---|
| 3080 HBM 带宽 | 760 GB/s |
| 3080 FP16 TC 峰值 | 119 TFLOPS |
| 3080 Roofline ridge | 156 FLOP/byte |
| TinyLlama KV cache | 2×22×4×2048×64×2B = 44 MB |
| TinyLlama 总权重 FP16 | ~2.2 GB |
| GQA KV 节省 | 8× (H=32, G=4) |
| Decode arithmetic intensity | ~1 FLOP/byte |
| 我的引擎 vs llama.cpp | 91.5 / 148.1 = 62% |
