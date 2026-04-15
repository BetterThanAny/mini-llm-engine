# Week 8 — 面试冲刺 (项目讲解 / 八股 / 投递)

> 本周不再加项目新功能。80% 时间用于面试准备,20% 预留给模拟面试暴露问题后的定向修复 (仅限 bug fix / 数据补充)。

---

## 做了什么

### 1. 面试材料三件套

| 文档 | 路径 | 用途 |
|---|---|---|
| 项目讲解稿 | `docs/interview_script.md` | 10 min 三项目主叙事 + 1 min / 3 min 自我介绍 + 追问预案 |
| 高频问答清单 | `docs/interview_qa.md` | 48 题,覆盖 CUDA / LLM 推理 / 项目追问 / 行为题 |
| 投递跟踪表 | `docs/application_tracker.md` | 投递记录模板 + 目标公司清单 (大厂 + AI 独角兽 + GPU 芯片) + 简历 checklist |

### 2. 面试主题覆盖 (对应 plan.md W8 day 49-52)

- **CUDA**:GPU 层级 / warp / coalesced / bank conflict / occupancy / reduction / softmax / GEMM 优化路径 / FlashAttn 原理
- **LLM 推理**:prefill vs decode / KV cache 公式 / GQA / PagedAttention / Continuous Batching / 量化 W8A16 vs W8A8 / TP
- **项目追问**:OWNERSHIP 逐条、62% 性能差距归因、INT8 反慢分析、扩展 70B 思路、KV OOM 方案
- **手写 kernel 题**:白板版 RMSNorm + online softmax 伪代码已整理到 qa.md Q42/Q43

### 3. 关键项目数据 (从 benchmark.csv / week7.md 汇总,上简历用)

| 指标 | 数值 | 备注 |
|---|---|---|
| 硬件 | RTX 3080 Laptop (10GB) | |
| 模型 | TinyLlama-1.1B FP16 | |
| GPU decode | **91.5 tok/s** | batch=1, flash attn |
| llama.cpp 基线 | 148.1 tok/s | 相同硬件 |
| 相对比 | **62%** | |
| TTFT | 24.2 ms (FP16) | prompt=128 |
| INT8 W8A16 decode | 86.3 tok/s | batch=1 |
| batch=8 aggregate | 93.0 tok/s | |
| 手写 CUDA kernel 数 | 5+ | RMSNorm / RoPE / Softmax / Flash Attn v1 / INT8 fused GEMV |

### 4. 模拟面试

- [ ] 对镜子讲 3 遍 10 min 项目稿 (计时)
- [ ] 找 1-2 位同学/老师做一次完整模拟面试
- [ ] 白板手写 reduction / softmax / RMSNorm 至少 3 次

---

## 产出清单

- `docs/interview_script.md` (项目讲解稿,含自我介绍两版)
- `docs/interview_qa.md` (48 题 Q&A)
- `docs/application_tracker.md` (投递表 + 目标公司清单)
- `mini-llm-engine/docs/week8.md` (本文件)

---

## 问题 / 风险

- **SysY 和 xv6 部分细节需用户本人填充**:interview_script.md 中用 `[填]` 占位的条目,必须在首次面试前回忆/翻旧 repo 补完,否则讲解卡壳
- **面试准备薄弱点** (自评,需持续修复):
  - [ ] `[填入个人薄弱点 1,如:手写 matmul tile 代码不熟]`
  - [ ] `[填入个人薄弱点 2,如:Tensor Parallelism 只懂理论,没跑过]`
  - [ ] `[填入个人薄弱点 3,如:PagedAttention 的 block table 细节]`
- **INT8 反慢的 roofline 解释** 需反复练,容易被追问细节
- **没有公开 demo 视频**:建议补 3-5 min 录屏放简历

---

## 下一步 (W9+)

### 持续投递
- 每周至少新投 5 家,周日复盘转化率
- 优先级:GPU/芯片厂 + AI 独角兽 > 大厂 > 云厂商

### 定向修复 (按模拟面试 + 真实面试反馈)
- 如被问到 CUDA Graph → 补做一个 demo commit
- 如被问到 FP8 / tensor core wmma → 单独 cuda-kernels/ 里写 mini demo
- 如被问到 batched GEMM → 把 `forward_gpu_batched` 真正改成 `cublasGemmStridedBatched`

### 不做的事
- 不加新模型 (不上 7B / Qwen)
- 不做训练反向
- 不做多卡 (纯时间投入,在面试阶段 ROI 低)

---

## Git tag

```
week8-done
```
