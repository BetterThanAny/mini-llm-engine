# Week 3: Transformer Inference Internals for TinyLlama-1.1B

> **Purpose**: Interview-ready technical notes. Each section is self-contained for a 5-minute verbal explanation.
>
> **TinyLlama-1.1B config reference** (used throughout):
> - `num_layers = 22`, `num_heads = 32`, `num_kv_heads = 4`, `head_dim = 64`
> - `d_model = 2048` (= num_heads × head_dim = 32 × 64)
> - `d_ffn = 5632` (intermediate dimension of FFN)
> - `vocab_size = 32000`, `max_seq_len = 2048`
> - Dtype: FP16 (2 bytes per element)

---

## 1. Prefill vs Decode: Compute Characterization

### The Core Difference

An LLM autoregressive inference has two distinct phases:

| Phase | Input | Attention op | Compute type |
|-------|-------|--------------|--------------|
| **Prefill** | All `S` prompt tokens simultaneously | Matrix × Matrix | Compute-bound |
| **Decode** | 1 new token at a time | Matrix × Vector | Memory-bound |

### Why Prefill is Compute-Bound (Matrix × Matrix)

During prefill, the `Q`, `K`, `V` projections and the attention score computation operate over the full prompt sequence `S`.

The dominant FLOP cost is `QK^T`:

```
Q: [S, d_model]  x  K^T: [d_model, S]  -->  [S, S]

FLOPs = 2 * S^2 * d_model
```

For TinyLlama, per layer, per head (head_dim = 64):

```
FLOPs(QK^T) = 2 * S^2 * head_dim = 2 * S^2 * 64
```

Over 32 heads and 22 layers with S = 128:

```
= 2 * 128^2 * 64 * 32 * 22
= 2 * 16384 * 64 * 704
= ~1.49 GFLOP  (just attention, not including projections)
```

The key point: FLOPs scale as **O(S^2)**. At long context, arithmetic intensity (FLOPs / bytes moved) is high enough to saturate the GPU's compute units. The GPU's tensor cores are busy.

### Why Decode is Memory-Bound (Matrix × Vector)

During decode, we generate exactly one new token. We need to load all weight matrices to compute one output row.

For a single linear layer `y = W x` where `W` is `[d_out, d_in]`:

```
FLOPs  = 2 * d_out * d_in         (multiply-adds)
Bytes  = 2 * d_out * d_in         (load W in FP16, one byte per weight x 2)
       + 2 * d_in                 (load x)
       + 2 * d_out                (write y)
```

Arithmetic intensity = FLOPs / Bytes ≈ 1 FLOP/byte for large matrices.

RTX 3080 specs:
- FP16 Tensor Core peak: ~119 TFLOP/s
- Memory bandwidth: ~760 GB/s
- Roofline balance point: 119e12 / 760e9 ≈ **156 FLOP/byte**

At decode time, arithmetic intensity ≈ 1 FLOP/byte << 156 FLOP/byte. The GPU is **starved for data**, not compute. The bottleneck is DRAM bandwidth, not arithmetic throughput.

### Summary: Roofline Intuition

```
Prefill (large S):
  Arithmetic Intensity = O(S)  -->  often above roofline  -->  compute-bound

Decode (S=1):
  Arithmetic Intensity ≈ 1 FLOP/byte  -->  far below roofline  -->  memory-bound
```

**Interview answer**: "Prefill processes a full prompt matrix, so QK^T is matrix-matrix and arithmetic intensity is high. Decode generates one token, so every weight matrix is loaded to produce just one output vector — arithmetic intensity collapses to ~1 FLOP/byte, well below the GPU's compute/bandwidth ratio, making it memory-bound."

---

## 2. KV Cache VRAM Formula

### Why KV Cache Exists

During autoregressive decode, at step `t` we need K and V for all previous tokens 1..t. Recomputing them every step would require rerunning all attention projections over the growing context — O(t) work per step, O(t^2) total. The KV cache stores computed K and V tensors so each step is O(1) attention (plus O(t) for the dot product with Q).

### Derivation

For each layer, each KV head, we store a K tensor and a V tensor. At maximum sequence length `max_seq_len = L`:

```
K_cache shape: [L, head_dim]  per KV head per layer
V_cache shape: [L, head_dim]  per KV head per layer
```

Total elements:

```
total_elements = 2              (K and V)
               * num_layers     (one cache per layer)
               * num_kv_heads   (one cache per KV head)
               * max_seq_len    (max tokens we cache)
               * head_dim       (size of each vector)
```

In bytes (FP16 = 2 bytes):

```
KV_cache_bytes = 2 * num_layers * num_kv_heads * max_seq_len * head_dim * sizeof(FP16)
```

### TinyLlama Numbers

Plugging in:

```
= 2 * 22 * 4 * 2048 * 64 * 2

= 2 * 22 * 4 * 2048 * 64 * 2
```

Step by step:

```
num_layers * num_kv_heads = 22 * 4 = 88
88 * max_seq_len = 88 * 2048 = 180,224
180,224 * head_dim = 180,224 * 64 = 11,534,336
11,534,336 * 2 (K+V) = 23,068,672
23,068,672 * 2 bytes (FP16) = 46,137,344 bytes
                             = 46,137,344 / (1024^2) MB
                             ≈ 44 MB
```

**TinyLlama KV cache at full context (2048 tokens): ~44 MB**

For comparison, model weights are ~2.2 GB (1.1B params × 2 bytes). The KV cache is small relative to weights, which is a direct consequence of GQA reducing num_kv_heads from 32 to 4.

### Per-Token Cost

```
KV_per_token = 2 * num_layers * num_kv_heads * head_dim * 2 bytes
             = 2 * 22 * 4 * 64 * 2
             = 22,528 bytes
             ≈ 22 KB per token
```

At 2048 tokens: 22 KB × 2048 = 44 MB. Consistent.

**Interview answer**: "KV cache formula is `2 × L × H_kv × S × d_head × dtype_bytes`. For TinyLlama at max context, that's 2×22×4×2048×64×2 = 44 MB — tiny because GQA reduced KV heads from 32 to 4."

---

## 3. Grouped Query Attention (GQA)

### The Attention Head Taxonomy

| Architecture | Query heads | KV heads | Ratio | Example models |
|---|---|---|---|---|
| **MHA** (Multi-Head Attention) | H | H | 1:1 | GPT-2, original Transformer |
| **MQA** (Multi-Query Attention) | H | 1 | H:1 | PaLM, Falcon |
| **GQA** (Grouped Query Attention) | H | G (1 < G < H) | H/G | TinyLlama, LLaMA-2, Mistral |

TinyLlama: `H = 32` query heads, `G = 4` KV heads. Each KV head serves `32/4 = 8` query heads.

### How GQA Works

In standard attention for one head:

```
Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V
```

In GQA, query head `i` maps to KV head `floor(i / (H/G))`:

```
group_id = i // (H // G)  = i // 8   for TinyLlama

Q_i attends to K_{group_id}, V_{group_id}
```

Groups 0..7 of Q-heads all use the same K_0, V_0. Groups 8..15 use K_1, V_1. Etc.

The K and V projections output only G=4 heads worth of data, not H=32.

### VRAM Savings

KV projection weight sizes:

```
MHA:  W_K, W_V each [d_model, H * d_head]  = [2048, 32 * 64] = [2048, 2048]  --> 2 * 2048 * 2048 * 2 bytes = 16 MB per layer
GQA:  W_K, W_V each [d_model, G * d_head]  = [2048,  4 * 64] = [2048,  256]  --> 2 * 2048 *  256 * 2 bytes =  2 MB per layer
```

Weight savings: 8x reduction in KV projection weights per layer.

KV cache savings: directly proportional — MHA would need 88×8 = 704 MB at full context, GQA needs 44 MB.

### Bandwidth Savings During Decode

During decode, KV cache is loaded from DRAM every step to compute attention. With MHA (H=32):

```
KV bytes loaded per decode step = 32 * 64 * 2 (K or V) * 2 * 22 * 2048 = 352 MB per step
```

With GQA (G=4):

```
KV bytes loaded per decode step = 4 * 64 * 2 * 2 * 22 * 2048 = 44 MB per step
```

8x fewer bytes moved per decode step. Since decode is memory-bound, this directly translates to ~8x lower attention latency per step.

### Accuracy vs Efficiency Tradeoff

GQA (G=4) produces slightly lower quality than MHA (G=32) but significantly better than MQA (G=1). Empirically, for models above ~7B params, GQA at G=8 matches MHA quality closely. For TinyLlama at 1.1B, the tradeoff is acceptable given 8x KV savings.

**Interview answer**: "GQA divides the H=32 query heads into G=4 groups; each group shares one K, V head. For TinyLlama this cuts KV cache from ~352 MB to ~44 MB (8x), and reduces memory bandwidth per decode step by 8x. Since decode is memory-bound, this directly reduces latency. Quality degrades slightly vs MHA but significantly better than MQA (G=1)."

---

## 4. RoPE (Rotary Position Encoding)

### Motivation

Standard learned position embeddings are added to token embeddings before the first layer. This has two problems for inference:
1. They don't generalize well beyond training length.
2. They can't be computed once and cached — they're entangled with content.

RoPE (Su et al., 2021) encodes position as a rotation applied to Q and K vectors in the attention head.

### The Rotation Mechanism

For a query or key vector `x` at position `m`, with head dimension `d_head`, partition `x` into pairs `(x_{2i}, x_{2i+1})` for `i = 0, 1, ..., d_head/2 - 1`.

Apply a 2D rotation to each pair:

```
[x'_{2i}  ]   [cos(m * theta_i)   -sin(m * theta_i)] [x_{2i}  ]
[x'_{2i+1}] = [sin(m * theta_i)    cos(m * theta_i)] [x_{2i+1}]
```

where the base frequencies are:

```
theta_i = base^(-2i / d_head)    (base = 10000 by default)
```

This gives lower frequencies for later dimensions (slower rotation), similar to sinusoidal PE but multiplicative rather than additive.

In compact notation, treating `x` as complex numbers `z_i = x_{2i} + j * x_{2i+1}`:

```
RoPE(z_i, m) = z_i * exp(j * m * theta_i)
```

### How It Encodes Relative Position

The key property: when computing `Q_m dot K_n` (query at position m, key at position n):

```
<RoPE(Q, m), RoPE(K, n)> = <Q, K> * f(m - n)
```

The dot product depends only on the **relative position** `m - n`, not on absolute positions. This emerges because:

```
exp(j*m*theta) * conj(exp(j*n*theta)) = exp(j*(m-n)*theta)
```

So the attention score is a function of relative distance — RoPE gives the model implicit relative position awareness without any learned parameters.

### Applied Per-Head

RoPE is applied independently to each attention head's Q and K vectors before the dot product:

```
for each head h:
    Q_h[m, :] = rope(Q_h[m, :], position=m)   # rotate Q at position m
    K_h[m, :] = rope(K_h[m, :], position=m)   # rotate K at position m
```

V vectors are NOT rotated — only Q and K.

### Why RoPE Is KV-Cache Friendly

This is critical: once `K_h[m, :]` is computed with RoPE applied at position `m`, it can be **stored in the KV cache as-is**. When a new query at position `t > m` attends to it:

```
score = RoPE(Q_t, t) dot RoPE(K_m, m)   <-- RoPE(K_m, m) already stored in cache
```

No recomputation needed. The rotation depends only on the token's own position, not on what other tokens are present. This contrasts with attention-bias approaches (like ALiBi) that may require recomputation.

Compare to absolute sinusoidal PE: sinusoidal PE is added to the embedding before layer 0, so it's baked into the residual stream. KV cache stores the post-PE values just fine. But sinusoidal PE doesn't give relative position information in the attention score, while RoPE does.

### TinyLlama RoPE Parameters

```
base = 10000
d_head = 64  -->  32 rotation pairs per head
theta_i = 10000^(-2i/64)  for i = 0, 1, ..., 31

theta_0  = 1.0          (fastest rotation, full cycle per ~6 positions)
theta_31 = 10000^(-1) = 0.0001  (slowest, cycle every ~62,832 positions)
```

The spread of frequencies allows the model to encode both local (theta_0 sensitive) and long-range (theta_31 sensitive) positional relationships.

**Interview answer**: "RoPE applies a 2D rotation to each pair of Q and K dimensions, parameterized by position m. The rotation matrix uses frequencies theta_i = 10000^(-2i/d_head). Because Q_m dot K_n depends only on m-n (relative position), the model gets relative PE for free. It's KV-cache friendly because K_m is rotated with its own position m and stored — no recomputation needed when future Q tokens attend to it."

---

## 5. SwiGLU Activation

### Standard FFN vs SwiGLU FFN

The standard Transformer FFN:

```
FFN(x) = W_2 * GELU(W_1 * x + b_1) + b_2
```

Two projections: up-projection `W_1: [d_model, d_ffn]`, down-projection `W_2: [d_ffn, d_model]`.

SwiGLU (Shazeer, 2020), used in LLaMA/TinyLlama:

```
FFN_SwiGLU(x) = W_down * (SiLU(W_gate * x) * (W_up * x))
```

Three projections: gate `W_gate: [d_model, d_ffn]`, up `W_up: [d_model, d_ffn]`, down `W_down: [d_ffn, d_model]`.

### The Gate × Up Structure

Two parallel linear projections of the input `x`:

```
gate_out = W_gate * x    [batch, d_ffn]
up_out   = W_up   * x    [batch, d_ffn]
```

Then element-wise:

```
hidden = SiLU(gate_out) * up_out
```

Then down-project:

```
output = W_down * hidden    [batch, d_model]
```

The `gate_out` path goes through SiLU and acts as a **soft gate** that selectively amplifies or suppresses elements of `up_out`.

### SiLU (Sigmoid Linear Unit)

```
SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
```

Properties:
- Smooth, non-monotonic for `x < 0` (has a small negative lobe around x ≈ -1.28)
- Output range: `(-0.28..., +inf)`
- Derivative is smooth everywhere, no "dead neuron" problem like ReLU

Compared to activation functions:

```
ReLU(x)  = max(0, x)             -- non-differentiable at 0, zero gradient for x<0
GELU(x)  ≈ x * Phi(x)           -- smooth, Gaussian-weighted gate
SiLU(x)  = x * sigmoid(x)       -- computationally simpler than GELU, similar properties
```

### Why Three Weight Matrices Instead of Two

Standard FFN: `2 * d_model * d_ffn` parameters (up + down).  
SwiGLU FFN: `3 * d_model * d_ffn` parameters (gate + up + down).

To keep parameter count constant vs standard FFN, models using SwiGLU typically reduce `d_ffn`:

```
d_ffn_swiglu = (2/3) * d_ffn_standard
```

TinyLlama uses `d_ffn = 5632 ≈ (2/3) * 8192`. This keeps compute roughly iso-FLOP compared to a vanilla Transformer.

### TinyLlama FFN Dimensions

```
W_gate: [2048, 5632]   --  2048 * 5632 * 2 bytes = 22.5 MB per layer
W_up:   [2048, 5632]   --  22.5 MB per layer
W_down: [5632, 2048]   --  22.5 MB per layer

FFN weight total per layer: 3 * 22.5 MB = 67.5 MB
Over 22 layers: 22 * 67.5 MB = 1.485 GB  (out of ~2.2 GB total)
```

FFN dominates model size; attention projections (`Q, K, V, O`) add:

```
W_Q: [2048, 2048] = 8 MB
W_K: [2048,  256] = 1 MB  (GQA: 4 heads)
W_V: [2048,  256] = 1 MB
W_O: [2048, 2048] = 8 MB

Attention total per layer: 18 MB
Over 22 layers: 396 MB
```

Sanity check: 1485 + 396 = 1881 MB + embeddings (~128 MB) ≈ 2.0 GB. Close to the expected ~2.2 GB.

### Why SwiGLU Empirically Outperforms ReLU/GELU

The multiplicative gating mechanism allows the network to:
1. **Adaptively suppress features**: `SiLU(gate)` near zero kills a dimension regardless of `up_out`
2. **Preserve gradient flow**: no hard zeros (unlike ReLU), gradients flow even for negative pre-activations
3. **Expressivity**: the gate path and the value path can specialize — the gate learns *when* to use information; the up path learns *what* information

Empirically (Shazeer 2020, LLaMA paper), SwiGLU reduces perplexity by ~0.5-1.0 points over GELU at equal FLOP budget. The improvement is consistent across model scales.

**Interview answer**: "SwiGLU has three weight matrices: gate, up, down. We compute `gate_out = W_gate * x` and `up_out = W_up * x` in parallel, then `hidden = SiLU(gate_out) * up_out`, then `output = W_down * hidden`. SiLU is `x * sigmoid(x)` — smooth, no dead neurons. The multiplicative gate lets the network adaptively suppress dimensions, giving better expressivity than ReLU. To compensate for the extra weight matrix, d_ffn is reduced to 2/3 of standard, keeping FLOP count iso-equivalent."

---

## Quick Reference: TinyLlama-1.1B Numbers

| Parameter | Value |
|---|---|
| `d_model` | 2048 |
| `num_layers` | 22 |
| `num_heads` (Q) | 32 |
| `num_kv_heads` | 4 |
| `head_dim` | 64 |
| `d_ffn` | 5632 |
| `vocab_size` | 32000 |
| `max_seq_len` | 2048 |
| KV cache @ 2048 tokens | ~44 MB |
| Total weights (FP16) | ~2.2 GB |
| FFN weights (FP16, 22L) | ~1.5 GB |
| Attention weights (FP16, 22L) | ~0.4 GB |

---

## Interview Cheat Sheet

| Topic | 1-line answer |
|---|---|
| Prefill bound | Compute-bound: QK^T is matrix×matrix, FLOPs scale as S^2, arithmetic intensity >> bandwidth limit |
| Decode bound | Memory-bound: every weight matrix loaded to produce 1 output vector, ~1 FLOP/byte vs 156 FLOP/byte balance point on RTX 3080 |
| KV cache formula | `2 × L × G × S × d_head × 2 bytes` = 44 MB for TinyLlama |
| GQA savings | 8x KV cache and bandwidth reduction: G=4 vs MHA H=32 |
| RoPE cache-friendly | K stored post-rotation at its position; future Q attends using its own position; no recompute needed |
| SwiGLU vs ReLU | Multiplicative gating + SiLU = soft adaptive feature selection, no dead neurons, ~0.5 ppl better |
