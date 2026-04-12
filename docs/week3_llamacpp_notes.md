# llama.cpp Internals — Technical Notes (Week 3)

**Purpose**: Interview prep / AI Infra deep-dive. Source: llama.cpp public codebase as of early 2025 (primarily `llama.cpp`, `ggml.c`, `ggml-cuda.cu`).

---

## Table of Contents

1. [High-Level Architecture](#1-high-level-architecture)
2. [Main Inference Call Stack](#2-main-inference-call-stack)
3. [KV Cache Implementation](#3-kv-cache-implementation)
4. [Sampling Chain](#4-sampling-chain)
5. [Notable Optimizations](#5-notable-optimizations)
6. [Quick-Reference Cheat Sheet](#6-quick-reference-cheat-sheet)

---

## 1. High-Level Architecture

llama.cpp is a self-contained C/C++ inference engine. Its dependency stack:

```
User / llama-cli
     |
llama.h / llama.cpp   ← model loading, KV cache, batch management, sampling
     |
ggml.h / ggml.c       ← tensor graph construction & CPU execution
ggml-cuda.cu          ← CUDA backend (kernels for matmul, rope, norm, etc.)
ggml-metal.m          ← Metal backend (Apple Silicon)
```

Key design principle: **lazy graph evaluation**. Operations are not executed when called. Instead, `ggml_*` calls build a directed acyclic computation graph (`ggml_cgraph`). The graph is dispatched to a backend (CPU/CUDA/Metal) in one shot via `ggml_graph_compute`.

---

## 2. Main Inference Call Stack

### 2.1 Entry Point: `llama_decode()`

```c
// llama.cpp
int llama_decode(
    struct llama_context * ctx,
    struct llama_batch     batch);
```

`llama_batch` is the public-facing structure for a single forward pass:

```c
struct llama_batch {
    int32_t   n_tokens;      // number of tokens in this batch
    llama_token * token;     // [n_tokens] — NULL for embeddings input
    float   * embd;          // [n_tokens * n_embd] — NULL for token input
    llama_pos  * pos;         // [n_tokens] position indices (for RoPE)
    int32_t  * n_seq_id;     // [n_tokens] how many sequences each token belongs to
    llama_seq_id ** seq_id;  // [n_tokens][n_seq_id[i]]
    int8_t   * logits;       // [n_tokens] — 1 means emit logits for this token
};
```

`llama_decode()` internally calls `llama_decode_internal()`.

### 2.2 `llama_decode_internal()` — the real work

Rough call sequence:

```
llama_decode_internal()
  ├── kv_cache slot allocation (llama_kv_cache_find_slot)
  ├── llama_build_graph()         ← construct the ggml DAG
  │     └── llm_build_llama()    ← model-specific graph builder
  │           ├── token embedding lookup
  │           ├── for each layer:
  │           │   ├── llm_build_norm()      (RMSNorm)
  │           │   ├── llm_build_attn()      (QKV projection, RoPE, KV-cache write, SDPA)
  │           │   └── llm_build_ffn()       (gate/up/down projections, SiLU)
  │           └── final norm + lm_head matmul
  ├── ggml_backend_graph_compute()  ← dispatch to CPU/CUDA/Metal
  └── extract logits from output tensor
```

### 2.3 Graph Construction: `llm_build_llama()`

This function returns a `ggml_cgraph *`. No GPU work happens here — it only records operations as graph nodes.

```c
// Simplified sketch of one transformer layer
struct ggml_tensor * cur;

// Input layernorm (RMSNorm)
cur = llm_build_norm(ctx0, inpL, hparams,
                     model.layers[il].attn_norm, NULL,
                     LLM_NORM_RMS, cb, il);

// QKV projections
struct ggml_tensor * Qcur = ggml_mul_mat(ctx0, model.layers[il].wq, cur);
struct ggml_tensor * Kcur = ggml_mul_mat(ctx0, model.layers[il].wk, cur);
struct ggml_tensor * Vcur = ggml_mul_mat(ctx0, model.layers[il].wv, cur);

// Apply RoPE to Q and K
Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, ...);
Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, ...);

// Write K and V into KV cache (ggml_cpy into cache tensors at cache slot positions)
ggml_build_forward_expand(gf, ggml_cpy(ctx0, Kcur, k_cache_view));
ggml_build_forward_expand(gf, ggml_cpy(ctx0, Vcur, v_cache_view));

// Flash attention or manual SDPA
cur = ggml_flash_attn_ext(ctx0, Qcur, k_cache_view, v_cache_view, ...);

// Output projection
cur = ggml_mul_mat(ctx0, model.layers[il].wo, cur);

// FFN ...
```

### 2.4 Prefill vs. Decode

These are not separate code paths — they use the same `llama_decode()`. The distinction is in the batch:

| Mode    | `n_tokens` | `logits` flags | KV cache behavior |
|---------|-----------|----------------|-------------------|
| Prefill | N (entire prompt) | only last token flagged | writes N new slots |
| Decode  | 1 (new token) | token flagged  | writes 1 new slot, reads all previous |

**Prefill optimization**: when `n_tokens > 1`, the attention mask is a lower-triangular causal mask so all tokens attend to their left context simultaneously. This is a large batched matmul — throughput-bound.

**Decode**: single token attends to the full KV cache — latency-bound, memory-bandwidth-bound.

For CUDA, `ggml_flash_attn_ext` dispatches different kernel tile sizes depending on head count and sequence length. The causal mask is built inside the kernel.

### 2.5 `ggml_backend_graph_compute()`

After graph construction:

```
ggml_backend_graph_compute(backend, graph)
  → ggml_backend_cuda_graph_compute()    (if CUDA backend)
      → for each node in topological order:
            ggml_cuda_compute_forward(node)
              → dispatch to specific CUDA kernel or cuBLAS call
```

For CPU, the graph is executed with a thread pool (`ggml_graph_compute_with_ctx`), parallelized at the tensor operation level.

---

## 3. KV Cache Implementation

### 3.1 Core Structures

```c
// llama.cpp — llama_kv_cache
struct llama_kv_cache {
    bool has_shift = false;  // whether RoPE shift has been applied

    uint32_t head = 0;       // current insertion pointer
    uint32_t size = 0;       // total capacity in cells
    uint32_t used = 0;       // number of non-empty cells

    uint32_t n = 0;          // context window for next compute

    std::vector<llama_kv_cell> cells;  // per-slot metadata

    // The actual K and V tensors — one per layer, stored in a ggml_context
    // Shape: [n_embd_head_k, n_head_kv, n_ctx, n_layer]  (conceptually)
    struct ggml_tensor * k_l[LLAMA_MAX_LAYERS];
    struct ggml_tensor * v_l[LLAMA_MAX_LAYERS];
};

struct llama_kv_cell {
    llama_pos pos   = -1;       // position index of token stored here (-1 = empty)
    llama_pos delta = 0;        // for RoPE shift (sliding window)

    std::set<llama_seq_id> seq_id;  // which sequences own this cell
};
```

Key points:
- K and V are stored **transposed** relative to naive layout. V is often stored as `[n_ctx, n_embd_head_v, n_head_kv, n_layer]` (column-major in the head dimension) for efficient gather during attention.
- All layers share one contiguous VRAM allocation via a single `ggml_backend_buffer`.

### 3.2 Slot Allocation: `llama_kv_cache_find_slot()`

llama.cpp uses a **linear scan with wrap-around** (not a true ring buffer):

```c
bool llama_kv_cache_find_slot(
    struct llama_kv_cache & cache,
    const struct llama_batch & batch) {

    // Count needed contiguous cells
    uint32_t n_tokens = batch.n_tokens;

    // Scan from cache.head looking for n_tokens contiguous free cells
    uint32_t n_tested = 0;
    while (true) {
        if (cache.head + n_tokens > cache.size) {
            cache.head = 0;  // wrap around
        }

        // Check if [head, head+n_tokens) are all free
        bool found = true;
        for (uint32_t i = 0; i < n_tokens; i++) {
            if (cache.cells[cache.head + i].pos >= 0) {
                cache.head += i + 1;
                found = false;
                break;
            }
        }
        if (found) break;

        n_tested += n_tokens;
        if (n_tested >= cache.size) return false;  // cache full
    }

    // Mark cells as occupied
    for (uint32_t i = 0; i < n_tokens; i++) {
        cache.cells[cache.head + i].pos = batch.pos[i];
        for (int s = 0; s < batch.n_seq_id[i]; s++) {
            cache.cells[cache.head + i].seq_id.insert(batch.seq_id[i][s]);
        }
    }
    return true;
}
```

Important properties:
- **Not strictly sequential**: slots can be non-contiguous after some tokens are freed (e.g., after `llama_kv_cache_seq_rm`).
- The `pos` field tracks the *logical* position (used for RoPE), not the physical slot index.
- For simple single-sequence inference, slots are filled contiguously from 0 upward — it behaves like a sequential buffer.

### 3.3 Multi-Sequence / Continuous Batching

For multi-sequence scenarios:
- Each token in the batch has a `seq_id` tag.
- `llama_kv_cache_seq_rm(cache, seq_id, p0, p1)` frees all slots belonging to `seq_id` in position range `[p0, p1)`.
- `llama_kv_cache_seq_cp(cache, seq_id_src, seq_id_dst, p0, p1)` copies sequence metadata (for beam search forking) — it only copies the cell metadata, not the actual K/V tensors. Actual K/V data sharing is done by adding the destination `seq_id` to the cell's `seq_id` set (copy-on-write is NOT implemented — both sequences share the physical slots until one diverges and the engine must allocate new slots).

### 3.4 KV Cache Views (Gather for Attention)

During attention computation, the Q tensor attends only to valid KV slots. This is handled via **ggml_view** operations:

```c
// Build a view of the K cache that covers cells [0, kv_cache.n)
// The actual gather of valid positions is done inside the CUDA flash-attn kernel
// using a "kq_mask" tensor that marks valid vs. invalid positions.
struct ggml_tensor * k_cache_view =
    ggml_view_3d(ctx0, kv_self.k_l[il],
                 n_embd_head_k, n_kv, n_head_kv,
                 /* strides */);
```

The `kq_mask` tensor is a `[n_kv, n_tokens_q]` float tensor where -INF means "don't attend" and 0.0 means "attend". It encodes both the causal mask and the valid-slot mask.

### 3.5 Sliding Window / RoPE Shift

For long contexts beyond `n_ctx`, llama.cpp supports **KV cache shifting**:
- `llama_kv_cache_seq_add(cache, seq_id, p0, p1, delta)` shifts position values of cells by `delta`.
- This enables re-using the cache by evicting the oldest slots and re-assigning their `pos` values, while adjusting RoPE accordingly.
- The `has_shift` flag tells the graph builder to apply a per-slot RoPE correction when reading from cache.

---

## 4. Sampling Chain

### 4.1 Data Structures

```c
// The vocabulary logits for one token position
typedef struct llama_token_data {
    llama_token id;   // token id
    float       logit; // raw logit from lm_head
    float       p;    // probability (after softmax)
} llama_token_data;

typedef struct llama_token_data_array {
    llama_token_data * data;
    size_t             size;
    bool               sorted;  // whether data is sorted by logit descending
} llama_token_data_array;
```

`llama_token_data_array` is the mutable working buffer passed through the sampling pipeline. Each stage modifies it in-place (filtering, re-sorting, computing `.p`).

### 4.2 `llama_sampling_sample()` — High-Level Entry

```c
llama_token llama_sampling_sample(
    struct llama_sampling_context * ctx_sampling,
    struct llama_context           * ctx_main,
    struct llama_context           * ctx_cfg,   // classifier-free guidance ctx (optional)
    int                              idx);
```

This calls `llama_sampling_sample_impl()` which:
1. Extracts logits from `ctx_main` for position `idx`.
2. Optionally applies CFG (blends logits from `ctx_cfg`).
3. Runs the sampler pipeline.
4. Returns the selected `llama_token`.

### 4.3 Sampler Pipeline: Exact Order

The order matters — each stage narrows the candidate set for the next:

```
Raw logits  (vocab_size floats)
     |
[1] llama_sample_repetition_penalties()
     |  penalizes recently generated tokens
     |  modifies: logit[id] /= penalty  (if logit > 0)
     |            logit[id] *= penalty  (if logit < 0)
     |
[2] llama_sample_classifier_free_guidance()   [optional]
     |  logit_guided = logit_main + cfg_scale * (logit_main - logit_negative)
     |
[3] llama_sample_softmax()    [implicit, lazy — called when p values needed]
     |
[4] llama_sample_temp()
     |  logit[i] /= temperature
     |  (temperature = 0 → greedy: just argmax, skip steps 5-7)
     |
[5] llama_sample_top_k()
     |  keep only top-k tokens by logit, discard rest
     |  sort descending, truncate to k
     |
[6] llama_sample_tail_free()  [optional, TFS]
     |  remove tokens in the tail of the second-derivative of sorted probs
     |
[7] llama_sample_typical()    [optional, locally typical sampling]
     |
[8] llama_sample_top_p()
     |  compute cumulative probability of sorted tokens
     |  keep tokens while cumsum < p, discard rest
     |
[9] llama_sample_min_p()      [optional]
     |  discard tokens with p < min_p * p_max
     |
[10] Mirostat [v1 or v2]      [replaces steps 5-9 if enabled]
     |  surprise-based adaptive sampling — adjusts k dynamically
     |  to maintain target perplexity (tau parameter)
     |
[11] llama_sample_token()
      weighted random sample from remaining candidates
      returns llama_token
```

### 4.4 Key Sampler Functions

**Temperature** (`llama_sample_temp`):

```c
void llama_sample_temp(
    struct llama_context * ctx,
    llama_token_data_array * candidates,
    float temp) {
    for (size_t i = 0; i < candidates->size; i++) {
        candidates->data[i].logit /= temp;
    }
}
// Note: temp=0 is handled separately as greedy (argmax) before this pipeline
```

**Top-K** (`llama_sample_top_k`):

```c
void llama_sample_top_k(
    struct llama_context * ctx,
    llama_token_data_array * candidates,
    int k,
    size_t min_keep) {

    k = std::max(k, (int)min_keep);
    k = std::min(k, (int)candidates->size);

    // Partial sort: only sort first k elements
    std::partial_sort(candidates->data, candidates->data + k,
                      candidates->data + candidates->size,
                      [](const llama_token_data & a, const llama_token_data & b) {
                          return a.logit > b.logit;
                      });

    candidates->size = k;
    candidates->sorted = true;
}
```

**Top-P (nucleus sampling)** (`llama_sample_top_p`):

```c
void llama_sample_top_p(
    struct llama_context * ctx,
    llama_token_data_array * candidates,
    float p,
    size_t min_keep) {

    // Requires sorted candidates + computed .p values (softmax applied)
    llama_sample_softmax(ctx, candidates);

    float cumsum = 0.0f;
    size_t last_idx = candidates->size;
    for (size_t i = 0; i < candidates->size; i++) {
        cumsum += candidates->data[i].p;
        if (cumsum >= p && i + 1 >= min_keep) {
            last_idx = i + 1;
            break;
        }
    }
    candidates->size = last_idx;
}
```

**Mirostat v2** (`llama_sample_token_mirostat_v2`):

```c
// Mirostat maintains a running estimate of model surprise (mu)
// tau = target surprise level, eta = learning rate
llama_token llama_sample_token_mirostat_v2(
    struct llama_context * ctx,
    llama_token_data_array * candidates,
    float tau,   // target entropy
    float eta,   // learning rate
    float * mu); // running mu (passed in/out)
// Dynamically truncates vocab to keep surprise near tau
// Updates: *mu -= eta * (observed_surprise - tau)
```

**Final sample** (`llama_sample_token`):

```c
llama_token llama_sample_token(
    struct llama_context * ctx,
    llama_token_data_array * candidates) {
    // Ensure .p is computed
    llama_sample_softmax(ctx, candidates);
    // Weighted random draw using std::discrete_distribution or manual CDF walk
    std::vector<float> probs(candidates->size);
    for (size_t i = 0; i < candidates->size; i++) probs[i] = candidates->data[i].p;
    std::discrete_distribution<int> dist(probs.begin(), probs.end());
    return candidates->data[dist(ctx->rng)].id;
}
```

### 4.5 Grammar Sampling

llama.cpp supports constrained generation via `llama_grammar`:

```c
void llama_sample_grammar(
    struct llama_context * ctx,
    llama_token_data_array * candidates,
    const struct llama_grammar * grammar);
```

This runs **before** temperature/top-k/top-p — it zeroes out logits of tokens that would not produce a valid continuation according to the grammar's PDA state. The grammar is defined as a set of production rules converted to a pushdown automaton.

---

## 5. Notable Optimizations

### 5.1 Quantization

llama.cpp invented the `GGUF`/`GGML` quantization format. Key schemes:

| Format | Bits/weight | Description |
|--------|-------------|-------------|
| Q4_0   | 4.5 bpw     | 32 weights per block, one f16 scale |
| Q4_1   | 5.0 bpw     | 32 weights per block, f16 scale + f16 min |
| Q5_K   | 5.5 bpw     | K-quant: 256-weight super-block with sub-block scales |
| Q6_K   | 6.5 bpw     | K-quant, best quality/size tradeoff |
| Q8_0   | 8.5 bpw     | Near-lossless, fast dequant |
| F16    | 16 bpw      | Half precision, no quantization |

**K-quants** (Q2_K through Q6_K) use a two-level block structure: a super-block of 256 weights contains 8 sub-blocks of 32 weights each. Sub-block scales are themselves quantized using the super-block scale — this reduces overhead from scale storage.

**CUDA dequantization path**: Quantized weights live in VRAM. CUDA kernels dequantize on-the-fly during matmul. For Q4_K matmul, a specialized kernel packs 2 weights per byte and processes them in warps. This avoids storing a full FP16 copy of weights — critical for VRAM budget.

```c
// ggml-cuda.cu — dispatch for quantized matmul
static void ggml_cuda_op_mul_mat(
    ggml_backend_cuda_context & ctx,
    ggml_tensor * dst) {
    // Selects kernel based on src0->type:
    // GGML_TYPE_Q4_K → mul_mat_q4_K_q8_1_cuda(...)
    // GGML_TYPE_F16  → ggml_cuda_op_mul_mat_cublas(...)
    // etc.
}
```

### 5.2 Flash Attention in ggml

`ggml_flash_attn_ext` is a fused attention kernel that avoids materializing the full `[n_q, n_kv]` attention weight matrix:

- Computes attention in tiles: load a tile of Q, iterate over tiles of K/V.
- Uses online softmax (log-sum-exp trick) to accumulate the output incrementally.
- Critical for long contexts: memory goes from O(n²) to O(n * tile_size).
- CUDA implementation in `ggml-cuda/fattn*.cu`. Dispatches to different tile configurations based on head size (64 or 128).

### 5.3 Memory Mapping (`mmap`)

Model weights are loaded via `mmap`:

```c
// llama.cpp — llama_model_loader
llama_mmap * mapping = new llama_mmap(file, prefetch, numa);
// Maps the GGUF file directly into virtual address space
// Weights are demand-paged from disk on first access
// For CPU-only inference, no explicit load needed — OS handles paging
// For GPU inference, weights are copied to VRAM on load
```

Benefits:
- OS-level page cache: if the same model is loaded by multiple processes, physical pages are shared.
- Fast startup: model appears loaded immediately (metadata parsed), weight pages fetched lazily.
- `mlock` can pin pages to prevent swapping during inference.

### 5.4 Batch Handling and Prompt Parallelism

For large prefill batches, llama.cpp splits the batch if it exceeds `n_ubatch` (micro-batch size):

```c
// llama_decode_internal
const int64_t n_ubatch = cparams.n_ubatch;
for (int64_t i = 0; i < n_tokens; i += n_ubatch) {
    // Process tokens [i, min(i + n_ubatch, n_tokens))
    // Each sub-batch runs a full forward pass
    // KV cache is updated incrementally
}
```

This prevents OOM on very long prompts. The downside is that early sub-batches cannot attend to later ones (causality is maintained anyway, but prompt parallelism is limited to the sub-batch).

### 5.5 NUMA Awareness

For CPU inference on multi-socket machines:

```c
struct llama_numa_strategy {
    // LLAMA_NUMA_STRATEGY_DISTRIBUTE: spread threads across NUMA nodes
    // LLAMA_NUMA_STRATEGY_ISOLATE: pin to one NUMA node
    // LLAMA_NUMA_STRATEGY_NUMACTL: use numactl settings
};
```

Thread affinity is set so that threads running on a NUMA node prefer memory allocated on that node, reducing cross-socket bandwidth.

### 5.6 Tensor Parallelism (Experimental)

As of early 2025, llama.cpp has experimental multi-GPU support via `llama_model_params.n_gpu_layers` + `split_mode`:

- `LLAMA_SPLIT_MODE_LAYER`: different transformer layers on different GPUs (pipeline parallelism).
- `LLAMA_SPLIT_MODE_ROW`: weight matrices split row-wise across GPUs (tensor parallelism with `ggml_backend_sched`).

The backend scheduler (`ggml_backend_sched`) routes individual ops to the appropriate device based on which device holds the primary tensor.

### 5.7 Speculative Decoding

`llama-speculative` uses a small draft model to generate candidate tokens, then batch-verifies them with the target model:

```
draft model → [t1, t2, ..., tN] candidate tokens
target model → verify all N+1 positions in one forward pass
              accept tokens where draft matches target distribution
              reject first mismatch, resample from target
```

This is implemented by constructing a batch with `logits=1` for all candidate positions and checking acceptance in post-processing. Achieves 2-3x speedup when draft model is 10-20x smaller.

---

## 6. Quick-Reference Cheat Sheet

| Concept | Key struct/function |
|---------|---------------------|
| Batch input | `llama_batch`, `llama_batch_init()` |
| Run forward pass | `llama_decode()` → `llama_decode_internal()` |
| Build compute graph | `llama_build_graph()` → `llm_build_llama()` |
| Execute graph | `ggml_backend_graph_compute()` |
| KV cache struct | `llama_kv_cache`, `llama_kv_cell` |
| Allocate KV slots | `llama_kv_cache_find_slot()` |
| Free KV slots | `llama_kv_cache_seq_rm()` |
| KV cache shift | `llama_kv_cache_seq_add()` |
| Logit buffer | `llama_token_data_array` |
| Full sampling entry | `llama_sampling_sample()` |
| Temperature | `llama_sample_temp()` |
| Top-K | `llama_sample_top_k()` |
| Top-P (nucleus) | `llama_sample_top_p()` |
| Mirostat | `llama_sample_token_mirostat_v2()` |
| Grammar constraint | `llama_sample_grammar()` |
| Repetition penalty | `llama_sample_repetition_penalties()` |
| Fused attention | `ggml_flash_attn_ext()` |
| Quantized matmul dispatch | `ggml_cuda_op_mul_mat()` |
| Weight mmap | `llama_mmap` |
| Micro-batch split | `n_ubatch` in `llama_cparams` |

---

## Interview Talking Points (10-minute structure)

**~2 min — Overview**: "llama.cpp uses lazy graph evaluation via ggml. You call ggml ops to build a DAG, then dispatch it to CPU/CUDA in one shot. This separates computation description from execution."

**~2 min — Inference loop**: "llama_decode() is called once per step. Prefill and decode use the same code path — the difference is batch size. For prefill, n_tokens is the full prompt; all tokens get processed with a causal mask. For decode, n_tokens=1."

**~2 min — KV cache**: "Each transformer layer has two tensors in VRAM: K cache and V cache of shape [n_ctx, n_head_kv, n_embd_head]. Slots are allocated by linear scan for n_tokens contiguous free cells. Each slot has a pos field for RoPE and a seq_id set for multi-sequence isolation."

**~2 min — Sampling**: "Sampling is a pipeline of in-place transformations on llama_token_data_array. The order is: repetition penalty → temperature → top-K (partial sort, truncate) → top-P (cumsum, truncate) → weighted sample. Temperature=0 short-circuits to argmax."

**~2 min — Optimizations**: "Three big ones: (1) K-quants with on-the-fly CUDA dequantization — keeps weights small in VRAM without a FP16 copy. (2) Flash attention to avoid O(n²) memory. (3) mmap for fast startup and OS-level weight sharing across processes."
