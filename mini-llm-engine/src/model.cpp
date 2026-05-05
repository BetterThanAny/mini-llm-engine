// model.cpp — Weight loading and CPU forward pass for TinyLlama-1.1B
// Binary format: MINILLM v1 (see scripts/dump_weights.py for spec)

#include "model.h"
#include "kv_cache.h"
#include "ops_cpu.h"
#include "ops_cuda.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// Binary format constants
// ---------------------------------------------------------------------------
static constexpr char     MAGIC[8]   = {'M','I','N','I','L','L','M','\0'};
static constexpr uint32_t FORMAT_VER = 1;

static constexpr uint32_t DTYPE_FP32 = 0;
static constexpr uint32_t DTYPE_FP16 = 1;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static uint32_t read_u32(FILE* f) {
    uint32_t v = 0;
    if (fread(&v, 4, 1, f) != 1)
        throw std::runtime_error("read_u32: unexpected EOF");
    return v;  // file is little-endian; host is assumed LE (x86/ARM)
}

static int32_t read_i32(FILE* f) {
    int32_t v = 0;
    if (fread(&v, 4, 1, f) != 1)
        throw std::runtime_error("read_i32: unexpected EOF");
    return v;
}

static void validate_kv_cache_capacity(const char* caller,
                                       const KVCache& kv_cache,
                                       int seq_len) {
    if (seq_len <= 0) {
        throw std::runtime_error(std::string(caller) +
                                 ": seq_len must be positive, got " +
                                 std::to_string(seq_len));
    }
    if (kv_cache.cur_len < 0 || kv_cache.max_seq_len < 0) {
        throw std::runtime_error(std::string(caller) +
                                 ": invalid KV cache state cur_len=" +
                                 std::to_string(kv_cache.cur_len) +
                                 " max_seq_len=" +
                                 std::to_string(kv_cache.max_seq_len));
    }
    if (seq_len > kv_cache.max_seq_len - kv_cache.cur_len) {
        throw std::runtime_error(std::string(caller) +
                                 ": KV cache capacity exceeded: cur_len=" +
                                 std::to_string(kv_cache.cur_len) +
                                 " seq_len=" + std::to_string(seq_len) +
                                 " max_seq_len=" +
                                 std::to_string(kv_cache.max_seq_len));
    }
}

// ---------------------------------------------------------------------------
// load_weights
// ---------------------------------------------------------------------------

ModelWeights load_weights(const std::string& path, const LlamaConfig& cfg) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f)
        throw std::runtime_error("load_weights: cannot open " + path);

    // ---- Validate header ----
    char magic[8] = {};
    if (fread(magic, 1, 8, f) != 8)
        throw std::runtime_error("load_weights: cannot read magic");
    if (memcmp(magic, MAGIC, 8) != 0)
        throw std::runtime_error("load_weights: bad magic bytes (not a MINILLM file)");

    uint32_t version     = read_u32(f);
    uint32_t num_tensors = read_u32(f);

    if (version != FORMAT_VER)
        throw std::runtime_error("load_weights: unsupported version " + std::to_string(version));

    printf("[load_weights] MINILLM v%u — %u tensors  file=%s\n",
           version, num_tensors, path.c_str());

    // ---- Pre-allocate ModelWeights skeleton ----
    ModelWeights mw;
    mw.embed_tokens = nullptr;
    mw.lm_head      = nullptr;
    mw.rms_final    = nullptr;
    mw.layers.resize(cfg.num_layers);
    for (auto& lw : mw.layers) {
        lw.attn_q   = nullptr;
        lw.attn_k   = nullptr;
        lw.attn_v   = nullptr;
        lw.attn_o   = nullptr;
        lw.ffn_gate = nullptr;
        lw.ffn_up   = nullptr;
        lw.ffn_down = nullptr;
        lw.rms_attn = nullptr;
        lw.rms_ffn  = nullptr;
    }

    // ---- Read tensors one by one ----
    for (uint32_t t = 0; t < num_tensors; ++t) {
        // name
        uint32_t name_len = read_u32(f);
        std::string name(name_len, '\0');
        if (fread(name.data(), 1, name_len, f) != name_len)
            throw std::runtime_error("load_weights: EOF reading tensor name");

        // shape
        uint32_t ndim = read_u32(f);
        if (ndim == 0 || ndim > 4)
            throw std::runtime_error("load_weights: ndim must be in [1, 4] for tensor " + name);

        int dims[4] = {1, 1, 1, 1};
        size_t numel = 1;
        for (uint32_t d = 0; d < ndim; ++d) {
            dims[d] = read_i32(f);
            if (dims[d] <= 0) {
                throw std::runtime_error(
                    "load_weights: tensor " + name + " has non-positive dim[" +
                    std::to_string(d) + "]=" + std::to_string(dims[d]));
            }
            if (numel > std::numeric_limits<size_t>::max() / static_cast<size_t>(dims[d])) {
                throw std::runtime_error("load_weights: tensor " + name + " shape is too large");
            }
            numel  *= static_cast<size_t>(dims[d]);
        }

        // dtype — always load as FP32 for CPU forward pass (W4)
        uint32_t dtype_code = read_u32(f);
        if (dtype_code != DTYPE_FP32 && dtype_code != DTYPE_FP16) {
            throw std::runtime_error(
                "load_weights: tensor " + name + " has unsupported dtype code " +
                std::to_string(dtype_code) + " (expected 0=fp32 or 1=fp16)");
        }
        size_t   elem_size  = (dtype_code == DTYPE_FP16) ? 2 : 4;
        if (numel > std::numeric_limits<size_t>::max() / elem_size) {
            throw std::runtime_error("load_weights: tensor " + name + " byte size is too large");
        }
        size_t   byte_count = numel * elem_size;

        // Allocate as FP32 CPU tensor
        Tensor* tensor = new Tensor(dims, static_cast<int>(ndim), DType::FP32, Device::CPU);

        if (dtype_code == DTYPE_FP32) {
            // Read directly
            if (fread(tensor->data, 1, byte_count, f) != byte_count)
                throw std::runtime_error("load_weights: EOF reading data for " + name);
        } else {
            // FP16 on disk — convert to FP32
            std::vector<uint16_t> tmp(numel);
            if (fread(tmp.data(), 2, numel, f) != numel)
                throw std::runtime_error("load_weights: EOF reading fp16 data for " + name);
            float* dst = tensor->fp32();
            for (size_t i = 0; i < numel; ++i) {
                // Manual fp16 → fp32 conversion (IEEE 754 layout)
                uint16_t h  = tmp[i];
                uint32_t sign     = (h >> 15) & 1u;
                int      exponent = (h >> 10) & 0x1F;  // signed: denormal loop may go negative
                uint32_t mantissa = h & 0x3FFu;
                uint32_t f32bits;
                if (exponent == 0) {
                    if (mantissa == 0) {
                        f32bits = sign << 31;
                    } else {
                        // Denormal: shift mantissa left until the implicit bit (bit 10) is set,
                        // decrementing exponent each step.  Using signed int prevents uint wraparound
                        // when more than one shift is needed (e.g. mantissa < 0x200).
                        exponent = 1;
                        while (!(mantissa & 0x400)) { mantissa <<= 1; --exponent; }
                        mantissa &= 0x3FF;
                        f32bits = (sign << 31) | ((uint32_t)(exponent + 127 - 15) << 23) | (mantissa << 13);
                    }
                } else if (exponent == 31) {
                    // Inf / NaN
                    f32bits = (sign << 31) | (0xFFu << 23) | (mantissa << 13);
                } else {
                    f32bits = (sign << 31) | ((exponent + 127 - 15) << 23) | (mantissa << 13);
                }
                memcpy(&dst[i], &f32bits, 4);
            }
        }

        // ---- Map name → ModelWeights field ----
        bool mapped = false;

        if (name == "embed_tokens") {
            mw.embed_tokens = tensor; mapped = true;
        } else if (name == "lm_head") {
            mw.lm_head = tensor; mapped = true;
        } else if (name == "norm") {
            mw.rms_final = tensor; mapped = true;
        } else if (name.size() >= 2 && name[0] == 'L') {
            // Parse "L{i}_{suffix}"
            size_t under = name.find('_');
            if (under != std::string::npos) {
                int layer_idx = std::stoi(name.substr(1, under - 1));
                std::string suffix = name.substr(under + 1);
                if (layer_idx >= 0 && layer_idx < cfg.num_layers) {
                    LayerWeights& lw = mw.layers[layer_idx];
                    if      (suffix == "q")    { lw.attn_q   = tensor; mapped = true; }
                    else if (suffix == "k")    { lw.attn_k   = tensor; mapped = true; }
                    else if (suffix == "v")    { lw.attn_v   = tensor; mapped = true; }
                    else if (suffix == "o")    { lw.attn_o   = tensor; mapped = true; }
                    else if (suffix == "gate") { lw.ffn_gate = tensor; mapped = true; }
                    else if (suffix == "up")   { lw.ffn_up   = tensor; mapped = true; }
                    else if (suffix == "down") { lw.ffn_down = tensor; mapped = true; }
                    else if (suffix == "n1")   { lw.rms_attn = tensor; mapped = true; }
                    else if (suffix == "n2")   { lw.rms_ffn  = tensor; mapped = true; }
                }
            }
        }

        if (!mapped) {
            fprintf(stderr, "[load_weights] WARNING: unknown tensor '%s' — skipping\n", name.c_str());
            delete tensor;
            tensor = nullptr;
        }

        // Print progress
        {
            const char* dtype_str = (dtype_code == DTYPE_FP16) ? "fp16→fp32" : "fp32";
            // Build shape string
            char shape_str[64] = {};
            int  off = 0;
            for (uint32_t d = 0; d < ndim; ++d) {
                off += snprintf(shape_str + off, sizeof(shape_str) - off,
                                d == 0 ? "%d" : "x%d", dims[d]);
            }
            printf("  [%u/%u] %-20s  shape=%-18s  dtype=%s\n",
                   t + 1, num_tensors, name.c_str(), shape_str, dtype_str);
        }
    }

    fclose(f);

    // ---- Validate all mandatory pointers are filled -------------------------
    // If any name failed to map, the pointer stays nullptr and forward() would
    // segfault.  Catch it here with a descriptive error instead.
    if (!mw.embed_tokens)
        throw std::runtime_error("load_weights: tensor 'embed_tokens' not found in file");
    if (!mw.lm_head)
        throw std::runtime_error("load_weights: tensor 'lm_head' not found in file");
    if (!mw.rms_final)
        throw std::runtime_error("load_weights: tensor 'norm' not found in file");
    for (int li = 0; li < cfg.num_layers; ++li) {
        const LayerWeights& lw = mw.layers[li];
        const char* missing = nullptr;
        if (!lw.attn_q)   missing = "q";
        else if (!lw.attn_k)   missing = "k";
        else if (!lw.attn_v)   missing = "v";
        else if (!lw.attn_o)   missing = "o";
        else if (!lw.ffn_gate) missing = "gate";
        else if (!lw.ffn_up)   missing = "up";
        else if (!lw.ffn_down) missing = "down";
        else if (!lw.rms_attn) missing = "n1";
        else if (!lw.rms_ffn)  missing = "n2";
        if (missing)
            throw std::runtime_error(
                "load_weights: layer " + std::to_string(li) +
                " missing tensor '" + missing + "'");
    }

    printf("[load_weights] Done.\n");
    return mw;
}

// ---------------------------------------------------------------------------
// free_weights
// ---------------------------------------------------------------------------

void free_weights(ModelWeights& w) {
    delete w.embed_tokens; w.embed_tokens = nullptr;
    delete w.lm_head;      w.lm_head      = nullptr;
    delete w.rms_final;    w.rms_final    = nullptr;

    for (auto& lw : w.layers) {
        delete lw.attn_q;   lw.attn_q   = nullptr;
        delete lw.attn_k;   lw.attn_k   = nullptr;
        delete lw.attn_v;   lw.attn_v   = nullptr;
        delete lw.attn_o;   lw.attn_o   = nullptr;
        delete lw.ffn_gate; lw.ffn_gate = nullptr;
        delete lw.ffn_up;   lw.ffn_up   = nullptr;
        delete lw.ffn_down; lw.ffn_down = nullptr;
        delete lw.rms_attn; lw.rms_attn = nullptr;
        delete lw.rms_ffn;  lw.rms_ffn  = nullptr;
    }
    w.layers.clear();
}

// ---------------------------------------------------------------------------
// forward_cpu
// ---------------------------------------------------------------------------

void forward_cpu(const ModelWeights& w, const LlamaConfig& cfg,
                 KVCache& kv_cache, const int* tokens, int seq_len,
                 int pos_offset, float* logits_out) {
    validate_kv_cache_capacity("forward_cpu", kv_cache, seq_len);

    const int H  = cfg.hidden_size;           // 2048
    const int NH = cfg.num_heads;             // 32
    const int KVH = cfg.num_kv_heads;         // 4
    const int HD  = cfg.head_dim;             // 64
    const int FD  = cfg.ffn_dim;              // 5632
    const int V   = cfg.vocab_size;           // 32000

    // ----------------------------------------------------------------
    // x = embed_tokens[tokens]  →  [seq_len, H]
    // ----------------------------------------------------------------
    std::vector<float> x(seq_len * H);
    {
        const float* emb = w.embed_tokens->fp32();
        for (int t = 0; t < seq_len; ++t)
            memcpy(x.data() + t * H, emb + tokens[t] * H, H * sizeof(float));
    }

    // Scratch buffers — allocated once, reused across layers
    std::vector<float> xn(seq_len * H);           // rms-normed hidden
    std::vector<float> q_buf(seq_len * NH  * HD); // queries
    std::vector<float> k_buf(seq_len * KVH * HD); // keys
    std::vector<float> v_buf(seq_len * KVH * HD); // values
    std::vector<float> attn_out(seq_len * H);      // attention output
    std::vector<float> gate_buf(seq_len * FD);
    std::vector<float> up_buf(seq_len * FD);
    std::vector<float> ffn_out(seq_len * H);
    std::vector<float> residual_add(seq_len * H);  // temp for residual

    // ----------------------------------------------------------------
    // Transformer layers
    // ----------------------------------------------------------------
    for (int i = 0; i < cfg.num_layers; ++i) {
        const LayerWeights& lw = w.layers[i];

        // ---------- Attention pre-norm ----------
        // xn = rms_norm(x, layer.rms_attn)
        rms_norm_cpu(x.data(), lw.rms_attn->fp32(), xn.data(),
                     seq_len, H, cfg.rms_eps);

        // ---------- QKV projections ----------
        // q = xn @ W_q^T   [seq_len, NH*HD]
        matmul_cpu(xn.data(), lw.attn_q->fp32(), q_buf.data(),
                   seq_len, NH * HD, H);
        // k = xn @ W_k^T   [seq_len, KVH*HD]
        matmul_cpu(xn.data(), lw.attn_k->fp32(), k_buf.data(),
                   seq_len, KVH * HD, H);
        // v = xn @ W_v^T   [seq_len, KVH*HD]
        matmul_cpu(xn.data(), lw.attn_v->fp32(), v_buf.data(),
                   seq_len, KVH * HD, H);

        // ---------- RoPE ----------
        rope_cpu(q_buf.data(), k_buf.data(),
                 seq_len, NH, KVH, HD, pos_offset);

        // ---------- KV cache write ----------
        // k_cache[i]: [max_seq_len, KVH, HD]
        float* kc = kv_cache.k_cache[i].fp32() + kv_cache.cur_len * (KVH * HD);
        float* vc = kv_cache.v_cache[i].fp32() + kv_cache.cur_len * (KVH * HD);
        memcpy(kc, k_buf.data(), seq_len * KVH * HD * sizeof(float));
        memcpy(vc, v_buf.data(), seq_len * KVH * HD * sizeof(float));

        int total_kv_len = kv_cache.cur_len + seq_len;  // full context length

        // ---------- Attention ----------
        // Pass full KV cache to attention (from position 0 to total_kv_len)
        attention_cpu(q_buf.data(),
                      kv_cache.k_cache[i].fp32(),
                      kv_cache.v_cache[i].fp32(),
                      attn_out.data(),
                      seq_len, total_kv_len,
                      NH, KVH, HD);

        // attn_proj = attn_out @ W_o^T   [seq_len, H]
        matmul_cpu(attn_out.data(), lw.attn_o->fp32(), residual_add.data(),
                   seq_len, H, H);

        // x = x + attn_proj
        for (int j = 0; j < seq_len * H; ++j)
            x[j] += residual_add[j];

        // ---------- FFN pre-norm ----------
        rms_norm_cpu(x.data(), lw.rms_ffn->fp32(), xn.data(),
                     seq_len, H, cfg.rms_eps);

        // ---------- SwiGLU FFN ----------
        // gate = xn @ W_gate^T   [seq_len, FD]
        matmul_cpu(xn.data(), lw.ffn_gate->fp32(), gate_buf.data(),
                   seq_len, FD, H);
        // up = xn @ W_up^T       [seq_len, FD]
        matmul_cpu(xn.data(), lw.ffn_up->fp32(), up_buf.data(),
                   seq_len, FD, H);

        // gate = silu(gate) * up  (in-place into gate_buf)
        silu_mul_cpu(gate_buf.data(), up_buf.data(), gate_buf.data(), seq_len * FD);

        // ffn = gate @ W_down^T  [seq_len, H]
        matmul_cpu(gate_buf.data(), lw.ffn_down->fp32(), ffn_out.data(),
                   seq_len, H, FD);

        // x = x + ffn
        for (int j = 0; j < seq_len * H; ++j)
            x[j] += ffn_out[j];
    }

    // ----------------------------------------------------------------
    // Final norm on last token only  →  [H]
    // ----------------------------------------------------------------
    std::vector<float> x_last_norm(H);
    // x_last = x[(seq_len-1)*H .. seq_len*H-1]
    rms_norm_cpu(x.data() + (seq_len - 1) * H,
                 w.rms_final->fp32(),
                 x_last_norm.data(),
                 /*rows=*/1, H, cfg.rms_eps);

    // ----------------------------------------------------------------
    // logits = x_last_norm @ lm_head^T   →  [V]
    // ----------------------------------------------------------------
    matmul_cpu(x_last_norm.data(), w.lm_head->fp32(), logits_out,
               /*M=*/1, /*N=*/V, /*K=*/H);

    // ----------------------------------------------------------------
    // Advance KV cache
    // ----------------------------------------------------------------
    kv_cache.cur_len += seq_len;
}

// ---------------------------------------------------------------------------
// weights_to_gpu — upload CPU FP32 weights to GPU as FP16
// ---------------------------------------------------------------------------

static Tensor* fp32_cpu_to_fp16_gpu(const Tensor* src) {
    size_t n = src->numel;
    std::vector<__half> tmp(n);
    const float* sp = src->fp32();
    for (size_t i = 0; i < n; i++) tmp[i] = __float2half(sp[i]);

    int dims[4] = {1, 1, 1, 1};
    for (int d = 0; d < src->ndim; d++) dims[d] = src->shape[d];
    Tensor* dst = new Tensor(dims, src->ndim, DType::FP16, Device::CUDA);
    cudaMemcpy(dst->data, tmp.data(), n * sizeof(__half), cudaMemcpyHostToDevice);
    return dst;
}

ModelWeights weights_to_gpu(const ModelWeights& cpu_w) {
    ModelWeights gpu_w;
    gpu_w.embed_tokens = fp32_cpu_to_fp16_gpu(cpu_w.embed_tokens);
    gpu_w.lm_head      = fp32_cpu_to_fp16_gpu(cpu_w.lm_head);
    gpu_w.rms_final    = fp32_cpu_to_fp16_gpu(cpu_w.rms_final);
    gpu_w.layers.resize(cpu_w.layers.size());
    for (size_t i = 0; i < cpu_w.layers.size(); i++) {
        const LayerWeights& cl = cpu_w.layers[i];
        LayerWeights& gl = gpu_w.layers[i];
        gl.attn_q   = fp32_cpu_to_fp16_gpu(cl.attn_q);
        gl.attn_k   = fp32_cpu_to_fp16_gpu(cl.attn_k);
        gl.attn_v   = fp32_cpu_to_fp16_gpu(cl.attn_v);
        gl.attn_o   = fp32_cpu_to_fp16_gpu(cl.attn_o);
        gl.ffn_gate = fp32_cpu_to_fp16_gpu(cl.ffn_gate);
        gl.ffn_up   = fp32_cpu_to_fp16_gpu(cl.ffn_up);
        gl.ffn_down = fp32_cpu_to_fp16_gpu(cl.ffn_down);
        gl.rms_attn = fp32_cpu_to_fp16_gpu(cl.rms_attn);
        gl.rms_ffn  = fp32_cpu_to_fp16_gpu(cl.rms_ffn);
    }
    printf("[weights_to_gpu] Uploaded %zu layers to GPU as FP16.\n", gpu_w.layers.size());
    return gpu_w;
}

// ---------------------------------------------------------------------------
// weights_to_int8 — per-row symmetric INT8 quantisation of GPU FP16 weights
// ---------------------------------------------------------------------------

static void quantize_tensor(const Tensor* fp16_tensor, Int8Tensor& out) {
    assert(fp16_tensor && fp16_tensor->ndim == 2);
    int rows = fp16_tensor->shape[0];
    int cols = fp16_tensor->shape[1];
    out.alloc(rows, cols);
    quantize_fp16_to_int8_gpu(fp16_tensor->fp16(), out.d_data, out.d_scale, rows, cols);
}

Int8ModelWeights weights_to_int8(const ModelWeights& gpu_fp16_w) {
    printf("[weights_to_int8] Quantising weights to INT8 (W8A16)...\n");
    Int8ModelWeights w8;

    // Embedding + final norm + lm_head: keep FP16 (non-GEMM or accuracy-critical)
    w8.embed_tokens = gpu_fp16_w.embed_tokens;
    w8.lm_head      = gpu_fp16_w.lm_head;
    w8.rms_final    = gpu_fp16_w.rms_final;

    w8.layers.resize(gpu_fp16_w.layers.size());
    for (size_t i = 0; i < gpu_fp16_w.layers.size(); ++i) {
        const LayerWeights& src = gpu_fp16_w.layers[i];
        Int8LayerWeights&   dst = w8.layers[i];

        quantize_tensor(src.attn_q,   dst.attn_q);
        quantize_tensor(src.attn_k,   dst.attn_k);
        quantize_tensor(src.attn_v,   dst.attn_v);
        quantize_tensor(src.attn_o,   dst.attn_o);
        quantize_tensor(src.ffn_gate, dst.ffn_gate);
        quantize_tensor(src.ffn_up,   dst.ffn_up);
        quantize_tensor(src.ffn_down, dst.ffn_down);

        // Norm weights: keep FP16 pointer (owned by gpu_fp16_w)
        dst.rms_attn = src.rms_attn;
        dst.rms_ffn  = src.rms_ffn;

        if ((i + 1) % 5 == 0 || i + 1 == gpu_fp16_w.layers.size())
            printf("  %zu/%zu layers quantised\n", i + 1, gpu_fp16_w.layers.size());
    }

    // Estimate VRAM savings
    size_t fp16_bytes = 0;
    size_t int8_bytes = 0;
    for (const auto& lw : gpu_fp16_w.layers) {
        for (const Tensor* t : {lw.attn_q, lw.attn_k, lw.attn_v, lw.attn_o,
                                 lw.ffn_gate, lw.ffn_up, lw.ffn_down}) {
            if (t) {
                fp16_bytes += t->bytes();
                int8_bytes += t->numel * sizeof(int8_t) + (size_t)t->shape[0] * sizeof(float);
            }
        }
    }
    printf("[weights_to_int8] Done. GEMM weight VRAM: FP16=%.1fMB → INT8=%.1fMB (%.1f%%)\n",
           fp16_bytes / 1e6, int8_bytes / 1e6, 100.0 * int8_bytes / fp16_bytes);
    return w8;
}

void free_int8_weights(Int8ModelWeights& w) {
    // Note: embed_tokens, lm_head, rms_final are shared with the FP16 model — do NOT free.
    for (auto& lw : w.layers) {
        lw.attn_q.free_mem();
        lw.attn_k.free_mem();
        lw.attn_v.free_mem();
        lw.attn_o.free_mem();
        lw.ffn_gate.free_mem();
        lw.ffn_up.free_mem();
        lw.ffn_down.free_mem();
        // rms_attn / rms_ffn are borrowed pointers — do not free
    }
    w.layers.clear();
}

// ---------------------------------------------------------------------------
// forward_gpu — GPU forward pass (FP16)
// ---------------------------------------------------------------------------

static void reshape2d(Tensor& t, int d0, int d1) {
    t.ndim = 2; t.shape[0] = d0; t.shape[1] = d1;
}
static void reshape3d(Tensor& t, int d0, int d1, int d2) {
    t.ndim = 3; t.shape[0] = d0; t.shape[1] = d1; t.shape[2] = d2;
}

void forward_gpu(const ModelWeights& w, const LlamaConfig& cfg,
                 KVCache& kv_cache, const int* tokens, int seq_len,
                 int pos_offset, float* logits_out)
{
    validate_kv_cache_capacity("forward_gpu", kv_cache, seq_len);

    const int H   = cfg.hidden_size;    // 2048
    const int NH  = cfg.num_heads;      // 32
    const int KVH = cfg.num_kv_heads;   // 4
    const int HD  = cfg.head_dim;       // 64
    const int FD  = cfg.ffn_dim;        // 5632
    const int V   = cfg.vocab_size;     // 32000

    // Upload tokens to GPU
    int* d_tokens = nullptr;
    cudaMalloc(&d_tokens, seq_len * sizeof(int));
    cudaMemcpy(d_tokens, tokens, seq_len * sizeof(int), cudaMemcpyHostToDevice);

    // x: [seq_len, H] FP16 CUDA — embedding lookup
    int dims2[2] = {seq_len, H};
    Tensor x(dims2, 2, DType::FP16, Device::CUDA);
    embed_cuda(*w.embed_tokens, d_tokens, x, seq_len);
    cudaFree(d_tokens);

    // Scratch buffers
    int dims_xn[2]   = {seq_len, H};
    int dims_q[2]    = {seq_len, NH * HD};
    int dims_kv[2]   = {seq_len, KVH * HD};
    int dims_attn[2] = {seq_len, H};
    int dims_proj[2] = {seq_len, H};
    int dims_gate[2] = {seq_len, FD};
    int dims_ffn[2]  = {seq_len, H};

    Tensor xn(dims_xn, 2, DType::FP16, Device::CUDA);
    Tensor q_buf(dims_q,  2, DType::FP16, Device::CUDA);
    Tensor k_buf(dims_kv, 2, DType::FP16, Device::CUDA);
    Tensor v_buf(dims_kv, 2, DType::FP16, Device::CUDA);
    Tensor attn_out(dims_attn, 2, DType::FP16, Device::CUDA);
    Tensor proj_out(dims_proj, 2, DType::FP16, Device::CUDA);
    Tensor gate_buf(dims_gate, 2, DType::FP16, Device::CUDA);
    Tensor up_buf(dims_gate,   2, DType::FP16, Device::CUDA);
    Tensor ffn_out(dims_ffn,   2, DType::FP16, Device::CUDA);

    for (int i = 0; i < cfg.num_layers; ++i) {
        const LayerWeights& lw = w.layers[i];

        // ── Attention pre-norm ────────────────────────────────────────────
        reshape2d(xn, seq_len, H);
        rms_norm_cuda(x, *lw.rms_attn, xn, cfg.rms_eps);

        // ── QKV projections ───────────────────────────────────────────────
        // xn [seq_len, H] @ W_q [NH*HD, H]^T → q [seq_len, NH*HD]
        reshape2d(q_buf, seq_len, NH * HD);
        gemm_fp16(xn, *lw.attn_q, q_buf);
        reshape2d(k_buf, seq_len, KVH * HD);
        gemm_fp16(xn, *lw.attn_k, k_buf);
        reshape2d(v_buf, seq_len, KVH * HD);
        gemm_fp16(xn, *lw.attn_v, v_buf);

        // ── RoPE: reshape to 3D first ─────────────────────────────────────
        reshape3d(q_buf, seq_len, NH,  HD);
        reshape3d(k_buf, seq_len, KVH, HD);
        rope_cuda(q_buf, k_buf, pos_offset);

        // ── KV cache write ────────────────────────────────────────────────
        // k_cache[i]: [max_seq_len, KVH, HD] FP16 CUDA
        // write new k/v at row cur_len..cur_len+seq_len
        cudaMemcpy(
            kv_cache.k_cache[i].fp16() + (size_t)kv_cache.cur_len * KVH * HD,
            k_buf.fp16(),
            (size_t)seq_len * KVH * HD * sizeof(__half),
            cudaMemcpyDeviceToDevice);
        cudaMemcpy(
            kv_cache.v_cache[i].fp16() + (size_t)kv_cache.cur_len * KVH * HD,
            v_buf.fp16(),
            (size_t)seq_len * KVH * HD * sizeof(__half),
            cudaMemcpyDeviceToDevice);

        int total_kv_len = kv_cache.cur_len + seq_len;

        // ── Attention (GQA, causal) ────────────────────────────────────────
        reshape3d(attn_out, seq_len, NH, HD);
        attention_cuda(q_buf, kv_cache.k_cache[i], kv_cache.v_cache[i],
                       attn_out, NH, KVH, total_kv_len);

        // ── Attention projection + residual ───────────────────────────────
        reshape2d(attn_out, seq_len, H);
        reshape2d(proj_out, seq_len, H);
        gemm_fp16(attn_out, *lw.attn_o, proj_out);
        add_inplace_cuda(x, proj_out);

        // ── FFN pre-norm ──────────────────────────────────────────────────
        reshape2d(xn, seq_len, H);
        rms_norm_cuda(x, *lw.rms_ffn, xn, cfg.rms_eps);

        // ── SwiGLU FFN ────────────────────────────────────────────────────
        reshape2d(gate_buf, seq_len, FD);
        gemm_fp16(xn, *lw.ffn_gate, gate_buf);
        reshape2d(up_buf, seq_len, FD);
        gemm_fp16(xn, *lw.ffn_up, up_buf);

        silu_cuda(gate_buf);                       // gate = silu(gate)
        mul_cuda(gate_buf, up_buf, gate_buf);      // gate = gate * up

        reshape2d(ffn_out, seq_len, H);
        gemm_fp16(gate_buf, *lw.ffn_down, ffn_out);
        add_inplace_cuda(x, ffn_out);
    }

    // ── Final norm on last token ──────────────────────────────────────────
    int dims_last[2] = {1, H};
    Tensor x_last(dims_last, 2, DType::FP16, Device::CUDA);
    cudaMemcpy(x_last.fp16(),
               x.fp16() + (size_t)(seq_len - 1) * H,
               H * sizeof(__half),
               cudaMemcpyDeviceToDevice);

    Tensor x_last_norm(dims_last, 2, DType::FP16, Device::CUDA);
    rms_norm_cuda(x_last, *w.rms_final, x_last_norm, cfg.rms_eps);

    // ── Logits: [1, H] @ lm_head [V, H]^T → [1, V] ──────────────────────
    int dims_logits[2] = {1, V};
    Tensor logits_gpu(dims_logits, 2, DType::FP16, Device::CUDA);
    gemm_fp16(x_last_norm, *w.lm_head, logits_gpu);

    // Copy logits from GPU FP16 → CPU FP32
    std::vector<__half> logits_h16(V);
    cudaMemcpy(logits_h16.data(), logits_gpu.fp16(),
               V * sizeof(__half), cudaMemcpyDeviceToHost);
    for (int j = 0; j < V; j++)
        logits_out[j] = __half2float(logits_h16[j]);

    // Advance KV cache
    kv_cache.cur_len += seq_len;
}

// ---------------------------------------------------------------------------
// forward_gpu_batched — process batch_size independent sequences
//
// Strategy: serialise over batch dimension — each item runs through the full
// transformer with its own KV cache.  This avoids a full batched-GEMM refactor
// while correctly exercising multi-sequence throughput.  For a production
// implementation, replace the loop with cublasGemmStridedBatched calls.
// ---------------------------------------------------------------------------

void forward_gpu_batched(const ModelWeights& w, const LlamaConfig& cfg,
                         std::vector<KVCache>& kv_caches,
                         const int* tokens,
                         int seq_len, int batch_size,
                         int pos_offset, float* logits_out)
{
    const int V = cfg.vocab_size;

    for (int b = 0; b < batch_size; ++b) {
        // Slice tokens for batch item b
        const int* b_tokens = tokens + b * seq_len;
        // logits for batch item b
        float* b_logits = logits_out + b * V;
        // Each item has its own KV cache
        forward_gpu(w, cfg, kv_caches[b], b_tokens, seq_len, pos_offset, b_logits);
    }
}

// ---------------------------------------------------------------------------
// forward_gpu_int8 — W8A16 forward pass
//
// For decode (seq_len=1): uses fused INT8 GEMV — reads INT8 weights directly,
// no temporary FP16 buffer, ~2x less memory traffic than FP16 GEMV.
//
// For prefill (seq_len>1): dequantises to FP16 buffer then calls cuBLAS GEMM.
// ---------------------------------------------------------------------------

// Helper: dequant Int8Tensor into a pre-allocated FP16 Tensor (prefill path)
static void dequant_into(const Int8Tensor& w, Tensor& buf) {
    buf.shape[0] = w.rows;
    buf.shape[1] = w.cols;
    dequant_int8_to_fp16(w.d_data, w.d_scale, buf.fp16(), w.rows, w.cols);
}

// Helper: fused INT8 matvec for decode (seq_len=1)
// x_in: [1, K] FP16, w: Int8Tensor [N, K], out: [1, N] FP16
static void int8_gemv(const Tensor& x_in, const Int8Tensor& w, Tensor& out) {
    int K = w.cols;
    int N = w.rows;
    gemv_int8_fp16(w.d_data, w.d_scale, x_in.fp16(), out.fp16(), N, K);
}

void forward_gpu_int8(const Int8ModelWeights& w, const LlamaConfig& cfg,
                      KVCache& kv_cache, const int* tokens, int seq_len,
                      int pos_offset, float* logits_out)
{
    validate_kv_cache_capacity("forward_gpu_int8", kv_cache, seq_len);

    const int H   = cfg.hidden_size;
    const int NH  = cfg.num_heads;
    const int KVH = cfg.num_kv_heads;
    const int HD  = cfg.head_dim;
    const int FD  = cfg.ffn_dim;
    const int V   = cfg.vocab_size;

    // Upload tokens
    int* d_tokens = nullptr;
    cudaMalloc(&d_tokens, seq_len * sizeof(int));
    cudaMemcpy(d_tokens, tokens, seq_len * sizeof(int), cudaMemcpyHostToDevice);

    // Embedding lookup (FP16 embed table, no quantisation needed)
    int dims2[2] = {seq_len, H};
    Tensor x(dims2, 2, DType::FP16, Device::CUDA);
    embed_cuda(*w.embed_tokens, d_tokens, x, seq_len);
    cudaFree(d_tokens);

    // Scratch buffers (same as forward_gpu)
    int dims_xn[2]   = {seq_len, H};
    int dims_q[2]    = {seq_len, NH * HD};
    int dims_kv[2]   = {seq_len, KVH * HD};
    int dims_attn[2] = {seq_len, H};
    int dims_proj[2] = {seq_len, H};
    int dims_gate[2] = {seq_len, FD};
    int dims_ffn[2]  = {seq_len, H};

    Tensor xn(dims_xn,   2, DType::FP16, Device::CUDA);
    Tensor q_buf(dims_q,  2, DType::FP16, Device::CUDA);
    Tensor k_buf(dims_kv, 2, DType::FP16, Device::CUDA);
    Tensor v_buf(dims_kv, 2, DType::FP16, Device::CUDA);
    Tensor attn_out(dims_attn, 2, DType::FP16, Device::CUDA);
    Tensor proj_out(dims_proj, 2, DType::FP16, Device::CUDA);
    Tensor gate_buf(dims_gate, 2, DType::FP16, Device::CUDA);
    Tensor up_buf(dims_gate,   2, DType::FP16, Device::CUDA);
    Tensor ffn_out(dims_ffn,   2, DType::FP16, Device::CUDA);

    // Pre-allocate dequantisation buffer — only needed for prefill (seq_len > 1).
    // For decode (seq_len == 1) we use fused INT8 GEMV and skip the buffer entirely.
    int dims_dq[2]  = {FD, H};  // large enough for any weight matrix
    Tensor dq_buf(dims_dq, 2, DType::FP16, Device::CUDA);

    const bool decode_path = (seq_len == 1);  // fused INT8 GEMV for single-token decode

    for (int i = 0; i < cfg.num_layers; ++i) {
        const Int8LayerWeights& lw = w.layers[i];

        // Attention pre-norm
        reshape2d(xn, seq_len, H);
        rms_norm_cuda(x, *lw.rms_attn, xn, cfg.rms_eps);

        // QKV projections
        reshape2d(q_buf, seq_len, NH * HD);
        reshape2d(k_buf, seq_len, KVH * HD);
        reshape2d(v_buf, seq_len, KVH * HD);
        if (decode_path) {
            int8_gemv(xn, lw.attn_q, q_buf);
            int8_gemv(xn, lw.attn_k, k_buf);
            int8_gemv(xn, lw.attn_v, v_buf);
        } else {
            dequant_into(lw.attn_q, dq_buf); gemm_fp16(xn, dq_buf, q_buf);
            dequant_into(lw.attn_k, dq_buf); gemm_fp16(xn, dq_buf, k_buf);
            dequant_into(lw.attn_v, dq_buf); gemm_fp16(xn, dq_buf, v_buf);
        }

        // RoPE
        reshape3d(q_buf, seq_len, NH,  HD);
        reshape3d(k_buf, seq_len, KVH, HD);
        rope_cuda(q_buf, k_buf, pos_offset);

        // KV cache write
        cudaMemcpy(
            kv_cache.k_cache[i].fp16() + (size_t)kv_cache.cur_len * KVH * HD,
            k_buf.fp16(),
            (size_t)seq_len * KVH * HD * sizeof(__half),
            cudaMemcpyDeviceToDevice);
        cudaMemcpy(
            kv_cache.v_cache[i].fp16() + (size_t)kv_cache.cur_len * KVH * HD,
            v_buf.fp16(),
            (size_t)seq_len * KVH * HD * sizeof(__half),
            cudaMemcpyDeviceToDevice);

        int total_kv_len = kv_cache.cur_len + seq_len;

        // Flash Attention
        reshape3d(attn_out, seq_len, NH, HD);
        attention_cuda(q_buf, kv_cache.k_cache[i], kv_cache.v_cache[i],
                       attn_out, NH, KVH, total_kv_len);

        // Attention projection + residual
        reshape2d(attn_out, seq_len, H);
        reshape2d(proj_out, seq_len, H);
        if (decode_path) {
            int8_gemv(attn_out, lw.attn_o, proj_out);
        } else {
            dequant_into(lw.attn_o, dq_buf); gemm_fp16(attn_out, dq_buf, proj_out);
        }
        add_inplace_cuda(x, proj_out);

        // FFN pre-norm
        reshape2d(xn, seq_len, H);
        rms_norm_cuda(x, *lw.rms_ffn, xn, cfg.rms_eps);

        // SwiGLU FFN
        reshape2d(gate_buf, seq_len, FD);
        reshape2d(up_buf, seq_len, FD);
        if (decode_path) {
            int8_gemv(xn, lw.ffn_gate, gate_buf);
            int8_gemv(xn, lw.ffn_up, up_buf);
        } else {
            dequant_into(lw.ffn_gate, dq_buf); gemm_fp16(xn, dq_buf, gate_buf);
            dequant_into(lw.ffn_up, dq_buf); gemm_fp16(xn, dq_buf, up_buf);
        }

        silu_cuda(gate_buf);
        mul_cuda(gate_buf, up_buf, gate_buf);

        reshape2d(ffn_out, seq_len, H);
        if (decode_path) {
            int8_gemv(gate_buf, lw.ffn_down, ffn_out);
        } else {
            dequant_into(lw.ffn_down, dq_buf); gemm_fp16(gate_buf, dq_buf, ffn_out);
        }
        add_inplace_cuda(x, ffn_out);
    }

    // Final norm on last token
    int dims_last[2] = {1, H};
    Tensor x_last(dims_last, 2, DType::FP16, Device::CUDA);
    cudaMemcpy(x_last.fp16(),
               x.fp16() + (size_t)(seq_len - 1) * H,
               H * sizeof(__half),
               cudaMemcpyDeviceToDevice);

    Tensor x_last_norm(dims_last, 2, DType::FP16, Device::CUDA);
    rms_norm_cuda(x_last, *w.rms_final, x_last_norm, cfg.rms_eps);

    // Logits: [1, H] @ lm_head [V, H]^T → [1, V]  (lm_head kept FP16)
    int dims_logits[2] = {1, V};
    Tensor logits_gpu(dims_logits, 2, DType::FP16, Device::CUDA);
    gemm_fp16(x_last_norm, *w.lm_head, logits_gpu);

    // Copy logits GPU FP16 → CPU FP32
    std::vector<__half> logits_h16(V);
    cudaMemcpy(logits_h16.data(), logits_gpu.fp16(),
               V * sizeof(__half), cudaMemcpyDeviceToHost);
    for (int j = 0; j < V; j++)
        logits_out[j] = __half2float(logits_h16[j]);

    kv_cache.cur_len += seq_len;
}
