#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>

#include <sentencepiece_processor.h>

#include <cuda_runtime.h>

#include "model.h"
#include "kv_cache.h"
#include "sampler.h"

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static void print_usage(const char* prog) {
    fprintf(stderr,
        "Usage: %s --model <path> --tokenizer <path> --prompt <text>\n"
        "          [--max_new_tokens N]   (default 128)\n"
        "          [--temperature T]      (default 0.0)\n"
        "          [--top_k K]            (default 1)\n"
        "          [--top_p P]            (default 1.0)\n"
        "          [--benchmark]          (print timing stats)\n"
        "          [--warmup N]           (warmup runs, default 3)\n"
        "          [--runs N]             (measure runs, default 5)\n"
        "          [--output_csv <path>]  (append timing to CSV)\n"
        "          [--device cpu|cuda]    (default cpu)\n"
        "          [--quant int8]         (W8A16 INT8 weight quant, requires --device cuda)\n"
        "          [--batch_size N]       (batch inference, N <= 8, requires --device cuda)\n",
        prog);
}

using Clock     = std::chrono::high_resolution_clock;
using TimePoint = std::chrono::time_point<Clock>;

static double ms_since(const TimePoint& start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

static float median(std::vector<float>& v) {
    std::sort(v.begin(), v.end());
    int n = static_cast<int>(v.size());
    return (n % 2 == 0) ? (v[n/2 - 1] + v[n/2]) / 2.0f : v[n/2];
}

// ---------------------------------------------------------------------------
// One full inference run: prefill + decode.
// Returns {ttft_ms, decode_tps, generated_token_count, output_text}.
// ---------------------------------------------------------------------------
struct RunResult {
    double ttft_ms;
    double decode_tps;
    int    gen_tokens;
    std::string output_text;
};

static RunResult run_inference(
    const ModelWeights& w,
    const LlamaConfig&  cfg,
    sentencepiece::SentencePieceProcessor& sp,
    const std::vector<int>& input_ids,
    int   max_new_tokens,
    const SamplerConfig& sampler_cfg,
    bool  verbose,
    bool  use_gpu        = false,
    const ModelWeights*     gpu_w   = nullptr,
    const Int8ModelWeights* int8_w  = nullptr)
{
    std::vector<float> logits(cfg.vocab_size);
    KVCache kv_cache(cfg, use_gpu ? Device::CUDA : Device::CPU);

    int prompt_len = static_cast<int>(input_ids.size());

    // Helper: dispatch to the correct forward function
    auto forward = [&](const int* toks, int len, int pos) {
        if (use_gpu && int8_w)
            forward_gpu_int8(*int8_w, cfg, kv_cache, toks, len, pos, logits.data());
        else if (use_gpu && gpu_w)
            forward_gpu(*gpu_w, cfg, kv_cache, toks, len, pos, logits.data());
        else
            forward_cpu(w, cfg, kv_cache, toks, len, pos, logits.data());
    };

    // --- Prefill (TTFT) ---
    TimePoint t0 = Clock::now();
    forward(input_ids.data(), prompt_len, /*pos_offset=*/0);
    int first_token = sample(logits.data(), cfg.vocab_size, sampler_cfg);
    double ttft_ms = ms_since(t0);

    if (verbose) {
        printf("[Prefill] TTFT = %.2f ms  (prompt_len=%d)\n", ttft_ms, prompt_len);
    }

    // --- Decode loop ---
    std::vector<int> output_ids;
    output_ids.push_back(first_token);

    constexpr int EOS_ID = 2;  // TinyLlama uses token id 2 as </s>

    TimePoint decode_start = Clock::now();
    for (int step = 0; step < max_new_tokens - 1; step++) {
        int cur_token = output_ids.back();
        if (cur_token == EOS_ID) break;

        int pos = prompt_len + step;
        forward(&cur_token, /*seq_len=*/1, pos);
        int next_token = sample(logits.data(), cfg.vocab_size, sampler_cfg);
        output_ids.push_back(next_token);

        if (next_token == EOS_ID) break;
    }
    double decode_elapsed_ms = ms_since(decode_start);
    int decode_tokens = static_cast<int>(output_ids.size()) - 1;  // exclude first_token (already counted)
    double decode_tps = (decode_elapsed_ms > 0.0 && decode_tokens > 0)
                        ? (decode_tokens * 1000.0 / decode_elapsed_ms)
                        : 0.0;

    // Decode token ids → text
    std::string output_text;
    sp.Decode(output_ids, &output_text);

    return {ttft_ms, decode_tps, static_cast<int>(output_ids.size()), output_text};
}

// ---------------------------------------------------------------------------
// Batched inference (batch_size > 1, CUDA only).
// Runs the same prompt for every item in the batch; reports aggregate tok/s.
// ---------------------------------------------------------------------------
struct BatchResult {
    double ttft_ms;         // time until first token (all batch items)
    double aggregate_tps;   // total generated tokens / decode time (all items)
    int    gen_tokens_each;
};

static BatchResult run_inference_batched(
    const ModelWeights& gpu_w,
    const LlamaConfig&  cfg,
    const std::vector<int>& input_ids,
    int   max_new_tokens,
    int   batch_size,
    const SamplerConfig& sampler_cfg,
    bool  verbose)
{
    const int prompt_len = static_cast<int>(input_ids.size());
    const int V          = cfg.vocab_size;

    // Build batch token buffer: same prompt repeated batch_size times
    std::vector<int> batch_tokens(batch_size * prompt_len);
    for (int b = 0; b < batch_size; ++b)
        for (int t = 0; t < prompt_len; ++t)
            batch_tokens[b * prompt_len + t] = input_ids[t];

    // One KVCache per batch item
    std::vector<KVCache> kv_caches;
    kv_caches.reserve(batch_size);
    for (int b = 0; b < batch_size; ++b)
        kv_caches.emplace_back(cfg, Device::CUDA);

    std::vector<float> logits(batch_size * V);
    std::vector<int>   cur_tokens(batch_size);

    // --- Prefill ---
    TimePoint t0 = Clock::now();
    forward_gpu_batched(gpu_w, cfg, kv_caches, batch_tokens.data(),
                        prompt_len, batch_size, /*pos_offset=*/0, logits.data());
    for (int b = 0; b < batch_size; ++b)
        cur_tokens[b] = sample(logits.data() + b * V, V, sampler_cfg);
    double ttft_ms = ms_since(t0);

    if (verbose)
        printf("[Prefill] TTFT = %.2f ms  (batch=%d  prompt_len=%d)\n",
               ttft_ms, batch_size, prompt_len);

    // --- Decode ---
    constexpr int EOS_ID = 2;
    std::vector<bool> done(batch_size, false);
    int gen_tokens = 1;  // first token from prefill

    TimePoint decode_start = Clock::now();
    for (int step = 0; step < max_new_tokens - 1; ++step) {
        // Check if all done
        bool all_done = true;
        for (int b = 0; b < batch_size; ++b)
            if (!done[b]) { all_done = false; break; }
        if (all_done) break;

        int pos = prompt_len + step;

        // Decode step: each item produces its cur_token
        for (int b = 0; b < batch_size; ++b) {
            if (done[b]) continue;
            forward_gpu(gpu_w, cfg, kv_caches[b], &cur_tokens[b],
                        /*seq_len=*/1, pos, logits.data() + b * V);
            cur_tokens[b] = sample(logits.data() + b * V, V, sampler_cfg);
            if (cur_tokens[b] == EOS_ID) done[b] = true;
        }
        ++gen_tokens;
    }

    double decode_ms = ms_since(decode_start);
    int total_decode_tokens = (gen_tokens - 1) * batch_size;
    double agg_tps = (decode_ms > 0 && total_decode_tokens > 0)
                     ? total_decode_tokens * 1000.0 / decode_ms
                     : 0.0;

    return {ttft_ms, agg_tps, gen_tokens};
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    // --- Argument parsing ---
    std::string model_path, tokenizer_path, prompt, output_csv;
    std::string device_str   = "cpu";
    std::string quant_str    = "";     // "" = fp16, "int8" = W8A16
    int   max_new_tokens = 128;
    float temperature    = 0.0f;
    int   top_k          = 1;
    float top_p          = 1.0f;
    bool  do_benchmark   = false;
    int   warmup_runs    = 3;
    int   measure_runs   = 5;
    int   seed           = 42;
    int   batch_size     = 1;

    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "--model")          && i+1 < argc) { model_path      = argv[++i]; }
        else if (!strcmp(argv[i], "--tokenizer")      && i+1 < argc) { tokenizer_path  = argv[++i]; }
        else if (!strcmp(argv[i], "--prompt")         && i+1 < argc) { prompt          = argv[++i]; }
        else if (!strcmp(argv[i], "--max_new_tokens") && i+1 < argc) { max_new_tokens  = atoi(argv[++i]); }
        else if (!strcmp(argv[i], "--temperature")    && i+1 < argc) { temperature     = static_cast<float>(atof(argv[++i])); }
        else if (!strcmp(argv[i], "--top_k")          && i+1 < argc) { top_k           = atoi(argv[++i]); }
        else if (!strcmp(argv[i], "--top_p")          && i+1 < argc) { top_p           = static_cast<float>(atof(argv[++i])); }
        else if (!strcmp(argv[i], "--seed")           && i+1 < argc) { seed            = atoi(argv[++i]); }
        else if (!strcmp(argv[i], "--benchmark"))                     { do_benchmark    = true; }
        else if (!strcmp(argv[i], "--warmup")         && i+1 < argc) { warmup_runs     = atoi(argv[++i]); }
        else if (!strcmp(argv[i], "--runs")           && i+1 < argc) { measure_runs    = atoi(argv[++i]); }
        else if (!strcmp(argv[i], "--output_csv")     && i+1 < argc) { output_csv      = argv[++i]; }
        else if (!strcmp(argv[i], "--device")         && i+1 < argc) { device_str      = argv[++i]; }
        else if (!strcmp(argv[i], "--quant")          && i+1 < argc) { quant_str       = argv[++i]; }
        else if (!strcmp(argv[i], "--batch_size")     && i+1 < argc) { batch_size      = atoi(argv[++i]); }
        else { print_usage(argv[0]); return 1; }
    }

    if (model_path.empty() || tokenizer_path.empty() || prompt.empty()) {
        fprintf(stderr, "Error: --model, --tokenizer, and --prompt are required.\n");
        print_usage(argv[0]);
        return 1;
    }

    // --- Load tokenizer ---
    sentencepiece::SentencePieceProcessor sp;
    auto status = sp.Load(tokenizer_path);
    if (!status.ok()) {
        fprintf(stderr, "Error loading tokenizer '%s': %s\n",
                tokenizer_path.c_str(), status.ToString().c_str());
        return 1;
    }

    // --- Tokenize prompt ---
    std::vector<int> input_ids;
    sp.Encode(prompt, &input_ids);
    printf("Prompt: \"%s\"\n", prompt.c_str());
    printf("Prompt tokens: %d\n", static_cast<int>(input_ids.size()));

    // --- Load model ---
    LlamaConfig cfg;
    printf("Model: TinyLlama-1.1B  hidden=%d  layers=%d  heads=%d  kv_heads=%d\n",
           cfg.hidden_size, cfg.num_layers, cfg.num_heads, cfg.num_kv_heads);
    ModelWeights w = load_weights(model_path, cfg);

    // --- GPU setup ---
    bool use_gpu = (device_str == "cuda");
    if (use_gpu) {
        int n = 0;
        cudaError_t err = cudaGetDeviceCount(&n);
        if (err != cudaSuccess || n == 0) {
            fprintf(stderr, "Warning: --device cuda requested but no CUDA device found (n=%d, err=%d: %s); falling back to CPU.\n",
                    n, (int)err, cudaGetErrorString(err));
            use_gpu = false;
        } else {
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, 0);
            printf("[CUDA] Device 0: %s  (compute %d.%d)\n", prop.name, prop.major, prop.minor);
        }
    }
    bool use_int8 = (quant_str == "int8");
    // INT8 batch inference is not implemented; warn and downgrade.
    if (use_int8 && batch_size > 1) {
        fprintf(stderr, "Warning: --quant int8 is not supported with --batch_size > 1 "
                        "(no batched INT8 path); falling back to FP16 for batch inference.\n");
        use_int8 = false;
    }
    ModelWeights    gpu_w{};
    Int8ModelWeights int8_w{};
    if (use_gpu) {
        if (use_int8) {
            printf("Device: CUDA (W8A16 INT8 weight quantisation)\n");
            gpu_w  = weights_to_gpu(w);
            int8_w = weights_to_int8(gpu_w);
        } else {
            printf("Device: CUDA (FP16)\n");
            gpu_w = weights_to_gpu(w);
        }
    } else {
        printf("Device: CPU (FP32)\n");
    }

    // --- Sampler config ---
    SamplerConfig sampler_cfg;
    sampler_cfg.temperature = temperature;
    sampler_cfg.top_k       = top_k;
    sampler_cfg.top_p       = top_p;
    sampler_cfg.seed        = seed;

    if (batch_size > 1 && !use_gpu) {
        fprintf(stderr, "Error: --batch_size > 1 requires --device cuda\n");
        return 1;
    }
    if (batch_size < 1 || batch_size > 8) {
        fprintf(stderr, "Error: --batch_size must be in [1, 8]\n");
        return 1;
    }

    if (!do_benchmark) {
        if (batch_size > 1) {
            // --- Batched single run (no text output — timing only) ---
            BatchResult br = run_inference_batched(gpu_w, cfg, input_ids, max_new_tokens,
                                                   batch_size, sampler_cfg, /*verbose=*/true);
            printf("\n=== Batch Timing (batch_size=%d) ===\n", batch_size);
            printf("  TTFT:               %.2f ms\n",  br.ttft_ms);
            printf("  Aggregate tok/s:    %.1f  (all %d streams)\n",
                   br.aggregate_tps, batch_size);
            printf("  Per-stream tok/s:   %.1f\n",
                   br.aggregate_tps / batch_size);
            printf("  Gen tokens/stream:  %d\n", br.gen_tokens_each);
        } else {
        // --- Single inference run ---
        RunResult result = run_inference(w, cfg, sp, input_ids, max_new_tokens, sampler_cfg,
                                         /*verbose=*/true, use_gpu,
                                         use_gpu ? &gpu_w : nullptr,
                                         (use_gpu && use_int8) ? &int8_w : nullptr);

        printf("\n=== Generated Text ===\n%s\n", result.output_text.c_str());
        printf("\n=== Timing ===\n");
        printf("  TTFT:        %.2f ms\n",  result.ttft_ms);
        printf("  Decode:      %.1f tok/s\n", result.decode_tps);
        printf("  Gen tokens:  %d\n", result.gen_tokens);
        }
    } else {
        // --- Benchmark mode ---
        printf("\n[Benchmark] warmup=%d  runs=%d  batch_size=%d\n",
               warmup_runs, measure_runs, batch_size);

        const Int8ModelWeights* p_int8 = (use_gpu && use_int8) ? &int8_w : nullptr;

        // Warmup
        for (int i = 0; i < warmup_runs; i++) {
            printf("  Warmup %d/%d ...\n", i+1, warmup_runs);
            if (batch_size > 1)
                run_inference_batched(gpu_w, cfg, input_ids, max_new_tokens,
                                      batch_size, sampler_cfg, false);
            else
                run_inference(w, cfg, sp, input_ids, max_new_tokens, sampler_cfg,
                              false, use_gpu, use_gpu ? &gpu_w : nullptr, p_int8);
        }

        // Measure
        std::vector<float> ttft_samples, tps_samples;
        int   last_gen_tokens = 0;
        double total_start_ms = 0.0;
        TimePoint bench_start = Clock::now();

        for (int i = 0; i < measure_runs; i++) {
            printf("  Run %d/%d ...\n", i+1, measure_runs);
            if (batch_size > 1) {
                BatchResult br = run_inference_batched(gpu_w, cfg, input_ids, max_new_tokens,
                                                       batch_size, sampler_cfg, false);
                ttft_samples.push_back(static_cast<float>(br.ttft_ms));
                tps_samples.push_back(static_cast<float>(br.aggregate_tps));
                last_gen_tokens = br.gen_tokens_each;
            } else {
                RunResult r = run_inference(w, cfg, sp, input_ids, max_new_tokens, sampler_cfg,
                                            false, use_gpu, use_gpu ? &gpu_w : nullptr, p_int8);
                ttft_samples.push_back(static_cast<float>(r.ttft_ms));
                tps_samples.push_back(static_cast<float>(r.decode_tps));
                last_gen_tokens = r.gen_tokens;
            }
        }
        double total_elapsed_ms = ms_since(bench_start);
        (void)total_start_ms;

        float median_ttft = median(ttft_samples);
        float median_tps  = median(tps_samples);

        printf("\n=== Benchmark Results (median of %d runs) ===\n", measure_runs);
        printf("  TTFT (ms):       %.2f\n", median_ttft);
        if (batch_size > 1) {
            printf("  Aggregate tok/s: %.1f  (batch=%d)\n", median_tps, batch_size);
            printf("  Per-stream tok/s:%.1f\n", median_tps / batch_size);
        } else {
            printf("  Decode (tok/s):  %.1f\n", median_tps);
        }
        printf("  Gen tokens:      %d\n",   last_gen_tokens);
        printf("  Total wall (ms): %.0f\n", total_elapsed_ms);

        // --- CSV output ---
        if (!output_csv.empty()) {
            bool write_header = false;
            // Check if file is new/empty
            {
                std::ifstream check(output_csv);
                write_header = !check.good() || check.peek() == std::ifstream::traits_type::eof();
            }
            std::ofstream csv(output_csv, std::ios::app);
            if (!csv.is_open()) {
                fprintf(stderr, "Warning: could not open CSV file '%s'\n", output_csv.c_str());
            } else {
                if (write_header) {
                    csv << "timestamp,engine,prompt_tokens,gen_tokens,ttft_ms,decode_tps,total_ms\n";
                }
                // Timestamp: ISO-8601
                std::time_t now_t = std::time(nullptr);
                char ts[32];
                std::strftime(ts, sizeof(ts), "%Y-%m-%dT%H:%M:%S", std::localtime(&now_t));
                csv << ts << ","
                    << (batch_size > 1
                        ? ("mini-llm-engine-cuda-bs" + std::to_string(batch_size))
                        : (use_gpu ? (use_int8 ? "mini-llm-engine-cuda-int8"
                                               : "mini-llm-engine-cuda")
                                   : "mini-llm-engine-cpu")) << ","
                    << input_ids.size() << ","
                    << last_gen_tokens << ","
                    << median_ttft << ","
                    << median_tps << ","
                    << total_elapsed_ms << "\n";
                printf("Timing written to: %s\n", output_csv.c_str());
            }
        }
    }

    if (use_gpu && use_int8) free_int8_weights(int8_w);
    if (use_gpu) free_weights(gpu_w);
    free_weights(w);
    return 0;
}
