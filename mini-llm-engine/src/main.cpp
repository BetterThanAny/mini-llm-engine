#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

// Stub entry point — will be fleshed out in W4.
// For now: parses CLI args and prints model config.

#include "model.h"
#include "kv_cache.h"
#include "sampler.h"

static void print_usage(const char* prog) {
    fprintf(stderr,
        "Usage: %s --model <path.gguf> --prompt <text>\n"
        "          [--max_new_tokens N] [--temperature T] [--top_k K] [--top_p P]\n",
        prog);
}

int main(int argc, char** argv) {
    std::string model_path, prompt;
    int   max_new_tokens = 128;
    float temperature    = 0.0f;
    int   top_k          = 1;
    float top_p          = 1.0f;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--model")          && i+1 < argc) { model_path = argv[++i]; }
        else if (!strcmp(argv[i], "--prompt")    && i+1 < argc) { prompt     = argv[++i]; }
        else if (!strcmp(argv[i], "--max_new_tokens") && i+1 < argc) { max_new_tokens = atoi(argv[++i]); }
        else if (!strcmp(argv[i], "--temperature")    && i+1 < argc) { temperature    = atof(argv[++i]); }
        else if (!strcmp(argv[i], "--top_k")          && i+1 < argc) { top_k          = atoi(argv[++i]); }
        else if (!strcmp(argv[i], "--top_p")          && i+1 < argc) { top_p          = atof(argv[++i]); }
        else { print_usage(argv[0]); return 1; }
    }

    if (model_path.empty() || prompt.empty()) {
        print_usage(argv[0]);
        return 1;
    }

    LlamaConfig cfg;  // TinyLlama defaults
    printf("Model: TinyLlama-1.1B  hidden=%d  layers=%d  heads=%d\n",
           cfg.hidden_size, cfg.num_layers, cfg.num_heads);
    printf("Prompt: \"%s\"\n", prompt.c_str());
    printf("max_new_tokens=%d  temperature=%.2f  top_k=%d\n",
           max_new_tokens, temperature, top_k);

    // TODO W4: load_weights → forward_pass → generate loop
    printf("\n[W4 TODO] Inference loop not yet implemented.\n");
    return 0;
}
