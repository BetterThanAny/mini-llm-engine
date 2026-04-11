#pragma once
#include <vector>
#include <cstdint>

struct SamplerConfig {
    float temperature = 1.0f;
    int   top_k       = 50;
    float top_p       = 0.9f;
    int   seed        = 42;
};

// Sample the next token from logits.
// logits: raw unnormalized scores, shape [vocab_size]
int sample(const float* logits, int vocab_size, const SamplerConfig& cfg);

// Greedy: argmax (temperature=0 shortcut)
int sample_greedy(const float* logits, int vocab_size);
