#include "sampler.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <vector>

int sample_greedy(const float* logits, int vocab_size) {
    int best = 0;
    float best_val = logits[0];
    for (int i = 1; i < vocab_size; i++) {
        if (logits[i] > best_val) {
            best_val = logits[i];
            best = i;
        }
    }
    return best;
}

int sample(const float* logits, int vocab_size, const SamplerConfig& cfg) {
    // Greedy shortcuts
    if (cfg.temperature == 0.0f || cfg.top_k == 1) {
        return sample_greedy(logits, vocab_size);
    }

    // Work on a mutable copy
    std::vector<float> probs(logits, logits + vocab_size);

    // 1. Apply temperature
    for (float& v : probs) v /= cfg.temperature;

    // 2. Top-K: zero out all but the top-k logits
    if (cfg.top_k > 0 && cfg.top_k < vocab_size) {
        // Find the k-th largest value via partial sort of indices
        std::vector<int> idx(vocab_size);
        std::iota(idx.begin(), idx.end(), 0);
        std::nth_element(idx.begin(), idx.begin() + cfg.top_k, idx.end(),
                         [&](int a, int b){ return probs[a] > probs[b]; });
        float kth_val = probs[idx[cfg.top_k - 1]];
        // Set values below the k-th threshold to -infinity
        for (float& v : probs) {
            if (v < kth_val) v = -std::numeric_limits<float>::infinity();
        }
    }

    // 3. Softmax
    float max_v = *std::max_element(probs.begin(), probs.end());
    float sum = 0.0f;
    for (float& v : probs) {
        v = std::exp(v - max_v);
        sum += v;
    }
    for (float& v : probs) v /= sum;

    // 4. Top-P (nucleus sampling)
    if (cfg.top_p < 1.0f) {
        // Sort indices by probability descending
        std::vector<int> idx(vocab_size);
        std::iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(),
                  [&](int a, int b){ return probs[a] > probs[b]; });

        // Find cutoff where cumsum exceeds top_p
        float cumsum = 0.0f;
        int cutoff = vocab_size - 1;
        for (int i = 0; i < vocab_size; i++) {
            cumsum += probs[idx[i]];
            if (cumsum > cfg.top_p) {
                cutoff = i;
                break;
            }
        }
        // Zero out tokens beyond cutoff
        for (int i = cutoff + 1; i < vocab_size; i++) {
            probs[idx[i]] = 0.0f;
        }
        // Renormalize
        float new_sum = 0.0f;
        for (float v : probs) new_sum += v;
        if (new_sum > 0.0f) {
            for (float& v : probs) v /= new_sum;
        }
    }

    // 5. Sample from distribution
    static std::mt19937 rng(static_cast<unsigned>(cfg.seed));
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float r = dist(rng);
    float cumsum = 0.0f;
    for (int i = 0; i < vocab_size; i++) {
        cumsum += probs[i];
        if (r <= cumsum) return i;
    }
    // Fallback: return last non-zero token
    for (int i = vocab_size - 1; i >= 0; i--) {
        if (probs[i] > 0.0f) return i;
    }
    return 0;
}
