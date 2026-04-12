"""
Sampling strategies for LLM token generation.

Supports:
- Greedy (temperature=0 or top_k=1)
- Temperature scaling
- Top-K filtering
- Top-P (nucleus) filtering
"""

import torch
import torch.nn.functional as F


def sample_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
) -> int:
    """
    Sample a single token from logits.

    Args:
        logits: [vocab_size] raw logits (unnormalized)
        temperature: 0 = greedy, >0 = softmax temperature
        top_k: keep only top-k tokens (0 = disabled)
        top_p: nucleus sampling threshold (1.0 = disabled)

    Returns:
        token id (int)
    """
    # Greedy decoding
    if temperature == 0.0 or top_k == 1:
        return int(logits.argmax())

    # Temperature scaling
    logits = logits.float()
    logits = logits / temperature

    # Top-K filtering: zero out all but top-k logits
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        values, _ = torch.topk(logits, top_k)
        min_val = values[-1]
        logits = logits.masked_fill(logits < min_val, float('-inf'))

    # Top-P (nucleus) filtering
    if top_p < 1.0:
        probs = F.softmax(logits, dim=-1)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # Remove tokens that push cumulative prob over top_p
        # Shift right so we always keep at least one token
        sorted_indices_to_remove = cumulative_probs - sorted_probs > top_p
        sorted_probs[sorted_indices_to_remove] = 0.0
        sorted_probs /= sorted_probs.sum()

        token = int(torch.multinomial(sorted_probs, 1))
        return int(sorted_indices[token])

    # Standard sampling from softmax
    probs = F.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, 1))
