"""
tests/test_flash_attn.py
────────────────────────
Correctness test for Flash Attention v1 (W6 kernel).

Implements both the naive 3-kernel attention and the Flash Attention v1
algorithm in PyTorch, then compares outputs.

This is a pure-Python reference test — it validates the *algorithm* is
correct, independent of the C++/CUDA implementation.  The C++ kernel
follows the same algorithm, so if this passes, the kernel correctness
argument is sound.

Run:
    python tests/test_flash_attn.py

Requirements: torch >= 2.0
"""

import sys
import math

try:
    import torch
except ModuleNotFoundError:
    print("SKIP: PyTorch is not installed")
    sys.exit(77)


# ── Reference implementations ──────────────────────────────────────────────

def naive_attention(Q, K, V, causal=True):
    """Standard 3-step attention (Q@K^T → softmax → @V)."""
    B, NH, S, HD = Q.shape
    scale = 1.0 / math.sqrt(HD)
    scores = torch.matmul(Q, K.transpose(-1, -2)) * scale   # [B, NH, S, kv_S]
    if causal:
        kv_S = K.shape[2]
        q_S  = Q.shape[2]
        kv_offset = kv_S - q_S
        # i-th query row can attend to kv positions 0..kv_offset+i
        mask = torch.ones(q_S, kv_S, device=Q.device, dtype=torch.bool)
        for i in range(q_S):
            mask[i, kv_offset + i + 1:] = False
        scores = scores.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float('-inf'))
    probs  = torch.softmax(scores.float(), dim=-1).to(Q.dtype)
    return torch.matmul(probs, V)


def flash_attn_v1(Q, K, V, Bc=64, causal=True):
    """
    Flash Attention v1 in PyTorch — tiled online softmax.
    Matches the algorithm in src/ops_cuda.cu flash_attn_kernel.

    Q : [B, NH, q_S, HD]
    K : [B, NH, kv_S, HD]   (already GQA-expanded or full)
    V : [B, NH, kv_S, HD]
    Returns output [B, NH, q_S, HD]
    """
    B, NH, q_S, HD = Q.shape
    kv_S = K.shape[2]
    scale = 1.0 / math.sqrt(HD)
    kv_offset = kv_S - q_S

    # Work in FP32 accumulators (mirrors kernel's float registers)
    Q_f = Q.float()
    K_f = K.float()
    V_f = V.float()

    O   = torch.zeros(B, NH, q_S, HD, device=Q.device, dtype=torch.float32)
    m   = torch.full( (B, NH, q_S),    float('-inf'), device=Q.device)
    l   = torch.zeros((B, NH, q_S),                   device=Q.device)

    num_tiles = math.ceil(kv_S / Bc)

    for t in range(num_tiles):
        tile_start = t * Bc
        tile_end   = min(tile_start + Bc, kv_S)

        K_tile = K_f[:, :, tile_start:tile_end, :]   # [B, NH, Bc, HD]
        V_tile = V_f[:, :, tile_start:tile_end, :]

        # Scores: [B, NH, q_S, Bc]
        s = torch.matmul(Q_f, K_tile.transpose(-1, -2)) * scale

        # Causal mask
        if causal:
            for j in range(tile_end - tile_start):
                kvi = tile_start + j
                for qi in range(q_S):
                    if kvi > kv_offset + qi:
                        s[:, :, qi, j] = float('-inf')

        # Online softmax update
        tile_max, _ = s.max(dim=-1)                        # [B, NH, q_S]
        m_new = torch.maximum(m, tile_max)
        alpha = torch.exp(m - m_new)                       # rescale factor
        exp_s = torch.exp(s - m_new.unsqueeze(-1))         # [B, NH, q_S, Bc]

        l = alpha * l + exp_s.sum(dim=-1)
        O = alpha.unsqueeze(-1) * O + torch.matmul(exp_s, V_tile)
        m = m_new

    # Normalise
    O = O / l.unsqueeze(-1)
    return O.to(Q.dtype)


# ── Test cases ──────────────────────────────────────────────────────────────

def run_test(desc, Q, K, V, atol=1e-2):
    """Compare naive vs flash output. Returns True if passed."""
    with torch.no_grad():
        ref = naive_attention(Q, K, V)
        fa  = flash_attn_v1(Q, K, V)

    max_err = (ref - fa).abs().max().item()
    passed  = max_err < atol

    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {desc:<45}  max_abs_err={max_err:.2e}  (tol={atol:.0e})")
    if not passed:
        print(f"           ref range=[{ref.min():.4f}, {ref.max():.4f}]")
        print(f"           fa  range=[{fa.min():.4f}, {fa.max():.4f}]")
    return passed


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype  = torch.float16
    print(f"Flash Attention v1 correctness test  (device={device}, dtype={dtype})")
    print(f"Tolerance: max_abs_error < 1e-2 (FP16 accumulation budget)")
    print()

    all_pass = True

    # TinyLlama config
    NH  = 32
    KVH = 4
    HD  = 64

    def make_qkv(B, q_S, kv_S, seed=0):
        torch.manual_seed(seed)
        Q = torch.randn(B, NH,  q_S,  HD, device=device, dtype=dtype)
        K = torch.randn(B, KVH, kv_S, HD, device=device, dtype=dtype)
        V = torch.randn(B, KVH, kv_S, HD, device=device, dtype=dtype)
        # Expand K/V to match NH (GQA → MHA for reference comparison)
        K = K.repeat_interleave(NH // KVH, dim=1)
        V = V.repeat_interleave(NH // KVH, dim=1)
        return Q, K, V

    # Test 1: decode step (q_len=1)
    Q, K, V = make_qkv(1, q_S=1, kv_S=128)
    all_pass &= run_test("decode  B=1  q=1   kv=128", Q, K, V)

    # Test 2: decode at longer context
    Q, K, V = make_qkv(1, q_S=1, kv_S=512)
    all_pass &= run_test("decode  B=1  q=1   kv=512", Q, K, V)

    # Test 3: decode at max context
    Q, K, V = make_qkv(1, q_S=1, kv_S=2048)
    all_pass &= run_test("decode  B=1  q=1   kv=2048", Q, K, V)

    # Test 4: prefill (q_len == kv_len, pure causal)
    Q, K, V = make_qkv(1, q_S=128, kv_S=128)
    all_pass &= run_test("prefill B=1  q=128 kv=128", Q, K, V)

    # Test 5: kv_len not a multiple of Bc=64
    Q, K, V = make_qkv(1, q_S=1, kv_S=100)
    all_pass &= run_test("decode  B=1  q=1   kv=100 (non-multiple of Bc)", Q, K, V)

    # Test 6: prefill with kv_len not multiple of Bc
    Q, K, V = make_qkv(1, q_S=70, kv_S=70)
    all_pass &= run_test("prefill B=1  q=70  kv=70  (non-multiple)", Q, K, V)

    # Test 7: generation step mid-context (q_len=1, kv_len=300)
    Q, K, V = make_qkv(1, q_S=1, kv_S=300)
    all_pass &= run_test("decode  B=1  q=1   kv=300", Q, K, V)

    print()
    if all_pass:
        print("All tests PASSED.")
        sys.exit(0)
    else:
        print("One or more tests FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    main()
