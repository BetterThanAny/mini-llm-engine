#!/usr/bin/env python3
"""
dump_weights.py — Export HuggingFace TinyLlama weights to MINILLM binary format.

Binary format (MINILLM v1):
  Header (16 bytes):
    magic[8]:     "MINILLM\0"  (7 printable chars + null byte)
    version:      uint32_t = 1  (little-endian)
    num_tensors:  uint32_t      (little-endian)

  Each tensor record:
    name_len:   uint32_t
    name:       char[name_len]  (no null terminator)
    ndim:       uint32_t
    shape:      int32_t[ndim]
    dtype:      uint32_t  (0=fp32, 1=fp16)
    data:       raw bytes (numel * elem_size)
"""

import argparse
import os
import struct
import sys

import torch
from transformers import AutoModelForCausalLM


DTYPE_FP32 = 0
DTYPE_FP16 = 1

MAGIC = b"MINILLM\x00"  # 7 printable + 1 null = 8 bytes
VERSION = 1


def build_tensor_map(num_layers: int) -> list[tuple[str, str]]:
    """Return ordered list of (binary_name, hf_param_name) pairs."""
    pairs = [
        ("embed_tokens", "model.embed_tokens.weight"),
        ("lm_head",      "lm_head.weight"),
        ("norm",         "model.norm.weight"),
    ]
    for i in range(num_layers):
        pairs += [
            (f"L{i}_q",    f"model.layers.{i}.self_attn.q_proj.weight"),
            (f"L{i}_k",    f"model.layers.{i}.self_attn.k_proj.weight"),
            (f"L{i}_v",    f"model.layers.{i}.self_attn.v_proj.weight"),
            (f"L{i}_o",    f"model.layers.{i}.self_attn.o_proj.weight"),
            (f"L{i}_gate", f"model.layers.{i}.mlp.gate_proj.weight"),
            (f"L{i}_up",   f"model.layers.{i}.mlp.up_proj.weight"),
            (f"L{i}_down", f"model.layers.{i}.mlp.down_proj.weight"),
            (f"L{i}_n1",   f"model.layers.{i}.input_layernorm.weight"),
            (f"L{i}_n2",   f"model.layers.{i}.post_attention_layernorm.weight"),
        ]
    return pairs


def write_tensor(f, name: str, tensor: torch.Tensor, out_dtype: str):
    """Write one tensor record to file f."""
    # Convert dtype
    if out_dtype == "fp16":
        tensor = tensor.to(torch.float16)
        dtype_code = DTYPE_FP16
    else:
        tensor = tensor.to(torch.float32)
        dtype_code = DTYPE_FP32

    tensor = tensor.contiguous()
    shape = list(tensor.shape)
    ndim = len(shape)
    raw_bytes = tensor.numpy().tobytes()

    # name_len + name
    name_bytes = name.encode("utf-8")
    f.write(struct.pack("<I", len(name_bytes)))
    f.write(name_bytes)

    # ndim + shape
    f.write(struct.pack("<I", ndim))
    for s in shape:
        f.write(struct.pack("<i", s))

    # dtype
    f.write(struct.pack("<I", dtype_code))

    # raw data
    f.write(raw_bytes)

    dtype_str = "fp16" if out_dtype == "fp16" else "fp32"
    shape_str = "x".join(str(s) for s in shape)
    print(f"  [{dtype_str}] {name:<20s}  shape={shape_str}")

    return len(raw_bytes)


def main():
    parser = argparse.ArgumentParser(
        description="Export HuggingFace TinyLlama weights to MINILLM binary format."
    )
    parser.add_argument(
        "--model",
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="HuggingFace model name or local path (default: TinyLlama/TinyLlama-1.1B-Chat-v1.0)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output .bin file path",
    )
    parser.add_argument(
        "--dtype",
        choices=["fp32", "fp16"],
        default="fp32",
        help="Output dtype (default: fp32)",
    )
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    model.eval()

    state_dict = model.state_dict()
    print(f"Loaded {len(state_dict)} parameters from HuggingFace checkpoint.")

    # Detect number of layers from state dict
    num_layers = 0
    for key in state_dict:
        if key.startswith("model.layers."):
            idx = int(key.split(".")[2])
            num_layers = max(num_layers, idx + 1)
    print(f"Detected {num_layers} transformer layers.")

    tensor_map = build_tensor_map(num_layers)

    # Validate all expected keys exist
    missing = []
    for bin_name, hf_name in tensor_map:
        if hf_name not in state_dict:
            missing.append(hf_name)
    if missing:
        print(f"ERROR: missing parameters in checkpoint:", file=sys.stderr)
        for m in missing:
            print(f"  {m}", file=sys.stderr)
        sys.exit(1)

    num_tensors = len(tensor_map)
    print(f"\nWriting {num_tensors} tensors to: {args.output}")
    print(f"Output dtype: {args.dtype}\n")

    total_bytes = 0
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    with open(args.output, "wb") as f:
        # Header
        f.write(MAGIC)                                   # 8 bytes
        f.write(struct.pack("<I", VERSION))              # 4 bytes
        f.write(struct.pack("<I", num_tensors))          # 4 bytes

        for bin_name, hf_name in tensor_map:
            tensor = state_dict[hf_name]
            nb = write_tensor(f, bin_name, tensor, args.dtype)
            total_bytes += nb

    total_mb = (os.path.getsize(args.output)) / (1024 * 1024)
    print(f"\nDone. Total file size: {total_mb:.1f} MB")


if __name__ == "__main__":
    main()
