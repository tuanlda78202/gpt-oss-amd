"""
dequantize_to_bf16.py
=====================

Create a **bfloat16 (bf16)** copy of a local GPT-OSS checkpoint and save it in
safe-serialized format (`safetensors`). This is useful when your original model
directory contains fp32/fp16 or quantized weights (e.g., 4/8-bit) and you want
a uniform bf16 checkpoint for consistent inference across toolchains that
expect/perform best with bf16 tensors.

The script:
  1) Resolves source/destination paths from a model *size* (20b or 120b),
     or lets you override them explicitly.
  2) Loads the tokenizer and model from `src` with `dtype=torch.bfloat16`
     (casting/dequantizing at load as applicable).
  3) Saves the tokenizer and model to `dst`, using `safe_serialization=True`
     so weights are written as `.safetensors`.
"""

import argparse
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def paths_for_size(size: str):
    size = size.lower()
    if size not in {"20b", "120b"}:
        raise ValueError("size must be one of: 20b, 120b")
    models_root = os.environ.get("MODELS_ROOT")
    src = os.path.join(models_root, f"gpt-oss-{size}")
    dst = os.path.join(models_root, f"gpt-oss-{size}-bf16")
    return src, dst


def main():
    p = argparse.ArgumentParser(
        description="Load a local GPT-OSS model and resave it as bf16 (safe-serialized)."
    )
    p.add_argument(
        "size",
        choices=["20b", "120b"],
        help="Model size to use when deriving default paths (20b or 120b).",
    )
    p.add_argument(
        "--src",
        default=None,
        help="Override source path (defaults to <MODELS_ROOT>/gpt-oss-<size>).",
    )
    p.add_argument(
        "--dst",
        default=None,
        help="Override destination path (defaults to <MODELS_ROOT>/gpt-oss-<size>-bf16).",
    )
    args = p.parse_args()

    default_src, default_dst = paths_for_size(args.size)
    src = args.src or default_src
    dst = args.dst or default_dst

    if not os.path.isdir(src):
        print(f"ERROR: Source path does not exist: {src}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading tokenizer from {src} ...")
    tok = AutoTokenizer.from_pretrained(src)

    print(f"Loading model from {src} (bf16, eager attn)...")
    model = AutoModelForCausalLM.from_pretrained(
        src,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )

    print(f"Saving to {dst} ...")
    tok.save_pretrained(dst)
    model.save_pretrained(dst, safe_serialization=True)

    print("Done.")


if __name__ == "__main__":
    main()
