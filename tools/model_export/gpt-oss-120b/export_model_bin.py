#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
export_model_bin.py (120B)
--------------------------
Exporter for GPT-OSS 120B checkpoints.

Differences from 20B:
- Supports multi-shard safetensors via index.json (typical for 120B)
- Same binary layout: [config header] + [ordered weights]
- FP4 dequantization on the fly when `.blocks` + `.scales` pairs are present

Usage
-----
# From an input directory that contains `model.safetensors.index.json` and shards
python export_model_bin.py \
  --input "/path/to/gpt-oss-120b/original" \
  --config "./config.json" \
  --output "../../gpt-oss-120b.bin"

# Or explicitly pass the index JSON
python export_model_bin.py \
  --input "/path/to/gpt-oss-120b/original/model.safetensors.index.json" \
  --config "./config.json" \
  --output "../../gpt-oss-120b.bin"

# Single-file inputs are also supported (mirrors 20B script)
python export_model_bin.py \
  --input "/path/to/model.safetensors" \
  --config "./config.json" \
  --output "../../gpt-oss-120b.bin"
"""

import argparse
import json
import math
import os
import re
import struct
from typing import Callable, Dict, List

import numpy as np
import torch
from safetensors.torch import safe_open

# ---- Config keys & ordering buckets (match 20B) ----

KEYS = [
    "vocab_size",
    "hidden_size",
    "num_experts",
    "experts_per_token",
    "intermediate_size",
    "num_hidden_layers",
    "head_dim",
    "num_attention_heads",
    "num_key_value_heads",
    "max_seq_len",
    "initial_context_length",
    "rope_theta",
    "rope_scaling_factor",
    "sliding_window",
    "swiglu_limit",
]

CATEGORIES = [
    #
    "embedding.weight",
    "unembedding.weight",
    "attn.norm.scale",
    "mlp.norm.scale",
    "norm.scale",
    # Attention
    "attn.qkv.weight",
    "attn.qkv.bias",
    "attn.out.weight",
    "attn.out.bias",
    "attn.sinks",
    # MoE
    "mlp.gate.weight",
    "mlp.gate.bias",
    "mlp.mlp1_weight",
    "mlp.mlp1_bias",
    "mlp.mlp2_weight",
    "mlp.mlp2_bias",
]

# ---- FP4 dequantization ----

FP4_VALUES = [
    +0.0,
    +0.5,
    +1.0,
    +1.5,
    +2.0,
    +3.0,
    +4.0,
    +6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
]


def dequantize_fp4(
    blocks: torch.Tensor,
    scales: torch.Tensor,
    dtype: torch.dtype = torch.bfloat16,
    rows_per_chunk: int = 16384 * 512,
) -> torch.Tensor:
    """Dequantize MXFP4 blocks/scales into `dtype` (default bfloat16)"""
    scales = scales.to(torch.int32) - 127
    assert blocks.shape[:-1] == scales.shape, (
        f"{blocks.shape=} does not match {scales.shape=}"
    )
    lut = torch.tensor(FP4_VALUES, dtype=dtype, device=blocks.device)

    *prefix_shape, G, B = blocks.shape
    rows_total = math.prod(prefix_shape) * G

    blocks = blocks.reshape(rows_total, B)
    scales = scales.reshape(rows_total, 1)

    out = torch.empty(rows_total, B * 2, dtype=dtype, device=blocks.device)

    for r0 in range(0, rows_total, rows_per_chunk):
        r1 = min(r0 + rows_per_chunk, rows_total)

        blk = blocks[r0:r1]
        exp = scales[r0:r1]

        idx_lo = (blk & 0x0F).to(torch.long)
        idx_hi = (blk >> 4).to(torch.long)

        sub = out[r0:r1]
        sub[:, 0::2] = lut[idx_lo]
        sub[:, 1::2] = lut[idx_hi]

        torch.ldexp(sub, exp, out=sub)

        del idx_lo, idx_hi, blk, exp

    return out.reshape(*prefix_shape, G, B * 2).view(*prefix_shape, G * B * 2)


# ---- Streaming providers so we don't materialize a giant dict ----


class TensorProvider:
    """Holds a callable to load (and optionally dequantize) a tensor on demand."""

    def __init__(self, loader: Callable[[], torch.Tensor]):
        self._loader = loader

    def load(self) -> torch.Tensor:
        return self._loader()


def collect_effective_keys_single_file(f) -> Dict[str, TensorProvider]:
    """
    Build mapping for a single .safetensors file using `safe_open` handle `f`.
    Pairs `.blocks` and `.scales` into a synthesized key without the suffix.
    """
    keys = set(f.keys())
    effective: Dict[str, TensorProvider] = {}

    # First pass: setup FP4 pairs
    for key in keys:
        if key.endswith(".blocks"):
            base = key[: -len(".blocks")]
            scales = base + ".scales"
            if scales in keys:
                # Create a provider that dequantizes on demand
                def make_loader(blocks_key=key, scales_key=scales):
                    def _ld():
                        blk = f.get_tensor(blocks_key)
                        scl = f.get_tensor(scales_key)
                        return dequantize_fp4(blk, scl, dtype=torch.bfloat16)

                    return _ld

                effective[base] = TensorProvider(make_loader())

    # Second pass: add the rest (excluding .scales and .blocks that already got paired)
    for key in keys:
        if key.endswith(".scales"):
            continue
        if key.endswith(".blocks") and key[: -len(".blocks")] in effective:
            continue
        if key not in effective:

            def make_plain_loader(k=key):
                def _ld():
                    return f.get_tensor(k)

                return _ld

            effective[key] = TensorProvider(make_plain_loader())

    return effective


def collect_effective_keys_from_index(
    index_path: str, base_dir: str
) -> Dict[str, TensorProvider]:
    """
    Build mapping using a safetensors index.json and multiple shards.
    Lazily opens the shard file that contains the requested tensor.
    """
    with open(index_path, "r") as jf:
        index = json.load(jf)
    weight_map: Dict[str, str] = index.get("weight_map", {})
    keys = set(weight_map.keys())

    def file_for_key(k: str) -> str:
        rel = weight_map[k]
        return os.path.join(base_dir, rel)

    effective: Dict[str, TensorProvider] = {}

    # FP4 pairs first
    for key in keys:
        if key.endswith(".blocks"):
            base = key[: -len(".blocks")]
            scales = base + ".scales"
            if scales in keys:
                blk_file = file_for_key(key)
                scl_file = file_for_key(scales)

                def make_loader(
                    blocks_key=key,
                    scales_key=scales,
                    blocks_path=blk_file,
                    scales_path=scl_file,
                ):
                    def _ld():
                        with safe_open(blocks_path, framework="pt", device="cpu") as fb:
                            blk = fb.get_tensor(blocks_key)
                        with safe_open(scales_path, framework="pt", device="cpu") as fs:
                            scl = fs.get_tensor(scales_key)
                        return dequantize_fp4(blk, scl, dtype=torch.bfloat16)

                    return _ld

                effective[base] = TensorProvider(make_loader())

    # Plain tensors from shards
    for key in keys:
        if key.endswith(".scales"):
            continue
        if key.endswith(".blocks") and key[: -len(".blocks")] in effective:
            continue
        if key not in effective:
            path = file_for_key(key)

            def make_plain_loader(k=key, p=path):
                def _ld():
                    with safe_open(p, framework="pt", device="cpu") as f:
                        return f.get_tensor(k)

                return _ld

            effective[key] = TensorProvider(make_plain_loader())

    return effective


# ---- Ordering ----

_BLOCK_RE = re.compile(r"block\.(\d+)\.")


def reorder_keys_for_write(all_keys: List[str]) -> List[str]:
    """Return keys ordered by CATEGORIES suffix + block index, then the rest sorted."""
    buckets = {cat: [] for cat in CATEGORIES}
    others: List[str] = []

    for key in all_keys:
        matched = False
        for cat in CATEGORIES:
            if key.endswith(cat):
                m = _BLOCK_RE.search(key)
                block_idx = int(m.group(1)) if m else -1
                buckets[cat].append((block_idx, key))
                matched = True
                break
        if not matched:
            others.append(key)

    ordered: List[str] = []
    for cat in CATEGORIES:
        for _, k in sorted(buckets[cat], key=lambda x: x[0]):
            ordered.append(k)

    ordered.extend(sorted(others))
    return ordered


# ---- Binary writing ----


def write_config_header(config: dict, fout) -> None:
    """Write config keys in fixed order as <i or <f little-endian."""
    for key in KEYS:
        val = config[key]
        if isinstance(val, int):
            fout.write(struct.pack("<i", val))
        elif isinstance(val, float):
            fout.write(struct.pack("<f", val))
        else:
            raise TypeError(f"Unsupported type for key {key}: {type(val)}")


def write_weights_streaming(
    providers: Dict[str, TensorProvider],
    ordered_keys: List[str],
    fout,
    out_dtype: str = "float32",
) -> None:
    """
    Stream tensors to file in the specified order.
    out_dtype: "float32" (default) or "bfloat16"
    """
    torch_cast = torch.float32 if out_dtype == "float32" else torch.bfloat16

    for name in ordered_keys:
        tens = providers[name].load().to(torch_cast).cpu()
        if torch_cast == torch.bfloat16:
            np_arr = tens.view(torch.uint16).numpy().astype(np.uint16, copy=False)
        else:
            np_arr = tens.numpy().astype(np.float32, copy=False)

        fout.write(np_arr.tobytes(order="C"))
        print(f"Wrote {name}  shape={tuple(tens.shape)} dtype={torch_cast}")


# ---- CLI ----


def parse_args():
    p = argparse.ArgumentParser(
        description="Export GPT-OSS 120B to single .bin (config + weights)."
    )
    p.add_argument(
        "--input",
        required=True,
        help=(
            "Path to input: a .safetensors file, an index.json, or a directory "
            "containing model.safetensors.index.json and shards"
        ),
    )
    p.add_argument("--config", required=True, help="Path to config.json")
    p.add_argument("--output", required=True, help="Output .bin path")
    p.add_argument(
        "--dtype",
        choices=["float32", "bfloat16"],
        default="float32",
        help="Output weights dtype (default float32)",
    )
    return p.parse_args()


def main():
    args = parse_args()

    # Load config JSON
    with open(args.config, "r") as cf:
        config = json.load(cf)

    in_path = args.input
    providers: Dict[str, TensorProvider]

    if os.path.isdir(in_path):
        index_path = os.path.join(in_path, "model.safetensors.index.json")
        if not os.path.exists(index_path):
            raise FileNotFoundError(
                f"Could not find index at {index_path}. Provide the index JSON explicitly via --input."
            )
        providers = collect_effective_keys_from_index(index_path, in_path)
        ordered = reorder_keys_for_write(list(providers.keys()))

    else:
        # Distinguish single-file vs index.json input by extension
        ext = os.path.splitext(in_path)[1].lower()
        if ext == ".json":
            base_dir = os.path.dirname(in_path)
            providers = collect_effective_keys_from_index(in_path, base_dir)
            ordered = reorder_keys_for_write(list(providers.keys()))
        else:
            # Single .safetensors file path
            with safe_open(in_path, framework="pt", device="cpu") as f:
                providers = collect_effective_keys_single_file(f)
                ordered = reorder_keys_for_write(list(providers.keys()))

    # Write binary
    with open(args.output, "wb") as fout:
        write_config_header(config, fout)
        write_weights_streaming(providers, ordered, fout, out_dtype=args.dtype)

    print(f"Done. Wrote {args.output}")


if __name__ == "__main__":
    main()
