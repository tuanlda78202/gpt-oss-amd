#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
export_model_bin.py
-------------------
One-shot exporter for GPT-OSS checkpoints:
- Reads a single input safetensors file that may contain FP4 `.blocks` + `.scales`
- Dequantizes FP4 on the fly (no intermediate files)
- Writes [config header + ordered weights] directly to a single .bin

Usage
-----
# Typical
python export_model_bin.py \
  --input original.safetensors \
  --config config.json \
  --output gpt-oss-20b.bin
"""

import argparse
import json
import math
import re
import struct
from typing import Callable, Dict, List

import numpy as np
import torch
from safetensors.torch import safe_open

# ---- Config keys & ordering buckets ----

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
    """
    Dequantize tensors stored in MXFP4 packed `blocks` with per-row `scales` into a full tensor of the requested `dtype`.
    
    Parameters:
        blocks (torch.Tensor): Packed FP4 data with shape (..., G, B) where each element contains two 4-bit values per output element.
        scales (torch.Tensor): Per-row scale bytes with shape matching `blocks.shape[:-1]`; values are interpreted as int32 and centered by subtracting 127.
        dtype (torch.dtype): Target tensor dtype for the dequantized output (default: `torch.bfloat16`).
        rows_per_chunk (int): Maximum number of rows processed at once to limit peak memory usage.
    
    Returns:
        torch.Tensor: Dequantized tensor with the same prefix dimensions as `blocks` but with the last two dimensions expanded so the embedding dimension equals G * B * 2, and elements cast to `dtype`.
    """
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
        """
        Create a TensorProvider that will load a tensor on demand using the given callable.
        
        Parameters:
            loader (Callable[[], torch.Tensor]): A zero-argument callable that, when invoked by `load()`, returns the requested `torch.Tensor`.
        """
        self._loader = loader

    def load(self) -> torch.Tensor:
        """
        Load and return the tensor produced by this provider's loader.
        
        Returns:
            torch.Tensor: The tensor produced by the stored loader callable.
        """
        return self._loader()


def collect_effective_keys(f) -> Dict[str, TensorProvider]:
    """
    Collect effective tensor names and create lazy providers, pairing FP4 ".blocks" with ".scales" into dequantized providers.
    
    For each key:
    - If both "X.blocks" and "X.scales" exist, exposes "X" with a TensorProvider that dequantizes on demand.
    - Keys ending with ".scales" are not exposed.
    - Other keys are exposed as passthrough TensorProviders that load the tensor as-is.
    
    Parameters:
        f: An opened safetensors-like object that supports `.keys()` and `.get_tensor(key)`.
    
    Returns:
        dict: Mapping from effective tensor name (str) to TensorProvider that lazily loads or dequantizes the tensor.
    """
    keys = set(f.keys())
    effective: Dict[str, TensorProvider] = {}

    # First pass: set up FP4 pairs
    for key in keys:
        if key.endswith(".blocks"):
            base = key[: -len(".blocks")]
            scales = base + ".scales"
            if scales in keys:
                # Create a provider that dequantizes on demand
                def make_loader(blocks_key=key, scales_key=scales):
                    """
                    Create a zero-argument loader that fetches FP4 `blocks` and `scales` tensors from the safetensors and returns their dequantized tensor.
                    
                    Parameters:
                        blocks_key (str): Key name of the FP4 blocks tensor in the safetensors.
                        scales_key (str): Key name of the corresponding FP4 scales tensor in the safetensors.
                    
                    Returns:
                        callable: A zero-argument callable that loads the tensors identified by `blocks_key` and `scales_key` and returns the dequantized tensor in bfloat16.
                    """
                    def _ld():
                        blk = f.get_tensor(blocks_key)
                        scl = f.get_tensor(scales_key)
                        return dequantize_fp4(blk, scl, dtype=torch.bfloat16)

                    return _ld

                effective[base] = TensorProvider(make_loader())

    # Second pass: add the rest (excluding .scales and .blocks that already got paired)
    for key in keys:
        if key.endswith(".scales"):
            continue  # consumed
        if key.endswith(".blocks"):
            base = key[: -len(".blocks")]
            if base in effective:
                continue  # already have the dequantized provider
        if key not in effective:
            # Plain tensor passthrough
            def make_plain_loader(k=key):
                """
                Create a zero-argument loader that fetches a tensor by key from the surrounding safetensors handle.
                
                Parameters:
                    k (str): Name of the tensor key to load from the `f` safetensors mapping.
                
                Returns:
                    loader (Callable[[], torch.Tensor]): A callable that, when invoked, returns the tensor for `k` from `f`.
                """
                def _ld():
                    return f.get_tensor(k)

                return _ld

            effective[key] = TensorProvider(make_plain_loader())

    return effective


# ---- Ordering ----

_BLOCK_RE = re.compile(r"block\.(\d+)\.")


def reorder_keys_for_write(all_keys: List[str]) -> List[str]:
    """
    Produce a deterministic ordering of tensor keys for binary output.
    
    The returned list places keys that end with any suffix from CATEGORIES first, iterating categories in CATEGORIES order and sorting keys within each category by the numeric block index extracted from "block.<n>." in the key (missing index sorts before numbered blocks). Keys that do not match any category are appended in lexicographic order.
    
    Returns:
        List[str]: Keys ordered by category suffix and block index, followed by remaining keys sorted alphabetically.
    """
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
    """
    Write the values from `config` into `fout` in the fixed order defined by `KEYS`.
    
    For each key in `KEYS`, writes the corresponding value from `config` as:
    - a 4-byte little-endian signed integer if the value is an `int`,
    - a 4-byte little-endian IEEE 754 float if the value is a `float`.
    
    Parameters:
        config (dict): Mapping containing all keys listed in `KEYS` with integer or float values.
        fout: Binary file-like object with a writable `write(bytes)` method.
    
    Raises:
        TypeError: If a value for a key in `KEYS` is not an `int` or `float`.
    """
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
    Write tensors from providers to the given binary file in the specified order and numeric format.
    
    Parameters:
        providers (Dict[str, TensorProvider]): Mapping from effective tensor name to a provider that supplies the tensor when loaded.
        ordered_keys (List[str]): Sequence of provider keys that determines the write order.
        fout: Binary file-like object with a write(bytes) method where tensor payloads will be written.
        out_dtype (str): Output numeric format; either "float32" or "bfloat16". When "bfloat16", the raw 16-bit bf16 payload is written as uint16 to preserve bit-level representation.
    """
    # np_dtype = (
    # np.float32 if out_dtype == "float32" else np.uint16
    # )  # bf16 stored as uint16 payload
    torch_cast = torch.float32 if out_dtype == "float32" else torch.bfloat16

    for name in ordered_keys:
        tens = providers[name].load().to(torch_cast).cpu()
        # For bfloat16, write the raw 16-bit payload (view as uint16) to avoid
        # accidental conversion.
        if torch_cast == torch.bfloat16:
            np_arr = tens.view(torch.uint16).numpy().astype(np.uint16, copy=False)
        else:
            np_arr = tens.numpy().astype(np.float32, copy=False)

        fout.write(np_arr.tobytes(order="C"))
        print(f"Wrote {name}  shape={tuple(tens.shape)} dtype={torch_cast}")


# ---- CLI ----


def parse_args():
    """
    Parse command-line arguments for the exporter.
    
    Returns:
        args (argparse.Namespace): Parsed arguments with attributes:
            - input (str): Path to the input .safetensors (may contain .blocks/.scales).
            - config (str): Path to the model config.json.
            - output (str): Path to the output .bin file.
            - dtype (str): Output weights dtype, either "float32" or "bfloat16".
    """
    p = argparse.ArgumentParser(
        description="Export GPT-OSS to single .bin (config + weights) in one step."
    )
    p.add_argument(
        "--input",
        required=True,
        help="Path to input .safetensors (may contain .blocks/.scales)",
    )
    p.add_argument("--config", required=True, help="Path to config.json")
    p.add_argument("--output", required=True, help="Output .bin path")
    p.add_argument(
        "--dtype",
        choices=["float32", "bfloat16"],
        default="float32",
        help="Output weights dtype (default float32, matching previous pipeline)",
    )
    return p.parse_args()


def main():
    """
    Execute the export pipeline: parse CLI arguments, load the config and input safetensors, and write a single .bin containing the fixed config header followed by ordered weight tensors.
    
    The input path, config JSON, output path, and output dtype are taken from the command-line arguments. Tensors are streamed (dequantizing FP4 on the fly when present) into the output file, and a completion message with the output path is printed.
    """
    args = parse_args()

    # Load config JSON
    with open(args.config, "r") as cf:
        config = json.load(cf)

    # Open safetensors once; build providers
    with safe_open(args.input, framework="pt", device="cpu") as f:
        providers = collect_effective_keys(f)
        all_keys = list(providers.keys())
        ordered = reorder_keys_for_write(all_keys)

        # Write binary in one pass
        with open(args.output, "wb") as fout:
            write_config_header(config, fout)
            write_weights_streaming(providers, ordered, fout, out_dtype=args.dtype)

    print(f"Done. Wrote {args.output}")


if __name__ == "__main__":
    main()