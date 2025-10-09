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
    """
    Dequantize MXFP4-encoded blocks using per-row scales into a tensor with the specified dtype.
    
    Blocks should contain packed 4-bit indices in their last dimension; scales contains one scale value per logical row and is interpreted as an unsigned byte biased by 127 (i.e., effective exponent = scales - 127). The function expands the 4-bit indices via the FP4 lookup table, applies the per-row scale as a base-2 exponent, and returns the resulting tensor cast to `dtype`.
    
    Parameters:
        blocks (torch.Tensor): Tensor of packed MXFP4 blocks. Its shape is (..., G, B) where the last dim B holds packed 4-bit indices.
        scales (torch.Tensor): Tensor of per-row scales. Its shape must match blocks.shape[:-1]; values are treated as unsigned bytes with bias 127.
        dtype (torch.dtype): Desired output dtype (default: torch.bfloat16).
        rows_per_chunk (int): Number of rows to process per chunk to control peak memory usage.
    
    Returns:
        torch.Tensor: Dequantized tensor with dtype `dtype` and shape (..., G * B * 2), where each packed byte in B expands to two dequantized values.
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
        Initialize the TensorProvider with a lazy loader for a tensor.
        
        Parameters:
            loader (Callable[[], torch.Tensor]): A zero-argument callable that returns the tensor when invoked; stored for on-demand loading.
        """
        self._loader = loader

    def load(self) -> torch.Tensor:
        """
        Load and return the tensor provided by this provider.
        
        Returns:
            torch.Tensor: The loaded tensor.
        """
        return self._loader()


def collect_effective_keys_single_file(f) -> Dict[str, TensorProvider]:
    """
    Map tensors in a single safetensors file to lazy TensorProvider loaders, synthesizing dequantized tensors for FP4 block/scale pairs.
    
    Parameters:
        f: A safetensors file handle (as returned by safe_open) containing tensors to expose.
    
    Returns:
        effective (Dict[str, TensorProvider]): A mapping from tensor names (base keys) to TensorProvider instances. For any matching pair of keys ending with `.blocks` and `.scales`, the returned key is the base name (without the suffixes) and its provider yields the dequantized tensor; other keys map to providers that load the raw tensor.
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
                    """
                    Create a zero-argument loader that reads FP4 blocks and scales from the enclosing safetensors file and returns a dequantized tensor.
                    
                    Parameters:
                        blocks_key (str): Key for the FP4 blocks tensor in the safetensors file.
                        scales_key (str): Key for the FP4 scales tensor in the safetensors file.
                    
                    Returns:
                        Callable[[], torch.Tensor]: A no-argument function that, when called, loads the blocks and scales using the provided keys and returns the dequantized tensor cast to bfloat16.
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
            continue
        if key.endswith(".blocks") and key[: -len(".blocks")] in effective:
            continue
        if key not in effective:

            def make_plain_loader(k=key):
                """
                Create a zero-argument loader that reads a tensor by key from the currently-open safetensors file.
                
                Parameters:
                    k (str): The tensor key to read from the safetensors file referenced as `f` in the enclosing scope.
                
                Returns:
                    Callable[[], torch.Tensor]: A no-argument function that, when called, returns the tensor for `k` from `f`.
                """
                def _ld():
                    return f.get_tensor(k)

                return _ld

            effective[key] = TensorProvider(make_plain_loader())

    return effective


def collect_effective_keys_from_index(
    index_path: str, base_dir: str
) -> Dict[str, TensorProvider]:
    """
    Build a mapping of tensor names to lazy loaders from a safetensors index and its shard files.
    
    Resolves entries in the index.json's weight_map to shard files under base_dir and constructs TensorProvider instances whose loaders open the containing shard on demand. If a pair of keys with suffixes `.blocks` and `.scales` is present, the mapping contains the base tensor name (without suffix) and its provider returns the FP4-dequantized tensor; other keys map to providers that return the tensor as stored in the shard.
    
    Parameters:
        index_path (str): Path to the index.json that contains the `weight_map`.
        base_dir (str): Directory used to resolve relative shard paths found in the index.
    
    Returns:
        Dict[str, TensorProvider]: A dictionary mapping tensor names to TensorProvider objects. For FP4-quantized tensors the key is the base name (without `.blocks`/`.scales`) and the provider returns a dequantized tensor; for all other keys the provider returns the raw tensor from its shard.
    """
    with open(index_path, "r") as jf:
        index = json.load(jf)
    weight_map: Dict[str, str] = index.get("weight_map", {})
    keys = set(weight_map.keys())

    def file_for_key(k: str) -> str:
        """
        Resolve the relative shard path for a tensor key into an absolute filesystem path.
        
        Parameters:
            k (str): Tensor key as listed in the index's weight_map.
        
        Returns:
            path (str): Absolute filesystem path to the safetensors shard containing the tensor.
        """
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
                    """
                    Create a zero-argument loader that reads FP4 `blocks` and `scales` tensors from two safetensors files and returns their dequantized tensor.
                    
                    Parameters:
                        blocks_key (str): Key name of the FP4 blocks tensor inside the blocks safetensors file.
                        scales_key (str): Key name of the FP4 scales tensor inside the scales safetensors file.
                        blocks_path (str): Filesystem path to the safetensors file containing the blocks tensor.
                        scales_path (str): Filesystem path to the safetensors file containing the scales tensor.
                    
                    Returns:
                        loader (Callable[[], torch.Tensor]): A callable that, when invoked, loads the specified tensors and returns the dequantized tensor cast to `torch.bfloat16`.
                    """
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
                """
                Create a zero-argument loader that opens a safetensors file and returns the tensor named by `k`.
                
                Parameters:
                    k (str): Tensor key to load from the safetensors file.
                    p (str): Path to the safetensors file.
                
                Returns:
                    loader (Callable[[], torch.Tensor]): A callable that, when invoked, opens `p` with safetensors and returns the tensor for `k`.
                """
                def _ld():
                    with safe_open(p, framework="pt", device="cpu") as f:
                        return f.get_tensor(k)

                return _ld

            effective[key] = TensorProvider(make_plain_loader())

    return effective


# ---- Ordering ----

_BLOCK_RE = re.compile(r"block\.(\d+)\.")


def reorder_keys_for_write(all_keys: List[str]) -> List[str]:
    """
    Order tensor keys for deterministic output.
    
    Keys whose name ends with one of the suffixes in CATEGORIES are grouped in the same order as CATEGORIES.
    Within each group keys are sorted by the integer in the first "block.<n>." token found in the key (keys without such a token are treated as having index -1).
    All other keys that do not match any category suffix are appended sorted alphabetically.
    
    Returns:
    	ordered_keys (List[str]): The input keys reordered according to the grouping and sorting rules above.
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
    Write the configuration values defined by KEYS to a binary stream in a fixed order.
    
    Each key listed in the module-level KEYS is read from `config` and written to `fout` as a little-endian 32-bit integer for int values or a little-endian 32-bit float for float values.
    
    Parameters:
        config (dict): Mapping of configuration keys to numeric values (int or float).
        fout: Writable binary file-like object (must support write(bytes)) positioned where the header should be written.
    
    Raises:
        TypeError: If a value for any key has an unsupported type.
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
    Write tensors to an open binary file in a deterministic order using provided lazy loaders.
    
    Parameters:
        providers (Dict[str, TensorProvider]): Mapping from tensor name to a provider that returns the tensor when loaded.
        ordered_keys (List[str]): Sequence of tensor names specifying the write order.
        fout: Binary file-like object opened for writing; raw tensor bytes are appended to it.
        out_dtype (str): Output storage dtype, either "float32" or "bfloat16". Selects how each tensor is cast before writing.
    
    Behavior:
        For each name in ordered_keys, the corresponding provider is loaded, the tensor is cast to the selected dtype and moved to CPU, and its raw bytes (C-contiguous) are written to fout. When `out_dtype` is "bfloat16", the 16-bit bfloat representation is written.
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
    """
    Parse command-line arguments for exporting a GPT-OSS 120B checkpoint into a single .bin file.
    
    The parser recognizes the following options: --input (path to a safetensors file, an index.json, or a directory containing model.safetensors.index.json and shard files), --config (path to config.json), --output (destination .bin path), and --dtype (output weights dtype: "float32" or "bfloat16").
    
    Returns:
        argparse.Namespace: Parsed arguments with attributes `input`, `config`, `output`, and `dtype`.
    """
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
    """
    Orchestrates the CLI export: loads config and input tensors, orders keys, and writes a single binary model file.
    
    Loads the configuration JSON from the path supplied by the CLI, collects tensors from either a safetensors shard directory (using model.safetensors.index.json), an explicit index JSON, or a single safetensors file, orders the tensor keys for deterministic output, writes the fixed-format config header and streamed weight bytes to the output path, and prints a completion message.
    
    Raises:
        FileNotFoundError: If an input directory is provided but the expected index file (model.safetensors.index.json) is missing.
    """
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