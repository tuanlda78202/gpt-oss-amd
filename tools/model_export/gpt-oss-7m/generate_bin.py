# Wrapping config and weights in to a .bin file

import argparse
from collections import OrderedDict
import json
import re
import struct

import numpy as np
from safetensors.torch import load_file
import torch

# Explicit list to ensure correct keys order
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


def binarize_config(cfg):
    config_bin = b""
    for key in KEYS:
        val = cfg[key]
        print("key", key, "val", val)
        if isinstance(val, int):
            config_bin += struct.pack("<i", val)
        elif isinstance(val, float):
            config_bin += struct.pack("<f", val)
        else:
            raise TypeError(f"Unsupported type for key {key}: {type(val)}")
    return config_bin


def binarize_weights(state_dict, dtype="float32"):
    # reorder weights such that it is compatible to Karpathy's llama2.c loading script
    buckets = {cat: [] for cat in CATEGORIES}
    others = []
    for key in state_dict:
        matched = False
        for cat in CATEGORIES:
            if key.endswith(cat):
                match = re.search(r"block\.(\d+)\.", key)
                block_idx = int(match.group(1)) if match else -1
                buckets[cat].append((block_idx, key))
                matched = True
                break
        if not matched:
            others.append(key)

    reordered = OrderedDict()
    for cat in CATEGORIES:
        sorted_bucket = sorted(buckets[cat], key=lambda x: x[0])
        for _, key in sorted_bucket:
            reordered[key] = state_dict[key]
    for key in sorted(others):
        reordered[key] = state_dict[key]

    # convert to binary
    np_dtype = np.float32 if dtype == "float32" else np.bfloat16
    torch_dtype = torch.float32 if dtype == "float32" else np.bfloat16
    weights_bin = b""
    for name, tensor in reordered.items():
        np_tensor = tensor.detach().cpu().to(torch_dtype).numpy().astype(np_dtype)
        weights_bin += np_tensor.tobytes()
        print(f"Binarized {name}")

    return weights_bin


def parseCLIArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, help="Path to config.json file")
    parser.add_argument(
        "-i", "--input", type=str, help="Path to input .safetensors file"
    )
    parser.add_argument("-o", "--output", type=str, help="Path to output .bin file")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parseCLIArgs()
    cfg = args.config
    inp = args.input
    out = args.output
    with open(args.config, "r") as f:
        config_json = json.load(f)

    config_bin = binarize_config(config_json)

    state_dict = load_file(inp)
    weights_bin = binarize_weights(state_dict)
    print("Writing to file ...")
    with open(out, "wb") as f:
        f.write(config_bin + weights_bin)
