import argparse
from collections import OrderedDict
import os

from safetensors.torch import load_file
from safetensors.torch import safe_open
from safetensors.torch import save_file
import torch


def convert_keys(state_dict_b):
    new_state_dict = OrderedDict()

    for k, v in state_dict_b.items():
        new_k = k
        # Attention:
        new_k = new_k.replace(".input_layernorm.weight", ".attn.norm.scale")
        new_k = new_k.replace("model.layers", "block")
        new_k = new_k.replace(".self_attn", ".attn")
        new_k = new_k.replace(".o_proj", ".out")
        new_k = new_k.replace(".post_attention_layernorm.weight",
                              ".mlp.norm.scale")
        new_k = new_k.replace(".experts.gate_up_proj_bias", ".mlp1_bias")
        new_k = new_k.replace(".experts.gate_up_proj", ".mlp1_weight")
        new_k = new_k.replace(".experts.down_proj_bias", ".mlp2_bias")
        new_k = new_k.replace(".experts.down_proj", ".mlp2_weight")
        new_k = new_k.replace(".router", ".gate")
        #
        new_k = new_k.replace("model.norm.weight", "norm.scale")
        new_k = new_k.replace("model.embed_tokens", "embedding")

        new_state_dict[new_k] = v
    return new_state_dict


def concat_qkv(state_dict, block_id):
    q_w = state_dict[f"block.{block_id}.attn.q_proj.weight"]
    k_w = state_dict[f"block.{block_id}.attn.k_proj.weight"]
    v_w = state_dict[f"block.{block_id}.attn.v_proj.weight"]

    qkv_w = torch.cat([q_w, k_w, v_w], dim=0)

    q_b = state_dict[f"block.{block_id}.attn.q_proj.bias"]
    k_b = state_dict[f"block.{block_id}.attn.k_proj.bias"]
    v_b = state_dict[f"block.{block_id}.attn.v_proj.bias"]

    qkv_b = torch.cat([q_b, k_b, v_b], dim=0)

    new_state_dict = OrderedDict(state_dict)
    new_state_dict[f"block.{block_id}.attn.qkv.weight"] = qkv_w
    new_state_dict[f"block.{block_id}.attn.qkv.bias"] = qkv_b

    del new_state_dict[f"block.{block_id}.attn.q_proj.weight"]
    del new_state_dict[f"block.{block_id}.attn.k_proj.weight"]
    del new_state_dict[f"block.{block_id}.attn.v_proj.weight"]
    del new_state_dict[f"block.{block_id}.attn.q_proj.bias"]
    del new_state_dict[f"block.{block_id}.attn.k_proj.bias"]
    del new_state_dict[f"block.{block_id}.attn.v_proj.bias"]

    return new_state_dict


def reshape_mlp(state_dict, block_id):
    mlp1_weight = state_dict[f"block.{block_id}.mlp.mlp1_weight"]
    mlp2_weight = state_dict[f"block.{block_id}.mlp.mlp2_weight"]

    new_state_dict = OrderedDict(state_dict)
    new_state_dict[
        f"block.{block_id}.mlp.mlp1_weight"] = mlp1_weight.transpose(
            1, 2).contiguous()
    new_state_dict[
        f"block.{block_id}.mlp.mlp2_weight"] = mlp2_weight.transpose(
            1, 2).contiguous()

    return new_state_dict


def add_unembedding(state_dict):
    embedding_weight = state_dict["embedding.weight"]
    unembedding_weight = embedding_weight.clone()

    new_state_dict = OrderedDict(state_dict)
    new_state_dict["unembedding.weight"] = unembedding_weight

    return new_state_dict


def parseCLIArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i",
                        "--input",
                        type=str,
                        required=True,
                        help="Path to the original .safetensors file")
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="model.safetensors",
        help=
        "Path to save the converted .safetensors file (default: model.safetensors)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parseCLIArgs()
    inp = args.input
    out = args.output

    # print keys
    with safe_open(inp, framework="pt", device="cpu") as f:
        keys = f.keys()
        print(f"Keys in original {inp}:")
        for key in keys:
            print(f"key: {key}, {f.get_tensor(key).shape}")

    # load and convert
    original_ckpt = load_file(inp)
    converted_ckpt = convert_keys(original_ckpt)

    # transformations
    ckpt = converted_ckpt
    ckpt = concat_qkv(ckpt, 0)
    ckpt = concat_qkv(ckpt, 1)
    ckpt = reshape_mlp(ckpt, 0)
    ckpt = reshape_mlp(ckpt, 1)
    ckpt = add_unembedding(ckpt)

    # save to requested output
    save_file(ckpt, out)

    with safe_open(out, framework="pt", device="cpu") as f:
        keys = f.keys()
        print(f"Keys in modified {out}:")
        for key in keys:
            print(f"key: {key}, {f.get_tensor(key).shape}")
