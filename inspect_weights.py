#!/usr/bin/env python3
"""Inspect Qwen3-TTS model weights from safetensors files.

Usage:
    python3 inspect_weights.py --model-dir qwen3-tts-0.6b
    python3 inspect_weights.py --model-dir qwen3-tts-1.7b
"""

import argparse
import json
import os
import struct
import sys


def read_safetensors_metadata(path):
    """Read tensor metadata from a safetensors file without loading data."""
    with open(path, "rb") as f:
        # First 8 bytes: little-endian u64 = JSON header length
        header_len_bytes = f.read(8)
        header_len = struct.unpack("<Q", header_len_bytes)[0]
        # Read JSON header
        header_json = f.read(header_len).decode("utf-8")
        header = json.loads(header_json)
    return header


def dtype_size(dtype_str):
    """Bytes per element for a safetensors dtype string."""
    sizes = {
        "F32": 4, "F16": 2, "BF16": 2, "F64": 8,
        "I8": 1, "I16": 2, "I32": 4, "I64": 8,
        "U8": 1, "U16": 2, "U32": 4, "U64": 8,
        "BOOL": 1,
    }
    return sizes.get(dtype_str, 0)


def format_shape(shape):
    return "[" + " x ".join(str(s) for s in shape) + "]"


def format_size_mb(nbytes):
    return f"{nbytes / (1024*1024):.1f} MB"


def inspect_model(model_dir):
    print(f"=== Inspecting model: {model_dir} ===\n")

    # Check for config.json
    config_path = os.path.join(model_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
        print("--- config.json ---")
        for key in sorted(config.keys()):
            val = config[key]
            if isinstance(val, (dict, list)) and len(str(val)) > 100:
                print(f"  {key}: <{type(val).__name__}, len={len(val)}>")
            else:
                print(f"  {key}: {val}")
        print()

    # Check for generation_config.json
    gen_config_path = os.path.join(model_dir, "generation_config.json")
    if os.path.exists(gen_config_path):
        with open(gen_config_path) as f:
            gen_config = json.load(f)
        print("--- generation_config.json ---")
        for key in sorted(gen_config.keys()):
            print(f"  {key}: {gen_config[key]}")
        print()

    # Find all safetensors files
    main_safetensors = []
    speech_safetensors = []

    for f in sorted(os.listdir(model_dir)):
        if f.endswith(".safetensors"):
            main_safetensors.append(os.path.join(model_dir, f))

    speech_dir = os.path.join(model_dir, "speech_tokenizer")
    if os.path.isdir(speech_dir):
        for f in sorted(os.listdir(speech_dir)):
            if f.endswith(".safetensors"):
                speech_safetensors.append(os.path.join(speech_dir, f))

    # Inspect main model tensors
    print("--- Main Model Tensors ---")
    all_tensors = {}
    total_bytes = 0

    for sf_path in main_safetensors:
        header = read_safetensors_metadata(sf_path)
        basename = os.path.basename(sf_path)
        for name, info in sorted(header.items()):
            if name == "__metadata__":
                continue
            shape = info["shape"]
            dtype = info["dtype"]
            nbytes = 1
            for s in shape:
                nbytes *= s
            nbytes *= dtype_size(dtype)
            total_bytes += nbytes
            print(f"  {name:70s}  {format_shape(shape):30s}  {dtype:5s}  {format_size_mb(nbytes):>10s}  ({basename})")
            all_tensors[name] = {"shape": shape, "dtype": dtype, "file": basename}

    print(f"\n  Total: {len(all_tensors)} tensors, {format_size_mb(total_bytes)}")
    print()

    # Categorize tensors
    categories = {
        "embed_tokens": [],
        "talker_layers": [],
        "talker_norm": [],
        "codec_head": [],
        "codec_embeddings": [],
        "spk_embeddings": [],
        "subtalker": [],
        "other": [],
    }

    for name in sorted(all_tensors.keys()):
        if "embed_tokens" in name:
            categories["embed_tokens"].append(name)
        elif "subtalker" in name or "code_predictor" in name:
            categories["subtalker"].append(name)
        elif "codec_head" in name:
            categories["codec_head"].append(name)
        elif "codec_embed" in name:
            categories["codec_embeddings"].append(name)
        elif "spk_embed" in name:
            categories["spk_embeddings"].append(name)
        elif ".layers." in name and "subtalker" not in name:
            categories["talker_layers"].append(name)
        elif "model.norm" in name:
            categories["talker_norm"].append(name)
        else:
            categories["other"].append(name)

    print("--- Tensor Categories ---")
    for cat, names in categories.items():
        if names:
            print(f"  {cat}: {len(names)} tensors")
            if len(names) <= 5:
                for n in names:
                    print(f"    {n}")
    print()

    # Inspect speech tokenizer tensors
    if speech_safetensors:
        print("--- Speech Tokenizer Tensors ---")
        speech_tensors = {}
        speech_bytes = 0

        for sf_path in speech_safetensors:
            header = read_safetensors_metadata(sf_path)
            basename = os.path.basename(sf_path)
            for name, info in sorted(header.items()):
                if name == "__metadata__":
                    continue
                shape = info["shape"]
                dtype = info["dtype"]
                nbytes = 1
                for s in shape:
                    nbytes *= s
                nbytes *= dtype_size(dtype)
                speech_bytes += nbytes
                print(f"  {name:70s}  {format_shape(shape):30s}  {dtype:5s}  {format_size_mb(nbytes):>10s}")
                speech_tensors[name] = {"shape": shape, "dtype": dtype}

        print(f"\n  Total: {len(speech_tensors)} tensors, {format_size_mb(speech_bytes)}")
        print()

        # Speech tokenizer config
        speech_config_path = os.path.join(speech_dir, "config.json")
        if os.path.exists(speech_config_path):
            with open(speech_config_path) as f:
                speech_config = json.load(f)
            print("--- Speech Tokenizer config.json ---")
            for key in sorted(speech_config.keys()):
                val = speech_config[key]
                if isinstance(val, (dict, list)) and len(str(val)) > 100:
                    print(f"  {key}: <{type(val).__name__}, len={len(val)}>")
                else:
                    print(f"  {key}: {val}")
            print()
    else:
        print("--- No speech tokenizer directory found ---")
        print()

    print("=== Done ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect Qwen3-TTS model weights")
    parser.add_argument("--model-dir", required=True, help="Model directory path")
    args = parser.parse_args()

    if not os.path.isdir(args.model_dir):
        print(f"Error: {args.model_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    inspect_model(args.model_dir)
