#!/usr/bin/env python3
"""
Simple test: check codec_head output for codec_bos embedding.
"""
import json
import struct
import numpy as np
import torch

MODEL_DIR = "qwen3-tts-0.6b"

def load_bf16_tensor(path, tensor_name, shape):
    """Load a bf16 tensor from safetensors file."""
    with open(path, 'rb') as f:
        header_size = struct.unpack('<Q', f.read(8))[0]
        header = json.loads(f.read(header_size).decode('utf-8'))
        meta = header[tensor_name]
        offset = meta['data_offsets'][0]
        size = meta['data_offsets'][1]
        f.seek(8 + header_size + offset)
        data = f.read(size)
        n_elements = np.prod(shape)
        bf16_data = np.frombuffer(data, dtype=np.uint16, count=n_elements)
        f32_data = np.zeros(n_elements, dtype=np.float32)
        for i in range(n_elements):
            f32_data.view(np.uint32)[i] = int(bf16_data[i]) << 16
        return torch.from_numpy(f32_data.reshape(shape))

print("=" * 60)
print("Loading codec_head and codec_embedding...")
print("=" * 60)

model_path = f"{MODEL_DIR}/model.safetensors"
codec_emb = load_bf16_tensor(model_path, 'talker.model.codec_embedding.weight', (3072, 1024))
codec_head = load_bf16_tensor(model_path, 'talker.codec_head.weight', (3072, 1024))

print(f"codec_embedding: {codec_emb.shape}")
print(f"codec_head: {codec_head.shape}")

# Special tokens
CODEC_PAD = 2148
CODEC_BOS = 2149
CODEC_EOS = 2150

print("\n" + "=" * 60)
print("Test 1: codec_head @ codec_bos_embedding")
print("=" * 60)

codec_bos_emb = codec_emb[CODEC_BOS]
logits = codec_head @ codec_bos_emb

print(f"codec_bos embedding norm: {torch.norm(codec_bos_emb):.4f}")
print(f"logits norm: {torch.norm(logits):.4f}")

# Find top tokens
top_vals, top_idx = torch.topk(logits, 15)
print(f"\nTop 15 tokens:")
for i, (idx, val) in enumerate(zip(top_idx, top_vals)):
    token_type = ""
    if idx == CODEC_PAD: token_type = " (PAD)"
    elif idx == CODEC_BOS: token_type = " (BOS)"
    elif idx == CODEC_EOS: token_type = " (EOS)"
    print(f"  {i+1}. token {idx.item()}{token_type}: logit={val.item():.4f}")

print(f"\nEOS logit: {logits[CODEC_EOS].item():.4f}, rank={(logits > logits[CODEC_EOS]).sum().item()}")
print(f"BOS logit: {logits[CODEC_BOS].item():.4f}, rank={(logits > logits[CODEC_BOS]).sum().item()}")
print(f"PAD logit: {logits[CODEC_PAD].item():.4f}, rank={(logits > logits[CODEC_PAD]).sum().item()}")

print("\n" + "=" * 60)
print("Test 2: codec_head @ codec_pad_embedding")
print("=" * 60)

codec_pad_emb = codec_emb[CODEC_PAD]
logits_pad = codec_head @ codec_pad_emb

print(f"codec_pad embedding norm: {torch.norm(codec_pad_emb):.4f}")
print(f"logits norm: {torch.norm(logits_pad):.4f}")

top_vals, top_idx = torch.topk(logits_pad, 15)
print(f"\nTop 15 tokens:")
for i, (idx, val) in enumerate(zip(top_idx, top_vals)):
    token_type = ""
    if idx == CODEC_PAD: token_type = " (PAD)"
    elif idx == CODEC_BOS: token_type = " (BOS)"
    elif idx == CODEC_EOS: token_type = " (EOS)"
    print(f"  {i+1}. token {idx.item()}{token_type}: logit={val.item():.4f}")

print(f"\nEOS logit: {logits_pad[CODEC_EOS].item():.4f}")

print("\n" + "=" * 60)
print("Test 3: codec_head @ speaker_embedding (serena=3066)")
print("=" * 60)

speaker_emb = codec_emb[3066]
logits_spk = codec_head @ speaker_emb

print(f"speaker embedding norm: {torch.norm(speaker_emb):.4f}")
print(f"logits norm: {torch.norm(logits_spk):.4f}")

top_vals, top_idx = torch.topk(logits_spk, 15)
print(f"\nTop 15 tokens:")
for i, (idx, val) in enumerate(zip(top_idx, top_vals)):
    token_type = ""
    if idx == CODEC_PAD: token_type = " (PAD)"
    elif idx == CODEC_BOS: token_type = " (BOS)"
    elif idx == CODEC_EOS: token_type = " (EOS)"
    print(f"  {i+1}. token {idx.item()}{token_type}: logit={val.item():.4f}")

print(f"\nEOS logit: {logits_spk[CODEC_EOS].item():.4f}")
