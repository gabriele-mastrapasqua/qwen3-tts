#!/usr/bin/env python3
"""
Reproduce C prefill in Python and compare hidden states.
This helps identify where the C implementation diverges from Python.
"""
import json
import struct
import numpy as np
import torch
import torch.nn.functional as F

MODEL_DIR = "qwen3-tts-0.6b"
TEXT = "Hello world"

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

def load_f32_tensor(path, tensor_name, shape):
    """Load a f32 tensor from safetensors file."""
    with open(path, 'rb') as f:
        header_size = struct.unpack('<Q', f.read(8))[0]
        header = json.loads(f.read(header_size).decode('utf-8'))
        meta = header[tensor_name]
        offset = meta['data_offsets'][0]
        size = meta['data_offsets'][1]
        f.seek(8 + header_size + offset)
        data = f.read(size)
        np_data = np.frombuffer(data, dtype=np.float32)
        return torch.from_numpy(np_data.reshape(shape))

print("=" * 60)
print("Loading model weights...")
print("=" * 60)

# Load weights
model_path = f"{MODEL_DIR}/model.safetensors"

text_emb = load_bf16_tensor(model_path, 'talker.model.text_embedding.weight', (151936, 2048))
text_proj_fc1_w = load_bf16_tensor(model_path, 'talker.text_projection.linear_fc1.weight', (2048, 2048))
text_proj_fc1_b = load_f32_tensor(model_path, 'talker.text_projection.linear_fc1.bias', (2048,))
text_proj_fc2_w = load_bf16_tensor(model_path, 'talker.text_projection.linear_fc2.weight', (1024, 2048))
text_proj_fc2_b = load_f32_tensor(model_path, 'talker.text_projection.linear_fc2.bias', (1024,))
codec_emb = load_bf16_tensor(model_path, 'talker.model.codec_embedding.weight', (3072, 1024))
codec_head = load_bf16_tensor(model_path, 'talker.codec_head.weight', (3072, 1024))
talker_norm = load_f32_tensor(model_path, 'talker.model.norm.weight', (1024,))

print(f"Loaded text_embedding: {text_emb.shape}")
print(f"Loaded text_proj_fc1: {text_proj_fc1_w.shape}, {text_proj_fc1_b.shape}")
print(f"Loaded text_proj_fc2: {text_proj_fc2_w.shape}, {text_proj_fc2_b.shape}")
print(f"Loaded codec_embedding: {codec_emb.shape}")
print(f"Loaded codec_head: {codec_head.shape}")

# Special tokens
CODEC_PAD = 2148
CODEC_BOS = 2149
CODEC_EOS = 2150
CODEC_THINK = 2155
CODEC_NO_THINK = 2154
CODEC_THINK_BOS = 2156
CODEC_THINK_EOS = 2157
TTS_BOS = 151672
TTS_EOS = 151673
TTS_PAD = 151648

# Speaker and language
SPEAKER_ID = 3066  # serena
LANGUAGE_ID = 2050  # English

print(f"\nSpecial tokens:")
print(f"  CODEC_PAD={CODEC_PAD}, CODEC_BOS={CODEC_BOS}, CODEC_EOS={CODEC_EOS}")
print(f"  TTS_PAD={TTS_PAD}, TTS_BOS={TTS_BOS}, TTS_EOS={TTS_EOS}")

# Build codec prefix (matching C)
codec_prefix = [CODEC_NO_THINK, CODEC_THINK_BOS, CODEC_THINK_EOS, SPEAKER_ID, CODEC_PAD, CODEC_BOS]
print(f"\nCodec prefix: {codec_prefix}")

# Simple test: just get codec embeddings
print("\n" + "=" * 60)
print("Testing codec embeddings...")
print("=" * 60)

for i, tok in enumerate(codec_prefix):
    emb = codec_emb[tok]
    norm = torch.norm(emb)
    print(f"  [{i}] token={tok}, emb norm={norm:.4f}, first 4: {emb[:4]}")

# Test codec_head logits from a simple input
print("\n" + "=" * 60)
print("Testing codec_head with codec_bos embedding...")
print("=" * 60)

codec_bos_emb = codec_emb[CODEC_BOS]
logits = codec_head @ codec_bos_emb
logits_norm = torch.norm(logits)
print(f"codec_bos embedding norm: {torch.norm(codec_bos_emb):.4f}")
print(f"logits norm: {logits_norm:.4f}")
print(f"logits first 10: {logits[:10]}")

# Find top tokens
top_vals, top_idx = torch.topk(logits, 10)
print(f"\nTop 10 tokens from codec_bos embedding:")
for i, (idx, val) in enumerate(zip(top_idx, top_vals)):
    token_type = "PAD" if idx == CODEC_PAD else "BOS" if idx == CODEC_BOS else "EOS" if idx == CODEC_EOS else ""
    print(f"  {i+1}. token {idx.item()} ({token_type}): logit={val.item():.4f}")

print(f"\nEOS logit: {logits[CODEC_EOS].item():.4f}")
print(f"BOS logit: {logits[CODEC_BOS].item():.4f}")
print(f"PAD logit: {logits[CODEC_PAD].item():.4f}")
