#!/usr/bin/env python3
"""
Simple test: load speech tokenizer and decode test codes.
This verifies if the speech decoder works correctly in Python.
"""
import torch
import torch.nn.functional as F
import json
import struct
import numpy as np
import soundfile as sf
from safetensors.torch import load_file

MODEL_DIR = "qwen3-tts-0.6b"

print("=" * 60)
print("Loading speech tokenizer weights...")
print("=" * 60)

st_path = f"{MODEL_DIR}/speech_tokenizer/model.safetensors"
W = load_file(st_path)

# Load config
with open(f"{MODEL_DIR}/speech_tokenizer/config.json") as f:
    cfg = json.load(f)

dec_cfg = cfg["decoder_config"]
print(f"Decoder config:")
print(f"  hidden_size: {dec_cfg['hidden_size']}")
print(f"  num_hidden_layers: {dec_cfg['num_hidden_layers']}")
print(f"  latent_dim: {dec_cfg['latent_dim']}")
print(f"  codebook_size: {dec_cfg['codebook_size']}")
print(f"  upsample_rates: {dec_cfg['upsample_rates']}")

# Create test codes (random)
n_frames = 50
n_codebooks = 16
np.random.seed(42)
test_codes = np.random.randint(100, 2000, size=(n_frames, n_codebooks))

print(f"\nTest codes: {n_frames} frames x {n_codebooks} codebooks")

# Simple dequantization test
print("\nStep 1: Codebook dequantization...")

# Load codebooks
emb0 = W["decoder.quantizer.rvq_first.vq.layers.0._codebook.embedding_sum"]
usage0 = W["decoder.quantizer.rvq_first.vq.layers.0._codebook.cluster_usage"]
cb0 = emb0 / usage0.clamp(min=1e-5).unsqueeze(1)

cb_rest = []
for k in range(15):
    emb = W[f"decoder.quantizer.rvq_rest.vq.layers.{k}._codebook.embedding_sum"]
    usage = W[f"decoder.quantizer.rvq_rest.vq.layers.{k}._codebook.cluster_usage"]
    cb_rest.append(emb / usage.clamp(min=1e-5).unsqueeze(1))

# Dequantize
codes_t = torch.tensor(test_codes, dtype=torch.long)

# Codebook 0
q_first = F.embedding(codes_t[:, 0], cb0)  # [T, 256]

# Codebooks 1-15
q_rest_sum = torch.zeros(n_frames, 256)
for k in range(15):
    q_rest_sum += F.embedding(codes_t[:, k + 1], cb_rest[k])

print(f"  q_first: {q_first.shape}, norm={q_first.norm():.4f}")
print(f"  q_rest_sum: {q_rest_sum.shape}, norm={q_rest_sum.norm():.4f}")

# Apply output projections
out_proj_first = W["decoder.quantizer.rvq_first.output_proj.weight"]  # [512, 256, 1]
out_proj_rest = W["decoder.quantizer.rvq_rest.output_proj.weight"]  # [512, 256, 1]

q_first_proj = F.conv1d(q_first.T.unsqueeze(0), out_proj_first)  # [1, 512, T]
q_rest_proj = F.conv1d(q_rest_sum.T.unsqueeze(0), out_proj_rest)  # [1, 512, T]

vq_out = (q_first_proj + q_rest_proj).squeeze(0)  # [512, T]
print(f"  VQ output: {vq_out.shape}, norm={vq_out.norm():.4f}")

# Pre-conv
pre_conv_w = W["decoder.pre_conv.conv.weight"]  # [1024, 512, 3]
pre_conv_b = W["decoder.pre_conv.conv.bias"]

# Causal padding: pad_left = 2
vq_padded = F.pad(vq_out, (2, 0))
pre_conv_out = F.conv1d(vq_padded.unsqueeze(0), pre_conv_w, pre_conv_b)  # [1, 1024, T]
print(f"  Pre-conv out: {pre_conv_out.shape}, norm={pre_conv_out.norm():.4f}")

# Input projection
inp_w = W["decoder.pre_transformer.input_proj.weight"]
inp_b = W["decoder.pre_transformer.input_proj.bias"]

latent = pre_conv_out.squeeze(0).T  # [T, 1024]
hidden = F.linear(latent, inp_w, inp_b)  # [T, 512]
print(f"  Hidden (after input_proj): {hidden.shape}, norm={hidden.norm():.4f}")

print("\n" + "=" * 60)
print("Basic decoder stages work in Python!")
print("=" * 60)

# Save intermediate for C comparison
with open("/tmp/python_vq_out.bin", "wb") as f:
    vq_out.T.numpy().astype(np.float32).tofile(f)
    
with open("/tmp/python_hidden.bin", "wb") as f:
    hidden.numpy().astype(np.float32).tofile(f)

print(f"\nSaved intermediates for C comparison:")
print(f"  /tmp/python_vq_out.bin: {n_frames}x256 f32")
print(f"  /tmp/python_hidden.bin: {n_frames}x512 f32")
