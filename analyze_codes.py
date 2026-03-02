#!/usr/bin/env python3
"""
Generate audio with Python reference and compare codes.
"""
import subprocess
import struct
import numpy as np
import os

MODEL_DIR = "qwen3-tts-0.6b"
TEXT = "Hello world"

# First, get C codes
print("=" * 60)
print("Step 1: Generate codes with C implementation")
print("=" * 60)

cmd = [
    './qwen_tts',
    '-d', MODEL_DIR,
    '--text', TEXT,
    '-o', '/tmp/c_test.wav',
    '--seed', '42',
    '--max-tokens', '20',
    '--debug'
]

result = subprocess.run(cmd, capture_output=True, text=True)
print(result.stderr[-3000:])  # Last 3000 chars

# Load C codes
codes_file = '/tmp/codec_codes.bin'
if os.path.exists(codes_file):
    with open(codes_file, 'rb') as f:
        n_frames, n_codebooks = struct.unpack('ii', f.read(8))
        codes = np.frombuffer(f.read(), dtype=np.int32).reshape(n_frames, n_codebooks)
    
    print(f"\nC codes: {n_frames} frames x {n_codebooks} codebooks")
    print(f"Code range: [{codes.min()}, {codes.max()}]")
    print(f"First 5 frames:")
    for i in range(min(5, n_frames)):
        print(f"  Frame {i}: {codes[i]}")
    
    # Check for special tokens
    CODEC_EOS = 2150
    CODEC_BOS = 2149
    CODEC_PAD = 2148
    
    eos_count = np.sum(codes == CODEC_EOS)
    bos_count = np.sum(codes == CODEC_BOS)
    pad_count = np.sum(codes == CODEC_PAD)
    
    print(f"\nSpecial tokens in codes:")
    print(f"  EOS (2150): {eos_count}")
    print(f"  BOS (2149): {bos_count}")
    print(f"  PAD (2148): {pad_count}")
    
    # Check code distribution
    print(f"\nCode distribution (first codebook):")
    unique, counts = np.unique(codes[:, 0], return_counts=True)
    sorted_idx = np.argsort(-counts)[:10]
    for idx in sorted_idx:
        print(f"  Code {unique[idx]}: {counts[idx]} times")
else:
    print("No codes file found!")
