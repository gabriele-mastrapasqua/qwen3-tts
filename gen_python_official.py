#!/usr/bin/env python3
"""Generate Hello World with official Python Qwen3-TTS"""
import torch
from transformers import AutoModel
import soundfile as sf

MODEL_DIR = "qwen3-tts-0.6b"
TEXT = "Hello world"

print("Loading Qwen3-TTS model...")
model = AutoModel.from_pretrained(MODEL_DIR, trust_remote_code=True)
model.eval()

print(f"Generating: '{TEXT}'")
audio = model.generate(TEXT)

print(f"Audio shape: {audio.shape}")
print(f"Sample rate: {model.config.sampling_rate}")

# Save
output_path = "/tmp/python_official_hello.wav"
sf.write(output_path, audio.squeeze().numpy(), model.config.sampling_rate)
print(f"Saved: {output_path}")
print(f"Duration: {len(audio.squeeze())/model.config.sampling_rate:.2f}s")
