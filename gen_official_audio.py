#!/usr/bin/env python3
"""Generate Hello World with official Qwen3-TTS Python"""
import torch
import soundfile as sf
import sys
sys.path.insert(0, '.')

from qwen_tts import Qwen3TTSModel

MODEL_DIR = "qwen3-tts-0.6b"
TEXT = "Hello world"
OUTPUT = "/tmp/python_official_hello.wav"

print("Loading Qwen3-TTS model (this may take a minute)...")
tts = Qwen3TTSModel.from_pretrained(MODEL_DIR)

print(f"Generating: '{TEXT}'")
wavs, sr = tts.generate_custom_voice(
    text=TEXT,
    language="English",
    speaker="Serena",
)

print(f"Audio saved: {OUTPUT}")
print(f"Sample rate: {sr}")
print(f"Duration: {len(wavs[0])/sr:.2f}s")

sf.write(OUTPUT, wavs[0], sr)
print("Done!")
