#!/usr/bin/env python3
"""
Generate reference audio using the official Qwen3-TTS Python pipeline.

Uses:
  - qwen_tts.inference.Qwen3TTSModel.from_pretrained()
  - generate_custom_voice(text, speaker, language)

Output: /tmp/python_ref.wav
"""

import sys
import time

import numpy as np
import soundfile as sf
import torch

from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel

MODEL_DIR = "/Users/gabrielemastrapasqua/source/personal/qwen-tts/qwen3-tts-0.6b"
TEXT = "Hello world"
SPEAKER = "Serena"
LANGUAGE = "English"
OUTPUT_PATH = "/tmp/python_ref.wav"
SEED = 42


def main():
    print(f"Loading model from: {MODEL_DIR}")
    t0 = time.time()

    model = Qwen3TTSModel.from_pretrained(
        MODEL_DIR,
        torch_dtype=torch.float32,
        device_map="cpu",
    )

    load_time = time.time() - t0
    print(f"Model loaded in {load_time:.1f}s")
    print(f"Device: {model.device}")

    # Print supported speakers and languages
    speakers = model.get_supported_speakers()
    languages = model.get_supported_languages()
    print(f"Supported speakers: {speakers}")
    print(f"Supported languages: {languages}")

    # Set seed for reproducibility
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print(f"\nGenerating speech:")
    print(f"  Text: \"{TEXT}\"")
    print(f"  Speaker: {SPEAKER}")
    print(f"  Language: {LANGUAGE}")

    t1 = time.time()

    wavs, sample_rate = model.generate_custom_voice(
        text=TEXT,
        speaker=SPEAKER,
        language=LANGUAGE,
    )

    gen_time = time.time() - t1
    print(f"Generation completed in {gen_time:.1f}s")

    # wavs is a list of np.ndarray (one per batch item); we have 1 item
    audio = wavs[0]

    print(f"\n--- Output info ---")
    print(f"  Sample rate: {sample_rate} Hz")
    print(f"  Number of waveforms returned: {len(wavs)}")
    print(f"  Audio dtype: {audio.dtype}")
    print(f"  Audio shape: {audio.shape}")
    print(f"  Total samples: {len(audio)}")
    print(f"  Duration: {len(audio) / sample_rate:.3f}s")
    print(f"  RMS: {np.sqrt(np.mean(audio ** 2)):.6f}")
    print(f"  Peak: {np.max(np.abs(audio)):.6f}")
    print(f"  Mean: {np.mean(audio):.6f}")

    # Compute number of codec frames: sample_rate=24000, decode_upsample_rate=1920
    # So each frame produces 1920 samples
    upsample_rate = 1920
    num_frames = len(audio) / upsample_rate
    print(f"  Upsample rate: {upsample_rate}")
    print(f"  Estimated codec frames: {num_frames:.1f} (= {len(audio)} / {upsample_rate})")

    # Save to WAV
    sf.write(OUTPUT_PATH, audio, sample_rate)
    print(f"\nSaved to: {OUTPUT_PATH}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
