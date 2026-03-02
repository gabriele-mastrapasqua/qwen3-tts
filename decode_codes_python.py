#!/usr/bin/env python3
"""Decode codec codes from C binary dump using Python speech decoder, then compare."""
import struct
import numpy as np
import torch
import soundfile as sf

CODES_BIN = "/tmp/codec_codes.bin"
C_WAV = "/tmp/test_det5.wav"
PY_WAV = "/tmp/test_det5_pydec.wav"

def main():
    # Read C binary dump: int32 array [frames * 16]
    with open(CODES_BIN, "rb") as f:
        data = f.read()
    n_ints = len(data) // 4
    all_ints = struct.unpack(f"{n_ints}i", data)
    # First 2 ints are header: [n_frames, n_codebooks]
    n_frames = all_ints[0]
    n_codebooks = all_ints[1]
    codes_flat = all_ints[2:]
    print(f"Read {n_frames} frames x 16 codebooks from {CODES_BIN}")

    # Reshape to [frames, 16] (codes_length x num_quantizers) - matches Python V2 format
    codes = np.array(codes_flat, dtype=np.int64).reshape(n_frames, 16)
    print(f"Codes shape: {codes.shape}  (frames x codebooks)")
    for f_idx in range(min(n_frames, 5)):
        print(f"  frame {f_idx}: {' '.join(str(codes[f_idx, cb]) for cb in range(16))}")

    # Load Python speech tokenizer
    from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
    model = Qwen3TTSModel.from_pretrained(
        "qwen3-tts-0.6b", torch_dtype=torch.float32, device_map="cpu"
    )
    speech_tokenizer = model.model.speech_tokenizer

    # Decode using Python speech tokenizer
    # V2 model expects audio_codes shape: (codes_length, num_quantizers) = (frames, 16)
    codes_tensor = torch.tensor(codes, dtype=torch.long)  # [frames, 16]
    print(f"\nDecoding with Python speech tokenizer...")
    print(f"  Input tensor shape: {codes_tensor.shape}")

    # Pass as dict with audio_codes key - decode handles 2D tensor as single sample
    encoded = {"audio_codes": codes_tensor}  # [frames, 16]
    wavs, sr = speech_tokenizer.decode(encoded)
    wav = wavs[0] if isinstance(wavs, list) else wavs
    if torch.is_tensor(wav):
        wav = wav.squeeze().numpy()
    elif isinstance(wav, np.ndarray):
        wav = wav.squeeze()

    print(f"  Output shape: {wav.shape}")
    print(f"  Duration: {len(wav)/24000:.3f}s")
    print(f"  RMS: {np.sqrt(np.mean(wav**2)):.6f}")
    print(f"  Peak: {np.max(np.abs(wav)):.6f}")

    sf.write(PY_WAV, wav, 24000)
    print(f"\nSaved Python-decoded audio to {PY_WAV}")

    # Also read C WAV for comparison
    c_wav, sr = sf.read(C_WAV)
    print(f"\nC audio: {len(c_wav)} samples, {len(c_wav)/sr:.3f}s, RMS={np.sqrt(np.mean(c_wav**2)):.6f}, Peak={np.max(np.abs(c_wav)):.6f}")
    print(f"Py audio: {len(wav)} samples, {len(wav)/24000:.3f}s, RMS={np.sqrt(np.mean(wav**2)):.6f}, Peak={np.max(np.abs(wav)):.6f}")

    # Compare waveforms
    min_len = min(len(c_wav), len(wav))
    if min_len > 0:
        diff = np.abs(c_wav[:min_len] - wav[:min_len])
        print(f"\nWaveform comparison (first {min_len} samples):")
        print(f"  Max diff: {diff.max():.6f}")
        print(f"  Mean diff: {diff.mean():.6f}")
        print(f"  Correlation: {np.corrcoef(c_wav[:min_len], wav[:min_len])[0,1]:.6f}")

if __name__ == "__main__":
    main()
