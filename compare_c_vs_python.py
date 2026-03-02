#!/usr/bin/env python3
"""
Compare C speech decoder output vs Python reference.
1. Generate codec codes with C implementation
2. Decode with Python speech tokenizer
3. Compare audio quality and save both outputs
"""
import subprocess
import sys
import os
import struct
import numpy as np

MODEL_DIR = "qwen3-tts-0.6b"
TEXT = "Hello world, this is a test."
OUTPUT_C_WAV = "/tmp/c_decoder.wav"
OUTPUT_PYTHON_WAV = "/tmp/python_decoder.wav"
CODES_BIN = "/tmp/codec_codes.bin"


def generate_codes_with_c():
    """Run C implementation to generate codec codes and audio."""
    print("=" * 60)
    print("Step 1: Generate codec codes with C implementation")
    print("=" * 60)
    
    cmd = [
        './qwen_tts',
        '-d', MODEL_DIR,
        '--text', TEXT,
        '-o', OUTPUT_C_WAV,
        '--seed', '42',
        '--max-tokens', '50',  # Limit frames for faster testing
        '--debug'
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stderr)
    
    if result.returncode != 0:
        print(f"C implementation failed: {result.stderr}")
        return False
    
    # Check if codes were dumped
    if not os.path.exists(CODES_BIN):
        print("Error: C implementation did not dump codec codes")
        return False
    
    print(f"C audio saved to: {OUTPUT_C_WAV}")
    return True


def load_codes():
    """Load codec codes dumped by C."""
    with open(CODES_BIN, 'rb') as f:
        n_frames, n_codebooks = struct.unpack('ii', f.read(8))
        codes = np.frombuffer(f.read(), dtype=np.int32).reshape(n_frames, n_codebooks)
    
    print(f"\nLoaded codes: {n_frames} frames x {n_codebooks} codebooks")
    print(f"Code range: [{codes.min()}, {codes.max()}]")
    print(f"First 3 frames:")
    for i in range(min(3, n_frames)):
        print(f"  Frame {i}: {codes[i]}")
    
    return codes, n_frames, n_codebooks


def decode_with_python(codes, n_frames, n_codebooks):
    """Decode codec codes using Python speech tokenizer."""
    print("\n" + "=" * 60)
    print("Step 2: Decode with Python speech tokenizer")
    print("=" * 60)
    
    try:
        import torch
        from transformers import AutoModel
    except ImportError as e:
        print(f"Missing dependencies: {e}")
        print("Install with: uv pip install torch transformers soundfile")
        return None
    
    st_dir = os.path.join(MODEL_DIR, "speech_tokenizer")
    
    try:
        print(f"Loading model from {st_dir}...")
        model = AutoModel.from_pretrained(st_dir, trust_remote_code=True)
        model.eval()
        print(f"Loaded model: {type(model).__name__}")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return None
    
    # Prepare codes tensor: [B, K, T]
    codes_tensor = torch.tensor(codes, dtype=torch.long).T.unsqueeze(0)  # [1, 16, T]
    print(f"Input codes tensor shape: {codes_tensor.shape}")
    
    with torch.no_grad():
        try:
            # Try different decode methods
            if hasattr(model, 'decode'):
                print("Using model.decode()...")
                audio = model.decode(codes_tensor)
            elif hasattr(model, 'decoder'):
                print("Using model.decoder(model.quantizer.decode())...")
                quantized = model.quantizer.decode(codes_tensor)
                print(f"Quantized shape: {quantized.shape}")
                audio = model.decoder(quantized)
            else:
                print(f"Available methods: {[m for m in dir(model) if not m.startswith('_')]}")
                # Try calling model directly
                audio = model(codes_tensor)
            
            audio_np = audio.squeeze().numpy()
            print(f"\nPython audio: shape={audio_np.shape}, min={audio_np.min():.4f}, "
                  f"max={audio_np.max():.4f}, mean={audio_np.mean():.6f}")
            
            return audio_np
            
        except Exception as e:
            print(f"Decode failed: {e}")
            import traceback
            traceback.print_exc()
            return None


def compare_audio(c_audio, py_audio):
    """Compare C and Python audio outputs."""
    print("\n" + "=" * 60)
    print("Step 3: Compare C vs Python audio")
    print("=" * 60)
    
    min_len = min(len(c_audio), len(py_audio))
    
    # Correlation
    corr = np.corrcoef(c_audio[:min_len], py_audio[:min_len])[0, 1]
    
    # Difference
    diff = c_audio[:min_len] - py_audio[:min_len]
    
    print(f"Correlation: {corr:.4f}")
    print(f"Max abs diff: {np.max(np.abs(diff)):.6f}")
    print(f"Mean abs diff: {np.mean(np.abs(diff)):.6f}")
    print(f"RMS diff: {np.sqrt(np.mean(diff**2)):.6f}")
    
    return corr, diff


def main():
    print("C vs Python Speech Decoder Comparison")
    print("=" * 60)
    
    # Step 1: Generate codes with C
    if not generate_codes_with_c():
        print("\nC code generation failed!")
        sys.exit(1)
    
    # Step 2: Load codes
    codes, n_frames, n_codebooks = load_codes()
    
    # Step 3: Decode with Python
    py_audio = decode_with_python(codes, n_frames, n_codebooks)
    
    if py_audio is None:
        print("\nPython decoding failed!")
        sys.exit(1)
    
    # Save Python audio
    import soundfile as sf
    sf.write(OUTPUT_PYTHON_WAV, py_audio, 24000)
    print(f"\nPython audio saved to: {OUTPUT_PYTHON_WAV}")
    print(f"Duration: {len(py_audio)/24000:.2f}s")
    
    # Step 4: Load C audio and compare
    try:
        c_audio, sr = sf.read(OUTPUT_C_WAV, dtype='float32')
        print(f"\nC audio: {len(c_audio)} samples, SR={sr}, duration={len(c_audio)/sr:.2f}s")
        
        if len(c_audio) > 0 and len(py_audio) > 0:
            compare_audio(c_audio, py_audio)
    except Exception as e:
        print(f"Could not load C audio for comparison: {e}")
    
    print("\n" + "=" * 60)
    print("Comparison complete!")
    print(f"C audio:      {OUTPUT_C_WAV}")
    print(f"Python audio: {OUTPUT_PYTHON_WAV}")
    print(f"Codes dump:   {CODES_BIN}")
    print("=" * 60)


if __name__ == "__main__":
    main()
