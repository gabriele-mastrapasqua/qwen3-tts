#!/usr/bin/env python3
import struct
import numpy as np

# Load codec codes dumped by C
with open('/tmp/codec_codes.bin', 'rb') as f:
    n_frames, n_codebooks = struct.unpack('ii', f.read(8))
    codes = np.frombuffer(f.read(), dtype=np.int32).reshape(n_frames, n_codebooks)

print(f"Loaded codes: {n_frames} frames x {n_codebooks} codebooks")
print(f"Code range: [{codes.min()}, {codes.max()}]")
print(f"First 3 frames codes:")
for i in range(min(3, n_frames)):
    print(f"  Frame {i}: {codes[i]}")

# Try to decode with Python
try:
    import torch
    from transformers import AutoModel
    
    st_dir = 'qwen3-tts-0.6b/speech_tokenizer'
    model = AutoModel.from_pretrained(st_dir, trust_remote_code=True)
    model.eval()
    
    # Prepare codes: [B, K, T]
    codes_tensor = torch.tensor(codes, dtype=torch.long).T.unsqueeze(0)  # [1, 16, T]
    print(f"\nInput tensor shape: {codes_tensor.shape}")
    
    with torch.no_grad():
        # Decode
        if hasattr(model, 'decode'):
            audio = model.decode(codes_tensor)
        elif hasattr(model, 'decoder'):
            # Get quantized output first
            quantized = model.quantizer.decode(codes_tensor)
            print(f"Quantized shape: {quantized.shape}")
            audio = model.decoder(quantized)
        else:
            # Try forward
            audio = model(codes_tensor)
    
    audio_np = audio.squeeze().numpy()
    print(f"\nPython audio: shape={audio_np.shape}, min={audio_np.min():.4f}, max={audio_np.max():.4f}, mean={audio_np.mean():.6f}")
    
    # Save
    import soundfile as sf
    sf.write('/tmp/python_decoder.wav', audio_np, 24000)
    print(f"Saved /tmp/python_decoder.wav ({len(audio_np)/24000:.2f}s)")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
