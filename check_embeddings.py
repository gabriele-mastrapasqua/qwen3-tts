#!/usr/bin/env python3
import struct
import numpy as np

def load_bf16_tensor(path, tensor_name, shape):
    """Load a bf16 tensor from safetensors file manually."""
    import json
    
    with open(path, 'rb') as f:
        # Read header size
        header_size = struct.unpack('<Q', f.read(8))[0]
        # Read header
        header_bytes = f.read(header_size)
        header = json.loads(header_bytes.decode('utf-8'))
        
        if tensor_name not in header:
            raise KeyError(f"Tensor {tensor_name} not found")
        
        meta = header[tensor_name]
        offset = meta['data_offsets'][0]
        size = meta['data_offsets'][1]
        
        # Read tensor data
        f.seek(8 + header_size + offset)
        data = f.read(size)
        
        # Convert bf16 to float32
        n_elements = np.prod(shape)
        bf16_data = np.frombuffer(data, dtype=np.uint16, count=n_elements)
        
        # bf16 to float32: pad with 16 zero bits
        f32_data = np.zeros(n_elements, dtype=np.float32)
        for i in range(n_elements):
            f32_data.view(np.uint32)[i] = int(bf16_data[i]) << 16
        
        return f32_data.reshape(shape)

# Load codec embedding and head
codec_emb = load_bf16_tensor('qwen3-tts-0.6b/model.safetensors', 
                             'talker.model.codec_embedding.weight', 
                             (3072, 1024))
codec_head = load_bf16_tensor('qwen3-tts-0.6b/model.safetensors',
                              'talker.codec_head.weight',
                              (3072, 1024))

print(f"Codec embedding shape: {codec_emb.shape}")
print(f"Codec head shape: {codec_head.shape}")

# Check special tokens
CODEC_PAD = 2148
CODEC_BOS = 2149
CODEC_EOS = 2150

print(f"\nCodec embedding norms:")
print(f"  PAD ({CODEC_PAD}): {np.linalg.norm(codec_emb[CODEC_PAD]):.4f}")
print(f"  BOS ({CODEC_BOS}): {np.linalg.norm(codec_emb[CODEC_BOS]):.4f}")
print(f"  EOS ({CODEC_EOS}): {np.linalg.norm(codec_emb[CODEC_EOS]):.4f}")

print(f"\nCodec head norms (special tokens):")
print(f"  PAD ({CODEC_PAD}): {np.linalg.norm(codec_head[CODEC_PAD]):.4f}")
print(f"  BOS ({CODEC_BOS}): {np.linalg.norm(codec_head[CODEC_BOS]):.4f}")
print(f"  EOS ({CODEC_EOS}): {np.linalg.norm(codec_head[CODEC_EOS]):.4f}")

# Check speaker embedding
SPEAKER_SERENA = 3066
print(f"\nSpeaker serena ({SPEAKER_SERENA}):")
print(f"  Codec embedding norm: {np.linalg.norm(codec_emb[SPEAKER_SERENA]):.4f}")
print(f"  Codec head norm: {np.linalg.norm(codec_head[SPEAKER_SERENA]):.4f}")

# Check some random codec codes
print(f"\nRandom codec code norms (first 10):")
for i in range(10):
    print(f"  {i}: emb={np.linalg.norm(codec_emb[i]):.4f}, head={np.linalg.norm(codec_head[i]):.4f}")
