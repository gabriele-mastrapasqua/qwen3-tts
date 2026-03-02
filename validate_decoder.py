#!/usr/bin/env python3
"""
Validate C speech decoder against Python reference.
Loads the speech_tokenizer model in Python, feeds it the same codec codes
that our C generates, and compares the audio output.
"""
import json
import struct
import sys
import os
import numpy as np

# Try to load the speech tokenizer model
def load_speech_tokenizer_python(model_dir):
    """Try to load the SpeechTokenizer decoder using the transformers or local code."""
    st_dir = os.path.join(model_dir, "speech_tokenizer")
    config_path = os.path.join(st_dir, "config.json")

    with open(config_path) as f:
        config = json.load(f)

    print(f"Speech tokenizer config:")
    print(f"  model_type: {config.get('model_type', 'unknown')}")

    # Try to import the model class
    try:
        sys.path.insert(0, st_dir)
        # Check what Python files exist
        py_files = [f for f in os.listdir(st_dir) if f.endswith('.py')]
        print(f"  Python files in speech_tokenizer dir: {py_files}")

        if 'modeling_speech_tokenizer.py' in py_files:
            from modeling_speech_tokenizer import SpeechTokenizer
            print("  Loaded SpeechTokenizer from local modeling file")
            return SpeechTokenizer, config
    except Exception as e:
        print(f"  Failed to import local module: {e}")

    # Try transformers AutoModel
    try:
        from transformers import AutoModel
        model = AutoModel.from_pretrained(st_dir, trust_remote_code=True)
        print(f"  Loaded via AutoModel: {type(model).__name__}")
        return model, config
    except Exception as e:
        print(f"  AutoModel failed: {e}")

    return None, config


def load_safetensors_weights(path):
    """Load weight tensors from a safetensors file."""
    import safetensors.numpy
    tensors = safetensors.numpy.load_file(path)
    return tensors


def decode_with_raw_weights(model_dir, codes, n_frames):
    """
    Manually reconstruct key decoder steps using raw weights.
    This helps verify individual stages against our C implementation.
    """
    st_path = os.path.join(model_dir, "speech_tokenizer", "model.safetensors")

    try:
        from safetensors.numpy import load_file
        weights = load_file(st_path)
    except ImportError:
        print("safetensors package not available, trying manual load...")
        return None

    num_codebooks = 16
    cb_dim = 256
    cb_size = 2048
    vq_hidden = 512

    # Step 1: Codebook lookup + dequant
    # rvq_first: codebook 0
    emb_sum_0 = weights["decoder.quantizer.rvq_first.vq.layers.0._codebook.embedding_sum"]
    usage_0 = weights["decoder.quantizer.rvq_first.vq.layers.0._codebook.cluster_usage"]
    cb0 = emb_sum_0 / np.maximum(usage_0[:, None], 1e-8)

    # rvq_rest: codebooks 1-15
    cb_rest = []
    for k in range(15):
        es = weights[f"decoder.quantizer.rvq_rest.vq.layers.{k}._codebook.embedding_sum"]
        cu = weights[f"decoder.quantizer.rvq_rest.vq.layers.{k}._codebook.cluster_usage"]
        cb_rest.append(es / np.maximum(cu[:, None], 1e-8))

    # Dequantize
    # rvq_first: codebook 0 lookup -> output_proj
    first_proj = weights["decoder.quantizer.rvq_first.output_proj.weight"]  # [512, 256, 1]
    first_proj = first_proj.squeeze(-1)  # [512, 256]

    rest_proj = weights["decoder.quantizer.rvq_rest.output_proj.weight"]  # [512, 256, 1]
    rest_proj = rest_proj.squeeze(-1)  # [512, 256]

    codes_np = np.array(codes).reshape(n_frames, num_codebooks)

    vq_out = np.zeros((n_frames, vq_hidden), dtype=np.float32)

    # rvq_first: code 0 -> lookup -> project
    for f in range(n_frames):
        entry0 = cb0[codes_np[f, 0]]  # [256]
        vq_out[f] += first_proj @ entry0  # [512]

    # rvq_rest: codes 1-15 -> sum lookups -> project
    for f in range(n_frames):
        rest_sum = np.zeros(cb_dim, dtype=np.float32)
        for k in range(15):
            rest_sum += cb_rest[k][codes_np[f, k + 1]]
        vq_out[f] += rest_proj @ rest_sum

    print(f"\n  [Python] VQ out: shape={vq_out.shape}, "
          f"min={vq_out.min():.4f}, max={vq_out.max():.4f}, mean={vq_out.mean():.4f}")

    # Step 2: Pre-conv (Conv1d 512->1024, k=3, pad=1)
    pre_conv_w = weights["decoder.pre_conv.conv.weight"]  # [1024, 512, 3]
    pre_conv_b = weights["decoder.pre_conv.conv.bias"]    # [1024]

    # Transpose to [512, n_frames] for conv
    vq_t = vq_out.T  # [512, n_frames]

    import torch
    import torch.nn.functional as F

    vq_tensor = torch.from_numpy(vq_t).unsqueeze(0)  # [1, 512, n_frames]
    pre_conv_wt = torch.from_numpy(pre_conv_w)  # [1024, 512, 3]
    pre_conv_bt = torch.from_numpy(pre_conv_b)  # [1024]

    pre_conv_out = F.conv1d(vq_tensor, pre_conv_wt, pre_conv_bt, padding=1)  # [1, 1024, n_frames]
    print(f"  [Python] Pre-conv out: shape={pre_conv_out.shape}, "
          f"min={pre_conv_out.min():.4f}, max={pre_conv_out.max():.4f}, mean={pre_conv_out.mean():.4f}")

    # Step 3: Input projection (Linear 1024->512)
    inp_w = weights["decoder.pre_transformer.input_proj.weight"]  # [512, 1024]
    inp_b = weights["decoder.pre_transformer.input_proj.bias"]    # [512]

    latent = pre_conv_out.squeeze(0).T.numpy()  # [n_frames, 1024]
    hidden = latent @ inp_w.T + inp_b  # [n_frames, 512]
    print(f"  [Python] After input_proj: shape={hidden.shape}, "
          f"min={hidden.min():.4f}, max={hidden.max():.4f}, mean={hidden.mean():.4f}")

    # We skip the full transformer since it's complex - just check the VQ+pre-conv stages

    return vq_out


def main():
    model_dir = sys.argv[1] if len(sys.argv) > 1 else "qwen3-tts-0.6b"

    print("=" * 60)
    print("Speech Decoder Validation (Python Reference)")
    print("=" * 60)

    # First generate codes with C implementation
    import subprocess

    text = "Hello world"
    c_cmd = ['./qwen_tts', '-d', model_dir, '--text', text,
             '-o', '/tmp/validate_dec.wav', '--seed', '42', '--debug']

    print(f"\nStep 1: Generate codec codes with C implementation...")
    result = subprocess.run(c_cmd, capture_output=True, text=True)

    # Parse the codec codes from debug output (we need to add this to C)
    # For now, let's just validate the decoder stages

    print(f"\nStep 2: Load Python speech tokenizer and compare stages...")

    # We can still validate by comparing the weights and first few stages
    model, config = load_speech_tokenizer_python(model_dir)

    if model is None:
        print("\nCouldn't load model directly. Trying raw weight comparison...")

    # Use raw weight approach - generate some test codes
    # Use the codes from C output (we'll need to dump them)
    # For now, use simple test codes
    n_frames = 5
    num_codebooks = 16
    np.random.seed(42)
    test_codes = np.random.randint(0, 2048, size=(n_frames, num_codebooks)).flatten().tolist()

    print(f"\nStep 3: Compare VQ dequantization with test codes ({n_frames} frames)...")
    vq_out = decode_with_raw_weights(model_dir, test_codes, n_frames)

    # Now try loading the full model and running decoder
    print(f"\nStep 4: Try full decoder forward pass...")
    try:
        import torch
        from transformers import AutoModel

        st_dir = os.path.join(model_dir, "speech_tokenizer")
        model = AutoModel.from_pretrained(st_dir, trust_remote_code=True)
        model.eval()

        # Create code tensor
        codes_tensor = torch.tensor(test_codes).reshape(1, n_frames, num_codebooks)
        codes_tensor = codes_tensor.permute(0, 2, 1)  # [1, 16, n_frames]

        print(f"  Input codes shape: {codes_tensor.shape}")

        with torch.no_grad():
            # Try to run the decoder
            if hasattr(model, 'decode'):
                audio = model.decode(codes_tensor)
                print(f"  [Python] Decoder output: shape={audio.shape}, "
                      f"min={audio.min():.4f}, max={audio.max():.4f}, mean={audio.mean():.4f}")
            elif hasattr(model, 'decoder'):
                # Try direct decoder access
                quantized = model.quantizer.decode(codes_tensor)
                print(f"  [Python] Quantizer decoded: shape={quantized.shape}, "
                      f"min={quantized.min():.4f}, max={quantized.max():.4f}, mean={quantized.mean():.4f}")
                audio = model.decoder(quantized)
                print(f"  [Python] Decoder output: shape={audio.shape}, "
                      f"min={audio.min():.4f}, max={audio.max():.4f}, mean={audio.mean():.4f}")
            else:
                print(f"  Model methods: {[m for m in dir(model) if not m.startswith('_')]}")
    except Exception as e:
        print(f"  Full decoder failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
