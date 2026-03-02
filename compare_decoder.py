#!/usr/bin/env python3
"""
Compare C speech decoder vs Python reference.
1. Loads codec codes dumped by C (--debug mode)
2. Runs them through the Python speech tokenizer decoder
3. Saves Python audio for comparison
"""
import struct
import sys
import os
import numpy as np

def load_codes(path="/tmp/codec_codes.bin"):
    """Load codec codes dumped by C implementation."""
    with open(path, "rb") as f:
        n_frames, n_codebooks = struct.unpack("ii", f.read(8))
        codes = np.frombuffer(f.read(), dtype=np.int32).reshape(n_frames, n_codebooks)
    print(f"Loaded {n_frames} frames x {n_codebooks} codebooks from {path}")
    print(f"  Code range: [{codes.min()}, {codes.max()}]")
    print(f"  First frame codes: {codes[0]}")
    return codes, n_frames, n_codebooks

def decode_with_python(codes, model_dir="qwen3-tts-0.6b"):
    """Decode codec codes using the Python speech tokenizer."""
    import torch

    st_dir = os.path.join(model_dir, "speech_tokenizer")

    # Try loading with the qwen_tts package
    try:
        sys.path.insert(0, st_dir)
        from transformers import AutoModel
        model = AutoModel.from_pretrained(st_dir, trust_remote_code=True)
        model.eval()
        print(f"Loaded model: {type(model).__name__}")
    except Exception as e:
        print(f"AutoModel failed: {e}")
        print("Trying to load decoder directly...")
        try:
            from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import (
                Qwen3TTSTokenizerV2Decoder,
            )
            from qwen_tts.core.tokenizer_12hz.configuration_qwen3_tts_tokenizer_v2 import (
                Qwen3TTSTokenizerV2DecoderConfig,
            )
            import json

            with open(os.path.join(st_dir, "config.json")) as f:
                full_config = json.load(f)

            dec_cfg = Qwen3TTSTokenizerV2DecoderConfig(**full_config["decoder_config"])
            model = Qwen3TTSTokenizerV2Decoder(dec_cfg)

            # Load weights
            from safetensors.torch import load_file
            weights = load_file(os.path.join(st_dir, "model.safetensors"))
            # Filter decoder weights
            dec_weights = {k.replace("decoder.", "", 1): v for k, v in weights.items()
                          if k.startswith("decoder.")}
            missing, unexpected = model.load_state_dict(dec_weights, strict=False)
            if missing:
                print(f"  Missing keys: {len(missing)}")
                for k in missing[:5]:
                    print(f"    {k}")
            if unexpected:
                print(f"  Unexpected keys: {len(unexpected)}")
            model.eval()
            print(f"Loaded decoder directly: {type(model).__name__}")
        except Exception as e2:
            print(f"Direct load also failed: {e2}")
            import traceback
            traceback.print_exc()
            return None

    # Prepare codes tensor: [B, K, T]
    n_frames, n_codebooks = codes.shape
    codes_tensor = torch.tensor(codes, dtype=torch.long)  # [T, K]
    codes_tensor = codes_tensor.T.unsqueeze(0)  # [1, K, T]
    print(f"  Input codes tensor: {codes_tensor.shape}")

    with torch.no_grad():
        # The model might be the full model or just the decoder
        if hasattr(model, 'decoder') and hasattr(model.decoder, 'quantizer'):
            # Full model - use decoder
            audio = model.decoder(codes_tensor)
        elif hasattr(model, 'quantizer'):
            # Direct decoder
            audio = model(codes_tensor)
        else:
            # Try to find the right method
            print(f"  Model attrs: {[a for a in dir(model) if not a.startswith('_')]}")
            # Try forward directly
            audio = model(codes_tensor)

    audio_np = audio.squeeze().numpy()
    print(f"  Python audio: shape={audio_np.shape}, min={audio_np.min():.4f}, "
          f"max={audio_np.max():.4f}, mean={audio_np.mean():.6f}")
    return audio_np

def decode_with_raw_weights(codes, model_dir="qwen3-tts-0.6b"):
    """
    Manual step-by-step decode using raw weights + PyTorch ops.
    This allows us to compare intermediate stages against C.
    """
    import torch
    import torch.nn.functional as F
    from safetensors.torch import load_file

    st_path = os.path.join(model_dir, "speech_tokenizer", "model.safetensors")
    W = load_file(st_path)

    n_frames, n_codebooks = codes.shape
    print(f"\n=== Step-by-step decode ({n_frames} frames) ===")

    # Step 1: VQ dequantize
    # rvq_first: codebook 0
    emb0 = W["decoder.quantizer.rvq_first.vq.layers.0._codebook.embedding_sum"]
    usage0 = W["decoder.quantizer.rvq_first.vq.layers.0._codebook.cluster_usage"]
    cb0 = emb0 / usage0.clamp(min=1e-5).unsqueeze(1)  # [2048, 256]

    # Lookup codebook 0
    codes_t = torch.tensor(codes, dtype=torch.long)
    q_first = F.embedding(codes_t[:, 0], cb0)  # [T, 256]
    q_first = q_first.T.unsqueeze(0)  # [1, 256, T]

    # Apply rvq_first output_proj (Conv1d 256->512, k=1)
    out_proj_first = W["decoder.quantizer.rvq_first.output_proj.weight"]  # [512, 256, 1]
    q_first = F.conv1d(q_first, out_proj_first)  # [1, 512, T]
    print(f"  VQ first: {q_first.shape}, min={q_first.min():.4f}, max={q_first.max():.4f}, mean={q_first.mean():.4f}")

    # rvq_rest: codebooks 1-15
    q_rest_sum = torch.zeros(1, 256, n_frames)
    for k in range(n_codebooks - 1):
        emb = W[f"decoder.quantizer.rvq_rest.vq.layers.{k}._codebook.embedding_sum"]
        usage = W[f"decoder.quantizer.rvq_rest.vq.layers.{k}._codebook.cluster_usage"]
        cb = emb / usage.clamp(min=1e-5).unsqueeze(1)
        q_k = F.embedding(codes_t[:, k + 1], cb)  # [T, 256]
        q_rest_sum += q_k.T.unsqueeze(0)  # accumulate in [1, 256, T]

    out_proj_rest = W["decoder.quantizer.rvq_rest.output_proj.weight"]  # [512, 256, 1]
    q_rest = F.conv1d(q_rest_sum, out_proj_rest)  # [1, 512, T]
    print(f"  VQ rest: {q_rest.shape}, min={q_rest.min():.4f}, max={q_rest.max():.4f}, mean={q_rest.mean():.4f}")

    vq_out = q_first + q_rest  # [1, 512, T]
    print(f"  VQ total: min={vq_out.min():.4f}, max={vq_out.max():.4f}, mean={vq_out.mean():.4f}")

    # Step 2: Pre-conv (CausalConv1d 512->1024, k=3)
    pre_conv_w = W["decoder.pre_conv.conv.weight"]  # [1024, 512, 3]
    pre_conv_b = W["decoder.pre_conv.conv.bias"]
    # Causal: pad left by (3-1)*1=2
    padded = F.pad(vq_out, (2, 0))
    pre_conv_out = F.conv1d(padded, pre_conv_w, pre_conv_b)  # [1, 1024, T]
    print(f"  Pre-conv: {pre_conv_out.shape}, min={pre_conv_out.min():.4f}, max={pre_conv_out.max():.4f}, mean={pre_conv_out.mean():.4f}")

    # Step 3: input_proj (Linear 1024->512)
    inp_w = W["decoder.pre_transformer.input_proj.weight"]  # [512, 1024]
    inp_b = W["decoder.pre_transformer.input_proj.bias"]
    hidden = pre_conv_out.transpose(1, 2)  # [1, T, 1024]
    hidden = F.linear(hidden, inp_w, inp_b)  # [1, T, 512]
    print(f"  Input proj: {hidden.shape}, min={hidden.min():.4f}, max={hidden.max():.4f}, mean={hidden.mean():.4f}")

    # We skip the transformer for now - just return intermediate values
    return {
        'vq_out': vq_out.squeeze().numpy(),
        'pre_conv': pre_conv_out.squeeze().numpy(),
        'input_proj': hidden.squeeze().numpy(),
    }


def main():
    model_dir = sys.argv[1] if len(sys.argv) > 1 else "qwen3-tts-0.6b"

    # Load codes
    codes, n_frames, n_codebooks = load_codes()

    # Step-by-step comparison
    intermediates = decode_with_raw_weights(codes, model_dir)

    # Full Python decode
    print("\n=== Full Python decode ===")
    audio = decode_with_python(codes, model_dir)

    if audio is not None:
        import soundfile as sf
        sf.write("/tmp/python_decode.wav", audio, 24000)
        print(f"  Saved /tmp/python_decode.wav ({len(audio)} samples, {len(audio)/24000:.2f}s)")

    # Also read C audio for comparison
    try:
        import soundfile as sf
        c_audio, sr = sf.read("/tmp/test_fixes.wav", dtype='float32')
        print(f"\n=== C audio ===")
        print(f"  Shape: {c_audio.shape}, SR: {sr}")
        print(f"  min={c_audio.min():.4f}, max={c_audio.max():.4f}, mean={c_audio.mean():.6f}")

        if audio is not None:
            min_len = min(len(audio), len(c_audio))
            corr = np.corrcoef(audio[:min_len], c_audio[:min_len])[0, 1]
            diff = audio[:min_len] - c_audio[:min_len]
            print(f"\n=== Comparison ===")
            print(f"  Correlation: {corr:.4f}")
            print(f"  Max abs diff: {np.max(np.abs(diff)):.4f}")
            print(f"  Mean abs diff: {np.mean(np.abs(diff)):.4f}")
    except Exception as e:
        print(f"Could not load C audio: {e}")


if __name__ == "__main__":
    main()
