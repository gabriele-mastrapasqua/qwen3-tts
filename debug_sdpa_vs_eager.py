#!/usr/bin/env python3
"""Compare SDPA vs Eager attention outputs at each layer to find divergence."""
import torch
import numpy as np
import sys

MODEL_DIR = "/Users/gabrielemastrapasqua/source/personal/qwen-tts/qwen3-tts-0.6b"
TEXT = "Hello world"
SPEAKER = "Serena"
LANGUAGE = "English"

GEN_PARAMS = dict(
    temperature=0.0,
    top_k=1,
    top_p=1.0,
    do_sample=False,
    subtalker_temperature=0.0,
    subtalker_top_k=1,
    subtalker_top_p=1.0,
    subtalker_dosample=False,
    repetition_penalty=1.0,
    max_new_tokens=20,
)


def run_with_attn(attn_impl):
    """Run generation with given attention implementation, capturing per-layer outputs."""
    from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
    model = Qwen3TTSModel.from_pretrained(
        MODEL_DIR, torch_dtype=torch.float32, device_map="cpu",
        attn_implementation=attn_impl,
    )

    talker = model.model.talker
    print(f"\n=== {attn_impl.upper()} ===")
    print(f"  Talker attn_implementation: {talker.config._attn_implementation}")

    # Hook into each talker layer to capture attention outputs
    layer_outputs = {}
    layer_inputs = {}

    def make_layer_hook(layer_idx):
        def hook(module, inp, out):
            if isinstance(out, tuple):
                h = out[0]
            else:
                h = out
            last_h = h[0, -1].detach().clone()
            if layer_idx not in layer_outputs:
                layer_outputs[layer_idx] = []
            layer_outputs[layer_idx].append(last_h)

            # Also capture input
            if isinstance(inp, tuple):
                h_in = inp[0]
            else:
                h_in = inp
            if layer_idx not in layer_inputs:
                layer_inputs[layer_idx] = []
            layer_inputs[layer_idx].append(h_in[0, -1].detach().clone())
        return hook

    # Hook into attention modules specifically
    attn_outputs = {}
    def make_attn_hook(layer_idx):
        def hook(module, inp, out):
            if isinstance(out, tuple):
                h = out[0]
            else:
                h = out
            last_h = h[0, -1].detach().clone()
            if layer_idx not in attn_outputs:
                attn_outputs[layer_idx] = []
            attn_outputs[layer_idx].append(last_h)
        return hook

    for i, layer in enumerate(talker.model.layers):
        layer.register_forward_hook(make_layer_hook(i))
        layer.self_attn.register_forward_hook(make_attn_hook(i))

    # Also capture codec codes
    frame_codes = []
    def talker_hook(module, inp, out):
        if hasattr(out, 'hidden_states') and out.hidden_states is not None:
            _, codec_ids = out.hidden_states
            if codec_ids is not None:
                frame_codes.append(codec_ids[0].detach().cpu().tolist())
    talker.register_forward_hook(talker_hook)

    # Generate
    wavs, sr = model.generate_custom_voice(
        text=TEXT, speaker=SPEAKER, language=LANGUAGE, **GEN_PARAMS
    )

    return layer_outputs, layer_inputs, attn_outputs, frame_codes


def main():
    print("Running SDPA...")
    sdpa_layer_out, sdpa_layer_in, sdpa_attn_out, sdpa_codes = run_with_attn("sdpa")

    print("\nRunning EAGER...")
    eager_layer_out, eager_layer_in, eager_attn_out, eager_codes = run_with_attn("eager")

    # Compare codes
    print("\n" + "="*70)
    print("CODEC CODE COMPARISON")
    print("="*70)
    n_frames = min(len(sdpa_codes), len(eager_codes))
    for f in range(n_frames):
        sc = sdpa_codes[f]
        ec = eager_codes[f]
        match = "OK" if sc == ec else "DIFFER"
        if match == "DIFFER":
            print(f"  frame {f}: {match}")
            print(f"    SDPA:  {sc}")
            print(f"    EAGER: {ec}")
        else:
            print(f"  frame {f}: {match} codes={sc}")

    # Compare layer outputs at first forward call (prefill)
    print("\n" + "="*70)
    print("LAYER OUTPUT COMPARISON (first forward call = prefill)")
    print("="*70)

    n_layers = len(sdpa_layer_out)
    for layer_idx in range(n_layers):
        if layer_idx in sdpa_layer_out and layer_idx in eager_layer_out:
            s = sdpa_layer_out[layer_idx][0]  # first forward call
            e = eager_layer_out[layer_idx][0]
            diff = (s - e).abs()
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()
            cos_sim = torch.nn.functional.cosine_similarity(s.unsqueeze(0), e.unsqueeze(0)).item()
            status = "OK" if max_diff < 1e-4 else "DIVERGE"
            if status == "DIVERGE" or layer_idx < 3 or layer_idx >= n_layers - 2:
                print(f"  Layer {layer_idx:2d}: max_diff={max_diff:.6e}  mean_diff={mean_diff:.6e}  cos_sim={cos_sim:.6f}  {status}")

    # Compare attention outputs at first forward call
    print("\n" + "="*70)
    print("ATTENTION OUTPUT COMPARISON (first forward call = prefill)")
    print("="*70)

    for layer_idx in range(n_layers):
        if layer_idx in sdpa_attn_out and layer_idx in eager_attn_out:
            s = sdpa_attn_out[layer_idx][0]
            e = eager_attn_out[layer_idx][0]
            diff = (s - e).abs()
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()
            cos_sim = torch.nn.functional.cosine_similarity(s.unsqueeze(0), e.unsqueeze(0)).item()
            status = "OK" if max_diff < 1e-4 else "DIVERGE"
            if status == "DIVERGE" or layer_idx < 3 or layer_idx >= n_layers - 2:
                print(f"  Layer {layer_idx:2d} attn: max_diff={max_diff:.6e}  mean_diff={mean_diff:.6e}  cos_sim={cos_sim:.6f}  {status}")

    # Compare at second forward call (first decode step)
    print("\n" + "="*70)
    print("LAYER OUTPUT COMPARISON (second forward call = first decode step)")
    print("="*70)
    for layer_idx in range(n_layers):
        if layer_idx in sdpa_layer_out and layer_idx in eager_layer_out:
            if len(sdpa_layer_out[layer_idx]) > 1 and len(eager_layer_out[layer_idx]) > 1:
                s = sdpa_layer_out[layer_idx][1]
                e = eager_layer_out[layer_idx][1]
                diff = (s - e).abs()
                max_diff = diff.max().item()
                cos_sim = torch.nn.functional.cosine_similarity(s.unsqueeze(0), e.unsqueeze(0)).item()
                status = "OK" if max_diff < 1e-4 else "DIVERGE"
                if status == "DIVERGE" or layer_idx < 3 or layer_idx >= n_layers - 2:
                    print(f"  Layer {layer_idx:2d}: max_diff={max_diff:.6e}  cos_sim={cos_sim:.6f}  {status}")


if __name__ == "__main__":
    main()
