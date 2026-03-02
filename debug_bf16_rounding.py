#!/usr/bin/env python3
"""Test if rounding float32 activations to bf16 between layers fixes audio quality.
Theory: model needs bf16 precision (rounding) at layer boundaries to work correctly."""
import torch
import numpy as np
import soundfile as sf

MODEL_DIR = "/Users/gabrielemastrapasqua/source/personal/qwen-tts/qwen3-tts-0.6b"

def analyze_freq(wav, sr, label):
    from scipy import signal
    freqs, psd = signal.welch(wav, sr, nperseg=2048)
    total = psd.sum()
    if total == 0:
        print(f"  {label}: SILENT")
        return
    lo_pct = 100 * psd[(freqs >= 50) & (freqs < 500)].sum() / total
    mid_pct = 100 * psd[(freqs >= 500) & (freqs < 4000)].sum() / total
    hi_pct = 100 * psd[(freqs >= 4000) & (freqs < 12000)].sum() / total
    print(f"  {label}: dur={len(wav)/sr:.2f}s RMS={np.sqrt(np.mean(wav**2)):.4f} Peak={np.max(np.abs(wav)):.4f} Low={lo_pct:.1f}% Mid={mid_pct:.1f}% High={hi_pct:.1f}%")


def install_bf16_rounding_hooks(model):
    """Install hooks on all transformer layers to round outputs to bf16 precision."""
    hooks = []

    # Hook the Talker layers
    talker = model.model.talker
    for i, layer in enumerate(talker.model.layers):
        def make_hook(idx):
            def hook(module, input, output):
                # output is (hidden_states, attn_weights) for decoder layers
                if isinstance(output, tuple):
                    h = output[0]
                    h_bf16 = h.to(torch.bfloat16).to(torch.float32)
                    return (h_bf16,) + output[1:]
                else:
                    return output.to(torch.bfloat16).to(torch.float32)
            return hook
        hooks.append(layer.register_forward_hook(make_hook(i)))

    # Hook the Code Predictor layers
    cp = talker.code_predictor
    for i, layer in enumerate(cp.model.layers):
        def make_cp_hook(idx):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    h = output[0]
                    h_bf16 = h.to(torch.bfloat16).to(torch.float32)
                    return (h_bf16,) + output[1:]
                else:
                    return output.to(torch.bfloat16).to(torch.float32)
            return hook
        hooks.append(layer.register_forward_hook(make_cp_hook(i)))

    print(f"  Installed {len(hooks)} bf16 rounding hooks")
    return hooks


def test(label, use_rounding, text, speaker, language):
    from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel

    greedy_params = dict(
        temperature=0.0, top_k=1, top_p=1.0, do_sample=False,
        subtalker_temperature=0.0, subtalker_top_k=1,
        subtalker_top_p=1.0, subtalker_dosample=False,
        repetition_penalty=1.0, max_new_tokens=200,
    )

    print(f"\n--- {label} ---")
    model = Qwen3TTSModel.from_pretrained(
        MODEL_DIR, dtype=torch.float32, device_map="cpu",
        attn_implementation="eager",
    )

    if use_rounding:
        install_bf16_rounding_hooks(model)

    # Capture codes
    frame_codes = []
    talker = model.model.talker
    def talker_hook(module, inp, out):
        if hasattr(out, 'hidden_states') and out.hidden_states is not None:
            _, codec_ids = out.hidden_states
            if codec_ids is not None:
                frame_codes.append(codec_ids[0].detach().cpu().tolist())
    talker.register_forward_hook(talker_hook)

    wavs, sr = model.generate_custom_voice(
        text=text, speaker=speaker, language=language, **greedy_params
    )
    wav = wavs[0] if isinstance(wavs, list) else wavs
    if torch.is_tensor(wav):
        wav = wav.squeeze().numpy()
    out_path = f"/tmp/debug_{label}.wav"
    sf.write(out_path, wav, sr)
    analyze_freq(wav, sr, label)
    print(f"  Frames: {len(frame_codes)}")
    del model
    return wav, frame_codes


def main():
    text = "Ciao mondo, come stai oggi?"
    speaker = "Serena"
    language = "Italian"

    # Test 1: Pure float32 (bad)
    w1, c1 = test("f32_no_round", False, text, speaker, language)

    # Test 2: Float32 with bf16 rounding (should be better)
    w2, c2 = test("f32_bf16_round", True, text, speaker, language)

    # Compare codes
    print(f"\n  Code comparison (no_round vs bf16_round):")
    n_frames = min(len(c1), len(c2))
    for i in range(min(5, n_frames)):
        match = "OK" if c1[i] == c2[i] else "DIFFER"
        print(f"    frame {i}: {match}")
        if match == "DIFFER":
            print(f"      no_round: {c1[i]}")
            print(f"      bf16_rnd: {c2[i]}")
    differ = sum(1 for i in range(n_frames) if c1[i] != c2[i])
    print(f"    Total: {differ}/{n_frames} frames differ")
    print(f"    f32 frames: {len(c1)}, bf16_round frames: {len(c2)}")


if __name__ == "__main__":
    main()
