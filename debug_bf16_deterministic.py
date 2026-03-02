#!/usr/bin/env python3
"""Deterministic comparison of bf16 vs f32 to isolate precision effect."""
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


def test(dtype, dtype_name, text, speaker, language, label):
    from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel

    greedy_params = dict(
        temperature=0.0,
        top_k=1,
        top_p=1.0,
        do_sample=False,
        subtalker_temperature=0.0,
        subtalker_top_k=1,
        subtalker_top_p=1.0,
        subtalker_dosample=False,
        repetition_penalty=1.0,
        max_new_tokens=200,  # limit to avoid infinite generation
    )

    print(f"\n--- {dtype_name} : {label} ---")
    model = Qwen3TTSModel.from_pretrained(
        MODEL_DIR, dtype=dtype, device_map="cpu",
        attn_implementation="eager",
    )

    # Capture codes for comparison
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
    out_path = f"/tmp/debug_det_{dtype_name}_{label}.wav"
    sf.write(out_path, wav, sr)
    analyze_freq(wav, sr, f"{dtype_name}_{label}")
    print(f"  Frames: {len(frame_codes)}")
    if frame_codes:
        # Print first 5 frames
        for i in range(min(5, len(frame_codes))):
            print(f"    frame {i}: {frame_codes[i][:4]}...")
    del model
    return wav, frame_codes


def main():
    # Test 1: Italian with Serena (where we saw the biggest difference)
    text = "Ciao mondo, come stai oggi?"
    w_f32, c_f32 = test(torch.float32, "f32", text, "Serena", "Italian", "ita_serena")
    w_bf16, c_bf16 = test(torch.bfloat16, "bf16", text, "Serena", "Italian", "ita_serena")

    # Compare codes
    print(f"\n  Code comparison (f32 vs bf16):")
    n_frames = min(len(c_f32), len(c_bf16))
    diverge_at = -1
    for i in range(n_frames):
        if c_f32[i] != c_bf16[i]:
            diverge_at = i
            print(f"    First divergence at frame {i}:")
            print(f"      f32:  {c_f32[i]}")
            print(f"      bf16: {c_bf16[i]}")
            break
    if diverge_at == -1:
        print(f"    All {n_frames} frames identical!")
    else:
        # Count how many differ
        differ = sum(1 for i in range(n_frames) if c_f32[i] != c_bf16[i])
        print(f"    {differ}/{n_frames} frames differ")

    # Test 2: English with Ryan
    print("\n" + "="*60)
    text2 = "Hello world, how are you doing today?"
    w2_f32, c2_f32 = test(torch.float32, "f32", text2, "Ryan", "English", "eng_ryan")
    w2_bf16, c2_bf16 = test(torch.bfloat16, "bf16", text2, "Ryan", "English", "eng_ryan")

    print(f"\n  Code comparison (f32 vs bf16):")
    n_frames = min(len(c2_f32), len(c2_bf16))
    diverge_at = -1
    for i in range(n_frames):
        if c2_f32[i] != c2_bf16[i]:
            diverge_at = i
            print(f"    First divergence at frame {i}:")
            print(f"      f32:  {c2_f32[i]}")
            print(f"      bf16: {c2_bf16[i]}")
            break
    if diverge_at == -1:
        print(f"    All {n_frames} frames identical!")
    else:
        differ = sum(1 for i in range(n_frames) if c2_f32[i] != c2_bf16[i])
        print(f"    {differ}/{n_frames} frames differ")


if __name__ == "__main__":
    main()
