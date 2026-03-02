#!/usr/bin/env python3
"""Test if rounding ALL linear layer outputs to bf16 fixes the issue."""
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


def install_all_linear_bf16_hooks(model):
    """Hook ALL nn.Linear modules to round their output to bf16."""
    hooks = []
    count = 0
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            # Only hook talker and code predictor, not the speech decoder
            if 'speech_tokenizer' in name or 'decoder' in name:
                continue
            def make_hook(n):
                def hook(module, input, output):
                    return output.to(torch.bfloat16).to(torch.float32)
                return hook
            hooks.append(module.register_forward_hook(make_hook(name)))
            count += 1
    print(f"  Installed {count} linear bf16 hooks")
    return hooks


def test(label, hook_fn, text, speaker, language):
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

    if hook_fn:
        hook_fn(model)

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

    # Pure bf16 reference
    print("=== Testing Italian Serena ===")
    w_bf16, c_bf16 = test("bf16_native", None, text, speaker, language)
    # Override above to use bf16 directly
    from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
    greedy_params = dict(
        temperature=0.0, top_k=1, top_p=1.0, do_sample=False,
        subtalker_temperature=0.0, subtalker_top_k=1,
        subtalker_top_p=1.0, subtalker_dosample=False,
        repetition_penalty=1.0, max_new_tokens=200,
    )
    print(f"\n--- bf16_native ---")
    model = Qwen3TTSModel.from_pretrained(
        MODEL_DIR, dtype=torch.bfloat16, device_map="cpu",
        attn_implementation="eager",
    )
    frame_codes_bf16 = []
    talker = model.model.talker
    def th(module, inp, out):
        if hasattr(out, 'hidden_states') and out.hidden_states is not None:
            _, codec_ids = out.hidden_states
            if codec_ids is not None:
                frame_codes_bf16.append(codec_ids[0].detach().cpu().tolist())
    talker.register_forward_hook(th)
    wavs, sr = model.generate_custom_voice(text=text, speaker=speaker, language=language, **greedy_params)
    wav = wavs[0] if isinstance(wavs, list) else wavs
    if torch.is_tensor(wav):
        wav = wav.squeeze().numpy()
    sf.write("/tmp/debug_bf16_native.wav", wav, sr)
    analyze_freq(wav, sr, "bf16_native")
    print(f"  Frames: {len(frame_codes_bf16)}")
    del model

    # f32 with all-linear hooks
    w_hook, c_hook = test("f32_all_linear_round", install_all_linear_bf16_hooks,
                          text, speaker, language)

    # Compare hook vs native bf16
    print(f"\n  Code comparison (all_linear_round vs native_bf16):")
    n_frames = min(len(c_hook), len(frame_codes_bf16))
    match_count = 0
    for i in range(n_frames):
        if c_hook[i] == frame_codes_bf16[i]:
            match_count += 1
    print(f"    {match_count}/{n_frames} frames match exactly")
    if n_frames > 0 and c_hook[0] != frame_codes_bf16[0]:
        print(f"    Frame 0:")
        print(f"      hook:   {c_hook[0]}")
        print(f"      native: {frame_codes_bf16[0]}")


if __name__ == "__main__":
    main()
