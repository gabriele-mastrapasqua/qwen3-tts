#!/usr/bin/env python3
"""Generate audio with Python ref and analyze frequency content."""
import torch
import numpy as np
import soundfile as sf

MODEL_DIR = "/Users/gabrielemastrapasqua/source/personal/qwen-tts/qwen3-tts-0.6b"

def analyze_freq(wav, sr, label):
    """Analyze frequency content of audio."""
    from scipy import signal
    freqs, psd = signal.welch(wav, sr, nperseg=2048)
    total = psd.sum()
    if total == 0:
        print(f"  {label}: SILENT")
        return

    bands = [
        ("Sub-bass (20-60Hz)", 20, 60),
        ("Bass (60-250Hz)", 60, 250),
        ("Low-mid (250-500Hz)", 250, 500),
        ("Mid (500-2kHz)", 500, 2000),
        ("Upper-mid (2-4kHz)", 2000, 4000),
        ("High (4-8kHz)", 4000, 8000),
        ("Very high (8-12kHz)", 8000, 12000),
    ]
    print(f"  {label}:")
    print(f"    Duration: {len(wav)/sr:.2f}s, RMS: {np.sqrt(np.mean(wav**2)):.4f}, Peak: {np.max(np.abs(wav)):.4f}")
    for name, lo, hi in bands:
        mask = (freqs >= lo) & (freqs < hi)
        pct = 100 * psd[mask].sum() / total
        print(f"    {name}: {pct:.1f}%")


def generate_and_analyze(text, speaker, language, label, gen_params=None):
    """Generate audio and analyze."""
    from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel

    default_params = dict(
        temperature=0.9,
        top_k=50,
        top_p=1.0,
        do_sample=True,
        subtalker_temperature=0.9,
        subtalker_top_k=50,
        subtalker_top_p=1.0,
        subtalker_dosample=True,
        repetition_penalty=1.05,
        max_new_tokens=2048,
    )
    if gen_params:
        default_params.update(gen_params)

    model = Qwen3TTSModel.from_pretrained(
        MODEL_DIR, torch_dtype=torch.float32, device_map="cpu",
        attn_implementation="eager",
    )

    print(f"\n{'='*60}")
    print(f"Generating: {label}")
    print(f"  text={text}, speaker={speaker}, lang={language}")
    print(f"  params={default_params}")

    wavs, sr = model.generate_custom_voice(
        text=text, speaker=speaker, language=language, **default_params
    )

    wav = wavs[0] if isinstance(wavs, list) else wavs
    if torch.is_tensor(wav):
        wav = wav.squeeze().numpy()

    out_path = f"/tmp/debug_{label}.wav"
    sf.write(out_path, wav, sr)
    print(f"  Saved to {out_path}")
    analyze_freq(wav, sr, label)
    return wav, sr


def main():
    # Test 1: English with Serena (greedy)
    generate_and_analyze(
        "Hello world, how are you doing today?", "Serena", "English",
        "eng_serena_greedy",
        gen_params=dict(temperature=0.0, top_k=1, do_sample=False,
                       subtalker_temperature=0.0, subtalker_top_k=1,
                       subtalker_dosample=False, repetition_penalty=1.0)
    )

    # Test 2: English with Serena (sampling, defaults)
    generate_and_analyze(
        "Hello world, how are you doing today?", "Serena", "English",
        "eng_serena_sampling",
    )

    # Test 3: Chinese with Serena (greedy - native language)
    generate_and_analyze(
        "你好世界，今天你好吗？", "Serena", "Chinese",
        "zho_serena_greedy",
        gen_params=dict(temperature=0.0, top_k=1, do_sample=False,
                       subtalker_temperature=0.0, subtalker_top_k=1,
                       subtalker_dosample=False, repetition_penalty=1.0)
    )

    # Test 4: English with Ryan (native English speaker)
    generate_and_analyze(
        "Hello world, how are you doing today?", "Ryan", "English",
        "eng_ryan_greedy",
        gen_params=dict(temperature=0.0, top_k=1, do_sample=False,
                       subtalker_temperature=0.0, subtalker_top_k=1,
                       subtalker_dosample=False, repetition_penalty=1.0)
    )


if __name__ == "__main__":
    main()
