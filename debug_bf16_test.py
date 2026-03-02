#!/usr/bin/env python3
"""Test audio quality with bfloat16 vs float32."""
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


def test_dtype(dtype, dtype_name, text, speaker, language, label, gen_params):
    from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
    print(f"\n--- {dtype_name} : {label} ---")
    model = Qwen3TTSModel.from_pretrained(
        MODEL_DIR, dtype=dtype, device_map="cpu",
        attn_implementation="eager",
    )
    wavs, sr = model.generate_custom_voice(
        text=text, speaker=speaker, language=language, **gen_params
    )
    wav = wavs[0] if isinstance(wavs, list) else wavs
    if torch.is_tensor(wav):
        wav = wav.squeeze().numpy()
    out_path = f"/tmp/debug_{dtype_name}_{label}.wav"
    sf.write(out_path, wav, sr)
    analyze_freq(wav, sr, f"{dtype_name}_{label}")
    del model
    return wav


def main():
    # Use sampling (defaults) - more realistic
    sampling_params = dict(
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

    # Test English with Ryan (native English speaker)
    text = "Hello world, how are you doing today?"
    speaker = "Ryan"
    language = "English"

    test_dtype(torch.float32, "f32", text, speaker, language, "eng_ryan", sampling_params)
    test_dtype(torch.bfloat16, "bf16", text, speaker, language, "eng_ryan", sampling_params)

    # Test Italian with Serena
    text2 = "Ciao mondo, come stai oggi?"
    test_dtype(torch.float32, "f32", text2, "Serena", "Italian", "ita_serena", sampling_params)
    test_dtype(torch.bfloat16, "bf16", text2, "Serena", "Italian", "ita_serena", sampling_params)


if __name__ == "__main__":
    main()
