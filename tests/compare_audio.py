#!/usr/bin/env python3
"""compare_audio.py — golden-reference correctness check for qwen-tts."""
import sys
import argparse
import numpy as np

try:
    import librosa
except ImportError:
    sys.stderr.write("compare_audio: librosa not installed (pip install librosa)\n")
    sys.exit(2)

SR = 24000
N_FFT = 1024
HOP = 256
N_MELS = 128

def log_mel(y):
    m = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS)
    return librosa.power_to_db(m + 1e-10)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("golden")
    ap.add_argument("output")
    ap.add_argument("--min-corr", type=float, default=0.98)
    ap.add_argument("--dur-tol", type=float, default=0.05)
    ap.add_argument("--label", default="")
    a = ap.parse_args()

    try:
        g, _ = librosa.load(a.golden, sr=SR, mono=True)
        o, _ = librosa.load(a.output, sr=SR, mono=True)
    except Exception as e:
        sys.stderr.write(f"compare_audio: load error: {e}\n")
        sys.exit(2)

    if g.size == 0 or o.size == 0:
        print(f"FAIL {a.label}: empty audio (golden={g.size} out={o.size} samples)")
        sys.exit(1)

    dur_g, dur_o = g.size / SR, o.size / SR
    dur_rel = abs(dur_o - dur_g) / dur_g

    n = min(g.size, o.size)
    mg, mo = log_mel(g[:n]), log_mel(o[:n])
    k = min(mg.shape[1], mo.shape[1])
    vg, vo = mg[:, :k].ravel(), mo[:, :k].ravel()
    corr = float(np.corrcoef(vg, vo)[0, 1]) if vg.std() > 0 and vo.std() > 0 else 0.0

    dur_ok = dur_rel <= a.dur_tol
    corr_ok = corr >= a.min_corr
    status = "PASS" if (dur_ok and corr_ok) else "FAIL"
    print(f"{status} {a.label}: mel_corr={corr:.5f} (>= {a.min_corr}) "
          f"dur={dur_o:.2f}s vs {dur_g:.2f}s (rel {dur_rel*100:.1f}% <= {a.dur_tol*100:.0f}%)")
    if status == "FAIL":
        if not corr_ok:
            sys.stderr.write(f"  mel-correlation too low: {corr:.5f} < {a.min_corr}\n")
        if not dur_ok:
            sys.stderr.write(f"  duration drift too large: {dur_rel*100:.1f}% > {a.dur_tol*100:.0f}%\n")
        sys.exit(1)
    sys.exit(0)

if __name__ == "__main__":
    main()
