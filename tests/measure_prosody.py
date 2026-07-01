#!/usr/bin/env python3
"""measure_prosody.py — objective prosody metrics for the structured-instruct test (§8.8).

Two modes:
  per-file metrics (default):
      measure_prosody.py a.wav [b.wav ...]
      -> TSV: file  dur_s  f0_med_hz  f0_std_hz  rms_db
      (dur for Tempo slot, f0_med for Pitch slot, rms_db for Intensity slot, f0_std for Expression slot)

  mel-movement (how far a clip moved from a neutral anchor of the SAME text):
      measure_prosody.py --move neutral.wav out.wav
      -> prints: <mel_corr>   (LOWER corr = moved MORE; 1.0 = identical)

Why these metrics: they let us check whether each structured slot actually DOES something —
Tempo:+X% should shorten dur_s, Pitch:higher should raise f0_med, Intensity:high should raise
rms_db, Expression should change f0_std/timbre. mel-movement isolates the emotional shift the
instruct produces over the same sentence rendered flat.
"""
import sys, os
import numpy as np

try:
    import librosa
except ImportError:
    sys.stderr.write("measure_prosody: librosa not installed (pip install librosa)\n")
    sys.exit(2)

SR = 24000
N_FFT = 1024
HOP = 256
N_MELS = 128


def per_file(path):
    y, _ = librosa.load(path, sr=SR, mono=True)
    dur = len(y) / SR
    rms = float(np.sqrt(np.mean(y ** 2)) + 1e-9)
    rms_db = 20.0 * np.log10(rms)
    f0_med = f0_std = float("nan")
    try:
        f0, _, _ = librosa.pyin(y, fmin=65, fmax=400, sr=SR, frame_length=2048)
        f0v = f0[np.isfinite(f0)]
        if f0v.size:
            f0_med = float(np.median(f0v))
            f0_std = float(np.std(f0v))
    except Exception:
        pass
    return dur, f0_med, f0_std, rms_db


def log_mel(path):
    y, _ = librosa.load(path, sr=SR, mono=True)
    S = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS)
    return librosa.power_to_db(S + 1e-10)


def mel_move(ref, out):
    a, b = log_mel(ref), log_mel(out)
    n = min(a.shape[1], b.shape[1])
    a, b = a[:, :n].ravel(), b[:, :n].ravel()
    if a.size == 0 or b.size == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "--move":
        print(f"{mel_move(args[1], args[2]):.4f}")
        sys.exit(0)
    paths = [a for a in args if a != "--header"]
    if "--header" in args or not paths:
        print("file\tdur_s\tf0_med_hz\tf0_std_hz\trms_db")
    for p in paths:
        d, fm, fs, r = per_file(p)
        print(f"{os.path.basename(p)}\t{d:.2f}\t{fm:.1f}\t{fs:.1f}\t{r:.1f}")
