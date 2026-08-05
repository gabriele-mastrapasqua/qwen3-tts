#!/usr/bin/env python3
"""Group para-event VARIANTS instead of classifying them.

Why this exists: `para_judge.py` (CNN14 / CLAP) answers "is this a sigh?" and on our material it
answers wrong — measured 2026-08-05 on the 0.6B seed sweep, every clip the ear calls a TOP sigh
comes back MISS with P<0.05, on both backends. Those taggers are trained on isolated, salient
AudioSet-style events; a 0.4 s "ehhh" inside an Italian sentence is drowned by the speech.

So we drop the naming problem, which is the hard one, and keep the useful one: **the same [tag]
with different seeds produces different VARIANTS of the event** (user, 2026-08-05: on the 1.7B too).
Hand-mapping those variants by ear is what makes tag discovery so slow. This tool clusters them, so
you listen to ONE representative per behaviour instead of every seed.

It describes the event region acoustically (no pretrained tagger, no labels):
  - F0 contour stats     -> pitch shape of the event ("eh!" rises, "ahhh" falls)
  - RMS envelope stats   -> attack/decay ("eh!" is short+sharp, "awhhh" is long+flat)
  - spectral centroid    -> brightness (laugh is brighter than a sigh)
  - voiced fraction, event duration up to the first pause

Use:
  ./.venv/bin/python para_variants.py --wavs <dir> --pattern 'sigh_*.wav' [--k 2] [--window 1.2]
"""
import argparse, glob, os, sys
import numpy as np

try:
    import librosa
except ImportError:
    sys.exit("need librosa: ./.venv/bin/pip install librosa")


def features(path, window):
    y, sr = librosa.load(path, sr=24000, duration=window)
    if y.size == 0:
        return None
    rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=256)[0]
    cen = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=256)[0]
    f0, voiced, _ = librosa.pyin(y, sr=sr, fmin=60, fmax=500,
                                 frame_length=1024, hop_length=256)
    f0v = f0[~np.isnan(f0)]
    # where the event stops: first frame under 20% of peak energy, after the onset
    thr = 0.2 * rms.max() if rms.max() > 0 else 0
    onset = int(np.argmax(rms > thr)) if thr else 0
    after = np.where(rms[onset:] < thr)[0]
    dur = (after[0] if after.size else len(rms) - onset) * 256 / sr

    def slope(x):
        return float(np.polyfit(np.arange(len(x)), x, 1)[0]) if len(x) > 2 else 0.0

    return np.array([
        float(np.log(f0v.mean() + 1e-6)) if f0v.size else 0.0,
        float(np.log(f0v.std() + 1e-6)) if f0v.size else 0.0,
        slope(np.log(f0v + 1e-6)) * 100 if f0v.size > 2 else 0.0,
        float(rms.mean()), float(rms.std()), slope(rms) * 100,
        float(cen.mean() / 1000), float(cen.std() / 1000),
        float(voiced.mean()), dur,
    ])


NAMES = ["f0_mean", "f0_std", "f0_slope", "rms_mean", "rms_std", "rms_slope",
         "centroid", "centroid_std", "voiced", "dur"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wavs", required=True)
    ap.add_argument("--pattern", default="*.wav")
    ap.add_argument("--window", type=float, default=1.2,
                    help="seconds from the start to analyse (the event sits at the head)")
    ap.add_argument("--k", type=int, default=0, help="clusters (0 = just print the distance matrix)")
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.wavs, a.pattern)))
    if not files:
        sys.exit(f"no wav matching {a.pattern} in {a.wavs}")
    X, keep = [], []
    for f in files:
        v = features(f, a.window)
        if v is not None:
            X.append(v); keep.append(f)
    X = np.array(X)
    Z = (X - X.mean(0)) / (X.std(0) + 1e-9)          # z-score so no feature dominates

    print(f"\n{'clip':28s}" + "".join(f"{n:>11s}" for n in NAMES))
    for f, row in zip(keep, X):
        print(f"  {os.path.basename(f)[:-4]:26s}" + "".join(f"{v:11.2f}" for v in row))

    D = np.sqrt(((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1))
    print(f"\ndistanza fra varianti (piu' alta = piu' diverse)\n{'':28s}" +
          "".join(f"{os.path.basename(f)[:-4][-6:]:>9s}" for f in keep))
    for f, row in zip(keep, D):
        print(f"  {os.path.basename(f)[:-4]:26s}" + "".join(f"{v:9.2f}" for v in row))

    if a.k > 1:
        rng = np.random.RandomState(0)
        C = Z[rng.choice(len(Z), a.k, replace=False)]
        for _ in range(50):
            lab = np.argmin(((Z[:, None] - C[None]) ** 2).sum(-1), axis=1)
            for j in range(a.k):
                if (lab == j).any():
                    C[j] = Z[lab == j].mean(0)
        print(f"\n{a.k} gruppi — ascolta UN rappresentante per gruppo:")
        for j in range(a.k):
            members = [os.path.basename(keep[i])[:-4] for i in range(len(keep)) if lab[i] == j]
            if not members:
                continue
            idx = [i for i in range(len(keep)) if lab[i] == j]
            rep = keep[idx[int(np.argmin(((Z[idx] - C[j]) ** 2).sum(-1)))]]
            print(f"  gruppo {j}: {', '.join(members)}   -> ascolta {os.path.basename(rep)}")


if __name__ == "__main__":
    main()
