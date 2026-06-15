#!/usr/bin/env python3
"""Interpretable ACOUSTIC emotion-distance: how far each emotion sits from a voice's own
neutral, in physical units (energy dB, pitch Hz, speed, brightness), and whether two voices
(e.g. preset ryan vs cloned galatea) move the SAME distance in the SAME direction.

Why audio features (not hidden-state cosine): the act-map residual cosine is noisy
(noise-floor ~0.48); these features are robust + readable and are what the ear actually hears.

Usage:
  tests/emo_distance.py LABEL neutral.wav emo1=clip1.wav emo2=clip2.wav ...
  (run once per voice; then eyeball the two tables, or pass --json to combine)
"""
import sys, json, argparse, math
import numpy as np, librosa

SR = 24000
def feats(path):
    y, _ = librosa.load(path, sr=SR, mono=True)
    dur = len(y) / SR
    rms = float(np.mean(librosa.feature.rms(y=y)))
    rms_db = 20 * math.log10(rms + 1e-9)
    f0, vflag, _ = librosa.pyin(y, fmin=70, fmax=400, sr=SR)
    f0v = f0[~np.isnan(f0)]
    f0_mean = float(np.mean(f0v)) if len(f0v) else 0.0
    f0_std = float(np.std(f0v)) if len(f0v) else 0.0
    cen = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=SR)))
    voiced = float(np.mean(vflag)) if vflag is not None else 0.0
    return {"dur": dur, "rms_db": rms_db, "f0_mean": f0_mean, "f0_std": f0_std,
            "centroid": cen, "voiced": voiced}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("label"); ap.add_argument("neutral")
    ap.add_argument("emos", nargs="+", help="emo=clip.wav")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    base = feats(a.neutral)
    rows = {}
    for spec in a.emos:
        name, path = spec.split("=", 1)
        rows[name] = feats(path)
    # readable deltas vs neutral
    print(f"\n=== {a.label}  (neutral: dur {base['dur']:.2f}s  energy {base['rms_db']:.1f}dB  "
          f"F0 {base['f0_mean']:.0f}±{base['f0_std']:.0f}Hz  bright {base['centroid']:.0f}Hz) ===")
    print(f"{'emotion':<10} {'Δdur(s)':>8} {'Δenergy(dB)':>11} {'ΔF0(Hz)':>8} {'ΔF0std':>7} {'Δbright':>8}   dist*")
    # normalization scales (rough, for the combined 'dist'): per-feature spread
    scales = {"dur":0.5, "rms_db":3.0, "f0_mean":20.0, "f0_std":15.0, "centroid":300.0}
    out = {"label": a.label, "neutral": base, "emotions": {}}
    for name, fv in rows.items():
        d = {k: fv[k]-base[k] for k in base}
        dist = math.sqrt(sum((d[k]/scales[k])**2 for k in scales))
        print(f"{name:<10} {d['dur']:>+8.2f} {d['rms_db']:>+11.1f} {d['f0_mean']:>+8.0f} "
              f"{d['f0_std']:>+7.0f} {d['centroid']:>+8.0f}   {dist:.2f}")
        out["emotions"][name] = {"delta": d, "dist": dist, "feats": fv}
    if a.json:
        print("JSON:" + json.dumps(out))

if __name__ == "__main__":
    main()
