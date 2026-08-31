#!/usr/bin/env python3
import argparse, io, json, os, random
from collections import Counter
import numpy as np
import pyarrow.parquet as pq
import soundfile as sf, librosa
from huggingface_hub import HfApi, hf_hub_download

SR = 24000
VOCALSOUND = "lmms-lab/vocalsound"
LABEL_MARKER = {"Laughter": "[laugh]", "Sigh": "[sigh]", "Cough": "[cough]"}

def load_events(which):
    """Download VocalSound, return {marker: [float32 24k mono arrays]} for the wanted labels."""
    fs = [f for f in HfApi().list_repo_files(VOCALSOUND, repo_type="dataset") if f.endswith(".parquet")]
    want = {k: v for k, v in LABEL_MARKER.items() if v in which}
    ev = {m: [] for m in want.values()}
    for rel in fs:
        p = hf_hub_download(VOCALSOUND, rel, repo_type="dataset")
        cols = pq.ParquetFile(p).read(columns=["audio", "answer"]).to_pydict()
        for au, lab in zip(cols["audio"], cols["answer"]):
            if lab not in want: continue
            y, sr = sf.read(io.BytesIO(au["bytes"]))
            if y.ndim > 1: y = y.mean(axis=1)
            y = y.astype("float32")
            if sr != SR: y = librosa.resample(y, orig_sr=sr, target_sr=SR)
            yt, _ = librosa.effects.trim(y, top_db=30)
            ev[want[lab]].append(yt if len(yt) > SR // 10 else y)
    return ev

def splice_point(carrier):
    """Return a sample index near a low-energy pause in the middle 45-70% of the carrier."""
    n = len(carrier)
    lo, hi = int(0.45 * n), int(0.72 * n)
    if hi - lo < SR // 20: return n
    win = SR // 50
    region = carrier[lo:hi]
    e = np.array([np.sum(region[i:i+win]**2) for i in range(0, len(region) - win, win)])
    if len(e) == 0: return n
    return lo + int(np.argmin(e)) * win

def insert_marker_text(text, marker, frac):
    words = text.split()
    if len(words) < 2: return f"{text} {marker}".strip()
    k = max(1, min(len(words) - 1, round(frac * len(words))))
    return " ".join(words[:k] + [marker] + words[k:])

def xfade_concat(a, b, ms=15):
    f = int(SR * ms / 1000)
    if len(a) < f or len(b) < f: return np.concatenate([a, b])
    ramp = np.linspace(0, 1, f, dtype="float32")
    mid = a[-f:] * (1 - ramp) + b[:f] * ramp
    return np.concatenate([a[:-f], mid, b[f:]])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--carriers", required=True, help="jsonl with neutral {audio,text} rows (multilingual)")
    ap.add_argument("--out_dir", default="data_paraling_aug")
    ap.add_argument("--markers", default="[laugh],[sigh]", help="comma list of markers to teach")
    ap.add_argument("--plain-frac", type=float, default=0.30, help="fraction of carriers kept PLAIN (anchor)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rnd = random.Random(args.seed)

    markers = [m.strip() for m in args.markers.split(",")]
    print("loading VocalSound events for:", markers)
    events = load_events(set(markers))
    for m in markers: print(f"  {m}: {len(events.get(m, []))} events")

    carriers = [json.loads(l) for l in open(args.carriers)]
    rnd.shuffle(carriers)
    wav_dir = os.path.join(args.out_dir, "wav24k"); os.makedirs(wav_dir, exist_ok=True)
    out_jsonl = os.path.join(args.out_dir, "train_raw.jsonl")

    rows, used = [], Counter()
    with open(out_jsonl, "w") as f:
        for i, c in enumerate(carriers):
            try:
                y, sr = librosa.load(c["audio"], sr=SR, mono=True)
            except Exception:
                continue
            text = (c.get("text") or "").strip()
            if not text: continue
            name = f"aug_{i:06d}"
            if rnd.random() < args.plain_frac:
                out_text, out_y, emo = text, y, "neutral"
                used["[plain]"] += 1
            else:
                m = rnd.choice(markers)
                pool = events.get(m) or []
                if not pool:
                    out_text, out_y, emo = text, y, "neutral"; used["[plain]"] += 1
                else:
                    ev = rnd.choice(pool)
                    sp = splice_point(y)
                    out_y = xfade_concat(xfade_concat(y[:sp], ev), y[sp:]) if sp < len(y) else xfade_concat(y, ev)
                    out_text = insert_marker_text(text, m, sp / max(1, len(y)))
                    emo = m.strip("[]"); used[m] += 1
            out = os.path.join(wav_dir, name + ".wav")
            sf.write(out, out_y, SR, subtype="PCM_16")
            LANG_FULL = {"IT":"Italian","DE":"German","ES":"Spanish","FR":"French","RU":"Russian",
                         "KO":"Korean","JA":"Japanese","EN":"English","ZH":"Chinese","PT":"Portuguese"}
            lang = LANG_FULL.get((c.get("lang") or "").upper(), "English")
            row = {"audio": out, "text": out_text, "ref_audio": out, "instruct": "",
                   "emotion": emo, "language": lang}
            f.write(json.dumps(row, ensure_ascii=False) + "\n"); rows.append(row)

    print(f"wrote {len(rows)} rows -> {out_jsonl}")
    print("composition:", dict(used))
    print("NEXT: prepare_data.py -> codes, then train_lora.py --layers 16-26 r32 (few epochs).")

if __name__ == "__main__":
    main()
