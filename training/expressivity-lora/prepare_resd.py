#!/usr/bin/env python3
# LICENSE: RESD is MIT -> research, commercial, and redistribution of derived LoRA weights
# are all permitted. Ship-safe. See DATASETS.md.
import argparse, io, json, os
from collections import Counter

import librosa
import pyarrow.parquet as pq
import soundfile as sf
from huggingface_hub import hf_hub_download

EMOTION_INSTRUCT = {
    "neutral":    "",
    "happiness":  "Speak happily, bright and warm, smiling through the words.",
    "anger":      "Speak with hot, furious anger, sharp and forceful.",
    "sadness":    "Speak with a sad, sorrowful, downcast tone, voice low and heavy.",
    "fear":       "Speak with fear, tense and trembling, your voice wary.",
    "disgust":    "Speak with physical disgust, repulsed and recoiling.",
    "enthusiasm": "Speak with bright, eager enthusiasm, energetic and excited.",
}

PARQUET = {
    "train": "data/train-00000-of-00001-1f5fe73d1293189c.parquet",
    "test":  "data/test-00000-of-00001-a2b788d59856c4ae.parquet",
}
REPO = "Aniemore/resd_annotated"

def iter_rows(split):
    path = hf_hub_download(REPO, PARQUET[split], repo_type="dataset")
    cols = pq.ParquetFile(path).read(columns=["name", "text", "emotion", "speech"]).to_pydict()
    for name, text, emo, speech in zip(cols["name"], cols["text"], cols["emotion"], cols["speech"]):
        yield name, text, emo, speech["bytes"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train", choices=["train", "test", "all"],
                    help="RESD split; 'all' merges train+test for a bit more data (~3.5h total)")
    ap.add_argument("--out_dir", default="data_resd")
    ap.add_argument("--histogram", action="store_true", help="print emotion counts and exit")
    args = ap.parse_args()

    splits = ["train", "test"] if args.split == "all" else [args.split]

    if args.histogram:
        c = Counter()
        for sp in splits:
            for _n, _t, emo, _b in iter_rows(sp):
                c[emo] += 1
        print("RESD emotion counts:", dict(c))
        return

    wav_dir = os.path.join(args.out_dir, "wav24k"); os.makedirs(wav_dir, exist_ok=True)
    out_jsonl = os.path.join(args.out_dir, "train_raw.jsonl")
    rows, skipped = [], 0
    for sp in splits:
        for name, text, emo, raw in iter_rows(sp):
            if emo not in EMOTION_INSTRUCT or not (text or "").strip():
                skipped += 1; continue
            out = os.path.join(wav_dir, f"{name}.wav")
            if not os.path.exists(out):
                y, sr = sf.read(io.BytesIO(raw))
                if y.ndim > 1:
                    y = y.mean(axis=1)
                if sr != 24000:
                    y = librosa.resample(y.astype("float32"), orig_sr=sr, target_sr=24000)
                sf.write(out, y, 24000, subtype="PCM_16")
            rows.append({"audio": out, "text": text.strip(), "ref_audio": out,
                         "instruct": EMOTION_INSTRUCT[emo], "emotion": emo})

    with open(out_jsonl, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} rows (skipped {skipped} unknown-emotion/empty-text) -> {out_jsonl}")
    print("emotions:", dict(Counter(r["emotion"] for r in rows)))
    print("NEXT: run the upstream prepare_data.py on this jsonl to add audio_codes,")
    print("      then train_lora.py --layers 0-27 (broad band) r32. See DATASETS.md.")

if __name__ == "__main__":
    main()
