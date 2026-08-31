#!/usr/bin/env python3
import argparse, io, json, os
import librosa, soundfile as sf
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

REPO = "japanese-asr/ja_asr.jsut_basic5000"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="data_jvnv", help="JVNV out_dir to APPEND neutral into")
    ap.add_argument("--limit", type=int, default=300, help="number of neutral clips to add")
    args = ap.parse_args()

    fs = sorted(f for f in HfApi().list_repo_files(REPO, repo_type="dataset") if f.endswith(".parquet"))
    wav_dir = os.path.join(args.out_dir, "wav24k"); os.makedirs(wav_dir, exist_ok=True)
    out_jsonl = os.path.join(args.out_dir, "train_raw.jsonl")

    added, i = 0, 0
    new_rows = []
    for rel in fs:
        if added >= args.limit: break
        p = hf_hub_download(REPO, rel, repo_type="dataset")
        cols = pq.ParquetFile(p).read(columns=["audio", "transcription"]).to_pydict()
        for au, txt in zip(cols["audio"], cols["transcription"]):
            if added >= args.limit: break
            i += 1
            txt = (txt or "").strip()
            if not txt: continue
            y, sr = sf.read(io.BytesIO(au["bytes"]))
            if y.ndim > 1: y = y.mean(axis=1)
            y = y.astype("float32")
            if sr != 24000: y = librosa.resample(y, orig_sr=sr, target_sr=24000)
            name = f"jsut_neutral_{i:05d}"
            out = os.path.join(wav_dir, name + ".wav")
            sf.write(out, y, 24000, subtype="PCM_16")
            new_rows.append({"audio": out, "text": txt, "ref_audio": out,
                             "instruct": "", "emotion": "neutral"})
            added += 1

    with open(out_jsonl, "a") as f:
        for r in new_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    total = sum(1 for _ in open(out_jsonl))
    print(f"appended {added} neutral rows -> {out_jsonl} (now {total} total)")

if __name__ == "__main__":
    main()
