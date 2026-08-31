#!/usr/bin/env python3
# LICENSE: CC-BY-SA-4.0.
import argparse, io, json, os
from collections import Counter

import librosa
import pyarrow.parquet as pq
import soundfile as sf
from huggingface_hub import HfApi, hf_hub_download

REPO = "asahi417/jvnv-emotional-speech-corpus"

EMOTION_INSTRUCT = {
    "anger":    "Speak with hot, furious anger, sharp and forceful.",
    "disgust":  "Speak with physical disgust, repulsed and recoiling.",
    "fear":     "Speak with fear, tense and trembling, your voice wary.",
    "happy":    "Speak happily, bright and warm, smiling through the words.",
    "sad":      "Speak with a sad, sorrowful, downcast tone, voice low and heavy.",
    "surprise": "Speak with surprise, startled and taken aback, held through the sentence.",
}

def parquet_paths():
    fs = HfApi().list_repo_files(REPO, repo_type="dataset")
    return sorted(f for f in fs if f.endswith(".parquet"))

def iter_rows(paths):
    for rel in paths:
        p = hf_hub_download(REPO, rel, repo_type="dataset")
        cols = pq.ParquetFile(p).read(columns=["audio", "speaker_id", "style"]).to_pydict()
        for au, spk, style in zip(cols["audio"], cols["speaker_id"], cols["style"]):
            yield spk, style, au["bytes"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="data_jvnv")
    ap.add_argument("--whisper_model", default="large-v3", help="faster-whisper model (large-v3/medium)")
    ap.add_argument("--histogram", action="store_true")
    args = ap.parse_args()

    paths = parquet_paths()
    if args.histogram:
        c = Counter(style for _spk, style, _b in iter_rows(paths))
        print("JVNV style counts:", dict(c))
        return

    import whisper
    asr = whisper.load_model(args.whisper_model, device="cuda")

    def transcribe(y16):
        return asr.transcribe(y16.astype("float32"), language="ja", fp16=True)["text"].strip()

    wav_dir = os.path.join(args.out_dir, "wav24k"); os.makedirs(wav_dir, exist_ok=True)
    out_jsonl = os.path.join(args.out_dir, "train_raw.jsonl")
    rows, skipped, i = [], 0, 0
    for spk, style, raw in iter_rows(paths):
        i += 1
        if style not in EMOTION_INSTRUCT:
            skipped += 1; continue
        y, sr = sf.read(io.BytesIO(raw))
        if y.ndim > 1: y = y.mean(axis=1)
        y = y.astype("float32")
        y16 = librosa.resample(y, orig_sr=sr, target_sr=16000) if sr != 16000 else y
        text = transcribe(y16)
        if not text:
            skipped += 1; continue
        y24 = librosa.resample(y, orig_sr=sr, target_sr=24000) if sr != 24000 else y
        name = f"{spk}_{style}_{i:05d}"
        out = os.path.join(wav_dir, name + ".wav")
        sf.write(out, y24, 24000, subtype="PCM_16")
        rows.append({"audio": out, "text": text, "ref_audio": out,
                     "instruct": EMOTION_INSTRUCT[style], "emotion": style})
        if i % 100 == 0:
            print(f"  ...{i} processed ({len(rows)} kept)", flush=True)

    with open(out_jsonl, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} rows (skipped {skipped}) -> {out_jsonl}")
    print("emotions:", dict(Counter(r["emotion"] for r in rows)))
    print("NEXT: prepare_data.py -> audio_codes, then train_lora.py --layers 0-27 r32.")

if __name__ == "__main__":
    main()
