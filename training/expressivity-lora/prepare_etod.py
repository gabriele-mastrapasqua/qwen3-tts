#!/usr/bin/env python3
# LICENSE: CC-BY-NC-SA-4.0 (research, NON-commercial). We do not redistribute the data,
# only train a micro-LoRA, so it is fine to USE; do not ship the data. See DATASETS.md.
import argparse, glob, json, os, re
from collections import Counter
import librosa, soundfile as sf

EMOTION_INSTRUCT = {
    "neutral": "",
    "happy":   "Speak happily, bright and warm, smiling through the words.",
    "angry":   "Speak with hot, furious anger, sharp and forceful.",
    "sad":     "Speak with a sad, sorrowful, downcast tone, voice low and heavy.",
}

def emo_from_num(n):
    if   1   <= n <= 100: return "neutral"
    elif 101 <= n <= 200: return "happy"
    elif 201 <= n <= 300: return "angry"
    elif 301 <= n <= 400: return "sad"
    return None

_NAME = re.compile(r"^([a-z]+)(\d{5})$")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="ETOD repo dir (contains Dataset/SpeechCorpus/Emotional)")
    ap.add_argument("--out_dir", default="data_etod")
    ap.add_argument("--histogram", action="store_true")
    args = ap.parse_args()

    wavs = sorted(glob.glob(os.path.join(args.root, "**", "Emotional", "**", "*.wav"), recursive=True))
    if not wavs:
        ap.error(f"no Emotional/**/*.wav under {args.root} (clone emotiontts_open_db and point --root at it)")

    if args.histogram:
        c = Counter()
        for w in wavs:
            m = _NAME.match(os.path.basename(w)[:-4])
            if m: c[emo_from_num(int(m.group(2)))] += 1
        print("ETOD emotion counts:", dict(c), "| total wav:", len(wavs))
        return

    wav_dir = os.path.join(args.out_dir, "wav24k"); os.makedirs(wav_dir, exist_ok=True)
    out_jsonl = os.path.join(args.out_dir, "train_raw.jsonl")
    rows, skipped = [], 0
    for w in wavs:
        base = os.path.basename(w)[:-4]
        m = _NAME.match(base)
        if not m: skipped += 1; continue
        emo = emo_from_num(int(m.group(2)))
        if emo not in EMOTION_INSTRUCT: skipped += 1; continue
        txt = None
        for sub in ("transcript", "script"):
            tp = os.path.join(os.path.dirname(os.path.dirname(w)), sub, base + ".txt")
            if os.path.exists(tp):
                txt = open(tp, encoding="utf-8-sig").read().strip(); break
        if not txt: skipped += 1; continue
        out = os.path.join(wav_dir, base + ".wav")
        if not os.path.exists(out):
            y, sr = librosa.load(w, sr=24000, mono=True)
            sf.write(out, y, 24000, subtype="PCM_16")
        rows.append({"audio": out, "text": txt, "ref_audio": out,
                     "instruct": EMOTION_INSTRUCT[emo], "emotion": emo})

    with open(out_jsonl, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} rows (skipped {skipped}) -> {out_jsonl}")
    print("emotions:", dict(Counter(r["emotion"] for r in rows)))
    print("NEXT: prepare_data.py -> audio_codes, then train_lora.py --layers 0-27 r32. See DATASETS.md.")

if __name__ == "__main__":
    main()
