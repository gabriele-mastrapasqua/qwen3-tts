#!/usr/bin/env python3
# LICENSE WARNING: NonverbalTTS annotations are CC BY-NC-SA (NonCommercial) and the audio
# inherits VoxCeleb/Expresso terms — RESEARCH/PROTOTYPE ONLY, not shippable. The HF repo's
# `apache-2.0` tag is contradicted by the README/paper. Pick a shippable base before any
# release. See DATASETS.md.
import argparse, io, json, os, re
from collections import Counter

EMOJI_MARKER = {
    "\U0001F923": "[laugh]",
    "\U0001F32C": "[breath]",
    "\U0001F624": "[sigh]",
    "\U0001F637": "[cough]",
    "\U0001F443": "[sniff]",
    "\U0001F616": "[grunt]",
    "\U0001F927": "[sneeze]",
    "\U0001F634": "[yawn]",
}

_EMOJI_RE = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF\U00002B00-\U00002BFF]"
    "[\U0000FE00-\U0000FE0F\U0001F3FB-\U0001F3FF\U0000200D]*"
)
_STRIP = re.compile("[\U0000FE00-\U0000FE0F\U0001F3FB-\U0001F3FF\U0000200D]")

def map_text(text, hist, kept, dropped):
    """Replace inline emoji with markers; strip unknowns. Mutates the counters.
    Returns (clean_text, markers_present_set)."""
    out, i, markers = [], 0, set()
    for m in _EMOJI_RE.finditer(text):
        out.append(text[i:m.start()])
        base = _STRIP.sub("", m.group(0))
        hist[base] += 1
        mark = EMOJI_MARKER.get(base)
        if mark:
            out.append(f" {mark} ")
            kept[base] += 1
            markers.add(mark)
        else:
            dropped[base] += 1
        i = m.end()
    out.append(text[i:])
    return re.sub(r"\s+", " ", "".join(out)).strip(), markers

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", default="deepvk/NonverbalTTS", help="HF dataset id")
    ap.add_argument("--split", default="train")
    ap.add_argument("--text-col", default="Result", help="consensus transcript column")
    ap.add_argument("--emotion-col", default="Emotion")
    ap.add_argument("--min-dnsmos", type=float, default=3.0,
                    help="drop clips below this DNSMOS (audio cleanliness); 0 = keep all")
    ap.add_argument("--max-rows", type=int, default=0, help="cap rows for a quick smoke (0 = all)")
    ap.add_argument("--keep-unmapped", action="store_true",
                    help="keep rows whose ONLY nonverbal emoji were unmapped/stripped")
    ap.add_argument("--cap-per-marker", type=int, default=0,
                    help="balance: cap clips per marker; skip a clip if ALL its markers are at cap "
                         "(0 = off). Tames the breath-heavy distribution that over-forces the LoRA.")
    ap.add_argument("--neutral", type=int, default=0,
                    help="ANCHOR: keep up to N marker-FREE (plain) clips as emotion=neutral/instruct='' "
                         "(0 = old behaviour: drop all plain clips). The plain baseline = 'force less'.")
    ap.add_argument("--histogram", action="store_true",
                    help="scan emoji frequency and EXIT (no audio written) — run this first")
    ap.add_argument("--out_dir", default="data_nv")
    args = ap.parse_args()

    from datasets import load_dataset, Audio
    ds = load_dataset(args.hf, split=args.split).cast_column("audio", Audio(decode=False))

    hist, kept, dropped = Counter(), Counter(), Counter()

    if args.histogram:
        for ex in ds:
            map_text(ex[args.text_col] or "", hist, kept, dropped)[0]
        print(f"emoji histogram over {len(ds)} rows of {args.hf}[{args.split}]:")
        for cp, n in hist.most_common():
            mark = EMOJI_MARKER.get(cp, "  (UNMAPPED -> stripped)")
            print(f"  U+{ord(cp):05X} {cp}  x{n:<6} -> {mark}")
        print(f"\nmapped occurrences: {sum(kept.values())} | stripped: {sum(dropped.values())}")
        return

    import librosa, soundfile as sf
    wav_dir = os.path.join(args.out_dir, "wav24k"); os.makedirs(wav_dir, exist_ok=True)
    out_jsonl = os.path.join(args.out_dir, "train_raw.jsonl")
    skipped_q, skipped_nonv, skipped_cap = 0, 0, 0
    n_written, n_neutral = 0, 0
    marker_clips = Counter()

    def write_clip(ex, idx, text, emo):
        a = ex["audio"]
        wav, sr = sf.read(io.BytesIO(a["bytes"]), dtype="float32", always_2d=False)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        y = librosa.resample(wav, orig_sr=sr, target_sr=24000) if sr != 24000 else wav
        out = os.path.join(wav_dir, f"{idx:06d}.wav")
        sf.write(out, y, 24000, subtype="PCM_16")
        f.write(json.dumps({"audio": out, "text": text, "ref_audio": out,
                            "instruct": "", "emotion": emo}, ensure_ascii=False) + "\n")

    with open(out_jsonl, "w") as f:
        for idx, ex in enumerate(ds):
            if args.min_dnsmos and (ex.get("dnsmos") or 0) < args.min_dnsmos:
                skipped_q += 1; continue
            text, markers = map_text(ex[args.text_col] or "", hist, kept, dropped)

            if not markers and not args.keep_unmapped:
                if args.neutral and n_neutral < args.neutral:
                    write_clip(ex, idx, text, "neutral"); n_neutral += 1
                else:
                    skipped_nonv += 1
                continue

            if args.cap_per_marker and markers and all(marker_clips[m] >= args.cap_per_marker for m in markers):
                skipped_cap += 1; continue
            for m in markers:
                marker_clips[m] += 1
            emo = (ex.get(args.emotion_col) or "neutral").strip().lower()
            write_clip(ex, idx, text, emo)
            n_written += 1
            if args.max_rows and n_written >= args.max_rows:
                break

    print(f"wrote {n_written} marked + {n_neutral} neutral = {n_written + n_neutral} rows -> {out_jsonl}")
    print(f"  skipped: {skipped_q} low-DNSMOS, {skipped_nonv} plain(no-anchor), {skipped_cap} over-cap")
    print(f"  marker CLIP counts (balanced): {dict(marker_clips)}")
    if dropped:
        print(f"  emoji stripped (unmapped): {sum(dropped.values())} occ across {len(dropped)} kinds "
              f"-> rerun with --histogram to inspect")
    print("NEXT: run the upstream prepare_data.py on this jsonl to add audio_codes, then train_lora.py.")

if __name__ == "__main__":
    main()
