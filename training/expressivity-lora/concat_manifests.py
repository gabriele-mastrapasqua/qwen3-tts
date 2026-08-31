#!/usr/bin/env python3
"""Concatenate several *_with_codes.jsonl manifests into one multi-speaker training file."""
import os, json, argparse, collections, sys

REQUIRED = ("text", "instruct", "emotion", "actor", "audio_codes")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("inputs", nargs="+", help="*_with_codes.jsonl files to merge, in order")
    ap.add_argument("--langs", default=None,
                    help="comma list, ONE per input file -> stamps `language` on every row of that file "
                         "(for the language-tagged FT). e.g. --langs Italian,English,English")
    ap.add_argument("--repeat", default=None,
                    help="comma list, ONE per input -> duplicate that file's rows N times (oversample a "
                         "minority language). e.g. --repeat 8,1,1")
    ap.add_argument("--allow-missing", action="store_true",
                    help="skip (don't fail on) rows missing required keys")
    a = ap.parse_args()

    langs = a.langs.split(",") if a.langs else None
    repeats = [int(x) for x in a.repeat.split(",")] if a.repeat else None
    if langs and len(langs) != len(a.inputs):
        print(f"[concat] FATAL: --langs has {len(langs)} entries but {len(a.inputs)} inputs", file=sys.stderr); sys.exit(2)
    if repeats and len(repeats) != len(a.inputs):
        print(f"[concat] FATAL: --repeat has {len(repeats)} entries but {len(a.inputs)} inputs", file=sys.stderr); sys.exit(2)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    merged, bad = [], 0
    by_source = collections.Counter()
    by_emo = collections.Counter()
    speakers = set()

    for fi, path in enumerate(a.inputs):
        if not os.path.exists(path):
            print(f"[concat] FATAL: input not found: {path}", file=sys.stderr); sys.exit(2)
        src = os.path.basename(os.path.dirname(path)) or os.path.basename(path)
        lang = langs[fi].strip() if langs else None
        rep = repeats[fi] if repeats else 1
        rows_this = []
        for ln, line in enumerate(open(path, encoding="utf-8"), 1):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            miss = [k for k in REQUIRED if k not in r or r[k] in (None, "")]
            miss = [k for k in miss if not (k == "instruct" and r.get("emotion") == "neutral")]
            if miss:
                bad += 1
                msg = f"[concat] {path}:{ln} missing {miss}"
                if a.allow_missing:
                    print(msg + " (skipped)", file=sys.stderr); continue
                print(msg + "  -> FAIL (re-run the encode step, or pass --allow-missing)",
                      file=sys.stderr)
                sys.exit(3)
            r["source"] = src
            if lang:
                r["language"] = lang
            rows_this.append(r)
        for _ in range(rep):
            for r in rows_this:
                merged.append(r)
                by_source[src] += 1
                by_emo[r["emotion"]] += 1
                speakers.add(f"{src}/{r['actor']}")
        tag = f"source '{src}'" + (f", lang '{lang}'" if lang else "") + (f", x{rep}" if rep > 1 else "")
        print(f"[concat] {path}: {len(rows_this)} rows x {rep} = {len(rows_this)*rep} ({tag})", flush=True)

    with open(a.out, "w", encoding="utf-8") as f:
        for r in merged:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n[concat] DONE -> {a.out}", flush=True)
    print(f"[concat]   total rows : {len(merged)}  ({bad} bad rows {'skipped' if a.allow_missing else 'n/a'})")
    print(f"[concat]   speakers   : {len(speakers)} unique (source/actor)")
    print(f"[concat]   by source  : {dict(by_source)}")
    print(f"[concat]   by emotion : {dict(by_emo)}")

if __name__ == "__main__":
    main()
