#!/usr/bin/env python3
import argparse, json, os
from collections import Counter

DEFAULT = [("emovo","IT"),("emodb_bb","DE"),("mesd_bb","ES"),("cafe_bb","FR"),
           ("resd","RU"),("etod","KO"),("jvnv","JA")]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/root/qwen-ft")
    ap.add_argument("--out", default="/root/qwen-ft/carriers.jsonl")
    ap.add_argument("--per-dir", type=int, default=150)
    args = ap.parse_args()
    out, by = [], Counter()
    for d, lang in DEFAULT:
        f = os.path.join(args.root, d, "train_raw.jsonl")
        if not os.path.exists(f): print(f"  skip {d} (no manifest)"); continue
        n = 0
        for line in open(f):
            if n >= args.per_dir: break
            r = json.loads(line)
            if r.get("emotion") != "neutral": continue
            txt = (r.get("text") or "").strip()
            if not txt: continue
            base = os.path.basename(r["audio"])
            out.append({"audio": os.path.join(args.root, d, "wav24k", base), "text": txt, "lang": lang})
            n += 1; by[lang] += 1
    with open(args.out, "w") as fo:
        for r in out: fo.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(out)} carriers -> {args.out}  by lang: {dict(by)}")

if __name__ == "__main__":
    main()
