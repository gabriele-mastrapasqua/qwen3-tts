#!/usr/bin/env python3
"""Per-codebook argmax-agreement matrix across code-predictor precisions.

Reads code dumps produced by QWEN_DUMP_CODES (one line per audio frame, 16 tokens).
Each file is one precision; the Talker is held at bf16 (QWEN_CP_PREC), so code0 is
identical run to run and any disagreement in c1..c15 is code-predictor drift alone.

Usage:
    quant_ladder.py bf16:bf16.codes int8:int8.codes int4:int4.codes

The first file is the reference. Reports overall agreement per precision, per-codebook
agreement, and the int4-vs-int8 pair.
"""
import sys

def load(path):
    rows = []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            rows.append([int(x) for x in parts])
    return rows

def col_agreement(a, b, col, nframes):
    """Fraction of frames where token `col` matches between dumps a and b."""
    same = sum(1 for i in range(nframes) if a[i][col] == b[i][col])
    return same / nframes if nframes else float("nan")

def overall_agreement(a, b, nframes):
    """Mean per-frame agreement over codebooks 1..15 (cols 1..15)."""
    tot = same = 0
    for i in range(nframes):
        for col in range(1, 16):
            tot += 1
            if a[i][col] == b[i][col]:
                same += 1
    return same / tot if tot else float("nan")

def main():
    args = sys.argv[1:]
    if len(args) < 2:
        print(__doc__)
        sys.exit(1)

    labels, paths = [], []
    for a in args:
        if ":" in a:
            lbl, p = a.split(":", 1)
        else:
            lbl, p = a, a
        labels.append(lbl)
        paths.append(p)

    dumps = [load(p) for p in paths]
    nframes = min(len(d) for d in dumps)
    if any(len(d) != nframes for d in dumps):
        print(f"WARN: frame counts differ {[len(d) for d in dumps]} — "
              f"truncating to {nframes} (Talker should be identical → investigate)")

    ref = dumps[0]
    code0_ok = True
    for d, lbl in zip(dumps[1:], labels[1:]):
        bad = sum(1 for i in range(nframes) if d[i][0] != ref[i][0])
        if bad:
            code0_ok = False
            print(f"WARN: code0 differs in {bad}/{nframes} frames for '{lbl}' "
                  f"— Talker NOT held fixed (use QWEN_CP_PREC, drop --int4/--int8). "
                  f"CP agreement below is contaminated.")
    if code0_ok:
        print(f"OK: code0 (Talker) identical across all {len(dumps)} dumps, "
              f"{nframes} frames → clean CP-isolated comparison.\n")

    print(f"=== Quant-ladder agreement vs reference '{labels[0]}' "
          f"(codebooks 1..15, {nframes} frames) ===")
    for d, lbl in zip(dumps[1:], labels[1:]):
        ov = overall_agreement(ref, d, nframes)
        print(f"  {lbl:>8} vs {labels[0]:<8}  overall = {100*ov:6.2f}%")

    others = list(zip(labels[1:], dumps[1:]))
    print("\n=== Per-codebook-index agreement vs '%s' (which residuals drift first) ==="
          % labels[0])
    hdr = "  cb# " + "".join(f"{lbl:>9}" for lbl, _ in others)
    print(hdr)
    for col in range(1, 16):
        cells = "".join(f"{100*col_agreement(ref, d, col, nframes):8.2f}%"
                        for _, d in others)
        print(f"  c{col:<3}{cells}")

    by = dict(zip(labels, dumps))
    if "int4" in by and "int8" in by:
        a, b = by["int4"], by["int8"]
        print("\n=== int4 vs int8 (int8 is the artifact-free gold) ===")
        print(f"  overall = {100*overall_agreement(a, b, nframes):6.2f}%")
        worst = sorted(range(1, 16),
                       key=lambda col: col_agreement(a, b, col, nframes))[:5]
        print("  lowest-agreement codebooks (where int4 diverges from int8 most):")
        for col in worst:
            print(f"    c{col:<3} {100*col_agreement(a, b, col, nframes):6.2f}%")

if __name__ == "__main__":
    main()
