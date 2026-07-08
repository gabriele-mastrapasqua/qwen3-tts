#!/usr/bin/env python3
"""dyad_mix.py — blend N emotion .qlsteer steering vectors into one dyad vector.

A .qlsteer is: 'QLST' magic (4B) + int32 L + int32 D + L*D float32 (row-major, per-layer).
A dyad = weighted element-wise sum of two (or more) same-shape vectors. Apply the result
with the usual `--ml-steer dyad.qlsteer --ml-weight 12 --ml-range 21-25` (no --emotion).

Usage:
  dyad_mix.py OUT.qlsteer A.qlsteer:0.5 B.qlsteer:0.5
  dyad_mix.py OUT.qlsteer A.qlsteer B.qlsteer          # equal weights (0.5/0.5)
"""
import struct, sys

MAGIC = 0x54534C51  # 'QLST'

def load(path):
    with open(path, "rb") as f:
        magic, L, D = struct.unpack("<Iii", f.read(12))
        if magic != MAGIC:
            sys.exit(f"{path}: bad magic {magic:#x} (not QLST)")
        n = L * D
        data = list(struct.unpack(f"<{n}f", f.read(n * 4)))
    return L, D, data

def main(argv):
    if len(argv) < 3:
        sys.exit(__doc__)
    out = argv[1]
    parts = argv[2:]
    specs = []
    for p in parts:
        if ":" in p:
            path, w = p.rsplit(":", 1); w = float(w)
        else:
            path, w = p, None
        specs.append([path, w])
    # default equal weights if none given
    if all(w is None for _, w in specs):
        for s in specs: s[1] = 1.0 / len(specs)
    elif any(w is None for _, w in specs):
        sys.exit("mix all-or-none: give a weight to every input or none")

    L0 = D0 = None
    acc = None
    for path, w in specs:
        L, D, data = load(path)
        if L0 is None:
            L0, D0 = L, D; acc = [0.0] * (L * D)
        elif (L, D) != (L0, D0):
            sys.exit(f"{path}: shape {L}x{D} != {L0}x{D0}")
        for i, v in enumerate(data):
            acc[i] += w * v
        print(f"  + {w:.3f} * {path}")

    with open(out, "wb") as f:
        f.write(struct.pack("<Iii", MAGIC, L0, D0))
        f.write(struct.pack(f"<{L0*D0}f", *acc))
    print(f"wrote {out}  ({L0}x{D0}, {len(specs)} inputs)")

if __name__ == "__main__":
    main(sys.argv)
