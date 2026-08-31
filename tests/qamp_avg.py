#!/usr/bin/env python3
"""Average N QWEN_ACT_MAP captures (.qamp) into one.

Usage: tests/qamp_avg.py OUT.qamp IN1.qamp IN2.qamp [IN3.qamp ...]
"""
import struct, sys

QAMP = 0x504D4151

def read(path):
    with open(path, "rb") as f:
        magic, L, D = struct.unpack("<Iii", f.read(12))
        if magic != QAMP: sys.exit(f"{path}: bad magic")
        return L, D, list(struct.unpack(f"<{L*D}f", f.read(L*D*4)))

def main():
    if len(sys.argv) < 3: sys.exit(__doc__)
    out, ins = sys.argv[1], sys.argv[2:]
    L, D, acc = read(ins[0])
    for p in ins[1:]:
        Li, Di, v = read(p)
        if (Li, Di) != (L, D): sys.exit(f"shape mismatch {p}")
        for i in range(L*D): acc[i] += v[i]
    n = len(ins)
    acc = [x / n for x in acc]
    with open(out, "wb") as f:
        f.write(struct.pack("<Iii", QAMP, L, D))
        f.write(struct.pack(f"<{L*D}f", *acc))
    print(f"averaged {n} captures -> {out} ({L}x{D})")

if __name__ == "__main__":
    main()
