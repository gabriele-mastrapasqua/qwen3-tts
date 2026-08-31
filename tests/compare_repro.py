#!/usr/bin/env python3
"""compare_repro.py — server-reproducibility gate with an fp-noise tolerance."""
import sys
import wave

MAX_LSB = 2
MAX_FRAC = 0.002

def read(path):
    w = wave.open(path)
    n = w.getnframes()
    raw = w.readframes(n)
    w.close()
    return [int.from_bytes(raw[i:i + 2], "little", signed=True)
            for i in range(0, len(raw), 2)]

def main():
    if len(sys.argv) < 3:
        print("usage: compare_repro.py ref.wav other.wav [...]")
        return 2
    ref = read(sys.argv[1])
    fail = 0
    for p in sys.argv[2:]:
        cur = read(p)
        if len(cur) != len(ref):
            print(f"FAIL: {p}: length {len(cur)} != {len(ref)} "
                  f"(trajectory fork — real state leak)")
            fail = 1
            continue
        ndiff, worst = 0, 0
        for a, b in zip(ref, cur):
            d = abs(a - b)
            if d:
                ndiff += 1
                if d > worst:
                    worst = d
        frac = ndiff / max(1, len(ref))
        ok = worst <= MAX_LSB and frac <= MAX_FRAC
        print(f"  {p}: ndiff={ndiff} ({frac*100:.3f}%) max|diff|={worst} LSB "
              f"-> {'ok' if ok else 'FAIL'}")
        if not ok:
            fail = 1
    if fail:
        print("FAIL: identical requests deviate beyond fp noise (state leak?)")
        return 1
    print("PASS: identical requests reproducible (within ±%d LSB fp noise)" % MAX_LSB)
    return 0

if __name__ == "__main__":
    sys.exit(main())
