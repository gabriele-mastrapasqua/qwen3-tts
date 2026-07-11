#!/usr/bin/env python3
"""compare_repro.py — server-reproducibility gate with an fp-noise tolerance.

The old gate was raw md5 equality. That is the WRONG oracle for this engine
(same reason test-golden uses mel-corr, not md5): the decode path carries a
known, benign ±1–2 LSB int16 jitter (≈ −90 dB, fp accumulation order — e.g.
threaded snake / BLAS internals), measured 2026-07-11 at ~0.05% of samples
with IDENTICAL length and codec codes. What this test must actually catch is
STATE LEAKING between requests: different lengths (trajectory fork), or
sample deviations beyond LSB noise.

PASS  = all runs: identical sample count, max |diff| <= MAX_LSB,
        differing samples <= MAX_FRAC.
"""
import sys
import wave

MAX_LSB = 2        # max per-sample deviation (int16 LSB)
MAX_FRAC = 0.002   # max fraction of differing samples (0.2%)


def read(path):
    w = wave.open(path)
    n = w.getnframes()
    raw = w.readframes(n)
    w.close()
    # int16 mono
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
