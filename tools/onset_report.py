#!/usr/bin/env python3
"""How many leading codec frames does every render share, and are they silent?

Reads a directory of paired <cell>.code0 / <cell>.wav produced by QWEN_DUMP_CODE0.
Reports the code0 opening of each cell, the leading silence of its waveform, and the
number of leading frames that are identical across EVERY cell in the set. Cells are
meant to differ in speaker, seed, text and sampling: what survives all of that is a
property of the model, not of the request.
"""
import sys, os, glob, wave, array


def lead_silence(path, thr=0.005):
    try:
        w = wave.open(path, "rb"); sr = w.getframerate()
        a = array.array("h"); a.frombytes(w.readframes(w.getnframes())); w.close()
    except Exception:
        return float("nan")
    t = thr * 32768.0
    i = 0
    while i < len(a) and abs(a[i]) < t:
        i += 1
    return i / float(sr)


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "."
    rows = []
    for p in sorted(glob.glob(os.path.join(d, "*.code0"))):
        codes = [l.strip() for l in open(p) if l.strip()]
        rows.append((os.path.basename(p)[:-6], codes[:8], lead_silence(p[:-6] + ".wav")))
    if not rows:
        print("no cells found in", d); return 2
    print("%-26s %-44s %s" % ("cell", "code0 first 8 frames", "lead_silence_s"))
    for n, c, l in rows:
        print("%-26s %-44s %.3f" % (n, " ".join(c), l))
    seqs = [r[1] for r in rows]
    k = 0
    while all(len(s) > k for s in seqs) and len({s[k] for s in seqs}) == 1:
        k += 1
    print("\nleading code0 frames identical across ALL %d cells "
          "(speaker x seed x text x sampling): %d" % (len(rows), k))
    print("at 12.5 Hz that is %.3f s of audio" % (k * 0.08))
    ls = [r[2] for r in rows if r[2] == r[2]]
    if ls:
        print("lead silence across cells: min %.3f  max %.3f  spread %.3f s"
              % (min(ls), max(ls), max(ls) - min(ls)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
