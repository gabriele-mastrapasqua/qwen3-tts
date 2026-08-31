#!/usr/bin/env python3
"""Signal-level screen for generated speech: clipping, DC, silence holes, clicks.

WHAT THIS IS FOR, AND WHAT IT IS NOT FOR
----------------------------------------
It answers "did the waveform break?" — clipped runs, a hole in the middle of a
sentence, a step discontinuity, a file that is all silence. Those are defects a
listener should not have to hunt for one clip at a time.

It does NOT answer "is this good speech". Nothing here scores pronunciation,
accent, prosody or naturalness, and a clean row is not a pass: a render can be
perfect by every number below and still be wrong to the ear. The ear is the
verdict; this only decides what to listen to first.

Usage:  python3 tools/wav_qc.py <dir-or-wav> [more...]
"""
import os, sys, wave, array, math

CLIP = 32700          # 16-bit peak minus a little headroom
SIL   = 0.005         # |s| below this fraction of full scale counts as silence
HOLE_MS = 1200        # an internal silence longer than this is reported. Natural
                      # sentence pauses run 0.5-0.9 s, so a lower bar flags correct speech.
STEP  = 0.35          # single-sample jump, as a fraction of full scale


def scan(path):
    with wave.open(path, "rb") as w:
        nch, sw, sr, n = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
        raw = w.readframes(n)
    if sw != 2:
        return {"file": path, "error": "sample width %d, expected 16-bit" % (sw * 8)}
    a = array.array("h")
    a.frombytes(raw)
    if nch > 1:
        a = array.array("h", a[::nch])
    n = len(a)
    if n == 0:
        return {"file": path, "error": "empty"}
    dur = n / float(sr)
    peak = 0; acc = 0.0; dc = 0.0
    clipped = 0; clip_run = 0; clip_run_max = 0
    prev = a[0]; step_max = 0; steps = 0
    sil_thr = SIL * 32768.0
    runs = []            # (start, end) silent runs, in samples
    run_start = None
    for i in range(n):
        s = a[i]
        v = s if s >= 0 else -s
        if v > peak: peak = v
        acc += float(s) * s
        dc += s
        if v >= CLIP:
            clipped += 1; clip_run += 1
            if clip_run > clip_run_max: clip_run_max = clip_run
        else:
            clip_run = 0
        d = s - prev
        if d < 0: d = -d
        if d > step_max: step_max = d
        if d > STEP * 32768.0: steps += 1
        prev = s
        if v < sil_thr:
            if run_start is None: run_start = i
        else:
            if run_start is not None:
                runs.append((run_start, i)); run_start = None
    if run_start is not None:
        runs.append((run_start, n))
    lead = (runs[0][1] / float(sr)) if runs and runs[0][0] == 0 else 0.0
    tail = ((n - runs[-1][0]) / float(sr)) if runs and runs[-1][1] == n else 0.0
    inner = [(b - a_) / float(sr) for a_, b in runs
             if a_ != 0 and b != n and (b - a_) / float(sr) * 1000.0 >= HOLE_MS]
    rms = math.sqrt(acc / n) / 32768.0
    return {
        "file": path, "sr": sr, "dur": dur, "peak": peak / 32768.0, "rms": rms,
        "dc": (dc / n) / 32768.0, "clipped": clipped, "clip_run": clip_run_max,
        "lead": lead, "tail": tail, "holes": inner,
        "step": step_max / 32768.0, "steps": steps,
    }


def verdict(r):
    bad, warn = [], []
    if r.get("error"):        bad.append(r["error"])
    if r.get("dur", 1) < 0.15: bad.append("almost no audio")
    if r.get("rms", 1) < 0.001: bad.append("silent")
    if r.get("clip_run", 0) >= 8:  bad.append("clipped run %d" % r["clip_run"])
    elif r.get("clipped", 0) > 0:  warn.append("clipped %d" % r["clipped"])
    if r.get("holes"):
        big = [h for h in r["holes"] if h >= 2.5]
        (bad if big else warn).append("hole %s" % ",".join("%.2fs" % h for h in r["holes"]))
    if r.get("steps", 0) > 0: warn.append("step x%d (max %.2f)" % (r["steps"], r["step"]))
    if abs(r.get("dc", 0)) > 0.02: warn.append("dc %.3f" % r["dc"])
    if r.get("tail", 0) > 1.5:  warn.append("tail silence %.1fs" % r["tail"])
    if r.get("lead", 0) > 1.0:  warn.append("lead silence %.1fs" % r["lead"])
    return ("FAIL" if bad else ("WARN" if warn else "ok")), "; ".join(bad + warn)


def collect(args):
    out = []
    for a in args:
        if os.path.isdir(a):
            for root, _, files in os.walk(a):
                for f in sorted(files):
                    if f.lower().endswith(".wav"):
                        out.append(os.path.join(root, f))
        elif a.lower().endswith(".wav"):
            out.append(a)
    return sorted(out)


def main():
    files = collect(sys.argv[1:])
    if not files:
        print("no wav files"); return 2
    print("%-58s %6s %6s %6s %5s %5s  %s" %
          ("file", "dur_s", "peak", "rms", "clip", "verd", "notes"))
    n_fail = n_warn = 0
    for p in files:
        try:
            r = scan(p)
        except Exception as e:                       # a file that cannot be read IS a finding
            r = {"file": p, "error": "unreadable: %s" % e}
        v, note = verdict(r)
        if v == "FAIL": n_fail += 1
        elif v == "WARN": n_warn += 1
        short = p if len(p) <= 58 else "..." + p[-55:]
        print("%-58s %6.2f %6.3f %6.4f %5d %5s  %s" %
              (short, r.get("dur", 0), r.get("peak", 0), r.get("rms", 0),
               r.get("clipped", 0), v, note))
    print("\n%d files: %d FAIL, %d WARN, %d ok" %
          (len(files), n_fail, n_warn, len(files) - n_fail - n_warn))
    print("A clean row means the waveform is intact, not that the speech is good.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
