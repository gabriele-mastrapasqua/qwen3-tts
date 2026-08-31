#!/usr/bin/env python3
"""serve_identity_gate.py — the gate that must pass BEFORE any KPI table is printed."""
import json, os, re, sys

def pct(v, q):
    v = sorted(x for x in v if x == x)
    if not v: return float("nan")
    return v[min(len(v)-1, int(round(q/100.0*(len(v)-1))))]

def engine_index(logpath):
    """One dict seed -> worker record for the WHOLE log. Seeds are globally unique
    (warm-up at +900000, each cell at +1000*C), so segmenting the log by [prefork-stats]
    is no longer needed - and no longer trusted: it was an assumption about flush order
    of two child processes' stderr, not a measurement."""
    life, req, dup = {}, {}, set()
    for ln in open(logpath, errors="replace"):
        m = re.search(r"\[LIFE\] pid=(\d+) seed=(\d+) parse=([\d.-]+) queue=([\d.-]+) "
                      r"pre_service=([\d.-]+) ttfa_after_admit=([\d.-]+) service=([\d.-]+) "
                      r"worker_total=([\d.-]+)", ln)
        if m:
            sd = int(m.group(2))
            if sd in life: dup.add(sd)
            life[sd] = dict(pid=m.group(1), seed=sd, queue=float(m.group(4)),
                            pre=float(m.group(5)), ttfa_admit=float(m.group(6)),
                            service=float(m.group(7)), total=float(m.group(8)))
            continue
        m = re.search(r"\[REQ\] pid=(\d+) seed=(\d+) tokens=(\d+) frames=(\d+) "
                      r"audio_s=([\d.]+) service_ms=([\d.]+)", ln)
        if m: req[int(m.group(2))] = dict(frames=int(m.group(4)), audio=float(m.group(5)))
    out = {}
    for sd, r in life.items():
        r = dict(r); r.update(req.get(sd, {})); out[sd] = r
    return out, dup

def run(d):
    def conc_of(f):
        m = re.search(r"_C(\d+)\.jsonl$", f) or re.search(r"_c(\d+)_requests\.jsonl$", f)
        return int(m.group(1)) if m else None
    cells = sorted([f for f in os.listdir(d)
                    if f.endswith(".jsonl") and conc_of(f) is not None],
                   key=conc_of)
    logs = [f for f in os.listdir(d) if f.endswith(".log")]
    summ = {r["conc"]: r for r in json.load(open(os.path.join(d,
            [f for f in os.listdir(d) if f.startswith("parallel_")][0])))}
    index, dup = engine_index(os.path.join(d, logs[0]))

    print(f"\n### IDENTITY GATE — {d}")
    table, fail = [], False
    if dup:
        print(f"  ❌ {len(dup)} seed(s) appear more than once in the worker log "
              f"({sorted(dup)[:5]}...) — requests are not distinguishable")
        fail = True
    for cell in cells:
        C = conc_of(cell)
        rows = [json.loads(l) for l in open(os.path.join(d, cell))]
        paired, unmatched = [], 0
        for r in rows:
            e = index.get(r["seed"])
            if e is None: unmatched += 1; continue
            r["e"] = e; paired.append(r)
        if not paired:
            print(f"  C={C}  ❌ NO PAIRS"); fail = True; continue
        da  = [abs(r["audio_s"] - r["e"].get("audio", float("nan")))*1000 for r in paired]
        ovh = [r["total_s"]*1000.0 - (r["e"]["pre"] + r["e"]["service"]) for r in paired]
        bad_audio = sum(1 for x in da if not (x < 1.0))
        bad_sign  = sum(1 for x in ovh if x < 0)
        ok = (unmatched == 0 and bad_audio == 0 and bad_sign == 0)
        fail = fail or not ok
        print(f"  C={C:<2} n={len(paired):<3} unmatched={unmatched}  "
              f"|audio_c-audio_e| max {max(da):.2f} ms (bad {bad_audio})  "
              f"overhead = client_total-(pre+service): p50 {pct(ovh,50):.1f} "
              f"p95 {pct(ovh,95):.1f} min {min(ovh):.1f} max {max(ovh):.1f} ms "
              f"(negative {bad_sign})   {'✅' if ok else '❌'}")
        s = summ.get(C, {})
        table.append((C, len(paired),
                      pct([r["ttfa_ms"] for r in paired], 50),
                      pct([r["ttfa_ms"] for r in paired], 95),
                      pct([r["stream_rtf"] for r in paired], 50),
                      pct([r["stream_rtf"] for r in paired], 95),
                      s.get("rejects", "?"), s.get("errors", "?"),
                      pct([r["audio_s"] for r in paired], 50)))
    if fail:
        print("\n  ❌ GATE FAILED — the table is NOT printed and these numbers are not used.")
        return True, []
    print("\n  ✅ GATE PASSED — every request pairs, audio agrees, no part exceeds the whole.")
    print("\n### FROZEN BASELINE")
    print(f"{'C':>3}{'n':>5}{'TTFA50':>9}{'TTFA95':>9}{'STREAM50':>11}{'STREAM95':>11}"
          f"{'rejects':>9}{'errors':>8}{'audio50':>10}")
    print("-"*75)
    for C, n, t50, t95, s50, s95, rj, er, au in table:
        print(f"{C:>3}{n:>5}{t50:>9.0f}{t95:>9.0f}{s50:>11.3f}{s95:>11.3f}"
              f"{str(rj):>9}{str(er):>8}{au:>9.2f}s")
    return False, table

if __name__ == "__main__":
    bad = 0
    for d in sys.argv[1:]:
        if os.path.isdir(d):
            r = run(d)
            if r is None or (isinstance(r, tuple) and r[0]):
                bad += 1
        else:
            print(f"not a directory: {d}"); bad += 1
    sys.exit(1 if bad else 0)
