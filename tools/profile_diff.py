#!/usr/bin/env python3
"""profile_diff.py — rank what STRUCTURALLY changes between two FAST profiles.

    tools/profile_diff.py C2/profile.json C4/profile.json

Not a UI: a sorted delta over the KPI, the semantic regions and the pool decomposition, so a
regime change can be read instead of guessed at. Regions come from each artifact's cost map.
"""
import argparse, collections, glob, json, os, sys


def regions(art):
    d = os.path.dirname(art)
    agg = collections.defaultdict(lambda: {"ns": 0, "calls": 0, "units": 0, "tasks": 0,
                                           "entered": 0, "ticks": 0, "dispatches": 0})
    for sub in ("costmap", "costmap-deep"):
        files = sorted(glob.glob(os.path.join(d, sub, "*.json")))
        if not files:
            continue
        for f in files:
            try:
                doc = json.load(open(f))
            except (OSError, ValueError):
                continue
            for t in doc.get("threads", []):
                for r in t.get("regions", []):
                    a = agg[r["name"]]
                    for k in ("ns", "calls", "units", "tasks", "entered", "ticks", "dispatches"):
                        a[k] += r.get(k, 0) or 0
        break                      # prefer costmap/, fall back to costmap-deep/
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a"); ap.add_argument("b")
    ap.add_argument("--top", type=int, default=25)
    x = ap.parse_args()
    A, B = json.load(open(x.a)), json.load(open(x.b))
    la = "C%s" % A["server"]["concurrency"]
    lb = "C%s" % B["server"]["concurrency"]

    print("PROFILE DIFF   %s -> %s" % (la, lb))
    print("  same host=%s  workers=%dx%d  build=%s" %
          (A["host"].get("cpu_model", "?"), A["server"]["workers"],
           A["server"]["threads_per_worker"], A["build"].get("simd")))
    print("\n  --- KPI ---")
    for k in ("ttfa_p50_ms", "ttfa_p95_ms", "stream_rtf_p50", "stream_rtf_p95"):
        va, vb = A["kpi"].get(k), B["kpi"].get(k)
        if va and vb:
            print("  %-16s %10.3f -> %10.3f   %+7.1f%%" % (k, va, vb, 100 * (vb - va) / va))

    ra, rb = regions(x.a), regions(x.b)
    rows = []
    for name in set(ra) | set(rb):
        na, nb = ra.get(name, {}).get("ns", 0), rb.get(name, {}).get("ns", 0)
        if max(na, nb) < 5e6:                  # ignore sub-5ms noise
            continue
        pa, pb = na / 1e6, nb / 1e6
        rows.append((abs(pb - pa), name, pa, pb))
    rows.sort(reverse=True)
    print("\n  --- semantic regions, ms total (ranked by absolute change) ---")
    for _, name, pa, pb in rows[:x.top]:
        d = ("%+7.1f%%" % (100 * (pb - pa) / pa)) if pa else "     new"
        print("  %-34s %9.1f -> %9.1f   %s" % (name, pa, pb, d))

    print("\n  --- pool decomposition ---")
    for name in sorted(set(ra) | set(rb)):
        a, b = ra.get(name, {}), rb.get(name, {})
        if not (a.get("tasks") or b.get("tasks") or a.get("ticks") or b.get("ticks")):
            continue
        print("  %-34s tasks %10d -> %-10d  entered %7d -> %-7d  ticks %9d -> %d" %
              (name, a.get("tasks", 0), b.get("tasks", 0), a.get("entered", 0),
               b.get("entered", 0), a.get("ticks", 0), b.get("ticks", 0)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
