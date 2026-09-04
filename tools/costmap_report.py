#!/usr/bin/env python3
"""costmap_report.py — read the cost-map JSON dumps and print the coarse cost map.

    tools/costmap_report.py profile/costmap/costmap-*.json \
        [--census profile/census/census-*.json] [--out costmap_summary.json]

The JSON carries the taxonomy with the data (name, parent, level, mode per row), so
this tool never re-declares the region tree and can never drift from the binary that
produced it.

Timing semantics, restated here because a report that does not say this is useless:
  * every "stack" region is INCLUSIVE of the regions entered inside it;
  * self = ns - child_ns is DERIVED, never measured;
  * rows with mode "derived" were not bracketed by a begin/end pair on one thread
    (the server request lifecycle crosses threads) and are reported separately;
  * regions are accumulated per OS thread; threads with different roles run
    CONCURRENTLY, so shares are given within a thread role, never as one flat 100%.
"""
import argparse, glob, json, os, sys
from collections import defaultdict


def load(paths):
    docs = []
    for p in paths:
        for f in sorted(glob.glob(p)) or ([p] if os.path.exists(p) else []):
            try:
                docs.append((f, json.load(open(f))))
            except Exception as e:
                print(f"WARNING: cannot read {f}: {e}", file=sys.stderr)
    return docs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json", nargs="+")
    ap.add_argument("--census", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--min-ms", type=float, default=0.0)
    a = ap.parse_args()

    docs = load(a.json)
    if not docs:
        print("no cost-map JSON found"); return 1

    # ---- merge: (role, region id) -> counters -------------------------------
    rows = defaultdict(lambda: {"calls": 0, "ns": 0, "child_ns": 0, "nest_mismatch": 0})
    meta = {}
    requests = 0
    level = 0
    integrity = defaultdict(int)
    pids, roles = set(), set()
    role_threads = defaultdict(set)
    for _, d in docs:
        requests += d.get("requests", 0)
        level = max(level, d.get("level", 0))
        pids.add(d.get("pid"))
        for t in d.get("threads", []):
            role = t.get("role", "?")
            roles.add(role)
            role_threads[role].add((d.get("pid"), t.get("tid")))
            for k in ("stack_overflow", "unbalanced", "leaked"):
                integrity[k] += t.get(k, 0)
            for r in t.get("regions", []):
                rid = r["id"]
                meta[rid] = r
                acc = rows[(role, rid)]
                acc["calls"] += r.get("calls", 0)
                acc["ns"] += r.get("ns", 0)
                acc["child_ns"] += r.get("child_ns", 0)
                acc["nest_mismatch"] += r.get("nest_mismatch", 0)

    nreq = requests if requests > 0 else 0
    print("COST MAP — profiler V1 step 2 (coarse semantic regions)")
    print("=" * 78)
    print(f"  processes={len(pids)}  level={level}  requests={requests or 'n/a'}  "
          f"thread roles={','.join(sorted(roles))}")
    print("  semantics: INCLUSIVE per region; self = ns - child_ns (derived); "
          "clock CLOCK_MONOTONIC")
    print("  shares are within a thread role: roles run concurrently and do NOT sum to 100%")

    # ---- per role, per component -------------------------------------------
    by_role = defaultdict(list)
    for (role, rid), acc in rows.items():
        by_role[role].append((rid, acc))

    out = {"requests": requests, "level": level, "processes": len(pids),
           "semantics": "inclusive", "exclusive_rule": "self_ns = ns - child_ns",
           "roles": {}, "integrity": dict(integrity), "regions": []}

    for role in sorted(by_role):
        print()
        nth = len(role_threads.get(role, ()))
        head = f"  --- thread role: {role} ({nth} thread{'s' if nth != 1 else ''}) "
        print(head + "-" * max(4, 78 - len(head)))
        # A role that never completed a request did start-up work (model warm-up on a
        # prefork worker's main thread).  Its ms/req would divide by requests that were
        # served on ANOTHER thread, so say so instead of printing a misleading rate.
        if not any(rid == 60 for rid, _ in by_role[role]):
            print("             (no request completed on this role: start-up / pre-warm work;"
                  " read the calls column, not ms/req)")
        print(f"  {'component':<10}{'region':<32}{'calls':>9}{'ms_total':>10}"
              f"{'ms/req':>9}{'%parent':>9}{'self%':>7}{'mode':>9}")
        entries = sorted(by_role[role], key=lambda x: (meta[x[0]]["component"], x[0]))
        idx = {rid: acc for rid, acc in entries}
        for rid, acc in entries:
            m = meta[rid]
            ms = acc["ns"] / 1e6
            if ms < a.min_ms:
                continue
            parent = m.get("parent", 0)
            pms = idx.get(parent, {}).get("ns", 0) / 1e6 if parent > 0 else 0.0
            ppct = (100.0 * ms / pms) if pms > 0 else None
            self_ms = (acc["ns"] - acc["child_ns"]) / 1e6
            selfpct = 100.0 * self_ms / ms if ms > 0 else 0.0
            name = m["name"]
            short = name.split(".", 1)[1] if "." in name else name
            s_req = f"{ms / nreq:>9.2f}" if nreq else f"{'-':>9}"
            s_par = f"{ppct:>9.1f}" if ppct is not None else f"{'-':>9}"
            print(f"  {m['component']:<10}{short:<32}{acc['calls']:>9}{ms:>10.1f}"
                  f"{s_req}{s_par}{selfpct:>7.1f}{m.get('mode', 'stack'):>9}")
            out["regions"].append({
                "role": role, "id": rid, "name": name,
                "component": m["component"], "parent": parent,
                "parent_name": meta.get(parent, {}).get("name", ""),
                "level": m.get("level", 1), "mode": m.get("mode", "stack"),
                "calls": acc["calls"], "ms_total": ms,
                "ms_per_request": (ms / nreq) if nreq else None,
                "pct_of_parent": None if pms <= 0 else ppct,
                "self_ms": self_ms, "self_pct": selfpct,
                "nest_mismatch": acc["nest_mismatch"],
            })

        # unaccounted inside each parent that has children
        kids = defaultdict(float)
        for rid, acc in entries:
            p = meta[rid].get("parent", 0)
            if p > 0:
                kids[p] += acc["ns"] / 1e6
        gaps = []
        for rid, acc in entries:
            # A derived row has no children by construction, and pool_dispatch's "gap"
            # is the caller running its own chunk, not unexplained time: neither is an
            # accounting hole, and listing them as one would be noise.
            if rid not in kids or meta[rid].get("mode") == "derived" \
                    or meta[rid].get("parent", 0) < 0:
                continue
            ms = acc["ns"] / 1e6
            un = ms - kids[rid]
            if ms > 0 and un / ms > 0.01:
                gaps.append((meta[rid]["name"], un, 100.0 * un / ms))
        if gaps:
            print(f"  {'':<9}unaccounted inside a parent (parent minus its children):")
            if level < 2:
                print(f"  {'':<9}  NOTE: this run is level {level}; regions declared level 2 were not"
                      f" recorded,\n  {'':<9}        so a parent's gap still contains them.")
            for n, un, pct in sorted(gaps, key=lambda g: -g[1]):
                print(f"  {'':<9}  {n:<40}{un:>9.1f} ms {pct:>6.1f}%")
            out["roles"].setdefault(role, {})["unaccounted"] = [
                {"parent": n, "ms": un, "pct": pct} for n, un, pct in gaps]

    # ---- sync / wait accounting --------------------------------------------
    print()
    print("  --- sync and wait accounting " + "-" * 47)
    disp = sum(acc["ns"] for (r, rid), acc in rows.items() if rid == 62) / 1e6
    wait = sum(acc["ns"] for (r, rid), acc in rows.items() if rid == 63) / 1e6
    subm = sum(acc["ns"] for (r, rid), acc in rows.items() if rid == 64) / 1e6
    ndisp = sum(acc["calls"] for (r, rid), acc in rows.items() if rid == 62)
    print(f"    pool dispatches                 {ndisp:>12}")
    print(f"    in qwen_parallel (inclusive)    {disp:>12.1f} ms")
    print(f"    waiting for worker completion   {wait:>12.1f} ms  "
          f"({100.0 * wait / disp if disp else 0:.1f}% of dispatch)")
    print(f"    waiting for the submit lock     {subm:>12.1f} ms  "
          f"({100.0 * subm / disp if disp else 0:.1f}% of dispatch)")
    print("    barrier synchronisation                  UNRESOLVED — this pool has no")
    print("      distinct barrier primitive; completion is a counter plus a condvar, so")
    print("      barrier time is not separable from 'waiting for worker completion'.")
    print(f"    caller running its own chunk    {disp - wait:>12.1f} ms  "
          f"({100.0 * (disp - wait) / disp if disp else 0:.1f}% of dispatch)  "
          "= dispatch minus wait, not a gap")
    print("    scheduler/admission idle        see runtime.admission (mode=derived).")
    out["sync"] = {"dispatches": ndisp, "dispatch_ms": disp, "wait_completion_ms": wait,
                   "submit_wait_ms": subm,
                   "unresolved": ["barrier_synchronisation"]}

    # ---- join with the shape census ----------------------------------------
    if a.census:
        cdocs = load([a.census])
        ccalls, cmacs = defaultdict(int), defaultdict(int)
        for _, d in cdocs:
            for r in d.get("rows", d.get("census", [])):
                comp = r.get("comp") or r.get("component") or "?"
                ccalls[comp] += r.get("calls", 0)
                cmacs[comp] += r.get("macs", 0)
        if ccalls:
            print()
            print("  --- join with the executed-path census " + "-" * 37)
            print(f"    {'component':<12}{'kernel calls':>14}{'GMAC':>12}"
                  f"{'region ms (total)':>20}")
            for comp in sorted(ccalls):
                ms = sum(acc["ns"] for (r, rid), acc in rows.items()
                         if meta[rid]["component"] == comp
                         and meta[rid].get("parent", 0) == 0) / 1e6
                print(f"    {comp:<12}{ccalls[comp]:>14}{cmacs[comp] / 1e9:>12.1f}"
                      f"{ms:>20.1f}")
            out["census_join"] = {c: {"calls": ccalls[c], "gmac": cmacs[c] / 1e9}
                                  for c in ccalls}

    # ---- integrity ----------------------------------------------------------
    print()
    print("  --- integrity " + "-" * 62)
    nm = sum(acc["nest_mismatch"] for acc in rows.values())
    bad = nm + integrity["stack_overflow"] + integrity["unbalanced"]
    print(f"    nesting mismatches (dynamic parent != declared)   {nm}")
    print(f"    region stack overflows                           {integrity['stack_overflow']}")
    print(f"    unbalanced ends                                  {integrity['unbalanced']}")
    print(f"    regions closed by unwind on an early return      {integrity['leaked']}")
    verdict = "PASS" if bad == 0 else "FAIL"
    print(f"    COST MAP GATE: {verdict}"
          + ("" if bad == 0 else "  -> the tree is not trustworthy, fix the markers"))
    out["verdict"] = verdict

    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)
        print(f"\n  machine-readable: {a.out}")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
