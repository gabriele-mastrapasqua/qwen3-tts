#!/usr/bin/env python3
"""envelope_report.py — a topology is a curve, not a benchmark point.

A production streaming server is judged by its whole load envelope: low latency when idle,
roughly linear scaling until the hardware saturates, and no cliff in between.  Ranking
topologies by their best single cell hides exactly the failure that matters.

This tool does NOT generate load.  It consumes the artifacts the canonical harnesses
already produce, so the qualification and the benchmarks stay the same experiment:

  serve_parallel_wave.py  parallel_*.json   one server per topology, swept C1..Cn
                                            without restarting between levels
  bench_suite.sh          the realistic/fast/diverse rungs, which are wave runs
  load_test.py            --json summaries, open-loop Poisson/uniform arrivals
  serve_soak.py           soak CSV, stability over time

    tools/envelope_report.py --wave parallel_*.json [--topology topology.json]
        [--store profiles/roofs] [--census census-*.json] [--costmap costmap-*.json]
        [--poisson load_summaries.json] [--out envelope.json]

Every derived classification is printed next to the raw values it came from, and every
threshold is named, because a heuristic that hardens into a hardware truth is how a report
starts lying on the next machine.
"""
import argparse, glob, json, os, statistics, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import topology as TP                                            # noqa: E402
import roofs as RF                                               # noqa: E402

# Named, tunable, and printed with the raw numbers beside them.  Not hardware truths.
TH = {
    "knee_scaling_efficiency": 0.80,   # below this, throughput has stopped scaling
    "cliff_ttfa_p95_growth": 2.0,      # p95 more than doubled vs the lowest load
    "cliff_scaling_efficiency": 0.50,  # ...while throughput barely moved
    "rt_limit": 1.0,                   # RTF at or above realtime
    "admission_dominant": 0.25,        # queue wait as a share of request wall
    "imbalance_ratio": 2.0,            # busiest/least busy worker assignment
    "domain_saturated": 0.90,          # one bandwidth domain at >90% of its own roof
    "domain_idle": 0.30,               # ...while another sits below 30%
    "batch_collapse": 0.70,            # effective batch fell to <70% of its best
    "underrun_tolerance_s": 0.05,      # playback gap below which a stream is healthy
}


def load_many(pats):
    out = []
    for p in pats or []:
        for f in sorted(glob.glob(p)) or ([p] if os.path.exists(p) else []):
            try:
                out.append(json.load(open(f)))
            except Exception as e:
                print(f"WARNING: cannot read {f}: {e}", file=sys.stderr)
    return out


def finite(x):
    return isinstance(x, (int, float)) and x == x


# ---- the two joins the wave harness does not have -----------------------------------

def effective_bandwidth(census_docs, costmap_docs):
    """Weight bytes per request from the census x wall per request from the cost map.

    At B=1 an int8 matvec reads one byte per MAC, so census MACs ARE the weight bytes;
    at B>1 the same weights serve B rows, hence macs/B.  Returns None when either half is
    missing — an unmeasured bandwidth must read UNKNOWN, never zero."""
    if not census_docs or not costmap_docs:
        return None
    serving = None
    for d in costmap_docs:
        for t in d.get("threads", []):
            if any(r["id"] == 60 for r in t.get("regions", [])):
                serving = (d, t)
    if not serving:
        return None
    cdoc, thread = serving
    nreq = cdoc.get("requests") or 0
    if not nreq:
        return None
    cen = next((c for c in census_docs if c.get("pid") == cdoc.get("pid")), None)
    if not cen:
        return None
    wall = {r["name"]: r["ns"] / 1e6 / nreq for r in thread.get("regions", [])}

    def weight_bytes(comp, only_b1):
        """Weight bytes per request for the rows that belong to this region.

        Both halves must be per REQUEST or the ratio is off by the request count.  And the
        rows must match the region: in the Talker, B==1 rows are decode while B>1 rows are
        prefill, so summing the component wholesale would divide prefill bytes by decode
        wall.  A wrapper row repeats its callee's MACs and is skipped."""
        tot = 0.0
        for r in cen.get("rows", []):
            if not r.get("calls") or r.get("comp") != comp:
                continue
            if r.get("kind") == 2:                    # QWEN_PATHK_WRAPPER
                continue
            B = max(r.get("B", 1), 1)
            if only_b1 and B != 1:
                continue
            tot += r["macs"] / B
        return tot / nreq

    out = {}
    for comp, region, b1 in (("talker", "talker.decode.total", True),
                             ("cp", "cp.decode.total", False)):
        ms, b = wall.get(region), weight_bytes(comp, b1)
        if ms and b:
            out[comp] = {"bytes_per_req": b, "ms_per_req": ms, "gbs": b / (ms * 1e6),
                         "region": region,
                         "rows": "B=1 only (decode)" if b1 else "all B (decode incl. prefill2)"}
    return out or None


def admission_share(costmap_docs):
    """runtime.admission / runtime.request.total, both mode=derived from job timestamps."""
    for d in costmap_docs or []:
        for t in d.get("threads", []):
            regs = {r["name"]: r for r in t.get("regions", [])}
            req, adm = regs.get("runtime.request.total"), regs.get("runtime.admission")
            if req and req.get("ns"):
                return {"admission_ms_per_req": (adm["ns"] / 1e6 / max(d.get("requests", 1), 1))
                        if adm else 0.0,
                        "request_ms_per_req": req["ns"] / 1e6 / max(d.get("requests", 1), 1),
                        "share": (adm["ns"] / req["ns"]) if adm else 0.0}
    return None


# ---- envelope -------------------------------------------------------------------------

def build_envelope(cells, topo, store, bw, adm):
    cells = sorted([c for c in cells if finite(c.get("conc"))], key=lambda c: c["conc"])
    if not cells:
        return None
    base = cells[0]
    n0, q0 = base["conc"], base.get("req_s") or 0.0
    ttfa0 = base.get("ttfa_p95") or float("nan")

    rows = []
    for c in cells:
        n = c["conc"]
        q = c.get("req_s") or 0.0
        eff = (q / (n / n0 * q0)) if q0 and n0 else float("nan")
        ok = c.get("ok") or 0
        # SLO goodput: throughput counting only the load that stayed realtime and did not
        # audibly gap.  Batching can lift req/s while the stream degrades, so raw scaling
        # efficiency must never be the success criterion on its own.  Starvation is judged
        # by MAGNITUDE (underrun p95 against a named tolerance), not by a boolean count:
        # a sub-millisecond gap is not a failed request.
        starved = c.get("starved_req") or 0
        un95 = c.get("underrun_p95")
        audible = finite(un95) and un95 > TH["underrun_tolerance_s"]
        goodput = q if not audible else 0.0
        if finite(c.get("rtf_p95")) and c["rtf_p95"] >= TH["rt_limit"]:
            goodput = 0.0
        rows.append({
            "conc": n, "ok": ok, "errors": c.get("errors", 0), "rejects": c.get("rejects", 0),
            "ttfa_p50": c.get("ttfa_p50"), "ttfa_p95": c.get("ttfa_p95"),
            "ttfa_max": c.get("ttfa_max"),
            "ttfb_p50": c.get("ttfb_p50"), "ttfb_p95": c.get("ttfb_p95"),
            "rtf_p50": c.get("rtf_p50"), "rtf_p95": c.get("rtf_p95"),
            "stream_p50": c.get("stream_p50"), "stream_p95": c.get("stream_p95"),
            "req_s": q, "batch_eff": c.get("batch_eff"),
            "assign": c.get("assign"), "cores": c.get("cores"), "csw_s": c.get("csw_s"),
            "starved_req": starved, "underrun_p95": un95,
            "audible_gap": audible,
            "scaling_efficiency": eff, "slo_goodput_req_s": goodput,
            "ttfa_p95_growth": (c.get("ttfa_p95") / ttfa0) if finite(ttfa0) and ttfa0 else None,
            "workers": c.get("workers", []),
        })

    sustainable = None
    for r in rows:
        healthy = (finite(r["rtf_p95"]) and r["rtf_p95"] < TH["rt_limit"]
                   and not r["errors"] and not r["rejects"] and not r["audible_gap"])
        if healthy:
            sustainable = r["conc"]
        else:
            break
    knee = next((r["conc"] for r in rows[1:]
                 if finite(r["scaling_efficiency"])
                 and r["scaling_efficiency"] < TH["knee_scaling_efficiency"]), None)
    post = [r for r in rows if knee and r["conc"] > knee]
    best_batch = max((r["batch_eff"] for r in rows if finite(r["batch_eff"])), default=None)

    cliffs = []

    def flag(name, cond, detail):
        if cond:
            cliffs.append({"cliff": name, "detail": detail})

    for r in rows:
        g, e = r["ttfa_p95_growth"], r["scaling_efficiency"]
        flag("ttfa_p95_growth_without_throughput",
             finite(g) and finite(e) and g > TH["cliff_ttfa_p95_growth"]
             and e < TH["cliff_scaling_efficiency"],
             f"C{r['conc']}: TTFA p95 x{g:.1f} vs C{n0} while scaling efficiency {e:.2f} "
             f"(thresholds {TH['cliff_ttfa_p95_growth']}x / {TH['cliff_scaling_efficiency']})")
        flag("rtf_crosses_realtime",
             finite(r["rtf_p95"]) and r["rtf_p95"] >= TH["rt_limit"],
             f"C{r['conc']}: TOTAL RTF p95 {r['rtf_p95']:.2f} >= {TH['rt_limit']}")
        flag("stream_rtf_crosses_realtime",
             finite(r["stream_p95"]) and r["stream_p95"] >= TH["rt_limit"],
             f"C{r['conc']}: STREAM RTF p95 {r['stream_p95']:.2f} >= {TH['rt_limit']}")
        flag("errors_rejects_or_starvation",
             bool(r["errors"] or r["rejects"] or r["audible_gap"]),
             f"C{r['conc']}: errors={r['errors']} rejects={r['rejects']} "
             f"starved={r['starved_req']} underrun p95 "
             f"{r['underrun_p95'] if finite(r['underrun_p95']) else float('nan'):.3f}s "
             f"(tolerance {TH['underrun_tolerance_s']}s)")
        flag("effective_batch_collapse",
             finite(r["batch_eff"]) and best_batch and r["conc"] > n0
             and r["batch_eff"] < TH["batch_collapse"] * best_batch,
             f"C{r['conc']}: effective batch {r['batch_eff']:.2f} vs best {best_batch:.2f}")
        asg = [w.get("assigned", 0) for w in r["workers"]]
        # Below one request per worker an idle worker is arithmetic, not imbalance.
        if len(asg) > 1 and sum(asg) and r["conc"] >= len(asg):
            hi, lo = max(asg), min(asg)
            flag("worker_imbalance",
                 lo == 0 or hi / max(lo, 1) > TH["imbalance_ratio"],
                 f"C{r['conc']}: assignments {asg} (ratio threshold {TH['imbalance_ratio']})")

    if adm and adm.get("share", 0) > TH["admission_dominant"]:
        cliffs.append({"cliff": "admission_dominant",
                       "detail": f"queue wait {adm['admission_ms_per_req']:.1f} ms of "
                                 f"{adm['request_ms_per_req']:.1f} ms per request "
                                 f"({100*adm['share']:.0f}%)"})
    elif adm is None:
        cliffs.append({"cliff": "admission_wait", "detail": "UNKNOWN — no cost map or "
                                                            "[LIFE] trace in this run"})

    # Oversubscription: the engine runs three independent pools (qwen_parallel, the speech
    # decoder's private sd_pool, and BLAS) on one mask.  Expected OS threads ~ 3K-1 per
    # worker; a large excess plus a high context-switch rate is the evidence we have today.
    kthr = (topo or {}).get("threads_per_worker") or 0
    for r in rows:
        for w in r["workers"]:
            got, exp = w.get("threads") or 0, max(3 * kthr - 1, 1)
            flag("pool_oversubscription_suspected",
                 kthr and got > 1.5 * exp and (r["csw_s"] or 0) > 1000,
                 f"C{r['conc']} worker{w.get('worker')}: {got} OS threads vs ~{exp} expected "
                 f"for {kthr} engine threads across qwen_parallel+sd_pool+BLAS, "
                 f"{r['csw_s']:.0f} ctx-switches/s")

    # Bandwidth utilisation, only through the comparison contract.
    missing = []
    if not bw:
        missing.append("no census+cost-map pair for the serving worker")
    if not topo:
        missing.append("no topology.json, so the numerator's cpu mask is unknown")
    if not (store or {}).get("entries"):
        missing.append("no roofs measured for this hardware")
    bwrep = {"status": "UNKNOWN", "reason": "; ".join(missing) or "unknown"}
    if bw and topo:
        per = []
        for w in topo.get("workers", []):
            mask = w.get("actual_mask")
            for comp, v in bw.items():
                num = {"scope": f"WORKER[{w['worker']}]", "mask": mask, "bench": "read",
                       "residency": "dram", "hw_fingerprint": store.get("hw_fingerprint")}
                per.append({"worker": w["worker"], "component": comp,
                            **RF.compare(store, num, v["gbs"])})
            break     # the cost map describes the worker that served
        bwrep = {"status": "OK" if any(p["verdict"] == "OK" for p in per) else "UNKNOWN",
                 "per_component": per}
        utils = [p.get("efficiency_pct") for p in per if p.get("efficiency_pct")]
        if utils and len(topo.get("workers", [])) > 1:
            flag("one_domain_saturated_other_idle",
                 max(utils) > 100 * TH["domain_saturated"]
                 and any((r["workers"] and len(r["workers"]) > 1
                          and (r["workers"][1].get("cores_mean") or 0) < 0.5) for r in rows),
                 f"worker0 at {max(utils):.0f}% of its own roof while worker1 is near idle")

    return {
        "v": 1,
        "topology_id": (topo or {}).get("topology_id", "UNKNOWN"),
        "topology": {k: (topo or {}).get(k) for k in
                     ("mode", "worker_count", "threads_per_worker", "masks_overlap")},
        "worker_masks": [w.get("actual_mask") for w in (topo or {}).get("workers", [])],
        "thresholds": TH,
        "cells": rows,
        "derived": {
            "low_load_latency_ms": {"conc": n0, "ttfa_p50": base.get("ttfa_p50"),
                                    "ttfa_p95": base.get("ttfa_p95")},
            "sustainable_concurrency": sustainable,
            "saturation_knee_conc": knee,
            "post_knee_degradation": (
                {"ttfa_p95_slope_per_conc":
                     (post[-1]["ttfa_p95"] - post[0]["ttfa_p95"]) /
                     max(post[-1]["conc"] - post[0]["conc"], 1)} if len(post) > 1 else None),
            "p95_growth_max": max((r["ttfa_p95_growth"] for r in rows
                                   if finite(r["ttfa_p95_growth"])), default=None),
            "scaling_efficiency_at_max_conc": rows[-1]["scaling_efficiency"],
            "slo_goodput_peak_req_s": max((r["slo_goodput_req_s"] for r in rows), default=0.0),
        },
        "admission": adm or {"status": "UNKNOWN"},
        "bandwidth": bwrep,
        "cliffs": cliffs,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wave", nargs="+", required=True)
    ap.add_argument("--topology", default="")
    ap.add_argument("--store", default="")
    ap.add_argument("--hardware", default="")
    ap.add_argument("--census", nargs="*", default=[])
    ap.add_argument("--costmap", nargs="*", default=[])
    ap.add_argument("--poisson", nargs="*", default=[])
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    cells = []
    for doc in load_many(a.wave):
        cells.extend(doc if isinstance(doc, list) else [doc])
    topo = json.load(open(a.topology)) if a.topology and os.path.isfile(a.topology) else None
    hw = json.load(open(a.hardware)) if a.hardware and os.path.isfile(a.hardware) else None
    store = (RF.load_store(a.store, RF.hw_fingerprint(hw)) if a.store
             else {"hw_fingerprint": RF.hw_fingerprint(hw), "entries": []})
    bw = effective_bandwidth(load_many(a.census), load_many(a.costmap))
    adm = admission_share(load_many(a.costmap))

    env = build_envelope(cells, topo, store, bw, adm)
    if not env:
        print("no wave cells found"); return 1

    print("SCALING ENVELOPE")
    print(f"  topology id   {env['topology_id']}   masks {env['worker_masks'] or 'UNKNOWN'}")
    print(f"  {'C':>3}{'ok':>5}{'err':>5}{'rej':>5}{'TTFA50':>8}{'TTFA95':>8}"
          f"{'RTF50':>7}{'RTF95':>7}{'STR95':>7}{'req/s':>7}{'batch':>7}"
          f"{'scal':>6}{'goodput':>9}{'assign':>10}")
    for r in env["cells"]:
        def f(x, w, p=2):
            return f"{x:>{w}.{p}f}" if finite(x) else f"{'-':>{w}}"
        print(f"  {r['conc']:>3}{r['ok']:>5}{r['errors']:>5}{r['rejects']:>5}"
              f"{f(r['ttfa_p50'],8,0)}{f(r['ttfa_p95'],8,0)}"
              f"{f(r['rtf_p50'],7)}{f(r['rtf_p95'],7)}{f(r['stream_p95'],7)}"
              f"{f(r['req_s'],7)}{f(r['batch_eff'],7)}{f(r['scaling_efficiency'],6)}"
              f"{f(r['slo_goodput_req_s'],9)}{str(r['assign']):>10}")
    d = env["derived"]
    print()
    print(f"  low-load latency        TTFA p50 {d['low_load_latency_ms']['ttfa_p50']:.0f} ms "
          f"at C{d['low_load_latency_ms']['conc']}")
    print(f"  sustainable concurrency {d['sustainable_concurrency']}"
          f"   (RTF p95 < {TH['rt_limit']}, no errors/rejects/starvation)")
    print(f"  saturation knee         {d['saturation_knee_conc']}"
          f"   (scaling efficiency < {TH['knee_scaling_efficiency']})")
    print(f"  p95 growth (max)        {d['p95_growth_max']:.2f}x"
          if finite(d["p95_growth_max"]) else "  p95 growth (max)        -")
    print(f"  SLO goodput peak        {d['slo_goodput_peak_req_s']:.2f} req/s")

    print()
    print("  BANDWIDTH UTILISATION (through the comparison contract)")
    if env["bandwidth"].get("per_component"):
        for p in env["bandwidth"]["per_component"]:
            if p["verdict"] == "OK":
                print(f"    worker{p['worker']} {p['component']:<7} {p['effective_gbs']:.1f}"
                      f" / {p['roof_gbs']:.1f} GB/s = {p['efficiency_pct']:.0f}%"
                      f"   roof {p['roof_scope']} mask {p['roof_mask']}")
            else:
                print(f"    worker{p['worker']} {p['component']:<7} {p['verdict']}: {p['reason']}")
    else:
        print(f"    {env['bandwidth']['status']} — {env['bandwidth'].get('reason','')}")

    print()
    print("  CLIFFS")
    if not env["cliffs"]:
        print("    none flagged")
    for c in env["cliffs"]:
        print(f"    [{c['cliff']}] {c['detail']}")
    print()
    print("  thresholds are named policy, not hardware truth: " +
          ", ".join(f"{k}={v}" for k, v in TH.items()))

    if a.out:
        json.dump(env, open(a.out, "w"), indent=1)
        print(f"\n  machine-readable: {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
