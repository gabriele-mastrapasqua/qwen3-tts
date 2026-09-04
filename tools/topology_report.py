#!/usr/bin/env python3
"""topology_report.py — the doctor block that makes the execution domain impossible to miss.

    tools/topology_report.py --hardware hardware.json [--topology topology.json]
                             [--store profiles/roofs] [--compare effective.json]

Prints SERVER TOPOLOGY, BANDWIDTH (per mask, with the full saturation curve) and, when a
numerator is supplied, CURRENT COMPARISON with the roof selected by the contract in
roofs.select_roof — including its refusals.  A missing roof prints ROOF UNKNOWN; a roof
from another execution domain prints NOT COMPARABLE.  Neither ever silently becomes a
host-roof percentage.

--compare takes a JSON list of numerators:
    [{"label":"talker decode","scope":"WORKER[0]","mask":"0-7","bench":"read",
      "residency":"dram","effective_gbs":59.2}]
"""
import argparse, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import topology as TP                                            # noqa: E402
import roofs as RF                                               # noqa: E402


def fmt_sweep(e):
    """The curve, never collapsed to its peak: on the c8a's cpus 0-7 two threads already
    reach ~97% of the local roof, which decides topology questions on its own."""
    parts = [f"{s['threads']}T {s['gbs']:.1f}" for s in e.get("sweep", [])
             if s.get("gbs") is not None]
    return "  ".join(parts) if parts else "(no sweep)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hardware", default="")
    ap.add_argument("--topology", default="")
    ap.add_argument("--store", default="")
    ap.add_argument("--compare", default="")
    a = ap.parse_args()

    hw = json.load(open(a.hardware)) if a.hardware and os.path.isfile(a.hardware) else None
    topo = (json.load(open(a.topology))
            if a.topology and os.path.isfile(a.topology) else None)
    host = topo["host"] if topo else TP.host_domain(hw)
    fp = RF.hw_fingerprint(hw)
    store = RF.load_store(a.store, fp) if a.store else {"hw_fingerprint": fp, "entries": []}

    print("SERVER TOPOLOGY")
    print(f"  host allowed CPUs   {host.get('cpus_allowed','?')}"
          f"   ({host.get('cpu_count') or '?'} cpus)")
    print(f"  physical cores      {host.get('physical_cores','?')}"
          f"   logical {host.get('logical_cpus','?')}   SMT {host.get('smt','?')}")
    print(f"  sockets             {host.get('sockets','?')}"
          f"   NUMA nodes {len(host.get('numa_nodes') or [])}")
    for d in host.get("llc_domains") or []:
        print(f"  LLC domain {d['id']}        cpus {d['cpus']}   {d['size']}")
    if not host.get("llc_domains"):
        print("  LLC domains         UNKNOWN (no sysfs cache topology)")
    print(f"  hardware fingerprint {fp}")

    if topo:
        print()
        print(f"  mode                {topo.get('mode','?')}"
              f"   workers {topo.get('worker_count','?')}"
              f"   threads/worker {topo.get('threads_per_worker','?')}")
        print(f"  topology id         {topo.get('topology_id','?')}")
        for w in topo.get("workers", []):
            print(f"  worker{w['worker']} pid {w['pid']}")
            print(f"    configured mask   {w.get('configured_mask','?')}")
            print(f"    actual mask       {w.get('actual_mask','?')}"
                  f"   ({w.get('cpu_count') or '?'} cpus, via {w.get('actual_mask_source','?')},"
                  f" confidence {w.get('mask_confidence','?')})")
            print(f"    LLC domain(s)     {w.get('llc_domains')}  span={w.get('llc_span','?')}")
            if w.get("numa_nodes"):
                print(f"    NUMA node(s)      {w.get('numa_nodes')}")
        print(f"  masks overlap       {'YES' if topo.get('masks_overlap') else 'none'}")
    else:
        print("\n  (no topology.json supplied: worker masks UNKNOWN for this run)")

    print()
    print("BANDWIDTH  (each roof belongs to ONE mask; a worker roof is never a divided host roof)")
    if not store.get("entries"):
        print("  ROOF UNKNOWN — no roofs measured for this hardware fingerprint")
        print("  measure with: tools/roofs.py measure --masks <host>,<worker0>,<worker1>")
    else:
        by_mask = {}
        for e in store["entries"]:
            by_mask.setdefault(TP.mask_norm(e["cpu_mask"]), []).append(e)
        for mask in sorted(by_mask):
            scopes = sorted({e["scope"] for e in by_mask[mask]})
            print(f"  mask {mask}   [{', '.join(scopes)}]")
            for e in sorted(by_mask[mask], key=lambda e: e["bench"]):
                print(f"    {e['bench']:<6}{e['gbs']:>8.1f} GB/s  peak@{e.get('threads')}T"
                      f"   90%@{e.get('t90')}T  95%@{e.get('t95')}T  99%@{e.get('t99')}T"
                      f"   [{e.get('residency','?')}, {e.get('access','?')}]")
                print(f"           saturation: {fmt_sweep(e)}")

    if a.compare and os.path.isfile(a.compare):
        print()
        print("CURRENT COMPARISON")
        for num in json.load(open(a.compare)):
            num.setdefault("hw_fingerprint", fp)
            eff = num.pop("effective_gbs", None)
            label = num.pop("label", "numerator")
            res = RF.compare(store, num, eff) if eff is not None else None
            print(f"  {label}")
            print(f"    numerator scope   {num.get('scope','?')}   mask {num.get('mask','?')}"
                  f"   bench {num.get('bench','?')}   residency {num.get('residency','?')}")
            if res is None:
                print("    VERDICT           ROOF UNKNOWN (no effective_gbs given)")
                continue
            if res["verdict"] != "OK":
                print(f"    VERDICT           {res['verdict']}")
                print(f"    reason            {res['reason']}")
                print(f"    effective         {eff:.1f} GB/s")
                print("    host comparison   NOT APPLICABLE — a host roof is not a "
                      "substitute for the roof of this mask")
                continue
            print(f"    selected roof     {res['roof_scope']}  mask {res['roof_mask']}"
                  f"  {res['roof_bench']}  peak@{res.get('roof_threads')}T")
            print(f"    effective         {eff:.1f} GB/s")
            print(f"    roof              {res['roof_gbs']:.1f} GB/s")
            print(f"    efficiency        {res['efficiency_pct']:.0f}%")
            hostm = TP.mask_norm(host.get("cpus_allowed", "?"))
            if TP.mask_norm(res["roof_mask"]) != hostm:
                print(f"    host comparison   NOT APPLICABLE (host mask {hostm} is a "
                      f"different execution domain)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
