#!/usr/bin/env python3
"""roofs.py — a bandwidth roof is not a scalar, and it is not divisible.

Measured on an AWS c8a.4xlarge: the host triads at ~113 GB/s, a process confined to
cpus 0-7 at ~54 GB/s, one confined to cpus 8-15 at ~54 GB/s.  A worker roof can therefore
never be obtained by dividing a host roof — not here, and not on Axion or any other part,
where the fabric may split very differently.  The only valid worker roof is one measured
under that worker's own mask.

Every entry carries what makes it comparable, so a comparison can be CHECKED and not
assumed: scope, cpu_mask, threads, bench type, working set, residency, the full thread
sweep and t90/t95/t99, plus provenance.

    tools/roofs.py measure --masks 0-15,0-7,8-15 [--membw ./membw] [--store DIR]
    tools/roofs.py show    [--store DIR]

Roofs are a property of (hardware, mask, benchmark) and NOT of the model, so they are
cached per hardware fingerprint and reused across runs and model switches.  They are
re-measured when that fingerprint changes, or when the benchmark's own source changes.
"""
import argparse, hashlib, json, os, platform, re, subprocess, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import topology as TP                                            # noqa: E402

BENCHES = ("copy", "triad", "read")


def sha256_file(p):
    if not os.path.isfile(p):
        return ""
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def hw_fingerprint(hardware=None):
    """Only what changes comparability: the silicon and its cache/NUMA geometry.
    Deliberately NOT the model, the binary or the source tree — a roof does not depend
    on those, and folding them in would force a pointless re-measurement on every build."""
    host = TP.host_domain(hardware)
    key = json.dumps({
        "cpu_model": host.get("cpu_model"),
        "physical_cores": host.get("physical_cores"),
        "logical_cpus": host.get("logical_cpus"),
        "smt": host.get("smt"),
        "sockets": host.get("sockets"),
        "llc": [(d["cpus"], d["size"]) for d in host.get("llc_domains", [])],
        "numa": [(n["cpus"]) for n in host.get("numa_nodes", [])],
        "machine": platform.machine(),
    }, sort_keys=True)
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def store_path(store_dir, fp):
    return os.path.join(store_dir, f"roofs_{fp}.json")


def load_store(store_dir, fp):
    p = store_path(store_dir, fp)
    if os.path.isfile(p):
        try:
            return json.load(open(p))
        except Exception:
            pass
    return {"v": 1, "hw_fingerprint": fp, "entries": []}


def save_store(store_dir, doc):
    os.makedirs(store_dir, exist_ok=True)
    p = store_path(store_dir, doc["hw_fingerprint"])
    json.dump(doc, open(p, "w"), indent=1)
    return p


def run_membw(membw, mask, threads=None, reps=5, l3_mb=None, label=""):
    cmd = [membw, "--json", "--reps", str(reps)]
    if mask and mask != "all":
        cmd += ["--cpus", mask]
    if threads:
        cmd += ["--threads", threads]
    if l3_mb:
        cmd += ["--l3-mb", str(l3_mb)]
    if label:
        cmd += ["--label", label]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if r.returncode != 0 or not r.stdout.strip():
        raise RuntimeError(f"membw failed for mask {mask}: {r.stderr.strip()[:200]}")
    return json.loads(r.stdout.strip().splitlines()[-1])


def entries_from_membw(doc, scope, membw_bin, membw_src):
    """One membw invocation describes ONE mask and yields one entry per benchmark kind:
    they are different ceilings (read-only vs read+write) and must not be merged."""
    out = []
    prov = {
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": platform.node(),
        "os": platform.system() + " " + platform.release(),
        "membw_binary_sha256": sha256_file(membw_bin),
        "membw_source_sha256": sha256_file(os.path.join(ROOT, "tests", "membw.c")),
        "membw_v": doc.get("v", 1),
    }
    for b in BENCHES:
        sub = doc.get(b)
        if not sub:
            continue                       # v1 membw: no per-kind block, no read kernel
        out.append({
            "scope": scope,
            "cpu_mask": doc.get("cpu_mask", "?"),
            "cpu_count": doc.get("cpu_count"),
            "threads": sub.get("peak_threads"),
            "bench": b,
            "access": sub.get("access", "?"),
            "working_set_mib": doc.get("working_set_mib"),
            "residency": doc.get("residency", "?"),
            "gbs": sub.get("peak_gbs"),
            "sweep": [{"threads": s["threads"], "gbs": s.get(f"{b}_gbs")}
                      for s in doc.get("sweep", [])],
            "t90": sub.get("t90"), "t95": sub.get("t95"), "t99": sub.get("t99"),
            "provenance": prov,
        })
    return out


def upsert(store, entries):
    """Key = (scope, cpu_mask, bench, residency).  A re-measurement replaces its own key
    and nothing else, so a worker roof can be added later without disturbing the host."""
    def key(e):
        return (e["scope"], e["cpu_mask"], e["bench"], e.get("residency"))
    idx = {key(e): i for i, e in enumerate(store["entries"])}
    for e in entries:
        k = key(e)
        if k in idx:
            store["entries"][idx[k]] = e
        else:
            idx[k] = len(store["entries"])
            store["entries"].append(e)
    return store


def stale(store, membw_bin):
    """Re-measure when the benchmark itself changed; the hardware fingerprint is already
    the store's filename, so a different machine simply uses a different store."""
    src = sha256_file(os.path.join(ROOT, "tests", "membw.c"))
    for e in store.get("entries", []):
        if e.get("provenance", {}).get("membw_source_sha256") != src:
            return True
    return False


# ---- B4: the comparison contract ----------------------------------------------------

def select_roof(store, numerator):
    """Return (entry, verdict, reason).

    verdict is one of:
      OK              a roof with a compatible execution scope exists
      ROOF UNKNOWN    no roof has been measured for this scope — say so, do not substitute
      NOT COMPARABLE  a roof exists but describes a different execution domain

    There is deliberately NO fallback from WORKER to HOST.  The comparison that started
    this work divided a numerator confined to cpus 0-7 (~59 GB/s) by a host roof measured
    over cpus 0-15 (~113 GB/s) and concluded "57% of roof" when the honest answer was
    "at roof".  That mistake is mechanical, so the guard against it must be mechanical too.
    """
    want_scope = numerator.get("scope")
    want_mask = TP.mask_norm(numerator.get("mask") or "?")
    want_bench = numerator.get("bench", "read")
    want_res = numerator.get("residency", "dram")
    want_fp = numerator.get("hw_fingerprint")

    if want_fp and store.get("hw_fingerprint") and want_fp != store["hw_fingerprint"]:
        return None, "NOT COMPARABLE", (
            f"different hardware: numerator fingerprint {want_fp}, "
            f"roof store {store['hw_fingerprint']}")
    if not want_mask or want_mask == "?":
        return None, "ROOF UNKNOWN", "the numerator does not state the mask it ran on"

    cands = [e for e in store.get("entries", []) if e.get("bench") == want_bench]
    if not cands:
        return None, "ROOF UNKNOWN", (
            f"no {want_bench} roof has been measured on this host — measure it; "
            f"there is no fallback")

    exact = [e for e in cands if TP.mask_norm(e.get("cpu_mask", "?")) == want_mask]
    if not exact:
        # Roofs exist, but every one of them describes a DIFFERENT execution domain.
        # This is the shape of the original mistake — a numerator confined to cpus 0-7
        # against a roof measured over cpus 0-15 — so it is refused as a scope error,
        # not softened into "unknown" and certainly not substituted.
        have = sorted({TP.mask_norm(e.get("cpu_mask", "?")) for e in cands})
        return None, "NOT COMPARABLE", (
            f"numerator ran on mask {want_mask}; the measured {want_bench} roofs describe "
            f"mask(s) {', '.join(have)} — different execution domain(s). "
            f"Measure mask {want_mask}; never divide or substitute another mask's roof")

    same_res = [e for e in exact if e.get("residency") == want_res]
    if not same_res:
        return None, "NOT COMPARABLE", (
            f"mask {want_mask} has a {want_bench} roof but at residency "
            f"{exact[0].get('residency')}, numerator is {want_res}")

    e = same_res[0]
    if want_scope and e["scope"] != want_scope:
        # Same mask, different label (e.g. a HOST roof measured on 0-7 on a smaller box):
        # allowed, but say it, because the scopes are only equivalent because the masks are.
        return e, "OK", (f"scope label differs ({e['scope']} vs {want_scope}) but the "
                         f"execution domain is identical (mask {want_mask})")
    return e, "OK", f"mask {want_mask}, {want_bench}, {want_res}-resident"


def compare(store, numerator, effective_gbs):
    e, verdict, reason = select_roof(store, numerator)
    out = {"effective_gbs": effective_gbs, "verdict": verdict, "reason": reason,
           "numerator": dict(numerator)}
    if verdict == "OK" and e and e.get("gbs"):
        out.update({"roof_gbs": e["gbs"], "roof_scope": e["scope"],
                    "roof_mask": e["cpu_mask"], "roof_bench": e["bench"],
                    "roof_threads": e.get("threads"),
                    "efficiency_pct": 100.0 * effective_gbs / e["gbs"]})
    return out


# ---- CLI -----------------------------------------------------------------------------

def cmd_measure(a):
    hw = json.load(open(a.hardware)) if a.hardware and os.path.isfile(a.hardware) else None
    fp = hw_fingerprint(hw)
    store = load_store(a.store, fp)
    if stale(store, a.membw) and not a.no_invalidate:
        print(f"  membw source changed since these roofs were taken -> re-measuring all")
        store["entries"] = []
    host_mask = TP.host_domain(hw)["cpus_allowed"]
    masks = [m.strip() for m in a.masks.split(",") if m.strip()] if a.masks else [host_mask]
    for m in masks:
        scope = "HOST" if TP.mask_norm(m) == TP.mask_norm(host_mask) else f"WORKER[{m}]"
        have = [e for e in store["entries"] if TP.mask_norm(e["cpu_mask"]) == TP.mask_norm(m)]
        if have and not a.force:
            print(f"  mask {m}: cached ({len(have)} entries) — --force to re-measure")
            continue
        print(f"  measuring mask {m} ({scope}) ...", flush=True)
        doc = run_membw(a.membw, m, a.threads, a.reps, a.l3_mb, label=scope)
        upsert(store, entries_from_membw(doc, scope, a.membw, None))
    p = save_store(a.store, store)
    print(f"  roofs -> {p}  ({len(store['entries'])} entries, hw {fp})")
    return 0


def cmd_show(a):
    hw = json.load(open(a.hardware)) if a.hardware and os.path.isfile(a.hardware) else None
    fp = hw_fingerprint(hw)
    store = load_store(a.store, fp)
    if not store["entries"]:
        print(f"  no roofs measured for hardware {fp}")
        return 1
    print(f"  roofs for hardware {fp}")
    print(f"    {'scope':<16}{'mask':<10}{'bench':<7}{'GB/s':>8}{'@T':>4}"
          f"{'t90':>5}{'t95':>5}{'t99':>5}  {'residency':<9} sweep")
    for e in sorted(store["entries"], key=lambda e: (e["cpu_mask"], e["bench"])):
        sw = " ".join(f"{s['threads']}T:{s['gbs']:.0f}" for s in e.get("sweep", [])
                      if s.get("gbs") is not None)
        print(f"    {e['scope']:<16}{e['cpu_mask']:<10}{e['bench']:<7}{e['gbs'] or 0:>8.1f}"
              f"{e.get('threads') or 0:>4}{e.get('t90') or 0:>5}{e.get('t95') or 0:>5}"
              f"{e.get('t99') or 0:>5}  {e.get('residency','?'):<9} {sw}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("measure", "show"):
        p = sub.add_parser(name)
        p.add_argument("--store", default=os.path.join(ROOT, "profiles", "roofs"))
        p.add_argument("--hardware", default="")
        if name == "measure":
            p.add_argument("--membw", default=os.path.join(ROOT, "membw"))
            p.add_argument("--masks", default="")
            p.add_argument("--threads", default="")
            p.add_argument("--reps", type=int, default=5)
            p.add_argument("--l3-mb", type=int, default=0)
            p.add_argument("--force", action="store_true")
            p.add_argument("--no-invalidate", action="store_true")
    a = ap.parse_args()
    return cmd_measure(a) if a.cmd == "measure" else cmd_show(a)


if __name__ == "__main__":
    sys.exit(main())
