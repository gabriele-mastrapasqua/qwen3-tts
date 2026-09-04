#!/usr/bin/env python3
"""topology.py — what execution domain did this run actually use?

A topology label like "2x8" or "1x8" is not an identity.  On a multi-CCX x86 part the
same label can mean different bandwidth domains:

  1 worker /  8 threads / mask 0-7    one CCX, one bandwidth domain
  1 worker /  8 threads / mask 0-15   8 threads free to roam BOTH domains
  1 worker / 16 threads / mask 0-15   16 threads over both domains
  2 workers / 8 threads / 0-7 + 8-15  two workers, one domain each

Those are four different experiments.  This module builds the identity that survives in
JSON, so a later comparison cannot silently treat them as the same configuration.

It joins information that already exists rather than adding machinery:
  * the engine's own `[TOPOLOGY] v=1 ...` line (configured AND actual mask, per process),
    with the legacy `prefork: worker N pid P cpus a-b threads K` line as a fallback for
    logs written before that line existed;
  * /proc/<pid>/status Cpus_allowed_list for the actual mask of a live process;
  * /sys/devices/system/cpu/*/cache/index3/shared_cpu_list for LLC domain membership,
    falling back to hardware.json's cache.l3_shared;
  * /sys/devices/system/node/ for NUMA membership.

    tools/topology.py --log server.log [--hardware hardware.json] [--out topology.json]
"""
import argparse, glob, json, os, re, sys


# ---- masks -------------------------------------------------------------------------

def mask_parse(spec):
    """'0-7', '0-3,8-11', '2' -> frozenset of ints.  Returns None when unparseable."""
    if not spec or spec in ("?", "unknown", "inherited", "all"):
        return None
    out = set()
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        m = re.fullmatch(r"(\d+)-(\d+)", part)
        if m:
            out.update(range(int(m.group(1)), int(m.group(2)) + 1))
            continue
        if part.isdigit():
            out.add(int(part))
            continue
        return None
    return frozenset(out) or None


def mask_str(cpus):
    """frozenset -> the compact form the kernel itself prints, so masks compare literally."""
    if not cpus:
        return "?"
    xs, out, i = sorted(cpus), [], 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[j] + 1:
            j += 1
        out.append(str(xs[i]) if i == j else f"{xs[i]}-{xs[j]}")
        i = j + 1
    return ",".join(out)


def mask_norm(spec):
    c = mask_parse(spec)
    return mask_str(c) if c else (spec or "?")


# ---- host ---------------------------------------------------------------------------

def _read(p):
    try:
        with open(p) as f:
            return f.read().strip()
    except OSError:
        return ""


def llc_domains(hardware=None):
    """[{id, cpus, size}] — the cache domains the masks will be tested against."""
    doms, seen = [], set()
    for p in sorted(glob.glob("/sys/devices/system/cpu/cpu*/cache/index*/level")):
        if _read(p) != "3":
            continue
        d = os.path.dirname(p)
        cl = _read(os.path.join(d, "shared_cpu_list"))
        if not cl or cl in seen:
            continue
        seen.add(cl)
        doms.append({"id": len(doms), "cpus": mask_norm(cl),
                     "size": _read(os.path.join(d, "size")) or "?"})
    if doms:
        return doms
    # No sysfs (macOS, or a container): hardware.json recorded the same relationship.
    for line in ((hardware or {}).get("cache", {}) or {}).get("l3_shared", []) or []:
        m = re.match(r"(\S+)\s+shared by cpus?\s+(.+)", line)
        if m:
            doms.append({"id": len(doms), "cpus": mask_norm(m.group(2)), "size": m.group(1)})
    return doms


def numa_nodes():
    out = []
    for p in sorted(glob.glob("/sys/devices/system/node/node*/cpulist")):
        nid = re.search(r"node(\d+)", p)
        out.append({"id": int(nid.group(1)) if nid else len(out), "cpus": mask_norm(_read(p))})
    return out


def host_domain(hardware=None):
    allowed = "?"
    st = _read(f"/proc/{os.getpid()}/status")
    for line in st.splitlines():
        if line.startswith("Cpus_allowed_list"):
            allowed = mask_norm(line.split(":", 1)[1].strip())
    hw = hardware or {}
    cpu = hw.get("cpu", {}) or {}
    return {
        "cpus_allowed": allowed,
        "cpu_count": len(mask_parse(allowed) or ()) or None,
        "cpu_model": cpu.get("model", "?"),
        "physical_cores": cpu.get("cores_physical"),
        "logical_cpus": cpu.get("cpus_logical"),
        "smt": cpu.get("smt", "?"),
        "sockets": cpu.get("sockets"),
        "llc_domains": llc_domains(hardware),
        "numa_nodes": numa_nodes(),
    }


def domains_of(mask, doms, key="cpus"):
    """Which LLC/NUMA domains a mask touches: [] unknown, [i] confined, [i,j] spanning."""
    c = mask_parse(mask)
    if not c or not doms:
        return []
    return [d["id"] for d in doms if mask_parse(d[key]) and (mask_parse(d[key]) & c)]


# ---- the run's own account of itself -------------------------------------------------

TOPO_RE = re.compile(
    r"\[TOPOLOGY\] v=1 worker=(\d+) pid=(\d+) configured_mask=(\S+) actual_mask=(\S+) "
    r"threads=(\d+) mode=(\S+)")
LEGACY_RE = re.compile(r"prefork: worker (\d+) pid (\d+) cpus ([\d\-,]+) threads (\d+)")


def workers_from_log(log_text):
    ws = [{"worker": int(m.group(1)), "pid": int(m.group(2)),
           "configured_mask": mask_norm(m.group(3)), "actual_mask_at_start": mask_norm(m.group(4)),
           "threads": int(m.group(5)), "mode": m.group(6), "source": "[TOPOLOGY]"}
          for m in TOPO_RE.finditer(log_text)]
    if ws:
        return ws
    # Older logs: the prefork line carries the configured mask only, and a single-process
    # run wrote nothing at all — which is exactly why [TOPOLOGY] was added.
    return [{"worker": int(m.group(1)), "pid": int(m.group(2)),
             "configured_mask": mask_norm(m.group(3)), "actual_mask_at_start": None,
             "threads": int(m.group(4)), "mode": "prefork", "source": "legacy prefork line"}
            for m in LEGACY_RE.finditer(log_text)]


def actual_mask_now(pid):
    for line in _read(f"/proc/{pid}/status").splitlines():
        if line.startswith("Cpus_allowed_list"):
            return mask_norm(line.split(":", 1)[1].strip())
    return None


def build(log_path=None, hardware=None, pids=None, threads_hint=None):
    text = ""
    if log_path and os.path.exists(log_path):
        with open(log_path, "rb") as f:
            text = f.read().decode("utf-8", "replace")
    ws = workers_from_log(text)

    # A single-process run whose log predates [TOPOLOGY]: keep it as a worker entry
    # instead of losing it, which is what every harness used to do.
    if not ws and pids:
        ws = [{"worker": i, "pid": p, "configured_mask": "inherited",
               "actual_mask_at_start": None, "threads": threads_hint or 0,
               "mode": "single", "source": "caller-supplied pid"} for i, p in enumerate(pids)]

    host = host_domain(hardware)
    lldoms, nodes = host["llc_domains"], host["numa_nodes"]
    for w in ws:
        live = actual_mask_now(w["pid"])
        # Three sources, best first, and the one that was used is always named.  An
        # archived run whose process is gone and whose log predates [TOPOLOGY] still has
        # the CONFIGURED mask, which is better than discarding the run — but it is
        # labelled `configured-only`, because "what was asked for" is not evidence of
        # "what the OS gave".
        if live:
            w["actual_mask"], w["actual_mask_source"] = live, "/proc/<pid>/status"
            w["mask_confidence"] = "observed"
        elif w.get("actual_mask_at_start"):
            w["actual_mask"] = w["actual_mask_at_start"]
            w["actual_mask_source"] = "[TOPOLOGY] line, at process start"
            w["mask_confidence"] = "observed"
        elif w.get("configured_mask") and w["configured_mask"] not in ("inherited", "?"):
            w["actual_mask"] = w["configured_mask"]
            w["actual_mask_source"] = "configured only (legacy log, process gone)"
            w["mask_confidence"] = "configured-only"
        else:
            w["actual_mask"], w["actual_mask_source"] = "unknown", "UNKNOWN"
            w["mask_confidence"] = "unknown"
        c = mask_parse(w["actual_mask"])
        w["cpu_count"] = len(c) if c else None
        w["llc_domains"] = domains_of(w["actual_mask"], lldoms)
        w["llc_span"] = ("unknown" if not w["llc_domains"] else
                         "single" if len(w["llc_domains"]) == 1 else "spanning")
        w["numa_nodes"] = domains_of(w["actual_mask"], nodes)

    overlap = False
    sets = [mask_parse(w["actual_mask"]) for w in ws]
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            if sets[i] and sets[j] and (sets[i] & sets[j]):
                overlap = True

    nthr = ws[0]["threads"] if ws else (threads_hint or 0)
    tid = (f"{len(ws)}W{nthr}T_m" + "|".join(w["actual_mask"] for w in ws)) if ws else "UNKNOWN"
    return {
        "v": 1,
        "source_log": log_path,
        "host": host,
        "mode": ws[0]["mode"] if ws else "unknown",
        "worker_count": len(ws),
        "threads_per_worker": nthr,
        "workers": ws,
        "masks_overlap": overlap,
        # The identity a later comparison must key on.  "1x8" is not enough: it does not
        # say whether those 8 threads were confined to one bandwidth domain or not.
        "topology_id": tid,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="")
    ap.add_argument("--hardware", default="")
    ap.add_argument("--pids", default="", help="comma-separated, for logs with no topology line")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    hw = json.load(open(a.hardware)) if a.hardware and os.path.exists(a.hardware) else None
    pids = [int(x) for x in a.pids.split(",") if x.strip().isdigit()]
    doc = build(a.log or None, hw, pids or None, a.threads or None)
    txt = json.dumps(doc, indent=1)
    if a.out:
        open(a.out, "w").write(txt + "\n")
    print(txt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
