#!/usr/bin/env python3
"""serve_memory_probe.py — how much memory does a multi-worker server ACTUALLY use?

Summing RSS over W processes counts every SHARED page W times. With pre-fork that is
exactly the wrong thing to measure, because sharing is the whole point: the topology
matrix reported 47.8 GB for 4x4 by summing RSS, and most of it was one physical copy
counted four times.

Pss (Proportional Set Size) divides each page by the number of processes mapping it,
so summing Pss across the tree gives the true footprint. This prints both, side by
side, plus the shared/private split that says WHERE the sharing is.

Usage:
  python3 tests/serve_memory_probe.py <pid>            # that process and its children
  python3 tests/serve_memory_probe.py --match qwen_tts # every matching process
"""
import os, re, sys


def rollup(pid):
    out = {}
    try:
        for line in open(f"/proc/{pid}/smaps_rollup"):
            m = re.match(r"(\w+):\s+(\d+) kB", line)
            if m:
                out[m.group(1)] = int(m.group(2))
    except OSError:
        return None
    try:
        out["cmd"] = open(f"/proc/{pid}/cmdline").read().replace("\0", " ").strip()[:60]
    except OSError:
        out["cmd"] = "?"
    return out


def children(pid):
    kids = []
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            st = open(f"/proc/{d}/stat").read().rsplit(") ", 1)[1].split()
            if int(st[1]) == int(pid):
                kids.append(int(d))
        except Exception:
            pass
    return kids


def main():
    if len(sys.argv) >= 3 and sys.argv[1] == "--match":
        pids = []
        for d in os.listdir("/proc"):
            if not d.isdigit():
                continue
            try:
                if sys.argv[2] in open(f"/proc/{d}/cmdline").read().replace("\0", " "):
                    pids.append(int(d))
            except Exception:
                pass
    else:
        root = int(sys.argv[1])
        pids = [root] + children(root)

    hdr = f"{'pid':>8}{'Rss MB':>9}{'Pss MB':>9}{'ShClean':>9}{'ShDirty':>9}{'PrClean':>9}{'PrDirty':>9}  cmd"
    print(hdr); print("-" * len(hdr))
    tot = {k: 0 for k in ("Rss", "Pss", "Shared_Clean", "Shared_Dirty",
                          "Private_Clean", "Private_Dirty")}
    n = 0
    for p in sorted(pids):
        r = rollup(p)
        if not r or "Pss" not in r:
            continue
        n += 1
        for k in tot:
            tot[k] += r.get(k, 0)
        print(f"{p:>8}{r.get('Rss',0)/1024:>9.0f}{r.get('Pss',0)/1024:>9.0f}"
              f"{r.get('Shared_Clean',0)/1024:>9.0f}{r.get('Shared_Dirty',0)/1024:>9.0f}"
              f"{r.get('Private_Clean',0)/1024:>9.0f}{r.get('Private_Dirty',0)/1024:>9.0f}"
              f"  {r['cmd']}")
    if not n:
        print("no processes found"); return
    print("-" * len(hdr))
    print(f"{'SUM':>8}{tot['Rss']/1024:>9.0f}{tot['Pss']/1024:>9.0f}"
          f"{tot['Shared_Clean']/1024:>9.0f}{tot['Shared_Dirty']/1024:>9.0f}"
          f"{tot['Private_Clean']/1024:>9.0f}{tot['Private_Dirty']/1024:>9.0f}"
          f"  ({n} processes)")
    print()
    print(f"  sum(RSS) = {tot['Rss']/1024:.0f} MB   <- counts every shared page once PER PROCESS")
    print(f"  sum(PSS) = {tot['Pss']/1024:.0f} MB   <- the real footprint")
    if tot["Rss"]:
        print(f"  sharing saves {(tot['Rss']-tot['Pss'])/1024:.0f} MB "
              f"({100.0*(tot['Rss']-tot['Pss'])/tot['Rss']:.0f}% of the naive sum)")
    # /proc/meminfo is the final arbiter: if Pss is right, MemAvailable agrees.
    mi = {}
    for line in open("/proc/meminfo"):
        m = re.match(r"(\w+):\s+(\d+) kB", line)
        if m: mi[m.group(1)] = int(m.group(2))
    print(f"  system: MemTotal {mi.get('MemTotal',0)/1048576:.1f} GB, "
          f"MemAvailable {mi.get('MemAvailable',0)/1048576:.1f} GB, "
          f"used {(mi.get('MemTotal',0)-mi.get('MemAvailable',0))/1048576:.1f} GB")


if __name__ == "__main__":
    main()
