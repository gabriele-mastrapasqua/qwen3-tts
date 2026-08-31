#!/usr/bin/env python3
"""serve_procstats.py — per-worker facts, read from /proc instead of guessed from htop."""
import os, re

def _read(p):
    try:
        return open(p).read()
    except OSError:
        return ""

def worker_pids_from_log(log_path):
    """[(index, pid, declared_cpu_range)] from the parent's startup lines.

    The parent prints `prefork: worker <i> pid <p> cpus <a>-<b> threads <k>`; parsing
    that is what ties a dispatcher worker INDEX to a pid, which /proc alone cannot do.
    Returns [] for a single-process topology, where the caller uses the pid it spawned.
    """
    out = []
    for m in re.finditer(r"prefork: worker (\d+) pid (\d+) cpus ([\d\-]+) threads (\d+)",
                         _read(log_path)):
        out.append((int(m.group(1)), int(m.group(2)), m.group(3), int(m.group(4))))
    return out

def proc_sample(pid):
    """One snapshot of a process: CPU ticks, affinity, threads, memory, switches."""
    d = {"pid": pid, "utime": 0, "stime": 0, "nthreads": 0, "cpus": "?",
         "rss_kb": 0, "pss_kb": 0, "vol": 0, "nonvol": 0, "alive": False}
    st = _read(f"/proc/{pid}/stat")
    if not st:
        return d
    d["alive"] = True
    try:
        f = st.rsplit(") ", 1)[1].split()
        d["utime"], d["stime"], d["nthreads"] = int(f[11]), int(f[12]), int(f[17])
    except Exception:
        pass
    for line in _read(f"/proc/{pid}/status").splitlines():
        if line.startswith("Cpus_allowed_list"):
            d["cpus"] = line.split(":", 1)[1].strip()
        elif line.startswith("VmHWM"):
            d["rss_kb"] = int(re.search(r"(\d+)", line).group(1))
    for t in os.listdir(f"/proc/{pid}/task") if os.path.isdir(f"/proc/{pid}/task") else []:
        s = _read(f"/proc/{pid}/task/{t}/status")
        for line in s.splitlines():
            if line.startswith("voluntary_ctxt_switches"):
                d["vol"] += int(re.search(r"(\d+)", line).group(1))
            elif line.startswith("nonvoluntary_ctxt_switches"):
                d["nonvol"] += int(re.search(r"(\d+)", line).group(1))
    for line in _read(f"/proc/{pid}/smaps_rollup").splitlines():
        m = re.match(r"(\w+):\s+(\d+) kB", line)
        if m and m.group(1) == "Pss":
            d["pss_kb"] = int(m.group(2))
    return d

def parse_prefork_stats(line):
    """The dispatcher's SIGUSR1 line -> {worker: {assigned, completed, active, B}}.

    Format (qwen_tts_server.c):
      [prefork-stats] mean_inflight X dispatched N rejected M · w0[asg=.. done=.. act=.. B=..] ...
    `B` is the TIME-WEIGHTED mean in-flight for that worker over the level, which is
    the effective batch it actually saw. A sample taken at the end of the level would
    miss the shape of the wave entirely.
    """
    out = {"workers": {}, "mean_inflight": None, "dispatched": None, "rejected": None}
    if not line:
        return out
    m = re.search(r"mean_inflight ([\d.]+)", line)
    if m: out["mean_inflight"] = float(m.group(1))
    m = re.search(r"dispatched (\d+)", line)
    if m: out["dispatched"] = int(m.group(1))
    m = re.search(r"rejected (\d+)", line)
    if m: out["rejected"] = int(m.group(1))
    for m in re.finditer(r"w(\d+)\[asg=(\d+) done=(\d+) act=(-?\d+)(?: B=([\d.]+))?\]", line):
        out["workers"][int(m.group(1))] = {
            "assigned": int(m.group(2)), "completed": int(m.group(3)),
            "active_end": int(m.group(4)),
            "B": float(m.group(5)) if m.group(5) else float("nan"),
        }
    return out

def per_worker_rows(before, after, stats, wall_s, hz=None):
    """Join a /proc delta with the dispatcher's own counters into printable rows."""
    hz = hz or os.sysconf("SC_CLK_TCK")
    rows = []
    for idx in sorted(before):
        b, a = before[idx], after.get(idx, before[idx])
        w = stats.get("workers", {}).get(idx, {})
        core_s = ((a["utime"] - b["utime"]) + (a["stime"] - b["stime"])) / hz
        rows.append({
            "worker": idx, "pid": a["pid"], "cpus": a["cpus"], "threads": a["nthreads"],
            "assigned": w.get("assigned", 0), "completed": w.get("completed", 0),
            "active_end": w.get("active_end", 0), "B": w.get("B", float("nan")),
            "core_s": core_s,
            "cores_mean": core_s / wall_s if wall_s > 0 else 0.0,
            "csw_s": (((a["vol"] - b["vol"]) + (a["nonvol"] - b["nonvol"])) / wall_s)
                     if wall_s > 0 else 0.0,
            "pss_mb": a["pss_kb"] / 1024.0,
        })
    return rows

def format_rows(rows, indent="      "):
    if not rows:
        return ""
    out = [f"{indent}{'w':>2} {'pid':>7} {'cpus':>7} {'thr':>4} {'asg':>4} {'done':>5} "
           f"{'B':>5} {'core-s':>8} {'cores':>6} {'csw/s':>8} {'PSS MB':>8}"]
    for r in rows:
        out.append(f"{indent}{r['worker']:>2} {r['pid']:>7} {r['cpus']:>7} {r['threads']:>4} "
                   f"{r['assigned']:>4} {r['completed']:>5} {r['B']:>5.2f} {r['core_s']:>8.1f} "
                   f"{r['cores_mean']:>6.2f} {r['csw_s']:>8.0f} {r['pss_mb']:>8.0f}")
    return "\n".join(out)
