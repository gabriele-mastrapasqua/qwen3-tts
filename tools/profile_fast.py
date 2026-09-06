#!/usr/bin/env python3
"""profile_fast.py — ONE command, ONE artifact: the FAST engine profile.

Runs the REAL production path (batched streaming server + prefork, declared topology), makes a
couple of seconds of audio, and writes a single artifact directory holding what it takes to say
where the request time went WITHOUT reading the source or grepping logs:

    profile.md            hierarchical report, hotspots, occupancy, findings
    profile.json          the same, machine-readable
    effective-config.txt  what the engine is ACTUALLY honouring  (--effective-config)
    dispatch-map.txt      the resolved dispatch                  (--dispatch-map)
    server.log            the run's own stderr, provenance included
    costmap/*.json        the raw per-thread region dumps

It consumes the existing machinery rather than duplicating it: --effective-config,
--dispatch-map, the cost map, and the server's own topology reporting.

    tools/profile_fast.py --model qwen3-tts-1.7b --conc 4 --out profiles/fast-c4
"""
import argparse, hashlib, json, os, re, shutil, signal, socket, subprocess, sys, time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEXT = ("The quick brown fox jumps over the lazy dog. "
        "A short sentence is enough to profile the whole path.")


def sh(cmd, **kw):
    return subprocess.run(cmd, shell=isinstance(cmd, str), capture_output=True,
                          text=True, cwd=ROOT, **kw)


def file_sha(path, n=16):
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for b in iter(lambda: f.read(1 << 20), b""):
                h.update(b)
        return h.hexdigest()[:n]
    except OSError:
        return None


def host_topology():
    t = {}
    o = sh("lscpu").stdout
    for key, pat in (("cpu_model", r"Model name:\s*(.+)"),
                     ("sockets", r"Socket\(s\):\s*(\d+)"),
                     ("cores_per_socket", r"Core\(s\) per socket:\s*(\d+)"),
                     ("threads_per_core", r"Thread\(s\) per core:\s*(\d+)"),
                     ("numa_nodes", r"NUMA node\(s\):\s*(\d+)"),
                     ("logical_cpus", r"^CPU\(s\):\s*(\d+)")):
        m = re.search(pat, o, re.M)
        if m:
            t[key] = m.group(1).strip()
    for k, p in (("smt_control", "/sys/devices/system/cpu/smt/control"),
                 ("online_cpus", "/sys/devices/system/cpu/online")):
        try:
            t[k] = open(p).read().strip()
        except OSError:
            t[k] = "unavailable"
    try:
        t["physical_cores"] = int(t.get("cores_per_socket", 0)) * int(t.get("sockets", 1))
    except (TypeError, ValueError):
        pass
    return t


def free_port(p):
    s = socket.socket()
    try:
        s.bind(("127.0.0.1", p)); return p
    except OSError:
        s.close(); s = socket.socket(); s.bind(("127.0.0.1", 0))
        p = s.getsockname()[1]
    finally:
        s.close()
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--bin", default="./qwen_tts")
    ap.add_argument("--out", required=True)
    ap.add_argument("--conc", type=int, default=4)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--port", type=int, default=9800)
    ap.add_argument("--level", type=int, default=1, help="cost-map level: 1 FAST, 2 DEEP")
    ap.add_argument("--deep-pass", action="store_true",
                    help="after the FAST pass, repeat the run at cost-map level 2 into "
                         "costmap-deep/ so the fine breakdown exists without perturbing the "
                         "timings the FAST pass reports")
    ap.add_argument("--no-profiler", action="store_true",
                    help="control arm: same run with the cost map off, for the overhead gate")
    a = ap.parse_args()

    out = os.path.join(ROOT, a.out) if not os.path.isabs(a.out) else a.out
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(os.path.join(out, "costmap"), exist_ok=True)
    port = free_port(a.port)

    binp = os.path.join(ROOT, a.bin) if not os.path.isabs(a.bin) else a.bin
    rev = sh("git rev-parse --short HEAD").stdout.strip() or "unknown"
    dirty = bool(sh("git status --porcelain").stdout.strip())

    env = dict(os.environ)
    if not a.no_profiler:
        env["QWEN_COST_MAP"] = str(a.level)
        env["QWEN_COSTMAP_JSON"] = os.path.join(out, "costmap", "cm-%d.json")

    # provenance the engine itself reports; never re-derived here
    for name, flag in (("effective-config.txt", "--effective-config"),
                       ("dispatch-map.txt", "--dispatch-map")):
        r = subprocess.run([binp, "-d", a.model, flag] if flag == "--dispatch-map"
                           else [binp, flag], capture_output=True, text=True, cwd=ROOT, env=env)
        open(os.path.join(out, name), "w").write(r.stdout + r.stderr)

    log = open(os.path.join(out, "server.log"), "w")
    srv = subprocess.Popen(
        [binp, "-d", a.model, "--int8", "--serve", str(port),
         "--batch-size", str(a.batch_size), "--prefork", str(a.workers),
         "--prefork-threads", str(a.threads), "--max-queue", "8",
         "--max-request-seconds", "120"],
        stdout=log, stderr=subprocess.STDOUT, cwd=ROOT, env=env)
    ok = False
    for _ in range(180):
        time.sleep(1)
        try:
            if "Server listening" in open(os.path.join(out, "server.log")).read():
                ok = True; break
        except OSError:
            pass
        if srv.poll() is not None:
            break
    if not ok:
        srv.kill(); sys.exit("server did not start; see %s/server.log" % out)

    import threading
    res = []
    def one(i, p=None):
        body = json.dumps({"text": TEXT, "speaker": "ryan", "language": "English",
                           "seed": 42, "temperature": 0}).encode()
        t0 = time.time(); first = None; n = 0
        try:
            req = urllib.request.Request("http://127.0.0.1:%d/v1/tts/stream" % (p or port),
                                         data=body, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=180) as r:
                while True:
                    c = r.read(8192)
                    if not c:
                        break
                    if first is None:
                        first = time.time() - t0
                    n += len(c)
        except Exception as e:                                    # noqa: BLE001
            res.append({"i": i, "error": str(e)}); return
        wall = time.time() - t0
        audio = n / 2 / 24000.0
        res.append({"i": i, "ttfa_ms": (first or 0) * 1000, "wall_s": wall,
                    "bytes": n, "audio_s": audio,
                    "stream_rtf": (wall - (first or 0)) / audio if audio > 0 else None,
                    "total_rtf": wall / audio if audio > 0 else None})
    ths = [threading.Thread(target=one, args=(i,)) for i in range(a.conc)]
    for t in ths: t.start()
    for t in ths: t.join()

    srv.send_signal(signal.SIGINT)
    try:
        srv.wait(timeout=40)
    except subprocess.TimeoutExpired:
        srv.kill(); srv.wait(timeout=20)
    log.close()

    if a.deep_pass and not a.no_profiler:
        # A separate DEEP run: FAST cannot afford one timestamp per CP layer/step, so the fine
        # breakdown is measured in its own pass and the FAST pass keeps the honest timings.
        os.makedirs(os.path.join(out, "costmap-deep"), exist_ok=True)
        denv = dict(env); denv["QWEN_COST_MAP"] = "2"
        denv["QWEN_COSTMAP_JSON"] = os.path.join(out, "costmap-deep", "cm-%d.json")
        dlog = open(os.path.join(out, "server-deep.log"), "w")
        dport = free_port(port + 37)
        d = subprocess.Popen(
            [binp, "-d", a.model, "--int8", "--serve", str(dport),
             "--batch-size", str(a.batch_size), "--prefork", str(a.workers),
             "--prefork-threads", str(a.threads), "--max-queue", "8",
             "--max-request-seconds", "120"],
            stdout=dlog, stderr=subprocess.STDOUT, cwd=ROOT, env=denv)
        for _ in range(180):
            time.sleep(1)
            if "Server listening" in open(os.path.join(out, "server-deep.log")).read():
                break
            if d.poll() is not None:
                break
        dth = [threading.Thread(target=one, args=(100 + i, dport)) for i in range(a.conc)]
        for t in dth: t.start()
        for t in dth: t.join()
        d.send_signal(signal.SIGINT)
        try:
            d.wait(timeout=40)
        except subprocess.TimeoutExpired:
            d.kill(); d.wait(timeout=20)
        dlog.close()

    cm = sorted(f for f in os.listdir(os.path.join(out, "costmap")) if f.endswith(".json"))
    report = ""
    if cm:
        r = sh([sys.executable, os.path.join(ROOT, "tools", "costmap_report.py")] +
               [os.path.join(out, "costmap", f) for f in cm])
        report = r.stdout + r.stderr

    good = [x for x in res if "error" not in x and x["i"] < 100]
    def pct(v, p):
        v = sorted(v)
        return v[min(len(v) - 1, int(round((len(v) - 1) * p)))] if v else None
    summary = {
        "schema": 1,
        "build": {"commit": rev, "dirty": dirty, "binary_sha256_16": file_sha(binp),
                  "simd": (re.search(r"simd=(\S+)",
                                     open(os.path.join(out, "dispatch-map.txt")).read(),
                                     re.I) or [None, None])[1]},
        "host": host_topology(),
        "server": {"workers": a.workers, "threads_per_worker": a.threads,
                   "batch_size": a.batch_size, "concurrency": a.conc,
                   "masks": re.findall(r"prefork: worker \d+ pid \d+ cpus (\S+)",
                                       open(os.path.join(out, "server.log")).read())},
        "requests": res,
        "kpi": {"n": len(good),
                "ttfa_p50_ms": pct([x["ttfa_ms"] for x in good], .5),
                "ttfa_p95_ms": pct([x["ttfa_ms"] for x in good], .95),
                "stream_rtf_p50": pct([x["stream_rtf"] for x in good if x["stream_rtf"]], .5),
                "stream_rtf_p95": pct([x["stream_rtf"] for x in good if x["stream_rtf"]], .95),
                "audio_s_total": sum(x["audio_s"] for x in good)},
        "profiler": {"enabled": not a.no_profiler, "level": None if a.no_profiler else a.level,
                     "costmap_dumps": len(cm)},
    }
    json.dump(summary, open(os.path.join(out, "profile.json"), "w"), indent=1, default=str)

    srvlog = open(os.path.join(out, "server.log")).read()
    findings = []
    if "SMT is ON" in srvlog:
        findings.append("SMT_SIBLING_OVERLAP — the server warned that SMT is on")
    if "OVERLAP" in srvlog:
        findings.append("SMT_SIBLING_OVERLAP — worker masks overlap")
    if "OVERRIDDEN" in open(os.path.join(out, "effective-config.txt")).read():
        findings.append("BLAS_THREAD_ESCAPE (contained) — an env was overridden by engine ownership")
    if "IGNORED" in open(os.path.join(out, "effective-config.txt")).read():
        findings.append("INERT_FLAG — a requested flag is not honoured by this build")
    if "above the batched int8 ceiling" in srvlog:
        findings.append("BATCH_LIMIT_FALLBACK — --batch-size above the int8 matmat ceiling")
    for line in report.splitlines():
        if "POOL UNDERFILLED" in line:
            findings.append("POOL_UNDERFILL — " + line.strip())
        elif "UNACCOUNTED" in line:
            findings.append("UNACCOUNTED_POOL_WORK — " + line.strip())
    m = re.search(r"unattributed[^\n]*?(\d+\.\d)%", report)
    if m and float(m.group(1)) > 10:
        findings.append("HIGH_UNATTRIBUTED_TIME — %s%% of a parent is unattributed" % m.group(1))

    with open(os.path.join(out, "profile.md"), "w") as f:
        f.write("# FAST engine profile\n\n")
        f.write("    commit %s%s   binary %s   simd %s\n" %
                (rev, " (dirty)" if dirty else "", summary["build"]["binary_sha256_16"],
                 summary["build"]["simd"]))
        h = summary["host"]
        f.write("    %s\n    %s physical cores, SMT %s, online %s, NUMA %s\n" %
                (h.get("cpu_model", "?"), h.get("physical_cores", "?"),
                 h.get("smt_control", "?"), h.get("online_cpus", "?"), h.get("numa_nodes", "?")))
        f.write("    %d workers x %d threads, batch %d, masks %s, concurrency %d\n\n" %
                (a.workers, a.threads, a.batch_size,
                 " | ".join(summary["server"]["masks"]) or "?", a.conc))
        k = summary["kpi"]
        f.write("## Request KPI\n\n")
        f.write("    requests %d   audio %.2f s\n" % (k["n"], k["audio_s_total"]))
        f.write("    TTFA        p50 %s ms   p95 %s ms\n" %
                (round(k["ttfa_p50_ms"] or 0), round(k["ttfa_p95_ms"] or 0)))
        f.write("    STREAM_RTF  p50 %s      p95 %s\n\n" %
                (round(k["stream_rtf_p50"] or 0, 3), round(k["stream_rtf_p95"] or 0, 3)))
        f.write("## Findings\n\n")
        f.write("\n".join("- " + x for x in findings) + ("\n" if findings else "- none\n"))
        f.write("\n## Execution map\n\n```\n%s\n```\n" % (report or "cost map not collected"))
        f.write("\n## Effective configuration\n\n```\n%s```\n" %
                open(os.path.join(out, "effective-config.txt")).read())
    print("artifact: %s" % out)
    print("  TTFA p50 %s ms  STREAM_RTF p50 %s  findings %d" %
          (round(k["ttfa_p50_ms"] or 0), round(k["stream_rtf_p50"] or 0, 3), len(findings)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
