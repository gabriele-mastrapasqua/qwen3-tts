#!/usr/bin/env python3
"""serve_topology_bench.py — which topology keeps TTFA low at LOW concurrency?"""
import argparse, csv as csvmod, json, os, re, signal, socket, statistics, subprocess, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import serve_procstats as PS

TOPOS = {
    "A": (1, 16, 16, {}),
    "B": (1, 16, 16, {"QWEN_THREADS_TALKER": "8"}),
    "C": (2, 8, 8, {}),
    "D": (4, 4, 4, {}),
    "E": (8, 2, 2, {}),
}

def wait_port(port, timeout=300):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with socket.create_connection(("127.0.0.1", port), 1.0):
                return True
        except OSError:
            time.sleep(0.5)
    return False

def tree_pids(root):
    out, seen = [root], {root}
    stack = [root]
    while stack:
        p = stack.pop()
        for d in os.listdir("/proc"):
            if not d.isdigit() or int(d) in seen:
                continue
            try:
                st = open(f"/proc/{d}/stat").read().rsplit(") ", 1)[1].split()
                if int(st[1]) == p:
                    seen.add(int(d)); out.append(int(d)); stack.append(int(d))
            except Exception:
                pass
    return out

def counters(pids):
    vol = nonvol = migr = ut = st_ = rss = pss = 0
    for pid in pids:
        try:
            tids = os.listdir(f"/proc/{pid}/task")
        except OSError:
            continue
        for t in tids:
            try:
                s = open(f"/proc/{pid}/task/{t}/status").read()
            except OSError:
                continue
            for l in s.splitlines():
                if l.startswith("voluntary_ctxt_switches"):
                    vol += int(re.search(r"(\d+)", l).group(1))
                elif l.startswith("nonvoluntary_ctxt_switches"):
                    nonvol += int(re.search(r"(\d+)", l).group(1))
            try:
                m = re.search(r"nr_migrations\s*:\s*(\d+)",
                              open(f"/proc/{pid}/task/{t}/sched").read())
                if m: migr += int(m.group(1))
            except OSError:
                pass
        try:
            f = open(f"/proc/{pid}/stat").read().rsplit(") ", 1)[1].split()
            ut += int(f[11]); st_ += int(f[12])
        except Exception:
            pass
        try:
            for l in open(f"/proc/{pid}/smaps_rollup"):
                m = re.match(r"(\w+):\s+(\d+) kB", l)
                if m and m.group(1) == "Rss": rss += int(m.group(2))
                if m and m.group(1) == "Pss": pss += int(m.group(2))
        except OSError:
            pass
    return dict(vol=vol, nonvol=nonvol, migr=migr, ut=ut, st=st_, rss=rss, pss=pss)

def start_server(a, topo, port, out):
    W, K, cap, env_extra = TOPOS[topo]
    env = dict(os.environ); env.update(env_extra)
    cmd = [a.bin, "-d", a.model, "--serve", str(port), "--batch-size", str(cap)]
    if a.precision == "int8":
        cmd.insert(3, "--int8")
    if W > 1:
        cmd += ["--prefork", str(W), "--prefork-threads", str(K)]
    else:
        cmd += ["-j", str(K)]
    log = os.path.join(out, f"{topo}_server.log")
    f = open(log, "wb")
    p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
    return p, f, log, (W, K, cap)

def run_level(a, port, conc, out, tag):
    reqs = max(a.min_requests, 3 * conc)
    if reqs > a.max_requests: reqs = a.max_requests
    js = os.path.join(out, f"{tag}.json")
    cs = os.path.join(out, f"{tag}.csv")
    cmd = ["python3", "tests/load_test.py",
           "--url", f"http://127.0.0.1:{port}",
           "--concurrency", str(conc), "--requests", str(reqs),
           "--arrival", a.arrival, "--arrival-seed", "7",
           "--speaker", "ryan", "--language", "English", "--seed", "42",
           "--no-save-audio", "--json", js, "--csv", cs,
           "--ttfa-budget-ms", str(a.budget)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    if not os.path.exists(js):
        return None, r.stdout[-800:] + r.stderr[-800:]
    data = json.load(open(js))
    row = data[0] if isinstance(data, list) else data
    if isinstance(row, dict) and "levels" in row:
        row = row["levels"][0]
    tot = []
    try:
        with open(cs) as fh:
            for rec in csvmod.DictReader(fh):
                for k in ("total_ms", "latency_ms", "elapsed_ms", "duration_ms"):
                    if k in rec and rec[k]:
                        tot.append(float(rec[k])); break
    except Exception:
        pass
    if tot:
        tot.sort()
        row["total_p50"] = tot[len(tot) // 2]
        row["total_p95"] = tot[min(len(tot) - 1, int(round(0.95 * (len(tot) - 1))))]
    return row, None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="/tmp/kai_curve")
    ap.add_argument("--topo", default="A,B,C,D,E")
    ap.add_argument("--conc", default="1,2,3,4,5,6,8,16")
    ap.add_argument("--arrival", default="poisson",
                    choices=["poisson", "all-at-once", "uniform"])
    ap.add_argument("--min-requests", type=int, default=12)
    ap.add_argument("--max-requests", type=int, default=32)
    ap.add_argument("--budget", type=int, default=500)
    ap.add_argument("--precision", default="int8")
    ap.add_argument("--bin", default="./qwen_tts")
    ap.add_argument("--port", type=int, default=9100)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    concs = [int(x) for x in a.conc.split(",")]
    rows = []
    port = a.port
    hz = os.sysconf("SC_CLK_TCK")

    for topo in a.topo.split(","):
        W, K, cap, _ = TOPOS[topo]
        print(f"\n=== topology {topo}: {W}x{K}, cap {cap}/worker, port {port} ===", flush=True)
        p, f, log, _ = start_server(a, topo, port, a.out)
        try:
            if not wait_port(port):
                print(f"  {topo}: server did not come up"); continue
            time.sleep(5)
            pids = tree_pids(p.pid)
            wmap = {i: pid for i, pid, _c, _t in PS.worker_pids_from_log(log)} or {0: p.pid}
            print(f"  processes: {len(pids)}", flush=True)
            for c in concs:
                tag = f"{topo}_c{c}"
                if W > 1:
                    try: p.send_signal(signal.SIGUSR1)
                    except Exception: pass
                    time.sleep(0.8)
                mark = os.path.getsize(log)
                c0 = counters(pids); t0 = time.time()
                pw0 = {i: PS.proc_sample(pid) for i, pid in wmap.items()}
                row, err = run_level(a, port, c, a.out, tag)
                wall = time.time() - t0; c1 = counters(pids)
                pw1 = {i: PS.proc_sample(pid) for i, pid in wmap.items()}
                stats_line = ""
                if W > 1:
                    try: p.send_signal(signal.SIGUSR1)
                    except Exception: pass
                    time.sleep(0.8)
                    try:
                        with open(log, "rb") as lf:
                            lf.seek(mark); tail = lf.read().decode(errors="replace")
                        ls = [l for l in tail.splitlines() if "[prefork-stats]" in l]
                        if ls: stats_line = ls[-1]
                    except Exception:
                        pass
                wrows = PS.per_worker_rows(pw0, pw1, PS.parse_prefork_stats(stats_line), wall)
                if row is None:
                    print(f"  {tag}: FAILED {err}"); continue
                row.update({
                    "topo": topo, "W": W, "K": K, "cap": cap, "conc": c,
                    "cores": ((c1["ut"] - c0["ut"]) + (c1["st"] - c0["st"])) / hz / wall,
                    "csw_s": ((c1["vol"] - c0["vol"]) + (c1["nonvol"] - c0["nonvol"])) / wall,
                    "migr": c1["migr"] - c0["migr"],
                    "rss_mb": c1["rss"] / 1024.0, "pss_mb": c1["pss"] / 1024.0,
                    "workers": wrows,
                })
                rows.append(row)
                print(f"  {tag}: TTFA p50 {row['ttfa_p50']:.0f} p95 {row['ttfa_p95']:.0f} ms · "
                      f"RTF {row['rtf_p50']:.2f} · Q {row['throughput_Q']:.2f} · "
                      f"B {row.get('mean_inflight', 0):.2f} · cores {row['cores']:.1f} · "
                      f"err {row['errors']}", flush=True)
                if wrows:
                    print(PS.format_rows(wrows), flush=True)
                json.dump(rows, open(os.path.join(a.out, "curve.json"), "w"), indent=1)
        finally:
            p.send_signal(signal.SIGTERM)
            try: p.wait(timeout=180)
            except subprocess.TimeoutExpired: p.kill(); p.wait()
            f.close()
            port += W + 2

    hdr = (f"{'topo':<6}{'c':>3}{'req/s':>7}{'Q':>6}{'TTFA50':>8}{'TTFA95':>8}"
           f"{'RTF50':>7}{'RTF95':>7}{'tot50':>8}{'tot95':>8}{'B':>6}"
           f"{'cores':>7}{'csw/s':>8}{'PSS GB':>8}{'err':>5}")
    print("\n" + hdr); print("-" * len(hdr))
    for r in rows:
        print(f"{r['topo']:<6}{r['conc']:>3}{r['ok']/r['wall_s']:>7.2f}{r['throughput_Q']:>6.2f}"
              f"{r['ttfa_p50']:>8.0f}{r['ttfa_p95']:>8.0f}{r['rtf_p50']:>7.2f}{r['rtf_p95']:>7.2f}"
              f"{r.get('total_p50', float('nan')):>8.0f}{r.get('total_p95', float('nan')):>8.0f}"
              f"{r.get('mean_inflight', 0):>6.2f}{r['cores']:>7.1f}{r['csw_s']:>8.0f}"
              f"{r['pss_mb']/1024:>8.1f}{r['errors']:>5}")
    print(f"\nTTFA budget {a.budget} ms · arrival {a.arrival} · the cell that matters is the")
    print("HIGHEST c whose TTFA p95 is still inside the budget, per topology.")
    for topo in a.topo.split(","):
        ins = [r for r in rows if r["topo"] == topo and r["ttfa_p95"] <= a.budget]
        best = max(ins, key=lambda r: r["conc"]) if ins else None
        print(f"  {topo}: " + (f"c={best['conc']} (TTFA p95 {best['ttfa_p95']:.0f} ms, "
                               f"Q {best['throughput_Q']:.2f})" if best else "never inside budget"))

if __name__ == "__main__":
    main()
