#!/usr/bin/env python3
"""serve_topology_probe.py — one big pool, or several small pinned ones?

THE QUESTION
The concurrency matrix showed the box already saturated by ONE request (15.1 of 16
cores, 112,857 context switches/s at c=1) and req/s flat from c=8. The thread curve
says where a single stream stops scaling. This asks the consequence: for the same 16
cores, is W workers of K threads each - pinned, independent, no shared pool - better
than one 16-thread pool, for concurrent streaming requests?

  1x16   one server, 16 threads          (today)
  2x8    two servers, 8 threads,  cores 0-7   / 8-15
  4x4    four servers, 4 threads, cores 0-3   / 4-7 / 8-11 / 12-15
  8x2    eight servers, 2 threads, cores 0-1  / ... / 14-15

Requests are dealt round-robin across the workers, so a topology's advantage has to
come from scheduling and locality, not from an easier load.

⚠️ MEMORY. Every worker is a full model copy. On the 1.7B with KleidiAI packed that
is ~13 GB each, so 4x4 needs ~53 GB and 8x2 does not fit in 64 GB. The script REFUSES
a topology it cannot hold rather than swapping and reporting a nonsense number - run
those cells on the 0.6B, where the shape of the curve still transfers.

Usage:
  python3 tests/serve_topology_probe.py --model DIR [--topo 1x16,2x8,4x4]
                                   [--conc 4,8,16] [--rounds 3] [--kleidi both]
"""
import argparse, json, os, re, signal, socket, subprocess, threading, time
import urllib.request

TEXT = ("The engine renders speech from text, one frame at a time. "
        "This sentence is long enough that every request stays in flight "
        "while the others arrive.")


def total_mem_gb():
    for line in open("/proc/meminfo"):
        if line.startswith("MemTotal"):
            return int(re.search(r"(\d+)", line).group(1)) / 1048576.0
    return 0.0


def wait_port(port, timeout=240):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with socket.create_connection(("127.0.0.1", port), 1.0):
                return True
        except OSError:
            time.sleep(0.5)
    return False


def one_request(port, results, lock):
    body = json.dumps({"text": TEXT, "speaker": "ryan", "language": "English",
                       "seed": 42}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/tts/stream", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time(); ttfb = None; n = 0
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            while True:
                # read1(), not read(): read(n) on a chunked response blocks until it
                # has n bytes, which turns TTFA into 'time to buffer 1.4 s of audio'.
                c = r.read1(65536)
                if not c: break
                if ttfb is None: ttfb = time.time() - t0
                n += len(c)
    except Exception as e:
        with lock: results.append({"err": str(e)})
        return
    total = time.time() - t0
    secs = n / 2.0 / 24000.0          # s16le @ 24 kHz
    with lock:
        results.append({"ttfa": ttfb, "total": total, "audio_s": secs,
                        "rtf": total / secs if secs > 0 else float("nan")})


def proc_counters(pid):
    vol = nonvol = migr = 0; rss = 0
    try:
        tids = os.listdir(f"/proc/{pid}/task")
    except OSError:
        return dict(vol=0, nonvol=0, migr=0, rss_kb=0, utime=0, stime=0)
    for tid in tids:
        try:
            st = open(f"/proc/{pid}/task/{tid}/status").read()
        except OSError:
            continue
        for line in st.splitlines():
            if line.startswith("voluntary_ctxt_switches"):
                vol += int(re.search(r"(\d+)", line).group(1))
            elif line.startswith("nonvoluntary_ctxt_switches"):
                nonvol += int(re.search(r"(\d+)", line).group(1))
        try:
            m = re.search(r"nr_migrations\s*:\s*(\d+)",
                          open(f"/proc/{pid}/task/{tid}/sched").read())
            if m: migr += int(m.group(1))
        except OSError:
            pass
    ut = st_ = 0
    try:
        f = open(f"/proc/{pid}/stat").read().rsplit(") ", 1)[1].split()
        ut, st_ = int(f[11]), int(f[12])
    except Exception:
        pass
    try:
        for line in open(f"/proc/{pid}/status"):
            if line.startswith("VmHWM"):
                rss = int(re.search(r"(\d+)", line).group(1))
    except OSError:
        pass
    return dict(vol=vol, nonvol=nonvol, migr=migr, rss_kb=rss, utime=ut, stime=st_)


def run_topology(a, W, K, kai, conc, port0, out):
    tag = f"{'on' if kai else 'off'}_{W}x{K}_c{conc}"
    env = dict(os.environ)
    env["QWEN_SERVE_PROFILE"] = "1"
    env["QWEN_BATCH_STATS"] = "1"
    if not kai:
        env["QWEN_NO_KAI_I8"] = "1"; env["QWEN_NO_KAI_BF16"] = "1"

    procs, logs, ports = [], [], []
    ncpu = os.cpu_count() or 16
    per = ncpu // W
    for w in range(W):
        port = port0 + w
        cpus = f"{w * per}-{w * per + per - 1}"
        log = os.path.join(out, f"{tag}_w{w}.log")
        cmd = ["taskset", "-c", cpus, a.bin, "-d", a.model, "--serve", str(port),
               "--batch-size", str(max(1, conc // W if W > 1 else conc)), "-j", str(K)]
        if a.precision == "int8":
            cmd.insert(cmd.index(a.bin) + 3, "--int8")
        f = open(log, "wb")
        procs.append(subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env))
        logs.append((log, f)); ports.append(port)
    try:
        for p in ports:
            if not wait_port(p):
                raise RuntimeError(f"worker on {p} did not come up")
        res, lock = [], threading.Lock()
        for i, p in enumerate(ports):      # warm every worker, not just the first
            one_request(p, res, lock)
        res.clear()

        c0 = [proc_counters(p.pid) for p in procs]
        t0 = time.time()
        for _ in range(a.rounds):
            ts = [threading.Thread(target=one_request,
                                   args=(ports[i % len(ports)], res, lock))
                  for i in range(conc)]           # round-robin across workers
            for t in ts: t.start()
            for t in ts: t.join()
        wall = time.time() - t0
        c1 = [proc_counters(p.pid) for p in procs]
    finally:
        for p in procs:
            p.send_signal(signal.SIGTERM)
        for p in procs:
            try: p.wait(timeout=120)
            except subprocess.TimeoutExpired: p.kill(); p.wait()
        for _, f in logs: f.close()

    ok = [r for r in res if "err" not in r]
    hz = os.sysconf("SC_CLK_TCK")
    cpu_s = sum((b["utime"] - a_["utime"] + b["stime"] - a_["stime"]) / hz
                for a_, b in zip(c0, c1))
    csw = sum((b["vol"] - a_["vol"]) + (b["nonvol"] - a_["nonvol"]) for a_, b in zip(c0, c1))
    migr = sum(b["migr"] - a_["migr"] for a_, b in zip(c0, c1))
    rss = sum(b["rss_kb"] for b in c1) / 1024.0

    def pct(v, q):
        if not v: return float("nan")
        v = sorted(v); return v[min(len(v) - 1, int(round(q * (len(v) - 1))))]
    return {
        "tag": tag, "W": W, "K": K, "kai": kai, "conc": conc,
        "n_ok": len(ok), "n_err": len(res) - len(ok),
        "req_s": len(ok) / wall if wall else 0,
        "ttfa_p50": pct([r["ttfa"] for r in ok if r["ttfa"]], .50),
        "ttfa_p95": pct([r["ttfa"] for r in ok if r["ttfa"]], .95),
        "rtf_p50": pct([r["rtf"] for r in ok], .50),
        "rtf_p95": pct([r["rtf"] for r in ok], .95),
        "cores": cpu_s / wall if wall else 0,
        "csw_s": csw / wall if wall else 0,
        "migr": migr, "rss_mb": rss,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="/tmp/kai_topo")
    ap.add_argument("--topo", default="1x16,2x8,4x4")
    ap.add_argument("--conc", default="4,8,16")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--precision", default="int8", choices=["int8", "bf16"])
    ap.add_argument("--kleidi", default="both", choices=["both", "on", "off"])
    ap.add_argument("--bin", default="./qwen_tts")
    ap.add_argument("--port", type=int, default=8940)
    ap.add_argument("--rss-per-worker-gb", type=float, default=0.0,
                    help="measured RSS of ONE worker; used to refuse a topology that "
                         "would not fit. 0 = do not check.")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    topos = [tuple(int(x) for x in t.split("x")) for t in a.topo.split(",")]
    concs = [int(x) for x in a.conc.split(",")]
    kais = {"both": [False, True], "on": [True], "off": [False]}[a.kleidi]
    mem = total_mem_gb()
    print(f"box memory {mem:.0f} GB, {os.cpu_count()} cpus")

    rows, port = [], a.port
    for W, K in topos:
        if a.rss_per_worker_gb > 0 and W * a.rss_per_worker_gb > mem * 0.85:
            print(f"  SKIP {W}x{K}: {W} workers x {a.rss_per_worker_gb:.1f} GB "
                  f"exceeds 85% of {mem:.0f} GB. Run this cell on the 0.6B.")
            continue
        for c in concs:
            for kai in kais:
                print(f"--- {W}x{K} c={c} kleidi={kai} ...", flush=True)
                try:
                    r = run_topology(a, W, K, kai, c, port, a.out)
                except Exception as e:
                    r = {"tag": f"{W}x{K}_c{c}", "W": W, "K": K, "kai": kai,
                         "conc": c, "error": str(e)}
                rows.append(r); port += max(W, 1) + 1
                json.dump(rows, open(os.path.join(a.out, "topo.json"), "w"), indent=1)

    hdr = (f"{'cell':<18}{'req/s':>7}{'TTFA50':>8}{'TTFA95':>8}{'RTF50':>7}{'RTF95':>7}"
           f"{'cores':>7}{'csw/s':>9}{'migr':>8}{'RSS MB':>8}{'err':>5}")
    print("\n" + hdr); print("-" * len(hdr))
    for r in rows:
        if "error" in r:
            print(f"{r['tag']:<18} ERROR {r['error']}"); continue
        print(f"{r['tag']:<18}{r['req_s']:>7.2f}{r['ttfa_p50']:>8.2f}{r['ttfa_p95']:>8.2f}"
              f"{r['rtf_p50']:>7.2f}{r['rtf_p95']:>7.2f}{r['cores']:>7.2f}"
              f"{r['csw_s']:>9.0f}{r['migr']:>8.0f}{r['rss_mb']:>8.0f}{r['n_err']:>5d}")

    print("\nvs 1x16 at the same concurrency and the same kleidi setting")
    for kai in kais:
        for c in concs:
            base = next((x for x in rows if x.get("W") == 1 and x.get("conc") == c
                         and x.get("kai") == kai and "error" not in x), None)
            if not base: continue
            for r in rows:
                if r.get("conc") != c or r.get("kai") != kai or "error" in r: continue
                if r["W"] == 1: continue
                print(f"  kleidi={str(kai):<5} c={c:<3} {r['W']}x{r['K']:<3} "
                      f"req/s {r['req_s']/base['req_s']:.2f}x  "
                      f"TTFA {base['ttfa_p50']/r['ttfa_p50']:.2f}x  "
                      f"RTF {base['rtf_p50']/r['rtf_p50']:.2f}x  "
                      f"csw {r['csw_s']/base['csw_s']:.2f}x  "
                      f"RSS {r['rss_mb']/base['rss_mb']:.2f}x")


if __name__ == "__main__":
    main()
