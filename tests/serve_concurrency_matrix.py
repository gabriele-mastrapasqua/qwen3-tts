#!/usr/bin/env python3
"""serve_concurrency_matrix.py — the concurrency matrix that decides whether the KleidiAI
backend is a throughput lever or a latency one.

THE QUESTION
Single-stream is B=1 and measured 1.02x end-to-end. The census says the same kernels
give 1.9-4.6x at B>=2. The server is the only place B>=2 actually exists, so this
sweeps concurrency and correlates, per cell:

    concurrency  ->  effective B  ->  which kernel fired  ->  speedup

WHAT IT REPORTS PER CELL
  client side : req/s, TTFA p50/p95 (time to FIRST BYTE on /v1/tts/stream, which is
                the real first-audio instant), RTF p50/p95
  server side : Talker ms/frame, CP ms/frame, mean active slots (= the effective B,
                straight from [serve-profile]), the shape-census B histogram over the
                projections, the GEMV-vs-GEMM split and GMAC per kernel from the
                batch audit
  os side     : context switches/s, CPU utilisation, core migrations, peak RSS

TOPOLOGY IS FIXED HERE, ON PURPOSE
One server, one pool, --batch-size and -j held constant across every cell. Changing
worker count or affinity in the same matrix would confound the concurrency curve with
a topology change; the topology series is a SECOND matrix, run after this one.

Usage:
  python3 tests/serve_concurrency_matrix.py --model DIR [--out DIR] [--rounds 3]
                                        [--conc 1,2,4,8,16] [--threads 16]
                                        [--batch-size 16] [--precision int8]
"""
import argparse, json, os, re, signal, socket, statistics, subprocess, sys, threading, time
import urllib.request

TEXT = ("The engine renders speech from text, one frame at a time. "
        "This sentence is long enough that every request stays in flight "
        "while the others arrive.")


def wait_port(port, timeout=180):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with socket.create_connection(("127.0.0.1", port), 1.0):
                return True
        except OSError:
            time.sleep(0.5)
    return False


def proc_snapshot(pid):
    """The three /proc files that answer 'was it computing or scheduling'."""
    out = {}
    try:
        st = open(f"/proc/{pid}/stat").read().rsplit(") ", 1)[1].split()
        out["utime"], out["stime"] = int(st[11]), int(st[12])
    except Exception:
        out["utime"] = out["stime"] = 0
    for line in _read(f"/proc/{pid}/status").splitlines():
        if line.startswith("VmHWM"):
            out["rss_kb"] = int(re.search(r"(\d+)", line).group(1))
    # Context switches and migrations are PER THREAD in /proc, and this server runs a
    # 16-wide pool: reading only the main thread reported 0 switches while the pool was
    # thrashing. Sum over /proc/PID/task/*, which is the whole point of the metric.
    vol = nonvol = migr = 0
    try:
        tids = os.listdir(f"/proc/{pid}/task")
    except OSError:
        tids = []
    for tid in tids:
        st = _read(f"/proc/{pid}/task/{tid}/status")
        for line in st.splitlines():
            if line.startswith("voluntary_ctxt_switches"):
                vol += int(re.search(r"(\d+)", line).group(1))
            elif line.startswith("nonvoluntary_ctxt_switches"):
                nonvol += int(re.search(r"(\d+)", line).group(1))
        m = re.search(r"nr_migrations\s*:\s*(\d+)", _read(f"/proc/{pid}/task/{tid}/sched"))
        if m: migr += int(m.group(1))
    out["vol"], out["nonvol"], out["migr"] = vol, nonvol, migr
    out["nthreads"] = len(tids)
    return out


def _read(p):
    try:
        return open(p).read()
    except Exception:
        return ""


def one_request(port, results, lock):
    """POST to /v1/tts/stream and time the FIRST BYTE. On the streaming endpoint that
    is the first-audio instant, which is what TTFA means; the total is what RTF needs."""
    body = json.dumps({"text": TEXT, "speaker": "ryan", "language": "English",
                       "seed": 42}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/tts/stream", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    ttfb = None
    n = 0
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            while True:
                # read1(), not read(): read(n) on a chunked response blocks until it
                # has n bytes, which turns TTFA into 'time to buffer 1.4 s of audio'.
                chunk = r.read1(65536)
                if not chunk:
                    break
                if ttfb is None:
                    ttfb = time.time() - t0
                n += len(chunk)
    except Exception as e:
        with lock:
            results.append({"err": str(e)})
        return
    total = time.time() - t0
    # /v1/tts/stream emits s16le mono at 24 kHz (qwen_tts_server.c:stream_http_callback);
    # urllib de-chunks for us, so n is raw PCM bytes and 2 bytes = 1 sample.
    secs = n / 2.0 / 24000.0
    with lock:
        results.append({"ttfa": ttfb, "total": total, "bytes": n, "audio_s": secs,
                        "rtf": total / secs if secs > 0 else float("nan")})


def parse_server_log(txt):
    out = {}
    m = re.search(r"\[serve-profile\] (\d+) frames, (\d+) slot-frames \(mean ([\d.]+) active slots\)", txt)
    if m:
        out["frames"] = int(m.group(1))
        out["slot_frames"] = int(m.group(2))
        out["mean_slots"] = float(m.group(3))
    for label, key in (("talker step (batched)", "talker_ms"),
                       ("code predictor (batched)", "cp_ms"),
                       ("speech decode + embed", "dec_ms"),
                       ("admission + prefill", "admit_ms")):
        m = re.search(re.escape(label) + r"\s+([\d.]+)\s", txt)
        if m:
            out[key] = float(m.group(1))
    if out.get("frames"):
        for k in ("talker_ms", "cp_ms", "dec_ms"):
            if k in out:
                out[k.replace("_ms", "_msf")] = out[k] / out["frames"]
    # batch audit: kernel -> GMAC, and the class split line
    kern = {}
    for m in re.finditer(r"^\s{2}(\S.*?)\s{2,}([\d.]+)\s+(\d+)\s+([\d.]+)%", txt, re.M):
        name = m.group(1).strip()
        if name in ("kernel",):
            continue
        kern[name] = {"gmac": float(m.group(2)), "calls": int(m.group(3)),
                      "share": float(m.group(4))}
    out["kernels"] = kern
    m = re.search(r"matrix-matrix\s+([\d.]+)%.*?GEMV\s+([\d.]+)%", txt, re.S)
    if m:
        out["gemm_pct"], out["gemv_pct"] = float(m.group(1)), float(m.group(2))
    # shape census: B histogram weighted by calls, over the projections only
    hist = {}
    for m in re.finditer(r"^census,\w+,\w+,(\d+),(\d+),(\d+),(\d+),", txt, re.M):
        B, calls = int(m.group(3)), int(m.group(4))
        hist[B] = hist.get(B, 0) + calls
    out["b_hist"] = hist
    if hist:
        tot = sum(hist.values())
        out["b_mean_calls"] = sum(b * c for b, c in hist.items()) / tot
    return out


def cell(args, kai_on, conc, port):
    env = dict(os.environ)
    env["QWEN_BATCH_STATS"] = "1"
    env["QWEN_SHAPE_CENSUS"] = "1"
    env["QWEN_SERVE_PROFILE"] = "1"
    if not kai_on:
        env["QWEN_NO_KAI_I8"] = "1"
        env["QWEN_NO_KAI_BF16"] = "1"
    tag = f"{'on' if kai_on else 'off'}_c{conc}"
    log_path = os.path.join(args.out, f"{tag}.log")
    cmd = [args.bin, "-d", args.model, "--serve", str(port),
           "--batch-size", str(args.batch_size), "-j", str(args.threads)]
    if args.precision == "int8":
        cmd.insert(3, "--int8")
    logf = open(log_path, "wb")
    p = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env)
    try:
        if not wait_port(port):
            raise RuntimeError("server did not come up")
        # Warm: the first request pays page faults and a cold pool on BOTH configs,
        # and leaving it in would flatter whichever ran first.
        res, lock = [], threading.Lock()
        one_request(port, res, lock)
        res.clear()

        s0 = proc_snapshot(p.pid)
        t0 = time.time()
        for _ in range(args.rounds):
            ts = [threading.Thread(target=one_request, args=(port, res, lock))
                  for _ in range(conc)]
            for t in ts: t.start()
            for t in ts: t.join()
        wall = time.time() - t0
        s1 = proc_snapshot(p.pid)
    finally:
        p.send_signal(signal.SIGTERM)     # clean exit -> atexit -> audit + census flush
        try:
            p.wait(timeout=120)
        except subprocess.TimeoutExpired:
            p.kill(); p.wait()
        logf.close()

    ok = [r for r in res if "err" not in r]
    errs = [r for r in res if "err" in r]
    hz = os.sysconf("SC_CLK_TCK")
    cpu_s = ((s1["utime"] - s0["utime"]) + (s1["stime"] - s0["stime"])) / hz
    csw = (s1.get("vol", 0) - s0.get("vol", 0)) + (s1.get("nonvol", 0) - s0.get("nonvol", 0))
    srv = parse_server_log(open(log_path, errors="replace").read())

    def pct(v, q):
        if not v: return float("nan")
        v = sorted(v)
        i = min(len(v) - 1, int(round(q * (len(v) - 1))))
        return v[i]

    return {
        "tag": tag, "kai": kai_on, "conc": conc,
        "n_ok": len(ok), "n_err": len(errs), "err_sample": errs[0]["err"] if errs else "",
        "wall_s": wall, "req_s": len(ok) / wall if wall > 0 else 0,
        "ttfa_p50": pct([r["ttfa"] for r in ok if r["ttfa"]], 0.50),
        "ttfa_p95": pct([r["ttfa"] for r in ok if r["ttfa"]], 0.95),
        "rtf_p50": pct([r["rtf"] for r in ok], 0.50),
        "rtf_p95": pct([r["rtf"] for r in ok], 0.95),
        "cpu_util": cpu_s / wall if wall > 0 else 0,
        "csw_s": csw / wall if wall > 0 else 0,
        "migr": s1.get("migr", 0) - s0.get("migr", 0),
        "nthreads": s1.get("nthreads", 0),
        "rss_mb": s1.get("rss_kb", 0) / 1024.0,
        **{k: srv.get(k) for k in ("talker_msf", "cp_msf", "dec_msf", "mean_slots",
                                   "gemm_pct", "gemv_pct", "b_mean_calls", "frames")},
        "b_hist": srv.get("b_hist", {}),
        "kernels": srv.get("kernels", {}),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="/tmp/kai_srv")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--conc", default="1,2,4,8,16")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--precision", default="int8", choices=["int8", "bf16"])
    ap.add_argument("--bin", default="./qwen_tts")
    ap.add_argument("--port", type=int, default=8917)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    concs = [int(x) for x in a.conc.split(",")]

    rows = []
    port = a.port
    for c in concs:
        for kai in (False, True):     # alternate so drift hits both equally
            print(f"--- running {'kleidi' if kai else 'baseline'} c={c} ...", flush=True)
            try:
                r = cell(a, kai, c, port)
            except Exception as e:
                r = {"tag": f"{'on' if kai else 'off'}_c{c}", "kai": kai, "conc": c,
                     "error": str(e)}
            rows.append(r)
            port += 1                 # never reuse a port a dying server still holds
            json.dump(rows, open(os.path.join(a.out, "matrix.json"), "w"), indent=1)

    hdr = (f"{'cell':<10}{'req/s':>7}{'TTFA50':>8}{'TTFA95':>8}{'RTF50':>7}{'RTF95':>7}"
           f"{'tk ms/f':>9}{'cp ms/f':>9}{'B':>6}{'GEMM%':>7}{'csw/s':>9}{'CPU':>6}"
           f"{'migr':>7}{'RSS MB':>8}")
    print("\n" + hdr); print("-" * len(hdr))
    for r in rows:
        if "error" in r:
            print(f"{r['tag']:<10} ERROR {r['error']}"); continue
        f = lambda k, d=0.0: (r.get(k) if r.get(k) is not None else d)
        print(f"{r['tag']:<10}{f('req_s'):>7.2f}{f('ttfa_p50'):>8.2f}{f('ttfa_p95'):>8.2f}"
              f"{f('rtf_p50'):>7.2f}{f('rtf_p95'):>7.2f}{f('talker_msf'):>9.2f}"
              f"{f('cp_msf'):>9.2f}{f('mean_slots'):>6.2f}{f('gemm_pct'):>7.1f}"
              f"{f('csw_s'):>9.0f}{f('cpu_util'):>6.1f}{f('migr'):>7.0f}{f('rss_mb'):>8.0f}")

    print("\nconcurrency -> effective B -> kernel -> speedup")
    for c in concs:
        off = next((x for x in rows if x.get("conc") == c and not x.get("kai") and "error" not in x), None)
        on = next((x for x in rows if x.get("conc") == c and x.get("kai") and "error" not in x), None)
        if not off or not on: continue
        def sp(k, inv=False):
            a_, b_ = off.get(k), on.get(k)
            if not a_ or not b_: return float("nan")
            return (b_ / a_) if inv else (a_ / b_)
        khist = ",".join(f"B{b}:{n}" for b, n in sorted(on.get("b_hist", {}).items()))
        print(f"  c={c:<3} B={on.get('mean_slots')}  req/s {sp('req_s', inv=True):.2f}x  "
              f"RTF {sp('rtf_p50'):.2f}x  TTFA {sp('ttfa_p50'):.2f}x  "
              f"tk {sp('talker_msf'):.2f}x  cp {sp('cp_msf'):.2f}x   [{khist}]")


if __name__ == "__main__":
    main()
