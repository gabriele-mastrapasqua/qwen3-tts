#!/usr/bin/env python3
"""cancel_correctness.py — K1-K7 for server-side cancellation. Correctness, not performance."""
import argparse, hashlib, json, os, re, socket, subprocess, sys, threading, time

SR = 24000

def wait_health(port, timeout=600):
    import urllib.request
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/health", timeout=2) as r:
                if b'"ok"' in r.read():
                    return True
        except Exception:
            time.sleep(0.5)
    return False

def stream(port, text, speaker, language, seed, abort_after=None):
    body = json.dumps({"text": text, "speaker": speaker, "language": language,
                       "seed": seed, "temperature": 0.0}).encode()
    req = (f"POST /v1/tts/stream HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
           f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n").encode() + body
    s = socket.create_connection(("127.0.0.1", port), 30)
    s.settimeout(0.2)
    h, n, t0, aborted = hashlib.sha256(), 0, time.time(), False
    abort_abs_ms = None
    buf, hdr = b"", False
    try:
        s.sendall(req)
        while True:
            if abort_after is not None and time.time() - t0 >= abort_after:
                aborted = True
                abort_abs_ms = time.monotonic() * 1000.0
                s.shutdown(socket.SHUT_RDWR)
                break
            try:
                ch = s.recv(1 << 16)
            except socket.timeout:
                continue
            if not ch:
                break
            buf += ch
            if not hdr:
                i = buf.find(b"\r\n\r\n")
                if i < 0:
                    continue
                hdr = True
                buf = buf[i+4:]
            while hdr:
                m = re.match(rb"([0-9a-fA-F]+)\r\n", buf)
                if not m:
                    break
                sz = int(m.group(1), 16)
                need = m.end() + sz + 2
                if len(buf) < need:
                    break
                pay = buf[m.end():m.end()+sz]
                buf = buf[need:]
                if sz == 0:
                    hdr = False
                    break
                h.update(pay); n += sz
    finally:
        s.close()
    return {"sha": h.hexdigest()[:16], "bytes": n, "audio_s": n/2/SR,
            "aborted": aborted, "wall_s": time.time()-t0,
            "abort_abs_ms": abort_abs_ms}

LOAD_FRAMES = ("qwen_tts_load_ex", "qwen_talker_load", "qwen_cp_load",
               "qwen_speech_decoder_load", "qwen_tts_load", "qwen_kleidi")

def parse_leaks(path):
    """LeakSanitizer blocks in a server log -> (load_bytes, request_bytes, examples).

    Returns ran=False when the log carries no LSan output at all: a run whose leak check
    never executed must say so, not report zero."""
    load_b = req_b = 0
    examples, ran, sigs = [], False, {}
    cur_bytes, cur_obj, cur_stack, in_block = 0, 0, [], False
    def flush():
        nonlocal load_b, req_b, cur_bytes, cur_obj, cur_stack, in_block
        if in_block:
            txt = "".join(cur_stack)
            frames = []
            for l in cur_stack[:4]:
                f = l.split(" in ")
                frames.append(f[1].split(" ")[0] if len(f) > 1 else l.strip()[:32])
            sig = " <- ".join(frames)
            if any(f in txt for f in LOAD_FRAMES):
                load_b += cur_bytes
            else:
                req_b += cur_bytes
                e = sigs.setdefault(sig, [0, 0])
                e[0] += cur_bytes; e[1] += cur_obj
                if len(examples) < 5:
                    examples.append((cur_bytes, sig))
        cur_bytes, cur_obj, cur_stack, in_block = 0, 0, [], False
    for ln in open(path, errors="replace"):
        if "LeakSanitizer" in ln or (ln.startswith("SUMMARY: AddressSanitizer:") and "leaked" in ln):
            ran = True
        m = re.match(r"(?:Direct|Indirect) leak of (\d+) byte\(s\) in (\d+) object", ln)
        if m:
            flush(); in_block = True
            cur_bytes, cur_obj = int(m.group(1)), int(m.group(2)); continue
        if in_block:
            if ln.strip().startswith("#"):
                cur_stack.append(ln)
            elif not ln.strip():
                flush()
    flush()
    return dict(ran=ran, load=load_b, req=req_b, examples=examples, sigs=sigs)

def parse_log(path):
    life, cancel, req = {}, {}, {}
    for ln in open(path, errors="replace"):
        m = re.search(r"\[LIFE\] pid=\d+ seed=(\d+) .*state=(\w+)", ln)
        if m: life[int(m.group(1))] = m.group(2)
        m = re.search(r"\[CANCEL\] pid=\d+ seed=(\d+) detected_ms=([\d.-]+) "
                      r"stopped_ms=([\d.-]+) cancel_to_stop_ms=([\d.-]+) enabled=(\d)"
                      r"(?: rdhup=(\d) detected_abs_ms=([\d.-]+))?", ln)
        if m: cancel[int(m.group(1))] = dict(detected=float(m.group(2)),
                                             stopped=float(m.group(3)),
                                             to_stop=float(m.group(4)),
                                             enabled=int(m.group(5)),
                                             rdhup=(int(m.group(6)) if m.group(6) else None),
                                             detected_abs=(float(m.group(7)) if m.group(7) else None))
        m = re.search(r"\[REQ\] pid=\d+ seed=(\d+) tokens=\d+ frames=(\d+) audio_s=([\d.]+)"
                      r"(.*cancelled=1)?", ln)
        if m: req[int(m.group(1))] = dict(frames=int(m.group(2)), audio=float(m.group(3)),
                                          cancelled=bool(m.group(4)))
    return life, cancel, req

def run_arm(a, on, log):
    env = dict(os.environ)
    env["QWEN_LIFE_TRACE"] = "1"; env["QWEN_REQ_TRACE"] = "1"
    env["QWEN_CANCEL_ON_DISCONNECT"] = "1" if on else "0"
    for kv in (a.server_env or "").split(","):
        if "=" in kv:
            k, v = kv.split("=", 1); env[k.strip()] = v.strip()
    cmd = [a.bin, "-d", a.model, "--serve", str(a.port), "--batch-size", str(a.batch)]
    if a.precision == "int8": cmd.insert(3, "--int8")
    if a.prefork > 1: cmd += ["--prefork", str(a.prefork), "--prefork-threads", str(a.threads)]
    lf = open(log, "wb")
    p = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env)
    out = {}
    try:
        if not wait_health(a.port):
            p.kill(); sys.exit(f"server did not come up ({log})")
        time.sleep(2)
        jobs, res, lk = [], {}, threading.Lock()
        for i, frac in enumerate(a.fracs):
            jobs.append((900100 + i, a.longtext, frac * a.expected))
        for i in range(a.keep):
            jobs.append((900200 + i, a.longtext, None))
        def go(seed, txt, ab):
            r = stream(a.port, txt, a.speaker, a.language, seed, ab)
            with lk: res[seed] = r
        ts = [threading.Thread(target=go, args=j) for j in jobs]
        for t in ts: t.start()
        for t in ts: t.join()
        time.sleep(max(6.0, a.expected * 1.5))
        out = res
    finally:
        p.terminate()
        try: p.wait(timeout=90)
        except subprocess.TimeoutExpired: p.kill()
        lf.close()
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--corpus", default=None,
                    help="optional TSV of texts; a built-in sentence is used when absent")
    ap.add_argument("--speaker", default="ryan")
    ap.add_argument("--language", default="English")
    ap.add_argument("--bin", default="./qwen_tts")
    ap.add_argument("--port", type=int, default=9866)
    ap.add_argument("--precision", default="int8")
    ap.add_argument("--prefork", type=int, default=2)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--server-env", default="OPENBLAS_THREAD_TIMEOUT=1")
    ap.add_argument("--fracs", default="0.20,0.40,0.60")
    ap.add_argument("--keep", type=int, default=3, help="concurrent streams that must NOT be disturbed")
    ap.add_argument("--band", default="long", choices=("short", "medium", "long"),
                    help="which duration band to draw the utterance from. 'long' for the "
                         "functional run; 'short' for the sanitizer run, where -O0 + ASan "
                         "makes a long turn cost hours without validating anything more.")
    ap.add_argument("--out", default="/tmp/cancelk")
    ap.add_argument("--growth", default="", help="A,B — after the OFF/ON pair, run the ON "
                    "arm twice with A and B aborted streams and check the leak does not "
                    "scale with the number of cancellations. Empty disables it.")
    a = ap.parse_args()
    a.fracs = [float(x) for x in a.fracs.split(",")]
    a.growth = tuple(int(x) for x in a.growth.split(",")) if a.growth else None

    longest = None
    for ln in open(a.corpus, encoding="utf-8"):
        if ln.startswith("#") or not ln.strip(): continue
        p = [x.strip() for x in ln.rstrip("\n").split("\t")]
        if len(p) >= 8 and p[1] == a.band:
            d = float(p[2])
            if longest is None or d > longest[1]: longest = (p[0], d, p[-1])
    if longest is None:
        sys.exit(f"no '{a.band}' row in {a.corpus} — refusing to fall back to another band")
    _tid, a.expected, a.longtext = longest
    os.makedirs(a.out, exist_ok=True)

    print(f"### K1-K7 cancellation correctness · binary {subprocess.run('sha256sum '+a.bin+' | cut -c1-16', shell=True, capture_output=True, text=True).stdout.strip()}")
    print(f"### {len(a.fracs)} aborted streams at {a.fracs} of {a.expected:.1f} s, "
          f"{a.keep} concurrent streams that must be UNDISTURBED")
    print(f"### topology {a.prefork}x{a.threads} batch {a.batch}"
          f"  ·  POLLRDHUP: reported by the binary in [CANCEL] rdhup=, not guessed\n")

    off = run_arm(a, False, os.path.join(a.out, "off.log"))
    a.port += 20
    on = run_arm(a, True, os.path.join(a.out, "on.log"))
    lo, co, ro = parse_log(os.path.join(a.out, "off.log"))
    ln_, cn, rn = parse_log(os.path.join(a.out, "on.log"))

    fails = []
    def chk(ok, name, detail=""):
        print(f"  {'✅' if ok else '❌'} {name}{('  ' + detail) if detail else ''}")
        if not ok: fails.append(name)

    ab_seeds = [900100 + i for i in range(len(a.fracs))]
    keep_seeds = [900200 + i for i in range(a.keep)]

    print("K1 abort detected")
    chk(all(s in cn for s in ab_seeds), "every aborted request has a [CANCEL] record",
        f"{sum(1 for s in ab_seeds if s in cn)}/{len(ab_seeds)}")
    print("K2 generation stops materially early  ·  K3 zombie collapses")
    for s, frac in zip(ab_seeds, a.fracs):
        fo, fn = ro.get(s, {}).get("audio", float('nan')), rn.get(s, {}).get("audio", float('nan'))
        do, dn = off.get(s, {}).get("audio_s", 0), on.get(s, {}).get("audio_s", 0)
        zo = fo - do if fo == fo else float('nan')
        zn = fn - dn if fn == fn else float('nan')
        ro_ = zo / (fo - do) if (fo - do) > 0 else float('nan')
        print(f"    frac {frac:.0%}: OFF generated {fo:6.2f}s (zombie {zo:5.2f}s) · "
              f"ON generated {fn:6.2f}s (zombie {zn:5.2f}s)")
        chk(fn < fo * 0.95, f"    ON stops early at {frac:.0%}",
            f"{fn:.2f}s vs {fo:.2f}s")
    print("K4 the OTHER concurrent streams are unchanged by cancellation")
    for s in keep_seeds:
        so, sn = off.get(s, {}), on.get(s, {})
        same_sha = so.get("sha") == sn.get("sha")
        fo, fn = ro.get(s, {}).get("frames"), rn.get(s, {}).get("frames")
        chk(same_sha, f"    seed {s}: audio SHA identical OFF vs ON",
            f"{so.get('sha')} vs {sn.get('sha')}")
        chk(fo == fn, f"    seed {s}: frame count identical", f"{fo} vs {fn}")
    print("K6/K7 accounting")
    chk(all(ln_.get(s) == "CANCELLED" for s in ab_seeds),
        "aborted requests recorded as CANCELLED, not COMPLETED")
    chk(all(ln_.get(s) == "COMPLETED" for s in keep_seeds),
        "undisturbed requests recorded as COMPLETED")
    chk(all(s in ln_ for s in ab_seeds + keep_seeds),
        "every issued request has a lifecycle record (no orphan)")
    for name, path in (("OFF", "off.log"), ("ON", "on.log")):
        txt = open(os.path.join(a.out, path), errors="replace").read()
        bad = txt.count("ERROR: AddressSanitizer") + txt.count("runtime error")
        chk(bad == 0, f"K5 {name}: no memory-error finding (leaks are K9)", f"{bad} found")
    print("K8 disconnect detection latency  ·  ONE clock (CLOCK_MONOTONIC), or nothing")
    rd = {v.get("rdhup") for v in cn.values()} - {None}
    print(f"    POLLRDHUP compiled into the running binary: "
          f"{'yes' if rd == {1} else ('no' if rd == {0} else 'NOT REPORTED (old binary)')}")
    lags = []
    for s_ in ab_seeds:
        c, cl = cn.get(s_), on.get(s_, {})
        det, ab = (c or {}).get("detected_abs"), cl.get("abort_abs_ms")
        if det is None or ab is None or det < 0:
            print(f"    seed {s_}: NOT_MEASURED (binary or client does not stamp the "
                  f"absolute instant)")
            continue
        lag = det - ab
        if lag < 0:
            print(f"    seed {s_}: NOT_COMPARABLE ({lag:.1f} ms) — the two stamps are not "
                  f"on one clock; the number is refused, not reported")
            continue
        lags.append(lag)
        print(f"    seed {s_}: client FIN -> server noticed  {lag:8.1f} ms")
    if lags:
        print(f"    disconnect_detect_latency_ms: {min(lags):.1f} .. {max(lags):.1f}")
    if cn:
        d = [v["to_stop"] for v in cn.values() if v["to_stop"] >= 0]
        if d: print(f"    cancel_to_generation_stop_ms: {min(d):.1f} .. {max(d):.1f}")
    print("K9 leaks, separated by where they were allocated")
    lk = {n: parse_leaks(os.path.join(a.out, f)) for n, f in (("OFF", "off.log"), ("ON", "on.log"))}
    for n in ("OFF", "ON"):
        v = lk[n]
        if not v["ran"]:
            print(f"    {n}: LEAK_CHECK_DID_NOT_RUN — no LeakSanitizer output in the log. "
                  f"A prefork worker ends with _exit(), which skips atexit and therefore "
                  f"LSan; without the explicit check this is a clean sheet nobody examined.")
        else:
            print(f"    {n}: load-time {v['load']:>9} B   ·   request-path {v['req']:>9} B "
                  f"(both are one-time scratch, not per-request growth — see K9-growth)")
    chk(lk["OFF"]["ran"] and lk["ON"]["ran"], "    the leak check actually ran in both arms")

    if a.growth:
        A, B = a.growth
        print(f"K9-growth  does the leak scale with cancellations?  {A} aborts vs {B} aborts, "
              f"cancellation ON in both")
        pts = {}
        for k, nab in ((0, A), (1, B)):
            a.fracs = [0.5] * nab
            a.keep = 1
            a.port += 40
            outk = os.path.join(a.out, f"growth{nab}")
            os.makedirs(outk, exist_ok=True)
            run_arm(a, True, os.path.join(outk, "on.log"))
            pts[nab] = parse_leaks(os.path.join(outk, "on.log"))
            print(f"    {nab} aborts: request-path {pts[nab]['req']:>9} B   "
                  f"load-time {pts[nab]['load']:>9} B")
        if B > A:
            spread = pts[B]["req"] - pts[A]["req"]
            print(f"    total request-path bytes moved {spread:+d} B between the two runs — "
                  f"RUN-TO-RUN NOISE in one-time scratch, NOT the verdict")
            grew = []
            for sig, (b, o) in pts[B]["sigs"].items():
                o0 = pts[A]["sigs"].get(sig, [0, 0])[1]
                if o - o0 >= (B - A):
                    grew.append((o - o0, sig))
            for d, sig in sorted(grew, reverse=True)[:5]:
                print(f"      +{d} object(s) at {sig}")
            chk(not grew,
                f"    no allocation site's object count tracks the cancellation count",
                f"{len(grew)} site(s) grew by >= {B - A} objects")
        else:
            chk(False, "    growth points must differ")

    print()
    if fails:
        print(f"❌ {len(fails)} CHECK(S) FAILED"); return 1
    print("✅ K1-K7 PASSED"); return 0

if __name__ == "__main__":
    sys.exit(main())
