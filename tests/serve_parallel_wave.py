#!/usr/bin/env python3
"""serve_parallel_wave.py — SYNCHRONIZED PARALLEL CAPACITY."""
import argparse, datetime, json, os, re, signal, socket, subprocess, sys, threading, time

RUN_DATE = datetime.date.today().isoformat()
import urllib.request
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import serve_procstats as PS

TEXTS = []

def load_texts(path, only_classes=None):
    """Mirror of load_test.py:load_texts, including the per-class interleave: the bank
    is written GROUPED by class, so a plain head-first round-robin would hand a small
    level nothing but `short` and flatter it."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for ln in f:
            ln = ln.rstrip("\n")
            if not ln.strip() or ln.lstrip().startswith("#"):
                continue
            if "\t" in ln:
                parts = [p.strip() for p in ln.split("\t")]
                cls = parts[1] if len(parts) >= 8 else parts[0]
                txt = parts[-1]
            else:
                cls, txt = "medium", ln.strip()
            if txt and (not only_classes or cls in only_classes):
                rows.append((cls, txt))
    by = {}
    for cls, txt in rows:
        by.setdefault(cls, []).append((cls, txt))
    out, i = [], 0
    while any(by.values()):
        for cls in sorted(by):
            if i < len(by[cls]):
                out.append(by[cls][i])
        if all(i >= len(v) - 1 for v in by.values()):
            break
        i += 1
    if only_classes and not rows:
        raise SystemExit(
            f"REFUSING TO RUN: {path} has no rows in classes {sorted(only_classes)}.\n"
            f"  A benchmark with an empty bank used to fall back to one hardcoded sentence\n"
            f"  and measure that instead, which is how a whole rung was run against\n"
            f"  \"Hello there.\" at 1.92 s while claiming a 3.68-7.92 s corpus.")
    return out or rows

def wait_port(port, timeout=400):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with socket.create_connection(("127.0.0.1", port), 1.0):
                return True
        except OSError:
            time.sleep(0.5)
    return False

def tree_pids(root):
    out, seen, stack = [root], {root}, [root]
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
    c = dict(vol=0, nonvol=0, migr=0, ut=0, st=0, rss=0, pss=0)
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
                    c["vol"] += int(re.search(r"(\d+)", l).group(1))
                elif l.startswith("nonvoluntary_ctxt_switches"):
                    c["nonvol"] += int(re.search(r"(\d+)", l).group(1))
            try:
                m = re.search(r"nr_migrations\s*:\s*(\d+)",
                              open(f"/proc/{pid}/task/{t}/sched").read())
                if m: c["migr"] += int(m.group(1))
            except OSError:
                pass
        try:
            f = open(f"/proc/{pid}/stat").read().rsplit(") ", 1)[1].split()
            c["ut"] += int(f[11]); c["st"] += int(f[12])
        except Exception:
            pass
        try:
            for l in open(f"/proc/{pid}/smaps_rollup"):
                m = re.match(r"(\w+):\s+(\d+) kB", l)
                if m and m.group(1) == "Rss": c["rss"] += int(m.group(2))
                if m and m.group(1) == "Pss": c["pss"] += int(m.group(2))
        except OSError:
            pass
    return c

SAVE_AUDIO_DIR = None

def _write_wav(path, pcm):
    """s16le mono 24 kHz, header written by hand — the harness already owns the framing."""
    import struct
    n = len(pcm)
    hdr = (b"RIFF" + struct.pack("<I", 36 + n) + b"WAVEfmt " +
           struct.pack("<IHHIIHH", 16, 1, 1, 24000, 24000 * 2, 2, 16) +
           b"data" + struct.pack("<I", n))
    with open(path, "wb") as f:
        f.write(hdr); f.write(pcm)

def stream_kpis(marks, total_s):
    """marks = [(t_rel_s, nbytes)] in arrival order, t_rel measured from request send.

    underrun_s simulates a zero-jitter-buffer player that starts the instant the first
    chunk lands and never pauses on purpose: between two chunk arrivals it plays what
    it holds, and any wall time it spends with an empty buffer is counted. Reported
    both as a total and as the worst single stall.
    """
    if len(marks) < 2:
        return {"stream_rtf": float("nan"), "underrun_s": float("nan"),
                "stall_max_s": float("nan"), "chunks": len(marks), "ratios": []}
    t_first = marks[0][0]
    rest_bytes = sum(nb for _, nb in marks[1:])
    rest_s = rest_bytes / 2.0 / 24000.0
    stream_rtf = (total_s - t_first) / rest_s if rest_s > 0 else float("nan")

    ratios, avail, play, prev_t, stall, stall_max = [], marks[0][1] / 48000.0, 0.0, t_first, 0.0, 0.0
    prebuf = 0.0
    for t, nb in marks[1:]:
        dur = nb / 2.0 / 24000.0
        gap = t - prev_t
        if dur > 0: ratios.append(gap / dur)
        prebuf = max(prebuf, (t - t_first) - avail)
        want = play + gap
        if want > avail:
            d = want - avail
            stall += d; stall_max = max(stall_max, d); play = avail
        else:
            play = want
        avail += dur; prev_t = t
    return {"stream_rtf": stream_rtf, "underrun_s": stall, "stall_max_s": stall_max,
            "prebuffer_s": max(0.0, prebuf), "chunks": len(marks), "ratios": ratios}

def one_request(port, out, lock, idx=0, speaker="ryan", language="English", seed=42):
    if not TEXTS:
        raise SystemExit("REFUSING TO RUN: the text bank is empty")
    cls, txt = TEXTS[idx % len(TEXTS)]
    body = json.dumps({"text": txt, "speaker": speaker, "language": language,
                       "temperature": 0.0, "seed": seed + idx}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/tts/stream", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time(); ttfa = None; n = 0; chunks = []; marks = []
    try:
        with urllib.request.urlopen(req, timeout=1200) as r:
            while True:
                ch = r.read1(65536)
                if not ch: break
                tnow = time.time() - t0
                if ttfa is None: ttfa = tnow
                marks.append((tnow, len(ch)))
                n += len(ch)
                if SAVE_AUDIO_DIR is not None: chunks.append(ch)
    except Exception as e:
        with lock: out.append({"err": str(e)})
        return
    total = time.time() - t0
    secs = n / 2.0 / 24000.0
    if SAVE_AUDIO_DIR is not None and chunks:
        os.makedirs(SAVE_AUDIO_DIR, exist_ok=True)
        _write_wav(os.path.join(SAVE_AUDIO_DIR,
                   f"{speaker}_r{idx:03d}_{cls}.wav"), b"".join(chunks))
    with lock:
        rec = {"cls": cls, "ttfa_ms": (ttfa or 0) * 1000.0, "total_s": total,
               "audio_s": secs, "seed": seed + idx, "idx": idx,
               "t_send": t0, "marks": marks,
               "rtf": total / secs if secs > 0 else float("nan")}
        rec.update(stream_kpis(marks, total))
        out.append(rec)

def canonical_ttfa(port, requests, speaker, language, seed, text_file, classes, out, tag):
    """Cross-check TTFA against tests/load_test.py at c=1, which decodes the chunk framing
    by hand and stamps TTFA on the first chunk. Returns its ttfa_p50 in ms, or None.

    A client that reads with a fixed-size read() blocks until it HAS that many bytes, so it
    times buffered audio rather than the first chunk. Any harness claiming to measure TTFA
    has to be checked against one that decodes the framing.
    """
    js = os.path.join(out, f"{tag}_oracle.json")
    cmd = ["python3", "tests/load_test.py", "--url", f"http://127.0.0.1:{port}",
           "--concurrency", "1", "--requests", str(requests),
           "--arrival", "all-at-once", "--speaker", speaker, "--language", language,
           "--seed", str(seed), "--text-file", text_file,
           "--no-save-audio", "--json", js]
    if classes:
        cmd += ["--classes", classes]
    try:
        subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        data = json.load(open(js))
        row = data[0] if isinstance(data, list) else data
        if isinstance(row, dict) and "levels" in row:
            row = row["levels"][0]
        return float(row["ttfa_p50"])
    except Exception:
        return None

def pct(v, q):
    """Percentile by NEAREST RANK, and deliberately not the interpolating definition
    tests/load_test.py uses. On a small even sample the two differ by a whole rank, so the
    cross-check below widens its tolerance instead of pretending they are comparable.
    """
    if not v: return float("nan")
    v = sorted(v)
    return v[min(len(v) - 1, int(round(q / 100.0 * (len(v) - 1))))]

ARRIVAL_TRUE_WAVE = "TRUE_SIMULTANEOUS_WAVE"

def workload_class(text_file, classes=""):
    """The workload is a property of the corpus and the class filter, never of a run's name."""
    base = os.path.basename(text_file)
    if not classes:
        stem = base[:-4] if base.endswith(".txt") else base
        if stem.endswith("_fast"):
            return "FAST_ENGINEERING"
        if stem.startswith("load_texts"):
            return "REALISTIC_QUALIFICATION"
    cl = {c.strip() for c in classes.split(",") if c.strip()}
    if cl == {"short"}:  return "PARALLEL_SHORT_DIVERSE"
    if cl == {"long"}:   return "PARALLEL_LONG_DIVERSE"
    if cl == {"medium"}: return "PARALLEL_MEDIUM_DIVERSE"
    if "diverse" in base or "corpus" in text_file:
        return "MIXED_PRODUCTION"
    return "UNSPECIFIED_WORKLOAD"

def result_slug(workload, arrival, topo, conc, model_label, precision):
    """A result filename has to be readable without opening it, e.g.
        <date>_true-wave_short-diverse_int8_2x8_c6.json
    """
    a = {"TRUE_SIMULTANEOUS_WAVE": "true-wave", "POISSON_OPEN_LOOP": "poisson",
         "CLOSED_LOOP_PARALLEL_SOAK": "closed-loop-soak"}.get(arrival, arrival.lower())
    w = workload.lower().replace("_", "-").replace("parallel-", "")
    return f"{w}_{a}_{model_label}-{precision}_{topo}_c{conc}"

def result_header(a, model_path, extra_env):
    def sh(cmd, default="UNKNOWN"):
        try:
            out = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
            v = out.stdout.strip()
            return v if v else default
        except Exception:
            return default
    mt = sh("curl -s -m 2 -H 'Metadata-Flavor: Google' "
            "http://169.254.169.254/computeMetadata/v1/instance/machine-type")
    if mt != "UNKNOWN": mt = mt.rsplit("/", 1)[-1]
    cpu = sh("lscpu | awk -F: '/Model name/{gsub(/^ +/,\"\",$2); print $2; exit}'")
    ncpu = sh("nproc")
    kern = sh("uname -sr")
    bsha = sh(f"sha256sum {a.bin} 2>/dev/null | cut -c1-16")
    brev = sh(f"{a.bin} --caps 2>/dev/null | awk '/build:/{{print $2}}'")
    commit = os.environ.get("QWEN_SOURCE_COMMIT") or sh("git rev-parse --short HEAD 2>/dev/null")
    dirty = os.environ.get("QWEN_SOURCE_DIRTY") or \
        sh("git diff --quiet HEAD 2>/dev/null && echo no || echo yes", "UNKNOWN")
    watched = ["OPENBLAS_THREAD_TIMEOUT", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
               "QWEN_KAI_QKV_FUSED", "QWEN_POOL_NARROW", "QWEN_POOL_SPIN",
               "QWEN_PREFIX_CACHE", "QWEN_NO_SMMLA", "QWEN_NO_BFMMLA"]
    env = dict(os.environ); env.update(extra_env)
    flags = " ".join(f"{k}={env.get(k, '(default)')}" for k in watched)
    print("### ─────────── RESULT IDENTITY ───────────")
    print(f"### machine_type=   {mt}    cpu_model= {cpu}  vcpu= {ncpu}")
    print(f"### kernel=         {kern}")
    print(f"### binary_sha256=  {bsha}    binary_build_tag= {brev}")
    print(f"### source_commit=  {commit}  dirty= {dirty}")
    print(f"### model=          {model_path}")
    print(f"### speaker=        {a.speaker}   language= {a.language}   seed_base= {a.seed}")
    wl = workload_class(a.text_file, a.classes)
    print(f"### benchmark_family=  {ARRIVAL_TRUE_WAVE}")
    print(f"### workload_class=    {wl}")
    print(f"### arrival_model=     {ARRIVAL_TRUE_WAVE}  (C requests at t=0, wait for ALL,"
          f" then the next wave)")
    print(f"### topology=          {a.topo}   concurrency= {a.conc}   waves= {a.waves}")
    backend = sh(a.bin + " --caps 2>/dev/null | sed -n 's/^  int8 dot: *//p'")
    print(f"### backend=           {backend}")
    print(f"### precision=         {a.precision}")
    print(f"### runtime_profile=   {a.server_env or '(compiled defaults)'}")
    print(f"### text_bank=         {os.path.basename(a.text_file)}"
          f"{'   classes= ' + a.classes if a.classes else ''}")
    print(f"### runtime_flags=  {flags}")
    print(f"### server_env=     {a.server_env or '(none: compiled defaults)'}")
    print("### ────────────────────────────────────────")
    return {
        "machine_type": mt, "cpu_model": cpu, "vcpu": ncpu, "kernel": kern,
        "binary_sha256": bsha, "binary_build_tag": brev,
        "source_commit": commit, "dirty": dirty,
        "model": model_path, "speaker": a.speaker, "language": a.language,
        "seed_base": a.seed,
        "benchmark_family": ARRIVAL_TRUE_WAVE, "workload_class": wl,
        "arrival_model": ARRIVAL_TRUE_WAVE, "topology": a.topo, "concurrency": a.conc,
        "waves": a.waves, "backend": backend, "precision": a.precision,
        "runtime_profile": a.server_env or "(compiled defaults)",
        "runtime_flags": {k: env.get(k, "(default)") for k in watched},
        "text_bank": os.path.basename(a.text_file), "classes": a.classes,
        "harness": os.path.basename(__file__), "run_date": RUN_DATE,
    }

PROFILE_DIR  = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "configs", "perf")
PROFILE_TOOL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "tools", "perf_profile.py")

def resolve_profile(a):
    """Compose the server environment from the named profile, then apply explicit overrides.

    Refuses to run with neither: a silent default is how a table gets printed for a
    configuration nobody chose.
    """
    if a.profile and a.no_profile:
        raise SystemExit("REFUSING TO RUN: --profile and --no-profile are mutually exclusive.")
    if not a.profile and not a.no_profile:
        try:
            avail = ", ".join(sorted(f[:-5] for f in os.listdir(PROFILE_DIR)
                                     if f.endswith(".json") and f != "schema.json"))
        except OSError:
            avail = "(configs/perf not found)"
        raise SystemExit(
            "REFUSING TO RUN: no --profile and no --no-profile.\n"
            "  A serving result is identified by its runtime configuration, and a flag left\n"
            "  at a default nobody chose is an invisible variable. Pass one of:\n"
            "     --profile <name>          the platform's declared profile\n"
            "     --no-profile '<reason>'   compiled defaults, on purpose\n"
            "  Available profiles: " + avail)
    explicit = dict(kv.split("=", 1) for kv in (a.server_env or "").split(",") if "=" in kv)
    if a.no_profile:
        print("### profile=           NONE - " + a.no_profile)
        a.profile_argv = []
        return a.server_env
    out = subprocess.run([sys.executable, PROFILE_TOOL, "server-env", a.profile],
                         capture_output=True, text=True)
    if out.returncode != 0:
        raise SystemExit("REFUSING TO RUN: profile %r did not resolve:\n%s"
                         % (a.profile, out.stderr or out.stdout))
    from_profile = dict(kv.split("=", 1) for kv in out.stdout.strip().split(",") if "=" in kv)
    merged = dict(from_profile)
    for k, v in explicit.items():
        if k in from_profile and from_profile[k] != v:
            print("### profile_override=  %s: %r -> %r (explicit)" % (k, from_profile[k], v))
        merged[k] = v
    print("### profile=           " + a.profile)
    a.profile_argv = profile_server_argv(a)
    if a.profile_argv:
        print("### profile_args=      " + " ".join(a.profile_argv))
    return ",".join("%s=%s" % (k, v) for k, v in sorted(merged.items()))

HARNESS_OWNED = {"--serve", "--batch-size", "--prefork", "--prefork-threads",
                 "--prefork-elastic", "-d", "--int8"}

def profile_server_argv(a):
    if not a.profile:
        return []
    out = subprocess.run([sys.executable, PROFILE_TOOL, "command", a.profile,
                          "--model", "MODEL", "--port", "0"],
                         capture_output=True, text=True)
    if out.returncode != 0:
        return []
    toks = out.stdout.split()
    try:
        toks = toks[toks.index("MODEL") + 1:]
    except ValueError:
        return []
    keep, i = [], 0
    while i < len(toks):
        t = toks[i]
        if not t.startswith("-"):
            i += 1
            continue
        val = toks[i + 1] if i + 1 < len(toks) and not toks[i + 1].startswith("-") else None
        if t not in HARNESS_OWNED:
            keep.append(t)
            if val is not None:
                keep.append(val)
        i += 2 if val is not None else 1
    return keep

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--out", default="/tmp/kai_par")
    ap.add_argument("--topo", default="2x8,4x4,8x2")
    ap.add_argument("--conc", default="1,2,3,4,5,6,8")
    ap.add_argument("--waves", type=int, default=3)
    ap.add_argument("--precision", default="int8")
    ap.add_argument("--bin", default="./qwen_tts")
    ap.add_argument("--port", type=int, default=9300)
    ap.add_argument("--text-file", default=os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "load_texts_en.txt"))
    ap.add_argument("--classes", default="", help="e.g. short,medium — filter the text bank")
    ap.add_argument("--speaker", default="ryan")
    ap.add_argument("--language", default="English")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--crosscheck-tol", type=float, default=0.15,
                    help="at C=1 the TTFA must stay within this fraction of the one "
                         "measured by tests/load_test.py, or the run is declared INVALID")
    ap.add_argument("--no-crosscheck", action="store_true")
    ap.add_argument("--server-env", default="", metavar="K=V,K=V",
                    help="env applied to the SERVER process only (A/B arms). Merged ON TOP "
                         "of --profile, and every override is announced.")
    ap.add_argument("--profile", default="", metavar="NAME",
                    help="deployment profile from configs/perf; supplies the server env")
    ap.add_argument("--no-profile", default="", metavar="REASON",
                    help="run with compiled defaults instead of a profile, stating why")
    ap.add_argument("--save-audio", default="", metavar="DIR",
                    help="write one WAV per request into DIR (quality gate; perturbs timing)")
    a = ap.parse_args()
    a.server_env = resolve_profile(a)
    global TEXTS
    TEXTS = load_texts(a.text_file,
                       set(x.strip() for x in a.classes.split(",") if x.strip()) or None)
    os.makedirs(a.out, exist_ok=True)
    label = a.label or os.path.basename(a.model.rstrip("/"))
    concs = [int(x) for x in a.conc.split(",")]
    hz = os.sysconf("SC_CLK_TCK")
    rows, port = [], a.port
    crosschecked = False
    bad_crosscheck = False

    RESULT_ID = result_header(a, a.model, dict(kv.split("=", 1) for kv in
                                               (a.server_env or "").split(",") if "=" in kv))
    print(f"### SYNCHRONIZED PARALLEL CAPACITY — model {label}")
    print(f"### {a.waves} waves per level; every wave fires C requests at t=0 and waits")
    print(f"### text bank {os.path.basename(a.text_file)} ({len(TEXTS)} texts"
          f"{', classes ' + a.classes if a.classes else ''}) — the SAME bank the Poisson")
    print(f"### harness uses, so the two tables differ only in the ARRIVAL PATTERN.")
    print(f"### for ALL of them. NOT comparable with the Poisson curve.\n")

    global SAVE_AUDIO_DIR
    if a.save_audio:
        SAVE_AUDIO_DIR = a.save_audio
        print(f"### ⚠️ --save-audio {a.save_audio}: this is a QUALITY run, not a timing run")
    for topo in a.topo.split(","):
        elastic = topo.endswith("e")
        W, K = (int(x) for x in topo.rstrip("e").split("x"))
        cap = max(1, 16 // W)
        cmd = [a.bin, "-d", a.model, "--serve", str(port), "--batch-size", str(cap)]
        if a.precision == "int8":
            cmd.insert(3, "--int8")
        cmd += ["--prefork", str(W), "--prefork-threads", str(8 if elastic else K)]
        if elastic:
            cmd += ["--prefork-elastic"]
        cmd += getattr(a, "profile_argv", [])
        log = os.path.join(a.out, f"{label}_{topo}.log")
        f = open(log, "wb")
        senv = dict(os.environ)
        want = {}
        for kv in (a.server_env or "").split(","):
            if "=" in kv:
                k, v = kv.split("=", 1)
                if v.strip() and (" " in v.strip() or "\t" in v.strip()):
                    sys.exit(f"--server-env: value of {k.strip()} is {v.strip()!r}, which "
                             f"contains whitespace — the separator is a COMMA, not a space. "
                             f"Got: {a.server_env!r}")
                senv[k.strip()] = v.strip(); want[k.strip()] = v.strip()
            elif kv.strip():
                sys.exit(f"--server-env: '{kv.strip()}' has no '=' — the list separator is a "
                         f"COMMA, not a space. Got: {a.server_env!r}")
        p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=senv)
        print(f"=== {label} {topo} (cap {cap}/worker, port {port}) ===", flush=True)
        try:
            if not wait_port(port):
                print("  server did not come up"); continue
            time.sleep(5)
            if want:
                for _ in range(120):
                    try:
                        if "[FLAGS]" in open(log, "rb").read().decode("utf-8", "replace"):
                            break
                    except OSError:
                        pass
                    time.sleep(0.5)
            if want:
                seen, txt = {}, ""
                try:
                    txt = open(log, "rb").read().decode("utf-8", "replace")
                except OSError:
                    pass
                for line in txt.splitlines():
                    if line.startswith("[FLAGS]"):
                        for tok in line.split()[2:]:
                            if "=" in tok:
                                k, v = tok.split("=", 1); seen[k] = v
                missing = {k: v for k, v in want.items()
                           if k.startswith("QWEN_") and seen.get(k) != v}
                if missing:
                    p.kill()
                    sys.exit(f"server did not receive {missing} (engine reported {seen}) — "
                             f"refusing to benchmark a configuration that is not running")
                print(f"  flags verified in the engine: "
                      f"{' '.join(f'{k}={v}' for k, v in sorted(seen.items())) or '(none)'}",
                      flush=True)
            pids = tree_pids(p.pid)
            res, lock = [], threading.Lock()
            for wi in range(min(4, W * 2)):
                one_request(port, res, lock, wi, a.speaker, a.language, a.seed + 900000)
            res.clear()
            wmap = {i: pid for i, pid, _cpus, _thr in PS.worker_pids_from_log(log)}

            for C in concs:
                try: p.send_signal(signal.SIGUSR1)
                except Exception: pass
                time.sleep(1.2)
                mark = os.path.getsize(log)
                c0 = counters(pids); t0 = time.time()
                pw0 = {i: PS.proc_sample(pid) for i, pid in wmap.items()}
                res.clear(); wave_s = []; ridx = 0
                for _ in range(a.waves):
                    w0 = time.time()
                    ts = []
                    for _k in range(C):
                        ts.append(threading.Thread(target=one_request,
                                                   args=(port, res, lock, ridx,
                                                         a.speaker, a.language,
                                                         a.seed + 1000 * C)))
                        ridx += 1
                    for t in ts: t.start()
                    for t in ts: t.join()
                    wave_s.append(time.time() - w0)
                wall = time.time() - t0; c1 = counters(pids)
                pw1 = {i: PS.proc_sample(pid) for i, pid in wmap.items()}
                try: p.send_signal(signal.SIGUSR1)
                except Exception: pass
                time.sleep(1.2)
                stats = ""
                try:
                    with open(log, "rb") as lf:
                        lf.seek(mark)
                        tail = lf.read().decode(errors="replace")
                    m = [l for l in tail.splitlines() if "[prefork-stats]" in l]
                    if m: stats = m[-1]
                except Exception:
                    pass
                ok = [r for r in res if "err" not in r]
                nerr = len(res) - len(ok)
                ps = PS.parse_prefork_stats(stats)
                wrows = PS.per_worker_rows(pw0, pw1, ps, wall)
                mi = re.search(r"mean_inflight ([\d.]+)", stats)
                asg = re.findall(r"w(\d+)\[asg=(\d+)", stats)
                rej = re.search(r"rejected (\d+)", stats)
                row = {
                    "model": label, "topo": topo, "W": W, "K": K, "cap": cap, "conc": C,
                    "waves": a.waves, "ok": len(ok), "errors": nerr,
                    "ttfa_p50": pct([r["ttfa_ms"] for r in ok], 50),
                    "ttfa_p95": pct([r["ttfa_ms"] for r in ok], 95),
                    "ttfa_max": max((r["ttfa_ms"] for r in ok), default=float("nan")),
                    "rtf_p50": pct([r["rtf"] for r in ok], 50),
                    "rtf_p95": pct([r["rtf"] for r in ok], 95),
                    "stream_p50": pct([r["stream_rtf"] for r in ok], 50),
                    "stream_p95": pct([r["stream_rtf"] for r in ok], 95),
                    "stream_max": max((r["stream_rtf"] for r in ok), default=float("nan")),
                    "underrun_p50": pct([r["underrun_s"] for r in ok], 50),
                    "underrun_p95": pct([r["underrun_s"] for r in ok], 95),
                    "underrun_max": max((r["underrun_s"] for r in ok), default=float("nan")),
                    "starved_req": sum(1 for r in ok if r.get("underrun_s", 0) > 0.001),
                    "prebuf_p50": pct([r["prebuffer_s"] for r in ok], 50),
                    "prebuf_p95": pct([r["prebuffer_s"] for r in ok], 95),
                    "prebuf_max": max((r["prebuffer_s"] for r in ok), default=float("nan")),
                    "gap_ratio_p50": pct([x for r in ok for x in r.get("ratios", [])], 50),
                    "gap_ratio_p95": pct([x for r in ok for x in r.get("ratios", [])], 95),
                    "gap_ratio_max": max((x for r in ok for x in r.get("ratios", [])),
                                         default=float("nan")),
                    "chunks_p50": pct([r["chunks"] for r in ok], 50),
                    "audio_p50": pct([r["audio_s"] for r in ok], 50),
                    "ttc_p50": pct(wave_s, 50), "ttc_p95": pct(wave_s, 95),
                    "req_s": len(ok) / wall if wall else 0,
                    "rejects": int(rej.group(1)) if rej else 0,
                    "batch_eff": float(mi.group(1)) if mi else float("nan"),
                    "assign": "/".join(v for _, v in asg) if asg else "-",
                    "cores": ((c1["ut"] - c0["ut"]) + (c1["st"] - c0["st"])) / hz / wall,
                    "csw_s": ((c1["vol"] - c0["vol"]) + (c1["nonvol"] - c0["nonvol"])) / wall,
                    "pss_mb": c1["pss"] / 1024.0,
                    "workers": wrows,
                }
                rows.append(row)
                if C == 1 and not a.no_crosscheck and not crosschecked:
                    crosschecked = True
                    orc = canonical_ttfa(port, a.waves, a.speaker, a.language, a.seed,
                                         a.text_file, a.classes, a.out, f"{label}_{topo}")
                    if orc is None:
                        print("  ⚠️  cross-check: the oracle did not run — TTFA UNVERIFIED",
                              flush=True)
                    else:
                        d = abs(row["ttfa_p50"] - orc) / orc if orc > 0 else 1.0
                        n_req = row.get("ok") or row.get("requests") or 0
                        tol = a.crosscheck_tol if n_req >= 8 else max(a.crosscheck_tol, 0.40)
                        ok_ = d <= tol
                        print(f"  {'✅' if ok_ else '❌'} cross-check C=1: this harness "
                              f"{row['ttfa_p50']:.0f} ms vs tests/load_test.py {orc:.0f} ms "
                              f"({d*100:.0f}% apart, tolerance {tol*100:.0f}% at n={n_req})"
                              + ("" if n_req >= 8 else
                                 "  [widened: at n<8 the median moves by a whole rank, and the "
                                 "two harnesses round percentiles differently]"),
                              flush=True)
                        if not ok_:
                            print("  ❌ the two disagree with NO contention to explain it: "
                                  "this harness is measuring something else. Run declared "
                                  "INVALID — fix the client, do not report these numbers.",
                                  flush=True)
                            bad_crosscheck = True
                print(f"  C={C:<2} TTFA p50 {row['ttfa_p50']:6.0f} p95 {row['ttfa_p95']:6.0f} "
                      f"max {row['ttfa_max']:6.0f} · RTF {row['rtf_p50']:.2f} · "
                      f"ttc {row['ttc_p50']:5.1f}s · {row['req_s']:.2f} req/s · "
                      f"B {row['batch_eff']:.2f} · asg {row['assign']} · "
                      f"cores {row['cores']:.1f} · err {nerr} rej {row['rejects']}", flush=True)
                print(f"     TOTAL_RTF p50 {row['rtf_p50']:.3f} p95 {row['rtf_p95']:.3f}"
                      f"  |  STREAM_RTF p50 {row['stream_p50']:.3f} p95 {row['stream_p95']:.3f}"
                      f" max {row['stream_max']:.3f}"
                      f"  |  underrun p50 {row['underrun_p50']*1000:.0f} ms"
                      f" p95 {row['underrun_p95']*1000:.0f} ms"
                      f" max {row['underrun_max']*1000:.0f} ms"
                      f" ({row['starved_req']}/{len(ok)} starved)"
                      f"  |  gap/dur p50 {row['gap_ratio_p50']:.2f}"
                      f" p95 {row['gap_ratio_p95']:.2f} max {row['gap_ratio_max']:.2f}"
                      f"  |  {row['chunks_p50']:.0f} chunks, {row['audio_p50']:.2f} s audio",
                      flush=True)
                print(f"     PREBUFFER needed for zero stall: p50 {row['prebuf_p50']*1000:.0f} ms"
                      f" p95 {row['prebuf_p95']*1000:.0f} ms max {row['prebuf_max']*1000:.0f} ms"
                      f"  ->  playback can start at TTFA+prebuf ="
                      f" {row['ttfa_p50'] + row['prebuf_p50']*1000:.0f} ms (p50),"
                      f" {row['ttfa_p95'] + row['prebuf_p95']*1000:.0f} ms (p95)", flush=True)
                with open(os.path.join(a.out,
                          f"{RUN_DATE}_{result_slug(workload_class(a.text_file, a.classes), ARRIVAL_TRUE_WAVE, topo, C, label, a.precision)}_requests.jsonl"), "w") as rf:
                    for r in ok: rf.write(json.dumps(r) + "\n")
                if wrows:
                    print(PS.format_rows(wrows), flush=True)
                json.dump(rows, open(os.path.join(a.out, f"parallel_{label}.json"), "w"), indent=1)
                slug = result_slug(workload_class(a.text_file, a.classes), ARRIVAL_TRUE_WAVE,
                                   topo, C, label, a.precision)
                json.dump({"result_id": RESULT_ID, "cell": row},
                          open(os.path.join(a.out, f"{RUN_DATE}_{slug}.json"), "w"),
                          indent=1, default=str)
        finally:
            p.send_signal(signal.SIGTERM)
            try: p.wait(timeout=240)
            except subprocess.TimeoutExpired: p.kill(); p.wait()
            f.close(); port += W + 2

    hdr = (f"{'topo':<6}{'C':>3}{'TTFA50':>8}{'TTFA95':>8}{'TTFAmax':>9}{'RTF50':>7}{'RTF95':>7}"
           f"{'ttc50':>7}{'ttc95':>7}{'req/s':>7}{'B':>6}{'assign':>12}{'cores':>7}"
           f"{'csw/s':>8}{'PSS GB':>8}{'rej':>5}{'err':>5}")
    print("\n" + hdr); print("-" * len(hdr))
    for r in rows:
        print(f"{r['topo']:<6}{r['conc']:>3}{r['ttfa_p50']:>8.0f}{r['ttfa_p95']:>8.0f}"
              f"{r['ttfa_max']:>9.0f}{r['rtf_p50']:>7.2f}{r['rtf_p95']:>7.2f}"
              f"{r['ttc_p50']:>7.1f}{r['ttc_p95']:>7.1f}{r['req_s']:>7.2f}"
              f"{r['batch_eff']:>6.2f}{r['assign']:>12}{r['cores']:>7.1f}"
              f"{r['csw_s']:>8.0f}{r['pss_mb']/1024:>8.1f}{r['rejects']:>5}{r['errors']:>5}")

    if bad_crosscheck:
        print("\n❌❌ TTFA CROSS-CHECK FAILED — the thresholds below are NOT meaningful.")
    print("\n### THE THRESHOLD (synchronized, all C at t=0)")
    for topo in a.topo.split(","):
        for budget in (500, 700, 1000):
            ins = [r for r in rows if r["topo"] == topo and r["errors"] == 0
                   and r["ttfa_p95"] <= budget]
            best = max(ins, key=lambda r: r["conc"]) if ins else None
            print(f"  {topo:<6} TTFA p95 < {budget:>4} ms  ->  " +
                  (f"C = {best['conc']}  (p95 {best['ttfa_p95']:.0f} ms, "
                   f"{best['req_s']:.2f} req/s)" if best else "never"))
    return 2 if bad_crosscheck else 0

if __name__ == "__main__":
    sys.exit(main() or 0)
