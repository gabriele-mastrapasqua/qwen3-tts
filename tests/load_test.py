#!/usr/bin/env python3
"""Load bench for the streaming TTS server, with first-audio latency as the primary metric.

Arrival modes: all-at-once (synchronised worst case, the default), poisson and uniform
(open loop, arrivals do not wait for the server). The generator seed is fixed and printed
so two runs are comparable.

Definitions (wall clock, client side):
  ttfa     first audio byte - request sent
  total    last audio byte  - request sent
  audio_s  PCM bytes / 2 / 24000 (int16 mono 24 kHz)
  RTF      total / audio_s
  Q        sum(audio_s) / wall

Usage:
  tests/load_test.py --speaker ryan --concurrency 1,2,4 --requests 8
  tests/load_test.py --concurrency 1,2,3,4 --arrival poisson --requests 6
  tests/load_test.py --concurrency 4 --arrival uniform --interval 1.5 --csv out.csv

Environment overrides for callers that cannot pass flags:
  QWEN_LT_ARRIVAL  QWEN_LT_RATE  QWEN_LT_INTERVAL  QWEN_LT_ARRIVAL_SEED
  QWEN_LT_TTFA_BUDGET_MS  QWEN_LT_SERVER_LOG  QWEN_LT_SERVICE_S
"""
import argparse, asyncio, csv, json, math, os, random, statistics, sys, time
from urllib.parse import urlparse

SR = 24000
BYTES_PER_SAMPLE = 2
DEFAULT_TEXTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "load_texts_en.txt")

NEIGHBOR_MS = 200.0

def load_texts(path, only_classes=None):
    """-> [(cls, text)]. Le righe sono '<classe>\\t<testo>'; '#' e vuote ignorate."""
    out = []
    with open(path, encoding="utf-8") as f:
        for ln in f:
            ln = ln.rstrip("\n")
            if not ln.strip() or ln.lstrip().startswith("#"):
                continue
            if "\t" in ln:
                cls, txt = ln.split("\t", 1)
            else:
                cls, txt = "medium", ln
            cls, txt = cls.strip(), txt.strip()
            if not txt:
                continue
            if only_classes and cls not in only_classes:
                continue
            out.append((cls, txt))
    if not out:
        sys.exit(f"{path}: no usable text (filter={only_classes})")

    buckets = {}
    for cls, txt in out:
        buckets.setdefault(cls, []).append(txt)
    order = list(buckets.keys())
    mixed = []
    i = 0
    while len(mixed) < len(out):
        for cls in order:
            b = buckets[cls]
            if i < len(b):
                mixed.append((cls, b[i]))
        i += 1
    return mixed

class ServerLogTail:
    """Read the server's stderr while it runs and stamp each line as it appears.

    The engine prints `[BATCH] done #N` without a clock, so the only way to know when it
    happened is to watch the file live. stderr in C is unbuffered, so the added delay is the
    sampling period (20 ms), not a library buffer. This is a correlation at tens of
    milliseconds, not tracing.
    """

    def __init__(self, path, poll_s=0.02):
        self.path = path
        self.poll_s = poll_s
        self.events = []
        self._stop = False
        self._fh = None
        self._buf = b""

    def open_at_end(self):
        """Opened and seeked to the end BEFORE the level starts, so warm-up events and the
        previous level's are not counted as belonging to this one."""
        try:
            self._fh = open(self.path, "rb")
            self._fh.seek(0, os.SEEK_END)
            return True
        except Exception:
            self._fh = None
            return False

    async def run(self, t0):
        if self._fh is None:
            return
        while not self._stop:
            try:
                data = self._fh.read()
            except Exception:
                data = b""
            if data:
                now = (time.perf_counter() - t0) * 1000.0
                self._buf += data
                *lines, self._buf = self._buf.split(b"\n")
                for ln in lines:
                    self.events.append((now, ln.decode("utf-8", "replace").rstrip()))
            await asyncio.sleep(self.poll_s)

    def stop(self):
        self._stop = True
        if self._fh:
            try:
                self._fh.close()
            except Exception:
                pass

    def done_events(self):
        """[(t_ms, cumulative_admitted)] from the `[BATCH] done #N (..., admitted=K)` lines."""
        out = []
        for t, ln in self.events:
            if "[BATCH] done" in ln:
                adm = None
                if "admitted=" in ln:
                    tail = ln.split("admitted=", 1)[1]
                    num = ""
                    for ch in tail:
                        if ch.isdigit():
                            num += ch
                        else:
                            break
                    adm = int(num) if num else None
                out.append((t, adm))
        return out

async def one_request(host, port, path, payload, req_id, cls, out_dir, save_audio,
                      timeout, t0, arrival_sched_ms, inflight, gate):
    """Stream one request, timing the FIRST body byte separately.

    `t0` is the level's time origin: every instant is relative to it, because spike
    attribution needs a timeline shared across requests.
    """
    body = json.dumps(payload).encode()
    head = (
        f"POST {path} HTTP/1.1\r\n"
        f"Host: {host}:{port}\r\n"
        f"Content-Type: application/json\r\n"
        f"Content-Length: {len(body)}\r\n"
        f"Connection: close\r\n\r\n"
    ).encode()

    rec = {"request_id": req_id, "class": cls, "text": payload["text"],
           "ttfa_ms": None, "total_ms": None, "audio_s": 0.0, "rtf": None,
           "bytes": 0, "status": None, "error": "",
           "arrival_sched_ms": arrival_sched_ms,
           "arrival_ms": None,
           "first_ms": None,
           "end_ms": None,
           "admit_wait_ms": 0.0}

    if gate is not None:
        g0 = time.perf_counter()
        await gate.acquire()
        rec["admit_wait_ms"] = (time.perf_counter() - g0) * 1000.0

    t_start = time.perf_counter()
    rec["arrival_ms"] = (t_start - t0) * 1000.0
    inflight.append(req_id)
    reader = writer = None
    try:
        reader, writer = await asyncio.wait_for(asyncio.open_connection(host, port), timeout)
        writer.write(head + body)
        await writer.drain()

        raw = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), timeout)
        status_line = raw.split(b"\r\n", 1)[0].decode("latin1")
        rec["status"] = int(status_line.split()[1]) if len(status_line.split()) > 1 else 0
        chunked = b"transfer-encoding: chunked" in raw.lower()

        chunks, first = [], None
        while True:
            if chunked:
                size_line = await asyncio.wait_for(reader.readuntil(b"\r\n"), timeout)
                n = int(size_line.strip().split(b";")[0], 16)
                if n == 0:
                    break
                data = await asyncio.wait_for(reader.readexactly(n), timeout)
                await reader.readexactly(2)
            else:
                data = await asyncio.wait_for(reader.read(65536), timeout)
                if not data:
                    break
            if data and first is None:
                first = time.perf_counter()
                rec["ttfa_ms"] = (first - t_start) * 1000.0
                rec["first_ms"] = (first - t0) * 1000.0
            chunks.append(data)

        payload_bytes = b"".join(chunks)
        t_end = time.perf_counter()
        rec["total_ms"] = (t_end - t_start) * 1000.0
        rec["end_ms"] = (t_end - t0) * 1000.0
        rec["bytes"] = len(payload_bytes)
        rec["audio_s"] = len(payload_bytes) / BYTES_PER_SAMPLE / SR
        if rec["audio_s"] > 0:
            rec["rtf"] = (rec["total_ms"] / 1000.0) / rec["audio_s"]
        else:
            rec["error"] = rec["error"] or "corpo vuoto"

        if save_audio and payload_bytes:
            _write_wav(os.path.join(out_dir, f"req{req_id:04d}_{cls}.wav"), payload_bytes)

    except Exception as e:                        # noqa: BLE001
        rec["error"] = f"{type(e).__name__}: {e}"
        t_end = time.perf_counter()
        rec["total_ms"] = (t_end - t_start) * 1000.0
        rec["end_ms"] = (t_end - t0) * 1000.0
    finally:
        try:
            inflight.remove(req_id)
        except ValueError:
            pass
        if gate is not None:
            gate.release()
        if writer is not None:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass
    return rec

def _write_wav(path, pcm):
    """Raw int16 PCM -> a real RIFF file. The server streams headerless PCM."""
    import struct
    n = len(pcm)
    hdr = (b"RIFF" + struct.pack("<I", 36 + n) + b"WAVEfmt " + struct.pack("<IHHIIHH", 16, 1, 1, SR, SR * 2, 2, 16)
           + b"data" + struct.pack("<I", n))
    with open(path, "wb") as f:
        f.write(hdr)
        f.write(pcm)

def arrival_offsets(mode, n, conc, rate, interval, service_s, seed):
    """Arrival instants (seconds from the start of the level) for n requests.

    The seed is FIXED and declared in the report: two runs with the same seed have the same
    arrival sequence, so a p95 difference is the machine and not another draw. Without
    that, over 6-8 requests, the noise of the Poisson process is easily larger than the
    effect being looked for.
    """
    if mode == "all-at-once":
        return [0.0] * n, None
    if mode == "uniform":
        gap = interval if interval and interval > 0 else (service_s / max(conc, 1))
        return [i * gap for i in range(n)], 1.0 / gap if gap > 0 else None
    if mode == "poisson":
        lam = rate if rate and rate > 0 else (max(conc, 1) / service_s)
        rng = random.Random(seed)
        out, t = [], 0.0
        for _ in range(n):
            out.append(t)
            t += -math.log(1.0 - rng.random()) / lam
        return out, lam
    raise SystemExit(f"modalita' di arrivo sconosciuta: {mode}")

async def run_level(args, host, port, texts, conc, service_s):
    out_dir = os.path.join(args.output_dir, f"c{conc}")
    if args.save_audio:
        os.makedirs(out_dir, exist_ok=True)

    tail = None
    if args.server_log:
        tail = ServerLogTail(args.server_log)
        if not tail.open_at_end():
            print(f"  ⚠️  --server-log {args.server_log}: cannot open, client-side attribution only")
            tail = None

    def payload_for(idx):
        cls, txt = texts[idx % len(texts)]
        return cls, {"text": txt, "speaker": args.speaker, "language": args.language,
                     "temperature": args.temperature, "seed": args.seed + idx}

    inflight = []
    gate = asyncio.Semaphore(args.max_inflight) if args.max_inflight > 0 else None
    records = []
    t0 = time.perf_counter()
    tail_task = asyncio.create_task(tail.run(t0)) if tail else None

    if args.arrival == "all-at-once" and args.duration:
        stop_at = t0 + args.duration
        sem = asyncio.Semaphore(conc)

        async def worker(idx):
            cls, pl = payload_for(idx)
            async with sem:
                if time.perf_counter() > stop_at:
                    return None
                return await one_request(host, port, args.path, pl, idx, cls, out_dir,
                                         args.save_audio, args.timeout, t0, 0.0, inflight, None)

        pending, idx = set(), 0
        while time.perf_counter() < stop_at or pending:
            while len(pending) < conc and time.perf_counter() < stop_at:
                pending.add(asyncio.create_task(worker(idx))); idx += 1
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            records.extend(r for r in (d.result() for d in done) if r)
        lam = None

    elif args.arrival == "all-at-once":
        sem = asyncio.Semaphore(conc)

        async def worker(idx):
            cls, pl = payload_for(idx)
            async with sem:
                return await one_request(host, port, args.path, pl, idx, cls, out_dir,
                                         args.save_audio, args.timeout, t0, 0.0, inflight, None)

        results = await asyncio.gather(*(worker(i) for i in range(args.requests)))
        records = [r for r in results if r]
        lam = None

    else:
        n = args.requests
        offs, lam = arrival_offsets(args.arrival, n, conc, args.rate, args.interval,
                                    service_s, args.arrival_seed)
        if args.duration:
            offs = [o for o in offs if o <= args.duration] or offs[:1]
        tasks = []
        for i, off in enumerate(offs):
            now = time.perf_counter() - t0
            if off > now:
                await asyncio.sleep(off - now)
            cls, pl = payload_for(i)
            tasks.append(asyncio.create_task(
                one_request(host, port, args.path, pl, i, cls, out_dir, args.save_audio,
                            args.timeout, t0, off * 1000.0, inflight, gate)))
        results = await asyncio.gather(*tasks)
        records = [r for r in results if r]

    wall = time.perf_counter() - t0
    if tail_task:
        tail.stop()
        try:
            await asyncio.wait_for(tail_task, 1.0)
        except Exception:
            tail_task.cancel()
    done_ev = tail.done_events() if tail else []
    attribute(records, done_ev, args.ttfa_budget_ms)
    return records, wall, len(records), lam, done_ev

def attribute(records, server_done_events, budget_ms):
    """For each request: what was happening in the server when it arrived, and while it
    waited for the first byte.

    The client-side columns (in flight, nearby arrivals, completions during the wait) are
    EXACT: we generate the arrivals and we see the last byte. The ones from the server log
    are an independent confirmation, timestamped by the tail.

    The control that makes the hypothesis falsifiable is `inflight_at_arrival == 0`:
    a spike on an idle machine is not contention, and it is the only case that rules
    out the scheduler without further instrumentation.
    """
    ev = [(r["arrival_ms"], r["end_ms"], r["request_id"]) for r in records
          if r["arrival_ms"] is not None]
    for r in records:
        a = r["arrival_ms"]
        f = r["first_ms"] if r["first_ms"] is not None else r["end_ms"]
        if a is None:
            continue
        others = [(s, e, i) for (s, e, i) in ev if i != r["request_id"]]
        r["inflight_at_arrival"] = sum(
            1 for (s, e, _i) in others if s is not None and s <= a and (e is None or e > a))
        r["arrivals_prev_200ms"] = sum(
            1 for (s, _e, _i) in others if s is not None and a - NEIGHBOR_MS <= s < a)
        r["finishes_prev_200ms"] = sum(
            1 for (_s, e, _i) in others if e is not None and a - NEIGHBOR_MS <= e < a)
        if f is None:
            r["arrivals_in_wait"] = r["finishes_in_wait"] = 0
            r["max_inflight_in_wait"] = r["inflight_at_arrival"]
        else:
            r["arrivals_in_wait"] = sum(
                1 for (s, _e, _i) in others if s is not None and a <= s <= f)
            r["finishes_in_wait"] = sum(
                1 for (_s, e, _i) in others if e is not None and a <= e <= f)
            hi = r["inflight_at_arrival"]
            for (s, _e, _i) in others:
                if s is not None and a <= s <= f:
                    hi += 1
            r["max_inflight_in_wait"] = hi
        if f is not None:
            r["srv_done_in_wait"] = sum(1 for (t, _k) in server_done_events if a <= t <= f)
            r["srv_done_prev_200ms"] = sum(
                1 for (t, _k) in server_done_events if a - NEIGHBOR_MS <= t < a)
            adm = [k for (t, k) in server_done_events if a <= t <= f and k is not None]
            adm0 = [k for (t, k) in server_done_events if t < a and k is not None]
            r["srv_admits_in_wait"] = (max(adm) - max(adm0)) if (adm and adm0) else None
        else:
            r["srv_done_in_wait"] = r["srv_done_prev_200ms"] = 0
            r["srv_admits_in_wait"] = None
        r["spike"] = bool(r["ttfa_ms"] is not None and r["ttfa_ms"] > budget_ms)
        r["attribution"] = _verdict(r)

def _verdict(r):
    """One short line saying what the spike COINCIDES with. Not a causal explanation: the
    coincidence itself, which is the hypothesis to confirm or falsify."""
    if not r.get("spike"):
        return ""
    if r.get("inflight_at_arrival", 0) == 0 and r.get("max_inflight_in_wait", 0) <= 1:
        return "machine IDLE -> NOT contention (prefill / model / cold start)"
    bits = []
    if r.get("arrivals_prev_200ms", 0):
        bits.append(f"arrival cluster (+{r['arrivals_prev_200ms']} in the previous 200 ms)")
    if r.get("finishes_in_wait", 0) or r.get("srv_done_in_wait", 0):
        n = max(r.get("finishes_in_wait", 0), r.get("srv_done_in_wait", 0))
        bits.append(f"waited for another to finish ({n} closed during the wait)")
    if r.get("srv_admits_in_wait"):
        bits.append(f"{r['srv_admits_in_wait']} others admitted while it waited")
    if not bits:
        bits.append(f"steady contention ({r.get('inflight_at_arrival', 0)} already in "
                    f"flight, no event inside the window)")
    return " · ".join(bits)

def pct(values, p):
    if not values:
        return float("nan")
    v = sorted(values)
    if len(v) == 1:
        return v[0]
    k = (len(v) - 1) * (p / 100.0)
    lo, hi = int(k), min(int(k) + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)

def mean_inflight(records, wall):
    """Requests in flight, time-averaged (integral of the overlaps / wall).
    It says whether the OFFERED load turned into the intended concurrency: with staggered
    arrivals "c=4" is a target, not a fact."""
    tot = 0.0
    hi = 0
    edges = []
    for r in records:
        if r["arrival_ms"] is None or r["end_ms"] is None:
            continue
        tot += (r["end_ms"] - r["arrival_ms"]) / 1000.0
        edges.append((r["arrival_ms"], 1))
        edges.append((r["end_ms"], -1))
    edges.sort()
    cur = 0
    for _t, d in edges:
        cur += d
        hi = max(hi, cur)
    return (tot / wall) if wall > 0 else 0.0, hi

def summarize(records, wall, conc, budget_ms, arrival, lam, seed, service_s):
    ok = [r for r in records if not r["error"] and r["status"] == 200]
    bad = [r for r in records if r not in ok]
    ttfa = [r["ttfa_ms"] for r in ok if r["ttfa_ms"] is not None]
    rtf = [r["rtf"] for r in ok if r["rtf"] is not None]
    audio = sum(r["audio_s"] for r in ok)
    over = [r for r in ok if r["ttfa_ms"] is not None and r["ttfa_ms"] > budget_ms]
    p50, p95 = pct(ttfa, 50), pct(ttfa, 95)
    mi, hi = mean_inflight(records, wall)
    return {
        "concurrency": conc, "requests": len(records), "ok": len(ok), "errors": len(bad),
        "wall_s": wall, "audio_s": audio, "throughput_Q": (audio / wall) if wall else 0.0,
        "ttfa_p50": p50, "ttfa_p95": p95, "ttfa_p99": pct(ttfa, 99),
        "ttfa_max": max(ttfa) if ttfa else float("nan"),
        "ttfa_mean": statistics.fmean(ttfa) if ttfa else float("nan"),
        "ttfa_budget_ms": budget_ms,
        "ttfa_over_budget": len(over),
        "ttfa_over_budget_pct": (100.0 * len(over) / len(ok)) if ok else float("nan"),
        "ttfa_stability": (p95 / p50) if (p50 and p50 == p50 and p50 > 0) else float("nan"),
        "ttfa_within_budget": bool(ok) and (p95 <= budget_ms),
        "rtf_p50": pct(rtf, 50), "rtf_p95": pct(rtf, 95),
        "arrival": arrival, "arrival_rate_hz": lam, "arrival_seed": seed,
        "service_s_estimate": service_s,
        "mean_inflight": mi, "max_inflight_observed": hi,
        "spikes": [{k: r[k] for k in ("request_id", "class", "ttfa_ms", "arrival_ms",
                                      "inflight_at_arrival", "arrivals_prev_200ms",
                                      "arrivals_in_wait", "finishes_in_wait",
                                      "srv_done_in_wait", "srv_admits_in_wait",
                                      "attribution")}
                   for r in sorted(over, key=lambda x: -x["ttfa_ms"])],
    }

def print_table(rows, budget_ms):
    base = next((s for s in rows if s["concurrency"] == 1), None)
    b95 = base["ttfa_p95"] if base else None
    print()
    print("=============== TTFA — LA METRICA DI PRIMA CLASSE")
    hdr = (f"{'conc':>5}{'req':>5}{'err':>4} | {'TTFA p50':>9}{'p95':>8}{'p99':>8}{'max':>8}"
           f" | {'degrad':>8}{'stab':>6}{'>budget':>9} | {'inflight':>9}{'Q':>7}"
           f"{'RTF p50':>9}{'p95':>7}")
    print(hdr)
    print("-" * len(hdr))
    for s in rows:
        deg = (s["ttfa_p95"] / b95) if (b95 and b95 > 0) else float("nan")
        over = f"{s['ttfa_over_budget']}/{s['ok']}"
        print(f"{s['concurrency']:>5}{s['requests']:>5}{s['errors']:>4} | "
              f"{s['ttfa_p50']:>9.0f}{s['ttfa_p95']:>8.0f}{s['ttfa_p99']:>8.0f}{s['ttfa_max']:>8.0f}"
              f" | {deg:>7.2f}x{s['ttfa_stability']:>6.2f}{over:>9} | "
              f"{s['mean_inflight']:>9.2f}{s['throughput_Q']:>7.2f}"
              f"{s['rtf_p50']:>9.2f}{s['rtf_p95']:>7.2f}")
    print()
    print(f"degrad   = TTFA p95(c) / TTFA p95(c=1). \"serving c users costs the worst one Nx\"")
    print(f"stab     = p95/p50. If it moves away from 1 there is a SPIKE even with a good median")
    print(f">budget  = requests with TTFA > {budget_ms:.0f} ms (the product target)")
    print(f"inflight = requests in flight, time-averaged: with staggered arrivals it says")
    print(f"           whether the offered load really became the intended concurrency")
    print(f"Q and RTF are SECONDARY. With staggered arrivals Q depends on the offered load and")
    print(f"is not a capacity measure: compare it only at equal --arrival")
    nmin = min((s["ok"] for s in rows), default=0)
    stag = any(s.get("arrival") != "all-at-once" for s in rows)
    if stag and nmin:
        cmax = max(s["concurrency"] for s in rows)
        reach = nmin * cmax / (nmin - 1 + cmax)
        print(f"NOTE with {nmin} requests per level the REACHABLE mean in-flight is")
        print(f"   N*c/(N-1+c) = {reach:.1f} at c={cmax}, not {cmax}: the level ends before")
        print(f"   reaching steady state. Fine for seeing the SHAPE of the degradation; to")
        print(f"   really measure c users in flight use --requests >= 8-16.")
    if nmin < 100:
        print(f"NOTE with {nmin} requests per level the p99 IS the maximum (N>=100 is needed for")
        print(f"   it to be a p99): read it as 'the worst one', not as a percentile.")
    if base and base.get("arrival") != "all-at-once" and base.get("max_inflight_observed", 0) > 1:
        print(f"NOTE the BASELINE is not clean: at c=1 up to {base['max_inflight_observed']}")
        print(f"   requests were seen in flight together — with Poisson, clusters happen even at")
        print(f"   low load. 'degrad' is therefore normalised on a c=1 that already contains")
        print(f"   contention: for a clean baseline use --arrival uniform, which by construction")
        print(f"   does not cluster.")

def print_verdict(rows, budget_ms):
    print()
    print(f"=============== VERDICT — TTFA p95 <= {budget_ms:.0f} ms (product target)")
    best = 0
    for s in sorted(rows, key=lambda r: r["concurrency"]):
        ok = s["ttfa_within_budget"] and s["errors"] == 0
        if ok:
            best = max(best, s["concurrency"])
        mark = "✅" if ok else "❌"
        why = []
        if s["errors"]:
            why.append(f"{s['errors']} errors")
        if not s["ttfa_within_budget"]:
            why.append(f"p95 {s['ttfa_p95']:.0f} ms > {budget_ms:.0f}")
        if s["ttfa_over_budget"]:
            why.append(f"{s['ttfa_over_budget']}/{s['ok']} requests over budget (max {s['ttfa_max']:.0f} ms)")
        print(f"  c={s['concurrency']:<3} {mark}  p95 {s['ttfa_p95']:>7.0f} ms"
              + (("   " + " · ".join(why)) if why else "   within budget, no spike"))
    print()
    if best:
        print(f"  -> HIGHEST CONCURRENCY HOLDING TTFA p95 <= {budget_ms:.0f} ms:  c = {best}")
    else:
        print(f"  -> HIGHEST CONCURRENCY HOLDING TTFA p95 <= {budget_ms:.0f} ms:  NONE (not even c=1)")
    print("     This is the product number: how many parallel conversations this machine")
    print("     serves without the user hearing silence. Q and RTF come after.")

def print_spikes(rows, budget_ms, limit=12):
    spikes = [(s["concurrency"], sp) for s in rows for sp in s["spikes"]]
    print()
    print(f"=============== SPIKE ATTRIBUTION (TTFA > {budget_ms:.0f} ms)")
    if not spikes:
        print("  no spike: no request over budget at any level.")
        return
    print("  The hypothesis to confirm or FALSIFY: \"the spike coincides with another")
    print("  request arriving or finishing\". The row that falsifies it is 'IDLE machine'.")
    print()
    hdr = (f"  {'c':>2}{'req':>5}{'class':>8}{'arrival':>9}{'TTFA':>8}{'flight':>6}"
           f"{'arr-200':>8}{'ends':>6}{'srv':>5}  coincidence")
    print(hdr)
    print("  " + "-" * (len(hdr) + 30))
    for c, sp in sorted(spikes, key=lambda x: -x[1]["ttfa_ms"])[:limit]:
        srv = sp["srv_done_in_wait"]
        print(f"  {c:>2}{sp['request_id']:>5}{sp['class']:>8}{sp['arrival_ms']:>8.0f}m"
              f"{sp['ttfa_ms']:>7.0f}m{sp['inflight_at_arrival']:>6}"
              f"{sp['arrivals_prev_200ms']:>8}{sp['finishes_in_wait']:>6}{srv:>5}  {sp['attribution']}")
    if len(spikes) > limit:
        print(f"  ... and {len(spikes) - limit} more (all in the CSV/JSON)")
    print()
    print("  flight  = requests already open when this one arrived (client side, exact)")
    print("  arr-200 = others that arrived in the previous 200 ms (cluster)")
    print("  ends    = others that CLOSED while this one waited for the first byte")
    print("  srv     = server `[BATCH] done` lines in the same window (--server-log)")

async def calibrate(args, host, port, texts):
    """One request alone on an idle machine -> S, the reference service time.

    It only sets the arrival pace when you do not pass one: interval = S/c makes "c"
    comparable across machines without hardwired numbers. It does not enter the
    statistics.

    The probe uses ONE text while the level uses a MIXTURE of lengths, so taking the
    probe duration as S costs an error proportional to the length ratio. S is therefore
    rescaled on the bank's mean length, and the pace is recalibrated on the c=1 level,
    the only one without contention.
    """
    mid = texts[len(texts) // 2]
    pl = {"text": mid[1], "speaker": args.speaker, "language": args.language,
          "temperature": args.temperature, "seed": args.seed}
    t0 = time.perf_counter()
    r = await one_request(host, port, args.path, pl, -1, mid[0], args.output_dir, False,
                          args.timeout, t0, 0.0, [], None)
    if r["error"] or not r["total_ms"]:
        return None, r
    mean_len = sum(len(t) for _c, t in texts) / len(texts)
    scale = mean_len / max(len(mid[1]), 1)
    return (r["total_ms"] / 1000.0) * scale, r

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    env = os.environ.get
    ap.add_argument("--url", default="http://localhost:8900")
    ap.add_argument("--path", default="/v1/tts/stream")
    ap.add_argument("--concurrency", default="1,2,4,8", help="sweep separato da virgole, es. 1,2,4,8")
    ap.add_argument("--requests", type=int, default=16, help="requests per concurrency level")
    ap.add_argument("--duration", type=float, default=0.0, help="secondi; scavalca --requests (soak)")
    ap.add_argument("--text-file", default=DEFAULT_TEXTS)
    ap.add_argument("--classes", default="", help="filtro sulle classi di testo, es. short,long")
    ap.add_argument("--speaker", default="ryan")
    ap.add_argument("--language", default="english")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--output-dir", default="/tmp/tts/loadtest")
    ap.add_argument("--no-save-audio", dest="save_audio", action="store_false")
    ap.add_argument("--csv", default="", help="write the per-request records here")
    ap.add_argument("--json", default="", help="write the per-level summary here")
    ap.add_argument("--arrival", default=env("QWEN_LT_ARRIVAL", "all-at-once"),
                    choices=["all-at-once", "poisson", "uniform"],
                    help="all-at-once = caso peggiore sincronizzato (default, storico); "
                         "poisson = inter-arrivi esponenziali; uniform = spaziatura fissa")
    ap.add_argument("--rate", type=float, default=float(env("QWEN_LT_RATE", "0") or 0),
                    help="requests/s for --arrival poisson (0 = calibrated: c/S)")
    ap.add_argument("--interval", type=float, default=float(env("QWEN_LT_INTERVAL", "0") or 0),
                    help="secondi fra arrivi per --arrival uniform (0 = calibrato: S/c)")
    ap.add_argument("--arrival-seed", type=int, default=int(env("QWEN_LT_ARRIVAL_SEED", "1234")),
                    help="arrival-process seed: FIXED and declared, so two runs see the "
                         "same arrival sequence")
    ap.add_argument("--service-s", type=float, default=float(env("QWEN_LT_SERVICE_S", "0") or 0),
                    help="duration of one request alone (s); 0 = measured with a probe")
    ap.add_argument("--max-inflight", type=int, default=0,
                    help="open-loop safety valve (0 = no cap). If it trips, "
                         "the wait is IN THE CLIENT, not the server, and is recorded separately")
    ap.add_argument("--ttfa-budget-ms", type=float,
                    default=float(env("QWEN_LT_TTFA_BUDGET_MS", "500")),
                    help="TTFA threshold for the verdict (default 500 ms)")
    ap.add_argument("--server-log", default=env("QWEN_LT_SERVER_LOG", ""),
                    help="server stderr, read live: correlates spikes with the "
                         "righe `[BATCH] done`")
    args = ap.parse_args()

    u = urlparse(args.url)
    host, port = u.hostname or "localhost", u.port or 80
    classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    texts = load_texts(args.text_file, classes or None)
    levels = [int(c) for c in args.concurrency.split(",") if c.strip()]

    print(f"target   {args.url}{args.path}")
    print(f"speaker  {args.speaker} · lingua {args.language} · temp {args.temperature}")
    print(f"testi    {len(texts)} da {os.path.basename(args.text_file)}"
          f"{' (' + ','.join(classes) + ')' if classes else ''}")
    mode = f"soak {args.duration:.0f}s" if args.duration else f"{args.requests} requests"
    print(f"sweep    concurrency {levels} · {mode} per level")
    print(f"budget   TTFA p95 <= {args.ttfa_budget_ms:.0f} ms  (product target)")

    service_s = args.service_s or None
    if args.arrival != "all-at-once" and not service_s and not (args.rate or args.interval):
        print("arrivals calibration: one request alone on an idle machine...", end=" ", flush=True)
        service_s, probe = asyncio.run(calibrate(args, host, port, texts))
        if service_s:
            print(f"S = {service_s:.2f} s  (audio {probe['audio_s']:.1f} s, "
                  f"TTFA {probe['ttfa_ms']:.0f} ms)")
        else:
            service_s = 10.0
            print(f"FALLITA ({probe['error']}) -> assumo S = {service_s:.0f} s")
    service_s = service_s or 10.0
    if args.arrival == "all-at-once":
        print("arrivi   all-at-once — tutte a t=0 (caso peggiore sincronizzato)")
    else:
        if args.arrival == "poisson":
            pace = (f"rate {args.rate:.3f} req/s (declared)" if args.rate
                     else f"rate = c/S = c/{service_s:.2f}s (calibrato)")
        else:
            pace = (f"interval {args.interval:.2f} s (declared)" if args.interval
                     else f"intervallo = S/c = {service_s:.2f}s/c (calibrato)")
        print(f"arrivi   {args.arrival} · {ritmo} · seme {args.arrival_seed} (FISSO: due corse "
              f"vedono gli stessi arrivi)")
    if args.server_log:
        print(f"log srv  {args.server_log}  (tail in diretta per l'attribuzione)")

    all_records, summaries = [], []
    for conc in levels:
        recs, wall, n, lam, dev = asyncio.run(run_level(args, host, port, texts, conc, service_s))
        for r in recs:
            r["concurrency"] = conc
        all_records.extend(recs)
        s = summarize(recs, wall, conc, args.ttfa_budget_ms, args.arrival, lam,
                      args.arrival_seed, service_s)
        s["path"] = args.path
        s["workload"] = "stream" if args.path.endswith("/stream") else "offline"
        summaries.append(s)
        if conc == 1 and args.arrival != "all-at-once" and not (args.rate or args.interval):
            tot = [r["total_ms"] / 1000.0 for r in recs if not r["error"] and r["total_ms"]]
            if tot:
                new_s = statistics.median(tot)
                if abs(new_s - service_s) / max(service_s, 1e-9) > 0.15:
                    print(f"      pace recalibrated on c=1 (no contention): "
                          f"S {service_s:.2f} s -> {new_s:.2f} s")
                service_s = new_s
        err = f" · {s['errors']} ERRORI" if s["errors"] else ""
        print(f"  c={conc:<3} {n:>3} req in {wall:6.1f}s · TTFA p50 {s['ttfa_p50']:.0f}ms "
              f"p95 {s['ttfa_p95']:.0f}ms max {s['ttfa_max']:.0f}ms · "
              f"{s['ttfa_over_budget']}/{s['ok']} over budget · Q {s['throughput_Q']:.2f}"
              f"{' · ' + str(len(dev)) + ' [BATCH] done' if dev else ''}{err}")

    print_table(summaries, args.ttfa_budget_ms)
    print_verdict(summaries, args.ttfa_budget_ms)
    print_spikes(summaries, args.ttfa_budget_ms)

    bad = [r for r in all_records if r["error"]]
    if bad:
        print(f"\n{len(bad)} failed requests; the first ones:")
        for r in bad[:5]:
            print(f"  c={r['concurrency']} req{r['request_id']} status={r['status']} {r['error']}")

    if args.csv and all_records:
        keys = sorted({k for r in all_records for k in r})
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in all_records:
                w.writerow({k: r.get(k, "") for k in keys})
        print(f"per-request records -> {args.csv}")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(summaries, f, indent=2, default=str)
        print(f"per-level summary -> {args.json}")
    if args.save_audio:
        print(f"audio                -> {args.output_dir}/c*/")

    return 1 if bad else 0

if __name__ == "__main__":
    sys.exit(main())
