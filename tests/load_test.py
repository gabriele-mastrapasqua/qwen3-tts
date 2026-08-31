#!/usr/bin/env python3
"""load_test.py — load bench for the streaming TTS server, with FIRST-AUDIO LATENCY as a
FIRST-CLASS METRIC.

The product objective, in words:

    serve 2-4 parallel requests with first audio <= 500 ms per user,
    without degrading and without spikes.

RTF and throughput are SECONDARY. A p99 that explodes is a quality incident in a call
centre even when the mean is excellent — so here the distribution matters more than the
mean, and spikes have to be ATTRIBUTED, not merely counted.

── WHY STAGGERED ARRIVALS (the reason this file changed) ────────────────────────

The historical mode (`--arrival all-at-once`) fires every request together at the start of
the level. That is the SYNCHRONISED WORST CASE, and it is useful: it stays the default and
stays comparable with every measurement taken so far. But it is not real traffic, and above
all it CANNOT DISTINGUISH two things we need to separate:

    first audio is late because the machine is saturated
    first audio is late because the request arrived while another was finishing

That distinction is exactly what has to be optimised (admission, priority for new arrivals,
split prefill). With everything arriving at t=0 every request is simultaneously "queued"
and "contended": no request ever arrives alone onto a loaded machine, so there is no
control.

Hence `--arrival poisson --rate <req/s>` and `--arrival uniform --interval <s>`: open loop,
arrivals do NOT wait for the server to free up. The generator's seed is fixed and DECLARED
(printed and written into the JSON) so two runs are comparable: without it a different p95
may be nothing but another draw of random numbers, and we would have credited an
optimisation for it.

── WHAT "c" MEANS WITH STAGGERED ARRIVALS ───────────────────────────────────────

In open loop `c` is not a semaphore. Without `--rate`/`--interval` the arrival pace is
CALIBRATED on the machine:

    mean inter-arrival interval = S / c        (S = the duration of ONE request alone)

By Little's law (L = lambda * W): if the server held c streams WITHOUT degrading, exactly c
would be in flight. So "c=4" means "an offered load such that a perfectly scaling server
would show 4 requests in flight". The number actually in flight is measured and printed
beside it: if it exceeds c, the server is NOT keeping up with the offered load — and that
is the result, not a defect of the measurement. S is measured with a probe request on an
idle machine (`--service-s` skips it if you already know it).

WITH STAGGERED ARRIVALS, THROUGHPUT IS NO LONGER A CAPACITY MEASURE: it depends on the
offered load, and if arrivals are sparse the machine sits idle between them. Throughput is
only meaningful compared AT EQUAL ARRIVAL MODE, never across modes. For pure capacity,
`all-at-once` remains.

── ATTRIBUTING THE SPIKES ───────────────────────────────────────────────────────

For every request above threshold, say WHAT WAS HAPPENING when it arrived:

  in flight on arrival     how many requests were already open (from the client: exact)
  arrivals in the last 200 ms  how many others started just before (from the client: exact)
  completions while waiting    how many others closed while this one waited
                           - from the client (last byte)  -> exact
                           - from the server ([BATCH] done) -> if you pass --server-log

The hypothesis to FALSIFY or confirm is "this spike coincides with the arrival or the
completion of another request". The control that makes it falsifiable is the case
`in flight on arrival = 0`: if a request arrives onto an IDLE machine and still pays 2 s of
first-audio latency, contention is not the cause — it is the prefill, the model, or a cold
start. Without that row the scheduler gets optimised for a problem that is not the
scheduler's.

`--server-log` reads the server's stderr LIVE (tail) and stamps a client timestamp on every
line: the engine prints `[BATCH] done #N (..., in-flight admitted=K)` without a clock, so
the time is ours, taken when the line appears. stderr in C is unbuffered, so the slack is
the file traversal, not a buffer. It is a correlation at ~ms, declared as such: admissions
are NOT logged, so the `admitted` counter is only visible when something finishes — for
arrivals the exact source remains the client.

── DEFINITIONS (all wall-clock, client side) ────────────────────────────────────
  ttfa       first audio byte     - request sent
  total      last audio byte      - request sent
  audio_s    PCM bytes / 2 / 24000 (int16 mono 24 kHz)
  RTF        total / audio_s                  (<1 = faster than realtime)
  Q          sum(audio_s) / wall              (audio-seconds per second)
  degradation  ttfa p95(c) / ttfa p95(c=1)    "serving 4 users costs the worst one 4.6x"
  stability    ttfa p95 / ttfa p50            if it separates, there is a spike even with
                                              a good median

Standard library only — asyncio plus a hand-written HTTP/1.1 client: the ARRIVAL TIME of
each chunk is exactly what is being measured, and a library's buffering would hide it.

Use:
  # start a server first:
  #   ./qwen_tts -d qwen3-tts-0.6b --serve 8900 --int8 --batch-size 4 -j 4
  tests/load_test.py --speaker ryan --concurrency 1,2,4 --requests 8
  tests/load_test.py --concurrency 1,2,3,4 --arrival poisson --requests 6 \
                     --server-log /tmp/tts/mini_bench/int8_server.log
  tests/load_test.py --concurrency 4 --arrival uniform --interval 1.5 --csv /tmp/l.csv

Environment variables (for callers reached through another script that does not pass the
flags — for example the mini-bench wrapper, which this file does not modify):
  QWEN_LT_ARRIVAL  QWEN_LT_RATE  QWEN_LT_INTERVAL  QWEN_LT_ARRIVAL_SEED
  QWEN_LT_TTFA_BUDGET_MS  QWEN_LT_SERVER_LOG  QWEN_LT_SERVICE_S
"""
import argparse, asyncio, csv, json, math, os, random, statistics, sys, time
from urllib.parse import urlparse

SR = 24000          # il server emette PCM grezzo int16 mono 24 kHz su /v1/tts/stream
BYTES_PER_SAMPLE = 2
# The DEFAULT text bank is English, and that is a phase decision rather than a taste one.
#
#   PHASE 1 (now): characterise the ENGINE on open Qwen3-TTS models, on rented servers.
#   Compute, bandwidth and disk are the same whatever the checkpoint is -- a finetune
#   only changes the tuning on top -- so open weights are enough to measure RTF, TTFA,
#   throughput, the concurrency ceiling and RSS, and there is no reason to ship private
#   weights to a machine you rent.
#
#   PHASE 2 (later): once the CPU side is understood, a finetune, its target language and a
#   language-identity gate come back. They are not discarded: a longer bank stays available
#   and is selected with --text-file.
#
# The wrong default does not fail, it silently measures open weights with another bank's
# texts. So the default is the one needed NOW, and the other is explicit.
DEFAULT_TEXTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "load_texts_en.txt")

# Co-occurrence window for attribution. 200 ms is not a magic number: it is the order of
# magnitude of one admission step + one frame (80 ms) + scheduling slack. Below it, real
# coincidences are lost; above it, everything "coincides with everything".
NEIGHBOR_MS = 200.0


# ── banco di testi ──────────────────────────────────────────────────────────
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
        sys.exit(f"{path}: nessun testo usabile (filtro={only_classes})")

    # ── INTERLEAVE BY CLASS, and it is not cosmetic ──────────────────────────────
    # The bank is written GROUPED by class (all the short ones, then medium, then long)
    # and the selection downstream is `texts[idx % len(texts)]`, i.e. round-robin from the
    # head. Measured consequence: a c=4 level with 4 requests emitted FOUR nearly identical
    # SHORT sentences, and over a full run of 8 the `long` class NEVER appeared — the
    # opposite of the "mixed lengths" design declared at the top of the bank.
    #
    # Why that is the worst case for exactly what we want to measure: with requests of
    # equal length every slot in the batch reaches EOS on the same frame, so the batch
    # fills and empties together and the RAGGED regime is never exercised — which is
    # precisely the regime of continuous batching in production, where finishing slots are
    # replaced while the others carry on.
    #
    # The interleave happens HERE rather than by reordering the file, because
    # `only_classes` may select a subset: the alternation has to hold over whatever
    # survives the filter. Class order = first appearance in the file (deterministic, no
    # randomness: the bank must stay reproducible between two runs).
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


# ── tail dello stderr del server ────────────────────────────────────────────
class ServerLogTail:
    """Read the server's stderr WHILE it runs and stamp each line as it appears.

    Needed because the engine prints `[BATCH] done #N` without a clock: the only way to
    know WHEN it happened, without touching the C, is to watch the file live. stderr in C
    is unbuffered, so the line appears at the instant of the event and the delay we add is
    the sampling period (20 ms), not a library buffer. Stated plainly: this is a
    correlation at tens of milliseconds, not tracing. It is enough to answer "did another
    request finish while this one waited".

    What the tail CANNOT see: ADMISSIONS. `sink_next_job` increments the counter without
    printing, so `admitted=K` is only readable attached to a `done`. For arrivals the
    exact source is the client, which generates them itself.
    """

    def __init__(self, path, poll_s=0.02):
        self.path = path
        self.poll_s = poll_s
        self.events = []          # [(t_rel_ms, riga)]
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


# ── una richiesta ───────────────────────────────────────────────────────────
async def one_request(host, port, path, payload, req_id, cls, out_dir, save_audio,
                      timeout, t0, arrival_sched_ms, inflight, gate):
    """Lancia una richiesta in streaming e cronometra a parte il PRIMO byte del corpo.

    `t0` e' l'origine dei tempi DEL LIVELLO: tutti gli istanti sono relativi a quello,
    perche' l'attribuzione dei picchi ha bisogno di una linea del tempo comune fra
    richieste diverse. Una durata da sola non permette di attribuire niente.
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
           # ── linea del tempo, relativa all'inizio del livello ──
           "arrival_sched_ms": arrival_sched_ms,   # quando DOVEVA arrivare
           "arrival_ms": None,                     # quando e' partita davvero
           "first_ms": None,                       # primo byte audio
           "end_ms": None,                         # ultimo byte
           "admit_wait_ms": 0.0}                   # attesa nel client (solo --max-inflight)

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

        # ── header ──────────────────────────────────────────────────────────
        raw = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), timeout)
        status_line = raw.split(b"\r\n", 1)[0].decode("latin1")
        rec["status"] = int(status_line.split()[1]) if len(status_line.split()) > 1 else 0
        chunked = b"transfer-encoding: chunked" in raw.lower()

        # ── corpo; il TTFA si timbra sul primo byte che e' davvero audio ─────
        chunks, first = [], None
        while True:
            if chunked:
                size_line = await asyncio.wait_for(reader.readuntil(b"\r\n"), timeout)
                n = int(size_line.strip().split(b";")[0], 16)
                if n == 0:
                    break
                data = await asyncio.wait_for(reader.readexactly(n), timeout)
                await reader.readexactly(2)          # CRLF finale
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

    except Exception as e:                        # noqa: BLE001 — ogni fallimento e' un dato
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
    """PCM int16 grezzo -> un vero file RIFF. Il server strema PCM senza header; un
    client che lo scrive dritto in .wav produce un file che nessun player apre."""
    import struct
    n = len(pcm)
    hdr = (b"RIFF" + struct.pack("<I", 36 + n) + b"WAVEfmt " + struct.pack("<IHHIIHH", 16, 1, 1, SR, SR * 2, 2, 16)
           + b"data" + struct.pack("<I", n))
    with open(path, "wb") as f:
        f.write(hdr)
        f.write(pcm)


# ── the arrival process ─────────────────────────────────────────────────────
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
            # inter-arrivi esponenziali = processo di Poisson (memoryless: e' il
            # modello standard di arrivi indipendenti, ed e' anche il caso in cui
            # i "grappoli" capitano da soli, senza doverli costruire a mano).
            t += -math.log(1.0 - rng.random()) / lam
        return out, lam
    raise SystemExit(f"modalita' di arrivo sconosciuta: {mode}")


# ── un livello di concorrenza ───────────────────────────────────────────────
async def run_level(args, host, port, texts, conc, service_s):
    out_dir = os.path.join(args.output_dir, f"c{conc}")
    if args.save_audio:
        os.makedirs(out_dir, exist_ok=True)

    tail = None
    if args.server_log:
        tail = ServerLogTail(args.server_log)
        if not tail.open_at_end():
            print(f"  ⚠️  --server-log {args.server_log}: non apribile, attribuzione solo lato client")
            tail = None

    def payload_for(idx):
        cls, txt = texts[idx % len(texts)]
        return cls, {"text": txt, "speaker": args.speaker, "language": args.language,
                     "temperature": args.temperature, "seed": args.seed + idx}

    inflight = []                     # ids of the requests open at this instant
    gate = asyncio.Semaphore(args.max_inflight) if args.max_inflight > 0 else None
    records = []
    t0 = time.perf_counter()
    tail_task = asyncio.create_task(tail.run(t0)) if tail else None

    if args.arrival == "all-at-once" and args.duration:
        # soak storico: si tengono `conc` richieste in volo finche' scade il tempo.
        # E' closed loop per costruzione, e resta com'era: e' il regime di tenuta.
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
        # THE SYNCHRONISED WORST CASE, and it stays the default: all at t=0, `conc` in
        # flight, the next starting when one finishes. It is the regime every historical
        # number in this project was taken under — changing it quietly would make all of
        # them incomparable.
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
        # ── ARRIVI SCAGLIONATI, open loop: nessuno trattiene le richieste. E' l'unico
        # modo di avere richieste che arrivano su una macchina GIA' carica ma non
        # sincronizzate con le altre — cioe' di poter attribuire un picco.
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


# ── spike attribution ───────────────────────────────────────────────────────
def attribute(records, server_done_events, budget_ms):
    """For each request: what was happening in the server when it arrived, and while it
    waited for the first byte.

    The client-side columns (in flight, nearby arrivals, completions during the wait) are
    EXACT: we generate the arrivals and we see the last byte. The ones from the server log
    are an independent confirmation, timestamped by the tail.

    Il controllo che rende l'ipotesi falsificabile e' `inflight_at_arrival == 0`:
    un picco su macchina scarica NON e' contesa, ed e' l'unico caso in cui si puo'
    escludere lo scheduler senza ricorrere ad altri strumenti.
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
            # in volo massimo durante l'attesa: il picco di contesa che ha davvero
            # visto, non quello all'istante dell'arrivo (che puo' essere basso).
            hi = r["inflight_at_arrival"]
            for (s, _e, _i) in others:
                if s is not None and a <= s <= f:
                    hi += 1
            r["max_inflight_in_wait"] = hi
        # conferma indipendente dal server: quante `[BATCH] done` sono state stampate
        # mentre questa richiesta aspettava il primo byte.
        if f is not None:
            r["srv_done_in_wait"] = sum(1 for (t, _k) in server_done_events if a <= t <= f)
            r["srv_done_prev_200ms"] = sum(
                1 for (t, _k) in server_done_events if a - NEIGHBOR_MS <= t < a)
            adm = [k for (t, k) in server_done_events if a <= t <= f and k is not None]
            adm0 = [k for (t, k) in server_done_events if t < a and k is not None]
            # the `admitted` counter is cumulative and we only see it attached to a
            # `done`: if it grew within the window, somebody else WAS admitted. With no
            # `done` inside the window nothing can be said -> None, which the report reads
            # as "n/a", not as "zero".
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


# ── statistiche ─────────────────────────────────────────────────────────────
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
        # ── TTFA: la metrica di prima classe ──
        "ttfa_p50": p50, "ttfa_p95": p95, "ttfa_p99": pct(ttfa, 99),
        "ttfa_max": max(ttfa) if ttfa else float("nan"),
        "ttfa_mean": statistics.fmean(ttfa) if ttfa else float("nan"),
        "ttfa_budget_ms": budget_ms,
        "ttfa_over_budget": len(over),
        "ttfa_over_budget_pct": (100.0 * len(over) / len(ok)) if ok else float("nan"),
        "ttfa_stability": (p95 / p50) if (p50 and p50 == p50 and p50 > 0) else float("nan"),
        "ttfa_within_budget": bool(ok) and (p95 <= budget_ms),
        # ── secondarie ──
        "rtf_p50": pct(rtf, 50), "rtf_p95": pct(rtf, 95),
        # ── how the load arrived (without this the row is not reproducible) ──
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


# ── stampa ──────────────────────────────────────────────────────────────────
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
    print(f"=============== VERDETTO — TTFA p95 <= {budget_ms:.0f} ms (PLAN 0.nonies, bersaglio di prodotto)")
    best = 0
    for s in sorted(rows, key=lambda r: r["concurrency"]):
        ok = s["ttfa_within_budget"] and s["errors"] == 0
        if ok:
            best = max(best, s["concurrency"])
        mark = "✅" if ok else "❌"
        why = []
        if s["errors"]:
            why.append(f"{s['errors']} errori")
        if not s["ttfa_within_budget"]:
            why.append(f"p95 {s['ttfa_p95']:.0f} ms > {budget_ms:.0f}")
        if s["ttfa_over_budget"]:
            why.append(f"{s['ttfa_over_budget']}/{s['ok']} richieste sopra soglia (max {s['ttfa_max']:.0f} ms)")
        print(f"  c={s['concurrency']:<3} {mark}  p95 {s['ttfa_p95']:>7.0f} ms"
              + (("   " + " · ".join(why)) if why else "   dentro soglia, nessun picco"))
    print()
    if best:
        print(f"  -> MASSIMA CONCORRENZA CHE REGGE TTFA p95 <= {budget_ms:.0f} ms:  c = {best}")
    else:
        print(f"  -> MASSIMA CONCORRENZA CHE REGGE TTFA p95 <= {budget_ms:.0f} ms:  NESSUNA (nemmeno c=1)")
    print("     E' il numero di prodotto: quante conversazioni parallele questa macchina")
    print("     serve senza che l'utente senta il silenzio. Q e RTF vengono dopo.")


def print_spikes(rows, budget_ms, limit=12):
    spikes = [(s["concurrency"], sp) for s in rows for sp in s["spikes"]]
    print()
    print(f"=============== ATTRIBUZIONE DEI PICCHI (TTFA > {budget_ms:.0f} ms)")
    if not spikes:
        print("  nessun picco: nessuna richiesta sopra soglia a nessun livello.")
        return
    print("  L'ipotesi da confermare o FALSIFICARE: \"il picco coincide con l'arrivo o la fine")
    print("  di un'altra richiesta\". La riga che la falsifica e' 'macchina SCARICA'.")
    print()
    hdr = (f"  {'c':>2}{'req':>5}{'classe':>8}{'arrivo':>9}{'TTFA':>8}{'volo':>6}"
           f"{'arr-200':>8}{'fini':>6}{'srv':>5}  coincidenza")
    print(hdr)
    print("  " + "-" * (len(hdr) + 30))
    for c, sp in sorted(spikes, key=lambda x: -x[1]["ttfa_ms"])[:limit]:
        srv = sp["srv_done_in_wait"]
        print(f"  {c:>2}{sp['request_id']:>5}{sp['class']:>8}{sp['arrival_ms']:>8.0f}m"
              f"{sp['ttfa_ms']:>7.0f}m{sp['inflight_at_arrival']:>6}"
              f"{sp['arrivals_prev_200ms']:>8}{sp['finishes_in_wait']:>6}{srv:>5}  {sp['attribution']}")
    if len(spikes) > limit:
        print(f"  ... e altri {len(spikes) - limit} (tutti nel CSV/JSON)")
    print()
    print("  volo    = richieste gia' aperte quando questa e' arrivata (lato client, esatto)")
    print("  arr-200 = altre arrivate nei 200 ms precedenti (grappolo)")
    print("  fini    = altre CHIUSE mentre questa aspettava il primo byte (lato client)")
    print("  srv     = righe `[BATCH] done` del server nella stessa finestra (--server-log)")


# ── calibrazione del ritmo di arrivo ────────────────────────────────────────
async def calibrate(args, host, port, texts):
    """Una richiesta SOLA su macchina scarica -> S, la durata di servizio di
    riferimento. Serve solo a scegliere il ritmo degli arrivi quando non lo passi:
    intervallo = S/c rende "c" confrontabile fra macchine diverse senza numeri
    cablati (su M1 e su un C3 la stessa `--rate` significherebbe carichi diversi).
    NON entra nelle statistiche: e' fuori dalla misura, per costruzione.

    ⚠️ La sonda usa UN testo, ma il livello ne usa una MISCELA di lunghezze diverse:
    prendere la durata della sonda come S paga un errore proporzionale al rapporto
    fra le lunghezze, e con banco misto e' un fattore 3-5 (misurato: sonda 17 s
    contro media reale ~3 s -> arrivi cinque volte troppo radi, e la concorrenza
    offerta non si realizza mai). Quindi si RISCALA sulla lunghezza media del banco,
    che e' il predittore piu' semplice della durata dell'audio. Resta una stima: il
    ritmo vero viene poi RICALIBRATO sul livello c=1, che e' l'unico senza contesa.
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
    ap.add_argument("--requests", type=int, default=16, help="richieste per livello di concorrenza")
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
    ap.add_argument("--csv", default="", help="scrive qui i record per richiesta")
    ap.add_argument("--json", default="", help="scrive qui il riepilogo per livello")
    # ── arrivi ──
    ap.add_argument("--arrival", default=env("QWEN_LT_ARRIVAL", "all-at-once"),
                    choices=["all-at-once", "poisson", "uniform"],
                    help="all-at-once = caso peggiore sincronizzato (default, storico); "
                         "poisson = inter-arrivi esponenziali; uniform = spaziatura fissa")
    ap.add_argument("--rate", type=float, default=float(env("QWEN_LT_RATE", "0") or 0),
                    help="richieste/s per --arrival poisson (0 = calibrato: c/S)")
    ap.add_argument("--interval", type=float, default=float(env("QWEN_LT_INTERVAL", "0") or 0),
                    help="secondi fra arrivi per --arrival uniform (0 = calibrato: S/c)")
    ap.add_argument("--arrival-seed", type=int, default=int(env("QWEN_LT_ARRIVAL_SEED", "1234")),
                    help="seme del processo di arrivo: FISSO e dichiarato, cosi' due corse "
                         "vedono la stessa sequenza di arrivi")
    ap.add_argument("--service-s", type=float, default=float(env("QWEN_LT_SERVICE_S", "0") or 0),
                    help="durata di una richiesta da sola (s); 0 = misurata con una sonda")
    ap.add_argument("--max-inflight", type=int, default=0,
                    help="valvola di sicurezza in open loop (0 = nessun tetto). ⚠️ se scatta, "
                         "l'attesa e' NEL CLIENT e non e' del server: viene registrata a parte")
    # ── soglia di prodotto ──
    ap.add_argument("--ttfa-budget-ms", type=float,
                    default=float(env("QWEN_LT_TTFA_BUDGET_MS", "500")),
                    help="soglia di TTFA per il verdetto (default 500 ms, PLAN 0.nonies)")
    ap.add_argument("--server-log", default=env("QWEN_LT_SERVER_LOG", ""),
                    help="stderr del server, letto in diretta: correla i picchi con le "
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
    mode = f"soak {args.duration:.0f}s" if args.duration else f"{args.requests} richieste"
    print(f"sweep    concorrenza {levels} · {mode} per livello")
    print(f"soglia   TTFA p95 <= {args.ttfa_budget_ms:.0f} ms  (bersaglio di prodotto)")

    # ── ritmo degli arrivi: calibrato una volta sola, a macchina scarica ──
    service_s = args.service_s or None
    if args.arrival != "all-at-once" and not service_s and not (args.rate or args.interval):
        print("arrivi   calibrazione: una richiesta sola a macchina scarica...", end=" ", flush=True)
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
            ritmo = (f"rate {args.rate:.3f} req/s (dichiarato)" if args.rate
                     else f"rate = c/S = c/{service_s:.2f}s (calibrato)")
        else:
            ritmo = (f"intervallo {args.interval:.2f} s (dichiarato)" if args.interval
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
        # THE ENDPOINT BELONGS IN THE ROW, not only on the command line. /v1/tts/stream
        # and /v1/tts produce the same audio but NOT the same metric: on the stream, first
        # audio is the first chunk (the silence the user actually lives through); on
        # /v1/tts the first byte arrives once generation has finished, so "first audio"
        # there equals total latency and is not a product constraint. Without this field
        # an aggregator combining two runs has no way to notice, and prints a false
        # verdict.
        s["path"] = args.path
        s["workload"] = "stream" if args.path.endswith("/stream") else "offline"
        summaries.append(s)
        # ── the real pace comes from the c=1 level: it is measured on the actual MIX of
        # texts and on this machine, while the probe is a single text. The probe is there
        # to start from; from here on the later levels are calibrated on a measured number.
        # The MEDIAN is used rather than the mean: with Poisson arrivals some clustering
        # happens even at c=1, and one request that waited in the queue would inflate the
        # mean -> arrivals too sparse at the levels after it, which is the very error being
        # corrected, again. ──
        if conc == 1 and args.arrival != "all-at-once" and not (args.rate or args.interval):
            tot = [r["total_ms"] / 1000.0 for r in recs if not r["error"] and r["total_ms"]]
            if tot:
                new_s = statistics.median(tot)
                if abs(new_s - service_s) / max(service_s, 1e-9) > 0.15:
                    print(f"      ritmo ricalibrato su c=1 (senza contesa): "
                          f"S {service_s:.2f} s -> {new_s:.2f} s")
                service_s = new_s
        err = f" · {s['errors']} ERRORI" if s["errors"] else ""
        print(f"  c={conc:<3} {n:>3} req in {wall:6.1f}s · TTFA p50 {s['ttfa_p50']:.0f}ms "
              f"p95 {s['ttfa_p95']:.0f}ms max {s['ttfa_max']:.0f}ms · "
              f"{s['ttfa_over_budget']}/{s['ok']} sopra soglia · Q {s['throughput_Q']:.2f}"
              f"{' · ' + str(len(dev)) + ' [BATCH] done' if dev else ''}{err}")

    print_table(summaries, args.ttfa_budget_ms)
    print_verdict(summaries, args.ttfa_budget_ms)
    print_spikes(summaries, args.ttfa_budget_ms)

    bad = [r for r in all_records if r["error"]]
    if bad:
        print(f"\n{len(bad)} richieste fallite; le prime:")
        for r in bad[:5]:
            print(f"  c={r['concurrency']} req{r['request_id']} status={r['status']} {r['error']}")

    if args.csv and all_records:
        keys = sorted({k for r in all_records for k in r})
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in all_records:
                w.writerow({k: r.get(k, "") for k in keys})
        print(f"record per richiesta -> {args.csv}")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(summaries, f, indent=2, default=str)
        print(f"riepilogo per livello -> {args.json}")
    if args.save_audio:
        print(f"audio                -> {args.output_dir}/c*/")

    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
