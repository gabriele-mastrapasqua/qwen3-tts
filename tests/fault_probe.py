#!/usr/bin/env python3
"""Provoked failures against a live qwen_tts server, each with its invariants.

Python stdlib only. Every case breaks ONE thing on purpose and then checks what the
server did about it from the server's own /v1/health, never from the client's opinion.

The central measure is MODEL WORK AFTER THE DISCONNECT: `frames_generated` (one per
Talker frame, 12.5 per second of audio, every request and path together) read just
before the client goes away and again once the server is quiet. A zombie inference --
the model generating for a client that no longer exists -- shows up as the rest of the
utterance in that difference.

Every zombie case first proves it is DISCRIMINATING: the same request, run to the end
once as a reference, must still have had at least --min-pending-s of audio to generate
at the instant of the disconnect. A fast machine that finished before the client left
cannot pass the case vacuously.

Zombie cases (run by `zombie`):

  rst-mid          /v1/tts/stream, batched slot: read ~1.5 s of audio, then a TCP RST
  fin-mid          the same, then close(): FIN, the socket gone
  rst-first        RST as soon as the first audio bytes arrive
  rst-mid-single   the single-job path (an `instruct` request runs on the clone
  fin-mid-single   worker, handle_tts_stream): RST / FIN mid-stream
  rst-wav          /v1/tts (WAV, no bytes until the end): RST during generation

Semantics cases (run by `semantics`):

  half-close-ok    shutdown(SHUT_WR) right after the request, then read everything: a
                   half-close is legal HTTP/1.1, so the stream must complete, whole
  stopped-reader   read ~1 s, then stop reading with the socket open (small receive
                   buffer): the blocked write must time out and end the request as gone
                   instead of holding the writer forever

Service cases (run by `service`, on the same server):

  neighbours       a healthy stream runs while three others die around it (RST, FIN, RST on
                   the first audio): its audio must equal its unloaded reference
  abort-loop       N mixed aborts (stream / WAV / single-job, RST / FIN / first audio): the
                   books balance, every abort is one client_gone, RSS does not grow

Books case (`books`, on a server started with QWEN_MAX_REQUEST_S=3, --batch-size 2,
--max-queue 1 and --metrics-port): a deterministic mixed workload -- completed,
disconnected, timed out, refused because the queue is full, invalid -- must move the
outcome counters by exactly the workload, in /v1/health AND in /metrics, and balance.

Every case also checks the session books: the request ends in exactly one outcome
(sessions +1, that outcome +1, nothing else), active returns to 0, `balanced` holds.

Usage:
  python3 tests/fault_probe.py --port P zombie
Exit 0 when every case holds, 1 otherwise; one OK/FAIL line per invariant.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import socket
import struct
import sys
import time
import urllib.request

FPS = 12.5            # Talker frames per second of audio
SR = 24000            # output sample rate, s16le mono

LONG_TEXT = (
    "The old lighthouse keeper climbed the spiral stairs every evening, counting each "
    "of the one hundred and twelve steps as his father had taught him. At the top he "
    "polished the great lens, trimmed the wick, and watched the last ships of the day "
    "turn toward the harbour. Some nights the fog rolled in so thick that the beam "
    "seemed to stop a few metres from the glass, and he would sit by the window with a "
    "cup of tea, listening to the foghorn and thinking about the sailors out there. "
    "In the morning he wrote everything in the logbook: the weather, the ships he had "
    "seen, and the small birds that rested on the railing before crossing the sea."
)
SINGLE_INSTRUCT = "Speak calmly, at a steady and even pace."

# ------------------------------------------------------------------ server state

def health(port: int) -> dict:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/health", timeout=10) as r:
        return json.loads(r.read())


def frames(port: int) -> int:
    v = health(port).get("frames_generated")
    if v is None:
        raise SystemExit("FAIL: /v1/health has no frames_generated -- this binary cannot "
                         "measure model work, refusing to report a zombie number")
    return int(v)


OUTCOMES = ("completed", "client_gone", "timeout", "rejected", "failed")


def books(port: int) -> dict:
    b = health(port).get("books")
    if b is None:
        raise SystemExit("FAIL: /v1/health has no books -- this binary keeps no session books")
    return b


def wait_books_idle(port: int, timeout: float = 60.0) -> dict | None:
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        b = books(port)
        if b.get("active") == 0:
            return b
        time.sleep(0.2)
    return None


def books_delta(b0: dict, b1: dict) -> dict:
    return {k: b1[k] - b0[k] for k in ("sessions",) + OUTCOMES}


def check_books(name: str, b0: dict, b1: dict | None, want: dict):
    """Exactly `want` moved (sessions = its sum), nothing else; idle and balanced."""
    if b1 is None:
        check(False, f"{name}: books idle (active back to 0)", "a session never ended")
        return
    d = books_delta(b0, b1)
    exp = {k: want.get(k, 0) for k in OUTCOMES}
    exp["sessions"] = sum(exp.values())
    moved = " ".join(f"{k}+{v}" for k, v in d.items() if v)
    check(d == exp and b1.get("balanced") is True and b1.get("anomalies") == 0,
          f"{name}: books -- one session per request, one outcome each, balanced",
          f"moved {moved or 'nothing'}; wanted " +
          " ".join(f"{k}+{v}" for k, v in exp.items() if v) +
          f"; balanced={b1.get('balanced')} anomalies={b1.get('anomalies')}")


def wait_quiet(port: int, stable_s: float = 1.2, timeout: float = 240.0):
    """Waits until frames_generated stops moving for `stable_s`. Returns the final
    count, or None on timeout. A zombie keeps the counter moving until its EOS, so
    the timeout must cover a whole utterance."""
    t0 = time.monotonic()
    last, since = frames(port), time.monotonic()
    while time.monotonic() - t0 < timeout:
        time.sleep(0.3)
        cur = frames(port)
        if cur != last:
            last, since = cur, time.monotonic()
        elif time.monotonic() - since >= stable_s:
            return cur
    return None

# ------------------------------------------------------------------ one client

class Stream:
    """One HTTP request on a raw socket, so a case can choose exactly how it dies."""

    def __init__(self, port: int, path: str, body: dict, timeout: float = 30.0,
                 rcvbuf: int = 0):
        data = json.dumps(body).encode()
        req = (f"POST {path} HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
               f"Content-Length: {len(data)}\r\nConnection: close\r\n\r\n").encode() + data
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        if rcvbuf:
            self.s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, rcvbuf)
        self.s.settimeout(timeout)
        self.s.connect(("127.0.0.1", port))
        self.s.sendall(req)
        self.buf = b""
        self.status = None
        self.headers = b""
        self.chunked = False
        self.audio = 0            # PCM bytes received (stream) or body bytes (WAV)
        self.sha = hashlib.sha256()
        self.eof = False
        self.complete = False     # terminal chunk seen (stream) / full body (WAV)
        self.clen = None
        self.body = b""           # first bytes of a non-chunked body (error messages)

    def _feed(self, ch: bytes):
        self.buf += ch
        if self.status is None:
            i = self.buf.find(b"\r\n\r\n")
            if i < 0:
                return
            self.headers = self.buf[:i]
            self.buf = self.buf[i + 4:]
            m = re.match(rb"HTTP/1\.[01] (\d+)", self.headers)
            self.status = int(m.group(1)) if m else -1
            self.chunked = b"chunked" in self.headers.lower()
            m = re.search(rb"(?i)content-length:\s*(\d+)", self.headers)
            self.clen = int(m.group(1)) if m else None
        if self.chunked:
            while True:
                m = re.match(rb"([0-9a-fA-F]+)\r\n", self.buf)
                if not m:
                    return
                sz = int(m.group(1), 16)
                need = m.end() + sz + 2
                if len(self.buf) < need:
                    return
                pay = self.buf[m.end():m.end() + sz]
                self.buf = self.buf[need:]
                if sz == 0:
                    self.complete = True
                    return
                self.sha.update(pay)
                self.audio += sz
        else:
            self.sha.update(self.buf)
            self.audio += len(self.buf)
            if len(self.body) < 4096:
                self.body += self.buf[:4096 - len(self.body)]
            self.buf = b""
            if self.clen is not None and self.audio >= self.clen:
                self.complete = True

    def read_some(self, deadline: float) -> bool:
        self.s.settimeout(max(0.01, min(0.2, deadline - time.monotonic())))
        try:
            ch = self.s.recv(1 << 16)
        except socket.timeout:
            return True
        except OSError:
            self.eof = True
            return False
        if not ch:
            self.eof = True
            return False
        self._feed(ch)
        return True

    def read_until_audio(self, seconds: float, timeout: float = 120.0) -> bool:
        want = int(seconds * SR * 2)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and not self.complete:
            if self.audio >= want and self.status is not None:
                return True
            if not self.read_some(deadline):
                break
        return self.audio >= want

    def read_all(self, timeout: float = 300.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and not self.complete:
            if not self.read_some(deadline):
                break
        self.close()
        return self

    def audio_s(self) -> float:
        return self.audio / 2 / SR

    def rst(self):
        """close() with SO_LINGER 0: the kernel sends a RST."""
        try:
            self.s.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
        except OSError:
            pass
        self.close()

    def fin(self):
        """A client that goes away politely: FIN, then the socket is closed. Whatever
        the server writes after this is answered with a RST by our kernel."""
        try:
            self.s.shutdown(socket.SHUT_WR)
        except OSError:
            pass
        self.close()

    def close(self):
        try:
            self.s.close()
        except OSError:
            pass

# ------------------------------------------------------------------ reporting

FAILS: list[str] = []


def check(ok: bool, name: str, detail: str = ""):
    print(f"  {'OK  ' if ok else 'FAIL'} {name}{('  ' + detail) if detail else ''}", flush=True)
    if not ok:
        FAILS.append(name)


def body_for(kind: str, seed: int) -> dict:
    b = {"text": LONG_TEXT, "speaker": "ryan", "language": "English",
         "seed": seed, "temperature": 0.0}
    if kind == "single":
        b["instruct"] = SINGLE_INSTRUCT
    return b

# ------------------------------------------------------------------ zombie cases

REF: dict[str, int] = {}
REF_AUDIO: dict[str, tuple] = {}


def reference(port: int, kind: str, seed: int) -> int:
    """Frames the whole utterance takes, measured on this binary with no disconnect."""
    key = f"{kind}:{seed}"
    if key in REF:
        return REF[key]
    if wait_quiet(port) is None:
        raise SystemExit("FAIL: server never became quiet before the reference")
    f0 = frames(port)
    path = "/v1/tts/stream"
    st = Stream(port, path, body_for(kind, seed)).read_all()
    f1 = wait_quiet(port)
    if st.status != 200 or not st.complete or f1 is None:
        raise SystemExit(f"FAIL: reference {kind} did not complete (status {st.status}, "
                         f"complete {st.complete})")
    REF[key] = f1 - f0
    REF_AUDIO[key] = (st.audio, st.sha.hexdigest()[:16])
    print(f"  ref  {kind:6s} seed {seed}: {REF[key]} frames = {REF[key] / FPS:.1f} s, "
          f"{st.audio_s():.1f} s delivered", flush=True)
    return REF[key]


def zombie_case(port: int, name: str, kind: str, path: str, how: str,
                after_audio_s: float, bound_frames: int, min_pending_s: float, seed: int):
    print(f"[{name}]", flush=True)
    ref = reference(port, kind, seed)
    if wait_quiet(port) is None:
        check(False, f"{name}: server quiet before the case")
        return
    b0 = books(port)
    f0 = frames(port)
    st = Stream(port, path, body_for(kind, seed))
    if path.endswith("/stream"):
        if after_audio_s <= 0:
            got = st.read_until_audio(1.0 / SR)      # the first bytes of audio
        else:
            got = st.read_until_audio(after_audio_s)
        if not got:
            check(False, f"{name}: stream delivered {after_audio_s:.1f} s before the disconnect",
                  f"status {st.status}, {st.audio_s():.2f} s")
            st.close()
            return
    else:
        # A WAV sends nothing until the end: wait until the server is visibly generating.
        want = f0 + int(after_audio_s * FPS)
        t_end = time.monotonic() + 120
        while frames(port) < want and time.monotonic() < t_end:
            time.sleep(0.05)
    f_at = frames(port)
    (st.rst if how == "rst" else st.fin)()
    f_end = wait_quiet(port)
    if f_end is None:
        check(False, f"{name}: server quiet after the disconnect", "still generating after 240 s")
        return
    gen_at = f_at - f0
    pending = ref - gen_at
    after = f_end - f_at
    check(pending >= min_pending_s * FPS,
          f"{name}: discriminating (work still pending at the disconnect)",
          f"{pending / FPS:.1f} s pending of {ref / FPS:.1f} s, need >= {min_pending_s:.0f} s")
    check(after <= bound_frames,
          f"{name}: model work after the disconnect",
          f"{after} frames = {after / FPS:.2f} s (bound {bound_frames} = "
          f"{bound_frames / FPS:.2f} s; {st.audio_s():.2f} s had been delivered)")
    check_books(name, b0, wait_books_idle(port), {"client_gone": 1})


def case_zombie(a):
    rst_bound = 3                  # the in-flight step, plus the health read racing it
    fin_bound = 2 * 10 + 3         # a FIN is seen at the next write: a chunk, plus the
                                   # decoder lagging the Talker by at most one more
    P = a.min_pending_s
    zombie_case(a.port, "rst-mid", "batch", "/v1/tts/stream", "rst", 1.5, rst_bound, P, 4242)
    zombie_case(a.port, "fin-mid", "batch", "/v1/tts/stream", "fin", 1.5, fin_bound, P, 4242)
    zombie_case(a.port, "rst-first", "batch", "/v1/tts/stream", "rst", 0.0, rst_bound, P, 4242)
    zombie_case(a.port, "rst-mid-single", "single", "/v1/tts/stream", "rst", 1.5, rst_bound, P, 4343)
    zombie_case(a.port, "fin-mid-single", "single", "/v1/tts/stream", "fin", 1.5, fin_bound, P, 4343)
    zombie_case(a.port, "rst-wav", "batch", "/v1/tts", "rst", 2.0, rst_bound, P, 4242)


def running(port: int) -> int:
    h = health(port)
    return int(h.get("num_requests_running", 0)) + int(h.get("num_requests_waiting", 0))


def wait_idle(port: int, timeout: float) -> float | None:
    """Seconds until the server holds no request (running + waiting == 0), or None."""
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if running(port) == 0:
            return time.monotonic() - t0
        time.sleep(0.2)
    return None


def case_semantics(a):
    seed = 4242
    print("[half-close-ok]", flush=True)
    ref = reference(a.port, "batch", seed)
    ref_bytes, ref_sha = REF_AUDIO[f"batch:{seed}"]
    b0 = books(a.port)
    st = Stream(a.port, "/v1/tts/stream", body_for("batch", seed))
    st.s.shutdown(socket.SHUT_WR)
    st.read_all()
    check_books("half-close-ok", b0, wait_books_idle(a.port), {"completed": 1})
    check(st.status == 200 and st.complete,
          "half-close-ok: a client that shut its write side still gets the whole stream",
          f"status {st.status}, complete {st.complete}, {st.audio_s():.1f} s")
    check(st.audio == ref_bytes,
          "half-close-ok: same audio length as the reference",
          f"{st.audio} vs {ref_bytes} bytes; sha {'identical' if st.sha.hexdigest()[:16] == ref_sha else 'differs'}")

    print("[stopped-reader]", flush=True)
    if wait_quiet(a.port) is None:
        check(False, "stopped-reader: server quiet before the case")
        return
    b0 = books(a.port)
    f0 = frames(a.port)
    st = Stream(a.port, "/v1/tts/stream", body_for("batch", seed), rcvbuf=4096)
    st.read_until_audio(1.0)
    f_stop = frames(a.port)
    t_stop = time.monotonic()
    ended = wait_idle(a.port, timeout=a.send_timeout_s + 40)
    f_end = frames(a.port)
    st.close()
    wait_quiet(a.port)
    check(ended is not None,
          "stopped-reader: the request ends while the client still holds the socket",
          f"after {ended:.1f} s" if ended is not None else "still held after "
          f"{a.send_timeout_s + 40:.0f} s")
    after = f_end - f_stop
    pending = ref - (f_stop - f0)
    check(ended is None or after < pending,
          "stopped-reader: generation stopped before the end of the utterance",
          f"{after} frames after the reader stopped, {pending} were pending")
    check_books("stopped-reader", b0, wait_books_idle(a.port), {"client_gone": 1})


NEIGHBOUR_TEXT = ("Thank you for calling. Your order left the warehouse this morning and "
                  "should arrive on Thursday before noon; you will receive a message with "
                  "the tracking number as soon as the courier scans the parcel.")


def rss_kb(pid: int) -> int:
    import subprocess
    out = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True)
    return int(out.stdout.strip() or 0)


def aborter(port: int, path: str, kind: str, how: str, seed: int, after_s: float,
            body: dict | None = None):
    st = Stream(port, path, body if body is not None else body_for(kind, seed))
    if path.endswith("/stream"):
        st.read_until_audio(after_s if after_s > 0 else 1.0 / SR)
    else:
        time.sleep(after_s)
    (st.rst if how == "rst" else st.fin)()


def case_service(a):
    import threading
    nb = {"text": NEIGHBOUR_TEXT, "speaker": "aiden", "language": "English",
          "seed": 5151, "temperature": 0.0}

    print("[neighbours]", flush=True)
    wait_quiet(a.port)
    ref = Stream(a.port, "/v1/tts/stream", nb).read_all()
    ref_sha = ref.sha.hexdigest()[:16]
    print(f"  ref  neighbour alone: {ref.audio_s():.2f} s  sha {ref_sha}", flush=True)
    wait_quiet(a.port)
    b0 = books(a.port)
    res = {}

    def neighbour():
        res["n"] = Stream(a.port, "/v1/tts/stream", nb).read_all()
    t = threading.Thread(target=neighbour)
    t.start()
    time.sleep(0.3)
    killers = [threading.Thread(target=aborter, args=args) for args in (
        (a.port, "/v1/tts/stream", "batch", "rst", 6001, 1.0),
        (a.port, "/v1/tts/stream", "batch", "fin", 6002, 1.5),
        (a.port, "/v1/tts/stream", "batch", "rst", 6003, 0.0))]
    for k in killers:
        k.start()
    for k in killers:
        k.join()
    t.join()
    n = res["n"]
    check(n.status == 200 and n.complete, "neighbours: the healthy stream completed",
          f"status {n.status}, complete {n.complete}")
    check(n.sha.hexdigest()[:16] == ref_sha and n.audio == ref.audio,
          "neighbours: its audio is identical to the unloaded reference",
          f"{n.audio_s():.2f} s sha {n.sha.hexdigest()[:16]} vs {ref.audio_s():.2f} s sha {ref_sha}")
    check_books("neighbours", b0, wait_books_idle(a.port), {"completed": 1, "client_gone": 3})

    print("[abort-loop]", flush=True)
    wait_quiet(a.port)
    rss0 = rss_kb(a.server_pid) if a.server_pid else 0
    b0 = books(a.port)
    # Two groups per round, so every request is admitted and none is refused: the batched
    # slots (4) take the first group at once; the single-job clone serves one request at a
    # time, and its queue is bounded by the batched occupancy, so it runs on its own.
    groups = [[("/v1/tts/stream", "batch", "rst", 1.0), ("/v1/tts/stream", "batch", "fin", 1.0),
               ("/v1/tts/stream", "batch", "rst", 0.0), ("/v1/tts", "batch", "rst", 1.5)],
              [("/v1/tts/stream", "single", "rst", 1.0), ("/v1/tts/stream", "single", "fin", 1.0)]]
    n_ab = 0
    for rnd in range(a.abort_rounds):
        for gi, plan in enumerate(groups):
            ts = [threading.Thread(target=aborter,
                                   args=(a.port, p, k, h, 7000 + rnd * 10 + gi * 5 + i, s))
                  for i, (p, k, h, s) in enumerate(plan)]
            for x in ts:
                x.start()
            for x in ts:
                x.join()
            wait_books_idle(a.port)
            n_ab += len(plan)
    b1 = wait_books_idle(a.port)
    wait_quiet(a.port)
    rss1 = rss_kb(a.server_pid) if a.server_pid else 0
    check_books(f"abort-loop ({n_ab} aborts)", b0, b1, {"client_gone": n_ab})
    if a.server_pid:
        g = rss1 / rss0 if rss0 else float("inf")
        check(g <= 1.10, "abort-loop: server RSS does not grow with the aborts",
              f"{rss0 / 1024:.0f} -> {rss1 / 1024:.0f} MB ({g:.3f}x, bound 1.10x)")


def http_json(port: int, path: str, body: dict, timeout: float = 120.0):
    st = Stream(port, path, body, timeout=timeout).read_all(timeout)
    return st.status, st


def metrics_books(mport: int) -> dict:
    with urllib.request.urlopen(f"http://127.0.0.1:{mport}/metrics", timeout=10) as r:
        txt = r.read().decode()
    names = {"sessions": "qwen_tts_worker_sessions_total",
             "completed": "qwen_tts_worker_terminated_ok_total",
             "client_gone": "qwen_tts_worker_terminated_client_gone_total",
             "timeout": "qwen_tts_worker_terminated_timeout_total",
             "rejected": "qwen_tts_worker_terminated_rejected_total",
             "failed": "qwen_tts_worker_terminated_failed_total",
             "active": "qwen_tts_worker_sessions_active",
             "balanced": "qwen_tts_worker_books_balanced",
             "anomalies": "qwen_tts_worker_books_anomalies_total"}
    out = {}
    for k, n in names.items():
        vals = [float(m.group(1)) for m in re.finditer(rf"^{n}{{[^}}]*}} ([0-9.]+)$", txt, re.M)]
        out[k] = sum(vals) if vals else None
    return out


def case_books(a):
    """A deterministic mixed workload; the counters must equal it, in both views."""
    import threading
    print("[books]", flush=True)
    wait_quiet(a.port)
    b0 = books(a.port)
    m0 = metrics_books(a.metrics_port) if a.metrics_port else None
    short = lambda i: {"text": "Good morning, everyone.", "speaker": "ryan",
                       "language": "English", "seed": 100 + i, "temperature": 0.0}
    mid = {"text": LONG_TEXT[:190], "speaker": "ryan", "language": "English",
           "seed": 200, "temperature": 0.0}

    # 3 completed
    for i in range(3):
        st = Stream(a.port, "/v1/tts/stream", short(i)).read_all()
        check(st.status == 200 and st.complete, f"books: short request {i} completed",
              f"status {st.status}")
    # 2 client_gone: RST on the first audio
    for i in range(2):
        aborter(a.port, "/v1/tts/stream", "batch", "rst", 0, 0.0, dict(mid, seed=300 + i))
    wait_books_idle(a.port)
    # 2 timeouts: a WAV longer than the 3 s budget -> 503
    for i in range(2):
        code, st = http_json(a.port, "/v1/tts", dict(mid, seed=400 + i))
        check(code == 503, f"books: WAV {i} stopped by the request budget (discriminating)",
              f"status {code}{' -- the machine finished inside the budget' if code == 200 else ''}")
    # queue full: 2 running + 1 queued hold the server, the 4th is refused
    holders = [Stream(a.port, "/v1/tts/stream", dict(mid, seed=500 + i)) for i in range(3)]
    t_end = time.monotonic() + 30
    while time.monotonic() < t_end:
        h = health(a.port)
        if h.get("num_requests_running") == 2 and h.get("num_requests_waiting") == 1:
            break
        time.sleep(0.05)
    code, st = http_json(a.port, "/v1/tts/stream", short(9))
    check(code == 503 and b"queue full" in st.body,
          "books: the request past slots + queue is refused with 503", f"status {code}")
    hs = [threading.Thread(target=x.read_all) for x in holders]
    for x in hs:
        x.start()
    for x in hs:
        x.join()
    # 1 invalid
    code, _ = http_json(a.port, "/v1/tts", {"text": ""})
    check(code == 400, "books: an empty text is refused with 400", f"status {code}")

    b1 = wait_books_idle(a.port)
    # holders: each ran into the 3 s budget (2 at once, then the queued one)
    check_books("books (mixed workload)", b0, b1,
                {"completed": 3, "client_gone": 2, "timeout": 2 + 3, "rejected": 2})
    if a.metrics_port and m0 is not None:
        m1 = metrics_books(a.metrics_port)
        missing = [k for k, v in m1.items() if v is None]
        check(not missing, "books: /metrics exports every books series",
              f"missing {missing}" if missing else "")
        if not missing:
            d = {k: int(m1[k] - m0[k]) for k in ("sessions",) + OUTCOMES}
            hd = books_delta(b0, b1)
            check(d == hd, "books: /metrics counters moved exactly like /v1/health",
                  f"metrics {d} vs health {hd}")
            check(m1["balanced"] == 1 and m1["active"] == 0 and m1["anomalies"] == 0,
                  "books: /metrics says balanced, nothing active, no anomaly",
                  f"balanced={m1['balanced']} active={m1['active']} anomalies={m1['anomalies']}")


CASES = {"zombie": case_zombie, "semantics": case_semantics, "service": case_service,
         "books": case_books}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--min-pending-s", type=float, default=10.0,
                    help="audio that must still be ungenerated at the disconnect")
    ap.add_argument("--send-timeout-s", type=float, default=5.0,
                    help="the server's send timeout (QWEN_STREAM_OUTPUT_SEND_TIMEOUT_MS)")
    ap.add_argument("--server-pid", type=int, default=0, help="for the RSS bound")
    ap.add_argument("--metrics-port", type=int, default=0)
    ap.add_argument("--abort-rounds", type=int, default=3)
    ap.add_argument("cases", nargs="+", choices=sorted(CASES))
    a = ap.parse_args()
    for c in a.cases:
        CASES[c](a)
    print()
    if FAILS:
        print(f"fault_probe: {len(FAILS)} invariant(s) FAILED")
        for f in FAILS:
            print(f"  - {f}")
        return 1
    print("fault_probe: all invariants hold")
    return 0


if __name__ == "__main__":
    sys.exit(main())
