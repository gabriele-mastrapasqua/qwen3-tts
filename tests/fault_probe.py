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
    st = Stream(a.port, "/v1/tts/stream", body_for("batch", seed))
    st.s.shutdown(socket.SHUT_WR)
    st.read_all()
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


CASES = {"zombie": case_zombie, "semantics": case_semantics}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--min-pending-s", type=float, default=10.0,
                    help="audio that must still be ungenerated at the disconnect")
    ap.add_argument("--send-timeout-s", type=float, default=5.0,
                    help="the server's send timeout (QWEN_STREAM_OUTPUT_SEND_TIMEOUT_MS)")
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
