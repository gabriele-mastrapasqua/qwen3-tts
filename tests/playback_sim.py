#!/usr/bin/env python3
"""playback_sim.py — client-observed playback metrics from one chunk-arrival timeline.

This module is the single definition of every per-request playback metric used by the
benchmark harnesses (``tests/soak_client.py``, ``tests/serve_parallel_wave.py``,
``tests/cadence_dump.py``).  Everything here is derived from CLIENT RECEIVE MARKS:
``(t_rel_s, nbytes[, blocked_s])`` in arrival order, ``t_rel_s`` measured from the moment
the request was sent, ``nbytes`` of s16le mono 24 kHz PCM, and optionally ``blocked_s``,
the wall time the client spent inside the read call that returned the chunk.

Receive-marker semantics (MT-1).  A mark is stamped when the client's chunked-transfer
reader returns one HTTP chunk to the harness.  It is therefore an application-level
arrival: socket, kernel and Python buffering all sit between the server's write and the
mark, and a reader that is late (GIL, other threads) can find several chunks already
queued and return them back-to-back.  A chunk whose read returned in less than
``COALESCE_S`` is counted in ``coalesced_chunks``: its true arrival was earlier than its
mark, so cadence metrics are UPPER bounds on the server's lateness.  Every cadence
quantity below is "client-observed" until a transport audit proves otherwise; the
detailed audit is in ``.work/professional-streaming-architecture.md`` (MT-1).

Definitions (all per request, seconds unless stated):

* ``ttfa``           = t_0, the arrival of the first non-empty audio chunk.
* ``stream_rtf``     = (t_done − t_0) / (audio delivered after the first chunk).  A MEAN
                       rate over the stream; it is blind to delivery quantization and is a
                       capacity metric, not a continuity proof.
* ``required_prebuffer`` = max(0, max_{i>=1} [(t_i − t_0) − A_i]) with A_i the audio
                       delivered before chunk i.  The smallest fixed delay after t_0 at
                       which a 1x player that then never pauses finishes without underrun.
* ``safe_play_start`` = max_i (t_i − A_i), i >= 0 (A_0 = 0).  The earliest absolute start
                       time after the request at which a 1x player could begin and finish
                       without underrun under the observed timeline.  It is computed by a
                       direct scan of the timeline, never as TTFA plus a percentile;
                       per request it satisfies safe_play_start = ttfa + required_prebuffer.
* zero-buffer player: starts at t_0, pauses only when its buffer is empty, resumes on the
                       next arrival.  ``underrun_total``, ``stall_max``, ``stall_count``.
* fixed-buffer player @B: starts when B seconds of audio are buffered (or the stream ended),
                       and after an underrun re-buffers B seconds before resuming.
                       ``stall_ms@B``, ``stalls@B``, ``stall_max_ms@B``, ``start_delay_ms@B``
                       (start relative to t_0).  At B = 0 it is the zero-buffer player.
* ``max_gap``        = largest inter-arrival gap after t_0 (raw cadence, ignores buffering).
* ``gap_ratio_max``  = max over chunks of (inter-arrival gap / chunk audio duration).

Aggregation (``summarize``): p50/p95 by nearest rank over requests; ``stall_rate@B`` is
the fraction of requests with at least one stall under the fixed-buffer player, and
``prebuffer_le_rate@B`` the fraction whose required_prebuffer <= B (a time-based start
delay of B is stall-free exactly for those requests).
"""
SR = 24000
BUFFERS_MS = (100, 250, 500, 1000)
COALESCE_S = 0.001          # a read that returns faster than this found data already queued


def audio_s(nbytes):
    return nbytes / 2.0 / SR


def pct(values, q):
    """Percentile by nearest rank (same convention as tests/serve_parallel_wave.py)."""
    v = sorted(x for x in values if x == x)
    if not v:
        return float("nan")
    return v[min(len(v) - 1, int(round(q / 100.0 * (len(v) - 1))))]


class Playhead:
    """Incremental zero-buffer playback state (optionally delayed by a fixed prebuffer),
    so a live client can ask 'how much has been HEARD?' while the stream still arrives."""

    def __init__(self, prebuffer_s=0.0):
        self.B = prebuffer_s
        self.t_first = None
        self.avail = 0.0
        self.played = 0.0
        self.t_prev = None
        self.stall_total = 0.0
        self.stall_count = 0
        self.worst_stall = 0.0
        self.deepest_deficit = float("-inf")
        self._in_stall = False

    def on_chunk(self, t, nbytes):
        """A chunk arrived at wall time t (absolute) carrying nbytes of PCM."""
        self.advance(t)
        if self.t_first is None:
            self.t_first = t
            self.t_prev = t + self.B
        self.avail += audio_s(nbytes)

    def advance(self, t):
        """Move the playhead to wall time t without new data."""
        if self.t_first is None:
            return
        start = self.t_first + self.B
        if t <= start:
            return
        prev = self.t_prev if self.t_prev is not None else start
        gap = t - prev
        if gap <= 0:
            return
        want = self.played + gap
        deficit = want - self.avail
        self.deepest_deficit = max(self.deepest_deficit, deficit)
        if deficit > 0:
            self.stall_total += deficit
            self.worst_stall = max(self.worst_stall, deficit)
            if not self._in_stall:
                self.stall_count += 1
                self._in_stall = True
            self.played = self.avail
        else:
            self._in_stall = False
            self.played = want
        self.t_prev = t

    @property
    def heard(self):
        return self.played

    @property
    def buffered_unheard(self):
        return max(0.0, self.avail - self.played)


def _norm(marks):
    """Return [(t, bytes, blocked_or_None)] in arrival order."""
    out = []
    for m in marks:
        t, nb = m[0], m[1]
        blocked = m[2] if len(m) > 2 else None
        out.append((float(t), int(nb), blocked))
    out.sort(key=lambda m: m[0])
    return out


def fixed_buffer_sim(marks, t_done, buffer_s):
    """Simulate a 1x player with an audio-based jitter buffer of ``buffer_s``.

    The player starts when at least ``buffer_s`` of audio has arrived (or when the
    stream has ended, whichever is first).  When its buffer runs dry it pauses, and it
    resumes only once ``buffer_s`` of unplayed audio is buffered again or the stream has
    ended.  Nothing after the last arrival can stall, so the simulation stops there.
    Returns start_delay_s (from t_0), stall_total_s, stall_max_s, stall_count.
    """
    ms = _norm(marks)
    if not ms:
        return float("nan"), float("nan"), float("nan"), 0
    t0 = ms[0][0]
    last = len(ms) - 1
    avail = 0.0
    played = 0.0
    playing = False
    start_at = None
    stall_from = None
    stall_total = 0.0
    stall_max = 0.0
    stall_count = 0
    for i, (t, nb, _b) in enumerate(ms):
        # Advance playback from the previous arrival to this one.
        if playing and i > 0:
            gap = t - ms[i - 1][0]
            lead = avail - played
            if gap <= lead:
                played += gap
            else:
                played = avail
                stall_from = ms[i - 1][0] + lead      # buffer ran dry here
                stall_count += 1
                playing = False
        avail += audio_s(nb)
        if not playing:
            can_start = (avail - played) >= buffer_s - 1e-12 or i == last
            if can_start:
                playing = True
                if start_at is None:
                    start_at = t
                elif stall_from is not None:
                    d = t - stall_from
                    stall_total += d
                    stall_max = max(stall_max, d)
                    stall_from = None
    return (start_at - t0 if start_at is not None else float("nan"),
            stall_total, stall_max, stall_count)


def timeline_kpis(marks, t_done, buffers_ms=BUFFERS_MS):
    """All per-request playback metrics from one arrival timeline.  Keys in seconds
    unless suffixed ``_ms``.  Requests with fewer than two chunks have no cadence."""
    ms = _norm(marks)
    n = len(ms)
    out = {
        "chunks": n,
        "coalesced_chunks": sum(1 for m in ms if m[2] is not None and m[2] < COALESCE_S),
        "blocked_reads_known": any(m[2] is not None for m in ms),
    }
    if n == 0:
        out.update({"ttfa_s": float("nan"), "delivered_s": 0.0})
    else:
        out["ttfa_s"] = ms[0][0]
        out["delivered_s"] = sum(audio_s(nb) for _t, nb, _b in ms)
    if n < 2:
        for k in ("stream_rtf", "required_prebuffer_s", "safe_play_start_s",
                  "underrun_total_s", "stall_max_s", "max_gap_s", "gap_ratio_max"):
            out[k] = float("nan")
        out["stall_count"] = 0
        for b in buffers_ms:
            out[f"start_delay_ms@{b}"] = float("nan")
            out[f"stall_ms@{b}"] = float("nan")
            out[f"stall_max_ms@{b}"] = float("nan")
            out[f"stalls@{b}"] = 0
        return out

    t0 = ms[0][0]
    rest = sum(audio_s(nb) for _t, nb, _b in ms[1:])
    out["stream_rtf"] = (t_done - t0) / rest if rest > 0 else float("nan")

    # required_prebuffer and safe_play_start by direct scan of the timeline.
    A = 0.0
    need = 0.0
    safe = float("-inf")
    for i, (t, nb, _b) in enumerate(ms):
        safe = max(safe, t - A)                  # i = 0 contributes t_0 (A_0 = 0)
        if i > 0:
            need = max(need, (t - t0) - A)
        A += audio_s(nb)
    out["required_prebuffer_s"] = max(0.0, need)
    out["safe_play_start_s"] = safe

    # Zero-buffer player.
    ph = Playhead(0.0)
    for t, nb, _b in ms:
        ph.on_chunk(t, nb)
    ph.advance(ms[-1][0])                        # nothing after the last arrival can stall
    out["underrun_total_s"] = ph.stall_total
    out["stall_max_s"] = ph.worst_stall
    out["stall_count"] = ph.stall_count

    # Raw cadence.
    gaps = [ms[i][0] - ms[i - 1][0] for i in range(1, n)]
    out["max_gap_s"] = max(gaps)
    ratios = [g / audio_s(ms[i][1]) for i, g in zip(range(1, n), gaps) if ms[i][1] > 0]
    out["gap_ratio_max"] = max(ratios) if ratios else float("nan")
    out["gap_ratios"] = ratios

    # Fixed audio-based jitter buffers.
    for b in buffers_ms:
        start, total, worst, count = fixed_buffer_sim(ms, t_done, b / 1000.0)
        out[f"start_delay_ms@{b}"] = start * 1000.0
        out[f"stall_ms@{b}"] = total * 1000.0
        out[f"stall_max_ms@{b}"] = worst * 1000.0
        out[f"stalls@{b}"] = count
    return out


def summarize(records, buffers_ms=BUFFERS_MS, prefix=""):
    """p50/p95 over per-request ``timeline_kpis`` dicts plus the fixed-buffer rates.
    ``records`` may carry the keys under ``prefix`` (e.g. a harness that renamed them)."""
    def col(key):
        return [r.get(prefix + key, float("nan")) for r in records if r.get(prefix + key) is not None]
    out = {}
    for key in ("ttfa_s", "stream_rtf", "required_prebuffer_s", "safe_play_start_s",
                "underrun_total_s", "stall_max_s", "max_gap_s", "gap_ratio_max"):
        v = col(key)
        out[f"{key}_p50"] = pct(v, 50)
        out[f"{key}_p95"] = pct(v, 95)
    valid = [r for r in records if r.get(prefix + "required_prebuffer_s", float("nan")) == r.get(prefix + "required_prebuffer_s", float("nan"))]
    out["n"] = len(valid)
    for b in buffers_ms:
        stalled = [r for r in valid if r.get(prefix + f"stalls@{b}", 0) > 0]
        out[f"stall_rate@{b}"] = (len(stalled) / len(valid)) if valid else float("nan")
        out[f"stall_ms@{b}_p50"] = pct(col(f"stall_ms@{b}"), 50)
        out[f"stall_ms@{b}_p95"] = pct(col(f"stall_ms@{b}"), 95)
        out[f"stall_max_ms@{b}_p95"] = pct(col(f"stall_max_ms@{b}"), 95)
        out[f"start_delay_ms@{b}_p95"] = pct(col(f"start_delay_ms@{b}"), 95)
        le = [r for r in valid if r.get(prefix + "required_prebuffer_s") <= b / 1000.0 + 1e-9]
        out[f"prebuffer_le_rate@{b}"] = (len(le) / len(valid)) if valid else float("nan")
    coal = col("coalesced_chunks")
    chunks = col("chunks")
    out["coalesced_chunk_share"] = (sum(coal) / sum(chunks)) if sum(chunks) else float("nan")
    return out


def format_summary(s, buffers_ms=BUFFERS_MS, indent="  "):
    """Two standard lines for a harness report."""
    ms = lambda x: f"{x * 1000:.0f}"
    l1 = (f"{indent}PLAYBACK (client-observed, n={s['n']}): required_prebuffer p50/p95 "
          f"{ms(s['required_prebuffer_s_p50'])}/{ms(s['required_prebuffer_s_p95'])} ms · "
          f"safe_play_start p50/p95 {ms(s['safe_play_start_s_p50'])}/{ms(s['safe_play_start_s_p95'])} ms · "
          f"stall_max p95 {ms(s['stall_max_s_p95'])} ms · underrun_total p95 "
          f"{ms(s['underrun_total_s_p95'])} ms · max_gap p95 {ms(s['max_gap_s_p95'])} ms")
    parts = []
    for b in buffers_ms:
        parts.append(f"@{b}ms stall_rate {s[f'stall_rate@{b}'] * 100:.0f}% "
                     f"(total p95 {s[f'stall_ms@{b}_p95']:.0f} ms, start p95 {s[f'start_delay_ms@{b}_p95']:.0f} ms)")
    l2 = f"{indent}FIXED BUFFER: " + " · ".join(parts)
    coal = s.get("coalesced_chunk_share", float("nan"))
    l3 = (f"{indent}RECEIVE FIDELITY: coalesced chunks {coal * 100:.1f}% of reads returned "
          f"already-queued data (cadence values are upper bounds on server lateness)"
          if coal == coal else f"{indent}RECEIVE FIDELITY: blocked-read times not recorded")
    return "\n".join((l1, l2, l3))


def analyse(marks, t_done, prebuffer_s=0.0):
    """Backward-compatible wrapper (tests/cadence_dump.py): the historical ``_ms`` keys plus
    the full ``timeline_kpis`` set.  ``prebuffer_s`` delays the zero-buffer playhead."""
    k = timeline_kpis(marks, t_done)
    if len(marks) < 2:
        first = audio_s(marks[0][1]) if marks else 0.0
        base = {"stream_rtf": float("nan"), "stall_count": 0, "total_stall_ms": 0.0,
                "worst_continuous_stall_ms": 0.0, "deepest_deficit_ms": 0.0,
                "min_prebuffer_ms": 0.0, "chunks": len(marks),
                "heard_s": first, "delivered_s": first}
        base.update(k)
        return base
    ph = Playhead(prebuffer_s)
    for t, nb in ((m[0], m[1]) for m in marks):
        ph.on_chunk(t, nb)
    ph.advance(t_done)
    base = {
        "stream_rtf": k["stream_rtf"],
        "stall_count": ph.stall_count,
        "total_stall_ms": ph.stall_total * 1000.0,
        "worst_continuous_stall_ms": ph.worst_stall * 1000.0,
        "deepest_deficit_ms": (ph.deepest_deficit if ph.deepest_deficit > float("-inf") else 0.0) * 1000.0,
        "min_prebuffer_ms": k["required_prebuffer_s"] * 1000.0,
        "chunks": len(marks),
        "heard_s": ph.heard,
        "delivered_s": ph.avail,
        "buffered_unheard_s": ph.buffered_unheard,
    }
    base.update({kk: vv for kk, vv in k.items() if kk not in base})
    return base


# --- deterministic synthetic timelines used by the self-test and tests/test_playback_sim.py

def synth(kind, chunk_s=0.5, n=8, t0=0.3, rtf=0.8):
    """Known-answer timelines.  Chunks carry ``chunk_s`` of audio each."""
    nb = int(chunk_s * 2 * SR)
    if kind == "smooth":            # every chunk arrives rtf * chunk_s after the previous
        return [(t0 + i * chunk_s * rtf, nb) for i in range(n)]
    if kind == "bursty":            # one chunk, then quanta of four chunks at once
        marks = [(t0, nb)]
        for k in range(1, 1 + (n - 1) // 4):
            marks += [(t0 + k * 4 * chunk_s * rtf, nb)] * 4
        return marks
    if kind == "late_first":        # tiny first chunk, then a long wait, then smooth
        small = int(0.08 * 2 * SR)
        marks = [(t0, small)]
        marks += [(t0 + 2.0 + i * chunk_s * rtf, nb) for i in range(n - 1)]
        return marks
    if kind == "repeated_gap":      # smooth, but every third chunk is 0.4 s late
        marks = []
        t = t0
        for i in range(n):
            if i > 0:
                t += chunk_s * rtf + (0.4 if i % 3 == 0 else 0.0)
            marks.append((t, nb))
        return marks
    raise ValueError(kind)


def _selftest():
    ok = True

    def check(name, got, want, tol=1e-6):
        nonlocal ok
        good = abs(got - want) <= tol
        ok = ok and good
        print(f"  {'ok  ' if good else 'FAIL'} {name}: got {got:.6f} want {want:.6f}")

    # Historical known answers (unchanged semantics).
    marks = [(1.0 + 0.5 * i, int(0.5 * 2 * SR)) for i in range(5)]
    r = analyse(marks, t_done=1.0 + 0.5 * 5)
    check("paced/stall_ms", r["total_stall_ms"], 0.0)
    check("paced/prebuffer_ms", r["min_prebuffer_ms"], 0.0)
    check("paced/stream_rtf", r["stream_rtf"], (0.5 * 5) / (0.5 * 4))
    marks = [(1.0, int(0.5 * 2 * SR)), (2.0, int(0.5 * 2 * SR))]
    r = analyse(marks, t_done=2.0)
    check("late/stall_ms", r["total_stall_ms"], 500.0, 1e-3)
    check("late/stall_count", r["stall_count"], 1)
    check("late/prebuffer_ms", r["min_prebuffer_ms"], 500.0, 1e-3)
    check("late/safe_play_start_s", r["safe_play_start_s"], 1.5)
    r = analyse(marks, t_done=2.5, prebuffer_s=0.5)
    check("prebuffered/stall_ms", r["total_stall_ms"], 0.0)

    # New synthetic families: smooth, bursty, late-first, repeated gaps.
    for kind in ("smooth", "bursty", "late_first", "repeated_gap"):
        m = synth(kind)
        k = timeline_kpis(m, m[-1][0] + 0.1)
        check(f"{kind}/identity safe=ttfa+prebuffer", k["safe_play_start_s"],
              k["ttfa_s"] + k["required_prebuffer_s"])
        check(f"{kind}/zero-buffer == fixed@0", fixed_buffer_sim(m, m[-1][0], 0.0)[1],
              k["underrun_total_s"])
    k = timeline_kpis(synth("smooth"), 4.0)
    check("smooth/prebuffer", k["required_prebuffer_s"], 0.0)
    check("smooth/stalls@100", k["stalls@100"], 0)
    k = timeline_kpis(synth("bursty", n=9), 4.0)                     # 0.5 s, then 2.0 s every 1.6 s
    check("bursty/prebuffer", k["required_prebuffer_s"], 1.1)        # 1.6 s wait, 0.5 s held
    check("bursty/max_gap", k["max_gap_s"], 1.6)
    check("bursty/stalls@250", k["stalls@250"], 1)
    check("bursty/stall_ms@250", k["stall_ms@250"], 1100.0)
    check("bursty/stalls@1000", k["stalls@1000"], 0)                 # waits for the quantum
    check("bursty/start_delay@1000", k["start_delay_ms@1000"], 1600.0)
    k = timeline_kpis(synth("late_first"), 6.0)
    check("late_first/prebuffer", k["required_prebuffer_s"], 2.0 - 0.08)
    check("late_first/safe_play_start", k["safe_play_start_s"], 0.3 + 2.0 - 0.08)
    check("late_first/stalls@1000", k["stalls@1000"], 0)             # starts once 1 s is held
    check("late_first/start_delay@1000", k["start_delay_ms@1000"], 2400.0)
    k = timeline_kpis(synth("repeated_gap"), 6.0)
    check("repeated_gap/stall_count", k["stall_count"], 2)           # i=3 and i=6
    check("repeated_gap/prebuffer", k["required_prebuffer_s"], 0.2)  # 0.4 late − 0.2 lead
    check("repeated_gap/stalls@500", k["stalls@500"], 2)             # 0.5 s buffer re-buffers twice
    check("repeated_gap/stalls@1000", k["stalls@1000"], 0)
    print("  " + ("✅ playback simulator self-test PASSED" if ok else "❌ FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(_selftest())
