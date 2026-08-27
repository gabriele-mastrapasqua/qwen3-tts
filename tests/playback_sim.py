#!/usr/bin/env python3
"""playback_sim.py — a consumer that plays audio at 1x realtime, from arrival timestamps.

WHY IT IS ITS OWN UNIT
  Every call-shaped KPI depends on this, and two of them - the barge-in trigger and the
  minimum prebuffer - are wrong in ways that look plausible if the maths is wrong. So it is
  a library with a self-test on synthetic timelines whose answers are known by hand.

THE MODEL
  Playback starts when the first chunk arrives, plus an optional prebuffer B. It advances
  with the wall clock and stalls when it has consumed everything delivered. It never pauses
  voluntarily.

      played(t) = min( avail(t) , played(t-) + (t - t-) )      for t >= t_first + B

  `heard` is played(t); `delivered` is avail(t); they are NOT the same number and the gap
  between them is buffered-but-unheard audio.

MINIMUM PREBUFFER
  With prebuffer B, at every arrival t_i the playhead must not have consumed more than what
  arrived strictly before it:

      (t_i - t_first - B) <= A(i)          A(i) = sum of durations of chunks 0..i-1
  =>  B >= (t_i - t_first) - A(i)
  =>  min_prebuffer = max(0, max_i[(t_i - t_first) - A(i)])

  The binding instants are exactly the arrivals, so the max over them is exact, not a bound.

⚠️ The v1 call-shaped run plays with ZERO intentional prebuffer, to observe the natural
cadence. min_prebuffer is COMPUTED from the timeline; it is not applied to the playback.
Applying one is a separate, explicit runtime parameter.
"""
SR = 24000


def audio_s(nbytes):
    return nbytes / 2.0 / SR


class Playhead:
    """Incremental playback state, so a live client can ask 'how much has been HEARD?'
    while the stream is still arriving."""

    def __init__(self, prebuffer_s=0.0):
        self.B = prebuffer_s
        self.t_first = None
        self.avail = 0.0          # audio delivered
        self.played = 0.0         # audio actually heard
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


def analyse(marks, t_done, prebuffer_s=0.0):
    """marks = [(t_rel, nbytes)] relative to t_send, in arrival order. Returns the KPIs."""
    if len(marks) < 2:
        return {"stream_rtf": float("nan"), "stall_count": 0, "total_stall_ms": 0.0,
                "worst_continuous_stall_ms": 0.0, "deepest_deficit_ms": 0.0,
                "min_prebuffer_ms": 0.0, "chunks": len(marks),
                "heard_s": audio_s(marks[0][1]) if marks else 0.0,
                "delivered_s": audio_s(marks[0][1]) if marks else 0.0}
    t_first = marks[0][0]
    ph = Playhead(prebuffer_s)
    for t, nb in marks:
        ph.on_chunk(t, nb)
    ph.advance(t_done)

    # minimum prebuffer: exact maximum over the binding instants
    A = 0.0
    need = 0.0
    for i, (t, nb) in enumerate(marks):
        if i > 0:
            need = max(need, (t - t_first) - A)
        A += audio_s(nb)

    total = sum(audio_s(nb) for _t, nb in marks)
    rest = total - audio_s(marks[0][1])
    return {
        "stream_rtf": (t_done - t_first) / rest if rest > 0 else float("nan"),
        "stall_count": ph.stall_count,
        "total_stall_ms": ph.stall_total * 1000.0,
        "worst_continuous_stall_ms": ph.worst_stall * 1000.0,
        "deepest_deficit_ms": (ph.deepest_deficit if ph.deepest_deficit > float("-inf") else 0.0) * 1000.0,
        "min_prebuffer_ms": max(0.0, need) * 1000.0,
        "chunks": len(marks),
        "heard_s": ph.heard,
        "delivered_s": ph.avail,
        "buffered_unheard_s": ph.buffered_unheard,
    }


def _selftest():
    """Synthetic timelines whose answers are known by hand."""
    ok = True

    def check(name, got, want, tol=1e-6):
        nonlocal ok
        good = abs(got - want) <= tol
        ok = ok and good
        print(f"  {'ok  ' if good else 'FAIL'} {name}: got {got:.6f} want {want:.6f}")

    # 1. perfectly paced: a 0.5 s chunk every 0.5 s -> no stall, no prebuffer needed
    marks = [(1.0 + 0.5 * i, int(0.5 * 2 * SR)) for i in range(5)]
    r = analyse(marks, t_done=1.0 + 0.5 * 5)
    check("paced/stall_ms", r["total_stall_ms"], 0.0)
    check("paced/prebuffer_ms", r["min_prebuffer_ms"], 0.0)
    check("paced/stream_rtf", r["stream_rtf"], (0.5 * 5) / (0.5 * 4))

    # 2. one late chunk: 0.5 s of audio, then a 1.0 s gap -> starves for exactly 0.5 s
    marks = [(1.0, int(0.5 * 2 * SR)), (2.0, int(0.5 * 2 * SR))]
    r = analyse(marks, t_done=2.0)
    check("late/stall_ms", r["total_stall_ms"], 500.0, 1e-3)
    check("late/stall_count", r["stall_count"], 1)
    check("late/prebuffer_ms", r["min_prebuffer_ms"], 500.0, 1e-3)

    # 3. running ahead: chunks arrive faster than realtime -> heard < delivered
    marks = [(1.0 + 0.1 * i, int(0.5 * 2 * SR)) for i in range(4)]
    r = analyse(marks, t_done=1.3)
    check("ahead/delivered_s", r["delivered_s"], 2.0)
    check("ahead/heard_s", r["heard_s"], 0.3, 1e-6)
    check("ahead/buffered_unheard_s", r["buffered_unheard_s"], 1.7, 1e-6)
    check("ahead/stall_ms", r["total_stall_ms"], 0.0)

    # 4. the prebuffer that fixes case 2 makes it gap-free
    marks = [(1.0, int(0.5 * 2 * SR)), (2.0, int(0.5 * 2 * SR))]
    r = analyse(marks, t_done=2.5, prebuffer_s=0.5)
    check("prebuffered/stall_ms", r["total_stall_ms"], 0.0)

    # 5. live Playhead matches the batch analysis
    ph = Playhead()
    for t, nb in [(1.0, int(0.5 * 2 * SR)), (2.0, int(0.5 * 2 * SR))]:
        ph.on_chunk(t, nb)
    ph.advance(2.0)
    check("live/heard_s", ph.heard, 0.5, 1e-6)
    check("live/stall_ms", ph.stall_total * 1000.0, 500.0, 1e-3)
    print("  " + ("✅ playback simulator self-test PASSED" if ok else "❌ FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(_selftest())
