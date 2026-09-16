#!/usr/bin/env python3
"""Deterministic tests for the client-observed playback metrics (tests/playback_sim.py).

Every timeline here has an answer derived by hand in the comments.  Times are seconds
since the request was sent; every chunk carries the audio duration written next to it.
"""
import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import playback_sim as pb

SR = pb.SR


def chunk(seconds):
    return int(seconds * 2 * SR)


class SafePlayStart(unittest.TestCase):
    def test_smooth_stream_needs_no_prebuffer(self):
        # 0.5 s chunks every 0.4 s: the player is never late.
        k = pb.timeline_kpis(pb.synth("smooth"), 4.0)
        self.assertEqual(k["required_prebuffer_s"], 0.0)
        self.assertAlmostEqual(k["safe_play_start_s"], k["ttfa_s"])
        self.assertEqual(k["stall_count"], 0)
        for b in pb.BUFFERS_MS:
            self.assertEqual(k[f"stalls@{b}"], 0)

    def test_safe_play_start_is_the_earliest_stall_free_start(self):
        # First chunk 0.5 s at 1.0; second 0.5 s at 2.0.  Starting at 1.5 the player
        # drains exactly when the second chunk lands; any earlier start stalls.
        marks = [(1.0, chunk(0.5)), (2.0, chunk(0.5))]
        k = pb.timeline_kpis(marks, 2.0)
        self.assertAlmostEqual(k["safe_play_start_s"], 1.5)
        self.assertAlmostEqual(k["required_prebuffer_s"], 0.5)
        # A player starting at safe_play_start is stall-free; 1 ms earlier is not.
        self.assertEqual(pb.analyse(marks, 2.5, prebuffer_s=0.5)["stall_count"], 0)
        self.assertEqual(pb.analyse(marks, 2.5, prebuffer_s=0.499)["stall_count"], 1)

    def test_per_request_identity_with_ttfa_plus_prebuffer(self):
        # safe_play_start is computed by a direct scan, never from percentiles; per
        # request it must still equal ttfa + required_prebuffer.
        for kind in ("smooth", "bursty", "late_first", "repeated_gap"):
            m = pb.synth(kind, n=9)
            k = pb.timeline_kpis(m, m[-1][0] + 0.2)
            self.assertAlmostEqual(k["safe_play_start_s"],
                                   k["ttfa_s"] + k["required_prebuffer_s"], msg=kind)

    def test_late_first_chunk(self):
        # 80 ms first chunk at 0.3, then nothing until 2.3: the player must wait
        # 2.0 − 0.08 = 1.92 s after first audio; safe start = 0.3 + 1.92.
        k = pb.timeline_kpis(pb.synth("late_first"), 6.0)
        self.assertAlmostEqual(k["required_prebuffer_s"], 1.92)
        self.assertAlmostEqual(k["safe_play_start_s"], 2.22)
        self.assertAlmostEqual(k["max_gap_s"], 2.0)
        self.assertAlmostEqual(k["stall_max_s"], 1.92)
        self.assertEqual(k["stall_count"], 1)


class FixedBufferPlayer(unittest.TestCase):
    def test_zero_buffer_equals_the_zero_buffer_player(self):
        for kind in ("smooth", "bursty", "late_first", "repeated_gap"):
            m = pb.synth(kind, n=9)
            k = pb.timeline_kpis(m, m[-1][0])
            start, total, worst, count = pb.fixed_buffer_sim(m, m[-1][0], 0.0)
            self.assertAlmostEqual(start, 0.0, msg=kind)
            self.assertAlmostEqual(total, k["underrun_total_s"], msg=kind)
            self.assertAlmostEqual(worst, k["stall_max_s"], msg=kind)
            self.assertEqual(count, k["stall_count"], msg=kind)

    def test_bursty_quanta(self):
        # 0.5 s at 0.3, then four 0.5 s chunks together at 1.9 and at 3.5.
        # @250/@500: start at 0.3 (0.5 s held), dry at 0.8, resume 1.9 -> one 1.1 s stall.
        # @1000: start when 1.0 s is held, i.e. at 1.9 (2.5 s held): no stall afterwards.
        k = pb.timeline_kpis(pb.synth("bursty", n=9), 4.0)
        self.assertAlmostEqual(k["required_prebuffer_s"], 1.1)
        for b in (100, 250, 500):
            self.assertEqual(k[f"stalls@{b}"], 1, msg=b)
            self.assertAlmostEqual(k[f"stall_ms@{b}"], 1100.0, msg=b)
            self.assertAlmostEqual(k[f"stall_max_ms@{b}"], 1100.0, msg=b)
        self.assertEqual(k["stalls@1000"], 0)
        self.assertAlmostEqual(k["start_delay_ms@1000"], 1600.0)

    def test_repeated_gaps_rebuffer(self):
        # 0.5 s chunks every 0.4 s, with a 0.4 s extra delay before chunks 3 and 6.
        # Lead grows 0.1 s per chunk; the 0.8 s gap exceeds the 0.7 s lead at chunk 3
        # by 0.1 s and the 0.9 s lead at chunk 6 by... no: lead at chunk 6 is 0.8 s,
        # deficit 0.0 -> only the required prebuffer at chunk 3 (0.1) and chunk 6 (0.2).
        k = pb.timeline_kpis(pb.synth("repeated_gap"), 6.0)
        self.assertAlmostEqual(k["required_prebuffer_s"], 0.2)
        self.assertEqual(k["stall_count"], 2)
        self.assertEqual(k["stalls@500"], 2)      # a 0.5 s buffer re-buffers at each gap
        self.assertEqual(k["stalls@1000"], 0)     # a 1.0 s buffer absorbs both

    def test_stream_end_starts_playback_even_below_buffer(self):
        # A 0.3 s utterance never reaches a 1.0 s buffer: the player starts at the last
        # chunk instead of never.
        marks = [(0.2, chunk(0.1)), (0.3, chunk(0.1)), (0.4, chunk(0.1))]
        start, total, worst, count = pb.fixed_buffer_sim(marks, 0.4, 1.0)
        self.assertAlmostEqual(start, 0.2)
        self.assertEqual(count, 0)
        self.assertEqual(total, 0.0)


class Degenerate(unittest.TestCase):
    def test_single_chunk_has_no_cadence(self):
        k = pb.timeline_kpis([(0.5, chunk(1.0))], 0.6)
        self.assertEqual(k["chunks"], 1)
        self.assertTrue(math.isnan(k["required_prebuffer_s"]))
        self.assertTrue(math.isnan(k["safe_play_start_s"]))
        self.assertEqual(k["stalls@250"], 0)
        s = pb.summarize([k])
        self.assertEqual(s["n"], 0)

    def test_coalesced_reads_are_counted(self):
        # Third field = time blocked in the read.  Two reads returning in 20 us are
        # already-queued data; the mark overstates their lateness.
        marks = [(0.3, chunk(0.5), 0.3), (0.7, chunk(0.5), 0.39), (0.70002, chunk(0.5), 0.00002)]
        k = pb.timeline_kpis(marks, 1.0)
        self.assertEqual(k["coalesced_chunks"], 1)
        self.assertTrue(k["blocked_reads_known"])
        s = pb.summarize([k])
        self.assertAlmostEqual(s["coalesced_chunk_share"], 1 / 3)

    def test_marks_without_blocked_time_still_work(self):
        k = pb.timeline_kpis([(0.3, chunk(0.5)), (0.7, chunk(0.5))], 1.0)
        self.assertEqual(k["coalesced_chunks"], 0)
        self.assertFalse(k["blocked_reads_known"])


class Summaries(unittest.TestCase):
    def test_stall_rate_and_prebuffer_cdf(self):
        recs = [pb.timeline_kpis(pb.synth(kind, n=9), 6.0)
                for kind in ("smooth", "bursty", "late_first", "repeated_gap")]
        s = pb.summarize(recs)
        self.assertEqual(s["n"], 4)
        # bursty (starts at once, 1.1 s dry) and repeated_gap (re-buffers at both gaps)
        # stall at 250 ms; smooth never; late_first does NOT stall because an audio-based
        # 250 ms buffer cannot start on its 80 ms first chunk and waits 2.0 s instead,
        # which is why start_delay is reported next to the stall rate.
        self.assertAlmostEqual(s["stall_rate@250"], 2 / 4)
        self.assertAlmostEqual(s["stall_rate@1000"], 0.0)
        late = recs[2]
        self.assertAlmostEqual(late["start_delay_ms@250"], 2000.0)
        self.assertAlmostEqual(s["prebuffer_le_rate@250"], 2 / 4)   # smooth, repeated_gap
        self.assertAlmostEqual(s["prebuffer_le_rate@1000"], 2 / 4)
        text = pb.format_summary(s)
        self.assertIn("safe_play_start", text)
        self.assertIn("@250ms stall_rate 50%", text)

    def test_percentile_is_nearest_rank(self):
        self.assertEqual(pb.pct([1, 2, 3, 4, 5], 50), 3)
        self.assertEqual(pb.pct([1, 2, 3, 4, 5], 95), 5)
        self.assertTrue(math.isnan(pb.pct([], 50)))


if __name__ == "__main__":
    unittest.main()
