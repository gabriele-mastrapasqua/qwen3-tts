#!/usr/bin/env python3
"""Small standard-library tests for the public soak helpers."""
import csv
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest

sys.path.insert(0, str(Path(__file__).parent))
import soak_client
import soak_drift
import serve_soak


class SoakTests(unittest.TestCase):
    def test_stratified_schedule_cycles_classes(self):
        rows = [("short", "a"), ("medium", "b"), ("long", "c")]
        picker = soak_client.make_picker(rows, worker=0, seed=42, schedule="stratified")
        self.assertEqual([picker(i)[0] for i in range(6)],
                         ["long", "medium", "short", "long", "medium", "short"])

    def test_stream_kpis_expose_zero_buffer_diagnostic(self):
        result = soak_client.stream_kpis(
            [(0.5, 48000), (1.5, 48000), (3.0, 48000)], 3.0
        )
        self.assertAlmostEqual(result["stream_rtf"], 1.25)
        self.assertAlmostEqual(result["underrun_s"], 0.5)
        self.assertAlmostEqual(result["stall_max_s"], 0.5)
        self.assertAlmostEqual(result["prebuffer_s"], 0.5)
        self.assertEqual(result["chunks"], 3)

    def test_stream_kpis_playback_fields_and_csv_layout(self):
        # 48000 bytes = 1.0 s of audio.  Third mark field = seconds blocked in the read;
        # the last read returned already-queued data (coalesced).
        marks = [(0.5, 48000, 0.5), (1.5, 48000, 1.0), (3.0, 48000, 1.5), (3.0001, 48000, 0.0001)]
        result = soak_client.stream_kpis(marks, 3.1)
        self.assertAlmostEqual(result["safe_play_start_ms"], 1000.0)   # 3.0 − 2.0 s held
        self.assertAlmostEqual(result["prebuffer_s"], 0.5)
        self.assertAlmostEqual(result["max_gap_s"], 1.5, places=3)
        self.assertEqual(result["coalesced_reads"], 1)
        # A 250 ms or 1 s buffer starts at 0.5 (1.0 s held), runs dry at 2.5, resumes at 3.0.
        self.assertEqual(result["stalls_at_250"], 1)
        self.assertAlmostEqual(result["stall_ms_at_250"], 500.0, places=1)
        self.assertEqual(result["stalls_at_1000"], 1)
        for column in soak_client.PLAYBACK_COLUMNS:
            self.assertIn(column, soak_client.CSV_COLUMNS)
            if column != "header_to_audio_ms":
                self.assertIn(column, result)
        self.assertEqual(soak_client.CSV_COLUMNS[-1], "error")
        self.assertEqual(soak_client.CSV_COLUMNS[-6], "is_probe")

    def test_stable_windows_are_assessed(self):
        fields = [
            "t_end_s", "worker", "i", "ttfa_ms", "total_ms", "bytes",
            "first_chunk_bytes", "audio_s", "stream_rtf", "is_probe", "class",
            "text_chars", "seed", "schedule", "error",
        ]
        with tempfile.TemporaryDirectory() as directory:
            request_path = os.path.join(directory, "requests.csv")
            with open(request_path, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                index = 0
                for window in range(3):
                    for cls in ("long", "medium", "short"):
                        for _ in range(2):
                            writer.writerow({
                                "t_end_s": 1 + window * 10 + index / 10,
                                "worker": 0, "i": index,
                                "ttfa_ms": 100, "total_ms": 500, "bytes": 24000,
                                "first_chunk_bytes": 4800, "audio_s": 0.5,
                                "stream_rtf": 0.8, "is_probe": 0, "class": cls,
                                "text_chars": 10, "seed": index,
                                "schedule": "stratified", "error": "",
                            })
                            index += 1
            resource_path = os.path.join(directory, "resources.csv")
            with open(resource_path, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=(
                    "elapsed_s", "rss_kb", "pss_kb", "anon_kb", "swap_kb",
                    "threads", "fds",
                ))
                writer.writeheader()
                for elapsed in (0, 10, 20, 30):
                    writer.writerow({
                        "elapsed_s": elapsed, "rss_kb": 100, "pss_kb": 90,
                        "anon_kb": 50, "swap_kb": 0, "threads": 4, "fds": 10,
                    })
            args = types.SimpleNamespace(
                warmup_s=0.0, window_s=10.0, min_per_window=6,
                min_per_class=2, min_per_class_p95=10, min_windows=3,
                max_mix_distance=0.20,
                max_ttfa_drift=30.0, max_stream_drift=20.0, strict_kpi=False,
            )
            self.assertEqual(soak_drift.analyze(directory, args), 0)
            summary = json.loads(Path(directory, "soak_summary.json").read_text())
            self.assertEqual(summary["status"], "PASS")
            self.assertEqual(summary["latency_kpi"]["status"], "PASS")
            self.assertTrue(all(item["status"] == "PARTIAL"
                                for item in summary["per_class"].values()))

    def test_intentional_503_is_not_an_inference_error(self):
        fields = list(soak_client.CSV_COLUMNS)
        with tempfile.TemporaryDirectory() as directory:
            request_path = os.path.join(directory, "requests.csv")
            with open(request_path, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                row = {field: "" for field in fields}
                row.update({"t_end_s": "1.0", "status": "503",
                            "outcome": "intentional_reject", "error": ""})
                writer.writerow(row)
            rows, errors, rejects = soak_drift.read_requests(request_path)
            self.assertEqual(rows, [])
            self.assertEqual(errors, [])
            self.assertEqual(len(rejects), 1)

    def test_profile_flag_check_uses_resolved_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            log_path = os.path.join(directory, "server.log")
            Path(log_path).write_text(
                "[FLAGS] v=1 QWEN_POOL_SPIN=4096 QWEN_DECODER_BATCH=0\n",
                encoding="utf-8",
            )
            args = types.SimpleNamespace(profile="test-profile")
            serve_soak.check_profile_flags(
                args, log_path,
                {"QWEN_POOL_SPIN": "4096", "QWEN_DECODER_BATCH": "0"},
            )

    def test_identity_accepts_tarball_source_revision(self):
        with tempfile.NamedTemporaryFile() as handle:
            handle.write(b"binary")
            handle.flush()
            result = serve_soak.identity(handle.name, "tree-revision")
        self.assertEqual(result["source_commit"], "tree-revision")


if __name__ == "__main__":
    unittest.main()
