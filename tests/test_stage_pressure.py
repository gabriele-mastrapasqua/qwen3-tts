#!/usr/bin/env python3
"""Offline checks for the QWEN_STAGE_TRACE log summariser."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import stage_pressure as S  # noqa: E402


def check(name, condition):
    print(f"{'ok  ' if condition else 'FAIL'} {name}")
    return condition


lines = [
    "[STAGE] v=1 pid=1 seq=1 clock=CLOCK_MONOTONIC domain=S start_ms=1000.0 end_ms=1060.0 active=2 step=2 admit_ms=1.0 prefill_ms=0.0 head_ms=2.0 sample_ms=1.0 cp_ms=5.0 decode_ms=20.0 talker_ms=30.0 output_ms=0.5 queue_wait_ms=0.0 serial_ms=1.0 wall_ms=60.0 dec_calls=1 dec_group_max=2 dec_frames=4 dec_ragged=1 dec_per_item=0 dec_external=0\n",
    "noise\n",
    "[STAGE] v=1 pid=1 seq=2 clock=CLOCK_MONOTONIC domain=S start_ms=1060.0 end_ms=1092.2 active=1 step=1 admit_ms=0.0 prefill_ms=0.0 head_ms=1.0 sample_ms=1.0 cp_ms=4.0 decode_ms=10.0 talker_ms=15.0 output_ms=0.2 queue_wait_ms=0.0 serial_ms=1.0 wall_ms=32.2 dec_calls=1 dec_group_max=1 dec_frames=1 dec_ragged=0 dec_per_item=1 dec_external=0\n",
]
rows = [r for r in (S.parse_line(x) for x in lines) if r]
result = S.summarise(rows)
ok = True
ok &= check("stage lines parse", len(rows) == 2)
ok &= check("active groups are retained", set(result["by_active"]) == {"1", "2"})
ok &= check("ragged and per-item shares are distinct", result["decoder"]["ragged_row_share"] == 0.5 and result["decoder"]["per_item_row_share"] == 0.5)
ok &= check("no client causality is claimed", "client-gap" in result["interpretation"])
joined = S.join_gaps(rows, [{"start_ms": 1030.0, "end_ms": 1080.0, "gap_ms": 50.0}])
ok &= check("absolute iteration bounds support offline overlap",
            joined["count"] == 1 and joined["gaps"][0]["iterations"] == 2)
print("all checks passed" if ok else "FAILED")
raise SystemExit(0 if ok else 1)
