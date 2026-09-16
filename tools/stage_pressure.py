#!/usr/bin/env python3
"""Summarise the default-off QWEN_STAGE_TRACE diagnostic.

This is intentionally a log-only helper.  ``[STAGE]`` timestamps and client playback
timestamps live in different processes, so this tool never claims that a long engine
iteration caused a particular client gap.  It reports phase pressure and candidate long
iterations; a later campaign helper may join it to client events using an explicit clock
alignment.
"""
import argparse
import json
import re
import statistics
import sys


STAGE_RE = re.compile(r"\[STAGE\]\s+(.*)$")
INT_FIELDS = {"pid", "seq", "active", "step", "dec_calls", "dec_group_max", "dec_frames",
              "dec_ragged", "dec_per_item", "dec_external"}
FLOAT_FIELDS = {"start_ms", "end_ms", "admit_ms", "prefill_ms", "head_ms", "sample_ms", "cp_ms", "decode_ms",
                "talker_ms", "output_ms", "queue_wait_ms", "serial_ms", "wall_ms"}
PHASES = ("admit_ms", "prefill_ms", "head_ms", "sample_ms", "cp_ms", "decode_ms",
          "talker_ms", "output_ms", "queue_wait_ms", "serial_ms")


def parse_line(line):
    m = STAGE_RE.search(line)
    if not m:
        return None
    row = {}
    for token in m.group(1).split():
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        try:
            if key in INT_FIELDS:
                row[key] = int(value)
            elif key in FLOAT_FIELDS:
                row[key] = float(value)
            else:
                row[key] = value
        except ValueError:
            row[key] = value
    return row if "wall_ms" in row else None


def load(paths):
    rows = []
    for path in paths:
        stream = sys.stdin if path == "-" else open(path, encoding="utf-8", errors="replace")
        try:
            for line in stream:
                row = parse_line(line)
                if row:
                    rows.append(row)
        finally:
            if stream is not sys.stdin:
                stream.close()
    return rows


def load_client_gaps(paths, minimum_ms=250.0):
    """Derive receive gaps from JSONL rows sharing the host monotonic clock.

    Rows without ``t_send_mono_ms`` are ignored rather than assigned a guessed clock
    offset. The result is receive-observed overlap evidence, not server causality.
    """
    gaps = []
    for path in paths:
        with open(path, encoding="utf-8", errors="replace") as stream:
            for line_no, line in enumerate(stream, 1):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                send = row.get("t_send_mono_ms")
                marks = row.get("marks")
                if not isinstance(send, (int, float)) or not isinstance(marks, list):
                    continue
                previous = None
                for mark in marks:
                    if not isinstance(mark, list) or not mark or not isinstance(mark[0], (int, float)):
                        continue
                    current = float(send) + float(mark[0]) * 1000.0
                    if previous is not None and current - previous >= minimum_ms:
                        gaps.append({"path": path, "line": line_no, "start_ms": previous,
                                     "end_ms": current, "gap_ms": current - previous,
                                     "idx": row.get("idx"), "cls": row.get("cls")})
                    previous = current
    return gaps


def join_gaps(rows, gaps):
    joined = []
    for gap in gaps:
        overlap = [r for r in rows if "start_ms" in r and "end_ms" in r
                   and r["start_ms"] < gap["end_ms"] and r["end_ms"] > gap["start_ms"]]
        phase_ms = {p: sum(float(r.get(p, 0.0)) for r in overlap) for p in PHASES}
        dominant = max(phase_ms, key=phase_ms.get) if overlap and any(phase_ms.values()) else None
        item = dict(gap)
        item.update({"iterations": len(overlap), "phase_ms": phase_ms,
                     "dominant_phase": dominant,
                     "active": sorted({r.get("active") for r in overlap if "active" in r}),
                     "decoder_group_max": max((r.get("dec_group_max", 0) for r in overlap), default=0)})
        joined.append(item)
    return {
        "minimum_gap_ms": min((g["gap_ms"] for g in gaps), default=None),
        "count": len(joined),
        "gaps": joined,
        "dominant_histogram": {p: sum(1 for g in joined if g["dominant_phase"] == p) for p in PHASES},
        "interpretation": "receive-gap overlap only; not server causality without validated clock/alignment",
    }


def percentile(values, p):
    if not values:
        return None
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    pos = (len(values) - 1) * p / 100.0
    lo, hi = int(pos), min(int(pos) + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


def stats(rows, field):
    values = [float(r[field]) for r in rows if field in r]
    return {"n": len(values), "p50_ms": percentile(values, 50),
            "p95_ms": percentile(values, 95), "mean_ms": statistics.mean(values) if values else None}


def summarise(rows, long_ms=250.0, gaps=None):
    long_rows = [r for r in rows if r.get("wall_ms", 0.0) >= long_ms]
    phase_sums = {p: sum(float(r.get(p, 0.0)) for r in long_rows) for p in PHASES}
    total = sum(phase_sums.values())
    dominant = sorted(phase_sums.items(), key=lambda x: x[1], reverse=True)
    groups = {}
    for active in sorted({r.get("active", -1) for r in rows}):
        subset = [r for r in rows if r.get("active", -1) == active]
        groups[str(active)] = {
            "count": len(subset),
            "wall": stats(subset, "wall_ms"),
            "decode": stats(subset, "decode_ms"),
            "talker": stats(subset, "talker_ms"),
            "decoder_group_histogram": {
                str(g): sum(1 for r in subset if r.get("dec_group_max") == g)
                for g in sorted({r.get("dec_group_max", 0) for r in subset})
            },
        }
    result = {
        "format": "qwen-stage-pressure-v1",
        "rows": len(rows),
        "clock": "CLOCK_MONOTONIC (engine process)",
        "long_iteration_threshold_ms": long_ms,
        "long_iterations": len(long_rows),
        "long_iteration_share": (len(long_rows) / len(rows)) if rows else None,
        "all": {p[:-3]: stats(rows, p) for p in ("wall_ms",) + PHASES},
        "by_active": groups,
        "decoder": {
            "ragged_row_share": (sum(1 for r in rows if r.get("dec_ragged", 0)) / len(rows)) if rows else None,
            "per_item_row_share": (sum(1 for r in rows if r.get("dec_per_item", 0)) / len(rows)) if rows else None,
            "external_row_share": (sum(1 for r in rows if r.get("dec_external", 0)) / len(rows)) if rows else None,
        },
        "long_iteration_phase_ms": phase_sums,
        "long_iteration_dominant_phase": dominant[0][0] if dominant and dominant[0][1] > 0 else None,
        "long_iteration_phase_share": ({p: v / total for p, v in phase_sums.items()} if total else {}),
        "interpretation": "candidate engine pressure only; not client-gap causality without clock alignment",
    }
    if gaps is not None:
        result["gap_join"] = join_gaps(rows, gaps)
    return result


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+", help="server stderr/stdout logs; use - for stdin")
    ap.add_argument("--long-ms", type=float, default=250.0,
                    help="candidate long engine iteration threshold (default: 250)")
    ap.add_argument("--client-jsonl", action="append", default=[],
                    help="optional wave JSONL with t_send_mono_ms/marks for receive-gap overlap")
    ap.add_argument("--gap-ms", type=float, default=250.0,
                    help="minimum client receive gap to join (default: 250)")
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = ap.parse_args(argv)
    gaps = load_client_gaps(args.client_jsonl, args.gap_ms) if args.client_jsonl else None
    result = summarise(load(args.logs), args.long_ms, gaps)
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print("STAGE PRESSURE (diagnostic; no client-gap causality)")
        print(f"rows={result['rows']} long_iterations={result['long_iterations']} "
              f"long_share={result['long_iteration_share']}")
        print("phase p50/p95 ms:")
        for name, value in result["all"].items():
            print(f"  {name:12s} {value['p50_ms']!s:>8} {value['p95_ms']!s:>8}")
        print("long dominant phase:", result["long_iteration_dominant_phase"] or "UNKNOWN")
        if result.get("gap_join") is not None:
            print(f"receive gaps >= {args.gap_ms:.0f} ms: {result['gap_join']['count']} "
                  "(overlap only; not causality)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
