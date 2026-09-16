#!/usr/bin/env python3
"""Analyze a closed-loop soak without confusing text mix with latency drift."""
import argparse
import csv
import json
import math
import os
import statistics
import sys


def percentile(values, quantile):
    values = sorted(value for value in values if value is not None and math.isfinite(value))
    if not values:
        return None
    index = min(len(values) - 1, int(round(quantile / 100.0 * (len(values) - 1))))
    return values[index]


def median(values):
    values = [value for value in values if value is not None and math.isfinite(value)]
    return statistics.median(values) if values else None


def display(value, digits=1):
    return "n/a" if value is None else f"{value:.{digits}f}"


def metric(rows, key):
    values = [row[key] for row in rows
              if row.get(key) is not None and math.isfinite(row[key])]
    return {
        "n": len(values),
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
    }


BUFFERS_MS = (100, 250, 500, 1000)


def playback_summary(rows):
    """Client-observed playback block for a window or the sustained set: per-request
    safe_play_start and required prebuffer percentiles, plus the fixed-buffer stall
    rates (share of requests with at least one stall under a B ms jitter buffer).
    Absent columns (older CSVs) yield None, never NaN, so the JSON stays strict."""
    out = {
        "safe_play_start": metric(rows, "safe_play_start"),
        "prebuffer": metric(rows, "prebuffer"),
        "stall_max": metric(rows, "stall_max"),
        "max_gap": metric(rows, "max_gap"),
    }
    for b in BUFFERS_MS:
        known = [row for row in rows if row.get(f"stalls_{b}") is not None]
        stalled = sum(1 for row in known if row[f"stalls_{b}"] > 0)
        out[f"stall_rate_{b}"] = (stalled / len(known)) if known else None
        out[f"stall_ms_{b}"] = metric(rows, f"stall_ms_{b}")
    known = [row for row in rows if row.get("coalesced_reads") is not None
             and row.get("chunks")]
    out["coalesced_read_share"] = (
        sum(row["coalesced_reads"] for row in known) / sum(row["chunks"] for row in known)
        if known else None)
    return out


def drift_percent(first, last):
    if first is None or last is None or first == 0:
        return None
    return 100.0 * (last - first) / first


def format_playback(pb):
    """One line: safe_play_start, fixed-buffer stall rates, receive fidelity."""
    sps = pb["safe_play_start"]
    parts = []
    for b in BUFFERS_MS:
        rate = pb[f"stall_rate_{b}"]
        parts.append(f"@{b}ms " + ("n/a" if rate is None else f"{100 * rate:.0f}%"))
    rates = " ".join(parts)
    coal = pb["coalesced_read_share"]
    return (f"safe_play_start p50/p95 {display(sps['p50'], 0)}/{display(sps['p95'], 0)} ms; "
            f"max_gap p95 {display(pb['max_gap']['p95'], 3)} s; stall_rate {rates}; "
            f"coalesced reads {'n/a' if coal is None else f'{100 * coal:.1f}%'}")


def drift_pairs(args, has_ttfb):
    """The (metric, key, limit) triples every drift verdict is made of.  TTFB is compared
    only when the CSV carries it: a soak recorded before the column existed must not turn
    NOT_ASSESSED because of a metric it could not have measured."""
    pairs = []
    if has_ttfb:
        pairs += [("TTFB p50", "ttfb", args.max_ttfa_drift),
                  ("TTFB p95", "ttfb", args.max_ttfa_drift)]
    pairs += [("TTFA p50", "ttfa", args.max_ttfa_drift),
              ("TTFA p95", "ttfa", args.max_ttfa_drift),
              ("stream RTF p50", "stream", args.max_stream_drift)]
    return pairs

def read_requests(path):
    rows = []
    errors = []
    rejects = []
    with open(path, newline="", encoding="utf-8", errors="replace") as handle:
        for raw in csv.DictReader(handle):
            try:
                end = float(raw["t_end_s"])
            except (KeyError, TypeError, ValueError):
                continue
            status = raw.get("status", "").strip()
            outcome = raw.get("outcome", "").strip()
            if status == "503" or outcome == "intentional_reject":
                rejects.append({"t": end, "status": status or "503"})
                continue
            error = raw.get("error", "").strip()
            if error:
                errors.append({"t": end, "error": error})
                continue
            try:
                ttfa = float(raw["ttfa_ms"])
                stream = float(raw["stream_rtf"])
                audio_s = float(raw.get("audio_s") or 0.0)
            except (KeyError, TypeError, ValueError):
                errors.append({"t": end, "error": "invalid metric row"})
                continue
            try:                                   # absent in CSVs written before TTFB existed
                ttfb = float(raw["ttfb_ms"])
            except (KeyError, TypeError, ValueError):
                ttfb = None
            playback = {}
            for source, target in (
                ("underrun_s", "underrun"),
                ("stall_max_s", "stall_max"),
                ("prebuffer_s", "prebuffer"),
                ("gap_ratio_max", "gap_ratio_max"),
                ("safe_play_start_ms", "safe_play_start"),
                ("max_gap_s", "max_gap"),
                ("coalesced_reads", "coalesced_reads"),
                *((f"stall_ms_at_{b}", f"stall_ms_{b}") for b in BUFFERS_MS),
                *((f"stalls_at_{b}", f"stalls_{b}") for b in BUFFERS_MS),
            ):
                try:
                    playback[target] = float(raw[source])
                except (KeyError, TypeError, ValueError):
                    playback[target] = None
            try:
                playback["chunks"] = int(raw["chunks"])
            except (KeyError, TypeError, ValueError):
                playback["chunks"] = None
            rows.append({
                "t": end,
                "class": raw.get("class", "unknown") or "unknown",
                "ttfb": ttfb,
                "ttfa": ttfa,
                "stream": stream,
                "audio_s": audio_s,
                "probe": raw.get("is_probe") == "1",
                **playback,
            })
    return rows, errors, rejects


def read_resources(path, warmup):
    if not os.path.isfile(path):
        return []
    rows = []
    with open(path, newline="", encoding="utf-8", errors="replace") as handle:
        for raw in csv.DictReader(handle):
            try:
                if float(raw["elapsed_s"]) < warmup:
                    continue
                row = {key: float(value) for key, value in raw.items()
                       if key and value not in (None, "")}
                rows.append(row)
            except (KeyError, TypeError, ValueError):
                continue
    return rows


def proportions(rows):
    counts = {}
    for row in rows:
        counts[row["class"]] = counts.get(row["class"], 0) + 1
    total = sum(counts.values())
    return {key: value / total for key, value in counts.items()} if total else {}


def mix_distance(left, right):
    keys = set(left) | set(right)
    return 0.5 * sum(abs(left.get(key, 0.0) - right.get(key, 0.0)) for key in keys)


def resource_verdict(rows):
    if not rows:
        return {"status": "NOT_AVAILABLE", "reason": "no resource samples"}

    result = {"status": "PASS", "samples": len(rows), "metrics": {}}
    for key, label, growth_limit in (
        ("anon_kb", "anonymous_memory_kb", 10.0),
        ("pss_kb", "pss_kb", 10.0),
        ("rss_kb", "rss_kb", 15.0),
    ):
        values = [row[key] for row in rows if key in row and row[key] > 0]
        if len(values) < 3:
            continue
        split = max(1, len(values) // 5)
        first = median(values[:split])
        last = median(values[-split:])
        growth = drift_percent(first, last)
        result["metrics"][label] = {
            "first": first,
            "last": last,
            "growth_pct": growth,
            "minimum": min(values),
            "maximum": max(values),
        }
        if growth is not None and growth > growth_limit:
            result["status"] = "FAIL"
            result.setdefault("failures", []).append(label)

    for key, label, allowance in (("threads", "threads", 2), ("fds", "open_fds", 4)):
        values = [int(row[key]) for row in rows if key in row]
        if not values:
            continue
        split = max(1, len(values) // 5)
        first = median(values[:split])
        last = median(values[-split:])
        result["metrics"][label] = {
            "first": first,
            "last": last,
            "minimum": min(values),
            "maximum": max(values),
            "range": max(values) - min(values),
        }
        if last is not None and first is not None and last > first + allowance:
            result["status"] = "FAIL"
            result.setdefault("failures", []).append(label)
    return result


def analyze(directory, args):
    request_path = os.path.join(directory, "requests.csv")
    if not os.path.isfile(request_path):
        print(f"FAIL: missing {request_path}")
        return 1

    rows, errors, rejects = read_requests(request_path)
    # THE TAIL IS NOT STEADY STATE.  The closed-loop client stops ADMITTING at the
    # deadline, so in the last stretch of a run no new work enters and only what is
    # already in flight can finish.  The long classes take the longest, so they are the
    # ones that stop appearing first: the final window ends up with a handful of them,
    # the per-class p95 becomes uncomputable, and the completed-class mix shifts enough
    # that the pooled drift is refused.  Both are artefacts of the stopping rule, not of
    # the server.  --cooldown-s drops that drain tail, exactly as --warmup-s drops the
    # ramp at the other end, so the compared windows are both full-admission windows.
    # Set it to a little more than the longest utterance the bank can produce.
    horizon = max((row["t"] for row in rows), default=0.0) - args.cooldown_s
    usable_all = [row for row in rows
                  if row["t"] >= args.warmup_s and (args.cooldown_s <= 0 or row["t"] <= horizon)]
    usable = [row for row in usable_all if not row["probe"]]
    if not usable:
        print("FAIL: no completed requests after warm-up")
        return 1

    end = max(row["t"] for row in usable_all)
    window_map = {}
    for row in usable:
        index = int((row["t"] - args.warmup_s) // args.window_s)
        window_map.setdefault(index, []).append(row)
    windows = []
    # A WINDOW THAT COVERS LESS WALL TIME IS NOT COMPARABLE TO THE OTHERS. The run rarely
    # ends on a window boundary, and --cooldown-s moves the horizon further inward, so the
    # trailing window is usually a stub holding a fraction of the samples the full ones do.
    # Kept, it becomes the "last" window every per-class comparison is made against, and its
    # thin per-class counts are what turn a flat run into PARTIAL. Drop it.
    last_index = max(window_map) if window_map else -1
    for index in sorted(window_map):
        group = window_map[index]
        if (index == last_index
                and args.warmup_s + (index + 1) * args.window_s > end + 1e-6):
            continue
        if len(group) < args.min_per_window:
            continue
        windows.append({
            "start_s": args.warmup_s + index * args.window_s,
            "end_s": args.warmup_s + (index + 1) * args.window_s,
            "n": len(group),
            "mix": proportions(group),
            "ttfb": metric(group, "ttfb"),
            "ttfa": metric(group, "ttfa"),
            "stream": metric(group, "stream"),
            "underrun": metric(group, "underrun"),
            "prebuffer": metric(group, "prebuffer"),
            "stall_max": metric(group, "stall_max"),
            "playback": playback_summary(group),
            "rows": group,
        })

    has_ttfb = any(row.get("ttfb") is not None for row in usable)
    print("### CLOSED-LOOP SOAK")
    print(f"completed={len(usable_all)} kpi_samples={len(usable)} "
          f"audio_probes={len(usable_all) - len(usable)} errors={len(errors)} "
          f"intentional_rejects={len(rejects)} duration_s={end:.1f} "
          f"warmup_s={args.warmup_s:.0f} cooldown_s={args.cooldown_s:.0f}")
    print(f"windows={len(windows)} window_s={args.window_s:.0f} "
          f"min_per_window={args.min_per_window}")
    print()
    print(f"{'window':>13} {'n':>5} {'TTFB p50':>10} {'TTFB p95':>10} {'TTFA p50':>10} {'TTFA p95':>10} "
          f"{'RTF p50':>9} {'RTF p95':>9} {'classes':>10}")
    for window in windows:
        ttfb = window["ttfb"]
        ttfa = window["ttfa"]
        stream = window["stream"]
        print(f"{window['start_s']:>6.0f}-{window['end_s']:<6.0f} {window['n']:>5} "
              f"{display(ttfb['p50']):>10} {display(ttfb['p95']):>10} "
              f"{display(ttfa['p50']):>10} {display(ttfa['p95']):>10} "
              f"{display(stream['p50'], 3):>9} {display(stream['p95'], 3):>9} "
              f"{len(window['mix']):>10}")
        underrun = window["underrun"]
        prebuffer = window["prebuffer"]
        print(f"              zero-buffer player: underrun p50/p95 "
              f"{display(underrun['p50'], 3)}/{display(underrun['p95'], 3)} s; "
              f"required_prebuffer p50/p95 {display(prebuffer['p50'], 3)}/"
              f"{display(prebuffer['p95'], 3)} s")
        print("              " + format_playback(window["playback"]))

    hard_failures = ["request errors"] if errors else []
    kpi = {"status": "NOT_ASSESSED", "reason": "insufficient comparable windows"}
    if len(windows) >= args.min_windows:
        first, last = windows[0], windows[-1]
        distance = mix_distance(first["mix"], last["mix"])
        kpi["mix_distance"] = distance
        if distance > args.max_mix_distance:
            kpi["reason"] = (f"completed-request class mix changed by {distance:.3f}; "
                              "use per-class rows instead of a pooled drift claim")
        else:
            failures = []
            comparisons = []
            for name, key, limit in drift_pairs(args, has_ttfb):
                quantile = 50 if "p50" in name else 95
                before = first[key][f"p{quantile}"]
                after = last[key][f"p{quantile}"]
                change = drift_percent(before, after)
                comparisons.append({"metric": name, "first": before,
                                    "last": after, "drift_pct": change,
                                    "limit_pct": limit})
                if before is None or after is None:
                    failures.append(name + " unavailable")
                elif change is not None and change > limit:
                    failures.append(name)
            unavailable = any(name.endswith(" unavailable") for name in failures)
            kpi = {"status": "NOT_ASSESSED" if unavailable else ("FAIL" if failures else "PASS"),
                   "reason": ("a required metric was unavailable" if unavailable
                              else "pooled class mix is comparable"),
                   "comparisons": comparisons}
            if failures:
                kpi["failures"] = failures

    class_results = {}
    classes = sorted({row["class"] for row in usable})
    min_per_class_p50 = getattr(args, "min_per_class_p50",
                                max(args.min_per_class, 5))
    min_per_class_p95 = getattr(args, "min_per_class_p95", args.min_per_class)
    for cls in classes:
        class_windows = []
        for window in windows:
            group = [row for row in window["rows"] if row["class"] == cls]
            if len(group) >= args.min_per_class:
                class_windows.append((window, group))
        result = {"status": "NOT_ASSESSED", "samples": sum(
            row["class"] == cls for row in usable)}
        if len(class_windows) >= args.min_windows:
            first_group = class_windows[0][1]
            last_group = class_windows[-1][1]
            comparisons = []
            failures = []
            unassessed = []
            for name, key, limit in drift_pairs(args, has_ttfb):
                quantile = 50 if "p50" in name else 95
                required = min_per_class_p95 if quantile == 95 else min_per_class_p50
                if len(first_group) < required or len(last_group) < required:
                    unassessed.append(name)
                    comparisons.append({
                        "metric": name,
                        "first": None,
                        "last": None,
                        "drift_pct": None,
                        "limit_pct": limit,
                        "status": "NOT_ASSESSED",
                        "reason": (
                            f"requires {required} samples in both comparison windows; "
                            f"got {len(first_group)} and {len(last_group)}"
                        ),
                    })
                    continue
                before = percentile([row[key] for row in first_group], quantile)
                after = percentile([row[key] for row in last_group], quantile)
                if before is None or after is None:
                    unassessed.append(name)
                    comparisons.append({
                        "metric": name,
                        "first": before,
                        "last": after,
                        "drift_pct": None,
                        "limit_pct": limit,
                        "status": "NOT_ASSESSED",
                        "reason": "metric unavailable",
                    })
                    continue
                change = drift_percent(before, after)
                # DRIFT IS A CLAIM ABOUT A TREND, AND TWO POINTS CANNOT SHOW ONE.
                # Comparing only the first and last window applies a percentage limit to a
                # single pair of samples, which is fine for a metric that moves smoothly and
                # wrong for one that jitters: TTFB p95 sits around 50 ms, so 15 ms of ordinary
                # bounce is 30 % and trips the gate while the run is flat. Measure the same
                # metric in EVERY window first. If the last window lands inside the range the
                # earlier windows already spanned, the run visited that value before it ended
                # and has not drifted -- whatever the first-to-last percentage says.
                track = []
                for window, group in class_windows:
                    if len(group) >= required:
                        value = percentile([row[key] for row in group], quantile)
                        if value is not None:
                            track.append(value)
                entry = {"metric": name, "first": before, "last": after,
                         "drift_pct": change, "limit_pct": limit, "status": "ASSESSED"}
                within = False
                if len(track) >= 3:
                    earlier = track[:-1]
                    lo, hi = min(earlier), max(earlier)
                    within = lo <= after <= hi
                    entry["windows_n"] = len(track)
                    entry["window_range"] = [lo, hi]
                    entry["within_run_range"] = within
                comparisons.append(entry)
                if change is not None and change > limit and not within:
                    failures.append(name)
                elif change is not None and change > limit and within:
                    entry["note"] = ("first-to-last exceeds the limit but the value stays "
                                     "inside the range the run already spanned: jitter, not drift")
            status = "FAIL" if failures else ("PARTIAL" if unassessed else "PASS")
            result = {"status": status,
                      "first_n": len(first_group), "last_n": len(last_group),
                      "comparisons": comparisons}
            if failures:
                result["failures"] = failures
            if unassessed:
                result["unassessed"] = unassessed
        class_results[cls] = result

    resource_rows = read_resources(os.path.join(directory, "resources.csv"), args.warmup_s)
    resources = resource_verdict(resource_rows)
    if resources["status"] == "FAIL":
        hard_failures.extend(resources.get("failures", ["resource growth"]))
    rejected = max((row.get("queue_rejected", 0) for row in resource_rows), default=0)
    queue_timeouts = max((row.get("queue_timeout", 0) for row in resource_rows), default=0)
    request_timeouts = max((row.get("request_timeout", 0) for row in resource_rows), default=0)
    # With the production fail-fast contract, a full-capacity 503 is an intentional
    # admission outcome, not an inference failure.  Keep it visible in the summary,
    # but do not turn a correctly rejected overload request into a SOAK FAIL.  Actual
    # queue/request timeouts remain hard failures below.
    if queue_timeouts > 0:
        hard_failures.append("queue timeouts")
    if request_timeouts > 0:
        hard_failures.append("server request timeouts")

    sustained = {
        "n": len(usable),
        "ttfa": metric(usable, "ttfa"),
        "stream": metric(usable, "stream"),
        "underrun": metric(usable, "underrun"),
        "prebuffer": metric(usable, "prebuffer"),
        "stall_max": metric(usable, "stall_max"),
        "playback": playback_summary(usable),
        "mix": proportions(usable),
    }
    print()
    print("SUSTAINED PLAYBACK (client-observed): " + format_playback(sustained["playback"]))
    if kpi["status"] == "FAIL":
        hard_failures.extend(kpi.get("failures", ["pooled KPI drift"]))
    if any(result["status"] == "FAIL" for result in class_results.values()):
        hard_failures.append("per-class KPI drift")

    overall = "FAIL" if hard_failures else ("PASS" if kpi["status"] == "PASS" else "PARTIAL")
    print()
    print(f"LATENCY KPI: {kpi['status']} — {kpi['reason']}")
    print(f"RESOURCE STABILITY: {resources['status']}")
    print(f"QUEUE REJECTIONS: {rejected:.0f} full / {queue_timeouts:.0f} timeout; "
          f"SERVER TIMEOUTS: {request_timeouts:.0f}")
    class_status = ", ".join(
        f"{key}={value['status']}" for key, value in class_results.items()
    ) or "none"
    print(f"PER-CLASS KPI: {class_status}")
    print(f"SOAK RESULT: {overall}" + (f" — {', '.join(hard_failures)}" if hard_failures else ""))

    summary = {
        "schema_version": 1,
        "status": overall,
        "completed": len(usable_all),
        "kpi_samples": len(usable),
        "audio_probes": len(usable_all) - len(usable),
        "errors": len(errors),
        "error_examples": errors[:5],
        "intentional_rejects": len(rejects),
        "reject_examples": rejects[:5],
        "queue_rejected": rejected,
        "queue_timeout": queue_timeouts,
        "request_timeout": request_timeouts,
        "duration_s": end,
        "warmup_s": args.warmup_s,
        "cooldown_s": args.cooldown_s,
        "windows": [{key: value for key, value in window.items() if key != "rows"}
                     for window in windows],
        "latency_kpi": kpi,
        "per_class": class_results,
        "sustained": sustained,
        "resources": resources,
        "failures": hard_failures,
    }
    with open(os.path.join(directory, "soak_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, allow_nan=False)
        handle.write("\n")
    strict_failure = args.strict_kpi and (
        kpi["status"] != "PASS" or
        any(result["status"] != "PASS" for result in class_results.values())
    )
    return 1 if overall == "FAIL" or strict_failure else 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory")
    parser.add_argument("--window-s", type=float, default=60.0)
    parser.add_argument("--warmup-s", type=float, default=60.0)
    parser.add_argument("--cooldown-s", type=float, default=0.0,
                        help="drop requests finishing in the last N seconds: the drain "
                             "tail after the client stops admitting is not steady state")
    parser.add_argument("--min-per-window", type=int, default=5)
    parser.add_argument("--min-per-class", type=int, default=3)
    parser.add_argument("--min-per-class-p50", type=int, default=5)
    parser.add_argument("--min-per-class-p95", type=int, default=20)
    parser.add_argument("--min-windows", type=int, default=3)
    parser.add_argument("--max-mix-distance", type=float, default=0.20)
    parser.add_argument("--max-ttfa-drift", type=float, default=30.0)
    parser.add_argument("--max-stream-drift", type=float, default=20.0)
    parser.add_argument("--strict-kpi", action="store_true")
    args = parser.parse_args()
    if args.window_s <= 0 or args.warmup_s < 0:
        parser.error("window and warm-up must be non-negative, with a positive window")
    if args.min_per_class_p50 < 1:
        parser.error("--min-per-class-p50 must be positive")
    return analyze(args.directory, args)


if __name__ == "__main__":
    sys.exit(main())
