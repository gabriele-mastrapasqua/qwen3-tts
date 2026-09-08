#!/usr/bin/env python3
"""Small admission falsifier: established streams, then one arrival.

The server is started by the caller.  This intentionally keeps the workload
separate from server startup and records accepted/rejected fifth requests
without turning the probe into a capacity benchmark.
"""
import argparse
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import serve_parallel_wave as SP


def request_one(port, text_row, seed, first_audio, barrier, out, lock, role,
                fifth_ref=None, speaker="ryan", language="English"):
    barrier.wait()
    cls, text = text_row
    body = json.dumps({"text": text, "speaker": speaker, "language": language,
                       "temperature": 0.0, "seed": seed}).encode()
    t0 = time.monotonic()
    t0_ms = t0 * 1000.0
    headers = {"Content-Type": "application/json"}
    if os.environ.get("QWEN_TTFA_TRACE"):
        headers["X-Qwen-F2-Client-Start-Monotonic-Ms"] = f"{t0_ms:.3f}"
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/tts/stream", data=body, headers=headers)
    rec = {"role": role, "class": cls, "seed": seed, "t0_mono_ms": t0_ms,
           "status": None, "error": None, "marks": []}
    if role == "fifth" and fifth_ref is not None:
        fifth_ref["t0_mono_ms"] = t0_ms
    try:
        with urllib.request.urlopen(req, timeout=180) as response:
            rec["status"] = int(response.status)
            rec["ttfb_ms"] = (time.monotonic() - t0) * 1000.0
            first = True
            total_bytes = 0
            while True:
                ch = response.read1(65536)
                if not ch:
                    break
                now = time.monotonic()
                if first:
                    first_audio.set()
                    first = False
                rec["marks"].append({"t_ms": (now - t0) * 1000.0,
                                     "abs_ms": now * 1000.0,
                                     "bytes": len(ch)})
                total_bytes += len(ch)
            rec["bytes"] = total_bytes
            rec["total_ms"] = (time.monotonic() - t0) * 1000.0
            if total_bytes:
                k = SP.stream_kpis(
                    [(m["t_ms"] / 1000.0, m["bytes"]) for m in rec["marks"]],
                    rec["total_ms"] / 1000.0)
                rec.update({"kpis": k, "audio_s": total_bytes / 2.0 / 24000.0})
    except urllib.error.HTTPError as exc:
        rec["status"] = int(exc.code)
        rec["error"] = f"HTTP {exc.code}"
        try:
            exc.read()
        except Exception:
            pass
        rec["total_ms"] = (time.monotonic() - t0) * 1000.0
    except Exception as exc:
        rec["error"] = repr(exc)
        rec["total_ms"] = (time.monotonic() - t0) * 1000.0
    with lock:
        out.append(rec)


def local_gap_stats(rec, inject_ms, window_ms=2000.0):
    marks = rec.get("marks", [])
    if len(marks) < 2:
        return {"before_max_gap_ms": None, "after_max_gap_ms": None,
                "window_max_gap_ms": None}
    gaps = []
    for a, b in zip(marks, marks[1:]):
        end = b["abs_ms"]
        start = a["abs_ms"]
        gaps.append((start, end, end - start))
    before = [g for _s, e, g in gaps if e <= inject_ms and e >= inject_ms - window_ms]
    after = [g for s, _e, g in gaps if s >= inject_ms and s <= inject_ms + window_ms]
    around = [g for s, e, g in gaps if e >= inject_ms - window_ms and s <= inject_ms + window_ms]
    return {"before_max_gap_ms": max(before) if before else None,
            "after_max_gap_ms": max(after) if after else None,
            "window_max_gap_ms": max(around) if around else None}


def run_rep(port, rows, base_seed, settle_ms, speaker, language, rep,
            established_count=4):
    out, lock = [], threading.Lock()
    events = [threading.Event() for _ in range(established_count)]
    barrier = threading.Barrier(established_count + 1)
    threads = []
    for i in range(established_count):
        t = threading.Thread(
            target=request_one,
            args=(port, rows[i % len(rows)], base_seed + rep * 100 + i,
                  events[i], barrier, out, lock, "established"),
            kwargs={"speaker": speaker, "language": language})
        threads.append(t)
        t.start()
    barrier.wait()
    first_ok = all(e.wait(timeout=120.0) for e in events)
    if first_ok:
        time.sleep(settle_ms / 1000.0)
    fifth_ref = {}
    fifth = threading.Thread(
        target=request_one,
        args=(port, rows[established_count % len(rows)], base_seed + rep * 100 + established_count,
              threading.Event(), threading.Barrier(1), out, lock, "fifth", fifth_ref),
        kwargs={"speaker": speaker, "language": language})
    inject_ms = time.monotonic() * 1000.0
    fifth_ref["scheduled_ms"] = inject_ms
    fifth.start()
    for t in threads:
        t.join()
    fifth.join()
    fifth_t0 = fifth_ref.get("t0_mono_ms", inject_ms)
    for rec in out:
        rec["rep"] = rep
        rec["established_ready_before_fifth"] = first_ok
        rec["fifth_start_mono_ms"] = fifth_t0
        if rec["role"] == "established":
            rec["local_gap"] = local_gap_stats(rec, fifth_t0)
    return {"rep": rep, "first_four_ready": first_ok,
            "fifth_scheduled_mono_ms": inject_ms, "fifth_start_mono_ms": fifth_t0,
            "requests": sorted(out, key=lambda r: (r["role"], r["seed"]))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--text-file", default="tests/load_texts_en.txt")
    ap.add_argument("--classes", default="long")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--settle-ms", type=float, default=500.0)
    ap.add_argument("--seed", type=int, default=9100)
    ap.add_argument("--speaker", default="ryan")
    ap.add_argument("--language", default="English")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    only = {x.strip() for x in a.classes.split(",") if x.strip()} or None
    rows = SP.load_texts(a.text_file, only)
    if len(rows) < 1:
        raise SystemExit("no text rows selected")
    if a.established_count < 1:
        raise SystemExit("--established-count must be positive")
    result = {"port": a.port, "repeats": a.repeats, "settle_ms": a.settle_ms,
              "established_count": a.established_count,
              "text_file": os.path.basename(a.text_file), "classes": sorted(only or []),
              "rows": len(rows), "repetitions": []}
    for rep in range(a.repeats):
        result["repetitions"].append(run_rep(
            a.port, rows, a.seed, a.settle_ms, a.speaker, a.language, rep,
            a.established_count))
        time.sleep(0.5)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    accepted = rejected = 0
    for wave in result["repetitions"]:
        for r in wave["requests"]:
            if r.get("status") == 200:
                accepted += 1
            elif r.get("status") == 503:
                rejected += 1
    print(json.dumps({"out": a.out, "accepted": accepted, "rejected_503": rejected,
                      "repetitions": len(result["repetitions"])}, sort_keys=True))


if __name__ == "__main__":
    main()
