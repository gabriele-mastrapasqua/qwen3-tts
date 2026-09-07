#!/usr/bin/env python3
"""Run one closed-loop streaming conversation for a soak run."""
import argparse
import csv
import json
import os
import random
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import playback_sim  # noqa: E402


def load_texts(path, wanted=None):
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            parts = [item.strip() for item in line.split("\t")]
            if len(parts) >= 8:
                item = (parts[1], parts[-1])
            elif len(parts) > 1:
                item = (parts[0], parts[-1])
            else:
                item = ("medium", parts[0])
            if wanted is None or item[0] in wanted:
                rows.append(item)
    return rows


def make_picker(rows, worker, seed, schedule):
    if schedule == "ordered":
        return lambda index: rows[(worker * 7 + index) % len(rows)]

    groups = {}
    for cls, text in rows:
        groups.setdefault(cls, []).append(text)
    classes = sorted(groups)
    rng = random.Random(seed + worker)
    for values in groups.values():
        rng.shuffle(values)

    def pick(index):
        cls = classes[(worker + index) % len(classes)]
        values = groups[cls]
        text = values[((worker + index) // len(classes)) % len(values)]
        return cls, text

    return pick


PLAYBACK_COLUMNS = (
    "safe_play_start_ms", "header_to_audio_ms", "max_gap_s", "coalesced_reads",
    "stall_ms_at_100", "stalls_at_100", "stall_ms_at_250", "stalls_at_250",
    "stall_ms_at_500", "stalls_at_500", "stall_ms_at_1000", "stalls_at_1000",
)


def stream_kpis(marks, total_s):
    """Return streaming and CLIENT-OBSERVED playback metrics for one response.

    ``marks`` contains ``(seconds_since_send, bytes_received[, seconds_blocked_in_read])``.
    Every definition lives in ``tests/playback_sim.py`` (single source): the legacy keys
    are kept for the CSV consumers, ``prebuffer_s`` is the required prebuffer, and the
    fixed-buffer fields simulate a real jitter buffer.  A mark is the return of the
    client's chunked read, so a late reader can coalesce arrivals; ``coalesced_reads``
    counts reads that returned already-queued data and bounds that effect.
    """
    k = playback_sim.timeline_kpis(marks, total_s)
    out = {
        "stream_rtf": k["stream_rtf"],
        "underrun_s": k["underrun_total_s"],
        "stall_max_s": k["stall_max_s"],
        "prebuffer_s": k["required_prebuffer_s"],
        "gap_ratio_max": k["gap_ratio_max"],
        "chunks": k["chunks"],
        "safe_play_start_ms": k["safe_play_start_s"] * 1000.0,
        "max_gap_s": k["max_gap_s"],
        "coalesced_reads": k["coalesced_chunks"],
    }
    for b in playback_sim.BUFFERS_MS:
        out[f"stall_ms_at_{b}"] = k[f"stall_ms@{b}"]
        out[f"stalls_at_{b}"] = k[f"stalls@{b}"]
    return out


CSV_COLUMNS = (
    "t_end_s", "worker", "i", "ttfb_ms", "ttfa_ms", "total_ms", "bytes",
    "first_chunk_bytes", "audio_s", "stream_rtf", "underrun_s",
    "stall_max_s", "prebuffer_s", "gap_ratio_max", "chunks",
    *PLAYBACK_COLUMNS,
    "is_probe", "class", "text_chars", "seed", "schedule", "error",
)


def one(port, text, speaker, language, seed, temperature, out_path, timeout):
    body = json.dumps({
        "text": text,
        "speaker": speaker,
        "language": language,
        "seed": seed,
        "temperature": temperature,
    }).encode()
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/tts/stream",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    started = time.time()
    received = first = 0
    first_at = None
    header_at = None
    marks = []
    handle = open(out_path, "wb") if out_path else None
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            # TTFB: urlopen returns once the status line + headers are parsed.  On the
            # batched server the header is written together with the first audio chunk,
            # so TTFB and TTFA are the same event today; they are stamped independently.
            header_at = time.time() - started
            while True:
                # read1 returns at most ONE HTTP chunk (or part of one) and does not wait
                # for that chunk's trailing CRLF.  ``blocked`` is the time spent inside the
                # call: a near-zero value means the data was already queued (coalesced
                # arrival or split chunk), so the mark overstates that chunk's lateness.
                t_call = time.time()
                chunk = response.read1(1 << 16)
                t_ret = time.time()
                if not chunk:
                    break
                if first_at is None:
                    first_at = t_ret - started
                    first = len(chunk)
                marks.append((t_ret - started, len(chunk), t_ret - t_call))
                received += len(chunk)
                if handle:
                    handle.write(chunk)
    except Exception as error:
        return None, str(error)
    finally:
        if handle:
            handle.close()

    total = time.time() - started
    audio_s = received / 2.0 / 24000.0
    kpis = stream_kpis(marks, total)
    kpis["header_to_audio_ms"] = ((first_at or 0.0) - (header_at or 0.0)) * 1000.0
    return {
        "ttfb_ms": (header_at or 0.0) * 1000.0,
        "ttfa_ms": (first_at or 0.0) * 1000.0,
        "total_ms": total * 1000.0,
        "bytes": received,
        "first_chunk_bytes": first,
        "audio_s": audio_s,
        **kpis,
    }, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--worker", type=int, required=True)
    parser.add_argument("--t0", type=float, required=True)
    parser.add_argument("--deadline", type=float, required=True)
    parser.add_argument("--bank", required=True)
    parser.add_argument("--classes", default="")
    parser.add_argument("--speaker", required=True)
    parser.add_argument("--language", default="English")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--request-timeout", type=float, default=180.0)
    parser.add_argument("--schedule", choices=("stratified", "ordered"), default="stratified")
    parser.add_argument("--schedule-seed", type=int, default=42)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--audio-dir", default="")
    parser.add_argument("--probe-every-min", type=int, default=5)
    args = parser.parse_args()

    wanted = {item.strip() for item in args.classes.split(",") if item.strip()} or None
    rows = load_texts(args.bank, wanted)
    if not rows:
        sys.exit("empty text bank")
    if args.probe_every_min < 1:
        sys.exit("--probe-every-min must be positive")
    if args.audio_dir:
        os.makedirs(args.audio_dir, exist_ok=True)

    pick = make_picker(rows, args.worker, args.schedule_seed, args.schedule)
    probe_cls, probe_text = rows[(args.worker + args.schedule_seed) % len(rows)]
    ticked = set()
    index = 0
    with open(args.csv, "w", newline="", buffering=1, encoding="utf-8") as handle:
        output = csv.writer(handle)
        output.writerow(CSV_COLUMNS)
        while time.time() < args.deadline:
            elapsed = time.time() - args.t0
            minute = int(elapsed // 60)
            probe = bool(args.audio_dir and minute % args.probe_every_min == 0
                         and minute not in ticked)
            if probe:
                ticked.add(minute)
                cls, text = probe_cls, probe_text
                seed = 900000 + args.worker
                output_path = os.path.join(
                    args.audio_dir, f"min{minute:03d}_w{args.worker}.pcm"
                )
            else:
                cls, text = pick(index)
                seed = args.schedule_seed + 1000 + args.worker * 100000 + index
                output_path = ""

            result, error = one(
                args.port, text, args.speaker, args.language, seed,
                args.temperature, output_path, args.request_timeout,
            )
            end = time.time() - args.t0
            tail = (int(probe), cls, len(text), seed, args.schedule)
            if error:
                blanks = [""] * (len(CSV_COLUMNS) - 3 - len(tail) - 1)
                output.writerow((f"{end:.3f}", args.worker, index, *blanks, *tail, error))
            else:
                output.writerow((
                    f"{end:.3f}", args.worker, index,
                    f"{result['ttfb_ms']:.1f}", f"{result['ttfa_ms']:.1f}", f"{result['total_ms']:.1f}",
                    result["bytes"], result["first_chunk_bytes"],
                    f"{result['audio_s']:.3f}", f"{result['stream_rtf']:.4f}",
                    f"{result['underrun_s']:.4f}", f"{result['stall_max_s']:.4f}",
                    f"{result['prebuffer_s']:.4f}", f"{result['gap_ratio_max']:.4f}",
                    result["chunks"],
                    f"{result['safe_play_start_ms']:.1f}", f"{result['header_to_audio_ms']:.1f}",
                    f"{result['max_gap_s']:.4f}", result["coalesced_reads"],
                    *(f"{result[f'stall_ms_at_{b}']:.1f}" if i == 0 else result[f"stalls_at_{b}"]
                      for b in playback_sim.BUFFERS_MS for i in (0, 1)),
                    *tail, "",
                ))
            index += 1


if __name__ == "__main__":
    main()
