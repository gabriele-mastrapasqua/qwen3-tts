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
    handle = open(out_path, "wb") if out_path else None
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            while True:
                chunk = response.read1(1 << 16)
                if not chunk:
                    break
                if first_at is None:
                    first_at = time.time() - started
                    first = len(chunk)
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
    remaining_s = (received - first) / 2.0 / 24000.0
    stream_rtf = ((total - first_at) / remaining_s
                  if first_at is not None and remaining_s > 0 else float("nan"))
    return {
        "ttfa_ms": (first_at or 0.0) * 1000.0,
        "total_ms": total * 1000.0,
        "bytes": received,
        "first_chunk_bytes": first,
        "audio_s": audio_s,
        "stream_rtf": stream_rtf,
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
        output.writerow((
            "t_end_s", "worker", "i", "ttfa_ms", "total_ms", "bytes",
            "first_chunk_bytes", "audio_s", "stream_rtf", "is_probe", "class",
            "text_chars", "seed", "schedule", "error",
        ))
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
            if error:
                output.writerow((
                    f"{end:.3f}", args.worker, index, "", "", "", "", "", "",
                    int(probe), cls, len(text), seed, args.schedule, error,
                ))
            else:
                output.writerow((
                    f"{end:.3f}", args.worker, index,
                    f"{result['ttfa_ms']:.1f}", f"{result['total_ms']:.1f}",
                    result["bytes"], result["first_chunk_bytes"],
                    f"{result['audio_s']:.3f}", f"{result['stream_rtf']:.4f}",
                    int(probe), cls, len(text), seed, args.schedule, "",
                ))
            index += 1


if __name__ == "__main__":
    main()
