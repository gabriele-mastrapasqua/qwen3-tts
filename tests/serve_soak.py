#!/usr/bin/env python3
"""Run a profile-aware closed-loop streaming soak for an open model."""
import argparse
import csv
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import signal
import struct
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROFILE_TOOL = os.path.join(ROOT, "tools", "perf_profile.py")
CLIENT = os.path.join(ROOT, "tests", "soak_client.py")
ANALYZER = os.path.join(ROOT, "tests", "soak_drift.py")
OPEN_MODELS = {
    "qwen3-tts-0.6b",
    "qwen3-tts-0.6b-base",
    "qwen3-tts-1.7b",
    "qwen3-tts-1.7b-base",
}


def parse_env(text):
    values = {}
    if not text:
        return values
    for item in text.split(","):
        if not item:
            continue
        if "=" not in item:
            raise SystemExit(f"invalid environment item {item!r}; use comma-separated KEY=VALUE")
        key, value = item.split("=", 1)
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            raise SystemExit(f"invalid environment key {key!r}")
        if any(char.isspace() for char in value):
            raise SystemExit(f"environment value for {key} contains whitespace")
        values[key] = value
    return values


def profile_command(args):
    if args.profile and args.no_profile:
        raise SystemExit("--profile and --no-profile are mutually exclusive")
    explicit = parse_env(args.server_env)
    if args.no_profile:
        if args.profile:
            raise SystemExit("--profile and --no-profile are mutually exclusive")
        argv = [args.bin, "-d", args.model]
        if args.precision == "int8":
            argv.append("--int8")
        elif args.precision == "int4":
            argv.append("--int4")
        argv += [
            "--serve", str(args.port),
            "--batch-size", str(args.batch_size),
            "--prefork", str(args.prefork),
            "--prefork-threads", str(args.prefork_threads),
        ]
        return argv, explicit, []
    if not args.profile:
        raise SystemExit("pass --profile NAME or --no-profile REASON")

    env_result = subprocess.run(
        [sys.executable, PROFILE_TOOL, "server-env", args.profile],
        cwd=ROOT, capture_output=True, text=True,
    )
    if env_result.returncode:
        raise SystemExit(env_result.stderr or env_result.stdout)
    environment = parse_env(env_result.stdout.strip())
    environment.update(explicit)

    command_result = subprocess.run(
        [sys.executable, PROFILE_TOOL, "command", args.profile,
         "--model", args.model, "--port", str(args.port)],
        cwd=ROOT, capture_output=True, text=True,
    )
    if command_result.returncode:
        raise SystemExit(command_result.stderr or command_result.stdout)
    tokens = shlex.split(command_result.stdout.strip())
    while tokens and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=.*", tokens[0]):
        tokens.pop(0)
    if not tokens:
        raise SystemExit("profile resolved to an empty server command")
    tokens[0] = args.bin

    forbidden_result = subprocess.run(
        [sys.executable, PROFILE_TOOL, "forbidden-env", args.profile],
        cwd=ROOT, capture_output=True, text=True,
    )
    if forbidden_result.returncode:
        raise SystemExit(forbidden_result.stderr or forbidden_result.stdout)
    forbidden = [line.strip() for line in forbidden_result.stdout.splitlines() if line.strip()]
    present = [key for key in forbidden if key in os.environ or key in explicit]
    if present:
        raise SystemExit(
            "refusing to run: profile forbids environment variables already present: "
            + ", ".join(present)
        )
    return tokens, environment, forbidden


def open_model_name(model):
    name = os.path.basename(os.path.normpath(model))
    if name not in OPEN_MODELS:
        allowed = ", ".join(sorted(OPEN_MODELS))
        raise SystemExit(f"model {name!r} is not an allowed open model ({allowed})")
    return name


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def identity(binary, source_commit=""):
    source_override = source_commit.strip() or os.environ.get("QWEN_SOURCE_COMMIT", "").strip()
    try:
        source = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True,
            stderr=subprocess.DEVNULL,
        ).strip())
    except (OSError, subprocess.CalledProcessError):
        source, dirty = "UNKNOWN", None
    if source_override:
        source = source_override
    return {
        "source_commit": source,
        "dirty": dirty,
        "binary_sha256": sha256(binary) if os.path.isfile(binary) else "UNKNOWN",
    }


def health(port):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/health", timeout=2) as response:
            raw = response.read()
        try:
            payload = json.loads(raw.decode())
            if isinstance(payload, dict):
                payload["_ok"] = payload.get("ok") is True or payload.get("status") == "ok"
                return payload
        except (UnicodeDecodeError, json.JSONDecodeError):
            pass
        return {"_ok": b'"ok"' in raw}
    except Exception:
        return {}


def live_probe(args):
    body = json.dumps({
        "text": "Warm up the streaming endpoint.",
        "speaker": args.speaker,
        "language": args.language,
        "seed": 7,
        "temperature": args.temperature,
    }).encode()
    request = urllib.request.Request(
        f"http://127.0.0.1:{args.port}/v1/tts/stream",
        data=body, headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=args.request_timeout) as response:
        return bool(response.read(1))


def wait_ready(process, args):
    deadline = time.time() + args.ready_timeout
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"server exited with status {process.returncode}")
        # The health endpoint may remain 503 while the streaming route is
        # already usable. The live request is the authoritative readiness test.
        try:
            if live_probe(args):
                return
        except (OSError, urllib.error.URLError, TimeoutError):
            pass
        time.sleep(1)
    raise RuntimeError(f"server did not become ready within {args.ready_timeout:.0f}s")


def check_profile_flags(args, log_path, expected_env):
    if not args.profile:
        return
    deadline = time.time() + 30
    while time.time() < deadline:
        seen = {}
        try:
            with open(log_path, encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    if line.startswith("[FLAGS]"):
                        seen = {
                            key: value for key, value in (
                                token.split("=", 1)
                                for token in line.split()[2:] if "=" in token
                            )
                        }
        except OSError:
            seen = {}
        if seen:
            wanted = {key: value for key, value in expected_env.items()
                      if key.startswith("QWEN_")}
            missing = {key: value for key, value in wanted.items()
                       if seen.get(key) != value}
            if missing:
                raise RuntimeError(
                    "profile flags were not confirmed by the server: "
                    + ", ".join(f"{key}={value}" for key, value in sorted(missing.items()))
                )
            return
        time.sleep(1)
    raise RuntimeError("server did not print a [FLAGS] line for profile verification")


def proc_tree(root_pid):
    if not root_pid or os.path.isdir("/proc"):
        parent = {}
        try:
            for name in os.listdir("/proc"):
                if not name.isdigit():
                    continue
                try:
                    line = Path("/proc", name, "stat").read_text()
                    tail = line.rsplit(")", 1)[1].split()
                    parent[int(name)] = int(tail[1])
                except (OSError, ValueError, IndexError):
                    continue
        except OSError:
            parent = {}
        selected = {root_pid}
        changed = True
        while changed:
            changed = False
            for pid, ppid in parent.items():
                if ppid in selected and pid not in selected:
                    selected.add(pid)
                    changed = True
        return sorted(selected)

    try:
        output = subprocess.check_output(
            ["ps", "-axo", "pid=,ppid=,rss=,nlwp=,%cpu="],
            text=True, stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return [root_pid]
    parent, stats = {}, {}
    for line in output.splitlines():
        fields = line.split()
        if len(fields) < 5:
            continue
        try:
            pid, ppid = int(fields[0]), int(fields[1])
            stats[pid] = (float(fields[2]), float(fields[3]), float(fields[4]))
            parent[pid] = ppid
        except ValueError:
            continue
    selected = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid, ppid in parent.items():
            if ppid in selected and pid not in selected:
                selected.add(pid)
                changed = True
    return sorted(selected)


def status_value(pid, key):
    try:
        for line in Path("/proc", str(pid), "status").read_text().splitlines():
            if line.startswith(key + ":"):
                return int(line.split()[1])
    except (OSError, ValueError, IndexError):
        pass
    return 0


def linux_sample(pids):
    result = {key: 0 for key in (
        "rss_kb", "pss_kb", "anon_kb", "swap_kb", "threads", "fds", "cpu_ticks",
    )}
    for pid in pids:
        result["rss_kb"] += status_value(pid, "VmRSS")
        result["anon_kb"] += status_value(pid, "RssAnon")
        result["swap_kb"] += status_value(pid, "VmSwap")
        result["threads"] += status_value(pid, "Threads")
        try:
            result["fds"] += len(os.listdir(Path("/proc", str(pid), "fd")))
        except OSError:
            pass
        try:
            pss = 0
            for line in Path("/proc", str(pid), "smaps_rollup").read_text().splitlines():
                if line.startswith("Pss:"):
                    pss += int(line.split()[1])
            result["pss_kb"] += pss
        except (OSError, ValueError, IndexError):
            pass
        try:
            fields = Path("/proc", str(pid), "stat").read_text().rsplit(")", 1)[1].split()
            result["cpu_ticks"] += int(fields[11]) + int(fields[12])
        except (OSError, ValueError, IndexError):
            pass
    return result


def sample_process_tree(root_pid):
    pids = proc_tree(root_pid)
    if os.path.isdir("/proc"):
        result = linux_sample(pids)
    else:
        result = {key: 0 for key in (
            "rss_kb", "pss_kb", "anon_kb", "swap_kb", "threads", "fds", "cpu_ticks",
        )}
        try:
            output = subprocess.check_output(
                ["ps", "-axo", "pid=,ppid=,rss=,nlwp=,%cpu="],
                text=True, stderr=subprocess.DEVNULL,
            )
            wanted = set(pids)
            for line in output.splitlines():
                fields = line.split()
                if len(fields) < 5 or int(fields[0]) not in wanted:
                    continue
                result["rss_kb"] += int(float(fields[2]))
                result["threads"] += int(float(fields[3]))
        except (OSError, subprocess.CalledProcessError, ValueError, IndexError):
            pass
        result["pss_kb"] = result["rss_kb"]
    result["pids"] = len(pids)
    return result


def process_group_members(group_id):
    members = set()
    if os.path.isdir("/proc"):
        try:
            names = os.listdir("/proc")
        except OSError:
            names = []
        for name in names:
            if not name.isdigit():
                continue
            try:
                fields = Path("/proc", name, "stat").read_text().rsplit(")", 1)[1].split()
                if int(fields[2]) == group_id:
                    members.add(int(name))
            except (OSError, ValueError, IndexError):
                continue
        return members
    try:
        output = subprocess.check_output(
            ["ps", "-axo", "pid=,pgid="], text=True, stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return members
    for line in output.splitlines():
        fields = line.split()
        try:
            if len(fields) >= 2 and int(fields[1]) == group_id:
                members.add(int(fields[0]))
        except ValueError:
            continue
    return members


def resource_loop(stop, root_pid, port, path, interval):
    fields = (
        "elapsed_s", "pids", "rss_kb", "pss_kb", "anon_kb", "swap_kb", "threads",
        "fds", "cpu_ticks", "queue_running", "queue_waiting", "queue_rejected",
        "queue_timeout", "request_timeout",
    )
    started = time.time()
    with open(path, "w", newline="", encoding="utf-8") as handle:
        output = csv.DictWriter(handle, fieldnames=fields)
        output.writeheader()
        while not stop.is_set():
            sample = sample_process_tree(root_pid)
            state = health(port)
            row = {key: sample.get(key, 0) for key in fields}
            row["elapsed_s"] = round(time.time() - started, 3)
            row["queue_running"] = state.get("num_requests_running", 0)
            row["queue_waiting"] = state.get("num_requests_waiting", 0)
            row["queue_rejected"] = state.get("rejected_queue_full", 0)
            row["queue_timeout"] = state.get("rejected_queue_timeout", 0)
            row["request_timeout"] = state.get("timed_out", 0)
            output.writerow(row)
            handle.flush()
            stop.wait(interval)


def terminate(process):
    if process is None:
        return
    running = process.poll() is None
    # Capture the dedicated group's members only while its leader is alive. If
    # the leader already exited, do not infer ownership from a reused group id.
    known = process_group_members(process.pid) if running else set()
    try:
        if running and process.pid in known:
            os.killpg(process.pid, signal.SIGTERM)
        elif running:
            process.terminate()
    except (OSError, ProcessLookupError):
        if running:
            try:
                process.terminate()
            except OSError:
                pass
    if running:
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            pass

    # The known set is from before SIGTERM, so a reused process group cannot
    # expand the cleanup scope. Recheck the group before touching a survivor.
    survivors = process_group_members(process.pid) & known
    for pid in survivors:
        if pid == os.getpid():
            continue
        try:
            os.kill(pid, signal.SIGTERM)
        except (OSError, ProcessLookupError):
            pass
    if survivors:
        time.sleep(0.2)
    survivors = process_group_members(process.pid) & known
    for pid in survivors:
        if pid == os.getpid():
            continue
        try:
            os.kill(pid, signal.SIGKILL)
        except (OSError, ProcessLookupError):
            pass
    if process.poll() is None:
        process.wait()


def to_wav(pcm_path):
    pcm = Path(pcm_path).read_bytes()
    wav_path = str(Path(pcm_path).with_suffix(".wav"))
    header = (
        b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVEfmt " +
        struct.pack("<IHHIIHH", 16, 1, 1, 24000, 48000, 2, 16) +
        b"data" + struct.pack("<I", len(pcm))
    )
    Path(wav_path).write_bytes(header + pcm)
    Path(pcm_path).unlink()


def merge_requests(out):
    paths = sorted(Path(out).glob("requests-w*.csv"))
    rows, fields = [], None
    for path in paths:
        with path.open(newline="", encoding="utf-8", errors="replace") as handle:
            reader = csv.DictReader(handle)
            if fields is None:
                fields = reader.fieldnames
            rows.extend(reader)
    if not fields:
        fields = [
            "t_end_s", "worker", "i", "ttfa_ms", "total_ms", "bytes",
            "first_chunk_bytes", "audio_s", "stream_rtf", "is_probe", "class",
            "text_chars", "seed", "schedule", "error",
        ]
    rows.sort(key=lambda row: float(row.get("t_end_s") or "inf"))
    with open(os.path.join(out, "requests.csv"), "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def safe_command(argv, model, binary):
    model_name = open_model_name(model)
    binary_name = os.path.basename(binary)
    return [model_name if item == model else binary_name if item == binary else item
            for item in argv]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="qwen3-tts-0.6b")
    parser.add_argument("--bin", default=os.path.join(ROOT, "qwen_tts"))
    parser.add_argument("--profile", default="")
    parser.add_argument("--no-profile", default="")
    parser.add_argument("--server-env", default="", metavar="KEY=VALUE,...")
    parser.add_argument("--port", type=int, default=9700)
    parser.add_argument("--bank", default=os.path.join(ROOT, "tests", "load_texts_en.txt"))
    parser.add_argument("--classes", default="")
    parser.add_argument("--speaker", default="ryan")
    parser.add_argument("--language", default="English")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--precision", choices=("default", "int8", "int4"), default="int8")
    parser.add_argument("--prefork", type=int, default=1)
    parser.add_argument("--prefork-threads", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--minutes", type=float, default=10.0)
    parser.add_argument("--request-timeout", type=float, default=180.0)
    parser.add_argument("--ready-timeout", type=float, default=600.0)
    parser.add_argument("--probe-every-min", type=int, default=5)
    parser.add_argument("--resource-every-s", type=float, default=15.0)
    parser.add_argument("--schedule", choices=("stratified", "ordered"), default="stratified")
    parser.add_argument("--schedule-seed", type=int, default=42)
    parser.add_argument("--warmup-s", type=float, default=60.0)
    parser.add_argument("--window-s", type=float, default=60.0)
    parser.add_argument("--min-per-window", type=int, default=5)
    parser.add_argument("--min-per-class", type=int, default=3)
    parser.add_argument("--min-per-class-p95", type=int, default=20)
    parser.add_argument("--min-windows", type=int, default=3)
    parser.add_argument("--max-mix-distance", type=float, default=0.20)
    parser.add_argument("--max-ttfa-drift", type=float, default=30.0)
    parser.add_argument("--max-stream-drift", type=float, default=20.0)
    parser.add_argument(
        "--source-commit",
        default=os.environ.get("QWEN_SOURCE_COMMIT", ""),
        help="source revision to record when the run is built outside a Git checkout",
    )
    parser.add_argument("--out", default="/tmp/qwen_tts_soak")
    parser.add_argument("--strict-kpi", action="store_true")
    args = parser.parse_args()

    if args.concurrency < 1 or args.minutes <= 0:
        parser.error("--concurrency must be positive and --minutes must be greater than zero")
    if args.resource_every_s <= 0:
        parser.error("--resource-every-s must be positive")
    if args.warmup_s < 0 or args.window_s <= 0:
        parser.error("--warmup-s must be non-negative and --window-s must be positive")
    if (args.min_per_window < 1 or args.min_per_class < 1 or
            args.min_per_class_p95 < 1 or args.min_windows < 1):
        parser.error("minimum sample and window counts must be positive")
    if args.max_mix_distance < 0 or args.max_ttfa_drift < 0 or args.max_stream_drift < 0:
        parser.error("KPI thresholds must be non-negative")
    if any(char.isspace() for char in args.source_commit):
        parser.error("--source-commit must not contain whitespace")
    if not os.path.isfile(args.bin) or not os.access(args.bin, os.X_OK):
        parser.error(f"executable not found: {args.bin}")
    if not os.path.isdir(args.model):
        parser.error(f"model directory not found: {args.model}")
    open_model_name(args.model)
    if not os.path.isfile(args.bank):
        parser.error(f"text bank not found: {args.bank}")

    out = args.out
    if os.path.isdir(out) and os.listdir(out):
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out = f"{out}-{stamp}"
        print(f"output exists; writing a recoverable new run to {out}")
    os.makedirs(os.path.join(out, "audio"), exist_ok=True)

    argv, server_env, forbidden = profile_command(args)
    environment = os.environ.copy()
    environment.update(server_env)
    command_for_manifest = safe_command(argv, args.model, args.bin)
    run_identity = identity(args.bin, args.source_commit)
    manifest = {
        "schema_version": 1,
        "benchmark_family": "closed_loop_soak",
        "model": open_model_name(args.model),
        "speaker": args.speaker,
        "language": args.language,
        "temperature": args.temperature,
        "text_bank": os.path.basename(args.bank),
        "classes": args.classes,
        "schedule": args.schedule,
        "schedule_seed": args.schedule_seed,
        "concurrency": args.concurrency,
        "duration_s": args.minutes * 60.0,
        "analysis": {
            "warmup_s": args.warmup_s,
            "window_s": args.window_s,
            "min_per_window": args.min_per_window,
            "min_per_class": args.min_per_class,
            "min_per_class_p95": args.min_per_class_p95,
            "min_windows": args.min_windows,
            "max_mix_distance": args.max_mix_distance,
            "max_ttfa_drift_pct": args.max_ttfa_drift,
            "max_stream_drift_pct": args.max_stream_drift,
        },
        "probe_every_min": args.probe_every_min,
        "resource_every_s": args.resource_every_s,
        "profile": args.profile or None,
        "no_profile_reason": args.no_profile or None,
        "server_argv": command_for_manifest,
        "server_env": server_env,
        "forbidden_env": forbidden,
        "identity": run_identity,
    }
    with open(os.path.join(out, "manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    with open(os.path.join(out, "server_command.txt"), "w", encoding="utf-8") as handle:
        prefix = " ".join(f"{key}={value}" for key, value in sorted(server_env.items()))
        handle.write((prefix + " " if prefix else "") + shlex.join(command_for_manifest) + "\n")

    server_log = os.path.join(out, "server.log")
    server_handle = open(server_log, "wb")
    server = clients = None
    client_processes = []
    resource_stop = threading.Event()
    resource_thread = None
    result = 1
    try:
        server = subprocess.Popen(
            argv, cwd=ROOT, env=environment, stdout=server_handle,
            stderr=subprocess.STDOUT, start_new_session=True,
        )
        wait_ready(server, args)
        check_profile_flags(args, server_log, server_env)
        if args.profile:
            print(f"profile verified: {args.profile}")
        print(f"server ready: model={open_model_name(args.model)} port={args.port}")

        resource_thread = threading.Thread(
            target=resource_loop,
            args=(resource_stop, server.pid, args.port,
                  os.path.join(out, "resources.csv"), args.resource_every_s),
            daemon=True,
        )
        resource_thread.start()
        started = time.time()
        deadline = started + args.minutes * 60.0
        for worker in range(args.concurrency):
            worker_csv = os.path.join(out, f"requests-w{worker}.csv")
            worker_log = open(os.path.join(out, f"worker-{worker}.log"), "wb")
            command = [
                sys.executable, CLIENT,
                "--port", str(args.port), "--worker", str(worker),
                "--t0", str(started), "--deadline", str(deadline),
                "--bank", args.bank, "--speaker", args.speaker,
                "--language", args.language, "--temperature", str(args.temperature),
                "--request-timeout", str(args.request_timeout),
                "--schedule", args.schedule, "--schedule-seed", str(args.schedule_seed),
                "--csv", worker_csv, "--audio-dir", os.path.join(out, "audio"),
                "--probe-every-min", str(args.probe_every_min),
            ]
            if args.classes:
                command += ["--classes", args.classes]
            process = subprocess.Popen(
                command, cwd=ROOT, stdout=worker_log, stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            process._soak_log = worker_log
            client_processes.append(process)
        print(f"soak running: {args.minutes:g} min, closed-loop concurrency={args.concurrency}")

        wait_timeout = args.minutes * 60.0 + args.request_timeout + 30.0
        wait_started = time.time()
        for process in client_processes:
            remaining = max(1.0, wait_timeout - (time.time() - wait_started))
            try:
                status = process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                raise RuntimeError("a soak worker exceeded the run deadline")
            if status != 0:
                raise RuntimeError(f"soak worker {client_processes.index(process)} exited with status {status}")
    except (OSError, RuntimeError, urllib.error.URLError) as error:
        print(f"SOAK RUN FAILED: {error}", file=sys.stderr)
    finally:
        for process in client_processes:
            terminate(process)
            log_handle = getattr(process, "_soak_log", None)
            if log_handle:
                log_handle.close()
        resource_stop.set()
        if resource_thread:
            resource_thread.join(timeout=5)
        terminate(server)
        server_handle.close()
        merge_requests(out)
        for pcm in Path(out, "audio").glob("*.pcm"):
            try:
                to_wav(pcm)
            except OSError as error:
                print(f"warning: could not convert {pcm}: {error}", file=sys.stderr)
        analysis_command = [
            sys.executable, ANALYZER, out,
            "--warmup-s", str(args.warmup_s),
            "--window-s", str(args.window_s),
            "--min-per-window", str(args.min_per_window),
            "--min-per-class", str(args.min_per_class),
            "--min-per-class-p95", str(args.min_per_class_p95),
            "--min-windows", str(args.min_windows),
            "--max-mix-distance", str(args.max_mix_distance),
            "--max-ttfa-drift", str(args.max_ttfa_drift),
            "--max-stream-drift", str(args.max_stream_drift),
        ]
        if args.strict_kpi:
            analysis_command.append("--strict-kpi")
        analysis = subprocess.run(
            analysis_command,
            cwd=ROOT,
        )
        result = analysis.returncode
    return result


if __name__ == "__main__":
    sys.exit(main())
