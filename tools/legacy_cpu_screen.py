#!/usr/bin/env python3
"""Fast, model-free legacy CPU candidate screen.

This is deliberately a screen, not a qualification suite. It records the host and
resolved dispatch first, then runs the existing bandwidth/roof/matmat tools with the
experimental AVX2 candidates disabled and enabled in fresh processes. A requested
candidate is only a policy override; compiled/runtime capability checks remain in the
engine and the dispatch/census output is the authority on what actually ran.

Example (physical cores, one 8-thread worker):
  make legacy-cpu-screen LEGACY_SCREEN_MODE=physical \
      LEGACY_SCREEN_THREADS=8 LEGACY_SCREEN_CPUS=0-7

For SMT, run the same command with a separate output directory and MODE=smt. The
script never silently combines the two modes in one manifest.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def command_prefix(cpus: str | None) -> list[str]:
    if not cpus:
        return []
    taskset = shutil.which("taskset")
    if not taskset:
        raise RuntimeError("--cpus requires taskset on this host")
    return [taskset, "-c", cpus]


def run_one(name: str, argv: list[str], out_dir: Path, env_overrides: dict[str, str],
            timeout: int, cpus: str | None, records: list[dict]) -> None:
    env = dict(os.environ)
    env.update(env_overrides)
    cmd = command_prefix(cpus) + argv
    started = time.time()
    try:
        proc = subprocess.run(cmd, cwd=ROOT, env=env, text=True,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              timeout=timeout)
        rc = proc.returncode
        output = proc.stdout
    except subprocess.TimeoutExpired as exc:
        rc = 124
        output = (exc.stdout or "") + f"\nTIMEOUT after {timeout}s\n"
    except OSError as exc:
        rc = 127
        output = f"EXEC ERROR: {exc}\n"
    path = out_dir / f"{name}.log"
    path.write_text(output)
    records.append({
        "name": name,
        "argv": cmd,
        "env_overrides": env_overrides,
        "returncode": rc,
        "seconds": round(time.time() - started, 3),
        "log": str(path),
    })
    status = "PASS" if rc == 0 else f"FAIL(rc={rc})"
    print(f"  {name:28s} {status:12s} {time.time() - started:7.1f}s")


def host_command(name: str, argv: list[str], out_dir: Path,
                 records: list[dict], timeout: int = 30) -> None:
    if not shutil.which(argv[0]):
        path = out_dir / f"{name}.log"
        path.write_text(f"NOT AVAILABLE: {argv[0]}\n")
        records.append({"name": name, "argv": argv, "returncode": 0,
                        "seconds": 0.0, "log": str(path), "optional": True,
                        "note": f"{argv[0]} not installed"})
        print(f"  {name:28s} {'SKIP':12s} {'optional tool unavailable':>7s}")
        return
    run_one(name, argv, out_dir, {}, timeout, None, records)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bin", required=True, help="built qwen_tts binary")
    p.add_argument("--membw", required=True, help="existing membw helper")
    p.add_argument("--roof", required=True, help="existing roof_matvec_int8 helper")
    p.add_argument("--out", required=True, help="directory for one manifest and raw logs")
    p.add_argument("--mode", choices=("physical", "smt", "unspecified"),
                    default="unspecified")
    p.add_argument("--threads", type=int, default=1,
                    help="threads for the candidate screen; record separately per mode")
    p.add_argument("--cpus", default=None,
                    help="optional taskset mask applied to every benchmark process")
    p.add_argument("--reps", type=int, default=2)
    p.add_argument("--roof-layers", default="2,28",
                    help="comma-separated CP-sized and Talker-sized roof layer counts")
    p.add_argument("--timeout", type=int, default=240)
    return p.parse_args()


def main() -> int:
    a = parse_args()
    if a.threads < 1:
        raise SystemExit("--threads must be positive")
    out = Path(a.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    binary = str(Path(a.bin).resolve())
    membw = str(Path(a.membw).resolve())
    roof = str(Path(a.roof).resolve())
    common = {"QWEN_AVX2_INT8_GEMV": "0", "QWEN_AVX2_Q4_GEMV": "0"}

    print(f"legacy CPU screen: {out}")
    print(f"  mode={a.mode} threads={a.threads} cpus={a.cpus or 'process default'}")

    # Host/source identity is kept as raw files so no parser can discard useful flags.
    host_command("uname", ["uname", "-a"], out, records)
    host_command("lscpu", ["lscpu"], out, records)
    host_command("lscpu_extended", ["lscpu", "-e=CPU,NODE,SOCKET,CORE,CACHE"], out, records)
    if shutil.which("numactl"):
        host_command("numactl", ["numactl", "-H"], out, records)
    if shutil.which("sysctl"):
        host_command("sysctl_cpu", ["sysctl", "-a"], out, records, timeout=30)
    host_command("git_head", ["git", "rev-parse", "HEAD"], out, records)
    host_command("git_status", ["git", "status", "--short"], out, records)
    host_command("compiler_version", [os.environ.get("CC", "cc"), "--version"], out, records)

    print("\nengine identity and baseline truth")
    run_one("caps", [binary, "--caps"], out, {}, a.timeout, None, records)
    run_one("dispatch_baseline", [binary, "--dispatch-map"], out, common,
             a.timeout, a.cpus, records)
    run_one("selftest_baseline", [binary, "--self-test"], out, common,
             a.timeout, a.cpus, records)

    candidate_envs = {
        "int8_candidate": {**common, "QWEN_AVX2_INT8_GEMV": "1"},
        "q4_candidate": {**common, "QWEN_AVX2_Q4_GEMV": "1"},
    }
    for label, env in candidate_envs.items():
        run_one(f"dispatch_{label}", [binary, "--dispatch-map"], out, env,
                a.timeout, a.cpus, records)
        run_one(f"selftest_{label}", [binary, "--self-test"], out, env,
                a.timeout, a.cpus, records)

    # Existing memory roof. The thread list includes the requested worker width and never
    # labels a run as physical/SMT unless the caller supplied that mode explicitly.
    thread_values = [1, 2, 4, a.threads]
    thread_list = ",".join(str(x) for i, x in enumerate(thread_values)
                            if x > 0 and x not in thread_values[:i])
    run_one("membw", [membw, "--threads", thread_list, "--reps", str(a.reps), "--json"],
            out, {}, a.timeout, a.cpus, records)

    print("\nINT8 B1 roof: current FMA versus experimental integer candidate")
    for layers in [int(x) for x in a.roof_layers.split(",") if x.strip()]:
        suffix = f"layers{layers}"
        for label, env in [("fma", common), ("avx2_integer", candidate_envs["int8_candidate"])]:
            run_one(f"roof_int8_{label}_{suffix}",
                    [roof, "--threads", str(a.threads), "--layers", str(layers),
                     "--reps", str(a.reps)], out, env, a.timeout, a.cpus, records)

    # The existing matmat bench contains the complete-call B=1 sequence and B=2/4/8
    # comparisons for INT8/Q4. Shape census is enabled so the report records the leaf.
    print("\nmatmat B1/B2/B4/B8 candidate screen")
    bench_base = [binary, "--matmat-bench", "-j", str(a.threads)]
    for label, env in [
        ("reference_fma", common),
        ("int8_integer", candidate_envs["int8_candidate"]),
        ("q4_integer", candidate_envs["q4_candidate"]),
    ]:
        screen_env = {**env, "QWEN_SHAPE_CENSUS": "1"}
        run_one(f"matmat_{label}", bench_base, out, screen_env,
                a.timeout, a.cpus, records)

    manifest = {
        "schema": "legacy-cpu-screen/v1",
        "created_unix": time.time(),
        "root": str(ROOT),
        "binary": binary,
        "mode": a.mode,
        "threads": a.threads,
        "cpus": a.cpus,
        "reps": a.reps,
        "roof_layers": [int(x) for x in a.roof_layers.split(",") if x.strip()],
        "notes": [
            "This is a diagnostic screen, not performance qualification.",
            "Candidate flags only request opt-in paths; runtime capability checks remain authoritative.",
            "Kernel B is not server concurrency C.",
            "Run physical and SMT modes as separate manifests.",
        ],
        "commands": records,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    failures = [r for r in records if r["returncode"] != 0]
    print("\nmanifest:", out / "manifest.json")
    print("logs:", out)
    print(f"result: {'PASS' if not failures else 'INCOMPLETE'} "
          f"({len(records) - len(failures)}/{len(records)} commands rc=0)")
    return 0 if not failures else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"legacy_cpu_screen: {exc}", file=sys.stderr)
        raise SystemExit(2)
