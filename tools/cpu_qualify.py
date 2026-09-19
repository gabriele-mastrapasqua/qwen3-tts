#!/usr/bin/env python3
"""One-command native CPU qualification for the old-CPU/v2 work.

This is an orchestration harness, not a synthetic ISA emulator.  It records the host and
resolved engine policy first, runs the compile/parity gates that are meaningful on the
current machine, then runs model-free A/B kernels.  With ``--model`` it also runs the
prepared Talker/CP batch oracle at the requested B values; ``--serve`` adds a short C1/C2/
C4/C8 continuous-server wave and requires the v2 census to be emitted.

Every result is labelled in the manifest and summary as one of:

``COMPILE VERIFIED``      translation units and guards compiled
``PARITY VERIFIED``       the available reference/correctness test passed
``NATIVE RUNTIME REQUIRED``  the host cannot execute that ISA branch here
``PERFORMANCE REQUIRED``  a measurement still has to be made on a real target

Examples::

    python3 tools/cpu_qualify.py --binary ./qwen_tts
    python3 tools/cpu_qualify.py --binary ./qwen_tts --model /models/qwen --serve

The output directory is intentionally outside tracked source (``profiles/`` is ignored).
"""

from __future__ import annotations

import argparse
import ctypes
import datetime as _dt
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def host_cmd(argv: list[str], timeout: int = 30) -> str:
    if not argv or not shutil.which(argv[0]):
        return ""
    try:
        p = subprocess.run(argv, cwd=ROOT, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, text=True, errors="replace",
                           timeout=timeout)
        return p.stdout
    except (OSError, subprocess.TimeoutExpired):
        return ""


def run_cmd(records: list[dict[str, Any]], name: str, argv: list[str], out: Path,
            env_overrides: dict[str, str] | None = None, timeout: int = 900,
            required: bool = True, label: str = "") -> dict[str, Any]:
    """Run one prepared command and persist its complete output."""
    env = os.environ.copy()
    overrides = dict(env_overrides or {})
    env.update(overrides)
    path = out / f"{safe_name(name)}.log"
    start = time.time()
    rc = 127
    text = ""
    try:
        p = subprocess.run(argv, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, text=True, errors="replace",
                           timeout=timeout)
        rc, text = p.returncode, p.stdout
    except subprocess.TimeoutExpired as exc:
        rc = 124
        value = exc.stdout or ""
        text = value.decode("utf-8", "replace") if isinstance(value, bytes) else value
        text += f"\nTIMEOUT after {timeout}s\n"
    except OSError as exc:
        text = f"EXEC ERROR: {exc}\n"
    path.write_text(text)
    rec = {
        "name": name,
        "argv": argv,
        "env_overrides": overrides,
        "returncode": rc,
        "seconds": round(time.time() - start, 3),
        "log": str(path),
        "required": required,
        "status": "PASS" if rc == 0 else "FAIL",
    }
    if label:
        rec["evidence"] = label
    records.append(rec)
    print(f"  {name:36s} {'PASS' if rc == 0 else f'FAIL(rc={rc})':12s}"
          f" {rec['seconds']:7.1f}s")
    return rec


def parse_args() -> argparse.Namespace:
    cpu_default = min(8, os.cpu_count() or 1)
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--binary", "--bin", default="./qwen_tts")
    p.add_argument("--out", default="", help="artifact directory (default: profiles/cpu-qualify/<timestamp>)")
    p.add_argument("--model", default="", help="model directory for batch/server qualification")
    p.add_argument("--serve", action="store_true", help="run the v2 C1/C2/C4/C8 wave (requires --model)")
    p.add_argument("--threads", type=int, default=cpu_default)
    p.add_argument("--batch-values", default="1,2,3,4,8,15,16,17,20,24,31,32")
    p.add_argument("--no-unchunked", action="store_true",
                   help="do not run the B>16 generic control next to the chunked run")
    p.add_argument("--chunk-max", type=int, default=16)
    p.add_argument("--concurrency", default="1,2,4,8")
    p.add_argument("--text-file", default="tests/load_texts_en.txt")
    p.add_argument("--classes", default="short")
    p.add_argument("--precision", default="int8")
    p.add_argument("--server-env", default="", metavar="K=V,K=V")
    p.add_argument("--timeout", type=int, default=900)
    return p.parse_args()


def capture_hardware(out: Path, records: list[dict[str, Any]], timeout: int) -> dict[str, Any]:
    raw = out / "hardware"
    raw.mkdir(exist_ok=True)
    uname = host_cmd(["uname", "-a"])
    (raw / "uname.txt").write_text(uname)
    for name, argv in (("lscpu", ["lscpu"]), ("numactl", ["numactl", "--hardware"]),
                       ("getconf", ["getconf", "-a"]),
                       ("sysctl_cpu", ["sysctl", "-n", "machdep.cpu.brand_string"]),
                       ("sysctl_features", ["sysctl", "-n", "machdep.cpu.features"]),
                       ("sysctl_leaf", ["sysctl", "-n", "machdep.cpu.leaf7_features"])):
        text = host_cmd(argv, timeout=timeout)
        if text:
            (raw / f"{name}.txt").write_text(text)
    if Path("/proc/cpuinfo").is_file():
        (raw / "proc_cpuinfo.txt").write_text(Path("/proc/cpuinfo").read_text(errors="replace"))
    aux: dict[str, Any] = {"AT_HWCAP": None, "AT_HWCAP2": None}
    if sys.platform.startswith("linux"):
        try:
            libc = ctypes.CDLL(None)
            libc.getauxval.argtypes = [ctypes.c_ulong]
            libc.getauxval.restype = ctypes.c_ulong
            aux = {"AT_HWCAP": hex(libc.getauxval(16)), "AT_HWCAP2": hex(libc.getauxval(26))}
        except (AttributeError, OSError):
            aux = {"AT_HWCAP": "unavailable", "AT_HWCAP2": "unavailable"}
    (raw / "auxv.json").write_text(json.dumps(aux, indent=2) + "\n")

    cpuinfo = (raw / "proc_cpuinfo.txt").read_text(errors="replace") if (raw / "proc_cpuinfo.txt").exists() else ""
    flag_match = re.search(r"^(?:flags|Features)\s*:\s*(.*)$", cpuinfo, re.M | re.I)
    flags = sorted(set((flag_match.group(1).lower().split() if flag_match else [])))
    host = {
        "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "uname": uname.strip(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": sys.version,
        "logical_cpus": os.cpu_count(),
        "cpu_flags_from_proc": flags,
        "auxv": aux,
        "raw_dir": str(raw),
    }
    (out / "hardware.json").write_text(json.dumps(host, indent=2) + "\n")

    box = ROOT / "tools" / "box_info.sh"
    if box.exists():
        run_cmd(records, "hardware_box_info", ["bash", str(box), "--out", str(out / "box-info.json")],
                out, timeout=timeout, required=False, label="hardware fingerprint")
    return host


def dispatch_and_engine(binary: Path, out: Path, records: list[dict[str, Any]], timeout: int) -> dict[str, Any]:
    run_cmd(records, "caps", [str(binary), "--caps"], out, timeout=timeout,
            label="resolved engine capabilities")
    dispatch_path = out / "dispatch.json"
    run_cmd(records, "dispatch_map", [str(binary), "--dispatch-map"], out,
            {"QWEN_DISPATCH_JSON": str(dispatch_path)}, timeout=timeout,
            label="resolved dispatch policy")
    dispatch: dict[str, Any] = {}
    try:
        dispatch = json.loads(dispatch_path.read_text())
    except (OSError, json.JSONDecodeError):
        pass
    fp = run_cmd(records, "profile_fingerprint", [sys.executable, "tools/profile_check.py",
                     "--fingerprint", "--bin", str(binary)], out, timeout=timeout,
                 label="source/binary provenance")
    try:
        log = Path(fp["log"]).read_text()
        first = log.find("{")
        if first >= 0:
            (out / "provenance.json").write_text(json.dumps(json.loads(log[first:]), indent=2) + "\n")
    except (OSError, json.JSONDecodeError):
        pass
    return dispatch


def make_checks(binary: Path, out: Path, records: list[dict[str, Any]], timeout: int) -> None:
    # These gates intentionally run in fresh processes.  A mocked capability test is policy
    # evidence; the native self-test is the only machine-code execution evidence.
    run_cmd(records, "test_v2_census", ["make", "test-v2-census"], out,
            timeout=timeout, label="PARITY VERIFIED: census/chunk policy unit")
    run_cmd(records, "flag_registry", [sys.executable, "tools/check_flag_registry.py"], out,
            timeout=timeout, label="COMPILE VERIFIED: declared runtime flags")
    run_cmd(records, "compile_isa_profiles", ["make", "check-isa"], out,
            timeout=timeout, label="COMPILE VERIFIED: available ISA translation units")
    run_cmd(records, "matmat_parity_native", ["make", "check-matmat-parity"], out,
            timeout=timeout, label="PARITY VERIFIED: native matmat reference")
    run_cmd(records, "matmat_parity_x86_control", ["make", "check-matmat-parity-x86"], out,
            timeout=timeout, required=False,
            label="PARITY VERIFIED where Rosetta control is available")
    run_cmd(records, "kai_dotprod_parity", ["make", "test-kai-dotprod"], out,
            timeout=timeout, label="PARITY VERIFIED: dotprod-only vendor pack/kernel ABI where available")
    run_cmd(records, "selftest_native", [str(binary), "--self-test"], out,
            timeout=timeout, label="PARITY VERIFIED: dispatched native self-test")
    run_cmd(records, "selftest_fallback", [str(binary), "--self-test"], out,
            {"QWEN_NO_SDOT": "1", "QWEN_NO_VNNI": "1", "QWEN_NO_AMX": "1"},
            timeout=timeout, label="PARITY VERIFIED: forced portable fallback")


def candidate_envs(dispatch: dict[str, Any]) -> dict[str, dict[str, str]]:
    baseline = {
        "QWEN_AVX2_INT8_GEMV": "0", "QWEN_AVX2_Q4_GEMV": "0",
        "QWEN_AVX512_INT8_GEMV": "0", "QWEN_AVX512_Q4_GEMV": "0",
        "QWEN_INT8_SDOT_MM": "0", "QWEN_Q4_SDOT_MM": "0",
        "QWEN_KAI_DOTPROD_GEMV": "0",
    }
    cls = str(dispatch.get("isa_class", ""))
    if cls.startswith("x86_") or platform.machine().lower() in ("x86_64", "amd64"):
        return {
            "baseline": baseline,
            "avx2_int8_gemv": {**baseline, "QWEN_AVX2_INT8_GEMV": "1"},
            "avx2_q4_gemv": {**baseline, "QWEN_AVX2_Q4_GEMV": "1"},
            "avx512bw_int8_gemv": {**baseline, "QWEN_AVX512_INT8_GEMV": "1"},
            "avx512bw_q4_gemv": {**baseline, "QWEN_AVX512_Q4_GEMV": "1"},
        }
    return {
        "baseline": baseline,
        "sdot_matmat": {**baseline, "QWEN_INT8_SDOT_MM": "1"},
        "q4_sdot_matmat": {**baseline, "QWEN_Q4_SDOT_MM": "1"},
        "kai_dotprod_gemv": {**baseline, "QWEN_KAI_DOTPROD_GEMV": "1"},
    }


def run_microbenches(binary: Path, dispatch: dict[str, Any], out: Path,
                     records: list[dict[str, Any]], threads: int, timeout: int) -> None:
    for name, overrides in candidate_envs(dispatch).items():
        env = dict(overrides)
        env.update({"QWEN_SHAPE_CENSUS": "1", "QWEN_V2_CENSUS": "1",
                    "QWEN_V2_CENSUS_JSON": str(out / f"v2-micro-{name}-%d.json")})
        run_cmd(records, f"microbench_{name}", [str(binary), "--matmat-bench", "-j", str(threads)],
                out, env, timeout=timeout, label="PERFORMANCE REQUIRED: kernel A/B")


def parse_values(raw: str) -> list[int]:
    vals = []
    for token in raw.split(","):
        try:
            value = int(token)
        except ValueError:
            continue
        if value > 0 and value not in vals:
            vals.append(value)
    return vals


def run_model_batch(binary: Path, model: Path, args: argparse.Namespace, out: Path,
                    records: list[dict[str, Any]], timeout: int) -> None:
    values = parse_values(args.batch_values)
    for B in values:
        arms = [("chunked", {"QWEN_BATCH_CHUNK_MAX_B": str(args.chunk_max)})]
        if B <= 16:
            arms = [("control", {})]
        elif args.no_unchunked:
            arms = [("chunked", {"QWEN_BATCH_CHUNK_MAX_B": str(args.chunk_max)})]
        else:
            arms = [("unchunked", {}), ("chunked", {"QWEN_BATCH_CHUNK_MAX_B": str(args.chunk_max)})]
        for arm, policy in arms:
            env = {"QWEN_BATCH_B": str(B), "QWEN_V2_CENSUS": "1",
                   "QWEN_V2_CENSUS_JSON": str(out / f"v2-batch-B{B}-{arm}-%d.json")}
            env.update(policy)
            run_cmd(records, f"batch_test_B{B}_{arm}",
                    [str(binary), "-d", str(model), "--batch-test", "-j", str(args.threads)],
                    out, env, timeout=timeout,
                    label="PARITY VERIFIED: Talker/CP batch oracle; census captured")


def server_env_merge(raw: str, census_pattern: str) -> str:
    vals: dict[str, str] = {}
    for token in raw.split(","):
        if "=" in token:
            k, v = token.split("=", 1)
            vals[k.strip()] = v.strip()
    vals["QWEN_V2_CENSUS"] = "1"
    vals["QWEN_V2_CENSUS_JSON"] = census_pattern
    return ",".join(f"{k}={v}" for k, v in sorted(vals.items()))


def run_server(binary: Path, model: Path, args: argparse.Namespace, out: Path,
               records: list[dict[str, Any]], timeout: int) -> None:
    topo = f"1x{max(1, args.threads)}"
    census = str(out / "v2-server-%d.json")
    env = server_env_merge(args.server_env, census)
    cmd = [sys.executable, "tests/serve_parallel_wave.py", "--model", str(model),
           "--bin", str(binary), "--topo", topo, "--conc", args.concurrency,
           "--waves", "1", "--batch-cap", str(max(parse_values(args.concurrency) or [1])),
           "--precision", args.precision, "--text-file", args.text_file,
           "--classes", args.classes, "--out", str(out / "server"), "--port", "9876",
           "--no-profile", "cpu-qualify explicit host qualification",
           "--server-env", env]
    rec = run_cmd(records, "server_v2_C1_C2_C4_C8", cmd, out,
                  {"QWEN_V2_CENSUS": "1", "QWEN_V2_CENSUS_JSON": census},
                  timeout=max(timeout, 1800), label="PERFORMANCE REQUIRED: v2 complete call")
    census_files = sorted(out.glob("v2-server-*.json"))
    rec["census_files"] = [str(p) for p in census_files]
    parallel = sorted((out / "server").glob("parallel_*.json")) if (out / "server").exists() else []
    rec["server_result_files"] = [str(p) for p in parallel]
    if not census_files:
        rec["status"] = "FAIL"
        rec["census_error"] = "server completed without QWEN_V2_CENSUS JSON"


def write_summary(out: Path, manifest: dict[str, Any]) -> None:
    records = manifest["commands"]
    lines = ["# Native CPU qualification", "", f"artifact: `{out}`", "",
             f"source: `{manifest.get('source_sha', 'unknown')}`",
             f"binary: `{manifest['binary']}` sha256 `{manifest['binary_sha256']}`",
             f"ISA class: `{manifest.get('isa_class', 'unknown')}`", "",
             "## Status"]
    for label, vals in manifest["status"].items():
        lines.append(f"- **{label}**: {vals}")
    lines += ["", "## Commands", "", "| command | result | evidence |", "|---|---|---|"]
    for r in records:
        lines.append(f"| `{r['name']}` | {r['status']} | {r.get('evidence', '')} |")
    lines += ["", "## Native limits", ""]
    for item in manifest["native_runtime_required"]:
        lines.append(f"- {item}")
    lines += ["", "## Artifacts", "", "- `hardware/` raw host evidence",
              "- `hardware.json` normalized host identity and flags",
              "- `dispatch.json` resolved engine policy",
              "- `v2-*.json` aggregated production/benchmark census files",
              "- command logs preserve the exact invocation and output"]
    (out / "summary.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()
    if args.threads < 1:
        raise SystemExit("--threads must be positive")
    binary = Path(args.binary).expanduser().resolve()
    if not binary.is_file():
        raise SystemExit(f"binary not found: {binary}")
    if args.serve and not args.model:
        raise SystemExit("--serve requires --model")
    model = Path(args.model).expanduser().resolve() if args.model else None
    if model and not model.is_dir():
        raise SystemExit(f"model directory not found: {model}")
    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out = Path(args.out).expanduser().resolve() if args.out else ROOT / "profiles" / "cpu-qualify" / stamp
    out.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    print(f"cpu-qualify: {out}")
    host = capture_hardware(out, records, args.timeout)
    dispatch = dispatch_and_engine(binary, out, records, args.timeout)
    make_checks(binary, out, records, args.timeout)
    run_microbenches(binary, dispatch, out, records, args.threads, args.timeout)
    if model:
        run_model_batch(binary, model, args, out, records, args.timeout)
        if args.serve:
            run_server(binary, model, args, out, records, args.timeout)

    try:
        source_sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                    capture_output=True, text=True, check=False).stdout.strip()
    except OSError:
        source_sha = "unknown"
    required_fail = [r for r in records if r.get("required") and r.get("status") != "PASS"]
    isa = str(dispatch.get("isa_class", "unknown"))
    native = [
        "x86_avx2: native runtime and performance still required on Haswell/Broadwell or Zen2/Zen3",
        "x86_avx512_no_vnni: native Skylake-SP runtime, frequency and complete-call A/B required",
        "x86_vnni/amx: native VNNI and AMX permission/shape qualification required",
        "arm_neon/dotprod: Linux HWCAP and old-Neoverse runtime qualification required",
        "arm_i8mm/kai: dotprod+i8mm packing and KAI GEMV/GEMM qualification required",
    ]
    status = {
        "COMPILE VERIFIED": "pass" if all(r["status"] == "PASS" for r in records
                                           if r["name"] in {"compile_isa_profiles", "flag_registry"}) else "fail",
        "PARITY VERIFIED": "pass" if all(r["status"] == "PASS" for r in records
                                          if r.get("evidence", "").startswith("PARITY VERIFIED")) else "fail",
        "PERFORMANCE REQUIRED": "recorded; native interpretation remains required",
        "NATIVE RUNTIME REQUIRED": "see native limits",
    }
    if not model:
        status["PERFORMANCE REQUIRED"] = "model-free kernel A/B recorded; model/v2 server run not requested"
        native.insert(0, "No --model was supplied: complete-call Talker/CP/decoder and C1/C2/C4/C8 server proof is pending")
    elif not args.serve:
        native.insert(0, "--serve was not requested: complete streaming C1/C2/C4/C8 proof is pending")
    manifest: dict[str, Any] = {
        "schema": "cpu-qualify/v1",
        "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "root": str(ROOT), "binary": str(binary), "binary_sha256": sha256_file(binary),
        "model": str(model) if model else None, "threads": args.threads,
        "host": host, "isa_class": isa, "source_sha": source_sha,
        "dispatch": dispatch, "status": status, "native_runtime_required": native,
        "commands": records, "return_code": 1 if required_fail else 0,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    write_summary(out, manifest)
    print(f"summary: {out / 'summary.md'}")
    print(f"result: {'FAIL' if required_fail else 'PASS'}  required_failures={len(required_fail)}")
    return 1 if required_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
