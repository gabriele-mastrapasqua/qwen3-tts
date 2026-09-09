#!/usr/bin/env python3
"""Validation for the server performance profiles."""
import json, os, subprocess, sys, tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import perf_profile as P  # noqa: E402

FAILURES = []

def check(name, cond, detail=""):
    print(f"{'ok  ' if cond else 'FAIL'} {name}" + (f"  — {detail}" if detail and not cond else ""))
    if not cond:
        FAILURES.append(name)

def rejects(name, mutate):
    """A profile mutated this way must be refused, not repaired."""
    prof, path = P.load("axion-16c-ttfa")
    prof = json.loads(json.dumps(prof))
    mutate(prof)
    with open(os.path.join(P.PERF, "schema.json")) as f:
        schema = json.load(f)
    try:
        P.check(prof, schema, schema, "$")
        errs = P.semantic(prof, path, engine=True)
        refused = bool(errs)
        why = errs[0] if errs else "accepted"
    except P.Bad as e:
        refused, why = True, str(e)
    check(name, refused, why)

r = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "perf_profile.py"),
                    "validate", "--engine", "1"], capture_output=True, text=True)
check("committed profiles validate", r.returncode == 0, r.stdout + r.stderr)

rejects("unknown field is refused", lambda p: p["server"].update(mystery_knob=1))
rejects("missing required field is refused", lambda p: p["server"].pop("batch_size"))
rejects("oversubscription is refused",
        lambda p: p["server"].update(threads_per_worker=16))
rejects("whitespace in an environment value is refused",
        lambda p: p["runtime"]["environment"].update(
            {"QWEN_PREFIX_CACHE": {"value": "1 QWEN_SD_INT8=1", "why": "the space-separated form"}}))
rejects("an environment variable the engine never declares is refused",
        lambda p: p["runtime"]["environment"].update(
            {"QWEN_NOT_A_REAL_FLAG": {"value": "1", "why": "typo"}}))
rejects("a setting fixed twice is refused",
        lambda p: p["launch"].update(extra_arguments=["--prefork", "4"]))
rejects("a malformed profile id is refused", lambda p: p["profile"].update(id="Axion 16C"))

raw, path = P.load_raw("recommended")
raw["server"] = {"prefork_workers": 4, "threads_per_worker": 4, "batch_size": 4}
try:
    P.shape_check(raw, path); ok = False
except P.Bad:
    ok = True
check("an alias carrying its own values is refused", ok)

a, _ = P.load("recommended")
b, _ = P.load("axion-16c-ttfa")
check("recommended resolves to axion-16c-ttfa", a == b)

prof, _ = P.load("recommended")
argv = P.argv(prof, "MODEL", 9000)
def val(flag):
    return argv[argv.index(flag) + 1] if flag in argv else None
check("resolved topology is the qualified 2x8",
      val("--prefork") == "2" and val("--prefork-threads") == "8",
      f"got {val('--prefork')}x{val('--prefork-threads')}")
check("resolved batch cap is the qualified 8", val("--batch-size") == "8", f"got {val('--batch-size')}")
check("resolved precision carries --int8", "--int8" in argv)
check("server-env is comma separated",
      "," in P.server_env(prof) and " " not in P.server_env(prof), P.server_env(prof))

for lane in ("amx-product", "vnni-product", "vnni-bf16-product", "turin-c8a-32c-vnni-product", "arm-product", "common-control"):
    lane_prof, _ = P.load(lane)
    lane_env = P.environ(lane_prof)
    check(f"{lane} pins the known-text streaming layout",
          lane_env.get("QWEN_TTS_STREAM_LAYOUT") == "1",
          f"got {lane_env.get('QWEN_TTS_STREAM_LAYOUT')!r}")

readme = open(os.path.join(P.PERF, "README.md")).read()
for token, why in ("2 workers x 8 threads", "the topology"), ("--int8", "the precision"), \
                  ("OPENBLAS_THREAD_TIMEOUT", "the OpenBLAS idle policy"):
    check(f"README still states {why}", token in readme)
for k, spec in prof["runtime"]["environment"].items():
    check(f"{k} carries a reason", bool(spec.get("why", "").strip()))

with tempfile.NamedTemporaryFile("w", suffix=".log", delete=False) as f:
    f.write("Server listening on http://0.0.0.0:9000\n")
    silent = f.name
r = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "perf_profile.py"),
                    "check-flags", "recommended", "--log", silent], capture_output=True, text=True)
check("a log without a [FLAGS] declaration fails the check", r.returncode != 0, r.stdout)
with open(silent, "a") as f:
    _env = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "perf_profile.py"),
                           "server-env", "recommended"], capture_output=True, text=True).stdout.strip()
    _qwen = " ".join(kv for kv in _env.split(",") if kv.startswith("QWEN_"))
    f.write(f"[FLAGS] v=1 pid=1 {_qwen}\n")
r = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "perf_profile.py"),
                    "check-flags", "recommended", "--log", silent], capture_output=True, text=True)
check("a declaration matching the profile passes", r.returncode == 0, r.stdout)
os.unlink(silent)

print(f"\n{'FAILED: ' + ', '.join(FAILURES) if FAILURES else 'all checks passed'}")
sys.exit(1 if FAILURES else 0)

# vnni-bf16-product: the VNNI lane for a CPU with native AVX-512 BF16 must claim the bf16 prefill
# (the f32 pin of vnni-product cost ~600 ms per admission on Zen5, 2026-09-09) and the measured spin.
bf16_prof, _ = P.load("vnni-bf16-product")
bf16_env = P.environ(bf16_prof)
check("vnni-bf16-product enables the native bf16 prefill",
      bf16_env.get("QWEN_PREFILL_MATMAT") == "1" and bf16_env.get("QWEN_NO_BF16_MATMUL") == "0" and bf16_env.get("QWEN_NO_BF16DOT") == "0", bf16_env)
check("vnni-bf16-product declares the bf16-native prefill backend", bf16_prof["parity"]["backends"]["prefill"] == "bf16-native")
check("VNNI product lanes carry the measured Turin spin budget",
      bf16_env.get("QWEN_POOL_SPIN") == "65536" and P.environ(P.load("vnni-product")[0]).get("QWEN_POOL_SPIN") == "65536")
