#!/usr/bin/env python3
"""Small offline tests for the strict cross-ISA serving-profile gate."""
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import perf_profile as P  # noqa: E402
import serving_profile as S  # noqa: E402


FAILURES = []


def check(name, condition, detail=""):
    print(f"{'ok  ' if condition else 'FAIL'} {name}" +
          (f" — {detail}" if detail and not condition else ""))
    if not condition:
        FAILURES.append(name)


def dispatch_for(isa, mode, *, batch, design, fused, kai, prefill, talker, q4, bf16):
    def row(ident, compiled, supported):
        return {"id": ident, "compiled": "yes" if compiled else "no",
                "supported": "yes" if supported else "no", "resolved": "ON"}

    return {
        "v": 1, "isa_class": isa, "build": "test", "simd": "test",
        "source_fp": "test:clean", "features": [
            row("decoder.design_d", design, design),
            row("decoder.fused_residual", fused, fused),
            row("kleidi.enabled", kai, kai),
        ],
        "serving": {
            "decoder_batch_requested": bool(batch),
            "decoder_mode": mode,
            "design_d_active": bool(design and mode == "ragged-design-d-int8"),
            "fused_residual_active": bool(fused),
            "stream_strip_active": bool(design),
            "decoder_pool": "engine",
            "talker_cp_int8_backend": talker,
            "q4_backend": q4,
            "bf16_backend": bf16,
            "prefill_matmat_active": bool(prefill),
            "prefill_reason": "test",
            "kleidi_active": bool(kai),
        },
    }


def run_case(profile_name, dispatch):
    profile, _ = P.load(profile_name)
    flags = dict(P.environ(profile))
    return S.evaluate(profile, dispatch, flags, profile_env=P.environ(profile))


cases = [
    ("amx-product", dispatch_for(
        "x86_amx", "ragged-design-d-int8", batch=1, design=True, fused=True,
        kai=False, prefill=True, talker="AMX INT8", q4="AMX Q4", bf16="AMX BF16")),
    ("vnni-product", dispatch_for(
        "x86_avx512vnni", "per-item-int8-vnni", batch=1, design=False, fused=False,
        kai=False, prefill=False, talker="INT8 VNNI", q4="Q4 VNNI", bf16="bf16 fixed-B twin")),
    ("arm-product", dispatch_for(
        "arm_i8mm_bf16", "per-item-int8-dotprod", batch=1, design=False, fused=False,
        kai=True, prefill=True, talker="KleidiAI INT8", q4="KleidiAI Q4", bf16="KleidiAI BF16")),
    ("common-control", dispatch_for(
        "x86_amx", "per-item-int8-vnni", batch=0, design=False, fused=False,
        kai=False, prefill=False, talker="INT8 VNNI", q4="Q4 VNNI", bf16="bf16 fixed-B twin")),
]
for name, dispatch in cases:
    result = run_case(name, dispatch)
    check(f"{name} accepts its resolved contract", result.get("profile_valid"),
          json.dumps(result.get("errors", [])))

bad = dispatch_for(
    "x86_amx", "per-item-design-d-int8", batch=1, design=True, fused=False,
    kai=False, prefill=True, talker="AMX INT8", q4="AMX Q4", bf16="AMX BF16")
result = run_case("amx-product", bad)
check("AMX product rejects a non-ragged decoder leaf", not result.get("profile_valid"),
      json.dumps(result.get("errors", [])))

bad = dispatch_for(
    "x86_avx512vnni", "per-item-int8-vnni", batch=1, design=False, fused=False,
    kai=False, prefill=False, talker="f32-accum twin", q4="Q4 VNNI", bf16="bf16 fixed-B twin")
result = run_case("vnni-product", bad)
check("VNNI product rejects an unintended generic Talker fallback", not result.get("profile_valid"),
      json.dumps(result.get("errors", [])))

profile, _ = P.load("amx-product")
argv = P.argv(profile, "MODEL", 8000)
check("host-unspecified profile does not invent prefork topology",
      "--prefork" not in argv and "--prefork-threads" not in argv)

validation = subprocess.run(
    [sys.executable, os.path.join(ROOT, "tools", "perf_profile.py"),
     "validate", "--engine", os.path.join(ROOT, "qwen_tts")],
    cwd=ROOT, capture_output=True, text=True,
)
check("all committed profiles validate", validation.returncode == 0,
      validation.stdout + validation.stderr)

print(f"\n{'FAILED: ' + ', '.join(FAILURES) if FAILURES else 'all checks passed'}")
sys.exit(1 if FAILURES else 0)
