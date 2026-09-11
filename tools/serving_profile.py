#!/usr/bin/env python3
"""Strict preflight for the ISA serving lanes.

The deployment profiles in configs/perf describe both the requested environment and,
for the four parity lanes, the resolved implementation that must be observed before
traffic is allowed.  This tool runs only --caps/--dispatch-map; it does not start a
server or a benchmark.

    tools/serving_profile.py preflight amx-product --binary ./qwen_tts \
        --out result/profile-preflight.json
    tools/serving_profile.py check vnni-product --dispatch dispatch.json \
        --flags caps.txt

The JSON summary is deliberately compact and is suitable for embedding in a benchmark
manifest.  An INVALID FOR PROFILE fallback exits non-zero.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import perf_profile as P  # noqa: E402


def parse_env_string(text):
    out = {}
    if not text:
        return out
    for item in text.split(","):
        if not item:
            continue
        if "=" not in item:
            raise ValueError("server-env entry has no '=': %r" % item)
        key, value = item.split("=", 1)
        key, value = key.strip(), value.strip()
        if not key or not value or any(ch.isspace() for ch in value):
            raise ValueError("invalid server-env entry: %r" % item)
        out[key] = value
    return out


def parse_flags(text):
    seen = {}
    for line in text.splitlines():
        if not line.startswith("[FLAGS]"):
            continue
        for token in line.split()[2:]:
            if "=" in token:
                key, value = token.split("=", 1)
                seen[key] = value
    return seen


def merge_profile_env(prof, overrides):
    """Apply explicit overrides while preserving the profile parity contract.

    ``parity.tunable_flags`` names host/ISA knobs that a campaign may vary.  The
    effective value still goes through the engine-owned dispatch and flag checks;
    all other values pinned by the profile remain immutable.
    """
    parity = prof.get("parity") or {}
    tunable = set(parity.get("tunable_flags", []))
    base = P.environ(prof)
    errors = []
    for key, value in overrides.items():
        if key in base and base[key] != value and key not in tunable:
            errors.append("override %s=%s changes the parity profile value %s" %
                          (key, value, base[key]))
        base[key] = value
    return base, errors


def feature_index(doc):
    return {row.get("id"): row for row in doc.get("features", [])}


def resolved_feature_status(features, feature_id, active):
    """Classify an optional path from the engine's own compiled/support row.

    A false boolean is not enough: a deliberately disabled path is a VALID FALLBACK,
    while a feature that is not compiled or not supported on this host is UNSUPPORTED.
    The distinction is part of the lane contract and prevents an ISA product profile
    from quietly accepting a missing backend.
    """
    if active:
        return "ACTIVE"
    row = features.get(feature_id, {})
    if row.get("compiled") == "no" or row.get("supported") == "no":
        return "UNSUPPORTED"
    return "VALID FALLBACK"


def status_allowed(wanted, actual):
    return actual in (wanted if isinstance(wanted, list) else [wanted])


def family_kind(value):
    v = (value or "").lower()
    if "kleidi" in v or "kai" in v:
        return "kai"
    if "amx" in v:
        return "amx"
    if "vnni" in v:
        return "vnni"
    if "bf16" in v:
        return "bf16"
    if "f32" in v or "sgemm" in v or "generic twin" in v:
        return "f32"
    return "unknown"


def expected_backend_ok(expected, actual, prefill_active=False):
    if expected in (None, "", "native-per-isa", "any"):
        return True
    if expected == "f32":
        # The resolver row is authoritative for the Talker prefill choice.  The generic
        # family probe is a B>1 matmat probe and may quite legitimately say "bf16 fixed-B
        # twin" even though the prefill predicate is OFF and the actual prefill call takes
        # its f32/BLAS fallback.
        return not prefill_active
    if expected == "bf16-native":
        return prefill_active and actual in ("amx", "bf16", "kai")
    return family_kind(actual) == expected


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def run_binary(binary, env, dispatch_path):
    probe_env = dict(os.environ)
    probe_env.update(env)
    probe_env["QWEN_DISPATCH_JSON"] = dispatch_path
    dispatch = subprocess.run([binary, "--dispatch-map"], env=probe_env,
                              capture_output=True, text=True, timeout=120)
    caps = subprocess.run([binary, "--caps"], env=probe_env,
                          capture_output=True, text=True, timeout=120)
    return dispatch, caps, probe_env


def evaluate(prof, dispatch, flags, binary=None, profile_env=None, errors=None,
             process_env=None):
    errors = list(errors or [])
    parity = prof.get("parity")
    if not parity:
        errors.append("profile has no parity contract")
        return {"profile_valid": False, "errors": errors}

    serving = dispatch.get("serving", {})
    features = feature_index(dispatch)
    isa = dispatch.get("isa_class", "unknown")
    if isa not in parity["isa_classes"]:
        errors.append("isa=%s is not allowed by lane %s (expected %s)" %
                      (isa, parity["lane"], ", ".join(parity["isa_classes"])))

    dec = parity["decoder"]
    mode = serving.get("decoder_mode")
    if not mode:
        mode = features.get("decoder.mode", {}).get("resolved")
    if mode not in dec["expected_modes"]:
        errors.append("resolved_decoder_mode=%r, expected one of %s" %
                      (mode, ", ".join(dec["expected_modes"])))
    batch = serving.get("decoder_batch_requested")
    if batch is None:
        raw = flags.get("QWEN_DECODER_BATCH")
        batch = raw not in (None, "", "0")
    if bool(batch) != bool(dec["requested_batch"]):
        errors.append("requested_decoder_batch=%r, expected %r" %
                      (bool(batch), bool(dec["requested_batch"])))

    if "bf16" in (mode or ""):
        precision = "bf16"
    elif "int8" in (mode or ""):
        precision = "int8"
    else:
        precision = "fp32"
    if precision != dec["precision"]:
        errors.append("decoder_precision=%s, expected %s" % (precision, dec["precision"]))

    # The server-level batch request is common, but only the AMX ragged leaf consumes
    # it as one decoder workset at this HEAD.  A requested batch that resolves to the
    # per-item leaf is therefore an intentional VALID FALLBACK for the VNNI/Arm product
    # lanes, while QWEN_DECODER_BATCH=0 makes the same per-item leaf ACTIVE in the common
    # control.  Keep this distinction in the machine-readable result instead of making
    # callers infer it from a mode string.
    if not mode:
        decoder_status = "UNSUPPORTED"
    elif dec["requested_batch"] and mode.startswith("per-item-"):
        decoder_status = "VALID FALLBACK"
    else:
        decoder_status = "ACTIVE"
    if decoder_status != dec["fallback_status"]:
        errors.append("decoder fallback status=%s, expected %s" %
                      (decoder_status, dec["fallback_status"]))

    design = bool(serving.get("design_d_active", False))
    fused = bool(serving.get("fused_residual_active", False))
    kai = bool(serving.get("kleidi_active", False))
    res1_v2 = bool(serving.get("res1_v2_active", False)) or \
        features.get("decoder.res1_v2", {}).get("resolved") == "ON"
    lane = bool(serving.get("decoder_lane_active", False)) or \
        features.get("decoder.lane", {}).get("resolved") == "ON"
    multislot = bool(serving.get("decoder_multislot_active", False)) or \
        features.get("decoder.multislot", {}).get("resolved") == "ON"
    actual_status = {
        "design_d": resolved_feature_status(features, "decoder.design_d", design),
        "fused_residual": resolved_feature_status(features, "decoder.fused_residual", fused),
        "kleidi": resolved_feature_status(features, "kleidi.enabled", kai),
        # Optional contracts (DL-4 conv, DL-1/DL-2 decoder lane): checked only when the
        # profile names them, so the older lanes keep their three-feature contract.
        "res1_v2": resolved_feature_status(features, "decoder.res1_v2", res1_v2),
        "decoder_lane": resolved_feature_status(features, "decoder.lane", lane),
        "multislot": resolved_feature_status(features, "decoder.multislot", multislot),
    }
    actual_status = {k: v for k, v in actual_status.items() if k in parity["features"]}
    for name, actual in actual_status.items():
        wanted = parity["features"][name]
        if not status_allowed(wanted, actual):
            errors.append("%s is %s, but the profile allows %s" %
                          (name, actual, ", ".join(wanted if isinstance(wanted, list) else [wanted])))

    talker = serving.get("talker_cp_int8_backend", "")
    q4 = serving.get("q4_backend", "")
    bf16 = serving.get("bf16_backend", "")
    prefill_active = bool(serving.get("prefill_matmat_active", False))
    backend_actual = {
        "talker_cp": family_kind(talker),
        # qwen_matmat_family_bf16() is a capability probe for a representative
        # matmat shape.  It is not the resolved prefill leaf when the prefill
        # predicate is off; in that case the dispatch map explicitly says the
        # operation takes the f32/BLAS fallback.
        "prefill": family_kind(bf16) if prefill_active else "f32",
        "q4": family_kind(q4),
    }
    for name, actual, raw in (
        ("talker_cp", backend_actual["talker_cp"], talker),
        ("q4", backend_actual["q4"], q4),
        ("prefill", backend_actual["prefill"], bf16),
    ):
        wanted = parity["backends"][name]
        if not expected_backend_ok(wanted, actual, prefill_active if name == "prefill" else False):
            errors.append("%s backend=%s (%s), expected %s" % (name, actual, raw, wanted))

    # Explicitly requested QWEN values must be visible in the engine's own [FLAGS].
    # Non-QWEN controls such as OPENBLAS_THREAD_TIMEOUT are process-environment settings,
    # not engine flags, so validate them against the exact probe environment instead of
    # incorrectly requiring them in [FLAGS]. Null entries are checked by the caller.
    if profile_env is not None:
        # Include streaming settings injected by perf_profile.environ(), not only the
        # runtime.environment map.  Otherwise decode quantum could drift while the
        # backend contract looked pinned.
        for key, value in profile_env.items():
            if key.startswith("QWEN_") and flags.get(key) != value:
                errors.append("engine [FLAGS] %s=%r, profile requested %r" %
                              (key, flags.get(key), value))
            elif not key.startswith("QWEN_") and process_env is not None \
                    and process_env.get(key) != value:
                errors.append("process environment %s=%r, profile requested %r" %
                              (key, process_env.get(key), value))

    summary = {
        "profile": prof["profile"]["id"],
        "lane": parity["lane"],
        "isa": isa,
        "profile_valid": not errors,
        "errors": errors,
        "requested_decoder_batch": int(bool(dec["requested_batch"])),
        "resolved_decoder_mode": mode,
        "decoder_precision": precision,
        "talker_backend": backend_actual["talker_cp"],
        "cp_backend": backend_actual["talker_cp"],
        "prefill_backend": backend_actual["prefill"],
        "q4_backend": backend_actual["q4"],
        "fused_residual_active": fused,
        "design_d_active": design,
        "kai_active": kai,
        "res1_v2_active": res1_v2,
        "decoder_lane_active": lane,
        "multislot_active": multislot,
        "decoder_lane_elastic": bool(serving.get("decoder_lane_elastic", False)),
        "fallback_detected": decoder_status == "VALID FALLBACK" or
                             any(v == "VALID FALLBACK" for v in actual_status.values()),
        "fallback_status": decoder_status,
        "expected_fallback_status": dec["fallback_status"],
        "decoder_fallback_status": decoder_status,
        "feature_status": actual_status,
        "resolved_dispatch": {
            "build": dispatch.get("build"),
            "simd": dispatch.get("simd"),
            "source_fp": dispatch.get("source_fp"),
            "decoder_pool": serving.get("decoder_pool"),
            "prefill_matmat_active": prefill_active,
            "prefill_reason": serving.get("prefill_reason"),
            "talker_cp_int8_backend": talker,
            "q4_backend": q4,
            "bf16_backend": bf16,
        },
        "profile_flags": P.environ(prof),
        "effective_profile_flags": profile_env,
        "tunable_flags": sorted(parity.get("tunable_flags", [])),
        "profile_forbidden_flags": P.forbidden_env(prof),
        "quality_gate": parity["quality_gate"],
    }
    if binary:
        summary["binary"] = binary
        try:
            summary["binary_sha256"] = sha256(binary)
        except OSError:
            summary["binary_sha256"] = None
    return summary


def write_summary(summary, path):
    if path:
        with open(path, "w") as f:
            json.dump(summary, f, indent=2, sort_keys=True)
            f.write("\n")


def preflight(args):
    prof, _ = P.load(args.profile)
    parity = prof.get("parity")
    if not parity:
        print("FAIL: %s has no parity contract" % args.profile, file=sys.stderr)
        return 2
    errors = []
    try:
        overrides = parse_env_string(args.server_env)
    except ValueError as e:
        print("FAIL: %s" % e, file=sys.stderr)
        return 2
    base, override_errors = merge_profile_env(prof, overrides)
    errors.extend(override_errors)
    for key in P.forbidden_env(prof):
        if key in os.environ and key not in overrides:
            errors.append("%s is declared absent by the profile but is present in the parent environment" % key)
        if key in overrides:
            errors.append("%s is declared absent by the profile but was supplied as an override" % key)
    if not os.path.isfile(args.binary) or not os.access(args.binary, os.X_OK):
        errors.append("binary is not executable: %s" % args.binary)
        summary = {"profile": args.profile, "lane": parity["lane"],
                   "profile_valid": False, "errors": errors}
        write_summary(summary, args.out)
        print(json.dumps(summary, sort_keys=True))
        return 1

    with tempfile.TemporaryDirectory(prefix="qwen-serving-profile-") as td:
        dispatch_path = os.path.join(td, "dispatch.json")
        try:
            dispatch_run, caps_run, probe_env = run_binary(args.binary, base, dispatch_path)
        except (OSError, subprocess.TimeoutExpired) as e:
            errors.append("probe failed: %s" % e)
            dispatch_run = caps_run = None
            probe_env = base
        dispatch = {}
        if dispatch_run is not None:
            if dispatch_run.returncode != 0:
                errors.append("--dispatch-map failed: %s" %
                              (dispatch_run.stderr or dispatch_run.stdout).strip()[-500:])
            if os.path.exists(dispatch_path):
                try:
                    dispatch = json.load(open(dispatch_path))
                except (OSError, json.JSONDecodeError) as e:
                    errors.append("invalid dispatch JSON: %s" % e)
        if caps_run is not None and caps_run.returncode != 0:
            errors.append("--caps failed: %s" % (caps_run.stderr or caps_run.stdout).strip()[-500:])
        flags = parse_flags((caps_run.stdout + "\n" + caps_run.stderr) if caps_run else "")
        summary = evaluate(prof, dispatch, flags, args.binary, base, errors,
                           process_env=probe_env)
        summary["host_env_overrides"] = sorted(overrides)
        summary["flags_observed"] = flags
        summary["source_fingerprint"] = dispatch.get("source_fp")
        summary["source_commit"] = dispatch.get("build")
        write_summary(summary, args.out)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0 if summary.get("profile_valid") else 1


def check(args):
    prof, _ = P.load(args.profile)
    dispatch = json.load(open(args.dispatch))
    flags = parse_flags(open(args.flags, errors="replace").read()) if args.flags else {}
    summary = evaluate(prof, dispatch, flags, args.binary, None)
    write_summary(summary, args.out)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary.get("profile_valid") else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("preflight")
    p.add_argument("profile")
    p.add_argument("--binary", default="./qwen_tts")
    p.add_argument("--server-env", default="", help="exact comma-separated overrides")
    p.add_argument("--out", default="")
    p.set_defaults(fn=preflight)
    c = sub.add_parser("check")
    c.add_argument("profile")
    c.add_argument("--dispatch", required=True)
    c.add_argument("--flags", default="")
    c.add_argument("--binary", default="")
    c.add_argument("--out", default="")
    c.set_defaults(fn=check)
    args = ap.parse_args()
    try:
        return args.fn(args)
    except P.Bad as e:
        print("FAIL: %s" % e, file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
