#!/usr/bin/env python3
"""dispatch_gate.py — expected-vs-observed dispatch for this host's ISA class.

    tools/dispatch_gate.py <dispatch.json> [--expect tools/dispatch_expect.json] [--warn-only]
    tools/dispatch_gate.py --selftest

Reads the JSON that `./qwen_tts --dispatch-map` writes (QWEN_DISPATCH_JSON=path) and:

  1. applies the per-class expectations of tools/dispatch_expect.json;
  2. applies two generic rules that need no expectation table:
       - a gate row that is compiled, supported, opt-out (no on_env) and still OFF
         without an explicit env switch is a fallback nobody asked for;
       - talker.prefill.f32_blas_fallback ON while talker.prefill.matmat_bf16 is
         compiled and supported is the 2026-09-03 bug, on any class.
       - QWEN_PREFILL_MATMAT=1 while no native BF16 unit is compiled/supported is an
         unsatisfied request, even when the class expectation historically allowed OFF.

Verdict per finding:
  SUSPICIOUS  expected ON, compiled+supported on this host, resolved OFF  -> exit 1
  MISMATCH    any other expected/observed disagreement                    -> exit 1
  NOTE        expectation refers to a feature this binary does not carry   -> informational

--warn-only prints the same lines but always exits 0 (for exploratory builds).
"""
import argparse, json, os, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_EXPECT = os.path.join(ROOT, "tools", "dispatch_expect.json")


def index(doc):
    rows = {}
    for f in doc.get("features", []):
        rows[f["id"]] = {
            "kind": "feature", "compiled": f.get("compiled") == "yes",
            "supported": f.get("supported") == "yes", "supported_raw": f.get("supported"),
            "resolved": f.get("resolved", ""), "env": f.get("env", ""),
            "env_value": f.get("env_value", ""), "reason": f.get("reason", ""),
            "on_env": "", "off_env": "",
        }
    for g in doc.get("gates", []):
        rows[g["id"]] = {
            "kind": "gate", "compiled": bool(g.get("compiled")),
            "supported": bool(g.get("supported")), "supported_raw": g.get("supported"),
            "resolved": "ON" if g.get("on") else "OFF", "env": g.get("off_env") or g.get("on_env") or "",
            "env_value": "", "reason": g.get("reason", ""),
            "on_env": g.get("on_env", ""), "off_env": g.get("off_env", ""),
        }
    return rows


def explicit_env_set(row):
    """True when an operator variable explains the state (then it is a choice, not a bug)."""
    for name in (row.get("env"), row.get("on_env"), row.get("off_env")):
        if name and os.environ.get(name) not in (None, ""):
            return True
    if row["kind"] == "feature" and row.get("env_value") not in ("", "unset"):
        return True
    return False


def evaluate(doc, expect_all):
    cls = doc.get("isa_class", "unknown")
    rows = index(doc)
    findings = []   # (verdict, id, text)
    expect = expect_all.get(cls, {})
    if not expect:
        findings.append(("NOTE", "-", f"no expectation table for isa_class={cls}; generic rules only"))

    for fid, want in expect.items():
        row = rows.get(fid)
        if row is None:
            findings.append(("NOTE", fid, f"expected {want}, feature not reported by this binary"))
            continue
        got = row["resolved"]
        if want == "ON":     ok = got.startswith("ON")      # "ON" and "ON*" both satisfy ON
        elif want == "OFF":  ok = (got == "OFF")
        else:                ok = got.startswith(want.rstrip("*"))
        if ok:
            continue
        if want.startswith("ON") and got == "OFF":
            if row["compiled"] and row["supported"] and not explicit_env_set(row):
                findings.append(("SUSPICIOUS", fid,
                                 f"expected ON for {cls}; compiled=yes supported=yes resolved=OFF; reason: {row['reason']}"))
            elif not row["compiled"]:
                findings.append(("NOTE", fid, f"expected ON for {cls} but NOT COMPILED in this binary (rebuild with the right SIMD=?)"))
            elif not row["supported"]:
                findings.append(("MISMATCH", fid, f"expected ON for {cls} but the host does not support it ({row['reason']})"))
            else:
                findings.append(("MISMATCH", fid, f"expected ON for {cls}; OFF by explicit env ({row['env']}); reason: {row['reason']}"))
        else:
            findings.append(("MISMATCH", fid, f"expected {want} for {cls}, observed {got}; reason: {row['reason']}"))

    # generic rule 1: an opt-out gate that is compiled+supported and OFF without an env switch
    for fid, row in rows.items():
        if row["kind"] != "gate" or fid in expect:
            continue
        if row["compiled"] and row["supported"] and row["resolved"] == "OFF" \
                and not row["on_env"] and not explicit_env_set(row):
            findings.append(("SUSPICIOUS", fid, f"opt-out gate compiled+supported but OFF: {row['reason']}"))

    # generic rule 2: the prefill fallback while the bf16 unit is there
    mb = rows.get("talker.prefill.matmat_bf16")
    fb = rows.get("talker.prefill.f32_blas_fallback")
    if mb and fb and fb["resolved"] == "ON" and mb["compiled"] and mb["supported"] \
            and "talker.prefill.matmat_bf16" not in expect and not explicit_env_set(mb):
        findings.append(("SUSPICIOUS", "talker.prefill.f32_blas_fallback",
                         "f32/SGEMM prefill selected although a bf16 matmat unit is compiled and supported"))

    # generic rule 3: an explicit native-prefill request must be satisfiable.  This is
    # deliberately checked even for x86_avx512vnni, whose historical expectation is
    # OFF: that class can still receive QWEN_PREFILL_MATMAT=1 and otherwise enter the
    # generic BF16 twin, which is neither AVX-512 BF16 nor AMX.
    requested_unavailable = (
        mb and mb.get("env") == "QWEN_PREFILL_MATMAT" and mb.get("env_value") == "1"
        and (not mb["compiled"] or not mb["supported"]
             or "no native" in mb.get("reason", "")
             or "no compiled" in mb.get("reason", "")))
    if requested_unavailable and not any(fid == "talker.prefill.matmat_bf16" for _, fid, _ in findings):
        findings.append(("SUSPICIOUS", "talker.prefill.matmat_bf16",
                         "QWEN_PREFILL_MATMAT=1 cannot be honored by this binary/CPU: "
                         "no native BF16 matmat unit; the run would use a fallback"))
    return cls, findings


def report(cls, findings, out=sys.stdout):
    order = {"SUSPICIOUS": 0, "MISMATCH": 1, "NOTE": 2}
    findings = sorted(findings, key=lambda t: (order[t[0]], t[1]))
    bad = sum(1 for v, _, _ in findings if v in ("SUSPICIOUS", "MISMATCH"))
    print(f"DISPATCH GATE  isa_class={cls}", file=out)
    if not findings:
        print("  [PASS] observed dispatch matches the expectation for this class", file=out)
    for v, fid, txt in findings:
        print(f"  [{v}] {fid}: {txt}", file=out)
    return bad


def selftest():
    """The regression test for the bug this gate exists for: an AVX-512-BF16 host whose
    prefill resolved to the f32/SGEMM fallback must be flagged SUSPICIOUS."""
    doc = {"isa_class": "x86_avx512bf16", "features": [
        {"id": "talker.prefill.matmat_bf16", "compiled": "yes", "supported": "yes",
         "env": "QWEN_PREFILL_MATMAT", "env_value": "unset", "resolved": "OFF",
         "reason": "predicate returned 0"},
        {"id": "talker.prefill.f32_blas_fallback", "compiled": "yes", "supported": "yes",
         "env": "", "env_value": "", "resolved": "ON", "reason": "bf16->f32 + SGEMM"},
        {"id": "decoder.int8", "compiled": "yes", "supported": "yes", "env": "QWEN_SD_INT8",
         "env_value": "0", "resolved": "OFF", "reason": "explicit env"},
    ], "gates": [
        {"id": "gate.int8.vnni", "kernel": "int8 VNNI", "compiled": True, "supported": True,
         "on": True, "off_env": "QWEN_NO_VNNI", "on_env": "", "reason": "default ON"},
        {"id": "gate.bf16.avx512", "kernel": "bf16 AVX-512", "compiled": True, "supported": True,
         "on": False, "off_env": "QWEN_NO_BF16_MATMUL", "on_env": "", "reason": "default ON"},
    ]}
    expect = json.load(open(DEFAULT_EXPECT))
    os.environ.pop("QWEN_PREFILL_MATMAT", None)
    os.environ.pop("QWEN_NO_BF16_MATMUL", None)
    cls, f = evaluate(doc, expect)
    ids = {(v, i) for v, i, _ in f}
    requested_doc = {"isa_class": "x86_avx512vnni", "features": [
        {"id": "talker.prefill.matmat_bf16", "compiled": "no", "supported": "no",
         "env": "QWEN_PREFILL_MATMAT", "env_value": "1", "resolved": "OFF",
         "reason": "explicit QWEN_PREFILL_MATMAT=1 but no native BF16 unit -> fallback"},
        {"id": "talker.prefill.f32_blas_fallback", "compiled": "yes", "supported": "yes",
         "env": "", "env_value": "", "resolved": "ON", "reason": "fallback"},
    ]}
    _, requested_findings = evaluate(requested_doc, expect)
    requested_ids = {(v, i) for v, i, _ in requested_findings}
    legacy_no_vnni_doc = {"isa_class": "x86_avx512f_no_vnni", "features": [
        {"id": "talker.prefill.matmat_bf16", "compiled": "no", "supported": "no",
         "env": "QWEN_PREFILL_MATMAT", "env_value": "unset", "resolved": "OFF",
         "reason": "AVX-512 BF16 unit not compiled"},
        {"id": "talker.prefill.f32_blas_fallback", "compiled": "yes", "supported": "yes",
         "env": "", "env_value": "", "resolved": "ON", "reason": "f32->SGEMM fallback"},
        {"id": "decoder.int8", "compiled": "yes", "supported": "yes",
         "env": "QWEN_SD_INT8", "env_value": "unset", "resolved": "OFF",
         "reason": "AVX2 decoder candidate pending qualification"},
        {"id": "decoder.res1_v2", "compiled": "yes", "supported": "yes",
         "env": "QWEN_SD_RES1_V2", "env_value": "unset", "resolved": "OFF",
         "reason": "opt-in default OFF"},
        {"id": "decoder.glue_fused", "compiled": "yes", "supported": "yes",
         "env": "QWEN_SD_GLUE", "env_value": "unset", "resolved": "OFF",
         "reason": "opt-in default OFF"},
        {"id": "matvec.int8.avx512bw-emulated-dot-gemv", "compiled": "yes", "supported": "yes",
         "env": "QWEN_AVX512_INT8_GEMV", "env_value": "unset", "resolved": "OFF",
         "reason": "opt-in default OFF"},
        {"id": "matvec.q4.avx512bw-emulated-dot-gemv", "compiled": "yes", "supported": "yes",
         "env": "QWEN_AVX512_Q4_GEMV", "env_value": "unset", "resolved": "OFF",
         "reason": "opt-in default OFF"},
        {"id": "matvec.int8.avx2-emulated-dot-gemv", "compiled": "yes", "supported": "yes",
         "env": "QWEN_AVX2_INT8_GEMV", "env_value": "unset", "resolved": "OFF",
         "reason": "opt-in default OFF"},
        {"id": "matvec.q4.avx2-emulated-dot-gemv", "compiled": "yes", "supported": "yes",
         "env": "QWEN_AVX2_Q4_GEMV", "env_value": "unset", "resolved": "OFF",
         "reason": "opt-in default OFF"},
    ], "gates": [
        {"id": "gate.int8.avx2", "kernel": "int8 AVX2 matmat", "compiled": True,
         "supported": True, "on": True, "off_env": "QWEN_NO_AVX2MM", "on_env": "",
         "reason": "default ON"},
        {"id": "gate.q4.avx2", "kernel": "q4 AVX2 matmat", "compiled": True,
         "supported": True, "on": True, "off_env": "QWEN_NO_AVX2MM", "on_env": "",
         "reason": "default ON"},
    ]}
    _, legacy_no_vnni_findings = evaluate(legacy_no_vnni_doc, expect)
    ok = ("SUSPICIOUS", "talker.prefill.matmat_bf16") in ids \
        and ("SUSPICIOUS", "gate.bf16.avx512") in ids \
        and ("MISMATCH", "decoder.int8") in ids \
        and not any(i == "gate.int8.vnni" for _, i, _ in f) \
        and ("SUSPICIOUS", "talker.prefill.matmat_bf16") in requested_ids \
        and not legacy_no_vnni_findings
    report(cls, f)
    print("legacy-no-vnni expectations", "PASS" if not legacy_no_vnni_findings else "FAIL")
    print("SELFTEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


REG_RE = re.compile(r'static const char \*const g_qwen_reported_flags\[\] = \{(.*?)\n\};', re.S)
GROUP_RE = re.compile(r'/\*\s*(.*?)\s*\*/')
# flags a dispatch decision does not exist for: they are diagnostics, thresholds or
# conditioning, and the map is right to leave them alone.  Everything else it does not
# name is genuinely "unknown" and is printed as such.
NON_DISPATCH_GROUPS = ("diagnostics", "precision, voice and conditioning", "GPU backends",
                       "server, admission and request batching")


def registry_groups(src_path):
    m = REG_RE.search(open(src_path, errors="replace").read())
    if not m:
        return {}
    groups, cur = {}, "ungrouped"
    for line in m.group(1).splitlines():
        g = GROUP_RE.search(line)
        if g:
            cur = g.group(1).split("—")[0].strip()
        for name in re.findall(r'"(QWEN_[A-Z0-9_]+)"', line):
            groups.setdefault(cur, []).append(name)
    return groups


def coverage(doc, registry_path, out=sys.stdout):
    """How much of the flag registry the dispatch map can account for.  The map must be
    able to say "I don't know", not only PASS: a flag it never resolves is a path the
    report is blind to."""
    groups = registry_groups(registry_path)
    named = set()
    for f in doc.get("features", []):
        if f.get("env"):
            named.add(f["env"])
    for g in doc.get("gates", []):
        for k in ("off_env", "on_env", "minb_env"):
            if g.get(k):
                named.add(g[k])
    # a gate row also explains the shape thresholds it reads
    named |= {"QWEN_NO_AMX", "QWEN_APPLE_MMLA", "QWEN_AMX_MIN_B", "QWEN_AMX_MIN_ROWS",
              "QWEN_AMX_BF16_MIN_B", "QWEN_AMX_INT8_MIN_B", "QWEN_AMX_BF16_MIN_COLS",
              "QWEN_AMX_INT8_MIN_COLS", "QWEN_AMX_Q4_MIN_COLS", "QWEN_NO_KAI_I8", "QWEN_NO_KAI_BF16"}
    total = sum(len(v) for v in groups.values())
    covered_total = 0; unknown_total = 0
    print(f"FEATURE COVERAGE  registry={total} flags in {len(groups)} groups; "
          f"map rows: {len(doc.get('features', []))} features + {len(doc.get('gates', []))} gates", file=out)
    print(f"  {'group':<44}{'flags':>6}{'resolved':>9}{'unknown':>8}  unknown flags", file=out)
    for grp, names in groups.items():
        cov = [n for n in names if n in named]
        miss = [n for n in names if n not in named]
        nondisp = any(grp.startswith(p) for p in NON_DISPATCH_GROUPS)
        covered_total += len(cov)
        if not nondisp:
            unknown_total += len(miss)
        tag = "" if not nondisp else "  (not a dispatch decision)"
        shown = ", ".join(miss[:6]) + (f", +{len(miss)-6} more" if len(miss) > 6 else "")
        print(f"  {grp:<44}{len(names):>6}{len(cov):>9}{len(miss) if not nondisp else 0:>8}  {shown if not nondisp else tag}", file=out)
    print(f"  {'TOTAL':<44}{total:>6}{covered_total:>9}{unknown_total:>8}", file=out)
    print(f"  resolved by the map: {covered_total}  ·  dispatch-relevant flags the map cannot explain: {unknown_total}"
          f"  ·  outside its scope: {total - covered_total - unknown_total}", file=out)
    return unknown_total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dispatch_json", nargs="?")
    ap.add_argument("--expect", default=DEFAULT_EXPECT)
    ap.add_argument("--warn-only", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--coverage", action="store_true", help="print how much of the flag registry the map explains")
    ap.add_argument("--registry", default=os.path.join(ROOT, "qwen_tts_kernels.c"))
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.coverage:
        doc = json.load(open(a.dispatch_json))
        coverage(doc, a.registry)
        return 0
    if not a.dispatch_json:
        ap.error("dispatch.json required (QWEN_DISPATCH_JSON=path ./qwen_tts --dispatch-map)")
    doc = json.load(open(a.dispatch_json))
    expect = json.load(open(a.expect))
    cls, findings = evaluate(doc, expect)
    bad = report(cls, findings)
    return 0 if (a.warn_only or bad == 0) else 1


if __name__ == "__main__":
    sys.exit(main())
