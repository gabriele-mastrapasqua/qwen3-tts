#!/usr/bin/env python3
"""census_report.py — what the engine ACTUALLY executed: call map, coverage, UNKNOWN/fallback.

    tools/census_report.py census-*.json [--dispatch dispatch.json] [--out summary.json]
                                          [--top N] [--warn-only]

Input: the JSON twins of the shape census (QWEN_SHAPE_CENSUS=1 QWEN_CENSUS_JSON=dir/census-%d.json,
one file per process; prefork workers each write their own).  Rows are keyed by the STABLE
PATH ID (qwen_tts_kernels.h QWEN_PATH_*), component, N, K, B, with the kernels
(QWEN_MMK_*) and leaves (QWEN_LEAF_*) that ran inside them.

Output:
  CALL MAP    per component, the executed paths: calls, calls/frame, B/N/K, GMAC, the
              instruction class that ran (VNNI, AVX512_BF16, AMX, SDOT, ..., BLAS, F32_FALLBACK,
              SCALAR) — from the leaf/kernel that was noted at the branch, not from a guess
  COVERAGE    per component: share of calls and GMAC by class; UNKNOWN = calls whose row
              carries neither a kernel nor a leaf (the report cannot say what ran)
  JOIN        selected (dispatch map) vs executed (census): SUSPICIOUS when a feature is
              resolved ON but its class never executed, or a fallback executed instead

Exit 1 on UNKNOWN > 0 or any SUSPICIOUS (0 with --warn-only).
"""
import argparse, glob, json, os, sys
from collections import defaultdict

OPTIMIZED = {"VNNI", "AVX512_BF16", "AMX", "SMMLA", "BFMMLA", "SDOT", "KLEIDI", "Q8REPACK", "AVX512F", "AVX2", "NEON"}
FALLBACK = {"F32_FALLBACK", "BF16_TWIN", "INT8_TWIN", "Q4_TWIN", "SCALAR"}

KERNEL_CLASS = {
    "bf16 BFMMLA (arm)": "BFMMLA", "bf16 AVX-512 dpbf16": "AVX512_BF16",
    "bf16 fixed-B twin": "BF16_TWIN", "bf16 generic twin": "BF16_TWIN",
    "int8 AMX tiles": "AMX", "int8 VNNI vpdpbusd": "VNNI", "int8 AVX2 maddubs": "AVX2",
    "int8 SMMLA (i8mm)": "SMMLA", "int8 SDOT loop over B": "SDOT", "int8 f32-accum twin": "INT8_TWIN",
    "q4   VNNI vpdpbusd": "VNNI", "q4   AVX2 maddubs": "AVX2", "q4   SMMLA (i8mm)": "SMMLA",
    "q4   B x matvec": "F32_FALLBACK", "q4   generic twin": "Q4_TWIN", "FORCED B x matvec": "F32_FALLBACK",
    "solo (B_eff==1)": None, "bf16 AMX tiles": "AMX", "q4   AMX tiles": "AMX", "q4   KleidiAI (arm)": "KLEIDI",
    "bf16 GEMV": None, "int8 GEMV": None, "q4   GEMV": None,        # entry markers: the leaf decides
    "int8 KleidiAI GEMM": "KLEIDI", "int8 KleidiAI GEMV": "KLEIDI", "bf16 KleidiAI GEMM": "KLEIDI",
    "bf16 KleidiAI GEMV": "KLEIDI", "q8_0 repack SMMLA": "Q8REPACK", "q8_0 repack GEMV": "Q8REPACK",
}
LEAF_CLASS = {"vnni": "VNNI", "dpbf16": "AVX512_BF16", "sdot": "SDOT", "avx512f": "AVX512F", "avx2": "AVX2",
              "neon": "NEON", "scalar": "SCALAR", "blas": "BLAS", "f32_fused": "F32_FALLBACK",
              "kleidi": "KLEIDI", "amx": "AMX", "delegated": "DELEGATED"}
KIND = {0: "call", 1: "slice", 2: "wrapper", 3: "transform"}


def row_classes(row):
    cls = set()
    for l in row.get("leaves", []):
        c = LEAF_CLASS.get(l)
        if c: cls.add(c)
    if not cls:
        for k in row.get("kernels", []):
            c = KERNEL_CLASS.get(k)
            if c: cls.add(c)
    return sorted(cls)


def load(files):
    procs, rows = [], {}
    frames = 0
    for f in files:
        d = json.load(open(f))
        procs.append({"file": f, "pid": d.get("pid"), "frames": d.get("frames", 0),
                      "threads": d.get("threads"), "dropped_ops": d.get("dropped_ops", 0)})
        frames += d.get("frames", 0)
        for r in d.get("rows", []):
            key = (r["comp"], r["path_id"], r["N"], r["K"], r["B"])
            a = rows.setdefault(key, {"comp": r["comp"], "path_id": r["path_id"], "path": r["path"],
                                       "kind": KIND.get(r.get("kind", 0), "call"), "N": r["N"], "K": r["K"],
                                       "B": r["B"], "calls": 0, "macs": 0, "kernels": set(), "leaves": set()})
            a["calls"] += r["calls"]; a["macs"] += r["macs"]
            a["kernels"] |= set(r.get("kernels", [])); a["leaves"] |= set(r.get("leaves", []))
    out = []
    for a in rows.values():
        a["kernels"] = sorted(a["kernels"]); a["leaves"] = sorted(a["leaves"])
        a["classes"] = row_classes(a)
        if a["classes"] == ["DELEGATED"]:
            # the entry only delegated to per-matrix calls whose rows carry the work
            a["kind"] = "wrapper"
        a["class"] = "+".join(a["classes"]) if a["classes"] else "UNKNOWN"
        if a["path"].startswith("decoder_"):
            a["B_note"] = "len<=2^k"
        out.append(a)
    return procs, out, frames


def bucket(cls_list):
    if not cls_list: return "unknown"
    if any(c in FALLBACK for c in cls_list): return "fallback"
    if any(c == "BLAS" for c in cls_list): return "blas"
    return "optimized"


def report(procs, rows, frames, dispatch, top, out=sys.stdout):
    comps = ["talker", "cp", "decoder", "other"]
    findings = []
    print(f"CENSUS  processes={len(procs)} frames={frames} rows={len(rows)} "
          f"dropped_ops={sum(p['dropped_ops'] for p in procs)}", file=out)
    if dispatch:
        print(f"        dispatch map: isa_class={dispatch.get('isa_class')} build={dispatch.get('build')} src={dispatch.get('source_fp', '?')}", file=out)
    dropped = sum(p["dropped_ops"] for p in procs)
    if dropped:
        findings.append(("SUSPICIOUS", "census", f"{dropped} ops dropped (table full): the map is incomplete"))

    # ---- call map ---------------------------------------------------------------
    print("\nCALL MAP  (kind=call rows; slice/wrapper/transform rows listed apart; decoder_* B = time length bucketed to the next power of two, GMAC exact)", file=out)
    hdr = f"  {'comp':<8}{'path':<28}{'B':>4}{'N':>7}{'K':>7}{'calls':>9}{'/frame':>8}{'GMAC':>10}  class (kernels | leaves)"
    print(hdr, file=out)
    for comp in comps:
        crow = [r for r in rows if r["comp"] == comp and r["kind"] == "call"]
        crow.sort(key=lambda r: -r["macs"])
        for r in crow[:top]:
            print(f"  {comp:<8}{r['path']:<28}{r['B']:>4}{r['N']:>7}{r['K']:>7}{r['calls']:>9}"
                  f"{(r['calls'] / frames if frames else 0):>8.2f}{r['macs'] / 1e9:>10.3f}  {r['class']}"
                  f"  ({'+'.join(r['kernels']) or '-'} | {'+'.join(r['leaves']) or '-'})", file=out)
        if len(crow) > top:
            print(f"  {comp:<8}... {len(crow) - top} more rows", file=out)
    other = [r for r in rows if r["kind"] != "call"]
    if other:
        print("  -- detail rows (not counted in coverage totals) --", file=out)
        for r in sorted(other, key=lambda r: (r["comp"], -r["macs"]))[:top]:
            print(f"  {r['comp']:<8}{r['path']:<28}{r['B']:>4}{r['N']:>7}{r['K']:>7}{r['calls']:>9}"
                  f"{'':>8}{r['macs'] / 1e9:>10.3f}  {r['kind']}: {r['class']}", file=out)

    # ---- coverage ---------------------------------------------------------------
    print("\nCOVERAGE  (kind=call rows; share of calls / GMAC by instruction class)", file=out)
    print(f"  {'comp':<8}{'calls':>9}{'GMAC':>10}  {'optimized':>10}{'blas':>8}{'fallback':>10}{'UNKNOWN':>9}   classes seen", file=out)
    cov = {}
    unknown_total = 0
    for comp in comps:
        crow = [r for r in rows if r["comp"] == comp and r["kind"] == "call"]
        if not crow: continue
        calls = sum(r["calls"] for r in crow); macs = sum(r["macs"] for r in crow)
        by = defaultdict(lambda: [0, 0])
        seen = set()
        for r in crow:
            b = bucket(r["classes"]); by[b][0] += r["calls"]; by[b][1] += r["macs"]; seen |= set(r["classes"])
        unk = by["unknown"][0]; unknown_total += unk
        pct = lambda b: (100.0 * by[b][1] / macs) if macs else 0.0
        cov[comp] = {"calls": calls, "gmac": macs / 1e9, "optimized_pct": pct("optimized"), "blas_pct": pct("blas"),
                     "fallback_pct": pct("fallback"), "unknown_calls": unk, "classes": sorted(seen)}
        print(f"  {comp:<8}{calls:>9}{macs / 1e9:>10.3f}  {pct('optimized'):>9.1f}%{pct('blas'):>7.1f}%"
              f"{pct('fallback'):>9.1f}%{unk:>9}   {', '.join(sorted(seen)) or '-'}", file=out)
        for r in crow:
            if bucket(r["classes"]) == "fallback":
                findings.append(("FALLBACK", f"{comp}.{r['path']}",
                                 f"B={r['B']} N={r['N']} K={r['K']} calls={r['calls']} class={r['class']}"))
            if bucket(r["classes"]) == "unknown":
                findings.append(("UNKNOWN", f"{comp}.{r['path']}",
                                 f"B={r['B']} N={r['N']} K={r['K']} calls={r['calls']}: no kernel and no leaf noted"))
    print(f"  UNKNOWN calls total: {unknown_total}", file=out)

    # ---- join with the dispatch map ------------------------------------------------
    if dispatch:
        feats = {f["id"]: f for f in dispatch.get("features", [])}
        gates = {g["id"]: g for g in dispatch.get("gates", [])}
        def executed(comp_set, pred):
            return sum(r["calls"] for r in rows if r["comp"] in comp_set and r["kind"] == "call" and pred(r))
        print("\nSELECTED vs EXECUTED", file=out)
        print(f"  {'feature':<32}{'resolved':>9}{'executed calls':>15}  verdict", file=out)
        checks = [
            ("talker.prefill.matmat_bf16", {"talker"},
             lambda r: r["B"] >= 2 and ("AVX512_BF16" in r["classes"] or "AMX" in r["classes"] or "BFMMLA" in r["classes"] or "KLEIDI" in r["classes"]) and r["path"].startswith(("matmat_bf16", "prefill_bf16", "matmat_bf16_native"))),
            ("talker.prefill.f32_blas_fallback", {"talker"}, lambda r: r["path"] == "prefill_f32_sgemm"),
            ("matvec.int8.vnni", {"talker", "cp"}, lambda r: r["B"] == 1 and "VNNI" in r["classes"] and "int8" in r["path"]),
            ("matvec.bf16.dpbf16", {"talker", "cp"}, lambda r: r["B"] == 1 and "AVX512_BF16" in r["classes"]),
            ("matvec.int8.sdot", {"talker", "cp"}, lambda r: r["B"] == 1 and "SDOT" in r["classes"] and "int8" in r["path"]),
            ("gate.int8.vnni", {"talker", "cp", "decoder"}, lambda r: r["B"] >= 2 and "VNNI" in r["classes"] and r["path"].startswith("matmat_int8")),
            ("gate.bf16.avx512", {"talker", "cp", "decoder"}, lambda r: r["B"] >= 2 and "AVX512_BF16" in r["classes"]),
            ("gate.int8.amx", {"talker", "cp", "decoder"}, lambda r: r["B"] >= 2 and "AMX" in r["classes"] and "int8" in r["path"]),
            ("gate.int8.smmla", {"talker", "cp", "decoder"}, lambda r: r["B"] >= 2 and "SMMLA" in r["classes"]),
            ("decoder.int8", {"decoder"}, lambda r: r["path"] == "decoder_conv_int8"),
        ]
        for fid, comps_, pred in checks:
            row = feats.get(fid) or gates.get(fid)
            if row is None: continue
            resolved = row.get("resolved") if "resolved" in row else ("ON" if row.get("on") else "OFF")
            compiled = (row.get("compiled") in ("yes", True))
            n = executed(comps_, pred)
            if not compiled:
                verdict = "not compiled"
            elif str(resolved).startswith("ON") and n == 0:
                # only suspicious when the workload could have used it: B>=2 paths need batching
                verdict = "SELECTED BUT NOT EXECUTED"
                findings.append(("SUSPICIOUS", fid, f"resolved {resolved}, executed calls = 0 in {sorted(comps_)}"))
            elif resolved == "OFF" and n > 0:
                verdict = "EXECUTED THOUGH OFF"
                findings.append(("SUSPICIOUS", fid, f"resolved OFF but {n} calls executed"))
            else:
                verdict = "ok"
            print(f"  {fid:<32}{str(resolved):>9}{n:>15}  {verdict}", file=out)
        # the bug of 2026-09-03, stated as a join rule
        mb = feats.get("talker.prefill.matmat_bf16")
        fb_calls = executed({"talker"}, lambda r: r["path"] == "prefill_f32_sgemm")
        if mb and str(mb.get("resolved", "")).startswith("ON") and fb_calls > 0:
            findings.append(("SUSPICIOUS", "talker.prefill.f32_blas_fallback",
                             f"bf16 prefill selected but {fb_calls} f32/SGEMM prefill calls executed"))

    # ---- verdict ------------------------------------------------------------------
    print("\nFINDINGS", file=out)
    order = {"SUSPICIOUS": 0, "UNKNOWN": 1, "FALLBACK": 2}
    findings.sort(key=lambda t: (order.get(t[0], 9), t[1]))
    if not findings:
        print("  none: every executed call is attributed, no fallback ran, selected == executed", file=out)
    for v, where, txt in findings:
        print(f"  [{v}] {where}: {txt}", file=out)
    bad = sum(1 for v, _, _ in findings if v in ("SUSPICIOUS", "UNKNOWN"))
    print(f"\nCENSUS GATE: {'PASS' if bad == 0 else 'FAIL'}  (UNKNOWN calls={unknown_total}, suspicious={sum(1 for v, _, _ in findings if v == 'SUSPICIOUS')}, "
          f"fallback rows={sum(1 for v, _, _ in findings if v == 'FALLBACK')})", file=out)
    return bad, {"processes": procs, "frames": frames, "coverage": cov, "unknown_calls": unknown_total,
                 "findings": [{"verdict": v, "where": w, "text": t} for v, w, t in findings],
                 "rows": [{k: r[k] for k in ("comp", "path_id", "path", "kind", "N", "K", "B", "calls", "macs", "kernels", "leaves", "class")} for r in rows]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", help="census-*.json (globs ok)")
    ap.add_argument("--dispatch", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--top", type=int, default=14)
    ap.add_argument("--warn-only", action="store_true")
    a = ap.parse_args()
    files = sorted({f for pat in a.files for f in glob.glob(pat)})
    if not files:
        print("no census files found — was the server run with QWEN_SHAPE_CENSUS=1 QWEN_CENSUS_JSON=dir/census-%d.json ?")
        return 1
    dispatch = json.load(open(a.dispatch)) if a.dispatch and os.path.isfile(a.dispatch) else None
    procs, rows, frames = load(files)
    bad, summary = report(procs, rows, frames, dispatch, a.top)
    if a.out:
        json.dump(summary, open(a.out, "w"), indent=1)
    return 0 if (a.warn_only or bad == 0) else 1


if __name__ == "__main__":
    sys.exit(main())
