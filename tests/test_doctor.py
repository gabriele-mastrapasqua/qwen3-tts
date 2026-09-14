#!/usr/bin/env python3
"""Offline checks for tools/doctor.py: no binary, no membw, no model.

The doctor is a predictor, so the tests pin (a) its calibration against the one host where
the answer is measured, (b) the topology rules on synthetic machines, (c) that what it
emits is a schema-valid profile every label of which is one the report explains.
"""
import json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import doctor as D          # noqa: E402
import perf_profile as PP   # noqa: E402

FAILURES = []


def check(name, cond, detail=""):
    print(f"{'ok  ' if cond else 'FAIL'} {name}" + (f"  — {detail}" if detail and not cond else ""))
    if not cond:
        FAILURES.append(name)


# ---- model bytes: the public shapes, in MB, as the engine's --matmat-tune declares them
mb17, mb06 = D.model_bytes("1.7b"), D.model_bytes("0.6b")
check("1.7B Talker int8 step ~1.41 GB", 1.38e9 < mb17["talker_step_bytes"] < 1.45e9, mb17)
check("0.6B Talker int8 step ~0.44 GB", 0.42e9 < mb06["talker_step_bytes"] < 0.46e9, mb06)
check("CP working set ~112 MB on both", abs(mb17["cp_ws_bytes"] - mb06["cp_ws_bytes"]) < 1 and 105e6 < mb17["cp_ws_bytes"] < 120e6, mb17)

# ---- calibration: the reference host (c4-standard-24, 2x6).  GEMV roof at 6T ~66 GB/s is what
# the measured Talker B2 step (23.7 ms) implies; the cache-resident rate ~81 GB/s is what the
# measured CP (16 ms/frame over 1.29 GB) implies.  Both are what `make roof-matvec` would report there.
p = D.predict("1.7b", K=6, B=2, q=4, gemv_gbs=66.0, llc_share_mb=21.67 * 6, amx=True, l3_gbs=81.0)
check("reference C4 (2x6, B2, q4) predicted rho within 0.78-0.92 (measured p50/p95 0.83/0.87)", 0.78 <= p["rho"] <= 0.92, p)
check("reference: Talker B2 term within 20 % of the measured 23.7 ms", abs(p["talker_ms"] - 23.7) / 23.7 < 0.2, p)
check("reference: CP term within 25 % of the measured 16 ms (cache-resident)", p["cp_fits_llc"] and abs(p["cp_ms"] - 17.6) / 17.6 < 0.25, p)
p8 = D.predict("1.7b", 6, 2, 8, 66.0, 130, True, l3_gbs=81.0)
p1 = D.predict("1.7b", 6, 2, 1, 66.0, 130, True, l3_gbs=81.0)
check("quantum floor: q1 slower than q4 slower than q8 (decoder glue amortization)", p1["rho"] > p["rho"] > p8["rho"], (p1["rho"], p["rho"], p8["rho"]))
check("B=1 costs less than B=2 per worker but serves half the streams", D.predict("1.7b", 6, 1, 4, 66.0, 130, True, l3_gbs=81.0)["rho"] < p["rho"])
check("non-AMX decoder is charged the [GUESS] factor", D.predict("1.7b", 6, 2, 4, 66.0, 130, False, l3_gbs=81.0)["dec_ms"] > p["dec_ms"])
check("no roof -> no prediction, not a fabricated one", D.predict("1.7b", 6, 2, 4, None, 130, True) is None)
check("CP that does not fit the LLC share is charged at the DRAM GEMV rate",
      D.predict("1.7b", 4, 1, 4, 58.0, 4 * 1.5, True, l3_gbs=99.0)["cp_ms"] > D.predict("1.7b", 4, 1, 4, 58.0, 4 * 32.5, True, l3_gbs=99.0)["cp_ms"])
# the 8-core Emerald Rapids box (2026-09-08): GEMV 58.1 GB/s @4T on mask 0-3, cache set 98.7 GB/s;
# @8T 98.1 / 156.9.  The old 8c AMX profile measured 2x4 C=2 at stream RTF 1.03 with the pre-Design-D decoder.
r8 = {4: {"mask": "0-3", "gemv_gbs": 58.12, "l3_gbs": 98.71, "source": "MEASURED"},
      8: {"mask": "0-7", "gemv_gbs": 98.08, "l3_gbs": 156.94, "source": "MEASURED"}}
g, l3, src = D.roof_at(r8, {"host": None, "worker": None, "source": {}}, 4)
check("roof_at returns the measured size verbatim", g == 58.12 and l3 == 98.71 and "MEASURED" in src, (g, l3, src))
g2, l32, src2 = D.roof_at(r8, {"host": {"sweep": [(1, 14), (2, 27), (4, 55), (8, 110)], "read_gbs": 110, "threads": 8, "mask": "0-7"}, "worker": None, "source": {"host": "MEASURED"}}, 2)
check("roof_at scales the nearest measured size by the membw sweep and labels it PREDICTED", 20 < g2 < 40 and "PREDICTED" in src2, (g2, src2))
g3, _, src3 = D.roof_at({}, {"host": {"sweep": [(4, 35.0)], "read_gbs": 35, "threads": 4, "mask": "0-3"}, "worker": None, "source": {"host": "MEASURED"}}, 4)
check("without the roof tool the membw read is scaled by the transferred factor and labelled weak", abs(g3 - 35.0 * D.CAL["gemv_over_membw_read"]) < 1e-6 and "weak" in src3, (g3, src3))

# ---- topology rules on synthetic machines
def ident(cores, llc_doms=1, per_core_mb=20.0, perf=None):
    doms = []
    per = cores // llc_doms
    for i in range(llc_doms):
        doms.append({"id": i, "cpus": f"{i*per}-{(i+1)*per-1}", "size": "32768K"})
    return {"online_cpus": cores, "physical_cores": cores, "online_mask": f"0-{cores-1}",
            "llc_domains": doms, "llc_per_core_mb": per_core_mb, "perf_cores": perf}


def bw(sweep):
    return {"host": {"mask": "x", "threads": max(t for t, _ in sweep), "read_gbs": max(v for _, v in sweep), "sweep": sweep},
            "worker": None, "source": {"host": "MEASURED"}}

t8 = D.choose_topology(ident(8, per_core_mb=32.5), bw([(1, 14), (2, 27), (4, 55), (8, 110)]), "x86_amx", roofs=r8)
check("8-core AMX box: 2x4 streams one request per worker (rho<=0.85 at B1), not two", (t8["W"], t8["K"], t8["B"], t8["C"]) == (2, 4, 1, 2), {k: v for k, v in t8.items() if k != "rows"})
check("8-core AMX box: the 1x8 single-worker rows predict a higher frontier, flagged as unmeasured beyond B2",
      any(r["W"] == 1 and r["B"] == 4 and r["rho"] <= 0.9 for r in t8["rows"]))
t8n = D.choose_topology(ident(8, per_core_mb=32.5), bw([(1, 14), (2, 27), (4, 55), (8, 110)]), "x86_amx",
                        roofs={4: {"mask": "0-3", "gemv_gbs": 30.0, "l3_gbs": 50.0, "source": "MEASURED"}, 8: {"mask": "0-7", "gemv_gbs": 98.08, "l3_gbs": 156.94, "source": "MEASURED"}})
check("when the rule-based shape cannot stream even B=1, the cost model overrides it and says so",
      t8n["W"] == 1 and t8n["override"] and "override" in t8n["rule"], {k: v for k, v in t8n.items() if k != "rows"})
t4 = D.choose_topology(ident(4), bw([(1, 14), (2, 27), (4, 55)]), "x86_avx512bf16")
check("4-core host -> one worker", (t4["W"], t4["K"]) == (1, 4), t4["rule"])
t16 = D.choose_topology(ident(16, llc_doms=2, per_core_mb=4.0), bw([(1, 15), (2, 30), (4, 60), (8, 90), (16, 165)]), "x86_avx512bf16")
check("16-core two-CCX host -> one worker per LLC domain (2x8)", (t16["W"], t16["K"]) == (2, 8) and "LLC" in t16["rule"], t16)
t12 = D.choose_topology(ident(12, per_core_mb=21.67), bw([(1, 7.5), (2, 15), (4, 30), (8, 55), (12, 79)]), "x86_amx",
                        roofs={6: {"mask": "0-5", "gemv_gbs": 66.0, "l3_gbs": 81.0, "source": "MEASURED"}})
check("12-core reference host -> 2x6 with cap 2 (the measured frontier)", (t12["W"], t12["K"], t12["B"]) == (2, 6, 2), {k: v for k, v in t12.items() if k != "rows"})
th = D.choose_topology(ident(8, perf=4), bw([(1, 20), (4, 55), (8, 60)]), "apple_m1")
check("hybrid P/E part -> one worker on the performance cores", (th["W"], th["K"]) == (1, 4), th["rule"])
check("every prediction row carries W, K, B, C, rho and a roof source",
      all(set(("W", "K", "B", "C", "rho", "bw_src")) <= set(r) for r in t8["rows"]) and t8["rows"])

# ---- bandwidth selection never divides a host roof
b = bw([(1, 14), (2, 27), (4, 55), (8, 110)])
v, src = D.bw_at(b, 4)
check("bw_at takes the host sweep point at the worker size, not peak/W", v == 55 and "sweep" in src, (v, src))
v, src = D.bw_at(b, 6)
check("bw_at without an exact point takes the nearest point BELOW (conservative)", v == 55 and "below" in src, (v, src))
b["worker"] = {"mask": "0-3", "threads": 4, "read_gbs": 48.0}
b["source"]["worker"] = "MEASURED"
v, src = D.bw_at(b, 4)
check("a measured worker-mask roof wins over the host sweep", v == 48.0 and "worker" in src, (v, src))

# ---- env sets: every ISA class resolves, aliases follow, values have no whitespace
for isa in ("x86_amx", "x86_avx512bf16", "x86_avx512vnni", "x86_avx2", "arm_i8mm_bf16", "apple_i8mm_bf16", "arm_dotprod", "apple_m1", None):
    rows = D.env_set(isa, set())
    keys = [r["key"] for r in rows]
    check(f"env set for {isa}: unique keys, common set present, no whitespace",
          len(keys) == len(set(keys)) and "OPENBLAS_THREAD_TIMEOUT" in keys and "QWEN_STREAM_DECODE_CHUNK" in keys
          and all(r["value"] is None or r["value"] == r["value"].strip() for r in rows), keys)
amx = {r["key"]: r for r in D.env_set("x86_amx", set())}
check("AMX set carries the Design-D streaming reference (SD_AMX_D, fused residual, engine pool, q4)",
      amx["QWEN_SD_AMX_D"]["value"] == "1" and amx["QWEN_SD_FUSED_RESIDUAL"]["value"] == "1"
      and amx["QWEN_SD_POOL"]["value"] == "engine" and amx["QWEN_STREAM_DECODE_CHUNK"]["value"] == "4")
check("AMX set pins the x86 spin, not the Arm one", amx["QWEN_POOL_SPIN"]["value"] == "4096")
arm = {r["key"]: r for r in D.env_set("arm_i8mm_bf16", set())}
check("Arm set pins the Arm spin and the KleidiAI n-chunk", arm["QWEN_POOL_SPIN"]["value"] == "65536" and arm["QWEN_KAI_NCHUNK"]["value"] == "384")
undecl = [r for r in D.env_set("x86_amx", {"QWEN_PREFIX_CACHE"}) if not r["declared"]]
check("a flag the binary does not declare is flagged NOT-DECLARED, never silently pinned",
      undecl and all(r["label"] == "NOT-DECLARED" for r in undecl) and "QWEN_SD_AMX_D" in [r["key"] for r in undecl])
rejected = {k for k, _ in D.DO_NOT_SET}
check("no rejected flag is in any recommended set",
      not any(r["key"] in rejected and r["value"] not in (None, "0") for isa in D.ISA_ENV for r in D.env_set(isa, set())))
check("every DO_NOT_SET flag is one the engine declares (registry drift guard)",
      rejected <= set(PP.engine_known_flags(os.path.join(ROOT, "qwen_tts")) or rejected))

# ---- argv is a server line
argv = D.build_argv({"W": 2, "K": 4, "B": 2}, port=9000)
check("argv: prefork server with cap, fail-fast queue, int8",
      argv[:1] == ["./qwen_tts"] and "--serve" in argv and "--prefork" in argv and argv[argv.index("--batch-size") + 1] == "2"
      and argv[argv.index("--max-queue") + 1] == "0" and "--int8" in argv, argv)
argv1 = D.build_argv({"W": 1, "K": 8, "B": 1})
check("argv: one worker uses -j, no prefork", "-j" in argv1 and "--prefork" not in argv1, argv1)

# ---- the draft is a schema-valid profile
idn = ident(8, per_core_mb=32.5)
idn.update({"arch": "x86_64", "cpu_model": "Test Xeon", "logical_cpus": 16, "smt": "off (off)", "numa_nodes": 1,
            "llc_mb": 260.0, "ram_gib": 31.3, "isa": ["avx512f", "amx_int8"], "online_mask": "0-7"})
topo = {"W": 2, "K": 4, "B": 2, "C": 4, "rule": "test"}
prof = D.draft_profile(idn, {"caps": {"simd": "amx"}}, topo, D.env_set("x86_amx", set()), "x86_amx", "doctor-test-2x4", b)
errs = D.validate_draft(prof)
check("draft profile validates against configs/perf/schema.json + semantic rules", errs == [], errs)
check("draft is unqualified and says why in every env entry",
      prof["qualification"]["status"] == "unqualified" and all(e["why"].startswith("[") for e in prof["runtime"]["environment"].values()))
check("draft carries the rejected flags as ABSENT with their verdict",
      prof["runtime"]["environment"]["QWEN_STREAM_LEAD_GATE"]["value"] is None)
check("draft argv builds through perf_profile.argv", "--prefork" in PP.argv(prof, "M", 8080) and "--batch-size" in PP.argv(prof, "M", 8080))
bad = json.loads(json.dumps(prof))
bad["server"]["threads_per_worker"] = 16
check("oversubscribed draft is refused by the same rules as a committed profile", D.validate_draft(bad) != [])

# ---- parsers on real output fragments
caps_txt = """qwen-tts compiled capabilities:
  build:            851fef6 · SIMD=amx · src=851fef6:clean · Sep  8 2026 09:38:22
  runtime cpu:      sse2 avx avx2 fma avx512f avx512bw avx512vnni avx512bf16 amx-int8
  x86 amx int8:     AMX ACTIVE (tile 16x64 int8 GEMM for batched matmat; QWEN_NO_AMX=1 disables)
  kernel selection (shape 2048x2048, asked to the dispatcher):
    B=1  (CLI, server c=1) matvec: bf16 -> NEON/scalar 2-row fused | int8 -> VNNI vpdpbusd | q4_0 -> VNNI vpdpbusd
    B=4                   matmat: bf16 -> bf16 AMX tiles | int8 -> int8 AMX tiles | q4_0 -> q4   AMX tiles
"""
c = D.parse_caps(caps_txt)
check("caps parser: build/simd/src, runtime cpu, AMX line, per-B kernel table",
      c.get("simd") == "amx" and c.get("src") == "851fef6:clean" and "amx-int8" in c.get("runtime_cpu", "")
      and c["kernel_by_B"][4]["int8"].startswith("int8 AMX") and c["kernel_by_B"][1]["int8"].startswith("VNNI"), c)
mm = D.parse_matmat_bench("""matmat-bench: B=8, threads=4
  [3072x1024]  (6.0 MB bf16)
     bf16   seq    4.16 ms   batch    0.27 ms   SPEEDUP 15.50x
     int8   seq    0.72 ms   batch    0.11 ms   SPEEDUP 6.35x
  [1024x3072]  (6.0 MB bf16)
     int4   seq    0.61 ms   batch    0.15 ms   SPEEDUP 4.03x
""")
check("matmat-bench parser", mm["3072x1024"]["int8"]["speedup"] == 6.35 and mm["1024x3072"]["int4"]["batch_ms"] == 0.15, mm)

# ---- the report renders every section with an empty bandwidth and no crash
rep = {"utc": "t", "elapsed_s": 0.1, "out": "x", "identity": idn, "bandwidth": {"host": None, "worker": None, "source": {}, "notes": ["n"]},
       "binary": {"bin": "b", "caps": c, "isa_class": "x86_amx", "features": {}, "gate_lines": []},
       "shapes": {"matmat_bench": {2: mm}, "threads": 4}, "topology": dict(topo, amx=True),
       "predictions": {"1.7b": [], "0.6b": []},
       "recommendation": {"argv": argv, "env": D.env_set("x86_amx", set()), "undeclared": [], "draft_path": "d", "draft_errors": [], "verify": ["v"], "alt": None}}
txt = D.render(rep)
check("report renders all eight sections and says UNKNOWN when there is no roof",
      all(f"{i}." in txt for i in range(1, 9)) and "[UNKNOWN] no GEMV roof" in txt)
g5 = D.classify_arm_gemv_scaling(159.9, 134.6, 143.4, 4.45)
check("Arm GEMV preflight classifies G5-like 4x8 contention as FAIL",
      g5["status"] == "FAIL" and "do not qualify 4x8" in g5["verdict"], g5)
good_arm = D.classify_arm_gemv_scaling(170.7, 286.0, 435.1, 1.57)
check("Arm GEMV preflight classifies G4-like 4x8 scaling as PASS",
      good_arm["status"] == "PASS" and "suitable" in good_arm["verdict"], good_arm)
check("report keeps the Arm GEMV preflight visible at the top",
      txt.splitlines()[4] == "ARM multi-worker GEMV scaling" and "VERDICT:" in txt, txt[:300])
# ---- ceilings: physics (bandwidth only) >= model (with decoder) >= floor (B=1), all from the same terms
c8 = D.ceiling("1.7b", 4, 8, 55.6, 4.0 * 8, False, l3_gbs=69.4)
check("c8a 4x8 (55.6 GB/s per CCX): physics (perfect batching, free decoder) far above the model ceiling B1..2; floor 4",
      c8["B_physics"] >= 4 and c8["B_model"] in (1, 2) and c8["C_floor"] == 4 and c8["C_physics"] >= c8["C_model"] >= c8["C_floor"], c8)
c16 = D.ceiling("1.7b", 2, 16, 87.5, 4.0 * 16, False, l3_gbs=109.0)
check("c8a 2x16 (87.5 GB/s): physics far above model -> the decoder is the wall on the wide shape",
      c16["C_physics"] >= 2 * c16["C_model"] and c16["dec_share"] > 0.4, c16)
c06 = D.ceiling("0.6b", 4, 8, 55.6, 4.0 * 8, False, l3_gbs=69.4)
check("0.6B has a higher ceiling than 1.7B on the same shape", c06["C_model"] > c8["C_model"] and c06["C_physics"] > c8["C_physics"], (c06, c8))
check("no roof -> no ceiling", D.ceiling("1.7b", 4, 8, None, 32, False) is None)
rep["ceiling"] = {"1.7b": [c8, c16], "0.6b": [c06]}
txt2 = D.render(rep)
rep_v = dict(rep, binary=dict(rep["binary"], isa_class="x86_avx512bf16"))
check("ceiling section lists the measured calibration points of the binary's ISA family",
      "calibration on this ISA family" in D.render(rep_v) and "FALSIFIED" in D.render(rep_v) and "1x32 B8 C8" in D.render(rep_v))
check("ceiling section renders one row per shape with the three numbers", txt2.count("1.7b   4x8") == 1 and txt2.count("1.7b   2x16") == 1 and txt2.count("0.6b   4x8") == 1 and "physics C (B)" in txt2 and "dec share" in txt2)
check("report explains its labels in a legend", "legend:" in txt and "[TRANSFERRED]" in txt)

# ---- the wave plan: the recommendation as DATA a runner executes in order, no shell chain
plan = D.wave_plan({"W": 4, "K": 8, "B": 1, "C": 4, "rule": "one worker per LLC domain"},
                   [], [{"W": 1, "K": 32, "B": 4, "C": 4, "rho": 0.37}], "profiles/doctor/x/profile-draft.json", isa="x86_amx")
labels = [r["label"] for r in plan["runs"]]
check("wave plan: both shapes x both models, then the A/B candidates on the recommended shape",
      [r["topo"] for r in plan["runs"][:4]] == ["4x8", "1x32", "4x8", "1x32"]
      and [r["model"] for r in plan["runs"][:4]] == ["qwen3-tts-1.7b"] * 2 + ["qwen3-tts-0.6b"] * 2
      and all(r["env"] for r in plan["runs"][4:]) and len(plan["runs"]) == 4 + len(D.CANDIDATES), labels)
check("wave plan: cold, warm, one past the prediction; cap = the predicted B of that shape",
      plan["runs"][0]["conc"] == [4, 4, 6] and plan["runs"][0]["cap"] == 1 and plan["runs"][1]["cap"] == 4, plan["runs"][0])
check("wave plan: labels unique", len(set(labels)) == len(labels), labels)
plan_vnni = D.wave_plan({"W": 4, "K": 8, "B": 1, "C": 4, "rule": "r"}, [], [], "d", isa="x86_avx512bf16")
check("wave plan: an A/B candidate for another ISA is not a run", len(plan_vnni["runs"]) == 2 and not any(r["env"] for r in plan_vnni["runs"]), plan_vnni["runs"])
import subprocess, tempfile
with tempfile.TemporaryDirectory() as td:
    pp = os.path.join(td, "wave-plan.json")
    with open(pp, "w") as f:
        json.dump(plan, f)
    dry = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "doctor_wave.py"), pp, "--dry-run"],
                         capture_output=True, text=True)
    check("doctor_wave --dry-run prints one serve_parallel_wave command per run and touches no server",
          dry.returncode == 0 and dry.stdout.count("serve_parallel_wave.py") == len(plan["runs"])
          and "--batch-cap 1" in dry.stdout and "--conc 4,4,6" in dry.stdout and "--profile profiles/doctor/x/profile-draft.json" in dry.stdout
          and "pgrep" not in dry.stdout, dry.stdout[-600:] + dry.stderr[-300:])
    dry2 = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "doctor_wave.py"), pp, "--dry-run", "--only", labels[1]],
                          capture_output=True, text=True)
    check("doctor_wave --only selects by label", dry2.stdout.count("serve_parallel_wave.py") == 1 and labels[1] in dry2.stdout, dry2.stdout[-300:])

print()
if FAILURES:
    print(f"FAILED: {len(FAILURES)}: {FAILURES}")
    sys.exit(1)
print("test_doctor: all checks passed")
