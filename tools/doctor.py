#!/usr/bin/env python3
"""doctor.py — one command, under a minute, no model: what is this box, what will the
engine pick on it, and how should the streaming server probably be run here.

    make doctor                      # default: ~10-40 s, reuses cached roofs when present
    make doctor DOCTOR_ARGS="--full" # also --self-test and the quick --matmat-tune grid
    tools/doctor.py --bin ./qwen_tts --membw /tmp/qwen_membw --out profiles/doctor/x

It composes what already exists rather than re-implementing it:

    tools/box_info.sh        identity, caches, NUMA, SMT/governor/cgroup gates
    tests/membw.c            measured bandwidth per cpu mask, cached in tools/roofs.py's store
    tests/roof_matvec_int8.c the Talker's OWN int8 GEMV streaming a 1.7B frame of distinct
                             weights from DRAM (--layers 28) and a CP-sized set from cache
                             (--layers 2), pinned to the candidate worker mask: the two
                             numbers the per-frame cost model is built on, MEASURED here
    ./qwen_tts --caps        what the binary compiled and what the host supports
    ./qwen_tts --dispatch-map + tools/dispatch_gate.py   the RESOLVED kernel choices
    ./qwen_tts --matmat-bench                            batched-vs-GEMV shapes, no model
    configs/perf/schema.json + tools/perf_profile.py     the draft profile it emits

Then it applies a small per-frame cost model calibrated on the streaming reference host
(GCP c4-standard-24, 2x6, 1.7B INT8, Design-D AMX decoder, 2026-09-07/08 evidence) to
PREDICT the realtime ratio per topology and batch, and writes a recommendation.

EVERY number carries a label, and the labels are the point:

    [MEASURED]     read from this machine, now
    [CACHED]       read from this machine on an earlier run (same hardware fingerprint)
    [TRANSFERRED]  a constant measured on the reference host, applied here unchanged
    [PREDICTED]    computed from measured + transferred inputs; error ~10-25 %
    [UNKNOWN]      nothing here supports a number; the doctor says so instead

A prediction is a starting point for `make bench-topo` / a Tier-A wave, never a claim.
Nothing here replaces `make cpu-check` (the qualification preflight) or a measurement.
"""
import argparse, json, os, platform, re, shutil, subprocess, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
try:
    import topology as TP          # noqa: E402
    import roofs as RF             # noqa: E402
    import perf_profile as PP      # noqa: E402
except Exception as e:             # pragma: no cover - the tools tree is always beside us
    TP = RF = PP = None
    _IMPORT_ERR = str(e)

FRAME_MS = 80.0                    # one codec frame of audio (12.5 Hz)
BUDGET_S_DEFAULT = 60

# ---------------------------------------------------------------------------------------
# Model shapes (public config.json values; the engine's --matmat-tune declares the same)
# ---------------------------------------------------------------------------------------
MODELS = {
    "1.7b": {"talker": dict(hidden=2048, heads=16, kv=8, hd=128, inter=6144, vocab=3072, layers=28),
             "cp":     dict(hidden=1024, heads=16, kv=8, hd=128, inter=3072, vocab=2048, layers=5),
             "codebooks": 16},
    "0.6b": {"talker": dict(hidden=1024, heads=16, kv=8, hd=128, inter=3072, vocab=3072, layers=28),
             "cp":     dict(hidden=1024, heads=16, kv=8, hd=128, inter=3072, vocab=2048, layers=5),
             "codebooks": 16},
}

# ---------------------------------------------------------------------------------------
# Calibration constants — measured on the reference host, TRANSFERRED elsewhere.
# Sources: .work/f1-fused-quantum-20260908.md, .work/p4-fused-residual-20260907.md,
#          .work/post-p4-evidence-synthesis-20260908.md (per-iteration model),
#          .work/p1-cadence-truth-20260907.md (CT-4 Talker B2/B1).
# ---------------------------------------------------------------------------------------
CAL = {
    "ref_host": "GCP c4-standard-24, Xeon 8581C, 12 physical cores SMT off, 2x6, 1.7B INT8",
    "ref_threads": 6,
    "talker_b2_over_b1": 1.10,          # CT-4: B2/B1 step ratio ~1.10 (measured)
    "l3_over_dram": 1.4,                # cache-resident GEMV rate / DRAM GEMV rate: 98.7/58 (8c ER 4T), ~81/66 (c4 6T, from CP 16 ms); used only when --layers 2 was not measured
    "gemv_over_membw_read": 1.6,        # the Talker GEMV out-streams membw's read loop (58 vs 35 GB/s at 4T): used only when the roof tool is unavailable
    "dec_glue_ms": 25.0,                # fused residual Design-D: iteration = (25 + items*q*9.5)/q ms/frame
    "dec_per_item_frame_ms": 9.5,       #   (measured fit at 6T; glue term scales ~sqrt(K), item term ~1/K)
    "dec_non_amx_factor": 1.5,          # [GUESS] no VNNI/Arm calibration of the ragged decoder exists
    "rho_hard": 0.90,                   # PLAN hard stream gate (STREAM_RTF p95 <= 0.90)
    "rho_preferred": 0.85,              # headroom the envelope wants at the operating point
    "model_error": "±10-20 % on rho; C4 on the reference host (2x6, GEMV 66 GB/s, cache set 81 GB/s): predicted 0.83, measured p50/p95 0.83/0.87",
}

# Measured points the cost model is checked against, per ISA family (STREAM_RTF p95 of the
# wave vs the rho predict() gives for that W x K x B).  Rendered under 8. CEILING so a reader
# sees how far to trust the model on THIS family before renting the next box.
CAL_POINTS = {
    "x86_amx": [
        ("GCP c4-standard-24 2x6 B2 C4", 0.87, 0.83, "reference host; 2026-09-07"),
    ],
    "x86_avx512bf16": [
        ("AWS c8a.8xlarge Zen5 4x8 B2 C8", 0.87, 1.05, "model pessimistic ~15 %; TOTAL p95 0.95-1.00 (2026-09-08)"),
        ("AWS c8a.8xlarge Zen5 2x16 B4 C8", 0.84, 0.74, "model optimistic ~12 %; TOTAL p95 0.94"),
        ("AWS c8a.8xlarge Zen5 2x16 B5 C10", 1.05, 0.80, "over the line where the model still says OK"),
        ("AWS c8a.8xlarge Zen5 1x32 B8 C8", 1.40, 0.50, "FALSIFIED: one 32-thread pool collapses; model blind to it"),
        ("AWS c8a.8xlarge Zen5 0.6B 4x8 B3 C12", 0.72, 1.00, "model pessimistic ~30 % on the small model"),
        ("AWS c8a.8xlarge Zen5 0.6B 2x16 B8 C16", 1.22, 0.75, "wide pool collapses on 0.6B too"),
        # one CCX, one worker pinned 0-7, ONE fixed short text, STAGE trace (2026-09-09): the lane law
        ("c8a one CCX 1x8@0-7 1.7B B1/B2/B3/B4", 1.172, 1.05, "STREAM p95 .674/.861/.991/1.172 = 40 ms + 13.5 ms x B: decoder 9.7 ms per decoded slot-frame (72 %), Talker +1.6, CP +1.8 per slot"),
        ("c8a one CCX 1x8@0-7 0.6B B1/B2/B3/B4", 0.889, 0.80, "STREAM p95 .433/.592/.737/.889 = 23 ms + 12.3 ms x B: same decoder, Talker stream 17 ms smaller"),
        ("c8a one CCX 1x2@0-1 1.7B B1", 0.858, None, "Talker 26.4 + CP 20.8 ms on TWO threads = the 8-thread cost: the weight stream saturates the CCX at 2 threads"),
        ("c8a two 1x4 lanes on one CCX, 1.7B B2 each", 1.28, 0.97, "two weight streams on one CCX halve each other (Talker 27.8 -> 63.7 ms/step): a CCX holds ONE step-lane"),
    ],
}

# ---------------------------------------------------------------------------------------
# Per-ISA serving sets.  Every entry: value, label, why.  None value = must be ABSENT.
# label: MEASURED-REF (a win measured on a reference host of this ISA), DEFAULT-PIN (already
# the compiled default, pinned so a later default change is visible), PREDICTED (carried
# from another ISA or from the cost model; verify), CANDIDATE (measured once, not in the
# reference), OFF (experimental, rejected or pending — must stay off).
# ---------------------------------------------------------------------------------------
COMMON_ENV = [
    ("OPENBLAS_NUM_THREADS", None, "ABSENT", "the engine sizes OpenBLAS per worker and backs off entirely when this is set"),
    ("OPENBLAS_THREAD_TIMEOUT", "1", "MEASURED-REF", "idle OpenBLAS workers park instead of spinning against the engine pool"),
    ("QWEN_PREFIX_CACHE", "1", "DEFAULT-PIN", "request-independent prompt head computed once"),
    ("QWEN_PREFILL_MATMAT", "1", "DEFAULT-PIN", "native bf16 matmat prefill where a matrix unit exists; pinned so a fallback is visible"),
    ("QWEN_DECODER_BATCH", "1", "DEFAULT-PIN", "one decoder pass for all active slots; the server sets it itself, the pin records it"),
    ("QWEN_STREAM_DECODE_CHUNK", "4", "MEASURED-REF", "F1: q4 = best C4 balance (prebuffer p95 201 ms, no 250 ms stalls); q8 = throughput control, q1 rejected"),
    ("QWEN_SERVER_ASYNC_OUTPUT", "0", "OFF", "OUT-1/2 implemented, default-off pending longer qualification; C3/C4 wave showed no KPI change"),
    ("QWEN_TTS_STREAM_LAYOUT", "1", "MEASURED-REF", "SL-1: official known-text dual-track layout is the current serving-generation reference; ICL/clone and live incremental text remain out of scope"),
]
ISA_ENV = {
    "x86_amx": [
        ("QWEN_SD_INT8", "1", "DEFAULT-PIN", "int8 speech-decoder convolutions (default with AVX-512 VNNI)"),
        ("QWEN_SD_AMX_D", "1", "MEASURED-REF", "Design-D INT8 AMX decoder with persistent B packs; parity + zero-error gates on the reference host"),
        ("QWEN_SD_FUSED_RESIDUAL", "1", "MEASURED-REF", "fused residual epilogues: C4 prebuffer p95 389->266 ms, STREAM p95 0.83->0.79"),
        ("QWEN_SD_STREAM_STRIP", "1", "MEASURED-REF", "warm output-range strip on the streaming decoder (SQ-1)"),
        ("QWEN_SD_RAG_MIN_PANELS", "2", "MEASURED-REF", "ragged panels go to the engine pool from 2 panels (compiled default 8 is the conservative A/B control)"),
        ("QWEN_SD_POOL", "engine", "MEASURED-REF", "decoder tiles on the engine-owned pool; a private team oversubscribes the slice"),
        ("QWEN_BLAS_OWN", "1", "MEASURED-REF", "OpenBLAS serial/partitioned under the engine pool (c8a execution-budget fix); WAV bit-identical"),
        ("QWEN_SD_AMX_BF16", "0", "MEASURED-REF", "bf16 AMX decoder route off in the reference; Design-D INT8 is the qualified route"),
        ("QWEN_CP_PREFILL2", "1", "DEFAULT-PIN", "two-position CP first pass; default with VNNI, pinned to be visible"),
        ("QWEN_POOL_SPIN", "4096", "MEASURED-REF", "x86 optimum: 0 costs +13 % RTF at C=1, 65536 (the Arm value) is worse on the other side"),
    ],
    "x86_avx512bf16": [
        ("QWEN_SD_INT8", "1", "DEFAULT-PIN", "int8 speech-decoder convolutions (default with AVX-512 VNNI)"),
        ("QWEN_CP_PREFILL2", "1", "DEFAULT-PIN", "two-position CP first pass"),
        ("QWEN_POOL_SPIN", "4096", "MEASURED-REF", "x86 compiled default; measured optimum on Zen5/EPYC and c4 VNNI"),
        ("QWEN_VNNI_GEMV_MR", "2", "MEASURED-REF", "two-row VNNI GEMV microkernel (c8a Zen5 profile)"),
        ("QWEN_SD_POOL", "engine", "PREDICTED", "engine-owned decoder pool is common code; qualified on AMX only — verify on this ISA"),
        ("QWEN_BLAS_OWN", "1", "PREDICTED", "common code, qualified on AMX only — verify"),
    ],
    "x86_avx512vnni": "x86_avx512bf16",
    "x86_avx2": [
        ("QWEN_POOL_SPIN", "4096", "DEFAULT-PIN", "x86 compiled default"),
    ],
    "arm_i8mm_bf16": [
        ("QWEN_KAI_NCHUNK", "384", "MEASURED-REF", "KleidiAI GEMM n-tiling: prefill p50 45.0->42.9 ms on Axion; 192/96 worse"),
        ("QWEN_POOL_SPIN", "65536", "MEASURED-REF", "Arm optimum: 4096 cost 40 % of the Code Predictor on Axion (context switches)"),
        ("QWEN_SD_POOL", "engine", "PREDICTED", "common code; Arm+KleidiAI is the parity reference for the decoder — verify"),
    ],
    "apple_i8mm_bf16": "arm_i8mm_bf16",
    "arm_dotprod": [
        ("QWEN_POOL_SPIN", "65536", "PREDICTED", "Linux/aarch64 compiled default"),
    ],
    "apple_m1": [],       # GCD dispatch: no spin knob; the M1 is a compile-check host, not a serving reference
}
# flags that exist, are declared, and must NOT be in a serving profile today
DO_NOT_SET = [
    ("QWEN_STREAM_LEAD_GATE", "REJECTED 2026-09-07: parks 95.8 % of steps, no stall improvement"),
    ("QWEN_DECODER_THREAD", "REJECTED 2026-09-07: same-pool decoder consumer loses to inline ragged decode"),
    ("QWEN_PREFILL_HELPER", "REJECTED 2026-09-07: cloned-context helper is not a serving fix"),
    ("QWEN_ADMIT_UTIL", "FALSIFIED 2026-09-08: fifth request interactive, the established four stall (stall@250 50 %)"),
    ("QWEN_VNNI_PREPACK", "REJECTED 2026-09-03 on Zen5: +1.1 % Talker, +3.5 % CP, +1.4 GB"),
    ("QWEN_AMX_PREPACK", "opt-in, not part of the qualified reference"),
    ("QWEN_STREAM_DECODE_CHUNK_BUSY", "keep 0/absent: a busy-chunk override was never part of a passing envelope"),
]
CANDIDATES = [  # (key, value, why, isa_class it applies to; None = every ISA)
    ("QWEN_AMX_MIN_B", "2", "measured +10 % RTF at C=4 on the old 8c AMX profile (pre Design-D); NOT re-validated on the streaming reference — A/B it", "x86_amx"),
]

# ---------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------
def sh(cmd, env=None, timeout=120, cwd=ROOT):
    e = dict(os.environ)
    if env:
        e.update({k: str(v) for k, v in env.items()})
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=cwd, env=e)
        return r.returncode, r.stdout, r.stderr
    except subprocess.TimeoutExpired:
        return 124, "", f"timeout after {timeout}s"
    except FileNotFoundError as ex:
        return 127, "", str(ex)


def read(p, default=""):
    try:
        with open(p) as f:
            return f.read().strip()
    except OSError:
        return default


def mask_len(spec):
    s = TP.mask_parse(spec) if TP else None
    return len(s) if s else 0


def online_mask(hw):
    """The CPUs the doctor may actually run on: cpus_allowed ∩ /sys online.  On a cloud
    x86 box with SMT disabled the sibling ids stay in the affinity mask but are offline,
    and a topology that names them measures nothing."""
    allowed = TP.host_domain(hw)["cpus_allowed"] if TP else "?"
    online = read("/sys/devices/system/cpu/online", "")
    a, o = (TP.mask_parse(allowed) if TP else None), (TP.mask_parse(online) if TP else None)
    if a and o:
        return TP.mask_str(a & o), (a != (a & o))
    if o:
        return TP.mask_str(o), False
    if a:
        return TP.mask_str(a), False
    n = (hw.get("cpu") or {}).get("cpus_logical") or (hw.get("cpu") or {}).get("cores_physical") or 0
    return (f"0-{n - 1}" if n > 1 else ("0" if n == 1 else "?")), False


def first_n(mask, n):
    cpus = sorted(TP.mask_parse(mask) or [])
    return TP.mask_str(frozenset(cpus[:n])) if cpus else "?"


def relp(p):
    r = os.path.relpath(p, ROOT)
    return r if not r.startswith("..") else os.path.abspath(p)


def fmt_ms(x):
    return "   n/a" if x is None else f"{x:6.1f}"


# ---------------------------------------------------------------------------------------
# 1. identity
# ---------------------------------------------------------------------------------------
def step_identity(out):
    p = os.path.join(out, "hardware.json")
    rc, so, se = sh(["bash", os.path.join(ROOT, "tools", "box_info.sh"), "--out", p], timeout=90)
    with open(os.path.join(out, "hardware.txt"), "w") as f:
        f.write(so + se)
    try:
        hw = json.load(open(p))
    except Exception:
        return None, f"box_info failed rc={rc}: {se.strip()[:200]}"
    return hw, None


def perf_cores(hw):
    """Hybrid parts (Apple P/E): the serving pool belongs on the performance cores only."""
    for line in (hw.get("cpu") or {}).get("perflevels") or []:
        m = re.match(r"Performance:\s*(\d+) physical cores", line)
        if m:
            return int(m.group(1))
    return None


def identity_summary(hw):
    cpu, cache, mem = hw.get("cpu", {}), hw.get("cache", {}), hw.get("memory", {})
    mask, offline_in_mask = online_mask(hw)
    doms = TP.llc_domains(hw) if TP else []
    return {
        "perf_cores": perf_cores(hw),
        "host": hw.get("host", {}).get("hostname"),
        "os": hw.get("host", {}).get("os"), "arch": hw.get("host", {}).get("arch"),
        "cloud": (hw.get("gcp") or {}).get("machine_type") or (hw.get("aws") or {}).get("instance_type"),
        "cpu_model": cpu.get("model"), "vendor": cpu.get("vendor"),
        "physical_cores": cpu.get("cores_physical"), "logical_cpus": cpu.get("cpus_logical"),
        "smt": cpu.get("smt"), "sockets": cpu.get("sockets"),
        "online_mask": mask, "online_cpus": mask_len(mask), "offline_siblings_in_mask": offline_in_mask,
        "llc_mb": cache.get("llc_mb"), "llc_per_core_mb": cache.get("llc_per_core_mb"),
        "llc_what": cache.get("llc_what"), "llc_domains": doms, "l2_mb": cache.get("l2_mb"),
        "numa_nodes": (hw.get("numa") or {}).get("nodes"),
        "ram_gib": round((mem.get("total_mb") or 0) / 1024.0, 1),
        "swap_mb": mem.get("swap_mb"), "thp": mem.get("thp"),
        "isa": (hw.get("flags") or {}).get("have", []),
        "gates": hw.get("gates", {}), "limits": hw.get("limits", {}),
        "warnings": hw.get("warnings", []),
    }


# ---------------------------------------------------------------------------------------
# 2. bandwidth — reuse the roofs store when the hardware fingerprint matches
# ---------------------------------------------------------------------------------------
def step_bandwidth(hw, ident, membw, store_dir, worker_mask, reps, measure=True):
    res = {"host": None, "worker": None, "source": {}, "notes": []}
    if not RF:
        res["notes"].append("roofs.py unavailable")
        return res
    fp = RF.hw_fingerprint(hw)
    store = RF.load_store(store_dir, fp)
    res["hw_fingerprint"] = fp
    host_mask = ident["online_mask"]
    nthr = ident["online_cpus"] or 1
    sweep = sorted({1, 2, 4, 8, 16, 32, nthr} & set(range(1, nthr + 1)))

    def cached(mask, bench="read"):
        for e in store.get("entries", []):
            if e.get("bench") == bench and TP.mask_norm(e.get("cpu_mask", "?")) == TP.mask_norm(mask):
                return e
        return None

    def measure_mask(mask, threads, scope):
        if not (membw and os.path.isfile(membw) and os.access(membw, os.X_OK)):
            res["notes"].append(f"membw binary missing ({membw}); build it with `make membw`")
            return None
        try:
            # macOS has no cpu affinity: measure unpinned ("all") and keep the mask label
            run_mask = "all" if platform.system() == "Darwin" else mask
            doc = RF.run_membw(membw, run_mask, ",".join(str(t) for t in threads), reps,
                               ident.get("llc_mb") or None, label=scope)
            if run_mask == "all":
                doc["cpu_mask"] = mask
        except Exception as ex:
            res["notes"].append(f"membw failed on mask {mask}: {str(ex)[:120]}")
            return None
        RF.upsert(store, RF.entries_from_membw(doc, scope, membw, None))
        RF.save_store(store_dir, store)
        return cached(mask)

    for key, mask, threads, scope in (("host", host_mask, sweep, "HOST"),
                                      ("worker", worker_mask, [mask_len(worker_mask)], f"WORKER[{worker_mask}]")):
        if not mask or mask == "?" or (key == "worker" and TP.mask_norm(mask) == TP.mask_norm(host_mask)):
            continue
        if key == "worker" and platform.system() == "Darwin":
            res["notes"].append("no cpu affinity on macOS: worker-mask roof not measurable, host sweep used")
            continue
        e = cached(mask)
        src = "CACHED"
        if (e is None or RF.stale(store, membw) if membw and os.path.isfile(membw) else e is None) and measure:
            e = measure_mask(mask, threads, scope)
            src = "MEASURED"
        if e is None:
            res["source"][key] = "UNKNOWN"
            continue
        allb = {b: cached(mask, b) for b in ("read", "copy", "triad")}
        res[key] = {
            "mask": mask, "threads": e.get("threads"),
            "read_gbs": e.get("gbs"),
            "copy_gbs": (allb["copy"] or {}).get("gbs"), "triad_gbs": (allb["triad"] or {}).get("gbs"),
            "sweep": [(s["threads"], s["gbs"]) for s in e.get("sweep", []) if s.get("gbs")],
            "t90": e.get("t90"), "provenance_utc": (e.get("provenance") or {}).get("utc"),
        }
        res["source"][key] = src
    return res


def bw_at(bwres, threads, prefer_worker=True):
    """read bandwidth [GB/s] available to ONE worker of `threads` threads.
    Order: the measured worker-mask roof for that size; else the host sweep point at that
    thread count (a roof of a SUBSET of the host measured over the host mask — labelled);
    never a division of the host peak."""
    w = bwres.get("worker")
    if prefer_worker and w and w.get("threads") == threads and w.get("read_gbs"):
        return w["read_gbs"], "worker-mask roof [%s]" % bwres["source"].get("worker", "?")
    h = bwres.get("host")
    if h and h.get("sweep"):
        pts = dict(h["sweep"])
        if threads in pts:
            return pts[threads], "host sweep @%dT [%s]" % (threads, bwres["source"].get("host", "?"))
        below = [t for t in pts if t < threads]
        if below:
            t = max(below)
            return pts[t], "host sweep @%dT (nearest below, conservative) [%s]" % (t, bwres["source"].get("host", "?"))
    return None, "UNKNOWN"


# ---------------------------------------------------------------------------------------
# 2b. the engine's own GEMV roof, per candidate worker size
# ---------------------------------------------------------------------------------------
def run_roof(roof_bin, mask, threads, layers, reps, out, tag):
    if not (roof_bin and os.path.isfile(roof_bin) and os.access(roof_bin, os.X_OK)):
        return None, f"roof tool missing ({roof_bin}); build it with `make roof-matvec`"
    jp = os.path.join(out, f"roof_{tag}.json")
    cmd = [roof_bin, "--threads", str(threads), "--layers", str(layers), "--reps", str(reps), "--json", jp]
    pinned = False
    if mask and mask != "?" and shutil.which("taskset"):
        cmd = ["taskset", "-c", mask] + cmd
        pinned = True
    rc, so, se = sh(cmd, timeout=120)
    with open(os.path.join(out, f"roof_{tag}.txt"), "w") as f:
        f.write(so + se)
    try:
        d = json.load(open(jp))
    except Exception:
        return None, f"roof tool failed rc={rc}: {(se or so).strip()[-160:]}"
    fr = d.get("frame", {})
    return {"threads": threads, "layers": layers, "mask": mask if pinned else "unpinned",
            "frame_ms": fr.get("ms"), "gbs": fr.get("gbs"), "frame_bytes": d.get("frame_bytes"),
            "read_only_gbs": d.get("read_only_roof_gbs"), "reps": reps}, None


def step_gemv_roofs(roof_bin, ident, sizes, out, reps=3):
    """sizes: {K: mask}.  For each worker size: DRAM frame (28 layers) + cache-resident set
    (2 layers ~ 96 MB, the CP working set).  Returns {K: {...}} plus notes."""
    res, notes = {}, []
    for K, mask in sorted(sizes.items()):
        dram, err = run_roof(roof_bin, mask, K, 28, reps, out, f"dram_{K}t")
        if err:
            notes.append(err)
            break
        l3, err2 = run_roof(roof_bin, mask, K, 2, reps, out, f"l3_{K}t")
        if err2:
            notes.append(err2)
        res[K] = {"mask": dram["mask"], "gemv_gbs": dram["gbs"], "talker17b_frame_ms": dram["frame_ms"],
                  "read_only_gbs": dram["read_only_gbs"],
                  "l3_gbs": (l3 or {}).get("gbs"), "l3_set_mb": ((l3 or {}).get("frame_bytes") or 0) / 1e6,
                  "source": "MEASURED"}
    return res, notes


ARM_GEMV_THREADS = 8
ARM_GEMV_GROUPS = 4


def _arm_gemv_write_json(out, result):
    path = os.path.join(out, "arm_gemv_scaling.json")
    result["artifact"] = relp(path)
    with open(path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    return result


def _arm_gemv_skip(out, reason):
    return _arm_gemv_write_json(out, {
        "schema_version": 1,
        "benchmark": "roof_matvec_int8",
        "status": "SKIPPED",
        "verdict": "SKIPPED / ARM multi-worker GEMV preflight not run",
        "reason": reason,
        "rows": [],
    })


def classify_arm_gemv_scaling(isolated_gbs, aggregate_2x8, aggregate_4x8,
                              per_worker_slowdown):
    """Classify the tested 4x8 shape without implying that the whole host is unusable."""
    if not all(x is not None and x > 0 for x in
               (isolated_gbs, aggregate_2x8, aggregate_4x8, per_worker_slowdown)):
        return {
            "status": "UNKNOWN",
            "verdict": "UNKNOWN / insufficient GEMV scaling measurements",
            "message": "the 1x8, 2x8 and 4x8 rows were not all measurable",
        }
    scale = aggregate_4x8 / isolated_gbs
    if aggregate_4x8 <= isolated_gbs:
        return {
            "status": "FAIL",
            "verdict": "FAIL / severe cross-worker contention; do not qualify 4x8 serving on this box",
            "message": "4x8 aggregate throughput is no higher than isolated 1x8",
        }
    if scale < 1.4 or per_worker_slowdown > 2.5:
        return {
            "status": "STRONG WARNING",
            "verdict": "STRONG WARNING / 4x8 serving topology not recommended",
            "message": "4x8 has severe cross-worker GEMV contention",
        }
    if scale < 2.0 or per_worker_slowdown > 2.0:
        return {
            "status": "WARNING",
            "verdict": "WARNING / topology-sensitive; validate alternate worker shapes",
            "message": "4x8 scaling is topology-sensitive",
        }
    return {
        "status": "PASS",
        "verdict": "PASS / topology appears suitable for 4x8 serving",
        "message": "4x8 aggregate scaling and per-worker slowdown are within the preflight bands",
    }


def _arm_gemv_groups(ident):
    cpus = sorted(TP.mask_parse(ident.get("online_mask") or "") or [])
    if len(cpus) < ARM_GEMV_THREADS * ARM_GEMV_GROUPS:
        return []
    return [TP.mask_str(frozenset(cpus[i * ARM_GEMV_THREADS:(i + 1) * ARM_GEMV_THREADS]))
            for i in range(ARM_GEMV_GROUPS)]


def _arm_gemv_spawn(roof_bin, mask, out, tag, reps):
    json_path = os.path.join(out, f"arm_gemv_{tag}.json")
    text_path = os.path.join(out, f"arm_gemv_{tag}.txt")
    cmd = [roof_bin, "--threads", str(ARM_GEMV_THREADS), "--layers", "28",
           "--reps", str(reps), "--json", json_path]
    taskset = shutil.which("taskset")
    if not taskset:
        return None, None, None, "taskset is unavailable; fixed CPU-mask preflight cannot run"
    cmd = [taskset, "-c", mask] + cmd
    try:
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True)
    except OSError as ex:
        return None, None, None, f"could not start {' '.join(cmd)}: {ex}"
    return proc, json_path, text_path, None


def _arm_gemv_collect(proc, json_path, text_path, mask):
    try:
        so, se = proc.communicate(timeout=180)
    except subprocess.TimeoutExpired:
        proc.kill()
        so, se = proc.communicate()
        se = (se or "") + "\ntimeout after 180s"
        rc = 124
    else:
        rc = proc.returncode
    with open(text_path, "w") as f:
        f.write(so + se)
    row = {"mask": mask, "returncode": rc, "stdout": so, "stderr": se,
           "text_artifact": relp(text_path), "json_artifact": relp(json_path)}
    try:
        with open(json_path) as f:
            doc = json.load(f)
        frame = doc.get("frame") or {}
        row["frame_ms"] = float(frame["ms"])
        row["gbs"] = float(frame["gbs"])
        row["frame_bytes"] = doc.get("frame_bytes")
        row["read_only_roof_gbs"] = doc.get("read_only_roof_gbs")
    except (OSError, ValueError, KeyError, TypeError) as ex:
        row["error"] = f"missing/invalid roof JSON: {ex}"
    return row


def _arm_gemv_run_shape(roof_bin, masks, out, shape, reps):
    procs = []
    for i, mask in enumerate(masks):
        proc, jp, tp, err = _arm_gemv_spawn(roof_bin, mask, out, f"{shape}_w{i}", reps)
        if err:
            for old, _, _, _ in procs:
                old.kill()
                old.wait()
            return {"shape": shape, "masks": masks, "workers": len(masks), "error": err, "runs": []}
        procs.append((proc, jp, tp, mask))
    runs = [_arm_gemv_collect(*item) for item in procs]
    good = [r for r in runs if r.get("frame_ms") and r.get("gbs")]
    row = {"shape": shape, "masks": masks, "workers": len(masks), "runs": runs}
    if len(good) != len(runs) or not good:
        row["error"] = "one or more roof workers did not produce a valid frame result"
        return row
    ms = [r["frame_ms"] for r in good]
    gbs = [r["gbs"] for r in good]
    row["worker_ms"] = {"min": min(ms), "max": max(ms), "mean": sum(ms) / len(ms)}
    row["per_worker_gbs"] = {"min": min(gbs), "max": max(gbs), "mean": sum(gbs) / len(gbs)}
    row["aggregate_gbs"] = sum(gbs)
    return row


def step_arm_gemv_preflight(roof_bin, ident, out, reps=3, enabled=True):
    """Short, fixed-mask simultaneous GEMV discriminator for Arm serving shapes.

    This is deliberately separate from the existing one-worker roof used by the cost
    model: 4x8 must be measured concurrently or shared-cache/fabric contention is hidden.
    """
    if not enabled:
        return _arm_gemv_skip(out, "disabled by command-line option")
    arch = str(ident.get("arch") or "").lower()
    if not (arch.startswith("aarch64") or arch.startswith("arm64") or arch.startswith("arm")):
        return _arm_gemv_skip(out, f"non-Arm architecture: {ident.get('arch') or 'unknown'}")
    if not (roof_bin and os.path.isfile(roof_bin) and os.access(roof_bin, os.X_OK)):
        return _arm_gemv_skip(out, f"roof tool missing or not executable ({roof_bin})")
    if not shutil.which("taskset"):
        return _arm_gemv_skip(out, "taskset unavailable; fixed disjoint CPU masks are required")
    groups = _arm_gemv_groups(ident)
    if len(groups) != ARM_GEMV_GROUPS:
        return _arm_gemv_skip(out, "fewer than 32 online CPUs; 4x8 fixed-mask preflight is not applicable")
    ram = ident.get("ram_gib")
    if ram and ram < 20:
        return _arm_gemv_skip(out, f"only {ram:.1f} GiB RAM; four concurrent roof workers need a larger safety margin")

    reps = max(1, int(reps))
    rows = [
        _arm_gemv_run_shape(roof_bin, groups[:1], out, "1x8", reps),
        _arm_gemv_run_shape(roof_bin, groups[:2], out, "2x8", reps),
        _arm_gemv_run_shape(roof_bin, groups[:4], out, "4x8", reps),
    ]
    isolated = rows[0].get("per_worker_gbs", {}).get("mean")
    for row in rows:
        row["scale_vs_1x8"] = (row.get("aggregate_gbs") / isolated
                                if isolated and row.get("aggregate_gbs") else None)
    row2 = rows[1] if len(rows) > 1 else {}
    row4 = rows[2] if len(rows) > 2 else {}
    agg2 = row2.get("aggregate_gbs")
    agg4 = row4.get("aggregate_gbs")
    ms4 = (row4.get("worker_ms") or {}).get("mean")
    slowdown = (ms4 / rows[0]["worker_ms"]["mean"]
                if ms4 and rows[0].get("worker_ms", {}).get("mean") else None)
    verdict = classify_arm_gemv_scaling(isolated, agg2, agg4, slowdown)
    result = {
        "schema_version": 1,
        "benchmark": "roof_matvec_int8",
        "status": verdict["status"],
        "verdict": verdict["verdict"],
        "message": verdict["message"],
        "threads_per_worker": ARM_GEMV_THREADS,
        "layers": 28,
        "reps": reps,
        "masks": groups,
        "rows": rows,
        "isolated_gbs": isolated,
        "two_x8_aggregate_gbs": agg2,
        "four_x8_aggregate_gbs": agg4,
        "four_x8_scale": (agg4 / isolated) if isolated and agg4 else None,
        "four_x8_per_worker_slowdown": slowdown,
        "next_topologies": ["2x16", "1x32"] if verdict["status"] not in ("PASS", "SKIPPED") else [],
        "raw_values": {
            "isolated_gbs": isolated,
            "two_x8_aggregate_gbs": agg2,
            "four_x8_aggregate_gbs": agg4,
            "four_x8_scale": (agg4 / isolated) if isolated and agg4 else None,
            "four_x8_per_worker_slowdown": slowdown,
        },
    }
    if verdict["status"] == "FAIL" and slowdown and slowdown >= 2.5:
        result["g5_like_warning"] = True
        result["g5_warning"] = (
            "WARNING: severe cross-worker GEMV contention detected.\n"
            "4x8 aggregate throughput does not scale from isolated 1x8 and\n"
            f"per-worker latency increases ~{slowdown:.1f}x.\n"
            "The default 4x8 Arm v2 serving topology is NOT recommended on\n"
            "this machine. Run topology sweep (2x16 / 1x32) before any soak\n"
            "or production qualification."
        )
    elif verdict["status"] == "STRONG WARNING":
        result["g5_like_warning"] = False
    return _arm_gemv_write_json(out, result)


def _arm_gemv_span(values):
    if not values:
        return "n/a"
    lo, hi = values.get("min"), values.get("max")
    if lo is None or hi is None:
        return "n/a"
    return f"{lo:.1f}" if abs(hi - lo) < 0.05 else f"{lo:.1f}-{hi:.1f}"


def render_arm_gemv_scaling(arm):
    lines = ["ARM multi-worker GEMV scaling",
             "-" * 63,
             "shape   worker ms        per-worker GB/s   aggregate GB/s   scale"]
    rows = arm.get("rows") or []
    for row in rows:
        if row.get("error"):
            lines.append(f"{row.get('shape', '?'):<7} ERROR: {row['error']}")
            continue
        lines.append(f"{row.get('shape', '?'):<7} "
                     f"{_arm_gemv_span(row.get('worker_ms')):<16} "
                     f"{_arm_gemv_span(row.get('per_worker_gbs')):<18} "
                     f"{row.get('aggregate_gbs', 0):<16.1f} "
                     f"{(row.get('scale_vs_1x8') or 0):.2f}x")
    lines.append("-" * 63)
    if arm.get("four_x8_per_worker_slowdown") is not None:
        lines.append(f"4x8 per-worker slowdown vs isolated: {arm['four_x8_per_worker_slowdown']:.2f}x")
        lines.append(f"4x8 aggregate scaling vs isolated:   {arm.get('four_x8_scale', 0):.2f}x")
    else:
        lines.append("4x8 per-worker slowdown vs isolated: n/a")
        lines.append("4x8 aggregate scaling vs isolated:   n/a")
    lines.append(f"VERDICT: {arm.get('verdict', 'UNKNOWN')}")
    if arm.get("g5_warning"):
        lines.extend(arm["g5_warning"].splitlines())
    elif arm.get("status") in ("STRONG WARNING", "WARNING"):
        lines.append("Next topology tests: 2x16, then 1x32; do not use 4x8 as the serving baseline yet.")
    elif arm.get("status") == "SKIPPED":
        lines.append(f"SKIPPED: {arm.get('reason', 'not applicable')}")
    return lines


def roof_at(roofs, bwres, K):
    """(gemv_gbs, l3_gbs, source) for a worker of K threads.  Measured for that K when the
    roof tool ran there; otherwise the nearest measured size scaled by the membw read
    sweep ratio [PREDICTED]; otherwise membw read x the transferred factor; otherwise None."""
    if K in roofs and roofs[K].get("gemv_gbs"):
        r = roofs[K]
        return r["gemv_gbs"], r.get("l3_gbs") or r["gemv_gbs"] * CAL["l3_over_dram"], f"GEMV roof @{K}T mask {r['mask']} [MEASURED]"
    if roofs:
        Km = min(roofs, key=lambda k: abs(k - K))
        r = roofs[Km]
        bwK, _ = bw_at(bwres, K, prefer_worker=False)
        bwKm, _ = bw_at(bwres, Km, prefer_worker=False)
        scale = (bwK / bwKm) if (bwK and bwKm) else (K / float(Km)) ** 0.5
        g = r["gemv_gbs"] * scale
        l3 = (r.get("l3_gbs") or r["gemv_gbs"] * CAL["l3_over_dram"]) * (K / float(Km))
        return g, l3, f"GEMV roof @{Km}T scaled to {K}T by membw sweep [PREDICTED]"
    bw, src = bw_at(bwres, K)
    if bw:
        g = bw * CAL["gemv_over_membw_read"]
        return g, g * CAL["l3_over_dram"], f"membw read x{CAL['gemv_over_membw_read']} ({src}) [PREDICTED, weak]"
    return None, None, "UNKNOWN"


# ---------------------------------------------------------------------------------------
# 3. binary: caps + resolved dispatch + gate
# ---------------------------------------------------------------------------------------
def parse_caps(txt):
    d = {"kernel_by_B": {}}
    for line in txt.splitlines():
        s = line.strip()
        m = re.match(r"build:\s*(\S+)\s*·\s*SIMD=(\S+)\s*·\s*src=(\S+)", s)
        if m:
            d["build"], d["simd"], d["src"] = m.group(1), m.group(2), m.group(3)
        if s.startswith("runtime cpu:"):
            d["runtime_cpu"] = s.split(":", 1)[1].strip()
        if s.startswith("matvec threads:"):
            d["threads"] = s.split(":", 1)[1].strip()
        for key in ("x86 amx int8", "x86 amx bf16", "int8 dot", "bf16 dot", "kleidi", "BLAS (prefill)"):
            if s.lower().startswith(key.lower() + ":"):
                d[key] = s.split(":", 1)[1].strip()
        m = re.match(r"B=(\d+)\s+\(([^)]*)\)?\s*(matvec|matmat):\s*(.*)", s) or re.match(r"B=(\d+)\s+(matvec|matmat):\s*(.*)", s)
        if m:
            g = m.groups()
            B, rest = int(g[0]), g[-1]
            picks = {}
            for part in rest.split("|"):
                if "->" in part:
                    k, v = part.split("->", 1)
                    picks[k.strip()] = v.strip()
            d["kernel_by_B"][B] = picks
    return d


def step_binary(bin_path, out, jthreads):
    r = {"bin": bin_path}
    rc, so, se = sh([bin_path, "--caps"], timeout=30)
    with open(os.path.join(out, "caps.txt"), "w") as f:
        f.write(so + se)
    r["caps"] = parse_caps(so + se) if rc == 0 else {"error": se.strip()[:200]}
    dj = os.path.join(out, "dispatch.json")
    rc, so, se = sh([bin_path, "--dispatch-map", "-j", str(jthreads)], env={"QWEN_DISPATCH_JSON": dj}, timeout=30)
    with open(os.path.join(out, "dispatch.txt"), "w") as f:
        f.write(so + se)
    try:
        dm = json.load(open(dj))
        r["isa_class"] = dm.get("isa_class")
        r["features"] = {ft["id"]: (ft.get("resolved"), ft.get("reason", "")) for ft in dm.get("features", [])}
    except Exception:
        r["isa_class"] = None
        r["features"] = {}
        r["dispatch_error"] = se.strip()[:200]
    rc, so, se = sh([sys.executable, os.path.join(ROOT, "tools", "dispatch_gate.py"), dj], timeout=30)
    r["gate_rc"] = rc
    r["gate_lines"] = [ln for ln in (so + se).splitlines() if re.search(r"SUSPICIOUS|MISMATCH|PASS|FAIL", ln)][:12]
    known = PP.engine_known_flags(bin_path) if PP else set()
    r["known_flags"] = sorted(known) if known else []
    return r


# ---------------------------------------------------------------------------------------
# 4. shapes without a model
# ---------------------------------------------------------------------------------------
def parse_matmat_bench(txt):
    res, shape = {}, None
    for line in txt.splitlines():
        m = re.match(r"\s*\[\s*(\d+)x\s*(\d+)\]", line)
        if m:
            shape = f"{m.group(1)}x{m.group(2)}"
            res[shape] = {}
            continue
        m = re.match(r"\s*(bf16|int8|int4|q4_0)\s+seq\s+([\d.]+) ms\s+batch\s+([\d.]+) ms\s+SPEEDUP\s+([\d.]+)x", line)
        if m and shape:
            res[shape][m.group(1)] = {"seq_ms": float(m.group(2)), "batch_ms": float(m.group(3)), "speedup": float(m.group(4))}
    return res


def step_shapes(bin_path, out, jthreads, Bs=(2, 4), tune=False, model_dir=None, budget_left=30):
    r = {"matmat_bench": {}, "threads": jthreads}
    for B in Bs:
        rc, so, se = sh([bin_path, "--matmat-bench", "-j", str(jthreads)], env={"QWEN_BATCH_B": B}, timeout=60)
        with open(os.path.join(out, f"matmat_bench_B{B}.txt"), "w") as f:
            f.write(so + se)
        r["matmat_bench"][B] = parse_matmat_bench(so) if rc == 0 else {"error": se.strip()[:120]}
    if tune and budget_left > 15:
        tj = os.path.join(out, "matmat_tune.json")
        cmd = [bin_path, "--matmat-tune", "-j", str(jthreads)] + (["-d", model_dir] if model_dir else [])
        rc, so, se = sh(cmd, env={"QWEN_TUNE_QUICK": 1, "QWEN_TUNE_JSON": tj}, timeout=max(20, int(budget_left)))
        with open(os.path.join(out, "matmat_tune.txt"), "w") as f:
            f.write(so + se)
        try:
            r["tune"] = json.load(open(tj))
        except Exception:
            r["tune"] = {"error": (se or so).strip()[-200:], "rc": rc}
    return r


# ---------------------------------------------------------------------------------------
# 5. cost model
# ---------------------------------------------------------------------------------------
def int8_bytes(d):
    qd, kd = d["heads"] * d["hd"], d["kv"] * d["hd"]
    per_layer = qd * d["hidden"] + 2 * kd * d["hidden"] + d["hidden"] * qd + 2 * d["inter"] * d["hidden"] + d["hidden"] * d["inter"]
    return per_layer * d["layers"], d["vocab"] * d["hidden"]


def model_bytes(name):
    m = MODELS[name]
    tk_body, tk_head = int8_bytes(m["talker"])
    cp_body, cp_head = int8_bytes(m["cp"])
    cp_ws = cp_body + m["codebooks"] * cp_head            # weights resident across the 16 CP steps
    return {
        "talker_step_bytes": tk_body + tk_head,          # read once per frame
        "cp_ws_bytes": cp_ws,                            # working set that wants to sit in LLC
        "cp_frame_bytes": m["codebooks"] * cp_body + m["codebooks"] * cp_head,  # re-read 16x per frame
    }


def predict(model, K, B, q, gemv_gbs, llc_share_mb, amx, l3_gbs=None, kcal=CAL):
    """per-worker frame time [ms] for B concurrent slots, K threads, decoder quantum q.
    gemv_gbs: the engine's int8 GEMV rate streaming weights from DRAM with K threads;
    l3_gbs:   the same kernel on a cache-resident CP-sized set (defaults to the transferred ratio)."""
    mb = model_bytes(model)
    if not gemv_gbs:
        return None
    l3_gbs = l3_gbs or gemv_gbs * kcal["l3_over_dram"]
    b_scale = 1.0 + (kcal["talker_b2_over_b1"] - 1.0) * (B - 1)         # B2=1.10, B3=1.20 (extrapolated)
    talker = mb["talker_step_bytes"] / (gemv_gbs * 1e9) * 1e3 * b_scale
    fits = (mb["cp_ws_bytes"] / 1e6) <= (llc_share_mb or 0)
    cp = mb["cp_frame_bytes"] / ((l3_gbs if fits else gemv_gbs) * 1e9) * 1e3 * b_scale
    kr = kcal["ref_threads"] / float(K)
    dec = (kcal["dec_glue_ms"] * (kr ** 0.5) + B * q * kcal["dec_per_item_frame_ms"] * kr) / q
    if not amx:
        dec *= kcal["dec_non_amx_factor"]
    frame = talker + cp + dec
    return {"talker_ms": talker, "cp_ms": cp, "dec_ms": dec, "frame_ms": frame,
            "rho": frame / FRAME_MS, "cp_fits_llc": fits, "B": B, "K": K, "q": q}


B_MATMAT_MAX = 16   # the int8 matmat family accepts B<=16 (dispatch-map matmat.int8.batch_ceiling)


def ceiling(model, W, K, gemv_gbs, llc_share_mb, amx, l3_gbs=None, q=4, kcal=CAL):
    """Three concurrency ceilings for one W x K shape, per worker then x W.

    physics : weights stream only — Talker + CP bytes at the measured GEMV roof, perfect
              batching (B2/B1 = 1.10 per extra slot), decoder free.  Nothing on this shape can
              stream more than W x B_physics; the gap to the next number is decoder + glue.
    model   : the full cost model (Talker + CP + decoder) at rho <= 1.0.  On a non-AMX ISA the
              decoder term is the [GUESS] x1.5 factor, so this number is the weakest.
    floor   : W x 1 — what the shape gives if the effective per-worker batch stays ~1 (short,
              desynchronised requests): the wave's `B` column says which ceiling applies.
    Every number is PREDICTED; B is capped at the matmat ceiling (16).
    """
    if not gemv_gbs:
        return None
    b_phys = b_model = 0
    for B in range(1, B_MATMAT_MAX + 1):
        p = predict(model, K, B, q, gemv_gbs, llc_share_mb, amx, l3_gbs=l3_gbs, kcal=kcal)
        if p["talker_ms"] + p["cp_ms"] <= FRAME_MS:
            b_phys = B
        if p["rho"] <= 1.0:
            b_model = B
    p1 = predict(model, K, 1, q, gemv_gbs, llc_share_mb, amx, l3_gbs=l3_gbs, kcal=kcal)
    pm = predict(model, K, max(b_model, 1), q, gemv_gbs, llc_share_mb, amx, l3_gbs=l3_gbs, kcal=kcal)
    return {"W": W, "K": K, "bw_gbs": gemv_gbs,
            "B_physics": b_phys, "C_physics": W * b_phys,
            "B_model": b_model, "C_model": W * b_model,
            "C_floor": W if p1["rho"] <= 1.0 else 0, "rho_B1": p1["rho"],
            "stream_ms_at_model": pm["talker_ms"] + pm["cp_ms"], "dec_ms_at_model": pm["dec_ms"],
            "dec_share": pm["dec_ms"] / pm["frame_ms"] if pm["frame_ms"] else 0.0}


def ceilings(ident, bwres, isa_class, roofs=None, q=4):
    roofs = roofs or {}
    cands, _ = candidate_topologies(ident)
    llc_per_core = ident.get("llc_per_core_mb") or 0.0
    amx = isa_class == "x86_amx"
    out = {}
    for model in ("1.7b", "0.6b"):
        rows = []
        for W, K in cands:
            g, l3, src = roof_at(roofs, bwres, K)
            c = ceiling(model, W, K, g, llc_per_core * K, amx, l3_gbs=l3, q=q)
            if c:
                c["bw_src"] = src
                rows.append(c)
        out[model] = rows
    return out


def candidate_topologies(ident):
    cores = ident.get("perf_cores") or ident.get("online_cpus") or ident.get("physical_cores") or 1
    doms = [d for d in (ident.get("llc_domains") or []) if d.get("cpus")]
    cands = []
    for W in (1, 2, 3, 4, 6, 8):
        if cores % W or cores // W < 2 and W > 1:
            continue
        K = cores // W
        if W > 1 and K < 2:
            continue
        cands.append((W, K))
    if (1, cores) not in cands:
        cands.insert(0, (1, cores))
    return sorted(set(cands)), len(doms)


def choose_topology(ident, bwres, isa_class, model="1.7b", q=4, roofs=None):
    roofs = roofs or {}
    cands, ndom = candidate_topologies(ident)
    cores = ident.get("online_cpus") or ident.get("physical_cores") or 1
    amx = isa_class == "x86_amx"
    llc_per_core = ident.get("llc_per_core_mb") or 0.0
    rows = []
    for W, K in cands:
        g, l3, src = roof_at(roofs, bwres, K)
        for B in (1, 2, 3, 4):
            if B > 2 and W > 1 and B * W > 8:
                continue
            p = predict(model, K, B, q, g, llc_per_core * K, amx, l3_gbs=l3)
            if p:
                p.update({"W": W, "bw_gbs": g, "l3_gbs": l3, "bw_src": src, "C": W * B})
                rows.append(p)
    # rule of evidence: two workers hold both first audio and RTF from 8 cores up (Arm 16c
    # sweep; c4 2x6); one worker exposes every stream to every admission's inline prefill
    # stall (108-240 ms) and a single lockstep loop.  Multi-LLC parts: one worker per LLC.
    if ident.get("perf_cores"):
        W, cores = 1, ident["perf_cores"]
    elif cores < 8:
        W = 1
    elif ndom > 1 and cores // ndom >= 4:
        W = ndom
    else:
        W = 2
    K = cores // W
    rule = ("hybrid P/E part -> one worker on the performance cores" if ident.get("perf_cores") else
            "cores<8 -> one worker" if cores < 8 else
            ("one worker per LLC domain" if (ndom > 1 and cores // ndom >= 4) else
             "two workers (reference rule; 1xN wins C=1 TTFA only)"))

    def frontier(W, K):
        """largest B whose predicted rho stays under the preferred gate.  Cap 2 is the
        measured frontier on the reference host; B>=3 is unmeasured anywhere and is offered
        only for a single worker, flagged."""
        fr = {r["B"]: r for r in rows if r["W"] == W and r["K"] == K}
        best = None
        for B in (1, 2, 3, 4):
            r = fr.get(B)
            if r and r["rho"] <= CAL["rho_preferred"] and (B <= 2 or W == 1):
                best = B
        return best, fr

    Bsel, fr = frontier(W, K)
    override = None
    if Bsel is None and rows:
        # the rule-based shape does not stream even one request per worker here: pick the
        # shape with the largest predicted streamable concurrency instead, and say so
        alts = []
        for (w2, k2) in cands:
            b2, _ = frontier(w2, k2)
            if b2:
                alts.append((w2 * b2, -w2, w2, k2, b2))
        if alts:
            alts.sort(reverse=True)
            _, _, w2, k2, b2 = alts[0]
            override = f"rule-based {W}x{K} predicts NOT STREAMABLE even at B=1 (rho {fr.get(1, {}).get('rho', float('nan')):.2f}); cost model prefers {w2}x{k2}"
            W, K, Bsel = w2, k2, b2
            rule = "cost-model override (" + rule + ")"
    if Bsel is None:
        Bsel = 1
    return {"W": W, "K": K, "B": Bsel, "C": W * Bsel, "rows": rows, "llc_domains": ndom,
            "rule": rule, "override": override}


# ---------------------------------------------------------------------------------------
# 6. recommendation + draft profile
# ---------------------------------------------------------------------------------------
def env_set(isa_class, known_flags):
    spec = ISA_ENV.get(isa_class, [])
    if isinstance(spec, str):
        spec = ISA_ENV[spec]
    rows = list(COMMON_ENV) + list(spec)
    out = []
    for k, v, label, why in rows:
        declared = (not k.startswith("QWEN_")) or (not known_flags) or (k in known_flags)
        out.append({"key": k, "value": v, "label": label if declared else "NOT-DECLARED", "why": why,
                    "declared": declared})
    return out


ISA_PROFILE_HINT = {"x86_amx": ("amx",), "x86_avx512bf16": ("vnni", "avx512"), "x86_avx512vnni": ("vnni", "avx512"),
                    "x86_avx2": ("avx2",), "arm_i8mm_bf16": ("axion", "arm", "graviton", "kleidi", "generic"),
                    "apple_i8mm_bf16": ("apple", "arm"), "arm_dotprod": ("arm",), "apple_m1": ("apple",)}


def related_profiles(isa_class):
    """Committed configs/perf profiles that describe this ISA family: the place a new box's
    profile is copied FROM, once the wave has replaced the doctor's predictions.  Matched on
    the profile id and its declared ISA features only (descriptions mention other ISAs)."""
    out = []
    if not PP or not os.path.isdir(PP.PERF):
        return out
    hints = ISA_PROFILE_HINT.get(isa_class, ())
    for fn in sorted(os.listdir(PP.PERF)):
        if not fn.endswith(".json") or fn == "schema.json":
            continue
        try:
            d = json.load(open(os.path.join(PP.PERF, fn)))
        except Exception:
            continue
        prof, hw = d.get("profile", {}), d.get("hardware") or {}
        pid = str(prof.get("id", fn[:-5])).lower()
        feats = " ".join(hw.get("isa_features_used") or []).lower()
        has_amx = "amx" in pid or "amx" in feats
        neutral = str(hw.get("architecture", "")).lower() == "any"
        hit = neutral or any(h in pid for h in hints) or (isa_class == "x86_amx" and has_amx) \
            or (isa_class.startswith("x86_avx512") and "avx512vnni" in feats)
        if not hit or (isa_class != "x86_amx" and has_amx and not neutral):
            continue
        status = (d.get("qualification") or {}).get("status") or ("alias" if prof.get("alias_of") else "unspecified")
        out.append(f"{pid} [{status}]")
    return out


def build_argv(topo, port=8080, model_dir="MODEL_DIR", cap=None):
    W, K = topo["W"], topo["K"]
    a = ["./qwen_tts", "-d", model_dir, "--int8", "--serve", str(port), "--batch-size", str(cap or topo["B"])]
    if W > 1:
        a += ["--prefork", str(W), "--prefork-threads", str(K)]
    else:
        a += ["-j", str(K)]
    a += ["--max-queue", "0", "--queue-timeout-ms", "0", "--max-request-seconds", "60"]
    return a


def draft_profile(ident, binary, topo, envrows, isa_class, name, bwres):
    env = {}
    for r in envrows:
        env[r["key"]] = {"value": r["value"],
                         "why": f"[{r['label']}] {r['why']}",
                         "revert": "unset" if r["value"] is not None else "set only for a measured control"}
    for k, why in DO_NOT_SET:
        env.setdefault(k, {"value": None, "why": f"[OFF] {why}", "revert": "set only in a dedicated falsifier"})
    simd = (binary.get("caps") or {}).get("simd", "auto")
    hostbw = (bwres.get("host") or {})
    return {
        "schema_version": 1,
        "profile": {
            "id": name,
            "description": ("DOCTOR DRAFT — predicted from this machine's identity and bandwidth plus "
                            "constants transferred from the streaming reference host. Nothing here is "
                            "qualified; run the verify commands the doctor printed, then replace the "
                            "predictions with measurements."),
            "objective": {"primary": "ttfa", "secondary": "realtime_factor",
                          "concurrency_range": [1, max(1, topo["C"])],
                          "preferred_concurrency": topo["C"], "quality_changes_allowed": False},
        },
        "hardware": {
            "architecture": ident.get("arch") or platform.machine(),
            "cpu_family": ident.get("cpu_model") or "unspecified",
            "logical_cpus": ident.get("logical_cpus") or 0,
            "physical_cores": ident.get("physical_cores") or 0,
            "threads_per_core": 1 if str(ident.get("smt", "")).startswith("off") else 2,
            "numa_nodes": ident.get("numa_nodes") or 1,
            "l3_mib": int(ident.get("llc_mb") or 0),
            "ram_gib": ident.get("ram_gib") or 0,
            "isa_features_used": list(ident.get("isa") or []),
            "notes": (f"online cpu mask {ident.get('online_mask')}; measured host read roof "
                      f"{hostbw.get('read_gbs')} GB/s @{hostbw.get('threads')}T on mask {hostbw.get('mask')} "
                      f"[{bwres.get('source', {}).get('host', 'UNKNOWN')}]. A worker roof belongs to its own mask; "
                      f"never divide the host roof."),
            "canonical_topology": f"{topo['W']}x{topo['K']} [PREDICTED by tools/doctor.py — {topo['rule']}]",
        },
        "build": {"target": "blas", "make": {"command": f"make blas SIMD={simd}", "variables": {"SIMD": simd}},
                  "requirements": {"blas": "OpenBLAS pthread build", "isa": isa_class or "unspecified"}},
        "server": {"prefork_workers": topo["W"], "threads_per_worker": topo["K"], "batch_size": topo["B"],
                   "max_queue": 0, "queue_timeout_ms": 0, "max_request_seconds": 60,
                   "max_text_chars": "unspecified",
                   "worker_affinity": f"engine core-major pinning over online mask {ident.get('online_mask')}; W={topo['W']} K={topo['K']}"},
        "streaming": {"first_chunk_ramp": [1, 2, 4], "decode_chunk": 4, "decode_chunk_busy": 0},
        "runtime": {
            "precision": {"talker_weights": "int8 per output row", "code_predictor_weights": "int8 per output row",
                          "speech_decoder_weights": "int8 where the ISA has int8 decoder kernels, fp32 otherwise",
                          "activations": "fp32 at the API boundary", "gemm_backend": isa_class or "unspecified"},
            "environment": env,
        },
        "launch": {"executable": "./qwen_tts", "extra_arguments": [], "server_env_serialization": "comma"},
        "qualification": {"status": "unqualified", "workload": "NONE — doctor prediction only",
                          "benchmark_family": "TRUE_SIMULTANEOUS_WAVE", "model": "qwen3-tts-1.7b --int8 (assumed)",
                          "notes": "Replace every [PREDICTED]/[TRANSFERRED] value with a Tier-A wave + playback envelope on this host."},
    }


def validate_draft(prof):
    if not PP:
        return ["perf_profile unavailable"]
    try:
        with open(os.path.join(PP.PERF, "schema.json")) as f:
            schema = json.load(f)
        PP.check(prof, schema, schema, "$")
        return PP.semantic(prof, "doctor-draft", engine=None)
    except PP.Bad as e:
        return [str(e)]


# ---------------------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------------------
def render(rep):
    L = []
    ident, bw, binr, shp, topo, rec = (rep["identity"], rep["bandwidth"], rep["binary"],
                                       rep["shapes"], rep["topology"], rep["recommendation"])
    arm = rep.get("arm_gemv_scaling") or {
        "status": "SKIPPED", "verdict": "SKIPPED / ARM multi-worker GEMV preflight not run",
        "reason": "not recorded in this report", "rows": [],
    }
    L.append(f"DOCTOR  {ident.get('host')}  {rep['utc']}   {rep['elapsed_s']:.1f}s   out={rep['out']}")
    L.append("=" * 100)
    L.append("legend: [MEASURED] here now · [CACHED] here earlier · [TRANSFERRED] reference-host constant · [PREDICTED] model · [UNKNOWN]")
    L.append("")
    L.extend(render_arm_gemv_scaling(arm))
    L.append("")
    L.append("1. MACHINE [MEASURED]")
    L.append(f"   {ident.get('cpu_model')}   cloud={ident.get('cloud') or '-'}   os={ident.get('os')}/{ident.get('arch')}")
    L.append(f"   physical cores {ident.get('physical_cores')}  logical {ident.get('logical_cpus')}  online mask {ident.get('online_mask')} ({ident.get('online_cpus')} cpus)  SMT {ident.get('smt')}  sockets {ident.get('sockets')}  NUMA {ident.get('numa_nodes')}")
    if ident.get("offline_siblings_in_mask"):
        L.append("   WARN affinity mask names OFFLINE cpus (SMT siblings): topology tools must use the online mask above")
    L.append(f"   LLC {ident.get('llc_mb')} MB ({ident.get('llc_what')}), {ident.get('llc_per_core_mb')} MB/core, L2 {ident.get('l2_mb')} MB/core, domains {len(ident.get('llc_domains') or [])}   RAM {ident.get('ram_gib')} GiB  swap {ident.get('swap_mb')} MB")
    L.append(f"   ISA: {' '.join(ident.get('isa') or [])}")
    g = ident.get("gates", {})
    L.append(f"   gates: SMT off {g.get('smt_off')} · governor {g.get('governor_performance')} · cgroup quota {g.get('no_cgroup_quota')}   (FAIL = every number afterwards describes another machine)")
    for w in ident.get("warnings") or []:
        L.append(f"   WARN {w}")
    L.append("")
    L.append("2. BANDWIDTH (read = the GEMV weight stream; the number that bounds Talker/CP at B=1)")
    h = bw.get("host")
    if h:
        L.append(f"   host mask {h['mask']} [{bw['source'].get('host')}]: read {h['read_gbs']} · copy {h.get('copy_gbs')} · triad {h.get('triad_gbs')} GB/s @{h['threads']}T   sweep: " +
                 "  ".join(f"{t}T:{v:.0f}" for t, v in h["sweep"]))
        if h["sweep"]:
            t1 = dict(h["sweep"]).get(1)
            knee = h.get("t90")
            if t1:
                L.append(f"   per-thread {t1:.1f} GB/s; 90 % of peak at {knee}T -> " + ("scales to the last core: core-bound, not DRAM-bound" if knee == h['threads'] else f"saturates before {h['threads']}T: DRAM-bound above {knee}T"))
    else:
        L.append(f"   host roof UNKNOWN: {'; '.join(bw.get('notes') or ['no measurement'])}")
    w = bw.get("worker")
    if w:
        L.append(f"   worker mask {w['mask']} [{bw['source'].get('worker')}]: read {w['read_gbs']} GB/s @{w['threads']}T  (this is the roof ONE worker of the suggested size actually gets)")
    for K, r in sorted((bw.get("gemv") or {}).items()):
        L.append(f"   GEMV roof @{K}T mask {r['mask']} [{r['source']}]: 1.7B Talker frame {r['talker17b_frame_ms']:.1f} ms = {r['gemv_gbs']:.1f} GB/s from DRAM · "
                 f"{'%.1f GB/s' % r['l3_gbs'] if r.get('l3_gbs') else 'n/a'} on a {r.get('l3_set_mb', 0):.0f} MB cache-resident set · read-only loop {r.get('read_only_gbs')} GB/s")
    if bw.get("gemv"):
        L.append("   (the GEMV roof is the engine's own kernel; membw's read loop under-reports it — the cost model uses the GEMV numbers)")
    for n in bw.get("notes") or []:
        L.append(f"   note: {n}")
    L.append("")
    caps = binr.get("caps", {})
    L.append("3. BINARY + DISPATCH [MEASURED]")
    L.append(f"   {binr.get('bin')}  build {caps.get('build')}  SIMD={caps.get('simd')}  src={caps.get('src')}   isa_class={binr.get('isa_class')}   threads-in-probe={shp.get('threads')}")
    L.append(f"   runtime cpu: {caps.get('runtime_cpu')}")
    for key in ("x86 amx int8", "x86 amx bf16", "int8 dot", "bf16 dot", "kleidi", "BLAS (prefill)"):
        if key in caps:
            L.append(f"   {key}: {caps[key]}")
    for B in sorted(caps.get("kernel_by_B", {})):
        L.append(f"   B={B:<2} " + " | ".join(f"{k} -> {v}" for k, v in caps["kernel_by_B"][B].items()))
    for ln in binr.get("gate_lines", []):
        L.append(f"   gate: {ln.strip()}")
    interesting = ("talker.prefill.matmat_bf16", "cp.prefill2", "decoder.int8", "decoder.pool", "pool.spin", "region.int8_runner", "matmat.int8.batch_ceiling", "kleidi.enabled")
    for fid in interesting:
        if fid in binr.get("features", {}):
            v, why = binr["features"][fid]
            L.append(f"   {fid:<32} {v:<10} {why[:70]}")
    L.append("")
    L.append("4. SHAPES WITHOUT A MODEL [MEASURED]  (--matmat-bench: batched matmat vs B x GEMV, hot cache; a SPEEDUP < 1 means batching does not pay for that width)")
    for B, tab in sorted(shp.get("matmat_bench", {}).items()):
        if "error" in tab:
            L.append(f"   B={B}: {tab['error']}")
            continue
        for shape, fmts in tab.items():
            L.append(f"   B={B} [{shape:>9}] " + "  ".join(f"{f}: seq {v['seq_ms']:.2f} batch {v['batch_ms']:.2f} x{v['speedup']:.1f}" for f, v in fmts.items()))
    if shp.get("tune"):
        t = shp["tune"]
        if "error" in t:
            L.append(f"   tune: {t['error']}")
        else:
            L.append(f"   tune (quick grid, {t.get('shapes_source', '')[:60]}...): {len(t.get('cells', []))} cells, recommend: {t.get('recommend', [])[:6]}")
    L.append("")
    L.append(f"5. COST MODEL [PREDICTED]  calibrated on {CAL['ref_host']} — {CAL['model_error']}")
    for model in ("1.7b", "0.6b"):
        mb = model_bytes(model)
        L.append(f"   {model}: Talker step {mb['talker_step_bytes']/1e6:.0f} MB int8/frame · CP working set {mb['cp_ws_bytes']/1e6:.0f} MB (re-read 16x/frame) · decoder {'AMX Design-D calibration' if topo.get('amx') else 'NO calibration on this ISA: x' + str(CAL['dec_non_amx_factor']) + ' [GUESS]'}")
    L.append(f"   per worker, decoder quantum q=4, frame budget {FRAME_MS:.0f} ms; rho = predicted frame time / 80 ms; gate {CAL['rho_hard']} hard / {CAL['rho_preferred']} preferred")
    L.append("   model  W x K   B  C   GEMV GB/s (source)                     talker    cp   dec  frame   rho   verdict")
    if not any(rep["predictions"][m] for m in ("1.7b", "0.6b")):
        L.append("   [UNKNOWN] no GEMV roof and no read roof for this host (build /tmp/qwen_roof_matvec with `make roof-matvec`, membw with `make membw`): no rho can be predicted")
    for model in ("1.7b", "0.6b"):
        for r in rep["predictions"][model]:
            verdict = ("OK" if r["rho"] <= CAL["rho_preferred"] else "MARGINAL" if r["rho"] <= CAL["rho_hard"] else "NOT STREAMABLE")
            if r["B"] >= 3:
                verdict += f" (B{r['B']} unmeasured anywhere)"
            L.append(f"   {model:<5}  {r['W']}x{r['K']:<3}   {r['B']}  {r['C']:<2}  {r['bw_gbs'] or 0:6.1f} {('(' + r['bw_src'] + ')')[:34]:<34} {fmt_ms(r['talker_ms'])} {fmt_ms(r['cp_ms'])} {fmt_ms(r['dec_ms'])} {fmt_ms(r['frame_ms'])}  {r['rho']:.2f}   {verdict}")
    L.append("")
    L.append("6. RECOMMENDATION  (a starting point, not a result)")
    L.append(f"   topology  {topo['W']}x{topo['K']}  [PREDICTED: {topo['rule']}]   batch cap {topo['B']}  [{'MEASURED-REF frontier (cap 2 on c4)' if topo['B'] == 2 else 'PREDICTED'}]   -> predicted streamable concurrency C{topo['C']} on 1.7B")
    if topo.get("override"):
        L.append(f"   NOTE {topo['override']}")
    if rec.get("alt"):
        L.append(f"   alternative {rec['alt']}")
    L.append("   argv: " + " ".join(rec["argv"]))
    L.append("   env (label · why):")
    for r in rec["env"]:
        val = "ABSENT" if r["value"] is None else r["value"]
        L.append(f"     {r['key']:<32} {val:<8} [{r['label']}] {r['why'][:80]}")
    L.append("   must stay OFF/absent (declared flags with a recorded verdict):")
    for k, why in DO_NOT_SET:
        L.append(f"     {k:<32} {why[:90]}")
    L.append("   candidates to A/B, not to pin:")
    for k, v, why, isa_only in CANDIDATES:
        if isa_only and isa_only != (rep["binary"].get("isa_class")):
            continue
        L.append(f"     {k}={v:<6} {why[:90]}")
    if rec.get("undeclared"):
        L.append(f"   WARN this binary does not declare: {', '.join(rec['undeclared'])} (older/newer build than the flag set; drop them)")
    if rec.get("related"):
        L.append("   committed profiles for this ISA family (copy the wave's winner FROM one of these, not from the draft's guesses): " + ", ".join(rec["related"]))
    if rec.get("draft_errors"):
        L.append(f"   draft profile validation: {rec['draft_errors']}")
    else:
        L.append(f"   draft profile: {rec['draft_path']}  (schema-valid, status=unqualified; copy into configs/perf/ only after the wave)")
    L.append("")
    L.append("7. VERIFY NEXT (in this order; each replaces a [PREDICTED] with a [MEASURED])")
    for c in rec["verify"]:
        L.append(f"   {c}")
    L.append("")
    L.append("8. CEILING [PREDICTED]  how many streams this host can hold at most, per shape, and what stands between the numbers")
    L.append("   physics = Talker+CP bytes at the measured GEMV roof with perfect batching and a free decoder: nothing on the shape streams more")
    L.append("   model   = the same plus the decoder term at rho <= 1.0" + ("  (decoder NOT calibrated on this ISA: [GUESS] x1.5, the weakest term)" if not (rep.get('topology') or {}).get('amx') else ""))
    L.append("   floor   = W x 1: what you get if the effective per-worker batch stays ~1 (short or desynchronised requests); the wave's B column decides which applies")
    L.append("   model  W x K   GEMV GB/s   physics C (B)   model C (B)   floor C   at model B: stream ms  dec ms  dec share")
    for model in ("1.7b", "0.6b"):
        for c in (rep.get("ceiling") or {}).get(model, []):
            L.append(f"   {model:<5}  {c['W']}x{c['K']:<3}  {c['bw_gbs'] or 0:8.1f}    {c['C_physics']:3d} ({c['B_physics']:2d})        {c['C_model']:3d} ({c['B_model']:2d})      {c['C_floor']:3d}        {c['stream_ms_at_model']:6.1f}   {c['dec_ms_at_model']:6.1f}   {c['dec_share']*100:4.0f} %")
    pts = CAL_POINTS.get((rep.get("binary") or {}).get("isa_class") or "", [])
    if pts:
        L.append("   [MEASURED] calibration on this ISA family (STREAM_RTF p95 measured vs rho predicted; the model's trust boundary):")
        for name, meas, pred, note in pts:
            L.append(f"     {name:<40} measured {meas:.2f}  predicted {(f'{pred:.2f}' if pred is not None else '  -  ')}   {note}")
    if rep.get("ceiling"):
        L.append("   reading: physics >> model means the decoder/glue is the wall, not bandwidth (the P4 structural cost); physics ~ model means the shape is bandwidth-bound")
        L.append("            and only a smaller weight stream (fewer CP re-reads, cache-resident CP, lower precision) raises it.  Past the model C the wave measures, the model guesses.")
    return "\n".join(L) + "\n"


def wave_plan(topo, preds17, others, draft_path, isa=None, candidates=CANDIDATES,
              models=("qwen3-tts-1.7b", "qwen3-tts-0.6b")):
    """The grid the doctor recommends, as DATA: tools/doctor_wave.py runs it in order.

    One run per (model, shape): concurrency [C, C, C+2] on the same server — the first level
    is the cold one, the second the warm repeat, the third one step past the prediction.
    Shapes = the recommended W x K plus the best alternative family, each at its predicted cap.
    Then the A/B candidates on the recommended shape at [C, C].  No pgrep, no waiting on
    another process: the runner executes the list sequentially and that is the whole gate.
    """
    shapes = [{"topo": f"{topo['W']}x{topo['K']}", "cap": int(topo["B"]), "C": int(topo["C"]), "why": topo.get("rule", "")}]
    for o in others[:1]:
        shapes.append({"topo": f"{o['W']}x{o['K']}", "cap": int(o["B"]), "C": int(o["C"]),
                       "why": f"alternative family, predicted rho {o['rho']:.2f}"})
    runs = []
    for m in models:
        short = m.replace("qwen3-tts-", "")
        for sh_ in shapes:
            runs.append({"label": f"{short}-{sh_['topo']}-cap{sh_['cap']}", "model": m, "topo": sh_["topo"],
                         "cap": sh_["cap"], "conc": [sh_["C"], sh_["C"], sh_["C"] + 2], "env": {}, "why": sh_["why"]})
    rec = shapes[0]
    for k, v, why, isa_only in candidates:
        if isa_only and isa_only != isa:
            continue   # an A/B for another ISA is noise here, not a run
        runs.append({"label": f"1.7b-{rec['topo']}-ab-{k.lower()}", "model": models[0], "topo": rec["topo"],
                     "cap": rec["cap"], "conc": [rec["C"], rec["C"]], "env": {k: v}, "why": f"A/B candidate: {why}"})
    return {"tool": "tools/doctor.py", "profile": draft_path, "waves": 1, "classes": "short",
            "runner": "python3 tools/doctor_wave.py <this file>", "runs": runs}


# ---------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bin", default=os.path.join(ROOT, "qwen_tts"))
    ap.add_argument("--membw", default=os.environ.get("MEMBW_BIN", "/tmp/qwen_membw"))
    ap.add_argument("--roof", default=os.environ.get("ROOF_MATVEC_BIN", "/tmp/qwen_roof_matvec"),
                    help="tests/roof_matvec_int8.c binary (make roof-matvec); the Talker/CP terms are MEASURED with it")
    ap.add_argument("--no-roof", action="store_true", help="skip the GEMV roof (falls back to membw x transferred factor)")
    ap.add_argument("--no-arm-gemv-preflight", action="store_true",
                    help="skip the fixed-mask Arm 1x8/2x8/4x8 GEMV scaling preflight")
    ap.add_argument("--arm-gemv-reps", type=int, default=3,
                    help="roof_matvec repetitions per worker for the Arm scaling preflight (default: 3)")
    ap.add_argument("--out", default=None, help="artifact dir (default profiles/doctor/<utc>_<host>)")
    ap.add_argument("--store", default=os.path.join(ROOT, "profiles", "roofs"), help="roofs store shared with `make roofs`")
    ap.add_argument("--budget", type=float, default=BUDGET_S_DEFAULT, help="seconds; the tune grid runs only inside it")
    ap.add_argument("--reps", type=int, default=2, help="membw repetitions (cpu-check uses 5)")
    ap.add_argument("--model", default=None, help="model dir for --matmat-tune shapes (optional, no weights read)")
    ap.add_argument("--full", action="store_true", help="also run --self-test and the quick --matmat-tune grid")
    ap.add_argument("--no-measure", action="store_true", help="never run membw; use cached roofs or say UNKNOWN")
    ap.add_argument("--json", action="store_true", help="print doctor.json to stdout instead of the report")
    ap.add_argument("--port", type=int, default=8080)
    a = ap.parse_args()
    if TP is None:
        print(f"doctor: cannot import tools/{{topology,roofs,perf_profile}}.py: {_IMPORT_ERR}", file=sys.stderr)
        return 2
    t0 = time.time()
    utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    host = platform.node().split(".")[0]
    out = a.out or os.path.join(ROOT, "profiles", "doctor", f"{time.strftime('%Y-%m-%d_%H%M%S')}_{host}")
    os.makedirs(out, exist_ok=True)
    if not os.access(a.bin, os.X_OK):
        print(f"doctor: binary {a.bin} missing or not executable: make blas", file=sys.stderr)
        return 1

    hw, err = step_identity(out)
    if hw is None:
        print(f"doctor: {err}", file=sys.stderr)
        return 1
    ident = identity_summary(hw)
    arm_gemv = step_arm_gemv_preflight(
        a.roof, ident, out, reps=a.arm_gemv_reps,
        enabled=not a.no_roof and not a.no_arm_gemv_preflight)

    # binary first: isa_class decides the decoder calibration and the env set
    cores = ident["online_cpus"] or ident["physical_cores"] or 1
    pre_topo = choose_topology(ident, {"host": None, "worker": None, "source": {}}, None)
    binr = step_binary(a.bin, out, pre_topo["K"])
    isa = binr.get("isa_class")

    worker_mask = first_n(ident["online_mask"], pre_topo["K"]) if pre_topo["W"] > 1 else None
    bw = step_bandwidth(hw, ident, a.membw, a.store, worker_mask, a.reps, measure=not a.no_measure)

    # the engine's own GEMV roof at the two worker sizes the recommendation will compare:
    # the rule-based worker (K threads on its mask) and the whole host (one worker, N threads)
    sizes = {pre_topo["K"]: worker_mask or ident["online_mask"]}
    if pre_topo["W"] > 1:
        sizes[cores] = ident["online_mask"]
    # every candidate worker width gets ITS OWN measured roof: scaling the 8T roof to 16T by
    # the membw sweep predicted 198 GB/s on a 4-CCX Zen5 where 2x16 measured as if 87; a
    # roof belongs to one mask (2026-09-08, c8a.8xlarge)
    for W_, K_ in candidate_topologies(ident)[0]:
        if K_ not in sizes and K_ >= 2:
            sizes[K_] = first_n(ident["online_mask"], K_)
    roofs, roof_notes = ({}, ["--no-roof"]) if a.no_roof else step_gemv_roofs(a.roof, ident, sizes, out, reps=a.reps + 1)
    bw["notes"] = (bw.get("notes") or []) + roof_notes
    bw["gemv"] = roofs

    topo = choose_topology(ident, bw, isa, roofs=roofs)
    topo["amx"] = isa == "x86_amx"
    preds = {m: [r for r in choose_topology(ident, bw, isa, model=m, roofs=roofs)["rows"]] for m in ("1.7b", "0.6b")}

    shp = step_shapes(a.bin, out, topo["K"], tune=a.full, model_dir=a.model, budget_left=a.budget - (time.time() - t0))
    if a.full:
        rc, so, se = sh([a.bin, "--self-test"], timeout=180)
        with open(os.path.join(out, "selftest.txt"), "w") as f:
            f.write(so + se)
        binr["self_test"] = "PASS" if rc == 0 else f"FAIL rc={rc}"

    known = set(binr.get("known_flags") or [])
    envrows = env_set(isa, known)
    undeclared = [r["key"] for r in envrows if not r["declared"]]
    envrows = [r for r in envrows if r["declared"]]
    argv = build_argv(topo, a.port)
    # alternative: the best-rho topology of the other family, if any, so the wave compares two shapes
    alt = None
    best = {}
    for r in preds["1.7b"]:
        if (r["B"] <= 2 or r["W"] == 1) and r["rho"] <= CAL["rho_preferred"]:
            key = (r["W"], r["K"])
            if key not in best or r["C"] > best[key]["C"]:
                best[key] = r
    others = sorted((v for k, v in best.items() if k != (topo["W"], topo["K"])), key=lambda r: (-r["C"], r["rho"]))
    if others:
        o = others[0]
        alt = f"{o['W']}x{o['K']} cap {o['B']} predicts C{o['C']} at rho {o['rho']:.2f} — the cost model cannot see admission/prefill coupling, so run both shapes in the wave"
    name = f"doctor-{host}-{topo['W']}x{topo['K']}".lower().replace("_", "-")
    prof = draft_profile(ident, binr, topo, envrows, isa, name, bw)
    errs = validate_draft(prof)
    dpath = os.path.join(out, "profile-draft.json")
    with open(dpath, "w") as f:
        json.dump(prof, f, indent=2)
    topos = ",".join(sorted({f"{r['W']}x{r['K']}" for r in preds['1.7b'] if r['rho'] <= 1.0} | {f"{topo['W']}x{topo['K']}"}))
    concs = sorted({1, topo["C"], topo["C"] + 1} | ({others[0]["C"]} if others else set()))
    plan = wave_plan(topo, preds["1.7b"], others, relp(dpath), isa=isa)
    ppath = os.path.join(out, "wave-plan.json")
    with open(ppath, "w") as f:
        json.dump(plan, f, indent=2)
    verify = [
        f"make cpu-check                                   # qualification preflight (self-test, roofs @5 reps, dispatch gate)",
        f"python3 tools/doctor_wave.py {relp(ppath)}   # THE grid above as one sequential run ({len(plan['runs'])} runs, `make doctor-wave` = LATEST); the lines below are what it does by hand",
        f"make roofs ROOF_MASKS={worker_mask or ident['online_mask']}                 # the worker roof for every mask the wave will use",
        f"make bench-topo BENCH_MODEL=qwen3-tts-1.7b BENCH_PROFILE=<copy of {relp(dpath)}> BENCH_TOPO={topos} BENCH_CONC={','.join(str(c) for c in concs)}",
        f"python3 tests/serve_parallel_wave.py --profile <id> --topo {topo['W']}x{topo['K']} --conc {','.join(str(c) for c in concs)} --waves 3   # playback envelope: safe_play_start, stall@250/500",
        "tools/perf_profile.py check-flags <id> --log server.log   # a flag is on when the process says so",
        "then: replace every [PREDICTED] in the draft with the measured value and set status=qualified",
    ]
    rec = {"argv": argv, "env": envrows, "undeclared": undeclared, "draft_path": relp(dpath), "related": related_profiles(isa),
           "draft_errors": errs, "verify": verify, "alt": alt}
    rep = {"tool": "tools/doctor.py", "utc": utc, "elapsed_s": time.time() - t0, "out": relp(out),
           "identity": ident, "arm_gemv_scaling": arm_gemv, "bandwidth": bw, "binary": binr, "shapes": shp,
           "topology": {k: v for k, v in topo.items() if k != "rows"}, "predictions": preds,
           "ceiling": ceilings(ident, bw, isa, roofs=roofs),
           "calibration": CAL, "recommendation": rec}
    with open(os.path.join(out, "doctor.json"), "w") as f:
        json.dump(rep, f, indent=1, default=str)
    txt = render(rep)
    with open(os.path.join(out, "doctor.txt"), "w") as f:
        f.write(txt)
    latest = os.path.join(os.path.dirname(out), "LATEST")
    try:
        if os.path.islink(latest) or os.path.exists(latest):
            os.remove(latest)
        os.symlink(os.path.basename(out), latest)
    except OSError:
        pass
    if a.json:
        print(json.dumps(rep, indent=1, default=str))
    else:
        sys.stdout.write(txt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
