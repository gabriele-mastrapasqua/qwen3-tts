# x86 optimization: AVX2, AVX-512 & VNNI — findings and how to benchmark

*Shareable write-up of the x86 SIMD work shipped in **v0.9.0** — what changed, why it helps (and
where it doesn't), the measured numbers, and copy-paste commands to benchmark it on your own CPU.*

---

## TL;DR

- The hot-path kernels (matvec + attention, ~90% of decode) now have **AVX2** and **AVX-512 / VNNI**
  implementations on x86, alongside the existing NEON+SDOT on ARM — with a scalar fallback and a
  runtime ISA guard. Decode threading runs on a **cross-OS pool** (was macOS-only).
- On x86 the **int8 kernel stack is a ~1.85× win at equal core count** (measured: scalar-bf16 `-j1`
  RTF 3.04 → VNNI-int8 `-j1` 1.64 on an EPYC 9555P / Zen5).
- The decode is **memory/cache-bound** (the Code Predictor re-reads its weights 16× per frame), so:
  - SIMD width (AVX2 → AVX-512) buys only a little on its own.
  - The big levers are **fewer weight bytes** (`--int8`, `--int4`) and **a CPU cache that fits the
    working set** (large L3 / 3D V-cache / Apple's SLC).
- **If you have a Zen4/Zen5 X3D desktop (e.g. Ryzen 9 7950X3D / 9950X3D): you are the best case.**
  V-cache + AVX-512/VNNI + high clock + bare metal is exactly what this workload wants.

---

## What changed in v0.9.0 (x86)

| Area | Before | v0.9.0 |
|---|---|---|
| matvec/attention SIMD on x86 | scalar | **AVX2+FMA**, **AVX-512** (`__m512` bf16 matvec), **AVX-512-VNNI** int8 dot |
| int8 dot | dequant→f32→FMA | native **`_mm512_dpbusd_epi32`** (VNNI), the x86 analog of ARM SDOT |
| decode threading | macOS/GCD only | **cross-OS pool** (GCD on macOS, persistent pthread pool on Linux/Windows) |
| quantization | int8 (Talker-only on small) | **int8** end-to-end (incl. 0.6B Talker) + **int4 (Q4_0)** on Talker + CP |
| build | `-march=native` | portable `-mavx2` default + `SIMD=` levels + runtime ISA guard |

Build levels:

```bash
make blas                    # SIMD=auto on Linux/x86: reads /proc/cpuinfo, probes the compiler,
                             # and picks the highest level BOTH support (announced as "[simd] auto -> …")
make blas SIMD=amx           # + AMX tiles, int8 and bf16 (Sapphire Rapids / Emerald Rapids, GCP c4)
make blas SIMD=avx512bf16    # + native bf16 dot VDPBF16PS (Zen4/5, Cooper Lake+)
make blas SIMD=avx512vnni    # + AVX-512-VNNI native int8 dot (Zen4+/Intel Ice Lake+)
make blas SIMD=avx512        # AVX-512 without VNNI (__m512 16-wide bf16 matvec)
make blas SIMD=portable      # AVX2 + FMA, any Haswell 2013+ — the level to ship a binary at
make blas SIMD=scalar        # no AVX2 (pre-2013 / portable fallback)
```

`auto` is a *host* decision, not a portable one: the binary it produces will fault on an older
CPU. Build `SIMD=portable` for anything you distribute, and pin the level explicitly in a
deployment profile so the box that reproduces your numbers compiles the same kernels.

`./qwen_tts --caps` prints what the binary actually compiled and the CPU it's running on.

---

## Why it's memory/cache-bound (the core insight)

The bottleneck is the **Code Predictor (CP)**: for every 80 ms audio frame it runs 15 sequential
passes, **re-reading its weights 16× per frame**. That is a lot of bytes pulled from memory per
frame, so the CPU spends most of its time *waiting on the memory subsystem*, not computing.

Consequences, all of which we measured:

1. **SIMD width helps only a little.** AVX2 over scalar was just **~+6%** on a Ryzen 6800H — you
   can't make memory traffic faster with wider math. VNNI over the AVX2 widen-path was **~+5%** on
   the EPYC. The big single-core win (1.85×) is the **int8 quantization itself** (half the bytes),
   not the SIMD instruction.
2. **Fewer weight bytes = the real lever.** bf16 CP ≈ 120 MB, int8 ≈ 60 MB, int4 ≈ 30 MB. On a
   memory-starved CPU, halving/quartering the bytes directly cuts the per-frame wait.
3. **Cache that fits the working set wins everything.** If the CP weights fit the cache the cores
   can actually see, the 16× re-read hits cache instead of DRAM. This is why Apple M1 (large
   system-level cache) reaches **sub-1.0 RTF**, and why a **3D V-cache** chip is so well suited.

---

## Measured findings

### Apple M1 (reference, ARM)
0.6B `--int8` is **faster than real-time in every mode** (CLI ~0.80–0.90, streaming ~0.81–0.89,
HTTP server warm ~0.88, cloned `.qvoice` ~0.93). See [performance.md](performance.md).

### Ryzen 7 6800H (Zen3+, 16 MB L3, no AVX-512, bare metal / WSL2)
- AVX2 only **~+6%** over scalar (memory-bound).
- **`--int4` is the lever** (multi-threaded): RTF **3.9 → 2.02** (−28% vs int8). 4 threads is the
  sweet spot; 8 regresses (memory bus saturates). int4 wins here because the smaller working set
  relieves the bus; on a bandwidth-rich chip like M1, int4 is *slower* than bf16.

### EPYC 9555P (Zen5 "Turin", full-width 512-bit AVX-512 + VNNI, 256 MB L3 split 32 MB/CCD)
The VNNI / AVX-512 validation box. `--caps` reports `int8 dot: VNNI _mm512_dpbusd_epi32 (native)`
and `--self-test` passes (kernels numerically correct).

| 0.6B, EPYC 9555P | RTF | CP ms/f |
|---|---|---|
| scalar bf16 `-j1` (≈ unoptimized) | 3.04 | 164.8 |
| **VNNI int8 `-j1` (v0.9.0)** | **1.64** | 79.3 |
| VNNI int8 `-j4` | 1.78 | 88.7 |
| int8 `-j4` (VNNI off → AVX2 widen) | 1.87 | 91.7 |
| int4 `-j4` | 2.06 | 108.4 |
| bf16 `-j4` | 1.90 | 95.1 |

Two takeaways:
- **The int8 stack is a real ~1.85× win at equal core count** (3.04 → 1.64). The kernel work is doing
  its job — the earlier "is anything even helping?" doubt was a *measurement* artifact, not the code.
- **This was a 4-vCPU VM**, and `-j1` (1.64) actually *beat* `-j4` (1.78): the hypervisor scatters
  vCPUs across different CCDs, so the 4 threads can't share one CCD's 32 MB L3. **On bare metal this
  doesn't happen** — threading scales (as on M1 and the 6800H). A VM slice of a many-core server is
  the worst case for single-stream latency.

### 2026-08-04 update — the AVX-512 parity round (Zen5, `SIMD=avx512bf16`)

The remaining AVX2-only hot paths got true AVX-512 twins, plus two new levers
(all runtime-switchable; full RTF matrix in `docs/hardware-testing.md` §5):

- **16-wide attention, rms_norm and bf16↔f32 conversions** (were AVX2 even on AVX-512 builds):
  branch-vs-main alone is **bf16 −17% / int4 −8%** at `-j1`.
- **Native bf16 dot (`VDPBF16PS`)** for the bf16 matvec: **−21% at `-j1`** (0.6B RTF 1.19 vs 1.51;
  −19% on 1.7B) — bf16 mode now *ties int8* single-thread. `QWEN_NO_BF16DOT=1` opts out.
- **q4-VNNI v3 as default + a fused-QKV VNNI twin** (QKV was still f32-dequant): 0.6B int4 `-j1`
  **1.05 vs int8 1.21** — **the first time int4 beats int8 on x86**. On 1.7B int8 remains the
  wall-clock king (1.74 vs 1.84); the old +21% int4 gap shrank to ~+6%.
- Recommended configs after this round: **0.6B → `--int4 -j4`**, **1.7B → `--int8 -j4`**; build
  with **`make blas SIMD=avx512bf16`** on Zen4/5 / Cooper Lake+ (falls back to `SIMD=avx512vnni`
  on VNNI-only chips like Ice Lake).

### Cross-device summary (0.6B, single-stream, best config)

| Device | SIMD + threads | Best RTF | Lever |
|---|---|---|---|
| Apple M1 | NEON + SDOT, 4-thread | **sub-1.0 (int8)** | big SLC fits the working set |
| Ryzen 7 6800H | AVX2, 4-thread, bare metal | 2.02 | `--int4` (small L3) |
| EPYC 9555P (Zen5, VM) | AVX-512 + VNNI + BF16, 4-thread | **0.95 (int4/int8)** | 2026-08-04 parity round; int4 wins `-j1` (1.05) |

---

### AMX: two consumers, two different jobs

On Sapphire Rapids and Emerald Rapids (GCP `c4`, AWS `m7i`, bare-metal Xeon 4th/5th gen) the
build picks up `amx_tile`, `amx_int8` and `amx_bf16`, and `--caps` says so:

```
  x86 amx int8:     AMX ACTIVE (tile 16x64 int8 GEMM for batched matmat; QWEN_NO_AMX=1 disables)
  x86 amx bf16:     AMX ACTIVE (tile 16x32 bf16 GEMM for batched matmat; QWEN_NO_AMX=1 disables)
```

The two are used by different parts of the engine, and a benchmark that turns *both* off
measures a sum of two unrelated things. Measured on an 8-core Emerald Rapids with SMT off,
1.7B open weights at int8, two pre-forked workers of four threads, three waves:

| control | C=1 TTFA | C=4 TTFA p50 / p95 | C=4 stream RTF | req/s |
|---|---:|---:|---:|---:|
| AMX fully on | 135 ms | 251 / 324 ms | 1.50 | 1.38 |
| `QWEN_NO_AMX_BF16=1` | **188 ms** | **374 / 556 ms** | 1.54 | 1.34 |
| `QWEN_NO_AMX_INT8=1` | 134 ms | 258 / 337 ms | **1.64** | **1.24** |
| `QWEN_NO_AMX=1` | 196 ms | 370 / 601 ms | 1.72 | 1.20 |

- **AMX BF16 is first audio.** It runs the prefill projections (`QWEN_PREFILL_MATMAT`), and
  removing it costs 39% of C=1 TTFA and 72% of C=4 p95 while leaving stream RTF alone.
- **AMX INT8 is sustained throughput.** It runs the batched decode GEMM, and removing it costs
  9% of RTF and 10% of throughput while leaving TTFA alone.
- They compose: disabling both gives the TTFA of the first row and the RTF of the second.

So on a machine with AMX the honest summary is *not* "AMX is worth 30%". It is: AMX BF16 pays
for the prefill, which is what a listener experiences as responsiveness, and AMX INT8 pays for
the decode, which is what decides how many concurrent streams the box carries.

**Where AMX is absent** — Zen4/Zen5, Ice Lake, anything before Sapphire Rapids — the same work
falls to the shape-aware AVX-512 paths, and the VNNI tiling is what matters there: with AMX
disabled, additionally setting `QWEN_NO_VNNI_TILE=1` cost a further 10% of throughput and 7% of
stream RTF on the same host.

### A bandwidth-poor box does not become fast because it has AMX

The same 8-core Emerald Rapids measures **82 GB/s** of Triad against 336 GB/s on a 16-core
Arm Neoverse-V2 host, and it shows: at C=4 it sustains stream RTF ~1.45 while the Arm box holds
0.72. First audio is competitive (C=4 p95 under 350 ms), sustained realtime past one stream is
not. AMX buys the compute-bound half of the problem; it cannot buy memory bandwidth, and this
workload re-reads the Code Predictor weights 16 times per frame. Choose an AMX box for TTFA
under concurrency, and count cores and GB/s for the number of realtime streams.

---

### A 16-core VNNI-only reference

The Scaleway reference host has 16 physical Zen 5 cores, AVX-512 VNNI/BF16 and no AMX. With
two workers of eight threads, the short-bank wave reaches TTFA 185/221 ms and stream RTF 1.11
at C=4. The full mixed-length bank reaches 640/1017 ms and RTF 1.20 at the same concurrency.
The 10-minute closed-loop soak is stable (`PASS`, 267 completed requests, zero errors), but its
pooled C=4 RTF is about 1.08 and its TTFA p95 is about 815 ms after warm-up. The complete tables,
Poisson checks and dispatch census are in
[`reference-scaleway-16c-vnni.md`](reference-scaleway-16c-vnni.md).

The C=4 census is the important part: 35.6% of counted work is INT8 GEMV and another 18.6% is
solo/single-slot work; only 32.9% is INT8 VNNI matrix-matrix. This is why a stronger VNNI GEMM
alone cannot close the gap to RTF 1.0. The profile keeps activation quantization, row-sum reuse,
GEMV MR=2 and the measured pool/decoder settings, while leaving NCHUNK and MR=4 as candidates.

### oneDNN as a bounded VNNI oracle

The same Scaleway host was measured with a small oneDNN harness using the dominant CP shapes.
oneDNN selected `brg_matmul:avx512_core_vnni` and reordered weights to `BA16a64b4a`. At B=1,
the engine was within roughly ten percent on the tested cells, so there is no general GEMV
instruction deficit large enough to explain the server gap. At B>1, oneDNN was often faster,
which makes its persistent RHS layout and blocking worth studying.

The comparison is intentionally not presented as an end-to-end speedup: oneDNN receives
pre-quantized activations and pays neither the engine's dynamic quantization nor its serving
wrapper. The full cells and the decision are in [`x86-oracle.md`](x86-oracle.md). The conclusion
so far is to transfer ideas selectively, not to add oneDNN or another runtime to the engine.

### VNNI parent prepack: reusable candidate, not a profile default

`QWEN_VNNI_PREPACK=1`/`all` builds an eligible INT8 RHS layout before prefork; `cp` and `talker`
limit the work to one component. The current layout is 16 output rows interleaved in groups of
four K values and is used by batched VNNI matmat, not by GEMV. On the correct AVX-512 BF16 + VNNI
Scaleway build, CP-only prepack created 41 matrices (about 102 MB) and produced byte-identical
audio. It left C=1 at 66/73 ms TTFA p50/p95; at C=4 it changed 167/318 ms to 178/319 ms and
total RTF 1.10 to 1.13. It is therefore kept opt-in for other hosts/shapes, but remains null in
the deployment profile.

### x86 shape controls

The x86 matrix paths expose the same kind of measured, runtime-selectable tiling control used by the ARM backend, without changing the default path:

```bash
# Optional output-row chunks; 0 or an unset variable keeps the original call shape.
QWEN_X86_NCHUNK=256
QWEN_AMX_NCHUNK=256
QWEN_VNNI_NCHUNK=128
QWEN_AVX512_NCHUNK=128
```

`QWEN_AMX_NCHUNK` applies to AMX INT8/BF16, `QWEN_VNNI_NCHUNK` to INT8 VNNI, and
`QWEN_AVX512_NCHUNK` to the AVX-512 BF16 fallback. A family-specific value overrides
`QWEN_X86_NCHUNK`; values are rounded down to the kernel row tile and invalid values are ignored.
The controls are experimental and should be qualified per CPU, shape, and objective.

For the small-batch boundary, `QWEN_VNNI_TILE_M4N2=1` selects an opt-in fixed `M4xN2`
implementation for observed `B=2` calls. It follows the ARM SMMLA principle of sharing the
activation loads across a small cross-product tile, but uses AVX-512 VNNI and the existing
row-sum correction. It is not enabled by the x86 profile: on the AWS screening campaign it
improved isolated B2 cells but did not establish an end-to-end RTF win. Do not infer that an
`M8xN2` tile is better; its register pressure and server coverage need separate evidence.

AMX batch gates can also be separated when one threshold is not suitable for every datatype:

```bash
QWEN_AMX_BF16_MIN_B=4
QWEN_AMX_INT8_MIN_B=8
```

If these are absent, both continue to use the existing `QWEN_AMX_MIN_B` fallback. Use
`QWEN_NO_AMX=1` for a VNNI-only control and `QWEN_NO_VNNI=1` for an AMX-only control.

The fused INT8 QKV path can use a stricter, independent lower bound when its
three-output working set makes `B=2` unattractive:

```bash
QWEN_AMX_INT8_QKV_MIN_B=4
```

This affects only fused QKV; other INT8 projections retain the normal AMX gate.
It is an experimental shape-specific control and is not enabled by the measured
profiles.

## Why a 3D V-cache chip (Zen4/Zen5 X3D) is the best case

The single thing that gets you toward/under RTF 1.0 on x86 is **a cache that fits the CP working
set**. A Ryzen X3D part stacks **64 MB of extra L3** on one CCD (≈96 MB total on that CCD):

- int8 CP (~60 MB) **fits** in the V-cache CCD → the 16×-per-frame re-read hits L3, not DRAM.
- High desktop clocks (up to ~5.7 GHz) + bare metal (real cores, no hypervisor scatter) + AVX-512/VNNI.

This is why an X3D chip running even the *old* build already does well, and the v0.9.0 int8 + VNNI
path should push it further. **Pro tip:** pin the job to the V-cache CCD so the working set stays in
the big L3:

```bash
# find which cores share the largest L3 (the V-cache CCD, usually cores 0-7):
cat /sys/devices/system/cpu/cpu0/cache/index3/shared_cpu_list
# pin a 4-thread run to that CCD:
taskset -c 0-7 ./qwen_tts -d qwen3-tts-0.6b --text "..." --int8 -j4 -o out.wav
```

---

## Try it yourself (build + benchmark)

```bash
# 1. get it (release v0.9.0)
git clone https://github.com/gabriele-mastrapasqua/qwen3-tts.git
cd qwen3-tts && git checkout v0.9.0
sudo apt install -y build-essential libopenblas-dev      # Linux/WSL2

# 2. build for your CPU (VNNI if you have AVX-512-VNNI: Zen4+/Intel Ice Lake+)
make blas SIMD=avx512vnni        # or: make blas   (portable AVX2)
./qwen_tts --caps                # confirm: "int8 dot: VNNI ... (native)"
./qwen_tts --self-test           # kernel numeric correctness (no model needed)

# 3. get a model and benchmark
./download_model.sh --model small
bash tests/x86_bench.sh          # builds scalar/avx2/avx512vnni, prints an RTF A/B table

# 4. your own sentence (int8 is the recommended config on x86):
./qwen_tts -d qwen3-tts-0.6b --text "Your text here." --int8 -j4 -o out.wav
```

`tests/x86_bench.sh` prints a clean table: it runs scalar-bf16 vs VNNI-int8 at the same core count
(so you see the kernel win in isolation) plus a `-j4` matrix across precisions.

### What to expect

- `--caps` should show **AVX2** (or VNNI on AVX-512) + a pthread pool, not `scalar`/`SINGLE-THREAD`.
- `--int8` should be **~1.5–2× faster than bf16**. On a **bare-metal, cache-rich (X3D) chip** the
  4-thread int8/int4 numbers should be the best — possibly approaching real-time on the 0.6B model.
- Useful to send back: the full `bash tests/x86_bench.sh` table + `./qwen_tts --caps`, plus your
  CPU model (`grep -m1 'model name' /proc/cpuinfo`) and L3 layout
  (`cat /sys/devices/system/cpu/cpu0/cache/index3/shared_cpu_list`). On an X3D part, also the
  `taskset -c 0-7 ... --int8 -j4` number (pinned to the V-cache CCD).

> Note: x86 single-stream RTF is memory/cache-bound, so the *chip's cache* matters as much as the
> code. Many-core server CPUs shine at **throughput** (many concurrent requests) rather than
> single-stream latency — that's a separate, future lever (batching).
