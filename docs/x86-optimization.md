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
make blas                    # x86 default: portable AVX2 + FMA (any Haswell 2013+)
make blas SIMD=avx512         # + AVX-512 (__m512 16-wide bf16 matvec)
make blas SIMD=avx512vnni    # + AVX-512-VNNI native int8 dot (Zen4+/Intel Ice Lake+)
make blas SIMD=scalar        # no AVX2 (pre-2013 / portable fallback)
```

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

### Cross-device summary (0.6B, single-stream, best config)

| Device | SIMD + threads | Best RTF | Lever |
|---|---|---|---|
| Apple M1 | NEON + SDOT, 4-thread | **sub-1.0 (int8)** | big SLC fits the working set |
| Ryzen 7 6800H | AVX2, 4-thread, bare metal | 2.02 | `--int4` (small L3) |
| EPYC 9555P (Zen5, VM) | AVX-512-VNNI | 1.64 | int8; VM limits threading |

---

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
