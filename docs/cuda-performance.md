# CUDA performance (NVIDIA) — RTF & throughput

The CUDA backend runs the whole per-frame pipeline **GPU-resident**: the Talker and Code
Predictor are fused steps (weights + KV + activations stay on the device, one sync per step,
captured into CUDA graphs), the ConvNet speech decoder runs on-device, and multi-request serving
batches the fused steps (matvec → matmat, each weight row read once for all B sequences).

The build is **multi-arch** (`sm_80/86/89/120` + PTX) so one binary runs on Ampere, Ada, and
Blackwell — **old and new NVIDIA GPUs alike** (RTX 30-, 40-, 50-series, workstation/datacenter).

## Reference GPU

All numbers below were measured on a **mainstream ~270 GB/s NVIDIA GPU (RTX 4060-class)** running
the **1.7B** model. Single-token TTS decode is **memory-bandwidth-bound** on weight reads, so the
figures scale (to first order) with a card's memory bandwidth — see the estimate table.

## Latency — single stream (RTF, 1.7B)

RTF = processing time ÷ audio duration; **< 1.0 = faster than real time**.

| Config | RTF | Note |
|--------|-----|------|
| Naive per-op GPU offload | 1.47 | H2D/D2H per matvec — transfer-bound, never wins |
| Resident fused (int8) | **0.55** | fused Talker+CP + resident decoder + CUDA graphs |
| Resident fused (`--quant-mixed`) | **0.44** | int4 Talker + int8 CP (best) |

**2.7–3.3× faster than the per-op baseline, comfortably sub-real-time.** All resident-decoder
changes are bit-identical to the CPU decoder (mel-corr 1.0); `--quant-mixed` is ear-validated.

Trajectory of the wins: `1.47 → 0.86` (fused Talker+CP) `→ 0.62` (resident decoder) `→ 0.55`
(cuBLAS pointwise + im2col/gemm convs + CUDA graphs) `→ 0.44` (mixed int4/int8 quant).

### Expected RTF by GPU (bandwidth-scaled estimate)

Decode is bandwidth-bound, so RTF roughly scales with memory bandwidth. Rough estimates for
`--quant-mixed` 1.7B (measured point in **bold**; others extrapolated, real numbers vary with
kernel efficiency and clocks):

| GPU (class) | Mem BW | Est. RTF (1.7B mixed) |
|-------------|-------:|----------------------:|
| RTX 3050 / 4060-class | ~270 GB/s | **0.44** (measured) |
| RTX 3060 | ~360 GB/s | ~0.33 |
| RTX 4070 | ~500 GB/s | ~0.24 |
| RTX 3090 / 4080 | ~700–940 GB/s | ~0.13–0.17 |
| RTX 4090 | ~1000 GB/s | ~0.12 |

The 0.6B model is proportionally faster (smaller Talker).

> ⚠️ **The scaling has a floor — measured on an A100 (2026-07-11).** A cloud **A100-SXM4-40GB**
> (HBM2, ~1.5 TB/s — 5–6× the reference card) measured **0.50–0.55**, not the ~0.1 the table
> would extrapolate: past the point where weights stream fast enough, single-stream decode
> becomes **kernel-launch-latency-bound** (hundreds of small dependent launches per frame,
> costlier on virtualized cloud GPUs). Big-bandwidth cards pay off in **batch throughput**,
> not single-stream latency — same lesson as Apple-silicon Metal.

## Measured: datacenter A100 (Verda cloud, A100-SXM4-40GB, 2026-07-11)

Full recipe (`QWEN_CUDA_FUSED_TALKER=1 QWEN_CUDA_CONVDEC=1`), seed-pinned, greedy:

| Config | RTF |
|--------|-----|
| 0.6B bf16 / int4 | **0.39** |
| 1.7B `--quant-mixed` | 0.55 |
| 1.7B `--quant-mixed` + `QWEN_CUDA_DP4A=1` | **0.50** |

- **dp4a (int4 weights × int8-quantized activations, integer `__dp4a` dots)** is a measured win on
  real NVIDIA: **1.7B Talker 8.4 → 5.6 ms/f (−33%)**, 0.6B Talker −19% / CP −16%, ear-validated —
  **now the DEFAULT for int4/quant-mixed** (`QWEN_CUDA_DP4A=0` reverts to the f32-act kernel).
- Without `QWEN_CUDA_CONVDEC=1` the speech decoder runs on the host CPU — on a weak cloud host
  that alone was the difference between RTF 0.94 and 0.39. **Always set both env vars.**
- **Batch throughput (B=8, 1.7B quant-mixed):** 8 concurrent requests = 30 s wall for 63.5 s of
  audio → **aggregate RTF 0.47, ~2.1× throughput**; per-request RTF in batch mode 1.27 (the known
  latency/throughput trade).

## Throughput — server batching (`--serve --batch-size N`)

With concurrent requests, the per-request matvecs become a **matmat** (each weight row read once
for all B sequences) — the compute-bound regime where the GPU shines. Enabled with
`QWEN_CUDA_BATCH=1`.

- **Batch independence (correctness):** a request's audio is **byte-identical** whether it runs
  solo or inside a batch of 8 (md5 match) — batching never changes or degrades any output.
- **Per-step (Talker+CP) throughput:** **3.35× at B=8** (Talker 4.1×, CP 2.7×) vs one sequence.
- **End-to-end server throughput:** **~3× at B=8** — both WAV and streaming. The speech decoder
  is amortized per frame (interleaved with generation), so a whole batch finishing together no
  longer serializes into a decode burst. Remaining gap to the 3.35× per-step ceiling is the
  non-batched prefill; longer clips trend toward the ceiling. Higher-bandwidth GPUs sustain
  larger effective batches.

Streaming (`/v1/tts/stream`) batches too: concurrent streams share the batched fused steps and
each gets its own incremental PCM chunks. WAV requests use the same incremental decoder internally
(bit-identical to the seam-free full decode, mel-corr 1.0) so they reach the same throughput.

## CUDA streaming server — 2026-09-16 (RTX A6000, 0.6B)

**Maturity: implemented and runtime-verified, NOT performance-qualified. Opt-in, not a default.**

What that means precisely, so nobody has to infer it:

| claim | status |
| --- | --- |
| compiles and runs (`--backend cuda --serve`) | **runtime verified** — A100 2026-09-15, A6000 2026-09-16 |
| batched steps produce the same codes as single-stream | **verified** — `--gpu-batch-bench` reads `0.00e+00` on three checks |
| audio is correct under concurrent load | **ear-validated** at C4 |
| meets a serving KPI target | **NOT established.** Every soak run to date is a 3-minute screen and every one reports `SOAK RESULT: FAIL — per-class KPI drift`. No 30-minute qualification has been run on any GPU. |
| is a default | **no.** It requires `--backend cuda` plus the resident-path env flags below. |

Treat the concurrency table below as a measured screen on one box, not as a supported envelope.

**Scope: this section is CUDA-only.** Everything below was measured with `--backend cuda` and the
resident GPU paths on. None of it describes or changes the CPU serving path, whose behaviour,
tuning and numbers live in `docs/server-batching.md` and `docs/serving-operations.md`. The two
paths are sized by different things — the CPU server by cores and cache, this one by GPU memory
bandwidth — and their numbers are not comparable.

### Concurrency the streaming server holds

RTX A6000 (768 GB/s) behind a 10-core EPYC 7402, 0.6B, `--precision default`, all resident paths
on, 3-minute closed-loop soaks:

| | C2 | C4 | C6 | C8 |
| --- | --- | --- | --- | --- |
| RTF p50 | 0.43 | **0.56** | 0.90 | 1.11 |
| stall rate @1000 ms | 0% | **4%** | 35% | 63% |
| safe_play_start p50/p95 | 0.69 / 1.6 s | 1.02 / 1.95 s | 1.65 / 4.0 s | 2.76 / 7.9 s |
| requests / 3 min | 57 | 91 | 93 | 101 |

**C4 comfortable, C6 usable.** Throughput keeps climbing to C8, so past C6 the limit is playback
quality rather than capacity. The knee scales with GPU memory bandwidth, not with core count: the
code predictor reads ~2.24 GB of weights per audio frame regardless of how many requests share
them, so the concurrency a card holds tracks its bandwidth almost linearly.

### Two configuration traps that silently cost most of the server

1. **`--precision default` is mandatory.** The backend seam is bf16-only, so `--int8` sends the
   *seam* back to the CPU. `tests/serve_soak.py` defaults `--precision` to int8, which produces a
   healthy-looking run with the GPU near 0%.
2. **Size the engine pool.** `--prefork-threads` defaults to `cpus/n` and that default is correct;
   passing `1` explicitly — as `tests/serve_soak.py` does — sizes the whole server pool to one
   thread. Measured at C4: RTF p50 0.68 → 0.58, stall@1000 11% → 4%, 82 → 91 requests, with the
   GPU-side cost unchanged. Four threads captures it, eight adds nothing.

Note that the server is deterministic for a **fixed** thread count and not across thread counts:
the reduction order follows the pool size. Hold it fixed for any A/B.

### What the batched path does on the GPU

`QWEN_CUDA_BATCH=1` steps every in-flight request together, so each weight row is read once for
all lanes. Since 2026-09-16 the batched bodies also:

- **replay from CUDA graphs**, cached per effective lane count. Only the single-stream bodies had
  graphs before; the server runs the batched ones, which were issuing ~1950 launches per frame.
- **compact idle lanes** (`QWEN_CUDA_BATCH_COMPACT=1`) so a server sized for 8 lanes serving 2
  does not pay for the empty ones. Worth 17% of code-predictor time at low occupancy. The KV stays
  addressed by slot, so a compacted lane still reads its own request's history.
- **run the code-predictor head on the GPU** — final norm, lm_head and argmax. The CPU fallback
  re-reads a 4 MB lm_head once *per lane*, fifteen times per frame; the GPU reads each weight row
  once for all lanes. 11.6 → 1.35 ms/frame, and codes are bit-identical.
- **size the per-lane accumulators to the actual lane count**, which is worth 34% at 4 lanes.

Measured effect of the 2026-09-16 work, `--gpu-batch-bench 8`:

| | before | after |
| --- | --- | --- |
| Talker step | 7.56 ms/frame | **4.75** (−37%) |
| Code predictor | 14.35 ms/frame | **10.44** (−27%) |
| aggregate GAIN | 4.73× | **6.82×** |

Every one of those is bit-identical: a fixed-seed temperature-0 streaming request returns the same
md5 before and after, and the three batched self-tests stay at `0.00e+00`.

### Batch size

`--batch-size 8` is the measured optimum and `QB_MAX` is 16. Wider batches are allowed but do not
pay: per-stream code-predictor cost is 2.05 ms at B=4, **1.34 at B=8**, 1.43 at B=12 and 1.47 at
B=16. Past eight lanes the kernel stops being weight-bound — the weights are read once regardless
— and becomes bound by per-lane registers.

### int8 on the GPU

int8 weights work on the resident path and are selected by `--int8` (the seam caveat above applies
only to the seam, not to the resident kernels). They help the **single-request** path, where the
kernel is weight-bound: Talker 5.23 → 4.64 ms/frame, code predictor 7.70 → 6.00. On the batched
path the gain is small — about 5% of RTF at C4 — because halving the bytes does not halve the time
of a kernel that is latency-bound rather than bandwidth-bound.

### The exact configuration behind the numbers above

```bash
QWEN_CUDA_FUSED_TALKER=1 QWEN_CUDA_DECODER=1 QWEN_CUDA_CONVDEC=1 \
QWEN_CUDA_BATCH=1 QWEN_CUDA_BATCH_COMPACT=1 \
  ./qwen_tts -d qwen3-tts-0.6b --backend cuda --serve 8000 \
  --batch-size 8 --prefork 1 --prefork-threads 8
```

All five resident-path flags are on, `--precision` is left at its default (never `--int8` on the
seam), and the engine pool is sized explicitly. Omitting any of them does not reproduce the table.

### Verifying a GPU build

```bash
./qwen_tts -d qwen3-tts-0.6b --backend cuda --gpu-selftest        # seam vs CPU
./qwen_tts -d qwen3-tts-0.6b --backend cuda --gpu-batch-bench 8   # batched exactness + throughput
```

`--gpu-batch-bench` prints three exactness checks, all of which must read `0.00e+00`: batched
Talker against single-stream, batched code predictor against single-stream, and **partial
occupancy against independent per-lane references** — the last one is the only check that
exercises lane compaction, since the first two step every lane. With `--int4` the first two read
non-zero by design: the single-stream q4 path uses the dp4a kernel, which quantises activations to
int8, so the two sides are deliberately different arithmetic (`QWEN_CUDA_DP4A=0` makes them agree).

## How to run

Build (pick your arch, or use the default multi-arch):

```bash
make cuda                    # multi-arch (sm_80/86/89/120 + PTX)
```

The CUDA toolkit prefix is auto-detected: `nvcc` on `PATH` (covers conda / environment-modules /
custom prefixes) → `/usr/local/cuda` (NVIDIA `.run`/`.deb` installers) → `/opt/cuda` (Arch Linux's
`cuda` package). If yours lives elsewhere, pass it explicitly:

```bash
make cuda CUDA_HOME=/path/to/cuda
```

Single stream (lowest latency):

```bash
QWEN_CUDA_FUSED_TALKER=1 QWEN_CUDA_CONVDEC=1 \
  ./qwen_tts -d qwen3-tts-1.7b --backend cuda --quant-mixed \
  --text "…" -s ryan -l English -o out.wav
```

Server with GPU batching (highest throughput):

```bash
QWEN_CUDA_FUSED_TALKER=1 QWEN_CUDA_CONVDEC=1 QWEN_CUDA_BATCH=1 \
  ./qwen_tts -d qwen3-tts-1.7b --backend cuda --quant-mixed \
  --serve 8000 --batch-size 8
```

Flags / env:
- `--backend cuda` — select the CUDA backend.
- `--int8` — int8 weights (Talker + CP). `--quant-mixed` — int4 Talker + int8 CP (fastest, same quality).
- `QWEN_CUDA_FUSED_TALKER=1` — GPU-resident fused Talker + Code Predictor.
- `QWEN_CUDA_DECODER=1` — speech-decoder pointwise convolutions through cuBLAS on the device.
- `QWEN_CUDA_CONVDEC=1` — GPU-resident ConvNet speech decoder (the whole conv stack, not just the
  pointwise convs above; the two are independent and the measurements in this document have both on).
- `QWEN_CUDA_BATCH=1` — GPU-batched fused steps for the server (`--batch-size N`, N ≤ 16; 8 is the
  measured optimum).
- `QWEN_CUDA_BATCH_COMPACT=1` — pack the stepping lanes together so idle slots cost nothing.
  Default off; verified exact against independent per-lane references.
- `QWEN_CUDA_BATCH_GRAPH=0` — replay the batched bodies with plain launches instead of CUDA
  graphs. Default on, bit-identical.
- `QWEN_CUDA_CP_HEAD=0` — put the code-predictor head (norm + lm_head + argmax) back on the CPU.
  Default on; the GPU path produces identical codes.
- `QWEN_CP_PROFILE=1` — break the batched code-predictor frame into seed / step / head and report
  every 500 frames.
- `QWEN_CUDA_MM_UNROLL`, `QWEN_CUDA_MM_TPB` — weight loads in flight (default 4) and block size
  (default 64) for the batched matmats. Both swept on an A6000; exposed to re-sweep elsewhere.
- `QWEN_CUDA_CUBLAS=1`, `QWEN_CUDA_TC=1` — route the wide batched matmats through cuBLAS, or
  through the portable wmma tensor-core GEMM. **Both default off and both change what the model
  generates**: tensor cores require bf16 activations, and at a fixed seed that moved the end of
  speech and produced a 25% shorter utterance. The code predictor is excluded from both by
  construction, since it picks the codes through an argmax.
- `QWEN_CUDA_TF32`, `QWEN_CUDA_VERBOSE` — TF32 in the seam's GEMMs; batched-state diagnostics.
- dp4a int4 matvec (integer `__dp4a`, activation quantized to int8 per 32-block, even/odd-
  deinterleaved to match q4_0 packing): **ON by default** since the A100 validation (−33% Talker
  ms/f on 1.7B, ear-validated). `QWEN_CUDA_DP4A=0` reverts to the f32-activation kernel
  (trajectory forks between the two — act-quant numerics, benign).

## Notes

- All GPU paths are validated against the CPU reference: the resident decoder and batched steps
  are bit-identical per sequence; `--quant-mixed` was ear-validated.
- CUDA vs CPU output differs benignly at the sampling level (matvec fp-order) — both are correct,
  each self-consistent; use RTF + mel-correlation to compare, not md5, across CPU↔GPU.
