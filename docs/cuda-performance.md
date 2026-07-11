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

## How to run

Build (pick your arch, or use the default multi-arch):

```bash
make cuda                    # multi-arch (sm_80/86/89/120 + PTX)
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
- `QWEN_CUDA_CONVDEC=1` — GPU-resident ConvNet speech decoder.
- `QWEN_CUDA_BATCH=1` — GPU-batched fused steps for the server (`--batch-size N`, N ≤ 8).
- dp4a int4 matvec (integer `__dp4a`, activation quantized to int8 per 32-block, even/odd-
  deinterleaved to match q4_0 packing): **ON by default** since the A100 validation (−33% Talker
  ms/f on 1.7B, ear-validated). `QWEN_CUDA_DP4A=0` reverts to the f32-activation kernel
  (trajectory forks between the two — act-quant numerics, benign).

## Notes

- All GPU paths are validated against the CPU reference: the resident decoder and batched steps
  are bit-identical per sequence; `--quant-mixed` was ear-validated.
- CUDA vs CPU output differs benignly at the sampling level (matvec fp-order) — both are correct,
  each self-consistent; use RTF + mel-correlation to compare, not md5, across CPU↔GPU.
