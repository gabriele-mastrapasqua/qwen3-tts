---
title: "Adding GPU backends to a pure-C TTS engine: Metal, CUDA, and the rented-Mac trick"
published: false
description: "How we bolted opt-in Apple Metal and NVIDIA CUDA backends onto a pure-C Qwen3-TTS engine — resident fused pipelines, server request-batching, and measuring it all on a Mac mini M2 rented by the hour. Plus the two 'obvious' optimizations we killed with data."
tags: c, cuda, metal, machinelearning
---

*Part of [qwen3-tts](https://github.com/gabriele-mastrapasqua/qwen3-tts) — a pure C inference engine for Qwen3-TTS.*

## TL;DR

The engine is pure C and **CPU by default**. We added two **opt-in** GPU backends that leave the CPU path untouched:

- **Apple Metal** (`make metal`) — 0.6B model **RTF 1.5 → 0.60 on an M1**, **0.36 on an M2 Pro**; streaming first-audio in **314 ms**.
- **NVIDIA CUDA** (`make cuda`) — 1.7B model **RTF 0.44** on a ~270 GB/s GPU (RTX 4060-class), scaling to ~0.12 on a 4090.
- **Server request-batching** on both (plus CPU): **~2.8× throughput on Metal, ~3.35× on CUDA** — serving N concurrent users in roughly the time of one.

And two optimizations that *looked* obvious and that we **measured and threw away** (ICB and MMA). More on that at the end — it's the most useful part.

RTF = processing time ÷ audio duration. **< 1.0 = faster than real time.**

---

## The one architectural idea: keep everything resident

The naive way to "use the GPU" is to offload one operation at a time — send the activation over, run a matmul, copy the result back. For a model that runs **16 sequential Code-Predictor passes per audio frame** plus a 28-layer Talker step per token, that's death by a thousand PCIe/round-trip cuts. We measured it: per-op offload was *slower* than the CPU.

The fix is the same on both backends: **make the whole step resident.** Weights, KV cache, and activations live on the device. A full decode step is encoded into **one command buffer / one kernel graph**, committed once, waited on once. The CPU orchestrates; it never babysits individual matmuls.

- On **CUDA** that's resident weights + CUDA Graphs + cuBLAS for the pointwise convs.
- On **Metal** that's MTLBuffers cached by pointer + one `MTLCommandBuffer` per step + simdgroup matvec kernels.

The biggest single win on Metal came from the **Code Predictor**. It was *sync-round-trip-bound* — 16 waits per frame. Moving the whole 16-pass RVQ loop (embed → 5 transformer layers → argmax, ×16) onto the device as **one command buffer with one sync/frame** took 0.6B from RTF 1.36 → 0.89. Direct kernel fusions (per-head RMSNorm+RoPE, residual-add+norm) took it to 0.72, and int4 weights to **0.60** — the practical floor on an M1, where the step is dispatch-bound.

---

## The rented-Mac trick (and why it works)

We develop on an M1. To see what an M2 does, we rented a **Scaleway Mac mini M2 Pro by the hour**. Two things made this painless:

**1. Metal shaders compile at runtime.** The MSL kernels are embedded as a string and compiled by the local Metal driver at startup (`newLibraryWithSource`). Consequence: a binary **built on an M1 drives an M2's GPU at full speed** — no rebuild needed, because the shaders are recompiled for whatever GPU is present. Only the CPU SIMD paths are baked in at build time.

**2. One curl to bootstrap a bare box.** A fresh macOS box has `curl` but not much else. A single script installs the Command Line Tools headlessly, clones the repo, pulls the models from HuggingFace, builds natively, and runs the benchmark:

```bash
curl -fsSL https://raw.githubusercontent.com/.../bootstrap_m2.sh | bash
```

That gave us the first **cross-backend RTF matrix on real silicon** — CPU vs Metal (M1 & M2) vs CUDA (a real NVIDIA GPU, not a datacenter part). A gotcha worth writing down: **macOS has no `timeout` and no `setsid`** (both GNU-only). If your bench scripts wrap `curl` in `timeout`, they silently no-op on a Mac.

Measured on the M2 Pro (0.6B, Metal, int8):

| Mode | RTF | First audio (TTFA) |
|---|---|---|
| CLI / warm server (single) | **0.36–0.39** | — |
| Streaming (single client) | 0.36 | **314 ms** |

int8 is the sweet spot on Apple Silicon: it's bandwidth-rich, so int4's nibble-unpacking doesn't pay off (that's the x86 lever).

---

## Batching is a throughput lever, not a speed-up

This one trips people up, so it's worth being blunt: **batching does not make a single request faster.** In batch mode each step does the work of B slots, so per-request RTF *rises*. What you buy is **throughput** — you serve B concurrent users in roughly the wall-clock of one, because each weight row is read once (from DRAM) and reused for all B (matvec → matmat).

We wired continuous request-batching (`--serve --batch-size N`) into all three backends. On the M2 Pro, 0.6B:

| Concurrent requests | Wall time | Per-request RTF | Throughput speed-up |
|---|---|---|---|
| 1 | 19.5 s | 1.01 | — |
| 2 | 21.9 s | 1.13 | **1.78×** |
| 4 | 27.7 s | 1.44 | **2.81×** |

So the honest per-request cost: RTF climbs from **1.01** (one request) to **1.44** (four) as the batch fills — the price of the B-wide work per step. What stays cheap is **first audio**: streaming TTFA is **314 ms** on the 0.6B and **517 ms** on the 1.7B *even under a full batch*, so it still feels responsive. You trade a lone request's **0.36** single-stream RTF for **~2.8× aggregate throughput**.

4 requests served in 27.7 s instead of ~78 s serial. The 1.7B model scales identically (2.82× at B=4). CUDA does **3.35×** at B=8. And crucially, the **batch output is bit-identical to single-stream** — batching never changes what a user hears.

### The bit-identity bug that taught us something

Our first batched Metal matvec accumulated **scalar** (element by element). The single-stream kernel accumulated with **`float4` dot products**. Same math, different floating-point order → a ~1e-2 difference per step. In isolation this is benign (the argmax still matches). But TTS has a **feedback loop**: this step's hidden state → sampled token → next step's embedding. That tiny difference compounded, flipped a few token choices, and the batched audio **diverged completely** from single-stream (mel-corr 0.44).

The fix was three lines: vectorize the batched matvec to `float4` so it matches the reference kernel's FP order **exactly**. Result: bit-identical (RMS-rel 0.000), and the batch server produced mel-corr **1.00000** vs single-stream.

**Lesson:** a numerically "close enough" kernel is not close enough inside a feedback loop. Match the reference's accumulation order.

---

## The two optimizations we killed with data

The most valuable engineering this session was *not shipping* two things that looked obviously good.

**ICB (Indirect Command Buffers) — the "CUDA-graphs for Metal" idea.** Pre-encode the step once, replay it. Before writing it, we profiled: the encode was only **12%** of the Talker step; the other 88% was GPU execution. ICB removes the 12% at best (~6–8% overall) for a large, invasive rewrite. And the Code Predictor — not the Talker — dominates the frame anyway. **Not built.**

**MMA matmat (simdgroup_float8x8) for batched matvecs.** The matrix units are ~4.6× faster than a simdgroup matvec on a big B=32 tile. But at our batch sizes (B≤8) the 8×8 tile is half-empty and the kernel underutilizes the GPU. Measured on the M2: **better scaling ratio (3.27× vs 2.81×) but a 2× worse baseline** → net loss. It only wins at large B. **Kept opt-in, off by default.**

Both were killed by a five-minute measurement instead of a two-day build. That's the whole point.

---

## Takeaways

1. **Resident-everything** beats per-op offload for autoregressive TTS — the round-trips, not the matmuls, are the enemy.
2. **Runtime-compiled Metal shaders** make cross-device testing trivial: build once, run on any Apple GPU.
3. **Batching = throughput, not latency.** Say so in your docs before someone benchmarks a single request and files a bug.
4. Inside a feedback loop, **match the reference kernel's FP order** or watch a "correct" kernel diverge.
5. **Measure before you build.** ICB and MMA both looked like free wins; the profiler said otherwise.

Full numbers, per-platform: [`docs/hardware-testing.md`](https://github.com/gabriele-mastrapasqua/qwen3-tts) (Metal) and [`docs/cuda-performance.md`](https://github.com/gabriele-mastrapasqua/qwen3-tts) (CUDA). It's all opt-in — the CPU path still ships as the default, pure C, zero heavy deps.
