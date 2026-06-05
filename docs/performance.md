# Performance

All benchmarks on Apple M1 8-core, 16 GB RAM, 4 threads.

## Summary

- **0.6B bf16**: RTF ~1.3–1.7 (full precision, reference quality)
- **0.6B `--int8`**: **RTF < 1.0 — faster than real-time — across CLI, streaming AND the HTTP server**
  (≈0.80 long / 0.90 short / 0.88 server-warm / 0.93 cloned `.qvoice`), no perceptible quality loss
- **1.7B**: RTF ~2.0–4.1 (bf16), ~1.8–2.4 with `--int8` (long text)
- Bottleneck is the Code Predictor (15 sequential autoregressive passes per frame)

## ⚡ Apple Silicon `--int8` sweet spot (sub-realtime, all delivery modes)

The headline result: on a 2020 **Apple M1** the 0.6B model with `--int8` is **faster than real-time
in every mode** — one-shot CLI, low-latency streaming, and the warm HTTP server — at near-bf16
quality. Measured 0.6B, M1 8-core, 4 threads, seed 42, speaker `ryan`:

| Mode | bf16 RTF | **`--int8` RTF** | First audio (TTFA) |
|---|---|---|---|
| CLI (short, ~4 s audio) | 1.5–1.8 | **0.90** | 0.96 s |
| CLI (long, ~14 s audio) | ~1.3 | **0.80** | — |
| **Streaming** `--stream` (short) | 1.5–1.8 | **0.89** | **0.46 s** |
| **Streaming** `--stream` (long) | ~1.3 | **0.81** | **0.50 s** |
| **HTTP server** `--serve` (warm) | ~1.3 | **0.88** | — |
| **Custom voice** `.qvoice` (streamed) | 1.34 | **0.93** | 0.47 s |

`--int8` quantizes the Talker + Code Predictor (native **SDOT** on ARM). The win comes from halving
the Code Predictor's per-frame weight traffic; longer audio amortizes prefill so RTF keeps dropping
(~0.80 on a paragraph). Streaming reaches first audio in **~0.5 s** while holding sub-1.0 RTF, and the
server (delta-prefill + embedding cache + decoder overlap) holds ~0.88 warm. Quality is validated by
ear vs bf16, including cloned `.qvoice` voices. See [Quantization](quantization.md) for the details.

## RTF Across Modes

Results from `make bench-full` (seed 42, speaker ryan):

| Config | 0.6B Short | 0.6B Long | 1.7B Short | 1.7B Long |
|--------|-----------|----------|-----------|----------|
| **CLI normal** | 1.37–1.71 | **1.29–1.32** | 4.10–4.40 | **1.97–2.11** |
| **CLI stream** | 1.30–1.31 | 1.30–1.33 | 2.59–4.01 | 2.06–2.43 |
| **Server cold** | 1.34 | — | — | — |
| **Server warm** | **1.33** | — | — | — |
| **1.7B INT8** | — | — | 3.69 | 2.15 |
| **1.7B Instruct** | — | — | 3.43 | — |

Streaming mode has **identical performance** to normal mode — the speech decoder
runs in a pipeline thread in both cases. Longer audio amortizes fixed costs (prefill).
Server mode adds warm caches, embedding cache, and decoder thread overlap on top.

## Per-Component Breakdown (0.6B, seed 42, CLI)

| Component | Short text (4.7s audio) | Long text (16.8s audio) |
|-----------|------------------------|------------------------|
| Prefill | 1,647ms | ~1,040ms |
| Talker | 24.6 ms/frame | ~22 ms/frame |
| Code Predictor | 76.3 ms/frame | ~60 ms/frame |
| Speech Decoder | overlapped (512ms drain) | overlapped |
| **Total** | **8.2s → RTF 1.74** | **~20s → ~RTF 1.4** |

The speech decoder runs in a **background thread** during generation, overlapping
most of its work with Talker+CP. Only the final "drain" (waiting for the last chunk)
adds to wall time. Prefill and per-frame costs amortize over longer audio, with
an asymptotic RTF approaching ~1.0.

## CPU vs GPU

| Hardware | 0.6B RTF (short) | 0.6B RTF (long) | Notes |
|----------|------------------|-----------------|-------|
| **This project (C, Apple M1 CPU)** | **1.39** | **1.26** | **Pure C, server warm, no GPU** |
| Python + PyTorch (Ryzen 9 7950X CPU) | 4.5–5.8 | — | Official Python, CPU-only |
| NVIDIA RTX 3090 | 0.52 | 0.68 | Python + PyTorch + FlashAttention 2 |
| NVIDIA RTX 4090 | 0.38 | 0.45 | Python + PyTorch + FlashAttention 2 |
| NVIDIA A100 | 0.28 | 0.35 | Data center GPU |
| NVIDIA H100 | 0.22 | 0.28 | Data center GPU |

> RTF = Real-Time Factor = processing_time / audio_duration. Lower is faster; <1.0 means faster than real-time.
> GPU benchmarks from [qwen3-tts.app](https://qwen3-tts.app/blog/qwen3-tts-performance-benchmarks-hardware-guide-2026).

### Key takeaways

**We're 3–4x faster than Python on CPU.** The official Python + PyTorch implementation
on a Ryzen 9 7950X (16-core Zen 4, 2022, DDR5) gets RTF 4.5–5.8. Our pure C engine on
an Apple M1 (8-core, 2020, LPDDR4X) gets RTF 1.3–1.7 — on older, slower hardware.
That's the difference between optimized C with SIMD (NEON/AVX) + BLAS and Python with
PyTorch overhead.

**GPUs get worse on long text, we get better.** GPU RTF degrades 18–31% from short
to long text (attention scales quadratically even with FlashAttention). Our RTF *improves*
7% on long text because fixed costs (prefill, speech decoder) amortize over more frames
while our per-token decode is constant-time (linear matvec, no quadratic attention).

For a CPU-only engine on a 2020 laptop, being within 2x of a consumer GPU (RTX 3090)
and 3–4x faster than the official Python CPU path is a solid result.

## Cross-device CPU benchmarks (ARM + x86)

Single-stream **0.6B**, deterministic (`--seed 42 -s ryan -l English`), each box's best
config in this repo. **Reproduce on your own machine** with the bundled A/B harness:

```bash
bash tests/x86_bench.sh            # builds scalar/AVX2/AVX-512 binaries, prints an RTF table
./qwen_tts --caps                  # what SIMD/threads your build actually uses
./qwen_tts --self-test             # kernel numeric correctness (ISA-independent, no model)
```

| Device | CPU / cache | RAM | SIMD + threads | Best 0.6B RTF | Best config |
|---|---|---|---|---|---|
| **Apple M1** (dev) | M1 8-core, large system-level cache, LPDDR4X | 16 GB | NEON + SDOT int8, GCD 4-thread | **~1.3 bf16 / sub-1.0 int8** | `--int8 -j4` |
| **Ryzen 7 6800H** (bare metal, WSL2) | Zen3+ 8C/16T, 16 MB L3, DDR5-4800 | 32 GB | AVX2 + FMA, pthread 4-thread | **2.02** | `--int4 -j4` |
| **EPYC 9555P** (Scaleway VM) | Zen5 "Turin" 64-core, 256 MB L3 (**32 MB/CCD**), AVX-512 + VNNI + BF16 | 16 GB / 4 vCPU | AVX-512-VNNI, pthread | **1.64** | `--int8 -j1` |

> EPYC 9555P cpuinfo AVX-512 flags: `avx512f avx512bw avx512vl avx512dq avx512cd avx512_vnni
> avx512_bf16 avx512vbmi avx512_vbmi2 avx512ifma avx512_bitalg avx512_vpopcntdq avx512_vp2intersect`.

### What each box teaches

- **Apple M1** is the single-stream king: its big system-level cache absorbs the Code
  Predictor's 16×-per-frame weight re-read, so `--int8` reaches/breaks RTF 1.0.
- **Ryzen 7 6800H** (16 MB L3, no AVX-512): memory-bandwidth-bound. AVX2 is only **~+6%** over
  scalar; the real lever is **fewer weight bytes → `--int4`** (multi-threaded), 3.9 → **2.02**.
  4 threads is the sweet spot; 8 regresses (memory bus saturates).
- **EPYC 9555P** (Zen5, full-width 512-bit AVX-512 + VNNI): proves the **VNNI int8 path is
  numerically correct** (`--self-test` PASS) and that the **int8 kernel stack is a real ~1.85×
  win at equal core count**:

  | EPYC 9555P, 0.6B | RTF | CP ms/f |
  |---|---|---|
  | scalar bf16 `-j1` (≈ unoptimized) | 3.04 | 164.8 |
  | **VNNI int8 `-j1` (this repo)** | **1.64** | 79.3 |
  | VNNI int8 `-j4` (VNNI on) | 1.78 | 88.7 |
  | int8 `-j4` (VNNI off → AVX2 widen) | 1.87 | 91.7 |
  | int4 `-j4` | 2.06 | 108.4 |
  | bf16 `-j4` | 1.90 | 95.1 |

  Note `-j1` (1.64) **beats** `-j4` (1.78): this was a 4-vCPU **VM**, where the hypervisor
  scatters vCPUs across different CCDs so the threads can't share one CCD's 32 MB L3.
  Threading scales on **bare metal** (M1, 6800H) but not in a multi-CCD VM slice. VNNI itself
  is ~5% over the AVX2 widen path (memory-bound caps it). int8 is the speed+quality floor;
  int4 loses here (nibble-unpack cost, working set doesn't fit the per-CCD L3).

### The one rule

Single-stream RTF is **memory/cache-bound** — the Code Predictor re-reads its weights 16× per
frame. SIMD width and thread count matter far less than (1) **fewer weight bytes**
(`--int8`/`--int4`) and (2) **a cache that fits the working set** (Apple's SLC, an X3D chip's
3D V-cache). Sub-1.0 single-stream is realistic on M1 / a desktop X3D; a many-core server CPU
is better spent on **throughput** (many concurrent requests), where its core count pays off.

## Optimization History

Starting from a baseline of **RTF ~3.5** (CLI), the following optimizations brought
performance to **RTF ~1.3–1.7** (up to 2.7x total speedup):

| Optimization | Speedup | Technique |
|---|---|---|
| Cache-line alignment (`posix_memalign(64)`) | **24%** | Aligned all BLAS/SIMD buffers and KV caches |
| Decoder thread overlap | **14-19%** | Speech decoder runs in background thread during generation |
| SIMD speech decoder | **11%** | Replaced scalar RMSNorm, RoPE, attention with NEON (ARM); AVX2 twins added for x86 (PLAN 21.3) |
| Persistent prefill buffers | **38% server** | Reuse buffers across generations (zero malloc in decode) |
| Text embedding cache | **14% server** | LRU cache for token embeddings (skip 2 matvec per cached token) |
| Batched VQ projection | minor | BLAS sgemm instead of per-frame scalar matvec |
| Pre-allocated sampling buffers | minor | Zero per-token malloc in generation loop |
| Top-k quickselect | **4× sampling** | O(n) quickselect replaces O(kn) selection sort |
| Streaming pipeline parallelism | **RTF 2.0→1.4** | Decoder thread runs in streaming mode too |
| SIMD element-wise ops | minor | NEON (ARM) + AVX2 (x86) for add/mul/scale inplace |
| Vectorized bf16→f32 codec embeds | **~6% on 1.7B** | Bulk SIMD conversion replaces scalar loops in embedding lookups |
| Auto-scaled thread count | minor | ncpus/2 (min 4) adapts to larger machines |

> **Platform reality (updated 2026-06-05):** the optimization table above is the Apple-M1 history,
> but the engine is **no longer Apple-only**. The hot matvec/attention kernels now have **NEON
> (+SDOT) and AVX2 twins** plus an **AVX-512/VNNI** int8 path (runtime-ISA-guarded), and decode
> threading runs on a **cross-OS pool** (GCD on macOS, pthread on Linux/Windows). x86 is validated
> — AVX2 + threading + int4 on a Ryzen 7 6800H, AVX-512/VNNI on an EPYC 9555P (Zen5); see
> "Cross-device CPU benchmarks" above. The remaining caveat is **physics, not missing code**: decode
> is memory/cache-bound, so SIMD/threading give modest gains and the chip's cache dominates
> single-stream RTF. Measure your box with `tests/x86_bench.sh`. Details in PLAN.md Phase 21.

## Running Benchmarks

```bash
make bench         # Quick: short+long text, normal+stream, both models
make bench-full    # Full: + server cold/warm, instruct, INT8, .qvoice
```

Auto-skips missing models and `.qvoice` files. Output includes RTF, audio duration,
and wall time for each configuration. Logs and WAVs saved in `/tmp/qwen_tts_bench/`.

## Key Architectural Decisions

- **SIMD-optimized kernels** — NEON (+ native SDOT int8) on ARM and **AVX2 + AVX-512/VNNI on x86**
  for the BF16/INT8 matrix-vector ops, with a scalar fallback and a runtime ISA guard (validated on
  Ryzen 6800H + EPYC 9555P, PLAN.md 21.3).
- **Cache-line aligned buffers** — 64B `posix_memalign` for optimal BLAS/SIMD throughput
- **Multi-threaded decode** — a **cross-OS thread pool**: GCD (`dispatch_apply`) on macOS, a
  persistent pthread pool on Linux/Windows (PLAN.md 21.2). Threading scales on bare metal; in a
  multi-CCD VM the hypervisor's vCPU placement can limit it. Prefill uses BLAS (`cblas_sgemm`).
- **Memory-mapped weights** — BF16 safetensors mmap'd directly, near-instant loading
- **Pipeline parallelism** — Speech decoder runs in background thread during Talker+CP generation
