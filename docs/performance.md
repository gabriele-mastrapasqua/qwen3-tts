# Performance

All benchmarks on Apple M1 8-core, 16 GB RAM, 4 threads.

## Summary

- **0.6B**: RTF ~1.3–1.7 depending on audio length and mode
- **1.7B**: RTF ~3.0–4.3 (BF16), ~2.5–3.6 with `--int8`
- Bottleneck is the Code Predictor (15 sequential autoregressive passes per frame)

## RTF Across Modes (0.6B)

|  | Short text (~5–8s audio) | Long text (~16s audio) |
|---|---|---|
| **CLI** | RTF 1.4–1.7 | ~RTF 1.3 |
| **CLI `--stream`** | RTF 1.4–1.7 | ~RTF 1.3 |
| **Server (cold)** | RTF 1.50 | RTF 1.28 |
| **Server (warm)** | RTF 1.39 | **RTF 1.26** |

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

## Optimization History

Starting from a baseline of **RTF ~3.5** (CLI), the following optimizations brought
performance to **RTF ~1.3–1.7** (up to 2.7x total speedup):

| Optimization | Speedup | Technique |
|---|---|---|
| Cache-line alignment (`posix_memalign(64)`) | **24%** | Aligned all BLAS/SIMD buffers and KV caches |
| Decoder thread overlap | **14-19%** | Speech decoder runs in background thread during generation |
| SIMD speech decoder | **11%** | Replaced scalar RMSNorm, RoPE, attention with NEON/AVX |
| Persistent prefill buffers | **38% server** | Reuse buffers across generations (zero malloc in decode) |
| Text embedding cache | **14% server** | LRU cache for token embeddings (skip 2 matvec per cached token) |
| Batched VQ projection | minor | BLAS sgemm instead of per-frame scalar matvec |
| Pre-allocated sampling buffers | minor | Zero per-token malloc in generation loop |
| Top-k quickselect | **4× sampling** | O(n) quickselect replaces O(kn) selection sort |
| Streaming pipeline parallelism | **RTF 2.0→1.4** | Decoder thread runs in streaming mode too |

All optimizations are cross-platform (POSIX standard `posix_memalign`, conditional NEON/AVX).
See [blog/optimization-notes.md](../blog/optimization-notes.md) for the full story.

## Key Architectural Decisions

- **SIMD-optimized kernels** — NEON on ARM, AVX on x86 for BF16/INT8 matrix-vector ops
- **Cache-line aligned buffers** — 64B `posix_memalign` for optimal BLAS/SIMD throughput
- **Multi-threaded inference** — GCD (`dispatch_apply`) on macOS, pthreads on Linux
- **Memory-mapped weights** — BF16 safetensors mmap'd directly, near-instant loading
- **Pipeline parallelism** — Speech decoder runs in background thread during Talker+CP generation
