# Weight Quantization

The `--int8` and `--int4` flags quantize Talker and Code Predictor (CP) weights at load time,
reducing memory usage and (for INT8) improving speed.

> **Updated 2026-06-03 — int8 now helps BOTH models.** The older claim that int8 had "no effect on
> 0.6B" was **wrong**: it only quantized the Talker, and the 0.6B Talker (hidden=1024) is too small
> to benefit. Once CP quantization was enabled (the CP is hidden=1024 and the bottleneck on **both**
> models), `--int8` wins big on **both**. With native int8 SDOT on Apple Silicon the 0.6B model goes
> **sub-realtime (RTF < 1.0) in every mode**: CLI ~0.90 short / **~0.80 long**, streaming ~0.81–0.89
> (first audio ~0.5 s), HTTP server warm ~0.88, cloned `.qvoice` ~0.93 — at near-bf16 quality. On 1.7B:
> **RTF 2.66 → 1.79 (−33%)**, Talker −23%, CP −29%. Full table in [Performance](performance.md).

> **x86 (updated 2026-06-05):** the int8 matvec now has **AVX2 + AVX-512/VNNI** twins (native
> `_mm512_dpbusd_epi32`) and decode runs on a cross-OS pthread pool — validated on a Ryzen 7 6800H
> (AVX2) and an EPYC 9555P / Zen5 (AVX-512/VNNI), where the int8 kernel is a ~1.85× win at equal core
> count. x86 single-stream RTF is memory/cache-bound (so it won't reach Apple's sub-1.0 without a
> cache-rich chip), but `--int8`/`--int4` are the right levers there too. Toggle SDOT/VNNI off with
> `QWEN_NO_SDOT=1` / `QWEN_NO_VNNI=1`. Measure your box: `bash tests/x86_bench.sh`. See PLAN.md 21.3.

## INT8 (Recommended on Apple Silicon, both models)

```bash
./qwen_tts -d qwen3-tts-1.7b --text "Hello world" --int8 -o hello.wav   # 1.7B: Talker+CP
./qwen_tts -d qwen3-tts-0.6b --text "Hello world" --int8 -o hello.wav   # 0.6B: CP (now a real win)
```

- Talker −23% (1.7B) + CP −29% (both models) with SDOT — reduced memory bandwidth + native int8 dot
- Good audio quality — minimal perceptual difference from BF16 (validated by ear, preset + custom voice)
- Halves Talker RAM usage on 1.7B (2.8 GB → 1.4 GB)
- Works with all features: server, streaming, custom voices (`.qvoice` re-quantized after override), instruct

## INT4 (Q4_0 — the fastest lever on Apple Silicon)

```bash
./qwen_tts -d qwen3-tts-0.6b --text "Hello world" --int4 -o hello.wav
```

- Q4_0 format: 32 weights per block, **fp16 per-block scale → 18 bytes/block** (llama.cpp layout;
  was 20 B with an f32 scale — the fp16 scale cut int4 weight traffic another 10% with negligible
  quality drift, teacher-forced ladder −0.8pp).
- On ARM the kernel is **SDOT-native** (int8-quantized activations, no per-nibble unpack tax);
  on AVX-512 x86 it's the VNNI v3 throughput kernel; batched ARM rides a **q4-SMMLA GEMM** (i8mm).
- Smallest memory footprint (¼ of bf16) — also the pick when RAM is tight.
- Quality: a touch more aggressive than int8 (per-block-32 scales are coarse for the CP's late
  residuals) — int8 stays the quality reference; int4 is ear-validated fine on 0.6B.
- **v0.16.0 — weighted-LSQ scales**: the load-time quantizer now picks each block scale by
  closed-form weighted least-squares (signed-max→-8 + LSQ rescale, w=v²) instead of naive
  absmax RTN. Same layout, same kernels, same bytes, same speed — measurably better weights:
  Talker word accuracy (teacher-forced code0) **83.9% → 90.9%**, and the 1.7B int4 duration
  stretch vs bf16 gold drops from **+71% to +22%** on the A/B sentence. Applies to every
  `--int4`/`--quant-mixed` config on every ISA (Metal/CUDA included — single quantizer).
  `QWEN_Q4_NAIVE=1` restores the old quantizer for A/B. Full study: `docs/quant-sub4.md`.

## Comparison (Apple M1 8-core, 16 GB, `-j4`, 2026-07 state)

| Config | 0.6B best RTF | 1.7B best RTF | Talker RAM (1.7B) |
|--------|--------------|---------------|-------------------|
| BF16 (default) | 1.3–1.8 | ~2.0 | 2.8 GB (mmap) |
| **INT8** | **0.69** | 1.79 | 1.4 GB |
| **INT4** | **0.51** ⚡ | 1.58 | 0.7 GB |
| quant-mixed (int4 Talker + int8 CP) | — | **~1.53** | ~1.0 GB |

## Recommendation — per platform (all measured on real silicon, 2026-07-11)

| Platform | Pick | Why (measured) |
|---|---|---|
| **Apple Silicon (M1+)** | `--int4` for speed, `--int8` for max quality | int4-SDOT is the fastest lever (0.6B: M1 **0.51**, **M4 0.32**); 1.7B best = `--quant-mixed` (M1 ~1.53, **M4 0.57**) |
| **x86 AVX-512/VNNI** (Zen4+, Ice Lake+) | **`--int8`** — for 1.7B too | EPYC Turin: int8 0.96 vs int4 1.05; **1.7B pure int8 1.22 beats quant-mixed 1.30** (quant-mixed is an Apple-silicon config). Build with `make blas SIMD=avx512vnni` |
| **x86 AVX2-only** (Zen3, small L3) | `--int4` multi-threaded | memory-starved: fewer weight bytes wins (Ryzen 6800H 3.9→2.02) |
| **ARM server** (Graviton3+, i8mm) | `--int8` single-stream; int8/int4 batched | 0.6B int8 **0.66**, 1.7B int8 **0.95** (sub-RT); batched matmats ride SMMLA (int8 2.1×, int4 1.6×) |
| **NVIDIA CUDA** | `--quant-mixed` (dp4a on by default) | A100: 0.50 (Talker −33% ms/f vs the f32-act kernel) — see [cuda-performance.md](cuda-performance.md) |

## Testing

```bash
make test-large-int8  # 1.7B INT8 tests (Italian + English, seed 42)
make test-large-int4  # 1.7B INT4 tests (Italian + English, seed 42)
make test-large-quant # All 1.7B quantization tests (INT8 + INT4)
```
