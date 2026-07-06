# GPU acceleration status — CPU / Metal / CUDA, per accelerated code point

Branch `feat/gpu-backends`. `make blas` (CPU-only) is the default and stays byte-identical;
Metal/CUDA are opt-in (`make metal` / `make cuda`, `--backend metal|cuda`). All numbers measured on
the **Apple M1 base** (8-core GPU, ~68 GB/s shared CPU+GPU). Correctness = `--gpu-selftest` per-op vs the
CPU kernels (bf16-exact within fp-order).

## The one rule that explains every number
- **Compute-bound** work (batched matmul = prefill / TTFA / server batch; ConvNet decoder): GPU **wins big
  even on M1** (matrix units, FLOPs > CPU). Measured **matmat 6.96×**.
- **Memory-bound** work (single-token decode: matvec / FFN B=1): GPU ≈ CPU **on M1 base** because both share
  68 GB/s DRAM — physics, not kernel quality. This win lives on high-BW Apple Silicon (Pro/Max/Ultra
  200–800 GB/s) and on discrete CUDA GPUs (1000+ GB/s).

## Master table

Status: ✅ = implemented + correctness-gated · 🔷 = implemented, GPU-compile-pending (nvcc; logic mirrors the
M1-validated Metal twin) · ⏳ = optimization TODO.

| # | Op (code point) | Bound | CPU — SIMD path | Metal — kernel + M1 result | CUDA | Commit |
|---|---|---|---|---|---|---|
| 1 | **matvec bf16** (Talker/CP decode) | mem | NEON (scalar fallback); no AVX2 | ✅ `matvec_bf16` simdgroup + `ushort4`/`float4` + `simd_sum`; per-op 0.25×, **fused 1.0–1.5×** | ✅ cuBLAS Sgemm resident (B=1) | 5c9826d |
| 2 | **matvec int8** | mem | NEON **SDOT**; AVX-512 VNNI (x86) | ✅ `matvec_int8` **simdgroup + char4** vectorized; rel 4e-3 | 🔷 `k_...`; ⏳ dp4a+q8_1 | 009ed26 |
| 3 | **matvec q4_0** | mem | NEON nibble-unpack | ✅ `matvec_q4_0` **simdgroup + simd_sum**; rel 3e-7 | ⏳ dp4a q4→q8_1 | 009ed26 |
| 4 | **matmat bf16** (prefill / server batch) | **cmp** | hand NEON; prefill via **BLAS/AMX** | ✅ **`simdgroup_matrix` 8×8 MMA**; **B=32 → 6.96×** ⭐ | ✅ cuBLAS Sgemm resident (→GemmEx bf16 sm_80+) | 921c542 |
| 5 | **rms_norm** | mem | NEON + **AVX2** | ✅ 2-level simd/threadgroup reduction; exact | 🔷 `k_rms_norm` block-reduce | 0267eb1 |
| 6 | **rope** (interleaved/neox) | cmp | scalar / NEON | ✅ `rope`; rel 9e-8 | 🔷 `k_rope` | 0267eb1 |
| 7 | **swiglu / silu** | mem | `vvexpf`/scalar | ✅ `swiglu`,`silu`; rel <2e-7 | 🔷 `k_swiglu`/`k_silu` | 0267eb1 |
| 8 | **add / mul / scale** | mem | scalar (auto-vec) | ✅ `eadd`/`emul`/`escale`; exact | 🔷 `k_add`/`k_mul`/`k_scale` | 0267eb1 |
| 9 | **FFN block** (rms→gate_up→swiglu→down→res) | mem(B1)/cmp(batch) | per-op CPU kernels | ✅ fused 1 cmdbuf resident: B=1 **1.07×**, **batched B=16 (MMA) 3.41×** | ⏳ fused (CUDA-graph) | 009ed26 |
| 10 | **attention** (causal GQA) | mem | NEON (f32 + bf16-KV) | ✅ `attention` direct online-softmax; rel 2.7e-7 (flash-vec = opt) | 🔷 `k_attention` | b3df324 |
| 11 | **decoder ConvNet** (480× upsample) | **cmp** | NEON/scalar; snake `sinf` | ✅ `conv1d`+`conv_transpose1d` (tap-solve)+`snake`; **exact** | 🔷 `k_conv1d`/`k_conv_transpose1d`/`k_snake` | b3df324 |
| 12 | **prefill GEMM** (TTFA floor) | **cmp** | **BLAS** (AMX), fp32 weights | ✅ `matmat_f32` `simdgroup_matrix` MMA; **exact** | ✅ cuBLAS Sgemm | b3df324 |
| 13 | **quantize** bf16→int8/q4 (load-time) | — | NEON | n/a (host) | n/a (host) | — |

**Metal column: COMPLETE** (19 ops, all `--gpu-selftest` PASS). **CUDA column: cuBLAS GEMM done + all compute
kernels written** (🔷 GPU-compile-pending — no nvcc on the M1 dev box; each mirrors a validated Metal twin).

### Measured M1 wins (compute-bound → GPU wins)
| block | CPU | Metal | speedup |
|---|---|---|---|
| matmat B=32 | 12.0 ms | 1.84 ms | **6.52×** |
| FFN batched B=16 (MMA, 1 cmdbuf) | 23.4 ms | 6.86 ms | **3.41×** |
| matvec fused (memory-bound) | 0.14 ms | 0.13 ms | ~1.0× (bandwidth ceiling, M1 base) |

### Optimization TODO (not missing ops)
1. **CUDA Graphs / one command buffer per Talker+CP step** — collapse the 16×5-pass launch overhead. (Metal
   fusion pattern proven by the batched FFN; the full per-step resident decode is the remaining integration.)
2. **int8/q4 dp4a decode** (halve bytes on the memory-bound CP path — the discrete-GPU/high-BW lever).
3. **NVIDIA box**: `make cuda NVCC_ARCH=sm_XX` → `--gpu-selftest --backend cuda` → real cuBLAS RTF + tensor-core GemmEx.

## Backend architecture (all three)
- **Seam** `qwen_tts_backend.{h,c}` — vtable (resolver metal→cuda→cpu) + global offload hooks
  `g_qwen_matvec_bf16_hook` / `g_qwen_matmat_bf16_hook` (NULL = CPU default; installed only with `--backend`).
- **Metal** `qwen_tts_metal.{h,m}` — clang ObjC + runtime-compiled MSL; **weights RESIDENT** (cached by
  pointer, uploaded once), IO buffers pooled; fused blocks via on-GPU `memoryBarrier` in one command buffer.
- **CUDA** `qwen_tts_cuda.{h,c}` — cuBLAS-first (gcc, no nvcc); **weights RESIDENT** (converted+uploaded once,
  cached), dX/dY reused. bf16 GemmEx tensor cores + custom decode matvec + CUDA Graphs = G3b (NVIDIA GPU).

## Ranked next (from the ggml/llama.cpp study — docs/gpu-accel-analysis.md + agent report)
1. **One command buffer / CUDA-graph per Talker+CP step** — kills the 16×5-pass launch-overhead trap (the
   Qwen-TTS-specific trap). Prerequisite for real decode RTF.
2. **Speech-decoder ConvNet offload + CPU/GPU overlap** — the one headline single-stream win on M1
   (compute-bound; also frees CPU bandwidth for the next frame's Talker/CP). ~1.2–1.5× e2e.
3. **Prefill/batch matmat** — Metal `mul_mm` (done as `matmat_bf16` MMA) wired into prefill; CUDA cuBLAS.
   Cuts the ~1.65 s TTFA floor; server batch 2–4× (M1), much more on discrete.
4. **Batched bf16 matvec** (`ncols_dst≤8`) — CP cross-request batching; M1-neutral, 5090 5–15× / ~270 GB/s-class 3–4×.
5. RMSNorm+RoPE fused into matvec epilogue · int8/q4 dp4a decode · flash-attn KV-resident.

## CUDA correctness gotchas (for the GPU run)
Row-major↔column-major (compute `Cᵀ=Bᵀ·Aᵀ`, swap operands — never transpose data; validate vs CPU golden) ·
bf16 GemmEx needs sm_80+ (Turing fp16 fallback) · tensor-core tiles want M/N/K %8 + 16-byte-aligned ·
**fp32 accumulate mandatory** (bf16 accumulate diverges from golden) · reproduce q8_1 act-quant + q4 −8 bias
bit-for-bit or late-codebook CP collapses · CUDA-Graph pointer capture (update params, not stale pointers).
