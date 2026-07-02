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

| # | Op (code point) | Bound | CPU — SIMD path | Metal — kernel + M1 result | CUDA — status | Commit |
|---|---|---|---|---|---|---|
| 1 | **matvec bf16** (Talker/CP decode) | mem | NEON (scalar fallback); no AVX2 | `matvec_bf16` simdgroup + `ushort4`/`float4` + `simd_sum`; per-op 0.25×, **fused 1.0–1.5×** | cuBLAS Sgemm resident (B=1) — DGX-run | 5c9826d |
| 2 | **matvec int8** (int8 CP/Talker) | mem | NEON **SDOT** (ARM); AVX-512 VNNI (x86, validated) | `matvec_int8` in-shader dequant; PASS rel 4e-3 | planned: `__dp4a`/dp4 + q8_1 act-quant (ggml `vecdotq`) | d0e2ff6 |
| 3 | **matvec q4_0** (int4 CP) | mem | NEON nibble-unpack; SDOT-q4 = TODO | `matvec_q4_0` in-shader `(nib-8)*s`; PASS rel 4e-7 | planned: dp4a q4→q8_1 | d0e2ff6 |
| 4 | **matmat bf16** (prefill / server batch) | **cmp** | hand NEON register-blocked; prefill via **BLAS/AMX** | `matmat_bf16` **`simdgroup_matrix` 8×8 MMA**; **B=32 → 6.96×** ⭐ | cuBLAS `Sgemm`/`GemmEx` (bf16 sm_80+) | 921c542 |
| 5 | **rms_norm** (+fused residual) | mem | NEON + **AVX2** | `rms_norm` 2-level simd/threadgroup reduction; PASS exact | planned `norm.cu` block-reduce | d0e2ff6 |
| 6 | **rope** (interleaved / neox) | cmp | scalar / NEON | `rope`; PASS rel 9e-8 | planned `rope.cu` (fuse into QKV epilogue) | d0e2ff6 |
| 7 | **swiglu / silu** | mem | `vvexpf` (Accelerate) / scalar | `swiglu`,`silu`; PASS rel <2e-7 | planned elementwise | d0e2ff6 |
| 8 | **add / mul / scale** | mem | scalar (auto-vec) | `eadd`/`emul`/`escale`; PASS exact | planned elementwise | d0e2ff6 |
| 9 | **FFN block** (rms→gate_up→swiglu→down→res) | mem(B1)/cmp(batch) | per-op CPU kernels | **`qwen_metal_ffn_swiglu` = ONE command buffer, resident activations**; B=1 **1.07×** | planned fused | db0ec05 |
| 10 | **attention** (causal GQA) | mem | NEON (f32 + bf16-KV) | **TODO** — port `flash_attn_ext_vec` (online softmax, KV resident) | TODO `fattn-vec` | — |
| 11 | **speech decoder ConvNet** (480× upsample, snake) | **cmp** | NEON/scalar; snake `sinf` scalar | **TODO** — `conv_transpose_1d` tap-solve + im2col/GEMM + snake (the honest M1 win) | TODO | — |
| 12 | **prefill GEMM** (TTFA floor) | **cmp** | **BLAS** (Accelerate AMX), fp32 weights | **TODO** — `mul_mm` MMA on bf16 (kill the bf16→f32 convert) | cuBLAS `GemmEx` | — |
| 13 | **quantize** bf16→int8/q4 (load-time) | — | NEON | n/a (host) | n/a (host) | — |

## Backend architecture (all three)
- **Seam** `qwen_tts_backend.{h,c}` — vtable (resolver metal→cuda→cpu) + global offload hooks
  `g_qwen_matvec_bf16_hook` / `g_qwen_matmat_bf16_hook` (NULL = CPU default; installed only with `--backend`).
- **Metal** `qwen_tts_metal.{h,m}` — clang ObjC + runtime-compiled MSL; **weights RESIDENT** (cached by
  pointer, uploaded once), IO buffers pooled; fused blocks via on-GPU `memoryBarrier` in one command buffer.
- **CUDA** `qwen_tts_cuda.{h,c}` — cuBLAS-first (gcc, no nvcc); **weights RESIDENT** (converted+uploaded once,
  cached), dX/dY reused. bf16 GemmEx tensor cores + custom decode matvec + CUDA Graphs = G3b (DGX).

## Ranked next (from the ggml/llama.cpp study — docs/gpu-accel-analysis.md + agent report)
1. **One command buffer / CUDA-graph per Talker+CP step** — kills the 16×5-pass launch-overhead trap (the
   Qwen-TTS-specific trap). Prerequisite for real decode RTF.
2. **Speech-decoder ConvNet offload + CPU/GPU overlap** — the one headline single-stream win on M1
   (compute-bound; also frees CPU bandwidth for the next frame's Talker/CP). ~1.2–1.5× e2e.
3. **Prefill/batch matmat** — Metal `mul_mm` (done as `matmat_bf16` MMA) wired into prefill; CUDA cuBLAS.
   Cuts the ~1.65 s TTFA floor; server batch 2–4× (M1), much more on discrete.
4. **Batched bf16 matvec** (`ncols_dst≤8`) — CP cross-request batching; M1-neutral, 5090 5–15× / GB10 3–4×.
5. RMSNorm+RoPE fused into matvec epilogue · int8/q4 dp4a decode · flash-attn KV-resident.

## CUDA correctness gotchas (for the DGX run)
Row-major↔column-major (compute `Cᵀ=Bᵀ·Aᵀ`, swap operands — never transpose data; validate vs CPU golden) ·
bf16 GemmEx needs sm_80+ (Turing fp16 fallback) · tensor-core tiles want M/N/K %8 + 16-byte-aligned ·
**fp32 accumulate mandatory** (bf16 accumulate diverges from golden) · reproduce q8_1 act-quant + q4 −8 bias
bit-for-bit or late-codebook CP collapses · CUDA-Graph pointer capture (update params, not stale pointers).
