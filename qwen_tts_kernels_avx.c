/*
 * qwen_tts_kernels_avx.c - x86 AVX/AVX2 kernel implementations
 *
 * Currently empty: all AVX kernels are inline in qwen_tts_kernels.c
 * guarded by #ifdef __AVX2__. This file is compiled as a separate
 * translation unit but contains no code.
 *
 * Reserved for future use if AVX kernels grow complex enough to
 * warrant splitting from the main kernels file.
 *
 * ACTUAL AVX2 coverage today (verified 2026-06-03, in qwen_tts_kernels.c):
 * ONLY these 5 auxiliary/elementwise ops have an #elif defined(__AVX2__) branch
 * (they are <3% of decode time per the CP microbench):
 * - qwen_rms_norm / qwen_rms_norm_residual / qwen_rms_norm_per_head
 * - qwen_bf16_accum_f32
 * - qwen_bf16_to_f32_vec
 *
 * EVERYTHING on the hot path FALLS TO SCALAR on x86 (no AVX2 branch):
 * - matvecs (bf16_matvec_fused, int8_matvec_fused, q4_0_matvec_inner,
 *   qwen_argmax_matvec_bf16/_int8) — the ~90% of decode time
 * - int8_matvec_sdot / quantize_act_int8 are ARM-only (#if __ARM_FEATURE_DOTPROD)
 * - qwen_causal_attention (+ _windowed, _bf16kv)
 * - RoPE, f32<->bf16 pack, swiglu/add/mul/scale, snake_activation
 * See PLAN.md 21.3 — adding AVX2 (then VNNI) to these is the open x86 work.
 *
 * Build requirement: -march=native or -mavx2 -mfma
 * Linux x86_64 also needs <immintrin.h> (provided by gcc/clang).
 */
