/*
 * qwen_tts_kernels_avx.c - x86 AVX/AVX2 kernel implementations
 *
 * Currently empty: all AVX2 kernels are inline in qwen_tts_kernels.c guarded by
 * #elif defined(__AVX2__). This file is a separate translation unit reserved for
 * a future split if the AVX paths grow large.
 *
 * ACTUAL AVX2 coverage (PLAN 21.3, 2026-06-03 — every NEON hot op now twinned):
 * HOT PATH (~90% of decode) — #elif defined(__AVX2__), 2-row fused / FMA:
 * - bf16_matvec_fused, int8_matvec_fused, q4_0_matvec_inner
 * - qwen_argmax_matvec_bf16, qwen_argmax_matvec_int8
 * - qwen_causal_attention (+ _windowed, _bf16kv): score dot + online-softmax
 *   accumulators (qwen_acc_corr/wt/scale_avx2) + bf16-KV variants
 * AUX (<3%): qwen_rms_norm{,_residual,_per_head}, qwen_bf16_accum_f32,
 *   qwen_bf16_to_f32_vec, qwen_quantize_bf16_to_int8 (load-time)
 * Helpers: qwen_hsum256_ps, qwen_loadu_bf16_8, qwen_dot_f32[_bf16]_avx2.
 *
 * Simple elementwise (qwen_silu/add/mul/scale, swiglu) are plain loops that the
 * compiler auto-vectorizes to AVX2 under -ffast-math — no hand-intrinsics needed.
 *
 * STILL SCALAR on x86 (NEON-only inline, scalar fallback compiles & is correct,
 * just not peak — follow-up): the inline f32<->bf16 pack + NeoX RoPE in
 * talker.c / code_predictor.c / speech_decoder.c, and qwen_snake_activation.
 * int8 uses widen->FMA (no AVX-VNNI yet — PLAN 21.3 [MED], needs the AVX512 box).
 *
 * Build: Linux x86 default is `-mavx2 -mfma` (Makefile SIMD=auto). NO -march=native
 * off-Mac (it SIGILLs on older field CPUs). `make blas SIMD=scalar` for pre-AVX2.
 * Linux x86_64 needs <immintrin.h> (gcc/clang ship it).
 */
