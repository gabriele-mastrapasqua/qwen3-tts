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
 * Active AVX optimizations (in qwen_tts_kernels.c):
 * - bf16_matvec_fused: AVX2 bf16→f32 conversion + fma
 * - int8_matvec_fused: AVX2 INT8 dot-product with scale
 * - qwen_rms_norm: AVX2 vectorized sum-of-squares + normalize
 * - qwen_causal_attention: AVX2 dot-product with FMA
 * - qwen_snake_activation: AVX2 vectorized
 * - qwen_bf16_accum_f32: AVX2 batch bf16→f32 accumulate
 *
 * Build requirement: -march=native or -mavx2 -mfma
 * Linux x86_64 also needs <immintrin.h> (provided by gcc/clang).
 */
