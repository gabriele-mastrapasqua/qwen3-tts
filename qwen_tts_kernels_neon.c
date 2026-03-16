/*
 * qwen_tts_kernels_neon.c - ARM NEON kernel implementations
 *
 * Currently empty: all NEON kernels are inline in qwen_tts_kernels.c
 * guarded by #ifdef __ARM_NEON. This file is compiled as a separate
 * translation unit but contains no code.
 *
 * Reserved for future use if NEON kernels grow complex enough to
 * warrant splitting from the main kernels file.
 *
 * Active NEON optimizations (in qwen_tts_kernels.c):
 * - bf16_matvec_fused: 2-row fused matvec, 32-element/iter, 8 accumulators
 * - int8_matvec_fused: 2-row fused INT8 matvec, 16-element/iter
 * - qwen_rms_norm: 8-element vectorized sum-of-squares + normalize
 * - qwen_causal_attention: 16-element unrolled dot-product, 4 accumulators
 * - qwen_causal_attention_windowed: same with sliding window
 * - qwen_snake_activation: 4-element vectorized (vvsinf on Apple)
 * - NeoX RoPE: 4-element vectorized in talker + speech decoder
 */
