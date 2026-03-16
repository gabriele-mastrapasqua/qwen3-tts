/*
 * qwen_tts_kernels_generic.c - Generic (scalar) kernel implementations
 *
 * Currently empty: all scalar fallbacks are inline in qwen_tts_kernels.c
 * inside #else blocks after each SIMD implementation. This file is compiled
 * as a separate translation unit but contains no code.
 *
 * The scalar paths are functionally equivalent to the SIMD paths and serve
 * as reference implementations + fallback for architectures without
 * NEON or AVX2 support.
 */
