/*
 * qwen_tts_kernels_impl.h - Architecture dispatch
 *
 * SIMD dispatch strategy:
 * All SIMD kernels (NEON on ARM, AVX on x86) are implemented inline in
 * qwen_tts_kernels.c using #ifdef __ARM_NEON / #ifdef __AVX2__ guards.
 * Each SIMD path has a scalar fallback in #else blocks.
 *
 * This is intentional: the codebase is small enough that a single-file
 * approach avoids indirection overhead and keeps the dispatch obvious.
 * The separate *_neon.c / *_avx.c / *_generic.c files exist as compilation
 * units in the Makefile but are currently empty — they are reserved for
 * future use if kernel complexity grows enough to warrant splitting.
 *
 * Cross-platform guarantee:
 * - ARM (Apple M1/M2, Linux aarch64): NEON intrinsics via <arm_neon.h>
 * - x86-64 (Linux, WSL2): AVX2 intrinsics via <immintrin.h>
 * - Fallback: scalar C loops (correct but slower)
 *
 * All buffers used by SIMD kernels MUST be 64-byte aligned (see
 * aligned_malloc() in qwen_tts_kernels.h). This matches both Apple
 * M-series (64B cache lines) and x86-64 (64B cache lines, AVX-512
 * alignment requirement).
 */

#ifndef QWEN_TTS_KERNELS_IMPL_H
#define QWEN_TTS_KERNELS_IMPL_H

/* Currently all dispatch is handled via #ifdef guards in qwen_tts_kernels.c.
 * This header is reserved for future dispatch macros if needed. */

#endif /* QWEN_TTS_KERNELS_IMPL_H */
