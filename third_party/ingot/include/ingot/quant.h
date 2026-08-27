/* ingot/quant.h — dequantization and quantized kernels for the ggml block
 * formats carried inside GGUF.
 *
 * This module is optional: build with INGOT_NO_QUANT defined (or just do not
 * link quant.o / kernels.o) and the container readers still work. Nobody who
 * only wants to read a file should pay for the SIMD.
 *
 * Two layers:
 *   1. ingot_dequant()  — spec-faithful scalar decode of a block format to
 *      f32. Writes into a caller-owned buffer. Never allocates.
 *   2. the kernels      — matvec / batched matmat straight off the quantized
 *      weights, which is the whole point of the format: dequantizing a 7B
 *      model to f32 costs 28 GB, multiplying against it in place costs zero.
 *
 * SPDX-License-Identifier: MIT */
#ifndef INGOT_QUANT_H
#define INGOT_QUANT_H

#include <stddef.h>
#include <stdint.h>

#include "ingot/dtype.h"
#include "ingot/gguf.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ── dequantization ─────────────────────────────────────────────────────── */

/* `src` holds `nelem` elements in `type`'s block layout; `dst` receives nelem
 * floats. `nelem` must be a multiple of the block size. Returns -1 for a type
 * this library cannot decode (ask ingot_type_can_dequant first). */
int ingot_dequant(int type, const void *src, size_t nelem, float *dst);

/* The same, straight off a GGUF tensor. `dst` needs t->nelem floats. */
int ingot_gguf_dequant(const ingot_gguf *g, const ingot_tensor *t, float *dst);

/* ── CPU dispatch ───────────────────────────────────────────────────────── */

typedef struct {
    int neon, dotprod, i8mm;      /* ARM: compiled AND present at runtime */
    int avx2, avx512, avx512_vnni;/* x86: same                            */
    int bf16;                     /* ARM FEAT_BF16 (BFDOT/BFMMLA)         */
    int f16c, avx512_bf16;        /* x86: VCVTPH2PS / VDPBF16PS           */
} ingot_cpu_caps;

ingot_cpu_caps ingot_cpu(void);

/* Cap the SIMD level by hand: "auto" (default), "scalar", "avx2", "vnni".
 * A level above what the CPU supports is silently downgraded. Returns the
 * effective level (0 scalar, 1 avx2/neon, 2 vnni/dotprod). The environment
 * variable INGOT_CAPS does the same thing without a code change.
 *
 * INGOT_CAPS_ASSUME=avx2|neon|dotprod|vnni goes the OTHER way: it trusts the
 * BUILD rather than CPUID, turning on a feature the CPU did not report. Only
 * what was compiled in can be forced, so it cannot conjure an instruction the
 * binary does not contain. It exists for emulators that under-report — Rosetta
 * 2 executes AVX2 without advertising it, and without this the x86 kernels
 * cannot be exercised on an Apple Silicon machine at all. A testing tool: it
 * will fault on hardware that genuinely lacks the feature. */
int ingot_cpu_set_level(const char *name);

/* ── thread injection ───────────────────────────────────────────────────────
 * The batched kernels parallelise over row strips. ingot does not own a thread
 * pool — every consumer already has one, and two pools fighting over the same
 * cores is a measurable loss. Hand yours in here; the default runs inline. */
typedef void (*ingot_range_fn)(size_t begin, size_t end, void *user);
typedef void (*ingot_parallel_for_fn)(size_t count, ingot_range_fn fn, void *user);
void ingot_set_parallel_for(ingot_parallel_for_fn fn);

/* ── kernels ────────────────────────────────────────────────────────────────
 * Layout: `weights` is a row-major [rows, cols] matrix of blocks, exactly as
 * stored in the GGUF (cols must be a multiple of the block size).
 *   matvec: output[rows]            = weights * input[cols]
 *   matmat: output[tokens][rows]    = input[tokens][cols] * weights^T
 *   dequant: output[rows*cols] f32
 * All return 0 on success. */

int ingot_q4_k_matvec(const void *weights, size_t rows, size_t cols,
                      const float *input, float *output);
int ingot_q4_k_matvec_scalar(const void *weights, size_t rows, size_t cols,
                             const float *input, float *output);
int ingot_q4_k_dequant(const void *weights, size_t rows, size_t cols, float *output);
int ingot_q4_k_matmat(const void *weights, size_t rows, size_t cols,
                      const float *input, float *output, size_t tokens);

int ingot_q5_k_matvec(const void *weights, size_t rows, size_t cols,
                      const float *input, float *output);
int ingot_q5_k_matvec_scalar(const void *weights, size_t rows, size_t cols,
                             const float *input, float *output);
int ingot_q5_k_dequant(const void *weights, size_t rows, size_t cols, float *output);
int ingot_q5_k_matmat(const void *weights, size_t rows, size_t cols,
                      const float *input, float *output, size_t tokens);

int ingot_q6_k_matvec(const void *weights, size_t rows, size_t cols,
                      const float *input, float *output);
int ingot_q6_k_dequant(const void *weights, size_t rows, size_t cols, float *output);
int ingot_q6_k_matmat(const void *weights, size_t rows, size_t cols,
                      const float *input, float *output, size_t tokens);

int ingot_q3_k_matvec(const void *weights, size_t rows, size_t cols,
                      const float *input, float *output);
int ingot_q3_k_dequant(const void *weights, size_t rows, size_t cols, float *output);
int ingot_q3_k_matmat(const void *weights, size_t rows, size_t cols,
                      const float *input, float *output, size_t tokens);

int ingot_q2_k_matvec(const void *weights, size_t rows, size_t cols,
                      const float *input, float *output);
int ingot_q2_k_dequant(const void *weights, size_t rows, size_t cols, float *output);
int ingot_q2_k_matmat(const void *weights, size_t rows, size_t cols,
                      const float *input, float *output, size_t tokens);

/* Q4_0: 32 weights in 18 bytes, split-half nibbles biased by 8. What Gemma 4
 * QAT checkpoints ship as, which is why it has a hand-written kernel. */
int ingot_q4_0_matvec(const void *weights, size_t rows, size_t cols,
                      const float *input, float *output);
int ingot_q4_0_dequant(const void *weights, size_t rows, size_t cols, float *output);

int ingot_q8_0_matvec(const void *weights, size_t rows, size_t cols,
                      const float *input, float *output);
int ingot_q8_0_dequant(const void *weights, size_t rows, size_t cols, float *output);
int ingot_q8_0_matmat(const void *weights, size_t rows, size_t cols,
                      const float *input, float *output, size_t tokens);

/* ── dense F16 / BF16 kernels ───────────────────────────────────────────────
 * `weights` is the little-endian 2-byte stream exactly as stored in the file
 * (zero-copy from the mapping), row-major [rows, cols]; input and output are
 * f32. The widening is exact, so the result is plain f32 arithmetic on the
 * stored values — a reordered sum, never an approximation. matvec is
 * single-threaded like the quantized ones; matmat parallelizes over rows
 * through ingot_set_parallel_for. */
int ingot_bf16_matvec(const void *weights, size_t rows, size_t cols,
                      const float *input, float *output);
int ingot_bf16_matmat(const void *weights, size_t rows, size_t cols,
                      const float *input, float *output, size_t tokens);
int ingot_bf16_dequant(const void *weights, size_t rows, size_t cols, float *output);
int ingot_f16_matvec(const void *weights, size_t rows, size_t cols,
                     const float *input, float *output);
int ingot_f16_matmat(const void *weights, size_t rows, size_t cols,
                     const float *input, float *output, size_t tokens);
int ingot_f16_dequant(const void *weights, size_t rows, size_t cols, float *output);

/* ── the precision contract ─────────────────────────────────────────────────
 * From two tokens up, Q4_K and Q5_K batched matmat quantize the ACTIVATIONS to
 * int8 by default: 1.5-2.5x faster, at a relative error around 2.4e-3 instead
 * of the last few ulp of a reorder. Everything else stays an exact reorder.
 *
 * This matters when the CPU result is the reference a GPU path is compared
 * against. Ask instead of guessing: a gate that picks 5e-3 where 1e-5 is owed
 * passes for the wrong reason, and one that picks 1e-5 where 5e-3 is owed
 * fails for the wrong reason.
 *   - ingot_matmat_is_exact(tokens): non-zero when this width takes the exact
 *     path.
 *   - the *_exact twins: always exact, whatever the default is.
 *   - INGOT_SDOT=0 in the environment: turn the int8 path off globally. */
int ingot_matmat_is_exact(size_t tokens);
int ingot_q4_k_matmat_exact(const void *weights, size_t rows, size_t cols,
                            const float *input, float *output, size_t tokens);
int ingot_q5_k_matmat_exact(const void *weights, size_t rows, size_t cols,
                            const float *input, float *output, size_t tokens);

/* ── type-generic entry points ──────────────────────────────────────────────
 * The calls above make you name the format at the call site, which is right
 * for an engine that has decided its weights are Q4_K and wrong for a loader
 * handed whatever GGUF the user downloaded. These work for EVERY type
 * ingot_type_can_dequant() accepts, dispatching to the hand-written kernel
 * when one exists and decoding row-by-row otherwise (same complexity, no SIMD
 * inner loop, no int8 activation path).
 *
 *   matvec:  output[rows]         = weights * input[cols]
 *   matmat:  output[tokens][rows] = input[tokens][cols] * weights^T
 *
 * ingot_has_kernel() says whether a type takes the fast route, so a caller
 * choosing a quantization can prefer one that does. */
int ingot_matvec(int type, const void *weights, size_t rows, size_t cols,
                 const float *input, float *output);
int ingot_matmat(int type, const void *weights, size_t rows, size_t cols,
                 const float *input, float *output, size_t tokens);
int ingot_dequant_matrix(int type, const void *weights, size_t rows, size_t cols,
                         float *output);
int ingot_has_kernel(int type);

/* The same, straight off a GGUF tensor: rows and cols come from `ne`, so the
 * ggml dimension flip (ne[0] is the INPUT width) happens once here instead of
 * at every call site. Rank 1 counts as a single row. */
int ingot_gguf_matvec(const ingot_gguf *g, const ingot_tensor *t,
                      const float *input, float *output);
int ingot_gguf_matmat(const ingot_gguf *g, const ingot_tensor *t,
                      const float *input, float *output, size_t tokens);
int ingot_gguf_dequant_matrix(const ingot_gguf *g, const ingot_tensor *t, float *output);

/* ── f32 -> block format ────────────────────────────────────────────────────
 * `count` must be a multiple of the block size and `out` must hold
 * count / block_elems * block_bytes bytes (ask ingot_type_nbytes).
 * Supported: F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K, Q3_K, Q4_K,
 * Q5_K, Q6_K. The K-quant encoders are unweighted least-squares fits; ggml
 * additionally weights them with an imatrix, which buys several percent of
 * relative error at 2-3 bits and almost nothing at 6. */
int ingot_quantize(int type, const float *values, size_t count, void *out);
int ingot_can_quantize(int type);

/* f32 -> Q4_K. `count` must be a multiple of 256; `out` needs count/256*144
 * bytes. The one direction a converter needs and no reader provides. */
int ingot_q4_k_quantize(const float *values, size_t count, void *out);

#ifdef __cplusplus
}
#endif
#endif
