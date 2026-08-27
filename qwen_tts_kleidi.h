/* qwen_tts_kleidi.h — Arm KleidiAI micro-kernels for GGUF Q4_0 weights. */
#ifndef QWEN_TTS_KLEIDI_H
#define QWEN_TTS_KLEIDI_H

#include <stddef.h>
#include <stdint.h>

/* Compiled in AND runnable on this CPU (i8mm for the GEMM, dotprod for the GEMV). */
int qwen_kleidi_supported(void);

/* Compiled in, runnable, and not switched off by QWEN_NO_KLEIDI=1. */
int qwen_kleidi_enabled(void);

/* Pack one Q4_0 weight matrix into KleidiAI's RHS layout, ONCE, at model load.
 *
 * `key` is the pointer the engine's kernels will later be called with (our own
 * q4_0_block_t array) — the registry is keyed by it, exactly like the speech
 * decoder's quantized-weight cache is keyed by its source pointer. `ggml_blocks`
 * points at the RAW GGUF tensor bytes: KleidiAI's `qsu4c32s16s0` source layout IS
 * ggml's block_q4_0 (fp16 scale, then 16 bytes packing k and k+16 in one byte), so
 * nothing is converted on the way in.
 *
 * rows = N (output features), cols = K (input features). Returns 1 on success. */
int qwen_kleidi_register_q4(const void *key, const uint8_t *ggml_blocks, int rows, int cols);

/* Y = W @ X using the packed RHS registered for `key`.
 *   B == 1: Y[rows]      , X[cols]        -> dotprod GEMV
 *   B >  1: Y[rows*B]    , X[cols*B]      -> i8mm GEMM (both column-major in B)
 * Returns 1 if KleidiAI ran, 0 if the caller must fall back. */
int qwen_kleidi_matmul_q4(float *Y, const void *key, const float *X, int rows, int cols, int B);

/* Compare KleidiAI against the engine's own q4 kernel on one packed matrix, with a
 * deterministic pseudo-random activation. Fills *max_abs and *rel (RMS of the
 * difference over RMS of the reference). Returns 1 if the comparison ran.
 * This is the gate the whole path has to pass BEFORE any speed number is quoted. */
int qwen_kleidi_selfcheck(const void *key, int rows, int cols, float *max_abs, float *rel);

/* How many weight matrices are packed, and how many bytes they hold. */
void qwen_kleidi_stats(int *n_packed, size_t *bytes);


/* ── INT8: our own per-row weight format, KleidiAI's compute ──────────────────────
 *
 * Our INT8 weights are int8 + ONE SCALE PER OUTPUT ROW, which is exactly KleidiAI's
 * `qsi8cxp` per-channel semantics: the packer takes our bytes and our scales VERBATIM,
 * with no requantization and no change of quantizer. Only the compute engine differs
 * -- and the ACTIVATION quantizer, which KleidiAI does inside its LHS pack
 * (asymmetric min/max with a zero point per row, against our symmetric amax/127).
 * Measured equivalent to within 3% of the relative error, Kleidi marginally better:
 * the design notes
 *
 * `key` is the int8 weight pointer the engine's kernels are called with. */
int qwen_kleidi_register_i8(const void *key, const int8_t *W, const float *scale,
                            int rows, int cols);
/* Engine layout: X is [cols,B], Y is [rows,B]. Transposes at B>1, so this is the
 * drop-in for call sites that have not been moved to the native layout yet. */
int qwen_kleidi_matmul_i8(float *Y, const void *key, const float *X, int rows, int cols, int B);
/* NATIVE layout: lhs is [B][cols] with `lhs_stride` BYTES between rows, dst is
 * [B][rows] with `dst_stride` BYTES between rows. This is the layout the engine's
 * producers already have (qwen_batch_proj_q's `src`, prefill's `Xn`), so on this
 * path the gather and the scatter disappear instead of being optimized. */
int qwen_kleidi_matmul_i8_native(float *dst, const void *key, const float *lhs,
                                 size_t lhs_stride, size_t dst_stride,
                                 int rows, int cols, int B);

/* ── Per-family gates (QWEN_KAI_OPS) ─────────────────────────────────────────────
 * The backend is NOT applied wholesale until we know which operations are sensitive.
 * A weight is tagged at registration with its component and family, and the matmul
 * declines unless that pair is enabled. QWEN_KAI_OPS is a comma list; a matrix runs
 * on KleidiAI if ANY token matches it:
 *
 *   all | none | prefill | talker | cp | qkv | o | ffn | heads | other
 *   talker.ffn | cp.qkv | ...                      (component-scoped)
 *
 * Default `all`, i.e. what was measured. This exists to bisect a quality drift down
 * to the first family that causes it, instead of choosing between the whole backend
 * and none of it. */
enum { QWEN_KAI_COMP_TALKER = 0, QWEN_KAI_COMP_CP, QWEN_KAI_COMP_N };
enum { QWEN_KAI_FAM_QKV = 0, QWEN_KAI_FAM_O, QWEN_KAI_FAM_FFN,
       QWEN_KAI_FAM_HEADS, QWEN_KAI_FAM_OTHER, QWEN_KAI_FAM_N };

int qwen_kleidi_register_i8_fam(const void *key, const int8_t *W, const float *scale,
                                int rows, int cols, int comp, int fam);
int qwen_kleidi_register_bf16_fam(const void *key, const uint16_t *W,
                                  int rows, int cols, int comp, int fam);
/* Is the bf16 PREFILL family on? Read by the prefill call site, which has no key. */
int qwen_kleidi_prefill_enabled(void);

/* Fused QKV: three matrices, one packed activation. Returns 1 if all three ran. */
/* Batched twin of qwen_kleidi_matmul_i8_qkv: one LHS pack and one barrier for the
 * three projections at B>=2. Returns 0 when it declines (not int8, shape mismatch,
 * QWEN_KAI_QKV_FUSED=0) and the caller must then do the three separate calls. */
int qwen_kleidi_matmul_i8_qkv_native(float *dq, float *dk, float *dv,
                                     const void *keyq, const void *keyk, const void *keyv,
                                     const float *lhs, size_t lhs_stride,
                                     int in_dim, int q_dim, int kv_dim, int B);

int qwen_kleidi_matmul_i8_qkv(float *q, float *k, float *v,
                              const void *keyq, const void *keyk, const void *keyv,
                              const float *x, int in_dim, int q_dim, int kv_dim);

/* ── BF16: the prefill path ────────────────────────────────────────────────────── */
int qwen_kleidi_register_bf16(const void *key, const uint16_t *W, int rows, int cols);
int qwen_kleidi_matmul_bf16(float *Y, const void *key, const float *X, int rows, int cols, int B);
int qwen_kleidi_matmul_bf16_native(float *dst, const void *key, const float *lhs,
                                   size_t lhs_stride, size_t dst_stride,
                                   int rows, int cols, int B);

/* Per-family switches, so a regression can be bisected without a rebuild:
 * QWEN_NO_KAI_I8=1 and QWEN_NO_KAI_BF16=1. Both default ON where the ISA allows. */
int qwen_kleidi_i8_enabled(void);
int qwen_kleidi_bf16_enabled(void);

/* Packed bytes per family, for the RSS line of any benchmark. */
void qwen_kleidi_stats_by_kind(int *n_q4, size_t *b_q4, int *n_i8, size_t *b_i8,
                               int *n_bf, size_t *b_bf);

#endif
