/* qwen_tts_q8repack.h — GGUF Q8_0 weights, repacked 4-row, run on ARM i8mm/dotprod.
 *
 * WHY THIS AND NOT KLEIDIAI
 * KleidiAI has no block-wise int8 RHS: every int8 packer it ships is `qsi8cxp`, one
 * scale per output channel (verified across the library - zero `qsi8c32` packers).
 * Feeding it a GGUF Q8_0 would mean dequantizing the per-32 scales and re-quantizing
 * per row, which throws away the only thing that makes Q8_0 better than our own int8.
 * ggml's ARM path instead has `q8_0_4x8_q8_0`, which keeps every fp16 scale and is a
 * pure byte shuffle. That is what this file ports - the layout and the instruction
 * strategy, not the ggml runtime. See the design notes
 */
#ifndef QWEN_TTS_Q8REPACK_H
#define QWEN_TTS_Q8REPACK_H

#include <stddef.h>
#include <stdint.h>

#define Q8_0_BLOCK_SIZE 32

/* One GGUF Q8_0 block, exactly as it sits in the file: 34 bytes. */
typedef struct {
    uint16_t d;                     /* fp16 scale */
    int8_t   qs[Q8_0_BLOCK_SIZE];
} q8_0_block_t;

/* Four rows interleaved, ggml's `block_q8_0x4`: 136 bytes for 4 rows x 32 weights =
 * 8.5 bit/weight, IDENTICAL to the source. The four scales are carried verbatim.
 *
 * qs holds 16 chunks of 8 bytes; chunk i is row (i%4) at byte offset (i/4)*8. So for
 * offset group g, qs[32g .. 32g+15] is rows 0,1 and qs[32g+16 .. 32g+31] is rows 2,3 -
 * which is precisely the 2x8 operand SMMLA wants, with no shuffling at compute time. */
typedef struct {
    uint16_t d[4];
    int8_t   qs[128];
} q8_0x4_block_t;

/* Compiled in AND runnable here (i8mm for the GEMM, dotprod for the GEMV). */
int qwen_q8r_supported(void);
int qwen_q8r_enabled(void);          /* the above, minus QWEN_NO_Q8REPACK=1 */

/* Repack rows x (cols/32) source blocks into the 4-row interleaved form.
 * `rows` must be a multiple of 4. Returns the destination byte count, 0 on refusal.
 * Byte count in == byte count out, by construction; the caller can assert it. */
size_t qwen_q8r_packed_bytes(int rows, int cols);
int    qwen_q8r_repack(q8_0x4_block_t *dst, const q8_0_block_t *src, int rows, int cols);

/* The inverse. Exists ONLY to prove the repack is bit-preserving: de-repack and
 * memcmp against the source must be byte-identical, not merely close. A shuffle that
 * is numerically close but not exact means a scale or a quant landed on the wrong row,
 * and that produces confident, plausible audio. */
int qwen_q8r_derepack(q8_0_block_t *dst, const q8_0x4_block_t *src, int rows, int cols);

/* Register a repacked matrix under the pointer the kernels will be called with. */
int qwen_q8r_register(const void *key, const q8_0_block_t *src, int rows, int cols);

/* Y = W @ X for a registered key. B==1 -> dotprod GEMV; B>1 -> i8mm SMMLA GEMM.
 * X is [cols, B] and Y is [rows, B], the engine's own convention.
 * Returns 1 if it ran, 0 if the caller must fall back. */
int qwen_q8r_matmul(float *Y, const void *key, const float *X, int rows, int cols, int B);

void qwen_q8r_stats(int *n_packed, size_t *bytes);

#endif
