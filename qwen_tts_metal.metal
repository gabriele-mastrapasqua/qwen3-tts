/*
 * qwen_tts_metal.metal - Metal compute shaders for Qwen3-TTS
 *
 * BF16 matrix-vector multiplication for Talker and Code Predictor.
 */

#include <metal_stdlib>
using namespace metal;

/* bf16 → f32 conversion: shift left by 16 bits */
static inline float bf16_to_f32(ushort bf) {
    return as_type<float>((uint(bf)) << 16);
}

/* ========================================================================
 * bf16 matvec: y[rows] = W_bf16[rows, cols] @ x[cols]
 *
 * Each thread computes one output row by iterating over all columns.
 * Simple and correct — no inter-thread reduction needed.
 *
 * Grid:  [ceil(rows / 256), 1, 1]
 * Group: [256, 1, 1]
 * ======================================================================== */

struct matvec_params {
    int rows;
    int cols;
};

kernel void matvec_bf16(
    device const ushort *W     [[buffer(0)]],  /* [rows, cols] bf16 */
    device const float  *x     [[buffer(1)]],  /* [cols] f32 */
    device float        *y     [[buffer(2)]],  /* [rows] f32 */
    constant matvec_params &p  [[buffer(3)]],
    uint tid                   [[thread_position_in_grid]])
{
    int row = (int)tid;
    if (row >= p.rows) return;

    int cols = p.cols;
    device const ushort *w_row = W + (long)row * cols;

    /* Accumulate dot product — 4-wide unroll for throughput */
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    int c = 0;
    for (; c + 3 < cols; c += 4) {
        acc0 += bf16_to_f32(w_row[c])     * x[c];
        acc1 += bf16_to_f32(w_row[c + 1]) * x[c + 1];
        acc2 += bf16_to_f32(w_row[c + 2]) * x[c + 2];
        acc3 += bf16_to_f32(w_row[c + 3]) * x[c + 3];
    }
    for (; c < cols; c++) {
        acc0 += bf16_to_f32(w_row[c]) * x[c];
    }

    y[row] = acc0 + acc1 + acc2 + acc3;
}
