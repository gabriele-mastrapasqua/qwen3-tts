/*
 * qwen_tts_metal.metal - Metal compute shaders for Qwen3-TTS
 *
 * BF16 matrix-vector multiplication optimized for Apple Silicon.
 * Uses simdgroup reduction for efficient per-row dot products.
 */

#include <metal_stdlib>
using namespace metal;

/* bf16 → f32 conversion */
static inline float bf16_to_f32(ushort bf) {
    return as_type<float>((uint(bf)) << 16);
}

/* ========================================================================
 * bf16 matvec: y[rows] = W_bf16[rows, cols] @ x[cols]
 *
 * Each simdgroup (32 threads) computes one output row.
 * Threads split the column dimension, then reduce via simd_sum.
 *
 * Grid:  [rows * 32, 1, 1]   (rows simdgroups × 32 threads each)
 * Group: [32, 1, 1]           (one simdgroup per threadgroup)
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
    uint tid                   [[thread_position_in_grid]],
    uint lane                  [[thread_index_in_simdgroup]])
{
    int row = (int)(tid / 32);
    if (row >= p.rows) return;

    int cols = p.cols;
    device const ushort *w_row = W + (long)row * cols;

    /* Each of 32 lanes handles a strided slice of columns */
    float acc = 0.0f;
    for (int c = (int)lane; c < cols; c += 32) {
        acc += bf16_to_f32(w_row[c]) * x[c];
    }

    /* Reduce across 32 lanes */
    acc = simd_sum(acc);

    /* Lane 0 writes the result */
    if (lane == 0) {
        y[row] = acc;
    }
}
