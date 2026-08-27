/* Quantized ggml kernels: dequant, matvec and batched matmat for the K-quant
 * family plus Q8_0, with NEON / AVX2 / ARM-SDOT / ARM-SMMLA paths.
 *
 * This is measured code, and the comments record the measurements and the
 * bugs that shaped it: the int8 activation default, the exactness contract,
 * the triple guard around the intrinsics. The numbers quoted throughout were
 * taken on an M1 unless another machine is named.
 *
 * SPDX-License-Identifier: MIT */
#include "ingot/quant.h"
#include "internal.h"

void *ingot_aligned_alloc(size_t alignment, size_t size);
void  ingot_aligned_free(void *ptr);
void  ingot_parallel_for(size_t count, ingot_range_fn fn, void *user);
typedef ingot_range_fn ingot_range_fn_t;

#include <math.h>
#include <stdalign.h>
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#if (defined(__ARM_NEON) || defined(__aarch64__)) && !defined(INGOT_DISABLE_NEON)
#include <arm_neon.h>
#define INGOT_HAVE_Q4_K_NEON 1
#endif

/* The AVX2 kernels lean on FMA, and -mavx2 alone does not imply -mfma: guard
 * on both, or a plain -mavx2 build dies on an always_inline mismatch. Every
 * AVX2 CPU since Haswell has FMA, so the pair costs nothing in practice. */
#if defined(__AVX2__) && defined(__FMA__) && !defined(INGOT_DISABLE_AVX2)
#include <immintrin.h>
#define INGOT_HAVE_Q4_K_AVX2 1
#endif

#define INGOT_QK_K 256
#define INGOT_Q4_K_BYTES 144
#define INGOT_Q5_K_BYTES 176

/* Argument validation for the SIMD dispatchers.
 *
 * Each scalar kernel checks its own arguments, but the NEON/SDOT/AVX2 paths
 * were written for callers inside a single engine that always passed valid
 * pointers, so the dispatchers jumped straight into them. For a library that
 * is not good enough: a null weight pointer has to come back as -1, not as a
 * segfault. Found by the guard case in tests/test_quant.c, which crashed the
 * original. */
static int qk_args_ok(const void *weights, const float *input,
                      const float *output, size_t rows, size_t cols,
                      size_t block_elems) {
    return weights != NULL && input != NULL && output != NULL &&
           rows != 0 && cols != 0 && cols % block_elems == 0;
}

static uint16_t read_u16(const unsigned char *p) {
    return (uint16_t)p[0] | ((uint16_t)p[1] << 8);
}

static float f16_to_f32(uint16_t value) {
    uint32_t sign = (uint32_t)(value & 0x8000u) << 16;
    int exponent = (int)((value >> 10) & 0x1fu);
    uint32_t fraction = value & 0x03ffu;
    uint32_t bits;
    if (exponent == 0) {
        if (fraction == 0) bits = sign;
        else {
            exponent = 1;
            while ((fraction & 0x0400u) == 0) { fraction <<= 1; exponent--; }
            bits = sign | ((uint32_t)(exponent + 112) << 23) |
                   ((fraction & 0x03ffu) << 13);
        }
    } else if (exponent == 31) bits = sign | 0x7f800000u | (fraction << 13);
    else bits = sign | ((uint32_t)(exponent + 112) << 23) | (fraction << 13);
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

static void scale_min(const unsigned char *scales, int index,
                      unsigned char *scale, unsigned char *minimum) {
    if (index < 4) {
        *scale = scales[index] & 63u;
        *minimum = scales[index + 4] & 63u;
    } else {
        *scale = (unsigned char)((scales[index + 4] & 0x0fu) |
                                 ((scales[index - 4] >> 6) << 4));
        *minimum = (unsigned char)((scales[index + 4] >> 4) |
                                   ((scales[index] >> 6) << 4));
    }
}

static float dot_block(const unsigned char *block, const float *input) {
    float d = f16_to_f32(read_u16(block));
    float dmin = f16_to_f32(read_u16(block + 2));
    const unsigned char *scales = block + 4;
    const unsigned char *quantized = block + 16;
    float sum = 0.0f;
    int scale_index = 0;
    for (int base = 0; base < INGOT_QK_K; base += 64) {
        unsigned char scale0, scale1, min0, min1;
        scale_min(scales, scale_index, &scale0, &min0);
        scale_min(scales, scale_index + 1, &scale1, &min1);
        float d0 = d * scale0, d1 = d * scale1;
        float m0 = dmin * min0, m1 = dmin * min1;
        for (int i = 0; i < 32; i++) {
            sum += (d0 * (quantized[i] & 0x0fu) - m0) * input[base + i];
            sum += (d1 * (quantized[i] >> 4) - m1) * input[base + i + 32];
        }
        quantized += 32;
        scale_index += 2;
    }
    return sum;
}

static void dequant_block(const unsigned char *block, float *output) {
    float d = f16_to_f32(read_u16(block));
    float dmin = f16_to_f32(read_u16(block + 2));
    const unsigned char *scales = block + 4;
    const unsigned char *quantized = block + 16;
    int scale_index = 0;
    for (int base = 0; base < INGOT_QK_K; base += 64) {
        unsigned char scale0, scale1, min0, min1;
        scale_min(scales, scale_index, &scale0, &min0);
        scale_min(scales, scale_index + 1, &scale1, &min1);
        float d0 = d * scale0, d1 = d * scale1;
        float m0 = dmin * min0, m1 = dmin * min1;
        for (int i = 0; i < 32; i++) {
            output[base + i] = d0 * (quantized[i] & 0x0fu) - m0;
            output[base + i + 32] = d1 * (quantized[i] >> 4) - m1;
        }
        quantized += 32;
        scale_index += 2;
    }
}

/* f32 -> Q4_K.  Inverse of dequant_block above, so read that first.
 *
 * A block is 256 weights in eight sub-blocks of 32.  Sub-block j is stored as
 * value = (d * sc[j]) * q - (dmin * m[j]), with q in [0,15] and sc[j], m[j]
 * six-bit integers: two levels of scaling, one per block and one per
 * sub-block.  So the fit goes bottom-up — per sub-block find the step and the
 * offset that span its range, then pick the two block-wide f16 factors that
 * express all eight of them in six bits.
 *
 * The offset is stored unsigned, which is why the minimum is clamped at zero:
 * a sub-block whose values are all positive keeps q = 0 mapping to 0 instead
 * of to its own minimum, and pays a little resolution for it. */
static uint16_t f32_to_f16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    uint32_t sign = (bits >> 16) & 0x8000u;
    int exponent = (int)((bits >> 23) & 0xffu) - 127 + 15;
    uint32_t fraction = bits & 0x007fffffu;
    if (exponent >= 31) return (uint16_t)(sign | 0x7c00u);   /* saturate */
    if (exponent <= 0) {
        if (exponent < -10) return (uint16_t)sign;           /* underflow */
        fraction |= 0x00800000u;
        uint32_t shift = (uint32_t)(14 - exponent);
        uint32_t rounded = (fraction + (1u << (shift - 1))) >> shift;
        return (uint16_t)(sign | rounded);
    }
    uint32_t rounded = (fraction + 0x00001000u) >> 13;
    if (rounded > 0x03ffu) { rounded = 0; exponent++; if (exponent >= 31)
        return (uint16_t)(sign | 0x7c00u); }
    return (uint16_t)(sign | ((uint32_t)exponent << 10) | rounded);
}

static void write_u16(unsigned char *p, uint16_t value) {
    p[0] = (unsigned char)(value & 0xffu);
    p[1] = (unsigned char)(value >> 8);
}

int ingot_q4_k_quantize(const float *values, size_t count, void *out) {
    if (values == NULL || out == NULL || count == 0 ||
        count % INGOT_QK_K != 0) return -1;
    unsigned char *dst = out;
    for (size_t block = 0; block < count / INGOT_QK_K; block++) {
        const float *source = values + block * INGOT_QK_K;
        unsigned char *target = dst + block * INGOT_Q4_K_BYTES;
        float steps[8], offsets[8];
        float max_step = 0.0f, max_offset = 0.0f;
        for (int j = 0; j < 8; j++) {
            const float *sub = source + j * 32;
            float low = sub[0], high = low;
            for (int i = 1; i < 32; i++) {
                if (sub[i] < low) low = sub[i];
                if (sub[i] > high) high = sub[i];
            }
            if (low > 0.0f) low = 0.0f;   /* q = 0 must be representable */
            if (high < 0.0f) high = 0.0f;
            /* Spanning min..max exactly is the obvious fit and not the best
             * one: a single outlier stretches the step and coarsens the other
             * thirty-one values.  Try a few narrower spans and keep whichever
             * minimises the squared error, clipping included.
             *
             * Measured, not assumed: span search plus the least-squares refit
             * below is worth about 7% of the error on both distributions the
             * test carries (spread 6.40 -> 5.98% rel L2, bell 7.76 -> 7.24%).
             * That is a modest win on a one-off offline cost, which is why it
             * is here; anyone expecting the fit to rescue 4 bits should read
             * those numbers first.  Bell-shaped values — what a weight matrix
             * actually holds — come out WORSE than evenly spread ones, because
             * min..max over 32 Gaussian samples spans ~5 sigma while the mass
             * sits within 1, so the step is wide relative to the RMS. */
            float best_step = (high - low) / 15.0f, best_offset = -low;
            float best_error = -1.0f;
            for (int candidate = 0; candidate <= 5; candidate++) {
                float shrink = 1.0f - 0.04f * (float)candidate;
                float centre = 0.5f * (high + low);
                float half = 0.5f * shrink * (high - low);
                float lo = centre - half, hi = centre + half;
                if (lo > 0.0f) lo = 0.0f;
                if (hi < 0.0f) hi = 0.0f;
                float step = (hi - lo) / 15.0f;
                if (!(step > 0.0f)) continue;
                /* Two rounds of: assign each value to a level, then re-solve
                 * step and offset by least squares against that assignment.
                 * The span above only picks the starting point; this is what
                 * moves the fit, because min..max is optimal for coverage and
                 * not for squared error. */
                for (int round = 0; round < 2; round++) {
                    float sum_q = 0.0f, sum_qq = 0.0f, sum_x = 0.0f, sum_xq = 0.0f;
                    for (int i = 0; i < 32; i++) {
                        long level = lrintf((sub[i] - lo) / step);
                        if (level < 0) level = 0;
                        if (level > 15) level = 15;
                        float q = (float)level;
                        sum_q += q; sum_qq += q * q;
                        sum_x += sub[i]; sum_xq += sub[i] * q;
                    }
                    float determinant = 32.0f * sum_qq - sum_q * sum_q;
                    if (!(fabsf(determinant) > 1e-12f)) break;
                    float fitted = (32.0f * sum_xq - sum_q * sum_x) / determinant;
                    float base = (sum_x - fitted * sum_q) / 32.0f;
                    /* The format stores the offset unsigned, so a fit that
                     * wants a positive base is not expressible: pin it at zero
                     * and re-solve the step alone. */
                    if (base > 0.0f) {
                        base = 0.0f;
                        if (sum_qq > 0.0f) fitted = sum_xq / sum_qq;
                    }
                    if (!(fitted > 0.0f)) break;
                    step = fitted; lo = base;
                }
                float error = 0.0f;
                for (int i = 0; i < 32; i++) {
                    long q = lrintf((sub[i] - lo) / step);
                    if (q < 0) q = 0;
                    if (q > 15) q = 15;
                    float diff = step * (float)q + lo - sub[i];
                    error += diff * diff;
                }
                if (best_error < 0.0f || error < best_error) {
                    best_error = error;
                    best_step = step;
                    best_offset = -lo;
                }
            }
            steps[j] = best_step;
            offsets[j] = best_offset;
            if (steps[j] > max_step) max_step = steps[j];
            if (offsets[j] > max_offset) max_offset = offsets[j];
        }
        /* The block factors are f16 and get read back as f16, so quantise the
         * sub-block indices against the ROUNDED d and dmin, not the ideal
         * ones — otherwise the two ends of the fit disagree by an ulp. */
        uint16_t d_half = f32_to_f16(max_step / 63.0f);
        uint16_t dmin_half = f32_to_f16(max_offset / 63.0f);
        write_u16(target, d_half);
        write_u16(target + 2, dmin_half);
        float d = f16_to_f32(d_half), dmin = f16_to_f32(dmin_half);
        unsigned char scale_bits[8], min_bits[8];
        for (int j = 0; j < 8; j++) {
            long scale = d > 0.0f ? lrintf(steps[j] / d) : 0;
            long minimum = dmin > 0.0f ? lrintf(offsets[j] / dmin) : 0;
            if (scale < 0) scale = 0;
            if (scale > 63) scale = 63;
            if (minimum < 0) minimum = 0;
            if (minimum > 63) minimum = 63;
            scale_bits[j] = (unsigned char)scale;
            min_bits[j] = (unsigned char)minimum;
        }
        /* Pack six-bit pairs the way scale_min() unpacks them. */
        unsigned char *scales = target + 4;
        for (int j = 0; j < 4; j++) {
            scales[j] = (unsigned char)(scale_bits[j] |
                        ((scale_bits[j + 4] >> 4) << 6));
            scales[j + 4] = (unsigned char)(min_bits[j] |
                            ((min_bits[j + 4] >> 4) << 6));
            scales[j + 8] = (unsigned char)((scale_bits[j + 4] & 0x0fu) |
                            ((min_bits[j + 4] & 0x0fu) << 4));
        }
        unsigned char *quantized = target + 16;
        for (int pair = 0; pair < 4; pair++) {
            int low_block = pair * 2, high_block = low_block + 1;
            float step_low = d * scale_bits[low_block];
            float step_high = d * scale_bits[high_block];
            float offset_low = dmin * min_bits[low_block];
            float offset_high = dmin * min_bits[high_block];
            for (int i = 0; i < 32; i++) {
                long low = step_low > 0.0f
                    ? lrintf((source[low_block * 32 + i] + offset_low) / step_low) : 0;
                long high = step_high > 0.0f
                    ? lrintf((source[high_block * 32 + i] + offset_high) / step_high) : 0;
                if (low < 0) low = 0;
                if (low > 15) low = 15;
                if (high < 0) high = 0;
                if (high > 15) high = 15;
                quantized[pair * 32 + i] =
                    (unsigned char)((unsigned)low | ((unsigned)high << 4));
            }
        }
    }
    return 0;
}

int ingot_q4_k_matvec_scalar(const void *weights, size_t rows, size_t cols,
                                  const float *input, float *output) {
    if (weights == NULL || input == NULL || output == NULL || rows == 0 ||
        cols == 0 || cols % INGOT_QK_K != 0) return -1;
    size_t blocks_per_row = cols / INGOT_QK_K;
    if (blocks_per_row > SIZE_MAX / INGOT_Q4_K_BYTES) return -1;
    size_t row_bytes = blocks_per_row * INGOT_Q4_K_BYTES;
    if (rows > SIZE_MAX / row_bytes) return -1;
    const unsigned char *source = (const unsigned char *)weights;
    for (size_t row = 0; row < rows; row++) {
        float sum = 0.0f;
        const unsigned char *row_data = source + row * row_bytes;
        for (size_t block = 0; block < blocks_per_row; block++)
            sum += dot_block(row_data + block * INGOT_Q4_K_BYTES,
                             input + block * INGOT_QK_K);
        output[row] = sum;
    }
    return 0;
}

#if defined(INGOT_HAVE_Q4_K_NEON)
static float32x4_t q4_k_u8x8_to_f32(uint8x8_t values, int high) {
    uint16x8_t values16 = vmovl_u8(values);
    uint32x4_t values32 = vmovl_u16(high ? vget_high_u16(values16) :
                                             vget_low_u16(values16));
    return vcvtq_f32_u32(values32);
}

/* Forward declaration for the single-row dot block used by dual-row remainder */
static float q4_k_dot_block_neon(const unsigned char *block, const float *input);

/* Process one Q4_K block: decode all 256 quants into 256 float values.
 * Uses NEON to deinterleave nibbles at 16 samples/iteration. */
static void q4_k_dequant_block_neon(const unsigned char *block, float *output,
                                     float d, float dmin,
                                     const unsigned char *scales) {
    for (int base = 0, scale_index = 0; base < INGOT_QK_K; base += 64, scale_index += 2) {
        unsigned char scale0, scale1, min0, min1;
        scale_min(scales, scale_index, &scale0, &min0);
        scale_min(scales, scale_index + 1, &scale1, &min1);
        float32x4_t d0v = vdupq_n_f32(d * scale0);
        float32x4_t d1v = vdupq_n_f32(d * scale1);
        float32x4_t m0v = vdupq_n_f32(dmin * min0);
        float32x4_t m1v = vdupq_n_f32(dmin * min1);
        /* 32 nibbles → 64 values, 8 at a time with NEON */
        for (int i = 0; i < 32; i += 8) {
            uint8x8_t packed = vld1_u8(block + 16 + (base/64) * 32 + i);
            /* Low nibbles (even positions, 0..31) */
            uint8x8_t lo = vand_u8(packed, vdup_n_u8(0x0f));
            uint16x8_t lo16 = vmovl_u8(lo);
            uint32x4_t lo32_0 = vmovl_u16(vget_low_u16(lo16));
            uint32x4_t lo32_1 = vmovl_u16(vget_high_u16(lo16));
            float32x4_t lo0 = vcvtq_f32_u32(lo32_0);
            float32x4_t lo1 = vcvtq_f32_u32(lo32_1);
            /* High nibbles (odd positions, 32..63) */
            uint8x8_t hi = vshr_n_u8(packed, 4);
            uint16x8_t hi16 = vmovl_u8(hi);
            uint32x4_t hi32_0 = vmovl_u16(vget_low_u16(hi16));
            uint32x4_t hi32_1 = vmovl_u16(vget_high_u16(hi16));
            float32x4_t hi0 = vcvtq_f32_u32(hi32_0);
            float32x4_t hi1 = vcvtq_f32_u32(hi32_1);
            /* Dequantize: d * scale * quant - dmin * min */
            vst1q_f32(output + base + i,     vsubq_f32(vmulq_f32(lo0, d0v), m0v));
            vst1q_f32(output + base + i + 4, vsubq_f32(vmulq_f32(lo1, d0v), m0v));
            vst1q_f32(output + base + i + 32, vsubq_f32(vmulq_f32(hi0, d1v), m1v));
            vst1q_f32(output + base + i + 36, vsubq_f32(vmulq_f32(hi1, d1v), m1v));
        }
    }
}

/* Dual-row Q4_K matvec: process 2 adjacent rows sharing the same input vector.
 * Halves input memory traffic and amortises control overhead. */
/* Q4_K rows, each decoded straight into the accumulator.
 *
 * This replaced a "dual" version that processed two rows per iteration and
 * decoded the FIRST of them into a 256-float scratch array before dotting it,
 * while the second went directly into accumulators — so half the rows paid a
 * 1 KB write-and-reread per super-block, and the two halves were ~90 lines of
 * duplicated logic that could drift apart.
 *
 * The same pattern is what made Q6_K 2.8x slower per element than it needed to
 * be. Measured here, best of three on an M-series Mac: [2048 x 1024]
 * 0.55 -> 0.46 ms and [3072 x 1024] 0.84 -> 0.70 ms, about 20%, while deleting
 * the duplicated half. */
static int ingot_q4_k_matvec_dual_neon(const void *weights, size_t rows,
    size_t cols, const float *input, float *output) {
    size_t blocks_per_row = cols / INGOT_QK_K;
    size_t row_bytes = blocks_per_row * INGOT_Q4_K_BYTES;
    const unsigned char *source = (const unsigned char *)weights;
    for (size_t row = 0; row < rows; row++) {
        const unsigned char *row_data = source + row * row_bytes;
        float sum = 0.0f;
        for (size_t b = 0; b < blocks_per_row; b++)
            sum += q4_k_dot_block_neon(row_data + b * INGOT_Q4_K_BYTES,
                                       input + b * INGOT_QK_K);
        output[row] = sum;
    }
    return 0;
}

/* Distribute the sum rather than materializing every weight.
 *
 *     SUM_j w_j x_j  =  d*scale * SUM_j (q_j x_j)  -  dmin*min * SUM_j x_j
 *
 * The form this replaces built each weight first — multiply by d*scale,
 * subtract dmin*min, THEN the multiply-add that matters — which is two vector
 * ops per four values spent on arithmetic the 32-weight sub-block could share.
 * Here the quants accumulate against the input, the inputs accumulate on their
 * own, and scale and min are applied once per sub-block.
 *
 * Folded into a running vector with vmlaq_n_f32 (min as a negative multiplier,
 * NEON has no vmlsq_n_f32), so a block still costs one horizontal reduction.
 * Fewer roundings than before, not more: the min is one subtraction per
 * sub-block rather than one per weight. Same identity as Q5_K above.
 *
 * NOTE for anyone measuring against a consumer's own Q4_K kernel: the SUM x_j
 * term does not depend on the row, so an engine that owns the whole matvec can
 * hoist it out of the row loop entirely and win again on top of this. That
 * needs a per-call preamble the row kernel reads, which is a different API
 * shape than this one — see mynah-slm's src/qmat.c. */
static float q4_k_dot_block_neon(const unsigned char *block, const float *input) {
    float d = f16_to_f32(read_u16(block));
    float dmin = f16_to_f32(read_u16(block + 2));
    const unsigned char *scales = block + 4;
    const unsigned char *quantized = block + 16;
    float32x4_t total = vdupq_n_f32(0.0f);
    const uint8x8_t nib = vdup_n_u8(0x0f);

    for (int base = 0, scale_index = 0; base < INGOT_QK_K; base += 64, scale_index += 2) {
        unsigned char scale0, scale1, min0, min1;
        scale_min(scales, scale_index, &scale0, &min0);
        scale_min(scales, scale_index + 1, &scale1, &min1);

        float32x4_t q0 = vdupq_n_f32(0.0f), q1 = vdupq_n_f32(0.0f);
        float32x4_t x0 = vdupq_n_f32(0.0f), x1 = vdupq_n_f32(0.0f);

        for (int i = 0; i < 32; i += 8) {
            const uint8x8_t packed = vld1_u8(quantized + i);
            const uint8x8_t lo = vand_u8(packed, nib);
            const uint8x8_t hi = vshr_n_u8(packed, 4);

            const float32x4_t xa = vld1q_f32(input + base + i);
            const float32x4_t xb = vld1q_f32(input + base + i + 4);
            const float32x4_t xc = vld1q_f32(input + base + i + 32);
            const float32x4_t xd = vld1q_f32(input + base + i + 36);

            q0 = vmlaq_f32(q0, q4_k_u8x8_to_f32(lo, 0), xa);
            q0 = vmlaq_f32(q0, q4_k_u8x8_to_f32(lo, 1), xb);
            q1 = vmlaq_f32(q1, q4_k_u8x8_to_f32(hi, 0), xc);
            q1 = vmlaq_f32(q1, q4_k_u8x8_to_f32(hi, 1), xd);
            x0 = vaddq_f32(x0, vaddq_f32(xa, xb));
            x1 = vaddq_f32(x1, vaddq_f32(xc, xd));
        }

        total = vmlaq_n_f32(total, q0, d * scale0);
        total = vmlaq_n_f32(total, q1, d * scale1);
        total = vmlaq_n_f32(total, x0, -(dmin * min0));
        total = vmlaq_n_f32(total, x1, -(dmin * min1));
        quantized += 32;
    }
    return vaddvq_f32(total);
}

#endif

#if defined(INGOT_HAVE_Q4_K_AVX2)
/* The AVX2 twin of the kernel above, same distribute-the-sum identity. */
static float q4_k_dot_block_avx2(const unsigned char *block, const float *input) {
    float d = f16_to_f32(read_u16(block));
    float dmin = f16_to_f32(read_u16(block + 2));
    const unsigned char *scales = block + 4;
    const unsigned char *quantized = block + 16;
    __m256 total = _mm256_setzero_ps();

    for (int base = 0, scale_index = 0; base < INGOT_QK_K; base += 64, scale_index += 2) {
        unsigned char scale0, scale1, min0, min1;
        scale_min(scales, scale_index, &scale0, &min0);
        scale_min(scales, scale_index + 1, &scale1, &min1);

        __m256 q0 = _mm256_setzero_ps(), q1 = _mm256_setzero_ps();
        __m256 x0 = _mm256_setzero_ps(), x1 = _mm256_setzero_ps();

        for (int i = 0; i < 32; i += 8) {
            const __m256i packed = _mm256_cvtepu8_epi32(
                _mm_loadl_epi64((const __m128i *)(const void *)(quantized + i)));
            const __m256 low  = _mm256_cvtepi32_ps(
                _mm256_and_si256(packed, _mm256_set1_epi32(15)));
            const __m256 high = _mm256_cvtepi32_ps(_mm256_srli_epi32(packed, 4));

            const __m256 xa = _mm256_loadu_ps(input + base + i);
            const __m256 xc = _mm256_loadu_ps(input + base + i + 32);

            q0 = _mm256_fmadd_ps(low, xa, q0);
            q1 = _mm256_fmadd_ps(high, xc, q1);
            x0 = _mm256_add_ps(x0, xa);
            x1 = _mm256_add_ps(x1, xc);
        }

        total = _mm256_fmadd_ps(q0, _mm256_set1_ps(d * scale0), total);
        total = _mm256_fmadd_ps(q1, _mm256_set1_ps(d * scale1), total);
        total = _mm256_fmadd_ps(x0, _mm256_set1_ps(-(dmin * min0)), total);
        total = _mm256_fmadd_ps(x1, _mm256_set1_ps(-(dmin * min1)), total);
        quantized += 32;
    }
    {
        __m128 h = _mm_add_ps(_mm256_castps256_ps128(total),
                              _mm256_extractf128_ps(total, 1));
        h = _mm_add_ps(h, _mm_movehl_ps(h, h));
        h = _mm_add_ss(h, _mm_shuffle_ps(h, h, 0x55));
        return _mm_cvtss_f32(h);
    }
}

static int ingot_q4_k_matvec_avx2(const void *weights, size_t rows, size_t cols,
                                       const float *input, float *output) {
    if (weights == NULL || input == NULL || output == NULL || rows == 0 ||
        cols == 0 || cols % INGOT_QK_K != 0) return -1;
    size_t blocks_per_row = cols / INGOT_QK_K;
    if (blocks_per_row > SIZE_MAX / INGOT_Q4_K_BYTES) return -1;
    size_t row_bytes = blocks_per_row * INGOT_Q4_K_BYTES;
    if (rows > SIZE_MAX / row_bytes) return -1;
    const unsigned char *source = (const unsigned char *)weights;
    for (size_t row = 0; row < rows; row++) {
        const unsigned char *row_data = source + row * row_bytes;
        float sum = 0.0f;
        for (size_t block = 0; block < blocks_per_row; block++)
            sum += q4_k_dot_block_avx2(row_data + block * INGOT_Q4_K_BYTES,
                                       input + block * INGOT_QK_K);
        output[row] = sum;
    }
    return 0;
}
#endif

/* The third belt, after the build define and the runtime check: the compiler
 * must actually target dotprod. On Linux the default is plain armv8-a, and
 * without this guard the define alone would break compilation of the vdotq
 * intrinsics. Apple clang defines it by default. */
#if defined(INGOT_HAVE_Q4_K_NEON) && defined(__ARM_FEATURE_DOTPROD)
#define INGOT_HAVE_Q4_K_SDOT 1

/* ── SDOT path: integer dot products against int8 activations ─────────────
 * Q4_K decodes as w[i] = d*scale_j*q[i] - dmin*min_j, with q in 0..15 and
 * (scale_j, min_j) constant over each 32-weight sub-block j.  Quantising the
 * activation to int8 (x[i] ~ sx * qx[i]) turns the dot into
 *
 *   row . x  ~  sx * SUM_j [ d*scale_j * <q_j, qx_j> - dmin*min_j * SUM(qx_j) ]
 *
 * so the inner loop is integer-only: <q_j, qx_j> rides vdotq_s32 on the raw
 * nibbles (0..15 needs no bias correction, unlike q4_0's -8 offset) and the
 * per-sub-block activation sums are computed ONCE for the whole matrix,
 * shared by every row.  Only the 8 per-sub-block scalars per block stay in
 * float, which is where the "decode ALU dominates int4" trap of the f32 path
 * disappears.
 *
 * Precision: int8 activations are an approximation, so this is NOT the exact
 * f32 reference the parity gates compare against — it is opt-in, see
 * ingot_q4_k_matvec. */

/* Quantises one 256-element activation super-block: absmax scale, int8
 * values, and the four 64-element sub-block sums the min term needs.  Returns
 * the scale (0 when the block is all zeros). */
static float q4_k_quantize_activation(const float *input, int8_t *quantized,
                                      int32_t sums[8]) {
    float32x4_t amax4 = vdupq_n_f32(0.0f);
    for (int i = 0; i < INGOT_QK_K; i += 4)
        amax4 = vmaxq_f32(amax4, vabsq_f32(vld1q_f32(input + i)));
    float amax = vmaxvq_f32(amax4);
    if (!(amax > 0.0f)) {
        memset(quantized, 0, INGOT_QK_K);
        memset(sums, 0, 8 * sizeof(*sums));
        return 0.0f;
    }
    float scale = amax / 127.0f;
    float inverse = 127.0f / amax;
    for (int group = 0; group < 8; group++) {
        int32x4_t sum4 = vdupq_n_s32(0);
        for (int i = 0; i < 32; i += 8) {
            int32x4_t lo = vcvtnq_s32_f32(
                vmulq_n_f32(vld1q_f32(input + group * 32 + i), inverse));
            int32x4_t hi = vcvtnq_s32_f32(
                vmulq_n_f32(vld1q_f32(input + group * 32 + i + 4), inverse));
            sum4 = vaddq_s32(sum4, vaddq_s32(lo, hi));
            int16x8_t packed = vcombine_s16(vqmovn_s32(lo), vqmovn_s32(hi));
            vst1_s8(quantized + group * 32 + i, vqmovn_s16(packed));
        }
        sums[group] = vaddvq_s32(sum4);
    }
    return scale;
}

/* Dot of one Q4_K super-block against the pre-quantised activation. */
static float q4_k_dot_block_sdot(const unsigned char *block,
                                 const int8_t *quantized, float activation_scale,
                                 const int32_t sums[8]) {
    float d = f16_to_f32(read_u16(block));
    float dmin = f16_to_f32(read_u16(block + 2));
    const unsigned char *scales = block + 4;
    const unsigned char *packed = block + 16;
    const uint8x16_t mask = vdupq_n_u8(0x0f);
    float total = 0.0f;
    /* Sub-block layout: group g holds weights [g*64, g*64+32) in the low
     * nibbles and [g*64+32, g*64+64) in the high nibbles, with scale indices
     * 2g and 2g+1 — the same pairing the f32 kernels use. */
    for (int group = 0; group < 4; group++) {
        unsigned char scale_low, scale_high, min_low, min_high;
        scale_min(scales, group * 2, &scale_low, &min_low);
        scale_min(scales, group * 2 + 1, &scale_high, &min_high);
        int32x4_t dot_low = vdupq_n_s32(0);
        int32x4_t dot_high = vdupq_n_s32(0);
        for (int i = 0; i < 32; i += 16) {
            uint8x16_t nibbles = vld1q_u8(packed + group * 32 + i);
            int8x16_t low = vreinterpretq_s8_u8(vandq_u8(nibbles, mask));
            int8x16_t high = vreinterpretq_s8_u8(vshrq_n_u8(nibbles, 4));
            dot_low = vdotq_s32(dot_low, low,
                                vld1q_s8(quantized + group * 64 + i));
            dot_high = vdotq_s32(dot_high, high,
                                 vld1q_s8(quantized + group * 64 + 32 + i));
        }
        total += d * (float)scale_low * (float)vaddvq_s32(dot_low) -
                 dmin * (float)min_low * (float)sums[group * 2];
        total += d * (float)scale_high * (float)vaddvq_s32(dot_high) -
                 dmin * (float)min_high * (float)sums[group * 2 + 1];
    }
    return total * activation_scale;
}

static int ingot_q4_k_matvec_sdot(const void *weights, size_t rows,
    size_t cols, const float *input, float *output) {
    if (weights == NULL || input == NULL || output == NULL || rows == 0 ||
        cols == 0 || cols % INGOT_QK_K != 0) return -1;
    size_t blocks_per_row = cols / INGOT_QK_K;
    if (blocks_per_row > SIZE_MAX / INGOT_Q4_K_BYTES) return -1;
    size_t row_bytes = blocks_per_row * INGOT_Q4_K_BYTES;
    if (rows > SIZE_MAX / row_bytes) return -1;
    /* The activation is quantised once and reused by every row; that is the
     * whole point of the format, so bail out rather than fall back if the
     * scratch cannot be had. */
    int8_t *quantized = malloc(cols);
    float *scales = malloc(blocks_per_row * sizeof(*scales));
    int32_t *sums = malloc(blocks_per_row * 8 * sizeof(*sums));
    if (quantized == NULL || scales == NULL || sums == NULL) {
        free(quantized); free(scales); free(sums);
        return -1;
    }
    for (size_t block = 0; block < blocks_per_row; block++)
        scales[block] = q4_k_quantize_activation(
            input + block * INGOT_QK_K, quantized + block * INGOT_QK_K,
            sums + block * 8);
    const unsigned char *source = (const unsigned char *)weights;
    for (size_t row = 0; row < rows; row++) {
        const unsigned char *row_data = source + row * row_bytes;
        float sum = 0.0f;
        for (size_t block = 0; block < blocks_per_row; block++)
            sum += q4_k_dot_block_sdot(row_data + block * INGOT_Q4_K_BYTES,
                                       quantized + block * INGOT_QK_K,
                                       scales[block], sums + block * 8);
        output[row] = sum;
    }
    free(quantized); free(scales); free(sums);
    return 0;
}
#endif

/* ── x86 int8: the VNNI / AVX2 twin of the SDOT path ────────────────────────
 * Same contract, same flat weight layout, same epilogue math as the ARM
 * section above. The extractors hand the quants over as UNSIGNED bytes
 * (0..15 for Q4_K, 0..31 for Q5_K), which is exactly the first operand
 * VPDPBUSD wants — no +128 bias trick, no sign-sum correction, unlike the
 * signed-weight kernels this pattern usually appears in. Where AVX-512VNNI
 * is not compiled the same dot rides VPMADDUBSW+VPMADDWD; the worst pair sum
 * is 31*127*2 = 7874, far from the i16 saturation edge, so both forms are
 * exact integer dots. The only approximation in the whole path is the
 * activation quantization, identical to ARM's (absmax/127, RNE). */
#if defined(INGOT_HAVE_Q4_K_AVX2)
#define INGOT_HAVE_QK_X86INT8 1
#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
#define INGOT_HAVE_QK_VNNI 1
#endif

static inline int32_t qk_hsum_epi32(__m256i v) {
    __m128i s = _mm_add_epi32(_mm256_castsi256_si128(v),
                              _mm256_extracti128_si256(v, 1));
    s = _mm_add_epi32(s, _mm_srli_si128(s, 8));
    s = _mm_add_epi32(s, _mm_srli_si128(s, 4));
    return _mm_cvtsi128_si32(s);
}

static inline __m256i qk_dot_u8s8(__m256i acc, __m256i w, __m256i x) {
#if defined(INGOT_HAVE_QK_VNNI)
    return _mm256_dpbusd_epi32(acc, w, x);
#else
    return _mm256_add_epi32(acc,
        _mm256_madd_epi16(_mm256_maddubs_epi16(w, x), _mm256_set1_epi16(1)));
#endif
}

/* The activation quantizer, semantics identical to the NEON one: absmax
 * scale, round-to-nearest-even (CVTPS2DQ under default MXCSR, matching
 * vcvtnq), saturating packs, and the eight 32-wide integer sums. */
static float q4_k_quantize_activation(const float *input, int8_t *quantized,
                                      int32_t sums[8]) {
    const __m256 absmask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
    __m256 amax8 = _mm256_setzero_ps();
    for (int i = 0; i < INGOT_QK_K; i += 8)
        amax8 = _mm256_max_ps(amax8,
            _mm256_and_ps(_mm256_loadu_ps(input + i), absmask));
    __m128 m = _mm_max_ps(_mm256_castps256_ps128(amax8),
                          _mm256_extractf128_ps(amax8, 1));
    m = _mm_max_ps(m, _mm_movehl_ps(m, m));
    m = _mm_max_ss(m, _mm_movehdup_ps(m));
    const float amax = _mm_cvtss_f32(m);
    if (!(amax > 0.0f)) {
        memset(quantized, 0, INGOT_QK_K);
        memset(sums, 0, 8 * sizeof(*sums));
        return 0.0f;
    }
    const float scale = amax / 127.0f;
    const __m256 inverse = _mm256_set1_ps(127.0f / amax);
    for (int group = 0; group < 8; group++) {
        __m256i sum8 = _mm256_setzero_si256();
        for (int i = 0; i < 32; i += 8) {
            const __m256i q = _mm256_cvtps_epi32(_mm256_mul_ps(
                _mm256_loadu_ps(input + group * 32 + i), inverse));
            sum8 = _mm256_add_epi32(sum8, q);
            const __m128i p16 = _mm_packs_epi32(_mm256_castsi256_si128(q),
                                                _mm256_extracti128_si256(q, 1));
            _mm_storel_epi64((__m128i *)(quantized + group * 32 + i),
                             _mm_packs_epi16(p16, p16));
        }
        sums[group] = qk_hsum_epi32(sum8);
    }
    return scale;
}

/* Dot of one Q4_K super-block against the pre-quantized activation — the
 * VPDPBUSD rendition of q4_k_dot_block_sdot, same sub-block pairing. */
static float q4_k_dot_block_x86(const unsigned char *block,
                                const int8_t *quantized, float activation_scale,
                                const int32_t sums[8]) {
    const float d = f16_to_f32(read_u16(block));
    const float dmin = f16_to_f32(read_u16(block + 2));
    const unsigned char *scales = block + 4;
    const unsigned char *packed = block + 16;
    const __m256i mask = _mm256_set1_epi8(0x0f);
    float total = 0.0f;
    for (int group = 0; group < 4; group++) {
        unsigned char scale_low, scale_high, min_low, min_high;
        scale_min(scales, group * 2, &scale_low, &min_low);
        scale_min(scales, group * 2 + 1, &scale_high, &min_high);
        const __m256i nib =
            _mm256_loadu_si256((const __m256i *)(packed + group * 32));
        const __m256i low = _mm256_and_si256(nib, mask);
        const __m256i high = _mm256_and_si256(_mm256_srli_epi16(nib, 4), mask);
        const __m256i dot_low = qk_dot_u8s8(_mm256_setzero_si256(), low,
            _mm256_loadu_si256((const __m256i *)(quantized + group * 64)));
        const __m256i dot_high = qk_dot_u8s8(_mm256_setzero_si256(), high,
            _mm256_loadu_si256((const __m256i *)(quantized + group * 64 + 32)));
        total += d * (float)scale_low * (float)qk_hsum_epi32(dot_low) -
                 dmin * (float)min_low * (float)sums[group * 2];
        total += d * (float)scale_high * (float)qk_hsum_epi32(dot_high) -
                 dmin * (float)min_high * (float)sums[group * 2 + 1];
    }
    return total * activation_scale;
}

static int ingot_q4_k_matvec_x86int8(const void *weights, size_t rows,
    size_t cols, const float *input, float *output) {
    if (weights == NULL || input == NULL || output == NULL || rows == 0 ||
        cols == 0 || cols % INGOT_QK_K != 0) return -1;
    size_t blocks_per_row = cols / INGOT_QK_K;
    if (blocks_per_row > SIZE_MAX / INGOT_Q4_K_BYTES) return -1;
    size_t row_bytes = blocks_per_row * INGOT_Q4_K_BYTES;
    if (rows > SIZE_MAX / row_bytes) return -1;
    int8_t *quantized = malloc(cols);
    float *scales = malloc(blocks_per_row * sizeof(*scales));
    int32_t *sums = malloc(blocks_per_row * 8 * sizeof(*sums));
    if (quantized == NULL || scales == NULL || sums == NULL) {
        free(quantized); free(scales); free(sums);
        return -1;
    }
    for (size_t block = 0; block < blocks_per_row; block++)
        scales[block] = q4_k_quantize_activation(
            input + block * INGOT_QK_K, quantized + block * INGOT_QK_K,
            sums + block * 8);
    const unsigned char *source = (const unsigned char *)weights;
    for (size_t row = 0; row < rows; row++) {
        const unsigned char *row_data = source + row * row_bytes;
        float sum = 0.0f;
        for (size_t block = 0; block < blocks_per_row; block++)
            sum += q4_k_dot_block_x86(row_data + block * INGOT_Q4_K_BYTES,
                                      quantized + block * INGOT_QK_K,
                                      scales[block], sums + block * 8);
        output[row] = sum;
    }
    free(quantized); free(scales); free(sums);
    return 0;
}
#endif /* x86 int8 */

/* Whether the int8 fast path is real on this machine: compiled in AND the
 * instruction set is there at runtime. One predicate shared by the matvec
 * opt-in, the batched default and ingot_matmat_is_exact, so the answer the
 * library gives is the path it actually takes. */
#if defined(INGOT_HAVE_Q4_K_SDOT) || defined(INGOT_HAVE_QK_X86INT8)
static int qk_int8_ready(void) {
#if defined(INGOT_HAVE_Q4_K_SDOT)
    return ingot_cpu().dotprod;
#elif defined(INGOT_HAVE_QK_VNNI)
    return ingot_cpu().avx512_vnni;
#else
    return ingot_cpu().avx2;
#endif
}
#endif

int ingot_q4_k_matvec_sdot_available(void) {
#if defined(INGOT_HAVE_Q4_K_SDOT)
    return ingot_cpu().dotprod ? 1 : 0;
#else
    return 0;
#endif
}

int ingot_q4_k_matvec_int8(const void *weights, size_t rows, size_t cols,
                                const float *input, float *output) {
    if (!qk_args_ok(weights, input, output, rows, cols, INGOT_QK_K)) return -1;
#if defined(INGOT_HAVE_Q4_K_SDOT)
    if (ingot_cpu().dotprod)
        return ingot_q4_k_matvec_sdot(weights, rows, cols, input, output);
#elif defined(INGOT_HAVE_QK_X86INT8)
    if (qk_int8_ready())
        return ingot_q4_k_matvec_x86int8(weights, rows, cols, input, output);
#endif
    return ingot_q4_k_matvec(weights, rows, cols, input, output);
}

int ingot_q4_k_matvec(const void *weights, size_t rows, size_t cols,
                           const float *input, float *output) {
    if (!qk_args_ok(weights, input, output, rows, cols, INGOT_QK_K)) return -1;
    ingot_cpu_caps caps = ingot_cpu();
    (void)caps;
#if defined(INGOT_HAVE_Q4_K_SDOT) || defined(INGOT_HAVE_QK_X86INT8)
    /* Opt-in, and it stays opt-in even though the batched path went int8 by
     * default: THIS function is the reference. Parity is measured against it,
     * and the batch gate requires `tokens == 1` to match it bit for bit, so
     * the approximation only enters here when asked for explicitly with
     * `INGOT_SDOT=1`. */
    static int sdot_env = -1;
    if (sdot_env < 0) {
        const char *value = getenv("INGOT_SDOT");
        sdot_env = value != NULL && value[0] == '1';
    }
    if (sdot_env && qk_int8_ready())
#if defined(INGOT_HAVE_Q4_K_SDOT)
        return ingot_q4_k_matvec_sdot(weights, rows, cols, input, output);
#else
        return ingot_q4_k_matvec_x86int8(weights, rows, cols, input, output);
#endif
#endif
#if defined(INGOT_HAVE_Q4_K_NEON)
    if (caps.neon)
        return ingot_q4_k_matvec_dual_neon(weights, rows, cols, input, output);
#endif
#if defined(INGOT_HAVE_Q4_K_AVX2)
    if (caps.avx2)
        return ingot_q4_k_matvec_avx2(weights, rows, cols, input, output);
#endif
    return ingot_q4_k_matvec_scalar(weights, rows, cols, input, output);
}

/* --- Batched quantised GEMM (the CPU mirror of the Metal/CUDA tiles) -------
 *
 * The batched path used to be a parallel_for over tokens calling the matvec
 * once per token, so a 4-bit matrix was decoded `tokens` times: the decode is
 * the expensive half of the kernel, and that is why the CPU forward grew by
 * ~0.85 s for every extra token (measured).  Here the nesting is inverted the
 * way a Metal tile or any blocked GEMM does it: a strip of rows is decoded
 * once into f32 and dotted against every token of a tile, so the decode cost
 * is divided by the tile width.
 *
 * Three levels, chosen so the working set stays in L1 on an M1 firestorm core
 * (128 KB): a token tile (INGOT_QK_TOKEN_TILE activations, 64 KB per k-block),
 * a row strip (INGOT_QK_ROW_TILE decoded rows, 8 KB), and the 256-wide
 * super-block itself.  Accumulators are INGOT_QK_ROW_TILE x tile floats.
 *
 * Numerics: same f32 decode as the matvec, but summed per super-block instead
 * of in one running accumulator, so results differ from
 * ingot_q4_k_matvec in the last ulps.  The matvec stays the reference the
 * GPU parity gates measure against. */
#define INGOT_QK_ROW_TILE 8
#define INGOT_QK_TOKEN_TILE 128
#define INGOT_QK_TOKEN_TILE_MIN 32
#define INGOT_QK_TOKEN_TILE_MAX 512

/* The tile width sets how often the matrix is decoded (once per tile) against
 * how much of the activation block stays resident while every row strip sweeps
 * it: tile * cols * 4 bytes, against 12 MB of shared L2 on an M1.
 *
 * That PRODUCT is what has to fit, so the tile is derived from cols instead of
 * being a constant. Measured on an M1 on AC power, 512 tokens, max of 9
 * interleaved runs (GMAC/s — an earlier battery-powered run of the same sweep
 * sat inside a ±25% noise band and two of its deltas did not survive):
 *
 *      cols   tile 64   tile 128   tile 256   tile 512     tile*cols*4 at 128
 *      6144     145.9      154.5      157.6      120.0     3 MB
 *     16384     143.2      140.8       83.6       46.1     8 MB
 *
 * Below the knee the choice is noise (64/128 at 16384 differ by 1.7%, under the
 * 4% verdict floor; 128/256 at 6144 likewise); past it the cliff is real — 256
 * loses 40% at cols=16384 and at cols=32768 the 4 MB rule beats a constant 128
 * by +40% (128.5 vs 91.7). That cliff is the point: the constant was fine on
 * the shapes krea2 uses today and wrong the first time a caller passes a wider
 * matrix. INGOT_QK_TILE=<n> still overrides, so it stays re-measurable
 * elsewhere: `make bench-gguf` with INGOT_BENCH_TOKENS large enough to
 * span more than one tile. */
#define INGOT_QK_TILE_BYTES (4u << 20)

static size_t qk_token_tile(size_t cols) {
    static long override = -1;
    if (override < 0) {
        override = 0;
        const char *value = getenv("INGOT_QK_TILE");
        if (value != NULL && value[0] != '\0') {
            char *end = NULL;
            unsigned long parsed = strtoul(value, &end, 10);
            if (end != value && *end == '\0' && parsed > 0 &&
                parsed <= INGOT_QK_TOKEN_TILE_MAX) override = (long)parsed;
        }
    }
    size_t tile;
    if (override > 0) {
        tile = (size_t)override;
    } else {
        tile = INGOT_QK_TILE_BYTES / (cols * sizeof(float));
        if (tile > INGOT_QK_TOKEN_TILE) tile = INGOT_QK_TOKEN_TILE;
        if (tile < INGOT_QK_TOKEN_TILE_MIN) tile = INGOT_QK_TOKEN_TILE_MIN;
    }
    /* The SMMLA worker indexes interleaved token PAIRS as
     * (token_begin + t) / 2, which is only right when every tile starts on
     * an even token. The derived tile can be odd (cols=11008 gives 95), and
     * so can the env override: round down to even, floor 2. */
    tile &= ~(size_t)1;
    if (tile < 2) tile = 2;
    return tile;
}

/* ── Lo stride delle attivazioni: quale loop aliasa, e su quale path ──────
 *
 * A tile sweeps `tile` tokens for every row strip, so the addresses that matter
 * are token t at t * stride, for t across the WHOLE tile — not just the four
 * tokens the innermost loop reads together. A Firestorm L1D is 128 KB / 8 ways
 * / 64 B lines = 256 sets, so token t lands in set (t * stride / 64) mod 256:
 * when stride/64 shares a large factor with 256, the tile's tokens pile onto
 * 256/gcd(stride/64, 256) set groups and thrash against 8 ways.
 *
 * Measured on the int8 path, M1, 1024x6144x128, matmul phase alone, best of
 * five — monotone in the number of set groups, which is the whole story:
 *
 *     set groups     4      8      16     >=32
 *     matmul ms    3.77   3.63   2.92   2.60-2.67
 *
 * cols is a multiple of 256, so an unpadded int8 stride (cols bytes) always
 * gives gcd >= 32, i.e. <= 8 groups: the bad end. `cols + 64` makes stride/64
 * equal to cols/64 + 1, odd for any legal cols, hence gcd 1 and all 256 groups
 * — provably the best case rather than a tuned constant. Wide sweep: every slow
 * stride was a multiple of 2048 (6144, 8192, 10240, 14336, 22528) and no other
 * was, and the plateau runs flat from +16 to +256 bytes.
 *
 * The f32 twin does NOT have this problem, which is why packing its activations
 * into a padded scratch was a 4.6% loss (the copy, and nothing bought). Its
 * stride is cols*4 bytes, which is far WORSE on paper — cols=6144 gives 2 set
 * groups, cols=8192 gives 1 — and yet sweeping cols over 1, 2, 4, 8 and 16
 * groups moves it not at all: 161-166 GMAC/s, flat. At 164 GMAC/s the f32
 * kernel is bound by decoding the weights, not by loading activations; the int8
 * kernel deleted the decode, and that is what EXPOSED the aliasing at ~300. The
 * lesson worth keeping: a latency this cache-shaped only becomes visible once
 * the thing that was hiding it is gone, so a padding experiment that fails is
 * evidence about which term dominates, not about the padding. */

/* acc[row][token] += dot(decoded_row, activation_token) over one super-block.
 * `act` points at the block slice of token 0; tokens are act_stride apart. */
static void qk_gemm_block(const float *decoded, size_t decoded_rows,
                          const float *act, size_t act_stride, size_t tokens,
                          float *acc, size_t acc_stride) {
#if defined(INGOT_HAVE_Q4_K_NEON)
    size_t row = 0;
    for (; row + 4 <= decoded_rows; row += 4) {
        const float *w0 = decoded + row * INGOT_QK_K;
        const float *w1 = w0 + INGOT_QK_K;
        const float *w2 = w1 + INGOT_QK_K;
        const float *w3 = w2 + INGOT_QK_K;
        size_t token = 0;
        for (; token + 4 <= tokens; token += 4) {
            const float *x = act + token * act_stride;
            /* 4 rows x 4 tokens: 8 loads feed 16 FMAs. The 2x4 shape measured
             * first only reached 4:3, and the M1 has 4 FP pipes against 3 load
             * ports; 16 accumulators + 8 operands still fit the 32 registers. */
            float32x4_t a[4][4];
            for (int r = 0; r < 4; r++)
                for (int t = 0; t < 4; t++) a[r][t] = vdupq_n_f32(0.0f);
            for (int i = 0; i < INGOT_QK_K; i += 4) {
                float32x4_t wv[4] = {vld1q_f32(w0 + i), vld1q_f32(w1 + i),
                                     vld1q_f32(w2 + i), vld1q_f32(w3 + i)};
                float32x4_t xv[4] = {vld1q_f32(x + i),
                                     vld1q_f32(x + act_stride + i),
                                     vld1q_f32(x + 2 * act_stride + i),
                                     vld1q_f32(x + 3 * act_stride + i)};
                for (int r = 0; r < 4; r++)
                    for (int t = 0; t < 4; t++)
                        a[r][t] = vfmaq_f32(a[r][t], wv[r], xv[t]);
            }
            for (int r = 0; r < 4; r++)
                for (int t = 0; t < 4; t++)
                    acc[(row + (size_t)r) * acc_stride + token + (size_t)t] +=
                        vaddvq_f32(a[r][t]);
        }
        for (; token < tokens; token++) {
            const float *x = act + token * act_stride;
            float32x4_t a[4];
            for (int r = 0; r < 4; r++) a[r] = vdupq_n_f32(0.0f);
            for (int i = 0; i < INGOT_QK_K; i += 4) {
                float32x4_t xv = vld1q_f32(x + i);
                a[0] = vfmaq_f32(a[0], vld1q_f32(w0 + i), xv);
                a[1] = vfmaq_f32(a[1], vld1q_f32(w1 + i), xv);
                a[2] = vfmaq_f32(a[2], vld1q_f32(w2 + i), xv);
                a[3] = vfmaq_f32(a[3], vld1q_f32(w3 + i), xv);
            }
            for (int r = 0; r < 4; r++)
                acc[(row + (size_t)r) * acc_stride + token] += vaddvq_f32(a[r]);
        }
    }
    for (; row < decoded_rows; row++) {
        const float *w = decoded + row * INGOT_QK_K;
        for (size_t token = 0; token < tokens; token++) {
            const float *x = act + token * act_stride;
            float32x4_t a0 = vdupq_n_f32(0.0f);
            for (int i = 0; i < INGOT_QK_K; i += 4)
                a0 = vfmaq_f32(a0, vld1q_f32(w + i), vld1q_f32(x + i));
            acc[row * acc_stride + token] += vaddvq_f32(a0);
        }
    }
#else
    for (size_t row = 0; row < decoded_rows; row++) {
        const float *w = decoded + row * INGOT_QK_K;
        for (size_t token = 0; token < tokens; token++) {
            const float *x = act + token * act_stride;
            float sum = 0.0f;
            for (int i = 0; i < INGOT_QK_K; i++) sum += w[i] * x[i];
            acc[row * acc_stride + token] += sum;
        }
    }
#endif
}

static void q4_k_decode_block(const unsigned char *block, float *output) {
#if defined(INGOT_HAVE_Q4_K_NEON)
    if (ingot_cpu().neon) {
        q4_k_dequant_block_neon(block, output, f16_to_f32(read_u16(block)),
                                f16_to_f32(read_u16(block + 2)), block + 4);
        return;
    }
#endif
    dequant_block(block, output);
}

typedef void (*qk_decode_fn)(const unsigned char *block, float *output);

typedef struct {
    const unsigned char *weights;
    const float *input;          /* [tokens][cols] row-major */
    float *output;               /* [tokens][rows] row-major */
    size_t rows, cols, tokens;
    size_t blocks_per_row, row_bytes, block_bytes;
    qk_decode_fn decode;
    size_t token_begin, token_count;
} qk_matmat_job_t;

static void qk_matmat_rows(size_t begin, size_t end, void *user) {
    const qk_matmat_job_t *job = user;
    /* alignas: on NEON a 16-byte load off a 16-aligned buffer never straddles a
     * line, so this is free here; on x86 built with -mavx512 a 64-byte load off
     * a 16-aligned one splits a line three times out of four. One word. */
    alignas(64) float decoded[INGOT_QK_ROW_TILE * INGOT_QK_K];
    alignas(64) float acc[INGOT_QK_ROW_TILE * INGOT_QK_TOKEN_TILE_MAX];
    for (size_t strip = begin; strip < end; strip++) {
        size_t row0 = strip * INGOT_QK_ROW_TILE;
        size_t strip_rows = job->rows - row0 < INGOT_QK_ROW_TILE ?
                            job->rows - row0 : INGOT_QK_ROW_TILE;
        memset(acc, 0, strip_rows * job->token_count * sizeof(*acc));
        for (size_t block = 0; block < job->blocks_per_row; block++) {
            for (size_t r = 0; r < strip_rows; r++)
                job->decode(job->weights + (row0 + r) * job->row_bytes +
                                block * job->block_bytes,
                            decoded + r * INGOT_QK_K);
            qk_gemm_block(decoded, strip_rows,
                          job->input + job->token_begin * job->cols +
                              block * INGOT_QK_K,
                          job->cols, job->token_count, acc, job->token_count);
        }
        for (size_t r = 0; r < strip_rows; r++)
            for (size_t t = 0; t < job->token_count; t++)
                job->output[(job->token_begin + t) * job->rows + row0 + r] =
                    acc[r * job->token_count + t];
    }
}

static int qk_matmat(const void *weights, size_t rows, size_t cols,
                     const float *input, float *output, size_t tokens,
                     size_t block_bytes, qk_decode_fn decode) {
    if (weights == NULL || input == NULL || output == NULL || rows == 0 ||
        cols == 0 || tokens == 0 || cols % INGOT_QK_K != 0) return -1;
    size_t blocks_per_row = cols / INGOT_QK_K;
    if (blocks_per_row > SIZE_MAX / block_bytes) return -1;
    size_t row_bytes = blocks_per_row * block_bytes;
    if (rows > SIZE_MAX / row_bytes) return -1;

    qk_matmat_job_t job = {
        (const unsigned char *)weights, input, output, rows, cols, tokens,
        blocks_per_row, row_bytes, block_bytes, decode, 0, 0,
    };
    size_t strips = (rows + INGOT_QK_ROW_TILE - 1) / INGOT_QK_ROW_TILE;
    size_t tile = qk_token_tile(cols);
    for (size_t base = 0; base < tokens; base += tile) {
        job.token_begin = base;
        job.token_count = tokens - base < tile ? tokens - base : tile;
        ingot_parallel_for(strips, qk_matmat_rows, &job);
    }
    return 0;
}

#if defined(INGOT_HAVE_Q4_K_SDOT) || defined(INGOT_HAVE_QK_X86INT8)
/* ── The int8 twin of the batched GEMM (opt-in) ──────────────────────────
 *
 * The f32 form above sits at ~52% of the machine's f32 peak, so the
 * next factor cannot come from scheduling: it has to come from the
 * arithmetic. SDOT does four int8 MACs per lane per instruction, so a
 * 256-wide super-block needs 16 `vdotq` where f32 needs 64 `vfma`.
 *
 * The catch is the epilogue. Q4_K carries a scale and an offset per
 * sub-block of 32, so the integer sums cannot simply be accumulated: eight
 * of them per super-block have to be reduced and scaled separately. Reducing
 * each with `vaddvq` would give the instruction count straight back, so four
 * tokens are processed together and their four accumulators collapse with
 * three `vpaddq` into one vector of four sums — one reduction instead of
 * four, and the scale then applies to the whole vector.
 *
 * Same contract as the matvec twin: the activations become int8, so this is
 * NOT the reference (rel ~2e-3, an approximation rather than a reordering).
 * `INGOT_SDOT=1` opts in, exactly as it does for the matvec. */

/* Q4_K quants into a flat 0..15 array: sub-block j is out[j*32 .. j*32+32),
 * which is the layout the scale/min pairing already implies (group g holds
 * sub-block 2g in its low nibbles and 2g+1 in its high ones). */
static void q4_k_nibbles(const unsigned char *block, int8_t *out) {
    const unsigned char *packed = block + 16;
#if defined(INGOT_HAVE_Q4_K_SDOT)
    const uint8x16_t mask = vdupq_n_u8(0x0f);
    for (int group = 0; group < 4; group++)
        for (int i = 0; i < 32; i += 16) {
            uint8x16_t nibbles = vld1q_u8(packed + group * 32 + i);
            vst1q_s8(out + group * 64 + i,
                     vreinterpretq_s8_u8(vandq_u8(nibbles, mask)));
            vst1q_s8(out + group * 64 + 32 + i,
                     vreinterpretq_s8_u8(vshrq_n_u8(nibbles, 4)));
        }
#else
    const __m128i mask = _mm_set1_epi8(0x0f);
    for (int group = 0; group < 4; group++)
        for (int i = 0; i < 32; i += 16) {
            const __m128i nibbles =
                _mm_loadu_si128((const __m128i *)(packed + group * 32 + i));
            _mm_storeu_si128((__m128i *)(out + group * 64 + i),
                             _mm_and_si128(nibbles, mask));
            _mm_storeu_si128((__m128i *)(out + group * 64 + 32 + i),
                _mm_and_si128(_mm_srli_epi16(nibbles, 4), mask));
        }
#endif
}

/* Q5_K into the same flat 0..31 array. The scale/min packing is byte for byte
 * the Q4_K one, so only the quant extraction differs: the fifth bit comes from
 * the qh plane, whose selected bit shifts every 64 weights, and adds 16. The
 * largest product is 31*127, so int8 weights against int8 activations still
 * accumulate comfortably inside int32. */
static void q5_k_quants(const unsigned char *block, int8_t *out) {
    const unsigned char *qh = block + 16;
    const unsigned char *qs = block + 48;
#if defined(INGOT_HAVE_Q4_K_SDOT)
    const uint8x16_t mask = vdupq_n_u8(0x0f);
    const uint8x16_t sixteen = vdupq_n_u8(16);
    for (int group = 0; group < 4; group++) {
        uint8x16_t u1 = vdupq_n_u8((unsigned char)(1u << (2 * group)));
        uint8x16_t u2 = vdupq_n_u8((unsigned char)(2u << (2 * group)));
        for (int i = 0; i < 32; i += 16) {
            uint8x16_t packed = vld1q_u8(qs + group * 32 + i);
            uint8x16_t high = vld1q_u8(qh + i);
            uint8x16_t lo = vaddq_u8(vandq_u8(packed, mask),
                                     vandq_u8(vtstq_u8(high, u1), sixteen));
            uint8x16_t hi = vaddq_u8(vshrq_n_u8(packed, 4),
                                     vandq_u8(vtstq_u8(high, u2), sixteen));
            vst1q_s8(out + group * 64 + i, vreinterpretq_s8_u8(lo));
            vst1q_s8(out + group * 64 + 32 + i, vreinterpretq_s8_u8(hi));
        }
    }
#else
    const __m128i mask = _mm_set1_epi8(0x0f);
    const __m128i sixteen = _mm_set1_epi8(16);
    const __m128i zero = _mm_setzero_si128();
    for (int group = 0; group < 4; group++) {
        const __m128i u1 = _mm_set1_epi8((char)(1u << (2 * group)));
        const __m128i u2 = _mm_set1_epi8((char)(2u << (2 * group)));
        for (int i = 0; i < 32; i += 16) {
            const __m128i packed =
                _mm_loadu_si128((const __m128i *)(qs + group * 32 + i));
            const __m128i high = _mm_loadu_si128((const __m128i *)(qh + i));
            /* cmpeq-with-zero is the inverted vtst: andnot re-inverts it */
            const __m128i no1 = _mm_cmpeq_epi8(_mm_and_si128(high, u1), zero);
            const __m128i no2 = _mm_cmpeq_epi8(_mm_and_si128(high, u2), zero);
            const __m128i lo = _mm_add_epi8(_mm_and_si128(packed, mask),
                                            _mm_andnot_si128(no1, sixteen));
            const __m128i hi = _mm_add_epi8(
                _mm_and_si128(_mm_srli_epi16(packed, 4), mask),
                _mm_andnot_si128(no2, sixteen));
            _mm_storeu_si128((__m128i *)(out + group * 64 + i), lo);
            _mm_storeu_si128((__m128i *)(out + group * 64 + 32 + i), hi);
        }
    }
#endif
}

typedef void (*qk_quant_fn)(const unsigned char *block, int8_t *out);

/* The activation scale is per (token, super-block) — the quantiser takes an
 * absmax over each 256-wide block — so it multiplies inside the block loop,
 * not once at the end. Both it and the sub-block sums are stored transposed,
 * block-major, so four consecutive tokens come in with one vector load. */
typedef struct {
    const unsigned char *weights;
    const int8_t *xq;            /* [tokens][xq_stride] int8 activations */
    const float *xscale;         /* [block][tokens] */
    const float *xsum;           /* [block][8][tokens], raw integer sums */
    float *output;               /* [tokens][rows] row-major */
    size_t rows, cols, tokens;
    size_t blocks_per_row, row_bytes, block_bytes;
    qk_quant_fn extract;
    size_t token_begin, token_count, xq_stride;
    /* A copy of xq with token PAIRS interleaved in 8-byte chunks ([t0 8B|t1 8B]
     * per column group): that is vmmlaq_s32's B operand ready to use, with no
     * vcombine in the hot loop. Built once per call, and only when the worker
     * is the SMMLA twin. */
    const int8_t *xq_int;
    size_t xint_stride;
} qk_sdot_job_t;

#if defined(INGOT_HAVE_Q4_K_SDOT)
static void qk_matmat_rows_sdot(size_t begin, size_t end, void *user) {
    const qk_sdot_job_t *job = user;
    alignas(64) int8_t nibbles[INGOT_QK_ROW_TILE * INGOT_QK_K];
    alignas(64) float scale[INGOT_QK_ROW_TILE * 8];
    alignas(64) float offset[INGOT_QK_ROW_TILE * 8];
    alignas(64) float acc[INGOT_QK_ROW_TILE * INGOT_QK_TOKEN_TILE_MAX];
    for (size_t strip = begin; strip < end; strip++) {
        size_t row0 = strip * INGOT_QK_ROW_TILE;
        size_t strip_rows = job->rows - row0 < INGOT_QK_ROW_TILE ?
                            job->rows - row0 : INGOT_QK_ROW_TILE;
        memset(acc, 0, strip_rows * job->token_count * sizeof(*acc));
        for (size_t block = 0; block < job->blocks_per_row; block++) {
            for (size_t r = 0; r < strip_rows; r++) {
                const unsigned char *source = job->weights +
                    (row0 + r) * job->row_bytes + block * job->block_bytes;
                job->extract(source, nibbles + r * INGOT_QK_K);
                float d = f16_to_f32(read_u16(source));
                float dmin = f16_to_f32(read_u16(source + 2));
                for (int j = 0; j < 8; j++) {
                    unsigned char s, m;
                    scale_min(source + 4, j, &s, &m);
                    scale[r * 8 + j] = d * (float)s;
                    offset[r * 8 + j] = dmin * (float)m;
                }
            }
            const int8_t *xq = job->xq +
                job->token_begin * job->xq_stride + block * INGOT_QK_K;
            const float *xsum = job->xsum +
                block * 8 * job->tokens + job->token_begin;
            for (size_t r = 0; r < strip_rows; r++) {
                const int8_t *w = nibbles + r * INGOT_QK_K;
                size_t token = 0;
                for (; token + 4 <= job->token_count; token += 4) {
                    const int8_t *x = xq + token * job->xq_stride;
                    float32x4_t total = vdupq_n_f32(0.0f);
                    for (int j = 0; j < 8; j++) {
                        int8x16_t w0 = vld1q_s8(w + j * 32);
                        int8x16_t w1 = vld1q_s8(w + j * 32 + 16);
                        int32x4_t d[4];
                        for (int t = 0; t < 4; t++) {
                            const int8_t *xt = x + (size_t)t * job->xq_stride + j * 32;
                            d[t] = vdotq_s32(vdotq_s32(vdupq_n_s32(0), w0,
                                                       vld1q_s8(xt)),
                                             w1, vld1q_s8(xt + 16));
                        }
                        int32x4_t sums = vpaddq_s32(vpaddq_s32(d[0], d[1]),
                                                    vpaddq_s32(d[2], d[3]));
                        total = vfmaq_n_f32(total, vcvtq_f32_s32(sums),
                                            scale[r * 8 + j]);
                        total = vfmsq_n_f32(total,
                                            vld1q_f32(xsum + (size_t)j * job->tokens + token),
                                            offset[r * 8 + j]);
                    }
                    float32x4_t out = vmulq_f32(total,
                        vld1q_f32(job->xscale + block * job->tokens +
                                  job->token_begin + token));
                    vst1q_f32(acc + r * job->token_count + token,
                              vaddq_f32(vld1q_f32(acc + r * job->token_count + token),
                                        out));
                }
                for (; token < job->token_count; token++) {
                    const int8_t *x = xq + token * job->xq_stride;
                    float total = 0.0f;
                    for (int j = 0; j < 8; j++) {
                        int32x4_t dot = vdotq_s32(
                            vdotq_s32(vdupq_n_s32(0), vld1q_s8(w + j * 32),
                                      vld1q_s8(x + j * 32)),
                            vld1q_s8(w + j * 32 + 16), vld1q_s8(x + j * 32 + 16));
                        total += scale[r * 8 + j] * (float)vaddvq_s32(dot) -
                                 offset[r * 8 + j] *
                                     xsum[(size_t)j * job->tokens + token];
                    }
                    acc[r * job->token_count + token] += total *
                        job->xscale[block * job->tokens + job->token_begin + token];
                }
            }
        }
        for (size_t r = 0; r < strip_rows; r++)
            for (size_t t = 0; t < job->token_count; t++)
                job->output[(job->token_begin + t) * job->rows + row0 + r] =
                    acc[r * job->token_count + t];
    }
}

#endif /* INGOT_HAVE_Q4_K_SDOT: the vdotq worker */

#if defined(INGOT_HAVE_Q4_K_SDOT) && defined(__ARM_FEATURE_MATMUL_INT8)
#define INGOT_HAVE_Q4_K_SMMLA 1

/* The i8mm twin: vmmlaq_s32 closes a 2-row × 2-token tile in i32 per
 * instruction over K=8 (32 MACs against vdotq's 16). The price is zipping the
 * operands: [row0|row1] and [token0|token1] out of vcombine of half vectors.
 * The accumulator comes out as [r0t0, r0t1, r1t0, r1t1], so the per-sub-block
 * scale becomes a vector [s_r0, s_r0, s_r1, s_r1] and the offset leans on the
 * same trick with the xsum pairs. Rows go in pairs and tokens in quadruples;
 * the tails (odd row, leftover tokens) fall back to the SDOT twin's vdotq
 * loop. If the bench ever says the zips dominate, the next step is to
 * pre-interleave the row pairs in the extract. */
static void qk_matmat_rows_smmla(size_t begin, size_t end, void *user) {
    const qk_sdot_job_t *job = user;
    alignas(64) int8_t nibbles[INGOT_QK_ROW_TILE * INGOT_QK_K];
    alignas(64) float scale[INGOT_QK_ROW_TILE * 8];
    alignas(64) float offset[INGOT_QK_ROW_TILE * 8];
    alignas(64) float acc[INGOT_QK_ROW_TILE * INGOT_QK_TOKEN_TILE_MAX];
    for (size_t strip = begin; strip < end; strip++) {
        size_t row0 = strip * INGOT_QK_ROW_TILE;
        size_t strip_rows = job->rows - row0 < INGOT_QK_ROW_TILE ?
                            job->rows - row0 : INGOT_QK_ROW_TILE;
        memset(acc, 0, strip_rows * job->token_count * sizeof(*acc));
        for (size_t block = 0; block < job->blocks_per_row; block++) {
            for (size_t r = 0; r < strip_rows; r++) {
                const unsigned char *source = job->weights +
                    (row0 + r) * job->row_bytes + block * job->block_bytes;
                job->extract(source, nibbles + r * INGOT_QK_K);
                float d = f16_to_f32(read_u16(source));
                float dmin = f16_to_f32(read_u16(source + 2));
                for (int j = 0; j < 8; j++) {
                    unsigned char s, m;
                    scale_min(source + 4, j, &s, &m);
                    scale[r * 8 + j] = d * (float)s;
                    offset[r * 8 + j] = dmin * (float)m;
                }
            }
            const int8_t *xq = job->xq +
                job->token_begin * job->xq_stride + block * INGOT_QK_K;
            const float *xsum = job->xsum +
                block * 8 * job->tokens + job->token_begin;
            size_t r = 0;
            for (; r + 2 <= strip_rows; r += 2) {
                /* v2: le due righe della coppia interleavate a chunk di 8 byte
                 * ([r0 8B|r1 8B] per k-chunk): 512 byte costruiti UNA volta e
                 * riusati da tutti i token del tile — gli operandi A escono
                 * con load dirette, niente vcombine nel loop caldo. */
                alignas(64) int8_t wint[2 * INGOT_QK_K];
                {
                    const int8_t *w0 = nibbles + r * INGOT_QK_K;
                    const int8_t *w1 = nibbles + (r + 1) * INGOT_QK_K;
                    for (int c = 0; c < INGOT_QK_K / 8; c++)
                        vst1q_s8(wint + c * 16,
                                 vcombine_s8(vld1_s8(w0 + c * 8),
                                             vld1_s8(w1 + c * 8)));
                }
                size_t token = 0;
                for (; token + 4 <= job->token_count; token += 4) {
                    size_t pair0 = (job->token_begin + token) / 2;
                    const int8_t *xp0 = job->xq_int + pair0 * job->xint_stride +
                                        block * 2 * INGOT_QK_K;
                    const int8_t *xp1 = xp0 + job->xint_stride;
                    float32x4_t total0 = vdupq_n_f32(0.0f);
                    float32x4_t total1 = vdupq_n_f32(0.0f);
                    for (int j = 0; j < 8; j++) {
                        const int8_t *wj = wint + j * 64;
                        int8x16_t A0 = vld1q_s8(wj);
                        int8x16_t A1 = vld1q_s8(wj + 16);
                        int8x16_t A2 = vld1q_s8(wj + 32);
                        int8x16_t A3 = vld1q_s8(wj + 48);
                        float sj0 = scale[r * 8 + j], sj1 = scale[(r + 1) * 8 + j];
                        float oj0 = offset[r * 8 + j], oj1 = offset[(r + 1) * 8 + j];
                        float32x4_t srow = {sj0, sj0, sj1, sj1};
                        float32x4_t orow = {oj0, oj0, oj1, oj1};
                        for (int p = 0; p < 2; p++) {
                            const int8_t *xj = (p == 0 ? xp0 : xp1) + j * 64;
                            int32x4_t s2 = vmmlaq_s32(
                                vmmlaq_s32(
                                    vmmlaq_s32(
                                        vmmlaq_s32(vdupq_n_s32(0),
                                                   A0, vld1q_s8(xj)),
                                        A1, vld1q_s8(xj + 16)),
                                    A2, vld1q_s8(xj + 32)),
                                A3, vld1q_s8(xj + 48));
                            float32x2_t xsd = vld1_f32(xsum + (size_t)j * job->tokens +
                                                       token + (size_t)(2 * p));
                            float32x4_t xsvec = vcombine_f32(xsd, xsd);
                            float32x4_t *total = p == 0 ? &total0 : &total1;
                            *total = vfmaq_f32(*total, vcvtq_f32_s32(s2), srow);
                            *total = vfmsq_f32(*total, xsvec, orow);
                        }
                    }
                    for (int p = 0; p < 2; p++) {
                        float32x4_t total = p == 0 ? total0 : total1;
                        float32x2_t xsc = vld1_f32(job->xscale + block * job->tokens +
                                                   job->token_begin + token + (size_t)(2 * p));
                        float32x4_t out = vmulq_f32(total, vcombine_f32(xsc, xsc));
                        float *a0 = acc + r * job->token_count + token + (size_t)(2 * p);
                        float *a1 = acc + (r + 1) * job->token_count + token + (size_t)(2 * p);
                        vst1_f32(a0, vadd_f32(vld1_f32(a0), vget_low_f32(out)));
                        vst1_f32(a1, vadd_f32(vld1_f32(a1), vget_high_f32(out)));
                    }
                }
                for (; token < job->token_count; token++) {
                    for (int rr = 0; rr < 2; rr++) {
                        const int8_t *w = nibbles + (r + (size_t)rr) * INGOT_QK_K;
                        const int8_t *xt = xq + token * job->xq_stride;
                        float total = 0.0f;
                        for (int j = 0; j < 8; j++) {
                            int32x4_t dot = vdotq_s32(
                                vdotq_s32(vdupq_n_s32(0), vld1q_s8(w + j * 32),
                                          vld1q_s8(xt + j * 32)),
                                vld1q_s8(w + j * 32 + 16), vld1q_s8(xt + j * 32 + 16));
                            total += scale[(r + (size_t)rr) * 8 + j] * (float)vaddvq_s32(dot) -
                                     offset[(r + (size_t)rr) * 8 + j] *
                                         xsum[(size_t)j * job->tokens + token];
                        }
                        acc[(r + (size_t)rr) * job->token_count + token] += total *
                            job->xscale[block * job->tokens + job->token_begin + token];
                    }
                }
            }
            for (; r < strip_rows; r++) {
                const int8_t *w = nibbles + r * INGOT_QK_K;
                for (size_t token = 0; token < job->token_count; token++) {
                    const int8_t *xt = xq + token * job->xq_stride;
                    float total = 0.0f;
                    for (int j = 0; j < 8; j++) {
                        int32x4_t dot = vdotq_s32(
                            vdotq_s32(vdupq_n_s32(0), vld1q_s8(w + j * 32),
                                      vld1q_s8(xt + j * 32)),
                            vld1q_s8(w + j * 32 + 16), vld1q_s8(xt + j * 32 + 16));
                        total += scale[r * 8 + j] * (float)vaddvq_s32(dot) -
                                 offset[r * 8 + j] *
                                     xsum[(size_t)j * job->tokens + token];
                    }
                    acc[r * job->token_count + token] += total *
                        job->xscale[block * job->tokens + job->token_begin + token];
                }
            }
        }
        for (size_t r = 0; r < strip_rows; r++)
            for (size_t t = 0; t < job->token_count; t++)
                job->output[(job->token_begin + t) * job->rows + row0 + r] =
                    acc[r * job->token_count + t];
    }
}
#endif /* INGOT_HAVE_Q4_K_SMMLA */

#if defined(INGOT_HAVE_Q4_K_SMMLA)
typedef struct {
    const int8_t *xq;
    int8_t *xq_int;
    size_t cols, xq_stride, xint_stride;
} qk_interleave_job_t;

/* Token pairs interleaved in 8-byte chunks, once per call (one copy of xq,
 * ~cols bytes per token: microseconds) and reused by every row strip. */
static void qk_interleave_pairs(size_t begin, size_t end, void *user) {
    const qk_interleave_job_t *job = user;
    for (size_t pair = begin; pair < end; pair++) {
        const int8_t *t0 = job->xq + (2 * pair) * job->xq_stride;
        const int8_t *t1 = t0 + job->xq_stride;
        int8_t *dst = job->xq_int + pair * job->xint_stride;
        for (size_t c = 0; c < job->cols / 8; c++)
            vst1q_s8(dst + c * 16, vcombine_s8(vld1_s8(t0 + c * 8),
                                               vld1_s8(t1 + c * 8)));
    }
}
#endif

#if defined(INGOT_HAVE_QK_X86INT8)
/* The VPDPBUSD worker — the x86 rendition of qk_matmat_rows_sdot. Four
 * tokens share each extracted weight vector; their four 8-lane accumulators
 * collapse with three VPHADDD into one vector of four sums, mirroring the
 * vpaddq collapse on ARM, and the per-sub-block scale/offset then apply to
 * the whole vector. Without AVX-512VNNI the dot inside qk_dot_u8s8 becomes
 * VPMADDUBSW+VPMADDWD — same integers, one more instruction. */
static void qk_matmat_rows_x86(size_t begin, size_t end, void *user) {
    const qk_sdot_job_t *job = user;
    alignas(64) int8_t nibbles[INGOT_QK_ROW_TILE * INGOT_QK_K];
    alignas(64) float scale[INGOT_QK_ROW_TILE * 8];
    alignas(64) float offset[INGOT_QK_ROW_TILE * 8];
    alignas(64) float acc[INGOT_QK_ROW_TILE * INGOT_QK_TOKEN_TILE_MAX];
    for (size_t strip = begin; strip < end; strip++) {
        size_t row0 = strip * INGOT_QK_ROW_TILE;
        size_t strip_rows = job->rows - row0 < INGOT_QK_ROW_TILE ?
                            job->rows - row0 : INGOT_QK_ROW_TILE;
        memset(acc, 0, strip_rows * job->token_count * sizeof(*acc));
        for (size_t block = 0; block < job->blocks_per_row; block++) {
            for (size_t r = 0; r < strip_rows; r++) {
                const unsigned char *source = job->weights +
                    (row0 + r) * job->row_bytes + block * job->block_bytes;
                job->extract(source, nibbles + r * INGOT_QK_K);
                float d = f16_to_f32(read_u16(source));
                float dmin = f16_to_f32(read_u16(source + 2));
                for (int j = 0; j < 8; j++) {
                    unsigned char s, m;
                    scale_min(source + 4, j, &s, &m);
                    scale[r * 8 + j] = d * (float)s;
                    offset[r * 8 + j] = dmin * (float)m;
                }
            }
            const int8_t *xq = job->xq +
                job->token_begin * job->xq_stride + block * INGOT_QK_K;
            const float *xsum = job->xsum +
                block * 8 * job->tokens + job->token_begin;
            for (size_t r = 0; r < strip_rows; r++) {
                const int8_t *w = nibbles + r * INGOT_QK_K;
                size_t token = 0;
                for (; token + 4 <= job->token_count; token += 4) {
                    const int8_t *x = xq + token * job->xq_stride;
                    __m128 total = _mm_setzero_ps();
                    for (int j = 0; j < 8; j++) {
                        const __m256i wv =
                            _mm256_loadu_si256((const __m256i *)(w + j * 32));
                        __m256i d0 = qk_dot_u8s8(_mm256_setzero_si256(), wv,
                            _mm256_loadu_si256((const __m256i *)(x + j * 32)));
                        __m256i d1 = qk_dot_u8s8(_mm256_setzero_si256(), wv,
                            _mm256_loadu_si256((const __m256i *)(x + job->xq_stride + j * 32)));
                        __m256i d2 = qk_dot_u8s8(_mm256_setzero_si256(), wv,
                            _mm256_loadu_si256((const __m256i *)(x + 2 * job->xq_stride + j * 32)));
                        __m256i d3 = qk_dot_u8s8(_mm256_setzero_si256(), wv,
                            _mm256_loadu_si256((const __m256i *)(x + 3 * job->xq_stride + j * 32)));
                        /* three VPHADDD leave [Σd0, Σd1, Σd2, Σd3] per lane;
                         * adding the two lanes closes the reduction */
                        const __m256i s = _mm256_hadd_epi32(
                            _mm256_hadd_epi32(d0, d1), _mm256_hadd_epi32(d2, d3));
                        const __m128i sums4 = _mm_add_epi32(
                            _mm256_castsi256_si128(s),
                            _mm256_extracti128_si256(s, 1));
                        total = _mm_fmadd_ps(_mm_cvtepi32_ps(sums4),
                                             _mm_set1_ps(scale[r * 8 + j]), total);
                        total = _mm_fnmadd_ps(
                            _mm_loadu_ps(xsum + (size_t)j * job->tokens + token),
                            _mm_set1_ps(offset[r * 8 + j]), total);
                    }
                    const __m128 out = _mm_mul_ps(total,
                        _mm_loadu_ps(job->xscale + block * job->tokens +
                                     job->token_begin + token));
                    float *a = acc + r * job->token_count + token;
                    _mm_storeu_ps(a, _mm_add_ps(_mm_loadu_ps(a), out));
                }
                for (; token < job->token_count; token++) {
                    const int8_t *x = xq + token * job->xq_stride;
                    float total = 0.0f;
                    for (int j = 0; j < 8; j++) {
                        const __m256i dot = qk_dot_u8s8(_mm256_setzero_si256(),
                            _mm256_loadu_si256((const __m256i *)(w + j * 32)),
                            _mm256_loadu_si256((const __m256i *)(x + j * 32)));
                        total += scale[r * 8 + j] * (float)qk_hsum_epi32(dot) -
                                 offset[r * 8 + j] *
                                     xsum[(size_t)j * job->tokens + token];
                    }
                    acc[r * job->token_count + token] += total *
                        job->xscale[block * job->tokens + job->token_begin + token];
                }
            }
        }
        for (size_t r = 0; r < strip_rows; r++)
            for (size_t t = 0; t < job->token_count; t++)
                job->output[(job->token_begin + t) * job->rows + row0 + r] =
                    acc[r * job->token_count + t];
    }
}
#endif /* INGOT_HAVE_QK_X86INT8 */

typedef struct {
    const float *input;
    int8_t *xq;
    float *xscale, *xsum;
    size_t cols, tokens, blocks_per_row, xq_stride;
} qk_quantise_job_t;

static void qk_quantise_tokens(size_t begin, size_t end, void *user) {
    const qk_quantise_job_t *job = user;
    for (size_t t = begin; t < end; t++)
        for (size_t block = 0; block < job->blocks_per_row; block++) {
            int32_t sums[8];
            job->xscale[block * job->tokens + t] = q4_k_quantize_activation(
                job->input + t * job->cols + block * INGOT_QK_K,
                job->xq + t * job->xq_stride + block * INGOT_QK_K, sums);
            for (int j = 0; j < 8; j++)
                job->xsum[(block * 8 + (size_t)j) * job->tokens + t] =
                    (float)sums[j];
        }
}

static int qk_matmat_sdot(const void *weights, size_t rows, size_t cols,
                          const float *input, float *output, size_t tokens,
                          size_t block_bytes, qk_quant_fn extract) {
    size_t blocks_per_row = cols / INGOT_QK_K;
    if (blocks_per_row > SIZE_MAX / block_bytes) return -1;
    size_t row_bytes = blocks_per_row * block_bytes;
    /* One cache line of stride padding, which here is free: xq is our own
     * buffer, so the four tokens the inner loop reads together stop landing in
     * the same L1 set without the copy that made it a loss on the f32 twin. */
    size_t xq_stride = cols + 64;
    if (rows > SIZE_MAX / row_bytes || tokens > SIZE_MAX / xq_stride) return -1;
    /* Quantised once for the whole call and reused by every row: that is the
     * entire point of the format, and the reason this pays more the wider the
     * batch. Roughly cols/4 bytes per token, a quarter of the f32 input. */
    int8_t *xq = ingot_aligned_alloc(64, tokens * xq_stride);
    float *xscale = ingot_aligned_alloc(64, blocks_per_row * tokens * sizeof(*xscale));
    float *xsum = ingot_aligned_alloc(64, blocks_per_row * 8 * tokens * sizeof(*xsum));
    if (xq == NULL || xscale == NULL || xsum == NULL) {
        ingot_aligned_free(xq); ingot_aligned_free(xscale); ingot_aligned_free(xsum);
        return -1;
    }
    qk_quantise_job_t quantise = {input, xq, xscale, xsum, cols, tokens,
                                  blocks_per_row, xq_stride};
    ingot_parallel_for(tokens, qk_quantise_tokens, &quantise);

    qk_sdot_job_t job = {
        (const unsigned char *)weights, xq, xscale, xsum, output,
        rows, cols, tokens, blocks_per_row, row_bytes, block_bytes, extract,
        0, 0, xq_stride, NULL, 0,
    };
    size_t strips = (rows + INGOT_QK_ROW_TILE - 1) / INGOT_QK_ROW_TILE;
    size_t tile = qk_token_tile(cols);
    /* Where i8mm exists (Grace has it, the M1 does not) the worker is the
     * SMMLA twin — same numeric contract apart from the order of the sums.
     * INGOT_SMMLA=0 is the kill-switch that puts vdotq back, and it doubles
     * as the A/B for the bench. */
#if defined(INGOT_HAVE_Q4_K_SDOT)
    ingot_range_fn_t worker = qk_matmat_rows_sdot;
#else
    ingot_range_fn_t worker = qk_matmat_rows_x86;
#endif
    int8_t *xq_int = NULL;
#if defined(INGOT_HAVE_Q4_K_SMMLA)
    static int smmla_env = -1;
    if (smmla_env < 0) {
        const char *value = getenv("INGOT_SMMLA");
        smmla_env = !(value != NULL && value[0] == '0');
    }
    if (smmla_env && ingot_cpu().i8mm && tokens >= 2) {
        /* Stride della coppia paddato di una linea, per la stessa ragione di
         * xq (§61): 2*cols/64 e' pari, +64 lo rende dispari -> 256 gruppi. */
        size_t xint_stride = 2 * cols + 64;
        size_t pairs = tokens / 2;
        if (pairs <= SIZE_MAX / xint_stride)
            xq_int = ingot_aligned_alloc(64, pairs * xint_stride);
        if (xq_int != NULL) {
            qk_interleave_job_t inter = {xq, xq_int, cols, xq_stride,
                                         xint_stride};
            ingot_parallel_for(pairs, qk_interleave_pairs, &inter);
            job.xq_int = xq_int;
            job.xint_stride = xint_stride;
            worker = qk_matmat_rows_smmla;
        }
        /* alloc fallita: si resta su SDOT, che non ne ha bisogno */
    }
#endif
    for (size_t base = 0; base < tokens; base += tile) {
        job.token_begin = base;
        job.token_count = tokens - base < tile ? tokens - base : tile;
        ingot_parallel_for(strips, worker, &job);
    }
    ingot_aligned_free(xq_int);
    ingot_aligned_free(xq); ingot_aligned_free(xscale); ingot_aligned_free(xsum);
    return 0;
}
#endif

/* ── The batched default: int8, with `INGOT_SDOT=0` to go back to exact ───
 *
 * This was opt-in at first, because the exact path is what the gates measure
 * against. It became the default for batches when the assumption that int8
 * only pays off on wide batches fell: looking for the threshold showed there
 * is no crossover at all. Kernel on an M1, rows=1024, int8/f32:
 *
 *     token       2      4      8     16     32     64    128
 *     cols  6144  2.11x  1.50x  1.57x  1.73x  1.66x  1.79x  1.84x
 *     cols 16384  2.53x  2.03x  1.95x  2.00x  1.91x  1.90x  1.99x
 *
 * int8 is never slower from two tokens up, so any threshold would be
 * arbitrary. A full CPU forward on real weights, 1024 tokens: 124.9 s exact
 * → 72.6 s, **1.72x**.
 *
 * `tokens == 1` does NOT come through here: it goes to the matvec, which
 * stays exact bit for bit (the gate checks it) because that is what parity is
 * measured against. The cost is ~2.4e-3 relative per GEMM, so `INGOT_SDOT=0`
 * turns all of it off and `ingot_matmat_is_exact()` says which path a call
 * will take, so whoever is comparing can pick the tolerance instead of
 * guessing it. */
#if defined(INGOT_HAVE_Q4_K_SDOT) || defined(INGOT_HAVE_QK_X86INT8)
static int qk_sdot_batched(void) {
    static int enabled = -1;
    if (enabled < 0) {
        const char *value = getenv("INGOT_SDOT");
        enabled = !(value != NULL && value[0] == '0');
    }
    return enabled;
}
#endif

int ingot_matmat_is_exact(size_t tokens) {
    if (tokens <= 1) return 1;
#if defined(INGOT_HAVE_Q4_K_SDOT) || defined(INGOT_HAVE_QK_X86INT8)
    return !(qk_sdot_batched() && qk_int8_ready());
#else
    return 1;
#endif
}

static int q4_k_matmat_maybe_int8(const void *weights, size_t rows, size_t cols,
                                  const float *input, float *output,
                                  size_t tokens, int allow_int8) {
    if (tokens == 1)
        return ingot_q4_k_matvec(weights, rows, cols, input, output);
#if defined(INGOT_HAVE_Q4_K_SDOT) || defined(INGOT_HAVE_QK_X86INT8)
    if (allow_int8 && qk_sdot_batched() && qk_int8_ready() &&
        weights != NULL && input != NULL && output != NULL &&
        rows != 0 && cols != 0 && cols % INGOT_QK_K == 0 &&
        qk_matmat_sdot(weights, rows, cols, input, output, tokens,
                       INGOT_Q4_K_BYTES, q4_k_nibbles) == 0)
        return 0;
#else
    (void)allow_int8;
#endif
    return qk_matmat(weights, rows, cols, input, output, tokens,
                     INGOT_Q4_K_BYTES, q4_k_decode_block);
}

int ingot_q4_k_matmat(const void *weights, size_t rows, size_t cols,
                           const float *input, float *output, size_t tokens) {
    return q4_k_matmat_maybe_int8(weights, rows, cols, input, output, tokens, 1);
}

int ingot_q4_k_matmat_exact(const void *weights, size_t rows, size_t cols,
                                 const float *input, float *output,
                                 size_t tokens) {
    return q4_k_matmat_maybe_int8(weights, rows, cols, input, output, tokens, 0);
}

int ingot_q4_k_dequant(const void *weights, size_t rows, size_t cols,
                            float *output) {
    if (weights == NULL || output == NULL || rows == 0 || cols == 0 ||
        cols % INGOT_QK_K != 0) return -1;
    size_t blocks_per_row = cols / INGOT_QK_K;
    if (blocks_per_row > SIZE_MAX / INGOT_Q4_K_BYTES) return -1;
    size_t row_bytes = blocks_per_row * INGOT_Q4_K_BYTES;
    if (rows > SIZE_MAX / row_bytes || rows * cols > SIZE_MAX / sizeof(float)) return -1;
    const unsigned char *source = (const unsigned char *)weights;
    for (size_t row = 0; row < rows; row++) {
        const unsigned char *row_data = source + row * row_bytes;
        float *row_output = output + row * cols;
        for (size_t block = 0; block < blocks_per_row; block++)
            dequant_block(row_data + block * INGOT_Q4_K_BYTES,
                          row_output + block * INGOT_QK_K);
    }
    return 0;
}

/* Q5_K super-block: d(f16) dmin(f16) scales[12] qh[32] qs[128]. Each weight is a
 * 5-bit value: the low 4 bits come from qs (like Q4_K) and the 5th bit from the
 * per-weight qh plane, whose selected bit shifts (u1/u2) every 64 weights. */
static void q5_scale_pointers(const unsigned char *block, float *d, float *dmin,
                              const unsigned char **scales,
                              const unsigned char **qh, const unsigned char **qs) {
    *d = f16_to_f32(read_u16(block));
    *dmin = f16_to_f32(read_u16(block + 2));
    *scales = block + 4;   /* 12 bytes, shared 6-bit scale/min packing with Q4_K */
    *qh = block + 16;      /* 32 bytes, one high bit per weight */
    *qs = block + 48;      /* 128 bytes, 4-bit low nibbles */
}

static float q5_dot_block(const unsigned char *block, const float *input) {
    float d, dmin;
    const unsigned char *scales, *qh, *qs;
    q5_scale_pointers(block, &d, &dmin, &scales, &qh, &qs);
    float sum = 0.0f;
    unsigned u1 = 1, u2 = 2;
    int scale_index = 0;
    for (int base = 0; base < INGOT_QK_K; base += 64) {
        unsigned char scale0, scale1, min0, min1;
        scale_min(scales, scale_index, &scale0, &min0);
        scale_min(scales, scale_index + 1, &scale1, &min1);
        float d0 = d * scale0, d1 = d * scale1;
        float m0 = dmin * min0, m1 = dmin * min1;
        for (int i = 0; i < 32; i++) {
            float low = (float)((qs[i] & 0x0fu) + ((qh[i] & u1) ? 16 : 0));
            float high = (float)((qs[i] >> 4) + ((qh[i] & u2) ? 16 : 0));
            sum += (d0 * low - m0) * input[base + i];
            sum += (d1 * high - m1) * input[base + i + 32];
        }
        qs += 32;
        scale_index += 2;
        u1 <<= 2;
        u2 <<= 2;
    }
    return sum;
}

static void q5_dequant_block(const unsigned char *block, float *output) {
    float d, dmin;
    const unsigned char *scales, *qh, *qs;
    q5_scale_pointers(block, &d, &dmin, &scales, &qh, &qs);
    unsigned u1 = 1, u2 = 2;
    int scale_index = 0;
    for (int base = 0; base < INGOT_QK_K; base += 64) {
        unsigned char scale0, scale1, min0, min1;
        scale_min(scales, scale_index, &scale0, &min0);
        scale_min(scales, scale_index + 1, &scale1, &min1);
        float d0 = d * scale0, d1 = d * scale1;
        float m0 = dmin * min0, m1 = dmin * min1;
        for (int i = 0; i < 32; i++) {
            output[base + i] = d0 * (float)((qs[i] & 0x0fu) + ((qh[i] & u1) ? 16 : 0)) - m0;
            output[base + i + 32] = d1 * (float)((qs[i] >> 4) + ((qh[i] & u2) ? 16 : 0)) - m1;
        }
        qs += 32;
        scale_index += 2;
        u1 <<= 2;
        u2 <<= 2;
    }
}

int ingot_q5_k_matvec_scalar(const void *weights, size_t rows, size_t cols,
                                  const float *input, float *output) {
    if (weights == NULL || input == NULL || output == NULL || rows == 0 ||
        cols == 0 || cols % INGOT_QK_K != 0) return -1;
    size_t blocks_per_row = cols / INGOT_QK_K;
    if (blocks_per_row > SIZE_MAX / INGOT_Q5_K_BYTES) return -1;
    size_t row_bytes = blocks_per_row * INGOT_Q5_K_BYTES;
    if (rows > SIZE_MAX / row_bytes) return -1;
    const unsigned char *source = (const unsigned char *)weights;
    for (size_t row = 0; row < rows; row++) {
        float sum = 0.0f;
        const unsigned char *row_data = source + row * row_bytes;
        for (size_t block = 0; block < blocks_per_row; block++)
            sum += q5_dot_block(row_data + block * INGOT_Q5_K_BYTES,
                                input + block * INGOT_QK_K);
        output[row] = sum;
    }
    return 0;
}

#if defined(INGOT_HAVE_Q4_K_NEON)
/* Distribute the sum instead of materializing every weight.
 *
 *     SUM_j w_j x_j  =  d*scale * SUM_j (q_j x_j)  -  dmin*min * SUM_j x_j
 *
 * The previous kernel built each weight before using it: multiply by d*scale,
 * subtract dmin*min, THEN the multiply-add that actually matters — two vector
 * ops per four values spent on arithmetic the sub-block could have shared. Here
 * the quants accumulate against the input and the inputs accumulate on their
 * own, and the scale and min are each applied once per 32-weight sub-block.
 *
 * Folded into a running vector with vmlaq_n_f32 (the min term as a negative
 * multiplier, since NEON has no vmlsq_n_f32) so a block still costs exactly one
 * horizontal reduction.
 *
 * Fewer roundings than before, not more: the min is now one subtraction per
 * sub-block rather than one per weight. Held to 1e-5 against dequant-then-dot
 * by test_quant's every-format gate. */
static float q5_k_dot_block_neon(const unsigned char *block, const float *input) {
    float d, dmin;
    const unsigned char *scales, *qh, *qs;
    q5_scale_pointers(block, &d, &dmin, &scales, &qh, &qs);
    float32x4_t total = vdupq_n_f32(0.0f);
    const uint8x8_t sixteen = vdup_n_u8(16);
    const uint8x8_t nib = vdup_n_u8(0x0f);
    unsigned char u1 = 1, u2 = 2;

    for (int base = 0, scale_index = 0; base < INGOT_QK_K; base += 64, scale_index += 2) {
        unsigned char scale0, scale1, min0, min1;
        scale_min(scales, scale_index, &scale0, &min0);
        scale_min(scales, scale_index + 1, &scale1, &min1);
        const uint8x8_t u1v = vdup_n_u8(u1), u2v = vdup_n_u8(u2);

        float32x4_t q0 = vdupq_n_f32(0.0f), q1 = vdupq_n_f32(0.0f);
        float32x4_t x0 = vdupq_n_f32(0.0f), x1 = vdupq_n_f32(0.0f);

        for (int i = 0; i < 32; i += 8) {
            const uint8x8_t packed = vld1_u8(qs + i);
            const uint8x8_t qhbits = vld1_u8(qh + i);
            /* 5th bit: add 16 to the nibble when the selected qh bit is set. */
            const uint8x8_t lo = vadd_u8(vand_u8(packed, nib),
                                         vand_u8(vtst_u8(qhbits, u1v), sixteen));
            const uint8x8_t hi = vadd_u8(vshr_n_u8(packed, 4),
                                         vand_u8(vtst_u8(qhbits, u2v), sixteen));

            const float32x4_t xa = vld1q_f32(input + base + i);
            const float32x4_t xb = vld1q_f32(input + base + i + 4);
            const float32x4_t xc = vld1q_f32(input + base + i + 32);
            const float32x4_t xd = vld1q_f32(input + base + i + 36);

            q0 = vmlaq_f32(q0, q4_k_u8x8_to_f32(lo, 0), xa);
            q0 = vmlaq_f32(q0, q4_k_u8x8_to_f32(lo, 1), xb);
            q1 = vmlaq_f32(q1, q4_k_u8x8_to_f32(hi, 0), xc);
            q1 = vmlaq_f32(q1, q4_k_u8x8_to_f32(hi, 1), xd);
            x0 = vaddq_f32(x0, vaddq_f32(xa, xb));
            x1 = vaddq_f32(x1, vaddq_f32(xc, xd));
        }

        total = vmlaq_n_f32(total, q0, d * scale0);
        total = vmlaq_n_f32(total, q1, d * scale1);
        total = vmlaq_n_f32(total, x0, -(dmin * min0));
        total = vmlaq_n_f32(total, x1, -(dmin * min1));

        qs += 32;
        u1 <<= 2;
        u2 <<= 2;
    }
    return vaddvq_f32(total);
}

static int ingot_q5_k_matvec_neon(const void *weights, size_t rows, size_t cols,
                                       const float *input, float *output) {
    if (weights == NULL || input == NULL || output == NULL || rows == 0 ||
        cols == 0 || cols % INGOT_QK_K != 0) return -1;
    size_t blocks_per_row = cols / INGOT_QK_K;
    if (blocks_per_row > SIZE_MAX / INGOT_Q5_K_BYTES) return -1;
    size_t row_bytes = blocks_per_row * INGOT_Q5_K_BYTES;
    if (rows > SIZE_MAX / row_bytes) return -1;
    const unsigned char *source = (const unsigned char *)weights;
    for (size_t row = 0; row < rows; row++) {
        const unsigned char *row_data = source + row * row_bytes;
        float sum = 0.0f;
        for (size_t block = 0; block < blocks_per_row; block++)
            sum += q5_k_dot_block_neon(row_data + block * INGOT_Q5_K_BYTES,
                                       input + block * INGOT_QK_K);
        output[row] = sum;
    }
    return 0;
}
#endif

#if defined(INGOT_HAVE_Q4_K_AVX2)
/* The AVX2 twin of the kernel above, same distribute-the-sum identity. x86 had
 * no Q5_K kernel at all before this and went through decode-then-dot.
 *
 * NEON's vtst_u8 has no single-instruction x86 equivalent, so the 5th bit is
 * built as: AND the selector, compare-equal against zero (0xFF where the bit is
 * CLEAR), then andnot against 16 — three ops where NEON needs two. */
static float q5_k_dot_block_avx2(const unsigned char *block, const float *input) {
    float d, dmin;
    const unsigned char *scales, *qh, *qs;
    q5_scale_pointers(block, &d, &dmin, &scales, &qh, &qs);
    __m256 total = _mm256_setzero_ps();
    const __m128i nib = _mm_set1_epi8(0x0f);
    const __m128i sixteen = _mm_set1_epi8(16);
    const __m128i zero = _mm_setzero_si128();
    unsigned char u1 = 1, u2 = 2;

    for (int base = 0, scale_index = 0; base < INGOT_QK_K; base += 64, scale_index += 2) {
        unsigned char scale0, scale1, min0, min1;
        scale_min(scales, scale_index, &scale0, &min0);
        scale_min(scales, scale_index + 1, &scale1, &min1);
        const __m128i u1v = _mm_set1_epi8((char)u1), u2v = _mm_set1_epi8((char)u2);

        __m256 q0 = _mm256_setzero_ps(), q1 = _mm256_setzero_ps();
        __m256 x0 = _mm256_setzero_ps(), x1 = _mm256_setzero_ps();

        for (int i = 0; i < 32; i += 8) {
            const __m128i packed = _mm_loadl_epi64((const __m128i *)(const void *)(qs + i));
            const __m128i qhbits = _mm_loadl_epi64((const __m128i *)(const void *)(qh + i));

            const __m128i b1 = _mm_andnot_si128(
                _mm_cmpeq_epi8(_mm_and_si128(qhbits, u1v), zero), sixteen);
            const __m128i b2 = _mm_andnot_si128(
                _mm_cmpeq_epi8(_mm_and_si128(qhbits, u2v), zero), sixteen);

            const __m128i lo = _mm_add_epi8(_mm_and_si128(packed, nib), b1);
            const __m128i hi = _mm_add_epi8(
                _mm_and_si128(_mm_srli_epi16(packed, 4), nib), b2);

            const __m256 xa = _mm256_loadu_ps(input + base + i);
            const __m256 xc = _mm256_loadu_ps(input + base + i + 32);

            q0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(lo)), xa, q0);
            q1 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(hi)), xc, q1);
            x0 = _mm256_add_ps(x0, xa);
            x1 = _mm256_add_ps(x1, xc);
        }

        total = _mm256_fmadd_ps(q0, _mm256_set1_ps(d * scale0), total);
        total = _mm256_fmadd_ps(q1, _mm256_set1_ps(d * scale1), total);
        total = _mm256_fmadd_ps(x0, _mm256_set1_ps(-(dmin * min0)), total);
        total = _mm256_fmadd_ps(x1, _mm256_set1_ps(-(dmin * min1)), total);

        qs += 32;
        u1 <<= 2;
        u2 <<= 2;
    }
    {
        __m128 h = _mm_add_ps(_mm256_castps256_ps128(total),
                              _mm256_extractf128_ps(total, 1));
        h = _mm_add_ps(h, _mm_movehl_ps(h, h));
        h = _mm_add_ss(h, _mm_shuffle_ps(h, h, 0x55));
        return _mm_cvtss_f32(h);
    }
}

static int ingot_q5_k_matvec_avx2(const void *weights, size_t rows, size_t cols,
                                  const float *input, float *output) {
    const size_t blocks_per_row = cols / INGOT_QK_K;
    if (blocks_per_row > SIZE_MAX / INGOT_Q5_K_BYTES) return -1;
    const size_t row_bytes = blocks_per_row * INGOT_Q5_K_BYTES;
    if (rows > SIZE_MAX / row_bytes) return -1;
    const unsigned char *source = (const unsigned char *)weights;
    for (size_t row = 0; row < rows; row++) {
        const unsigned char *row_data = source + row * row_bytes;
        float sum = 0.0f;
        for (size_t block = 0; block < blocks_per_row; block++)
            sum += q5_k_dot_block_avx2(row_data + block * INGOT_Q5_K_BYTES,
                                       input + block * INGOT_QK_K);
        output[row] = sum;
    }
    return 0;
}
#endif

static int kquant_apply(const void *weights, size_t rows, size_t cols,
                        const float *input, float *output, size_t block_bytes,
                        void (*dequant)(const unsigned char *, float *), int matvec);

int ingot_q5_k_matvec(const void *weights, size_t rows, size_t cols,
                           const float *input, float *output) {
    if (!qk_args_ok(weights, input, output, rows, cols, INGOT_QK_K)) return -1;
#if defined(INGOT_HAVE_Q4_K_NEON)
    if (ingot_cpu().neon)
        return ingot_q5_k_matvec_neon(weights, rows, cols, input, output);
#endif
#if defined(INGOT_HAVE_Q4_K_AVX2)
    if (ingot_cpu().avx2)
        return ingot_q5_k_matvec_avx2(weights, rows, cols, input, output);
#endif
    return ingot_q5_k_matvec_scalar(weights, rows, cols, input, output);
}

int ingot_q5_k_dequant(const void *weights, size_t rows, size_t cols,
                            float *output) {
    if (weights == NULL || output == NULL || rows == 0 || cols == 0 ||
        cols % INGOT_QK_K != 0) return -1;
    size_t blocks_per_row = cols / INGOT_QK_K;
    if (blocks_per_row > SIZE_MAX / INGOT_Q5_K_BYTES) return -1;
    size_t row_bytes = blocks_per_row * INGOT_Q5_K_BYTES;
    if (rows > SIZE_MAX / row_bytes || rows * cols > SIZE_MAX / sizeof(float)) return -1;
    const unsigned char *source = (const unsigned char *)weights;
    for (size_t row = 0; row < rows; row++) {
        const unsigned char *row_data = source + row * row_bytes;
        float *row_output = output + row * cols;
        for (size_t block = 0; block < blocks_per_row; block++)
            q5_dequant_block(row_data + block * INGOT_Q5_K_BYTES,
                             row_output + block * INGOT_QK_K);
    }
    return 0;
}

/* Q3_K super-block (110 bytes): hmask[32] qs[64] scales[12] d(f16). Values are
 * 3-bit: 2 low bits from qs (four planes per byte, shift 0/2/4/6) and a high
 * bit from hmask; a cleared high bit subtracts 4 (ggml convention). The 16
 * 6-bit sub-block scales are packed across scales[12] and used as (sc - 32). */
#define INGOT_Q3_K_BYTES 110
#define INGOT_Q2_K_BYTES 84

static int q3_k_scale(const unsigned char *scales, int index) {
    int sc;
    if (index < 4)
        sc = (scales[index] & 0x0f) | (((scales[8 + index] >> 0) & 3) << 4);
    else if (index < 8)
        sc = (scales[index] & 0x0f) | (((scales[4 + index] >> 2) & 3) << 4);
    else if (index < 12)
        sc = (scales[index - 8] >> 4) | (((scales[index] >> 4) & 3) << 4);
    else
        sc = (scales[index - 8] >> 4) | (((scales[index - 4] >> 6) & 3) << 4);
    return sc - 32;
}

static void q3_dequant_block(const unsigned char *block, float *output) {
    const unsigned char *hmask = block;
    const unsigned char *qs = block + 32;
    const unsigned char *scales = block + 96;
    float d = f16_to_f32(read_u16(block + 108));
    int scale_index = 0;
    unsigned char m = 1;
    for (int chunk = 0; chunk < INGOT_QK_K; chunk += 128) {
        int shift = 0;
        for (int j = 0; j < 4; j++) {
            for (int half = 0; half < 2; half++) {
                float dl = d * (float)q3_k_scale(scales, scale_index++);
                for (int l = 0; l < 16; l++) {
                    int position = half * 16 + l;
                    int quant = (qs[position] >> shift) & 3;
                    if ((hmask[position] & m) == 0) quant -= 4;
                    *output++ = dl * (float)quant;
                }
            }
            shift += 2;
            m <<= 1;
        }
        qs += 32;
    }
}

static void q2_dequant_block(const unsigned char *block, float *output) {
    const unsigned char *scales = block;
    const unsigned char *qs = block + 16;
    float d = f16_to_f32(read_u16(block + 80));
    float dmin = f16_to_f32(read_u16(block + 82));
    int scale_index = 0;
    for (int chunk = 0; chunk < INGOT_QK_K; chunk += 128) {
        int shift = 0;
        for (int j = 0; j < 4; j++) {
            for (int half = 0; half < 2; half++) {
                unsigned char sc = scales[scale_index++];
                float dl = d * (float)(sc & 0x0f);
                float ml = dmin * (float)(sc >> 4);
                for (int l = 0; l < 16; l++)
                    *output++ = dl * (float)((qs[half * 16 + l] >> shift) & 3) - ml;
            }
            shift += 2;
        }
        qs += 32;
    }
}

typedef void (*kquant_dequant_fn)(const unsigned char *, float *);

/* ── the decode-then-dot inner product ──────────────────────────────────────
 * Four accumulators: enough to hide FMA latency on both ISAs, and a reordered
 * sum is within the 1e-5 budget the parity test grants the exact path. The
 * compiler cannot do this on its own — reordering float sums needs
 * -ffast-math, which this library does not build with. */
typedef float (*qk_dot_fn)(const float *, const float *, size_t);

static float qk_dot_scalar(const float *a, const float *b, size_t n) {
    float sum = 0.0f;
    for (size_t i = 0; i < n; i++) sum += a[i] * b[i];
    return sum;
}

#if defined(INGOT_HAVE_Q4_K_NEON)
static float qk_dot_neon(const float *a, const float *b, size_t n) {
    float32x4_t acc0 = vdupq_n_f32(0.0f), acc1 = vdupq_n_f32(0.0f);
    float32x4_t acc2 = vdupq_n_f32(0.0f), acc3 = vdupq_n_f32(0.0f);
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        acc0 = vfmaq_f32(acc0, vld1q_f32(a + i),      vld1q_f32(b + i));
        acc1 = vfmaq_f32(acc1, vld1q_f32(a + i + 4),  vld1q_f32(b + i + 4));
        acc2 = vfmaq_f32(acc2, vld1q_f32(a + i + 8),  vld1q_f32(b + i + 8));
        acc3 = vfmaq_f32(acc3, vld1q_f32(a + i + 12), vld1q_f32(b + i + 12));
    }
    float sum = vaddvq_f32(vaddq_f32(vaddq_f32(acc0, acc1),
                                     vaddq_f32(acc2, acc3)));
    for (; i < n; i++) sum += a[i] * b[i];
    return sum;
}
#endif

#if defined(INGOT_HAVE_Q4_K_AVX2)
static float qk_dot_avx2(const float *a, const float *b, size_t n) {
    __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps(), acc3 = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 32 <= n; i += 32) {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i),      _mm256_loadu_ps(b + i),      acc0);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8),  _mm256_loadu_ps(b + i + 8),  acc1);
        acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 16), _mm256_loadu_ps(b + i + 16), acc2);
        acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 24), _mm256_loadu_ps(b + i + 24), acc3);
    }
    const __m256 acc = _mm256_add_ps(_mm256_add_ps(acc0, acc1),
                                     _mm256_add_ps(acc2, acc3));
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(acc),
                          _mm256_extractf128_ps(acc, 1));
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_movehdup_ps(s));
    float sum = _mm_cvtss_f32(s);
    for (; i < n; i++) sum += a[i] * b[i];
    return sum;
}
#endif

static qk_dot_fn qk_dot_pick(void) {
    const ingot_cpu_caps caps = ingot_cpu();
    (void)caps;
#if defined(INGOT_HAVE_Q4_K_NEON)
    if (caps.neon) return qk_dot_neon;
#endif
#if defined(INGOT_HAVE_Q4_K_AVX2)
    if (caps.avx2) return qk_dot_avx2;
#endif
    return qk_dot_scalar;
}

static int kquant_apply(const void *weights, size_t rows, size_t cols,
                        const float *input, float *output, size_t block_bytes,
                        kquant_dequant_fn dequant, int matvec) {
    if (weights == NULL || output == NULL || rows == 0 || cols == 0 ||
        cols % INGOT_QK_K != 0 || (matvec && input == NULL)) return -1;
    size_t blocks_per_row = cols / INGOT_QK_K;
    if (blocks_per_row > SIZE_MAX / block_bytes) return -1;
    size_t row_bytes = blocks_per_row * block_bytes;
    if (rows > SIZE_MAX / row_bytes || rows * cols > SIZE_MAX / sizeof(float)) return -1;
    const unsigned char *source = (const unsigned char *)weights;
    const qk_dot_fn dot = qk_dot_pick();
    float values[INGOT_QK_K];
    for (size_t row = 0; row < rows; row++) {
        const unsigned char *row_data = source + row * row_bytes;
        if (matvec) {
            float sum = 0.0f;
            for (size_t block = 0; block < blocks_per_row; block++) {
                dequant(row_data + block * block_bytes, values);
                sum += dot(values, input + block * INGOT_QK_K, INGOT_QK_K);
            }
            output[row] = sum;
        } else {
            for (size_t block = 0; block < blocks_per_row; block++)
                dequant(row_data + block * block_bytes,
                        output + row * cols + block * INGOT_QK_K);
        }
    }
    return 0;
}

/* ── fused Q2_K and Q3_K matvecs ────────────────────────────────────────────
 * Both types had NO vector matvec on EITHER architecture and went through
 * kquant_apply: dequantize a 256-float scratch per block, then dot it. That is
 * 1 KB written and re-read per 84 or 110 bytes of weights, the same round trip
 * removed from Q6_K and then from Q8_0, where it was worth 4.6x and 2.8x.
 *
 * Both formats share a shape: 16 groups of 16 values, each group taking one
 * scale and mapping to 16 CONTIGUOUS inputs — so a group is one accumulator and
 * the scale is applied once per 16 weights instead of once per weight.
 *
 * The 2-bit fields sit at four shifts of the same byte, and the shift is a loop
 * variable rather than a constant, so vshr_n_u8 (constant shift only) is out.
 * NEON's vshlq_u8 takes a signed shift VECTOR where negative means right, which
 * is the whole reason this reads the way it does. */
#if defined(INGOT_HAVE_Q4_K_NEON)
static float q2_k_dot_block_neon(const unsigned char *block, const float *input) {
    const unsigned char *scales = block;
    const unsigned char *qs = block + 16;
    const float d    = f16_to_f32(read_u16(block + 80));
    const float dmin = f16_to_f32(read_u16(block + 82));

    float32x4_t total = vdupq_n_f32(0.0f);
    const uint8x16_t three = vdupq_n_u8(3);
    int scale_index = 0;
    const float *in = input;

    for (int chunk = 0; chunk < INGOT_QK_K; chunk += 128) {
        for (int j = 0; j < 4; j++) {
            const int8x16_t sh = vdupq_n_s8((signed char)(-2 * j));
            for (int half = 0; half < 2; half++) {
                const unsigned char sc = scales[scale_index++];
                const uint8x16_t packed = vld1q_u8(qs + half * 16);
                const uint8x16_t q = vandq_u8(vshlq_u8(packed, sh), three);

                const uint16x8_t l16 = vmovl_u8(vget_low_u8(q));
                const uint16x8_t h16 = vmovl_u8(vget_high_u8(q));

                const float32x4_t xa = vld1q_f32(in), xb = vld1q_f32(in + 4);
                const float32x4_t xc = vld1q_f32(in + 8), xd = vld1q_f32(in + 12);

                float32x4_t qa = vdupq_n_f32(0.0f);
                qa = vmlaq_f32(qa, vcvtq_f32_u32(vmovl_u16(vget_low_u16(l16))), xa);
                qa = vmlaq_f32(qa, vcvtq_f32_u32(vmovl_u16(vget_high_u16(l16))), xb);
                qa = vmlaq_f32(qa, vcvtq_f32_u32(vmovl_u16(vget_low_u16(h16))), xc);
                qa = vmlaq_f32(qa, vcvtq_f32_u32(vmovl_u16(vget_high_u16(h16))), xd);

                const float32x4_t xsum =
                    vaddq_f32(vaddq_f32(xa, xb), vaddq_f32(xc, xd));

                total = vmlaq_n_f32(total, qa, d * (float)(sc & 0x0f));
                total = vmlaq_n_f32(total, xsum, -(dmin * (float)(sc >> 4)));
                in += 16;
            }
        }
        qs += 32;
    }
    return vaddvq_f32(total);
}

static float q3_k_dot_block_neon(const unsigned char *block, const float *input) {
    const unsigned char *hmask = block;
    const unsigned char *qs = block + 32;
    const unsigned char *scales = block + 96;
    const float d = f16_to_f32(read_u16(block + 108));

    float32x4_t total = vdupq_n_f32(0.0f);
    const uint8x16_t three = vdupq_n_u8(3);
    const uint8x16_t four  = vdupq_n_u8(4);
    const uint8x16_t zero  = vdupq_n_u8(0);
    int scale_index = 0;
    unsigned char m = 1;
    const float *in = input;

    for (int chunk = 0; chunk < INGOT_QK_K; chunk += 128) {
        for (int j = 0; j < 4; j++) {
            const int8x16_t sh = vdupq_n_s8((signed char)(-2 * j));
            const uint8x16_t mv = vdupq_n_u8(m);
            for (int half = 0; half < 2; half++) {
                const float dl = d * (float)q3_k_scale(scales, scale_index++);
                const uint8x16_t packed = vld1q_u8(qs + half * 16);
                const uint8x16_t hm = vld1q_u8(hmask + half * 16);

                /* quant is 0..3, minus 4 wherever the high-mask bit is CLEAR. */
                const uint8x16_t q = vandq_u8(vshlq_u8(packed, sh), three);
                const uint8x16_t clear = vceqq_u8(vandq_u8(hm, mv), zero);
                const int8x16_t qv = vsubq_s8(vreinterpretq_s8_u8(q),
                                              vreinterpretq_s8_u8(vandq_u8(clear, four)));

                const int16x8_t l16 = vmovl_s8(vget_low_s8(qv));
                const int16x8_t h16 = vmovl_s8(vget_high_s8(qv));

                float32x4_t qa = vdupq_n_f32(0.0f);
                qa = vmlaq_f32(qa, vcvtq_f32_s32(vmovl_s16(vget_low_s16(l16))),
                               vld1q_f32(in));
                qa = vmlaq_f32(qa, vcvtq_f32_s32(vmovl_s16(vget_high_s16(l16))),
                               vld1q_f32(in + 4));
                qa = vmlaq_f32(qa, vcvtq_f32_s32(vmovl_s16(vget_low_s16(h16))),
                               vld1q_f32(in + 8));
                qa = vmlaq_f32(qa, vcvtq_f32_s32(vmovl_s16(vget_high_s16(h16))),
                               vld1q_f32(in + 12));

                total = vmlaq_n_f32(total, qa, dl);
                in += 16;
            }
            m <<= 1;
        }
        qs += 32;
    }
    return vaddvq_f32(total);
}
#endif

#if defined(INGOT_HAVE_Q4_K_AVX2)
static inline float qk_hsum256(__m256 v) {
    __m128 h = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    h = _mm_add_ps(h, _mm_movehl_ps(h, h));
    h = _mm_add_ss(h, _mm_shuffle_ps(h, h, 0x55));
    return _mm_cvtss_f32(h);
}

static float q2_k_dot_block_avx2(const unsigned char *block, const float *input) {
    const unsigned char *scales = block;
    const unsigned char *qs = block + 16;
    const float d    = f16_to_f32(read_u16(block + 80));
    const float dmin = f16_to_f32(read_u16(block + 82));

    __m256 total = _mm256_setzero_ps();
    int scale_index = 0;
    const float *in = input;

    for (int chunk = 0; chunk < INGOT_QK_K; chunk += 128) {
        for (int j = 0; j < 4; j++) {
            const int shift = 2 * j;
            for (int half = 0; half < 2; half++) {
                const unsigned char sc = scales[scale_index++];
                const unsigned char *p = qs + half * 16;

                __m256 qa = _mm256_setzero_ps(), xs = _mm256_setzero_ps();
                for (int k = 0; k < 16; k += 8) {
                    const __m256i w = _mm256_cvtepu8_epi32(
                        _mm_loadl_epi64((const __m128i *)(const void *)(p + k)));
                    const __m256i q = _mm256_and_si256(_mm256_srli_epi32(w, shift),
                                                       _mm256_set1_epi32(3));
                    const __m256 x = _mm256_loadu_ps(in + k);
                    qa = _mm256_fmadd_ps(_mm256_cvtepi32_ps(q), x, qa);
                    xs = _mm256_add_ps(xs, x);
                }
                total = _mm256_fmadd_ps(qa, _mm256_set1_ps(d * (float)(sc & 0x0f)), total);
                total = _mm256_fmadd_ps(xs, _mm256_set1_ps(-(dmin * (float)(sc >> 4))), total);
                in += 16;
            }
        }
        qs += 32;
    }
    return qk_hsum256(total);
}

static float q3_k_dot_block_avx2(const unsigned char *block, const float *input) {
    const unsigned char *hmask = block;
    const unsigned char *qs = block + 32;
    const unsigned char *scales = block + 96;
    const float d = f16_to_f32(read_u16(block + 108));

    __m256 total = _mm256_setzero_ps();
    int scale_index = 0;
    unsigned char m = 1;
    const float *in = input;

    for (int chunk = 0; chunk < INGOT_QK_K; chunk += 128) {
        for (int j = 0; j < 4; j++) {
            const int shift = 2 * j;
            const __m256i mv = _mm256_set1_epi32((int)(unsigned int)m);
            for (int half = 0; half < 2; half++) {
                const float dl = d * (float)q3_k_scale(scales, scale_index++);
                const unsigned char *p = qs + half * 16;
                const unsigned char *h = hmask + half * 16;

                __m256 qa = _mm256_setzero_ps();
                for (int k = 0; k < 16; k += 8) {
                    const __m256i w = _mm256_cvtepu8_epi32(
                        _mm_loadl_epi64((const __m128i *)(const void *)(p + k)));
                    const __m256i hb = _mm256_cvtepu8_epi32(
                        _mm_loadl_epi64((const __m128i *)(const void *)(h + k)));
                    __m256i q = _mm256_and_si256(_mm256_srli_epi32(w, shift),
                                                 _mm256_set1_epi32(3));
                    /* 0xFFFFFFFF where the mask bit is CLEAR, then subtract 4. */
                    const __m256i clear = _mm256_cmpeq_epi32(
                        _mm256_and_si256(hb, mv), _mm256_setzero_si256());
                    q = _mm256_sub_epi32(q, _mm256_and_si256(clear, _mm256_set1_epi32(4)));
                    qa = _mm256_fmadd_ps(_mm256_cvtepi32_ps(q),
                                         _mm256_loadu_ps(in + k), qa);
                }
                total = _mm256_fmadd_ps(qa, _mm256_set1_ps(dl), total);
                in += 16;
            }
            m <<= 1;
        }
        qs += 32;
    }
    return qk_hsum256(total);
}
#endif

/* One row loop for both, since the only differences are the block size and the
 * dot. */
#define INGOT_QK_FUSED_MATVEC(NAME, BYTES, DOT)                                \
    static int NAME(const void *weights, size_t rows, size_t cols,             \
                    const float *input, float *output) {                       \
        const size_t blocks_per_row = cols / INGOT_QK_K;                       \
        if (blocks_per_row > SIZE_MAX / (BYTES)) return -1;                    \
        const size_t row_bytes = blocks_per_row * (BYTES);                     \
        if (rows > SIZE_MAX / row_bytes) return -1;                            \
        const unsigned char *source = (const unsigned char *)weights;          \
        for (size_t row = 0; row < rows; row++) {                              \
            const unsigned char *row_data = source + row * row_bytes;          \
            float sum = 0.0f;                                                  \
            for (size_t b = 0; b < blocks_per_row; b++)                        \
                sum += DOT(row_data + b * (BYTES), input + b * INGOT_QK_K);    \
            output[row] = sum;                                                 \
        }                                                                      \
        return 0;                                                              \
    }

#if defined(INGOT_HAVE_Q4_K_NEON)
INGOT_QK_FUSED_MATVEC(ingot_q2_k_matvec_neon, INGOT_Q2_K_BYTES, q2_k_dot_block_neon)
INGOT_QK_FUSED_MATVEC(ingot_q3_k_matvec_neon, INGOT_Q3_K_BYTES, q3_k_dot_block_neon)
#endif
#if defined(INGOT_HAVE_Q4_K_AVX2)
INGOT_QK_FUSED_MATVEC(ingot_q2_k_matvec_avx2, INGOT_Q2_K_BYTES, q2_k_dot_block_avx2)
INGOT_QK_FUSED_MATVEC(ingot_q3_k_matvec_avx2, INGOT_Q3_K_BYTES, q3_k_dot_block_avx2)
#endif

int ingot_q3_k_matvec(const void *weights, size_t rows, size_t cols,
                           const float *input, float *output) {
    if (!qk_args_ok(weights, input, output, rows, cols, INGOT_QK_K)) return -1;
#if defined(INGOT_HAVE_Q4_K_NEON)
    if (ingot_cpu().neon)
        return ingot_q3_k_matvec_neon(weights, rows, cols, input, output);
#endif
#if defined(INGOT_HAVE_Q4_K_AVX2)
    if (ingot_cpu().avx2)
        return ingot_q3_k_matvec_avx2(weights, rows, cols, input, output);
#endif
    return kquant_apply(weights, rows, cols, input, output,
                        INGOT_Q3_K_BYTES, q3_dequant_block, 1);
}

int ingot_q3_k_dequant(const void *weights, size_t rows, size_t cols,
                            float *output) {
    return kquant_apply(weights, rows, cols, NULL, output,
                        INGOT_Q3_K_BYTES, q3_dequant_block, 0);
}

int ingot_q2_k_matvec(const void *weights, size_t rows, size_t cols,
                           const float *input, float *output) {
    if (!qk_args_ok(weights, input, output, rows, cols, INGOT_QK_K)) return -1;
#if defined(INGOT_HAVE_Q4_K_NEON)
    if (ingot_cpu().neon)
        return ingot_q2_k_matvec_neon(weights, rows, cols, input, output);
#endif
#if defined(INGOT_HAVE_Q4_K_AVX2)
    if (ingot_cpu().avx2)
        return ingot_q2_k_matvec_avx2(weights, rows, cols, input, output);
#endif
    return kquant_apply(weights, rows, cols, input, output,
                        INGOT_Q2_K_BYTES, q2_dequant_block, 1);
}

int ingot_q2_k_dequant(const void *weights, size_t rows, size_t cols,
                            float *output) {
    return kquant_apply(weights, rows, cols, NULL, output,
                        INGOT_Q2_K_BYTES, q2_dequant_block, 0);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * GGML Q6_K — 256 values per 210-byte block, 6-bit quantization
 * ═══════════════════════════════════════════════════════════════════════════ */

#define INGOT_Q6_K_BYTES 210

/* A LAYOUT TRAP, and the reason this file has two independent decoders.
 *
 * An earlier version of this kernel read the block as
 * { half d; uint8 ql[128]; uint8 qh[64]; int8 scales[16] } and walked the
 * quants linearly. Both halves are wrong. In ggml, block_q6_K is
 * { uint8 ql[128]; uint8 qh[64]; int8 scales[16]; half d } — d is at the END,
 * the only K-quant besides Q3_K where it is — and the quants are interleaved
 * in two halves of 128, each half holding four quarters at four different bit
 * offsets of the same qh byte, with scales taken at is, is+2, is+4, is+6.
 *
 * Reading d out of ql[0..1] and the quants in the wrong order produced values
 * off by orders of magnitude, silently, on every Q6_K tensor. It survived
 * because the wrong layout still fits exactly in 210 bytes (nothing to crash
 * on) and because it was only ever compared against a matvec that shared the
 * same misreading.
 *
 * Caught by cross-checking against src/dequant.c, an independent decoder
 * validated against llama.cpp's own dequantizer. The other five K-quants
 * agree bit-for-bit between the two;
 * only this one did not. tests/test_formats.c pins the comparison. */
static void q6_dequant_block(const unsigned char *block, float *output) {
    const unsigned char *ql = block;
    const unsigned char *qh = block + 128;
    const signed char *sc = (const signed char *)(block + 192);
    const float d = f16_to_f32(read_u16(block + 208));
    float *out = output;
    for (int half = 0; half < 2; half++) {
        for (int i = 0; i < 32; i++) {
            const int is = i / 16;
            const int q1 = (int)((ql[i]      & 0x0f) | (((qh[i] >> 0) & 3) << 4)) - 32;
            const int q2 = (int)((ql[i + 32] & 0x0f) | (((qh[i] >> 2) & 3) << 4)) - 32;
            const int q3 = (int)((ql[i]      >> 4)   | (((qh[i] >> 4) & 3) << 4)) - 32;
            const int q4 = (int)((ql[i + 32] >> 4)   | (((qh[i] >> 6) & 3) << 4)) - 32;
            out[i]      = d * (float)sc[is]     * (float)q1;
            out[i + 32] = d * (float)sc[is + 2] * (float)q2;
            out[i + 64] = d * (float)sc[is + 4] * (float)q3;
            out[i + 96] = d * (float)sc[is + 6] * (float)q4;
        }
        out += 128;
        ql += 64;
        qh += 32;
        sc += 8;
    }
}


#if defined(INGOT_HAVE_Q4_K_NEON)
/* Fused Q6_K dot: decode straight into the accumulator, never into memory.
 *
 * The previous kernel went through kquant_apply, which dequantizes a block
 * into a 256-float scratch array and then dots it. That is 1 KB written and
 * re-read per 210 bytes of weights, with a scalar decode loop — and it made
 * Q6_K 2.8x slower PER ELEMENT than Q4_K on the same shape, despite moving
 * only 1.46x the bytes. Since Q6_K holds 45% of a Q4_K_M checkpoint (the
 * token embedding and every ffn_down), that gap was most of a decoder's time.
 *
 * Layout, and it is the trap documented above q6_dequant_block: quants are
 * interleaved in two halves of 128, four quarters per half taken at four bit
 * offsets of the same qh byte, scales at is, is+2, is+4, is+6 — and `d` lives
 * at the END of the block, not the start.
 *
 * Each run of 16 outputs shares one scale, so the scale multiply is hoisted
 * out of the element loop: accumulate q*x for the group, scale once. */
static float q6_k_dot_block_neon(const unsigned char *block, const float *input) {
    const unsigned char *ql = block;
    const unsigned char *qh = block + 128;
    const signed char   *sc = (const signed char *)(block + 192);
    const float d = f16_to_f32(read_u16(block + 208));

    const uint8x8_t nib  = vdup_n_u8(0x0f);
    const uint8x8_t two  = vdup_n_u8(0x03);
    const int8x8_t  bias = vdup_n_s8(32);

    float total = 0.0f;

    for (int half = 0; half < 2; half++) {
        const float *in = input + half * 128;

        /* is = i/16, so two groups of 16 per half; each group is two 8-wide
         * steps sharing one set of four scales. */
        for (int is = 0; is < 2; is++) {
            float32x4_t a1 = vdupq_n_f32(0.0f), a2 = vdupq_n_f32(0.0f);
            float32x4_t a3 = vdupq_n_f32(0.0f), a4 = vdupq_n_f32(0.0f);

            for (int k = 0; k < 16; k += 8) {
                const int i = is * 16 + k;
                const uint8x8_t la = vld1_u8(ql + i);
                const uint8x8_t lb = vld1_u8(ql + i + 32);
                const uint8x8_t h  = vld1_u8(qh + i);

                /* (low nibble | high 2 bits << 4) - 32, in [-32, 31]. */
                const int8x8_t q1 = vsub_s8(vreinterpret_s8_u8(
                    vorr_u8(vand_u8(la, nib), vshl_n_u8(vand_u8(h, two), 4))), bias);
                const int8x8_t q2 = vsub_s8(vreinterpret_s8_u8(
                    vorr_u8(vand_u8(lb, nib), vshl_n_u8(vand_u8(vshr_n_u8(h, 2), two), 4))), bias);
                const int8x8_t q3 = vsub_s8(vreinterpret_s8_u8(
                    vorr_u8(vshr_n_u8(la, 4), vshl_n_u8(vand_u8(vshr_n_u8(h, 4), two), 4))), bias);
                const int8x8_t q4 = vsub_s8(vreinterpret_s8_u8(
                    vorr_u8(vshr_n_u8(lb, 4), vshl_n_u8(vand_u8(vshr_n_u8(h, 6), two), 4))), bias);

#define Q6_ACC(acc, q, off)                                                        \
    do {                                                                           \
        const int16x8_t w_ = vmovl_s8(q);                                          \
        acc = vfmaq_f32(acc, vcvtq_f32_s32(vmovl_s16(vget_low_s16(w_))),           \
                        vld1q_f32(in + (off) + i));                                \
        acc = vfmaq_f32(acc, vcvtq_f32_s32(vmovl_s16(vget_high_s16(w_))),          \
                        vld1q_f32(in + (off) + i + 4));                            \
    } while (0)

                Q6_ACC(a1, q1, 0);
                Q6_ACC(a2, q2, 32);
                Q6_ACC(a3, q3, 64);
                Q6_ACC(a4, q4, 96);
#undef Q6_ACC
            }

            total += d * ((float)sc[is]     * vaddvq_f32(a1) +
                          (float)sc[is + 2] * vaddvq_f32(a2) +
                          (float)sc[is + 4] * vaddvq_f32(a3) +
                          (float)sc[is + 6] * vaddvq_f32(a4));
        }
        ql += 64;
        qh += 32;
        sc += 8;
    }
    return total;
}

static int ingot_q6_k_matvec_neon(const void *weights, size_t rows, size_t cols,
                                  const float *input, float *output) {
    if (weights == NULL || input == NULL || output == NULL || rows == 0 ||
        cols == 0 || cols % INGOT_QK_K != 0) return -1;
    size_t blocks_per_row = cols / INGOT_QK_K;
    if (blocks_per_row > SIZE_MAX / INGOT_Q6_K_BYTES) return -1;
    size_t row_bytes = blocks_per_row * INGOT_Q6_K_BYTES;
    if (rows > SIZE_MAX / row_bytes) return -1;
    const unsigned char *source = (const unsigned char *)weights;
    for (size_t row = 0; row < rows; row++) {
        const unsigned char *row_data = source + row * row_bytes;
        float sum = 0.0f;
        for (size_t block = 0; block < blocks_per_row; block++)
            sum += q6_k_dot_block_neon(row_data + block * INGOT_Q6_K_BYTES,
                                       input + block * INGOT_QK_K);
        output[row] = sum;
    }
    return 0;
}
#endif /* INGOT_HAVE_Q4_K_NEON */

#if defined(INGOT_HAVE_Q4_K_AVX2)
/* The AVX2 twin of the NEON kernel above, and it was worth 4.65x on an LM head
 * because until now x86 had NO vector path for Q6_K at all: the type fell
 * through to kquant_apply, which is exactly the dequantize-into-a-256-float-
 * scratch-then-dot loop that the NEON commit removed on ARM.
 *
 * Same layout traps, same shape, one x86 wrinkle worth stating: _mm_srli_epi16
 * shifts 16-bit lanes, so every byte-wise right shift has to be masked
 * afterwards or bits from the neighbouring byte walk in. The NEON side gets
 * vshr_n_u8 and needs no mask, which is precisely the kind of difference that
 * makes a "port" quietly wrong.
 *
 * Each run of 16 outputs shares one scale, so the scale multiply is hoisted out
 * of the element loop and the four group accumulators fold into the running
 * total with one FMA each. */
static inline float q6_hsum256(__m256 v) {
    __m128 a = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    a = _mm_add_ps(a, _mm_movehl_ps(a, a));
    a = _mm_add_ss(a, _mm_shuffle_ps(a, a, 0x55));
    return _mm_cvtss_f32(a);
}

static float q6_k_dot_block_avx2(const unsigned char *block, const float *input) {
    const unsigned char *ql = block;
    const unsigned char *qh = block + 128;
    const signed char   *sc = (const signed char *)(block + 192);
    const float d = f16_to_f32(read_u16(block + 208));

    const __m128i nib  = _mm_set1_epi8(0x0f);
    const __m128i two  = _mm_set1_epi8(0x03);
    const __m128i bias = _mm_set1_epi8(32);

    __m256 total = _mm256_setzero_ps();

    for (int half = 0; half < 2; half++) {
        const float *in = input + half * 128;

        for (int is = 0; is < 2; is++) {
            __m256 a1 = _mm256_setzero_ps(), a2 = _mm256_setzero_ps();
            __m256 a3 = _mm256_setzero_ps(), a4 = _mm256_setzero_ps();

            for (int k = 0; k < 16; k += 8) {
                const int i = is * 16 + k;
                const __m128i la = _mm_loadl_epi64((const __m128i *)(const void *)(ql + i));
                const __m128i lb = _mm_loadl_epi64((const __m128i *)(const void *)(ql + i + 32));
                const __m128i h  = _mm_loadl_epi64((const __m128i *)(const void *)(qh + i));

                const __m128i q1 = _mm_sub_epi8(
                    _mm_or_si128(_mm_and_si128(la, nib),
                                 _mm_slli_epi16(_mm_and_si128(h, two), 4)), bias);
                const __m128i q2 = _mm_sub_epi8(
                    _mm_or_si128(_mm_and_si128(lb, nib),
                                 _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(h, 2), two), 4)),
                    bias);
                const __m128i q3 = _mm_sub_epi8(
                    _mm_or_si128(_mm_and_si128(_mm_srli_epi16(la, 4), nib),
                                 _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(h, 4), two), 4)),
                    bias);
                const __m128i q4 = _mm_sub_epi8(
                    _mm_or_si128(_mm_and_si128(_mm_srli_epi16(lb, 4), nib),
                                 _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(h, 6), two), 4)),
                    bias);

#define Q6_ACC_AVX2(acc, q, off)                                               \
    acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q)),         \
                          _mm256_loadu_ps(in + (off) + i), acc)

                Q6_ACC_AVX2(a1, q1, 0);
                Q6_ACC_AVX2(a2, q2, 32);
                Q6_ACC_AVX2(a3, q3, 64);
                Q6_ACC_AVX2(a4, q4, 96);
#undef Q6_ACC_AVX2
            }

            total = _mm256_fmadd_ps(a1, _mm256_set1_ps(d * (float)sc[is]),     total);
            total = _mm256_fmadd_ps(a2, _mm256_set1_ps(d * (float)sc[is + 2]), total);
            total = _mm256_fmadd_ps(a3, _mm256_set1_ps(d * (float)sc[is + 4]), total);
            total = _mm256_fmadd_ps(a4, _mm256_set1_ps(d * (float)sc[is + 6]), total);
        }
        ql += 64;
        qh += 32;
        sc += 8;
    }
    return q6_hsum256(total);
}

static int ingot_q6_k_matvec_avx2(const void *weights, size_t rows, size_t cols,
                                  const float *input, float *output) {
    if (weights == NULL || input == NULL || output == NULL || rows == 0 ||
        cols == 0 || cols % INGOT_QK_K != 0) return -1;
    size_t blocks_per_row = cols / INGOT_QK_K;
    if (blocks_per_row > SIZE_MAX / INGOT_Q6_K_BYTES) return -1;
    size_t row_bytes = blocks_per_row * INGOT_Q6_K_BYTES;
    if (rows > SIZE_MAX / row_bytes) return -1;
    const unsigned char *source = (const unsigned char *)weights;
    for (size_t row = 0; row < rows; row++) {
        const unsigned char *row_data = source + row * row_bytes;
        float sum = 0.0f;
        for (size_t block = 0; block < blocks_per_row; block++)
            sum += q6_k_dot_block_avx2(row_data + block * INGOT_Q6_K_BYTES,
                                       input + block * INGOT_QK_K);
        output[row] = sum;
    }
    return 0;
}
#endif

static int ingot_q6_k_matvec_s(const void *weights, size_t rows, size_t cols,
                                     const float *input, float *output) {
    return kquant_apply(weights, rows, cols, input, output,
                        INGOT_Q6_K_BYTES, q6_dequant_block, 1);
}

int ingot_q6_k_matvec(const void *weights, size_t rows, size_t cols,
                           const float *input, float *output) {
#if defined(INGOT_HAVE_Q4_K_NEON)
    if (ingot_cpu().neon)
        return ingot_q6_k_matvec_neon(weights, rows, cols, input, output);
#endif
#if defined(INGOT_HAVE_Q4_K_AVX2)
    if (ingot_cpu().avx2)
        return ingot_q6_k_matvec_avx2(weights, rows, cols, input, output);
#endif
    return ingot_q6_k_matvec_s(weights, rows, cols, input, output);
}

int ingot_q6_k_dequant(const void *weights, size_t rows, size_t cols,
                            float *output) {
    return kquant_apply(weights, rows, cols, NULL, output,
                        INGOT_Q6_K_BYTES, q6_dequant_block, 0);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * GGML Q8_0 — 32 values per 34-byte block, 8-bit quantization
 * ═══════════════════════════════════════════════════════════════════════════ */

#define INGOT_Q8_0_K 32

/* ── Q4_0 ─────────────────────────────────────────────────────────────────────
 * 32 weights in 18 bytes: { half d; uint8 qs[16] }. Element order is
 * SPLIT-HALF, not interleaved — qs[i] & 0x0f is element i, qs[i] >> 4 is
 * element i + 16 — and the stored nibble is biased by 8, so the value is
 * (nibble - 8) * d.
 *
 * The SIMD shape here is qwen-tts's (load 16 bytes, split the nibbles, widen,
 * FMA against several accumulators so the chain is not latency-bound), but NOT
 * its element order: qwen-tts exports its own checkpoints with lo/hi
 * interleaved and zips them back together with vzipq_s16. ggml does not, and
 * copying that zip would have produced a kernel that runs, matches nothing,
 * and is wrong in the same silent way the Q6_K layout trap was. The ggml order
 * is the friendlier one anyway: low nibbles map contiguously onto x[0..15] and
 * high nibbles onto x[16..31], so no shuffle is needed at all.
 *
 * Q4_0 mattered enough to write because it is what Google's Gemma 4 QAT
 * checkpoints ship as — a model that would otherwise have run on the
 * decode-a-row-then-dot generic path. */
#define INGOT_Q4_0_K     32
#define INGOT_Q4_0_BYTES 18

static void q4_0_dequant_block(const unsigned char *block, float *output) {
    const float d = f16_to_f32(read_u16(block));
    const unsigned char *q = block + 2;
    for (int i = 0; i < 16; i++) {
        output[i]      = d * (float)((int)(q[i] & 0x0f) - 8);
        output[i + 16] = d * (float)((int)(q[i] >> 4)   - 8);
    }
}

#if defined(INGOT_HAVE_Q4_K_NEON)
static float q4_0_dot_block_neon(const unsigned char *block, const float *x) {
    const float d = f16_to_f32(read_u16(block));
    const uint8x16_t raw = vld1q_u8(block + 2);
    const uint8x8_t  eight = vdup_n_u8(8);

    /* (nibble - 8) via an unsigned widening subtract, then reinterpreted: the
     * result 0..15 minus 8 wraps to the right two's-complement int16. */
    const uint8x16_t lo = vandq_u8(raw, vdupq_n_u8(0x0f));
    const uint8x16_t hi = vshrq_n_u8(raw, 4);
    const int16x8_t l0 = vreinterpretq_s16_u16(vsubl_u8(vget_low_u8(lo),  eight));
    const int16x8_t l1 = vreinterpretq_s16_u16(vsubl_u8(vget_high_u8(lo), eight));
    const int16x8_t h0 = vreinterpretq_s16_u16(vsubl_u8(vget_low_u8(hi),  eight));
    const int16x8_t h1 = vreinterpretq_s16_u16(vsubl_u8(vget_high_u8(hi), eight));

    float32x4_t a0 = vdupq_n_f32(0.0f), a1 = vdupq_n_f32(0.0f);
    float32x4_t a2 = vdupq_n_f32(0.0f), a3 = vdupq_n_f32(0.0f);

#define Q40_FMA(acc, v, half, off)                                             \
    acc = vfmaq_f32(acc, vcvtq_f32_s32(vmovl_s16(vget_##half##_s16(v))),       \
                    vld1q_f32(x + (off)))
    Q40_FMA(a0, l0, low,   0);  Q40_FMA(a1, l0, high,  4);
    Q40_FMA(a2, l1, low,   8);  Q40_FMA(a3, l1, high, 12);
    Q40_FMA(a0, h0, low,  16);  Q40_FMA(a1, h0, high, 20);
    Q40_FMA(a2, h1, low,  24);  Q40_FMA(a3, h1, high, 28);
#undef Q40_FMA

    /* One scale multiply per block instead of 32: every element in a Q4_0
     * block shares d. */
    return d * vaddvq_f32(vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3)));
}

static int ingot_q4_0_matvec_neon(const void *weights, size_t rows, size_t cols,
                                  const float *input, float *output) {
    const size_t blocks = cols / INGOT_Q4_0_K;
    const size_t row_bytes = blocks * INGOT_Q4_0_BYTES;
    const unsigned char *source = (const unsigned char *)weights;
    for (size_t row = 0; row < rows; row++) {
        const unsigned char *row_data = source + row * row_bytes;
        float sum = 0.0f;
        for (size_t b = 0; b < blocks; b++)
            sum += q4_0_dot_block_neon(row_data + b * INGOT_Q4_0_BYTES,
                                       input + b * INGOT_Q4_0_K);
        output[row] = sum;
    }
    return 0;
}
#endif

#if defined(INGOT_HAVE_Q4_K_AVX2)
static float q4_0_dot_block_avx2(const unsigned char *block, const float *x) {
    const float d = f16_to_f32(read_u16(block));
    const __m128i raw = _mm_loadu_si128((const __m128i *)(block + 2));
    const __m128i lo  = _mm_and_si128(raw, _mm_set1_epi8(0x0f));
    const __m128i hi  = _mm_and_si128(_mm_srli_epi16(raw, 4), _mm_set1_epi8(0x0f));
    const __m256i bias = _mm256_set1_epi32(8);

    __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();

#define Q40_AVX(acc, src, sel, off)                                            \
    do {                                                                       \
        const __m256i w_ = _mm256_sub_epi32(                                   \
            _mm256_cvtepu8_epi32(_mm_srli_si128(src, sel)), bias);             \
        acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(w_),                          \
                              _mm256_loadu_ps(x + (off)), acc);                \
    } while (0)
    Q40_AVX(acc0, lo, 0,  0);  Q40_AVX(acc1, lo, 8,  8);
    Q40_AVX(acc0, hi, 0, 16);  Q40_AVX(acc1, hi, 8, 24);
#undef Q40_AVX

    const __m256 sum = _mm256_add_ps(acc0, acc1);
    __m128 v = _mm_add_ps(_mm256_castps256_ps128(sum), _mm256_extractf128_ps(sum, 1));
    v = _mm_hadd_ps(v, v);
    v = _mm_hadd_ps(v, v);
    return d * _mm_cvtss_f32(v);
}

static int ingot_q4_0_matvec_avx2(const void *weights, size_t rows, size_t cols,
                                  const float *input, float *output) {
    const size_t blocks = cols / INGOT_Q4_0_K;
    const size_t row_bytes = blocks * INGOT_Q4_0_BYTES;
    const unsigned char *source = (const unsigned char *)weights;
    for (size_t row = 0; row < rows; row++) {
        const unsigned char *row_data = source + row * row_bytes;
        float sum = 0.0f;
        for (size_t b = 0; b < blocks; b++)
            sum += q4_0_dot_block_avx2(row_data + b * INGOT_Q4_0_BYTES,
                                       input + b * INGOT_Q4_0_K);
        output[row] = sum;
    }
    return 0;
}
#endif

static int q4_0_args_ok(const void *w, const float *in, const float *out,
                        size_t rows, size_t cols) {
    if (w == NULL || out == NULL || rows == 0 || cols == 0) return 0;
    if (cols % INGOT_Q4_0_K != 0) return 0;
    if (in == NULL) return 0;
    const size_t blocks = cols / INGOT_Q4_0_K;
    if (blocks > SIZE_MAX / INGOT_Q4_0_BYTES) return 0;
    if (rows > SIZE_MAX / (blocks * INGOT_Q4_0_BYTES)) return 0;
    return 1;
}

static int ingot_q4_0_matvec_scalar(const void *weights, size_t rows, size_t cols,
                                    const float *input, float *output) {
    const size_t blocks = cols / INGOT_Q4_0_K;
    const size_t row_bytes = blocks * INGOT_Q4_0_BYTES;
    const unsigned char *source = (const unsigned char *)weights;
    float values[INGOT_Q4_0_K];
    for (size_t row = 0; row < rows; row++) {
        const unsigned char *row_data = source + row * row_bytes;
        float sum = 0.0f;
        for (size_t b = 0; b < blocks; b++) {
            q4_0_dequant_block(row_data + b * INGOT_Q4_0_BYTES, values);
            const float *x = input + b * INGOT_Q4_0_K;
            for (int i = 0; i < INGOT_Q4_0_K; i++) sum += values[i] * x[i];
        }
        output[row] = sum;
    }
    return 0;
}

int ingot_q4_0_matvec(const void *weights, size_t rows, size_t cols,
                      const float *input, float *output) {
    if (!q4_0_args_ok(weights, input, output, rows, cols)) return -1;
#if defined(INGOT_HAVE_Q4_K_NEON)
    if (ingot_cpu().neon)
        return ingot_q4_0_matvec_neon(weights, rows, cols, input, output);
#endif
#if defined(INGOT_HAVE_Q4_K_AVX2)
    if (ingot_cpu().avx2)
        return ingot_q4_0_matvec_avx2(weights, rows, cols, input, output);
#endif
    return ingot_q4_0_matvec_scalar(weights, rows, cols, input, output);
}

int ingot_q4_0_dequant(const void *weights, size_t rows, size_t cols,
                       float *output) {
    if (weights == NULL || output == NULL || rows == 0 || cols == 0) return -1;
    if (cols % INGOT_Q4_0_K != 0) return -1;
    const size_t blocks = cols / INGOT_Q4_0_K;
    if (blocks > SIZE_MAX / INGOT_Q4_0_BYTES) return -1;
    const size_t row_bytes = blocks * INGOT_Q4_0_BYTES;
    if (rows > SIZE_MAX / row_bytes || rows > SIZE_MAX / cols) return -1;
    const unsigned char *source = (const unsigned char *)weights;
    for (size_t row = 0; row < rows; row++)
        for (size_t b = 0; b < blocks; b++)
            q4_0_dequant_block(source + row * row_bytes + b * INGOT_Q4_0_BYTES,
                               output + row * cols + b * INGOT_Q4_0_K);
    return 0;
}

#define INGOT_Q8_0_BYTES 34

static void q8_0_dequant_block(const unsigned char *block, float *output) {
    float d = f16_to_f32(read_u16(block));
    const signed char *qs = (const signed char *)(block + 2);
    for (int i = 0; i < INGOT_Q8_0_K; i++)
        output[i] = (float)qs[i] * d;
}

/* ── fused Q8_0 matvec ──────────────────────────────────────────────────────
 * Q8_0 is the simplest format in the file — 32 int8 and one f16 scale, w = d*q
 * — and until now it was the SLOWEST per element of the K-quants, because it
 * was the only one still going through dequantize-a-block-into-scratch and dot
 * it. That is 128 bytes written and re-read per 34 bytes of weights, and it is
 * the same round trip the Q6_K commit removed.
 *
 * Measured on a 100352 x 1024 LM head at Q8_0: 14.8 G elem/s, against Q6_K's
 * 24-30 on the identical shape. A format that decodes with one multiply had no
 * business being half the speed of one that decodes from three bit planes.
 *
 * The scale is applied ONCE per 32-weight block against the block accumulator,
 * and the block accumulators fold into a row-level vector, so a row costs one
 * horizontal reduction rather than one per block. */
#if defined(INGOT_HAVE_Q4_K_NEON)
static int ingot_q8_0_matvec_neon(const void *weights, size_t rows, size_t cols,
                                  const float *input, float *output) {
    const size_t blocks = cols / INGOT_Q8_0_K;
    const size_t row_bytes = blocks * INGOT_Q8_0_BYTES;
    const unsigned char *source = (const unsigned char *)weights;

    for (size_t row = 0; row < rows; row++) {
        const unsigned char *row_data = source + row * row_bytes;
        float32x4_t total = vdupq_n_f32(0.0f);

        for (size_t b = 0; b < blocks; b++) {
            const unsigned char *block = row_data + b * INGOT_Q8_0_BYTES;
            const float d = f16_to_f32(read_u16(block));
            const signed char *qs = (const signed char *)(block + 2);
            const float *in = input + b * INGOT_Q8_0_K;

            float32x4_t acc = vdupq_n_f32(0.0f);
            for (int i = 0; i < INGOT_Q8_0_K; i += 16) {
                const int8x16_t v  = vld1q_s8(qs + i);
                const int16x8_t lo = vmovl_s8(vget_low_s8(v));
                const int16x8_t hi = vmovl_s8(vget_high_s8(v));
                acc = vmlaq_f32(acc, vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo))),
                                vld1q_f32(in + i));
                acc = vmlaq_f32(acc, vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo))),
                                vld1q_f32(in + i + 4));
                acc = vmlaq_f32(acc, vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi))),
                                vld1q_f32(in + i + 8));
                acc = vmlaq_f32(acc, vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi))),
                                vld1q_f32(in + i + 12));
            }
            total = vmlaq_n_f32(total, acc, d);
        }
        output[row] = vaddvq_f32(total);
    }
    return 0;
}
#endif

#if defined(INGOT_HAVE_Q4_K_AVX2)
static int ingot_q8_0_matvec_avx2(const void *weights, size_t rows, size_t cols,
                                  const float *input, float *output) {
    const size_t blocks = cols / INGOT_Q8_0_K;
    const size_t row_bytes = blocks * INGOT_Q8_0_BYTES;
    const unsigned char *source = (const unsigned char *)weights;

    for (size_t row = 0; row < rows; row++) {
        const unsigned char *row_data = source + row * row_bytes;
        __m256 total = _mm256_setzero_ps();

        for (size_t b = 0; b < blocks; b++) {
            const unsigned char *block = row_data + b * INGOT_Q8_0_BYTES;
            const float d = f16_to_f32(read_u16(block));
            const signed char *qs = (const signed char *)(block + 2);
            const float *in = input + b * INGOT_Q8_0_K;

            __m256 acc = _mm256_setzero_ps();
            for (int i = 0; i < INGOT_Q8_0_K; i += 8) {
                const __m128i v = _mm_loadl_epi64((const __m128i *)(const void *)(qs + i));
                acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(v)),
                                      _mm256_loadu_ps(in + i), acc);
            }
            total = _mm256_fmadd_ps(acc, _mm256_set1_ps(d), total);
        }
        __m128 h = _mm_add_ps(_mm256_castps256_ps128(total),
                              _mm256_extractf128_ps(total, 1));
        h = _mm_add_ps(h, _mm_movehl_ps(h, h));
        h = _mm_add_ss(h, _mm_shuffle_ps(h, h, 0x55));
        output[row] = _mm_cvtss_f32(h);
    }
    return 0;
}
#endif

static int ingot_q8_0_matvec_s(const void *weights, size_t rows, size_t cols,
                                     const float *input, float *output) {
    if (weights == NULL || input == NULL || output == NULL || rows == 0 ||
        cols == 0 || cols % INGOT_Q8_0_K != 0) return -1;
    size_t blocks = cols / INGOT_Q8_0_K;
    if (blocks > SIZE_MAX / INGOT_Q8_0_BYTES) return -1;
    size_t row_bytes = blocks * INGOT_Q8_0_BYTES;
    if (rows > SIZE_MAX / row_bytes) return -1;
    const unsigned char *source = (const unsigned char *)weights;
    const qk_dot_fn dot = qk_dot_pick();
    float values[INGOT_Q8_0_K];
    for (size_t row = 0; row < rows; row++) {
        const unsigned char *row_data = source + row * row_bytes;
        float sum = 0.0f;
        for (size_t b = 0; b < blocks; b++) {
            q8_0_dequant_block(row_data + b * INGOT_Q8_0_BYTES, values);
            sum += dot(values, input + b * INGOT_Q8_0_K, INGOT_Q8_0_K);
        }
        output[row] = sum;
    }
    return 0;
}

int ingot_q8_0_matvec(const void *weights, size_t rows, size_t cols,
                           const float *input, float *output) {
    /* The argument check lived only in the scalar body, so it has to happen
     * before the dispatch now that there is something to dispatch to. */
    if (weights == NULL || input == NULL || output == NULL || rows == 0 ||
        cols == 0 || cols % INGOT_Q8_0_K != 0) return -1;
    {
        const size_t blocks = cols / INGOT_Q8_0_K;
        if (blocks > SIZE_MAX / INGOT_Q8_0_BYTES) return -1;
        if (rows > SIZE_MAX / (blocks * INGOT_Q8_0_BYTES)) return -1;
    }
#if defined(INGOT_HAVE_Q4_K_NEON)
    if (ingot_cpu().neon)
        return ingot_q8_0_matvec_neon(weights, rows, cols, input, output);
#endif
#if defined(INGOT_HAVE_Q4_K_AVX2)
    if (ingot_cpu().avx2)
        return ingot_q8_0_matvec_avx2(weights, rows, cols, input, output);
#endif
    return ingot_q8_0_matvec_s(weights, rows, cols, input, output);
}

/* A STRIDE TRAP, and the reason Q8_0 dequant has its own test. An earlier
 * version of this got it wrong and nothing caught it: kquant_apply
 * strides by 256 ELEMENTS per decoder call, so it has to be handed the
 * 256-wide decoder and the 272-byte stride (eight 34-byte blocks), exactly as
 * ingot_q8_0_matmat does. Handing it the 32-value block / 34-byte pair made it
 * read one eighth of every row and leave seven eighths of the output
 * uninitialised. Pinned by tests/test_quant.c: matvec vs dequant-then-dot. */
static void q8_0_decode_super(const unsigned char *block, float *output);

int ingot_q8_0_dequant(const void *weights, size_t rows, size_t cols,
                            float *output) {
    return kquant_apply(weights, rows, cols, NULL, output,
                        (INGOT_QK_K / INGOT_Q8_0_K) * INGOT_Q8_0_BYTES,
                        q8_0_decode_super, 0);
}

/* The other k-quants through the same batched core.  Their decoders are the
 * scalar ones (only Q4_K has a NEON dequant), which costs more per block but is
 * paid once per token tile instead of once per token, so the batched form still
 * wins by the tile width.  Q4_K_M checkpoints carry a handful of Q5_K layers,
 * and those used to be the only ones left on the per-token path. */
static int q5_k_matmat_maybe_int8(const void *weights, size_t rows, size_t cols,
                                  const float *input, float *output,
                                  size_t tokens, int allow_int8) {
    if (tokens == 1)
        return ingot_q5_k_matvec(weights, rows, cols, input, output);
#if defined(INGOT_HAVE_Q4_K_SDOT) || defined(INGOT_HAVE_QK_X86INT8)
    /* Worth wiring for this model in particular: a Q4_K_M checkpoint carries
     * 63 Q5_K tensors — the first three blocks whole, all of txtfusion and the
     * final projection — so leaving Q5_K on the f32 path would have left most
     * of the early network out of the int8 route. */
    if (allow_int8 && qk_sdot_batched() && qk_int8_ready() &&
        weights != NULL && input != NULL && output != NULL &&
        rows != 0 && cols != 0 && cols % INGOT_QK_K == 0 &&
        qk_matmat_sdot(weights, rows, cols, input, output, tokens,
                       INGOT_Q5_K_BYTES, q5_k_quants) == 0)
        return 0;
#else
    (void)allow_int8;
#endif
    return qk_matmat(weights, rows, cols, input, output, tokens,
                     INGOT_Q5_K_BYTES, q5_dequant_block);
}

int ingot_q5_k_matmat(const void *weights, size_t rows, size_t cols,
                           const float *input, float *output, size_t tokens) {
    return q5_k_matmat_maybe_int8(weights, rows, cols, input, output, tokens, 1);
}

int ingot_q5_k_matmat_exact(const void *weights, size_t rows, size_t cols,
                                 const float *input, float *output,
                                 size_t tokens) {
    return q5_k_matmat_maybe_int8(weights, rows, cols, input, output, tokens, 0);
}

int ingot_q3_k_matmat(const void *weights, size_t rows, size_t cols,
                           const float *input, float *output, size_t tokens) {
    if (tokens == 1)
        return ingot_q3_k_matvec(weights, rows, cols, input, output);
    return qk_matmat(weights, rows, cols, input, output, tokens,
                     INGOT_Q3_K_BYTES, q3_dequant_block);
}

int ingot_q2_k_matmat(const void *weights, size_t rows, size_t cols,
                           const float *input, float *output, size_t tokens) {
    if (tokens == 1)
        return ingot_q2_k_matvec(weights, rows, cols, input, output);
    return qk_matmat(weights, rows, cols, input, output, tokens,
                     INGOT_Q2_K_BYTES, q2_dequant_block);
}

/* Q6_K and Q8_0 through the same batched core. Q8_0 is the odd one: its block
 * holds 32 values, not 256, so eight of them are wrapped into one 256-wide
 * decode (8 * 34 bytes) and the core does not need to know. */
int ingot_q6_k_matmat(const void *weights, size_t rows, size_t cols,
                           const float *input, float *output, size_t tokens) {
    if (tokens == 1)
        return ingot_q6_k_matvec(weights, rows, cols, input, output);
    return qk_matmat(weights, rows, cols, input, output, tokens,
                     INGOT_Q6_K_BYTES, q6_dequant_block);
}

static void q8_0_decode_super(const unsigned char *block, float *output) {
    for (int i = 0; i < INGOT_QK_K / INGOT_Q8_0_K; i++)
        q8_0_dequant_block(block + (size_t)i * INGOT_Q8_0_BYTES,
                           output + (size_t)i * INGOT_Q8_0_K);
}

int ingot_q8_0_matmat(const void *weights, size_t rows, size_t cols,
                           const float *input, float *output, size_t tokens) {
    if (tokens == 1)
        return ingot_q8_0_matvec(weights, rows, cols, input, output);
    return qk_matmat(weights, rows, cols, input, output, tokens,
                     (INGOT_QK_K / INGOT_Q8_0_K) * INGOT_Q8_0_BYTES,
                     q8_0_decode_super);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Dense F16 / BF16 — the weights exactly as stored, no conversion pass
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * The alternative these kernels remove is the load-time f32 blow-up: a BF16
 * checkpoint converted on open doubles its footprint before the first token.
 * Multiplying THROUGH the stored bytes keeps the mapping zero-copy and reads
 * half the memory per dot, which is what a bandwidth-bound matvec actually
 * pays for. Widening BF16 (a 16-bit shift) and F16 (FCVTL / VCVTPH2PS) is
 * exact, so the result is ordinary f32 arithmetic on the stored values — a
 * reordered sum, never an approximation.
 *
 * Batched form: four tokens share every widened weight vector, so the widen
 * cost is divided by the register tile the way the quantized GEMM divides its
 * decode cost. No BFDOT/BFMMLA here yet: both need the ACTIVATIONS truncated
 * to bf16 too, which is a precision decision the consumer should opt into,
 * not inherit — it goes in with a bench when a machine with FEAT_BF16 is on
 * the desk. */

static inline float dense_at(const unsigned char *p, int f16) {
    const uint16_t v = (uint16_t)((uint16_t)p[0] | ((uint16_t)p[1] << 8));
    return f16 ? ingot_f16_to_f32(v) : ingot_bf16_to_f32(v);
}

typedef struct {
    const unsigned char *w;
    const float *input;          /* [tokens][cols] row-major */
    float *output;               /* [tokens][rows] row-major */
    size_t rows, cols, tokens;
    int f16;
} dense_job_t;

static void dense_rows(size_t begin, size_t end, void *user) {
    const dense_job_t *job = user;
    const size_t cols = job->cols, tokens = job->tokens;
    const int f16 = job->f16;

#if defined(INGOT_HAVE_Q4_K_NEON) && defined(__aarch64__)
    if (ingot_cpu().neon) {
        for (size_t row = begin; row < end; row++) {
            const unsigned char *wrow = job->w + row * cols * 2;
            size_t t = 0;
            for (; t + 4 <= tokens; t += 4) {
                const float *x0 = job->input + t * cols;
                const float *x1 = x0 + cols, *x2 = x1 + cols, *x3 = x2 + cols;
                float32x4_t a0 = vdupq_n_f32(0.0f), a1 = a0, a2 = a0, a3 = a0;
                size_t c = 0;
                for (; c + 8 <= cols; c += 8) {
                    float32x4_t wlo, whi;
                    if (f16) {
                        /* byte load: wrow follows the file layout and owes
                         * no 2-byte alignment (same fix as the dtype.c lanes) */
                        const float16x8_t h = vreinterpretq_f16_u8(
                            vld1q_u8(wrow + 2 * c));
                        wlo = vcvt_f32_f16(vget_low_f16(h));
                        whi = vcvt_f32_f16(vget_high_f16(h));
                    } else {
                        const uint16x8_t h = vreinterpretq_u16_u8(
                            vld1q_u8(wrow + 2 * c));
                        wlo = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(h), 16));
                        whi = vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(h), 16));
                    }
                    a0 = vfmaq_f32(a0, wlo, vld1q_f32(x0 + c));
                    a0 = vfmaq_f32(a0, whi, vld1q_f32(x0 + c + 4));
                    a1 = vfmaq_f32(a1, wlo, vld1q_f32(x1 + c));
                    a1 = vfmaq_f32(a1, whi, vld1q_f32(x1 + c + 4));
                    a2 = vfmaq_f32(a2, wlo, vld1q_f32(x2 + c));
                    a2 = vfmaq_f32(a2, whi, vld1q_f32(x2 + c + 4));
                    a3 = vfmaq_f32(a3, wlo, vld1q_f32(x3 + c));
                    a3 = vfmaq_f32(a3, whi, vld1q_f32(x3 + c + 4));
                }
                float s0 = vaddvq_f32(a0), s1 = vaddvq_f32(a1);
                float s2 = vaddvq_f32(a2), s3 = vaddvq_f32(a3);
                for (; c < cols; c++) {
                    const float wv = dense_at(wrow + 2 * c, f16);
                    s0 += wv * x0[c]; s1 += wv * x1[c];
                    s2 += wv * x2[c]; s3 += wv * x3[c];
                }
                job->output[(t + 0) * job->rows + row] = s0;
                job->output[(t + 1) * job->rows + row] = s1;
                job->output[(t + 2) * job->rows + row] = s2;
                job->output[(t + 3) * job->rows + row] = s3;
            }
            for (; t < tokens; t++) {
                const float *x = job->input + t * cols;
                float32x4_t a0 = vdupq_n_f32(0.0f), a1 = vdupq_n_f32(0.0f);
                size_t c = 0;
                for (; c + 8 <= cols; c += 8) {
                    float32x4_t wlo, whi;
                    if (f16) {
                        /* byte load: wrow follows the file layout and owes
                         * no 2-byte alignment (same fix as the dtype.c lanes) */
                        const float16x8_t h = vreinterpretq_f16_u8(
                            vld1q_u8(wrow + 2 * c));
                        wlo = vcvt_f32_f16(vget_low_f16(h));
                        whi = vcvt_f32_f16(vget_high_f16(h));
                    } else {
                        const uint16x8_t h = vreinterpretq_u16_u8(
                            vld1q_u8(wrow + 2 * c));
                        wlo = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(h), 16));
                        whi = vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(h), 16));
                    }
                    a0 = vfmaq_f32(a0, wlo, vld1q_f32(x + c));
                    a1 = vfmaq_f32(a1, whi, vld1q_f32(x + c + 4));
                }
                float s = vaddvq_f32(vaddq_f32(a0, a1));
                for (; c < cols; c++) s += dense_at(wrow + 2 * c, f16) * x[c];
                job->output[t * job->rows + row] = s;
            }
        }
        return;
    }
#endif

#if defined(INGOT_HAVE_Q4_K_AVX2)
    {
        int simd = ingot_cpu().avx2;
#if defined(__F16C__)
        if (f16 && !ingot_cpu().f16c) simd = 0;
#else
        if (f16) simd = 0;
#endif
        if (simd) {
            for (size_t row = begin; row < end; row++) {
                const unsigned char *wrow = job->w + row * cols * 2;
                size_t t = 0;
                for (; t + 4 <= tokens; t += 4) {
                    const float *x0 = job->input + t * cols;
                    const float *x1 = x0 + cols, *x2 = x1 + cols, *x3 = x2 + cols;
                    __m256 a0 = _mm256_setzero_ps(), a1 = a0, a2 = a0, a3 = a0;
                    size_t c = 0;
                    for (; c + 8 <= cols; c += 8) {
                        const __m128i h = _mm_loadu_si128(
                            (const __m128i *)(wrow + 2 * c));
                        __m256 w8;
#if defined(__F16C__)
                        if (f16) w8 = _mm256_cvtph_ps(h); else
#endif
                        w8 = _mm256_castsi256_ps(_mm256_slli_epi32(
                                 _mm256_cvtepu16_epi32(h), 16));
                        a0 = _mm256_fmadd_ps(w8, _mm256_loadu_ps(x0 + c), a0);
                        a1 = _mm256_fmadd_ps(w8, _mm256_loadu_ps(x1 + c), a1);
                        a2 = _mm256_fmadd_ps(w8, _mm256_loadu_ps(x2 + c), a2);
                        a3 = _mm256_fmadd_ps(w8, _mm256_loadu_ps(x3 + c), a3);
                    }
                    float s0, s1, s2, s3;
                    {
                        __m128 r0 = _mm_add_ps(_mm256_castps256_ps128(a0),
                                               _mm256_extractf128_ps(a0, 1));
                        __m128 r1 = _mm_add_ps(_mm256_castps256_ps128(a1),
                                               _mm256_extractf128_ps(a1, 1));
                        __m128 r2 = _mm_add_ps(_mm256_castps256_ps128(a2),
                                               _mm256_extractf128_ps(a2, 1));
                        __m128 r3 = _mm_add_ps(_mm256_castps256_ps128(a3),
                                               _mm256_extractf128_ps(a3, 1));
                        r0 = _mm_add_ps(r0, _mm_movehl_ps(r0, r0));
                        r1 = _mm_add_ps(r1, _mm_movehl_ps(r1, r1));
                        r2 = _mm_add_ps(r2, _mm_movehl_ps(r2, r2));
                        r3 = _mm_add_ps(r3, _mm_movehl_ps(r3, r3));
                        s0 = _mm_cvtss_f32(_mm_add_ss(r0, _mm_movehdup_ps(r0)));
                        s1 = _mm_cvtss_f32(_mm_add_ss(r1, _mm_movehdup_ps(r1)));
                        s2 = _mm_cvtss_f32(_mm_add_ss(r2, _mm_movehdup_ps(r2)));
                        s3 = _mm_cvtss_f32(_mm_add_ss(r3, _mm_movehdup_ps(r3)));
                    }
                    for (; c < cols; c++) {
                        const float wv = dense_at(wrow + 2 * c, f16);
                        s0 += wv * x0[c]; s1 += wv * x1[c];
                        s2 += wv * x2[c]; s3 += wv * x3[c];
                    }
                    job->output[(t + 0) * job->rows + row] = s0;
                    job->output[(t + 1) * job->rows + row] = s1;
                    job->output[(t + 2) * job->rows + row] = s2;
                    job->output[(t + 3) * job->rows + row] = s3;
                }
                for (; t < tokens; t++) {
                    const float *x = job->input + t * cols;
                    __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
                    size_t c = 0;
                    for (; c + 16 <= cols; c += 16) {
                        const __m128i h0 = _mm_loadu_si128(
                            (const __m128i *)(wrow + 2 * c));
                        const __m128i h1 = _mm_loadu_si128(
                            (const __m128i *)(wrow + 2 * c + 16));
                        __m256 w0, w1;
#if defined(__F16C__)
                        if (f16) { w0 = _mm256_cvtph_ps(h0); w1 = _mm256_cvtph_ps(h1); } else
#endif
                        {
                            w0 = _mm256_castsi256_ps(_mm256_slli_epi32(
                                     _mm256_cvtepu16_epi32(h0), 16));
                            w1 = _mm256_castsi256_ps(_mm256_slli_epi32(
                                     _mm256_cvtepu16_epi32(h1), 16));
                        }
                        a0 = _mm256_fmadd_ps(w0, _mm256_loadu_ps(x + c), a0);
                        a1 = _mm256_fmadd_ps(w1, _mm256_loadu_ps(x + c + 8), a1);
                    }
                    const __m256 acc = _mm256_add_ps(a0, a1);
                    __m128 r = _mm_add_ps(_mm256_castps256_ps128(acc),
                                          _mm256_extractf128_ps(acc, 1));
                    r = _mm_add_ps(r, _mm_movehl_ps(r, r));
                    float s = _mm_cvtss_f32(_mm_add_ss(r, _mm_movehdup_ps(r)));
                    for (; c < cols; c++) s += dense_at(wrow + 2 * c, f16) * x[c];
                    job->output[t * job->rows + row] = s;
                }
            }
            return;
        }
    }
#endif

    for (size_t row = begin; row < end; row++) {
        const unsigned char *wrow = job->w + row * cols * 2;
        for (size_t t = 0; t < tokens; t++) {
            const float *x = job->input + t * cols;
            float s = 0.0f;
            for (size_t c = 0; c < cols; c++)
                s += dense_at(wrow + 2 * c, f16) * x[c];
            job->output[t * job->rows + row] = s;
        }
    }
}

static int dense_args_ok(const void *weights, const float *input, float *output,
                         size_t rows, size_t cols, size_t tokens) {
    if (weights == NULL || input == NULL || output == NULL ||
        rows == 0 || cols == 0 || tokens == 0) return 0;
    if (cols > SIZE_MAX / 2 || rows > SIZE_MAX / (cols * 2)) return 0;
    if (tokens > SIZE_MAX / cols) return 0;
    return 1;
}

static int dense_matvec(const void *weights, size_t rows, size_t cols,
                        const float *input, float *output, int f16) {
    if (!dense_args_ok(weights, input, output, rows, cols, 1)) return -1;
    dense_job_t job = {(const unsigned char *)weights, input, output,
                       rows, cols, 1, f16};
    /* Serial on purpose, like every other matvec here: the consumer splits
     * rows over its own pool, and a nested parallel_for into that same pool
     * is a deadlock waiting for a schedule. */
    dense_rows(0, rows, &job);
    return 0;
}

static int dense_matmat(const void *weights, size_t rows, size_t cols,
                        const float *input, float *output, size_t tokens,
                        int f16) {
    if (!dense_args_ok(weights, input, output, rows, cols, tokens)) return -1;
    dense_job_t job = {(const unsigned char *)weights, input, output,
                       rows, cols, tokens, f16};
    ingot_parallel_for(rows, dense_rows, &job);
    return 0;
}

int ingot_bf16_matvec(const void *weights, size_t rows, size_t cols,
                      const float *input, float *output) {
    return dense_matvec(weights, rows, cols, input, output, 0);
}
int ingot_bf16_matmat(const void *weights, size_t rows, size_t cols,
                      const float *input, float *output, size_t tokens) {
    return dense_matmat(weights, rows, cols, input, output, tokens, 0);
}
int ingot_f16_matvec(const void *weights, size_t rows, size_t cols,
                     const float *input, float *output) {
    return dense_matvec(weights, rows, cols, input, output, 1);
}
int ingot_f16_matmat(const void *weights, size_t rows, size_t cols,
                     const float *input, float *output, size_t tokens) {
    return dense_matmat(weights, rows, cols, input, output, tokens, 1);
}

/* The dequant twins ride the bulk converters from dtype.c. */
int ingot_bf16_dequant(const void *weights, size_t rows, size_t cols,
                       float *output) {
    if (weights == NULL || output == NULL || rows == 0 || cols == 0) return -1;
    if (cols > SIZE_MAX / 2 || rows > SIZE_MAX / (cols * 2)) return -1;
    ingot_bf16_block_to_f32((const unsigned char *)weights, rows * cols, output);
    return 0;
}
int ingot_f16_dequant(const void *weights, size_t rows, size_t cols,
                      float *output) {
    if (weights == NULL || output == NULL || rows == 0 || cols == 0) return -1;
    if (cols > SIZE_MAX / 2 || rows > SIZE_MAX / (cols * 2)) return -1;
    ingot_f16_block_to_f32((const unsigned char *)weights, rows * cols, output);
    return 0;
}
