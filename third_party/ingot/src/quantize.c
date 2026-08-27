/* f32 -> ggml block formats.
 *
 * These are the reference quantizers, written from the format definitions and
 * pinned by round-trip tests that measure relative L2 against what the bit
 * width can actually deliver (a packing bug lands orders of magnitude above
 * the floor, so the budget is a real gate and not a rubber stamp).
 *
 * Q4_K lives in kernels.c — it came with the kernels and is measured there.
 *
 * SPDX-License-Identifier: MIT */
#include "ingot/quant.h"
#include "internal.h"

#include <float.h>
#include <math.h>

static void put_f16(unsigned char *p, float v) {
    const uint16_t h = ingot_f32_to_f16(v);
    p[0] = (unsigned char)(h & 0xff);
    p[1] = (unsigned char)(h >> 8);
}

/* Symmetric block scale: the largest magnitude decides, and its SIGN is kept
 * so the whole range maps onto the negative extreme of the grid — that is why
 * `max` is tracked separately from `amax` instead of just using fabsf. */
static void block_extremes(const float *x, int n, float *amax, float *max) {
    *amax = 0.0f;
    *max = 0.0f;
    for (int i = 0; i < n; i++) {
        const float v = fabsf(x[i]);
        if (v > *amax) { *amax = v; *max = x[i]; }
    }
}

static void block_min_max(const float *x, int n, float *min, float *max) {
    *min = FLT_MAX;
    *max = -FLT_MAX;
    for (int i = 0; i < n; i++) {
        if (x[i] < *min) *min = x[i];
        if (x[i] > *max) *max = x[i];
    }
}

static int clampi(int v, int low, int high) {
    return v < low ? low : (v > high ? high : v);
}

/* Q8_0, 34B: d = amax/127, q = round(x/d). */
static void q_q8_0(const float *x, unsigned char *out) {
    float amax = 0.0f;
    for (int i = 0; i < 32; i++) { const float v = fabsf(x[i]); if (v > amax) amax = v; }
    const float d = amax / 127.0f;
    const float id = d != 0.0f ? 1.0f / d : 0.0f;
    put_f16(out, d);
    for (int i = 0; i < 32; i++)
        out[2 + i] = (unsigned char)(signed char)clampi((int)lroundf(x[i] * id), -127, 127);
}

/* Q4_0, 18B: symmetric, 16 levels centred on 8. */
static void q_q4_0(const float *x, unsigned char *out) {
    float amax, max;
    block_extremes(x, 32, &amax, &max);
    const float d = max / -8.0f;
    const float id = d != 0.0f ? 1.0f / d : 0.0f;
    put_f16(out, d);
    for (int i = 0; i < 16; i++) {
        const int lo = clampi((int)(x[i] * id + 8.5f), 0, 15);
        const int hi = clampi((int)(x[i + 16] * id + 8.5f), 0, 15);
        out[2 + i] = (unsigned char)(lo | (hi << 4));
    }
}

/* Q4_1, 20B: affine, so a block that never crosses zero keeps its resolution.
 * x = d*q + m with q in 0..15. */
static void q_q4_1(const float *x, unsigned char *out) {
    float min, max;
    block_min_max(x, 32, &min, &max);
    const float d = (max - min) / 15.0f;
    const float id = d != 0.0f ? 1.0f / d : 0.0f;
    put_f16(out, d);
    put_f16(out + 2, min);
    for (int i = 0; i < 16; i++) {
        const int lo = clampi((int)((x[i] - min) * id + 0.5f), 0, 15);
        const int hi = clampi((int)((x[i + 16] - min) * id + 0.5f), 0, 15);
        out[4 + i] = (unsigned char)(lo | (hi << 4));
    }
}

/* Q5_0, 22B: Q4_0 with a fifth bit lifted into a 32-bit plane. */
static void q_q5_0(const float *x, unsigned char *out) {
    float amax, max;
    block_extremes(x, 32, &amax, &max);
    const float d = max / -16.0f;
    const float id = d != 0.0f ? 1.0f / d : 0.0f;
    put_f16(out, d);
    uint32_t qh = 0;
    for (int i = 0; i < 16; i++) {
        const int lo = clampi((int)(x[i] * id + 16.5f), 0, 31);
        const int hi = clampi((int)(x[i + 16] * id + 16.5f), 0, 31);
        out[6 + i] = (unsigned char)((lo & 0x0f) | ((hi & 0x0f) << 4));
        qh |= (uint32_t)((lo >> 4) & 1u) << i;
        qh |= (uint32_t)((hi >> 4) & 1u) << (i + 16);
    }
    for (int i = 0; i < 4; i++) out[2 + i] = (unsigned char)((qh >> (8 * i)) & 0xff);
}

/* Q5_1, 24B: Q4_1's affine form at 32 levels. */
static void q_q5_1(const float *x, unsigned char *out) {
    float min, max;
    block_min_max(x, 32, &min, &max);
    const float d = (max - min) / 31.0f;
    const float id = d != 0.0f ? 1.0f / d : 0.0f;
    put_f16(out, d);
    put_f16(out + 2, min);
    uint32_t qh = 0;
    for (int i = 0; i < 16; i++) {
        const int lo = clampi((int)((x[i] - min) * id + 0.5f), 0, 31);
        const int hi = clampi((int)((x[i + 16] - min) * id + 0.5f), 0, 31);
        out[8 + i] = (unsigned char)((lo & 0x0f) | ((hi & 0x0f) << 4));
        qh |= (uint32_t)((lo >> 4) & 1u) << i;
        qh |= (uint32_t)((hi >> 4) & 1u) << (i + 16);
    }
    for (int i = 0; i < 4; i++) out[4 + i] = (unsigned char)((qh >> (8 * i)) & 0xff);
}

/* Q6_K, 210B: sixteen sub-blocks of 16 share one f16 super-scale, each with a
 * SIGNED 8-bit scale of its own. Quants are 6 bits biased by 32, split between
 * a low-nibble plane and a 2-bit high plane. */
static void q_q6_k(const float *x, unsigned char *out) {
    float scales[16];
    float max_scale = 0.0f;
    for (int sub = 0; sub < 16; sub++) {
        float amax, max;
        block_extremes(x + sub * 16, 16, &amax, &max);
        scales[sub] = max / -32.0f;
        if (fabsf(scales[sub]) > fabsf(max_scale)) max_scale = scales[sub];
    }
    const float d = max_scale / -128.0f;
    const float id = d != 0.0f ? 1.0f / d : 0.0f;

    signed char sc[16];
    for (int sub = 0; sub < 16; sub++)
        sc[sub] = (signed char)clampi((int)lroundf(scales[sub] * id), -128, 127);

    unsigned char ql[128] = {0}, qh[64] = {0};
    for (int sub = 0; sub < 16; sub++) {
        const float ds = d * (float)sc[sub];
        const float ids = ds != 0.0f ? 1.0f / ds : 0.0f;
        for (int i = 0; i < 16; i++) {
            const int q = clampi((int)lroundf(x[sub * 16 + i] * ids) + 32, 0, 63);
            /* The interleave the decoder expects: two halves of 128, and
             * inside each half the four quarters are (low nibble | high 2
             * bits) at four different bit offsets of the same qh byte. */
            const int idx = sub * 16 + i;
            const int half = idx / 128;
            const int within = idx % 128;
            const int quarter = within / 32;
            const int pos = within % 32;
            ql[half * 64 + (quarter % 2) * 32 + pos] |=
                (unsigned char)((q & 0x0f) << (4 * (quarter / 2)));
            qh[half * 32 + pos] |= (unsigned char)(((q >> 4) & 3) << (2 * quarter));
        }
    }
    memcpy(out, ql, 128);
    memcpy(out + 128, qh, 64);
    memcpy(out + 192, sc, 16);
    put_f16(out + 208, d);
}

/* ── sub-block fitting for the K-quants ─────────────────────────────────────
 * Same discipline as Q4_K in kernels.c: spanning min..max exactly is optimal
 * for coverage and not for squared error, so try a few narrower spans and
 * refit step and offset by least squares against the resulting assignment.
 * Deliberately a second copy rather than a refactor of Q4_K's: sharing it
 * would let a change made for Q2_K move bytes Q4_K already emits. */
static void fit_scale_min(const float *x, int n, int maxq,
                          float *step_out, float *offset_out) {
    float low, high;
    block_min_max(x, n, &low, &high);
    if (low > 0.0f) low = 0.0f;    /* q = 0 must stay representable */
    if (high < 0.0f) high = 0.0f;

    float best_step = 0.0f, best_offset = 0.0f, best_error = -1.0f;
    for (int candidate = 0; candidate <= 5; candidate++) {
        const float shrink = 1.0f - 0.04f * (float)candidate;
        const float centre = 0.5f * (high + low);
        const float half = 0.5f * shrink * (high - low);
        float lo = centre - half, hi = centre + half;
        if (lo > 0.0f) lo = 0.0f;
        if (hi < 0.0f) hi = 0.0f;
        float step = (hi - lo) / (float)maxq;
        if (!(step > 0.0f)) continue;

        for (int round = 0; round < 2; round++) {
            float sum_q = 0.0f, sum_qq = 0.0f, sum_x = 0.0f, sum_xq = 0.0f;
            for (int i = 0; i < n; i++) {
                const float q = (float)clampi((int)lrintf((x[i] - lo) / step), 0, maxq);
                sum_q += q; sum_qq += q * q;
                sum_x += x[i]; sum_xq += x[i] * q;
            }
            const float determinant = (float)n * sum_qq - sum_q * sum_q;
            if (!(fabsf(determinant) > 1e-12f)) break;
            float fitted = ((float)n * sum_xq - sum_q * sum_x) / determinant;
            float base = (sum_x - fitted * sum_q) / (float)n;
            /* The offset is stored unsigned, so a fit wanting a positive base
             * is not expressible: pin it at zero and re-solve the step. */
            if (base > 0.0f) {
                base = 0.0f;
                if (sum_qq > 0.0f) fitted = sum_xq / sum_qq;
            }
            if (!(fitted > 0.0f)) break;
            step = fitted; lo = base;
        }

        float error = 0.0f;
        for (int i = 0; i < n; i++) {
            const int q = clampi((int)lrintf((x[i] - lo) / step), 0, maxq);
            const float diff = step * (float)q + lo - x[i];
            error += diff * diff;
        }
        if (best_error < 0.0f || error < best_error) {
            best_error = error; best_step = step; best_offset = -lo;
        }
    }
    *step_out = best_step;
    *offset_out = best_offset;
}

/* Symmetric fit onto a grid that is NOT symmetric: Q3_K spends its eight
 * levels on -4..3, so a sub-block whose extreme is positive wastes one level.
 * Two least-squares rounds over the clamped assignment, which is what moves
 * the fit once the starting scale is set by the extreme. */
static float fit_signed_scale(const float *x, int n, int qmin, int qmax) {
    float amax, extreme;
    block_extremes(x, n, &amax, &extreme);
    if (!(amax > 0.0f)) return 0.0f;
    float scale = extreme < 0.0f ? extreme / (float)qmin : extreme / (float)qmax;
    if (!(scale > 0.0f)) return 0.0f;
    for (int round = 0; round < 2; round++) {
        float sum_qq = 0.0f, sum_xq = 0.0f;
        for (int i = 0; i < n; i++) {
            const float q = (float)clampi((int)lrintf(x[i] / scale), qmin, qmax);
            sum_qq += q * q; sum_xq += x[i] * q;
        }
        if (!(sum_qq > 0.0f)) break;
        const float fitted = sum_xq / sum_qq;
        if (!(fitted > 0.0f)) break;
        scale = fitted;
    }
    return scale;
}

/* Where sub-block `is` lives inside a Q2_K / Q3_K super-block. The four 2-bit
 * fields of one byte belong to FOUR different sub-blocks, so the walk is by
 * (base, shift, half) — the same order dq_q2_k and dq_q3_k read in. */
static void k2_placement(int is, int *byte_base, int *shift) {
    const int within = is % 8;
    *byte_base = (is < 8 ? 0 : 32) + (within % 2) * 16;
    *shift = (within / 2) * 2;
}

/* Q2_K, 84B: 16B of scale|min nibble pairs + 64B of 2-bit quants + d,dmin(f16).
 * x = d*sc*q - dmin*m over sixteen sub-blocks of 16, q in 0..3. */
static void q_q2_k(const float *x, unsigned char *out) {
    float steps[16], offsets[16];
    float max_step = 0.0f, max_offset = 0.0f;
    for (int j = 0; j < 16; j++) {
        fit_scale_min(x + j * 16, 16, 3, &steps[j], &offsets[j]);
        if (steps[j] > max_step) max_step = steps[j];
        if (offsets[j] > max_offset) max_offset = offsets[j];
    }

    /* Four-bit sub-indices, so the block factors divide by 15. They are stored
     * as f16 and read back as f16, so the indices must be solved against the
     * ROUNDED values or the two ends of the fit disagree by an ulp. */
    memset(out, 0, 84);
    const uint16_t d_half = ingot_f32_to_f16(max_step / 15.0f);
    const uint16_t dmin_half = ingot_f32_to_f16(max_offset / 15.0f);
    put_f16(out + 80, ingot_f16_to_f32(d_half));
    put_f16(out + 82, ingot_f16_to_f32(dmin_half));
    const float d = ingot_f16_to_f32(d_half), dmin = ingot_f16_to_f32(dmin_half);

    unsigned char *q = out + 16;
    for (int is = 0; is < 16; is++) {
        const int sc = d > 0.0f ? clampi((int)lroundf(steps[is] / d), 0, 15) : 0;
        const int mn = dmin > 0.0f ? clampi((int)lroundf(offsets[is] / dmin), 0, 15) : 0;
        out[is] = (unsigned char)(sc | (mn << 4));

        const float step = d * (float)sc, offset = dmin * (float)mn;
        const float inv = step > 0.0f ? 1.0f / step : 0.0f;
        int byte_base, shift;
        k2_placement(is, &byte_base, &shift);
        for (int l = 0; l < 16; l++) {
            const int level = clampi((int)lroundf((x[is * 16 + l] + offset) * inv), 0, 3);
            q[byte_base + l] |= (unsigned char)(level << shift);
        }
    }
}

/* Q3_K, 110B: 32B high-bit mask + 64B of 2-bit quants + 12B of 6-bit scales +
 * d(f16). x = d*sc*level with level in -4..3, the high bit INVERTED (a clear
 * bit subtracts 4). Every fitted step is positive, so the signed 6-bit scale
 * field is used across 0..31 and d stays positive — self-consistent with
 * dq_q3_k, and no resolution is lost because the other sign is never needed. */
static void q_q3_k(const float *x, unsigned char *out) {
    float steps[16];
    float max_step = 0.0f;
    for (int j = 0; j < 16; j++) {
        steps[j] = fit_signed_scale(x + j * 16, 16, -4, 3);
        if (steps[j] > max_step) max_step = steps[j];
    }

    memset(out, 0, 110);
    const uint16_t d_half = ingot_f32_to_f16(max_step / 31.0f);
    put_f16(out + 108, ingot_f16_to_f32(d_half));
    const float d = ingot_f16_to_f32(d_half);

    unsigned char *hmask = out, *q = out + 32, *sc6 = out + 96;
    for (int is = 0; is < 16; is++) {
        const int sc = d > 0.0f ? clampi((int)lroundf(steps[is] / d), 0, 31) : 0;
        /* Stored biased by 32: low nibble in sc6[0..7], high two bits packed
         * four to a byte in sc6[8..11], exactly as dq_q3_k unpacks them. */
        const unsigned u = (unsigned)(sc + 32);
        if (is < 8) sc6[is] |= (unsigned char)(u & 0x0f);
        else        sc6[is - 8] |= (unsigned char)((u & 0x0f) << 4);
        sc6[8 + (is % 4)] |= (unsigned char)((u >> 4) << (2 * (is / 4)));

        const float step = d * (float)sc;
        const float inv = step > 0.0f ? 1.0f / step : 0.0f;
        int byte_base, shift;
        k2_placement(is, &byte_base, &shift);
        const unsigned char m = (unsigned char)(1u << ((is / 8) * 4 + (is % 8) / 2));
        const int mask_base = (is % 2) * 16;
        for (int l = 0; l < 16; l++) {
            const int level = clampi((int)lroundf(x[is * 16 + l] * inv), -4, 3);
            int low2 = level;
            if (level >= 0) hmask[mask_base + l] |= m;
            else            low2 = level + 4;
            q[byte_base + l] |= (unsigned char)(low2 << shift);
        }
    }
}

/* Q5_K, 176B: Q4_K's layout plus a fifth bit per quant in qh[32].
 * x = d*sc*(q | bit<<4) - dmin*m over eight sub-blocks of 32; the qh bit pair
 * advances by two every group of 64. */
static void q_q5_k(const float *x, unsigned char *out) {
    float steps[8], offsets[8];
    float max_step = 0.0f, max_offset = 0.0f;
    for (int j = 0; j < 8; j++) {
        fit_scale_min(x + j * 32, 32, 31, &steps[j], &offsets[j]);
        if (steps[j] > max_step) max_step = steps[j];
        if (offsets[j] > max_offset) max_offset = offsets[j];
    }

    memset(out, 0, 176);
    const uint16_t d_half = ingot_f32_to_f16(max_step / 63.0f);
    const uint16_t dmin_half = ingot_f32_to_f16(max_offset / 63.0f);
    put_f16(out, ingot_f16_to_f32(d_half));
    put_f16(out + 2, ingot_f16_to_f32(dmin_half));
    const float d = ingot_f16_to_f32(d_half), dmin = ingot_f16_to_f32(dmin_half);

    unsigned char scale_bits[8], min_bits[8];
    for (int j = 0; j < 8; j++) {
        scale_bits[j] = (unsigned char)(d > 0.0f ? clampi((int)lroundf(steps[j] / d), 0, 63) : 0);
        min_bits[j] = (unsigned char)(dmin > 0.0f ? clampi((int)lroundf(offsets[j] / dmin), 0, 63) : 0);
    }
    /* Six-bit pairs, packed the way k4_scale_min() unpacks them. */
    unsigned char *scales = out + 4;
    for (int j = 0; j < 4; j++) {
        scales[j] = (unsigned char)(scale_bits[j] | ((scale_bits[j + 4] >> 4) << 6));
        scales[j + 4] = (unsigned char)(min_bits[j] | ((min_bits[j + 4] >> 4) << 6));
        scales[j + 8] = (unsigned char)((scale_bits[j + 4] & 0x0fu) |
                                        ((min_bits[j + 4] & 0x0fu) << 4));
    }

    unsigned char *qh = out + 16, *q = out + 48;
    for (int g = 0; g < 4; g++) {
        const int si = 2 * g, base = 64 * g;
        const float step_lo = d * (float)scale_bits[si], off_lo = dmin * (float)min_bits[si];
        const float step_hi = d * (float)scale_bits[si + 1], off_hi = dmin * (float)min_bits[si + 1];
        const float inv_lo = step_lo > 0.0f ? 1.0f / step_lo : 0.0f;
        const float inv_hi = step_hi > 0.0f ? 1.0f / step_hi : 0.0f;
        for (int i = 0; i < 32; i++) {
            const int lo = clampi((int)lroundf((x[base + i] + off_lo) * inv_lo), 0, 31);
            const int hi = clampi((int)lroundf((x[base + 32 + i] + off_hi) * inv_hi), 0, 31);
            q[32 * g + i] = (unsigned char)((lo & 0x0f) | ((hi & 0x0f) << 4));
            if (lo & 16) qh[i] |= (unsigned char)(1u << (2 * g));
            if (hi & 16) qh[i] |= (unsigned char)(2u << (2 * g));
        }
    }
}

typedef void (*quant_block_fn)(const float *, unsigned char *);

static quant_block_fn quantizer_for(int type) {
    switch (type) {
    case INGOT_TYPE_Q4_0: return q_q4_0;
    case INGOT_TYPE_Q4_1: return q_q4_1;
    case INGOT_TYPE_Q5_0: return q_q5_0;
    case INGOT_TYPE_Q5_1: return q_q5_1;
    case INGOT_TYPE_Q8_0: return q_q8_0;
    case INGOT_TYPE_Q2_K: return q_q2_k;
    case INGOT_TYPE_Q3_K: return q_q3_k;
    case INGOT_TYPE_Q5_K: return q_q5_k;
    case INGOT_TYPE_Q6_K: return q_q6_k;
    default: return NULL;
    }
}

int ingot_can_quantize(int type) {
    return type == INGOT_TYPE_Q4_K || quantizer_for(type) != NULL ||
           type == INGOT_TYPE_F32 || type == INGOT_TYPE_F16 || type == INGOT_TYPE_BF16;
}

int ingot_quantize(int type, const float *values, size_t count, void *out) {
    if (values == NULL || out == NULL) return -1;
    uint64_t blk_elems, blk_bytes;
    if (ingot_type_geometry(type, &blk_elems, &blk_bytes) != 0) return -1;
    if (count % blk_elems != 0) return -1;

    unsigned char *dst = (unsigned char *)out;
    switch (type) {
    case INGOT_TYPE_F32:
        memcpy(dst, values, count * sizeof(float));
        return 0;
    case INGOT_TYPE_F16:
        for (size_t i = 0; i < count; i++) put_f16(dst + 2 * i, values[i]);
        return 0;
    case INGOT_TYPE_BF16:
        ingot_f32_block_to_bf16(values, count, dst);
        return 0;
    case INGOT_TYPE_Q4_K:
        return ingot_q4_k_quantize(values, count, out);
    default: break;
    }
    const quant_block_fn fn = quantizer_for(type);
    if (fn == NULL) return -1;
    const size_t blocks = count / (size_t)blk_elems;
    for (size_t b = 0; b < blocks; b++)
        fn(values + b * (size_t)blk_elems, dst + b * (size_t)blk_bytes);
    return 0;
}
