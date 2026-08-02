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

typedef void (*quant_block_fn)(const float *, unsigned char *);

static quant_block_fn quantizer_for(int type) {
    switch (type) {
    case INGOT_TYPE_Q4_0: return q_q4_0;
    case INGOT_TYPE_Q4_1: return q_q4_1;
    case INGOT_TYPE_Q5_0: return q_q5_0;
    case INGOT_TYPE_Q5_1: return q_q5_1;
    case INGOT_TYPE_Q8_0: return q_q8_0;
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
