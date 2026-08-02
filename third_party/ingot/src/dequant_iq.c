/* The codebook families: IQ1, IQ2, IQ3, the ternary TQ formats, and the
 * microscaling FP4 pair.
 *
 * These were the last hole in ingot's coverage, and they were held open on
 * purpose. Unlike every other ggml format, these do not compute their values
 * from a formula: each indexes a hand-trained lookup grid of 256 to 2048
 * entries. Typing those out is how you get a decoder that returns plausible
 * numbers and wrong ones, and no round-trip test catches a self-consistent
 * mistake.
 *
 * What unblocked them: src/iq_tables.inc is EXTRACTED from llama.cpp's own
 * `gguf` package by tools/gen_iq_tables.py, and tests/test_oracle.c checks
 * every decoder here against that same package's dequantizer, block by block,
 * on random data that exercises the whole codebook. Derived, then verified.
 *
 * SPDX-License-Identifier: MIT */
#include "ingot/quant.h"
#include "internal.h"

#include <math.h>

#include "iq_tables.inc"

#define QK_K 256

static float f16at(const unsigned char *p) { return ingot_f16_to_f32(ingot_ld_u16(p)); }

/* Sign bit i of a 7-bit sign index: set means negate. */
static float sign_of(unsigned char bits, int i) {
    return ((bits >> i) & 1u) ? -1.0f : 1.0f;
}

/* ── IQ2_XXS, 66B ───────────────────────────────────────────────────────────
 * d(f16) + eight uint32 PAIRS. In each pair the first word holds four grid
 * indices (one per byte), the second holds four 7-bit sign indices packed at
 * 7-bit strides plus a 4-bit scale in its top nibble. One pair covers 32
 * values: four groups of eight. */
static void dq_iq2_xxs(const unsigned char *src, size_t nelem, float *dst) {
    for (size_t b = 0; b < nelem / QK_K; b++) {
        const unsigned char *blk = src + b * 66;
        const float d = f16at(blk);
        const unsigned char *qs = blk + 2;
        float *out = dst + b * QK_K;
        for (int pair = 0; pair < 8; pair++) {
            const unsigned char *a = qs + pair * 8;
            const uint32_t hi = ingot_ld_u32(a + 4);
            const float db = d * (0.5f + (float)(hi >> 28)) * 0.25f;
            for (int g = 0; g < 4; g++) {
                const int index = a[g];
                const unsigned char signs = ingot_iq_ksigns[(hi >> (7 * g)) & 0x7fu];
                for (int i = 0; i < 8; i++)
                    out[pair * 32 + g * 8 + i] =
                        db * (float)ingot_iq2xxs_grid[index * 8 + i] * sign_of(signs, i);
            }
        }
    }
}

/* ── IQ2_XS, 74B ────────────────────────────────────────────────────────────
 * d(f16) + 32 uint16 + eight scale bytes. Each uint16 is a 9-bit grid index
 * and a 7-bit sign index; each scale byte holds two 4-bit scales, and one
 * scale covers two consecutive groups of eight. */
static void dq_iq2_xs(const unsigned char *src, size_t nelem, float *dst) {
    for (size_t b = 0; b < nelem / QK_K; b++) {
        const unsigned char *blk = src + b * 74;
        const float d = f16at(blk);
        const unsigned char *qs = blk + 2;
        const unsigned char *scales = blk + 2 + 64;
        float *out = dst + b * QK_K;
        for (int g = 0; g < 32; g++) {
            const uint16_t q = ingot_ld_u16(qs + 2 * g);
            const int scale = (scales[g / 4] >> (4 * ((g / 2) % 2))) & 0x0f;
            const float db = d * (0.5f + (float)scale) * 0.25f;
            const unsigned char signs = ingot_iq_ksigns[(q >> 9) & 0x7fu];
            const int index = q & 511;
            for (int i = 0; i < 8; i++)
                out[g * 8 + i] = db * (float)ingot_iq2xs_grid[index * 8 + i] * sign_of(signs, i);
        }
    }
}

/* ── IQ2_S, 82B ─────────────────────────────────────────────────────────────
 * d(f16) + qs(32) + signs(32) + qh(8) + scales(8). Here the signs get a plane
 * of their own instead of a codebook index, and the grid index gains two high
 * bits from qh, reaching the 1024-entry grid. */
static void dq_iq2_s(const unsigned char *src, size_t nelem, float *dst) {
    for (size_t b = 0; b < nelem / QK_K; b++) {
        const unsigned char *blk = src + b * 82;
        const float d = f16at(blk);
        const unsigned char *qs = blk + 2;
        const unsigned char *signs = blk + 2 + 32;
        const unsigned char *qh = blk + 2 + 64;
        const unsigned char *scales = blk + 2 + 72;
        float *out = dst + b * QK_K;
        for (int g = 0; g < 32; g++) {
            const int scale = (scales[g / 4] >> (4 * ((g / 2) % 2))) & 0x0f;
            const float db = d * (0.5f + (float)scale) * 0.25f;
            const int high = (qh[g / 4] >> (2 * (g % 4))) & 3;
            const int index = qs[g] | (high << 8);
            for (int i = 0; i < 8; i++)
                out[g * 8 + i] =
                    db * (float)ingot_iq2s_grid[index * 8 + i] * sign_of(signs[g], i);
        }
    }
}

/* ── IQ3_XXS, 98B ───────────────────────────────────────────────────────────
 * d(f16) + qs(64) + eight uint32 of packed signs and scales. The grid is four
 * values wide here, so one qs byte covers four outputs and a sign group of
 * eight spans two qs bytes. */
static void dq_iq3_xxs(const unsigned char *src, size_t nelem, float *dst) {
    for (size_t b = 0; b < nelem / QK_K; b++) {
        const unsigned char *blk = src + b * 98;
        const float d = f16at(blk);
        const unsigned char *qs = blk + 2;
        const unsigned char *scales = blk + 2 + 64;
        float *out = dst + b * QK_K;
        for (int v = 0; v < QK_K; v++) {
            const int word = v / 32;
            const uint32_t s = ingot_ld_u32(scales + word * 4);
            const float db = d * (0.5f + (float)(s >> 28)) * 0.5f;
            const int group = (v % 32) / 8;
            const unsigned char signs = ingot_iq_ksigns[(s >> (7 * group)) & 0x7fu];
            out[v] = db * (float)ingot_iq3xxs_grid[qs[v / 4] * 4 + (v % 4)] *
                     sign_of(signs, v % 8);
        }
    }
}

/* ── IQ3_S, 110B ────────────────────────────────────────────────────────────
 * d(f16) + qs(64) + qh(8) + signs(32) + scales(4). One high bit per qs byte
 * lifts the index into the 512-entry grid; the scale is an odd multiplier
 * (1 + 2*s) rather than the (0.5 + s)/4 form the IQ2 family uses. */
static void dq_iq3_s(const unsigned char *src, size_t nelem, float *dst) {
    for (size_t b = 0; b < nelem / QK_K; b++) {
        const unsigned char *blk = src + b * 110;
        const float d = f16at(blk);
        const unsigned char *qs = blk + 2;
        const unsigned char *qh = blk + 2 + 64;
        const unsigned char *signs = blk + 2 + 72;
        const unsigned char *scales = blk + 2 + 104;
        float *out = dst + b * QK_K;
        for (int v = 0; v < QK_K; v++) {
            const int chunk = v / 32;
            const int scale = (scales[chunk / 2] >> (4 * (chunk % 2))) & 0x0f;
            const float db = d * (float)(1 + 2 * scale);
            const int byte = v / 4;
            const int index = qs[byte] | (((qh[byte / 8] >> (byte % 8)) & 1) << 8);
            out[v] = db * (float)ingot_iq3s_grid[index * 4 + (v % 4)] *
                     sign_of(signs[v / 8], v % 8);
        }
    }
}

/* ── IQ1_S, 50B ─────────────────────────────────────────────────────────────
 * d(f16) + qs(32) + eight uint16. Each uint16 carries four 3-bit index
 * extensions, a 3-bit scale and one sign bit that flips a constant offset —
 * this is the family where the grid is ternary and the offset does the rest. */
static void dq_iq1_s(const unsigned char *src, size_t nelem, float *dst) {
    for (size_t b = 0; b < nelem / QK_K; b++) {
        const unsigned char *blk = src + b * 50;
        const float d = f16at(blk);
        const unsigned char *qs = blk + 2;
        const unsigned char *qh = blk + 2 + 32;
        float *out = dst + b * QK_K;
        for (int word = 0; word < 8; word++) {
            const uint16_t h = ingot_ld_u16(qh + 2 * word);
            const float dl = d * (float)(2 * ((h >> 12) & 7) + 1);
            const float delta = (h & 0x8000u) ? -0.125f : 0.125f;
            for (int sub = 0; sub < 4; sub++) {
                const int index = qs[word * 4 + sub] | (((h >> (3 * sub)) & 7) << 8);
                for (int i = 0; i < 8; i++)
                    out[word * 32 + sub * 8 + i] =
                        dl * ((float)ingot_iq1s_grid[index * 8 + i] + delta);
            }
        }
    }
}

/* ── IQ1_M, 56B ─────────────────────────────────────────────────────────────
 * qs(32) + qh(16) + scales(8). The only format with no f16 scale of its own:
 * `d` is reassembled from the top nibble of each of the four scale words. */
static void dq_iq1_m(const unsigned char *src, size_t nelem, float *dst) {
    for (size_t b = 0; b < nelem / QK_K; b++) {
        const unsigned char *blk = src + b * 56;
        const unsigned char *qs = blk;
        const unsigned char *qh = blk + 32;
        const unsigned char *scales = blk + 48;
        uint16_t bits = 0;
        for (int u = 0; u < 4; u++)
            bits |= (uint16_t)((ingot_ld_u16(scales + 2 * u) & 0xf000u) >> (12 - 4 * u));
        const float d = ingot_f16_to_f32(bits);
        float *out = dst + b * QK_K;
        for (int j = 0; j < 8; j++) {
            for (int s = 0; s < 2; s++) {
                const int m = j * 2 + s;              /* which 3-bit scale   */
                const uint16_t word = ingot_ld_u16(scales + 2 * (m / 4));
                const int scale = (word >> (3 * (m % 4))) & 7;
                const float dl = d * (float)(2 * scale + 1);
                for (int t = 0; t < 2; t++) {
                    const int k = j * 4 + s * 2 + t;  /* qs byte / qh nibble */
                    const int nib = (qh[k / 2] >> (4 * (k % 2))) & 0x0f;
                    const int index = qs[k] | ((nib & 7) << 8);
                    const float delta = (nib & 8) ? -0.125f : 0.125f;
                    for (int i = 0; i < 8; i++)
                        out[k * 8 + i] =
                            dl * ((float)ingot_iq1s_grid[index * 8 + i] + delta);
                }
            }
        }
    }
}

/* ── TQ1_0, 54B ─────────────────────────────────────────────────────────────
 * Ternary, packed in base 3: five values per byte for the first 48 bytes, four
 * per byte for the qh tail. Decoding is a multiply by a power of three that
 * WRAPS in eight bits, then a fixed-point divide — the trick that makes a
 * base-3 digit extractable without a division. */
static void dq_tq1_0(const unsigned char *src, size_t nelem, float *dst) {
    static const unsigned char pow3[5] = { 1, 3, 9, 27, 81 };
    for (size_t b = 0; b < nelem / QK_K; b++) {
        const unsigned char *blk = src + b * 54;
        const unsigned char *qs = blk;
        const unsigned char *qh = blk + 48;
        const float d = f16at(blk + 52);
        float *out = dst + b * QK_K;
        size_t v = 0;
        for (int m = 0; m < 5; m++)
            for (int k = 0; k < 32; k++) {
                const unsigned char q = (unsigned char)(qs[k] * pow3[m]);
                out[v++] = d * (float)((int)(((unsigned)q * 3u) >> 8) - 1);
            }
        for (int m = 0; m < 5; m++)
            for (int k = 0; k < 16; k++) {
                const unsigned char q = (unsigned char)(qs[32 + k] * pow3[m]);
                out[v++] = d * (float)((int)(((unsigned)q * 3u) >> 8) - 1);
            }
        for (int m = 0; m < 4; m++)
            for (int k = 0; k < 4; k++) {
                const unsigned char q = (unsigned char)(qh[k] * pow3[m]);
                out[v++] = d * (float)((int)(((unsigned)q * 3u) >> 8) - 1);
            }
    }
}

/* ── TQ2_0, 66B ─────────────────────────────────────────────────────────────
 * Ternary again, but two bits per value: four values per byte, biased by one. */
static void dq_tq2_0(const unsigned char *src, size_t nelem, float *dst) {
    for (size_t b = 0; b < nelem / QK_K; b++) {
        const unsigned char *blk = src + b * 66;
        const float d = f16at(blk + 64);
        float *out = dst + b * QK_K;
        size_t v = 0;
        for (int half = 0; half < 2; half++)
            for (int m = 0; m < 4; m++)
                for (int k = 0; k < 32; k++)
                    out[v++] = d * (float)((int)((blk[half * 32 + k] >> (2 * m)) & 3) - 1);
    }
}

/* ── MXFP4, 17B for 32 values ───────────────────────────────────────────────
 * The OCP microscaling format gpt-oss ships in: one E8M0 exponent byte and
 * sixteen bytes of E2M1 nibbles read through a 16-entry table. */
static const int8_t MXFP4_VALUES[16] = { 0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12 };

static float e8m0_half(unsigned char e) {
    const uint32_t bits = (e < 2) ? (0x00200000u << e) : ((uint32_t)(e - 1) << 23);
    float out;
    memcpy(&out, &bits, sizeof out);
    return out;
}

static void dq_mxfp4(const unsigned char *src, size_t nelem, float *dst) {
    for (size_t b = 0; b < nelem / 32; b++) {
        const unsigned char *blk = src + b * 17;
        const float d = e8m0_half(blk[0]);
        for (int i = 0; i < 16; i++) {
            dst[b * 32 + i]      = d * (float)MXFP4_VALUES[blk[1 + i] & 0x0f];
            dst[b * 32 + i + 16] = d * (float)MXFP4_VALUES[blk[1 + i] >> 4];
        }
    }
}

/* ── NVFP4, 36B for 64 values ───────────────────────────────────────────────
 * Four sub-blocks of sixteen, each with an unsigned E4M3 scale byte. */
static float ue4m3(unsigned char x) {
    if (x == 0 || x == 0x7f) return 0.0f;
    const int exponent = (x >> 3) & 0x0f;
    const float mantissa = (float)(x & 7);
    const float raw = (exponent == 0)
                          ? mantissa * 0.001953125f          /* 2^-9 */
                          : (1.0f + mantissa / 8.0f) * ldexpf(1.0f, exponent - 7);
    return raw * 0.5f;
}

static void dq_nvfp4(const unsigned char *src, size_t nelem, float *dst) {
    for (size_t b = 0; b < nelem / 64; b++) {
        const unsigned char *blk = src + b * 36;
        for (int sub = 0; sub < 4; sub++) {
            const float d = ue4m3(blk[sub]);
            const unsigned char *qs = blk + 4 + sub * 8;
            float *out = dst + b * 64 + sub * 16;
            for (int i = 0; i < 8; i++) {
                out[i]     = d * (float)MXFP4_VALUES[qs[i] & 0x0f];
                out[i + 8] = d * (float)MXFP4_VALUES[qs[i] >> 4];
            }
        }
    }
}

/* ── IQ4_NL / IQ4_XS ────────────────────────────────────────────────────────
 * The 16 levels (ingot_iq4nl_values, generated) are spaced to match a
 * Gaussian rather than evenly, which is what makes these beat Q4_0 at the
 * same bit width. No sign plane and no big grid, so they are the cheap end
 * of the IQ family. */
/* IQ4_NL, 18B: d(f16) + 16B of nibbles over 32 values, split low-half /
 * high-half exactly like Q4_0. */
static void dq_iq4_nl(const unsigned char *src, size_t nelem, float *dst) {
    for (size_t b = 0; b < nelem / 32; b++) {
        const unsigned char *blk = src + b * 18;
        const float d = f16at(blk);
        const unsigned char *q = blk + 2;
        for (int i = 0; i < 16; i++) {
            dst[b * 32 + i]      = d * (float)ingot_iq4nl_values[q[i] & 0x0f];
            dst[b * 32 + i + 16] = d * (float)ingot_iq4nl_values[q[i] >> 4];
        }
    }
}

/* IQ4_XS, 136B: d(f16) + scales_h(u16) + scales_l[4] + 128B of nibbles.
 * Eight sub-blocks of 32, each with a 6-bit scale split across the two scale
 * planes: four low bits in a nibble of scales_l, two high bits in a bit pair
 * of scales_h, biased by 32. */
static void dq_iq4_xs(const unsigned char *src, size_t nelem, float *dst) {
    for (size_t b = 0; b < nelem / QK_K; b++) {
        const unsigned char *blk = src + b * 136;
        const float d = f16at(blk);
        const uint16_t scales_h = ingot_ld_u16(blk + 2);
        const unsigned char *scales_l = blk + 4;
        const unsigned char *q = blk + 8;
        float *out = dst + b * QK_K;
        for (int sub = 0; sub < 8; sub++) {
            const int lo = (scales_l[sub / 2] >> (4 * (sub % 2))) & 0x0f;
            const int hi = (scales_h >> (2 * sub)) & 3;
            const float dl = d * (float)(((hi << 4) | lo) - 32);
            for (int i = 0; i < 16; i++) {
                out[i]      = dl * (float)ingot_iq4nl_values[q[i] & 0x0f];
                out[i + 16] = dl * (float)ingot_iq4nl_values[q[i] >> 4];
            }
            out += 32;
            q += 16;
        }
    }
}

/* Dispatched from ingot_dequant(); returns -1 for anything not handled here so
 * the caller can fall through to the classic formats. */
int ingot_dequant_codebook(int type, const void *src, size_t nelem, float *dst);
int ingot_dequant_codebook(int type, const void *src, size_t nelem, float *dst) {
    const unsigned char *p = (const unsigned char *)src;
    switch (type) {
    case INGOT_TYPE_IQ2_XXS: dq_iq2_xxs(p, nelem, dst); return 0;
    case INGOT_TYPE_IQ2_XS:  dq_iq2_xs(p, nelem, dst);  return 0;
    case INGOT_TYPE_IQ2_S:   dq_iq2_s(p, nelem, dst);   return 0;
    case INGOT_TYPE_IQ3_XXS: dq_iq3_xxs(p, nelem, dst); return 0;
    case INGOT_TYPE_IQ3_S:   dq_iq3_s(p, nelem, dst);   return 0;
    case INGOT_TYPE_IQ1_S:   dq_iq1_s(p, nelem, dst);   return 0;
    case INGOT_TYPE_IQ1_M:   dq_iq1_m(p, nelem, dst);   return 0;
    case INGOT_TYPE_TQ1_0:   dq_tq1_0(p, nelem, dst);   return 0;
    case INGOT_TYPE_TQ2_0:   dq_tq2_0(p, nelem, dst);   return 0;
    case INGOT_TYPE_MXFP4:   dq_mxfp4(p, nelem, dst);   return 0;
    case INGOT_TYPE_NVFP4:   dq_nvfp4(p, nelem, dst);   return 0;
    case INGOT_TYPE_IQ4_NL:  dq_iq4_nl(p, nelem, dst);  return 0;
    case INGOT_TYPE_IQ4_XS:  dq_iq4_xs(p, nelem, dst);  return 0;
    default: return -1;
    }
}
