/* Scalar, spec-faithful dequantization of the ggml block formats.
 *
 * These decoders are validated against llama.cpp's own
 * `gguf.quants.dequantize` rather than against a second reading of the same
 * spec by the same person — see tests/test_oracle.c.
 *
 * The block layouts are described inline because a wrong nibble here produces
 * plausible-looking garbage, not a crash, and the next person to touch it
 * deserves better than the spec's field names.
 *
 * SPDX-License-Identifier: MIT */
#include "ingot/quant.h"
#include "internal.h"

int ingot_dequant_codebook(int type, const void *src, size_t nelem, float *dst);

#define QK_K 256

/* ── SIMD lanes for the hot decoders ────────────────────────────────────────
 * Compile-time dispatch, same rule as dtype.c: dequant.c is linked by
 * core-only consumers, so it cannot call ingot_cpu(), and a TU built with
 * -mavx2 may assume AVX2 exactly as the compiler does for its own codegen.
 *
 * Bit-parity with the scalar loops is structural, not lucky: every float
 * product below is exact in f32 — an f16 value carries 11 significand bits
 * and the integer factors at most 8 more, so no product ever rounds — which
 * makes the operation order irrelevant. The oracle test holds whichever body
 * is compiled to llama.cpp's reference values, bit for bit. */
#if !defined(__BYTE_ORDER__) || __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
#if (defined(__ARM_NEON) || defined(__aarch64__)) && !defined(INGOT_DISABLE_NEON)
#include <arm_neon.h>
#define INGOT_DEQ_NEON 1
#elif defined(__AVX2__) && !defined(INGOT_DISABLE_AVX2)
#include <immintrin.h>
#define INGOT_DEQ_AVX2 1
#endif
#endif

#if defined(INGOT_DEQ_NEON)
/* 8 bytes -> two f32x4, unsigned and signed flavours. */
static inline void u8x8_f32(uint8x8_t v, float32x4_t *lo, float32x4_t *hi) {
    const uint16x8_t w = vmovl_u8(v);
    *lo = vcvtq_f32_u32(vmovl_u16(vget_low_u16(w)));
    *hi = vcvtq_f32_u32(vmovl_u16(vget_high_u16(w)));
}
static inline void s8x8_f32(int8x8_t v, float32x4_t *lo, float32x4_t *hi) {
    const int16x8_t w = vmovl_s8(v);
    *lo = vcvtq_f32_s32(vmovl_s16(vget_low_s16(w)));
    *hi = vcvtq_f32_s32(vmovl_s16(vget_high_s16(w)));
}
#endif

static float f16(const unsigned char *p) { return ingot_f16_to_f32(ingot_ld_u16(p)); }

/* Q4_K / Q5_K pack eight 6-bit scales and eight 6-bit mins into 12 bytes.
 * ggml calls this get_scale_min_k4; the first four pairs live in the low six
 * bits of scales[0..7], the last four are split across the high two bits of
 * scales[0..3] and the nibbles of scales[8..11]. */
static void k4_scale_min(const unsigned char *scales, int index,
                         unsigned char *scale, unsigned char *min) {
    if (index < 4) {
        *scale = scales[index] & 63u;
        *min   = scales[index + 4] & 63u;
    } else {
        *scale = (unsigned char)((scales[index + 4] & 0x0fu) | ((scales[index - 4] >> 6) << 4));
        *min   = (unsigned char)((scales[index + 4] >> 4)    | ((scales[index] >> 6) << 4));
    }
}

/* ── 32-value blocks ────────────────────────────────────────────────────── */

/* 18B: d(f16) + 16B of nibbles. Element j and j+16 share a byte. */
static void dq_q4_0(const unsigned char *src, size_t nelem, float *dst) {
    for (size_t b = 0; b < nelem / 32; b++) {
        const unsigned char *blk = src + b * 18;
        const float d = f16(blk);
        const unsigned char *q = blk + 2;
        for (int i = 0; i < 16; i++) {
            dst[b * 32 + i]      = d * (float)((int)(q[i] & 0x0f) - 8);
            dst[b * 32 + i + 16] = d * (float)((int)(q[i] >> 4) - 8);
        }
    }
}

/* 20B: d,m(f16) + 16B nibbles. Affine instead of symmetric: x = d*q + m. */
static void dq_q4_1(const unsigned char *src, size_t nelem, float *dst) {
    for (size_t b = 0; b < nelem / 32; b++) {
        const unsigned char *blk = src + b * 20;
        const float d = f16(blk), m = f16(blk + 2);
        const unsigned char *q = blk + 4;
        for (int i = 0; i < 16; i++) {
            dst[b * 32 + i]      = d * (float)(q[i] & 0x0f) + m;
            dst[b * 32 + i + 16] = d * (float)(q[i] >> 4) + m;
        }
    }
}

/* 22B: d(f16) + qh(4B, one bit per element) + 16B nibbles. The fifth bit for
 * element j is bit j of qh, for j+16 it is bit j+16. */
static void dq_q5_0(const unsigned char *src, size_t nelem, float *dst) {
    for (size_t b = 0; b < nelem / 32; b++) {
        const unsigned char *blk = src + b * 22;
        const float d = f16(blk);
        const uint32_t qh = ingot_ld_u32(blk + 2);
        const unsigned char *q = blk + 6;
        for (int i = 0; i < 16; i++) {
            const int lo = (int)(q[i] & 0x0f) | (int)(((qh >> i) & 1u) << 4);
            const int hi = (int)(q[i] >> 4)   | (int)(((qh >> (i + 16)) & 1u) << 4);
            dst[b * 32 + i]      = d * (float)(lo - 16);
            dst[b * 32 + i + 16] = d * (float)(hi - 16);
        }
    }
}

/* 24B: d,m(f16) + qh(4B) + 16B nibbles. Q5_0's bits with Q4_1's affine form. */
static void dq_q5_1(const unsigned char *src, size_t nelem, float *dst) {
    for (size_t b = 0; b < nelem / 32; b++) {
        const unsigned char *blk = src + b * 24;
        const float d = f16(blk), m = f16(blk + 2);
        const uint32_t qh = ingot_ld_u32(blk + 4);
        const unsigned char *q = blk + 8;
        for (int i = 0; i < 16; i++) {
            const int lo = (int)(q[i] & 0x0f) | (int)(((qh >> i) & 1u) << 4);
            const int hi = (int)(q[i] >> 4)   | (int)(((qh >> (i + 16)) & 1u) << 4);
            dst[b * 32 + i]      = d * (float)lo + m;
            dst[b * 32 + i + 16] = d * (float)hi + m;
        }
    }
}

/* 34B: d(f16) + 32 int8. */
static void dq_q8_0(const unsigned char *src, size_t nelem, float *dst) {
    for (size_t b = 0; b < nelem / 32; b++) {
        const unsigned char *blk = src + b * 34;
        const float d = f16(blk);
        const signed char *q = (const signed char *)(blk + 2);
        float *out = dst + b * 32;
#if defined(INGOT_DEQ_NEON)
        for (int i = 0; i < 32; i += 8) {
            float32x4_t lo, hi;
            s8x8_f32(vld1_s8(q + i), &lo, &hi);
            vst1q_f32(out + i,     vmulq_n_f32(lo, d));
            vst1q_f32(out + i + 4, vmulq_n_f32(hi, d));
        }
#elif defined(INGOT_DEQ_AVX2)
        const __m256 dv = _mm256_set1_ps(d);
        for (int i = 0; i < 32; i += 8) {
            const __m128i v = _mm_loadl_epi64((const __m128i *)(q + i));
            const __m256 f = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(v));
            _mm256_storeu_ps(out + i, _mm256_mul_ps(f, dv));
        }
#else
        for (int i = 0; i < 32; i++) out[i] = d * (float)q[i];
#endif
    }
}

/* ── 256-value super-blocks ─────────────────────────────────────────────── */

/* Q2_K, 84B: 16B of packed scale/min nibbles + 64B of 2-bit quants +
 * d,dmin(f16). x = d*sc*q - dmin*m over sixteen sub-blocks of 16. The four
 * 2-bit fields of one byte belong to FOUR different sub-blocks, which is why
 * the walk advances by `shift` (0,2,4,6) rather than by address. */
static void dq_q2_k(const unsigned char *src, size_t nelem, float *dst) {
    for (size_t b = 0; b < nelem / QK_K; b++) {
        const unsigned char *blk = src + b * 84;
        const unsigned char *scales = blk;
        const unsigned char *q = blk + 16;
        const float d = f16(blk + 80), dmin = f16(blk + 82);
        float *out = dst + b * QK_K;
        int is = 0, o = 0;
        for (int base = 0; base < QK_K; base += 128) {
            for (int shift = 0; shift < 8; shift += 2) {
                for (int half = 0; half < 32; half += 16) {
                    const unsigned char sc = scales[is++];
                    const float dl = d * (float)(sc & 0x0f);
                    const float ml = dmin * (float)(sc >> 4);
                    for (int l = 0; l < 16; l++)
                        out[o++] = dl * (float)((q[base / 4 + half + l] >> shift) & 3) - ml;
                }
            }
        }
    }
}

/* Q3_K, 110B: 32B high-bit mask + 64B of 2-bit quants + 12B of 6-bit scales +
 * d(f16). x = d*(sc-32)*(q - 4*(hbit==0)): the high bit is INVERTED, so a
 * clear bit subtracts 4. hmask is one bit per element, so the byte index stays
 * in 0..31 and the walking bit distinguishes the two halves of 128. */
static void dq_q3_k(const unsigned char *src, size_t nelem, float *dst) {
    for (size_t b = 0; b < nelem / QK_K; b++) {
        const unsigned char *blk = src + b * 110;
        const unsigned char *hmask = blk;
        const unsigned char *q = blk + 32;
        const unsigned char *sc6 = blk + 96;
        const float d = f16(blk + 108);
        float *out = dst + b * QK_K;
        signed char sc[16];
        for (int i = 0; i < 16; i++) {
            const int lo = (i < 8) ? (sc6[i] & 0x0f) : (sc6[i - 8] >> 4);
            const int hi = (sc6[8 + (i % 4)] >> (2 * (i / 4))) & 3;
            sc[i] = (signed char)((lo | (hi << 4)) - 32);
        }
        int is = 0, o = 0;
        unsigned char m = 1;
        for (int base = 0; base < QK_K; base += 128) {
            for (int shift = 0; shift < 8; shift += 2) {
                for (int half = 0; half < 32; half += 16) {
                    const float dl = d * (float)sc[is++];
                    for (int l = 0; l < 16; l++) {
                        const int lo2 = (q[base / 4 + half + l] >> shift) & 3;
                        const int sub = (hmask[half + l] & m) ? 0 : 4;
                        out[o++] = dl * (float)(lo2 - sub);
                    }
                }
                m = (unsigned char)(m << 1);
            }
        }
    }
}

/* Q4_K, 144B: d,dmin(f16) + 12B scales/mins + 128B nibbles.
 * x = d*sc*q - dmin*m over eight sub-blocks of 32. */
static void dq_q4_k(const unsigned char *src, size_t nelem, float *dst) {
    for (size_t b = 0; b < nelem / QK_K; b++) {
        const unsigned char *blk = src + b * 144;
        const float d = f16(blk), dmin = f16(blk + 2);
        const unsigned char *scales = blk + 4;
        const unsigned char *q = blk + 16;
        float *out = dst + b * QK_K;
        for (int base = 0, si = 0; base < QK_K; base += 64, si += 2) {
            unsigned char sc0, mn0, sc1, mn1;
            k4_scale_min(scales, si, &sc0, &mn0);
            k4_scale_min(scales, si + 1, &sc1, &mn1);
            const float d0 = d * sc0, m0 = dmin * mn0;
            const float d1 = d * sc1, m1 = dmin * mn1;
#if defined(INGOT_DEQ_NEON)
            const float32x4_t m0v = vdupq_n_f32(m0), m1v = vdupq_n_f32(m1);
            for (int i = 0; i < 32; i += 8) {
                const uint8x8_t byte = vld1_u8(q + i);
                float32x4_t a, c;
                u8x8_f32(vand_u8(byte, vdup_n_u8(0x0f)), &a, &c);
                vst1q_f32(out + base + i,     vsubq_f32(vmulq_n_f32(a, d0), m0v));
                vst1q_f32(out + base + i + 4, vsubq_f32(vmulq_n_f32(c, d0), m0v));
                u8x8_f32(vshr_n_u8(byte, 4), &a, &c);
                vst1q_f32(out + base + i + 32, vsubq_f32(vmulq_n_f32(a, d1), m1v));
                vst1q_f32(out + base + i + 36, vsubq_f32(vmulq_n_f32(c, d1), m1v));
            }
#elif defined(INGOT_DEQ_AVX2)
            const __m256 d0v = _mm256_set1_ps(d0), m0v = _mm256_set1_ps(m0);
            const __m256 d1v = _mm256_set1_ps(d1), m1v = _mm256_set1_ps(m1);
            const __m128i nib = _mm_set1_epi8(0x0f);
            for (int i = 0; i < 32; i += 8) {
                const __m128i byte = _mm_loadl_epi64((const __m128i *)(q + i));
                const __m128i lo = _mm_and_si128(byte, nib);
                const __m128i hi = _mm_and_si128(_mm_srli_epi16(byte, 4), nib);
                const __m256 a = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(lo));
                const __m256 c = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(hi));
                _mm256_storeu_ps(out + base + i,
                                 _mm256_sub_ps(_mm256_mul_ps(a, d0v), m0v));
                _mm256_storeu_ps(out + base + i + 32,
                                 _mm256_sub_ps(_mm256_mul_ps(c, d1v), m1v));
            }
#else
            for (int i = 0; i < 32; i++) {
                out[base + i]      = d0 * (float)(q[i] & 0x0f) - m0;
                out[base + i + 32] = d1 * (float)(q[i] >> 4) - m1;
            }
#endif
            q += 32;
        }
    }
}

/* Q5_K, 176B: Q4_K plus a fifth bit per quant in qh[32].
 * x = d*sc*(q | bit<<4) - dmin*m; the qh bit pair advances by two every group
 * of 64 (1,2 -> 4,8 -> 16,32 -> 64,128). */
static void dq_q5_k(const unsigned char *src, size_t nelem, float *dst) {
    for (size_t b = 0; b < nelem / QK_K; b++) {
        const unsigned char *blk = src + b * 176;
        const float d = f16(blk), dmin = f16(blk + 2);
        const unsigned char *scales = blk + 4;
        const unsigned char *qh = blk + 16;
        const unsigned char *q = blk + 48;
        float *out = dst + b * QK_K;
        unsigned u1 = 1, u2 = 2;
        for (int base = 0, si = 0; base < QK_K; base += 64, si += 2) {
            unsigned char sc0, mn0, sc1, mn1;
            k4_scale_min(scales, si, &sc0, &mn0);
            k4_scale_min(scales, si + 1, &sc1, &mn1);
            const float d0 = d * sc0, m0 = dmin * mn0;
            const float d1 = d * sc1, m1 = dmin * mn1;
#if defined(INGOT_DEQ_NEON)
            const float32x4_t m0v = vdupq_n_f32(m0), m1v = vdupq_n_f32(m1);
            const uint8x8_t u1v = vdup_n_u8((uint8_t)u1);
            const uint8x8_t u2v = vdup_n_u8((uint8_t)u2);
            const uint8x8_t sixteen = vdup_n_u8(16);
            for (int i = 0; i < 32; i += 8) {
                const uint8x8_t byte = vld1_u8(q + i);
                const uint8x8_t hbit = vld1_u8(qh + i);
                const uint8x8_t hi0 = vand_u8(vtst_u8(hbit, u1v), sixteen);
                const uint8x8_t hi1 = vand_u8(vtst_u8(hbit, u2v), sixteen);
                float32x4_t a, c;
                u8x8_f32(vadd_u8(vand_u8(byte, vdup_n_u8(0x0f)), hi0), &a, &c);
                vst1q_f32(out + base + i,     vsubq_f32(vmulq_n_f32(a, d0), m0v));
                vst1q_f32(out + base + i + 4, vsubq_f32(vmulq_n_f32(c, d0), m0v));
                u8x8_f32(vadd_u8(vshr_n_u8(byte, 4), hi1), &a, &c);
                vst1q_f32(out + base + i + 32, vsubq_f32(vmulq_n_f32(a, d1), m1v));
                vst1q_f32(out + base + i + 36, vsubq_f32(vmulq_n_f32(c, d1), m1v));
            }
#elif defined(INGOT_DEQ_AVX2)
            const __m256 d0v = _mm256_set1_ps(d0), m0v = _mm256_set1_ps(m0);
            const __m256 d1v = _mm256_set1_ps(d1), m1v = _mm256_set1_ps(m1);
            const __m128i nib = _mm_set1_epi8(0x0f);
            const __m128i u1v = _mm_set1_epi8((char)u1);
            const __m128i u2v = _mm_set1_epi8((char)u2);
            const __m128i sixteen = _mm_set1_epi8(16);
            const __m128i zero = _mm_setzero_si128();
            for (int i = 0; i < 32; i += 8) {
                const __m128i byte = _mm_loadl_epi64((const __m128i *)(q + i));
                const __m128i hbit = _mm_loadl_epi64((const __m128i *)(qh + i));
                /* cmpeq-with-zero is the inverted test: andnot re-inverts it */
                const __m128i no0 = _mm_cmpeq_epi8(_mm_and_si128(hbit, u1v), zero);
                const __m128i no1 = _mm_cmpeq_epi8(_mm_and_si128(hbit, u2v), zero);
                const __m128i hi0 = _mm_andnot_si128(no0, sixteen);
                const __m128i hi1 = _mm_andnot_si128(no1, sixteen);
                const __m128i lo = _mm_add_epi8(_mm_and_si128(byte, nib), hi0);
                const __m128i hi = _mm_add_epi8(
                    _mm_and_si128(_mm_srli_epi16(byte, 4), nib), hi1);
                const __m256 a = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(lo));
                const __m256 c = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(hi));
                _mm256_storeu_ps(out + base + i,
                                 _mm256_sub_ps(_mm256_mul_ps(a, d0v), m0v));
                _mm256_storeu_ps(out + base + i + 32,
                                 _mm256_sub_ps(_mm256_mul_ps(c, d1v), m1v));
            }
#else
            for (int i = 0; i < 32; i++) {
                const int hi0 = (qh[i] & u1) ? 16 : 0;
                const int hi1 = (qh[i] & u2) ? 16 : 0;
                out[base + i]      = d0 * (float)((q[i] & 0x0f) + hi0) - m0;
                out[base + i + 32] = d1 * (float)((q[i] >> 4) + hi1) - m1;
            }
#endif
            q += 32;
            u1 <<= 2;
            u2 <<= 2;
        }
    }
}

/* Q6_K, 210B: ql[128] low nibbles + qh[64] bit pairs + 16 SIGNED 8-bit scales
 * + d(f16). x = d*sc*(q-32), no mins. Two halves of 128, eight scales each. */
static void dq_q6_k(const unsigned char *src, size_t nelem, float *dst) {
    for (size_t b = 0; b < nelem / QK_K; b++) {
        const unsigned char *blk = src + b * 210;
        const unsigned char *ql = blk;
        const unsigned char *qh = blk + 128;
        const signed char *sc = (const signed char *)(blk + 192);
        const float d = f16(blk + 208);
        float *out = dst + b * QK_K;
        for (int half = 0; half < 2; half++) {
#if defined(INGOT_DEQ_NEON)
            const int8x8_t bias = vdup_n_s8(32);
            const uint8x8_t nib = vdup_n_u8(0x0f), two = vdup_n_u8(3);
            for (int i = 0; i < 32; i += 8) {
                const int is = i / 16;
                const float f1 = d * (float)sc[is],     f2 = d * (float)sc[is + 2];
                const float f3 = d * (float)sc[is + 4], f4 = d * (float)sc[is + 6];
                const uint8x8_t l0 = vld1_u8(ql + i);
                const uint8x8_t l1 = vld1_u8(ql + i + 32);
                const uint8x8_t h  = vld1_u8(qh + i);
                const int8x8_t q1 = vsub_s8(vreinterpret_s8_u8(vorr_u8(
                    vand_u8(l0, nib), vshl_n_u8(vand_u8(h, two), 4))), bias);
                const int8x8_t q2 = vsub_s8(vreinterpret_s8_u8(vorr_u8(
                    vand_u8(l1, nib), vshl_n_u8(vand_u8(vshr_n_u8(h, 2), two), 4))), bias);
                const int8x8_t q3 = vsub_s8(vreinterpret_s8_u8(vorr_u8(
                    vshr_n_u8(l0, 4), vshl_n_u8(vand_u8(vshr_n_u8(h, 4), two), 4))), bias);
                const int8x8_t q4 = vsub_s8(vreinterpret_s8_u8(vorr_u8(
                    vshr_n_u8(l1, 4), vshl_n_u8(vshr_n_u8(h, 6), 4))), bias);
                float32x4_t a, c;
                s8x8_f32(q1, &a, &c);
                vst1q_f32(out + i,      vmulq_n_f32(a, f1));
                vst1q_f32(out + i + 4,  vmulq_n_f32(c, f1));
                s8x8_f32(q2, &a, &c);
                vst1q_f32(out + i + 32, vmulq_n_f32(a, f2));
                vst1q_f32(out + i + 36, vmulq_n_f32(c, f2));
                s8x8_f32(q3, &a, &c);
                vst1q_f32(out + i + 64, vmulq_n_f32(a, f3));
                vst1q_f32(out + i + 68, vmulq_n_f32(c, f3));
                s8x8_f32(q4, &a, &c);
                vst1q_f32(out + i + 96, vmulq_n_f32(a, f4));
                vst1q_f32(out + i + 100, vmulq_n_f32(c, f4));
            }
#elif defined(INGOT_DEQ_AVX2)
            const __m128i bias = _mm_set1_epi8(32);
            const __m128i nib = _mm_set1_epi8(0x0f), two = _mm_set1_epi8(3);
            for (int i = 0; i < 32; i += 8) {
                const int is = i / 16;
                const float f1 = d * (float)sc[is],     f2 = d * (float)sc[is + 2];
                const float f3 = d * (float)sc[is + 4], f4 = d * (float)sc[is + 6];
                const __m128i l0 = _mm_loadl_epi64((const __m128i *)(ql + i));
                const __m128i l1 = _mm_loadl_epi64((const __m128i *)(ql + i + 32));
                const __m128i h  = _mm_loadl_epi64((const __m128i *)(qh + i));
                const __m128i lo0 = _mm_and_si128(l0, nib);
                const __m128i lo1 = _mm_and_si128(l1, nib);
                const __m128i hi0 = _mm_and_si128(_mm_srli_epi16(l0, 4), nib);
                const __m128i hi1 = _mm_and_si128(_mm_srli_epi16(l1, 4), nib);
                const __m128i b1 = _mm_slli_epi16(_mm_and_si128(h, two), 4);
                const __m128i b2 = _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(h, 2), two), 4);
                const __m128i b3 = _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(h, 4), two), 4);
                const __m128i b4 = _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(h, 6), two), 4);
                const __m128i q1 = _mm_sub_epi8(_mm_or_si128(lo0, b1), bias);
                const __m128i q2 = _mm_sub_epi8(_mm_or_si128(lo1, b2), bias);
                const __m128i q3 = _mm_sub_epi8(_mm_or_si128(hi0, b3), bias);
                const __m128i q4 = _mm_sub_epi8(_mm_or_si128(hi1, b4), bias);
                _mm256_storeu_ps(out + i, _mm256_mul_ps(
                    _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q1)), _mm256_set1_ps(f1)));
                _mm256_storeu_ps(out + i + 32, _mm256_mul_ps(
                    _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q2)), _mm256_set1_ps(f2)));
                _mm256_storeu_ps(out + i + 64, _mm256_mul_ps(
                    _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q3)), _mm256_set1_ps(f3)));
                _mm256_storeu_ps(out + i + 96, _mm256_mul_ps(
                    _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q4)), _mm256_set1_ps(f4)));
            }
#else
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
#endif
            out += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
}


/* ── the formats added to complete the set ──────────────────────────────── */

/* Q8_1, 40B: d,s as f32 (not f16 — this is the one block that widened when
 * ggml moved it off ggml_half2) + 32 int8. `s` is the sum of the quants times
 * d, kept so an int8 dot product can fold the other operand's offset into one
 * multiply; it plays no part in decoding a weight.
 *
 * The 36-byte f16 pair was ingot's first guess and it was wrong. Caught by
 * cross-checking every geometry against GGML_QUANT_SIZES, which is now a
 * standing test rather than something done once. */
static void dq_q8_1(const unsigned char *src, size_t nelem, float *dst) {
    for (size_t b = 0; b < nelem / 32; b++) {
        const unsigned char *blk = src + b * 40;
        const uint32_t bits = ingot_ld_u32(blk);
        float d;
        memcpy(&d, &bits, sizeof d);
        const signed char *q = (const signed char *)(blk + 8);
        for (int i = 0; i < 32; i++) dst[b * 32 + i] = d * (float)q[i];
    }
}

/* Q8_K, 292B: d(f32, not f16) + 256 int8 + 16 int16 sub-block sums. The sums
 * are an accelerator for the int8 dot path, not part of the value. */
static void dq_q8_k(const unsigned char *src, size_t nelem, float *dst) {
    for (size_t b = 0; b < nelem / QK_K; b++) {
        const unsigned char *blk = src + b * 292;
        const uint32_t bits = ingot_ld_u32(blk);
        float d;
        memcpy(&d, &bits, sizeof d);
        const signed char *q = (const signed char *)(blk + 4);
        for (int i = 0; i < QK_K; i++) dst[b * QK_K + i] = d * (float)q[i];
    }
}

/* ── entry points ───────────────────────────────────────────────────────── */
int ingot_dequant(int type, const void *src, size_t nelem, float *dst) {
    if (src == NULL || dst == NULL) return -1;
    uint64_t blk_elems, blk_bytes;
    if (ingot_type_geometry(type, &blk_elems, &blk_bytes) != 0) return -1;
    if (nelem % blk_elems != 0) return -1;
    const unsigned char *p = (const unsigned char *)src;

    switch (type) {
    case INGOT_TYPE_F32:  memcpy(dst, src, nelem * sizeof(float)); return 0;
    case INGOT_TYPE_F16:
        ingot_f16_block_to_f32(p, nelem, dst);
        return 0;
    case INGOT_TYPE_BF16:
        ingot_bf16_block_to_f32(p, nelem, dst);
        return 0;
    case INGOT_TYPE_F64:
        for (size_t i = 0; i < nelem; i++) {
            double v;
            memcpy(&v, p + 8 * i, 8);
            dst[i] = (float)v;
        }
        return 0;
    case INGOT_TYPE_I8:
        for (size_t i = 0; i < nelem; i++) dst[i] = (float)(signed char)p[i];
        return 0;
    case INGOT_TYPE_I16:
        for (size_t i = 0; i < nelem; i++) dst[i] = (float)(int16_t)ingot_ld_u16(p + 2 * i);
        return 0;
    case INGOT_TYPE_I32:
        for (size_t i = 0; i < nelem; i++) dst[i] = (float)(int32_t)ingot_ld_u32(p + 4 * i);
        return 0;
    case INGOT_TYPE_I64:
        for (size_t i = 0; i < nelem; i++) dst[i] = (float)(int64_t)ingot_ld_u64(p + 8 * i);
        return 0;
    case INGOT_TYPE_Q4_0: dq_q4_0(p, nelem, dst); return 0;
    case INGOT_TYPE_Q4_1: dq_q4_1(p, nelem, dst); return 0;
    case INGOT_TYPE_Q5_0: dq_q5_0(p, nelem, dst); return 0;
    case INGOT_TYPE_Q5_1: dq_q5_1(p, nelem, dst); return 0;
    case INGOT_TYPE_Q8_0: dq_q8_0(p, nelem, dst); return 0;
    case INGOT_TYPE_Q8_1: dq_q8_1(p, nelem, dst); return 0;
    case INGOT_TYPE_Q8_K: dq_q8_k(p, nelem, dst); return 0;
    case INGOT_TYPE_Q2_K: dq_q2_k(p, nelem, dst); return 0;
    case INGOT_TYPE_Q3_K: dq_q3_k(p, nelem, dst); return 0;
    case INGOT_TYPE_Q4_K: dq_q4_k(p, nelem, dst); return 0;
    case INGOT_TYPE_Q5_K: dq_q5_k(p, nelem, dst); return 0;
    case INGOT_TYPE_Q6_K: dq_q6_k(p, nelem, dst); return 0;
    /* The codebook families live in dequant_iq.c: they carry 200 KiB of
     * extracted lookup tables, worth keeping out of this file. It returns -1
     * for anything it does not handle either, so this stays the one place a
     * caller has to ask. */
    default: return ingot_dequant_codebook(type, src, nelem, dst);
    }
}

int ingot_gguf_dequant(const ingot_gguf *g, const ingot_tensor *t, float *dst) {
    const void *src = ingot_gguf_data(g, t);
    if (src == NULL || dst == NULL || t->nelem > SIZE_MAX) return -1;
    return ingot_dequant(t->type, src, (size_t)t->nelem, dst);
}

#ifdef INGOT_NO_KERNELS
/* The per-format matrix wrappers normally live in kernels.c, which has a NEON
 * decode for Q4_K. In a build without the kernels they fall back to these
 * scalar ones, so the API surface does not change with the build flags. */
static int matrix_dequant(int type, const void *w, size_t rows, size_t cols, float *out) {
    uint64_t blk_elems, blk_bytes;
    if (w == NULL || out == NULL || rows == 0 || cols == 0 ||
        ingot_type_geometry(type, &blk_elems, &blk_bytes) != 0 ||
        cols % blk_elems != 0 || rows > SIZE_MAX / cols) return -1;
    return ingot_dequant(type, w, rows * cols, out);
}

int ingot_q4_k_dequant(const void *w, size_t rows, size_t cols, float *out) {
    return matrix_dequant(INGOT_TYPE_Q4_K, w, rows, cols, out);
}
int ingot_q5_k_dequant(const void *w, size_t rows, size_t cols, float *out) {
    return matrix_dequant(INGOT_TYPE_Q5_K, w, rows, cols, out);
}
int ingot_q6_k_dequant(const void *w, size_t rows, size_t cols, float *out) {
    return matrix_dequant(INGOT_TYPE_Q6_K, w, rows, cols, out);
}
int ingot_q3_k_dequant(const void *w, size_t rows, size_t cols, float *out) {
    return matrix_dequant(INGOT_TYPE_Q3_K, w, rows, cols, out);
}
int ingot_q2_k_dequant(const void *w, size_t rows, size_t cols, float *out) {
    return matrix_dequant(INGOT_TYPE_Q2_K, w, rows, cols, out);
}
int ingot_q8_0_dequant(const void *w, size_t rows, size_t cols, float *out) {
    return matrix_dequant(INGOT_TYPE_Q8_0, w, rows, cols, out);
}
#endif /* INGOT_NO_KERNELS */
