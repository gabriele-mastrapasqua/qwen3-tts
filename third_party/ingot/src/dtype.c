/* SPDX-License-Identifier: MIT */
#include "ingot/dtype.h"
#include "internal.h"

#include <string.h>

/* ── SIMD for the bulk conversions ──────────────────────────────────────────
 * dtype.c is container-half code, so there is no ingot_cpu() here: runtime
 * dispatch lives in the quant half and core-only links must stay clean. The
 * rule is compile-time instead — a TU built with -mavx2/-mf16c may use them
 * unconditionally, exactly as the compiler itself already does for its own
 * codegen, and on aarch64 NEON is baseline. The SIMD bodies read the file
 * bytes as host-endian words, so they are little-endian-host only; a
 * big-endian build falls back to the byte-by-byte scalar loops. */
#if !defined(__BYTE_ORDER__) || __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
#if (defined(__ARM_NEON) || defined(__aarch64__)) && !defined(INGOT_DISABLE_NEON)
#include <arm_neon.h>
#define INGOT_DTYPE_NEON 1
#endif
#if defined(__AVX2__) && !defined(INGOT_DISABLE_AVX2)
#include <immintrin.h>
#define INGOT_DTYPE_AVX2 1
#if defined(__AVX512F__) && defined(__AVX512BW__)
#define INGOT_DTYPE_AVX512 1
#endif
#if defined(__F16C__)
#define INGOT_DTYPE_F16C 1
#endif
#endif
#endif

/* ── ggml block geometry ────────────────────────────────────────────────────
 * The numbers are the sizeof() of ggml's block_* structs. They are listed here
 * for every type ggml defines, including the ones ingot cannot dequantize:
 * knowing the geometry is what lets a file open and validate its byte ranges,
 * and it is what turns "unknown type 23" into "IQ4_XS, not dequantizable". */
int ingot_type_geometry(int type, uint64_t *block_elems, uint64_t *block_bytes) {
    uint64_t e, b;
    switch (type) {
    case INGOT_TYPE_F32:     e = 1;   b = 4;   break;
    case INGOT_TYPE_F16:     e = 1;   b = 2;   break;
    case INGOT_TYPE_BF16:    e = 1;   b = 2;   break;
    case INGOT_TYPE_F64:     e = 1;   b = 8;   break;
    case INGOT_TYPE_I8:      e = 1;   b = 1;   break;
    case INGOT_TYPE_I16:     e = 1;   b = 2;   break;
    case INGOT_TYPE_I32:     e = 1;   b = 4;   break;
    case INGOT_TYPE_I64:     e = 1;   b = 8;   break;
    /* 32-value blocks */
    case INGOT_TYPE_Q4_0:    e = 32;  b = 18;  break;  /* d + 16B nibbles      */
    case INGOT_TYPE_Q4_1:    e = 32;  b = 20;  break;  /* d,m + 16B nibbles    */
    case INGOT_TYPE_Q5_0:    e = 32;  b = 22;  break;  /* d + qh4 + 16B        */
    case INGOT_TYPE_Q5_1:    e = 32;  b = 24;  break;  /* d,m + qh4 + 16B      */
    case INGOT_TYPE_Q8_0:    e = 32;  b = 34;  break;  /* d + 32 int8          */
    case INGOT_TYPE_Q8_1:    e = 32;  b = 40;  break;  /* d,s f32 + 32 int8    */
    case INGOT_TYPE_IQ4_NL:  e = 32;  b = 18;  break;
    /* 256-value super-blocks */
    case INGOT_TYPE_Q2_K:    e = 256; b = 84;  break;
    case INGOT_TYPE_Q3_K:    e = 256; b = 110; break;
    case INGOT_TYPE_Q4_K:    e = 256; b = 144; break;
    case INGOT_TYPE_Q5_K:    e = 256; b = 176; break;
    case INGOT_TYPE_Q6_K:    e = 256; b = 210; break;
    case INGOT_TYPE_Q8_K:    e = 256; b = 292; break;
    case INGOT_TYPE_IQ2_XXS: e = 256; b = 66;  break;
    case INGOT_TYPE_IQ2_XS:  e = 256; b = 74;  break;
    case INGOT_TYPE_IQ2_S:   e = 256; b = 82;  break;
    case INGOT_TYPE_IQ3_XXS: e = 256; b = 98;  break;
    case INGOT_TYPE_IQ3_S:   e = 256; b = 110; break;
    case INGOT_TYPE_IQ1_S:   e = 256; b = 50;  break;
    case INGOT_TYPE_IQ1_M:   e = 256; b = 56;  break;
    case INGOT_TYPE_IQ4_XS:  e = 256; b = 136; break;
    case INGOT_TYPE_TQ1_0:   e = 256; b = 54;  break;
    case INGOT_TYPE_TQ2_0:   e = 256; b = 66;  break;
    /* Microscaling formats: a shared exponent per small group. */
    case INGOT_TYPE_MXFP4:   e = 32;  b = 17;  break;  /* E8M0 + E2M1 nibbles */
    case INGOT_TYPE_NVFP4:   e = 64;  b = 36;  break;  /* 4x UE4M3 + nibbles  */
    case INGOT_TYPE_Q1_0:    e = 128; b = 18;  break;
    default: return -1;
    }
    if (block_elems) *block_elems = e;
    if (block_bytes) *block_bytes = b;
    return 0;
}

const char *ingot_type_name(int type) {
    switch (type) {
    case INGOT_TYPE_F32:     return "F32";
    case INGOT_TYPE_F16:     return "F16";
    case INGOT_TYPE_BF16:    return "BF16";
    case INGOT_TYPE_F64:     return "F64";
    case INGOT_TYPE_I8:      return "I8";
    case INGOT_TYPE_I16:     return "I16";
    case INGOT_TYPE_I32:     return "I32";
    case INGOT_TYPE_I64:     return "I64";
    case INGOT_TYPE_Q4_0:    return "Q4_0";
    case INGOT_TYPE_Q4_1:    return "Q4_1";
    case INGOT_TYPE_Q5_0:    return "Q5_0";
    case INGOT_TYPE_Q5_1:    return "Q5_1";
    case INGOT_TYPE_Q8_0:    return "Q8_0";
    case INGOT_TYPE_Q8_1:    return "Q8_1";
    case INGOT_TYPE_Q2_K:    return "Q2_K";
    case INGOT_TYPE_Q3_K:    return "Q3_K";
    case INGOT_TYPE_Q4_K:    return "Q4_K";
    case INGOT_TYPE_Q5_K:    return "Q5_K";
    case INGOT_TYPE_Q6_K:    return "Q6_K";
    case INGOT_TYPE_Q8_K:    return "Q8_K";
    case INGOT_TYPE_IQ2_XXS: return "IQ2_XXS";
    case INGOT_TYPE_IQ2_XS:  return "IQ2_XS";
    case INGOT_TYPE_IQ2_S:   return "IQ2_S";
    case INGOT_TYPE_IQ3_XXS: return "IQ3_XXS";
    case INGOT_TYPE_IQ3_S:   return "IQ3_S";
    case INGOT_TYPE_IQ1_S:   return "IQ1_S";
    case INGOT_TYPE_IQ1_M:   return "IQ1_M";
    case INGOT_TYPE_IQ4_NL:  return "IQ4_NL";
    case INGOT_TYPE_IQ4_XS:  return "IQ4_XS";
    case INGOT_TYPE_TQ1_0:   return "TQ1_0";
    case INGOT_TYPE_TQ2_0:   return "TQ2_0";
    case INGOT_TYPE_MXFP4:   return "MXFP4";
    case INGOT_TYPE_NVFP4:   return "NVFP4";
    case INGOT_TYPE_Q1_0:    return "Q1_0";
    default:                 return "UNKNOWN";
    }
}

int ingot_type_is_quantized(int type) {
    uint64_t e, b;
    if (ingot_type_geometry(type, &e, &b) != 0) return 0;
    return e > 1;
}

int ingot_type_can_dequant(int type) {
    switch (type) {
    case INGOT_TYPE_F32:  case INGOT_TYPE_F16:  case INGOT_TYPE_BF16:
    case INGOT_TYPE_F64:
    case INGOT_TYPE_I8:   case INGOT_TYPE_I16:  case INGOT_TYPE_I32:
    case INGOT_TYPE_I64:
    case INGOT_TYPE_Q4_0: case INGOT_TYPE_Q4_1:
    case INGOT_TYPE_Q5_0: case INGOT_TYPE_Q5_1:
    case INGOT_TYPE_Q8_0:
    case INGOT_TYPE_Q2_K: case INGOT_TYPE_Q3_K: case INGOT_TYPE_Q4_K:
    case INGOT_TYPE_Q5_K: case INGOT_TYPE_Q6_K:
    case INGOT_TYPE_Q8_1: case INGOT_TYPE_Q8_K:
    case INGOT_TYPE_IQ4_NL: case INGOT_TYPE_IQ4_XS:
    case INGOT_TYPE_IQ1_S:  case INGOT_TYPE_IQ1_M:
    case INGOT_TYPE_IQ2_XXS: case INGOT_TYPE_IQ2_XS: case INGOT_TYPE_IQ2_S:
    case INGOT_TYPE_IQ3_XXS: case INGOT_TYPE_IQ3_S:
    case INGOT_TYPE_TQ1_0:  case INGOT_TYPE_TQ2_0:
    case INGOT_TYPE_MXFP4:  case INGOT_TYPE_NVFP4:
        return 1;
    default:
        /* Only Q1_0 is left out: llama.cpp's own reference package carries no
         * decoder for it, so there is nothing to check against and it would be
         * guesswork. Its geometry is known, so such a file still opens,
         * accounts its bytes exactly, and fails by name. */
        return 0;
    }
}

int ingot_type_nbytes(int type, uint64_t nelem, uint64_t *out) {
    uint64_t e, b;
    if (out == NULL || ingot_type_geometry(type, &e, &b) != 0) return -1;
    if (nelem % e != 0) return -1;
    uint64_t blocks = nelem / e;
    if (b != 0 && blocks > UINT64_MAX / b) return -1;
    *out = blocks * b;
    return 0;
}

/* ── safetensors dtypes ─────────────────────────────────────────────────── */
size_t ingot_dtype_size(ingot_dtype dtype) {
    switch (dtype) {
    case INGOT_DT_F64: case INGOT_DT_I64: case INGOT_DT_U64: return 8;
    case INGOT_DT_F32: case INGOT_DT_I32: case INGOT_DT_U32: return 4;
    case INGOT_DT_F16: case INGOT_DT_BF16:
    case INGOT_DT_I16: case INGOT_DT_U16: return 2;
    case INGOT_DT_F8_E4M3: case INGOT_DT_F8_E5M2:
    case INGOT_DT_I8: case INGOT_DT_U8: case INGOT_DT_BOOL: return 1;
    default: return 0;
    }
}

const char *ingot_dtype_name(ingot_dtype dtype) {
    switch (dtype) {
    case INGOT_DT_F64:     return "F64";
    case INGOT_DT_F32:     return "F32";
    case INGOT_DT_F16:     return "F16";
    case INGOT_DT_BF16:    return "BF16";
    case INGOT_DT_F8_E4M3: return "F8_E4M3";
    case INGOT_DT_F8_E5M2: return "F8_E5M2";
    case INGOT_DT_I8:      return "I8";
    case INGOT_DT_U8:      return "U8";
    case INGOT_DT_I16:     return "I16";
    case INGOT_DT_U16:     return "U16";
    case INGOT_DT_I32:     return "I32";
    case INGOT_DT_U32:     return "U32";
    case INGOT_DT_I64:     return "I64";
    case INGOT_DT_U64:     return "U64";
    case INGOT_DT_BOOL:    return "BOOL";
    default:               return "UNKNOWN";
    }
}

ingot_dtype ingot_dtype_from_name(const char *name) {
    if (name == NULL) return INGOT_DT_UNKNOWN;
    static const struct { const char *name; ingot_dtype dtype; } table[] = {
        {"F64", INGOT_DT_F64},   {"F32", INGOT_DT_F32},  {"F16", INGOT_DT_F16},
        {"BF16", INGOT_DT_BF16}, {"F8_E4M3", INGOT_DT_F8_E4M3},
        {"F8_E5M2", INGOT_DT_F8_E5M2},
        {"I8", INGOT_DT_I8},     {"U8", INGOT_DT_U8},    {"I16", INGOT_DT_I16},
        {"U16", INGOT_DT_U16},   {"I32", INGOT_DT_I32},  {"U32", INGOT_DT_U32},
        {"I64", INGOT_DT_I64},   {"U64", INGOT_DT_U64},  {"BOOL", INGOT_DT_BOOL},
    };
    for (size_t i = 0; i < sizeof(table) / sizeof(table[0]); i++)
        if (strcmp(name, table[i].name) == 0) return table[i].dtype;
    return INGOT_DT_UNKNOWN;
}

/* ── scalar conversions ─────────────────────────────────────────────────── */
float ingot_f16_to_f32(uint16_t value) {
    uint32_t sign = (uint32_t)(value & 0x8000u) << 16;
    int exponent = (value >> 10) & 0x1f;
    uint32_t frac = value & 0x03ffu;
    uint32_t bits;
    if (exponent == 0) {
        if (frac == 0) {
            bits = sign;                      /* +/- zero */
        } else {                              /* subnormal: normalize it */
            exponent = 1;
            while ((frac & 0x0400u) == 0) { frac <<= 1; exponent--; }
            bits = sign | ((uint32_t)(exponent + 112) << 23) | ((frac & 0x03ffu) << 13);
        }
    } else if (exponent == 31) {
        /* Inf passes through; NaN gets the quiet bit, payload preserved.
         * That is what FCVTL and VCVTPH2PS both do (IEEE: converting a
         * signaling NaN signals and returns it quieted), and the SIMD lanes
         * of ingot_f16_block_to_f32 are exactly those instructions — the
         * scalar tail has to agree with them bit for bit. */
        bits = sign | 0x7f800000u | (frac << 13) | (frac != 0 ? 0x00400000u : 0u);
    } else {
        bits = sign | ((uint32_t)(exponent + 112) << 23) | (frac << 13);
    }
    float out;
    memcpy(&out, &bits, sizeof(out));
    return out;
}

uint16_t ingot_f32_to_f16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    uint32_t sign = (bits >> 16) & 0x8000u;
    int exponent = (int)((bits >> 23) & 0xffu) - 127;
    uint32_t frac = bits & 0x007fffffu;
    if (exponent == 128) {   /* inf / nan: keep a non-zero mantissa non-zero */
        return (uint16_t)(sign | 0x7c00u | (frac != 0 ? 0x0200u : 0u));
    }
    if (exponent > 15) return (uint16_t)(sign | 0x7c00u);          /* overflow */
    if (exponent < -24) return (uint16_t)sign;                     /* underflow */
    if (exponent < -14) {                                          /* subnormal */
        frac |= 0x00800000u;
        int shift = -exponent - 14;
        uint32_t sub = frac >> (13 + shift);
        if ((frac >> (12 + shift)) & 1u) sub++;                    /* round half up */
        return (uint16_t)(sign | sub);
    }
    uint32_t half = (uint32_t)(exponent + 15) << 10 | (frac >> 13);
    if ((frac & 0x1fffu) > 0x1000u ||
        ((frac & 0x1fffu) == 0x1000u && (half & 1u))) half++;      /* nearest-even */
    return (uint16_t)(sign | half);
}

float ingot_bf16_to_f32(uint16_t value) {
    uint32_t bits = (uint32_t)value << 16;
    float out;
    memcpy(&out, &bits, sizeof(out));
    return out;
}

uint16_t ingot_f32_to_bf16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    if (((bits >> 23) & 0xffu) == 0xffu && (bits & 0x007fffffu) != 0)
        return (uint16_t)((bits >> 16) | 0x0040u);          /* keep nan a nan */
    uint32_t rounded = bits + 0x7fffu + ((bits >> 16) & 1u); /* nearest-even */
    return (uint16_t)(rounded >> 16);
}

/* FP8 E4M3 in the OCP "fn" flavour torchao/transformer-engine checkpoints use:
 * no infinities, exponent bias 7, 0x7f/0xff are NaN. */
float ingot_f8_e4m3_to_f32(uint8_t value) {
    uint32_t sign = (uint32_t)(value & 0x80u) << 24;
    int exponent = (value >> 3) & 0x0f;
    uint32_t frac = value & 0x07u;
    uint32_t bits;
    if (exponent == 0) {
        if (frac == 0) {
            bits = sign;
        } else {
            exponent = 1;
            while ((frac & 0x08u) == 0) { frac <<= 1; exponent--; }
            bits = sign | ((uint32_t)(exponent + 120) << 23) | ((frac & 0x07u) << 20);
        }
    } else if (exponent == 15 && frac == 7) {
        bits = sign | 0x7fc00000u;                       /* the single NaN */
    } else {
        bits = sign | ((uint32_t)(exponent + 120) << 23) | (frac << 20);
    }
    float out;
    memcpy(&out, &bits, sizeof(out));
    return out;
}

/* FP8 E5M2: an f16 with the low 8 mantissa bits chopped off, so the exponent
 * bias and the inf/nan encodings are f16's. */
float ingot_f8_e5m2_to_f32(uint8_t value) {
    return ingot_f16_to_f32((uint16_t)((uint16_t)value << 8));
}

/* ── bulk conversions ───────────────────────────────────────────────────────
 * BF16→F32 is a 16-bit shift into the high half of the word; F16→F32 is what
 * FCVTL / VCVTPH2PS do in hardware, IEEE-exact including subnormals. Both are
 * widening, so the SIMD result is bit-identical to the scalar loop — the
 * conversion parity test enforces that, subnormals and NaNs included. */

void ingot_bf16_block_to_f32(const unsigned char *p, size_t nelem, float *dst) {
    size_t i = 0;
#if defined(INGOT_DTYPE_AVX512)
    for (; i + 16 <= nelem; i += 16) {
        const __m256i h = _mm256_loadu_si256((const __m256i *)(p + 2 * i));
        const __m512i w = _mm512_slli_epi32(_mm512_cvtepu16_epi32(h), 16);
        _mm512_storeu_ps(dst + i, _mm512_castsi512_ps(w));
    }
#elif defined(INGOT_DTYPE_AVX2)
    for (; i + 8 <= nelem; i += 8) {
        const __m128i h = _mm_loadu_si128((const __m128i *)(p + 2 * i));
        const __m256i w = _mm256_slli_epi32(_mm256_cvtepu16_epi32(h), 16);
        _mm256_storeu_ps(dst + i, _mm256_castsi256_ps(w));
    }
#elif defined(INGOT_DTYPE_NEON)
    for (; i + 8 <= nelem; i += 8) {
        /* byte load: these converters accept any source alignment, and
         * vld1q_u16 would promise the compiler 2-byte alignment it may not have */
        const uint16x8_t h = vreinterpretq_u16_u8(vld1q_u8(p + 2 * i));
        vst1q_f32(dst + i,     vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(h), 16)));
        vst1q_f32(dst + i + 4, vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(h), 16)));
    }
#endif
    for (; i < nelem; i++)
        dst[i] = ingot_bf16_to_f32(ingot_ld_u16(p + 2 * i));
}

void ingot_f16_block_to_f32(const unsigned char *p, size_t nelem, float *dst) {
    size_t i = 0;
#if defined(INGOT_DTYPE_AVX512)
    for (; i + 16 <= nelem; i += 16)
        _mm512_storeu_ps(dst + i,
            _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(p + 2 * i))));
#elif defined(INGOT_DTYPE_F16C)
    for (; i + 8 <= nelem; i += 8)
        _mm256_storeu_ps(dst + i,
            _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(p + 2 * i))));
#elif defined(INGOT_DTYPE_NEON) && defined(__aarch64__)
    for (; i + 4 <= nelem; i += 4) {
        /* byte load: any source alignment (see the BF16 twin above) */
        const float16x4_t h = vreinterpret_f16_u8(vld1_u8(p + 2 * i));
        vst1q_f32(dst + i, vcvt_f32_f16(h));
    }
#endif
    for (; i < nelem; i++)
        dst[i] = ingot_f16_to_f32(ingot_ld_u16(p + 2 * i));
}

/* F32→BF16 is the write-side twin (quantize/convert tooling). Narrowing, so
 * the vector body re-implements the exact scalar semantics: round to nearest
 * even via bits + 0x7fff + lsb, NaN kept NaN by forcing the quiet bit. The
 * AVX-512BF16 native instruction is deliberately NOT used here: VCVTNEPS2BF16
 * flushes subnormal inputs to zero no matter what MXCSR says, which breaks
 * byte-parity with the scalar path. */
void ingot_f32_block_to_bf16(const float *src, size_t nelem, unsigned char *dst) {
    size_t i = 0;
#if defined(INGOT_DTYPE_AVX2)
    const __m256i c7fff = _mm256_set1_epi32(0x7fff);
    const __m256i one   = _mm256_set1_epi32(1);
    const __m256i expm  = _mm256_set1_epi32(0x7f800000);
    const __m256i quiet = _mm256_set1_epi32(0x0040);
    for (; i + 8 <= nelem; i += 8) {
        const __m256i bits = _mm256_castps_si256(_mm256_loadu_ps(src + i));
        const __m256i lsb  = _mm256_and_si256(_mm256_srli_epi32(bits, 16), one);
        __m256i h = _mm256_srli_epi32(
            _mm256_add_epi32(_mm256_add_epi32(bits, c7fff), lsb), 16);
        /* NaN: exponent all ones and mantissa non-zero -> (bits>>16)|quiet */
        const __m256i isexp = _mm256_cmpeq_epi32(_mm256_and_si256(bits, expm), expm);
        const __m256i isman = _mm256_cmpeq_epi32(
            _mm256_andnot_si256(expm, _mm256_and_si256(bits, _mm256_set1_epi32(0x7fffffff))),
            _mm256_setzero_si256());
        const __m256i isnan = _mm256_andnot_si256(isman, isexp);
        const __m256i nanh  = _mm256_or_si256(_mm256_srli_epi32(bits, 16), quiet);
        h = _mm256_blendv_epi8(h, nanh, isnan);
        /* 32-bit lanes hold 16-bit values: pack and undo the lane interleave */
        const __m256i packed = _mm256_packus_epi32(h, h);
        const __m256i lanes  = _mm256_permute4x64_epi64(packed, 0xd8);
        _mm_storeu_si128((__m128i *)(dst + 2 * i), _mm256_castsi256_si128(lanes));
    }
#elif defined(INGOT_DTYPE_NEON)
    const uint32x4_t c7fff = vdupq_n_u32(0x7fff);
    const uint32x4_t expm  = vdupq_n_u32(0x7f800000);
    const uint32x4_t manm  = vdupq_n_u32(0x007fffff);
    for (; i + 4 <= nelem; i += 4) {
        const uint32x4_t bits = vreinterpretq_u32_f32(vld1q_f32(src + i));
        const uint32x4_t lsb  = vandq_u32(vshrq_n_u32(bits, 16), vdupq_n_u32(1));
        const uint32x4_t rnd  = vaddq_u32(vaddq_u32(bits, c7fff), lsb);
        uint16x4_t h = vshrn_n_u32(rnd, 16);
        const uint32x4_t isnan = vandq_u32(
            vceqq_u32(vandq_u32(bits, expm), expm),
            vcgtq_u32(vandq_u32(bits, manm), vdupq_n_u32(0)));
        const uint16x4_t nanh = vorr_u16(vshrn_n_u32(bits, 16), vdup_n_u16(0x0040));
        h = vbsl_u16(vmovn_u32(isnan), nanh, h);
        /* byte store: dst may sit at any alignment; vst1_u16 would be UB there
         * (UBSan on GB10 caught exactly this on an odd destination) */
        vst1_u8(dst + 2 * i, vreinterpret_u8_u16(h));
    }
#endif
    for (; i < nelem; i++) {
        const uint16_t h = ingot_f32_to_bf16(src[i]);
        dst[2 * i]     = (unsigned char)(h & 0xff);
        dst[2 * i + 1] = (unsigned char)(h >> 8);
    }
}

int ingot_dtype_to_f32(ingot_dtype dtype, const void *src, size_t nelem, float *dst) {
    if (src == NULL || dst == NULL) return -1;
    const unsigned char *p = (const unsigned char *)src;
    switch (dtype) {
    case INGOT_DT_F32:
        memcpy(dst, src, nelem * sizeof(float));
        return 0;
    case INGOT_DT_F64:
        for (size_t i = 0; i < nelem; i++) {
            double v;
            memcpy(&v, p + i * 8, 8);
            dst[i] = (float)v;
        }
        return 0;
    case INGOT_DT_F16:
        ingot_f16_block_to_f32(p, nelem, dst);
        return 0;
    case INGOT_DT_BF16:
        ingot_bf16_block_to_f32(p, nelem, dst);
        return 0;
    case INGOT_DT_F8_E4M3:
        for (size_t i = 0; i < nelem; i++) dst[i] = ingot_f8_e4m3_to_f32(p[i]);
        return 0;
    case INGOT_DT_F8_E5M2:
        for (size_t i = 0; i < nelem; i++) dst[i] = ingot_f8_e5m2_to_f32(p[i]);
        return 0;
    case INGOT_DT_I8:
        for (size_t i = 0; i < nelem; i++) dst[i] = (float)(signed char)p[i];
        return 0;
    case INGOT_DT_U8: case INGOT_DT_BOOL:
        for (size_t i = 0; i < nelem; i++) dst[i] = (float)p[i];
        return 0;
    case INGOT_DT_I16:
        for (size_t i = 0; i < nelem; i++) {
            int16_t v = (int16_t)((uint16_t)p[2 * i] | ((uint16_t)p[2 * i + 1] << 8));
            dst[i] = (float)v;
        }
        return 0;
    case INGOT_DT_U16:
        for (size_t i = 0; i < nelem; i++)
            dst[i] = (float)((uint16_t)p[2 * i] | ((uint16_t)p[2 * i + 1] << 8));
        return 0;
    case INGOT_DT_I32:
        for (size_t i = 0; i < nelem; i++) {
            int32_t v;
            memcpy(&v, p + i * 4, 4);
            dst[i] = (float)v;
        }
        return 0;
    case INGOT_DT_U32:
        for (size_t i = 0; i < nelem; i++) {
            uint32_t v;
            memcpy(&v, p + i * 4, 4);
            dst[i] = (float)v;
        }
        return 0;
    case INGOT_DT_I64:
        for (size_t i = 0; i < nelem; i++) {
            int64_t v;
            memcpy(&v, p + i * 8, 8);
            dst[i] = (float)v;
        }
        return 0;
    case INGOT_DT_U64:
        for (size_t i = 0; i < nelem; i++) {
            uint64_t v;
            memcpy(&v, p + i * 8, 8);
            dst[i] = (float)v;
        }
        return 0;
    default:
        return -1;
    }
}
