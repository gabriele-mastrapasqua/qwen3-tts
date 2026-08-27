/* ingot — GGUF + safetensors reader/writer in C11.
 * https://github.com/mynah-org/ingot   SPDX-License-Identifier: MIT
 *
 * GENERATED FILE — do not edit. Regenerate with tools/amalgamate.py.
 * The source of truth is src/ and include/ingot/.
 *
 * Drop ingot.h and ingot.c into your project and build the .c like any
 * other source file. Needs -lpthread -lm. Define INGOT_NO_KERNELS to
 * leave out dequantization and the SIMD kernels.
 */

/* Feature-test macros, hoisted: they only take effect before the
 * first system header, which the concatenation below would break. */
#define _POSIX_C_SOURCE 200809L
#if defined(__APPLE__)
#define _DARWIN_C_SOURCE 1
#endif
#define _POSIX_C_SOURCE 200809L   /* ftello/off_t under -std=c11 on glibc */

#include "ingot.h"

/* ═══ src/internal.h ═══ */
/* Shared internals. Not installed, not part of the public API.
 * SPDX-License-Identifier: MIT */

#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ── errors ─────────────────────────────────────────────────────────────── */
static inline void ingot_err(char *err, size_t errsz, const char *fmt, ...) {
    if (err == NULL || errsz == 0) return;
    va_list args;
    va_start(args, fmt);
    vsnprintf(err, errsz, fmt, args);
    va_end(args);
}

/* ── overflow-checked arithmetic ────────────────────────────────────────── */
static inline int ingot_add_u64(uint64_t a, uint64_t b, uint64_t *out) {
    if (b > UINT64_MAX - a) return -1;
    *out = a + b;
    return 0;
}
static inline int ingot_mul_u64(uint64_t a, uint64_t b, uint64_t *out) {
    if (a != 0 && b > UINT64_MAX / a) return -1;
    *out = a * b;
    return 0;
}

/* ── bulk dtype conversions (dtype.c) ───────────────────────────────────────
 * The vectorized bodies behind ingot_dtype_to_f32 and the F16/BF16 cases of
 * ingot_dequant. `p` is the little-endian file bytes, not necessarily aligned.
 * Byte-for-byte identical to the per-element scalar functions — widening
 * float conversions are exact, and the tests hold them to that. */
void ingot_bf16_block_to_f32(const unsigned char *p, size_t nelem, float *dst);
void ingot_f16_block_to_f32(const unsigned char *p, size_t nelem, float *dst);
void ingot_f32_block_to_bf16(const float *src, size_t nelem, unsigned char *dst);

/* ── little-endian loads ────────────────────────────────────────────────────
 * Explicit, byte by byte: both container formats are defined little-endian and
 * a memcpy would silently do the wrong thing on a big-endian host. */
static inline uint16_t ingot_ld_u16(const unsigned char *p) {
    return (uint16_t)((uint16_t)p[0] | ((uint16_t)p[1] << 8));
}
static inline uint32_t ingot_ld_u32(const unsigned char *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}
static inline uint64_t ingot_ld_u64(const unsigned char *p) {
    uint64_t v = 0;
    for (int i = 7; i >= 0; i--) v = (v << 8) | p[i];
    return v;
}

/* ── name index ─────────────────────────────────────────────────────────────
 * FNV-1a plus open addressing over a power-of-two table, slots holding
 * index+1 so that 0 means empty. Turns tensor lookup from the O(n) strcmp
 * walk every source project shipped into O(1) — it matters: a 400-tensor
 * model looked up 400 times is 160k strcmp for nothing. */
typedef struct {
    size_t   *slots;
    uint64_t *hashes;    /* parallel to the caller's item array */
    size_t    capacity;  /* power of two, 0 when unbuilt */
} ingot_index;

static inline uint64_t ingot_hash(const char *name, size_t len) {
    uint64_t h = UINT64_C(1469598103934665603);
    for (size_t i = 0; i < len; i++) {
        h ^= (unsigned char)name[i];
        h *= UINT64_C(1099511628211);
    }
    return h;
}
static inline uint64_t ingot_hash_str(const char *name) {
    return ingot_hash(name, strlen(name));
}

/* Build an index over `count` items; `name_of(items, i)` yields each name. */
typedef const char *(*ingot_name_fn)(const void *items, size_t i);

static inline int ingot_index_build(ingot_index *ix, const void *items,
                                    size_t count, ingot_name_fn name_of) {
    memset(ix, 0, sizeof(*ix));
    if (count == 0) return 0;
    if (count > SIZE_MAX / 4) return -1;
    size_t capacity = 16;
    while (capacity < count * 2) capacity *= 2;
    ix->slots  = (size_t *)calloc(capacity, sizeof(*ix->slots));
    ix->hashes = (uint64_t *)calloc(count, sizeof(*ix->hashes));
    if (ix->slots == NULL || ix->hashes == NULL) {
        free(ix->slots); free(ix->hashes);
        memset(ix, 0, sizeof(*ix));
        return -1;
    }
    ix->capacity = capacity;
    const size_t mask = capacity - 1;
    for (size_t i = 0; i < count; i++) {
        ix->hashes[i] = ingot_hash_str(name_of(items, i));
        size_t slot = (size_t)ix->hashes[i] & mask;
        while (ix->slots[slot] != 0) slot = (slot + 1) & mask;
        ix->slots[slot] = i + 1;
    }
    return 0;
}

/* Returns the item index, or (size_t)-1 when absent. */
static inline size_t ingot_index_find(const ingot_index *ix, const void *items,
                                      ingot_name_fn name_of, const char *name) {
    if (ix->capacity == 0 || name == NULL) return (size_t)-1;
    const uint64_t h = ingot_hash_str(name);
    const size_t mask = ix->capacity - 1;
    size_t slot = (size_t)h & mask;
    for (;;) {
        const size_t stored = ix->slots[slot];
        if (stored == 0) return (size_t)-1;
        const size_t i = stored - 1;
        if (ix->hashes[i] == h && strcmp(name_of(items, i), name) == 0) return i;
        slot = (slot + 1) & mask;
    }
}

static inline void ingot_index_free(ingot_index *ix) {
    free(ix->slots);
    free(ix->hashes);
    memset(ix, 0, sizeof(*ix));
}

/* ── misc ───────────────────────────────────────────────────────────────── */
static inline char *ingot_strdup(const char *s) {
    size_t n = strlen(s) + 1;
    char *copy = (char *)malloc(n);
    if (copy != NULL) memcpy(copy, s, n);
    return copy;
}

static inline char *ingot_strndup(const char *s, size_t n) {
    char *copy = (char *)malloc(n + 1);
    if (copy != NULL) { memcpy(copy, s, n); copy[n] = '\0'; }
    return copy;
}


/* ═══ src/dtype.c ═══ */
/* SPDX-License-Identifier: MIT */

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

/* ═══ src/gguf.c ═══ */
/* GGUF v2/v3 reader.
 *
 * Two layers of defence, because a model file is untrusted input like any
 * other: the hardening (overflow-checked offsets, explicit little-endian
 * loads, bounded metadata skipping, post-mmap revalidation of every payload
 * range), and the geometric invariants — Q8_0 sizing, `offset % alignment`,
 * "a block never straddles a row" — that catch a file which is structurally
 * well-formed but describes something impossible.
 *
 * SPDX-License-Identifier: MIT */


#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#define GGUF_MAGIC 0x46554747u          /* "GGUF" little-endian */
#define GGUF_MAX_STRING (16u << 20)     /* 16 MiB: a chat template can be big */
#define GGUF_MAX_COUNT  (1u << 28)      /* tensors / KVs / array elements      */

typedef struct {
    char          *path;
    int            fd;
    uint64_t       size;
    unsigned char *map;
    uint64_t       data_base;
    uint64_t       alignment;
    uint32_t       version;
} shard_t;

struct ingot_kv {
    char                *key;
    int                  type;
    int                  arr_type;      /* -1 unless type == INGOT_KV_ARRAY */
    uint64_t             arr_len;
    const unsigned char *payload;       /* into shard 0's mapping */
    uint64_t             payload_bytes;
    char                *strval;        /* NUL-terminated copy, scalar strings */
    uint64_t            *str_off;       /* per-element offsets, string arrays  */
    uint64_t            *str_len;       /* parse-time-validated lengths        */
};

struct ingot_gguf {
    shard_t      *shards;
    uint32_t      nshard;
    ingot_tensor *tensors;
    size_t        ntensor;
    ingot_index   tensor_index;
    ingot_kv     *kvs;
    size_t        nkv;
    ingot_index   kv_index;
    char        **names;                /* owned tensor names */
};

/* ── cursor over a mapped shard ─────────────────────────────────────────── */
typedef struct { const unsigned char *base; uint64_t size, pos; } cursor;

static int cur_take(cursor *c, uint64_t n, const unsigned char **out) {
    uint64_t end;
    if (ingot_add_u64(c->pos, n, &end) != 0 || end > c->size) return -1;
    *out = c->base + c->pos;
    c->pos = end;
    return 0;
}
static int cur_u32(cursor *c, uint32_t *v) {
    const unsigned char *p;
    if (cur_take(c, 4, &p) != 0) return -1;
    *v = ingot_ld_u32(p);
    return 0;
}
static int cur_u64(cursor *c, uint64_t *v) {
    const unsigned char *p;
    if (cur_take(c, 8, &p) != 0) return -1;
    *v = ingot_ld_u64(p);
    return 0;
}
/* A GGUF string is a u64 length followed by raw bytes. Returns a view, not a
 * copy: the mapping outlives every view we hand out. */
static int cur_str(cursor *c, const char **s, uint64_t *len) {
    uint64_t n;
    const unsigned char *p;
    if (cur_u64(c, &n) != 0 || n > GGUF_MAX_STRING) return -1;
    if (cur_take(c, n, &p) != 0) return -1;
    *s = (const char *)p;
    *len = n;
    return 0;
}

static int kv_scalar_bytes(int type, uint64_t *out) {
    switch (type) {
    case INGOT_KV_UINT8: case INGOT_KV_INT8: case INGOT_KV_BOOL: *out = 1; return 0;
    case INGOT_KV_UINT16: case INGOT_KV_INT16: *out = 2; return 0;
    case INGOT_KV_UINT32: case INGOT_KV_INT32: case INGOT_KV_FLOAT32: *out = 4; return 0;
    case INGOT_KV_UINT64: case INGOT_KV_INT64: case INGOT_KV_FLOAT64: *out = 8; return 0;
    default: return -1;
    }
}

static uint64_t align_up(uint64_t value, uint64_t alignment, int *ok) {
    /* alignment must be a sane power of two: the spec default is 32, and a
     * hostile file that declares 2^63 must not wrap the addition below. */
    if (alignment == 0 || alignment > (1u << 20) ||
        (alignment & (alignment - 1)) != 0 || value > UINT64_MAX - alignment + 1) {
        *ok = 0;
        return 0;
    }
    *ok = 1;
    return (value + alignment - 1) & ~(alignment - 1);
}

/* ── metadata ───────────────────────────────────────────────────────────── */
/* A hostile float KV (NaN, inf, 1e30) must not hit the out-of-range
 * float->integer conversion, which is UB and ISA-dependent: saturate,
 * NaN goes to 0. */
static int64_t f64_to_i64_sat(double v) {
    if (v != v) return 0;
    if (v >= 9223372036854775807.0) return INT64_MAX;
    if (v <= -9223372036854775808.0) return INT64_MIN;
    return (int64_t)v;
}

static uint64_t f64_to_u64_sat(double v) {
    if (v != v || v <= 0.0) return 0;
    if (v >= 18446744073709551615.0) return UINT64_MAX;
    return (uint64_t)v;
}

static void kv_free_one(ingot_kv *kv) {
    free(kv->key);
    free(kv->strval);
    free(kv->str_off);
    free(kv->str_len);
}

static int parse_kv(ingot_gguf *g, cursor *c, uint64_t count,
                    char *err, size_t errsz) {
    if (count > GGUF_MAX_COUNT) {
        ingot_err(err, errsz, "GGUF declares %" PRIu64 " metadata entries", count);
        return -1;
    }
    g->kvs = (ingot_kv *)calloc(count ? (size_t)count : 1, sizeof(*g->kvs));
    if (g->kvs == NULL) { ingot_err(err, errsz, "out of memory"); return -1; }

    for (uint64_t i = 0; i < count; i++) {
        ingot_kv *kv = &g->kvs[i];
        const char *key;
        uint64_t keylen;
        uint32_t type;
        if (cur_str(c, &key, &keylen) != 0 || cur_u32(c, &type) != 0) {
            ingot_err(err, errsz, "GGUF metadata truncated at entry %" PRIu64, i);
            return -1;
        }
        kv->key = ingot_strndup(key, (size_t)keylen);
        if (kv->key == NULL) { ingot_err(err, errsz, "out of memory"); return -1; }
        g->nkv = (size_t)i + 1;          /* freeable from here on */
        kv->type = (int)type;
        kv->arr_type = -1;

        if (type == INGOT_KV_STRING) {
            const char *s;
            uint64_t n;
            if (cur_str(c, &s, &n) != 0) {
                ingot_err(err, errsz, "GGUF string value truncated for '%s'", kv->key);
                return -1;
            }
            kv->payload = (const unsigned char *)s;
            kv->payload_bytes = n;
            kv->strval = ingot_strndup(s, (size_t)n);
            if (kv->strval == NULL) { ingot_err(err, errsz, "out of memory"); return -1; }
        } else if (type == INGOT_KV_ARRAY) {
            uint32_t elem;
            uint64_t n;
            if (cur_u32(c, &elem) != 0 || cur_u64(c, &n) != 0 || n > GGUF_MAX_COUNT) {
                ingot_err(err, errsz, "GGUF array header invalid for '%s'", kv->key);
                return -1;
            }
            kv->arr_type = (int)elem;
            kv->arr_len = n;
            if (elem == INGOT_KV_ARRAY) {
                ingot_err(err, errsz, "GGUF nested arrays are not allowed ('%s')", kv->key);
                return -1;
            }
            if (elem == INGOT_KV_STRING) {
                /* We have to walk the array to find where the next KV starts,
                 * so recording each element's offset while walking is free —
                 * and it is what makes a 150k-token vocabulary indexable in
                 * O(1) instead of O(n) per lookup. */
                if (n != 0) {
                    kv->str_off = (uint64_t *)calloc((size_t)n, sizeof(*kv->str_off));
                    kv->str_len = (uint64_t *)calloc((size_t)n, sizeof(*kv->str_len));
                    if (kv->str_off == NULL || kv->str_len == NULL) {
                        ingot_err(err, errsz, "out of memory");
                        return -1;
                    }
                }
                for (uint64_t k = 0; k < n; k++) {
                    const char *s;
                    uint64_t len;
                    kv->str_off[k] = c->pos;
                    if (cur_str(c, &s, &len) != 0) {
                        ingot_err(err, errsz,
                                  "GGUF string array '%s' truncated at %" PRIu64, kv->key, k);
                        return -1;
                    }
                    kv->str_len[k] = len;
                }
                kv->payload = c->base;    /* offsets are absolute in the mapping */
            } else {
                uint64_t esz, total;
                const unsigned char *p;
                if (kv_scalar_bytes((int)elem, &esz) != 0) {
                    ingot_err(err, errsz, "GGUF array '%s' has element type %u", kv->key, elem);
                    return -1;
                }
                if (ingot_mul_u64(n, esz, &total) != 0 || cur_take(c, total, &p) != 0) {
                    ingot_err(err, errsz, "GGUF array '%s' runs past the file", kv->key);
                    return -1;
                }
                kv->payload = p;
                kv->payload_bytes = total;
            }
        } else {
            uint64_t esz;
            const unsigned char *p;
            if (kv_scalar_bytes((int)type, &esz) != 0) {
                ingot_err(err, errsz, "GGUF metadata '%s' has type %u", kv->key, type);
                return -1;
            }
            if (cur_take(c, esz, &p) != 0) {
                ingot_err(err, errsz, "GGUF metadata '%s' truncated", kv->key);
                return -1;
            }
            kv->payload = p;
            kv->payload_bytes = esz;
        }
    }
    g->nkv = (size_t)count;
    return 0;
}

static const char *kv_name_of(const void *items, size_t i) {
    return ((const ingot_kv *)items)[i].key;
}
static const char *tensor_name_of(const void *items, size_t i) {
    return ((const ingot_tensor *)items)[i].name;
}

/* ── one shard ──────────────────────────────────────────────────────────── */
static void gguf_shard_close(shard_t *s) {
    if (s->map != NULL && s->map != MAP_FAILED) munmap(s->map, (size_t)s->size);
    if (s->fd >= 0) close(s->fd);
    free(s->path);
    memset(s, 0, sizeof(*s));
    s->fd = -1;
}

static int shard_map(shard_t *s, const char *path, char *err, size_t errsz) {
    memset(s, 0, sizeof(*s));
    s->fd = open(path, O_RDONLY);
    if (s->fd < 0) {
        ingot_err(err, errsz, "cannot open '%s': %s", path, strerror(errno));
        return -1;
    }
    struct stat st;
    if (fstat(s->fd, &st) != 0 || st.st_size < 24) {
        ingot_err(err, errsz, "'%s' is not a GGUF file (too small)", path);
        gguf_shard_close(s);
        return -1;
    }
    s->size = (uint64_t)st.st_size;
    /* On a 32-bit size_t the cast below would map size mod 2^32 bytes while
     * every bounds check trusts the full u64: refuse instead. The /2 also
     * keeps nelem*sizeof(float) products from wrapping downstream. */
    if (s->size > SIZE_MAX / 2) {
        ingot_err(err, errsz, "'%s' is too large for this address space", path);
        gguf_shard_close(s);
        return -1;
    }
    s->path = ingot_strdup(path);
    s->map = (unsigned char *)mmap(NULL, (size_t)s->size, PROT_READ, MAP_PRIVATE, s->fd, 0);
    if (s->path == NULL || s->map == MAP_FAILED) {
        ingot_err(err, errsz, "cannot mmap '%s': %s", path, strerror(errno));
        gguf_shard_close(s);
        return -1;
    }
    return 0;
}

/* Parse one shard's header. `first` also fills in the metadata KVs; later
 * shards only contribute tensors (llama.cpp repeats the KVs in every shard,
 * and we have no use for the copies). Appends to g->tensors. */
static int shard_parse(ingot_gguf *g, uint32_t index, int first,
                       char *err, size_t errsz) {
    shard_t *s = &g->shards[index];
    cursor c = { s->map, s->size, 0 };
    uint32_t magic, version;
    uint64_t ntensor, nkv;

    if (cur_u32(&c, &magic) != 0 || magic != GGUF_MAGIC) {
        ingot_err(err, errsz, "'%s' is not a GGUF file (bad magic)", s->path);
        return -1;
    }
    /* Split, because the message reads `version`: folding the two together
     * printed a variable the failed read had never written. */
    if (cur_u32(&c, &version) != 0) {
        ingot_err(err, errsz, "'%s' ends inside the GGUF header", s->path);
        return -1;
    }
    if (version < 2 || version > 3) {
        ingot_err(err, errsz, "unsupported GGUF version %u in '%s' (need 2 or 3)",
                  version, s->path);
        return -1;
    }
    s->version = version;
    s->alignment = 32;                   /* the spec default */
    if (cur_u64(&c, &ntensor) != 0 || cur_u64(&c, &nkv) != 0 ||
        ntensor > GGUF_MAX_COUNT) {
        ingot_err(err, errsz, "GGUF header invalid in '%s'", s->path);
        return -1;
    }

    if (first) {
        if (parse_kv(g, &c, nkv, err, errsz) != 0) return -1;
        /* Linear scan on purpose: the name index does not exist yet (it is
         * built once every shard has been parsed), and there are a few dozen
         * KVs, not a few thousand. */
        for (size_t i = 0; i < g->nkv; i++) {
            uint64_t a;
            if (strcmp(g->kvs[i].key, "general.alignment") == 0 &&
                ingot_kv_u64(&g->kvs[i], &a) == 0) s->alignment = a;
        }
    } else {
        /* Later shards: skip the KV block without recording it, but with the
         * same bounds checking — a corrupt tail shard must not be trusted
         * just because shard 0 was fine. */
        ingot_gguf skip;
        memset(&skip, 0, sizeof(skip));
        if (parse_kv(&skip, &c, nkv, err, errsz) != 0) {
            for (size_t i = 0; i < skip.nkv; i++) kv_free_one(&skip.kvs[i]);
            free(skip.kvs);
            return -1;
        }
        for (size_t i = 0; i < skip.nkv; i++) {
            if (strcmp(skip.kvs[i].key, "general.alignment") == 0) {
                uint64_t a;
                if (ingot_kv_u64(&skip.kvs[i], &a) == 0) s->alignment = a;
            }
            kv_free_one(&skip.kvs[i]);
        }
        free(skip.kvs);
        s->alignment = s->alignment ? s->alignment : 32;
    }

    const size_t base = g->ntensor;
    if (ntensor > SIZE_MAX / sizeof(ingot_tensor) - base) {
        ingot_err(err, errsz, "GGUF tensor count overflows");
        return -1;
    }
    ingot_tensor *grown = (ingot_tensor *)realloc(
        g->tensors, (base + (size_t)ntensor + 1) * sizeof(*grown));
    if (grown == NULL) { ingot_err(err, errsz, "out of memory"); return -1; }
    g->tensors = grown;
    char **names = (char **)realloc(g->names, (base + (size_t)ntensor + 1) * sizeof(*names));
    if (names == NULL) { ingot_err(err, errsz, "out of memory"); return -1; }
    g->names = names;
    memset(g->tensors + base, 0, ((size_t)ntensor + 1) * sizeof(*g->tensors));
    memset(g->names + base, 0, ((size_t)ntensor + 1) * sizeof(*g->names));

    for (uint64_t i = 0; i < ntensor; i++) {
        ingot_tensor *t = &g->tensors[base + (size_t)i];
        const char *name;
        uint64_t namelen;
        uint32_t rank, type;
        if (cur_str(&c, &name, &namelen) != 0) {
            ingot_err(err, errsz, "GGUF tensor table truncated at %" PRIu64, i);
            return -1;
        }
        g->names[base + (size_t)i] = ingot_strndup(name, (size_t)namelen);
        if (g->names[base + (size_t)i] == NULL) {
            ingot_err(err, errsz, "out of memory");
            return -1;
        }
        g->ntensor = base + (size_t)i + 1;      /* freeable from here on */
        t->name = g->names[base + (size_t)i];
        t->shard = index;

        if (cur_u32(&c, &rank) != 0) {
            ingot_err(err, errsz, "'%s' ends inside the entry for tensor '%s'",
                      s->path, t->name);
            return -1;
        }
        if (rank > INGOT_MAX_RANK) {
            ingot_err(err, errsz, "tensor '%s' has rank %u (max %d)",
                      t->name, rank, INGOT_MAX_RANK);
            return -1;
        }
        t->rank = rank;
        uint64_t nelem = 1;
        for (uint32_t d = 0; d < rank; d++) {
            if (cur_u64(&c, &t->ne[d]) != 0 || t->ne[d] == 0 ||
                ingot_mul_u64(nelem, t->ne[d], &nelem) != 0) {
                ingot_err(err, errsz, "tensor '%s' has an invalid shape", t->name);
                return -1;
            }
        }
        t->nelem = nelem;
        if (cur_u32(&c, &type) != 0 || cur_u64(&c, &t->offset) != 0) {
            ingot_err(err, errsz, "tensor '%s' header truncated", t->name);
            return -1;
        }
        t->type = (int)type;

        uint64_t blk_elems, blk_bytes;
        if (ingot_type_geometry(t->type, &blk_elems, &blk_bytes) != 0) {
            ingot_err(err, errsz, "tensor '%s': unknown ggml type %u", t->name, type);
            return -1;
        }
        /* A quantized block must fit whole inside the fastest dimension: ggml
         * lays blocks out along ne[0], so a row that is not a multiple of the
         * block size would make blocks straddle rows and every kernel would
         * read the wrong scales. */
        if (blk_elems > 1 && (rank == 0 || t->ne[0] % blk_elems != 0)) {
            ingot_err(err, errsz, "tensor '%s': %s needs ne[0] to be a multiple of %"
                      PRIu64, t->name, ingot_type_name(t->type), blk_elems);
            return -1;
        }
        if (ingot_type_nbytes(t->type, nelem, &t->nbytes) != 0) {
            ingot_err(err, errsz, "tensor '%s': %s cannot hold %" PRIu64 " elements",
                      t->name, ingot_type_name(t->type), nelem);
            return -1;
        }
    }

    int ok;
    s->data_base = align_up(c.pos, s->alignment, &ok);
    if (!ok || s->data_base > s->size) {
        ingot_err(err, errsz, "GGUF alignment %" PRIu64 " is invalid in '%s'",
                  s->alignment, s->path);
        return -1;
    }
    /* Every payload revalidated against the real file size, after the header
     * has told us where the data section starts. */
    for (size_t i = base; i < g->ntensor; i++) {
        const ingot_tensor *t = &g->tensors[i];
        uint64_t absolute, end;
        if (t->offset % s->alignment != 0) {
            ingot_err(err, errsz, "tensor '%s' is not aligned to %" PRIu64,
                      t->name, s->alignment);
            return -1;
        }
        if (ingot_add_u64(s->data_base, t->offset, &absolute) != 0 ||
            ingot_add_u64(absolute, t->nbytes, &end) != 0 || end > s->size) {
            ingot_err(err, errsz, "tensor '%s' payload lies outside '%s'", t->name, s->path);
            return -1;
        }
    }
    return 0;
}

/* ── split-file naming: <prefix>-00001-of-00003.gguf ────────────────────── */
static int split_expand(const char *path, char ***out, uint32_t *count) {
    const size_t len = strlen(path);
    const char suffix[] = ".gguf";
    const size_t slen = sizeof(suffix) - 1;
    const size_t tlen = 15;                         /* "-NNNNN-of-NNNNN" */
    if (len < slen + tlen) return 0;
    if (strcmp(path + len - slen, suffix) != 0) return 0;
    const char *tail = path + len - slen - tlen;
    if (tail[0] != '-' || tail[6] != '-' ||
        tail[7] != 'o' || tail[8] != 'f' || tail[9] != '-') return 0;
    unsigned no = 0, total = 0;
    for (int i = 1; i <= 5; i++) {
        if (tail[i] < '0' || tail[i] > '9') return 0;
        no = no * 10 + (unsigned)(tail[i] - '0');
    }
    for (int i = 10; i <= 14; i++) {
        if (tail[i] < '0' || tail[i] > '9') return 0;
        total = total * 10 + (unsigned)(tail[i] - '0');
    }
    if (total < 1 || total > 999 || no < 1 || no > total) return 0;
    if (total == 1) return 0;

    const size_t prefix = (size_t)(tail - path);
    char **paths = (char **)calloc(total, sizeof(*paths));
    if (paths == NULL) return -1;
    for (unsigned i = 0; i < total; i++) {
        const size_t n = prefix + tlen + slen + 1;
        paths[i] = (char *)malloc(n);
        if (paths[i] == NULL) {
            for (unsigned k = 0; k < i; k++) free(paths[k]);
            free(paths);
            return -1;
        }
        memcpy(paths[i], path, prefix);
        snprintf(paths[i] + prefix, n - prefix, "-%05u-of-%05u.gguf", i + 1, total);
    }
    *out = paths;
    *count = total;
    return 1;
}

/* ── open / close ───────────────────────────────────────────────────────── */
static int open_shards(ingot_gguf **out, char *const *paths, uint32_t n,
                       char *err, size_t errsz) {
    *out = NULL;
    ingot_gguf *g = (ingot_gguf *)calloc(1, sizeof(*g));
    if (g == NULL) { ingot_err(err, errsz, "out of memory"); return -1; }
    g->shards = (shard_t *)calloc(n, sizeof(*g->shards));
    if (g->shards == NULL) {
        free(g);
        ingot_err(err, errsz, "out of memory");
        return -1;
    }
    for (uint32_t i = 0; i < n; i++) g->shards[i].fd = -1;

    for (uint32_t i = 0; i < n; i++) {
        if (shard_map(&g->shards[i], paths[i], err, errsz) != 0) goto fail;
        g->nshard = i + 1;
        if (shard_parse(g, i, i == 0, err, errsz) != 0) goto fail;
    }
    if (ingot_index_build(&g->tensor_index, g->tensors, g->ntensor, tensor_name_of) != 0 ||
        ingot_index_build(&g->kv_index, g->kvs, g->nkv, kv_name_of) != 0) {
        ingot_err(err, errsz, "out of memory building the name index");
        goto fail;
    }
    /* A name present in two shards makes every lookup ambiguous; better to say
     * so than to silently pick one. */
    for (size_t i = 0; i < g->ntensor; i++) {
        if (ingot_index_find(&g->tensor_index, g->tensors, tensor_name_of,
                             g->tensors[i].name) != i) {
            ingot_err(err, errsz, "tensor '%s' appears more than once", g->tensors[i].name);
            goto fail;
        }
    }
    *out = g;
    return 0;
fail:
    ingot_gguf_close(g);
    return -1;
}

int ingot_gguf_open(ingot_gguf **out, const char *path, char *err, size_t errsz) {
    if (out == NULL || path == NULL) {
        ingot_err(err, errsz, "ingot_gguf_open: null argument");
        return -1;
    }
    char *one = (char *)path;   /* open_shards never writes through it */
    return open_shards(out, &one, 1, err, errsz);
}

int ingot_gguf_open_split(ingot_gguf **out, const char *path, char *err, size_t errsz) {
    if (out == NULL || path == NULL) {
        ingot_err(err, errsz, "ingot_gguf_open_split: null argument");
        return -1;
    }
    char **paths = NULL;
    uint32_t n = 0;
    const int split = split_expand(path, &paths, &n);
    if (split < 0) { ingot_err(err, errsz, "out of memory"); return -1; }
    if (split == 0) return ingot_gguf_open(out, path, err, errsz);
    const int rc = open_shards(out, paths, n, err, errsz);
    for (uint32_t i = 0; i < n; i++) free(paths[i]);
    free(paths);
    return rc;
}

void ingot_gguf_close(ingot_gguf *g) {
    if (g == NULL) return;
    for (size_t i = 0; i < g->nkv; i++) kv_free_one(&g->kvs[i]);
    free(g->kvs);
    if (g->names != NULL)
        for (size_t i = 0; i < g->ntensor; i++) free(g->names[i]);
    free(g->names);
    free(g->tensors);
    ingot_index_free(&g->tensor_index);
    ingot_index_free(&g->kv_index);
    for (uint32_t i = 0; i < g->nshard; i++) gguf_shard_close(&g->shards[i]);
    free(g->shards);
    free(g);
}

/* ── tensors ────────────────────────────────────────────────────────────── */
size_t ingot_gguf_count(const ingot_gguf *g) { return g != NULL ? g->ntensor : 0; }

const ingot_tensor *ingot_gguf_at(const ingot_gguf *g, size_t index) {
    return (g != NULL && index < g->ntensor) ? &g->tensors[index] : NULL;
}

const ingot_tensor *ingot_gguf_find(const ingot_gguf *g, const char *name) {
    if (g == NULL) return NULL;
    const size_t i = ingot_index_find(&g->tensor_index, g->tensors, tensor_name_of, name);
    return i == (size_t)-1 ? NULL : &g->tensors[i];
}

const void *ingot_gguf_data(const ingot_gguf *g, const ingot_tensor *t) {
    if (g == NULL || t == NULL || t->shard >= g->nshard) return NULL;
    const shard_t *s = &g->shards[t->shard];
    uint64_t absolute, end;
    if (ingot_add_u64(s->data_base, t->offset, &absolute) != 0 ||
        ingot_add_u64(absolute, t->nbytes, &end) != 0 || end > s->size) return NULL;
    return s->map + absolute;
}

int ingot_gguf_read(const ingot_gguf *g, const ingot_tensor *t, uint64_t offset,
                    void *dst, size_t nbytes, char *err, size_t errsz) {
    if (g == NULL || t == NULL || t->shard >= g->nshard || (dst == NULL && nbytes != 0) ||
        offset > t->nbytes || (uint64_t)nbytes > t->nbytes - offset) {
        ingot_err(err, errsz, "ingot_gguf_read: range outside tensor");
        return -1;
    }
    const shard_t *s = &g->shards[t->shard];
    uint64_t absolute;
    if (ingot_add_u64(s->data_base, t->offset, &absolute) != 0 ||
        ingot_add_u64(absolute, offset, &absolute) != 0 ||
        absolute > s->size || (uint64_t)nbytes > s->size - absolute) {
        ingot_err(err, errsz, "ingot_gguf_read: range outside file");
        return -1;
    }
    size_t done = 0;
    while (done < nbytes) {
        const ssize_t got = pread(s->fd, (unsigned char *)dst + done, nbytes - done,
                                  (off_t)(absolute + done));
        if (got <= 0) {
            ingot_err(err, errsz, "read of tensor '%s' failed: %s", t->name,
                      got == 0 ? "unexpected EOF" : strerror(errno));
            return -1;
        }
        done += (size_t)got;
    }
    return 0;
}

int ingot_gguf_mapping(const ingot_gguf *g, uint32_t shard,
                       const void **base, size_t *size) {
    if (g == NULL || shard >= g->nshard || base == NULL || size == NULL) return -1;
    *base = g->shards[shard].map;
    *size = (size_t)g->shards[shard].size;
    return 0;
}

void ingot_gguf_shape_row_major(const ingot_tensor *t, uint64_t *shape) {
    if (t == NULL || shape == NULL) return;
    for (uint32_t i = 0; i < t->rank; i++) shape[i] = t->ne[t->rank - 1 - i];
}

/* ── metadata ───────────────────────────────────────────────────────────── */
size_t ingot_gguf_kv_count(const ingot_gguf *g) { return g != NULL ? g->nkv : 0; }

const ingot_kv *ingot_gguf_kv_at(const ingot_gguf *g, size_t index) {
    return (g != NULL && index < g->nkv) ? &g->kvs[index] : NULL;
}

const ingot_kv *ingot_gguf_kv_find(const ingot_gguf *g, const char *key) {
    if (g == NULL) return NULL;
    const size_t i = ingot_index_find(&g->kv_index, g->kvs, kv_name_of, key);
    return i == (size_t)-1 ? NULL : &g->kvs[i];
}

const char *ingot_kv_key(const ingot_kv *kv)  { return kv != NULL ? kv->key : NULL; }
int         ingot_kv_type(const ingot_kv *kv) { return kv != NULL ? kv->type : -1; }
int         ingot_kv_arr_type(const ingot_kv *kv) { return kv != NULL ? kv->arr_type : -1; }

int ingot_kv_arr_len(const ingot_kv *kv, uint64_t *out) {
    if (kv == NULL || out == NULL || kv->type != INGOT_KV_ARRAY) return -1;
    *out = kv->arr_len;
    return 0;
}

int ingot_kv_str(const ingot_kv *kv, const char **out) {
    if (kv == NULL || out == NULL || kv->type != INGOT_KV_STRING) return -1;
    *out = kv->strval;
    return 0;
}

/* One decoder for every integer width, so a model that stores head_count as
 * u32 and another that stores it as u64 both read through _u64 without the
 * caller caring. */
static int kv_read_scalar(int type, const unsigned char *p,
                          uint64_t *u, int64_t *i, double *f) {
    switch (type) {
    case INGOT_KV_UINT8:  *u = p[0]; *i = (int64_t)p[0]; *f = (double)p[0]; return 0;
    case INGOT_KV_BOOL:   *u = p[0] ? 1 : 0; *i = *u ? 1 : 0; *f = (double)*i; return 0;
    case INGOT_KV_INT8:   *i = (int8_t)p[0]; *u = (uint64_t)*i; *f = (double)*i; return 0;
    case INGOT_KV_UINT16: *u = ingot_ld_u16(p); *i = (int64_t)*u; *f = (double)*u; return 0;
    case INGOT_KV_INT16:  *i = (int16_t)ingot_ld_u16(p); *u = (uint64_t)*i; *f = (double)*i; return 0;
    case INGOT_KV_UINT32: *u = ingot_ld_u32(p); *i = (int64_t)*u; *f = (double)*u; return 0;
    case INGOT_KV_INT32:  *i = (int32_t)ingot_ld_u32(p); *u = (uint64_t)*i; *f = (double)*i; return 0;
    case INGOT_KV_UINT64: *u = ingot_ld_u64(p); *i = (int64_t)*u; *f = (double)*u; return 0;
    case INGOT_KV_INT64:  *i = (int64_t)ingot_ld_u64(p); *u = (uint64_t)*i; *f = (double)*i; return 0;
    case INGOT_KV_FLOAT32: {
        const uint32_t bits = ingot_ld_u32(p);
        float v;
        memcpy(&v, &bits, sizeof(v));
        *f = (double)v; *i = f64_to_i64_sat((double)v); *u = f64_to_u64_sat((double)v);
        return 0;
    }
    case INGOT_KV_FLOAT64: {
        const uint64_t bits = ingot_ld_u64(p);
        double v;
        memcpy(&v, &bits, sizeof(v));
        *f = v; *i = f64_to_i64_sat(v); *u = f64_to_u64_sat(v);
        return 0;
    }
    default: return -1;
    }
}

int ingot_kv_u64(const ingot_kv *kv, uint64_t *out) {
    uint64_t u; int64_t i; double f;
    if (kv == NULL || out == NULL || kv_read_scalar(kv->type, kv->payload, &u, &i, &f) != 0)
        return -1;
    *out = u;
    return 0;
}
int ingot_kv_i64(const ingot_kv *kv, int64_t *out) {
    uint64_t u; int64_t i; double f;
    if (kv == NULL || out == NULL || kv_read_scalar(kv->type, kv->payload, &u, &i, &f) != 0)
        return -1;
    *out = i;
    return 0;
}
int ingot_kv_f64(const ingot_kv *kv, double *out) {
    uint64_t u; int64_t i; double f;
    if (kv == NULL || out == NULL || kv_read_scalar(kv->type, kv->payload, &u, &i, &f) != 0)
        return -1;
    *out = f;
    return 0;
}
int ingot_kv_bool(const ingot_kv *kv, int *out) {
    uint64_t u;
    if (kv == NULL || out == NULL || kv->type != INGOT_KV_BOOL ||
        ingot_kv_u64(kv, &u) != 0) return -1;
    *out = u != 0;
    return 0;
}

int ingot_kv_arr_str(const ingot_kv *kv, uint64_t index, const char **out, size_t *len) {
    if (kv == NULL || out == NULL || len == NULL || kv->type != INGOT_KV_ARRAY ||
        kv->arr_type != INGOT_KV_STRING || index >= kv->arr_len ||
        kv->str_off == NULL || kv->str_len == NULL) return -1;
    /* The length was bounds-checked at parse time; re-reading it from the
     * mapping would trust bytes the file may have changed since (MAP_PRIVATE
     * does not snapshot unfaulted pages). */
    const unsigned char *p = kv->payload + kv->str_off[index];
    *len = (size_t)kv->str_len[index];
    *out = (const char *)(p + 8);
    return 0;
}

int ingot_kv_arr_f32(const ingot_kv *kv, uint64_t index, float *out) {
    if (kv == NULL || out == NULL || kv->type != INGOT_KV_ARRAY || index >= kv->arr_len)
        return -1;
    uint64_t esz;
    if (kv_scalar_bytes(kv->arr_type, &esz) != 0) return -1;
    uint64_t u; int64_t i; double f;
    if (kv_read_scalar(kv->arr_type, kv->payload + index * esz, &u, &i, &f) != 0) return -1;
    *out = (float)f;
    return 0;
}

int ingot_kv_arr_i64(const ingot_kv *kv, uint64_t index, int64_t *out) {
    if (kv == NULL || out == NULL || kv->type != INGOT_KV_ARRAY || index >= kv->arr_len)
        return -1;
    uint64_t esz;
    if (kv_scalar_bytes(kv->arr_type, &esz) != 0) return -1;
    uint64_t u; int64_t i; double f;
    if (kv_read_scalar(kv->arr_type, kv->payload + index * esz, &u, &i, &f) != 0) return -1;
    *out = i;
    return 0;
}

/* ── convenience ────────────────────────────────────────────────────────── */
const char *ingot_gguf_arch(const ingot_gguf *g) {
    const ingot_kv *kv = ingot_gguf_kv_find(g, "general.architecture");
    const char *value = NULL;
    if (kv == NULL || ingot_kv_str(kv, &value) != 0 || value == NULL) return "";
    return value;
}
uint64_t ingot_gguf_alignment(const ingot_gguf *g) {
    return (g != NULL && g->nshard > 0) ? g->shards[0].alignment : 0;
}
uint32_t ingot_gguf_version(const ingot_gguf *g) {
    return (g != NULL && g->nshard > 0) ? g->shards[0].version : 0;
}
uint32_t ingot_gguf_shard_count(const ingot_gguf *g) {
    return g != NULL ? g->nshard : 0;
}
uint64_t ingot_gguf_data_base(const ingot_gguf *g, uint32_t shard) {
    return (g != NULL && shard < g->nshard) ? g->shards[shard].data_base : 0;
}

/* ═══ src/safetensors.c ═══ */
/* safetensors reader.
 *
 * The JSON is hand-rolled so there is no cJSON to vendor. Every tensor is
 * validated exactly (`elements * dtype_size == data_size`), the shard bundle
 * refuses duplicate names instead of letting the first one win, and nothing
 * here has a fixed cap: a reader with a 1024-tensor array loads half a model
 * and says nothing, which is the failure this deliberately does not have.
 *
 * SPDX-License-Identifier: MIT */
#if defined(__APPLE__)
/* madvise/MADV_DONTNEED are outside the strict POSIX namespace on Darwin. */
#endif


#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

typedef struct {
    char          *path;
    int            fd;
    uint64_t       size;
    unsigned char *map;
    uint64_t       data_start;
} st_shard;

struct ingot_st {
    st_shard        *shards;
    uint32_t         nshard;
    ingot_st_tensor *tensors;
    size_t           ntensor, tensor_cap;
    char           **names;
    ingot_index      index;
    char           **meta_keys, **meta_vals;
    size_t           nmeta;
    int              drop_cache;
};

/* ── a small, bounded JSON reader ───────────────────────────────────────── */
typedef struct { const char *p, *end; } jcur;

static void jws(jcur *j) {
    while (j->p < j->end && (*j->p == ' ' || *j->p == '\t' ||
                             *j->p == '\n' || *j->p == '\r')) j->p++;
}
static int jchar(jcur *j, char expected) {
    jws(j);
    if (j->p >= j->end || *j->p != expected) return -1;
    j->p++;
    return 0;
}
static int jpeek(jcur *j, char expected) {
    jws(j);
    return (j->p < j->end && *j->p == expected) ? 0 : -1;
}

static void utf8_put(char **dst, uint32_t cp) {
    unsigned char *o = (unsigned char *)*dst;
    if (cp < 0x80) { *o++ = (unsigned char)cp; }
    else if (cp < 0x800) {
        *o++ = (unsigned char)(0xc0 | (cp >> 6));
        *o++ = (unsigned char)(0x80 | (cp & 0x3f));
    } else if (cp < 0x10000) {
        *o++ = (unsigned char)(0xe0 | (cp >> 12));
        *o++ = (unsigned char)(0x80 | ((cp >> 6) & 0x3f));
        *o++ = (unsigned char)(0x80 | (cp & 0x3f));
    } else {
        *o++ = (unsigned char)(0xf0 | (cp >> 18));
        *o++ = (unsigned char)(0x80 | ((cp >> 12) & 0x3f));
        *o++ = (unsigned char)(0x80 | ((cp >> 6) & 0x3f));
        *o++ = (unsigned char)(0x80 | (cp & 0x3f));
    }
    *dst = (char *)o;
}

static int jhex4(const char *p, uint32_t *out) {
    uint32_t v = 0;
    for (int i = 0; i < 4; i++) {
        const char c = p[i];
        v <<= 4;
        if (c >= '0' && c <= '9') v |= (uint32_t)(c - '0');
        else if (c >= 'a' && c <= 'f') v |= (uint32_t)(c - 'a' + 10);
        else if (c >= 'A' && c <= 'F') v |= (uint32_t)(c - 'A' + 10);
        else return -1;
    }
    *out = v;
    return 0;
}

/* Allocates a NUL-terminated, unescaped copy. Tensor names really do contain
 * dots and slashes, and occasionally an escape; the caller always wants a
 * C string, so unescaping here beats handing out views nobody can strcmp. */
static int jstring(jcur *j, char **out) {
    jws(j);
    if (j->p >= j->end || *j->p != '"') return -1;
    j->p++;
    const char *start = j->p;
    /* worst case: every \uXXXX becomes 3 bytes, so the raw span is an upper
     * bound on the decoded length. */
    const char *scan = start;
    while (scan < j->end && *scan != '"') {
        if (*scan == '\\') { scan++; if (scan >= j->end) return -1; }
        scan++;
    }
    if (scan >= j->end) return -1;
    char *buffer = (char *)malloc((size_t)(scan - start) + 1);
    if (buffer == NULL) return -1;
    char *w = buffer;
    while (j->p < scan) {
        if (*j->p != '\\') { *w++ = *j->p++; continue; }
        j->p++;
        if (j->p >= scan) { free(buffer); return -1; }
        switch (*j->p) {
        case 'n': *w++ = '\n'; j->p++; break;
        case 't': *w++ = '\t'; j->p++; break;
        case 'r': *w++ = '\r'; j->p++; break;
        case 'b': *w++ = '\b'; j->p++; break;
        case 'f': *w++ = '\f'; j->p++; break;
        case '"': case '\\': case '/': *w++ = *j->p++; break;
        case 'u': {
            if (j->p + 5 > scan) { free(buffer); return -1; }
            uint32_t cp;
            if (jhex4(j->p + 1, &cp) != 0) { free(buffer); return -1; }
            j->p += 5;
            if (cp >= 0xd800 && cp <= 0xdbff && j->p + 6 <= scan &&
                j->p[0] == '\\' && j->p[1] == 'u') {
                uint32_t low;
                if (jhex4(j->p + 2, &low) == 0 && low >= 0xdc00 && low <= 0xdfff) {
                    cp = 0x10000 + ((cp - 0xd800) << 10) + (low - 0xdc00);
                    j->p += 6;
                }
            }
            utf8_put(&w, cp);
            break;
        }
        default: free(buffer); return -1;
        }
    }
    *w = '\0';
    j->p = scan + 1;
    *out = buffer;
    return 0;
}

static int ju64(jcur *j, uint64_t *out) {
    jws(j);
    if (j->p >= j->end || *j->p < '0' || *j->p > '9') return -1;
    uint64_t v = 0;
    while (j->p < j->end && *j->p >= '0' && *j->p <= '9') {
        const uint64_t d = (uint64_t)(*j->p - '0');
        if (v > (UINT64_MAX - d) / 10) return -1;
        v = v * 10 + d;
        j->p++;
    }
    *out = v;
    return 0;
}

static int jskip(jcur *j, unsigned depth) {
    if (depth > 32) return -1;
    jws(j);
    if (j->p >= j->end) return -1;
    if (*j->p == '"') {
        char *tmp = NULL;
        if (jstring(j, &tmp) != 0) return -1;
        free(tmp);
        return 0;
    }
    if (*j->p == '{' || *j->p == '[') {
        const char close = (*j->p == '{') ? '}' : ']';
        const int object = (*j->p == '{');
        j->p++;
        if (jpeek(j, close) == 0) { j->p++; return 0; }
        for (;;) {
            if (object) {
                char *key = NULL;
                if (jstring(j, &key) != 0) return -1;
                free(key);
                if (jchar(j, ':') != 0) return -1;
            }
            if (jskip(j, depth + 1) != 0) return -1;
            jws(j);
            if (j->p >= j->end) return -1;
            if (*j->p == close) { j->p++; return 0; }
            if (*j->p != ',') return -1;
            j->p++;
        }
    }
    /* number, true, false, null */
    while (j->p < j->end && *j->p != ',' && *j->p != '}' && *j->p != ']') j->p++;
    return 0;
}

/* ── tensor table ───────────────────────────────────────────────────────── */
static int tensors_push(ingot_st *st, char *name, const ingot_st_tensor *t,
                        char *err, size_t errsz) {
    if (st->ntensor == st->tensor_cap) {
        const size_t next = st->tensor_cap == 0 ? 64 : st->tensor_cap * 2;
        ingot_st_tensor *grown = (ingot_st_tensor *)realloc(st->tensors, next * sizeof(*grown));
        char **names = (char **)realloc(st->names, next * sizeof(*names));
        if (grown != NULL) st->tensors = grown;
        if (names != NULL) st->names = names;
        if (grown == NULL || names == NULL) {
            ingot_err(err, errsz, "out of memory");
            return -1;
        }
        st->tensor_cap = next;
    }
    st->names[st->ntensor] = name;
    st->tensors[st->ntensor] = *t;
    st->tensors[st->ntensor].name = name;
    st->ntensor++;
    return 0;
}

static int parse_tensor_object(jcur *j, const char *name, ingot_st_tensor *t,
                               char *err, size_t errsz) {
    int has_dtype = 0, has_shape = 0, has_offsets = 0;
    uint64_t begin = 0, end = 0;
    memset(t, 0, sizeof(*t));
    t->dtype = INGOT_DT_UNKNOWN;
    if (jchar(j, '{') != 0) {
        ingot_err(err, errsz, "tensor '%s' is not an object", name);
        return -1;
    }
    if (jpeek(j, '}') == 0) {
        ingot_err(err, errsz, "tensor '%s' has an empty descriptor", name);
        return -1;
    }
    for (;;) {
        char *key = NULL;
        if (jstring(j, &key) != 0 || jchar(j, ':') != 0) {
            free(key);
            ingot_err(err, errsz, "tensor '%s' has a malformed field", name);
            return -1;
        }
        int bad = 0;
        if (strcmp(key, "dtype") == 0) {
            char *dtype = NULL;
            if (jstring(j, &dtype) != 0) bad = 1;
            else { t->dtype = ingot_dtype_from_name(dtype); has_dtype = 1; free(dtype); }
        } else if (strcmp(key, "shape") == 0) {
            if (jchar(j, '[') != 0) bad = 1;
            else if (jpeek(j, ']') == 0) { j->p++; t->rank = 0; has_shape = 1; }
            else {
                for (;;) {
                    if (t->rank >= INGOT_MAX_RANK || ju64(j, &t->shape[t->rank]) != 0) {
                        ingot_err(err, errsz, "tensor '%s' has rank > %d or a bad dimension",
                                  name, INGOT_MAX_RANK);
                        free(key);
                        return -1;
                    }
                    t->rank++;
                    jws(j);
                    if (j->p >= j->end) { bad = 1; break; }
                    if (*j->p == ']') { j->p++; has_shape = 1; break; }
                    if (*j->p != ',') { bad = 1; break; }
                    j->p++;
                }
            }
        } else if (strcmp(key, "data_offsets") == 0) {
            if (jchar(j, '[') != 0 || ju64(j, &begin) != 0 || jchar(j, ',') != 0 ||
                ju64(j, &end) != 0 || jchar(j, ']') != 0 || end < begin) bad = 1;
            else has_offsets = 1;
        } else if (jskip(j, 0) != 0) {
            bad = 1;
        }
        free(key);
        if (bad) {
            ingot_err(err, errsz, "tensor '%s' has a malformed descriptor", name);
            return -1;
        }
        jws(j);
        if (j->p >= j->end) {
            ingot_err(err, errsz, "tensor '%s' descriptor is truncated", name);
            return -1;
        }
        if (*j->p == '}') { j->p++; break; }
        if (*j->p != ',') {
            ingot_err(err, errsz, "tensor '%s' descriptor is malformed", name);
            return -1;
        }
        j->p++;
    }
    if (!has_dtype || !has_shape || !has_offsets) {
        ingot_err(err, errsz, "tensor '%s' is missing dtype, shape or data_offsets", name);
        return -1;
    }
    t->offset = begin;
    t->nbytes = end - begin;
    t->nelem = 1;
    for (uint32_t d = 0; d < t->rank; d++) {
        if (t->shape[d] == 0) { t->nelem = 0; break; }
        if (ingot_mul_u64(t->nelem, t->shape[d], &t->nelem) != 0) {
            ingot_err(err, errsz, "tensor '%s' shape overflows", name);
            return -1;
        }
    }
    /* Exact byte accounting, for the dtypes we know. An unknown dtype is kept
     * (so a caller can still see the tensor and its bytes) but never sized. */
    const size_t item = ingot_dtype_size(t->dtype);
    if (item != 0) {
        uint64_t expected;
        if (ingot_mul_u64(t->nelem, item, &expected) != 0 || expected != t->nbytes) {
            ingot_err(err, errsz,
                      "tensor '%s': %s of %" PRIu64 " elements needs %" PRIu64
                      " bytes, header says %" PRIu64,
                      name, ingot_dtype_name(t->dtype), t->nelem,
                      (uint64_t)(t->nelem * item), t->nbytes);
            return -1;
        }
    }
    return 0;
}

static int parse_metadata(ingot_st *st, jcur *j, char *err, size_t errsz) {
    if (jchar(j, '{') != 0) return -1;
    if (jpeek(j, '}') == 0) { j->p++; return 0; }
    for (;;) {
        char *key = NULL, *value = NULL;
        if (jstring(j, &key) != 0 || jchar(j, ':') != 0) { free(key); return -1; }
        jws(j);
        if (j->p < j->end && *j->p == '"') {
            if (jstring(j, &value) != 0) { free(key); return -1; }
            char **keys = (char **)realloc(st->meta_keys, (st->nmeta + 1) * sizeof(*keys));
            char **vals = (char **)realloc(st->meta_vals, (st->nmeta + 1) * sizeof(*vals));
            if (keys != NULL) st->meta_keys = keys;
            if (vals != NULL) st->meta_vals = vals;
            if (keys == NULL || vals == NULL) {
                free(key); free(value);
                ingot_err(err, errsz, "out of memory");
                return -1;
            }
            st->meta_keys[st->nmeta] = key;
            st->meta_vals[st->nmeta] = value;
            st->nmeta++;
        } else {                       /* non-string metadata: kept out */
            free(key);
            if (jskip(j, 0) != 0) return -1;
        }
        jws(j);
        if (j->p >= j->end) return -1;
        if (*j->p == '}') { j->p++; return 0; }
        if (*j->p != ',') return -1;
        j->p++;
    }
}

static int parse_header(ingot_st *st, uint32_t shard, const char *json, size_t len,
                        char *err, size_t errsz) {
    jcur j = { json, json + len };
    if (jchar(&j, '{') != 0) {
        ingot_err(err, errsz, "safetensors header is not a JSON object");
        return -1;
    }
    if (jpeek(&j, '}') == 0) return 0;
    for (;;) {
        char *name = NULL;
        if (jstring(&j, &name) != 0 || jchar(&j, ':') != 0) {
            free(name);
            ingot_err(err, errsz, "safetensors header is malformed");
            return -1;
        }
        if (strcmp(name, "__metadata__") == 0) {
            free(name);
            const int rc = (shard == 0) ? parse_metadata(st, &j, err, errsz) : jskip(&j, 0);
            if (rc != 0) {
                ingot_err(err, errsz, "safetensors __metadata__ is malformed");
                return -1;
            }
        } else {
            ingot_st_tensor t;
            if (parse_tensor_object(&j, name, &t, err, errsz) != 0) {
                free(name);
                return -1;
            }
            t.shard = shard;
            if (tensors_push(st, name, &t, err, errsz) != 0) {
                free(name);
                return -1;
            }
        }
        jws(&j);
        if (j.p >= j.end) {
            ingot_err(err, errsz, "safetensors header is truncated");
            return -1;
        }
        if (*j.p == '}') return 0;
        if (*j.p != ',') {
            ingot_err(err, errsz, "safetensors header is malformed");
            return -1;
        }
        j.p++;
    }
}

/* ── shards ─────────────────────────────────────────────────────────────── */
static void st_shard_close(st_shard *s, int drop_cache) {
    if (s->map != NULL && s->map != MAP_FAILED) munmap(s->map, (size_t)s->size);
    /* The page cache is dropped AFTER munmap (the kernel keeps pages that are
     * still mapped) and BEFORE close (it needs the descriptor). posix_madvise
     * is not enough: on Linux, for a mapped file, it does not touch the page
     * cache — only posix_fadvise on the descriptor does. */
#if !defined(__APPLE__)
    if (drop_cache && s->fd >= 0) (void)posix_fadvise(s->fd, 0, 0, POSIX_FADV_DONTNEED);
#else
    (void)drop_cache;
#endif
    if (s->fd >= 0) close(s->fd);
    free(s->path);
    memset(s, 0, sizeof(*s));
    s->fd = -1;
}

static int shard_open(ingot_st *st, uint32_t index, const char *path,
                      char *err, size_t errsz) {
    st_shard *s = &st->shards[index];
    memset(s, 0, sizeof(*s));
    s->fd = open(path, O_RDONLY);
    if (s->fd < 0) {
        ingot_err(err, errsz, "cannot open '%s': %s", path, strerror(errno));
        return -1;
    }
    struct stat sb;
    if (fstat(s->fd, &sb) != 0 || sb.st_size < 8) {
        ingot_err(err, errsz, "'%s' is too small to be safetensors", path);
        st_shard_close(s, 0);
        return -1;
    }
    s->size = (uint64_t)sb.st_size;
    /* refuse files a 32-bit size_t cannot map in full (see gguf.c) */
    if (s->size > SIZE_MAX / 2) {
        ingot_err(err, errsz, "'%s' is too large for this address space", path);
        st_shard_close(s, 0);
        return -1;
    }
    s->path = ingot_strdup(path);
    s->map = (unsigned char *)mmap(NULL, (size_t)s->size, PROT_READ, MAP_PRIVATE, s->fd, 0);
    if (s->path == NULL || s->map == MAP_FAILED) {
        ingot_err(err, errsz, "cannot mmap '%s': %s", path, strerror(errno));
        st_shard_close(s, 0);
        return -1;
    }
    const uint64_t header_len = ingot_ld_u64(s->map);
    if (header_len > s->size - 8) {
        ingot_err(err, errsz, "'%s' declares a %" PRIu64 "-byte header in a %" PRIu64
                  "-byte file", path, header_len, s->size);
        return -1;
    }
    s->data_start = 8 + header_len;
    /* safetensors pads the JSON so the data section starts 8-byte aligned, and
     * every zero-copy typed read in this library relies on that. A container
     * that breaks it is malformed, and quietly accepting it would hand out
     * misaligned f32/bf16 pointers. */
    if ((s->data_start & 7u) != 0) {
        ingot_err(err, errsz, "'%s' has an unaligned data section (header not padded to 8)",
                  path);
        return -1;
    }
    if (parse_header(st, index, (const char *)s->map + 8, (size_t)header_len, err, errsz) != 0)
        return -1;

    const uint64_t data_bytes = s->size - s->data_start;
    for (size_t i = 0; i < st->ntensor; i++) {
        const ingot_st_tensor *t = &st->tensors[i];
        if (t->shard != index) continue;
        if (t->offset > data_bytes || t->nbytes > data_bytes - t->offset) {
            ingot_err(err, errsz, "tensor '%s' payload lies outside '%s'", t->name, path);
            return -1;
        }
    }
    return 0;
}

static const char *st_name_of(const void *items, size_t i) {
    return ((const ingot_st_tensor *)items)[i].name;
}

int ingot_st_open_shards(ingot_st **out, const char *const *paths, size_t count,
                         char *err, size_t errsz) {
    if (out == NULL || paths == NULL || count == 0 || count > UINT32_MAX) {
        ingot_err(err, errsz, "ingot_st_open_shards: bad arguments");
        return -1;
    }
    *out = NULL;
    ingot_st *st = (ingot_st *)calloc(1, sizeof(*st));
    if (st == NULL) { ingot_err(err, errsz, "out of memory"); return -1; }
    st->shards = (st_shard *)calloc(count, sizeof(*st->shards));
    if (st->shards == NULL) {
        free(st);
        ingot_err(err, errsz, "out of memory");
        return -1;
    }
    for (size_t i = 0; i < count; i++) st->shards[i].fd = -1;
    for (size_t i = 0; i < count; i++) {
        if (paths[i] == NULL) { ingot_err(err, errsz, "null shard path"); goto fail; }
        st->nshard = (uint32_t)i + 1;
        if (shard_open(st, (uint32_t)i, paths[i], err, errsz) != 0) goto fail;
    }
    if (ingot_index_build(&st->index, st->tensors, st->ntensor, st_name_of) != 0) {
        ingot_err(err, errsz, "out of memory building the name index");
        goto fail;
    }
    for (size_t i = 0; i < st->ntensor; i++) {
        if (ingot_index_find(&st->index, st->tensors, st_name_of, st->tensors[i].name) != i) {
            ingot_err(err, errsz, "tensor '%s' appears in more than one shard",
                      st->tensors[i].name);
            goto fail;
        }
    }
    *out = st;
    return 0;
fail:
    ingot_st_close(st);
    return -1;
}

int ingot_st_open(ingot_st **out, const char *path, char *err, size_t errsz) {
    if (out == NULL || path == NULL) {
        ingot_err(err, errsz, "ingot_st_open: null argument");
        return -1;
    }
    struct stat sb;
    if (stat(path, &sb) == 0 && S_ISDIR(sb.st_mode))
        return ingot_st_open_dir(out, path, err, errsz);
    return ingot_st_open_shards(out, &path, 1, err, errsz);
}

/* ── directory resolution ───────────────────────────────────────────────── */
typedef struct { char **items; size_t count, cap; } strlist;

static int strlist_push(strlist *l, char *value) {
    if (l->count == l->cap) {
        const size_t next = l->cap == 0 ? 8 : l->cap * 2;
        char **grown = (char **)realloc(l->items, next * sizeof(*grown));
        if (grown == NULL) return -1;
        l->items = grown;
        l->cap = next;
    }
    l->items[l->count++] = value;
    return 0;
}
static void strlist_free(strlist *l) {
    for (size_t i = 0; i < l->count; i++) free(l->items[i]);
    free(l->items);
    memset(l, 0, sizeof(*l));
}
static int strcmp_qsort(const void *a, const void *b) {
    return strcmp(*(const char *const *)a, *(const char *const *)b);
}
static int strlist_has(const strlist *l, const char *value) {
    for (size_t i = 0; i < l->count; i++)
        if (strcmp(l->items[i], value) == 0) return 1;
    return 0;
}

static char *join_path(const char *dir, const char *name) {
    const size_t n = strlen(dir) + strlen(name) + 2;
    char *path = (char *)malloc(n);
    if (path != NULL) snprintf(path, n, "%s/%s", dir, name);
    return path;
}

static int ends_with(const char *s, const char *suffix) {
    const size_t a = strlen(s), b = strlen(suffix);
    return a >= b && strcmp(s + a - b, suffix) == 0;
}

/* model.safetensors.index.json: { "metadata": {...}, "weight_map": { name: file } } */
static int read_index_json(const char *path, strlist *files) {
    const int fd = open(path, O_RDONLY);
    if (fd < 0) return -1;
    struct stat sb;
    if (fstat(fd, &sb) != 0 || sb.st_size <= 0 || sb.st_size > (64 << 20)) {
        close(fd);
        return -1;
    }
    char *text = (char *)malloc((size_t)sb.st_size);
    if (text == NULL) { close(fd); return -1; }
    ssize_t done = 0;
    while (done < sb.st_size) {
        const ssize_t got = pread(fd, text + done, (size_t)(sb.st_size - done), done);
        if (got <= 0) { free(text); close(fd); return -1; }
        done += got;
    }
    close(fd);

    jcur j = { text, text + sb.st_size };
    int rc = -1;
    if (jchar(&j, '{') != 0) goto done;
    if (jpeek(&j, '}') == 0) goto done;
    for (;;) {
        char *key = NULL;
        if (jstring(&j, &key) != 0 || jchar(&j, ':') != 0) { free(key); goto done; }
        const int is_map = strcmp(key, "weight_map") == 0;
        free(key);
        if (!is_map) {
            if (jskip(&j, 0) != 0) goto done;
        } else {
            if (jchar(&j, '{') != 0) goto done;
            if (jpeek(&j, '}') == 0) { j.p++; }
            else for (;;) {
                char *tensor = NULL, *file = NULL;
                if (jstring(&j, &tensor) != 0 || jchar(&j, ':') != 0 ||
                    jstring(&j, &file) != 0) { free(tensor); free(file); goto done; }
                free(tensor);
                if (strlist_has(files, file)) free(file);
                else if (strlist_push(files, file) != 0) { free(file); goto done; }
                jws(&j);
                if (j.p >= j.end) goto done;
                if (*j.p == '}') { j.p++; break; }
                if (*j.p != ',') goto done;
                j.p++;
            }
            rc = 0;
        }
        jws(&j);
        if (j.p >= j.end) break;
        if (*j.p == '}') break;
        if (*j.p != ',') goto done;
        j.p++;
    }
done:
    free(text);
    return (rc == 0 && files->count > 0) ? 0 : -1;
}

int ingot_st_open_dir(ingot_st **out, const char *dir, char *err, size_t errsz) {
    if (out == NULL || dir == NULL) {
        ingot_err(err, errsz, "ingot_st_open_dir: null argument");
        return -1;
    }
    strlist names = {0}, paths = {0};
    int rc = -1;

    char *index_path = join_path(dir, "model.safetensors.index.json");
    if (index_path == NULL) { ingot_err(err, errsz, "out of memory"); return -1; }
    const int have_index = read_index_json(index_path, &names) == 0;
    free(index_path);

    if (!have_index) {
        char *single = join_path(dir, "model.safetensors");
        struct stat sb;
        if (single != NULL && stat(single, &sb) == 0 && S_ISREG(sb.st_mode)) {
            free(single);
            /* Held in a local so a failed push frees it — the other three call
             * sites already do this. */
            char *only = ingot_strdup("model.safetensors");
            if (only == NULL || strlist_push(&names, only) != 0) {
                free(only);
                ingot_err(err, errsz, "out of memory");
                goto out;
            }
        } else {
            free(single);
            DIR *d = opendir(dir);
            if (d == NULL) {
                ingot_err(err, errsz, "cannot open directory '%s': %s", dir, strerror(errno));
                goto out;
            }
            const struct dirent *entry;
            while ((entry = readdir(d)) != NULL) {
                if (!ends_with(entry->d_name, ".safetensors")) continue;
                char *copy = ingot_strdup(entry->d_name);
                if (copy == NULL || strlist_push(&names, copy) != 0) {
                    free(copy);
                    closedir(d);
                    ingot_err(err, errsz, "out of memory");
                    goto out;
                }
            }
            closedir(d);
        }
    }
    if (names.count == 0) {
        ingot_err(err, errsz, "no .safetensors file in '%s'", dir);
        goto out;
    }
    qsort(names.items, names.count, sizeof(*names.items), strcmp_qsort);
    for (size_t i = 0; i < names.count; i++) {
        char *full = join_path(dir, names.items[i]);
        if (full == NULL || strlist_push(&paths, full) != 0) {
            free(full);
            ingot_err(err, errsz, "out of memory");
            goto out;
        }
    }
    rc = ingot_st_open_shards(out, (const char *const *)paths.items, paths.count, err, errsz);
out:
    strlist_free(&names);
    strlist_free(&paths);
    return rc;
}

void ingot_st_close(ingot_st *st) {
    if (st == NULL) return;
    if (st->names != NULL)
        for (size_t i = 0; i < st->ntensor; i++) free(st->names[i]);
    free(st->names);
    free(st->tensors);
    ingot_index_free(&st->index);
    for (size_t i = 0; i < st->nmeta; i++) { free(st->meta_keys[i]); free(st->meta_vals[i]); }
    free(st->meta_keys);
    free(st->meta_vals);
    for (uint32_t i = 0; i < st->nshard; i++) st_shard_close(&st->shards[i], st->drop_cache);
    free(st->shards);
    free(st);
}

/* ── accessors ──────────────────────────────────────────────────────────── */
size_t ingot_st_count(const ingot_st *st) { return st != NULL ? st->ntensor : 0; }

const ingot_st_tensor *ingot_st_at(const ingot_st *st, size_t index) {
    return (st != NULL && index < st->ntensor) ? &st->tensors[index] : NULL;
}

const ingot_st_tensor *ingot_st_find(const ingot_st *st, const char *name) {
    if (st == NULL) return NULL;
    const size_t i = ingot_index_find(&st->index, st->tensors, st_name_of, name);
    return i == (size_t)-1 ? NULL : &st->tensors[i];
}

const void *ingot_st_data(const ingot_st *st, const ingot_st_tensor *t) {
    if (st == NULL || t == NULL || t->shard >= st->nshard) return NULL;
    const st_shard *s = &st->shards[t->shard];
    uint64_t absolute, end;
    if (ingot_add_u64(s->data_start, t->offset, &absolute) != 0 ||
        ingot_add_u64(absolute, t->nbytes, &end) != 0 || end > s->size) return NULL;
    return s->map + absolute;
}

int ingot_st_read(const ingot_st *st, const ingot_st_tensor *t, uint64_t offset,
                  void *dst, size_t nbytes, char *err, size_t errsz) {
    if (st == NULL || t == NULL || t->shard >= st->nshard || (dst == NULL && nbytes != 0) ||
        offset > t->nbytes || (uint64_t)nbytes > t->nbytes - offset) {
        ingot_err(err, errsz, "ingot_st_read: range outside tensor");
        return -1;
    }
    const st_shard *s = &st->shards[t->shard];
    uint64_t absolute;
    if (ingot_add_u64(s->data_start, t->offset, &absolute) != 0 ||
        ingot_add_u64(absolute, offset, &absolute) != 0 ||
        absolute > s->size || (uint64_t)nbytes > s->size - absolute) {
        ingot_err(err, errsz, "ingot_st_read: range outside file");
        return -1;
    }
    size_t done = 0;
    while (done < nbytes) {
        const ssize_t got = pread(s->fd, (unsigned char *)dst + done, nbytes - done,
                                  (off_t)(absolute + done));
        if (got <= 0) {
            ingot_err(err, errsz, "read of tensor '%s' failed: %s", t->name,
                      got == 0 ? "unexpected EOF" : strerror(errno));
            return -1;
        }
        done += (size_t)got;
    }
    return 0;
}

int ingot_st_to_f32(const ingot_st *st, const ingot_st_tensor *t, float *dst) {
    const void *src = ingot_st_data(st, t);
    if (src == NULL || dst == NULL) return -1;
    if (t->nelem > SIZE_MAX) return -1;
    return ingot_dtype_to_f32(t->dtype, src, (size_t)t->nelem, dst);
}

uint32_t ingot_st_shard_count(const ingot_st *st) { return st != NULL ? st->nshard : 0; }

const char *ingot_st_shard_path(const ingot_st *st, uint32_t shard) {
    return (st != NULL && shard < st->nshard) ? st->shards[shard].path : NULL;
}

int ingot_st_mapping(const ingot_st *st, uint32_t shard, const void **base, size_t *size) {
    if (st == NULL || shard >= st->nshard || base == NULL || size == NULL) return -1;
    *base = st->shards[shard].map;
    *size = (size_t)st->shards[shard].size;
    return 0;
}

int ingot_st_metadata(const ingot_st *st, const char *key, const char **out) {
    if (st == NULL || key == NULL || out == NULL) return -1;
    for (size_t i = 0; i < st->nmeta; i++) {
        if (strcmp(st->meta_keys[i], key) == 0) { *out = st->meta_vals[i]; return 0; }
    }
    return -1;
}

void ingot_st_prefault(ingot_st *st) {
    if (st == NULL) return;
    volatile char sink = 0;
    for (uint32_t s = 0; s < st->nshard; s++) {
        const st_shard *sh = &st->shards[s];
        if (sh->map == NULL) continue;
        for (uint64_t off = 0; off < sh->size; off += 4096) sink = (char)(sink + sh->map[off]);
    }
    (void)sink;
}

void ingot_st_dontneed(ingot_st *st) {
    if (st == NULL) return;
    for (uint32_t s = 0; s < st->nshard; s++) {
        st_shard *sh = &st->shards[s];
        if (sh->map == NULL || sh->size == 0) continue;
#ifdef __APPLE__
        madvise(sh->map, (size_t)sh->size, MADV_DONTNEED);
#else
        posix_madvise(sh->map, (size_t)sh->size, POSIX_MADV_DONTNEED);
#endif
    }
}

void ingot_st_set_drop_cache(ingot_st *st, int drop) {
    if (st != NULL) st->drop_cache = drop ? 1 : 0;
}

/* ═══ src/wfile.c ═══ */
/* One handle for either container. See ingot/wfile.h for why this layer is
 * thin on purpose.
 *
 * SPDX-License-Identifier: MIT */

#include <stdio.h>

#ifndef INGOT_NO_KERNELS
#endif

struct ingot_wfile {
    ingot_container container;
    ingot_gguf     *gguf;
    ingot_st       *st;
    ingot_wtensor  *tensors;
    size_t          ntensor;
    ingot_index     index;
};

static const char *w_name_of(const void *items, size_t i) {
    return ((const ingot_wtensor *)items)[i].name;
}

static int build_from_gguf(ingot_wfile *w, char *err, size_t errsz) {
    w->ntensor = ingot_gguf_count(w->gguf);
    w->tensors = (ingot_wtensor *)calloc(w->ntensor ? w->ntensor : 1, sizeof(*w->tensors));
    if (w->tensors == NULL) { ingot_err(err, errsz, "out of memory"); return -1; }
    for (size_t i = 0; i < w->ntensor; i++) {
        const ingot_tensor *t = ingot_gguf_at(w->gguf, i);
        ingot_wtensor *o = &w->tensors[i];
        o->name = t->name;
        o->ggml_type = t->type;
        o->rank = t->rank;
        /* ggml stores ne[0] as the fastest dimension; everything outside ggml
         * reads shapes the other way round, so this is where the flip lives —
         * once, explicitly, instead of in every consumer. */
        ingot_gguf_shape_row_major(t, o->shape);
        o->nelem = t->nelem;
        o->nbytes = t->nbytes;
        o->data = ingot_gguf_data(w->gguf, t);
        switch (t->type) {
        case INGOT_TYPE_F32:  o->dtype = INGOT_DT_F32;  break;
        case INGOT_TYPE_F16:  o->dtype = INGOT_DT_F16;  break;
        case INGOT_TYPE_BF16: o->dtype = INGOT_DT_BF16; break;
        case INGOT_TYPE_F64:  o->dtype = INGOT_DT_F64;  break;
        case INGOT_TYPE_I8:   o->dtype = INGOT_DT_I8;   break;
        case INGOT_TYPE_I16:  o->dtype = INGOT_DT_I16;  break;
        case INGOT_TYPE_I32:  o->dtype = INGOT_DT_I32;  break;
        case INGOT_TYPE_I64:  o->dtype = INGOT_DT_I64;  break;
        default:              o->dtype = INGOT_DT_UNKNOWN; break;  /* a block type */
        }
        if (o->data == NULL) {
            ingot_err(err, errsz, "tensor '%s' has no readable payload", t->name);
            return -1;
        }
    }
    return 0;
}

static int build_from_st(ingot_wfile *w, char *err, size_t errsz) {
    w->ntensor = ingot_st_count(w->st);
    w->tensors = (ingot_wtensor *)calloc(w->ntensor ? w->ntensor : 1, sizeof(*w->tensors));
    if (w->tensors == NULL) { ingot_err(err, errsz, "out of memory"); return -1; }
    for (size_t i = 0; i < w->ntensor; i++) {
        const ingot_st_tensor *t = ingot_st_at(w->st, i);
        ingot_wtensor *o = &w->tensors[i];
        o->name = t->name;
        o->dtype = t->dtype;
        o->ggml_type = -1;
        o->rank = t->rank;
        for (uint32_t d = 0; d < t->rank; d++) o->shape[d] = t->shape[d];
        o->nelem = t->nelem;
        o->nbytes = t->nbytes;
        o->data = ingot_st_data(w->st, t);
        if (o->data == NULL) {
            ingot_err(err, errsz, "tensor '%s' has no readable payload", t->name);
            return -1;
        }
    }
    return 0;
}

int ingot_wfile_open(ingot_wfile **out, const char *path, char *err, size_t errsz) {
    if (out == NULL || path == NULL) {
        ingot_err(err, errsz, "ingot_wfile_open: null argument");
        return -1;
    }
    *out = NULL;
    unsigned char magic[4] = {0};
    FILE *probe = fopen(path, "rb");
    if (probe != NULL) {
        if (fread(magic, 1, 4, probe) != 4) memset(magic, 0, sizeof magic);
        fclose(probe);
    }
    ingot_wfile *w = (ingot_wfile *)calloc(1, sizeof(*w));
    if (w == NULL) { ingot_err(err, errsz, "out of memory"); return -1; }

    if (memcmp(magic, "GGUF", 4) == 0) {
        w->container = INGOT_CONTAINER_GGUF;
        if (ingot_gguf_open_split(&w->gguf, path, err, errsz) != 0) goto fail;
        if (build_from_gguf(w, err, errsz) != 0) goto fail;
    } else {
        w->container = INGOT_CONTAINER_SAFETENSORS;
        if (ingot_st_open(&w->st, path, err, errsz) != 0) goto fail;
        if (build_from_st(w, err, errsz) != 0) goto fail;
    }
    if (ingot_index_build(&w->index, w->tensors, w->ntensor, w_name_of) != 0) {
        ingot_err(err, errsz, "out of memory building the name index");
        goto fail;
    }
    *out = w;
    return 0;
fail:
    ingot_wfile_close(w);
    return -1;
}

void ingot_wfile_close(ingot_wfile *w) {
    if (w == NULL) return;
    ingot_index_free(&w->index);
    free(w->tensors);
    ingot_gguf_close(w->gguf);
    ingot_st_close(w->st);
    free(w);
}

ingot_container ingot_wfile_container(const ingot_wfile *w) {
    return w != NULL ? w->container : INGOT_CONTAINER_GGUF;
}
size_t ingot_wfile_count(const ingot_wfile *w) { return w != NULL ? w->ntensor : 0; }

const ingot_wtensor *ingot_wfile_at(const ingot_wfile *w, size_t index) {
    return (w != NULL && index < w->ntensor) ? &w->tensors[index] : NULL;
}

const ingot_wtensor *ingot_wfile_find(const ingot_wfile *w, const char *name) {
    if (w == NULL) return NULL;
    const size_t i = ingot_index_find(&w->index, w->tensors, w_name_of, name);
    return i == (size_t)-1 ? NULL : &w->tensors[i];
}

int ingot_wfile_to_f32(const ingot_wfile *w, const ingot_wtensor *t, float *dst) {
    if (w == NULL || t == NULL || t->data == NULL || dst == NULL) return -1;
    if (t->nelem > SIZE_MAX) return -1;
    if (t->ggml_type >= 0) {
#ifdef INGOT_NO_KERNELS
        (void)dst;
        return -1;
#else
        return ingot_dequant(t->ggml_type, t->data, (size_t)t->nelem, dst);
#endif
    }
    return ingot_dtype_to_f32(t->dtype, t->data, (size_t)t->nelem, dst);
}

const ingot_gguf *ingot_wfile_gguf(const ingot_wfile *w) {
    return (w != NULL && w->container == INGOT_CONTAINER_GGUF) ? w->gguf : NULL;
}
ingot_st *ingot_wfile_st(const ingot_wfile *w) {
    return (w != NULL && w->container == INGOT_CONTAINER_SAFETENSORS) ? w->st : NULL;
}

/* ═══ src/write.c ═══ */
/* GGUF and safetensors writers.
 *
 * Containers only: which layers stay exact, which get quantized, whether a
 * LoRA is baked in — all of that is conversion policy and belongs to the
 * caller, not to a loader.
 *
 * SPDX-License-Identifier: MIT */
#include <sys/types.h>

#ifndef INGOT_NO_KERNELS
#endif

#include <stdio.h>

/* ── shared: a growable byte buffer ─────────────────────────────────────── */
typedef struct { unsigned char *p; size_t len, cap; } wbuf;

static int wbuf_put(wbuf *b, const void *src, size_t n) {
    if (b->len + n > b->cap) {
        size_t cap = b->cap ? b->cap * 2 : 1024;
        while (cap < b->len + n) cap *= 2;
        unsigned char *grown = (unsigned char *)realloc(b->p, cap);
        if (grown == NULL) return -1;
        b->p = grown;
        b->cap = cap;
    }
    memcpy(b->p + b->len, src, n);
    b->len += n;
    return 0;
}
static int wbuf_u32(wbuf *b, uint32_t v) {
    unsigned char x[4] = { (unsigned char)v, (unsigned char)(v >> 8),
                           (unsigned char)(v >> 16), (unsigned char)(v >> 24) };
    return wbuf_put(b, x, 4);
}
static int wbuf_u64(wbuf *b, uint64_t v) {
    unsigned char x[8];
    for (int i = 0; i < 8; i++) x[i] = (unsigned char)(v >> (8 * i));
    return wbuf_put(b, x, 8);
}
static int wbuf_str(wbuf *b, const char *s) {
    const size_t n = strlen(s);
    return wbuf_u64(b, n) != 0 ? -1 : wbuf_put(b, s, n);
}

/* ── GGUF ───────────────────────────────────────────────────────────────── */
#define GGUF_ALIGNMENT 32

typedef struct {
    char          *name;
    int            type;
    uint32_t       rank;
    uint64_t       ne[INGOT_MAX_RANK];
    uint64_t       nbytes;
    const void    *data;
    unsigned char *owned;     /* non-NULL when we quantized it ourselves */
} wtensor;

struct ingot_gguf_writer {
    wbuf      kv;             /* the serialized KV block, built as we go */
    uint64_t  nkv;
    int       alignment_kv;   /* general.alignment already appended (save() ran) */
    wtensor  *tensors;
    size_t    ntensor, cap;
    int       failed;
};

ingot_gguf_writer *ingot_gguf_writer_new(void) {
    return (ingot_gguf_writer *)calloc(1, sizeof(ingot_gguf_writer));
}

void ingot_gguf_writer_free(ingot_gguf_writer *w) {
    if (w == NULL) return;
    for (size_t i = 0; i < w->ntensor; i++) {
        free(w->tensors[i].name);
        free(w->tensors[i].owned);
    }
    free(w->tensors);
    free(w->kv.p);
    free(w);
}

/* Every KV goes in as key, type tag, payload — the order the spec fixes. A
 * failure anywhere latches `failed`, so a caller can chain twenty calls and
 * check once at save() instead of twenty times. */
static int kv_head(ingot_gguf_writer *w, const char *key, uint32_t type) {
    if (w == NULL || key == NULL) return -1;
    if (wbuf_str(&w->kv, key) != 0 || wbuf_u32(&w->kv, type) != 0) {
        w->failed = 1;
        return -1;
    }
    w->nkv++;
    return 0;
}

int ingot_gguf_kv_string(ingot_gguf_writer *w, const char *key, const char *value) {
    if (value == NULL || kv_head(w, key, INGOT_KV_STRING) != 0) return -1;
    if (wbuf_str(&w->kv, value) != 0) { w->failed = 1; return -1; }
    return 0;
}
int ingot_gguf_kv_u32(ingot_gguf_writer *w, const char *key, uint32_t value) {
    if (kv_head(w, key, INGOT_KV_UINT32) != 0) return -1;
    if (wbuf_u32(&w->kv, value) != 0) { w->failed = 1; return -1; }
    return 0;
}
int ingot_gguf_kv_u64(ingot_gguf_writer *w, const char *key, uint64_t value) {
    if (kv_head(w, key, INGOT_KV_UINT64) != 0) return -1;
    if (wbuf_u64(&w->kv, value) != 0) { w->failed = 1; return -1; }
    return 0;
}
int ingot_gguf_kv_i32(ingot_gguf_writer *w, const char *key, int32_t value) {
    if (kv_head(w, key, INGOT_KV_INT32) != 0) return -1;
    if (wbuf_u32(&w->kv, (uint32_t)value) != 0) { w->failed = 1; return -1; }
    return 0;
}
int ingot_gguf_kv_f32(ingot_gguf_writer *w, const char *key, float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof bits);
    if (kv_head(w, key, INGOT_KV_FLOAT32) != 0) return -1;
    if (wbuf_u32(&w->kv, bits) != 0) { w->failed = 1; return -1; }
    return 0;
}
int ingot_gguf_kv_bool(ingot_gguf_writer *w, const char *key, int value) {
    const unsigned char b = value ? 1 : 0;
    if (kv_head(w, key, INGOT_KV_BOOL) != 0) return -1;
    if (wbuf_put(&w->kv, &b, 1) != 0) { w->failed = 1; return -1; }
    return 0;
}

static int kv_array_head(ingot_gguf_writer *w, const char *key,
                         uint32_t elem, size_t count) {
    if (kv_head(w, key, INGOT_KV_ARRAY) != 0) return -1;
    if (wbuf_u32(&w->kv, elem) != 0 || wbuf_u64(&w->kv, count) != 0) {
        w->failed = 1;
        return -1;
    }
    return 0;
}

int ingot_gguf_kv_array_string(ingot_gguf_writer *w, const char *key,
                               const char *const *values, size_t count) {
    if (values == NULL || kv_array_head(w, key, INGOT_KV_STRING, count) != 0) return -1;
    for (size_t i = 0; i < count; i++) {
        if (values[i] == NULL || wbuf_str(&w->kv, values[i]) != 0) {
            w->failed = 1;
            return -1;
        }
    }
    return 0;
}
int ingot_gguf_kv_array_f32(ingot_gguf_writer *w, const char *key,
                            const float *values, size_t count) {
    if (values == NULL || kv_array_head(w, key, INGOT_KV_FLOAT32, count) != 0) return -1;
    for (size_t i = 0; i < count; i++) {
        uint32_t bits;
        memcpy(&bits, &values[i], sizeof bits);
        if (wbuf_u32(&w->kv, bits) != 0) { w->failed = 1; return -1; }
    }
    return 0;
}
int ingot_gguf_kv_array_i32(ingot_gguf_writer *w, const char *key,
                            const int32_t *values, size_t count) {
    if (values == NULL || kv_array_head(w, key, INGOT_KV_INT32, count) != 0) return -1;
    for (size_t i = 0; i < count; i++)
        if (wbuf_u32(&w->kv, (uint32_t)values[i]) != 0) { w->failed = 1; return -1; }
    return 0;
}

/* Returns NULL on both a rejected shape and a failed allocation, but only the
 * second latches `failed`. The distinction matters: a caller may legitimately
 * probe a type, get told no, and carry on writing the file — latching there
 * would poison a writer for asking a question. */
static wtensor *tensor_slot(ingot_gguf_writer *w, const char *name, int type,
                            uint32_t rank, const uint64_t *ne, uint64_t *nelem) {
    if (w == NULL || name == NULL || ne == NULL || rank == 0 || rank > INGOT_MAX_RANK)
        return NULL;
    uint64_t count = 1;
    for (uint32_t d = 0; d < rank; d++) {
        if (ne[d] == 0 || ingot_mul_u64(count, ne[d], &count) != 0) return NULL;
    }
    uint64_t bytes;
    if (ingot_type_nbytes(type, count, &bytes) != 0) return NULL;
    /* The same invariant the reader enforces: a block must fit whole inside
     * the fastest dimension, or every kernel reads the wrong scales. */
    uint64_t blk_elems, blk_bytes;
    if (ingot_type_geometry(type, &blk_elems, &blk_bytes) != 0) return NULL;
    if (blk_elems > 1 && ne[0] % blk_elems != 0) return NULL;

    if (w->ntensor == w->cap) {
        const size_t next = w->cap ? w->cap * 2 : 16;
        wtensor *grown = (wtensor *)realloc(w->tensors, next * sizeof(*grown));
        if (grown == NULL) { w->failed = 1; return NULL; }
        w->tensors = grown;
        w->cap = next;
    }
    wtensor *t = &w->tensors[w->ntensor];
    memset(t, 0, sizeof(*t));
    t->name = ingot_strdup(name);
    if (t->name == NULL) { w->failed = 1; return NULL; }
    t->type = type;
    t->rank = rank;
    for (uint32_t d = 0; d < rank; d++) t->ne[d] = ne[d];
    t->nbytes = bytes;
    if (nelem != NULL) *nelem = count;
    return t;
}

int ingot_gguf_add_tensor(ingot_gguf_writer *w, const char *name, int type,
                          uint32_t rank, const uint64_t *ne, const void *data) {
    if (data == NULL) return -1;
    wtensor *t = tensor_slot(w, name, type, rank, ne, NULL);
    if (t == NULL) return -1;
    t->data = data;
    w->ntensor++;
    return 0;
}

int ingot_gguf_add_f32(ingot_gguf_writer *w, const char *name, int type,
                       uint32_t rank, const uint64_t *ne, const float *values) {
#ifdef INGOT_NO_KERNELS
    (void)w; (void)name; (void)type; (void)rank; (void)ne; (void)values;
    return -1;
#else
    if (values == NULL) return -1;
    uint64_t nelem = 0;
    wtensor *t = tensor_slot(w, name, type, rank, ne, &nelem);
    if (t == NULL) return -1;
    t->owned = (unsigned char *)malloc((size_t)t->nbytes);
    if (t->owned == NULL) { free(t->name); w->failed = 1; return -1; }
    /* No encoder for this type is an answer, not a fault: the slot is rolled
     * back and the writer stays usable. */
    if (nelem > SIZE_MAX || ingot_quantize(type, values, (size_t)nelem, t->owned) != 0) {
        free(t->owned);
        free(t->name);
        memset(t, 0, sizeof(*t));
        return -1;
    }
    t->data = t->owned;
    w->ntensor++;
    return 0;
#endif
}

static int pad_file(FILE *f, uint64_t alignment) {
    static const unsigned char zero[GGUF_ALIGNMENT] = {0};
    const off_t pos = ftello(f);   /* ftell's long truncates past 2 GiB on LP32 */
    if (pos < 0) return -1;
    const size_t rem = (size_t)((uint64_t)pos % alignment);
    if (rem == 0) return 0;
    const size_t need = (size_t)(alignment - rem);
    return fwrite(zero, 1, need, f) == need ? 0 : -1;
}

int ingot_gguf_writer_save(ingot_gguf_writer *w, const char *path,
                           char *err, size_t errsz) {
    if (w == NULL || path == NULL) {
        ingot_err(err, errsz, "ingot_gguf_writer_save: null argument");
        return -1;
    }
    if (w->failed) {
        ingot_err(err, errsz, "the writer already failed (out of memory, or a "
                              "tensor whose shape does not fit its block type)");
        return -1;
    }
    /* The alignment is written as a KV so a reader does not have to assume the
     * default; we always use 32, which is the default, so old readers agree.
     * save() may legitimately run twice (retry after ENOSPC, two paths): the
     * KV must be appended exactly once or the header count lies. */
    if (!w->alignment_kv) {
        if (ingot_gguf_kv_u32(w, "general.alignment", GGUF_ALIGNMENT) != 0 || w->failed) {
            ingot_err(err, errsz, "out of memory");
            return -1;
        }
        w->alignment_kv = 1;
    }

    /* The tensor table needs each payload's offset, which depends on the table
     * length, which depends on the offsets only through their encoded width —
     * u64 always. So one pass to compute, one to write. */
    wbuf table = {0};
    uint64_t offset = 0;
    int ok = 1;
    for (size_t i = 0; i < w->ntensor && ok; i++) {
        const wtensor *t = &w->tensors[i];
        ok = wbuf_str(&table, t->name) == 0 && wbuf_u32(&table, t->rank) == 0;
        for (uint32_t d = 0; d < t->rank && ok; d++) ok = wbuf_u64(&table, t->ne[d]) == 0;
        ok = ok && wbuf_u32(&table, (uint32_t)t->type) == 0 &&
             wbuf_u64(&table, offset) == 0;
        offset += (t->nbytes + GGUF_ALIGNMENT - 1) & ~((uint64_t)GGUF_ALIGNMENT - 1);
    }
    if (!ok) {
        free(table.p);
        ingot_err(err, errsz, "out of memory building the tensor table");
        return -1;
    }

    FILE *f = fopen(path, "wb");
    if (f == NULL) {
        free(table.p);
        ingot_err(err, errsz, "cannot open '%s' for writing", path);
        return -1;
    }
    wbuf head = {0};
    ok = wbuf_u32(&head, 0x46554747u) == 0 &&      /* "GGUF" */
         wbuf_u32(&head, 3) == 0 &&
         wbuf_u64(&head, w->ntensor) == 0 &&
         wbuf_u64(&head, w->nkv) == 0;
    ok = ok && fwrite(head.p, 1, head.len, f) == head.len;
    ok = ok && (w->kv.len == 0 || fwrite(w->kv.p, 1, w->kv.len, f) == w->kv.len);
    ok = ok && (table.len == 0 || fwrite(table.p, 1, table.len, f) == table.len);
    ok = ok && pad_file(f, GGUF_ALIGNMENT) == 0;
    for (size_t i = 0; i < w->ntensor && ok; i++) {
        const wtensor *t = &w->tensors[i];
        ok = fwrite(t->data, 1, (size_t)t->nbytes, f) == (size_t)t->nbytes &&
             pad_file(f, GGUF_ALIGNMENT) == 0;
    }
    free(head.p);
    free(table.p);
    if (fclose(f) != 0) ok = 0;
    if (!ok) {
        ingot_err(err, errsz, "write to '%s' failed", path);
        return -1;
    }
    return 0;
}

/* ── safetensors ────────────────────────────────────────────────────────── */
typedef struct {
    char        *name;
    ingot_dtype  dtype;
    uint32_t     rank;
    uint64_t     shape[INGOT_MAX_RANK];
    uint64_t     nbytes;
    const void  *data;
} stentry;

struct ingot_st_writer {
    stentry *entries;
    size_t   count, cap;
    wbuf     metadata;        /* JSON fragments, already escaped */
    size_t   nmeta;
    int      failed;
};

ingot_st_writer *ingot_st_writer_new(void) {
    return (ingot_st_writer *)calloc(1, sizeof(ingot_st_writer));
}

void ingot_st_writer_free(ingot_st_writer *w) {
    if (w == NULL) return;
    for (size_t i = 0; i < w->count; i++) free(w->entries[i].name);
    free(w->entries);
    free(w->metadata.p);
    free(w);
}

/* JSON string escaping. Tensor names really do contain characters that need
 * it, and a writer that emits invalid JSON produces a file no reader can open
 * — including this one, which is how it would be found. */
static int json_string(wbuf *b, const char *s) {
    if (wbuf_put(b, "\"", 1) != 0) return -1;
    for (const unsigned char *p = (const unsigned char *)s; *p != 0; p++) {
        char esc[8];
        const char *out = NULL;
        size_t n = 1;
        switch (*p) {
        case '"':  out = "\\\""; n = 2; break;
        case '\\': out = "\\\\"; n = 2; break;
        case '\n': out = "\\n";  n = 2; break;
        case '\r': out = "\\r";  n = 2; break;
        case '\t': out = "\\t";  n = 2; break;
        default:
            if (*p < 0x20) {
                snprintf(esc, sizeof esc, "\\u%04x", *p);
                out = esc;
                n = 6;
            } else {
                esc[0] = (char)*p;
                out = esc;
                n = 1;
            }
        }
        if (wbuf_put(b, out, n) != 0) return -1;
    }
    return wbuf_put(b, "\"", 1);
}

int ingot_st_writer_metadata(ingot_st_writer *w, const char *key, const char *value) {
    if (w == NULL || key == NULL || value == NULL) return -1;
    if (w->nmeta != 0 && wbuf_put(&w->metadata, ",", 1) != 0) { w->failed = 1; return -1; }
    if (json_string(&w->metadata, key) != 0 || wbuf_put(&w->metadata, ":", 1) != 0 ||
        json_string(&w->metadata, value) != 0) {
        w->failed = 1;
        return -1;
    }
    w->nmeta++;
    return 0;
}

int ingot_st_writer_add(ingot_st_writer *w, const char *name, ingot_dtype dtype,
                        uint32_t rank, const uint64_t *shape, const void *data) {
    if (w == NULL || name == NULL || shape == NULL || data == NULL ||
        rank == 0 || rank > INGOT_MAX_RANK) return -1;
    /* An unknown dtype or an impossible shape is a rejected argument, not a
     * broken writer: return -1 and leave the writer usable. Only allocation
     * failures below latch. */
    const size_t item = ingot_dtype_size(dtype);
    if (item == 0) return -1;
    uint64_t nelem = 1;
    for (uint32_t d = 0; d < rank; d++)
        if (shape[d] == 0 || ingot_mul_u64(nelem, shape[d], &nelem) != 0) return -1;
    uint64_t bytes;
    if (ingot_mul_u64(nelem, item, &bytes) != 0) return -1;

    if (w->count == w->cap) {
        const size_t next = w->cap ? w->cap * 2 : 16;
        stentry *grown = (stentry *)realloc(w->entries, next * sizeof(*grown));
        if (grown == NULL) { w->failed = 1; return -1; }
        w->entries = grown;
        w->cap = next;
    }
    stentry *e = &w->entries[w->count];
    memset(e, 0, sizeof(*e));
    e->name = ingot_strdup(name);
    if (e->name == NULL) { w->failed = 1; return -1; }
    e->dtype = dtype;
    e->rank = rank;
    for (uint32_t d = 0; d < rank; d++) e->shape[d] = shape[d];
    e->nbytes = bytes;
    e->data = data;
    w->count++;
    return 0;
}

int ingot_st_writer_save(ingot_st_writer *w, const char *path,
                         char *err, size_t errsz) {
    if (w == NULL || path == NULL) {
        ingot_err(err, errsz, "ingot_st_writer_save: null argument");
        return -1;
    }
    if (w->failed) {
        ingot_err(err, errsz, "the writer already failed (out of memory, or an "
                              "unsupported dtype)");
        return -1;
    }
    wbuf json = {0};
    int ok = wbuf_put(&json, "{", 1) == 0;
    if (w->nmeta != 0) {
        ok = ok && wbuf_put(&json, "\"__metadata__\":{", 16) == 0 &&
             wbuf_put(&json, w->metadata.p, w->metadata.len) == 0 &&
             wbuf_put(&json, "}", 1) == 0;
    }
    uint64_t offset = 0;
    for (size_t i = 0; i < w->count && ok; i++) {
        const stentry *e = &w->entries[i];
        char numbers[64];
        if (i != 0 || w->nmeta != 0) ok = wbuf_put(&json, ",", 1) == 0;
        ok = ok && json_string(&json, e->name) == 0 &&
             wbuf_put(&json, ":{\"dtype\":\"", 11) == 0 &&
             wbuf_put(&json, ingot_dtype_name(e->dtype),
                      strlen(ingot_dtype_name(e->dtype))) == 0 &&
             wbuf_put(&json, "\",\"shape\":[", 11) == 0;
        for (uint32_t d = 0; d < e->rank && ok; d++) {
            const int n = snprintf(numbers, sizeof numbers, "%s%llu",
                                   d ? "," : "", (unsigned long long)e->shape[d]);
            ok = n > 0 && wbuf_put(&json, numbers, (size_t)n) == 0;
        }
        const int n = snprintf(numbers, sizeof numbers, "],\"data_offsets\":[%llu,%llu]}",
                               (unsigned long long)offset,
                               (unsigned long long)(offset + e->nbytes));
        ok = ok && n > 0 && wbuf_put(&json, numbers, (size_t)n) == 0;
        offset += e->nbytes;
    }
    ok = ok && wbuf_put(&json, "}", 1) == 0;
    /* Pad so that 8 (the length prefix) + header is a multiple of 8: every
     * zero-copy typed read downstream depends on it. */
    const size_t pad = (8 - ((8 + json.len) % 8)) % 8;
    for (size_t i = 0; i < pad && ok; i++) ok = wbuf_put(&json, " ", 1) == 0;
    if (!ok) {
        free(json.p);
        ingot_err(err, errsz, "out of memory building the safetensors header");
        return -1;
    }

    FILE *f = fopen(path, "wb");
    if (f == NULL) {
        free(json.p);
        ingot_err(err, errsz, "cannot open '%s' for writing", path);
        return -1;
    }
    unsigned char prefix[8];
    for (int i = 0; i < 8; i++) prefix[i] = (unsigned char)((uint64_t)json.len >> (8 * i));
    ok = fwrite(prefix, 1, 8, f) == 8 && fwrite(json.p, 1, json.len, f) == json.len;
    for (size_t i = 0; i < w->count && ok; i++)
        ok = w->entries[i].nbytes == 0 ||
             fwrite(w->entries[i].data, 1, (size_t)w->entries[i].nbytes, f) ==
                 (size_t)w->entries[i].nbytes;
    free(json.p);
    if (fclose(f) != 0) ok = 0;
    if (!ok) {
        ingot_err(err, errsz, "write to '%s' failed", path);
        return -1;
    }
    return 0;
}


#ifndef INGOT_NO_KERNELS
/* ─── the optional quantization half ─────────────────────────────── */

/* ═══ src/cpu.c ═══ */
/* CPU feature detection and the thread-injection point.
 *
 * The usual two ways of getting this wrong are picking the SIMD level at BUILD
 * time, through -D flags a Makefile has to remember to pass, and doing proper
 * runtime detection on one architecture only. Here the "compiled" half comes
 * from the compiler's own predefined macros — so there is no build flag to
 * forget — and the "runtime" half is checked on every platform.
 *
 * SPDX-License-Identifier: MIT */
/* posix_memalign, below. glibc hides it under -std=c11 without this, and an
 * implicitly declared function is an error on current compilers and undefined
 * behaviour on the ones that let it through. macOS declares it either way,
 * which is why only Linux ever complained. */
#if defined(__APPLE__)
/* And <sys/sysctl.h> is outside the strict POSIX namespace on Darwin, so the
 * line above alone breaks the SDK headers. Same pair as in safetensors.c. */
#endif


#include <pthread.h>
#include <stdlib.h>

#if defined(__APPLE__)
#include <sys/sysctl.h>
#elif defined(__linux__) && defined(__aarch64__)
#include <sys/auxv.h>
#include <asm/hwcap.h>
#endif
#if defined(__x86_64__) || defined(__i386__)
#if defined(__GNUC__) || defined(__clang__)
#include <cpuid.h>
#endif
#endif

/* What this translation unit was actually compiled with. */
#if defined(__ARM_NEON) || defined(__aarch64__)
#define INGOT_COMPILED_NEON 1
#endif
#if defined(__ARM_FEATURE_DOTPROD)
#define INGOT_COMPILED_DOTPROD 1
#endif
#if defined(__ARM_FEATURE_MATMUL_INT8)
#define INGOT_COMPILED_I8MM 1
#endif
#if defined(__AVX2__)
#define INGOT_COMPILED_AVX2 1
#endif
#if defined(__AVX512F__)
#define INGOT_COMPILED_AVX512 1
#endif
#if defined(__AVX512VNNI__)
#define INGOT_COMPILED_AVX512_VNNI 1
#endif
#if defined(__F16C__)
#define INGOT_COMPILED_F16C 1
#endif
#if defined(__AVX512BF16__)
#define INGOT_COMPILED_AVX512_BF16 1
#endif
#if defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC) || defined(__ARM_FEATURE_BF16)
#define INGOT_COMPILED_ARM_BF16 1
#endif

static ingot_cpu_caps cached;
static int cap_level_cap = -1;          /* -1 = auto */
static pthread_once_t caps_once = PTHREAD_ONCE_INIT;

/* Name to cap level, with no locking of any kind: init_caps runs INSIDE
 * pthread_once and must never re-enter it. -2 means the name is not one we
 * know. */
#define INGOT_LEVEL_UNKNOWN (-2)
static int level_from_name(const char *name) {
    if (name == NULL || strcmp(name, "auto") == 0) return -1;
    if (strcmp(name, "scalar") == 0) return 0;
    if (strcmp(name, "avx2") == 0 || strcmp(name, "neon") == 0) return 1;
    if (strcmp(name, "vnni") == 0 || strcmp(name, "dotprod") == 0) return 2;
    return INGOT_LEVEL_UNKNOWN;
}

/* The effective level of `caps` once `cap_level_cap` is applied. */
static int level_of(ingot_cpu_caps caps) {
    if (caps.avx512_vnni || caps.dotprod) return 2;
    if (caps.avx2 || caps.neon) return 1;
    return 0;
}

static ingot_cpu_caps capped(ingot_cpu_caps caps) {
    if (cap_level_cap >= 0) {
        if (cap_level_cap < 2) {
            caps.avx512_vnni = 0; caps.i8mm = 0; caps.dotprod = 0;
            caps.bf16 = 0; caps.avx512_bf16 = 0;
        }
        if (cap_level_cap < 1) {
            caps.avx2 = 0; caps.avx512 = 0; caps.neon = 0;
            caps.f16c = 0;
        }
    }
    return caps;
}

#if defined(__APPLE__) && defined(__aarch64__)
static int sysctl_flag(const char *name) {
    int value = 0;
    size_t size = sizeof(value);
    return sysctlbyname(name, &value, &size, NULL, 0) == 0 && value != 0;
}
#endif

#if defined(__x86_64__) || defined(__i386__)
static unsigned long long xgetbv0(void) {
    unsigned int eax, edx;
    __asm__ __volatile__(".byte 0x0f, 0x01, 0xd0" : "=a"(eax), "=d"(edx) : "c"(0));
    return ((unsigned long long)edx << 32) | eax;
}
#endif

static void init_caps(void) {
    memset(&cached, 0, sizeof(cached));

#if defined(INGOT_COMPILED_NEON)
    cached.neon = 1;                     /* NEON is baseline on aarch64 */
#endif
#if defined(INGOT_COMPILED_DOTPROD)
#if defined(__APPLE__)
    cached.dotprod = sysctl_flag("hw.optional.arm.FEAT_DotProd");
#elif defined(__linux__) && defined(__aarch64__) && defined(HWCAP_ASIMDDP)
    cached.dotprod = (getauxval(AT_HWCAP) & HWCAP_ASIMDDP) != 0;
#else
    cached.dotprod = 1;                  /* compiled for it, cannot ask: trust it */
#endif
#endif
#if defined(INGOT_COMPILED_I8MM)
#if defined(__APPLE__)
    cached.i8mm = sysctl_flag("hw.optional.arm.FEAT_I8MM");
#elif defined(__linux__) && defined(__aarch64__) && defined(HWCAP2_I8MM)
    cached.i8mm = (getauxval(AT_HWCAP2) & HWCAP2_I8MM) != 0;
#else
    cached.i8mm = 1;
#endif
#endif
#if defined(INGOT_COMPILED_ARM_BF16)
#if defined(__APPLE__)
    cached.bf16 = sysctl_flag("hw.optional.arm.FEAT_BF16");
#elif defined(__linux__) && defined(__aarch64__) && defined(HWCAP2_BF16)
    cached.bf16 = (getauxval(AT_HWCAP2) & HWCAP2_BF16) != 0;
#else
    cached.bf16 = 1;
#endif
#endif

#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
    {
        unsigned int a = 0, b = 0, c = 0, d = 0;
        int os_ymm = 0, os_zmm = 0;
        if (__get_cpuid(1, &a, &b, &c, &d) && ((c >> 27) & 1)) {   /* OSXSAVE */
            const unsigned long long xcr0 = xgetbv0();
            os_ymm = (xcr0 & 0x6) == 0x6;                          /* XMM|YMM  */
            os_zmm = os_ymm && (xcr0 & 0xe0) == 0xe0;              /* + ZMM    */
#if defined(INGOT_COMPILED_F16C)
            cached.f16c = os_ymm && ((c >> 29) & 1);               /* leaf 1 ECX */
#endif
        }
#if !defined(INGOT_COMPILED_AVX512) && !defined(INGOT_COMPILED_AVX512_VNNI) && \
    !defined(INGOT_COMPILED_AVX512_BF16)
        (void)os_zmm;   /* only the AVX-512 checks below read it */
#endif
        if (__get_cpuid_count(7, 0, &a, &b, &c, &d)) {
#if defined(INGOT_COMPILED_AVX2)
            cached.avx2 = os_ymm && ((b >> 5) & 1);
#endif
#if defined(INGOT_COMPILED_AVX512)
            cached.avx512 = os_zmm && ((b >> 16) & 1);
#endif
#if defined(INGOT_COMPILED_AVX512_VNNI)
            cached.avx512_vnni = os_zmm && ((c >> 11) & 1);
#endif
        }
#if defined(INGOT_COMPILED_AVX512_BF16)
        /* AVX512-BF16 lives in leaf 7 SUBLEAF 1 (EAX bit 5), not subleaf 0. */
        if (__get_cpuid_count(7, 1, &a, &b, &c, &d))
            cached.avx512_bf16 = os_zmm && ((a >> 5) & 1);
#endif
    }
#endif

    /* INGOT_CAPS caps the level without a rebuild, which is how you bisect a
     * bug that only shows up on one SIMD path. Parsed inline, NOT through
     * ingot_cpu_set_level(): we are inside pthread_once here, and that function
     * enters it again — recursing on the same pthread_once_t is undefined and
     * killed the process outright. An unknown name is ignored rather than
     * fatal: an environment variable must not be able to stop a library. */
    const char *env = getenv("INGOT_CAPS");
    if (env != NULL) {
        const int level = level_from_name(env);
        if (level != INGOT_LEVEL_UNKNOWN) cap_level_cap = level;
    }

    /* INGOT_CAPS_ASSUME goes the other way: trust the BUILD instead of CPUID.
     *
     * It exists because CPUID is not always telling the truth. Rosetta 2
     * executes AVX2 but does not advertise it in leaf 7, so on an Apple
     * Silicon machine — which for many of us is the only place x86 code can be
     * run at all — every AVX2 kernel in this library silently falls back to
     * scalar. They are then neither tested nor measured, and an untested
     * kernel is the one that is wrong.
     *
     * This can only switch on what the compiler was already told to emit, so
     * it cannot conjure an instruction the binary does not contain. It CAN
     * fault on a machine that genuinely lacks the feature, which is why it is
     * opt-in, off by default, and a testing tool rather than a tuning knob.
     *
     * AVX-512 is deliberately not forceable: it additionally needs the OS to
     * have enabled ZMM state, and that is not something a build flag knows. */
    const char *assume = getenv("INGOT_CAPS_ASSUME");
    if (assume != NULL) {
        const int level = level_from_name(assume);
        if (level >= 1) {
#if defined(INGOT_COMPILED_NEON)
            cached.neon = 1;
#endif
#if defined(INGOT_COMPILED_AVX2)
            cached.avx2 = 1;
#endif
#if defined(INGOT_COMPILED_F16C)
            cached.f16c = 1;
#endif
        }
        if (level >= 2) {
#if defined(INGOT_COMPILED_DOTPROD)
            cached.dotprod = 1;
#endif
#if defined(INGOT_COMPILED_I8MM)
            cached.i8mm = 1;
#endif
        }
    }
}

ingot_cpu_caps ingot_cpu(void) {
    pthread_once(&caps_once, init_caps);
    return capped(cached);
}

int ingot_cpu_set_level(const char *name) {
    pthread_once(&caps_once, init_caps);
    const int level = level_from_name(name);
    if (level == INGOT_LEVEL_UNKNOWN) return -1;
    cap_level_cap = level;
    return level_of(capped(cached));
}

/* ── thread injection ───────────────────────────────────────────────────── */
static void parallel_for_serial(size_t count, ingot_range_fn fn, void *user) {
    if (count != 0) fn(0, count, user);
}

static ingot_parallel_for_fn parallel_for_impl = parallel_for_serial;

void ingot_set_parallel_for(ingot_parallel_for_fn fn) {
    parallel_for_impl = (fn != NULL) ? fn : parallel_for_serial;
}

void ingot_parallel_for(size_t count, ingot_range_fn fn, void *user);
void ingot_parallel_for(size_t count, ingot_range_fn fn, void *user) {
    parallel_for_impl(count, fn, user);
}

/* ── aligned allocation ─────────────────────────────────────────────────── */
void *ingot_aligned_alloc(size_t alignment, size_t size);
void  ingot_aligned_free(void *ptr);

void *ingot_aligned_alloc(size_t alignment, size_t size) {
    if (alignment == 0 || (alignment & (alignment - 1)) != 0) return NULL;
    if (alignment < sizeof(void *)) alignment = sizeof(void *);
    void *ptr = NULL;
    const size_t padded = (size + alignment - 1) & ~(alignment - 1);
    if (posix_memalign(&ptr, alignment, padded == 0 ? alignment : padded) != 0) return NULL;
    return ptr;
}

void ingot_aligned_free(void *ptr) { free(ptr); }

/* ═══ src/dequant.c ═══ */
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

/* ═══ src/dequant_iq.c ═══ */
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

#include <math.h>

/* ─── inlined iq_tables.inc ─── */
/* GENERATED by tools/gen_iq_tables.py from llama.cpp's `gguf` package.
 * Do not edit: regenerate instead. See the script for why these are
 * extracted rather than typed.
 * SPDX-License-Identifier: MIT */

/* IQ1_S: 2048 entries x 8 values, map (-1, 0, 1) */
static const int8_t ingot_iq1s_grid[16384] = {
      -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,    1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
       0,    0,   -1,   -1,   -1,   -1,   -1,   -1,   -1,    1,   -1,   -1,   -1,   -1,   -1,   -1,
       1,    1,   -1,   -1,   -1,   -1,   -1,   -1,    0,   -1,    0,   -1,   -1,   -1,   -1,   -1,
       0,    0,    0,   -1,   -1,   -1,   -1,   -1,   -1,   -1,    1,   -1,   -1,   -1,   -1,   -1,
       1,   -1,    1,   -1,   -1,   -1,   -1,   -1,   -1,    1,    1,   -1,   -1,   -1,   -1,   -1,
       1,    1,    1,   -1,   -1,   -1,   -1,   -1,    0,    0,   -1,    0,   -1,   -1,   -1,   -1,
       0,   -1,    0,    0,   -1,   -1,   -1,   -1,   -1,    0,    0,    0,   -1,   -1,   -1,   -1,
       1,    0,    0,    0,   -1,   -1,   -1,   -1,    0,    0,    1,    0,   -1,   -1,   -1,   -1,
      -1,   -1,   -1,    1,   -1,   -1,   -1,   -1,    1,   -1,   -1,    1,   -1,   -1,   -1,   -1,
      -1,    1,   -1,    1,   -1,   -1,   -1,   -1,    1,    1,   -1,    1,   -1,   -1,   -1,   -1,
       0,    0,    0,    1,   -1,   -1,   -1,   -1,   -1,   -1,    1,    1,   -1,   -1,   -1,   -1,
       1,   -1,    1,    1,   -1,   -1,   -1,   -1,   -1,    1,    1,    1,   -1,   -1,   -1,   -1,
       1,    1,    1,    1,   -1,   -1,   -1,   -1,   -1,    0,   -1,   -1,    0,   -1,   -1,   -1,
       0,    0,   -1,   -1,    0,   -1,   -1,   -1,    0,   -1,    0,   -1,    0,   -1,   -1,   -1,
      -1,    0,    0,   -1,    0,   -1,   -1,   -1,    1,    0,    0,   -1,    0,   -1,   -1,   -1,
       0,    1,    0,   -1,    0,   -1,   -1,   -1,    1,    1,    0,   -1,    0,   -1,   -1,   -1,
       0,    0,    1,   -1,    0,   -1,   -1,   -1,    0,   -1,   -1,    0,    0,   -1,   -1,   -1,
       1,    0,   -1,    0,    0,   -1,   -1,   -1,    0,    1,   -1,    0,    0,   -1,   -1,   -1,
       1,   -1,    0,    0,    0,   -1,   -1,   -1,    0,    0,    0,    0,    0,   -1,   -1,   -1,
       1,    1,    0,    0,    0,   -1,   -1,   -1,    0,   -1,    1,    0,    0,   -1,   -1,   -1,
      -1,    0,    1,    0,    0,   -1,   -1,   -1,    1,    0,    1,    0,    0,   -1,   -1,   -1,
      -1,    1,    1,    0,    0,   -1,   -1,   -1,    0,    0,   -1,    1,    0,   -1,   -1,   -1,
       0,   -1,    0,    1,    0,   -1,   -1,   -1,   -1,    0,    0,    1,    0,   -1,   -1,   -1,
       1,    0,    0,    1,    0,   -1,   -1,   -1,    0,    0,    1,    1,    0,   -1,   -1,   -1,
      -1,   -1,   -1,   -1,    1,   -1,   -1,   -1,    1,   -1,   -1,   -1,    1,   -1,   -1,   -1,
      -1,    1,   -1,   -1,    1,   -1,   -1,   -1,    1,    1,   -1,   -1,    1,   -1,   -1,   -1,
       0,    0,    0,   -1,    1,   -1,   -1,   -1,   -1,   -1,    1,   -1,    1,   -1,   -1,   -1,
       1,   -1,    1,   -1,    1,   -1,   -1,   -1,   -1,    1,    1,   -1,    1,   -1,   -1,   -1,
       1,    1,    1,   -1,    1,   -1,   -1,   -1,    0,    0,   -1,    0,    1,   -1,   -1,   -1,
       0,   -1,    0,    0,    1,   -1,   -1,   -1,    0,    1,    0,    0,    1,   -1,   -1,   -1,
      -1,    0,    1,    0,    1,   -1,   -1,   -1,    0,    1,    1,    0,    1,   -1,   -1,   -1,
      -1,   -1,   -1,    1,    1,   -1,   -1,   -1,    1,   -1,   -1,    1,    1,   -1,   -1,   -1,
      -1,    1,   -1,    1,    1,   -1,   -1,   -1,    1,    1,   -1,    1,    1,   -1,   -1,   -1,
       0,   -1,    0,    1,    1,   -1,   -1,   -1,    0,    0,    0,    1,    1,   -1,   -1,   -1,
       0,    1,    0,    1,    1,   -1,   -1,   -1,   -1,   -1,    1,    1,    1,   -1,   -1,   -1,
       1,   -1,    1,    1,    1,   -1,   -1,   -1,   -1,    1,    1,    1,    1,   -1,   -1,   -1,
       1,    1,    1,    1,    1,   -1,   -1,   -1,    0,   -1,    0,   -1,   -1,    0,   -1,   -1,
      -1,    0,    0,   -1,   -1,    0,   -1,   -1,    1,    0,    0,   -1,   -1,    0,   -1,   -1,
       0,    0,    1,   -1,   -1,    0,   -1,   -1,    0,   -1,   -1,    0,   -1,    0,   -1,   -1,
       0,    1,   -1,    0,   -1,    0,   -1,   -1,    0,    0,    0,    0,   -1,    0,   -1,   -1,
       1,    1,    0,    0,   -1,    0,   -1,   -1,   -1,    0,    1,    0,   -1,    0,   -1,   -1,
       0,    0,    1,    0,   -1,    0,   -1,   -1,    0,   -1,    0,    1,   -1,    0,   -1,   -1,
       0,    1,    0,    1,   -1,    0,   -1,   -1,    0,    0,    1,    1,   -1,    0,   -1,   -1,
       0,   -1,   -1,   -1,    0,    0,   -1,   -1,   -1,    0,   -1,   -1,    0,    0,   -1,   -1,
       0,    0,   -1,   -1,    0,    0,   -1,   -1,    1,    0,   -1,   -1,    0,    0,   -1,   -1,
       0,    0,    0,   -1,    0,    0,   -1,   -1,   -1,    1,    0,   -1,    0,    0,   -1,   -1,
       1,    1,    0,   -1,    0,    0,   -1,   -1,    0,    1,    1,   -1,    0,    0,   -1,   -1,
      -1,   -1,   -1,    0,    0,    0,   -1,   -1,    0,    0,   -1,    0,    0,    0,   -1,   -1,
       1,    1,   -1,    0,    0,    0,   -1,   -1,   -1,   -1,    0,    0,    0,    0,   -1,   -1,
       0,   -1,    0,    0,    0,    0,   -1,   -1,   -1,    0,    0,    0,    0,    0,   -1,   -1,
       0,    0,    0,    0,    0,    0,   -1,   -1,    1,    0,    0,    0,    0,    0,   -1,   -1,
       0,    1,    0,    0,    0,    0,   -1,   -1,   -1,   -1,    1,    0,    0,    0,   -1,   -1,
       1,   -1,    1,    0,    0,    0,   -1,   -1,    0,    0,    1,    0,    0,    0,   -1,   -1,
      -1,    1,    1,    0,    0,    0,   -1,   -1,    1,    1,    1,    0,    0,    0,   -1,   -1,
       0,   -1,   -1,    1,    0,    0,   -1,   -1,    0,   -1,    0,    1,    0,    0,   -1,   -1,
       0,    0,    0,    1,    0,    0,   -1,   -1,   -1,    1,    0,    1,    0,    0,   -1,   -1,
       1,    1,    0,    1,    0,    0,   -1,   -1,    0,   -1,    1,    1,    0,    0,   -1,   -1,
      -1,    0,    1,    1,    0,    0,   -1,   -1,    0,    0,    1,    1,    0,    0,   -1,   -1,
       1,    0,    1,    1,    0,    0,   -1,   -1,    0,    1,    1,    1,    0,    0,   -1,   -1,
      -1,    0,    0,   -1,    1,    0,   -1,   -1,    0,    1,    0,   -1,    1,    0,   -1,   -1,
       0,   -1,   -1,    0,    1,    0,   -1,   -1,   -1,    0,   -1,    0,    1,    0,   -1,   -1,
      -1,   -1,    0,    0,    1,    0,   -1,   -1,    1,   -1,    0,    0,    1,    0,   -1,   -1,
       0,    0,    0,    0,    1,    0,   -1,   -1,   -1,    1,    0,    0,    1,    0,   -1,   -1,
      -1,   -1,    1,    0,    1,    0,   -1,   -1,    0,   -1,    1,    0,    1,    0,   -1,   -1,
       1,    0,    1,    0,    1,    0,   -1,   -1,    0,    1,    1,    0,    1,    0,   -1,   -1,
       0,    0,   -1,    1,    1,    0,   -1,   -1,    0,   -1,    0,    1,    1,    0,   -1,   -1,
      -1,    0,    0,    1,    1,    0,   -1,   -1,    0,    1,    0,    1,    1,    0,   -1,   -1,
      -1,   -1,   -1,   -1,   -1,    1,   -1,   -1,    1,   -1,   -1,   -1,   -1,    1,   -1,   -1,
      -1,    1,   -1,   -1,   -1,    1,   -1,   -1,    1,    1,   -1,   -1,   -1,    1,   -1,   -1,
       0,    0,    0,   -1,   -1,    1,   -1,   -1,   -1,   -1,    1,   -1,   -1,    1,   -1,   -1,
       1,   -1,    1,   -1,   -1,    1,   -1,   -1,   -1,    1,    1,   -1,   -1,    1,   -1,   -1,
       1,    1,    1,   -1,   -1,    1,   -1,   -1,    0,    0,   -1,    0,   -1,    1,   -1,   -1,
       0,   -1,    0,    0,   -1,    1,   -1,   -1,    1,    0,    0,    0,   -1,    1,   -1,   -1,
       0,    0,    1,    0,   -1,    1,   -1,   -1,   -1,   -1,   -1,    1,   -1,    1,   -1,   -1,
       1,   -1,   -1,    1,   -1,    1,   -1,   -1,   -1,    1,   -1,    1,   -1,    1,   -1,   -1,
       1,    1,   -1,    1,   -1,    1,   -1,   -1,    0,    0,    0,    1,   -1,    1,   -1,   -1,
      -1,   -1,    1,    1,   -1,    1,   -1,   -1,    1,   -1,    1,    1,   -1,    1,   -1,   -1,
      -1,    1,    1,    1,   -1,    1,   -1,   -1,    1,    1,    1,    1,   -1,    1,   -1,   -1,
       0,    0,   -1,   -1,    0,    1,   -1,   -1,    0,   -1,    0,   -1,    0,    1,   -1,   -1,
      -1,    0,    0,   -1,    0,    1,   -1,   -1,    0,    1,    0,   -1,    0,    1,   -1,   -1,
      -1,    0,    1,   -1,    0,    1,   -1,   -1,    0,    0,    1,   -1,    0,    1,   -1,   -1,
       0,   -1,   -1,    0,    0,    1,   -1,   -1,   -1,   -1,    0,    0,    0,    1,   -1,   -1,
       0,   -1,    0,    0,    0,    1,   -1,   -1,    0,    0,    0,    0,    0,    1,   -1,   -1,
       0,   -1,    1,    0,    0,    1,   -1,   -1,   -1,    0,    1,    0,    0,    1,   -1,   -1,
       0,    1,    1,    0,    0,    1,   -1,   -1,    0,   -1,    0,    1,    0,    1,   -1,   -1,
      -1,    0,    0,    1,    0,    1,   -1,   -1,    1,    0,    0,    1,    0,    1,   -1,   -1,
       0,    1,    0,    1,    0,    1,   -1,   -1,    0,    0,    1,    1,    0,    1,   -1,   -1,
      -1,   -1,   -1,   -1,    1,    1,   -1,   -1,    1,   -1,   -1,   -1,    1,    1,   -1,   -1,
      -1,    1,   -1,   -1,    1,    1,   -1,   -1,    1,    1,   -1,   -1,    1,    1,   -1,   -1,
       0,    0,    0,   -1,    1,    1,   -1,   -1,   -1,   -1,    1,   -1,    1,    1,   -1,   -1,
       1,   -1,    1,   -1,    1,    1,   -1,   -1,   -1,    1,    1,   -1,    1,    1,   -1,   -1,
       1,    1,    1,   -1,    1,    1,   -1,   -1,    0,    0,   -1,    0,    1,    1,   -1,   -1,
       0,   -1,    0,    0,    1,    1,   -1,   -1,    0,    1,    0,    0,    1,    1,   -1,   -1,
       0,   -1,    1,    0,    1,    1,   -1,   -1,    0,    0,    1,    0,    1,    1,   -1,   -1,
      -1,   -1,   -1,    1,    1,    1,   -1,   -1,    1,   -1,   -1,    1,    1,    1,   -1,   -1,
       0,    0,   -1,    1,    1,    1,   -1,   -1,   -1,    1,   -1,    1,    1,    1,   -1,   -1,
       1,    1,   -1,    1,    1,    1,   -1,   -1,    0,    0,    0,    1,    1,    1,   -1,   -1,
      -1,   -1,    1,    1,    1,    1,   -1,   -1,    1,   -1,    1,    1,    1,    1,   -1,   -1,
      -1,    1,    1,    1,    1,    1,   -1,   -1,    1,    1,    1,    1,    1,    1,   -1,   -1,
      -1,   -1,    0,   -1,   -1,   -1,    0,   -1,    0,   -1,    0,   -1,   -1,   -1,    0,   -1,
      -1,    0,    0,   -1,   -1,   -1,    0,   -1,    0,    1,    0,   -1,   -1,   -1,    0,   -1,
      -1,    0,    1,   -1,   -1,   -1,    0,   -1,    0,    0,    1,   -1,   -1,   -1,    0,   -1,
       0,   -1,   -1,    0,   -1,   -1,    0,   -1,   -1,    0,   -1,    0,   -1,   -1,    0,   -1,
      -1,   -1,    0,    0,   -1,   -1,    0,   -1,    0,    0,    0,    0,   -1,   -1,    0,   -1,
      -1,    1,    0,    0,   -1,   -1,    0,   -1,    0,   -1,    1,    0,   -1,   -1,    0,   -1,
      -1,    0,    1,    0,   -1,   -1,    0,   -1,    0,    0,    1,    0,   -1,   -1,    0,   -1,
       0,    1,    1,    0,   -1,   -1,    0,   -1,    0,   -1,    0,    1,   -1,   -1,    0,   -1,
      -1,    0,    0,    1,   -1,   -1,    0,   -1,    1,    0,    0,    1,   -1,   -1,    0,   -1,
       0,   -1,    1,    1,   -1,   -1,    0,   -1,    0,    0,    1,    1,   -1,   -1,    0,   -1,
       0,   -1,   -1,   -1,    0,   -1,    0,   -1,   -1,    0,   -1,   -1,    0,   -1,    0,   -1,
       1,    0,   -1,   -1,    0,   -1,    0,   -1,    0,    1,   -1,   -1,    0,   -1,    0,   -1,
      -1,   -1,    0,   -1,    0,   -1,    0,   -1,    1,   -1,    0,   -1,    0,   -1,    0,   -1,
       0,    0,    0,   -1,    0,   -1,    0,   -1,   -1,    1,    0,   -1,    0,   -1,    0,   -1,
       0,   -1,    1,   -1,    0,   -1,    0,   -1,   -1,    0,    1,   -1,    0,   -1,    0,   -1,
       0,    1,    1,   -1,    0,   -1,    0,   -1,    0,    0,   -1,    0,    0,   -1,    0,   -1,
       1,    1,   -1,    0,    0,   -1,    0,   -1,   -1,   -1,    0,    0,    0,   -1,    0,   -1,
       0,   -1,    0,    0,    0,   -1,    0,   -1,    1,   -1,    0,    0,    0,   -1,    0,   -1,
      -1,    0,    0,    0,    0,   -1,    0,   -1,    0,    0,    0,    0,    0,   -1,    0,   -1,
       1,    0,    0,    0,    0,   -1,    0,   -1,    0,    1,    0,    0,    0,   -1,    0,   -1,
      -1,   -1,    1,    0,    0,   -1,    0,   -1,    0,    0,    1,    0,    0,   -1,    0,   -1,
      -1,    0,   -1,    1,    0,   -1,    0,   -1,    1,   -1,    0,    1,    0,   -1,    0,   -1,
       0,    0,    0,    1,    0,   -1,    0,   -1,    0,   -1,    1,    1,    0,   -1,    0,   -1,
      -1,    0,    1,    1,    0,   -1,    0,   -1,    0,   -1,    0,   -1,    1,   -1,    0,   -1,
      -1,    0,    0,   -1,    1,   -1,    0,   -1,    1,    0,    0,   -1,    1,   -1,    0,   -1,
       0,    0,    1,   -1,    1,   -1,    0,   -1,   -1,   -1,   -1,    0,    1,   -1,    0,   -1,
       1,    0,   -1,    0,    1,   -1,    0,   -1,    0,    1,   -1,    0,    1,   -1,    0,   -1,
       1,   -1,    0,    0,    1,   -1,    0,   -1,    0,    0,    0,    0,    1,   -1,    0,   -1,
      -1,    1,    0,    0,    1,   -1,    0,   -1,    1,    1,    0,    0,    1,   -1,    0,   -1,
      -1,    0,    1,    0,    1,   -1,    0,   -1,    1,    0,    1,    0,    1,   -1,    0,   -1,
       0,    0,   -1,    1,    1,   -1,    0,   -1,    0,   -1,    0,    1,    1,   -1,    0,   -1,
      -1,    0,    0,    1,    1,   -1,    0,   -1,    1,    0,    0,    1,    1,   -1,    0,   -1,
       0,    0,    1,    1,    1,   -1,    0,   -1,    0,   -1,   -1,   -1,   -1,    0,    0,   -1,
       1,    0,   -1,   -1,   -1,    0,    0,   -1,    0,    1,   -1,   -1,   -1,    0,    0,   -1,
      -1,    0,    0,   -1,   -1,    0,    0,   -1,    0,    0,    0,   -1,   -1,    0,    0,   -1,
      -1,    1,    0,   -1,   -1,    0,    0,   -1,    0,    1,    0,   -1,   -1,    0,    0,   -1,
       0,   -1,    1,   -1,   -1,    0,    0,   -1,    1,    0,    1,   -1,   -1,    0,    0,   -1,
       0,   -1,   -1,    0,   -1,    0,    0,   -1,    0,    0,   -1,    0,   -1,    0,    0,   -1,
       1,    0,   -1,    0,   -1,    0,    0,   -1,   -1,    1,   -1,    0,   -1,    0,    0,   -1,
       1,    1,   -1,    0,   -1,    0,    0,   -1,    0,   -1,    0,    0,   -1,    0,    0,   -1,
      -1,    0,    0,    0,   -1,    0,    0,   -1,    0,    0,    0,    0,   -1,    0,    0,   -1,
       1,    0,    0,    0,   -1,    0,    0,   -1,    0,    1,    0,    0,   -1,    0,    0,   -1,
       1,   -1,    1,    0,   -1,    0,    0,   -1,    0,    0,    1,    0,   -1,    0,    0,   -1,
      -1,    1,    1,    0,   -1,    0,    0,   -1,   -1,    0,   -1,    1,   -1,    0,    0,   -1,
       0,    1,   -1,    1,   -1,    0,    0,   -1,   -1,   -1,    0,    1,   -1,    0,    0,   -1,
      -1,    0,    0,    1,   -1,    0,    0,   -1,    0,    0,    0,    1,   -1,    0,    0,   -1,
      -1,    1,    0,    1,   -1,    0,    0,   -1,    0,    1,    0,    1,   -1,    0,    0,   -1,
       1,    1,    0,    1,   -1,    0,    0,   -1,    0,   -1,    1,    1,   -1,    0,    0,   -1,
      -1,    0,    1,    1,   -1,    0,    0,   -1,    0,    0,    1,    1,   -1,    0,    0,   -1,
       0,    1,    1,    1,   -1,    0,    0,   -1,    1,   -1,   -1,   -1,    0,    0,    0,   -1,
       0,    0,   -1,   -1,    0,    0,    0,   -1,    1,    1,   -1,   -1,    0,    0,    0,   -1,
       0,   -1,    0,   -1,    0,    0,    0,   -1,   -1,    0,    0,   -1,    0,    0,    0,   -1,
       0,    0,    0,   -1,    0,    0,    0,   -1,    1,    0,    0,   -1,    0,    0,    0,   -1,
       0,    1,    0,   -1,    0,    0,    0,   -1,   -1,   -1,    1,   -1,    0,    0,    0,   -1,
       1,   -1,    1,   -1,    0,    0,    0,   -1,    0,    0,    1,   -1,    0,    0,    0,   -1,
      -1,    1,    1,   -1,    0,    0,    0,   -1,    1,    1,    1,   -1,    0,    0,    0,   -1,
       0,   -1,   -1,    0,    0,    0,    0,   -1,   -1,    0,   -1,    0,    0,    0,    0,   -1,
       0,    0,   -1,    0,    0,    0,    0,   -1,    1,    0,   -1,    0,    0,    0,    0,   -1,
       0,   -1,    0,    0,    0,    0,    0,   -1,    1,   -1,    0,    0,    0,    0,    0,   -1,
      -1,    0,    0,    0,    0,    0,    0,   -1,    0,    0,    0,    0,    0,    0,    0,   -1,
       1,    0,    0,    0,    0,    0,    0,   -1,    0,    1,    0,    0,    0,    0,    0,   -1,
       1,    1,    0,    0,    0,    0,    0,   -1,    0,   -1,    1,    0,    0,    0,    0,   -1,
      -1,    0,    1,    0,    0,    0,    0,   -1,    0,    0,    1,    0,    0,    0,    0,   -1,
       1,    0,    1,    0,    0,    0,    0,   -1,    0,    1,    1,    0,    0,    0,    0,   -1,
      -1,   -1,   -1,    1,    0,    0,    0,   -1,    1,   -1,   -1,    1,    0,    0,    0,   -1,
      -1,    0,   -1,    1,    0,    0,    0,   -1,    0,    0,   -1,    1,    0,    0,    0,   -1,
      -1,    1,   -1,    1,    0,    0,    0,   -1,    1,    1,   -1,    1,    0,    0,    0,   -1,
      -1,   -1,    0,    1,    0,    0,    0,   -1,    0,   -1,    0,    1,    0,    0,    0,   -1,
      -1,    0,    0,    1,    0,    0,    0,   -1,    0,    0,    0,    1,    0,    0,    0,   -1,
       1,    0,    0,    1,    0,    0,    0,   -1,    0,    1,    0,    1,    0,    0,    0,   -1,
       1,    1,    0,    1,    0,    0,    0,   -1,   -1,   -1,    1,    1,    0,    0,    0,   -1,
       1,   -1,    1,    1,    0,    0,    0,   -1,    0,    0,    1,    1,    0,    0,    0,   -1,
       0,   -1,   -1,   -1,    1,    0,    0,   -1,   -1,    0,   -1,   -1,    1,    0,    0,   -1,
       0,    0,   -1,   -1,    1,    0,    0,   -1,    1,    0,   -1,   -1,    1,    0,    0,   -1,
       0,    0,    0,   -1,    1,    0,    0,   -1,    1,    0,    0,   -1,    1,    0,    0,   -1,
      -1,    1,    0,   -1,    1,    0,    0,   -1,    1,    1,    0,   -1,    1,    0,    0,   -1,
       0,   -1,    1,   -1,    1,    0,    0,   -1,    1,    0,    1,   -1,    1,    0,    0,   -1,
      -1,   -1,   -1,    0,    1,    0,    0,   -1,    1,   -1,   -1,    0,    1,    0,    0,   -1,
      -1,    0,   -1,    0,    1,    0,    0,   -1,    0,    0,   -1,    0,    1,    0,    0,   -1,
      -1,    1,   -1,    0,    1,    0,    0,   -1,    1,    1,   -1,    0,    1,    0,    0,   -1,
       0,   -1,    0,    0,    1,    0,    0,   -1,    0,    0,    0,    0,    1,    0,    0,   -1,
       1,    0,    0,    0,    1,    0,    0,   -1,   -1,    1,    0,    0,    1,    0,    0,   -1,
       0,    1,    0,    0,    1,    0,    0,   -1,    0,   -1,    1,    0,    1,    0,    0,   -1,
      -1,    0,    1,    0,    1,    0,    0,   -1,    0,    0,    1,    0,    1,    0,    0,   -1,
      -1,    1,    1,    0,    1,    0,    0,   -1,    0,    1,    1,    0,    1,    0,    0,   -1,
       1,    1,    1,    0,    1,    0,    0,   -1,    1,    0,   -1,    1,    1,    0,    0,   -1,
       1,    1,   -1,    1,    1,    0,    0,   -1,    1,   -1,    0,    1,    1,    0,    0,   -1,
       0,    0,    0,    1,    1,    0,    0,   -1,   -1,    0,    1,    1,    1,    0,    0,   -1,
       0,    1,    1,    1,    1,    0,    0,   -1,    0,   -1,    0,   -1,   -1,    1,    0,   -1,
       1,    0,    0,   -1,   -1,    1,    0,   -1,    0,    0,    1,   -1,   -1,    1,    0,   -1,
       0,   -1,   -1,    0,   -1,    1,    0,   -1,   -1,    0,   -1,    0,   -1,    1,    0,   -1,
       1,    0,   -1,    0,   -1,    1,    0,   -1,    0,    1,   -1,    0,   -1,    1,    0,   -1,
      -1,   -1,    0,    0,   -1,    1,    0,   -1,    0,    0,    0,    0,   -1,    1,    0,   -1,
      -1,    1,    0,    0,   -1,    1,    0,   -1,    1,    1,    0,    0,   -1,    1,    0,   -1,
      -1,   -1,    1,    0,   -1,    1,    0,   -1,    0,   -1,    1,    0,   -1,    1,    0,   -1,
      -1,    0,    1,    0,   -1,    1,    0,   -1,    1,    0,    1,    0,   -1,    1,    0,   -1,
       0,    1,    1,    0,   -1,    1,    0,   -1,    0,    0,   -1,    1,   -1,    1,    0,   -1,
       0,   -1,    0,    1,   -1,    1,    0,   -1,   -1,    0,    0,    1,   -1,    1,    0,   -1,
       0,    0,    1,    1,   -1,    1,    0,   -1,   -1,   -1,    0,   -1,    0,    1,    0,   -1,
       1,   -1,    0,   -1,    0,    1,    0,   -1,    0,    0,    0,   -1,    0,    1,    0,   -1,
       1,    1,    0,   -1,    0,    1,    0,   -1,    0,   -1,    1,   -1,    0,    1,    0,   -1,
       0,    0,    1,   -1,    0,    1,    0,   -1,    1,   -1,   -1,    0,    0,    1,    0,   -1,
      -1,    0,   -1,    0,    0,    1,    0,   -1,    0,    0,   -1,    0,    0,    1,    0,   -1,
      -1,    1,   -1,    0,    0,    1,    0,   -1,    0,   -1,    0,    0,    0,    1,    0,   -1,
      -1,    0,    0,    0,    0,    1,    0,   -1,    0,    0,    0,    0,    0,    1,    0,   -1,
       1,    0,    0,    0,    0,    1,    0,   -1,    0,    1,    0,    0,    0,    1,    0,   -1,
       1,    1,    0,    0,    0,    1,    0,   -1,   -1,   -1,    1,    0,    0,    1,    0,   -1,
       0,    0,    1,    0,    0,    1,    0,   -1,    1,    1,    1,    0,    0,    1,    0,   -1,
       0,    1,   -1,    1,    0,    1,    0,   -1,    0,   -1,    0,    1,    0,    1,    0,   -1,
       1,   -1,    0,    1,    0,    1,    0,   -1,    0,    0,    0,    1,    0,    1,    0,   -1,
      -1,    1,    0,    1,    0,    1,    0,   -1,    0,   -1,    1,    1,    0,    1,    0,   -1,
       1,    0,    1,    1,    0,    1,    0,   -1,    0,    1,    1,    1,    0,    1,    0,   -1,
       0,    1,   -1,   -1,    1,    1,    0,   -1,    1,    0,    0,   -1,    1,    1,    0,   -1,
      -1,    0,    1,   -1,    1,    1,    0,   -1,    1,    0,    1,   -1,    1,    1,    0,   -1,
      -1,    0,   -1,    0,    1,    1,    0,   -1,    1,    0,   -1,    0,    1,    1,    0,   -1,
       0,    1,   -1,    0,    1,    1,    0,   -1,   -1,   -1,    0,    0,    1,    1,    0,   -1,
       1,   -1,    0,    0,    1,    1,    0,   -1,    0,    0,    0,    0,    1,    1,    0,   -1,
      -1,    1,    0,    0,    1,    1,    0,   -1,    0,   -1,    1,    0,    1,    1,    0,   -1,
       1,    0,    1,    0,    1,    1,    0,   -1,    0,    1,    1,    0,    1,    1,    0,   -1,
       0,    0,   -1,    1,    1,    1,    0,   -1,    0,   -1,    0,    1,    1,    1,    0,   -1,
       1,    0,    0,    1,    1,    1,    0,   -1,    1,    1,    0,    1,    1,    1,    0,   -1,
      -1,   -1,   -1,   -1,   -1,   -1,    1,   -1,    1,   -1,   -1,   -1,   -1,   -1,    1,   -1,
      -1,    1,   -1,   -1,   -1,   -1,    1,   -1,    1,    1,   -1,   -1,   -1,   -1,    1,   -1,
       0,    0,    0,   -1,   -1,   -1,    1,   -1,   -1,   -1,    1,   -1,   -1,   -1,    1,   -1,
       1,   -1,    1,   -1,   -1,   -1,    1,   -1,    0,    0,    1,   -1,   -1,   -1,    1,   -1,
      -1,    1,    1,   -1,   -1,   -1,    1,   -1,    1,    1,    1,   -1,   -1,   -1,    1,   -1,
       0,    0,   -1,    0,   -1,   -1,    1,   -1,    0,   -1,    0,    0,   -1,   -1,    1,   -1,
       0,    1,    0,    0,   -1,   -1,    1,   -1,    0,   -1,    1,    0,   -1,   -1,    1,   -1,
       0,    0,    1,    0,   -1,   -1,    1,   -1,   -1,   -1,   -1,    1,   -1,   -1,    1,   -1,
       1,   -1,   -1,    1,   -1,   -1,    1,   -1,   -1,    1,   -1,    1,   -1,   -1,    1,   -1,
       1,    1,   -1,    1,   -1,   -1,    1,   -1,    0,    0,    0,    1,   -1,   -1,    1,   -1,
      -1,   -1,    1,    1,   -1,   -1,    1,   -1,    1,   -1,    1,    1,   -1,   -1,    1,   -1,
       0,    0,    1,    1,   -1,   -1,    1,   -1,   -1,    1,    1,    1,   -1,   -1,    1,   -1,
       1,    1,    1,    1,   -1,   -1,    1,   -1,    0,    0,   -1,   -1,    0,   -1,    1,   -1,
       0,   -1,    0,   -1,    0,   -1,    1,   -1,   -1,    0,    0,   -1,    0,   -1,    1,   -1,
       0,    1,    0,   -1,    0,   -1,    1,   -1,    0,    0,    1,   -1,    0,   -1,    1,   -1,
       1,   -1,   -1,    0,    0,   -1,    1,   -1,   -1,    0,   -1,    0,    0,   -1,    1,   -1,
       0,    1,   -1,    0,    0,   -1,    1,   -1,    0,    0,    0,    0,    0,   -1,    1,   -1,
      -1,    1,    0,    0,    0,   -1,    1,   -1,    1,    1,    0,    0,    0,   -1,    1,   -1,
       0,   -1,    1,    0,    0,   -1,    1,   -1,   -1,    0,    1,    0,    0,   -1,    1,   -1,
       0,    0,    1,    0,    0,   -1,    1,   -1,    1,    0,    1,    0,    0,   -1,    1,   -1,
       0,    0,   -1,    1,    0,   -1,    1,   -1,   -1,   -1,    0,    1,    0,   -1,    1,   -1,
       1,    0,    0,    1,    0,   -1,    1,   -1,    0,    1,    0,    1,    0,   -1,    1,   -1,
       0,    0,    1,    1,    0,   -1,    1,   -1,    0,   -1,   -1,   -1,    1,   -1,    1,   -1,
      -1,    1,   -1,   -1,    1,   -1,    1,   -1,    1,    1,   -1,   -1,    1,   -1,    1,   -1,
       0,   -1,    0,   -1,    1,   -1,    1,   -1,    0,    0,    0,   -1,    1,   -1,    1,   -1,
      -1,   -1,    1,   -1,    1,   -1,    1,   -1,    1,   -1,    1,   -1,    1,   -1,    1,   -1,
      -1,    1,    1,   -1,    1,   -1,    1,   -1,    1,    1,    1,   -1,    1,   -1,    1,   -1,
       0,    0,   -1,    0,    1,   -1,    1,   -1,    0,   -1,    0,    0,    1,   -1,    1,   -1,
       1,    0,    0,    0,    1,   -1,    1,   -1,    0,    1,    0,    0,    1,   -1,    1,   -1,
       0,    0,    1,    0,    1,   -1,    1,   -1,    0,   -1,   -1,    1,    1,   -1,    1,   -1,
      -1,    1,   -1,    1,    1,   -1,    1,   -1,    1,    1,   -1,    1,    1,   -1,    1,   -1,
       0,   -1,    0,    1,    1,   -1,    1,   -1,    0,    0,    0,    1,    1,   -1,    1,   -1,
      -1,   -1,    1,    1,    1,   -1,    1,   -1,    1,   -1,    1,    1,    1,   -1,    1,   -1,
      -1,    1,    1,    1,    1,   -1,    1,   -1,    1,    1,    1,    1,    1,   -1,    1,   -1,
       0,    0,   -1,   -1,   -1,    0,    1,   -1,   -1,    0,    0,   -1,   -1,    0,    1,   -1,
       1,    0,    0,   -1,   -1,    0,    1,   -1,    0,    1,    0,   -1,   -1,    0,    1,   -1,
       0,    0,    1,   -1,   -1,    0,    1,   -1,   -1,    0,   -1,    0,   -1,    0,    1,   -1,
       0,    0,   -1,    0,   -1,    0,    1,   -1,    1,    0,   -1,    0,   -1,    0,    1,   -1,
       0,    1,   -1,    0,   -1,    0,    1,   -1,    1,   -1,    0,    0,   -1,    0,    1,   -1,
       0,    0,    0,    0,   -1,    0,    1,   -1,   -1,    1,    0,    0,   -1,    0,    1,   -1,
       1,    1,    0,    0,   -1,    0,    1,   -1,    1,    0,    1,    0,   -1,    0,    1,   -1,
       0,    0,   -1,    1,   -1,    0,    1,   -1,    0,   -1,    0,    1,   -1,    0,    1,   -1,
      -1,    0,    0,    1,   -1,    0,    1,   -1,    0,    1,    0,    1,   -1,    0,    1,   -1,
       0,   -1,    1,    1,   -1,    0,    1,   -1,    0,    0,    1,    1,   -1,    0,    1,   -1,
       0,    1,   -1,   -1,    0,    0,    1,   -1,    0,    0,    0,   -1,    0,    0,    1,   -1,
       0,   -1,    1,   -1,    0,    0,    1,   -1,    0,    1,    1,   -1,    0,    0,    1,   -1,
      -1,   -1,   -1,    0,    0,    0,    1,   -1,    0,    0,   -1,    0,    0,    0,    1,   -1,
      -1,    1,   -1,    0,    0,    0,    1,   -1,    0,   -1,    0,    0,    0,    0,    1,   -1,
      -1,    0,    0,    0,    0,    0,    1,   -1,    0,    0,    0,    0,    0,    0,    1,   -1,
       0,    1,    0,    0,    0,    0,    1,   -1,    1,   -1,    1,    0,    0,    0,    1,   -1,
       0,    0,    1,    0,    0,    0,    1,   -1,   -1,    1,    1,    0,    0,    0,    1,   -1,
       0,    1,   -1,    1,    0,    0,    1,   -1,   -1,   -1,    0,    1,    0,    0,    1,   -1,
      -1,    0,    0,    1,    0,    0,    1,   -1,    0,    0,    0,    1,    0,    0,    1,   -1,
      -1,    1,    0,    1,    0,    0,    1,   -1,    1,    1,    0,    1,    0,    0,    1,   -1,
       0,   -1,    1,    1,    0,    0,    1,   -1,   -1,    0,    1,    1,    0,    0,    1,   -1,
       1,    0,    1,    1,    0,    0,    1,   -1,    0,    1,    1,    1,    0,    0,    1,   -1,
       0,    0,   -1,   -1,    1,    0,    1,   -1,   -1,   -1,    0,   -1,    1,    0,    1,   -1,
       1,   -1,    0,   -1,    1,    0,    1,   -1,    0,    1,    0,   -1,    1,    0,    1,   -1,
       0,    0,    1,   -1,    1,    0,    1,   -1,    0,   -1,   -1,    0,    1,    0,    1,   -1,
       0,    1,   -1,    0,    1,    0,    1,   -1,    0,    0,    0,    0,    1,    0,    1,   -1,
      -1,   -1,    1,    0,    1,    0,    1,   -1,    0,   -1,    1,    0,    1,    0,    1,   -1,
       0,    1,    1,    0,    1,    0,    1,   -1,   -1,    0,   -1,    1,    1,    0,    1,   -1,
       1,    0,   -1,    1,    1,    0,    1,   -1,   -1,   -1,    0,    1,    1,    0,    1,   -1,
       1,    1,    0,    1,    1,    0,    1,   -1,   -1,   -1,   -1,   -1,   -1,    1,    1,   -1,
       1,   -1,   -1,   -1,   -1,    1,    1,   -1,   -1,    1,   -1,   -1,   -1,    1,    1,   -1,
       1,    1,   -1,   -1,   -1,    1,    1,   -1,    0,    0,    0,   -1,   -1,    1,    1,   -1,
      -1,   -1,    1,   -1,   -1,    1,    1,   -1,    1,   -1,    1,   -1,   -1,    1,    1,   -1,
      -1,    1,    1,   -1,   -1,    1,    1,   -1,    1,    1,    1,   -1,   -1,    1,    1,   -1,
       0,    0,   -1,    0,   -1,    1,    1,   -1,    0,   -1,    0,    0,   -1,    1,    1,   -1,
      -1,    0,    0,    0,   -1,    1,    1,   -1,    0,    0,    1,    0,   -1,    1,    1,   -1,
      -1,   -1,   -1,    1,   -1,    1,    1,   -1,    1,   -1,   -1,    1,   -1,    1,    1,   -1,
      -1,    1,   -1,    1,   -1,    1,    1,   -1,    1,    1,   -1,    1,   -1,    1,    1,   -1,
      -1,   -1,    1,    1,   -1,    1,    1,   -1,    1,   -1,    1,    1,   -1,    1,    1,   -1,
      -1,    1,    1,    1,   -1,    1,    1,   -1,    1,    1,    1,    1,   -1,    1,    1,   -1,
       0,    1,   -1,   -1,    0,    1,    1,   -1,    0,   -1,    0,   -1,    0,    1,    1,   -1,
      -1,    0,    0,   -1,    0,    1,    1,   -1,    0,    1,    0,   -1,    0,    1,    1,   -1,
       0,    0,    1,   -1,    0,    1,    1,   -1,    1,    0,   -1,    0,    0,    1,    1,   -1,
       0,    1,   -1,    0,    0,    1,    1,   -1,    1,   -1,    0,    0,    0,    1,    1,   -1,
       0,    0,    0,    0,    0,    1,    1,   -1,    0,   -1,    1,    0,    0,    1,    1,   -1,
      -1,    0,    1,    0,    0,    1,    1,   -1,    1,    0,    1,    0,    0,    1,    1,   -1,
       0,    1,    1,    0,    0,    1,    1,   -1,    0,    0,   -1,    1,    0,    1,    1,   -1,
      -1,   -1,    0,    1,    0,    1,    1,   -1,    1,    0,    0,    1,    0,    1,    1,   -1,
       0,    1,    0,    1,    0,    1,    1,   -1,   -1,    0,    1,    1,    0,    1,    1,   -1,
       0,    0,    1,    1,    0,    1,    1,   -1,   -1,   -1,   -1,   -1,    1,    1,    1,   -1,
       1,   -1,   -1,   -1,    1,    1,    1,   -1,   -1,    1,   -1,   -1,    1,    1,    1,   -1,
       1,    1,   -1,   -1,    1,    1,    1,   -1,   -1,   -1,    1,   -1,    1,    1,    1,   -1,
       1,   -1,    1,   -1,    1,    1,    1,   -1,   -1,    1,    1,   -1,    1,    1,    1,   -1,
       1,    1,    1,   -1,    1,    1,    1,   -1,    0,    0,   -1,    0,    1,    1,    1,   -1,
       0,   -1,    0,    0,    1,    1,    1,   -1,    1,    0,    0,    0,    1,    1,    1,   -1,
       0,    1,    0,    0,    1,    1,    1,   -1,    0,    0,    1,    0,    1,    1,    1,   -1,
      -1,   -1,   -1,    1,    1,    1,    1,   -1,    1,   -1,   -1,    1,    1,    1,    1,   -1,
      -1,    1,   -1,    1,    1,    1,    1,   -1,    1,    1,   -1,    1,    1,    1,    1,   -1,
       0,    0,    0,    1,    1,    1,    1,   -1,   -1,   -1,    1,    1,    1,    1,    1,   -1,
       1,   -1,    1,    1,    1,    1,    1,   -1,   -1,    1,    1,    1,    1,    1,    1,   -1,
       1,    1,    1,    1,    1,    1,    1,   -1,    0,    0,   -1,   -1,   -1,   -1,   -1,    0,
       0,   -1,    0,   -1,   -1,   -1,   -1,    0,    1,    0,    0,   -1,   -1,   -1,   -1,    0,
       0,    0,    1,   -1,   -1,   -1,   -1,    0,    0,    1,   -1,    0,   -1,   -1,   -1,    0,
       1,   -1,    0,    0,   -1,   -1,   -1,    0,    0,    0,    0,    0,   -1,   -1,   -1,    0,
      -1,    1,    0,    0,   -1,   -1,   -1,    0,    1,    1,    0,    0,   -1,   -1,   -1,    0,
       0,   -1,    1,    0,   -1,   -1,   -1,    0,   -1,    0,    1,    0,   -1,   -1,   -1,    0,
       1,    0,    1,    0,   -1,   -1,   -1,    0,   -1,    0,    0,    1,   -1,   -1,   -1,    0,
       0,    1,    0,    1,   -1,   -1,   -1,    0,    0,   -1,    1,    1,   -1,   -1,   -1,    0,
       1,    0,    1,    1,   -1,   -1,   -1,    0,   -1,   -1,   -1,   -1,    0,   -1,   -1,    0,
       0,   -1,   -1,   -1,    0,   -1,   -1,    0,   -1,    0,   -1,   -1,    0,   -1,   -1,    0,
       1,    0,   -1,   -1,    0,   -1,   -1,    0,    0,    1,   -1,   -1,    0,   -1,   -1,    0,
       1,   -1,    0,   -1,    0,   -1,   -1,    0,    0,    0,    0,   -1,    0,   -1,   -1,    0,
       1,    0,    0,   -1,    0,   -1,   -1,    0,   -1,    1,    0,   -1,    0,   -1,   -1,    0,
       1,    1,    0,   -1,    0,   -1,   -1,    0,    0,   -1,    1,   -1,    0,   -1,   -1,    0,
       1,    0,    1,   -1,    0,   -1,   -1,    0,    0,    1,    1,   -1,    0,   -1,   -1,    0,
       0,    0,   -1,    0,    0,   -1,   -1,    0,   -1,    1,   -1,    0,    0,   -1,   -1,    0,
       1,    1,   -1,    0,    0,   -1,   -1,    0,    0,   -1,    0,    0,    0,   -1,   -1,    0,
      -1,    0,    0,    0,    0,   -1,   -1,    0,    0,    0,    0,    0,    0,   -1,   -1,    0,
       1,    0,    0,    0,    0,   -1,   -1,    0,    0,    1,    0,    0,    0,   -1,   -1,    0,
       1,    1,    0,    0,    0,   -1,   -1,    0,    0,    0,    1,    0,    0,   -1,   -1,    0,
      -1,    1,    1,    0,    0,   -1,   -1,    0,    1,    1,    1,    0,    0,   -1,   -1,    0,
       0,   -1,   -1,    1,    0,   -1,   -1,    0,   -1,    0,   -1,    1,    0,   -1,   -1,    0,
       1,    0,   -1,    1,    0,   -1,   -1,    0,   -1,   -1,    0,    1,    0,   -1,   -1,    0,
       1,   -1,    0,    1,    0,   -1,   -1,    0,    0,    0,    0,    1,    0,   -1,   -1,    0,
      -1,   -1,    1,    1,    0,   -1,   -1,    0,    0,   -1,    1,    1,    0,   -1,   -1,    0,
       1,   -1,    1,    1,    0,   -1,   -1,    0,    0,    0,   -1,   -1,    1,   -1,   -1,    0,
       0,   -1,    0,   -1,    1,   -1,   -1,    0,   -1,    0,    0,   -1,    1,   -1,   -1,    0,
       1,    0,    0,   -1,    1,   -1,   -1,    0,    0,    0,    1,   -1,    1,   -1,   -1,    0,
       0,   -1,   -1,    0,    1,   -1,   -1,    0,    1,   -1,    0,    0,    1,   -1,   -1,    0,
       0,    0,    0,    0,    1,   -1,   -1,    0,    1,    1,    0,    0,    1,   -1,   -1,    0,
      -1,    0,    1,    0,    1,   -1,   -1,    0,    0,    1,    1,    0,    1,   -1,   -1,    0,
       0,    1,   -1,    1,    1,   -1,   -1,    0,   -1,    0,    0,    1,    1,   -1,   -1,    0,
       0,    0,    1,    1,    1,   -1,   -1,    0,    0,   -1,   -1,   -1,   -1,    0,   -1,    0,
       0,    0,    0,   -1,   -1,    0,   -1,    0,    0,    1,    0,   -1,   -1,    0,   -1,    0,
       0,    1,    1,   -1,   -1,    0,   -1,    0,    0,    0,   -1,    0,   -1,    0,   -1,    0,
      -1,    1,   -1,    0,   -1,    0,   -1,    0,    1,    1,   -1,    0,   -1,    0,   -1,    0,
       0,   -1,    0,    0,   -1,    0,   -1,    0,   -1,    0,    0,    0,   -1,    0,   -1,    0,
       0,    0,    0,    0,   -1,    0,   -1,    0,    1,    0,    0,    0,   -1,    0,   -1,    0,
       0,   -1,    1,    0,   -1,    0,   -1,    0,    1,   -1,    1,    0,   -1,    0,   -1,    0,
       0,    0,    1,    0,   -1,    0,   -1,    0,   -1,    1,    1,    0,   -1,    0,   -1,    0,
       1,    1,    1,    0,   -1,    0,   -1,    0,    0,   -1,   -1,    1,   -1,    0,   -1,    0,
       1,    0,   -1,    1,   -1,    0,   -1,    0,    0,    1,   -1,    1,   -1,    0,   -1,    0,
      -1,   -1,    0,    1,   -1,    0,   -1,    0,    1,   -1,    0,    1,   -1,    0,   -1,    0,
       0,    0,    0,    1,   -1,    0,   -1,    0,   -1,   -1,    1,    1,   -1,    0,   -1,    0,
       0,   -1,    1,    1,   -1,    0,   -1,    0,    0,    1,    1,    1,   -1,    0,   -1,    0,
       0,   -1,   -1,   -1,    0,    0,   -1,    0,    1,   -1,   -1,   -1,    0,    0,   -1,    0,
       0,    0,   -1,   -1,    0,    0,   -1,    0,    1,    1,   -1,   -1,    0,    0,   -1,    0,
       0,   -1,    0,   -1,    0,    0,   -1,    0,   -1,    0,    0,   -1,    0,    0,   -1,    0,
       0,    0,    0,   -1,    0,    0,   -1,    0,    1,    0,    0,   -1,    0,    0,   -1,    0,
       0,    1,    0,   -1,    0,    0,   -1,    0,   -1,   -1,    1,   -1,    0,    0,   -1,    0,
       0,    0,    1,   -1,    0,    0,   -1,    0,    1,    1,    1,   -1,    0,    0,   -1,    0,
       0,   -1,   -1,    0,    0,    0,   -1,    0,   -1,    0,   -1,    0,    0,    0,   -1,    0,
       0,    0,   -1,    0,    0,    0,   -1,    0,    1,    0,   -1,    0,    0,    0,   -1,    0,
       0,    1,   -1,    0,    0,    0,   -1,    0,   -1,   -1,    0,    0,    0,    0,   -1,    0,
       0,   -1,    0,    0,    0,    0,   -1,    0,   -1,    0,    0,    0,    0,    0,   -1,    0,
       0,    0,    0,    0,    0,    0,   -1,    0,    1,    0,    0,    0,    0,    0,   -1,    0,
      -1,    1,    0,    0,    0,    0,   -1,    0,    0,    1,    0,    0,    0,    0,   -1,    0,
       0,   -1,    1,    0,    0,    0,   -1,    0,   -1,    0,    1,    0,    0,    0,   -1,    0,
       0,    0,    1,    0,    0,    0,   -1,    0,    1,    0,    1,    0,    0,    0,   -1,    0,
       0,    1,    1,    0,    0,    0,   -1,    0,    1,   -1,   -1,    1,    0,    0,   -1,    0,
      -1,    0,   -1,    1,    0,    0,   -1,    0,    0,    0,   -1,    1,    0,    0,   -1,    0,
      -1,    1,   -1,    1,    0,    0,   -1,    0,    0,   -1,    0,    1,    0,    0,   -1,    0,
      -1,    0,    0,    1,    0,    0,   -1,    0,    0,    0,    0,    1,    0,    0,   -1,    0,
       1,    0,    0,    1,    0,    0,   -1,    0,    0,    1,    0,    1,    0,    0,   -1,    0,
       1,    1,    0,    1,    0,    0,   -1,    0,    0,    0,    1,    1,    0,    0,   -1,    0,
      -1,    1,    1,    1,    0,    0,   -1,    0,    1,    1,    1,    1,    0,    0,   -1,    0,
       0,   -1,   -1,   -1,    1,    0,   -1,    0,    0,    0,   -1,   -1,    1,    0,   -1,    0,
       0,    1,   -1,   -1,    1,    0,   -1,    0,   -1,    0,    0,   -1,    1,    0,   -1,    0,
       0,    0,    0,   -1,    1,    0,   -1,    0,   -1,    1,    0,   -1,    1,    0,   -1,    0,
       1,    1,    0,   -1,    1,    0,   -1,    0,    0,   -1,    1,   -1,    1,    0,   -1,    0,
      -1,    0,    1,   -1,    1,    0,   -1,    0,    0,    1,    1,   -1,    1,    0,   -1,    0,
      -1,   -1,   -1,    0,    1,    0,   -1,    0,    1,   -1,   -1,    0,    1,    0,   -1,    0,
       0,    0,   -1,    0,    1,    0,   -1,    0,   -1,    1,   -1,    0,    1,    0,   -1,    0,
      -1,   -1,    0,    0,    1,    0,   -1,    0,    0,   -1,    0,    0,    1,    0,   -1,    0,
       1,   -1,    0,    0,    1,    0,   -1,    0,    0,    0,    0,    0,    1,    0,   -1,    0,
       1,    0,    0,    0,    1,    0,   -1,    0,    0,    1,    0,    0,    1,    0,   -1,    0,
       1,   -1,    1,    0,    1,    0,   -1,    0,    0,    0,    1,    0,    1,    0,   -1,    0,
      -1,    1,    1,    0,    1,    0,   -1,    0,    0,   -1,   -1,    1,    1,    0,   -1,    0,
       0,    0,   -1,    1,    1,    0,   -1,    0,    1,    1,   -1,    1,    1,    0,   -1,    0,
      -1,    0,    0,    1,    1,    0,   -1,    0,    0,    0,    0,    1,    1,    0,   -1,    0,
       0,   -1,    1,    1,    1,    0,   -1,    0,   -1,    0,    1,    1,    1,    0,   -1,    0,
       1,    0,    1,    1,    1,    0,   -1,    0,    0,    0,   -1,   -1,   -1,    1,   -1,    0,
       0,   -1,    0,   -1,   -1,    1,   -1,    0,    0,    0,    0,   -1,   -1,    1,   -1,    0,
       1,    1,    0,   -1,   -1,    1,   -1,    0,    0,    0,    1,   -1,   -1,    1,   -1,    0,
       1,   -1,   -1,    0,   -1,    1,   -1,    0,    0,    1,   -1,    0,   -1,    1,   -1,    0,
      -1,   -1,    0,    0,   -1,    1,   -1,    0,    0,    0,    0,    0,   -1,    1,   -1,    0,
      -1,    1,    0,    0,   -1,    1,   -1,    0,    0,   -1,    1,    0,   -1,    1,   -1,    0,
      -1,    0,    1,    0,   -1,    1,   -1,    0,    1,    0,    1,    0,   -1,    1,   -1,    0,
       0,    1,    1,    0,   -1,    1,   -1,    0,    0,    0,   -1,    1,   -1,    1,   -1,    0,
       0,   -1,    0,    1,   -1,    1,   -1,    0,   -1,    0,    0,    1,   -1,    1,   -1,    0,
       1,    0,    0,    1,   -1,    1,   -1,    0,    0,    1,    0,    1,   -1,    1,   -1,    0,
       0,    0,    1,    1,   -1,    1,   -1,    0,    0,   -1,   -1,   -1,    0,    1,   -1,    0,
       0,    0,   -1,   -1,    0,    1,   -1,    0,    1,    0,   -1,   -1,    0,    1,   -1,    0,
       1,    1,   -1,   -1,    0,    1,   -1,    0,   -1,   -1,    0,   -1,    0,    1,   -1,    0,
      -1,    0,    0,   -1,    0,    1,   -1,    0,    0,    0,    0,   -1,    0,    1,   -1,    0,
      -1,    1,    0,   -1,    0,    1,   -1,    0,    0,   -1,    1,   -1,    0,    1,   -1,    0,
      -1,    0,    1,   -1,    0,    1,   -1,    0,    1,    0,    1,   -1,    0,    1,   -1,    0,
      -1,   -1,   -1,    0,    0,    1,   -1,    0,    0,    0,   -1,    0,    0,    1,   -1,    0,
       1,    1,   -1,    0,    0,    1,   -1,    0,    0,   -1,    0,    0,    0,    1,   -1,    0,
       1,   -1,    0,    0,    0,    1,   -1,    0,   -1,    0,    0,    0,    0,    1,   -1,    0,
       0,    0,    0,    0,    0,    1,   -1,    0,    1,    0,    0,    0,    0,    1,   -1,    0,
       0,    1,    0,    0,    0,    1,   -1,    0,   -1,   -1,    1,    0,    0,    1,   -1,    0,
       1,   -1,    1,    0,    0,    1,   -1,    0,    0,    0,    1,    0,    0,    1,   -1,    0,
       1,    0,    1,    0,    0,    1,   -1,    0,    1,    1,    1,    0,    0,    1,   -1,    0,
       1,    0,   -1,    1,    0,    1,   -1,    0,    0,    1,   -1,    1,    0,    1,   -1,    0,
       1,   -1,    0,    1,    0,    1,   -1,    0,    0,    0,    0,    1,    0,    1,   -1,    0,
       1,    0,    0,    1,    0,    1,   -1,    0,   -1,    1,    0,    1,    0,    1,   -1,    0,
       0,   -1,    1,    1,    0,    1,   -1,    0,   -1,    0,    1,    1,    0,    1,   -1,    0,
       1,    0,    1,    1,    0,    1,   -1,    0,    0,    1,    1,    1,    0,    1,   -1,    0,
       1,    0,    0,   -1,    1,    1,   -1,    0,   -1,    0,   -1,    0,    1,    1,   -1,    0,
       1,    0,   -1,    0,    1,    1,   -1,    0,    0,    1,   -1,    0,    1,    1,   -1,    0,
       0,    0,    0,    0,    1,    1,   -1,    0,   -1,    1,    0,    0,    1,    1,   -1,    0,
       1,    1,    0,    0,    1,    1,   -1,    0,   -1,    0,    1,    0,    1,    1,   -1,    0,
       0,    1,    1,    0,    1,    1,   -1,    0,   -1,    0,    0,    1,    1,    1,   -1,    0,
       0,    0,    1,    1,    1,    1,   -1,    0,    0,   -1,   -1,   -1,   -1,   -1,    0,    0,
      -1,    0,   -1,   -1,   -1,   -1,    0,    0,    0,    0,   -1,   -1,   -1,   -1,    0,    0,
       1,    0,   -1,   -1,   -1,   -1,    0,    0,    0,    1,   -1,   -1,   -1,   -1,    0,    0,
       1,   -1,    0,   -1,   -1,   -1,    0,    0,    0,    0,    0,   -1,   -1,   -1,    0,    0,
       1,    1,    0,   -1,   -1,   -1,    0,    0,    0,   -1,    1,   -1,   -1,   -1,    0,    0,
      -1,    0,    1,   -1,   -1,   -1,    0,    0,    0,    1,    1,   -1,   -1,   -1,    0,    0,
      -1,   -1,   -1,    0,   -1,   -1,    0,    0,    0,    0,   -1,    0,   -1,   -1,    0,    0,
      -1,    1,   -1,    0,   -1,   -1,    0,    0,    0,   -1,    0,    0,   -1,   -1,    0,    0,
      -1,    0,    0,    0,   -1,   -1,    0,    0,    0,    0,    0,    0,   -1,   -1,    0,    0,
       1,    0,    0,    0,   -1,   -1,    0,    0,    0,    1,    0,    0,   -1,   -1,    0,    0,
       0,    0,    1,    0,   -1,   -1,    0,    0,   -1,    1,    1,    0,   -1,   -1,    0,    0,
       1,    0,   -1,    1,   -1,   -1,    0,    0,    0,    1,   -1,    1,   -1,   -1,    0,    0,
       0,    0,    0,    1,   -1,   -1,    0,    0,   -1,    1,    0,    1,   -1,   -1,    0,    0,
      -1,   -1,    1,    1,   -1,   -1,    0,    0,    0,   -1,    1,    1,   -1,   -1,    0,    0,
       1,    0,    1,    1,   -1,   -1,    0,    0,    0,    1,    1,    1,   -1,   -1,    0,    0,
       0,    0,   -1,   -1,    0,   -1,    0,    0,   -1,    1,   -1,   -1,    0,   -1,    0,    0,
       0,    1,   -1,   -1,    0,   -1,    0,    0,    1,    1,   -1,   -1,    0,   -1,    0,    0,
       0,   -1,    0,   -1,    0,   -1,    0,    0,   -1,    0,    0,   -1,    0,   -1,    0,    0,
       0,    0,    0,   -1,    0,   -1,    0,    0,    1,    0,    0,   -1,    0,   -1,    0,    0,
      -1,    1,    0,   -1,    0,   -1,    0,    0,    0,    1,    0,   -1,    0,   -1,    0,    0,
      -1,   -1,    1,   -1,    0,   -1,    0,    0,    0,    0,    1,   -1,    0,   -1,    0,    0,
       1,    0,    1,   -1,    0,   -1,    0,    0,   -1,    1,    1,   -1,    0,   -1,    0,    0,
       1,    1,    1,   -1,    0,   -1,    0,    0,    0,   -1,   -1,    0,    0,   -1,    0,    0,
      -1,    0,   -1,    0,    0,   -1,    0,    0,    0,    0,   -1,    0,    0,   -1,    0,    0,
       1,    0,   -1,    0,    0,   -1,    0,    0,    0,    1,   -1,    0,    0,   -1,    0,    0,
      -1,   -1,    0,    0,    0,   -1,    0,    0,    0,   -1,    0,    0,    0,   -1,    0,    0,
       1,   -1,    0,    0,    0,   -1,    0,    0,   -1,    0,    0,    0,    0,   -1,    0,    0,
       0,    0,    0,    0,    0,   -1,    0,    0,    1,    0,    0,    0,    0,   -1,    0,    0,
      -1,    1,    0,    0,    0,   -1,    0,    0,    0,    1,    0,    0,    0,   -1,    0,    0,
       1,    1,    0,    0,    0,   -1,    0,    0,    0,   -1,    1,    0,    0,   -1,    0,    0,
      -1,    0,    1,    0,    0,   -1,    0,    0,    0,    0,    1,    0,    0,   -1,    0,    0,
       1,    0,    1,    0,    0,   -1,    0,    0,    0,    1,    1,    0,    0,   -1,    0,    0,
       1,   -1,   -1,    1,    0,   -1,    0,    0,    0,    0,   -1,    1,    0,   -1,    0,    0,
       0,   -1,    0,    1,    0,   -1,    0,    0,   -1,    0,    0,    1,    0,   -1,    0,    0,
       0,    0,    0,    1,    0,   -1,    0,    0,    1,    0,    0,    1,    0,   -1,    0,    0,
       0,    1,    0,    1,    0,   -1,    0,    0,   -1,   -1,    1,    1,    0,   -1,    0,    0,
       0,    0,    1,    1,    0,   -1,    0,    0,    1,    1,    1,    1,    0,   -1,    0,    0,
       0,   -1,   -1,   -1,    1,   -1,    0,    0,    1,    0,   -1,   -1,    1,   -1,    0,    0,
       1,   -1,    0,   -1,    1,   -1,    0,    0,    0,    0,    0,   -1,    1,   -1,    0,    0,
       1,    1,    0,   -1,    1,   -1,    0,    0,    0,   -1,    1,   -1,    1,   -1,    0,    0,
      -1,    0,    1,   -1,    1,   -1,    0,    0,    1,   -1,   -1,    0,    1,   -1,    0,    0,
       0,    0,   -1,    0,    1,   -1,    0,    0,    1,    1,   -1,    0,    1,   -1,    0,    0,
       0,   -1,    0,    0,    1,   -1,    0,    0,   -1,    0,    0,    0,    1,   -1,    0,    0,
       0,    0,    0,    0,    1,   -1,    0,    0,    1,    0,    0,    0,    1,   -1,    0,    0,
       0,    1,    0,    0,    1,   -1,    0,    0,    1,   -1,    1,    0,    1,   -1,    0,    0,
       0,    0,    1,    0,    1,   -1,    0,    0,    0,    0,   -1,    1,    1,   -1,    0,    0,
      -1,   -1,    0,    1,    1,   -1,    0,    0,    1,   -1,    0,    1,    1,   -1,    0,    0,
       0,    0,    0,    1,    1,   -1,    0,    0,    0,    1,    0,    1,    1,   -1,    0,    0,
       1,    1,    0,    1,    1,   -1,    0,    0,   -1,    0,    1,    1,    1,   -1,    0,    0,
      -1,    0,   -1,   -1,   -1,    0,    0,    0,    0,    0,   -1,   -1,   -1,    0,    0,    0,
       0,   -1,    0,   -1,   -1,    0,    0,    0,   -1,    0,    0,   -1,   -1,    0,    0,    0,
       0,    0,    0,   -1,   -1,    0,    0,    0,    1,    0,    0,   -1,   -1,    0,    0,    0,
      -1,    1,    0,   -1,   -1,    0,    0,    0,    0,    1,    0,   -1,   -1,    0,    0,    0,
       0,   -1,    1,   -1,   -1,    0,    0,    0,    0,    0,    1,   -1,   -1,    0,    0,    0,
      -1,    1,    1,   -1,   -1,    0,    0,    0,    1,    1,    1,   -1,   -1,    0,    0,    0,
       0,   -1,   -1,    0,   -1,    0,    0,    0,   -1,    0,   -1,    0,   -1,    0,    0,    0,
       0,    0,   -1,    0,   -1,    0,    0,    0,    1,    0,   -1,    0,   -1,    0,    0,    0,
       0,    1,   -1,    0,   -1,    0,    0,    0,    1,    1,   -1,    0,   -1,    0,    0,    0,
      -1,   -1,    0,    0,   -1,    0,    0,    0,    0,   -1,    0,    0,   -1,    0,    0,    0,
      -1,    0,    0,    0,   -1,    0,    0,    0,    0,    0,    0,    0,   -1,    0,    0,    0,
       1,    0,    0,    0,   -1,    0,    0,    0,   -1,    1,    0,    0,   -1,    0,    0,    0,
       0,    1,    0,    0,   -1,    0,    0,    0,    1,    1,    0,    0,   -1,    0,    0,    0,
       0,   -1,    1,    0,   -1,    0,    0,    0,    1,   -1,    1,    0,   -1,    0,    0,    0,
      -1,    0,    1,    0,   -1,    0,    0,    0,    0,    0,    1,    0,   -1,    0,    0,    0,
       1,    0,    1,    0,   -1,    0,    0,    0,    0,    1,    1,    0,   -1,    0,    0,    0,
      -1,   -1,   -1,    1,   -1,    0,    0,    0,   -1,    1,   -1,    1,   -1,    0,    0,    0,
       1,    1,   -1,    1,   -1,    0,    0,    0,    0,   -1,    0,    1,   -1,    0,    0,    0,
      -1,    0,    0,    1,   -1,    0,    0,    0,    0,    0,    0,    1,   -1,    0,    0,    0,
       1,    0,    0,    1,   -1,    0,    0,    0,    0,    1,    0,    1,   -1,    0,    0,    0,
       0,   -1,    1,    1,   -1,    0,    0,    0,   -1,    0,    1,    1,   -1,    0,    0,    0,
       0,    0,    1,    1,   -1,    0,    0,    0,    1,    1,    1,    1,   -1,    0,    0,    0,
       0,   -1,   -1,   -1,    0,    0,    0,    0,    1,   -1,   -1,   -1,    0,    0,    0,    0,
      -1,    0,   -1,   -1,    0,    0,    0,    0,    0,    0,   -1,   -1,    0,    0,    0,    0,
       1,    0,   -1,   -1,    0,    0,    0,    0,    0,    1,   -1,   -1,    0,    0,    0,    0,
      -1,   -1,    0,   -1,    0,    0,    0,    0,    0,   -1,    0,   -1,    0,    0,    0,    0,
       1,   -1,    0,   -1,    0,    0,    0,    0,   -1,    0,    0,   -1,    0,    0,    0,    0,
       0,    0,    0,   -1,    0,    0,    0,    0,    1,    0,    0,   -1,    0,    0,    0,    0,
       0,    1,    0,   -1,    0,    0,    0,    0,    1,    1,    0,   -1,    0,    0,    0,    0,
       0,   -1,    1,   -1,    0,    0,    0,    0,   -1,    0,    1,   -1,    0,    0,    0,    0,
       0,    0,    1,   -1,    0,    0,    0,    0,    1,    0,    1,   -1,    0,    0,    0,    0,
       0,    1,    1,   -1,    0,    0,    0,    0,   -1,   -1,   -1,    0,    0,    0,    0,    0,
       0,   -1,   -1,    0,    0,    0,    0,    0,    1,   -1,   -1,    0,    0,    0,    0,    0,
      -1,    0,   -1,    0,    0,    0,    0,    0,    0,    0,   -1,    0,    0,    0,    0,    0,
       1,    0,   -1,    0,    0,    0,    0,    0,   -1,    1,   -1,    0,    0,    0,    0,    0,
       0,    1,   -1,    0,    0,    0,    0,    0,   -1,   -1,    0,    0,    0,    0,    0,    0,
       0,   -1,    0,    0,    0,    0,    0,    0,    1,   -1,    0,    0,    0,    0,    0,    0,
      -1,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
       1,    0,    0,    0,    0,    0,    0,    0,   -1,    1,    0,    0,    0,    0,    0,    0,
       0,    1,    0,    0,    0,    0,    0,    0,    1,    1,    0,    0,    0,    0,    0,    0,
      -1,   -1,    1,    0,    0,    0,    0,    0,    0,   -1,    1,    0,    0,    0,    0,    0,
      -1,    0,    1,    0,    0,    0,    0,    0,    0,    0,    1,    0,    0,    0,    0,    0,
       1,    0,    1,    0,    0,    0,    0,    0,   -1,    1,    1,    0,    0,    0,    0,    0,
       0,    1,    1,    0,    0,    0,    0,    0,    1,    1,    1,    0,    0,    0,    0,    0,
       0,   -1,   -1,    1,    0,    0,    0,    0,   -1,    0,   -1,    1,    0,    0,    0,    0,
       0,    0,   -1,    1,    0,    0,    0,    0,    0,    1,   -1,    1,    0,    0,    0,    0,
       1,    1,   -1,    1,    0,    0,    0,    0,   -1,   -1,    0,    1,    0,    0,    0,    0,
       0,   -1,    0,    1,    0,    0,    0,    0,   -1,    0,    0,    1,    0,    0,    0,    0,
       0,    0,    0,    1,    0,    0,    0,    0,    1,    0,    0,    1,    0,    0,    0,    0,
      -1,    1,    0,    1,    0,    0,    0,    0,    0,    1,    0,    1,    0,    0,    0,    0,
       0,   -1,    1,    1,    0,    0,    0,    0,   -1,    0,    1,    1,    0,    0,    0,    0,
       0,    0,    1,    1,    0,    0,    0,    0,    1,    0,    1,    1,    0,    0,    0,    0,
       0,    1,    1,    1,    0,    0,    0,    0,   -1,   -1,   -1,   -1,    1,    0,    0,    0,
       0,   -1,   -1,   -1,    1,    0,    0,    0,    1,   -1,   -1,   -1,    1,    0,    0,    0,
      -1,    0,   -1,   -1,    1,    0,    0,    0,    1,    0,   -1,   -1,    1,    0,    0,    0,
      -1,    1,   -1,   -1,    1,    0,    0,    0,    0,    1,   -1,   -1,    1,    0,    0,    0,
       0,   -1,    0,   -1,    1,    0,    0,    0,   -1,    0,    0,   -1,    1,    0,    0,    0,
       0,    0,    0,   -1,    1,    0,    0,    0,   -1,    1,    0,   -1,    1,    0,    0,    0,
       0,    1,    0,   -1,    1,    0,    0,    0,   -1,   -1,    1,   -1,    1,    0,    0,    0,
       0,   -1,    1,   -1,    1,    0,    0,    0,    1,   -1,    1,   -1,    1,    0,    0,    0,
      -1,    0,    1,   -1,    1,    0,    0,    0,    0,    0,    1,   -1,    1,    0,    0,    0,
       1,    0,    1,   -1,    1,    0,    0,    0,   -1,    1,    1,   -1,    1,    0,    0,    0,
       0,    1,    1,   -1,    1,    0,    0,    0,    0,   -1,   -1,    0,    1,    0,    0,    0,
       0,    0,   -1,    0,    1,    0,    0,    0,    1,    0,   -1,    0,    1,    0,    0,    0,
      -1,    1,   -1,    0,    1,    0,    0,    0,    0,    1,   -1,    0,    1,    0,    0,    0,
       1,    1,   -1,    0,    1,    0,    0,    0,   -1,   -1,    0,    0,    1,    0,    0,    0,
       0,   -1,    0,    0,    1,    0,    0,    0,    1,   -1,    0,    0,    1,    0,    0,    0,
      -1,    0,    0,    0,    1,    0,    0,    0,    0,    0,    0,    0,    1,    0,    0,    0,
       1,    0,    0,    0,    1,    0,    0,    0,   -1,    1,    0,    0,    1,    0,    0,    0,
       0,    1,    0,    0,    1,    0,    0,    0,    1,    1,    0,    0,    1,    0,    0,    0,
       0,   -1,    1,    0,    1,    0,    0,    0,   -1,    0,    1,    0,    1,    0,    0,    0,
       0,    0,    1,    0,    1,    0,    0,    0,    0,    1,    1,    0,    1,    0,    0,    0,
       1,   -1,   -1,    1,    1,    0,    0,    0,    0,    0,   -1,    1,    1,    0,    0,    0,
       1,    0,   -1,    1,    1,    0,    0,    0,   -1,    1,   -1,    1,    1,    0,    0,    0,
       0,    1,   -1,    1,    1,    0,    0,    0,    1,    1,   -1,    1,    1,    0,    0,    0,
       0,   -1,    0,    1,    1,    0,    0,    0,    0,    0,    0,    1,    1,    0,    0,    0,
       1,    1,    0,    1,    1,    0,    0,    0,    1,   -1,    1,    1,    1,    0,    0,    0,
       0,    0,    1,    1,    1,    0,    0,    0,    1,    0,    1,    1,    1,    0,    0,    0,
      -1,    1,    1,    1,    1,    0,    0,    0,    0,    1,    1,    1,    1,    0,    0,    0,
      -1,    0,   -1,   -1,   -1,    1,    0,    0,    0,    0,   -1,   -1,   -1,    1,    0,    0,
       1,    0,   -1,   -1,   -1,    1,    0,    0,    0,    1,   -1,   -1,   -1,    1,    0,    0,
      -1,   -1,    0,   -1,   -1,    1,    0,    0,    0,    0,    0,   -1,   -1,    1,    0,    0,
      -1,    1,    0,   -1,   -1,    1,    0,    0,    0,   -1,    1,   -1,   -1,    1,    0,    0,
       1,    1,    1,   -1,   -1,    1,    0,    0,    0,    0,   -1,    0,   -1,    1,    0,    0,
      -1,    1,   -1,    0,   -1,    1,    0,    0,    1,    1,   -1,    0,   -1,    1,    0,    0,
       0,   -1,    0,    0,   -1,    1,    0,    0,   -1,    0,    0,    0,   -1,    1,    0,    0,
       0,    0,    0,    0,   -1,    1,    0,    0,    1,    0,    0,    0,   -1,    1,    0,    0,
      -1,    1,    0,    0,   -1,    1,    0,    0,    0,    1,    0,    0,   -1,    1,    0,    0,
      -1,   -1,    1,    0,   -1,    1,    0,    0,    1,   -1,    1,    0,   -1,    1,    0,    0,
      -1,    0,    1,    0,   -1,    1,    0,    0,    0,    0,    1,    0,   -1,    1,    0,    0,
       1,   -1,   -1,    1,   -1,    1,    0,    0,    0,    1,   -1,    1,   -1,    1,    0,    0,
      -1,   -1,    0,    1,   -1,    1,    0,    0,    1,   -1,    0,    1,   -1,    1,    0,    0,
       0,    0,    0,    1,   -1,    1,    0,    0,   -1,    1,    0,    1,   -1,    1,    0,    0,
       0,   -1,    1,    1,   -1,    1,    0,    0,    0,    1,    1,    1,   -1,    1,    0,    0,
       0,   -1,   -1,   -1,    0,    1,    0,    0,    1,   -1,   -1,   -1,    0,    1,    0,    0,
       0,    0,   -1,   -1,    0,    1,    0,    0,    1,    1,   -1,   -1,    0,    1,    0,    0,
       0,   -1,    0,   -1,    0,    1,    0,    0,   -1,    0,    0,   -1,    0,    1,    0,    0,
       0,    0,    0,   -1,    0,    1,    0,    0,    1,    0,    0,   -1,    0,    1,    0,    0,
       0,    1,    0,   -1,    0,    1,    0,    0,    0,    0,    1,   -1,    0,    1,    0,    0,
       0,   -1,   -1,    0,    0,    1,    0,    0,   -1,    0,   -1,    0,    0,    1,    0,    0,
       0,    0,   -1,    0,    0,    1,    0,    0,    1,    0,   -1,    0,    0,    1,    0,    0,
       0,    1,   -1,    0,    0,    1,    0,    0,   -1,   -1,    0,    0,    0,    1,    0,    0,
       0,   -1,    0,    0,    0,    1,    0,    0,    1,   -1,    0,    0,    0,    1,    0,    0,
      -1,    0,    0,    0,    0,    1,    0,    0,    0,    0,    0,    0,    0,    1,    0,    0,
       1,    0,    0,    0,    0,    1,    0,    0,   -1,    1,    0,    0,    0,    1,    0,    0,
       0,    1,    0,    0,    0,    1,    0,    0,    1,    1,    0,    0,    0,    1,    0,    0,
       0,   -1,    1,    0,    0,    1,    0,    0,   -1,    0,    1,    0,    0,    1,    0,    0,
       0,    0,    1,    0,    0,    1,    0,    0,    1,    0,    1,    0,    0,    1,    0,    0,
       0,    1,    1,    0,    0,    1,    0,    0,    0,   -1,   -1,    1,    0,    1,    0,    0,
       0,    0,   -1,    1,    0,    1,    0,    0,    0,    1,   -1,    1,    0,    1,    0,    0,
       0,   -1,    0,    1,    0,    1,    0,    0,   -1,    0,    0,    1,    0,    1,    0,    0,
       0,    0,    0,    1,    0,    1,    0,    0,    1,    0,    0,    1,    0,    1,    0,    0,
      -1,    1,    0,    1,    0,    1,    0,    0,    0,    1,    0,    1,    0,    1,    0,    0,
       0,    0,    1,    1,    0,    1,    0,    0,   -1,    0,   -1,   -1,    1,    1,    0,    0,
      -1,    1,   -1,   -1,    1,    1,    0,    0,    0,    0,    0,   -1,    1,    1,    0,    0,
       1,    1,    0,   -1,    1,    1,    0,    0,   -1,   -1,    1,   -1,    1,    1,    0,    0,
       0,    0,    1,   -1,    1,    1,    0,    0,    1,    0,    1,   -1,    1,    1,    0,    0,
       0,    1,    1,   -1,    1,    1,    0,    0,    0,    0,   -1,    0,    1,    1,    0,    0,
      -1,    1,   -1,    0,    1,    1,    0,    0,    0,    1,   -1,    0,    1,    1,    0,    0,
       0,   -1,    0,    0,    1,    1,    0,    0,    0,    0,    0,    0,    1,    1,    0,    0,
       1,    0,    0,    0,    1,    1,    0,    0,   -1,    1,    0,    0,    1,    1,    0,    0,
       0,    1,    0,    0,    1,    1,    0,    0,    1,   -1,    1,    0,    1,    1,    0,    0,
       0,    0,    1,    0,    1,    1,    0,    0,   -1,    1,    1,    0,    1,    1,    0,    0,
       1,    1,    1,    0,    1,    1,    0,    0,    0,   -1,   -1,    1,    1,    1,    0,    0,
       1,    1,   -1,    1,    1,    1,    0,    0,    1,   -1,    0,    1,    1,    1,    0,    0,
       0,    0,    0,    1,    1,    1,    0,    0,    1,    0,    0,    1,    1,    1,    0,    0,
      -1,    1,    0,    1,    1,    1,    0,    0,    1,    1,    0,    1,    1,    1,    0,    0,
       0,   -1,    1,    1,    1,    1,    0,    0,    0,    0,   -1,   -1,   -1,   -1,    1,    0,
      -1,    0,    0,   -1,   -1,   -1,    1,    0,    1,    0,    0,   -1,   -1,   -1,    1,    0,
       0,    1,    0,   -1,   -1,   -1,    1,    0,    0,    0,    1,   -1,   -1,   -1,    1,    0,
      -1,    0,   -1,    0,   -1,   -1,    1,    0,   -1,   -1,    0,    0,   -1,   -1,    1,    0,
       0,    0,    0,    0,   -1,   -1,    1,    0,    1,    0,    0,    0,   -1,   -1,    1,    0,
      -1,    1,    0,    0,   -1,   -1,    1,    0,    1,    1,    0,    0,   -1,   -1,    1,    0,
       0,   -1,    1,    0,   -1,   -1,    1,    0,   -1,    0,    1,    0,   -1,   -1,    1,    0,
       1,    0,    1,    0,   -1,   -1,    1,    0,    0,    1,    1,    0,   -1,   -1,    1,    0,
       0,   -1,   -1,    1,   -1,   -1,    1,    0,    1,    0,    0,    1,   -1,   -1,    1,    0,
       0,    0,    1,    1,   -1,   -1,    1,    0,    0,   -1,   -1,   -1,    0,   -1,    1,    0,
      -1,    0,   -1,   -1,    0,   -1,    1,    0,    1,    0,   -1,   -1,    0,   -1,    1,    0,
       0,    1,   -1,   -1,    0,   -1,    1,    0,    1,   -1,    0,   -1,    0,   -1,    1,    0,
       0,    0,    0,   -1,    0,   -1,    1,    0,    0,   -1,    1,   -1,    0,   -1,    1,    0,
       1,   -1,    1,   -1,    0,   -1,    1,    0,    1,    0,    1,   -1,    0,   -1,    1,    0,
       0,    1,    1,   -1,    0,   -1,    1,    0,    0,    0,   -1,    0,    0,   -1,    1,    0,
       0,    1,   -1,    0,    0,   -1,    1,    0,    0,   -1,    0,    0,    0,   -1,    1,    0,
       0,    0,    0,    0,    0,   -1,    1,    0,    1,    0,    0,    0,    0,   -1,    1,    0,
       0,    1,    0,    0,    0,   -1,    1,    0,    0,    0,    1,    0,    0,   -1,    1,    0,
       1,    0,    1,    0,    0,   -1,    1,    0,    1,    1,    1,    0,    0,   -1,    1,    0,
      -1,    0,   -1,    1,    0,   -1,    1,    0,    1,    1,   -1,    1,    0,   -1,    1,    0,
       1,   -1,    0,    1,    0,   -1,    1,    0,    0,    0,    0,    1,    0,   -1,    1,    0,
       0,   -1,    1,    1,    0,   -1,    1,    0,    1,    0,    1,    1,    0,   -1,    1,    0,
       0,    1,    1,    1,    0,   -1,    1,    0,    0,   -1,    0,   -1,    1,   -1,    1,    0,
       1,    0,    0,   -1,    1,   -1,    1,    0,    0,    1,    0,   -1,    1,   -1,    1,    0,
      -1,   -1,   -1,    0,    1,   -1,    1,    0,    0,   -1,   -1,    0,    1,   -1,    1,    0,
       1,    0,   -1,    0,    1,   -1,    1,    0,    0,    0,    0,    0,    1,   -1,    1,    0,
       1,    0,    0,    0,    1,   -1,    1,    0,   -1,    1,    0,    0,    1,   -1,    1,    0,
      -1,   -1,    1,    0,    1,   -1,    1,    0,    0,    0,   -1,    1,    1,   -1,    1,    0,
       0,   -1,    0,    1,    1,   -1,    1,    0,    1,    0,    0,    1,    1,   -1,    1,    0,
       0,    0,    1,    1,    1,   -1,    1,    0,    0,   -1,    0,   -1,   -1,    0,    1,    0,
       1,   -1,    0,   -1,   -1,    0,    1,    0,    0,    0,    0,   -1,   -1,    0,    1,    0,
       1,    0,    0,   -1,   -1,    0,    1,    0,    1,    1,    0,   -1,   -1,    0,    1,    0,
       0,   -1,    1,   -1,   -1,    0,    1,    0,    1,    0,    1,   -1,   -1,    0,    1,    0,
       0,    1,    1,   -1,   -1,    0,    1,    0,   -1,   -1,   -1,    0,   -1,    0,    1,    0,
       1,   -1,   -1,    0,   -1,    0,    1,    0,    0,    0,   -1,    0,   -1,    0,    1,    0,
      -1,    1,   -1,    0,   -1,    0,    1,    0,    1,    1,   -1,    0,   -1,    0,    1,    0,
       0,   -1,    0,    0,   -1,    0,    1,    0,   -1,    0,    0,    0,   -1,    0,    1,    0,
       0,    0,    0,    0,   -1,    0,    1,    0,    1,    0,    0,    0,   -1,    0,    1,    0,
       0,    1,    0,    0,   -1,    0,    1,    0,    1,    1,    0,    0,   -1,    0,    1,    0,
      -1,   -1,    1,    0,   -1,    0,    1,    0,    1,   -1,    1,    0,   -1,    0,    1,    0,
       0,    0,    1,    0,   -1,    0,    1,    0,   -1,    0,   -1,    1,   -1,    0,    1,    0,
       0,    0,   -1,    1,   -1,    0,    1,    0,    0,    1,   -1,    1,   -1,    0,    1,    0,
      -1,   -1,    0,    1,   -1,    0,    1,    0,    1,   -1,    0,    1,   -1,    0,    1,    0,
      -1,    0,    0,    1,   -1,    0,    1,    0,    0,    0,    0,    1,   -1,    0,    1,    0,
       1,    0,    0,    1,   -1,    0,    1,    0,   -1,    1,    0,    1,   -1,    0,    1,    0,
       1,    1,    0,    1,   -1,    0,    1,    0,    0,   -1,    1,    1,   -1,    0,    1,    0,
      -1,    0,    1,    1,   -1,    0,    1,    0,    0,    1,    1,    1,   -1,    0,    1,    0,
       0,    0,   -1,   -1,    0,    0,    1,    0,   -1,    1,   -1,   -1,    0,    0,    1,    0,
       1,    1,   -1,   -1,    0,    0,    1,    0,    0,   -1,    0,   -1,    0,    0,    1,    0,
       0,    0,    0,   -1,    0,    0,    1,    0,    1,    0,    0,   -1,    0,    0,    1,    0,
       0,    1,    0,   -1,    0,    0,    1,    0,   -1,    0,   -1,    0,    0,    0,    1,    0,
       0,    0,   -1,    0,    0,    0,    1,    0,    1,    0,   -1,    0,    0,    0,    1,    0,
       0,    1,   -1,    0,    0,    0,    1,    0,   -1,   -1,    0,    0,    0,    0,    1,    0,
       0,   -1,    0,    0,    0,    0,    1,    0,   -1,    0,    0,    0,    0,    0,    1,    0,
       0,    0,    0,    0,    0,    0,    1,    0,    1,    0,    0,    0,    0,    0,    1,    0,
       0,    1,    0,    0,    0,    0,    1,    0,    0,   -1,    1,    0,    0,    0,    1,    0,
      -1,    0,    1,    0,    0,    0,    1,    0,    0,    0,    1,    0,    0,    0,    1,    0,
       1,    0,    1,    0,    0,    0,    1,    0,    0,    1,    1,    0,    0,    0,    1,    0,
       1,    0,   -1,    1,    0,    0,    1,    0,    0,    1,   -1,    1,    0,    0,    1,    0,
       1,    1,   -1,    1,    0,    0,    1,    0,    0,   -1,    0,    1,    0,    0,    1,    0,
       0,    0,    0,    1,    0,    0,    1,    0,    1,    0,    0,    1,    0,    0,    1,    0,
       0,    1,    0,    1,    0,    0,    1,    0,    1,    1,    0,    1,    0,    0,    1,    0,
       1,   -1,    1,    1,    0,    0,    1,    0,    0,    0,    1,    1,    0,    0,    1,    0,
       1,    0,    1,    1,    0,    0,    1,    0,   -1,    1,    1,    1,    0,    0,    1,    0,
       1,   -1,   -1,   -1,    1,    0,    1,    0,    0,    1,   -1,   -1,    1,    0,    1,    0,
       0,    0,    0,   -1,    1,    0,    1,    0,   -1,   -1,    1,   -1,    1,    0,    1,    0,
       1,    0,    1,   -1,    1,    0,    1,    0,   -1,    1,    1,   -1,    1,    0,    1,    0,
       0,    1,    1,   -1,    1,    0,    1,    0,   -1,   -1,   -1,    0,    1,    0,    1,    0,
       0,    0,   -1,    0,    1,    0,    1,    0,   -1,    1,   -1,    0,    1,    0,    1,    0,
       1,    1,   -1,    0,    1,    0,    1,    0,    0,   -1,    0,    0,    1,    0,    1,    0,
      -1,    0,    0,    0,    1,    0,    1,    0,    0,    0,    0,    0,    1,    0,    1,    0,
       1,    0,    0,    0,    1,    0,    1,    0,   -1,    1,    0,    0,    1,    0,    1,    0,
       1,    1,    0,    0,    1,    0,    1,    0,   -1,   -1,    1,    0,    1,    0,    1,    0,
       0,    0,    1,    0,    1,    0,    1,    0,   -1,    1,    1,    0,    1,    0,    1,    0,
      -1,   -1,   -1,    1,    1,    0,    1,    0,    1,   -1,   -1,    1,    1,    0,    1,    0,
       0,    0,   -1,    1,    1,    0,    1,    0,    1,    1,   -1,    1,    1,    0,    1,    0,
      -1,    0,    0,    1,    1,    0,    1,    0,    1,    0,    0,    1,    1,    0,    1,    0,
      -1,    1,    0,    1,    1,    0,    1,    0,    0,    1,    0,    1,    1,    0,    1,    0,
      -1,   -1,    1,    1,    1,    0,    1,    0,   -1,    0,    1,    1,    1,    0,    1,    0,
       1,    0,    1,    1,    1,    0,    1,    0,    1,    1,    1,    1,    1,    0,    1,    0,
       1,    0,    0,   -1,   -1,    1,    1,    0,    0,    1,    0,   -1,   -1,    1,    1,    0,
       0,    0,    1,   -1,   -1,    1,    1,    0,    0,   -1,   -1,    0,   -1,    1,    1,    0,
       1,   -1,    0,    0,   -1,    1,    1,    0,    0,    0,    0,    0,   -1,    1,    1,    0,
       1,    1,    0,    0,   -1,    1,    1,    0,    0,   -1,    1,    0,   -1,    1,    1,    0,
       0,    1,    1,    0,   -1,    1,    1,    0,    0,    0,   -1,    1,   -1,    1,    1,    0,
       0,   -1,    0,    1,   -1,    1,    1,    0,   -1,    1,    0,    1,   -1,    1,    1,    0,
       1,    0,    1,    1,   -1,    1,    1,    0,    0,   -1,   -1,   -1,    0,    1,    1,    0,
      -1,    0,   -1,   -1,    0,    1,    1,    0,   -1,   -1,    0,   -1,    0,    1,    1,    0,
       0,    0,    0,   -1,    0,    1,    1,    0,    0,   -1,    1,   -1,    0,    1,    1,    0,
      -1,    0,    1,   -1,    0,    1,    1,    0,    1,    0,    1,   -1,    0,    1,    1,    0,
       0,    1,    1,   -1,    0,    1,    1,    0,   -1,   -1,   -1,    0,    0,    1,    1,    0,
       0,   -1,   -1,    0,    0,    1,    1,    0,    0,    0,   -1,    0,    0,    1,    1,    0,
       1,    0,   -1,    0,    0,    1,    1,    0,   -1,    1,   -1,    0,    0,    1,    1,    0,
       0,   -1,    0,    0,    0,    1,    1,    0,   -1,    0,    0,    0,    0,    1,    1,    0,
       0,    0,    0,    0,    0,    1,    1,    0,    1,    0,    0,    0,    0,    1,    1,    0,
       0,    1,    0,    0,    0,    1,    1,    0,   -1,   -1,    1,    0,    0,    1,    1,    0,
       0,    0,    1,    0,    0,    1,    1,    0,    1,    1,    1,    0,    0,    1,    1,    0,
       1,   -1,   -1,    1,    0,    1,    1,    0,   -1,    0,   -1,    1,    0,    1,    1,    0,
       1,    1,   -1,    1,    0,    1,    1,    0,    0,    0,    0,    1,    0,    1,    1,    0,
       0,   -1,    1,    1,    0,    1,    1,    0,   -1,    0,    1,    1,    0,    1,    1,    0,
       0,    0,    1,    1,    0,    1,    1,    0,    0,    1,    1,    1,    0,    1,    1,    0,
       0,   -1,    0,   -1,    1,    1,    1,    0,    1,    0,    0,   -1,    1,    1,    1,    0,
      -1,    1,    0,   -1,    1,    1,    1,    0,    0,   -1,   -1,    0,    1,    1,    1,    0,
      -1,    0,   -1,    0,    1,    1,    1,    0,    0,    1,   -1,    0,    1,    1,    1,    0,
      -1,   -1,    0,    0,    1,    1,    1,    0,    0,    0,    0,    0,    1,    1,    1,    0,
      -1,    1,    0,    0,    1,    1,    1,    0,    1,    1,    0,    0,    1,    1,    1,    0,
      -1,    0,    1,    0,    1,    1,    1,    0,    0,    0,    1,    0,    1,    1,    1,    0,
       0,    1,    1,    0,    1,    1,    1,    0,    1,    0,   -1,    1,    1,    1,    1,    0,
      -1,    0,    0,    1,    1,    1,    1,    0,   -1,    1,    0,    1,    1,    1,    1,    0,
       1,    1,    0,    1,    1,    1,    1,    0,    1,    0,    1,    1,    1,    1,    1,    0,
      -1,   -1,   -1,   -1,   -1,   -1,   -1,    1,    1,   -1,   -1,   -1,   -1,   -1,   -1,    1,
      -1,    1,   -1,   -1,   -1,   -1,   -1,    1,    1,    1,   -1,   -1,   -1,   -1,   -1,    1,
      -1,   -1,    1,   -1,   -1,   -1,   -1,    1,    1,   -1,    1,   -1,   -1,   -1,   -1,    1,
      -1,    1,    1,   -1,   -1,   -1,   -1,    1,    1,    1,    1,   -1,   -1,   -1,   -1,    1,
       0,    0,   -1,    0,   -1,   -1,   -1,    1,   -1,   -1,    0,    0,   -1,   -1,   -1,    1,
       0,   -1,    0,    0,   -1,   -1,   -1,    1,   -1,    0,    0,    0,   -1,   -1,   -1,    1,
       1,    0,    0,    0,   -1,   -1,   -1,    1,    0,    1,    0,    0,   -1,   -1,   -1,    1,
       0,    0,    1,    0,   -1,   -1,   -1,    1,   -1,   -1,   -1,    1,   -1,   -1,   -1,    1,
       1,   -1,   -1,    1,   -1,   -1,   -1,    1,   -1,    1,   -1,    1,   -1,   -1,   -1,    1,
       1,    1,   -1,    1,   -1,   -1,   -1,    1,    0,    0,    0,    1,   -1,   -1,   -1,    1,
      -1,   -1,    1,    1,   -1,   -1,   -1,    1,    1,   -1,    1,    1,   -1,   -1,   -1,    1,
      -1,    1,    1,    1,   -1,   -1,   -1,    1,    1,    1,    1,    1,   -1,   -1,   -1,    1,
       0,    0,   -1,   -1,    0,   -1,   -1,    1,    0,   -1,    0,   -1,    0,   -1,   -1,    1,
      -1,    0,    0,   -1,    0,   -1,   -1,    1,    1,    0,    0,   -1,    0,   -1,   -1,    1,
       0,    1,    0,   -1,    0,   -1,   -1,    1,    0,    0,    1,   -1,    0,   -1,   -1,    1,
       0,   -1,   -1,    0,    0,   -1,   -1,    1,   -1,    0,   -1,    0,    0,   -1,   -1,    1,
       0,    1,   -1,    0,    0,   -1,   -1,    1,   -1,   -1,    0,    0,    0,   -1,   -1,    1,
       1,   -1,    0,    0,    0,   -1,   -1,    1,    0,    0,    0,    0,    0,   -1,   -1,    1,
       1,    0,    0,    0,    0,   -1,   -1,    1,   -1,    1,    0,    0,    0,   -1,   -1,    1,
       0,    1,    0,    0,    0,   -1,   -1,    1,   -1,    0,    1,    0,    0,   -1,   -1,    1,
       1,    0,    1,    0,    0,   -1,   -1,    1,    0,    1,    1,    0,    0,   -1,   -1,    1,
       0,    0,   -1,    1,    0,   -1,   -1,    1,    0,    1,   -1,    1,    0,   -1,   -1,    1,
      -1,    0,    0,    1,    0,   -1,   -1,    1,    1,    0,    0,    1,    0,   -1,   -1,    1,
       0,    1,    0,    1,    0,   -1,   -1,    1,    0,    0,    1,    1,    0,   -1,   -1,    1,
      -1,   -1,   -1,   -1,    1,   -1,   -1,    1,    1,   -1,   -1,   -1,    1,   -1,   -1,    1,
      -1,    1,   -1,   -1,    1,   -1,   -1,    1,    1,    1,   -1,   -1,    1,   -1,   -1,    1,
       0,    0,    0,   -1,    1,   -1,   -1,    1,   -1,   -1,    1,   -1,    1,   -1,   -1,    1,
       1,   -1,    1,   -1,    1,   -1,   -1,    1,   -1,    1,    1,   -1,    1,   -1,   -1,    1,
       1,    1,    1,   -1,    1,   -1,   -1,    1,    0,   -1,    0,    0,    1,   -1,   -1,    1,
      -1,    0,    0,    0,    1,   -1,   -1,    1,    0,    1,    0,    0,    1,   -1,   -1,    1,
       0,    0,    1,    0,    1,   -1,   -1,    1,   -1,   -1,   -1,    1,    1,   -1,   -1,    1,
       1,   -1,   -1,    1,    1,   -1,   -1,    1,   -1,    1,   -1,    1,    1,   -1,   -1,    1,
       1,    1,   -1,    1,    1,   -1,   -1,    1,    0,    0,    0,    1,    1,   -1,   -1,    1,
      -1,   -1,    1,    1,    1,   -1,   -1,    1,    1,   -1,    1,    1,    1,   -1,   -1,    1,
      -1,    1,    1,    1,    1,   -1,   -1,    1,    1,    1,    1,    1,    1,   -1,   -1,    1,
      -1,    0,    0,   -1,   -1,    0,   -1,    1,    0,    1,    0,   -1,   -1,    0,   -1,    1,
       0,   -1,   -1,    0,   -1,    0,   -1,    1,   -1,    0,   -1,    0,   -1,    0,   -1,    1,
       0,   -1,    0,    0,   -1,    0,   -1,    1,    0,    0,    0,    0,   -1,    0,   -1,    1,
       1,    1,    0,    0,   -1,    0,   -1,    1,    0,   -1,    1,    0,   -1,    0,   -1,    1,
      -1,    0,    1,    0,   -1,    0,   -1,    1,    0,    1,    1,    0,   -1,    0,   -1,    1,
      -1,    0,    0,    1,   -1,    0,   -1,    1,    0,    1,    0,    1,   -1,    0,   -1,    1,
       0,   -1,   -1,   -1,    0,    0,   -1,    1,    0,    1,   -1,   -1,    0,    0,   -1,    1,
       1,   -1,    0,   -1,    0,    0,   -1,    1,    0,    0,    0,   -1,    0,    0,   -1,    1,
       1,    1,    0,   -1,    0,    0,   -1,    1,    1,    0,    1,   -1,    0,    0,   -1,    1,
       0,    1,    1,   -1,    0,    0,   -1,    1,   -1,   -1,   -1,    0,    0,    0,   -1,    1,
       0,   -1,   -1,    0,    0,    0,   -1,    1,    0,    0,   -1,    0,    0,    0,   -1,    1,
      -1,    1,   -1,    0,    0,    0,   -1,    1,    0,   -1,    0,    0,    0,    0,   -1,    1,
      -1,    0,    0,    0,    0,    0,   -1,    1,    0,    0,    0,    0,    0,    0,   -1,    1,
       1,    0,    0,    0,    0,    0,   -1,    1,    0,    1,    0,    0,    0,    0,   -1,    1,
       1,    1,    0,    0,    0,    0,   -1,    1,    0,    0,    1,    0,    0,    0,   -1,    1,
       1,    0,    1,    0,    0,    0,   -1,    1,   -1,    1,    1,    0,    0,    0,   -1,    1,
       1,    1,    1,    0,    0,    0,   -1,    1,    0,   -1,   -1,    1,    0,    0,   -1,    1,
      -1,    0,   -1,    1,    0,    0,   -1,    1,    1,    0,   -1,    1,    0,    0,   -1,    1,
       0,    1,   -1,    1,    0,    0,   -1,    1,   -1,   -1,    0,    1,    0,    0,   -1,    1,
       1,   -1,    0,    1,    0,    0,   -1,    1,    0,    0,    0,    1,    0,    0,   -1,    1,
      -1,    1,    0,    1,    0,    0,   -1,    1,    1,    0,    1,    1,    0,    0,   -1,    1,
       0,   -1,    0,   -1,    1,    0,   -1,    1,    1,    0,    0,   -1,    1,    0,   -1,    1,
       0,    1,    0,   -1,    1,    0,   -1,    1,    0,    0,    1,   -1,    1,    0,   -1,    1,
       0,   -1,   -1,    0,    1,    0,   -1,    1,   -1,    0,   -1,    0,    1,    0,   -1,    1,
       0,    1,   -1,    0,    1,    0,   -1,    1,    1,    1,   -1,    0,    1,    0,   -1,    1,
      -1,   -1,    0,    0,    1,    0,   -1,    1,    0,    0,    0,    0,    1,    0,   -1,    1,
       0,    1,    0,    0,    1,    0,   -1,    1,    1,    1,    0,    0,    1,    0,   -1,    1,
       0,   -1,    1,    0,    1,    0,   -1,    1,    1,    0,    1,    0,    1,    0,   -1,    1,
       1,    1,    1,    0,    1,    0,   -1,    1,    0,    0,   -1,    1,    1,    0,   -1,    1,
       0,   -1,    0,    1,    1,    0,   -1,    1,    1,    1,    0,    1,    1,    0,   -1,    1,
      -1,    0,    1,    1,    1,    0,   -1,    1,   -1,   -1,   -1,   -1,   -1,    1,   -1,    1,
       1,   -1,   -1,   -1,   -1,    1,   -1,    1,   -1,    1,   -1,   -1,   -1,    1,   -1,    1,
       1,    1,   -1,   -1,   -1,    1,   -1,    1,    0,    0,    0,   -1,   -1,    1,   -1,    1,
      -1,   -1,    1,   -1,   -1,    1,   -1,    1,    1,   -1,    1,   -1,   -1,    1,   -1,    1,
      -1,    1,    1,   -1,   -1,    1,   -1,    1,    1,    1,    1,   -1,   -1,    1,   -1,    1,
       0,   -1,   -1,    0,   -1,    1,   -1,    1,    0,    0,   -1,    0,   -1,    1,   -1,    1,
       0,   -1,    0,    0,   -1,    1,   -1,    1,   -1,    0,    0,    0,   -1,    1,   -1,    1,
       0,    1,    0,    0,   -1,    1,   -1,    1,    0,    0,    1,    0,   -1,    1,   -1,    1,
       0,    1,    1,    0,   -1,    1,   -1,    1,   -1,   -1,   -1,    1,   -1,    1,   -1,    1,
       1,   -1,   -1,    1,   -1,    1,   -1,    1,   -1,    1,   -1,    1,   -1,    1,   -1,    1,
       1,    1,   -1,    1,   -1,    1,   -1,    1,    0,    0,    0,    1,   -1,    1,   -1,    1,
      -1,   -1,    1,    1,   -1,    1,   -1,    1,    1,   -1,    1,    1,   -1,    1,   -1,    1,
      -1,    1,    1,    1,   -1,    1,   -1,    1,    1,    1,    1,    1,   -1,    1,   -1,    1,
       0,    0,   -1,   -1,    0,    1,   -1,    1,    1,    0,   -1,   -1,    0,    1,   -1,    1,
       0,   -1,    0,   -1,    0,    1,   -1,    1,   -1,    0,    0,   -1,    0,    1,   -1,    1,
       1,    0,    0,   -1,    0,    1,   -1,    1,    0,    0,    1,   -1,    0,    1,   -1,    1,
       0,   -1,   -1,    0,    0,    1,   -1,    1,   -1,    0,   -1,    0,    0,    1,   -1,    1,
       1,    0,   -1,    0,    0,    1,   -1,    1,    0,    1,   -1,    0,    0,    1,   -1,    1,
      -1,   -1,    0,    0,    0,    1,   -1,    1,    1,   -1,    0,    0,    0,    1,   -1,    1,
       0,    0,    0,    0,    0,    1,   -1,    1,    1,    1,    0,    0,    0,    1,   -1,    1,
       0,   -1,    1,    0,    0,    1,   -1,    1,   -1,    0,    1,    0,    0,    1,   -1,    1,
       0,    0,   -1,    1,    0,    1,   -1,    1,    1,    0,    0,    1,    0,    1,   -1,    1,
       0,    1,    0,    1,    0,    1,   -1,    1,    0,    0,    1,    1,    0,    1,   -1,    1,
      -1,   -1,   -1,   -1,    1,    1,   -1,    1,    1,   -1,   -1,   -1,    1,    1,   -1,    1,
      -1,    1,   -1,   -1,    1,    1,   -1,    1,    1,    1,   -1,   -1,    1,    1,   -1,    1,
       0,    0,    0,   -1,    1,    1,   -1,    1,   -1,   -1,    1,   -1,    1,    1,   -1,    1,
       1,   -1,    1,   -1,    1,    1,   -1,    1,   -1,    1,    1,   -1,    1,    1,   -1,    1,
       1,    1,    1,   -1,    1,    1,   -1,    1,    0,    0,   -1,    0,    1,    1,   -1,    1,
       0,   -1,    0,    0,    1,    1,   -1,    1,   -1,    0,    0,    0,    1,    1,   -1,    1,
       1,    0,    0,    0,    1,    1,   -1,    1,   -1,   -1,   -1,    1,    1,    1,   -1,    1,
       1,   -1,   -1,    1,    1,    1,   -1,    1,   -1,    1,   -1,    1,    1,    1,   -1,    1,
       1,    1,   -1,    1,    1,    1,   -1,    1,    0,    0,    0,    1,    1,    1,   -1,    1,
      -1,   -1,    1,    1,    1,    1,   -1,    1,    1,   -1,    1,    1,    1,    1,   -1,    1,
      -1,    1,    1,    1,    1,    1,   -1,    1,    1,    1,    1,    1,    1,    1,   -1,    1,
       0,    0,   -1,   -1,   -1,   -1,    0,    1,    0,   -1,    0,   -1,   -1,   -1,    0,    1,
       1,    0,    0,   -1,   -1,   -1,    0,    1,   -1,    1,    0,   -1,   -1,   -1,    0,    1,
       0,    1,    0,   -1,   -1,   -1,    0,    1,    0,    0,    1,   -1,   -1,   -1,    0,    1,
       0,   -1,   -1,    0,   -1,   -1,    0,    1,    1,    0,   -1,    0,   -1,   -1,    0,    1,
       0,    1,   -1,    0,   -1,   -1,    0,    1,    0,    0,    0,    0,   -1,   -1,    0,    1,
      -1,    1,    0,    0,   -1,   -1,    0,    1,    1,    1,    0,    0,   -1,   -1,    0,    1,
       0,    1,    1,    0,   -1,   -1,    0,    1,    1,    1,    1,    0,   -1,   -1,    0,    1,
       0,    0,   -1,    1,   -1,   -1,    0,    1,    0,   -1,    0,    1,   -1,   -1,    0,    1,
      -1,    0,    0,    1,   -1,   -1,    0,    1,    1,    0,    0,    1,   -1,   -1,    0,    1,
       0,    1,    0,    1,   -1,   -1,    0,    1,    0,    0,    1,    1,   -1,   -1,    0,    1,
       0,   -1,   -1,   -1,    0,   -1,    0,    1,   -1,    0,   -1,   -1,    0,   -1,    0,    1,
       1,    0,   -1,   -1,    0,   -1,    0,    1,    0,    1,   -1,   -1,    0,   -1,    0,    1,
      -1,   -1,    0,   -1,    0,   -1,    0,    1,    0,    0,    0,   -1,    0,   -1,    0,    1,
      -1,    1,    0,   -1,    0,   -1,    0,    1,    1,    1,    0,   -1,    0,   -1,    0,    1,
       0,   -1,    1,   -1,    0,   -1,    0,    1,   -1,    0,    1,   -1,    0,   -1,    0,    1,
       1,    0,    1,   -1,    0,   -1,    0,    1,    0,    1,    1,   -1,    0,   -1,    0,    1,
      -1,   -1,   -1,    0,    0,   -1,    0,    1,    0,    0,   -1,    0,    0,   -1,    0,    1,
      -1,   -1,    0,    0,    0,   -1,    0,    1,    0,   -1,    0,    0,    0,   -1,    0,    1,
      -1,    0,    0,    0,    0,   -1,    0,    1,    0,    0,    0,    0,    0,   -1,    0,    1,
       1,    0,    0,    0,    0,   -1,    0,    1,    0,    1,    0,    0,    0,   -1,    0,    1,
       1,   -1,    1,    0,    0,   -1,    0,    1,    0,    0,    1,    0,    0,   -1,    0,    1,
      -1,    0,   -1,    1,    0,   -1,    0,    1,    1,    0,   -1,    1,    0,   -1,    0,    1,
       1,   -1,    0,    1,    0,   -1,    0,    1,    0,    0,    0,    1,    0,   -1,    0,    1,
      -1,    1,    0,    1,    0,   -1,    0,    1,    0,   -1,    1,    1,    0,   -1,    0,    1,
      -1,    0,    1,    1,    0,   -1,    0,    1,    1,    0,    1,    1,    0,   -1,    0,    1,
       0,    1,    1,    1,    0,   -1,    0,    1,    0,    0,   -1,   -1,    1,   -1,    0,    1,
       0,   -1,    0,   -1,    1,   -1,    0,    1,   -1,    0,    0,   -1,    1,   -1,    0,    1,
       0,    1,    0,   -1,    1,   -1,    0,    1,    0,    0,    1,   -1,    1,   -1,    0,    1,
      -1,    0,   -1,    0,    1,   -1,    0,    1,    1,    0,   -1,    0,    1,   -1,    0,    1,
       0,    1,   -1,    0,    1,   -1,    0,    1,   -1,   -1,    0,    0,    1,   -1,    0,    1,
       1,   -1,    0,    0,    1,   -1,    0,    1,    0,    0,    0,    0,    1,   -1,    0,    1,
      -1,    1,    0,    0,    1,   -1,    0,    1,    1,    0,    1,    0,    1,   -1,    0,    1,
       0,    1,    1,    0,    1,   -1,    0,    1,    0,    0,   -1,    1,    1,   -1,    0,    1,
      -1,    0,    0,    1,    1,   -1,    0,    1,    1,    0,    0,    1,    1,   -1,    0,    1,
       0,    1,    1,    1,    1,   -1,    0,    1,    0,   -1,   -1,   -1,   -1,    0,    0,    1,
      -1,    0,   -1,   -1,   -1,    0,    0,    1,    1,    0,   -1,   -1,   -1,    0,    0,    1,
      -1,   -1,    0,   -1,   -1,    0,    0,    1,    0,    0,    0,   -1,   -1,    0,    0,    1,
      -1,    1,    0,   -1,   -1,    0,    0,    1,    1,    0,    1,   -1,   -1,    0,    0,    1,
      -1,   -1,   -1,    0,   -1,    0,    0,    1,    1,    1,   -1,    0,   -1,    0,    0,    1,
       0,   -1,    0,    0,   -1,    0,    0,    1,   -1,    0,    0,    0,   -1,    0,    0,    1,
       0,    0,    0,    0,   -1,    0,    0,    1,    1,    0,    0,    0,   -1,    0,    0,    1,
      -1,    1,    0,    0,   -1,    0,    0,    1,    0,    1,    0,    0,   -1,    0,    0,    1,
      -1,   -1,    1,    0,   -1,    0,    0,    1,    0,   -1,    1,    0,   -1,    0,    0,    1,
       1,   -1,    1,    0,   -1,    0,    0,    1,    0,    0,    1,    0,   -1,    0,    0,    1,
      -1,    0,   -1,    1,   -1,    0,    0,    1,    1,    0,   -1,    1,   -1,    0,    0,    1,
       1,   -1,    0,    1,   -1,    0,    0,    1,   -1,    0,    0,    1,   -1,    0,    0,    1,
       0,    0,    0,    1,   -1,    0,    0,    1,   -1,    1,    0,    1,   -1,    0,    0,    1,
       0,   -1,    1,    1,   -1,    0,    0,    1,    0,    1,    1,    1,   -1,    0,    0,    1,
      -1,   -1,   -1,   -1,    0,    0,    0,    1,    0,    0,   -1,   -1,    0,    0,    0,    1,
      -1,    1,   -1,   -1,    0,    0,    0,    1,    1,    1,   -1,   -1,    0,    0,    0,    1,
      -1,   -1,    0,   -1,    0,    0,    0,    1,    0,   -1,    0,   -1,    0,    0,    0,    1,
      -1,    0,    0,   -1,    0,    0,    0,    1,    0,    0,    0,   -1,    0,    0,    0,    1,
       1,    0,    0,   -1,    0,    0,    0,    1,    0,    1,    0,   -1,    0,    0,    0,    1,
       0,   -1,    1,   -1,    0,    0,    0,    1,    0,    0,    1,   -1,    0,    0,    0,    1,
       0,    1,    1,   -1,    0,    0,    0,    1,    1,    1,    1,   -1,    0,    0,    0,    1,
       0,   -1,   -1,    0,    0,    0,    0,    1,   -1,    0,   -1,    0,    0,    0,    0,    1,
       0,    0,   -1,    0,    0,    0,    0,    1,    1,    0,   -1,    0,    0,    0,    0,    1,
       0,    1,   -1,    0,    0,    0,    0,    1,   -1,   -1,    0,    0,    0,    0,    0,    1,
       0,   -1,    0,    0,    0,    0,    0,    1,    1,   -1,    0,    0,    0,    0,    0,    1,
      -1,    0,    0,    0,    0,    0,    0,    1,    0,    0,    0,    0,    0,    0,    0,    1,
       1,    0,    0,    0,    0,    0,    0,    1,   -1,    1,    0,    0,    0,    0,    0,    1,
       0,    1,    0,    0,    0,    0,    0,    1,    1,    1,    0,    0,    0,    0,    0,    1,
       0,   -1,    1,    0,    0,    0,    0,    1,   -1,    0,    1,    0,    0,    0,    0,    1,
       0,    0,    1,    0,    0,    0,    0,    1,    1,    0,    1,    0,    0,    0,    0,    1,
       0,    1,    1,    0,    0,    0,    0,    1,    0,   -1,   -1,    1,    0,    0,    0,    1,
       0,    0,   -1,    1,    0,    0,    0,    1,   -1,    1,   -1,    1,    0,    0,    0,    1,
       0,   -1,    0,    1,    0,    0,    0,    1,    1,   -1,    0,    1,    0,    0,    0,    1,
      -1,    0,    0,    1,    0,    0,    0,    1,    0,    0,    0,    1,    0,    0,    0,    1,
       1,    0,    0,    1,    0,    0,    0,    1,    0,    1,    0,    1,    0,    0,    0,    1,
       1,    1,    0,    1,    0,    0,    0,    1,   -1,   -1,    1,    1,    0,    0,    0,    1,
       1,   -1,    1,    1,    0,    0,    0,    1,    0,    0,    1,    1,    0,    0,    0,    1,
      -1,    1,    1,    1,    0,    0,    0,    1,    1,    1,    1,    1,    0,    0,    0,    1,
       0,   -1,   -1,   -1,    1,    0,    0,    1,   -1,    0,   -1,   -1,    1,    0,    0,    1,
      -1,   -1,    0,   -1,    1,    0,    0,    1,    0,    0,    0,   -1,    1,    0,    0,    1,
       0,    1,    0,   -1,    1,    0,    0,    1,   -1,   -1,    1,   -1,    1,    0,    0,    1,
       1,    0,    1,   -1,    1,    0,    0,    1,    0,    1,    1,   -1,    1,    0,    0,    1,
       0,    0,   -1,    0,    1,    0,    0,    1,   -1,    1,   -1,    0,    1,    0,    0,    1,
       0,    1,   -1,    0,    1,    0,    0,    1,    0,   -1,    0,    0,    1,    0,    0,    1,
       1,   -1,    0,    0,    1,    0,    0,    1,    0,    0,    0,    0,    1,    0,    0,    1,
       1,    0,    0,    0,    1,    0,    0,    1,    0,    1,    0,    0,    1,    0,    0,    1,
       0,    0,    1,    0,    1,    0,    0,    1,   -1,    1,    1,    0,    1,    0,    0,    1,
       1,   -1,   -1,    1,    1,    0,    0,    1,   -1,    0,   -1,    1,    1,    0,    0,    1,
       0,    1,   -1,    1,    1,    0,    0,    1,    1,    1,   -1,    1,    1,    0,    0,    1,
       1,   -1,    0,    1,    1,    0,    0,    1,   -1,    0,    0,    1,    1,    0,    0,    1,
       0,    0,    0,    1,    1,    0,    0,    1,   -1,    0,    1,    1,    1,    0,    0,    1,
       1,    0,    1,    1,    1,    0,    0,    1,    0,    1,    1,    1,    1,    0,    0,    1,
       0,    0,   -1,   -1,   -1,    1,    0,    1,    1,    0,    0,   -1,   -1,    1,    0,    1,
       0,    1,    0,   -1,   -1,    1,    0,    1,    0,    0,    1,   -1,   -1,    1,    0,    1,
       0,   -1,   -1,    0,   -1,    1,    0,    1,    1,    0,   -1,    0,   -1,    1,    0,    1,
      -1,   -1,    0,    0,   -1,    1,    0,    1,    1,   -1,    0,    0,   -1,    1,    0,    1,
       0,    0,    0,    0,   -1,    1,    0,    1,    1,    0,    0,    0,   -1,    1,    0,    1,
       1,    1,    0,    0,   -1,    1,    0,    1,   -1,    0,    1,    0,   -1,    1,    0,    1,
       0,    0,    1,    0,   -1,    1,    0,    1,    0,    0,   -1,    1,   -1,    1,    0,    1,
       0,   -1,    0,    1,   -1,    1,    0,    1,    1,    0,    0,    1,   -1,    1,    0,    1,
       0,    1,    0,    1,   -1,    1,    0,    1,    0,    0,    1,    1,   -1,    1,    0,    1,
      -1,    0,   -1,   -1,    0,    1,    0,    1,    1,    0,   -1,   -1,    0,    1,    0,    1,
       0,    1,   -1,   -1,    0,    1,    0,    1,   -1,   -1,    0,   -1,    0,    1,    0,    1,
       1,   -1,    0,   -1,    0,    1,    0,    1,    0,    0,    0,   -1,    0,    1,    0,    1,
      -1,    1,    0,   -1,    0,    1,    0,    1,    1,    1,    0,   -1,    0,    1,    0,    1,
      -1,   -1,    1,   -1,    0,    1,    0,    1,    0,   -1,    1,   -1,    0,    1,    0,    1,
      -1,    0,    1,   -1,    0,    1,    0,    1,    1,    0,    1,   -1,    0,    1,    0,    1,
      -1,   -1,   -1,    0,    0,    1,    0,    1,    1,   -1,   -1,    0,    0,    1,    0,    1,
       0,    0,   -1,    0,    0,    1,    0,    1,   -1,    1,   -1,    0,    0,    1,    0,    1,
       1,    1,   -1,    0,    0,    1,    0,    1,    0,   -1,    0,    0,    0,    1,    0,    1,
      -1,    0,    0,    0,    0,    1,    0,    1,    0,    0,    0,    0,    0,    1,    0,    1,
       1,    0,    0,    0,    0,    1,    0,    1,    0,    1,    0,    0,    0,    1,    0,    1,
       1,   -1,    1,    0,    0,    1,    0,    1,    0,    0,    1,    0,    0,    1,    0,    1,
       1,    0,    1,    0,    0,    1,    0,    1,    1,    1,    1,    0,    0,    1,    0,    1,
       0,   -1,   -1,    1,    0,    1,    0,    1,   -1,    0,   -1,    1,    0,    1,    0,    1,
      -1,   -1,    0,    1,    0,    1,    0,    1,    1,   -1,    0,    1,    0,    1,    0,    1,
       0,    0,    0,    1,    0,    1,    0,    1,    1,    1,    0,    1,    0,    1,    0,    1,
       0,   -1,    1,    1,    0,    1,    0,    1,    1,    0,    1,    1,    0,    1,    0,    1,
       0,    0,   -1,   -1,    1,    1,    0,    1,    0,    0,    0,   -1,    1,    1,    0,    1,
       0,    0,    1,   -1,    1,    1,    0,    1,   -1,    0,   -1,    0,    1,    1,    0,    1,
       1,    0,   -1,    0,    1,    1,    0,    1,    0,    1,   -1,    0,    1,    1,    0,    1,
      -1,   -1,    0,    0,    1,    1,    0,    1,    0,    0,    0,    0,    1,    1,    0,    1,
      -1,    1,    0,    0,    1,    1,    0,    1,    0,   -1,    1,    0,    1,    1,    0,    1,
       0,    0,   -1,    1,    1,    1,    0,    1,    0,   -1,    0,    1,    1,    1,    0,    1,
      -1,    0,    0,    1,    1,    1,    0,    1,    0,    0,    0,    1,    1,    1,    0,    1,
       1,    0,    0,    1,    1,    1,    0,    1,   -1,   -1,   -1,   -1,   -1,   -1,    1,    1,
       1,   -1,   -1,   -1,   -1,   -1,    1,    1,   -1,    1,   -1,   -1,   -1,   -1,    1,    1,
       1,    1,   -1,   -1,   -1,   -1,    1,    1,    0,    0,    0,   -1,   -1,   -1,    1,    1,
      -1,   -1,    1,   -1,   -1,   -1,    1,    1,    1,   -1,    1,   -1,   -1,   -1,    1,    1,
      -1,    1,    1,   -1,   -1,   -1,    1,    1,    1,    1,    1,   -1,   -1,   -1,    1,    1,
       0,    0,   -1,    0,   -1,   -1,    1,    1,    0,   -1,    0,    0,   -1,   -1,    1,    1,
      -1,    0,    0,    0,   -1,   -1,    1,    1,    1,    0,    0,    0,   -1,   -1,    1,    1,
       0,    1,    0,    0,   -1,   -1,    1,    1,   -1,   -1,   -1,    1,   -1,   -1,    1,    1,
       1,   -1,   -1,    1,   -1,   -1,    1,    1,   -1,    1,   -1,    1,   -1,   -1,    1,    1,
       1,    1,   -1,    1,   -1,   -1,    1,    1,    0,    0,    0,    1,   -1,   -1,    1,    1,
      -1,   -1,    1,    1,   -1,   -1,    1,    1,    1,   -1,    1,    1,   -1,   -1,    1,    1,
      -1,    1,    1,    1,   -1,   -1,    1,    1,    1,    1,    1,    1,   -1,   -1,    1,    1,
       0,    0,   -1,   -1,    0,   -1,    1,    1,    0,    1,   -1,   -1,    0,   -1,    1,    1,
       0,   -1,    0,   -1,    0,   -1,    1,    1,   -1,    0,    0,   -1,    0,   -1,    1,    1,
       1,    0,    0,   -1,    0,   -1,    1,    1,    0,    1,    0,   -1,    0,   -1,    1,    1,
       1,    1,    0,   -1,    0,   -1,    1,    1,    1,    0,   -1,    0,    0,   -1,    1,    1,
       0,    1,   -1,    0,    0,   -1,    1,    1,    0,   -1,    0,    0,    0,   -1,    1,    1,
       0,    0,    0,    0,    0,   -1,    1,    1,   -1,    1,    0,    0,    0,   -1,    1,    1,
       1,    1,    0,    0,    0,   -1,    1,    1,    0,   -1,    1,    0,    0,   -1,    1,    1,
      -1,    0,    1,    0,    0,   -1,    1,    1,    0,    0,   -1,    1,    0,   -1,    1,    1,
      -1,   -1,    0,    1,    0,   -1,    1,    1,    1,   -1,    0,    1,    0,   -1,    1,    1,
       1,    0,    0,    1,    0,   -1,    1,    1,    0,    1,    0,    1,    0,   -1,    1,    1,
       1,   -1,   -1,   -1,    1,   -1,    1,    1,   -1,    1,   -1,   -1,    1,   -1,    1,    1,
       1,    1,   -1,   -1,    1,   -1,    1,    1,   -1,   -1,    0,   -1,    1,   -1,    1,    1,
       0,    1,    0,   -1,    1,   -1,    1,    1,    1,   -1,    1,   -1,    1,   -1,    1,    1,
      -1,    1,    1,   -1,    1,   -1,    1,    1,    1,    1,    1,   -1,    1,   -1,    1,    1,
       0,    0,   -1,    0,    1,   -1,    1,    1,    0,   -1,    0,    0,    1,   -1,    1,    1,
       1,    0,    0,    0,    1,   -1,    1,    1,    0,    1,    0,    0,    1,   -1,    1,    1,
       0,    0,    1,    0,    1,   -1,    1,    1,   -1,   -1,   -1,    1,    1,   -1,    1,    1,
       1,   -1,   -1,    1,    1,   -1,    1,    1,   -1,    1,   -1,    1,    1,   -1,    1,    1,
       1,    1,   -1,    1,    1,   -1,    1,    1,    0,    0,    0,    1,    1,   -1,    1,    1,
      -1,   -1,    1,    1,    1,   -1,    1,    1,    1,   -1,    1,    1,    1,   -1,    1,    1,
      -1,    1,    1,    1,    1,   -1,    1,    1,    1,    1,    1,    1,    1,   -1,    1,    1,
       0,    1,    0,   -1,   -1,    0,    1,    1,    0,    0,    1,   -1,   -1,    0,    1,    1,
       0,   -1,   -1,    0,   -1,    0,    1,    1,   -1,    0,   -1,    0,   -1,    0,    1,    1,
      -1,   -1,    0,    0,   -1,    0,    1,    1,   -1,    0,    0,    0,   -1,    0,    1,    1,
       0,    0,    0,    0,   -1,    0,    1,    1,   -1,    1,    0,    0,   -1,    0,    1,    1,
       1,    1,    0,    0,   -1,    0,    1,    1,    0,   -1,    1,    0,   -1,    0,    1,    1,
       0,    0,    1,    0,   -1,    0,    1,    1,    1,    0,    1,    0,   -1,    0,    1,    1,
      -1,    1,    1,    0,   -1,    0,    1,    1,    0,    1,    1,    0,   -1,    0,    1,    1,
       0,    0,   -1,    1,   -1,    0,    1,    1,    1,    0,   -1,   -1,    0,    0,    1,    1,
       0,    1,   -1,   -1,    0,    0,    1,    1,   -1,   -1,    0,   -1,    0,    0,    1,    1,
       1,   -1,    0,   -1,    0,    0,    1,    1,    0,    0,    0,   -1,    0,    0,    1,    1,
      -1,    1,    0,   -1,    0,    0,    1,    1,    1,    0,    1,   -1,    0,    0,    1,    1,
       0,    1,    1,   -1,    0,    0,    1,    1,    1,   -1,   -1,    0,    0,    0,    1,    1,
       0,    0,   -1,    0,    0,    0,    1,    1,    0,   -1,    0,    0,    0,    0,    1,    1,
      -1,    0,    0,    0,    0,    0,    1,    1,    0,    0,    0,    0,    0,    0,    1,    1,
       1,    0,    0,    0,    0,    0,    1,    1,    0,    1,    0,    0,    0,    0,    1,    1,
       0,    0,    1,    0,    0,    0,    1,    1,    1,    1,    1,    0,    0,    0,    1,    1,
       0,   -1,   -1,    1,    0,    0,    1,    1,   -1,    0,   -1,    1,    0,    0,    1,    1,
       0,    0,   -1,    1,    0,    0,    1,    1,    1,    0,   -1,    1,    0,    0,    1,    1,
       0,    1,   -1,    1,    0,    0,    1,    1,    1,   -1,    0,    1,    0,    0,    1,    1,
       0,    0,    0,    1,    0,    0,    1,    1,   -1,    1,    0,    1,    0,    0,    1,    1,
       0,    0,   -1,   -1,    1,    0,    1,    1,    0,   -1,    0,   -1,    1,    0,    1,    1,
       1,    0,    0,   -1,    1,    0,    1,    1,    1,    1,    0,   -1,    1,    0,    1,    1,
       0,   -1,    1,   -1,    1,    0,    1,    1,    0,    0,    1,   -1,    1,    0,    1,    1,
      -1,    0,   -1,    0,    1,    0,    1,    1,    1,    0,   -1,    0,    1,    0,    1,    1,
       1,    1,   -1,    0,    1,    0,    1,    1,    1,   -1,    0,    0,    1,    0,    1,    1,
       0,    0,    0,    0,    1,    0,    1,    1,    1,    0,    0,    0,    1,    0,    1,    1,
      -1,    1,    0,    0,    1,    0,    1,    1,   -1,   -1,    1,    0,    1,    0,    1,    1,
       1,   -1,    1,    0,    1,    0,    1,    1,    1,    0,   -1,    1,    1,    0,    1,    1,
      -1,   -1,    0,    1,    1,    0,    1,    1,    0,    0,    0,    1,    1,    0,    1,    1,
       1,    0,    0,    1,    1,    0,    1,    1,    0,    1,    0,    1,    1,    0,    1,    1,
       0,   -1,    1,    1,    1,    0,    1,    1,   -1,    0,    1,    1,    1,    0,    1,    1,
       1,    0,    1,    1,    1,    0,    1,    1,   -1,   -1,   -1,   -1,   -1,    1,    1,    1,
       1,   -1,   -1,   -1,   -1,    1,    1,    1,   -1,    1,   -1,   -1,   -1,    1,    1,    1,
       1,    1,   -1,   -1,   -1,    1,    1,    1,   -1,   -1,    1,   -1,   -1,    1,    1,    1,
       1,   -1,    1,   -1,   -1,    1,    1,    1,   -1,    1,    1,   -1,   -1,    1,    1,    1,
       1,    1,    1,   -1,   -1,    1,    1,    1,    0,   -1,    0,    0,   -1,    1,    1,    1,
      -1,    0,    0,    0,   -1,    1,    1,    1,    1,    0,    0,    0,   -1,    1,    1,    1,
       0,    1,    0,    0,   -1,    1,    1,    1,   -1,   -1,   -1,    1,   -1,    1,    1,    1,
       1,   -1,   -1,    1,   -1,    1,    1,    1,   -1,    1,   -1,    1,   -1,    1,    1,    1,
       1,    1,   -1,    1,   -1,    1,    1,    1,    0,    0,    0,    1,   -1,    1,    1,    1,
      -1,   -1,    1,    1,   -1,    1,    1,    1,    1,   -1,    1,    1,   -1,    1,    1,    1,
      -1,    1,    1,    1,   -1,    1,    1,    1,    1,    1,    1,    1,   -1,    1,    1,    1,
       0,    0,   -1,   -1,    0,    1,    1,    1,   -1,    0,    0,   -1,    0,    1,    1,    1,
       0,    1,    0,   -1,    0,    1,    1,    1,    0,   -1,    1,   -1,    0,    1,    1,    1,
       0,    0,    1,   -1,    0,    1,    1,    1,    0,   -1,   -1,    0,    0,    1,    1,    1,
      -1,   -1,    0,    0,    0,    1,    1,    1,    0,    0,    0,    0,    0,    1,    1,    1,
       1,    1,    0,    0,    0,    1,    1,    1,    0,   -1,    1,    0,    0,    1,    1,    1,
       1,    0,    1,    0,    0,    1,    1,    1,    0,    1,    1,    0,    0,    1,    1,    1,
      -1,   -1,    0,    1,    0,    1,    1,    1,    1,    0,    0,    1,    0,    1,    1,    1,
      -1,   -1,   -1,   -1,    1,    1,    1,    1,    1,   -1,   -1,   -1,    1,    1,    1,    1,
      -1,    1,   -1,   -1,    1,    1,    1,    1,    1,    1,   -1,   -1,    1,    1,    1,    1,
      -1,   -1,    1,   -1,    1,    1,    1,    1,    1,   -1,    1,   -1,    1,    1,    1,    1,
      -1,    1,    1,   -1,    1,    1,    1,    1,    1,    1,    1,   -1,    1,    1,    1,    1,
       0,   -1,    0,    0,    1,    1,    1,    1,   -1,    0,    0,    0,    1,    1,    1,    1,
       1,    0,    0,    0,    1,    1,    1,    1,   -1,   -1,   -1,    1,    1,    1,    1,    1,
       1,   -1,   -1,    1,    1,    1,    1,    1,   -1,    1,   -1,    1,    1,    1,    1,    1,
       1,    1,   -1,    1,    1,    1,    1,    1,    0,    0,    0,    1,    1,    1,    1,    1,
      -1,   -1,    1,    1,    1,    1,    1,    1,    1,   -1,    1,    1,    1,    1,    1,    1,
      -1,    1,    1,    1,    1,    1,    1,    1,    1,    1,    1,    1,    1,    1,    1,    1,
};

#define INGOT_IQ1S_GRID_ENTRIES 2048
#define INGOT_IQ1S_GRID_WIDTH   8

/* IQ2_XXS: 256 entries x 8 values, map (8, 25, 43) */
static const int8_t ingot_iq2xxs_grid[2048] = {
       8,    8,    8,    8,    8,    8,    8,    8,   43,    8,    8,    8,    8,    8,    8,    8,
      25,   25,    8,    8,    8,    8,    8,    8,    8,   43,    8,    8,    8,    8,    8,    8,
      43,   43,    8,    8,    8,    8,    8,    8,   25,    8,   25,    8,    8,    8,    8,    8,
       8,   25,   25,    8,    8,    8,    8,    8,    8,    8,   43,    8,    8,    8,    8,    8,
      43,    8,   43,    8,    8,    8,    8,    8,    8,   43,   43,    8,    8,    8,    8,    8,
      43,   43,   43,    8,    8,    8,    8,    8,   25,    8,    8,   25,    8,    8,    8,    8,
       8,   25,    8,   25,    8,    8,    8,    8,    8,    8,   25,   25,    8,    8,    8,    8,
       8,   43,   25,   25,    8,    8,    8,    8,   25,    8,   43,   25,    8,    8,    8,    8,
       8,   25,   43,   25,    8,    8,    8,    8,    8,    8,    8,   43,    8,    8,    8,    8,
      43,    8,    8,   43,    8,    8,    8,    8,   43,   43,    8,   43,    8,    8,    8,    8,
      43,    8,   43,   43,    8,    8,    8,    8,   25,    8,    8,    8,   25,    8,    8,    8,
       8,   25,    8,    8,   25,    8,    8,    8,    8,    8,   25,    8,   25,    8,    8,    8,
      25,   25,   25,    8,   25,    8,    8,    8,    8,    8,    8,   25,   25,    8,    8,    8,
       8,   25,    8,   43,   25,    8,    8,    8,    8,   43,   25,   43,   25,    8,    8,    8,
       8,    8,    8,    8,   43,    8,    8,    8,   43,    8,    8,    8,   43,    8,    8,    8,
      43,    8,   43,    8,   43,    8,    8,    8,   43,    8,    8,   43,   43,    8,    8,    8,
      25,    8,    8,    8,    8,   25,    8,    8,    8,   25,    8,    8,    8,   25,    8,    8,
       8,    8,   25,    8,    8,   25,    8,    8,   25,    8,   43,    8,    8,   25,    8,    8,
       8,   25,   43,    8,    8,   25,    8,    8,    8,    8,    8,   25,    8,   25,    8,    8,
      43,    8,    8,   25,    8,   25,    8,    8,    8,   43,    8,   25,    8,   25,    8,    8,
       8,    8,   43,   25,    8,   25,    8,    8,   25,    8,    8,   43,    8,   25,    8,    8,
       8,   25,    8,   43,    8,   25,    8,    8,    8,    8,   25,   43,    8,   25,    8,    8,
       8,   25,   43,   43,    8,   25,    8,    8,    8,    8,    8,    8,   25,   25,    8,    8,
      43,    8,    8,    8,   25,   25,    8,    8,    8,   43,    8,    8,   25,   25,    8,    8,
       8,    8,   43,    8,   25,   25,    8,    8,   43,   25,    8,   25,   25,   25,    8,    8,
      25,   43,   43,   25,   25,   25,    8,    8,    8,    8,    8,   43,   25,   25,    8,    8,
      25,    8,   25,   43,   25,   25,    8,    8,   25,   43,    8,    8,   43,   25,    8,    8,
       8,    8,   25,    8,   43,   25,    8,    8,    8,    8,    8,   25,   43,   25,    8,    8,
       8,   25,    8,   43,   43,   25,    8,    8,    8,   25,   43,   43,   43,   25,    8,    8,
       8,    8,    8,    8,    8,   43,    8,    8,   25,   25,    8,    8,    8,   43,    8,    8,
       8,   43,    8,    8,    8,   43,    8,    8,    8,   25,   25,    8,    8,   43,    8,    8,
       8,   43,   43,    8,    8,   43,    8,    8,   25,    8,    8,   25,    8,   43,    8,    8,
       8,   25,    8,   25,    8,   43,    8,    8,    8,    8,   25,   25,    8,   43,    8,    8,
      43,    8,   25,   25,    8,   43,    8,    8,    8,   43,    8,   43,    8,   43,    8,    8,
       8,   25,    8,    8,   25,   43,    8,    8,    8,    8,    8,   25,   25,   43,    8,    8,
      43,    8,    8,    8,   43,   43,    8,    8,    8,   25,   25,    8,   43,   43,    8,    8,
      25,    8,    8,    8,    8,    8,   25,    8,    8,   25,    8,    8,    8,    8,   25,    8,
       8,    8,   25,    8,    8,    8,   25,    8,   25,    8,   43,    8,    8,    8,   25,    8,
       8,    8,    8,   25,    8,    8,   25,    8,    8,    8,   43,   25,    8,    8,   25,    8,
       8,   25,    8,   43,    8,    8,   25,    8,    8,    8,   25,   43,    8,    8,   25,    8,
      25,   25,   25,   43,    8,    8,   25,    8,    8,    8,    8,    8,   25,    8,   25,    8,
       8,   43,    8,    8,   25,    8,   25,    8,    8,    8,   43,    8,   25,    8,   25,    8,
       8,    8,   25,   25,   25,    8,   25,    8,   43,   43,   25,   25,   25,    8,   25,    8,
       8,    8,    8,   43,   25,    8,   25,    8,    8,   25,   43,    8,   43,    8,   25,    8,
      25,   25,    8,   25,   43,    8,   25,    8,    8,    8,    8,    8,    8,   25,   25,    8,
       8,   43,    8,    8,    8,   25,   25,    8,    8,    8,   43,    8,    8,   25,   25,    8,
      25,   25,   43,    8,    8,   25,   25,    8,   25,   43,    8,   25,    8,   25,   25,    8,
       8,    8,    8,   43,    8,   25,   25,    8,    8,   43,   25,    8,   25,   25,   25,    8,
      43,    8,   43,   25,   25,   25,   25,    8,    8,    8,    8,    8,   43,   25,   25,    8,
      43,   25,   25,    8,   43,   25,   25,    8,   25,    8,    8,    8,    8,   43,   25,    8,
       8,   25,    8,    8,    8,   43,   25,    8,    8,    8,   25,    8,    8,   43,   25,    8,
       8,    8,    8,   25,    8,   43,   25,    8,   25,    8,    8,   43,    8,   43,   25,    8,
       8,    8,    8,    8,   25,   43,   25,    8,   25,   25,    8,    8,   25,   43,   25,    8,
       8,    8,   43,   43,   25,   43,   25,    8,   25,    8,   25,   25,   43,   43,   25,    8,
       8,    8,    8,    8,    8,    8,   43,    8,   43,    8,    8,    8,    8,    8,   43,    8,
      43,   43,    8,    8,    8,    8,   43,    8,    8,   25,    8,   25,    8,    8,   43,    8,
      25,    8,   43,   25,    8,    8,   43,    8,    8,    8,    8,   43,    8,    8,   43,    8,
      43,    8,    8,   43,    8,    8,   43,    8,   25,   43,   43,    8,   25,    8,   43,    8,
       8,   43,    8,   25,   25,    8,   43,    8,    8,    8,    8,    8,   43,    8,   43,    8,
      43,    8,    8,    8,   43,    8,   43,    8,   25,    8,    8,    8,    8,   25,   43,    8,
       8,   25,    8,    8,    8,   25,   43,    8,    8,    8,   25,    8,    8,   25,   43,    8,
       8,    8,    8,   25,    8,   25,   43,    8,   43,   25,   25,   25,    8,   25,   43,    8,
       8,    8,    8,    8,   25,   25,   43,    8,   25,    8,    8,   25,   25,   25,   43,    8,
       8,   25,   43,   25,   25,   25,   43,    8,    8,    8,   25,   43,   43,   25,   43,    8,
       8,   43,    8,    8,    8,   43,   43,    8,    8,    8,   43,    8,    8,   43,   43,    8,
       8,   25,   25,   43,    8,   43,   43,    8,    8,   25,    8,   25,   43,   43,   43,    8,
      25,    8,    8,    8,    8,    8,    8,   25,    8,   25,    8,    8,    8,    8,    8,   25,
       8,    8,   25,    8,    8,    8,    8,   25,    8,   43,   25,    8,    8,    8,    8,   25,
      25,    8,   43,    8,    8,    8,    8,   25,    8,   25,   43,    8,    8,    8,    8,   25,
       8,    8,    8,   25,    8,    8,    8,   25,    8,   43,    8,   25,    8,    8,    8,   25,
      43,   25,   25,   25,    8,    8,    8,   25,    8,    8,   43,   25,    8,    8,    8,   25,
      25,    8,    8,   43,    8,    8,    8,   25,    8,   25,    8,   43,    8,    8,    8,   25,
       8,    8,   25,   43,    8,    8,    8,   25,    8,    8,    8,    8,   25,    8,    8,   25,
       8,    8,   43,    8,   25,    8,    8,   25,   25,    8,   43,   25,   25,    8,    8,   25,
       8,    8,    8,   43,   25,    8,    8,   25,   25,   25,    8,   43,   25,    8,    8,   25,
      25,    8,    8,    8,   43,    8,    8,   25,    8,    8,   25,    8,   43,    8,    8,   25,
       8,   43,    8,   25,   43,    8,    8,   25,   43,   25,   25,   25,   43,    8,    8,   25,
       8,   43,   43,   25,   43,    8,    8,   25,    8,    8,    8,    8,    8,   25,    8,   25,
       8,   43,    8,    8,    8,   25,    8,   25,    8,    8,   43,    8,    8,   25,    8,   25,
       8,    8,    8,   43,    8,   25,    8,   25,   25,   43,   25,   43,    8,   25,    8,   25,
      43,    8,   25,    8,   25,   25,    8,   25,    8,   25,   43,    8,   25,   25,    8,   25,
       8,    8,    8,    8,   43,   25,    8,   25,   25,    8,    8,    8,    8,   43,    8,   25,
       8,   25,    8,    8,    8,   43,    8,   25,    8,    8,   25,    8,    8,   43,    8,   25,
       8,    8,    8,   25,    8,   43,    8,   25,   25,   25,    8,   25,    8,   43,    8,   25,
       8,    8,    8,    8,   25,   43,    8,   25,    8,   43,   25,   25,   25,   43,    8,   25,
      25,    8,   43,   25,   25,   43,    8,   25,   43,    8,    8,   43,   25,   43,    8,   25,
      25,   25,    8,   25,   43,   43,    8,   25,    8,    8,   25,   43,   43,   43,    8,   25,
       8,    8,    8,    8,    8,    8,   25,   25,    8,   43,    8,    8,    8,    8,   25,   25,
      25,    8,   25,    8,    8,    8,   25,   25,   25,   43,   25,    8,    8,    8,   25,   25,
       8,    8,   43,    8,    8,    8,   25,   25,    8,    8,    8,   43,    8,    8,   25,   25,
       8,   43,    8,   43,    8,    8,   25,   25,    8,   25,    8,    8,   25,    8,   25,   25,
      43,    8,    8,   25,   25,    8,   25,   25,    8,   25,   43,   43,   25,    8,   25,   25,
      25,    8,   25,   43,   43,    8,   25,   25,    8,    8,   25,   43,    8,   25,   25,   25,
      43,    8,   25,   43,    8,   25,   25,   25,   43,   43,    8,    8,   25,   25,   25,   25,
      25,    8,    8,    8,   43,   25,   25,   25,    8,   25,   25,   25,   43,   25,   25,   25,
       8,    8,    8,    8,    8,   43,   25,   25,   25,    8,   25,    8,    8,   43,   25,   25,
      25,   43,   25,    8,    8,   43,   25,   25,    8,   25,   43,   25,    8,   43,   25,   25,
       8,    8,    8,   25,   25,   43,   25,   25,    8,   43,    8,    8,   43,   43,   25,   25,
       8,   25,    8,    8,    8,    8,   43,   25,    8,    8,   25,    8,    8,    8,   43,   25,
       8,    8,    8,   25,    8,    8,   43,   25,    8,   43,   43,   25,    8,    8,   43,   25,
       8,    8,    8,    8,   25,    8,   43,   25,   25,   25,   25,   25,   25,    8,   43,   25,
       8,   43,   25,    8,   43,    8,   43,   25,    8,    8,   43,   25,   43,    8,   43,   25,
       8,    8,    8,    8,    8,   25,   43,   25,   25,   25,    8,    8,    8,   25,   43,   25,
       8,    8,   25,    8,   25,   25,   43,   25,   43,    8,   25,    8,   25,   25,   43,   25,
       8,   25,    8,   43,   25,   25,   43,   25,   43,    8,    8,   25,    8,   43,   43,   25,
       8,    8,    8,    8,    8,    8,    8,   43,   43,    8,    8,    8,    8,    8,    8,   43,
      43,   43,    8,    8,    8,    8,    8,   43,   25,    8,    8,   25,    8,    8,    8,   43,
      43,    8,    8,   43,    8,    8,    8,   43,    8,   25,    8,    8,   25,    8,    8,   43,
       8,   43,   25,    8,   25,    8,    8,   43,    8,    8,    8,   25,   25,    8,    8,   43,
      25,    8,   25,    8,   43,    8,    8,   43,   25,    8,    8,    8,    8,   25,    8,   43,
       8,   25,    8,    8,    8,   25,    8,   43,    8,    8,   25,    8,    8,   25,    8,   43,
      25,   25,   25,    8,    8,   25,    8,   43,    8,    8,    8,   25,    8,   25,    8,   43,
       8,    8,   43,   25,    8,   25,    8,   43,    8,    8,    8,    8,   25,   25,    8,   43,
      43,   25,    8,   25,   25,   25,    8,   43,    8,   25,   25,   43,   25,   25,    8,   43,
      25,   43,    8,    8,   43,   25,    8,   43,    8,    8,    8,   25,   43,   25,    8,   43,
       8,    8,   43,   25,   43,   25,    8,   43,   43,    8,    8,    8,    8,   43,    8,   43,
       8,   25,    8,    8,   25,   43,    8,   43,   25,    8,   25,    8,   43,   43,    8,   43,
       8,   25,    8,    8,    8,    8,   25,   43,    8,    8,   25,    8,    8,    8,   25,   43,
       8,   25,   43,    8,    8,    8,   25,   43,    8,    8,    8,   25,    8,    8,   25,   43,
      25,    8,   43,   43,    8,    8,   25,   43,   43,   25,   25,    8,   25,    8,   25,   43,
       8,    8,    8,   43,   25,    8,   25,   43,   25,   25,    8,   25,   43,    8,   25,   43,
       8,    8,    8,    8,    8,   25,   25,   43,   43,    8,   43,    8,    8,   25,   25,   43,
       8,   25,    8,   25,    8,   25,   25,   43,   25,    8,   25,   25,   25,   25,   25,   43,
      25,    8,    8,   43,    8,   43,   25,   43,    8,    8,   43,    8,   25,   43,   25,   43,
      43,    8,    8,    8,    8,    8,   43,   43,    8,    8,   25,   25,    8,    8,   43,   43,
      25,   25,    8,   43,    8,    8,   43,   43,   25,   43,    8,    8,   25,    8,   43,   43,
       8,    8,    8,    8,   43,    8,   43,   43,    8,   43,   25,    8,    8,   25,   43,   43,
       8,    8,   25,   25,    8,   43,   43,   43,    8,   25,    8,    8,   25,   43,   43,   43,
};

#define INGOT_IQ2XXS_GRID_ENTRIES 256
#define INGOT_IQ2XXS_GRID_WIDTH   8

/* IQ2_XS: 512 entries x 8 values, map (8, 25, 43) */
static const int8_t ingot_iq2xs_grid[4096] = {
       8,    8,    8,    8,    8,    8,    8,    8,   43,    8,    8,    8,    8,    8,    8,    8,
      25,   25,    8,    8,    8,    8,    8,    8,    8,   43,    8,    8,    8,    8,    8,    8,
      43,   43,    8,    8,    8,    8,    8,    8,   25,    8,   25,    8,    8,    8,    8,    8,
       8,   25,   25,    8,    8,    8,    8,    8,   43,   25,   25,    8,    8,    8,    8,    8,
      25,   43,   25,    8,    8,    8,    8,    8,    8,    8,   43,    8,    8,    8,    8,    8,
      43,    8,   43,    8,    8,    8,    8,    8,   25,   25,   43,    8,    8,    8,    8,    8,
       8,   43,   43,    8,    8,    8,    8,    8,   25,    8,    8,   25,    8,    8,    8,    8,
       8,   25,    8,   25,    8,    8,    8,    8,   43,   25,    8,   25,    8,    8,    8,    8,
      25,   43,    8,   25,    8,    8,    8,    8,    8,    8,   25,   25,    8,    8,    8,    8,
      43,    8,   25,   25,    8,    8,    8,    8,   25,   25,   25,   25,    8,    8,    8,    8,
       8,   43,   25,   25,    8,    8,    8,    8,   25,    8,   43,   25,    8,    8,    8,    8,
       8,   25,   43,   25,    8,    8,    8,    8,    8,    8,    8,   43,    8,    8,    8,    8,
      43,    8,    8,   43,    8,    8,    8,    8,   25,   25,    8,   43,    8,    8,    8,    8,
       8,   43,    8,   43,    8,    8,    8,    8,   25,    8,   25,   43,    8,    8,    8,    8,
       8,   25,   25,   43,    8,    8,    8,    8,   25,   43,   25,   43,    8,    8,    8,    8,
       8,    8,   43,   43,    8,    8,    8,    8,   25,    8,    8,    8,   25,    8,    8,    8,
       8,   25,    8,    8,   25,    8,    8,    8,   43,   25,    8,    8,   25,    8,    8,    8,
      25,   43,    8,    8,   25,    8,    8,    8,    8,    8,   25,    8,   25,    8,    8,    8,
      43,    8,   25,    8,   25,    8,    8,    8,   25,   25,   25,    8,   25,    8,    8,    8,
       8,   43,   25,    8,   25,    8,    8,    8,   43,   43,   25,    8,   25,    8,    8,    8,
      25,    8,   43,    8,   25,    8,    8,    8,    8,   25,   43,    8,   25,    8,    8,    8,
       8,    8,    8,   25,   25,    8,    8,    8,   43,    8,    8,   25,   25,    8,    8,    8,
      25,   25,    8,   25,   25,    8,    8,    8,    8,   43,    8,   25,   25,    8,    8,    8,
      25,    8,   25,   25,   25,    8,    8,    8,    8,   25,   25,   25,   25,    8,    8,    8,
       8,    8,   43,   25,   25,    8,    8,    8,    8,   43,   43,   25,   25,    8,    8,    8,
      25,    8,    8,   43,   25,    8,    8,    8,    8,   25,    8,   43,   25,    8,    8,    8,
       8,    8,   25,   43,   25,    8,    8,    8,    8,    8,    8,    8,   43,    8,    8,    8,
      43,    8,    8,    8,   43,    8,    8,    8,   25,   25,    8,    8,   43,    8,    8,    8,
       8,   43,    8,    8,   43,    8,    8,    8,   25,    8,   25,    8,   43,    8,    8,    8,
       8,   25,   25,    8,   43,    8,    8,    8,    8,    8,   43,    8,   43,    8,    8,    8,
      25,    8,    8,   25,   43,    8,    8,    8,    8,   25,    8,   25,   43,    8,    8,    8,
       8,    8,   25,   25,   43,    8,    8,    8,   25,   25,   25,   25,   43,    8,    8,    8,
       8,    8,    8,   43,   43,    8,    8,    8,   43,   43,    8,   43,   43,    8,    8,    8,
      25,    8,    8,    8,    8,   25,    8,    8,    8,   25,    8,    8,    8,   25,    8,    8,
      43,   25,    8,    8,    8,   25,    8,    8,   25,   43,    8,    8,    8,   25,    8,    8,
       8,    8,   25,    8,    8,   25,    8,    8,   43,    8,   25,    8,    8,   25,    8,    8,
      25,   25,   25,    8,    8,   25,    8,    8,    8,   43,   25,    8,    8,   25,    8,    8,
      25,    8,   43,    8,    8,   25,    8,    8,    8,   25,   43,    8,    8,   25,    8,    8,
       8,    8,    8,   25,    8,   25,    8,    8,   43,    8,    8,   25,    8,   25,    8,    8,
      25,   25,    8,   25,    8,   25,    8,    8,    8,   43,    8,   25,    8,   25,    8,    8,
      25,    8,   25,   25,    8,   25,    8,    8,    8,   25,   25,   25,    8,   25,    8,    8,
      43,   25,   25,   25,    8,   25,    8,    8,    8,    8,   43,   25,    8,   25,    8,    8,
      25,    8,    8,   43,    8,   25,    8,    8,    8,   25,    8,   43,    8,   25,    8,    8,
       8,    8,   25,   43,    8,   25,    8,    8,    8,    8,    8,    8,   25,   25,    8,    8,
      43,    8,    8,    8,   25,   25,    8,    8,   25,   25,    8,    8,   25,   25,    8,    8,
       8,   43,    8,    8,   25,   25,    8,    8,   25,    8,   25,    8,   25,   25,    8,    8,
       8,   25,   25,    8,   25,   25,    8,    8,    8,    8,   43,    8,   25,   25,    8,    8,
      25,    8,    8,   25,   25,   25,    8,    8,    8,   25,    8,   25,   25,   25,    8,    8,
       8,    8,   25,   25,   25,   25,    8,    8,   25,    8,   43,   25,   25,   25,    8,    8,
       8,    8,    8,   43,   25,   25,    8,    8,   25,    8,    8,    8,   43,   25,    8,    8,
       8,   25,    8,    8,   43,   25,    8,    8,    8,    8,   25,    8,   43,   25,    8,    8,
      43,   25,   43,    8,   43,   25,    8,    8,    8,    8,    8,   25,   43,   25,    8,    8,
      43,    8,    8,   25,   43,   25,    8,    8,    8,   25,    8,   43,   43,   25,    8,    8,
       8,    8,    8,    8,    8,   43,    8,    8,   43,    8,    8,    8,    8,   43,    8,    8,
      25,   25,    8,    8,    8,   43,    8,    8,    8,   43,    8,    8,    8,   43,    8,    8,
      43,   43,    8,    8,    8,   43,    8,    8,   25,    8,   25,    8,    8,   43,    8,    8,
       8,   25,   25,    8,    8,   43,    8,    8,    8,    8,   43,    8,    8,   43,    8,    8,
      25,   25,   43,    8,    8,   43,    8,    8,   25,    8,    8,   25,    8,   43,    8,    8,
       8,   25,    8,   25,    8,   43,    8,    8,    8,    8,   25,   25,    8,   43,    8,    8,
       8,   43,   25,   25,    8,   43,    8,    8,    8,    8,    8,   43,    8,   43,    8,    8,
       8,    8,   43,   43,    8,   43,    8,    8,   43,   43,   43,   43,    8,   43,    8,    8,
      25,    8,    8,    8,   25,   43,    8,    8,    8,   25,    8,    8,   25,   43,    8,    8,
       8,    8,   25,    8,   25,   43,    8,    8,    8,    8,    8,   25,   25,   43,    8,    8,
      25,    8,    8,   43,   25,   43,    8,    8,   25,   43,    8,   43,   25,   43,    8,    8,
       8,    8,    8,    8,   43,   43,    8,    8,    8,    8,   43,    8,   43,   43,    8,    8,
       8,   43,   43,    8,   43,   43,    8,    8,   43,   25,   25,   43,   43,   43,    8,    8,
       8,    8,   43,   43,   43,   43,    8,    8,   25,    8,    8,    8,    8,    8,   25,    8,
       8,   25,    8,    8,    8,    8,   25,    8,   43,   25,    8,    8,    8,    8,   25,    8,
      25,   43,    8,    8,    8,    8,   25,    8,    8,    8,   25,    8,    8,    8,   25,    8,
      43,    8,   25,    8,    8,    8,   25,    8,   25,   25,   25,    8,    8,    8,   25,    8,
       8,   43,   25,    8,    8,    8,   25,    8,   25,    8,   43,    8,    8,    8,   25,    8,
       8,   25,   43,    8,    8,    8,   25,    8,    8,    8,    8,   25,    8,    8,   25,    8,
      43,    8,    8,   25,    8,    8,   25,    8,   25,   25,    8,   25,    8,    8,   25,    8,
       8,   43,    8,   25,    8,    8,   25,    8,   25,    8,   25,   25,    8,    8,   25,    8,
       8,   25,   25,   25,    8,    8,   25,    8,    8,    8,   43,   25,    8,    8,   25,    8,
      43,   43,   43,   25,    8,    8,   25,    8,   25,    8,    8,   43,    8,    8,   25,    8,
       8,   25,    8,   43,    8,    8,   25,    8,    8,    8,   25,   43,    8,    8,   25,    8,
       8,    8,    8,    8,   25,    8,   25,    8,   43,    8,    8,    8,   25,    8,   25,    8,
      25,   25,    8,    8,   25,    8,   25,    8,    8,   43,    8,    8,   25,    8,   25,    8,
      25,    8,   25,    8,   25,    8,   25,    8,    8,   25,   25,    8,   25,    8,   25,    8,
       8,    8,   43,    8,   25,    8,   25,    8,   25,    8,    8,   25,   25,    8,   25,    8,
       8,   25,    8,   25,   25,    8,   25,    8,    8,    8,   25,   25,   25,    8,   25,    8,
       8,    8,    8,   43,   25,    8,   25,    8,    8,   25,   25,   43,   25,    8,   25,    8,
      43,   25,   25,   43,   25,    8,   25,    8,   25,    8,    8,    8,   43,    8,   25,    8,
       8,   25,    8,    8,   43,    8,   25,    8,   43,   25,    8,    8,   43,    8,   25,    8,
       8,    8,   25,    8,   43,    8,   25,    8,    8,    8,    8,   25,   43,    8,   25,    8,
       8,    8,   43,   25,   43,    8,   25,    8,    8,    8,    8,    8,    8,   25,   25,    8,
      43,    8,    8,    8,    8,   25,   25,    8,   25,   25,    8,    8,    8,   25,   25,    8,
       8,   43,    8,    8,    8,   25,   25,    8,   25,    8,   25,    8,    8,   25,   25,    8,
       8,   25,   25,    8,    8,   25,   25,    8,    8,    8,   43,    8,    8,   25,   25,    8,
      25,    8,    8,   25,    8,   25,   25,    8,    8,   25,    8,   25,    8,   25,   25,    8,
      25,   43,    8,   25,    8,   25,   25,    8,    8,    8,   25,   25,    8,   25,   25,    8,
       8,   25,   43,   25,    8,   25,   25,    8,    8,    8,    8,   43,    8,   25,   25,    8,
      25,    8,    8,    8,   25,   25,   25,    8,    8,   25,    8,    8,   25,   25,   25,    8,
       8,    8,   25,    8,   25,   25,   25,    8,    8,    8,    8,   25,   25,   25,   25,    8,
       8,    8,    8,    8,   43,   25,   25,    8,    8,   25,   25,    8,   43,   25,   25,    8,
      25,   43,    8,   25,   43,   25,   25,    8,   25,    8,    8,    8,    8,   43,   25,    8,
       8,   25,    8,    8,    8,   43,   25,    8,    8,    8,   25,    8,    8,   43,   25,    8,
      43,    8,   25,    8,    8,   43,   25,    8,    8,    8,    8,   25,    8,   43,   25,    8,
       8,   25,   25,   25,    8,   43,   25,    8,   43,   25,    8,   43,    8,   43,   25,    8,
       8,    8,    8,    8,   25,   43,   25,    8,   25,   25,    8,    8,   25,   43,   25,    8,
      43,   25,   43,   25,   25,   43,   25,    8,   25,    8,   25,   25,   43,   43,   25,    8,
      25,   43,   43,   43,   43,   43,   25,    8,    8,    8,    8,    8,    8,    8,   43,    8,
      43,    8,    8,    8,    8,    8,   43,    8,   25,   25,    8,    8,    8,    8,   43,    8,
       8,   43,    8,    8,    8,    8,   43,    8,   43,   43,    8,    8,    8,    8,   43,    8,
      25,    8,   25,    8,    8,    8,   43,    8,    8,   25,   25,    8,    8,    8,   43,    8,
       8,    8,   43,    8,    8,    8,   43,    8,   25,    8,    8,   25,    8,    8,   43,    8,
       8,   25,    8,   25,    8,    8,   43,    8,    8,    8,   25,   25,    8,    8,   43,    8,
       8,    8,    8,   43,    8,    8,   43,    8,    8,    8,   43,   43,    8,    8,   43,    8,
      25,    8,    8,    8,   25,    8,   43,    8,    8,   25,    8,    8,   25,    8,   43,    8,
       8,    8,   25,    8,   25,    8,   43,    8,    8,    8,    8,   25,   25,    8,   43,    8,
       8,   43,    8,   25,   25,    8,   43,    8,   25,   25,   43,   25,   25,    8,   43,    8,
       8,    8,    8,    8,   43,    8,   43,    8,   43,    8,   43,    8,   43,    8,   43,    8,
       8,    8,    8,   43,   43,    8,   43,    8,    8,   43,   43,   43,   43,    8,   43,    8,
      25,    8,    8,    8,    8,   25,   43,    8,    8,   25,    8,    8,    8,   25,   43,    8,
       8,    8,   25,    8,    8,   25,   43,    8,   25,   43,   43,    8,    8,   25,   43,    8,
       8,    8,    8,   25,    8,   25,   43,    8,    8,    8,    8,    8,   25,   25,   43,    8,
      25,    8,    8,   25,   25,   25,   43,    8,   43,    8,   25,   25,   25,   25,   43,    8,
      25,   43,   25,   43,   25,   25,   43,    8,   25,    8,    8,    8,   43,   25,   43,    8,
      43,   43,   25,    8,   43,   25,   43,    8,   43,   25,   43,   43,   43,   25,   43,    8,
       8,    8,    8,    8,    8,   43,   43,    8,    8,   43,    8,    8,    8,   43,   43,    8,
      43,   43,    8,    8,    8,   43,   43,    8,    8,    8,   43,    8,    8,   43,   43,    8,
      25,   25,   25,   25,    8,   43,   43,    8,    8,   43,    8,   43,    8,   43,   43,    8,
      43,    8,   43,   43,    8,   43,   43,    8,    8,   43,   43,   25,   25,   43,   43,    8,
       8,    8,   25,   43,   25,   43,   43,    8,    8,   43,    8,    8,   43,   43,   43,    8,
       8,    8,   43,    8,   43,   43,   43,    8,   43,    8,    8,   43,   43,   43,   43,    8,
       8,   43,    8,   43,   43,   43,   43,    8,   43,   43,    8,   43,   43,   43,   43,    8,
      25,    8,    8,    8,    8,    8,    8,   25,    8,   25,    8,    8,    8,    8,    8,   25,
      43,   25,    8,    8,    8,    8,    8,   25,   25,   43,    8,    8,    8,    8,    8,   25,
       8,    8,   25,    8,    8,    8,    8,   25,   43,    8,   25,    8,    8,    8,    8,   25,
      25,   25,   25,    8,    8,    8,    8,   25,    8,   43,   25,    8,    8,    8,    8,   25,
      25,    8,   43,    8,    8,    8,    8,   25,    8,   25,   43,    8,    8,    8,    8,   25,
       8,    8,    8,   25,    8,    8,    8,   25,   43,    8,    8,   25,    8,    8,    8,   25,
      25,   25,    8,   25,    8,    8,    8,   25,    8,   43,    8,   25,    8,    8,    8,   25,
      43,   43,    8,   25,    8,    8,    8,   25,   25,    8,   25,   25,    8,    8,    8,   25,
       8,   25,   25,   25,    8,    8,    8,   25,    8,    8,   43,   25,    8,    8,    8,   25,
      25,   25,   43,   25,    8,    8,    8,   25,   25,    8,    8,   43,    8,    8,    8,   25,
       8,   25,    8,   43,    8,    8,    8,   25,    8,    8,   25,   43,    8,    8,    8,   25,
       8,    8,    8,    8,   25,    8,    8,   25,   43,    8,    8,    8,   25,    8,    8,   25,
      25,   25,    8,    8,   25,    8,    8,   25,    8,   43,    8,    8,   25,    8,    8,   25,
      25,    8,   25,    8,   25,    8,    8,   25,    8,   25,   25,    8,   25,    8,    8,   25,
       8,    8,   43,    8,   25,    8,    8,   25,   25,    8,    8,   25,   25,    8,    8,   25,
       8,   25,    8,   25,   25,    8,    8,   25,    8,    8,   25,   25,   25,    8,    8,   25,
       8,    8,    8,   43,   25,    8,    8,   25,   25,   25,    8,   43,   25,    8,    8,   25,
      43,    8,   43,   43,   25,    8,    8,   25,   25,    8,    8,    8,   43,    8,    8,   25,
       8,   25,    8,    8,   43,    8,    8,   25,    8,    8,   25,    8,   43,    8,    8,   25,
      43,    8,   25,    8,   43,    8,    8,   25,   25,   43,   43,    8,   43,    8,    8,   25,
       8,    8,    8,   25,   43,    8,    8,   25,    8,    8,    8,    8,    8,   25,    8,   25,
      43,    8,    8,    8,    8,   25,    8,   25,   25,   25,    8,    8,    8,   25,    8,   25,
       8,   43,    8,    8,    8,   25,    8,   25,   25,    8,   25,    8,    8,   25,    8,   25,
       8,   25,   25,    8,    8,   25,    8,   25,   25,   43,   25,    8,    8,   25,    8,   25,
       8,    8,   43,    8,    8,   25,    8,   25,   25,    8,    8,   25,    8,   25,    8,   25,
       8,   25,    8,   25,    8,   25,    8,   25,    8,    8,   25,   25,    8,   25,    8,   25,
       8,    8,    8,   43,    8,   25,    8,   25,    8,   25,   25,   43,    8,   25,    8,   25,
      25,    8,    8,    8,   25,   25,    8,   25,    8,   25,    8,    8,   25,   25,    8,   25,
       8,    8,   25,    8,   25,   25,    8,   25,    8,   25,   43,    8,   25,   25,    8,   25,
       8,    8,    8,   25,   25,   25,    8,   25,   43,   43,   25,   43,   25,   25,    8,   25,
       8,    8,    8,    8,   43,   25,    8,   25,   43,   43,    8,    8,   43,   25,    8,   25,
       8,   25,    8,   25,   43,   25,    8,   25,    8,    8,   25,   25,   43,   25,    8,   25,
      25,    8,    8,    8,    8,   43,    8,   25,    8,   25,    8,    8,    8,   43,    8,   25,
       8,    8,   25,    8,    8,   43,    8,   25,    8,    8,    8,   25,    8,   43,    8,   25,
      25,   25,    8,   25,    8,   43,    8,   25,    8,   25,   25,   25,    8,   43,    8,   25,
      43,    8,   43,   25,    8,   43,    8,   25,    8,    8,    8,    8,   25,   43,    8,   25,
      25,    8,   25,    8,   25,   43,    8,   25,    8,   25,    8,   25,   25,   43,    8,   25,
       8,    8,   25,   25,   25,   43,    8,   25,   25,   43,   43,   25,   25,   43,    8,   25,
       8,   25,    8,    8,   43,   43,    8,   25,    8,    8,    8,    8,    8,    8,   25,   25,
      43,    8,    8,    8,    8,    8,   25,   25,   25,   25,    8,    8,    8,    8,   25,   25,
       8,   43,    8,    8,    8,    8,   25,   25,   25,    8,   25,    8,    8,    8,   25,   25,
       8,   25,   25,    8,    8,    8,   25,   25,    8,    8,   43,    8,    8,    8,   25,   25,
       8,   43,   43,    8,    8,    8,   25,   25,   25,    8,    8,   25,    8,    8,   25,   25,
       8,   25,    8,   25,    8,    8,   25,   25,    8,    8,   25,   25,    8,    8,   25,   25,
       8,    8,    8,   43,    8,    8,   25,   25,   25,    8,    8,    8,   25,    8,   25,   25,
       8,   25,    8,    8,   25,    8,   25,   25,    8,    8,   25,    8,   25,    8,   25,   25,
      25,   25,   25,    8,   25,    8,   25,   25,    8,    8,    8,   25,   25,    8,   25,   25,
      43,    8,    8,   25,   25,    8,   25,   25,    8,    8,    8,    8,   43,    8,   25,   25,
       8,   25,    8,   25,   43,    8,   25,   25,   43,   43,   43,   43,   43,    8,   25,   25,
      25,    8,    8,    8,    8,   25,   25,   25,    8,   25,    8,    8,    8,   25,   25,   25,
       8,    8,   25,    8,    8,   25,   25,   25,   25,    8,   43,    8,    8,   25,   25,   25,
       8,    8,    8,   25,    8,   25,   25,   25,    8,    8,   43,   25,    8,   25,   25,   25,
      25,    8,    8,   43,    8,   25,   25,   25,   25,    8,   43,   43,    8,   25,   25,   25,
       8,    8,    8,    8,   25,   25,   25,   25,    8,   43,    8,    8,   25,   25,   25,   25,
       8,    8,    8,   43,   25,   25,   25,   25,    8,   43,    8,   43,   25,   25,   25,   25,
      25,    8,   43,    8,   43,   25,   25,   25,    8,   43,   43,   25,   43,   25,   25,   25,
      25,    8,   43,   43,   43,   25,   25,   25,    8,    8,    8,    8,    8,   43,   25,   25,
       8,   25,   25,    8,    8,   43,   25,   25,   25,    8,    8,   25,    8,   43,   25,   25,
       8,    8,   25,   25,    8,   43,   25,   25,   25,   43,   25,   43,    8,   43,   25,   25,
      43,   43,   25,    8,   25,   43,   25,   25,    8,    8,    8,   25,   25,   43,   25,   25,
      43,    8,    8,   25,   25,   43,   25,   25,   25,   25,    8,   43,   43,   43,   25,   25,
      25,    8,    8,    8,    8,    8,   43,   25,    8,   25,    8,    8,    8,    8,   43,   25,
       8,    8,   25,    8,    8,    8,   43,   25,    8,    8,    8,   25,    8,    8,   43,   25,
       8,   25,   25,   25,    8,    8,   43,   25,   43,    8,   43,   25,    8,    8,   43,   25,
      43,   25,    8,   43,    8,    8,   43,   25,   25,   43,   43,   43,    8,    8,   43,   25,
       8,    8,    8,    8,   25,    8,   43,   25,    8,   25,   43,    8,   43,    8,   43,   25,
      43,   43,    8,   25,   43,    8,   43,   25,   43,    8,   25,   43,   43,    8,   43,   25,
       8,    8,    8,    8,    8,   25,   43,   25,   43,   25,   25,    8,    8,   25,   43,   25,
       8,    8,   25,    8,   25,   25,   43,   25,    8,    8,    8,   25,   25,   25,   43,   25,
      25,   25,    8,   25,   25,   25,   43,   25,    8,   25,   43,   43,   25,   25,   43,   25,
      25,    8,    8,    8,    8,   43,   43,   25,   43,   43,   43,   25,    8,   43,   43,   25,
      25,   25,   43,    8,   25,   43,   43,   25,   43,   25,    8,    8,   43,   43,   43,   25,
       8,   25,   25,   25,   43,   43,   43,   25,   43,    8,   43,   25,   43,   43,   43,   25,
       8,    8,    8,    8,    8,    8,    8,   43,   43,    8,    8,    8,    8,    8,    8,   43,
      25,   25,    8,    8,    8,    8,    8,   43,    8,   43,    8,    8,    8,    8,    8,   43,
      25,    8,   25,    8,    8,    8,    8,   43,    8,   25,   25,    8,    8,    8,    8,   43,
       8,    8,   43,    8,    8,    8,    8,   43,   43,   43,   43,    8,    8,    8,    8,   43,
      25,    8,    8,   25,    8,    8,    8,   43,    8,   25,    8,   25,    8,    8,    8,   43,
       8,    8,   25,   25,    8,    8,    8,   43,    8,    8,    8,   43,    8,    8,    8,   43,
      43,    8,    8,   43,    8,    8,    8,   43,    8,   43,   43,   43,    8,    8,    8,   43,
      43,   43,   43,   43,    8,    8,    8,   43,   25,    8,    8,    8,   25,    8,    8,   43,
       8,   25,    8,    8,   25,    8,    8,   43,   43,   25,    8,    8,   25,    8,    8,   43,
       8,    8,   25,    8,   25,    8,    8,   43,    8,    8,    8,   25,   25,    8,    8,   43,
      25,    8,   25,   25,   25,    8,    8,   43,   25,   43,   25,   25,   25,    8,    8,   43,
       8,    8,    8,    8,   43,    8,    8,   43,    8,    8,   43,    8,   43,    8,    8,   43,
       8,    8,    8,   43,   43,    8,    8,   43,   43,    8,    8,   43,   43,    8,    8,   43,
       8,    8,   43,   43,   43,    8,    8,   43,    8,   43,   43,   43,   43,    8,    8,   43,
      25,    8,    8,    8,    8,   25,    8,   43,    8,   25,    8,    8,    8,   25,    8,   43,
       8,    8,   25,    8,    8,   25,    8,   43,   43,    8,   25,    8,    8,   25,    8,   43,
      25,   25,   25,    8,    8,   25,    8,   43,    8,    8,    8,   25,    8,   25,    8,   43,
       8,    8,   43,   25,    8,   25,    8,   43,   25,   43,    8,   43,    8,   25,    8,   43,
       8,    8,    8,    8,   25,   25,    8,   43,    8,   25,    8,   25,   25,   25,    8,   43,
      25,   25,   43,   43,   25,   25,    8,   43,    8,   43,   25,    8,   43,   25,    8,   43,
      43,   43,   43,   25,   43,   25,    8,   43,    8,    8,    8,    8,    8,   43,    8,   43,
       8,   43,    8,    8,    8,   43,    8,   43,   25,   25,   43,    8,    8,   43,    8,   43,
      43,   43,   25,   25,    8,   43,    8,   43,    8,    8,    8,   43,    8,   43,    8,   43,
      43,    8,    8,   43,    8,   43,    8,   43,    8,   43,   43,   43,    8,   43,    8,   43,
      43,   25,    8,    8,   25,   43,    8,   43,   43,    8,   43,    8,   43,   43,    8,   43,
       8,    8,    8,   43,   43,   43,    8,   43,    8,   43,    8,   43,   43,   43,    8,   43,
      43,   25,   25,   43,   43,   43,    8,   43,    8,   43,   43,   43,   43,   43,    8,   43,
      25,    8,    8,    8,    8,    8,   25,   43,    8,   25,    8,    8,    8,    8,   25,   43,
       8,    8,   25,    8,    8,    8,   25,   43,    8,    8,    8,   25,    8,    8,   25,   43,
      43,   25,   25,   25,    8,    8,   25,   43,    8,   25,    8,   43,    8,    8,   25,   43,
       8,    8,    8,    8,   25,    8,   25,   43,   43,    8,   43,    8,   25,    8,   25,   43,
       8,   25,   43,   25,   25,    8,   25,   43,   43,   25,   25,   25,   43,    8,   25,   43,
      25,   43,    8,   43,   43,    8,   25,   43,    8,    8,    8,    8,    8,   25,   25,   43,
      25,   25,    8,    8,    8,   25,   25,   43,    8,   25,    8,   25,    8,   25,   25,   43,
       8,    8,   25,   25,    8,   25,   25,   43,    8,   43,   25,   25,    8,   25,   25,   43,
      25,   43,   43,    8,   25,   25,   25,   43,    8,    8,   25,   43,   25,   25,   25,   43,
      43,    8,   25,   43,   25,   25,   25,   43,   25,    8,    8,   25,   43,   25,   25,   43,
      25,    8,   25,   25,    8,   43,   25,   43,   43,   25,   43,   43,    8,   43,   25,   43,
      25,   43,    8,   25,   25,   43,   25,   43,   25,   25,   25,    8,   43,   43,   25,   43,
       8,    8,   43,   25,   43,   43,   25,   43,    8,    8,    8,    8,    8,    8,   43,   43,
      43,    8,    8,    8,    8,    8,   43,   43,    8,   43,    8,    8,    8,    8,   43,   43,
      43,   43,    8,    8,    8,    8,   43,   43,    8,    8,   43,    8,    8,    8,   43,   43,
      43,   43,   43,    8,    8,    8,   43,   43,    8,    8,   43,   43,    8,    8,   43,   43,
      25,    8,   25,   25,   25,    8,   43,   43,   25,   43,   25,   25,   25,    8,   43,   43,
      43,   25,   43,   43,   25,    8,   43,   43,    8,    8,    8,    8,   43,    8,   43,   43,
      43,    8,    8,    8,   43,    8,   43,   43,    8,   43,    8,    8,   43,    8,   43,   43,
      43,   43,   43,    8,   43,    8,   43,   43,    8,    8,    8,   43,   43,    8,   43,   43,
       8,    8,   43,   43,   43,    8,   43,   43,    8,    8,    8,   25,    8,   25,   43,   43,
      25,   25,   25,   43,    8,   25,   43,   43,   25,   25,   43,   25,   43,   25,   43,   43,
       8,   43,   25,   43,   43,   25,   43,   43,   43,   43,    8,    8,    8,   43,   43,   43,
       8,    8,   43,    8,    8,   43,   43,   43,   43,    8,   43,    8,    8,   43,   43,   43,
       8,   43,   43,    8,    8,   43,   43,   43,    8,    8,   43,   43,    8,   43,   43,   43,
       8,   43,   43,   43,    8,   43,   43,   43,    8,   25,    8,    8,   25,   43,   43,   43,
       8,   25,    8,   43,   25,   43,   43,   43,   43,   25,    8,   43,   25,   43,   43,   43,
       8,   43,   43,    8,   43,   43,   43,   43,   43,   43,   43,    8,   43,   43,   43,   43,
      25,    8,   25,   43,   43,   43,   43,   43,   43,   43,   43,   43,   43,   43,   43,   43,
};

#define INGOT_IQ2XS_GRID_ENTRIES 512
#define INGOT_IQ2XS_GRID_WIDTH   8

/* IQ2_S: 1024 entries x 8 values, map (8, 25, 43) */
static const int8_t ingot_iq2s_grid[8192] = {
       8,    8,    8,    8,    8,    8,    8,    8,   43,    8,    8,    8,    8,    8,    8,    8,
      25,   25,    8,    8,    8,    8,    8,    8,    8,   43,    8,    8,    8,    8,    8,    8,
      43,   43,    8,    8,    8,    8,    8,    8,   25,    8,   25,    8,    8,    8,    8,    8,
       8,   25,   25,    8,    8,    8,    8,    8,   43,   25,   25,    8,    8,    8,    8,    8,
      25,   43,   25,    8,    8,    8,    8,    8,    8,    8,   43,    8,    8,    8,    8,    8,
      43,    8,   43,    8,    8,    8,    8,    8,   25,   25,   43,    8,    8,    8,    8,    8,
       8,   43,   43,    8,    8,    8,    8,    8,   25,    8,    8,   25,    8,    8,    8,    8,
       8,   25,    8,   25,    8,    8,    8,    8,   43,   25,    8,   25,    8,    8,    8,    8,
      25,   43,    8,   25,    8,    8,    8,    8,    8,    8,   25,   25,    8,    8,    8,    8,
      43,    8,   25,   25,    8,    8,    8,    8,   25,   25,   25,   25,    8,    8,    8,    8,
       8,   43,   25,   25,    8,    8,    8,    8,   25,    8,   43,   25,    8,    8,    8,    8,
       8,   25,   43,   25,    8,    8,    8,    8,   43,   25,   43,   25,    8,    8,    8,    8,
      25,   43,   43,   25,    8,    8,    8,    8,    8,    8,    8,   43,    8,    8,    8,    8,
      43,    8,    8,   43,    8,    8,    8,    8,   25,   25,    8,   43,    8,    8,    8,    8,
       8,   43,    8,   43,    8,    8,    8,    8,   25,    8,   25,   43,    8,    8,    8,    8,
       8,   25,   25,   43,    8,    8,    8,    8,    8,    8,   43,   43,    8,    8,    8,    8,
      25,   25,   43,   43,    8,    8,    8,    8,   43,   43,   43,   43,    8,    8,    8,    8,
      25,    8,    8,    8,   25,    8,    8,    8,    8,   25,    8,    8,   25,    8,    8,    8,
      43,   25,    8,    8,   25,    8,    8,    8,   25,   43,    8,    8,   25,    8,    8,    8,
       8,    8,   25,    8,   25,    8,    8,    8,   43,    8,   25,    8,   25,    8,    8,    8,
      25,   25,   25,    8,   25,    8,    8,    8,    8,   43,   25,    8,   25,    8,    8,    8,
      25,    8,   43,    8,   25,    8,    8,    8,    8,   25,   43,    8,   25,    8,    8,    8,
       8,    8,    8,   25,   25,    8,    8,    8,   43,    8,    8,   25,   25,    8,    8,    8,
      25,   25,    8,   25,   25,    8,    8,    8,    8,   43,    8,   25,   25,    8,    8,    8,
      25,    8,   25,   25,   25,    8,    8,    8,    8,   25,   25,   25,   25,    8,    8,    8,
      43,   25,   25,   25,   25,    8,    8,    8,   25,   43,   25,   25,   25,    8,    8,    8,
       8,    8,   43,   25,   25,    8,    8,    8,   25,   25,   43,   25,   25,    8,    8,    8,
       8,   43,   43,   25,   25,    8,    8,    8,   25,    8,    8,   43,   25,    8,    8,    8,
       8,   25,    8,   43,   25,    8,    8,    8,    8,    8,   25,   43,   25,    8,    8,    8,
      43,    8,   25,   43,   25,    8,    8,    8,   25,   25,   25,   43,   25,    8,    8,    8,
      25,    8,   43,   43,   25,    8,    8,    8,    8,   25,   43,   43,   25,    8,    8,    8,
       8,    8,    8,    8,   43,    8,    8,    8,   43,    8,    8,    8,   43,    8,    8,    8,
      25,   25,    8,    8,   43,    8,    8,    8,    8,   43,    8,    8,   43,    8,    8,    8,
      25,    8,   25,    8,   43,    8,    8,    8,    8,   25,   25,    8,   43,    8,    8,    8,
       8,    8,   43,    8,   43,    8,    8,    8,   43,   43,   43,    8,   43,    8,    8,    8,
      25,    8,    8,   25,   43,    8,    8,    8,    8,   25,    8,   25,   43,    8,    8,    8,
      43,   25,    8,   25,   43,    8,    8,    8,   25,   43,    8,   25,   43,    8,    8,    8,
       8,    8,   25,   25,   43,    8,    8,    8,   25,   25,   25,   25,   43,    8,    8,    8,
       8,    8,    8,   43,   43,    8,    8,    8,   25,   25,    8,   43,   43,    8,    8,    8,
      43,   43,    8,   43,   43,    8,    8,    8,    8,   25,   25,   43,   43,    8,    8,    8,
      43,    8,   43,   43,   43,    8,    8,    8,   25,    8,    8,    8,    8,   25,    8,    8,
       8,   25,    8,    8,    8,   25,    8,    8,   43,   25,    8,    8,    8,   25,    8,    8,
      25,   43,    8,    8,    8,   25,    8,    8,    8,    8,   25,    8,    8,   25,    8,    8,
      43,    8,   25,    8,    8,   25,    8,    8,   25,   25,   25,    8,    8,   25,    8,    8,
       8,   43,   25,    8,    8,   25,    8,    8,   25,    8,   43,    8,    8,   25,    8,    8,
       8,   25,   43,    8,    8,   25,    8,    8,   43,   25,   43,    8,    8,   25,    8,    8,
      25,   43,   43,    8,    8,   25,    8,    8,    8,    8,    8,   25,    8,   25,    8,    8,
      43,    8,    8,   25,    8,   25,    8,    8,   25,   25,    8,   25,    8,   25,    8,    8,
       8,   43,    8,   25,    8,   25,    8,    8,   43,   43,    8,   25,    8,   25,    8,    8,
      25,    8,   25,   25,    8,   25,    8,    8,    8,   25,   25,   25,    8,   25,    8,    8,
      43,   25,   25,   25,    8,   25,    8,    8,   25,   43,   25,   25,    8,   25,    8,    8,
       8,    8,   43,   25,    8,   25,    8,    8,   43,    8,   43,   25,    8,   25,    8,    8,
      25,   25,   43,   25,    8,   25,    8,    8,   25,    8,    8,   43,    8,   25,    8,    8,
       8,   25,    8,   43,    8,   25,    8,    8,   43,   25,    8,   43,    8,   25,    8,    8,
      25,   43,    8,   43,    8,   25,    8,    8,    8,    8,   25,   43,    8,   25,    8,    8,
      25,   25,   25,   43,    8,   25,    8,    8,    8,   43,   25,   43,    8,   25,    8,    8,
      25,    8,   43,   43,    8,   25,    8,    8,    8,   25,   43,   43,    8,   25,    8,    8,
       8,    8,    8,    8,   25,   25,    8,    8,   43,    8,    8,    8,   25,   25,    8,    8,
      25,   25,    8,    8,   25,   25,    8,    8,    8,   43,    8,    8,   25,   25,    8,    8,
      43,   43,    8,    8,   25,   25,    8,    8,   25,    8,   25,    8,   25,   25,    8,    8,
       8,   25,   25,    8,   25,   25,    8,    8,   43,   25,   25,    8,   25,   25,    8,    8,
      25,   43,   25,    8,   25,   25,    8,    8,    8,    8,   43,    8,   25,   25,    8,    8,
      25,   25,   43,    8,   25,   25,    8,    8,    8,   43,   43,    8,   25,   25,    8,    8,
      25,    8,    8,   25,   25,   25,    8,    8,    8,   25,    8,   25,   25,   25,    8,    8,
      43,   25,    8,   25,   25,   25,    8,    8,   25,   43,    8,   25,   25,   25,    8,    8,
       8,    8,   25,   25,   25,   25,    8,    8,   43,    8,   25,   25,   25,   25,    8,    8,
      25,   25,   25,   25,   25,   25,    8,    8,    8,   43,   25,   25,   25,   25,    8,    8,
      25,    8,   43,   25,   25,   25,    8,    8,    8,   25,   43,   25,   25,   25,    8,    8,
       8,    8,    8,   43,   25,   25,    8,    8,   43,    8,    8,   43,   25,   25,    8,    8,
      25,   25,    8,   43,   25,   25,    8,    8,    8,   43,    8,   43,   25,   25,    8,    8,
      25,    8,   25,   43,   25,   25,    8,    8,    8,   25,   25,   43,   25,   25,    8,    8,
       8,    8,   43,   43,   25,   25,    8,    8,   25,    8,    8,    8,   43,   25,    8,    8,
       8,   25,    8,    8,   43,   25,    8,    8,   43,   25,    8,    8,   43,   25,    8,    8,
      25,   43,    8,    8,   43,   25,    8,    8,    8,    8,   25,    8,   43,   25,    8,    8,
      25,   25,   25,    8,   43,   25,    8,    8,    8,    8,    8,   25,   43,   25,    8,    8,
      25,   25,    8,   25,   43,   25,    8,    8,    8,   43,    8,   25,   43,   25,    8,    8,
      25,    8,   25,   25,   43,   25,    8,    8,    8,   25,   25,   25,   43,   25,    8,    8,
       8,    8,   43,   25,   43,   25,    8,    8,   25,    8,    8,   43,   43,   25,    8,    8,
       8,   25,    8,   43,   43,   25,    8,    8,    8,    8,   25,   43,   43,   25,    8,    8,
       8,    8,    8,    8,    8,   43,    8,    8,   43,    8,    8,    8,    8,   43,    8,    8,
      25,   25,    8,    8,    8,   43,    8,    8,    8,   43,    8,    8,    8,   43,    8,    8,
      25,    8,   25,    8,    8,   43,    8,    8,    8,   25,   25,    8,    8,   43,    8,    8,
      43,   25,   25,    8,    8,   43,    8,    8,   25,   43,   25,    8,    8,   43,    8,    8,
       8,    8,   43,    8,    8,   43,    8,    8,   25,   25,   43,    8,    8,   43,    8,    8,
      43,   43,   43,    8,    8,   43,    8,    8,   25,    8,    8,   25,    8,   43,    8,    8,
       8,   25,    8,   25,    8,   43,    8,    8,   43,   25,    8,   25,    8,   43,    8,    8,
      25,   43,    8,   25,    8,   43,    8,    8,    8,    8,   25,   25,    8,   43,    8,    8,
      43,    8,   25,   25,    8,   43,    8,    8,   25,   25,   25,   25,    8,   43,    8,    8,
       8,   43,   25,   25,    8,   43,    8,    8,   25,    8,   43,   25,    8,   43,    8,    8,
       8,   25,   43,   25,    8,   43,    8,    8,    8,    8,    8,   43,    8,   43,    8,    8,
      25,   25,    8,   43,    8,   43,    8,    8,    8,   25,   25,   43,    8,   43,    8,    8,
      43,   43,   43,   43,    8,   43,    8,    8,   25,    8,    8,    8,   25,   43,    8,    8,
       8,   25,    8,    8,   25,   43,    8,    8,    8,    8,   25,    8,   25,   43,    8,    8,
      43,    8,   25,    8,   25,   43,    8,    8,   25,   25,   25,    8,   25,   43,    8,    8,
       8,   43,   25,    8,   25,   43,    8,    8,   25,    8,   43,    8,   25,   43,    8,    8,
       8,    8,    8,   25,   25,   43,    8,    8,   25,   25,    8,   25,   25,   43,    8,    8,
       8,   43,    8,   25,   25,   43,    8,    8,   25,    8,   25,   25,   25,   43,    8,    8,
       8,   25,   25,   25,   25,   43,    8,    8,    8,    8,   43,   25,   25,   43,    8,    8,
      25,    8,    8,   43,   25,   43,    8,    8,    8,    8,   25,   43,   25,   43,    8,    8,
       8,    8,    8,    8,   43,   43,    8,    8,   25,    8,   25,    8,   43,   43,    8,    8,
       8,   25,   25,    8,   43,   43,    8,    8,   43,    8,   43,    8,   43,   43,    8,    8,
       8,   43,   43,    8,   43,   43,    8,    8,   43,   43,   43,    8,   43,   43,    8,    8,
       8,    8,   25,   25,   43,   43,    8,    8,   25,   43,   25,   43,   43,   43,    8,    8,
      25,    8,    8,    8,    8,    8,   25,    8,    8,   25,    8,    8,    8,    8,   25,    8,
      43,   25,    8,    8,    8,    8,   25,    8,   25,   43,    8,    8,    8,    8,   25,    8,
       8,    8,   25,    8,    8,    8,   25,    8,   43,    8,   25,    8,    8,    8,   25,    8,
      25,   25,   25,    8,    8,    8,   25,    8,    8,   43,   25,    8,    8,    8,   25,    8,
      25,    8,   43,    8,    8,    8,   25,    8,    8,   25,   43,    8,    8,    8,   25,    8,
      43,   25,   43,    8,    8,    8,   25,    8,    8,    8,    8,   25,    8,    8,   25,    8,
      43,    8,    8,   25,    8,    8,   25,    8,   25,   25,    8,   25,    8,    8,   25,    8,
       8,   43,    8,   25,    8,    8,   25,    8,   25,    8,   25,   25,    8,    8,   25,    8,
       8,   25,   25,   25,    8,    8,   25,    8,   43,   25,   25,   25,    8,    8,   25,    8,
      25,   43,   25,   25,    8,    8,   25,    8,    8,    8,   43,   25,    8,    8,   25,    8,
      43,    8,   43,   25,    8,    8,   25,    8,   25,   25,   43,   25,    8,    8,   25,    8,
       8,   43,   43,   25,    8,    8,   25,    8,   25,    8,    8,   43,    8,    8,   25,    8,
       8,   25,    8,   43,    8,    8,   25,    8,   43,   25,    8,   43,    8,    8,   25,    8,
       8,    8,   25,   43,    8,    8,   25,    8,   25,   25,   25,   43,    8,    8,   25,    8,
       8,   43,   25,   43,    8,    8,   25,    8,   25,    8,   43,   43,    8,    8,   25,    8,
       8,   25,   43,   43,    8,    8,   25,    8,    8,    8,    8,    8,   25,    8,   25,    8,
      43,    8,    8,    8,   25,    8,   25,    8,   25,   25,    8,    8,   25,    8,   25,    8,
       8,   43,    8,    8,   25,    8,   25,    8,   43,   43,    8,    8,   25,    8,   25,    8,
      25,    8,   25,    8,   25,    8,   25,    8,    8,   25,   25,    8,   25,    8,   25,    8,
      43,   25,   25,    8,   25,    8,   25,    8,   25,   43,   25,    8,   25,    8,   25,    8,
       8,    8,   43,    8,   25,    8,   25,    8,   43,    8,   43,    8,   25,    8,   25,    8,
      25,   25,   43,    8,   25,    8,   25,    8,    8,   43,   43,    8,   25,    8,   25,    8,
      25,    8,    8,   25,   25,    8,   25,    8,    8,   25,    8,   25,   25,    8,   25,    8,
      43,   25,    8,   25,   25,    8,   25,    8,   25,   43,    8,   25,   25,    8,   25,    8,
       8,    8,   25,   25,   25,    8,   25,    8,   43,    8,   25,   25,   25,    8,   25,    8,
      25,   25,   25,   25,   25,    8,   25,    8,    8,   43,   25,   25,   25,    8,   25,    8,
      25,    8,   43,   25,   25,    8,   25,    8,    8,   25,   43,   25,   25,    8,   25,    8,
       8,    8,    8,   43,   25,    8,   25,    8,   43,    8,    8,   43,   25,    8,   25,    8,
      25,   25,    8,   43,   25,    8,   25,    8,    8,   43,    8,   43,   25,    8,   25,    8,
      25,    8,   25,   43,   25,    8,   25,    8,    8,   25,   25,   43,   25,    8,   25,    8,
      25,    8,    8,    8,   43,    8,   25,    8,    8,   25,    8,    8,   43,    8,   25,    8,
      25,   43,    8,    8,   43,    8,   25,    8,    8,    8,   25,    8,   43,    8,   25,    8,
      25,   25,   25,    8,   43,    8,   25,    8,   25,    8,   43,    8,   43,    8,   25,    8,
       8,   25,   43,    8,   43,    8,   25,    8,    8,    8,    8,   25,   43,    8,   25,    8,
      25,   25,    8,   25,   43,    8,   25,    8,   25,    8,   25,   25,   43,    8,   25,    8,
       8,   25,   25,   25,   43,    8,   25,    8,   25,    8,    8,   43,   43,    8,   25,    8,
       8,   25,    8,   43,   43,    8,   25,    8,    8,    8,   25,   43,   43,    8,   25,    8,
       8,    8,    8,    8,    8,   25,   25,    8,   43,    8,    8,    8,    8,   25,   25,    8,
      25,   25,    8,    8,    8,   25,   25,    8,    8,   43,    8,    8,    8,   25,   25,    8,
      25,    8,   25,    8,    8,   25,   25,    8,    8,   25,   25,    8,    8,   25,   25,    8,
      43,   25,   25,    8,    8,   25,   25,    8,   25,   43,   25,    8,    8,   25,   25,    8,
       8,    8,   43,    8,    8,   25,   25,    8,   25,   25,   43,    8,    8,   25,   25,    8,
       8,   43,   43,    8,    8,   25,   25,    8,   25,    8,    8,   25,    8,   25,   25,    8,
       8,   25,    8,   25,    8,   25,   25,    8,   43,   25,    8,   25,    8,   25,   25,    8,
      25,   43,    8,   25,    8,   25,   25,    8,    8,    8,   25,   25,    8,   25,   25,    8,
      43,    8,   25,   25,    8,   25,   25,    8,   25,   25,   25,   25,    8,   25,   25,    8,
       8,   43,   25,   25,    8,   25,   25,    8,   25,    8,   43,   25,    8,   25,   25,    8,
       8,   25,   43,   25,    8,   25,   25,    8,    8,    8,    8,   43,    8,   25,   25,    8,
      43,    8,    8,   43,    8,   25,   25,    8,   25,   25,    8,   43,    8,   25,   25,    8,
       8,   43,    8,   43,    8,   25,   25,    8,   25,    8,   25,   43,    8,   25,   25,    8,
       8,   25,   25,   43,    8,   25,   25,    8,    8,    8,   43,   43,    8,   25,   25,    8,
      25,    8,    8,    8,   25,   25,   25,    8,    8,   25,    8,    8,   25,   25,   25,    8,
      43,   25,    8,    8,   25,   25,   25,    8,   25,   43,    8,    8,   25,   25,   25,    8,
       8,    8,   25,    8,   25,   25,   25,    8,   43,    8,   25,    8,   25,   25,   25,    8,
      25,   25,   25,    8,   25,   25,   25,    8,    8,   43,   25,    8,   25,   25,   25,    8,
      25,    8,   43,    8,   25,   25,   25,    8,    8,   25,   43,    8,   25,   25,   25,    8,
       8,    8,    8,   25,   25,   25,   25,    8,   43,    8,    8,   25,   25,   25,   25,    8,
      25,   25,    8,   25,   25,   25,   25,    8,    8,   43,    8,   25,   25,   25,   25,    8,
      25,    8,   25,   25,   25,   25,   25,    8,    8,   25,   25,   25,   25,   25,   25,    8,
       8,    8,   43,   25,   25,   25,   25,    8,   25,    8,    8,   43,   25,   25,   25,    8,
       8,   25,    8,   43,   25,   25,   25,    8,    8,    8,   25,   43,   25,   25,   25,    8,
       8,    8,    8,    8,   43,   25,   25,    8,   25,   25,    8,    8,   43,   25,   25,    8,
       8,   43,    8,    8,   43,   25,   25,    8,   25,    8,   25,    8,   43,   25,   25,    8,
       8,   25,   25,    8,   43,   25,   25,    8,    8,    8,   43,    8,   43,   25,   25,    8,
      25,    8,    8,   25,   43,   25,   25,    8,    8,   25,    8,   25,   43,   25,   25,    8,
       8,    8,   25,   25,   43,   25,   25,    8,    8,    8,    8,   43,   43,   25,   25,    8,
      43,   43,   43,   43,   43,   25,   25,    8,   25,    8,    8,    8,    8,   43,   25,    8,
       8,   25,    8,    8,    8,   43,   25,    8,   43,   25,    8,    8,    8,   43,   25,    8,
      25,   43,    8,    8,    8,   43,   25,    8,    8,    8,   25,    8,    8,   43,   25,    8,
      25,   25,   25,    8,    8,   43,   25,    8,    8,   43,   25,    8,    8,   43,   25,    8,
      25,    8,   43,    8,    8,   43,   25,    8,    8,    8,    8,   25,    8,   43,   25,    8,
      43,    8,    8,   25,    8,   43,   25,    8,   25,   25,    8,   25,    8,   43,   25,    8,
       8,   43,    8,   25,    8,   43,   25,    8,   25,    8,   25,   25,    8,   43,   25,    8,
       8,   25,   25,   25,    8,   43,   25,    8,    8,    8,   43,   25,    8,   43,   25,    8,
      25,    8,    8,   43,    8,   43,   25,    8,    8,   25,    8,   43,    8,   43,   25,    8,
       8,    8,    8,    8,   25,   43,   25,    8,   43,    8,    8,    8,   25,   43,   25,    8,
      25,   25,    8,    8,   25,   43,   25,    8,    8,   43,    8,    8,   25,   43,   25,    8,
      25,    8,   25,    8,   25,   43,   25,    8,    8,   25,   25,    8,   25,   43,   25,    8,
       8,    8,   43,    8,   25,   43,   25,    8,   25,    8,    8,   25,   25,   43,   25,    8,
       8,   25,    8,   25,   25,   43,   25,    8,    8,    8,   25,   25,   25,   43,   25,    8,
      25,   43,   43,   25,   25,   43,   25,    8,   43,    8,   43,   43,   25,   43,   25,    8,
       8,   25,    8,    8,   43,   43,   25,    8,    8,    8,   25,    8,   43,   43,   25,    8,
       8,    8,    8,   25,   43,   43,   25,    8,   43,   25,   25,   25,   43,   43,   25,    8,
       8,    8,    8,    8,    8,    8,   43,    8,   43,    8,    8,    8,    8,    8,   43,    8,
      25,   25,    8,    8,    8,    8,   43,    8,    8,   43,    8,    8,    8,    8,   43,    8,
      25,    8,   25,    8,    8,    8,   43,    8,    8,   25,   25,    8,    8,    8,   43,    8,
      43,   25,   25,    8,    8,    8,   43,    8,   25,   43,   25,    8,    8,    8,   43,    8,
       8,    8,   43,    8,    8,    8,   43,    8,   25,   25,   43,    8,    8,    8,   43,    8,
      43,   43,   43,    8,    8,    8,   43,    8,   25,    8,    8,   25,    8,    8,   43,    8,
       8,   25,    8,   25,    8,    8,   43,    8,    8,    8,   25,   25,    8,    8,   43,    8,
      43,    8,   25,   25,    8,    8,   43,    8,   25,   25,   25,   25,    8,    8,   43,    8,
       8,   25,   43,   25,    8,    8,   43,    8,    8,    8,    8,   43,    8,    8,   43,    8,
      43,   43,    8,   43,    8,    8,   43,    8,    8,   25,   25,   43,    8,    8,   43,    8,
      43,   43,   43,   43,    8,    8,   43,    8,   25,    8,    8,    8,   25,    8,   43,    8,
       8,   25,    8,    8,   25,    8,   43,    8,    8,    8,   25,    8,   25,    8,   43,    8,
      43,    8,   25,    8,   25,    8,   43,    8,   25,   25,   25,    8,   25,    8,   43,    8,
      25,    8,   43,    8,   25,    8,   43,    8,    8,    8,    8,   25,   25,    8,   43,    8,
      43,    8,    8,   25,   25,    8,   43,    8,   25,   25,    8,   25,   25,    8,   43,    8,
      25,    8,   25,   25,   25,    8,   43,    8,    8,   25,   25,   25,   25,    8,   43,    8,
       8,    8,   43,   25,   25,    8,   43,    8,   25,    8,    8,   43,   25,    8,   43,    8,
       8,   25,    8,   43,   25,    8,   43,    8,    8,    8,   25,   43,   25,    8,   43,    8,
       8,    8,    8,    8,   43,    8,   43,    8,   43,   43,    8,    8,   43,    8,   43,    8,
      43,    8,   43,    8,   43,    8,   43,    8,    8,   43,   43,    8,   43,    8,   43,    8,
      43,   43,   43,    8,   43,    8,   43,    8,    8,   25,    8,   25,   43,    8,   43,    8,
       8,    8,   25,   25,   43,    8,   43,    8,    8,   43,    8,   43,   43,    8,   43,    8,
      43,   43,    8,   43,   43,    8,   43,    8,    8,   43,   43,   43,   43,    8,   43,    8,
      25,    8,    8,    8,    8,   25,   43,    8,    8,   25,    8,    8,    8,   25,   43,    8,
      43,   25,    8,    8,    8,   25,   43,    8,   25,   43,    8,    8,    8,   25,   43,    8,
       8,    8,   25,    8,    8,   25,   43,    8,   25,   25,   25,    8,    8,   25,   43,    8,
       8,   43,   25,    8,    8,   25,   43,    8,   25,    8,   43,    8,    8,   25,   43,    8,
       8,   25,   43,    8,    8,   25,   43,    8,    8,    8,    8,   25,    8,   25,   43,    8,
      43,    8,    8,   25,    8,   25,   43,    8,   25,   25,    8,   25,    8,   25,   43,    8,
       8,   43,    8,   25,    8,   25,   43,    8,   25,    8,   25,   25,    8,   25,   43,    8,
       8,   25,   25,   25,    8,   25,   43,    8,    8,    8,   43,   25,    8,   25,   43,    8,
      25,    8,    8,   43,    8,   25,   43,    8,    8,   25,    8,   43,    8,   25,   43,    8,
       8,    8,   25,   43,    8,   25,   43,    8,    8,    8,    8,    8,   25,   25,   43,    8,
      25,   25,    8,    8,   25,   25,   43,    8,    8,   43,    8,    8,   25,   25,   43,    8,
      25,    8,   25,    8,   25,   25,   43,    8,    8,   25,   25,    8,   25,   25,   43,    8,
       8,    8,   43,    8,   25,   25,   43,    8,   25,    8,    8,   25,   25,   25,   43,    8,
       8,   25,    8,   25,   25,   25,   43,    8,    8,    8,   25,   25,   25,   25,   43,    8,
      43,   25,   43,   25,   25,   25,   43,    8,    8,    8,    8,   43,   25,   25,   43,    8,
      25,    8,    8,    8,   43,   25,   43,    8,    8,   25,    8,    8,   43,   25,   43,    8,
       8,    8,   25,    8,   43,   25,   43,    8,    8,    8,    8,   25,   43,   25,   43,    8,
      25,   43,   25,   25,   43,   25,   43,    8,    8,    8,    8,    8,    8,   43,   43,    8,
      25,   25,    8,    8,    8,   43,   43,    8,   25,    8,   25,    8,    8,   43,   43,    8,
       8,   25,   25,    8,    8,   43,   43,    8,   25,    8,    8,   25,    8,   43,   43,    8,
       8,   25,    8,   25,    8,   43,   43,    8,    8,    8,   25,   25,    8,   43,   43,    8,
      43,   43,    8,   43,    8,   43,   43,    8,   43,   43,   43,   43,    8,   43,   43,    8,
      25,    8,    8,    8,   25,   43,   43,    8,    8,   25,    8,    8,   25,   43,   43,    8,
       8,    8,   25,    8,   25,   43,   43,    8,   25,   25,   25,   43,   25,   43,   43,    8,
      43,   43,    8,    8,   43,   43,   43,    8,   43,    8,   43,    8,   43,   43,   43,    8,
       8,   25,   43,   25,   43,   43,   43,    8,    8,   43,    8,   43,   43,   43,   43,    8,
      43,   43,    8,   43,   43,   43,   43,    8,   25,    8,    8,    8,    8,    8,    8,   25,
       8,   25,    8,    8,    8,    8,    8,   25,   43,   25,    8,    8,    8,    8,    8,   25,
      25,   43,    8,    8,    8,    8,    8,   25,    8,    8,   25,    8,    8,    8,    8,   25,
      43,    8,   25,    8,    8,    8,    8,   25,   25,   25,   25,    8,    8,    8,    8,   25,
       8,   43,   25,    8,    8,    8,    8,   25,   43,   43,   25,    8,    8,    8,    8,   25,
      25,    8,   43,    8,    8,    8,    8,   25,    8,   25,   43,    8,    8,    8,    8,   25,
      43,   25,   43,    8,    8,    8,    8,   25,    8,    8,    8,   25,    8,    8,    8,   25,
      43,    8,    8,   25,    8,    8,    8,   25,   25,   25,    8,   25,    8,    8,    8,   25,
       8,   43,    8,   25,    8,    8,    8,   25,   43,   43,    8,   25,    8,    8,    8,   25,
      25,    8,   25,   25,    8,    8,    8,   25,    8,   25,   25,   25,    8,    8,    8,   25,
      43,   25,   25,   25,    8,    8,    8,   25,   25,   43,   25,   25,    8,    8,    8,   25,
       8,    8,   43,   25,    8,    8,    8,   25,   43,    8,   43,   25,    8,    8,    8,   25,
      25,   25,   43,   25,    8,    8,    8,   25,   25,    8,    8,   43,    8,    8,    8,   25,
       8,   25,    8,   43,    8,    8,    8,   25,    8,    8,   25,   43,    8,    8,    8,   25,
      25,   25,   25,   43,    8,    8,    8,   25,    8,   43,   25,   43,    8,    8,    8,   25,
      25,    8,   43,   43,    8,    8,    8,   25,    8,   25,   43,   43,    8,    8,    8,   25,
       8,    8,    8,    8,   25,    8,    8,   25,   43,    8,    8,    8,   25,    8,    8,   25,
      25,   25,    8,    8,   25,    8,    8,   25,    8,   43,    8,    8,   25,    8,    8,   25,
      25,    8,   25,    8,   25,    8,    8,   25,    8,   25,   25,    8,   25,    8,    8,   25,
      43,   25,   25,    8,   25,    8,    8,   25,   25,   43,   25,    8,   25,    8,    8,   25,
       8,    8,   43,    8,   25,    8,    8,   25,   43,    8,   43,    8,   25,    8,    8,   25,
      25,   25,   43,    8,   25,    8,    8,   25,   25,    8,    8,   25,   25,    8,    8,   25,
       8,   25,    8,   25,   25,    8,    8,   25,   43,   25,    8,   25,   25,    8,    8,   25,
      25,   43,    8,   25,   25,    8,    8,   25,    8,    8,   25,   25,   25,    8,    8,   25,
      43,    8,   25,   25,   25,    8,    8,   25,   25,   25,   25,   25,   25,    8,    8,   25,
       8,   43,   25,   25,   25,    8,    8,   25,   25,    8,   43,   25,   25,    8,    8,   25,
       8,   25,   43,   25,   25,    8,    8,   25,    8,    8,    8,   43,   25,    8,    8,   25,
      43,    8,    8,   43,   25,    8,    8,   25,   25,   25,    8,   43,   25,    8,    8,   25,
       8,   43,    8,   43,   25,    8,    8,   25,   25,    8,   25,   43,   25,    8,    8,   25,
       8,   25,   25,   43,   25,    8,    8,   25,    8,    8,   43,   43,   25,    8,    8,   25,
      25,    8,    8,    8,   43,    8,    8,   25,    8,   25,    8,    8,   43,    8,    8,   25,
       8,    8,   25,    8,   43,    8,    8,   25,   43,    8,   25,    8,   43,    8,    8,   25,
      25,   25,   25,    8,   43,    8,    8,   25,    8,   43,   25,    8,   43,    8,    8,   25,
       8,   25,   43,    8,   43,    8,    8,   25,    8,    8,    8,   25,   43,    8,    8,   25,
      25,   25,    8,   25,   43,    8,    8,   25,    8,   43,    8,   25,   43,    8,    8,   25,
      25,    8,   25,   25,   43,    8,    8,   25,    8,   25,   25,   25,   43,    8,    8,   25,
       8,    8,   43,   25,   43,    8,    8,   25,   25,    8,    8,   43,   43,    8,    8,   25,
       8,   25,    8,   43,   43,    8,    8,   25,    8,    8,    8,    8,    8,   25,    8,   25,
      43,    8,    8,    8,    8,   25,    8,   25,   25,   25,    8,    8,    8,   25,    8,   25,
       8,   43,    8,    8,    8,   25,    8,   25,   43,   43,    8,    8,    8,   25,    8,   25,
      25,    8,   25,    8,    8,   25,    8,   25,    8,   25,   25,    8,    8,   25,    8,   25,
      43,   25,   25,    8,    8,   25,    8,   25,   25,   43,   25,    8,    8,   25,    8,   25,
       8,    8,   43,    8,    8,   25,    8,   25,   43,    8,   43,    8,    8,   25,    8,   25,
      25,   25,   43,    8,    8,   25,    8,   25,    8,   43,   43,    8,    8,   25,    8,   25,
      25,    8,    8,   25,    8,   25,    8,   25,    8,   25,    8,   25,    8,   25,    8,   25,
      43,   25,    8,   25,    8,   25,    8,   25,   25,   43,    8,   25,    8,   25,    8,   25,
       8,    8,   25,   25,    8,   25,    8,   25,   43,    8,   25,   25,    8,   25,    8,   25,
      25,   25,   25,   25,    8,   25,    8,   25,    8,   43,   25,   25,    8,   25,    8,   25,
      25,    8,   43,   25,    8,   25,    8,   25,    8,   25,   43,   25,    8,   25,    8,   25,
       8,    8,    8,   43,    8,   25,    8,   25,   43,    8,    8,   43,    8,   25,    8,   25,
      25,   25,    8,   43,    8,   25,    8,   25,    8,   43,    8,   43,    8,   25,    8,   25,
      25,    8,   25,   43,    8,   25,    8,   25,    8,   25,   25,   43,    8,   25,    8,   25,
       8,    8,   43,   43,    8,   25,    8,   25,   25,    8,    8,    8,   25,   25,    8,   25,
       8,   25,    8,    8,   25,   25,    8,   25,   43,   25,    8,    8,   25,   25,    8,   25,
      25,   43,    8,    8,   25,   25,    8,   25,    8,    8,   25,    8,   25,   25,    8,   25,
      43,    8,   25,    8,   25,   25,    8,   25,   25,   25,   25,    8,   25,   25,    8,   25,
       8,   43,   25,    8,   25,   25,    8,   25,   25,    8,   43,    8,   25,   25,    8,   25,
       8,   25,   43,    8,   25,   25,    8,   25,    8,    8,    8,   25,   25,   25,    8,   25,
      43,    8,    8,   25,   25,   25,    8,   25,   25,   25,    8,   25,   25,   25,    8,   25,
       8,   43,    8,   25,   25,   25,    8,   25,   25,    8,   25,   25,   25,   25,    8,   25,
       8,   25,   25,   25,   25,   25,    8,   25,    8,    8,   43,   25,   25,   25,    8,   25,
      43,   43,   43,   25,   25,   25,    8,   25,   25,    8,    8,   43,   25,   25,    8,   25,
       8,   25,    8,   43,   25,   25,    8,   25,    8,    8,   25,   43,   25,   25,    8,   25,
       8,    8,    8,    8,   43,   25,    8,   25,   43,    8,    8,    8,   43,   25,    8,   25,
      25,   25,    8,    8,   43,   25,    8,   25,    8,   43,    8,    8,   43,   25,    8,   25,
      25,    8,   25,    8,   43,   25,    8,   25,    8,   25,   25,    8,   43,   25,    8,   25,
       8,    8,   43,    8,   43,   25,    8,   25,   25,    8,    8,   25,   43,   25,    8,   25,
       8,   25,    8,   25,   43,   25,    8,   25,    8,    8,   25,   25,   43,   25,    8,   25,
       8,    8,    8,   43,   43,   25,    8,   25,   25,   25,   43,   43,   43,   25,    8,   25,
      25,    8,    8,    8,    8,   43,    8,   25,    8,   25,    8,    8,    8,   43,    8,   25,
      25,   43,    8,    8,    8,   43,    8,   25,    8,    8,   25,    8,    8,   43,    8,   25,
      43,    8,   25,    8,    8,   43,    8,   25,   25,   25,   25,    8,    8,   43,    8,   25,
       8,   43,   25,    8,    8,   43,    8,   25,   25,    8,   43,    8,    8,   43,    8,   25,
       8,   25,   43,    8,    8,   43,    8,   25,    8,    8,    8,   25,    8,   43,    8,   25,
      43,    8,    8,   25,    8,   43,    8,   25,   25,   25,    8,   25,    8,   43,    8,   25,
       8,   43,    8,   25,    8,   43,    8,   25,   25,    8,   25,   25,    8,   43,    8,   25,
       8,   25,   25,   25,    8,   43,    8,   25,    8,    8,   43,   25,    8,   43,    8,   25,
       8,   25,    8,   43,    8,   43,    8,   25,    8,    8,   25,   43,    8,   43,    8,   25,
       8,    8,    8,    8,   25,   43,    8,   25,   43,    8,    8,    8,   25,   43,    8,   25,
      25,   25,    8,    8,   25,   43,    8,   25,    8,   43,    8,    8,   25,   43,    8,   25,
      25,    8,   25,    8,   25,   43,    8,   25,    8,   25,   25,    8,   25,   43,    8,   25,
       8,    8,   43,    8,   25,   43,    8,   25,   25,    8,    8,   25,   25,   43,    8,   25,
       8,   25,    8,   25,   25,   43,    8,   25,    8,    8,   25,   25,   25,   43,    8,   25,
       8,    8,    8,   43,   25,   43,    8,   25,   43,   25,   25,   43,   25,   43,    8,   25,
      25,    8,    8,    8,   43,   43,    8,   25,    8,   25,    8,    8,   43,   43,    8,   25,
       8,    8,   25,    8,   43,   43,    8,   25,    8,    8,    8,   25,   43,   43,    8,   25,
       8,    8,    8,    8,    8,    8,   25,   25,   43,    8,    8,    8,    8,    8,   25,   25,
      25,   25,    8,    8,    8,    8,   25,   25,    8,   43,    8,    8,    8,    8,   25,   25,
      25,    8,   25,    8,    8,    8,   25,   25,    8,   25,   25,    8,    8,    8,   25,   25,
      43,   25,   25,    8,    8,    8,   25,   25,   25,   43,   25,    8,    8,    8,   25,   25,
       8,    8,   43,    8,    8,    8,   25,   25,   43,    8,   43,    8,    8,    8,   25,   25,
      25,   25,   43,    8,    8,    8,   25,   25,    8,   43,   43,    8,    8,    8,   25,   25,
      25,    8,    8,   25,    8,    8,   25,   25,    8,   25,    8,   25,    8,    8,   25,   25,
      43,   25,    8,   25,    8,    8,   25,   25,   25,   43,    8,   25,    8,    8,   25,   25,
       8,    8,   25,   25,    8,    8,   25,   25,   43,    8,   25,   25,    8,    8,   25,   25,
      25,   25,   25,   25,    8,    8,   25,   25,    8,   43,   25,   25,    8,    8,   25,   25,
      25,    8,   43,   25,    8,    8,   25,   25,    8,   25,   43,   25,    8,    8,   25,   25,
       8,    8,    8,   43,    8,    8,   25,   25,   43,    8,    8,   43,    8,    8,   25,   25,
      25,   25,    8,   43,    8,    8,   25,   25,    8,   43,    8,   43,    8,    8,   25,   25,
      25,    8,   25,   43,    8,    8,   25,   25,    8,   25,   25,   43,    8,    8,   25,   25,
      25,    8,    8,    8,   25,    8,   25,   25,    8,   25,    8,    8,   25,    8,   25,   25,
      43,   25,    8,    8,   25,    8,   25,   25,   25,   43,    8,    8,   25,    8,   25,   25,
       8,    8,   25,    8,   25,    8,   25,   25,   43,    8,   25,    8,   25,    8,   25,   25,
      25,   25,   25,    8,   25,    8,   25,   25,    8,   43,   25,    8,   25,    8,   25,   25,
      25,    8,   43,    8,   25,    8,   25,   25,    8,   25,   43,    8,   25,    8,   25,   25,
       8,    8,    8,   25,   25,    8,   25,   25,   43,    8,    8,   25,   25,    8,   25,   25,
      25,   25,    8,   25,   25,    8,   25,   25,    8,   43,    8,   25,   25,    8,   25,   25,
      25,    8,   25,   25,   25,    8,   25,   25,    8,   25,   25,   25,   25,    8,   25,   25,
       8,    8,   43,   25,   25,    8,   25,   25,   25,    8,    8,   43,   25,    8,   25,   25,
       8,   25,    8,   43,   25,    8,   25,   25,    8,    8,   25,   43,   25,    8,   25,   25,
       8,    8,    8,    8,   43,    8,   25,   25,   25,   25,    8,    8,   43,    8,   25,   25,
       8,   43,    8,    8,   43,    8,   25,   25,   25,    8,   25,    8,   43,    8,   25,   25,
       8,   25,   25,    8,   43,    8,   25,   25,    8,    8,   43,    8,   43,    8,   25,   25,
      25,    8,    8,   25,   43,    8,   25,   25,    8,   25,    8,   25,   43,    8,   25,   25,
       8,    8,   25,   25,   43,    8,   25,   25,   25,   43,   43,   25,   43,    8,   25,   25,
       8,    8,    8,   43,   43,    8,   25,   25,   25,    8,    8,    8,    8,   25,   25,   25,
       8,   25,    8,    8,    8,   25,   25,   25,   43,   25,    8,    8,    8,   25,   25,   25,
      25,   43,    8,    8,    8,   25,   25,   25,    8,    8,   25,    8,    8,   25,   25,   25,
      43,    8,   25,    8,    8,   25,   25,   25,   25,   25,   25,    8,    8,   25,   25,   25,
       8,   43,   25,    8,    8,   25,   25,   25,   25,    8,   43,    8,    8,   25,   25,   25,
       8,   25,   43,    8,    8,   25,   25,   25,    8,    8,    8,   25,    8,   25,   25,   25,
      43,    8,    8,   25,    8,   25,   25,   25,   25,   25,    8,   25,    8,   25,   25,   25,
       8,   43,    8,   25,    8,   25,   25,   25,   25,    8,   25,   25,    8,   25,   25,   25,
       8,   25,   25,   25,    8,   25,   25,   25,    8,    8,   43,   25,    8,   25,   25,   25,
      25,    8,    8,   43,    8,   25,   25,   25,    8,   25,    8,   43,    8,   25,   25,   25,
       8,    8,   25,   43,    8,   25,   25,   25,    8,    8,    8,    8,   25,   25,   25,   25,
      43,    8,    8,    8,   25,   25,   25,   25,   25,   25,    8,    8,   25,   25,   25,   25,
       8,   43,    8,    8,   25,   25,   25,   25,   25,    8,   25,    8,   25,   25,   25,   25,
       8,   25,   25,    8,   25,   25,   25,   25,    8,    8,   43,    8,   25,   25,   25,   25,
      25,    8,    8,   25,   25,   25,   25,   25,    8,   25,    8,   25,   25,   25,   25,   25,
       8,    8,   25,   25,   25,   25,   25,   25,    8,    8,    8,   43,   25,   25,   25,   25,
      25,    8,    8,    8,   43,   25,   25,   25,    8,   25,    8,    8,   43,   25,   25,   25,
       8,    8,   25,    8,   43,   25,   25,   25,   43,   25,   43,    8,   43,   25,   25,   25,
       8,    8,    8,   25,   43,   25,   25,   25,    8,    8,    8,    8,    8,   43,   25,   25,
      43,    8,    8,    8,    8,   43,   25,   25,   25,   25,    8,    8,    8,   43,   25,   25,
       8,   43,    8,    8,    8,   43,   25,   25,   25,    8,   25,    8,    8,   43,   25,   25,
       8,   25,   25,    8,    8,   43,   25,   25,    8,    8,   43,    8,    8,   43,   25,   25,
      25,    8,    8,   25,    8,   43,   25,   25,    8,   25,    8,   25,    8,   43,   25,   25,
       8,    8,   25,   25,    8,   43,   25,   25,   43,   43,   25,   25,    8,   43,   25,   25,
       8,    8,    8,   43,    8,   43,   25,   25,   25,    8,    8,    8,   25,   43,   25,   25,
       8,   25,    8,    8,   25,   43,   25,   25,    8,    8,   25,    8,   25,   43,   25,   25,
       8,    8,    8,   25,   25,   43,   25,   25,    8,    8,    8,    8,   43,   43,   25,   25,
      25,   43,   25,    8,   43,   43,   25,   25,   25,   25,    8,   43,   43,   43,   25,   25,
       8,   43,   43,   43,   43,   43,   25,   25,   25,    8,    8,    8,    8,    8,   43,   25,
       8,   25,    8,    8,    8,    8,   43,   25,   43,   25,    8,    8,    8,    8,   43,   25,
       8,    8,   25,    8,    8,    8,   43,   25,   43,    8,   25,    8,    8,    8,   43,   25,
      25,   25,   25,    8,    8,    8,   43,   25,    8,   43,   25,    8,    8,    8,   43,   25,
      25,    8,   43,    8,    8,    8,   43,   25,    8,   25,   43,    8,    8,    8,   43,   25,
       8,    8,    8,   25,    8,    8,   43,   25,   25,   25,    8,   25,    8,    8,   43,   25,
       8,   43,    8,   25,    8,    8,   43,   25,   25,    8,   25,   25,    8,    8,   43,   25,
       8,   25,   25,   25,    8,    8,   43,   25,    8,    8,   43,   25,    8,    8,   43,   25,
       8,   25,    8,   43,    8,    8,   43,   25,    8,    8,   25,   43,    8,    8,   43,   25,
       8,    8,    8,    8,   25,    8,   43,   25,   43,    8,    8,    8,   25,    8,   43,   25,
      25,   25,    8,    8,   25,    8,   43,   25,    8,   43,    8,    8,   25,    8,   43,   25,
      25,    8,   25,    8,   25,    8,   43,   25,    8,   25,   25,    8,   25,    8,   43,   25,
       8,    8,   43,    8,   25,    8,   43,   25,   25,    8,    8,   25,   25,    8,   43,   25,
       8,   25,    8,   25,   25,    8,   43,   25,    8,    8,   25,   25,   25,    8,   43,   25,
       8,    8,    8,   43,   25,    8,   43,   25,   25,   43,   25,   43,   25,    8,   43,   25,
       8,   25,    8,    8,   43,    8,   43,   25,    8,    8,   25,    8,   43,    8,   43,   25,
       8,    8,    8,   25,   43,    8,   43,   25,   43,   25,   25,   25,   43,    8,   43,   25,
      25,    8,   43,   43,   43,    8,   43,   25,    8,    8,    8,    8,    8,   25,   43,   25,
      25,   25,    8,    8,    8,   25,   43,   25,    8,   43,    8,    8,    8,   25,   43,   25,
      25,    8,   25,    8,    8,   25,   43,   25,    8,   25,   25,    8,    8,   25,   43,   25,
       8,    8,   43,    8,    8,   25,   43,   25,   25,    8,    8,   25,    8,   25,   43,   25,
       8,   25,    8,   25,    8,   25,   43,   25,    8,    8,   25,   25,    8,   25,   43,   25,
       8,    8,    8,   43,    8,   25,   43,   25,   25,    8,    8,    8,   25,   25,   43,   25,
       8,   25,    8,    8,   25,   25,   43,   25,    8,    8,   25,    8,   25,   25,   43,   25,
       8,    8,    8,   25,   25,   25,   43,   25,   43,   43,    8,   25,   25,   25,   43,   25,
       8,   43,   43,   25,   25,   25,   43,   25,   43,    8,   25,   43,   25,   25,   43,   25,
       8,    8,    8,    8,   43,   25,   43,   25,    8,   25,   25,   43,   43,   25,   43,   25,
      25,    8,    8,    8,    8,   43,   43,   25,    8,   25,    8,    8,    8,   43,   43,   25,
       8,    8,   25,    8,    8,   43,   43,   25,   25,   25,   43,   25,    8,   43,   43,   25,
       8,   43,   25,   43,    8,   43,   43,   25,    8,    8,    8,    8,   25,   43,   43,   25,
      43,   43,   43,    8,   25,   43,   43,   25,   43,    8,    8,   25,   43,   43,   43,   25,
      25,    8,   43,   43,   43,   43,   43,   25,    8,    8,    8,    8,    8,    8,    8,   43,
      43,    8,    8,    8,    8,    8,    8,   43,   25,   25,    8,    8,    8,    8,    8,   43,
       8,   43,    8,    8,    8,    8,    8,   43,   25,    8,   25,    8,    8,    8,    8,   43,
       8,   25,   25,    8,    8,    8,    8,   43,   25,   43,   25,    8,    8,    8,    8,   43,
       8,    8,   43,    8,    8,    8,    8,   43,   25,   25,   43,    8,    8,    8,    8,   43,
      25,    8,    8,   25,    8,    8,    8,   43,    8,   25,    8,   25,    8,    8,    8,   43,
       8,    8,   25,   25,    8,    8,    8,   43,   43,    8,   25,   25,    8,    8,    8,   43,
      25,   25,   25,   25,    8,    8,    8,   43,    8,   43,   25,   25,    8,    8,    8,   43,
      25,    8,   43,   25,    8,    8,    8,   43,    8,    8,    8,   43,    8,    8,    8,   43,
      25,   25,    8,   43,    8,    8,    8,   43,   25,    8,   25,   43,    8,    8,    8,   43,
       8,   25,   25,   43,    8,    8,    8,   43,   25,    8,    8,    8,   25,    8,    8,   43,
       8,   25,    8,    8,   25,    8,    8,   43,   25,   43,    8,    8,   25,    8,    8,   43,
       8,    8,   25,    8,   25,    8,    8,   43,   43,    8,   25,    8,   25,    8,    8,   43,
      25,   25,   25,    8,   25,    8,    8,   43,    8,   43,   25,    8,   25,    8,    8,   43,
      25,    8,   43,    8,   25,    8,    8,   43,    8,   25,   43,    8,   25,    8,    8,   43,
       8,    8,    8,   25,   25,    8,    8,   43,   43,    8,    8,   25,   25,    8,    8,   43,
      25,   25,    8,   25,   25,    8,    8,   43,    8,   43,    8,   25,   25,    8,    8,   43,
      25,    8,   25,   25,   25,    8,    8,   43,    8,   25,   25,   25,   25,    8,    8,   43,
      25,    8,    8,   43,   25,    8,    8,   43,    8,   25,    8,   43,   25,    8,    8,   43,
       8,    8,   25,   43,   25,    8,    8,   43,   25,   43,   43,   43,   25,    8,    8,   43,
       8,    8,    8,    8,   43,    8,    8,   43,   25,   25,    8,    8,   43,    8,    8,   43,
      43,   43,    8,    8,   43,    8,    8,   43,   25,    8,   25,    8,   43,    8,    8,   43,
       8,   25,   25,    8,   43,    8,    8,   43,   25,    8,    8,   25,   43,    8,    8,   43,
       8,   25,    8,   25,   43,    8,    8,   43,    8,    8,   25,   25,   43,    8,    8,   43,
      25,    8,    8,    8,    8,   25,    8,   43,    8,   25,    8,    8,    8,   25,    8,   43,
      43,   25,    8,    8,    8,   25,    8,   43,   25,   43,    8,    8,    8,   25,    8,   43,
       8,    8,   25,    8,    8,   25,    8,   43,   43,    8,   25,    8,    8,   25,    8,   43,
      25,   25,   25,    8,    8,   25,    8,   43,    8,   43,   25,    8,    8,   25,    8,   43,
      25,    8,   43,    8,    8,   25,    8,   43,    8,    8,    8,   25,    8,   25,    8,   43,
      43,    8,    8,   25,    8,   25,    8,   43,   25,   25,    8,   25,    8,   25,    8,   43,
       8,   43,    8,   25,    8,   25,    8,   43,   25,    8,   25,   25,    8,   25,    8,   43,
       8,   25,   25,   25,    8,   25,    8,   43,    8,    8,   43,   25,    8,   25,    8,   43,
      25,    8,    8,   43,    8,   25,    8,   43,    8,   25,    8,   43,    8,   25,    8,   43,
       8,    8,   25,   43,    8,   25,    8,   43,    8,    8,    8,    8,   25,   25,    8,   43,
      43,    8,    8,    8,   25,   25,    8,   43,   25,   25,    8,    8,   25,   25,    8,   43,
       8,   43,    8,    8,   25,   25,    8,   43,   25,    8,   25,    8,   25,   25,    8,   43,
       8,   25,   25,    8,   25,   25,    8,   43,    8,    8,   43,    8,   25,   25,    8,   43,
      25,    8,    8,   25,   25,   25,    8,   43,    8,   25,    8,   25,   25,   25,    8,   43,
       8,    8,   25,   25,   25,   25,    8,   43,    8,    8,    8,   43,   25,   25,    8,   43,
      43,   43,    8,   43,   25,   25,    8,   43,   25,    8,    8,    8,   43,   25,    8,   43,
       8,   25,    8,    8,   43,   25,    8,   43,    8,    8,   25,    8,   43,   25,    8,   43,
      25,   43,   43,    8,   43,   25,    8,   43,    8,    8,    8,   25,   43,   25,    8,   43,
       8,    8,    8,    8,    8,   43,    8,   43,   25,   25,    8,    8,    8,   43,    8,   43,
      25,    8,   25,    8,    8,   43,    8,   43,    8,   25,   25,    8,    8,   43,    8,   43,
      25,    8,    8,   25,    8,   43,    8,   43,    8,   25,    8,   25,    8,   43,    8,   43,
       8,    8,   25,   25,    8,   43,    8,   43,   43,    8,   43,   43,    8,   43,    8,   43,
      25,    8,    8,    8,   25,   43,    8,   43,    8,   25,    8,    8,   25,   43,    8,   43,
       8,    8,    8,   25,   25,   43,    8,   43,   25,   25,   43,   25,   25,   43,    8,   43,
      43,    8,   43,    8,   43,   43,    8,   43,    8,   43,   25,   25,   43,   43,    8,   43,
      43,   43,   25,   25,   43,   43,    8,   43,   43,    8,    8,   43,   43,   43,    8,   43,
      43,    8,   43,   43,   43,   43,    8,   43,   25,    8,    8,    8,    8,    8,   25,   43,
       8,   25,    8,    8,    8,    8,   25,   43,   25,   43,    8,    8,    8,    8,   25,   43,
       8,    8,   25,    8,    8,    8,   25,   43,   43,    8,   25,    8,    8,    8,   25,   43,
      25,   25,   25,    8,    8,    8,   25,   43,    8,   43,   25,    8,    8,    8,   25,   43,
       8,   25,   43,    8,    8,    8,   25,   43,    8,    8,    8,   25,    8,    8,   25,   43,
      43,    8,    8,   25,    8,    8,   25,   43,   25,   25,    8,   25,    8,    8,   25,   43,
       8,   43,    8,   25,    8,    8,   25,   43,   25,    8,   25,   25,    8,    8,   25,   43,
       8,   25,   25,   25,    8,    8,   25,   43,    8,    8,   43,   25,    8,    8,   25,   43,
      25,    8,    8,   43,    8,    8,   25,   43,    8,   25,    8,   43,    8,    8,   25,   43,
       8,    8,   25,   43,    8,    8,   25,   43,    8,    8,    8,    8,   25,    8,   25,   43,
      25,   25,    8,    8,   25,    8,   25,   43,   25,    8,   25,    8,   25,    8,   25,   43,
       8,   25,   25,    8,   25,    8,   25,   43,   25,    8,    8,   25,   25,    8,   25,   43,
       8,   25,    8,   25,   25,    8,   25,   43,    8,    8,   25,   25,   25,    8,   25,   43,
      43,   43,   25,   25,   25,    8,   25,   43,   25,    8,    8,    8,   43,    8,   25,   43,
       8,   25,    8,    8,   43,    8,   25,   43,    8,    8,   25,    8,   43,    8,   25,   43,
       8,    8,    8,   25,   43,    8,   25,   43,   43,   25,   43,   43,   43,    8,   25,   43,
       8,    8,    8,    8,    8,   25,   25,   43,   43,    8,    8,    8,    8,   25,   25,   43,
      25,   25,    8,    8,    8,   25,   25,   43,    8,   43,    8,    8,    8,   25,   25,   43,
      25,    8,   25,    8,    8,   25,   25,   43,    8,   25,   25,    8,    8,   25,   25,   43,
       8,    8,   43,    8,    8,   25,   25,   43,   25,    8,    8,   25,    8,   25,   25,   43,
       8,   25,    8,   25,    8,   25,   25,   43,    8,    8,   25,   25,    8,   25,   25,   43,
       8,    8,    8,   43,    8,   25,   25,   43,   43,   25,   25,   43,    8,   25,   25,   43,
      25,    8,    8,    8,   25,   25,   25,   43,    8,   25,    8,    8,   25,   25,   25,   43,
       8,    8,   25,    8,   25,   25,   25,   43,    8,    8,    8,   25,   25,   25,   25,   43,
       8,   43,   25,   43,   25,   25,   25,   43,   25,    8,   43,   43,   25,   25,   25,   43,
       8,    8,    8,    8,   43,   25,   25,   43,   43,   25,    8,   25,   43,   25,   25,   43,
       8,   25,   43,   25,   43,   25,   25,   43,   25,    8,    8,    8,    8,   43,   25,   43,
       8,   25,    8,    8,    8,   43,   25,   43,    8,    8,   25,    8,    8,   43,   25,   43,
      43,   25,   43,    8,    8,   43,   25,   43,    8,    8,    8,   25,    8,   43,   25,   43,
      25,   43,   43,   43,    8,   43,   25,   43,    8,    8,    8,    8,   25,   43,   25,   43,
      25,   43,    8,   25,   25,   43,   25,   43,   43,    8,   25,   25,   25,   43,   25,   43,
       8,    8,   25,   43,   43,   43,   25,   43,    8,    8,    8,    8,    8,    8,   43,   43,
      25,   25,    8,    8,    8,    8,   43,   43,   43,   43,    8,    8,    8,    8,   43,   43,
       8,   25,   25,    8,    8,    8,   43,   43,   43,    8,   43,    8,    8,    8,   43,   43,
      43,   43,   43,    8,    8,    8,   43,   43,   25,    8,    8,   25,    8,    8,   43,   43,
       8,   25,    8,   25,    8,    8,   43,   43,    8,    8,   25,   25,    8,    8,   43,   43,
      43,    8,   43,   43,    8,    8,   43,   43,   43,   43,   43,   43,    8,    8,   43,   43,
       8,    8,    8,   25,   25,    8,   43,   43,   25,   25,   43,   25,   25,    8,   43,   43,
      43,    8,    8,    8,   43,    8,   43,   43,   43,   43,    8,    8,   43,    8,   43,   43,
      43,    8,   43,    8,   43,    8,   43,   43,    8,   43,   43,    8,   43,    8,   43,   43,
      43,   43,   43,    8,   43,    8,   43,   43,   43,    8,    8,   43,   43,    8,   43,   43,
       8,   43,    8,   43,   43,    8,   43,   43,   43,   43,    8,   43,   43,    8,   43,   43,
       8,   43,   43,   43,   43,    8,   43,   43,   25,    8,    8,    8,    8,   25,   43,   43,
       8,   25,    8,    8,    8,   25,   43,   43,    8,    8,   25,    8,    8,   25,   43,   43,
       8,    8,    8,   25,    8,   25,   43,   43,   25,   43,    8,   43,    8,   25,   43,   43,
       8,   25,   43,   43,    8,   25,   43,   43,    8,    8,    8,    8,   25,   25,   43,   43,
      25,   43,   25,    8,   25,   25,   43,   43,   25,    8,   25,   25,   43,   25,   43,   43,
      43,   43,    8,    8,    8,   43,   43,   43,    8,   43,   43,    8,    8,   43,   43,   43,
      43,    8,   43,   43,    8,   43,   43,   43,    8,   25,   25,   25,   25,   43,   43,   43,
      43,   25,    8,   43,   25,   43,   43,   43,    8,   43,    8,    8,   43,   43,   43,   43,
      43,   43,    8,    8,   43,   43,   43,   43,    8,    8,   43,    8,   43,   43,   43,   43,
      43,    8,   43,    8,   43,   43,   43,   43,    8,   43,   43,    8,   43,   43,   43,   43,
       8,   43,    8,   43,   43,   43,   43,   43,   43,   43,   43,   43,   43,   43,   43,   43,
};

#define INGOT_IQ2S_GRID_ENTRIES 1024
#define INGOT_IQ2S_GRID_WIDTH   8

/* IQ3_XXS: 256 entries x 4 values, map (4, 12, 20, 28, 36, 44, 52, 62) */
static const int8_t ingot_iq3xxs_grid[1024] = {
       4,    4,    4,    4,   20,    4,    4,    4,   36,    4,    4,    4,   12,   12,    4,    4,
      28,   12,    4,    4,   62,   12,    4,    4,    4,   20,    4,    4,   20,   20,    4,    4,
      12,   28,    4,    4,   20,   36,    4,    4,   28,   62,    4,    4,   44,   62,    4,    4,
      12,    4,   12,    4,   28,    4,   12,    4,    4,   12,   12,    4,   20,   12,   12,    4,
      12,   20,   12,    4,   44,   20,   12,    4,    4,   28,   12,    4,   20,   28,   12,    4,
      12,   36,   12,    4,   36,   44,   12,    4,    4,   62,   12,    4,    4,    4,   20,    4,
      20,    4,   20,    4,   36,    4,   20,    4,   12,   12,   20,    4,    4,   20,   20,    4,
      20,   20,   20,    4,   12,   28,   20,    4,   28,   28,   20,    4,   62,   28,   20,    4,
      12,   44,   20,    4,   62,   44,   20,    4,   44,   62,   20,    4,   12,    4,   28,    4,
      62,    4,   28,    4,    4,   12,   28,    4,   20,   12,   28,    4,   44,   20,   28,    4,
       4,   62,   28,    4,   28,   12,   36,    4,   62,   28,   36,    4,   36,   36,   36,    4,
      62,   44,   36,    4,   28,   62,   36,    4,   44,   62,   36,    4,   12,    4,   44,    4,
      62,    4,   44,    4,   20,   28,   44,    4,   20,   44,   44,    4,   44,   28,   52,    4,
      36,   52,   52,    4,    4,   12,   62,    4,   36,   12,   62,    4,   52,   12,   62,    4,
      28,   36,   62,    4,   12,   52,   62,    4,   12,    4,    4,   12,   28,    4,    4,   12,
       4,   12,    4,   12,   20,   12,    4,   12,   12,   20,    4,   12,   28,   20,    4,   12,
       4,   28,    4,   12,   20,   28,    4,   12,   36,   28,    4,   12,   62,   36,    4,   12,
       4,   44,    4,   12,    4,    4,   12,   12,   20,    4,   12,   12,   12,   12,   12,   12,
       4,   20,   12,   12,   20,   20,   12,   12,   12,    4,   20,   12,   28,    4,   20,   12,
       4,   12,   20,   12,   20,   12,   20,   12,   12,   20,   20,   12,    4,   28,   20,   12,
      20,   62,   20,   12,    4,    4,   28,   12,   20,    4,   28,   12,    4,   20,   28,   12,
      12,   28,   28,   12,   52,   36,   28,   12,   52,   52,   28,   12,   12,    4,   36,   12,
      44,    4,   36,   12,    4,   44,   36,   12,    4,   20,   44,   12,   36,   20,   44,   12,
      52,   36,   44,   12,   12,   62,   44,   12,   44,    4,   52,   12,   20,   20,   62,   12,
       4,   36,   62,   12,    4,    4,    4,   20,   20,    4,    4,   20,   12,   12,    4,   20,
      28,   12,    4,   20,    4,   20,    4,   20,   20,   20,    4,   20,   52,   20,    4,   20,
      12,   28,    4,   20,   20,   36,    4,   20,   12,    4,   12,   20,   28,    4,   12,   20,
      44,    4,   12,   20,    4,   12,   12,   20,   20,   12,   12,   20,   12,   20,   12,   20,
       4,   28,   12,   20,   28,   52,   12,   20,   62,   52,   12,   20,    4,   62,   12,   20,
       4,    4,   20,   20,   20,    4,   20,   20,   12,   12,   20,   20,   62,   12,   20,   20,
       4,   20,   20,   20,   20,   20,   20,   20,   62,   28,   20,   20,    4,   36,   20,   20,
      44,   44,   20,   20,   12,    4,   28,   20,    4,   12,   28,   20,   36,   12,   28,   20,
       4,   62,   28,   20,   36,   62,   28,   20,   44,   28,   36,   20,   28,   44,   36,   20,
      28,    4,   44,   20,   62,   20,   44,   20,   12,   36,   44,   20,   36,   62,   44,   20,
      12,    4,   62,   20,   28,    4,   62,   20,   52,   12,   62,   20,   44,   36,   62,   20,
      12,    4,    4,   28,    4,   12,    4,   28,   20,   12,    4,   28,   12,   20,    4,   28,
      28,   20,    4,   28,    4,   44,    4,   28,   44,   52,    4,   28,   20,   62,    4,   28,
       4,    4,   12,   28,   20,    4,   12,   28,    4,   20,   12,   28,   12,   28,   12,   28,
      36,   36,   12,   28,   52,   36,   12,   28,   12,    4,   20,   28,   28,    4,   20,   28,
       4,   12,   20,   28,   44,   20,   20,   28,   20,   44,   20,   28,   20,   62,   20,   28,
      12,   12,   28,   28,   28,   28,   28,   28,    4,   28,   36,   28,   62,   36,   36,   28,
      20,   62,   36,   28,    4,    4,   44,   28,   52,    4,   44,   28,   20,   20,   44,   28,
      44,   44,   44,   28,   36,   12,   52,   28,   52,   28,   52,   28,   28,   52,   52,   28,
      28,   28,   62,   28,    4,   52,   62,   28,   36,    4,    4,   36,   62,   12,    4,   36,
      44,   28,    4,   36,   62,   28,    4,   36,   28,   44,    4,   36,   62,   44,    4,   36,
      36,   62,   12,   36,    4,   20,   20,   36,   62,   28,   20,   36,    4,   36,   20,   36,
       4,   52,   20,   36,   52,   52,   20,   36,   62,    4,   28,   36,   44,   36,   28,   36,
      36,    4,   36,   36,   12,   44,   36,   36,   36,   52,   36,   36,   44,   20,   44,   36,
      28,   36,   44,   36,    4,   62,   44,   36,   44,    4,   62,   36,    4,   12,   62,   36,
      20,   12,   62,   36,    4,   28,   62,   36,   20,   12,    4,   44,   12,   36,    4,   44,
       4,   62,    4,   44,    4,    4,   12,   44,   52,    4,   12,   44,   52,   20,   12,   44,
      44,   44,   12,   44,   36,   12,   20,   44,   20,   28,   20,   44,   20,   62,   20,   44,
      20,    4,   28,   44,   28,   44,   28,   44,    4,   12,   36,   44,   28,   20,   36,   44,
      62,   20,   36,   44,   20,   62,   36,   44,   20,    4,   44,   44,   12,   28,   44,   44,
       4,   44,   52,   44,   36,   20,   62,   44,   20,   36,   62,   44,   36,   20,    4,   52,
      36,   36,    4,   52,   52,   36,    4,   52,   36,   52,    4,   52,   12,   20,   12,   52,
      12,   52,   12,   52,   62,   12,   20,   52,   36,   52,   20,   52,    4,   28,   28,   52,
      52,   28,   28,   52,   36,   36,   36,   52,   44,    4,   44,   52,   20,   44,   44,   52,
      28,   28,   52,   52,   28,    4,   62,   52,   12,   20,   62,   52,   28,    4,    4,   62,
      44,    4,    4,   62,   62,    4,    4,   62,    4,   12,    4,   62,   20,   28,    4,   62,
      20,   44,    4,   62,   52,   20,   12,   62,    4,   36,   12,   62,   20,   12,   20,   62,
      44,   36,   20,   62,   20,   44,   20,   62,    4,    4,   28,   62,   44,   12,   28,   62,
      28,   28,   28,   62,    4,   52,   28,   62,   12,   20,   36,   62,   12,   36,   36,   62,
       4,    4,   44,   62,   20,    4,   44,   62,   36,   20,   44,   62,    4,   28,   52,   62,
};

#define INGOT_IQ3XXS_GRID_ENTRIES 256
#define INGOT_IQ3XXS_GRID_WIDTH   4

/* IQ3_S: 512 entries x 4 values, map (1, 3, 5, 7, 9, 11, 13, 15) */
static const int8_t ingot_iq3s_grid[2048] = {
       1,    1,    1,    1,    3,    1,    1,    1,    5,    1,    1,    1,   11,    1,    1,    1,
      15,    1,    1,    1,    1,    3,    1,    1,    3,    3,    1,    1,    5,    3,    1,    1,
       9,    3,    1,    1,   13,    3,    1,    1,    1,    5,    1,    1,    3,    5,    1,    1,
      11,    5,    1,    1,    7,    7,    1,    1,    1,    9,    1,    1,    5,    9,    1,    1,
      11,    9,    1,    1,   15,    9,    1,    1,    3,   11,    1,    1,    7,   11,    1,    1,
       1,   13,    1,    1,    5,   13,    1,    1,    3,   15,    1,    1,    9,   15,    1,    1,
      15,   15,    1,    1,    1,    1,    3,    1,    3,    1,    3,    1,    5,    1,    3,    1,
       9,    1,    3,    1,    1,    3,    3,    1,    3,    3,    3,    1,   11,    3,    3,    1,
       1,    5,    3,    1,    7,    5,    3,    1,   15,    5,    3,    1,    3,    7,    3,    1,
      11,    7,    3,    1,    9,    9,    3,    1,    3,   13,    3,    1,   11,   13,    3,    1,
       5,   15,    3,    1,    1,    1,    5,    1,    3,    1,    5,    1,   11,    1,    5,    1,
      15,    1,    5,    1,    1,    3,    5,    1,    7,    3,    5,    1,   13,    3,    5,    1,
       3,    5,    5,    1,   11,    5,    5,    1,    1,    7,    5,    1,    9,    7,    5,    1,
       5,    9,    5,    1,   11,    9,    5,    1,   15,    9,    5,    1,    3,   11,    5,    1,
       7,   11,    5,    1,    1,   15,    5,    1,    7,   15,    5,    1,    7,    1,    7,    1,
       3,    3,    7,    1,   11,    3,    7,    1,    1,    5,    7,    1,    5,    5,    7,    1,
       3,    7,    7,    1,    7,    7,    7,    1,   13,    7,    7,    1,    9,    9,    7,    1,
       1,   11,    7,    1,    5,   11,    7,    1,   15,   13,    7,    1,    3,   15,    7,    1,
      11,   15,    7,    1,    1,    1,    9,    1,    7,    3,    9,    1,   15,    3,    9,    1,
       3,    5,    9,    1,    9,    5,    9,    1,    5,    7,    9,    1,    1,    9,    9,    1,
       7,    9,    9,    1,    3,   11,    9,    1,    1,   15,    9,    1,    5,    1,   11,    1,
       9,    1,   11,    1,    1,    5,   11,    1,    5,    5,   11,    1,   13,    5,   11,    1,
       7,    7,   11,    1,    3,    9,   11,    1,   11,    9,   11,    1,   15,    9,   11,    1,
      13,   13,   11,    1,    7,   15,   11,    1,   13,    1,   13,    1,    3,    3,   13,    1,
       7,    3,   13,    1,    3,    7,   13,    1,    5,   11,   13,    1,    3,   15,   13,    1,
       1,    1,   15,    1,    5,    1,   15,    1,    9,    1,   15,    1,    1,    5,   15,    1,
       5,    5,   15,    1,   13,    5,   15,    1,    7,    7,   15,    1,    1,   11,   15,    1,
       9,   11,   15,    1,    1,    1,    1,    3,    3,    1,    1,    3,    5,    1,    1,    3,
       9,    1,    1,    3,    1,    3,    1,    3,    3,    3,    1,    3,    7,    3,    1,    3,
      11,    3,    1,    3,   15,    3,    1,    3,    1,    5,    1,    3,    5,    5,    1,    3,
       3,    7,    1,    3,    9,    7,    1,    3,   13,    7,    1,    3,    9,   11,    1,    3,
      13,   11,    1,    3,    3,   13,    1,    3,    5,   15,    1,    3,    1,    1,    3,    3,
       3,    1,    3,    3,    7,    1,    3,    3,   13,    1,    3,    3,    1,    3,    3,    3,
       9,    3,    3,    3,    3,    5,    3,    3,    1,    7,    3,    3,    7,    7,    3,    3,
       3,    9,    3,    3,    1,   11,    3,    3,    5,   11,    3,    3,    1,   15,    3,    3,
      13,   15,    3,    3,    1,    1,    5,    3,    5,    3,    5,    3,   11,    3,    5,    3,
      15,    3,    5,    3,    1,    5,    5,    3,    9,    5,    5,    3,    5,    7,    5,    3,
       1,    9,    5,    3,    7,    9,    5,    3,   11,   11,    5,    3,    1,   13,    5,    3,
       5,   15,    5,    3,    3,    1,    7,    3,    9,    1,    7,    3,   15,    1,    7,    3,
       1,    3,    7,    3,    7,    3,    7,    3,    3,    5,    7,    3,   15,    5,    7,    3,
       1,    7,    7,    3,    9,    7,    7,    3,    3,    9,    7,    3,    5,   13,    7,    3,
       1,   15,    7,    3,    7,    1,    9,    3,   11,    1,    9,    3,    5,    3,    9,    3,
       9,    3,    9,    3,    3,    7,    9,    3,    7,    7,    9,    3,    5,    9,    9,    3,
      13,    9,    9,    3,    1,   11,    9,    3,    9,   11,    9,    3,    3,    1,   11,    3,
       1,    3,   11,    3,    7,    3,   11,    3,    3,    5,   11,    3,    1,    7,   11,    3,
       5,    7,   11,    3,    3,   11,   11,    3,    1,    5,   13,    3,    9,    5,   13,    3,
      15,    5,   13,    3,    9,    9,   13,    3,   13,    9,   13,    3,    3,    1,   15,    3,
       7,    1,   15,    3,    1,    3,   15,    3,    5,    3,   15,    3,    3,    5,   15,    3,
      11,    7,   15,    3,    3,    9,   15,    3,    5,   13,   15,    3,    1,   15,   15,    3,
       1,    1,    1,    5,    3,    1,    1,    5,    7,    1,    1,    5,   11,    1,    1,    5,
      15,    1,    1,    5,    1,    3,    1,    5,    5,    3,    1,    5,    9,    3,    1,    5,
      13,    3,    1,    5,    3,    5,    1,    5,    7,    5,    1,    5,   15,    5,    1,    5,
       1,    7,    1,    5,    5,    7,    1,    5,    3,    9,    1,    5,    7,    9,    1,    5,
      11,    9,    1,    5,    1,   11,    1,    5,    5,   11,    1,    5,   15,   13,    1,    5,
       1,   15,    1,    5,    7,   15,    1,    5,   11,   15,    1,    5,    1,    1,    3,    5,
       5,    1,    3,    5,    1,    3,    3,    5,    7,    3,    3,    5,   15,    3,    3,    5,
       5,    5,    3,    5,   11,    5,    3,    5,    3,    7,    3,    5,    9,    7,    3,    5,
       5,    9,    3,    5,    3,   11,    3,    5,    3,    1,    5,    5,    9,    1,    5,    5,
      15,    1,    5,    5,    3,    5,    5,    5,    7,    5,    5,    5,    1,    7,    5,    5,
      15,    7,    5,    5,    3,    9,    5,    5,    7,   11,    5,    5,   15,   11,    5,    5,
       3,   15,    5,    5,    9,   15,    5,    5,    1,    1,    7,    5,    5,    1,    7,    5,
      11,    1,    7,    5,    3,    3,    7,    5,    5,    5,    7,    5,    9,    5,    7,    5,
       3,    7,    7,    5,    7,    7,    7,    5,    5,    9,    7,    5,    1,   11,    7,    5,
      13,   13,    7,    5,    3,    1,    9,    5,   15,    1,    9,    5,    1,    5,    9,    5,
       7,    5,    9,    5,    5,    7,    9,    5,   11,    7,    9,    5,    3,    9,    9,    5,
       5,   15,    9,    5,   11,   15,    9,    5,    9,    1,   11,    5,    3,    3,   11,    5,
       5,    5,   11,    5,   15,    7,   11,    5,    1,    9,   11,    5,    7,   11,   11,    5,
       1,   15,   11,    5,    1,    1,   13,    5,    5,    1,   13,    5,   15,    1,   13,    5,
       3,    5,   13,    5,   11,   11,   13,    5,    3,   13,   13,    5,   11,    1,   15,    5,
       3,    3,   15,    5,   13,    5,   15,    5,    1,    7,   15,    5,    7,    9,   15,    5,
       1,   11,   15,    5,    5,    1,    1,    7,    3,    3,    1,    7,    7,    3,    1,    7,
      11,    3,    1,    7,   15,    3,    1,    7,    5,    5,    1,    7,    3,    7,    1,    7,
       7,    7,    1,    7,   11,    7,    1,    7,    5,    9,    1,    7,    9,    9,    1,    7,
      15,    9,    1,    7,    3,   11,    1,    7,    7,   13,    1,    7,    3,   15,    1,    7,
       3,    1,    3,    7,    7,    1,    3,    7,   11,    1,    3,    7,    9,    3,    3,    7,
       3,    5,    3,    7,    7,    5,    3,    7,    1,    9,    3,    7,    1,   13,    3,    7,
       5,   15,    3,    7,   13,   15,    3,    7,    1,    1,    5,    7,    5,    3,    5,    7,
       1,    5,    5,    7,    5,    7,    5,    7,    9,    7,    5,    7,    1,   11,    5,    7,
       3,    1,    7,    7,    1,    3,    7,    7,    9,    3,    7,    7,    3,    5,    7,    7,
       7,    5,    7,    7,   15,    5,    7,    7,    1,    7,    7,    7,    3,    9,    7,    7,
       7,    9,    7,    7,   15,    9,    7,    7,   11,   11,    7,    7,    7,   15,    7,    7,
       7,    1,    9,    7,    3,    3,    9,    7,   13,    3,    9,    7,    5,    5,    9,    7,
       3,    7,    9,    7,    5,   11,    9,    7,    1,   13,    9,    7,    9,   13,    9,    7,
       3,    1,   11,    7,    1,    3,   11,    7,    5,    3,   11,    7,   11,    5,   11,    7,
       5,    7,   11,    7,    9,    9,   11,    7,   13,   11,   11,    7,    7,   15,   11,    7,
      13,    3,   13,    7,    3,    9,   13,    7,    3,    1,   15,    7,    7,    1,   15,    7,
       1,    5,   15,    7,    5,    5,   15,    7,   11,    7,   15,    7,    1,    1,    1,    9,
       9,    1,    1,    9,    5,    3,    1,    9,    1,    5,    1,    9,    9,    5,    1,    9,
      15,    5,    1,    9,    5,    7,    1,    9,    3,    9,    1,    9,    1,   11,    1,    9,
       1,   15,    1,    9,    5,    1,    3,    9,   15,    1,    3,    9,    3,    3,    3,    9,
       7,    3,    3,    9,    5,    5,    3,    9,    1,    7,    3,    9,   11,    7,    3,    9,
       7,    9,    3,    9,    3,   11,    3,    9,   11,   11,    3,    9,    3,    1,    5,    9,
       7,    1,    5,    9,    1,    3,    5,    9,   11,    3,    5,    9,    3,    5,    5,    9,
       7,    7,    5,    9,    1,    9,    5,    9,   15,   11,    5,    9,    5,   13,    5,    9,
       1,   15,    5,    9,    9,    1,    7,    9,    3,    3,    7,    9,    7,    3,    7,    9,
       1,    5,    7,    9,    5,    5,    7,    9,    3,    7,    7,    9,   11,    7,    7,    9,
       1,    1,    9,    9,    5,    1,    9,    9,    9,    5,    9,    9,   15,    7,    9,    9,
       1,    9,    9,    9,    3,   15,    9,    9,   11,    1,   11,    9,   15,    1,   11,    9,
       3,    5,   11,    9,    5,   13,   11,    9,    7,    3,   13,    9,    9,    7,   13,    9,
       1,   13,   13,    9,    1,    3,   15,    9,   11,    3,   15,    9,    1,    7,   15,    9,
       7,    9,   15,    9,    3,   11,   15,    9,    5,    1,    1,   11,    1,    3,    1,   11,
       9,    3,    1,   11,    5,    5,    1,   11,    1,    9,    1,   11,    9,    9,    1,   11,
      15,    9,    1,   11,    5,   11,    1,   11,   13,   13,    1,   11,    9,   15,    1,   11,
       3,    1,    3,   11,    7,    1,    3,   11,   11,    1,    3,   11,    5,    3,    3,   11,
       3,    5,    3,   11,    5,    7,    3,   11,    5,   15,    3,   11,    1,    1,    5,   11,
       3,    3,    5,   11,    7,    5,    5,   11,    1,    7,    5,   11,   13,    7,    5,   11,
       7,   11,    5,   11,    5,    1,    7,   11,   15,    1,    7,   11,    1,    3,    7,   11,
      15,    5,    7,   11,    9,    9,    7,   11,    3,   11,    7,   11,   11,   13,    7,   11,
       7,   15,    7,   11,    3,    1,    9,   11,    9,    1,    9,   11,    1,    5,    9,   11,
       5,    7,    9,   11,   13,    9,    9,   11,    5,    3,   11,   11,   13,    5,   11,   11,
       3,   11,   11,   11,    7,   11,   11,   11,    5,    9,   13,   11,    5,    1,   15,   11,
       9,    1,   15,   11,    5,    5,   15,   11,    3,    3,    1,   13,    7,    3,    1,   13,
      11,    3,    1,   13,    3,    7,    1,   13,    7,    7,    1,   13,    1,   13,    1,   13,
       1,    1,    3,   13,    1,    5,    3,   13,   15,    5,    3,   13,    9,   13,    3,   13,
       5,    3,    5,   13,    9,    7,    5,   13,    5,    9,    5,   13,   11,   11,    5,   13,
       5,   13,    5,   13,    1,   15,    5,   13,    1,    1,    7,   13,    9,    3,    7,   13,
       3,    5,    7,   13,    1,    9,    7,   13,   11,    5,    9,   13,    7,    9,    9,   13,
       5,   13,    9,   13,    1,    1,   11,   13,    7,    1,   11,   13,    9,    7,   11,   13,
       1,   13,   11,   13,   11,    1,   13,   13,    1,    9,   13,   13,    3,    3,   15,   13,
       7,    3,   15,   13,    1,    1,    1,   15,    9,    1,    1,   15,   15,    1,    1,   15,
       1,    5,    1,   15,    5,    5,    1,   15,   13,    7,    1,   15,    1,    9,    1,   15,
       9,   11,    1,   15,    5,   13,    1,   15,    5,    1,    3,   15,    3,    3,    3,   15,
       9,    5,    3,   15,    7,    9,    3,   15,   11,    9,    3,   15,    3,    1,    5,   15,
       9,    1,    5,   15,    1,    3,    5,   15,   13,    3,    5,   15,    3,    5,    5,   15,
       1,    7,    5,   15,    3,   11,    5,   15,    5,    1,    7,   15,    5,    7,    7,   15,
      11,    7,    7,   15,    7,   11,    7,   15,    3,    1,    9,   15,   11,    1,    9,   15,
       7,    3,    9,   15,    1,    5,    9,   15,    1,   11,    9,   15,    5,    5,   11,   15,
       5,    9,   11,   15,    5,    1,   13,   15,    3,    7,   13,   15,    1,    1,   15,   15,
};

#define INGOT_IQ3S_GRID_ENTRIES 512
#define INGOT_IQ3S_GRID_WIDTH   4

/* Shared 7-bit sign codebook (IQ2_XXS, IQ3_XXS): bit i of entry k
 * negates value i of the 8-wide group. */
static const uint8_t ingot_iq_ksigns[128] = {
    0x00, 0x81, 0x82, 0x03, 0x84, 0x05, 0x06, 0x87, 0x88, 0x09, 0x0a, 0x8b, 0x0c, 0x8d, 0x8e, 0x0f,
    0x90, 0x11, 0x12, 0x93, 0x14, 0x95, 0x96, 0x17, 0x18, 0x99, 0x9a, 0x1b, 0x9c, 0x1d, 0x1e, 0x9f,
    0xa0, 0x21, 0x22, 0xa3, 0x24, 0xa5, 0xa6, 0x27, 0x28, 0xa9, 0xaa, 0x2b, 0xac, 0x2d, 0x2e, 0xaf,
    0x30, 0xb1, 0xb2, 0x33, 0xb4, 0x35, 0x36, 0xb7, 0xb8, 0x39, 0x3a, 0xbb, 0x3c, 0xbd, 0xbe, 0x3f,
    0xc0, 0x41, 0x42, 0xc3, 0x44, 0xc5, 0xc6, 0x47, 0x48, 0xc9, 0xca, 0x4b, 0xcc, 0x4d, 0x4e, 0xcf,
    0x50, 0xd1, 0xd2, 0x53, 0xd4, 0x55, 0x56, 0xd7, 0xd8, 0x59, 0x5a, 0xdb, 0x5c, 0xdd, 0xde, 0x5f,
    0x60, 0xe1, 0xe2, 0x63, 0xe4, 0x65, 0x66, 0xe7, 0xe8, 0x69, 0x6a, 0xeb, 0x6c, 0xed, 0xee, 0x6f,
    0xf0, 0x71, 0x72, 0xf3, 0x74, 0xf5, 0xf6, 0x77, 0x78, 0xf9, 0xfa, 0x7b, 0xfc, 0x7d, 0x7e, 0xff,
};

/* IQ4_NL / IQ4_XS: 16 non-linear levels */
static const int8_t ingot_iq4nl_values[16] = {
    -127, -104,  -83,  -65,  -49,  -35,  -22,  -10,    1,   13,   25,   38,   53,   69,   89,  113,
};

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

/* ═══ src/quantize.c ═══ */
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

/* ═══ src/kernels.c ═══ */
/* Quantized ggml kernels: dequant, matvec and batched matmat for the K-quant
 * family plus Q8_0, with NEON / AVX2 / ARM-SDOT / ARM-SMMLA paths.
 *
 * This is measured code, and the comments record the measurements and the
 * bugs that shaped it: the int8 activation default, the exactness contract,
 * the triple guard around the intrinsics. The numbers quoted throughout were
 * taken on an M1 unless another machine is named.
 *
 * SPDX-License-Identifier: MIT */

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

/* ═══ src/generic.c ═══ */
/* Type-generic entry points.
 *
 * The specialized kernels (ingot_q4_k_matvec and friends) require the caller
 * to know the format at the call site. That is right for an engine that has
 * decided its weights are Q4_K, and wrong for everything else: a loader that
 * opens whatever GGUF it is handed needs one call that works for any type it
 * can decode.
 *
 * So: dispatch to the hand-written kernel when there is one, and fall back to
 * decode-a-row-then-dot otherwise. The fallback is not slow in the way people
 * expect — it decodes each row ONCE per call, which is the same complexity as
 * the specialized matvec; what it gives up is the SIMD inner loop and, for
 * matmat, the int8 activation path.
 *
 * SPDX-License-Identifier: MIT */

void ingot_parallel_for(size_t count, ingot_range_fn fn, void *user);

typedef int (*matvec_fn)(const void *, size_t, size_t, const float *, float *);
typedef int (*matmat_fn)(const void *, size_t, size_t, const float *, float *, size_t);
typedef int (*deqmat_fn)(const void *, size_t, size_t, float *);

static void specialized(int type, matvec_fn *mv, matmat_fn *mm, deqmat_fn *dq) {
    *mv = NULL; *mm = NULL; *dq = NULL;
    switch (type) {
    case INGOT_TYPE_Q2_K: *mv = ingot_q2_k_matvec; *mm = ingot_q2_k_matmat; *dq = ingot_q2_k_dequant; break;
    case INGOT_TYPE_Q3_K: *mv = ingot_q3_k_matvec; *mm = ingot_q3_k_matmat; *dq = ingot_q3_k_dequant; break;
    case INGOT_TYPE_Q4_K: *mv = ingot_q4_k_matvec; *mm = ingot_q4_k_matmat; *dq = ingot_q4_k_dequant; break;
    case INGOT_TYPE_Q5_K: *mv = ingot_q5_k_matvec; *mm = ingot_q5_k_matmat; *dq = ingot_q5_k_dequant; break;
    case INGOT_TYPE_Q6_K: *mv = ingot_q6_k_matvec; *mm = ingot_q6_k_matmat; *dq = ingot_q6_k_dequant; break;
    case INGOT_TYPE_Q4_0: *mv = ingot_q4_0_matvec; *dq = ingot_q4_0_dequant; break;
    case INGOT_TYPE_Q8_0: *mv = ingot_q8_0_matvec; *mm = ingot_q8_0_matmat; *dq = ingot_q8_0_dequant; break;
    case INGOT_TYPE_BF16: *mv = ingot_bf16_matvec; *mm = ingot_bf16_matmat; *dq = ingot_bf16_dequant; break;
    case INGOT_TYPE_F16:  *mv = ingot_f16_matvec;  *mm = ingot_f16_matmat;  *dq = ingot_f16_dequant;  break;
    default: break;
    }
}

int ingot_has_kernel(int type) {
    matvec_fn mv; matmat_fn mm; deqmat_fn dq;
    specialized(type, &mv, &mm, &dq);
    return mv != NULL;
}

/* Row geometry shared by every generic path. Returns -1 on anything the type
 * cannot represent: unknown type, a row that is not a whole number of blocks,
 * or an overflow. */
static int row_geometry(int type, size_t rows, size_t cols, size_t *row_bytes) {
    uint64_t blk_elems, blk_bytes;
    if (rows == 0 || cols == 0 || ingot_type_geometry(type, &blk_elems, &blk_bytes) != 0)
        return -1;
    if (cols % blk_elems != 0) return -1;
    const uint64_t blocks = cols / blk_elems;
    if (blocks > UINT64_MAX / blk_bytes) return -1;
    const uint64_t bytes = blocks * blk_bytes;
    if (bytes > SIZE_MAX || rows > SIZE_MAX / (size_t)bytes) return -1;
    if (rows > SIZE_MAX / cols) return -1;
    *row_bytes = (size_t)bytes;
    return 0;
}

typedef struct {
    int type;
    const unsigned char *w;
    size_t rows, cols, row_bytes, tokens;
    const float *x;
    float *y;
    int failed;
} job;

/* One scratch row per worker range, not per row: the allocation is amortised
 * over the whole range, and a failure is recorded rather than thrown, because
 * there is nothing to throw to from inside a parallel_for. */
static void generic_range(size_t begin, size_t end, void *user) {
    job *j = (job *)user;
    float *row = (float *)malloc(j->cols * sizeof(float));
    if (row == NULL) { j->failed = 1; return; }
    for (size_t r = begin; r < end && !j->failed; r++) {
        if (ingot_dequant(j->type, j->w + r * j->row_bytes, j->cols, row) != 0) {
            j->failed = 1;
            break;
        }
        for (size_t t = 0; t < j->tokens; t++) {
            const float *x = j->x + t * j->cols;
            float sum = 0.0f;
            for (size_t i = 0; i < j->cols; i++) sum += row[i] * x[i];
            j->y[t * j->rows + r] = sum;
        }
    }
    free(row);
}

static void dequant_range(size_t begin, size_t end, void *user) {
    job *j = (job *)user;
    for (size_t r = begin; r < end && !j->failed; r++)
        if (ingot_dequant(j->type, j->w + r * j->row_bytes, j->cols,
                          j->y + r * j->cols) != 0) j->failed = 1;
}

int ingot_matvec(int type, const void *weights, size_t rows, size_t cols,
                 const float *input, float *output) {
    return ingot_matmat(type, weights, rows, cols, input, output, 1);
}

int ingot_matmat(int type, const void *weights, size_t rows, size_t cols,
                 const float *input, float *output, size_t tokens) {
    if (weights == NULL || input == NULL || output == NULL || tokens == 0) return -1;
    size_t row_bytes;
    if (row_geometry(type, rows, cols, &row_bytes) != 0) return -1;
    if (tokens > SIZE_MAX / rows) return -1;

    matvec_fn mv; matmat_fn mm; deqmat_fn dq;
    specialized(type, &mv, &mm, &dq);
    if (mm != NULL) return mm(weights, rows, cols, input, output, tokens);
    if (mv != NULL && tokens == 1) return mv(weights, rows, cols, input, output);
    if (!ingot_type_can_dequant(type)) return -1;

    job j = { type, (const unsigned char *)weights, rows, cols, row_bytes,
              tokens, input, output, 0 };
    ingot_parallel_for(rows, generic_range, &j);
    return j.failed ? -1 : 0;
}

int ingot_dequant_matrix(int type, const void *weights, size_t rows, size_t cols,
                         float *output) {
    if (weights == NULL || output == NULL) return -1;
    size_t row_bytes;
    if (row_geometry(type, rows, cols, &row_bytes) != 0) return -1;

    matvec_fn mv; matmat_fn mm; deqmat_fn dq;
    specialized(type, &mv, &mm, &dq);
    if (dq != NULL) return dq(weights, rows, cols, output);
    if (!ingot_type_can_dequant(type)) return -1;

    job j = { type, (const unsigned char *)weights, rows, cols, row_bytes,
              1, NULL, output, 0 };
    ingot_parallel_for(rows, dequant_range, &j);
    return j.failed ? -1 : 0;
}

/* ── straight off a GGUF tensor ─────────────────────────────────────────────
 * ggml stores ne[0] as the fastest dimension, which for a 2D weight matrix is
 * the input width. So rows = ne[1] and cols = ne[0] — the flip that every
 * consumer otherwise writes out by hand at each call site, and gets wrong
 * once. Rank 1 is treated as a single row. */
static int tensor_shape(const ingot_tensor *t, size_t *rows, size_t *cols) {
    if (t == NULL || t->rank == 0 || t->rank > 2) return -1;
    *cols = (size_t)t->ne[0];
    *rows = (t->rank == 2) ? (size_t)t->ne[1] : 1;
    return (*rows != 0 && *cols != 0) ? 0 : -1;
}

int ingot_gguf_matvec(const ingot_gguf *g, const ingot_tensor *t,
                      const float *input, float *output) {
    return ingot_gguf_matmat(g, t, input, output, 1);
}

int ingot_gguf_matmat(const ingot_gguf *g, const ingot_tensor *t,
                      const float *input, float *output, size_t tokens) {
    size_t rows, cols;
    const void *w = ingot_gguf_data(g, t);
    if (w == NULL || tensor_shape(t, &rows, &cols) != 0) return -1;
    return ingot_matmat(t->type, w, rows, cols, input, output, tokens);
}

int ingot_gguf_dequant_matrix(const ingot_gguf *g, const ingot_tensor *t, float *output) {
    size_t rows, cols;
    const void *w = ingot_gguf_data(g, t);
    if (w == NULL || tensor_shape(t, &rows, &cols) != 0) return -1;
    return ingot_dequant_matrix(t->type, w, rows, cols, output);
}

#endif /* INGOT_NO_KERNELS */
