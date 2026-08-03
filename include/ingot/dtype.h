/* ingot/dtype.h — the type systems of the two containers, plus the scalar
 * conversions everybody re-implements.
 *
 * The two formats have genuinely different type systems: GGUF carries *ggml*
 * types, which are block-quantized (a "type" is a layout of N values in M
 * bytes), while safetensors carries plain element dtypes. This header keeps
 * them apart on purpose — pretending they are one enum costs more than it
 * saves.
 *
 * SPDX-License-Identifier: MIT */
#ifndef INGOT_DTYPE_H
#define INGOT_DTYPE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define INGOT_MAX_RANK 8

/* ── ggml tensor types (the numeric values are the on-disk ones) ────────── */
enum {
    INGOT_TYPE_F32      = 0,
    INGOT_TYPE_F16      = 1,
    INGOT_TYPE_Q4_0     = 2,
    INGOT_TYPE_Q4_1     = 3,
    INGOT_TYPE_Q5_0     = 6,
    INGOT_TYPE_Q5_1     = 7,
    INGOT_TYPE_Q8_0     = 8,
    INGOT_TYPE_Q8_1     = 9,
    INGOT_TYPE_Q2_K     = 10,
    INGOT_TYPE_Q3_K     = 11,
    INGOT_TYPE_Q4_K     = 12,
    INGOT_TYPE_Q5_K     = 13,
    INGOT_TYPE_Q6_K     = 14,
    INGOT_TYPE_Q8_K     = 15,
    INGOT_TYPE_IQ2_XXS  = 16,
    INGOT_TYPE_IQ2_XS   = 17,
    INGOT_TYPE_IQ3_XXS  = 18,
    INGOT_TYPE_IQ1_S    = 19,
    INGOT_TYPE_IQ4_NL   = 20,
    INGOT_TYPE_IQ3_S    = 21,
    INGOT_TYPE_IQ2_S    = 22,
    INGOT_TYPE_IQ4_XS   = 23,
    INGOT_TYPE_I8       = 24,
    INGOT_TYPE_I16      = 25,
    INGOT_TYPE_I32      = 26,
    INGOT_TYPE_I64      = 27,
    INGOT_TYPE_F64      = 28,
    INGOT_TYPE_IQ1_M    = 29,
    INGOT_TYPE_BF16     = 30,
    INGOT_TYPE_TQ1_0    = 34,
    INGOT_TYPE_TQ2_0    = 35,
    INGOT_TYPE_MXFP4    = 39,
    INGOT_TYPE_NVFP4    = 40,
    INGOT_TYPE_Q1_0     = 41
};

/* Block geometry: how many elements live in one block and how many bytes that
 * block occupies. Returns 0 on success, -1 for a type this build does not know
 * at all.
 *
 * KNOWING the geometry and being able to DEQUANTIZE are two different things.
 * Every type listed above has a geometry — so a file using it opens, its byte
 * accounting is exact, and an error message can name the type — but only the
 * ones for which ingot_type_can_dequant() is non-zero can be turned into f32
 * by this library. That is the difference between
 *   "tensor 'x' is IQ4_XS, which ingot cannot dequantize"
 * and
 *   "unknown ggml type 23". */
int ingot_type_geometry(int type, uint64_t *block_elems, uint64_t *block_bytes);
const char *ingot_type_name(int type);
int ingot_type_is_quantized(int type);
int ingot_type_can_dequant(int type);

/* Byte size of a whole tensor of `type` holding `nelem` elements. Returns -1
 * on an unknown type, on nelem not being a multiple of the block size, or on
 * overflow. */
int ingot_type_nbytes(int type, uint64_t nelem, uint64_t *out);

/* ── safetensors element dtypes ─────────────────────────────────────────── */
typedef enum {
    INGOT_DT_F32 = 0,
    INGOT_DT_F64,
    INGOT_DT_F16,
    INGOT_DT_BF16,
    INGOT_DT_F8_E4M3,
    INGOT_DT_F8_E5M2,
    INGOT_DT_I8,
    INGOT_DT_U8,
    INGOT_DT_I16,
    INGOT_DT_U16,
    INGOT_DT_I32,
    INGOT_DT_U32,
    INGOT_DT_I64,
    INGOT_DT_U64,
    INGOT_DT_BOOL,
    INGOT_DT_UNKNOWN = -1
} ingot_dtype;

size_t      ingot_dtype_size(ingot_dtype dtype);   /* 0 when unknown */
const char *ingot_dtype_name(ingot_dtype dtype);
ingot_dtype ingot_dtype_from_name(const char *name);

/* ── scalar conversions ─────────────────────────────────────────────────── */
/* f16 handles subnormals and inf/nan properly. The naive shift-only version,
 * which three of the five projects this library was distilled from shipped,
 * silently flushes subnormals to zero. */
float    ingot_f16_to_f32(uint16_t value);
uint16_t ingot_f32_to_f16(float value);
float    ingot_bf16_to_f32(uint16_t value);
uint16_t ingot_f32_to_bf16(float value);   /* round-to-nearest-even */
float    ingot_f8_e4m3_to_f32(uint8_t value);
float    ingot_f8_e5m2_to_f32(uint8_t value);

/* Bulk conversion of a whole buffer — all the safetensors side ever needs,
 * since it has no block types. Returns -1 only for INGOT_DT_UNKNOWN; every
 * other dtype converts, rounding where f32 cannot represent the value
 * exactly (BOOL becomes 0.0/1.0, U64/I64 above 2^24..2^53 lose low bits). */
int ingot_dtype_to_f32(ingot_dtype dtype, const void *src, size_t nelem, float *dst);

#ifdef __cplusplus
}
#endif
#endif
