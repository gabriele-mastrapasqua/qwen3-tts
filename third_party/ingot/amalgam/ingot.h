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

#ifndef INGOT_AMALGAM_H
#define INGOT_AMALGAM_H

#ifdef __cplusplus
extern "C" {
#endif

/* ═══ include/ingot/dtype.h ═══ */
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

#include <stddef.h>
#include <stdint.h>


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

/* ═══ include/ingot/gguf.h ═══ */
/* ingot/gguf.h — GGUF v2/v3 reader.
 *
 * What it does: opens the container, validates every offset against the file
 * size, and hands back tensor descriptors plus a zero-copy pointer to each
 * payload. Tensors stay in the type they were stored in — dequantization is a
 * separate, explicit call (see ingot/quant.h), never a side effect of opening.
 *
 * What it does not do: no compute graph, no layout rewriting, no tokenizer.
 * The metadata key/value store is exposed raw, so a caller that needs
 * `tokenizer.ggml.tokens` gets the array and does its own thing with it.
 *
 * Errors are returned as a code plus a message in the caller's buffer. This
 * library never writes to stderr.
 *
 * SPDX-License-Identifier: MIT */

#include <stddef.h>
#include <stdint.h>



typedef struct ingot_gguf ingot_gguf;
typedef struct ingot_kv   ingot_kv;

/* GGUF metadata value types (the numeric values are the on-disk ones). */
enum {
    INGOT_KV_UINT8 = 0, INGOT_KV_INT8   = 1,  INGOT_KV_UINT16 = 2,
    INGOT_KV_INT16 = 3, INGOT_KV_UINT32 = 4,  INGOT_KV_INT32  = 5,
    INGOT_KV_FLOAT32 = 6, INGOT_KV_BOOL = 7,  INGOT_KV_STRING = 8,
    INGOT_KV_ARRAY = 9, INGOT_KV_UINT64 = 10, INGOT_KV_INT64  = 11,
    INGOT_KV_FLOAT64 = 12
};

typedef struct {
    const char *name;
    int         type;                     /* INGOT_TYPE_*                     */
    uint32_t    rank;
    uint64_t    ne[INGOT_MAX_RANK];       /* ggml order: ne[0] is the fastest */
    uint64_t    nelem;
    uint64_t    nbytes;                   /* exact on-disk size               */
    uint64_t    offset;                   /* relative to the shard data base  */
    uint32_t    shard;                    /* 0 unless the file was split      */
} ingot_tensor;

/* ── open / close ───────────────────────────────────────────────────────── */

/* Contract shared by every open() here: on success *out owns a handle the
 * caller must close; on failure *out is set to NULL. That second half is
 * deliberate — it makes an ignored return code fail safely — but it means you
 * must not pass a variable that still owns a live handle, because the old
 * pointer is overwritten, not closed. `make test-leaks` catches that mistake. */

/* Open one GGUF file. */
int  ingot_gguf_open(ingot_gguf **out, const char *path, char *err, size_t errsz);

/* Open a split model. `path` may be any shard whose name follows the
 * llama.cpp convention `<prefix>-00001-of-000NN.gguf`; every shard is opened
 * and the tensor tables are merged. A path without that suffix behaves exactly
 * like ingot_gguf_open(). */
int  ingot_gguf_open_split(ingot_gguf **out, const char *path, char *err, size_t errsz);

void ingot_gguf_close(ingot_gguf *g);

/* ── tensors ────────────────────────────────────────────────────────────── */

size_t              ingot_gguf_count(const ingot_gguf *g);
const ingot_tensor *ingot_gguf_at(const ingot_gguf *g, size_t index);
const ingot_tensor *ingot_gguf_find(const ingot_gguf *g, const char *name); /* O(1) */

/* Zero-copy pointer into the mmap. Valid until ingot_gguf_close. NULL when the
 * tensor does not belong to this handle. */
const void *ingot_gguf_data(const ingot_gguf *g, const ingot_tensor *t);

/* The same bytes through pread, for callers that cannot or will not touch the
 * mapping (uploading straight to a device, reading a file on a network mount
 * where a page fault storm is the wrong shape of I/O). Returns 0 on success. */
int ingot_gguf_read(const ingot_gguf *g, const ingot_tensor *t, uint64_t offset,
                    void *dst, size_t nbytes, char *err, size_t errsz);

/* The whole read-only mapping of one shard, so a GPU backend can register it
 * with no copy. Returns -1 when `shard` is out of range. */
int ingot_gguf_mapping(const ingot_gguf *g, uint32_t shard,
                       const void **base, size_t *size);

/* Shape in row-major order (ne reversed), which is what everything outside
 * ggml expects. Writes `rank` entries into `shape`. */
void ingot_gguf_shape_row_major(const ingot_tensor *t, uint64_t *shape);

/* ── metadata ───────────────────────────────────────────────────────────── */

size_t          ingot_gguf_kv_count(const ingot_gguf *g);
const ingot_kv *ingot_gguf_kv_at(const ingot_gguf *g, size_t index);
const ingot_kv *ingot_gguf_kv_find(const ingot_gguf *g, const char *key); /* O(1) */

const char *ingot_kv_key(const ingot_kv *kv);
int         ingot_kv_type(const ingot_kv *kv);

/* Scalar accessors. Each returns 0 on success and -1 when the stored type is
 * not convertible (an int KV reads fine through _u64/_i64/_f64; a string does
 * not). Scalar strings are NUL-terminated copies owned by the handle. */
int ingot_kv_str(const ingot_kv *kv, const char **out);
int ingot_kv_u64(const ingot_kv *kv, uint64_t *out);
int ingot_kv_i64(const ingot_kv *kv, int64_t *out);
int ingot_kv_f64(const ingot_kv *kv, double *out);
int ingot_kv_bool(const ingot_kv *kv, int *out);

/* Arrays. `ingot_kv_arr_type` returns the element type, `_len` the count.
 * String elements come back as a pointer into the mapping plus a length — they
 * are NOT NUL-terminated, because a 150k-entry vocabulary is not worth 150k
 * mallocs. Indexing is O(1): the string offsets are recorded while parsing,
 * which the parser has to walk anyway. */
int ingot_kv_arr_type(const ingot_kv *kv);
int ingot_kv_arr_len(const ingot_kv *kv, uint64_t *out);
int ingot_kv_arr_str(const ingot_kv *kv, uint64_t index, const char **out, size_t *len);
int ingot_kv_arr_f32(const ingot_kv *kv, uint64_t index, float *out);
int ingot_kv_arr_i64(const ingot_kv *kv, uint64_t index, int64_t *out);

/* Convenience wrappers over the above, for the three keys every caller wants.
 * `ingot_gguf_arch` returns "" rather than NULL when the key is absent, so it
 * is safe to strcmp directly. */
const char *ingot_gguf_arch(const ingot_gguf *g);
uint64_t    ingot_gguf_alignment(const ingot_gguf *g);
uint32_t    ingot_gguf_version(const ingot_gguf *g);
uint32_t    ingot_gguf_shard_count(const ingot_gguf *g);
/* Absolute file offset where a shard's payload section starts. Only needed to
 * turn a tensor's relative offset into an absolute one — for a device upload
 * that wants file offsets rather than pointers. Returns 0 for a bad shard. */
uint64_t    ingot_gguf_data_base(const ingot_gguf *g, uint32_t shard);

/* ═══ include/ingot/safetensors.h ═══ */
/* ingot/safetensors.h — safetensors reader (single file, directory, or an
 * explicit list of shards).
 *
 * Same contract as the GGUF side: mmap, validate every range against the file,
 * hand back zero-copy pointers, never convert behind the caller's back, never
 * write to stderr.
 *
 * SPDX-License-Identifier: MIT */

#include <stddef.h>
#include <stdint.h>



typedef struct ingot_st ingot_st;

typedef struct {
    const char *name;
    ingot_dtype dtype;
    uint32_t    rank;
    uint64_t    shape[INGOT_MAX_RANK];   /* row-major, exactly as written */
    uint64_t    nelem;
    uint64_t    offset;                  /* relative to the shard data start */
    uint64_t    nbytes;
    uint32_t    shard;
} ingot_st_tensor;

/* ── open / close ───────────────────────────────────────────────────────── */

/* Contract shared by every open() here: on success *out owns a handle the
 * caller must close; on failure *out is set to NULL. That second half is
 * deliberate — it makes an ignored return code fail safely — but it means you
 * must not pass a variable that still owns a live handle, because the old
 * pointer is overwritten, not closed. `make test-leaks` catches that mistake. */

/* A file, or a directory (in which case this behaves as ingot_st_open_dir). */
int ingot_st_open(ingot_st **out, const char *path, char *err, size_t errsz);

/* Resolve a model directory, in this order:
 *   1. model.safetensors.index.json  — the weight_map names the shards
 *   2. model.safetensors             — the single-file case
 *   3. every *.safetensors in the directory, opened in sorted order
 * There is no fixed cap on shards or tensors: a model that does not fit is an
 * error with a message, never a silent truncation. */
int ingot_st_open_dir(ingot_st **out, const char *dir, char *err, size_t errsz);

/* An explicit shard list, for callers that already know the layout. */
int ingot_st_open_shards(ingot_st **out, const char *const *paths, size_t count,
                         char *err, size_t errsz);

void ingot_st_close(ingot_st *st);

/* ── tensors ────────────────────────────────────────────────────────────── */

size_t                 ingot_st_count(const ingot_st *st);
const ingot_st_tensor *ingot_st_at(const ingot_st *st, size_t index);
const ingot_st_tensor *ingot_st_find(const ingot_st *st, const char *name); /* O(1) */

const void *ingot_st_data(const ingot_st *st, const ingot_st_tensor *t);
int         ingot_st_read(const ingot_st *st, const ingot_st_tensor *t, uint64_t offset,
                          void *dst, size_t nbytes, char *err, size_t errsz);

/* Cast a whole tensor into a caller-owned f32 buffer of `t->nelem` floats.
 * F32 is a memcpy; BF16/F16/F8/int types are converted. -1 when the dtype has
 * no f32 form. */
int ingot_st_to_f32(const ingot_st *st, const ingot_st_tensor *t, float *dst);

/* ── shards and metadata ────────────────────────────────────────────────── */

uint32_t    ingot_st_shard_count(const ingot_st *st);
const char *ingot_st_shard_path(const ingot_st *st, uint32_t shard);
int         ingot_st_mapping(const ingot_st *st, uint32_t shard,
                             const void **base, size_t *size);

/* A key from the header's "__metadata__" object of shard 0 (string values
 * only). Returns 0 and a NUL-terminated pointer owned by the handle. */
int ingot_st_metadata(const ingot_st *st, const char *key, const char **out);

/* ── page-cache ergonomics ──────────────────────────────────────────────────
 * These exist because a checkpoint read once and then uploaded to a device
 * otherwise leaves its whole copy in the page cache — on unified memory that
 * cache competes with the GPU's own allocations. Measured: 23 GB held for a
 * 12B checkpoint alongside 46 GB for a second model, on a 121 GiB machine.
 *
 *  - prefault: touch every page now, so a later first-touch storm does not
 *    happen mid-inference (worth it on NAS/slow NVMe).
 *  - dontneed: drop the mapped pages, keeping the handle open.
 *  - set_drop_cache: on close, also drop the file's page cache. Do NOT use it
 *    for files re-read on every run: reading them back from NVMe is not free. */
void ingot_st_prefault(ingot_st *st);
void ingot_st_dontneed(ingot_st *st);
void ingot_st_set_drop_cache(ingot_st *st, int drop);

/* ═══ include/ingot/quant.h ═══ */
/* ingot/quant.h — dequantization and quantized kernels for the ggml block
 * formats carried inside GGUF.
 *
 * This module is optional: build with INGOT_NO_QUANT defined (or just do not
 * link quant.o / kernels.o) and the container readers still work. Nobody who
 * only wants to read a file should pay for the SIMD.
 *
 * Two layers:
 *   1. ingot_dequant()  — spec-faithful scalar decode of a block format to
 *      f32. Writes into a caller-owned buffer. Never allocates.
 *   2. the kernels      — matvec / batched matmat straight off the quantized
 *      weights, which is the whole point of the format: dequantizing a 7B
 *      model to f32 costs 28 GB, multiplying against it in place costs zero.
 *
 * SPDX-License-Identifier: MIT */

#include <stddef.h>
#include <stdint.h>



/* ── dequantization ─────────────────────────────────────────────────────── */

/* `src` holds `nelem` elements in `type`'s block layout; `dst` receives nelem
 * floats. `nelem` must be a multiple of the block size. Returns -1 for a type
 * this library cannot decode (ask ingot_type_can_dequant first). */
int ingot_dequant(int type, const void *src, size_t nelem, float *dst);

/* The same, straight off a GGUF tensor. `dst` needs t->nelem floats. */
int ingot_gguf_dequant(const ingot_gguf *g, const ingot_tensor *t, float *dst);

/* ── CPU dispatch ───────────────────────────────────────────────────────── */

typedef struct {
    int neon, dotprod, i8mm;      /* ARM: compiled AND present at runtime */
    int avx2, avx512, avx512_vnni;/* x86: same                            */
    int bf16;                     /* ARM FEAT_BF16 (BFDOT/BFMMLA)         */
    int f16c, avx512_bf16;        /* x86: VCVTPH2PS / VDPBF16PS           */
} ingot_cpu_caps;

ingot_cpu_caps ingot_cpu(void);

/* Cap the SIMD level by hand: "auto" (default), "scalar", "avx2", "vnni".
 * A level above what the CPU supports is silently downgraded. Returns the
 * effective level (0 scalar, 1 avx2/neon, 2 vnni/dotprod). The environment
 * variable INGOT_CAPS does the same thing without a code change.
 *
 * INGOT_CAPS_ASSUME=avx2|neon|dotprod|vnni goes the OTHER way: it trusts the
 * BUILD rather than CPUID, turning on a feature the CPU did not report. Only
 * what was compiled in can be forced, so it cannot conjure an instruction the
 * binary does not contain. It exists for emulators that under-report — Rosetta
 * 2 executes AVX2 without advertising it, and without this the x86 kernels
 * cannot be exercised on an Apple Silicon machine at all. A testing tool: it
 * will fault on hardware that genuinely lacks the feature. */
int ingot_cpu_set_level(const char *name);

/* ── thread injection ───────────────────────────────────────────────────────
 * The batched kernels parallelise over row strips. ingot does not own a thread
 * pool — every consumer already has one, and two pools fighting over the same
 * cores is a measurable loss. Hand yours in here; the default runs inline. */
typedef void (*ingot_range_fn)(size_t begin, size_t end, void *user);
typedef void (*ingot_parallel_for_fn)(size_t count, ingot_range_fn fn, void *user);
void ingot_set_parallel_for(ingot_parallel_for_fn fn);

/* ── kernels ────────────────────────────────────────────────────────────────
 * Layout: `weights` is a row-major [rows, cols] matrix of blocks, exactly as
 * stored in the GGUF (cols must be a multiple of the block size).
 *   matvec: output[rows]            = weights * input[cols]
 *   matmat: output[tokens][rows]    = input[tokens][cols] * weights^T
 *   dequant: output[rows*cols] f32
 * All return 0 on success. */

int ingot_q4_k_matvec(const void *weights, size_t rows, size_t cols,
                      const float *input, float *output);
int ingot_q4_k_matvec_scalar(const void *weights, size_t rows, size_t cols,
                             const float *input, float *output);
int ingot_q4_k_dequant(const void *weights, size_t rows, size_t cols, float *output);
int ingot_q4_k_matmat(const void *weights, size_t rows, size_t cols,
                      const float *input, float *output, size_t tokens);

int ingot_q5_k_matvec(const void *weights, size_t rows, size_t cols,
                      const float *input, float *output);
int ingot_q5_k_matvec_scalar(const void *weights, size_t rows, size_t cols,
                             const float *input, float *output);
int ingot_q5_k_dequant(const void *weights, size_t rows, size_t cols, float *output);
int ingot_q5_k_matmat(const void *weights, size_t rows, size_t cols,
                      const float *input, float *output, size_t tokens);

int ingot_q6_k_matvec(const void *weights, size_t rows, size_t cols,
                      const float *input, float *output);
int ingot_q6_k_dequant(const void *weights, size_t rows, size_t cols, float *output);
int ingot_q6_k_matmat(const void *weights, size_t rows, size_t cols,
                      const float *input, float *output, size_t tokens);

int ingot_q3_k_matvec(const void *weights, size_t rows, size_t cols,
                      const float *input, float *output);
int ingot_q3_k_dequant(const void *weights, size_t rows, size_t cols, float *output);
int ingot_q3_k_matmat(const void *weights, size_t rows, size_t cols,
                      const float *input, float *output, size_t tokens);

int ingot_q2_k_matvec(const void *weights, size_t rows, size_t cols,
                      const float *input, float *output);
int ingot_q2_k_dequant(const void *weights, size_t rows, size_t cols, float *output);
int ingot_q2_k_matmat(const void *weights, size_t rows, size_t cols,
                      const float *input, float *output, size_t tokens);

/* Q4_0: 32 weights in 18 bytes, split-half nibbles biased by 8. What Gemma 4
 * QAT checkpoints ship as, which is why it has a hand-written kernel. */
int ingot_q4_0_matvec(const void *weights, size_t rows, size_t cols,
                      const float *input, float *output);
int ingot_q4_0_dequant(const void *weights, size_t rows, size_t cols, float *output);

int ingot_q8_0_matvec(const void *weights, size_t rows, size_t cols,
                      const float *input, float *output);
int ingot_q8_0_dequant(const void *weights, size_t rows, size_t cols, float *output);
int ingot_q8_0_matmat(const void *weights, size_t rows, size_t cols,
                      const float *input, float *output, size_t tokens);

/* ── dense F16 / BF16 kernels ───────────────────────────────────────────────
 * `weights` is the little-endian 2-byte stream exactly as stored in the file
 * (zero-copy from the mapping), row-major [rows, cols]; input and output are
 * f32. The widening is exact, so the result is plain f32 arithmetic on the
 * stored values — a reordered sum, never an approximation. matvec is
 * single-threaded like the quantized ones; matmat parallelizes over rows
 * through ingot_set_parallel_for. */
int ingot_bf16_matvec(const void *weights, size_t rows, size_t cols,
                      const float *input, float *output);
int ingot_bf16_matmat(const void *weights, size_t rows, size_t cols,
                      const float *input, float *output, size_t tokens);
int ingot_bf16_dequant(const void *weights, size_t rows, size_t cols, float *output);
int ingot_f16_matvec(const void *weights, size_t rows, size_t cols,
                     const float *input, float *output);
int ingot_f16_matmat(const void *weights, size_t rows, size_t cols,
                     const float *input, float *output, size_t tokens);
int ingot_f16_dequant(const void *weights, size_t rows, size_t cols, float *output);

/* ── the precision contract ─────────────────────────────────────────────────
 * From two tokens up, Q4_K and Q5_K batched matmat quantize the ACTIVATIONS to
 * int8 by default: 1.5-2.5x faster, at a relative error around 2.4e-3 instead
 * of the last few ulp of a reorder. Everything else stays an exact reorder.
 *
 * This matters when the CPU result is the reference a GPU path is compared
 * against. Ask instead of guessing: a gate that picks 5e-3 where 1e-5 is owed
 * passes for the wrong reason, and one that picks 1e-5 where 5e-3 is owed
 * fails for the wrong reason.
 *   - ingot_matmat_is_exact(tokens): non-zero when this width takes the exact
 *     path.
 *   - the *_exact twins: always exact, whatever the default is.
 *   - INGOT_SDOT=0 in the environment: turn the int8 path off globally. */
int ingot_matmat_is_exact(size_t tokens);
int ingot_q4_k_matmat_exact(const void *weights, size_t rows, size_t cols,
                            const float *input, float *output, size_t tokens);
int ingot_q5_k_matmat_exact(const void *weights, size_t rows, size_t cols,
                            const float *input, float *output, size_t tokens);

/* ── type-generic entry points ──────────────────────────────────────────────
 * The calls above make you name the format at the call site, which is right
 * for an engine that has decided its weights are Q4_K and wrong for a loader
 * handed whatever GGUF the user downloaded. These work for EVERY type
 * ingot_type_can_dequant() accepts, dispatching to the hand-written kernel
 * when one exists and decoding row-by-row otherwise (same complexity, no SIMD
 * inner loop, no int8 activation path).
 *
 *   matvec:  output[rows]         = weights * input[cols]
 *   matmat:  output[tokens][rows] = input[tokens][cols] * weights^T
 *
 * ingot_has_kernel() says whether a type takes the fast route, so a caller
 * choosing a quantization can prefer one that does. */
int ingot_matvec(int type, const void *weights, size_t rows, size_t cols,
                 const float *input, float *output);
int ingot_matmat(int type, const void *weights, size_t rows, size_t cols,
                 const float *input, float *output, size_t tokens);
int ingot_dequant_matrix(int type, const void *weights, size_t rows, size_t cols,
                         float *output);
int ingot_has_kernel(int type);

/* The same, straight off a GGUF tensor: rows and cols come from `ne`, so the
 * ggml dimension flip (ne[0] is the INPUT width) happens once here instead of
 * at every call site. Rank 1 counts as a single row. */
int ingot_gguf_matvec(const ingot_gguf *g, const ingot_tensor *t,
                      const float *input, float *output);
int ingot_gguf_matmat(const ingot_gguf *g, const ingot_tensor *t,
                      const float *input, float *output, size_t tokens);
int ingot_gguf_dequant_matrix(const ingot_gguf *g, const ingot_tensor *t, float *output);

/* ── f32 -> block format ────────────────────────────────────────────────────
 * `count` must be a multiple of the block size and `out` must hold
 * count / block_elems * block_bytes bytes (ask ingot_type_nbytes).
 * Supported: F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K, Q3_K, Q4_K,
 * Q5_K, Q6_K. The K-quant encoders are unweighted least-squares fits; ggml
 * additionally weights them with an imatrix, which buys several percent of
 * relative error at 2-3 bits and almost nothing at 6. */
int ingot_quantize(int type, const float *values, size_t count, void *out);
int ingot_can_quantize(int type);

/* f32 -> Q4_K. `count` must be a multiple of 256; `out` needs count/256*144
 * bytes. The one direction a converter needs and no reader provides. */
int ingot_q4_k_quantize(const float *values, size_t count, void *out);

/* ═══ include/ingot/wfile.h ═══ */
/* ingot/wfile.h — one handle for either container.
 *
 * Sniff the magic and open a GGUF or a safetensors behind the same API, so the
 * engine above never learns which file its weights came from and there is a
 * single code path after load.
 *
 * The trade is deliberate. A wfile tensor is normalised — row-major shape, one
 * dtype vocabulary — which means the ggml block types collapse into
 * INGOT_DT_UNKNOWN with a `ggml_type` field to say which one. Quantized
 * weights are still handed over as-is; use ingot/quant.h on the raw pointer,
 * or ingot_wfile_to_f32() when a float buffer is what you want.
 *
 * If you already know which container you have, use ingot/gguf.h or
 * ingot/safetensors.h directly: this layer buys uniformity, not power.
 *
 * SPDX-License-Identifier: MIT */

#include <stddef.h>
#include <stdint.h>



typedef struct ingot_wfile ingot_wfile;

typedef enum { INGOT_CONTAINER_GGUF = 0, INGOT_CONTAINER_SAFETENSORS = 1 } ingot_container;

typedef struct {
    const char *name;
    ingot_dtype dtype;       /* INGOT_DT_UNKNOWN for a ggml block type */
    int         ggml_type;   /* -1 when the source was safetensors      */
    uint32_t    rank;
    uint64_t    shape[INGOT_MAX_RANK];   /* ROW-MAJOR, both containers   */
    uint64_t    nelem;
    uint64_t    nbytes;
    const void *data;        /* into the mapping, read-only              */
} ingot_wtensor;

/* Contract shared by every open() here: on success *out owns a handle the
 * caller must close; on failure *out is set to NULL. That second half is
 * deliberate — it makes an ignored return code fail safely — but it means you
 * must not pass a variable that still owns a live handle, because the old
 * pointer is overwritten, not closed. `make test-leaks` catches that mistake. */

/* `path` may be a .gguf, a .safetensors, or a model directory. */
int  ingot_wfile_open(ingot_wfile **out, const char *path, char *err, size_t errsz);
void ingot_wfile_close(ingot_wfile *w);

ingot_container      ingot_wfile_container(const ingot_wfile *w);
size_t               ingot_wfile_count(const ingot_wfile *w);
const ingot_wtensor *ingot_wfile_at(const ingot_wfile *w, size_t index);
const ingot_wtensor *ingot_wfile_find(const ingot_wfile *w, const char *name); /* O(1) */

/* `dst` needs t->nelem floats. Handles both the ggml block types and the
 * safetensors dtypes. -1 when the format has no f32 decode in this build. */
int ingot_wfile_to_f32(const ingot_wfile *w, const ingot_wtensor *t, float *dst);

/* The underlying handle, when you need something only one side has (GGUF
 * metadata, safetensors page-cache control). NULL for the other container. */
const ingot_gguf *ingot_wfile_gguf(const ingot_wfile *w);
ingot_st         *ingot_wfile_st(const ingot_wfile *w);

/* ═══ include/ingot/write.h ═══ */
/* ingot/write.h — writing GGUF and safetensors.
 *
 * Enough to build a converter and, just as usefully, to build fixtures: a
 * reader whose tests can only consume files somebody else wrote is a reader
 * with a blind spot. Round-tripping through these writers is how the reader's
 * corner cases get exercised without a model on disk.
 *
 * Both writers hold POINTERS to the caller's tensor data until save() — they
 * do not copy. The one exception is ingot_gguf_add_f32(), which has to
 * allocate because it quantizes on the way in. Keep your buffers alive until
 * you have called save().
 *
 * SPDX-License-Identifier: MIT */

#include <stddef.h>
#include <stdint.h>



/* ── GGUF ───────────────────────────────────────────────────────────────── */

typedef struct ingot_gguf_writer ingot_gguf_writer;

ingot_gguf_writer *ingot_gguf_writer_new(void);
void               ingot_gguf_writer_free(ingot_gguf_writer *w);

/* Metadata. Keys are copied; string values are copied; array payloads are
 * copied too, because they are small and the alternative is a lifetime rule
 * nobody remembers. All return 0 on success. */
int ingot_gguf_kv_string(ingot_gguf_writer *w, const char *key, const char *value);
int ingot_gguf_kv_u32(ingot_gguf_writer *w, const char *key, uint32_t value);
int ingot_gguf_kv_u64(ingot_gguf_writer *w, const char *key, uint64_t value);
int ingot_gguf_kv_i32(ingot_gguf_writer *w, const char *key, int32_t value);
int ingot_gguf_kv_f32(ingot_gguf_writer *w, const char *key, float value);
int ingot_gguf_kv_bool(ingot_gguf_writer *w, const char *key, int value);
int ingot_gguf_kv_array_string(ingot_gguf_writer *w, const char *key,
                               const char *const *values, size_t count);
int ingot_gguf_kv_array_f32(ingot_gguf_writer *w, const char *key,
                            const float *values, size_t count);
int ingot_gguf_kv_array_i32(ingot_gguf_writer *w, const char *key,
                            const int32_t *values, size_t count);

/* A tensor already in its final block format. `ne` is in GGML order (ne[0] is
 * the fastest dimension); `data` must hold exactly ingot_type_nbytes() bytes
 * and stay alive until save(). */
int ingot_gguf_add_tensor(ingot_gguf_writer *w, const char *name, int type,
                          uint32_t rank, const uint64_t *ne, const void *data);

/* The converter's entry point: hand over f32 and a target type, and the
 * writer quantizes into a buffer it owns. Falls back to an error when the
 * target has no encoder — ask ingot_can_quantize() first if you want to pick
 * a fallback rather than fail. */
int ingot_gguf_add_f32(ingot_gguf_writer *w, const char *name, int type,
                       uint32_t rank, const uint64_t *ne, const float *values);

/* Writes magic, version 3, the KV block, the tensor table, and the payloads
 * each padded to `general.alignment` (32 unless you set it). */
int ingot_gguf_writer_save(ingot_gguf_writer *w, const char *path,
                           char *err, size_t errsz);

/* ── safetensors ────────────────────────────────────────────────────────── */

typedef struct ingot_st_writer ingot_st_writer;

ingot_st_writer *ingot_st_writer_new(void);
void             ingot_st_writer_free(ingot_st_writer *w);

int ingot_st_writer_metadata(ingot_st_writer *w, const char *key, const char *value);
int ingot_st_writer_add(ingot_st_writer *w, const char *name, ingot_dtype dtype,
                        uint32_t rank, const uint64_t *shape, const void *data);
/* The header is padded with spaces so the data section starts 8-byte aligned,
 * which is the contract every reader (including this one) relies on. */
int ingot_st_writer_save(ingot_st_writer *w, const char *path,
                         char *err, size_t errsz);

#ifdef __cplusplus
}
#endif
#endif /* INGOT_AMALGAM_H */
