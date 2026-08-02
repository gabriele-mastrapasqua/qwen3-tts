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
#ifndef INGOT_GGUF_H
#define INGOT_GGUF_H

#include <stddef.h>
#include <stdint.h>

#include "ingot/dtype.h"

#ifdef __cplusplus
extern "C" {
#endif

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

#ifdef __cplusplus
}
#endif
#endif
