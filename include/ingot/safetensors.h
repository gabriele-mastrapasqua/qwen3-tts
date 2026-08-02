/* ingot/safetensors.h — safetensors reader (single file, directory, or an
 * explicit list of shards).
 *
 * Same contract as the GGUF side: mmap, validate every range against the file,
 * hand back zero-copy pointers, never convert behind the caller's back, never
 * write to stderr.
 *
 * SPDX-License-Identifier: MIT */
#ifndef INGOT_SAFETENSORS_H
#define INGOT_SAFETENSORS_H

#include <stddef.h>
#include <stdint.h>

#include "ingot/dtype.h"

#ifdef __cplusplus
extern "C" {
#endif

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

#ifdef __cplusplus
}
#endif
#endif
