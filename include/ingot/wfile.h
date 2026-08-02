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
#ifndef INGOT_WFILE_H
#define INGOT_WFILE_H

#include <stddef.h>
#include <stdint.h>

#include "ingot/dtype.h"
#include "ingot/gguf.h"
#include "ingot/safetensors.h"

#ifdef __cplusplus
extern "C" {
#endif

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

#ifdef __cplusplus
}
#endif
#endif
