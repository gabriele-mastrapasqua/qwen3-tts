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
#ifndef INGOT_WRITE_H
#define INGOT_WRITE_H

#include <stddef.h>
#include <stdint.h>

#include "ingot/dtype.h"

#ifdef __cplusplus
extern "C" {
#endif

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
#endif
