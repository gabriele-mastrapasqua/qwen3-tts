/* GGUF and safetensors writers.
 *
 * Containers only: which layers stay exact, which get quantized, whether a
 * LoRA is baked in — all of that is conversion policy and belongs to the
 * caller, not to a loader.
 *
 * SPDX-License-Identifier: MIT */
#define _POSIX_C_SOURCE 200809L   /* ftello/off_t under -std=c11 on glibc */
#include <sys/types.h>
#include "ingot/write.h"
#include "ingot/gguf.h"   /* the INGOT_KV_* tags */
#include "internal.h"

#ifndef INGOT_NO_KERNELS
#include "ingot/quant.h"
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
