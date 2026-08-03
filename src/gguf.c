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
#define _POSIX_C_SOURCE 200809L

#include "ingot/gguf.h"
#include "internal.h"

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
