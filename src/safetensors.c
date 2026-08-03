/* safetensors reader.
 *
 * The JSON is hand-rolled so there is no cJSON to vendor. Every tensor is
 * validated exactly (`elements * dtype_size == data_size`), the shard bundle
 * refuses duplicate names instead of letting the first one win, and nothing
 * here has a fixed cap: a reader with a 1024-tensor array loads half a model
 * and says nothing, which is the failure this deliberately does not have.
 *
 * SPDX-License-Identifier: MIT */
#define _POSIX_C_SOURCE 200809L
#if defined(__APPLE__)
/* madvise/MADV_DONTNEED are outside the strict POSIX namespace on Darwin. */
#define _DARWIN_C_SOURCE 1
#endif

#include "ingot/safetensors.h"
#include "internal.h"

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
