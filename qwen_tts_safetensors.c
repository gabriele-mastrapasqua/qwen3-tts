/*
 * qwen_tts_safetensors.c - Standard HuggingFace safetensors reader with mmap
 * Format: 8-byte header_size (uint64 LE) + JSON header + raw tensor data
 */

#include "qwen_tts_safetensors.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <dirent.h>

/* ========================================================================
 * Minimal JSON parser for safetensors header
 * ======================================================================== */

static void skip_whitespace(const char **p) {
    while (**p == ' ' || **p == '\n' || **p == '\r' || **p == '\t') (*p)++;
}

static int parse_string(const char **p, char *out, size_t max_len) {
    skip_whitespace(p);
    if (**p != '"') return -1;
    (*p)++;
    size_t i = 0;
    while (**p && **p != '"' && i < max_len - 1) {
        if (**p == '\\') {
            (*p)++;
            if (**p == 'n') out[i++] = '\n';
            else if (**p == 't') out[i++] = '\t';
            else if (**p == '"') out[i++] = '"';
            else if (**p == '\\') out[i++] = '\\';
            else out[i++] = **p;
        } else {
            out[i++] = **p;
        }
        (*p)++;
    }
    out[i] = '\0';
    if (**p != '"') return -1;
    (*p)++;
    return 0;
}

static int64_t parse_int(const char **p) {
    skip_whitespace(p);
    int64_t val = 0;
    int neg = 0;
    if (**p == '-') { neg = 1; (*p)++; }
    while (**p >= '0' && **p <= '9') {
        val = val * 10 + (**p - '0');
        (*p)++;
    }
    return neg ? -val : val;
}

static safetensor_dtype_t parse_dtype(const char *s) {
    if (strcmp(s, "F32") == 0) return DTYPE_F32;
    if (strcmp(s, "F16") == 0) return DTYPE_F16;
    if (strcmp(s, "BF16") == 0) return DTYPE_BF16;
    if (strcmp(s, "I32") == 0) return DTYPE_I32;
    if (strcmp(s, "I64") == 0) return DTYPE_I64;
    if (strcmp(s, "BOOL") == 0) return DTYPE_BOOL;
    return DTYPE_UNKNOWN;
}

static int parse_tensor_entry(const char **p, safetensor_t *t) {
    skip_whitespace(p);
    if (**p != '{') return -1;
    (*p)++;

    t->dtype = DTYPE_UNKNOWN;
    t->ndim = 0;
    t->data_offset = 0;
    t->data_size = 0;

    while (**p && **p != '}') {
        skip_whitespace(p);
        if (**p == ',') { (*p)++; continue; }

        char key[64];
        if (parse_string(p, key, sizeof(key)) != 0) return -1;
        skip_whitespace(p);
        if (**p != ':') return -1;
        (*p)++;
        skip_whitespace(p);

        if (strcmp(key, "dtype") == 0) {
            char dtype_str[32];
            if (parse_string(p, dtype_str, sizeof(dtype_str)) != 0) return -1;
            t->dtype = parse_dtype(dtype_str);
        } else if (strcmp(key, "shape") == 0) {
            if (**p != '[') return -1;
            (*p)++;
            t->ndim = 0;
            while (**p && **p != ']' && t->ndim < 8) {
                skip_whitespace(p);
                if (**p == ',') { (*p)++; continue; }
                t->shape[t->ndim++] = parse_int(p);
            }
            if (**p == ']') (*p)++;
        } else if (strcmp(key, "data_offsets") == 0) {
            if (**p != '[') return -1;
            (*p)++;
            skip_whitespace(p);
            size_t start = (size_t)parse_int(p);
            skip_whitespace(p);
            if (**p == ',') (*p)++;
            skip_whitespace(p);
            size_t end = (size_t)parse_int(p);
            t->data_offset = start;
            t->data_size = end - start;
            skip_whitespace(p);
            if (**p == ']') (*p)++;
        } else {
            /* Skip unknown value */
            if (**p == '"') {
                (*p)++;
                while (**p && **p != '"') {
                    if (**p == '\\') (*p)++;
                    if (**p) (*p)++;
                }
                if (**p == '"') (*p)++;
            } else if (**p == '[') {
                int depth = 1; (*p)++;
                while (**p && depth > 0) {
                    if (**p == '[') depth++;
                    else if (**p == ']') depth--;
                    (*p)++;
                }
            } else if (**p == '{') {
                int depth = 1; (*p)++;
                while (**p && depth > 0) {
                    if (**p == '{') depth++;
                    else if (**p == '}') depth--;
                    (*p)++;
                }
            } else {
                while (**p && **p != ',' && **p != '}') (*p)++;
            }
        }
    }
    if (**p == '}') (*p)++;
    return 0;
}

static int parse_header(safetensors_file_t *sf) {
    const char *p = sf->header_json;
    skip_whitespace(&p);
    if (*p != '{') return -1;
    p++;

    sf->num_tensors = 0;

    while (*p && *p != '}' && sf->num_tensors < SAFETENSORS_MAX_TENSORS) {
        skip_whitespace(&p);
        if (*p == ',') { p++; continue; }
        if (*p == '}') break;

        char name[256];
        if (parse_string(&p, name, sizeof(name)) != 0) return -1;
        skip_whitespace(&p);
        if (*p != ':') return -1;
        p++;

        if (strcmp(name, "__metadata__") == 0) {
            skip_whitespace(&p);
            if (*p == '{') {
                int depth = 1; p++;
                while (*p && depth > 0) {
                    if (*p == '{') depth++;
                    else if (*p == '}') depth--;
                    p++;
                }
            }
            continue;
        }

        safetensor_t *t = &sf->tensors[sf->num_tensors];
        snprintf(t->name, sizeof(t->name), "%s", name);
        if (parse_tensor_entry(&p, t) != 0) return -1;
        sf->num_tensors++;
    }
    return 0;
}

/* ========================================================================
 * Single file operations
 * ======================================================================== */

safetensors_file_t *safetensors_open(const char *path) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) return NULL;

    struct stat st;
    if (fstat(fd, &st) < 0) { close(fd); return NULL; }

    size_t file_size = (size_t)st.st_size;
    if (file_size < 8) { close(fd); return NULL; }

    void *data = mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (data == MAP_FAILED) return NULL;

    uint64_t header_size = 0;
    memcpy(&header_size, data, 8);
    if (header_size > file_size - 8) { munmap(data, file_size); return NULL; }

    safetensors_file_t *sf = calloc(1, sizeof(safetensors_file_t));
    if (!sf) { munmap(data, file_size); return NULL; }

    sf->path = strdup(path);
    sf->data = data;
    sf->file_size = file_size;
    sf->header_size = (size_t)header_size;

    sf->header_json = malloc(header_size + 1);
    if (!sf->header_json) { safetensors_close(sf); return NULL; }
    memcpy(sf->header_json, (char *)data + 8, header_size);
    sf->header_json[header_size] = '\0';

    if (parse_header(sf) != 0) { safetensors_close(sf); return NULL; }

    return sf;
}

void safetensors_close(safetensors_file_t *sf) {
    if (!sf) return;
    if (sf->data) munmap(sf->data, sf->file_size);
    free(sf->path);
    free(sf->header_json);
    free(sf);
}

const void *safetensors_data(const safetensors_file_t *sf, const safetensor_t *t) {
    return (const char *)sf->data + 8 + sf->header_size + t->data_offset;
}

int64_t safetensor_numel(const safetensor_t *t) {
    int64_t n = 1;
    for (int i = 0; i < t->ndim; i++) n *= t->shape[i];
    return n;
}

static float bf16_to_f32(uint16_t bf16) {
    uint32_t f32 = ((uint32_t)bf16) << 16;
    float result;
    memcpy(&result, &f32, sizeof(float));
    return result;
}

float *safetensors_get_f32(const safetensors_file_t *sf, const safetensor_t *t) {
    int64_t n = safetensor_numel(t);
    if (n <= 0) return NULL;

    float *out = malloc(n * sizeof(float));
    if (!out) return NULL;

    const void *data = safetensors_data(sf, t);
    if (!data) { free(out); return NULL; }

    switch (t->dtype) {
        case DTYPE_F32:
            memcpy(out, data, n * sizeof(float));
            break;
        case DTYPE_BF16: {
            const uint16_t *src = (const uint16_t *)data;
            for (int64_t i = 0; i < n; i++) out[i] = bf16_to_f32(src[i]);
            break;
        }
        default:
            free(out);
            return NULL;
    }
    return out;
}

int safetensor_is_bf16(const safetensor_t *t) {
    return t && t->dtype == DTYPE_BF16;
}

uint16_t *safetensors_get_bf16_direct(const safetensors_file_t *sf, const safetensor_t *t) {
    if (!sf || !t || t->dtype != DTYPE_BF16) return NULL;
    return (uint16_t *)safetensors_data(sf, t);
}

void safetensor_print(const safetensor_t *t) {
    const char *dtype_names[] = {"F32", "F16", "BF16", "I32", "I64", "BOOL"};
    const char *dtype_name = t->dtype >= 0 && t->dtype <= 5 ?
                             dtype_names[t->dtype] : "UNKNOWN";
    printf("%s: dtype=%s, shape=[", t->name, dtype_name);
    for (int i = 0; i < t->ndim; i++) {
        printf("%ld%s", (long)t->shape[i], i < t->ndim - 1 ? ", " : "");
    }
    printf("]\n");
}

void safetensors_print_all(const safetensors_file_t *sf) {
    printf("File: %s (%d tensors)\n", sf->path, sf->num_tensors);
    for (int i = 0; i < sf->num_tensors; i++) safetensor_print(&sf->tensors[i]);
}

/* ========================================================================
 * Multi-shard operations
 * ======================================================================== */

multi_safetensors_t *multi_safetensors_open(const char *model_dir) {
    multi_safetensors_t *ms = calloc(1, sizeof(multi_safetensors_t));
    if (!ms) return NULL;

    char path[4096];

    /* Try single file first */
    snprintf(path, sizeof(path), "%s/model.safetensors", model_dir);
    safetensors_file_t *sf = safetensors_open(path);
    if (sf) {
        ms->shards[0] = sf;
        ms->num_shards = 1;
        return ms;
    }

    /* Scan directory for shard files (model-00001-of-NNNNN.safetensors) */
    DIR *dir = opendir(model_dir);
    if (!dir) { free(ms); return NULL; }

    struct dirent *entry;
    char shard_names[SAFETENSORS_MAX_SHARDS][256];
    int n_shards = 0;

    while ((entry = readdir(dir)) != NULL && n_shards < SAFETENSORS_MAX_SHARDS) {
        if (strncmp(entry->d_name, "model-", 6) == 0 &&
            strstr(entry->d_name, ".safetensors") != NULL) {
            snprintf(shard_names[n_shards], sizeof(shard_names[n_shards]),
                     "%s", entry->d_name);
            n_shards++;
        }
    }
    closedir(dir);

    if (n_shards == 0) {
        fprintf(stderr, "multi_safetensors_open: no safetensors files in %s\n", model_dir);
        free(ms);
        return NULL;
    }

    /* Sort shard names to ensure consistent ordering */
    qsort(shard_names, n_shards, 256, (int(*)(const void*,const void*))strcmp);

    for (int i = 0; i < n_shards; i++) {
        snprintf(path, sizeof(path), "%s/%s", model_dir, shard_names[i]);
        ms->shards[i] = safetensors_open(path);
        if (!ms->shards[i]) {
            fprintf(stderr, "multi_safetensors_open: failed to open %s\n", path);
            multi_safetensors_close(ms);
            return NULL;
        }
    }
    ms->num_shards = n_shards;
    return ms;
}

void multi_safetensors_close(multi_safetensors_t *ms) {
    if (!ms) return;
    for (int i = 0; i < ms->num_shards; i++) {
        safetensors_close(ms->shards[i]);
    }
    free(ms);
}

const safetensor_t *multi_safetensors_find(const multi_safetensors_t *ms,
                                            const char *name,
                                            safetensors_file_t **out_sf) {
    for (int s = 0; s < ms->num_shards; s++) {
        safetensors_file_t *sf = ms->shards[s];
        for (int i = 0; i < sf->num_tensors; i++) {
            if (strcmp(sf->tensors[i].name, name) == 0) {
                if (out_sf) *out_sf = sf;
                return &sf->tensors[i];
            }
        }
    }
    return NULL;
}
