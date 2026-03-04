/*
 * qwen_tts_safetensors.c - Simple binary weights loader
 * Format: QWTS magic + length-prefixed tensor metadata + tensor data
 */

#include "qwen_tts_safetensors.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>

/* ========================================================================
 * Single file operations
 * ======================================================================== */

safetensors_file_t *safetensors_open(const char *path) {
    char bin_path[4096];
    
    /* Check if path already ends with weights.bin */
    if (strstr(path, "weights.bin")) {
        snprintf(bin_path, sizeof(bin_path), "%s", path);
    } else {
        /* Try model_dir/weights.bin first */
        snprintf(bin_path, sizeof(bin_path), "%s/weights.bin", path);
    }
    int fd = open(bin_path, O_RDONLY);
    if (fd < 0) {
        /* Try current directory */
        snprintf(bin_path, sizeof(bin_path), "weights.bin");
        fd = open(bin_path, O_RDONLY);
    }
    if (fd < 0) {
        fprintf(stderr, "Error: Cannot open weights.bin (tried %s and weights.bin)\n", path);
        return NULL;
    }
    
    /* Read header */
    char magic[4];
    if (read(fd, magic, 4) != 4 || memcmp(magic, "QWTS", 4) != 0) {
        close(fd); return NULL;
    }
    
    uint32_t version, num_tensors, metadata_size;
    if (read(fd, &version, 4) != 4 || version != 1) { close(fd); return NULL; }
    if (read(fd, &num_tensors, 4) != 4 || num_tensors > SAFETENSORS_MAX_TENSORS) {
        close(fd); return NULL;
    }
    if (read(fd, &metadata_size, 4) != 4) { close(fd); return NULL; }
    
    safetensors_file_t *sf = calloc(1, sizeof(safetensors_file_t));
    if (!sf) { close(fd); return NULL; }
    
    sf->path = strdup(path);
    sf->fd = fd;
    sf->num_tensors = num_tensors;
    
    /* Read tensor metadata */
    for (uint32_t i = 0; i < num_tensors; i++) {
        safetensor_t *t = &sf->tensors[i];
        
        /* name_len (2 bytes) */
        uint16_t name_len;
        if (read(fd, &name_len, 2) != 2) {
            fprintf(stderr, "[DEBUG] Failed to read name_len for tensor %u\n", i);
            safetensors_close(sf); return NULL;
        }
        if (name_len > 255) {
            fprintf(stderr, "[DEBUG] name_len too large (%u) for tensor %u\n", name_len, i);
            safetensors_close(sf); return NULL;
        }
        
        /* name (variable, NOT null-terminated) */
        memset(t->name, 0, sizeof(t->name));
        if (read(fd, t->name, name_len) != name_len) {
            fprintf(stderr, "[DEBUG] Failed to read name for tensor %u\n", i);
            safetensors_close(sf); return NULL;
        }
        t->name[name_len] = '\0';  /* Ensure null termination */
        
        /* dtype (1 byte) */
        uint8_t dtype_code;
        if (read(fd, &dtype_code, 1) != 1) { safetensors_close(sf); return NULL; }
        t->dtype = (safetensor_dtype_t)dtype_code;
        
        /* ndim (1 byte) */
        if (read(fd, &t->ndim, 1) != 1) { safetensors_close(sf); return NULL; }
        
        /* shape (8 x 4 bytes) */
        for (int d = 0; d < 8; d++) {
            if (read(fd, &t->shape[d], 4) != 4) { safetensors_close(sf); return NULL; }
        }
        
        /* data_offset (8 bytes) */
        if (read(fd, &t->data_offset, 8) != 8) { safetensors_close(sf); return NULL; }
        
        /* data_size (8 bytes) */
        if (read(fd, &t->data_size, 8) != 8) { safetensors_close(sf); return NULL; }
    }
    
    /* Cache: read entire data section into memory */
    size_t total_data_size = sf->tensors[num_tensors - 1].data_offset + sf->tensors[num_tensors - 1].data_size;
    /* Align to 8 bytes */
    if (total_data_size % 8 != 0) total_data_size += 8 - (total_data_size % 8);
    
    sf->data = malloc(total_data_size);
    if (!sf->data) { safetensors_close(sf); return NULL; }
    
    /* Read in chunks to avoid read() limits */
    size_t total_read = 0;
    const size_t chunk_size = 256 * 1024 * 1024;  /* 256 MB chunks */
    while (total_read < total_data_size) {
        size_t to_read = total_data_size - total_read;
        if (to_read > chunk_size) to_read = chunk_size;
        
        ssize_t bytes_read = read(fd, (char *)sf->data + total_read, to_read);
        if (bytes_read < 0 || bytes_read == 0) {
            free(sf->data);
            sf->data = NULL;
            safetensors_close(sf);
            return NULL;
        }
        total_read += bytes_read;
    }
    
    return sf;
}

void safetensors_close(safetensors_file_t *sf) {
    if (!sf) return;
    if (sf->fd >= 0) close(sf->fd);
    free(sf->data);
    free(sf->path);
    free(sf);
}

const void *safetensors_data(const safetensors_file_t *sf, const safetensor_t *t) {
    if (!sf || !sf->data || !t) return NULL;
    return (const char *)sf->data + t->data_offset;
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

    safetensors_file_t *sf = safetensors_open(model_dir);
    if (!sf) {
        free(ms);
        return NULL;
    }
    
    ms->shards[0] = sf;
    ms->num_shards = 1;
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
        /* Check tensor 397 specifically */
        if (sf->num_tensors > 397) {
            /* Use strncmp to avoid strlen issues */
            if (strncmp(sf->tensors[397].name, name, 256) == 0) {
                if (out_sf) *out_sf = sf;
                return &sf->tensors[397];
            }
        }
        /* Check all tensors */
        for (int i = 0; i < sf->num_tensors; i++) {
            if (strncmp(sf->tensors[i].name, name, 256) == 0) {
                if (out_sf) *out_sf = sf;
                return &sf->tensors[i];
            }
        }
    }
    return NULL;
}
