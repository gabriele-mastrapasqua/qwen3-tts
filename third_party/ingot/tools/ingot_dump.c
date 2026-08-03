/* ingot-dump — print what is inside a GGUF or safetensors file.
 *
 * Detects the container from the magic, so `ingot-dump anything` works. Point
 * it at a directory and it resolves the shards the way ingot_st_open_dir does.
 *
 * SPDX-License-Identifier: MIT */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ingot/gguf.h"
#include "ingot/quant.h"
#include "ingot/safetensors.h"

static void print_shape(const uint64_t *dims, uint32_t rank) {
    printf("[");
    for (uint32_t i = 0; i < rank; i++)
        printf("%llu%s", (unsigned long long)dims[i], i + 1 < rank ? ", " : "");
    printf("]");
}

static void print_kv(const ingot_kv *kv) {
    printf("  %-40s ", ingot_kv_key(kv));
    const int type = ingot_kv_type(kv);
    if (type == INGOT_KV_STRING) {
        const char *s = NULL;
        ingot_kv_str(kv, &s);
        /* Chat templates are kilobytes; show enough to identify, not to flood. */
        if (s != NULL && strlen(s) > 72) printf("str  \"%.72s...\" (%zu bytes)\n", s, strlen(s));
        else printf("str  \"%s\"\n", s != NULL ? s : "");
    } else if (type == INGOT_KV_ARRAY) {
        uint64_t len = 0;
        ingot_kv_arr_len(kv, &len);
        printf("arr  %llu x type %d", (unsigned long long)len, ingot_kv_arr_type(kv));
        if (ingot_kv_arr_type(kv) == INGOT_KV_STRING && len > 0) {
            const char *s = NULL;
            size_t slen = 0;
            if (ingot_kv_arr_str(kv, 0, &s, &slen) == 0)
                printf("  first=\"%.*s\"", (int)(slen > 24 ? 24 : slen), s);
        }
        printf("\n");
    } else {
        double f = 0;
        int64_t i = 0;
        ingot_kv_f64(kv, &f);
        ingot_kv_i64(kv, &i);
        if (f == (double)i) printf("num  %lld\n", (long long)i);
        else printf("num  %g\n", f);
    }
}

static int dump_gguf(const char *path, int verbose) {
    char err[512] = {0};
    ingot_gguf *g = NULL;
    if (ingot_gguf_open_split(&g, path, err, sizeof err) != 0) {
        fprintf(stderr, "ingot-dump: %s\n", err);
        return 1;
    }
    printf("GGUF v%u  %s\n", ingot_gguf_version(g), path);
    printf("  architecture: %s\n", ingot_gguf_arch(g));
    printf("  alignment:    %llu\n", (unsigned long long)ingot_gguf_alignment(g));
    printf("  shards:       %u\n", ingot_gguf_shard_count(g));
    printf("  metadata:     %zu keys\n", ingot_gguf_kv_count(g));
    printf("  tensors:      %zu\n\n", ingot_gguf_count(g));

    if (verbose) {
        printf("metadata:\n");
        for (size_t i = 0; i < ingot_gguf_kv_count(g); i++) print_kv(ingot_gguf_kv_at(g, i));
        printf("\n");
    }

    /* A per-type census is usually what you actually opened the file for:
     * it tells you at a glance whether this is a Q4_K_M with a handful of
     * Q5_K/Q6_K layers, and whether anything in it is undecodable. */
    size_t counts[64] = {0};
    uint64_t bytes[64] = {0};
    uint64_t total = 0;
    for (size_t i = 0; i < ingot_gguf_count(g); i++) {
        const ingot_tensor *t = ingot_gguf_at(g, i);
        if (t->type >= 0 && t->type < 64) { counts[t->type]++; bytes[t->type] += t->nbytes; }
        total += t->nbytes;
        if (verbose) {
            printf("  %-48s %-8s ", t->name, ingot_type_name(t->type));
            print_shape(t->ne, t->rank);
            printf("  %llu B\n", (unsigned long long)t->nbytes);
        }
    }
    if (verbose) printf("\n");
    printf("type census:\n");
    for (int t = 0; t < 64; t++) {
        if (counts[t] == 0) continue;
        printf("  %-8s %5zu tensors  %8.2f MiB%s\n", ingot_type_name(t), counts[t],
               (double)bytes[t] / (1024.0 * 1024.0),
               ingot_type_can_dequant(t) ? "" : "   (ingot cannot dequantize this)");
    }
    printf("  %-8s %5zu tensors  %8.2f MiB\n", "TOTAL", ingot_gguf_count(g),
           (double)total / (1024.0 * 1024.0));
    ingot_gguf_close(g);
    return 0;
}

static int dump_st(const char *path, int verbose) {
    char err[512] = {0};
    ingot_st *st = NULL;
    if (ingot_st_open(&st, path, err, sizeof err) != 0) {
        fprintf(stderr, "ingot-dump: %s\n", err);
        return 1;
    }
    printf("safetensors  %s\n", path);
    printf("  shards:  %u\n", ingot_st_shard_count(st));
    for (uint32_t s = 0; s < ingot_st_shard_count(st); s++)
        printf("    [%u] %s\n", s, ingot_st_shard_path(st, s));
    printf("  tensors: %zu\n", ingot_st_count(st));
    const char *format = NULL;
    if (ingot_st_metadata(st, "format", &format) == 0) printf("  format:  %s\n", format);
    printf("\n");

    uint64_t total = 0;
    for (size_t i = 0; i < ingot_st_count(st); i++) {
        const ingot_st_tensor *t = ingot_st_at(st, i);
        total += t->nbytes;
        if (verbose) {
            printf("  %-56s %-8s ", t->name, ingot_dtype_name(t->dtype));
            print_shape(t->shape, t->rank);
            printf("  %llu B  (shard %u)\n", (unsigned long long)t->nbytes, t->shard);
        }
    }
    printf("%stotal: %.2f MiB\n", verbose ? "\n" : "", (double)total / (1024.0 * 1024.0));
    ingot_st_close(st);
    return 0;
}

int main(int argc, char **argv) {
    int verbose = 0;
    const char *path = NULL;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-v") == 0 || strcmp(argv[i], "--verbose") == 0) verbose = 1;
        else path = argv[i];
    }
    if (path == NULL) {
        fprintf(stderr,
                "usage: ingot-dump [-v] <file.gguf | file.safetensors | model-dir>\n"
                "  -v  list every tensor and metadata key, not just the summary\n");
        return 2;
    }
    /* Sniff the magic; a directory can only be safetensors. */
    unsigned char magic[4] = {0};
    FILE *f = fopen(path, "rb");
    if (f != NULL) {
        if (fread(magic, 1, 4, f) != 4) memset(magic, 0, sizeof magic);
        fclose(f);
    }
    return memcmp(magic, "GGUF", 4) == 0 ? dump_gguf(path, verbose)
                                         : dump_st(path, verbose);
}
