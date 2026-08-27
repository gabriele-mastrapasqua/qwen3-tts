/* Mutation fuzz: write a valid file, flip a few bytes, open it, and touch
 * everything a consumer would touch.
 *
 * The hand-written malformed fixtures in test_gguf.c and test_safetensors.c
 * break a file in the ways a person thinks of. This breaks it in the ways
 * nobody thinks of, and the interesting half is not the rejections: it is the
 * ~15% of mutants that still parse, because those exercise the accept path
 * with values no valid writer would ever produce.
 *
 * Not part of `make test` (it is a loop, not an assertion). Run it with
 * `make fuzz`, and under the memory checker with `make fuzz-leaks`.
 *
 * SPDX-License-Identifier: MIT */
#define _POSIX_C_SOURCE 200809L /* mkdtemp under -std=c11 on glibc */
#define _DARWIN_C_SOURCE        /* macOS: keep the full namespace alongside it */
#include <ingot/gguf.h>
#include <ingot/safetensors.h>
#include <ingot/wfile.h>
#include <ingot/write.h>
#include <ingot/quant.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
static uint32_t rng = 1;
static uint32_t nx(void){ rng = rng*1664525u+1013904223u; return rng>>8; }
static char dir[128];
static int opened, rejected;

static void build(const char *path, int gguf) {
    if (gguf) {
        ingot_gguf_writer *w = ingot_gguf_writer_new();
        float v[256]; for (int i=0;i<256;i++) v[i]=(float)i*0.01f;
        const char *vocab[] = {"a","bb","ccc"};
        const uint64_t ne1[2]={256,1}, ne2[2]={8,4};
        ingot_gguf_kv_string(w,"general.architecture","fuzz");
        ingot_gguf_kv_array_string(w,"tokenizer.ggml.tokens",vocab,3);
        ingot_gguf_kv_u32(w,"fuzz.n",7);
        ingot_gguf_add_f32(w,"q4k.weight",INGOT_TYPE_Q4_K,2,ne1,v);
        ingot_gguf_add_tensor(w,"f32.weight",INGOT_TYPE_F32,2,ne2,v);
        char e[256]; ingot_gguf_writer_save(w,path,e,sizeof e);
        ingot_gguf_writer_free(w);
    } else {
        ingot_st_writer *w = ingot_st_writer_new();
        float v[24]; for (int i=0;i<24;i++) v[i]=(float)i;
        const uint64_t s[2]={2,3}, s2[1]={4};
        ingot_st_writer_metadata(w,"format","pt");
        ingot_st_writer_add(w,"a.weight",INGOT_DT_F32,2,s,v);
        ingot_st_writer_add(w,"b.weight",INGOT_DT_BF16,1,s2,v);
        char e[256]; ingot_st_writer_save(w,path,e,sizeof e);
        ingot_st_writer_free(w);
    }
}

static void exercise(const char *path) {
    char err[256];
    ingot_wfile *w = NULL;
    if (ingot_wfile_open(&w, path, err, sizeof err) != 0) { rejected++; return; }
    opened++;
    /* touch everything a consumer would touch */
    for (size_t i = 0; i < ingot_wfile_count(w); i++) {
        const ingot_wtensor *t = ingot_wfile_at(w, i);
        if (t == NULL || t->data == NULL) continue;
        (void)ingot_wfile_find(w, t->name);
        if (t->nelem > 0 && t->nelem < (1u<<20)) {
            float *f = malloc((size_t)t->nelem * sizeof(float));
            if (f) { (void)ingot_wfile_to_f32(w, t, f); free(f); }
        }
    }
    const ingot_gguf *g = ingot_wfile_gguf(w);
    if (g != NULL) {
        for (size_t i = 0; i < ingot_gguf_kv_count(g); i++) {
            const ingot_kv *kv = ingot_gguf_kv_at(g, i);
            const char *s; uint64_t u, n; double f; int64_t v; size_t l;
            (void)ingot_kv_str(kv,&s); (void)ingot_kv_u64(kv,&u);
            (void)ingot_kv_f64(kv,&f); (void)ingot_kv_i64(kv,&v);
            if (ingot_kv_arr_len(kv,&n)==0) for (uint64_t k=0;k<n && k<64;k++) {
                (void)ingot_kv_arr_str(kv,k,&s,&l); (void)ingot_kv_arr_f32(kv,k,(float*)&f);
            }
        }
    }
    ingot_wfile_close(w);
}

int main(int argc, char **argv) {
    const int rounds = argc > 1 ? atoi(argv[1]) : 4000;
    strcpy(dir, "/tmp/ingot_fuzz_XXXXXX");
    if (!mkdtemp(dir)) return 1;
    char good[2][256], mutated[256];
    snprintf(good[0], sizeof good[0], "%s/g.gguf", dir);
    snprintf(good[1], sizeof good[1], "%s/g.safetensors", dir);
    snprintf(mutated, sizeof mutated, "%s/m.bin", dir);
    build(good[0], 1); build(good[1], 0);
    for (int r = 0; r < rounds; r++) {
        const char *src = good[r & 1];
        FILE *in = fopen(src, "rb");
        fseek(in, 0, SEEK_END); long n = ftell(in); fseek(in, 0, SEEK_SET);
        unsigned char *buf = malloc((size_t)n);
        if (fread(buf, 1, (size_t)n, in) != (size_t)n) return 1;
        fclose(in);
        const int flips = 1 + (int)(nx() % 8);
        for (int k = 0; k < flips; k++) buf[nx() % (uint32_t)n] = (unsigned char)nx();
        /* the right suffix so wfile picks the right container */
        char path[300];
        snprintf(path, sizeof path, "%s/m%s", dir, (r & 1) ? ".safetensors" : ".gguf");
        FILE *out = fopen(path, "wb"); fwrite(buf, 1, (size_t)n, out); fclose(out);
        free(buf);
        exercise(path);
        unlink(path);
    }
    unlink(good[0]); unlink(good[1]); rmdir(dir);
    printf("%d rounds: %d opened, %d rejected — no crash\n", rounds, opened, rejected);
    return 0;
}
