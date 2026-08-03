/* Writers, checked by round trip: build a file, read it back with this
 * library's own reader, and require every field to survive.
 *
 * A round trip through one implementation cannot prove the format is right —
 * a writer and a reader that share a misunderstanding agree perfectly. What it
 * DOES prove is that the two halves are consistent, that padding and alignment
 * work out, and that the reader's corner cases can be exercised without a
 * model on disk. For "is this really GGUF", see tools/check_against_python.py
 * and the block layouts cross-checked in test_formats.c.
 *
 * SPDX-License-Identifier: MIT */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "ingot/gguf.h"
#include "ingot/quant.h"
#include "ingot/safetensors.h"
#include "ingot/write.h"

static int failures;
static int checks;

#define CHECK(cond, ...) do {                                        \
    checks++;                                                        \
    if (!(cond)) { printf("  FAIL: "); printf(__VA_ARGS__);          \
                   printf("  (%s:%d)\n", __FILE__, __LINE__);        \
                   failures++; }                                     \
    else { printf("  ok:   "); printf(__VA_ARGS__); printf("\n"); }  \
} while (0)

static char dir[128];
static char *in_tmp(const char *name) {
    static char path[256];
    snprintf(path, sizeof path, "%s/%s", dir, name);
    return path;
}

static void test_gguf_roundtrip(void) {
    printf("GGUF write -> read\n");
    char err[256] = {0};
    ingot_gguf_writer *w = ingot_gguf_writer_new();
    CHECK(w != NULL, "writer allocates");
    if (w == NULL) return;

    static const char *vocab[] = { "<pad>", "hello", "world" };
    const float scores[3] = { 0.5f, -1.25f, 3.0f };
    const int32_t types[3] = { 1, 1, 2 };

    int ok = ingot_gguf_kv_string(w, "general.architecture", "roundtrip") == 0;
    ok &= ingot_gguf_kv_string(w, "general.name", "a \"quoted\" name") == 0;
    ok &= ingot_gguf_kv_u32(w, "rt.head_count", 32) == 0;
    ok &= ingot_gguf_kv_u64(w, "rt.big", 1ull << 40) == 0;
    ok &= ingot_gguf_kv_i32(w, "rt.negative", -7) == 0;
    ok &= ingot_gguf_kv_f32(w, "rt.freq_base", 10000.0f) == 0;
    ok &= ingot_gguf_kv_bool(w, "rt.flag", 1) == 0;
    ok &= ingot_gguf_kv_array_string(w, "tokenizer.ggml.tokens", vocab, 3) == 0;
    ok &= ingot_gguf_kv_array_f32(w, "tokenizer.ggml.scores", scores, 3) == 0;
    ok &= ingot_gguf_kv_array_i32(w, "tokenizer.ggml.token_type", types, 3) == 0;
    CHECK(ok, "every metadata kind accepted");

    /* One f32 tensor stored verbatim, one quantized on the way in. ne is in
     * GGML order, so [256, 4] means 4 rows of 256. */
    float raw[64];
    for (int i = 0; i < 64; i++) raw[i] = (float)i * 0.25f - 8.0f;
    float wide[4 * 256];
    for (int i = 0; i < 4 * 256; i++) wide[i] = sinf((float)i * 0.013f) * 0.05f;

    const uint64_t ne_raw[2] = { 16, 4 };
    const uint64_t ne_wide[2] = { 256, 4 };
    ok = ingot_gguf_add_tensor(w, "raw.weight", INGOT_TYPE_F32, 2, ne_raw, raw) == 0;
    ok &= ingot_gguf_add_f32(w, "q4k.weight", INGOT_TYPE_Q4_K, 2, ne_wide, wide) == 0;
    ok &= ingot_gguf_add_f32(w, "q6k.weight", INGOT_TYPE_Q6_K, 2, ne_wide, wide) == 0;
    ok &= ingot_gguf_add_f32(w, "q80.weight", INGOT_TYPE_Q8_0, 2, ne_wide, wide) == 0;
    CHECK(ok, "tensors accepted, three of them quantized on the way in");

    /* A shape whose fastest dimension is not a whole number of blocks must be
     * refused at add time, not discovered by the reader later. */
    const uint64_t ne_bad[2] = { 100, 4 };
    CHECK(ingot_gguf_add_f32(w, "bad.weight", INGOT_TYPE_Q4_K, 2, ne_bad, wide) != 0,
          "a row that is not a whole number of blocks is refused");

    const char *path = in_tmp("rt.gguf");
    CHECK(ingot_gguf_writer_save(w, path, err, sizeof err) == 0, "save (%s)", err);
    ingot_gguf_writer_free(w);

    ingot_gguf *g = NULL;
    CHECK(ingot_gguf_open(&g, path, err, sizeof err) == 0 && g != NULL,
          "reads back (%s)", err);
    if (g == NULL) return;

    CHECK(ingot_gguf_count(g) == 4, "4 tensors");
    CHECK(strcmp(ingot_gguf_arch(g), "roundtrip") == 0, "architecture survives");
    CHECK(ingot_gguf_alignment(g) == 32, "alignment written and read");

    const ingot_kv *kv = ingot_gguf_kv_find(g, "general.name");
    const char *s = NULL;
    CHECK(kv != NULL && ingot_kv_str(kv, &s) == 0 && strcmp(s, "a \"quoted\" name") == 0,
          "a string with quotes survives");
    uint64_t u = 0;
    kv = ingot_gguf_kv_find(g, "rt.big");
    CHECK(kv != NULL && ingot_kv_u64(kv, &u) == 0 && u == (1ull << 40), "u64 survives");
    int64_t i64 = 0;
    kv = ingot_gguf_kv_find(g, "rt.negative");
    CHECK(kv != NULL && ingot_kv_i64(kv, &i64) == 0 && i64 == -7, "a negative i32 survives");
    int flag = 0;
    kv = ingot_gguf_kv_find(g, "rt.flag");
    CHECK(kv != NULL && ingot_kv_bool(kv, &flag) == 0 && flag == 1, "bool survives");

    kv = ingot_gguf_kv_find(g, "tokenizer.ggml.tokens");
    uint64_t len = 0;
    size_t slen = 0;
    CHECK(kv != NULL && ingot_kv_arr_len(kv, &len) == 0 && len == 3 &&
          ingot_kv_arr_str(kv, 1, &s, &slen) == 0 && slen == 5 &&
          memcmp(s, "hello", 5) == 0, "a string array survives, indexable");
    kv = ingot_gguf_kv_find(g, "tokenizer.ggml.scores");
    float score = 0;
    CHECK(kv != NULL && ingot_kv_arr_f32(kv, 1, &score) == 0 && score == -1.25f,
          "an f32 array survives");
    kv = ingot_gguf_kv_find(g, "tokenizer.ggml.token_type");
    CHECK(kv != NULL && ingot_kv_arr_i64(kv, 2, &i64) == 0 && i64 == 2,
          "an i32 array survives");

    const ingot_tensor *t = ingot_gguf_find(g, "raw.weight");
    CHECK(t != NULL && t->type == INGOT_TYPE_F32 && t->ne[0] == 16 && t->ne[1] == 4 &&
          memcmp(ingot_gguf_data(g, t), raw, sizeof raw) == 0,
          "an f32 tensor comes back bit-exact");

    /* The quantized ones come back within their format's error, which also
     * proves the payload landed at the offset the table claims. */
    static const struct { const char *name; double budget; } q[] = {
        { "q4k.weight", 0.09 }, { "q6k.weight", 0.03 }, { "q80.weight", 0.01 },
    };
    for (size_t k = 0; k < 3; k++) {
        const ingot_tensor *qt = ingot_gguf_find(g, q[k].name);
        if (qt == NULL) { CHECK(0, "%s present", q[k].name); continue; }
        float *out = malloc((size_t)qt->nelem * sizeof(float));
        double err2 = 0, ref2 = 0;
        const int rc = ingot_gguf_dequant(g, qt, out) == 0;
        for (uint64_t i = 0; i < qt->nelem && rc; i++) {
            const double d = (double)out[i] - (double)wide[i];
            err2 += d * d;
            ref2 += (double)wide[i] * (double)wide[i];
        }
        const double rel = (rc && ref2 > 0) ? sqrt(err2 / ref2) : 1e9;
        CHECK(rel < q[k].budget, "%s round-trips, rel_l2 %.4f", q[k].name, rel);
        free(out);
    }

    /* And the whole point of writing GGUF: the kernels read it in place. */
    const ingot_tensor *qt = ingot_gguf_find(g, "q4k.weight");
    if (qt != NULL) {
        float x[256], y[4] = {0}, deq[4 * 256];
        for (int i = 0; i < 256; i++) x[i] = sinf((float)i * 0.07f);
        const int rc = ingot_gguf_matvec(g, qt, x, y) == 0 &&
                       ingot_gguf_dequant_matrix(g, qt, deq) == 0;
        double worst = 0;
        for (int r = 0; r < 4 && rc; r++) {
            double manual = 0;
            for (int i = 0; i < 256; i++) manual += (double)deq[r * 256 + i] * (double)x[i];
            const double d = fabs((double)y[r] - manual) / fmax(1.0, fabs(manual));
            if (d > worst) worst = d;
        }
        CHECK(rc && worst < 1e-4,
              "ingot_gguf_matvec runs straight off the file we just wrote (rel %.1e)", worst);
    }

    ingot_gguf_close(g);
    unlink(path);
}

static void test_st_roundtrip(void) {
    printf("safetensors write -> read\n");
    char err[256] = {0};
    ingot_st_writer *w = ingot_st_writer_new();
    CHECK(w != NULL, "writer allocates");
    if (w == NULL) return;

    float f32[6] = { 1, -2.5f, 3.25f, 0, 7, -0.5f };
    uint16_t bf16[4] = { 0x3f80, 0x4000, 0xbf80, 0x3f00 };
    int64_t ids[2] = { 7, -3 };
    const uint64_t s23[2] = { 2, 3 }, s4[1] = { 4 }, s2[1] = { 2 };

    int ok = ingot_st_writer_metadata(w, "format", "pt") == 0;
    ok &= ingot_st_writer_metadata(w, "note", "a\nnewline and a \"quote\"") == 0;
    ok &= ingot_st_writer_add(w, "a.weight", INGOT_DT_F32, 2, s23, f32) == 0;
    ok &= ingot_st_writer_add(w, "b.weight", INGOT_DT_BF16, 1, s4, bf16) == 0;
    ok &= ingot_st_writer_add(w, "c.ids", INGOT_DT_I64, 1, s2, ids) == 0;
    CHECK(ok, "metadata and three dtypes accepted");
    CHECK(ingot_st_writer_add(w, "bad", INGOT_DT_UNKNOWN, 1, s2, ids) != 0,
          "an unknown dtype is refused at add time");

    const char *path = in_tmp("rt.safetensors");
    CHECK(ingot_st_writer_save(w, path, err, sizeof err) == 0, "save (%s)", err);
    ingot_st_writer_free(w);

    ingot_st *st = NULL;
    CHECK(ingot_st_open(&st, path, err, sizeof err) == 0 && st != NULL,
          "reads back (%s)", err);
    if (st == NULL) return;

    CHECK(ingot_st_count(st) == 3, "3 tensors");
    const char *meta = NULL;
    CHECK(ingot_st_metadata(st, "format", &meta) == 0 && strcmp(meta, "pt") == 0,
          "metadata survives");
    CHECK(ingot_st_metadata(st, "note", &meta) == 0 &&
          strcmp(meta, "a\nnewline and a \"quote\"") == 0,
          "escapes survive the JSON round trip");

    const ingot_st_tensor *a = ingot_st_find(st, "a.weight");
    CHECK(a != NULL && a->dtype == INGOT_DT_F32 && a->rank == 2 &&
          a->shape[0] == 2 && a->shape[1] == 3 && a->nbytes == 24 &&
          memcmp(ingot_st_data(st, a), f32, sizeof f32) == 0,
          "F32 [2,3] comes back bit-exact");
    const ingot_st_tensor *b = ingot_st_find(st, "b.weight");
    float back[4] = {0};
    CHECK(b != NULL && b->dtype == INGOT_DT_BF16 &&
          ingot_st_to_f32(st, b, back) == 0 && back[0] == 1.0f && back[2] == -1.0f,
          "BF16 survives and converts");
    const ingot_st_tensor *c = ingot_st_find(st, "c.ids");
    CHECK(c != NULL && c->dtype == INGOT_DT_I64 &&
          memcmp(ingot_st_data(st, c), ids, sizeof ids) == 0, "I64 comes back bit-exact");

    ingot_st_close(st);
    unlink(path);
}

/* A file written by the writer, opened as a split shard set of one, through
 * the container-agnostic layer: three subsystems agreeing on one artefact. */
static void test_cross_layer(void) {
    printf("writer -> wfile\n");
    char err[256] = {0};
    ingot_gguf_writer *w = ingot_gguf_writer_new();
    if (w == NULL) { CHECK(0, "writer"); return; }
    float values[32];
    for (int i = 0; i < 32; i++) values[i] = (float)i;
    const uint64_t ne[2] = { 8, 4 };
    ingot_gguf_kv_string(w, "general.architecture", "x");
    ingot_gguf_add_tensor(w, "m.weight", INGOT_TYPE_F32, 2, ne, values);
    const char *path = in_tmp("cross.gguf");
    CHECK(ingot_gguf_writer_save(w, path, err, sizeof err) == 0, "save (%s)", err);
    ingot_gguf_writer_free(w);

    ingot_gguf *g = NULL;
    CHECK(ingot_gguf_open_split(&g, path, err, sizeof err) == 0 && g != NULL,
          "open_split accepts a name without the shard suffix");
    if (g != NULL) {
        CHECK(ingot_gguf_shard_count(g) == 1, "one shard");
        CHECK(ingot_gguf_data_base(g, 0) % 32 == 0, "the data base is aligned");
        ingot_gguf_close(g);
    }
    unlink(path);
}

/* Split files: written as two real shards and reopened through the naming
 * convention. This path had an implementation and no test — the shard-name
 * arithmetic is exactly the kind of code that is wrong by one character. */
static void test_split(void) {
    printf("split GGUF: two shards, one handle\n");
    char err[256] = {0};
    float a_values[32], b_values[32];
    for (int i = 0; i < 32; i++) { a_values[i] = (float)i; b_values[i] = -(float)i; }
    const uint64_t ne[2] = { 8, 4 };

    char first[256], second[256];
    snprintf(first, sizeof first, "%s/m-00001-of-00002.gguf", dir);
    snprintf(second, sizeof second, "%s/m-00002-of-00002.gguf", dir);

    ingot_gguf_writer *w = ingot_gguf_writer_new();
    ingot_gguf_kv_string(w, "general.architecture", "split");
    ingot_gguf_kv_u32(w, "split.count", 2);
    ingot_gguf_add_tensor(w, "a.weight", INGOT_TYPE_F32, 2, ne, a_values);
    CHECK(ingot_gguf_writer_save(w, first, err, sizeof err) == 0, "shard 1 written (%s)", err);
    ingot_gguf_writer_free(w);

    w = ingot_gguf_writer_new();
    ingot_gguf_kv_string(w, "general.architecture", "split");
    ingot_gguf_kv_u32(w, "split.count", 2);
    ingot_gguf_add_tensor(w, "b.weight", INGOT_TYPE_F32, 2, ne, b_values);
    CHECK(ingot_gguf_writer_save(w, second, err, sizeof err) == 0, "shard 2 written (%s)", err);
    ingot_gguf_writer_free(w);

    /* Either shard name must resolve the whole set. */
    for (int from = 0; from < 2; from++) {
        ingot_gguf *g = NULL;
        const char *path = from == 0 ? first : second;
        if (ingot_gguf_open_split(&g, path, err, sizeof err) != 0 || g == NULL) {
            CHECK(0, "open_split from shard %d (%s)", from + 1, err);
            continue;
        }
        const ingot_tensor *ta = ingot_gguf_find(g, "a.weight");
        const ingot_tensor *tb = ingot_gguf_find(g, "b.weight");
        const int ok = ingot_gguf_shard_count(g) == 2 && ingot_gguf_count(g) == 2 &&
                       ta != NULL && tb != NULL && ta->shard == 0 && tb->shard == 1 &&
                       memcmp(ingot_gguf_data(g, ta), a_values, sizeof a_values) == 0 &&
                       memcmp(ingot_gguf_data(g, tb), b_values, sizeof b_values) == 0;
        CHECK(ok, "open_split from shard %d sees both, with the right payloads", from + 1);
        ingot_gguf_close(g);
    }

    /* And plain open() must see only the one file it was given. */
    ingot_gguf *single = NULL;
    CHECK(ingot_gguf_open(&single, first, err, sizeof err) == 0 && single != NULL &&
          ingot_gguf_count(single) == 1 && ingot_gguf_find(single, "b.weight") == NULL,
          "plain open() stays on one shard");
    ingot_gguf_close(single);

    /* A missing sibling must fail cleanly, not half-open. */
    unlink(second);
    ingot_gguf *broken = NULL;
    CHECK(ingot_gguf_open_split(&broken, first, err, sizeof err) != 0 && broken == NULL,
          "a missing shard is refused ('%s')", err);
    unlink(first);
}

int main(void) {
    strcpy(dir, "/tmp/ingot_w_XXXXXX");
    if (mkdtemp(dir) == NULL) { perror("mkdtemp"); return 1; }
    test_gguf_roundtrip();
    test_st_roundtrip();
    test_cross_layer();
    test_split();
    rmdir(dir);
    printf("\n%d checks, %d failures\n", checks, failures);
    return failures != 0;
}
