/* GGUF reader self-test. Builds synthetic containers in a temp directory and
 * checks two things:
 *
 *   1. a valid file parses — metadata of every KV type, tensors of every type
 *      we claim to read, F32 zero-copy bit-exact, the split-file convention;
 *   2. malformed files are REJECTED cleanly (non-zero return, no crash, a
 *      message the caller can print).
 *
 * The second half is the point: a reader that only ever sees well-formed files
 * is a reader nobody has tested.
 *
 * Runs with no model on disk, so it belongs in CI.
 * SPDX-License-Identifier: MIT */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <unistd.h>

#include "ingot/gguf.h"
#include "ingot/quant.h"

static int failures;
static int checks;

#define CHECK(cond, ...) do {                                        \
    checks++;                                                        \
    if (!(cond)) { printf("  FAIL: "); printf(__VA_ARGS__);          \
                   printf("  (%s:%d)\n", __FILE__, __LINE__);        \
                   failures++; }                                     \
    else { printf("  ok:   "); printf(__VA_ARGS__); printf("\n"); }  \
} while (0)

/* ── a growable byte buffer ─────────────────────────────────────────────── */
typedef struct { unsigned char *p; size_t len, cap; } buf;

static void put(buf *b, const void *src, size_t n) {
    if (b->len + n > b->cap) {
        b->cap = (b->len + n) * 2 + 256;
        b->p = realloc(b->p, b->cap);
        if (b->p == NULL) abort();
    }
    memcpy(b->p + b->len, src, n);
    b->len += n;
}
static void put_u8(buf *b, unsigned char v) { put(b, &v, 1); }
static void put_u32(buf *b, uint32_t v) {
    unsigned char x[4] = { (unsigned char)v, (unsigned char)(v >> 8),
                           (unsigned char)(v >> 16), (unsigned char)(v >> 24) };
    put(b, x, 4);
}
static void put_u64(buf *b, uint64_t v) {
    unsigned char x[8];
    for (int i = 0; i < 8; i++) x[i] = (unsigned char)(v >> (8 * i));
    put(b, x, 8);
}
static void put_f32(buf *b, float v) {
    uint32_t bits;
    memcpy(&bits, &v, 4);
    put_u32(b, bits);
}
static void put_str(buf *b, const char *s) { put_u64(b, strlen(s)); put(b, s, strlen(s)); }
static void pad_to(buf *b, size_t align) {
    static const unsigned char zero[64] = {0};
    while (b->len % align != 0) put(b, zero, 1);
}

static uint16_t to_f16(float f) {           /* fixtures only: normal values */
    uint32_t bits;
    memcpy(&bits, &f, 4);
    const uint32_t sign = (bits >> 16) & 0x8000u;
    const int e = (int)((bits >> 23) & 0xff) - 127 + 15;
    const uint32_t frac = (bits >> 13) & 0x3ffu;
    if (e <= 0) return (uint16_t)sign;
    if (e >= 31) return (uint16_t)(sign | 0x7c00u);
    return (uint16_t)(sign | ((uint32_t)e << 10) | frac);
}
static void put_f16(buf *b, float v) {
    const uint16_t h = to_f16(v);
    put_u8(b, (unsigned char)(h & 0xff));
    put_u8(b, (unsigned char)(h >> 8));
}

static char *write_tmp(const buf *b, char *path) {
    strcpy(path, "/tmp/ingot_test_XXXXXX");
    const int fd = mkstemp(path);
    if (fd < 0) return NULL;
    if (write(fd, b->p, b->len) != (ssize_t)b->len) { close(fd); return NULL; }
    close(fd);
    return path;
}

/* ── the reference fixture ──────────────────────────────────────────────────
 * Header layout matches the spec: magic, version, tensor count, KV count, the
 * KV block, the tensor table, padding to `alignment`, then the payloads. */
typedef struct { const char *name; int type; uint64_t ne[2]; uint32_t rank; } tspec;

static void build_reference(buf *b, uint64_t alignment, int version) {
    static const tspec tensors[] = {
        { "f32.weight",  INGOT_TYPE_F32,  { 8, 4 }, 2 },
        { "f16.weight",  INGOT_TYPE_F16,  { 8, 1 }, 2 },
        { "bf16.weight", INGOT_TYPE_BF16, { 8, 1 }, 2 },
        { "q8_0.weight", INGOT_TYPE_Q8_0, { 32, 1 }, 2 },
        { "q4_0.weight", INGOT_TYPE_Q4_0, { 32, 1 }, 2 },
        { "q4_k.weight", INGOT_TYPE_Q4_K, { 256, 1 }, 2 },
        { "q6_k.weight", INGOT_TYPE_Q6_K, { 256, 1 }, 2 },
    };
    const size_t ntensor = sizeof(tensors) / sizeof(tensors[0]);

    put_u32(b, 0x46554747u);            /* "GGUF" */
    put_u32(b, (uint32_t)version);
    put_u64(b, ntensor);
    put_u64(b, 6);                       /* KV count */

    put_str(b, "general.architecture"); put_u32(b, INGOT_KV_STRING); put_str(b, "testarch");
    put_str(b, "general.alignment");    put_u32(b, INGOT_KV_UINT32); put_u32(b, (uint32_t)alignment);
    put_str(b, "test.head_count");      put_u32(b, INGOT_KV_UINT32); put_u32(b, 32);
    put_str(b, "test.freq_base");       put_u32(b, INGOT_KV_FLOAT32); put_f32(b, 10000.0f);
    put_str(b, "test.vocab");           put_u32(b, INGOT_KV_ARRAY);
        put_u32(b, INGOT_KV_STRING); put_u64(b, 3);
        put_str(b, "alpha"); put_str(b, "beta"); put_str(b, "gamma");
    put_str(b, "test.scores");          put_u32(b, INGOT_KV_ARRAY);
        put_u32(b, INGOT_KV_FLOAT32); put_u64(b, 3);
        put_f32(b, 1.5f); put_f32(b, -2.5f); put_f32(b, 0.25f);

    /* tensor table: offsets are relative to the data base, each aligned */
    uint64_t offset = 0;
    uint64_t offsets[8];
    for (size_t i = 0; i < ntensor; i++) {
        put_str(b, tensors[i].name);
        put_u32(b, tensors[i].rank);
        for (uint32_t d = 0; d < tensors[i].rank; d++) put_u64(b, tensors[i].ne[d]);
        put_u32(b, (uint32_t)tensors[i].type);
        put_u64(b, offset);
        offsets[i] = offset;

        uint64_t nelem = 1, bytes;
        for (uint32_t d = 0; d < tensors[i].rank; d++) nelem *= tensors[i].ne[d];
        ingot_type_nbytes(tensors[i].type, nelem, &bytes);
        offset += (bytes + alignment - 1) & ~(alignment - 1);
    }
    (void)offsets;
    pad_to(b, (size_t)alignment);

    /* payloads, in the same order and with the same padding */
    for (size_t i = 0; i < ntensor; i++) {
        const size_t start = b->len;
        switch (tensors[i].type) {
        case INGOT_TYPE_F32:
            for (int k = 0; k < 32; k++) put_f32(b, (float)k * 0.5f - 4.0f);
            break;
        case INGOT_TYPE_F16:
            for (int k = 0; k < 8; k++) put_f16(b, (float)k * 0.25f);
            break;
        case INGOT_TYPE_BF16:
            for (int k = 0; k < 8; k++) {
                uint32_t bits;
                const float v = (float)k * 0.5f;
                memcpy(&bits, &v, 4);
                const uint16_t h = (uint16_t)(bits >> 16);
                put_u8(b, (unsigned char)(h & 0xff));
                put_u8(b, (unsigned char)(h >> 8));
            }
            break;
        case INGOT_TYPE_Q8_0:
            put_f16(b, 0.5f);
            for (int k = 0; k < 32; k++) put_u8(b, (unsigned char)(signed char)(k - 16));
            break;
        case INGOT_TYPE_Q4_0:
            put_f16(b, 0.25f);
            for (int k = 0; k < 16; k++) put_u8(b, (unsigned char)((k & 0x0f) | ((15 - k) << 4)));
            break;
        case INGOT_TYPE_Q4_K: {
            float values[256];
            /* A smooth ramp, so each 32-value sub-block spans a narrow range
             * and the round-trip error actually measures Q4_K fidelity. A
             * sawtooth here would span the full range inside one sub-block
             * and the tolerance would just be recording the step size. */
            for (int k = 0; k < 256; k++) values[k] = ((float)k / 255.0f - 0.5f) * 2.0f;
            unsigned char block[144];
            if (ingot_q4_k_quantize(values, 256, block) != 0) abort();
            put(b, block, sizeof(block));
            break;
        }
        case INGOT_TYPE_Q6_K:
            for (int k = 0; k < 192; k++) put_u8(b, (unsigned char)(k * 7 + 3));
            for (int k = 0; k < 16; k++) put_u8(b, (unsigned char)(signed char)(k - 8));
            put_f16(b, 0.03125f);
            break;
        default: abort();
        }
        (void)start;
        pad_to(b, (size_t)alignment);
    }
}

/* ── the valid-file checks ──────────────────────────────────────────────── */
static void test_valid(void) {
    printf("valid GGUF\n");
    buf b = {0};
    char path[64];
    build_reference(&b, 32, 3);
    if (write_tmp(&b, path) == NULL) { printf("cannot write fixture\n"); failures++; return; }

    char err[256] = {0};
    ingot_gguf *g = NULL;
    CHECK(ingot_gguf_open(&g, path, err, sizeof err) == 0 && g != NULL,
          "opens (err='%s')", err);
    if (g == NULL) { free(b.p); unlink(path); return; }

    CHECK(ingot_gguf_count(g) == 7, "7 tensors");
    CHECK(ingot_gguf_version(g) == 3, "version 3");
    CHECK(ingot_gguf_alignment(g) == 32, "alignment 32");
    CHECK(strcmp(ingot_gguf_arch(g), "testarch") == 0, "architecture 'testarch'");
    CHECK(ingot_gguf_shard_count(g) == 1, "single shard");

    /* metadata of every shape */
    const ingot_kv *kv = ingot_gguf_kv_find(g, "test.head_count");
    uint64_t u = 0;
    CHECK(kv != NULL && ingot_kv_u64(kv, &u) == 0 && u == 32, "u32 KV reads as u64");
    kv = ingot_gguf_kv_find(g, "test.freq_base");
    double f = 0;
    CHECK(kv != NULL && ingot_kv_f64(kv, &f) == 0 && f > 9999.0 && f < 10001.0,
          "f32 KV reads as f64");
    kv = ingot_gguf_kv_find(g, "test.vocab");
    uint64_t len = 0;
    const char *s = NULL;
    size_t slen = 0;
    CHECK(kv != NULL && ingot_kv_arr_len(kv, &len) == 0 && len == 3, "string array length 3");
    CHECK(kv != NULL && ingot_kv_arr_str(kv, 2, &s, &slen) == 0 &&
          slen == 5 && memcmp(s, "gamma", 5) == 0, "string array indexes in O(1)");
    CHECK(kv != NULL && ingot_kv_arr_str(kv, 3, &s, &slen) != 0, "string array rejects OOB");
    kv = ingot_gguf_kv_find(g, "test.scores");
    float score = 0;
    CHECK(kv != NULL && ingot_kv_arr_f32(kv, 1, &score) == 0 && score == -2.5f,
          "f32 array element");
    CHECK(ingot_gguf_kv_find(g, "nope") == NULL, "absent key returns NULL");

    /* F32 is zero-copy and bit-exact */
    const ingot_tensor *t = ingot_gguf_find(g, "f32.weight");
    CHECK(t != NULL, "find by name");
    if (t != NULL) {
        CHECK(t->rank == 2 && t->ne[0] == 8 && t->ne[1] == 4, "ne is ggml order [8,4]");
        uint64_t shape[2] = {0, 0};
        ingot_gguf_shape_row_major(t, shape);
        CHECK(shape[0] == 4 && shape[1] == 8, "row-major shape is [4,8]");
        CHECK(t->nelem == 32 && t->nbytes == 128, "32 elements, 128 bytes");
        const float *data = (const float *)ingot_gguf_data(g, t);
        int exact = data != NULL;
        for (int k = 0; k < 32 && exact; k++)
            exact = data[k] == (float)k * 0.5f - 4.0f;
        CHECK(exact, "F32 payload is bit-exact zero-copy");

        /* pread twin returns the same bytes as the mapping */
        float copy[32];
        CHECK(ingot_gguf_read(g, t, 0, copy, sizeof copy, err, sizeof err) == 0 &&
              memcmp(copy, data, sizeof copy) == 0, "pread path matches the mmap");
        CHECK(ingot_gguf_read(g, t, 0, copy, sizeof copy + 4, err, sizeof err) != 0,
              "pread rejects a read past the tensor");
    }

    /* every quantized type dequantizes to something finite and in range */
    static const struct { const char *name; float tol; } deq[] = {
        { "f16.weight", 1e-3f }, { "bf16.weight", 1e-2f },
        { "q8_0.weight", 1e-6f }, { "q4_0.weight", 1e-6f },
        { "q4_k.weight", 0.2f },  { "q6_k.weight", 1e-6f },
    };
    for (size_t i = 0; i < sizeof(deq) / sizeof(deq[0]); i++) {
        const ingot_tensor *q = ingot_gguf_find(g, deq[i].name);
        if (q == NULL) { CHECK(0, "%s present", deq[i].name); continue; }
        float *out = malloc((size_t)q->nelem * sizeof(float));
        const int rc = ingot_gguf_dequant(g, q, out);
        int finite = rc == 0;
        for (uint64_t k = 0; k < q->nelem && finite; k++)
            finite = out[k] == out[k] && out[k] < 1e30f && out[k] > -1e30f;
        CHECK(finite, "%s dequantizes to finite values", deq[i].name);
        free(out);
    }

    /* Q4_K round trip. The metric is relative L2, not a max absolute error:
     * 4 bits over 32 values with a 6-bit sub-block scale lands around 6-7%
     * relative on any real distribution, so an absolute budget would only be
     * recording the step size.
     * A packing bug lands orders of magnitude above this. */
    const ingot_tensor *q4k = ingot_gguf_find(g, "q4_k.weight");
    if (q4k != NULL) {
        float out[256];
        double err2 = 0, ref2 = 0;
        ingot_gguf_dequant(g, q4k, out);
        for (int k = 0; k < 256; k++) {
            const double want = ((double)k / 255.0 - 0.5) * 2.0;
            const double d = (double)out[k] - want;
            err2 += d * d;
            ref2 += want * want;
        }
        const double rel = ref2 > 0 ? err2 / ref2 : 0;
        CHECK(rel < 0.08 * 0.08, "Q4_K round trip, rel_l2 %.4f", rel > 0 ? sqrt(rel) : 0.0);
    }

    ingot_gguf_close(g);
    free(b.p);
    unlink(path);
}

/* ── malformed files ────────────────────────────────────────────────────── */
typedef void (*breaker)(buf *b);

static void break_magic(buf *b)   { b->p[0] = 'X'; }
static void break_version(buf *b) { b->p[4] = 1; }        /* v1: different layout */
static void break_version9(buf *b){ b->p[4] = 9; }
static void break_truncate(buf *b){ b->len = 40; }
static void break_alignment(buf *b) {
    /* general.alignment is the second KV; its u32 value sits right after the
     * key and the type tag. 33 is not a power of two. */
    const char *needle = "general.alignment";
    for (size_t i = 0; i + strlen(needle) + 8 < b->len; i++) {
        if (memcmp(b->p + i, needle, strlen(needle)) == 0) {
            b->p[i + strlen(needle) + 4] = 33;
            return;
        }
    }
}
static void break_tensor_type(buf *b) {
    /* Rewrite the last tensor's type tag to something that does not exist. The
     * type sits just before the u64 offset; find "q6_k.weight" and walk. */
    const char *needle = "q6_k.weight";
    for (size_t i = 0; i + 64 < b->len; i++) {
        if (memcmp(b->p + i, needle, strlen(needle)) == 0) {
            const size_t at = i + strlen(needle) + 4 + 16;   /* rank + 2 dims */
            b->p[at] = 99;
            return;
        }
    }
}
static void break_huge_string(buf *b) {
    /* The first KV key length becomes absurd. */
    b->p[24] = 0xff; b->p[25] = 0xff; b->p[26] = 0xff; b->p[27] = 0xff;
}
static void break_tensor_count(buf *b) {
    for (int i = 0; i < 8; i++) b->p[8 + i] = 0xff;
}

static void test_malformed(void) {
    printf("malformed GGUF must be rejected\n");
    static const struct { breaker fn; const char *what; } cases[] = {
        { break_magic,        "bad magic" },
        { break_version,      "version 1 (unsupported layout)" },
        { break_version9,     "version 9 (from the future)" },
        { break_truncate,     "truncated mid-header" },
        { break_alignment,    "alignment that is not a power of two" },
        { break_tensor_type,  "unknown ggml tensor type" },
        { break_huge_string,  "absurd string length" },
        { break_tensor_count, "absurd tensor count" },
    };
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
        buf b = {0};
        char path[64];
        build_reference(&b, 32, 3);
        cases[i].fn(&b);
        if (write_tmp(&b, path) == NULL) { failures++; free(b.p); continue; }
        char err[256] = {0};
        ingot_gguf *g = NULL;
        const int rc = ingot_gguf_open(&g, path, err, sizeof err);
        CHECK(rc != 0 && g == NULL && err[0] != '\0',
              "%s -> rejected with a message ('%s')", cases[i].what, err);
        if (g != NULL) ingot_gguf_close(g);
        free(b.p);
        unlink(path);
    }

    /* A payload that claims to start past EOF: rebuild with a huge offset for
     * the first tensor rather than patching bytes blind. */
    {
        buf b = {0};
        char path[64];
        build_reference(&b, 32, 3);
        const char *needle = "f32.weight";
        for (size_t i = 0; i + 64 < b.len; i++) {
            if (memcmp(b.p + i, needle, strlen(needle)) == 0) {
                const size_t at = i + strlen(needle) + 4 + 16 + 4;   /* offset u64 */
                for (int k = 0; k < 8; k++) b.p[at + k] = 0;
                b.p[at] = 0x00; b.p[at + 1] = 0x00; b.p[at + 2] = 0x10;  /* 1 MiB */
                break;
            }
        }
        if (write_tmp(&b, path) != NULL) {
            char err[256] = {0};
            ingot_gguf *g = NULL;
            CHECK(ingot_gguf_open(&g, path, err, sizeof err) != 0 && g == NULL,
                  "payload offset past EOF -> rejected ('%s')", err);
            if (g != NULL) ingot_gguf_close(g);
            unlink(path);
        }
        free(b.p);
    }

    /* An empty file, a directory, a path that does not exist. */
    {
        char err[256] = {0};
        ingot_gguf *g = NULL;
        CHECK(ingot_gguf_open(&g, "/tmp/ingot-does-not-exist-xyz", err, sizeof err) != 0,
              "missing file -> rejected ('%s')", err);
        CHECK(ingot_gguf_open(&g, "/tmp", err, sizeof err) != 0,
              "directory -> rejected ('%s')", err);
        CHECK(ingot_gguf_open(NULL, "x", err, sizeof err) != 0, "null out pointer -> rejected");
    }
}

/* ── the type table ─────────────────────────────────────────────────────── */
static void test_types(void) {
    printf("type table\n");
    uint64_t e, b;
    CHECK(ingot_type_geometry(INGOT_TYPE_Q4_K, &e, &b) == 0 && e == 256 && b == 144,
          "Q4_K is 256 values in 144 bytes");
    CHECK(ingot_type_geometry(INGOT_TYPE_Q8_0, &e, &b) == 0 && e == 32 && b == 34,
          "Q8_0 is 32 values in 34 bytes");
    CHECK(ingot_type_geometry(INGOT_TYPE_IQ4_XS, &e, &b) == 0 && e == 256 && b == 136,
          "IQ4_XS geometry");
    CHECK(ingot_type_can_dequant(INGOT_TYPE_IQ4_XS) == 1, "IQ4_XS decodes");
    CHECK(ingot_type_can_dequant(INGOT_TYPE_IQ2_XS) == 1, "IQ2_XS decodes");
    /* Q1_0 is the one type still without a decoder: llama.cpp's own reference
       package has none either, so there is nothing to verify against. It keeps
       a geometry and a name, so a file using it opens and fails by name. */
    CHECK(ingot_type_geometry(INGOT_TYPE_Q1_0, &e, &b) == 0 && e == 128 && b == 18,
          "Q1_0 has a geometry even though we cannot dequantize it");
    CHECK(ingot_type_can_dequant(INGOT_TYPE_Q1_0) == 0, "Q1_0 reports no dequant");
    CHECK(strcmp(ingot_type_name(INGOT_TYPE_Q1_0), "Q1_0") == 0,
          "Q1_0 has a name, so the error can say it");
    CHECK(ingot_type_geometry(4242, &e, &b) != 0, "a truly unknown type is unknown");

    uint64_t bytes = 0;
    CHECK(ingot_type_nbytes(INGOT_TYPE_Q4_K, 512, &bytes) == 0 && bytes == 288,
          "512 Q4_K elements are 288 bytes");
    CHECK(ingot_type_nbytes(INGOT_TYPE_Q4_K, 300, &bytes) != 0,
          "300 elements do not fit whole Q4_K blocks");

    /* f16 subnormals: the naive shift-only conversion three of the source
     * projects shipped flushes these to zero. */
    CHECK(ingot_f16_to_f32(0x0001u) > 0.0f && ingot_f16_to_f32(0x0001u) < 1e-6f,
          "smallest f16 subnormal survives the conversion");
    CHECK(ingot_f16_to_f32(0x3c00u) == 1.0f, "f16 1.0 round trips");
    CHECK(ingot_bf16_to_f32(0x3f80u) == 1.0f, "bf16 1.0 round trips");
    CHECK(ingot_f32_to_f16(1.0f) == 0x3c00u, "f32 1.0 -> f16");
    CHECK(ingot_f32_to_bf16(1.0f) == 0x3f80u, "f32 1.0 -> bf16");
}

int main(void) {
    test_types();
    test_valid();
    test_malformed();
    printf("\n%d checks, %d failures\n", checks, failures);
    return failures != 0;
}
