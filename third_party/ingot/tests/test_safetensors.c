/* safetensors reader self-test: a valid single file, a sharded directory
 * resolved three different ways, and the malformed cases none of the four
 * source readers this was distilled from ever tested.
 *
 * SPDX-License-Identifier: MIT */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "ingot/safetensors.h"

static int failures;
static int checks;

#define CHECK(cond, ...) do {                                        \
    checks++;                                                        \
    if (!(cond)) { printf("  FAIL: "); printf(__VA_ARGS__);          \
                   printf("  (%s:%d)\n", __FILE__, __LINE__);        \
                   failures++; }                                     \
    else { printf("  ok:   "); printf(__VA_ARGS__); printf("\n"); }  \
} while (0)

/* ── fixture writing ────────────────────────────────────────────────────── */

/* Writes 8-byte header length + JSON padded with spaces to an 8-byte boundary
 * + the data blob, which is what the reference implementation produces. */
static int write_st(const char *path, const char *json, const void *data, size_t nbytes) {
    size_t jlen = strlen(json);
    const size_t pad = (8 - ((8 + jlen) % 8)) % 8;
    FILE *f = fopen(path, "wb");
    if (f == NULL) return -1;
    unsigned char header[8];
    const uint64_t total = jlen + pad;
    for (int i = 0; i < 8; i++) header[i] = (unsigned char)(total >> (8 * i));
    fwrite(header, 1, 8, f);
    fwrite(json, 1, jlen, f);
    for (size_t i = 0; i < pad; i++) fputc(' ', f);
    if (nbytes != 0) fwrite(data, 1, nbytes, f);
    fclose(f);
    return 0;
}

/* Same, but the caller controls the declared header length — the only way to
 * build a container that lies about itself. */
static int write_st_raw(const char *path, uint64_t declared_len,
                        const char *json, const void *data, size_t nbytes) {
    FILE *f = fopen(path, "wb");
    if (f == NULL) return -1;
    unsigned char header[8];
    for (int i = 0; i < 8; i++) header[i] = (unsigned char)(declared_len >> (8 * i));
    fwrite(header, 1, 8, f);
    fwrite(json, 1, strlen(json), f);
    if (nbytes != 0) fwrite(data, 1, nbytes, f);
    fclose(f);
    return 0;
}

static char tmpdir[128];

static void make_tmpdir(void) {
    strcpy(tmpdir, "/tmp/ingot_st_XXXXXX");
    if (mkdtemp(tmpdir) == NULL) { perror("mkdtemp"); exit(1); }
}

static char *in_tmp(const char *name) {
    static char path[256];
    snprintf(path, sizeof path, "%s/%s", tmpdir, name);
    return path;
}

static void rm_tmpdir(void) {
    char cmd[512];
    snprintf(cmd, sizeof cmd, "rm -rf '%s'", tmpdir);
    if (system(cmd) != 0) printf("  (warning: could not clean %s)\n", tmpdir);
}

/* ── the valid case ─────────────────────────────────────────────────────── */
static void test_valid(void) {
    printf("valid safetensors\n");
    /* 4 f32 + 4 bf16 + 2 i64 = 16 + 8 + 16 = 40 bytes */
    unsigned char data[40] = {0};
    float f[4] = { 1.0f, -2.5f, 3.25f, 0.0f };
    memcpy(data, f, 16);
    /* bf16 1.0, 2.0, -1.0, 0.5 */
    const uint16_t bf[4] = { 0x3f80, 0x4000, 0xbf80, 0x3f00 };
    for (int i = 0; i < 4; i++) {
        data[16 + 2 * i]     = (unsigned char)(bf[i] & 0xff);
        data[16 + 2 * i + 1] = (unsigned char)(bf[i] >> 8);
    }
    const int64_t ids[2] = { 7, -3 };
    memcpy(data + 24, ids, 16);

    const char *json =
        "{\"__metadata__\":{\"format\":\"pt\",\"note\":\"line\\nbreak\"},"
        "\"a.weight\":{\"dtype\":\"F32\",\"shape\":[2,2],\"data_offsets\":[0,16]},"
        "\"b.weight\":{\"dtype\":\"BF16\",\"shape\":[4],\"data_offsets\":[16,24]},"
        "\"c.ids\":{\"dtype\":\"I64\",\"shape\":[2],\"data_offsets\":[24,40]}}";

    const char *path = in_tmp("model.safetensors");
    if (write_st(path, json, data, sizeof data) != 0) { failures++; return; }

    char err[256] = {0};
    ingot_st *st = NULL;
    CHECK(ingot_st_open(&st, path, err, sizeof err) == 0 && st != NULL,
          "opens (err='%s')", err);
    if (st == NULL) return;

    CHECK(ingot_st_count(st) == 3, "3 tensors, __metadata__ not counted as one");
    const ingot_st_tensor *a = ingot_st_find(st, "a.weight");
    CHECK(a != NULL && a->dtype == INGOT_DT_F32 && a->rank == 2 &&
          a->shape[0] == 2 && a->shape[1] == 2 && a->nelem == 4 && a->nbytes == 16,
          "a.weight is F32 [2,2], 16 bytes");
    if (a != NULL) {
        const float *p = (const float *)ingot_st_data(st, a);
        CHECK(p != NULL && p[0] == 1.0f && p[1] == -2.5f && p[2] == 3.25f,
              "F32 payload is bit-exact zero-copy");
    }

    const ingot_st_tensor *b = ingot_st_find(st, "b.weight");
    CHECK(b != NULL && b->dtype == INGOT_DT_BF16, "b.weight is BF16");
    if (b != NULL) {
        float out[4] = {0};
        CHECK(ingot_st_to_f32(st, b, out) == 0 && out[0] == 1.0f && out[1] == 2.0f &&
              out[2] == -1.0f && out[3] == 0.5f, "BF16 converts to f32");
    }

    const ingot_st_tensor *c = ingot_st_find(st, "c.ids");
    if (c != NULL) {
        float out[2] = {0};
        CHECK(ingot_st_to_f32(st, c, out) == 0 && out[0] == 7.0f && out[1] == -3.0f,
              "I64 converts to f32");
    }

    const char *meta = NULL;
    CHECK(ingot_st_metadata(st, "format", &meta) == 0 && strcmp(meta, "pt") == 0,
          "__metadata__ is exposed, not skipped");
    CHECK(ingot_st_metadata(st, "note", &meta) == 0 && strcmp(meta, "line\nbreak") == 0,
          "JSON escapes are decoded");
    CHECK(ingot_st_metadata(st, "absent", &meta) != 0, "absent metadata key fails");

    CHECK(ingot_st_find(st, "nope") == NULL, "absent tensor returns NULL");
    CHECK(ingot_st_shard_count(st) == 1, "single shard");

    /* the pread twin agrees with the mapping */
    if (a != NULL) {
        float copy[4];
        CHECK(ingot_st_read(st, a, 0, copy, sizeof copy, err, sizeof err) == 0 &&
              memcmp(copy, ingot_st_data(st, a), sizeof copy) == 0,
              "pread path matches the mmap");
    }

    ingot_st_prefault(st);
    ingot_st_dontneed(st);
    CHECK(a == NULL || ((const float *)ingot_st_data(st, a))[0] == 1.0f,
          "data survives prefault + dontneed");

    ingot_st_close(st);
}

/* ── sharding ───────────────────────────────────────────────────────────── */
static void test_shards(void) {
    printf("sharded models\n");
    char sub[256];
    snprintf(sub, sizeof sub, "%s/sharded", tmpdir);
    if (mkdir(sub, 0755) != 0) { failures++; return; }

    float one[2] = { 1.0f, 2.0f }, two[2] = { 3.0f, 4.0f };
    char p1[300], p2[300], idx[300];
    snprintf(p1, sizeof p1, "%s/model-00001-of-00002.safetensors", sub);
    snprintf(p2, sizeof p2, "%s/model-00002-of-00002.safetensors", sub);
    snprintf(idx, sizeof idx, "%s/model.safetensors.index.json", sub);

    write_st(p1, "{\"x\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[0,8]}}",
             one, sizeof one);
    write_st(p2, "{\"y\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[0,8]}}",
             two, sizeof two);

    char err[256] = {0};
    ingot_st *st = NULL;

    /* 1. no index.json yet: the directory glob must find both, in order */
    CHECK(ingot_st_open_dir(&st, sub, err, sizeof err) == 0 && st != NULL,
          "directory glob finds the shards (err='%s')", err);
    if (st != NULL) {
        CHECK(ingot_st_count(st) == 2 && ingot_st_shard_count(st) == 2,
              "2 tensors across 2 shards");
        const ingot_st_tensor *y = ingot_st_find(st, "y");
        CHECK(y != NULL && y->shard == 1, "y lives in shard 1");
        if (y != NULL) {
            const float *p = (const float *)ingot_st_data(st, y);
            CHECK(p != NULL && p[0] == 3.0f, "cross-shard data pointer is right");
        }
        ingot_st_close(st);
        st = NULL;
    }

    /* 2. with an index.json naming only the first shard, only that one opens */
    FILE *f = fopen(idx, "wb");
    if (f != NULL) {
        fputs("{\"metadata\":{\"total_size\":16},"
              "\"weight_map\":{\"x\":\"model-00001-of-00002.safetensors\"}}", f);
        fclose(f);
    }
    CHECK(ingot_st_open_dir(&st, sub, err, sizeof err) == 0 && st != NULL &&
          ingot_st_shard_count(st) == 1 && ingot_st_find(st, "y") == NULL,
          "index.json wins over the glob and selects its shards");
    if (st != NULL) { ingot_st_close(st); st = NULL; }

    /* 3. ingot_st_open on a directory delegates */
    CHECK(ingot_st_open(&st, sub, err, sizeof err) == 0 && st != NULL,
          "open() on a directory delegates to open_dir");
    if (st != NULL) { ingot_st_close(st); st = NULL; }

    /* 4. a name in two shards is ambiguous and must be refused */
    unlink(idx);
    char p3[300];
    snprintf(p3, sizeof p3, "%s/model-00003-of-00003.safetensors", sub);
    write_st(p3, "{\"x\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[0,8]}}",
             two, sizeof two);
    CHECK(ingot_st_open_dir(&st, sub, err, sizeof err) != 0,
          "a duplicated tensor name across shards is rejected ('%s')", err);
    if (st != NULL) { ingot_st_close(st); st = NULL; }
    unlink(p3);
}

/* ── malformed ──────────────────────────────────────────────────────────── */
static void test_malformed(void) {
    printf("malformed safetensors must be rejected\n");
    const float data[4] = { 1, 2, 3, 4 };
    char err[256];
    ingot_st *st = NULL;

    struct { const char *what; const char *json; uint64_t declared; size_t bytes; } cases[] = {
        { "header longer than the file",
          "{\"a\":{\"dtype\":\"F32\",\"shape\":[4],\"data_offsets\":[0,16]}}", 1u << 30, 16 },
        { "data_offsets reversed",
          "{\"a\":{\"dtype\":\"F32\",\"shape\":[4],\"data_offsets\":[16,0]}}", 0, 16 },
        { "byte count disagrees with shape x dtype",
          "{\"a\":{\"dtype\":\"F32\",\"shape\":[4],\"data_offsets\":[0,8]}}", 0, 16 },
        { "payload past the end of the file",
          "{\"a\":{\"dtype\":\"F32\",\"shape\":[4],\"data_offsets\":[64,80]}}", 0, 16 },
        { "missing dtype",
          "{\"a\":{\"shape\":[4],\"data_offsets\":[0,16]}}", 0, 16 },
        { "missing data_offsets",
          "{\"a\":{\"dtype\":\"F32\",\"shape\":[4]}}", 0, 16 },
        { "truncated JSON",
          "{\"a\":{\"dtype\":\"F32\",\"shape\":[4],", 0, 16 },
        { "header is not an object",
          "[1,2,3]", 0, 16 },
    };

    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
        const char *path = in_tmp("broken.safetensors");
        if (cases[i].declared != 0)
            write_st_raw(path, cases[i].declared, cases[i].json, data, cases[i].bytes);
        else
            write_st(path, cases[i].json, data, cases[i].bytes);
        err[0] = '\0';
        st = NULL;
        CHECK(ingot_st_open(&st, path, err, sizeof err) != 0 && st != NULL == 0,
              "%s -> rejected ('%s')", cases[i].what, err);
        if (st != NULL) ingot_st_close(st);
        unlink(path);
    }

    /* A header not padded to 8 makes every zero-copy typed pointer misaligned;
     * the reference writer always pads, so a file that does not is malformed. */
    {
        const char *path = in_tmp("unaligned.safetensors");
        const char *json = "{\"a\":{\"dtype\":\"F32\",\"shape\":[4],\"data_offsets\":[0,16]}}";
        write_st_raw(path, strlen(json), json, data, sizeof data);
        err[0] = '\0';
        st = NULL;
        CHECK(ingot_st_open(&st, path, err, sizeof err) != 0,
              "unpadded header (data section not 8-aligned) -> rejected ('%s')", err);
        if (st != NULL) ingot_st_close(st);
        unlink(path);
    }

    /* An unknown dtype is not fatal: the tensor is visible with its bytes, it
     * simply has no element size. A converter still wants to see it. */
    {
        const char *path = in_tmp("exotic.safetensors");
        write_st(path, "{\"a\":{\"dtype\":\"F4_E2M1\",\"shape\":[4],\"data_offsets\":[0,16]}}",
                 data, sizeof data);
        err[0] = '\0';
        st = NULL;
        const int rc = ingot_st_open(&st, path, err, sizeof err);
        const ingot_st_tensor *t = (rc == 0) ? ingot_st_find(st, "a") : NULL;
        CHECK(rc == 0 && t != NULL && t->dtype == INGOT_DT_UNKNOWN && t->nbytes == 16,
              "an unknown dtype is kept, not fatal");
        if (st != NULL) ingot_st_close(st);
        unlink(path);
    }

    CHECK(ingot_st_open(&st, "/tmp/ingot-no-such-file-xyz", err, sizeof err) != 0,
          "missing file -> rejected");
    CHECK(ingot_st_open_dir(&st, "/tmp/ingot-no-such-dir-xyz", err, sizeof err) != 0,
          "missing directory -> rejected");
}

int main(void) {
    make_tmpdir();
    test_valid();
    test_shards();
    test_malformed();
    rm_tmpdir();
    printf("\n%d checks, %d failures\n", checks, failures);
    return failures != 0;
}
