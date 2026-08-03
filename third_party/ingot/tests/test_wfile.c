/* wfile: one handle for either container. Checks that the same engine code
 * really does not learn which file the weights came from, which is the whole
 * property this layer exists to provide.
 *
 * SPDX-License-Identifier: MIT */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "ingot/wfile.h"

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

/* The same logical tensor written into both containers: a [2,3] f32 matrix.
 * GGUF stores it as ne = [3,2] (fastest dimension first), safetensors as
 * shape = [2,3]. After wfile they must look identical, which is the whole
 * claim of this layer. */
static const float VALUES[6] = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f };

static void put_u32(FILE *f, uint32_t v) {
    for (int i = 0; i < 4; i++) fputc((int)((v >> (8 * i)) & 0xff), f);
}
static void put_u64(FILE *f, uint64_t v) {
    for (int i = 0; i < 8; i++) fputc((int)((v >> (8 * i)) & 0xff), f);
}
static void put_str(FILE *f, const char *s) {
    put_u64(f, strlen(s));
    fwrite(s, 1, strlen(s), f);
}

static void write_gguf(const char *path) {
    FILE *f = fopen(path, "wb");
    if (f == NULL) { failures++; return; }
    put_u32(f, 0x46554747u);
    put_u32(f, 3);
    put_u64(f, 1);                       /* one tensor  */
    put_u64(f, 1);                       /* one KV      */
    put_str(f, "general.architecture");
    put_u32(f, 8 /* string */);
    put_str(f, "demo");
    put_str(f, "m.weight");
    put_u32(f, 2);
    put_u64(f, 3);                       /* ne[0], the fastest dimension */
    put_u64(f, 2);
    put_u32(f, 0 /* F32 */);
    put_u64(f, 0);
    while (ftell(f) % 32 != 0) fputc(0, f);
    fwrite(VALUES, 1, sizeof VALUES, f);
    fclose(f);
}

static void write_st(const char *path) {
    const char *json = "{\"m.weight\":{\"dtype\":\"F32\",\"shape\":[2,3],"
                       "\"data_offsets\":[0,24]}}";
    size_t jlen = strlen(json);
    const size_t pad = (8 - ((8 + jlen) % 8)) % 8;
    FILE *f = fopen(path, "wb");
    if (f == NULL) { failures++; return; }
    put_u64(f, jlen + pad);
    fwrite(json, 1, jlen, f);
    for (size_t i = 0; i < pad; i++) fputc(' ', f);
    fwrite(VALUES, 1, sizeof VALUES, f);
    fclose(f);
}

int main(void) {
    strcpy(dir, "/tmp/ingot_wf_XXXXXX");
    if (mkdtemp(dir) == NULL) { perror("mkdtemp"); return 1; }
    char gguf[256], st[256];
    snprintf(gguf, sizeof gguf, "%s/m.gguf", dir);
    snprintf(st, sizeof st, "%s/model.safetensors", dir);
    write_gguf(gguf);
    write_st(st);

    printf("wfile hides the container\n");
    char err[256] = {0};
    ingot_wfile *a = NULL, *b = NULL;
    CHECK(ingot_wfile_open(&a, gguf, err, sizeof err) == 0, "opens a GGUF (%s)", err);
    CHECK(ingot_wfile_open(&b, st, err, sizeof err) == 0, "opens a safetensors (%s)", err);
    if (a == NULL || b == NULL) { failures++; goto out; }

    CHECK(ingot_wfile_container(a) == INGOT_CONTAINER_GGUF, "container reported as GGUF");
    CHECK(ingot_wfile_container(b) == INGOT_CONTAINER_SAFETENSORS,
          "container reported as safetensors");

    const ingot_wtensor *ta = ingot_wfile_find(a, "m.weight");
    const ingot_wtensor *tb = ingot_wfile_find(b, "m.weight");
    CHECK(ta != NULL && tb != NULL, "same name found in both");
    if (ta == NULL || tb == NULL) goto out;

    CHECK(ta->rank == tb->rank && ta->shape[0] == tb->shape[0] &&
          ta->shape[1] == tb->shape[1],
          "same row-major shape [%llu,%llu] from ne=[3,2] and shape=[2,3]",
          (unsigned long long)ta->shape[0], (unsigned long long)ta->shape[1]);
    CHECK(ta->shape[0] == 2 && ta->shape[1] == 3, "and it is [2,3], not [3,2]");
    CHECK(ta->dtype == tb->dtype && ta->dtype == INGOT_DT_F32, "same dtype");
    CHECK(ta->nelem == tb->nelem && ta->nelem == 6, "same element count");
    CHECK(ta->nbytes == tb->nbytes && ta->nbytes == 24, "same byte count");
    CHECK(memcmp(ta->data, tb->data, 24) == 0, "identical payload bytes");
    CHECK(ta->ggml_type == 0 && tb->ggml_type == -1,
          "the ggml type is still reachable on the GGUF side only");

    float fa[6] = {0}, fb[6] = {0};
    CHECK(ingot_wfile_to_f32(a, ta, fa) == 0 && ingot_wfile_to_f32(b, tb, fb) == 0 &&
          memcmp(fa, fb, sizeof fa) == 0 && fa[5] == 6.0f,
          "to_f32 agrees across containers");

    CHECK(ingot_wfile_gguf(a) != NULL && ingot_wfile_gguf(b) == NULL,
          "the GGUF handle is reachable only for a GGUF");
    CHECK(ingot_wfile_st(b) != NULL && ingot_wfile_st(a) == NULL,
          "the safetensors handle is reachable only for a safetensors");
    CHECK(strcmp(ingot_gguf_arch(ingot_wfile_gguf(a)), "demo") == 0,
          "GGUF metadata still reachable through the escape hatch");

    CHECK(ingot_wfile_find(a, "nope") == NULL, "absent name returns NULL");
    /* A separate variable on purpose: a failed open sets *out to NULL, so
     * reusing one that still owns a handle would leak it. `make test-leaks`
     * catches exactly this. */
    ingot_wfile *missing = NULL;
    CHECK(ingot_wfile_open(&missing, "/tmp/ingot-nope-xyz", err, sizeof err) != 0 &&
          missing == NULL, "missing file rejected, out pointer cleared");

out:
    ingot_wfile_close(a);
    ingot_wfile_close(b);
    unlink(gguf);
    unlink(st);
    rmdir(dir);
    printf("\n%d checks, %d failures\n", checks, failures);
    return failures != 0;
}
