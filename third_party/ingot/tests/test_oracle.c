/* The only test here that can catch a shared misunderstanding of a format.
 *
 * Every other test compares ingot against ingot: a round trip through one
 * implementation proves consistency, not correctness — a writer and a reader
 * that share a wrong idea agree perfectly. This one decodes fixtures whose
 * expected output came from llama.cpp's own `gguf` package: a different
 * implementation, in a different language, by the people who define the
 * format.
 *
 * The fixtures are pseudo-random block bytes, not quantized real data, so the
 * whole codebook gets walked rather than the handful of entries a smooth
 * signal would touch.
 *
 * Regenerate with:  python3 tools/gen_reference.py   (needs the gguf package)
 * Missing fixtures are reported and skipped, so a machine without Python can
 * still run the rest of the suite.
 *
 * SPDX-License-Identifier: MIT */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ingot/quant.h"

static int failures;
static int checks;
static int skipped;

#define CHECK(cond, ...) do {                                        \
    checks++;                                                        \
    if (!(cond)) { printf("  FAIL: "); printf(__VA_ARGS__);          \
                   printf("  (%s:%d)\n", __FILE__, __LINE__);        \
                   failures++; }                                     \
    else { printf("  ok:   "); printf(__VA_ARGS__); printf("\n"); }  \
} while (0)

#ifndef INGOT_FIXTURES
#define INGOT_FIXTURES "tests/fixtures"
#endif

static void *slurp(const char *path, size_t *size) {
    FILE *f = fopen(path, "rb");
    if (f == NULL) return NULL;
    fseek(f, 0, SEEK_END);
    const long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (n <= 0) { fclose(f); return NULL; }
    void *buffer = malloc((size_t)n);
    if (buffer == NULL || fread(buffer, 1, (size_t)n, f) != (size_t)n) {
        free(buffer);
        fclose(f);
        return NULL;
    }
    fclose(f);
    *size = (size_t)n;
    return buffer;
}

int main(void) {
    char path[512];
    snprintf(path, sizeof path, "%s/manifest.txt", INGOT_FIXTURES);
    size_t manifest_size = 0;
    char *manifest = (char *)slurp(path, &manifest_size);
    if (manifest == NULL) {
        printf("no fixtures at %s — run: python3 tools/gen_reference.py\n", INGOT_FIXTURES);
        printf("(skipping the oracle comparison, not failing: it needs the gguf package)\n");
        return 0;
    }

    /* Geometry first, for EVERY type ggml defines — including the ones with no
     * decoder here. Byte accounting is a claim ingot makes about a format even
     * when it cannot decode it, and getting it wrong mis-sizes a whole tensor
     * without ever looking odd. */
    snprintf(path, sizeof path, "%s/geometry.txt", INGOT_FIXTURES);
    size_t geometry_size = 0;
    char *geometry = (char *)slurp(path, &geometry_size);
    if (geometry != NULL) {
        printf("block geometry vs GGML_QUANT_SIZES\n");
        size_t seen = 0, wrong_geometry = 0;
        char worst_name[64] = "";
        for (char *line = strtok(geometry, "\n"); line != NULL; line = strtok(NULL, "\n")) {
            char name[64];
            int type = 0;
            unsigned block_size = 0, type_size = 0;
            if (sscanf(line, "%63s %d %u %u", name, &type, &block_size, &type_size) != 4)
                continue;
            seen++;
            uint64_t elems = 0, bytes = 0;
            if (ingot_type_geometry(type, &elems, &bytes) != 0 ||
                elems != block_size || bytes != type_size) {
                wrong_geometry++;
                snprintf(worst_name, sizeof worst_name, "%s", name);
            }
        }
        CHECK(wrong_geometry == 0, "%zu types agree on block size and byte size%s%s",
              seen, wrong_geometry ? ", first mismatch: " : "", worst_name);
        free(geometry);
    }

    printf("decoders vs llama.cpp's own dequantizer\n");
    for (char *line = strtok(manifest, "\n"); line != NULL; line = strtok(NULL, "\n")) {
        char name[64];
        int type = 0;
        unsigned block_size = 0, type_size = 0, nblocks = 0;
        if (sscanf(line, "%63s %d %u %u %u", name, &type, &block_size, &type_size,
                   &nblocks) != 5) continue;

        /* Two independent statements about the format, both worth checking:
         * that ingot agrees on the geometry, and that it agrees on the values. */
        uint64_t elems = 0, bytes = 0;
        if (ingot_type_geometry(type, &elems, &bytes) != 0 ||
            elems != block_size || bytes != type_size) {
            CHECK(0, "%-8s geometry: ingot says %llu/%llu, the reference says %u/%u",
                  name, (unsigned long long)elems, (unsigned long long)bytes,
                  block_size, type_size);
            continue;
        }

        size_t raw_size = 0, ref_size = 0;
        snprintf(path, sizeof path, "%s/%s.bin", INGOT_FIXTURES, name);
        unsigned char *raw = (unsigned char *)slurp(path, &raw_size);
        snprintf(path, sizeof path, "%s/%s.ref", INGOT_FIXTURES, name);
        float *reference = (float *)slurp(path, &ref_size);
        if (raw == NULL || reference == NULL) {
            printf("  skip: %s (fixture missing)\n", name);
            skipped++;
            free(raw);
            free(reference);
            continue;
        }
        const size_t nelem = (size_t)nblocks * block_size;

        if (!ingot_type_can_dequant(type)) {
            printf("  skip: %-8s (ingot has no decoder; geometry checked above)\n", name);
            skipped++;
            free(raw);
            free(reference);
            continue;
        }

        float *got = (float *)malloc(nelem * sizeof(float));
        const int rc = ingot_dequant(type, raw, nelem, got);

        /* The budget is f32 rounding, not tolerance for a layout mistake: the
         * reference computes in float32 too, so agreement should be to a few
         * ulp. A wrong grid entry or a misread offset lands whole orders of
         * magnitude away, which is why this can be tight. */
        size_t wrong = 0;
        double worst = 0;
        for (size_t i = 0; i < nelem && rc == 0; i++) {
            const double a = got[i], b = reference[i];
            const double scale = fmax(1e-6, fabs(b));
            const double rel = fabs(a - b) / scale;
            if (rel > worst) worst = rel;
            if (rel > 1e-5) wrong++;
        }
        if (rc != 0) {
            CHECK(0, "%-8s decoder returned -1 (claims it can decode, then does not)", name);
        } else {
            CHECK(wrong == 0, "%-8s %6zu values, worst rel %.2e%s", name, nelem, worst,
                  wrong ? "  <-- MISMATCH" : "");
        }
        free(raw);
        free(reference);
        free(got);
    }
    free(manifest);

    printf("\n%d checks, %d failures, %d skipped\n", checks, failures, skipped);
    return failures != 0;
}
