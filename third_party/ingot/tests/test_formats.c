/* Every format, three ways:
 *
 *   1. round trip — quantize a realistic signal, decode it, measure relative
 *      L2 against what the bit width can actually deliver. A packing bug lands
 *      orders of magnitude above the budget, so this is a real gate; the
 *      budgets are the measured numbers plus headroom, printed on every run so
 *      a drift is visible before it trips.
 *   2. generic vs specialized — ingot_matvec(type, ...) must agree with
 *      ingot_dequant_matrix() followed by a hand-written dot, for EVERY type
 *      the library claims to decode. This is what makes "works for whatever
 *      file you were handed" a checked claim rather than a hope.
 *   3. generic matmat vs per-token matvec, on awkward shapes.
 *
 * SPDX-License-Identifier: MIT */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

static uint32_t rng = 0x9E3779B9u;
static float rnd_unit(void) {
    rng = rng * 1664525u + 1013904223u;
    return (float)(rng >> 8) / 16777216.0f - 0.5f;
}
/* Sum of twelve uniforms: the bell a weight matrix actually has, no rand(). */
static float rnd_bell(void) {
    float acc = 0;
    for (int k = 0; k < 12; k++) acc += rnd_unit() + 0.5f;
    return 0.02f * (acc - 6.0f);
}

static double rel_l2(const float *a, const float *b, size_t n) {
    double err = 0, ref = 0;
    for (size_t i = 0; i < n; i++) {
        const double d = (double)a[i] - (double)b[i];
        err += d * d;
        ref += (double)b[i] * (double)b[i];
    }
    return ref > 0 ? sqrt(err / ref) : sqrt(err);
}

/* Every type the library can decode. `budget` is the round-trip relative-L2
 * ceiling for the ones it can also encode; -1 means encode-only-not-supported,
 * so the format is exercised with pseudo-random block bytes instead. */
static const struct { int type; const char *name; double budget; } FORMATS[] = {
    { INGOT_TYPE_F32,    "F32",    1e-9 },
    { INGOT_TYPE_F16,    "F16",    2e-3 },
    { INGOT_TYPE_BF16,   "BF16",   2e-2 },
    { INGOT_TYPE_Q8_0,   "Q8_0",   1e-2 },
    { INGOT_TYPE_Q4_0,   "Q4_0",   0.15 },
    { INGOT_TYPE_Q4_1,   "Q4_1",   0.15 },
    { INGOT_TYPE_Q5_0,   "Q5_0",   0.08 },
    { INGOT_TYPE_Q5_1,   "Q5_1",   0.08 },
    { INGOT_TYPE_Q4_K,   "Q4_K",   0.09 },
    { INGOT_TYPE_Q6_K,   "Q6_K",   0.03 },
    { INGOT_TYPE_Q2_K,   "Q2_K",   -1 },
    { INGOT_TYPE_Q3_K,   "Q3_K",   -1 },
    { INGOT_TYPE_Q5_K,   "Q5_K",   -1 },
    { INGOT_TYPE_Q8_1,   "Q8_1",   -1 },
    { INGOT_TYPE_Q8_K,   "Q8_K",   -1 },
    { INGOT_TYPE_IQ4_NL, "IQ4_NL", -1 },
    { INGOT_TYPE_IQ4_XS, "IQ4_XS", -1 },
    { INGOT_TYPE_IQ1_S,  "IQ1_S",  -1 },
    { INGOT_TYPE_IQ1_M,  "IQ1_M",  -1 },
    { INGOT_TYPE_IQ2_XXS,"IQ2_XXS",-1 },
    { INGOT_TYPE_IQ2_XS, "IQ2_XS", -1 },
    { INGOT_TYPE_IQ2_S,  "IQ2_S",  -1 },
    { INGOT_TYPE_IQ3_XXS,"IQ3_XXS",-1 },
    { INGOT_TYPE_IQ3_S,  "IQ3_S",  -1 },
    { INGOT_TYPE_TQ1_0,  "TQ1_0",  -1 },
    { INGOT_TYPE_TQ2_0,  "TQ2_0",  -1 },
    { INGOT_TYPE_MXFP4,  "MXFP4",  -1 },
    { INGOT_TYPE_NVFP4,  "NVFP4",  -1 },
};
static const size_t NFORMAT = sizeof(FORMATS) / sizeof(FORMATS[0]);

/* Fill `bytes` of block data for `type` covering rows x cols elements, either
 * by quantizing `reference` or, when there is no encoder, with deterministic
 * pseudo-random bytes whose f16 scale fields are kept in a moderate range so
 * no inf or nan reaches the comparison. */
static int fill_blocks(int type, const float *reference, size_t nelem,
                       unsigned char *w, size_t bytes, int can_encode) {
    if (can_encode) return ingot_quantize(type, reference, nelem, w);
    uint64_t blk_elems, blk_bytes;
    if (ingot_type_geometry(type, &blk_elems, &blk_bytes) != 0) return -1;
    for (size_t i = 0; i < bytes; i++) {
        rng = rng * 1664525u + 1013904223u;
        w[i] = (unsigned char)(rng >> 24);
    }
    for (size_t b = 0; b * blk_bytes < bytes; b++)
        for (uint64_t k = 1; k < blk_bytes; k += 2)
            w[b * blk_bytes + k] = (unsigned char)(w[b * blk_bytes + k] & 0x3f);
    /* Taming the odd bytes covers the f16 scale of most formats but not, say,
     * MXFP4's E8M0 exponent byte at offset 0, where 0xFF means 2^127. Rather
     * than special-case each layout, re-roll any block whose decode is not
     * finite — the same trick tools/gen_reference.py uses. */
    float *probe = (float *)malloc((size_t)blk_elems * sizeof(float));
    if (probe == NULL) return -1;
    for (size_t b = 0; b * blk_bytes < bytes; b++) {
        unsigned char *blk = w + b * blk_bytes;
        for (int attempt = 0; attempt < 64; attempt++) {
            int finite = ingot_dequant(type, blk, (size_t)blk_elems, probe) == 0;
            for (uint64_t i = 0; i < blk_elems && finite; i++)
                if (!(probe[i] == probe[i]) || probe[i] > 1e30f || probe[i] < -1e30f)
                    finite = 0;
            if (finite) break;
            for (uint64_t k = 0; k < blk_bytes; k++) {
                rng = rng * 1664525u + 1013904223u;
                blk[k] = (unsigned char)((rng >> 24) & (k % 2 ? 0x3fu : 0xffu));
            }
        }
    }
    free(probe);
    return 0;
}

static void test_roundtrip(void) {
    printf("round trip: quantize -> dequantize\n");
    const size_t n = 4096;
    float *values = malloc(n * sizeof(float));
    float *decoded = malloc(n * sizeof(float));
    for (size_t i = 0; i < n; i++) values[i] = rnd_bell();

    for (size_t f = 0; f < NFORMAT; f++) {
        if (FORMATS[f].budget < 0) continue;
        CHECK(ingot_can_quantize(FORMATS[f].type), "%s reports an encoder", FORMATS[f].name);
        uint64_t bytes = 0;
        if (ingot_type_nbytes(FORMATS[f].type, n, &bytes) != 0) {
            CHECK(0, "%s byte size", FORMATS[f].name);
            continue;
        }
        unsigned char *w = malloc((size_t)bytes);
        const int rc = ingot_quantize(FORMATS[f].type, values, n, w) == 0 &&
                       ingot_dequant(FORMATS[f].type, w, n, decoded) == 0;
        const double rel = rc ? rel_l2(decoded, values, n) : 1e9;
        CHECK(rc && rel < FORMATS[f].budget, "%-6s rel_l2 %.5f  (budget %.5f)",
              FORMATS[f].name, rel, FORMATS[f].budget);
        free(w);
    }
    free(values);
    free(decoded);
}

static void test_generic_matvec(void) {
    printf("ingot_matvec(type, ...) vs dequant-then-dot, every decodable type\n");
    const size_t rows = 5, cols = 512;
    float *reference = malloc(rows * cols * sizeof(float));
    float *decoded = malloc(rows * cols * sizeof(float));
    float *x = malloc(cols * sizeof(float));
    float *y = malloc(rows * sizeof(float));
    for (size_t i = 0; i < cols; i++) x[i] = rnd_unit();

    for (size_t f = 0; f < NFORMAT; f++) {
        const int type = FORMATS[f].type;
        const int can_encode = FORMATS[f].budget >= 0;
        uint64_t bytes = 0;
        if (ingot_type_nbytes(type, rows * cols, &bytes) != 0) {
            CHECK(0, "%s byte size", FORMATS[f].name);
            continue;
        }
        unsigned char *w = malloc((size_t)bytes);
        for (size_t i = 0; i < rows * cols; i++) reference[i] = rnd_bell();
        if (fill_blocks(type, reference, rows * cols, w, (size_t)bytes, can_encode) != 0) {
            CHECK(0, "%s could not be filled", FORMATS[f].name);
            free(w);
            continue;
        }

        const int rc = ingot_dequant_matrix(type, w, rows, cols, decoded) == 0 &&
                       ingot_matvec(type, w, rows, cols, x, y) == 0;
        double worst = 0;
        int finite = rc;
        for (size_t r = 0; r < rows && finite; r++) {
            double manual = 0;
            for (size_t i = 0; i < cols; i++)
                manual += (double)decoded[r * cols + i] * (double)x[i];
            const double d = fabs((double)y[r] - manual);
            const double sc = fmax(1.0, fabs(manual));
            if (d / sc > worst) worst = d / sc;
            if (!(y[r] == y[r])) finite = 0;
        }
        /* 1e-4 is an f32-accumulation budget, not a bit-exactness claim: the
         * kernel sums in f32 in its own order while the reference here sums in
         * double, and with 512 columns of random block data that parts company
         * around 1e-5. It is still three to seven orders of magnitude below
         * what a layout bug produces — the Q6_K one this suite caught was off
         * by 6017 in absolute terms. */
        CHECK(finite && worst < 1e-4, "%-6s %s   worst rel %.2e", FORMATS[f].name,
              ingot_has_kernel(type) ? "[kernel] " : "[generic]", worst);
        free(w);
    }
    free(reference); free(decoded); free(x); free(y);
}

static void test_generic_matmat(void) {
    printf("ingot_matmat(type, ...) vs per-token matvec\n");
    /* Awkward on purpose: 13 rows, 7 tokens, 768 columns (3 super-blocks). */
    const size_t rows = 13, cols = 768, tokens = 7;
    float *reference = malloc(rows * cols * sizeof(float));
    float *x = malloc(tokens * cols * sizeof(float));
    float *batched = malloc(tokens * rows * sizeof(float));
    float *single = malloc(tokens * rows * sizeof(float));
    for (size_t i = 0; i < tokens * cols; i++) x[i] = rnd_unit();

    for (size_t f = 0; f < NFORMAT; f++) {
        const int type = FORMATS[f].type;
        const int can_encode = FORMATS[f].budget >= 0;
        uint64_t bytes = 0;
        if (ingot_type_nbytes(type, rows * cols, &bytes) != 0) continue;
        unsigned char *w = malloc((size_t)bytes);
        for (size_t i = 0; i < rows * cols; i++) reference[i] = rnd_bell();
        if (fill_blocks(type, reference, rows * cols, w, (size_t)bytes, can_encode) != 0) {
            free(w);
            continue;
        }
        int rc = 1;
        for (size_t t = 0; t < tokens; t++)
            rc &= ingot_matvec(type, w, rows, cols, x + t * cols, single + t * rows) == 0;
        rc &= ingot_matmat(type, w, rows, cols, x, batched, tokens) == 0;

        /* The int8 activation path only exists for Q4_K/Q5_K; everything else
         * is an exact reorder. Ask, do not guess. */
        const int exact = ingot_matmat_is_exact(tokens) ||
                          (type != INGOT_TYPE_Q4_K && type != INGOT_TYPE_Q5_K);
        /* 5e-3 is the int8-activation budget on REAL weights; these blocks are
         * pseudo-random, so their dynamic range is worse than anything a
         * trained matrix has and the per-block int8 step is coarser. 2e-2
         * keeps the check meaningful without pretending random data behaves
         * like a model. */
        const double budget = exact ? 1e-5 : 2e-2;
        const double rel = rc ? rel_l2(batched, single, tokens * rows) : 1e9;
        CHECK(rc && rel < budget, "%-6s rel_l2 %.2e < %.0e (%s)", FORMATS[f].name,
              rel, budget, exact ? "exact" : "int8");
        free(w);
    }
    free(reference); free(x); free(batched); free(single);
}

static void test_unsupported(void) {
    printf("formats deliberately left out fail by name, not by surprise\n");
    /* One type is left: llama.cpp's reference package has no decoder for Q1_0
     * either, so implementing it would be guesswork with nothing to check
     * against. It keeps its geometry so a file using it still opens. */
    static const int absent[] = { INGOT_TYPE_Q1_0 };
    unsigned char block[512] = {0};
    float out[256] = {0}, x[256] = {0};
    for (size_t i = 0; i < sizeof(absent) / sizeof(absent[0]); i++) {
        uint64_t e, b;
        const int has_geometry = ingot_type_geometry(absent[i], &e, &b) == 0;
        const int named = strcmp(ingot_type_name(absent[i]), "UNKNOWN") != 0;
        uint64_t bs = 0;
        ingot_type_geometry(absent[i], &bs, NULL);
        const size_t n = (size_t)bs;
        const int refuses = ingot_type_can_dequant(absent[i]) == 0 &&
                            ingot_dequant(absent[i], block, n, out) != 0 &&
                            ingot_matvec(absent[i], block, 1, n, x, out) != 0 &&
                            ingot_quantize(absent[i], out, n, block) != 0;
        CHECK(has_geometry && named && refuses,
              "%-8s: geometry known, named, every decode path refuses",
              ingot_type_name(absent[i]));
    }
}

/* ── the two decoder lineages against each other ────────────────────────────
 * src/dequant.c and src/kernels.c hold INDEPENDENT decoders for the same six
 * K-quants: the scalar reference and the kernel's own fast decode. Two
 * implementations written from the same spec by different hands agreeing
 * bit-for-bit is real evidence; a single implementation agreeing with itself
 * is not.
 *
 * This is the check that caught a Q6_K decoder reading `d` out of the first
 * two bytes when ggml puts it in the last two, and walking the quants linearly
 * instead of in the two-halves-of-128 interleave — a misreading that still
 * fits in exactly 210 bytes, so nothing else would have found it. Keep this
 * test. */
static void test_decoder_lineages(void) {
    printf("dequant.c vs kernels.c: two lineages, same answer\n");
    static const struct { int type; const char *name;
                          int (*kernel_dequant)(const void *, size_t, size_t, float *); } pairs[] = {
        { INGOT_TYPE_Q2_K, "Q2_K", ingot_q2_k_dequant },
        { INGOT_TYPE_Q3_K, "Q3_K", ingot_q3_k_dequant },
        { INGOT_TYPE_Q4_K, "Q4_K", ingot_q4_k_dequant },
        { INGOT_TYPE_Q5_K, "Q5_K", ingot_q5_k_dequant },
        { INGOT_TYPE_Q6_K, "Q6_K", ingot_q6_k_dequant },
        { INGOT_TYPE_Q8_0, "Q8_0", ingot_q8_0_dequant },
    };
    const size_t rows = 4, cols = 512, n = rows * cols;
    float *a = malloc(n * sizeof(float));
    float *b = malloc(n * sizeof(float));
    for (size_t f = 0; f < sizeof(pairs) / sizeof(pairs[0]); f++) {
        uint64_t bytes = 0, blk_elems = 0, blk_bytes = 0;
        ingot_type_nbytes(pairs[f].type, n, &bytes);
        ingot_type_geometry(pairs[f].type, &blk_elems, &blk_bytes);
        unsigned char *w = malloc((size_t)bytes);
        rng = 0x51ED270Bu;
        fill_blocks(pairs[f].type, NULL, n, w, (size_t)bytes, 0);
        const int rc = ingot_dequant(pairs[f].type, w, n, a) == 0 &&
                       pairs[f].kernel_dequant(w, rows, cols, b) == 0;
        size_t differing = 0;
        for (size_t i = 0; i < n && rc; i++) if (a[i] != b[i]) differing++;
        CHECK(rc && differing == 0, "%-5s identical in both decoders (%zu/%zu differ)",
              pairs[f].name, differing, n);
        free(w);
    }
    free(a);
    free(b);
}

int main(void) {
    test_decoder_lineages();
    test_roundtrip();
    test_generic_matvec();
    test_generic_matmat();
    test_unsupported();
    printf("\n%d checks, %d failures\n", checks, failures);
    return failures != 0;
}
