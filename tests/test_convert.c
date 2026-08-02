/* Bulk dtype conversions: the SIMD bodies must be byte-for-byte identical to
 * the per-element scalar functions. Widening conversions (BF16/F16 -> F32) are
 * exhaustive — all 65536 inputs — because they can be. The narrowing one
 * (F32 -> BF16) gets every edge class plus a deterministic pseudo-random
 * sweep. Odd lengths and misaligned sources exercise the vector tails.
 *
 * SPDX-License-Identifier: MIT */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "ingot/dtype.h"

/* Internal entry points (src/internal.h): exported symbols, not public API. */
void ingot_bf16_block_to_f32(const unsigned char *p, size_t nelem, float *dst);
void ingot_f16_block_to_f32(const unsigned char *p, size_t nelem, float *dst);
void ingot_f32_block_to_bf16(const float *src, size_t nelem, unsigned char *dst);

static int failures;
static int checks;

#define CHECK(cond, ...) do {                                        \
    checks++;                                                        \
    if (!(cond)) { printf("  FAIL: "); printf(__VA_ARGS__);          \
                   printf("  (%s:%d)\n", __FILE__, __LINE__);        \
                   failures++; }                                     \
    else { printf("  ok:   "); printf(__VA_ARGS__); printf("\n"); }  \
} while (0)

static uint32_t f32_bits(float f) {
    uint32_t b;
    memcpy(&b, &f, sizeof(b));
    return b;
}

/* Deterministic 32-bit LCG: the sweep must be reproducible in a bug report. */
static uint32_t lcg(uint32_t *state) {
    *state = *state * 1664525u + 1013904223u;
    return *state;
}

static void exhaustive_widening(void) {
    printf("widening conversions, all 65536 inputs\n");
    enum { N = 65536 };
    unsigned char *src = malloc(N * 2);
    float *simd  = malloc(N * sizeof(float));
    float *ref   = malloc(N * sizeof(float));
    for (int i = 0; i < N; i++) {
        src[2 * i]     = (unsigned char)(i & 0xff);
        src[2 * i + 1] = (unsigned char)(i >> 8);
    }

    ingot_bf16_block_to_f32(src, N, simd);
    for (int i = 0; i < N; i++) ref[i] = ingot_bf16_to_f32((uint16_t)i);
    CHECK(memcmp(simd, ref, N * sizeof(float)) == 0,
          "BF16->F32 bit-identical on every input");

    ingot_f16_block_to_f32(src, N, simd);
    for (int i = 0; i < N; i++) ref[i] = ingot_f16_to_f32((uint16_t)i);
    int mismatch = -1;
    for (int i = 0; i < N; i++)
        if (f32_bits(simd[i]) != f32_bits(ref[i])) { mismatch = i; break; }
    CHECK(mismatch < 0,
          "F16->F32 bit-identical on every input (subnormals, inf, NaN)");
    if (mismatch >= 0)
        printf("        first mismatch at f16 0x%04x: simd %08x ref %08x\n",
               mismatch, f32_bits(simd[mismatch]), f32_bits(ref[mismatch]));

    free(src); free(simd); free(ref);
}

static void narrowing_bf16(void) {
    printf("F32->BF16\n");
    /* Every class an RNE narrowing has to get right. */
    static const uint32_t edges[] = {
        0x00000000u, 0x80000000u,             /* +0 -0                       */
        0x00000001u, 0x807fffffu,             /* smallest subnormals         */
        0x00800000u, 0x80800000u,             /* smallest normals            */
        0x7f7fffffu, 0xff7fffffu,             /* largest finite (rounds inf) */
        0x7f800000u, 0xff800000u,             /* inf                         */
        0x7fc00000u, 0x7f800001u,             /* qNaN, sNaN (must stay NaN)  */
        0xffc12345u, 0x7f812345u,             /* NaN payloads                */
        0x3f800000u, 0x3f808000u,             /* 1.0, exact tie -> even      */
        0x3f818000u, 0x3f807fffu,             /* tie -> odd rounds up, below */
        0x40490fdbu, 0xc0490fdbu,             /* pi                          */
        0x00008000u, 0x00018000u,             /* subnormal ties              */
    };
    enum { NEDGE = sizeof(edges) / sizeof(edges[0]), NRAND = 1 << 18 };
    const size_t n = NEDGE + NRAND;
    float *src = malloc(n * sizeof(float));
    unsigned char *simd = malloc(n * 2);
    unsigned char *ref  = malloc(n * 2);
    for (size_t i = 0; i < NEDGE; i++) memcpy(&src[i], &edges[i], 4);
    uint32_t state = 0x1234567u;
    for (size_t i = NEDGE; i < n; i++) {
        const uint32_t b = lcg(&state);
        memcpy(&src[i], &b, 4);
    }

    ingot_f32_block_to_bf16(src, n, simd);
    for (size_t i = 0; i < n; i++) {
        const uint16_t h = ingot_f32_to_bf16(src[i]);
        ref[2 * i]     = (unsigned char)(h & 0xff);
        ref[2 * i + 1] = (unsigned char)(h >> 8);
    }
    size_t mismatch = n;
    for (size_t i = 0; i < n; i++)
        if (simd[2 * i] != ref[2 * i] || simd[2 * i + 1] != ref[2 * i + 1]) {
            mismatch = i; break;
        }
    CHECK(mismatch == n,
          "byte-identical on %zu inputs (edges + LCG sweep)", n);
    if (mismatch < n)
        printf("        first mismatch at %zu (f32 %08x): simd %02x%02x ref %02x%02x\n",
               mismatch, f32_bits(src[mismatch]),
               simd[2 * mismatch + 1], simd[2 * mismatch],
               ref[2 * mismatch + 1], ref[2 * mismatch]);

    free(src); free(simd); free(ref);
}

static void tails_and_alignment(void) {
    printf("vector tails and misaligned sources\n");
    /* A buffer deliberately used at +1: mmap'd tensor payloads are aligned,
     * but nothing in the API promises it, and vld1q/loadu must not care. */
    enum { MAX = 70 };
    unsigned char raw[1 + MAX * 2];
    float out[MAX], ref[MAX];
    uint32_t state = 0xdeadbeefu;
    for (size_t i = 0; i < sizeof(raw); i++) raw[i] = (unsigned char)lcg(&state);
    const unsigned char *p = raw + 1;

    int bad = 0;
    for (size_t n = 0; n <= MAX && !bad; n++) {
        memset(out, 0xa5, sizeof(out));
        ingot_bf16_block_to_f32(p, n, out);
        for (size_t i = 0; i < n; i++)
            ref[i] = ingot_bf16_to_f32((uint16_t)(p[2 * i] | (p[2 * i + 1] << 8)));
        if (memcmp(out, ref, n * sizeof(float)) != 0) bad = 1;

        memset(out, 0xa5, sizeof(out));
        ingot_f16_block_to_f32(p, n, out);
        for (size_t i = 0; i < n; i++)
            ref[i] = ingot_f16_to_f32((uint16_t)(p[2 * i] | (p[2 * i + 1] << 8)));
        for (size_t i = 0; i < n; i++)
            if (f32_bits(out[i]) != f32_bits(ref[i])) bad = 1;
    }
    CHECK(!bad, "every length 0..%d from an odd address matches", MAX);

    /* The narrowing twin, same treatment, unaligned destination. */
    float fsrc[MAX];
    unsigned char draw[1 + MAX * 2 + 1], dref[MAX * 2];   /* +1: the sentinel */
    for (size_t i = 0; i < MAX; i++) {
        const uint32_t b = lcg(&state);
        memcpy(&fsrc[i], &b, 4);
    }
    bad = 0;
    for (size_t n = 0; n <= MAX && !bad; n++) {
        memset(draw, 0x5a, sizeof(draw));
        ingot_f32_block_to_bf16(fsrc, n, draw + 1);
        for (size_t i = 0; i < n; i++) {
            const uint16_t h = ingot_f32_to_bf16(fsrc[i]);
            dref[2 * i]     = (unsigned char)(h & 0xff);
            dref[2 * i + 1] = (unsigned char)(h >> 8);
        }
        if (memcmp(draw + 1, dref, n * 2) != 0) bad = 1;
        /* and the byte after the write must be untouched */
        if (draw[1 + n * 2] != 0x5a) bad = 1;
    }
    CHECK(!bad, "F32->BF16 every length 0..%d, no overrun past the end", MAX);
}

static void through_the_public_api(void) {
    printf("through ingot_dtype_to_f32\n");
    const uint16_t vals[5] = { 0x3f80, 0x0001, 0x8000, 0x7f81, 0xff80 };
    unsigned char bytes[10];
    for (int i = 0; i < 5; i++) {
        bytes[2 * i]     = (unsigned char)(vals[i] & 0xff);
        bytes[2 * i + 1] = (unsigned char)(vals[i] >> 8);
    }
    float out[5];
    CHECK(ingot_dtype_to_f32(INGOT_DT_BF16, bytes, 5, out) == 0 &&
              f32_bits(out[0]) == 0x3f800000u,
          "BF16 case routes through the block converter");
    CHECK(ingot_dtype_to_f32(INGOT_DT_F16, bytes, 5, out) == 0 &&
              f32_bits(out[0]) == 0x3ff00000u,
          "F16 case routes through the block converter");
}

int main(void) {
    exhaustive_widening();
    narrowing_bf16();
    tails_and_alignment();
    through_the_public_api();
    printf("\n%d checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
}
