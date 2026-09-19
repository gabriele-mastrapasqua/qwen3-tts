/* Host-side parity smoke for the opt-in KleidiAI dotprod GEMV candidates.
 * This is independent of the model loader: it proves the vendor packer/kernel
 * ABI and arithmetic on a dotprod host before a rented Linux box is used for
 * complete-call measurements. */
#include "qwen_tts_kleidi.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int close_enough(float got, float ref) {
    return isfinite(got) && fabsf(got - ref) <= 0.25f * (fabsf(ref) + 1.0f);
}

static int q4_case(void) {
    const int rows = 7, cols = 64, nb = cols / 32;
    uint8_t *raw = (uint8_t *)calloc((size_t)rows * nb, 18);
    float *x = (float *)malloc((size_t)cols * sizeof *x);
    float *y = (float *)malloc((size_t)rows * sizeof *y);
    if (!raw || !x || !y) return 0;
    for (int k = 0; k < cols; k++) x[k] = (float)((k % 11) - 5) * 0.17f;
    for (int r = 0; r < rows; r++) {
        for (int b = 0; b < nb; b++) {
            uint8_t *blk = raw + ((size_t)r * nb + b) * 18;
            /* f16(1.0). */
            blk[0] = 0x00; blk[1] = 0x3c;
            for (int i = 0; i < 16; i++) {
                uint8_t lo = (uint8_t)((i + r + b) & 15);
                uint8_t hi = (uint8_t)((3 * i + r + b + 1) & 15);
                blk[2 + i] = (uint8_t)(lo | (hi << 4));
            }
        }
    }
    if (!qwen_kleidi_dotprod_register_q4(raw, raw, rows, cols) ||
        !qwen_kleidi_dotprod_matmul_q4(y, raw, x, rows, cols, 1)) {
        free(raw); free(x); free(y); return 0;
    }
    for (int r = 0; r < rows; r++) {
        float ref = 0.0f;
        for (int k = 0; k < cols; k++) {
            const uint8_t *blk = raw + ((size_t)r * nb + k / 32) * 18;
            int i = k % 32;
            int q = i < 16 ? (blk[2 + i] & 15) : (blk[2 + i - 16] >> 4);
            ref += (float)(q - 8) * x[k];
        }
        if (!close_enough(y[r], ref)) {
            fprintf(stderr, "dotprod q4 mismatch row=%d got=%g ref=%g\n", r, y[r], ref);
            free(raw); free(x); free(y); return 0;
        }
    }
    free(raw); free(x); free(y);
    return 1;
}

static int i8_case(void) {
    const int rows = 9, cols = 64;
    int8_t *w = (int8_t *)malloc((size_t)rows * cols);
    float *scale = (float *)malloc((size_t)rows * sizeof *scale);
    float *x = (float *)malloc((size_t)cols * sizeof *x);
    float *y = (float *)malloc((size_t)rows * sizeof *y);
    if (!w || !scale || !x || !y) return 0;
    for (int k = 0; k < cols; k++) x[k] = (float)((k % 13) - 6) * 0.11f;
    for (int r = 0; r < rows; r++) {
        scale[r] = 0.01f * (float)(r + 1);
        for (int k = 0; k < cols; k++) w[(size_t)r * cols + k] = (int8_t)(((r * 5 + k * 3) % 31) - 15);
    }
    if (!qwen_kleidi_dotprod_register_i8(w, w, scale, rows, cols) ||
        !qwen_kleidi_dotprod_matmul_i8(y, w, x, rows, cols, 1)) {
        free(w); free(scale); free(x); free(y); return 0;
    }
    for (int r = 0; r < rows; r++) {
        float ref = 0.0f;
        for (int k = 0; k < cols; k++) ref += (float)w[(size_t)r * cols + k] * scale[r] * x[k];
        if (!close_enough(y[r], ref)) {
            fprintf(stderr, "dotprod int8 mismatch row=%d got=%g ref=%g\n", r, y[r], ref);
            free(w); free(scale); free(x); free(y); return 0;
        }
    }
    free(w); free(scale); free(x); free(y);
    return 1;
}

int main(void) {
    if (!qwen_kleidi_dotprod_compiled() || !qwen_kleidi_dotprod_supported()) {
        puts("SKIP: dotprod runtime unavailable");
        return 0;
    }
    if (setenv("QWEN_KAI_DOTPROD_GEMV", "1", 1) != 0 ||
        !qwen_kleidi_dotprod_enabled()) {
        fprintf(stderr, "dotprod candidate did not enable\n");
        return 1;
    }
    if (!q4_case() || !i8_case()) return 1;
    puts("PASS: dotprod-only Q4/int8 GEMV pack and kernel parity");
    return 0;
}
