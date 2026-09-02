/* Compare one AMX B=32 call with two AMX B=16 calls. */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "../qwen_tts_kernels.h"

static uint32_t rng_state = 0x3c6ef35fu;

static float random_value(void) {
    rng_state = rng_state * 1664525u + 1013904223u;
    return ((float)(rng_state >> 8) / 16777216.0f) * 2.0f - 1.0f;
}

static void *aligned_zero(size_t bytes) {
    void *p = NULL;
    size_t size = (bytes + 63u) & ~63u;
    if (size == 0) size = 64;
    if (posix_memalign(&p, 64, size) != 0) return NULL;
    memset(p, 0, size);
    return p;
}

static double now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e9 + (double)ts.tv_nsec;
}

static double median(double *values, int count) {
    for (int i = 1; i < count; i++) {
        double value = values[i];
        int j = i;
        while (j > 0 && values[j - 1] > value) {
            values[j] = values[j - 1];
            j--;
        }
        values[j] = value;
    }
    return values[count / 2];
}

static double max_abs_diff(const float *a, const float *b, size_t n) {
    double max_err = 0.0;
    for (size_t i = 0; i < n; i++) {
        double err = fabs((double)a[i] - (double)b[i]);
        if (err > max_err) max_err = err;
    }
    return max_err;
}

static void split_inputs(float *x0, float *x1, const float *x, int cols) {
    for (int k = 0; k < cols; k++) {
        for (int b = 0; b < 16; b++) x0[(size_t)k * 16 + b] = x[(size_t)k * 32 + b];
        for (int b = 0; b < 16; b++) x1[(size_t)k * 16 + b] = x[(size_t)k * 32 + 16 + b];
    }
}

static void join_outputs(float *dst, const float *y0, const float *y1, int rows) {
    for (int r = 0; r < rows; r++) {
        for (int b = 0; b < 16; b++) dst[(size_t)r * 32 + b] = y0[(size_t)r * 16 + b];
        for (int b = 0; b < 16; b++) dst[(size_t)r * 32 + 16 + b] = y1[(size_t)r * 16 + b];
    }
}

int main(int argc, char **argv) {
    const int rows = argc > 1 ? atoi(argv[1]) : 2048;
    const int cols = argc > 2 ? atoi(argv[2]) : 2048;
    int reps = argc > 3 ? atoi(argv[3]) : 30;
    if (reps < 5) reps = 5;
    if (rows < 16 || (rows & 15) != 0 || cols < 64) {
        fprintf(stderr, "rows must be a positive multiple of 16 and cols must be at least 64\n");
        return 2;
    }
    if (setenv("QWEN_AMX_B32", "1", 1) != 0) return 2;
    qwen_set_threads(1);

    const size_t weights = (size_t)rows * cols;
    const size_t outputs = (size_t)rows * 32;
    int8_t *W = (int8_t *)aligned_zero(weights);
    float *scale = (float *)aligned_zero((size_t)rows * sizeof(float));
    float *x = (float *)aligned_zero((size_t)cols * 32 * sizeof(float));
    float *x0 = (float *)aligned_zero((size_t)cols * 16 * sizeof(float));
    float *x1 = (float *)aligned_zero((size_t)cols * 16 * sizeof(float));
    float *y0 = (float *)aligned_zero((size_t)rows * 16 * sizeof(float));
    float *y1 = (float *)aligned_zero((size_t)rows * 16 * sizeof(float));
    float *baseline = (float *)aligned_zero(outputs * sizeof(float));
    float *got = (float *)aligned_zero(outputs * sizeof(float));
    if (!W || !scale || !x || !x0 || !x1 || !y0 || !y1 || !baseline || !got) {
        fprintf(stderr, "allocation failed\n");
        return 2;
    }
    for (size_t i = 0; i < weights; i++) W[i] = (int8_t)(random_value() * 127.0f);
    for (int r = 0; r < rows; r++) scale[r] = 0.002f + 0.001f * fabsf(random_value());
    for (size_t i = 0; i < (size_t)cols * 32; i++) x[i] = random_value();
    split_inputs(x0, x1, x, cols);

    qwen_amx_weight_cache_reset();
    qwen_matmat_int8(y0, W, scale, x0, rows, cols, 16);
    qwen_matmat_int8(y1, W, scale, x1, rows, cols, 16);
    join_outputs(baseline, y0, y1, rows);
    qwen_amx_weight_cache_reset();
    int native = qwen_matmat_int8_amx_b32(got, W, scale, x, rows, cols);
    if (!native) {
        printf("amx-b32-bench path=not-selected\n");
        return 0;
    }
    const double err = max_abs_diff(got, baseline, outputs);
    if (err > 1e-5) {
        fprintf(stderr, "FAIL: max_abs=%.3e\n", err);
        return 1;
    }

    qwen_amx_weight_cache_reset();
    double t0 = now_ns();
    qwen_matmat_int8(y0, W, scale, x0, rows, cols, 16);
    qwen_matmat_int8(y1, W, scale, x1, rows, cols, 16);
    const double cold_b16 = now_ns() - t0;
    qwen_amx_weight_cache_reset();
    t0 = now_ns();
    qwen_matmat_int8_amx_b32(got, W, scale, x, rows, cols);
    const double cold_b32 = now_ns() - t0;

    qwen_amx_weight_cache_reset();
    qwen_matmat_int8(y0, W, scale, x0, rows, cols, 16);
    qwen_matmat_int8(y1, W, scale, x1, rows, cols, 16);
    qwen_matmat_int8_amx_b32(got, W, scale, x, rows, cols);
    double b16_samples[64], b32_samples[64];
    if (reps > 64) reps = 64;
    for (int i = 0; i < reps; i++) {
        t0 = now_ns();
        qwen_matmat_int8(y0, W, scale, x0, rows, cols, 16);
        qwen_matmat_int8(y1, W, scale, x1, rows, cols, 16);
        b16_samples[i] = now_ns() - t0;
        t0 = now_ns();
        qwen_matmat_int8_amx_b32(got, W, scale, x, rows, cols);
        b32_samples[i] = now_ns() - t0;
    }
    const double warm_b16 = median(b16_samples, reps);
    const double warm_b32 = median(b32_samples, reps);
    printf("amx-b32-bench rows=%d cols=%d reps=%d cold_two_b16_ns=%.0f cold_b32_ns=%.0f "
           "warm_two_b16_ns=%.0f warm_b32_ns=%.0f speedup=%.3fx max_abs=%.3e\n",
           rows, cols, reps, cold_b16, cold_b32, warm_b16, warm_b32,
           warm_b16 / warm_b32, err);

    free(W); free(scale); free(x); free(x0); free(x1);
    free(y0); free(y1); free(baseline); free(got);
    return 0;
}
