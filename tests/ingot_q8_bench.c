/* ingot_q8_bench.c — is it worth wiring the vendored Q8_0 into the Talker? */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>

#include "ingot/dtype.h"
#include "ingot/quant.h"
#include "qwen_tts_kernels.h"

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static uint16_t f32_to_bf16(float f) {
    uint32_t b; memcpy(&b, &f, 4);
    return (uint16_t)((b + 0x7fff + ((b >> 16) & 1)) >> 16);
}
static float bf16_to_f32(uint16_t h) {
    uint32_t b = (uint32_t)h << 16; float f; memcpy(&f, &b, 4); return f;
}

static double rel_err(const float *y, const float *ref, int n) {
    double num = 0, den = 0;
    for (int i = 0; i < n; i++) { double d = y[i] - ref[i]; num += d * d; den += (double)ref[i] * ref[i]; }
    return den > 0 ? sqrt(num / den) : 0.0;
}

static void bench_shape(int rows, int cols, int iters) {
    printf("\n=== %d x %d  (%.1f MB in bf16)\n", rows, cols,
           (double)rows * cols * 2 / 1e6);

    size_t n = (size_t)rows * cols;
    uint16_t *Wb  = malloc(n * sizeof(uint16_t));
    float    *Wf  = malloc(n * sizeof(float));
    int8_t   *Wi  = malloc(n);
    float    *Ws  = malloc((size_t)rows * sizeof(float));
    float    *x   = malloc((size_t)cols * sizeof(float));
    float    *y_ref = malloc((size_t)rows * sizeof(float));
    float    *y_i8  = malloc((size_t)rows * sizeof(float));
    float    *y_q8  = malloc((size_t)rows * sizeof(float));
    if (!Wb || !Wf || !Wi || !Ws || !x || !y_ref || !y_i8 || !y_q8) { printf("OOM\n"); return; }

    srand(1234);
    for (size_t i = 0; i < n; i++) {
        double u = (double)rand() / RAND_MAX - 0.5;
        double v = u * 0.05;
        if ((rand() % 1000) == 0) v *= 20.0;
        Wb[i] = f32_to_bf16((float)v);
        Wf[i] = bf16_to_f32(Wb[i]);
    }
    for (int i = 0; i < cols; i++) x[i] = (float)((double)rand() / RAND_MAX - 0.5);

    for (int r = 0; r < rows; r++) {
        double s = 0;
        for (int c = 0; c < cols; c++) s += (double)Wf[(size_t)r * cols + c] * x[c];
        y_ref[r] = (float)s;
    }

    qwen_quantize_bf16_to_int8(Wb, rows, cols, Wi, Ws);

    uint64_t q8_bytes = 0;
    if (ingot_type_nbytes(INGOT_TYPE_Q8_0, n, &q8_bytes) != 0) { printf("nbytes fallito\n"); return; }
    void *Wq8 = malloc((size_t)q8_bytes);
    if (!Wq8) { printf("OOM q8\n"); return; }
    if (ingot_quantize(INGOT_TYPE_Q8_0, Wf, n, Wq8) != 0) { printf("quantize fallito\n"); return; }

    printf("  byte/peso   int8 %.4f   Q8_0 %.4f  (+%.1f%%)\n",
           1.0, (double)q8_bytes / n, ((double)q8_bytes / n - 1.0) * 100.0);
    printf("  kernel Q8_0 dedicato: %s\n", ingot_has_kernel(INGOT_TYPE_Q8_0) ? "si" : "NO (decode riga per riga)");

    qwen_matvec_int8(y_i8, Wi, Ws, x, rows, cols);
    ingot_matvec(INGOT_TYPE_Q8_0, Wq8, (size_t)rows, (size_t)cols, x, y_q8);

    double t0 = now_s();
    for (int i = 0; i < iters; i++) qwen_matvec_int8(y_i8, Wi, Ws, x, rows, cols);
    double t_i8 = (now_s() - t0) / iters;

    t0 = now_s();
    for (int i = 0; i < iters; i++) ingot_matvec(INGOT_TYPE_Q8_0, Wq8, (size_t)rows, (size_t)cols, x, y_q8);
    double t_q8 = (now_s() - t0) / iters;

    printf("  tempo       int8 %8.3f us   Q8_0 %8.3f us   -> %s %.2fx\n",
           t_i8 * 1e6, t_q8 * 1e6,
           t_q8 < t_i8 ? "Q8_0 piu' veloce" : "int8 piu' veloce",
           t_q8 < t_i8 ? t_i8 / t_q8 : t_q8 / t_i8);
    printf("  errore rel. int8 %.3e        Q8_0 %.3e    -> %s\n",
           rel_err(y_i8, y_ref, rows), rel_err(y_q8, y_ref, rows),
           rel_err(y_q8, y_ref, rows) < rel_err(y_i8, y_ref, rows) ? "Q8_0 piu' preciso" : "int8 piu' preciso");

    free(Wb); free(Wf); free(Wi); free(Ws); free(x);
    free(y_ref); free(y_i8); free(y_q8); free(Wq8);
}

int main(int argc, char **argv) {
    int iters = argc > 1 ? atoi(argv[1]) : 200;
    int threads = argc > 2 ? atoi(argv[2]) : 1;

    qwen_set_threads(threads);

    printf("matvec: motore int8 (scala per riga) contro ingot Q8_0 (scala per blocco di 32)\n");
    printf("%d iterazioni per cella · motore a %d thread · ingot single-thread (by design)\n",
           iters, threads);
    if (threads != 1)
        printf("⚠️  confronto NON a parita' di core: rilancia con '%d 1' per il kernel puro\n", iters);

    bench_shape(2048, 2048, iters);
    bench_shape(12288, 2048, iters);
    bench_shape(2048, 6144, iters);
    return 0;
}
