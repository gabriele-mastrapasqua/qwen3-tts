/* ingot_q8_bench.c — vale la pena cablare Q8_0 di ingot nel Talker? (task T2)
 *
 * LA DOMANDA CORTA CHE PRECEDE QUELLA LUNGA. Il matvec int8 del motore converte ogni
 * peso a f32 e fa una FMA in virgola mobile (qwen_tts_kernels.c:2270-2310:
 * vcvtq_f32_s32 + vfmaq_f32) — cioe' butta via il vantaggio dell'int8. ingot ha un
 * Q8_0 con kernel NEON/AVX2/SDOT/SMMLA/VNNI verificati contro llama.cpp. Prima di
 * riscrivere il caricamento e il dispatch del Talker, si misura il kernel da solo:
 * se non e' piu' veloce, il cablaggio e' lavoro buttato.
 *
 * DUE ASSI, non uno. La velocita' da sola non basta: i due formati non hanno la stessa
 * granularita' di scala, quindi vanno confrontati anche in ERRORE contro la stessa
 * reference f32.
 *
 *   motore int8   una scala per RIGA         (max|W_row|/127)      1,00 byte/peso
 *   ingot Q8_0    una scala ogni 32 pesi     (blocchi da 34 byte)  1,0625 byte/peso
 *
 * Q8_0 costa il 6% di byte in piu' e ha una granularita' di scala 64x piu' fine. Se e'
 * anche piu' veloce, non c'e' motivo di tenersi quello a righe.
 *
 * Shape reali del Talker 1.7B (hidden 2048): la proiezione quadrata e la FFN, che sono
 * i due regimi diversi (la seconda e' quella che domina la banda).
 *
 * Build:  make ingot-q8-bench     (oppure la riga in fondo a questo commento)
 *   cc -O3 -march=native -Ithird_party/ingot/include -I. tests/ingot_q8_bench.c \
 *      qwen_tts_kernels.o qwen_tts_kernels_neon.o qwen_tts_kernels_generic.o \
 *      qwen_tts_kernels_avx.o qwen_tts_thread.o third_party/ingot/libingot.a \
 *      -o ingot_q8_bench -lm -lpthread
 */
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
    return (uint16_t)((b + 0x7fff + ((b >> 16) & 1)) >> 16);   /* round-to-nearest-even */
}
static float bf16_to_f32(uint16_t h) {
    uint32_t b = (uint32_t)h << 16; float f; memcpy(&f, &b, 4); return f;
}

/* errore relativo L2 fra un risultato e la reference f32 */
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

    /* pesi con una coda: e' l'outlier per riga che punisce la scala per-riga, ed e'
     * realistico — le matrici di un transformer hanno pochi valori molto grandi. */
    srand(1234);
    for (size_t i = 0; i < n; i++) {
        double u = (double)rand() / RAND_MAX - 0.5;
        double v = u * 0.05;
        if ((rand() % 1000) == 0) v *= 20.0;        /* 0,1% di outlier */
        Wb[i] = f32_to_bf16((float)v);
        Wf[i] = bf16_to_f32(Wb[i]);                 /* stessa sorgente per entrambi */
    }
    for (int i = 0; i < cols; i++) x[i] = (float)((double)rand() / RAND_MAX - 0.5);

    /* reference esatta in f32, dalla stessa sorgente bf16 */
    for (int r = 0; r < rows; r++) {
        double s = 0;
        for (int c = 0; c < cols; c++) s += (double)Wf[(size_t)r * cols + c] * x[c];
        y_ref[r] = (float)s;
    }

    /* --- motore: int8 per-riga --- */
    qwen_quantize_bf16_to_int8(Wb, rows, cols, Wi, Ws);

    /* --- ingot: Q8_0 per-blocco-32 --- */
    uint64_t q8_bytes = 0;
    if (ingot_type_nbytes(INGOT_TYPE_Q8_0, n, &q8_bytes) != 0) { printf("nbytes fallito\n"); return; }
    void *Wq8 = malloc((size_t)q8_bytes);
    if (!Wq8) { printf("OOM q8\n"); return; }
    if (ingot_quantize(INGOT_TYPE_Q8_0, Wf, n, Wq8) != 0) { printf("quantize fallito\n"); return; }

    printf("  byte/peso   int8 %.4f   Q8_0 %.4f  (+%.1f%%)\n",
           1.0, (double)q8_bytes / n, ((double)q8_bytes / n - 1.0) * 100.0);
    printf("  kernel Q8_0 dedicato: %s\n", ingot_has_kernel(INGOT_TYPE_Q8_0) ? "si" : "NO (decode riga per riga)");

    /* warm-up (entrambi, così la cache non favorisce il primo) */
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

    /* PARITA' DI CORE, altrimenti non si misura il kernel ma il thread pool.
     * ingot_matvec e' single-thread PER SCELTA (quant.h:117: "matvec is
     * single-threaded like the quantized ones; matmat parallelizes over rows
     * through ingot_set_parallel_for") — ingot non possiede un pool apposta,
     * se lo fa iniettare dal consumatore. Il matvec int8 del motore invece
     * si sparpaglia su g_n_threads. Confrontarli a thread diversi misura
     * quanti core ha il motore, che non e' la domanda. */
    qwen_set_threads(threads);

    printf("matvec: motore int8 (scala per riga) contro ingot Q8_0 (scala per blocco di 32)\n");
    printf("%d iterazioni per cella · motore a %d thread · ingot single-thread (by design)\n",
           iters, threads);
    if (threads != 1)
        printf("⚠️  confronto NON a parita' di core: rilancia con '%d 1' per il kernel puro\n", iters);

    /* le shape vere del Talker 1.7B: hidden 2048, FFN intermedia 6144 */
    bench_shape(2048, 2048, iters);    /* wo / qkv: la proiezione quadrata */
    bench_shape(12288, 2048, iters);   /* gate_up fuso: 2 x 6144 */
    bench_shape(2048, 6144, iters);    /* down: quella che domina la banda */
    return 0;
}
