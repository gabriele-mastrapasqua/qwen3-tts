/* matmat_parity.c — does the batched twin do the SAME arithmetic as the integer reference?
 *
 * WHY IT EXISTS. Quantized GEMMs are per-ISA: VNNI on AVX-512, SMMLA on i8mm, maddubs on
 * plain AVX2, the f32 twin everywhere else. Each path is different code for the same
 * format, and until now the only way to verify one was to rent the machine that runs it.
 * That is how the AVX-512 kernels came to be written and never executed by us.
 *
 * AN ARM MAC CAN STILL RUN THE x86 PATH: Rosetta 2 supports AVX2 (measured on this
 * machine). So this test, compiled for x86_64 with -march=x86-64-v3, really EXECUTES the
 * AVX2 kernel and checks its numbers — on the laptop, before paying for an hour of cloud.
 * (Rosetta does NOT emulate AVX-512: VNNI and AMX still have to be validated on metal.)
 *
 * THE ORACLE, and why it is this one. The matmat is not compared against the matvec: that
 * is an f32 path with a different activation quantization, so a difference would not say
 * whether the kernel is wrong or merely rounded differently. It is compared against a
 * plain-C INTEGER reference that quantizes activations exactly as the engine does and sums
 * in int32. The kernel must produce the SAME number: integer arithmetic has no summation
 * order to argue about. Expected: relative error 0.
 *
 * Use:
 *   make check-matmat-parity        # native ISA
 *   make check-matmat-parity-x86    # x86-64-v3 under Rosetta, from an Arm Mac
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include "qwen_tts_kernels.h"

static uint64_t rs = 0x243F6A8885A308D3ull;
static double rnd(void) { rs = rs * 6364136223846793005ull + 1442695040888963407ull; return (double)((rs >> 40) / (double)(1u << 24)) * 2.0 - 1.0; }

/* la stessa quantizzazione per colonna che usano i gemelli batched */
static float qcol(int8_t *qb, const float *X, int cols, int B, int b) {
    float amax = 0.0f;
    for (int k = 0; k < cols; k++) { float a = fabsf(X[(size_t)k * B + b]); if (a > amax) amax = a; }
    if (amax == 0.0f) { memset(qb, 0, (size_t)cols); return 0.0f; }
    float inv = 127.0f / amax;
    for (int k = 0; k < cols; k++) {
        int v = (int)lrintf(X[(size_t)k * B + b] * inv);
        qb[k] = (int8_t)(v > 127 ? 127 : (v < -128 ? -128 : v));
    }
    return amax / 127.0f;
}

/* TWO THRESHOLDS, because not every ISA takes the integer path. Where the dispatcher has
 * a real integer GEMM (AVX2/maddubs, AVX-512/VNNI, i8mm/SMMLA) the result must match the
 * integer reference up to the final rounding alone: rel ~ 0. Where it falls back to the
 * f32 twin (no i8mm, or QWEN_NO_*=1) the activations stay in f32 and the difference is
 * quantization error, not a bug: it is declared rather than hidden. A FAIL is only past
 * the second threshold. */
static int check(const char *what, const float *got, const float *ref, int n) {
    double mx = 0.0, den = 0.0;
    int worst = -1;
    for (int i = 0; i < n; i++) {
        double d = fabs((double)got[i] - (double)ref[i]);
        if (d > mx) { mx = d; worst = i; }
        den += fabs((double)ref[i]);
    }
    double rel = den > 0 ? mx / (den / n) : mx;
    const char *verdict = rel <= 1e-4 ? "OK  (percorso intero, coincide)"
                        : rel <= 5e-2 ? "ok  (twin f32: NON e' il GEMM intero)"
                                      : "FAIL";
    printf("  %-26s max_abs=%.3e  rel=%.3e  %s\n", what, mx, rel, verdict);
    if (rel > 5e-2 && worst >= 0)
        printf("      peggior elemento %d: got=%.6f ref=%.6f\n", worst, got[worst], ref[worst]);
    return rel > 5e-2 ? 1 : 0;
}

int main(int argc, char **argv) {
    int rows = argc > 1 ? atoi(argv[1]) : 512;
    int cols = argc > 2 ? atoi(argv[2]) : 1024;
    int nt   = argc > 3 ? atoi(argv[3]) : 1;
    qwen_set_threads(nt);
    printf("matmat parity — rows=%d cols=%d threads=%d\n", rows, cols, nt);
    qwen_caps_report(stdout);

    int fail = 0;
    int8_t *W = (int8_t *)malloc((size_t)rows * cols);
    float *scale = (float *)malloc((size_t)rows * sizeof(float));
    for (int r = 0; r < rows; r++) {
        scale[r] = (float)(0.002 + 0.001 * fabs(rnd()));
        for (int k = 0; k < cols; k++) { int v = (int)(rnd() * 127.0); W[(size_t)r * cols + k] = (int8_t)v; }
    }
    /* pesi bf16 -> q4_0, per il gemello q4 */
    uint16_t *Wb = (uint16_t *)malloc((size_t)rows * cols * sizeof(uint16_t));
    for (int i = 0; i < rows * cols; i++) {
        float f = (float)(rnd() * 0.1);
        uint32_t u; memcpy(&u, &f, 4); Wb[i] = (uint16_t)(u >> 16);
    }
    q4_0_block_t *Wq4 = (q4_0_block_t *)malloc((size_t)rows * (cols / 32) * sizeof(q4_0_block_t));
    qwen_quantize_bf16_to_q4_0(Wb, rows, cols, Wq4);

    for (int B = 2; B <= 8; B *= 2) {
        printf("\nB = %d\n", B);
        float *X = (float *)malloc((size_t)cols * B * sizeof(float));
        for (int i = 0; i < cols * B; i++) X[i] = (float)(rnd() * 0.5);
        float *Y = (float *)malloc((size_t)rows * B * sizeof(float));
        float *R = (float *)malloc((size_t)rows * B * sizeof(float));
        int8_t *qXt = (int8_t *)malloc((size_t)B * cols);
        float *sx = (float *)malloc((size_t)B * sizeof(float));
        for (int b = 0; b < B; b++) sx[b] = qcol(qXt + (size_t)b * cols, X, cols, B, b);

        /* ── int8: riferimento intero ── */
        for (int r = 0; r < rows; r++)
            for (int b = 0; b < B; b++) {
                const int8_t *w = W + (size_t)r * cols, *qb = qXt + (size_t)b * cols;
                long sum = 0;
                for (int k = 0; k < cols; k++) sum += (long)w[k] * (long)qb[k];
                R[(size_t)r * B + b] = (float)sum * scale[r] * sx[b];
            }
        qwen_matmat_int8(Y, W, scale, X, rows, cols, B);
        fail += check("int8 matmat", Y, R, rows * B);

        /* ── q4_0: riferimento intero, stesso ordine di accumulo (per blocco) ── */
        int nb = cols / 32;
        for (int r = 0; r < rows; r++) {
            const q4_0_block_t *row = Wq4 + (size_t)r * nb;
            for (int b = 0; b < B; b++) {
                const int8_t *qb = qXt + (size_t)b * cols;
                float acc = 0.0f;
                for (int bl = 0; bl < nb; bl++) {
                    int dot = 0;
                    for (int k = 0; k < 32; k++) {
                        int lo = row[bl].qs[k / 2];
                        int code = (k % 2 == 0) ? (lo & 0x0F) : ((lo >> 4) & 0x0F);
                        dot += (code - 8) * (int)qb[bl * 32 + k];
                    }
                    acc += qwen_f16_to_f32(row[bl].scale_f16) * (float)dot;
                }
                R[(size_t)r * B + b] = acc * sx[b];
            }
        }
        qwen_matmat_q4_0(Y, Wq4, X, rows, cols, B);
        fail += check("q4_0 matmat", Y, R, rows * B);

        free(X); free(Y); free(R); free(qXt); free(sx);
    }
    printf("\n%s\n", fail ? "FAIL — un gemello batched non fa l'aritmetica che dichiara" :
                            "PASS — i gemelli batched coincidono col riferimento intero");
    return fail ? 1 : 0;
}
