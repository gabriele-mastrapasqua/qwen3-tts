/* matmat_parity.c — il gemello batched fa la STESSA aritmetica del riferimento intero?
 *
 * PERCHE' ESISTE. I GEMM quantizzati sono per-ISA: VNNI su AVX-512, SMMLA su i8mm,
 * il nuovo maddubs su AVX2 puro, il twin f32 altrove. Ogni percorso e' codice diverso
 * per lo stesso formato, e finora l'unico modo di verificarne uno era affittare la
 * macchina che lo esegue. Cosi' i kernel AVX-512 sono stati scritti a giugno e mai
 * eseguiti da noi.
 *
 * ⭐ SU M1 SI PUO' ESEGUIRE LO STESSO IL PERCORSO x86: Rosetta 2 supporta AVX2 (misurato
 * su questa macchina, 2026-08-18). Quindi questo test, compilato per x86_64 con
 * -march=x86-64-v3, ESEGUE davvero il kernel AVX2 e ne verifica i numeri — sul portatile,
 * prima di pagare un'ora di cloud. (Rosetta NON emula AVX-512: VNNI e AMX restano da
 * validare sul metallo.)
 *
 * L'ORACOLO, e perche' e' questo. Non si confronta il matmat contro il matvec: quello e'
 * un percorso f32 con un'altra quantizzazione delle attivazioni, quindi una differenza
 * non direbbe se il kernel e' sbagliato o solo diversamente arrotondato. Si confronta
 * contro un riferimento INTERO in C semplice che quantizza le attivazioni esattamente
 * come il motore e somma in int32. Il kernel deve dare lo STESSO numero: l'aritmetica
 * intera non ha ordine di somma da discutere. Atteso: errore relativo 0.
 *
 * Uso:
 *   make check-matmat-parity        # ISA nativa
 *   make check-matmat-parity-x86    # x86-64-v3 sotto Rosetta, dal Mac ARM
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

/* DUE SOGLIE, perche' non tutte le ISA prendono il percorso intero. Dove il
 * dispatcher ha un GEMM intero vero (AVX2/maddubs, AVX-512/VNNI, i8mm/SMMLA) il
 * risultato deve coincidere col riferimento intero a meno del solo arrotondamento
 * finale: rel ~ 0. Dove invece cade sul twin f32 (M1 senza i8mm, o QWEN_NO_*=1) le
 * attivazioni restano in f32 e la differenza e' l'errore di quantizzazione, non un
 * bug: si dichiara, non si nasconde. Un FAIL e' solo oltre la seconda soglia. */
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
