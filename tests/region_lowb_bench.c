/* region_lowb_bench.c — X86-2: what does the in-region INT8 runner really cost at low B?
 *
 * The engine's AMX INT8 gate starts at B>=4 (QWEN_AMX_MIN_B / QWEN_AMX_INT8_MIN_B).  An
 * oneDNN oracle picked an AMX BRGEMM even at B1/B2, but that is an oracle with a different
 * threading, layout and quantisation contract.  This harness asks the question for OUR path:
 * the COMPLETE contract a region pays per projection -- AMX activation pack (once per thread,
 * per call, as the runner does it today), packed-RHS lookup, the matmul, and the output scale
 * -- against the VNNI row-block runner doing the same work.
 *
 * It runs the runner exactly as the region does: qwen_parallel over the engine pool, each
 * worker calling qwen_region_i8_run(tid, nt).  The two arms are INTERLEAVED inside one process
 * and pinned with qwen_mm_force(), and the reported number is the MEDIAN of R alternating
 * rounds: comparing arms across two processes on this box swung 14% on an unchanged
 * configuration, which is larger than the effect under test.
 *
 * Shapes are the real 1.7B projections (CP h=1024 q=2048 kv=1024 inter=3072;
 * Talker h=2048 q=2048 kv=1024 inter=6144).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include "qwen_tts_kernels.h"
#include "qwen_tts_thread.h"

static double now_ms(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e3 + t.tv_nsec / 1e6;
}
static int cmpd(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return x < y ? -1 : (x > y ? 1 : 0);
}
static double median(double *v, int n) {
    qsort(v, (size_t)n, sizeof(double), cmpd);
    return n & 1 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}
static uint64_t rs = 0x243F6A8885A308D3ull;
static int8_t rq(void) {
    rs = rs * 6364136223846793005ull + 1442695040888963407ull;
    return (int8_t)((int)((rs >> 40) & 0xFF) - 128);
}

typedef struct {
    float *Y, *Yk, *Yv;
    const int8_t *W, *Wk, *Wv;
    const float *sw, *swk, *swv;
    const int8_t *qXt; const float *sx;
    int rows, kv_rows, cols, B, qkv;
} job_t;

static void job_task(size_t tid, size_t nt, void *vc) {
    job_t *j = (job_t *)vc;
    if (j->qkv)
        qwen_region_i8_run_qkv(j->Y, j->Yk, j->Yv, j->W, j->sw, j->Wk, j->swk,
                               j->Wv, j->swv, j->qXt, j->sx,
                               j->rows, j->kv_rows, j->cols, j->B, tid, nt);
    else
        qwen_region_i8_run(j->Y, j->W, j->sw, j->qXt, j->sx,
                           j->rows, j->cols, j->B, tid, nt);
}

typedef struct { const char *name; int rows, kv_rows, cols, qkv; } shape_t;

int main(int argc, char **argv) {
    int nt   = argc > 1 ? atoi(argv[1]) : 8;
    int iters= argc > 2 ? atoi(argv[2]) : 200;
    int maxB = argc > 3 ? atoi(argv[3]) : 4;
    int rounds = argc > 4 ? atoi(argv[4]) : 7;
    if (rounds > 64) rounds = 64;
    qwen_set_threads(nt);
    setvbuf(stdout, NULL, _IONBF, 0);

    static const shape_t shapes[] = {
        { "CP QKV      ",  2048, 1024, 1024, 1 },
        { "CP WO       ",  1024,    0, 2048, 0 },
        { "CP Gate/Up  ",  6144,    0, 1024, 0 },
        { "CP Down     ",  1024,    0, 3072, 0 },
        { "TK QKV      ",  2048, 1024, 2048, 1 },
        { "TK WO       ",  2048,    0, 2048, 0 },
        { "TK Gate/Up  ", 12288,    0, 2048, 0 },
        { "TK Down     ",  2048,    0, 6144, 0 },
    };
    const int ns = (int)(sizeof shapes / sizeof shapes[0]);

    printf("region low-B bench: threads=%d iters=%d  (complete in-region contract per call)\n", nt, iters);
    printf("%-13s %5s %6s %6s   %9s %9s   %8s  %s\n",
           "projection", "B", "rows", "cols", "VNNI us", "AMX us", "AMX vs", "winner");

    for (int s = 0; s < ns; s++) {
        const shape_t *sh = &shapes[s];
        int rows = sh->rows, kvr = sh->kv_rows, cols = sh->cols;
        size_t wn = (size_t)rows * cols, wkn = (size_t)kvr * cols;
        int8_t *W  = (int8_t *)aligned_alloc(64, (wn + 63) & ~(size_t)63);
        int8_t *Wk = kvr ? (int8_t *)aligned_alloc(64, (wkn + 63) & ~(size_t)63) : NULL;
        int8_t *Wv = kvr ? (int8_t *)aligned_alloc(64, (wkn + 63) & ~(size_t)63) : NULL;
        float *sw  = (float *)malloc((size_t)rows * sizeof(float));
        float *swk = kvr ? (float *)malloc((size_t)kvr * sizeof(float)) : NULL;
        float *swv = kvr ? (float *)malloc((size_t)kvr * sizeof(float)) : NULL;
        float *Y   = (float *)aligned_alloc(64, (size_t)rows * 64 * sizeof(float));
        float *Yk  = kvr ? (float *)aligned_alloc(64, (size_t)kvr * 64 * sizeof(float)) : NULL;
        float *Yv  = kvr ? (float *)aligned_alloc(64, (size_t)kvr * 64 * sizeof(float)) : NULL;
        int8_t *qXt= (int8_t *)aligned_alloc(64, ((size_t)cols * 64 + 63) & ~(size_t)63);
        float *sx  = (float *)malloc(64 * sizeof(float));
        if (!W || !sw || !Y || !qXt || !sx || (kvr && (!Wk || !Wv || !swk || !swv || !Yk || !Yv))) {
            printf("  OOM on %s\n", sh->name); return 1;
        }
        for (size_t i = 0; i < wn; i++) W[i] = rq();
        for (size_t i = 0; i < wkn; i++) { Wk[i] = rq(); Wv[i] = rq(); }
        for (int r = 0; r < rows; r++) sw[r] = 0.01f;
        for (int r = 0; r < kvr; r++) { swk[r] = 0.01f; swv[r] = 0.01f; }
        for (size_t i = 0; i < (size_t)cols * 64; i++) qXt[i] = rq();
        for (int b = 0; b < 64; b++) sx[b] = 0.02f;

        for (int B = 1; B <= maxB; B++) {
            job_t j = { Y, Yk, Yv, W, Wk, Wv, sw, swk, swv, qXt, sx, rows, kvr, cols, B, sh->qkv };
            double amx[64], vnni[64];
            for (int r = 0; r < rounds; r++) {
                for (int arm = 0; arm < 2; arm++) {
                    qwen_mm_force(arm == 0 ? QWEN_MMK_INT8_AMX : QWEN_MMK_INT8_VNNI);
                    for (int w = 0; w < 3; w++) qwen_parallel((size_t)nt, job_task, &j);
                    double t0 = now_ms();
                    for (int it = 0; it < iters; it++) qwen_parallel((size_t)nt, job_task, &j);
                    double t1 = now_ms();
                    (arm == 0 ? amx : vnni)[r] = (t1 - t0) * 1e3 / iters;
                }
            }
            qwen_mm_force(0);
            double a = median(amx, rounds), v = median(vnni, rounds);
            printf("%-13s %5d %6d %6d   %9.2f %9.2f   %+7.1f%%  %s\n",
                   sh->name, B, rows, cols, v, a, (a - v) / v * 100.0,
                   a < v ? "AMX" : "VNNI");
        }
        free(W); free(Wk); free(Wv); free(sw); free(swk); free(swv);
        free(Y); free(Yk); free(Yv); free(qXt); free(sx);
    }
    return 0;
}
