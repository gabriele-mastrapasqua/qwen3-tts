/* qwen_tts_sd_gemm.c - decoder SGEMM under the engine's own execution budget.
 *
 * With QWEN_BLAS_OWN=1 OpenBLAS is held at one thread, so it never runs a worker team
 * of its own inside a prefork worker.  The decoder's SGEMMs are then partitioned here
 * across the engine pool: C[:, n0:n1] = op(A)·op(B)[:, n0:n1] (or a row block of C when
 * M is the wider dimension).  Each slice is an exact sub-problem — no reduction is split —
 * so the math is unchanged; only who executes it.  Inside an existing parallel region the
 * call runs inline on the current worker (the pool has a single job slot). */
#include "qwen_tts_kernels.h"
#include "qwen_tts_thread.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

/* QWEN_SD_SGEMM_CENSUS=1: count every decoder SGEMM shape and its wall time, print at
 * exit.  Diagnostic only — a shape table update per call — never in a run that produces
 * a number.  It answers one question: how many shape families carry the BLAS time. */
typedef struct { int ta, tb, M, N, K; unsigned long calls; double ns; } sdg_shape_t;
#define SDG_SHAPES 256
static sdg_shape_t g_sdg_shapes[SDG_SHAPES];
static int g_sdg_nshapes = 0, g_sdg_census = -1;
static void sdg_census_dump(void) {
    if (g_sdg_nshapes == 0) return;
    double tot = 0; for (int i = 0; i < g_sdg_nshapes; i++) tot += g_sdg_shapes[i].ns;
    /* sort by time, descending (tiny table, insertion sort is fine) */
    for (int i = 1; i < g_sdg_nshapes; i++) { sdg_shape_t t = g_sdg_shapes[i]; int j = i - 1;
        while (j >= 0 && g_sdg_shapes[j].ns < t.ns) { g_sdg_shapes[j + 1] = g_sdg_shapes[j]; j--; }
        g_sdg_shapes[j + 1] = t; }
    fprintf(stderr, "[SD_SGEMM_CENSUS] pid=%d shapes=%d total_ms=%.1f (ta tb M N K calls ms cum%%)\n",
            (int)getpid(), g_sdg_nshapes, tot / 1e6);
    double cum = 0;
    for (int i = 0; i < g_sdg_nshapes; i++) { sdg_shape_t *s = &g_sdg_shapes[i]; cum += s->ns;
        fprintf(stderr, "[SD_SGEMM_CENSUS] %c %c M=%d N=%d K=%d calls=%lu ms=%.2f cum=%.1f%%\n",
                s->ta == 111 ? 'N' : 'T', s->tb == 111 ? 'N' : 'T', s->M, s->N, s->K,
                s->calls, s->ns / 1e6, 100.0 * cum / tot); }
}
static int sdg_census_on(void) {
    if (g_sdg_census < 0) { const char *e = getenv("QWEN_SD_SGEMM_CENSUS");
        g_sdg_census = (e && e[0] == '1') ? 1 : 0; if (g_sdg_census) atexit(sdg_census_dump); }
    return g_sdg_census;
}
static void sdg_census_add(int ta, int tb, int M, int N, int K, double ns) {
    for (int i = 0; i < g_sdg_nshapes; i++) { sdg_shape_t *s = &g_sdg_shapes[i];
        if (s->ta == ta && s->tb == tb && s->M == M && s->N == N && s->K == K) { s->calls++; s->ns += ns; return; } }
    if (g_sdg_nshapes < SDG_SHAPES) { sdg_shape_t *s = &g_sdg_shapes[g_sdg_nshapes++];
        s->ta = ta; s->tb = tb; s->M = M; s->N = N; s->K = K; s->calls = 1; s->ns = ns; }
}
static double sdg_now_ns(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e9 + t.tv_nsec; }

#ifdef USE_BLAS
#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#define SDG_ORDER enum CBLAS_ORDER
#define SDG_TRANS enum CBLAS_TRANSPOSE
#else
#include <cblas.h>
#define SDG_ORDER CBLAS_ORDER
#define SDG_TRANS CBLAS_TRANSPOSE
#endif

typedef struct {
    int order, ta, tb, M, N, K, lda, ldb, ldc, split_n, chunk;
    float alpha, beta;
    const float *A, *B;
    float *C;
} sdg_job_t;

static void sdg_task(size_t tid, size_t nt, void *vj) {
    (void)nt;
    sdg_job_t *j = (sdg_job_t *)vj;
    int lo = (int)tid * j->chunk, hi = lo + j->chunk;
    if (j->split_n) {
        if (hi > j->N) hi = j->N;
        if (lo >= hi) return;
        const float *B = (j->tb == CblasNoTrans) ? j->B + lo : j->B + (size_t)lo * j->ldb;
        cblas_sgemm((SDG_ORDER)j->order, (SDG_TRANS)j->ta, (SDG_TRANS)j->tb,
                    j->M, hi - lo, j->K, j->alpha, j->A, j->lda, B, j->ldb,
                    j->beta, j->C + lo, j->ldc);
    } else {
        if (hi > j->M) hi = j->M;
        if (lo >= hi) return;
        const float *A = (j->ta == CblasNoTrans) ? j->A + (size_t)lo * j->lda : j->A + lo;
        cblas_sgemm((SDG_ORDER)j->order, (SDG_TRANS)j->ta, (SDG_TRANS)j->tb,
                    hi - lo, j->N, j->K, j->alpha, A, j->lda, j->B, j->ldb,
                    j->beta, j->C + (size_t)lo * j->ldc, j->ldc);
    }
}

static void sdg_run(int order, int ta, int tb, int M, int N, int K, float alpha,
                    const float *A, int lda, const float *B, int ldb, float beta,
                    float *C, int ldc);
void qwen_sd_sgemm(int order, int ta, int tb, int M, int N, int K, float alpha,
                   const float *A, int lda, const float *B, int ldb, float beta,
                   float *C, int ldc) {
    if (!sdg_census_on()) { sdg_run(order, ta, tb, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc); return; }
    double t0 = sdg_now_ns();
    sdg_run(order, ta, tb, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
    sdg_census_add(ta, tb, M, N, K, sdg_now_ns() - t0);
    /* prefork workers die by signal, so the table is also printed every 1000 calls */
    static unsigned long calls = 0;
    if ((++calls % 100) == 0) sdg_census_dump();
}
static void sdg_run(int order, int ta, int tb, int M, int N, int K, float alpha,
                    const float *A, int lda, const float *B, int ldb, float beta,
                    float *C, int ldc) {
    int nt = qwen_get_threads();
    /* Small problems, single thread, inside a region, or BLAS still owning its team:
     * plain call.  The threshold keeps a ~us-scale GEMM from paying a pool dispatch. */
    if (!qwen_blas_own_effective() || nt <= 1 || order != (int)CblasRowMajor ||
        qwen_parallel_active() || (double)M * (double)N * (double)K < 262144.0) {
        cblas_sgemm((SDG_ORDER)order, (SDG_TRANS)ta, (SDG_TRANS)tb, M, N, K,
                    alpha, A, lda, B, ldb, beta, C, ldc);
        return;
    }
    sdg_job_t j = { order, ta, tb, M, N, K, lda, ldb, ldc, 0, 0, alpha, beta, A, B, C };
    int dim = N >= M ? N : M;
    j.split_n = (N >= M);
    int chunks = dim / 32;              /* every slice keeps >= 32 columns (or rows) */
    if (chunks > nt) chunks = nt;
    if (chunks < 2) {
        cblas_sgemm((SDG_ORDER)order, (SDG_TRANS)ta, (SDG_TRANS)tb, M, N, K,
                    alpha, A, lda, B, ldb, beta, C, ldc);
        return;
    }
    j.chunk = (dim + chunks - 1) / chunks;
    j.chunk = (j.chunk + 15) & ~15;     /* 16-aligned slices keep the BLAS kernels on their fast path */
    chunks = (dim + j.chunk - 1) / j.chunk;
    qwen_parallel((size_t)chunks, sdg_task, &j);
}
#else
void qwen_sd_sgemm(int order, int ta, int tb, int M, int N, int K, float alpha,
                   const float *A, int lda, const float *B, int ldb, float beta,
                   float *C, int ldc) {
    (void)order; (void)ta; (void)tb; (void)M; (void)N; (void)K; (void)alpha; (void)A;
    (void)lda; (void)B; (void)ldb; (void)beta; (void)C; (void)ldc;
}
#endif
