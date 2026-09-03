/* Exact-shape B=1 INT8 GEMV A/B for the x86 VNNI path (Track A / A1).
 *
 * Times the COMPLETE public entry points (qwen_matvec_int8 /
 * qwen_matvec_int8_qkv), not just the inner dot, so activation quantization,
 * row-sum lookup, thread dispatch and the epilogue are all inside the measured
 * region.  Run it twice, once with QWEN_VNNI_UACT=0 and once with =1, and
 * compare the reported ns/call and the y checksum (which must be identical). */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "../qwen_tts_kernels.h"
#include "../qwen_tts_thread.h"

void qwen_set_threads(int n);

static uint32_t rng_state = 0x4d595df4u;
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
static double median(double *v, int n) {
    for (int i = 1; i < n; i++) {
        double x = v[i]; int j = i;
        while (j > 0 && v[j - 1] > x) { v[j] = v[j - 1]; j--; }
        v[j] = x;
    }
    return v[n / 2];
}
/* Bit-exact fingerprint of the output vector. */
static uint64_t fnv(const float *y, int n) {
    uint64_t h = 1469598103934665603ull;
    const unsigned char *p = (const unsigned char *)y;
    for (size_t i = 0; i < (size_t)n * sizeof(float); i++) {
        h ^= p[i]; h *= 1099511628211ull;
    }
    return h;
}

enum { MAXREP = 201 };

/* Weight working set per shape, in MB.  The served model streams ~1.7 GB of
 * INT8 weights, so a single-matrix micro-bench would sit in L2/L3 and measure a
 * regime the engine never sees.  With a pool larger than L3 every call reads a
 * different matrix and the shape is DRAM-resident, like production. */
static int pool_mb(void) {
    const char *e = getenv("QWEN_B1_BENCH_POOL_MB");
    int v = e ? atoi(e) : 192;
    return v < 0 ? 0 : v;
}
enum { MAXCOPIES = 64 };

static void run_gemv(const char *name, int out_dim, int in_dim, int reps) {
    qwen_vnni_row_sums_reset();
    const size_t wbytes = (size_t)out_dim * in_dim;
    int copies = 1;
    if (pool_mb() > 0) {
        copies = (int)(((size_t)pool_mb() * 1000000u + wbytes - 1) / wbytes);
        if (copies < 1) copies = 1;
        if (copies > MAXCOPIES) copies = MAXCOPIES;
    }
    int8_t *W[MAXCOPIES];
    for (int c = 0; c < copies; c++) {
        W[c] = (int8_t *)aligned_zero(wbytes);
        if (!W[c]) { fprintf(stderr, "alloc failed\n"); return; }
    }
    float *scale = (float *)aligned_zero((size_t)out_dim * sizeof(float));
    float *x = (float *)aligned_zero((size_t)in_dim * sizeof(float));
    float *y = (float *)aligned_zero((size_t)out_dim * sizeof(float));
    if (!scale || !x || !y) { fprintf(stderr, "alloc failed\n"); return; }
    for (size_t i = 0; i < wbytes; i++) W[0][i] = (int8_t)(random_value() * 127.0f);
    for (int c = 1; c < copies; c++) memcpy(W[c], W[0], wbytes);
    for (int i = 0; i < out_dim; i++) scale[i] = 0.002f + 0.001f * fabsf(random_value());
    for (int i = 0; i < in_dim; i++) x[i] = random_value();

    /* Warm every copy so the row-sum cache is built and not timed. */
    for (int c = 0; c < copies; c++)
        qwen_matvec_int8(y, W[c], scale, x, out_dim, in_dim);
    uint64_t fp = fnv(y, out_dim);

    static double s[MAXREP];
    if (reps > MAXREP) reps = MAXREP;
    for (int r = 0; r < reps; r++) {
        const int8_t *w = W[r % copies];
        double t0 = now_ns();
        qwen_matvec_int8(y, w, scale, x, out_dim, in_dim);
        s[r] = now_ns() - t0;
    }
    double ns = median(s, reps);
    double gb = (double)wbytes / 1e9;
    printf("gemv  %-22s N=%-5d K=%-5d pool=%4dMB ns=%9.0f  weight_GBps=%7.2f  fp=%016llx\n",
           name, out_dim, in_dim, (int)(wbytes * (size_t)copies / 1000000u),
           ns, gb / (ns / 1e9), (unsigned long long)fp);
    fflush(stdout);
    for (int c = 0; c < copies; c++) free(W[c]);
    free(scale); free(x); free(y);
}

static void run_qkv(const char *name, int in_dim, int q_dim, int kv_dim, int reps) {
    qwen_vnni_row_sums_reset();
    const size_t qb = (size_t)q_dim * in_dim, kb = (size_t)kv_dim * in_dim;
    const size_t setb = qb + 2 * kb;
    int copies = 1;
    if (pool_mb() > 0) {
        copies = (int)(((size_t)pool_mb() * 1000000u + setb - 1) / setb);
        if (copies < 1) copies = 1;
        if (copies > MAXCOPIES / 3) copies = MAXCOPIES / 3;
    }
    int8_t *WQ[MAXCOPIES / 3], *WK[MAXCOPIES / 3], *WV[MAXCOPIES / 3];
    for (int c = 0; c < copies; c++) {
        WQ[c] = (int8_t *)aligned_zero(qb);
        WK[c] = (int8_t *)aligned_zero(kb);
        WV[c] = (int8_t *)aligned_zero(kb);
        if (!WQ[c] || !WK[c] || !WV[c]) { fprintf(stderr, "alloc failed\n"); return; }
    }
    int8_t *Wq = WQ[0], *Wk = WK[0], *Wv = WV[0];
    float *sq = (float *)aligned_zero((size_t)q_dim * sizeof(float));
    float *sk = (float *)aligned_zero((size_t)kv_dim * sizeof(float));
    float *sv = (float *)aligned_zero((size_t)kv_dim * sizeof(float));
    float *x = (float *)aligned_zero((size_t)in_dim * sizeof(float));
    float *q = (float *)aligned_zero((size_t)q_dim * sizeof(float));
    float *k = (float *)aligned_zero((size_t)kv_dim * sizeof(float));
    float *v = (float *)aligned_zero((size_t)kv_dim * sizeof(float));
    if (!sq || !sk || !sv || !x || !q || !k || !v) {
        fprintf(stderr, "alloc failed\n"); return;
    }
    for (size_t i = 0; i < qb; i++) Wq[i] = (int8_t)(random_value() * 127.0f);
    for (size_t i = 0; i < kb; i++) {
        Wk[i] = (int8_t)(random_value() * 127.0f);
        Wv[i] = (int8_t)(random_value() * 127.0f);
    }
    for (int i = 0; i < q_dim; i++) sq[i] = 0.002f + 0.001f * fabsf(random_value());
    for (int i = 0; i < kv_dim; i++) {
        sk[i] = 0.002f + 0.001f * fabsf(random_value());
        sv[i] = 0.002f + 0.001f * fabsf(random_value());
    }
    for (int i = 0; i < in_dim; i++) x[i] = random_value();
    for (int c = 1; c < copies; c++) {
        memcpy(WQ[c], WQ[0], qb); memcpy(WK[c], WK[0], kb); memcpy(WV[c], WV[0], kb);
    }

    for (int c = 0; c < copies; c++)
        qwen_matvec_int8_qkv(q, k, v, WQ[c], sq, WK[c], sk, WV[c], sv,
                             x, in_dim, q_dim, kv_dim);
    uint64_t fp = fnv(q, q_dim) ^ fnv(k, kv_dim) ^ fnv(v, kv_dim);

    static double s[MAXREP];
    if (reps > MAXREP) reps = MAXREP;
    for (int r = 0; r < reps; r++) {
        const int i = r % copies;
        double t0 = now_ns();
        qwen_matvec_int8_qkv(q, k, v, WQ[i], sq, WK[i], sk, WV[i], sv,
                             x, in_dim, q_dim, kv_dim);
        s[r] = now_ns() - t0;
    }
    double ns = median(s, reps);
    double gb = (double)setb / 1e9;
    printf("qkv   %-22s N=%-5d K=%-5d pool=%4dMB ns=%9.0f  weight_GBps=%7.2f  fp=%016llx\n",
           name, q_dim + 2 * kv_dim, in_dim, (int)(setb * (size_t)copies / 1000000u),
           ns, gb / (ns / 1e9), (unsigned long long)fp);
    fflush(stdout);
    for (int c = 0; c < copies; c++) { free(WQ[c]); free(WK[c]); free(WV[c]); }
    free(sq); free(sk); free(sv);
    free(x); free(q); free(k); free(v);
}

/* B=2 small GEMM on the same pooled weights.  X is [cols][B] (column major per
 * K), Y is [rows][B], exactly the layout the served engine hands to
 * qwen_matmat_int8(). */
static void run_matmat(const char *name, int out_dim, int in_dim, int B, int reps) {
    qwen_vnni_row_sums_reset();
    const size_t wbytes = (size_t)out_dim * in_dim;
    int copies = 1;
    if (pool_mb() > 0) {
        copies = (int)(((size_t)pool_mb() * 1000000u + wbytes - 1) / wbytes);
        if (copies < 1) copies = 1;
        if (copies > MAXCOPIES) copies = MAXCOPIES;
    }
    int8_t *W[MAXCOPIES];
    for (int c = 0; c < copies; c++) {
        W[c] = (int8_t *)aligned_zero(wbytes);
        if (!W[c]) { fprintf(stderr, "alloc failed\n"); return; }
    }
    float *scale = (float *)aligned_zero((size_t)out_dim * sizeof(float));
    float *X = (float *)aligned_zero((size_t)in_dim * B * sizeof(float));
    float *Y = (float *)aligned_zero((size_t)out_dim * B * sizeof(float));
    if (!scale || !X || !Y) { fprintf(stderr, "alloc failed\n"); return; }
    for (size_t i = 0; i < wbytes; i++) W[0][i] = (int8_t)(random_value() * 127.0f);
    for (int c = 1; c < copies; c++) memcpy(W[c], W[0], wbytes);
    for (int i = 0; i < out_dim; i++) scale[i] = 0.002f + 0.001f * fabsf(random_value());
    for (int i = 0; i < in_dim * B; i++) X[i] = random_value();

    for (int c = 0; c < copies; c++)
        qwen_matmat_int8(Y, W[c], scale, X, out_dim, in_dim, B);
    uint64_t fp = fnv(Y, out_dim * B);

    static double s[MAXREP];
    if (reps > MAXREP) reps = MAXREP;
    for (int r = 0; r < reps; r++) {
        const int8_t *w = W[r % copies];
        double t0 = now_ns();
        qwen_matmat_int8(Y, w, scale, X, out_dim, in_dim, B);
        s[r] = now_ns() - t0;
    }
    double ns = median(s, reps);
    double gb = (double)wbytes / 1e9;
    printf("mm%-2d  %-22s N=%-5d K=%-5d pool=%4dMB ns=%9.0f  weight_GBps=%7.2f  fp=%016llx\n",
           B, name, out_dim, in_dim, (int)(wbytes * (size_t)copies / 1000000u),
           ns, gb / (ns / 1e9), (unsigned long long)fp);
    fflush(stdout);
    for (int c = 0; c < copies; c++) free(W[c]);
    free(scale); free(X); free(Y);
}

int main(int argc, char **argv) {
    int threads = argc > 1 ? atoi(argv[1]) : 8;
    int reps = argc > 2 ? atoi(argv[2]) : 101;
    if (threads < 1) threads = 1;
    if (reps < 3) reps = 3;
    qwen_set_threads(threads);
    const char *u = getenv("QWEN_VNNI_UACT");
    printf("x86-b1-gemv-bench threads=%d reps=%d UACT=%s MR=%s\n",
           threads, reps, u ? u : "0",
           getenv("QWEN_VNNI_GEMV_MR") ? getenv("QWEN_VNNI_GEMV_MR") : "2");
#if !defined(__AVX512VNNI__)
    printf("SKIP: binary was not built with AVX-512 VNNI\n");
    return 0;
#else
    /* Dominant B=1 INT8 shapes from the CP/talker shape census. */
    run_gemv("cp_down",        1024, 3072, reps);
    run_gemv("cp_o_proj",      1024, 2048, reps);
    run_gemv("cp_gate_up",     6144, 1024, reps);
    run_gemv("cp_lm_head",     2048, 1024, reps);
    run_gemv("cp_small",       1024, 1024, reps);
    run_gemv("talker_gate_up", 3072, 2048, reps);
    run_qkv ("cp_qkv",         1024, 2048, 1024, reps);
    if (getenv("QWEN_B1_BENCH_B2")) {
        printf("--- B=2 matmat, same shapes/pool ---\n");
        run_matmat("cp_down",        1024, 3072, 2, reps);
        run_matmat("cp_o_proj",      1024, 2048, 2, reps);
        run_matmat("cp_gate_up",     6144, 1024, 2, reps);
        run_matmat("cp_lm_head",     2048, 1024, 2, reps);
        run_matmat("cp_small",       1024, 1024, 2, reps);
        run_matmat("talker_gate_up", 3072, 2048, 2, reps);
        run_matmat("cp_qkv_q",       2048, 1024, 2, reps);
    }
    qwen_census_report(NULL);
    qwen_matmat_stats_report(NULL);
    return 0;
#endif
}
