/* prefill_bench.c — what does a Talker prefill actually pay for?
 *
 * The matmat kernels cap a call at B=16 (the AMX tile is 16 rows x 64 bytes and the accumulator
 * is configured colsb = B*4), so a prefill of n positions is ceil(n/16) calls. End-to-end timing
 * on an 8-core Emerald Rapids fits prefill_ms ~= 20*ceil(n/16) + 2.2*n, i.e. a large fixed cost
 * per call. This harness says WHAT that fixed cost is before anyone rewrites a kernel to avoid
 * it: activation packing, per-call overhead, or the weight stream.
 *
 *   ./prefill_bench [rows] [cols] [threads]
 *
 * Prototype (AMX builds only): the same work with two accumulator tiles, so one pass over the
 * weights serves 32 positions instead of 16. It is checked against the shipped path before it is
 * timed — a faster wrong answer is not a result.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <time.h>
#include "qwen_tts_kernels.h"

#if defined(__AMX_BF16__) && defined(__AMX_TILE__)
#include <immintrin.h>
#define HAVE_AMX 1
#else
#define HAVE_AMX 0
#endif

static double now_ms(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e3 + t.tv_nsec / 1e6;
}
static uint64_t rs = 0x243F6A8885A308D3ull;
static double rnd(void) {
    rs = rs * 6364136223846793005ull + 1442695040888963407ull;
    return (double)((rs >> 40) / (double)(1u << 24)) * 2.0 - 1.0;
}
static inline uint16_t f2b(float f) { uint32_t u; memcpy(&u, &f, 4); return (uint16_t)(u >> 16); }
static inline float b2f(uint16_t b) { uint32_t u = (uint32_t)b << 16; float f; memcpy(&f, &u, 4); return f; }

/* the loop shape the talker prefill uses today: pack a chunk, one matmat call per chunk */
static void path_current(float *Y, const uint16_t *W, const float *Xn, int n, int in_dim,
                         int out_dim, float *xT, float *yT, double *pack_ms, double *mm_ms) {
    for (int s0 = 0; s0 < n; s0 += 16) {
        int B = n - s0; if (B > 16) B = 16;
        double t0 = now_ms();
        for (int b = 0; b < B; b++) {
            const float *xr = Xn + (int64_t)(s0 + b) * in_dim;
            for (int k = 0; k < in_dim; k++) xT[(int64_t)k * B + b] = xr[k];
        }
        double t1 = now_ms();
        qwen_matmat_bf16(yT, W, xT, out_dim, in_dim, B);
        double t2 = now_ms();
        for (int b = 0; b < B; b++) {
            float *yr = Y + (int64_t)(s0 + b) * out_dim;
            for (int o = 0; o < out_dim; o++) yr[o] = yT[(int64_t)o * B + b];
        }
        *pack_ms += (t1 - t0); *mm_ms += (t2 - t1);
    }
}

#if HAVE_AMX
/* the packing the AMX path expects: per 32-wide k chunk, 16 rows of B bf16 pairs */
static void pack_act(uint16_t *p, const uint16_t *Xb, int cols, int kfull, int B) {
    const int nch = kfull >> 5;
    const size_t cstride = (size_t)B * 2;
    for (int kc = 0; kc < nch; kc++) {
        uint16_t *dst = p + (size_t)kc * 32 * (size_t)B;
        for (int nn = 0; nn < B; nn++) {
            const uint16_t *src = Xb + (size_t)nn * cols + (size_t)kc * 32;
            for (int j = 0; j < 16; j++)
                memcpy(dst + (size_t)j * cstride + (size_t)nn * 2, src + 2 * j, 2 * sizeof(uint16_t));
        }
    }
}
typedef struct { uint8_t palette, start_row, rsvd[14]; uint16_t colsb[16]; uint8_t rows[16]; } tcfg_t;

/* one weight load, two accumulators: 32 positions per pass over W instead of 16 */
static void amx_b32(float *Y, const uint16_t *W, const uint16_t *pA, const uint16_t *pB,
                    int rows, int cols, int nb) {
    const int kfull = cols & ~31, nch = kfull >> 5;
    const size_t wstride = (size_t)cols * sizeof(uint16_t);
    tcfg_t cfg; memset(&cfg, 0, sizeof cfg);
    cfg.palette = 1;
    for (int t = 0; t < 6; t++) { cfg.rows[t] = 16; cfg.colsb[t] = 64; }
    _tile_loadconfig(&cfg);
    float c0[256] __attribute__((aligned(64))), c1[256] __attribute__((aligned(64)));
    for (int r = 0; r + 15 < rows; r += 16) {
        _tile_zero(0); _tile_zero(1);
        for (int kc = 0; kc < nch; kc++) {
            _tile_loadd(4, pA + (size_t)kc * 32 * 16, 64);
            _tile_loadd(2, W + (size_t)r * cols + (size_t)kc * 32, wstride);
            _tile_dpbf16ps(0, 2, 4);
            if (nb > 16) {
                _tile_loadd(5, pB + (size_t)kc * 32 * 16, 64);
                _tile_dpbf16ps(1, 2, 5);
            }
        }
        _tile_stored(0, c0, 64);
        if (nb > 16) _tile_stored(1, c1, 64);
        for (int m = 0; m < 16; m++)
            for (int b = 0; b < nb; b++)
                Y[(size_t)(r + m) * nb + b] = (b < 16) ? c0[m * 16 + b] : c1[m * 16 + (b - 16)];
    }
    _tile_release();
}
#endif

int main(int argc, char **argv) {
    int fail = 0;
    int rows = argc > 1 ? atoi(argv[1]) : 2048;
    int cols = argc > 2 ? atoi(argv[2]) : 2048;
    int nt   = argc > 3 ? atoi(argv[3]) : 1;
    qwen_set_threads(nt);
    printf("prefill-bench  W=%dx%d (%.1f MB bf16)  threads=%d  AMX prototype=%s\n",
           rows, cols, rows * (double)cols * 2 / 1e6, nt, HAVE_AMX ? "yes" : "no (not an AMX build)");

    uint16_t *W = (uint16_t *)aligned_alloc(64, (size_t)rows * cols * sizeof(uint16_t));
    for (size_t i = 0; i < (size_t)rows * cols; i++) W[i] = f2b((float)(rnd() * 0.05));
    const int NMAX = 128;
    float *Xn = (float *)malloc((size_t)NMAX * cols * sizeof(float));
    for (size_t i = 0; i < (size_t)NMAX * cols; i++) Xn[i] = (float)(rnd() * 0.5);
    float *Y  = (float *)malloc((size_t)NMAX * rows * sizeof(float));
    float *xT = (float *)malloc((size_t)cols * 16 * sizeof(float));
    float *yT = (float *)malloc((size_t)rows * 16 * sizeof(float));

    /* First touch of an 8 MB weight matrix is a page-fault benchmark, not a kernel one. */
    { double a = 0, b = 0; for (int w = 0; w < 3; w++)
        path_current(Y, W, Xn, 32, cols, rows, xT, yT, &a, &b); }

    printf("\n  %4s %6s %10s %10s %10s   %s\n", "n", "calls", "total ms", "pack ms", "matmat ms",
           "ms per 16-position call");
    const int NS[] = { 14, 21, 29, 37, 53, 64, 128 };
    const int REP = 5;
    for (unsigned i = 0; i < sizeof NS / sizeof *NS; i++) {
        int n = NS[i], calls = (n + 15) / 16;
        double best = 1e30, bpk = 0, bmm = 0;
        for (int r = 0; r < REP; r++) {
            double pk = 0, mm = 0, t0 = now_ms();
            path_current(Y, W, Xn, n, cols, rows, xT, yT, &pk, &mm);
            double tot = now_ms() - t0;
            if (tot < best) { best = tot; bpk = pk; bmm = mm; }
        }
        printf("  %4d %6d %10.2f %10.2f %10.2f   %.2f\n", n, calls, best, bpk, bmm, bmm / calls);
    }
    printf("  (best of %d after warm-up; packing is the pack column, everything else is the "
           "call itself)\n", REP);

    /* The decomposition the table above cannot give: cost as a function of B. A prefill of n
       positions is one full call per 16 plus one partial call, so if cost(B) is flat the
       partial call is as expensive as a full one and the only lever is fewer calls. If instead
       cost(B) has steps, some widths are falling off a kernel gate and the lever is the gate. */
    printf("\n  cost of ONE matmat call against batch width\n");
    printf("  %4s %10s %10s   %s\n", "B", "ms", "ms/pos", "rel L2 vs B x matvec");
    float *yref = (float *)malloc((size_t)rows * sizeof(float));
    for (int B = 1; B <= 16; B++) {
        for (int b = 0; b < B; b++) {
            const float *xr = Xn + (int64_t)b * cols;
            for (int k = 0; k < cols; k++) xT[(int64_t)k * B + b] = xr[k];
        }
        qwen_matmat_bf16(yT, W, xT, rows, cols, B);          /* warm */
        /* Correctness at THIS width: the batched kernel must agree with the matvec it replaces.
           Relative L2 over the whole result, not a per-element ratio: with a per-element metric
           a single near-zero reference value makes the error explode and reports a mismatch
           that is an artefact of the denominator. This is the measure --self-test uses. */
        double l2n = 0, l2d = 0;
        for (int b = 0; b < B; b++) {
            qwen_matvec_bf16(yref, W, Xn + (int64_t)b * cols, rows, cols);
            for (int r = 0; r < rows; r++) {
                double d = (double)yT[(size_t)r * B + b] - yref[r];
                l2n += d * d; l2d += (double)yref[r] * yref[r];
            }
        }
        double worst = l2d > 0 ? sqrt(l2n / l2d) : 0.0;
        double best = 1e30;
        for (int r = 0; r < 5; r++) {
            double t0 = now_ms();
            qwen_matmat_bf16(yT, W, xT, rows, cols, B);
            double d = now_ms() - t0;
            if (d < best) best = d;
        }
        printf("  %4d %10.2f %10.3f   %s (%.1e)\n", B, best, best / B,
               worst < 3e-2 ? "OK" : "MISMATCH", worst);
        if (worst >= 3e-2) fail = 1;
    }
    free(yref);
    printf("  %s\n", fail ? "FAIL: a batch width disagrees with the matvec reference"
                           : "PASS: every batch width agrees with the matvec reference");

    /* Same question for int8, which is the precision deployments actually run. */
    {
        int8_t *Wq = (int8_t *)malloc((size_t)rows * cols);
        float *sc = (float *)malloc((size_t)rows * sizeof(float));
        for (int r = 0; r < rows; r++) {
            sc[r] = (float)(0.002 + 0.001 * fabs(rnd()));
            for (int k = 0; k < cols; k++) Wq[(size_t)r * cols + k] = (int8_t)(rnd() * 127.0);
        }
        printf("\n  the same, int8 (correctness for int8 lives in make check-matmat-parity and\n"
               "  --self-test, which compare against an integer reference; this is timing only)\n");
        printf("  %4s %10s %10s\n", "B", "ms", "ms/pos");
        for (int B = 1; B <= 16; B++) {
            for (int b = 0; b < B; b++) {
                const float *xr = Xn + (int64_t)b * cols;
                for (int k = 0; k < cols; k++) xT[(int64_t)k * B + b] = xr[k];
            }
            qwen_matmat_int8(yT, Wq, sc, xT, rows, cols, B);
            double best = 1e30;
            for (int r = 0; r < 5; r++) {
                double t0 = now_ms();
                qwen_matmat_int8(yT, Wq, sc, xT, rows, cols, B);
                double d = now_ms() - t0;
                if (d < best) best = d;
            }
            printf("  %4d %10.2f %10.3f\n", B, best, best / B);
        }
        free(Wq); free(sc);
    }

#if HAVE_AMX
    if (nt == 1) {
        printf("\n  B=32 prototype (one weight pass per 32 positions) vs two B=16 calls\n");
        int kfull = cols & ~31;
        uint16_t *Xb = (uint16_t *)malloc((size_t)32 * cols * sizeof(uint16_t));
        uint16_t *pA = (uint16_t *)aligned_alloc(64, (size_t)kfull * 16 * sizeof(uint16_t));
        uint16_t *pB = (uint16_t *)aligned_alloc(64, (size_t)kfull * 16 * sizeof(uint16_t));
        float *Y32 = (float *)malloc((size_t)rows * 32 * sizeof(float));
        for (int b = 0; b < 32; b++)
            for (int k = 0; k < cols; k++) Xb[(size_t)b * cols + k] = f2b(Xn[(int64_t)b * cols + k]);
        double pack32 = 1e30, kern32 = 1e30, cur32 = 1e30, pk = 0, mm = 0;
        for (int r = 0; r < 5; r++) {
            double p0 = now_ms();
            pack_act(pA, Xb, cols, kfull, 16);
            pack_act(pB, Xb + (size_t)16 * cols, cols, kfull, 16);
            double p1 = now_ms();
            amx_b32(Y32, W, pA, pB, rows, cols, 32);
            double p2 = now_ms();
            if (p1 - p0 < pack32) pack32 = p1 - p0;
            if (p2 - p1 < kern32) kern32 = p2 - p1;
            double a = 0, b = 0, t0 = now_ms();
            path_current(Y, W, Xn, 32, cols, rows, xT, yT, &a, &b);
            double t1 = now_ms() - t0;
            if (t1 < cur32) { cur32 = t1; pk = a; mm = b; }
        }

        /* correctness before speed: the prototype must agree with the shipped path */
        double worst = 0;
        for (int b = 0; b < 32; b++)
            for (int r = 0; r < (rows & ~15); r++) {
                double got = Y32[(size_t)r * 32 + b], ref = Y[(int64_t)b * rows + r];
                double d = fabs(got - ref) / (fabs(ref) + 1e-3);
                if (d > worst) worst = d;
            }
        printf("    correctness vs shipped path: worst relative %.2e  %s\n", worst,
               worst < 2e-2 ? "OK (both go through bf16)" : "MISMATCH - the prototype is wrong");
        printf("    shipped  (2 x B=16): %7.2f ms   (pack %.2f + matmat %.2f)\n", cur32, pk, mm);
        printf("    prototype (1 x B=32): %7.2f ms   (pack %.2f + kernel %.2f)\n",
               pack32 + kern32, pack32, kern32);
        printf("    => %+.1f%% on the 32-position case\n",
               100.0 * ((pack32 + kern32) - cur32) / cur32);
    }
#endif
    return fail;
}
