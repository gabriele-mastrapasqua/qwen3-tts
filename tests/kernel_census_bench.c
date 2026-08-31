/* kernel_census_bench.c - shape-aware kernel census against Arm KleidiAI */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>
#include <float.h>

#include "../qwen_tts_kernels.h"
#include "../qwen_tts_thread.h"

#if defined(__aarch64__) && defined(__ARM_FEATURE_MATMUL_INT8) && defined(__ARM_FEATURE_DOTPROD)
#define KAI_HAVE_INT8 1
#include "kai/kai_common.h"
#include "kai/ukernels/matmul/pack/kai_lhs_quant_pack_qai8dxp_f32.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qsi8cxp_qsi8cx_neon.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp_qsi8cxp_interface.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm.h"
#else
#define KAI_HAVE_INT8 0
#endif

#if defined(__aarch64__) && defined(__ARM_FEATURE_BF16)
#define KAI_HAVE_BF16 1
#include "kai/kai_common.h"
#include "kai/ukernels/matmul/pack/kai_lhs_quant_pack_bf16p1x4_f32_neon.h"
#include "kai/ukernels/matmul/pack/kai_lhs_quant_pack_bf16p8x4_f32_neon.h"
#include "kai/ukernels/matmul/pack/kai_rhs_quant_pack_kxn_bf16p12x4biasf32_f32_neon.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_bf16p_bf16p/kai_matmul_clamp_f32_bf16p_bf16p_interface.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_bf16p_bf16p/kai_matmul_clamp_f32_bf16p1x4_bf16p12x4b_1x36_neon_dot.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_bf16p_bf16p/kai_matmul_clamp_f32_bf16p8x4_bf16p12x4b_8x12_neon_mmla.h"
#else
#define KAI_HAVE_BF16 0
#endif

typedef struct { const char *comp, *op; int N, K; double calls_per_frame; } shape_t;

static const shape_t SHAPES_17B[] = {
    { "talker", "qkv",      4096, 2048, 28.00 },
    { "talker", "o_proj",   2048, 2048, 28.00 },
    { "talker", "gate_up", 12288, 2048, 28.00 },
    { "talker", "down",     2048, 6144, 28.00 },
    { "talker", "codec_head", 3072, 2048, 1.00 },
    { "cp",     "qkv",      4096, 1024, 75.00 },
    { "cp",     "o_proj",   1024, 2048, 75.00 },
    { "cp",     "gate_up",  6144, 1024, 75.00 },
    { "cp",     "down",     1024, 3072, 75.00 },
    { "cp",     "lm_head",  2048, 1024, 15.00 },
    { "cp",     "mtp_proj", 1024, 2048,  1.00 },
};
static const shape_t SHAPES_06B[] = {
    { "talker", "qkv",      4096, 1024, 27.45 },
    { "talker", "o_proj",   1024, 2048, 27.75 },
    { "talker", "gate_up",  6144, 1024, 27.45 },
    { "talker", "down",     1024, 3072, 27.45 },
    { "talker", "codec_head", 3072, 1024, 1.00 },
    { "cp",     "qkv",      4096, 1024, 78.43 },
    { "cp",     "o_proj",   1024, 2048, 78.43 },
    { "cp",     "gate_up",  6144, 1024, 78.43 },
    { "cp",     "down",     1024, 3072, 78.43 },
    { "cp",     "lm_head",  2048, 1024, 14.71 },
};
static const shape_t PREFILL_17B[] = {
    { "prefill", "wq",      2048, 2048, 28.0 },
    { "prefill", "wk",      1024, 2048, 28.0 },
    { "prefill", "wv",      1024, 2048, 28.0 },
    { "prefill", "wo",      2048, 2048, 28.0 },
    { "prefill", "gate_up",12288, 2048, 28.0 },
    { "prefill", "down",    2048, 6144, 28.0 },
};
static const shape_t PREFILL_06B[] = {
    { "prefill", "wq",      2048, 1024, 28.0 },
    { "prefill", "wk",      1024, 1024, 28.0 },
    { "prefill", "wv",      1024, 1024, 28.0 },
    { "prefill", "wo",      1024, 2048, 28.0 },
    { "prefill", "gate_up", 6144, 1024, 28.0 },
    { "prefill", "down",    1024, 3072, 28.0 },
};

static double now_ns(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec * 1e9 + (double)t.tv_nsec;
}
static int cmp_dbl(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return x < y ? -1 : (x > y ? 1 : 0);
}
static double median(double *v, int n) {
    qsort(v, (size_t)n, sizeof(double), cmp_dbl);
    return n & 1 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}
static void *xaligned(size_t bytes) {
    void *p = NULL;
    if (bytes == 0) bytes = 64;
    if (posix_memalign(&p, 64, (bytes + 63) & ~(size_t)63) != 0) {
        fprintf(stderr, "alloc failed (%zu bytes)\n", bytes); exit(1);
    }
    memset(p, 0, (bytes + 63) & ~(size_t)63);
    return p;
}
static uint32_t rng_s = 0x2b3c4d5eu;
static float frand(void) {
    rng_s = rng_s * 1664525u + 1013904223u;
    return ((float)(rng_s >> 8) / (float)(1u << 24)) * 2.0f - 1.0f;
}

typedef struct {
    int N, K;
    float    *Wf32;
    uint16_t *Wbf16;
    int8_t   *Wi8;
    float    *Wscale;
    float    *Wi8_deq;
    float    *Wbf16_deq;
} weights_t;

static void weights_build(weights_t *w, int N, int K) {
    w->N = N; w->K = K;
    w->Wf32      = (float *)xaligned((size_t)N * K * sizeof(float));
    w->Wbf16     = (uint16_t *)xaligned((size_t)N * K * sizeof(uint16_t));
    w->Wi8       = (int8_t *)xaligned((size_t)N * K);
    w->Wscale    = (float *)xaligned((size_t)N * sizeof(float));
    w->Wi8_deq   = (float *)xaligned((size_t)N * K * sizeof(float));
    w->Wbf16_deq = (float *)xaligned((size_t)N * K * sizeof(float));
    for (int n = 0; n < N; n++) {
        float amax = 0.0f;
        float *row = w->Wf32 + (size_t)n * K;
        for (int k = 0; k < K; k++) {
            row[k] = frand() * 0.05f;
            float a = fabsf(row[k]); if (a > amax) amax = a;
        }
        float s = amax > 0.0f ? amax / 127.0f : 1.0f;
        w->Wscale[n] = s;
        for (int k = 0; k < K; k++) {
            int v = (int)lrintf(row[k] / s);
            if (v > 127) v = 127; if (v < -128) v = -128;
            w->Wi8[(size_t)n * K + k] = (int8_t)v;
            w->Wi8_deq[(size_t)n * K + k] = (float)v * s;
            uint32_t u; memcpy(&u, &row[k], 4);
            uint16_t b = (uint16_t)(u >> 16);
            w->Wbf16[(size_t)n * K + k] = b;
            uint32_t back = (uint32_t)b << 16;
            memcpy(&w->Wbf16_deq[(size_t)n * K + k], &back, 4);
        }
    }
}
static void weights_free(weights_t *w) {
    free(w->Wf32); free(w->Wbf16); free(w->Wi8); free(w->Wscale);
    free(w->Wi8_deq); free(w->Wbf16_deq);
}

static void acts_build(int K, int B, float **Xbk, float **Xkb) {
    float *a = (float *)xaligned((size_t)K * B * sizeof(float));
    float *t = (float *)xaligned((size_t)K * B * sizeof(float));
    for (int b = 0; b < B; b++)
        for (int k = 0; k < K; k++) {
            float v = frand();
            a[(size_t)b * K + k] = v;
            t[(size_t)k * B + b] = v;
        }
    *Xbk = a; *Xkb = t;
}
static double *reference(const float *W, const float *Xkb, int N, int K, int B) {
    double *R = (double *)xaligned((size_t)N * B * sizeof(double));
    for (int n = 0; n < N; n++)
        for (int b = 0; b < B; b++) {
            double acc = 0.0;
            const float *w = W + (size_t)n * K;
            for (int k = 0; k < K; k++) acc += (double)w[k] * (double)Xkb[(size_t)k * B + b];
            R[(size_t)n * B + b] = acc;
        }
    return R;
}
static double rel_err(const float *Y, const double *R, int N, int B) {
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < (size_t)N * B; i++) {
        double d = (double)Y[i] - R[i];
        num += d * d; den += R[i] * R[i];
    }
    return den > 0.0 ? sqrt(num / den) : 0.0;
}

static FILE *g_csv = NULL;
static void csv_header(void) {
    fprintf(g_csv, "comp,op,dtype,N,K,B,calls_per_frame,impl,mode,isa,tile,mr,nr,kr,"
                   "m_step,n_step,threads,rhs_pack_ns,rhs_pack_bytes,"
                   "gather_ns,lhspack_ns,kernel_ns,out_ns,ns_compare,ns_dropin,"
                   "gmac_s,err_vs_f32,err_vs_qw,speedup\n");
    fflush(g_csv);
}
typedef struct {
    const char *comp, *op, *dtype, *impl, *mode, *isa, *tile;
    int N, K, B, mr, nr, kr, m_step, n_step, threads;
    double calls_per_frame, rhs_pack_ns; size_t rhs_pack_bytes;
    double gather, lhspack, kernel, out;
    double ns_compare;
    double ns_dropin;
    double err_f32, err_qw;
} row_t;
static void row_emit(const row_t *r, double baseline_ns) {
    double g = (double)r->N * r->K * r->B / (r->ns_compare > 0 ? r->ns_compare : 1);
    fprintf(g_csv, "%s,%s,%s,%d,%d,%d,%.2f,%s,%s,%s,%s,%d,%d,%d,%d,%d,%d,"
                   "%.0f,%zu,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.3f,%.3e,%.3e,%.3f\n",
            r->comp, r->op, r->dtype, r->N, r->K, r->B, r->calls_per_frame,
            r->impl, r->mode, r->isa, r->tile, r->mr, r->nr, r->kr,
            r->m_step, r->n_step, r->threads,
            r->rhs_pack_ns, r->rhs_pack_bytes,
            r->gather, r->lhspack, r->kernel, r->out, r->ns_compare, r->ns_dropin,
            g, r->err_f32, r->err_qw,
            baseline_ns > 0 && r->ns_compare > 0 ? baseline_ns / r->ns_compare : 0.0);
    fflush(g_csv);
}

static void eng_gather(float *Xt, const float *src, int n, int dim) {
    for (int j = 0; j < n; j++) {
        const float *s = src + (size_t)j * dim;
        for (int k = 0; k < dim; k++) Xt[(size_t)k * n + j] = s[k];
    }
}
static void eng_scatter(float *dst, const float *Yt, int n, int rows) {
    for (int r = 0; r < rows; r++) {
        const float *yr = Yt + (size_t)r * n;
        for (int j = 0; j < n; j++) dst[(size_t)j * rows + r] = yr[j];
    }
}
static void kai_gather(float *Xbk, const float *src, int n, int dim) {
    for (int j = 0; j < n; j++)
        memcpy(Xbk + (size_t)j * dim, src + (size_t)j * dim, (size_t)dim * sizeof(float));
}
static void kai_scatter(float *dst, const float *D, int n, int rows) {
    for (int j = 0; j < n; j++)
        memcpy(dst + (size_t)j * rows, D + (size_t)j * rows, (size_t)rows * sizeof(float));
}
static void xpose_in(float *dst, const float *Xkb, int K, int B) {
    if (B == 1) { memcpy(dst, Xkb, (size_t)K * sizeof(float)); return; }
    for (int b = 0; b < B; b++)
        for (int k = 0; k < K; k++) dst[(size_t)b * K + k] = Xkb[(size_t)k * B + b];
}
static void xpose_out(float *Y, const float *D, int N, int B) {
    if (B == 1) { memcpy(Y, D, (size_t)N * sizeof(float)); return; }
    for (int n = 0; n < N; n++)
        for (int b = 0; b < B; b++) Y[(size_t)n * B + b] = D[(size_t)b * N + n];
}

static double bench_our(int int8, const weights_t *w, const float *Xkb, float *Y,
                        int B, int iters, int reps) {
    double *t = (double *)xaligned((size_t)reps * sizeof(double));
    for (int r = 0; r < reps; r++) {
        double t0 = now_ns();
        for (int i = 0; i < iters; i++) {
            if (int8) {
                if (B == 1) qwen_matvec_int8(Y, w->Wi8, w->Wscale, Xkb, w->N, w->K);
                else        qwen_matmat_int8(Y, w->Wi8, w->Wscale, Xkb, w->N, w->K, B);
            } else {
                if (B == 1) qwen_matvec_bf16(Y, w->Wbf16, Xkb, w->N, w->K);
                else        qwen_matmat_bf16(Y, w->Wbf16, Xkb, w->N, w->K, B);
            }
        }
        t[r] = (now_ns() - t0) / iters;
    }
    double m = median(t, reps); free(t);
    return m;
}

#if KAI_HAVE_INT8
typedef struct {
    const char *name, *isa, *tile;
    struct kai_matmul_clamp_f32_qai8dxp_qsi8cxp_ukernel uk;
} kai_i8_cand_t;
static kai_i8_cand_t KAI_I8[] = {
#define UKI(sfx) { \
    kai_get_m_step_matmul_clamp_f32_##sfx, kai_get_n_step_matmul_clamp_f32_##sfx, \
    kai_get_mr_matmul_clamp_f32_##sfx,     kai_get_nr_matmul_clamp_f32_##sfx, \
    kai_get_kr_matmul_clamp_f32_##sfx,     kai_get_sr_matmul_clamp_f32_##sfx, \
    kai_get_lhs_packed_offset_matmul_clamp_f32_##sfx, \
    kai_get_rhs_packed_offset_matmul_clamp_f32_##sfx, \
    kai_get_dst_offset_matmul_clamp_f32_##sfx, \
    kai_get_dst_size_matmul_clamp_f32_##sfx, \
    kai_run_matmul_clamp_f32_##sfx }
    { "qai8dxp1x4_qsi8cxp4x4_1x4_dotprod",  "dotprod", "1x4",
      UKI(qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod) },
    { "qai8dxp1x8_qsi8cxp4x8_1x4_dotprod",  "dotprod", "1x4",
      UKI(qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod) },
    { "qai8dxp4x4_qsi8cxp4x4_16x4_dotprod", "dotprod", "16x4",
      UKI(qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod) },
    { "qai8dxp4x8_qsi8cxp4x8_16x4_i8mm",    "i8mm",    "16x4",
      UKI(qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm) },
#undef UKI
};
#define KAI_I8_N ((int)(sizeof(KAI_I8) / sizeof(KAI_I8[0])))

typedef struct {
    const kai_i8_cand_t *c; int M, N, K;
    const void *lhs_p, *rhs_p; float *dst;
} kai_i8_job_t;
static void kai_i8_task(size_t tid, size_t nt, void *vc) {
    kai_i8_job_t *j = (kai_i8_job_t *)vc;
    const size_t ns = j->c->uk.get_n_step();
    size_t nblk = ((size_t)j->N + ns - 1) / ns;
    size_t n0 = tid * nblk / nt * ns, n1 = (tid + 1) * nblk / nt * ns;
    if (n1 > (size_t)j->N) n1 = (size_t)j->N;
    if (n0 >= n1) return;
    const uint8_t *rhs = (const uint8_t *)j->rhs_p + j->c->uk.get_rhs_packed_offset(n0, (size_t)j->K);
    float *dst = (float *)((uint8_t *)j->dst +
                 j->c->uk.get_dst_offset(0, n0, (size_t)j->N * sizeof(float)));
    j->c->uk.run_matmul((size_t)j->M, n1 - n0, (size_t)j->K, j->lhs_p, rhs, dst,
                        (size_t)j->N * sizeof(float), sizeof(float), -FLT_MAX, FLT_MAX);
}
static void *kai_i8_pack_rhs(const weights_t *w, size_t nr, size_t kr, size_t sr,
                             size_t *bytes, double *ns) {
    *bytes = kai_get_rhs_packed_size_rhs_pack_nxk_qsi8cxp_qsi8cx_neon((size_t)w->N, (size_t)w->K, nr, kr, sr);
    void *p = xaligned(*bytes);
    struct kai_rhs_pack_qsi8cx_params rp = { .lhs_zero_point = 1, .scale_multiplier = 1.0f };
    double t0 = now_ns();
    kai_run_rhs_pack_nxk_qsi8cxp_qsi8cx_neon(1, (size_t)w->N, (size_t)w->K, nr, kr, sr,
                                             w->Wi8, NULL, w->Wscale, p, 0, &rp);
    *ns = now_ns() - t0;
    return p;
}
#endif

#if KAI_HAVE_BF16
typedef struct {
    const char *name, *isa, *tile;
    struct kai_matmul_clamp_f32_bf16p_bf16p_ukernel uk;
    void (*lhs_pack)(size_t, size_t, size_t, size_t, size_t, size_t, const void *, size_t, void *);
    size_t (*lhs_size)(size_t, size_t, size_t, size_t, size_t);
    int only_m1;
} kai_bf_cand_t;
static kai_bf_cand_t KAI_BF[] = {
#define UKB(sfx) { \
    kai_get_m_step_matmul_clamp_f32_##sfx, kai_get_n_step_matmul_clamp_f32_##sfx, \
    kai_get_mr_matmul_clamp_f32_##sfx,     kai_get_nr_matmul_clamp_f32_##sfx, \
    kai_get_kr_matmul_clamp_f32_##sfx,     kai_get_sr_matmul_clamp_f32_##sfx, \
    kai_get_lhs_packed_offset_matmul_clamp_f32_##sfx, \
    kai_get_rhs_packed_offset_matmul_clamp_f32_##sfx, \
    kai_get_dst_offset_matmul_clamp_f32_##sfx, \
    kai_get_dst_size_matmul_clamp_f32_##sfx, \
    kai_run_matmul_clamp_f32_##sfx }
    { "bf16p1x4_bf16p12x4b_1x36_dot",  "neon_dot",  "1x36",
      UKB(bf16p1x4_bf16p12x4b_1x36_neon_dot),
      kai_run_lhs_quant_pack_bf16p1x4_f32_neon,
      kai_get_lhs_packed_size_lhs_quant_pack_bf16p1x4_f32_neon, 1 },
    { "bf16p8x4_bf16p12x4b_8x12_mmla", "neon_mmla", "8x12",
      UKB(bf16p8x4_bf16p12x4b_8x12_neon_mmla),
      kai_run_lhs_quant_pack_bf16p8x4_f32_neon,
      kai_get_lhs_packed_size_lhs_quant_pack_bf16p8x4_f32_neon, 0 },
#undef UKB
};
#define KAI_BF_N ((int)(sizeof(KAI_BF) / sizeof(KAI_BF[0])))
typedef struct {
    const kai_bf_cand_t *c; int M, N, K;
    const void *lhs_p, *rhs_p; float *dst;
} kai_bf_job_t;
static void kai_bf_task(size_t tid, size_t nt, void *vc) {
    kai_bf_job_t *j = (kai_bf_job_t *)vc;
    const size_t ns = j->c->uk.get_n_step();
    size_t nblk = ((size_t)j->N + ns - 1) / ns;
    size_t n0 = tid * nblk / nt * ns, n1 = (tid + 1) * nblk / nt * ns;
    if (n1 > (size_t)j->N) n1 = (size_t)j->N;
    if (n0 >= n1) return;
    const uint8_t *rhs = (const uint8_t *)j->rhs_p + j->c->uk.get_rhs_packed_offset(n0, (size_t)j->K);
    float *dst = (float *)((uint8_t *)j->dst +
                 j->c->uk.get_dst_offset(0, n0, (size_t)j->N * sizeof(float)));
    j->c->uk.run_matmul((size_t)j->M, n1 - n0, (size_t)j->K, j->lhs_p, rhs, dst,
                        (size_t)j->N * sizeof(float), sizeof(float), -FLT_MAX, FLT_MAX);
}
static void *kai_bf_pack_rhs(const weights_t *w, size_t nr, size_t kr, size_t sr,
                             size_t *bytes, double *ns) {
    *bytes = kai_get_rhs_packed_size_rhs_quant_pack_kxn_bf16p12x4biasf32_f32_neon(
                 (size_t)w->N, (size_t)w->K, nr, kr);
    void  *p    = xaligned(*bytes);
    float *bias = (float *)xaligned((size_t)w->N * sizeof(float));
    float *WkxN = (float *)xaligned((size_t)w->N * w->K * sizeof(float));
    double t0 = now_ns();
    for (int n = 0; n < w->N; n++)
        for (int k = 0; k < w->K; k++)
            WkxN[(size_t)k * w->N + n] = w->Wf32[(size_t)n * w->K + k];
    kai_run_rhs_quant_pack_kxn_bf16p12x4biasf32_f32_neon(
        1, (size_t)w->N, (size_t)w->K, nr, kr, sr,
        (size_t)w->N * sizeof(float), WkxN, bias, NULL, p, 0, NULL);
    *ns = now_ns() - t0;
    free(bias); free(WkxN);
    return p;
}
#endif

static int g_threads = 1, g_reps = 5, g_iters = 0, g_check = 1;

static int iters_for(double macs) {
    int it = g_iters > 0 ? g_iters : (int)(2e9 / (macs > 0 ? macs : 1));
    if (it < 3) it = 3;
    if (it > 2000) it = 2000;
    return it;
}

static void cell(const shape_t *S, const weights_t *w, int B, int int8) {
    float *Xbk, *Xkb;
    acts_build(w->K, B, &Xbk, &Xkb);
    float *Y   = (float *)xaligned((size_t)w->N * B * sizeof(float));
    float *Yk  = (float *)xaligned((size_t)w->N * B * sizeof(float));
    float *dst = (float *)xaligned((size_t)w->N * B * sizeof(float));
    float *Xtm = (float *)xaligned((size_t)w->K * B * sizeof(float));
    int it = iters_for((double)w->N * w->K * B);

    double *ref_f32 = NULL, *ref_qw = NULL;
    if (g_check) {
        ref_f32 = reference(w->Wf32, Xkb, w->N, w->K, B);
        ref_qw  = reference(int8 ? w->Wi8_deq : w->Wbf16_deq, Xkb, w->N, w->K, B);
    }
    row_t base;
    memset(&base, 0, sizeof base);
    base.comp = S->comp; base.op = S->op; base.dtype = int8 ? "int8" : "bf16";
    base.N = w->N; base.K = w->K; base.B = B; base.threads = g_threads;
    base.calls_per_frame = S->calls_per_frame;

    bench_our(int8, w, Xkb, Y, B, 1, 1);
    double ours = bench_our(int8, w, Xkb, Y, B, it, g_reps);
    {
        row_t r = base;
        r.impl = "OURS"; r.mode = "engine";
        r.isa = int8 ? "dotprod/smmla" : "bfmmla"; r.tile = "-";
        double t0 = now_ns();
        for (int i = 0; i < it; i++) eng_gather(Xtm, Xbk, B, w->K);
        r.gather = (now_ns() - t0) / it;
        t0 = now_ns();
        for (int i = 0; i < it; i++) eng_scatter(dst, Y, B, w->N);
        r.out = (now_ns() - t0) / it;
        r.kernel = ours; r.ns_compare = ours; r.ns_dropin = r.gather + ours + r.out;
        if (g_check) { r.err_f32 = rel_err(Y, ref_f32, w->N, B);
                       r.err_qw  = rel_err(Y, ref_qw,  w->N, B); }
        row_emit(&r, ours);
    }

#if KAI_HAVE_INT8
    if (int8) for (int c = 0; c < KAI_I8_N; c++) {
        kai_i8_cand_t *cd = &KAI_I8[c];
        const size_t mr = cd->uk.get_mr(), nr = cd->uk.get_nr();
        const size_t kr = cd->uk.get_kr(), sr = cd->uk.get_sr();
        size_t rhs_bytes; double rhs_ns;
        void *rhs_p = kai_i8_pack_rhs(w, nr, kr, sr, &rhs_bytes, &rhs_ns);
        void *lhs_p = xaligned(kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_f32(
                                   (size_t)B, (size_t)w->K, mr, kr, sr));
        float *lhsT = (float *)xaligned((size_t)B * w->K * sizeof(float));
        float *D    = (float *)xaligned((size_t)B * w->N * sizeof(float));

        for (int mode = 0; mode < 2; mode++) {
            double *tg = (double *)xaligned((size_t)g_reps * sizeof(double));
            double *tp = (double *)xaligned((size_t)g_reps * sizeof(double));
            double *tk = (double *)xaligned((size_t)g_reps * sizeof(double));
            double *to = (double *)xaligned((size_t)g_reps * sizeof(double));
            for (int rr = 0; rr < g_reps; rr++) {
                double a = 0, bb = 0, cc = 0, dd = 0, t0;
                for (int i = 0; i < it; i++) {
                    t0 = now_ns();
                    if (mode) xpose_in(lhsT, Xkb, w->K, B); else kai_gather(lhsT, Xbk, B, w->K);
                    a += now_ns() - t0;
                    t0 = now_ns();
                    kai_run_lhs_quant_pack_qai8dxp_f32((size_t)B, (size_t)w->K, mr, kr, sr, 0,
                                                       lhsT, (size_t)w->K * sizeof(float), lhs_p);
                    bb += now_ns() - t0;
                    t0 = now_ns();
                    kai_i8_job_t job = { cd, B, w->N, w->K, lhs_p, rhs_p, D };
                    if (g_threads > 1) qwen_parallel((size_t)g_threads, kai_i8_task, &job);
                    else               kai_i8_task(0, 1, &job);
                    cc += now_ns() - t0;
                    t0 = now_ns();
                    if (mode) xpose_out(Yk, D, w->N, B); else kai_scatter(dst, D, B, w->N);
                    dd += now_ns() - t0;
                }
                tg[rr] = a / it; tp[rr] = bb / it; tk[rr] = cc / it; to[rr] = dd / it;
            }
            row_t r = base;
            r.impl = cd->name; r.mode = mode ? "xpose" : "native";
            r.isa = cd->isa; r.tile = cd->tile;
            r.mr = (int)mr; r.nr = (int)nr; r.kr = (int)kr;
            r.m_step = (int)cd->uk.get_m_step(); r.n_step = (int)cd->uk.get_n_step();
            r.rhs_pack_ns = rhs_ns; r.rhs_pack_bytes = rhs_bytes;
            r.gather = median(tg, g_reps); r.lhspack = median(tp, g_reps);
            r.kernel = median(tk, g_reps); r.out = median(to, g_reps);
            r.ns_compare = r.lhspack + r.kernel;
            r.ns_dropin  = r.gather + r.lhspack + r.kernel + r.out;
            if (g_check) {
                xpose_out(Yk, D, w->N, B);
                r.err_f32 = rel_err(Yk, ref_f32, w->N, B);
                r.err_qw  = rel_err(Yk, ref_qw,  w->N, B);
            }
            row_emit(&r, ours);
            free(tg); free(tp); free(tk); free(to);
        }
        free(rhs_p); free(lhs_p); free(lhsT); free(D);
    }
#endif
#if KAI_HAVE_BF16
    if (!int8) for (int c = 0; c < KAI_BF_N; c++) {
        kai_bf_cand_t *cd = &KAI_BF[c];
        if (cd->only_m1 && B != 1) continue;
        const size_t mr = cd->uk.get_mr(), nr = cd->uk.get_nr();
        const size_t kr = cd->uk.get_kr(), sr = cd->uk.get_sr();
        size_t rhs_bytes; double rhs_ns;
        void *rhs_p = kai_bf_pack_rhs(w, nr, kr, sr, &rhs_bytes, &rhs_ns);
        void *lhs_p = xaligned(cd->lhs_size((size_t)B, (size_t)w->K, mr, kr, sr));
        float *lhsT = (float *)xaligned((size_t)B * w->K * sizeof(float));
        float *D    = (float *)xaligned((size_t)B * w->N * sizeof(float));

        for (int mode = 0; mode < 2; mode++) {
            double *tg = (double *)xaligned((size_t)g_reps * sizeof(double));
            double *tp = (double *)xaligned((size_t)g_reps * sizeof(double));
            double *tk = (double *)xaligned((size_t)g_reps * sizeof(double));
            double *to = (double *)xaligned((size_t)g_reps * sizeof(double));
            for (int rr = 0; rr < g_reps; rr++) {
                double a = 0, bb = 0, cc = 0, dd = 0, t0;
                for (int i = 0; i < it; i++) {
                    t0 = now_ns();
                    if (mode) xpose_in(lhsT, Xkb, w->K, B); else kai_gather(lhsT, Xbk, B, w->K);
                    a += now_ns() - t0;
                    t0 = now_ns();
                    cd->lhs_pack((size_t)B, (size_t)w->K, mr, kr, sr, 0,
                                 lhsT, (size_t)w->K * sizeof(float), lhs_p);
                    bb += now_ns() - t0;
                    t0 = now_ns();
                    kai_bf_job_t job = { cd, B, w->N, w->K, lhs_p, rhs_p, D };
                    if (g_threads > 1) qwen_parallel((size_t)g_threads, kai_bf_task, &job);
                    else               kai_bf_task(0, 1, &job);
                    cc += now_ns() - t0;
                    t0 = now_ns();
                    if (mode) xpose_out(Yk, D, w->N, B); else kai_scatter(dst, D, B, w->N);
                    dd += now_ns() - t0;
                }
                tg[rr] = a / it; tp[rr] = bb / it; tk[rr] = cc / it; to[rr] = dd / it;
            }
            row_t r = base;
            r.impl = cd->name; r.mode = mode ? "xpose" : "native";
            r.isa = cd->isa; r.tile = cd->tile;
            r.mr = (int)mr; r.nr = (int)nr; r.kr = (int)kr;
            r.m_step = (int)cd->uk.get_m_step(); r.n_step = (int)cd->uk.get_n_step();
            r.rhs_pack_ns = rhs_ns; r.rhs_pack_bytes = rhs_bytes;
            r.gather = median(tg, g_reps); r.lhspack = median(tp, g_reps);
            r.kernel = median(tk, g_reps); r.out = median(to, g_reps);
            r.ns_compare = r.lhspack + r.kernel;
            r.ns_dropin  = r.gather + r.lhspack + r.kernel + r.out;
            if (g_check) {
                xpose_out(Yk, D, w->N, B);
                r.err_f32 = rel_err(Yk, ref_f32, w->N, B);
                r.err_qw  = rel_err(Yk, ref_qw,  w->N, B);
            }
            row_emit(&r, ours);
            free(tg); free(tp); free(tk); free(to);
        }
        free(rhs_p); free(lhs_p); free(lhsT); free(D);
    }
#endif
    free(ref_f32); free(ref_qw);
    free(Xbk); free(Xkb); free(Y); free(Yk); free(dst); free(Xtm);
}

static void qkv_pass(const char *comp, int Nq, int Nk, int K, const int *BS, int nBS) {
#if !KAI_HAVE_INT8
    (void)comp; (void)Nq; (void)Nk; (void)K; (void)BS; (void)nBS;
#else
    weights_t wq, wk, wv;
    weights_build(&wq, Nq, K); weights_build(&wk, Nk, K); weights_build(&wv, Nk, K);
    for (int bi = 0; bi < nBS; bi++) {
        int B = BS[bi];
        if (B <= 0) continue;
        kai_i8_cand_t *cd = &KAI_I8[B == 1 ? 1 : 3];
        const size_t mr = cd->uk.get_mr(), nr = cd->uk.get_nr();
        const size_t kr = cd->uk.get_kr(), sr = cd->uk.get_sr();
        float *Xbk, *Xkb; acts_build(K, B, &Xbk, &Xkb);
        int it = iters_for((double)(Nq + 2 * Nk) * K * B);
        size_t rb; double rn;
        void *rp_q = kai_i8_pack_rhs(&wq, nr, kr, sr, &rb, &rn);
        void *rp_k = kai_i8_pack_rhs(&wk, nr, kr, sr, &rb, &rn);
        void *rp_v = kai_i8_pack_rhs(&wv, nr, kr, sr, &rb, &rn);
        void *lp = xaligned(kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_f32(
                                (size_t)B, (size_t)K, mr, kr, sr));
        float *Dq = (float *)xaligned((size_t)B * Nq * sizeof(float));
        float *Dk = (float *)xaligned((size_t)B * Nk * sizeof(float));
        float *Dv = (float *)xaligned((size_t)B * Nk * sizeof(float));
        float *Yq = (float *)xaligned((size_t)B * Nq * sizeof(float));
        float *Yk2 = (float *)xaligned((size_t)B * Nk * sizeof(float));
        float *Yv = (float *)xaligned((size_t)B * Nk * sizeof(float));
        double *t = (double *)xaligned((size_t)g_reps * sizeof(double));

        bench_our(1, &wq, Xkb, Yq, B, 1, 1);
        for (int rr = 0; rr < g_reps; rr++) {
            double t0 = now_ns();
            for (int i = 0; i < it; i++) {
                if (B == 1) { qwen_matvec_int8(Yq,  wq.Wi8, wq.Wscale, Xkb, Nq, K);
                              qwen_matvec_int8(Yk2, wk.Wi8, wk.Wscale, Xkb, Nk, K);
                              qwen_matvec_int8(Yv,  wv.Wi8, wv.Wscale, Xkb, Nk, K); }
                else        { qwen_matmat_int8(Yq,  wq.Wi8, wq.Wscale, Xkb, Nq, K, B);
                              qwen_matmat_int8(Yk2, wk.Wi8, wk.Wscale, Xkb, Nk, K, B);
                              qwen_matmat_int8(Yv,  wv.Wi8, wv.Wscale, Xkb, Nk, K, B); }
            }
            t[rr] = (now_ns() - t0) / it;
        }
        double ours3 = median(t, g_reps);

        for (int shared = 0; shared < 2; shared++) {
            for (int rr = 0; rr < g_reps; rr++) {
                double t0 = now_ns();
                for (int i = 0; i < it; i++) {
                    if (shared)
                        kai_run_lhs_quant_pack_qai8dxp_f32((size_t)B, (size_t)K, mr, kr, sr, 0,
                                                           Xbk, (size_t)K * sizeof(float), lp);
                    for (int s = 0; s < 3; s++) {
                        if (!shared)
                            kai_run_lhs_quant_pack_qai8dxp_f32((size_t)B, (size_t)K, mr, kr, sr, 0,
                                                               Xbk, (size_t)K * sizeof(float), lp);
                        kai_i8_job_t j = { cd, B, s == 0 ? Nq : Nk, K, lp,
                                           s == 0 ? rp_q : (s == 1 ? rp_k : rp_v),
                                           s == 0 ? Dq : (s == 1 ? Dk : Dv) };
                        if (g_threads > 1) qwen_parallel((size_t)g_threads, kai_i8_task, &j);
                        else               kai_i8_task(0, 1, &j);
                    }
                }
                t[rr] = (now_ns() - t0) / it;
            }
            double v = median(t, g_reps);
            row_t r; memset(&r, 0, sizeof r);
            r.comp = comp; r.op = "qkv3"; r.dtype = "int8";
            r.N = Nq + 2 * Nk; r.K = K; r.B = B; r.threads = g_threads;
            r.mr = (int)mr; r.nr = (int)nr; r.kr = (int)kr;
            r.m_step = (int)cd->uk.get_m_step(); r.n_step = (int)cd->uk.get_n_step();
            r.isa = cd->isa; r.tile = cd->tile;
            r.impl = cd->name; r.mode = shared ? "kai_1pack" : "kai_3packs";
            r.kernel = v; r.ns_compare = v; r.ns_dropin = v;
            if (shared == 0) {
                row_t o; memset(&o, 0, sizeof o);
                o.comp = comp; o.op = "qkv3"; o.dtype = "int8";
                o.N = Nq + 2 * Nk; o.K = K; o.B = B; o.threads = g_threads;
                o.impl = "OURS_3calls"; o.mode = "engine";
                o.isa = "dotprod/smmla"; o.tile = "-";
                o.kernel = ours3; o.ns_compare = ours3; o.ns_dropin = ours3;
                row_emit(&o, ours3);
            }
            row_emit(&r, ours3);
        }
        free(t); free(rp_q); free(rp_k); free(rp_v); free(lp);
        free(Dq); free(Dk); free(Dv); free(Yq); free(Yk2); free(Yv);
        free(Xbk); free(Xkb);
    }
    weights_free(&wq); weights_free(&wk); weights_free(&wv);
#endif
}

int main(int argc, char **argv) {
    const char *model = "1.7b", *csv_path = NULL;
    int do_int8 = 1, do_bf16 = 1, do_qkv = 1;
    int blist[16] = { 1, 2, 4, 8, 16 }, nb = 5;
    int plist[16] = { 1, 4, 5, 8, 16, 32, 64 }, npf = 7;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--model") && i + 1 < argc) model = argv[++i];
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc) g_threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--iters") && i + 1 < argc) g_iters = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--reps") && i + 1 < argc) g_reps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--csv") && i + 1 < argc) csv_path = argv[++i];
        else if (!strcmp(argv[i], "--int8-only")) { do_bf16 = 0; }
        else if (!strcmp(argv[i], "--bf16-only")) { do_int8 = 0; do_qkv = 0; }
        else if (!strcmp(argv[i], "--qkv-only"))  { do_int8 = 0; do_bf16 = 0; }
        else if (!strcmp(argv[i], "--no-qkv"))    { do_qkv = 0; }
        else if (!strcmp(argv[i], "--no-check"))  { g_check = 0; }
        else if (!strcmp(argv[i], "--b") && i + 1 < argc) {
            nb = 0;
            for (char *tk = strtok(argv[++i], ","); tk && nb < 16; tk = strtok(NULL, ",")) blist[nb++] = atoi(tk);
        } else if (!strcmp(argv[i], "--prefill-b") && i + 1 < argc) {
            npf = 0;
            for (char *tk = strtok(argv[++i], ","); tk && npf < 16; tk = strtok(NULL, ",")) plist[npf++] = atoi(tk);
        } else {
            fprintf(stderr,
                "usage: %s [--model 0.6b|1.7b] [--threads N] [--iters N] [--reps N]\n"
                "          [--b 1,2,4,8,16] [--prefill-b 1,4,5,8,16,32,64] [--csv FILE]\n"
                "          [--int8-only|--bf16-only|--qkv-only] [--no-qkv] [--no-check]\n", argv[0]);
            return 2;
        }
    }
    if (g_threads <= 0) { qwen_init_threads(); g_threads = qwen_get_threads(); }
    else qwen_set_threads(g_threads);

    g_csv = csv_path ? fopen(csv_path, "w") : stdout;
    if (!g_csv) { perror("csv"); return 1; }
    fprintf(stderr, "kernel census: model=%s threads=%d kleidi_int8=%d kleidi_bf16=%d\n",
            model, g_threads, KAI_HAVE_INT8, KAI_HAVE_BF16);
    csv_header();

    const int is06 = !strncmp(model, "0.6", 3);
    const shape_t *dec = is06 ? SHAPES_06B : SHAPES_17B;
    int ndec = is06 ? (int)(sizeof(SHAPES_06B) / sizeof(*SHAPES_06B))
                    : (int)(sizeof(SHAPES_17B) / sizeof(*SHAPES_17B));
    const shape_t *pre = is06 ? PREFILL_06B : PREFILL_17B;
    int npre = (int)(sizeof(PREFILL_17B) / sizeof(*PREFILL_17B));

    if (do_int8)
        for (int s = 0; s < ndec; s++) {
            weights_t w; weights_build(&w, dec[s].N, dec[s].K);
            for (int bi = 0; bi < nb; bi++) if (blist[bi] > 0) cell(&dec[s], &w, blist[bi], 1);
            weights_free(&w);
        }
    if (do_bf16)
        for (int s = 0; s < npre; s++) {
            weights_t w; weights_build(&w, pre[s].N, pre[s].K);
            for (int bi = 0; bi < npf; bi++) if (plist[bi] > 0) cell(&pre[s], &w, plist[bi], 0);
            weights_free(&w);
        }
    if (do_qkv) {
        qkv_pass("cp", 2048, 1024, 1024, blist, nb);
        if (!is06) qkv_pass("talker", 2048, 1024, 2048, blist, nb);
    }
    if (csv_path) fclose(g_csv);
    return 0;
}
