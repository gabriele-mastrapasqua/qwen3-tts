/* onednn_w8a8_bench.c — oneDNN against the SAME cells the KleidiAI census already measured. */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>
#include "dnnl.h"

#define CK(x) do { dnnl_status_t s_ = (x); if (s_ != dnnl_success) { \
    fprintf(stderr, "FAIL %s -> %d (%s:%d)\n", #x, (int)s_, __FILE__, __LINE__); exit(1); } } while (0)

static double now_ns(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e9 + ts.tv_nsec;
}
static int cmp_d(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}
static double median(double *v, int n) { qsort(v, n, sizeof *v, cmp_d); return v[n / 2]; }

static uint32_t rng_s = 0x2b3c4d5eu;
static float frand(void) {
    rng_s = rng_s * 1664525u + 1013904223u;
    return ((float)(rng_s >> 8) / 8388608.0f) - 1.0f;
}

typedef struct { const char *comp, *op; int N, K; } shape_t;
static const shape_t SHAPES[] = {
    { "cp",     "qkv",      4096, 1024 },
    { "cp",     "gate_up",  6144, 1024 },
    { "cp",     "down",     1024, 3072 },
    { "cp",     "lm_head",  2048, 1024 },
    { "talker", "gate_up", 12288, 2048 },
    { "talker", "down",     2048, 6144 },
};

static void quant_tensor(int8_t *q, float *scale, const float *X, int M, int K) {
    float amax = 0.0f;
    for (size_t i = 0; i < (size_t)M * K; i++) { float a = fabsf(X[i]); if (a > amax) amax = a; }
    float s = amax > 0 ? amax / 127.0f : 1.0f;
    *scale = s;
    float inv = 1.0f / s;
    for (size_t i = 0; i < (size_t)M * K; i++) {
        int v = (int)lrintf(X[i] * inv);
        q[i] = (int8_t)(v > 127 ? 127 : v < -127 ? -127 : v);
    }
}

static dnnl_memory_t mem_from(dnnl_engine_t eng, const_dnnl_memory_desc_t md, void *h) {
    dnnl_memory_t m;
    CK(dnnl_memory_create(&m, md, eng, h));
    return m;
}

static void cell(dnnl_engine_t eng, dnnl_stream_t str, const shape_t *S, int B, int reps) {
    const int N = S->N, K = S->K, M = B;

    int8_t *W  = (int8_t *)malloc((size_t)N * K);
    float  *Ws = (float *)malloc((size_t)N * sizeof(float));
    float  *X  = (float *)malloc((size_t)M * K * sizeof(float));
    int8_t *Xq = (int8_t *)malloc((size_t)M * K);
    float  *Xs = (float *)malloc(sizeof(float));
    float  *Y  = (float *)malloc((size_t)M * N * sizeof(float));
    if (!W || !Ws || !X || !Xq || !Xs || !Y) { fprintf(stderr, "oom\n"); exit(1); }

    for (size_t i = 0; i < (size_t)N * K; i++) W[i] = (int8_t)(frand() * 127.0f);
    for (int n = 0; n < N; n++) Ws[n] = 0.01f + 0.001f * (float)(n % 7);
    for (size_t i = 0; i < (size_t)M * K; i++) X[i] = frand();
    quant_tensor(Xq, Xs, X, M, K);

    int8_t *Wkn = (int8_t *)malloc((size_t)K * N);

    dnnl_dims_t sd = {M, K}, wd = {K, N}, dd = {M, N};
    dnnl_memory_desc_t smd, wmd_any, dmd, wmd_plain;
    CK(dnnl_memory_desc_create_with_tag(&smd, 2, sd, dnnl_s8, dnnl_ab));
    CK(dnnl_memory_desc_create_with_tag(&wmd_any, 2, wd, dnnl_s8, dnnl_format_tag_any));
    CK(dnnl_memory_desc_create_with_tag(&wmd_plain, 2, wd, dnnl_s8, dnnl_ab));
    CK(dnnl_memory_desc_create_with_tag(&dmd, 2, dd, dnnl_f32, dnnl_ab));

    dnnl_primitive_attr_t attr;
    CK(dnnl_primitive_attr_create(&attr));
    CK(dnnl_primitive_attr_set_scales_mask(attr, DNNL_ARG_SRC, 0));
    CK(dnnl_primitive_attr_set_scales_mask(attr, DNNL_ARG_WEIGHTS, 1 << 1));

    dnnl_primitive_desc_t pd;
    dnnl_status_t st = dnnl_matmul_primitive_desc_create(&pd, eng, smd, wmd_any, NULL, dmd, attr);
    if (st != dnnl_success) {
        printf("onednn,%s,%s,int8,%d,%d,%d,NO_PRIMITIVE,,,,,,,\n", S->comp, S->op, N, K, B);
        return;
    }
    const char *impl = NULL;
    dnnl_primitive_desc_query(pd, dnnl_query_impl_info_str, 0, &impl);
    const_dnnl_memory_desc_t wmd_req = dnnl_primitive_desc_query_md(pd, dnnl_query_weights_md, 0);
    size_t wpacked_bytes = dnnl_memory_desc_get_size(wmd_req);

    void *Wpacked = malloc(wpacked_bytes);
    double t0 = now_ns();
    for (int k = 0; k < K; k++)
        for (int n = 0; n < N; n++) Wkn[(size_t)k * N + n] = W[(size_t)n * K + k];
    dnnl_memory_t w_plain = mem_from(eng, wmd_plain, Wkn);
    dnnl_memory_t w_pack  = mem_from(eng, wmd_req, Wpacked);
    dnnl_primitive_desc_t rpd;
    dnnl_primitive_t rp;
    if (dnnl_reorder_primitive_desc_create(&rpd, wmd_plain, eng, wmd_req, eng, NULL) == dnnl_success) {
        CK(dnnl_primitive_create(&rp, rpd));
        dnnl_exec_arg_t ra[2] = { { DNNL_ARG_FROM, w_plain }, { DNNL_ARG_TO, w_pack } };
        CK(dnnl_primitive_execute(rp, str, 2, ra));
        CK(dnnl_stream_wait(str));
        dnnl_primitive_destroy(rp); dnnl_primitive_desc_destroy(rpd);
    } else {
        memcpy(Wpacked, Wkn, wpacked_bytes < (size_t)K * N ? wpacked_bytes : (size_t)K * N);
    }
    double rhs_pack_ns = now_ns() - t0;

    dnnl_primitive_t prim;
    CK(dnnl_primitive_create(&prim, pd));
    dnnl_memory_t m_src = mem_from(eng, smd, Xq);
    dnnl_memory_t m_dst = mem_from(eng, dmd, Y);

    dnnl_dims_t s1 = {1}, s2 = {N};
    dnnl_memory_desc_t sc_src_md, sc_wei_md;
    CK(dnnl_memory_desc_create_with_tag(&sc_src_md, 1, s1, dnnl_f32, dnnl_x));
    CK(dnnl_memory_desc_create_with_tag(&sc_wei_md, 1, s2, dnnl_f32, dnnl_x));
    dnnl_memory_t m_ssrc = mem_from(eng, sc_src_md, Xs);
    dnnl_memory_t m_swei = mem_from(eng, sc_wei_md, Ws);

    dnnl_exec_arg_t args[5] = {
        { DNNL_ARG_SRC, m_src }, { DNNL_ARG_WEIGHTS, w_pack }, { DNNL_ARG_DST, m_dst },
        { DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC, m_ssrc },
        { DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, m_swei },
    };

    CK(dnnl_primitive_execute(prim, str, 5, args));
    CK(dnnl_stream_wait(str));
    double num = 0, den = 0;
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            double acc = 0;
            for (int k = 0; k < K; k++)
                acc += (double)X[(size_t)m * K + k] * (double)W[(size_t)n * K + k] * (double)Ws[n];
            double got = Y[(size_t)m * N + n];
            num += (got - acc) * (got - acc);
            den += acc * acc;
        }
    double err = den > 0 ? sqrt(num / den) : -1.0;

    double macs = (double)M * N * K;
    int iters = (int)(2e8 / macs); if (iters < 1) iters = 1; if (iters > 2000) iters = 2000;
    for (int w = 0; w < 3; w++) {
        for (int i = 0; i < iters; i++) dnnl_primitive_execute(prim, str, 5, args);
        dnnl_stream_wait(str);
    }
    double runs[5];
    for (int r = 0; r < 5; r++) {
        double a = now_ns();
        for (int i = 0; i < iters; i++) dnnl_primitive_execute(prim, str, 5, args);
        CK(dnnl_stream_wait(str));
        runs[r] = (now_ns() - a) / iters;
    }
    double ns = median(runs, 5);

    printf("onednn,%s,%s,int8,%d,%d,%d,%s,%.0f,%zu,%.0f,%.3f,%.3e\n",
           S->comp, S->op, N, K, B, impl ? impl : "?",
           rhs_pack_ns, wpacked_bytes, ns, macs / ns, err);
    fflush(stdout);

    dnnl_primitive_destroy(prim); dnnl_primitive_desc_destroy(pd);
    dnnl_primitive_attr_destroy(attr);
    dnnl_memory_destroy(m_src); dnnl_memory_destroy(m_dst);
    dnnl_memory_destroy(m_ssrc); dnnl_memory_destroy(m_swei);
    dnnl_memory_destroy(w_plain); dnnl_memory_destroy(w_pack);
    dnnl_memory_desc_destroy(smd); dnnl_memory_desc_destroy(wmd_any);
    dnnl_memory_desc_destroy(wmd_plain); dnnl_memory_desc_destroy(dmd);
    dnnl_memory_desc_destroy(sc_src_md); dnnl_memory_desc_destroy(sc_wei_md);
    free(W); free(Ws); free(X); free(Xq); free(Xs); free(Y); free(Wkn); free(Wpacked);
}

static uint16_t f32_to_bf16_rne(float f) {
    union { float f; uint32_t u; } v = { f };
    uint32_t u = v.u;
    uint32_t lsb = (u >> 16) & 1u;
    u += 0x7fffu + lsb;
    return (uint16_t)(u >> 16);
}
static float bf16_to_f32(uint16_t h) {
    union { float f; uint32_t u; } v; v.u = (uint32_t)h << 16; return v.f;
}

static void cell_bf16(dnnl_engine_t eng, dnnl_stream_t str, const shape_t *S, int B) {
    const int N = S->N, K = S->K, M = B;
    float *Wf = (float *)malloc((size_t)N * K * sizeof(float));
    float *Xf = (float *)malloc((size_t)M * K * sizeof(float));
    uint16_t *Wb = (uint16_t *)malloc((size_t)K * N * sizeof(uint16_t));
    uint16_t *Xb = (uint16_t *)malloc((size_t)M * K * sizeof(uint16_t));
    float *Y = (float *)malloc((size_t)M * N * sizeof(float));
    if (!Wf || !Xf || !Wb || !Xb || !Y) { fprintf(stderr, "oom\n"); exit(1); }
    for (size_t i = 0; i < (size_t)N * K; i++) Wf[i] = frand() * 0.05f;
    for (size_t i = 0; i < (size_t)M * K; i++) Xf[i] = frand();
    for (size_t i = 0; i < (size_t)M * K; i++) Xb[i] = f32_to_bf16_rne(Xf[i]);

    dnnl_dims_t sd = {M, K}, wd = {K, N}, dd = {M, N};
    dnnl_memory_desc_t smd, wmd_any, wmd_plain, dmd;
    CK(dnnl_memory_desc_create_with_tag(&smd, 2, sd, dnnl_bf16, dnnl_ab));
    CK(dnnl_memory_desc_create_with_tag(&wmd_any, 2, wd, dnnl_bf16, dnnl_format_tag_any));
    CK(dnnl_memory_desc_create_with_tag(&wmd_plain, 2, wd, dnnl_bf16, dnnl_ab));
    CK(dnnl_memory_desc_create_with_tag(&dmd, 2, dd, dnnl_f32, dnnl_ab));

    dnnl_primitive_desc_t pd;
    if (dnnl_matmul_primitive_desc_create(&pd, eng, smd, wmd_any, NULL, dmd, NULL) != dnnl_success) {
        printf("onednn,%s,%s,bf16,%d,%d,%d,NO_PRIMITIVE,,,,,\n", S->comp, S->op, N, K, B);
        return;
    }
    const char *impl = NULL;
    dnnl_primitive_desc_query(pd, dnnl_query_impl_info_str, 0, &impl);
    const_dnnl_memory_desc_t wreq = dnnl_primitive_desc_query_md(pd, dnnl_query_weights_md, 0);
    size_t wbytes = dnnl_memory_desc_get_size(wreq);
    void *Wpacked = malloc(wbytes);

    double t0 = now_ns();
    for (int k = 0; k < K; k++)
        for (int n = 0; n < N; n++) Wb[(size_t)k * N + n] = f32_to_bf16_rne(Wf[(size_t)n * K + k]);
    dnnl_memory_t w_plain, w_pack;
    CK(dnnl_memory_create(&w_plain, wmd_plain, eng, Wb));
    CK(dnnl_memory_create(&w_pack, wreq, eng, Wpacked));
    dnnl_primitive_desc_t rpd; dnnl_primitive_t rp;
    if (dnnl_reorder_primitive_desc_create(&rpd, wmd_plain, eng, wreq, eng, NULL) == dnnl_success) {
        CK(dnnl_primitive_create(&rp, rpd));
        dnnl_exec_arg_t ra[2] = { { DNNL_ARG_FROM, w_plain }, { DNNL_ARG_TO, w_pack } };
        CK(dnnl_primitive_execute(rp, str, 2, ra)); CK(dnnl_stream_wait(str));
        dnnl_primitive_destroy(rp); dnnl_primitive_desc_destroy(rpd);
    }
    double pack_ns = now_ns() - t0;

    dnnl_primitive_t prim; CK(dnnl_primitive_create(&prim, pd));
    dnnl_memory_t m_src, m_dst;
    CK(dnnl_memory_create(&m_src, smd, eng, Xb));
    CK(dnnl_memory_create(&m_dst, dmd, eng, Y));
    dnnl_exec_arg_t args[3] = { { DNNL_ARG_SRC, m_src }, { DNNL_ARG_WEIGHTS, w_pack },
                                { DNNL_ARG_DST, m_dst } };
    CK(dnnl_primitive_execute(prim, str, 3, args)); CK(dnnl_stream_wait(str));

    double num = 0, den = 0;
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            double acc = 0;
            for (int k = 0; k < K; k++) acc += (double)Xf[(size_t)m * K + k] * (double)Wf[(size_t)n * K + k];
            double got = Y[(size_t)m * N + n];
            num += (got - acc) * (got - acc); den += acc * acc;
        }
    double err = den > 0 ? sqrt(num / den) : -1.0;

    double macs = (double)M * N * K;
    int iters = (int)(2e8 / macs); if (iters < 1) iters = 1; if (iters > 2000) iters = 2000;
    for (int w = 0; w < 3; w++) { for (int i = 0; i < iters; i++) dnnl_primitive_execute(prim, str, 3, args); dnnl_stream_wait(str); }
    double runs[5];
    for (int r = 0; r < 5; r++) {
        double a = now_ns();
        for (int i = 0; i < iters; i++) dnnl_primitive_execute(prim, str, 3, args);
        CK(dnnl_stream_wait(str));
        runs[r] = (now_ns() - a) / iters;
    }
    double ns = median(runs, 5);
    printf("onednn,%s,%s,bf16,%d,%d,%d,%s,%.0f,%zu,%.0f,%.3f,%.3e\n",
           S->comp, S->op, N, K, B, impl ? impl : "?", pack_ns, wbytes, ns, macs / ns, err);
    fflush(stdout);
    (void)bf16_to_f32;
    dnnl_primitive_destroy(prim); dnnl_primitive_desc_destroy(pd);
    dnnl_memory_destroy(m_src); dnnl_memory_destroy(m_dst);
    dnnl_memory_destroy(w_plain); dnnl_memory_destroy(w_pack);
    dnnl_memory_desc_destroy(smd); dnnl_memory_desc_destroy(wmd_any);
    dnnl_memory_desc_destroy(wmd_plain); dnnl_memory_desc_destroy(dmd);
    free(Wf); free(Xf); free(Wb); free(Xb); free(Y); free(Wpacked);
}

static const shape_t PREFILL[] = {
    { "prefill", "wq",       2048, 2048 },
    { "prefill", "wk",       1024, 2048 },
    { "prefill", "wo",       2048, 2048 },
    { "prefill", "gate_up", 12288, 2048 },
    { "prefill", "down",     2048, 6144 },
};

int main(int argc, char **argv) {
    int reps = argc > 1 ? atoi(argv[1]) : 5;
    const dnnl_version_t *v = dnnl_version();
    fprintf(stderr, "# oneDNN %d.%d.%d cpu_runtime=%d  OMP_NUM_THREADS=%s\n",
            v->major, v->minor, v->patch, (int)v->cpu_runtime,
            getenv("OMP_NUM_THREADS") ? getenv("OMP_NUM_THREADS") : "(unset)");
    dnnl_engine_t eng; dnnl_stream_t str;
    CK(dnnl_engine_create(&eng, dnnl_cpu, 0));
    CK(dnnl_stream_create(&str, eng, dnnl_stream_default_flags));
    printf("backend,comp,op,dtype,N,K,B,impl,rhs_pack_ns,rhs_pack_bytes,kernel_ns,gmac_s,err_vs_f32\n");
    int BS[] = {1, 2, 4, 8, 16};
    for (size_t i = 0; i < sizeof SHAPES / sizeof *SHAPES; i++)
        for (size_t b = 0; b < sizeof BS / sizeof *BS; b++)
            cell(eng, str, &SHAPES[i], BS[b], reps);
    int PB[] = {5, 8, 16};
    for (size_t i = 0; i < sizeof PREFILL / sizeof *PREFILL; i++)
        for (size_t b = 0; b < sizeof PB / sizeof *PB; b++)
            cell_bf16(eng, str, &PREFILL[i], PB[b]);
    dnnl_stream_destroy(str); dnnl_engine_destroy(eng);
    return 0;
}
