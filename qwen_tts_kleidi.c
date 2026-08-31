/* qwen_tts_kleidi.c — Arm KleidiAI micro-kernels for GGUF Q4_0 weights. */
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdatomic.h>
#include <pthread.h>
#include <math.h>
#include "qwen_tts_kleidi.h"
#include "qwen_tts_kernels.h"
#include "qwen_tts_thread.h"
#include <float.h>
#ifdef __linux__
#include <sys/auxv.h>
#if defined(__aarch64__) || defined(__arm__)
#include <asm/hwcap.h>
#endif
#endif
#ifdef __APPLE__
#include <sys/sysctl.h>
#endif

#if defined(__aarch64__) && defined(__ARM_FEATURE_MATMUL_INT8) && defined(__ARM_FEATURE_DOTPROD)
#define QWEN_KLEIDI_BUILD 1
#include "kai/kai_common.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0.h"
#include "kai/ukernels/matmul/pack/kai_lhs_quant_pack_qsi8d32p_f32.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm.h"
#include "kai/ukernels/matmul/pack/kai_lhs_quant_pack_qai8dxp_f32.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qsi8cxp_qsi8cx_neon.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm.h"
#if defined(__ARM_FEATURE_BF16)
#define QWEN_KLEIDI_BF16_BUILD 1
#include "kai/ukernels/matmul/pack/kai_lhs_quant_pack_bf16p1x4_f32_neon.h"
#include "kai/ukernels/matmul/pack/kai_lhs_quant_pack_bf16p8x4_f32_neon.h"
#include "kai/ukernels/matmul/pack/kai_rhs_quant_pack_kxn_bf16p12x4biasf32_f32_neon.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_bf16p_bf16p/kai_matmul_clamp_f32_bf16p1x4_bf16p12x4b_1x36_neon_dot.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_bf16p_bf16p/kai_matmul_clamp_f32_bf16p8x4_bf16p12x4b_8x12_neon_mmla.h"
#else
#define QWEN_KLEIDI_BF16_BUILD 0
#endif
#else
#define QWEN_KLEIDI_BUILD 0
#define QWEN_KLEIDI_BF16_BUILD 0
#endif

#define QWEN_KAI_SCRATCH(name, type)                                                  \
    static __thread type *g_kais_##name = NULL;                                       \
    static __thread size_t g_kais_cap_##name = 0;                                     \
    static type *kai_scratch_##name(size_t nelem) {                                   \
        size_t need = nelem * sizeof(type);                                           \
        if (need > g_kais_cap_##name) {                                               \
            void *np = NULL;                                                          \
            if (posix_memalign(&np, 64, need) != 0) return NULL;                       \
            free(g_kais_##name);                                                      \
            g_kais_##name = (type *)np; g_kais_cap_##name = need;                     \
        }                                                                             \
        return g_kais_##name;                                                         \
    }

#define KAI_BL 32

static int kleidi_cpu_ok(void) {
#if !QWEN_KLEIDI_BUILD
    return 0;
#elif defined(__linux__)
    unsigned long h1 = getauxval(AT_HWCAP), h2 = getauxval(AT_HWCAP2);
    int dot = 0, i8mm = 0;
#ifdef HWCAP_ASIMDDP
    dot = (h1 & HWCAP_ASIMDDP) != 0;
#endif
#ifdef HWCAP2_I8MM
    i8mm = (h2 & HWCAP2_I8MM) != 0;
#endif
    (void)h1; (void)h2;
    return dot && i8mm;
#elif defined(__APPLE__)
    int v = 0; size_t sz = sizeof v;
    int dot = (sysctlbyname("hw.optional.arm.FEAT_DotProd", &v, &sz, NULL, 0) == 0 && v);
    v = 0; sz = sizeof v;
    int i8mm = (sysctlbyname("hw.optional.arm.FEAT_I8MM", &v, &sz, NULL, 0) == 0 && v);
    return dot && i8mm;
#else
    return 0;
#endif
}

int qwen_kleidi_supported(void) { return QWEN_KLEIDI_BUILD && kleidi_cpu_ok(); }

int qwen_kleidi_enabled(void) {
    static atomic_int cached = -1;
    int v = atomic_load_explicit(&cached, memory_order_relaxed);
    if (v < 0) {
        const char *e = getenv("QWEN_NO_KLEIDI");
        v = qwen_kleidi_supported() && !(e && e[0] == '1');
        atomic_store_explicit(&cached, v, memory_order_relaxed);
    }
    return v;
}

double qwen_kbf_pack_ms = 0.0, qwen_kbf_gemm_ms = 0.0;

double qwen_kbf_busy_mean_ms = 0.0, qwen_kbf_busy_max_ms = 0.0;
#if QWEN_KLEIDI_BUILD

enum { KAI_KIND_Q4 = 0, KAI_KIND_I8, KAI_KIND_BF16, KAI_KIND_N };
typedef struct {
    const void *key;
    void       *rhs;
    int         rows, cols;
    int         kind;
    int         comp, fam;
    size_t      bytes;
} kai_entry_t;

static kai_entry_t  *g_kai;
static int           g_kai_n, g_kai_cap;
static size_t        g_kai_bytes;
static pthread_mutex_t g_kai_mx = PTHREAD_MUTEX_INITIALIZER;

static size_t g_kai_bytes_kind[KAI_KIND_N];
static int    g_kai_n_kind[KAI_KIND_N];

static const kai_entry_t *kai_lookup_kind(const void *key, int kind) {
    int n = atomic_load_explicit((_Atomic int *)&g_kai_n, memory_order_acquire);
    for (int i = 0; i < n; i++)
        if (g_kai[i].key == key && g_kai[i].kind == kind) return &g_kai[i];
    return NULL;
}
static const kai_entry_t *kai_lookup(const void *key) { return kai_lookup_kind(key, KAI_KIND_Q4); }

static int g_kai_ops[QWEN_KAI_COMP_N][QWEN_KAI_FAM_N];
static int g_kai_ops_prefill = 1;
static atomic_int g_kai_ops_parsed = 0;
static const char *const KAI_FAM_NAME[QWEN_KAI_FAM_N] = { "qkv", "o", "ffn", "heads", "other" };
static const char *const KAI_COMP_NAME[QWEN_KAI_COMP_N] = { "talker", "cp" };

static void kai_ops_parse(void) {
    if (atomic_exchange_explicit(&g_kai_ops_parsed, 1, memory_order_relaxed)) return;
    const char *e = getenv("QWEN_KAI_OPS");
    if (!e || !e[0]) {
        for (int c = 0; c < QWEN_KAI_COMP_N; c++)
            for (int f = 0; f < QWEN_KAI_FAM_N; f++) g_kai_ops[c][f] = 1;
        g_kai_ops_prefill = 1;
        return;
    }
    g_kai_ops_prefill = 0;
    char buf[512]; snprintf(buf, sizeof buf, "%s", e);
    for (char *tok = strtok(buf, ","); tok; tok = strtok(NULL, ",")) {
        while (*tok == ' ') tok++;
        if (!strcmp(tok, "none")) continue;
        if (!strcmp(tok, "all")) {
            for (int c = 0; c < QWEN_KAI_COMP_N; c++)
                for (int f = 0; f < QWEN_KAI_FAM_N; f++) g_kai_ops[c][f] = 1;
            g_kai_ops_prefill = 1; continue;
        }
        if (!strcmp(tok, "prefill")) { g_kai_ops_prefill = 1; continue; }
        char *dot = strchr(tok, '.');
        const char *cs = NULL, *fs = tok;
        if (dot) { *dot = 0; cs = tok; fs = dot + 1; }
        for (int c = 0; c < QWEN_KAI_COMP_N; c++) {
            if (cs && strcmp(cs, KAI_COMP_NAME[c])) continue;
            if (!cs && !strcmp(fs, KAI_COMP_NAME[c])) {
                for (int f = 0; f < QWEN_KAI_FAM_N; f++) g_kai_ops[c][f] = 1;
                continue;
            }
            for (int f = 0; f < QWEN_KAI_FAM_N; f++)
                if (!strcmp(fs, KAI_FAM_NAME[f])) g_kai_ops[c][f] = 1;
        }
    }
}
static inline int kai_op_on(int comp, int fam) {
    kai_ops_parse();
    if (comp < 0 || comp >= QWEN_KAI_COMP_N) comp = QWEN_KAI_COMP_TALKER;
    if (fam < 0 || fam >= QWEN_KAI_FAM_N) fam = QWEN_KAI_FAM_OTHER;
    return g_kai_ops[comp][fam];
}
int qwen_kleidi_prefill_enabled(void) {
    if (!qwen_kleidi_bf16_enabled()) return 0;
    kai_ops_parse();
    return g_kai_ops_prefill;
}

static int kai_insert_fam(const void *key, void *rhs, int rows, int cols, int kind,
                          size_t sz, int comp, int fam);
static int kai_insert(const void *key, void *rhs, int rows, int cols, int kind, size_t sz) {
    return kai_insert_fam(key, rhs, rows, cols, kind, sz,
                          QWEN_KAI_COMP_TALKER, QWEN_KAI_FAM_OTHER);
}
static int kai_insert_fam(const void *key, void *rhs, int rows, int cols, int kind,
                          size_t sz, int comp, int fam) {
    pthread_mutex_lock(&g_kai_mx);
    for (int i = 0; i < g_kai_n; i++)
        if (g_kai[i].key == key && g_kai[i].kind == kind) {
            pthread_mutex_unlock(&g_kai_mx); free(rhs); return 1;
        }
    if (g_kai_n == g_kai_cap) {
        int cap = g_kai_cap ? g_kai_cap * 2 : 256;
        kai_entry_t *p = (kai_entry_t *)realloc(g_kai, (size_t)cap * sizeof *p);
        if (!p) { pthread_mutex_unlock(&g_kai_mx); free(rhs); return 0; }
        g_kai = p; g_kai_cap = cap;
    }
    g_kai[g_kai_n] = (kai_entry_t){ key, rhs, rows, cols, kind, comp, fam, sz };
    atomic_store_explicit((_Atomic int *)&g_kai_n, g_kai_n + 1, memory_order_release);
    g_kai_bytes += sz;
    g_kai_bytes_kind[kind] += sz;
    g_kai_n_kind[kind]++;
    pthread_mutex_unlock(&g_kai_mx);
    return 1;
}

int qwen_kleidi_register_q4(const void *key, const uint8_t *ggml_blocks, int rows, int cols) {
    if (!qwen_kleidi_enabled() || !key || !ggml_blocks) return 0;
    if (cols % KAI_BL != 0) return 0;

    const size_t nr = kai_get_nr_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm();
    const size_t kr = kai_get_kr_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm();
    const size_t sr = kai_get_sr_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm();

    size_t sz = kai_get_rhs_packed_size_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0(
                    (size_t)rows, (size_t)cols, nr, kr, KAI_BL);
    void *rhs = aligned_malloc(sz);
    if (!rhs) return 0;

    struct kai_rhs_pack_qs4cxs1s0_param params = { .lhs_zero_point = 1, .rhs_zero_point = 8 };
    kai_run_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0(
        1, (size_t)rows, (size_t)cols, nr, kr, sr, KAI_BL,
        ggml_blocks, NULL, rhs, 0, &params);

    return kai_insert(key, rhs, rows, cols, KAI_KIND_Q4, sz);
}

static __thread int g_kai_bypass;

QWEN_KAI_SCRATCH(lhs, uint8_t)
QWEN_KAI_SCRATCH(xt,  float)
QWEN_KAI_SCRATCH(yt,  float)

typedef struct {
    const kai_entry_t *e;
    const void *lhs_packed;
    float *dst;
    size_t m, n, k, dst_stride_row;
    int gemm;
} kai_job_t;

static void kai_task(size_t tid, size_t nt, void *ctx) {
    kai_job_t *j = (kai_job_t *)ctx;
    const size_t n_step = j->gemm
        ? kai_get_n_step_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm()
        : kai_get_n_step_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod();
    size_t tiles = (j->n + n_step - 1) / n_step;
    size_t t0 = tiles * tid / nt, t1 = tiles * (tid + 1) / nt;
    size_t n0 = t0 * n_step;
    size_t n1 = (t1 * n_step < j->n) ? t1 * n_step : j->n;
    if (n0 >= n1) return;

    if (j->gemm) {
        const size_t roff = kai_get_rhs_packed_offset_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm(n0, j->k, KAI_BL);
        const size_t doff = kai_get_dst_offset_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm(0, n0, j->dst_stride_row);
        kai_run_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm(
            j->m, n1 - n0, j->k, KAI_BL, j->lhs_packed,
            (const uint8_t *)j->e->rhs + roff,
            (float *)((uint8_t *)j->dst + doff),
            j->dst_stride_row, sizeof(float), -FLT_MAX, FLT_MAX);
    } else {
        const size_t roff = kai_get_rhs_packed_offset_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod(n0, j->k, KAI_BL);
        const size_t doff = kai_get_dst_offset_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod(0, n0, j->dst_stride_row);
        kai_run_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod(
            j->m, n1 - n0, j->k, KAI_BL, j->lhs_packed,
            (const uint8_t *)j->e->rhs + roff,
            (float *)((uint8_t *)j->dst + doff),
            j->dst_stride_row, sizeof(float), -FLT_MAX, FLT_MAX);
    }
}

int qwen_kleidi_matmul_q4(float *Y, const void *key, const float *X, int rows, int cols, int B) {
    if (!qwen_kleidi_enabled() || B < 1 || g_kai_bypass) return 0;
    const kai_entry_t *e = kai_lookup(key);
    if (!e || e->rows != rows || e->cols != cols) return 0;

    const int gemm = (B > 1);
    const size_t mr = gemm ? kai_get_mr_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm()
                           : kai_get_mr_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod();
    const size_t kr = gemm ? kai_get_kr_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm()
                           : kai_get_kr_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod();
    const size_t sr = gemm ? kai_get_sr_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm()
                           : kai_get_sr_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod();

    const float *lhs_src = X;
    size_t lhs_stride = (size_t)cols * sizeof(float);
    if (B > 1) {
        float *xt = kai_scratch_xt((size_t)B * cols);
        if (!xt) return 0;
        for (int b = 0; b < B; b++)
            for (int c = 0; c < cols; c++) xt[(size_t)b * cols + c] = X[(size_t)c * B + b];
        lhs_src = xt;
    }

    size_t lhs_sz = kai_get_lhs_packed_size_lhs_quant_pack_qsi8d32p_f32((size_t)B, (size_t)cols, KAI_BL, mr, kr, sr);
    uint8_t *lhs_packed = kai_scratch_lhs(lhs_sz);
    if (!lhs_packed) return 0;
    kai_run_lhs_quant_pack_qsi8d32p_f32((size_t)B, (size_t)cols, KAI_BL, mr, kr, sr, 0,
                                        lhs_src, lhs_stride, lhs_packed);

    float *dst = Y;
    if (B > 1) {
        dst = kai_scratch_yt((size_t)B * rows);
        if (!dst) return 0;
    }

    kai_job_t job = { e, lhs_packed, dst, (size_t)B, (size_t)rows, (size_t)cols,
                      (size_t)rows * sizeof(float), gemm };
    size_t nt = (size_t)qwen_get_threads();
    if (nt < 1) nt = 1;
    if ((size_t)rows < nt * 16) nt = 1;
    if (nt == 1) kai_task(0, 1, &job);
    else         qwen_parallel(nt, kai_task, &job);

    if (B > 1) {
        for (int b = 0; b < B; b++)
            for (int r = 0; r < rows; r++) Y[(size_t)r * B + b] = dst[(size_t)b * rows + r];
    }
    return 1;
}

int qwen_kleidi_selfcheck(const void *key, int rows, int cols, float *max_abs, float *rel) {
    if (!qwen_kleidi_enabled() || !kai_lookup(key)) return 0;
    float *x  = (float *)malloc((size_t)cols * sizeof(float));
    float *ya = (float *)malloc((size_t)rows * sizeof(float));
    float *yb = (float *)malloc((size_t)rows * sizeof(float));
    if (!x || !ya || !yb) { free(x); free(ya); free(yb); return 0; }

    uint32_t st = 12345u;
    for (int i = 0; i < cols; i++) {
        st = st * 1664525u + 1013904223u;
        x[i] = ((float)((st >> 8) & 0xFFFF) / 32768.0f) - 1.0f;
    }

    if (!qwen_kleidi_matmul_q4(ya, key, x, rows, cols, 1)) { free(x); free(ya); free(yb); return 0; }
    g_kai_bypass = 1;
    qwen_matvec_q4_0(yb, (const q4_0_block_t *)key, x, rows, cols);
    g_kai_bypass = 0;

    double worst = 0.0, se = 0.0, sr = 0.0;
    for (int i = 0; i < rows; i++) {
        double d = (double)ya[i] - (double)yb[i];
        if (d < 0) d = -d;
        if (d > worst) worst = d;
        se += d * d;
        sr += (double)yb[i] * (double)yb[i];
    }
    if (max_abs) *max_abs = (float)worst;
    if (rel)     *rel = (float)((sr > 0.0) ? sqrt(se / sr) : 0.0);
    free(x); free(ya); free(yb);
    return 1;
}

void qwen_kleidi_stats(int *n_packed, size_t *bytes) {
    if (n_packed) *n_packed = g_kai_n;
    if (bytes)    *bytes    = g_kai_bytes;
}

#define KI8_GEMV(f) kai_##f##_matmul_clamp_f32_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod
#define KI8_GEMM(f) kai_##f##_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm

int qwen_kleidi_i8_enabled(void) {
    static atomic_int cached = -1;
    int v = atomic_load_explicit(&cached, memory_order_relaxed);
    if (v < 0) {
        const char *e = getenv("QWEN_NO_KAI_I8");
        v = qwen_kleidi_enabled() && !(e && e[0] && e[0] != '0');
        atomic_store_explicit(&cached, v, memory_order_relaxed);
    }
    return v;
}

int qwen_kleidi_register_i8(const void *key, const int8_t *W, const float *scale,
                            int rows, int cols) {
    return qwen_kleidi_register_i8_fam(key, W, scale, rows, cols,
                                       QWEN_KAI_COMP_TALKER, QWEN_KAI_FAM_OTHER);
}
int qwen_kleidi_register_i8_fam(const void *key, const int8_t *W, const float *scale,
                                int rows, int cols, int comp, int fam) {
    if (!qwen_kleidi_i8_enabled() || !key || !W || !scale) return 0;
    if (rows <= 0 || cols <= 0) return 0;
    if (kai_lookup_kind(key, KAI_KIND_I8)) return 1;

    const size_t nr = KI8_GEMM(get_nr)(), kr = KI8_GEMM(get_kr)(), sr = KI8_GEMM(get_sr)();
    size_t sz = kai_get_rhs_packed_size_rhs_pack_nxk_qsi8cxp_qsi8cx_neon(
                    (size_t)rows, (size_t)cols, nr, kr, sr);
    void *rhs = aligned_malloc(sz);
    if (!rhs) return 0;

    struct kai_rhs_pack_qsi8cx_params params = { .lhs_zero_point = 1, .scale_multiplier = 1.0f };
    kai_run_rhs_pack_nxk_qsi8cxp_qsi8cx_neon(1, (size_t)rows, (size_t)cols, nr, kr, sr,
                                             W, NULL, scale, rhs, 0, &params);
    return kai_insert_fam(key, rhs, rows, cols, KAI_KIND_I8, sz, comp, fam);
}

typedef struct {
    const kai_entry_t *e;
    const void *lhs_packed;
    float *dst;
    size_t m, n, k, dst_stride_row;
    int gemm;
} kai_i8_job_t;

static void kai_i8_task(size_t tid, size_t nt, void *ctx) {
    kai_i8_job_t *j = (kai_i8_job_t *)ctx;
    const size_t n_step = j->gemm ? KI8_GEMM(get_n_step)() : KI8_GEMV(get_n_step)();
    size_t tiles = (j->n + n_step - 1) / n_step;
    size_t t0 = tiles * tid / nt, t1 = tiles * (tid + 1) / nt;
    size_t n0 = t0 * n_step;
    size_t n1 = (t1 * n_step < j->n) ? t1 * n_step : j->n;
    if (n0 >= n1) return;
    if (j->gemm) {
        const size_t roff = KI8_GEMM(get_rhs_packed_offset)(n0, j->k);
        const size_t doff = KI8_GEMM(get_dst_offset)(0, n0, j->dst_stride_row);
        KI8_GEMM(run)(j->m, n1 - n0, j->k, j->lhs_packed,
                      (const uint8_t *)j->e->rhs + roff,
                      (float *)((uint8_t *)j->dst + doff),
                      j->dst_stride_row, sizeof(float), -FLT_MAX, FLT_MAX);
    } else {
        const size_t roff = KI8_GEMV(get_rhs_packed_offset)(n0, j->k);
        const size_t doff = KI8_GEMV(get_dst_offset)(0, n0, j->dst_stride_row);
        KI8_GEMV(run)(j->m, n1 - n0, j->k, j->lhs_packed,
                      (const uint8_t *)j->e->rhs + roff,
                      (float *)((uint8_t *)j->dst + doff),
                      j->dst_stride_row, sizeof(float), -FLT_MAX, FLT_MAX);
    }
}

QWEN_KAI_SCRATCH(i8lhs, uint8_t)
QWEN_KAI_SCRATCH(i8xt,  float)
QWEN_KAI_SCRATCH(i8yt,  float)

static const uint8_t *kai_i8_pack_lhs(const float *lhs, size_t lhs_stride,
                                      int cols, int B, int gemm) {
    const size_t mr = gemm ? KI8_GEMM(get_mr)() : KI8_GEMV(get_mr)();
    const size_t kr = gemm ? KI8_GEMM(get_kr)() : KI8_GEMV(get_kr)();
    const size_t sr = gemm ? KI8_GEMM(get_sr)() : KI8_GEMV(get_sr)();
    size_t sz = kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_f32(
                    (size_t)B, (size_t)cols, mr, kr, sr);
    uint8_t *p = kai_scratch_i8lhs(sz);
    if (!p) return NULL;
    kai_run_lhs_quant_pack_qai8dxp_f32((size_t)B, (size_t)cols, mr, kr, sr, 0,
                                       lhs, lhs_stride, p);
    return p;
}

static const uint8_t *kai_i8_pack_lhs_sym(const float *lhs, size_t lhs_stride,
                                          int cols, int B, int gemm) {
    const size_t mr = gemm ? KI8_GEMM(get_mr)() : KI8_GEMV(get_mr)();
    const size_t kr = gemm ? KI8_GEMM(get_kr)() : KI8_GEMV(get_kr)();
    const size_t sr = gemm ? KI8_GEMM(get_sr)() : KI8_GEMV(get_sr)();
    const size_t blk = kr / sr;
    const size_t k_int = (((size_t)cols + 31) / 32) * 32;
    const size_t stride = mr * (k_int + sizeof(int32_t) + sizeof(float));
    const size_t nblocks = ((size_t)B + mr - 1) / mr;

    uint8_t *base = kai_scratch_i8lhs(nblocks * stride);
    if (!base) return NULL;
    memset(base, 0, nblocks * stride);

    for (int b = 0; b < B; b++) {
        const float *x = (const float *)((const uint8_t *)lhs + (size_t)b * lhs_stride);
        float amax = 0.0f;
        for (int k = 0; k < cols; k++) { float a = fabsf(x[k]); if (a > amax) amax = a; }
        const float s = amax > 0.0f ? amax / 127.0f : 0.0f;
        const float inv = amax > 0.0f ? 127.0f / amax : 0.0f;

        uint8_t *blkbase = base + (size_t)(b / mr) * stride;
        const size_t row = (size_t)b % mr;
        int8_t *q = (int8_t *)(blkbase + row * blk);
        for (int k = 0; k < cols; k++) {
            int v = (int)lrintf(x[k] * inv);
            if (v > 127) v = 127; else if (v < -128) v = -128;
            q[(size_t)(k / blk) * mr * blk + (size_t)(k % blk)] = (int8_t)v;
        }
        *(int32_t *)(blkbase + mr * k_int + row * sizeof(int32_t)) = 0;
        *(float *)(blkbase + mr * k_int + mr * sizeof(int32_t) + row * sizeof(float)) = s;
    }
    return base;
}

static int kai_lhs_sym_mode(void) {
    static atomic_int cached = -1;
    int v = atomic_load_explicit(&cached, memory_order_relaxed);
    if (v < 0) {
        const char *e = getenv("QWEN_KAI_LHS");
        v = (e && (e[0] == 's' || e[0] == 'S'));
        atomic_store_explicit(&cached, v, memory_order_relaxed);
    }
    return v;
}

static int kai_i8_run_packed(const kai_entry_t *e, float *dst, const void *lhs_packed,
                             size_t dst_stride, int rows, int cols, int B, int gemm) {
    kai_i8_job_t job = { e, lhs_packed, dst, (size_t)B, (size_t)rows, (size_t)cols,
                         dst_stride, gemm };
    size_t nt = (size_t)qwen_get_threads();
    if (nt < 1) nt = 1;
    if ((size_t)rows < nt * 16) nt = 1;
    if (nt == 1) kai_i8_task(0, 1, &job);
    else         qwen_parallel(nt, kai_i8_task, &job);
    return 1;
}

static int kai_i8_run(const kai_entry_t *e, float *dst, const float *lhs,
                      size_t lhs_stride, size_t dst_stride, int rows, int cols, int B) {
    const int gemm = (B > 1);
    const uint8_t *lp = kai_lhs_sym_mode()
        ? kai_i8_pack_lhs_sym(lhs, lhs_stride, cols, B, gemm)
        : kai_i8_pack_lhs(lhs, lhs_stride, cols, B, gemm);
    if (!lp) return 0;
    return kai_i8_run_packed(e, dst, lp, dst_stride, rows, cols, B, gemm);
}

typedef struct {
    const kai_entry_t *e[3];
    float  *dst[3];
    size_t  dst_stride[3];
    size_t  n[3];
    size_t  tiles[3];
    size_t  cum[4];
    const void *lhs_packed;
    size_t  m, k;
    int     gemm;
} kai_i8_qkv_job_t;

static void kai_i8_qkv_task(size_t tid, size_t nt, void *ctx) {
    kai_i8_qkv_job_t *j = (kai_i8_qkv_job_t *)ctx;
    const size_t n_step = j->gemm ? KI8_GEMM(get_n_step)() : KI8_GEMV(get_n_step)();
    const size_t T = j->cum[3];
    size_t t0 = T * tid / nt, t1 = T * (tid + 1) / nt;
    if (t0 >= t1) return;

    for (int i = 0; i < 3; i++) {
        size_t a = t0 > j->cum[i] ? t0 : j->cum[i];
        size_t b = t1 < j->cum[i + 1] ? t1 : j->cum[i + 1];
        if (a >= b) continue;
        size_t lt0 = a - j->cum[i], lt1 = b - j->cum[i];
        size_t n0 = lt0 * n_step;
        size_t n1 = (lt1 * n_step < j->n[i]) ? lt1 * n_step : j->n[i];
        if (n0 >= n1) continue;
        if (j->gemm) {
            const size_t roff = KI8_GEMM(get_rhs_packed_offset)(n0, j->k);
            const size_t doff = KI8_GEMM(get_dst_offset)(0, n0, j->dst_stride[i]);
            KI8_GEMM(run)(j->m, n1 - n0, j->k, j->lhs_packed,
                          (const uint8_t *)j->e[i]->rhs + roff,
                          (float *)((uint8_t *)j->dst[i] + doff),
                          j->dst_stride[i], sizeof(float), -FLT_MAX, FLT_MAX);
        } else {
            const size_t roff = KI8_GEMV(get_rhs_packed_offset)(n0, j->k);
            const size_t doff = KI8_GEMV(get_dst_offset)(0, n0, j->dst_stride[i]);
            KI8_GEMV(run)(j->m, n1 - n0, j->k, j->lhs_packed,
                          (const uint8_t *)j->e[i]->rhs + roff,
                          (float *)((uint8_t *)j->dst[i] + doff),
                          j->dst_stride[i], sizeof(float), -FLT_MAX, FLT_MAX);
        }
    }
}

static int kai_qkv_fused(void) {
    static atomic_int cached = -1;
    int v = atomic_load_explicit(&cached, memory_order_relaxed);
    if (v < 0) {
        const char *e = getenv("QWEN_KAI_QKV_FUSED");
        v = !(e && e[0] == '0');
        atomic_store_explicit(&cached, v, memory_order_relaxed);
    }
    return v;
}

int qwen_kleidi_matmul_i8_qkv_native(float *dq, float *dk, float *dv,
                                     const void *keyq, const void *keyk, const void *keyv,
                                     const float *lhs, size_t lhs_stride,
                                     int in_dim, int q_dim, int kv_dim, int B) {
    if (!qwen_kleidi_i8_enabled() || g_kai_bypass || B < 1) return 0;
    if (!kai_qkv_fused()) return 0;
    const kai_entry_t *eq = kai_lookup_kind(keyq, KAI_KIND_I8);
    const kai_entry_t *ek = kai_lookup_kind(keyk, KAI_KIND_I8);
    const kai_entry_t *ev = kai_lookup_kind(keyv, KAI_KIND_I8);
    if (!eq || !ek || !ev) return 0;
    if (eq->rows != q_dim || ek->rows != kv_dim || ev->rows != kv_dim) return 0;
    if (eq->cols != in_dim || ek->cols != in_dim || ev->cols != in_dim) return 0;
    if (!kai_op_on(eq->comp, eq->fam)) return 0;

    const int gemm = (B > 1);
    const uint8_t *lp = kai_lhs_sym_mode()
        ? kai_i8_pack_lhs_sym(lhs, lhs_stride, in_dim, B, gemm)
        : kai_i8_pack_lhs(lhs, lhs_stride, in_dim, B, gemm);
    if (!lp) return 0;

    kai_i8_qkv_job_t job;
    job.e[0] = eq; job.e[1] = ek; job.e[2] = ev;
    job.dst[0] = dq; job.dst[1] = dk; job.dst[2] = dv;
    job.dst_stride[0] = (size_t)q_dim  * sizeof(float);
    job.dst_stride[1] = (size_t)kv_dim * sizeof(float);
    job.dst_stride[2] = (size_t)kv_dim * sizeof(float);
    job.n[0] = (size_t)q_dim; job.n[1] = (size_t)kv_dim; job.n[2] = (size_t)kv_dim;
    job.lhs_packed = lp;
    job.m = (size_t)B; job.k = (size_t)in_dim; job.gemm = gemm;

    const size_t n_step = gemm ? KI8_GEMM(get_n_step)() : KI8_GEMV(get_n_step)();
    job.cum[0] = 0;
    for (int i = 0; i < 3; i++) {
        job.tiles[i] = (job.n[i] + n_step - 1) / n_step;
        job.cum[i + 1] = job.cum[i] + job.tiles[i];
    }
    size_t nt = (size_t)qwen_get_threads();
    if (nt < 1) nt = 1;
    const size_t rows_total = job.n[0] + job.n[1] + job.n[2];
    if (rows_total < nt * 16) nt = 1;
    if (nt == 1) kai_i8_qkv_task(0, 1, &job);
    else         qwen_parallel(nt, kai_i8_qkv_task, &job);
    return 1;
}

int qwen_kleidi_matmul_i8_qkv(float *q, float *k, float *v,
                              const void *keyq, const void *keyk, const void *keyv,
                              const float *x, int in_dim, int q_dim, int kv_dim) {
    if (!qwen_kleidi_i8_enabled() || g_kai_bypass) return 0;
    const kai_entry_t *eq = kai_lookup_kind(keyq, KAI_KIND_I8);
    const kai_entry_t *ek = kai_lookup_kind(keyk, KAI_KIND_I8);
    const kai_entry_t *ev = kai_lookup_kind(keyv, KAI_KIND_I8);
    if (!eq || !ek || !ev) return 0;
    if (eq->rows != q_dim || ek->rows != kv_dim || ev->rows != kv_dim) return 0;
    if (eq->cols != in_dim || ek->cols != in_dim || ev->cols != in_dim) return 0;
    if (!kai_op_on(eq->comp, eq->fam)) return 0;

    const uint8_t *lp = kai_lhs_sym_mode()
        ? kai_i8_pack_lhs_sym(x, (size_t)in_dim * sizeof(float), in_dim, 1, 0)
        : kai_i8_pack_lhs(x, (size_t)in_dim * sizeof(float), in_dim, 1, 0);
    if (!lp) return 0;

    if (!kai_qkv_fused())
        return kai_i8_run_packed(eq, q, lp, (size_t)q_dim  * sizeof(float), q_dim,  in_dim, 1, 0)
            && kai_i8_run_packed(ek, k, lp, (size_t)kv_dim * sizeof(float), kv_dim, in_dim, 1, 0)
            && kai_i8_run_packed(ev, v, lp, (size_t)kv_dim * sizeof(float), kv_dim, in_dim, 1, 0);

    kai_i8_qkv_job_t job;
    job.e[0] = eq; job.e[1] = ek; job.e[2] = ev;
    job.dst[0] = q; job.dst[1] = k; job.dst[2] = v;
    job.dst_stride[0] = (size_t)q_dim  * sizeof(float);
    job.dst_stride[1] = (size_t)kv_dim * sizeof(float);
    job.dst_stride[2] = (size_t)kv_dim * sizeof(float);
    job.n[0] = (size_t)q_dim; job.n[1] = (size_t)kv_dim; job.n[2] = (size_t)kv_dim;
    job.lhs_packed = lp;
    job.m = 1; job.k = (size_t)in_dim; job.gemm = 0;

    const size_t n_step = KI8_GEMV(get_n_step)();
    job.cum[0] = 0;
    for (int i = 0; i < 3; i++) {
        job.tiles[i] = (job.n[i] + n_step - 1) / n_step;
        job.cum[i + 1] = job.cum[i] + job.tiles[i];
    }

    size_t nt = (size_t)qwen_get_threads();
    if (nt < 1) nt = 1;
    const size_t rows_total = job.n[0] + job.n[1] + job.n[2];
    if (rows_total < nt * 16) nt = 1;
    if (nt == 1) kai_i8_qkv_task(0, 1, &job);
    else         qwen_parallel(nt, kai_i8_qkv_task, &job);
    return 1;
}

int qwen_kleidi_matmul_i8_native(float *dst, const void *key, const float *lhs,
                                 size_t lhs_stride, size_t dst_stride,
                                 int rows, int cols, int B) {
    if (!qwen_kleidi_i8_enabled() || B < 1 || g_kai_bypass) return 0;
    const kai_entry_t *e = kai_lookup_kind(key, KAI_KIND_I8);
    if (!e || e->rows != rows || e->cols != cols) return 0;
    if (!kai_op_on(e->comp, e->fam)) return 0;
    if (!kai_op_on(e->comp, e->fam)) return 0;
    return kai_i8_run(e, dst, lhs, lhs_stride, dst_stride, rows, cols, B);
}

int qwen_kleidi_matmul_i8(float *Y, const void *key, const float *X,
                          int rows, int cols, int B) {
    if (!qwen_kleidi_i8_enabled() || B < 1 || g_kai_bypass) return 0;
    const kai_entry_t *e = kai_lookup_kind(key, KAI_KIND_I8);
    if (!e || e->rows != rows || e->cols != cols) return 0;

    const float *lhs = X;
    size_t lhs_stride = (size_t)cols * sizeof(float);
    float *dst = Y;
    size_t dst_stride = (size_t)rows * sizeof(float);
    if (B > 1) {
        float *xt = kai_scratch_i8xt((size_t)B * cols);
        float *yt = kai_scratch_i8yt((size_t)B * rows);
        if (!xt || !yt) return 0;
        for (int b = 0; b < B; b++)
            for (int c = 0; c < cols; c++) xt[(size_t)b * cols + c] = X[(size_t)c * B + b];
        lhs = xt; dst = yt;
    }
    if (!kai_i8_run(e, dst, lhs, lhs_stride, dst_stride, rows, cols, B)) return 0;
    if (B > 1)
        for (int b = 0; b < B; b++)
            for (int r = 0; r < rows; r++) Y[(size_t)r * B + b] = dst[(size_t)b * rows + r];
    return 1;
}

#if QWEN_KLEIDI_BF16_BUILD
#define KBF_GEMV(f) kai_##f##_matmul_clamp_f32_bf16p1x4_bf16p12x4b_1x36_neon_dot
#define KBF_GEMM(f) kai_##f##_matmul_clamp_f32_bf16p8x4_bf16p12x4b_8x12_neon_mmla

int qwen_kleidi_bf16_enabled(void) {
    static atomic_int cached = -1;
    int v = atomic_load_explicit(&cached, memory_order_relaxed);
    if (v < 0) {
        const char *e = getenv("QWEN_NO_KAI_BF16");
        v = qwen_kleidi_enabled() && !(e && e[0] && e[0] != '0');
        atomic_store_explicit(&cached, v, memory_order_relaxed);
    }
    return v;
}

int qwen_kleidi_register_bf16(const void *key, const uint16_t *W, int rows, int cols) {
    return qwen_kleidi_register_bf16_fam(key, W, rows, cols,
                                         QWEN_KAI_COMP_TALKER, QWEN_KAI_FAM_OTHER);
}
int qwen_kleidi_register_bf16_fam(const void *key, const uint16_t *W, int rows, int cols,
                                  int comp, int fam) {
    if (!qwen_kleidi_bf16_enabled() || !key || !W) return 0;
    if (rows <= 0 || cols <= 0) return 0;
    if (kai_lookup_kind(key, KAI_KIND_BF16)) return 1;

    const size_t nr = KBF_GEMM(get_nr)(), kr = KBF_GEMM(get_kr)(), sr = KBF_GEMM(get_sr)();
    size_t sz = kai_get_rhs_packed_size_rhs_quant_pack_kxn_bf16p12x4biasf32_f32_neon(
                    (size_t)rows, (size_t)cols, nr, kr);
    void *rhs = aligned_malloc(sz);
    if (!rhs) return 0;
    float *kxn = (float *)malloc((size_t)rows * (size_t)cols * sizeof(float));
    float *bias = (float *)calloc((size_t)rows, sizeof(float));
    if (!kxn || !bias) { free(kxn); free(bias); free(rhs); return 0; }
    for (int n = 0; n < rows; n++)
        for (int k = 0; k < cols; k++) {
            uint32_t u = (uint32_t)W[(size_t)n * cols + k] << 16;
            float f; memcpy(&f, &u, 4);
            kxn[(size_t)k * rows + n] = f;
        }
    kai_run_rhs_quant_pack_kxn_bf16p12x4biasf32_f32_neon(
        1, (size_t)rows, (size_t)cols, nr, kr, sr,
        (size_t)rows * sizeof(float), kxn, bias, NULL, rhs, 0, NULL);
    free(kxn); free(bias);
    return kai_insert_fam(key, rhs, rows, cols, KAI_KIND_BF16, sz, comp, fam);
}

typedef struct {
    const kai_entry_t *e;
    const void *lhs_packed;
    float *dst;
    size_t m, n, k, dst_stride_row, n_chunk;
    int gemm;
} kai_bf_job_t;

static int kbf_trace(void) {
    static int t = -1;
    if (t < 0) { const char *e = getenv("QWEN_TTFA_TRACE"); t = (e && e[0] && e[0] != '0'); }
    return t;
}
static double kbf_now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}
static double kbf_cpu_ms(void) {
#if defined(CLOCK_THREAD_CPUTIME_ID)
    struct timespec ts;
    if (clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts) == 0)
        return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
#endif
    return -1.0;
}
static double kbf_busy[64];
static double kbf_st[64], kbf_en[64], kbf_cpu[64];
static int kbf_tiles[64], kbf_seen[64];

static void kai_bf_task(size_t tid, size_t nt, void *ctx) {
    kai_bf_job_t *j = (kai_bf_job_t *)ctx;
    const int _tr = kbf_trace();
    double _b0 = _tr ? kbf_now_ms() : 0.0;
    double _c0 = _tr ? kbf_cpu_ms() : 0.0;
    if (_tr && tid < 64) { kbf_st[tid] = _b0; kbf_seen[tid] = 1; kbf_tiles[tid] = 0; }
    const size_t n_step = j->gemm ? KBF_GEMM(get_n_step)() : KBF_GEMV(get_n_step)();
    size_t tiles = (j->n + n_step - 1) / n_step;
    size_t t0 = tiles * tid / nt, t1 = tiles * (tid + 1) / nt;
    size_t n0 = t0 * n_step;
    size_t n1 = (t1 * n_step < j->n) ? t1 * n_step : j->n;
    if (_tr && tid < 64) kbf_tiles[tid] = (int)(t1 - t0);
    if (n0 >= n1) { if (_tr && tid < 64) { kbf_busy[tid] = 0.0; kbf_en[tid] = kbf_now_ms(); } return; }
    if (j->gemm) {
        size_t chunk = n1 - n0;
        if (j->n_chunk > 0) {
            size_t c = j->n_chunk - (j->n_chunk % n_step);
            if (c >= n_step) chunk = c;
        }
        for (size_t c0 = n0; c0 < n1; c0 += chunk) {
            size_t c1 = c0 + chunk < n1 ? c0 + chunk : n1;
            const size_t roff = KBF_GEMM(get_rhs_packed_offset)(c0, j->k);
            const size_t doff = KBF_GEMM(get_dst_offset)(0, c0, j->dst_stride_row);
            KBF_GEMM(run)(j->m, c1 - c0, j->k, j->lhs_packed,
                          (const uint8_t *)j->e->rhs + roff,
                          (float *)((uint8_t *)j->dst + doff),
                          j->dst_stride_row, sizeof(float), -FLT_MAX, FLT_MAX);
        }
    } else {
        const size_t roff = KBF_GEMV(get_rhs_packed_offset)(n0, j->k);
        const size_t doff = KBF_GEMV(get_dst_offset)(0, n0, j->dst_stride_row);
        KBF_GEMV(run)(j->m, n1 - n0, j->k, j->lhs_packed,
                      (const uint8_t *)j->e->rhs + roff,
                      (float *)((uint8_t *)j->dst + doff),
                      j->dst_stride_row, sizeof(float), -FLT_MAX, FLT_MAX);
    }
    if (_tr && tid < 64) { double _e = kbf_now_ms(); kbf_busy[tid] = _e - _b0; kbf_en[tid] = _e;
                           double _c1 = kbf_cpu_ms();
                           kbf_cpu[tid] = (_c0 >= 0 && _c1 >= 0) ? _c1 - _c0 : -1.0; }
}

QWEN_KAI_SCRATCH(bflhs, uint8_t)
QWEN_KAI_SCRATCH(bfxt,  float)
QWEN_KAI_SCRATCH(bfyt,  float)

static int kai_bf_run(const kai_entry_t *e, float *dst, const float *lhs,
                      size_t lhs_stride, size_t dst_stride, int rows, int cols, int B) {
    const int gemm = (B > 1);
    const size_t mr = gemm ? KBF_GEMM(get_mr)() : KBF_GEMV(get_mr)();
    const size_t kr = gemm ? KBF_GEMM(get_kr)() : KBF_GEMV(get_kr)();
    const size_t sr = gemm ? KBF_GEMM(get_sr)() : KBF_GEMV(get_sr)();
    size_t lhs_sz = gemm
        ? kai_get_lhs_packed_size_lhs_quant_pack_bf16p8x4_f32_neon((size_t)B, (size_t)cols, mr, kr, sr)
        : kai_get_lhs_packed_size_lhs_quant_pack_bf16p1x4_f32_neon((size_t)B, (size_t)cols, mr, kr, sr);
    uint8_t *lhs_packed = kai_scratch_bflhs(lhs_sz);
    if (!lhs_packed) return 0;
    const int tr = kbf_trace();
    double t0 = tr ? kbf_now_ms() : 0.0;
    if (gemm) kai_run_lhs_quant_pack_bf16p8x4_f32_neon((size_t)B, (size_t)cols, mr, kr, sr, 0,
                                                       lhs, lhs_stride, lhs_packed);
    else      kai_run_lhs_quant_pack_bf16p1x4_f32_neon((size_t)B, (size_t)cols, mr, kr, sr, 0,
                                                       lhs, lhs_stride, lhs_packed);
    if (tr) { qwen_kbf_pack_ms += kbf_now_ms() - t0; t0 = kbf_now_ms(); }

    static int nchunk = -1;
    if (nchunk < 0) { const char *e2 = getenv("QWEN_KAI_NCHUNK"); nchunk = (e2 && *e2) ? atoi(e2) : 384; }
    kai_bf_job_t job = { e, lhs_packed, dst, (size_t)B, (size_t)rows, (size_t)cols,
                         dst_stride, (size_t)(nchunk > 0 ? nchunk : 0), gemm };
    size_t nt = (size_t)qwen_get_threads();
    if (nt < 1) nt = 1;
    if ((size_t)rows < nt * 16) nt = 1;
    static int rep = -1;
    if (rep < 0) { const char *e4 = getenv("QWEN_KAI_REPEAT"); rep = (e4 && *e4 && *e4 != '0'); }
    double r_t0 = rep ? kbf_now_ms() : 0.0;
    if (nt == 1) kai_bf_task(0, 1, &job);
    else         qwen_parallel(nt, kai_bf_task, &job);
    if (rep) {
        double r_t1 = kbf_now_ms();
        if (nt == 1) kai_bf_task(0, 1, &job);
        else         qwen_parallel(nt, kai_bf_task, &job);
        double r_t2 = kbf_now_ms();
        fprintf(stderr, "[KAIREP] v=1 m=%d n=%d k=%d cold_ms=%.4f hot_ms=%.4f ratio=%.3f\n",
                B, rows, cols, r_t1 - r_t0, r_t2 - r_t1,
                (r_t1 - r_t0) > 0 ? (r_t2 - r_t1) / (r_t1 - r_t0) : 0.0);
    }
    if (tr) {
        double _rend = kbf_now_ms();
        qwen_kbf_gemm_ms += _rend - t0;
        double smin = 1e300, smax = -1e300, emin = 1e300, emax = -1e300;
        double bmin = 1e300, bmax = 0, bsum = 0; int tmin = 1 << 30, tmax = 0, nw = 0;
        int last_start_tid = -1, last_end_tid = -1;
        for (size_t i = 0; i < nt && i < 64; i++) {
            if (!kbf_seen[i]) continue;
            nw++;
            if (kbf_st[i] < smin) smin = kbf_st[i];
            if (kbf_st[i] > smax) { smax = kbf_st[i]; last_start_tid = (int)i; }
            if (kbf_en[i] < emin) emin = kbf_en[i];
            if (kbf_en[i] > emax) { emax = kbf_en[i]; last_end_tid = (int)i; }
            if (kbf_busy[i] < bmin) bmin = kbf_busy[i];
            if (kbf_busy[i] > bmax) bmax = kbf_busy[i];
            bsum += kbf_busy[i];
            if (kbf_tiles[i] < tmin) tmin = kbf_tiles[i];
            if (kbf_tiles[i] > tmax) tmax = kbf_tiles[i];
        }
        double cpu_sum = 0, cpu_max = 0, off_sum = 0; int cpu_n = 0;
        for (size_t i = 0; i < nt && i < 64; i++) {
            if (!kbf_seen[i] || kbf_cpu[i] < 0) continue;
            cpu_n++; cpu_sum += kbf_cpu[i];
            if (kbf_cpu[i] > cpu_max) cpu_max = kbf_cpu[i];
            double off = kbf_busy[i] - kbf_cpu[i];
            off_sum += off > 0 ? off : 0.0;
        }
        double _entry = (nw > 0) ? smax - t0 : 0.0;
        double _first = (nw > 0) ? smin - t0 : 0.0;
        double _exec  = (nw > 0) ? (emax - smax) - (emin > smax ? emin - smax : 0.0) : 0.0;
        double _exit  = (nw > 0) ? emax - emin : 0.0;
        if (nw > 0)
            fprintf(stderr, "[GEMMT] v=1 m=%d n=%d k=%d workers=%d region_ms=%.4f "
                    "start_skew_ms=%.4f finish_tail_ms=%.4f busy_min=%.4f busy_mean=%.4f "
                    "busy_max=%.4f tiles_min=%d tiles_max=%d util=%.3f "
                    "first_start_ms=%.4f entry_ms=%.4f exec_ms=%.4f exit_ms=%.4f "
                    "last_start_tid=%d last_end_tid=%d cpu_n=%d cpu_sum=%.4f "
                    "cpu_max=%.4f offcpu_sum=%.4f\n",
                    (int)B, rows, cols, nw, _rend - t0, smax - smin, emax - emin,
                    bmin, bsum / nw, bmax, tmin, tmax,
                    (_rend - t0) > 0 ? bsum / ((double)nw * (_rend - t0)) : 0.0,
                    _first, _entry, _exec, _exit, last_start_tid, last_end_tid,
                    cpu_n, cpu_sum, cpu_max, off_sum);
        for (size_t i = 0; i < nt && i < 64; i++) kbf_seen[i] = 0;
        double sum = 0, mx = 0;
        size_t cnt = nt < 64 ? nt : 64;
        for (size_t i = 0; i < cnt; i++) { sum += kbf_busy[i]; if (kbf_busy[i] > mx) mx = kbf_busy[i]; }
        qwen_kbf_busy_mean_ms += cnt ? sum / (double)cnt : 0.0;
        qwen_kbf_busy_max_ms  += mx;
    }
    return 1;
}

int qwen_kleidi_matmul_bf16_native(float *dst, const void *key, const float *lhs,
                                   size_t lhs_stride, size_t dst_stride,
                                   int rows, int cols, int B) {
    if (!qwen_kleidi_bf16_enabled() || B < 1 || g_kai_bypass) return 0;
    const kai_entry_t *e = kai_lookup_kind(key, KAI_KIND_BF16);
    if (!e || e->rows != rows || e->cols != cols) return 0;
    if (!kai_op_on(e->comp, e->fam)) return 0;
    if (!kai_op_on(e->comp, e->fam)) return 0;
    return kai_bf_run(e, dst, lhs, lhs_stride, dst_stride, rows, cols, B);
}

int qwen_kleidi_matmul_bf16(float *Y, const void *key, const float *X,
                            int rows, int cols, int B) {
    if (!qwen_kleidi_bf16_enabled() || B < 1 || g_kai_bypass) return 0;
    const kai_entry_t *e = kai_lookup_kind(key, KAI_KIND_BF16);
    if (!e || e->rows != rows || e->cols != cols) return 0;
    const float *lhs = X;
    size_t lhs_stride = (size_t)cols * sizeof(float);
    float *dst = Y;
    size_t dst_stride = (size_t)rows * sizeof(float);
    if (B > 1) {
        float *xt = kai_scratch_bfxt((size_t)B * cols);
        float *yt = kai_scratch_bfyt((size_t)B * rows);
        if (!xt || !yt) return 0;
        for (int b = 0; b < B; b++)
            for (int c = 0; c < cols; c++) xt[(size_t)b * cols + c] = X[(size_t)c * B + b];
        lhs = xt; dst = yt;
    }
    if (!kai_bf_run(e, dst, lhs, lhs_stride, dst_stride, rows, cols, B)) return 0;
    if (B > 1)
        for (int b = 0; b < B; b++)
            for (int r = 0; r < rows; r++) Y[(size_t)r * B + b] = dst[(size_t)b * rows + r];
    return 1;
}
#else
int qwen_kleidi_bf16_enabled(void) { return 0; }
int qwen_kleidi_register_bf16(const void *k, const uint16_t *W, int r, int c) {
    (void)k; (void)W; (void)r; (void)c; return 0;
}
int qwen_kleidi_matmul_bf16(float *Y, const void *k, const float *X, int r, int c, int B) {
    (void)Y; (void)k; (void)X; (void)r; (void)c; (void)B; return 0;
}
int qwen_kleidi_matmul_bf16_native(float *d, const void *k, const float *l, size_t ls,
                                   size_t ds, int r, int c, int B) {
    (void)d; (void)k; (void)l; (void)ls; (void)ds; (void)r; (void)c; (void)B; return 0;
}
#endif

void qwen_kleidi_stats_by_kind(int *n_q4, size_t *b_q4, int *n_i8, size_t *b_i8,
                               int *n_bf, size_t *b_bf) {
    if (n_q4) *n_q4 = g_kai_n_kind[KAI_KIND_Q4];
    if (b_q4) *b_q4 = g_kai_bytes_kind[KAI_KIND_Q4];
    if (n_i8) *n_i8 = g_kai_n_kind[KAI_KIND_I8];
    if (b_i8) *b_i8 = g_kai_bytes_kind[KAI_KIND_I8];
    if (n_bf) *n_bf = g_kai_n_kind[KAI_KIND_BF16];
    if (b_bf) *b_bf = g_kai_bytes_kind[KAI_KIND_BF16];
}

#else

int qwen_kleidi_register_q4(const void *k, const uint8_t *b, int r, int c) {
    (void)k; (void)b; (void)r; (void)c; return 0;
}
int qwen_kleidi_matmul_q4(float *Y, const void *k, const float *X, int r, int c, int B) {
    (void)Y; (void)k; (void)X; (void)r; (void)c; (void)B; return 0;
}
int qwen_kleidi_selfcheck(const void *k, int r, int c, float *a, float *rel) {
    (void)k; (void)r; (void)c; (void)a; (void)rel; return 0;
}
void qwen_kleidi_stats(int *n, size_t *b) { if (n) *n = 0; if (b) *b = 0; }
int qwen_kleidi_i8_enabled(void) { return 0; }
int qwen_kleidi_bf16_enabled(void) { return 0; }
int qwen_kleidi_register_i8(const void *k, const int8_t *W, const float *s, int r, int c) {
    (void)k; (void)W; (void)s; (void)r; (void)c; return 0;
}
int qwen_kleidi_register_i8_fam(const void *k, const int8_t *W, const float *s, int r,
                                int c, int cm, int f) {
    (void)k; (void)W; (void)s; (void)r; (void)c; (void)cm; (void)f; return 0;
}
int qwen_kleidi_register_bf16_fam(const void *k, const uint16_t *W, int r, int c,
                                  int cm, int f) {
    (void)k; (void)W; (void)r; (void)c; (void)cm; (void)f; return 0;
}
int qwen_kleidi_prefill_enabled(void) { return 0; }
int qwen_kleidi_matmul_i8(float *Y, const void *k, const float *X, int r, int c, int B) {
    (void)Y; (void)k; (void)X; (void)r; (void)c; (void)B; return 0;
}
int qwen_kleidi_matmul_i8_native(float *d, const void *k, const float *l, size_t ls,
                                 size_t ds, int r, int c, int B) {
    (void)d; (void)k; (void)l; (void)ls; (void)ds; (void)r; (void)c; (void)B; return 0;
}
int qwen_kleidi_matmul_i8_qkv_native(float *dq, float *dk, float *dv, const void *a,
                                     const void *b, const void *c, const float *x,
                                     size_t ls, int i, int q, int kv, int B) {
    (void)dq; (void)dk; (void)dv; (void)a; (void)b; (void)c; (void)x; (void)ls;
    (void)i; (void)q; (void)kv; (void)B; return 0;
}
int qwen_kleidi_matmul_i8_qkv(float *q, float *k, float *v, const void *a, const void *b,
                              const void *c, const float *x, int i, int qd, int kd) {
    (void)q; (void)k; (void)v; (void)a; (void)b; (void)c; (void)x; (void)i; (void)qd; (void)kd;
    return 0;
}
int qwen_kleidi_register_bf16(const void *k, const uint16_t *W, int r, int c) {
    (void)k; (void)W; (void)r; (void)c; return 0;
}
int qwen_kleidi_matmul_bf16(float *Y, const void *k, const float *X, int r, int c, int B) {
    (void)Y; (void)k; (void)X; (void)r; (void)c; (void)B; return 0;
}
int qwen_kleidi_matmul_bf16_native(float *d, const void *k, const float *l, size_t ls,
                                   size_t ds, int r, int c, int B) {
    (void)d; (void)k; (void)l; (void)ls; (void)ds; (void)r; (void)c; (void)B; return 0;
}
void qwen_kleidi_stats_by_kind(int *a, size_t *b, int *c, size_t *d, int *e, size_t *f) {
    if (a) *a = 0; if (b) *b = 0; if (c) *c = 0; if (d) *d = 0; if (e) *e = 0; if (f) *f = 0;
}

#endif
