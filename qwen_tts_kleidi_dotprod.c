/*
 * Arm KleidiAI dotprod-only GEMV candidates.
 *
 * The regular qwen_tts_kleidi.c backend owns the i8mm packing contract used by
 * the B>1 paths.  Its dotprod 1x kernels are nevertheless independently useful
 * on dotprod-only CPUs.  Keep a separate registry and a separate packed RHS so
 * a dotprod-only build can qualify B=1 without claiming that the i8mm GEMM
 * layout is valid for it.
 *
 * This file is deliberately opt-in (QWEN_KAI_DOTPROD_GEMV=1).  It is a
 * qualification candidate, not a production performance claim.
 */
#include "qwen_tts_kleidi.h"

#include <stdatomic.h>
#include <stddef.h>
#include <stdint.h>
#include <float.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>

#ifdef __linux__
#include <sys/auxv.h>
#if defined(__aarch64__) || defined(__arm__)
#include <asm/hwcap.h>
#endif
#endif
#ifdef __APPLE__
#include <sys/sysctl.h>
#endif

#if defined(__aarch64__) && defined(__ARM_FEATURE_DOTPROD)
#define QWEN_KAI_DOTPROD_BUILD 1
#include "kai/kai_common.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0.h"
#include "kai/ukernels/matmul/pack/kai_lhs_quant_pack_qsi8d32p_f32.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod.h"
#include "kai/ukernels/matmul/pack/kai_lhs_quant_pack_qai8dxp_f32.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qsi8cxp_qsi8cx_neon.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod.h"
#else
#define QWEN_KAI_DOTPROD_BUILD 0
#endif

int qwen_kleidi_dotprod_compiled(void) { return QWEN_KAI_DOTPROD_BUILD; }

static int dotprod_cpu_ok(void) {
#if !QWEN_KAI_DOTPROD_BUILD
    return 0;
#elif defined(__linux__)
    unsigned long h = getauxval(AT_HWCAP);
#ifdef HWCAP_ASIMDDP
    return (h & HWCAP_ASIMDDP) != 0;
#else
    (void)h;
    return 0;
#endif
#elif defined(__APPLE__)
    int v = 0;
    size_t sz = sizeof v;
    return sysctlbyname("hw.optional.arm.FEAT_DotProd", &v, &sz, NULL, 0) == 0 && v;
#else
    return 0;
#endif
}

int qwen_kleidi_dotprod_supported(void) {
    return QWEN_KAI_DOTPROD_BUILD && dotprod_cpu_ok();
}

int qwen_kleidi_dotprod_enabled(void) {
    static atomic_int cached = -1;
    int v = atomic_load_explicit(&cached, memory_order_relaxed);
    if (v < 0) {
        const char *e = getenv("QWEN_KAI_DOTPROD_GEMV");
        const char *off = getenv("QWEN_NO_KLEIDI");
        v = qwen_kleidi_dotprod_supported() && e && e[0] == '1' && !(off && off[0] == '1');
        atomic_store_explicit(&cached, v, memory_order_relaxed);
    }
    return v;
}

#if QWEN_KAI_DOTPROD_BUILD

enum { DOT_KIND_Q4 = 0, DOT_KIND_I8, DOT_KIND_N };
typedef struct {
    const void *key;
    void *rhs;
    int rows;
    int cols;
    int kind;
    size_t bytes;
} dot_entry_t;

static dot_entry_t *g_dot;
static _Atomic int g_dot_n;
static int g_dot_cap;
static pthread_mutex_t g_dot_mu = PTHREAD_MUTEX_INITIALIZER;

static void *dot_aligned_alloc(size_t bytes) {
    void *p = NULL;
    if (posix_memalign(&p, 64, bytes ? bytes : 64) != 0) return NULL;
    return p;
}

static const dot_entry_t *dot_lookup(const void *key, int kind) {
    int n = atomic_load_explicit(&g_dot_n, memory_order_acquire);
    for (int i = 0; i < n; i++)
        if (g_dot[i].key == key && g_dot[i].kind == kind) return &g_dot[i];
    return NULL;
}

static int dot_insert(const void *key, void *rhs, int rows, int cols, int kind, size_t bytes) {
    pthread_mutex_lock(&g_dot_mu);
    if (dot_lookup(key, kind)) {
        pthread_mutex_unlock(&g_dot_mu);
        free(rhs);
        return 1;
    }
    if (g_dot_n == g_dot_cap) {
        int cap = g_dot_cap ? g_dot_cap * 2 : 128;
        dot_entry_t *p = (dot_entry_t *)realloc(g_dot, (size_t)cap * sizeof *p);
        if (!p) {
            pthread_mutex_unlock(&g_dot_mu);
            free(rhs);
            return 0;
        }
        g_dot = p;
        g_dot_cap = cap;
    }
    int at = atomic_load_explicit(&g_dot_n, memory_order_relaxed);
    g_dot[at] = (dot_entry_t){ key, rhs, rows, cols, kind, bytes };
    atomic_store_explicit(&g_dot_n, at + 1, memory_order_release);
    pthread_mutex_unlock(&g_dot_mu);
    return 1;
}

int qwen_kleidi_dotprod_register_q4(const void *key, const uint8_t *ggml_blocks,
                                    int rows, int cols) {
    if (!qwen_kleidi_dotprod_enabled() || !key || !ggml_blocks || rows <= 0 || cols <= 0)
        return 0;
    if (cols % 32 != 0 || dot_lookup(key, DOT_KIND_Q4)) return dot_lookup(key, DOT_KIND_Q4) != NULL;

    const size_t nr = kai_get_nr_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod();
    const size_t kr = kai_get_kr_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod();
    const size_t sr = kai_get_sr_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod();
    const size_t bytes = kai_get_rhs_packed_size_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0(
        (size_t)rows, (size_t)cols, nr, kr, 32);
    void *rhs = dot_aligned_alloc(bytes);
    if (!rhs) return 0;
    struct kai_rhs_pack_qs4cxs1s0_param params = { .lhs_zero_point = 1, .rhs_zero_point = 8 };
    kai_run_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0(
        1, (size_t)rows, (size_t)cols, nr, kr, sr, 32,
        ggml_blocks, NULL, rhs, 0, &params);
    return dot_insert(key, rhs, rows, cols, DOT_KIND_Q4, bytes);
}

int qwen_kleidi_dotprod_matmul_q4(float *Y, const void *key, const float *X,
                                  int rows, int cols, int B) {
    if (!qwen_kleidi_dotprod_enabled() || !Y || !X || B != 1) return 0;
    const dot_entry_t *e = dot_lookup(key, DOT_KIND_Q4);
    if (!e || e->rows != rows || e->cols != cols) return 0;
    const size_t mr = kai_get_mr_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod();
    const size_t kr = kai_get_kr_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod();
    const size_t sr = kai_get_sr_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod();
    const size_t bytes = kai_get_lhs_packed_size_lhs_quant_pack_qsi8d32p_f32(
        1, (size_t)cols, 32, mr, kr, sr);
    void *lhs = dot_aligned_alloc(bytes);
    if (!lhs) return 0;
    kai_run_lhs_quant_pack_qsi8d32p_f32(1, (size_t)cols, 32, mr, kr, sr, 0,
                                        X, (size_t)cols * sizeof(float), lhs);
    kai_run_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod(
        1, (size_t)rows, (size_t)cols, 32, lhs, e->rhs, Y,
        (size_t)rows * sizeof(float), sizeof(float), -FLT_MAX, FLT_MAX);
    free(lhs);
    return 1;
}

int qwen_kleidi_dotprod_register_i8(const void *key, const int8_t *W, const float *scale,
                                    int rows, int cols) {
    if (!qwen_kleidi_dotprod_enabled() || !key || !W || !scale || rows <= 0 || cols <= 0)
        return 0;
    if (dot_lookup(key, DOT_KIND_I8)) return 1;
    const size_t nr = kai_get_nr_matmul_clamp_f32_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod();
    const size_t kr = kai_get_kr_matmul_clamp_f32_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod();
    const size_t sr = kai_get_sr_matmul_clamp_f32_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod();
    const size_t bytes = kai_get_rhs_packed_size_rhs_pack_nxk_qsi8cxp_qsi8cx_neon(
        (size_t)rows, (size_t)cols, nr, kr, sr);
    void *rhs = dot_aligned_alloc(bytes);
    if (!rhs) return 0;
    struct kai_rhs_pack_qsi8cx_params params = { .lhs_zero_point = 1, .scale_multiplier = 1.0f };
    kai_run_rhs_pack_nxk_qsi8cxp_qsi8cx_neon(
        1, (size_t)rows, (size_t)cols, nr, kr, sr,
        W, NULL, scale, rhs, 0, &params);
    return dot_insert(key, rhs, rows, cols, DOT_KIND_I8, bytes);
}

int qwen_kleidi_dotprod_matmul_i8(float *Y, const void *key, const float *X,
                                  int rows, int cols, int B) {
    if (!qwen_kleidi_dotprod_enabled() || !Y || !X || B != 1) return 0;
    const dot_entry_t *e = dot_lookup(key, DOT_KIND_I8);
    if (!e || e->rows != rows || e->cols != cols) return 0;
    const size_t mr = kai_get_mr_matmul_clamp_f32_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod();
    const size_t kr = kai_get_kr_matmul_clamp_f32_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod();
    const size_t sr = kai_get_sr_matmul_clamp_f32_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod();
    const size_t bytes = kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_f32(
        1, (size_t)cols, mr, kr, sr);
    void *lhs = dot_aligned_alloc(bytes);
    if (!lhs) return 0;
    kai_run_lhs_quant_pack_qai8dxp_f32(1, (size_t)cols, mr, kr, sr, 0,
                                       X, (size_t)cols * sizeof(float), lhs);
    kai_run_matmul_clamp_f32_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod(
        1, (size_t)rows, (size_t)cols, lhs, e->rhs, Y,
        (size_t)rows * sizeof(float), sizeof(float), -FLT_MAX, FLT_MAX);
    free(lhs);
    return 1;
}

#else

int qwen_kleidi_dotprod_register_q4(const void *k, const uint8_t *b, int r, int c) {
    (void)k; (void)b; (void)r; (void)c; return 0;
}
int qwen_kleidi_dotprod_matmul_q4(float *y, const void *k, const float *x, int r, int c, int B) {
    (void)y; (void)k; (void)x; (void)r; (void)c; (void)B; return 0;
}
int qwen_kleidi_dotprod_register_i8(const void *k, const int8_t *w, const float *s, int r, int c) {
    (void)k; (void)w; (void)s; (void)r; (void)c; return 0;
}
int qwen_kleidi_dotprod_matmul_i8(float *y, const void *k, const float *x, int r, int c, int B) {
    (void)y; (void)k; (void)x; (void)r; (void)c; (void)B; return 0;
}

#endif
