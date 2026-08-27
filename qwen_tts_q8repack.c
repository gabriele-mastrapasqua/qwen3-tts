/* qwen_tts_q8repack.c — see the header for why this exists instead of KleidiAI.
 *
 * THE LAYOUT, AND WHY IT COSTS NOTHING
 * ggml's `make_block_q8_0x4` is one loop:
 *     chunk i (8 bytes)  <-  row (i % 4), byte offset (i / 4) * 8
 * Sixteen chunks, 128 bytes, plus the four fp16 scales copied verbatim. Nothing is
 * dequantized, nothing is rescaled, and 136 bytes in = 136 bytes out. That is what
 * "lossless" means here, and `qwen_q8r_derepack` exists to prove it by memcmp rather
 * than by argument.
 *
 * WHY THE INTERLEAVE IS EXACTLY WHAT SMMLA WANTS
 * For offset group g (weights j = 8g..8g+7), the chunks for rows 0..3 are consecutive:
 *     qs[32g .. 32g+15]  = rows 0,1  ->  a 2x8 int8 operand
 *     qs[32g+16 .. +31]  = rows 2,3  ->  another one
 * `vmmlaq_s32(acc, A, B)` multiplies two 2x8 int8 operands into a 2x2 int32 tile, so a
 * plain `vld1q_s8` is the whole load. This is the repack paying for itself: our own
 * int8 SMMLA kernel synthesises that pairing in registers on every call
 * (qwen_tts_kernels.c:3113), reading two rows `cols` bytes apart.
 *
 * ACTIVATIONS ARE PER-32 TOO
 * The weights keep a scale every 32 values, so the activation is quantized the same
 * way (int8 + fp16 scale per 32). Both sides per-block is the whole point: a per-row
 * activation scale would give back the precision the format was chosen for.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdatomic.h>
#include <pthread.h>
#include <time.h>
#include "qwen_tts_q8repack.h"
#include "qwen_tts_kernels.h"
#include "qwen_tts_thread.h"

#if defined(__aarch64__) && defined(__ARM_NEON)
#include <arm_neon.h>
#define Q8R_NEON 1
#else
#define Q8R_NEON 0
#endif
#if defined(__ARM_FEATURE_MATMUL_INT8)
#define Q8R_I8MM 1
#else
#define Q8R_I8MM 0
#endif
#if defined(__ARM_FEATURE_DOTPROD)
#define Q8R_DOT 1
#else
#define Q8R_DOT 0
#endif

#ifdef __linux__
#include <sys/auxv.h>
/* asm/hwcap.h is ARM-only and absent on x86 Linux, where this translation unit is
 * still compiled (the NEON body below is guarded, the file is not). The only reader
 * of these macros is already behind __aarch64__. */
#if defined(__aarch64__) || defined(__arm__)
#include <asm/hwcap.h>
#endif
#endif
#ifdef __APPLE__
#include <sys/sysctl.h>
#endif

/* Runtime feature check, asked of the OS. Same reasoning as the KleidiAI path: a
 * binary built with +i8mm on a CPU without it is a SIGILL with no diagnostic. */
static int q8r_cpu(int *has_dot, int *has_i8mm) {
    int d = 0, m = 0;
#if defined(__linux__) && defined(__aarch64__)
    unsigned long h1 = getauxval(AT_HWCAP), h2 = getauxval(AT_HWCAP2);
#ifdef HWCAP_ASIMDDP
    d = (h1 & HWCAP_ASIMDDP) != 0;
#endif
#ifdef HWCAP2_I8MM
    m = (h2 & HWCAP2_I8MM) != 0;
#endif
    (void)h1; (void)h2;
#elif defined(__APPLE__) && defined(__aarch64__)
    int v = 0; size_t sz = sizeof v;
    d = (sysctlbyname("hw.optional.arm.FEAT_DotProd", &v, &sz, NULL, 0) == 0 && v);
    v = 0; sz = sizeof v;
    m = (sysctlbyname("hw.optional.arm.FEAT_I8MM", &v, &sz, NULL, 0) == 0 && v);
#endif
    if (has_dot)  *has_dot  = d && Q8R_DOT;
    if (has_i8mm) *has_i8mm = m && Q8R_I8MM;
    return (d && Q8R_DOT) || (m && Q8R_I8MM);
}

int qwen_q8r_supported(void) {
#if Q8R_NEON
    return q8r_cpu(NULL, NULL);
#else
    return 0;
#endif
}

int qwen_q8r_enabled(void) {
    static atomic_int cached = -1;
    int v = atomic_load_explicit(&cached, memory_order_relaxed);
    if (v < 0) {
        const char *e = getenv("QWEN_NO_Q8REPACK");
        v = qwen_q8r_supported() && !(e && e[0] == '1');
        atomic_store_explicit(&cached, v, memory_order_relaxed);
    }
    return v;
}

/* ── repack / de-repack ───────────────────────────────────────────────────────── */

size_t qwen_q8r_packed_bytes(int rows, int cols) {
    if (rows <= 0 || cols <= 0 || rows % 4 || cols % Q8_0_BLOCK_SIZE) return 0;
    return (size_t)(rows / 4) * (cols / Q8_0_BLOCK_SIZE) * sizeof(q8_0x4_block_t);
}

int qwen_q8r_repack(q8_0x4_block_t *dst, const q8_0_block_t *src, int rows, int cols) {
    if (!dst || !src || rows % 4 || cols % Q8_0_BLOCK_SIZE) return 0;
    const int nb = cols / Q8_0_BLOCK_SIZE;
    for (int r0 = 0; r0 < rows; r0 += 4) {
        for (int b = 0; b < nb; b++) {
            q8_0x4_block_t *o = dst++;
            for (int i = 0; i < 4; i++) o->d[i] = src[(size_t)(r0 + i) * nb + b].d;
            for (int i = 0; i < 16; i++) {
                const int row = i % 4, off = (i / 4) * 8;
                memcpy(&o->qs[i * 8], &src[(size_t)(r0 + row) * nb + b].qs[off], 8);
            }
        }
    }
    return 1;
}

int qwen_q8r_derepack(q8_0_block_t *dst, const q8_0x4_block_t *src, int rows, int cols) {
    if (!dst || !src || rows % 4 || cols % Q8_0_BLOCK_SIZE) return 0;
    const int nb = cols / Q8_0_BLOCK_SIZE;
    for (int r0 = 0; r0 < rows; r0 += 4) {
        for (int b = 0; b < nb; b++) {
            const q8_0x4_block_t *o = src++;
            for (int i = 0; i < 4; i++) dst[(size_t)(r0 + i) * nb + b].d = o->d[i];
            for (int i = 0; i < 16; i++) {
                const int row = i % 4, off = (i / 4) * 8;
                memcpy(&dst[(size_t)(r0 + row) * nb + b].qs[off], &o->qs[i * 8], 8);
            }
        }
    }
    return 1;
}

/* ── registry (same shape as the KleidiAI one, keyed by weight pointer) ────────── */
typedef struct {
    const void     *key;
    q8_0x4_block_t *packed;
    int             rows, cols;
    size_t          bytes;
} q8r_entry_t;

static q8r_entry_t *g_q8r;
static int          g_q8r_n, g_q8r_cap;
static size_t       g_q8r_bytes;
static pthread_mutex_t g_q8r_mx = PTHREAD_MUTEX_INITIALIZER;

static const q8r_entry_t *q8r_lookup(const void *key) {
    int n = atomic_load_explicit((_Atomic int *)&g_q8r_n, memory_order_acquire);
    for (int i = 0; i < n; i++)
        if (g_q8r[i].key == key) return &g_q8r[i];
    return NULL;
}

int qwen_q8r_register(const void *key, const q8_0_block_t *src, int rows, int cols) {
    if (!qwen_q8r_enabled() || !key || !src) return 0;
    size_t sz = qwen_q8r_packed_bytes(rows, cols);
    if (!sz) return 0;
    /* The byte count must be identical to the source. Asserting it here is cheap and
     * turns a layout mistake into a refusal instead of a silent misread. */
    size_t src_bytes = (size_t)rows * (cols / Q8_0_BLOCK_SIZE) * sizeof(q8_0_block_t);
    if (sz != src_bytes) {
        fprintf(stderr, "Q8repack: packed %zu != source %zu bytes - refusing\n", sz, src_bytes);
        return 0;
    }
    q8_0x4_block_t *p = (q8_0x4_block_t *)aligned_malloc(sz);
    if (!p) return 0;
    if (!qwen_q8r_repack(p, src, rows, cols)) { free(p); return 0; }

    pthread_mutex_lock(&g_q8r_mx);
    if (g_q8r_n == g_q8r_cap) {
        int cap = g_q8r_cap ? g_q8r_cap * 2 : 256;
        q8r_entry_t *n = (q8r_entry_t *)realloc(g_q8r, (size_t)cap * sizeof *n);
        if (!n) { pthread_mutex_unlock(&g_q8r_mx); free(p); return 0; }
        g_q8r = n; g_q8r_cap = cap;
    }
    g_q8r[g_q8r_n] = (q8r_entry_t){ key, p, rows, cols, sz };
    atomic_store_explicit((_Atomic int *)&g_q8r_n, g_q8r_n + 1, memory_order_release);
    g_q8r_bytes += sz;
    pthread_mutex_unlock(&g_q8r_mx);
    return 1;
}

void qwen_q8r_stats(int *n_packed, size_t *bytes) {
    if (n_packed) *n_packed = g_q8r_n;
    if (bytes)    *bytes    = g_q8r_bytes;
}

/* ── activation quantization: int8 + fp16 scale, every 32 values ──────────────── */

/* The scalar reference. Kept, and reachable with QWEN_Q8_SCALAR_ACT=1, for two
 * reasons: it is the definition of what the vector version must reproduce, and it is
 * the "before" side of the before/after measurement without rebuilding. */
static void q8r_quant_act_scalar(q8_0_block_t *dst, const float *x, int cols) {
    const int nb = cols / Q8_0_BLOCK_SIZE;
    for (int b = 0; b < nb; b++) {
        const float *s = x + (size_t)b * Q8_0_BLOCK_SIZE;
        float amax = 0.0f;
        for (int j = 0; j < Q8_0_BLOCK_SIZE; j++) {
            float a = fabsf(s[j]);
            if (a > amax) amax = a;
        }
        const float d = amax / 127.0f;
        const float id = d ? 1.0f / d : 0.0f;
        dst[b].d = qwen_f32_to_f16(d);
        for (int j = 0; j < Q8_0_BLOCK_SIZE; j++) {
            float v = s[j] * id;
            int q = (int)lrintf(v);
            if (q > 127) q = 127; else if (q < -128) q = -128;
            dst[b].qs[j] = (int8_t)q;
        }
    }
}

#if Q8R_NEON
/* NEON. Same arithmetic, same per-block scale, same rounding - only the loop is
 * vectorized. Nothing here touches the FORMAT: the scale stays one fp16 per 32
 * weights, which is the whole reason Q8_0 was chosen over a per-row int8.
 *
 * Rounding: `lrintf` above uses the default mode, round-half-to-even, and
 * `vcvtnq_s32_f32` is exactly that. `vcvtaq_s32_f32` (ties away from zero) would
 * differ on exact halves and silently make this a different quantizer.
 *
 * No clamp: id = 127/amax by construction, so |v * id| <= 127 for every element of
 * the block. The scalar version clamps defensively; here the bound is the definition. */
static void q8r_quant_act_neon(q8_0_block_t *dst, const float *x, int cols) {
    const int nb = cols / Q8_0_BLOCK_SIZE;
    for (int b = 0; b < nb; b++) {
        const float *s = x + (size_t)b * Q8_0_BLOCK_SIZE;
        float32x4_t v0 = vld1q_f32(s +  0), v1 = vld1q_f32(s +  4);
        float32x4_t v2 = vld1q_f32(s +  8), v3 = vld1q_f32(s + 12);
        float32x4_t v4 = vld1q_f32(s + 16), v5 = vld1q_f32(s + 20);
        float32x4_t v6 = vld1q_f32(s + 24), v7 = vld1q_f32(s + 28);

        float32x4_t m0 = vmaxq_f32(vabsq_f32(v0), vabsq_f32(v1));
        float32x4_t m1 = vmaxq_f32(vabsq_f32(v2), vabsq_f32(v3));
        float32x4_t m2 = vmaxq_f32(vabsq_f32(v4), vabsq_f32(v5));
        float32x4_t m3 = vmaxq_f32(vabsq_f32(v6), vabsq_f32(v7));
        float amax = vmaxvq_f32(vmaxq_f32(vmaxq_f32(m0, m1), vmaxq_f32(m2, m3)));

        const float d  = amax / 127.0f;
        const float id = d ? 1.0f / d : 0.0f;
        dst[b].d = qwen_f32_to_f16(d);

        int32x4_t q0 = vcvtnq_s32_f32(vmulq_n_f32(v0, id));
        int32x4_t q1 = vcvtnq_s32_f32(vmulq_n_f32(v1, id));
        int32x4_t q2 = vcvtnq_s32_f32(vmulq_n_f32(v2, id));
        int32x4_t q3 = vcvtnq_s32_f32(vmulq_n_f32(v3, id));
        int32x4_t q4 = vcvtnq_s32_f32(vmulq_n_f32(v4, id));
        int32x4_t q5 = vcvtnq_s32_f32(vmulq_n_f32(v5, id));
        int32x4_t q6 = vcvtnq_s32_f32(vmulq_n_f32(v6, id));
        int32x4_t q7 = vcvtnq_s32_f32(vmulq_n_f32(v7, id));

        int16x8_t h0 = vcombine_s16(vmovn_s32(q0), vmovn_s32(q1));
        int16x8_t h1 = vcombine_s16(vmovn_s32(q2), vmovn_s32(q3));
        int16x8_t h2 = vcombine_s16(vmovn_s32(q4), vmovn_s32(q5));
        int16x8_t h3 = vcombine_s16(vmovn_s32(q6), vmovn_s32(q7));

        vst1q_s8(dst[b].qs +  0, vcombine_s8(vmovn_s16(h0), vmovn_s16(h1)));
        vst1q_s8(dst[b].qs + 16, vcombine_s8(vmovn_s16(h2), vmovn_s16(h3)));
    }
}
#endif

static void q8r_quant_act(q8_0_block_t *dst, const float *x, int cols) {
#if Q8R_NEON
    static int scalar = -1;
    if (scalar < 0) { const char *e = getenv("QWEN_Q8_SCALAR_ACT"); scalar = (e && e[0] == '1'); }
    if (!scalar) { q8r_quant_act_neon(dst, x, cols); return; }
#endif
    q8r_quant_act_scalar(dst, x, cols);
}

/* ── the kernels ──────────────────────────────────────────────────────────────── */

typedef struct {
    const q8_0x4_block_t *W;
    const q8_0_block_t   *A;      /* quantized activations, B rows of nb blocks */
    float *Y;
    int rows, cols, B, nb;
} q8r_job_t;

/* GEMV, B = 1. dotprod, not a one-row GEMM: for a single activation vector SMMLA
 * would leave half of every 2x2 tile idle, which is exactly the asymmetry ggml
 * encodes by pairing an i8mm GEMM with a dotprod GEMV. */
static void q8r_gemv_rows(const q8r_job_t *j, int r0, int r1) {
#if Q8R_NEON && Q8R_DOT
    /* Faithful port of ggml's `ggml_gemv_q8_0_4x8_q8_0` NEON+dotprod inner loop
     * (llama.cpp ggml/src/ggml-cpu/arch/arm/repack.cpp). The first version of this
     * function was a simplification of it, and the simplification is what cost the
     * performance: it extracted the four int32 lanes to scalars, converted five fp16
     * scales one at a time, and accumulated in a scalar array. Per 32-block that is
     * 4 SIMD->GPR transfers + 5 scalar fp16 conversions + 12 scalar FP ops, against
     * the 5 vector instructions below - and SIMD->GPR moves are exactly the transfer
     * ARM cores are slowest at.
     *
     * The two things that make it work, and that the simplification threw away:
     *   - the four weight scales are CONTIGUOUS fp16 in the block (`d[4]`), so
     *     `vld1_f16` loads all four in one instruction and `vcvt_f32_f16` widens them
     *     in one more. No per-row conversion.
     *   - the float accumulator stays in a float32x4 register across every block of
     *     the row group; nothing returns to memory or to a general register until the
     *     final store. */
    const int nb = j->nb;
    for (int r = r0; r < r1; r += 4) {
        const q8_0x4_block_t *b_ptr = j->W + (size_t)(r / 4) * nb;
        const q8_0_block_t   *a_ptr = j->A;
        float32x4_t acc = vdupq_n_f32(0.0f);
        for (int b = 0; b < nb; b++, a_ptr++, b_ptr++) {
            int8x16x4_t b_low  = vld1q_s8_x4((const int8_t *)b_ptr->qs);
            int8x16x4_t b_high = vld1q_s8_x4((const int8_t *)b_ptr->qs + 64);
            float16x4_t bd     = vld1_f16((const __fp16 *)b_ptr->d);
            int8x8x4_t  a_ch   = vld1_s8_x4(a_ptr->qs);
            int8x16_t   a0 = vcombine_s8(a_ch.val[0], a_ch.val[0]);
            int8x16_t   a1 = vcombine_s8(a_ch.val[1], a_ch.val[1]);
            int8x16_t   a2 = vcombine_s8(a_ch.val[2], a_ch.val[2]);
            int8x16_t   a3 = vcombine_s8(a_ch.val[3], a_ch.val[3]);
            float16x4_t ad = vld1_dup_f16((const __fp16 *)&a_ptr->d);

            int32x4_t ret0 = vdupq_n_s32(0), ret1 = vdupq_n_s32(0);
            ret0 = vdotq_s32(ret0, b_low.val[0],  a0);
            ret1 = vdotq_s32(ret1, b_low.val[1],  a0);
            ret0 = vdotq_s32(ret0, b_low.val[2],  a1);
            ret1 = vdotq_s32(ret1, b_low.val[3],  a1);
            ret0 = vdotq_s32(ret0, b_high.val[0], a2);
            ret1 = vdotq_s32(ret1, b_high.val[1], a2);
            ret0 = vdotq_s32(ret0, b_high.val[2], a3);
            ret1 = vdotq_s32(ret1, b_high.val[3], a3);
            int32x4_t ret = vpaddq_s32(ret0, ret1);   /* -> [row0, row1, row2, row3] */

            acc = vfmaq_f32(acc, vcvtq_f32_s32(ret),
                            vmulq_f32(vcvt_f32_f16(ad), vcvt_f32_f16(bd)));
        }
        vst1q_f32(j->Y + r, acc);
    }
#else
    (void)j; (void)r0; (void)r1;
#endif
}

/* GEMM, B >= 2. Two activation rows and two weight rows per SMMLA. */
#if Q8R_NEON && Q8R_I8MM
static void q8r_gemm_rows(const q8r_job_t *j, int r0, int r1) {
    const int nb = j->nb, B = j->B, rows = j->rows;
    for (int bb = 0; bb < B; bb += 2) {
        const int b1 = (bb + 1 < B) ? bb + 1 : bb;      /* odd B: duplicate, discard */
        for (int r = r0; r < r1; r += 4) {
            const q8_0x4_block_t *w = j->W + (size_t)(r / 4) * nb;
            float o[4][2] = { { 0, 0 }, { 0, 0 }, { 0, 0 }, { 0, 0 } };
            for (int b = 0; b < nb; b++) {
                const q8_0_block_t *a0 = &j->A[(size_t)bb * nb + b];
                const q8_0_block_t *a1 = &j->A[(size_t)b1 * nb + b];
                const int8_t *qs = w[b].qs;
                int32x4_t acc01 = vdupq_n_s32(0), acc23 = vdupq_n_s32(0);
                for (int g = 0; g < 4; g++) {
                    /* A: 2 activation rows x 8 values.  B: 2 weight rows x 8 values. */
                    int8x16_t A = vcombine_s8(vld1_s8(a0->qs + g * 8), vld1_s8(a1->qs + g * 8));
                    acc01 = vmmlaq_s32(acc01, A, vld1q_s8(qs + g * 32));
                    acc23 = vmmlaq_s32(acc23, A, vld1q_s8(qs + g * 32 + 16));
                }
                /* vmmla lays the 2x2 tile out as [a0.w0, a0.w1, a1.w0, a1.w1] */
                const float d0 = qwen_f16_to_f32(a0->d), d1 = qwen_f16_to_f32(a1->d);
                const float w0 = qwen_f16_to_f32(w[b].d[0]), w1 = qwen_f16_to_f32(w[b].d[1]);
                const float w2 = qwen_f16_to_f32(w[b].d[2]), w3 = qwen_f16_to_f32(w[b].d[3]);
                o[0][0] += (float)vgetq_lane_s32(acc01, 0) * w0 * d0;
                o[1][0] += (float)vgetq_lane_s32(acc01, 1) * w1 * d0;
                o[0][1] += (float)vgetq_lane_s32(acc01, 2) * w0 * d1;
                o[1][1] += (float)vgetq_lane_s32(acc01, 3) * w1 * d1;
                o[2][0] += (float)vgetq_lane_s32(acc23, 0) * w2 * d0;
                o[3][0] += (float)vgetq_lane_s32(acc23, 1) * w3 * d0;
                o[2][1] += (float)vgetq_lane_s32(acc23, 2) * w2 * d1;
                o[3][1] += (float)vgetq_lane_s32(acc23, 3) * w3 * d1;
            }
            for (int i = 0; i < 4; i++) {
                j->Y[(size_t)(r + i) * B + bb] = o[i][0];
                if (b1 != bb) j->Y[(size_t)(r + i) * B + b1] = o[i][1];
            }
        }
    }
    (void)rows;
}
#endif

static void q8r_task(size_t tid, size_t nt, void *ctx) {
    q8r_job_t *j = (q8r_job_t *)ctx;
    int nblk = j->rows / 4;
    int t0 = (int)((size_t)nblk * tid / nt), t1 = (int)((size_t)nblk * (tid + 1) / nt);
    int r0 = t0 * 4, r1 = t1 * 4;
    if (r0 >= r1) return;
#if Q8R_NEON && Q8R_I8MM
    if (j->B > 1) { q8r_gemm_rows(j, r0, r1); return; }
#endif
    q8r_gemv_rows(j, r0, r1);
}

/* Per-thread scratch for the quantized activations, same mould as the rest of the
 * engine: TLS pointer, high-water growth, owned by the calling thread. */
static __thread q8_0_block_t *g_q8r_act;
static __thread size_t        g_q8r_act_cap;

int qwen_q8r_matmul(float *Y, const void *key, const float *X, int rows, int cols, int B) {
    if (!qwen_q8r_enabled() || B < 1) return 0;
    const q8r_entry_t *e = q8r_lookup(key);
    if (!e || e->rows != rows || e->cols != cols) return 0;
#if !(Q8R_NEON && Q8R_DOT)
    return 0;                       /* no GEMV means no path at all */
#else
    int has_dot = 0, has_i8mm = 0;
    q8r_cpu(&has_dot, &has_i8mm);
    if (B > 1 && !has_i8mm) return 0;      /* declared: no i8mm, no batched path */
    if (!has_dot) return 0;

    const int nb = cols / Q8_0_BLOCK_SIZE;
    size_t need = (size_t)B * nb * sizeof(q8_0_block_t);
    if (need > g_q8r_act_cap) {
        void *p = NULL;
        if (posix_memalign(&p, 64, need) != 0) return 0;
        free(g_q8r_act);
        g_q8r_act = (q8_0_block_t *)p;
        g_q8r_act_cap = need;
    }
    if (B == 1) {
        q8r_quant_act(g_q8r_act, X, cols);
    } else {
        /* X is [cols, B]; the kernel wants each batch row contiguous. */
        float *tmp = (float *)malloc((size_t)cols * sizeof(float));
        if (!tmp) return 0;
        for (int b = 0; b < B; b++) {
            for (int c = 0; c < cols; c++) tmp[c] = X[(size_t)c * B + b];
            q8r_quant_act(g_q8r_act + (size_t)b * nb, tmp, cols);
        }
        free(tmp);
    }

    q8r_job_t job = { e->packed, g_q8r_act, Y, rows, cols, B, nb };
    size_t nt = (size_t)qwen_get_threads();
    if (nt < 1) nt = 1;
    if (rows < (int)nt * 16) nt = 1;
    if (nt == 1) q8r_task(0, 1, &job);
    else         qwen_parallel(nt, q8r_task, &job);
    return 1;
#endif
}


/* ── the vertical slice: one real Q8_0 matrix, end to end ─────────────────────────
 *
 * Two questions, answered separately because they fail differently:
 *   1. STRUCTURAL - is the repack bit-preserving? De-repack and memcmp. A shuffle
 *      that is "numerically close" has put a scale or a quant on the wrong row, and
 *      that produces confident, plausible audio rather than an obvious break.
 *   2. NUMERICAL  - does the kernel compute the same product as a scalar reference
 *      built from the SOURCE blocks? Same quantized values on both sides, so any
 *      difference is the kernel's, not the format's. fp32 accumulation order differs,
 *      so the bar is a relative error near the fp32 epsilon, not zero.
 */
#include "ingot/gguf.h"
#include "ingot/dtype.h"

int qwen_q8r_selftest(void *out, const char *gguf_path, const char *tensor_name) {
    FILE *f = out ? (FILE *)out : stderr;
    int has_dot = 0, has_i8mm = 0;
    q8r_cpu(&has_dot, &has_i8mm);
    fprintf(f, "Q8repack self-test\n");
    fprintf(f, "  build: NEON=%d dotprod=%d i8mm=%d   cpu: dotprod=%d i8mm=%d\n",
            Q8R_NEON, Q8R_DOT, Q8R_I8MM, has_dot, has_i8mm);
    if (!qwen_q8r_supported()) {
        fprintf(f, "  -> not supported on this build/CPU: nothing to test\n");
        return 1;
    }

    ingot_gguf *g = NULL;
    char err[256] = "";
    if (ingot_gguf_open(&g, gguf_path, err, sizeof err) != 0) {
        fprintf(f, "  cannot open %s: %s\n", gguf_path, err);
        return 1;
    }
    const char *name = tensor_name ? tensor_name : "blk.0.attn_q.weight";
    const ingot_tensor *t = ingot_gguf_find(g, name);
    if (!t) { fprintf(f, "  tensor %s not found\n", name); ingot_gguf_close(g); return 1; }
    if (t->type != INGOT_TYPE_Q8_0) {
        fprintf(f, "  %s is %s, not Q8_0 - wrong artifact\n", name, ingot_type_name(t->type));
        ingot_gguf_close(g); return 1;
    }
    uint64_t shp[INGOT_MAX_RANK] = { 0 };
    ingot_gguf_shape_row_major(t, shp);
    const int rows = (int)shp[0], cols = (int)shp[1];
    const q8_0_block_t *src = (const q8_0_block_t *)ingot_gguf_data(g, t);
    if (!src || rows % 4 || cols % Q8_0_BLOCK_SIZE) {
        fprintf(f, "  %s [%d,%d] not usable (rows%%4=%d cols%%32=%d)\n",
                name, rows, cols, rows % 4, cols % Q8_0_BLOCK_SIZE);
        ingot_gguf_close(g); return 1;
    }
    const int nb = cols / Q8_0_BLOCK_SIZE;
    fprintf(f, "  matrix: %s [%d, %d] = %d blocks/row, %d Q8_0 blocks\n",
            name, rows, cols, nb, rows * nb);

    /* 1. structural */
    size_t packed_bytes = qwen_q8r_packed_bytes(rows, cols);
    size_t src_bytes    = (size_t)rows * nb * sizeof(q8_0_block_t);
    q8_0x4_block_t *pk  = (q8_0x4_block_t *)malloc(packed_bytes);
    q8_0_block_t   *back = (q8_0_block_t *)malloc(src_bytes);
    if (!pk || !back) { free(pk); free(back); ingot_gguf_close(g); return 1; }
    qwen_q8r_repack(pk, src, rows, cols);
    qwen_q8r_derepack(back, pk, rows, cols);
    int bytes_same = (packed_bytes == src_bytes);
    int exact = (memcmp(back, src, src_bytes) == 0);
    fprintf(f, "  bytes: source %zu, packed %zu -> %s\n", src_bytes, packed_bytes,
            bytes_same ? "IDENTICAL" : "DIFFERENT  <-- not a pure shuffle");
    fprintf(f, "  de-repack == source: %s\n",
            exact ? "BIT-IDENTICAL" : "MISMATCH  <-- the repack loses or moves data");
    if (!exact) {
        size_t i = 0;
        const uint8_t *a = (const uint8_t *)back, *b = (const uint8_t *)src;
        while (i < src_bytes && a[i] == b[i]) i++;
        fprintf(f, "     first differing byte at offset %zu (block %zu)\n",
                i, i / sizeof(q8_0_block_t));
    }

    /* 1.bis the vectorized activation quantizer must reproduce the scalar one EXACTLY.
     * It is the same format and the same rounding mode; anything else would be a
     * different quantizer wearing the same name. */
#if Q8R_NEON
    {
        int nbq = cols / Q8_0_BLOCK_SIZE;
        float *xt = (float *)malloc((size_t)cols * sizeof(float));
        q8_0_block_t *as = (q8_0_block_t *)malloc((size_t)nbq * sizeof(q8_0_block_t));
        q8_0_block_t *an = (q8_0_block_t *)malloc((size_t)nbq * sizeof(q8_0_block_t));
        if (xt && as && an) {
            uint32_t st2 = 24680u;
            for (int i = 0; i < cols; i++) {
                st2 = st2 * 1664525u + 1013904223u;
                xt[i] = (((float)((st2 >> 8) & 0xFFFF) / 32768.0f) - 1.0f) * (1 + (i % 5));
            }
            q8r_quant_act_scalar(as, xt, cols);
            q8r_quant_act_neon(an, xt, cols);
            int same = (memcmp(as, an, (size_t)nbq * sizeof(q8_0_block_t)) == 0);
            fprintf(f, "  activation quantizer NEON == scalar: %s\n",
                    same ? "BIT-IDENTICAL" : "MISMATCH  <-- not the same quantizer");
        }
        free(xt); free(as); free(an);
    }
#endif

    /* 2. numerical, against a scalar reference over the SOURCE blocks */
    float *x  = (float *)malloc((size_t)cols * sizeof(float));
    float *y  = (float *)malloc((size_t)rows * sizeof(float));
    float *yr = (float *)malloc((size_t)rows * sizeof(float));
    q8_0_block_t *act = (q8_0_block_t *)malloc((size_t)nb * sizeof(q8_0_block_t));
    if (!x || !y || !yr || !act) { free(x); free(y); free(yr); free(act);
                                   free(pk); free(back); ingot_gguf_close(g); return 1; }
    uint32_t st = 987654321u;
    for (int i = 0; i < cols; i++) {
        st = st * 1664525u + 1013904223u;
        x[i] = ((float)((st >> 8) & 0xFFFF) / 32768.0f) - 1.0f;
    }
    q8r_quant_act(act, x, cols);
    for (int r = 0; r < rows; r++) {
        double acc = 0.0;
        for (int b = 0; b < nb; b++) {
            const q8_0_block_t *w = &src[(size_t)r * nb + b];
            int32_t s = 0;
            for (int j = 0; j < Q8_0_BLOCK_SIZE; j++) s += (int32_t)w->qs[j] * (int32_t)act[b].qs[j];
            acc += (double)s * (double)qwen_f16_to_f32(w->d) * (double)qwen_f16_to_f32(act[b].d);
        }
        yr[r] = (float)acc;
    }

    /* run the real path through the registry, exactly as inference would */
    int ok_gemv = 0, ok_gemm = 0;
    double worst_v = 0.0, worst_m = 0.0, ref_rms = 0.0;
    for (int r = 0; r < rows; r++) ref_rms += (double)yr[r] * (double)yr[r];
    ref_rms = sqrt(ref_rms / rows);

    if (qwen_q8r_register(pk /* any unique key */, src, rows, cols)) {
        ok_gemv = qwen_q8r_matmul(y, pk, x, rows, cols, 1);
        if (ok_gemv)
            for (int r = 0; r < rows; r++) {
                double d = fabs((double)y[r] - (double)yr[r]);
                if (d > worst_v) worst_v = d;
            }
        if (has_i8mm) {
            const int B = 4;
            float *Xb = (float *)malloc((size_t)cols * B * sizeof(float));
            float *Yb = (float *)malloc((size_t)rows * B * sizeof(float));
            if (Xb && Yb) {
                for (int c = 0; c < cols; c++)
                    for (int b = 0; b < B; b++) Xb[(size_t)c * B + b] = x[c];
                ok_gemm = qwen_q8r_matmul(Yb, pk, Xb, rows, cols, B);
                if (ok_gemm)
                    for (int r = 0; r < rows; r++)
                        for (int b = 0; b < B; b++) {
                            double d = fabs((double)Yb[(size_t)r * B + b] - (double)yr[r]);
                            if (d > worst_m) worst_m = d;
                        }
            }
            free(Xb); free(Yb);
        }
    }

    fprintf(f, "  GEMV (B=1, dotprod): %s", ok_gemv ? "" : "DID NOT RUN\n");
    if (ok_gemv) fprintf(f, "max|diff| %.3g, relative %.3g%%\n", worst_v, 100.0 * worst_v / ref_rms);
    if (has_i8mm) {
        fprintf(f, "  GEMM (B=4, i8mm):    %s", ok_gemm ? "" : "DID NOT RUN\n");
        if (ok_gemm) fprintf(f, "max|diff| %.3g, relative %.3g%%\n", worst_m, 100.0 * worst_m / ref_rms);
    } else {
        fprintf(f, "  GEMM (B>1, i8mm):    skipped - this CPU has no i8mm\n");
    }

    int pass = bytes_same && exact && ok_gemv &&
               (100.0 * worst_v / ref_rms) < 0.01 &&
               (!has_i8mm || (ok_gemm && (100.0 * worst_m / ref_rms) < 0.01));
    fprintf(f, "  => %s\n", pass ? "PASS" : "FAIL");

    free(x); free(y); free(yr); free(act); free(pk); free(back);
    ingot_gguf_close(g);
    return pass ? 0 : 1;
}


/* ── microbenchmark on the Code Predictor's real shapes ───────────────────────────
 * The CP is where B=1 hurts: five layers, fifteen lm_heads, all of it fifteen times
 * per frame. Timing the kernel on synthetic data of the RIGHT shapes isolates the
 * kernel from everything else in the pipeline, which is the only way to tell a slow
 * kernel from a slow pipeline. Compared against our own INT8 GEMV on the same shape,
 * because that is the number Q8_0 has to approach. */
static double q8r_now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

int qwen_q8r_bench(void *out) {
    FILE *f = out ? (FILE *)out : stderr;
    if (!qwen_q8r_supported()) { fprintf(f, "Q8repack: not supported here\n"); return 1; }
    struct { const char *what; int rows, cols; int per_frame; } shapes[] = {
        { "CP attn_q      ", 2048, 1024,  5 * 15 },
        { "CP attn_k/v    ", 1024, 1024, 10 * 15 },
        { "CP attn_output ", 1024, 2048,  5 * 15 },
        { "CP ffn gate+up ", 6144, 1024,  5 * 15 },
        { "CP ffn_down    ", 1024, 3072,  5 * 15 },
        { "CP lm_head     ", 2048, 1024, 15 },   /* one head per pass */
        { "Talker attn_q  ", 2048, 2048,  0 },
        { "Talker gate+up ", 12288, 2048, 0 },
    };
    const int ns = (int)(sizeof shapes / sizeof shapes[0]);
    void *keep[16] = { 0 };
    fprintf(f, "Q8_0 GEMV microbench (B=1), threads=%d\n", qwen_get_threads());
    fprintf(f, "  %-16s %11s %11s %9s %10s\n", "shape", "q8_0 us", "int8 us", "ratio", "per frame");
    double tot_q8 = 0, tot_i8 = 0;
    for (int i = 0; i < ns; i++) {
        const int rows = shapes[i].rows, cols = shapes[i].cols;
        const int nb = cols / Q8_0_BLOCK_SIZE;
        q8_0_block_t *W = (q8_0_block_t *)malloc((size_t)rows * nb * sizeof(q8_0_block_t));
        int8_t *Wi = (int8_t *)aligned_malloc((size_t)rows * cols);
        float  *sc = (float *)malloc((size_t)rows * sizeof(float));
        float  *x  = (float *)malloc((size_t)cols * sizeof(float));
        float  *y  = (float *)malloc((size_t)rows * sizeof(float));
        if (!W || !Wi || !sc || !x || !y) { free(W); free(Wi); free(sc); free(x); free(y); continue; }
        uint32_t st = 1u + (uint32_t)i;
        for (size_t k = 0; k < (size_t)rows * nb; k++) {
            W[k].d = qwen_f32_to_f16(0.01f);
            for (int j = 0; j < Q8_0_BLOCK_SIZE; j++) { st = st * 1664525u + 1013904223u; W[k].qs[j] = (int8_t)(st >> 24); }
        }
        for (size_t k = 0; k < (size_t)rows * cols; k++) { st = st * 1664525u + 1013904223u; Wi[k] = (int8_t)(st >> 24); }
        for (int r = 0; r < rows; r++) sc[r] = 0.01f;
        for (int c = 0; c < cols; c++) x[c] = 0.5f - (float)(c % 7) / 7.0f;

        /* NOT freed until the end: the registry is keyed by pointer, and malloc reuses
         * freed addresses - a later shape then matched an earlier, stale entry and the
         * call silently declined. The engine never hits this (its keys are live model
         * weights), but a benchmark that frees as it goes does. */
        keep[i] = W;
        if (!qwen_q8r_register(W, W, rows, cols)) { fprintf(f, "  %-16s register failed\n", shapes[i].what); goto next; }
        /* Warm BOTH thoroughly first: the first shape measured used to pay the page
         * faults of its own fresh allocation while the second contender ran on pages
         * already touched, which flattered whichever went second. Then time each twice
         * in opposite order and keep the minimum - the floor is the kernel, the excess
         * is the machine. */
        for (int w = 0; w < 20; w++) {
            qwen_q8r_matmul(y, W, x, rows, cols, 1);
            qwen_matvec_int8(y, Wi, sc, x, rows, cols);
        }
        const int REP = 50;
        double q8us = 1e18, i8us = 1e18;
        for (int pass = 0; pass < 2; pass++) {
            double a0 = q8r_now_ms();
            for (int r = 0; r < REP; r++) qwen_q8r_matmul(y, W, x, rows, cols, 1);
            double a1 = q8r_now_ms();
            for (int r = 0; r < REP; r++) qwen_matvec_int8(y, Wi, sc, x, rows, cols);
            double a2 = q8r_now_ms();
            double q = (a1 - a0) * 1000.0 / REP, i = (a2 - a1) * 1000.0 / REP;
            if (q < q8us) q8us = q;
            if (i < i8us) i8us = i;
            /* second pass: swap the order so neither contender is always the warm one */
            a0 = q8r_now_ms();
            for (int r = 0; r < REP; r++) qwen_matvec_int8(y, Wi, sc, x, rows, cols);
            a1 = q8r_now_ms();
            for (int r = 0; r < REP; r++) qwen_q8r_matmul(y, W, x, rows, cols, 1);
            a2 = q8r_now_ms();
            i = (a1 - a0) * 1000.0 / REP; q = (a2 - a1) * 1000.0 / REP;
            if (q < q8us) q8us = q;
            if (i < i8us) i8us = i;
        }
        fprintf(f, "  %-16s %11.1f %11.1f %8.2fx", shapes[i].what, q8us, i8us, q8us / i8us);
        if (shapes[i].per_frame) {
            double dq = q8us * shapes[i].per_frame / 1000.0, di = i8us * shapes[i].per_frame / 1000.0;
            fprintf(f, "  %6.2f vs %.2f ms/f", dq, di);
            tot_q8 += dq; tot_i8 += di;
        }
        fprintf(f, "\n");
    next:
        free(Wi); free(sc); free(x); free(y);
    }
    for (int i = 0; i < ns; i++) free(keep[i]);
    fprintf(f, "  CP total (15 passes/frame):  q8_0 %.2f ms/f  vs  int8 %.2f ms/f\n", tot_q8, tot_i8);

    /* The activation quantizer on its own. It is the part of the Q8_0 GEMV that has no
     * counterpart in our int8 path (which quantizes the activation once per vector,
     * not once per 32), so it is the first place to look for the gap. */
    fprintf(f, "\n  activation quantization alone (one call, us):\n");
    fprintf(f, "  %-10s %12s %12s %8s\n", "cols", "scalar", "NEON", "speedup");
    for (int ci = 0; ci < 4; ci++) {
        const int cols = (int[]){ 1024, 2048, 3072, 6144 }[ci];
        const int nbq = cols / Q8_0_BLOCK_SIZE;
        float *xt = (float *)malloc((size_t)cols * sizeof(float));
        q8_0_block_t *ab = (q8_0_block_t *)malloc((size_t)nbq * sizeof(q8_0_block_t));
        if (!xt || !ab) { free(xt); free(ab); continue; }
        for (int i = 0; i < cols; i++) xt[i] = 0.37f - (float)(i % 11) / 11.0f;
        for (int w = 0; w < 200; w++) { q8r_quant_act_scalar(ab, xt, cols); }
        const int R = 2000;
        double t0 = q8r_now_ms();
        for (int r = 0; r < R; r++) q8r_quant_act_scalar(ab, xt, cols);
        double t1 = q8r_now_ms();
#if Q8R_NEON
        for (int w = 0; w < 200; w++) q8r_quant_act_neon(ab, xt, cols);
        double t2 = q8r_now_ms();
        for (int r = 0; r < R; r++) q8r_quant_act_neon(ab, xt, cols);
        double t3 = q8r_now_ms();
        double sc = (t1 - t0) * 1000.0 / R, ne = (t3 - t2) * 1000.0 / R;
        fprintf(f, "  %-10d %11.2f %12.2f %7.2fx\n", cols, sc, ne, sc / ne);
#else
        fprintf(f, "  %-10d %11.2f %12s %8s\n", cols, (t1 - t0) * 1000.0 / R, "n/a", "n/a");
#endif
        free(xt); free(ab);
    }
    return 0;
}
