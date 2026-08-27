/*
 * qwen_tts_talker.c - Talker LLM forward pass with KV cache
 * Implements Qwen3-based autoregressive transformer with:
 * - GQA (Grouped Query Attention) with 2:1 ratio
 * - Per-head Q/K RMSNorm
 * - NeoX split-half RoPE (NOT interleaved)
 * - SwiGLU MLP
 */

#include "qwen_tts.h"
#include "qwen_tts_kernels.h"
#include "ingot/safetensors.h"
#include "qwen_tts_batch.h"
#include "qwen_tts_kleidi.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdatomic.h>
#if defined(__APPLE__) || defined(__unix__) || defined(__linux__)
#include <sys/mman.h>      /* madvise: hand the mmapped layer projections back (S3) */
#include <unistd.h>        /* sysconf(_SC_PAGESIZE) */
#define QWEN_HAVE_MADVISE 1
#endif

/* aligned_malloc/aligned_calloc now in qwen_tts_kernels.h */

/* ── Activation-map capture (QWEN_ACT_MAP=path) ──────────────────────────────
 * Read-only diagnostic for the emotion/instruct representation analysis: over a
 * whole generation, accumulate the MEAN of the Talker residual stream after EACH
 * layer (plus the final-norm hidden) — a [num_layers+1 × hidden] fingerprint of
 * where, layer by layer, an instruct/emotion shifts the activations. Contrast two
 * runs (neutral-instruct vs emotion-instruct) with tests/act_map_diff.py. Off
 * unless QWEN_ACT_MAP is set; zero cost otherwise. */
static const char *g_actmap_path   = NULL;
static double    **g_actmap_acc    = NULL;   /* [num_layers+1][hidden] */
static int         g_actmap_layers = 0;      /* num_layers + 1 (final hidden) */
static int         g_actmap_dim    = 0;
static long        g_actmap_frames = 0;
static int         g_actmap_probed = 0;

static void actmap_dump(void) {
    if (!g_actmap_path || g_actmap_frames <= 0 || !g_actmap_acc) return;
    FILE *f = fopen(g_actmap_path, "wb");
    if (!f) return;
    uint32_t magic = 0x504D4151;             /* 'QAMP' */
    int32_t L = g_actmap_layers, D = g_actmap_dim;
    fwrite(&magic, 4, 1, f); fwrite(&L, 4, 1, f); fwrite(&D, 4, 1, f);
    for (int l = 0; l < g_actmap_layers; l++)
        for (int i = 0; i < g_actmap_dim; i++) {
            float v = (float)(g_actmap_acc[l][i] / (double)g_actmap_frames);
            fwrite(&v, 4, 1, f);
        }
    fclose(f);
    fprintf(stderr, "  [QWEN_ACT_MAP] wrote %d layers x %d dim (%ld frames) -> %s\n",
            g_actmap_layers, g_actmap_dim, g_actmap_frames, g_actmap_path);
}

static void actmap_init(int num_layers, int h) {
    if (g_actmap_probed) return;
    g_actmap_probed = 1;
    const char *p = getenv("QWEN_ACT_MAP");
    if (!p || !*p) return;
    g_actmap_path   = p;
    g_actmap_layers = num_layers + 1;        /* +1 = final-norm hidden */
    g_actmap_dim    = h;
    g_actmap_acc    = (double **)calloc(g_actmap_layers, sizeof(double *));
    if (!g_actmap_acc) { g_actmap_path = NULL; return; }
    for (int l = 0; l < g_actmap_layers; l++) g_actmap_acc[l] = (double *)calloc(h, sizeof(double));
    atexit(actmap_dump);
}

static inline void actmap_accum(int layer, const float *x, int h) {
    if (!g_actmap_path || !g_actmap_acc) return;
    double *a = g_actmap_acc[layer];
    for (int i = 0; i < h; i++) a[i] += (double)x[i];
}

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif
#ifdef __AVX2__
#include <immintrin.h>
#endif

#ifdef USE_BLAS
#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif
#endif

/* ========================================================================
 * bf16 helpers
 * ======================================================================== */

static inline float bf16_to_f32(uint16_t bf) {
    uint32_t bits = (uint32_t)bf << 16;
    float val; memcpy(&val, &bits, sizeof(float));
    return val;
}

static inline uint16_t f32_to_bf16(float val) {
    uint32_t bits;
    memcpy(&bits, &val, sizeof(float));
    return (uint16_t)(bits >> 16);
}

static uint16_t *get_bf16(void *ms, const char *name) {
    const ingot_st_tensor *t = ingot_st_find((ingot_st *)ms, name);
    if (!t || t->dtype != INGOT_DT_BF16) return NULL;
    return (uint16_t *)(uintptr_t)ingot_st_data((ingot_st *)ms, t);
}

/* Same signature and same ownership as the old safetensors_get_f32: the
 * caller frees. ingot converts into a buffer that is ours from the start. */
static float *get_f32(void *ms, const char *name) {
    const ingot_st_tensor *t = ingot_st_find((ingot_st *)ms, name);
    if (!t) return NULL;
    float *out = malloc((size_t)t->nelem * sizeof(float));
    if (!out || ingot_st_to_f32((ingot_st *)ms, t, out) != 0) {
        free(out);
        return NULL;
    }
    return out;
}

/* Convert f32 vector to bf16 (NEON-vectorized) */
static void f32_to_bf16_vec(uint16_t *dst, const float *src, int64_t n) {
#ifdef __ARM_NEON
    int64_t i = 0;
    for (; i + 7 < n; i += 8) {
        /* Load 8 f32 values, extract upper 16 bits (bf16 truncation) */
        uint32x4_t u0 = vreinterpretq_u32_f32(vld1q_f32(src + i));
        uint32x4_t u1 = vreinterpretq_u32_f32(vld1q_f32(src + i + 4));
        uint16x4_t lo = vshrn_n_u32(u0, 16);
        uint16x4_t hi = vshrn_n_u32(u1, 16);
        vst1q_u16(dst + i, vcombine_u16(lo, hi));
    }
    for (; i < n; i++) dst[i] = f32_to_bf16(src[i]);
#elif defined(__AVX512F__)
    int64_t i = 0;
    for (; i + 15 < n; i += 16) {
        /* Truncate 16 f32 to their top 16 bits (bf16, same semantics as the
         * AVX2/NEON paths), narrow 16×u32 -> 16×u16 with vpmovdw. */
        __m512i u = _mm512_srli_epi32(_mm512_castps_si512(_mm512_loadu_ps(src + i)), 16);
        _mm256_storeu_si256((__m256i *)(dst + i), _mm512_cvtepi32_epi16(u));
    }
    for (; i < n; i++) dst[i] = f32_to_bf16(src[i]);
#elif defined(__AVX2__)
    int64_t i = 0;
    for (; i + 7 < n; i += 8) {
        /* Truncate each f32 to its top 16 bits (bf16), then pack 8×u32 -> 8×u16 */
        __m256i u = _mm256_srli_epi32(_mm256_castps_si256(_mm256_loadu_ps(src + i)), 16);
        __m128i packed = _mm_packus_epi32(_mm256_castsi256_si128(u),
                                          _mm256_extracti128_si256(u, 1));
        _mm_storeu_si128((__m128i *)(dst + i), packed);
    }
    for (; i < n; i++) dst[i] = f32_to_bf16(src[i]);
#else
    for (int64_t i = 0; i < n; i++) dst[i] = f32_to_bf16(src[i]);
#endif
}

/* Convert bf16 matrix to f32 (NEON-vectorized, multi-threaded) */
static void bf16_to_f32_matrix(float *dst, const uint16_t *src, int64_t n) {
#ifdef __ARM_NEON
    int64_t i = 0;
    for (; i + 7 < n; i += 8) {
        uint16x8_t v = vld1q_u16(src + i);
        uint32x4_t lo = vshll_n_u16(vget_low_u16(v), 16);
        uint32x4_t hi = vshll_n_u16(vget_high_u16(v), 16);
        vst1q_f32(dst + i,     vreinterpretq_f32_u32(lo));
        vst1q_f32(dst + i + 4, vreinterpretq_f32_u32(hi));
    }
    for (; i < n; i++) dst[i] = bf16_to_f32(src[i]);
#elif defined(__AVX512F__)
    int64_t i = 0;
    for (; i + 15 < n; i += 16) {
        __m256i v = _mm256_loadu_si256((const __m256i *)(src + i));
        __m512i w = _mm512_slli_epi32(_mm512_cvtepu16_epi32(v), 16);
        _mm512_storeu_ps(dst + i, _mm512_castsi512_ps(w));
    }
    for (; i < n; i++) dst[i] = bf16_to_f32(src[i]);
#elif defined(__AVX2__)
    int64_t i = 0;
    for (; i + 7 < n; i += 8) {
        __m128i v = _mm_loadu_si128((const __m128i *)(src + i));
        __m256i w = _mm256_slli_epi32(_mm256_cvtepu16_epi32(v), 16);
        _mm256_storeu_ps(dst + i, _mm256_castsi256_ps(w));
    }
    for (; i < n; i++) dst[i] = bf16_to_f32(src[i]);
#else
    for (int64_t i = 0; i < n; i++) dst[i] = bf16_to_f32(src[i]);
#endif
}

/* ========================================================================
 * S3 — the prefill reads the QUANTIZED weights (QWEN_PREFILL_QUANT=1)
 * ========================================================================
 *
 * WHY THIS EXISTS. On a quantized server the decode is int8/q4/q6 but the PREFILL
 * is not: with BLAS (all our builds) it runs the use_matmat=0 path, which for EVERY
 * layer of EVERY request expands the bf16 weights into the per-layer f32 scratch
 * (bf16_to_f32_matrix). So the bf16 copies are not a load-time residue that a
 * madvise could drop — they are a live reader on the request path, and dropping
 * them would only buy a page-fault storm on the next request. Measured (PLAN 0.nonies
 * S3): 1.7B --int8 holds the mmapped bf16 (up to 3.6 GB) AND
 * the heap-fused gate_up bf16 (~1.4 GB) AND the quantized copies (~1.8 GB) at once —
 * which is what the OOM kill on this 16 GB Mac actually was.
 *
 * WHAT CHANGES. Only the SOURCE of the f32 scratch: same buffers, same BLAS sgemm,
 * same numerics downstream. Dequantizing int8 reads HALF the bytes of the bf16
 * expansion (and q4/q6 fewer still), so this should be neutral-to-faster; if it is
 * slower, the dequant is written wrong, not the idea.
 *
 * WHAT IT COSTS. The prompt KV is then computed from the quantized weights instead
 * of the bf16 ones. That is a REAL change of the audio, not an internal detail —
 * it makes prefill consistent with a decode that is already quantized, but it has
 * to be listened to, not assumed. Hence the env switch, default OFF.
 *
 * Dispatch order mirrors the decode step exactly: q6 > q4 > int8 > bf16.
 * Formats (kept in sync with qwen_tts_kernels.c, which owns the quantizers):
 *   int8 : w = q * scale[row]                       (per-row absmax)
 *   q4_0 : w = (nibble - 8) * f16(scale), qs[i] low nibble = col 2i, high = 2i+1
 *   q6_0 : delegated to qwen_dequant_row_q6_0 (the parity gate's own reader) */

static void tk_deq_int8_matrix(float *restrict dst, const int8_t *restrict q,
                               const float *restrict scale, int rows, int cols) {
    for (int r = 0; r < rows; r++) {
        const int8_t *qr = q + (size_t)r * cols;
        float *dr = dst + (size_t)r * cols;
        const float s = scale[r];
        for (int i = 0; i < cols; i++) dr[i] = (float)qr[i] * s;
    }
}

static void tk_deq_q4_matrix(float *restrict dst, const q4_0_block_t *W,
                             int rows, int cols) {
    const int bpr = cols / Q4_0_BLOCK_SIZE;
    for (int r = 0; r < rows; r++) {
        const q4_0_block_t *row = W + (size_t)r * bpr;
        float *dr = dst + (size_t)r * cols;
        for (int b = 0; b < bpr; b++) {
            const float s = qwen_f16_to_f32(row[b].scale_f16);
            const uint8_t *qs = row[b].qs;
            float *db = dr + (size_t)b * Q4_0_BLOCK_SIZE;
            for (int i = 0; i < 16; i++) {
                db[2 * i]     = (float)((int)(qs[i] & 0x0F) - 8) * s;
                db[2 * i + 1] = (float)((int)(qs[i] >> 4)   - 8) * s;
            }
        }
    }
}

static void tk_deq_q6_matrix(float *restrict dst, const q6_0_block_t *W,
                             int rows, int cols) {
    const int bpr = cols / Q6_0_BLOCK_SIZE;
    for (int r = 0; r < rows; r++)
        qwen_dequant_row_q6_0(dst + (size_t)r * cols, W + (size_t)r * bpr, cols);
}

/* QWEN_PREFILL_QUANT=1 → the prefill scratch is rebuilt from the quantized weights.
 * Default OFF: turning it on changes the generated audio (see above), so it is a
 * decision taken after listening, not a build-time default. */
static int tk_prefill_quant_enabled(void) {
    static atomic_int on = -1;
    int v = atomic_load_explicit(&on, memory_order_relaxed);
    if (v < 0) {
        const char *e = getenv("QWEN_PREFILL_QUANT");
        v = (e && e[0] == '1');
        atomic_store_explicit(&on, v, memory_order_relaxed);
    }
    return v;
}

/* Fill the per-layer f32 scratch from the cheapest weight set present (or from bf16
 * when the switch is off / the layer was never quantized). */
static void tk_prefill_weight_f32(float *dst, const uint16_t *Wb,
                                  const int8_t *Wi, const float *Ws,
                                  const q4_0_block_t *W4, const q6_0_block_t *W6,
                                  int rows, int cols, int quant) {
    if (quant && W6)            tk_deq_q6_matrix(dst, W6, rows, cols);
    else if (quant && W4)       tk_deq_q4_matrix(dst, W4, rows, cols);
    else if (quant && Wi && Ws) tk_deq_int8_matrix(dst, Wi, Ws, rows, cols);
    else                        bf16_to_f32_matrix(dst, Wb, (int64_t)rows * cols);
}

/* Owner of the heap-fused gate_up bf16 copies. qwen_talker_load runs ONLY for the
 * base ctx; qwen_tts_clone_for_worker byte-copies the layer array (`*w = *base`), so
 * every worker clone holds the SAME pointers while qwen_tts_unload frees them exactly
 * once, through the base. Releasing from a clone would therefore be a double free at
 * shutdown — so only the owner is allowed to release. */
static qwen_tts_ctx_t *g_tk_bf16_owner = NULL;
static atomic_int g_tk_bf16_released = 0;

/* Use centralized NEON+multi-threaded matvec from qwen_tts_kernels.c */
#define matvec_bf16_local qwen_matvec_bf16

/* ========================================================================
 * RoPE - NeoX SPLIT-HALF STYLE
 * Splits head into first half and second half: [x1..., x2...]
 * Rotated: [x1*cos - x2*sin, x2*cos + x1*sin]
 * ======================================================================== */

static void apply_rope_neox_inplace(float *x, int n_heads, int head_dim,
                                    const float *cos_cache,
                                    const float *sin_cache, int pos) {
    int half = head_dim / 2;
    const float *cos_ptr = cos_cache + (int64_t)pos * half;
    const float *sin_ptr = sin_cache + (int64_t)pos * half;

    for (int h = 0; h < n_heads; h++) {
        float *xh = x + h * head_dim;
#ifdef __ARM_NEON
        int i = 0;
        for (; i + 3 < half; i += 4) {
            float32x4_t c = vld1q_f32(cos_ptr + i);
            float32x4_t s = vld1q_f32(sin_ptr + i);
            float32x4_t v1 = vld1q_f32(xh + i);
            float32x4_t v2 = vld1q_f32(xh + i + half);
            vst1q_f32(xh + i,        vmlsq_f32(vmulq_f32(v1, c), v2, s));
            vst1q_f32(xh + i + half, vmlaq_f32(vmulq_f32(v2, c), v1, s));
        }
        for (; i < half; i++) {
            float x1 = xh[i], x2 = xh[i + half];
            xh[i]        = x1 * cos_ptr[i] - x2 * sin_ptr[i];
            xh[i + half] = x2 * cos_ptr[i] + x1 * sin_ptr[i];
        }
#elif defined(__AVX2__)
        int i = 0;
        for (; i + 8 <= half; i += 8) {
            __m256 c = _mm256_loadu_ps(cos_ptr + i);
            __m256 s = _mm256_loadu_ps(sin_ptr + i);
            __m256 v1 = _mm256_loadu_ps(xh + i);
            __m256 v2 = _mm256_loadu_ps(xh + i + half);
            _mm256_storeu_ps(xh + i,        _mm256_fmsub_ps(v1, c, _mm256_mul_ps(v2, s)));
            _mm256_storeu_ps(xh + i + half, _mm256_fmadd_ps(v2, c, _mm256_mul_ps(v1, s)));
        }
        for (; i < half; i++) {
            float x1 = xh[i], x2 = xh[i + half];
            xh[i]        = x1 * cos_ptr[i] - x2 * sin_ptr[i];
            xh[i + half] = x2 * cos_ptr[i] + x1 * sin_ptr[i];
        }
#else
        for (int i = 0; i < half; i++) {
            float x1 = xh[i];
            float x2 = xh[i + half];
            xh[i]        = x1 * cos_ptr[i] - x2 * sin_ptr[i];
            xh[i + half] = x2 * cos_ptr[i] + x1 * sin_ptr[i];
        }
#endif
    }
}

/* ========================================================================
 * KV Cache Growth
 * ======================================================================== */

static int kv_cache_grow(qwen_tts_ctx_t *ctx, int required) {
    if (required <= ctx->kv_max) return 0;

    int new_max = ctx->kv_max;
    while (new_max < required) new_max *= 2;

    int kv_dim = ctx->config.num_kv_heads * ctx->config.head_dim;

    uint16_t *new_k = (uint16_t *)aligned_malloc((int64_t)ctx->config.num_layers * new_max * kv_dim * sizeof(uint16_t));
    uint16_t *new_v = (uint16_t *)aligned_malloc((int64_t)ctx->config.num_layers * new_max * kv_dim * sizeof(uint16_t));
    if (!new_k || !new_v) { free(new_k); free(new_v); return -1; }

    for (int layer = 0; layer < ctx->config.num_layers; layer++) {
        int64_t old_off = (int64_t)layer * ctx->kv_max * kv_dim;
        int64_t new_off = (int64_t)layer * new_max * kv_dim;
        memcpy(new_k + new_off, ctx->kv_cache_k + old_off, (int64_t)ctx->kv_len * kv_dim * sizeof(uint16_t));
        memcpy(new_v + new_off, ctx->kv_cache_v + old_off, (int64_t)ctx->kv_len * kv_dim * sizeof(uint16_t));
    }
    free(ctx->kv_cache_k); free(ctx->kv_cache_v);
    ctx->kv_cache_k = new_k; ctx->kv_cache_v = new_v;
    ctx->kv_max = new_max;

    return 0;
}

/* ========================================================================
 * Weight Loading
 * ======================================================================== */

/* Alloc int8 buffers if absent, then (re)quantize from the current bf16 pointer.
 * Reusing existing buffers makes this safe to call a second time (e.g. after a
 * WDELTA voice override swaps the bf16 weights from CV to Base). */
static void tk_qz(int8_t **dst, float **scale, const uint16_t *src, int rows, int cols) {
    if (!*dst)   *dst   = (int8_t *)aligned_malloc((size_t)rows * cols);
    if (!*scale) *scale = (float *)aligned_malloc((size_t)rows * sizeof(float));
    if (*dst && *scale) qwen_quantize_bf16_to_int8(src, rows, cols, *dst, *scale);
}

/* (Re)quantize Talker weights to INT8 from the current bf16 pointers. Gated to
 * 1.7B (hidden>=2048); the 0.6B Talker is too small to benefit. */
void qwen_talker_quantize_int8(qwen_tts_ctx_t *ctx) {
    qwen_tts_config_t *c = &ctx->config;
    /* The old `hidden < 2048` gate (0.6B Talker stayed bf16 under --int8) was a denormal-hang
     * workaround, since fixed by FTZ (qwen_ftz_on in every matvec worker) + fused int8 qkv —
     * same fix that unblocked CP int8 at hidden=1024. Dropping it lets --int8 quantize the
     * 0.6B Talker too (int8 = the measured quality floor / gold), for the Talker-step speedup. */
    if (!ctx->use_int8) return;
    int h = c->hidden_size;
    int q_dim = c->num_heads * c->head_dim;
    int kv_dim = c->num_kv_heads * c->head_dim;
    int inter = c->intermediate_size;
    for (int i = 0; i < c->num_layers; i++) {
        qwen_talker_layer_t *l = &ctx->layers[i];
        tk_qz(&l->wq_int8, &l->wq_scale, l->wq_bf16, q_dim, h);
        tk_qz(&l->wk_int8, &l->wk_scale, l->wk_bf16, kv_dim, h);
        tk_qz(&l->wv_int8, &l->wv_scale, l->wv_bf16, kv_dim, h);
        tk_qz(&l->wo_int8, &l->wo_scale, l->wo_bf16, h, q_dim);
        tk_qz(&l->gate_up_fused_int8, &l->gate_up_fused_scale, l->gate_up_fused_bf16, 2 * inter, h);
        tk_qz(&l->down_int8, &l->down_scale, l->down_bf16, h, inter);
    }
}

/* ── Per-layer MIXED map: int8 where the profile says it matters, q6_0 elsewhere ──
 *
 * The dynamic (imatrix-style) bit allocation, but scored on LANGUAGE IDENTITY instead of on
 * perplexity — which is the whole point. llama.cpp's mixed K-quants allocate bits by
 * looking at a proxy loss; here they are allocated by looking at the thing the
 * listener actually hears. The layer set comes from the MEASURED sensitivity profile
 * (28 layers x "int6 everywhere EXCEPT L", scored at the breaking point), NOT from
 * the accent-delta prior — that prior was falsified three separate times: the
 * critical layers sit in the middle and high (L13 L26 L12 L24 L20 L11), while L00/L03,
 * the peak of the accent delta, are among the LEAST critical.
 *
 * Weight traffic vs today's all-int8 Talker:
 *     7 of 28 at int8  ->  0.8594 B/weight  (-14.1%)
 *     6 of 28 at int8  ->  0.8527 B/weight  (-14.7%)
 *
 * A layer belongs to EXACTLY ONE set. Quantizing a tensor twice would give
 * q6(int8(W)), which keeps the int8 error while looking like a perfectly valid
 * file — the same silent-corruption trap the fakequant scripts guard against with
 * disjoint --only regexes.
 *
 * Returns the number of layers left at int8, or -1 if the spec cannot be parsed. */
/* Per-layer format plan: 8 = int8 (per-row, today's gold), 6 = q6_0, 4 = q4_0.
 * THREE levels, not two, and that is the point of the whole track: the answer to
 * "int4 loses the language identity" is not "never use int4", it is "use it ONLY on the layers
 * the profile says are insensitive, and buy back the bits elsewhere". Quantizing
 * everything to int4 was measured and it drops the language ~1 seed in 5; spending
 * those same average bits unevenly is a different proposition, and it has to be
 * measured on the language identity layer by layer rather than assumed.
 *
 * Traffic per weight: int8 1.0, q6_0 0.8125, q4_0 0.5625.
 *   top6           (6x int8, 22x q6)                  0.8527  -14.7%
 *   tri6  (6x int8, 16x q6, 6x q4 on the least critical) 0.7991  -20.1%
 *
 * Spec grammar:
 *   top6 | top7        int8 on the measured top-N, q6_0 on the rest        (default top6)
 *   none               q6_0 everywhere — the floor, to price the map itself
 *   tri6               int8 top6 / q4_0 bottom6 / q6_0 in between
 *   8=13,26,12;4=10,3  explicit: int8 list, int4 list, everything else q6_0
 * Returns the number of int8 layers, or -1 if unparseable. */
enum { TK_FMT_Q6 = 6, TK_FMT_INT8 = 8, TK_FMT_Q4 = 4 };

/* The 28 Talker layers ordered by the MEASURED sensitivity profile, most critical
 * first (28 layers x "int6 everywhere EXCEPT L", 2 seeds, scored at the breaking
 * point — PLAN T2.next).
 *
 * ⚠️ READ THIS BEFORE TRUSTING THE ORDER. The 10-seed campaign (PLAN T2.topn)
 * FALSIFIED the idea that this ranking is usable: protecting the 7 LEAST critical
 * layers scored 97.1% language identity against 96.9% for the 7 most critical — indistinguishable,
 * with the seed-to-seed spread larger than the gap between maps. What matters is HOW
 * MANY layers stay at int8, not WHICH. The order is kept because it makes "top-N" a
 * reproducible, documented choice of N layers — NOT because it is optimal. Anyone
 * re-deriving a map from it is repeating a falsified assumption. */
static const int TK_RANK[28] = {
    13, 26, 12, 24, 20, 11, 21, 16, 14,  6, 17,  7, 27,  1,
     8, 22,  2, 19,  5,  4, 25, 18,  9, 15, 23,  0,  3, 10
};

static int tk_parse_layer_plan(const char *spec, int n_layers, unsigned char *fmt) {
    /* "rest=4" / "rest=6" sets what the UNLISTED layers get. Default q6_0.
     * rest=4 is the interesting one: int8 where it matters + q4_0 everywhere else is
     * -34.4% weight traffic at N=6, and BOTH halves already have fast, validated
     * kernels on every ISA — so if language identity holds, no new kernel is needed at all. */
    int fill = TK_FMT_Q6;
    if (spec && strstr(spec, "rest=4")) fill = TK_FMT_Q4;
    else if (spec && strstr(spec, "rest=8")) fill = TK_FMT_INT8;
    for (int i = 0; i < n_layers; i++) fmt[i] = (unsigned char)fill;

    #define TK_TOP_N(n, f) do {                                           \
        for (int _i = 0; _i < (n) && _i < 28; _i++)                       \
            if (TK_RANK[_i] < n_layers) fmt[TK_RANK[_i]] = (f);           \
    } while (0)
    #define TK_BOT_N(n, f) do {                                           \
        for (int _i = 0; _i < (n) && _i < 28; _i++)                       \
            if (TK_RANK[27 - _i] < n_layers) fmt[TK_RANK[27 - _i]] = (f); \
    } while (0)

    if (!spec || !*spec || !strcmp(spec, "top6") || !strcmp(spec, "1")) {
        TK_TOP_N(6, TK_FMT_INT8); return 6;
    }
    if (!strcmp(spec, "top7")) { TK_TOP_N(7, TK_FMT_INT8); return 7; }
    if (!strcmp(spec, "none")) return 0;
    if (!strcmp(spec, "tri6")) {                       /* int8 top6 / q4 bottom6 / q6 rest */
        TK_TOP_N(6, TK_FMT_INT8);
        TK_BOT_N(6, TK_FMT_Q4);
        return 6;
    }
    /* q4N — the ladder that PLAN T2.q4map calls for: int8 on N layers, q4_0 on ALL
     * the others. N is what is being searched for; the identity of the N is not. */
    if (!strncmp(spec, "q4n", 3)) {
        int n = atoi(spec + 3);
        if (n < 0 || n > n_layers) return -1;
        for (int i = 0; i < n_layers; i++) fmt[i] = TK_FMT_Q4;
        TK_TOP_N(n, TK_FMT_INT8);
        return n;
    }

    /* explicit: "8=13,26,12;4=10,3,0[;rest=4]" — a bare list is read as the int8 set */
    int n8 = 0, cur = TK_FMT_INT8;
    const char *p = spec;
    while (*p) {
        if (!strncmp(p, "rest=", 5)) { p += 6; continue; }   /* already handled above */
        if ((*p == '8' || *p == '6' || *p == '4') && p[1] == '=') {
            cur = *p - '0';
            p += 2;
            continue;
        }
        if (*p == ';' || *p == ',' || *p == ' ') { p++; continue; }
        char *end;
        long v = strtol(p, &end, 10);
        if (end == p) return -1;
        if (v >= 0 && v < n_layers) {
            if (fmt[v] == TK_FMT_INT8 && cur != TK_FMT_INT8) n8--;
            if (cur == TK_FMT_INT8 && fmt[v] != TK_FMT_INT8) n8++;
            fmt[v] = (unsigned char)cur;
        }
        p = end;
    }
    if (fill == TK_FMT_INT8)
        for (int i = 0; i < n_layers; i++) if (fmt[i] == TK_FMT_INT8) n8++;
    #undef TK_TOP_N
    #undef TK_BOT_N
    return n8;
}

/* Does any layer actually carry Q6_0 weights? The batched Talker dispatches on the
 * bf16/int8/q4 pointers only, so a q6 layer would quietly run BF16 there — but a map
 * made of int8 + q4_0 (e.g. q4nN) is perfectly safe under batching, since both of
 * those formats already have batched paths. Gating on the FLAG would have banned the
 * one configuration worth serving; gate on what the map actually allocated. */
int qwen_talker_has_q6(const qwen_tts_ctx_t *ctx) {
    for (int i = 0; i < ctx->config.num_layers; i++)
        if (ctx->layers[i].wq_q6 || ctx->layers[i].wo_q6 ||
            ctx->layers[i].gate_up_fused_q6 || ctx->layers[i].down_q6)
            return 1;
    return 0;
}

int qwen_talker_quantize_mixed_int6(qwen_tts_ctx_t *ctx, const char *spec) {
    qwen_tts_config_t *c = &ctx->config;
    int h = c->hidden_size;
    int q_dim = c->num_heads * c->head_dim;
    int kv_dim = c->num_kv_heads * c->head_dim;
    int inter = c->intermediate_size;
    if (h % Q6_0_BLOCK_SIZE || q_dim % Q6_0_BLOCK_SIZE || inter % Q6_0_BLOCK_SIZE) {
        fprintf(stderr, "  [mixed-int6] dims not a multiple of %d — refusing\n", Q6_0_BLOCK_SIZE);
        return -1;
    }
    unsigned char fmt[256];
    int nl = c->num_layers > 256 ? 256 : c->num_layers;
    int n_int8 = tk_parse_layer_plan(spec, nl, fmt);
    if (n_int8 < 0) {
        fprintf(stderr, "  [mixed-int6] cannot parse layer spec '%s'\n", spec ? spec : "");
        return -1;
    }

    int h_bpr = h / Q6_0_BLOCK_SIZE, qd_bpr = q_dim / Q6_0_BLOCK_SIZE,
        i_bpr = inter / Q6_0_BLOCK_SIZE;
    int n_q6 = 0, n_q4 = 0;
    for (int i = 0; i < nl; i++) {
        qwen_talker_layer_t *l = &ctx->layers[i];
        if (fmt[i] == TK_FMT_INT8) {
            /* int8 per-row absmax — byte for byte what ships today */
            tk_qz(&l->wq_int8, &l->wq_scale, l->wq_bf16, q_dim, h);
            tk_qz(&l->wk_int8, &l->wk_scale, l->wk_bf16, kv_dim, h);
            tk_qz(&l->wv_int8, &l->wv_scale, l->wv_bf16, kv_dim, h);
            tk_qz(&l->wo_int8, &l->wo_scale, l->wo_bf16, h, q_dim);
            tk_qz(&l->gate_up_fused_int8, &l->gate_up_fused_scale, l->gate_up_fused_bf16, 2 * inter, h);
            tk_qz(&l->down_int8, &l->down_scale, l->down_bf16, h, inter);
        } else if (fmt[i] == TK_FMT_Q4) {
            #define TK_Q4(dst, src, rows, bpr) do {                                          \
                l->dst = (q4_0_block_t *)aligned_malloc((size_t)(rows) * (bpr) * sizeof(q4_0_block_t)); \
                if (!l->dst) return -1;                                                       \
                qwen_quantize_bf16_to_q4_0(l->src, (rows), (bpr) * Q4_0_BLOCK_SIZE, l->dst);  \
            } while (0)
            TK_Q4(wq_q4, wq_bf16, q_dim, h_bpr);
            TK_Q4(wk_q4, wk_bf16, kv_dim, h_bpr);
            TK_Q4(wv_q4, wv_bf16, kv_dim, h_bpr);
            TK_Q4(wo_q4, wo_bf16, h, qd_bpr);
            TK_Q4(gate_up_fused_q4, gate_up_fused_bf16, 2 * inter, h_bpr);
            TK_Q4(down_q4, down_bf16, h, i_bpr);
            #undef TK_Q4
            n_q4++;
        } else {
            #define TK_Q6(dst, src, rows, bpr) do {                                          \
                l->dst = (q6_0_block_t *)aligned_malloc((size_t)(rows) * (bpr) * sizeof(q6_0_block_t)); \
                if (!l->dst) return -1;                                                       \
                qwen_quantize_bf16_to_q6_0(l->src, (rows), (bpr) * Q6_0_BLOCK_SIZE, l->dst);  \
            } while (0)
            TK_Q6(wq_q6, wq_bf16, q_dim, h_bpr);
            TK_Q6(wk_q6, wk_bf16, kv_dim, h_bpr);
            TK_Q6(wv_q6, wv_bf16, kv_dim, h_bpr);
            TK_Q6(wo_q6, wo_bf16, h, qd_bpr);
            TK_Q6(gate_up_fused_q6, gate_up_fused_bf16, 2 * inter, h_bpr);
            TK_Q6(down_q6, down_bf16, h, i_bpr);
            #undef TK_Q6
            n_q6++;
        }
    }
    if (!ctx->silent) {
        /* Print the plan per layer, not just the counts: the whole track turns on
         * WHICH layers got which format, and a run whose log does not say it cannot
         * be reproduced or argued with later. */
        fprintf(stderr, "  [mixed-int6] plan:");
        for (int i = 0; i < nl; i++) fprintf(stderr, "%c", fmt[i] == TK_FMT_INT8 ? '8' :
                                                          (fmt[i] == TK_FMT_Q4 ? '4' : '6'));
        double bpw = ((double)n_int8 * 1.0 + (double)n_q6 * 0.8125 + (double)n_q4 * 0.5625) / nl;
        fprintf(stderr, "  (L00..L%02d)  int8 %d / q6 %d / q4 %d — %.4f B/weight, "
                        "%+.1f%% weight traffic vs all-int8\n",
                nl - 1, n_int8, n_q6, n_q4, bpw, (bpw - 1.0) * 100.0);
    }
    return n_int8;
}

int qwen_talker_load(qwen_tts_ctx_t *ctx) {
    qwen_tts_config_t *c = &ctx->config;
    int h = c->hidden_size;
    int q_dim = c->num_heads * c->head_dim;
    int kv_dim = c->num_kv_heads * c->head_dim;

    if (!ctx->silent)
        fprintf(stderr, "Loading Talker weights (hidden=%d, head_dim=%d, layers=%d)...\n",
                h, c->head_dim, c->num_layers);

    /* Text embeddings */
    ctx->tok_embeddings_bf16 = get_bf16(ctx->safetensors, "talker.model.text_embedding.weight");
    if (!ctx->tok_embeddings_bf16) {
        fprintf(stderr, "Error: cannot find talker.model.text_embedding.weight\n");
        return -1;
    }

    /* Text projection */
    ctx->text_proj_fc1_bf16 = get_bf16(ctx->safetensors, "talker.text_projection.linear_fc1.weight");
    ctx->text_proj_fc1_bias = get_f32(ctx->safetensors, "talker.text_projection.linear_fc1.bias");
    ctx->text_proj_fc2_bf16 = get_bf16(ctx->safetensors, "talker.text_projection.linear_fc2.weight");
    ctx->text_proj_fc2_bias = get_f32(ctx->safetensors, "talker.text_projection.linear_fc2.bias");

    /* Codec head + embedding */
    ctx->codec_head_bf16 = get_bf16(ctx->safetensors, "talker.codec_head.weight");
    ctx->codec_embedding_bf16 = get_bf16(ctx->safetensors, "talker.model.codec_embedding.weight");

    /* Final norm */
    ctx->talker_norm = get_f32(ctx->safetensors, "talker.model.norm.weight");

    /* Per-layer weights */
    for (int i = 0; i < c->num_layers; i++) {
        qwen_talker_layer_t *l = &ctx->layers[i];
        char name[256];

        #define LOAD_BF16(field, fmt, ...) do { \
            snprintf(name, sizeof(name), fmt, ##__VA_ARGS__); \
            l->field = get_bf16(ctx->safetensors, name); \
            if (!l->field) { fprintf(stderr, "Error: cannot find %s\n", name); return -1; } \
        } while(0)

        #define LOAD_F32(field, fmt, ...) do { \
            snprintf(name, sizeof(name), fmt, ##__VA_ARGS__); \
            l->field = get_f32(ctx->safetensors, name); \
            if (!l->field) { fprintf(stderr, "Error: cannot find %s\n", name); return -1; } \
        } while(0)

        LOAD_BF16(wq_bf16, "talker.model.layers.%d.self_attn.q_proj.weight", i);
        LOAD_BF16(wk_bf16, "talker.model.layers.%d.self_attn.k_proj.weight", i);
        LOAD_BF16(wv_bf16, "talker.model.layers.%d.self_attn.v_proj.weight", i);
        LOAD_BF16(wo_bf16, "talker.model.layers.%d.self_attn.o_proj.weight", i);
        LOAD_F32(q_norm, "talker.model.layers.%d.self_attn.q_norm.weight", i);
        LOAD_F32(k_norm, "talker.model.layers.%d.self_attn.k_norm.weight", i);
        LOAD_F32(input_norm, "talker.model.layers.%d.input_layernorm.weight", i);
        LOAD_F32(post_attn_norm, "talker.model.layers.%d.post_attention_layernorm.weight", i);
        LOAD_BF16(gate_bf16, "talker.model.layers.%d.mlp.gate_proj.weight", i);
        LOAD_BF16(up_bf16, "talker.model.layers.%d.mlp.up_proj.weight", i);
        LOAD_BF16(down_bf16, "talker.model.layers.%d.mlp.down_proj.weight", i);

        /* Fuse gate+up: interleave rows [gate_row0, up_row0, gate_row1, ...] */
        {
            size_t row_bytes = (size_t)h * sizeof(uint16_t);
            l->gate_up_fused_bf16 = (uint16_t *)aligned_malloc(2 * (size_t)c->intermediate_size * row_bytes);
            for (int r = 0; r < c->intermediate_size; r++) {
                memcpy(l->gate_up_fused_bf16 + (size_t)(2 * r) * h,
                       l->gate_bf16 + (size_t)r * h, row_bytes);
                memcpy(l->gate_up_fused_bf16 + (size_t)(2 * r + 1) * h,
                       l->up_bf16 + (size_t)r * h, row_bytes);
            }
        }

        #undef LOAD_BF16
        #undef LOAD_F32
    }

    /* Talker precision normally follows --int8/--int4. QWEN_TALKER_PREC={bf16|int8|int4}
     * DECOUPLES it from the CP so the quant-ladder can hold the CP at bf16 and vary ONLY
     * the Talker — measuring how much Talker quant moves code0 (= the WORDS). No env →
     * unchanged behavior. (int8 still no-ops on 0.6B per the hidden>=2048 gate below.) */
    int tk_do_int8 = ctx->use_int8;
    int tk_do_int4 = ctx->use_int4;
    const char *tk_prec = getenv("QWEN_TALKER_PREC");
    if (tk_prec && *tk_prec) {
        tk_do_int8 = !strcmp(tk_prec, "int8");
        tk_do_int4 = !strcmp(tk_prec, "int4");
        if (!ctx->silent)
            fprintf(stderr, "  [QWEN_TALKER_PREC=%s] Talker precision decoupled from CP\n", tk_prec);
    }

    /* INT8 quantization of Talker weights (--int8; 1.7B only, hidden>=2048).
     * Extracted into qwen_talker_quantize_int8() so it can be re-run after a
     * WDELTA voice override (re-quantize the Base weights, not stale CV ones). */
    /* --quant-mixed-int6 (QWEN_TALKER_MIXED_INT6=top6|top7|none|"13,26,..."):
     * REPLACES the all-int8 Talker with the per-layer map. It deliberately rides on
     * top of --int8 so the Code Predictor is quantized identically in both arms —
     * the A/B has to differ in the Talker and NOWHERE else, or the comparison is
     * measuring two things at once. */
    const char *tk_mix6 = getenv("QWEN_TALKER_MIXED_INT6");
    if (tk_mix6 && *tk_mix6 && tk_do_int8) {
        if (qwen_talker_quantize_mixed_int6(ctx, tk_mix6) < 0) {
            fprintf(stderr, "  [mixed-int6] FAILED — refusing to fall back silently to int8\n");
            return -1;
        }
        tk_do_int8 = 0;   /* the map already quantized every layer, one format each */
    }

    if (tk_do_int8) {
        int save = ctx->use_int8; ctx->use_int8 = 1;
        if (c->hidden_size >= 2048 && !ctx->silent)
            fprintf(stderr, "  Quantizing Talker weights to INT8 (per-row absmax)...\n");
        qwen_talker_quantize_int8(ctx);
        ctx->use_int8 = save;
        if (c->hidden_size >= 2048 && !ctx->silent)
            fprintf(stderr, "  Talker INT8 quantization done (%d layers)\n", c->num_layers);
    }

    /* Q4_0 quantization of Talker weights (optional, enabled by --int4 flag) */
    if (tk_do_int4) {
        if (!ctx->silent)
            fprintf(stderr, "  Quantizing Talker weights to Q4_0 (4-bit)...\n");
        int inter = c->intermediate_size;
        int q_bpr = h / Q4_0_BLOCK_SIZE;          /* blocks per row for hidden dim */
        int qd_bpr = q_dim / Q4_0_BLOCK_SIZE;     /* blocks per row for q_dim */
        int i_bpr = inter / Q4_0_BLOCK_SIZE;       /* blocks per row for inter dim */
        for (int i = 0; i < c->num_layers; i++) {
            qwen_talker_layer_t *l = &ctx->layers[i];

            l->wq_q4 = (q4_0_block_t *)aligned_malloc((size_t)q_dim * q_bpr * sizeof(q4_0_block_t));
            qwen_quantize_bf16_to_q4_0(l->wq_bf16, q_dim, h, l->wq_q4);

            l->wk_q4 = (q4_0_block_t *)aligned_malloc((size_t)kv_dim * q_bpr * sizeof(q4_0_block_t));
            qwen_quantize_bf16_to_q4_0(l->wk_bf16, kv_dim, h, l->wk_q4);

            l->wv_q4 = (q4_0_block_t *)aligned_malloc((size_t)kv_dim * q_bpr * sizeof(q4_0_block_t));
            qwen_quantize_bf16_to_q4_0(l->wv_bf16, kv_dim, h, l->wv_q4);

            l->wo_q4 = (q4_0_block_t *)aligned_malloc((size_t)h * qd_bpr * sizeof(q4_0_block_t));
            qwen_quantize_bf16_to_q4_0(l->wo_bf16, h, q_dim, l->wo_q4);

            l->gate_up_fused_q4 = (q4_0_block_t *)aligned_malloc((size_t)2 * inter * q_bpr * sizeof(q4_0_block_t));
            qwen_quantize_bf16_to_q4_0(l->gate_up_fused_bf16, 2 * inter, h, l->gate_up_fused_q4);

            l->down_q4 = (q4_0_block_t *)aligned_malloc((size_t)h * i_bpr * sizeof(q4_0_block_t));
            qwen_quantize_bf16_to_q4_0(l->down_bf16, h, inter, l->down_q4);
        }
        if (!ctx->silent)
            fprintf(stderr, "  Talker Q4_0 quantization done (%d layers)\n", c->num_layers);
    }

    /* Allocate KV cache (bf16 — halves memory vs f32) */
    int initial_kv_max = 2048;
    int64_t kv_size = (int64_t)c->num_layers * initial_kv_max * kv_dim;
    ctx->kv_cache_k = (uint16_t *)aligned_calloc(kv_size, sizeof(uint16_t));
    ctx->kv_cache_v = (uint16_t *)aligned_calloc(kv_size, sizeof(uint16_t));
    ctx->kv_max = initial_kv_max;
    ctx->kv_len = 0;

    /* Allocate decode buffers (single-token step) — 64B aligned for NEON/BLAS */
    ctx->dec_x = (float *)aligned_calloc(h, sizeof(float));
    ctx->dec_x_norm = (float *)aligned_malloc(h * sizeof(float));
    ctx->dec_q = (float *)aligned_malloc(q_dim * sizeof(float));
    ctx->dec_k = (float *)aligned_malloc(kv_dim * sizeof(float));
    ctx->dec_v = (float *)aligned_malloc(kv_dim * sizeof(float));
    ctx->dec_attn_out = (float *)aligned_malloc(q_dim * sizeof(float));
    ctx->dec_proj_out = (float *)aligned_malloc(h * sizeof(float));
    ctx->dec_gate = (float *)aligned_malloc(2 * c->intermediate_size * sizeof(float));
    ctx->dec_up = NULL;  /* unused: gate buffer holds fused gate+up */
    ctx->dec_ffn_out = (float *)aligned_malloc(h * sizeof(float));
    /* SwiGLU tmp buffer: max of Talker inter and CP inter (CP allocated later, but inter is known) */
    int swiglu_size = c->intermediate_size;
    if (c->cp_intermediate_size > swiglu_size) swiglu_size = c->cp_intermediate_size;
    ctx->swiglu_tmp = (float *)aligned_malloc(swiglu_size * sizeof(float));

    /* Allocate RoPE cache */
    int rope_max = 8192;
    int half_dim = c->head_dim / 2;
    ctx->rope_inv_freq = (float *)aligned_malloc(half_dim * sizeof(float));
    ctx->rope_cos = (float *)aligned_malloc((int64_t)rope_max * half_dim * sizeof(float));
    ctx->rope_sin = (float *)aligned_malloc((int64_t)rope_max * half_dim * sizeof(float));

    for (int i = 0; i < half_dim; i++)
        ctx->rope_inv_freq[i] = 1.0f / powf(c->rope_theta, (float)(2 * i) / c->head_dim);

    for (int pos = 0; pos < rope_max; pos++) {
        for (int i = 0; i < half_dim; i++) {
            float angle = (float)pos * ctx->rope_inv_freq[i];
            ctx->rope_cos[pos * half_dim + i] = cosf(angle);
            ctx->rope_sin[pos * half_dim + i] = sinf(angle);
        }
    }
    ctx->rope_cache_len = rope_max;

    /* S3: remember which ctx owns the heap-fused gate_up bf16 copies (see
     * g_tk_bf16_owner). Only this ctx may release them. */
    g_tk_bf16_owner = ctx;

    if (!ctx->silent) {
        fprintf(stderr, "  Talker: %d layers loaded, KV cache %d slots\n", c->num_layers, initial_kv_max);
        fprintf(stderr, "  q_dim=%d kv_dim=%d (head_dim=%d), NeoX RoPE theta=%.0f\n",
                q_dim, kv_dim, c->head_dim, c->rope_theta);
    }

    return 0;
}

/* ========================================================================
 * Single-token Talker Step
 * ======================================================================== */

/* GPU-resident fused Talker step (qwen_tts_cuda_talker.cu). When set, both the decode step
 * and the sequential prefill delegate to it (it builds its OWN device KV from pos 0). Disabled
 * automatically when emotion steering is active (the fused step doesn't apply ml_steer yet). */
void *g_cuda_talker_state = NULL;
/* audit MED-2: the fused single-stream states above/below hold ONE device KV, so they belong to
 * exactly ONE ctx (the one they were init'd with). Guard every delegation with an owner check:
 * a clone ctx (server --workers N, batched single-worker clone) must fall through to the CPU
 * path instead of clobbering the owner's device KV mid-request (cross-request audio corruption). */
void *g_gpu_fused_owner = NULL;   /* the qwen_tts_ctx_t* the fused states were created for */
/* GPU-resident BATCHED Talker step (throughput path). When set, qwen_batch_talker_step_ragged
 * delegates to it (maintains its OWN device KV [B][kv_max][kvd]; seeded per slot on admit via
 * qwen_cuda_talker_batch_upload_slot). Created by qwen_tts_serve_continuous when QWEN_CUDA_BATCH=1. */
void *g_cuda_talker_batch_state = NULL;
#ifdef QWEN_HAVE_METAL
/* GPU-resident fused Talker step (Metal, G2). Same delegation as CUDA. */
void *g_metal_talker_state = NULL;
void *g_metal_talker_batch_state = NULL;   /* batched fused Talker step (server throughput) */
extern void qwen_metal_talker_step(void *state, const float *embed, float *hidden_out, int pos);
extern void qwen_metal_talker_get_dec_x(void *state, float *out);
extern void qwen_metal_talker_batch_step(void *state, const float *embeds, const int *pos_arr, float *hidden_out);
#endif
#ifdef QWEN_HAVE_CUDA
extern void qwen_cuda_talker_batch_step(void *state, const float *embeds, const int *pos_arr, float *hidden_out);
extern void qwen_cuda_talker_step(void *state, const float *embed, float *hidden_out, int pos);
extern void qwen_cuda_talker_get_dec_x(void *state, float *out);
#endif

int qwen_talker_step(qwen_tts_ctx_t *ctx, float *embed, float *hidden_out) {
    qwen_mm_component(QWEN_COMP_TALKER);   /* MAC audit: attribute this step's kernels */
    qwen_tts_config_t *c = &ctx->config;
    int h = c->hidden_size;
    int q_dim = c->num_heads * c->head_dim;
    int kv_dim = c->num_kv_heads * c->head_dim;
    int inter = c->intermediate_size;
    int pos = ctx->kv_len;
    float eps = c->rms_norm_eps;

    /* Fused-vs-CPU dispatch is per-REQUEST (ml_steer_weight), not per-frame (w_eff):
     * mixing fused steps (device KV only) and CPU steered steps (host KV only) inside one
     * request leaves each cache missing the other side's rows — a pulse schedule
     * (--ml-frames N) would read garbage after the pulse. A steered request runs fully on
     * the CPU path (the fused step doesn't apply ml_steer yet); qwen_tts_generate forces a
     * full prefill for it so the host KV is coherent (issue #19 part 2). */
#ifdef QWEN_HAVE_CUDA
    if (g_cuda_talker_state && ctx == g_gpu_fused_owner &&
        !(ctx->ml_steer && ctx->ml_steer_weight != 0.0f)) {
        qwen_cuda_talker_step(g_cuda_talker_state, embed, hidden_out, pos);
        /* The fused step skips the host state; refresh ctx->dec_x from the device residual
         * so the prefill→decode handoff (last_hidden = rms_norm(dec_x, talker_norm)) stays
         * correct across server delta-reuse requests and streaming chunk seeds (issue #19). */
        qwen_cuda_talker_get_dec_x(g_cuda_talker_state, ctx->dec_x);
        ctx->kv_len = pos + 1;
        return 0;
    }
#endif
#ifdef QWEN_HAVE_METAL
    if (g_metal_talker_state && ctx == g_gpu_fused_owner &&
        !(ctx->ml_steer && ctx->ml_steer_weight != 0.0f)) {
        qwen_metal_talker_step(g_metal_talker_state, embed, hidden_out, pos);
        qwen_metal_talker_get_dec_x(g_metal_talker_state, ctx->dec_x);
        ctx->kv_len = pos + 1;
        return 0;
    }
#endif

    if (kv_cache_grow(ctx, pos + 1) != 0) return -1;

    actmap_init(c->num_layers, h);   /* QWEN_ACT_MAP: one-time probe (no-op if unset) */

    memcpy(ctx->dec_x, embed, h * sizeof(float));

    for (int layer = 0; layer < c->num_layers; layer++) {
        qwen_talker_layer_t *l = &ctx->layers[layer];

        /* 1. Input RMSNorm */
        qwen_rms_norm(ctx->dec_x_norm, ctx->dec_x, l->input_norm, 1, h, eps);

        /* 2. QKV projections (unified dispatch — single barrier for all 3).
         * q6 goes FIRST: under --quant-mixed-int6 a layer has either the q6 or the
         * int8 pointers, never both, so the order only matters as documentation of
         * intent — but it keeps "the cheapest format present wins" true everywhere. */
        if (l->wq_q6)
            qwen_matvec_q6_0_qkv(ctx->dec_q, ctx->dec_k, ctx->dec_v,
                                  l->wq_q6, l->wk_q6, l->wv_q6,
                                  ctx->dec_x_norm, h, q_dim, kv_dim);
        else if (l->wq_q4)
            qwen_matvec_q4_0_qkv(ctx->dec_q, ctx->dec_k, ctx->dec_v,
                                  l->wq_q4, l->wk_q4, l->wv_q4,
                                  ctx->dec_x_norm, h, q_dim, kv_dim);
        else if (l->wq_int8)
            qwen_matvec_int8_qkv(ctx->dec_q, ctx->dec_k, ctx->dec_v,
                                  l->wq_int8, l->wq_scale,
                                  l->wk_int8, l->wk_scale,
                                  l->wv_int8, l->wv_scale,
                                  ctx->dec_x_norm, h, q_dim, kv_dim);
        else
            qwen_matvec_bf16_qkv(ctx->dec_q, ctx->dec_k, ctx->dec_v,
                                  l->wq_bf16, l->wk_bf16, l->wv_bf16,
                                  ctx->dec_x_norm, h, q_dim, kv_dim);

        /* 3. Q/K RMSNorm per-head */
        qwen_rms_norm_per_head(ctx->dec_q, l->q_norm, 1, c->num_heads, c->head_dim, eps);
        qwen_rms_norm_per_head(ctx->dec_k, l->k_norm, 1, c->num_kv_heads, c->head_dim, eps);

        /* 4. NeoX split-half RoPE */
        apply_rope_neox_inplace(ctx->dec_q, c->num_heads, c->head_dim,
                                ctx->rope_cos, ctx->rope_sin, pos);
        apply_rope_neox_inplace(ctx->dec_k, c->num_kv_heads, c->head_dim,
                                ctx->rope_cos, ctx->rope_sin, pos);

        /* 5. Append KV to cache (convert f32→bf16) */
        int64_t kv_offset = (int64_t)layer * ctx->kv_max * kv_dim + (int64_t)pos * kv_dim;
        f32_to_bf16_vec(ctx->kv_cache_k + kv_offset, ctx->dec_k, kv_dim);
        f32_to_bf16_vec(ctx->kv_cache_v + kv_offset, ctx->dec_v, kv_dim);

        /* 6. Causal GQA attention (bf16 KV cache) */
        float scale = 1.0f / sqrtf((float)c->head_dim);
        uint16_t *layer_k = ctx->kv_cache_k + (int64_t)layer * ctx->kv_max * kv_dim;
        uint16_t *layer_v = ctx->kv_cache_v + (int64_t)layer * ctx->kv_max * kv_dim;
        qwen_causal_attention_bf16kv(ctx->dec_attn_out, ctx->dec_q, layer_k, layer_v,
                                     1, pos + 1, c->num_heads, c->num_kv_heads,
                                     c->head_dim, scale, pos);

        /* 7. Output projection */
        if (l->wo_q6)
            qwen_matvec_q6_0(ctx->dec_proj_out, l->wo_q6, ctx->dec_attn_out, h, q_dim);
        else if (l->wo_q4)
            qwen_matvec_q4_0(ctx->dec_proj_out, l->wo_q4, ctx->dec_attn_out, h, q_dim);
        else if (l->wo_int8)
            qwen_matvec_int8(ctx->dec_proj_out, l->wo_int8, l->wo_scale,
                              ctx->dec_attn_out, h, q_dim);
        else
            matvec_bf16_local(ctx->dec_proj_out, l->wo_bf16, ctx->dec_attn_out, h, q_dim);

        /* 8. Fused residual-add + post-attention RMSNorm (saves one pass over dec_x) */
        qwen_rms_norm_residual(ctx->dec_x_norm, ctx->dec_x, ctx->dec_proj_out,
                               l->post_attn_norm, h, eps);

        /* 9. Fused gate+up SwiGLU FFN (single matvec, x loaded once) */
        if (l->gate_up_fused_q6)
            qwen_matvec_q6_0(ctx->dec_gate, l->gate_up_fused_q6, ctx->dec_x_norm,
                              2 * inter, h);
        else if (l->gate_up_fused_q4)
            qwen_matvec_q4_0(ctx->dec_gate, l->gate_up_fused_q4, ctx->dec_x_norm,
                              2 * inter, h);
        else if (l->gate_up_fused_int8)
            qwen_matvec_int8(ctx->dec_gate, l->gate_up_fused_int8, l->gate_up_fused_scale,
                              ctx->dec_x_norm, 2 * inter, h);
        else
            qwen_matvec_bf16(ctx->dec_gate, l->gate_up_fused_bf16, ctx->dec_x_norm,
                              2 * inter, h);
        qwen_swiglu_inplace(ctx->dec_gate, ctx->swiglu_tmp, inter);

        /* Down projection */
        if (l->down_q6)
            qwen_matvec_q6_0(ctx->dec_proj_out, l->down_q6, ctx->dec_gate, h, inter);
        else if (l->down_q4)
            qwen_matvec_q4_0(ctx->dec_proj_out, l->down_q4, ctx->dec_gate, h, inter);
        else if (l->down_int8)
            qwen_matvec_int8(ctx->dec_proj_out, l->down_int8, l->down_scale,
                              ctx->dec_gate, h, inter);
        else
            qwen_matvec_bf16(ctx->dec_proj_out, l->down_bf16, ctx->dec_gate, h, inter);

        /* Fused residual-add + next layer's input RMSNorm (or just add for last layer) */
        if (layer + 1 < c->num_layers) {
            qwen_rms_norm_residual(ctx->dec_x_norm, ctx->dec_x, ctx->dec_proj_out,
                                   ctx->layers[layer + 1].input_norm, h, eps);
        } else {
            for (int i = 0; i < h; i++) ctx->dec_x[i] += ctx->dec_proj_out[i];
        }

        if (g_actmap_path) actmap_accum(layer, ctx->dec_x, h);  /* per-layer residual (zero cost when off) */

        /* Multi-layer emotion steering: rotate the residual toward the captured
         * emotion direction at late layers (L21-25 carry emotion identity), but
         * PRESERVE the residual's L2 norm so we change TONE, not ENERGY. A plain
         * additive bias every frame drives the autoregressive loop into an
         * energy-collapse spiral (fades to silence) — norm-preservation fixes that.
         * Then re-norm the next layer's input so the rotation propagates. */
        if (ctx->ml_steer && ctx->ml_steer_w_eff != 0.0f &&
            layer >= ctx->ml_steer_l0 && layer <= ctx->ml_steer_l1) {
            const float *sv = ctx->ml_steer + (size_t)layer * ctx->ml_steer_dim;
            float w = ctx->ml_steer_w_eff;
            float n0 = 0.0f, n1 = 0.0f;
            for (int i = 0; i < h; i++) n0 += ctx->dec_x[i] * ctx->dec_x[i];
            for (int i = 0; i < h; i++) ctx->dec_x[i] += w * sv[i];
            for (int i = 0; i < h; i++) n1 += ctx->dec_x[i] * ctx->dec_x[i];
            if (n1 > 1e-12f) {
                float s = sqrtf(n0 / n1);            /* restore the pre-steer energy */
                for (int i = 0; i < h; i++) ctx->dec_x[i] *= s;
            }
            if (layer + 1 < c->num_layers)
                qwen_rms_norm(ctx->dec_x_norm, ctx->dec_x, ctx->layers[layer + 1].input_norm, 1, h, eps);
        }
    }

    /* Final RMSNorm */
    qwen_rms_norm(hidden_out, ctx->dec_x, ctx->talker_norm, 1, h, eps);

    if (g_actmap_path) { actmap_accum(c->num_layers, hidden_out, h); g_actmap_frames++; }

    ctx->kv_len = pos + 1;
    return 0;
}

/* ========================================================================
 * Prefill (multi-token)
 * ======================================================================== */

/* plan_v4 D1: prefill projection straight off bf16 weights, no per-layer
 * bf16->f32 conversion. Y[seq][out] = Xn[seq][in] @ W_bf16^T via qwen_matmat_bf16
 * (weight streamed once per <=16-col tile, bf16 = half the DRAM of the f32 convert).
 * matmat wants X as [in][B] and Y as [out][B], B<=16 (the fixed-B fast path caps at
 * 16), so tile seq into <=16 chunks and transpose in/out — cheap O(seq*dim) vs the
 * O(seq*dim*dim) gemm. Env-gated (QWEN_PREFILL_MATMAT=1) so we can A/B vs BLAS: on
 * M1 Accelerate sgemm rides the AMX coprocessor, so this only wins where BLAS is
 * weak/absent (the #else scalar path, Linux OpenBLAS) or when the convert dominates. */
static void prefill_proj_matmat(float *Y, const uint16_t *W, const float *Xn,
                                int seq, int in_dim, int out_dim,
                                float *xT, float *yT) {
    /* ── the native path ──────────────────────────────────────────────────────────
     * Xn is ALREADY [seq][in_dim] and Y is ALREADY [seq][out_dim], which is exactly
     * KleidiAI's LHS and dst layout. Both transposes below exist only because OUR
     * matmat wants [in][B]/[out][B]; on this path they disappear rather than being
     * optimized. The <=16 tiling goes with them - KleidiAI takes any m, so the real
     * 21-token prompt runs as ONE m=21 call instead of 16+5, and the B=5 cliff our
     * own bf16 kernel has (403 us at B=4, 735 us at B=5, measured) never happens. */
    qwen_census_op("prefill_bf16_native", out_dim, in_dim, seq);
    if (qwen_kleidi_prefill_enabled() &&
        qwen_kleidi_matmul_bf16_native(Y, W, Xn,
                                       (size_t)in_dim * sizeof(float),
                                       (size_t)out_dim * sizeof(float),
                                       out_dim, in_dim, seq)) {
        /* Book it: a native path that skips the dispatcher also skips MMSTAT, and an
         * audit that silently loses the whole prefill is worse than no audit. */
        if (qwen_matmat_stats_enabled() || qwen_census_enabled())
            qwen_matmat_stats_note(QWEN_MMK_KLEIDI_BF16,
                                   (long long)out_dim * in_dim * seq);
        return;
    }
    for (int s0 = 0; s0 < seq; s0 += 16) {
        int B = seq - s0; if (B > 16) B = 16;
        for (int b = 0; b < B; b++) {
            const float *xr = Xn + (int64_t)(s0 + b) * in_dim;
            for (int k = 0; k < in_dim; k++) xT[(int64_t)k * B + b] = xr[k];
        }
        qwen_matmat_bf16(yT, W, xT, out_dim, in_dim, B);   /* yT[out][B] */
        for (int b = 0; b < B; b++) {
            float *yr = Y + (int64_t)(s0 + b) * out_dim;
            for (int o = 0; o < out_dim; o++) yr[o] = yT[(int64_t)o * B + b];
        }
    }
}

/* ========================================================================
 * S3 — stop paying for the memory twice: release the bf16 layer weights
 * ========================================================================
 *
 * ONLY legal once the prefill reads the quantized weights (QWEN_PREFILL_QUANT=1),
 * because until then the bf16 copies are a live reader on every request. Gated by
 * its OWN switch (QWEN_FREE_BF16=1, default off) so the two halves can be measured
 * apart: quantized prefill alone changes the audio, the release alone changes the RSS.
 *
 * Two different kinds of memory, two different mechanisms:
 *
 *  - `gate_up_fused_bf16` is a HEAP copy (aligned_malloc, gate/up rows interleaved
 *    at load — ~50 MB/layer, ~1.4 GB on the 1.7B). madvise does nothing for it:
 *    it needs free(). Freed + NULLed, so a reader that slipped through crashes
 *    instead of silently reading zeros.
 *  - the six projections mmapped from the safetensors are handed back with
 *    madvise(MADV_DONTNEED) over the PAGE-ALIGNED INNER range of each tensor, so a
 *    neighbouring tensor sharing the first/last page is never touched. The pointer
 *    stays valid: touching it refaults from the file. That is deliberate — it is
 *    what keeps `layers[0].wq_bf16 != NULL` true, and with it the batched-path
 *    guards in qwen_tts.c (qwen_tts_generate_batch / _multi / serve_continuous),
 *    the --self-test oracle, and the CUDA/Metal uploaders, all of which read bf16.
 *
 * WHAT IS NOT TOUCHED: text/codec embeddings, the codec head, the x-vector, the
 * Code Predictor. Only the per-layer projections of the Talker.
 *
 * REFUSALS (a wrong release is a crash or, worse, silent garbage):
 *  1. every layer must carry a complete quantized set for all six tensors —
 *     otherwise some layer would still need its bf16;
 *  2. no layer may be q6: the batched projection (qwen_batch_proj_q) dispatches on
 *     q4 > int8 > bf16 and has NO q6 kernel, so a q6 layer under batching falls back
 *     to `gate_up_fused_bf16` — which we would just have freed;
 *  3. the free always goes through the OWNER ctx, whoever triggered it.
 *
 * WHO TRIGGERS IT vs WHOSE MEMORY IT IS. In the batched server the prefill runs on a
 * HELPER CLONE, not on the base ctx, so "only the owner may trigger" would mean the
 * release never happens on exactly the path that needs it. So: any ctx may trigger,
 * but the free/NULL is applied to g_tk_bf16_owner's layer array — the one
 * qwen_tts_unload will free. Clones keep their byte-copied (now stale) pointer; that
 * is inert, because refusal 1 guarantees every layer has a quantized set and EVERY
 * reader — decode step, prefill, qwen_batch_proj_q — checks the quantized pointers
 * first. The triggering clone's own copy is NULLed too, to shrink that window.
 *
 * KNOWN LIMIT, stated rather than hidden: the CUDA/Metal Talker uploaders take the
 * fused gate_up as (q4, int8+scale, bf16) and pick by precision. Refusal 1 means the
 * quantized set is always there when we free, so they never need the bf16 — but they
 * are opt-in builds (`make blas` never compiles them) and have not been exercised
 * with QWEN_FREE_BF16=1. Measure there before turning it on with a GPU backend.
 *
 * WHEN. At the first prefill, not at load: --wdelta / WOVR / --expr rebuild the fused
 * gate_up copy AFTER qwen_tts_load returns (main.c), and freeing it earlier would
 * make those overrides silently no-op on the FFN. By the first prefill every weight
 * override has been applied. */
static int tk_layer_fully_quantized(const qwen_talker_layer_t *l) {
#define TK_Q_OK(i8, sc, q4, q6) ((l->q6) != NULL || (l->q4) != NULL || \
                                 ((l->i8) != NULL && (l->sc) != NULL))
    return TK_Q_OK(wq_int8, wq_scale, wq_q4, wq_q6) &&
           TK_Q_OK(wk_int8, wk_scale, wk_q4, wk_q6) &&
           TK_Q_OK(wv_int8, wv_scale, wv_q4, wv_q6) &&
           TK_Q_OK(wo_int8, wo_scale, wo_q4, wo_q6) &&
           TK_Q_OK(gate_up_fused_int8, gate_up_fused_scale,
                   gate_up_fused_q4, gate_up_fused_q6) &&
           TK_Q_OK(down_int8, down_scale, down_q4, down_q6);
#undef TK_Q_OK
}

/* madvise(MADV_DONTNEED) the page-aligned inner range of [ptr, ptr+nbytes), but ONLY
 * if it really lies inside one of the safetensors mappings. The check is not
 * cosmetic: --wdelta/--expr REPLACE a weight pointer with an owned heap buffer
 * (main.c: "can't write to mmap'd weights"), and MADV_DONTNEED on anonymous memory
 * discards its CONTENT — it would zero a patched weight matrix without any error.
 * Returns the number of bytes actually handed back. */
static size_t tk_madv_dontneed_mapped(const qwen_tts_ctx_t *ctx,
                                      const void *ptr, size_t nbytes) {
#ifdef QWEN_HAVE_MADVISE
    if (!ptr || !nbytes || !ctx->safetensors) return 0;
    ingot_st *st = (ingot_st *)ctx->safetensors;
    uint32_t ns = ingot_st_shard_count(st);
    for (uint32_t sh = 0; sh < ns; sh++) {
        const void *base = NULL; size_t size = 0;
        if (ingot_st_mapping(st, sh, &base, &size) != 0 || !base) continue;
        const unsigned char *b = (const unsigned char *)base;
        const unsigned char *q = (const unsigned char *)ptr;
        if (q < b || q + nbytes > b + size) continue;         /* not this shard */
        long pg = sysconf(_SC_PAGESIZE);
        if (pg <= 0) return 0;
        uintptr_t lo = ((uintptr_t)q + (uintptr_t)pg - 1) & ~((uintptr_t)pg - 1);
        uintptr_t hi = ((uintptr_t)q + nbytes) & ~((uintptr_t)pg - 1);
        if (hi <= lo) return 0;                                /* smaller than a page */
        if (madvise((void *)lo, (size_t)(hi - lo), MADV_DONTNEED) != 0) return 0;
        return (size_t)(hi - lo);
    }
    return 0;
#else
    (void)ctx; (void)ptr; (void)nbytes;
    return 0;
#endif
}

static void tk_release_bf16(qwen_tts_ctx_t *caller) {
    static atomic_int free_env = -1;
    int on = atomic_load_explicit(&free_env, memory_order_relaxed);
    if (on < 0) {
        const char *e = getenv("QWEN_FREE_BF16");
        on = (e && e[0] == '1');
        atomic_store_explicit(&free_env, on, memory_order_relaxed);
    }
    if (!on) return;
    /* Refusal 3: the memory belongs to the owner ctx, so that is the one we edit —
     * whichever ctx (base or worker/prefill-helper clone) got here first. The
     * exchange makes it exactly-once when several workers arrive together. */
    qwen_tts_ctx_t *ctx = g_tk_bf16_owner;
    if (!ctx) return;
    if (atomic_exchange(&g_tk_bf16_released, 1) != 0) return;

    qwen_tts_config_t *c = &ctx->config;
    /* Refusal 2 */
    if (qwen_talker_has_q6(ctx)) {
        fprintf(stderr, "  [free-bf16] refused: a q6 layer is present and the batched "
                            "projection has no q6 kernel (it would fall back to the freed bf16)\n");
        return;
    }
    /* Refusal 1 */
    for (int i = 0; i < c->num_layers; i++) {
        if (!tk_layer_fully_quantized(&ctx->layers[i])) {
            fprintf(stderr, "  [free-bf16] refused: layer %d is not fully quantized — "
                                "the prefill would still need its bf16\n", i);
            return;
        }
    }

    int h = c->hidden_size;
    int q_dim = c->num_heads * c->head_dim;
    int kv_dim = c->num_kv_heads * c->head_dim;
    int inter = c->intermediate_size;
    size_t freed = 0, advised = 0;
    for (int i = 0; i < c->num_layers; i++) {
        qwen_talker_layer_t *l = &ctx->layers[i];
        if (l->gate_up_fused_bf16) {
            freed += (size_t)2 * inter * h * sizeof(uint16_t);
            free(l->gate_up_fused_bf16);
            l->gate_up_fused_bf16 = NULL;   /* crash a stray reader, never feed it zeros */
            /* and in the clone that triggered this, if it is not the owner */
            if (caller && caller != ctx) caller->layers[i].gate_up_fused_bf16 = NULL;
        }
        advised += tk_madv_dontneed_mapped(ctx, l->wq_bf16, (size_t)q_dim * h * 2);
        advised += tk_madv_dontneed_mapped(ctx, l->wk_bf16, (size_t)kv_dim * h * 2);
        advised += tk_madv_dontneed_mapped(ctx, l->wv_bf16, (size_t)kv_dim * h * 2);
        advised += tk_madv_dontneed_mapped(ctx, l->wo_bf16, (size_t)h * q_dim * 2);
        advised += tk_madv_dontneed_mapped(ctx, l->down_bf16, (size_t)h * inter * 2);
        /* gate/up: consumed once at load to build the fused copy, dead ever since */
        advised += tk_madv_dontneed_mapped(ctx, l->gate_bf16, (size_t)inter * h * 2);
        advised += tk_madv_dontneed_mapped(ctx, l->up_bf16, (size_t)inter * h * 2);
    }
    /* Printed even under --silent/server: QWEN_FREE_BF16=1 is an explicit opt-in and
     * this single line is the only evidence that it fired (the server runs silent). */
    fprintf(stderr, "  [free-bf16] released %.0f MB heap (fused gate_up) + "
                        "%.0f MB mmapped layer projections handed back\n",
                (double)freed / (1024.0 * 1024.0), (double)advised / (1024.0 * 1024.0));
}

/* The prefill weight for a slot: the untouched original when a quantized artifact has
 * replaced the decode pointer, otherwise the same pointer as always. */
#define PREFW(L, F) ((uint16_t *)((L)->F##_bf16_pref ? (L)->F##_bf16_pref : (L)->F##_bf16))

/* ── Pre-pack every weight KleidiAI will be asked for, ONCE, at load ──────────────
 *
 * The kernels can register lazily on a miss, but then the FIRST request pays every
 * pack - and TTFA is one of the numbers this work is judged on, so the cost belongs
 * to load time where it is visible. It also makes the native entries usable from the
 * very first call: those do not lazily register, by design (they are on the hot path
 * and must stay a single lookup).
 *
 * Scope is deliberate: the int8 decode weights and the bf16 PREFILL weights, which
 * are not always the same pointer - a quantized artifact replaces the decode pointer
 * and leaves the original for prefill (PREFW). Registering the decode bf16 pointer
 * instead would pack a matrix nothing reads. */
void qwen_kleidi_prepack(qwen_tts_ctx_t *ctx) {
    if (!ctx || (!qwen_kleidi_i8_enabled() && !qwen_kleidi_bf16_enabled())) return;
    qwen_tts_config_t *c = &ctx->config;
    const int h = c->hidden_size;
    const int q_dim = c->num_heads * c->head_dim;
    const int kv_dim = c->num_kv_heads * c->head_dim;
    const int inter = c->intermediate_size;

    for (int i = 0; i < c->num_layers; i++) {
        qwen_talker_layer_t *l = &ctx->layers[i];
        struct { const int8_t *w; const float *s; const uint16_t *b; int rows, cols, fam; } t[] = {
            { l->wq_int8, l->wq_scale, PREFW(l, wq), q_dim,  h,     QWEN_KAI_FAM_QKV },
            { l->wk_int8, l->wk_scale, PREFW(l, wk), kv_dim, h,     QWEN_KAI_FAM_QKV },
            { l->wv_int8, l->wv_scale, PREFW(l, wv), kv_dim, h,     QWEN_KAI_FAM_QKV },
            { l->wo_int8, l->wo_scale, PREFW(l, wo), h,      q_dim, QWEN_KAI_FAM_O   },
            { l->gate_up_fused_int8, l->gate_up_fused_scale, PREFW(l, gate_up_fused),
              2 * inter, h, QWEN_KAI_FAM_FFN },
            { l->down_int8, l->down_scale, PREFW(l, down), h, inter, QWEN_KAI_FAM_FFN },
        };
        for (size_t k = 0; k < sizeof t / sizeof t[0]; k++) {
            if (t[k].w && t[k].s)
                qwen_kleidi_register_i8_fam(t[k].w, t[k].w, t[k].s, t[k].rows, t[k].cols,
                                            QWEN_KAI_COMP_TALKER, t[k].fam);
            if (t[k].b)
                qwen_kleidi_register_bf16_fam(t[k].b, t[k].b, t[k].rows, t[k].cols,
                                              QWEN_KAI_COMP_TALKER, t[k].fam);
        }
    }
    /* The Code Predictor: five layers, and the fifteen lm_heads that run once per
     * codebook per frame - the single hottest matrix count in the engine. */
    for (int i = 0; i < c->cp_num_layers; i++) {
        qwen_cp_layer_t *l = &ctx->cp_layers[i];
        const int ch = c->cp_hidden_size;
        const int cq = c->cp_num_heads * c->cp_head_dim;
        const int ckv = c->cp_num_kv_heads * c->cp_head_dim;
        const int ci = c->cp_intermediate_size;
        struct { const int8_t *w; const float *s; int rows, cols, fam; } t[] = {
            { l->wq_int8, l->wq_scale, cq,  ch, QWEN_KAI_FAM_QKV },
            { l->wk_int8, l->wk_scale, ckv, ch, QWEN_KAI_FAM_QKV },
            { l->wv_int8, l->wv_scale, ckv, ch, QWEN_KAI_FAM_QKV },
            { l->wo_int8, l->wo_scale, ch,  cq, QWEN_KAI_FAM_O   },
            { l->gate_up_fused_int8, l->gate_up_fused_scale, 2 * ci, ch, QWEN_KAI_FAM_FFN },
            { l->down_int8, l->down_scale, ch, ci, QWEN_KAI_FAM_FFN },
        };
        for (size_t k = 0; k < sizeof t / sizeof t[0]; k++)
            if (t[k].w && t[k].s)
                qwen_kleidi_register_i8_fam(t[k].w, t[k].w, t[k].s, t[k].rows, t[k].cols,
                                            QWEN_KAI_COMP_CP, t[k].fam);
    }
    for (int g = 0; g < 15; g++)
        if (ctx->cp_lm_head_int8[g] && ctx->cp_lm_head_scale[g])
            qwen_kleidi_register_i8_fam(ctx->cp_lm_head_int8[g], ctx->cp_lm_head_int8[g],
                                        ctx->cp_lm_head_scale[g], c->codebook_size,
                                        c->cp_hidden_size, QWEN_KAI_COMP_CP, QWEN_KAI_FAM_HEADS);
    if (ctx->cp_mtp_proj_int8 && ctx->cp_mtp_proj_scale)
        qwen_kleidi_register_i8_fam(ctx->cp_mtp_proj_int8, ctx->cp_mtp_proj_int8,
                                    ctx->cp_mtp_proj_scale, c->cp_hidden_size,
                                    ctx->cp_emb_dim, QWEN_KAI_COMP_CP, QWEN_KAI_FAM_OTHER);
}

/* ══════════════════════════════════════════════════════════════════════════════════
 * PREFIX CACHE (opt-in: QWEN_PREFIX_CACHE=1)
 *
 * The first `inst_len + role_len + codec_len` positions of every prompt are a pure
 * function of (model, instruct, think mode, language_id, speaker_id) - the text has not
 * started yet. Their per-layer K/V are therefore identical across every request that
 * shares that tuple, and prefill is deterministic (no sampling), so reusing them is
 * exact, not an approximation.
 *
 * WHAT IS CACHED, AND WHY IN f32 AND NOT bf16
 * The KV cache the DECODE phase reads is bf16. But the PREFILL attends over the f32
 * K/V it just computed, deliberately, to avoid a bf16 round-trip. If the prefix came
 * back as bf16 the attention over those positions would see different numbers and the
 * audio would not be bit-identical. So the prefix is kept in f32, post-k_norm and
 * post-RoPE - byte for byte what the cold path had in pref_k/pref_v at that moment.
 * Cost: 28 layers x pos x 1024 x 4 B x 2 = 229 KB per position. Nine positions is
 * 2.1 MB, a 38-token instruct is 10.8 MB. Once per process, not per request.
 *
 * ONE ENTRY, NO LRU, NO EVICTION. Deliberately: a server serves one voice and one
 * language for its whole life, so a second entry has no user. Adding a policy before
 * the gain is measured is how caches become the bug.
 *
 * CONCURRENCY. The server synthesizes from TWO threads that can overlap: the batched
 * scheduler and the single-job clone worker (instruct / stream / voice_design). So the
 * entry has a three-state lifecycle - EMPTY -> FILLING -> READY - and exactly one
 * thread ever fills it (CAS on the state). READY is published with a release store and
 * read with an acquire load, so a reader that sees READY sees every byte written before
 * it. And once READY it is NEVER freed or replaced: a second key simply does not cache.
 * That is what makes "no eviction policy" a safety property and not just a shortcut -
 * there is no window in which one thread frees what another is reading.
 * ══════════════════════════════════════════════════════════════════════════════════ */
typedef struct {
    int   len, n_layers, kv_dim;
    int   speaker_id, language_id, think_mode;
    uint64_t ihash;              /* FNV-1a of the instruct + role + codec token ids */
    const void *model_tag;       /* ctx->layers: a different model must never match */
    float *k, *v;                /* [n_layers][len][kv_dim], f32, post-norm post-RoPE */
} qwen_prefix_cache_t;

/* ── Why FOUR slots and not one (measured 2026-08-24) ──────────────────────────────
 * The first version had a single entry and never replaced it. It filled on the very
 * first synthesis of the process - which in the server is the PRE-WARM, and the
 * pre-warm runs with language_id = -1 while every real request runs with a real
 * language. Result: filled once with the wrong key, then hits=0 misses=7, forever.
 * (That is what the PREFIXCACHE MISS-WHY line exists to say out loud: `len 8/9
 * lang -1/2050`.)
 *
 * Four append-only slots fix it without bringing back a policy: a slot is claimed once
 * by CAS, filled, published with a release store, and NEVER freed or reused. So there
 * is still no eviction, no LRU, and no window where one thread frees what another
 * reads - and a warm-up with an odd key costs one slot instead of the whole cache.
 * Four covers pre-warm + a couple of (speaker, language) pairs; past that the cache
 * simply stops taking new keys, which is a bounded and boring failure. */
#define QWEN_PFX_SLOTS 4
enum { PFX_EMPTY = 0, PFX_FILLING = 1, PFX_READY = 2 };
static qwen_prefix_cache_t g_pfx[QWEN_PFX_SLOTS];
static atomic_int g_pfx_state[QWEN_PFX_SLOTS];
static atomic_long g_pfx_hits, g_pfx_miss;

/* ── Default ON since 2026-08-24, and the evidence for it ─────────────────────────
 * Correctness is not a risk here: the audio is BIT-IDENTICAL to the uncached path (8
 * requests out of 8 on the box, both with and without an instruct prompt), because the
 * prefix comes back in f32 exactly as the cold path had it. So the only question was
 * whether it is worth the memory, and it is:
 *   prefill  124.6 -> 103.5 ms (-17%) with no instruct, 214.4 -> 129.0 ms (-40%) with;
 *   server 2x8, synchronized: TTFA p50 -16/-19% and p95 better at C=1, 3 AND 4, moving
 *   the p95<500ms threshold from C=3 to C=4.
 * Poisson came out neutral (p50 better at c=1, p95 within noise on 12-32 sample cells) -
 * which is expected: under Poisson the prefills do not queue up, so there is nothing for
 * the saving to compound against.
 * Cost: 3.9 MB per worker without instruct, 12.6 MB with. The pre-warm happens AFTER the
 * fork, so it is per worker and NOT shared copy-on-write.
 * QWEN_PREFIX_CACHE=0 turns it off. */
int qwen_prefix_cache_enabled(void) {
    static atomic_int cached = -1;
    int v = atomic_load_explicit(&cached, memory_order_relaxed);
    if (v < 0) {
        const char *e = getenv("QWEN_PREFIX_CACHE");
        v = !(e && e[0] == '0');
        atomic_store_explicit(&cached, v, memory_order_relaxed);
    }
    return v;
}

uint64_t qwen_prefix_hash(const int *toks, int n) {
    uint64_t h = 1469598103934665603ULL;
    for (int i = 0; i < n; i++) {
        uint32_t t = (uint32_t)toks[i];
        for (int b = 0; b < 4; b++) { h ^= (t >> (8 * b)) & 0xff; h *= 1099511628211ULL; }
    }
    return h;
}

void qwen_talker_prefix_key(qwen_tts_ctx_t *ctx, int prefix_len, int speaker_id,
                            int language_id, int think_mode, uint64_t ihash) {
    ctx->pfx_len   = prefix_len;
    ctx->pfx_spk   = speaker_id;
    ctx->pfx_lang  = language_id;
    ctx->pfx_think = think_mode;
    ctx->pfx_ihash = ihash;
}

static int pfx_match(const qwen_prefix_cache_t *e, const qwen_tts_ctx_t *ctx) {
    return e->len == ctx->pfx_len
        && e->model_tag == (const void *)ctx->layers
        && e->n_layers == ctx->config.num_layers
        && e->kv_dim == ctx->config.num_kv_heads * ctx->config.head_dim
        && e->speaker_id == ctx->pfx_spk && e->language_id == ctx->pfx_lang
        && e->think_mode == ctx->pfx_think && e->ihash == ctx->pfx_ihash;
}

/* The entry to reuse for this request, or NULL. Acquire on the slot state pairs with
 * the release store in the filler, so a READY slot is fully visible. */
static const qwen_prefix_cache_t *pfx_find(const qwen_tts_ctx_t *ctx) {
    if (ctx->pfx_len <= 0) return NULL;
    for (int i = 0; i < QWEN_PFX_SLOTS; i++)
        if (atomic_load_explicit(&g_pfx_state[i], memory_order_acquire) == PFX_READY
            && pfx_match(&g_pfx[i], ctx)) return &g_pfx[i];
    return NULL;
}

void qwen_prefix_cache_stats(int *n_ready, size_t *bytes) {
    int n = 0; size_t b = 0;
    for (int i = 0; i < QWEN_PFX_SLOTS; i++)
        if (atomic_load_explicit(&g_pfx_state[i], memory_order_acquire) == PFX_READY) {
            n++;
            b += (size_t)g_pfx[i].n_layers * g_pfx[i].len * g_pfx[i].kv_dim * sizeof(float) * 2;
        }
    if (n_ready) *n_ready = n;
    if (bytes) *bytes = b;
}

__attribute__((destructor)) static void pfx_report(void) {
    long h = atomic_load(&g_pfx_hits), m = atomic_load(&g_pfx_miss);
    if (!h && !m) return;
    int n; size_t b; qwen_prefix_cache_stats(&n, &b);
    fprintf(stderr, "PREFIXCACHE hits=%ld misses=%ld slots_ready=%d bytes=%zu\n", h, m, n, b);
}

/* After fork(): a child inherits every READY slot as copy-on-write read-only pages that
 * all workers share for free. Nothing to do - which is the point of never freeing. */

/* Wall time of the prefill itself, printed unconditionally under QWEN_TTFA_TRACE.
 * The engine's own "Prefill: N ms" line sits behind !ctx->silent, and the server sets
 * silent=1 - so on the one configuration where the prefix cache can hit, the number it
 * changes is invisible. This is that number. */
static double pfx_now_ms(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

int qwen_talker_prefill(qwen_tts_ctx_t *ctx, float *input_embeds, int seq_len) {
    qwen_mm_component(QWEN_COMP_TALKER);   /* MAC audit: attribute this step's kernels */
    static int trace = -1;
    if (trace < 0) { const char *e = getenv("QWEN_TTFA_TRACE"); trace = (e && e[0] && e[0] != '0'); }
    double pfx_t0 = trace ? pfx_now_ms() : 0.0;
    qwen_tts_config_t *c = &ctx->config;
    int h = c->hidden_size;
    int q_dim = c->num_heads * c->head_dim;
    int kv_dim = c->num_kv_heads * c->head_dim;
    int inter = c->intermediate_size;
    float eps = c->rms_norm_eps;

    /* plan_v4 D1: route the prefill projections through qwen_matmat_bf16 (direct
     * bf16 weights, no per-layer f32 conversion). MEASURED on M1: Accelerate's
     * sgemm rides the AMX matrix coprocessor and beats NEON matmat ~4.7× for the
     * prefill gemm, so with USE_BLAS this LOSES (kept off by default). Without BLAS
     * it replaces the naive scalar triple-loop (a large win) AND drops the f32
     * convert, so it's the default there. QWEN_PREFILL_MATMAT=1/0 forces either way
     * (e.g. to bench a weak-BLAS Linux box where matmat may win). */
    static __thread int mm_env = -1;
    if (mm_env < 0) {
        const char *e = getenv("QWEN_PREFILL_MATMAT");
        if (e) mm_env = (e[0] == '1');
#ifdef USE_BLAS
        /* S19: measured on the c3 with the per-request TTFA trace (0.6B, server, -j4),
         * and the six arms separate TWO effects that are almost exactly half and half:
         *
         *   build          prefill   TTFA     what it isolates
         *   AMX   + BLAS     342 ms  648 ms
         *   AMX   + matmat   111 ms  306 ms   tile + no f32 materialisation
         *   AMX off + BLAS   343 ms  630 ms
         *   AMX off + matmat 220 ms  409 ms   ONLY no f32 materialisation
         *   VNNI build+BLAS  320 ms  612 ms
         *   VNNI build+matmat 231 ms 420 ms   idem, on a binary with no AMX at all
         *
         * The BLAS path materialises EVERY layer's weights in f32 on EVERY prefill and
         * then runs an fp32 sgemm; matmat_bf16 streams the bf16 weights as they are.
         * Skipping that conversion is worth ~120 ms of the 231 ms saved and needs no
         * matrix unit: the VNNI-only build does not even compile -mavx512bf16 and still
         * takes a third off TTFA. The tile is the other ~110 ms.
         *
         * ⚠️ AND THEN THE PRODUCT BENCH SAID NO, on the build without a tile unit. Widening
         * the default to all AVX-512 was tried and reverted the same hour: on the VNNI-only
         * build the Poisson probe (which runs at -j2, half the cores, like the product) got
         * WORSE, p50 787/784/1076 -> 1052/1442/1609 ms at c=1/2/3, even though the
         * single-request trace at -j4 had it winning 612 -> 420 ms. The reason is threading,
         * not arithmetic: OpenBLAS parallelises its own sgemm, matmat_bf16 rides the caller's
         * pool, so at -j4 it wins and at -j2 it loses. A trace at full threads is not a
         * product measurement, and this is the second time today that mistake was available.
         *
         * So the default stays tied to the TILE, where the win is large enough to survive
         * both thread budgets. Same weights either way: arithmetic order, not precision.
         * ⚠️ Ear-validated on OSS presets only (samples/tests/2026-08-19_prefill-matmat-amx);
         * the language-identity check on the real finetune is still owed - see PLAN S19. */
        /* ARM (2026-08-21, GCP c4a / Axion Neoverse-V2): stessa mossa dell'AMX, e qui
         * it passes the check that had rejected it on x86. Measured on a finetuned
         * checkpoint, --int8, a 21-token prefill, BLAS -> matmat:
         *
         *   -j 1   2014 -> 1439 ms      -j 8    434 -> 244 ms
         *   -j 2   1207 ->  754 ms      -j 16   310 -> 167 ms
         *   -j 4    679 ->  413 ms
         *
         * Vince a OGNI budget di thread, ed e' la differenza con la vicenda x86 del 19/08:
         * la' il default fu legato al TILE perche' sulla build VNNI-only il banco a -j2
         * (il budget del prodotto) peggiorava, mentre il trace a -j4 migliorava. Qui non
         * c'e' quel bivio: OpenBLAS parallelizza il suo sgemm e matmat cavalca il pool del
         * chiamante, ma anche a un thread solo il matmat resta davanti.
         * Language identity verified BEFORE changing the default: five voices,
         * language-identity scoring, 97.8-99.3% against 97.9-99.5% for the BLAS path, off-target
         * language 0.0-0.1%, durations normal.
         * ⚠️ Cambia l'ordine aritmetico (attivazioni troncate a bf16), quindi l'audio non e'
         * bit-identico: QWEN_PREFILL_MATMAT=0 riporta a BLAS senza ricompilare. */
        else mm_env = qwen_amx_bf16_available() || qwen_arm_bf16_matmat_available();
#else
        else mm_env = 1;   /* no BLAS: matmat_bf16 >> scalar, and skips the convert */
#endif
    }
    int use_matmat = mm_env;
    /* S3: the quantized source only applies to the dequant-into-f32-scratch path.
     * QWEN_PREFILL_MATMAT=1 streams the bf16 weights straight into qwen_matmat_bf16
     * and has no quantized twin, so it stays on bf16 — and, for the same reason, the
     * bf16 release below is CONDITIONAL on not being in that path. */
    int pref_quant = !use_matmat && tk_prefill_quant_enabled();
    if (pref_quant) tk_release_bf16(ctx);   /* no-op unless QWEN_FREE_BF16=1 */
    static __thread float *pp_xT = NULL, *pp_yT = NULL;
    static __thread int pp_cap = 0;
    if (use_matmat) {
        int need_in = (h > inter ? h : inter);       /* max projection input dim */
        int need_out = 2 * inter;                     /* max projection output dim */
        int cap = (need_in > need_out ? need_in : need_out) * 16;
        if (cap > pp_cap) {
            float *nx = (float *)realloc(pp_xT, (size_t)need_in * 16 * sizeof(float));
            float *ny = (float *)realloc(pp_yT, (size_t)need_out * 16 * sizeof(float));
            if (nx) pp_xT = nx;
            if (ny) pp_yT = ny;
            if (!pp_xT || !pp_yT) { use_matmat = 0; } else { pp_cap = cap; }
        }
    }

    if (!ctx->silent) fprintf(stderr, "  Prefill: %d tokens, hidden=%d\n", seq_len, h);

    if (kv_cache_grow(ctx, seq_len) != 0) return -1;

    /* Allocate/grow persistent prefill buffers (reused across generations in server mode) */
    if (seq_len > ctx->pref_seq_cap) {
        free(ctx->pref_residual); free(ctx->pref_q); free(ctx->pref_k); free(ctx->pref_v);
        free(ctx->pref_x_norm); free(ctx->pref_attn_out); free(ctx->pref_gate); free(ctx->pref_proj);
        ctx->pref_residual = (float *)aligned_malloc((int64_t)seq_len * h * sizeof(float));
        ctx->pref_q = (float *)aligned_malloc((int64_t)seq_len * q_dim * sizeof(float));
        ctx->pref_k = (float *)aligned_malloc((int64_t)seq_len * kv_dim * sizeof(float));
        ctx->pref_v = (float *)aligned_malloc((int64_t)seq_len * kv_dim * sizeof(float));
        ctx->pref_x_norm = (float *)aligned_malloc((int64_t)seq_len * h * sizeof(float));
        ctx->pref_attn_out = (float *)aligned_malloc((int64_t)seq_len * q_dim * sizeof(float));
        ctx->pref_gate = (float *)aligned_malloc((int64_t)seq_len * 2 * inter * sizeof(float));
        ctx->pref_proj = (float *)aligned_malloc((int64_t)seq_len * h * sizeof(float));
        ctx->pref_seq_cap = seq_len;
    }
    /* Allocate persistent weight conversion buffers (fixed size, allocated once) */
    if (!ctx->pref_wq_f32) {
        ctx->pref_wq_f32 = (float *)aligned_malloc((int64_t)q_dim * h * sizeof(float));
        ctx->pref_wk_f32 = (float *)aligned_malloc((int64_t)kv_dim * h * sizeof(float));
        ctx->pref_wv_f32 = (float *)aligned_malloc((int64_t)kv_dim * h * sizeof(float));
        ctx->pref_wo_f32 = (float *)aligned_malloc((int64_t)h * q_dim * sizeof(float));
        ctx->pref_gate_up_f32 = (float *)aligned_malloc((int64_t)2 * inter * h * sizeof(float));
        ctx->pref_down_f32 = (float *)aligned_malloc((int64_t)h * inter * sizeof(float));
    }

    float *residual = ctx->pref_residual;
    float *pref_q = ctx->pref_q;
    float *pref_k = ctx->pref_k;
    float *pref_v = ctx->pref_v;
    float *pref_x_norm = ctx->pref_x_norm;
    float *pref_attn_out = ctx->pref_attn_out;
    float *pref_gate = ctx->pref_gate;
    float *pref_proj = ctx->pref_proj;
    float *wq_f32 = ctx->pref_wq_f32;
    float *wk_f32 = ctx->pref_wk_f32;
    float *wv_f32 = ctx->pref_wv_f32;
    float *wo_f32 = ctx->pref_wo_f32;
    float *gate_up_f32 = ctx->pref_gate_up_f32;
    float *down_f32 = ctx->pref_down_f32;

    if (!residual || !pref_q || !pref_k || !pref_v || !pref_x_norm ||
        !pref_attn_out || !pref_gate || !pref_proj ||
        !wq_f32 || !wk_f32 || !wv_f32 || !wo_f32 || !gate_up_f32 || !down_f32) {
        fprintf(stderr, "Error: prefill allocation failed\n");
        return -1;
    }

    /* ── warm prefix? ──────────────────────────────────────────────────────────── */
    const qwen_prefix_cache_t *pfx = qwen_prefix_cache_enabled() ? pfx_find(ctx) : NULL;
    const int pos0  = pfx ? pfx->len : 0;
    const int n_new = seq_len - pos0;
    /* Cold, but worth saving for the next request with the same key? */
    /* Claim the right to fill: EMPTY -> FILLING, exactly once, whoever gets there first.
     * A loser simply does not cache this request; it will hit on the next one. */
    qwen_prefix_cache_t *fill = NULL;
    if (qwen_prefix_cache_enabled() && !pfx && ctx->pfx_len > 0 && ctx->pfx_len < seq_len) {
        for (int i = 0; i < QWEN_PFX_SLOTS && !fill; i++) {
            int expect = PFX_EMPTY;
            if (!atomic_compare_exchange_strong_explicit(&g_pfx_state[i], &expect, PFX_FILLING,
                                                         memory_order_acq_rel, memory_order_relaxed))
                continue;                                   /* someone else owns this slot */
            size_t n = (size_t)c->num_layers * ctx->pfx_len * kv_dim;
            g_pfx[i].k = (float *)malloc(n * sizeof(float));
            g_pfx[i].v = (float *)malloc(n * sizeof(float));
            if (g_pfx[i].k && g_pfx[i].v) fill = &g_pfx[i];
            else {
                free(g_pfx[i].k); free(g_pfx[i].v); g_pfx[i].k = g_pfx[i].v = NULL;
                atomic_store_explicit(&g_pfx_state[i], PFX_EMPTY, memory_order_release);
                break;                                      /* out of memory: stop trying */
            }
        }
    }
    const int pfx_saving = (fill != NULL);
    float *pref_kn = pref_k + (int64_t)pos0 * kv_dim;   /* where the NEW K rows go */
    float *pref_vn = pref_v + (int64_t)pos0 * kv_dim;
    /* Counted, and announced ONCE unconditionally. The server runs with ctx->silent=1,
     * so a message behind that flag is a message nobody will ever read - and "is the
     * cache actually hitting?" is precisely the question a silent server has to answer. */
    if (pos0 > 0) {
        long n = atomic_fetch_add_explicit(&g_pfx_hits, 1, memory_order_relaxed) + 1;
        if (n == 1 || !ctx->silent)
            fprintf(stderr, "  Prefix cache HIT: %d/%d positions reused, computing %d\n",
                    pos0, seq_len, n_new);
    } else if (qwen_prefix_cache_enabled() && ctx->pfx_len > 0) {
        atomic_fetch_add_explicit(&g_pfx_miss, 1, memory_order_relaxed);
    }

    memcpy(residual, input_embeds + (int64_t)pos0 * h, (int64_t)n_new * h * sizeof(float));

    if (ctx->debug) {
        /* Debug: print first position embedding values */
        fprintf(stderr, "[PREFILL] input_embeds[0][:8]:");
        for (int j = 0; j < 8 && j < h; j++) fprintf(stderr, " %.6f", residual[j]);
        fprintf(stderr, "\n");
    }

    for (int layer = 0; layer < c->num_layers; layer++) {
        qwen_talker_layer_t *l = &ctx->layers[layer];

        /* Warm prefix: rows [0,pos0) of THIS layer's K/V come back from the cache in
         * f32, post-norm and post-RoPE - exactly the bytes the cold path had here. */
        if (pos0 > 0) {
            size_t off = (size_t)layer * pos0 * kv_dim, nb = (size_t)pos0 * kv_dim * sizeof(float);
            memcpy(pref_k, pfx->k + off, nb);
            memcpy(pref_v, pfx->v + off, nb);
        }

        /* Fill the per-layer f32 scratch (skipped in D1 matmat mode, which reads the
         * bf16 weights directly). SOURCE: the quantized weights when QWEN_PREFILL_QUANT=1
         * and they exist, the bf16 otherwise — same buffers, same sgemm below. */
        if (!use_matmat) {
            tk_prefill_weight_f32(wq_f32, PREFW(l, wq), l->wq_int8, l->wq_scale,
                                  l->wq_q4, l->wq_q6, q_dim, h, pref_quant);
            tk_prefill_weight_f32(wk_f32, PREFW(l, wk), l->wk_int8, l->wk_scale,
                                  l->wk_q4, l->wk_q6, kv_dim, h, pref_quant);
            tk_prefill_weight_f32(wv_f32, PREFW(l, wv), l->wv_int8, l->wv_scale,
                                  l->wv_q4, l->wv_q6, kv_dim, h, pref_quant);
            tk_prefill_weight_f32(wo_f32, PREFW(l, wo), l->wo_int8, l->wo_scale,
                                  l->wo_q4, l->wo_q6, h, q_dim, pref_quant);
            tk_prefill_weight_f32(gate_up_f32, PREFW(l, gate_up_fused),
                                  l->gate_up_fused_int8, l->gate_up_fused_scale,
                                  l->gate_up_fused_q4, l->gate_up_fused_q6,
                                  2 * inter, h, pref_quant);
            tk_prefill_weight_f32(down_f32, PREFW(l, down), l->down_int8, l->down_scale,
                                  l->down_q4, l->down_q6, h, inter, pref_quant);
        }

        /* 1. Input RMSNorm for all positions */
        qwen_rms_norm(pref_x_norm, residual, l->input_norm, n_new, h, eps);

        /* 2. QKV projections */
        if (use_matmat) {
            prefill_proj_matmat(pref_q, PREFW(l, wq), pref_x_norm, n_new, h, q_dim,  pp_xT, pp_yT);
            prefill_proj_matmat(pref_kn, PREFW(l, wk), pref_x_norm, n_new, h, kv_dim, pp_xT, pp_yT);
            prefill_proj_matmat(pref_vn, PREFW(l, wv), pref_x_norm, n_new, h, kv_dim, pp_xT, pp_yT);
        } else {
#ifdef USE_BLAS
        /* x_norm[n_new, h] × W^T[h, out_dim] = out[n_new, out_dim] */
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    n_new, q_dim, h, 1.0f,
                    pref_x_norm, h, wq_f32, h, 0.0f, pref_q, q_dim);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    n_new, kv_dim, h, 1.0f,
                    pref_x_norm, h, wk_f32, h, 0.0f, pref_kn, kv_dim);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    n_new, kv_dim, h, 1.0f,
                    pref_x_norm, h, wv_f32, h, 0.0f, pref_vn, kv_dim);
#else
        for (int s = 0; s < n_new; s++) {
            const float *xs = pref_x_norm + (int64_t)s * h;
            float *qs = pref_q + (int64_t)s * q_dim;
            float *ks = pref_kn + (int64_t)s * kv_dim;
            float *vs = pref_vn + (int64_t)s * kv_dim;
            for (int o = 0; o < q_dim; o++) {
                float sum = 0.0f;
                const float *row = wq_f32 + (int64_t)o * h;
                for (int i = 0; i < h; i++) sum += row[i] * xs[i];
                qs[o] = sum;
            }
            for (int o = 0; o < kv_dim; o++) {
                float sum = 0.0f;
                const float *row = wk_f32 + (int64_t)o * h;
                for (int i = 0; i < h; i++) sum += row[i] * xs[i];
                ks[o] = sum;
            }
            for (int o = 0; o < kv_dim; o++) {
                float sum = 0.0f;
                const float *row = wv_f32 + (int64_t)o * h;
                for (int i = 0; i < h; i++) sum += row[i] * xs[i];
                vs[o] = sum;
            }
        }
#endif
        }

        /* 3. Q/K RMSNorm per-head */
        qwen_rms_norm_per_head(pref_q, l->q_norm, n_new, c->num_heads, c->head_dim, eps);
        qwen_rms_norm_per_head(pref_kn, l->k_norm, n_new, c->num_kv_heads, c->head_dim, eps);

        /* 4. NeoX split-half RoPE for all positions */
        for (int s = 0; s < n_new; s++) {
            /* ABSOLUTE position s+pos0: with a warm prefix the first computed row is not
             * position 0, and RoPE is the one place where that has to be said out loud. */
            apply_rope_neox_inplace(pref_q + (int64_t)s * q_dim, c->num_heads, c->head_dim,
                                    ctx->rope_cos, ctx->rope_sin, s + pos0);
            apply_rope_neox_inplace(pref_kn + (int64_t)s * kv_dim, c->num_kv_heads, c->head_dim,
                                    ctx->rope_cos, ctx->rope_sin, s + pos0);
        }

        /* Cold path, saving: the prefix rows are FINAL here - projected, k_norm'd and
         * RoPE'd. Anything later in the layer does not touch them. */
        if (pfx_saving) {
            size_t off = (size_t)layer * ctx->pfx_len * kv_dim;
            size_t nb  = (size_t)ctx->pfx_len * kv_dim * sizeof(float);
            memcpy(fill->k + off, pref_k, nb);
            memcpy(fill->v + off, pref_v, nb);
        }

        /* 5. Store KV into cache (convert f32→bf16) */
        int64_t cache_base = (int64_t)layer * ctx->kv_max * kv_dim;
        /* The whole [0, n_new) goes to the cache: with a warm prefix, rows [0,pos0)
         * were memcpy'd back into pref_k/pref_v above, so this stays one call and the
         * decode phase still finds every position it will attend to. */
        f32_to_bf16_vec(ctx->kv_cache_k + cache_base, pref_k, (int64_t)seq_len * kv_dim);
        f32_to_bf16_vec(ctx->kv_cache_v + cache_base, pref_v, (int64_t)seq_len * kv_dim);

        /* 6. Causal GQA attention — prefill uses f32 Q/K/V directly (not from cache)
         * since we just computed them. This avoids bf16 roundtrip during prefill. */
        float scale = 1.0f / sqrtf((float)c->head_dim);
        qwen_causal_attention(pref_attn_out, pref_q, pref_k, pref_v,
                              n_new, seq_len, c->num_heads, c->num_kv_heads,
                              c->head_dim, scale, pos0);

        /* 7. Output projection + residual */
        if (use_matmat) {
            prefill_proj_matmat(pref_proj, PREFW(l, wo), pref_attn_out, n_new, q_dim, h, pp_xT, pp_yT);
            for (int64_t i = 0; i < (int64_t)n_new * h; i++)
                residual[i] += pref_proj[i];
        } else {
#ifdef USE_BLAS
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    n_new, h, q_dim, 1.0f,
                    pref_attn_out, q_dim, wo_f32, q_dim, 0.0f, pref_proj, h);
        for (int64_t i = 0; i < (int64_t)n_new * h; i++)
            residual[i] += pref_proj[i];
#else
        for (int s = 0; s < n_new; s++) {
            float *xs = residual + (int64_t)s * h;
            const float *attn = pref_attn_out + (int64_t)s * q_dim;
            for (int o = 0; o < h; o++) {
                float sum = 0.0f;
                const float *row = wo_f32 + (int64_t)o * q_dim;
                for (int i = 0; i < q_dim; i++) sum += row[i] * attn[i];
                xs[o] += sum;
            }
        }
#endif
        }

        /* 8. Post-attention RMSNorm */
        qwen_rms_norm(pref_x_norm, residual, l->post_attn_norm, n_new, h, eps);

        /* 9. SwiGLU FFN (fused gate+up: interleaved [g0,u0,g1,u1,...]) */
        if (use_matmat) {
            prefill_proj_matmat(pref_gate, PREFW(l, gate_up_fused), pref_x_norm, n_new, h, 2 * inter, pp_xT, pp_yT);
        } else {
#ifdef USE_BLAS
        /* Single sgemm: output is [n_new, 2*inter] interleaved */
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    n_new, 2 * inter, h, 1.0f,
                    pref_x_norm, h, gate_up_f32, h, 0.0f, pref_gate, 2 * inter);
#else
        for (int s = 0; s < n_new; s++) {
            const float *xs = pref_x_norm + (int64_t)s * h;
            float *out = pref_gate + (int64_t)s * 2 * inter;
            for (int o = 0; o < 2 * inter; o++) {
                float sum = 0.0f;
                const float *row = gate_up_f32 + (int64_t)o * h;
                for (int i = 0; i < h; i++) sum += row[i] * xs[i];
                out[o] = sum;
            }
        }
#endif
        }

        /* SiLU(gate) * up on interleaved pairs, compact to stride=inter.
         * Uses batch vvexpf on macOS for faster exp. */
        for (int s = 0; s < n_new; s++) {
            float *src = pref_gate + (int64_t)s * 2 * inter;
            float *dst = pref_gate + (int64_t)s * inter;
            qwen_swiglu_inplace(src, ctx->swiglu_tmp, inter);
            /* swiglu_inplace writes result to src[0..inter-1], copy to compacted dst */
            if (dst != src)
                memcpy(dst, src, inter * sizeof(float));
        }

        /* Down projection + residual (compacted: lda=inter) */
        if (use_matmat) {
            prefill_proj_matmat(pref_proj, l->down_bf16, pref_gate, n_new, inter, h, pp_xT, pp_yT);
            for (int64_t i = 0; i < (int64_t)n_new * h; i++)
                residual[i] += pref_proj[i];
        } else {
#ifdef USE_BLAS
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    n_new, h, inter, 1.0f,
                    pref_gate, inter, down_f32, inter, 0.0f, pref_proj, h);
        for (int64_t i = 0; i < (int64_t)n_new * h; i++)
            residual[i] += pref_proj[i];
#else
        for (int s = 0; s < n_new; s++) {
            float *xs = residual + (int64_t)s * h;
            const float *gs = pref_gate + (int64_t)s * inter;
            for (int o = 0; o < h; o++) {
                float sum = 0.0f;
                const float *row = down_f32 + (int64_t)o * inter;
                for (int i = 0; i < inter; i++) sum += row[i] * gs[i];
                xs[o] += sum;
            }
        }
#endif
        }

        if (ctx->debug) {
            fprintf(stderr, "  Layer %d/%d done", layer + 1, c->num_layers);
            /* Print first position residual to detect NaN */
            fprintf(stderr, " res[:4]=[%.4f,%.4f,%.4f,%.4f]",
                    residual[0], residual[1], residual[2], residual[3]);
            fprintf(stderr, "\n");
        }
    }

    if (pfx_saving) {
        fill->len = ctx->pfx_len; fill->n_layers = c->num_layers; fill->kv_dim = kv_dim;
        fill->speaker_id = ctx->pfx_spk; fill->language_id = ctx->pfx_lang;
        fill->think_mode = ctx->pfx_think; fill->ihash = ctx->pfx_ihash;
        fill->model_tag = (const void *)ctx->layers;
        /* Release: everything above (both memcpy'd tensors and every key field) is
         * visible to any thread that later reads PFX_READY with acquire. */
        atomic_store_explicit(&g_pfx_state[fill - g_pfx], PFX_READY, memory_order_release);
        fprintf(stderr, "  Prefix cache FILLED slot %d: %d positions (spk %d, lang %d), %.2f MB\n",
                (int)(fill - g_pfx), fill->len, fill->speaker_id, fill->language_id,
                (double)c->num_layers * fill->len * kv_dim * sizeof(float) * 2 / 1048576.0);
    }

    ctx->kv_len = seq_len;

    if (ctx->debug) {
        /* Debug: print last position hidden state before and after norm */
        float *last_pos = residual + (int64_t)(n_new - 1) * h;
        fprintf(stderr, "[PREFILL] last_hidden[:8]:");
        for (int j = 0; j < 8 && j < h; j++) fprintf(stderr, " %.6f", last_pos[j]);
        fprintf(stderr, "\n");
        /* Apply norm temporarily for debug */
        float *normed_tmp = (float *)malloc(h * sizeof(float));
        qwen_rms_norm(normed_tmp, last_pos, ctx->talker_norm, 1, h, c->rms_norm_eps);
        fprintf(stderr, "[PREFILL] after_norm[:8]:");
        for (int j = 0; j < 8 && j < h; j++) fprintf(stderr, " %.6f", normed_tmp[j]);
        fprintf(stderr, "\n");
        free(normed_tmp);
    }

    /* Copy last position to dec_x for use in generation */
    memcpy(ctx->dec_x, residual + (int64_t)(n_new - 1) * h, h * sizeof(float));

    /* Buffers persist in ctx for reuse across generations (server mode) */

    if (!ctx->silent) fprintf(stderr, "  Prefill complete (%d tokens in KV cache)\n", seq_len);
    if (trace)
        fprintf(stderr, "PREFILLMS %.1f positions=%d computed=%d reused=%d\n",
                pfx_now_ms() - pfx_t0, seq_len, n_new, pos0);
    return 0;
}

/* ========================================================================
 * OPT-IN BATCHED Talker step (feat/batching) — see qwen_tts_batch.h.
 * ADDITIVE: does not touch qwen_talker_step above. Reuses the per-vector
 * kernels looped over B; batches ONLY the matvecs via qwen_matmat_bf16.
 * v1: bf16 weights, B sequences in lockstep. Layout: activations [B][dim];
 * the matmat wants [dim][B] so we gather/scatter around each call.
 * ======================================================================== */

/* gather src[B][dim] (row b at b*srcstride) -> Xt[dim][B] */
/* gather src[idx[j]][dim] -> Xt[dim][n]  (idx NULL = identity 0..n-1)
 *
 * `idx` is what makes the batched path follow the ACTIVE slots instead of the allocated
 * width (PLAN T5.piano 2b/3). Columns are packed contiguously for the matmat and put
 * back where they belong on the way out, so any pattern of idle slots costs nothing —
 * including holes in the middle, which the high-water mark alone could not remove.
 * Nothing here is ISA-specific: it changes WHICH columns the GEMM sees, and every
 * matmat twin (NEON/SDOT, AVX2, AVX-512/VNNI, scalar) benefits identically. */
static void batch_gather(float *Xt, const float *src, int n, int dim, int srcstride,
                         const int *idx) {
    for (int j = 0; j < n; j++) {
        const float *s = src + (size_t)(idx ? idx[j] : j) * srcstride;
        for (int k = 0; k < dim; k++) Xt[(size_t)k * n + j] = s[k];
    }
}
/* scatter Yt[rows][n] -> dst[idx[j]][rows] */
static void batch_scatter(float *dst, const float *Yt, int n, int rows, const int *idx) {
    for (int r = 0; r < rows; r++) {
        const float *yr = Yt + (size_t)r * n;
        for (int j = 0; j < n; j++) dst[(size_t)(idx ? idx[j] : j) * rows + r] = yr[j];
    }
}

/* Batched projection dst[B][rows] = W @ src[B][cols] (src row b at b*srcstride).
 * Default: one batched matmat (weights read once). force_matvec (or QWEN_BATCH_NOMATMUL=1)
 * falls back to B per-column matvecs — bit-matches single-stream; a diagnostic to isolate
 * the matmat from the wiring. Xt/Yt are caller-provided scratch ([cols*B] / [rows*B]).
 * Shared by the batched Talker AND Code Predictor (different dims). */
static atomic_int g_batch_nomatmul = -1;  /* audit #10: race-free one-time env cache */
/* ONE COLUMN GOES THROUGH THE MATVEC, and that is not a fallback — it is the right kernel.
 * Every matmat twin declares a min_b (qwen_tts_kernels.c g_mm_gate): 2 for VNNI / AVX2 /
 * SMMLA / SDOT / BFMMLA, and 4 for AMX. So a one-column matmat fires NONE of them and
 * lands on the generic tail, while the single-column kernel — the one the CLI uses all
 * day — is qwen_matvec_*.
 *
 * Measured 2026-08-20 on Emerald Rapids, and it is why this exists: making the codec head
 * honest at one active slot (act_idx/B_eff packed by the solo shortcut) narrowed it from
 * B_eff=4 to B_eff=1 and cost 1.8x on a solo request (service_s 9.3 -> 16.5 s) — the head
 * had been riding AMX only because B_eff was stale at the allocated width. Correctness
 * wants the narrow width; speed wants the kernel that suits it. Both, here. */
void qwen_batch_proj(float *dst, const uint16_t *W, const float *src,
                     int rows, int cols, int srcstride, int B, const int *idx,
                     int force_matvec, float *Xt, float *Yt) {
    int nomatmul = atomic_load_explicit(&g_batch_nomatmul, memory_order_relaxed);
    if (nomatmul < 0) {
        nomatmul = getenv("QWEN_BATCH_NOMATMUL") ? 1 : 0;
        atomic_store_explicit(&g_batch_nomatmul, nomatmul, memory_order_relaxed);
    }
    if (nomatmul || force_matvec || B == 1) {
        for (int j = 0; j < B; j++) {
            int b = idx ? idx[j] : j;
            qwen_matvec_bf16(dst + (size_t)b * rows, W, src + (size_t)b * srcstride, rows, cols);
        }
    } else {
        batch_gather(Xt, src, B, cols, srcstride, idx);
        qwen_matmat_bf16(Yt, W, Xt, rows, cols, B);
        batch_scatter(dst, Yt, B, rows, idx);
    }
}
/* thin wrapper for the Talker step. Width is bb->B_eff, NOT bb->B: the columns above
 * the highest active slot are pure waste, and on M1 that waste is not free (T5.piano). */
static void batch_proj(qwen_batch_t *bb, float *dst, const uint16_t *W,
                       const float *src, int rows, int cols, int srcstride) {
    qwen_batch_proj(dst, W, src, rows, cols, srcstride,
                    bb->B_eff > 0 ? bb->B_eff : bb->B, bb->act_idx,
                    bb->force_matvec, bb->Xt, bb->Yt);
}

/* Bytes of WEIGHTS one pass over this projection pulls from memory, by precision.
 * Exact for what it counts (q4_0 carries its scale inside the block; int8 carries one
 * f32 scale per row), and deliberately blind to activations and KV — see the header. */
static long long qwen_proj_weight_bytes(const uint16_t *Wb, const int8_t *Wi,
                                        const q4_0_block_t *Wq, int rows, int cols) {
    long long n = (long long)rows * (long long)cols;
    if (Wq) return (n / 32) * (long long)sizeof(q4_0_block_t);
    if (Wi) return n + (long long)rows * 4;
    (void)Wb; return n * 2;
}

/* Precision-aware batched projection (B2): picks the kernel by which weight set is
 * present, mirroring the single-stream dispatch — q4 (Wq) > int8 (Wi+Wscale) > bf16
 * (Wb). The batched int8/int4 twins (qwen_matmat_int8/q4_0) read the weight ONCE for
 * all B chunks; int4 amortizes the nibble unpack (the M1 lever). force_matvec /
 * QWEN_BATCH_NOMATMUL falls back to B per-column matvecs (bit-exact diagnostic),
 * still precision-correct. Shared by the batched Talker AND Code Predictor. */
void qwen_batch_proj_q(float *dst,
                       const uint16_t *Wb, const int8_t *Wi, const float *Wscale,
                       const q4_0_block_t *Wq,
                       const float *src, int rows, int cols, int srcstride,
                       int B, const int *idx, int force_matvec, float *Xt, float *Yt) {
    if (g_batch_nomatmul < 0) g_batch_nomatmul = getenv("QWEN_BATCH_NOMATMUL") ? 1 : 0;
    /* B == 1 -> matvec, for the min_b reason spelled out above qwen_batch_proj. */
    if (force_matvec || g_batch_nomatmul || B == 1) {
        /* audit (PLAN 0.nonies S10): this is the deliberate bit-exact reference path —
         * B separate matvecs, weights read B times. Counted so a curve taken with it on
         * by accident cannot be mistaken for the machine's batching behaviour. */
        if (qwen_matmat_stats_enabled() && !(force_matvec || g_batch_nomatmul)) {
            /* one column by width, not by request: counted as SOLO so a report cannot
             * read it as "somebody left the bit-exact diagnostic path on". */
            qwen_matmat_stats_note(QWEN_MMK_SOLO, (long long)rows * cols);
            qwen_matmat_stats_note_bytes(qwen_proj_weight_bytes(Wb, Wi, Wq, rows, cols));
        } else if (qwen_matmat_stats_enabled()) {
            qwen_matmat_stats_note(QWEN_MMK_FORCED_MATVEC, (long long)rows * cols * B);
            /* B separate matvecs: the weight block is streamed once PER COLUMN. That
             * factor B is the whole difference this counter exists to show. */
            qwen_matmat_stats_note_bytes(qwen_proj_weight_bytes(Wb, Wi, Wq, rows, cols) * B);
        }
        for (int j = 0; j < B; j++) {
            int b = idx ? idx[j] : j;
            const float *s = src + (size_t)b * srcstride; float *d = dst + (size_t)b * rows;
            if (Wq)      qwen_matvec_q4_0(d, Wq, s, rows, cols);
            else if (Wi) qwen_matvec_int8(d, Wi, Wscale, s, rows, cols);
            else         qwen_matvec_bf16(d, Wb, s, rows, cols);
        }
    } else {
        /* batched: the weight block is read ONCE and reused across the B columns —
         * so the byte count does NOT carry a factor B, whatever B is. */
        if (qwen_matmat_stats_enabled())
            qwen_matmat_stats_note_bytes(qwen_proj_weight_bytes(Wb, Wi, Wq, rows, cols));
        /* ── native layout: no gather, no scatter ────────────────────────────────
         * `src` is [slot][cols] and `dst` is [slot][rows] - already the layout
         * KleidiAI consumes. batch_gather below TRANSPOSES into [cols][slot] purely
         * because our own kernels want that, and batch_scatter undoes it. Measured
         * on Neoverse V2 those two cost as much as the matmul itself at B=16, and
         * they also leave the kernel on a cold cache: the SAME KleidiAI kernel ran
         * 27.8 us native and 57.6 us through the transposes.
         *
         * The slots must be contiguous and ascending for the packer's single
         * (pointer, stride) pair to describe them. That is the normal case - both
         * orchestrators seed slots in order - and when it is not, the code falls
         * through to the transposing path rather than guessing. */
        int contig = 1;
        if (idx) for (int j = 0; j < B; j++) if (idx[j] != j) { contig = 0; break; }
        if (contig) {
            /* Census FIRST: a native path that skips the dispatcher also skips the
             * shape census, and the B histogram then shows only the calls that fell
             * back - which reads as "the server never batches" when it does. */
            if (Wi) qwen_census_op("matmat_int8_native", rows, cols, B);
            else if (!Wq) qwen_census_op("matmat_bf16_native", rows, cols, B);
            if (Wi && qwen_kleidi_matmul_i8_native(dst, Wi, src,
                                                   (size_t)srcstride * sizeof(float),
                                                   (size_t)rows * sizeof(float),
                                                   rows, cols, B)) {
                if (qwen_matmat_stats_enabled() || qwen_census_enabled())
                    qwen_matmat_stats_note(B > 1 ? QWEN_MMK_KLEIDI_I8 : QWEN_MMK_KLEIDI_I8_GEMV,
                                           (long long)rows * cols * B);
                return;
            }
            if (!Wq && !Wi && qwen_kleidi_matmul_bf16_native(dst, Wb, src,
                                                   (size_t)srcstride * sizeof(float),
                                                   (size_t)rows * sizeof(float),
                                                   rows, cols, B)) {
                if (qwen_matmat_stats_enabled() || qwen_census_enabled())
                    qwen_matmat_stats_note(B > 1 ? QWEN_MMK_KLEIDI_BF16 : QWEN_MMK_KLEIDI_BF16_GEMV,
                                           (long long)rows * cols * B);
                return;
            }
        }
        batch_gather(Xt, src, B, cols, srcstride, idx);
        if (Wq)      qwen_matmat_q4_0(Yt, Wq, Xt, rows, cols, B);
        else if (Wi) qwen_matmat_int8(Yt, Wi, Wscale, Xt, rows, cols, B);
        else         qwen_matmat_bf16(Yt, Wb, Xt, rows, cols, B);
        batch_scatter(dst, Yt, B, rows, idx);
    }
}
/* ── One QKV, one pack, one barrier — the B>=2 twin of qwen_kleidi_matmul_i8_qkv ──
 * Q, K and V read the SAME activation. Called through qwen_batch_proj_q they pack it
 * three times and pay three fork-join barriers; measured on the batched server that is
 * 26.0% of all dispatches (the design notes.
 *
 * Declines - and falls back to exactly the three calls it replaces - unless every
 * condition the fused kernel needs holds: int8 weights, no q4, batching actually on,
 * contiguous ascending slots (the packer describes them with ONE pointer+stride), and
 * B > 1. The fallback is not a degraded path, it IS the previous behaviour. */
void qwen_batch_proj_qkv(float *dq, float *dk, float *dv,
                         const uint16_t *Wqb, const int8_t *Wqi, const float *Wqs,
                         const q4_0_block_t *Wqq,
                         const uint16_t *Wkb, const int8_t *Wki, const float *Wks,
                         const q4_0_block_t *Wkq,
                         const uint16_t *Wvb, const int8_t *Wvi, const float *Wvs,
                         const q4_0_block_t *Wvq,
                         const float *src, int q_rows, int kv_rows, int cols,
                         int srcstride, int B, const int *idx, int force_matvec,
                         float *Xt, float *Yt) {
    if (g_batch_nomatmul < 0) g_batch_nomatmul = getenv("QWEN_BATCH_NOMATMUL") ? 1 : 0;
    int contig = 1;
    if (idx) for (int j = 0; j < B; j++) if (idx[j] != j) { contig = 0; break; }
    if (B > 1 && contig && !force_matvec && !g_batch_nomatmul &&
        Wqi && Wki && Wvi && !Wqq && !Wkq && !Wvq) {
        /* Census AFTER success, NOT before. qwen_batch_proj_q records the census first
         * because nothing downstream of it will; here the fallback below is three
         * qwen_batch_proj_q calls that record themselves. Recording first therefore
         * DOUBLE-COUNTS every QKV whenever the fused path declines - which is exactly
         * the QWEN_KAI_QKV_FUSED=0 arm, i.e. the baseline an A/B is measured against.
         * Caught on 2026-08-24 by the OFF arm reporting a qkv_native row it could not
         * have executed. */
        if (qwen_kleidi_matmul_i8_qkv_native(dq, dk, dv, Wqi, Wki, Wvi, src,
                                             (size_t)srcstride * sizeof(float),
                                             cols, q_rows, kv_rows, B)) {
            if (qwen_census_enabled())
                qwen_census_op("matmat_int8_qkv_native", q_rows + 2 * kv_rows, cols, B);
            if (qwen_matmat_stats_enabled() || qwen_census_enabled()) {
                qwen_matmat_stats_note(QWEN_MMK_KLEIDI_I8,
                                       (long long)(q_rows + 2 * kv_rows) * cols * B);
                /* The three weight blocks are still read once each; what the fusion
                 * removes is two packs and two barriers, not bytes. */
                qwen_matmat_stats_note_bytes(
                    qwen_proj_weight_bytes(Wqb, Wqi, Wqq, q_rows, cols) +
                    qwen_proj_weight_bytes(Wkb, Wki, Wkq, kv_rows, cols) +
                    qwen_proj_weight_bytes(Wvb, Wvi, Wvq, kv_rows, cols));
            }
            return;
        }
    }
    qwen_batch_proj_q(dq, Wqb, Wqi, Wqs, Wqq, src, q_rows,  cols, srcstride, B, idx,
                      force_matvec, Xt, Yt);
    qwen_batch_proj_q(dk, Wkb, Wki, Wks, Wkq, src, kv_rows, cols, srcstride, B, idx,
                      force_matvec, Xt, Yt);
    qwen_batch_proj_q(dv, Wvb, Wvi, Wvs, Wvq, src, kv_rows, cols, srcstride, B, idx,
                      force_matvec, Xt, Yt);
}

/* bb-scratch wrapper for the Talker step. */
static void batch_proj_q(qwen_batch_t *bb, float *dst,
                         const uint16_t *Wb, const int8_t *Wi, const float *Wscale,
                         const q4_0_block_t *Wq,
                         const float *src, int rows, int cols, int srcstride) {
    qwen_batch_proj_q(dst, Wb, Wi, Wscale, Wq, src, rows, cols, srcstride,
                      bb->B_eff > 0 ? bb->B_eff : bb->B, bb->act_idx,
                      bb->force_matvec, bb->Xt, bb->Yt);
}

/* QWEN_BATCH_NO_BEFF=1 pins the projections back to the allocated B — the "before" of
 * T5.piano #2, so the win can be A/B'd on any box without a rebuild. Same shape of
 * switch as QWEN_BATCH_NO_SOLO, and it must exist on every ISA for the same reason:
 * the size of the win is machine-dependent and has to be measured where it runs. */
/* See qwen_tts_batch.h for the invariant this exists to hold, and what breaking it did. */
void qwen_batch_pack_active(qwen_batch_t *bb, const uint8_t *active) {
    int n = 0;
    if (active && !qwen_batch_beff_disabled()) {
        /* Pack the ACTIVE slots, not the high-water mark: a slot freed in the middle
         * would otherwise keep paying, and both orchestrators seed every slot up front
         * and drain them in arbitrary order, so middle holes are the normal case. */
        for (int b = 0; b < bb->B; b++) if (active[b]) bb->act_idx[n++] = b;
        if (n < 1) { bb->act_idx[0] = 0; n = 1; }
    } else {
        for (int b = 0; b < bb->B; b++) bb->act_idx[b] = b;
        n = bb->B;
    }
    bb->B_eff = n;
}

int qwen_batch_beff_disabled(void) {
    static atomic_int off = -1;
    int v = atomic_load_explicit(&off, memory_order_relaxed);
    if (v < 0) {
        const char *e = getenv("QWEN_BATCH_NO_BEFF");
        v = (e && e[0] == '1');
        atomic_store_explicit(&off, v, memory_order_relaxed);
    }
    return v;
}

int qwen_batch_solo_disabled(void) {
    static atomic_int off = -1;
    int v = atomic_load_explicit(&off, memory_order_relaxed);
    if (v < 0) {
        const char *e = getenv("QWEN_BATCH_NO_SOLO");
        v = (e && e[0] == '1');
        atomic_store_explicit(&off, v, memory_order_relaxed);
    }
    return v;
}

qwen_batch_t *qwen_batch_alloc(qwen_tts_ctx_t *ctx, int B, int kv_max) {
    qwen_tts_config_t *c = &ctx->config;
    if (B < 1 || B > 64 || kv_max < 1) return NULL;
    /* The batched projection (qwen_batch_proj_q) has been precision-aware for a while:
     * q4 > int8 > bf16. So the gate is "is there ANY weight set this path can read",
     * not "is there a bf16 one" — otherwise freeing the bf16 (S3) would stop the
     * batched server from starting even though every kernel it uses is quantized.
     * (q6 has no batched kernel, which is what qwen_talker_has_q6 is for.) */
    if (ctx->layers[0].wq_bf16 == NULL && ctx->layers[0].wq_int8 == NULL &&
        ctx->layers[0].wq_q4 == NULL) return NULL;
    qwen_batch_t *bb = (qwen_batch_t *)calloc(1, sizeof(qwen_batch_t));
    if (!bb) return NULL;
    bb->B = B; bb->h = c->hidden_size; bb->q_dim = c->num_heads * c->head_dim;
    bb->kv_dim = c->num_kv_heads * c->head_dim; bb->inter = c->intermediate_size;
    bb->num_layers = c->num_layers; bb->kv_max = kv_max; bb->kv_len = 0;
    /* Full width until a step computes the active high-water mark: callers that never
     * pass an `active` mask (bench, self-test) must keep running at B. */
    bb->B_eff = B;
    for (int b = 0; b < B; b++) bb->act_idx[b] = b;
    int h = bb->h, qd = bb->q_dim, kvd = bb->kv_dim, inter = bb->inter;
    int maxrows = 2 * inter; if (qd > maxrows) maxrows = qd; if (h > maxrows) maxrows = h;
    int maxcols = h; if (qd > maxcols) maxcols = qd; if (inter > maxcols) maxcols = inter;
#define A(n) (float *)aligned_calloc((size_t)(n), sizeof(float))
    bb->x = A(B * h); bb->x_norm = A(B * h); bb->q = A(B * qd);
    bb->k = A(B * kvd); bb->v = A(B * kvd); bb->attn_out = A(B * qd);
    bb->proj_out = A(B * h); bb->gate = A((size_t)B * 2 * inter); bb->swiglu_tmp = A(inter);
    bb->Xt = A((size_t)maxcols * B); bb->Yt = A((size_t)maxrows * B);
#undef A
    size_t kvN = (size_t)B * bb->num_layers * kv_max * kvd;
    bb->kv_k = (uint16_t *)aligned_calloc(kvN, sizeof(uint16_t));
    bb->kv_v = (uint16_t *)aligned_calloc(kvN, sizeof(uint16_t));

    /* ---- Code Predictor batched buffers (cp_kv_max = 64, matching single-stream) ---- */
    bb->cp_h = c->cp_hidden_size; bb->cp_q_dim = c->cp_num_heads * c->cp_head_dim;
    bb->cp_kv_dim = c->cp_num_kv_heads * c->cp_head_dim; bb->cp_inter = c->cp_intermediate_size;
    bb->cp_num_layers = c->cp_num_layers; bb->cp_kv_max = 64;
    int ch = bb->cp_h, cqd = bb->cp_q_dim, ckvd = bb->cp_kv_dim, cint = bb->cp_inter;
    int cmaxrows = 2 * cint; if (cqd > cmaxrows) cmaxrows = cqd; if (ch > cmaxrows) cmaxrows = ch;
    int cmaxcols = ch; if (cqd > cmaxcols) cmaxcols = cqd; if (cint > cmaxcols) cmaxcols = cint;
#define AC(n) (float *)aligned_calloc((size_t)(n), sizeof(float))
    bb->cp_x = AC(B * ch); bb->cp_x_norm = AC(B * ch); bb->cp_q = AC(B * cqd);
    bb->cp_k = AC(B * ckvd); bb->cp_v = AC(B * ckvd); bb->cp_attn = AC(B * cqd);
    bb->cp_proj = AC(B * ch); bb->cp_gate = AC((size_t)B * 2 * cint); bb->cp_swiglu_tmp = AC(cint);
    bb->cp_Xt = AC((size_t)cmaxcols * B); bb->cp_Yt = AC((size_t)cmaxrows * B);
#undef AC
    size_t ckvN = (size_t)B * bb->cp_num_layers * bb->cp_kv_max * ckvd;
    bb->cp_kv_k = (uint16_t *)aligned_calloc(ckvN, sizeof(uint16_t));
    bb->cp_kv_v = (uint16_t *)aligned_calloc(ckvN, sizeof(uint16_t));

    if (!bb->x || !bb->x_norm || !bb->q || !bb->k || !bb->v || !bb->attn_out ||
        !bb->proj_out || !bb->gate || !bb->swiglu_tmp || !bb->Xt || !bb->Yt ||
        !bb->kv_k || !bb->kv_v || !bb->cp_x || !bb->cp_q || !bb->cp_gate ||
        !bb->cp_Xt || !bb->cp_Yt || !bb->cp_kv_k || !bb->cp_kv_v) { qwen_batch_free(bb); return NULL; }
    return bb;
}

void qwen_batch_free(qwen_batch_t *bb) {
    if (!bb) return;
    free(bb->x); free(bb->x_norm); free(bb->q); free(bb->k); free(bb->v);
    free(bb->attn_out); free(bb->proj_out); free(bb->gate); free(bb->swiglu_tmp);
    free(bb->Xt); free(bb->Yt); free(bb->kv_k); free(bb->kv_v);
    free(bb->cp_x); free(bb->cp_x_norm); free(bb->cp_q); free(bb->cp_k); free(bb->cp_v);
    free(bb->cp_attn); free(bb->cp_proj); free(bb->cp_gate); free(bb->cp_swiglu_tmp);
    free(bb->cp_Xt); free(bb->cp_Yt); free(bb->cp_kv_k); free(bb->cp_kv_v);
    free(bb);
}

/* Core batched Talker step. pos_arr[b] = each sequence's CURRENT position (where
 * its new K/V is written, and the RoPE/attention query position); NULL = lockstep
 * (all sequences at the shared bb->kv_len, as the bench/self-test use). active[b]=0
 * skips that sequence's per-seq work (finished/ragged-EOS); NULL = all active. The
 * batched matvecs still process all B rows (compaction is a later optimization);
 * inactive outputs are ignored by the caller. The caller advances pos_arr; in
 * lockstep mode this advances bb->kv_len by one. */
static int batch_talker_step_impl(qwen_tts_ctx_t *ctx, qwen_batch_t *bb,
                                  const float *embeds, const int *pos_arr,
                                  const uint8_t *active, float *hidden_out) {
    qwen_tts_config_t *c = &ctx->config;
    int B = bb->B, h = bb->h, qd = bb->q_dim, kvd = bb->kv_dim, inter = bb->inter;
    float eps = c->rms_norm_eps;
    if (ctx->layers[0].wq_bf16 == NULL) return -2;
    int maxpos = 0;
    for (int b = 0; b < B; b++) { int p = pos_arr ? pos_arr[b] : bb->kv_len; if (p > maxpos) maxpos = p; }
    if (maxpos + 1 > bb->kv_max) return -1;

    /* T5.piano #2 — the projections run at (highest active slot + 1), not at the
     * allocated B. Columns above that index belong to inactive slots and the
     * orchestrator discards their output, so computing them is work thrown away.
     *
     * ARCHITECTURE-NEUTRAL: wasted columns are wasted on every ISA. What varies is HOW
     * MUCH they cost, and that is set by the machine's bandwidth/compute ratio — the
     * premise in qwen_tts_batch.h ("near-free on bandwidth-bound ARM") is only true
     * where the kernel is fully bandwidth-bound, which is a property of the box, not of
     * the instruction set. Measured on Apple M1/NEON+SDOT; MUST be re-measured on
     * x86-64 with AVX2-only and with AVX-512/VNNI, and on arm64 Linux (Graviton/GB10),
     * because those have different ratios and different matmat kernels. See
     * PLAN "Da validare sul metallo x86/arm64".
     *
     * Safe by construction on any ISA: rows [0, B_eff) are the leading rows of the same
     * [B][dim] buffers, so nothing is re-laid out and inactive slots keep their stale
     * (ignored) values. Idle slots BELOW the high-water mark still cost — closing those
     * needs slot compaction (#2b), which must move a KV block. */
    qwen_batch_pack_active(bb, active);
    #define POS_B(b) (pos_arr ? pos_arr[b] : bb->kv_len)
    #define ACTIVE_B(b) (!active || active[b])
    memcpy(bb->x, embeds, (size_t)B * h * sizeof(float));
    float scale = 1.0f / sqrtf((float)c->head_dim);

    for (int layer = 0; layer < c->num_layers; layer++) {
        qwen_talker_layer_t *l = &ctx->layers[layer];
        /* 1. input RMSNorm (per sequence) */
        for (int b = 0; b < B; b++)
            qwen_rms_norm(bb->x_norm + (size_t)b * h, bb->x + (size_t)b * h, l->input_norm, 1, h, eps);
        /* 2. QKV (batched, precision-aware) */
        qwen_batch_proj_qkv(bb->q, bb->k, bb->v,
                            l->wq_bf16, l->wq_int8, l->wq_scale, l->wq_q4,
                            l->wk_bf16, l->wk_int8, l->wk_scale, l->wk_q4,
                            l->wv_bf16, l->wv_int8, l->wv_scale, l->wv_q4,
                            bb->x_norm, qd, kvd, h, h,
                            bb->B_eff > 0 ? bb->B_eff : bb->B, bb->act_idx,
                            bb->force_matvec, bb->Xt, bb->Yt);
        /* 3-5. per-head norm, RoPE, append KV — per sequence (at its own position) */
        for (int b = 0; b < B; b++) {
            if (!ACTIVE_B(b)) continue;
            int pos = POS_B(b);
            qwen_rms_norm_per_head(bb->q + (size_t)b * qd,  l->q_norm, 1, c->num_heads,    c->head_dim, eps);
            qwen_rms_norm_per_head(bb->k + (size_t)b * kvd, l->k_norm, 1, c->num_kv_heads, c->head_dim, eps);
            apply_rope_neox_inplace(bb->q + (size_t)b * qd,  c->num_heads,    c->head_dim, ctx->rope_cos, ctx->rope_sin, pos);
            apply_rope_neox_inplace(bb->k + (size_t)b * kvd, c->num_kv_heads, c->head_dim, ctx->rope_cos, ctx->rope_sin, pos);
            size_t kvbase = ((size_t)b * bb->num_layers + layer) * bb->kv_max * kvd + (size_t)pos * kvd;
            f32_to_bf16_vec(bb->kv_k + kvbase, bb->k + (size_t)b * kvd, kvd);
            f32_to_bf16_vec(bb->kv_v + kvbase, bb->v + (size_t)b * kvd, kvd);
        }
        /* 6. causal GQA attention — per sequence, against its own KV (length pos+1) */
        for (int b = 0; b < B; b++) {
            if (!ACTIVE_B(b)) continue;
            int pos = POS_B(b);
            size_t lbase = ((size_t)b * bb->num_layers + layer) * bb->kv_max * kvd;
            qwen_causal_attention_bf16kv(bb->attn_out + (size_t)b * qd, bb->q + (size_t)b * qd,
                                         bb->kv_k + lbase, bb->kv_v + lbase, 1, pos + 1,
                                         c->num_heads, c->num_kv_heads, c->head_dim, scale, pos);
        }
        /* 7. O projection (batched, precision-aware) */
        batch_proj_q(bb, bb->proj_out, l->wo_bf16, l->wo_int8, l->wo_scale, l->wo_q4, bb->attn_out, h, qd, qd);
        /* 8. residual + post-attn RMSNorm (per seq; x += proj_out in place) */
        for (int b = 0; b < B; b++)
            qwen_rms_norm_residual(bb->x_norm + (size_t)b * h, bb->x + (size_t)b * h,
                                   bb->proj_out + (size_t)b * h, l->post_attn_norm, h, eps);
        /* 9. gate+up (batched, precision-aware) + SwiGLU per seq */
        batch_proj_q(bb, bb->gate, l->gate_up_fused_bf16, l->gate_up_fused_int8, l->gate_up_fused_scale,
                          l->gate_up_fused_q4, bb->x_norm, 2 * inter, h, h);
        for (int b = 0; b < B; b++)
            qwen_swiglu_inplace(bb->gate + (size_t)b * 2 * inter, bb->swiglu_tmp, inter);
        /* 10. down (batched, precision-aware) — swiglu output is first `inter` of each 2*inter row */
        batch_proj_q(bb, bb->proj_out, l->down_bf16, l->down_int8, l->down_scale, l->down_q4,
                          bb->gate, h, inter, 2 * inter);
        /* 11. residual (+ next layer's input norm, or plain add on the last layer) */
        if (layer + 1 < c->num_layers) {
            for (int b = 0; b < B; b++)
                qwen_rms_norm_residual(bb->x_norm + (size_t)b * h, bb->x + (size_t)b * h,
                                       bb->proj_out + (size_t)b * h, ctx->layers[layer + 1].input_norm, h, eps);
        } else {
            for (int b = 0; b < B; b++) {
                float *xb = bb->x + (size_t)b * h, *pb = bb->proj_out + (size_t)b * h;
                for (int i = 0; i < h; i++) xb[i] += pb[i];
            }
        }
    }
    /* final RMSNorm per sequence */
    for (int b = 0; b < B; b++)
        qwen_rms_norm(hidden_out + (size_t)b * h, bb->x + (size_t)b * h, ctx->talker_norm, 1, h, eps);
    if (!pos_arr) bb->kv_len = bb->kv_len + 1;   /* lockstep advance */
    #undef POS_B
    #undef ACTIVE_B
    return 0;
}

/* Lockstep wrapper (bench/self-test): all B sequences share bb->kv_len. */
int qwen_batch_talker_step(qwen_tts_ctx_t *ctx, qwen_batch_t *bb,
                           const float *embeds, float *hidden_out) {
    return batch_talker_step_impl(ctx, bb, embeds, NULL, NULL, hidden_out);
}

/* Ragged wrapper (orchestrator): each sequence at its own pos_arr[b]; active[b]=0
 * skips finished sequences. The caller advances pos_arr[b] for active sequences. */
int qwen_batch_talker_step_ragged(qwen_tts_ctx_t *ctx, qwen_batch_t *bb,
                                  const float *embeds, const int *pos_arr,
                                  const uint8_t *active, float *hidden_out) {
#ifdef QWEN_HAVE_CUDA
    /* GPU batched path: the resident device KV is authoritative (seeded per slot on admit).
     * Processes all B rows (inactive slots' outputs are ignored by the orchestrator). */
    extern void *g_cuda_talker_batch_state;
    if (g_cuda_talker_batch_state) {
        qwen_cuda_talker_batch_step(g_cuda_talker_batch_state, embeds, pos_arr, hidden_out);
        return 0;
    }
#endif
#ifdef QWEN_HAVE_METAL
    if (g_metal_talker_batch_state) {
        qwen_metal_talker_batch_step(g_metal_talker_batch_state, embeds, pos_arr, hidden_out);
        return 0;
    }
#endif
    /* ── B_eff == 1: run the ORDINARY single-stream step ────────────────────────
     *
     * WHY. The batched projections run at bb->B width: with one slot occupied out of
     * four, three columns are computed and thrown away. The header calls that
     * "near-free on bandwidth-bound ARM" — true only where the kernel is FULLY
     * bandwidth-bound, which is a property of the machine (bandwidth/compute ratio),
     * not of the ISA. Wherever it is not, the waste is paid in time on ARM and on x86
     * alike. Measured on M1: -45% RTF at concurrency 1 once this shortcut is taken.
     * ⚠️ To re-measure on x86-64 (AVX2-only and AVX-512/VNNI) and arm64 Linux: the
     * batched matmat twins differ per ISA, so the size of the win differs too.
     *
     * A server spends most of its life below full occupancy, so this is the common
     * case, not the corner. And the fix needs no new kernel: bb's KV is laid out
     * [B][layers][kv_max][kv_dim], so ONE slot's region is contiguous and has exactly
     * the single-stream layout — point ctx at it, take the step that the CLI already
     * takes (with its fast matvec path), then put ctx back.
     *
     * ⚠️ REPRODUCIBILITY CAVEAT, stated because it is real: batched matmat and matvec
     * are valid fp-order variants of each other, so a sequence that starts at B_eff=2
     * and continues at B_eff=1 changes fp order mid-generation — its output then
     * depends on WHEN its neighbours finished. Set QWEN_BATCH_NO_SOLO=1 to keep every
     * step on the batched path when bit-reproducibility under concurrency matters more
     * than latency.
     * ⚠️ Scheduler thread only: this borrows ctx's KV fields, as the server already
     * does for speaker_id per admission. */
    if (!qwen_batch_solo_disabled()) {
        int n_act = 0, only = -1;
        if (active) {
            for (int b = 0; b < bb->B; b++) if (active[b]) { n_act++; only = b; }
        } else {
            n_act = bb->B; only = 0;
        }
        if (n_act == 1) {
            /* audit: the single-slot shortcut is BY DESIGN, not a fallback — but it has to
             * appear in the report, or a server that spends its life at one active slot
             * would look like it had no batched work at all. MACs are the step's projection
             * count (talker only; the CP's own solo shortcut is not counted). */
            if (qwen_matmat_stats_enabled()) {
                const qwen_tts_config_t *cc = &ctx->config;
                long long hh = cc->hidden_size, qd2 = (long long)cc->num_heads * cc->head_dim;
                long long kvd2 = (long long)cc->num_kv_heads * cc->head_dim, it2 = cc->intermediate_size;
                long long per_layer = qd2 * hh + 2 * kvd2 * hh + hh * qd2 + 2 * it2 * hh + hh * it2;
                qwen_matmat_stats_note(QWEN_MMK_SOLO, per_layer * cc->num_layers);
                /* Same weights, one column: the single-slot step reads the whole Talker
                 * once per frame. Counting it keeps BW_utilization honest at B=1, which
                 * is the baseline every BatchEfficiency is divided by.
                 * Summed PER LAYER and per tensor, not from one global precision flag:
                 * with a mixed map (--quant-mixed-int6) layer 3 can be int8 while layer
                 * 4 is q4, and a single flag would misreport the traffic by ~2x. */
                long long wb_solo = 0;
                for (int li = 0; li < cc->num_layers; li++) {
                    const qwen_talker_layer_t *L = &ctx->layers[li];
                    struct { long long n; const void *q6, *q4, *i8; } T[] = {
                        { qd2 * hh,      L->wq_q6, L->wq_q4, L->wq_int8 },
                        { kvd2 * hh,     L->wk_q6, L->wk_q4, L->wk_int8 },
                        { kvd2 * hh,     L->wv_q6, L->wv_q4, L->wv_int8 },
                        { hh * qd2,      L->wo_q6, L->wo_q4, L->wo_int8 },
                        { 2 * it2 * hh,  L->gate_up_fused_q6, L->gate_up_fused_q4, L->gate_up_fused_int8 },
                        { hh * it2,      L->down_q6, L->down_q4, L->down_int8 },
                    };
                    for (unsigned t = 0; t < sizeof(T) / sizeof(T[0]); t++) {
                        if      (T[t].q6) wb_solo += (T[t].n / 32) * (long long)sizeof(q6_0_block_t);
                        else if (T[t].q4) wb_solo += (T[t].n / 32) * (long long)sizeof(q4_0_block_t);
                        else if (T[t].i8) wb_solo += T[t].n + T[t].n / hh * 4;
                        else              wb_solo += T[t].n * 2;
                    }
                }
                qwen_matmat_stats_note_bytes(wb_solo);
            }
            size_t slot = (size_t)only * bb->num_layers * bb->kv_max * bb->kv_dim;
            uint16_t *sk = ctx->kv_cache_k, *sv = ctx->kv_cache_v;
            int smax = ctx->kv_max, slen = ctx->kv_len;
            ctx->kv_cache_k = bb->kv_k + slot;
            ctx->kv_cache_v = bb->kv_v + slot;
            ctx->kv_max = bb->kv_max;
            ctx->kv_len = pos_arr ? pos_arr[only] : bb->kv_len;
            int rc = qwen_talker_step(ctx, (float *)(uintptr_t)embeds + (size_t)only * bb->h,
                                      hidden_out + (size_t)only * bb->h);
            ctx->kv_cache_k = sk; ctx->kv_cache_v = sv;
            ctx->kv_max = smax; ctx->kv_len = slen;
            /* This step happened HERE, so act_idx/B_eff must say so: the frame loop reads
             * them for the codec head immediately after. Leaving them at whatever the last
             * batched step packed is what derailed generation on 2026-08-20. */
            qwen_batch_pack_active(bb, active);
            return rc;
        }
    }

    return batch_talker_step_impl(ctx, bb, embeds, pos_arr, active, hidden_out);
}

int qwen_batch_self_test(qwen_tts_ctx_t *ctx) {
    qwen_tts_config_t *c = &ctx->config; int h = c->hidden_size;
    if (ctx->layers[0].wq_bf16 == NULL) {
        fprintf(stderr, "batch-test: model is not bf16 (v1 batched path is bf16-only)\n");
        return 1;
    }
    const char *be = getenv("QWEN_BATCH_B");
    int B = be ? atoi(be) : 8; if (B < 1 || B > 64) B = 8;
    const int K = 8, kv_max = 64;
    qwen_batch_t *bb = qwen_batch_alloc(ctx, B, kv_max);
    if (!bb) { fprintf(stderr, "batch-test: alloc failed\n"); return 1; }
    float *embeds_all = (float *)malloc((size_t)K * h * sizeof(float));
    float *embedsB    = (float *)malloc((size_t)B * h * sizeof(float));
    float *href       = (float *)malloc((size_t)K * h * sizeof(float));
    float *hbat       = (float *)malloc((size_t)B * h * sizeof(float));
    uint64_t rng = 0xABCDEF123456789ull;
#define RF (((double)((rng = rng * 6364136223846793005ull + 1442695040888963407ull) >> 40)) / (double)(1u << 24) * 2.0 - 1.0)
    for (int i = 0; i < K * h; i++) embeds_all[i] = (float)(RF * 0.1);
    /* direct probe: matmat(B=1) vs matvec on the REAL layer-0 wq (is it fp-order or a bug?) */
    {
        int qd = c->num_heads * c->head_dim;
        float *Yt = (float *)malloc((size_t)qd * sizeof(float));
        float *yv = (float *)malloc((size_t)qd * sizeof(float));
        qwen_matmat_bf16(Yt, ctx->layers[0].wq_bf16, embeds_all, qd, h, 1);
        qwen_matvec_bf16(yv, ctx->layers[0].wq_bf16, embeds_all, qd, h);
        double mx = 0, l2n = 0, l2d = 0;
        for (int r = 0; r < qd; r++) { double d = (double)Yt[r] - yv[r]; if (fabs(d) > mx) mx = fabs(d);
            l2n += d * d; l2d += (double)yv[r] * yv[r]; }
        fprintf(stderr, "  probe wq matmat(B=1) vs matvec: max_abs=%.3e  L2_rel=%.3e\n", mx, l2d > 0 ? sqrt(l2n / l2d) : 0);
        free(Yt); free(yv);
    }
    /* single-stream reference (fresh KV) */
    int saved_kv = ctx->kv_len; ctx->kv_len = 0;
    for (int s = 0; s < K; s++) qwen_talker_step(ctx, embeds_all + (size_t)s * h, href + (size_t)s * h);
    ctx->kv_len = saved_kv;

    /* Run the K-step batched sequence (B identical chunks) in a given mode; return
     * the max per-step hidden L2 error vs the single-stream reference. */
    double err_matvec = 0.0, err_matmat = 0.0;
    for (int mode = 0; mode < 2; mode++) {
        bb->kv_len = 0; bb->force_matvec = (mode == 0);   /* mode 0 = matvec wiring check, 1 = real matmat */
        double maxl2 = 0.0;
        for (int s = 0; s < K; s++) {
            for (int b = 0; b < B; b++) memcpy(embedsB + (size_t)b * h, embeds_all + (size_t)s * h, h * sizeof(float));
            if (qwen_batch_talker_step(ctx, bb, embedsB, hbat) != 0) { fprintf(stderr, "batch-test: step failed\n"); break; }
            double l2n = 0, l2d = 0;
            for (int b = 0; b < B; b++) for (int i = 0; i < h; i++) {
                double d = (double)hbat[(size_t)b * h + i] - href[(size_t)s * h + i];
                l2n += d * d; l2d += (double)href[(size_t)s * h + i] * href[(size_t)s * h + i];
            }
            double l2 = l2d > 0 ? sqrt(l2n / l2d) : 0; if (l2 > maxl2) maxl2 = l2;
        }
        if (mode == 0) err_matvec = maxl2; else err_matmat = maxl2;
    }
    /* Correctness gate = the WIRING (matvec mode) must be bit-identical to single-stream.
     * The real matmat path diverges only by fp accumulation ORDER (6e-7/op, see the probe),
     * amplified through the 28-layer residual stream — a valid alternative kernel like int8,
     * to be validated end-to-end by audio mel-corr (not hidden bit-match). */
    int pass = err_matvec < 1e-5;
    fprintf(stderr, "batch-test: B=%d K=%d\n", B, K);
    fprintf(stderr, "  Talker wiring (matvec mode) vs single-stream: L2_rel=%.2e  %s (must be bit-exact)\n",
            err_matvec, err_matvec < 1e-5 ? "PASS" : "FAIL");
    fprintf(stderr, "  Talker batched matmat vs single-stream:       L2_rel=%.2e  (fp-order amplification, benign — validate via audio)\n",
            err_matmat);

    /* ---- batched Code Predictor: codes must match single-stream qwen_cp_predict ----
     * (greedy argmax → exact in matvec mode; matmat mode may flip a few argmaxes on
     * near-ties, reported as a code-disagreement rate.) */
    {
        int vocab = c->codec_vocab_size > 0 ? c->codec_vocab_size : 1024;
        float *th = (float *)malloc((size_t)B * h * sizeof(float));
        int   *c0 = (int *)malloc((size_t)B * sizeof(int));
        int   *ref = (int *)malloc((size_t)B * 15 * sizeof(int));
        int   *bat = (int *)malloc((size_t)B * 15 * sizeof(int));
        for (int i = 0; i < B * h; i++) th[i] = (float)(RF * 0.5);
        for (int b = 0; b < B; b++) c0[b] = (int)((RF * 0.5 + 0.5) * (vocab - 1));
        /* single-stream reference per frame */
        for (int b = 0; b < B; b++) qwen_cp_predict(ctx, th + (size_t)b * h, c0[b], ref + (size_t)b * 15);
        int cp_unsupported = 0, diff_mv = 0, diff_mm = 0;
        for (int mode = 0; mode < 2; mode++) {
            bb->force_matvec = (mode == 0);
            if (qwen_batch_cp_predict(ctx, bb, th, c0, bat, NULL) == -2) { cp_unsupported = 1; break; }
            int diff = 0;
            for (int i = 0; i < B * 15; i++) if (bat[i] != ref[i]) diff++;
            if (mode == 0) diff_mv = diff; else diff_mm = diff;
        }
        if (cp_unsupported) {
            fprintf(stderr, "  CP batched: skipped (non-bf16 model)\n");
        } else {
            fprintf(stderr, "  CP wiring (matvec mode) vs single-stream: %d/%d codes differ  %s (must be 0)\n",
                    diff_mv, B * 15, diff_mv == 0 ? "PASS" : "FAIL");
            fprintf(stderr, "  CP batched matmat vs single-stream:       %d/%d codes differ  (argmax flips on near-ties; validate via audio)\n",
                    diff_mm, B * 15);
            if (diff_mv != 0) pass = 0;
        }
        free(th); free(c0); free(ref); free(bat);
    }

    fprintf(stderr, "batch-test: %s\n", pass ? "PASS" : "FAIL");
    free(embeds_all); free(embedsB); free(href); free(hbat); qwen_batch_free(bb);
    return pass ? 0 : 1;
}

/* End-to-end batched-compute throughput bench: real model, B frames of Talker step +
 * CP predict, batched vs single-stream. Measures the actual per-frame compute speedup
 * (the autoregressive bulk; excludes prefill/sampling/decode/scheduler — the remaining
 * integration). `./qwen_tts -d <model> --batch-bench`. */
int qwen_batch_bench(qwen_tts_ctx_t *ctx) {
    qwen_tts_config_t *c = &ctx->config; int h = c->hidden_size;
    if (ctx->layers[0].wq_bf16 == NULL || !ctx->cp_lm_head_bf16[0]) {
        fprintf(stderr, "batch-bench: needs a bf16 model (v1)\n"); return 1;
    }
    const char *be = getenv("QWEN_BATCH_B"); int B = be ? atoi(be) : 8; if (B < 1 || B > 64) B = 8;
    const int K = 50;                         /* frames per sequence */
    qwen_batch_t *bb = qwen_batch_alloc(ctx, B, K + 4);
    if (!bb) { fprintf(stderr, "batch-bench: alloc failed\n"); return 1; }
    float *emb = (float *)malloc((size_t)K * h * sizeof(float));
    float *embB = (float *)malloc((size_t)B * h * sizeof(float));
    float *hid = (float *)malloc((size_t)B * h * sizeof(float));
    float *hid1 = (float *)malloc((size_t)h * sizeof(float));
    int *codes = (int *)malloc((size_t)B * 15 * sizeof(int));
    int *codes1 = (int *)malloc(15 * sizeof(int));
    int *c0 = (int *)malloc((size_t)B * sizeof(int));
    uint64_t rng = 0x1234567ull;
#define RBF (((double)((rng = rng * 6364136223846793005ull + 1442695040888963407ull) >> 40)) / (double)(1u << 24) * 2.0 - 1.0)
    for (int i = 0; i < K * h; i++) emb[i] = (float)(RBF * 0.1);
    for (int b = 0; b < B; b++) c0[b] = 1;
    struct timespec t0, t1;
#define NOW(t) clock_gettime(CLOCK_MONOTONIC, &(t))
#define MS(a,b) (((b).tv_sec-(a).tv_sec)*1e3 + ((b).tv_nsec-(a).tv_nsec)*1e-6)

    /* single-stream: B sequences x K frames (Talker step + CP predict) */
    NOW(t0);
    for (int seq = 0; seq < B; seq++) {
        ctx->kv_len = 0;
        for (int s = 0; s < K; s++) {
            qwen_talker_step(ctx, emb + (size_t)s * h, hid1);
            qwen_cp_predict(ctx, hid1, 1, codes1);
        }
    }
    NOW(t1); double t_single = MS(t0, t1);

    /* batched: K steps, each doing B frames */
    bb->kv_len = 0; bb->force_matvec = 0;
    NOW(t0);
    for (int s = 0; s < K; s++) {
        for (int b = 0; b < B; b++) memcpy(embB + (size_t)b * h, emb + (size_t)s * h, h * sizeof(float));
        qwen_batch_talker_step(ctx, bb, embB, hid);
        qwen_batch_cp_predict(ctx, bb, hid, c0, codes, NULL);
    }
    NOW(t1); double t_batch = MS(t0, t1);

    double frames = (double)B * K;
    fprintf(stderr, "batch-bench: B=%d K=%d (%.0f frames) threads=%d\n", B, K, frames, qwen_get_threads());
    fprintf(stderr, "  single-stream: %8.1f ms  (%6.1f frames/s)\n", t_single, frames / (t_single * 1e-3));
    fprintf(stderr, "  batched:       %8.1f ms  (%6.1f frames/s)\n", t_batch,  frames / (t_batch  * 1e-3));
    fprintf(stderr, "  SPEEDUP: %.2fx\n", t_single / t_batch);
    free(emb); free(embB); free(hid); free(hid1); free(codes); free(codes1); free(c0);
    qwen_batch_free(bb);
    ctx->kv_len = 0;
    return 0;
}
