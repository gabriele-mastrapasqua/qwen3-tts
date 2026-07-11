/*
 * qwen_tts_code_predictor.c - Code Predictor (MTP) forward pass
 * Generates codebooks 1-15 for each audio frame.
 *
 * Architecture: 5-layer Qwen3 transformer with GQA, QK-norm, NeoX RoPE.
 * Per frame: prefill (talker_hidden, code0_embed), then 14 autoregressive steps.
 */

#include "qwen_tts.h"
#include "qwen_tts_kernels.h"
#include "qwen_tts_safetensors.h"
#include "qwen_tts_batch.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* aligned_malloc/aligned_calloc now in qwen_tts_kernels.h */

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif
#ifdef __AVX2__
#include <immintrin.h>
#endif

/* ========================================================================
 * CP micro-benchmark (compile with -DCP_MICROBENCH, e.g. `make cp-microbench`)
 *
 * Partitions the per-frame Code Predictor time among its sub-operations so we
 * can see what dominates the ~87 ms/f (QKV proj, attention, FFN, norms,
 * lm_head, embed). Zero overhead and zero footprint when CP_MICROBENCH is
 * undefined — all macros expand to nothing. Single-threaded orchestration
 * (the matvecs spawn threads internally, but cp_layer_body runs serially),
 * so a file-scope timestamp threaded through the frame is correct.
 * ======================================================================== */
#ifdef CP_MICROBENCH
#include <sys/time.h>
typedef enum {
    CPB_EMBED, CPB_INNORM, CPB_QKV, CPB_QKNORM, CPB_ROPE, CPB_KVSTORE,
    CPB_ATTN, CPB_OPROJ, CPB_RESNORM, CPB_FFN_GU, CPB_SWIGLU, CPB_FFN_DOWN,
    CPB_LMHEAD, CPB_N
} cpb_slot_t;
static double cpb_acc[CPB_N];
static double cpb_t;  /* last timestamp (ms) */
static const char *cpb_name[CPB_N] = {
    "Embed+project", "Input norm", "QKV proj", "Q/K norm", "RoPE", "KV store",
    "Attention", "O proj", "Resid+Norm", "FFN gate_up", "SwiGLU", "FFN down",
    "lm_head argmax"
};
static inline double cpb_now(void) {
    struct timeval tv; gettimeofday(&tv, NULL);
    return (double)tv.tv_sec * 1000.0 + (double)tv.tv_usec / 1000.0;
}
#define CPB_RESET()    do { cpb_t = cpb_now(); } while (0)
#define CPB_MARK(slot) do { double _n = cpb_now(); cpb_acc[slot] += _n - cpb_t; cpb_t = _n; } while (0)
void qwen_cp_microbench_report(int frames) {
    double tot = 0; for (int i = 0; i < CPB_N; i++) tot += cpb_acc[i];
    fprintf(stderr, "\n  === CP micro-bench (%d frames, -DCP_MICROBENCH) ===\n", frames);
    for (int i = 0; i < CPB_N; i++)
        fprintf(stderr, "    %-16s %8.1f ms  %6.3f ms/f  %5.1f%%\n",
                cpb_name[i], cpb_acc[i], frames > 0 ? cpb_acc[i] / frames : 0,
                tot > 0 ? 100.0 * cpb_acc[i] / tot : 0);
    fprintf(stderr, "    %-16s %8.1f ms  %6.3f ms/f  (CP total measured here)\n",
            "TOTAL", tot, frames > 0 ? tot / frames : 0);
}
#else
#define CPB_RESET()
#define CPB_MARK(slot)
#endif

/* ========================================================================
 * Quant-ladder instrumentation (env-gated; truly zero overhead when off).
 *
 *   QWEN_DUMP_CODES=<path>  append one line "code0 c1 c2 ... c15" per frame
 *                           (the 16 codebook tokens). Run the synth at each
 *                           CP precision (see QWEN_CP_PREC) → one file each →
 *                           tests/quant_ladder.py computes the per-codebook
 *                           argmax-agreement matrix across {bf16,int8,int4,q2}.
 *   QWEN_FFN_SPARSITY[=eps]  count post-SwiGLU activations with |x| < eps
 *                           (default 1e-4) → contextual-sparsity headroom %.
 *                           Reported to stderr at process exit.
 *
 * This is the cheap "measure first" instrument behind PLAN.md future-research C:
 * it tells us WHERE and HOW MUCH int4 drifts vs int8/bf16 before we build any
 * speculative-decode / sparsity / roughness machinery.
 * ======================================================================== */
static FILE  *ql_codes_fp   = NULL;
static int    ql_init_done  = 0;
static int    ql_ffn_on     = 0;
static float  ql_ffn_eps    = 1e-4f;
static long   ql_ffn_total  = 0;
static long   ql_ffn_zero   = 0;

static void ql_report_atexit(void) {
    if (ql_codes_fp) { fclose(ql_codes_fp); ql_codes_fp = NULL; }
    if (ql_ffn_on && ql_ffn_total > 0)
        fprintf(stderr, "  [QWEN_FFN_SPARSITY] post-SwiGLU |x|<%.0e: %ld/%ld = %.2f%% (sparsity headroom)\n",
                (double)ql_ffn_eps, ql_ffn_zero, ql_ffn_total,
                100.0 * (double)ql_ffn_zero / (double)ql_ffn_total);
}

static void ql_init(void) {
    if (ql_init_done) return;
    ql_init_done = 1;
    const char *p = getenv("QWEN_DUMP_CODES");
    if (p && *p) ql_codes_fp = fopen(p, "w");
    const char *s = getenv("QWEN_FFN_SPARSITY");
    if (s) {
        ql_ffn_on = 1;
        double e = atof(s);
        if (e > 0) ql_ffn_eps = (float)e;
    }
    if (ql_codes_fp || ql_ffn_on) atexit(ql_report_atexit);
}

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

/* Convert f32 vector to bf16 (NEON-vectorized) */
static void f32_to_bf16_vec(uint16_t *dst, const float *src, int64_t n) {
#ifdef __ARM_NEON
    int64_t i = 0;
    for (; i + 7 < n; i += 8) {
        uint32x4_t u0 = vreinterpretq_u32_f32(vld1q_f32(src + i));
        uint32x4_t u1 = vreinterpretq_u32_f32(vld1q_f32(src + i + 4));
        uint16x4_t lo = vshrn_n_u32(u0, 16);
        uint16x4_t hi = vshrn_n_u32(u1, 16);
        vst1q_u16(dst + i, vcombine_u16(lo, hi));
    }
    for (; i < n; i++) dst[i] = f32_to_bf16(src[i]);
#elif defined(__AVX2__)
    int64_t i = 0;
    for (; i + 7 < n; i += 8) {
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

static uint16_t *get_bf16(void *ms, const char *name) {
    safetensors_file_t *sf = NULL;
    const safetensor_t *t = multi_safetensors_find((multi_safetensors_t *)ms, name, &sf);
    return (t && sf) ? (uint16_t *)safetensors_get_bf16_direct(sf, t) : NULL;
}

static float *get_f32(void *ms, const char *name) {
    safetensors_file_t *sf = NULL;
    const safetensor_t *t = multi_safetensors_find((multi_safetensors_t *)ms, name, &sf);
    return (t && sf) ? (float *)safetensors_get_f32(sf, t) : NULL;
}

/* Use centralized NEON+multi-threaded matvec from qwen_tts_kernels.c */
#define matvec_bf16 qwen_matvec_bf16

/* ========================================================================
 * RoPE - NeoX split-half
 * ======================================================================== */

static void apply_rope_neox(float *x, int n_heads, int head_dim,
                            const float *cos_cache, const float *sin_cache, int pos) {
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
 * Weight Loading
 * ======================================================================== */

/* Alloc int8 buffers if absent, then (re)quantize from the current bf16 pointer.
 * Reusing buffers makes this safe to call again after a WDELTA voice override. */
static void cp_qz(int8_t **dst, float **scale, const uint16_t *src, int rows, int cols) {
    if (!*dst)   *dst   = (int8_t *)aligned_malloc((size_t)rows * cols);
    if (!*scale) *scale = (float *)aligned_malloc((size_t)rows * sizeof(float));
    if (*dst && *scale) qwen_quantize_bf16_to_int8(src, rows, cols, *dst, *scale);
}

/* (Re)quantize Code Predictor weights (+ lm_heads) to INT8 from current bf16.
 * CP is hidden=1024 on both models; the denormal hang is fixed (FTZ + fused
 * qkv), so this is enabled for both. */
void qwen_cp_quantize_int8(qwen_tts_ctx_t *ctx) {
    qwen_tts_config_t *c = &ctx->config;
    if (!ctx->use_int8) return;
    int cp_h = c->cp_hidden_size;
    int cp_q_dim = c->cp_num_heads * c->cp_head_dim;
    int cp_kv_dim = c->cp_num_kv_heads * c->cp_head_dim;
    int cp_inter = c->cp_intermediate_size;
    for (int i = 0; i < c->cp_num_layers; i++) {
        qwen_cp_layer_t *l = &ctx->cp_layers[i];
        cp_qz(&l->wq_int8, &l->wq_scale, l->wq_bf16, cp_q_dim, cp_h);
        cp_qz(&l->wk_int8, &l->wk_scale, l->wk_bf16, cp_kv_dim, cp_h);
        cp_qz(&l->wv_int8, &l->wv_scale, l->wv_bf16, cp_kv_dim, cp_h);
        cp_qz(&l->wo_int8, &l->wo_scale, l->wo_bf16, cp_h, cp_q_dim);
        cp_qz(&l->gate_up_fused_int8, &l->gate_up_fused_scale, l->gate_up_fused_bf16, 2 * cp_inter, cp_h);
        cp_qz(&l->down_int8, &l->down_scale, l->down_bf16, cp_h, cp_inter);
    }
    for (int g = 0; g < 15; g++)
        if (ctx->cp_lm_head_bf16[g])
            cp_qz(&ctx->cp_lm_head_int8[g], &ctx->cp_lm_head_scale[g],
                  ctx->cp_lm_head_bf16[g], c->codebook_size, cp_h);
}

/* Alloc Q4_0 buffer if absent, then (re)quantize from the current bf16 pointer. */
static void cp_qz_q4(q4_0_block_t **dst, const uint16_t *src, int rows, int cols) {
    int bpr = cols / Q4_0_BLOCK_SIZE;
    if (!*dst) *dst = (q4_0_block_t *)aligned_malloc((size_t)rows * bpr * sizeof(q4_0_block_t));
    if (*dst) qwen_quantize_bf16_to_q4_0(src, rows, cols, *dst);
}

/* Q2_0 variant for the hybrid FFN path (QWEN_CP_Q2_FFN=1). */
static void cp_qz_q2(q2_0_block_t **dst, const uint16_t *src, int rows, int cols) {
    int bpr = cols / Q2_0_BLOCK_SIZE;
    if (!*dst) *dst = (q2_0_block_t *)aligned_malloc((size_t)rows * bpr * sizeof(q2_0_block_t));
    if (*dst) qwen_quantize_bf16_to_q2_0(src, rows, cols, *dst);
}

/* Lazily build the per-layer Q2_0 copy of the FFN down weight used by the
 * --roughness knob. Quantized from the bf16 mmap (never freed) so it is
 * independent of the active quant mode (works under bf16/int8/int4). */
static void cp_build_roughness(qwen_tts_ctx_t *ctx) {
    if (ctx->cp_rough_built) return;
    qwen_tts_config_t *c = &ctx->config;
    int cp_h = c->cp_hidden_size, cp_inter = c->cp_intermediate_size;
    for (int i = 0; i < c->cp_num_layers; i++) {
        qwen_cp_layer_t *l = &ctx->cp_layers[i];
        if (l->down_bf16 && !l->down_q2_rough)
            cp_qz_q2(&l->down_q2_rough, l->down_bf16, cp_h, cp_inter);
    }
    ctx->cp_rough_built = 1;
}

/* ---- Quant-ladder decomposition knobs (env-gated; reset-then-quantize fix-ups
 * applied AFTER the uniform QWEN_CP_PREC pass). Let us answer "is the late-codebook
 * drift in the shared transformer or the per-codebook lm_heads?" (QWEN_CP_LMHEAD_PREC)
 * and "which LAYERS tolerate low bits?" (QWEN_CP_LAYER_PREC) — the per-tensor/per-layer
 * mixed-precision (DeepSeek-style) feasibility test. ---- */
#define CP_FREE(p) do { if (p) { free(p); (p) = NULL; } } while (0)

/* Reset one CP layer's quant buffers → bf16 dispatch (frees q4/q2/int8 + scales). */
static void cp_layer_to_bf16(qwen_cp_layer_t *l) {
    CP_FREE(l->wq_q4); CP_FREE(l->wk_q4); CP_FREE(l->wv_q4); CP_FREE(l->wo_q4);
    CP_FREE(l->gate_up_fused_q4); CP_FREE(l->down_q4);
    CP_FREE(l->gate_up_fused_q2); CP_FREE(l->down_q2);
    CP_FREE(l->wq_int8); CP_FREE(l->wq_scale);
    CP_FREE(l->wk_int8); CP_FREE(l->wk_scale);
    CP_FREE(l->wv_int8); CP_FREE(l->wv_scale);
    CP_FREE(l->wo_int8); CP_FREE(l->wo_scale);
    CP_FREE(l->gate_up_fused_int8); CP_FREE(l->gate_up_fused_scale);
    CP_FREE(l->down_int8); CP_FREE(l->down_scale);
}

/* (Re)quantize one CP layer to a named precision {bf16|int8|int4}. */
static void cp_layer_quantize(qwen_tts_ctx_t *ctx, int layer, const char *prec) {
    qwen_tts_config_t *c = &ctx->config;
    int cp_h = c->cp_hidden_size;
    int cp_q_dim = c->cp_num_heads * c->cp_head_dim;
    int cp_kv_dim = c->cp_num_kv_heads * c->cp_head_dim;
    int cp_inter = c->cp_intermediate_size;
    qwen_cp_layer_t *l = &ctx->cp_layers[layer];
    cp_layer_to_bf16(l);
    if (!strcmp(prec, "int4")) {
        cp_qz_q4(&l->wq_q4, l->wq_bf16, cp_q_dim, cp_h);
        cp_qz_q4(&l->wk_q4, l->wk_bf16, cp_kv_dim, cp_h);
        cp_qz_q4(&l->wv_q4, l->wv_bf16, cp_kv_dim, cp_h);
        cp_qz_q4(&l->wo_q4, l->wo_bf16, cp_h, cp_q_dim);
        cp_qz_q4(&l->gate_up_fused_q4, l->gate_up_fused_bf16, 2 * cp_inter, cp_h);
        cp_qz_q4(&l->down_q4, l->down_bf16, cp_h, cp_inter);
    } else if (!strcmp(prec, "int8")) {
        cp_qz(&l->wq_int8, &l->wq_scale, l->wq_bf16, cp_q_dim, cp_h);
        cp_qz(&l->wk_int8, &l->wk_scale, l->wk_bf16, cp_kv_dim, cp_h);
        cp_qz(&l->wv_int8, &l->wv_scale, l->wv_bf16, cp_kv_dim, cp_h);
        cp_qz(&l->wo_int8, &l->wo_scale, l->wo_bf16, cp_h, cp_q_dim);
        cp_qz(&l->gate_up_fused_int8, &l->gate_up_fused_scale, l->gate_up_fused_bf16, 2 * cp_inter, cp_h);
        cp_qz(&l->down_int8, &l->down_scale, l->down_bf16, cp_h, cp_inter);
    } /* "bf16" → leave reset */
}

/* (Re)quantize all 15 lm_heads to a named precision {bf16|int8|int4}. */
static void cp_lmhead_quantize(qwen_tts_ctx_t *ctx, const char *prec) {
    qwen_tts_config_t *c = &ctx->config;
    int cp_h = c->cp_hidden_size;
    for (int g = 0; g < 15; g++) {
        CP_FREE(ctx->cp_lm_head_q4[g]);
        CP_FREE(ctx->cp_lm_head_int8[g]);
        CP_FREE(ctx->cp_lm_head_scale[g]);
        if (!ctx->cp_lm_head_bf16[g]) continue;
        if (!strcmp(prec, "int4"))
            cp_qz_q4(&ctx->cp_lm_head_q4[g], ctx->cp_lm_head_bf16[g], c->codebook_size, cp_h);
        else if (!strcmp(prec, "int8"))
            cp_qz(&ctx->cp_lm_head_int8[g], &ctx->cp_lm_head_scale[g],
                  ctx->cp_lm_head_bf16[g], c->codebook_size, cp_h);
    }
}

/* (Re)quantize Code Predictor weights (+ lm_heads) to Q4_0 from current bf16.
 * This is THE bandwidth lever on memory-bound CPUs: q4 weights are ¼ the bytes of
 * bf16, halving DRAM traffic vs int8 on the CP (re-read 16×/frame). --int4 only. */
void qwen_cp_quantize_q4(qwen_tts_ctx_t *ctx) {
    qwen_tts_config_t *c = &ctx->config;
    if (!ctx->use_int4) return;
    int cp_h = c->cp_hidden_size;
    int cp_q_dim = c->cp_num_heads * c->cp_head_dim;
    int cp_kv_dim = c->cp_num_kv_heads * c->cp_head_dim;
    int cp_inter = c->cp_intermediate_size;
    /* Hybrid: optionally drop the big FFN matrices to 2-bit to shrink the working set
     * below int4. Granular & experimental: QWEN_CP_Q2_FFN = both|1 / gateup / down —
     * lets us find WHICH matrix tolerates q2 (they have different sensitivity). */
    const char *e = getenv("QWEN_CP_Q2_FFN");
    int q2_gateup = e && (!strcmp(e, "1") || !strcmp(e, "both") || !strcmp(e, "gateup"));
    int q2_down   = e && (!strcmp(e, "1") || !strcmp(e, "both") || !strcmp(e, "down"));
    for (int i = 0; i < c->cp_num_layers; i++) {
        qwen_cp_layer_t *l = &ctx->cp_layers[i];
        cp_qz_q4(&l->wq_q4, l->wq_bf16, cp_q_dim, cp_h);
        cp_qz_q4(&l->wk_q4, l->wk_bf16, cp_kv_dim, cp_h);
        cp_qz_q4(&l->wv_q4, l->wv_bf16, cp_kv_dim, cp_h);
        cp_qz_q4(&l->wo_q4, l->wo_bf16, cp_h, cp_q_dim);
        if (q2_gateup) cp_qz_q2(&l->gate_up_fused_q2, l->gate_up_fused_bf16, 2 * cp_inter, cp_h);
        else           cp_qz_q4(&l->gate_up_fused_q4, l->gate_up_fused_bf16, 2 * cp_inter, cp_h);
        if (q2_down)   cp_qz_q2(&l->down_q2, l->down_bf16, cp_h, cp_inter);
        else           cp_qz_q4(&l->down_q4, l->down_bf16, cp_h, cp_inter);
    }
    for (int g = 0; g < 15; g++)
        if (ctx->cp_lm_head_bf16[g])
            cp_qz_q4(&ctx->cp_lm_head_q4[g], ctx->cp_lm_head_bf16[g], c->codebook_size, cp_h);
}

int qwen_cp_load(qwen_tts_ctx_t *ctx) {
    qwen_tts_config_t *c = &ctx->config;
    int cp_h = c->cp_hidden_size;
    int cp_q_dim = c->cp_num_heads * c->cp_head_dim;
    int cp_kv_dim = c->cp_num_kv_heads * c->cp_head_dim;

    if (!ctx->silent)
        fprintf(stderr, "Loading Code Predictor weights (hidden=%d, layers=%d)...\n",
                cp_h, c->cp_num_layers);

    /* Final norm */
    ctx->cp_norm = get_f32(ctx->safetensors, "talker.code_predictor.model.norm.weight");

    /* Per-layer weights */
    for (int i = 0; i < c->cp_num_layers; i++) {
        qwen_cp_layer_t *l = &ctx->cp_layers[i];
        char name[256];
        #define CP_LOAD_BF16(field, fmt, ...) do { \
            snprintf(name, sizeof(name), fmt, ##__VA_ARGS__); \
            l->field = get_bf16(ctx->safetensors, name); \
        } while(0)
        #define CP_LOAD_F32(field, fmt, ...) do { \
            snprintf(name, sizeof(name), fmt, ##__VA_ARGS__); \
            l->field = get_f32(ctx->safetensors, name); \
        } while(0)

        CP_LOAD_BF16(wq_bf16, "talker.code_predictor.model.layers.%d.self_attn.q_proj.weight", i);
        CP_LOAD_BF16(wk_bf16, "talker.code_predictor.model.layers.%d.self_attn.k_proj.weight", i);
        CP_LOAD_BF16(wv_bf16, "talker.code_predictor.model.layers.%d.self_attn.v_proj.weight", i);
        CP_LOAD_BF16(wo_bf16, "talker.code_predictor.model.layers.%d.self_attn.o_proj.weight", i);
        CP_LOAD_F32(q_norm, "talker.code_predictor.model.layers.%d.self_attn.q_norm.weight", i);
        CP_LOAD_F32(k_norm, "talker.code_predictor.model.layers.%d.self_attn.k_norm.weight", i);
        CP_LOAD_F32(input_norm, "talker.code_predictor.model.layers.%d.input_layernorm.weight", i);
        CP_LOAD_F32(post_attn_norm, "talker.code_predictor.model.layers.%d.post_attention_layernorm.weight", i);
        CP_LOAD_BF16(gate_bf16, "talker.code_predictor.model.layers.%d.mlp.gate_proj.weight", i);
        CP_LOAD_BF16(up_bf16, "talker.code_predictor.model.layers.%d.mlp.up_proj.weight", i);
        CP_LOAD_BF16(down_bf16, "talker.code_predictor.model.layers.%d.mlp.down_proj.weight", i);

        /* Fuse gate+up: interleave rows [gate_row0, up_row0, gate_row1, ...] */
        {
            size_t row_bytes = (size_t)cp_h * sizeof(uint16_t);
            l->gate_up_fused_bf16 = (uint16_t *)aligned_malloc(2 * (size_t)c->cp_intermediate_size * row_bytes);
            for (int r = 0; r < c->cp_intermediate_size; r++) {
                memcpy(l->gate_up_fused_bf16 + (size_t)(2 * r) * cp_h,
                       l->gate_bf16 + (size_t)r * cp_h, row_bytes);
                memcpy(l->gate_up_fused_bf16 + (size_t)(2 * r + 1) * cp_h,
                       l->up_bf16 + (size_t)r * cp_h, row_bytes);
            }
        }

        #undef CP_LOAD_BF16
        #undef CP_LOAD_F32
    }

    /* LM heads and codec embeddings for codebooks 1-15 */
    for (int g = 0; g < 15; g++) {
        char name[256];
        snprintf(name, sizeof(name), "talker.code_predictor.lm_head.%d.weight", g);
        ctx->cp_lm_head_bf16[g] = get_bf16(ctx->safetensors, name);
        snprintf(name, sizeof(name), "talker.code_predictor.model.codec_embedding.%d.weight", g);
        ctx->cp_codec_emb_bf16[g] = get_bf16(ctx->safetensors, name);
    }

    /* small_to_mtp_projection: projects talker_hidden -> cp_hidden (only when they differ) */
    int talker_h = c->hidden_size;
    if (talker_h != cp_h) {
        ctx->cp_mtp_proj_bf16 = get_bf16(ctx->safetensors, "talker.code_predictor.small_to_mtp_projection.weight");
        /* Bias is BF16 in safetensors — convert to f32 */
        uint16_t *bias_bf16 = get_bf16(ctx->safetensors, "talker.code_predictor.small_to_mtp_projection.bias");
        if (bias_bf16) {
            ctx->cp_mtp_proj_bias = (float *)aligned_malloc(cp_h * sizeof(float));
            for (int i = 0; i < cp_h; i++) ctx->cp_mtp_proj_bias[i] = bf16_to_f32(bias_bf16[i]);
        } else {
            ctx->cp_mtp_proj_bias = NULL;
        }
        ctx->cp_emb_dim = talker_h;  /* CP embeddings have talker_hidden dim */
        /* perf item 5 (2026-07-11): when the CP is quantized, this projection was the
         * one weight left permanently bf16 — [cp_h × talker_h] ≈ 4 MB re-read 16×/frame
         * (~64 MB/frame of unquantized traffic on 1.7B). Build an int8 twin (int8 even
         * under --int4: the projection feeds the whole CP residual stream, int8 is the
         * near-lossless pick). The bf16 pointer stays valid (mmap) for the GPU paths. */
        if ((ctx->use_int8 || ctx->use_int4) && ctx->cp_mtp_proj_bf16) {
            cp_qz(&ctx->cp_mtp_proj_int8, &ctx->cp_mtp_proj_scale,
                  ctx->cp_mtp_proj_bf16, cp_h, talker_h);
            if (!ctx->silent)
                fprintf(stderr, "  MTP projection quantized to INT8 (%d x %d)\n", cp_h, talker_h);
        }
        if (!ctx->silent)
            fprintf(stderr, "  MTP projection: %d -> %d\n", talker_h, cp_h);
    } else {
        ctx->cp_mtp_proj_bf16 = NULL;
        ctx->cp_mtp_proj_bias = NULL;
        ctx->cp_emb_dim = cp_h;      /* CP embeddings have cp_hidden dim (same as talker) */
    }

    /* Allocate CP KV cache (bf16 — needs 17 positions max: 2 prefill + 14 steps + margin) */
    int cp_kv_max = 64;
    int64_t cp_kv_size = (int64_t)c->cp_num_layers * cp_kv_max * cp_kv_dim;
    ctx->cp_kv_k = (uint16_t *)aligned_calloc(cp_kv_size, sizeof(uint16_t));
    ctx->cp_kv_v = (uint16_t *)aligned_calloc(cp_kv_size, sizeof(uint16_t));
    ctx->cp_kv_max = cp_kv_max;
    ctx->cp_kv_len = 0;

    /* Allocate CP decode buffers */
    ctx->cp_dec_x = (float *)aligned_malloc(cp_h * sizeof(float));
    ctx->cp_dec_q = (float *)aligned_malloc(cp_q_dim * sizeof(float));
    ctx->cp_dec_k = (float *)aligned_malloc(cp_kv_dim * sizeof(float));
    ctx->cp_dec_v = (float *)aligned_malloc(cp_kv_dim * sizeof(float));
    ctx->cp_dec_attn_out = (float *)aligned_malloc(cp_q_dim * sizeof(float));
    ctx->cp_dec_gate = (float *)aligned_malloc(2 * c->cp_intermediate_size * sizeof(float));
    ctx->cp_dec_up = NULL;  /* unused: gate buffer holds fused gate+up */
    ctx->cp_dec_ffn_out = (float *)aligned_malloc(cp_h * sizeof(float));

    /* CP RoPE cache (same theta as talker) */
    int half_dim = c->cp_head_dim / 2;
    ctx->cp_rope_cos = (float *)aligned_malloc((int64_t)cp_kv_max * half_dim * sizeof(float));
    ctx->cp_rope_sin = (float *)aligned_malloc((int64_t)cp_kv_max * half_dim * sizeof(float));
    for (int pos = 0; pos < cp_kv_max; pos++) {
        for (int i = 0; i < half_dim; i++) {
            float angle = (float)pos * (1.0f / powf(c->rope_theta, (float)(2*i) / c->cp_head_dim));
            ctx->cp_rope_cos[pos * half_dim + i] = cosf(angle);
            ctx->cp_rope_sin[pos * half_dim + i] = sinf(angle);
        }
    }
    ctx->cp_rope_cache_len = cp_kv_max;

    /* INT8 quantization of CP weights (optional, enabled by --int8 flag).
     * The old cp_h>=2048 gate was a workaround for the denormal hang (fixed June
     * 2026 via FTZ + fused int8 qkv). CP is hidden=1024 on BOTH models and is the
     * per-frame bottleneck (~90% matvec), so quantizing it helps 0.6B and 1.7B. */
    /* CP precision normally follows --int8/--int4, which ALSO quantize the Talker.
     * QWEN_CP_PREC={bf16|int8|int4} DECOUPLES the CP precision from the Talker so the
     * quant-ladder measurement can hold the Talker (hence code0) fixed and vary ONLY
     * the CP — otherwise code0 drifts run-to-run and the per-codebook agreement is
     * contaminated by Talker quantization. No env → unchanged --int8/--int4 behavior. */
    int cp_do_int8 = ctx->use_int8;
    int cp_do_int4 = ctx->use_int4;
    const char *cp_prec = getenv("QWEN_CP_PREC");
    if (cp_prec && *cp_prec) {
        cp_do_int8 = !strcmp(cp_prec, "int8");
        cp_do_int4 = !strcmp(cp_prec, "int4");
        if (!ctx->silent)
            fprintf(stderr, "  [QWEN_CP_PREC=%s] CP precision decoupled from Talker\n", cp_prec);
    }

    if (cp_do_int8) {
        if (!ctx->silent)
            fprintf(stderr, "  Quantizing CP weights to INT8 (per-row absmax)...\n");
        int save = ctx->use_int8; ctx->use_int8 = 1;   /* force CP int8 even if Talker isn't */
        qwen_cp_quantize_int8(ctx);  /* extracted — re-runnable after WDELTA override */
        ctx->use_int8 = save;
        if (!ctx->silent)
            fprintf(stderr, "  INT8 quantization done (%d layers + 15 lm_heads)\n", c->cp_num_layers);
    }

    /* Q4_0 quantization of CP weights (--int4). CP is the memory-bound bottleneck;
     * q4 (¼ the bytes of bf16) is the biggest bandwidth lever on x86/CPU. */
    if (cp_do_int4) {
        if (!ctx->silent)
            fprintf(stderr, "  Quantizing CP weights to Q4_0 (--int4)...\n");
        int save = ctx->use_int4; ctx->use_int4 = 1;   /* force CP int4 even if Talker isn't */
        qwen_cp_quantize_q4(ctx);
        ctx->use_int4 = save;
        if (!ctx->silent)
            fprintf(stderr, "  Q4_0 quantization done (%d layers + 15 lm_heads)\n", c->cp_num_layers);
    }

    /* Decomposition fix-ups (applied after the uniform pass above). Exp 1: override
     * lm_head precision independently of the shared transformer (QWEN_CP_LMHEAD_PREC).
     * Exp 2: per-layer precision (QWEN_CP_LAYER_PREC=p0,p1,p2,p3,p4). Both {bf16|int8|int4}. */
    {
        const char *lmh = getenv("QWEN_CP_LMHEAD_PREC");
        if (lmh && *lmh) {
            if (!ctx->silent)
                fprintf(stderr, "  [QWEN_CP_LMHEAD_PREC=%s] lm_head precision decoupled from transformer\n", lmh);
            cp_lmhead_quantize(ctx, lmh);
        }
        const char *lp = getenv("QWEN_CP_LAYER_PREC");
        if (lp && *lp) {
            char buf[256];
            strncpy(buf, lp, sizeof(buf) - 1); buf[sizeof(buf) - 1] = 0;
            int li = 0;
            for (char *tok = strtok(buf, ","); tok && li < c->cp_num_layers; tok = strtok(NULL, ","), li++) {
                if (!ctx->silent)
                    fprintf(stderr, "  [QWEN_CP_LAYER_PREC] layer %d -> %s\n", li, tok);
                cp_layer_quantize(ctx, li, tok);
            }
        }
    }

    if (!ctx->silent)
        fprintf(stderr, "  Code Predictor: %d layers loaded, q_dim=%d kv_dim=%d%s\n",
                c->cp_num_layers, cp_q_dim, cp_kv_dim,
                ctx->use_int4 ? " [INT4]" : (ctx->use_int8 ? " [INT8]" : ""));

    return 0;
}

/* ========================================================================
 * Single CP transformer step at given position
 * ======================================================================== */

static void cp_layer_body(qwen_tts_ctx_t *ctx, float *x, float *x_norm, int pos, int layer) {
    qwen_tts_config_t *c = &ctx->config;
    int cp_h = c->cp_hidden_size;
    int cp_q_dim = c->cp_num_heads * c->cp_head_dim;
    int cp_kv_dim = c->cp_num_kv_heads * c->cp_head_dim;
    int cp_inter = c->cp_intermediate_size;
    float eps = c->rms_norm_eps;
    float attn_scale = 1.0f / sqrtf((float)c->cp_head_dim);
    qwen_cp_layer_t *l = &ctx->cp_layers[layer];
    float *proj = ctx->cp_dec_ffn_out; /* reuse buffer */

    /* x_norm already contains RMSNorm(x) on entry */

    /* QKV projections (unified dispatch — single barrier for all 3) */
    if (l->wq_q4) {
        qwen_matvec_q4_0_qkv(ctx->cp_dec_q, ctx->cp_dec_k, ctx->cp_dec_v,
                              l->wq_q4, l->wk_q4, l->wv_q4,
                              x_norm, cp_h, cp_q_dim, cp_kv_dim);
    } else if (l->wq_int8) {
        qwen_matvec_int8_qkv(ctx->cp_dec_q, ctx->cp_dec_k, ctx->cp_dec_v,
                              l->wq_int8, l->wq_scale,
                              l->wk_int8, l->wk_scale,
                              l->wv_int8, l->wv_scale,
                              x_norm, cp_h, cp_q_dim, cp_kv_dim);
    } else {
        qwen_matvec_bf16_qkv(ctx->cp_dec_q, ctx->cp_dec_k, ctx->cp_dec_v,
                              l->wq_bf16, l->wk_bf16, l->wv_bf16,
                              x_norm, cp_h, cp_q_dim, cp_kv_dim);
    }
    CPB_MARK(CPB_QKV);

    /* Q/K RMSNorm per-head */
    qwen_rms_norm_per_head(ctx->cp_dec_q, l->q_norm, 1, c->cp_num_heads, c->cp_head_dim, eps);
    qwen_rms_norm_per_head(ctx->cp_dec_k, l->k_norm, 1, c->cp_num_kv_heads, c->cp_head_dim, eps);
    CPB_MARK(CPB_QKNORM);

    /* NeoX RoPE */
    apply_rope_neox(ctx->cp_dec_q, c->cp_num_heads, c->cp_head_dim,
                    ctx->cp_rope_cos, ctx->cp_rope_sin, pos);
    apply_rope_neox(ctx->cp_dec_k, c->cp_num_kv_heads, c->cp_head_dim,
                    ctx->cp_rope_cos, ctx->cp_rope_sin, pos);
    CPB_MARK(CPB_ROPE);

    /* Store KV in cache (convert f32→bf16) */
    int64_t kv_off = (int64_t)layer * ctx->cp_kv_max * cp_kv_dim + (int64_t)pos * cp_kv_dim;
    f32_to_bf16_vec(ctx->cp_kv_k + kv_off, ctx->cp_dec_k, cp_kv_dim);
    f32_to_bf16_vec(ctx->cp_kv_v + kv_off, ctx->cp_dec_v, cp_kv_dim);
    CPB_MARK(CPB_KVSTORE);

    /* Causal GQA attention (bf16 KV cache) */
    uint16_t *layer_k = ctx->cp_kv_k + (int64_t)layer * ctx->cp_kv_max * cp_kv_dim;
    uint16_t *layer_v = ctx->cp_kv_v + (int64_t)layer * ctx->cp_kv_max * cp_kv_dim;
    qwen_causal_attention_bf16kv(ctx->cp_dec_attn_out, ctx->cp_dec_q, layer_k, layer_v,
                                 1, pos + 1, c->cp_num_heads, c->cp_num_kv_heads,
                                 c->cp_head_dim, attn_scale, pos);
    CPB_MARK(CPB_ATTN);

    /* Output projection */
    if (l->wo_q4)
        qwen_matvec_q4_0(proj, l->wo_q4, ctx->cp_dec_attn_out, cp_h, cp_q_dim);
    else if (l->wo_int8)
        qwen_matvec_int8(proj, l->wo_int8, l->wo_scale, ctx->cp_dec_attn_out, cp_h, cp_q_dim);
    else
        matvec_bf16(proj, l->wo_bf16, ctx->cp_dec_attn_out, cp_h, cp_q_dim);
    CPB_MARK(CPB_OPROJ);

    /* Fused residual-add + post-attention RMSNorm (saves one pass over x) */
    qwen_rms_norm_residual(x_norm, x, proj, l->post_attn_norm, cp_h, eps);
    CPB_MARK(CPB_RESNORM);

    /* Fused gate+up SwiGLU FFN (single matvec, x loaded once) */
    if (l->gate_up_fused_q2)
        qwen_matvec_q2_0(ctx->cp_dec_gate, l->gate_up_fused_q2, x_norm, 2 * cp_inter, cp_h);
    else if (l->gate_up_fused_q4)
        qwen_matvec_q4_0(ctx->cp_dec_gate, l->gate_up_fused_q4, x_norm, 2 * cp_inter, cp_h);
    else if (l->gate_up_fused_int8)
        qwen_matvec_int8(ctx->cp_dec_gate, l->gate_up_fused_int8, l->gate_up_fused_scale,
                          x_norm, 2 * cp_inter, cp_h);
    else
        matvec_bf16(ctx->cp_dec_gate, l->gate_up_fused_bf16, x_norm, 2 * cp_inter, cp_h);
    CPB_MARK(CPB_FFN_GU);
    qwen_swiglu_inplace(ctx->cp_dec_gate, ctx->swiglu_tmp, cp_inter);
    CPB_MARK(CPB_SWIGLU);

    /* Contextual-sparsity probe: how many FFN activations are ~0 (would let us
     * skip the matching down-proj columns). Env-gated, off the hot path. */
    if (ql_ffn_on) {
        long z = 0;
        for (int i = 0; i < cp_inter; i++)
            if (fabsf(ctx->cp_dec_gate[i]) < ql_ffn_eps) z++;
        ql_ffn_zero  += z;
        ql_ffn_total += cp_inter;
    }

    /* Down projection */
    if (l->down_q2)
        qwen_matvec_q2_0(proj, l->down_q2, ctx->cp_dec_gate, cp_h, cp_inter);
    else if (l->down_q4)
        qwen_matvec_q4_0(proj, l->down_q4, ctx->cp_dec_gate, cp_h, cp_inter);
    else if (l->down_int8)
        qwen_matvec_int8(proj, l->down_int8, l->down_scale, ctx->cp_dec_gate, cp_h, cp_inter);
    else
        matvec_bf16(proj, l->down_bf16, ctx->cp_dec_gate, cp_h, cp_inter);
    CPB_MARK(CPB_FFN_DOWN);

    /* Roughness knob: blend a q2 version of the down output into the high-precision
     * one. `down` is the causal driver of the texture/roughness effect (q2-on-down =
     * "death metal"); blending dials it in continuously. 0 = off (no extra work). */
    if (ctx->cp_roughness > 0.0f && l->down_q2_rough) {
        float proj_q2[2048];  /* cp_h is 1024 on both models */
        qwen_matvec_q2_0(proj_q2, l->down_q2_rough, ctx->cp_dec_gate, cp_h, cp_inter);
        float r = ctx->cp_roughness;
        for (int i = 0; i < cp_h; i++) proj[i] = (1.0f - r) * proj[i] + r * proj_q2[i];
    }

    /* Fused residual-add + next layer's input RMSNorm (or just add for last layer) */
    if (layer + 1 < c->cp_num_layers) {
        qwen_rms_norm_residual(x_norm, x, proj, ctx->cp_layers[layer + 1].input_norm, cp_h, eps);
    } else {
        for (int i = 0; i < cp_h; i++) x[i] += proj[i];
    }
    CPB_MARK(CPB_RESNORM);
}

/* GPU-resident fused CP step (qwen_tts_cuda_talker.cu). Set alongside the fused Talker.
 * Emotion steer is a pre-step input add (done in qwen_cp_predict), so the fused step is emo-safe;
 * only cp_roughness (the q2-down blend) forces the CPU path. */
void *g_cuda_cp_state = NULL;
/* GPU-resident BATCHED CP step (throughput path). batch_cp_transformer_step delegates to it;
 * the per-seq argmax/embed between passes stay on CPU (in qwen_batch_cp_predict). */
void *g_cuda_cp_batch_state = NULL;
#ifdef QWEN_HAVE_METAL
void *g_metal_cp_state = NULL;   /* GPU-resident fused CP step (Metal, G2) */
void *g_metal_cp_frame_state = NULL;   /* device-frame CP (whole 16-pass loop on GPU, 1 sync/frame) */
void *g_metal_cp_batch_state = NULL;   /* batched CP step (server throughput) */
extern void qwen_metal_cp_batch_step(void *state, float *x, const int *pos_arr);
#endif
#ifdef QWEN_HAVE_CUDA
extern void qwen_cuda_cp_step(void *state, float *x, int pos);
extern void qwen_cuda_cp_batch_step(void *state, float *x, const int *pos_arr);
#endif

static void cp_transformer_step(qwen_tts_ctx_t *ctx, float *x, float *x_norm, int pos) {
    qwen_tts_config_t *c = &ctx->config;
    int cp_h = c->cp_hidden_size;

    extern void *g_gpu_fused_owner;   /* audit MED-2: delegate only for the owning ctx */
#ifdef QWEN_HAVE_CUDA
    if (g_cuda_cp_state && ctx == g_gpu_fused_owner && ctx->cp_roughness <= 0.0f) {
        qwen_cuda_cp_step(g_cuda_cp_state, x, pos);   /* x updated in place (residual stream) */
        return;
    }
#endif
#ifdef QWEN_HAVE_METAL
    if (g_metal_cp_state && ctx == g_gpu_fused_owner && ctx->cp_roughness <= 0.0f) {
        extern void qwen_metal_cp_step(void *, float *, int);
        qwen_metal_cp_step(g_metal_cp_state, x, pos);
        return;
    }
#endif

    /* First layer: standard input RMSNorm, then body produces fused norm for next */
    qwen_rms_norm(x_norm, x, ctx->cp_layers[0].input_norm, 1, cp_h, c->rms_norm_eps);
    CPB_MARK(CPB_INNORM);
    for (int layer = 0; layer < c->cp_num_layers; layer++)
        cp_layer_body(ctx, x, x_norm, pos, layer);
}

/* ========================================================================
 * Code Predictor: generate codebooks 1-15
 *
 * For each frame:
 * - Prefill: pos=0 = talker_hidden, pos=1 = codec_embed(code0)
 * - Then 14 autoregressive steps (pos=2..15), each feeding the previous codebook embed
 * - After each step: apply final norm, compute logits via lm_head, sample
 * ======================================================================== */

/* ── CP 2-token prefill (perf item 3, 2026-07-11) ────────────────────────────
 * Frame positions 0 (projected talker hidden) and 1 (code0 embed) are BOTH known
 * before the RVQ loop starts — the only legal intra-frame batching the sequential
 * argmax feedback permits. Run them as ONE B=2 matmat pass per projection so each
 * 5-layer weight matrix is read from DRAM once instead of twice (−1 of 16 full
 * weight sweeps ≈ −6% CP traffic). QUANTIZED-CP ONLY (uniform int8 or q4 across
 * all 5 layers): the bf16 path keeps the sequential steps so the bf16 golden
 * trajectory (the quality anchor) is untouched. matmat fp-order differs from
 * matvec → the quantized trajectories fork (same class of intentional fork as
 * the int8-SDOT argmax). Per-pos attention/rope/norm use the exact kernels the
 * sequential path uses. Kill-switch: QWEN_CP_PREFILL2=0. */
static int cp_prefill2_mode(qwen_tts_ctx_t *ctx) {   /* 0=off, 1=int8, 2=q4 (weights+env only) */
    static __thread int cached = -2;                 /* -2 = unprobed */
    if (cached != -2) return cached;
    /* OPT-IN (QWEN_CP_PREFILL2=1). Measured on M1 2026-07-11: perf NEUTRAL (B=2 int8
     * matmat ≈ 2× SDOT matvec — SDOT-seq is already near-optimal there), so the M1
     * default keeps the sequential path (no trajectory change). The win case is
     * x86/VNNI + bandwidth-starved boxes, where matmat_int8 rides the VNNI GEMM —
     * measure there before flipping the default. Curiosity worth an ear-test: with
     * pf2 ON the teacher-forced gold-agreement IMPROVES (int8 78.2 -> 82.2% — the
     * f32-act B=2 matmat is MORE precise than int8-act matvec on pos 0/1). */
    const char *e = getenv("QWEN_CP_PREFILL2");
    if (!(e && e[0] == '1')) return cached = 0;
    int all8 = 1, all4 = 1;
    for (int l = 0; l < ctx->config.cp_num_layers; l++) {
        qwen_cp_layer_t *L = &ctx->cp_layers[l];
        if (L->gate_up_fused_q2 || L->down_q2) { all8 = 0; all4 = 0; break; }
        if (!(L->wq_int8 && L->wk_int8 && L->wv_int8 && L->wo_int8 &&
              L->gate_up_fused_int8 && L->down_int8)) all8 = 0;
        if (!(L->wq_q4 && L->wk_q4 && L->wv_q4 && L->wo_q4 &&
              L->gate_up_fused_q4 && L->down_q4)) all4 = 0;
    }
    return cached = all8 ? 1 : (all4 ? 2 : 0);
}

static inline void cp_ilv2(float *X2, const float *a, const float *b, int n) {
    for (int i = 0; i < n; i++) { X2[2 * i] = a[i]; X2[2 * i + 1] = b[i]; }
}
static inline void cp_dcol2(float *dst, const float *Y2, int n, int p) {
    for (int i = 0; i < n; i++) dst[i] = Y2[2 * i + p];
}

static void cp_prefill2(qwen_tts_ctx_t *ctx, int mode, float *x0, float *x1) {
    qwen_tts_config_t *c = &ctx->config;
    int cp_h  = c->cp_hidden_size;
    int qd    = c->cp_num_heads * c->cp_head_dim;
    int kvd   = c->cp_num_kv_heads * c->cp_head_dim;
    int inter = c->cp_intermediate_size;
    float eps = c->rms_norm_eps;
    float attn_scale = 1.0f / sqrtf((float)c->cp_head_dim);

    /* grow-once per-thread scratch (~190 KB): all the B=2 interleaved views */
    static __thread float *S = NULL; static __thread size_t S_cap = 0;
    size_t need = (size_t)(2*cp_h /*X2*/ + 2*qd /*Q2*/ + 2*kvd /*K2*/ + 2*kvd /*V2*/
                 + qd /*attn0*/ + 2*qd /*A2*/ + 2*cp_h /*P2*/ + 4*inter /*G2*/
                 + 2*inter /*g0*/ + 2*inter /*GI2*/ + 2*cp_h /*D2*/ + 2*cp_h /*xn0+xn1*/);
    if (need > S_cap) {
        float *ns = (float *)realloc(S, need * sizeof(float));
        if (!ns) return;                       /* OOM: caller keeps prior KV; unreachable in practice */
        S = ns; S_cap = need;
    }
    float *X2 = S,            *Q2 = X2 + 2*cp_h, *K2 = Q2 + 2*qd,  *V2 = K2 + 2*kvd;
    float *attn0 = V2 + 2*kvd, *A2 = attn0 + qd, *P2 = A2 + 2*qd,  *G2 = P2 + 2*cp_h;
    float *g0 = G2 + 4*inter,  *GI2 = g0 + 2*inter, *D2 = GI2 + 2*inter;
    float *xn0 = D2 + 2*cp_h,  *xn1 = xn0 + cp_h;

#define CP_MM2(Y, W8, S8, W4, ROWS, COLS) do {                                   \
        if (mode == 1) qwen_matmat_int8((Y), (W8), (S8), X2loc, (ROWS), (COLS), 2); \
        else           qwen_matmat_q4_0((Y), (W4), X2loc, (ROWS), (COLS), 2);       \
    } while (0)

    float *xs[2] = { x0, x1 };
    for (int layer = 0; layer < c->cp_num_layers; layer++) {
        qwen_cp_layer_t *l = &ctx->cp_layers[layer];

        qwen_rms_norm(xn0, x0, l->input_norm, 1, cp_h, eps);
        qwen_rms_norm(xn1, x1, l->input_norm, 1, cp_h, eps);
        cp_ilv2(X2, xn0, xn1, cp_h);
        { const float *X2loc = X2;
          CP_MM2(Q2, l->wq_int8, l->wq_scale, l->wq_q4, qd,  cp_h);
          CP_MM2(K2, l->wk_int8, l->wk_scale, l->wk_q4, kvd, cp_h);
          CP_MM2(V2, l->wv_int8, l->wv_scale, l->wv_q4, kvd, cp_h); }

        for (int p = 0; p < 2; p++) {
            float *q = ctx->cp_dec_q, *k = ctx->cp_dec_k, *v = ctx->cp_dec_v;
            cp_dcol2(q, Q2, qd, p); cp_dcol2(k, K2, kvd, p); cp_dcol2(v, V2, kvd, p);
            qwen_rms_norm_per_head(q, l->q_norm, 1, c->cp_num_heads, c->cp_head_dim, eps);
            qwen_rms_norm_per_head(k, l->k_norm, 1, c->cp_num_kv_heads, c->cp_head_dim, eps);
            apply_rope_neox(q, c->cp_num_heads, c->cp_head_dim, ctx->cp_rope_cos, ctx->cp_rope_sin, p);
            apply_rope_neox(k, c->cp_num_kv_heads, c->cp_head_dim, ctx->cp_rope_cos, ctx->cp_rope_sin, p);
            int64_t kv_off = (int64_t)layer * ctx->cp_kv_max * kvd + (int64_t)p * kvd;
            f32_to_bf16_vec(ctx->cp_kv_k + kv_off, k, kvd);
            f32_to_bf16_vec(ctx->cp_kv_v + kv_off, v, kvd);
            uint16_t *layer_k = ctx->cp_kv_k + (int64_t)layer * ctx->cp_kv_max * kvd;
            uint16_t *layer_v = ctx->cp_kv_v + (int64_t)layer * ctx->cp_kv_max * kvd;
            qwen_causal_attention_bf16kv(p == 0 ? attn0 : ctx->cp_dec_attn_out, q,
                                         layer_k, layer_v, 1, p + 1,
                                         c->cp_num_heads, c->cp_num_kv_heads,
                                         c->cp_head_dim, attn_scale, p);
        }

        cp_ilv2(A2, attn0, ctx->cp_dec_attn_out, qd);
        { const float *X2loc = A2;
          CP_MM2(P2, l->wo_int8, l->wo_scale, l->wo_q4, cp_h, qd); }
        for (int p = 0; p < 2; p++) {
            float *x = xs[p];
            for (int i = 0; i < cp_h; i++) x[i] += P2[2 * i + p];
        }

        qwen_rms_norm(xn0, x0, l->post_attn_norm, 1, cp_h, eps);
        qwen_rms_norm(xn1, x1, l->post_attn_norm, 1, cp_h, eps);
        cp_ilv2(X2, xn0, xn1, cp_h);
        { const float *X2loc = X2;
          CP_MM2(G2, l->gate_up_fused_int8, l->gate_up_fused_scale, l->gate_up_fused_q4,
                 2 * inter, cp_h); }
        cp_dcol2(g0, G2, 2 * inter, 0);
        cp_dcol2(ctx->cp_dec_gate, G2, 2 * inter, 1);
        qwen_swiglu_inplace(g0, ctx->swiglu_tmp, inter);
        qwen_swiglu_inplace(ctx->cp_dec_gate, ctx->swiglu_tmp, inter);
        cp_ilv2(GI2, g0, ctx->cp_dec_gate, inter);
        { const float *X2loc = GI2;
          CP_MM2(D2, l->down_int8, l->down_scale, l->down_q4, cp_h, inter); }
        for (int p = 0; p < 2; p++) {
            float *x = xs[p];
            for (int i = 0; i < cp_h; i++) x[i] += D2[2 * i + p];
        }
    }
#undef CP_MM2
}

/* Apply small_to_mtp_projection: projects from emb_dim to cp_hidden.
 * If no projection needed (0.6B), just copies the first cp_h elements.
 * src has dim=emb_dim, dst has dim=cp_h. */
static void cp_mtp_project(qwen_tts_ctx_t *ctx, float *dst, const float *src) {
    int cp_h = ctx->config.cp_hidden_size;
    if (ctx->cp_mtp_proj_bf16) {
        /* Linear: dst = W @ src + bias, W is [cp_h, emb_dim]. Prefer the int8 twin
         * when the CP is quantized (perf item 5: ¼ the bytes of the 16×-reread bf16). */
        int emb_dim = ctx->cp_emb_dim;
        if (ctx->cp_mtp_proj_int8)
            qwen_matvec_int8(dst, ctx->cp_mtp_proj_int8, ctx->cp_mtp_proj_scale,
                             src, cp_h, emb_dim);
        else
            matvec_bf16(dst, ctx->cp_mtp_proj_bf16, src, cp_h, emb_dim);
        if (ctx->cp_mtp_proj_bias) {
            for (int i = 0; i < cp_h; i++) dst[i] += ctx->cp_mtp_proj_bias[i];
        }
    } else {
        memcpy(dst, src, cp_h * sizeof(float));
    }
}

int qwen_cp_predict(qwen_tts_ctx_t *ctx, float *talker_hidden, int code0, int *out_codes) {
    qwen_tts_config_t *c = &ctx->config;
    int cp_h = c->cp_hidden_size;
    int emb_dim = ctx->cp_emb_dim;  /* talker_hidden for 1.7B, cp_hidden for 0.6B */

#ifdef QWEN_HAVE_METAL
    /* Device-frame CP: whole 16-pass RVQ loop + argmax + embed on GPU, 1 sync/frame (the M1 win).
     * Falls back to the CPU/per-pass loop for steer/roughness/teacher-forcing. */
    { extern void *g_metal_cp_frame_state, *g_gpu_fused_owner;
      extern void qwen_metal_cp_frame(void *, const float *, int, int *);
      if (g_metal_cp_frame_state && ctx == g_gpu_fused_owner &&
          ctx->cp_roughness <= 0.0f && !ctx->tf_ref_codes) {
        qwen_metal_cp_frame(g_metal_cp_frame_state, talker_hidden, code0, out_codes);
        return 0;
      } }
#endif

    ql_init();  /* one-time env probe (QWEN_DUMP_CODES / QWEN_FFN_SPARSITY) */

    if (ctx->cp_roughness > 0.0f && !ctx->cp_rough_built) cp_build_roughness(ctx);

    /* Reset CP KV cache for this frame */
    ctx->cp_kv_len = 0;

    /* Pre-allocated buffers reused across frames (avoid per-frame malloc) */
    float *cp_x = ctx->cp_dec_x;
    float *cp_normed = ctx->cp_dec_attn_out; /* reuse: not overlapping with transformer step output */
    float *x_norm = ctx->cp_dec_ffn_out;     /* reuse: scratch for transformer step */

    CPB_RESET();
    /* Step 0: process talker hidden state (project if needed) */
    cp_mtp_project(ctx, cp_x, talker_hidden);

    /* Step-1 input (code0 embed via TALKER's codec embedding, NOT CP's) is known
     * NOW too — compute it up front so positions 0+1 can prefill together. */
    static __thread float *x1in = NULL;
    if (!x1in) { x1in = (float *)malloc((size_t)cp_h * sizeof(float)); if (!x1in) return -1; }
    {
        int h = c->hidden_size;
        if (ctx->codec_embedding_bf16 && code0 >= 0 && code0 < c->codec_vocab_size) {
            float emb_buf[4096];
            qwen_bf16_to_f32_vec(emb_buf, ctx->codec_embedding_bf16 + (int64_t)code0 * h, h);
            cp_mtp_project(ctx, x1in, emb_buf);
        } else {
            memset(x1in, 0, cp_h * sizeof(float));
        }
    }
    CPB_MARK(CPB_EMBED);

    /* Fused 2-token prefill (quantized CP only; see cp_prefill2). Roughness blends
     * per-step → sequential path. CP_MICROBENCH keeps the per-step marks meaningful. */
    int pf2 = 0;
#ifndef CP_MICROBENCH
    if (ctx->cp_roughness <= 0.0f) pf2 = cp_prefill2_mode(ctx);
#endif
    if (pf2) {
        cp_prefill2(ctx, pf2, cp_x, x1in);
        memcpy(cp_x, x1in, (size_t)cp_h * sizeof(float));   /* continue on pos-1's stream */
    } else {
        cp_transformer_step(ctx, cp_x, x_norm, 0);
        memcpy(cp_x, x1in, (size_t)cp_h * sizeof(float));
        CPB_MARK(CPB_EMBED);
        cp_transformer_step(ctx, cp_x, x_norm, 1);
    }

    /* Predict codebook 1: fused argmax+matvec (greedy — avoids writing 2048 logits) */
    qwen_rms_norm(cp_normed, cp_x, ctx->cp_norm, 1, cp_h, c->rms_norm_eps);
    if (ctx->cp_lm_head_q4[0])
        out_codes[0] = qwen_argmax_matvec_q4_0(cp_normed, ctx->cp_lm_head_q4[0], cp_h, c->codebook_size);
    else if (ctx->cp_lm_head_int8[0])
        out_codes[0] = qwen_argmax_matvec_int8(cp_normed, ctx->cp_lm_head_int8[0],
                                                ctx->cp_lm_head_scale[0], cp_h, c->codebook_size);
    else
        out_codes[0] = qwen_argmax_matvec_bf16(cp_normed, ctx->cp_lm_head_bf16[0], cp_h, c->codebook_size);
    CPB_MARK(CPB_LMHEAD);

    /* Steps 2-15: generate codebooks 2-15. */
    for (int g = 1; g < 15; g++) {
        /* Teacher-forcing (quant-ladder): feed the REFERENCE prev code, not this
         * precision's own prediction, so step g sees identical inputs across all
         * precisions → its disagreement is pure step-g quant drift. */
        int prev_code = ctx->tf_ref_codes ? ctx->tf_ref_codes[g - 1] : out_codes[g - 1];
        int pos = g + 1;

        /* Embed previous code using CP codec_emb[g-1] (NOT [g]).
         * Vectorized bf16→f32 conversion (NEON/AVX2 via qwen_bf16_to_f32_vec). */
        if (ctx->cp_codec_emb_bf16[g - 1] && prev_code >= 0 && prev_code < c->codebook_size) {
            float emb_buf[4096];
            const uint16_t *e = ctx->cp_codec_emb_bf16[g - 1] + (int64_t)prev_code * emb_dim;
            qwen_bf16_to_f32_vec(emb_buf, e, emb_dim);
            cp_mtp_project(ctx, cp_x, emb_buf);
        } else {
            memset(cp_x, 0, cp_h * sizeof(float));
        }
        CPB_MARK(CPB_EMBED);

        cp_transformer_step(ctx, cp_x, x_norm, pos);

        /* Fused argmax+matvec (greedy) */
        qwen_rms_norm(cp_normed, cp_x, ctx->cp_norm, 1, cp_h, c->rms_norm_eps);
        if (ctx->cp_lm_head_q4[g])
            out_codes[g] = qwen_argmax_matvec_q4_0(cp_normed, ctx->cp_lm_head_q4[g], cp_h, c->codebook_size);
        else if (ctx->cp_lm_head_int8[g])
            out_codes[g] = qwen_argmax_matvec_int8(cp_normed, ctx->cp_lm_head_int8[g],
                                                    ctx->cp_lm_head_scale[g], cp_h, c->codebook_size);
        else
            out_codes[g] = qwen_argmax_matvec_bf16(cp_normed, ctx->cp_lm_head_bf16[g], cp_h, c->codebook_size);
        CPB_MARK(CPB_LMHEAD);
    }

    /* Quant-ladder dump: the 16 codebook tokens for this frame (code0 from the
     * Talker — held fixed across precisions — then the 15 CP codes). */
    if (ql_codes_fp) {
        fprintf(ql_codes_fp, "%d", code0);
        for (int g = 0; g < 15; g++) fprintf(ql_codes_fp, " %d", out_codes[g]);
        fputc('\n', ql_codes_fp);
    }

    return 0;
}

/* ========================================================================
 * OPT-IN BATCHED Code Predictor (feat/batching) — see qwen_tts_batch.h.
 * ADDITIVE: qwen_cp_predict above is untouched. B frames in lockstep through
 * the 16-step CP; reuses the per-vector kernels looped over B and batches the
 * matvecs via qwen_batch_proj_q (PRECISION-AWARE: q4 > int8 > bf16, matching
 * the single-stream dispatch — the old "v1: bf16 only" note was stale). No
 * roughness in the batched path (steering IS supported). Mirrors
 * cp_layer_body / cp_transformer_step.
 * ======================================================================== */

/* Precision-aware CP lm_head argmax (B2): mirror the single-stream dispatch
 * (q4 > int8 > bf16) so the batched codes match the quantized single-stream path
 * — a bf16 lm_head in int8/int4 mode forks the trajectory (CP codes feed back). */
static int cp_lm_argmax(qwen_tts_ctx_t *ctx, const float *normed, int g, int ch, int vocab) {
    if (ctx->cp_lm_head_q4[g])
        return qwen_argmax_matvec_q4_0(normed, ctx->cp_lm_head_q4[g], ch, vocab);
    if (ctx->cp_lm_head_int8[g])
        return qwen_argmax_matvec_int8(normed, ctx->cp_lm_head_int8[g], ctx->cp_lm_head_scale[g], ch, vocab);
    return qwen_argmax_matvec_bf16(normed, ctx->cp_lm_head_bf16[g], ch, vocab);
}

static void batch_cp_layer(qwen_tts_ctx_t *ctx, qwen_batch_t *bb,
                           float *x, float *x_norm, int pos, int layer, const int *active) {
    qwen_tts_config_t *c = &ctx->config;
    int B = bb->B, ch = bb->cp_h, cqd = bb->cp_q_dim, ckvd = bb->cp_kv_dim, cint = bb->cp_inter;
    float eps = c->rms_norm_eps, ascale = 1.0f / sqrtf((float)c->cp_head_dim);
    int fm = bb->force_matvec;
    qwen_cp_layer_t *l = &ctx->cp_layers[layer];
#define CP_SKIP(b) (active && !active[b])   /* slot compaction: inactive skip per-slot vector work */

    /* x_norm = RMSNorm(x) already on entry. QKV (batched, precision-aware) */
    qwen_batch_proj_q(bb->cp_q, l->wq_bf16, l->wq_int8, l->wq_scale, l->wq_q4, x_norm, cqd,  ch, ch, B, fm, bb->cp_Xt, bb->cp_Yt);
    qwen_batch_proj_q(bb->cp_k, l->wk_bf16, l->wk_int8, l->wk_scale, l->wk_q4, x_norm, ckvd, ch, ch, B, fm, bb->cp_Xt, bb->cp_Yt);
    qwen_batch_proj_q(bb->cp_v, l->wv_bf16, l->wv_int8, l->wv_scale, l->wv_q4, x_norm, ckvd, ch, ch, B, fm, bb->cp_Xt, bb->cp_Yt);
    for (int b = 0; b < B; b++) {
        if (CP_SKIP(b)) continue;
        qwen_rms_norm_per_head(bb->cp_q + (size_t)b * cqd,  l->q_norm, 1, c->cp_num_heads,    c->cp_head_dim, eps);
        qwen_rms_norm_per_head(bb->cp_k + (size_t)b * ckvd, l->k_norm, 1, c->cp_num_kv_heads, c->cp_head_dim, eps);
        apply_rope_neox(bb->cp_q + (size_t)b * cqd,  c->cp_num_heads,    c->cp_head_dim, ctx->cp_rope_cos, ctx->cp_rope_sin, pos);
        apply_rope_neox(bb->cp_k + (size_t)b * ckvd, c->cp_num_kv_heads, c->cp_head_dim, ctx->cp_rope_cos, ctx->cp_rope_sin, pos);
        size_t kvbase = ((size_t)b * bb->cp_num_layers + layer) * bb->cp_kv_max * ckvd + (size_t)pos * ckvd;
        f32_to_bf16_vec(bb->cp_kv_k + kvbase, bb->cp_k + (size_t)b * ckvd, ckvd);
        f32_to_bf16_vec(bb->cp_kv_v + kvbase, bb->cp_v + (size_t)b * ckvd, ckvd);
        size_t lbase = ((size_t)b * bb->cp_num_layers + layer) * bb->cp_kv_max * ckvd;
        qwen_causal_attention_bf16kv(bb->cp_attn + (size_t)b * cqd, bb->cp_q + (size_t)b * cqd,
                                     bb->cp_kv_k + lbase, bb->cp_kv_v + lbase, 1, pos + 1,
                                     c->cp_num_heads, c->cp_num_kv_heads, c->cp_head_dim, ascale, pos);
    }
    /* O projection (batched, precision-aware) + residual + post-attn norm */
    qwen_batch_proj_q(bb->cp_proj, l->wo_bf16, l->wo_int8, l->wo_scale, l->wo_q4, bb->cp_attn, ch, cqd, cqd, B, fm, bb->cp_Xt, bb->cp_Yt);
    for (int b = 0; b < B; b++) {
        if (CP_SKIP(b)) continue;
        qwen_rms_norm_residual(x_norm + (size_t)b * ch, x + (size_t)b * ch,
                               bb->cp_proj + (size_t)b * ch, l->post_attn_norm, ch, eps);
    }
    /* gate+up (batched, precision-aware) + SwiGLU per frame */
    qwen_batch_proj_q(bb->cp_gate, l->gate_up_fused_bf16, l->gate_up_fused_int8, l->gate_up_fused_scale,
                      l->gate_up_fused_q4, x_norm, 2 * cint, ch, ch, B, fm, bb->cp_Xt, bb->cp_Yt);
    for (int b = 0; b < B; b++) {
        if (CP_SKIP(b)) continue;
        qwen_swiglu_inplace(bb->cp_gate + (size_t)b * 2 * cint, bb->cp_swiglu_tmp, cint);
    }
    /* down (batched, precision-aware) */
    qwen_batch_proj_q(bb->cp_proj, l->down_bf16, l->down_int8, l->down_scale, l->down_q4,
                      bb->cp_gate, ch, cint, 2 * cint, B, fm, bb->cp_Xt, bb->cp_Yt);
    /* residual (+ next layer input norm, or plain add on last layer) */
    if (layer + 1 < c->cp_num_layers) {
        for (int b = 0; b < B; b++) {
            if (CP_SKIP(b)) continue;
            qwen_rms_norm_residual(x_norm + (size_t)b * ch, x + (size_t)b * ch,
                                   bb->cp_proj + (size_t)b * ch, ctx->cp_layers[layer + 1].input_norm, ch, eps);
        }
    } else {
        for (int b = 0; b < B; b++) {
            if (CP_SKIP(b)) continue;
            float *xb = x + (size_t)b * ch, *pb = bb->cp_proj + (size_t)b * ch;
            for (int i = 0; i < ch; i++) xb[i] += pb[i];
        }
    }
#undef CP_SKIP
}

static void batch_cp_transformer_step(qwen_tts_ctx_t *ctx, qwen_batch_t *bb,
                                      float *x, float *x_norm, int pos, const int *active) {
    qwen_tts_config_t *c = &ctx->config;
    int B = bb->B, ch = bb->cp_h; float eps = c->rms_norm_eps;
#ifdef QWEN_HAVE_CUDA
    /* GPU batched path: the whole 5-layer step runs on device (x = [B][cp_h] residual, updated
     * in place). Lockstep — all B at the same pos. The caller's per-seq argmax/embed stay on CPU. */
    extern void *g_cuda_cp_batch_state;
    if (g_cuda_cp_batch_state && B <= 16) {
        int pos_arr[16]; for (int b = 0; b < B; b++) pos_arr[b] = pos;
        qwen_cuda_cp_batch_step(g_cuda_cp_batch_state, x, pos_arr);
        return;
    }
#endif
#ifdef QWEN_HAVE_METAL
    if (g_metal_cp_batch_state && B <= 8) {
        int pos_arr[8]; for (int b = 0; b < B; b++) pos_arr[b] = pos;
        qwen_metal_cp_batch_step(g_metal_cp_batch_state, x, pos_arr);
        return;
    }
#endif
    for (int b = 0; b < B; b++) {
        if (active && !active[b]) continue;
        qwen_rms_norm(x_norm + (size_t)b * ch, x + (size_t)b * ch, ctx->cp_layers[0].input_norm, 1, ch, eps);
    }
    for (int layer = 0; layer < c->cp_num_layers; layer++)
        batch_cp_layer(ctx, bb, x, x_norm, pos, layer, active);
}

int qwen_batch_cp_predict(qwen_tts_ctx_t *ctx, qwen_batch_t *bb,
                          const float *talker_hidden, const int *code0, int *out_codes,
                          const int *active) {
    qwen_tts_config_t *c = &ctx->config;
    int B = bb->B, ch = bb->cp_h, h = c->hidden_size, emb_dim = ctx->cp_emb_dim;
    /* Sanity: the mmapped bf16 weights must exist (they always do after a successful load —
     * quantized runs keep the bf16 pointers as the embedding/GPU source). audit 2026-07-11:
     * the old "v1 CPU: bf16 only" framing was stale — batch_cp_layer dispatches q4>int8>bf16
     * via qwen_batch_proj_q, so the CPU batched CP has been quantization-aware for a while. */
    if (!ctx->cp_layers[0].wq_bf16 || !ctx->cp_lm_head_bf16[0]) return -2;
    float *cx = bb->cp_x, *cxn = bb->cp_x_norm;
    float emb_buf[4096], normed[2048];

#define CPB_SKIP(b) (active && !active[b])   /* slot compaction (MED-5): see qwen_tts_batch.h */

    /* step 0: project talker hidden */
    for (int b = 0; b < B; b++) {
        if (CPB_SKIP(b)) { memset(cx + (size_t)b * ch, 0, ch * sizeof(float)); continue; }
        cp_mtp_project(ctx, cx + (size_t)b * ch, talker_hidden + (size_t)b * h);
    }
    batch_cp_transformer_step(ctx, bb, cx, cxn, 0, active);

    /* step 1: embed code0 via the Talker codec embedding */
    for (int b = 0; b < B; b++) {
        if (CPB_SKIP(b)) continue;   /* cx slot already zeroed at step 0 */
        int code0_b = code0[b];
        if (ctx->codec_embedding_bf16 && code0_b >= 0 && code0_b < c->codec_vocab_size) {
            qwen_bf16_to_f32_vec(emb_buf, ctx->codec_embedding_bf16 + (int64_t)code0_b * h, h);
            cp_mtp_project(ctx, cx + (size_t)b * ch, emb_buf);
        } else memset(cx + (size_t)b * ch, 0, ch * sizeof(float));
    }
    batch_cp_transformer_step(ctx, bb, cx, cxn, 1, active);

    /* codebook 0 per frame (greedy argmax) */
    for (int b = 0; b < B; b++) {
        if (CPB_SKIP(b)) { out_codes[(size_t)b * 15 + 0] = 0; continue; }
        qwen_rms_norm(normed, cx + (size_t)b * ch, ctx->cp_norm, 1, ch, c->rms_norm_eps);
        out_codes[(size_t)b * 15 + 0] = cp_lm_argmax(ctx, normed, 0, ch, c->codebook_size);
    }

    /* codebooks 1-14 */
    for (int g = 1; g < 15; g++) {
        int pos = g + 1;
        for (int b = 0; b < B; b++) {
            if (CPB_SKIP(b)) continue;
            int prev = out_codes[(size_t)b * 15 + (g - 1)];
            if (ctx->cp_codec_emb_bf16[g - 1] && prev >= 0 && prev < c->codebook_size) {
                qwen_bf16_to_f32_vec(emb_buf, ctx->cp_codec_emb_bf16[g - 1] + (int64_t)prev * emb_dim, emb_dim);
                cp_mtp_project(ctx, cx + (size_t)b * ch, emb_buf);
            } else memset(cx + (size_t)b * ch, 0, ch * sizeof(float));
        }
        batch_cp_transformer_step(ctx, bb, cx, cxn, pos, active);
        for (int b = 0; b < B; b++) {
            if (CPB_SKIP(b)) { out_codes[(size_t)b * 15 + g] = 0; continue; }
            qwen_rms_norm(normed, cx + (size_t)b * ch, ctx->cp_norm, 1, ch, c->rms_norm_eps);
            out_codes[(size_t)b * 15 + g] = cp_lm_argmax(ctx, normed, g, ch, c->codebook_size);
        }
    }
    return 0;
#undef CPB_SKIP
}
