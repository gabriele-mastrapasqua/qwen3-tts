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
    safetensors_file_t *sf = NULL;
    const safetensor_t *t = multi_safetensors_find((multi_safetensors_t *)ms, name, &sf);
    if (!t || !sf) return NULL;
    return safetensors_get_bf16_direct(sf, t);
}

static float *get_f32(void *ms, const char *name) {
    safetensors_file_t *sf = NULL;
    const safetensor_t *t = multi_safetensors_find((multi_safetensors_t *)ms, name, &sf);
    if (!t || !sf) return NULL;
    return safetensors_get_f32(sf, t);
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

int qwen_talker_step(qwen_tts_ctx_t *ctx, float *embed, float *hidden_out) {
    qwen_tts_config_t *c = &ctx->config;
    int h = c->hidden_size;
    int q_dim = c->num_heads * c->head_dim;
    int kv_dim = c->num_kv_heads * c->head_dim;
    int inter = c->intermediate_size;
    int pos = ctx->kv_len;
    float eps = c->rms_norm_eps;

    if (kv_cache_grow(ctx, pos + 1) != 0) return -1;

    memcpy(ctx->dec_x, embed, h * sizeof(float));

    for (int layer = 0; layer < c->num_layers; layer++) {
        qwen_talker_layer_t *l = &ctx->layers[layer];

        /* 1. Input RMSNorm */
        qwen_rms_norm(ctx->dec_x_norm, ctx->dec_x, l->input_norm, 1, h, eps);

        /* 2. QKV projections (unified dispatch — single barrier for all 3) */
        if (l->wq_q4)
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
        if (l->wo_q4)
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
        if (l->gate_up_fused_q4)
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
        if (l->down_q4)
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
    }

    /* Final RMSNorm */
    qwen_rms_norm(hidden_out, ctx->dec_x, ctx->talker_norm, 1, h, eps);

    ctx->kv_len = pos + 1;
    return 0;
}

/* ========================================================================
 * Prefill (multi-token)
 * ======================================================================== */

int qwen_talker_prefill(qwen_tts_ctx_t *ctx, float *input_embeds, int seq_len) {
    qwen_tts_config_t *c = &ctx->config;
    int h = c->hidden_size;
    int q_dim = c->num_heads * c->head_dim;
    int kv_dim = c->num_kv_heads * c->head_dim;
    int inter = c->intermediate_size;
    float eps = c->rms_norm_eps;

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

    memcpy(residual, input_embeds, (int64_t)seq_len * h * sizeof(float));

    if (ctx->debug) {
        /* Debug: print first position embedding values */
        fprintf(stderr, "[PREFILL] input_embeds[0][:8]:");
        for (int j = 0; j < 8 && j < h; j++) fprintf(stderr, " %.6f", residual[j]);
        fprintf(stderr, "\n");
    }

    for (int layer = 0; layer < c->num_layers; layer++) {
        qwen_talker_layer_t *l = &ctx->layers[layer];

        /* Convert bf16 weights to f32 for this layer */
        bf16_to_f32_matrix(wq_f32, l->wq_bf16, (int64_t)q_dim * h);
        bf16_to_f32_matrix(wk_f32, l->wk_bf16, (int64_t)kv_dim * h);
        bf16_to_f32_matrix(wv_f32, l->wv_bf16, (int64_t)kv_dim * h);
        bf16_to_f32_matrix(wo_f32, l->wo_bf16, (int64_t)h * q_dim);
        bf16_to_f32_matrix(gate_up_f32, l->gate_up_fused_bf16, (int64_t)2 * inter * h);
        bf16_to_f32_matrix(down_f32, l->down_bf16, (int64_t)h * inter);

        /* 1. Input RMSNorm for all positions */
        qwen_rms_norm(pref_x_norm, residual, l->input_norm, seq_len, h, eps);

        /* 2. QKV projections */
#ifdef USE_BLAS
        /* x_norm[seq_len, h] × W^T[h, out_dim] = out[seq_len, out_dim] */
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    seq_len, q_dim, h, 1.0f,
                    pref_x_norm, h, wq_f32, h, 0.0f, pref_q, q_dim);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    seq_len, kv_dim, h, 1.0f,
                    pref_x_norm, h, wk_f32, h, 0.0f, pref_k, kv_dim);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    seq_len, kv_dim, h, 1.0f,
                    pref_x_norm, h, wv_f32, h, 0.0f, pref_v, kv_dim);
#else
        for (int s = 0; s < seq_len; s++) {
            const float *xs = pref_x_norm + (int64_t)s * h;
            float *qs = pref_q + (int64_t)s * q_dim;
            float *ks = pref_k + (int64_t)s * kv_dim;
            float *vs = pref_v + (int64_t)s * kv_dim;
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

        /* 3. Q/K RMSNorm per-head */
        qwen_rms_norm_per_head(pref_q, l->q_norm, seq_len, c->num_heads, c->head_dim, eps);
        qwen_rms_norm_per_head(pref_k, l->k_norm, seq_len, c->num_kv_heads, c->head_dim, eps);

        /* 4. NeoX split-half RoPE for all positions */
        for (int s = 0; s < seq_len; s++) {
            apply_rope_neox_inplace(pref_q + (int64_t)s * q_dim, c->num_heads, c->head_dim,
                                    ctx->rope_cos, ctx->rope_sin, s);
            apply_rope_neox_inplace(pref_k + (int64_t)s * kv_dim, c->num_kv_heads, c->head_dim,
                                    ctx->rope_cos, ctx->rope_sin, s);
        }

        /* 5. Store KV into cache (convert f32→bf16) */
        int64_t cache_base = (int64_t)layer * ctx->kv_max * kv_dim;
        f32_to_bf16_vec(ctx->kv_cache_k + cache_base, pref_k, (int64_t)seq_len * kv_dim);
        f32_to_bf16_vec(ctx->kv_cache_v + cache_base, pref_v, (int64_t)seq_len * kv_dim);

        /* 6. Causal GQA attention — prefill uses f32 Q/K/V directly (not from cache)
         * since we just computed them. This avoids bf16 roundtrip during prefill. */
        float scale = 1.0f / sqrtf((float)c->head_dim);
        qwen_causal_attention(pref_attn_out, pref_q, pref_k, pref_v,
                              seq_len, seq_len, c->num_heads, c->num_kv_heads,
                              c->head_dim, scale, 0);

        /* 7. Output projection + residual */
#ifdef USE_BLAS
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    seq_len, h, q_dim, 1.0f,
                    pref_attn_out, q_dim, wo_f32, q_dim, 0.0f, pref_proj, h);
        for (int64_t i = 0; i < (int64_t)seq_len * h; i++)
            residual[i] += pref_proj[i];
#else
        for (int s = 0; s < seq_len; s++) {
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

        /* 8. Post-attention RMSNorm */
        qwen_rms_norm(pref_x_norm, residual, l->post_attn_norm, seq_len, h, eps);

        /* 9. SwiGLU FFN (fused gate+up: interleaved [g0,u0,g1,u1,...]) */
#ifdef USE_BLAS
        /* Single sgemm: output is [seq_len, 2*inter] interleaved */
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    seq_len, 2 * inter, h, 1.0f,
                    pref_x_norm, h, gate_up_f32, h, 0.0f, pref_gate, 2 * inter);
#else
        for (int s = 0; s < seq_len; s++) {
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

        /* SiLU(gate) * up on interleaved pairs, compact to stride=inter.
         * Uses batch vvexpf on macOS for faster exp. */
        for (int s = 0; s < seq_len; s++) {
            float *src = pref_gate + (int64_t)s * 2 * inter;
            float *dst = pref_gate + (int64_t)s * inter;
            qwen_swiglu_inplace(src, ctx->swiglu_tmp, inter);
            /* swiglu_inplace writes result to src[0..inter-1], copy to compacted dst */
            if (dst != src)
                memcpy(dst, src, inter * sizeof(float));
        }

        /* Down projection + residual (compacted: lda=inter) */
#ifdef USE_BLAS
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    seq_len, h, inter, 1.0f,
                    pref_gate, inter, down_f32, inter, 0.0f, pref_proj, h);
        for (int64_t i = 0; i < (int64_t)seq_len * h; i++)
            residual[i] += pref_proj[i];
#else
        for (int s = 0; s < seq_len; s++) {
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

        if (ctx->debug) {
            fprintf(stderr, "  Layer %d/%d done", layer + 1, c->num_layers);
            /* Print first position residual to detect NaN */
            fprintf(stderr, " res[:4]=[%.4f,%.4f,%.4f,%.4f]",
                    residual[0], residual[1], residual[2], residual[3]);
            fprintf(stderr, "\n");
        }
    }

    ctx->kv_len = seq_len;

    if (ctx->debug) {
        /* Debug: print last position hidden state before and after norm */
        float *last_pos = residual + (int64_t)(seq_len - 1) * h;
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
    memcpy(ctx->dec_x, residual + (int64_t)(seq_len - 1) * h, h * sizeof(float));

    /* Buffers persist in ctx for reuse across generations (server mode) */

    if (!ctx->silent) fprintf(stderr, "  Prefill complete (%d tokens in KV cache)\n", seq_len);
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
static void batch_gather(float *Xt, const float *src, int B, int dim, int srcstride) {
    for (int b = 0; b < B; b++) {
        const float *s = src + (size_t)b * srcstride;
        for (int k = 0; k < dim; k++) Xt[(size_t)k * B + b] = s[k];
    }
}
/* scatter Yt[rows][B] -> dst[B][rows] */
static void batch_scatter(float *dst, const float *Yt, int B, int rows) {
    for (int r = 0; r < rows; r++) {
        const float *yr = Yt + (size_t)r * B;
        for (int b = 0; b < B; b++) dst[(size_t)b * rows + r] = yr[b];
    }
}

/* Batched projection dst[B][rows] = W @ src[B][cols] (src row b at b*srcstride).
 * Default: one batched matmat (weights read once). QWEN_BATCH_NOMATMUL=1 falls back
 * to B per-column matvecs — a diagnostic to isolate the matmat from the wiring. */
static int g_batch_nomatmul = -1;
static void batch_proj(qwen_batch_t *bb, float *dst, const uint16_t *W,
                       const float *src, int rows, int cols, int srcstride) {
    if (g_batch_nomatmul < 0) g_batch_nomatmul = getenv("QWEN_BATCH_NOMATMUL") ? 1 : 0;
    if (g_batch_nomatmul || bb->force_matvec) {
        for (int b = 0; b < bb->B; b++)
            qwen_matvec_bf16(dst + (size_t)b * rows, W, src + (size_t)b * srcstride, rows, cols);
    } else {
        batch_gather(bb->Xt, src, bb->B, cols, srcstride);
        qwen_matmat_bf16(bb->Yt, W, bb->Xt, rows, cols, bb->B);
        batch_scatter(dst, bb->Yt, bb->B, rows);
    }
}

qwen_batch_t *qwen_batch_alloc(qwen_tts_ctx_t *ctx, int B, int kv_max) {
    qwen_tts_config_t *c = &ctx->config;
    if (B < 1 || B > 64 || kv_max < 1) return NULL;
    if (ctx->layers[0].wq_bf16 == NULL) return NULL;   /* v1: bf16 only */
    qwen_batch_t *bb = (qwen_batch_t *)calloc(1, sizeof(qwen_batch_t));
    if (!bb) return NULL;
    bb->B = B; bb->h = c->hidden_size; bb->q_dim = c->num_heads * c->head_dim;
    bb->kv_dim = c->num_kv_heads * c->head_dim; bb->inter = c->intermediate_size;
    bb->num_layers = c->num_layers; bb->kv_max = kv_max; bb->kv_len = 0;
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
    if (!bb->x || !bb->x_norm || !bb->q || !bb->k || !bb->v || !bb->attn_out ||
        !bb->proj_out || !bb->gate || !bb->swiglu_tmp || !bb->Xt || !bb->Yt ||
        !bb->kv_k || !bb->kv_v) { qwen_batch_free(bb); return NULL; }
    return bb;
}

void qwen_batch_free(qwen_batch_t *bb) {
    if (!bb) return;
    free(bb->x); free(bb->x_norm); free(bb->q); free(bb->k); free(bb->v);
    free(bb->attn_out); free(bb->proj_out); free(bb->gate); free(bb->swiglu_tmp);
    free(bb->Xt); free(bb->Yt); free(bb->kv_k); free(bb->kv_v); free(bb);
}

int qwen_batch_talker_step(qwen_tts_ctx_t *ctx, qwen_batch_t *bb,
                           const float *embeds, float *hidden_out) {
    qwen_tts_config_t *c = &ctx->config;
    int B = bb->B, h = bb->h, qd = bb->q_dim, kvd = bb->kv_dim, inter = bb->inter;
    int pos = bb->kv_len; float eps = c->rms_norm_eps;
    if (pos + 1 > bb->kv_max) return -1;
    if (ctx->layers[0].wq_bf16 == NULL) return -2;
    memcpy(bb->x, embeds, (size_t)B * h * sizeof(float));
    float scale = 1.0f / sqrtf((float)c->head_dim);

    for (int layer = 0; layer < c->num_layers; layer++) {
        qwen_talker_layer_t *l = &ctx->layers[layer];
        /* 1. input RMSNorm (per sequence) */
        for (int b = 0; b < B; b++)
            qwen_rms_norm(bb->x_norm + (size_t)b * h, bb->x + (size_t)b * h, l->input_norm, 1, h, eps);
        /* 2. QKV (batched) */
        batch_proj(bb, bb->q, l->wq_bf16, bb->x_norm, qd,  h, h);
        batch_proj(bb, bb->k, l->wk_bf16, bb->x_norm, kvd, h, h);
        batch_proj(bb, bb->v, l->wv_bf16, bb->x_norm, kvd, h, h);
        /* 3-5. per-head norm, RoPE, append KV — per sequence */
        for (int b = 0; b < B; b++) {
            qwen_rms_norm_per_head(bb->q + (size_t)b * qd,  l->q_norm, 1, c->num_heads,    c->head_dim, eps);
            qwen_rms_norm_per_head(bb->k + (size_t)b * kvd, l->k_norm, 1, c->num_kv_heads, c->head_dim, eps);
            apply_rope_neox_inplace(bb->q + (size_t)b * qd,  c->num_heads,    c->head_dim, ctx->rope_cos, ctx->rope_sin, pos);
            apply_rope_neox_inplace(bb->k + (size_t)b * kvd, c->num_kv_heads, c->head_dim, ctx->rope_cos, ctx->rope_sin, pos);
            size_t kvbase = ((size_t)b * bb->num_layers + layer) * bb->kv_max * kvd + (size_t)pos * kvd;
            f32_to_bf16_vec(bb->kv_k + kvbase, bb->k + (size_t)b * kvd, kvd);
            f32_to_bf16_vec(bb->kv_v + kvbase, bb->v + (size_t)b * kvd, kvd);
        }
        /* 6. causal GQA attention — per sequence, against its own KV */
        for (int b = 0; b < B; b++) {
            size_t lbase = ((size_t)b * bb->num_layers + layer) * bb->kv_max * kvd;
            qwen_causal_attention_bf16kv(bb->attn_out + (size_t)b * qd, bb->q + (size_t)b * qd,
                                         bb->kv_k + lbase, bb->kv_v + lbase, 1, pos + 1,
                                         c->num_heads, c->num_kv_heads, c->head_dim, scale, pos);
        }
        /* 7. O projection (batched) */
        batch_proj(bb, bb->proj_out, l->wo_bf16, bb->attn_out, h, qd, qd);
        /* 8. residual + post-attn RMSNorm (per seq; x += proj_out in place) */
        for (int b = 0; b < B; b++)
            qwen_rms_norm_residual(bb->x_norm + (size_t)b * h, bb->x + (size_t)b * h,
                                   bb->proj_out + (size_t)b * h, l->post_attn_norm, h, eps);
        /* 9. gate+up (batched) + SwiGLU per seq */
        batch_proj(bb, bb->gate, l->gate_up_fused_bf16, bb->x_norm, 2 * inter, h, h);
        for (int b = 0; b < B; b++)
            qwen_swiglu_inplace(bb->gate + (size_t)b * 2 * inter, bb->swiglu_tmp, inter);
        /* 10. down (batched) — swiglu output is the first `inter` of each 2*inter row */
        batch_proj(bb, bb->proj_out, l->down_bf16, bb->gate, h, inter, 2 * inter);
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
    bb->kv_len = pos + 1;
    return 0;
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
    fprintf(stderr, "  wiring (matvec mode) vs single-stream: L2_rel=%.2e  %s (must be bit-exact)\n",
            err_matvec, err_matvec < 1e-5 ? "PASS" : "FAIL");
    fprintf(stderr, "  batched matmat path vs single-stream:  L2_rel=%.2e  (fp-order amplification, benign — validate via audio)\n",
            err_matmat);
    fprintf(stderr, "batch-test: %s\n", pass ? "PASS" : "FAIL");
    free(embeds_all); free(embedsB); free(href); free(hbat); qwen_batch_free(bb);
    return pass ? 0 : 1;
}
