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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef __ARM_NEON
#include <arm_neon.h>
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

    uint16_t *new_k = (uint16_t *)malloc((int64_t)ctx->config.num_layers * new_max * kv_dim * sizeof(uint16_t));
    uint16_t *new_v = (uint16_t *)malloc((int64_t)ctx->config.num_layers * new_max * kv_dim * sizeof(uint16_t));
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
            l->gate_up_fused_bf16 = (uint16_t *)malloc(2 * (size_t)c->intermediate_size * row_bytes);
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

    /* Allocate KV cache (bf16 — halves memory vs f32) */
    int initial_kv_max = 2048;
    int64_t kv_size = (int64_t)c->num_layers * initial_kv_max * kv_dim;
    ctx->kv_cache_k = (uint16_t *)calloc(kv_size, sizeof(uint16_t));
    ctx->kv_cache_v = (uint16_t *)calloc(kv_size, sizeof(uint16_t));
    ctx->kv_max = initial_kv_max;
    ctx->kv_len = 0;

    /* Allocate decode buffers (single-token step) */
    ctx->dec_x = (float *)calloc(h, sizeof(float));
    ctx->dec_x_norm = (float *)malloc(h * sizeof(float));
    ctx->dec_q = (float *)malloc(q_dim * sizeof(float));
    ctx->dec_k = (float *)malloc(kv_dim * sizeof(float));
    ctx->dec_v = (float *)malloc(kv_dim * sizeof(float));
    ctx->dec_attn_out = (float *)malloc(q_dim * sizeof(float));
    ctx->dec_proj_out = (float *)malloc(h * sizeof(float));
    ctx->dec_gate = (float *)malloc(2 * c->intermediate_size * sizeof(float));
    ctx->dec_up = NULL;  /* unused: gate buffer holds fused gate+up */
    ctx->dec_ffn_out = (float *)malloc(h * sizeof(float));

    /* Allocate RoPE cache */
    int rope_max = 8192;
    int half_dim = c->head_dim / 2;
    ctx->rope_inv_freq = (float *)malloc(half_dim * sizeof(float));
    ctx->rope_cos = (float *)malloc((int64_t)rope_max * half_dim * sizeof(float));
    ctx->rope_sin = (float *)malloc((int64_t)rope_max * half_dim * sizeof(float));

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

        /* 7. Output projection + residual */
        matvec_bf16_local(ctx->dec_proj_out, l->wo_bf16, ctx->dec_attn_out, h, q_dim);
        for (int i = 0; i < h; i++) ctx->dec_x[i] += ctx->dec_proj_out[i];

        /* 8. Post-attention RMSNorm */
        qwen_rms_norm(ctx->dec_x_norm, ctx->dec_x, l->post_attn_norm, 1, h, eps);

        /* 9. Fused gate+up SwiGLU FFN (single matvec, x loaded once) */
        qwen_matvec_bf16(ctx->dec_gate, l->gate_up_fused_bf16, ctx->dec_x_norm,
                          2 * inter, h);
        /* In-place SwiGLU: interleaved [g0,u0,g1,u1,...] → [silu(g0)*u0, ...] */
        for (int o = 0; o < inter; o++) {
            float g = ctx->dec_gate[2 * o];
            float u = ctx->dec_gate[2 * o + 1];
            ctx->dec_gate[o] = g / (1.0f + expf(-g)) * u;
        }

        /* Down projection + residual */
        qwen_matvec_bf16(ctx->dec_proj_out, l->down_bf16, ctx->dec_gate, h, inter);
        for (int i = 0; i < h; i++) ctx->dec_x[i] += ctx->dec_proj_out[i];
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

    /* Allocate prefill working buffer (separate from dec_x which is single-token) */
    float *residual = (float *)malloc((int64_t)seq_len * h * sizeof(float));
    float *pref_q = (float *)malloc((int64_t)seq_len * q_dim * sizeof(float));
    float *pref_k = (float *)malloc((int64_t)seq_len * kv_dim * sizeof(float));
    float *pref_v = (float *)malloc((int64_t)seq_len * kv_dim * sizeof(float));
    float *pref_x_norm = (float *)malloc((int64_t)seq_len * h * sizeof(float));
    float *pref_attn_out = (float *)malloc((int64_t)seq_len * q_dim * sizeof(float));
    float *pref_gate = (float *)malloc((int64_t)seq_len * 2 * inter * sizeof(float));
    float *pref_proj = (float *)malloc((int64_t)seq_len * h * sizeof(float));

    /* Temp f32 weight buffers for prefill matmul (converted per-layer) */
    float *wq_f32 = (float *)malloc((int64_t)q_dim * h * sizeof(float));
    float *wk_f32 = (float *)malloc((int64_t)kv_dim * h * sizeof(float));
    float *wv_f32 = (float *)malloc((int64_t)kv_dim * h * sizeof(float));
    float *wo_f32 = (float *)malloc((int64_t)h * q_dim * sizeof(float));
    float *gate_up_f32 = (float *)malloc((int64_t)2 * inter * h * sizeof(float));
    float *down_f32 = (float *)malloc((int64_t)h * inter * sizeof(float));

    if (!residual || !pref_q || !pref_k || !pref_v || !pref_x_norm ||
        !pref_attn_out || !pref_gate || !pref_proj ||
        !wq_f32 || !wk_f32 || !wv_f32 || !wo_f32 || !gate_up_f32 || !down_f32) {
        fprintf(stderr, "Error: prefill allocation failed\n");
        free(residual); free(pref_q); free(pref_k); free(pref_v); free(pref_x_norm);
        free(pref_attn_out); free(pref_gate); free(pref_proj);
        free(wq_f32); free(wk_f32); free(wv_f32); free(wo_f32);
        free(gate_up_f32); free(down_f32);
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

        /* SiLU(gate) * up on interleaved pairs, compact to stride=inter */
        for (int s = 0; s < seq_len; s++) {
            float *src = pref_gate + (int64_t)s * 2 * inter;
            float *dst = pref_gate + (int64_t)s * inter;
            for (int o = 0; o < inter; o++) {
                float g = src[2 * o];
                float u = src[2 * o + 1];
                dst[o] = g / (1.0f + expf(-g)) * u;
            }
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

    /* Free prefill buffers */
    free(residual);
    free(pref_q); free(pref_k); free(pref_v);
    free(pref_x_norm); free(pref_attn_out);
    free(pref_gate); free(pref_proj);
    free(wq_f32); free(wk_f32); free(wv_f32); free(wo_f32);
    free(gate_up_f32); free(down_f32);

    if (!ctx->silent) fprintf(stderr, "  Prefill complete (%d tokens in KV cache)\n", seq_len);
    return 0;
}
