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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef __ARM_NEON
#include <arm_neon.h>
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
            l->gate_up_fused_bf16 = (uint16_t *)malloc(2 * (size_t)c->cp_intermediate_size * row_bytes);
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
            ctx->cp_mtp_proj_bias = (float *)malloc(cp_h * sizeof(float));
            for (int i = 0; i < cp_h; i++) ctx->cp_mtp_proj_bias[i] = bf16_to_f32(bias_bf16[i]);
        } else {
            ctx->cp_mtp_proj_bias = NULL;
        }
        ctx->cp_emb_dim = talker_h;  /* CP embeddings have talker_hidden dim */
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
    ctx->cp_kv_k = (uint16_t *)calloc(cp_kv_size, sizeof(uint16_t));
    ctx->cp_kv_v = (uint16_t *)calloc(cp_kv_size, sizeof(uint16_t));
    ctx->cp_kv_max = cp_kv_max;
    ctx->cp_kv_len = 0;

    /* Allocate CP decode buffers */
    ctx->cp_dec_x = (float *)malloc(cp_h * sizeof(float));
    ctx->cp_dec_q = (float *)malloc(cp_q_dim * sizeof(float));
    ctx->cp_dec_k = (float *)malloc(cp_kv_dim * sizeof(float));
    ctx->cp_dec_v = (float *)malloc(cp_kv_dim * sizeof(float));
    ctx->cp_dec_attn_out = (float *)malloc(cp_q_dim * sizeof(float));
    ctx->cp_dec_gate = (float *)malloc(2 * c->cp_intermediate_size * sizeof(float));
    ctx->cp_dec_up = NULL;  /* unused: gate buffer holds fused gate+up */
    ctx->cp_dec_ffn_out = (float *)malloc(cp_h * sizeof(float));

    /* CP RoPE cache (same theta as talker) */
    int half_dim = c->cp_head_dim / 2;
    ctx->cp_rope_cos = (float *)malloc((int64_t)cp_kv_max * half_dim * sizeof(float));
    ctx->cp_rope_sin = (float *)malloc((int64_t)cp_kv_max * half_dim * sizeof(float));
    for (int pos = 0; pos < cp_kv_max; pos++) {
        for (int i = 0; i < half_dim; i++) {
            float angle = (float)pos * (1.0f / powf(c->rope_theta, (float)(2*i) / c->cp_head_dim));
            ctx->cp_rope_cos[pos * half_dim + i] = cosf(angle);
            ctx->cp_rope_sin[pos * half_dim + i] = sinf(angle);
        }
    }
    ctx->cp_rope_cache_len = cp_kv_max;

    /* INT8 quantization of CP weights (optional, enabled by --int8 flag) */
    if (ctx->use_int8) {
        if (!ctx->silent)
            fprintf(stderr, "  Quantizing CP weights to INT8 (per-row absmax)...\n");
        int cp_inter = c->cp_intermediate_size;
        for (int i = 0; i < c->cp_num_layers; i++) {
            qwen_cp_layer_t *l = &ctx->cp_layers[i];

            /* QKV + O projections */
            l->wq_int8 = (int8_t *)malloc((size_t)cp_q_dim * cp_h);
            l->wq_scale = (float *)malloc(cp_q_dim * sizeof(float));
            qwen_quantize_bf16_to_int8(l->wq_bf16, cp_q_dim, cp_h, l->wq_int8, l->wq_scale);

            l->wk_int8 = (int8_t *)malloc((size_t)cp_kv_dim * cp_h);
            l->wk_scale = (float *)malloc(cp_kv_dim * sizeof(float));
            qwen_quantize_bf16_to_int8(l->wk_bf16, cp_kv_dim, cp_h, l->wk_int8, l->wk_scale);

            l->wv_int8 = (int8_t *)malloc((size_t)cp_kv_dim * cp_h);
            l->wv_scale = (float *)malloc(cp_kv_dim * sizeof(float));
            qwen_quantize_bf16_to_int8(l->wv_bf16, cp_kv_dim, cp_h, l->wv_int8, l->wv_scale);

            l->wo_int8 = (int8_t *)malloc((size_t)cp_h * cp_q_dim);
            l->wo_scale = (float *)malloc(cp_h * sizeof(float));
            qwen_quantize_bf16_to_int8(l->wo_bf16, cp_h, cp_q_dim, l->wo_int8, l->wo_scale);

            /* Fused gate+up + down */
            l->gate_up_fused_int8 = (int8_t *)malloc((size_t)2 * cp_inter * cp_h);
            l->gate_up_fused_scale = (float *)malloc(2 * cp_inter * sizeof(float));
            qwen_quantize_bf16_to_int8(l->gate_up_fused_bf16, 2 * cp_inter, cp_h,
                                        l->gate_up_fused_int8, l->gate_up_fused_scale);

            l->down_int8 = (int8_t *)malloc((size_t)cp_h * cp_inter);
            l->down_scale = (float *)malloc(cp_h * sizeof(float));
            qwen_quantize_bf16_to_int8(l->down_bf16, cp_h, cp_inter, l->down_int8, l->down_scale);
        }

        /* LM heads */
        for (int g = 0; g < 15; g++) {
            if (ctx->cp_lm_head_bf16[g]) {
                ctx->cp_lm_head_int8[g] = (int8_t *)malloc((size_t)c->codebook_size * cp_h);
                ctx->cp_lm_head_scale[g] = (float *)malloc(c->codebook_size * sizeof(float));
                qwen_quantize_bf16_to_int8(ctx->cp_lm_head_bf16[g], c->codebook_size, cp_h,
                                            ctx->cp_lm_head_int8[g], ctx->cp_lm_head_scale[g]);
            }
        }
        if (!ctx->silent)
            fprintf(stderr, "  INT8 quantization done (%d layers + 15 lm_heads)\n", c->cp_num_layers);
    }

    if (!ctx->silent)
        fprintf(stderr, "  Code Predictor: %d layers loaded, q_dim=%d kv_dim=%d%s\n",
                c->cp_num_layers, cp_q_dim, cp_kv_dim,
                ctx->use_int8 ? " [INT8]" : "");

    return 0;
}

/* ========================================================================
 * Single CP transformer step at given position
 * ======================================================================== */

static void cp_transformer_step(qwen_tts_ctx_t *ctx, float *x, float *x_norm, int pos) {
    qwen_tts_config_t *c = &ctx->config;
    int cp_h = c->cp_hidden_size;
    int cp_q_dim = c->cp_num_heads * c->cp_head_dim;
    int cp_kv_dim = c->cp_num_kv_heads * c->cp_head_dim;
    int cp_inter = c->cp_intermediate_size;
    float eps = c->rms_norm_eps;

    for (int layer = 0; layer < c->cp_num_layers; layer++) {
        qwen_cp_layer_t *l = &ctx->cp_layers[layer];

        /* 1. Input RMSNorm */
        qwen_rms_norm(x_norm, x, l->input_norm, 1, cp_h, eps);

        /* 2. QKV projections (unified dispatch — single barrier for all 3) */
        if (l->wq_int8) {
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

        /* 3. Q/K RMSNorm per-head */
        qwen_rms_norm_per_head(ctx->cp_dec_q, l->q_norm, 1, c->cp_num_heads, c->cp_head_dim, eps);
        qwen_rms_norm_per_head(ctx->cp_dec_k, l->k_norm, 1, c->cp_num_kv_heads, c->cp_head_dim, eps);

        /* 4. NeoX RoPE */
        apply_rope_neox(ctx->cp_dec_q, c->cp_num_heads, c->cp_head_dim,
                        ctx->cp_rope_cos, ctx->cp_rope_sin, pos);
        apply_rope_neox(ctx->cp_dec_k, c->cp_num_kv_heads, c->cp_head_dim,
                        ctx->cp_rope_cos, ctx->cp_rope_sin, pos);

        /* 5. Store KV in cache (convert f32→bf16) */
        int64_t kv_off = (int64_t)layer * ctx->cp_kv_max * cp_kv_dim + (int64_t)pos * cp_kv_dim;
        f32_to_bf16_vec(ctx->cp_kv_k + kv_off, ctx->cp_dec_k, cp_kv_dim);
        f32_to_bf16_vec(ctx->cp_kv_v + kv_off, ctx->cp_dec_v, cp_kv_dim);

        /* 6. Causal GQA attention (bf16 KV cache) */
        float scale = 1.0f / sqrtf((float)c->cp_head_dim);
        uint16_t *layer_k = ctx->cp_kv_k + (int64_t)layer * ctx->cp_kv_max * cp_kv_dim;
        uint16_t *layer_v = ctx->cp_kv_v + (int64_t)layer * ctx->cp_kv_max * cp_kv_dim;
        qwen_causal_attention_bf16kv(ctx->cp_dec_attn_out, ctx->cp_dec_q, layer_k, layer_v,
                                     1, pos + 1, c->cp_num_heads, c->cp_num_kv_heads,
                                     c->cp_head_dim, scale, pos);

        /* 7. Output projection + residual */
        float *proj = ctx->cp_dec_ffn_out; /* reuse buffer */
        if (l->wo_int8)
            qwen_matvec_int8(proj, l->wo_int8, l->wo_scale, ctx->cp_dec_attn_out, cp_h, cp_q_dim);
        else
            matvec_bf16(proj, l->wo_bf16, ctx->cp_dec_attn_out, cp_h, cp_q_dim);
        for (int i = 0; i < cp_h; i++) x[i] += proj[i];

        /* 8. Post-attention RMSNorm */
        qwen_rms_norm(x_norm, x, l->post_attn_norm, 1, cp_h, eps);

        /* 9. Fused gate+up SwiGLU FFN (single matvec, x loaded once) */
        if (l->gate_up_fused_int8)
            qwen_matvec_int8(ctx->cp_dec_gate, l->gate_up_fused_int8, l->gate_up_fused_scale,
                              x_norm, 2 * cp_inter, cp_h);
        else
            matvec_bf16(ctx->cp_dec_gate, l->gate_up_fused_bf16, x_norm, 2 * cp_inter, cp_h);
        for (int o = 0; o < cp_inter; o++) {
            float g = ctx->cp_dec_gate[2 * o];
            float u = ctx->cp_dec_gate[2 * o + 1];
            ctx->cp_dec_gate[o] = g / (1.0f + expf(-g)) * u;
        }

        /* Down projection + residual */
        if (l->down_int8)
            qwen_matvec_int8(proj, l->down_int8, l->down_scale, ctx->cp_dec_gate, cp_h, cp_inter);
        else
            matvec_bf16(proj, l->down_bf16, ctx->cp_dec_gate, cp_h, cp_inter);
        for (int i = 0; i < cp_h; i++) x[i] += proj[i];
    }
}

/* ========================================================================
 * Code Predictor: generate codebooks 1-15
 *
 * For each frame:
 * - Prefill: pos=0 = talker_hidden, pos=1 = codec_embed(code0)
 * - Then 14 autoregressive steps (pos=2..15), each feeding the previous codebook embed
 * - After each step: apply final norm, compute logits via lm_head, sample
 * ======================================================================== */

/* Apply small_to_mtp_projection: projects from emb_dim to cp_hidden.
 * If no projection needed (0.6B), just copies the first cp_h elements.
 * src has dim=emb_dim, dst has dim=cp_h. */
static void cp_mtp_project(qwen_tts_ctx_t *ctx, float *dst, const float *src) {
    int cp_h = ctx->config.cp_hidden_size;
    if (ctx->cp_mtp_proj_bf16) {
        /* Linear: dst = W @ src + bias, W is [cp_h, emb_dim] in bf16 */
        int emb_dim = ctx->cp_emb_dim;
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

    /* Reset CP KV cache for this frame */
    ctx->cp_kv_len = 0;

    /* Pre-allocated buffers reused across frames (avoid per-frame malloc) */
    float *cp_x = ctx->cp_dec_x;
    float *cp_normed = ctx->cp_dec_attn_out; /* reuse: not overlapping with transformer step output */
    float *x_norm = ctx->cp_dec_ffn_out;     /* reuse: scratch for transformer step */

    /* Step 0: process talker hidden state (project if needed) */
    cp_mtp_project(ctx, cp_x, talker_hidden);
    cp_transformer_step(ctx, cp_x, x_norm, 0);

    /* Step 1: embed code0 using TALKER's codec embedding (NOT CP's).
     * The embedding has dim=hidden_size (talker), so project to cp_hidden. */
    {
        int h = c->hidden_size;  /* talker hidden = embedding dim for talker codec emb */
        float emb_buf[4096];  /* max talker_hidden is 2048, but use 4096 for safety */
        if (ctx->codec_embedding_bf16 && code0 >= 0 && code0 < c->codec_vocab_size) {
            const uint16_t *e = ctx->codec_embedding_bf16 + (int64_t)code0 * h;
            for (int i = 0; i < h; i++) emb_buf[i] = bf16_to_f32(e[i]);
        } else {
            memset(emb_buf, 0, h * sizeof(float));
        }
        cp_mtp_project(ctx, cp_x, emb_buf);
    }
    cp_transformer_step(ctx, cp_x, x_norm, 1);

    /* Predict codebook 1: fused argmax+matvec (greedy — avoids writing 2048 logits) */
    qwen_rms_norm(cp_normed, cp_x, ctx->cp_norm, 1, cp_h, c->rms_norm_eps);
    if (ctx->cp_lm_head_int8[0])
        out_codes[0] = qwen_argmax_matvec_int8(cp_normed, ctx->cp_lm_head_int8[0],
                                                ctx->cp_lm_head_scale[0], cp_h, c->codebook_size);
    else
        out_codes[0] = qwen_argmax_matvec_bf16(cp_normed, ctx->cp_lm_head_bf16[0], cp_h, c->codebook_size);

    /* Steps 2-15: generate codebooks 2-15. */
    for (int g = 1; g < 15; g++) {
        int prev_code = out_codes[g - 1];
        int pos = g + 1;

        /* Embed previous code using CP codec_emb[g-1] (NOT [g]).
         * CP embeddings have dim=emb_dim (talker_hidden for 1.7B), project to cp_hidden. */
        float emb_buf[4096];
        if (ctx->cp_codec_emb_bf16[g - 1] && prev_code >= 0 && prev_code < c->codebook_size) {
            const uint16_t *e = ctx->cp_codec_emb_bf16[g - 1] + (int64_t)prev_code * emb_dim;
            for (int i = 0; i < emb_dim; i++) emb_buf[i] = bf16_to_f32(e[i]);
            cp_mtp_project(ctx, cp_x, emb_buf);
        } else {
            memset(cp_x, 0, cp_h * sizeof(float));
        }

        cp_transformer_step(ctx, cp_x, x_norm, pos);

        /* Fused argmax+matvec (greedy) */
        qwen_rms_norm(cp_normed, cp_x, ctx->cp_norm, 1, cp_h, c->rms_norm_eps);
        if (ctx->cp_lm_head_int8[g])
            out_codes[g] = qwen_argmax_matvec_int8(cp_normed, ctx->cp_lm_head_int8[g],
                                                    ctx->cp_lm_head_scale[g], cp_h, c->codebook_size);
        else
            out_codes[g] = qwen_argmax_matvec_bf16(cp_normed, ctx->cp_lm_head_bf16[g], cp_h, c->codebook_size);
    }

    return 0;
}
