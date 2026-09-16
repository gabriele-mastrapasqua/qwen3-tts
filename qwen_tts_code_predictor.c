/* qwen_tts_code_predictor.c - Code Predictor (MTP) forward pass */
#include "qwen_tts.h"
#include "qwen_tts_kernels.h"
#include "qwen_tts_costmap.h"
#include "ingot/safetensors.h"
#include "qwen_tts_batch.h"
#include "qwen_tts_thread.h"
#include "qwen_tts_kleidi.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif
#ifdef __AVX2__
#include <immintrin.h>
#endif

#ifdef CP_MICROBENCH
#include <sys/time.h>
typedef enum {
    CPB_EMBED, CPB_INNORM, CPB_QKV, CPB_QKNORM, CPB_ROPE, CPB_KVSTORE,
    CPB_ATTN, CPB_OPROJ, CPB_RESNORM, CPB_FFN_GU, CPB_SWIGLU, CPB_FFN_DOWN,
    CPB_LMHEAD, CPB_N
} cpb_slot_t;
static double cpb_acc[CPB_N];
static double cpb_t;
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
    const ingot_st_tensor *t = ingot_st_find((ingot_st *)ms, name);
    if (!t || t->dtype != INGOT_DT_BF16) return NULL;
    return (uint16_t *)(uintptr_t)ingot_st_data((ingot_st *)ms, t);
}

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

#define matvec_bf16 qwen_matvec_bf16

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

static void cp_qz(int8_t **dst, float **scale, const uint16_t *src, int rows, int cols) {
    if (!*dst)   *dst   = (int8_t *)aligned_malloc((size_t)rows * cols);
    if (!*scale) *scale = (float *)aligned_malloc((size_t)rows * sizeof(float));
    if (*dst && *scale) qwen_quantize_bf16_to_int8(src, rows, cols, *dst, *scale);
}

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

static void cp_qz_q4(q4_0_block_t **dst, const uint16_t *src, int rows, int cols) {
    int bpr = cols / Q4_0_BLOCK_SIZE;
    if (!*dst) *dst = (q4_0_block_t *)aligned_malloc((size_t)rows * bpr * sizeof(q4_0_block_t));
    if (*dst) qwen_quantize_bf16_to_q4_0(src, rows, cols, *dst);
}

static void cp_qz_q2(q2_0_block_t **dst, const uint16_t *src, int rows, int cols) {
    int bpr = cols / Q2_0_BLOCK_SIZE;
    if (!*dst) *dst = (q2_0_block_t *)aligned_malloc((size_t)rows * bpr * sizeof(q2_0_block_t));
    if (*dst) qwen_quantize_bf16_to_q2_0(src, rows, cols, *dst);
}

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

#define CP_FREE(p) do { if (p) { free(p); (p) = NULL; } } while (0)

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
    }
}

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

void qwen_cp_quantize_q4(qwen_tts_ctx_t *ctx) {
    qwen_tts_config_t *c = &ctx->config;
    if (!ctx->use_int4) return;
    int cp_h = c->cp_hidden_size;
    int cp_q_dim = c->cp_num_heads * c->cp_head_dim;
    int cp_kv_dim = c->cp_num_kv_heads * c->cp_head_dim;
    int cp_inter = c->cp_intermediate_size;
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

    ctx->cp_norm = get_f32(ctx->safetensors, "talker.code_predictor.model.norm.weight");

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

    for (int g = 0; g < 15; g++) {
        char name[256];
        snprintf(name, sizeof(name), "talker.code_predictor.lm_head.%d.weight", g);
        ctx->cp_lm_head_bf16[g] = get_bf16(ctx->safetensors, name);
        snprintf(name, sizeof(name), "talker.code_predictor.model.codec_embedding.%d.weight", g);
        ctx->cp_codec_emb_bf16[g] = get_bf16(ctx->safetensors, name);
    }

    int talker_h = c->hidden_size;
    if (talker_h != cp_h) {
        ctx->cp_mtp_proj_bf16 = get_bf16(ctx->safetensors, "talker.code_predictor.small_to_mtp_projection.weight");
        uint16_t *bias_bf16 = get_bf16(ctx->safetensors, "talker.code_predictor.small_to_mtp_projection.bias");
        if (bias_bf16) {
            ctx->cp_mtp_proj_bias = (float *)aligned_malloc(cp_h * sizeof(float));
            for (int i = 0; i < cp_h; i++) ctx->cp_mtp_proj_bias[i] = bf16_to_f32(bias_bf16[i]);
        } else {
            ctx->cp_mtp_proj_bias = NULL;
        }
        ctx->cp_emb_dim = talker_h;
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
        ctx->cp_emb_dim = cp_h;
    }

    int cp_kv_max = 64;
    int64_t cp_kv_size = (int64_t)c->cp_num_layers * cp_kv_max * cp_kv_dim;
    ctx->cp_kv_k = (uint16_t *)aligned_calloc(cp_kv_size, sizeof(uint16_t));
    ctx->cp_kv_v = (uint16_t *)aligned_calloc(cp_kv_size, sizeof(uint16_t));
    ctx->cp_kv_max = cp_kv_max;
    ctx->cp_kv_len = 0;

    ctx->cp_dec_x = (float *)aligned_malloc(cp_h * sizeof(float));
    ctx->cp_dec_q = (float *)aligned_malloc(cp_q_dim * sizeof(float));
    ctx->cp_dec_k = (float *)aligned_malloc(cp_kv_dim * sizeof(float));
    ctx->cp_dec_v = (float *)aligned_malloc(cp_kv_dim * sizeof(float));
    ctx->cp_dec_attn_out = (float *)aligned_malloc(cp_q_dim * sizeof(float));
    ctx->cp_dec_gate = (float *)aligned_malloc(2 * c->cp_intermediate_size * sizeof(float));
    ctx->cp_dec_up = NULL;
    ctx->cp_dec_ffn_out = (float *)aligned_malloc(cp_h * sizeof(float));

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
        int save = ctx->use_int8; ctx->use_int8 = 1;
        qwen_cp_quantize_int8(ctx);
        ctx->use_int8 = save;
        if (!ctx->silent)
            fprintf(stderr, "  INT8 quantization done (%d layers + 15 lm_heads)\n", c->cp_num_layers);
    }

    if (cp_do_int4) {
        if (!ctx->silent)
            fprintf(stderr, "  Quantizing CP weights to Q4_0 (--int4)...\n");
        int save = ctx->use_int4; ctx->use_int4 = 1;
        qwen_cp_quantize_q4(ctx);
        ctx->use_int4 = save;
        if (!ctx->silent)
            fprintf(stderr, "  Q4_0 quantization done (%d layers + 15 lm_heads)\n", c->cp_num_layers);
    }

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

static void cp_layer_body(qwen_tts_ctx_t *ctx, float *x, float *x_norm, int pos, int layer) {
    qwen_tts_config_t *c = &ctx->config;
    int cp_h = c->cp_hidden_size;
    int cp_q_dim = c->cp_num_heads * c->cp_head_dim;
    int cp_kv_dim = c->cp_num_kv_heads * c->cp_head_dim;
    int cp_inter = c->cp_intermediate_size;
    float eps = c->rms_norm_eps;
    float attn_scale = 1.0f / sqrtf((float)c->cp_head_dim);
    qwen_cp_layer_t *l = &ctx->cp_layers[layer];
    float *proj = ctx->cp_dec_ffn_out;

    qwen_region_begin2(QWEN_RGN_CP_D_QKV);
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
    qwen_region_end2(QWEN_RGN_CP_D_QKV);

    qwen_region_begin2(QWEN_RGN_CP_D_ATTN);
    qwen_rms_norm_per_head(ctx->cp_dec_q, l->q_norm, 1, c->cp_num_heads, c->cp_head_dim, eps);
    qwen_rms_norm_per_head(ctx->cp_dec_k, l->k_norm, 1, c->cp_num_kv_heads, c->cp_head_dim, eps);
    CPB_MARK(CPB_QKNORM);

    apply_rope_neox(ctx->cp_dec_q, c->cp_num_heads, c->cp_head_dim,
                    ctx->cp_rope_cos, ctx->cp_rope_sin, pos);
    apply_rope_neox(ctx->cp_dec_k, c->cp_num_kv_heads, c->cp_head_dim,
                    ctx->cp_rope_cos, ctx->cp_rope_sin, pos);
    CPB_MARK(CPB_ROPE);

    int64_t kv_off = (int64_t)layer * ctx->cp_kv_max * cp_kv_dim + (int64_t)pos * cp_kv_dim;
    f32_to_bf16_vec(ctx->cp_kv_k + kv_off, ctx->cp_dec_k, cp_kv_dim);
    f32_to_bf16_vec(ctx->cp_kv_v + kv_off, ctx->cp_dec_v, cp_kv_dim);
    CPB_MARK(CPB_KVSTORE);

    uint16_t *layer_k = ctx->cp_kv_k + (int64_t)layer * ctx->cp_kv_max * cp_kv_dim;
    uint16_t *layer_v = ctx->cp_kv_v + (int64_t)layer * ctx->cp_kv_max * cp_kv_dim;
    qwen_causal_attention_bf16kv(ctx->cp_dec_attn_out, ctx->cp_dec_q, layer_k, layer_v,
                                 1, pos + 1, c->cp_num_heads, c->cp_num_kv_heads,
                                 c->cp_head_dim, attn_scale, pos);
    CPB_MARK(CPB_ATTN);
    qwen_region_end2(QWEN_RGN_CP_D_ATTN);

    qwen_region_begin2(QWEN_RGN_CP_D_OPROJ);
    if (l->wo_q4)
        qwen_matvec_q4_0(proj, l->wo_q4, ctx->cp_dec_attn_out, cp_h, cp_q_dim);
    else if (l->wo_int8)
        qwen_matvec_int8(proj, l->wo_int8, l->wo_scale, ctx->cp_dec_attn_out, cp_h, cp_q_dim);
    else
        matvec_bf16(proj, l->wo_bf16, ctx->cp_dec_attn_out, cp_h, cp_q_dim);
    CPB_MARK(CPB_OPROJ);

    qwen_rms_norm_residual(x_norm, x, proj, l->post_attn_norm, cp_h, eps);
    CPB_MARK(CPB_RESNORM);
    qwen_region_end2(QWEN_RGN_CP_D_OPROJ);

    qwen_region_begin2(QWEN_RGN_CP_D_GATEUP);
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
    qwen_region_end2(QWEN_RGN_CP_D_GATEUP);

    qwen_region_begin2(QWEN_RGN_CP_D_DOWN);
    if (ql_ffn_on) {
        long z = 0;
        for (int i = 0; i < cp_inter; i++)
            if (fabsf(ctx->cp_dec_gate[i]) < ql_ffn_eps) z++;
        ql_ffn_zero  += z;
        ql_ffn_total += cp_inter;
    }

    if (l->down_q2)
        qwen_matvec_q2_0(proj, l->down_q2, ctx->cp_dec_gate, cp_h, cp_inter);
    else if (l->down_q4)
        qwen_matvec_q4_0(proj, l->down_q4, ctx->cp_dec_gate, cp_h, cp_inter);
    else if (l->down_int8)
        qwen_matvec_int8(proj, l->down_int8, l->down_scale, ctx->cp_dec_gate, cp_h, cp_inter);
    else
        matvec_bf16(proj, l->down_bf16, ctx->cp_dec_gate, cp_h, cp_inter);
    CPB_MARK(CPB_FFN_DOWN);

    if (ctx->cp_roughness > 0.0f && l->down_q2_rough) {
        float proj_q2[2048];
        qwen_matvec_q2_0(proj_q2, l->down_q2_rough, ctx->cp_dec_gate, cp_h, cp_inter);
        float r = ctx->cp_roughness;
        for (int i = 0; i < cp_h; i++) proj[i] = (1.0f - r) * proj[i] + r * proj_q2[i];
    }

    if (layer + 1 < c->cp_num_layers) {
        qwen_rms_norm_residual(x_norm, x, proj, ctx->cp_layers[layer + 1].input_norm, cp_h, eps);
    } else {
        for (int i = 0; i < cp_h; i++) x[i] += proj[i];
    }
    CPB_MARK(CPB_RESNORM);
    qwen_region_end2(QWEN_RGN_CP_D_DOWN);
}

void *g_cuda_cp_state = NULL;
void *g_cuda_cp_batch_state = NULL;
#ifdef QWEN_HAVE_METAL
void *g_metal_cp_state = NULL;
void *g_metal_cp_frame_state = NULL;
void *g_metal_cp_batch_state = NULL;
extern void qwen_metal_cp_batch_step(void *state, float *x, const int *pos_arr);
#endif
#ifdef QWEN_HAVE_CUDA
extern void qwen_cuda_cp_step(void *state, float *x, int pos);
extern void qwen_cuda_cp_batch_step(void *state, float *x, const int *pos_arr, const uint8_t *active);
#endif

static void cp_transformer_step(qwen_tts_ctx_t *ctx, float *x, float *x_norm, int pos) {
    qwen_tts_config_t *c = &ctx->config;
    int cp_h = c->cp_hidden_size;

    extern void *g_gpu_fused_owner;
#ifdef QWEN_HAVE_CUDA
    if (g_cuda_cp_state && ctx == g_gpu_fused_owner && ctx->cp_roughness <= 0.0f) {
        qwen_cuda_cp_step(g_cuda_cp_state, x, pos);
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

    qwen_rms_norm(x_norm, x, ctx->cp_layers[0].input_norm, 1, cp_h, c->rms_norm_eps);
    CPB_MARK(CPB_INNORM);
    for (int layer = 0; layer < c->cp_num_layers; layer++)
        cp_layer_body(ctx, x, x_norm, pos, layer);
}

/* Env/arch half of the cp_prefill2 decision (default ON with AVX-512 VNNI, opt-in
 * elsewhere).  The other half needs the weights: every CP layer int8 or int4. */
int qwen_cp_prefill2_requested(void) {
    const char *e = getenv("QWEN_CP_PREFILL2");
#if defined(__AVX512VNNI__)
    return !(e && e[0] == '0');
#else
    return (e && e[0] == '1');
#endif
}

static int cp_prefill2_mode(qwen_tts_ctx_t *ctx) {
    static __thread int cached = -2;
    if (cached != -2) return cached;
    if (!qwen_cp_prefill2_requested()) return cached = 0;
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

static void cp_prefill2_body(qwen_tts_ctx_t *ctx, int mode, float *x0, float *x1);
static void cp_prefill2(qwen_tts_ctx_t *ctx, int mode, float *x0, float *x1) {
    qwen_region_begin(QWEN_RGN_CP_PREFILL);
    cp_prefill2_body(ctx, mode, x0, x1);
    qwen_region_end(QWEN_RGN_CP_PREFILL);
}
static void cp_prefill2_body(qwen_tts_ctx_t *ctx, int mode, float *x0, float *x1) {
    qwen_tts_config_t *c = &ctx->config;
    int cp_h  = c->cp_hidden_size;
    int qd    = c->cp_num_heads * c->cp_head_dim;
    int kvd   = c->cp_num_kv_heads * c->cp_head_dim;
    int inter = c->cp_intermediate_size;
    float eps = c->rms_norm_eps;
    float attn_scale = 1.0f / sqrtf((float)c->cp_head_dim);

    static __thread float *S = NULL; static __thread size_t S_cap = 0;
    size_t need = (size_t)(2*cp_h   + 2*qd   + 2*kvd   + 2*kvd
                 + qd   + 2*qd   + 2*cp_h   + 4*inter
                 + 2*inter   + 2*inter   + 2*cp_h   + 2*cp_h  );
    if (need > S_cap) {
        float *ns = (float *)realloc(S, need * sizeof(float));
        if (!ns) return;
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

static void cp_mtp_project(qwen_tts_ctx_t *ctx, float *dst, const float *src) {
    int cp_h = ctx->config.cp_hidden_size;
    if (ctx->cp_mtp_proj_bf16) {
        int emb_dim = ctx->cp_emb_dim;
        if (ctx->cp_mtp_proj_q4)
            qwen_matvec_q4_0(dst, ctx->cp_mtp_proj_q4, src, cp_h, emb_dim);
        else if (ctx->cp_mtp_proj_int8)
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

static double cpx_embed, cpx_proj, cpx_step, cpx_norm, cpx_head;

/* Batched-path breakdown.  The cpx_* counters above only instrument qwen_cp_predict, the
 * single-request path; the server runs qwen_batch_cp_predict, and serving profiles put that
 * at roughly half of all work with no way to see inside it.  QWEN_CP_PROFILE=1 splits it into
 * the three things the loop actually alternates between, 15 times per frame:
 *   seed   - embedding lookup / MTP projection that feeds the next codebook  (CPU)
 *   step   - the 5-layer transformer pass                                    (GPU when batched)
 *   head   - rms_norm + lm_head + argmax over the codebook                   (CPU)
 * `step` on the CUDA path also carries three synchronous copies and a full stream sync per
 * call, so a large `step` share does not by itself mean the GPU is busy. */
static double cpb_seed, cpb_step, cpb_head;
static long   cpb_frames;
static int cpb_on(void) {
    static int t = -1;
    if (t < 0) { const char *e = getenv("QWEN_CP_PROFILE"); t = (e && e[0] && e[0] != '0'); }
    return t;
}
static double cpb_now(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}
static void cpb_report(void) {
    double sum = cpb_seed + cpb_step + cpb_head;
    if (sum <= 0.0) return;
    fprintf(stderr,
            "[CP] frames=%ld  seed %.0f ms (%.1f%%)  step %.0f ms (%.1f%%)  head %.0f ms (%.1f%%)"
            "  | per frame: seed %.3f  step %.3f  head %.3f ms\n",
            cpb_frames, cpb_seed, 100.0*cpb_seed/sum, cpb_step, 100.0*cpb_step/sum,
            cpb_head, 100.0*cpb_head/sum,
            cpb_frames ? cpb_seed/(double)cpb_frames : 0.0,
            cpb_frames ? cpb_step/(double)cpb_frames : 0.0,
            cpb_frames ? cpb_head/(double)cpb_frames : 0.0);
}
static int cpx_on(void) {
    static int t = -1;
    if (t < 0) { const char *e = getenv("QWEN_TTFA_TRACE"); t = (e && e[0] && e[0] != '0'); }
    return t;
}
static double cpx_now(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

int qwen_cp_predict(qwen_tts_ctx_t *ctx, float *talker_hidden, int code0, int *out_codes) {
    qwen_mm_component(QWEN_COMP_CP);
    qwen_tts_config_t *c = &ctx->config;
    int cp_h = c->cp_hidden_size;
    int emb_dim = ctx->cp_emb_dim;

#ifdef QWEN_HAVE_METAL
    { extern void *g_metal_cp_frame_state, *g_gpu_fused_owner;
      extern void qwen_metal_cp_frame(void *, const float *, int, int *);
      if (g_metal_cp_frame_state && ctx == g_gpu_fused_owner &&
          ctx->cp_roughness <= 0.0f && !ctx->tf_ref_codes) {
        qwen_metal_cp_frame(g_metal_cp_frame_state, talker_hidden, code0, out_codes);
        return 0;
      } }
#endif

    ql_init();
    qwen_region_begin(QWEN_RGN_CP_DECODE);

    if (ctx->cp_roughness > 0.0f && !ctx->cp_rough_built) cp_build_roughness(ctx);

    ctx->cp_kv_len = 0;

    float *cp_x = ctx->cp_dec_x;
    float *cp_normed = ctx->cp_dec_attn_out;
    float *x_norm = ctx->cp_dec_ffn_out;

    CPB_RESET();
    cp_mtp_project(ctx, cp_x, talker_hidden);

    static __thread float *x1in = NULL;
    if (!x1in) { x1in = (float *)malloc((size_t)cp_h * sizeof(float));
                 if (!x1in) { qwen_region_end(QWEN_RGN_CP_DECODE); return -1; } }
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

    int pf2 = 0;
#ifndef CP_MICROBENCH
    if (ctx->cp_roughness <= 0.0f) pf2 = cp_prefill2_mode(ctx);
#endif
    if (pf2) {
        cp_prefill2(ctx, pf2, cp_x, x1in);
        memcpy(cp_x, x1in, (size_t)cp_h * sizeof(float));
    } else {
        cp_transformer_step(ctx, cp_x, x_norm, 0);
        memcpy(cp_x, x1in, (size_t)cp_h * sizeof(float));
        CPB_MARK(CPB_EMBED);
        cp_transformer_step(ctx, cp_x, x_norm, 1);
    }

    qwen_rms_norm(cp_normed, cp_x, ctx->cp_norm, 1, cp_h, c->rms_norm_eps);
    qwen_region_begin(QWEN_RGN_CP_D_LMHEAD);
    if (ctx->cp_lm_head_q4[0])
        out_codes[0] = qwen_argmax_matvec_q4_0(cp_normed, ctx->cp_lm_head_q4[0], cp_h, c->codebook_size);
    else if (ctx->cp_lm_head_int8[0])
        out_codes[0] = qwen_argmax_matvec_int8(cp_normed, ctx->cp_lm_head_int8[0],
                                                ctx->cp_lm_head_scale[0], cp_h, c->codebook_size);
    else
        out_codes[0] = qwen_argmax_matvec_bf16(cp_normed, ctx->cp_lm_head_bf16[0], cp_h, c->codebook_size);
    qwen_region_end(QWEN_RGN_CP_D_LMHEAD);
    CPB_MARK(CPB_LMHEAD);

    const int _cx = cpx_on();
    double _cm = _cx ? cpx_now() : 0.0, _cf0 = _cm;
    if (_cx) cpx_embed = cpx_proj = cpx_step = cpx_norm = cpx_head = 0.0;
    for (int g = 1; g < 15; g++) {
        if (_cx) _cm = cpx_now();
        int prev_code = ctx->tf_ref_codes ? ctx->tf_ref_codes[g - 1] : out_codes[g - 1];
        int pos = g + 1;

        if (ctx->cp_codec_emb_bf16[g - 1] && prev_code >= 0 && prev_code < c->codebook_size) {
            float emb_buf[4096];
            const uint16_t *e = ctx->cp_codec_emb_bf16[g - 1] + (int64_t)prev_code * emb_dim;
            qwen_bf16_to_f32_vec(emb_buf, e, emb_dim);
            if (_cx) { double _t = cpx_now(); cpx_embed += _t - _cm; _cm = _t; }
            cp_mtp_project(ctx, cp_x, emb_buf);
            if (_cx) { double _t = cpx_now(); cpx_proj += _t - _cm; _cm = _t; }
        } else {
            memset(cp_x, 0, cp_h * sizeof(float));
        }
        CPB_MARK(CPB_EMBED);

        if (_cx) _cm = cpx_now();
        cp_transformer_step(ctx, cp_x, x_norm, pos);
        if (_cx) { double _t = cpx_now(); cpx_step += _t - _cm; _cm = _t; }

        qwen_rms_norm(cp_normed, cp_x, ctx->cp_norm, 1, cp_h, c->rms_norm_eps);
        if (_cx) { double _t = cpx_now(); cpx_norm += _t - _cm; _cm = _t; }
        qwen_region_begin(QWEN_RGN_CP_D_LMHEAD);
        if (ctx->cp_lm_head_q4[g])
            out_codes[g] = qwen_argmax_matvec_q4_0(cp_normed, ctx->cp_lm_head_q4[g], cp_h, c->codebook_size);
        else if (ctx->cp_lm_head_int8[g])
            out_codes[g] = qwen_argmax_matvec_int8(cp_normed, ctx->cp_lm_head_int8[g],
                                                    ctx->cp_lm_head_scale[g], cp_h, c->codebook_size);
        else
            out_codes[g] = qwen_argmax_matvec_bf16(cp_normed, ctx->cp_lm_head_bf16[g], cp_h, c->codebook_size);
        qwen_region_end(QWEN_RGN_CP_D_LMHEAD);
        CPB_MARK(CPB_LMHEAD);
        if (_cx) { double _t = cpx_now(); cpx_head += _t - _cm; _cm = _t; }
    }
    if (_cx) {
        double _tot = cpx_now() - _cf0;
        double _sum = cpx_embed + cpx_proj + cpx_step + cpx_norm + cpx_head;
        fprintf(stderr, "[CPX] v=1 embed=%.4f proj=%.4f step=%.4f norm=%.4f head=%.4f "
                "sum=%.4f total=%.4f unacc=%.4f\n",
                cpx_embed, cpx_proj, cpx_step, cpx_norm, cpx_head, _sum, _tot, _tot - _sum);
    }

    if (ql_codes_fp) {
        fprintf(ql_codes_fp, "%d", code0);
        for (int g = 0; g < 15; g++) fprintf(ql_codes_fp, " %d", out_codes[g]);
        fputc('\n', ql_codes_fp);
    }

    qwen_region_end(QWEN_RGN_CP_DECODE);
    return 0;
}

static int cp_lm_argmax(qwen_tts_ctx_t *ctx, const float *normed, int g, int ch, int vocab) {
    if (ctx->cp_lm_head_q4[g])
        return qwen_argmax_matvec_q4_0(normed, ctx->cp_lm_head_q4[g], ch, vocab);
    if (ctx->cp_lm_head_int8[g])
        return qwen_argmax_matvec_int8(normed, ctx->cp_lm_head_int8[g], ctx->cp_lm_head_scale[g], ch, vocab);
    return qwen_argmax_matvec_bf16(normed, ctx->cp_lm_head_bf16[g], ch, vocab);
}

static void batch_cp_layer(qwen_tts_ctx_t *ctx, qwen_batch_t *bb,
                           float *x, float *x_norm, int pos, int layer, const uint8_t *active) {
    qwen_tts_config_t *c = &ctx->config;
    int B = bb->B, ch = bb->cp_h, cqd = bb->cp_q_dim, ckvd = bb->cp_kv_dim, cint = bb->cp_inter;
    float eps = c->rms_norm_eps, ascale = 1.0f / sqrtf((float)c->cp_head_dim);
    int fm = bb->force_matvec;
    int BW = bb->B_eff > 0 ? bb->B_eff : B;
    qwen_cp_layer_t *l = &ctx->cp_layers[layer];
#define CP_SKIP(b) (active && !active[b])

    qwen_region_begin2(QWEN_RGN_CP_D_QKV);
    qwen_batch_proj_qkv(bb->cp_q, bb->cp_k, bb->cp_v,
                        l->wq_bf16, l->wq_int8, l->wq_scale, l->wq_q4,
                        l->wk_bf16, l->wk_int8, l->wk_scale, l->wk_q4,
                        l->wv_bf16, l->wv_int8, l->wv_scale, l->wv_q4,
                        x_norm, cqd, ckvd, ch, ch, BW, bb->act_idx, fm,
                        bb->cp_Xt, bb->cp_Yt);
    qwen_region_end2(QWEN_RGN_CP_D_QKV);
    qwen_region_begin2(QWEN_RGN_CP_D_ATTN);
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
    qwen_region_end2(QWEN_RGN_CP_D_ATTN);
    qwen_region_begin2(QWEN_RGN_CP_D_OPROJ);
    qwen_batch_proj_q(bb->cp_proj, l->wo_bf16, l->wo_int8, l->wo_scale, l->wo_q4, bb->cp_attn, ch, cqd, cqd, BW, bb->act_idx, fm, bb->cp_Xt, bb->cp_Yt);
    for (int b = 0; b < B; b++) {
        if (CP_SKIP(b)) continue;
        qwen_rms_norm_residual(x_norm + (size_t)b * ch, x + (size_t)b * ch,
                               bb->cp_proj + (size_t)b * ch, l->post_attn_norm, ch, eps);
    }
    qwen_region_end2(QWEN_RGN_CP_D_OPROJ);
    qwen_region_begin2(QWEN_RGN_CP_D_GATEUP);
    qwen_batch_proj_q(bb->cp_gate, l->gate_up_fused_bf16, l->gate_up_fused_int8, l->gate_up_fused_scale,
                      l->gate_up_fused_q4, x_norm, 2 * cint, ch, ch, BW, bb->act_idx, fm, bb->cp_Xt, bb->cp_Yt);
    for (int b = 0; b < B; b++) {
        if (CP_SKIP(b)) continue;
        qwen_swiglu_inplace(bb->cp_gate + (size_t)b * 2 * cint, bb->cp_swiglu_tmp, cint);
    }
    qwen_region_end2(QWEN_RGN_CP_D_GATEUP);
    qwen_region_begin2(QWEN_RGN_CP_D_DOWN);
    qwen_batch_proj_q(bb->cp_proj, l->down_bf16, l->down_int8, l->down_scale, l->down_q4,
                      bb->cp_gate, ch, cint, 2 * cint, BW, bb->act_idx, fm, bb->cp_Xt, bb->cp_Yt);
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
    qwen_region_end2(QWEN_RGN_CP_D_DOWN);
#undef CP_SKIP
}

/* ---- one CP transformer step as ONE persistent parallel region -------------------------
 * The dispatched path leaves and re-enters the pool 20 times per step (4 projections x 5
 * layers) and runs every per-slot section (q/k norm, rope, KV store, attention, residual
 * norms, swiglu) on the loop thread with the pool idle.  Here the whole team enters once,
 * the projections run as the same VNNI row blocks the dispatched path uses, and the per-slot
 * sections run one slot per thread between spin barriers.  Math and kernels are unchanged,
 * so the outputs are bit-identical; QWEN_CP_REGION=0 restores the dispatched path. */
typedef struct {
    qwen_tts_ctx_t *ctx; qwen_batch_t *bb; float *x, *x_norm; int pos;
    int BW; const int *idx;
    int8_t *qx; float *swtmp; float sx[16];
    int arm_kai;                 /* 1: KleidiAI prepared-state runner (Arm) */
    const void *kai_lhs_packed;  /* one pack shared by the whole region team */
    int kai_prep_failed;
    qwen_barrier_t bar;
    /* frame mode only: the whole 15-group decode inside one region */
    const float *talker_hidden; const int *code0; int *out_codes;
} cp_region_t;

/* Same two runners as the Talker region: x86 row blocks take a k-major int8 panel with
 * per-column scales, KleidiAI quantises a row-major f32 activation itself.  The name is
 * kept so the frame-region body (VNNI-only, still off on Arm) compiles unchanged. */
static void cp_region_gather_quant(cp_region_t *r, const float *src, int b, int j,
                                   int cols, int srcstride) {
    const float *s = src + (size_t)b * srcstride;
    if (r->arm_kai) {
        memcpy(r->bb->cp_Xt + (size_t)j * cols, s, (size_t)cols * sizeof(float));
        return;
    }
    float *Xt = r->bb->cp_Xt;
    for (int k = 0; k < cols; k++) Xt[(size_t)k * r->BW + j] = s[k];
    r->sx[j] = qwen_region_i8_quant_col(r->qx + (size_t)j * cols, Xt, cols, r->BW, j);
}
static void cp_region_scatter(cp_region_t *r, float *dst, const float *Y, int b, int j, int rows) {
    float *d = dst + (size_t)b * rows;
    if (r->arm_kai) { memcpy(d, Y + (size_t)j * rows, (size_t)rows * sizeof(float)); return; }
    for (int i = 0; i < rows; i++) d[i] = Y[(size_t)i * r->BW + j];
}

/* Pack the row-major activation once.  qwen_kleidi_*_region_prep() uses TLS scratch,
 * so the leader's packed buffer remains valid for all workers until the phase barrier
 * following the run releases it for the next projection. */
static const void *cp_region_kai_prep(cp_region_t *r, int cols, size_t tid) {
    if (tid == 0) {
        r->kai_lhs_packed = qwen_kleidi_i8_region_prep(
            r->bb->cp_Xt, (size_t)cols * sizeof(float), cols, r->BW);
        r->kai_prep_failed = (r->kai_lhs_packed == NULL);
    }
    qwen_barrier_wait(&r->bar);
    if (r->kai_prep_failed) {
        if (tid == 0) fprintf(stderr, "[cp] KAI region LHS prep failed\n");
        abort();
    }
    return r->kai_lhs_packed;
}

static void cp_region_run_proj(cp_region_t *r, const int8_t *W, const float *sw,
                               int rows, int cols, size_t tid, size_t nt) {
    if (r->arm_kai) {
        const void *lp = cp_region_kai_prep(r, cols, tid);
        qwen_kleidi_i8_region_run(W, r->bb->cp_Yt, (size_t)rows * sizeof(float), lp,
                                  rows, cols, r->BW, tid, nt);
        return;
    }
    qwen_region_i8_run(r->bb->cp_Yt, W, sw, r->qx, r->sx, rows, cols, r->BW, tid, nt);
}
static void cp_region_run_qkv(cp_region_t *r, const int8_t *Wq, const float *sq,
                              const int8_t *Wk, const float *sk,
                              const int8_t *Wv, const float *sv,
                              float *Yk, float *Yv, int q_rows, int kv_rows, int cols,
                              size_t tid, size_t nt) {
    if (r->arm_kai) {
        const void *lp = cp_region_kai_prep(r, cols, tid);
        qwen_kleidi_i8_qkv_region_run(Wq, Wk, Wv, r->bb->cp_Yt, Yk, Yv, lp,
                                      q_rows, kv_rows, cols, r->BW, tid, nt);
        return;
    }
    qwen_region_i8_run_qkv(r->bb->cp_Yt, Yk, Yv, Wq, sq, Wk, sk, Wv, sv,
                           r->qx, r->sx, q_rows, kv_rows, cols, r->BW, tid, nt);
}

/* ph[] = {qkv, attn, out_proj, gate_up, down, other}, or NULL when profiling is off.
 * The four projections are the four DISTINCT AMX gate decisions in this loop (the fused
 * QKV is judged on q+2kv), so lumping them hides exactly what the AMX census must see.
 * 'other' catches every remaining barrier interval so the six always sum to the whole. */
static void cp_region_layers(cp_region_t *r, size_t tid, size_t nt, int pos, uint64_t *ph) {
    uint64_t lmark = ph ? qwen_costmap_now_ns() : 0;
#define CPL_ACC(k) do { if (ph) { uint64_t _n = qwen_costmap_now_ns(); \
                                  ph[(k)] += _n - lmark; lmark = _n; } } while (0)
    qwen_tts_ctx_t *ctx = r->ctx; qwen_batch_t *bb = r->bb; qwen_tts_config_t *c = &ctx->config;
    const int BW = r->BW, ch = bb->cp_h, cqd = bb->cp_q_dim, ckvd = bb->cp_kv_dim, cint = bb->cp_inter;
    const float eps = c->rms_norm_eps, ascale = 1.0f / sqrtf((float)c->cp_head_dim);
    float *Yt = bb->cp_Yt;
#define RSLOT(j) (r->idx ? r->idx[j] : (j))
#define RMINE(j) ((size_t)(j) % nt == tid)
    for (int L = 0; L < c->cp_num_layers; L++) {
        qwen_cp_layer_t *l = &ctx->cp_layers[L];
        float *Yk = Yt + (size_t)cqd * BW, *Yv = Yt + (size_t)(cqd + ckvd) * BW;
        cp_region_run_qkv(r, l->wq_int8, l->wq_scale, l->wk_int8, l->wk_scale,
                          l->wv_int8, l->wv_scale, Yk, Yv, cqd, ckvd, ch, tid, nt);
        qwen_barrier_wait(&r->bar);
        CPL_ACC(0);
        for (int j = 0; j < BW; j++) if (RMINE(j)) {
            int b = RSLOT(j);
            cp_region_scatter(r, bb->cp_q, Yt, b, j, cqd);
            cp_region_scatter(r, bb->cp_k, Yk, b, j, ckvd);
            cp_region_scatter(r, bb->cp_v, Yv, b, j, ckvd);
            qwen_rms_norm_per_head(bb->cp_q + (size_t)b * cqd,  l->q_norm, 1, c->cp_num_heads,    c->cp_head_dim, eps);
            qwen_rms_norm_per_head(bb->cp_k + (size_t)b * ckvd, l->k_norm, 1, c->cp_num_kv_heads, c->cp_head_dim, eps);
            apply_rope_neox(bb->cp_q + (size_t)b * cqd,  c->cp_num_heads,    c->cp_head_dim, ctx->cp_rope_cos, ctx->cp_rope_sin, pos);
            apply_rope_neox(bb->cp_k + (size_t)b * ckvd, c->cp_num_kv_heads, c->cp_head_dim, ctx->cp_rope_cos, ctx->cp_rope_sin, pos);
            size_t kvbase = ((size_t)b * bb->cp_num_layers + L) * bb->cp_kv_max * ckvd + (size_t)pos * ckvd;
            f32_to_bf16_vec(bb->cp_kv_k + kvbase, bb->cp_k + (size_t)b * ckvd, ckvd);
            f32_to_bf16_vec(bb->cp_kv_v + kvbase, bb->cp_v + (size_t)b * ckvd, ckvd);
            size_t lbase = ((size_t)b * bb->cp_num_layers + L) * bb->cp_kv_max * ckvd;
            qwen_causal_attention_bf16kv(bb->cp_attn + (size_t)b * cqd, bb->cp_q + (size_t)b * cqd,
                                         bb->cp_kv_k + lbase, bb->cp_kv_v + lbase, 1, pos + 1,
                                         c->cp_num_heads, c->cp_num_kv_heads, c->cp_head_dim, ascale, pos);
            cp_region_gather_quant(r, bb->cp_attn, b, j, cqd, cqd);
        }
        qwen_barrier_wait(&r->bar);
        CPL_ACC(1);
        cp_region_run_proj(r, l->wo_int8, l->wo_scale, ch, cqd, tid, nt);
        qwen_barrier_wait(&r->bar);
        CPL_ACC(2);
        for (int j = 0; j < BW; j++) if (RMINE(j)) {
            int b = RSLOT(j);
            cp_region_scatter(r, bb->cp_proj, Yt, b, j, ch);
            qwen_rms_norm_residual(r->x_norm + (size_t)b * ch, r->x + (size_t)b * ch,
                                   bb->cp_proj + (size_t)b * ch, l->post_attn_norm, ch, eps);
            cp_region_gather_quant(r, r->x_norm, b, j, ch, ch);
        }
        qwen_barrier_wait(&r->bar);
        CPL_ACC(5);
        cp_region_run_proj(r, l->gate_up_fused_int8, l->gate_up_fused_scale, 2 * cint, ch, tid, nt);
        qwen_barrier_wait(&r->bar);
        CPL_ACC(3);
        for (int j = 0; j < BW; j++) if (RMINE(j)) {
            int b = RSLOT(j);
            cp_region_scatter(r, bb->cp_gate, Yt, b, j, 2 * cint);
            qwen_swiglu_inplace(bb->cp_gate + (size_t)b * 2 * cint, r->swtmp + (size_t)j * cint, cint);
            cp_region_gather_quant(r, bb->cp_gate, b, j, cint, 2 * cint);
        }
        qwen_barrier_wait(&r->bar);
        CPL_ACC(5);
        cp_region_run_proj(r, l->down_int8, l->down_scale, ch, cint, tid, nt);
        qwen_barrier_wait(&r->bar);
        CPL_ACC(4);
        for (int j = 0; j < BW; j++) if (RMINE(j)) {
            int b = RSLOT(j);
            cp_region_scatter(r, bb->cp_proj, Yt, b, j, ch);
            if (L + 1 < c->cp_num_layers) {
                qwen_rms_norm_residual(r->x_norm + (size_t)b * ch, r->x + (size_t)b * ch,
                                       bb->cp_proj + (size_t)b * ch, ctx->cp_layers[L + 1].input_norm, ch, eps);
                cp_region_gather_quant(r, r->x_norm, b, j, ch, ch);
            } else {
                float *xb = r->x + (size_t)b * ch, *pb = bb->cp_proj + (size_t)b * ch;
                for (int i = 0; i < ch; i++) xb[i] += pb[i];
            }
        }
        qwen_barrier_wait(&r->bar);
        CPL_ACC(5);
    }
#undef RSLOT
#undef RMINE
}
#undef CPL_ACC

static void cp_region_task(size_t tid, size_t nt, void *v) {
    cp_region_t *r = (cp_region_t *)v;
    const int BW = r->BW, ch = r->bb->cp_h;
    for (int j = 0; j < BW; j++)
        if ((size_t)j % nt == tid)
            cp_region_gather_quant(r, r->x_norm, r->idx ? r->idx[j] : j, j, ch, ch);
    qwen_barrier_wait(&r->bar);
    cp_region_layers(r, tid, nt, r->pos, NULL);
}

/* Can this step run as one region?  Decided once per process for the CP shapes (they never
 * change) and re-checked for the cheap per-call conditions. */
static int cp_region_mode = -1;   /* 0 off, 1 x86 row blocks, 2 Arm KAI prepared state */
static int cp_region_arm(void) { return cp_region_mode == 2; }
static int cp_region_ok(qwen_tts_ctx_t *ctx, qwen_batch_t *bb, int BW) {
    if (cp_region_mode < 0) {
        const char *e = getenv("QWEN_CP_REGION");
        qwen_cp_layer_t *l = &ctx->cp_layers[0];
        const int want = !(e && e[0] == '0') && qwen_parallel_team() >= 2 && bb->B >= 2 &&
                    l->wq_int8 && l->wk_int8 && l->wv_int8 && l->wo_int8 &&
                    l->gate_up_fused_int8 && l->down_int8 &&
                    !l->wq_q4 && !l->wk_q4 && !l->wv_q4 && !l->wo_q4 && !l->gate_up_fused_q4 && !l->down_q4;
        const int vnni = want &&
                    qwen_region_i8_qkv_usable(bb->cp_q_dim, bb->cp_kv_dim, bb->cp_h, 2) &&
                    qwen_region_i8_usable(bb->cp_h, bb->cp_q_dim, 2) &&
                    qwen_region_i8_usable(2 * bb->cp_inter, bb->cp_h, 2) &&
                    qwen_region_i8_usable(bb->cp_h, bb->cp_inter, 2);
        /* The Arm runner reuses the prepared-state API the Talker region now wires. */
        const int kai = !vnni && want &&
                    qwen_kleidi_i8_qkv_region_usable(l->wq_int8, l->wk_int8, l->wv_int8,
                                                     bb->cp_q_dim, bb->cp_kv_dim, bb->cp_h, 2) &&
                    qwen_kleidi_i8_region_usable(l->wo_int8, bb->cp_h, bb->cp_q_dim, 2) &&
                    qwen_kleidi_i8_region_usable(l->gate_up_fused_int8, 2 * bb->cp_inter, bb->cp_h, 2) &&
                    qwen_kleidi_i8_region_usable(l->down_int8, bb->cp_h, bb->cp_inter, 2);
        cp_region_mode = vnni ? 1 : (kai ? 2 : 0);
        fprintf(stderr, "[cp] transformer step as one parallel region: %s (team %d)\n",
                cp_region_mode == 1 ? "ON" : cp_region_mode == 2 ? "ON (KleidiAI prepared state)" : "off",
                qwen_parallel_team());
    }
    if (cp_region_mode == 0 || bb->force_matvec || BW < 2 || BW > 16) return 0;
    if (cp_region_mode == 1)
        return qwen_region_i8_qkv_usable(bb->cp_q_dim, bb->cp_kv_dim, bb->cp_h, BW) &&
               qwen_region_i8_usable(bb->cp_h, bb->cp_q_dim, BW) &&
               qwen_region_i8_usable(2 * bb->cp_inter, bb->cp_h, BW) &&
               qwen_region_i8_usable(bb->cp_h, bb->cp_inter, BW);
    return 1;
}

/* One grow-once scratch pair for both region entry points: qx holds BW quantised activation
 * columns of the widest operand, swtmp the per-slot swiglu temporary. */
static int cp_region_scratch(int BW, size_t maxc, int cint, int8_t **pqx, float **psw) {
    static int8_t *qx = NULL; static float *swtmp = NULL; static size_t qx_cap = 0, sw_cap = 0;
    size_t need = (size_t)BW * maxc + 64, swn = (size_t)BW * cint;
    if (need > qx_cap) { free(qx); qx = (int8_t *)aligned_alloc(64, (need + 63) & ~(size_t)63); qx_cap = qx ? need : 0; }
    if (swn > sw_cap) { free(swtmp); swtmp = (float *)malloc(swn * sizeof(float)); sw_cap = swtmp ? swn : 0; }
    *pqx = qx; *psw = swtmp;
    return qx && swtmp;
}

static void batch_cp_transformer_step(qwen_tts_ctx_t *ctx, qwen_batch_t *bb,
                                      float *x, float *x_norm, int pos, const uint8_t *active) {
    qwen_tts_config_t *c = &ctx->config;
    int B = bb->B, ch = bb->cp_h; float eps = c->rms_norm_eps;
#ifdef QWEN_HAVE_CUDA
    extern void *g_cuda_cp_batch_state;
    if (g_cuda_cp_batch_state && B <= 16) {
        int pos_arr[16]; for (int b = 0; b < B; b++) pos_arr[b] = pos;
        /* `active` must reach the device: a lane the caller is not stepping keeps a stale
         * position, and cp_kv_max is 64, so indexing with it runs off the cache. */
        qwen_cuda_cp_batch_step(g_cuda_cp_batch_state, x, pos_arr, active);
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
    {
        int BW = bb->B_eff > 0 ? bb->B_eff : B;
        if (cp_region_ok(ctx, bb, BW)) {
            int8_t *qx = NULL; float *swtmp = NULL;
            size_t maxc = (size_t)(bb->cp_inter > bb->cp_q_dim ? bb->cp_inter : bb->cp_q_dim);
            if (maxc < (size_t)ch) maxc = ch;
            if (cp_region_scratch(BW, maxc, bb->cp_inter, &qx, &swtmp)) {
                cp_region_t r; memset(&r, 0, sizeof r);
                r.ctx = ctx; r.bb = bb; r.x = x; r.x_norm = x_norm; r.pos = pos;
                r.BW = BW; r.idx = bb->act_idx; r.qx = qx; r.swtmp = swtmp;
                r.arm_kai = cp_region_arm();
                int team = qwen_parallel_team();
                qwen_barrier_init(&r.bar, team);
                qwen_parallel((size_t)team, cp_region_task, &r);
                return;
            }
        }
    }
    for (int layer = 0; layer < c->cp_num_layers; layer++)
        batch_cp_layer(ctx, bb, x, x_norm, pos, layer, active);
}

/* ---- B-batched code-predictor head/projection ---------------------------------------
 * At concurrency >= 2 the MTP projection and every lm_head were run once per slot as
 * B=1 GEMVs, i.e. the same 2 MB weight was streamed once per slot per group.  These run
 * the active slots through one int8 matmat (same per-column quantiser, exact int32 dots,
 * same scaling expression), so the weight leaves DRAM once per step.  They return 0 when
 * the int8 matmat path is not the one that would run, and the caller keeps the per-slot
 * path; QWEN_CP_BATCH_HEAD=0 forces the per-slot path. */
static int cp_batch_head_enabled(void) {
    static int on = -1;
    if (on < 0) { const char *e = getenv("QWEN_CP_BATCH_HEAD"); on = !(e && e[0] == '0'); }
    return on;
}
static int cp_batch_mtp(qwen_tts_ctx_t *ctx, qwen_batch_t *bb, float *cx,
                        const float *const *src, const uint8_t *active) {
    int B = bb->B, ch = ctx->config.cp_hidden_size, ed = ctx->cp_emb_dim;
    if (!cp_batch_head_enabled() || !ctx->cp_mtp_proj_int8 || ctx->cp_mtp_proj_q4) return 0;
    int idx[64], BW = 0;
    for (int b = 0; b < B && BW < 64; b++) if (!active || active[b]) idx[BW++] = b;
    if (BW < 2 || BW > 16 || !qwen_region_i8_usable(ch, ed, BW)) return 0;
    float *Xt = bb->cp_Xt, *Yt = bb->cp_Yt;
    for (int j = 0; j < BW; j++) { const float *x = src[idx[j]]; for (int k = 0; k < ed; k++) Xt[(size_t)k * BW + j] = x[k]; }
    qwen_matmat_int8(Yt, ctx->cp_mtp_proj_int8, ctx->cp_mtp_proj_scale, Xt, ch, ed, BW);
    const float *bias = ctx->cp_mtp_proj_bias;
    for (int j = 0; j < BW; j++) {
        float *d = cx + (size_t)idx[j] * ch;
        for (int i = 0; i < ch; i++) d[i] = Yt[(size_t)i * BW + j];
        if (bias) for (int i = 0; i < ch; i++) d[i] += bias[i];
    }
    return 1;
}
static int cp_batch_lm(qwen_tts_ctx_t *ctx, qwen_batch_t *bb, const float *normed_rows, int g,
                       int *out_codes, const uint8_t *active) {
    int B = bb->B, ch = ctx->config.cp_hidden_size, vocab = ctx->config.codebook_size;
    if (!cp_batch_head_enabled() || !ctx->cp_lm_head_int8[g] || ctx->cp_lm_head_q4[g]) return 0;
    int idx[64], BW = 0;
    for (int b = 0; b < B && BW < 64; b++) if (!active || active[b]) idx[BW++] = b;
    if (BW < 2 || BW > 16 || !qwen_region_i8_usable(vocab, ch, BW)) return 0;
    float *Xt = bb->cp_Xt, *Yt = bb->cp_Yt;
    for (int j = 0; j < BW; j++) { const float *x = normed_rows + (size_t)idx[j] * ch; for (int k = 0; k < ch; k++) Xt[(size_t)k * BW + j] = x[k]; }
    qwen_matmat_int8(Yt, ctx->cp_lm_head_int8[g], ctx->cp_lm_head_scale[g], Xt, vocab, ch, BW);
    for (int j = 0; j < BW; j++) {
        int best = 0; float bv = Yt[j];
        for (int o = 1; o < vocab; o++) { float v = Yt[(size_t)o * BW + j]; if (v > bv) { bv = v; best = o; } }
        out_codes[(size_t)idx[j] * 15 + g] = best;
    }
    return 1;
}

/* ---- the whole CP frame as ONE parallel region ----------------------------------------
 * Even with the step region and the batched heads on, a 16-step frame still leaves and
 * re-enters the pool 47 times: 16 steps, 16 MTP projections and 15 lm_heads, each one a
 * submit/wake/wait pair with the gather, the scatter and the argmax running on the loop
 * thread while the team idles.  The sequence is fully decidable up front (every embedding
 * row index is either code0 or an argmax this frame produced), so the team can enter once
 * and run the entire decode.  Kernels, per-column quantiser, accumulation and argmax order
 * are the ones the dispatched path uses, so the codes are bit-identical.
 * QWEN_CP_FRAME_REGION=0 restores the per-call path. */
static void cp_region_frame_task(size_t tid, size_t nt, void *v) {
    cp_region_t *r = (cp_region_t *)v;
    qwen_tts_ctx_t *ctx = r->ctx; qwen_batch_t *bb = r->bb; qwen_tts_config_t *c = &ctx->config;
    const int BW = r->BW, ch = bb->cp_h, ed = ctx->cp_emb_dim, h = c->hidden_size;
    const int vocab = c->codebook_size, estride = (h > ed ? h : ed);
    const float eps = c->rms_norm_eps;
    float *Xt = bb->cp_Xt, *Yt = bb->cp_Yt, *embs = bb->cp_gate;
#define RSLOT(j) (r->idx ? r->idx[j] : (j))
#define RMINE(j) ((size_t)(j) % nt == tid)
    /* Phase attribution for the path the SERVER actually runs.  The whole frame lives inside
     * one held region, so cp.decode had no children at all.  Accumulate plain nanoseconds in
     * locals across the 16 steps and submit FOUR derived durations once per frame per worker:
     * no begin/end pair per step or per layer, which is what made the pool markers unaffordable.
     * Cost is 5 clock reads per step per worker -- ~0.02% of a frame. */
    const int prof = qwen_costmap_level() != 0;
    uint64_t ph_mtp = 0, ph_head = 0, tmark = 0;
    uint64_t phl[6] = { 0, 0, 0, 0, 0, 0 };   /* qkv, attn, out_proj, gate_up, down, other */
#define CPB_T0() do { if (prof) tmark = qwen_costmap_now_ns(); } while (0)
#define CPB_ACC(acc) do { if (prof) { uint64_t _n = qwen_costmap_now_ns(); (acc) += _n - tmark; \
                                      tmark = _n; } } while (0)
    for (int s = 0; s < 16; s++) {
        CPB_T0();
        /* MTP projection: this step's source row per slot, quantised, then one matmat. */
        for (int j = 0; j < BW; j++) if (RMINE(j)) {
            int b = RSLOT(j);
            const float *src;
            if (s == 0) {
                src = r->talker_hidden + (size_t)b * h;
            } else {
                float *e = embs + (size_t)j * estride;
                if (s == 1)
                    qwen_bf16_to_f32_vec(e, ctx->codec_embedding_bf16 + (int64_t)r->code0[b] * h, h);
                else
                    qwen_bf16_to_f32_vec(e, ctx->cp_codec_emb_bf16[s - 2] +
                                         (int64_t)r->out_codes[(size_t)b * 15 + (s - 2)] * ed, ed);
                src = e;
            }
            for (int k = 0; k < ed; k++) Xt[(size_t)k * BW + j] = src[k];
            r->sx[j] = qwen_region_i8_quant_col(r->qx + (size_t)j * ed, Xt, ed, BW, j);
        }
        qwen_barrier_wait(&r->bar);
        qwen_region_i8_run(Yt, ctx->cp_mtp_proj_int8, ctx->cp_mtp_proj_scale, r->qx, r->sx,
                      ch, ed, BW, tid, nt);
        qwen_barrier_wait(&r->bar);
        for (int j = 0; j < BW; j++) if (RMINE(j)) {
            int b = RSLOT(j);
            cp_region_scatter(r, r->x, Yt, b, j, ch);
            if (ctx->cp_mtp_proj_bias) {
                float *d = r->x + (size_t)b * ch;
                for (int i = 0; i < ch; i++) d[i] += ctx->cp_mtp_proj_bias[i];
            }
            qwen_rms_norm(r->x_norm + (size_t)b * ch, r->x + (size_t)b * ch,
                          ctx->cp_layers[0].input_norm, 1, ch, eps);
            cp_region_gather_quant(r, r->x_norm, b, j, ch, ch);
        }
        qwen_barrier_wait(&r->bar);
        CPB_ACC(ph_mtp);
        cp_region_layers(r, tid, nt, s, prof ? phl : NULL);
        CPB_T0();                 /* cp_region_layers accounts for its own interval in phl[] */
        if (s == 0) continue;                        /* the first step has no head */
        {
            const int g = s - 1;
            for (int j = 0; j < BW; j++) if (RMINE(j)) {
                int b = RSLOT(j);
                qwen_rms_norm(r->x_norm + (size_t)b * ch, r->x + (size_t)b * ch, ctx->cp_norm, 1, ch, eps);
                cp_region_gather_quant(r, r->x_norm, b, j, ch, ch);
            }
            qwen_barrier_wait(&r->bar);
            qwen_region_i8_run(Yt, ctx->cp_lm_head_int8[g], ctx->cp_lm_head_scale[g], r->qx, r->sx,
                          vocab, ch, BW, tid, nt);
            qwen_barrier_wait(&r->bar);
            for (int j = 0; j < BW; j++) if (RMINE(j)) {
                int b = RSLOT(j), best = 0; float bv = Yt[j];
                for (int o = 1; o < vocab; o++) {
                    float vv = Yt[(size_t)o * BW + j];
                    if (vv > bv) { bv = vv; best = o; }
                }
                r->out_codes[(size_t)b * 15 + g] = best;
            }
            CPB_ACC(ph_head);
        }
    }
    if (prof) {
        /* ONE submission per worker per frame, four derived durations. */
        qwen_region_add_ns(QWEN_RGN_CPB_MTP, ph_mtp);
        qwen_region_add_ns(QWEN_RGN_CPB_QKV,    phl[0]);
        qwen_region_add_ns(QWEN_RGN_CPB_ATTN,   phl[1]);
        qwen_region_add_ns(QWEN_RGN_CPB_WO,     phl[2]);
        qwen_region_add_ns(QWEN_RGN_CPB_GATEUP, phl[3]);
        qwen_region_add_ns(QWEN_RGN_CPB_DOWN,   phl[4]);
        qwen_region_add_ns(QWEN_RGN_CPB_LOTHER, phl[5]);
        qwen_region_add_ns(QWEN_RGN_CPB_LMHEAD, ph_head);
    }
#undef CPB_T0
#undef CPB_ACC
#undef RSLOT
#undef RMINE
}

static int cp_frame_region_enabled(void) {
    static int on = -1;
    if (on < 0) { const char *e = getenv("QWEN_CP_FRAME_REGION"); on = !(e && e[0] == '0'); }
    return on;
}

/* Returns 1 when the whole frame ran as one region (out_codes filled for every active
 * slot), 0 when any precondition fails and the caller must run its own sequence. */
static int cp_frame_region_run(qwen_tts_ctx_t *ctx, qwen_batch_t *bb, const float *talker_hidden,
                               const int *code0, int *out_codes, const uint8_t *active) {
    qwen_tts_config_t *c = &ctx->config;
    const int B = bb->B, ch = bb->cp_h, ed = ctx->cp_emb_dim, h = c->hidden_size;
    const int vocab = c->codebook_size, cint = bb->cp_inter, cqd = bb->cp_q_dim;
    const int BW = bb->B_eff > 0 ? bb->B_eff : B;
    if (!cp_frame_region_enabled() || !cp_batch_head_enabled()) return 0;
#ifdef QWEN_HAVE_CUDA
    { extern void *g_cuda_cp_batch_state; if (g_cuda_cp_batch_state) return 0; }
#endif
#ifdef QWEN_HAVE_METAL
    if (g_metal_cp_batch_state) return 0;
#endif
    if (B > 64 || BW < 2 || BW > 16) return 0;
    if (!cp_region_ok(ctx, bb, BW)) return 0;
    /* act_idx must be exactly the active set (QWEN_BATCH_NO_BEFF packs inactive slots too) */
    { int nact = 0;
      for (int b = 0; b < B; b++) if (!active || active[b]) nact++;
      if (nact != BW) return 0; }
    if (!ctx->cp_mtp_proj_bf16 || !ctx->cp_mtp_proj_int8 || ctx->cp_mtp_proj_q4) return 0;
    if (!ctx->codec_embedding_bf16) return 0;
    for (int g = 0; g < 15; g++) if (!ctx->cp_lm_head_int8[g] || ctx->cp_lm_head_q4[g]) return 0;
    for (int g = 1; g < 15; g++) if (!ctx->cp_codec_emb_bf16[g - 1]) return 0;
    if (!qwen_region_i8_usable(ch, ed, BW) || !qwen_region_i8_usable(vocab, ch, BW)) return 0;
    /* the shared batch scratch must hold the two extra operands as well */
    int cmaxcols = ch; if (cqd > cmaxcols) cmaxcols = cqd; if (cint > cmaxcols) cmaxcols = cint;
    int cmaxrows = 2 * cint; if (cqd > cmaxrows) cmaxrows = cqd; if (ch > cmaxrows) cmaxrows = ch;
    const int estride = h > ed ? h : ed;
    if (ed > cmaxcols || vocab > cmaxrows || ed > h) return 0;
    if ((long long)BW * estride > (long long)B * 2 * cint) return 0;
    for (int b = 0; b < B; b++) {
        if (active && !active[b]) continue;
        if (code0[b] < 0 || code0[b] >= c->codec_vocab_size) return 0;
    }
    size_t maxc = (size_t)cmaxcols; if ((size_t)ed > maxc) maxc = (size_t)ed;
    int8_t *qx = NULL; float *swtmp = NULL;
    if (!cp_region_scratch(BW, maxc, cint, &qx, &swtmp)) return 0;
    cp_region_t r; memset(&r, 0, sizeof r);
    r.ctx = ctx; r.bb = bb; r.x = bb->cp_x; r.x_norm = bb->cp_x_norm; r.pos = 0;
    r.BW = BW; r.idx = bb->act_idx; r.qx = qx; r.swtmp = swtmp;
    r.talker_hidden = talker_hidden; r.code0 = code0; r.out_codes = out_codes;
    int team = qwen_parallel_team();
    static int announced = 0;
    if (!announced) {
        announced = 1;
        fprintf(stderr, "[cp] whole decode frame as one parallel region: ON (team %d, BW %d)\n", team, BW);
    }
    qwen_barrier_init(&r.bar, team);
    qwen_parallel((size_t)team, cp_region_frame_task, &r);
    return 1;
}

int qwen_batch_cp_predict(qwen_tts_ctx_t *ctx, qwen_batch_t *bb,
                          const float *talker_hidden, const int *code0, int *out_codes,
                          const uint8_t *active) {
    qwen_tts_config_t *c = &ctx->config;
    int B = bb->B, ch = bb->cp_h, h = c->hidden_size, emb_dim = ctx->cp_emb_dim;
    if (!ctx->cp_layers[0].wq_bf16 || !ctx->cp_lm_head_bf16[0]) return -2;

    /* The single-request path tags its kernels QWEN_COMP_CP; the batched one did
     * not, so every batched CP projection was attributed to whichever component
     * ran last (the Talker).  Tag here and restore on exit so the CP tag does not
     * leak into the caller's own work either. */
    const int prev_comp = qwen_mm_component_get();
    qwen_mm_component(QWEN_COMP_CP);

    if (!qwen_batch_solo_disabled() && active) {
        int n_act = 0, only = -1;
        for (int b = 0; b < B; b++) if (active[b]) { n_act++; only = b; }
        if (n_act == 1) {
            size_t slot = (size_t)only * bb->cp_num_layers * bb->cp_kv_max * bb->cp_kv_dim;
            uint16_t *sk = ctx->cp_kv_k, *sv = ctx->cp_kv_v;
            int smax = ctx->cp_kv_max, slen = ctx->cp_kv_len;
            ctx->cp_kv_k = bb->cp_kv_k + slot;
            ctx->cp_kv_v = bb->cp_kv_v + slot;
            ctx->cp_kv_max = bb->cp_kv_max;
            ctx->cp_kv_len = 0;
            memset(out_codes, 0, (size_t)B * 15 * sizeof(int));
            int rc = qwen_cp_predict(ctx,
                                     (float *)(uintptr_t)talker_hidden + (size_t)only * h,
                                     code0[only], out_codes + (size_t)only * 15);
            ctx->cp_kv_k = sk; ctx->cp_kv_v = sv;
            ctx->cp_kv_max = smax; ctx->cp_kv_len = slen;
            qwen_batch_pack_active(bb, active);
            qwen_mm_component(prev_comp);
            return rc;
        }
    }

    qwen_region_begin(QWEN_RGN_CP_DECODE);
    qwen_batch_pack_active(bb, active);

    float *cx = bb->cp_x, *cxn = bb->cp_x_norm;

#define CPB_SKIP(b) (active && !active[b])

    if (cp_frame_region_run(ctx, bb, talker_hidden, code0, out_codes, active)) {
        for (int b = 0; b < B; b++) {
            if (!CPB_SKIP(b)) continue;
            memset(cx + (size_t)b * ch, 0, (size_t)ch * sizeof(float));
            for (int g = 0; g < 15; g++) out_codes[(size_t)b * 15 + g] = 0;
        }
        qwen_mm_component(prev_comp);
        qwen_region_end(QWEN_RGN_CP_DECODE);
        return 0;
    }

    {
        const float *srcs[64]; int any = 0;
        for (int b = 0; b < B && b < 64; b++) { srcs[b] = talker_hidden + (size_t)b * h; if (!CPB_SKIP(b)) any = 1; }
        int done = (any && B <= 64) ? cp_batch_mtp(ctx, bb, cx, srcs, active) : 0;
        for (int b = 0; b < B; b++) {
            if (CPB_SKIP(b)) { memset(cx + (size_t)b * ch, 0, ch * sizeof(float)); continue; }
            if (!done) cp_mtp_project(ctx, cx + (size_t)b * ch, talker_hidden + (size_t)b * h);
        }
    }
    batch_cp_transformer_step(ctx, bb, cx, cxn, 0, active);

    {
        /* embeddings of all active slots first (bb->cp_Yt is free here), then one projection */
        float *embs = bb->cp_gate;   /* B x 2*cint floats, unused between steps: room for B x h */
        uint8_t ok[64]; const float *srcs[64]; int nok = 0;
        for (int b = 0; b < B && b < 64; b++) {
            ok[b] = 0; srcs[b] = embs + (size_t)b * h;
            if (CPB_SKIP(b)) continue;
            int code0_b = code0[b];
            if (ctx->codec_embedding_bf16 && code0_b >= 0 && code0_b < c->codec_vocab_size) {
                qwen_bf16_to_f32_vec(embs + (size_t)b * h, ctx->codec_embedding_bf16 + (int64_t)code0_b * h, h);
                ok[b] = 1; nok++;
            }
        }
        int done = 0;
        if (nok >= 2 && B <= 64) {
            uint8_t act2[64]; for (int b = 0; b < B; b++) act2[b] = ok[b];
            done = cp_batch_mtp(ctx, bb, cx, srcs, act2);
        }
        for (int b = 0; b < B; b++) {
            if (CPB_SKIP(b)) continue;
            if (!ok[b]) { memset(cx + (size_t)b * ch, 0, ch * sizeof(float)); continue; }
            if (!done) cp_mtp_project(ctx, cx + (size_t)b * ch, srcs[b]);
        }
    }
    batch_cp_transformer_step(ctx, bb, cx, cxn, 1, active);

    {
        qwen_region_begin(QWEN_RGN_CP_D_LMHEAD);
#ifdef QWEN_HAVE_CUDA
        /* CUDA-only: the head runs on the device when the batched step just ran there.
         * Deliberately duplicated rather than factored with the block below -- the CPU code in
         * this function is left exactly as it was, and everything CUDA lives inside the ifdef. */
        {
            extern void *g_cuda_cp_batch_state;
            extern int qwen_cuda_cp_batch_head(void *, int, const unsigned char *, int *, int);
            if (g_cuda_cp_batch_state && B <= 16 &&
                qwen_cuda_cp_batch_head(g_cuda_cp_batch_state, 0, active, out_codes, 15)) {
                for (int b = 0; b < B; b++)
                    if (CPB_SKIP(b)) out_codes[(size_t)b * 15 + 0] = 0;
                qwen_region_end(QWEN_RGN_CP_D_LMHEAD);
                goto cp_head_done_g0;
            }
        }
#endif
        for (int b = 0; b < B; b++) {
            if (CPB_SKIP(b)) { out_codes[(size_t)b * 15 + 0] = 0; continue; }
            qwen_rms_norm(cxn + (size_t)b * ch, cx + (size_t)b * ch, ctx->cp_norm, 1, ch, c->rms_norm_eps);
        }
        if (!cp_batch_lm(ctx, bb, cxn, 0, out_codes, active))
            for (int b = 0; b < B; b++) {
                if (CPB_SKIP(b)) continue;
                out_codes[(size_t)b * 15 + 0] = cp_lm_argmax(ctx, cxn + (size_t)b * ch, 0, ch, c->codebook_size);
            }
        qwen_region_end(QWEN_RGN_CP_D_LMHEAD);
#ifdef QWEN_HAVE_CUDA
        cp_head_done_g0: ;
#endif
    }

    double _cpb_g0 = cpb_on() ? cpb_now() : 0.0;   /* start of pass g's seed section */
    for (int g = 1; g < 15; g++) {
        int pos = g + 1;
        {
            float *embs = bb->cp_gate;
            uint8_t ok[64]; const float *srcs[64]; int nok = 0;
            for (int b = 0; b < B && b < 64; b++) {
                ok[b] = 0; srcs[b] = embs + (size_t)b * emb_dim;
                if (CPB_SKIP(b)) continue;
                int prev = out_codes[(size_t)b * 15 + (g - 1)];
                if (ctx->cp_codec_emb_bf16[g - 1] && prev >= 0 && prev < c->codebook_size) {
                    qwen_bf16_to_f32_vec(embs + (size_t)b * emb_dim, ctx->cp_codec_emb_bf16[g - 1] + (int64_t)prev * emb_dim, emb_dim);
                    ok[b] = 1; nok++;
                }
            }
            int done = 0;
            if (nok >= 2 && B <= 64) {
                uint8_t act2[64]; for (int b = 0; b < B; b++) act2[b] = ok[b];
                done = cp_batch_mtp(ctx, bb, cx, srcs, act2);
            }
            for (int b = 0; b < B; b++) {
                if (CPB_SKIP(b)) continue;
                if (!ok[b]) { memset(cx + (size_t)b * ch, 0, ch * sizeof(float)); continue; }
                if (!done) cp_mtp_project(ctx, cx + (size_t)b * ch, srcs[b]);
            }
        }
        const int _cpb = cpb_on();
        double _cpb_m = _cpb ? cpb_now() : 0.0;
        if (_cpb) { cpb_seed += _cpb_m - _cpb_g0; }
        batch_cp_transformer_step(ctx, bb, cx, cxn, pos, active);
        if (_cpb) { double _t = cpb_now(); cpb_step += _t - _cpb_m; _cpb_m = _t; }
        {
            qwen_region_begin(QWEN_RGN_CP_D_LMHEAD);
#ifdef QWEN_HAVE_CUDA
        /* CUDA-only: the head runs on the device when the batched step just ran there.
         * Deliberately duplicated rather than factored with the block below -- the CPU code in
         * this function is left exactly as it was, and everything CUDA lives inside the ifdef. */
        {
            extern void *g_cuda_cp_batch_state;
            extern int qwen_cuda_cp_batch_head(void *, int, const unsigned char *, int *, int);
            if (g_cuda_cp_batch_state && B <= 16 &&
                qwen_cuda_cp_batch_head(g_cuda_cp_batch_state, g, active, out_codes, 15)) {
                for (int b = 0; b < B; b++)
                    if (CPB_SKIP(b)) out_codes[(size_t)b * 15 + g] = 0;
                qwen_region_end(QWEN_RGN_CP_D_LMHEAD);
                goto cp_head_done_gl;
            }
        }
#endif
            for (int b = 0; b < B; b++) {
                if (CPB_SKIP(b)) { out_codes[(size_t)b * 15 + g] = 0; continue; }
                qwen_rms_norm(cxn + (size_t)b * ch, cx + (size_t)b * ch, ctx->cp_norm, 1, ch, c->rms_norm_eps);
            }
            if (!cp_batch_lm(ctx, bb, cxn, g, out_codes, active))
                for (int b = 0; b < B; b++) {
                    if (CPB_SKIP(b)) continue;
                    out_codes[(size_t)b * 15 + g] = cp_lm_argmax(ctx, cxn + (size_t)b * ch, g, ch, c->codebook_size);
                }
            qwen_region_end(QWEN_RGN_CP_D_LMHEAD);
#ifdef QWEN_HAVE_CUDA
            cp_head_done_gl: ;
#endif
        }
        if (_cpb) { cpb_head += cpb_now() - _cpb_m; _cpb_g0 = cpb_now(); }
    }
    if (cpb_on()) { cpb_frames++; if ((cpb_frames % 500) == 0) cpb_report(); }
    qwen_mm_component(prev_comp);
    qwen_region_end(QWEN_RGN_CP_DECODE);
    return 0;
#undef CPB_SKIP
}
