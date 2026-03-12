/*
 * qwen_tts_speech_decoder.c - Speech Decoder (ConvNet) forward pass
 * Converts 16 codebook codes per frame → 24kHz audio waveform
 *
 * Architecture:
 * 1. VQ dequant (16 codebooks × 2048 × 256) → sum → project to 512
 * 2. Pre-conv (512→1024, k=3, causal)
 * 3. Pre-transformer (8 layers, hidden=512, sliding window=72, layer_scale)
 * 4. Output proj (512→1024)
 * 5. ConvNeXt upsample (2 blocks, 2x each)
 * 6. Initial conv (1024→1536, k=7)
 * 7. 4 Decoder upsample blocks (rates: 8,5,4,3) with 3 residual blocks each
 * 8. Final snake + conv (96→1, k=7) → audio
 *
 * Tensor naming from safetensors:
 *   decoder.upsample.{0,1}.0.conv.{weight,bias}        - ConvNeXt ConvTranspose
 *   decoder.upsample.{0,1}.1.{dwconv.conv,norm,...}     - ConvNeXt block
 *   decoder.decoder.0.conv.{weight,bias}                - initial conv
 *   decoder.decoder.{1-4}.block.0.{alpha,beta}          - snake before upsample
 *   decoder.decoder.{1-4}.block.1.conv.{weight,bias}    - ConvTranspose upsample
 *   decoder.decoder.{1-4}.block.{2-4}.{act1,conv1,act2,conv2} - ResBlocks
 *   decoder.decoder.5.{alpha,beta}                      - final snake
 *   decoder.decoder.6.conv.{weight,bias}                - final conv
 */

#include "qwen_tts.h"
#include "qwen_tts_kernels.h"
#include "qwen_tts_safetensors.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Cache-line aligned allocation for BLAS/SIMD performance.
 * 64-byte alignment matches Apple M1/M2 cache line size.
 * Falls back to malloc if posix_memalign unavailable. */
static inline void *aligned_malloc(size_t size) {
    void *ptr = NULL;
    if (posix_memalign(&ptr, 64, size) != 0) return NULL;
    return ptr;
}
static inline void *aligned_calloc(size_t count, size_t size) {
    size_t total = count * size;
    void *ptr = aligned_malloc(total);
    if (ptr) memset(ptr, 0, total);
    return ptr;
}

#ifdef USE_BLAS
#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif
#define CONV_TILE_MAX_BYTES (256 * 1024 * 1024)
#endif

static const float *get_f32(void *ms, const char *name) {
    safetensors_file_t *sf = NULL;
    const safetensor_t *t = multi_safetensors_find((multi_safetensors_t *)ms, name, &sf);
    return (t && sf) ? (const float *)safetensors_data(sf, t) : NULL;
}

/* Causal Conv1d: out_len = (in_len + pad_left - kernel) / stride + 1 */
static int conv1d_out_len(int in_len, int kernel, int stride, int pad_left) {
    return (in_len + pad_left - kernel) / stride + 1;
}

/* Causal ConvTranspose1d: out_len = (in_len-1)*stride + kernel - (kernel-stride) = in_len*stride */
static int conv_transpose1d_out_len(int in_len, int kernel, int stride) {
    return (in_len - 1) * stride + kernel - (kernel - stride);
}

/* Snake activation dispatched through qwen_snake_activation() kernel
 * (NEON/Accelerate-optimized in qwen_tts_kernels.c) */
#define snake_activation qwen_snake_activation

#ifndef USE_BLAS
/* Naive causal Conv1d: [out_ch, in_ch, kernel], pad_left=(kernel-1)*dilation */
static void causal_conv1d_naive(float *out, const float *in,
                                const float *weight, const float *bias,
                                int in_ch, int out_ch, int length,
                                int kernel, int dilation) {
    int pad_left = (kernel - 1) * dilation;
    for (int oc = 0; oc < out_ch; oc++) {
        float b = bias ? bias[oc] : 0;
        for (int t = 0; t < length; t++) {
            float sum = b;
            for (int ic = 0; ic < in_ch; ic++) {
                for (int k = 0; k < kernel; k++) {
                    int in_pos = t - pad_left + k * dilation;
                    if (in_pos >= 0 && in_pos < length) {
                        sum += weight[((int64_t)oc * in_ch + ic) * kernel + k]
                             * in[(int64_t)ic * length + in_pos];
                    }
                }
            }
            out[(int64_t)oc * length + t] = sum;
        }
    }
}

/* Naive causal ConvTranspose1d: [in_ch, out_ch, kernel], stride, trim right by (kernel-stride) */
static void causal_conv_transpose1d_naive(float *out, const float *in,
                                          const float *weight, const float *bias,
                                          int in_ch, int out_ch, int in_len, int out_len,
                                          int kernel, int stride) {
    memset(out, 0, (int64_t)out_ch * out_len * sizeof(float));
    int full_len = (in_len - 1) * stride + kernel;
    int trim_right = kernel - stride;

    for (int ic = 0; ic < in_ch; ic++) {
        for (int t = 0; t < in_len; t++) {
            float x = in[(int64_t)ic * in_len + t];
            for (int k = 0; k < kernel; k++) {
                int out_pos = t * stride + k;
                if (out_pos < full_len - trim_right && out_pos < out_len) {
                    for (int oc = 0; oc < out_ch; oc++) {
                        out[(int64_t)oc * out_len + out_pos] +=
                            x * weight[((int64_t)ic * out_ch + oc) * kernel + k];
                    }
                }
            }
        }
    }
    if (bias) {
        for (int oc = 0; oc < out_ch; oc++)
            for (int t = 0; t < out_len; t++)
                out[(int64_t)oc * out_len + t] += bias[oc];
    }
}
#endif /* !USE_BLAS */

#ifdef USE_BLAS
/* Add bias to channel-first output [channels, length] */
static void conv_add_bias(float *out, const float *bias, int channels, int length) {
    if (!bias) return;
    for (int c = 0; c < channels; c++) {
        float b = bias[c];
        float *row = out + (int64_t)c * length;
        for (int t = 0; t < length; t++)
            row[t] += b;
    }
}

/* BLAS causal Conv1d: im2col + sgemm (k>1), direct sgemm (k=1) */
static void causal_conv1d_blas(float *out, const float *in,
                               const float *weight, const float *bias,
                               int in_ch, int out_ch, int length,
                               int kernel, int dilation) {
    if (kernel == 1) {
        /* k=1: weight is [out_ch, in_ch], direct matmul */
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    out_ch, length, in_ch,
                    1.0f, weight, in_ch,
                    in, length,
                    0.0f, out, length);
        conv_add_bias(out, bias, out_ch, length);
        return;
    }

    /* im2col + sgemm for k>1 */
    int pad_left = (kernel - 1) * dilation;
    int64_t col_rows = (int64_t)in_ch * kernel;

    /* Tile along time if im2col buffer would exceed limit */
    int64_t max_tile = CONV_TILE_MAX_BYTES / (col_rows * (int64_t)sizeof(float));
    if (max_tile < 1) max_tile = 1;
    if (max_tile > length) max_tile = length;

    float *col = (float *)aligned_malloc(col_rows * max_tile * sizeof(float));

    for (int ts = 0; ts < length; ts += (int)max_tile) {
        int tile = ((int64_t)ts + max_tile > length) ? length - ts : (int)max_tile;

        /* Build im2col: col[in_ch*kernel, tile] */
        memset(col, 0, col_rows * tile * sizeof(float));
        for (int ic = 0; ic < in_ch; ic++) {
            for (int k = 0; k < kernel; k++) {
                float *col_row = col + ((int64_t)ic * kernel + k) * tile;
                for (int t = 0; t < tile; t++) {
                    int in_pos = (t + ts) - pad_left + k * dilation;
                    if (in_pos >= 0 && in_pos < length)
                        col_row[t] = in[(int64_t)ic * length + in_pos];
                }
            }
        }

        /* sgemm: out_tile = weight[out_ch, col_rows] × col[col_rows, tile] */
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    out_ch, tile, (int)col_rows,
                    1.0f, weight, (int)col_rows,
                    col, tile,
                    0.0f, out + ts, length);
    }

    free(col);
    conv_add_bias(out, bias, out_ch, length);
}

/* BLAS causal ConvTranspose1d: per-kernel sgemm + scatter */
static void causal_conv_transpose1d_blas(float *out, const float *in,
                                         const float *weight, const float *bias,
                                         int in_ch, int out_ch, int in_len, int out_len,
                                         int kernel, int stride) {
    memset(out, 0, (int64_t)out_ch * out_len * sizeof(float));
    int full_len = (in_len - 1) * stride + kernel;
    int trim_right = kernel - stride;

    /* Per-kernel-position: extract weight slice, sgemm, scatter */
    float *wk = (float *)aligned_malloc((int64_t)in_ch * out_ch * sizeof(float));
    float *rk = (float *)aligned_malloc((int64_t)out_ch * in_len * sizeof(float));

    for (int k = 0; k < kernel; k++) {
        /* Extract W_k[in_ch, out_ch] from weight[in_ch, out_ch, kernel] */
        for (int ic = 0; ic < in_ch; ic++)
            for (int oc = 0; oc < out_ch; oc++)
                wk[(int64_t)ic * out_ch + oc] =
                    weight[((int64_t)ic * out_ch + oc) * kernel + k];

        /* rk[out_ch, in_len] = W_k^T[out_ch, in_ch] × in[in_ch, in_len] */
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    out_ch, in_len, in_ch,
                    1.0f, wk, out_ch,
                    in, in_len,
                    0.0f, rk, in_len);

        /* Scatter to strided output positions */
        for (int oc = 0; oc < out_ch; oc++) {
            const float *src = rk + (int64_t)oc * in_len;
            float *dst = out + (int64_t)oc * out_len;
            for (int t = 0; t < in_len; t++) {
                int out_pos = t * stride + k;
                if (out_pos < full_len - trim_right && out_pos < out_len)
                    dst[out_pos] += src[t];
            }
        }
    }

    free(wk);
    free(rk);
    conv_add_bias(out, bias, out_ch, out_len);
}
#endif /* USE_BLAS */

/* Dispatch wrappers */
static void causal_conv1d(float *out, const float *in,
                           const float *weight, const float *bias,
                           int in_ch, int out_ch, int length,
                           int kernel, int dilation) {
#ifdef USE_BLAS
    causal_conv1d_blas(out, in, weight, bias, in_ch, out_ch, length, kernel, dilation);
#else
    causal_conv1d_naive(out, in, weight, bias, in_ch, out_ch, length, kernel, dilation);
#endif
}

static void causal_conv_transpose1d(float *out, const float *in,
                                     const float *weight, const float *bias,
                                     int in_ch, int out_ch, int in_len, int out_len,
                                     int kernel, int stride) {
#ifdef USE_BLAS
    causal_conv_transpose1d_blas(out, in, weight, bias, in_ch, out_ch, in_len, out_len, kernel, stride);
#else
    causal_conv_transpose1d_naive(out, in, weight, bias, in_ch, out_ch, in_len, out_len, kernel, stride);
#endif
}

/* ========================================================================
 * Weight Loading
 * ======================================================================== */

int qwen_speech_decoder_load(qwen_tts_ctx_t *ctx) {
    qwen_tts_config_t *c = &ctx->config;
    void *ms = ctx->speech_safetensors;
    qwen_speech_decoder_t *sd = &ctx->speech_dec;

    if (!ctx->silent) fprintf(stderr, "Loading Speech Decoder weights...\n");

    int cb_dim = QWEN_TTS_CODEBOOK_DIM;
    int cb_size = c->codebook_size;

    /* Codebook 0 (rvq_first) - dequantize from EMA */
    const float *emb_sum = get_f32(ms, "decoder.quantizer.rvq_first.vq.layers.0._codebook.embedding_sum");
    const float *usage = get_f32(ms, "decoder.quantizer.rvq_first.vq.layers.0._codebook.cluster_usage");
    if (emb_sum && usage) {
        sd->codebook[0] = (float *)malloc((int64_t)cb_size * cb_dim * sizeof(float));
        for (int i = 0; i < cb_size; i++)
            for (int d = 0; d < cb_dim; d++)
                sd->codebook[0][(int64_t)i * cb_dim + d] = emb_sum[(int64_t)i * cb_dim + d] / fmaxf(usage[i], 1e-5f);
    }

    /* Codebooks 1-15 (rvq_rest) */
    for (int k = 0; k < 15; k++) {
        char es_name[128], cu_name[128];
        snprintf(es_name, sizeof(es_name), "decoder.quantizer.rvq_rest.vq.layers.%d._codebook.embedding_sum", k);
        snprintf(cu_name, sizeof(cu_name), "decoder.quantizer.rvq_rest.vq.layers.%d._codebook.cluster_usage", k);
        emb_sum = get_f32(ms, es_name);
        usage = get_f32(ms, cu_name);
        if (emb_sum && usage) {
            sd->codebook[k + 1] = (float *)malloc((int64_t)cb_size * cb_dim * sizeof(float));
            for (int i = 0; i < cb_size; i++)
                for (int d = 0; d < cb_dim; d++)
                    sd->codebook[k + 1][(int64_t)i * cb_dim + d] = emb_sum[(int64_t)i * cb_dim + d] / fmaxf(usage[i], 1e-5f);
        }
    }

    /* VQ projections */
    sd->rvq_first_output_proj = get_f32(ms, "decoder.quantizer.rvq_first.output_proj.weight");
    sd->rvq_rest_output_proj = get_f32(ms, "decoder.quantizer.rvq_rest.output_proj.weight");

    /* Pre-conv */
    sd->pre_conv_weight = get_f32(ms, "decoder.pre_conv.conv.weight");
    sd->pre_conv_bias = get_f32(ms, "decoder.pre_conv.conv.bias");

    /* Pre-transformer */
    sd->input_proj_weight = get_f32(ms, "decoder.pre_transformer.input_proj.weight");
    sd->input_proj_bias = get_f32(ms, "decoder.pre_transformer.input_proj.bias");
    sd->final_norm_weight = get_f32(ms, "decoder.pre_transformer.norm.weight");
    sd->output_proj_weight = get_f32(ms, "decoder.pre_transformer.output_proj.weight");
    sd->output_proj_bias = get_f32(ms, "decoder.pre_transformer.output_proj.bias");

    sd->pre_layers = (qwen_sd_pre_layer_t *)calloc(c->dec_num_layers, sizeof(qwen_sd_pre_layer_t));
    for (int i = 0; i < c->dec_num_layers; i++) {
        qwen_sd_pre_layer_t *l = &sd->pre_layers[i];
        char name[128];
        snprintf(name, sizeof(name), "decoder.pre_transformer.layers.%d.input_layernorm.weight", i);
        l->attn_norm = get_f32(ms, name);
        snprintf(name, sizeof(name), "decoder.pre_transformer.layers.%d.self_attn.q_proj.weight", i);
        l->attn_q = get_f32(ms, name);
        snprintf(name, sizeof(name), "decoder.pre_transformer.layers.%d.self_attn.k_proj.weight", i);
        l->attn_k = get_f32(ms, name);
        snprintf(name, sizeof(name), "decoder.pre_transformer.layers.%d.self_attn.v_proj.weight", i);
        l->attn_v = get_f32(ms, name);
        snprintf(name, sizeof(name), "decoder.pre_transformer.layers.%d.self_attn.o_proj.weight", i);
        l->attn_o = get_f32(ms, name);
        snprintf(name, sizeof(name), "decoder.pre_transformer.layers.%d.self_attn_layer_scale.scale", i);
        l->attn_layer_scale = get_f32(ms, name);
        snprintf(name, sizeof(name), "decoder.pre_transformer.layers.%d.post_attention_layernorm.weight", i);
        l->ffn_norm = get_f32(ms, name);
        snprintf(name, sizeof(name), "decoder.pre_transformer.layers.%d.mlp.gate_proj.weight", i);
        l->ffn_gate = get_f32(ms, name);
        snprintf(name, sizeof(name), "decoder.pre_transformer.layers.%d.mlp.up_proj.weight", i);
        l->ffn_up = get_f32(ms, name);
        snprintf(name, sizeof(name), "decoder.pre_transformer.layers.%d.mlp.down_proj.weight", i);
        l->ffn_down = get_f32(ms, name);
        snprintf(name, sizeof(name), "decoder.pre_transformer.layers.%d.mlp_layer_scale.scale", i);
        l->ffn_layer_scale = get_f32(ms, name);
    }

    /* RoPE cache for pre-transformer (NeoX split-half) */
    int half_dim = c->dec_head_dim / 2;
    sd->rope_cos = (float *)malloc(8000 * half_dim * sizeof(float));
    sd->rope_sin = (float *)malloc(8000 * half_dim * sizeof(float));
    for (int pos = 0; pos < 8000; pos++) {
        for (int i = 0; i < half_dim; i++) {
            float angle = pos / powf(c->dec_rope_theta, (float)(2*i) / c->dec_head_dim);
            sd->rope_cos[pos * half_dim + i] = cosf(angle);
            sd->rope_sin[pos * half_dim + i] = sinf(angle);
        }
    }

    /* ConvNeXt upsample blocks (2 blocks) */
    for (int b = 0; b < 2; b++) {
        qwen_sd_convnext_t *cn = &sd->convnext[b];
        char name[128];
        /* ConvTranspose1d is sub-layer 0 */
        snprintf(name, sizeof(name), "decoder.upsample.%d.0.conv.weight", b);
        cn->conv_weight = get_f32(ms, name);
        snprintf(name, sizeof(name), "decoder.upsample.%d.0.conv.bias", b);
        cn->conv_bias = get_f32(ms, name);
        /* ConvNeXt block is sub-layer 1 */
        snprintf(name, sizeof(name), "decoder.upsample.%d.1.dwconv.conv.weight", b);
        cn->dwconv_weight = get_f32(ms, name);
        snprintf(name, sizeof(name), "decoder.upsample.%d.1.dwconv.conv.bias", b);
        cn->dwconv_bias = get_f32(ms, name);
        snprintf(name, sizeof(name), "decoder.upsample.%d.1.pwconv1.weight", b);
        cn->pwconv1_weight = get_f32(ms, name);
        snprintf(name, sizeof(name), "decoder.upsample.%d.1.pwconv1.bias", b);
        cn->pwconv1_bias = get_f32(ms, name);
        snprintf(name, sizeof(name), "decoder.upsample.%d.1.pwconv2.weight", b);
        cn->pwconv2_weight = get_f32(ms, name);
        snprintf(name, sizeof(name), "decoder.upsample.%d.1.pwconv2.bias", b);
        cn->pwconv2_bias = get_f32(ms, name);
        snprintf(name, sizeof(name), "decoder.upsample.%d.1.norm.weight", b);
        cn->norm_weight = get_f32(ms, name);
        snprintf(name, sizeof(name), "decoder.upsample.%d.1.norm.bias", b);
        cn->norm_bias = get_f32(ms, name);
        snprintf(name, sizeof(name), "decoder.upsample.%d.1.gamma", b);
        cn->gamma = get_f32(ms, name);
    }

    /* Initial conv: decoder.decoder.0 */
    sd->initial_conv_weight = get_f32(ms, "decoder.decoder.0.conv.weight");
    sd->initial_conv_bias = get_f32(ms, "decoder.decoder.0.conv.bias");

    /* Decoder upsample blocks: decoder.decoder.{1-4} */
    for (int b = 0; b < 4; b++) {
        qwen_sd_upsample_block_t *ub = &sd->upsample_blocks[b];
        int bi = b + 1; /* tensor index: 1-4 */
        char name[128];

        /* Snake before upsample: block.0.{alpha,beta} */
        snprintf(name, sizeof(name), "decoder.decoder.%d.block.0.alpha", bi);
        ub->upsample.snake_alpha = get_f32(ms, name);
        snprintf(name, sizeof(name), "decoder.decoder.%d.block.0.beta", bi);
        ub->upsample.snake_beta = get_f32(ms, name);

        /* ConvTranspose upsample: block.1.conv.{weight,bias} */
        snprintf(name, sizeof(name), "decoder.decoder.%d.block.1.conv.weight", bi);
        ub->upsample.conv_weight = get_f32(ms, name);
        snprintf(name, sizeof(name), "decoder.decoder.%d.block.1.conv.bias", bi);
        ub->upsample.conv_bias = get_f32(ms, name);

        /* 3 residual blocks: block.{2,3,4} */
        for (int r = 0; r < 3; r++) {
            int ri = r + 2; /* tensor index: 2,3,4 */
            snprintf(name, sizeof(name), "decoder.decoder.%d.block.%d.act1.alpha", bi, ri);
            ub->res_blocks[r].snake1_alpha = get_f32(ms, name);
            snprintf(name, sizeof(name), "decoder.decoder.%d.block.%d.act1.beta", bi, ri);
            ub->res_blocks[r].snake1_beta = get_f32(ms, name);
            snprintf(name, sizeof(name), "decoder.decoder.%d.block.%d.conv1.conv.weight", bi, ri);
            ub->res_blocks[r].conv1_weight = get_f32(ms, name);
            snprintf(name, sizeof(name), "decoder.decoder.%d.block.%d.conv1.conv.bias", bi, ri);
            ub->res_blocks[r].conv1_bias = get_f32(ms, name);
            snprintf(name, sizeof(name), "decoder.decoder.%d.block.%d.act2.alpha", bi, ri);
            ub->res_blocks[r].snake2_alpha = get_f32(ms, name);
            snprintf(name, sizeof(name), "decoder.decoder.%d.block.%d.act2.beta", bi, ri);
            ub->res_blocks[r].snake2_beta = get_f32(ms, name);
            snprintf(name, sizeof(name), "decoder.decoder.%d.block.%d.conv2.conv.weight", bi, ri);
            ub->res_blocks[r].conv2_weight = get_f32(ms, name);
            snprintf(name, sizeof(name), "decoder.decoder.%d.block.%d.conv2.conv.bias", bi, ri);
            ub->res_blocks[r].conv2_bias = get_f32(ms, name);
        }
    }

    /* Final snake: decoder.decoder.5 */
    sd->final_snake.alpha = get_f32(ms, "decoder.decoder.5.alpha");
    sd->final_snake.beta = get_f32(ms, "decoder.decoder.5.beta");

    /* Final conv: decoder.decoder.6 */
    sd->final_conv_weight = get_f32(ms, "decoder.decoder.6.conv.weight");
    sd->final_conv_bias = get_f32(ms, "decoder.decoder.6.conv.bias");

    /* Debug: verify pre_conv weights right after loading */
    if (sd->pre_conv_weight) {
        fprintf(stderr, "  [LOAD] pre_conv_w[:5]: [%.6f, %.6f, %.6f, %.6f, %.6f] bias[:3]: [%.6f, %.6f, %.6f]\n",
                sd->pre_conv_weight[0], sd->pre_conv_weight[1], sd->pre_conv_weight[2],
                sd->pre_conv_weight[3], sd->pre_conv_weight[4],
                sd->pre_conv_bias[0], sd->pre_conv_bias[1], sd->pre_conv_bias[2]);
    } else {
        fprintf(stderr, "  [LOAD] pre_conv_weight is NULL!\n");
    }

    if (!ctx->silent) {
        fprintf(stderr, "  Codebooks: 16/16 (dequantized from EMA)\n");
        fprintf(stderr, "  Pre-transformer: %d layers, input_proj=%s\n",
                c->dec_num_layers, sd->input_proj_weight ? "ok" : "MISSING");
        fprintf(stderr, "  ConvNeXt upsample: %s\n",
                (sd->convnext[0].conv_weight && sd->convnext[1].conv_weight) ? "ok" : "MISSING");
        fprintf(stderr, "  Conv decoder: initial=%s, final=%s\n",
                sd->initial_conv_weight ? "ok" : "MISSING",
                sd->final_conv_weight ? "ok" : "MISSING");
        fprintf(stderr, "  Upsample blocks: [%s %s %s %s]\n",
                sd->upsample_blocks[0].upsample.conv_weight ? "ok" : "MISSING",
                sd->upsample_blocks[1].upsample.conv_weight ? "ok" : "MISSING",
                sd->upsample_blocks[2].upsample.conv_weight ? "ok" : "MISSING",
                sd->upsample_blocks[3].upsample.conv_weight ? "ok" : "MISSING");
        fprintf(stderr, "  Final snake: %s\n",
                sd->final_snake.alpha ? "ok" : "MISSING");
    }

    return 0;
}

/* ========================================================================
 * Decode: codes → audio
 * ======================================================================== */

int qwen_speech_decoder_decode(qwen_tts_ctx_t *ctx, const int *codes, int n_frames,
                                float **audio_out, int *n_samples) {
    qwen_speech_decoder_t *sd = &ctx->speech_dec;
    qwen_tts_config_t *c = &ctx->config;

    int cb_dim = QWEN_TTS_CODEBOOK_DIM;
    int vq_hidden = 512;
    int latent_dim = 1024;

    if (!ctx->silent)
        fprintf(stderr, "  Speech decoder: %d frames -> audio...\n", n_frames);

    /* Debug: check if weights are still intact */
    if (ctx->debug && sd->pre_conv_weight) {
        fprintf(stderr, "[DECODER] ENTRY pre_conv_w[:5]: [%.6f, %.6f, %.6f, %.6f, %.6f]\n",
                sd->pre_conv_weight[0], sd->pre_conv_weight[1], sd->pre_conv_weight[2],
                sd->pre_conv_weight[3], sd->pre_conv_weight[4]);
    }

    /* Step 1: VQ dequant + output projection (batched with BLAS) */
    float *vq_out = (float *)aligned_calloc((int64_t)n_frames * vq_hidden, sizeof(float));

    /* Gather codebook embeddings into matrices for batched projection */
    float *emb_first = (float *)aligned_malloc((int64_t)n_frames * cb_dim * sizeof(float));
    float *emb_rest = (float *)aligned_calloc((int64_t)n_frames * cb_dim, sizeof(float));

    for (int f = 0; f < n_frames; f++) {
        /* Codebook 0 (rvq_first) */
        int code0 = codes[f * 16];
        if (code0 >= 0 && code0 < c->codebook_size && sd->codebook[0]) {
            memcpy(emb_first + (int64_t)f * cb_dim,
                   sd->codebook[0] + (int64_t)code0 * cb_dim, cb_dim * sizeof(float));
        } else {
            memset(emb_first + (int64_t)f * cb_dim, 0, cb_dim * sizeof(float));
        }

        /* Codebooks 1-15 (rvq_rest): sum embeddings */
        float *rest_row = emb_rest + (int64_t)f * cb_dim;
        for (int k = 1; k < 16; k++) {
            int code = codes[f * 16 + k];
            if (code >= 0 && code < c->codebook_size && sd->codebook[k]) {
                const float *emb = sd->codebook[k] + (int64_t)code * cb_dim;
                for (int d = 0; d < cb_dim; d++) rest_row[d] += emb[d];
            }
        }
    }

    /* Batched projection: vq_out[n_frames, vq_hidden] = emb[n_frames, cb_dim] × W^T[cb_dim, vq_hidden] */
#ifdef USE_BLAS
    if (sd->rvq_first_output_proj) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    n_frames, vq_hidden, cb_dim, 1.0f,
                    emb_first, cb_dim, sd->rvq_first_output_proj, cb_dim,
                    0.0f, vq_out, vq_hidden);
    }
    if (sd->rvq_rest_output_proj) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    n_frames, vq_hidden, cb_dim, 1.0f,
                    emb_rest, cb_dim, sd->rvq_rest_output_proj, cb_dim,
                    1.0f, vq_out, vq_hidden);  /* accumulate */
    }
#else
    for (int f = 0; f < n_frames; f++) {
        if (sd->rvq_first_output_proj) {
            const float *emb = emb_first + (int64_t)f * cb_dim;
            for (int o = 0; o < vq_hidden; o++) {
                float sum = 0;
                for (int i = 0; i < cb_dim; i++)
                    sum += sd->rvq_first_output_proj[(int64_t)o * cb_dim + i] * emb[i];
                vq_out[(int64_t)f * vq_hidden + o] += sum;
            }
        }
        if (sd->rvq_rest_output_proj) {
            const float *rest = emb_rest + (int64_t)f * cb_dim;
            for (int o = 0; o < vq_hidden; o++) {
                float sum = 0;
                for (int i = 0; i < cb_dim; i++)
                    sum += sd->rvq_rest_output_proj[(int64_t)o * cb_dim + i] * rest[i];
                vq_out[(int64_t)f * vq_hidden + o] += sum;
            }
        }
    }
#endif
    free(emb_first); free(emb_rest);

    /* Debug: dump first frame's RVQ output */
    if (ctx->debug) {
        float rms0 = 0;
        for (int i = 0; i < vq_hidden; i++) rms0 += vq_out[i] * vq_out[i];
        rms0 = sqrtf(rms0 / vq_hidden);
        fprintf(stderr, "[DECODER] RVQ out frame 0 [:5]: [%.6f, %.6f, %.6f, %.6f, %.6f] RMS=%.6f\n",
                vq_out[0], vq_out[1], vq_out[2], vq_out[3], vq_out[4], rms0);
        /* Also check the transposed buffer */
    }

    /* Step 2: Pre-conv (512→1024, k=3, causal, pad_left=2) */
    /* Need channel-first format for conv: [vq_hidden, n_frames] */
    float *vq_cf = (float *)aligned_malloc((int64_t)vq_hidden * n_frames * sizeof(float));
    for (int f = 0; f < n_frames; f++)
        for (int d = 0; d < vq_hidden; d++)
            vq_cf[(int64_t)d * n_frames + f] = vq_out[(int64_t)f * vq_hidden + d];
    free(vq_out);

    /* Debug: check transposed buffer */
    if (ctx->debug) {
        float rms_cf = 0;
        for (int d = 0; d < vq_hidden; d++) {
            float v = vq_cf[(int64_t)d * n_frames + 0]; /* frame 0, channel d */
            rms_cf += v * v;
        }
        rms_cf = sqrtf(rms_cf / vq_hidden);
        fprintf(stderr, "[DECODER] vq_cf frame 0 ch[:5]: [%.6f, %.6f, %.6f, %.6f, %.6f] RMS=%.6f\n",
                vq_cf[0 * n_frames + 0], vq_cf[1 * n_frames + 0],
                vq_cf[2 * n_frames + 0], vq_cf[3 * n_frames + 0],
                vq_cf[4 * n_frames + 0], rms_cf);
        /* Check pre_conv weight */
        fprintf(stderr, "[DECODER] pre_conv_w[:5]: [%.6f, %.6f, %.6f, %.6f, %.6f] bias[:3]: [%.6f, %.6f, %.6f]\n",
                sd->pre_conv_weight[0], sd->pre_conv_weight[1], sd->pre_conv_weight[2],
                sd->pre_conv_weight[3], sd->pre_conv_weight[4],
                sd->pre_conv_bias[0], sd->pre_conv_bias[1], sd->pre_conv_bias[2]);
    }

    float *pre_conv_out = (float *)aligned_calloc((int64_t)latent_dim * n_frames, sizeof(float));
    causal_conv1d(pre_conv_out, vq_cf, sd->pre_conv_weight, sd->pre_conv_bias,
                  vq_hidden, latent_dim, n_frames, 3, 1);
    free(vq_cf);

    /* Debug: dump pre_conv output */
    if (ctx->debug) {
        fprintf(stderr, "[DECODER] pre_conv out frame 0 [:5]: [%.6f, %.6f, %.6f, %.6f, %.6f]\n",
                pre_conv_out[0 * n_frames + 0], pre_conv_out[1 * n_frames + 0],
                pre_conv_out[2 * n_frames + 0], pre_conv_out[3 * n_frames + 0],
                pre_conv_out[4 * n_frames + 0]);
    }

    /* Step 3: Transpose to row-major + input_proj (1024→512) */
    int dec_hidden = 512;
    float *hidden = (float *)aligned_malloc((int64_t)n_frames * dec_hidden * sizeof(float));
#ifdef USE_BLAS
    /* Transpose pre_conv_out from channel-first [1024, n_frames] to row-major [n_frames, 1024] */
    float *pre_conv_rm = (float *)aligned_malloc((int64_t)n_frames * latent_dim * sizeof(float));
    for (int f = 0; f < n_frames; f++)
        for (int d = 0; d < latent_dim; d++)
            pre_conv_rm[(int64_t)f * latent_dim + d] = pre_conv_out[(int64_t)d * n_frames + f];
    free(pre_conv_out);
    /* hidden[n_frames, 512] = pre_conv_rm[n_frames, 1024] × W^T[1024, 512] */
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                n_frames, dec_hidden, latent_dim, 1.0f,
                pre_conv_rm, latent_dim,
                sd->input_proj_weight, latent_dim,
                0.0f, hidden, dec_hidden);
    free(pre_conv_rm);
    if (sd->input_proj_bias) {
        for (int f = 0; f < n_frames; f++)
            for (int o = 0; o < dec_hidden; o++)
                hidden[(int64_t)f * dec_hidden + o] += sd->input_proj_bias[o];
    }
#else
    for (int f = 0; f < n_frames; f++) {
        for (int o = 0; o < dec_hidden; o++) {
            float sum = sd->input_proj_bias ? sd->input_proj_bias[o] : 0;
            for (int i = 0; i < latent_dim; i++)
                sum += sd->input_proj_weight[(int64_t)o * latent_dim + i] * pre_conv_out[(int64_t)i * n_frames + f];
            hidden[(int64_t)f * dec_hidden + o] = sum;
        }
    }
    free(pre_conv_out);
#endif

    /* Debug: dump input_proj output */
    if (ctx->debug) {
        fprintf(stderr, "[DECODER] input_proj out frame 0 [:5]: [%.6f, %.6f, %.6f, %.6f, %.6f]\n",
                hidden[0], hidden[1], hidden[2], hidden[3], hidden[4]);
    }

    /* Step 4: Pre-transformer (8 layers with sliding window causal attention) */
    int dec_inter = 1024;
    int n_heads = 16;
    int head_dim = c->dec_head_dim; /* 64 */
    int qkv_dim = n_heads * head_dim;
    int window = 72;
    float eps = c->dec_rms_norm_eps;
    int half_hd = head_dim / 2;

    float *q = (float *)aligned_malloc((int64_t)n_frames * qkv_dim * sizeof(float));
    float *kk = (float *)aligned_malloc((int64_t)n_frames * qkv_dim * sizeof(float));
    float *vv = (float *)aligned_malloc((int64_t)n_frames * qkv_dim * sizeof(float));
    float *x_norm = (float *)aligned_malloc((int64_t)n_frames * dec_hidden * sizeof(float));
    float *attn_out = (float *)aligned_malloc((int64_t)n_frames * qkv_dim * sizeof(float));

    for (int layer = 0; layer < c->dec_num_layers; layer++) {
        qwen_sd_pre_layer_t *l = &sd->pre_layers[layer];

        /* Input RMSNorm (NEON-optimized) */
        qwen_rms_norm(x_norm, hidden, l->attn_norm, n_frames, dec_hidden, eps);

        /* Debug after input RMSNorm for layer 0 */
        if (ctx->debug && layer == 0) {
            fprintf(stderr, "[DECODER] Layer 0 input_norm frame 0 [:5]: [%.6f, %.6f, %.6f, %.6f, %.6f]\n",
                    x_norm[0], x_norm[1], x_norm[2], x_norm[3], x_norm[4]);
        }

        /* QKV projections */
#ifdef USE_BLAS
        /* x_norm[n_frames, dec_hidden] × W^T = out[n_frames, qkv_dim] */
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    n_frames, qkv_dim, dec_hidden, 1.0f,
                    x_norm, dec_hidden, l->attn_q, dec_hidden, 0.0f, q, qkv_dim);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    n_frames, qkv_dim, dec_hidden, 1.0f,
                    x_norm, dec_hidden, l->attn_k, dec_hidden, 0.0f, kk, qkv_dim);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    n_frames, qkv_dim, dec_hidden, 1.0f,
                    x_norm, dec_hidden, l->attn_v, dec_hidden, 0.0f, vv, qkv_dim);
#else
        for (int s = 0; s < n_frames; s++) {
            const float *xs = x_norm + s * dec_hidden;
            float *qs = q + s * qkv_dim;
            float *ks = kk + s * qkv_dim;
            float *vs = vv + s * qkv_dim;
            for (int o = 0; o < qkv_dim; o++) {
                float sum_q = 0, sum_k = 0, sum_v = 0;
                for (int i = 0; i < dec_hidden; i++) {
                    sum_q += l->attn_q[(int64_t)o * dec_hidden + i] * xs[i];
                    sum_k += l->attn_k[(int64_t)o * dec_hidden + i] * xs[i];
                    sum_v += l->attn_v[(int64_t)o * dec_hidden + i] * xs[i];
                }
                qs[o] = sum_q; ks[o] = sum_k; vs[o] = sum_v;
            }
        }
#endif

        /* Debug QKV for layer 0 */
        if (ctx->debug && layer == 0) {
            fprintf(stderr, "[DECODER] Layer 0 Q frame 0 [:5]: [%.6f, %.6f, %.6f, %.6f, %.6f]\n",
                    q[0], q[1], q[2], q[3], q[4]);
            fprintf(stderr, "[DECODER] Layer 0 K frame 0 [:5]: [%.6f, %.6f, %.6f, %.6f, %.6f]\n",
                    kk[0], kk[1], kk[2], kk[3], kk[4]);
            fprintf(stderr, "[DECODER] Layer 0 V frame 0 [:5]: [%.6f, %.6f, %.6f, %.6f, %.6f]\n",
                    vv[0], vv[1], vv[2], vv[3], vv[4]);
        }

        /* NeoX split-half RoPE (NO QK-norm for pre-transformer) */
        for (int s = 0; s < n_frames; s++) {
            const float *cos_ptr = sd->rope_cos + s * half_hd;
            const float *sin_ptr = sd->rope_sin + s * half_hd;
            for (int h = 0; h < n_heads; h++) {
                float *qh = q + s * qkv_dim + h * head_dim;
                float *kh = kk + s * qkv_dim + h * head_dim;
                for (int i = 0; i < half_hd; i++) {
                    float q1 = qh[i], q2 = qh[i + half_hd];
                    float k1 = kh[i], k2 = kh[i + half_hd];
                    float co = cos_ptr[i], si = sin_ptr[i];
                    qh[i]           = q1 * co - q2 * si;
                    qh[i + half_hd] = q2 * co + q1 * si;
                    kh[i]           = k1 * co - k2 * si;
                    kh[i + half_hd] = k2 * co + k1 * si;
                }
            }
        }

        /* Sliding window causal attention with proper softmax */
        float scale = 1.0f / sqrtf((float)head_dim);
        for (int sq = 0; sq < n_frames; sq++) {
            float *out = attn_out + sq * qkv_dim;
            memset(out, 0, qkv_dim * sizeof(float));
            int sk_start = (sq - window + 1 > 0) ? sq - window + 1 : 0;

            for (int h = 0; h < n_heads; h++) {
                const float *qh = q + sq * qkv_dim + h * head_dim;
                float *oh = out + h * head_dim;

                /* Compute scores and find max for numerical stability */
                int n_keys = sq - sk_start + 1;
                float scores[n_keys];
                float max_score = -1e30f;
                for (int j = 0; j < n_keys; j++) {
                    int sk = sk_start + j;
                    const float *kh = kk + sk * qkv_dim + h * head_dim;
                    float dot = 0;
                    for (int d = 0; d < head_dim; d++) dot += qh[d] * kh[d];
                    scores[j] = dot * scale;
                    if (scores[j] > max_score) max_score = scores[j];
                }

                /* Softmax */
                float sum_exp = 0;
                for (int j = 0; j < n_keys; j++) {
                    scores[j] = expf(scores[j] - max_score);
                    sum_exp += scores[j];
                }
                float inv_sum = 1.0f / sum_exp;

                /* Weighted sum of values */
                for (int j = 0; j < n_keys; j++) {
                    int sk = sk_start + j;
                    const float *vh = vv + sk * qkv_dim + h * head_dim;
                    float w = scores[j] * inv_sum;
                    for (int d = 0; d < head_dim; d++) oh[d] += vh[d] * w;
                }
            }
        }

        /* Output proj + layer_scale + residual */
#ifdef USE_BLAS
        {
            /* proj[n_frames, dec_hidden] = attn_out[n_frames, qkv_dim] × attn_o^T */
            float *oproj = x_norm; /* reuse x_norm as temp (same size: n_frames * dec_hidden) */
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        n_frames, dec_hidden, qkv_dim, 1.0f,
                        attn_out, qkv_dim, l->attn_o, qkv_dim,
                        0.0f, oproj, dec_hidden);
            for (int s = 0; s < n_frames; s++) {
                float *xs = hidden + s * dec_hidden;
                float *ps = oproj + s * dec_hidden;
                if (l->attn_layer_scale) {
                    for (int o = 0; o < dec_hidden; o++)
                        xs[o] += ps[o] * l->attn_layer_scale[o];
                } else {
                    for (int o = 0; o < dec_hidden; o++)
                        xs[o] += ps[o];
                }
            }
        }
#else
        for (int s = 0; s < n_frames; s++) {
            float *xs = hidden + s * dec_hidden;
            const float *attn = attn_out + s * qkv_dim;
            for (int o = 0; o < dec_hidden; o++) {
                float sum = 0;
                for (int i = 0; i < qkv_dim; i++)
                    sum += l->attn_o[(int64_t)o * qkv_dim + i] * attn[i];
                if (l->attn_layer_scale)
                    sum *= l->attn_layer_scale[o];
                xs[o] += sum;
            }
        }
#endif

        /* Debug after attention + residual */
        if (ctx->debug && layer == 0) {
            fprintf(stderr, "[DECODER] Layer 0 after attn+res frame 0 [:5]: [%.6f, %.6f, %.6f, %.6f, %.6f]\n",
                    hidden[0], hidden[1], hidden[2], hidden[3], hidden[4]);
        }

        /* Post-attn RMSNorm (NEON-optimized) */
        qwen_rms_norm(x_norm, hidden, l->ffn_norm, n_frames, dec_hidden, eps);

        /* SwiGLU FFN: down_proj(SiLU(gate_proj(x)) * up_proj(x)) + layer_scale + residual */
#ifdef USE_BLAS
        {
            float *ffn_gate = (float *)aligned_malloc((int64_t)n_frames * dec_inter * sizeof(float));
            float *ffn_up = (float *)aligned_malloc((int64_t)n_frames * dec_inter * sizeof(float));
            /* gate[n_frames, dec_inter] = x_norm[n_frames, dec_hidden] × W_gate^T */
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        n_frames, dec_inter, dec_hidden, 1.0f,
                        x_norm, dec_hidden, l->ffn_gate, dec_hidden,
                        0.0f, ffn_gate, dec_inter);
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        n_frames, dec_inter, dec_hidden, 1.0f,
                        x_norm, dec_hidden, l->ffn_up, dec_hidden,
                        0.0f, ffn_up, dec_inter);
            /* SiLU(gate) * up */
            for (int64_t i = 0; i < (int64_t)n_frames * dec_inter; i++)
                ffn_gate[i] = (ffn_gate[i] / (1.0f + expf(-ffn_gate[i]))) * ffn_up[i];
            free(ffn_up);
            /* down[n_frames, dec_hidden] = ffn_gate[n_frames, dec_inter] × W_down^T */
            float *ffn_down_out = ffn_up = (float *)aligned_malloc((int64_t)n_frames * dec_hidden * sizeof(float));
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        n_frames, dec_hidden, dec_inter, 1.0f,
                        ffn_gate, dec_inter, l->ffn_down, dec_inter,
                        0.0f, ffn_down_out, dec_hidden);
            free(ffn_gate);
            /* layer_scale + residual */
            for (int s = 0; s < n_frames; s++) {
                float *hs = hidden + s * dec_hidden;
                float *ds = ffn_down_out + s * dec_hidden;
                if (l->ffn_layer_scale) {
                    for (int o = 0; o < dec_hidden; o++)
                        hs[o] += ds[o] * l->ffn_layer_scale[o];
                } else {
                    for (int o = 0; o < dec_hidden; o++)
                        hs[o] += ds[o];
                }
            }
            free(ffn_down_out);
        }
#else
        for (int s = 0; s < n_frames; s++) {
            const float *xs = x_norm + s * dec_hidden;
            float *hs = hidden + s * dec_hidden;

            /* gate and up projections */
            float gate_up[dec_inter * 2]; /* VLA */
            for (int o = 0; o < dec_inter; o++) {
                float sum_g = 0, sum_u = 0;
                for (int i = 0; i < dec_hidden; i++) {
                    sum_g += l->ffn_gate[(int64_t)o * dec_hidden + i] * xs[i];
                    sum_u += l->ffn_up[(int64_t)o * dec_hidden + i] * xs[i];
                }
                /* SiLU on gate, multiply by up */
                gate_up[o] = (sum_g / (1.0f + expf(-sum_g))) * sum_u;
            }

            /* down projection + layer_scale + residual */
            for (int o = 0; o < dec_hidden; o++) {
                float sum = 0;
                for (int i = 0; i < dec_inter; i++)
                    sum += l->ffn_down[(int64_t)o * dec_inter + i] * gate_up[i];
                if (l->ffn_layer_scale)
                    sum *= l->ffn_layer_scale[o];
                hs[o] += sum;
            }
        }
#endif

        /* Per-layer debug */
        if (ctx->debug) {
            fprintf(stderr, "[DECODER] Layer %d out frame 0 [:5]: [%.6f, %.6f, %.6f, %.6f, %.6f]\n",
                    layer, hidden[0], hidden[1], hidden[2], hidden[3], hidden[4]);
        }
    }

    free(q); free(kk); free(vv); free(x_norm); free(attn_out);

    /* Debug: after pre-transformer */
    if (ctx->debug) {
        fprintf(stderr, "[DECODER] pre-trans out frame 0 [:5]: [%.6f, %.6f, %.6f, %.6f, %.6f]\n",
                hidden[0], hidden[1], hidden[2], hidden[3], hidden[4]);
    }

    /* Step 5: Final RMSNorm + Output proj (512→1024) */
    if (sd->final_norm_weight) {
        qwen_rms_norm(hidden, hidden, sd->final_norm_weight, n_frames, dec_hidden, eps);
    }

    if (ctx->debug) {
        fprintf(stderr, "[DECODER] final_norm frame 0 [:5]: [%.6f, %.6f, %.6f, %.6f, %.6f]\n",
                hidden[0], hidden[1], hidden[2], hidden[3], hidden[4]);
    }

    float *latent_out = (float *)aligned_malloc((int64_t)n_frames * latent_dim * sizeof(float));
#ifdef USE_BLAS
    /* latent_out[n_frames, 1024] = hidden[n_frames, 512] × W^T[512, 1024] */
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                n_frames, latent_dim, dec_hidden, 1.0f,
                hidden, dec_hidden,
                sd->output_proj_weight, dec_hidden,
                0.0f, latent_out, latent_dim);
    if (sd->output_proj_bias) {
        for (int f = 0; f < n_frames; f++)
            for (int o = 0; o < latent_dim; o++)
                latent_out[(int64_t)f * latent_dim + o] += sd->output_proj_bias[o];
    }
#else
    for (int f = 0; f < n_frames; f++) {
        for (int o = 0; o < latent_dim; o++) {
            float sum = sd->output_proj_bias ? sd->output_proj_bias[o] : 0;
            for (int i = 0; i < dec_hidden; i++)
                sum += sd->output_proj_weight[(int64_t)o * dec_hidden + i] * hidden[(int64_t)f * dec_hidden + i];
            latent_out[(int64_t)f * latent_dim + o] = sum;
        }
    }
#endif
    free(hidden);

    /* Step 6: Transpose to channel-first [1024, n_frames] */
    float *signal = (float *)aligned_malloc((int64_t)latent_dim * n_frames * sizeof(float));
    for (int f = 0; f < n_frames; f++)
        for (int d = 0; d < latent_dim; d++)
            signal[(int64_t)d * n_frames + f] = latent_out[(int64_t)f * latent_dim + d];
    free(latent_out);

    /* Debug: after output_proj */
    if (ctx->debug) {
        fprintf(stderr, "[DECODER] output_proj out frame 0 [:5]: [%.6f, %.6f, %.6f, %.6f, %.6f]\n",
                signal[0 * n_frames + 0], signal[1 * n_frames + 0],
                signal[2 * n_frames + 0], signal[3 * n_frames + 0],
                signal[4 * n_frames + 0]);
    }
    if (ctx->debug) fprintf(stderr, "[DECODER] ConvNeXt upsample...\n");
    int cur_ch = latent_dim;
    int cur_len = n_frames;

    /* Step 7: ConvNeXt upsample (2 blocks, 2x each) */
    for (int b = 0; b < 2; b++) {
        qwen_sd_convnext_t *cn = &sd->convnext[b];
        if (!cn->conv_weight) { fprintf(stderr, "ERROR: ConvNeXt block %d weights missing!\n", b); free(signal); return -1; }

        int new_len = conv_transpose1d_out_len(cur_len, 2, 2);

        /* Full ConvTranspose1d 2x upsample: [1024, 1024, 2] */
        float *up_out = (float *)aligned_calloc((int64_t)cur_ch * new_len, sizeof(float));
        causal_conv_transpose1d(up_out, signal, cn->conv_weight, cn->conv_bias,
                                 cur_ch, cur_ch, cur_len, new_len, 2, 2);
        free(signal); signal = up_out; cur_len = new_len;

        /* ConvNeXt block: DW conv → LayerNorm → PW1 → GELU → PW2 → gamma → residual */
        float *residual = (float *)aligned_malloc((int64_t)cur_ch * cur_len * sizeof(float));
        memcpy(residual, signal, (int64_t)cur_ch * cur_len * sizeof(float));

        /* Depthwise conv (k=7, groups=cur_ch, pad_left=6) */
        float *dw_out = (float *)aligned_calloc((int64_t)cur_ch * cur_len, sizeof(float));
        for (int ci = 0; ci < cur_ch; ci++) {
            for (int t = 0; t < cur_len; t++) {
                float sum = cn->dwconv_bias ? cn->dwconv_bias[ci] : 0;
                for (int k = 0; k < 7; k++) {
                    int in_pos = t - (6 - k);
                    if (in_pos >= 0 && in_pos < cur_len)
                        sum += cn->dwconv_weight[(int64_t)ci * 7 + k] * signal[(int64_t)ci * cur_len + in_pos];
                }
                dw_out[(int64_t)ci * cur_len + t] = sum;
            }
        }
        memcpy(signal, dw_out, (int64_t)cur_ch * cur_len * sizeof(float));
        free(dw_out);

        /* LayerNorm per timestep (over channels) */
        for (int t = 0; t < cur_len; t++) {
            float sum = 0, sum_sq = 0;
            for (int ci = 0; ci < cur_ch; ci++) {
                float val = signal[(int64_t)ci * cur_len + t];
                sum += val; sum_sq += val * val;
            }
            float mean = sum / cur_ch;
            float var = sum_sq / cur_ch - mean * mean;
            float inv_std = 1.0f / sqrtf(var + 1e-5f);
            for (int ci = 0; ci < cur_ch; ci++) {
                float *p = &signal[(int64_t)ci * cur_len + t];
                *p = (*p - mean) * inv_std * cn->norm_weight[ci] + cn->norm_bias[ci];
            }
        }

        /* PW1: 1024→4096 (pointwise = 1x1 conv = matmul per timestep) */
        int pw_dim = cur_ch * 4;
        float *pw1_out = (float *)aligned_malloc((int64_t)pw_dim * cur_len * sizeof(float));
#ifdef USE_BLAS
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    pw_dim, cur_len, cur_ch,
                    1.0f, cn->pwconv1_weight, cur_ch,
                    signal, cur_len,
                    0.0f, pw1_out, cur_len);
        conv_add_bias(pw1_out, cn->pwconv1_bias, pw_dim, cur_len);
#else
        for (int t = 0; t < cur_len; t++) {
            for (int o = 0; o < pw_dim; o++) {
                float sum = cn->pwconv1_bias ? cn->pwconv1_bias[o] : 0;
                for (int i = 0; i < cur_ch; i++)
                    sum += cn->pwconv1_weight[(int64_t)o * cur_ch + i] * signal[(int64_t)i * cur_len + t];
                pw1_out[(int64_t)o * cur_len + t] = sum;
            }
        }
#endif

        /* Exact GELU: x * 0.5 * (1 + erf(x / sqrt(2))) */
        for (int64_t i = 0; i < (int64_t)pw_dim * cur_len; i++) {
            float x = pw1_out[i];
            pw1_out[i] = 0.5f * x * (1.0f + erff(x * 0.7071067811865476f));
        }

        /* PW2: 4096→1024 */
#ifdef USE_BLAS
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    cur_ch, cur_len, pw_dim,
                    1.0f, cn->pwconv2_weight, pw_dim,
                    pw1_out, cur_len,
                    0.0f, signal, cur_len);
        conv_add_bias(signal, cn->pwconv2_bias, cur_ch, cur_len);
#else
        for (int t = 0; t < cur_len; t++) {
            for (int o = 0; o < cur_ch; o++) {
                float sum = cn->pwconv2_bias ? cn->pwconv2_bias[o] : 0;
                for (int i = 0; i < pw_dim; i++)
                    sum += cn->pwconv2_weight[(int64_t)o * pw_dim + i] * pw1_out[(int64_t)i * cur_len + t];
                signal[(int64_t)o * cur_len + t] = sum;
            }
        }
#endif
        free(pw1_out);

        /* Gamma + residual */
        for (int ci = 0; ci < cur_ch; ci++) {
            float g = cn->gamma[ci];
            for (int t = 0; t < cur_len; t++)
                signal[(int64_t)ci * cur_len + t] = residual[(int64_t)ci * cur_len + t]
                    + signal[(int64_t)ci * cur_len + t] * g;
        }
        free(residual);
    }

    if (ctx->debug) fprintf(stderr, "[DECODER] Initial conv...\n");

    /* Step 8: Initial conv (1024→1536, k=7, pad_left=6) */
    if (!sd->initial_conv_weight) { free(signal); return -1; }
    int new_ch = 1536;
    int new_len = conv1d_out_len(cur_len, 7, 1, 6);
    float *conv_out = (float *)aligned_calloc((int64_t)new_ch * new_len, sizeof(float));
    causal_conv1d(conv_out, signal, sd->initial_conv_weight, sd->initial_conv_bias,
                  cur_ch, new_ch, cur_len, 7, 1);
    free(signal); signal = conv_out; cur_ch = new_ch; cur_len = new_len;

    /* Step 9: 4 Decoder upsample blocks */
    int up_rates[4] = {8, 5, 4, 3};
    int out_channels[4] = {768, 384, 192, 96};

    if (ctx->debug) fprintf(stderr, "[DECODER] Upsample blocks...\n");
    for (int b = 0; b < 4; b++) {
        qwen_sd_upsample_block_t *ub = &sd->upsample_blocks[b];
        int rate = up_rates[b];
        int kernel = rate * 2;
        int out_ch = out_channels[b];

        if (!ub->upsample.conv_weight) {
            fprintf(stderr, "ERROR: Upsample block %d weights missing!\n", b);
            free(signal); return -1;
        }

        /* Snake activation before upsample */
        if (ub->upsample.snake_alpha && ub->upsample.snake_beta)
            snake_activation(signal, cur_ch, cur_len, ub->upsample.snake_alpha, ub->upsample.snake_beta);

        /* ConvTranspose1d upsample: [in_ch, out_ch, kernel] */
        int up_len = conv_transpose1d_out_len(cur_len, kernel, rate);
        float *up_out = (float *)aligned_calloc((int64_t)out_ch * up_len, sizeof(float));
        causal_conv_transpose1d(up_out, signal, ub->upsample.conv_weight, ub->upsample.conv_bias,
                                 cur_ch, out_ch, cur_len, up_len, kernel, rate);
        free(signal); signal = up_out; cur_ch = out_ch; cur_len = up_len;

        /* 3 residual blocks with dilations [1, 3, 9] */
        int dilations[3] = {1, 3, 9};
        for (int r = 0; r < 3; r++) {
            int dil = dilations[r];

            float *res = (float *)aligned_malloc((int64_t)cur_ch * cur_len * sizeof(float));
            memcpy(res, signal, (int64_t)cur_ch * cur_len * sizeof(float));

            /* Snake 1 */
            if (ub->res_blocks[r].snake1_alpha && ub->res_blocks[r].snake1_beta)
                snake_activation(signal, cur_ch, cur_len,
                                  ub->res_blocks[r].snake1_alpha, ub->res_blocks[r].snake1_beta);

            /* Conv1 (k=7, dilation, causal): [ch, ch, 7] */
            float *c1_out = (float *)aligned_calloc((int64_t)cur_ch * cur_len, sizeof(float));
            causal_conv1d(c1_out, signal, ub->res_blocks[r].conv1_weight, ub->res_blocks[r].conv1_bias,
                          cur_ch, cur_ch, cur_len, 7, dil);
            memcpy(signal, c1_out, (int64_t)cur_ch * cur_len * sizeof(float));
            free(c1_out);

            /* Snake 2 */
            if (ub->res_blocks[r].snake2_alpha && ub->res_blocks[r].snake2_beta)
                snake_activation(signal, cur_ch, cur_len,
                                  ub->res_blocks[r].snake2_alpha, ub->res_blocks[r].snake2_beta);

            /* Conv2 (k=1): [ch, ch, 1] */
            float *c2_out = (float *)aligned_calloc((int64_t)cur_ch * cur_len, sizeof(float));
            causal_conv1d(c2_out, signal, ub->res_blocks[r].conv2_weight, ub->res_blocks[r].conv2_bias,
                          cur_ch, cur_ch, cur_len, 1, 1);

            /* Residual add */
            for (int64_t i = 0; i < (int64_t)cur_ch * cur_len; i++)
                signal[i] = res[i] + c2_out[i];
            free(c2_out);
            free(res);
        }

        if (ctx->debug) fprintf(stderr, "[DECODER]   Block %d done: ch=%d, len=%d\n", b+1, cur_ch, cur_len);
    }

    if (ctx->debug) fprintf(stderr, "[DECODER] Final conv...\n");

    /* Step 10: Final Snake + Conv (96→1, k=7) */
    if (!sd->final_snake.alpha || !sd->final_conv_weight) {
        fprintf(stderr, "ERROR: Final snake/conv weights missing!\n");
        free(signal); return -1;
    }

    /* Final snake activation */
    snake_activation(signal, cur_ch, cur_len, sd->final_snake.alpha, sd->final_snake.beta);

    /* Final conv: [1, 96, 7] */
    int audio_len = conv1d_out_len(cur_len, 7, 1, 6);
    float *audio = (float *)aligned_calloc(audio_len, sizeof(float));
    for (int t = 0; t < audio_len; t++) {
        float sum = sd->final_conv_bias ? sd->final_conv_bias[0] : 0;
        for (int ic = 0; ic < cur_ch; ic++) {
            for (int k = 0; k < 7; k++) {
                int in_pos = t - (6 - k);
                if (in_pos >= 0 && in_pos < cur_len)
                    sum += sd->final_conv_weight[(int64_t)ic * 7 + k] * signal[(int64_t)ic * cur_len + in_pos];
            }
        }
        audio[t] = sum;
    }
    free(signal);

    /* Clamp to [-1, 1] */
    for (int i = 0; i < audio_len; i++) {
        if (audio[i] < -1.0f) audio[i] = -1.0f;
        if (audio[i] > 1.0f) audio[i] = 1.0f;
    }

    *audio_out = audio;
    *n_samples = audio_len;

    if (!ctx->silent)
        fprintf(stderr, "  Speech decoder output: %d samples (%.2fs @ 24kHz)\n",
                audio_len, (float)audio_len / 24000.0f);

    return 0;
}

/* ========================================================================
 * Streaming Incremental Decode
 *
 * Instead of re-decoding ALL accumulated frames each streaming chunk (O(n²)),
 * this processes only NEW frames through VQ → pre-transformer (with KV cache),
 * caches the latent output, and runs the conv decoder on a small window
 * (context + new frames) for O(1) per chunk.
 *
 * Audio output is exactly 1920 samples per codec frame (by design: 4×480× upsample).
 * ======================================================================== */

/* Initialize/reset streaming state */
void qwen_sd_stream_init(qwen_sd_stream_state_t *st) {
    memset(st, 0, sizeof(*st));
}

/* Free streaming state buffers */
void qwen_sd_stream_free(qwen_sd_stream_state_t *st) {
    for (int i = 0; i < QWEN_SD_STREAM_MAX_LAYERS; i++) {
        free(st->k_cache[i]); st->k_cache[i] = NULL;
        free(st->v_cache[i]); st->v_cache[i] = NULL;
    }
    free(st->latent_cache); st->latent_cache = NULL;
    free(st->vq_pad); st->vq_pad = NULL;
    memset(st, 0, sizeof(*st));
}

/* Run conv decoder (ConvNeXt + initial conv + upsample blocks + final conv)
 * on a signal in channel-first format [latent_dim, n_frames].
 * Returns audio samples. This is the same pipeline as steps 7-10 in the
 * full decode, extracted as a helper to avoid duplication. */
static int conv_decoder_forward(qwen_tts_ctx_t *ctx,
                                 float *signal, int cur_ch, int cur_len,
                                 float **audio_out, int *n_samples_out) {
    qwen_speech_decoder_t *sd = &ctx->speech_dec;

    /* ConvNeXt upsample (2 blocks, 2x each → 4x total) */
    for (int b = 0; b < 2; b++) {
        qwen_sd_convnext_t *cn = &sd->convnext[b];
        if (!cn->conv_weight) { free(signal); return -1; }

        int new_len = conv_transpose1d_out_len(cur_len, 2, 2);
        float *up_out = (float *)aligned_calloc((int64_t)cur_ch * new_len, sizeof(float));
        causal_conv_transpose1d(up_out, signal, cn->conv_weight, cn->conv_bias,
                                 cur_ch, cur_ch, cur_len, new_len, 2, 2);
        free(signal); signal = up_out; cur_len = new_len;

        /* Depthwise conv (k=7, pad=6) */
        float *dw_out = (float *)aligned_calloc((int64_t)cur_ch * cur_len, sizeof(float));
        for (int c = 0; c < cur_ch; c++) {
            for (int t = 0; t < cur_len; t++) {
                float sum = cn->dwconv_bias ? cn->dwconv_bias[c] : 0;
                for (int k = 0; k < 7; k++) {
                    int in_pos = t - 6 + k;
                    if (in_pos >= 0 && in_pos < cur_len)
                        sum += cn->dwconv_weight[c * 7 + k] * signal[(int64_t)c * cur_len + in_pos];
                }
                dw_out[(int64_t)c * cur_len + t] = sum;
            }
        }

        float *residual = signal; signal = dw_out;

        /* LayerNorm (per-timestep) */
        for (int t = 0; t < cur_len; t++) {
            float mean = 0, var = 0;
            for (int c = 0; c < cur_ch; c++) mean += signal[(int64_t)c * cur_len + t];
            mean /= cur_ch;
            for (int c = 0; c < cur_ch; c++) {
                float d = signal[(int64_t)c * cur_len + t] - mean;
                var += d * d;
            }
            var = 1.0f / sqrtf(var / cur_ch + 1e-5f);
            for (int c = 0; c < cur_ch; c++) {
                float x = (signal[(int64_t)c * cur_len + t] - mean) * var;
                signal[(int64_t)c * cur_len + t] = x * cn->norm_weight[c] + cn->norm_bias[c];
            }
        }

        /* Pointwise convs: pw1 (1024→4096, GELU), pw2 (4096→1024) */
        int pw_dim = 4096;
        float *pw1_out = (float *)aligned_malloc((int64_t)pw_dim * cur_len * sizeof(float));
#ifdef USE_BLAS
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    pw_dim, cur_len, cur_ch, 1.0f,
                    cn->pwconv1_weight, cur_ch, signal, cur_len,
                    0.0f, pw1_out, cur_len);
        if (cn->pwconv1_bias)
            for (int i = 0; i < pw_dim; i++)
                for (int t = 0; t < cur_len; t++)
                    pw1_out[(int64_t)i * cur_len + t] += cn->pwconv1_bias[i];
#else
        for (int o = 0; o < pw_dim; o++)
            for (int t = 0; t < cur_len; t++) {
                float sum = cn->pwconv1_bias ? cn->pwconv1_bias[o] : 0;
                for (int i = 0; i < cur_ch; i++)
                    sum += cn->pwconv1_weight[(int64_t)o * cur_ch + i] * signal[(int64_t)i * cur_len + t];
                pw1_out[(int64_t)o * cur_len + t] = sum;
            }
#endif
        /* Exact GELU */
        for (int64_t i = 0; i < (int64_t)pw_dim * cur_len; i++)
            pw1_out[i] = 0.5f * pw1_out[i] * (1.0f + erff(pw1_out[i] * 0.7071067811865476f));

        /* pw2 */
        memset(signal, 0, (int64_t)cur_ch * cur_len * sizeof(float));
#ifdef USE_BLAS
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    cur_ch, cur_len, pw_dim, 1.0f,
                    cn->pwconv2_weight, pw_dim, pw1_out, cur_len,
                    0.0f, signal, cur_len);
        if (cn->pwconv2_bias)
            for (int o = 0; o < cur_ch; o++)
                for (int t = 0; t < cur_len; t++)
                    signal[(int64_t)o * cur_len + t] += cn->pwconv2_bias[o];
#else
        for (int o = 0; o < cur_ch; o++)
            for (int t = 0; t < cur_len; t++) {
                float sum = cn->pwconv2_bias ? cn->pwconv2_bias[o] : 0;
                for (int i = 0; i < pw_dim; i++)
                    sum += cn->pwconv2_weight[(int64_t)o * pw_dim + i] * pw1_out[(int64_t)i * cur_len + t];
                signal[(int64_t)o * cur_len + t] = sum;
            }
#endif
        free(pw1_out);

        /* Gamma + residual */
        for (int ci = 0; ci < cur_ch; ci++) {
            float g = cn->gamma[ci];
            for (int t = 0; t < cur_len; t++)
                signal[(int64_t)ci * cur_len + t] = residual[(int64_t)ci * cur_len + t]
                    + signal[(int64_t)ci * cur_len + t] * g;
        }
        free(residual);
    }

    /* Initial conv (1024→1536, k=7, pad_left=6) */
    if (!sd->initial_conv_weight) { free(signal); return -1; }
    int new_ch = 1536;
    int new_len = conv1d_out_len(cur_len, 7, 1, 6);
    float *conv_out = (float *)aligned_calloc((int64_t)new_ch * new_len, sizeof(float));
    causal_conv1d(conv_out, signal, sd->initial_conv_weight, sd->initial_conv_bias,
                  cur_ch, new_ch, cur_len, 7, 1);
    free(signal); signal = conv_out; cur_ch = new_ch; cur_len = new_len;

    /* 4 Decoder upsample blocks */
    int up_rates[4] = {8, 5, 4, 3};
    int out_channels[4] = {768, 384, 192, 96};

    for (int b = 0; b < 4; b++) {
        qwen_sd_upsample_block_t *ub = &sd->upsample_blocks[b];
        int rate = up_rates[b];
        int kernel = rate * 2;
        int out_ch = out_channels[b];

        if (!ub->upsample.conv_weight) { free(signal); return -1; }

        if (ub->upsample.snake_alpha && ub->upsample.snake_beta)
            snake_activation(signal, cur_ch, cur_len, ub->upsample.snake_alpha, ub->upsample.snake_beta);

        int up_len = conv_transpose1d_out_len(cur_len, kernel, rate);
        float *up_out = (float *)aligned_calloc((int64_t)out_ch * up_len, sizeof(float));
        causal_conv_transpose1d(up_out, signal, ub->upsample.conv_weight, ub->upsample.conv_bias,
                                 cur_ch, out_ch, cur_len, up_len, kernel, rate);
        free(signal); signal = up_out; cur_ch = out_ch; cur_len = up_len;

        int dilations[3] = {1, 3, 9};
        for (int r = 0; r < 3; r++) {
            int dil = dilations[r];
            float *res = (float *)aligned_malloc((int64_t)cur_ch * cur_len * sizeof(float));
            memcpy(res, signal, (int64_t)cur_ch * cur_len * sizeof(float));

            if (ub->res_blocks[r].snake1_alpha && ub->res_blocks[r].snake1_beta)
                snake_activation(signal, cur_ch, cur_len,
                                  ub->res_blocks[r].snake1_alpha, ub->res_blocks[r].snake1_beta);

            float *c1_out = (float *)aligned_calloc((int64_t)cur_ch * cur_len, sizeof(float));
            causal_conv1d(c1_out, signal, ub->res_blocks[r].conv1_weight, ub->res_blocks[r].conv1_bias,
                          cur_ch, cur_ch, cur_len, 7, dil);
            memcpy(signal, c1_out, (int64_t)cur_ch * cur_len * sizeof(float));
            free(c1_out);

            if (ub->res_blocks[r].snake2_alpha && ub->res_blocks[r].snake2_beta)
                snake_activation(signal, cur_ch, cur_len,
                                  ub->res_blocks[r].snake2_alpha, ub->res_blocks[r].snake2_beta);

            float *c2_out = (float *)aligned_calloc((int64_t)cur_ch * cur_len, sizeof(float));
            causal_conv1d(c2_out, signal, ub->res_blocks[r].conv2_weight, ub->res_blocks[r].conv2_bias,
                          cur_ch, cur_ch, cur_len, 1, 1);

            for (int64_t i = 0; i < (int64_t)cur_ch * cur_len; i++)
                signal[i] = res[i] + c2_out[i];
            free(c2_out);
            free(res);
        }
    }

    /* Final Snake + Conv (96→1, k=7) */
    if (!sd->final_snake.alpha || !sd->final_conv_weight) { free(signal); return -1; }
    snake_activation(signal, cur_ch, cur_len, sd->final_snake.alpha, sd->final_snake.beta);

    int audio_len = conv1d_out_len(cur_len, 7, 1, 6);
    float *audio = (float *)aligned_calloc(audio_len, sizeof(float));
    for (int t = 0; t < audio_len; t++) {
        float sum = sd->final_conv_bias ? sd->final_conv_bias[0] : 0;
        for (int ic = 0; ic < cur_ch; ic++) {
            for (int k = 0; k < 7; k++) {
                int in_pos = t - (6 - k);
                if (in_pos >= 0 && in_pos < cur_len)
                    sum += sd->final_conv_weight[(int64_t)ic * 7 + k] * signal[(int64_t)ic * cur_len + in_pos];
            }
        }
        audio[t] = sum;
    }
    free(signal);

    for (int i = 0; i < audio_len; i++) {
        if (audio[i] < -1.0f) audio[i] = -1.0f;
        if (audio[i] > 1.0f) audio[i] = 1.0f;
    }

    *audio_out = audio;
    *n_samples_out = audio_len;
    return 0;
}

/* Incremental streaming decode: process only new_frames through VQ→pre-transformer
 * (using KV cache), cache latent output, run conv decoder on windowed latent.
 * Returns only NEW audio samples (not previously emitted ones). */
int qwen_speech_decoder_decode_streaming(qwen_tts_ctx_t *ctx,
                                          const int *new_codes, int new_frames,
                                          float **audio_out, int *n_samples) {
    qwen_speech_decoder_t *sd = &ctx->speech_dec;
    qwen_sd_stream_state_t *st = &ctx->sd_stream;
    qwen_tts_config_t *c = &ctx->config;

    int cb_dim = QWEN_TTS_CODEBOOK_DIM;
    int vq_hidden = 512;
    int latent_dim = 1024;
    int dec_hidden = 512;
    int dec_inter = 1024;
    int n_heads = 16;
    int head_dim = c->dec_head_dim;
    int qkv_dim = n_heads * head_dim;
    int window = 72;
    float eps = c->dec_rms_norm_eps;
    int half_hd = head_dim / 2;

    /* === Step 1: VQ dequant for new frames only === */
    /* Output: vq_out row-major [new_frames, 512] */
    float *vq_out = (float *)aligned_calloc((int64_t)new_frames * vq_hidden, sizeof(float));
    float *cb_sum = (float *)aligned_malloc(cb_dim * sizeof(float));

    for (int f = 0; f < new_frames; f++) {
        int code0 = new_codes[f * 16];
        if (code0 >= 0 && code0 < c->codebook_size && sd->codebook[0]) {
            const float *emb = sd->codebook[0] + (int64_t)code0 * cb_dim;
            if (sd->rvq_first_output_proj) {
                for (int o = 0; o < vq_hidden; o++) {
                    float sum = 0;
                    for (int i = 0; i < cb_dim; i++)
                        sum += sd->rvq_first_output_proj[(int64_t)o * cb_dim + i] * emb[i];
                    vq_out[(int64_t)f * vq_hidden + o] += sum;
                }
            }
        }
        memset(cb_sum, 0, cb_dim * sizeof(float));
        for (int k = 1; k < 16; k++) {
            int code = new_codes[f * 16 + k];
            if (code >= 0 && code < c->codebook_size && sd->codebook[k]) {
                const float *emb = sd->codebook[k] + (int64_t)code * cb_dim;
                for (int d = 0; d < cb_dim; d++) cb_sum[d] += emb[d];
            }
        }
        if (sd->rvq_rest_output_proj) {
            for (int o = 0; o < vq_hidden; o++) {
                float sum = 0;
                for (int i = 0; i < cb_dim; i++)
                    sum += sd->rvq_rest_output_proj[(int64_t)o * cb_dim + i] * cb_sum[i];
                vq_out[(int64_t)f * vq_hidden + o] += sum;
            }
        }
    }
    free(cb_sum);

    /* === Step 2: Pre-conv on new frames with padding from previous chunk === */
    /* VQ output is row-major [new_frames, 512]. Transpose to channel-first [512, new_frames]
     * for conv1d, prepending 2 frames of padding. */
    int pad_frames = st->vq_pad_valid ? 2 : 0;
    int conv_in_len = pad_frames + new_frames;
    float *vq_cf = (float *)aligned_calloc((int64_t)vq_hidden * conv_in_len, sizeof(float));

    /* Copy padding */
    if (st->vq_pad_valid && st->vq_pad) {
        for (int ch = 0; ch < vq_hidden; ch++)
            for (int t = 0; t < 2; t++)
                vq_cf[(int64_t)ch * conv_in_len + t] = st->vq_pad[(int64_t)ch * 2 + t];
    }
    /* Copy new VQ output (transpose row→channel-first) */
    for (int f = 0; f < new_frames; f++)
        for (int ch = 0; ch < vq_hidden; ch++)
            vq_cf[(int64_t)ch * conv_in_len + pad_frames + f] = vq_out[(int64_t)f * vq_hidden + ch];

    /* Save last 2 frames of VQ output (channel-first) as padding for next chunk */
    if (!st->vq_pad) st->vq_pad = (float *)aligned_malloc(vq_hidden * 2 * sizeof(float));
    int save_start = (conv_in_len >= 2) ? conv_in_len - 2 : 0;
    int save_count = (conv_in_len >= 2) ? 2 : conv_in_len;
    for (int ch = 0; ch < vq_hidden; ch++)
        for (int t = 0; t < save_count; t++)
            st->vq_pad[(int64_t)ch * 2 + (2 - save_count + t)] =
                vq_cf[(int64_t)ch * conv_in_len + save_start + t];
    if (save_count < 2) {
        /* Zero-fill the earlier positions if we had fewer than 2 frames */
        for (int ch = 0; ch < vq_hidden; ch++)
            for (int t = 0; t < 2 - save_count; t++)
                st->vq_pad[(int64_t)ch * 2 + t] = 0;
    }
    st->vq_pad_valid = 1;
    free(vq_out);

    /* Pre-conv (512→1024, k=3, causal, pad_left=2) */
    float *pre_conv_out = (float *)aligned_calloc((int64_t)latent_dim * conv_in_len, sizeof(float));
    causal_conv1d(pre_conv_out, vq_cf, sd->pre_conv_weight, sd->pre_conv_bias,
                  vq_hidden, latent_dim, conv_in_len, 3, 1);
    free(vq_cf);

    /* Take only the last new_frames from pre_conv output */
    /* The first pad_frames outputs may have been computed with actual previous context */

    /* === Step 3: Input proj on new frames (1024→512, row-major) === */
    float *hidden = (float *)aligned_malloc((int64_t)new_frames * dec_hidden * sizeof(float));
#ifdef USE_BLAS
    /* Transpose new portion [1024, new_frames] → [new_frames, 1024] */
    float *pre_conv_rm = (float *)aligned_malloc((int64_t)new_frames * latent_dim * sizeof(float));
    for (int f = 0; f < new_frames; f++)
        for (int d = 0; d < latent_dim; d++)
            pre_conv_rm[(int64_t)f * latent_dim + d] = pre_conv_out[(int64_t)d * conv_in_len + pad_frames + f];
    free(pre_conv_out);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                new_frames, dec_hidden, latent_dim, 1.0f,
                pre_conv_rm, latent_dim,
                sd->input_proj_weight, latent_dim,
                0.0f, hidden, dec_hidden);
    free(pre_conv_rm);
    if (sd->input_proj_bias)
        for (int f = 0; f < new_frames; f++)
            for (int o = 0; o < dec_hidden; o++)
                hidden[(int64_t)f * dec_hidden + o] += sd->input_proj_bias[o];
#else
    for (int f = 0; f < new_frames; f++) {
        for (int o = 0; o < dec_hidden; o++) {
            float sum = sd->input_proj_bias ? sd->input_proj_bias[o] : 0;
            for (int i = 0; i < latent_dim; i++)
                sum += sd->input_proj_weight[(int64_t)o * latent_dim + i]
                     * pre_conv_out[(int64_t)i * conv_in_len + pad_frames + f];
            hidden[(int64_t)f * dec_hidden + o] = sum;
        }
    }
    free(pre_conv_out);
#endif

    /* === Step 4: Pre-transformer with KV cache === */
    /* Ensure KV cache is allocated */
    int total_frames = st->kv_len + new_frames;
    if (total_frames > st->kv_alloc) {
        int new_alloc = total_frames + 256; /* grow with headroom */
        for (int l = 0; l < c->dec_num_layers; l++) {
            st->k_cache[l] = (float *)realloc(st->k_cache[l], (int64_t)new_alloc * qkv_dim * sizeof(float));
            st->v_cache[l] = (float *)realloc(st->v_cache[l], (int64_t)new_alloc * qkv_dim * sizeof(float));
        }
        st->kv_alloc = new_alloc;
    }

    float *q = (float *)aligned_malloc((int64_t)new_frames * qkv_dim * sizeof(float));
    float *new_k = (float *)aligned_malloc((int64_t)new_frames * qkv_dim * sizeof(float));
    float *new_v = (float *)aligned_malloc((int64_t)new_frames * qkv_dim * sizeof(float));
    float *x_norm = (float *)aligned_malloc((int64_t)new_frames * dec_hidden * sizeof(float));
    float *attn_out = (float *)aligned_malloc((int64_t)new_frames * qkv_dim * sizeof(float));

    for (int layer = 0; layer < c->dec_num_layers; layer++) {
        qwen_sd_pre_layer_t *l = &sd->pre_layers[layer];

        /* Input RMSNorm (NEON-optimized) */
        qwen_rms_norm(x_norm, hidden, l->attn_norm, new_frames, dec_hidden, eps);

        /* QKV projections for new frames */
#ifdef USE_BLAS
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    new_frames, qkv_dim, dec_hidden, 1.0f,
                    x_norm, dec_hidden, l->attn_q, dec_hidden, 0.0f, q, qkv_dim);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    new_frames, qkv_dim, dec_hidden, 1.0f,
                    x_norm, dec_hidden, l->attn_k, dec_hidden, 0.0f, new_k, qkv_dim);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    new_frames, qkv_dim, dec_hidden, 1.0f,
                    x_norm, dec_hidden, l->attn_v, dec_hidden, 0.0f, new_v, qkv_dim);
#else
        for (int s = 0; s < new_frames; s++) {
            const float *xs = x_norm + s * dec_hidden;
            float *qs = q + s * qkv_dim;
            float *ks = new_k + s * qkv_dim;
            float *vs = new_v + s * qkv_dim;
            for (int o = 0; o < qkv_dim; o++) {
                float sum_q = 0, sum_k = 0, sum_v = 0;
                for (int i = 0; i < dec_hidden; i++) {
                    sum_q += l->attn_q[(int64_t)o * dec_hidden + i] * xs[i];
                    sum_k += l->attn_k[(int64_t)o * dec_hidden + i] * xs[i];
                    sum_v += l->attn_v[(int64_t)o * dec_hidden + i] * xs[i];
                }
                qs[o] = sum_q; ks[o] = sum_k; vs[o] = sum_v;
            }
        }
#endif

        /* NeoX split-half RoPE using absolute positions */
        for (int s = 0; s < new_frames; s++) {
            int abs_pos = st->kv_len + s;
            const float *cos_ptr = sd->rope_cos + abs_pos * half_hd;
            const float *sin_ptr = sd->rope_sin + abs_pos * half_hd;
            for (int h = 0; h < n_heads; h++) {
                float *qh = q + s * qkv_dim + h * head_dim;
                float *kh = new_k + s * qkv_dim + h * head_dim;
                for (int i = 0; i < half_hd; i++) {
                    float q1 = qh[i], q2 = qh[i + half_hd];
                    float k1 = kh[i], k2 = kh[i + half_hd];
                    float co = cos_ptr[i], si = sin_ptr[i];
                    qh[i]           = q1 * co - q2 * si;
                    qh[i + half_hd] = q2 * co + q1 * si;
                    kh[i]           = k1 * co - k2 * si;
                    kh[i + half_hd] = k2 * co + k1 * si;
                }
            }
        }

        /* Append new K, V to cache for this layer */
        memcpy(st->k_cache[layer] + (int64_t)st->kv_len * qkv_dim,
               new_k, (int64_t)new_frames * qkv_dim * sizeof(float));
        memcpy(st->v_cache[layer] + (int64_t)st->kv_len * qkv_dim,
               new_v, (int64_t)new_frames * qkv_dim * sizeof(float));

        /* Sliding window causal attention: Q from new frames, K/V from cache */
        float scale = 1.0f / sqrtf((float)head_dim);
        for (int sq = 0; sq < new_frames; sq++) {
            int abs_sq = st->kv_len + sq;
            float *out = attn_out + sq * qkv_dim;
            memset(out, 0, qkv_dim * sizeof(float));
            int sk_start = (abs_sq - window + 1 > 0) ? abs_sq - window + 1 : 0;
            int sk_end = abs_sq; /* inclusive */

            for (int h = 0; h < n_heads; h++) {
                const float *qh = q + sq * qkv_dim + h * head_dim;
                float *oh = out + h * head_dim;

                int n_keys = sk_end - sk_start + 1;
                float *scores = (float *)alloca(n_keys * sizeof(float));
                float max_score = -1e30f;
                for (int j = 0; j < n_keys; j++) {
                    int sk = sk_start + j;
                    const float *kh = st->k_cache[layer] + (int64_t)sk * qkv_dim + h * head_dim;
                    float dot = 0;
                    for (int d = 0; d < head_dim; d++) dot += qh[d] * kh[d];
                    scores[j] = dot * scale;
                    if (scores[j] > max_score) max_score = scores[j];
                }

                float sum_exp = 0;
                for (int j = 0; j < n_keys; j++) {
                    scores[j] = expf(scores[j] - max_score);
                    sum_exp += scores[j];
                }
                float inv_sum = 1.0f / sum_exp;

                for (int j = 0; j < n_keys; j++) {
                    int sk = sk_start + j;
                    const float *vh = st->v_cache[layer] + (int64_t)sk * qkv_dim + h * head_dim;
                    float w = scores[j] * inv_sum;
                    for (int d = 0; d < head_dim; d++) oh[d] += vh[d] * w;
                }
            }
        }

        /* Output proj + layer_scale + residual */
#ifdef USE_BLAS
        {
            float *oproj = x_norm;
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        new_frames, dec_hidden, qkv_dim, 1.0f,
                        attn_out, qkv_dim, l->attn_o, qkv_dim,
                        0.0f, oproj, dec_hidden);
            for (int s = 0; s < new_frames; s++) {
                float *xs = hidden + s * dec_hidden;
                float *ps = oproj + s * dec_hidden;
                if (l->attn_layer_scale) {
                    for (int o = 0; o < dec_hidden; o++) xs[o] += ps[o] * l->attn_layer_scale[o];
                } else {
                    for (int o = 0; o < dec_hidden; o++) xs[o] += ps[o];
                }
            }
        }
#else
        for (int s = 0; s < new_frames; s++) {
            float *xs = hidden + s * dec_hidden;
            const float *attn = attn_out + s * qkv_dim;
            for (int o = 0; o < dec_hidden; o++) {
                float sum = 0;
                for (int i = 0; i < qkv_dim; i++)
                    sum += l->attn_o[(int64_t)o * qkv_dim + i] * attn[i];
                if (l->attn_layer_scale) sum *= l->attn_layer_scale[o];
                xs[o] += sum;
            }
        }
#endif

        /* Post-attn RMSNorm (NEON-optimized) */
        qwen_rms_norm(x_norm, hidden, l->ffn_norm, new_frames, dec_hidden, eps);

        /* SwiGLU FFN */
#ifdef USE_BLAS
        {
            float *ffn_gate = (float *)aligned_malloc((int64_t)new_frames * dec_inter * sizeof(float));
            float *ffn_up = (float *)aligned_malloc((int64_t)new_frames * dec_inter * sizeof(float));
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        new_frames, dec_inter, dec_hidden, 1.0f,
                        x_norm, dec_hidden, l->ffn_gate, dec_hidden,
                        0.0f, ffn_gate, dec_inter);
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        new_frames, dec_inter, dec_hidden, 1.0f,
                        x_norm, dec_hidden, l->ffn_up, dec_hidden,
                        0.0f, ffn_up, dec_inter);
            for (int64_t i = 0; i < (int64_t)new_frames * dec_inter; i++)
                ffn_gate[i] = (ffn_gate[i] / (1.0f + expf(-ffn_gate[i]))) * ffn_up[i];
            free(ffn_up);
            float *ffn_down_out = (float *)aligned_malloc((int64_t)new_frames * dec_hidden * sizeof(float));
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        new_frames, dec_hidden, dec_inter, 1.0f,
                        ffn_gate, dec_inter, l->ffn_down, dec_inter,
                        0.0f, ffn_down_out, dec_hidden);
            free(ffn_gate);
            for (int s = 0; s < new_frames; s++) {
                float *hs = hidden + s * dec_hidden;
                float *ds = ffn_down_out + s * dec_hidden;
                if (l->ffn_layer_scale) {
                    for (int o = 0; o < dec_hidden; o++) hs[o] += ds[o] * l->ffn_layer_scale[o];
                } else {
                    for (int o = 0; o < dec_hidden; o++) hs[o] += ds[o];
                }
            }
            free(ffn_down_out);
        }
#else
        for (int s = 0; s < new_frames; s++) {
            const float *xs = x_norm + s * dec_hidden;
            float *hs = hidden + s * dec_hidden;
            float gate_up[dec_inter * 2];
            for (int o = 0; o < dec_inter; o++) {
                float sum_g = 0, sum_u = 0;
                for (int i = 0; i < dec_hidden; i++) {
                    sum_g += l->ffn_gate[(int64_t)o * dec_hidden + i] * xs[i];
                    sum_u += l->ffn_up[(int64_t)o * dec_hidden + i] * xs[i];
                }
                gate_up[o] = (sum_g / (1.0f + expf(-sum_g))) * sum_u;
            }
            for (int o = 0; o < dec_hidden; o++) {
                float sum = 0;
                for (int i = 0; i < dec_inter; i++)
                    sum += l->ffn_down[(int64_t)o * dec_inter + i] * gate_up[i];
                if (l->ffn_layer_scale) sum *= l->ffn_layer_scale[o];
                hs[o] += sum;
            }
        }
#endif
    }

    /* Update KV cache length (after all layers processed) */
    st->kv_len += new_frames;

    free(q); free(new_k); free(new_v); free(x_norm); free(attn_out);

    /* === Step 5: Final RMSNorm + Output proj (512→1024) on new frames === */
    if (sd->final_norm_weight) {
        qwen_rms_norm(hidden, hidden, sd->final_norm_weight, new_frames, dec_hidden, eps);
    }

    /* Grow latent cache if needed */
    if (st->latent_frames + new_frames > st->latent_alloc) {
        int new_alloc = st->latent_frames + new_frames + 256;
        st->latent_cache = (float *)realloc(st->latent_cache,
            (int64_t)new_alloc * latent_dim * sizeof(float));
        st->latent_alloc = new_alloc;
    }

    /* Output proj new frames → append to latent cache [row-major: frames × 1024] */
    float *lat_dst = st->latent_cache + (int64_t)st->latent_frames * latent_dim;
#ifdef USE_BLAS
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                new_frames, latent_dim, dec_hidden, 1.0f,
                hidden, dec_hidden,
                sd->output_proj_weight, dec_hidden,
                0.0f, lat_dst, latent_dim);
    if (sd->output_proj_bias)
        for (int f = 0; f < new_frames; f++)
            for (int o = 0; o < latent_dim; o++)
                lat_dst[(int64_t)f * latent_dim + o] += sd->output_proj_bias[o];
#else
    for (int f = 0; f < new_frames; f++) {
        for (int o = 0; o < latent_dim; o++) {
            float sum = sd->output_proj_bias ? sd->output_proj_bias[o] : 0;
            for (int i = 0; i < dec_hidden; i++)
                sum += sd->output_proj_weight[(int64_t)o * dec_hidden + i] * hidden[(int64_t)f * dec_hidden + i];
            lat_dst[(int64_t)f * latent_dim + o] = sum;
        }
    }
#endif
    st->latent_frames += new_frames;
    free(hidden);

    /* === Step 6: Windowed conv decoder === */
    /* Take last (RF + new_frames) from latent cache, or all if fewer */
    int conv_rf = QWEN_SD_STREAM_CONV_RF;
    int window_frames = st->latent_frames;
    int context_frames = 0;
    if (window_frames > conv_rf + new_frames) {
        context_frames = conv_rf;
        window_frames = conv_rf + new_frames;
    } else {
        context_frames = window_frames - new_frames;
    }
    int window_start = st->latent_frames - window_frames;

    /* Transpose window to channel-first [1024, window_frames] for conv decoder */
    float *signal = (float *)aligned_malloc((int64_t)latent_dim * window_frames * sizeof(float));
    const float *lat_src = st->latent_cache + (int64_t)window_start * latent_dim;
    for (int f = 0; f < window_frames; f++)
        for (int d = 0; d < latent_dim; d++)
            signal[(int64_t)d * window_frames + f] = lat_src[(int64_t)f * latent_dim + d];

    /* Run conv decoder (ConvNeXt + initial conv + upsample blocks + final conv) */
    float *full_audio = NULL;
    int full_samples = 0;
    int ret = conv_decoder_forward(ctx, signal, latent_dim, window_frames,
                                    &full_audio, &full_samples);
    if (ret != 0) return ret;

    /* Extract only the new audio (skip context portion) */
    /* Audio is exactly 1920 samples per latent frame */
    int context_samples = context_frames * 1920;
    int new_samples = full_samples - context_samples;
    if (new_samples <= 0) {
        free(full_audio);
        *audio_out = NULL;
        *n_samples = 0;
        return 0;
    }

    float *new_audio = (float *)aligned_malloc(new_samples * sizeof(float));
    memcpy(new_audio, full_audio + context_samples, new_samples * sizeof(float));
    free(full_audio);

    st->frames_decoded += new_frames;
    st->samples_produced += new_samples;

    *audio_out = new_audio;
    *n_samples = new_samples;
    return 0;
}
