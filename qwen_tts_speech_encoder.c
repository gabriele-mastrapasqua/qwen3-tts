/*
 * qwen_tts_speech_encoder.c - Speech Tokenizer Encoder
 * Converts 24kHz audio waveform → 16 codebook codes per frame (12.5 Hz)
 *
 * Architecture (inverse of decoder):
 * 1. Conv encoder: initial conv → 4× (ResBlock + ELU + downsample) → ELU → final conv
 *    - Downsample rates: [4, 5, 6, 8] (channels: 1→64→128→256→512→1024→512)
 *    - All convolutions are causal (left-padded)
 *    - ELU activation (NOT Snake)
 * 2. Encoder transformer (8 layers, hidden=512, heads=8, window=250)
 *    - LayerNorm (with bias), NOT RMSNorm
 *    - GELU MLP (fc1→GELU→fc2), NOT SwiGLU
 *    - Causal attention with sliding window 250
 *    - NeoX split-half RoPE
 * 3. Downsample conv (stride=2, k=4, no bias)
 * 4. RVQ quantization (16 codebooks: 1 semantic + 15 acoustic)
 *    - Input projection (512→256)
 *    - Nearest-neighbor search in codebook (L2 distance)
 *    - Residual subtraction for each codebook level
 *
 * Weights from speech_tokenizer/model.safetensors (encoder.* prefix)
 * Quantizer codebooks reused from decoder (already loaded)
 */

#include "qwen_tts.h"
#include "qwen_tts_safetensors.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef USE_BLAS
#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif
#endif

/* ========================================================================
 * Helper: get f32 tensor from speech safetensors
 * ======================================================================== */

static const float *enc_get_f32(void *ms, const char *name) {
    safetensors_file_t *sf = NULL;
    const safetensor_t *t = multi_safetensors_find((multi_safetensors_t *)ms, name, &sf);
    return (t && sf) ? (const float *)safetensors_data(sf, t) : NULL;
}

/* ========================================================================
 * Conv helpers (same as decoder but extracted here)
 * ======================================================================== */

/* Causal Conv1d output length */
static int enc_conv1d_out_len(int in_len, int kernel, int stride) {
    int pad_left = kernel - stride;  /* causal: all padding on left */
    return (in_len + pad_left - kernel) / stride + 1;
}

/* ELU activation: x if x > 0, else exp(x) - 1 */
static void elu_activation(float *data, int n) {
    for (int i = 0; i < n; i++)
        if (data[i] < 0) data[i] = expf(data[i]) - 1.0f;
}

/* Causal Conv1d: [out_ch, in_ch, kernel], pad_left = kernel - stride */
static void enc_causal_conv1d(float *out, const float *in,
                               const float *weight, const float *bias,
                               int in_ch, int out_ch, int in_len,
                               int kernel, int stride, int dilation) {
    int pad_left = (kernel - 1) * dilation;
    if (stride > 1) pad_left = kernel - stride; /* for strided convs */
    int out_len = (in_len + pad_left - (kernel - 1) * dilation - 1) / stride + 1;

#ifdef USE_BLAS
    if (kernel == 1 && stride == 1) {
        /* k=1: direct matmul */
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    out_ch, in_len, in_ch, 1.0f,
                    weight, in_ch, in, in_len, 0.0f, out, out_len);
        if (bias)
            for (int oc = 0; oc < out_ch; oc++)
                for (int t = 0; t < out_len; t++)
                    out[(int64_t)oc * out_len + t] += bias[oc];
        return;
    }
#endif

    memset(out, 0, (int64_t)out_ch * out_len * sizeof(float));
    for (int oc = 0; oc < out_ch; oc++) {
        float b = bias ? bias[oc] : 0;
        for (int t = 0; t < out_len; t++) {
            float sum = b;
            for (int ic = 0; ic < in_ch; ic++) {
                for (int k = 0; k < kernel; k++) {
                    int in_pos = t * stride - pad_left + k * dilation;
                    if (in_pos >= 0 && in_pos < in_len)
                        sum += weight[((int64_t)oc * in_ch + ic) * kernel + k]
                             * in[(int64_t)ic * in_len + in_pos];
                }
            }
            out[(int64_t)oc * out_len + t] = sum;
        }
    }
}

/* ========================================================================
 * Encoder Conv Layer Weights
 * ======================================================================== */

typedef struct {
    const float *weight;
    const float *bias;
} enc_conv_t;

typedef struct {
    /* ResBlock: block.1 (conv k=3) + block.3 (conv k=1) */
    enc_conv_t conv1;  /* dim→dim/2, k=3 */
    enc_conv_t conv2;  /* dim/2→dim, k=1 */
} enc_resblock_t;

typedef struct {
    /* Conv encoder layers */
    enc_conv_t initial_conv;     /* 1→64, k=7 */
    enc_resblock_t resblocks[4]; /* dim/2→dim, for each stage */
    enc_conv_t stride_convs[4];  /* 64→128 k=8/s=4, 128→256 k=10/s=5, etc. */
    enc_conv_t final_conv;       /* 1024→512, k=3 */

    /* Encoder transformer (8 layers) */
    struct {
        const float *attn_norm_w, *attn_norm_b;    /* LayerNorm */
        const float *attn_q, *attn_k, *attn_v, *attn_o; /* [512, 512] */
        const float *attn_layer_scale;              /* [512] */
        const float *ffn_norm_w, *ffn_norm_b;       /* LayerNorm */
        const float *ffn_fc1, *ffn_fc2;            /* [2048,512] and [512,2048] */
        const float *ffn_layer_scale;               /* [512] */
    } transformer[8];

    /* Downsample conv */
    const float *downsample_weight; /* [512, 512, 4], stride=2, no bias */

    /* RVQ input projections (encoder-side, different from decoder's output projections) */
    const float *rvq_semantic_input_proj;  /* [256, 512, 1] — for codebook 0 */
    const float *rvq_acoustic_input_proj;  /* [256, 512, 1] — for codebooks 1-15 */

    /* RoPE cache */
    float *rope_cos; /* [max_pos, head_dim/2] */
    float *rope_sin;

    int loaded;
} qwen_speech_encoder_t;

/* ========================================================================
 * Weight Loading
 * ======================================================================== */

static qwen_speech_encoder_t g_encoder;

int qwen_speech_encoder_load(qwen_tts_ctx_t *ctx) {
    void *ms = ctx->speech_safetensors;
    qwen_speech_encoder_t *enc = &g_encoder;
    memset(enc, 0, sizeof(*enc));
    int ok = 0;

    /* Initial conv: encoder.encoder.layers.0 */
    enc->initial_conv.weight = enc_get_f32(ms, "encoder.encoder.layers.0.conv.weight");
    enc->initial_conv.bias = enc_get_f32(ms, "encoder.encoder.layers.0.conv.bias");
    if (enc->initial_conv.weight) ok++;

    /* 4 stages: resblock at layers [1,4,7,10], stride conv at layers [3,6,9,12] */
    int res_layers[] = {1, 4, 7, 10};
    int stride_layers[] = {3, 6, 9, 12};

    for (int i = 0; i < 4; i++) {
        char buf[128];
        /* ResBlock */
        snprintf(buf, sizeof(buf), "encoder.encoder.layers.%d.block.1.conv.weight", res_layers[i]);
        enc->resblocks[i].conv1.weight = enc_get_f32(ms, buf);
        snprintf(buf, sizeof(buf), "encoder.encoder.layers.%d.block.1.conv.bias", res_layers[i]);
        enc->resblocks[i].conv1.bias = enc_get_f32(ms, buf);
        snprintf(buf, sizeof(buf), "encoder.encoder.layers.%d.block.3.conv.weight", res_layers[i]);
        enc->resblocks[i].conv2.weight = enc_get_f32(ms, buf);
        snprintf(buf, sizeof(buf), "encoder.encoder.layers.%d.block.3.conv.bias", res_layers[i]);
        enc->resblocks[i].conv2.bias = enc_get_f32(ms, buf);
        if (enc->resblocks[i].conv1.weight) ok++;

        /* Stride conv */
        snprintf(buf, sizeof(buf), "encoder.encoder.layers.%d.conv.weight", stride_layers[i]);
        enc->stride_convs[i].weight = enc_get_f32(ms, buf);
        snprintf(buf, sizeof(buf), "encoder.encoder.layers.%d.conv.bias", stride_layers[i]);
        enc->stride_convs[i].bias = enc_get_f32(ms, buf);
        if (enc->stride_convs[i].weight) ok++;
    }

    /* Final conv: encoder.encoder.layers.14 */
    enc->final_conv.weight = enc_get_f32(ms, "encoder.encoder.layers.14.conv.weight");
    enc->final_conv.bias = enc_get_f32(ms, "encoder.encoder.layers.14.conv.bias");
    if (enc->final_conv.weight) ok++;

    /* Encoder transformer (8 layers) */
    for (int l = 0; l < 8; l++) {
        char buf[128];
        #define ENC_LOAD(field, suffix) do { \
            snprintf(buf, sizeof(buf), "encoder.encoder_transformer.layers.%d." suffix, l); \
            enc->transformer[l].field = enc_get_f32(ms, buf); \
        } while(0)

        ENC_LOAD(attn_norm_w, "input_layernorm.weight");
        ENC_LOAD(attn_norm_b, "input_layernorm.bias");
        ENC_LOAD(attn_q, "self_attn.q_proj.weight");
        ENC_LOAD(attn_k, "self_attn.k_proj.weight");
        ENC_LOAD(attn_v, "self_attn.v_proj.weight");
        ENC_LOAD(attn_o, "self_attn.o_proj.weight");
        ENC_LOAD(attn_layer_scale, "self_attn_layer_scale.scale");
        ENC_LOAD(ffn_norm_w, "post_attention_layernorm.weight");
        ENC_LOAD(ffn_norm_b, "post_attention_layernorm.bias");
        ENC_LOAD(ffn_fc1, "mlp.fc1.weight");
        ENC_LOAD(ffn_fc2, "mlp.fc2.weight");
        ENC_LOAD(ffn_layer_scale, "mlp_layer_scale.scale");
        #undef ENC_LOAD

        if (enc->transformer[l].attn_q) ok++;
    }

    /* Downsample */
    enc->downsample_weight = enc_get_f32(ms, "encoder.downsample.conv.weight");
    if (enc->downsample_weight) ok++;

    /* RVQ input projections (encoder-specific) */
    enc->rvq_semantic_input_proj = enc_get_f32(ms,
        "encoder.quantizer.semantic_residual_vector_quantizer.input_proj.weight");
    enc->rvq_acoustic_input_proj = enc_get_f32(ms,
        "encoder.quantizer.acoustic_residual_vector_quantizer.input_proj.weight");
    if (enc->rvq_semantic_input_proj) ok++;
    if (enc->rvq_acoustic_input_proj) ok++;

    /* RoPE cache: theta=10000.0, head_dim=64, max_pos=8000 */
    int half_dim = 32; /* 64 / 2 */
    enc->rope_cos = (float *)malloc(8000 * half_dim * sizeof(float));
    enc->rope_sin = (float *)malloc(8000 * half_dim * sizeof(float));
    for (int pos = 0; pos < 8000; pos++) {
        for (int i = 0; i < half_dim; i++) {
            float angle = (float)pos / powf(10000.0f, (float)(2 * i) / 64.0f);
            enc->rope_cos[pos * half_dim + i] = cosf(angle);
            enc->rope_sin[pos * half_dim + i] = sinf(angle);
        }
    }

    enc->loaded = 1;
    if (!ctx->silent)
        fprintf(stderr, "  Speech encoder: %d/15 components loaded\n", ok);

    return (ok >= 12) ? 0 : -1; /* need conv encoder + transformer + downsample */
}

/* ========================================================================
 * Encode: audio samples → codec codes
 * ======================================================================== */

int qwen_speech_encoder_encode(qwen_tts_ctx_t *ctx, const float *audio, int n_samples,
                                int **codes_out, int *n_frames_out) {
    qwen_speech_encoder_t *enc = &g_encoder;
    qwen_speech_decoder_t *sd = &ctx->speech_dec;

    if (!enc->loaded) {
        fprintf(stderr, "ERROR: Speech encoder not loaded\n");
        return -1;
    }

    int cb_dim = QWEN_TTS_CODEBOOK_DIM; /* 256 */

    /* === Stage 1: Conv Encoder === */
    /* Input: [1, n_samples] channel-first */
    int cur_ch = 1, cur_len = n_samples;
    float *signal = (float *)malloc((int64_t)cur_ch * cur_len * sizeof(float));
    memcpy(signal, audio, cur_len * sizeof(float));

    if (ctx->debug)
        fprintf(stderr, "[ENC] Input: %d samples\n", n_samples);

    /* Initial conv: 1→64, k=7, s=1 */
    {
        int out_ch = 64, kernel = 7;
        int out_len = enc_conv1d_out_len(cur_len, kernel, 1);
        float *out = (float *)calloc((int64_t)out_ch * out_len, sizeof(float));
        enc_causal_conv1d(out, signal, enc->initial_conv.weight, enc->initial_conv.bias,
                          cur_ch, out_ch, cur_len, kernel, 1, 1);
        free(signal); signal = out; cur_ch = out_ch; cur_len = out_len;
    }

    if (ctx->debug)
        fprintf(stderr, "[ENC] After initial conv: ch=%d, len=%d\n", cur_ch, cur_len);

    /* 4 stages: ResBlock → ELU → stride conv */
    int out_channels[] = {128, 256, 512, 1024};
    int strides[] = {4, 5, 6, 8};
    int kernels[] = {8, 10, 12, 16};

    for (int stage = 0; stage < 4; stage++) {
        /* ResBlock: ELU → conv1(ch→ch/2, k=3) → ELU → conv2(ch/2→ch, k=1) + residual */
        {
            int half_ch = cur_ch / 2;
            float *residual = (float *)malloc((int64_t)cur_ch * cur_len * sizeof(float));
            memcpy(residual, signal, (int64_t)cur_ch * cur_len * sizeof(float));

            /* ELU */
            elu_activation(signal, (int64_t)cur_ch * cur_len);

            /* Conv1: ch→ch/2, k=3 */
            float *c1 = (float *)calloc((int64_t)half_ch * cur_len, sizeof(float));
            enc_causal_conv1d(c1, signal, enc->resblocks[stage].conv1.weight,
                              enc->resblocks[stage].conv1.bias,
                              cur_ch, half_ch, cur_len, 3, 1, 1);

            /* ELU */
            elu_activation(c1, (int64_t)half_ch * cur_len);

            /* Conv2: ch/2→ch, k=1 */
            float *c2 = (float *)calloc((int64_t)cur_ch * cur_len, sizeof(float));
            enc_causal_conv1d(c2, c1, enc->resblocks[stage].conv2.weight,
                              enc->resblocks[stage].conv2.bias,
                              half_ch, cur_ch, cur_len, 1, 1, 1);
            free(c1);

            /* Residual */
            for (int64_t i = 0; i < (int64_t)cur_ch * cur_len; i++)
                signal[i] = residual[i] + c2[i];
            free(residual); free(c2);
        }

        /* ELU before stride conv */
        elu_activation(signal, (int64_t)cur_ch * cur_len);

        /* Stride conv */
        int out_ch = out_channels[stage];
        int out_len = enc_conv1d_out_len(cur_len, kernels[stage], strides[stage]);
        float *out = (float *)calloc((int64_t)out_ch * out_len, sizeof(float));
        enc_causal_conv1d(out, signal, enc->stride_convs[stage].weight,
                          enc->stride_convs[stage].bias,
                          cur_ch, out_ch, cur_len, kernels[stage], strides[stage], 1);
        free(signal); signal = out; cur_ch = out_ch; cur_len = out_len;

        if (ctx->debug)
            fprintf(stderr, "[ENC] After stage %d: ch=%d, len=%d\n", stage, cur_ch, cur_len);
    }

    /* ELU */
    elu_activation(signal, (int64_t)cur_ch * cur_len);

    /* Final conv: 1024→512, k=3 */
    {
        int out_ch = 512;
        int out_len = enc_conv1d_out_len(cur_len, 3, 1);
        float *out = (float *)calloc((int64_t)out_ch * out_len, sizeof(float));
        enc_causal_conv1d(out, signal, enc->final_conv.weight, enc->final_conv.bias,
                          cur_ch, out_ch, cur_len, 3, 1, 1);
        free(signal); signal = out; cur_ch = out_ch; cur_len = out_len;
    }

    if (ctx->debug)
        fprintf(stderr, "[ENC] After conv encoder: ch=%d, len=%d (%.1f Hz)\n",
                cur_ch, cur_len, (float)cur_len / ((float)n_samples / 24000.0f));

    /* === Stage 2: Encoder Transformer === */
    /* Transpose signal from channel-first [512, len] to row-major [len, 512] */
    int n_seq = cur_len;
    int hidden = 512;
    float *h_buf = (float *)malloc((int64_t)n_seq * hidden * sizeof(float));
    for (int f = 0; f < n_seq; f++)
        for (int d = 0; d < hidden; d++)
            h_buf[(int64_t)f * hidden + d] = signal[(int64_t)d * n_seq + f];
    free(signal);

    int n_heads = 8, head_dim = 64, inter = 2048;
    int qkv_dim = n_heads * head_dim;
    int window = 250;
    int half_hd = head_dim / 2;
    float eps = 1e-5f;

    float *q = (float *)malloc((int64_t)n_seq * qkv_dim * sizeof(float));
    float *kk = (float *)malloc((int64_t)n_seq * qkv_dim * sizeof(float));
    float *vv = (float *)malloc((int64_t)n_seq * qkv_dim * sizeof(float));
    float *x_norm = (float *)malloc((int64_t)n_seq * hidden * sizeof(float));
    float *attn_out = (float *)malloc((int64_t)n_seq * qkv_dim * sizeof(float));

    for (int layer = 0; layer < 8; layer++) {
        /* LayerNorm (with bias) */
        for (int s = 0; s < n_seq; s++) {
            const float *xs = h_buf + s * hidden;
            float *xn = x_norm + s * hidden;
            float mean = 0, var = 0;
            for (int i = 0; i < hidden; i++) mean += xs[i];
            mean /= hidden;
            for (int i = 0; i < hidden; i++) {
                float d = xs[i] - mean;
                var += d * d;
            }
            float inv_std = 1.0f / sqrtf(var / hidden + eps);
            for (int i = 0; i < hidden; i++)
                xn[i] = (xs[i] - mean) * inv_std * enc->transformer[layer].attn_norm_w[i]
                         + enc->transformer[layer].attn_norm_b[i];
        }

        /* QKV projections */
#ifdef USE_BLAS
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    n_seq, qkv_dim, hidden, 1.0f,
                    x_norm, hidden, enc->transformer[layer].attn_q, hidden, 0.0f, q, qkv_dim);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    n_seq, qkv_dim, hidden, 1.0f,
                    x_norm, hidden, enc->transformer[layer].attn_k, hidden, 0.0f, kk, qkv_dim);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    n_seq, qkv_dim, hidden, 1.0f,
                    x_norm, hidden, enc->transformer[layer].attn_v, hidden, 0.0f, vv, qkv_dim);
#else
        for (int s = 0; s < n_seq; s++) {
            const float *xs = x_norm + s * hidden;
            for (int o = 0; o < qkv_dim; o++) {
                float sq = 0, sk = 0, sv = 0;
                for (int i = 0; i < hidden; i++) {
                    sq += enc->transformer[layer].attn_q[(int64_t)o * hidden + i] * xs[i];
                    sk += enc->transformer[layer].attn_k[(int64_t)o * hidden + i] * xs[i];
                    sv += enc->transformer[layer].attn_v[(int64_t)o * hidden + i] * xs[i];
                }
                q[s * qkv_dim + o] = sq;
                kk[s * qkv_dim + o] = sk;
                vv[s * qkv_dim + o] = sv;
            }
        }
#endif

        /* NeoX split-half RoPE */
        for (int s = 0; s < n_seq; s++) {
            const float *cos_ptr = enc->rope_cos + s * half_hd;
            const float *sin_ptr = enc->rope_sin + s * half_hd;
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

        /* Sliding window causal attention */
        float scale = 1.0f / sqrtf((float)head_dim);
        for (int sq = 0; sq < n_seq; sq++) {
            float *out = attn_out + sq * qkv_dim;
            memset(out, 0, qkv_dim * sizeof(float));
            int sk_start = (sq - window + 1 > 0) ? sq - window + 1 : 0;

            for (int h = 0; h < n_heads; h++) {
                const float *qh = q + sq * qkv_dim + h * head_dim;
                float *oh = out + h * head_dim;
                int n_keys = sq - sk_start + 1;

                /* Stack-allocate scores (bounded by window size) */
                float *scores = (float *)alloca(n_keys * sizeof(float));
                float max_score = -1e30f;
                for (int j = 0; j < n_keys; j++) {
                    int sk = sk_start + j;
                    const float *kh = kk + sk * qkv_dim + h * head_dim;
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
                    const float *vh = vv + sk * qkv_dim + h * head_dim;
                    float w = scores[j] * inv_sum;
                    for (int d = 0; d < head_dim; d++) oh[d] += vh[d] * w;
                }
            }
        }

        /* Output proj + layer_scale + residual */
#ifdef USE_BLAS
        {
            float *oproj = x_norm; /* reuse as temp */
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        n_seq, hidden, qkv_dim, 1.0f,
                        attn_out, qkv_dim, enc->transformer[layer].attn_o, qkv_dim,
                        0.0f, oproj, hidden);
            for (int s = 0; s < n_seq; s++) {
                float *xs = h_buf + s * hidden;
                float *ps = oproj + s * hidden;
                const float *ls = enc->transformer[layer].attn_layer_scale;
                if (ls) {
                    for (int o = 0; o < hidden; o++) xs[o] += ps[o] * ls[o];
                } else {
                    for (int o = 0; o < hidden; o++) xs[o] += ps[o];
                }
            }
        }
#else
        for (int s = 0; s < n_seq; s++) {
            float *xs = h_buf + s * hidden;
            const float *attn = attn_out + s * qkv_dim;
            for (int o = 0; o < hidden; o++) {
                float sum = 0;
                for (int i = 0; i < qkv_dim; i++)
                    sum += enc->transformer[layer].attn_o[(int64_t)o * qkv_dim + i] * attn[i];
                const float *ls = enc->transformer[layer].attn_layer_scale;
                if (ls) sum *= ls[o];
                xs[o] += sum;
            }
        }
#endif

        /* Post-attn LayerNorm */
        for (int s = 0; s < n_seq; s++) {
            const float *xs = h_buf + s * hidden;
            float *xn = x_norm + s * hidden;
            float mean = 0, var = 0;
            for (int i = 0; i < hidden; i++) mean += xs[i];
            mean /= hidden;
            for (int i = 0; i < hidden; i++) {
                float d = xs[i] - mean;
                var += d * d;
            }
            float inv_std = 1.0f / sqrtf(var / hidden + eps);
            for (int i = 0; i < hidden; i++)
                xn[i] = (xs[i] - mean) * inv_std * enc->transformer[layer].ffn_norm_w[i]
                         + enc->transformer[layer].ffn_norm_b[i];
        }

        /* GELU MLP: fc1→GELU→fc2 + layer_scale + residual */
#ifdef USE_BLAS
        {
            float *fc1_out = (float *)malloc((int64_t)n_seq * inter * sizeof(float));
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        n_seq, inter, hidden, 1.0f,
                        x_norm, hidden, enc->transformer[layer].ffn_fc1, hidden,
                        0.0f, fc1_out, inter);
            /* Exact GELU */
            for (int64_t i = 0; i < (int64_t)n_seq * inter; i++)
                fc1_out[i] = 0.5f * fc1_out[i] * (1.0f + erff(fc1_out[i] * 0.7071067811865476f));

            float *fc2_out = (float *)malloc((int64_t)n_seq * hidden * sizeof(float));
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        n_seq, hidden, inter, 1.0f,
                        fc1_out, inter, enc->transformer[layer].ffn_fc2, inter,
                        0.0f, fc2_out, hidden);
            free(fc1_out);

            for (int s = 0; s < n_seq; s++) {
                float *xs = h_buf + s * hidden;
                float *fs = fc2_out + s * hidden;
                const float *ls = enc->transformer[layer].ffn_layer_scale;
                if (ls) {
                    for (int o = 0; o < hidden; o++) xs[o] += fs[o] * ls[o];
                } else {
                    for (int o = 0; o < hidden; o++) xs[o] += fs[o];
                }
            }
            free(fc2_out);
        }
#else
        for (int s = 0; s < n_seq; s++) {
            const float *xs = x_norm + s * hidden;
            float *hs = h_buf + s * hidden;
            float *fc1 = (float *)malloc(inter * sizeof(float));
            for (int o = 0; o < inter; o++) {
                float sum = 0;
                for (int i = 0; i < hidden; i++)
                    sum += enc->transformer[layer].ffn_fc1[(int64_t)o * hidden + i] * xs[i];
                fc1[o] = 0.5f * sum * (1.0f + erff(sum * 0.7071067811865476f));
            }
            for (int o = 0; o < hidden; o++) {
                float sum = 0;
                for (int i = 0; i < inter; i++)
                    sum += enc->transformer[layer].ffn_fc2[(int64_t)o * inter + i] * fc1[i];
                const float *ls = enc->transformer[layer].ffn_layer_scale;
                if (ls) sum *= ls[o];
                hs[o] += sum;
            }
            free(fc1);
        }
#endif

        if (ctx->debug && layer == 0)
            fprintf(stderr, "[ENC] Transformer L0 out [:5]: [%.6f, %.6f, %.6f, %.6f, %.6f]\n",
                    h_buf[0], h_buf[1], h_buf[2], h_buf[3], h_buf[4]);
    }

    free(q); free(kk); free(vv); free(x_norm); free(attn_out);

    /* Transpose back to channel-first [512, n_seq] */
    signal = (float *)malloc((int64_t)hidden * n_seq * sizeof(float));
    for (int f = 0; f < n_seq; f++)
        for (int d = 0; d < hidden; d++)
            signal[(int64_t)d * n_seq + f] = h_buf[(int64_t)f * hidden + d];
    free(h_buf);

    cur_ch = hidden; cur_len = n_seq;

    if (ctx->debug)
        fprintf(stderr, "[ENC] After transformer: ch=%d, len=%d\n", cur_ch, cur_len);

    /* === Stage 3: Downsample conv (stride=2, k=4, no bias) === */
    {
        int out_len = enc_conv1d_out_len(cur_len, 4, 2);
        float *out = (float *)calloc((int64_t)cur_ch * out_len, sizeof(float));
        enc_causal_conv1d(out, signal, enc->downsample_weight, NULL,
                          cur_ch, cur_ch, cur_len, 4, 2, 1);
        free(signal); signal = out; cur_len = out_len;
    }

    int n_frames = cur_len;
    if (ctx->debug)
        fprintf(stderr, "[ENC] After downsample: len=%d (%.1f Hz)\n",
                n_frames, (float)n_frames / ((float)n_samples / 24000.0f));

    /* === Stage 4: RVQ Quantization === */
    /* Project from hidden (512) to codebook dim (256) */
    /* Transpose to row-major [n_frames, 512] for projection */
    float *enc_hidden = (float *)malloc((int64_t)n_frames * hidden * sizeof(float));
    for (int f = 0; f < n_frames; f++)
        for (int d = 0; d < hidden; d++)
            enc_hidden[(int64_t)f * hidden + d] = signal[(int64_t)d * n_frames + f];
    free(signal);

    /* Allocate output codes */
    int *codes = (int *)malloc((int64_t)n_frames * 16 * sizeof(int));
    memset(codes, 0, (int64_t)n_frames * 16 * sizeof(int));

    /* Codebook 0 (semantic / rvq_first) */
    {
        /* Project: [n_frames, 512] → [n_frames, 256] using encoder's semantic input_proj */
        float *projected = (float *)malloc((int64_t)n_frames * cb_dim * sizeof(float));
        const float *proj_w = enc->rvq_semantic_input_proj; /* [256, 512, 1] = [256, 512] */
#ifdef USE_BLAS
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    n_frames, cb_dim, hidden, 1.0f,
                    enc_hidden, hidden, proj_w, hidden,
                    0.0f, projected, cb_dim);
#else
        for (int f = 0; f < n_frames; f++)
            for (int o = 0; o < cb_dim; o++) {
                float sum = 0;
                for (int i = 0; i < hidden; i++)
                    sum += proj_w[(int64_t)o * hidden + i] * enc_hidden[(int64_t)f * hidden + i];
                projected[(int64_t)f * cb_dim + o] = sum;
            }
#endif

        /* Nearest neighbor in codebook 0 */
        const float *codebook = sd->codebook[0]; /* [2048, 256] */
        int cb_size = ctx->config.codebook_size;

        /* Precompute ||e||² for each codebook entry */
        float *cb_norm2 = (float *)malloc(cb_size * sizeof(float));
        for (int e = 0; e < cb_size; e++) {
            float sum = 0;
            for (int d = 0; d < cb_dim; d++) {
                float v = codebook[(int64_t)e * cb_dim + d];
                sum += v * v;
            }
            cb_norm2[e] = sum;
        }

        for (int f = 0; f < n_frames; f++) {
            const float *x = projected + (int64_t)f * cb_dim;
            /* Compute x_norm2 */
            float x_norm2 = 0;
            for (int d = 0; d < cb_dim; d++) x_norm2 += x[d] * x[d];

            /* L2 = x_norm2 + e_norm2 - 2*x·e → find min */
            int best_idx = 0;
            float best_dist = 1e30f;
            for (int e = 0; e < cb_size; e++) {
                float dot = 0;
                for (int d = 0; d < cb_dim; d++)
                    dot += x[d] * codebook[(int64_t)e * cb_dim + d];
                float dist = x_norm2 + cb_norm2[e] - 2.0f * dot;
                if (dist < best_dist) { best_dist = dist; best_idx = e; }
            }
            codes[f * 16] = best_idx;

            /* Subtract quantized from projected for residual */
            for (int d = 0; d < cb_dim; d++)
                projected[(int64_t)f * cb_dim + d] -= codebook[(int64_t)best_idx * cb_dim + d];
        }
        free(cb_norm2);

        /* Codebooks 1-15 (acoustic / rvq_rest) */
        /* Re-project using encoder's acoustic input_proj for the residual */
        float *residual = (float *)malloc((int64_t)n_frames * cb_dim * sizeof(float));
        const float *rest_proj_w = enc->rvq_acoustic_input_proj;
#ifdef USE_BLAS
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    n_frames, cb_dim, hidden, 1.0f,
                    enc_hidden, hidden, rest_proj_w, hidden,
                    0.0f, residual, cb_dim);
#else
        for (int f = 0; f < n_frames; f++)
            for (int o = 0; o < cb_dim; o++) {
                float sum = 0;
                for (int i = 0; i < hidden; i++)
                    sum += rest_proj_w[(int64_t)o * hidden + i] * enc_hidden[(int64_t)f * hidden + i];
                residual[(int64_t)f * cb_dim + o] = sum;
            }
#endif

        /* Subtract the semantic codebook's quantized contribution from residual */
        /* Wait — actually the rvq_first and rvq_rest operate on independent projections.
         * The residual for rvq_rest starts from its own projection, and each codebook
         * within rvq_rest does residual subtraction from the previous level. */

        for (int k = 1; k < 16; k++) {
            const float *cb = sd->codebook[k];

            /* Precompute cb norms */
            float *cn2 = (float *)malloc(cb_size * sizeof(float));
            for (int e = 0; e < cb_size; e++) {
                float sum = 0;
                for (int d = 0; d < cb_dim; d++) {
                    float v = cb[(int64_t)e * cb_dim + d];
                    sum += v * v;
                }
                cn2[e] = sum;
            }

            for (int f = 0; f < n_frames; f++) {
                float *x = residual + (int64_t)f * cb_dim;
                float x_n2 = 0;
                for (int d = 0; d < cb_dim; d++) x_n2 += x[d] * x[d];

                int best = 0;
                float best_d = 1e30f;
                for (int e = 0; e < cb_size; e++) {
                    float dot = 0;
                    for (int d = 0; d < cb_dim; d++)
                        dot += x[d] * cb[(int64_t)e * cb_dim + d];
                    float dist = x_n2 + cn2[e] - 2.0f * dot;
                    if (dist < best_d) { best_d = dist; best = e; }
                }
                codes[f * 16 + k] = best;

                /* Subtract for next level */
                for (int d = 0; d < cb_dim; d++)
                    x[d] -= cb[(int64_t)best * cb_dim + d];
            }
            free(cn2);
        }

        free(residual);
        free(projected);
    }

    free(enc_hidden);

    if (!ctx->silent)
        fprintf(stderr, "  Speech encoder: %d samples → %d frames (%d codebooks)\n",
                n_samples, n_frames, 16);

    if (ctx->debug && n_frames > 0) {
        fprintf(stderr, "[ENC] ref_codes frame 0: [%d", codes[0]);
        for (int k = 1; k < 16; k++) fprintf(stderr, ", %d", codes[k]);
        fprintf(stderr, "]\n");
        if (n_frames > 1) {
            fprintf(stderr, "[ENC] ref_codes frame 1: [%d", codes[16]);
            for (int k = 1; k < 16; k++) fprintf(stderr, ", %d", codes[16 + k]);
            fprintf(stderr, "]\n");
        }
    }

    *codes_out = codes;
    *n_frames_out = n_frames;
    return 0;
}
