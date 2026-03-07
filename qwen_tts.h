/*
 * qwen_tts.h - Qwen3-TTS Pure C Inference Engine
 *
 * Supports Qwen3-TTS-12Hz-0.6B-CustomVoice and Qwen3-TTS-12Hz-1.7B-CustomVoice models.
 */

#ifndef QWEN_TTS_H
#define QWEN_TTS_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <pthread.h>

/* ========================================================================
 * Constants
 * ======================================================================== */

#define QWEN_TTS_SAMPLE_RATE         24000
#define QWEN_TTS_FRAME_RATE          12.5
#define QWEN_TTS_HOP_SAMPLES         1920  /* 24000 / 12.5 */

/* Model size limits */
#define QWEN_TTS_MAX_TALKER_LAYERS   28
#define QWEN_TTS_MAX_CP_LAYERS       5
#define QWEN_TTS_MAX_DECODER_LAYERS  8

/* Vocabularies */
#define QWEN_TTS_TEXT_VOCAB_SIZE     151936
#define QWEN_TTS_CODEC_VOCAB_SIZE    3072
#define QWEN_TTS_CODEBOOK_SIZE       2048
#define QWEN_TTS_NUM_CODEBOOKS       16
#define QWEN_TTS_CODEBOOK_DIM        256

/* Special token IDs - Text side */
#define QWEN_TTS_TOK_IM_START        151644
#define QWEN_TTS_TOK_IM_END          151645
#define QWEN_TTS_TOK_ENDOFTEXT       151643
#define QWEN_TTS_TTS_BOS             151672
#define QWEN_TTS_TTS_EOS             151673
#define QWEN_TTS_TTS_PAD             151671

/* Special token IDs - Codec side */
#define QWEN_TTS_CODEC_PAD           2148
#define QWEN_TTS_CODEC_BOS           2149
#define QWEN_TTS_CODEC_EOS           2150
#define QWEN_TTS_CODEC_THINK         2154
#define QWEN_TTS_CODEC_NO_THINK      2155
#define QWEN_TTS_CODEC_THINK_BOS     2156
#define QWEN_TTS_CODEC_THINK_EOS     2157

/* Language IDs (codec vocab) */
#define QWEN_TTS_LANG_CHINESE        2055
#define QWEN_TTS_LANG_ENGLISH        2050
#define QWEN_TTS_LANG_JAPANESE       2058
#define QWEN_TTS_LANG_KOREAN         2064
#define QWEN_TTS_LANG_GERMAN         2053
#define QWEN_TTS_LANG_FRENCH         2061
#define QWEN_TTS_LANG_RUSSIAN        2069
#define QWEN_TTS_LANG_PORTUGUESE     2071
#define QWEN_TTS_LANG_SPANISH        2054
#define QWEN_TTS_LANG_ITALIAN        2070

/* Speaker IDs (CustomVoice) */
#define QWEN_TTS_SPEAKER_SERENA      3066
#define QWEN_TTS_SPEAKER_VIVIAN      3065
#define QWEN_TTS_SPEAKER_UNCLE_FU    3010
#define QWEN_TTS_SPEAKER_RYAN        3061
#define QWEN_TTS_SPEAKER_AIDEN       2861
#define QWEN_TTS_SPEAKER_ONO_ANNA    2873
#define QWEN_TTS_SPEAKER_SOHEE       2864
#define QWEN_TTS_SPEAKER_ERIC        2875
#define QWEN_TTS_SPEAKER_DYLAN       2878

/* ========================================================================
 * Model Configuration
 * ======================================================================== */

typedef struct {
    /* Talker (Qwen3 LLM backbone) */
    int text_hidden_size;        /* 2048 */
    int hidden_size;             /* 1024 (0.6B) or 2048 (1.7B) */
    int num_layers;              /* 28 */
    int num_heads;               /* 16 */
    int num_kv_heads;            /* 8 (GQA 2:1) */
    int head_dim;                /* 128 */
    int intermediate_size;       /* 3072 (0.6B) or 6144 (1.7B) */
    int codec_vocab_size;        /* 3072 */
    int codebook_size;           /* 2048 */
    float rms_norm_eps;          /* 1e-6 */
    float rope_theta;            /* 1e6 */
    
    /* Code Predictor (MTP module) */
    int cp_hidden_size;          /* 1024 */
    int cp_num_layers;           /* 5 */
    int cp_num_heads;            /* 16 */
    int cp_num_kv_heads;         /* 8 */
    int cp_head_dim;             /* 128 */
    int cp_intermediate_size;    /* 3072 */
    
    /* Speech Decoder */
    int dec_hidden_size;         /* 512 */
    int dec_num_layers;          /* 8 (pre-transformer) */
    int dec_latent_dim;          /* 1024 */
    int dec_codebook_dim;        /* 512 (after VQ projection) */
    int dec_decoder_dim;         /* 1536 */
    int dec_num_heads;           /* 16 */
    int dec_head_dim;            /* 64 */
    int dec_intermediate_size;   /* 1024 */
    int dec_num_quantizers;      /* 16 */
    int dec_sliding_window;      /* 72 */
    float dec_rope_theta;        /* 10000 */
    float dec_rms_norm_eps;      /* 1e-5 */
    int dec_upsample_rates[4];   /* [8, 5, 4, 3] */
    int dec_convnext_ratios[2];  /* [2, 2] */
} qwen_tts_config_t;

/* ========================================================================
 * Talker Layer Weights
 * ======================================================================== */

typedef struct {
    /* QKV projections (bf16) */
    uint16_t *wq_bf16;           /* [q_dim, hidden] = [2048, 1024] */
    uint16_t *wk_bf16;           /* [kv_dim, hidden] = [1024, 1024] */
    uint16_t *wv_bf16;           /* [kv_dim, hidden] = [1024, 1024] */
    uint16_t *wo_bf16;           /* [hidden, q_dim] = [1024, 2048] */
    
    /* Q/K RMSNorm (f32, per-head) */
    float *q_norm;               /* [head_dim] = [128] */
    float *k_norm;               /* [head_dim] = [128] */
    
    /* Layer norms (f32) */
    float *input_norm;           /* [hidden] */
    float *post_attn_norm;       /* [hidden] */
    
    /* SwiGLU MLP (bf16) */
    uint16_t *gate_bf16;         /* [inter, hidden] */
    uint16_t *up_bf16;           /* [inter, hidden] */
    uint16_t *down_bf16;         /* [hidden, inter] */

    /* Fused gate+up for optimization */
    uint16_t *gate_up_fused_bf16; /* [2*inter, hidden] */
} qwen_talker_layer_t;

/* ========================================================================
 * Code Predictor Layer Weights
 * ======================================================================== */

typedef struct {
    /* QKV projections (bf16) */
    uint16_t *wq_bf16;
    uint16_t *wk_bf16;
    uint16_t *wv_bf16;
    uint16_t *wo_bf16;
    
    /* Q/K RMSNorm (f32) */
    float *q_norm;
    float *k_norm;
    
    /* Layer norms (f32) */
    float *input_norm;
    float *post_attn_norm;
    
    /* SwiGLU MLP (bf16) */
    uint16_t *gate_bf16;
    uint16_t *up_bf16;
    uint16_t *down_bf16;

    /* Fused gate+up for optimization */
    uint16_t *gate_up_fused_bf16; /* [2*inter, hidden] */
} qwen_cp_layer_t;

/* ========================================================================
 * Speech Decoder Weights
 * ======================================================================== */

/* Pre-transformer layer */
typedef struct {
    const float *attn_norm;            /* input_layernorm [512] */
    const float *attn_q;               /* q_proj [1024, 512] */
    const float *attn_k;               /* k_proj [1024, 512] */
    const float *attn_v;               /* v_proj [1024, 512] */
    const float *attn_o;               /* o_proj [512, 1024] */
    const float *attn_layer_scale;     /* self_attn_layer_scale [512] */
    const float *ffn_norm;             /* post_attention_layernorm [512] */
    const float *ffn_gate;             /* gate_proj [1024, 512] */
    const float *ffn_up;               /* up_proj [1024, 512] */
    const float *ffn_down;             /* down_proj [512, 1024] */
    const float *ffn_layer_scale;      /* mlp_layer_scale [512] */
} qwen_sd_pre_layer_t;

/* ConvNeXt upsample block */
typedef struct {
    const float *conv_weight;          /* [1024, 1024, 2] */
    const float *conv_bias;
    const float *dwconv_weight;        /* [1024, 1, 7] (depthwise) */
    const float *dwconv_bias;
    const float *pwconv1_weight;       /* [4096, 1024] */
    const float *pwconv1_bias;
    const float *pwconv2_weight;       /* [1024, 4096] */
    const float *pwconv2_bias;
    const float *norm_weight;          /* [1024] */
    const float *norm_bias;
    const float *gamma;                /* [1024] */
} qwen_sd_convnext_t;

/* Upsample block (decoder) */
typedef struct {
    struct {
        const float *conv_weight;    /* [in_ch, out_ch, kernel] */
        const float *conv_bias;      /* [out_ch] */
        const float *snake_alpha;    /* [in_ch] (log-space) */
        const float *snake_beta;     /* [in_ch] (log-space) */
    } upsample;
    struct {
        const float *conv1_weight;   /* [ch, ch, 7] */
        const float *conv1_bias;
        const float *conv2_weight;   /* [ch, ch, 1] */
        const float *conv2_bias;
        const float *snake1_alpha;   /* [ch] (log-space) */
        const float *snake1_beta;
        const float *snake2_alpha;   /* [ch] (log-space) */
        const float *snake2_beta;
    } res_blocks[3];
} qwen_sd_upsample_block_t;

/* Full speech decoder state */
typedef struct {
    /* Codebook embeddings (dequantized from EMA) */
    float *codebook[16];         /* 16 × [2048, 256] */
    
    /* VQ projections */
    const float *rvq_first_input_proj;   /* [256, 512, 1] */
    const float *rvq_first_output_proj;  /* [512, 256, 1] */
    const float *rvq_rest_input_proj;    /* [256, 512, 1] */
    const float *rvq_rest_output_proj;   /* [512, 256, 1] */
    
    /* Pre-conv */
    const float *pre_conv_weight;  /* [1024, 512, 3] */
    const float *pre_conv_bias;    /* [1024] */
    
    /* Pre-transformer */
    qwen_sd_pre_layer_t *pre_layers;  /* 8 layers */
    const float *input_proj_weight;   /* [512, 1024] */
    const float *input_proj_bias;     /* [512] */
    const float *final_norm_weight;   /* [512] - RMSNorm before output_proj */
    const float *output_proj_weight;  /* [1024, 512] */
    const float *output_proj_bias;    /* [1024] */
    
    /* RoPE cache for pre-transformer */
    float *rope_cos;
    float *rope_sin;
    
    /* ConvNeXt upsample blocks */
    qwen_sd_convnext_t convnext[2];
    
    /* Initial conv */
    const float *initial_conv_weight;  /* [1536, 1024, 7] */
    const float *initial_conv_bias;    /* [1536] */
    
    /* Decoder upsample blocks */
    qwen_sd_upsample_block_t upsample_blocks[4];
    
    /* Final conv */
    const float *final_conv_weight;  /* [1, 96, 7] */
    const float *final_conv_bias;    /* [1] */
    
    /* Snake activation params (log-space) */
    struct {
        const float *alpha;
        const float *beta;
    } final_snake;
} qwen_speech_decoder_t;

/* ========================================================================
 * Main Context Structure
 * ======================================================================== */

typedef struct {
    /* Model directory */
    char model_dir[512];
    
    /* Configuration */
    qwen_tts_config_t config;
    
    /* Silence flag */
    int silent;
    int debug;
    
    /* Sampling parameters */
    float temperature;
    int top_k;
    float top_p;
    float rep_penalty;
    int max_tokens;
    float cp_temperature;
    int cp_top_k;
    
    /* Speaker and language */
    int speaker_id;
    int language_id;
    
    /* Random seed */
    uint32_t seed;
    
    /* Safetensors handles */
    void *safetensors;           /* Main model */
    void *speech_safetensors;    /* Speech decoder */
    
    /* Talker weights */
    uint16_t *tok_embeddings_bf16;  /* [vocab, text_hidden] */
    uint16_t *text_proj_fc1_bf16;   /* [text_hidden, text_hidden] */
    float *text_proj_fc1_bias;
    uint16_t *text_proj_fc2_bf16;   /* [hidden, text_hidden] */
    float *text_proj_fc2_bias;
    uint16_t *codec_embedding_bf16; /* [codec_vocab, hidden] */
    uint16_t *codec_head_bf16;      /* [codec_vocab, hidden] */
    float *talker_norm;             /* [hidden] */
    qwen_talker_layer_t layers[QWEN_TTS_MAX_TALKER_LAYERS];
    
    /* Code Predictor weights */
    float *cp_norm;
    qwen_cp_layer_t cp_layers[QWEN_TTS_MAX_CP_LAYERS];
    uint16_t *cp_codec_emb_bf16[15];  /* 15 × [codebook_size, emb_dim] */
    uint16_t *cp_lm_head_bf16[15];    /* 15 × [codebook_size, cp_hidden] */
    int cp_emb_dim;                   /* embedding dim: talker_hidden for 1.7B, cp_hidden for 0.6B */
    uint16_t *cp_mtp_proj_bf16;       /* [cp_hidden, talker_hidden] or NULL if same size */
    float *cp_mtp_proj_bias;          /* [cp_hidden] or NULL */
    
    /* Speech decoder */
    qwen_speech_decoder_t speech_dec;
    
    /* KV cache (Talker) */
    float *kv_cache_k;
    float *kv_cache_v;
    int kv_max;
    int kv_len;
    
    /* KV cache (Code Predictor) */
    float *cp_kv_k;
    float *cp_kv_v;
    int cp_kv_max;
    int cp_kv_len;
    
    /* Decode buffers (single token) */
    float *dec_x;
    float *dec_x_norm;
    float *dec_q;
    float *dec_k;
    float *dec_v;
    float *dec_attn_out;
    float *dec_proj_out;
    float *dec_gate;
    float *dec_up;
    float *dec_ffn_out;
    
    /* CP decode buffers */
    float *cp_dec_x;
    float *cp_dec_q;
    float *cp_dec_k;
    float *cp_dec_v;
    float *cp_dec_attn_out;
    float *cp_dec_gate;
    float *cp_dec_up;
    float *cp_dec_ffn_out;
    
    /* Prefill buffers */
    float *pref_x;
    float *pref_x_norm;
    float *pref_q;
    float *pref_k;
    float *pref_v;
    float *pref_attn_out;
    float *pref_ffn_out;
    int pref_seq_cap;
    
    /* RoPE caches */
    float *rope_cos;
    float *rope_sin;
    float *rope_inv_freq;
    int rope_cache_len;
    
    float *cp_rope_cos;
    float *cp_rope_sin;
    int cp_rope_cache_len;
    
    /* Logits buffer */
    float *logits;
    
    /* Generation state */
    int *codec_codes;
    int codec_frames;
    int codec_frames_cap;
    int *prev_tokens;
    int n_prev_tokens;
    int prev_tokens_cap;
    
    /* Audio output buffer */
    float *audio_buf;
    int audio_samples;
} qwen_tts_ctx_t;

/* ========================================================================
 * API Functions
 * ======================================================================== */

#ifdef __cplusplus
extern "C" {
#endif

/* Load model from directory */
qwen_tts_ctx_t *qwen_tts_load(const char *model_dir);

/* Unload model and free resources */
void qwen_tts_unload(qwen_tts_ctx_t *ctx);

/* Set speaker ID */
void qwen_tts_set_speaker(qwen_tts_ctx_t *ctx, int speaker_id);

/* Set language by name */
void qwen_tts_set_language(qwen_tts_ctx_t *ctx, const char *language);

/* Get language ID from name */
int qwen_tts_language_id(const char *name);

/* Get speaker ID from name */
int qwen_tts_speaker_id(const char *name);

/* Generate speech from text */
int qwen_tts_generate(qwen_tts_ctx_t *ctx, const char *text,
                      float **out_samples, int *out_n_samples);

/* Write WAV file */
int qwen_tts_write_wav(const char *path, const float *samples, int n_samples, int sample_rate);

#ifdef __cplusplus
}
#endif

#endif /* QWEN_TTS_H */
