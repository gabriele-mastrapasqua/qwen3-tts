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

#include "qwen_tts_kernels.h"
#include "qwen_tts_voice_clone.h"

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

    /* INT8 quantized weights (optional, allocated if --int8 flag is set) */
    int8_t *wq_int8;              /* [q_dim, hidden] */
    float  *wq_scale;             /* [q_dim] per-row scale */
    int8_t *wk_int8;              /* [kv_dim, hidden] */
    float  *wk_scale;             /* [kv_dim] */
    int8_t *wv_int8;              /* [kv_dim, hidden] */
    float  *wv_scale;             /* [kv_dim] */
    int8_t *wo_int8;              /* [hidden, q_dim] */
    float  *wo_scale;             /* [hidden] */
    int8_t *gate_up_fused_int8;   /* [2*inter, hidden] */
    float  *gate_up_fused_scale;  /* [2*inter] */
    int8_t *down_int8;            /* [hidden, inter] */
    float  *down_scale;           /* [hidden] */

    /* Q4_0 quantized weights (optional, allocated if --int4 flag is set) */
    q4_0_block_t *wq_q4;              /* [q_dim, hidden/32 blocks] */
    q4_0_block_t *wk_q4;              /* [kv_dim, hidden/32 blocks] */
    q4_0_block_t *wv_q4;              /* [kv_dim, hidden/32 blocks] */
    q4_0_block_t *wo_q4;              /* [hidden, q_dim/32 blocks] */
    q4_0_block_t *gate_up_fused_q4;   /* [2*inter, hidden/32 blocks] */
    q4_0_block_t *down_q4;            /* [hidden, inter/32 blocks] */
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

    /* INT8 quantized weights (optional, allocated if --int8 flag is set) */
    int8_t *wq_int8;           /* [q_dim, hidden] */
    float  *wq_scale;          /* [q_dim] per-row scale */
    int8_t *wk_int8;           /* [kv_dim, hidden] */
    float  *wk_scale;          /* [kv_dim] */
    int8_t *wv_int8;           /* [kv_dim, hidden] */
    float  *wv_scale;          /* [kv_dim] */
    int8_t *wo_int8;           /* [hidden, q_dim] */
    float  *wo_scale;          /* [hidden] */
    int8_t *gate_up_fused_int8; /* [2*inter, hidden] */
    float  *gate_up_fused_scale; /* [2*inter] */
    int8_t *down_int8;         /* [hidden, inter] */
    float  *down_scale;        /* [hidden] */

    /* Q4_0 quantized weights (optional, allocated if --int4 flag is set) */
    q4_0_block_t *wq_q4;              /* [q_dim, hidden/32 blocks] */
    q4_0_block_t *wk_q4;              /* [kv_dim, hidden/32 blocks] */
    q4_0_block_t *wv_q4;              /* [kv_dim, hidden/32 blocks] */
    q4_0_block_t *wo_q4;              /* [hidden, q_dim/32 blocks] */
    q4_0_block_t *gate_up_fused_q4;   /* [2*inter, hidden/32 blocks] */
    q4_0_block_t *down_q4;            /* [hidden, inter/32 blocks] */

    /* Q2_0 FFN (EXPERIMENTAL hybrid: --int4 + QWEN_CP_Q2_FFN=1). Shrinks the
     * biggest, most quant-tolerant matrices below int4 to reduce DRAM bytes. */
    q2_0_block_t *gate_up_fused_q2;   /* [2*inter, hidden/32 blocks] */
    q2_0_block_t *down_q2;            /* [hidden, inter/32 blocks] */

    /* Q2_0 copy of down used ONLY by the --roughness knob (feat/expressivity):
     * built lazily from down_bf16 independent of the active quant mode, blended
     * into the high-precision down output to dial in texture roughness. */
    q2_0_block_t *down_q2_rough;      /* [hidden, inter/32 blocks], NULL until roughness>0 */
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

/* Speech decoder streaming state (incremental decode) */
#define QWEN_SD_STREAM_MAX_LAYERS 8
#define QWEN_SD_STREAM_CONV_RF 20  /* Conv decoder receptive field in latent frames */

typedef struct {
    /* Pre-transformer KV cache: [n_layers][alloc × qkv_dim], row-major */
    float *k_cache[QWEN_SD_STREAM_MAX_LAYERS];
    float *v_cache[QWEN_SD_STREAM_MAX_LAYERS];
    int kv_len;         /* total frames processed (absolute, monotonic) */
    int kv_alloc;       /* allocated PHYSICAL capacity in frames */
    int kv_base;        /* plan_v4 D2: absolute frame stored at physical slot 0.
                         * Attention only ever reads the last `window` (72) frames,
                         * so the cache is compacted to O(window+chunk) instead of
                         * growing with total stream length. physical = abs - kv_base;
                         * RoPE is relative so the continuous abs timeline is unchanged. */

    /* Pre-transformer output cache (latent_out): row-major [alloc × latent_dim] */
    float *latent_cache;   /* [latent_alloc, 1024] */
    int latent_frames;     /* total latent frames produced (absolute, monotonic) */
    int latent_alloc;      /* allocated PHYSICAL capacity in frames */
    int latent_base;       /* plan_v4 D2: absolute frame at physical slot 0. The
                            * conv decoder only reads the last conv_rf+chunk frames,
                            * so trim like the KV cache. physical = abs - latent_base. */

    /* Pre-conv left padding: last 2 timesteps of VQ output, channel-first [512, 2] */
    float *vq_pad;
    int vq_pad_valid;

    /* Tracking */
    int frames_decoded;    /* total codec frames processed */
    int samples_produced;  /* total audio samples produced */
    int initialized;       /* 1 after first call */
} qwen_sd_stream_state_t;

/* ========================================================================
 * Audio Callback (for streaming)
 * ======================================================================== */

/* Called with each decoded audio chunk during streaming generation.
 * samples: float PCM in [-1, 1], n_samples: count, userdata: opaque pointer.
 * Return 0 to continue, non-zero to abort generation. */
typedef int (*qwen_tts_audio_cb)(const float *samples, int n_samples, void *userdata);

/* ========================================================================
 * Main Context Structure
 * ======================================================================== */

typedef struct qwen_tts_ctx {
    /* Model directory */
    char model_dir[512];
    
    /* Configuration */
    qwen_tts_config_t config;
    
    /* Silence flag */
    int silent;
    int debug;
    int use_int8;  /* INT8 quantized Code Predictor weights */
    int use_int4;  /* Q4_0 quantized Talker weights (1.7B only) */
    
    /* Sampling parameters */
    float temperature;
    int top_k;
    float top_p;
    float rep_penalty;
    int max_tokens;
    float cp_temperature;
    int cp_top_k;
    int greedy_warmup;  /* initial frames sampled greedily (temp=0) for cross-model stability */

    /* Speaker and language */
    int speaker_id;
    int language_id;

    /* Instruct text (style/emotion control, 1.7B only; required for VoiceDesign) */
    char *instruct;

    /* VoiceDesign mode (no preset speakers, voice created from instruct) */
    int voice_design;

    /* Voice clone mode (Base model only) */
    int voice_clone;             /* 1 = voice clone active */
    int xvector_only;            /* 1 = x-vector only (no ICL), 0 = ICL mode */
    float *speaker_embedding;    /* [hidden_size] speaker embedding from ref audio */
    float max_ref_seconds;       /* Max ref audio duration for embedding (0=all, default 15) */
    char *ref_audio_path;        /* Path to reference audio file */
    char *ref_text;              /* Reference text for ICL mode */

    /* Cached ICL data (for .qvoice save/load) */
    int *cached_ref_codes;       /* [cached_ref_n_frames × 16] codec tokens from speech encoder */
    int cached_ref_n_frames;     /* Number of reference codec frames */
    /* Override weight buffers (WDELTA/WOVR/--expr) that REPLACED mmap pointers — tracked so unload
     * frees them (leaks-audit #3). A worker clone shares this list via the shallow `*w=*base` copy;
     * only the base's unload frees it (qwen_tts_free_clone leaves it). */
    void **owned_overrides;
    int    n_owned_overrides;
    int    cap_owned_overrides;
    int icl_frames_cap;          /* --icl-frames N: cap ICL ref frames to dilute the prosody
                                    anchor (more emotion room). 0 = use all (default). */
    int graft_mode;              /* --graft: ignore cached ref_codes on a lite .qvoice and clone
                                    via the x-vector path (CV weights + instruct emote = the graft).
                                    Same 25MB file: default = faithful ICL, --graft = emotive. */

    /* Base model type */
    int is_base_model;           /* 1 = Base model, 0 = CustomVoice/VoiceDesign */
    int speaker_enc_dim;         /* Speaker encoder output dim (1024 for 0.6B, 2048 for 1.7B) */

    /* Streaming */
    int stream;                  /* Enable streaming (decode chunks during generation) */
    int stream_chunk_frames;     /* Frames per chunk (default: 10 = 0.8s audio) */
    qwen_tts_audio_cb audio_cb;  /* Audio callback for streaming */
    void *audio_cb_userdata;

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
    int8_t   *cp_lm_head_int8[15];   /* INT8 quantized lm_head (optional) */
    float    *cp_lm_head_scale[15];  /* per-row scales for INT8 lm_head */
    q4_0_block_t *cp_lm_head_q4[15]; /* Q4_0 quantized lm_head (optional, --int4) */
    int cp_emb_dim;                   /* embedding dim: talker_hidden for 1.7B, cp_hidden for 0.6B */
    uint16_t *cp_mtp_proj_bf16;       /* [cp_hidden, talker_hidden] or NULL if same size */
    float *cp_mtp_proj_bias;          /* [cp_hidden] or NULL */
    
    /* Speech decoder */
    qwen_speech_decoder_t speech_dec;
    qwen_sd_stream_state_t sd_stream;  /* Streaming incremental decode state */

    /* Speaker encoder (ECAPA-TDNN, Base model only) */
    qwen_speaker_encoder_t speaker_enc;
    
    /* KV cache (Talker) — stored as bf16 to halve memory and improve cache utilization */
    uint16_t *kv_cache_k;
    uint16_t *kv_cache_v;
    int kv_max;
    int kv_len;

    /* --batch orchestration: prefill_only=1 makes qwen_tts_generate stop right after
     * prefill (KV populated, dec_x = last position), so the batched orchestrator can
     * capture each chunk's KV + seed hidden. bg_text_content_len stashes the text
     * token count (for the per-chunk EOS-boost heuristic). Not used on the normal path. */
    int prefill_only;
    int bg_text_content_len;

    /* KV cache (Code Predictor) — stored as bf16 */
    uint16_t *cp_kv_k;
    uint16_t *cp_kv_v;
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
    float *swiglu_tmp;  /* Temp buffer for batch vvexpf in SwiGLU */
    
    /* CP decode buffers */
    float *cp_dec_x;
    float *cp_dec_q;
    float *cp_dec_k;
    float *cp_dec_v;
    float *cp_dec_attn_out;
    float *cp_dec_gate;
    float *cp_dec_up;
    float *cp_dec_ffn_out;
    
    /* Prefill buffers (persisted across generations for server mode) */
    float *pref_residual;
    float *pref_x_norm;
    float *pref_q;
    float *pref_k;
    float *pref_v;
    float *pref_attn_out;
    float *pref_gate;
    float *pref_proj;
    int pref_seq_cap;
    /* Prefill weight conversion buffers (fixed size, allocated once) */
    float *pref_wq_f32;
    float *pref_wk_f32;
    float *pref_wv_f32;
    float *pref_wo_f32;
    float *pref_gate_up_f32;
    float *pref_down_f32;
    
    /* RoPE caches */
    float *rope_cos;
    float *rope_sin;
    float *rope_inv_freq;
    int rope_cache_len;
    
    float *cp_rope_cos;
    float *cp_rope_sin;
    int cp_rope_cache_len;
    
    /* Text embedding temp buffers (reused across embed_one_text_token calls) */
    float *emb_tmp1;
    float *emb_tmp2;

    /* Logits buffer */
    float *logits;
    
    /* Generation state */
    int *codec_codes;
    int codec_frames;
    int codec_frames_cap;
    int *prev_tokens;
    int n_prev_tokens;
    int prev_tokens_cap;

    /* Quant-ladder teacher-forcing: when non-NULL, the CP autoregression feeds
     * these reference codebook-1..15 codes (instead of its own predictions) so
     * every precision sees identical per-step inputs → isolates CP quant drift.
     * Set per-frame by the generation loop in QWEN_TF_CODES replay mode; else NULL. */
    const int *tf_ref_codes;

    /* ---- Expressivity / prosody control (feat/expressivity) ----
     * All default to 0/NULL → the normal path is bit-identical (no overhead). */
    float  cp_roughness;     /* 0..1: blend q2-down into the high-prec down output
                              *        (texture "roughness" knob; 0 = off). */
    int    cp_rough_built;   /* lazy-build guard for the per-layer down_q2_rough copies. */

    /* Multi-layer Talker steering (emotion identity lives at late layers, not the
     * single cp_x point). ml_steer is [num_layers+1][hidden]: after Talker layer L
     * the residual stream gets dec_x += ml_steer_weight * ml_steer[L]. Layers outside
     * [ml_steer_l0, ml_steer_l1] are skipped (zero vectors or range gate). Built from
     * two QWEN_ACT_MAP captures (emotion − neutral); see tests/act_map_steer.py. */
    float *ml_steer;         /* [(num_layers+1) * hidden] per-layer steer, or NULL. */
    int    ml_steer_layers;  /* num_layers+1 */
    int    ml_steer_dim;     /* hidden */
    float  ml_steer_weight;  /* injection scale (base, per-frame schedule applied on top) */
    int    ml_steer_l0, ml_steer_l1;  /* inclusive layer range to inject (e.g. 21..25) */
    /* Per-frame application SCHEDULE (the fix for the energy-collapse spiral): a fixed
     * additive bias every autoregressive frame compounds toward silence. Instead apply a
     * decaying / first-N-frames "mood-set" pulse. The gen loop sets ml_steer_w_eff before
     * each talker step; talker_step uses w_eff (NOT ml_steer_weight) and is 0 during prefill. */
    float  ml_steer_decay;   /* per-frame multiplier g: eff = weight * g^frame (1.0 = no decay) */
    int    ml_steer_frames;  /* apply only first N generation frames (0 = all frames) */
    float  ml_steer_w_eff;   /* runtime: effective weight for the current talker step (0 = off) */

    /* Audio output buffer */
    float *audio_buf;
    int audio_samples;

    /* Cached tokenizer (avoid re-loading from disk each call) */
    void *cached_tokenizer;

    /* Text embedding cache (server optimization) */
    float *cached_tts_pad_embed;   /* [hidden] — computed once at load */
    float *cached_tts_bos_embed;   /* [hidden] */
    float *cached_tts_eos_embed;   /* [hidden] */

    /* LRU token embedding cache: token_id → float[hidden]
     * Open-addressing hash map with linear probing. */
    struct {
        int *keys;          /* token IDs (-1 = empty) */
        float *values;      /* [capacity × hidden] embeddings */
        uint32_t *access;   /* access counter for LRU eviction */
        int capacity;       /* hash map capacity (power of 2) */
        int count;          /* entries currently cached */
        uint32_t clock;     /* global access counter */
    } emb_cache;

    /* Delta prefill cache: store previous prompt embeddings for KV reuse.
     * On repeat calls with same speaker/language, the prompt prefix
     * (role + codec) is identical — we skip re-prefilling those tokens. */
    float *prev_input_embeds;    /* [prev_prefill_len × hidden] */
    int prev_prefill_len;        /* length of cached embeddings */

} qwen_tts_ctx_t;

/* ========================================================================
 * API Functions
 * ======================================================================== */

#ifdef __cplusplus
extern "C" {
#endif

/* Load model from directory */
qwen_tts_ctx_t *qwen_tts_load(const char *model_dir);
qwen_tts_ctx_t *qwen_tts_load_ex(const char *model_dir, int silent, int use_int8, int use_int4);

/* Unload model and free resources */
void qwen_tts_unload(qwen_tts_ctx_t *ctx);

/* Track a malloc'd weight-override buffer (WDELTA/WOVR/--expr) so qwen_tts_unload frees it
 * (leaks-audit #3). No-op on NULL. */
void qwen_track_override(qwen_tts_ctx_t *ctx, void *ptr);

/* Concurrent server support: clone a loaded context into an independent worker
 * context that SHARES read-only weights/voice/RoPE but owns fresh per-request
 * mutable buffers (KV caches, work buffers, caches). Free with
 * qwen_tts_free_clone — NEVER qwen_tts_unload (that frees shared weights). */
qwen_tts_ctx_t *qwen_tts_clone_for_worker(const qwen_tts_ctx_t *base);
void qwen_tts_free_clone(qwen_tts_ctx_t *ctx);

/* Set speaker ID */
void qwen_tts_set_speaker(qwen_tts_ctx_t *ctx, int speaker_id);

/* Set language by name */
void qwen_tts_set_language(qwen_tts_ctx_t *ctx, const char *language);

/* Get language ID from name */
int qwen_tts_language_id(const char *name);

/* Get speaker ID from name */
int qwen_tts_speaker_id(const char *name);

/* Set audio callback for streaming (called with each decoded chunk) */
void qwen_tts_set_audio_callback(qwen_tts_ctx_t *ctx, qwen_tts_audio_cb cb, void *userdata);

/* Generate speech from text */
int qwen_tts_generate(qwen_tts_ctx_t *ctx, const char *text,
                      float **out_samples, int *out_n_samples);

/* Batched long-form generation: step `nc` independent chunks through Talker+CP
 * together (weight-stationary), decode per chunk, concatenate (chunk_pause seconds
 * of silence between chunks). Returns -2 if the model can't use the bf16 batched
 * path (caller falls back to sequential). Used by --batch (Milestone B). */
int qwen_tts_generate_batch(qwen_tts_ctx_t *ctx, char **chunks, int nc,
                            float chunk_pause, float **out_samples, int *out_n_samples);

/* ── Server request-batching (vLLM-style) ──────────────────────────────────
 * One independent request per batch slot: different text, speaker, language and
 * sampling params, producing SEPARATE outputs. This is the engine the concurrent
 * server steps N users' requests through Talker+CP together (weight-stationary).
 * Distinct from --batch (which splits ONE long text into chunks and concatenates).
 *
 * Per-request voice is limited to PRESET speakers (carried in the prompt → per-seq
 * KV via prefill). A loaded custom .qvoice / quant mode is per-SERVER (shared
 * weights), set at startup; not switchable per slot. */
typedef struct {
    const char *text;
    int   speaker_id;     /* preset speaker for this request */
    int   language_id;    /* language id for this request */
    float temperature;
    int   top_k;
    float top_p;
    float rep_penalty;
    uint32_t seed;
    int   greedy_warmup;
    int   want_stream;    /* 1 = stream this request's frames incrementally (S3) */
} qwen_batch_req_t;

/* Step `nc` independent requests together; each writes its OWN audio buffer into
 * out_samples[i] (malloc'd, caller frees) and out_n_samples[i]. Per-slot sampling
 * params + per-slot RNG state reproduce single-stream output bit-for-bit. Returns
 * -2 if the model can't use the bf16 batched path. nc is capped internally per
 * group (GMAX); the server scheduler controls admission. */
int qwen_tts_generate_batch_multi(qwen_tts_ctx_t *ctx,
                                  const qwen_batch_req_t *reqs, int nc,
                                  float **out_samples, int *out_n_samples);

/* ── Continuous request-batching driver (vLLM-style, S2) ────────────────────
 * A persistent frame-stepping loop over up to `max_batch` slots. When a slot's
 * request hits EOS it is finalized (decoded → delivered) and the freed slot is
 * immediately refilled with a queued request (compact+refill / continuous
 * batching) — no waiting for the slowest in a group. The host (server) supplies
 * a job source + delivery sink via callbacks; the driver owns ctx + the batch
 * state and is the sole synthesizer. */
typedef struct {
    void *ud;
    /* Pull the next BATCH request. Fill *req + *tag (opaque handle the host uses
     * to route the response). Return 1 if one was dequeued, 0 if none. When
     * block!=0 the driver is fully idle and the callback MAY block until a job
     * arrives or shutdown (return 0 on shutdown). req->text must stay valid until
     * on_done is called for that tag (the driver does not copy it). */
    int (*next_job)(void *ud, qwen_batch_req_t *req, void **tag, int block);
    /* Non-streaming: deliver a finished request's full audio (malloc'd, host
     * frees) or NULL on failure. For a streaming request this is the END marker
     * (samples=NULL,n=0) → host finishes the chunked response. Driver is done
     * with `tag` after this returns. */
    void (*on_done)(void *ud, void *tag, float *samples, int n_samples);
    /* Streaming only (req.want_stream): one incremental PCM chunk for `tag` as
     * its frame is decoded (driver frees `samples` after this returns). May be
     * NULL → driver won't stream even if want_stream is set. */
    void (*on_chunk)(void *ud, void *tag, float *samples, int n_samples);
    /* Return non-zero while the server should keep running. */
    int (*running)(void *ud);
} qwen_batch_sink_t;

/* Run the continuous-batching loop until running()==0 and the queue drains.
 * Returns 0 on clean exit, -2 if the model can't use the bf16 batched path. */
int qwen_tts_serve_continuous(qwen_tts_ctx_t *ctx, int max_batch, qwen_batch_sink_t *sink);

/* Write WAV file */
int qwen_tts_write_wav(const char *path, const float *samples, int n_samples, int sample_rate);

/* Speech encoder: audio → codec codes (for ICL voice clone) */
int qwen_speech_encoder_load(qwen_tts_ctx_t *ctx);
int qwen_speech_encoder_encode(qwen_tts_ctx_t *ctx, const float *audio, int n_samples,
                                int **codes_out, int *n_frames_out);

#ifdef __cplusplus
}
#endif

#endif /* QWEN_TTS_H */
