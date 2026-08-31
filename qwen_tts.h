/* qwen_tts.h - Qwen3-TTS Pure C Inference Engine */
#ifndef QWEN_TTS_H
#define QWEN_TTS_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <pthread.h>

#include "qwen_tts_kernels.h"
#include "qwen_tts_voice_clone.h"

#define QWEN_TTS_SAMPLE_RATE         24000
#define QWEN_TTS_FRAME_RATE          12.5
#define QWEN_TTS_HOP_SAMPLES         1920

#define QWEN_TTS_MAX_TALKER_LAYERS   28
#define QWEN_TTS_MAX_CP_LAYERS       5
#define QWEN_TTS_MAX_DECODER_LAYERS  8

#define QWEN_TTS_TEXT_VOCAB_SIZE     151936
#define QWEN_TTS_CODEC_VOCAB_SIZE    3072
#define QWEN_TTS_CODEBOOK_SIZE       2048
#define QWEN_TTS_NUM_CODEBOOKS       16
#define QWEN_TTS_CODEBOOK_DIM        256

#define QWEN_TTS_TOK_IM_START        151644
#define QWEN_TTS_TOK_IM_END          151645
#define QWEN_TTS_TOK_ENDOFTEXT       151643
#define QWEN_TTS_TTS_BOS             151672
#define QWEN_TTS_TTS_EOS             151673
#define QWEN_TTS_TTS_PAD             151671

#define QWEN_TTS_CODEC_PAD           2148
#define QWEN_TTS_CODEC_BOS           2149
#define QWEN_TTS_CODEC_EOS           2150
#define QWEN_TTS_CODEC_THINK         2154
#define QWEN_TTS_CODEC_NO_THINK      2155
#define QWEN_TTS_CODEC_THINK_BOS     2156
#define QWEN_TTS_CODEC_THINK_EOS     2157

typedef enum {
    QWEN_EOS_OFF  = 0,
    QWEN_EOS_V1   = 1,
    QWEN_EOS_V2   = 2,
    QWEN_EOS_TOPK = 3,
} qwen_eos_strategy_t;

const char *qwen_tts_eos_strategy_name(int strategy);
int         qwen_tts_eos_strategy_parse(const char *name);

int qwen_tts_load_speaker_map(qwen_tts_ctx_t *ctx, const char *path);

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

#define QWEN_TTS_SPEAKER_SERENA      3066
#define QWEN_TTS_SPEAKER_VIVIAN      3065
#define QWEN_TTS_SPEAKER_UNCLE_FU    3010
#define QWEN_TTS_SPEAKER_RYAN        3061
#define QWEN_TTS_SPEAKER_AIDEN       2861
#define QWEN_TTS_SPEAKER_ONO_ANNA    2873
#define QWEN_TTS_SPEAKER_SOHEE       2864
#define QWEN_TTS_SPEAKER_ERIC        2875
#define QWEN_TTS_SPEAKER_DYLAN       2878

typedef struct {
    int text_hidden_size;
    int hidden_size;
    int num_layers;
    int num_heads;
    int num_kv_heads;
    int head_dim;
    int intermediate_size;
    int codec_vocab_size;
    int codebook_size;
    float rms_norm_eps;
    float rope_theta;

    int cp_hidden_size;
    int cp_num_layers;
    int cp_num_heads;
    int cp_num_kv_heads;
    int cp_head_dim;
    int cp_intermediate_size;

    int dec_hidden_size;
    int dec_num_layers;
    int dec_latent_dim;
    int dec_codebook_dim;
    int dec_decoder_dim;
    int dec_num_heads;
    int dec_head_dim;
    int dec_intermediate_size;
    int dec_num_quantizers;
    int dec_sliding_window;
    float dec_rope_theta;
    float dec_rms_norm_eps;
    int dec_upsample_rates[4];
    int dec_convnext_ratios[2];
} qwen_tts_config_t;

typedef struct {
    uint16_t *wq_bf16;
    uint16_t *wk_bf16;
    uint16_t *wv_bf16;
    uint16_t *wo_bf16;

    float *q_norm;
    float *k_norm;

    float *input_norm;
    float *post_attn_norm;

    uint16_t *gate_bf16;
    uint16_t *up_bf16;
    uint16_t *down_bf16;

    uint16_t *gate_up_fused_bf16;

    int8_t *wq_int8;
    float  *wq_scale;
    int8_t *wk_int8;
    float  *wk_scale;
    int8_t *wv_int8;
    float  *wv_scale;
    int8_t *wo_int8;
    float  *wo_scale;
    int8_t *gate_up_fused_int8;
    float  *gate_up_fused_scale;
    int8_t *down_int8;
    float  *down_scale;

    q4_0_block_t *wq_q4;
    q4_0_block_t *wk_q4;
    q4_0_block_t *wv_q4;
    q4_0_block_t *wo_q4;
    q4_0_block_t *gate_up_fused_q4;
    q4_0_block_t *down_q4;

    q6_0_block_t *wq_q6;
    q6_0_block_t *wk_q6;
    q6_0_block_t *wv_q6;
    q6_0_block_t *wo_q6;
    q6_0_block_t *gate_up_fused_q6;
    q6_0_block_t *down_q6;

    const uint16_t *wq_bf16_pref, *wk_bf16_pref, *wv_bf16_pref, *wo_bf16_pref;
    const uint16_t *gate_up_fused_bf16_pref, *down_bf16_pref;
} qwen_talker_layer_t;

typedef struct {
    uint16_t *wq_bf16;
    uint16_t *wk_bf16;
    uint16_t *wv_bf16;
    uint16_t *wo_bf16;

    float *q_norm;
    float *k_norm;

    float *input_norm;
    float *post_attn_norm;

    uint16_t *gate_bf16;
    uint16_t *up_bf16;
    uint16_t *down_bf16;

    uint16_t *gate_up_fused_bf16;

    int8_t *wq_int8;
    float  *wq_scale;
    int8_t *wk_int8;
    float  *wk_scale;
    int8_t *wv_int8;
    float  *wv_scale;
    int8_t *wo_int8;
    float  *wo_scale;
    int8_t *gate_up_fused_int8;
    float  *gate_up_fused_scale;
    int8_t *down_int8;
    float  *down_scale;

    q4_0_block_t *wq_q4;
    q4_0_block_t *wk_q4;
    q4_0_block_t *wv_q4;
    q4_0_block_t *wo_q4;
    q4_0_block_t *gate_up_fused_q4;
    q4_0_block_t *down_q4;

    q2_0_block_t *gate_up_fused_q2;
    q2_0_block_t *down_q2;

    q2_0_block_t *down_q2_rough;
} qwen_cp_layer_t;

typedef struct {
    const float *attn_norm;
    const float *attn_q;
    const float *attn_k;
    const float *attn_v;
    const float *attn_o;
    const float *attn_layer_scale;
    const float *ffn_norm;
    const float *ffn_gate;
    const float *ffn_up;
    const float *ffn_down;
    const float *ffn_layer_scale;
} qwen_sd_pre_layer_t;

typedef struct {
    const float *conv_weight;
    const float *conv_bias;
    const float *dwconv_weight;
    const float *dwconv_bias;
    const float *pwconv1_weight;
    const float *pwconv1_bias;
    const float *pwconv2_weight;
    const float *pwconv2_bias;
    const float *norm_weight;
    const float *norm_bias;
    const float *gamma;
} qwen_sd_convnext_t;

typedef struct {
    struct {
        const float *conv_weight;
        const float *conv_bias;
        const float *snake_alpha;
        const float *snake_beta;
    } upsample;
    struct {
        const float *conv1_weight;
        const float *conv1_bias;
        const float *conv2_weight;
        const float *conv2_bias;
        const float *snake1_alpha;
        const float *snake1_beta;
        const float *snake2_alpha;
        const float *snake2_beta;
    } res_blocks[3];
} qwen_sd_upsample_block_t;

typedef struct {
    float *codebook[16];

    const float *rvq_first_input_proj;
    const float *rvq_first_output_proj;
    const float *rvq_rest_input_proj;
    const float *rvq_rest_output_proj;

    const float *pre_conv_weight;
    const float *pre_conv_bias;

    qwen_sd_pre_layer_t *pre_layers;
    const float *input_proj_weight;
    const float *input_proj_bias;
    const float *final_norm_weight;
    const float *output_proj_weight;
    const float *output_proj_bias;

    float *rope_cos;
    float *rope_sin;

    qwen_sd_convnext_t convnext[2];

    const float *initial_conv_weight;
    const float *initial_conv_bias;

    qwen_sd_upsample_block_t upsample_blocks[4];

    float *convt_packed[6];

    const float *final_conv_weight;
    const float *final_conv_bias;

    struct {
        const float *alpha;
        const float *beta;
    } final_snake;
} qwen_speech_decoder_t;

#define QWEN_SD_STREAM_MAX_LAYERS 8
#define QWEN_SD_STREAM_CONV_RF 20

typedef struct {
    float *k_cache[QWEN_SD_STREAM_MAX_LAYERS];
    float *v_cache[QWEN_SD_STREAM_MAX_LAYERS];
    int kv_len;
    int kv_alloc;
    int kv_base;

    float *latent_cache;
    int latent_frames;
    int latent_alloc;
    int latent_base;

    float *vq_pad;
    int vq_pad_valid;

    float *cs_cn_dw_tail[2];
    float *cs_init_tail;
    float *cs_up_carry[4];
    float *cs_res_tail[4][3];
    float *cs_final_tail;
    int    cs_alloc;
    int    cs_warm;

    int frames_decoded;
    int samples_produced;
    int initialized;
} qwen_sd_stream_state_t;

typedef struct {
    qwen_sd_stream_state_t *st;
    const int *codes;
    int nframes;
    float *audio;
    int n_samples;
    int rc;
} qwen_sd_batch_item_t;

typedef int (*qwen_tts_audio_cb)(const float *samples, int n_samples, void *userdata);

typedef struct qwen_tts_ctx {
    char model_dir[512];

    qwen_tts_config_t config;

    int silent;
    int debug;
    int use_int8;
    int use_int4;

    float temperature;
    int top_k;
    float top_p;
    float rep_penalty;
    int max_tokens;
    float cp_temperature;
    int cp_top_k;
    int greedy_warmup;

    int   eos_strategy;
    int   eos_suppress_frames;
    float eos_frames_per_token;
    float eos_start_multiple;
    int   eos_overhead_frames;
    float eos_ramp_per_frame;
    float eos_ramp_cap;
    int   eos_topk;

    int speaker_id;
    int language_id;

    char **spk_names;
    int   *spk_slots;
    struct {
        char talker_linear[224];   int talker_n, talker_eligible;
        char cp_linear[224];       int cp_n, cp_eligible;
        char cp_heads[224];        int cp_heads_n;
        char extras[224];          int extras_n;
    } src;

    int    spk_count;

    char *instruct;

    int voice_design;

    int voice_clone;
    int xvector_only;
    float *speaker_embedding;
    float max_ref_seconds;
    char *ref_audio_path;
    char *ref_text;

    char *emo_ref_path;
    char *emo_ref_text;

    int *cached_ref_codes;
    int cached_ref_n_frames;
    void **owned_overrides;
    int    n_owned_overrides;
    int    cap_owned_overrides;
    int icl_frames_cap;
    int graft_mode;

    int is_base_model;
    int speaker_enc_dim;

    int stream;
    int stream_chunk_frames;
    qwen_tts_audio_cb audio_cb;
    void *audio_cb_userdata;

    uint32_t seed;

    void *safetensors;
    void *speech_safetensors;

    uint16_t *tok_embeddings_bf16;
    uint16_t *text_proj_fc1_bf16;
    float *text_proj_fc1_bias;
    uint16_t *text_proj_fc2_bf16;
    q4_0_block_t *text_proj_fc1_q4;
    q4_0_block_t *text_proj_fc2_q4;
    float *text_proj_fc2_bias;
    uint16_t *codec_embedding_bf16;
    uint16_t *codec_head_bf16;
    q4_0_block_t *codec_head_q4;
    float *talker_norm;
    qwen_talker_layer_t layers[QWEN_TTS_MAX_TALKER_LAYERS];

    float *cp_norm;
    qwen_cp_layer_t cp_layers[QWEN_TTS_MAX_CP_LAYERS];
    uint16_t *cp_codec_emb_bf16[15];
    uint16_t *cp_lm_head_bf16[15];
    int8_t   *cp_lm_head_int8[15];
    float    *cp_lm_head_scale[15];
    q4_0_block_t *cp_lm_head_q4[15];
    int cp_emb_dim;
    uint16_t *cp_mtp_proj_bf16;
    float *cp_mtp_proj_bias;
    int8_t *cp_mtp_proj_int8;
    float *cp_mtp_proj_scale;
    q4_0_block_t *cp_mtp_proj_q4;

    qwen_speech_decoder_t speech_dec;
    qwen_sd_stream_state_t sd_stream;

    qwen_speaker_encoder_t speaker_enc;

    int      pfx_len, pfx_spk, pfx_lang, pfx_think;
    uint64_t pfx_ihash;

    uint16_t *kv_cache_k;
    uint16_t *kv_cache_v;
    int kv_max;
    int kv_len;

    int prefill_only;
    int bg_text_content_len;

    uint16_t *cp_kv_k;
    uint16_t *cp_kv_v;
    int cp_kv_max;
    int cp_kv_len;

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
    float *swiglu_tmp;

    float *cp_dec_x;
    float *cp_dec_q;
    float *cp_dec_k;
    float *cp_dec_v;
    float *cp_dec_attn_out;
    float *cp_dec_gate;
    float *cp_dec_up;
    float *cp_dec_ffn_out;

    float *pref_residual;
    float *pref_x_norm;
    float *pref_q;
    float *pref_k;
    float *pref_v;
    float *pref_attn_out;
    float *pref_gate;
    float *pref_proj;
    int pref_seq_cap;
    float *pref_wq_f32;
    float *pref_wk_f32;
    float *pref_wv_f32;
    float *pref_wo_f32;
    float *pref_gate_up_f32;
    float *pref_down_f32;

    float *rope_cos;
    float *rope_sin;
    float *rope_inv_freq;
    int rope_cache_len;

    float *cp_rope_cos;
    float *cp_rope_sin;
    int cp_rope_cache_len;

    float *emb_tmp1;
    float *emb_tmp2;

    float *logits;

    int *codec_codes;
    int codec_frames;
    int codec_frames_cap;
    int *prev_tokens;
    int n_prev_tokens;
    int prev_tokens_cap;

    const int *tf_ref_codes;

    float  cp_roughness;
    int    cp_rough_built;

    float *ml_steer;
    int    ml_steer_layers;
    int    ml_steer_dim;
    float  ml_steer_weight;
    int    ml_steer_l0, ml_steer_l1;
    float  ml_steer_decay;
    int    ml_steer_frames;
    float  ml_steer_w_eff;

    float *audio_buf;
    int audio_samples;

    void *cached_tokenizer;

    float *cached_tts_pad_embed;
    float *cached_tts_bos_embed;
    float *cached_tts_eos_embed;

    struct {
        int *keys;
        float *values;
        uint32_t *access;
        int capacity;
        int count;
        uint32_t clock;
    } emb_cache;

    float *prev_input_embeds;
    int prev_prefill_len;

} qwen_tts_ctx_t;

#ifdef __cplusplus
extern "C" {
#endif

qwen_tts_ctx_t *qwen_tts_load(const char *model_dir);
qwen_tts_ctx_t *qwen_tts_load_ex(const char *model_dir, int silent, int use_int8, int use_int4);

void qwen_kleidi_prepack(qwen_tts_ctx_t *ctx);

int qwen_tts_serve_prefork(qwen_tts_ctx_t *ctx, int port, int workers,
                           int threads_per, int max_batch);

void qwen_tts_unload(qwen_tts_ctx_t *ctx);

void qwen_track_override(qwen_tts_ctx_t *ctx, void *ptr);

qwen_tts_ctx_t *qwen_tts_clone_for_worker(const qwen_tts_ctx_t *base);
void qwen_tts_free_clone(qwen_tts_ctx_t *ctx);

void qwen_tts_set_speaker(qwen_tts_ctx_t *ctx, int speaker_id);

void qwen_tts_set_language(qwen_tts_ctx_t *ctx, const char *language);

int qwen_tts_language_id(const char *name);

int qwen_tts_speaker_id(const char *name);

int qwen_tts_resolve_speaker(const qwen_tts_ctx_t *ctx, const char *name);

void qwen_tts_list_speakers(const qwen_tts_ctx_t *ctx);

void qwen_tts_set_audio_callback(qwen_tts_ctx_t *ctx, qwen_tts_audio_cb cb, void *userdata);

int qwen_tts_generate(qwen_tts_ctx_t *ctx, const char *text,
                      float **out_samples, int *out_n_samples);

int qwen_tts_generate_batch(qwen_tts_ctx_t *ctx, char **chunks, int nc,
                            float chunk_pause, float **out_samples, int *out_n_samples);

typedef struct {
    const char *text;
    int   speaker_id;
    int   language_id;
    float temperature;
    int   top_k;
    float top_p;
    float rep_penalty;
    uint32_t seed;
    int   greedy_warmup;
    int   want_stream;
} qwen_batch_req_t;

int qwen_tts_generate_batch_multi(qwen_tts_ctx_t *ctx,
                                  const qwen_batch_req_t *reqs, int nc,
                                  float **out_samples, int *out_n_samples);

typedef struct {
    void *ud;
    int (*next_job)(void *ud, qwen_batch_req_t *req, void **tag, int block);
    void (*on_done)(void *ud, void *tag, float *samples, int n_samples);
    void (*on_chunk)(void *ud, void *tag, float *samples, int n_samples);
    int (*running)(void *ud);
    int (*cancelled)(void *ud, void *tag);
    void (*on_reject)(void *ud, void *tag, const char *reason);
} qwen_batch_sink_t;

int qwen_tts_batch_max_prompt(void);
int qwen_tts_batch_max_frames(void);

void qwen_admit_probe_read(unsigned long long *seq, double *ts_ms, double *last_iter_ms);

int qwen_tts_serve_continuous(qwen_tts_ctx_t *ctx, int max_batch, qwen_batch_sink_t *sink);

int qwen_tts_write_wav(const char *path, const float *samples, int n_samples, int sample_rate);

int qwen_speech_encoder_load(qwen_tts_ctx_t *ctx);
int qwen_speech_encoder_encode(qwen_tts_ctx_t *ctx, const float *audio, int n_samples,
                                int **codes_out, int *n_frames_out);

#ifdef __cplusplus
}
#endif

#endif
