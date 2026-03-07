/*
 * qwen_tts.c - Qwen3-TTS Pure C Inference Engine
 * Main pipeline: text → Talker → Code Predictor → Speech Decoder → audio
 */

#include "qwen_tts.h"
#include "qwen_tts_kernels.h"
#include "qwen_tts_safetensors.h"
#include "qwen_tts_tokenizer.h"
#include "qwen_tts_audio.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>

int qwen_verbose = 0;

static double time_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

/* Language/Speaker mapping */
typedef struct { const char *name; int id; } lang_entry_t;
static const lang_entry_t lang_table[] = {
    {"Chinese", 2055}, {"English", 2050}, {"Japanese", 2058}, {"Korean", 2064},
    {"German", 2053}, {"French", 2061}, {"Russian", 2069}, {"Portuguese", 2071},
    {"Spanish", 2054}, {"Italian", 2070}, {NULL, -1}
};

int qwen_tts_language_id(const char *name) {
    if (!name) return -1;
    for (int i = 0; lang_table[i].name; i++)
        if (strcasecmp(name, lang_table[i].name) == 0) return lang_table[i].id;
    return -1;
}

typedef struct { const char *name; int id; } spk_entry_t;
static const spk_entry_t spk_table[] = {
    {"serena", 3066}, {"vivian", 3065}, {"uncle_fu", 3010}, {"ryan", 3061},
    {"aiden", 2861}, {"ono_anna", 2873}, {"sohee", 2864}, {"eric", 2875},
    {"dylan", 2878}, {NULL, -1}
};

int qwen_tts_speaker_id(const char *name) {
    if (!name) return -1;
    for (int i = 0; spk_table[i].name; i++)
        if (strcasecmp(name, spk_table[i].name) == 0) return spk_table[i].id;
    return -1;
}

/* JSON helpers */
static const char *json_find_key(const char *json, const char *key) {
    char pattern[256]; snprintf(pattern, sizeof(pattern), "\"%s\"", key);
    const char *p = strstr(json, pattern);
    if (!p) return NULL;
    p += strlen(pattern);
    while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r' || *p == ':') p++;
    return p;
}
static int json_get_int(const char *json, const char *key, int def) {
    const char *p = json_find_key(json, key); return p ? atoi(p) : def;
}
static float json_get_float(const char *json, const char *key, float def) {
    const char *p = json_find_key(json, key); return p ? (float)atof(p) : def;
}
static char *read_file(const char *path, long *out_len) {
    FILE *f = fopen(path, "r"); if (!f) return NULL;
    fseek(f, 0, SEEK_END); long len = ftell(f); fseek(f, 0, SEEK_SET);
    char *buf = (char *)malloc(len + 1); if (!buf) { fclose(f); return NULL; }
    if ((long)fread(buf, 1, len, f) != len) { free(buf); fclose(f); return NULL; }
    buf[len] = '\0'; fclose(f); if (out_len) *out_len = len; return buf;
}

/* Config loading */
static int load_config(qwen_tts_ctx_t *ctx) {
    char path[1024]; snprintf(path, sizeof(path), "%s/config.json", ctx->model_dir);
    long len; char *json = read_file(path, &len); if (!json) return -1;
    qwen_tts_config_t *c = &ctx->config;

    const char *tc_start = strstr(json, "\"talker_config\"");
    if (!tc_start) { free(json); return -1; }
    const char *p = strchr(tc_start, '{'); if (!p) { free(json); return -1; }
    
    /* Find the closing brace of talker_config (including nested code_predictor_config) */
    int brace = 1; const char *tc_end = p + 1;
    while (*tc_end && brace > 0) { if (*tc_end == '{') brace++; else if (*tc_end == '}') brace--; tc_end++; }
    
    long tc_len = tc_end - p; char *tc_json = (char *)malloc(tc_len + 1);
    memcpy(tc_json, p, tc_len); tc_json[tc_len] = '\0';
    
    /* Build a flat version of talker_config with nested objects removed.
     * This prevents json_find_key from matching keys inside nested objects
     * like code_predictor_config (whose fields shadow talker-level fields). */
    char *talker_only_json = strdup(tc_json);
    {
        /* Repeatedly find and blank out nested {...} blocks */
        char *scan = talker_only_json;
        while (1) {
            /* Find next key whose value is an object (opening brace) */
            char *q = scan;
            char *nested_open = NULL;
            while (*q) {
                if (*q == '"') {
                    /* Skip string */
                    q++;
                    while (*q && *q != '"') { if (*q == '\\') q++; q++; }
                    if (*q) q++;
                    /* After key string, skip whitespace and colon */
                    while (*q == ' ' || *q == '\t' || *q == '\n' || *q == '\r' || *q == ':') q++;
                    if (*q == '{') { nested_open = q; break; }
                } else {
                    q++;
                }
            }
            if (!nested_open) break;
            /* Find matching close brace */
            int depth = 1;
            char *r = nested_open + 1;
            while (*r && depth > 0) {
                if (*r == '{') depth++;
                else if (*r == '}') depth--;
                r++;
            }
            /* Blank out the nested object (replace with spaces) */
            memset(nested_open, ' ', r - nested_open);
            scan = r;
        }
    }
    
    c->text_hidden_size = json_get_int(talker_only_json, "text_hidden_size", 2048);
    c->hidden_size = json_get_int(talker_only_json, "hidden_size", 1024);
    c->num_layers = json_get_int(talker_only_json, "num_hidden_layers", 28);
    c->num_heads = json_get_int(talker_only_json, "num_attention_heads", 16);
    c->num_kv_heads = json_get_int(talker_only_json, "num_key_value_heads", 8);
    c->head_dim = json_get_int(talker_only_json, "head_dim", 128);
    c->intermediate_size = json_get_int(talker_only_json, "intermediate_size", 3072);
    c->codec_vocab_size = json_get_int(talker_only_json, "codec_vocab_size", 3072);
    c->codebook_size = json_get_int(talker_only_json, "codebook_size", 2048);
    c->rms_norm_eps = json_get_float(talker_only_json, "rms_norm_eps", 1e-6f);
    c->rope_theta = json_get_float(talker_only_json, "rope_theta", 1e6f);
    free(talker_only_json);
    
    fprintf(stderr, "[CONFIG] After talker parse: num_layers=%d\n", c->num_layers);

    const char *cp_start = strstr(tc_json, "\"code_predictor_config\"");
    if (cp_start) {
        const char *cp_open = strchr(cp_start, '{');
        if (cp_open) {
            const char *cp_close = strchr(cp_open, '}');
            if (cp_close) {
                long cp_len = cp_close - cp_open + 1; char *cp_json = (char *)malloc(cp_len + 1);
                memcpy(cp_json, cp_open, cp_len); cp_json[cp_len] = '\0';
                c->cp_hidden_size = json_get_int(cp_json, "hidden_size", 1024);
                c->cp_num_layers = json_get_int(cp_json, "num_hidden_layers", 5);
                fprintf(stderr, "[CONFIG] After CP parse: cp_num_layers=%d, talker num_layers=%d\n", c->cp_num_layers, c->num_layers);
                c->cp_num_heads = json_get_int(cp_json, "num_attention_heads", 16);
                c->cp_num_kv_heads = json_get_int(cp_json, "num_key_value_heads", 8);
                c->cp_head_dim = json_get_int(cp_json, "head_dim", 128);
                c->cp_intermediate_size = json_get_int(cp_json, "intermediate_size", 3072);
                free(cp_json);
            }
        }
    }
    free(tc_json); free(json);

    snprintf(path, sizeof(path), "%s/speech_tokenizer/config.json", ctx->model_dir);
    json = read_file(path, &len);
    if (!json) {
        snprintf(path, sizeof(path), "speech_tokenizer_config.json");
        json = read_file(path, &len);
    }
    if (json) {
        const char *dc_start = strstr(json, "\"decoder_config\"");
        if (dc_start) {
            const char *dc_open = strchr(dc_start, '{');
            if (dc_open) {
                const char *dc_close = dc_open + 1; int brace = 1;
                while (*dc_close && brace > 0) { if (*dc_close == '{') brace++; else if (*dc_close == '}') brace--; dc_close++; }
                long dc_len = dc_close - dc_open; char *dc_json = (char *)malloc(dc_len + 1);
                memcpy(dc_json, dc_open, dc_len); dc_json[dc_len] = '\0';
                c->dec_hidden_size = json_get_int(dc_json, "hidden_size", 512);
                c->dec_num_layers = json_get_int(dc_json, "num_hidden_layers", 8);
                c->dec_latent_dim = json_get_int(dc_json, "latent_dim", 1024);
                c->dec_codebook_dim = json_get_int(dc_json, "codebook_dim", 512);
                c->dec_decoder_dim = json_get_int(dc_json, "decoder_dim", 1536);
                c->dec_num_heads = json_get_int(dc_json, "num_attention_heads", 16);
                c->dec_head_dim = json_get_int(dc_json, "head_dim", 64);
                c->dec_intermediate_size = json_get_int(dc_json, "intermediate_size", 1024);
                c->dec_num_quantizers = json_get_int(dc_json, "num_quantizers", 16);
                c->dec_sliding_window = json_get_int(dc_json, "sliding_window", 72);
                c->dec_rope_theta = json_get_float(dc_json, "rope_theta", 10000.0f);
                c->dec_rms_norm_eps = json_get_float(dc_json, "rms_norm_eps", 1e-5f);
                free(dc_json);
            }
        }
        free(json);
    }
    c->codebook_size = QWEN_TTS_CODEBOOK_SIZE;
    c->codec_vocab_size = QWEN_TTS_CODEC_VOCAB_SIZE;
    return 0;
}

static inline float bf16_to_f32(uint16_t bf) {
    uint32_t bits = (uint32_t)bf << 16; float val; memcpy(&val, &bits, sizeof(float)); return val;
}

/* Use centralized NEON+multi-threaded matvec from qwen_tts_kernels.c */
#define matvec_bf16 qwen_matvec_bf16

/* External functions */
extern int qwen_talker_load(qwen_tts_ctx_t *ctx);
extern int qwen_cp_load(qwen_tts_ctx_t *ctx);
extern int qwen_speech_decoder_load(qwen_tts_ctx_t *ctx);
extern int qwen_talker_prefill(qwen_tts_ctx_t *ctx, float *input_embeds, int seq_len);
extern int qwen_talker_step(qwen_tts_ctx_t *ctx, float *embed, float *hidden_out);
extern int qwen_cp_predict(qwen_tts_ctx_t *ctx, float *talker_hidden, int code0, int *out_codes);
extern int qwen_speech_decoder_decode(qwen_tts_ctx_t *ctx, const int *codes, int n_frames, float **audio_out, int *n_samples);
extern int qwen_tts_sample(float *logits, int vocab_size, float temp, int top_k, float top_p, float rep_penalty, int *prev_tokens, int n_prev);
extern void qwen_set_seed(uint32_t seed);

/* Embed a single text token: text_embedding → text_projection(SiLU) → out[hidden] */
static void embed_one_text_token(qwen_tts_ctx_t *ctx, int tid, float *out) {
    int th = ctx->config.text_hidden_size, h = ctx->config.hidden_size;
    float *text_emb = (float *)malloc(th * sizeof(float));
    float *fc1_out = (float *)malloc(th * sizeof(float));
    const uint16_t *emb = ctx->tok_embeddings_bf16 + (int64_t)tid * th;
    for (int j = 0; j < th; j++) text_emb[j] = bf16_to_f32(emb[j]);
    if (ctx->text_proj_fc1_bf16 && ctx->text_proj_fc2_bf16) {
        matvec_bf16(fc1_out, ctx->text_proj_fc1_bf16, text_emb, th, th);
        if (ctx->text_proj_fc1_bias) for (int j = 0; j < th; j++) fc1_out[j] += ctx->text_proj_fc1_bias[j];
        for (int j = 0; j < th; j++) fc1_out[j] = fc1_out[j] / (1.0f + expf(-fc1_out[j])); /* SiLU */
        matvec_bf16(out, ctx->text_proj_fc2_bf16, fc1_out, h, th);
        if (ctx->text_proj_fc2_bias) for (int j = 0; j < h; j++) out[j] += ctx->text_proj_fc2_bias[j];
    } else {
        memcpy(out, text_emb, h * sizeof(float));
    }
    free(text_emb); free(fc1_out);
}

/* Load model */
qwen_tts_ctx_t *qwen_tts_load(const char *model_dir) {
    qwen_tts_ctx_t *ctx = (qwen_tts_ctx_t *)calloc(1, sizeof(qwen_tts_ctx_t)); if (!ctx) return NULL;
    strncpy(ctx->model_dir, model_dir, sizeof(ctx->model_dir) - 1);
    ctx->temperature = 0.9f; ctx->top_k = 50; ctx->top_p = 1.0f; ctx->rep_penalty = 1.05f;
    ctx->max_tokens = 8192; ctx->cp_temperature = 0.0f; ctx->cp_top_k = 1;
    /* Default speaker: Ryan (3061) - native English speaker
     * Serena (3066) and others are Chinese speakers which may cause issues with English */
    ctx->speaker_id = 3061; ctx->language_id = -1; ctx->seed = (uint32_t)time(NULL);
    ctx->silent = 0; ctx->debug = 0;

    /* Load config from model_dir or current dir */
    char config_path[1024];
    snprintf(config_path, sizeof(config_path), "%s/config.json", ctx->model_dir);
    if (load_config(ctx) != 0) {
        /* Try current directory */
        snprintf(config_path, sizeof(config_path), "config.json");
        if (load_config(ctx) != 0) { free(ctx); return NULL; }
    }
    
    qwen_tts_config_t *c = &ctx->config;
    if (!ctx->silent) {
        fprintf(stderr, "Config: hidden=%d text_hidden=%d layers=%d heads=%d/%d head_dim=%d inter=%d\n",
                c->hidden_size, c->text_hidden_size, c->num_layers, c->num_heads, c->num_kv_heads, c->head_dim, c->intermediate_size);
        fprintf(stderr, "  Code Predictor: hidden=%d layers=%d heads=%d head_dim=%d\n",
                c->cp_hidden_size, c->cp_num_layers, c->cp_num_heads, c->cp_head_dim);
        fprintf(stderr, "  Codec: vocab=%d codebooks=%d entries=%d\n", c->codec_vocab_size, c->dec_num_quantizers, c->codebook_size);
    }

    /* Load safetensors using qwen-asr loader (mmap-based, working) */
    ctx->safetensors = multi_safetensors_open(ctx->model_dir);
    if (!ctx->safetensors) {
        fprintf(stderr, "Error: Failed to load model from %s\n", ctx->model_dir);
        free(ctx); return NULL;
    }
    ctx->speech_safetensors = ctx->safetensors; /* Same file for now */

    if (!ctx->silent) fprintf(stderr, "Threads: %d\n", qwen_get_threads());

    double t0 = time_ms();
    if (qwen_talker_load(ctx) != 0 || qwen_cp_load(ctx) != 0 || qwen_speech_decoder_load(ctx) != 0) {
        multi_safetensors_close(ctx->safetensors);
        if (ctx->speech_safetensors != ctx->safetensors) multi_safetensors_close(ctx->speech_safetensors);
        free(ctx); return NULL;
    }
    if (!ctx->silent) fprintf(stderr, "Model loaded in %.0f ms\n", time_ms() - t0);
    return ctx;
}

void qwen_tts_unload(qwen_tts_ctx_t *ctx) {
    if (!ctx) return;
    /* Free malloc'd fused weights (gate_up are the only malloc'd weight copies) */
    for (int i = 0; i < ctx->config.num_layers; i++) free(ctx->layers[i].gate_up_fused_bf16);
    for (int i = 0; i < ctx->config.cp_num_layers; i++) free(ctx->cp_layers[i].gate_up_fused_bf16);
    /* Free malloc'd codebooks (EMA-reconstructed, not from safetensors) */
    for (int i = 0; i < 16; i++) free(ctx->speech_dec.codebook[i]);
    free(ctx->speech_dec.pre_layers);
    free(ctx->speech_dec.rope_cos); free(ctx->speech_dec.rope_sin);
    /* Close safetensors (all get_bf16/get_f32 pointers point into this data) */
    multi_safetensors_close(ctx->safetensors);
    if (ctx->speech_safetensors != ctx->safetensors)
        multi_safetensors_close(ctx->speech_safetensors);
    /* Free runtime buffers */
    free(ctx->kv_cache_k); free(ctx->kv_cache_v); free(ctx->cp_kv_k); free(ctx->cp_kv_v);
    free(ctx->dec_x); free(ctx->dec_x_norm); free(ctx->dec_q); free(ctx->dec_k); free(ctx->dec_v);
    free(ctx->dec_attn_out); free(ctx->dec_proj_out); free(ctx->dec_gate); free(ctx->dec_up); free(ctx->dec_ffn_out);
    free(ctx->cp_dec_x); free(ctx->cp_dec_q); free(ctx->cp_dec_k); free(ctx->cp_dec_v);
    free(ctx->cp_dec_attn_out); free(ctx->cp_dec_gate); free(ctx->cp_dec_up); free(ctx->cp_dec_ffn_out);
    free(ctx->pref_x); free(ctx->pref_x_norm); free(ctx->pref_q); free(ctx->pref_k); free(ctx->pref_v);
    free(ctx->pref_attn_out); free(ctx->pref_ffn_out);
    free(ctx->rope_cos); free(ctx->rope_sin); free(ctx->rope_inv_freq);
    free(ctx->cp_rope_cos); free(ctx->cp_rope_sin);
    free(ctx->logits); free(ctx->codec_codes); free(ctx->prev_tokens); free(ctx->audio_buf);
    free(ctx);
}

void qwen_tts_set_speaker(qwen_tts_ctx_t *ctx, int speaker_id) { ctx->speaker_id = speaker_id; }
void qwen_tts_set_language(qwen_tts_ctx_t *ctx, const char *language) {
    ctx->language_id = qwen_tts_language_id(language);
    /* Set appropriate speaker based on language */
    if (ctx->language_id == QWEN_TTS_LANG_ENGLISH) {
        ctx->speaker_id = 3061;  /* Ryan - native English */
    } else if (ctx->language_id == QWEN_TTS_LANG_CHINESE) {
        ctx->speaker_id = 3066;  /* Serena - native Chinese */
    } else if (ctx->language_id == QWEN_TTS_LANG_JAPANESE) {
        ctx->speaker_id = 2873;  /* Ono Anna - native Japanese */
    } else if (ctx->language_id == QWEN_TTS_LANG_KOREAN) {
        ctx->speaker_id = 2864;  /* Sohee - native Korean */
    }
    /* For other languages, keep current speaker or default to Ryan */
}

/* Codec embedding lookup */
static void lookup_codec_embed(qwen_tts_ctx_t *ctx, int token_id, float *out) {
    int h = ctx->config.hidden_size;
    if (token_id < 0 || token_id >= ctx->config.codec_vocab_size) { memset(out, 0, h * sizeof(float)); return; }
    const uint16_t *emb = ctx->codec_embedding_bf16 + (int64_t)token_id * h;
    for (int j = 0; j < h; j++) out[j] = bf16_to_f32(emb[j]);
}

/* Generate speech from text.
 *
 * DUAL-TRACK ARCHITECTURE (matching official Qwen3-TTS Python):
 *
 * The full template string "<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
 * is BPE-encoded as raw text (NOT using special token IDs).
 * Then split: role_prefix = first 3 BPE tokens, text_content = tokens[3:-5], suffix discarded.
 *
 * NON-STREAMING PREFILL (default mode):
 *   [role_0, role_1, role_2]                     -- text-only, no codec pairing
 *   [tts_pad+codec_0, ..., tts_pad+codec_{K-3}]  -- pad+codec prefix (without last 2)
 *   [tts_bos + codec_pad]                         -- bos paired with codec pad
 *   [text_0+codec_pad, ..., text_N+codec_pad]     -- all text content with codec_pad
 *   [tts_eos + codec_pad]                         -- eos paired with codec_pad
 *   [tts_pad + codec_bos]                         -- final: pad + bos
 *
 * Generation: every frame gets tts_pad (text side) + codec_embed(sum_all_codes)
 */
int qwen_tts_generate(qwen_tts_ctx_t *ctx, const char *text, float **out_samples, int *out_n_samples) {
    double t_start = time_ms();
    int h = ctx->config.hidden_size;
    qwen_set_seed(ctx->seed);

    /* Build token sequence matching Python:
     * [<|im_start|>, assistant, \n, ...BPE(text)..., <|im_end|>, \n, <|im_start|>, assistant, \n]
     * Special tokens use their IDs directly; only the user text is BPE-encoded.
     * Role prefix = [:3], text_content = [3:-5], suffix [-5:] discarded.
     */
    int32_t *text_tokens = NULL;
    int text_token_len = 0;
    qwen_tokenizer_t *tok = qwen_tokenizer_load(ctx->model_dir);
    if (tok) {
        text_tokens = qwen_tokenizer_encode(tok, text, &text_token_len);
        qwen_tokenizer_free(tok);
    }
    if (!text_tokens || text_token_len == 0) {
        fprintf(stderr, "Error: text tokenization failed\n");
        free(text_tokens);
        return -1;
    }

    /* Assemble: [im_start, assistant, \n] + text_tokens + [im_end, \n, im_start, assistant, \n] */
    int role_len = 3;
    int suffix_len = 5;
    int all_len = role_len + text_token_len + suffix_len;
    int32_t *all_tokens = (int32_t *)malloc(all_len * sizeof(int32_t));
    int pos_t = 0;
    all_tokens[pos_t++] = 151644;  /* <|im_start|> */
    all_tokens[pos_t++] = 77091;   /* assistant */
    all_tokens[pos_t++] = 198;     /* \n */
    memcpy(all_tokens + pos_t, text_tokens, text_token_len * sizeof(int32_t));
    pos_t += text_token_len;
    all_tokens[pos_t++] = 151645;  /* <|im_end|> */
    all_tokens[pos_t++] = 198;     /* \n */
    all_tokens[pos_t++] = 151644;  /* <|im_start|> */
    all_tokens[pos_t++] = 77091;   /* assistant */
    all_tokens[pos_t++] = 198;     /* \n */
    free(text_tokens);

    int text_content_len = all_len - role_len - suffix_len;  /* = text_token_len */

    if (!ctx->silent) {
        fprintf(stderr, "Text: \"%s\" (template: %d BPE tokens, text_content: %d)\n",
                text, all_len, text_content_len);
    }

    /* Build codec-side prefix:
     * With language: [THINK, THINK_BOS, language_id, THINK_EOS, speaker, PAD, BOS]
     * Without language: [NO_THINK, THINK_BOS, THINK_EOS, speaker, PAD, BOS]
     */
    int codec_tokens[16];
    int codec_len = 0;
    if (ctx->language_id >= 0) {
        codec_tokens[codec_len++] = QWEN_TTS_CODEC_THINK;
        codec_tokens[codec_len++] = QWEN_TTS_CODEC_THINK_BOS;
        codec_tokens[codec_len++] = ctx->language_id;
        codec_tokens[codec_len++] = QWEN_TTS_CODEC_THINK_EOS;
    } else {
        codec_tokens[codec_len++] = QWEN_TTS_CODEC_NO_THINK;
        codec_tokens[codec_len++] = QWEN_TTS_CODEC_THINK_BOS;
        codec_tokens[codec_len++] = QWEN_TTS_CODEC_THINK_EOS;
    }
    codec_tokens[codec_len++] = ctx->speaker_id;
    codec_tokens[codec_len++] = QWEN_TTS_CODEC_PAD;
    codec_tokens[codec_len++] = QWEN_TTS_CODEC_BOS;

    /* Pre-compute key embeddings */
    float *tts_pad_embed = (float *)malloc(h * sizeof(float));
    float *tts_bos_embed = (float *)malloc(h * sizeof(float));
    float *tts_eos_embed = (float *)malloc(h * sizeof(float));
    embed_one_text_token(ctx, QWEN_TTS_TTS_PAD, tts_pad_embed);
    embed_one_text_token(ctx, QWEN_TTS_TTS_BOS, tts_bos_embed);
    embed_one_text_token(ctx, QWEN_TTS_TTS_EOS, tts_eos_embed);

    float *codec_pad_embed = (float *)malloc(h * sizeof(float));
    float *codec_bos_embed = (float *)malloc(h * sizeof(float));
    lookup_codec_embed(ctx, QWEN_TTS_CODEC_PAD, codec_pad_embed);
    lookup_codec_embed(ctx, QWEN_TTS_CODEC_BOS, codec_bos_embed);

    /*
     * Build prefill: role(3) + pad_codec(codec_len-1) + text_content+eos(N+1) + final(1)
     *
     * Section 1: Role prefix (3 positions) - text-only, NO codec pairing
     * Section 2: tts_pad*(codec_len-2) + tts_bos  paired with  codec[0..codec_len-2]
     * Section 3: text_content[0..N-1] + tts_eos  paired with  codec_pad * (N+1)
     * Section 4: tts_pad + codec_bos  (1 position)
     */
    int sec2_len = codec_len - 1;  /* codec tokens without the last (BOS) */
    int sec3_len = text_content_len + 1;  /* text tokens + tts_eos */
    int prefill_len = role_len + sec2_len + sec3_len + 1;

    float *input_embeds = (float *)calloc((int64_t)prefill_len * h, sizeof(float));
    float *tmp_embed = (float *)malloc(h * sizeof(float));
    int pos = 0;

    /* Section 1: Role prefix (text-only, no codec) */
    for (int i = 0; i < role_len; i++) {
        embed_one_text_token(ctx, all_tokens[i], input_embeds + (int64_t)pos * h);
        if (ctx->debug) {
            float *e = input_embeds + (int64_t)pos * h;
            fprintf(stderr, "[PROMPT] pos=%d role token=%d embed[:5]=[%.4f,%.4f,%.4f,%.4f,%.4f]\n",
                    pos, all_tokens[i], e[0], e[1], e[2], e[3], e[4]);
        }
        pos++;
    }

    /* Section 2: tts_pad/tts_bos + codec prefix (without last element) */
    for (int i = 0; i < sec2_len; i++) {
        float *dst = input_embeds + (int64_t)pos * h;
        /* Text side: tts_pad for all except last which is tts_bos */
        if (i < sec2_len - 1) {
            memcpy(dst, tts_pad_embed, h * sizeof(float));
        } else {
            memcpy(dst, tts_bos_embed, h * sizeof(float));
        }
        /* Codec side: codec_tokens[i] */
        lookup_codec_embed(ctx, codec_tokens[i], tmp_embed);
        for (int j = 0; j < h; j++) dst[j] += tmp_embed[j];
        pos++;
    }

    /* Section 3: text content + tts_eos, all paired with codec_pad */
    for (int i = 0; i < sec3_len; i++) {
        float *dst = input_embeds + (int64_t)pos * h;
        /* Text side */
        if (i < text_content_len) {
            embed_one_text_token(ctx, all_tokens[role_len + i], dst);
        } else {
            /* Last position of section 3: tts_eos */
            memcpy(dst, tts_eos_embed, h * sizeof(float));
        }
        /* Codec side: codec_pad */
        for (int j = 0; j < h; j++) dst[j] += codec_pad_embed[j];
        pos++;
    }

    /* Section 4: tts_pad + codec_bos (final position) */
    {
        float *dst = input_embeds + (int64_t)pos * h;
        memcpy(dst, tts_pad_embed, h * sizeof(float));
        for (int j = 0; j < h; j++) dst[j] += codec_bos_embed[j];
        pos++;
    }

    free(all_tokens);
    free(tmp_embed);
    free(tts_bos_embed);
    free(tts_eos_embed);

    free(codec_pad_embed);
    free(codec_bos_embed);

    if (!ctx->silent) {
        fprintf(stderr, "Speaker: %d, Language: %d\n", ctx->speaker_id, ctx->language_id);
        fprintf(stderr, "Prefill: %d positions (role=%d, codec=%d, text+eos=%d, final=1)\n",
                prefill_len, role_len, sec2_len, sec3_len);
    }

    /* Debug: check speech decoder weights before prefill */
    if (ctx->debug && ctx->speech_dec.pre_conv_weight) {
        fprintf(stderr, "[CORR] pre-prefill: pre_conv_w[0]=%.6f\n", ctx->speech_dec.pre_conv_weight[0]);
    }

    /* Talker prefill */
    double t_prefill = time_ms();
    if (qwen_talker_prefill(ctx, input_embeds, prefill_len) != 0) {
        free(input_embeds); free(tts_pad_embed);
        return -1;
    }
    free(input_embeds);
    if (!ctx->silent) fprintf(stderr, "  Prefill: %.0f ms\n", time_ms() - t_prefill);

    /* Debug: check speech decoder weights after prefill */
    if (ctx->debug && ctx->speech_dec.pre_conv_weight) {
        fprintf(stderr, "[CORR] post-prefill: pre_conv_w[0]=%.6f\n", ctx->speech_dec.pre_conv_weight[0]);
    }

    /* Get hidden state from last prefill position (apply final norm) */
    float *last_hidden = (float *)malloc(h * sizeof(float));
    qwen_rms_norm(last_hidden, ctx->dec_x, ctx->talker_norm, 1, h, ctx->config.rms_norm_eps);

    /* Autoregressive generation */
    int max_frames = ctx->max_tokens;
    ctx->codec_codes = (int *)realloc(ctx->codec_codes, (int64_t)max_frames * 16 * sizeof(int));
    ctx->codec_frames = 0;
    ctx->prev_tokens = (int *)realloc(ctx->prev_tokens, max_frames * sizeof(int));
    ctx->n_prev_tokens = 0;
    ctx->logits = (float *)realloc(ctx->logits, ctx->config.codec_vocab_size * sizeof(float));

    double t_cp_total = 0;
    float *step_embed = (float *)malloc(h * sizeof(float));

    for (int frame = 0; frame < max_frames; frame++) {
        /* Codec head: logits = codec_head @ last_hidden */
        matvec_bf16(ctx->logits, ctx->codec_head_bf16, last_hidden, ctx->config.codec_vocab_size, h);

        /* Clip logits */
        for (int t = 0; t < ctx->config.codec_vocab_size; t++) {
            if (ctx->logits[t] > 100.0f) ctx->logits[t] = 100.0f;
            if (ctx->logits[t] < -100.0f) ctx->logits[t] = -100.0f;
        }

        /* Suppress special tokens (>= 2048) except EOS (2150) */
        for (int t = 2048; t < ctx->config.codec_vocab_size; t++)
            if (t != QWEN_TTS_CODEC_EOS) ctx->logits[t] = -1e30f;

        /* Suppress EOS for first 2 frames */
        if (frame < 2) ctx->logits[QWEN_TTS_CODEC_EOS] = -1e30f;

        /* Debug logging */
        if (ctx->debug && frame < 30) {
            float eos_logit = ctx->logits[QWEN_TTS_CODEC_EOS];
            int eos_rank = 0;
            for (int t = 0; t < ctx->config.codec_vocab_size; t++)
                if (ctx->logits[t] > eos_logit) eos_rank++;
            fprintf(stderr, "  [frame %d] EOS logit=%.2f rank=%d\n", frame, eos_logit, eos_rank);
        }

        /* Sample code0 */
        int code0 = qwen_tts_sample(ctx->logits, ctx->config.codec_vocab_size,
                                     ctx->temperature, ctx->top_k, ctx->top_p,
                                     ctx->rep_penalty, ctx->prev_tokens, ctx->n_prev_tokens);

        if (code0 == QWEN_TTS_CODEC_EOS) {
            if (!ctx->silent) fprintf(stderr, "  EOS at frame %d\n", frame);
            break;
        }

        ctx->prev_tokens[ctx->n_prev_tokens++] = code0;

        /* Code Predictor: generate codebooks 1-15 */
        int codes[16]; codes[0] = code0;
        double t_cp_start = time_ms();
        qwen_cp_predict(ctx, last_hidden, code0, codes + 1);
        t_cp_total += time_ms() - t_cp_start;

        memcpy(ctx->codec_codes + (int64_t)ctx->codec_frames * 16, codes, 16 * sizeof(int));
        ctx->codec_frames++;

        /* Debug: dump codes for all frames */
        if (ctx->debug) {
            fprintf(stderr, "  [frame %d] codes:", frame);
            for (int g = 0; g < 16; g++) fprintf(stderr, " %d", codes[g]);
            fprintf(stderr, "\n");
        }

        /* Debug: check for weight corruption */
        if (ctx->debug && frame == 0 && ctx->speech_dec.pre_conv_weight) {
            fprintf(stderr, "[CORR] post-frame0: pre_conv_w[0]=%.6f\n", ctx->speech_dec.pre_conv_weight[0]);
        }

        if (!ctx->silent && frame % 50 == 0 && frame > 0)
            fprintf(stderr, "\r  Frame %d/%d (%.1fs audio)...", frame, max_frames, frame / 12.5);

        /* Build next input embedding (non-streaming mode):
         * codec_side: codec_embed(code0) + sum of CP codec_embeds(codes 1-15)
         * text_side: always tts_pad (all text was in prefill)
         */
        lookup_codec_embed(ctx, code0, step_embed);
        for (int g = 0; g < 15; g++) {
            int code_g = codes[g + 1];
            if (ctx->cp_codec_emb_bf16[g] && code_g >= 0 && code_g < ctx->config.codebook_size) {
                const uint16_t *emb = ctx->cp_codec_emb_bf16[g] + (int64_t)code_g * h;
                for (int j = 0; j < h; j++) step_embed[j] += bf16_to_f32(emb[j]);
            }
        }

        /* Text side: always tts_pad in non-streaming mode */
        for (int j = 0; j < h; j++) step_embed[j] += tts_pad_embed[j];

        /* Talker step */
        if (qwen_talker_step(ctx, step_embed, last_hidden) != 0) {
            free(step_embed); free(last_hidden); free(tts_pad_embed);
            return -1;
        }
    }

    free(step_embed);
    free(last_hidden);
    free(tts_pad_embed);

    double t_talker_end = time_ms();
    if (!ctx->silent) {
        fprintf(stderr, "\n  Generated %d frames (%.1fs audio)\n", ctx->codec_frames, ctx->codec_frames / 12.5);
        fprintf(stderr, "  Talker: %.0f ms, Code Predictor: %.0f ms\n",
                t_talker_end - t_prefill - t_cp_total, t_cp_total);
    }

    /* Speech decoder */
    if (ctx->codec_frames == 0) {
        *out_samples = NULL; *out_n_samples = 0;
        return 0;
    }

    double t_dec_start = time_ms();
    float *audio; int n_samples;
    if (qwen_speech_decoder_decode(ctx, ctx->codec_codes, ctx->codec_frames, &audio, &n_samples) != 0)
        return -1;
    if (!ctx->silent)
        fprintf(stderr, "  Speech decoder: %.0f ms\n", time_ms() - t_dec_start);

    *out_samples = audio;
    *out_n_samples = n_samples;

    if (!ctx->silent) {
        fprintf(stderr, "Audio: %.1fs generated in %.1fs (%.1fx realtime)\n",
                (float)n_samples / 24000.0f, (time_ms() - t_start) / 1000.0f,
                (float)n_samples / 24000.0f / ((time_ms() - t_start) / 1000.0f));
    }

    return 0;
}
