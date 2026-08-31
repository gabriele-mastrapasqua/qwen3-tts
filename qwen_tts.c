/* qwen_tts.c - Qwen3-TTS Pure C Inference Engine */
#include "qwen_tts.h"
#include "qwen_tts_voice_clone.h"
#include "qwen_tts_kernels.h"
#include "ingot/safetensors.h"
#include "qwen_tts_tokenizer.h"
#include "qwen_tts_audio.h"
#include "qwen_tts_batch.h"
#include "qwen_tts_thread.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <unistd.h>
#include <stdatomic.h>

int qwen_verbose = 0;

const char *qwen_tts_eos_strategy_name(int strategy) {
    switch (strategy) {
        case QWEN_EOS_OFF:  return "off";
        case QWEN_EOS_V1:   return "v1";
        case QWEN_EOS_V2:   return "v2";
        case QWEN_EOS_TOPK: return "topk";
        default:            return "?";
    }
}

int qwen_tts_eos_strategy_parse(const char *name) {
    if (!name) return -1;
    if (!strcmp(name, "off")  || !strcmp(name, "none")) return QWEN_EOS_OFF;
    if (!strcmp(name, "v1")   || !strcmp(name, "1"))    return QWEN_EOS_V1;
    if (!strcmp(name, "v2")   || !strcmp(name, "2"))    return QWEN_EOS_V2;
    if (!strcmp(name, "topk") || !strcmp(name, "ref"))  return QWEN_EOS_TOPK;
    return -1;
}

static double time_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

static _Atomic unsigned long long g_admit_seq = 0;
static _Atomic double g_admit_ts = 0.0;
static _Atomic double g_admit_last_iter = 0.0;

void qwen_admit_probe_read(unsigned long long *seq, double *ts_ms, double *last_iter_ms) {
    if (seq)          *seq          = atomic_load_explicit(&g_admit_seq, memory_order_relaxed);
    if (ts_ms)        *ts_ms        = atomic_load_explicit(&g_admit_ts, memory_order_relaxed);
    if (last_iter_ms) *last_iter_ms = atomic_load_explicit(&g_admit_last_iter, memory_order_relaxed);
}

double qwen_mono_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

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

int qwen_tts_resolve_speaker(const qwen_tts_ctx_t *ctx, const char *name) {
    if (!name) return -1;
    if (ctx) {
        for (int i = 0; i < ctx->spk_count; i++)
            if (strcasecmp(name, ctx->spk_names[i]) == 0) return ctx->spk_slots[i];
    }
    return qwen_tts_speaker_id(name);
}

static void parse_spk_id_table(qwen_tts_ctx_t *ctx, const char *cfg_raw);

int qwen_tts_load_speaker_map(qwen_tts_ctx_t *ctx, const char *path) {
    if (!ctx || !path) return -1;
    char cfg[4096];
    struct stat st;
    if (stat(path, &st) == 0 && S_ISDIR(st.st_mode))
        snprintf(cfg, sizeof(cfg), "%s/config.json", path);
    else
        snprintf(cfg, sizeof(cfg), "%s", path);

    FILE *f = fopen(cfg, "rb");
    if (!f) { fprintf(stderr, "Error: --speaker-map: cannot open %s\n", cfg); return -1; }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    if (n <= 0) { fclose(f); fprintf(stderr, "Error: --speaker-map: %s is empty\n", cfg); return -1; }
    char *raw = (char *)malloc((size_t)n + 1);
    if (!raw) { fclose(f); return -1; }
    if (fread(raw, 1, (size_t)n, f) != (size_t)n) { free(raw); fclose(f); return -1; }
    raw[n] = 0; fclose(f);

    for (int i = 0; i < ctx->spk_count; i++) free(ctx->spk_names[i]);
    free(ctx->spk_names); free(ctx->spk_slots);
    ctx->spk_names = NULL; ctx->spk_slots = NULL; ctx->spk_count = 0;

    parse_spk_id_table(ctx, raw);
    free(raw);
    if (ctx->spk_count <= 0) {
        fprintf(stderr, "Error: --speaker-map: no \"spk_id\" table in %s\n", cfg);
        return -1;
    }
    return ctx->spk_count;
}

void qwen_tts_list_speakers(const qwen_tts_ctx_t *ctx) {
    if (ctx && ctx->spk_count > 0) {
        fprintf(stderr, "Speakers declared by this model (config.json talker_config.spk_id): %d\n",
                ctx->spk_count);
        for (int i = 0; i < ctx->spk_count; i++)
            fprintf(stderr, "  %-28s slot %d\n", ctx->spk_names[i], ctx->spk_slots[i]);
        fprintf(stderr,
                "Note: a pool's config lists every TRAINED slot. The voices actually shipped are\n"
                "      those in voices.json next to the model, which is usually a subset.\n");
        return;
    }
    fprintf(stderr, "This model declares no speakers of its own; built-in presets:\n");
    for (int i = 0; spk_table[i].name; i++)
        fprintf(stderr, "  %-28s slot %d\n", spk_table[i].name, spk_table[i].id);
}

static void parse_spk_id_table(qwen_tts_ctx_t *ctx, const char *cfg_raw) {
    const char *p = strstr(cfg_raw, "\"spk_id\"");
    if (!p) return;
    p = strchr(p + 8, '{');
    if (!p) return;
    p++;

    int cap = 16;
    ctx->spk_names = (char **)malloc((size_t)cap * sizeof(char *));
    ctx->spk_slots = (int *)malloc((size_t)cap * sizeof(int));
    if (!ctx->spk_names || !ctx->spk_slots) { free(ctx->spk_names); free(ctx->spk_slots);
                                              ctx->spk_names = NULL; ctx->spk_slots = NULL; return; }

    while (*p && *p != '}') {
        while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r' || *p == ',') p++;
        if (*p != '"') break;
        const char *ns = ++p;
        while (*p && *p != '"') p++;
        if (*p != '"') break;
        size_t nlen = (size_t)(p - ns);
        p++;
        while (*p == ' ' || *p == ':' || *p == '\t' || *p == '\n' || *p == '\r') p++;
        char *end = NULL;
        long slot = strtol(p, &end, 10);
        if (end == p) break;
        p = end;

        if (ctx->spk_count == cap) {
            cap *= 2;
            char **nn = (char **)realloc(ctx->spk_names, (size_t)cap * sizeof(char *));
            int   *ns2 = (int *)realloc(ctx->spk_slots, (size_t)cap * sizeof(int));
            if (!nn || !ns2) { free(nn ? nn : ctx->spk_names); free(ns2 ? ns2 : ctx->spk_slots);
                               ctx->spk_names = NULL; ctx->spk_slots = NULL; ctx->spk_count = 0; return; }
            ctx->spk_names = nn; ctx->spk_slots = ns2;
        }
        char *nm = (char *)malloc(nlen + 1);
        if (!nm) return;
        memcpy(nm, ns, nlen); nm[nlen] = '\0';
        ctx->spk_names[ctx->spk_count] = nm;
        ctx->spk_slots[ctx->spk_count] = (int)slot;
        ctx->spk_count++;
    }
}

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

static const char *json_match_brace(const char *p, int depth) {
    int in_str = 0;
    while (*p && depth > 0) {
        char ch = *p;
        if (in_str) {
            if (ch == '\\' && p[1]) p++;
            else if (ch == '"') in_str = 0;
        } else if (ch == '"') in_str = 1;
        else if (ch == '{') depth++;
        else if (ch == '}') depth--;
        p++;
    }
    return p;
}

static int load_config(qwen_tts_ctx_t *ctx) {
    char path[1024]; snprintf(path, sizeof(path), "%s/config.json", ctx->model_dir);
    long len; char *json = read_file(path, &len); if (!json) return -1;
    qwen_tts_config_t *c = &ctx->config;

    const char *tc_start = strstr(json, "\"talker_config\"");
    if (!tc_start) { free(json); return -1; }
    const char *p = strchr(tc_start, '{'); if (!p) { free(json); return -1; }

    const char *tc_end = json_match_brace(p + 1, 1);

    long tc_len = tc_end - p; char *tc_json = (char *)malloc(tc_len + 1);
    if (!tc_json) { free(json); return -1; }
    memcpy(tc_json, p, tc_len); tc_json[tc_len] = '\0';

    char *talker_only_json = strdup(tc_json);
    if (!talker_only_json) { free(tc_json); free(json); return -1; }
    {
        char *scan = talker_only_json;
        while (1) {
            char *q = scan;
            char *nested_open = NULL;
            while (*q) {
                if (*q == '"') {
                    q++;
                    while (*q && *q != '"') { if (*q == '\\') q++; q++; }
                    if (*q) q++;
                    while (*q == ' ' || *q == '\t' || *q == '\n' || *q == '\r' || *q == ':') q++;
                    if (*q == '{') { nested_open = q; break; }
                } else {
                    q++;
                }
            }
            if (!nested_open) break;
            char *r = (char *)json_match_brace(nested_open + 1, 1);
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
                if (!cp_json) { free(tc_json); free(json); return -1; }
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
                const char *dc_close = json_match_brace(dc_open + 1, 1);
                long dc_len = dc_close - dc_open; char *dc_json = (char *)malloc(dc_len + 1);
                if (!dc_json) { free(json); return -1; }
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

#define matvec_bf16 qwen_matvec_bf16

extern int qwen_talker_load(qwen_tts_ctx_t *ctx);
extern int qwen_cp_load(qwen_tts_ctx_t *ctx);
extern int qwen_speech_decoder_load(qwen_tts_ctx_t *ctx);
extern int qwen_talker_prefill(qwen_tts_ctx_t *ctx, float *input_embeds, int seq_len);
extern void qwen_talker_prefix_key(qwen_tts_ctx_t *ctx, int prefix_len, int speaker_id,
                                   int language_id, int think_mode, uint64_t ihash);
extern uint64_t qwen_prefix_hash(const int *toks, int n);
extern int qwen_prefix_cache_enabled(void);
extern int qwen_talker_step(qwen_tts_ctx_t *ctx, float *embed, float *hidden_out);
extern int qwen_cp_predict(qwen_tts_ctx_t *ctx, float *talker_hidden, int code0, int *out_codes);
#ifdef CP_MICROBENCH
extern void qwen_cp_microbench_report(int frames);
#endif
extern int qwen_speech_decoder_decode(qwen_tts_ctx_t *ctx, const int *codes, int n_frames, float **audio_out, int *n_samples);
extern int qwen_speech_decoder_decode_streaming(qwen_tts_ctx_t *ctx, const int *new_codes, int new_frames, float **audio_out, int *n_samples);
extern int qwen_speech_decoder_decode_streaming_st(qwen_tts_ctx_t *ctx, qwen_sd_stream_state_t *st, const int *new_codes, int new_frames, float **audio_out, int *n_samples);
extern int qwen_speech_decoder_decode_streaming_batch(qwen_tts_ctx_t *ctx, qwen_sd_batch_item_t *items, int n_items);
extern void qwen_sd_stream_init(qwen_sd_stream_state_t *st);
extern void qwen_sd_stream_free(qwen_sd_stream_state_t *st);
extern int qwen_tts_sample(float *logits, int vocab_size, float temp, int top_k, float top_p, float rep_penalty, int *prev_tokens, int n_prev);
extern void qwen_set_seed(uint32_t seed);
extern uint32_t qwen_get_seed(void);

void embed_one_text_token_compute(qwen_tts_ctx_t *ctx, int tid, float *out) {
    qwen_mm_component(QWEN_COMP_TALKER);
    int th = ctx->config.text_hidden_size, h = ctx->config.hidden_size;
    float *text_emb = ctx->emb_tmp1;
    float *fc1_out = ctx->emb_tmp2;
    const uint16_t *emb = ctx->tok_embeddings_bf16 + (int64_t)tid * th;
    for (int j = 0; j < th; j++) text_emb[j] = bf16_to_f32(emb[j]);
    if (ctx->text_proj_fc1_bf16 && ctx->text_proj_fc2_bf16) {
        if (ctx->text_proj_fc1_q4) qwen_matvec_q4_0(fc1_out, ctx->text_proj_fc1_q4, text_emb, th, th);
        else                       matvec_bf16(fc1_out, ctx->text_proj_fc1_bf16, text_emb, th, th);
        if (ctx->text_proj_fc1_bias) for (int j = 0; j < th; j++) fc1_out[j] += ctx->text_proj_fc1_bias[j];
        for (int j = 0; j < th; j++) fc1_out[j] = fc1_out[j] / (1.0f + expf(-fc1_out[j]));
        if (ctx->text_proj_fc2_q4) qwen_matvec_q4_0(out, ctx->text_proj_fc2_q4, fc1_out, h, th);
        else                       matvec_bf16(out, ctx->text_proj_fc2_bf16, fc1_out, h, th);
        if (ctx->text_proj_fc2_bias) for (int j = 0; j < h; j++) out[j] += ctx->text_proj_fc2_bias[j];
    } else {
        memcpy(out, text_emb, h * sizeof(float));
    }
}

#define EMB_CACHE_CAPACITY 2048

static void emb_cache_init(qwen_tts_ctx_t *ctx) {
    int cap = EMB_CACHE_CAPACITY;
    int h = ctx->config.hidden_size;
    ctx->emb_cache.capacity = cap;
    ctx->emb_cache.count = 0;
    ctx->emb_cache.clock = 0;
    ctx->emb_cache.keys = (int *)aligned_malloc(cap * sizeof(int));
    ctx->emb_cache.values = (float *)aligned_malloc((size_t)cap * h * sizeof(float));
    ctx->emb_cache.access = (uint32_t *)aligned_calloc(cap, sizeof(uint32_t));
    for (int i = 0; i < cap; i++) ctx->emb_cache.keys[i] = -1;
}

static void emb_cache_free(qwen_tts_ctx_t *ctx) {
    free(ctx->emb_cache.keys);
    free(ctx->emb_cache.values);
    free(ctx->emb_cache.access);
    ctx->emb_cache.keys = NULL;
    ctx->emb_cache.values = NULL;
    ctx->emb_cache.access = NULL;
    ctx->emb_cache.capacity = 0;
    ctx->emb_cache.count = 0;
}

static const float *emb_cache_get(qwen_tts_ctx_t *ctx, int tid) {
    int cap = ctx->emb_cache.capacity;
    int h = ctx->config.hidden_size;
    int mask = cap - 1;
    int idx = (tid * 2654435761u) & mask;

    for (int probe = 0; probe < cap; probe++) {
        int slot = (idx + probe) & mask;
        if (ctx->emb_cache.keys[slot] == tid) {
            ctx->emb_cache.access[slot] = ++ctx->emb_cache.clock;
            return ctx->emb_cache.values + (size_t)slot * h;
        }
        if (ctx->emb_cache.keys[slot] == -1) {
            ctx->emb_cache.keys[slot] = tid;
            ctx->emb_cache.access[slot] = ++ctx->emb_cache.clock;
            ctx->emb_cache.count++;
            float *dst = ctx->emb_cache.values + (size_t)slot * h;
            embed_one_text_token_compute(ctx, tid, dst);
            return dst;
        }
    }

    uint32_t min_access = UINT32_MAX;
    int victim = 0;
    for (int i = 0; i < cap; i++) {
        if (ctx->emb_cache.access[i] < min_access) {
            min_access = ctx->emb_cache.access[i];
            victim = i;
        }
    }
    ctx->emb_cache.keys[victim] = tid;
    ctx->emb_cache.access[victim] = ++ctx->emb_cache.clock;
    float *dst = ctx->emb_cache.values + (size_t)victim * h;
    embed_one_text_token_compute(ctx, tid, dst);
    return dst;
}

static void embed_one_text_token(qwen_tts_ctx_t *ctx, int tid, float *out) {
    int h = ctx->config.hidden_size;
    if (ctx->cached_tts_pad_embed) {
        if (tid == QWEN_TTS_TTS_PAD) { memcpy(out, ctx->cached_tts_pad_embed, h * sizeof(float)); return; }
        if (tid == QWEN_TTS_TTS_BOS) { memcpy(out, ctx->cached_tts_bos_embed, h * sizeof(float)); return; }
        if (tid == QWEN_TTS_TTS_EOS) { memcpy(out, ctx->cached_tts_eos_embed, h * sizeof(float)); return; }
    }
    if (ctx->emb_cache.capacity > 0) {
        const float *cached = emb_cache_get(ctx, tid);
        memcpy(out, cached, h * sizeof(float));
        return;
    }
    embed_one_text_token_compute(ctx, tid, out);
}

#define DT_CHUNK_FRAMES 10

typedef struct {
    pthread_mutex_t mutex;
    pthread_cond_t  cond;
    int *codes;
    int  capacity;
    int  write_pos;
    int  read_pos;
    int  done;

    float *audio_buf;
    int    audio_len;
    int    audio_cap;

    qwen_tts_audio_cb audio_cb;
    void *audio_cb_userdata;
    _Atomic int cb_aborted;

    qwen_tts_ctx_t *ctx;
    double decode_ms;
    double first_chunk_ms;
    int    chunk_frames;
    int    trim_head_left;
} decoder_thread_t;

static void dt_init(decoder_thread_t *dt, qwen_tts_ctx_t *ctx, int max_frames) {
    pthread_mutex_init(&dt->mutex, NULL);
    pthread_cond_init(&dt->cond, NULL);
    dt->capacity = max_frames;
    dt->codes = (int *)malloc((size_t)max_frames * 16 * sizeof(int));
    dt->write_pos = 0;
    dt->read_pos = 0;
    dt->done = 0;
    dt->ctx = ctx;
    dt->decode_ms = 0;
    dt->first_chunk_ms = 0;
    dt->chunk_frames = (ctx->stream_chunk_frames > 0) ? ctx->stream_chunk_frames : DT_CHUNK_FRAMES;
    dt->audio_cb = NULL;
    dt->audio_cb_userdata = NULL;
    dt->cb_aborted = 0;
    dt->trim_head_left = 0;
    dt->audio_cap = max_frames * 1920 + 4096;
    dt->audio_buf = (float *)aligned_malloc(dt->audio_cap * sizeof(float));
    dt->audio_len = 0;
}

static void dt_free(decoder_thread_t *dt) {
    pthread_mutex_destroy(&dt->mutex);
    pthread_cond_destroy(&dt->cond);
    free(dt->codes);
    free(dt->audio_buf);
    dt->audio_buf = NULL;
}

static void dt_push_frames(decoder_thread_t *dt, const int *frame_codes, int n_frames) {
    pthread_mutex_lock(&dt->mutex);
    if (n_frames < 0 || dt->write_pos + n_frames > dt->capacity) {
        pthread_mutex_unlock(&dt->mutex);
        return;
    }
    memcpy(dt->codes + dt->write_pos * 16, frame_codes, n_frames * 16 * sizeof(int));
    dt->write_pos += n_frames;
    pthread_cond_signal(&dt->cond);
    pthread_mutex_unlock(&dt->mutex);
}

static void dt_finish(decoder_thread_t *dt) {
    pthread_mutex_lock(&dt->mutex);
    dt->done = 1;
    pthread_cond_signal(&dt->cond);
    pthread_mutex_unlock(&dt->mutex);
}

static void dt_append_audio(decoder_thread_t *dt, const float *samples, int n) {
    if (n <= 0) return;
    if (dt->audio_len + n > dt->audio_cap) {
        size_t newcap = ((size_t)dt->audio_len + (size_t)n) * 2;
        float *nb = (float *)realloc(dt->audio_buf, newcap * sizeof(float));
        if (!nb) return;
        dt->audio_buf = nb;
        dt->audio_cap = newcap > (size_t)0x7FFFFFFF ? 0x7FFFFFFF : (int)newcap;
    }
    memcpy(dt->audio_buf + dt->audio_len, samples, (size_t)n * sizeof(float));
    dt->audio_len += n;
}

static void *decoder_thread_fn(void *arg) {
    decoder_thread_t *dt = (decoder_thread_t *)arg;
    qwen_tts_ctx_t *ctx = dt->ctx;

    for (;;) {
        int avail, is_done;
        pthread_mutex_lock(&dt->mutex);
        int target = dt->chunk_frames;
        if (dt->first_chunk_ms == 0 && dt->ctx->stream && target > 2)
            target = 2;
        while (dt->write_pos - dt->read_pos < target && !dt->done)
            pthread_cond_wait(&dt->cond, &dt->mutex);
        avail = dt->write_pos - dt->read_pos;
        is_done = dt->done;
        pthread_mutex_unlock(&dt->mutex);

        if (avail <= 0 && is_done) break;
        if (avail <= 0) continue;

        const int *chunk_codes = dt->codes + dt->read_pos * 16;
        float *chunk_audio = NULL;
        int chunk_samples = 0;

        double t0 = 0;
        struct timeval tv;
        gettimeofday(&tv, NULL);
        t0 = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;

        if (dt->cb_aborted) { dt->read_pos += avail; continue; }

        if (qwen_speech_decoder_decode_streaming(ctx, chunk_codes, avail,
                                                   &chunk_audio, &chunk_samples) == 0) {
            if (chunk_samples > 0 && chunk_audio) {
                float *emit = chunk_audio;
                int emit_n = chunk_samples;
                if (dt->trim_head_left > 0) {
                    int cut = dt->trim_head_left < emit_n ? dt->trim_head_left : emit_n;
                    emit += cut; emit_n -= cut; dt->trim_head_left -= cut;
                }
                if (emit_n > 0) {
                    if (dt->first_chunk_ms == 0) {
                        struct timeval tvf; gettimeofday(&tvf, NULL);
                        dt->first_chunk_ms = tvf.tv_sec * 1000.0 + tvf.tv_usec / 1000.0;
                    }
                    if (dt->audio_cb) {
                        int ret = dt->audio_cb(emit, emit_n, dt->audio_cb_userdata);
                        if (ret != 0) dt->cb_aborted = 1;
                        dt->audio_len += emit_n;
                    } else {
                        dt_append_audio(dt, emit, emit_n);
                    }
                }
            }
            free(chunk_audio);
        }

        struct timeval tv2;
        gettimeofday(&tv2, NULL);
        dt->decode_ms += (tv2.tv_sec * 1000.0 + tv2.tv_usec / 1000.0) - t0;

        dt->read_pos += avail;
    }

    return NULL;
}

static void qwen_weights_thp_advise(ingot_st *st, const char *what) {
#if defined(__linux__) && defined(MADV_HUGEPAGE)
    const char *e = getenv("QWEN_THP");
    if (!e || e[0] != '1' || !st) return;
    size_t n = ingot_st_count(st);
    uintptr_t lo = (uintptr_t)-1, hi = 0;
    for (size_t i = 0; i < n; i++) {
        const ingot_st_tensor *t = ingot_st_at(st, i);
        if (!t) continue;
        const void *d = ingot_st_data(st, t);
        if (!d) continue;
        uintptr_t a = (uintptr_t)d;
        if (a < lo) lo = a;
        if (a + t->nbytes > hi) hi = a + t->nbytes;
    }
    if (hi <= lo) return;
    long pg = sysconf(_SC_PAGESIZE); if (pg <= 0) pg = 4096;
    uintptr_t alo = lo & ~((uintptr_t)pg - 1);
    uintptr_t ahi = (hi + (uintptr_t)pg - 1) & ~((uintptr_t)pg - 1);
    int rc = madvise((void *)alo, (size_t)(ahi - alo), MADV_HUGEPAGE);
    fprintf(stderr, "  QWEN_THP: madvise(MADV_HUGEPAGE) su %.1f MB di pesi (%s): %s\n",
            (double)(ahi - alo) / (1024.0 * 1024.0), what,
            rc == 0 ? "ok" : "rifiutata dal kernel (THP disabilitate?)");
#else
    (void)st; (void)what;
#endif
}

qwen_tts_ctx_t *qwen_tts_load_ex(const char *model_dir, int silent, int use_int8, int use_int4) {
    qwen_tts_ctx_t *ctx = (qwen_tts_ctx_t *)calloc(1, sizeof(qwen_tts_ctx_t)); if (!ctx) return NULL;
    strncpy(ctx->model_dir, model_dir, sizeof(ctx->model_dir) - 1);
    ctx->temperature = 0.9f; ctx->top_k = 50; ctx->top_p = 1.0f; ctx->rep_penalty = 1.05f;
    ctx->eos_strategy         = QWEN_EOS_V1;
    ctx->eos_suppress_frames  = 2;
    ctx->eos_frames_per_token = 3.0f;
    ctx->eos_start_multiple   = 2.0f;
    ctx->eos_overhead_frames  = 18;
    ctx->eos_ramp_per_frame   = 0.5f;
    ctx->eos_ramp_cap         = 10.0f;
    ctx->eos_topk             = 50;
    ctx->max_tokens = 8192; ctx->cp_temperature = 0.9f; ctx->cp_top_k = 50;
    ctx->stream_chunk_frames = 10;
    ctx->speaker_id = 3061; ctx->language_id = -1; ctx->seed = (uint32_t)time(NULL);
    ctx->silent = silent; ctx->debug = 0;
    ctx->use_int8 = use_int8; ctx->use_int4 = use_int4;

    char config_path[1024];
    snprintf(config_path, sizeof(config_path), "%s/config.json", ctx->model_dir);
    if (load_config(ctx) != 0) {
        snprintf(config_path, sizeof(config_path), "config.json");
        if (load_config(ctx) != 0) { free(ctx); return NULL; }
    }

    qwen_tts_config_t *c = &ctx->config;

    {
        char cfg_path[1024];
        snprintf(cfg_path, sizeof(cfg_path), "%s/config.json", ctx->model_dir);
        long cfg_len;
        char *cfg_raw = read_file(cfg_path, &cfg_len);
        if (cfg_raw) {
            const char *mt = strstr(cfg_raw, "\"tts_model_type\"");
            if (mt) {
                const char *val = strchr(mt + 16, '"');
                if (val) {
                    val++;
                    if (strncmp(val, "base", 4) == 0) ctx->is_base_model = 1;
                }
            }

            const char *sec = strstr(cfg_raw, "\"speaker_encoder_config\"");
            if (sec) {
                ctx->speaker_enc_dim = json_get_int(sec, "enc_dim", 1024);
            }

            if (!ctx->voice_design && !ctx->is_base_model) {
                const char *spk = strstr(cfg_raw, "\"spk_id\"");
                if (spk) {
                    const char *p = spk + 8;
                    while (*p == ' ' || *p == ':' || *p == '\t' || *p == '\n') p++;
                    if (*p == '{') {
                        p++;
                        while (*p == ' ' || *p == '\t' || *p == '\n') p++;
                        if (*p == '}') ctx->voice_design = 1;
                    }
                }
            }

            if (!ctx->voice_design)
                parse_spk_id_table(ctx, cfg_raw);

            free(cfg_raw);
        }
    }

    if (!ctx->silent) {
        fprintf(stderr, "Config: hidden=%d text_hidden=%d layers=%d heads=%d/%d head_dim=%d inter=%d\n",
                c->hidden_size, c->text_hidden_size, c->num_layers, c->num_heads, c->num_kv_heads, c->head_dim, c->intermediate_size);
        fprintf(stderr, "  Code Predictor: hidden=%d layers=%d heads=%d head_dim=%d\n",
                c->cp_hidden_size, c->cp_num_layers, c->cp_num_heads, c->cp_head_dim);
        fprintf(stderr, "  Codec: vocab=%d codebooks=%d entries=%d\n", c->codec_vocab_size, c->dec_num_quantizers, c->codebook_size);
        if (ctx->voice_design) {
            fprintf(stderr, "  Mode: VoiceDesign (no preset speakers)\n");
            if (c->hidden_size < 2048)
                fprintf(stderr, "  Warning: VoiceDesign requires a 1.7B model; results may be incorrect\n");
        }
    }

    char st_err[256] = "";
    ingot_st *st_main = NULL;
    if (ingot_st_open_dir(&st_main, ctx->model_dir, st_err, sizeof st_err) != 0) {
        fprintf(stderr, "Error: Failed to load model from %s: %s\n",
                ctx->model_dir, st_err);
        free(ctx); return NULL;
    }
    ctx->safetensors = st_main;
    qwen_weights_thp_advise(st_main, "talker + code predictor");
    char speech_dir[4096];
    snprintf(speech_dir, sizeof(speech_dir), "%s/speech_tokenizer", ctx->model_dir);
    ingot_st *st_speech = NULL;
    if (ingot_st_open_dir(&st_speech, speech_dir, st_err, sizeof st_err) != 0) {
        fprintf(stderr, "Error: Failed to load speech tokenizer from %s: %s\n",
                speech_dir, st_err);
        ingot_st_close(ctx->safetensors);
        free(ctx); return NULL;
    }
    ctx->speech_safetensors = st_speech;
    qwen_weights_thp_advise(st_speech, "speech tokenizer");

    if (!ctx->silent) fprintf(stderr, "Threads: %d\n", qwen_get_threads());

    double t0 = time_ms();
    if (qwen_talker_load(ctx) != 0 || qwen_cp_load(ctx) != 0 || qwen_speech_decoder_load(ctx) != 0) {
        ingot_st_close(ctx->safetensors);
        ingot_st_close(ctx->speech_safetensors);
        free(ctx); return NULL;
    }

    if (ctx->is_base_model) {
        if (ctx->speaker_enc_dim > 0)
            ctx->speaker_enc.enc_dim = ctx->speaker_enc_dim;
        if (qwen_speaker_encoder_load(&ctx->speaker_enc, ctx->safetensors) != 0) {
            fprintf(stderr, "Warning: failed to load speaker encoder (voice cloning unavailable)\n");
        } else if (!ctx->silent) {
            fprintf(stderr, "  Speaker encoder: ECAPA-TDNN (enc_dim=%d)\n", ctx->speaker_enc.enc_dim);
        }
    }

    int th = ctx->config.text_hidden_size;
    int h = ctx->config.hidden_size;
    ctx->emb_tmp1 = (float *)aligned_malloc(th * sizeof(float));
    ctx->emb_tmp2 = (float *)aligned_malloc(th * sizeof(float));

    ctx->cached_tts_pad_embed = (float *)aligned_malloc(h * sizeof(float));
    ctx->cached_tts_bos_embed = (float *)aligned_malloc(h * sizeof(float));
    ctx->cached_tts_eos_embed = (float *)aligned_malloc(h * sizeof(float));
    embed_one_text_token_compute(ctx, QWEN_TTS_TTS_PAD, ctx->cached_tts_pad_embed);
    embed_one_text_token_compute(ctx, QWEN_TTS_TTS_BOS, ctx->cached_tts_bos_embed);
    embed_one_text_token_compute(ctx, QWEN_TTS_TTS_EOS, ctx->cached_tts_eos_embed);
    if (!ctx->silent) {
        fprintf(stderr, "  tts_pad_embed[:3]=[%.6f,%.6f,%.6f]\n",
                ctx->cached_tts_pad_embed[0], ctx->cached_tts_pad_embed[1], ctx->cached_tts_pad_embed[2]);
    }

    emb_cache_init(ctx);

    if (!ctx->silent) fprintf(stderr, "Model loaded in %.0f ms\n", time_ms() - t0);
    return ctx;
}

qwen_tts_ctx_t *qwen_tts_load(const char *model_dir) {
    return qwen_tts_load_ex(model_dir, 0, 0, 0);
}

void qwen_track_override(qwen_tts_ctx_t *ctx, void *ptr) {
    if (!ctx || !ptr) return;
    if (ctx->n_owned_overrides >= ctx->cap_owned_overrides) {
        int nc = ctx->cap_owned_overrides ? ctx->cap_owned_overrides * 2 : 64;
        void **t = (void **)realloc(ctx->owned_overrides, (size_t)nc * sizeof(void *));
        if (!t) return;
        ctx->owned_overrides = t;
        ctx->cap_owned_overrides = nc;
    }
    ctx->owned_overrides[ctx->n_owned_overrides++] = ptr;
}

void qwen_tts_unload(qwen_tts_ctx_t *ctx) {
    if (!ctx) return;
    for (int i = 0; i < ctx->n_owned_overrides; i++) free(ctx->owned_overrides[i]);
    free(ctx->owned_overrides);
    for (int i = 0; i < ctx->config.num_layers; i++) free(ctx->layers[i].gate_up_fused_bf16);
    for (int i = 0; i < ctx->config.cp_num_layers; i++) free(ctx->cp_layers[i].gate_up_fused_bf16);
    for (int i = 0; i < ctx->config.cp_num_layers; i++) free(ctx->cp_layers[i].down_q2_rough);
    for (int i = 0; i < 16; i++) free(ctx->speech_dec.codebook[i]);
    for (int i = 0; i < 6; i++) free(ctx->speech_dec.convt_packed[i]);
    free(ctx->speech_dec.pre_layers);
    free(ctx->speech_dec.rope_cos); free(ctx->speech_dec.rope_sin);
    ingot_st_close(ctx->safetensors);
    ingot_st_close(ctx->speech_safetensors);
    free(ctx->instruct);
    for (int i = 0; i < ctx->spk_count; i++) free(ctx->spk_names[i]);
    free(ctx->spk_names);
    free(ctx->spk_slots);
    free(ctx->speaker_embedding);
    free(ctx->ref_audio_path);
    free(ctx->ref_text);
    free(ctx->emo_ref_path);
    free(ctx->emo_ref_text);
    free(ctx->kv_cache_k); free(ctx->kv_cache_v); free(ctx->cp_kv_k); free(ctx->cp_kv_v);
    free(ctx->dec_x); free(ctx->dec_x_norm); free(ctx->dec_q); free(ctx->dec_k); free(ctx->dec_v);
    free(ctx->dec_attn_out); free(ctx->dec_proj_out); free(ctx->dec_gate); free(ctx->dec_up); free(ctx->dec_ffn_out);
    free(ctx->cp_dec_x); free(ctx->cp_dec_q); free(ctx->cp_dec_k); free(ctx->cp_dec_v);
    free(ctx->cp_dec_attn_out); free(ctx->cp_dec_gate); free(ctx->cp_dec_up); free(ctx->cp_dec_ffn_out);
    free(ctx->pref_residual); free(ctx->pref_x_norm); free(ctx->pref_q);
    free(ctx->pref_k); free(ctx->pref_v); free(ctx->pref_attn_out);
    free(ctx->pref_gate); free(ctx->pref_proj);
    free(ctx->pref_wq_f32); free(ctx->pref_wk_f32); free(ctx->pref_wv_f32);
    free(ctx->pref_wo_f32); free(ctx->pref_gate_up_f32); free(ctx->pref_down_f32);
    free(ctx->rope_cos); free(ctx->rope_sin); free(ctx->rope_inv_freq);
    free(ctx->cp_rope_cos); free(ctx->cp_rope_sin);
    free(ctx->emb_tmp1); free(ctx->emb_tmp2);
    free(ctx->cached_tts_pad_embed); free(ctx->cached_tts_bos_embed); free(ctx->cached_tts_eos_embed);
    emb_cache_free(ctx);
    free(ctx->logits); free(ctx->codec_codes); free(ctx->prev_tokens); free(ctx->audio_buf);
    free(ctx->prev_input_embeds); free(ctx->cached_ref_codes);
    if (ctx->cached_tokenizer) qwen_tokenizer_free((qwen_tokenizer_t *)ctx->cached_tokenizer);
    free(ctx);
}

qwen_tts_ctx_t *qwen_tts_clone_for_worker(const qwen_tts_ctx_t *base) {
    if (!base) return NULL;
    qwen_tts_ctx_t *w = (qwen_tts_ctx_t *)malloc(sizeof(qwen_tts_ctx_t));
    if (!w) return NULL;
    *w = *base;

    const qwen_tts_config_t *c = &w->config;
    int h        = c->hidden_size;
    int th       = c->text_hidden_size;
    int q_dim    = c->num_heads * c->head_dim;
    int kv_dim   = c->num_kv_heads * c->head_dim;
    int cp_h     = c->cp_hidden_size;
    int cp_q_dim = c->cp_num_heads * c->cp_head_dim;
    int cp_kv_dim= c->cp_num_kv_heads * c->cp_head_dim;
    int swiglu_size = c->intermediate_size > c->cp_intermediate_size
                      ? c->intermediate_size : c->cp_intermediate_size;

    int talker_kv_max = 2048;
    int64_t kv_size = (int64_t)c->num_layers * talker_kv_max * kv_dim;
    w->kv_cache_k = (uint16_t *)aligned_calloc(kv_size, sizeof(uint16_t));
    w->kv_cache_v = (uint16_t *)aligned_calloc(kv_size, sizeof(uint16_t));
    w->kv_max = talker_kv_max; w->kv_len = 0;
    w->dec_x        = (float *)aligned_calloc(h, sizeof(float));
    w->dec_x_norm   = (float *)aligned_malloc(h * sizeof(float));
    w->dec_q        = (float *)aligned_malloc(q_dim * sizeof(float));
    w->dec_k        = (float *)aligned_malloc(kv_dim * sizeof(float));
    w->dec_v        = (float *)aligned_malloc(kv_dim * sizeof(float));
    w->dec_attn_out = (float *)aligned_malloc(q_dim * sizeof(float));
    w->dec_proj_out = (float *)aligned_malloc(h * sizeof(float));
    w->dec_gate     = (float *)aligned_malloc(2 * c->intermediate_size * sizeof(float));
    w->dec_up       = NULL;
    w->dec_ffn_out  = (float *)aligned_malloc(h * sizeof(float));
    w->swiglu_tmp   = (float *)aligned_malloc(swiglu_size * sizeof(float));

    int cp_kv_max = 64;
    int64_t cp_kv_size = (int64_t)c->cp_num_layers * cp_kv_max * cp_kv_dim;
    w->cp_kv_k = (uint16_t *)aligned_calloc(cp_kv_size, sizeof(uint16_t));
    w->cp_kv_v = (uint16_t *)aligned_calloc(cp_kv_size, sizeof(uint16_t));
    w->cp_kv_max = cp_kv_max; w->cp_kv_len = 0;
    w->cp_dec_x        = (float *)aligned_malloc(cp_h * sizeof(float));
    w->cp_dec_q        = (float *)aligned_malloc(cp_q_dim * sizeof(float));
    w->cp_dec_k        = (float *)aligned_malloc(cp_kv_dim * sizeof(float));
    w->cp_dec_v        = (float *)aligned_malloc(cp_kv_dim * sizeof(float));
    w->cp_dec_attn_out = (float *)aligned_malloc(cp_q_dim * sizeof(float));
    w->cp_dec_gate     = (float *)aligned_malloc(2 * c->cp_intermediate_size * sizeof(float));
    w->cp_dec_up       = NULL;
    w->cp_dec_ffn_out  = (float *)aligned_malloc(cp_h * sizeof(float));

    w->emb_tmp1 = (float *)aligned_malloc(th * sizeof(float));
    w->emb_tmp2 = (float *)aligned_malloc(th * sizeof(float));

    memset(&w->emb_cache, 0, sizeof(w->emb_cache));
    emb_cache_init(w);

    w->pref_residual = w->pref_x_norm = w->pref_q = NULL;
    w->pref_k = w->pref_v = w->pref_attn_out = w->pref_gate = w->pref_proj = NULL;
    w->pref_seq_cap = 0;
    w->pref_wq_f32 = w->pref_wk_f32 = w->pref_wv_f32 = NULL;
    w->pref_wo_f32 = w->pref_gate_up_f32 = w->pref_down_f32 = NULL;
    w->logits = NULL;
    w->codec_codes = NULL; w->codec_frames = 0; w->codec_frames_cap = 0;
    w->prev_tokens = NULL; w->n_prev_tokens = 0; w->prev_tokens_cap = 0;
    w->prev_input_embeds = NULL; w->prev_prefill_len = 0;
    w->audio_buf = NULL; w->audio_samples = 0;
    memset(&w->sd_stream, 0, sizeof(w->sd_stream));

    w->cp_rough_built = 0;
    for (int i = 0; i < c->cp_num_layers; i++) w->cp_layers[i].down_q2_rough = NULL;

    w->cached_tokenizer = NULL;
    w->instruct = NULL;
    w->tf_ref_codes = NULL;
    w->stream = 0; w->audio_cb = NULL; w->audio_cb_userdata = NULL;

    return w;
}

void qwen_tts_free_clone(qwen_tts_ctx_t *ctx) {
    if (!ctx) return;
    free(ctx->kv_cache_k); free(ctx->kv_cache_v); free(ctx->cp_kv_k); free(ctx->cp_kv_v);
    free(ctx->dec_x); free(ctx->dec_x_norm); free(ctx->dec_q); free(ctx->dec_k); free(ctx->dec_v);
    free(ctx->dec_attn_out); free(ctx->dec_proj_out); free(ctx->dec_gate); free(ctx->dec_up); free(ctx->dec_ffn_out);
    free(ctx->swiglu_tmp);
    free(ctx->cp_dec_x); free(ctx->cp_dec_q); free(ctx->cp_dec_k); free(ctx->cp_dec_v);
    free(ctx->cp_dec_attn_out); free(ctx->cp_dec_gate); free(ctx->cp_dec_up); free(ctx->cp_dec_ffn_out);
    free(ctx->pref_residual); free(ctx->pref_x_norm); free(ctx->pref_q);
    free(ctx->pref_k); free(ctx->pref_v); free(ctx->pref_attn_out);
    free(ctx->pref_gate); free(ctx->pref_proj);
    free(ctx->pref_wq_f32); free(ctx->pref_wk_f32); free(ctx->pref_wv_f32);
    free(ctx->pref_wo_f32); free(ctx->pref_gate_up_f32); free(ctx->pref_down_f32);
    free(ctx->emb_tmp1); free(ctx->emb_tmp2);
    emb_cache_free(ctx);
    free(ctx->logits); free(ctx->codec_codes); free(ctx->prev_tokens); free(ctx->audio_buf);
    free(ctx->prev_input_embeds);
    for (int i = 0; i < ctx->config.cp_num_layers; i++) free(ctx->cp_layers[i].down_q2_rough);
    free(ctx->instruct);
    if (ctx->cached_tokenizer) qwen_tokenizer_free((qwen_tokenizer_t *)ctx->cached_tokenizer);
    free(ctx);
}

void qwen_tts_set_audio_callback(qwen_tts_ctx_t *ctx, qwen_tts_audio_cb cb, void *userdata) {
    ctx->audio_cb = cb;
    ctx->audio_cb_userdata = userdata;
}

void qwen_tts_set_speaker(qwen_tts_ctx_t *ctx, int speaker_id) { ctx->speaker_id = speaker_id; }
void qwen_tts_set_language(qwen_tts_ctx_t *ctx, const char *language) {
    ctx->language_id = qwen_tts_language_id(language);
    if (ctx->language_id == QWEN_TTS_LANG_ENGLISH) {
        ctx->speaker_id = 3061;
    } else if (ctx->language_id == QWEN_TTS_LANG_CHINESE) {
        ctx->speaker_id = 3066;
    } else if (ctx->language_id == QWEN_TTS_LANG_JAPANESE) {
        ctx->speaker_id = 2873;
    } else if (ctx->language_id == QWEN_TTS_LANG_KOREAN) {
        ctx->speaker_id = 2864;
    }
}

static void spk_debug_row(qwen_tts_ctx_t *ctx, int token_id, const uint16_t *emb, int h) {
    static int done[64]; static int ndone = 0;
    const char *e = getenv("QWEN_SPK_DEBUG");
    if (!e || e[0] != '1') return;
    for (int i = 0; i < ndone; i++) if (done[i] == token_id) return;
    if (ndone < 64) done[ndone++] = token_id;
    uint64_t fnv = 1469598103934665603ULL;
    const uint8_t *b = (const uint8_t *)emb;
    for (size_t i = 0; i < (size_t)h * 2; i++) { fnv ^= b[i]; fnv *= 1099511628211ULL; }
    double l2 = 0.0;
    for (int i = 0; i < h; i++) { float v = bf16_to_f32(emb[i]); l2 += (double)v * v; }
    fprintf(stderr, "[SPK] token %d  src=talker.model.codec_embedding.weight row %d  dtype=bf16  dim=%d"
                    "  fnv1a=%016llx  L2=%.6f  first=[%.6f %.6f %.6f]  last=%.6f\n",
            token_id, token_id, h, (unsigned long long)fnv, sqrt(l2),
            bf16_to_f32(emb[0]), bf16_to_f32(emb[1]), bf16_to_f32(emb[2]), bf16_to_f32(emb[h-1]));
    (void)ctx;
}

static void lookup_codec_embed(qwen_tts_ctx_t *ctx, int token_id, float *out) {
    int h = ctx->config.hidden_size;
    if (token_id < 0 || token_id >= ctx->config.codec_vocab_size) { memset(out, 0, h * sizeof(float)); return; }
    const uint16_t *emb = ctx->codec_embedding_bf16 + (int64_t)token_id * h;
    spk_debug_row(ctx, token_id, emb, h);
    qwen_bf16_to_f32_vec(out, emb, h);
}

int qwen_tts_generate(qwen_tts_ctx_t *ctx, const char *text, float **out_samples, int *out_n_samples) {
    double t_start = time_ms();
    int h = ctx->config.hidden_size;
    qwen_set_seed(ctx->seed);

    int32_t *instruct_tokens = NULL;
    int instruct_token_len = 0;
    qwen_tokenizer_t *tok = (qwen_tokenizer_t *)ctx->cached_tokenizer;
    if (!tok) {
        tok = qwen_tokenizer_load(ctx->model_dir);
        if (tok) ctx->cached_tokenizer = tok;
    }

    if (ctx->instruct && ctx->instruct[0] && tok) {
        int inst_len = (int)strlen(ctx->instruct);
        int tmpl_len = inst_len + 64;
        char *instruct_tmpl = (char *)malloc(tmpl_len);
        snprintf(instruct_tmpl, tmpl_len, "<|im_start|>user\n%s<|im_end|>\n", ctx->instruct);
        instruct_tokens = qwen_tokenizer_encode(tok, instruct_tmpl, &instruct_token_len);
        free(instruct_tmpl);
        if (!ctx->silent && instruct_tokens)
            fprintf(stderr, "Instruct: \"%s\" (%d tokens)\n", ctx->instruct, instruct_token_len);
    }

    int32_t *text_tokens = NULL;
    int text_token_len = 0;
    int32_t *ref_text_tokens = NULL;
    int ref_text_token_len = 0;
    if (tok) {
        text_tokens = qwen_tokenizer_encode_para(tok, text, &text_token_len);
        const char *icl_text = (ctx->emo_ref_path && ctx->emo_ref_text) ? ctx->emo_ref_text
                             : ((ctx->voice_clone && !ctx->xvector_only) ? ctx->ref_text : NULL);
        if (icl_text) {
            ref_text_tokens = qwen_tokenizer_encode(tok, icl_text, &ref_text_token_len);
            if (!ctx->silent && ref_text_tokens)
                fprintf(stderr, "Ref text: \"%s\" (%d tokens)\n", icl_text, ref_text_token_len);
        }
    }
    if (!text_tokens || text_token_len == 0) {
        fprintf(stderr, "Error: text tokenization failed\n");
        free(text_tokens);
        free(instruct_tokens);
        free(ref_text_tokens);
        return -1;
    }

    int role_len = 3;
    int suffix_len = 5;
    int all_len = role_len + text_token_len + suffix_len;
    int32_t *all_tokens = (int32_t *)malloc(all_len * sizeof(int32_t));
    int pos_t = 0;
    all_tokens[pos_t++] = 151644;
    all_tokens[pos_t++] = 77091;
    all_tokens[pos_t++] = 198;
    memcpy(all_tokens + pos_t, text_tokens, text_token_len * sizeof(int32_t));
    pos_t += text_token_len;
    all_tokens[pos_t++] = 151645;
    all_tokens[pos_t++] = 198;
    all_tokens[pos_t++] = 151644;
    all_tokens[pos_t++] = 77091;
    all_tokens[pos_t++] = 198;
    free(text_tokens);

    int text_content_len = all_len - role_len - suffix_len;

    if (!ctx->silent) {
        fprintf(stderr, "Text: \"%s\" (template: %d BPE tokens, text_content: %d)\n",
                text, all_len, text_content_len);
        if (ctx->eos_strategy == QWEN_EOS_V1 || ctx->eos_strategy == QWEN_EOS_V2) {
            float expected = (ctx->eos_strategy == QWEN_EOS_V2)
                ? (float)ctx->eos_overhead_frames + text_content_len * ctx->eos_frames_per_token
                : text_content_len * ctx->eos_frames_per_token;
            fprintf(stderr, "EOS strategy: %s (suppress=%d, ramp starts at frame %.0f, "
                            "slope=%.2f, cap=%.1f)\n",
                    qwen_tts_eos_strategy_name(ctx->eos_strategy), ctx->eos_suppress_frames,
                    ctx->eos_start_multiple * expected, ctx->eos_ramp_per_frame, ctx->eos_ramp_cap);
        } else {
            fprintf(stderr, "EOS strategy: %s (suppress=%d%s)\n",
                    qwen_tts_eos_strategy_name(ctx->eos_strategy), ctx->eos_suppress_frames,
                    ctx->eos_strategy == QWEN_EOS_TOPK ? ", lift to k-th logit" : "");
        }
    }

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
    if (ctx->voice_clone && ctx->speaker_embedding) {
        codec_tokens[codec_len++] = -1;
    } else if (!ctx->voice_design) {
        codec_tokens[codec_len++] = ctx->speaker_id;
    }
    codec_tokens[codec_len++] = QWEN_TTS_CODEC_PAD;
    codec_tokens[codec_len++] = QWEN_TTS_CODEC_BOS;

    const float *tts_pad_embed = ctx->cached_tts_pad_embed;
    const float *tts_bos_embed = ctx->cached_tts_bos_embed;
    const float *tts_eos_embed = ctx->cached_tts_eos_embed;

    float *codec_pad_embed = (float *)aligned_malloc(h * sizeof(float));
    float *codec_bos_embed = (float *)aligned_malloc(h * sizeof(float));
    lookup_codec_embed(ctx, QWEN_TTS_CODEC_PAD, codec_pad_embed);
    lookup_codec_embed(ctx, QWEN_TTS_CODEC_BOS, codec_bos_embed);

    int *ref_codes = NULL;
    int ref_n_frames = 0;
    int ref_codes_owned = 0;
    int icl_mode = 0;

    if (ctx->graft_mode && ctx->cached_ref_codes && !ctx->silent)
        fprintf(stderr, "ICL: --graft -> ignoring %d ref frames, cloning via x-vector (emotive)\n",
                ctx->cached_ref_n_frames);

    if (ctx->emo_ref_path && ctx->emo_ref_text && ref_text_tokens && ref_text_token_len > 0) {
        float *emo_samples = NULL;
        int emo_n_samples = 0, emo_sr = 0;
        if (qwen_read_wav(ctx->emo_ref_path, &emo_samples, &emo_n_samples, &emo_sr) != 0) {
            fprintf(stderr, "Error: failed to read emotion reference %s\n", ctx->emo_ref_path);
        } else {
            if (emo_sr != QWEN_TTS_SAMPLE_RATE && !ctx->silent)
                fprintf(stderr, "Warning: emo-ref sample rate %d, expected %d\n",
                        emo_sr, QWEN_TTS_SAMPLE_RATE);
            qwen_trim_trailing_silence(emo_samples, &emo_n_samples, emo_sr, ctx->silent);
            if (qwen_speech_encoder_encode(ctx, emo_samples, emo_n_samples,
                                           &ref_codes, &ref_n_frames) != 0) {
                fprintf(stderr, "Error: speech encoder failed on emotion reference\n");
                ref_codes = NULL; ref_n_frames = 0;
            } else {
                icl_mode = 1;
                ref_codes_owned = 1;
                if (!ctx->silent)
                    fprintf(stderr, "Emotion-by-example: %d ref frames from %s (identity unchanged)\n",
                            ref_n_frames, ctx->emo_ref_path);
            }
            free(emo_samples);
        }
    }
    else if (ctx->voice_clone && !ctx->graft_mode && ctx->cached_ref_codes && ctx->cached_ref_n_frames > 0
        && ctx->ref_text && ref_text_tokens && ref_text_token_len > 0) {
        ref_codes = ctx->cached_ref_codes;
        ref_n_frames = ctx->cached_ref_n_frames;
        icl_mode = 1;
        if (!ctx->silent)
            fprintf(stderr, "ICL: using %d cached ref frames from .qvoice\n", ref_n_frames);
    }
    else if (ctx->voice_clone && !ctx->xvector_only && !ctx->graft_mode && ctx->ref_text
             && ref_text_tokens && ref_text_token_len > 0) {
        icl_mode = 1;
        float *ref_audio_samples = NULL;
        int ref_n_samples = 0, ref_sr = 0;
        if (qwen_read_wav(ctx->ref_audio_path, &ref_audio_samples, &ref_n_samples, &ref_sr) != 0) {
            fprintf(stderr, "Error: failed to read reference audio %s\n", ctx->ref_audio_path);
            icl_mode = 0;
        } else {
            if (ref_sr != QWEN_TTS_SAMPLE_RATE && !ctx->silent)
                fprintf(stderr, "Warning: ref audio sample rate %d, expected %d\n",
                        ref_sr, QWEN_TTS_SAMPLE_RATE);
            qwen_trim_trailing_silence(ref_audio_samples, &ref_n_samples, ref_sr, ctx->silent);
            if (qwen_speech_encoder_encode(ctx, ref_audio_samples, ref_n_samples,
                                            &ref_codes, &ref_n_frames) != 0) {
                fprintf(stderr, "Error: speech encoder failed\n");
                icl_mode = 0;
            }
            free(ref_audio_samples);
            ref_codes_owned = 1;
        }
    }

    if (icl_mode && ref_n_frames > 0) {
        int cap = ctx->icl_frames_cap;
        { const char *e = getenv("QWEN_ICL_FRAMES"); if (e && e[0]) cap = atoi(e); }
        if (cap > 0 && cap < ref_n_frames) {
            if (!ctx->silent)
                fprintf(stderr, "ICL: capping ref frames %d -> %d (anchor dilution)\n",
                        ref_n_frames, cap);
            ref_n_frames = cap;
        }
    }

    int sec2_len = codec_len - 1;
    int inst_len = instruct_tokens ? instruct_token_len : 0;

    int sec3_len, sec4_len;
    if (icl_mode) {
        sec3_len = ref_text_token_len + text_content_len + 1;
        sec4_len = ref_n_frames + 1;
    } else {
        sec3_len = text_content_len + 1;
        sec4_len = 1;
    }
    int prefill_len = inst_len + role_len + sec2_len + sec3_len + sec4_len;

    float *input_embeds = (float *)aligned_calloc((int64_t)prefill_len * h, sizeof(float));
    float *tmp_embed = (float *)aligned_malloc(h * sizeof(float));
    int pos = 0;

    {
        int cacheable = !icl_mode && !ctx->voice_clone;
        uint64_t ih = 0;
        if (cacheable) {
            ih = qwen_prefix_hash(all_tokens, role_len);
            if (inst_len) ih ^= qwen_prefix_hash(instruct_tokens, inst_len) * 1099511628211ULL;
            ih ^= qwen_prefix_hash(codec_tokens, sec2_len) * 14695981039346656037ULL;
        }
        qwen_talker_prefix_key(ctx, cacheable ? (inst_len + role_len + sec2_len) : 0,
                               ctx->speaker_id, ctx->language_id, 0, ih);
    }

    for (int i = 0; i < inst_len; i++) {
        embed_one_text_token(ctx, instruct_tokens[i], input_embeds + (int64_t)pos * h);
        pos++;
    }
    free(instruct_tokens);

    for (int i = 0; i < role_len; i++) {
        embed_one_text_token(ctx, all_tokens[i], input_embeds + (int64_t)pos * h);
        if (ctx->debug) {
            float *e = input_embeds + (int64_t)pos * h;
            fprintf(stderr, "[PROMPT] pos=%d role token=%d embed[:5]=[%.4f,%.4f,%.4f,%.4f,%.4f]\n",
                    pos, all_tokens[i], e[0], e[1], e[2], e[3], e[4]);
        }
        pos++;
    }

    for (int i = 0; i < sec2_len; i++) {
        float *dst = input_embeds + (int64_t)pos * h;
        if (i < sec2_len - 1) {
            memcpy(dst, tts_pad_embed, h * sizeof(float));
        } else {
            memcpy(dst, tts_bos_embed, h * sizeof(float));
        }
        if (codec_tokens[i] == -1 && ctx->voice_clone && ctx->speaker_embedding) {
            float emb_norm = 0;
            for (int j = 0; j < h; j++) emb_norm += ctx->speaker_embedding[j] * ctx->speaker_embedding[j];
            emb_norm = sqrtf(emb_norm);

            float ref_norm = 0;
            {
                float tmp_ref[4096];
                lookup_codec_embed(ctx, 3061, tmp_ref);
                for (int j = 0; j < h; j++) ref_norm += tmp_ref[j] * tmp_ref[j];
                ref_norm = sqrtf(ref_norm);
            }

            float scale = (ref_norm > 0.1f && emb_norm > 0.1f) ? ref_norm / emb_norm : 1.0f;
            float spk_scale_env = 1.0f;
            { const char *sse = getenv("QWEN_SPK_SCALE"); if (sse && sse[0]) spk_scale_env = (float)atof(sse); }
            scale *= spk_scale_env;
            for (int j = 0; j < h; j++) dst[j] += ctx->speaker_embedding[j] * scale;

            if (!ctx->silent && fabsf(scale - 1.0f) > 0.01f)
                fprintf(stderr, "  Speaker embedding norm scaled: %.2f -> %.2f (scale=%.4f, QWEN_SPK_SCALE=%.2f)\n",
                        emb_norm, emb_norm * scale, scale, spk_scale_env);
            if (ctx->debug)
                fprintf(stderr, "[PROMPT] pos=%d SPEAKER EMBED injected (h=%d, raw_norm=%.4f, target_norm=%.4f, scale=%.4f)\n",
                        pos, h, emb_norm, ref_norm, scale);

        } else {
            lookup_codec_embed(ctx, codec_tokens[i], tmp_embed);
            for (int j = 0; j < h; j++) dst[j] += tmp_embed[j];
        }
        pos++;
    }

    if (icl_mode) {
        for (int i = 0; i < sec3_len; i++) {
            float *dst = input_embeds + (int64_t)pos * h;
            if (i < ref_text_token_len) {
                embed_one_text_token(ctx, ref_text_tokens[i], dst);
            } else if (i < ref_text_token_len + text_content_len) {
                embed_one_text_token(ctx, all_tokens[role_len + (i - ref_text_token_len)], dst);
            } else {
                memcpy(dst, tts_eos_embed, h * sizeof(float));
            }
            for (int j = 0; j < h; j++) dst[j] += codec_pad_embed[j];
            pos++;
        }

        for (int i = 0; i < sec4_len; i++) {
            float *dst = input_embeds + (int64_t)pos * h;
            memcpy(dst, tts_pad_embed, h * sizeof(float));
            if (i == 0) {
                for (int j = 0; j < h; j++) dst[j] += codec_bos_embed[j];
            } else {
                int frame = i - 1;
                int code0 = ref_codes[frame * 16];
                lookup_codec_embed(ctx, code0, tmp_embed);
                for (int j = 0; j < h; j++) dst[j] += tmp_embed[j];
                for (int g = 0; g < 15; g++) {
                    int code_g = ref_codes[frame * 16 + g + 1];
                    if (ctx->cp_codec_emb_bf16[g] && code_g >= 0
                        && code_g < ctx->config.codebook_size) {
                        const uint16_t *emb = ctx->cp_codec_emb_bf16[g]
                                              + (int64_t)code_g * h;
                        qwen_bf16_accum_f32(dst, emb, h);
                    }
                }
            }
            pos++;
        }
    } else {
        for (int i = 0; i < sec3_len; i++) {
            float *dst = input_embeds + (int64_t)pos * h;
            if (i < text_content_len) {
                embed_one_text_token(ctx, all_tokens[role_len + i], dst);
            } else {
                memcpy(dst, tts_eos_embed, h * sizeof(float));
            }
            for (int j = 0; j < h; j++) dst[j] += codec_pad_embed[j];
            pos++;
        }

        {
            float *dst = input_embeds + (int64_t)pos * h;
            memcpy(dst, tts_pad_embed, h * sizeof(float));
            for (int j = 0; j < h; j++) dst[j] += codec_bos_embed[j];
            pos++;
        }
    }

    free(all_tokens);
    free(tmp_embed);
    free(ref_text_tokens);
    if (ref_codes_owned) free(ref_codes);

    free(codec_pad_embed);
    free(codec_bos_embed);

    if (!ctx->silent) {
        if (ctx->voice_clone)
            fprintf(stderr, "Voice clone: %s (x-vector%s)\n",
                    ctx->ref_audio_path ? ctx->ref_audio_path : "(loaded from file)",
                    ctx->xvector_only ? " only" : " + ICL");
        else
            fprintf(stderr, "Speaker: %d, Language: %d\n", ctx->speaker_id, ctx->language_id);
        if (icl_mode)
            fprintf(stderr, "Prefill: %d positions (instruct=%d, role=%d, codec=%d, "
                    "icl_text=%d, icl_codes=%d)\n",
                    prefill_len, inst_len, role_len, sec2_len, sec3_len, sec4_len);
        else
            fprintf(stderr, "Prefill: %d positions (instruct=%d, role=%d, codec=%d, "
                    "text+eos=%d, final=%d)\n",
                    prefill_len, inst_len, role_len, sec2_len, sec3_len, sec4_len);
    }

    if (ctx->debug && ctx->speech_dec.pre_conv_weight) {
        fprintf(stderr, "[CORR] pre-prefill: pre_conv_w[0]=%.6f\n", ctx->speech_dec.pre_conv_weight[0]);
    }

    int delta_start = 0;
    if (ctx->prev_input_embeds && ctx->prev_prefill_len > 0) {
        int max_match = (prefill_len < ctx->prev_prefill_len) ? prefill_len : ctx->prev_prefill_len;
        for (int t = 0; t < max_match; t++) {
            if (memcmp(input_embeds + (int64_t)t * h,
                       ctx->prev_input_embeds + (int64_t)t * h,
                       h * sizeof(float)) != 0)
                break;
            delta_start = t + 1;
        }
    }

    if (delta_start >= prefill_len) delta_start = 0;

#if defined(QWEN_HAVE_CUDA) || defined(QWEN_HAVE_METAL)
    {
        extern void *g_gpu_fused_owner;
        int fused_owner = 0;
#ifdef QWEN_HAVE_CUDA
        { extern void *g_cuda_talker_state;
          if (g_cuda_talker_state && ctx == g_gpu_fused_owner) fused_owner = 1; }
#endif
#ifdef QWEN_HAVE_METAL
        { extern void *g_metal_talker_state;
          if (g_metal_talker_state && ctx == g_gpu_fused_owner) fused_owner = 1; }
#endif
        if (fused_owner && ctx->ml_steer && ctx->ml_steer_weight != 0.0f) delta_start = 0;
    }
#endif

    ctx->kv_len = delta_start;

    ctx->ml_steer_w_eff = 0.0f;

    if (prefill_len > ctx->rope_cache_len) {
        fprintf(stderr, "Error: prompt too long (%d tokens > RoPE cache %d); shorten the text.\n",
                prefill_len, ctx->rope_cache_len);
        free(input_embeds);
        return -1;
    }

    double t_prefill = time_ms();
    if (delta_start < prefill_len) {
        if (delta_start > 0) {
            float *dummy_hidden = (float *)malloc(h * sizeof(float));
            for (int t = delta_start; t < prefill_len; t++) {
                if (qwen_talker_step(ctx, input_embeds + (int64_t)t * h, dummy_hidden) != 0) {
                    free(input_embeds); free(dummy_hidden);
                    return -1;
                }
            }
            free(dummy_hidden);
        } else {
            if (qwen_talker_prefill(ctx, input_embeds, prefill_len) != 0) {
                free(input_embeds);
                return -1;
            }
        }
    }
#ifdef QWEN_HAVE_CUDA
    {
        extern void *g_cuda_talker_state, *g_gpu_fused_owner;
        extern void qwen_cuda_talker_upload_kv(void *, qwen_tts_ctx_t *, int);
        if (g_cuda_talker_state && ctx == g_gpu_fused_owner && delta_start == 0 &&
            !(ctx->ml_steer && ctx->ml_steer_w_eff != 0.0f))
            qwen_cuda_talker_upload_kv(g_cuda_talker_state, ctx, ctx->kv_len);
    }
#endif
#ifdef QWEN_HAVE_METAL
    {
        extern void *g_metal_talker_state, *g_gpu_fused_owner;
        extern void qwen_metal_talker_upload_kv(void *, qwen_tts_ctx_t *, int);
        if (g_metal_talker_state && ctx == g_gpu_fused_owner && delta_start == 0 &&
            !(ctx->ml_steer && ctx->ml_steer_w_eff != 0.0f))
            qwen_metal_talker_upload_kv(g_metal_talker_state, ctx, ctx->kv_len);
    }
#endif
    double prefill_ms = time_ms() - t_prefill;
    if (!ctx->silent) {
        if (delta_start > 0)
            fprintf(stderr, "  Prefill: %.0f ms (delta: %d new tokens, %d cached)\n",
                    prefill_ms, prefill_len - delta_start, delta_start);
        else
            fprintf(stderr, "  Prefill: %.0f ms\n", prefill_ms);
    }

    if (!ctx->prev_input_embeds || ctx->prev_prefill_len < prefill_len) {
        free(ctx->prev_input_embeds);
        ctx->prev_input_embeds = (float *)malloc((int64_t)prefill_len * h * sizeof(float));
    }
    if (ctx->prev_input_embeds) {
        memcpy(ctx->prev_input_embeds, input_embeds, (int64_t)prefill_len * h * sizeof(float));
        ctx->prev_prefill_len = prefill_len;
    }

    free(input_embeds);

    if (ctx->debug && ctx->speech_dec.pre_conv_weight) {
        fprintf(stderr, "[CORR] post-prefill: pre_conv_w[0]=%.6f\n", ctx->speech_dec.pre_conv_weight[0]);
    }

    if (ctx->prefill_only) {
        ctx->bg_text_content_len = text_content_len;
        return 0;
    }

    float *last_hidden = (float *)malloc(h * sizeof(float));
    qwen_rms_norm(last_hidden, ctx->dec_x, ctx->talker_norm, 1, h, ctx->config.rms_norm_eps);

    int max_frames = ctx->max_tokens;
    if (max_frames > ctx->rope_cache_len - prefill_len)
        max_frames = ctx->rope_cache_len - prefill_len;
    ctx->codec_codes = (int *)realloc(ctx->codec_codes, (int64_t)max_frames * 16 * sizeof(int));
    ctx->codec_frames = 0;
    ctx->prev_tokens = (int *)realloc(ctx->prev_tokens, max_frames * sizeof(int));
    ctx->n_prev_tokens = 0;
    ctx->logits = (float *)realloc(ctx->logits, ctx->config.codec_vocab_size * sizeof(float));

    double t_cp_total = 0, t_talker_step_total = 0, t_embed_total = 0;
    float *step_embed = (float *)malloc(h * sizeof(float));

    int   *tf_codes = NULL;
    int    tf_nframes = 0;
    int    tf_cb_keep = 0;
    { const char *k = getenv("QWEN_TF_CB_KEEP"); if (k && *k) { tf_cb_keep = atoi(k); if (tf_cb_keep < 0) tf_cb_keep = 0; if (tf_cb_keep > 16) tf_cb_keep = 16; } }
    int    tf_prefix_mode = 0;
    { const char *k = getenv("QWEN_TF_PREFIX"); tf_prefix_mode = (k && k[0] && k[0] != '0'); }
    FILE  *code0_fp = NULL;
    {
        const char *c0p = getenv("QWEN_DUMP_CODE0");
        if (c0p && *c0p) code0_fp = fopen(c0p, "w");
    }
    {
        const char *tfp = getenv("QWEN_TF_CODES");
        if (tfp && *tfp) {
            FILE *tf = fopen(tfp, "r");
            if (tf) {
                int cap = 256;
                tf_codes = (int *)malloc((size_t)cap * 16 * sizeof(int));
                char line[1024];
                while (fgets(line, sizeof(line), tf)) {
                    int c[16], n = 0;
                    char *p = line;
                    while (n < 16) {
                        char *end; long v = strtol(p, &end, 10);
                        if (end == p) break;
                        c[n++] = (int)v; p = end;
                    }
                    if (n < 16) continue;
                    if (tf_nframes >= cap) {
                        cap *= 2;
                        tf_codes = (int *)realloc(tf_codes, (size_t)cap * 16 * sizeof(int));
                    }
                    memcpy(tf_codes + (size_t)tf_nframes * 16, c, 16 * sizeof(int));
                    tf_nframes++;
                }
                fclose(tf);
                if (!ctx->silent)
                    fprintf(stderr, "  [QWEN_TF_CODES] teacher-forcing replay: %d reference frames\n", tf_nframes);
            }
        }
    }

    decoder_thread_t dt_state;
    pthread_t dt_thread;
    int dt_no_overlap = (getenv("QWEN_NO_OVERLAP") != NULL);
#ifdef QWEN_HAVE_CUDA
    { extern int g_cuda_decoder_conv_on; if (g_cuda_decoder_conv_on && !ctx->stream) dt_no_overlap = 1; }
#endif
    qwen_sd_stream_init(&ctx->sd_stream);
    dt_init(&dt_state, ctx, max_frames);
    if (ctx->stream && ctx->audio_cb) {
        dt_state.audio_cb = ctx->audio_cb;
        dt_state.audio_cb_userdata = ctx->audio_cb_userdata;
    }
    if (icl_mode) {
        int trim_frames = 2;
        const char *e = getenv("QWEN_ICL_TRIM_FRAMES");
        if (e) trim_frames = atoi(e);
        if (trim_frames > 0) dt_state.trim_head_left = trim_frames * 1920;
    }
    if (!dt_no_overlap) {
        pthread_create(&dt_thread, NULL, decoder_thread_fn, &dt_state);
        int nt = qwen_get_threads();
        int gen_blas = nt > 1 ? nt - 1 : 1;
        { const char *e = getenv("QWEN_BLAS_GEN_THREADS");
          if (e && atoi(e) > 0) gen_blas = atoi(e); }
        qwen_blas_set_threads(gen_blas);
    }

    for (int frame = 0; frame < max_frames; frame++) {
        qwen_census_frame();
        if (ctx->codec_head_q4)
            qwen_matvec_q4_0(ctx->logits, ctx->codec_head_q4, last_hidden, ctx->config.codec_vocab_size, h);
        else
            matvec_bf16(ctx->logits, ctx->codec_head_bf16, last_hidden, ctx->config.codec_vocab_size, h);

        for (int t = 0; t < ctx->config.codec_vocab_size; t++) {
            if (ctx->logits[t] > 100.0f) ctx->logits[t] = 100.0f;
            if (ctx->logits[t] < -100.0f) ctx->logits[t] = -100.0f;
        }

        for (int t = 2048; t < ctx->config.codec_vocab_size; t++)
            if (t != QWEN_TTS_CODEC_EOS) ctx->logits[t] = -1e30f;

        if (ctx->eos_strategy != QWEN_EOS_OFF && frame < ctx->eos_suppress_frames)
            ctx->logits[QWEN_TTS_CODEC_EOS] = -1e30f;

        if (ctx->eos_strategy == QWEN_EOS_V1 || ctx->eos_strategy == QWEN_EOS_V2) {
            float expected = (ctx->eos_strategy == QWEN_EOS_V2)
                ? (float)ctx->eos_overhead_frames + text_content_len * ctx->eos_frames_per_token
                : text_content_len * ctx->eos_frames_per_token;
            float boost_start = ctx->eos_start_multiple * expected;
            if (expected > 0.0f && (float)frame > boost_start) {
                float boost = ctx->eos_ramp_per_frame * ((float)frame - boost_start);
                if (boost > ctx->eos_ramp_cap) boost = ctx->eos_ramp_cap;
                ctx->logits[QWEN_TTS_CODEC_EOS] += boost;
            }
        } else if (ctx->eos_strategy == QWEN_EOS_TOPK) {
            if (frame >= ctx->eos_suppress_frames) {
                float top[64];
                int kmax = (int)(sizeof(top) / sizeof(top[0]));
                int k = ctx->eos_topk > 0 ? ctx->eos_topk : 50;
                if (k > ctx->config.codec_vocab_size) k = ctx->config.codec_vocab_size;
                if (k > kmax) k = kmax;
                for (int i = 0; i < k; i++) top[i] = -1e30f;
                for (int t = 0; t < ctx->config.codec_vocab_size; t++) {
                    float v = ctx->logits[t];
                    if (v <= top[k - 1]) continue;
                    int i = k - 1;
                    while (i > 0 && top[i - 1] < v) { top[i] = top[i - 1]; i--; }
                    top[i] = v;
                }
                if (ctx->logits[QWEN_TTS_CODEC_EOS] < top[k - 1])
                    ctx->logits[QWEN_TTS_CODEC_EOS] = top[k - 1];
            }
        }

        if (ctx->debug && frame < 30) {
            float eos_logit = ctx->logits[QWEN_TTS_CODEC_EOS];
            int eos_rank = 0;
            for (int t = 0; t < ctx->config.codec_vocab_size; t++)
                if (ctx->logits[t] > eos_logit) eos_rank++;
            fprintf(stderr, "  [frame %d] EOS logit=%.2f rank=%d\n", frame, eos_logit, eos_rank);
        }

        float frame_temp = ctx->temperature;
        int frame_top_k = ctx->top_k;
        if (ctx->greedy_warmup > 0 && frame < ctx->greedy_warmup) {
            frame_temp = 0.0f;
            frame_top_k = 1;
        }
        int code0 = qwen_tts_sample(ctx->logits, ctx->config.codec_vocab_size,
                                     frame_temp, frame_top_k, ctx->top_p,
                                     ctx->rep_penalty, ctx->prev_tokens, ctx->n_prev_tokens);

        if (code0_fp) fprintf(code0_fp, "%d\n", code0);

        if (tf_codes) {
            if (frame >= tf_nframes) {
                if (!tf_prefix_mode) break;
                free(tf_codes); tf_codes = NULL; ctx->tf_ref_codes = NULL;
            }
        }
        if (tf_codes) {
            if (tf_cb_keep < 1) code0 = tf_codes[(int64_t)frame * 16 + 0];
            ctx->tf_ref_codes = tf_codes + (int64_t)frame * 16 + 1;
        }

        if (code0 == QWEN_TTS_CODEC_EOS && tf_codes && tf_cb_keep >= 1)
            code0 = tf_codes[(int64_t)frame * 16 + 0];

        if (code0 == QWEN_TTS_CODEC_EOS) {
            if (!ctx->silent) fprintf(stderr, "  EOS at frame %d\n", frame);
            break;
        }

        ctx->prev_tokens[ctx->n_prev_tokens++] = code0;

        int codes[16]; codes[0] = code0;
        double t_cp_start = time_ms();
        qwen_cp_predict(ctx, last_hidden, code0, codes + 1);
        t_cp_total += time_ms() - t_cp_start;

        if (tf_codes)
            memcpy(codes + tf_cb_keep, tf_codes + (int64_t)frame * 16 + tf_cb_keep,
                   (size_t)(16 - tf_cb_keep) * sizeof(int));

        memcpy(ctx->codec_codes + (int64_t)ctx->codec_frames * 16, codes, 16 * sizeof(int));
        ctx->codec_frames++;

        dt_push_frames(&dt_state, codes, 1);

        if (ctx->debug) {
            fprintf(stderr, "  [frame %d] codes:", frame);
            for (int g = 0; g < 16; g++) fprintf(stderr, " %d", codes[g]);
            fprintf(stderr, "\n");
        }

        if (ctx->debug && frame == 0 && ctx->speech_dec.pre_conv_weight) {
            fprintf(stderr, "[CORR] post-frame0: pre_conv_w[0]=%.6f\n", ctx->speech_dec.pre_conv_weight[0]);
        }

        if (!ctx->silent && frame % 50 == 0 && frame > 0)
            fprintf(stderr, "\r  Frame %d/%d (%.1fs audio)...", frame, max_frames, frame / 12.5);

        if (ctx->stream && ctx->audio_cb && dt_state.cb_aborted) {
            if (!ctx->silent) fprintf(stderr, "\n  Streaming aborted by callback\n");
            break;
        }

        double t_embed_start = time_ms();
        lookup_codec_embed(ctx, code0, step_embed);
        for (int g = 0; g < 15; g++) {
            int code_g = codes[g + 1];
            if (ctx->cp_codec_emb_bf16[g] && code_g >= 0 && code_g < ctx->config.codebook_size) {
                const uint16_t *emb = ctx->cp_codec_emb_bf16[g] + (int64_t)code_g * h;
                qwen_bf16_accum_f32(step_embed, emb, h);
            }
        }

        for (int j = 0; j < h; j++) step_embed[j] += tts_pad_embed[j];
        t_embed_total += time_ms() - t_embed_start;

        if (ctx->debug && frame < 2) {
            fprintf(stderr, "  [frame %d] step_embed[:5]=[%.6f,%.6f,%.6f,%.6f,%.6f]\n",
                    frame, step_embed[0], step_embed[1], step_embed[2], step_embed[3], step_embed[4]);
            fprintf(stderr, "  [frame %d] PRE last_hidden[:5]=[%.6f,%.6f,%.6f,%.6f,%.6f]\n",
                    frame, last_hidden[0], last_hidden[1], last_hidden[2], last_hidden[3], last_hidden[4]);
        }
        if (ctx->ml_steer && ctx->ml_steer_weight != 0.0f) {
            if (ctx->ml_steer_frames > 0 && frame >= ctx->ml_steer_frames) {
                ctx->ml_steer_w_eff = 0.0f;
            } else {
                float g = ctx->ml_steer_decay > 0.0f ? ctx->ml_steer_decay : 1.0f;
                ctx->ml_steer_w_eff = ctx->ml_steer_weight * powf(g, (float)frame);
            }
        }
        double t_step_start = time_ms();
        if (qwen_talker_step(ctx, step_embed, last_hidden) != 0) {
            free(step_embed); free(last_hidden);
            dt_finish(&dt_state);
            if (dt_no_overlap) decoder_thread_fn(&dt_state); else pthread_join(dt_thread, NULL);
            qwen_blas_set_threads(qwen_get_threads());
            qwen_sd_stream_free(&ctx->sd_stream); dt_free(&dt_state);
            return -1;
        }
        t_talker_step_total += time_ms() - t_step_start;
        if (ctx->debug && frame < 2) {
            fprintf(stderr, "  [frame %d] POST last_hidden[:5]=[%.6f,%.6f,%.6f,%.6f,%.6f]\n",
                    frame, last_hidden[0], last_hidden[1], last_hidden[2], last_hidden[3], last_hidden[4]);
        }
    }

    free(step_embed);
    free(last_hidden);
    if (tf_codes) { free(tf_codes); ctx->tf_ref_codes = NULL; }
    if (code0_fp) fclose(code0_fp);

    double t_talker_end = time_ms();
    double t_total_gen = t_talker_end - t_prefill - prefill_ms;
    double t_codec_head = t_total_gen - t_talker_step_total - t_cp_total - t_embed_total;
    if (!ctx->silent) {
        fprintf(stderr, "\n  Generated %d frames (%.1fs audio)\n", ctx->codec_frames, ctx->codec_frames / 12.5);
        fprintf(stderr, "  Talker step: %.0f ms (%.1f ms/f), Code Predictor: %.0f ms (%.1f ms/f)\n",
                t_talker_step_total, ctx->codec_frames > 0 ? t_talker_step_total / ctx->codec_frames : 0,
                t_cp_total, ctx->codec_frames > 0 ? t_cp_total / ctx->codec_frames : 0);
        fprintf(stderr, "  Embed: %.0f ms, Codec head+sampling: %.0f ms\n", t_embed_total, t_codec_head);
#ifdef CP_MICROBENCH
        qwen_cp_microbench_report(ctx->codec_frames);
#endif
    }

    if (ctx->codec_frames == 0) {
        dt_finish(&dt_state);
        if (dt_no_overlap) decoder_thread_fn(&dt_state); else pthread_join(dt_thread, NULL);
        qwen_blas_set_threads(qwen_get_threads());
        qwen_sd_stream_free(&ctx->sd_stream); dt_free(&dt_state);
        *out_samples = NULL; *out_n_samples = 0;
        return 0;
    }

    float *audio; int n_samples;
    double t_dec_start = time_ms();

    dt_finish(&dt_state);
    if (dt_no_overlap) decoder_thread_fn(&dt_state); else pthread_join(dt_thread, NULL);
    qwen_blas_set_threads(qwen_get_threads());
    qwen_sd_stream_free(&ctx->sd_stream);

    double dt_decode_ms = dt_state.decode_ms;
    double dt_drain_ms = time_ms() - t_dec_start;
    double ttfa_ms = (dt_state.first_chunk_ms > 0) ? dt_state.first_chunk_ms - t_start : -1;

    if (dt_state.audio_cb) {
        audio = NULL;
        n_samples = dt_state.audio_len;
        dt_free(&dt_state);
    } else {
        audio = dt_state.audio_buf;
        n_samples = dt_state.audio_len;
        dt_state.audio_buf = NULL;
        dt_free(&dt_state);
    }

    if (!ctx->silent)
        fprintf(stderr, "  Speech decoder: %.0f ms total (%.0f ms drain after gen)\n",
                dt_decode_ms, dt_drain_ms);

    *out_samples = audio;
    *out_n_samples = n_samples;

    if (!ctx->silent) {
        float audio_dur = (float)n_samples / 24000.0f;
        float proc_time = (time_ms() - t_start) / 1000.0f;
        fprintf(stderr, "Audio: %.1fs generated in %.1fs (RTF %.2f)\n",
                audio_dur, proc_time, proc_time / audio_dur);
        if (ttfa_ms >= 0)
            fprintf(stderr, "  TTFA: %.0f ms (first audio chunk, %d-frame chunk)\n",
                    ttfa_ms, dt_state.chunk_frames);
    }

    return 0;
}

int qwen_tts_batch_max_prompt(void) {
    static int v = -1;
    if (v < 0) {
        const char *e = getenv("QWEN_BATCH_MAX_PROMPT");
        v = (e && atoi(e) > 0) ? atoi(e) : 512;
        if (v < 32) v = 32;
    }
    return v;
}
int qwen_tts_batch_max_frames(void) {
    static int v = -1;
    if (v < 0) {
        const char *e = getenv("QWEN_BATCH_MAX_FRAMES");
        v = (e && atoi(e) > 0) ? atoi(e) : 600;
        if (v < 32) v = 32;
    }
    return v;
}

int qwen_tts_generate_batch(qwen_tts_ctx_t *ctx, char **chunks, int nc,
                            float chunk_pause, float **out_samples, int *out_n_samples) {
    if (nc <= 0) { *out_samples = NULL; *out_n_samples = 0; return 0; }
    if (ctx->layers[0].wq_bf16 == NULL) return -2;
    int h = ctx->config.hidden_size;
    int kvd = ctx->config.num_kv_heads * ctx->config.head_dim;
    int num_layers = ctx->config.num_layers;
    int vocab = ctx->config.codec_vocab_size;
    int cb = ctx->config.codebook_size;
    float eps = ctx->config.rms_norm_eps;
    const int GMAX = 8;
    int GEN_CAP = ctx->max_tokens;
    { int lim = qwen_tts_batch_max_frames(); if (GEN_CAP > lim) GEN_CAP = lim; }
    if (GEN_CAP < 32) GEN_CAP = 32;
    const int SR = QWEN_TTS_SAMPLE_RATE;

    float *out = NULL; size_t out_n = 0, out_cap = 0;
    #define BG_APPEND(src, cnt) do {                                           \
        size_t _c = (cnt);                                                     \
        if (out_n + _c > out_cap) { out_cap = (out_n + _c) * 2 + 4096;         \
            float *_t = (float *)realloc(out, out_cap * sizeof(float));        \
            if (!_t) { free(out); return -1; } out = _t; }                     \
        if (src) memcpy(out + out_n, (src), _c * sizeof(float));               \
        else memset(out + out_n, 0, _c * sizeof(float));                       \
        out_n += _c;                                                           \
    } while (0)

    qwen_set_seed(ctx->seed);

    for (int g0 = 0; g0 < nc; g0 += GMAX) {
        int B = nc - g0 < GMAX ? nc - g0 : GMAX;

        int *prompt_len = (int *)calloc(B, sizeof(int));
        int *tcl = (int *)calloc(B, sizeof(int));
        float *seed_hidden = (float *)malloc((size_t)B * h * sizeof(float));
        uint16_t **tk = (uint16_t **)calloc(B, sizeof(uint16_t *));
        uint16_t **tv = (uint16_t **)calloc(B, sizeof(uint16_t *));
        int maxpl = 0, ok = 1;
        ctx->prefill_only = 1;
        for (int b = 0; b < B && ok; b++) {
            ctx->prev_prefill_len = 0;
            if (qwen_tts_generate(ctx, chunks[g0 + b], NULL, NULL) != 0) { ok = 0; break; }
            int pl = ctx->kv_len; prompt_len[b] = pl; tcl[b] = ctx->bg_text_content_len;
            qwen_rms_norm(seed_hidden + (size_t)b * h, ctx->dec_x, ctx->talker_norm, 1, h, eps);
            size_t bytes = (size_t)num_layers * pl * kvd * sizeof(uint16_t);
            tk[b] = (uint16_t *)malloc(bytes); tv[b] = (uint16_t *)malloc(bytes);
            if (!tk[b] || !tv[b]) { ok = 0; break; }
            for (int L = 0; L < num_layers; L++) {
                memcpy(tk[b] + (size_t)L * pl * kvd,
                       ctx->kv_cache_k + (size_t)L * ctx->kv_max * kvd, (size_t)pl * kvd * sizeof(uint16_t));
                memcpy(tv[b] + (size_t)L * pl * kvd,
                       ctx->kv_cache_v + (size_t)L * ctx->kv_max * kvd, (size_t)pl * kvd * sizeof(uint16_t));
            }
            if (pl > maxpl) maxpl = pl;
        }
        ctx->prefill_only = 0;
        if (!ok) {
            for (int b = 0; b < B; b++) { free(tk[b]); free(tv[b]); }
            free(tk); free(tv); free(prompt_len); free(tcl); free(seed_hidden); free(out);
            return -1;
        }

        int kv_max = maxpl + GEN_CAP + 4;
        qwen_batch_t *bb = qwen_batch_alloc(ctx, B, kv_max);
        if (bb && getenv("QWEN_BATCH_FORCE_MATVEC")) bb->force_matvec = 1;
        if (!bb) {
            for (int b = 0; b < B; b++) { free(tk[b]); free(tv[b]); }
            free(tk); free(tv); free(prompt_len); free(tcl); free(seed_hidden); free(out);
            return -1;
        }
        for (int b = 0; b < B; b++) {
            int pl = prompt_len[b];
            for (int L = 0; L < num_layers; L++) {
                size_t dst = ((size_t)b * num_layers + L) * kv_max * kvd;
                memcpy(bb->kv_k + dst, tk[b] + (size_t)L * pl * kvd, (size_t)pl * kvd * sizeof(uint16_t));
                memcpy(bb->kv_v + dst, tv[b] + (size_t)L * pl * kvd, (size_t)pl * kvd * sizeof(uint16_t));
            }
            free(tk[b]); free(tv[b]);
        }
        free(tk); free(tv);

        int *pos = (int *)malloc((size_t)B * sizeof(int));
        uint8_t *active = (uint8_t *)malloc((size_t)B);
        int *nprev = (int *)calloc(B, sizeof(int));
        int *chframes = (int *)calloc(B, sizeof(int));
        int **prev_tok = (int **)malloc((size_t)B * sizeof(int *));
        int **chcodes = (int **)malloc((size_t)B * sizeof(int *));
        float *last_hidden = (float *)malloc((size_t)B * h * sizeof(float));
        float *logits = (float *)malloc((size_t)vocab * sizeof(float));
        float *step_embed = (float *)malloc((size_t)B * h * sizeof(float));
        int *code0 = (int *)malloc((size_t)B * sizeof(int));
        int *cpcodes = (int *)malloc((size_t)B * 15 * sizeof(int));
        for (int b = 0; b < B; b++) {
            pos[b] = prompt_len[b]; active[b] = 1;
            prev_tok[b] = (int *)malloc((size_t)GEN_CAP * sizeof(int));
            chcodes[b] = (int *)malloc((size_t)GEN_CAP * 16 * sizeof(int));
        }
        memcpy(last_hidden, seed_hidden, (size_t)B * h * sizeof(float));
        const float *tts_pad = ctx->cached_tts_pad_embed;
        int n_active = B;

        for (int frame = 0; frame < GEN_CAP && n_active > 0; frame++) {
            qwen_census_frame_at(2);
            for (int b = 0; b < B; b++) {
                if (!active[b]) { code0[b] = 0; continue; }
                if (ctx->codec_head_q4)
                    qwen_matvec_q4_0(logits, ctx->codec_head_q4, last_hidden + (size_t)b * h, vocab, h);
                else
                    matvec_bf16(logits, ctx->codec_head_bf16, last_hidden + (size_t)b * h, vocab, h);
                for (int t = 0; t < vocab; t++) { if (logits[t] > 100.0f) logits[t] = 100.0f; if (logits[t] < -100.0f) logits[t] = -100.0f; }
                for (int t = 2048; t < vocab; t++) if (t != QWEN_TTS_CODEC_EOS) logits[t] = -1e30f;
                if (frame < 2) logits[QWEN_TTS_CODEC_EOS] = -1e30f;
                int ef = tcl[b] * 3, bs = ef * 2;
                if (ef > 0 && frame > bs) { float bo = 0.5f * (frame - bs); if (bo > 10.0f) bo = 10.0f; logits[QWEN_TTS_CODEC_EOS] += bo; }
                float ft = ctx->temperature; int ftk = ctx->top_k;
                if (ctx->greedy_warmup > 0 && frame < ctx->greedy_warmup) { ft = 0.0f; ftk = 1; }
                int c0 = qwen_tts_sample(logits, vocab, ft, ftk, ctx->top_p, ctx->rep_penalty, prev_tok[b], nprev[b]);
                if (c0 == QWEN_TTS_CODEC_EOS || chframes[b] >= GEN_CAP) { active[b] = 0; n_active--; code0[b] = 0; continue; }
                code0[b] = c0; prev_tok[b][nprev[b]++] = c0;
            }
            if (n_active == 0) break;

            qwen_batch_cp_predict(ctx, bb, last_hidden, code0, cpcodes, NULL);

            for (int b = 0; b < B; b++) {
                float *se = step_embed + (size_t)b * h;
                if (!active[b]) { memset(se, 0, (size_t)h * sizeof(float)); continue; }
                int frame16[16]; frame16[0] = code0[b];
                for (int g = 0; g < 15; g++) frame16[g + 1] = cpcodes[(size_t)b * 15 + g];
                memcpy(chcodes[b] + (size_t)chframes[b] * 16, frame16, 16 * sizeof(int));
                chframes[b]++;
                lookup_codec_embed(ctx, code0[b], se);
                for (int g = 0; g < 15; g++) {
                    int cg = frame16[g + 1];
                    if (ctx->cp_codec_emb_bf16[g] && cg >= 0 && cg < cb)
                        qwen_bf16_accum_f32(se, ctx->cp_codec_emb_bf16[g] + (size_t)cg * h, h);
                }
                for (int j = 0; j < h; j++) se[j] += tts_pad[j];
            }

            if (qwen_batch_talker_step_ragged(ctx, bb, step_embed, pos, active, last_hidden) != 0) break;

            for (int b = 0; b < B; b++) if (active[b]) pos[b]++;
        }

        for (int b = 0; b < B; b++) {
            if (chframes[b] <= 0) continue;
            if ((g0 + b) > 0 && out_n > 0 && chunk_pause > 0) BG_APPEND(NULL, (size_t)(chunk_pause * SR));
            float *aud = NULL; int an = 0;
            if (qwen_speech_decoder_decode(ctx, chcodes[b], chframes[b], &aud, &an) == 0 && aud && an > 0)
                BG_APPEND(aud, (size_t)an);
            free(aud);
        }

        for (int b = 0; b < B; b++) { free(prev_tok[b]); free(chcodes[b]); }
        free(pos); free(active); free(nprev); free(chframes); free(prev_tok); free(chcodes);
        free(last_hidden); free(logits); free(step_embed); free(code0); free(cpcodes);
        free(prompt_len); free(tcl); free(seed_hidden);
        qwen_batch_free(bb);
    }

    #undef BG_APPEND
    *out_samples = out; *out_n_samples = (int)out_n;
    return 0;
}

int qwen_tts_generate_batch_multi(qwen_tts_ctx_t *ctx,
                                  const qwen_batch_req_t *reqs, int nc,
                                  float **out_samples, int *out_n_samples) {
    if (nc <= 0) return 0;
    if (ctx->layers[0].wq_bf16 == NULL) return -2;
    int h = ctx->config.hidden_size;
    int kvd = ctx->config.num_kv_heads * ctx->config.head_dim;
    int num_layers = ctx->config.num_layers;
    int vocab = ctx->config.codec_vocab_size;
    int cb = ctx->config.codebook_size;
    float eps = ctx->config.rms_norm_eps;
    const int GMAX = 8;
    int GEN_CAP = ctx->max_tokens;
    { int lim = qwen_tts_batch_max_frames(); if (GEN_CAP > lim) GEN_CAP = lim; }
    if (GEN_CAP < 32) GEN_CAP = 32;

    for (int i = 0; i < nc; i++) { out_samples[i] = NULL; out_n_samples[i] = 0; }

    for (int g0 = 0; g0 < nc; g0 += GMAX) {
        int B = nc - g0 < GMAX ? nc - g0 : GMAX;

        int *prompt_len = (int *)calloc(B, sizeof(int));
        int *tcl = (int *)calloc(B, sizeof(int));
        float *seed_hidden = (float *)malloc((size_t)B * h * sizeof(float));
        uint16_t **tk = (uint16_t **)calloc(B, sizeof(uint16_t *));
        uint16_t **tv = (uint16_t **)calloc(B, sizeof(uint16_t *));
        float *p_temp = (float *)malloc((size_t)B * sizeof(float));
        int   *p_topk = (int *)malloc((size_t)B * sizeof(int));
        float *p_topp = (float *)malloc((size_t)B * sizeof(float));
        float *p_rep  = (float *)malloc((size_t)B * sizeof(float));
        int   *p_gw   = (int *)malloc((size_t)B * sizeof(int));
        uint32_t *rng = (uint32_t *)malloc((size_t)B * sizeof(uint32_t));
        int maxpl = 0, ok = 1;
        int sv_spk = ctx->speaker_id, sv_lang = ctx->language_id;
        ctx->prefill_only = 1;
        for (int b = 0; b < B && ok; b++) {
            const qwen_batch_req_t *rq = &reqs[g0 + b];
            ctx->speaker_id = rq->speaker_id;
            ctx->language_id = rq->language_id;
            ctx->prev_prefill_len = 0;
            if (qwen_tts_generate(ctx, rq->text, NULL, NULL) != 0) { ok = 0; break; }
            int pl = ctx->kv_len; prompt_len[b] = pl; tcl[b] = ctx->bg_text_content_len;
            qwen_rms_norm(seed_hidden + (size_t)b * h, ctx->dec_x, ctx->talker_norm, 1, h, eps);
            p_temp[b] = rq->temperature; p_topk[b] = rq->top_k; p_topp[b] = rq->top_p;
            p_rep[b]  = rq->rep_penalty; p_gw[b] = rq->greedy_warmup; rng[b] = rq->seed;
            size_t bytes = (size_t)num_layers * pl * kvd * sizeof(uint16_t);
            tk[b] = (uint16_t *)malloc(bytes); tv[b] = (uint16_t *)malloc(bytes);
            if (!tk[b] || !tv[b]) { ok = 0; break; }
            for (int L = 0; L < num_layers; L++) {
                memcpy(tk[b] + (size_t)L * pl * kvd,
                       ctx->kv_cache_k + (size_t)L * ctx->kv_max * kvd, (size_t)pl * kvd * sizeof(uint16_t));
                memcpy(tv[b] + (size_t)L * pl * kvd,
                       ctx->kv_cache_v + (size_t)L * ctx->kv_max * kvd, (size_t)pl * kvd * sizeof(uint16_t));
            }
            if (pl > maxpl) maxpl = pl;
        }
        ctx->prefill_only = 0;
        ctx->speaker_id = sv_spk; ctx->language_id = sv_lang;
        if (!ok) {
            for (int b = 0; b < B; b++) { free(tk[b]); free(tv[b]); }
            free(tk); free(tv); free(prompt_len); free(tcl); free(seed_hidden);
            free(p_temp); free(p_topk); free(p_topp); free(p_rep); free(p_gw); free(rng);
            return -1;
        }

        int kv_max = maxpl + GEN_CAP + 4;
        qwen_batch_t *bb = qwen_batch_alloc(ctx, B, kv_max);
        if (bb && getenv("QWEN_BATCH_FORCE_MATVEC")) bb->force_matvec = 1;
        if (!bb) {
            for (int b = 0; b < B; b++) { free(tk[b]); free(tv[b]); }
            free(tk); free(tv); free(prompt_len); free(tcl); free(seed_hidden);
            free(p_temp); free(p_topk); free(p_topp); free(p_rep); free(p_gw); free(rng);
            return -1;
        }
        for (int b = 0; b < B; b++) {
            int pl = prompt_len[b];
            for (int L = 0; L < num_layers; L++) {
                size_t dst = ((size_t)b * num_layers + L) * kv_max * kvd;
                memcpy(bb->kv_k + dst, tk[b] + (size_t)L * pl * kvd, (size_t)pl * kvd * sizeof(uint16_t));
                memcpy(bb->kv_v + dst, tv[b] + (size_t)L * pl * kvd, (size_t)pl * kvd * sizeof(uint16_t));
            }
            free(tk[b]); free(tv[b]);
        }
        free(tk); free(tv);

        int *pos = (int *)malloc((size_t)B * sizeof(int));
        uint8_t *active = (uint8_t *)malloc((size_t)B);
        int *nprev = (int *)calloc(B, sizeof(int));
        int *chframes = (int *)calloc(B, sizeof(int));
        int **prev_tok = (int **)malloc((size_t)B * sizeof(int *));
        int **chcodes = (int **)malloc((size_t)B * sizeof(int *));
        float *last_hidden = (float *)malloc((size_t)B * h * sizeof(float));
        float *logits = (float *)malloc((size_t)vocab * sizeof(float));
        float *step_embed = (float *)malloc((size_t)B * h * sizeof(float));
        int *code0 = (int *)malloc((size_t)B * sizeof(int));
        int *cpcodes = (int *)malloc((size_t)B * 15 * sizeof(int));
        for (int b = 0; b < B; b++) {
            pos[b] = prompt_len[b]; active[b] = 1;
            prev_tok[b] = (int *)malloc((size_t)GEN_CAP * sizeof(int));
            chcodes[b] = (int *)malloc((size_t)GEN_CAP * 16 * sizeof(int));
        }
        memcpy(last_hidden, seed_hidden, (size_t)B * h * sizeof(float));
        const float *tts_pad = ctx->cached_tts_pad_embed;
        int n_active = B;

        for (int frame = 0; frame < GEN_CAP && n_active > 0; frame++) {
            qwen_census_frame_at(2);
            for (int b = 0; b < B; b++) {
                if (!active[b]) { code0[b] = 0; continue; }
                if (ctx->codec_head_q4)
                    qwen_matvec_q4_0(logits, ctx->codec_head_q4, last_hidden + (size_t)b * h, vocab, h);
                else
                    matvec_bf16(logits, ctx->codec_head_bf16, last_hidden + (size_t)b * h, vocab, h);
                for (int t = 0; t < vocab; t++) { if (logits[t] > 100.0f) logits[t] = 100.0f; if (logits[t] < -100.0f) logits[t] = -100.0f; }
                for (int t = 2048; t < vocab; t++) if (t != QWEN_TTS_CODEC_EOS) logits[t] = -1e30f;
                if (frame < 2) logits[QWEN_TTS_CODEC_EOS] = -1e30f;
                int ef = tcl[b] * 3, bs = ef * 2;
                if (ef > 0 && frame > bs) { float bo = 0.5f * (frame - bs); if (bo > 10.0f) bo = 10.0f; logits[QWEN_TTS_CODEC_EOS] += bo; }
                float ft = p_temp[b]; int ftk = p_topk[b];
                if (p_gw[b] > 0 && frame < p_gw[b]) { ft = 0.0f; ftk = 1; }
                qwen_set_seed(rng[b]);
                int c0 = qwen_tts_sample(logits, vocab, ft, ftk, p_topp[b], p_rep[b], prev_tok[b], nprev[b]);
                rng[b] = qwen_get_seed();
                if (c0 == QWEN_TTS_CODEC_EOS || chframes[b] >= GEN_CAP) { active[b] = 0; n_active--; code0[b] = 0; continue; }
                code0[b] = c0; prev_tok[b][nprev[b]++] = c0;
            }
            if (n_active == 0) break;

            qwen_batch_cp_predict(ctx, bb, last_hidden, code0, cpcodes, NULL);

            for (int b = 0; b < B; b++) {
                float *se = step_embed + (size_t)b * h;
                if (!active[b]) { memset(se, 0, (size_t)h * sizeof(float)); continue; }
                int frame16[16]; frame16[0] = code0[b];
                for (int g = 0; g < 15; g++) frame16[g + 1] = cpcodes[(size_t)b * 15 + g];
                memcpy(chcodes[b] + (size_t)chframes[b] * 16, frame16, 16 * sizeof(int));
                chframes[b]++;
                lookup_codec_embed(ctx, code0[b], se);
                for (int g = 0; g < 15; g++) {
                    int cg = frame16[g + 1];
                    if (ctx->cp_codec_emb_bf16[g] && cg >= 0 && cg < cb)
                        qwen_bf16_accum_f32(se, ctx->cp_codec_emb_bf16[g] + (size_t)cg * h, h);
                }
                for (int j = 0; j < h; j++) se[j] += tts_pad[j];
            }

            if (qwen_batch_talker_step_ragged(ctx, bb, step_embed, pos, active, last_hidden) != 0) break;

            for (int b = 0; b < B; b++) if (active[b]) pos[b]++;
        }

        for (int b = 0; b < B; b++) {
            if (chframes[b] <= 0) continue;
            float *aud = NULL; int an = 0;
            if (qwen_speech_decoder_decode(ctx, chcodes[b], chframes[b], &aud, &an) == 0 && aud && an > 0) {
                out_samples[g0 + b] = aud; out_n_samples[g0 + b] = an;
            } else {
                free(aud);
            }
        }

        for (int b = 0; b < B; b++) { free(prev_tok[b]); free(chcodes[b]); }
        free(pos); free(active); free(nprev); free(chframes); free(prev_tok); free(chcodes);
        free(last_hidden); free(logits); free(step_embed); free(code0); free(cpcodes);
        free(prompt_len); free(tcl); free(seed_hidden);
        free(p_temp); free(p_topk); free(p_topp); free(p_rep); free(p_gw); free(rng);
        qwen_batch_free(bb);
    }

    return 0;
}

typedef struct prefilled_s {
    void *tag;
    qwen_batch_req_t req;
    int ok;
    const char *reject_reason;
    int pl;
    int tcl;
    uint16_t *kv_k, *kv_v;
    float *last_hidden;
    double ts_admitted;
    double ts_prefill_start;
    double ts_prefill_done;
    double ts_state_ready;
    double ts_pfq_push;
    struct prefilled_s *next;
} prefilled_t;

typedef struct {
    prefilled_t *head, *tail;
    int count, cap, shutdown;
    pthread_mutex_t mtx;
    pthread_cond_t not_empty, not_full;
} prefill_q_t;

static void pfq_init(prefill_q_t *q, int cap) {
    q->head = q->tail = NULL; q->count = 0; q->cap = cap; q->shutdown = 0;
    pthread_mutex_init(&q->mtx, NULL);
    pthread_cond_init(&q->not_empty, NULL); pthread_cond_init(&q->not_full, NULL);
}
static void pfq_destroy(prefill_q_t *q) {
    pthread_mutex_destroy(&q->mtx);
    pthread_cond_destroy(&q->not_empty); pthread_cond_destroy(&q->not_full);
}
static int pfq_push(prefill_q_t *q, prefilled_t *p) {
    pthread_mutex_lock(&q->mtx);
    while (q->count >= q->cap && !q->shutdown) pthread_cond_wait(&q->not_full, &q->mtx);
    if (q->shutdown) { pthread_mutex_unlock(&q->mtx); return 0; }
    p->next = NULL;
    if (q->tail) q->tail->next = p; else q->head = p;
    q->tail = p; q->count++;
    pthread_cond_signal(&q->not_empty);
    pthread_mutex_unlock(&q->mtx);
    return 1;
}
static prefilled_t *pfq_pop(prefill_q_t *q, int block) {
    pthread_mutex_lock(&q->mtx);
    if (block) while (q->count == 0 && !q->shutdown) pthread_cond_wait(&q->not_empty, &q->mtx);
    if (q->count == 0) { pthread_mutex_unlock(&q->mtx); return NULL; }
    prefilled_t *p = q->head; q->head = p->next; if (!q->head) q->tail = NULL;
    q->count--;
    pthread_cond_signal(&q->not_full);
    pthread_mutex_unlock(&q->mtx);
    return p;
}
static void pfq_shutdown(prefill_q_t *q) {
    pthread_mutex_lock(&q->mtx);
    q->shutdown = 1;
    pthread_cond_broadcast(&q->not_empty); pthread_cond_broadcast(&q->not_full);
    pthread_mutex_unlock(&q->mtx);
}
static void prefilled_free(prefilled_t *p) {
    if (!p) return;
    free(p->kv_k); free(p->kv_v); free(p->last_hidden); free(p);
}

typedef struct {
    qwen_tts_ctx_t *pf_ctx;
    qwen_batch_sink_t *sink;
    prefill_q_t *q;
    int num_layers, kvd, h, MAXPROMPT;
    float eps;
} prefill_helper_arg_t;

static void *prefill_helper_main(void *arg) {
    prefill_helper_arg_t *a = (prefill_helper_arg_t *)arg;
    qwen_tts_ctx_t *pf = a->pf_ctx;
    for (;;) {
        qwen_batch_req_t req; void *tag = NULL;
        if (!a->sink->next_job(a->sink->ud, &req, &tag, 1)) break;
        double _t_admitted = qwen_mono_ms();
        pf->speaker_id = req.speaker_id; pf->language_id = req.language_id;
        pf->prev_prefill_len = 0; pf->prefill_only = 1;
        double _t_pf_start = qwen_mono_ms();
        int prc = qwen_tts_generate(pf, req.text, NULL, NULL);
        double _t_pf_done = qwen_mono_ms();
        pf->prefill_only = 0;
        int pl = pf->kv_len;
        prefilled_t *p = (prefilled_t *)calloc(1, sizeof(prefilled_t));
        if (!p) { a->sink->on_done(a->sink->ud, tag, NULL, 0); continue; }
        p->tag = tag; p->req = req;
        p->reject_reason = (prc != 0) ? "prefill failed"
                         : (pl <= 0)  ? "prefill produced nothing"
                         : (pl > a->MAXPROMPT) ? "prompt too long for a batch slot"
                         : NULL;
        p->ts_admitted = _t_admitted; p->ts_prefill_start = _t_pf_start;
        p->ts_prefill_done = _t_pf_done;
        if (prc == 0 && pl > 0 && pl <= a->MAXPROMPT) {
            size_t klen = (size_t)a->num_layers * pl * a->kvd;
            p->kv_k = (uint16_t *)malloc(klen * sizeof(uint16_t));
            p->kv_v = (uint16_t *)malloc(klen * sizeof(uint16_t));
            p->last_hidden = (float *)malloc((size_t)a->h * sizeof(float));
            if (p->kv_k && p->kv_v && p->last_hidden) {
                for (int L = 0; L < a->num_layers; L++) {
                    size_t d = (size_t)L * pl * a->kvd;
                    size_t s = (size_t)L * pf->kv_max * a->kvd;
                    memcpy(p->kv_k + d, pf->kv_cache_k + s, (size_t)pl * a->kvd * sizeof(uint16_t));
                    memcpy(p->kv_v + d, pf->kv_cache_v + s, (size_t)pl * a->kvd * sizeof(uint16_t));
                }
                qwen_rms_norm(p->last_hidden, pf->dec_x, pf->talker_norm, 1, a->h, a->eps);
                p->ok = 1; p->pl = pl; p->tcl = pf->bg_text_content_len;
            } else {
                free(p->kv_k); free(p->kv_v); free(p->last_hidden);
                p->kv_k = p->kv_v = NULL; p->last_hidden = NULL;
            }
        }
        p->ts_state_ready = qwen_mono_ms();
        p->ts_pfq_push = qwen_mono_ms();
        if (!pfq_push(a->q, p)) { prefilled_free(p); break; }
    }
    pfq_shutdown(a->q);
    return NULL;
}

typedef struct dec_job {
    int slot;
    int nframes;
    int *codes;
    void *tag;
    int is_final;
    int stream;
    int first;
    struct dec_job *next;
} dec_job_t;

typedef struct {
    pthread_mutex_t m;
    pthread_cond_t  cv;
    dec_job_t *head, *tail;
    int running;
    qwen_tts_ctx_t *ctx;
    qwen_batch_sink_t *sink;
    qwen_sd_stream_state_t *sstate;
    atomic_int *busy;
    int batch;
    int first_group;
    int trace;
} dec_pool_t;

static void dec_push(dec_pool_t *dp, dec_job_t *j) {
    pthread_mutex_lock(&dp->m);
    j->next = NULL;
    if (j->first && dp->head) {
        j->next = dp->head; dp->head = j;
    } else {
        if (dp->tail) dp->tail->next = j; else dp->head = j;
        dp->tail = j;
    }
    pthread_cond_signal(&dp->cv);
    pthread_mutex_unlock(&dp->m);
}

#define DEC_GROUP_MAX 16

static void *dec_worker_main(void *arg) {
    dec_pool_t *dp = (dec_pool_t *)arg;
    dec_job_t *grp[DEC_GROUP_MAX];
    qwen_sd_batch_item_t items[DEC_GROUP_MAX];
    for (;;) {
        pthread_mutex_lock(&dp->m);
        while (!dp->head && dp->running) pthread_cond_wait(&dp->cv, &dp->m);
        dec_job_t *j = dp->head;
        if (!j) { pthread_mutex_unlock(&dp->m); break; }
        dp->head = j->next; if (!dp->head) dp->tail = NULL;

        int ng = 0;
        grp[ng++] = j;
        if (dp->batch && j->stream && j->nframes > 0 &&
            (dp->first_group || !j->first)) {
            dec_job_t *prev = NULL, *it = dp->head;
            while (it && ng < DEC_GROUP_MAX) {
                int dup = 0;
                for (int i = 0; i < ng; i++) if (grp[i]->slot == it->slot) { dup = 1; break; }
                if (!dup && it->stream && it->nframes > 0) {
                    dec_job_t *take = it;
                    it = it->next;
                    if (prev) prev->next = take->next; else dp->head = take->next;
                    if (dp->tail == take) dp->tail = prev;
                    take->next = NULL;
                    grp[ng++] = take;
                    continue;
                }
                prev = it; it = it->next;
            }
        }
        pthread_mutex_unlock(&dp->m);

        int _dw_first = 0;
        for (int i = 0; i < ng; i++) if (grp[i]->first) { _dw_first = 1; break; }
        double _dw_t0 = dp->trace ? qwen_mono_ms() : 0.0;
        if (ng > 1) {
            for (int i = 0; i < ng; i++) {
                items[i].st = &dp->sstate[grp[i]->slot];
                items[i].codes = grp[i]->codes;
                items[i].nframes = grp[i]->nframes;
                items[i].audio = NULL; items[i].n_samples = 0; items[i].rc = 0;
            }
            qwen_speech_decoder_decode_streaming_batch(dp->ctx, items, ng);
            for (int i = 0; i < ng; i++) {
                if (items[i].rc == 0 && items[i].audio && items[i].n_samples > 0)
                    dp->sink->on_chunk(dp->sink->ud, grp[i]->tag,
                                       items[i].audio, items[i].n_samples);
                free(items[i].audio);
            }
        } else if (j->stream) {
            if (j->nframes > 0) {
                float *aud = NULL; int an = 0;
                if (qwen_speech_decoder_decode_streaming_st(dp->ctx, &dp->sstate[j->slot],
                        j->codes, j->nframes, &aud, &an) == 0 && aud && an > 0)
                    dp->sink->on_chunk(dp->sink->ud, j->tag, aud, an);
                free(aud);
            }
        } else if (j->is_final) {
            float *aud = NULL; int an = 0;
            if (j->nframes > 0 &&
                qwen_speech_decoder_decode(dp->ctx, j->codes, j->nframes, &aud, &an) == 0
                && aud && an > 0) dp->sink->on_done(dp->sink->ud, j->tag, aud, an);
            else { free(aud); dp->sink->on_done(dp->sink->ud, j->tag, NULL, 0); }
        }
        if (dp->trace) {
            fprintf(stderr, "[DECODE] v=1 pid=%d placement=THREADED clock=CLOCK_MONOTONIC "
                            "domain=S group=%d first=%d batch=%d first_group=%d dur_ms=%.3f\n",
                    (int)getpid(), ng, _dw_first, dp->batch, dp->first_group,
                    qwen_mono_ms() - _dw_t0);
        }

        for (int i = 0; i < ng; i++) {
            dec_job_t *g = grp[i];
            if (g->stream && g->is_final) {
                qwen_sd_stream_free(&dp->sstate[g->slot]);
                dp->sink->on_done(dp->sink->ud, g->tag, NULL, 0);
            }
            free(g->codes);
            if (g->is_final) atomic_store(&dp->busy[g->slot], 0);
            else atomic_fetch_sub(&dp->busy[g->slot], 1);
            free(g);
        }
    }
    return NULL;
}

static void dec_enqueue(dec_pool_t *dp, int slot, const int *codes, int nframes,
                        void *tag, int is_final, int stream, int first) {
    dec_job_t *j = (dec_job_t *)calloc(1, sizeof(dec_job_t));
    if (!j) return;
    j->slot = slot; j->nframes = nframes; j->tag = tag;
    j->is_final = is_final; j->stream = stream; j->first = first;
    if (nframes > 0) {
        j->codes = (int *)malloc((size_t)nframes * 16 * sizeof(int));
        if (!j->codes) { free(j); return; }
        memcpy(j->codes, codes, (size_t)nframes * 16 * sizeof(int));
    }
    atomic_fetch_add(&dp->busy[slot], 1);
    dec_push(dp, j);
}

int qwen_tts_serve_continuous(qwen_tts_ctx_t *ctx, int B, qwen_batch_sink_t *sink) {
    if (B < 1) B = 1;
    int want_cuda_batch = 0;
#ifdef QWEN_HAVE_CUDA
    { extern void *g_cuda_talker_state, *g_cuda_cp_state;
      want_cuda_batch = (getenv("QWEN_CUDA_BATCH") && g_cuda_talker_state && g_cuda_cp_state && B <= 8); }
#endif
    int want_metal_batch = 0;
#ifdef QWEN_HAVE_METAL
    { extern void *g_metal_talker_state;
      want_metal_batch = (getenv("QWEN_METAL_BATCH") && g_metal_talker_state && B <= 8); }
#endif
    if (ctx->layers[0].wq_bf16 == NULL && !want_cuda_batch && !want_metal_batch) return -2;
    int h = ctx->config.hidden_size;
    int kvd = ctx->config.num_kv_heads * ctx->config.head_dim;
    int num_layers = ctx->config.num_layers;
    int vocab = ctx->config.codec_vocab_size;
    int cb = ctx->config.codebook_size;
    float eps = ctx->config.rms_norm_eps;
    int GEN_CAP = ctx->max_tokens;
    { int lim = qwen_tts_batch_max_frames(); if (GEN_CAP > lim) GEN_CAP = lim; }
    if (GEN_CAP < 32) GEN_CAP = 32;
    const int MAXPROMPT = qwen_tts_batch_max_prompt();
    int kv_max = MAXPROMPT + GEN_CAP + 4;
    int force_matvec = getenv("QWEN_BATCH_FORCE_MATVEC") ? 1 : 0;

    qwen_batch_t *bb = qwen_batch_alloc(ctx, B, kv_max);
    if (!bb) return -1;
    bb->force_matvec = force_matvec;

    int cuda_batch = 0;
#ifdef QWEN_HAVE_CUDA
    extern void *g_cuda_talker_state, *g_cuda_cp_state, *g_cuda_talker_batch_state, *g_cuda_cp_batch_state;
    extern void *qwen_cuda_talker_batch_init(void *, int);
    extern void *qwen_cuda_cp_batch_init(void *, int);
    extern void  qwen_cuda_talker_batch_upload_slot(void *, int, const uint16_t *, const uint16_t *, int, int);
    extern void  qwen_cuda_talker_batch_free(void *);
    extern void  qwen_cuda_cp_batch_free(void *);
    if (want_cuda_batch) {
        g_cuda_talker_batch_state = qwen_cuda_talker_batch_init(g_cuda_talker_state, B);
        g_cuda_cp_batch_state = qwen_cuda_cp_batch_init(g_cuda_cp_state, B);
        cuda_batch = (g_cuda_talker_batch_state && g_cuda_cp_batch_state);
        if (cuda_batch) fprintf(stderr, "[serve] GPU batched Talker+CP ENABLED (B=%d, matvec->matmat)\n", B);
        else fprintf(stderr, "[serve] GPU batched init failed — falling back to CPU batch path\n");
    } else if (getenv("QWEN_CUDA_BATCH") && B > 8) {
        fprintf(stderr, "[serve] QWEN_CUDA_BATCH: batch-size %d > 8 (QB_MAX) — using CPU batch path\n", B);
    }
#endif

    int metal_batch = 0;
#ifdef QWEN_HAVE_METAL
    extern void *g_metal_talker_state, *g_metal_talker_batch_state, *g_metal_cp_batch_state;
    extern void *qwen_metal_talker_batch_init(void *, int);
    extern void *qwen_metal_cp_batch_init(void *, int);
    extern void  qwen_metal_talker_batch_upload_slot(void *, int, const uint16_t *, const uint16_t *, int, int);
    extern void  qwen_metal_talker_batch_free(void *);
    extern void  qwen_metal_cp_batch_free(void *);
    if (want_metal_batch) {
        g_metal_talker_batch_state = qwen_metal_talker_batch_init(g_metal_talker_state, B);
        if (!getenv("QWEN_METAL_BATCH_NOCP"))
            g_metal_cp_batch_state = qwen_metal_cp_batch_init(g_metal_talker_state, B);
        metal_batch = (g_metal_talker_batch_state != NULL);
        if (metal_batch) fprintf(stderr, "[serve] Metal batched Talker+CP ENABLED (B=%d, matvec->matmat)\n", B);
        else fprintf(stderr, "[serve] Metal batched init failed — falling back to CPU batch path\n");
    } else if (getenv("QWEN_METAL_BATCH") && B > 8) {
        fprintf(stderr, "[serve] QWEN_METAL_BATCH: batch-size %d > 8 — using CPU batch path\n", B);
    }
#endif

    uint8_t *active = (uint8_t *)calloc(B, 1);
    void **tag = (void **)calloc(B, sizeof(void *));
    int *pos = (int *)calloc(B, sizeof(int));
    int *tcl = (int *)calloc(B, sizeof(int));
    float *p_temp = (float *)malloc((size_t)B * sizeof(float));
    int *p_topk = (int *)malloc((size_t)B * sizeof(int));
    float *p_topp = (float *)malloc((size_t)B * sizeof(float));
    float *p_rep = (float *)malloc((size_t)B * sizeof(float));
    int *p_gw = (int *)malloc((size_t)B * sizeof(int));
    uint32_t *rng = (uint32_t *)malloc((size_t)B * sizeof(uint32_t));
    int *nprev = (int *)calloc(B, sizeof(int));
    int *chframes = (int *)calloc(B, sizeof(int));
    int *sframe = (int *)calloc(B, sizeof(int));
    int *decpos = (int *)calloc(B, sizeof(int));
    int **prev_tok = (int **)malloc((size_t)B * sizeof(int *));
    int **chcodes = (int **)malloc((size_t)B * sizeof(int *));
    float *last_hidden = (float *)calloc((size_t)B * h, sizeof(float));
    float *logits = (float *)malloc((size_t)B * vocab * sizeof(float));
    float *step_embed = (float *)malloc((size_t)B * h * sizeof(float));
    int *code0 = (int *)malloc((size_t)B * sizeof(int));
    int *cpcodes = (int *)malloc((size_t)B * 15 * sizeof(int));
    uint8_t *want_stream = (uint8_t *)calloc(B, 1);
    qwen_sd_stream_state_t *sstate = (qwen_sd_stream_state_t *)calloc(B, sizeof(qwen_sd_stream_state_t));
    int amort = (cuda_batch || getenv("QWEN_AMORT_CPU")) && !getenv("QWEN_NO_AMORT");
    float **acc_aud = (float **)calloc(B, sizeof(float *));
    int *acc_n = (int *)calloc(B, sizeof(int));
    int *acc_cap = (int *)calloc(B, sizeof(int));
    for (int b = 0; b < B; b++) {
        prev_tok[b] = (int *)malloc((size_t)GEN_CAP * sizeof(int));
        chcodes[b] = (int *)malloc((size_t)GEN_CAP * 16 * sizeof(int));
    }
    const float *tts_pad = ctx->cached_tts_pad_embed;
    int n_active = 0;

    dec_pool_t dpool; pthread_t dec_thr; int dec_on = 0;
    atomic_int *dec_busy = (atomic_int *)calloc(B, sizeof(atomic_int));
    if (getenv("QWEN_DECODER_THREAD") && dec_busy) {
        memset(&dpool, 0, sizeof dpool);
        pthread_mutex_init(&dpool.m, NULL); pthread_cond_init(&dpool.cv, NULL);
        dpool.running = 1; dpool.sink = sink; dpool.sstate = sstate; dpool.busy = dec_busy;
        dpool.batch = (getenv("QWEN_DECODER_BATCH") &&
                       atoi(getenv("QWEN_DECODER_BATCH")) != 0) ? 1 : 0;
        dpool.first_group = (getenv("QWEN_DEC_FIRSTCHUNK_GROUP") &&
                             atoi(getenv("QWEN_DEC_FIRSTCHUNK_GROUP")) != 0) ? 1 : 0;
        dpool.trace = (getenv("QWEN_TTFA_TRACE") &&
                       atoi(getenv("QWEN_TTFA_TRACE")) != 0) ? 1 : 0;
        dpool.ctx = qwen_tts_clone_for_worker(ctx);
        if (dpool.ctx && pthread_create(&dec_thr, NULL, dec_worker_main, &dpool) == 0) {
            dec_on = 1;
            fprintf(stderr, "[serve] decoder thread ENABLED (decode leaves the frame loop)\n");
        } else {
            if (dpool.ctx) qwen_tts_free_clone(dpool.ctx);
            fprintf(stderr, "[serve] decoder thread requested but clone/thread failed — staying inline\n");
        }
    }

    int ttfa_prio = 0;
    { const char *e = getenv("QWEN_TTFA_PRIORITY"); if (e) ttfa_prio = atoi(e);
      if (ttfa_prio < 0) ttfa_prio = 0; if (ttfa_prio > 8) ttfa_prio = 8; }
    uint8_t *prio_mask = ttfa_prio ? (uint8_t *)calloc(B, 1) : NULL;
    int *frozen = ttfa_prio ? (int *)calloc(B, sizeof(int)) : NULL;
    if (ttfa_prio && (!prio_mask || !frozen)) ttfa_prio = 0;
    int freeze_cap = ttfa_prio * 2;
    { const char *e = getenv("QWEN_TTFA_FREEZE_CAP"); if (e && atoi(e) >= 0) freeze_cap = atoi(e); }
    int prio_strict = getenv("QWEN_TTFA_PRIO_STRICT") ? 1 : 0;

    int prof_on = (getenv("QWEN_SERVE_PROFILE") || getenv("QWEN_BATCH_STATS")) ? 1 : 0;

    int dec_batch = (getenv("QWEN_DECODER_BATCH") &&
                     atoi(getenv("QWEN_DECODER_BATCH")) != 0) ? 1 : 0;
    int ttfa_trace = (getenv("QWEN_TTFA_TRACE") && atoi(getenv("QWEN_TTFA_TRACE")) != 0);
    int admit_m1 = (getenv("QWEN_ADMIT_M1") && atoi(getenv("QWEN_ADMIT_M1")) != 0);
    long long m1_admitted = 0, m1_rejected = 0, m1_cancelled = 0, m1_first_audio = 0;
    long long m1_scan = 0, m1_noslot = 0, m1_nojob = 0, m1_tick = 0;
    double *t2_admitted   = ttfa_trace ? (double *)calloc(B, sizeof(double)) : NULL;
    double *t2_pf_start   = ttfa_trace ? (double *)calloc(B, sizeof(double)) : NULL;
    double *t2_pf_done    = ttfa_trace ? (double *)calloc(B, sizeof(double)) : NULL;
    double *t2_state_rdy  = ttfa_trace ? (double *)calloc(B, sizeof(double)) : NULL;
    double *t2_pfq_push   = ttfa_trace ? (double *)calloc(B, sizeof(double)) : NULL;
    double *t2_pfq_pop    = ttfa_trace ? (double *)calloc(B, sizeof(double)) : NULL;
    double *t2_installed  = ttfa_trace ? (double *)calloc(B, sizeof(double)) : NULL;
    double *t2_frame1     = ttfa_trace ? (double *)calloc(B, sizeof(double)) : NULL;
    double *t2_audio1     = ttfa_trace ? (double *)calloc(B, sizeof(double)) : NULL;
    unsigned int *t2_seed = ttfa_trace ? (unsigned int *)calloc(B, sizeof(unsigned int)) : NULL;
    int *t2_helper        = ttfa_trace ? (int *)calloc(B, sizeof(int)) : NULL;
    int *t2_batch_at_inst = ttfa_trace ? (int *)calloc(B, sizeof(int)) : NULL;
    int *t2_qdepth_at_pop = ttfa_trace ? (int *)calloc(B, sizeof(int)) : NULL;
    int *t2_emitted       = ttfa_trace ? (int *)calloc(B, sizeof(int)) : NULL;
    unsigned long long *t2_adm_seq = ttfa_trace ?
        (unsigned long long *)calloc(B, sizeof(unsigned long long)) : NULL;
#define T2_FIRST_AUDIO(bb_) do {                                                             \
    if (ttfa_trace && t2_emitted && !t2_emitted[(bb_)]) {                                    \
        t2_audio1[(bb_)] = qwen_mono_ms(); t2_emitted[(bb_)] = 1;                            \
        fprintf(stderr,                                                                      \
            "[TTFA2] v=2 seed=%u path=%s slot=%d clock=CLOCK_MONOTONIC domain=S "            \
            "dec_thread=%d dec_batch=%d admitted=%.3f prefill_start=%.3f prefill_done=%.3f " \
            "state_ready=%.3f pfq_push=%.3f pfq_pop=%.3f installed=%.3f frame1=%.3f "        \
            "audio1=%.3f batch_at_install=%d pfq_depth_at_pop=%d adm_seq=%llu\n",  \
            t2_seed[(bb_)], t2_helper[(bb_)] ? "HELPER" : "INLINE", (bb_),                   \
            dec_on, dec_batch,                                                               \
            t2_admitted[(bb_)], t2_pf_start[(bb_)], t2_pf_done[(bb_)], t2_state_rdy[(bb_)],  \
            t2_pfq_push[(bb_)], t2_pfq_pop[(bb_)], t2_installed[(bb_)], t2_frame1[(bb_)],    \
            t2_audio1[(bb_)], t2_batch_at_inst[(bb_)], t2_qdepth_at_pop[(bb_)],  \
            t2_adm_seq[(bb_)]);  \
    }                                                                                        \
} while (0)
    int *db_pending = dec_batch ? (int *)calloc(B, sizeof(int)) : NULL;
    int *db_target  = dec_batch ? (int *)calloc(B, sizeof(int)) : NULL;
    int *db_slot    = dec_batch ? (int *)calloc(B, sizeof(int)) : NULL;
    qwen_sd_batch_item_t *db_items = dec_batch ?
        (qwen_sd_batch_item_t *)calloc(B, sizeof(qwen_sd_batch_item_t)) : NULL;
    if (dec_batch && (!db_pending || !db_target || !db_slot || !db_items)) dec_batch = 0;
    if (dec_batch && dec_on) dec_batch = 0;
    long long db_calls = 0, db_slots_sum = 0; int db_max = 0;

    int st_bt = 0, st_bd = 0, st_tt = 0, st_td = 0;
    { const char *e;
      if ((e = getenv("QWEN_BATCH_TALKER")))  { st_bt = atoi(e); if (st_bt < 0) st_bt = 0; }
      if ((e = getenv("QWEN_BATCH_DECODER"))) { st_bd = atoi(e); if (st_bd < 0) st_bd = 0; }
      if ((e = getenv("QWEN_THREADS_TALKER")))  { st_tt = atoi(e); if (st_tt < 0) st_tt = 0; }
      if ((e = getenv("QWEN_THREADS_DECODER"))) { st_td = atoi(e); if (st_td < 0) st_td = 0; } }
    if (st_bt > B) st_bt = B;
    if (st_bd > B) st_bd = B;
    uint8_t *width_mask = st_bt ? (uint8_t *)calloc(B, 1) : NULL;
    if (st_bt && !width_mask) st_bt = 0;
    uint8_t *m1_mask = (uint8_t *)calloc(B, 1);
    int rr_cursor = 0;
    int st_th_base = qwen_get_threads();
    if (st_tt || st_td) {
        int hard = st_th_base;
        if (st_tt > hard) hard = st_tt;
        if (st_td > hard) hard = st_td;
        if (hard != st_th_base) qwen_set_threads(hard);
        fprintf(stderr, "[serve] stage threads: talker %d · decoder %d (pool %d)\n",
                st_tt ? st_tt : st_th_base, st_td ? st_td : st_th_base, qwen_get_threads_hard());
    }
    if (st_bt || st_bd) {
        char wt[16], wd[16];
        if (st_bt) snprintf(wt, sizeof wt, "%d", st_bt); else snprintf(wt, sizeof wt, "all");
        if (st_bd) snprintf(wd, sizeof wd, "%d", st_bd); else snprintf(wd, sizeof wd, "all");
        fprintf(stderr, "[serve] stage batch width: talker %s · decoder %s (of %d slots)\n",
                wt, wd, B);
    }

    int g_stream_dec_chunk = 8;
    { const char *e = getenv("QWEN_STREAM_DECODE_CHUNK");
      if (e && atoi(e) > 0) g_stream_dec_chunk = atoi(e);
      if (g_stream_dec_chunk > 32) g_stream_dec_chunk = 32; }

    int g_dec_chunk_busy = 0;
    { const char *e = getenv("QWEN_STREAM_DECODE_CHUNK_BUSY");
      if (e && atoi(e) > 0) g_dec_chunk_busy = atoi(e);
      if (g_dec_chunk_busy > 32) g_dec_chunk_busy = 32; }

    int blas_solo = 0, blas_busy = 0, blas_now = 0;
    { const char *e = getenv("QWEN_SERVE_BLAS");      if (e && atoi(e) > 0) blas_solo = atoi(e); }
    { const char *e = getenv("QWEN_SERVE_BLAS_BUSY"); if (e && atoi(e) > 0) blas_busy = atoi(e); }
    if (blas_solo || blas_busy)
        fprintf(stderr, "[serve] BLAS budget: %d with one slot · %d from two up\n",
                blas_solo ? blas_solo : qwen_get_threads(),
                blas_busy ? blas_busy : qwen_get_threads());

    int g_gang_lead = 4, g_gang_min = 2;
    { const char *e = getenv("QWEN_DECODER_GANG_LEAD"); if (e && atoi(e) > 0) g_gang_lead = atoi(e); }
    { const char *e = getenv("QWEN_DECODER_GANG_MIN");  if (e && atoi(e) > 0) g_gang_min  = atoi(e); }
    if (g_dec_chunk_busy || g_gang_lead != 4 || g_gang_min != 2) {
        char busy[16];
        if (g_dec_chunk_busy) snprintf(busy, sizeof busy, "%d", g_dec_chunk_busy);
        else                  snprintf(busy, sizeof busy, "off");
        fprintf(stderr, "[serve] decode chunk %d (busy %s) · gang lead>=%d join>=%d\n",
                g_stream_dec_chunk, busy, g_gang_lead, g_gang_min);
    }
    double pf_admit = 0, pf_talker = 0, pf_head = 0, pf_cp = 0, pf_decode = 0, pf_wait = 0, pf_samp = 0, pf_final = 0;
    double pf_m1 = 0;
    const int rq_trace = getenv("QWEN_REQ_TRACE") ? 1 : 0;
    unsigned int *rq_seed = (unsigned int *)calloc((size_t)B, sizeof(unsigned int));
    int *rq_tok = (int *)calloc((size_t)B, sizeof(int));
    double *rq_t0 = (double *)calloc((size_t)B, sizeof(double));
    double pf_t0_loop = time_ms(), pf_mark = 0;
    long long pf_frames = 0, pf_slotframes = 0, pf_stepframes = 0;
    #define PF_START() do { if (prof_on) pf_mark = time_ms(); } while (0)
    #define PF_END(acc) do { if (prof_on) (acc) += time_ms() - pf_mark; } while (0)

    #define RELEASE_SLOT(b) do {                                                   \
        active[b] = 0; tag[b] = NULL; n_active--;                                  \
    } while (0)

    #define FINALIZE_SLOT(b) do {                                                  \
        double _pf_f0 = prof_on ? time_ms() : 0;                                   \
        if (rq_trace)                                                              \
            fprintf(stderr, "[REQ] pid=%d seed=%u tokens=%d frames=%d audio_s=%.3f " \
                            "service_ms=%.1f\n", (int)getpid(), rq_seed[b],        \
                    rq_tok[b], chframes[b], (double)chframes[b] / 12.5,            \
                    time_ms() - rq_t0[b]);                                         \
        if (dec_on && (want_stream[b] || !amort)) {                                \
               \
            if (want_stream[b])                                                    \
                dec_enqueue(&dpool, b, chcodes[b] + (size_t)decpos[b] * 16,         \
                            chframes[b] - decpos[b], tag[b], 1, 1, 0);             \
            else                                                                    \
                dec_enqueue(&dpool, b, chcodes[b], chframes[b], tag[b], 1, 0, 0);   \
            decpos[b] = chframes[b]; want_stream[b] = 0;                            \
            RELEASE_SLOT(b);                                                        \
            if (prof_on) { double _d = time_ms() - _pf_f0; pf_final += _d; pf_mark += _d; } \
            break;                                                                  \
        }                                                                           \
            \
        if ((want_stream[b] || amort) && chframes[b] > decpos[b]) {                 \
            float *_ta = NULL; int _tn = 0;                                        \
            if (qwen_speech_decoder_decode_streaming_st(ctx, &sstate[b],            \
                    chcodes[b] + (size_t)decpos[b] * 16,                            \
                    chframes[b] - decpos[b], &_ta, &_tn) == 0 && _ta && _tn > 0) {  \
                decpos[b] = chframes[b];                                            \
                if (want_stream[b]) sink->on_chunk(sink->ud, tag[b], _ta, _tn);     \
                else {                                                              \
                    if (acc_n[b] + _tn > acc_cap[b]) {                              \
                        acc_cap[b] = (acc_n[b] + _tn) * 2;                          \
                        acc_aud[b] = (float *)realloc(acc_aud[b],                   \
                                        (size_t)acc_cap[b] * sizeof(float));        \
                    }                                                               \
                    if (acc_aud[b]) { memcpy(acc_aud[b] + acc_n[b], _ta,            \
                        (size_t)_tn * sizeof(float)); acc_n[b] += _tn; }            \
                }                                                                   \
            }                                                                       \
            free(_ta);                                                              \
        }                                                                           \
        if (want_stream[b]) {                                                      \
            qwen_sd_stream_free(&sstate[b]); want_stream[b] = 0;                   \
            sink->on_done(sink->ud, tag[b], NULL, 0);                             \
        } else if (amort) {                                                        \
            qwen_sd_stream_free(&sstate[b]);                                       \
            if (acc_n[b] > 0) { sink->on_done(sink->ud, tag[b], acc_aud[b], acc_n[b]); \
                                acc_aud[b] = NULL; acc_cap[b] = 0; acc_n[b] = 0; } \
            else sink->on_done(sink->ud, tag[b], NULL, 0);                         \
        } else {                                                                   \
            float *aud = NULL; int an = 0;                                         \
            if (chframes[b] > 0 &&                                                 \
                qwen_speech_decoder_decode(ctx, chcodes[b], chframes[b], &aud, &an) == 0 \
                && aud && an > 0) sink->on_done(sink->ud, tag[b], aud, an);        \
            else { free(aud); sink->on_done(sink->ud, tag[b], NULL, 0); }         \
        }                                                                          \
        RELEASE_SLOT(b);                                                           \
        if (prof_on) { double _d = time_ms() - _pf_f0; pf_final += _d; pf_mark += _d; } \
    } while (0)

    #define CANCEL_SLOT(b) do {                                                    \
        if (rq_trace)                                                              \
            fprintf(stderr, "[REQ] pid=%d seed=%u tokens=%d frames=%d audio_s=%.3f " \
                            "service_ms=%.1f cancelled=1 frames_after_cancel=0\n", \
                    (int)getpid(), rq_seed[b], rq_tok[b], chframes[b],              \
                    (double)chframes[b] / 12.5, time_ms() - rq_t0[b]);              \
        qwen_sd_stream_free(&sstate[b]);                                           \
        want_stream[b] = 0;                                                        \
        if (acc_aud[b]) { free(acc_aud[b]); acc_aud[b] = NULL; }                   \
        acc_cap[b] = 0; acc_n[b] = 0;                                              \
        void *_t = tag[b];                                                         \
        RELEASE_SLOT(b);                                                           \
        sink->on_done(sink->ud, _t, NULL, 0);                                      \
    } while (0)

#ifdef QWEN_HAVE_CUDA
#define ADMIT_UPLOAD_CUDA(b_) do { if (cuda_batch) qwen_cuda_talker_batch_upload_slot(g_cuda_talker_batch_state, (b_), bb->kv_k, bb->kv_v, kv_max, pos[(b_)]); } while (0)
#else
#define ADMIT_UPLOAD_CUDA(b_) do { } while (0)
#endif
#ifdef QWEN_HAVE_METAL
#define ADMIT_UPLOAD_METAL(b_) do { if (metal_batch) qwen_metal_talker_batch_upload_slot(g_metal_talker_batch_state, (b_), bb->kv_k, bb->kv_v, kv_max, pos[(b_)]); } while (0)
#else
#define ADMIT_UPLOAD_METAL(b_) do { } while (0)
#endif

#define ADMIT_PREFILL(b_, req_, prc_, pl_) do {                                            \
        int _sv_spk = ctx->speaker_id, _sv_lang = ctx->language_id;                        \
        ctx->speaker_id = (req_).speaker_id; ctx->language_id = (req_).language_id;        \
        ctx->prev_prefill_len = 0; ctx->prefill_only = 1;                                  \
        double _tt0 = ttfa_trace ? qwen_mono_ms() : 0;                                     \
        (prc_) = qwen_tts_generate(ctx, (req_).text, NULL, NULL);                          \
        double _tt1 = ttfa_trace ? qwen_mono_ms() : 0;                                     \
        ctx->prefill_only = 0;                                                             \
        if (ttfa_trace) {                                                                  \
                        \
            t2_helper[(b_)] = 0;            t2_seed[(b_)]      = (req_).seed;              \
            t2_admitted[(b_)] = _tt0;       t2_pf_start[(b_)]  = _tt0;                     \
            t2_pf_done[(b_)]  = _tt1;       t2_state_rdy[(b_)] = _tt1;                     \
            t2_pfq_push[(b_)] = 0;          t2_pfq_pop[(b_)]   = 0;                        \
            t2_frame1[(b_)] = 0; t2_audio1[(b_)] = 0; t2_emitted[(b_)] = 0;                \
            t2_batch_at_inst[(b_)] = n_active; t2_qdepth_at_pop[(b_)] = -1;                \
            t2_adm_seq[(b_)] = atomic_load_explicit(&g_admit_seq, memory_order_relaxed);   \
        }                                                                                  \
        ctx->speaker_id = _sv_spk; ctx->language_id = _sv_lang;                            \
        (pl_) = ctx->kv_len;                                                               \
    } while (0)

#define ADMIT_INSTALL(b_, req_, tag_, pl_) do {                                            \
        for (int _L = 0; _L < num_layers; _L++) {                                          \
            size_t _dst = ((size_t)(b_) * num_layers + _L) * kv_max * kvd;                 \
            memcpy(bb->kv_k + _dst, ctx->kv_cache_k + (size_t)_L * ctx->kv_max * kvd,      \
                   (size_t)(pl_) * kvd * sizeof(uint16_t));                                \
            memcpy(bb->kv_v + _dst, ctx->kv_cache_v + (size_t)_L * ctx->kv_max * kvd,      \
                   (size_t)(pl_) * kvd * sizeof(uint16_t));                                \
        }                                                                                  \
        qwen_rms_norm(last_hidden + (size_t)(b_) * h, ctx->dec_x, ctx->talker_norm, 1, h, eps); \
        tcl[(b_)] = ctx->bg_text_content_len;                                              \
        pos[(b_)] = (pl_);                                                                 \
        ADMIT_UPLOAD_CUDA((b_)); ADMIT_UPLOAD_METAL((b_));                                 \
        p_temp[(b_)] = (req_).temperature; p_topk[(b_)] = (req_).top_k;                    \
        p_topp[(b_)] = (req_).top_p; p_rep[(b_)] = (req_).rep_penalty;                     \
        p_gw[(b_)] = (req_).greedy_warmup; rng[(b_)] = (req_).seed;                        \
        nprev[(b_)] = 0; chframes[(b_)] = 0; sframe[(b_)] = 0; decpos[(b_)] = 0;           \
        if (rq_trace) { rq_seed[(b_)] = (req_).seed; rq_tok[(b_)] = (pl_);                 \
                        rq_t0[(b_)] = time_ms(); }                                         \
        want_stream[(b_)] = ((req_).want_stream && sink->on_chunk) ? 1 : 0;                \
        if (want_stream[(b_)]) qwen_sd_stream_init(&sstate[(b_)]);                         \
        tag[(b_)] = (tag_); active[(b_)] = 1; n_active++;                                  \
        if (ttfa_trace) t2_installed[(b_)] = qwen_mono_ms();                               \
    } while (0)

#define SAMPLE_SLOT(b_, c0_, stop_) do {                                                   \
        (stop_) = 0; (c0_) = 0;                                                            \
                                                    \
        if (sink->cancelled && tag[(b_)] && sink->cancelled(sink->ud, tag[(b_)])) {         \
            CANCEL_SLOT((b_)); code0[(b_)] = 0; (stop_) = 1; break;                         \
        }                                                                                  \
        float *_lg = logits + (size_t)(b_) * vocab;                                         \
        for (int _t = 0; _t < vocab; _t++) {                                                \
            if (_lg[_t] >  100.0f) _lg[_t] =  100.0f;                                       \
            if (_lg[_t] < -100.0f) _lg[_t] = -100.0f; }                                     \
        for (int _t = 2048; _t < vocab; _t++) if (_t != QWEN_TTS_CODEC_EOS) _lg[_t] = -1e30f; \
        int _sf = sframe[(b_)];                                                             \
        if (_sf < 2) _lg[QWEN_TTS_CODEC_EOS] = -1e30f;                                      \
        int _ef = tcl[(b_)] * 3, _bs = _ef * 2;                                             \
        if (_ef > 0 && _sf > _bs) { float _bo = 0.5f * (_sf - _bs);                          \
                                    if (_bo > 10.0f) _bo = 10.0f;                            \
                                    _lg[QWEN_TTS_CODEC_EOS] += _bo; }                        \
        float _ft = p_temp[(b_)]; int _ftk = p_topk[(b_)];                                   \
        if (p_gw[(b_)] > 0 && _sf < p_gw[(b_)]) { _ft = 0.0f; _ftk = 1; }                    \
        qwen_set_seed(rng[(b_)]);                                                            \
        int _c0 = qwen_tts_sample(_lg, vocab, _ft, _ftk, p_topp[(b_)], p_rep[(b_)],          \
                                  prev_tok[(b_)], nprev[(b_)]);                              \
        rng[(b_)] = qwen_get_seed();                                                         \
        if (_c0 == QWEN_TTS_CODEC_EOS || chframes[(b_)] >= GEN_CAP ||                        \
            pos[(b_)] >= kv_max - 1) {                                                       \
            FINALIZE_SLOT((b_)); code0[(b_)] = 0; (stop_) = 1; break;                         \
        }                                                                                    \
        code0[(b_)] = _c0; prev_tok[(b_)][nprev[(b_)]++] = _c0; (c0_) = _c0;                 \
    } while (0)

#define RECORD_FRAME_AND_EMBED(b_) do {                                                     \
        float *_se = step_embed + (size_t)(b_) * h;                                          \
        int _f16[16]; _f16[0] = code0[(b_)];                                                 \
        for (int _g = 0; _g < 15; _g++) _f16[_g + 1] = cpcodes[(size_t)(b_) * 15 + _g];      \
        memcpy(chcodes[(b_)] + (size_t)chframes[(b_)] * 16, _f16, 16 * sizeof(int));         \
        chframes[(b_)]++;                                                                    \
        lookup_codec_embed(ctx, code0[(b_)], _se);                                           \
        for (int _g = 0; _g < 15; _g++) {                                                    \
            int _cg = _f16[_g + 1];                                                          \
            if (ctx->cp_codec_emb_bf16[_g] && _cg >= 0 && _cg < cb)                          \
                qwen_bf16_accum_f32(_se, ctx->cp_codec_emb_bf16[_g] + (size_t)_cg * h, h);   \
        }                                                                                    \
        for (int _j = 0; _j < h; _j++) _se[_j] += tts_pad[_j];                               \
    } while (0)

    int use_helper = qwen_parallel_is_reentrant();
    qwen_tts_ctx_t *pf_ctx = use_helper ? qwen_tts_clone_for_worker(ctx) : NULL;
    prefill_q_t pfq; pthread_t pf_thr; prefill_helper_arg_t pf_arg;
    if (pf_ctx) {
        int cap = (B < 2) ? B : 2; if (cap < 1) cap = 1;
        { const char *e = getenv("QWEN_QUEUE_PREFILL");
          if (e && atoi(e) > 0) { cap = atoi(e); if (cap > B) cap = B; } }
        pfq_init(&pfq, cap);
        pf_arg.pf_ctx = pf_ctx; pf_arg.sink = sink; pf_arg.q = &pfq;
        pf_arg.num_layers = num_layers; pf_arg.kvd = kvd; pf_arg.h = h;
        pf_arg.MAXPROMPT = MAXPROMPT; pf_arg.eps = eps;
        if (pthread_create(&pf_thr, NULL, prefill_helper_main, &pf_arg) != 0) {
            qwen_tts_free_clone(pf_ctx); pf_ctx = NULL; pfq_destroy(&pfq);
        }
    }
    use_helper = (pf_ctx != NULL);
    if (admit_m1 && use_helper) {
        fprintf(stderr, "[serve] QWEN_ADMIT_M1 ignored: the prefill HELPER path is active "
                        "(M1 is defined against INLINE admission)\n");
        admit_m1 = 0;
    }
    if (admit_m1)
        fprintf(stderr, "[serve] M1 early first-frame admission ON (one request per iteration)\n");

    double _t2_prev_iter = 0.0;
    while (sink->running(sink->ud) || n_active > 0) {
        if (ttfa_trace) {
            double _now = qwen_mono_ms();
            {
                long long _fr = 0; for (int _b = 0; _b < B; _b++) _fr += chframes[_b];
                fprintf(stderr,
                    "[ITER] v=2 pid=%d seq=%llu ts=%.3f clock=CLOCK_MONOTONIC domain=S prof=%d "
                    "n_active=%d frames_cum=%lld pf_admit=%.3f pf_talker=%.3f pf_head=%.3f "
                    "pf_samp=%.3f pf_cp=%.3f pf_decode=%.3f pf_final=%.3f pf_wait=%.3f\n",
                    (int)getpid(),
                    (unsigned long long)atomic_load_explicit(&g_admit_seq, memory_order_relaxed) + 1,
                    _now, prof_on, n_active, _fr,
                    pf_admit, pf_talker, pf_head, pf_samp, pf_cp, pf_decode, pf_final, pf_wait);
            }
            atomic_store_explicit(&g_admit_last_iter,
                _t2_prev_iter > 0 ? _now - _t2_prev_iter : 0.0, memory_order_relaxed);
            atomic_store_explicit(&g_admit_ts, _now, memory_order_relaxed);
            atomic_fetch_add_explicit(&g_admit_seq, 1, memory_order_relaxed);
            _t2_prev_iter = _now;
            if ((m1_tick++ % 1000) == 0)
                fprintf(stderr, "[M1STAT] v=1 pid=%d ts=%.3f clock=CLOCK_MONOTONIC domain=S "
                                "m1=%s admitted=%lld first_audio_same_iter=%lld rejected=%lld "
                                "stopped_first_frame=%lld scans=%lld no_free_slot=%lld "
                                "empty_queue=%lld m1_ms=%.1f\n",
                        (int)getpid(), _now, admit_m1 ? "ON" : "OFF", m1_admitted,
                        m1_first_audio, m1_rejected, m1_cancelled, m1_scan, m1_noslot,
                        m1_nojob, pf_m1);
        }
        PF_START();
        for (int b = 0; b < B; b++) {
            if (active[b]) continue;
            if (dec_on && atomic_load(&dec_busy[b]) != 0) continue;
            if (use_helper) {
                if (!sink->running(sink->ud) && n_active > 0) break;
                int block = (n_active == 0);
                double pf_w0 = prof_on ? time_ms() : 0;
                double _t_pop_pre = ttfa_trace ? qwen_mono_ms() : 0.0;
                prefilled_t *p = pfq_pop(&pfq, block);
                double _t_pop = ttfa_trace ? qwen_mono_ms() : 0.0;
                (void)_t_pop_pre;
                if (prof_on) { double d = time_ms() - pf_w0; pf_wait += d; pf_mark += d; }
                if (!p) break;
                if (!p->ok) {
                    if (sink->on_reject)
                        sink->on_reject(sink->ud, p->tag,
                                        p->reject_reason ? p->reject_reason : "could not admit the request");
                    else
                        sink->on_done(sink->ud, p->tag, NULL, 0);
                    prefilled_free(p); continue;
                }
                for (int L = 0; L < num_layers; L++) {
                    size_t dst = ((size_t)b * num_layers + L) * kv_max * kvd;
                    memcpy(bb->kv_k + dst, p->kv_k + (size_t)L * p->pl * kvd, (size_t)p->pl * kvd * sizeof(uint16_t));
                    memcpy(bb->kv_v + dst, p->kv_v + (size_t)L * p->pl * kvd, (size_t)p->pl * kvd * sizeof(uint16_t));
                }
                memcpy(last_hidden + (size_t)b * h, p->last_hidden, (size_t)h * sizeof(float));
                tcl[b] = p->tcl; pos[b] = p->pl;
#ifdef QWEN_HAVE_CUDA
                if (cuda_batch) qwen_cuda_talker_batch_upload_slot(g_cuda_talker_batch_state, b, bb->kv_k, bb->kv_v, kv_max, pos[b]);
#endif
#ifdef QWEN_HAVE_METAL
                if (metal_batch) qwen_metal_talker_batch_upload_slot(g_metal_talker_batch_state, b, bb->kv_k, bb->kv_v, kv_max, pos[b]);
#endif
                p_temp[b] = p->req.temperature; p_topk[b] = p->req.top_k; p_topp[b] = p->req.top_p;
                p_rep[b] = p->req.rep_penalty; p_gw[b] = p->req.greedy_warmup; rng[b] = p->req.seed;
                nprev[b] = 0; chframes[b] = 0; sframe[b] = 0; decpos[b] = 0;
                if (rq_trace) { rq_seed[b] = p->req.seed; rq_tok[b] = p->pl; rq_t0[b] = time_ms(); }
                if (ttfa_trace) {
                    t2_helper[b]   = 1;
                    t2_seed[b]     = p->req.seed;
                    t2_admitted[b] = p->ts_admitted;    t2_pf_start[b]  = p->ts_prefill_start;
                    t2_pf_done[b]  = p->ts_prefill_done; t2_state_rdy[b] = p->ts_state_ready;
                    t2_pfq_push[b] = p->ts_pfq_push;    t2_pfq_pop[b]   = _t_pop;
                    t2_frame1[b] = 0; t2_audio1[b] = 0; t2_emitted[b] = 0;
                    t2_batch_at_inst[b] = n_active;     t2_qdepth_at_pop[b] = pfq.count;
                    t2_adm_seq[b] = atomic_load_explicit(&g_admit_seq, memory_order_relaxed);
                }
                want_stream[b] = (p->req.want_stream && sink->on_chunk) ? 1 : 0;
                if (want_stream[b] || amort) qwen_sd_stream_init(&sstate[b]);
                acc_n[b] = 0;
                tag[b] = p->tag; active[b] = 1; n_active++;
                if (ttfa_trace) t2_installed[b] = qwen_mono_ms();
                prefilled_free(p);
                continue;
            }
            if (!sink->running(sink->ud)) break;
            int block = (n_active == 0);
            qwen_batch_req_t req;
            void *t = NULL;
            double pf_w1 = prof_on ? time_ms() : 0;
            int pf_got = sink->next_job(sink->ud, &req, &t, block);
            if (prof_on) { double d = time_ms() - pf_w1; pf_wait += d; pf_mark += d; }
            if (!pf_got) {
                if (block) break;
                continue;
            }
            int prc = 0, pl = 0;
            ADMIT_PREFILL(b, req, prc, pl);
            if (prc != 0 || pl <= 0 || pl > MAXPROMPT) {
                const char *why = (pl > MAXPROMPT) ? "prompt too long for a batch slot"
                                                   : "prefill failed";
                if (sink->on_reject) sink->on_reject(sink->ud, t, why);
                else sink->on_done(sink->ud, t, NULL, 0);
                continue;
            }
            ADMIT_INSTALL(b, req, t, pl);
        }

        if (n_active == 0) {
            if (!sink->running(sink->ud)) break;
            continue;
        }

        PF_END(pf_admit);

        if (blas_solo || blas_busy) {
            int want = (n_active > 1) ? blas_busy : blas_solo;
            if (want <= 0) want = qwen_get_threads();
            if (want != blas_now) { qwen_blas_set_threads(want); blas_now = want; }
        }

        uint8_t *step_active = active;
        if (ttfa_prio && n_active > 1) {
            int newest = -1, starving = 0, established = 0;
            for (int b = 0; b < B; b++) {
                if (!active[b]) continue;
                if (frozen[b] >= freeze_cap) starving = 1;
                if (sframe[b] >= ttfa_prio) established = 1;
                if (sframe[b] < ttfa_prio && (newest < 0 || sframe[b] < sframe[newest])) newest = b;
            }
            if (newest >= 0 && !starving && (established || !prio_strict)) {
                memset(prio_mask, 0, (size_t)B);
                prio_mask[newest] = 1;
                step_active = prio_mask;
            }
        }
        if (st_bt && step_active == active && n_active > st_bt) {
            memset(width_mask, 0, (size_t)B);
            int picked = 0;
            for (int k = 0; k < B && picked < st_bt; k++) {
                int b = (rr_cursor + k) % B;
                if (!active[b]) continue;
                width_mask[b] = 1; picked++;
                rr_cursor = (b + 1) % B;
            }
            step_active = width_mask;
        }
        if (ttfa_prio) {
            for (int b = 0; b < B; b++)
                frozen[b] = (active[b] && !step_active[b]) ? frozen[b] + 1 : 0;
        }
        if (st_tt) qwen_set_threads_soft(st_tt);

        qwen_batch_pack_active(bb, step_active);
        PF_START();
        qwen_batch_proj(logits, ctx->codec_head_bf16, last_hidden, vocab, h, h,
                        bb->B_eff > 0 ? bb->B_eff : B, bb->act_idx,
                        force_matvec, bb->Xt, bb->Yt);
        PF_END(pf_head);
        PF_START();
        for (int b = 0; b < B; b++) {
            if (!step_active[b]) { code0[b] = 0; continue; }
            int sm_c0 = 0, sm_stop = 0;
            SAMPLE_SLOT(b, sm_c0, sm_stop);
            if (sm_stop) continue;
        }
        PF_END(pf_samp);
        if (n_active == 0) continue;

        PF_START();
        qwen_batch_cp_predict(ctx, bb, last_hidden, code0, cpcodes, step_active);
        PF_END(pf_cp);
        qwen_census_frame_at(1);
        if (prof_on) {
            pf_frames++; pf_slotframes += n_active;
            for (int b = 0; b < B; b++) if (step_active[b]) pf_stepframes++;
        }

        PF_START();
        if (st_td) qwen_set_threads_soft(st_td);
        for (int b = 0; b < B; b++) {
            if (!step_active[b]) {
                memset(step_embed + (size_t)b * h, 0, (size_t)h * sizeof(float));
                continue;
            }
            RECORD_FRAME_AND_EMBED(b);

            if (want_stream[b] || amort) {
                int pending = chframes[b] - decpos[b];
                int target;
                if      (decpos[b] == 0) target = 1;
                else if (decpos[b] < 4)  target = 2;
                else if (decpos[b] < 12) target = 4;
                else if (g_dec_chunk_busy && n_active > 1) target = g_dec_chunk_busy;
                else                     target = g_stream_dec_chunk;
                if (ttfa_trace && t2_frame1[b] == 0.0 && pending > 0) t2_frame1[b] = qwen_mono_ms();
                if (dec_batch) {
                    db_pending[b] = pending;
                    db_target[b] = target;
                    continue;
                }
                if (pending < target) continue;
                if (dec_on && want_stream[b]) {
                    dec_enqueue(&dpool, b, chcodes[b] + (size_t)decpos[b] * 16, pending,
                                tag[b], 0, 1, decpos[b] == 0);
                    decpos[b] = chframes[b];
                    continue;
                }
                float *aud = NULL; int an = 0;
                if (qwen_speech_decoder_decode_streaming_st(ctx, &sstate[b],
                        chcodes[b] + (size_t)decpos[b] * 16, pending, &aud, &an) == 0
                    && aud && an > 0) {
                    decpos[b] = chframes[b];
                    T2_FIRST_AUDIO(b);
                    if (want_stream[b]) sink->on_chunk(sink->ud, tag[b], aud, an);
                    else {
                        if (acc_n[b] + an > acc_cap[b]) {
                            acc_cap[b] = (acc_n[b] + an) * 2;
                            acc_aud[b] = (float *)realloc(acc_aud[b], (size_t)acc_cap[b] * sizeof(float));
                        }
                        if (acc_aud[b]) { memcpy(acc_aud[b] + acc_n[b], aud, (size_t)an * sizeof(float)); acc_n[b] += an; }
                    }
                }
                free(aud);
            }
        }

        int m1_slot_i = -1; unsigned int m1_seed_i = 0; double m1_ts_i = 0;
        if (admit_m1 && sink->running(sink->ud)) {
            m1_scan++;
            int m1_free_slot = 0;
            for (int b = 0; b < B; b++) {
                if (active[b]) continue;
                if (dec_on && atomic_load(&dec_busy[b]) != 0) continue;
                m1_free_slot = 1;
                qwen_batch_req_t req;
                void *jt = NULL;
                double _m_t0 = ttfa_trace ? qwen_mono_ms() : 0;
                if (!sink->next_job(sink->ud, &req, &jt, 0)) { m1_nojob++; break; }
                double _mf0 = prof_on ? time_ms() : 0;
                int prc = 0, pl = 0;
                double _m_pf0 = ttfa_trace ? qwen_mono_ms() : 0;
                ADMIT_PREFILL(b, req, prc, pl);
                double _m_pf1 = ttfa_trace ? qwen_mono_ms() : 0;
                if (prc != 0 || pl <= 0 || pl > MAXPROMPT) {
                    const char *why = (pl > MAXPROMPT) ? "prompt too long for a batch slot"
                                                       : "prefill failed";
                    if (sink->on_reject) sink->on_reject(sink->ud, jt, why);
                    else sink->on_done(sink->ud, jt, NULL, 0);
                    m1_rejected++;
                    if (prof_on) { double _d = time_ms() - _mf0; pf_m1 += _d; pf_mark += _d; }
                    break;
                }
                int _m_nbefore = n_active;
                ADMIT_INSTALL(b, req, jt, pl);

                memset(m1_mask, 0, (size_t)B);
                m1_mask[b] = 1;
                qwen_batch_pack_active(bb, m1_mask);
                int _m_beff = bb->B_eff;
                qwen_batch_proj(logits, ctx->codec_head_bf16, last_hidden, vocab, h, h,
                                bb->B_eff > 0 ? bb->B_eff : B, bb->act_idx,
                                force_matvec, bb->Xt, bb->Yt);
                double _m_h1 = ttfa_trace ? qwen_mono_ms() : 0;
                int _m_c0 = 0, _m_stop = 0;
                SAMPLE_SLOT(b, _m_c0, _m_stop);
                double _m_s1 = ttfa_trace ? qwen_mono_ms() : 0;
                if (_m_stop) {
                    m1_cancelled++;
                    if (prof_on) { double _d = time_ms() - _mf0; pf_m1 += _d; pf_mark += _d; }
                    break;
                }
                qwen_batch_cp_predict(ctx, bb, last_hidden, code0, cpcodes, m1_mask);
                double _m_cp1 = ttfa_trace ? qwen_mono_ms() : 0;
                RECORD_FRAME_AND_EMBED(b);
                qwen_census_frame_at(1);
                if (prof_on) { pf_frames++; pf_slotframes++; pf_stepframes++; }
                double _m_e1 = ttfa_trace ? qwen_mono_ms() : 0;

                if (want_stream[b] || amort) {
                    if (ttfa_trace && t2_frame1[b] == 0.0) t2_frame1[b] = qwen_mono_ms();
                    if (dec_batch) { db_pending[b] = 1; db_target[b] = 1; }
                }
                if (step_active != active) step_active[b] = 1;

                m1_admitted++;
                m1_slot_i = b; m1_seed_i = req.seed; m1_ts_i = _m_e1;
                if (prof_on) { double _d = time_ms() - _mf0; pf_m1 += _d; pf_mark += _d; }
                if (ttfa_trace)
                    fprintf(stderr,
                        "[M1] v=1 pid=%d seed=%u slot=%d ts=%.3f clock=CLOCK_MONOTONIC "
                        "domain=S scan_ms=%.3f prefill_ms=%.3f head_ms=%.3f samp_ms=%.3f "
                        "cp_ms=%.3f embed_ms=%.3f narrow_ms=%.3f b_eff_first=%d "
                        "n_active_before=%d n_active_after=%d dec_batch=%d want_stream=%d "
                        "adm_seq=%llu\n",
                        (int)getpid(), req.seed, b, _m_e1,
                        _m_pf0 - _m_t0, _m_pf1 - _m_pf0, _m_h1 - _m_pf1, _m_s1 - _m_h1,
                        _m_cp1 - _m_s1, _m_e1 - _m_cp1, _m_e1 - _m_pf1, _m_beff,
                        _m_nbefore, n_active, dec_batch, want_stream[b],
                        (unsigned long long)atomic_load_explicit(&g_admit_seq,
                                                                memory_order_relaxed));
                break;
            }
            if (!m1_free_slot) m1_noslot++;
        }

        if (dec_batch) {
            int leader = 0;
            for (int b = 0; b < B; b++)
                if (db_pending[b] >= db_target[b] && db_target[b] >= g_gang_lead) { leader = 1; break; }
            int nit = 0;
            for (int pass = 0; pass < 2; pass++)
            for (int b = 0; b < B; b++) {
                int must = (db_pending[b] > 0) && (db_pending[b] >= db_target[b]);
                int join = (db_pending[b] > 0) && leader && (db_pending[b] >= g_gang_min) && !must;
                int fire = pass == 0 ? must : join;
                if (!fire) continue;
                if (st_bd && nit >= st_bd) break;
                db_slot[nit] = b;
                db_items[nit].st = &sstate[b];
                db_items[nit].codes = chcodes[b] + (size_t)decpos[b] * 16;
                db_items[nit].nframes = db_pending[b];
                db_items[nit].audio = NULL; db_items[nit].n_samples = 0; db_items[nit].rc = 0;
                nit++;
            }
            if (nit > 0) {
                int _di_first = 0;
                for (int _i = 0; _i < nit; _i++) if (decpos[db_slot[_i]] == 0) { _di_first = 1; break; }
                double _di_t0 = ttfa_trace ? qwen_mono_ms() : 0.0;
                qwen_speech_decoder_decode_streaming_batch(ctx, db_items, nit);
                if (ttfa_trace)
                    fprintf(stderr, "[DECODE] v=1 pid=%d placement=INLINE clock=CLOCK_MONOTONIC "
                                    "domain=S group=%d first=%d batch=%d first_group=%d dur_ms=%.3f\n",
                            (int)getpid(), nit, _di_first, dec_batch, 1,
                            qwen_mono_ms() - _di_t0);
                if (prof_on) { db_calls++; db_slots_sum += nit; if (nit > db_max) db_max = nit; }
                for (int i = 0; i < nit; i++) {
                    int b = db_slot[i];
                    float *aud = db_items[i].audio;
                    int an = db_items[i].n_samples;
                    if (db_items[i].rc == 0 && aud && an > 0) {
                        T2_FIRST_AUDIO(b);
                        decpos[b] += db_items[i].nframes;
                        if (want_stream[b]) sink->on_chunk(sink->ud, tag[b], aud, an);
                        else {
                            if (acc_n[b] + an > acc_cap[b]) {
                                acc_cap[b] = (acc_n[b] + an) * 2;
                                acc_aud[b] = (float *)realloc(acc_aud[b], (size_t)acc_cap[b] * sizeof(float));
                            }
                            if (acc_aud[b]) { memcpy(acc_aud[b] + acc_n[b], aud, (size_t)an * sizeof(float)); acc_n[b] += an; }
                        }
                    }
                    free(aud);
                }
            }
            for (int b = 0; b < B; b++) { db_pending[b] = 0; db_target[b] = 0; }
        }

        if (m1_slot_i >= 0) {
            int _got = (ttfa_trace && t2_emitted && t2_emitted[m1_slot_i]) ? 1 : 0;
            if (_got) m1_first_audio++;
            if (ttfa_trace)
                fprintf(stderr, "[M1AUDIO] v=1 pid=%d seed=%u slot=%d "
                                "clock=CLOCK_MONOTONIC domain=S same_iteration=%d "
                                "frame_ts=%.3f audio_ts=%.3f delay_ms=%.3f\n",
                        (int)getpid(), m1_seed_i, m1_slot_i, _got, m1_ts_i,
                        _got ? t2_audio1[m1_slot_i] : 0.0,
                        _got ? t2_audio1[m1_slot_i] - m1_ts_i : -1.0);
        }

        if (st_td) qwen_set_threads_soft(st_tt ? st_tt : st_th_base);

        PF_END(pf_decode);

        PF_START();
        int pf_rc = qwen_batch_talker_step_ragged(ctx, bb, step_embed, pos, step_active, last_hidden);
        PF_END(pf_talker);
        if (pf_rc != 0) {
            for (int b = 0; b < B; b++) if (active[b]) {
                if (want_stream[b]) { qwen_sd_stream_free(&sstate[b]); want_stream[b] = 0; }
                sink->on_done(sink->ud, tag[b], NULL, 0); active[b] = 0; tag[b] = NULL; n_active--;
            }
            break;
        }
        for (int b = 0; b < B; b++) if (step_active[b]) { pos[b]++; sframe[b]++; }
    }

    if (prof_on && pf_frames > 0) {
        double wall = time_ms() - pf_t0_loop;
        double acc = pf_admit + pf_talker + pf_head + pf_samp + pf_cp + pf_decode + pf_final;
        fprintf(stderr,
            "\n[serve-profile] %lld frames, %lld slot-frames (mean %.2f active slots), loop %.1f s\n",
            pf_frames, pf_slotframes, (double)pf_slotframes / (double)pf_frames, wall / 1000.0);
        if (pf_stepframes != pf_slotframes)
            fprintf(stderr, "  stepped slots/frame %.2f (of %.2f admitted): a narrowing policy is ON "
                            "(D2 priority and/or QWEN_BATCH_TALKER)\n",
                    (double)pf_stepframes / (double)pf_frames,
                    (double)pf_slotframes / (double)pf_frames);
        if (st_bt || st_bd || st_tt || st_td || getenv("QWEN_QUEUE_PREFILL")) {
            char wt[16], wd[16];
            if (st_bt) snprintf(wt, sizeof wt, "%d", st_bt); else snprintf(wt, sizeof wt, "all");
            if (st_bd) snprintf(wd, sizeof wd, "%d", st_bd); else snprintf(wd, sizeof wd, "all");
            fprintf(stderr, "  stage policy: batch talker=%s decoder=%s · threads talker=%d decoder=%d"
                            " · prefill queue=%s\n",
                    wt, wd, st_tt ? st_tt : st_th_base, st_td ? st_td : st_th_base,
                    getenv("QWEN_QUEUE_PREFILL") ? getenv("QWEN_QUEUE_PREFILL") : "default");
        }
        fprintf(stderr, "  %-28s %10s %8s   %s\n", "stage", "ms", "share", "what it tells you");
        struct { const char *n; double v; const char *w; } rows[7] = {
            { "talker step (batched)",  pf_talker, "kernel/ISA: where a real batched GEMM shows up — or fails to" },
            { "code predictor (batched)", pf_cp,   "15 sequential passes per frame; re-reads its weights 16x" },
            { "codec head (GEMM)",      pf_head,   "vocab x hidden, bf16 and NOT quantized — 12.6 MB on the 1.7B" },
            { "sampling (per slot)",    pf_samp,   "top-k + rep-penalty over the codec vocab, once per slot per frame" },
            { "finalize (full decode)", pf_final,  "a non-streaming request decodes its WHOLE utterance here, in one burst" },
            { "speech decode + embed",  pf_decode, "cross-slot batched since S13: read the slots-per-call line, not this share" },
            { "admission + prefill",    pf_admit,  "what a new arrival costs the requests already in flight" },
        };
        double pf_other[1] = { pf_wait };  (void)pf_other;
        for (int i = 0; i < 7; i++)
            fprintf(stderr, "  %-28s %10.0f %7.1f%%   %s\n", rows[i].n, rows[i].v,
                    acc > 0 ? 100.0 * rows[i].v / acc : 0.0, rows[i].w);
        fprintf(stderr, "  %-28s %10.0f %7.1f%%   %s\n", "(blocked: no work queued)", pf_wait,
                wall > 0 ? 100.0 * pf_wait / wall : 0.0,
                "share of WALL, not of work: high = the box is idle, not saturated");
        if (db_calls > 0)
            fprintf(stderr, "  %-28s %10lld %7.2f    %s\n", "decoder batch: calls / mean",
                    db_calls, (double)db_slots_sum / (double)db_calls,
                    "mean slots per decoder call (max seen: see below). 1.00 = not batching");
        if (db_calls > 0)
            fprintf(stderr, "  %-28s %10d %7s    %s\n", "decoder batch: max slots",
                    db_max, "",
                    "if this stays at 1 the gang policy never formed a batch — read WHY, not the RTF");
        fprintf(stderr, "  %-28s %10.0f %7.1f%%   %s\n", "(unaccounted)", wall - acc - pf_wait,
                wall > 0 ? 100.0 * (wall - acc - pf_wait) / wall : 0.0,
                "share of WALL; large here means the buckets miss something");
        fflush(stderr);
    }
    fprintf(stderr, "[serve] M1 early first frame: %s · admitted %lld · "
                    "first-audio-same-iter %lld · rejected %lld · stopped-on-first-frame %lld"
                    " · scans %lld (no free slot %lld · empty queue %lld) · %.0f ms\n",
            admit_m1 ? "ON" : "OFF", m1_admitted, m1_first_audio, m1_rejected,
            m1_cancelled, m1_scan, m1_noslot, m1_nojob, pf_m1);
    fflush(stderr);

    if (dec_on) {
        pthread_mutex_lock(&dpool.m);
        dpool.running = 0;
        pthread_cond_broadcast(&dpool.cv);
        pthread_mutex_unlock(&dpool.m);
        pthread_join(dec_thr, NULL);
        pthread_mutex_destroy(&dpool.m); pthread_cond_destroy(&dpool.cv);
        if (dpool.ctx) qwen_tts_free_clone(dpool.ctx);
        dec_on = 0;
    }
    free(dec_busy);
    free(db_pending); free(db_target); free(db_slot); free(db_items);
    free(t2_admitted); free(t2_pf_start); free(t2_pf_done); free(t2_state_rdy);
    free(t2_pfq_push); free(t2_pfq_pop); free(t2_installed); free(t2_frame1);
    free(t2_audio1); free(t2_seed); free(t2_helper); free(t2_batch_at_inst);
    free(t2_qdepth_at_pop); free(t2_emitted); free(t2_adm_seq);
    free(width_mask);
    free(m1_mask);
    free(prio_mask); free(frozen);

    #undef PF_START
    #undef PF_END
    #undef FINALIZE_SLOT
    #undef CANCEL_SLOT
    #undef RELEASE_SLOT

    if (use_helper) {
        pfq_shutdown(&pfq);
        pthread_join(pf_thr, NULL);
        prefilled_t *p;
        while ((p = pfq_pop(&pfq, 0)) != NULL) {
            sink->on_done(sink->ud, p->tag, NULL, 0);
            prefilled_free(p);
        }
        pfq_destroy(&pfq);
        qwen_tts_free_clone(pf_ctx);
    }

    for (int b = 0; b < B; b++) { if (active[b] && (want_stream[b] || amort)) qwen_sd_stream_free(&sstate[b]); free(prev_tok[b]); free(chcodes[b]); free(acc_aud[b]); }
    free(want_stream); free(sstate); free(acc_aud); free(acc_n); free(acc_cap);
    free(active); free(tag); free(pos); free(tcl);
    free(p_temp); free(p_topk); free(p_topp); free(p_rep); free(p_gw); free(rng);
    free(nprev); free(chframes); free(sframe); free(decpos); free(prev_tok); free(chcodes);
    free(last_hidden); free(logits); free(step_embed); free(code0); free(cpcodes);
    qwen_batch_free(bb);
#ifdef QWEN_HAVE_CUDA
    if (cuda_batch) {
        qwen_cuda_talker_batch_free(g_cuda_talker_batch_state); g_cuda_talker_batch_state = NULL;
        qwen_cuda_cp_batch_free(g_cuda_cp_batch_state); g_cuda_cp_batch_state = NULL;
    }
#endif
#ifdef QWEN_HAVE_METAL
    if (metal_batch) {
        qwen_metal_talker_batch_free(g_metal_talker_batch_state); g_metal_talker_batch_state = NULL;
        qwen_metal_cp_batch_free(g_metal_cp_batch_state); g_metal_cp_batch_state = NULL;
    }
#endif
    return 0;
}
