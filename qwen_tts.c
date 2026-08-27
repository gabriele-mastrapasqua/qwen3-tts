/*
 * qwen_tts.c - Qwen3-TTS Pure C Inference Engine
 * Main pipeline: text → Talker → Code Predictor → Speech Decoder → audio
 */

#include "qwen_tts.h"
#include "qwen_tts_voice_clone.h"
#include "qwen_tts_kernels.h"
#include "ingot/safetensors.h"
#include "qwen_tts_tokenizer.h"
#include "qwen_tts_audio.h"
#include "qwen_tts_batch.h"
#include "qwen_tts_thread.h"   /* qwen_parallel_is_reentrant() (A1 prefill helper gate) */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <unistd.h>
#include <stdatomic.h>   /* leaks-audit #6: _Atomic flag for the cross-thread cb_aborted */

int qwen_verbose = 0;

/* ── EOS strategy names ──────────────────────────────────────────────────────
 * Next to the parser so a new strategy cannot be added without a name.
 * Why this is a switch at all: the design notes. */
const char *qwen_tts_eos_strategy_name(int strategy) {
    switch (strategy) {
        case QWEN_EOS_OFF:  return "off";   /* = nano-vllm, i.e. their production */
        case QWEN_EOS_V1:   return "v1";    /* historic proportional ramp          */
        case QWEN_EOS_V2:   return "v2";    /* affine ramp (fixed overhead added)  */
        case QWEN_EOS_TOPK: return "topk";  /* = their PyTorch EosBoost processor  */
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

/* CLOCK_MONOTONIC twin, for the TTFA decomposition ONLY (schema TTFA2).
 * time_ms() above is gettimeofday(), i.e. CLOCK_REALTIME, and it is used elsewhere in this
 * file: it is deliberately NOT changed. But CLOCK_REALTIME cannot be subtracted from the
 * server's srv_now_ms(), which is CLOCK_MONOTONIC, so every instant that ever enters a
 * decomposition comes from HERE and declares it. Same process tree as the server worker =
 * one clock domain (domain S). */
/* STEP 3A probe state. Relaxed atomics: readers want a recent snapshot, not a barrier. */
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

int qwen_tts_resolve_speaker(const qwen_tts_ctx_t *ctx, const char *name) {
    if (!name) return -1;
    /* The model's own table wins: a finetune may reuse a preset name for a
     * different slot, and its declaration is authoritative for its own weights. */
    if (ctx) {
        for (int i = 0; i < ctx->spk_count; i++)
            if (strcasecmp(name, ctx->spk_names[i]) == 0) return ctx->spk_slots[i];
    }
    return qwen_tts_speaker_id(name);
}

/* dichiarata piu' sotto: il parser sta vicino al caricamento del modello */
static void parse_spk_id_table(qwen_tts_ctx_t *ctx, const char *cfg_raw);

/* Ricarica la tabella nomi->slot da un ALTRO modello (dir o config.json).
 *
 * Serve agli INNESTI. La mappa nomi->slot vive in config.json, non nei pesi:
 * un modello innestato (Base + delta) porta i pesi del finetune ma il config del
 * genitore, quindi a pool-voice name non risolve e l'unica via era passare lo slot
 * numerico a mano. Con --speaker-map si punta al finetune di provenienza e i nomi
 * tornano a funzionare ovunque.
 *
 * Gli slot restano quelli del file passato: e' proprio il punto — sono i numeri con
 * cui quei pesi sono stati addestrati. */
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

/* Parse `"spk_id": {"name": 3000, ...}` out of a raw config.json.
 *
 * Deliberately a small scanner rather than a JSON library: the engine has no JSON
 * dependency, and the shape here is fixed by the model format. Tolerates whitespace
 * and an empty object (VoiceDesign). */
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
        while (*p && *p != '"') p++;          /* speaker names carry no escapes */
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
/* Walk forward from `p` until the brace `depth` returns to 0, skipping any braces
 * that occur inside double-quoted strings (config JSON is trusted, but a brace in a
 * string value would otherwise miscount). Returns a pointer just past the matching
 * '}' (or at the terminating NUL). Call with `depth` = the count already opened. */
static const char *json_match_brace(const char *p, int depth) {
    int in_str = 0;
    while (*p && depth > 0) {
        char ch = *p;
        if (in_str) {
            if (ch == '\\' && p[1]) p++;        /* skip escaped char inside string */
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
    
    /* Find the closing brace of talker_config (including nested code_predictor_config) */
    const char *tc_end = json_match_brace(p + 1, 1);
    
    long tc_len = tc_end - p; char *tc_json = (char *)malloc(tc_len + 1);
    if (!tc_json) { free(json); return -1; }            /* leaks-audit #9: OOM NULL-check */
    memcpy(tc_json, p, tc_len); tc_json[tc_len] = '\0';
    
    /* Build a flat version of talker_config with nested objects removed.
     * This prevents json_find_key from matching keys inside nested objects
     * like code_predictor_config (whose fields shadow talker-level fields). */
    char *talker_only_json = strdup(tc_json);
    if (!talker_only_json) { free(tc_json); free(json); return -1; }   /* leaks-audit #9 */
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
            /* Find matching close brace (string-aware) */
            char *r = (char *)json_match_brace(nested_open + 1, 1);
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
                if (!cp_json) { free(tc_json); free(json); return -1; }  /* leaks-audit #9 (talker_only_json already freed at line 152) */
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
                if (!dc_json) { free(json); return -1; }            /* leaks-audit #9: OOM NULL-check */
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

/* Embed a single text token: text_embedding → text_projection(SiLU) → out[hidden]
 * Computes the full projection (bf16 lookup + fc1 SiLU + fc2). */
void embed_one_text_token_compute(qwen_tts_ctx_t *ctx, int tid, float *out) {
    qwen_mm_component(QWEN_COMP_TALKER);   /* text_projection is Talker work, done at prompt-build time */
    int th = ctx->config.text_hidden_size, h = ctx->config.hidden_size;
    float *text_emb = ctx->emb_tmp1;
    float *fc1_out = ctx->emb_tmp2;
    const uint16_t *emb = ctx->tok_embeddings_bf16 + (int64_t)tid * th;
    for (int j = 0; j < th; j++) text_emb[j] = bf16_to_f32(emb[j]);
    if (ctx->text_proj_fc1_bf16 && ctx->text_proj_fc2_bf16) {
        if (ctx->text_proj_fc1_q4) qwen_matvec_q4_0(fc1_out, ctx->text_proj_fc1_q4, text_emb, th, th);
        else                       matvec_bf16(fc1_out, ctx->text_proj_fc1_bf16, text_emb, th, th);
        if (ctx->text_proj_fc1_bias) for (int j = 0; j < th; j++) fc1_out[j] += ctx->text_proj_fc1_bias[j];
        for (int j = 0; j < th; j++) fc1_out[j] = fc1_out[j] / (1.0f + expf(-fc1_out[j])); /* SiLU */
        if (ctx->text_proj_fc2_q4) qwen_matvec_q4_0(out, ctx->text_proj_fc2_q4, fc1_out, h, th);
        else                       matvec_bf16(out, ctx->text_proj_fc2_bf16, fc1_out, h, th);
        if (ctx->text_proj_fc2_bias) for (int j = 0; j < h; j++) out[j] += ctx->text_proj_fc2_bias[j];
    } else {
        memcpy(out, text_emb, h * sizeof(float));
    }
}

/* ── LRU embedding cache ─────────────────────────────────────────────── */

#define EMB_CACHE_CAPACITY 2048  /* power of 2, holds up to ~1500 tokens before eviction */

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

/* Lookup or compute+insert. Returns pointer to cached embedding (valid until next eviction). */
static const float *emb_cache_get(qwen_tts_ctx_t *ctx, int tid) {
    int cap = ctx->emb_cache.capacity;
    int h = ctx->config.hidden_size;
    int mask = cap - 1;  /* cap is power of 2 */
    int idx = (tid * 2654435761u) & mask;  /* Knuth multiplicative hash */

    /* Linear probe: find existing or empty slot */
    for (int probe = 0; probe < cap; probe++) {
        int slot = (idx + probe) & mask;
        if (ctx->emb_cache.keys[slot] == tid) {
            /* Cache hit */
            ctx->emb_cache.access[slot] = ++ctx->emb_cache.clock;
            return ctx->emb_cache.values + (size_t)slot * h;
        }
        if (ctx->emb_cache.keys[slot] == -1) {
            /* Empty slot — compute and insert */
            ctx->emb_cache.keys[slot] = tid;
            ctx->emb_cache.access[slot] = ++ctx->emb_cache.clock;
            ctx->emb_cache.count++;
            float *dst = ctx->emb_cache.values + (size_t)slot * h;
            embed_one_text_token_compute(ctx, tid, dst);
            return dst;
        }
    }

    /* Table full (load factor ~75% with cap=2048) — evict LRU entry */
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

/* Embed a text token with caching. Checks special tokens first, then LRU cache. */
static void embed_one_text_token(qwen_tts_ctx_t *ctx, int tid, float *out) {
    int h = ctx->config.hidden_size;
    /* Fast path: pre-computed special tokens */
    if (ctx->cached_tts_pad_embed) {
        if (tid == QWEN_TTS_TTS_PAD) { memcpy(out, ctx->cached_tts_pad_embed, h * sizeof(float)); return; }
        if (tid == QWEN_TTS_TTS_BOS) { memcpy(out, ctx->cached_tts_bos_embed, h * sizeof(float)); return; }
        if (tid == QWEN_TTS_TTS_EOS) { memcpy(out, ctx->cached_tts_eos_embed, h * sizeof(float)); return; }
    }
    /* LRU cache path (server mode) */
    if (ctx->emb_cache.capacity > 0) {
        const float *cached = emb_cache_get(ctx, tid);
        memcpy(out, cached, h * sizeof(float));
        return;
    }
    /* Fallback: compute directly */
    embed_one_text_token_compute(ctx, tid, out);
}

/* ── Decoder Thread (pipeline overlap) ────────────────────────────────
 * Runs speech decoder in background while Talker+CP generates more frames.
 * Uses the existing streaming decoder path (qwen_speech_decoder_decode_streaming).
 *
 * Protocol:
 *   Main thread pushes frames via dt_push_frames() → signals condvar
 *   Decoder thread wakes, decodes chunk, appends audio to growing buffer
 *   Main thread calls dt_finish() → sets done flag, joins thread
 *   Audio is collected from dt->audio_buf after join
 */

#define DT_CHUNK_FRAMES 10  /* decode every N frames (match streaming chunk) */

typedef struct {
    /* Shared state (protected by mutex) */
    pthread_mutex_t mutex;
    pthread_cond_t  cond;
    int *codes;             /* ringbuffer: [capacity * 16] */
    int  capacity;          /* max frames in buffer */
    int  write_pos;         /* frames written by producer */
    int  read_pos;          /* frames consumed by decoder */
    int  done;              /* producer signals no more frames */

    /* Decoder output (owned by decoder thread) */
    float *audio_buf;       /* growing audio buffer (NULL if streaming via cb) */
    int    audio_len;       /* samples written */
    int    audio_cap;       /* capacity */

    /* Streaming callback (if set, audio goes to callback instead of buffer) */
    qwen_tts_audio_cb audio_cb;
    void *audio_cb_userdata;
    _Atomic int cb_aborted; /* set to 1 if callback returns non-zero (leaks-audit #6: read by the
                             * main thread + written by the decoder thread → atomic, not a plain int) */

    /* Context for decoder */
    qwen_tts_ctx_t *ctx;
    double decode_ms;       /* total decode time */
    double first_chunk_ms;  /* abs timestamp (gettimeofday ms) of first emitted chunk; 0 = none yet */
    int    chunk_frames;    /* frames to wait for per chunk (from ctx->stream_chunk_frames) */
    int    trim_head_left;  /* ICL onset fix: samples still to drop from the head of output */
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
    /* Frames per decode chunk: wire --stream-chunk (ctx->stream_chunk_frames),
     * fall back to the historical default if unset. Smaller = lower TTFA. */
    dt->chunk_frames = (ctx->stream_chunk_frames > 0) ? ctx->stream_chunk_frames : DT_CHUNK_FRAMES;
    dt->audio_cb = NULL;
    dt->audio_cb_userdata = NULL;
    dt->cb_aborted = 0;
    dt->trim_head_left = 0;
    /* Pre-allocate audio for ~max_frames worth of audio */
    dt->audio_cap = max_frames * 1920 + 4096;  /* 1920 samples/frame + margin */
    dt->audio_buf = (float *)aligned_malloc(dt->audio_cap * sizeof(float));
    dt->audio_len = 0;
}

static void dt_free(decoder_thread_t *dt) {
    pthread_mutex_destroy(&dt->mutex);
    pthread_cond_destroy(&dt->cond);
    free(dt->codes);
    /* Leaks-audit fix (2026-07, #1 HIGH): free audio_buf here. The normal-mode
     * caller that takes ownership NULLs dt->audio_buf *before* calling us, so this
     * is free(NULL) (no-op) on that path; on the streaming, talker-step-error, and
     * codec_frames==0 paths the caller does NOT take ownership, so this frees the
     * ~63 MB (max_frames*1920) allocation that used to leak on every such request. */
    free(dt->audio_buf);
    dt->audio_buf = NULL;
}

static void dt_push_frames(decoder_thread_t *dt, const int *frame_codes, int n_frames) {
    pthread_mutex_lock(&dt->mutex);
    if (n_frames < 0 || dt->write_pos + n_frames > dt->capacity) {
        /* Bounds guard (currently unreachable: caller pushes 1 frame at a time and
         * capacity is sized to max_frames). Drop the overflow rather than corrupt heap. */
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
        /* Leaks-audit fix (#5): the old int `(audio_len+n)*2` overflowed on long audio → an
         * undersized realloc then a memcpy heap-overflow. Size the growth in size_t and NULL-check
         * realloc before the memcpy. (audio_len stays int — bounded well under INT_MAX by the
         * sample-count pipeline; clamp the stored cap to avoid an int-cast UB at the extreme.) */
        size_t newcap = ((size_t)dt->audio_len + (size_t)n) * 2;
        float *nb = (float *)realloc(dt->audio_buf, newcap * sizeof(float));
        if (!nb) return;   /* OOM: keep the old buffer and drop this chunk rather than crash */
        dt->audio_buf = nb;
        dt->audio_cap = newcap > (size_t)0x7FFFFFFF ? 0x7FFFFFFF : (int)newcap;
    }
    memcpy(dt->audio_buf + dt->audio_len, samples, (size_t)n * sizeof(float));
    dt->audio_len += n;
}

static void *decoder_thread_fn(void *arg) {
    decoder_thread_t *dt = (decoder_thread_t *)arg;
    qwen_tts_ctx_t *ctx = dt->ctx;

    /* ctx->sd_stream is initialized by main thread before launching us.
     * We are the sole user of sd_stream during generation — main thread
     * only touches Talker/CP state, never speech decoder state. */

    for (;;) {
        int avail, is_done;
        pthread_mutex_lock(&dt->mutex);
        /* Ramped chunking: in streaming mode, emit a small FIRST chunk for low
         * TTFA, then fall back to the full chunk size for throughput. The
         * one-time small chunk costs a little extra decode (conv_rf recompute)
         * but only once, so overall RTF is unaffected. */
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

        /* Decode available frames */
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
                /* ICL onset trim: drop the first N frames of decoder output — the
                 * reference->target cold-start produces a "tud" transient at frame 0. */
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

/* Load model */
/* ── QWEN_THP=1: chiedi le huge page sui pesi mmappati (2026-08-21, ARM epic A5) ──
 *
 * Il carico e' memory-bound e il Code Predictor rilegge i suoi pesi 16 volte per frame.
 * A pagine da 4 KiB i 3.85 GB del 1.7B sono ~940 000 pagine contro le poche migliaia di
 * entry del TLB di secondo livello: una quota del tempo se ne va in page-table walk, e
 * nessun kernel SIMD la recupera.
 *
 * Su Linux le THP stanno in modalita' `madvise` su quasi tutte le VM cloud (verificato sul
 * c4a: `always [madvise] never`), quindi senza QUESTA chiamata non arrivano: non e' una
 * cosa che il kernel fa da solo.
 *
 * Perche' dal nostro lato e non dentro ingot: la mappa la crea la libreria vendorizzata e
 * una modifica li' sparirebbe alla prossima ri-vendorizzazione. L'intervallo si ricostruisce
 * dai tensori che ingot espone gia' (ingot_st_at + ingot_st_data + nbytes), quindi qui basta
 * una madvise sull'estensione complessiva.
 *
 * DEFAULT OFF: non cambia un bit del risultato, ma cambia come il kernel alloca la memoria
 * (una THP e' 2 MiB: su una macchina piena puo' costare RSS e latenza di fault). Si accende
 * per misurare e diventa default solo col numero in mano. Fuori da Linux, o su un kernel
 * senza MADV_HUGEPAGE, e' un no-op silenzioso. */
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
        if (!d) continue;                       /* tensore non mappato: niente da consigliare */
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
    /* EOS: V1 stays the default so existing behaviour is bit-identical unless
     * asked otherwise. V2 is opt-in until the short-text bank validates it. */
    ctx->eos_strategy         = QWEN_EOS_V1;
    ctx->eos_suppress_frames  = 2;
    ctx->eos_frames_per_token = 3.0f;
    ctx->eos_start_multiple   = 2.0f;
    ctx->eos_overhead_frames  = 18;
    ctx->eos_ramp_per_frame   = 0.5f;
    ctx->eos_ramp_cap         = 10.0f;
    ctx->eos_topk             = 50;
    ctx->max_tokens = 8192; ctx->cp_temperature = 0.9f; ctx->cp_top_k = 50;
    ctx->stream_chunk_frames = 10; /* default: 10 frames = 0.8s audio per chunk */
    /* Default speaker: Ryan (3061) - native English speaker
     * Serena (3066) and others are Chinese speakers which may cause issues with English */
    ctx->speaker_id = 3061; ctx->language_id = -1; ctx->seed = (uint32_t)time(NULL);
    ctx->silent = silent; ctx->debug = 0;
    ctx->use_int8 = use_int8; ctx->use_int4 = use_int4;

    /* Load config from model_dir or current dir */
    char config_path[1024];
    snprintf(config_path, sizeof(config_path), "%s/config.json", ctx->model_dir);
    if (load_config(ctx) != 0) {
        /* Try current directory */
        snprintf(config_path, sizeof(config_path), "config.json");
        if (load_config(ctx) != 0) { free(ctx); return NULL; }
    }
    
    qwen_tts_config_t *c = &ctx->config;

    /* Auto-detect model type from config.json */
    {
        char cfg_path[1024];
        snprintf(cfg_path, sizeof(cfg_path), "%s/config.json", ctx->model_dir);
        long cfg_len;
        char *cfg_raw = read_file(cfg_path, &cfg_len);
        if (cfg_raw) {
            /* Detect Base model: "tts_model_type": "base" */
            const char *mt = strstr(cfg_raw, "\"tts_model_type\"");
            if (mt) {
                const char *val = strchr(mt + 16, '"');
                if (val) {
                    val++;  /* skip opening quote */
                    if (strncmp(val, "base", 4) == 0) ctx->is_base_model = 1;
                }
            }

            /* Parse speaker_encoder_config enc_dim (1024 for 0.6B, 2048 for 1.7B) */
            const char *sec = strstr(cfg_raw, "\"speaker_encoder_config\"");
            if (sec) {
                ctx->speaker_enc_dim = json_get_int(sec, "enc_dim", 1024);
            }

            /* VoiceDesign: "spk_id": {} (empty object) */
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

            /* Speakers declared by the model itself. Base-derived finetunes (all
             * finetuned checkpoints) list theirs here; upstream Base and VoiceDesign
             * leave it empty, in which case the built-in presets remain the only table. */
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

    /* Load safetensors through ingot (mmap, index.json-aware, no caps) */
    char st_err[256] = "";
    ingot_st *st_main = NULL;
    if (ingot_st_open_dir(&st_main, ctx->model_dir, st_err, sizeof st_err) != 0) {
        fprintf(stderr, "Error: Failed to load model from %s: %s\n",
                ctx->model_dir, st_err);
        free(ctx); return NULL;
    }
    ctx->safetensors = st_main;
    qwen_weights_thp_advise(st_main, "talker + code predictor");
    /* Speech tokenizer is in a separate subdirectory */
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

    /* Load speaker encoder for Base models */
    if (ctx->is_base_model) {
        if (ctx->speaker_enc_dim > 0)
            ctx->speaker_enc.enc_dim = ctx->speaker_enc_dim;
        if (qwen_speaker_encoder_load(&ctx->speaker_enc, ctx->safetensors) != 0) {
            fprintf(stderr, "Warning: failed to load speaker encoder (voice cloning unavailable)\n");
        } else if (!ctx->silent) {
            fprintf(stderr, "  Speaker encoder: ECAPA-TDNN (enc_dim=%d)\n", ctx->speaker_enc.enc_dim);
        }
    }

    /* Pre-allocate text embedding temp buffers */
    int th = ctx->config.text_hidden_size;
    int h = ctx->config.hidden_size;
    ctx->emb_tmp1 = (float *)aligned_malloc(th * sizeof(float));
    ctx->emb_tmp2 = (float *)aligned_malloc(th * sizeof(float));

    /* Pre-compute special token embeddings (used every request) */
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

    /* Initialize LRU embedding cache (8MB for 2048 slots × 1024 hidden) */
    emb_cache_init(ctx);

    if (!ctx->silent) fprintf(stderr, "Model loaded in %.0f ms\n", time_ms() - t0);
    return ctx;
}

qwen_tts_ctx_t *qwen_tts_load(const char *model_dir) {
    return qwen_tts_load_ex(model_dir, 0, 0, 0);
}

/* Leaks-audit #3: record a malloc'd override buffer so unload frees it. Growable list. */
void qwen_track_override(qwen_tts_ctx_t *ctx, void *ptr) {
    if (!ctx || !ptr) return;
    if (ctx->n_owned_overrides >= ctx->cap_owned_overrides) {
        int nc = ctx->cap_owned_overrides ? ctx->cap_owned_overrides * 2 : 64;
        void **t = (void **)realloc(ctx->owned_overrides, (size_t)nc * sizeof(void *));
        if (!t) return;   /* OOM: skip tracking this one (leak it) rather than crash */
        ctx->owned_overrides = t;
        ctx->cap_owned_overrides = nc;
    }
    ctx->owned_overrides[ctx->n_owned_overrides++] = ptr;
}

void qwen_tts_unload(qwen_tts_ctx_t *ctx) {
    if (!ctx) return;
    /* Leaks-audit #3: free the WDELTA/WOVR/--expr override buffers that replaced mmap pointers.
     * A worker clone shares this list (shallow copy) and must be freed BEFORE its base, so only
     * the base reaches here with the list populated. */
    for (int i = 0; i < ctx->n_owned_overrides; i++) free(ctx->owned_overrides[i]);
    free(ctx->owned_overrides);
    /* Free malloc'd fused weights (gate_up are the only malloc'd weight copies) */
    for (int i = 0; i < ctx->config.num_layers; i++) free(ctx->layers[i].gate_up_fused_bf16);
    for (int i = 0; i < ctx->config.cp_num_layers; i++) free(ctx->cp_layers[i].gate_up_fused_bf16);
    for (int i = 0; i < ctx->config.cp_num_layers; i++) free(ctx->cp_layers[i].down_q2_rough);
    /* Free pre-converted F32 codec embeddings */
    /* codec_embedding_f32 removed — vectorized bf16→f32 conversion used instead */
    /* Free malloc'd codebooks (EMA-reconstructed, not from safetensors) */
    for (int i = 0; i < 16; i++) free(ctx->speech_dec.codebook[i]);
    free(ctx->speech_dec.pre_layers);
    free(ctx->speech_dec.rope_cos); free(ctx->speech_dec.rope_sin);
    /* Close safetensors (all get_bf16/get_f32 pointers point into this data) */
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
    /* Free runtime buffers */
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

/* ── Worker clone (concurrent server) ───────────────────────────────────
 *
 * Produce an independent context that SHARES all read-only state with `base`
 * (mmapped weights, quantized arrays, codebooks, RoPE caches, the cloned
 * voice's overridden/quantized weights + speaker embedding, precomputed
 * special-token embeddings) but owns FRESH copies of every buffer that the
 * generation path mutates (KV caches, per-step work buffers, the embedding
 * LRU cache, the delta-prefill cache, sampling params). This lets N server
 * workers each run a synthesis concurrently without aliasing each other's
 * state, while paying the weight memory only once.
 *
 * Sizes mirror the allocation sites in qwen_talker_load / qwen_cp_load /
 * qwen_tts_load_ex exactly (config is identical across clones). The lazily
 * grown buffers (pref_*, logits, codec_codes, prev_tokens, prev_input_embeds,
 * sd_stream.*) start NULL and are realloc'd per worker on first request.
 *
 * Free a clone with qwen_tts_free_clone (frees ONLY the per-worker buffers —
 * never the shared weights/safetensors). NEVER pass a clone to
 * qwen_tts_unload (it would close the shared safetensors / free shared
 * weights, corrupting the base and the other clones). */
qwen_tts_ctx_t *qwen_tts_clone_for_worker(const qwen_tts_ctx_t *base) {
    if (!base) return NULL;
    qwen_tts_ctx_t *w = (qwen_tts_ctx_t *)malloc(sizeof(qwen_tts_ctx_t));
    if (!w) return NULL;
    *w = *base;   /* share every pointer by default; override the mutable ones below */

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

    /* Talker KV cache + decode buffers (see qwen_talker_load) */
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

    /* CP KV cache + decode buffers (see qwen_cp_load) */
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

    /* Text-embedding temp buffers (mutated per embed call) */
    w->emb_tmp1 = (float *)aligned_malloc(th * sizeof(float));
    w->emb_tmp2 = (float *)aligned_malloc(th * sizeof(float));

    /* Per-worker LRU embedding cache (emb_cache_init reads w->emb_cache.* fresh) */
    memset(&w->emb_cache, 0, sizeof(w->emb_cache));
    emb_cache_init(w);

    /* Lazily grown buffers — start empty, realloc'd per worker on first request */
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

    /* Per-worker CP roughness buffers: `*w = *base` byte-copied base's cp_layers
     * (down_q2_rough pointers + cp_rough_built). Detach them so each worker builds
     * and owns its own (freed in qwen_tts_free_clone) — never shares/double-frees base's. */
    w->cp_rough_built = 0;
    for (int i = 0; i < c->cp_num_layers; i++) w->cp_layers[i].down_q2_rough = NULL;

    /* Per-worker tokenizer: loaded lazily on first generate (avoids sharing a
     * single tokenizer across threads). instruct/tf/streaming reset per request. */
    w->cached_tokenizer = NULL;
    w->instruct = NULL;
    w->tf_ref_codes = NULL;
    w->stream = 0; w->audio_cb = NULL; w->audio_cb_userdata = NULL;

    return w;
}

/* Free a worker clone: ONLY the per-worker buffers (mirrors the runtime-buffer
 * subset of qwen_tts_unload). Shared weights/safetensors/rope/voice belong to
 * the base ctx and must NOT be touched here. */
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
    /* Per-worker lazily-built CP roughness buffers (NULL unless cp_roughness>0 used). */
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

/* Codec embedding lookup — vectorized bf16→f32 conversion */
/* QWEN_SPK_DEBUG=1: prove - not assume - that every backend resolves and reads the
 * SAME speaker row. The conditioning of this finetune is one row of codec_embedding
 * selected by a pool id (3000-3009), so if two runs differ here the cause is not the
 * quantization of the transformer weights at all. Printed once per token id. */
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

    /* Tokenize instruct if provided (1.7B only).
     * Format: "<|im_start|>user\n{instruct}<|im_end|>\n"
     * These tokens get embedded via text_projection and prepended to input_embeds. */
    int32_t *instruct_tokens = NULL;
    int instruct_token_len = 0;
    /* Use cached tokenizer if available, otherwise load and cache */
    qwen_tokenizer_t *tok = (qwen_tokenizer_t *)ctx->cached_tokenizer;
    if (!tok) {
        tok = qwen_tokenizer_load(ctx->model_dir);
        /* Cache for future calls */
        if (tok) ctx->cached_tokenizer = tok;
    }

    if (ctx->instruct && ctx->instruct[0] && tok) {
        /* Build instruct template: <|im_start|>user\n{instruct}<|im_end|>\n */
        int inst_len = (int)strlen(ctx->instruct);
        int tmpl_len = inst_len + 64;
        char *instruct_tmpl = (char *)malloc(tmpl_len);
        snprintf(instruct_tmpl, tmpl_len, "<|im_start|>user\n%s<|im_end|>\n", ctx->instruct);
        instruct_tokens = qwen_tokenizer_encode(tok, instruct_tmpl, &instruct_token_len);
        free(instruct_tmpl);
        if (!ctx->silent && instruct_tokens)
            fprintf(stderr, "Instruct: \"%s\" (%d tokens)\n", ctx->instruct, instruct_token_len);
    }

    /* Build token sequence matching Python:
     * [<|im_start|>, assistant, \n, ...BPE(text)..., <|im_end|>, \n, <|im_start|>, assistant, \n]
     * Special tokens use their IDs directly; only the user text is BPE-encoded.
     * Role prefix = [:3], text_content = [3:-5], suffix [-5:] discarded.
     */
    int32_t *text_tokens = NULL;
    int text_token_len = 0;
    int32_t *ref_text_tokens = NULL;
    int ref_text_token_len = 0;
    if (tok) {
        text_tokens = qwen_tokenizer_encode_para(tok, text, &text_token_len);
        /* ICL mode: also tokenize reference text. --emo-ref brings its own transcript and takes
         * precedence — its codec anchor is what carries the emotion (and it does NOT require
         * voice_clone, so it works on CustomVoice with a preset speaker). */
        const char *icl_text = (ctx->emo_ref_path && ctx->emo_ref_text) ? ctx->emo_ref_text
                             : ((ctx->voice_clone && !ctx->xvector_only) ? ctx->ref_text : NULL);
        if (icl_text) {
            ref_text_tokens = qwen_tokenizer_encode(tok, icl_text, &ref_text_token_len);
            if (!ctx->silent && ref_text_tokens)
                fprintf(stderr, "Ref text: \"%s\" (%d tokens)\n", icl_text, ref_text_token_len);
        }
        /* tok is cached in ctx->cached_tokenizer — do not free */
    }
    if (!text_tokens || text_token_len == 0) {
        fprintf(stderr, "Error: text tokenization failed\n");
        free(text_tokens);
        free(instruct_tokens);   /* audit #7: were leaked on this error path */
        free(ref_text_tokens);
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
        /* The EOS policy is part of the experiment, so it must be in the log:
         * two runs with different strategies are not the same measurement. The
         * ladder harness parses this line into every clip record. */
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

    /* Build codec-side prefix:
     * CustomVoice with language: [THINK, THINK_BOS, language_id, THINK_EOS, speaker, PAD, BOS]
     * CustomVoice without language: [NO_THINK, THINK_BOS, THINK_EOS, speaker, PAD, BOS]
     * VoiceDesign with language: [THINK, THINK_BOS, language_id, THINK_EOS, PAD, BOS]
     * VoiceDesign without language: [NO_THINK, THINK_BOS, THINK_EOS, PAD, BOS]
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
    /* Speaker position in codec prefix:
     * - CustomVoice: discrete speaker token from spk_id
     * - VoiceDesign: no speaker (voice from instruct)
     * - Voice Clone: continuous speaker embedding replaces token
     *   We use -1 as sentinel; the embedding loop handles it specially. */
    if (ctx->voice_clone && ctx->speaker_embedding) {
        codec_tokens[codec_len++] = -1;  /* placeholder — will use speaker_embedding */
    } else if (!ctx->voice_design) {
        codec_tokens[codec_len++] = ctx->speaker_id;
    }
    codec_tokens[codec_len++] = QWEN_TTS_CODEC_PAD;
    codec_tokens[codec_len++] = QWEN_TTS_CODEC_BOS;

    /* Special token embeddings: use pre-computed cache from load time */
    const float *tts_pad_embed = ctx->cached_tts_pad_embed;
    const float *tts_bos_embed = ctx->cached_tts_bos_embed;
    const float *tts_eos_embed = ctx->cached_tts_eos_embed;

    float *codec_pad_embed = (float *)aligned_malloc(h * sizeof(float));
    float *codec_bos_embed = (float *)aligned_malloc(h * sizeof(float));
    lookup_codec_embed(ctx, QWEN_TTS_CODEC_PAD, codec_pad_embed);
    lookup_codec_embed(ctx, QWEN_TTS_CODEC_BOS, codec_bos_embed);

    /* === ICL mode: use cached ref_codes (.qvoice) or encode reference audio === */
    int *ref_codes = NULL;
    int ref_n_frames = 0;
    int ref_codes_owned = 0;  /* 1 if we allocated ref_codes and must free it */
    int icl_mode = 0;

    if (ctx->graft_mode && ctx->cached_ref_codes && !ctx->silent)
        fprintf(stderr, "ICL: --graft -> ignoring %d ref frames, cloning via x-vector (emotive)\n",
                ctx->cached_ref_n_frames);

    /* --emo-ref (EMOTION BY EXAMPLE): encode the emotional reference and use ONLY its codec
     * tokens as the ICL anchor. Checked FIRST so it wins over any clone-supplied ref_codes.
     * Crucially this does NOT set/need voice_clone: the speaker slot below stays whatever it
     * already was (preset -s on CustomVoice, or a loaded clone's x-vector), so identity and
     * emotion come from two independent places. That is the whole point on the 0.6B, which has
     * no steerable emotion subspace but shares the codec vocabulary with the 1.7B. */
    if (ctx->emo_ref_path && ctx->emo_ref_text && ref_text_tokens && ref_text_token_len > 0) {
        float *emo_samples = NULL;
        int emo_n_samples = 0, emo_sr = 0;
        if (qwen_read_wav(ctx->emo_ref_path, &emo_samples, &emo_n_samples, &emo_sr) != 0) {
            fprintf(stderr, "Error: failed to read emotion reference %s\n", ctx->emo_ref_path);
        } else {
            if (emo_sr != QWEN_TTS_SAMPLE_RATE && !ctx->silent)
                fprintf(stderr, "Warning: emo-ref sample rate %d, expected %d\n",
                        emo_sr, QWEN_TTS_SAMPLE_RATE);
            /* Same trailing-fade trim as the clone paths: a fade-out tail poisons the anchor. */
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
    /* Check for cached ref_codes from .qvoice file (skipped in --graft mode) */
    else if (ctx->voice_clone && !ctx->graft_mode && ctx->cached_ref_codes && ctx->cached_ref_n_frames > 0
        && ctx->ref_text && ref_text_tokens && ref_text_token_len > 0) {
        ref_codes = ctx->cached_ref_codes;
        ref_n_frames = ctx->cached_ref_n_frames;
        icl_mode = 1;
        if (!ctx->silent)
            fprintf(stderr, "ICL: using %d cached ref frames from .qvoice\n", ref_n_frames);
    }
    /* Otherwise encode from ref audio file */
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
            /* Same trailing-fade trim as the ECAPA path: the ICL codec prefix is
             * what carries prosody, so a fade-out tail here is the real culprit. */
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

    /* --icl-frames N (or QWEN_ICL_FRAMES): cap the reference codec frames fed as the ICL
     * prosody anchor. The ref_codes are BOTH the identity carrier AND the prosody template
     * that damps instruct/emotion (RUN-6). Trimming the anchor to the first N frames leaves
     * the speaker-embedding + ref_text identity binding intact while freeing room for emotion
     * — interpolating ICL (faithful/flat) toward the x-vector graft (expressive/looser).
     * 0 = use all (default). Keeps the contiguous head so the codec sequence stays coherent. */
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

    /*
     * Build prefill:
     *
     * Section 0: Instruct tokens (text-only, NO codec pairing) — only if instruct provided
     * Section 1: Role prefix (3 positions) - text-only, NO codec pairing
     * Section 2: tts_pad*(codec_len-2) + tts_bos  paired with  codec[0..codec_len-2]
     *
     * Normal mode (non-ICL):
     *   Section 3: text_content[0..N-1] + tts_eos  paired with  codec_pad * (N+1)
     *   Section 4: tts_pad + codec_bos  (1 position)
     *
     * ICL mode:
     *   Section 3': ref_text + target_text + tts_eos  paired with  codec_pad
     *   Section 4': tts_pad * (ref_n_frames+1)  paired with  codec_bos + ref_code_embeds
     */
    int sec2_len = codec_len - 1;  /* codec tokens without the last (BOS) */
    int inst_len = instruct_tokens ? instruct_token_len : 0;

    int sec3_len, sec4_len;
    if (icl_mode) {
        sec3_len = ref_text_token_len + text_content_len + 1;  /* ref_text + text + eos */
        sec4_len = ref_n_frames + 1;                           /* bos + ref_codes */
    } else {
        sec3_len = text_content_len + 1;  /* text + eos */
        sec4_len = 1;                     /* bos */
    }
    int prefill_len = inst_len + role_len + sec2_len + sec3_len + sec4_len;

    float *input_embeds = (float *)aligned_calloc((int64_t)prefill_len * h, sizeof(float));
    float *tmp_embed = (float *)aligned_malloc(h * sizeof(float));
    int pos = 0;

    /* ── Prefix-cache key ─────────────────────────────────────────────────────────
     * The first `inst_len + role_len + sec2_len` positions are built from the instruct
     * tokens, the ChatML role tokens and the codec prefix - and the codec prefix IS
     * [think mode, THINK_BOS, language_id, THINK_EOS, speaker_id, PAD]. Not one of them
     * depends on the text. So hashing exactly those three arrays gives a key that
     * changes if and only if the cacheable prefix changes; there is no separate
     * speaker/language field to forget to compare.
     *
     * Disabled for ICL and voice clone: there the codec side carries a CONTINUOUS
     * speaker embedding (and, in ICL, reference audio), so equal token ids do not mean
     * equal input. Rather than hash a float vector, those paths simply do not cache. */
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

    /* Section 0: Instruct tokens (text-only, no codec) */
    for (int i = 0; i < inst_len; i++) {
        embed_one_text_token(ctx, instruct_tokens[i], input_embeds + (int64_t)pos * h);
        pos++;
    }
    free(instruct_tokens);

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
        /* Codec side: codec_tokens[i] or speaker embedding for voice clone */
        if (codec_tokens[i] == -1 && ctx->voice_clone && ctx->speaker_embedding) {
            /* Voice clone: use continuous speaker embedding.
             * For cross-model injection (ECAPA embedding into CustomVoice), scale
             * the embedding norm to match the model's preset speaker norms.
             * This is needed because ECAPA and codec embedding tables have different
             * norm ranges (e.g., ECAPA ~17 vs CustomVoice ~14.5 on 1.7B). */
            float emb_norm = 0;
            for (int j = 0; j < h; j++) emb_norm += ctx->speaker_embedding[j] * ctx->speaker_embedding[j];
            emb_norm = sqrtf(emb_norm);

            /* Compute target norm from a reference preset speaker (ryan=3061) */
            float ref_norm = 0;
            {
                float tmp_ref[4096];
                lookup_codec_embed(ctx, 3061, tmp_ref);  /* ryan */
                for (int j = 0; j < h; j++) ref_norm += tmp_ref[j] * tmp_ref[j];
                ref_norm = sqrtf(ref_norm);
            }

            float scale = (ref_norm > 0.1f && emb_norm > 0.1f) ? ref_norm / emb_norm : 1.0f;
            /* Relax-identity lever (QWEN_SPK_SCALE, default 1.0 → no change): scale the speaker
             * embedding contribution. <1.0 loosens the identity clamp on the register/pitch range
             * (lets emotional steering reach bigger F0 excursions, like VoiceDesign), at the cost
             * of some voice fidelity; >1.0 tightens identity. Experimental, env-gated. */
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
        /* Section 3' (ICL): ref_text + target_text + tts_eos, all paired with codec_pad */
        for (int i = 0; i < sec3_len; i++) {
            float *dst = input_embeds + (int64_t)pos * h;
            if (i < ref_text_token_len) {
                /* Reference text tokens */
                embed_one_text_token(ctx, ref_text_tokens[i], dst);
            } else if (i < ref_text_token_len + text_content_len) {
                /* Target text tokens */
                embed_one_text_token(ctx, all_tokens[role_len + (i - ref_text_token_len)], dst);
            } else {
                /* tts_eos at the end */
                memcpy(dst, tts_eos_embed, h * sizeof(float));
            }
            for (int j = 0; j < h; j++) dst[j] += codec_pad_embed[j];
            pos++;
        }

        /* Section 4' (ICL): tts_pad + (codec_bos + ref_code_embeds) */
        for (int i = 0; i < sec4_len; i++) {
            float *dst = input_embeds + (int64_t)pos * h;
            /* Text side: tts_pad */
            memcpy(dst, tts_pad_embed, h * sizeof(float));
            /* Codec side */
            if (i == 0) {
                /* First position: codec_bos */
                for (int j = 0; j < h; j++) dst[j] += codec_bos_embed[j];
            } else {
                /* Ref code frame: sum 16 codebook embeddings */
                int frame = i - 1;
                /* Codebook 0: talker's codec_embedding */
                int code0 = ref_codes[frame * 16];
                lookup_codec_embed(ctx, code0, tmp_embed);
                for (int j = 0; j < h; j++) dst[j] += tmp_embed[j];
                /* Codebooks 1-15: CP codec embeddings */
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
    }

    free(all_tokens);
    free(tmp_embed);
    /* tts_pad/bos/eos_embed are ctx-owned cache — do not free */
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

    /* Debug: check speech decoder weights before prefill */
    if (ctx->debug && ctx->speech_dec.pre_conv_weight) {
        fprintf(stderr, "[CORR] pre-prefill: pre_conv_w[0]=%.6f\n", ctx->speech_dec.pre_conv_weight[0]);
    }

    /* Delta prefill: compare with previous embeddings to find reusable KV prefix.
     * For server mode, consecutive calls with the same speaker/language share
     * the role+codec prefix (~8-9 tokens), so we skip re-prefilling those. */
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

    /* BUGFIX (2026-06-03): a FULL prefix match (delta_start == prefill_len) — i.e. two
     * identical consecutive server requests — skipped the `if (delta_start < prefill_len)`
     * block entirely, so NO prefill/step ran and ctx->dec_x (read below to seed
     * last_hidden = the first generated frame) stayed STALE from the PREVIOUS request's
     * last token step. Identical requests therefore diverged (verified: 3 identical reqs
     * -> 291884/311084/257324 B, even -j1 temp0). Fix: on a full match, do a full fresh
     * prefill (delta_start = 0) so the request is bit-identical to a cold run. The
     * delta-prefill optimization is preserved for PARTIAL matches (the real server case:
     * same speaker/language prefix, different text) — those re-step the new tokens via
     * the sequential path and correctly repopulate dec_x at the last position. */
    if (delta_start >= prefill_len) delta_start = 0;

#if defined(QWEN_HAVE_CUDA) || defined(QWEN_HAVE_METAL)
    /* Issue #19 (part 2): fused GPU Talker + emotion steering. Steered steps run on the
     * CPU path and read the HOST KV, but a fused delta-prefill writes the device KV only,
     * so host rows [delta_start, prefill_len) may be stale from an earlier request. A
     * steered decode would then attend to the previous request's text. Force a full fresh
     * prefill (host + device KV both repopulated) whenever this request will steer. */
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

    /* Reset KV cache to the reusable prefix length */
    ctx->kv_len = delta_start;

    /* No emotion steering during prefill (the gen loop sets w_eff per frame). Reset here
     * so a server-reused ctx doesn't inherit the previous request's effective weight. */
    ctx->ml_steer_w_eff = 0.0f;

    /* Leaks-audit #3 MED: the RoPE cos/sin cache holds rope_cache_len (8192) positions.
     * Prefill applies RoPE at positions [0, prefill_len); a longer prompt would index past
     * the cache -> heap overread -> garbage rotations/audio. Refuse it rather than corrupt. */
    if (prefill_len > ctx->rope_cache_len) {
        fprintf(stderr, "Error: prompt too long (%d tokens > RoPE cache %d); shorten the text.\n",
                prefill_len, ctx->rope_cache_len);
        free(input_embeds);
        return -1;
    }

    double t_prefill = time_ms();
    if (delta_start < prefill_len) {
        if (delta_start > 0) {
            /* Sequential prefill only for the server delta-reuse case (the BLAS
             * batch prefill assumes it processes the full sequence from pos 0).
             * NOTE: quantized (int8/int4) mode also uses the BATCHED path now —
             * the bf16 weights are still mmap-resident (quantization doesn't free
             * them), so the batched sgemm prefill works and is ~2x faster than the
             * sequential int8 step path (cuts TTFA). Generation still uses int8. */
            float *dummy_hidden = (float *)malloc(h * sizeof(float));
            for (int t = delta_start; t < prefill_len; t++) {
                if (qwen_talker_step(ctx, input_embeds + (int64_t)t * h, dummy_hidden) != 0) {
                    free(input_embeds); free(dummy_hidden);
                    return -1;
                }
            }
            free(dummy_hidden);
        } else {
            /* Full BLAS batch prefill (first call, no delta, no quantization) */
            if (qwen_talker_prefill(ctx, input_embeds, prefill_len) != 0) {
                free(input_embeds);
                return -1;
            }
        }
    }
#ifdef QWEN_HAVE_CUDA
    /* Fused GPU Talker KV sync (issue #19). The device KV is the source of truth in fused mode:
     *  - delta_start == 0 (full CPU batched prefill): host kv_cache_k/v is freshly populated
     *    by qwen_talker_prefill → mirror it to the device once.
     *  - delta_start > 0 (server delta-reuse): the new tokens were stepped through the fused
     *    GPU step, which writes the device KV DIRECTLY and skips the host KV. The matching
     *    prefix [0, delta_start) is still valid on the device from the previous request, so
     *    the device KV is ALREADY correct — uploading here would clobber it with the stale
     *    host KV (the previous request's text) and make the model replay the old prompt. */
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

    /* Cache current embeddings for delta prefill on next call */
    if (!ctx->prev_input_embeds || ctx->prev_prefill_len < prefill_len) {
        free(ctx->prev_input_embeds);
        ctx->prev_input_embeds = (float *)malloc((int64_t)prefill_len * h * sizeof(float));
    }
    if (ctx->prev_input_embeds) {
        memcpy(ctx->prev_input_embeds, input_embeds, (int64_t)prefill_len * h * sizeof(float));
        ctx->prev_prefill_len = prefill_len;
    }

    free(input_embeds);

    /* Debug: check speech decoder weights after prefill */
    if (ctx->debug && ctx->speech_dec.pre_conv_weight) {
        fprintf(stderr, "[CORR] post-prefill: pre_conv_w[0]=%.6f\n", ctx->speech_dec.pre_conv_weight[0]);
    }

    /* --batch prefill-only: KV is populated and ctx->dec_x holds the last prefill
     * position's pre-norm hidden. The orchestrator captures both; stop here (no
     * generation/decode). Additive: prefill_only is 0 on the normal path. */
    if (ctx->prefill_only) {
        ctx->bg_text_content_len = text_content_len;
        return 0;
    }

    /* Get hidden state from last prefill position (apply final norm) */
    float *last_hidden = (float *)malloc(h * sizeof(float));
    qwen_rms_norm(last_hidden, ctx->dec_x, ctx->talker_norm, 1, h, ctx->config.rms_norm_eps);

    /* Autoregressive generation */
    int max_frames = ctx->max_tokens;
    /* Leaks-audit #3 MED: generation RoPE position = prefill_len + frame; cap frames so the
     * last position stays within the RoPE cache (prefill_len <= rope_cache_len is guaranteed
     * by the prefill guard above). Without this, a run to max_tokens past 8192 reads OOB. */
    if (max_frames > ctx->rope_cache_len - prefill_len)
        max_frames = ctx->rope_cache_len - prefill_len;
    ctx->codec_codes = (int *)realloc(ctx->codec_codes, (int64_t)max_frames * 16 * sizeof(int));
    ctx->codec_frames = 0;
    ctx->prev_tokens = (int *)realloc(ctx->prev_tokens, max_frames * sizeof(int));
    ctx->n_prev_tokens = 0;
    ctx->logits = (float *)realloc(ctx->logits, ctx->config.codec_vocab_size * sizeof(float));

    double t_cp_total = 0, t_talker_step_total = 0, t_embed_total = 0;
    float *step_embed = (float *)malloc(h * sizeof(float));

    /* Quant-ladder teacher-forcing (QWEN_TF_CODES=<bf16-reference>.codes): REPLAY
     * the reference 16-codes-per-frame stream — override code0 and feed the
     * reference codebook-1..15 back into the Talker (rails identical to bf16) while
     * the CP at the current precision RECORDS what it WOULD predict. Every precision
     * then sees bit-identical Talker hidden states and CP inputs → the per-codebook
     * disagreement vs reference is PURE CP quant drift, free of trajectory fork.
     * Only the autoregressive feedback coupling (CP codes → next Talker step) makes
     * a free-running comparison meaningless; this isolates it. NULL → normal synth. */
    int   *tf_codes = NULL;     /* nframes × 16 reference codes */
    int    tf_nframes = 0;
    /* QWEN_TF_CB_KEEP=N (codec-VC experiment): in TF replay, KEEP the model's own first N codebooks
     * (predicted with the TARGET voice loaded) and override only codebooks N..15 from the reference
     * clip. N=0 (default) = override all 16 (pure replay). N=1 keeps the coarse cb0 (timbre-ish) from
     * the target voice while taking the fine cb1-15 articulation from a real cough -> cross-voice splice. */
    int    tf_cb_keep = 0;
    { const char *k = getenv("QWEN_TF_CB_KEEP"); if (k && *k) { tf_cb_keep = atoi(k); if (tf_cb_keep < 0) tf_cb_keep = 0; if (tf_cb_keep > 16) tf_cb_keep = 16; } }
    FILE  *code0_fp = NULL;     /* QWEN_DUMP_CODE0: Talker's greedy code0 prediction per frame */
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

    /* Launch decoder thread for pipeline overlap (always — both streaming and normal).
     * In streaming mode, the decoder thread calls audio_cb directly.
     * In normal mode, it accumulates audio to a buffer. */
    decoder_thread_t dt_state;
    pthread_t dt_thread;
    /* DIAGNOSTIC: QWEN_NO_OVERLAP=1 runs the speech decoder SYNCHRONOUSLY (no overlap
     * pthread) — used to test whether the decoder thread is the source of the intermittent
     * -j1 temp0 non-determinism. With it set, the thread is never spawned; frames buffer
     * into dt->codes during generation and are drained by a single synchronous
     * decoder_thread_fn() call at the end (done=1 → it processes all and returns). */
    int dt_no_overlap = (getenv("QWEN_NO_OVERLAP") != NULL);
#ifdef QWEN_HAVE_CUDA
    /* M3 scheduling — MEASURED on GB10: on ONE GPU, overlapping generation (fused Talker/CP) with
     * the resident decoder makes it WORSE (RTF 0.78 vs 0.62) — they contend for the same SMs, so
     * generation slows (14→20 ms/f) while the decoder overlaps. Sequential wins. So force
     * synchronous decode when the resident decoder is on and NOT streaming (full-file: RTF matters).
     * Streaming keeps overlap (TTFA matters — sync would push first audio to the very end). */
    { extern int g_cuda_decoder_conv_on; if (g_cuda_decoder_conv_on && !ctx->stream) dt_no_overlap = 1; }
#endif
    qwen_sd_stream_init(&ctx->sd_stream);
    dt_init(&dt_state, ctx, max_frames);
    if (ctx->stream && ctx->audio_cb) {
        dt_state.audio_cb = ctx->audio_cb;
        dt_state.audio_cb_userdata = ctx->audio_cb_userdata;
    }
    /* ICL onset fix: drop the first generated frame(s) of audio. In ICL mode the Talker is
     * primed with reference frames (which are NOT decoded), so the decoder cold-starts on an
     * already-loud first frame → a "tud" transient. The non-ICL/qvoice path ramps from silence
     * and is clean. Tunable via QWEN_ICL_TRIM_FRAMES (default 1, 0 disables). */
    if (icl_mode) {
        int trim_frames = 2;  /* ear-tuned: 2 frames (160ms) starts from silence, kills the "tud" */
        const char *e = getenv("QWEN_ICL_TRIM_FRAMES");
        if (e) trim_frames = atoi(e);
        if (trim_frames > 0) dt_state.trim_head_left = trim_frames * 1920;
    }
    if (!dt_no_overlap) {
        pthread_create(&dt_thread, NULL, decoder_thread_fn, &dt_state);
        /* From here the decoder thread is the BLAS user, and it runs alongside
         * our matvec pool. Prefill (above) had BLAS to itself and kept all -j
         * threads; hand one back now. Measured on 4-core Neoverse-N1 (0.6B
         * --int4 -j4): BLAS at 3 gives RTF 1.47 stream / 1.51 file against 1.54
         * / 1.59 at 4, with TTFA untouched because prefill still ran wide.
         * Dropping to 1 is a disaster (RTF 2.14): the decoder becomes the
         * bottleneck. A strict `ours + BLAS == cores` split is wrong -- mild
         * oversubscription wins, because our pool has serial stretches in which
         * the cores must be free to go to the decoder. */
        int nt = qwen_get_threads();
        /* rental-prep (pr17 §5.5): the per-mode OPENBLAS optimum on N1 differs (file=2,
         * stream=3 at 4 vCPU) — QWEN_BLAS_GEN_THREADS overrides the generation-phase
         * default so the box can sweep it without rebuilds. No-op on Accelerate. */
        int gen_blas = nt > 1 ? nt - 1 : 1;
        { const char *e = getenv("QWEN_BLAS_GEN_THREADS");
          if (e && atoi(e) > 0) gen_blas = atoi(e); }
        qwen_blas_set_threads(gen_blas);
    }

    for (int frame = 0; frame < max_frames; frame++) {
        qwen_census_frame();   /* shape census: the denominator for calls/frame */
        /* Codec head: logits = codec_head @ last_hidden */
        if (ctx->codec_head_q4)
            qwen_matvec_q4_0(ctx->logits, ctx->codec_head_q4, last_hidden, ctx->config.codec_vocab_size, h);
        else
            matvec_bf16(ctx->logits, ctx->codec_head_bf16, last_hidden, ctx->config.codec_vocab_size, h);

        /* Clip logits */
        for (int t = 0; t < ctx->config.codec_vocab_size; t++) {
            if (ctx->logits[t] > 100.0f) ctx->logits[t] = 100.0f;
            if (ctx->logits[t] < -100.0f) ctx->logits[t] = -100.0f;
        }

        /* Suppress special tokens (>= 2048) except EOS (2150) */
        for (int t = 2048; t < ctx->config.codec_vocab_size; t++)
            if (t != QWEN_TTS_CODEC_EOS) ctx->logits[t] = -1e30f;

        /* Suppress EOS for the leading frames (every strategy except OFF) */
        if (ctx->eos_strategy != QWEN_EOS_OFF && frame < ctx->eos_suppress_frames)
            ctx->logits[QWEN_TTS_CODEC_EOS] = -1e30f;

        /* ── EOS assist, switchable ─────────────────────────────────────────
         * There is nothing to "conform" to: their PyTorch path and their
         * production runtime disagree with each other. Measurements behind V2:
         * the design notes. */
        if (ctx->eos_strategy == QWEN_EOS_V1 || ctx->eos_strategy == QWEN_EOS_V2) {
            /* V1 assumes the clip length scales with the token count. It does
             * not: onset and leading/trailing silence are FIXED cost. Measured
             * on one finetuned checkpoint and voice, the real ratio is 5.75
             * frames per token on a 4-token text but 3.19 on a 62-token one, so
             * a purely proportional threshold sits right on top of short clips
             * — "Abeg wait." generated 23 frames against a V1 threshold of 24.
             * V2 adds that fixed overhead K back, which is what makes the
             * threshold track reality at both ends. */
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
            /* Their PyTorch EosBoostLogitsProcessor: from step `suppress_frames`
             * onward, lift EOS *to* the k-th highest logit when it sits below,
             * so it survives top-k filtering. NOTE: under greedy this cannot
             * change the argmax — the k-th logit is never the maximum — so it is
             * inert by construction unless sampling. Kept for parity work. */
            if (frame >= ctx->eos_suppress_frames) {
                float top[64];
                int kmax = (int)(sizeof(top) / sizeof(top[0]));
                int k = ctx->eos_topk > 0 ? ctx->eos_topk : 50;
                if (k > ctx->config.codec_vocab_size) k = ctx->config.codec_vocab_size;
                if (k > kmax) k = kmax;
                for (int i = 0; i < k; i++) top[i] = -1e30f;
                for (int t = 0; t < ctx->config.codec_vocab_size; t++) {
                    float v = ctx->logits[t];
                    if (v <= top[k - 1]) continue;   /* below the running k-th */
                    int i = k - 1;
                    while (i > 0 && top[i - 1] < v) { top[i] = top[i - 1]; i--; }
                    top[i] = v;
                }
                if (ctx->logits[QWEN_TTS_CODEC_EOS] < top[k - 1])
                    ctx->logits[QWEN_TTS_CODEC_EOS] = top[k - 1];
            }
        }

        /* Debug logging */
        if (ctx->debug && frame < 30) {
            float eos_logit = ctx->logits[QWEN_TTS_CODEC_EOS];
            int eos_rank = 0;
            for (int t = 0; t < ctx->config.codec_vocab_size; t++)
                if (ctx->logits[t] > eos_logit) eos_rank++;
            fprintf(stderr, "  [frame %d] EOS logit=%.2f rank=%d\n", frame, eos_logit, eos_rank);
        }

        /* Sample code0 — use greedy for warmup frames to reduce cross-model divergence */
        float frame_temp = ctx->temperature;
        int frame_top_k = ctx->top_k;
        if (ctx->greedy_warmup > 0 && frame < ctx->greedy_warmup) {
            frame_temp = 0.0f;
            frame_top_k = 1;
        }
        int code0 = qwen_tts_sample(ctx->logits, ctx->config.codec_vocab_size,
                                     frame_temp, frame_top_k, ctx->top_p,
                                     ctx->rep_penalty, ctx->prev_tokens, ctx->n_prev_tokens);

        /* Quant-ladder: record the Talker's would-be code0 prediction (greedy, this is
         * pre-override). vs the bf16 reference code0 column → Talker quant sensitivity on
         * the WORDS. In TF mode last_hidden is on the bf16 rails, so this isolates it. */
        if (code0_fp) fprintf(code0_fp, "%d\n", code0);

        /* Teacher-forcing replay: ride the reference rails (code0 + CP feedback). */
        if (tf_codes) {
            if (frame >= tf_nframes) break;
            /* codec-VC: keep the model's own code0 (target-voice timbre) when tf_cb_keep>=1 */
            if (tf_cb_keep < 1) code0 = tf_codes[(int64_t)frame * 16 + 0];
            ctx->tf_ref_codes = tf_codes + (int64_t)frame * 16 + 1;
        }

        /* In codec-VC mode the model's own code0 may sample EOS early; fall back to the reference
         * code0 so the replay runs the full clip length instead of stopping. */
        if (code0 == QWEN_TTS_CODEC_EOS && tf_codes && tf_cb_keep >= 1)
            code0 = tf_codes[(int64_t)frame * 16 + 0];

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

        /* TF replay: the CP just RECORDED its predictions (via QWEN_DUMP_CODES); now
         * overwrite with the reference so the Talker's next input stays on the bf16
         * rails (identical hidden states for every precision). */
        if (tf_codes)
            memcpy(codes + tf_cb_keep, tf_codes + (int64_t)frame * 16 + tf_cb_keep,
                   (size_t)(16 - tf_cb_keep) * sizeof(int));

        memcpy(ctx->codec_codes + (int64_t)ctx->codec_frames * 16, codes, 16 * sizeof(int));
        ctx->codec_frames++;

        /* Push frame to decoder thread for pipeline overlap */
        dt_push_frames(&dt_state, codes, 1);

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

        /* Check if streaming callback was aborted by decoder thread */
        if (ctx->stream && ctx->audio_cb && dt_state.cb_aborted) {
            if (!ctx->silent) fprintf(stderr, "\n  Streaming aborted by callback\n");
            break;
        }

        /* Build next input embedding:
         * codec_side: codec_embed(code0) + sum of CP codec_embeds(codes 1-15)
         * text_side: always tts_pad (all text was in prefill)
         */
        double t_embed_start = time_ms();
        lookup_codec_embed(ctx, code0, step_embed);
        for (int g = 0; g < 15; g++) {
            int code_g = codes[g + 1];
            if (ctx->cp_codec_emb_bf16[g] && code_g >= 0 && code_g < ctx->config.codebook_size) {
                const uint16_t *emb = ctx->cp_codec_emb_bf16[g] + (int64_t)code_g * h;
                qwen_bf16_accum_f32(step_embed, emb, h);
            }
        }

        /* Text side: always tts_pad in non-streaming mode */
        for (int j = 0; j < h; j++) step_embed[j] += tts_pad_embed[j];
        t_embed_total += time_ms() - t_embed_start;

        /* Talker step */
        if (ctx->debug && frame < 2) {
            fprintf(stderr, "  [frame %d] step_embed[:5]=[%.6f,%.6f,%.6f,%.6f,%.6f]\n",
                    frame, step_embed[0], step_embed[1], step_embed[2], step_embed[3], step_embed[4]);
            fprintf(stderr, "  [frame %d] PRE last_hidden[:5]=[%.6f,%.6f,%.6f,%.6f,%.6f]\n",
                    frame, last_hidden[0], last_hidden[1], last_hidden[2], last_hidden[3], last_hidden[4]);
        }
        /* Multi-layer emotion steer schedule: per-frame effective weight (mood-set pulse,
         * not a constant bias). 0 during prefill (set here, per generation frame). */
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
            /* Leaks-audit fix (decoder-thread USE-AFTER-FREE): the decoder thread was spawned
             * with &dt_state on THIS stack frame. Returning here without finishing+joining it
             * leaves a live thread reading freed stack memory (UAF) and leaks dt_state buffers.
             * Mirror the normal-exit cleanup (cf. the codec_frames==0 path below). */
            dt_finish(&dt_state);
            if (dt_no_overlap) decoder_thread_fn(&dt_state); else pthread_join(dt_thread, NULL);
            qwen_blas_set_threads(qwen_get_threads());   /* decoder gone: BLAS may spread again */
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

    /* Speech decoder */
    if (ctx->codec_frames == 0) {
        dt_finish(&dt_state);
        if (dt_no_overlap) decoder_thread_fn(&dt_state); else pthread_join(dt_thread, NULL);
        qwen_blas_set_threads(qwen_get_threads());
        qwen_sd_stream_free(&ctx->sd_stream); dt_free(&dt_state);
        *out_samples = NULL; *out_n_samples = 0;
        return 0;
    }

    /* Signal decoder thread that generation is done, join, collect audio */
    float *audio; int n_samples;
    double t_dec_start = time_ms();

    dt_finish(&dt_state);
    if (dt_no_overlap) decoder_thread_fn(&dt_state); else pthread_join(dt_thread, NULL);
    qwen_blas_set_threads(qwen_get_threads());
    qwen_sd_stream_free(&ctx->sd_stream);

    double dt_decode_ms = dt_state.decode_ms;
    double dt_drain_ms = time_ms() - t_dec_start;
    /* TTFA = time from generation start to first emitted audio chunk. */
    double ttfa_ms = (dt_state.first_chunk_ms > 0) ? dt_state.first_chunk_ms - t_start : -1;

    if (dt_state.audio_cb) {
        /* Streaming mode: audio was already sent via callback, return empty */
        audio = NULL;
        n_samples = dt_state.audio_len;  /* track for reporting */
        dt_free(&dt_state);
    } else {
        /* Normal mode: collect audio from decoder thread buffer */
        audio = dt_state.audio_buf;
        n_samples = dt_state.audio_len;
        dt_state.audio_buf = NULL;  /* ownership transferred */
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

/* ============================================================================
 * BATCHED long-form generation (Milestone B).
 *
 * Synthesizes `nc` independent text chunks by stepping them through the Talker
 * + Code Predictor TOGETHER (weight-stationary batched matmat → the weights are
 * read once and reused across all chunks in flight), instead of one-after-another.
 * Each chunk keeps its own KV / position / sampling state (ragged: prompts prefill
 * to different lengths, chunks hit EOS at different frames). Audio is decoded per
 * chunk (seam-free, same as Milestone A) and concatenated.
 *
 * Strategy: process chunks in groups of <= GMAX. Per group: (1) prefill each chunk
 * via the normal single-stream path (prefill_only) and capture its KV + seed hidden;
 * (2) batched ragged generation with per-chunk sampling/EOS; (3) decode + concat.
 *
 * v1 = bf16 batched step kernels (returns -2 if the model has no bf16 weights — the
 * caller falls back to sequential). int8/int4 batched-step twins come next; until
 * then a quantized model still works here via its mmap-resident bf16 weights.
 * The output is a "valid alternative kernel" (fp-order differs like int8) → validate
 * by ear/mel-corr, not bit-match. Returns 0 on success. */
int qwen_tts_generate_batch(qwen_tts_ctx_t *ctx, char **chunks, int nc,
                            float chunk_pause, float **out_samples, int *out_n_samples) {
    if (nc <= 0) { *out_samples = NULL; *out_n_samples = 0; return 0; }
    if (ctx->layers[0].wq_bf16 == NULL) return -2;   /* bf16 batched step only (v1) */
    int h = ctx->config.hidden_size;
    int kvd = ctx->config.num_kv_heads * ctx->config.head_dim;
    int num_layers = ctx->config.num_layers;
    int vocab = ctx->config.codec_vocab_size;
    int cb = ctx->config.codebook_size;
    float eps = ctx->config.rms_norm_eps;
    const int GMAX = 8;
    int GEN_CAP = ctx->max_tokens; if (GEN_CAP > 600) GEN_CAP = 600; if (GEN_CAP < 32) GEN_CAP = 32;
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

        /* ---- Phase 1: prefill each chunk, capture KV + seed hidden + lengths ---- */
        int *prompt_len = (int *)calloc(B, sizeof(int));
        int *tcl = (int *)calloc(B, sizeof(int));
        float *seed_hidden = (float *)malloc((size_t)B * h * sizeof(float));
        uint16_t **tk = (uint16_t **)calloc(B, sizeof(uint16_t *));
        uint16_t **tv = (uint16_t **)calloc(B, sizeof(uint16_t *));
        int maxpl = 0, ok = 1;
        ctx->prefill_only = 1;
        for (int b = 0; b < B && ok; b++) {
            ctx->prev_prefill_len = 0;   /* cold prefill (no cross-chunk KV reuse) */
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
        /* Diagnostic: force the batched proj to do B matvecs (bit-exact to single-
         * stream) instead of the fp-reordering matmat — isolates wiring bugs from the
         * benign matmat trajectory fork. QWEN_BATCH_FORCE_MATVEC=1. */
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

        /* ---- Phase 2: batched ragged generation ---- */
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
            /* The census denominator. It used to be incremented ONLY in the
             * single-request loop, so every calls/frame reported for the BATCHED server
             * was divided by the handful of frames the single-job clone happened to run
             * (14 against several hundred, measured 2026-08-24) - a number that reads
             * like a per-frame rate and is not one. */
            qwen_census_frame_at(2);
            /* 1. per-chunk codec head + sample code0 */
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

            /* 2. batched Code Predictor (lockstep; inactive use code0=0, ignored) */
            qwen_batch_cp_predict(ctx, bb, last_hidden, code0, cpcodes, NULL);

            /* 3. per-chunk: record frame + build next-step embedding */
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

            /* 4. batched ragged Talker step -> next last_hidden */
            if (qwen_batch_talker_step_ragged(ctx, bb, step_embed, pos, active, last_hidden) != 0) break;

            /* 5. advance each active chunk's position */
            for (int b = 0; b < B; b++) if (active[b]) pos[b]++;
        }

        /* ---- Phase 3: decode each chunk (seam-free full decode) + concat ---- */
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

/* ── Server request-batching engine ──────────────────────────────────────────
 * N independent requests (own text/speaker/lang/sampling/seed), stepped together
 * through Talker+CP weight-stationary, each producing a SEPARATE output buffer.
 * Mirrors qwen_tts_generate_batch but: (a) per-slot sampling params + RNG state
 * (reproduces single-stream bit-for-bit), (b) per-slot speaker/language applied at
 * prefill, (c) outputs are NOT concatenated — out_samples[i]/out_n_samples[i] per
 * request. Caller frees each out_samples[i]. */
int qwen_tts_generate_batch_multi(qwen_tts_ctx_t *ctx,
                                  const qwen_batch_req_t *reqs, int nc,
                                  float **out_samples, int *out_n_samples) {
    if (nc <= 0) return 0;
    if (ctx->layers[0].wq_bf16 == NULL) return -2;   /* bf16 batched step only */
    int h = ctx->config.hidden_size;
    int kvd = ctx->config.num_kv_heads * ctx->config.head_dim;
    int num_layers = ctx->config.num_layers;
    int vocab = ctx->config.codec_vocab_size;
    int cb = ctx->config.codebook_size;
    float eps = ctx->config.rms_norm_eps;
    const int GMAX = 8;
    int GEN_CAP = ctx->max_tokens; if (GEN_CAP > 600) GEN_CAP = 600; if (GEN_CAP < 32) GEN_CAP = 32;

    for (int i = 0; i < nc; i++) { out_samples[i] = NULL; out_n_samples[i] = 0; }

    for (int g0 = 0; g0 < nc; g0 += GMAX) {
        int B = nc - g0 < GMAX ? nc - g0 : GMAX;

        /* ---- Phase 1: per-request prefill (own speaker/language) ---- */
        int *prompt_len = (int *)calloc(B, sizeof(int));
        int *tcl = (int *)calloc(B, sizeof(int));
        float *seed_hidden = (float *)malloc((size_t)B * h * sizeof(float));
        uint16_t **tk = (uint16_t **)calloc(B, sizeof(uint16_t *));
        uint16_t **tv = (uint16_t **)calloc(B, sizeof(uint16_t *));
        /* per-slot sampling params + RNG state */
        float *p_temp = (float *)malloc((size_t)B * sizeof(float));
        int   *p_topk = (int *)malloc((size_t)B * sizeof(int));
        float *p_topp = (float *)malloc((size_t)B * sizeof(float));
        float *p_rep  = (float *)malloc((size_t)B * sizeof(float));
        int   *p_gw   = (int *)malloc((size_t)B * sizeof(int));
        uint32_t *rng = (uint32_t *)malloc((size_t)B * sizeof(uint32_t));
        int maxpl = 0, ok = 1;
        /* save ctx voice/sampling state to restore after (prefill mutates speaker/lang) */
        int sv_spk = ctx->speaker_id, sv_lang = ctx->language_id;
        ctx->prefill_only = 1;
        for (int b = 0; b < B && ok; b++) {
            const qwen_batch_req_t *rq = &reqs[g0 + b];
            ctx->speaker_id = rq->speaker_id;
            ctx->language_id = rq->language_id;
            ctx->prev_prefill_len = 0;   /* cold prefill per request */
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

        /* ---- Phase 2: batched ragged generation, per-slot sampling ---- */
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
            qwen_census_frame_at(2);   /* same site id: neither of these two runs under --serve */
            /* 1. per-slot codec head + sample code0 (own params + RNG state) */
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

            /* 2. batched Code Predictor */
            qwen_batch_cp_predict(ctx, bb, last_hidden, code0, cpcodes, NULL);

            /* 3. per-slot: record frame + build next-step embedding */
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

            /* 4. batched ragged Talker step */
            if (qwen_batch_talker_step_ragged(ctx, bb, step_embed, pos, active, last_hidden) != 0) break;

            /* 5. advance active positions */
            for (int b = 0; b < B; b++) if (active[b]) pos[b]++;
        }

        /* ---- Phase 3: decode each request into its OWN output buffer ---- */
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

/* ── Continuous-batching driver (S2) ─────────────────────────────────────────
 * Persistent frame-stepping loop over `B` slots. Free slots are refilled from the
 * job source every frame (continuous/vLLM-style); EOS'd requests are decoded and
 * delivered immediately, freeing their slot for the next queued request — no
 * waiting for the slowest in a static group. */
/* ================= A1: async admission-prefill pipeline =================
 * The ~1-2s single-stream prefill of a newly admitted request used to run inline in
 * the scheduler's frame loop, stalling every active slot. A1 moves it to a HELPER
 * thread with its own cloned ctx: the helper prefills, snapshots the KV + seed hidden
 * into a bounded ready-queue, and the scheduler admits pre-prefilled slots without
 * stalling. Gated on qwen_parallel_is_reentrant(): only a reentrant kernel pool
 * (macOS GCD) lets the helper's qwen_parallel run concurrently with the scheduler's;
 * on a non-reentrant pool (Linux pthread / Win32) we keep the inline prefill (correct,
 * blocking) — the pre-A1 behavior. */
typedef struct prefilled_s {
    void *tag;
    qwen_batch_req_t req;
    int ok;                 /* 0 = prefill failed / rejected */
    int pl;                 /* prefill length (frames) */
    int tcl;                /* bg_text_content_len */
    uint16_t *kv_k, *kv_v;  /* [num_layers * pl * kvd] snapshot from the clone */
    float *last_hidden;     /* [h] */
    /* TTFA2 · CLOCK_MONOTONIC, domain S. Stamped in the HELPER thread, before a slot
     * exists, which is why they live on the job and not in a per-slot array. */
    double ts_admitted;     /* next_job() returned: the job left the server's ready queue */
    double ts_prefill_start;/* immediately before qwen_tts_generate(prefill_only) */
    double ts_prefill_done; /* prefill returned                                        */
    double ts_state_ready;  /* KV snapshot + last_hidden copied: the state is publishable */
    double ts_pfq_push;     /* pfq_push() returned: resident in the prefetched-job queue */
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
/* Returns 1 if queued, 0 if the queue was shut down (caller owns p and must free it). */
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
/* block=1 waits for a job (or shutdown); block=0 returns NULL immediately if empty. */
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
    qwen_tts_ctx_t *pf_ctx;          /* cloned ctx owned by the helper */
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
        if (!a->sink->next_job(a->sink->ud, &req, &tag, 1)) break;  /* shutdown + drained */
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
                p->kv_k = p->kv_v = NULL; p->last_hidden = NULL;  /* p->ok stays 0 */
            }
        }
        p->ts_state_ready = qwen_mono_ms();
        /* Stamped BEFORE the push, and it has to be: once pfq_push returns, the driver may
         * already have popped and freed `p`. Writing to it afterwards is a use-after-free
         * and a data race. "Handed to the queue" is the correct semantic event anyway. */
        p->ts_pfq_push = qwen_mono_ms();
        if (!pfq_push(a->q, p)) { prefilled_free(p); break; }  /* shutdown mid-run */
    }
    pfq_shutdown(a->q);   /* wake a scheduler blocked on pop */
    return NULL;
}


/* ══ Decoder off the scheduler thread (PLAN 0.nonies S11) ═════════════════════
 *
 * WHY. Measured 2026-08-18 on four concurrent requests: the speech decoder is
 * 48.6% of the scheduler loop on the streaming endpoint and 29.8% on the WAV one —
 * more than the Code Predictor, and far more than the Talker. And it is the ONE
 * stage that shares nothing between slots: the Talker and the CP at least read each
 * weight once for all B, the decoder does N slots' work N times.
 *
 * The cost is not just throughput. The scheduler thread is the only thread that
 * advances every slot, so while it decodes ONE request's audio, every other user's
 * generation is stopped. That lands on THEIR time-to-first-audio and on their p95 —
 * the spike a listener notices — not on the latency of the request being decoded.
 * With TTFA as the product target, this is the largest remaining lever.
 *
 * WHAT THIS DOES. One worker thread with its OWN ctx clone (weights shared, scratch
 * private) takes the decode work off the loop. The scheduler enqueues a copy of the
 * frames and goes straight back to generating; the worker decodes and delivers.
 *
 * WHY ONE THREAD AND NOT A POOL. One thread makes per-slot ordering free (a FIFO
 * queue preserves it) and makes the per-slot streaming state single-owner, which is
 * the part that would otherwise need a lock per slot. It already removes the decode
 * from the critical path, which is the whole point; whether a second decoder thread
 * buys anything is a question for a box where the decode is compute-bound, and the
 * measurement to answer it (`make membw`, knee at 1 thread on M1) already exists.
 *
 * SLOT REUSE IS THE TRAP. A slot the scheduler frees can be re-admitted immediately
 * while the worker still holds jobs for its previous occupant — which would hand the
 * new request the old request's decoder state. dec_busy[] blocks admission until the
 * worker is done with that slot; it is not an optimisation, it is correctness.
 *
 * Off by default: QWEN_DECODER_THREAD=1 enables it. It changes WHO calls on_chunk /
 * on_done, not WHAT they are called with, so the audio is byte-identical — the gate
 * is `make test-serve-stream-batch` (corr 1.0 + exact sample count). */
typedef struct dec_job {
    int slot;
    int nframes;
    int *codes;          /* owned copy, 16 ints per frame */
    void *tag;
    int is_final;        /* last job for this slot: flush, free state, on_done */
    int stream;          /* streaming slot -> on_chunk per job; else full decode at final */
    int first;           /* first chunk of a request: jumps the queue (see dec_push) */
    struct dec_job *next;
} dec_job_t;

typedef struct {
    pthread_mutex_t m;
    pthread_cond_t  cv;
    dec_job_t *head, *tail;
    int running;
    qwen_tts_ctx_t *ctx;                 /* private clone */
    qwen_batch_sink_t *sink;
    qwen_sd_stream_state_t *sstate;      /* per-slot, owned by this thread once enabled */
    atomic_int *busy;                    /* per-slot: jobs not yet finished */
    int batch;                           /* QWEN_DECODER_BATCH: group jobs across slots */
    /* STEP 3D, DIAGNOSTIC ONLY, default 0 = today's behaviour byte for byte.
     * The INLINE decoder groups a request's FIRST chunk with other slots; this worker
     * deliberately does not (see the comment at the grouping site). Same intent, different
     * mechanism, and the difference lands on the exact chunk TTFA measures - so a
     * placement-only A/B cannot leave it uncontrolled. QWEN_DEC_FIRSTCHUNK_GROUP=1 makes the
     * worker group first chunks TOO, matching INLINE. It is never a production setting. */
    int first_group;
    int trace;                           /* QWEN_TTFA_TRACE, for the [DECODE] record */
} dec_pool_t;

/* FIFO, except that a request's FIRST chunk jumps to the head.
 *
 * WHY, and it was measured the hard way: moving the decode off the loop raised
 * throughput (+19%) and made TTFA WORSE (p50 551 -> 1047 ms). Taking the decode out of
 * the generation path put it into a shared queue, where a brand-new request's first
 * chunk — one frame, a few milliseconds of work — waited behind full chunks belonging
 * to requests that were already producing audio. Those requests have a buffer and can
 * afford to wait; the new one has nothing, and its user is staring at silence.
 *
 * Only the FIRST chunk is promoted, so a long request cannot starve the others: after
 * its first frame it is an ordinary citizen. This is the decoder-queue form of the
 * "priority to a new request's first frames" lever (PLAN T5.5 D2). */
static void dec_push(dec_pool_t *dp, dec_job_t *j) {
    pthread_mutex_lock(&dp->m);
    j->next = NULL;
    if (j->first && dp->head) {
        j->next = dp->head; dp->head = j;   /* queue non-empty -> tail stays valid */
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
        if (!j) { pthread_mutex_unlock(&dp->m); break; }   /* shut down and drained */
        dp->head = j->next; if (!dp->head) dp->tail = NULL;

        /* QWEN_DECODER_BATCH: take with it the FIRST queued job of every OTHER slot and
         * decode them in one call, so the decoder weights are read once for the group
         * instead of once per slot. Still holding the lock — the queue is being edited.
         *
         * Per-slot order is preserved because at most one job per slot is taken, and it
         * is that slot's earliest one; across slots the order does not matter, they are
         * independent streams delivered to different clients.
         *
         * A `first` job is deliberately NOT grouped: it jumped the queue precisely
         * because someone is sitting in silence waiting for their first audio, and one
         * frame decoded alone is the fastest way to give it to them. Grouping it would
         * spend the very latency the queue jump exists to buy. */
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
            /* non-streaming: the whole utterance decodes here, off the loop */
            float *aud = NULL; int an = 0;
            if (j->nframes > 0 &&
                qwen_speech_decoder_decode(dp->ctx, j->codes, j->nframes, &aud, &an) == 0
                && aud && an > 0) dp->sink->on_done(dp->sink->ud, j->tag, aud, an);
            else { free(aud); dp->sink->on_done(dp->sink->ud, j->tag, NULL, 0); }
        }
        if (dp->trace) {
            /* Decode compute, measured WHERE IT RUNS. With the thread on, the driver only
             * enqueues, so PF_END(pf_decode) collapses to ~0 and this is the only place the
             * cost exists. Without it the hypothesis "iterations shorten while decode
             * compute does not change" cannot be evaluated at all. Same clock and schema as
             * TTFA2: CLOCK_MONOTONIC, domain S. */
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
    atomic_fetch_add(&dp->busy[slot], 1);   /* cleared by the worker; blocks re-admission */
    dec_push(dp, j);
}

int qwen_tts_serve_continuous(qwen_tts_ctx_t *ctx, int B, qwen_batch_sink_t *sink) {
    if (B < 1) B = 1;
    /* The CPU batch path is bf16-only; the GPU batched path (QWEN_CUDA_BATCH) handles int8/q4
     * on the device, so the bf16 requirement is waived when it will be used. */
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
    if (ctx->layers[0].wq_bf16 == NULL && !want_cuda_batch && !want_metal_batch) return -2;   /* CPU batched step is bf16 only */
    int h = ctx->config.hidden_size;
    int kvd = ctx->config.num_kv_heads * ctx->config.head_dim;
    int num_layers = ctx->config.num_layers;
    int vocab = ctx->config.codec_vocab_size;
    int cb = ctx->config.codebook_size;
    float eps = ctx->config.rms_norm_eps;
    int GEN_CAP = ctx->max_tokens; if (GEN_CAP > 600) GEN_CAP = 600; if (GEN_CAP < 32) GEN_CAP = 32;
    const int MAXPROMPT = 512;                       /* per-slot prompt KV budget */
    int kv_max = MAXPROMPT + GEN_CAP + 4;
    int force_matvec = getenv("QWEN_BATCH_FORCE_MATVEC") ? 1 : 0;

    qwen_batch_t *bb = qwen_batch_alloc(ctx, B, kv_max);
    if (!bb) return -1;
    bb->force_matvec = force_matvec;

    /* ---- GPU batched Talker+CP (throughput path): opt-in via QWEN_CUDA_BATCH, needs the fused
     * single states (QWEN_CUDA_FUSED_TALKER --backend cuda). The GPU batched steps maintain their
     * own device KV (seeded per slot on admit); the 3 batched calls below delegate automatically. */
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

    /* Metal batched Talker+CP (throughput): opt-in QWEN_METAL_BATCH, needs the fused single Talker
     * (QWEN_METAL_FUSED_TALKER --backend metal). Both batch states derive from g_metal_talker_state. */
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
        if (!getenv("QWEN_METAL_BATCH_NOCP"))   /* diag: keep CP on CPU to isolate talker-batch vs CP-batch */
            g_metal_cp_batch_state = qwen_metal_cp_batch_init(g_metal_talker_state, B);
        metal_batch = (g_metal_talker_batch_state != NULL);
        if (metal_batch) fprintf(stderr, "[serve] Metal batched Talker+CP ENABLED (B=%d, matvec->matmat)\n", B);
        else fprintf(stderr, "[serve] Metal batched init failed — falling back to CPU batch path\n");
    } else if (getenv("QWEN_METAL_BATCH") && B > 8) {
        fprintf(stderr, "[serve] QWEN_METAL_BATCH: batch-size %d > 8 — using CPU batch path\n", B);
    }
#endif

    /* per-slot state */
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
    int *sframe = (int *)calloc(B, sizeof(int));   /* per-slot frame counter since admit */
    int *decpos = (int *)calloc(B, sizeof(int));   /* frames already handed to the decoder, per slot */
    int **prev_tok = (int **)malloc((size_t)B * sizeof(int *));
    int **chcodes = (int **)malloc((size_t)B * sizeof(int *));
    float *last_hidden = (float *)calloc((size_t)B * h, sizeof(float));
    float *logits = (float *)malloc((size_t)B * vocab * sizeof(float));  /* A2: per-slot [B][vocab] */
    float *step_embed = (float *)malloc((size_t)B * h * sizeof(float));
    int *code0 = (int *)malloc((size_t)B * sizeof(int));
    int *cpcodes = (int *)malloc((size_t)B * 15 * sizeof(int));
    /* per-slot streaming decoder state (S3): when a slot's request wants streaming
     * we decode its frames incrementally with its own state and emit via on_chunk. */
    uint8_t *want_stream = (uint8_t *)calloc(B, 1);
    qwen_sd_stream_state_t *sstate = (qwen_sd_stream_state_t *)calloc(B, sizeof(qwen_sd_stream_state_t));
    /* Amortized WAV decode: on the GPU-batched path, decode non-streaming slots INCREMENTALLY too
     * (per frame, interleaved with gen) and accumulate — avoids the serial full-decode burst when
     * a whole batch finishes together (measured ~2.4x→~3.4x at B=8, matching the streaming path).
     * Same streaming decoder the /stream endpoint uses (ear-validated). */
    /* QWEN_NO_AMORT=1 → full seam-free decode (A/B).
     * QWEN_AMORT_CPU=1 → also amortize on the CPU path. WHY IT MATTERS HERE: a
     * non-streaming request decodes its WHOLE utterance in one burst inside the scheduler
     * thread, which owns the loop — so every other slot stops while it happens. That is a
     * serial section in the middle of a concurrent server, and it lands on other users'
     * p95, not on the finishing request's own latency. Amortizing spreads the same work
     * one frame at a time, interleaved with generation, using the SAME streaming decoder
     * the /stream endpoint already uses on CPU.
     * ⚠️ It is NOT byte-identical to the one-shot decode (chunk boundaries in a
     * convolutional decoder), so it is opt-in until an ear pass says the seams are clean —
     * the project rule: the metric may choose what to listen to, never whether it is good. */
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

    /* ── decoder off the scheduler thread (S11), opt-in ── */
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
        dpool.ctx = qwen_tts_clone_for_worker(ctx);   /* weights shared, scratch private */
        if (dpool.ctx && pthread_create(&dec_thr, NULL, dec_worker_main, &dpool) == 0) {
            dec_on = 1;
            fprintf(stderr, "[serve] decoder thread ENABLED (decode leaves the frame loop)\n");
        } else {
            if (dpool.ctx) qwen_tts_free_clone(dpool.ctx);
            fprintf(stderr, "[serve] decoder thread requested but clone/thread failed — staying inline\n");
        }
    }

    /* ── Per-stage profile of the frame loop (PLAN 0.nonies: "profilare dove migliorare") ──
     *
     * A concurrency curve says WHETHER the server scales; it does not say WHERE the time
     * goes, and without that the next optimisation is a guess. The five buckets below are
     * the five things a frame does, and they answer different questions: `talker` and `cp`
     * are kernel/ISA work (that is where a real batched GEMM shows up, or fails to);
     * `head` is one vocab x hidden GEMM plus sampling; `decode` is the speech decoder,
     * which is per-slot and NOT batched; `admit` is prefill and scheduling, i.e. the part
     * that hurts everyone else's p95 when a new request arrives.
     *
     * Wall-clock per stage, single scheduler thread, so the buckets add up to the loop's
     * own time — the gap to total runtime is the loop waiting for work, which is itself
     * the signal that the box is idle rather than saturated.
     *
     * QWEN_SERVE_PROFILE=1 (or QWEN_BATCH_STATS=1, since they answer the same question
     * from two sides) prints it when the loop exits. */
    /* ── D2: priority to a new arrival's first frames (PLAN T5.5 D2), opt-in ──
     *
     * WHY. TTFA is the product's first metric, and at concurrency 4 it is ~4x the
     * concurrency-1 value (299 ms -> 1207 ms measured) for one reason: a new request's
     * first frame waits for a FULL-WIDTH batched step, and on a machine without a real
     * matrix-matrix instruction that step costs roughly B times a solo step. Nothing in
     * the pipeline is slow — the newcomer is simply paying for everyone else.
     *
     * WHAT IT DOES. For its first QWEN_TTFA_PRIORITY frames a freshly admitted slot is
     * stepped ALONE: the mask handed to the Talker and the CP contains only that slot, so
     * the B_eff==1 shortcut takes the fast single-stream path and its first audio comes
     * out at concurrency-1 speed. The streams already running are frozen for those few
     * frames — which is exactly the trade worth making, because they have audio buffered
     * and the newcomer has none. Two frames of freeze is 160 ms of THEIR buffer against
     * hundreds of ms off the newcomer's silence.
     *
     * ⚠️ It costs the others, so it is bounded and off by default: only the first N frames,
     * only one slot at a time (the newest), and only when more than one slot is active.
     * ⚠️ It also changes fp order for the frozen slots' neighbours exactly like the
     * B_eff==1 shortcut does — same caveat, same switch (QWEN_BATCH_NO_SOLO=1) if
     * bit-reproducibility under concurrency matters more than latency. */
    int ttfa_prio = 0;
    { const char *e = getenv("QWEN_TTFA_PRIORITY"); if (e) ttfa_prio = atoi(e);
      if (ttfa_prio < 0) ttfa_prio = 0; if (ttfa_prio > 8) ttfa_prio = 8; }
    uint8_t *prio_mask = ttfa_prio ? (uint8_t *)calloc(B, 1) : NULL;
    int *frozen = ttfa_prio ? (int *)calloc(B, sizeof(int)) : NULL;   /* consecutive frames held back */
    if (ttfa_prio && (!prio_mask || !frozen)) ttfa_prio = 0;
    /* Freeze cap: a running slot may be held back at most this many frames in a row.
     * WHY IT EXISTS — measured, not anticipated: with Poisson arrivals at a rate the box
     * could not drain, priority made the tail WORSE (p95 5.5 s -> 8.0 s), because every
     * new arrival froze the streams that were trying to finish, and nothing ever drained.
     * The lever assumes spare capacity; under overload it is exactly the wrong policy.
     * The cap turns that from a failure mode into a bounded cost. Default 2x the priority
     * window, overridable with QWEN_TTFA_FREEZE_CAP. */
    int freeze_cap = ttfa_prio * 2;
    { const char *e = getenv("QWEN_TTFA_FREEZE_CAP"); if (e && atoi(e) >= 0) freeze_cap = atoi(e); }
    int prio_strict = getenv("QWEN_TTFA_PRIO_STRICT") ? 1 : 0;

    int prof_on = (getenv("QWEN_SERVE_PROFILE") || getenv("QWEN_BATCH_STATS")) ? 1 : 0;

    /* Cross-slot batched speech decode. OFF by default: the per-slot path stays the
     * reference, and this one is only a win when several slots are streaming at once.
     * Mutually exclusive with the decoder THREAD for the inline path — when the thread
     * owns the decode the batching happens inside the worker instead (see dec_worker_main). */
    int dec_batch = (getenv("QWEN_DECODER_BATCH") &&
                     atoi(getenv("QWEN_DECODER_BATCH")) != 0) ? 1 : 0;
    /* S19 · TTFA decomposto PER RICHIESTA (QWEN_TTFA_TRACE=1).
     * Il banco misura il TTFA da fuori e sa dire QUANTO, non DOVE. Questi quattro
     * istanti dicono dove: quanto e' costato il prefill, quanto si e' aspettato il
     * primo frame (contesa col resto del batch), quanto la prima decodifica. Senza
     * questa riga qualunque lavoro sul TTFA e' una scommessa su quale terzo attaccare. */
    int ttfa_trace = (getenv("QWEN_TTFA_TRACE") && atoi(getenv("QWEN_TTFA_TRACE")) != 0);
    /* ── M1 · EARLY FIRST-FRAME ADMISSION. OFF by default; `1` opts in.
     * The driver admits once per iteration, at the top, immediately upstream of the codec
     * head - so a slot admitted there produces its first frame in that same iteration
     * (measured: 338/338 requests cross ZERO iteration boundaries between `installed` and
     * `frame1`). A second admission point that only ADMITS is therefore worth nothing: on a
     * serial driver with inline prefill it moves the prefill earlier and leaves the first
     * frame exactly where it was. What M1 adds is an admission that ALSO emits the frame,
     * placed at the single boundary where the new slot's first chunk can still join the
     * batched decode call of the CURRENT iteration.
     * Diagnostic, one request per iteration, and it changes nothing when unset.
     * See the design notes */
    int admit_m1 = (getenv("QWEN_ADMIT_M1") && atoi(getenv("QWEN_ADMIT_M1")) != 0);
    long long m1_admitted = 0, m1_rejected = 0, m1_cancelled = 0, m1_first_audio = 0;
    /* Why M1 did NOT fire is as much an observation as why it did: a scan that found no
     * free slot and a scan that found an empty queue are different worlds, and without
     * these two counters a low admission count is a guess. */
    long long m1_scan = 0, m1_noslot = 0, m1_nojob = 0, m1_tick = 0;
    /* TTFA2 · schema v2. The S19 names are GONE on purpose: `tt_admit` was stamped after
     * prefill and was never admit, and a diagnostic that lies about which event it timed is
     * worse than no diagnostic. Every array below is CLOCK_MONOTONIC, domain S.
     * HELPER fields are copied off the job at install; INLINE fields are stamped here.
     * The two paths do NOT share a field set and are never pooled. */
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
    /* first_audio_ready is stamped HERE, immediately before the sink is called, i.e. BEFORE
     * any socket code. If it were taken inside the delivery path, model production and
     * delivery would stay confounded - which is the whole point of this decomposition. */
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
    if (dec_batch && dec_on) dec_batch = 0;   /* the worker batches on its own side */
    long long db_calls = 0, db_slots_sum = 0; int db_max = 0;

    /* ══ S14 · POLITICHE PER STADIO — un solo --batch-size non basta ═══════════════
     *
     * WHY. Today ONE number governs Talker, Code Predictor and speech decoder together,
     * and the three stages do not have the same profile — MEASURED on this bench at c=4,
     * not assumed: CP 34-51%, decoder 30-60%, Talker 12-19%. vLLM reached the same
     * conclusion by architecture and pays for it in their config: max_num_seqs is PER
     * STAGE and tuned PER TASK (10 on Base, 1 on CustomVoice — our class).
     *
     * WHAT THIS ADDS. Three knobs per stage, all defaulting to today's behaviour, so a
     * server started without them is byte-for-byte the server of an hour ago:
     *
     *   width    QWEN_BATCH_TALKER=n / QWEN_BATCH_DECODER=n   how many slots step together
     *   threads  QWEN_THREADS_TALKER=n / QWEN_THREADS_DECODER=n   runners for that stage
     *   queue    QWEN_QUEUE_PREFILL=n                          depth of the ready queue
     *
     * The interesting cell is the ASYMMETRIC one — (talker 4, decoder 1) against its
     * opposite — because the two stages fail differently: the Talker is a weight-streaming
     * matvec that wants columns to share the read, while the decoder is a convolution
     * that pays a receptive-field context per CALL. One number cannot ask for both.
     *
     * ⚠️ FAIRNESS, not truncation. Narrowing the Talker to n slots does NOT mean "the
     * first n always run": the window rotates (round-robin cursor), so over consecutive
     * frames every active slot advances. Picking "the first n" would look identical on a
     * throughput plot and starve the tail — an error that only shows up as a p99 nobody
     * can explain. */
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
    int rr_cursor = 0;                         /* rotating start of the Talker window */
    /* The pool keeps the LARGEST budget any stage asks for and each stage narrows with a
     * soft set — off macOS resizing the pool means join+respawn, which at 12.5 frames/s
     * per stage would cost more than the policy can buy (see qwen_set_threads_soft). */
    /* ⚠️ CAVEAT DICHIARATO: g_n_threads è globale al processo. Con il THREAD decoder
     * acceso (QWEN_DECODER_THREAD=1) il worker gira in parallelo allo scheduler e vede
     * il budget che lo scheduler ha impostato in quel momento — quindi lì "per stadio"
     * diventa "l'ultimo che ha scritto", non una separazione vera. Le due leve non si
     * compongono, e il thread decoder è spento di default: la combinazione va misurata
     * prima di essere raccomandata, non dedotta. */
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

    /* S17 · THE TWO LEVERS vLLM-Omni USES ON THE SAME STAGE, made measurable here.
     *
     * Their Code2Wav does exactly what our speech decoder does, and their design doc
     * names two knobs we did not have:
     *
     *   (a) a chunk that is much LONGER than ours in steady state (codec_chunk_frames
     *       = 25 frames, ~2 s of audio, against our 8), because a non-autoregressive
     *       decoder pays its receptive-field context and its per-call overhead on EVERY
     *       call. We measured the same shape from the other end: one frame per call
     *       costs 4.2x the decoder time of a single end-of-request decode.
     *   (b) that chunk chosen BY LOAD ("initial chunk size auto-selected based on server
     *       load"): small when one request is alone and the only thing that matters is
     *       its first audio, large when the box is busy and per-call overhead is being
     *       paid N times over.
     *
     * QWEN_STREAM_DECODE_CHUNK_BUSY is (b): the steady-state chunk to use when more than
     * one slot is active. 0 = off, which keeps the single value for every load and is the
     * behaviour every number before 2026-08-19 was taken with. It does NOT touch the
     * ramp's first chunks (1, 2, 4): TTFA belongs to the newcomer, and the load lever
     * must not be able to spend it.
     *
     * ⚠️ It buys decoder time with chunk latency, and only for slots that are already
     * being heard: a longer chunk arrives later. That is invisible while audio is playing
     * and RTF < 1, and audible as a gap the moment RTF > 1 — so it is a lever to measure
     * per model and per concurrency, not a default to raise once. */
    int g_dec_chunk_busy = 0;
    { const char *e = getenv("QWEN_STREAM_DECODE_CHUNK_BUSY");
      if (e && atoi(e) > 0) g_dec_chunk_busy = atoi(e);
      if (g_dec_chunk_busy > 32) g_dec_chunk_busy = 32; }

    /* S17 · the gang policy's two thresholds, previously hard-coded at 4 and 2. They set
     * how WIDE the cross-slot decode batch gets, and the profile already prints the ratio
     * they move (slots per decoder call against mean active slots) — measured 1.33 of
     * 1.42 on the c3, so the headroom is small but the cost of finding out is one env. */
    /* S17.E · IL BUDGET BLAS DEL SERVER, che finora non esisteva.
     *
     * `qwen_set_threads()` lega OpenBLAS a `-j` all'avvio e il percorso CLI lo restringe
     * ancora in generazione (`gen_blas = nt-1`), ma questo loop non ha mai chiamato
     * `qwen_blas_set_threads()`: il server eredita `-j` e basta, per ogni fase e per ogni
     * carico. MISURATO sul c3 (0.6B, int8, -j2, match, REQS=8):
     *
     *     BLAS      c=1              c=2              c=4
     *     1         RTF 1.18 Q 0.90  3.37 / 0.59      5.44 / 0.69
     *     2         RTF 1.03 Q 0.99  2.69 / 0.74      4.19 / 0.87
     *     4         RTF 2.44 Q 0.42  2.50 / 0.80      3.74 / 0.97
     *
     * Il punto ottimo SI SPOSTA col carico: a richiesta singola vince 2, a c=4 vince 4, e
     * scegliere male costa fino al 140% (da CLI: -j2 con BLAS 4 = RTF 2.34 contro 0.99).
     * La regola che spiega la tabella e' che il totale dei thread richiesti deve restare
     * vicino ai core: a uno slot il nostro pool e' quasi fermo e la BLAS puo' allargarsi,
     * a quattro slot sono gli slot stessi a riempire la macchina.
     *
     * Default: NESSUNA chiamata, cioe' esattamente il comportamento con cui e' stato preso
     * ogni numero precedente. QWEN_SERVE_BLAS = valore con un solo slot attivo,
     * QWEN_SERVE_BLAS_BUSY = valore da due in su. OPENBLAS_NUM_THREADS nell'ambiente vince
     * comunque su entrambi (qwen_blas_set_threads esce subito se e' impostata), perche' chi
     * sta tarando a mano non va scavalcato. */
    int blas_solo = 0, blas_busy = 0, blas_now = 0;
    { const char *e = getenv("QWEN_SERVE_BLAS");      if (e && atoi(e) > 0) blas_solo = atoi(e); }
    { const char *e = getenv("QWEN_SERVE_BLAS_BUSY"); if (e && atoi(e) > 0) blas_busy = atoi(e); }
    if (blas_solo || blas_busy)
        fprintf(stderr, "[serve] BLAS budget: %d con uno slot · %d da due in su\n",
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
    double pf_m1 = 0;   /* M1's own time, handed back to the bucket that encloses it */
    /* ── QWEN_REQ_TRACE: one line per finished request, so worker load imbalance can be
     * ATTRIBUTED to the requests each worker was given rather than inferred from frame
     * counts. Joined to the harness by `seed`, which is unique per request in a level. */
    const int rq_trace = getenv("QWEN_REQ_TRACE") ? 1 : 0;
    unsigned int *rq_seed = (unsigned int *)calloc((size_t)B, sizeof(unsigned int));
    int *rq_tok = (int *)calloc((size_t)B, sizeof(int));
    double *rq_t0 = (double *)calloc((size_t)B, sizeof(double));
    double pf_t0_loop = time_ms(), pf_mark = 0;
    long long pf_frames = 0, pf_slotframes = 0, pf_stepframes = 0;
    #define PF_START() do { if (prof_on) pf_mark = time_ms(); } while (0)
    #define PF_END(acc) do { if (prof_on) (acc) += time_ms() - pf_mark; } while (0)

    /* Finalize slot b: streaming → free state + end-of-stream marker (frames were
     * already emitted via on_chunk); non-streaming → full decode + deliver. */
    /* ⚠️ Timed separately, and the time is handed back to whatever bucket encloses it.
     * First measurement (2026-08-18) attributed the FULL end-of-request decode to
     * "sampling", which then read as 30% of the loop — 14.6 ms to pick one token out of
     * 3072, which is absurd on its face and was the clue. For a non-streaming request the
     * whole utterance is decoded HERE, in one burst, inside the scheduler thread: that is
     * a real property worth its own line (it is what the amortized path exists to avoid),
     * not something to hide inside another stage. */
    /* ── The part EOS and cancellation SHARE: give the slot back. Nothing here delivers
     * audio, decodes, or reports a completion — it is only the release. Keeping it in one
     * macro is what stops cancellation from growing a second, divergent teardown. */
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
            /* the decoder thread owns delivery for this slot: hand it the tail     \
             * (streaming) or the whole utterance (WAV) and return immediately —    \
             * the loop must not stop to decode for one user while the others wait. \
             * on_done is called by the worker, in order, after the last chunk. */  \
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
        /* flush the tail of the ramped chunking: whatever was accumulated but not  \
         * yet decoded belongs to this request and must go out before it ends. */   \
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

    /* ── CANCELLED IS NOT A SUCCESSFUL EOS.
     *
     * FINALIZE_SLOT, on the streaming path, hands the undecoded remainder to the decoder
     * pool (dec_enqueue) or decodes the tail and SENDS IT with sink->on_chunk. Both are
     * exactly the work cancellation exists to avoid, and the second writes to a socket we
     * already know is dead. So this branch skips the delivery and keeps the release:
     *
     *   skipped : tail flush · dec_enqueue · tail decode · the completion [REQ] record
     *   shared  : free the stream state, free the accumulator, RELEASE_SLOT
     *   host    : on_done(tag, NULL, 0) so it can close the response and free the tag;
     *             it records the request as CANCELLED from its own per-job flag
     *
     * `frames_after_cancel_observed` is 0 by construction here: the slot stops advancing
     * in the same iteration the cancel is seen. It is still reported, because a future
     * change that made it non-zero must be visible rather than silent. */
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

    /* ══ THE THREE HALVES BOTH ADMISSION SITES SHARE ═════════════════════════════════
     *
     * Extracted VERBATIM from the top-of-loop inline path so the M1 early-first-frame point
     * cannot grow a second, divergent admission - the same reason RELEASE_SLOT exists for
     * teardown. None of them contains loop control flow: the call site owns `continue` /
     * `break` and the blocking policy, which is the only thing that legitimately differs
     * between the two sites (the top of the loop may block when fully idle; M1 never does).
     *
     *   ADMIT_PREFILL           single-stream prefill into ctx + the TTFA2 INLINE stamps.
     *                           Leaves prc_/pl_ for the caller to validate; restores the
     *                           saved speaker/language exactly as the original did.
     *   ADMIT_INSTALL           copy the KV into slot b and make the slot live.
     *   SAMPLE_SLOT             mask + sample one slot. Sets stop_=1 when the slot went
     *                           away under it (cancelled) or ended (EOS / caps).
     *   RECORD_FRAME_AND_EMBED  append the 16-code frame and build the next embedding.  */
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
            /* INLINE: the fallback path. It has NO pfq and NO helper, so those fields      \
             * stay 0 and the record declares path=INLINE - they are FORBIDDEN in the       \
             * decomposition rather than reported as zero-valued observations. */           \
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
        /* E2: has this request's client gone? Checked per slot, per frame, and ONLY for    \
         * this slot - the neighbouring rows of the same batch are untouched, exactly as    \
         * they are when one of them hits EOS. */                                           \
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

    /* ---- A1: spawn the async prefill helper (reentrant pool only) ---- */
    /* A1 e' OPT-IN e di default SPENTO su Linux/Windows: qwen_parallel_is_reentrant()
     * legge QWEN_PREFILL_HELPER. Misurato il 2026-08-18 sul c3 — accenderlo PEGGIORA il
     * TTFA p95 del ~30% a c=2/c=4, perche' con un solo job slot l'helper e il frame loop
     * si serializzano sul mutex del pool invece di sovrapporsi. La tabella e il perche'
     * stanno in qwen_tts_thread.c sopra is_reentrant(). Su macOS (GCD) resta acceso. */
    int use_helper = qwen_parallel_is_reentrant();
    qwen_tts_ctx_t *pf_ctx = use_helper ? qwen_tts_clone_for_worker(ctx) : NULL;
    prefill_q_t pfq; pthread_t pf_thr; prefill_helper_arg_t pf_arg;
    if (pf_ctx) {
        /* S14 · queue limit, per stage. This one is the ADMISSION queue: how many
         * already-prefilled requests may wait for a slot. Default 2 (one prefill hidden
         * behind generation) — raising it hides more prefill but also lets a request sit
         * finished-prefilling and un-served, which is TTFA the client is already paying
         * and nobody is measuring. Lower it to 1 to make the helper lock-step with the
         * scheduler. QWEN_QUEUE_PREFILL. */
        int cap = (B < 2) ? B : 2; if (cap < 1) cap = 1;   /* one prefill hidden behind gen */
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
    use_helper = (pf_ctx != NULL);   /* fell back to inline prefill if clone/thread failed */
    /* M1 is defined against the INLINE admission path - the production path on POSIX. On
     * the helper path the ready queue is drained by another thread into pfq, so a second
     * scheduler-thread scan would race it for jobs and the two admissions would stamp
     * different provenance on requests of the same run. Refused rather than half-supported,
     * and said out loud so a run cannot silently measure nothing. */
    if (admit_m1 && use_helper) {
        fprintf(stderr, "[serve] QWEN_ADMIT_M1 ignored: the prefill HELPER path is active "
                        "(M1 is defined against INLINE admission)\n");
        admit_m1 = 0;
    }
    if (admit_m1)
        fprintf(stderr, "[serve] M1 early first-frame admission ON (one request per iteration)\n");

    double _t2_prev_iter = 0.0;
    while (sink->running(sink->ud) || n_active > 0) {
        /* ---- admit queued jobs into free slots ---- */
        if (ttfa_trace) {
            /* THIS instant is an admission opportunity: the only point in the loop where a
             * queued request can enter a slot. Published so a request can be matched to the
             * opportunity it actually waited for, instead of the cadence being assumed.
             *
             * STEP 3C also emits it as an INDEPENDENT stream, with the production loop's own
             * phase accumulators snapshotted. Two clock domains are kept apart on purpose:
             *   iteration boundaries  qwen_mono_ms()  CLOCK_MONOTONIC   (domain S)
             *   phase accumulators    time_ms()       CLOCK_REALTIME    (domain P)
             * Consecutive differences WITHIN a domain are valid. The two are compared for
             * consistency and never subtracted from one another. PF_* is production code and
             * is left exactly as it is. */
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
            /* ⚠️ The counters CANNOT be reported at shutdown: the load harness ends the
             * run with kill -9, so no teardown print in a prefork worker ever executes.
             * That is exactly how the first two attempts at an evidence line produced an
             * empty grep and a gate that passed on nothing. Emit them LIVE instead. */
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
            /* A freed slot is NOT reusable until the decoder thread has finished the
             * previous occupant: its streaming state lives per slot, and handing it to
             * a new request would splice two voices together. Correctness, not tuning. */
            if (dec_on && atomic_load(&dec_busy[b]) != 0) continue;
            if (use_helper) {
                /* A1: admit a slot that the helper thread already prefilled — the
                 * scheduler never blocks on prefill, only on the (cheap) KV copy. */
                if (!sink->running(sink->ud) && n_active > 0) break;  /* draining: stop admitting */
                int block = (n_active == 0);   /* fully idle → wait for the first ready job */
                /* The blocking pop is NOT admission work — it is the loop waiting for a
                 * customer. Charging it to "admit" made an idle server look like it was
                 * spending 60% of its time on scheduling (first live run, 2026-08-18).
                 * Time it separately and hand the same amount back to the admit mark. */
                double pf_w0 = prof_on ? time_ms() : 0;
                double _t_pop_pre = ttfa_trace ? qwen_mono_ms() : 0.0;
                prefilled_t *p = pfq_pop(&pfq, block);
                double _t_pop = ttfa_trace ? qwen_mono_ms() : 0.0;
                (void)_t_pop_pre;
                if (prof_on) { double d = time_ms() - pf_w0; pf_wait += d; pf_mark += d; }
                if (!p) break;                 /* nothing ready (or shutdown) this frame */
                if (!p->ok) { sink->on_done(sink->ud, p->tag, NULL, 0); prefilled_free(p); continue; }
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
                    t2_helper[b]   = 1;                 /* PRODUCTION path */
                    t2_seed[b]     = p->req.seed;
                    t2_admitted[b] = p->ts_admitted;    t2_pf_start[b]  = p->ts_prefill_start;
                    t2_pf_done[b]  = p->ts_prefill_done; t2_state_rdy[b] = p->ts_state_ready;
                    t2_pfq_push[b] = p->ts_pfq_push;    t2_pfq_pop[b]   = _t_pop;
                    t2_frame1[b] = 0; t2_audio1[b] = 0; t2_emitted[b] = 0;
                    /* scheduler state AT the pop, not afterwards */
                    t2_batch_at_inst[b] = n_active;     t2_qdepth_at_pop[b] = pfq.count;
                    t2_adm_seq[b] = atomic_load_explicit(&g_admit_seq, memory_order_relaxed);
                }
                want_stream[b] = (p->req.want_stream && sink->on_chunk) ? 1 : 0;
                if (want_stream[b] || amort) qwen_sd_stream_init(&sstate[b]);
                acc_n[b] = 0;
                tag[b] = p->tag; active[b] = 1; n_active++;
                if (ttfa_trace) t2_installed[b] = qwen_mono_ms();   /* KV in the slot */
                prefilled_free(p);
                continue;
            }
            /* ---- inline fallback (non-reentrant pool): prefill blocks the batch ---- */
            if (!sink->running(sink->ud)) break;
            int block = (n_active == 0);   /* block only when fully idle (no spin) */
            qwen_batch_req_t req;
            void *t = NULL;
            double pf_w1 = prof_on ? time_ms() : 0;
            int pf_got = sink->next_job(sink->ud, &req, &t, block);
            if (prof_on) { double d = time_ms() - pf_w1; pf_wait += d; pf_mark += d; }
            if (!pf_got) {
                if (block) break;          /* shutdown while idle */
                continue;                  /* nothing queued right now */
            }
            /* prefill this request (single-stream prefill_only), capture KV into slot b */
            int prc = 0, pl = 0;
            ADMIT_PREFILL(b, req, prc, pl);
            if (prc != 0 || pl <= 0 || pl > MAXPROMPT) {
                sink->on_done(sink->ud, t, NULL, 0);   /* reject (prefill fail / too long) */
                continue;
            }
            ADMIT_INSTALL(b, req, t, pl);
        }

        if (n_active == 0) {
            if (!sink->running(sink->ud)) break;
            continue;
        }

        PF_END(pf_admit);

        /* S17.E · la BLAS segue il carico. Si chiama solo quando il valore CAMBIA: la
         * openblas_set_num_threads() non e' gratis e n_active si muove poco. */
        if (blas_solo || blas_busy) {
            int want = (n_active > 1) ? blas_busy : blas_solo;
            if (want <= 0) want = qwen_get_threads();
            if (want != blas_now) { qwen_blas_set_threads(want); blas_now = want; }
        }

        /* Which slots move this frame: normally all the active ones. With D2 on, a slot
         * still inside its priority window takes the frame for itself. */
        uint8_t *step_active = active;
        if (ttfa_prio && n_active > 1) {
            int newest = -1, starving = 0, established = 0;
            for (int b = 0; b < B; b++) {
                if (!active[b]) continue;
                if (frozen[b] >= freeze_cap) starving = 1;      /* someone waited long enough */
                if (sframe[b] >= ttfa_prio) established = 1;    /* has audio buffered already */
                if (sframe[b] < ttfa_prio && (newest < 0 || sframe[b] < sframe[newest])) newest = b;
            }
            /* QWEN_TTFA_PRIO_STRICT=1 borrows time ONLY from an established stream —
             * one that already has audio buffered. The argument for it: if every active
             * slot is also new, priority merely serializes their starts rather than
             * trading one user's buffer for another's silence.
             *
             * It is NOT the default, because the argument lost to the measurement. With
             * four simultaneous arrivals, serializing the starts improved BOTH p50 (652 ->
             * 420 ms) and p95 (1197 -> 1032) — stepping one slot alone is so much cheaper
             * than stepping four together on a machine with no matrix-matrix instruction
             * that the early starters get their first frames far sooner while the last one
             * barely loses. The tidy-sounding rule would have thrown that away. Kept
             * available, because on silicon where the batched step is genuinely near-free
             * the arithmetic may well flip back. */
            if (newest >= 0 && !starving && (established || !prio_strict)) {
                memset(prio_mask, 0, (size_t)B);
                prio_mask[newest] = 1;
                step_active = prio_mask;
            }
        }
        /* S14 · Talker+CP batch width. Applied AFTER D2, and only if D2 did not already
         * narrow the step: the priority mask is one slot, which is narrower than any
         * width this can ask for, and overriding it would silently disable the TTFA
         * lever the moment someone set a width. The window ROTATES so no slot starves. */
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
        if (st_tt) qwen_set_threads_soft(st_tt);   /* Talker+CP thread budget */

        /* ---- one frame: batched codec head (A2), then per-slot mask + sample ----
         * Was B separate matvec_bf16 calls (codec_head re-read B times). qwen_batch_proj
         * reads the weight ONCE for all B under the matmat path; under force_matvec it is
         * B matvecs, bit-identical to the old per-slot path (so --batch-test stays exact).
         * Inactive slots compute a (finite, ignored) head — same as the batched CP/step. */
        /* The codec head is a vocab x hidden GEMM (3072 x 2048 on the 1.7B), so it earns
         * the same treatment as the layer projections: compute only the active columns,
         * through the map the Talker step just packed. Inactive slots keep stale logits
         * and are skipped by the loop below, exactly as before. */
        /* act_idx/B_eff descrivono I SLOT DI QUESTO FRAME, e vanno scritti QUI — prima
         * di usarli — non ereditati dal passo del Talker in fondo al ciclo.
         *
         * Perche' (orecchio, 2026-08-20): una richiesta ammessa a riga ~3262 diventa attiva
         * DOPO l'ultimo passo del Talker, quindi al suo primo frame non e' ancora in
         * act_idx e la sua colonna di logit non viene calcolata. Il campionatore legge
         * valori stantii, il primo code0 esce sbagliato, e un solo token sbagliato in testa
         * manda il modello a farfugliare qualche secondo prima di riallinearsi al testo:
         * ~4 s di suoni che non sono neither the target language nor English, poi la frase giusta fino in
         * fondo. Colpisce solo quando lo slot nuovo NON e' quello che act_idx gia' nomina —
         * cioe' dopo una sovrapposizione — ed e' per questo che sfuggiva alla media del LID
         * (67% invece di 0) mentre all'orecchio era ovvia. */
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

        /* ---- batched CP over all slots (inactive use code0=0) ---- */
        PF_START();
        qwen_batch_cp_predict(ctx, bb, last_hidden, code0, cpcodes, step_active);
        PF_END(pf_cp);
        /* The census denominator for the BATCHED SERVER. The other two generation loops
         * are instrumented too and both measured ZERO frames under `--serve --batch-size 4`
         * (2026-08-24): this while() is the one that runs, and dividing a call count by a
         * loop that never executed produces a per-frame rate that is not one. */
        qwen_census_frame_at(1);
        if (prof_on) {
            pf_frames++; pf_slotframes += n_active;
            /* Slots ADMITTED vs slots actually STEPPED this frame. They differ as soon as
             * a narrowing policy is on (D2 priority, S14 talker width), and keeping only
             * the first would make a width of 1 look like a batch of 4 that scales badly. */
            for (int b = 0; b < B; b++) if (step_active[b]) pf_stepframes++;
        }

        /* ---- record frame + build next embedding ---- */
        PF_START();
        if (st_td) qwen_set_threads_soft(st_td);   /* S14: decoder thread budget */
        for (int b = 0; b < B; b++) {
            if (!step_active[b]) {
                memset(step_embed + (size_t)b * h, 0, (size_t)h * sizeof(float));
                continue;
            }
            RECORD_FRAME_AND_EMBED(b);

            /* S3: decode incrementally — streaming → emit via on_chunk; amortized WAV
             * (GPU batch) → accumulate into acc_aud[b], delivered whole at FINALIZE.
             *
             * ⚠️ IN CHUNK A RAMPA, non un frame per volta. MEASURED 2026-08-18: decoding one
             * frame per call costs 31.1 s of decoder time for the same audio that a single
             * end-of-request decode does in 7.5 s — 4.2x, because a convolutional decoder pays
             * its receptive-field context and its per-call overhead on EVERY call. That is why
             * the streaming endpoint showed the decoder at 58.5% of the scheduler loop while
             * the WAV endpoint showed 29.8%.
             *
             * The ramp keeps what the small first chunk bought (TTFA: first audio after ONE
             * frame, ~80 ms of audio, not 800) and stops paying for it afterwards: 1, 2, 4,
             * then QWEN_STREAM_DECODE_CHUNK frames (default 8). The trade is explicit — a
             * later chunk arrives up to 8 frames after its first token, which is invisible to
             * a listener who is already hearing audio, but each extra frame in a chunk is
             * decoder work NOT repeated.
             * ⚠️ Chunk boundaries are where a convolutional decoder can crack: this changes
             * WHERE the seams are, so it wants the D0 listening protocol before the default
             * moves further. QWEN_STREAM_DECODE_CHUNK=1 restores the old behaviour exactly. */
            if (want_stream[b] || amort) {
                int pending = chframes[b] - decpos[b];
                int target;
                if      (decpos[b] == 0) target = 1;   /* first audio out as early as possible */
                else if (decpos[b] < 4)  target = 2;
                else if (decpos[b] < 12) target = 4;
                else if (g_dec_chunk_busy && n_active > 1) target = g_dec_chunk_busy;
                else                     target = g_stream_dec_chunk;
                if (ttfa_trace && t2_frame1[b] == 0.0 && pending > 0) t2_frame1[b] = qwen_mono_ms();
                if (dec_batch) {
                    /* Do NOT decode here. Record what this slot has pending and let the
                     * pass after this loop decide the batch — the whole point is that the
                     * decoder reads its weights once for every slot, and it cannot do that
                     * from inside a per-slot loop. See the gang policy below. */
                    db_pending[b] = pending;
                    db_target[b] = target;
                    continue;
                }
                if (pending < target) continue;        /* accumulate; flushed at FINALIZE */
                if (dec_on && want_stream[b]) {        /* hand it to the decoder thread */
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
                    else {   /* amortized WAV: append */
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
        /* ══ M1 · EARLY FIRST-FRAME ADMISSION ═══════════════════════════════════════
         * QWEN_ADMIT_M1=1. Diagnostic, OFF by default, at most ONE request per iteration.
         *
         * WHY HERE AND NOWHERE ELSE. This is the only instant in an iteration where a
         * newly admitted slot can still emit audio in the SAME iteration: head, sampler
         * and CP for the established slots are done, db_pending/db_target are populated,
         * and the cross-slot decode call three lines below has not run yet. A slot admitted
         * here runs its own narrow head+sample+CP, records one frame, and fires as a `must`
         * in that decode call - exactly as its first frame fires today, one iteration
         * sooner. Admitting AFTER the decode call (or anywhere else) buys nothing: the
         * first chunk would wait for the next iteration regardless, and the only thing
         * that moved would be the queue-time label. See §2-3 of
         * the design notes
         *
         * WHAT IT DELIBERATELY DOES NOT TOUCH. Decoder placement, decoder batching and the
         * gang policy, batch capacity, the prefill policy, cancellation, and the order the
         * ready queue is served in. The new slot contributes one extra ragged item to a
         * decode call that was going to happen anyway.
         *
         * THE COST, measured before this was written: the narrow pass is 9.53-9.63 ms p95
         * and is 98.8 % Code Predictor (the design notes.
         * Its time is charged to pf_m1 and handed back to the enclosing pf_decode mark, so
         * the decode line keeps meaning "decode".
         *
         * ⚠️ The first frame is computed at B_eff=1 rather than batched with the others,
         * which qwen_tts_talker.c:2430-2449 documents as a different, valid fp-order
         * variant. Output is therefore NOT bit-identical to the same request admitted at
         * the top of the loop - which is why byte identity is not a gate for this flag. */
        if (admit_m1 && sink->running(sink->ud)) {
            m1_scan++;
            int m1_free_slot = 0;
            for (int b = 0; b < B; b++) {
                if (active[b]) continue;
                /* same slot-reuse rule as the top-of-loop site: a freed slot is not
                 * reusable until the decoder thread released it. Correctness, not tuning. */
                if (dec_on && atomic_load(&dec_busy[b]) != 0) continue;
                m1_free_slot = 1;
                qwen_batch_req_t req;
                void *jt = NULL;
                double _m_t0 = ttfa_trace ? qwen_mono_ms() : 0;
                /* NEVER blocks: this is the middle of an iteration with work in flight. */
                if (!sink->next_job(sink->ud, &req, &jt, 0)) { m1_nojob++; break; }
                double _mf0 = prof_on ? time_ms() : 0;
                int prc = 0, pl = 0;
                double _m_pf0 = ttfa_trace ? qwen_mono_ms() : 0;
                ADMIT_PREFILL(b, req, prc, pl);
                double _m_pf1 = ttfa_trace ? qwen_mono_ms() : 0;
                if (prc != 0 || pl <= 0 || pl > MAXPROMPT) {
                    sink->on_done(sink->ud, jt, NULL, 0);   /* reject, exactly as above */
                    m1_rejected++;
                    if (prof_on) { double _d = time_ms() - _mf0; pf_m1 += _d; pf_mark += _d; }
                    break;
                }
                int _m_nbefore = n_active;
                ADMIT_INSTALL(b, req, jt, pl);

                /* ---- the narrow first frame: this slot alone, at B_eff = 1 ---- */
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
                    /* the client was already gone, or the slot ended on its first frame.
                     * SAMPLE_SLOT has done the full teardown; nothing of this iteration
                     * is left holding the slot. */
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

                /* Hand the frame to the decode call below. decpos[b]==0 so its own ramp
                 * asks for exactly 1 frame, which is `must`: no gang is opened and no
                 * established slot is dragged into a 1-frame call. */
                if (want_stream[b] || amort) {
                    if (ttfa_trace && t2_frame1[b] == 0.0) t2_frame1[b] = qwen_mono_ms();
                    if (dec_batch) { db_pending[b] = 1; db_target[b] = 1; }
                    /* !dec_batch: the frame simply stays pending and is decoded by the
                     * next iteration's per-slot path, one iteration earlier than today. */
                }
                /* the Talker step at the bottom of this iteration must advance it too.
                 * When step_active aliases `active` the slot is already enrolled; under
                 * the D2 priority or S14 width mask it is a separate buffer and has to be
                 * said explicitly, or the slot would sit with its KV installed and never
                 * move. qwen_batch_talker_step_ragged repacks act_idx from step_active,
                 * so the narrow pack above does not leak into it. */
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
                break;                       /* one request per iteration, by design */
            }
            if (!m1_free_slot) m1_noslot++;
        }

        /* ---- CROSS-SLOT BATCHED DECODE (QWEN_DECODER_BATCH=1) --------------------
         * THE PHASE PROBLEM, and why a batch is not formed by "slots that happen to be
         * ready". Every active slot produces exactly one frame per iteration, so with
         * the ramp every slot ends up on the same chunk size (8) but on a DIFFERENT
         * iteration — its phase is set by when the request was admitted. Batching only
         * the slots whose triggers coincide would therefore give B=1 almost always, and
         * the change would measure as "no effect" for the wrong reason.
         *
         * GANG POLICY. A slot's own ramp still decides when a decode MUST happen, so no
         * slot ever waits longer than it does today and the first chunk is still one
         * frame delivered immediately. What is added is: once a slot with a grown-up
         * chunk (target >= 4) fires, every other slot holding >= 2 frames joins the same
         * call. Nothing is padded — the batched decoder takes ragged lengths — so a
         * joiner contributes exactly the frames it has and no wasted frame is computed.
         *
         * The policy is also self-aligning, which is the part that makes it worth having:
         * after a gang round every joiner is back at 0 pending, so they reach their next
         * target on the SAME iteration, and from then on they fire together without
         * anything having to coordinate them.
         *
         * A new arrival's tiny chunks (target 1 and 2) deliberately do NOT open a gang:
         * they must go out immediately for TTFA, and dragging the established slots into
         * a 1-frame call would hand them the per-call cost the ramp exists to avoid. */
        if (dec_batch) {
            int leader = 0;
            for (int b = 0; b < B; b++)
                if (db_pending[b] >= db_target[b] && db_target[b] >= g_gang_lead) { leader = 1; break; }
            int nit = 0;
            /* S14 · decoder batch width. Two passes, because the cap must not be spent on
             * joiners: a slot whose OWN ramp says "decode now" has a deadline, a joiner is
             * only there to share the weight read. Filling the batch with joiners and
             * pushing a must-fire slot to the next frame would turn a width knob into an
             * added latency, which is precisely the confusion this task exists to remove.
             * A deferred joiner loses nothing — its frames stay pending and it fires next
             * iteration with more of them. */
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
                        else {   /* amortized WAV: append */
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

        /* Did the slot M1 admitted a few lines ago actually get its first audio out in
         * THIS iteration? That is the link the whole hypothesis turns on, so it is
         * recorded as an observation rather than inferred from the TTFA moving. */
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

        if (st_td) qwen_set_threads_soft(st_tt ? st_tt : st_th_base);   /* back to the Talker budget */

        PF_END(pf_decode);

        /* ---- batched ragged Talker step over active slots ---- */
        PF_START();
        int pf_rc = qwen_batch_talker_step_ragged(ctx, bb, step_embed, pos, step_active, last_hidden);
        PF_END(pf_talker);
        if (pf_rc != 0) {
            /* fatal step error: fail all active slots */
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
    /* M1 · positive evidence, printed whether or not the flag was on and whether or not
     * profiling is on. A run that reports `admitted 0` did NOT exercise the path and its
     * comparison is NOT EVALUATED - never a PASS. Printed for the OFF arm too, so the
     * absence of the path in A is an observation rather than an assumption. The two
     * not-fired counters are here for the same reason: a low admission count must be
     * explainable as "the queue was empty" or "no slot was free", not guessed at.
     * ⚠️ NOT inside the prof_on block: that block needs QWEN_SERVE_PROFILE, and the line
     * that decides whether a cell is evaluated at all cannot depend on a profiling flag. */
    fprintf(stderr, "[serve] M1 early first frame: %s · admitted %lld · "
                    "first-audio-same-iter %lld · rejected %lld · stopped-on-first-frame %lld"
                    " · scans %lld (no free slot %lld · empty queue %lld) · %.0f ms\n",
            admit_m1 ? "ON" : "OFF", m1_admitted, m1_first_audio, m1_rejected,
            m1_cancelled, m1_scan, m1_noslot, m1_nojob, pf_m1);
    fflush(stderr);

    /* Stop the decoder thread FIRST and drain it: it still owns per-slot streaming
     * state and the client sockets of any request whose last chunks are queued.
     * Freeing that state before it drains would be a use-after-free, and closing the
     * loop before it delivers would truncate audio a listener is hearing. */
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

    /* A1: stop the prefill helper and reclaim any not-yet-admitted prefilled jobs.
     * The helper unblocks from sink->next_job when the server shuts jq down (serve_batched
     * calls jq_shutdown before joining this scheduler thread), or from a blocked push via
     * pfq_shutdown here; either way it exits and this join returns. Leftover ready jobs get
     * a failure on_done so their clients aren't left hanging. */
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
