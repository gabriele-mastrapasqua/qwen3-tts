/*
 * qwen_tts.c - Qwen3-TTS Pure C Inference Engine
 * Main pipeline: text → Talker → Code Predictor → Speech Decoder → audio
 */

#include "qwen_tts.h"
#include "qwen_tts_voice_clone.h"
#include "qwen_tts_kernels.h"
#include "qwen_tts_safetensors.h"
#include "qwen_tts_tokenizer.h"
#include "qwen_tts_audio.h"
#include "qwen_tts_batch.h"
#include "qwen_tts_thread.h"   /* qwen_parallel_is_reentrant() (A1 prefill helper gate) */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <sys/mman.h>
#include <unistd.h>
#include <stdatomic.h>   /* leaks-audit #6: _Atomic flag for the cross-thread cb_aborted */

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
extern int qwen_talker_step(qwen_tts_ctx_t *ctx, float *embed, float *hidden_out);
extern int qwen_cp_predict(qwen_tts_ctx_t *ctx, float *talker_hidden, int code0, int *out_codes);
#ifdef CP_MICROBENCH
extern void qwen_cp_microbench_report(int frames);
#endif
extern int qwen_speech_decoder_decode(qwen_tts_ctx_t *ctx, const int *codes, int n_frames, float **audio_out, int *n_samples);
extern int qwen_speech_decoder_decode_streaming(qwen_tts_ctx_t *ctx, const int *new_codes, int new_frames, float **audio_out, int *n_samples);
extern int qwen_speech_decoder_decode_streaming_st(qwen_tts_ctx_t *ctx, qwen_sd_stream_state_t *st, const int *new_codes, int new_frames, float **audio_out, int *n_samples);
extern void qwen_sd_stream_init(qwen_sd_stream_state_t *st);
extern void qwen_sd_stream_free(qwen_sd_stream_state_t *st);
extern int qwen_tts_sample(float *logits, int vocab_size, float temp, int top_k, float top_p, float rep_penalty, int *prev_tokens, int n_prev);
extern void qwen_set_seed(uint32_t seed);
extern uint32_t qwen_get_seed(void);

/* Embed a single text token: text_embedding → text_projection(SiLU) → out[hidden]
 * Computes the full projection (bf16 lookup + fc1 SiLU + fc2). */
void embed_one_text_token_compute(qwen_tts_ctx_t *ctx, int tid, float *out) {
    int th = ctx->config.text_hidden_size, h = ctx->config.hidden_size;
    float *text_emb = ctx->emb_tmp1;
    float *fc1_out = ctx->emb_tmp2;
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
qwen_tts_ctx_t *qwen_tts_load_ex(const char *model_dir, int silent, int use_int8, int use_int4) {
    qwen_tts_ctx_t *ctx = (qwen_tts_ctx_t *)calloc(1, sizeof(qwen_tts_ctx_t)); if (!ctx) return NULL;
    strncpy(ctx->model_dir, model_dir, sizeof(ctx->model_dir) - 1);
    ctx->temperature = 0.9f; ctx->top_k = 50; ctx->top_p = 1.0f; ctx->rep_penalty = 1.05f;
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

    /* Load safetensors using qwen-asr loader (mmap-based, working) */
    ctx->safetensors = multi_safetensors_open(ctx->model_dir);
    if (!ctx->safetensors) {
        fprintf(stderr, "Error: Failed to load model from %s\n", ctx->model_dir);
        free(ctx); return NULL;
    }
    /* Speech tokenizer is in a separate subdirectory */
    char speech_dir[4096];
    snprintf(speech_dir, sizeof(speech_dir), "%s/speech_tokenizer", ctx->model_dir);
    ctx->speech_safetensors = multi_safetensors_open(speech_dir);
    if (!ctx->speech_safetensors) {
        fprintf(stderr, "Error: Failed to load speech tokenizer from %s\n", speech_dir);
        multi_safetensors_close(ctx->safetensors);
        free(ctx); return NULL;
    }

    if (!ctx->silent) fprintf(stderr, "Threads: %d\n", qwen_get_threads());

    double t0 = time_ms();
    if (qwen_talker_load(ctx) != 0 || qwen_cp_load(ctx) != 0 || qwen_speech_decoder_load(ctx) != 0) {
        multi_safetensors_close(ctx->safetensors);
        multi_safetensors_close(ctx->speech_safetensors);
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
    multi_safetensors_close(ctx->safetensors);
    multi_safetensors_close(ctx->speech_safetensors);
    free(ctx->instruct);
    free(ctx->speaker_embedding);
    free(ctx->ref_audio_path);
    free(ctx->ref_text);
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
static void lookup_codec_embed(qwen_tts_ctx_t *ctx, int token_id, float *out) {
    int h = ctx->config.hidden_size;
    if (token_id < 0 || token_id >= ctx->config.codec_vocab_size) { memset(out, 0, h * sizeof(float)); return; }
    const uint16_t *emb = ctx->codec_embedding_bf16 + (int64_t)token_id * h;
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
        /* ICL mode: also tokenize reference text */
        if (ctx->voice_clone && !ctx->xvector_only && ctx->ref_text) {
            ref_text_tokens = qwen_tokenizer_encode(tok, ctx->ref_text, &ref_text_token_len);
            if (!ctx->silent && ref_text_tokens)
                fprintf(stderr, "Ref text: \"%s\" (%d tokens)\n", ctx->ref_text, ref_text_token_len);
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

    /* Check for cached ref_codes from .qvoice file (skipped in --graft mode) */
    if (ctx->voice_clone && !ctx->graft_mode && ctx->cached_ref_codes && ctx->cached_ref_n_frames > 0
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
    /* Fused GPU Talker: the CPU batched prefill populated the host bf16 KV; upload it to the
     * device cache once so the fused decode steps (which read the device KV) see the prompt. */
    {
        extern void *g_cuda_talker_state;
        extern void qwen_cuda_talker_upload_kv(void *, qwen_tts_ctx_t *, int);
        if (g_cuda_talker_state && !(ctx->ml_steer && ctx->ml_steer_w_eff != 0.0f))
            qwen_cuda_talker_upload_kv(g_cuda_talker_state, ctx, ctx->kv_len);
    }
#endif
#ifdef QWEN_HAVE_METAL
    {
        extern void *g_metal_talker_state;
        extern void qwen_metal_talker_upload_kv(void *, qwen_tts_ctx_t *, int);
        if (g_metal_talker_state && !(ctx->ml_steer && ctx->ml_steer_w_eff != 0.0f))
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
    if (!dt_no_overlap)
        pthread_create(&dt_thread, NULL, decoder_thread_fn, &dt_state);

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

        /* EOS boosting: after 2x expected duration, gently boost EOS logit.
         * Heuristic: ~3 frames per BPE token. Start boosting at 2x expected,
         * linearly increasing by 0.5 per frame beyond that threshold. */
        {
            int expected_frames = text_content_len * 3;
            int boost_start = expected_frames * 2;
            if (expected_frames > 0 && frame > boost_start) {
                float boost = 0.5f * (frame - boost_start);
                if (boost > 10.0f) boost = 10.0f;  /* cap at +10 */
                ctx->logits[QWEN_TTS_CODEC_EOS] += boost;
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
        qwen_sd_stream_free(&ctx->sd_stream); dt_free(&dt_state);
        *out_samples = NULL; *out_n_samples = 0;
        return 0;
    }

    /* Signal decoder thread that generation is done, join, collect audio */
    float *audio; int n_samples;
    double t_dec_start = time_ms();

    dt_finish(&dt_state);
    if (dt_no_overlap) decoder_thread_fn(&dt_state); else pthread_join(dt_thread, NULL);
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
            /* 1. per-chunk codec head + sample code0 */
            for (int b = 0; b < B; b++) {
                if (!active[b]) { code0[b] = 0; continue; }
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
            qwen_batch_cp_predict(ctx, bb, last_hidden, code0, cpcodes);

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
            /* 1. per-slot codec head + sample code0 (own params + RNG state) */
            for (int b = 0; b < B; b++) {
                if (!active[b]) { code0[b] = 0; continue; }
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
            qwen_batch_cp_predict(ctx, bb, last_hidden, code0, cpcodes);

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
        pf->speaker_id = req.speaker_id; pf->language_id = req.language_id;
        pf->prev_prefill_len = 0; pf->prefill_only = 1;
        int prc = qwen_tts_generate(pf, req.text, NULL, NULL);
        pf->prefill_only = 0;
        int pl = pf->kv_len;
        prefilled_t *p = (prefilled_t *)calloc(1, sizeof(prefilled_t));
        if (!p) { a->sink->on_done(a->sink->ud, tag, NULL, 0); continue; }
        p->tag = tag; p->req = req;
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
        if (!pfq_push(a->q, p)) { prefilled_free(p); break; }  /* shutdown mid-run */
    }
    pfq_shutdown(a->q);   /* wake a scheduler blocked on pop */
    return NULL;
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
    int amort = cuda_batch && !getenv("QWEN_NO_AMORT");   /* QWEN_NO_AMORT=1 → full seam-free decode (A/B) */
    float **acc_aud = (float **)calloc(B, sizeof(float *));
    int *acc_n = (int *)calloc(B, sizeof(int));
    int *acc_cap = (int *)calloc(B, sizeof(int));
    for (int b = 0; b < B; b++) {
        prev_tok[b] = (int *)malloc((size_t)GEN_CAP * sizeof(int));
        chcodes[b] = (int *)malloc((size_t)GEN_CAP * 16 * sizeof(int));
    }
    const float *tts_pad = ctx->cached_tts_pad_embed;
    int n_active = 0;

    /* Finalize slot b: streaming → free state + end-of-stream marker (frames were
     * already emitted via on_chunk); non-streaming → full decode + deliver. */
    #define FINALIZE_SLOT(b) do {                                                  \
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
        active[b] = 0; tag[b] = NULL; n_active--;                                  \
    } while (0)

    /* ---- A1: spawn the async prefill helper (reentrant pool only) ---- */
    int use_helper = qwen_parallel_is_reentrant();
    qwen_tts_ctx_t *pf_ctx = use_helper ? qwen_tts_clone_for_worker(ctx) : NULL;
    prefill_q_t pfq; pthread_t pf_thr; prefill_helper_arg_t pf_arg;
    if (pf_ctx) {
        int cap = (B < 2) ? B : 2; if (cap < 1) cap = 1;   /* one prefill hidden behind gen */
        pfq_init(&pfq, cap);
        pf_arg.pf_ctx = pf_ctx; pf_arg.sink = sink; pf_arg.q = &pfq;
        pf_arg.num_layers = num_layers; pf_arg.kvd = kvd; pf_arg.h = h;
        pf_arg.MAXPROMPT = MAXPROMPT; pf_arg.eps = eps;
        if (pthread_create(&pf_thr, NULL, prefill_helper_main, &pf_arg) != 0) {
            qwen_tts_free_clone(pf_ctx); pf_ctx = NULL; pfq_destroy(&pfq);
        }
    }
    use_helper = (pf_ctx != NULL);   /* fell back to inline prefill if clone/thread failed */

    while (sink->running(sink->ud) || n_active > 0) {
        /* ---- admit queued jobs into free slots ---- */
        for (int b = 0; b < B; b++) {
            if (active[b]) continue;
            if (use_helper) {
                /* A1: admit a slot that the helper thread already prefilled — the
                 * scheduler never blocks on prefill, only on the (cheap) KV copy. */
                if (!sink->running(sink->ud) && n_active > 0) break;  /* draining: stop admitting */
                int block = (n_active == 0);   /* fully idle → wait for the first ready job */
                prefilled_t *p = pfq_pop(&pfq, block);
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
                nprev[b] = 0; chframes[b] = 0; sframe[b] = 0;
                want_stream[b] = (p->req.want_stream && sink->on_chunk) ? 1 : 0;
                if (want_stream[b] || amort) qwen_sd_stream_init(&sstate[b]);
                acc_n[b] = 0;
                tag[b] = p->tag; active[b] = 1; n_active++;
                prefilled_free(p);
                continue;
            }
            /* ---- inline fallback (non-reentrant pool): prefill blocks the batch ---- */
            if (!sink->running(sink->ud)) break;
            int block = (n_active == 0);   /* block only when fully idle (no spin) */
            qwen_batch_req_t req;
            void *t = NULL;
            if (!sink->next_job(sink->ud, &req, &t, block)) {
                if (block) break;          /* shutdown while idle */
                continue;                  /* nothing queued right now */
            }
            /* prefill this request (single-stream prefill_only), capture KV into slot b */
            int sv_spk = ctx->speaker_id, sv_lang = ctx->language_id;
            ctx->speaker_id = req.speaker_id; ctx->language_id = req.language_id;
            ctx->prev_prefill_len = 0; ctx->prefill_only = 1;
            int prc = qwen_tts_generate(ctx, req.text, NULL, NULL);
            ctx->prefill_only = 0;
            ctx->speaker_id = sv_spk; ctx->language_id = sv_lang;
            int pl = ctx->kv_len;
            if (prc != 0 || pl <= 0 || pl > MAXPROMPT) {
                sink->on_done(sink->ud, t, NULL, 0);   /* reject (prefill fail / too long) */
                continue;
            }
            for (int L = 0; L < num_layers; L++) {
                size_t dst = ((size_t)b * num_layers + L) * kv_max * kvd;
                memcpy(bb->kv_k + dst, ctx->kv_cache_k + (size_t)L * ctx->kv_max * kvd, (size_t)pl * kvd * sizeof(uint16_t));
                memcpy(bb->kv_v + dst, ctx->kv_cache_v + (size_t)L * ctx->kv_max * kvd, (size_t)pl * kvd * sizeof(uint16_t));
            }
            qwen_rms_norm(last_hidden + (size_t)b * h, ctx->dec_x, ctx->talker_norm, 1, h, eps);
            tcl[b] = ctx->bg_text_content_len;
            pos[b] = pl;
#ifdef QWEN_HAVE_CUDA
            if (cuda_batch) qwen_cuda_talker_batch_upload_slot(g_cuda_talker_batch_state, b, bb->kv_k, bb->kv_v, kv_max, pos[b]);
#endif
#ifdef QWEN_HAVE_METAL
            if (metal_batch) qwen_metal_talker_batch_upload_slot(g_metal_talker_batch_state, b, bb->kv_k, bb->kv_v, kv_max, pos[b]);
#endif
            p_temp[b] = req.temperature; p_topk[b] = req.top_k; p_topp[b] = req.top_p;
            p_rep[b] = req.rep_penalty; p_gw[b] = req.greedy_warmup; rng[b] = req.seed;
            nprev[b] = 0; chframes[b] = 0; sframe[b] = 0;
            want_stream[b] = (req.want_stream && sink->on_chunk) ? 1 : 0;
            if (want_stream[b]) qwen_sd_stream_init(&sstate[b]);
            tag[b] = t; active[b] = 1; n_active++;
        }

        if (n_active == 0) {
            if (!sink->running(sink->ud)) break;
            continue;
        }

        /* ---- one frame: batched codec head (A2), then per-slot mask + sample ----
         * Was B separate matvec_bf16 calls (codec_head re-read B times). qwen_batch_proj
         * reads the weight ONCE for all B under the matmat path; under force_matvec it is
         * B matvecs, bit-identical to the old per-slot path (so --batch-test stays exact).
         * Inactive slots compute a (finite, ignored) head — same as the batched CP/step. */
        qwen_batch_proj(logits, ctx->codec_head_bf16, last_hidden, vocab, h, h,
                        B, force_matvec, bb->Xt, bb->Yt);
        for (int b = 0; b < B; b++) {
            if (!active[b]) { code0[b] = 0; continue; }
            float *lg = logits + (size_t)b * vocab;
            for (int t = 0; t < vocab; t++) { if (lg[t] > 100.0f) lg[t] = 100.0f; if (lg[t] < -100.0f) lg[t] = -100.0f; }
            for (int t = 2048; t < vocab; t++) if (t != QWEN_TTS_CODEC_EOS) lg[t] = -1e30f;
            int sf = sframe[b];
            if (sf < 2) lg[QWEN_TTS_CODEC_EOS] = -1e30f;
            int ef = tcl[b] * 3, bs = ef * 2;
            if (ef > 0 && sf > bs) { float bo = 0.5f * (sf - bs); if (bo > 10.0f) bo = 10.0f; lg[QWEN_TTS_CODEC_EOS] += bo; }
            float ft = p_temp[b]; int ftk = p_topk[b];
            if (p_gw[b] > 0 && sf < p_gw[b]) { ft = 0.0f; ftk = 1; }
            qwen_set_seed(rng[b]);
            int c0 = qwen_tts_sample(lg, vocab, ft, ftk, p_topp[b], p_rep[b], prev_tok[b], nprev[b]);
            rng[b] = qwen_get_seed();
            if (c0 == QWEN_TTS_CODEC_EOS || chframes[b] >= GEN_CAP || pos[b] >= kv_max - 1) {
                FINALIZE_SLOT(b); code0[b] = 0; continue;
            }
            code0[b] = c0; prev_tok[b][nprev[b]++] = c0;
        }
        if (n_active == 0) continue;

        /* ---- batched CP over all slots (inactive use code0=0) ---- */
        qwen_batch_cp_predict(ctx, bb, last_hidden, code0, cpcodes);

        /* ---- record frame + build next embedding ---- */
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

            /* S3: decode this frame incrementally — streaming → emit via on_chunk; amortized WAV
             * (GPU batch) → accumulate into acc_aud[b], delivered whole at FINALIZE. Interleaving
             * the decode with gen avoids the serial full-decode burst when the batch finishes. */
            if (want_stream[b] || amort) {
                float *aud = NULL; int an = 0;
                if (qwen_speech_decoder_decode_streaming_st(ctx, &sstate[b],
                        chcodes[b] + (size_t)(chframes[b] - 1) * 16, 1, &aud, &an) == 0
                    && aud && an > 0) {
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

        /* ---- batched ragged Talker step over active slots ---- */
        if (qwen_batch_talker_step_ragged(ctx, bb, step_embed, pos, active, last_hidden) != 0) {
            /* fatal step error: fail all active slots */
            for (int b = 0; b < B; b++) if (active[b]) {
                if (want_stream[b]) { qwen_sd_stream_free(&sstate[b]); want_stream[b] = 0; }
                sink->on_done(sink->ud, tag[b], NULL, 0); active[b] = 0; tag[b] = NULL; n_active--;
            }
            break;
        }
        for (int b = 0; b < B; b++) if (active[b]) { pos[b]++; sframe[b]++; }
    }

    #undef FINALIZE_SLOT

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
    free(nprev); free(chframes); free(sframe); free(prev_tok); free(chcodes);
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
