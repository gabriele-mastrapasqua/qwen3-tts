/*
 * qwen_tts_server.c - Minimal HTTP server for Qwen3-TTS
 *
 * Single-threaded, no external dependencies. Handles one request at a time.
 * Endpoints:
 *   POST /v1/tts          — generate speech, return WAV
 *   POST /v1/tts/stream   — generate speech, return chunked raw PCM
 *   GET  /v1/speakers     — list available speakers
 *   GET  /v1/health       — health check
 *   POST /v1/audio/speech — OpenAI-compatible TTS endpoint
 */

#include "qwen_tts_server.h"
#include "qwen_tts.h"
#include "qwen_tts_thread.h"   /* qwen_parallel_is_reentrant() */
#include "qwen_tts_emotion.h"  /* qwen_tts_apply_emotion() — server --emotion support */
#include "qwen_tts_compose.h"  /* inline per-sentence emotion markup ([joy]…[sad]…) */
#include "qwen_tts_audio.h"    /* qwen_audio_apply_gain / qwen_audio_time_stretch */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <signal.h>
#include <errno.h>
#include <sys/time.h>
#include <pthread.h>

/* Max accepted request text length (chars). Guards against a single huge body
 * blowing up the tokenizer / generation time / memory. ~1500 words of TTS is
 * already far beyond any reasonable single request. */
#define MAX_TTS_TEXT 8192

/* Serializes synthesis on the shared ctx. The accept loop is single-threaded today
 * (one request at a time), so this is UNCONTENDED — it's the correctness foundation
 * for when the server gains per-connection concurrency (continuous batching). With a
 * shared mutable ctx, any future threading MUST hold this around parse+generate. */
static pthread_mutex_t g_synth_lock = PTHREAD_MUTEX_INITIALIZER;

/* When 1, synthesis is serialized under g_synth_lock even across worker threads.
 * Set at startup iff (n_workers >= 2 AND the kernel thread pool is NOT reentrant):
 * on the pthread/Win32 backend two workers calling qwen_parallel at once would
 * corrupt the single global job slot, so we must serialize. On GCD it stays 0
 * (dispatch_apply is concurrent-safe) → true request-level parallelism. With a
 * single worker (or inline mode) there is no concurrency, so it also stays 0. */
static int g_serialize_synth = 0;

static inline float clampf(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

/* ── Simple JSON helpers ─────────────────────────────────────────────── */

/* Extract a string value for a key from JSON. Returns malloc'd string or NULL. */
static char *json_extract_string(const char *json, const char *key) {
    char pattern[256];
    snprintf(pattern, sizeof(pattern), "\"%s\"", key);
    const char *p = strstr(json, pattern);
    if (!p) return NULL;
    p += strlen(pattern);
    while (*p == ' ' || *p == '\t' || *p == '\n' || *p == ':') p++;
    if (*p != '"') return NULL;
    p++;
    const char *end = p;
    while (*end && *end != '"') {
        /* Leaks-audit #4 MED: only skip the escaped char if it isn't the NUL
         * terminator. A body ending in a trailing backslash (\\\0) used to step
         * over the NUL and walk out-of-bounds heap -> crash / huge len. */
        if (*end == '\\' && end[1]) end++;
        end++;
    }
    int len = (int)(end - p);
    char *result = (char *)malloc(len + 1);
    if (!result) return NULL;
    memcpy(result, p, len);
    result[len] = '\0';
    return result;
}

/* Extract a numeric value for a key. Returns default if not found. */
static double json_extract_number(const char *json, const char *key, double def) {
    char pattern[256];
    snprintf(pattern, sizeof(pattern), "\"%s\"", key);
    const char *p = strstr(json, pattern);
    if (!p) return def;
    p += strlen(pattern);
    while (*p == ' ' || *p == '\t' || *p == '\n' || *p == ':') p++;
    if (*p == '"') return def; /* it's a string, not a number */
    return atof(p);
}

/* ── HTTP helpers ────────────────────────────────────────────────────── */

/* Read full HTTP request into buffer. Returns total bytes read, or -1. */
static int read_request(int fd, char *buf, int buf_size) {
    int total = 0;
    int content_length = -1;
    int header_end = -1;

    while (total < buf_size - 1) {
        int n = (int)read(fd, buf + total, buf_size - 1 - total);
        if (n <= 0) break;
        total += n;
        buf[total] = '\0';

        /* Look for end of headers */
        if (header_end < 0) {
            char *hend = strstr(buf, "\r\n\r\n");
            if (hend) {
                header_end = (int)(hend - buf) + 4;
                /* Parse Content-Length */
                char *cl = strcasestr(buf, "Content-Length:");
                if (cl) content_length = atoi(cl + 15);
                else content_length = 0;
                /* leaks-audit #10: clamp the untrusted Content-Length. The buffer is fixed
                 * (buf_size), so a body that can't fit is capped rather than waited on (limits a
                 * slowloris-style hold); a negative/garbage value is treated as 0. A full fix would
                 * also set a socket read timeout (SO_RCVTIMEO) at accept time — follow-up. */
                if (content_length < 0) content_length = 0;
                if (content_length > buf_size - 1) content_length = buf_size - 1;
            }
        }

        /* Check if we have the full body */
        if (header_end >= 0) {
            int body_received = total - header_end;
            if (body_received >= content_length) break;
        }
    }
    return total;
}

/* Send HTTP response with headers + body */
static void send_response(int fd, int status, const char *content_type,
                          const void *body, int body_len) {
    const char *status_text = (status == 200) ? "OK" :
                              (status == 400) ? "Bad Request" :
                              (status == 404) ? "Not Found" :
                              (status == 405) ? "Method Not Allowed" :
                              "Internal Server Error";
    char header[512];
    int hlen = snprintf(header, sizeof(header),
        "HTTP/1.1 %d %s\r\n"
        "Content-Type: %s\r\n"
        "Content-Length: %d\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "Connection: close\r\n"
        "\r\n",
        status, status_text, content_type, body_len);
    write(fd, header, hlen);
    if (body && body_len > 0) write(fd, body, body_len);
}

static void send_json(int fd, int status, const char *json) {
    send_response(fd, status, "application/json", json, (int)strlen(json));
}

static void send_error(int fd, int status, const char *msg) {
    char json[512];
    snprintf(json, sizeof(json), "{\"error\":\"%s\"}", msg);
    send_json(fd, status, json);
}

/* ── Streaming response (chunked transfer encoding) ──────────────── */

typedef struct {
    int fd;
    int total_samples;
    float volume;   /* per-chunk gain (emotion/volume); 1.0 = no-op */
} stream_http_state_t;

static void send_chunked_header(int fd) {
    const char *header =
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: audio/pcm\r\n"
        "X-Sample-Rate: 24000\r\n"
        "X-Sample-Format: s16le\r\n"
        "X-Channels: 1\r\n"
        "Transfer-Encoding: chunked\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "Connection: close\r\n"
        "\r\n";
    write(fd, header, strlen(header));
}

static int stream_http_callback(const float *samples, int n_samples, void *userdata) {
    stream_http_state_t *st = (stream_http_state_t *)userdata;
    float g = st->volume;
    /* Convert float to s16le (applying the emotion/volume gain per chunk) */
    int16_t *pcm = (int16_t *)malloc(n_samples * sizeof(int16_t));
    for (int i = 0; i < n_samples; i++) {
        float s = samples[i] * g;
        if (s < -1.0f) s = -1.0f;
        if (s > 1.0f) s = 1.0f;
        pcm[i] = (int16_t)(s * 32767);
    }
    /* Send as HTTP chunk: hex_size\r\n + data + \r\n */
    int data_len = n_samples * 2;
    char chunk_header[32];
    int chlen = snprintf(chunk_header, sizeof(chunk_header), "%x\r\n", data_len);
    write(st->fd, chunk_header, chlen);
    write(st->fd, pcm, data_len);
    write(st->fd, "\r\n", 2);
    free(pcm);
    st->total_samples += n_samples;
    return 0;
}

static void send_chunked_end(int fd) {
    write(fd, "0\r\n\r\n", 5);
}

/* Adapter: feed a composer span's PCM through the HTTP chunk encoder (applies per-chunk gain). */
static void compose_stream_emit(const float *pcm, int n, void *user) {
    stream_http_callback(pcm, n, user);
}

/* ── WAV in-memory builder ───────────────────────────────────────────── */

static void *build_wav(const float *samples, int n_samples, int *out_size) {
    int sample_rate = QWEN_TTS_SAMPLE_RATE;
    int bits = 16, channels = 1;
    int data_size = n_samples * channels * (bits / 8);
    int file_size = 36 + data_size;
    int total = 44 + data_size;
    char *wav = (char *)malloc(total);
    char *p = wav;

    /* RIFF header */
    memcpy(p, "RIFF", 4); p += 4;
    memcpy(p, &file_size, 4); p += 4;
    memcpy(p, "WAVEfmt ", 8); p += 8;
    int fmt_size = 16; memcpy(p, &fmt_size, 4); p += 4;
    short audio_fmt = 1; memcpy(p, &audio_fmt, 2); p += 2;
    short ch = channels; memcpy(p, &ch, 2); p += 2;
    memcpy(p, &sample_rate, 4); p += 4;
    int byte_rate = sample_rate * channels * (bits / 8);
    memcpy(p, &byte_rate, 4); p += 4;
    short block_align = channels * (bits / 8);
    memcpy(p, &block_align, 2); p += 2;
    short bps = bits; memcpy(p, &bps, 2); p += 2;
    memcpy(p, "data", 4); p += 4;
    memcpy(p, &data_size, 4); p += 4;

    /* PCM samples */
    int16_t *pcm = (int16_t *)p;
    for (int i = 0; i < n_samples; i++) {
        float s = samples[i];
        if (s < -1.0f) s = -1.0f;
        if (s > 1.0f) s = 1.0f;
        pcm[i] = (int16_t)(s * 32767);
    }

    *out_size = total;
    return wav;
}

/* ── Request handlers ────────────────────────────────────────────────── */

static void handle_health(int fd) {
    send_json(fd, 200, "{\"status\":\"ok\"}");
}

static void handle_speakers(int fd) {
    const char *json =
        "{\"speakers\":["
        "{\"name\":\"ryan\",\"language\":\"English\",\"gender\":\"male\"},"
        "{\"name\":\"aiden\",\"language\":\"English\",\"gender\":\"male\"},"
        "{\"name\":\"vivian\",\"language\":\"Chinese\",\"gender\":\"female\"},"
        "{\"name\":\"serena\",\"language\":\"Chinese\",\"gender\":\"female\"},"
        "{\"name\":\"uncle_fu\",\"language\":\"Chinese\",\"gender\":\"male\"},"
        "{\"name\":\"dylan\",\"language\":\"Chinese\",\"gender\":\"male\"},"
        "{\"name\":\"eric\",\"language\":\"Chinese\",\"gender\":\"male\"},"
        "{\"name\":\"ono_anna\",\"language\":\"Japanese\",\"gender\":\"female\"},"
        "{\"name\":\"sohee\",\"language\":\"Korean\",\"gender\":\"female\"}"
        "]}";
    send_json(fd, 200, json);
}

/* Reset per-request context to clean defaults (prevents state leaking between requests) */
static void reset_request_state(qwen_tts_ctx_t *ctx) {
    /* Reset speaker and language.
     * If a .qvoice is loaded (voice_clone mode), preserve the language
     * from the voice metadata — the user shouldn't need to specify it. */
    if (!ctx->voice_clone) {
        ctx->speaker_id = 3061;   /* ryan */
        ctx->language_id = 2050;  /* English */
    }
    /* In voice_clone mode, speaker_id and language_id stay as set by .qvoice */

    /* Reset sampling params to defaults */
    ctx->temperature = 0.5f;
    ctx->top_k = 50;
    ctx->top_p = 1.0f;
    ctx->rep_penalty = 1.05f;

    /* Reset transient flags */
    ctx->voice_design = 0;
    free(ctx->instruct);
    ctx->instruct = NULL;

    /* Clear any emotion steering from a prior request (must not leak between requests).
     * qwen_tts_apply_emotion also clears it, but a request with NO emotion needs this. */
    if (ctx->cp_steer_vec) { free(ctx->cp_steer_vec); ctx->cp_steer_vec = NULL; ctx->cp_steer_dim = 0; }
    ctx->cp_roughness = 0.0f;

    /* Fresh seed per request (time-based) */
    struct timeval tv;
    gettimeofday(&tv, NULL);
    ctx->seed = (uint32_t)(tv.tv_sec ^ tv.tv_usec);
}

/* Apply TTS params from JSON body to context. Returns text (malloc'd) or NULL on error.
 * out_volume/out_rate receive the effective DSP gain/tempo (from --emotion recipe or
 * explicit "volume"/"rate"), to be applied to the rendered audio by the caller. */
static char *parse_tts_request(qwen_tts_ctx_t *ctx, const char *body,
                               float *out_volume, float *out_rate) {
    /* Start from clean defaults — prevents state leaking between requests */
    reset_request_state(ctx);

    char *text = json_extract_string(body, "text");
    if (!text) {
        /* Try OpenAI-compatible "input" field */
        text = json_extract_string(body, "input");
    }
    if (!text || text[0] == '\0') {
        free(text);
        return NULL;
    }
    if (strlen(text) > MAX_TTS_TEXT) {   /* reject oversized input (DoS / OOM guard) */
        free(text);
        return NULL;
    }

    char *speaker = json_extract_string(body, "speaker");
    if (!speaker) speaker = json_extract_string(body, "voice");
    if (speaker) {
        int sid = qwen_tts_speaker_id(speaker);
        if (sid >= 0) ctx->speaker_id = sid;
        free(speaker);
    }

    char *language = json_extract_string(body, "language");  /* kept for the emotion resolver below */
    if (language) {
        int lid = qwen_tts_language_id(language);
        if (lid >= 0) ctx->language_id = lid;
    }

    /* Instruct (1.7B only) */
    free(ctx->instruct);
    ctx->instruct = json_extract_string(body, "instruct");

    /* Voice design mode */
    char *vd = json_extract_string(body, "voice_design");
    if (vd) {
        if (strcmp(vd, "true") == 0 || strcmp(vd, "1") == 0) ctx->voice_design = 1;
        free(vd);
    }

    /* Sampling params (override defaults only if provided), clamped to sane ranges so
     * a bad client value can't crash sampling or produce garbage (e.g. negative top_k,
     * top_p outside [0,1], runaway temperature). */
    /* Cap temperature at 2.0: above that (with top_p=1/top_k=0) sampling is so flat the
     * model may never emit EOS and runs to max_frames — a degenerate near-runaway. 2.0 is
     * already far past the 0.5 default. */
    ctx->temperature = clampf((float)json_extract_number(body, "temperature", ctx->temperature), 0.0f, 2.0f);
    ctx->top_k       = (int)json_extract_number(body, "top_k", ctx->top_k);
    if (ctx->top_k < 0) ctx->top_k = 0;
    if (ctx->top_k > ctx->config.codec_vocab_size) ctx->top_k = ctx->config.codec_vocab_size;
    ctx->top_p       = clampf((float)json_extract_number(body, "top_p", ctx->top_p), 0.0f, 1.0f);
    ctx->rep_penalty = clampf((float)json_extract_number(body, "rep_penalty", ctx->rep_penalty), 0.5f, 2.0f);

    /* Seed (optional: 0 or negative = keep time-based from reset) */
    int seed = (int)json_extract_number(body, "seed", -1);
    if (seed >= 0) ctx->seed = (uint32_t)seed;

    /* Inline paralinguistics ([laugh]/[sigh] -> validated onomatopoeia in the active voice,
     * one generation), mirroring the CLI. Runs on ALL requests so the substituted text flows
     * into either the plain or the per-sentence-compose path. Pin the validated seed + bump
     * temperature only when the client didn't set them. */
    {
        int vivian_id = qwen_tts_speaker_id("vivian");
        int para_voice = (vivian_id >= 0 && ctx->speaker_id == vivian_id) ? 1 : 0;
        int did = 0, para_seed = 7; float para_temp = 1.1f;
        char *sub = qwen_compose_para_substitute(text, para_voice, &did, &para_seed, &para_temp);
        if (sub && did) {
            free(text); text = sub;
            if (seed < 0) ctx->seed = (uint32_t)para_seed;
            if (strstr(body, "\"temperature\"") == NULL) ctx->temperature = para_temp;
        } else {
            free(sub);
        }
    }

    /* Emotion (CLI --emotion parity): sets the CP steering vector for
     * (emotion, language) on ctx (applied during generation for BOTH the full
     * and streaming paths) + returns the effective volume/rate DSP. Explicit
     * "volume"/"rate" in the body override the recipe value. Best-effort: an
     * unknown emotion degrades to volume/rate only (or no-op). */
    float eff_vol = 1.0f, eff_rate = 1.0f;
    int vol_present  = strstr(body, "\"volume\"") != NULL;
    int rate_present = strstr(body, "\"rate\"") != NULL;
    float req_vol  = (float)json_extract_number(body, "volume", 1.0);
    float req_rate = (float)json_extract_number(body, "rate", 1.0);
    char *emotion = json_extract_string(body, "emotion");
    if (emotion && emotion[0]) {
        qwen_tts_apply_emotion(ctx, emotion, NULL, language,
                               1.0f, 0, 0.0f, 0, req_vol, vol_present, req_rate, rate_present,
                               &eff_vol, &eff_rate, 0);
    } else {
        eff_vol  = vol_present  ? req_vol  : 1.0f;
        eff_rate = rate_present ? req_rate : 1.0f;
    }
    free(emotion);
    free(language);
    if (out_volume) *out_volume = eff_vol;
    if (out_rate)   *out_rate   = eff_rate;

    return text;
}

static double server_time_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

static void handle_tts(qwen_tts_ctx_t *ctx, int fd, const char *body) {
    float volume = 1.0f, rate = 1.0f;
    char *text = parse_tts_request(ctx, body, &volume, &rate);
    if (!text) {
        send_error(fd, 400, "missing, empty, or oversized 'text' (max 8192 chars)");
        return;
    }
    if (ctx->voice_design && ctx->config.hidden_size < 2048) {
        send_error(fd, 400, "voice_design requires the 1.7B VoiceDesign model");
        free(text);
        return;
    }

    fprintf(stderr, "[HTTP] TTS: \"%s\" (speaker=%d, lang=%d, seed=%u)\n",
            text, ctx->speaker_id, ctx->language_id, ctx->seed);
    double t0 = server_time_ms();

    /* Disable streaming for this path — full decode */
    ctx->stream = 0;
    ctx->audio_cb = NULL;

    float *audio = NULL;
    int n_samples = 0;

    /* Per-sentence dynamic emotion: if the text carries inline markup ([joy]…[sad]…, [pause],
     * fillers), synthesize span-by-span with each span's own emotion and concatenate — same
     * mechanism as the CLI --compose / auto-detected --text. Plain text takes the fast path. */
    if (qwen_compose_has_markup(text)) {
        char *language = json_extract_string(body, "language");
        qwen_cspan_t *spans = NULL; int nspans = 0;
        if (qwen_compose_parse(text, &spans, &nspans) != 0 || nspans == 0) {
            send_error(fd, 500, "markup parse failed");
            free(language); free(text); return;
        }
        fprintf(stderr, "[HTTP] inline markup -> per-sentence compose (%d spans)\n", nspans);
        int rc = qwen_compose_render_buffer(ctx, spans, nspans, language, 0.12f, &audio, &n_samples, 1);
        qwen_compose_free_spans(spans, nspans);
        free(language);
        if (rc != 0 || !audio || n_samples == 0) {
            send_error(fd, 500, "generation failed");
            free(audio); free(text); return;
        }
    } else if (qwen_tts_generate(ctx, text, &audio, &n_samples) != 0 || !audio || n_samples == 0) {
        send_error(fd, 500, "generation failed");
        free(text);
        free(audio);
        return;
    }

    /* Emotion/volume/rate DSP: gain then pitch-preserving tempo (matches CLI --emotion). */
    if (volume != 1.0f) qwen_audio_apply_gain(audio, n_samples, volume);
    if (rate != 1.0f) {
        float *stretched = NULL; int stretched_n = 0;
        if (qwen_audio_time_stretch(audio, n_samples, rate, QWEN_TTS_SAMPLE_RATE, &stretched, &stretched_n) == 0) {
            free(audio); audio = stretched; n_samples = stretched_n;
        }
    }

    /* Build WAV in memory and send */
    int wav_size = 0;
    void *wav = build_wav(audio, n_samples, &wav_size);
    free(audio);
    free(text);

    send_response(fd, 200, "audio/wav", wav, wav_size);
    free(wav);

    double elapsed = server_time_ms() - t0;
    float audio_secs = (float)n_samples / QWEN_TTS_SAMPLE_RATE;
    fprintf(stderr, "[HTTP] Sent %d bytes WAV (%.2fs audio) in %.1fs (RTF %.2f)\n",
            wav_size, audio_secs, elapsed / 1000.0, (elapsed / 1000.0) / audio_secs);
}

static void handle_tts_stream(qwen_tts_ctx_t *ctx, int fd, const char *body) {
    float volume = 1.0f, rate = 1.0f;
    char *text = parse_tts_request(ctx, body, &volume, &rate);
    (void)rate;  /* pitch-preserving tempo isn't applied on the streaming path (needs full buffer) */
    if (!text) {
        send_error(fd, 400, "missing, empty, or oversized 'text' (max 8192 chars)");
        return;
    }
    if (ctx->voice_design && ctx->config.hidden_size < 2048) {
        send_error(fd, 400, "voice_design requires the 1.7B VoiceDesign model");
        free(text);
        return;
    }

    fprintf(stderr, "[HTTP] TTS stream: \"%s\" (speaker=%d, lang=%d, seed=%u)\n",
            text, ctx->speaker_id, ctx->language_id, ctx->seed);
    double t0 = server_time_ms();

    /* Set up streaming (emotion steering is already set on ctx; volume applied per chunk) */
    stream_http_state_t state = { .fd = fd, .total_samples = 0, .volume = volume };
    ctx->stream = 1;
    ctx->stream_chunk_frames = 10;
    qwen_tts_set_audio_callback(ctx, stream_http_callback, &state);

    /* Send chunked response header */
    send_chunked_header(fd);

    /* Per-sentence dynamic emotion: inline markup streams span-by-span — each sentence is
     * synthesized with its own emotion and flushed as it completes (low time-to-first-audio),
     * so a single request can switch mood paragraph by paragraph. Plain text streams as one take. */
    if (qwen_compose_has_markup(text)) {
        char *language = json_extract_string(body, "language");
        qwen_cspan_t *spans = NULL; int nspans = 0;
        if (qwen_compose_parse(text, &spans, &nspans) == 0 && nspans > 0) {
            fprintf(stderr, "[HTTP] inline markup -> per-sentence compose stream (%d spans)\n", nspans);
            ctx->stream = 0;        /* each span is a full internal decode; we emit its buffer per span */
            ctx->audio_cb = NULL;
            qwen_compose_render_stream(ctx, spans, nspans, language, 0.12f,
                                       compose_stream_emit, &state, 1);
        }
        qwen_compose_free_spans(spans, nspans);
        free(language);
        free(text);
    } else {
    float *audio = NULL;
    int n_samples = 0;
    qwen_tts_generate(ctx, text, &audio, &n_samples);
    free(audio);
    free(text);
    }

    /* Terminate chunked encoding */
    send_chunked_end(fd);

    /* Clean up streaming state */
    ctx->stream = 0;
    ctx->audio_cb = NULL;

    double elapsed = server_time_ms() - t0;
    float audio_secs = (float)state.total_samples / QWEN_TTS_SAMPLE_RATE;
    fprintf(stderr, "[HTTP] Streamed %d samples (%.2fs audio) in %.1fs (RTF %.2f)\n",
            state.total_samples, audio_secs, elapsed / 1000.0, (elapsed / 1000.0) / audio_secs);
}

/* ── Per-connection handling ─────────────────────────────────────────────
 *
 * Reads + routes + responds on one connection, then closes it. Runs either on
 * the acceptor thread (single-worker inline mode) or on a worker thread (pool
 * mode). It only ever touches its OWN `ctx` — in pool mode each worker has an
 * independent clone, so there is no shared mutable state EXCEPT the kernel
 * thread pool: when that backend is not concurrent-safe, g_serialize_synth is
 * set and the synthesis dispatch is wrapped in g_synth_lock. */
static void handle_connection(qwen_tts_ctx_t *ctx, int client_fd,
                              struct sockaddr_in client_addr) {
    char *buf = (char *)malloc(1024 * 1024); /* 1MB max request */
    if (!buf) { close(client_fd); return; }
    int total = read_request(client_fd, buf, 1024 * 1024);
    if (total <= 0) { free(buf); close(client_fd); return; }

    /* Parse method and path */
    char method[16] = {0}, path[256] = {0};
    sscanf(buf, "%15s %255s", method, path);

    /* Find body (after \r\n\r\n) */
    const char *body = strstr(buf, "\r\n\r\n");
    if (body) body += 4;
    else body = "";

    /* inet_ntop into a local buffer (inet_ntoa's static buffer is not
     * thread-safe across concurrent workers). */
    char client_ip[INET_ADDRSTRLEN] = {0};
    inet_ntop(AF_INET, &client_addr.sin_addr, client_ip, sizeof(client_ip));
    fprintf(stderr, "[HTTP] %s %s %s from %s\n", method, path,
            (strcmp(method, "POST") == 0 && body[0]) ? "(has body)" : "", client_ip);

    /* Handle CORS preflight */
    if (strcmp(method, "OPTIONS") == 0) {
        const char *cors =
            "HTTP/1.1 204 No Content\r\n"
            "Access-Control-Allow-Origin: *\r\n"
            "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
            "Access-Control-Allow-Headers: Content-Type\r\n"
            "Connection: close\r\n\r\n";
        write(client_fd, cors, strlen(cors));
    }
    else if (strcmp(path, "/v1/health") == 0 && strcmp(method, "GET") == 0) {
        handle_health(client_fd);
    }
    else if (strcmp(path, "/v1/speakers") == 0 && strcmp(method, "GET") == 0) {
        handle_speakers(client_fd);
    }
    /* Synthesis: per-worker ctx makes these independent; only serialize when the
     * kernel thread pool itself is not concurrent-safe (g_serialize_synth). */
    else if (strcmp(path, "/v1/tts") == 0 && strcmp(method, "POST") == 0) {
        if (g_serialize_synth) pthread_mutex_lock(&g_synth_lock);
        handle_tts(ctx, client_fd, body);
        if (g_serialize_synth) pthread_mutex_unlock(&g_synth_lock);
    }
    else if (strcmp(path, "/v1/tts/stream") == 0 && strcmp(method, "POST") == 0) {
        if (g_serialize_synth) pthread_mutex_lock(&g_synth_lock);
        handle_tts_stream(ctx, client_fd, body);
        if (g_serialize_synth) pthread_mutex_unlock(&g_synth_lock);
    }
    else if (strcmp(path, "/v1/audio/speech") == 0 && strcmp(method, "POST") == 0) {
        if (g_serialize_synth) pthread_mutex_lock(&g_synth_lock);
        handle_tts(ctx, client_fd, body);   /* OpenAI-compatible: same as /v1/tts */
        if (g_serialize_synth) pthread_mutex_unlock(&g_synth_lock);
    }
    else {
        send_error(client_fd, 404, "not found");
    }

    free(buf);
    close(client_fd);
}

/* ── Connection queue (acceptor → worker pool) ───────────────────────────── */

#define CONN_QUEUE_CAP 256

typedef struct {
    int fds[CONN_QUEUE_CAP];
    int head, tail, count;
    pthread_mutex_t mtx;
    pthread_cond_t not_empty;
    pthread_cond_t not_full;
    int shutdown;            /* 1 = no more work; workers drain then exit */
} conn_queue_t;

static void cq_init(conn_queue_t *q) {
    q->head = q->tail = q->count = 0;
    q->shutdown = 0;
    pthread_mutex_init(&q->mtx, NULL);
    pthread_cond_init(&q->not_empty, NULL);
    pthread_cond_init(&q->not_full, NULL);
}

static void cq_push(conn_queue_t *q, int fd) {
    pthread_mutex_lock(&q->mtx);
    while (q->count == CONN_QUEUE_CAP && !q->shutdown)
        pthread_cond_wait(&q->not_full, &q->mtx);   /* backpressure */
    if (q->shutdown) { pthread_mutex_unlock(&q->mtx); close(fd); return; }
    q->fds[q->tail] = fd;
    q->tail = (q->tail + 1) % CONN_QUEUE_CAP;
    q->count++;
    pthread_cond_signal(&q->not_empty);
    pthread_mutex_unlock(&q->mtx);
}

/* Returns a client fd, or -1 when the queue is shut down and drained. */
static int cq_pop(conn_queue_t *q) {
    pthread_mutex_lock(&q->mtx);
    while (q->count == 0 && !q->shutdown)
        pthread_cond_wait(&q->not_empty, &q->mtx);
    if (q->count == 0 && q->shutdown) { pthread_mutex_unlock(&q->mtx); return -1; }
    int fd = q->fds[q->head];
    q->head = (q->head + 1) % CONN_QUEUE_CAP;
    q->count--;
    pthread_cond_signal(&q->not_full);
    pthread_mutex_unlock(&q->mtx);
    return fd;
}

static void cq_shutdown(conn_queue_t *q) {
    pthread_mutex_lock(&q->mtx);
    q->shutdown = 1;
    pthread_cond_broadcast(&q->not_empty);
    pthread_cond_broadcast(&q->not_full);
    pthread_mutex_unlock(&q->mtx);
}

typedef struct {
    qwen_tts_ctx_t *ctx;
    conn_queue_t *q;
    int id;
} worker_arg_t;

static void *worker_main(void *arg) {
    worker_arg_t *wa = (worker_arg_t *)arg;
    for (;;) {
        int fd = cq_pop(wa->q);
        if (fd < 0) break;   /* shutdown + drained */
        handle_connection(wa->ctx, fd, (struct sockaddr_in){0});
    }
    return NULL;
}

/* ── Main server loop ────────────────────────────────────────────────── */

static volatile int server_running = 1;

static void sigint_handler(int sig) {
    (void)sig;
    server_running = 0;
}

static int setup_listen_socket(int port) {
    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0) { perror("socket"); return -1; }
    int opt = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
    struct sockaddr_in addr = {
        .sin_family = AF_INET,
        .sin_addr.s_addr = INADDR_ANY,
        .sin_port = htons(port)
    };
    if (bind(server_fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        perror("bind"); close(server_fd); return -1;
    }
    if (listen(server_fd, 16) < 0) {
        perror("listen"); close(server_fd); return -1;
    }
    return server_fd;
}

static void install_signal_handlers(void) {
    struct sigaction sa = { .sa_handler = sigint_handler };
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0; /* no SA_RESTART — let accept() return EINTR */
    sigaction(SIGINT, &sa, NULL);
    sigaction(SIGTERM, &sa, NULL);
    signal(SIGPIPE, SIG_IGN);
}

static void print_banner(int port, int n_workers) {
    fprintf(stderr, "Server listening on http://0.0.0.0:%d", port);
    if (n_workers > 1)
        fprintf(stderr, " (%d workers%s)", n_workers,
                g_serialize_synth ? ", synthesis serialized: non-reentrant thread pool" : "");
    fprintf(stderr, "\nEndpoints:\n");
    fprintf(stderr, "  POST /v1/tts          — generate speech (returns WAV)\n");
    fprintf(stderr, "  POST /v1/tts/stream   — generate speech (chunked PCM stream)\n");
    fprintf(stderr, "  POST /v1/audio/speech — OpenAI-compatible TTS\n");
    fprintf(stderr, "  GET  /v1/speakers     — list speakers\n");
    fprintf(stderr, "  GET  /v1/health       — health check\n\n");
    fprintf(stderr, "Press Ctrl+C to stop.\n\n");
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Continuous/dynamic request-batching server (vLLM-style, opt-in --batch-size N)
 *
 * Architecture (distinct from the --workers pool, which runs N independent
 * single-stream synths re-reading weights N×):
 *   - ONE scheduler thread OWNS ctx and is the SOLE synthesizer. It pops jobs,
 *     groups the batchable ones, and steps them together through Talker+CP
 *     weight-stationary (qwen_tts_generate_batch_multi) — weights read once.
 *   - A reader pool reads+parses HTTP into jobs (never touches ctx synthesis)
 *     and hands fd ownership to the scheduler.
 *   - Opportunistic batching with ZERO added latency: a batch synth takes
 *     seconds, so concurrent requests pile up in the queue meanwhile; the next
 *     batch drains all waiting jobs (no linger window needed).
 *   - Requests the batch engine can't do (instruct / voice_design / streaming)
 *     run as single jobs on the scheduler thread (still the sole ctx user).
 * ═══════════════════════════════════════════════════════════════════════════ */

enum { JOB_BATCH = 0, JOB_SINGLE = 1 };

typedef struct batch_job {
    int fd;
    int kind;              /* JOB_BATCH (preset voice, batchable) or JOB_SINGLE */
    int is_stream;         /* stream this request (batched streaming or single worker) */
    int header_sent;       /* JOB_BATCH streaming: chunked header already written */
    char *text;            /* owned (JOB_BATCH) */
    char *body;            /* owned (JOB_SINGLE: re-parsed on scheduler ctx) */
    qwen_batch_req_t req;  /* JOB_BATCH: req.text aliases ->text */
    struct batch_job *next;
} batch_job_t;

typedef struct {
    batch_job_t *head, *tail;
    int count;
    pthread_mutex_t mtx;
    pthread_cond_t not_empty;
    int shutdown;
} job_queue_t;

static void jq_init(job_queue_t *q) {
    q->head = q->tail = NULL; q->count = 0; q->shutdown = 0;
    pthread_mutex_init(&q->mtx, NULL);
    pthread_cond_init(&q->not_empty, NULL);
}
static void jq_push(job_queue_t *q, batch_job_t *j) {
    j->next = NULL;
    pthread_mutex_lock(&q->mtx);
    if (q->tail) q->tail->next = j; else q->head = j;
    q->tail = j; q->count++;
    pthread_cond_signal(&q->not_empty);
    pthread_mutex_unlock(&q->mtx);
}
/* Blocking pop. Returns NULL only on shutdown+drained. */
static batch_job_t *jq_pop(job_queue_t *q) {
    pthread_mutex_lock(&q->mtx);
    while (q->count == 0 && !q->shutdown)
        pthread_cond_wait(&q->not_empty, &q->mtx);
    if (q->count == 0 && q->shutdown) { pthread_mutex_unlock(&q->mtx); return NULL; }
    batch_job_t *j = q->head;
    q->head = j->next; if (!q->head) q->tail = NULL;
    q->count--;
    pthread_mutex_unlock(&q->mtx);
    return j;
}
/* Non-blocking pop. Returns NULL if empty. */
static batch_job_t *jq_trypop(job_queue_t *q) {
    pthread_mutex_lock(&q->mtx);
    if (q->count == 0) { pthread_mutex_unlock(&q->mtx); return NULL; }
    batch_job_t *j = q->head;
    q->head = j->next; if (!q->head) q->tail = NULL;
    q->count--;
    pthread_mutex_unlock(&q->mtx);
    return j;
}
static void jq_shutdown(job_queue_t *q) {
    pthread_mutex_lock(&q->mtx);
    q->shutdown = 1;
    pthread_cond_broadcast(&q->not_empty);
    pthread_mutex_unlock(&q->mtx);
}
static void job_free(batch_job_t *j) {
    if (!j) return;
    free(j->text); free(j->body); free(j);
}

/* Parse a TTS request body into a qwen_batch_req_t WITHOUT mutating ctx (the
 * scheduler owns ctx; readers must be read-only on it). Resolves speaker/language
 * names + defaults using ctx config only. *needs_single set when the batch engine
 * can't serve it (instruct or voice_design → must run single-stream). Returns
 * malloc'd text or NULL on bad/oversized input. */
static char *parse_batch_req(qwen_tts_ctx_t *ctx, int def_speaker_id, int def_language_id,
                             const char *body,
                             qwen_batch_req_t *req, int *needs_single) {
    *needs_single = 0;
    char *text = json_extract_string(body, "text");
    if (!text) text = json_extract_string(body, "input");
    if (!text || text[0] == '\0' || strlen(text) > MAX_TTS_TEXT) { free(text); return NULL; }

    /* defaults (mirror reset_request_state). audit #5: use the start-of-server snapshot,
     * NOT live ctx->speaker_id/language_id which the scheduler mutates per admission. */
    if (ctx->voice_clone) { req->speaker_id = def_speaker_id; req->language_id = def_language_id; }
    else { req->speaker_id = 3061 /*ryan*/; req->language_id = 2050 /*English*/; }
    req->temperature = 0.5f; req->top_k = 50; req->top_p = 1.0f; req->rep_penalty = 1.05f;
    req->greedy_warmup = ctx->greedy_warmup;
    struct timeval tv; gettimeofday(&tv, NULL);
    req->seed = (uint32_t)(tv.tv_sec ^ tv.tv_usec);

    char *speaker = json_extract_string(body, "speaker");
    if (!speaker) speaker = json_extract_string(body, "voice");
    if (speaker) { int sid = qwen_tts_speaker_id(speaker); if (sid >= 0) req->speaker_id = sid; free(speaker); }
    char *language = json_extract_string(body, "language");
    if (language) { int lid = qwen_tts_language_id(language); if (lid >= 0) req->language_id = lid; free(language); }

    req->temperature = clampf((float)json_extract_number(body, "temperature", req->temperature), 0.0f, 2.0f);
    req->top_k = (int)json_extract_number(body, "top_k", req->top_k);
    if (req->top_k < 0) req->top_k = 0;
    if (req->top_k > ctx->config.codec_vocab_size) req->top_k = ctx->config.codec_vocab_size;
    req->top_p = clampf((float)json_extract_number(body, "top_p", req->top_p), 0.0f, 1.0f);
    req->rep_penalty = clampf((float)json_extract_number(body, "rep_penalty", req->rep_penalty), 0.5f, 2.0f);
    int seed = (int)json_extract_number(body, "seed", -1);
    if (seed >= 0) req->seed = (uint32_t)seed;

    /* instruct / voice_design can't go through the preset-voice batch engine */
    char *instruct = json_extract_string(body, "instruct");
    if (instruct && instruct[0]) *needs_single = 1;
    free(instruct);
    char *vd = json_extract_string(body, "voice_design");
    if (vd) { if (strcmp(vd, "true") == 0 || strcmp(vd, "1") == 0) *needs_single = 1; free(vd); }

    req->text = NULL;  /* caller sets to the owned text pointer */
    return text;
}

/* Send a finished request's audio as a WAV response. */
static void respond_wav(int fd, const float *audio, int n_samples) {
    if (!audio || n_samples <= 0) { send_error(fd, 500, "generation failed"); return; }
    int wav_size = 0;
    void *wav = build_wav(audio, n_samples, &wav_size);
    send_response(fd, 200, "audio/wav", wav, wav_size);
    free(wav);
}

/* Reader thread: pop a client fd, read+route. Synthesis is deferred to jobs;
 * only non-synth endpoints answer inline. */
/* def_speaker_id/def_language_id: audit #5 — snapshot of the ctx voice-clone defaults
 * taken ONCE at server start (single-threaded). Readers must NOT read ctx->speaker_id/
 * language_id live: the scheduler transiently overwrites+restores those per admission,
 * so a concurrent read would race and pick a wrong default. */
typedef struct { qwen_tts_ctx_t *ctx; conn_queue_t *cq; job_queue_t *jq; job_queue_t *jq_single;
                 int def_speaker_id; int def_language_id; } reader_arg_t;

static void *reader_main(void *arg) {
    reader_arg_t *ra = (reader_arg_t *)arg;
    for (;;) {
        int fd = cq_pop(ra->cq);
        if (fd < 0) break;
        char *buf = (char *)malloc(1024 * 1024);
        if (!buf) { close(fd); continue; }
        int total = read_request(fd, buf, 1024 * 1024);
        if (total <= 0) { free(buf); close(fd); continue; }
        char method[16] = {0}, path[256] = {0};
        sscanf(buf, "%15s %255s", method, path);
        const char *body = strstr(buf, "\r\n\r\n");
        body = body ? body + 4 : "";

        if (strcmp(method, "OPTIONS") == 0) {
            const char *cors = "HTTP/1.1 204 No Content\r\nAccess-Control-Allow-Origin: *\r\n"
                "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\nAccess-Control-Allow-Headers: Content-Type\r\n"
                "Connection: close\r\n\r\n";
            write(fd, cors, strlen(cors)); close(fd);
        } else if (strcmp(path, "/v1/health") == 0 && strcmp(method, "GET") == 0) {
            handle_health(fd); close(fd);
        } else if (strcmp(path, "/v1/speakers") == 0 && strcmp(method, "GET") == 0) {
            handle_speakers(fd); close(fd);
        } else if (strcmp(method, "POST") == 0 &&
                   (strcmp(path, "/v1/tts") == 0 || strcmp(path, "/v1/audio/speech") == 0 ||
                    strcmp(path, "/v1/tts/stream") == 0)) {
            int is_stream = (strcmp(path, "/v1/tts/stream") == 0);
            batch_job_t *j = (batch_job_t *)calloc(1, sizeof(batch_job_t));
            j->fd = fd;
            int needs_single = 0;
            char *text = parse_batch_req(ra->ctx, ra->def_speaker_id, ra->def_language_id, body, &j->req, &needs_single);
            if (!text) { send_error(fd, 400, "missing, empty, or oversized 'text' (max 8192 chars)"); close(fd); free(j); free(buf); continue; }
            if (needs_single) {
                /* instruct / voice_design can't batch → dedicated worker (clone ctx) */
                j->kind = JOB_SINGLE; j->is_stream = is_stream;
                j->body = strdup(body); j->text = text;  /* text freed with job */
                jq_push(ra->jq_single, j);
            } else {
                /* preset voice → continuous batch; stream requests are batched AND
                 * streamed (S3): each slot's frame is emitted as produced. */
                j->kind = JOB_BATCH; j->is_stream = is_stream;
                j->req.want_stream = is_stream;
                j->text = text; j->req.text = j->text;
                jq_push(ra->jq, j);
            }
        } else {
            send_error(fd, 404, "not found"); close(fd);
        }
        free(buf);
    }
    return NULL;
}

/* ── Continuous-batching driver glue (sink callbacks over the job queue) ──── */

typedef struct {
    job_queue_t *jq;     /* batch jobs */
    int *running;        /* &server_running */
    int admitted, done;  /* counters for the [BATCH] log */
} sink_ctx_t;

/* next_job: pop a batch job; block when the driver is fully idle. */
static int sink_next_job(void *ud, qwen_batch_req_t *req, void **tag, int block) {
    sink_ctx_t *sc = (sink_ctx_t *)ud;
    batch_job_t *j = block ? jq_pop(sc->jq) : jq_trypop(sc->jq);
    if (!j) return 0;                 /* none / shutdown */
    *req = j->req;                    /* req.text aliases j->text (valid until on_done) */
    *tag = j;
    sc->admitted++;
    return 1;
}

/* Write one float PCM buffer as an HTTP chunk (s16le). */
static void send_pcm_chunk(int fd, const float *samples, int n) {
    int16_t *pcm = (int16_t *)malloc((size_t)n * sizeof(int16_t));
    if (!pcm) return;
    for (int i = 0; i < n; i++) {
        float s = samples[i]; if (s < -1.0f) s = -1.0f; if (s > 1.0f) s = 1.0f;
        pcm[i] = (int16_t)(s * 32767);
    }
    int data_len = n * 2;
    char ch[32]; int chlen = snprintf(ch, sizeof(ch), "%x\r\n", data_len);
    write(fd, ch, chlen); write(fd, pcm, data_len); write(fd, "\r\n", 2);
    free(pcm);
}

/* on_chunk (streaming): emit one incremental PCM chunk for this request. */
static void sink_on_chunk(void *ud, void *tag, float *samples, int n_samples) {
    (void)ud;
    batch_job_t *j = (batch_job_t *)tag;
    if (n_samples <= 0 || !samples) return;
    if (!j->header_sent) { send_chunked_header(j->fd); j->header_sent = 1; }
    send_pcm_chunk(j->fd, samples, n_samples);
}

/* on_done: finish this request + close its connection. Streaming → chunked end;
 * non-streaming → full WAV. */
static void sink_on_done(void *ud, void *tag, float *samples, int n_samples) {
    sink_ctx_t *sc = (sink_ctx_t *)ud;
    batch_job_t *j = (batch_job_t *)tag;
    if (j->is_stream) {
        if (!j->header_sent) { send_chunked_header(j->fd); j->header_sent = 1; }
        send_chunked_end(j->fd);
    } else {
        respond_wav(j->fd, samples, n_samples);
        free(samples);
    }
    close(j->fd);
    int streamed = j->is_stream;
    job_free(j);
    sc->done++;
    fprintf(stderr, "[BATCH] done #%d (%s, in-flight admitted=%d)\n",
            sc->done, streamed ? "streamed" : "wav", sc->admitted);
}

static int sink_running(void *ud) {
    sink_ctx_t *sc = (sink_ctx_t *)ud;
    return *sc->running;
}

/* Continuous-batching scheduler thread: the sole batch synthesizer (owns ctx). */
typedef struct { qwen_tts_ctx_t *ctx; job_queue_t *jq; int max_batch; } sched_arg_t;
static void *scheduler_main(void *arg) {
    sched_arg_t *sa = (sched_arg_t *)arg;
    sink_ctx_t sc = { .jq = sa->jq, .running = (int *)&server_running, .admitted = 0, .done = 0 };
    qwen_batch_sink_t sink = {
        .ud = &sc, .next_job = sink_next_job, .on_done = sink_on_done,
        .on_chunk = sink_on_chunk, .running = sink_running,
    };
    qwen_tts_serve_continuous(sa->ctx, sa->max_batch, &sink);
    return NULL;
}

/* Single-job worker: streaming / instruct / voice_design on a CLONE ctx so it
 * never stalls the batch. audit #6: if the clone allocation failed (reject=1) we do
 * NOT fall back to the shared scheduler ctx — that ctx is synthesizing continuously,
 * so two threads on it would corrupt KV/dec_x. Instead we drain the queue with 503
 * (clean error) rather than serve corrupted audio. */
typedef struct { qwen_tts_ctx_t *ctx; job_queue_t *jq; int reject; } single_arg_t;
static void *single_worker_main(void *arg) {
    single_arg_t *sw = (single_arg_t *)arg;
    for (;;) {
        batch_job_t *j = jq_pop(sw->jq);
        if (!j) break;   /* shutdown + drained */
        if (sw->reject) {
            send_error(j->fd, 503, "single-job worker unavailable (clone alloc failed)");
            close(j->fd); job_free(j);
            continue;
        }
        sw->ctx->stream = 0; sw->ctx->audio_cb = NULL;
        if (j->is_stream) handle_tts_stream(sw->ctx, j->fd, j->body);
        else handle_tts(sw->ctx, j->fd, j->body);
        close(j->fd); job_free(j);
    }
    return NULL;
}

/* Batched server entry: reader pool + continuous-batching scheduler + single worker. */
int qwen_tts_serve_batched(qwen_tts_ctx_t *ctx, int port, int max_batch) {
    if (max_batch < 2) max_batch = 2;
    int server_fd = setup_listen_socket(port);
    if (server_fd < 0) return -1;
    install_signal_handlers();
    ctx->silent = 1;

    int n_readers = max_batch; if (n_readers < 2) n_readers = 2; if (n_readers > 16) n_readers = 16;

    conn_queue_t cq; cq_init(&cq);
    job_queue_t jq; jq_init(&jq);            /* batch jobs → continuous scheduler */
    job_queue_t jq_single; jq_init(&jq_single);  /* stream/instruct/voice_design → single worker */

    pthread_t *readers = (pthread_t *)calloc(n_readers, sizeof(pthread_t));
    reader_arg_t *rargs = (reader_arg_t *)calloc(n_readers, sizeof(reader_arg_t));
    /* audit #5: snapshot the voice-clone defaults now, before the scheduler (which
     * mutates ctx->speaker_id/language_id per admission) starts. */
    int def_spk = ctx->speaker_id, def_lang = ctx->language_id;
    for (int i = 0; i < n_readers; i++) {
        rargs[i].ctx = ctx; rargs[i].cq = &cq; rargs[i].jq = &jq; rargs[i].jq_single = &jq_single;
        rargs[i].def_speaker_id = def_spk; rargs[i].def_language_id = def_lang;
        pthread_create(&readers[i], NULL, reader_main, &rargs[i]);
    }
    /* continuous-batching scheduler (owns base ctx) */
    pthread_t sched;
    sched_arg_t sarg = { .ctx = ctx, .jq = &jq, .max_batch = max_batch };
    pthread_create(&sched, NULL, scheduler_main, &sarg);
    /* single-job worker on a CLONE so stream/instruct never stalls the batch
     * (clone shares weights+voice; NULL → fall back to shared ctx). */
    qwen_tts_ctx_t *single_ctx = qwen_tts_clone_for_worker(ctx);
    pthread_t single_thr;
    /* audit #6: on clone failure, run the worker in reject mode (503) rather than share
     * the scheduler's ctx (which would corrupt in-flight batch synthesis). */
    single_arg_t swarg = { .ctx = single_ctx ? single_ctx : ctx, .jq = &jq_single, .reject = (single_ctx == NULL) };
    pthread_create(&single_thr, NULL, single_worker_main, &swarg);

    fprintf(stderr, "Server listening on http://0.0.0.0:%d (continuous request-batching: max_batch=%d, %d readers%s)\n",
            port, max_batch, n_readers, single_ctx ? ", +1 single-job clone" : "");
    fprintf(stderr, "Endpoints:\n"
            "  POST /v1/tts          — generate speech (returns WAV, BATCHED)\n"
            "  POST /v1/tts/stream   — generate speech (chunked PCM, single clone)\n"
            "  POST /v1/audio/speech — OpenAI-compatible TTS (BATCHED)\n"
            "  GET  /v1/speakers     — list speakers\n"
            "  GET  /v1/health       — health check\n\n"
            "Press Ctrl+C to stop.\n\n");

    while (server_running) {
        struct sockaddr_in client_addr; socklen_t client_len = sizeof(client_addr);
        int client_fd = accept(server_fd, (struct sockaddr *)&client_addr, &client_len);
        if (client_fd < 0) { if (errno == EINTR) continue; perror("accept"); continue; }
        cq_push(&cq, client_fd);
    }

    close(server_fd);
    cq_shutdown(&cq);
    for (int i = 0; i < n_readers; i++) pthread_join(readers[i], NULL);
    jq_shutdown(&jq);
    jq_shutdown(&jq_single);
    pthread_join(sched, NULL);
    pthread_join(single_thr, NULL);
    if (single_ctx) qwen_tts_free_clone(single_ctx);
    free(readers); free(rargs);
    fprintf(stderr, "\nServer stopped.\n");
    return 0;
}

int qwen_tts_serve_ex(qwen_tts_ctx_t *ctx, int port, int n_workers) {
    if (n_workers < 1) n_workers = 1;
    int server_fd = setup_listen_socket(port);
    if (server_fd < 0) return -1;
    install_signal_handlers();

    /* Suppress model output during request handling */
    ctx->silent = 1;

    /* ── Single-worker: original inline accept loop (zero extra memory) ── */
    if (n_workers == 1) {
        print_banner(port, 1);
        while (server_running) {
            struct sockaddr_in client_addr;
            socklen_t client_len = sizeof(client_addr);
            int client_fd = accept(server_fd, (struct sockaddr *)&client_addr, &client_len);
            if (client_fd < 0) {
                if (errno == EINTR) continue;
                perror("accept");
                continue;
            }
            handle_connection(ctx, client_fd, client_addr);
        }
        close(server_fd);
        fprintf(stderr, "\nServer stopped.\n");
        return 0;
    }

    /* ── Multi-worker: acceptor thread + worker pool ──
     * Decide serialization: on a non-reentrant kernel pool (pthread/Win32) two
     * workers calling qwen_parallel at once would corrupt its single job slot. */
    g_serialize_synth = !qwen_parallel_is_reentrant();

    /* Clone n_workers-1 independent contexts (worker 0 reuses the base ctx).
     * Clones SHARE the mmapped weights + loaded voice, so only KV/work buffers
     * cost extra memory per worker. */
    qwen_tts_ctx_t **ctxs = (qwen_tts_ctx_t **)calloc(n_workers, sizeof(*ctxs));
    pthread_t *threads = (pthread_t *)calloc(n_workers, sizeof(pthread_t));
    worker_arg_t *args = (worker_arg_t *)calloc(n_workers, sizeof(worker_arg_t));
    if (!ctxs || !threads || !args) {
        fprintf(stderr, "Error: worker pool allocation failed\n");
        free(ctxs); free(threads); free(args); close(server_fd); return -1;
    }
    ctxs[0] = ctx;
    int spawned = n_workers;
    for (int i = 1; i < n_workers; i++) {
        ctxs[i] = qwen_tts_clone_for_worker(ctx);
        if (!ctxs[i]) {
            fprintf(stderr, "Warning: failed to clone worker %d; running with %d workers\n", i, i);
            spawned = i;
            break;
        }
    }

    conn_queue_t q;
    cq_init(&q);
    for (int i = 0; i < spawned; i++) {
        args[i].ctx = ctxs[i];
        args[i].q = &q;
        args[i].id = i;
        pthread_create(&threads[i], NULL, worker_main, &args[i]);
    }

    print_banner(port, spawned);

    while (server_running) {
        struct sockaddr_in client_addr;
        socklen_t client_len = sizeof(client_addr);
        int client_fd = accept(server_fd, (struct sockaddr *)&client_addr, &client_len);
        if (client_fd < 0) {
            if (errno == EINTR) continue;
            perror("accept");
            continue;
        }
        cq_push(&q, client_fd);
    }

    /* Graceful shutdown: stop accepting, drain queue, join workers. */
    close(server_fd);
    cq_shutdown(&q);
    for (int i = 0; i < spawned; i++)
        pthread_join(threads[i], NULL);

    /* Free clones (worker 0 = base ctx, owned by the caller). */
    for (int i = 1; i < spawned; i++)
        qwen_tts_free_clone(ctxs[i]);

    free(ctxs); free(threads); free(args);
    fprintf(stderr, "\nServer stopped.\n");
    return 0;
}

int qwen_tts_serve(qwen_tts_ctx_t *ctx, int port) {
    return qwen_tts_serve_ex(ctx, port, 1);
}
