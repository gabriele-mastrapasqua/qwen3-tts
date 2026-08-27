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

/* sched_setaffinity / cpu_set_t / CPU_ZERO are GNU extensions: the macro has to
 * come before ANY header, not next to the code that uses them. */
#ifdef __linux__
#ifndef _GNU_SOURCE
#define _GNU_SOURCE 1
#endif
#endif
#include "qwen_tts_server.h"
#include "qwen_tts_kernels.h"
#include <dlfcn.h>
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
#include <poll.h>
/* POLLRDHUP is a Linux extension gated on _GNU_SOURCE - which THIS FILE defines above,
 * before any header, so on Linux it IS compiled in. Do not probe it with a bare
 * `gcc -E -dM -include poll.h`: that answers a question about the default environment,
 * not about this translation unit, and reports 0 while the binary uses 0x2038.
 * macOS has no equivalent; there the pre-check falls back to POLLHUP|POLLERR|POLLNVAL,
 * which is weaker - it cannot see a half-close - and the write() return then carries the
 * detection alone. That is exactly why the write result is checked as well and never
 * replaced by the poll. QWEN_HAVE_RDHUP is emitted in the [CANCEL] record so a log says
 * which of the two paths actually ran, instead of leaving it to be guessed. */
#ifndef POLLRDHUP
#define QWEN_POLL_GONE (POLLHUP | POLLERR | POLLNVAL)
#define QWEN_HAVE_RDHUP 0
#else
#define QWEN_POLL_GONE (POLLRDHUP | POLLHUP | POLLERR | POLLNVAL)
#define QWEN_HAVE_RDHUP 1
#endif
#include <sys/time.h>
#include <stdatomic.h>

/* Built with AddressSanitizer? GCC says __SANITIZE_ADDRESS__, clang says __has_feature. */
#if defined(__SANITIZE_ADDRESS__)
#  define QWEN_ASAN 1
#elif defined(__has_feature)
#  if __has_feature(address_sanitizer)
#    define QWEN_ASAN 1
#  endif
#endif
#ifdef QWEN_ASAN
#  include <sanitizer/lsan_interface.h>
#endif
#include <pthread.h>

/* Max accepted request text length (chars). Guards against a single huge body
 * blowing up the tokenizer / generation time / memory. ~1500 words of TTS is
 * already far beyond any reasonable single request. */
#define MAX_TTS_TEXT 8192

/* Serializes synthesis on the shared ctx. The accept loop is single-threaded today
 * (one request at a time), so this is UNCONTENDED — it's the correctness foundation
 * for when the server gains per-connection concurrency (continuous batching). With a
 * shared mutable ctx, any future threading MUST hold this around parse+generate. */
/* Defined next to the prefork dispatcher; declared here because the non-batched
 * connection path above it must report completions too. */
static void srv_conn_close(int fd);

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
    /* 503 rendeva "Internal Server Error", che e' fuorviante: un server pieno non e'
     * rotto. La divisione e' quella di RFC 9110 / RFC 6585 e la rispettano tutti i server
     * di inferenza: 503 = IL SERVER non ha capacita' ora; 429 = QUESTO CLIENT ha superato
     * una quota. Noi emettiamo 503 (non abbiamo quote per cliente); il 429 sta qui per
     * quando le aggiungeremo. */
    const char *status_text = (status == 200) ? "OK" :
                              (status == 400) ? "Bad Request" :
                              (status == 404) ? "Not Found" :
                              (status == 405) ? "Method Not Allowed" :
                              (status == 429) ? "Too Many Requests" :
                              (status == 503) ? "Service Unavailable" :
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

/* ── QWEN_CANCEL_ON_DISCONNECT — stop generating for a request whose client has gone.
 * DEFAULT OFF: both arms of the A/B are then the SAME binary, and the OFF arm reproduces
 * the historical behaviour exactly. the design notes carries the register entry. */
static int qwen_cancel_on_disconnect(void) {
    static int v = -1;
    if (v < 0) { const char *e = getenv("QWEN_CANCEL_ON_DISCONNECT"); v = (e && e[0] == '1'); }
    return v;
}

/* ── Is this peer gone? TWO-SIDED, and both sides are needed.
 *
 * The poll is a fast path: it sees a FIN without waiting for a write to fail, which would
 * otherwise cost one whole chunk of latency. It is NOT the mechanism, because the race
 *      poll says connected -> peer disconnects -> write
 * is unavoidable. So the write() return is checked as well, and it stays checked: ignoring
 * write returns is precisely what produced this problem.
 *
 * client_gone == true on ANY of:
 *   pre-check   POLLRDHUP | POLLHUP | POLLERR | POLLNVAL
 *   write()     EPIPE, ECONNRESET, ENOTCONN, EBADF
 * EAGAIN and EINTR are NOT disconnects - a busy socket must not become a cancellation. */
static int peer_hung_up(int fd) {
    struct pollfd p = { .fd = fd, .events = QWEN_POLL_GONE, .revents = 0 };
    if (poll(&p, 1, 0) > 0 && (p.revents & QWEN_POLL_GONE))
        return 1;
    return 0;
}

static int write_all_or_gone(int fd, const void *buf, size_t n) {
    const char *p = (const char *)buf;
    size_t left = n;
    while (left > 0) {
        ssize_t w = write(fd, p, left);
        if (w > 0) { p += w; left -= (size_t)w; continue; }
        if (w < 0 && (errno == EINTR || errno == EAGAIN)) continue;
        if (w < 0 && (errno == EPIPE || errno == ECONNRESET ||
                      errno == ENOTCONN || errno == EBADF)) return -1;
        return -1;                      /* any other hard error: treat as gone */
    }
    return 0;
}

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

/* ── STATO DEL SERVIZIO, condiviso fra reader e scheduler ─────────────────────
 *
 * Esiste per una ragione sola: /v1/health deve dire la VERITA'. Prima rispondeva
 * `{"status":"ok"}` statico — 200 anche con lo scheduler morto e il server che drenava
 * 503. Un bilanciatore decide da li' dove mandare il traffico, quindi una salute che
 * mente non e' un dettaglio cosmetico: e' il fondamento sbagliato sotto qualunque
 * architettura a piu' processi, e peggiora le cose invece di migliorarle.
 *
 * Gli stessi contatori sono anche il minimo per essere diagnosticabili in produzione:
 * oggi coda e in-volo esistono solo su stderr. I nomi seguono di proposito quelli che
 * vLLM ha reso lo standard di fatto (num_requests_running / num_requests_waiting), cosi'
 * un router LLM-aware o un Prometheus li trovano dove se li aspetta. */
typedef struct {
    atomic_int sched_alive;      /* 0 finche' lo scheduler non e' partito, 0 se e' morto */
    atomic_int running;          /* richieste attualmente in generazione */
    atomic_int waiting;          /* richieste in coda, non ancora ammesse */
    atomic_int admitted, done;
    atomic_int rejected_full;    /* coda piena  -> 503 */
    atomic_int rejected_stale;   /* scaduta in coda -> 503 */
    int queue_max;               /* quante possono ASPETTARE oltre quelle in esecuzione */
    int slots;                   /* --batch-size: quante ne esegue insieme */
    int queue_timeout_ms;        /* 0 = nessuna scadenza */
} server_state_t;

static server_state_t g_srv;

/* -1 = automatico (2x gli slot). Impostati da main.c prima di partire. */
static int g_cfg_max_queue = -1;
static int g_cfg_queue_timeout_ms = 0;

void qwen_tts_server_set_limits(int max_queue, int queue_timeout_ms) {
    g_cfg_max_queue = max_queue;
    g_cfg_queue_timeout_ms = queue_timeout_ms;
}

/* ── Request handlers ────────────────────────────────────────────────── */

static void handle_health(int fd) {
    int alive = atomic_load(&g_srv.sched_alive);
    int waiting = atomic_load(&g_srv.waiting);
    char json[512];
    /* `status` resta "ok"/"unavailable" per non rompere i controlli esistenti che
     * cercano quella stringa; tutto il resto e' additivo. */
    snprintf(json, sizeof(json),
             "{\"status\":\"%s\",\"scheduler\":\"%s\","
             "\"num_requests_running\":%d,\"num_requests_waiting\":%d,"
             "\"queue_max\":%d,\"queue_timeout_ms\":%d,"
             "\"admitted\":%d,\"done\":%d,"
             "\"rejected_queue_full\":%d,\"rejected_queue_timeout\":%d}",
             alive ? "ok" : "unavailable", alive ? "running" : "down",
             atomic_load(&g_srv.running), waiting,
             g_srv.queue_max, g_srv.queue_timeout_ms,
             atomic_load(&g_srv.admitted), atomic_load(&g_srv.done),
             atomic_load(&g_srv.rejected_full), atomic_load(&g_srv.rejected_stale));
    /* 503 quando lo scheduler non c'e': e' il segnale con cui un bilanciatore toglie
     * questo backend dalla rotazione invece di continuare a mandargli chiamate. */
    send_json(fd, alive ? 200 : 503, json);
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
     * The emotion path is the Talker ml_steer (qlsteer) — cleared just below. */
    ctx->cp_roughness = 0.0f;
    if (ctx->ml_steer) { free(ctx->ml_steer); ctx->ml_steer = NULL; ctx->ml_steer_layers = 0; }

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
        /* qwen_tts_resolve_speaker, NOT qwen_tts_speaker_id: the latter knows only the 9
         * hardcoded CustomVoice presets and returns -1 for every voice of a finetuned
         * pool — and -1 was then silently dropped, so a request for "a pool voice" was
         * served by the DEFAULT slot. Measured 2026-08-17: 98% language identity from the CLI vs
         * 14.5% from the server, same model/text/seed, because the server was rendering
         * a different voice. Same class of silent failure as PLAN fact F9, on the
         * serving path, where nobody had looked. */
        int sid = qwen_tts_resolve_speaker(ctx, speaker);
        if (sid >= 0) ctx->speaker_id = sid;
        else fprintf(stderr, "[server] unknown speaker '%s' — falling back to the default "
                             "voice (this is almost never what you want)\n", speaker);
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
        char *sub = qwen_compose_para_substitute(text, para_voice, ctx->config.hidden_size < 2048,
                                                 &did, &para_seed, &para_temp);
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
        qwen_tts_apply_emotion(ctx, emotion, language,
                               0.0f, 0, req_vol, vol_present, req_rate, rate_present,
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
    /* Per-request chunk size (idea from PR #17). The default stays 10 frames
     * (0.8s): with the exact stateful conv decoder a chunk boundary no longer
     * costs a context re-decode, so a bigger chunk buys throughput only by
     * amortizing BLAS/dispatch — while coarsening mid-stream latency. Clients
     * that want throughput over smoothness can raise it. TTFA is set by the
     * 2-frame first chunk either way. */
    int chunk_frames = (int)json_extract_number(body, "chunk_frames", 10);
    if (chunk_frames < 2)   chunk_frames = 2;
    if (chunk_frames > 250) chunk_frames = 250;
    ctx->stream_chunk_frames = chunk_frames;
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
    if (!buf) { srv_conn_close(client_fd); return; }
    int total = read_request(client_fd, buf, 1024 * 1024);
    if (total <= 0) { free(buf); srv_conn_close(client_fd); return; }

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
    srv_conn_close(client_fd);
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
    if (q->shutdown) { pthread_mutex_unlock(&q->mtx); srv_conn_close(fd); return; }
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

static volatile sig_atomic_t server_running = 1;   /* audit: written from a signal handler */

static void sigint_handler(int sig) {
    (void)sig;
    server_running = 0;
}

/* audit MED-3: a client that connects and never sends data must not hold a reader
 * (or, in single-worker mode, the whole server) forever — bound the socket reads.
 * 30s is generous for any legit request; streaming WRITES are unaffected (send side). */
static void set_client_timeout(int fd) {
    struct timeval tv = { .tv_sec = 30, .tv_usec = 0 };
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
}


/* ── PREFORK DISPATCH: the parent owns the listener, the children own the work ────
 *
 * SO_REUSEPORT hashes connections, it does not balance them: measured on the Axion,
 * four prefork workers under 16 concurrent requests used ~11 of 16 cores and reached
 * 0.49 req/s against 0.75 for four independent processes fed round-robin. The kernel
 * was handing one worker six connections and another two.
 *
 * So the parent keeps the ONLY listening socket, accepts, and hands the accepted
 * descriptor to the least-loaded child over a unix socketpair with SCM_RIGHTS. The
 * parent never touches the traffic - it passes the descriptor and forgets it. The
 * child answers the client directly.
 *
 * Backpressure instead of rejection: a hard cap of `cap` in-flight per worker, and
 * when every worker is at cap the parent simply stops polling the listener. The
 * connection waits in the kernel backlog instead of being refused, which is what
 * "zero rejected" means here.
 *
 * The child posts one byte back per finished connection. That is the only thing it
 * has to tell the parent, and it is why srv_conn_close() exists: every path that
 * closes a client descriptor goes through it, so the count cannot drift.
 */
static int g_conn_chan_fd = -1;   /* child: receives accepted descriptors from the parent */
static int g_conn_done_fd = -1;   /* child: posts one byte per finished connection */

static void srv_conn_close(int fd) {
    if (fd >= 0) close(fd);
    if (g_conn_done_fd >= 0) {
        char b = 1;
        ssize_t r = write(g_conn_done_fd, &b, 1);
        (void)r;   /* the parent is gone or the pipe is full: neither is worth dying for */
    }
}

#if defined(__linux__)
static int srv_send_fd(int chan, int fd) {
    char dummy = 'F';
    struct iovec iov = { .iov_base = &dummy, .iov_len = 1 };
    char cbuf[CMSG_SPACE(sizeof(int))];
    memset(cbuf, 0, sizeof cbuf);
    struct msghdr msg = { .msg_iov = &iov, .msg_iovlen = 1,
                          .msg_control = cbuf, .msg_controllen = sizeof cbuf };
    struct cmsghdr *cm = CMSG_FIRSTHDR(&msg);
    cm->cmsg_level = SOL_SOCKET; cm->cmsg_type = SCM_RIGHTS;
    cm->cmsg_len = CMSG_LEN(sizeof(int));
    memcpy(CMSG_DATA(cm), &fd, sizeof(int));
    ssize_t n;
    do { n = sendmsg(chan, &msg, 0); } while (n < 0 && errno == EINTR);
    return n > 0 ? 0 : -1;
}

static int srv_recv_fd(int chan) {
    char dummy;
    struct iovec iov = { .iov_base = &dummy, .iov_len = 1 };
    char cbuf[CMSG_SPACE(sizeof(int))];
    memset(cbuf, 0, sizeof cbuf);
    struct msghdr msg = { .msg_iov = &iov, .msg_iovlen = 1,
                          .msg_control = cbuf, .msg_controllen = sizeof cbuf };
    /* NOT retried on EINTR: the child blocks here, and SIGTERM must be able to break
     * it out or the worker can never be stopped by a signal. Retrying forever here
     * left five processes alive after pkill -TERM. */
    ssize_t n = recvmsg(chan, &msg, 0);
    if (n < 0 && errno == EINTR) return -3;   /* caller re-checks server_running */
    if (n <= 0) return -1;              /* parent closed: the child should stop */
    struct cmsghdr *cm = CMSG_FIRSTHDR(&msg);
    if (!cm || cm->cmsg_type != SCM_RIGHTS) return -2;
    int fd; memcpy(&fd, CMSG_DATA(cm), sizeof(int));
    return fd;
}
#endif /* __linux__ */

static int setup_listen_socket(int port) {
    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0) { perror("socket"); return -1; }
    int opt = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
#ifdef SO_REUSEPORT
    /* --prefork: every worker binds the SAME port and the kernel hashes incoming
     * connections across the listening sockets. That is what lets the pre-fork
     * workers keep the existing serve loops UNCHANGED - no shared accept fd to thread
     * through, no acceptor process, and no thundering herd. Harmless with one worker. */
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEPORT, &opt, sizeof(opt));
#endif
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
    /* Provenienza in cima al log del server: quando un banco raccoglie lo stderr, la
     * prima riga dell'artefatto dice DA QUALE binario e con QUALI flag sono usciti i
     * numeri. Senza, due corse non sono confrontabili e non c'e' modo di scoprirlo dopo. */
    qwen_provenance_report(stderr);
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
    double enq_ms;         /* quando e' entrata in coda: serve per la scadenza di attesa */
    /* ── QWEN_LIFE_TRACE: the pre-service segments, in monotonic ms. At C=6 a request was
     * measured spending ~570 ms between leaving the client and being served, while the
     * engine ran that same request UNDER realtime. Guessing which segment holds it is
     * exactly what this avoids: every boundary is stamped, and the segments must reconcile
     * with the worker-side total or the instrumentation is declared incomplete. */
    double t_recv, t_parsed, t_admit, t_first;
    /* TTFA2 domain S. A userspace stamp is NOT "byte out": the kernel decides when a byte
     * leaves and nothing here observes it. These two are exactly what CAN be observed. */
    double t_write_attempt;   /* immediately BEFORE the first send() for this request  */
    double t_write_complete;  /* immediately AFTER the first send() that returned > 0  */
    /* STEP 3A: the driver's admission-opportunity state AS SEEN AT ENQUEUE. With these a
     * request can be told which opportunity it waited for and how far into an iteration it
     * landed - instead of the cadence being inferred from a correlation. */
    unsigned long long enq_adm_seq;
    double enq_adm_ts;        /* instant of the last admission opportunity before enqueue */
    double enq_last_iter_ms;  /* duration of the iteration that ended at that opportunity */
    unsigned int life_seed;
    /* ── Cancellation state. Lives on the JOB and not on the fd: an fd can be closed and
     * recycled by another connection while this slot still exists, and the cancel flag
     * would then be read from an unrelated client. The job outlives the slot because
     * on_done() is the last thing the driver's finalisation does (qwen_tts.h:900-905). */
    int client_gone;              /* set once by the HTTP layer, read by sink_cancelled */
    int cancelled;                /* the driver dropped this slot: CANCELLED != COMPLETED */
    double t_abort_detected;      /* when the server first observed the disconnect */
    double t_cancel_stop;         /* when the driver actually released the slot */
    struct batch_job *next;
} batch_job_t;

/* The ONLY evidence that a client stamp and a server stamp share a CLOCK_MONOTONIC origin
 * is that they were taken on the same booted kernel. Without a matching boot_id a
 * cross-domain subtraction is NOT AVAILABLE - it is not "probably fine". */
static const char *qwen_boot_id(void) {
    static char id[64] = {0};
    if (!id[0]) {
        FILE *f = fopen("/proc/sys/kernel/random/boot_id", "r");
        if (f) { if (!fgets(id, sizeof id, f)) id[0] = 0; fclose(f); }
        for (char *p = id; *p; p++) if (*p == '\n') *p = 0;
        if (!id[0]) snprintf(id, sizeof id, "NOT_AVAILABLE");
    }
    return id;
}

static double srv_now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

typedef struct {
    batch_job_t *head, *tail;
    int count;
    int cap;               /* 0 = illimitata (il vecchio comportamento) */
    pthread_mutex_t mtx;
    pthread_cond_t not_empty;
    int shutdown;
} job_queue_t;

static void jq_init(job_queue_t *q) {
    q->head = q->tail = NULL; q->count = 0; q->cap = 0; q->shutdown = 0;
    pthread_mutex_init(&q->mtx, NULL);
    pthread_cond_init(&q->not_empty, NULL);
}
/* Ritorna 1 se accodata, 0 se la coda e' PIENA (il chiamante deve rifiutare).
 *
 * ⚠️ PERCHE' UN TETTO. Prima questa coda era illimitata: la quarta richiesta veniva
 * accodata e il client aspettava all'infinito — nessun 503, nessuna scadenza. Non e' un
 * problema di prestazioni, e' un problema di CONTRATTO: il sovraccarico si manifestava
 * come silenzio, e un cliente sa gestire un rifiuto ma non sa gestire un servizio che non
 * risponde mai. E' anche lo stesso buco che vLLM ha ancora aperto (issue #18826: "la coda
 * puo' crescere indefinitamente... fino a OOM").
 *
 * Il tetto NON e' pensato per rifiutare in condizioni normali: e' 2x gli slot, cioe' al
 * massimo due tempi di generazione di attesa. Una coda piu' lunga non "assorbe un picco",
 * accumula un arretrato che nessuno vuole piu' quando lo servi. */
static int jq_push(job_queue_t *q, batch_job_t *j) {
    j->next = NULL;
    j->enq_ms = srv_now_ms();
    if (getenv("QWEN_TTFA_TRACE"))
        qwen_admit_probe_read(&j->enq_adm_seq, &j->enq_adm_ts, &j->enq_last_iter_ms);
    pthread_mutex_lock(&q->mtx);
    /* ⚠️ IL TETTO E' SUL TOTALE NEL SISTEMA, NON SULLA LUNGHEZZA DELLA CODA.
     * `cap` e' quante possono ASPETTARE oltre quelle in esecuzione, quindi la condizione e'
     *      in_esecuzione + in_attesa < slot + cap
     * La prima versione confrontava solo `count >= cap` ed era rotta nel caso che contava di
     * piu': la coda e' l'UNICA strada per entrare nello scheduler — anche a macchina scarica
     * una richiesta ci passa e lo scheduler la preleva — quindi con cap=0 la condizione era
     * sempre vera e NIENTE entrava mai. Misurato il 2026-08-20: `--max-queue 0` ha rifiutato
     * 56 richieste su 56, zero servite. Il test l'ha preso al primo colpo, che e' esattamente
     * perche' esiste. */
    if (q->cap >= 0 && atomic_load(&g_srv.running) + q->count >= g_srv.slots + q->cap) {
        pthread_mutex_unlock(&q->mtx); return 0;
    }
    if (q->tail) q->tail->next = j; else q->head = j;
    q->tail = j; q->count++;
    atomic_store(&g_srv.waiting, q->count);
    pthread_cond_signal(&q->not_empty);
    pthread_mutex_unlock(&q->mtx);
    return 1;
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
    atomic_store(&g_srv.waiting, q->count);
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
    atomic_store(&g_srv.waiting, q->count);
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
    /* same fix as the single-request path above: the batched scheduler must resolve pool
     * voices through the model's own table, or every batched request is served by the
     * default slot without a word in the log. */
    if (speaker) {
        int sid = qwen_tts_resolve_speaker(ctx, speaker);
        if (sid >= 0) req->speaker_id = sid;
        else fprintf(stderr, "[server] unknown speaker '%s' — falling back to the default "
                             "voice (this is almost never what you want)\n", speaker);
        free(speaker);
    }
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
        if (!buf) { srv_conn_close(fd); continue; }
        int total = read_request(fd, buf, 1024 * 1024);
        if (total <= 0) { free(buf); srv_conn_close(fd); continue; }
        char method[16] = {0}, path[256] = {0};
        sscanf(buf, "%15s %255s", method, path);
        const char *body = strstr(buf, "\r\n\r\n");
        body = body ? body + 4 : "";

        if (strcmp(method, "OPTIONS") == 0) {
            const char *cors = "HTTP/1.1 204 No Content\r\nAccess-Control-Allow-Origin: *\r\n"
                "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\nAccess-Control-Allow-Headers: Content-Type\r\n"
                "Connection: close\r\n\r\n";
            write(fd, cors, strlen(cors)); srv_conn_close(fd);
        } else if (strcmp(path, "/v1/health") == 0 && strcmp(method, "GET") == 0) {
            handle_health(fd); srv_conn_close(fd);
        } else if (strcmp(path, "/v1/speakers") == 0 && strcmp(method, "GET") == 0) {
            handle_speakers(fd); srv_conn_close(fd);
        } else if (strcmp(method, "POST") == 0 &&
                   (strcmp(path, "/v1/tts") == 0 || strcmp(path, "/v1/audio/speech") == 0 ||
                    strcmp(path, "/v1/tts/stream") == 0)) {
            int is_stream = (strcmp(path, "/v1/tts/stream") == 0);
            double _t_recv = srv_now_ms();
            batch_job_t *j = (batch_job_t *)calloc(1, sizeof(batch_job_t));
            j->fd = fd;
            int needs_single = 0;
            char *text = parse_batch_req(ra->ctx, ra->def_speaker_id, ra->def_language_id, body, &j->req, &needs_single);
            if (!text) { send_error(fd, 400, "missing, empty, or oversized 'text' (max 8192 chars)"); srv_conn_close(fd); free(j); free(buf); continue; }
            if (needs_single) {
                /* instruct / voice_design can't batch → dedicated worker (clone ctx) */
                j->kind = JOB_SINGLE; j->is_stream = is_stream;
                j->body = strdup(body); j->text = text;  /* text freed with job */
                j->t_recv = _t_recv; j->t_parsed = srv_now_ms();
                if (!jq_push(ra->jq_single, j)) {
                    atomic_fetch_add(&g_srv.rejected_full, 1);
                    send_error(fd, 503, "server at capacity: queue full");
                    srv_conn_close(fd); job_free(j); free(buf); continue;
                }
            } else {
                /* preset voice → continuous batch; stream requests are batched AND
                 * streamed (S3): each slot's frame is emitted as produced. */
                j->kind = JOB_BATCH; j->is_stream = is_stream;
                j->req.want_stream = is_stream;
                j->text = text; j->req.text = j->text;
                j->t_recv = _t_recv; j->t_parsed = srv_now_ms();
                if (!jq_push(ra->jq, j)) {
                    atomic_fetch_add(&g_srv.rejected_full, 1);
                    send_error(fd, 503, "server at capacity: queue full");
                    srv_conn_close(fd); job_free(j); free(buf); continue;
                }
            }
        } else {
            send_error(fd, 404, "not found"); srv_conn_close(fd);
        }
        free(buf);
    }
    return NULL;
}

/* ── Continuous-batching driver glue (sink callbacks over the job queue) ──── */

typedef struct {
    job_queue_t *jq;     /* batch jobs */
    volatile sig_atomic_t *running;   /* &server_running */
    int admitted, done;  /* counters for the [BATCH] log */
} sink_ctx_t;

/* next_job: pop a batch job; block when the driver is fully idle. */
static int sink_next_job(void *ud, qwen_batch_req_t *req, void **tag, int block) {
    sink_ctx_t *sc = (sink_ctx_t *)ud;
    batch_job_t *j;
    /* SCADENZA DI ATTESA. Una richiesta che ha aspettato oltre il budget non verra' mai
     * servita in tempo: consegnarle audio in ritardo e' peggio che dirle di no, perche'
     * intanto ha occupato uno slot che sarebbe servito a una richiesta ancora viva. E' la
     * stessa idea della "viabilita' binaria" dello streaming applicata all'AMMISSIONE:
     * sotto la soglia il servizio e' buono, sopra non serve a nessuno. */
    for (;;) {
        j = block ? jq_pop(sc->jq) : jq_trypop(sc->jq);
        if (!j) return 0;                 /* none / shutdown */
        if (g_srv.queue_timeout_ms > 0 &&
            srv_now_ms() - j->enq_ms > (double)g_srv.queue_timeout_ms) {
            atomic_fetch_add(&g_srv.rejected_stale, 1);
            send_error(j->fd, 503, "server at capacity: queued too long");
            srv_conn_close(j->fd); job_free(j);
            continue;                      /* prova la prossima, non abbandonare il giro */
        }
        break;
    }
    atomic_fetch_add(&g_srv.running, 1);
    atomic_fetch_add(&g_srv.admitted, 1);
    j->t_admit = srv_now_ms();        /* handed to the engine: the pre-service wait ends here */
    j->life_seed = j->req.seed;
    *req = j->req;                    /* req.text aliases j->text (valid until on_done) */
    *tag = j;
    sc->admitted++;
    /* Log the ADMISSION, not only the completion. Without this line the only server-side
     * timestamp a load test can correlate against is `[BATCH] done`, so a slow request can
     * be tied to another one FINISHING but never to another one STARTING — and "a new
     * arrival costs the requests already in flight" is precisely the hypothesis we are
     * trying to confirm or kill. Cheap: one line per request, not per frame. */
    fprintf(stderr, "[BATCH] admit #%d (in-flight admitted=%d, done=%d)\n",
            sc->admitted, sc->admitted - sc->done, sc->done);
    return 1;
}

/* Write one float PCM buffer as an HTTP chunk (s16le). */
static int send_pcm_chunk(int fd, const float *samples, int n) {
    int16_t *pcm = (int16_t *)malloc((size_t)n * sizeof(int16_t));
    if (!pcm) return 0;
    for (int i = 0; i < n; i++) {
        float s = samples[i]; if (s < -1.0f) s = -1.0f; if (s > 1.0f) s = 1.0f;
        pcm[i] = (int16_t)(s * 32767);
    }
    int data_len = n * 2;
    char ch[32]; int chlen = snprintf(ch, sizeof(ch), "%x\r\n", data_len);
    int gone = 0;
    if (write_all_or_gone(fd, ch, (size_t)chlen) < 0) gone = 1;
    else if (write_all_or_gone(fd, pcm, (size_t)data_len) < 0) gone = 1;
    else if (write_all_or_gone(fd, "\r\n", 2) < 0) gone = 1;
    free(pcm);
    return gone;
}

/* on_chunk (streaming): emit one incremental PCM chunk for this request. */
static void sink_on_chunk(void *ud, void *tag, float *samples, int n_samples) {
    (void)ud;
    batch_job_t *j = (batch_job_t *)tag;
    if (n_samples <= 0 || !samples) return;
    if (j->t_first == 0.0) j->t_first = srv_now_ms();
    if (j->t_write_attempt == 0.0) j->t_write_attempt = srv_now_ms();
    if (!j->header_sent) { send_chunked_header(j->fd); j->header_sent = 1; }
    if (!j->client_gone && peer_hung_up(j->fd)) {
        j->client_gone = 1; j->t_abort_detected = srv_now_ms();
    }
    int _gone = send_pcm_chunk(j->fd, samples, n_samples);
    if (j->t_write_complete == 0.0 && !_gone) j->t_write_complete = srv_now_ms();
    if (_gone && !j->client_gone) {
        j->client_gone = 1; j->t_abort_detected = srv_now_ms();
    }
}

/* The driver asks, per slot per frame, whether this request should stop. */
static int sink_cancelled(void *ud, void *tag) {
    (void)ud;
    batch_job_t *j = (batch_job_t *)tag;
    if (!j || !qwen_cancel_on_disconnect()) return 0;
    if (!j->client_gone && j->fd >= 0 && peer_hung_up(j->fd)) {
        j->client_gone = 1; j->t_abort_detected = srv_now_ms();
    }
    if (j->client_gone && j->t_cancel_stop == 0.0) j->t_cancel_stop = srv_now_ms();
    return j->client_gone;
}

static int qwen_life_trace(void) {
    static int v = -1;
    if (v < 0) v = getenv("QWEN_LIFE_TRACE") ? 1 : 0;
    return v;
}
static void qwen_life_emit(batch_job_t *j) {
    if (!qwen_life_trace()) return;
    const double d = srv_now_ms();
    if (j->t_first == 0.0) j->t_first = d;
    /* CANCELLED != COMPLETED, and the three phases of a cancellation are separate:
     *   disconnect_detect  the server noticing the peer is gone
     *   cancel_to_stop     the driver dropping the row once it knows
     * If a cancellation looks slow, this says which half it was. */
    if (j->client_gone)
        fprintf(stderr, "[CANCEL] pid=%d seed=%u detected_ms=%.1f stopped_ms=%.1f "
                        "cancel_to_stop_ms=%.1f enabled=%d rdhup=%d "
                        "detected_abs_ms=%.1f\n",
                (int)getpid(), j->life_seed,
                j->t_abort_detected > 0 ? j->t_abort_detected - j->t_recv : -1.0,
                j->t_cancel_stop > 0 ? j->t_cancel_stop - j->t_recv : -1.0,
                (j->t_cancel_stop > 0 && j->t_abort_detected > 0)
                    ? j->t_cancel_stop - j->t_abort_detected : -1.0,
                qwen_cancel_on_disconnect(), QWEN_HAVE_RDHUP,
                /* ABSOLUTE CLOCK_MONOTONIC. A detection latency is only meaningful
                 * against a client timestamp taken from the SAME clock; subtracting a
                 * client CLOCK_REALTIME from a server-relative offset is not a
                 * measurement, and that mistake produced a phantom 2.8 s lag once. */
                j->t_abort_detected > 0 ? j->t_abort_detected : -1.0);
    if (getenv("QWEN_TTFA_TRACE"))
        fprintf(stderr, "[PATH] v=2 seed=%u pid=%d clock=CLOCK_MONOTONIC domain=S "
                        "boot_id=%s recv=%.3f parsed=%.3f enqueued=%.3f admitted=%.3f "
                        "write_attempt=%.3f write_complete=%.3f enq_adm_seq=%llu "
                        "enq_adm_ts=%.3f enq_last_iter_ms=%.3f\n",
                j->life_seed, (int)getpid(), qwen_boot_id(),
                j->t_recv, j->t_parsed, j->enq_ms, j->t_admit,
                j->t_write_attempt, j->t_write_complete,
                j->enq_adm_seq, j->enq_adm_ts, j->enq_last_iter_ms);
    fprintf(stderr, "[LIFE] pid=%d seed=%u parse=%.1f queue=%.1f pre_service=%.1f "
                    "ttfa_after_admit=%.1f service=%.1f worker_total=%.1f%s\n",
            (int)getpid(), j->life_seed,
            j->t_parsed - j->t_recv,     /* HTTP read + parse                       */
            j->t_admit  - j->enq_ms,     /* sat in the ready queue                  */
            j->t_admit  - j->t_recv,     /* everything before the engine sees it    */
            j->t_first  - j->t_admit,    /* engine start to first audio byte        */
            d - j->t_admit,              /* service                                 */
            d - j->t_recv,               /* worker-side total                       */
            j->client_gone ? " state=CANCELLED" : " state=COMPLETED");
}

/* on_done: finish this request + close its connection. Streaming → chunked end;
 * non-streaming → full WAV. */
static void sink_on_done(void *ud, void *tag, float *samples, int n_samples) {
    sink_ctx_t *sc = (sink_ctx_t *)ud;
    batch_job_t *j = (batch_job_t *)tag;
    qwen_life_emit(j);
    if (j->is_stream) {
        if (!j->header_sent) { send_chunked_header(j->fd); j->header_sent = 1; }
        send_chunked_end(j->fd);
    } else {
        respond_wav(j->fd, samples, n_samples);
        free(samples);
    }
    srv_conn_close(j->fd);
    int streamed = j->is_stream;
    job_free(j);
    sc->done++;
    atomic_fetch_add(&g_srv.done, 1);
    atomic_fetch_sub(&g_srv.running, 1);
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
    sink_ctx_t sc = { .jq = sa->jq, .running = &server_running, .admitted = 0, .done = 0 };
    qwen_batch_sink_t sink = {
        .ud = &sc, .next_job = sink_next_job, .on_done = sink_on_done,
        .on_chunk = sink_on_chunk, .running = sink_running,
        .cancelled = sink_cancelled,
    };
    atomic_store(&g_srv.sched_alive, 1);
    int rc = qwen_tts_serve_continuous(sa->ctx, sa->max_batch, &sink);
    /* Da qui in poi /v1/health deve dire la verita': o e' uno spegnimento ordinato, o lo
     * scheduler e' morto — in entrambi i casi questo backend non serve piu' richieste, e
     * un bilanciatore lo deve sapere PRIMA di mandargliene un'altra. */
    atomic_store(&g_srv.sched_alive, 0);
    /* audit MED-1: if the batch driver died (alloc failure / no bf16 weights) while the
     * server is still up, readers keep queueing JOB_BATCH into an unbounded queue nobody
     * pops → every batch client hangs forever, silently. Become a 503-drain instead
     * (mirrors the single-worker reject path) until shutdown. */
    if (rc != 0 && server_running) {
        fprintf(stderr, "[BATCH] FATAL: continuous scheduler failed (rc=%d) — "
                        "draining batch jobs with 503 until shutdown\n", rc);
        for (;;) {
            batch_job_t *j = jq_pop(sa->jq);
            if (!j) break;   /* shutdown + drained */
            send_error(j->fd, 503, "batch scheduler unavailable (startup failure)");
            srv_conn_close(j->fd); job_free(j);
        }
    }
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
            srv_conn_close(j->fd); job_free(j);
            continue;
        }
        sw->ctx->stream = 0; sw->ctx->audio_cb = NULL;
        if (j->is_stream) handle_tts_stream(sw->ctx, j->fd, j->body);
        else handle_tts(sw->ctx, j->fd, j->body);
        srv_conn_close(j->fd); job_free(j);
    }
    return NULL;
}


/* ── Pre-warm: pay the first generation's cost at startup, not on the first user ──
 *
 * MEASURED (PLAN T5.5 A2): the first call costs ~12% more than the second (43.4 s ->
 * 38.1 s, then 38.3 — the effect is gone by the third). That is allocation, page faults
 * on the weights, the thread pool ramping and the speech decoder's cold buffers. Today
 * the first customer of the day pays all of it; moving it to startup is free.
 *
 * It must be a REAL generation, not a warm read: the talker alone leaves the Code
 * Predictor and the speech decoder cold, and the decoder's first invocation is a
 * measurable part of that 12% (T5.5 D4). So we synthesize a short sentence end-to-end
 * and throw the audio away.
 *
 * WHERE it runs matters: before any thread is created, while this thread still owns
 * ctx exclusively. In the batched server the scheduler thread takes ownership right
 * after — pre-warming later would race it.
 *
 * QWEN_NO_PREWARM=1 skips it (server up in the time it takes to load, for a test that
 * measures the cold path on purpose). */

/* ── On the SERVER, the memory levers are on by default ──────────────────────
 *
 * QWEN_PREFILL_QUANT rebuilds the prefill's f32 scratch from the quantized weights
 * instead of the bf16, and QWEN_FREE_BF16 then releases what that makes dead: 4.0 GB
 * on the 1.7B at int8 (1344 MB heap + 2685 MB mapped, against a 5.4 GB peak), plus a
 * prefill that drops from ~3.5 s to ~0.7 s on the same model.
 *
 * WHY DEFAULT-ON HERE AND NOWHERE ELSE. Turning them on changes the generated audio —
 * the prompt's KV is computed with int8 weights, and that KV conditions everything
 * after it — so this is a product decision, not a build-time default. It was taken by
 * ear on 2026-08-18 ("audio perfetti") and it applies where the memory is money: a
 * rented box, where RSS decides how many instances fit. The CLI keeps the old
 * behaviour so `make test-golden` stays a stable reference rather than becoming a
 * moving one.
 *
 * setenv with overwrite=0: an explicit QWEN_PREFILL_QUANT=0 from the operator still
 * wins. And it runs BEFORE the pre-warm, whose first prefill is what triggers the
 * release. */
/* Il decoder batchato fra slot: acceso di default sul server, e SOLO lui.
 *
 * Batchare il decoder ha migliorato entrambe le colonne insieme — a c=4 throughput
 * +8,1% e TTFA p50 -8,9%, p95 -6,1% — che nella giornata e' stata l'unica modifica
 * senza compromesso da dichiarare. E l'audio non si muove: 81 campioni su 161280
 * differiscono di UN LSB su 32768, correlazione 1.00000000, e l'ascolto dell'utente
 * (2026-08-18) ha promosso entrambi i bracci come indistinguibili.
 *
 * Sul server e basta, perche' con uno slot solo non c'e' niente da batchare: la CLI
 * non guadagnerebbe nulla e il golden resta un riferimento fermo.
 *
 * ⚠️ Il THREAD decoder (QWEN_DECODER_THREAD) NON viene acceso qui: quello toglie il
 * decode dal percorso critico ma contende i core, e sul nostro banco il TTFA non era
 * migliorato. Sono due leve diverse e vanno accese sulla base di due misure diverse. */
static void server_default_decoder_batch(qwen_tts_ctx_t *ctx) {
    (void)ctx;
    if (getenv("QWEN_SERVER_NO_DECODER_BATCH")) return;
    setenv("QWEN_DECODER_BATCH", "1", 0);      /* un QWEN_DECODER_BATCH=0 esplicito vince */
    fprintf(stderr, "[serve] batched speech decoder ON by default (one pass over the decoder "
                    "weights for all active slots) — QWEN_DECODER_BATCH=0 to opt out\n");
}

/* ⛔ QUANTIZED PREFILL IS **NOT** A DEFAULT — MEASURED 2026-08-18, on the customer's
 * finetune, and it is the reason this function no longer turns anything on.
 *
 * WHAT HAPPENED. The lever was validated by ear on OSS models in English, where it is
 * harmless: the audio is clean, the memory saving is real (4.0 GB on the 1.7B int8) and
 * the prefill gets faster. On a finetuned checkpoint with `--quant-mixed-int6=q4n14`,
 * one pool voice, same texts and same seeds, six clips per arm:
 *
 *     prefill quant OFF   language identity 96.3% mean, worst clip 86.1%
 *     prefill quant ON    language identity 38.0% mean, three clips at 0.0 / 1.4 / 11.7%
 *
 * The failure mode is the one this project keeps meeting: the audio stays clean, the
 * duration stays normal, nothing rasps — and the model DRIFTS INTO ENGLISH, losing the
 * the target language the finetune exists for. No signal-level metric sees it; only the language-identity check
 * does. The isolation was clean: an arm with the batched decoder off instead scored
 * identically to the all-on arm, clip by clip, so the decoder is exonerated and the
 * prefill is not.
 *
 * WHY IT IS BELIEVABLE. The accent lives in weight changes of at most 0.002 spread over
 * 2008 codebook rows — small in magnitude, systematic in extent. Quantizing the prefill
 * is exactly the perturbation that erases that: the PROMPT encoding is where the accent
 * is decided, and it turns out to be far more sensitive than the per-step path, which
 * runs on the same quantized weights and keeps the accent fine.
 *
 * SO: off by default, available with QWEN_PREFILL_QUANT=1 for a base/OSS model where
 * memory matters more than an accent that isn't there. The memory saving goes with it —
 * freeing the bf16 requires the prefill to read the quantized weights — and that is a
 * real cost, stated rather than traded away silently: the 1.7B --int8 was OOM-killed on
 * a 16 GB Mac without it. On a server box with RAM, correctness wins. */
static void server_default_memory_levers(qwen_tts_ctx_t *ctx) {
    int quantized = ctx->layers && (ctx->layers[0].wq_int8 || ctx->layers[0].wq_q4 || ctx->layers[0].wq_q6);
    if (!quantized) return;                            /* nothing to read instead of bf16 */
    /* Report the EFFECTIVE state, not the intended one. The previous version printed
     * "memory levers ON by default" even when an explicit QWEN_PREFILL_QUANT=0 had
     * turned them off — a log that describes intentions instead of behaviour is how a
     * bench measures one configuration believing it measured another. */
    const char *e = getenv("QWEN_PREFILL_QUANT");
    int on = (e && e[0] && e[0] != '0');
    if (on) {
        setenv("QWEN_FREE_BF16", "1", 0);              /* only meaningful together */
        fprintf(stderr, "[serve] quantized prefill ON (explicitly requested): frees the bf16 "
                        "(~4 GB on the 1.7B) but MEASURABLY COSTS THE ACCENT on a finetune — "
                        "language identity 96%% -> 38%% on that finetune, 2026-08-18. Base/OSS models only.\n");
    } else {
        fprintf(stderr, "[serve] quantized prefill OFF (default since 2026-08-18: it loses the "
                        "language identity on finetunes) — QWEN_PREFILL_QUANT=1 to opt in on a base model\n");
    }
}

static double prewarm_now_ms(void) {
    struct timeval tv; gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}
static void server_prewarm(qwen_tts_ctx_t *ctx) {
    if (getenv("QWEN_NO_PREWARM")) return;
    int sv_silent = ctx->silent;
    ctx->silent = 1;
    float *aud = NULL; int n = 0;
    double t0 = prewarm_now_ms();
    int rc = qwen_tts_generate(ctx, "Warm up.", &aud, &n);
    free(aud);
    ctx->silent = sv_silent;
    reset_request_state(ctx);
    if (rc == 0 && n > 0)
        fprintf(stderr, "[serve] pre-warm: %.0f ms, %.2f s of audio discarded "
                        "(the first user no longer pays it)\n", prewarm_now_ms() - t0, n / 24000.0);
    else
        fprintf(stderr, "[serve] pre-warm skipped (rc=%d)\n", rc);
}

/* Batched server entry: reader pool + continuous-batching scheduler + single worker. */
int qwen_tts_serve_batched(qwen_tts_ctx_t *ctx, int port, int max_batch) {
    if (max_batch < 2) max_batch = 2;
    /* Under --prefork the PARENT owns the only listening socket and feeds us
     * descriptors, so binding here would be a second listener competing for the
     * same port - which is exactly the SO_REUSEPORT behaviour this replaced. */
    int server_fd = (g_conn_chan_fd >= 0) ? -1 : setup_listen_socket(port);
    if (server_fd < 0 && g_conn_chan_fd < 0) return -1;
    install_signal_handlers();
    ctx->silent = 1;
    server_default_memory_levers(ctx);
    server_default_decoder_batch(ctx);
    server_prewarm(ctx);          /* while this thread still owns ctx alone */

    int n_readers = max_batch; if (n_readers < 2) n_readers = 2; if (n_readers > 16) n_readers = 16;

    conn_queue_t cq; cq_init(&cq);
    job_queue_t jq; jq_init(&jq);            /* batch jobs → continuous scheduler */
    /* ── LA CODA SI TARA SUL BUDGET DI LATENZA, NON SUGLI SLOT ────────────────────
     *
     * Il tetto era 2x gli slot. Misurato il 2026-08-20 con traffico realistico e raffiche
     * da 10: con 3 slot e ~39 s di lavoro per slot sotto raffica, una coda profonda 6 vale
     * fino a ~78 s di attesa — e infatti il TTFA p95 degli arrivi NORMALI e' schizzato a
     * **28 secondi**, perche' una chiamata normale finiva in fila dietro cinque richieste
     * della raffica. Il p50 restava 418 ms: il server non era lento, era la coda a essere
     * piu' profonda del budget.
     *
     * ⚠️ UNA CODA PIU' PROFONDA DEL BUDGET DI LATENZA NON ASSORBE UN PICCO: produce
     * risposte che nessuno vuole piu'. E' la "viabilita' binaria" dello streaming (VoxServe)
     * applicata all'ammissione: sotto la soglia il servizio e' buono, sopra non serve a
     * nessuno, e servire tardi e' peggio che dire di no — perche' intanto quello slot
     * sarebbe servito a una richiesta ancora viva.
     *
     * E c'e' una ragione di ARCHITETTURA piu' forte della latenza: se sopra c'e' un
     * bilanciatore, la coda deve stare LI', non qui. Harchol-Balter (SIGMETRICS 2009)
     * dimostra che una coda centrale condivisa E' un M/GI/n, mentre code per-server
     * impegnano una richiesta su un worker prima di sapere quale si liberera' per primo.
     * HAProxy e' costruito cosi': `maxconn` sul server dice quante ne puo' ESEGUIRE, la
     * coda la tiene lui, e oltre `maxqueue` ridistribuisce ad altri server. Un worker che
     * accoda ruba al bilanciatore l'unica informazione con cui potrebbe fare meglio.
     *
     * Quindi:  0 = NESSUNA CODA, 503 immediato quando gli slot sono pieni (dietro un
     *              bilanciatore e' la configurazione giusta)
     *          N = al massimo N in attesa
     *         <0 = automatico: 1, una sola posizione di grazia per la richiesta che arriva
     *              qualche centinaio di ms prima che uno slot si liberi
     * Il vecchio comportamento illimitato — quello in cui la quarta richiesta aspettava per
     * sempre senza risposta — resta raggiungibile SOLO con QWEN_QUEUE_UNBOUNDED=1, perche'
     * e' il difetto, non una configurazione. */
    jq.cap = (g_cfg_max_queue >= 0) ? g_cfg_max_queue : 1;
    if (getenv("QWEN_QUEUE_UNBOUNDED")) {
        jq.cap = -1;
        fprintf(stderr, "[serve] ⚠️  QWEN_QUEUE_UNBOUNDED: coda ILLIMITATA — una richiesta in "
                        "eccesso aspettera' senza limite e senza errore. E' il comportamento "
                        "di prima del 2026-08-20, tenuto solo per A/B.\n");
    }
    g_srv.queue_max = jq.cap;
    g_srv.slots = max_batch;
    g_srv.queue_timeout_ms = g_cfg_queue_timeout_ms;
    fprintf(stderr, "[serve] %d slot · possono attendere %d (totale nel sistema %d) · scadenza %s\n",
            max_batch, jq.cap, max_batch + jq.cap,
            g_srv.queue_timeout_ms > 0 ? "attiva" : "nessuna");
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
        int client_fd;
        if (g_conn_chan_fd >= 0) {
#if defined(__linux__)
            /* prefork: the parent owns the listener and hands us descriptors. A closed
             * channel means the parent is gone, which is our shutdown signal. */
            client_fd = srv_recv_fd(g_conn_chan_fd);
            if (client_fd == -1) break;      /* parent gone */
            if (client_fd < 0) continue;     /* EINTR or a malformed message */
            /* ── elastic allocation, child half ──
             * The parent may have widened or narrowed our core slice while we were
             * idle. Read what the kernel actually granted - never what we asked for -
             * and match the thread budget to it. qwen_set_threads_soft does NOT touch
             * the pool, so this costs nothing and cannot deadlock; the pool was
             * spawned at the WIDEST slice this worker can ever hold. Checked here
             * because this is the moment work arrives, i.e. exactly when the
             * allocation last changed. */
            {
                cpu_set_t got; CPU_ZERO(&got);
                if (sched_getaffinity(0, sizeof got, &got) == 0) {
                    int n = CPU_COUNT(&got);
                    if (n > 0 && n != qwen_get_threads()) qwen_set_threads_soft(n);
                }
            }
#else
            break;
#endif
        } else {
            struct sockaddr_in client_addr; socklen_t client_len = sizeof(client_addr);
            client_fd = accept(server_fd, (struct sockaddr *)&client_addr, &client_len);
            if (client_fd < 0) { if (errno == EINTR) continue; perror("accept"); continue; }
        }
        set_client_timeout(client_fd);
        cq_push(&cq, client_fd);
    }

    if (server_fd >= 0) close(server_fd);
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
    server_default_memory_levers(ctx);
    server_default_decoder_batch(ctx);
    server_prewarm(ctx);          /* before any worker thread exists */

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
            set_client_timeout(client_fd);
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
        set_client_timeout(client_fd);
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

/* ══════════════════════════════════════════════════════════════════════════════
 * PRE-FORK: pack ONCE, then fork — so the workers share the weights
 *
 * The topology matrix measured 4 workers x 4 pinned cores at 1.89x the throughput
 * and 1.92x the RTF of one 16-thread pool. The blocker was memory: four independent
 * servers meant four copies of everything, and with the KleidiAI packs that is
 * ~13 GB each.
 *
 * The fix is the oldest one there is. The parent loads the model AND packs every
 * KleidiAI RHS, and only then forks. Those pages are never written again, so
 * copy-on-write never copies them: one physical copy, W workers reading it. The
 * measurement to trust is the sum of Pss across the children, NOT the sum of their
 * RSS - RSS counts every shared page once per process and would report four copies
 * that do not exist.
 *
 * Each child binds the same port with SO_REUSEPORT and runs the ORDINARY serve loop,
 * so nothing about request handling changes.
 *
 * Two things must be rebuilt in the child, because fork() does not carry them:
 *   - the thread pool. Its worker threads do not survive, but its struct does, and
 *     it would claim to have K-1 workers that no longer exist - qwen_parallel would
 *     then wait forever. qwen_threadpool_after_fork() drops that state so the next
 *     qwen_set_threads() spawns a real pool.
 *   - the affinity mask, set per child to its own slice of cores.
 */
#if defined(__linux__)
#include <sched.h>
#include <sys/wait.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/un.h>

static volatile sig_atomic_t g_prefork_stop = 0;
static volatile sig_atomic_t g_prefork_dump = 0;
/* Dump every counter a worker owns. Called by the prefork child before _exit(), where
 * atexit and destructors do not run. dlsym for the yield tracer so the engine never
 * depends on the LD_PRELOAD shim being present. */
static void qwen_worker_dump_counters(void) {
    qwen_pool_stats_report();            /* no-op unless built -DQWEN_POOL_STATS */
    if (qwen_census_enabled()) qwen_census_report(NULL);
    if (qwen_matmat_stats_enabled()) qwen_matmat_stats_report(NULL);  /* kernel mix, MACs, bytes */
    void (*yt)(void) = (void (*)(void))dlsym(RTLD_DEFAULT, "yieldtrace_report");
    if (yt) yt();
    fflush(stderr);
}

static void prefork_parent_sig(int sig) { (void)sig; g_prefork_stop = 1; }
/* SIGUSR1: print the per-worker counters and RESET them. A capacity bench needs the
 * assignment and the mean in-flight PER LEVEL, and restarting the server for every
 * level would cost more than the measurement. */
static void prefork_dump_sig(int sig) { (void)sig; g_prefork_dump = 1; }


/* ── ELASTIC CORE ALLOCATION ──────────────────────────────────────────────────────
 *
 * The static topologies each win a different band: 2x8 is best up to three
 * simultaneous callers (TTFA p95 461 ms at C=3), 4x4 is what holds at eight. Elastic
 * allocation tries to have both by moving the cores instead of the requests, on the
 * schedule the measurements suggested:
 *
 *     1 busy worker   ->  8 cores to it       (a B=1 stream saturates at 8 threads)
 *     2 busy          ->  8 + 8
 *     3 busy          ->  8 + 4 + 4
 *     4 or more       ->  4 each
 *
 * Slices are contiguous and DISJOINT among busy workers. Idle workers are parked on
 * the last core: their mask is irrelevant while they hold no work, and the plan is
 * recomputed and applied BEFORE a descriptor is handed to a worker, so nobody ever
 * starts a request while holding an overlapping mask.
 *
 * At one busy worker this deliberately leaves 8 cores unused. That is the finding,
 * not an oversight: past 8 threads a single stream REGRESSES (0.63x per doubling,
 * context switches 10k -> 220k/s).
 */
static void elastic_plan(int workers, int ncpu, const int *active, int *slice) {
    int busy = 0;
    for (int w = 0; w < workers; w++) if (active[w] > 0) busy++;
    for (int w = 0; w < workers; w++) slice[w] = 0;
    if (busy <= 0) { slice[0] = ncpu / 2; return; }
    if (busy >= 4) {
        for (int w = 0; w < workers; w++) if (active[w] > 0) slice[w] = ncpu / busy;
        return;
    }
    const int big = ncpu / 2;
    const int rest = (busy > 1) ? (ncpu - big) / (busy - 1) : 0;
    int seen = 0;
    for (int w = 0; w < workers; w++)
        if (active[w] > 0) slice[w] = (seen++ == 0) ? big : rest;
}

/* Apply a plan: contiguous disjoint ranges for the busy, the last core for the idle.
 * Returns 1 if anything actually changed, so the common case costs one memcmp. */
static int elastic_apply(int workers, int ncpu, const int *slice, const pid_t *kids,
                         int *base_out) {
    int changed = 0, next = 0;
    for (int w = 0; w < workers; w++) {
        if (kids[w] <= 0) continue;
        int lo, hi;
        if (slice[w] > 0) { lo = next; hi = next + slice[w] - 1; next += slice[w]; }
        else              { lo = hi = ncpu - 1; }      /* idle parking, see the note */
        if (hi >= ncpu) hi = ncpu - 1;
        if (base_out[2 * w] == lo && base_out[2 * w + 1] == hi) continue;
        cpu_set_t set; CPU_ZERO(&set);
        for (int c = lo; c <= hi; c++) CPU_SET(c, &set);
        if (sched_setaffinity(kids[w], sizeof set, &set) != 0) {
            perror("sched_setaffinity(child)");
            continue;                       /* leave base_out so we retry next time */
        }
        base_out[2 * w] = lo; base_out[2 * w + 1] = hi;
        changed = 1;
    }
    return changed;
}

int qwen_tts_serve_prefork(qwen_tts_ctx_t *ctx, int port, int workers,
                           int threads_per, int max_batch) {
    const int elastic = getenv("QWEN_PREFORK_ELASTIC") &&
                        atoi(getenv("QWEN_PREFORK_ELASTIC")) != 0;
    if (workers < 1) workers = 1;
    const int ncpu = (int)sysconf(_SC_NPROCESSORS_ONLN);
    const int per = ncpu / workers > 0 ? ncpu / workers : 1;
    if (threads_per < 1) threads_per = per;
    const int cap = max_batch >= 1 ? max_batch : 1;   /* hard in-flight cap per worker */

    int listen_fd = setup_listen_socket(port);
    if (listen_fd < 0) return -1;

    int (*sp)[2] = (int (*)[2])calloc((size_t)workers, sizeof(int[2]));
    pid_t *kids = (pid_t *)calloc((size_t)workers, sizeof(pid_t));
    long long *assigned = (long long *)calloc((size_t)workers, sizeof(long long));
    long long *completed = (long long *)calloc((size_t)workers, sizeof(long long));
    int *active = (int *)calloc((size_t)workers, sizeof(int));
    int *slice = (int *)calloc((size_t)workers, sizeof(int));
    /* current [lo,hi] per worker, so a re-plan that changes nothing costs a compare */
    int *cur = (int *)malloc((size_t)workers * 2 * sizeof(int));
    long long rejected = 0, replans = 0;
    if (cur) for (int w = 0; w < 2 * workers; w++) cur[w] = -1;
    if (!sp || !kids || !assigned || !completed || !active || !slice || !cur) return -1;

    fprintf(stderr, "prefork: %d workers x %d threads, %d cpus (%d per worker), "
                    "cap %d in flight each, port %d%s\n",
            workers, threads_per, ncpu, per, cap, port,
            elastic ? " · ELASTIC core allocation" : "");
    if (elastic && threads_per < ncpu / 2)
        fprintf(stderr, "prefork: ⚠️  elastic wants --prefork-threads >= %d (the widest "
                        "slice); the soft budget can only shrink, never grow past the "
                        "pool actually spawned\n", ncpu / 2);
    fflush(stderr);

    for (int w = 0; w < workers; w++) {
        if (socketpair(AF_UNIX, SOCK_STREAM, 0, sp[w]) != 0) { perror("socketpair"); return -1; }
        pid_t pid = fork();
        if (pid < 0) { perror("fork"); break; }
        if (pid == 0) {
            /* ── child ── */
            close(listen_fd);                 /* the parent is the only listener */
            for (int p2 = 0; p2 <= w; p2++) close(sp[p2][0]);
            g_conn_chan_fd = sp[w][1];
            g_conn_done_fd = sp[w][1];        /* same socketpair, other direction */
            cpu_set_t set; CPU_ZERO(&set);
            for (int c = w * per; c < (w + 1) * per && c < ncpu; c++) CPU_SET(c, &set);
            if (sched_setaffinity(0, sizeof(set), &set) != 0) perror("sched_setaffinity");
            qwen_threadpool_after_fork();     /* the inherited pool has no threads */
            qwen_set_threads(threads_per);
            fprintf(stderr, "prefork: worker %d pid %d cpus %d-%d threads %d\n",
                    w, (int)getpid(), w * per, w * per + per - 1, threads_per);
            int rc = (max_batch >= 2) ? qwen_tts_serve_batched(ctx, port, max_batch)
                                      : qwen_tts_serve_ex(ctx, port, 1);
            /* ── A prefork worker ends with _exit(), so NOTHING registered with atexit or
             * as a destructor runs here. On 2026-08-24 that silently voided three
             * instrumented passes at once: the shape census reported frames=0, POOL_STATS
             * reported dispatch=6 and the LD_PRELOAD yield tracer reported 15 — all of them
             * the PARENT's numbers, for a campaign the parent takes no part in. The counters
             * are dumped explicitly instead. _exit stays: the child must not flush stdio it
             * inherited nor run the parent's handlers. */
            qwen_worker_dump_counters();
#ifdef QWEN_ASAN
            /* The same _exit() that voided the counters voids LeakSanitizer, which runs
             * from atexit. Without this call a prefork worker - where ALL request-path
             * allocation happens - is never leak-checked, and a sanitizer run reports a
             * clean sheet on something it never examined. Recoverable variant: it prints
             * and returns, so the exit code the harness reads still comes from the run.
             * Compiled out of any non-sanitizer build, so the production binary is
             * untouched. */
            __lsan_do_recoverable_leak_check();
#endif
            _exit(rc == 0 ? 0 : 1);
        }
        kids[w] = pid;
        close(sp[w][1]);                      /* the parent keeps only its end */
    }

    struct sigaction sa = { .sa_handler = prefork_parent_sig };
    sigemptyset(&sa.sa_mask); sa.sa_flags = 0;
    sigaction(SIGINT, &sa, NULL);
    sigaction(SIGTERM, &sa, NULL);
    signal(SIGPIPE, SIG_IGN);

    /* ── the dispatcher ──
     * One poll over the worker channels plus, ONLY when somebody has capacity, the
     * listener. Dropping the listener out of the set when every worker is full is the
     * backpressure: the connection waits in the kernel backlog instead of being
     * refused, and nothing is queued in user space where it would age invisibly. */
    struct sigaction su = { .sa_handler = prefork_dump_sig };
    sigemptyset(&su.sa_mask); su.sa_flags = 0;
    sigaction(SIGUSR1, &su, NULL);

    struct pollfd *pfd = (struct pollfd *)calloc((size_t)workers + 1, sizeof(struct pollfd));
    if (!pfd) return -1;
    long long dispatched = 0;
    /* Time-weighted mean in-flight: the effective batch the workers actually saw, which
     * a sample-at-the-end would miss entirely. */
    double act_area = 0.0, act_time = 0.0;
    /* Per worker as well as globally: "worker 3 sat idle" has to be a number, not an
     * impression from htop. Time-weighted, because a sample at the end of a level
     * misses the shape of the wave entirely. */
    double *act_area_w = (double *)calloc((size_t)workers, sizeof(double));
    if (!act_area_w) return -1;
    struct timespec tprev; clock_gettime(CLOCK_MONOTONIC, &tprev);
    while (!g_prefork_stop) {
        int nf = 0, free_slots = 0;
        for (int w = 0; w < workers; w++) {
            if (kids[w] <= 0) continue;
            pfd[nf].fd = sp[w][0]; pfd[nf].events = POLLIN; pfd[nf].revents = 0;
            nf++;
            if (active[w] < cap) free_slots++;
        }
        if (nf == 0) break;
        int li = -1;
        if (free_slots > 0) {
            li = nf;
            pfd[nf].fd = listen_fd; pfd[nf].events = POLLIN; pfd[nf].revents = 0;
            nf++;
        }
        int r = poll(pfd, (nfds_t)nf, 1000);
        {
            struct timespec tn; clock_gettime(CLOCK_MONOTONIC, &tn);
            double dt = (double)(tn.tv_sec - tprev.tv_sec) +
                        (double)(tn.tv_nsec - tprev.tv_nsec) * 1e-9;
            tprev = tn;
            int tot = 0;
            for (int w = 0; w < workers; w++) {
                if (kids[w] <= 0) continue;
                tot += active[w];
                act_area_w[w] += (double)active[w] * dt;
            }
            act_area += (double)tot * dt; act_time += dt;
        }
        if (g_prefork_dump) {
            g_prefork_dump = 0;
            fprintf(stderr, "[prefork-stats] mean_inflight %.3f dispatched %lld rejected %lld ·",
                    act_time > 0 ? act_area / act_time : 0.0, dispatched, rejected);
            for (int w = 0; w < workers; w++)
                fprintf(stderr, " w%d[asg=%lld done=%lld act=%d B=%.2f]",
                        w, assigned[w], completed[w], active[w],
                        act_time > 0 ? act_area_w[w] / act_time : 0.0);
            if (elastic) {
                fprintf(stderr, " · slices");
                for (int w = 0; w < workers; w++)
                    fprintf(stderr, " %d-%d", cur[2 * w], cur[2 * w + 1]);
                fprintf(stderr, " replans=%lld", replans);
            }
            fprintf(stderr, "\n"); fflush(stderr);
            for (int w = 0; w < workers; w++) { assigned[w] = 0; completed[w] = 0; act_area_w[w] = 0.0; }
            dispatched = 0; rejected = 0; act_area = 0.0; act_time = 0.0;
        }
        if (r < 0) { if (errno == EINTR) continue; perror("poll"); break; }

        /* completions first: they free the slots the next accept will want */
        int idx = 0;
        for (int w = 0; w < workers; w++) {
            if (kids[w] <= 0) continue;
            struct pollfd *p = &pfd[idx++];
            if (!(p->revents & (POLLIN | POLLHUP | POLLERR))) continue;
            char buf[256];
            ssize_t n = read(sp[w][0], buf, sizeof buf);
            if (n > 0) {
                completed[w] += n;
                active[w] -= (int)n;
                if (active[w] < 0) active[w] = 0;
                if (elastic) {
                    elastic_plan(workers, ncpu, active, slice);
                    replans += elastic_apply(workers, ncpu, slice, kids, cur);
                }
            } else if (n == 0 || (n < 0 && errno != EINTR && errno != EAGAIN)) {
                fprintf(stderr, "prefork: worker %d (pid %d) channel closed\n", w, (int)kids[w]);
                close(sp[w][0]); kids[w] = -1; active[w] = 0;
            }
        }
        if (li < 0 || !(pfd[li].revents & POLLIN)) continue;

        struct sockaddr_in ca; socklen_t cl = sizeof(ca);
        int cfd = accept(listen_fd, (struct sockaddr *)&ca, &cl);
        if (cfd < 0) { if (errno == EINTR || errno == EAGAIN) continue; perror("accept"); continue; }

        /* least loaded, ties to the lowest index so a light load stays on few cores */
        int best = -1;
        for (int w = 0; w < workers; w++) {
            if (kids[w] <= 0 || active[w] >= cap) continue;
            if (best < 0 || active[w] < active[best]) best = w;
        }
        if (best < 0) {          /* cannot happen: the listener was only polled with capacity */
            rejected++;
            send_error(cfd, 503, "all workers at capacity");
            close(cfd);
            continue;
        }
        set_client_timeout(cfd);
        if (elastic) {
            /* Count this worker as busy FIRST, then re-plan and apply, and only then
             * hand over the descriptor: the child reads its mask when the fd arrives,
             * so applying afterwards would start the request on the old slice. */
            active[best]++;
            elastic_plan(workers, ncpu, active, slice);
            replans += elastic_apply(workers, ncpu, slice, kids, cur);
            active[best]--;
        }
        if (srv_send_fd(sp[best][0], cfd) != 0) {
            rejected++;
            close(cfd);
            continue;
        }
        close(cfd);              /* the child owns it now */
        active[best]++; assigned[best]++; dispatched++;
        /* QWEN_LIFE_TRACE: what the parent knew when it chose. The listener is only polled
         * while free_slots > 0 and `best` is picked among workers under cap, so by
         * construction a full worker cannot be chosen over an idle one — this line is here
         * to make that CHECKABLE rather than argued from the source. */
        if (getenv("QWEN_LIFE_TRACE")) {
            fprintf(stderr, "[DISP] seq=%lld w=%d free_slots_before=%d cap=%d act=", 
                    dispatched, best, free_slots, cap);
            for (int w = 0; w < workers; w++) fprintf(stderr, "%s%d", w ? "," : "", active[w]);
            fprintf(stderr, "\n");
        }
        if ((dispatched % 64) == 0) {
            fprintf(stderr, "prefork: dispatched %lld ·", dispatched);
            for (int w = 0; w < workers; w++)
                fprintf(stderr, " w%d[a=%d asg=%lld done=%lld]", w, active[w], assigned[w], completed[w]);
            fprintf(stderr, " rejected=%lld\n", rejected);
        }
    }

    for (int w = 0; w < workers; w++) if (kids[w] > 0) kill(kids[w], SIGTERM);
    int alive = 0;
    for (int w = 0; w < workers; w++) if (kids[w] > 0) alive++;
    while (alive > 0) {
        int status = 0;
        pid_t got = waitpid(-1, &status, 0);
        if (got > 0) { alive--; continue; }
        if (errno == EINTR) continue;
        break;
    }
    fprintf(stderr, "\nprefork: FINAL  dispatched=%lld rejected=%lld\n", dispatched, rejected);
    for (int w = 0; w < workers; w++)
        fprintf(stderr, "  worker %d: assigned=%lld completed=%lld still-active=%d\n",
                w, assigned[w], completed[w], active[w]);
    free(pfd); free(sp); free(kids); free(assigned); free(completed); free(active);
    free(act_area_w); free(slice); free(cur);
    close(listen_fd);
    return 0;
}
#else
int qwen_tts_serve_prefork(qwen_tts_ctx_t *ctx, int port, int workers,
                           int threads_per, int max_batch) {
    (void)workers; (void)threads_per;
    fprintf(stderr, "prefork: not supported on this platform, running one server\n");
    return max_batch >= 2 ? qwen_tts_serve_batched(ctx, port, max_batch)
                          : qwen_tts_serve_ex(ctx, port, 1);
}
#endif
