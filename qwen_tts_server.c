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
#include <ctype.h>
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
/* Hard ceiling, independent of any other setting: nothing longer is ever read. */
#define MAX_TTS_TEXT 8192
#define QWEN_STR2(x) #x
#define QWEN_STR(x) QWEN_STR2(x)

/* ── The input limit is DERIVED from the generation limit ────────────────────────
 * The two used to be unrelated, and the gap was most of the input space: 8192
 * characters were accepted while the 60 s cap could complete roughly 1900, so every
 * request between those two numbers was guaranteed to end in a timeout or an error.
 * A limit that lets a caller submit work the server cannot finish is not a limit.
 *
 * ⚠️ THE RATE BELOW IS PROVISIONAL AND MUST BE RE-DERIVED PER DEPLOYMENT.
 * It comes from a SINGLE observation -- 2000 characters produced 48 s of audio in 63.6 s
 * of compute on a 0.6B at one concurrent stream, i.e. ~0.024 s of audio and ~0.032 s of
 * compute per character, or ~31 characters per second of budget. A second measurement on
 * the same machine then refused 1000 characters at the same 60 s, which is twice the cost
 * per character, so the two disagree by 2x and the difference is almost certainly machine
 * load rather than anything about the text. One point is not a rate.
 *
 * What IS settled is the SHAPE: the input limit must be derived from the generation cap
 * rather than set beside it. Before, 8192 characters were accepted while the cap could
 * finish far fewer, so most of the accepted input space was guaranteed to fail. A limit
 * that lets a caller submit work the server cannot finish is not a limit.
 *
 * Until it is measured properly -- a length sweep on a quiet machine, at the concurrency
 * the deployment actually runs -- treat 30 as a placeholder that errs towards refusing,
 * and set --max-text-chars explicitly for anything that matters. */
#define QWEN_CHARS_PER_CAP_SECOND 30

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
/* Set by read_request when the declared body exceeded the buffer. Thread-local: one
 * connection per reader thread, and a shared flag would blame the wrong request. */
static _Thread_local int g_req_too_large;

static int read_request(int fd, char *buf, int buf_size) {
    g_req_too_large = 0;
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
                if (content_length > buf_size - 1) {
                    /* Do not parse a truncated body as if it were the whole request: a JSON
                     * object cut in half parses as "missing text" and the caller is told the
                     * wrong thing. Flag it so the router can answer 413. */
                    g_req_too_large = 1;
                    content_length = buf_size - 1;
                }
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

/* Escape a message for embedding in JSON. Without this a quote or a control byte from
 * an echoed field name breaks the envelope, and a client parsing our error gets a parse
 * failure instead of the reason -- which is also how a caller-controlled string turns
 * into a response-splitting primitive. Everything outside printable ASCII is escaped. */
static void json_escape(char *dst, size_t dstsz, const char *src) {
    size_t j = 0;
    for (const unsigned char *p = (const unsigned char *)src; *p && j + 8 < dstsz; p++) {
        switch (*p) {
            case '"':  if (j + 2 < dstsz) { dst[j++]='\\'; dst[j++]='"';  } break;
            case '\\': if (j + 2 < dstsz) { dst[j++]='\\'; dst[j++]='\\'; } break;
            case '\n': if (j + 2 < dstsz) { dst[j++]='\\'; dst[j++]='n';  } break;
            case '\r': if (j + 2 < dstsz) { dst[j++]='\\'; dst[j++]='r';  } break;
            case '\t': if (j + 2 < dstsz) { dst[j++]='\\'; dst[j++]='t';  } break;
            default:
                if (*p < 0x20 || *p > 0x7e) j += (size_t)snprintf(dst + j, dstsz - j, "\\u%04x", *p);
                else dst[j++] = (char)*p;
        }
    }
    dst[j < dstsz ? j : dstsz - 1] = '\0';
}

/* The OpenAI error envelope, which is an OBJECT and not a string. Their own clients read
 * error.message, so a flat {"error":"..."} makes the official SDKs report undefined --
 * vLLM shipped exactly that shape and had to fix it (vllm#12886). `param` names the
 * offending field when we know it, which is what turns a rejection into a one-line fix. */
static const char *api_error_type(int status) {
    if (status == 404) return "not_found_error";
    if (status == 429) return "rate_limit_error";
    if (status >= 500) return "api_error";
    return "invalid_request_error";
}

static void send_api_error(int fd, int status, const char *msg, const char *param) {
    char emsg[768], eparam[128], json[1200];
    json_escape(emsg, sizeof(emsg), msg ? msg : "");
    if (param && *param) {
        json_escape(eparam, sizeof(eparam), param);
        snprintf(json, sizeof(json),
                 "{\"error\":{\"message\":\"%s\",\"type\":\"%s\",\"param\":\"%s\",\"code\":null}}",
                 emsg, api_error_type(status), eparam);
    } else {
        snprintf(json, sizeof(json),
                 "{\"error\":{\"message\":\"%s\",\"type\":\"%s\",\"param\":null,\"code\":null}}",
                 emsg, api_error_type(status));
    }
    send_json(fd, status, json);
}

static void send_error(int fd, int status, const char *msg) {
    send_api_error(fd, status, msg, NULL);
}

/* ── Streaming response (chunked transfer encoding) ──────────────── */

typedef struct {
    int fd;
    int total_samples;
    float volume;   /* per-chunk gain (emotion/volume); 1.0 = no-op */
} stream_http_state_t;

/* ── QWEN_CANCEL_ON_DISCONNECT — stop generating for a request whose client has gone.
 * DEFAULT OFF: both arms of the A/B are then the SAME binary, and the OFF arm reproduces
 * the historical behaviour exactly. */
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

/* ── SERVICE STATE, shared between the readers and the scheduler ──────────────
 *
 * It exists for one reason: /v1/health has to tell the TRUTH. It used to answer a static
 * `{"status":"ok"}` — 200 even with the scheduler dead and the server draining 503s. A
 * load balancer decides where to send traffic from that answer, so health that lies is not
 * a cosmetic detail: it is the wrong foundation under any multi-process architecture, and
 * it makes things worse rather than better.
 *
 * The same counters are also the minimum needed to be diagnosable in production, where
 * queue depth and in-flight count otherwise exist only on stderr. The names deliberately
 * follow the ones vLLM made the de facto standard (num_requests_running /
 * num_requests_waiting), so an LLM-aware router or a Prometheus scrape finds them where it
 * expects them. */
typedef struct {
    atomic_int sched_alive;      /* 0 finche' lo scheduler non e' partito, 0 se e' morto */
    atomic_int running;          /* richieste attualmente in generazione */
    atomic_int waiting;          /* richieste in coda, non ancora ammesse */
    atomic_int admitted, done;
    atomic_int rejected_full;    /* coda piena  -> 503 */
    atomic_int rejected_stale;   /* scaduta in coda -> 503 */
    atomic_int timed_out;        /* exceeded the wall-clock cap while in service */
    int queue_max;               /* quante possono ASPETTARE oltre quelle in esecuzione */
    int slots;                   /* --batch-size: quante ne esegue insieme */
    int queue_timeout_ms;        /* 0 = nessuna scadenza */
    int max_request_ms;          /* 0 = no cap; enforced in sink_cancelled */
    int max_text_chars;          /* 0 = derive it from max_request_ms */
} server_state_t;

static server_state_t g_srv;

/* -1 = automatico (2x gli slot). Impostati da main.c prima di partire. */
static int g_cfg_max_queue = -1;
static int g_cfg_queue_timeout_ms = 0;
/* Default ON. A limit that ships disabled is not a limit -- the queue deadline shipped
 * at 0 for weeks for exactly that reason, and it is listed as an open item in its own
 * right. 60 s is chosen from measurement, not from taste: on the long bank (~24 s of
 * audio) the p95 request takes 41.5 s at C=8 and 32.2 s at C=6, so 60 s clears the
 * slowest legitimate traffic measured with ~45 % of headroom while cutting the worst
 * case from the ~11 minutes the token ceiling allows down to one minute.
 *
 * Raise it for deployments that legitimately synthesise several minutes in one request;
 * 0 disables it entirely. Both are one flag away, and the effective value is printed at
 * startup so nobody has to infer it. */
static int g_cfg_max_request_ms = 60000;
static int g_cfg_max_text_chars = 0;   /* 0 = derived, see srv_max_text_chars */

/* ── H4/H5: the strict serving profile ───────────────────────────────────────────
 * The CLI operator and an untrusted HTTP caller want OPPOSITE defaults, and the
 * forgiving behaviour is right for the CLI. Under --serve the default is strict:
 * an input the server cannot honour is REFUSED rather than silently replaced by
 * something else. Every rule stays switchable -- QWEN_SERVER_STRICT=0 or --no-strict
 * restores the permissive behaviour -- because a deployment may have a caller that
 * depends on it, and finding that out through a 400 in production is not the plan.
 *
 * The specific bug this exists to stop already happened once on this path: an
 * unresolvable speaker fell back to the DEFAULT voice, and the only symptom was that
 * the server rendered a different voice from the CLI for the same request. */
/* One rejection reason per in-flight request. Thread-local: the single-request path
 * runs one connection per thread, and a shared buffer would report another caller's
 * error to this one. */
static _Thread_local char g_req_err[256];
static int g_cfg_strict = 1;
void qwen_tts_server_set_strict(int on) { g_cfg_strict = on ? 1 : 0; }
static int qwen_server_strict(void) {
    static int v = -1;
    if (v < 0) {
        const char *e = getenv("QWEN_SERVER_STRICT");
        v = (e && *e) ? (*e != '0') : g_cfg_strict;
    }
    return v;
}

/* ── Well-formedness, before anything reads a field ─────────────────────────────
 * The field extractors scan for `"key"` and do not care what surrounds it, so a body
 * like {"text":"hi","evil"key":1} -- which is not JSON at all -- was accepted and
 * synthesised. Anything that reaches the extractors must first BE a JSON object.
 *
 * A full parser is not needed and not wanted here: this validates the grammar and
 * builds nothing, so there is no allocation for a caller to grow. Depth is bounded
 * because unbounded nesting is a stack-exhaustion primitive against anything that
 * later walks the document, and no legitimate request for this API nests at all.
 *
 * Returns 0 if `body` is a single well-formed JSON object, -1 otherwise. */
#define QWEN_JSON_MAX_DEPTH 16

static const char *js_ws(const char *p) {
    while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') p++;
    return p;
}
static const char *js_value(const char *p, int depth, const char **why);

static const char *js_string(const char *p, const char **why) {
    if (*p != '"') { *why = "expected a string"; return NULL; }
    p++;
    for (;;) {
        unsigned char ch = (unsigned char)*p;
        if (ch == '\0') { *why = "unterminated string"; return NULL; }
        if (ch == '"')  return p + 1;
        if (ch < 0x20)  { *why = "control character in string"; return NULL; }
        if (ch == '\\') {
            p++;
            switch (*p) {
                case '"': case '\\': case '/': case 'b': case 'f':
                case 'n': case 'r': case 't': p++; break;
                case 'u':
                    p++;
                    for (int i = 0; i < 4; i++, p++)
                        if (!isxdigit((unsigned char)*p)) { *why = "bad \\u escape"; return NULL; }
                    break;
                default: *why = "bad escape in string"; return NULL;
            }
            continue;
        }
        p++;
    }
}

static const char *js_number(const char *p, const char **why) {
    const char *start = p;
    if (*p == '-') p++;
    if (*p == '0') p++;
    else if (isdigit((unsigned char)*p)) { while (isdigit((unsigned char)*p)) p++; }
    else { *why = "bad number"; return NULL; }
    if (*p == '.') { p++; if (!isdigit((unsigned char)*p)) { *why = "bad number"; return NULL; }
                     while (isdigit((unsigned char)*p)) p++; }
    if (*p == 'e' || *p == 'E') {
        p++; if (*p == '+' || *p == '-') p++;
        if (!isdigit((unsigned char)*p)) { *why = "bad exponent"; return NULL; }
        while (isdigit((unsigned char)*p)) p++;
    }
    /* Reject a number so long it cannot be a real parameter: it is never a legitimate
     * request and it is a cheap way to make a downstream converter work hard. */
    if (p - start > 40) { *why = "number too long"; return NULL; }
    return p;
}

static const char *js_value(const char *p, int depth, const char **why) {
    if (depth > QWEN_JSON_MAX_DEPTH) { *why = "nesting too deep"; return NULL; }
    p = js_ws(p);
    switch (*p) {
        case '"': return js_string(p, why);
        case '{': {
            p = js_ws(p + 1);
            if (*p == '}') return p + 1;
            for (;;) {
                p = js_ws(p);
                p = js_string(p, why); if (!p) return NULL;
                p = js_ws(p);
                if (*p != ':') { *why = "expected ':' after a key"; return NULL; }
                p = js_value(p + 1, depth + 1, why); if (!p) return NULL;
                p = js_ws(p);
                if (*p == ',') { p++; continue; }
                if (*p == '}') return p + 1;
                *why = "expected ',' or '}'"; return NULL;
            }
        }
        case '[': {
            p = js_ws(p + 1);
            if (*p == ']') return p + 1;
            for (;;) {
                p = js_value(p, depth + 1, why); if (!p) return NULL;
                p = js_ws(p);
                if (*p == ',') { p++; continue; }
                if (*p == ']') return p + 1;
                *why = "expected ',' or ']'"; return NULL;
            }
        }
        case 't': if (!strncmp(p, "true", 4))  return p + 4; break;
        case 'f': if (!strncmp(p, "false", 5)) return p + 5; break;
        case 'n': if (!strncmp(p, "null", 4))  return p + 4; break;
        default:  return js_number(p, why);
    }
    *why = "unexpected token";
    return NULL;
}

static int json_validate_object(const char *body, char *err, size_t errsz) {
    const char *why = "malformed JSON";
    const char *p = js_ws(body ? body : "");
    if (*p != '{') {
        snprintf(err, errsz, "body must be a JSON object");
        return -1;
    }
    p = js_value(p, 0, &why);
    if (!p) { snprintf(err, errsz, "malformed JSON: %s", why); return -1; }
    p = js_ws(p);
    if (*p) { snprintf(err, errsz, "malformed JSON: trailing data after the object"); return -1; }
    return 0;
}

/* Name the limit that ACTUALLY bound. Two different ceilings can produce the same
 * number, and telling a caller about the wrong one sends them to change the wrong knob. */
static void srv_text_limit_reason(char *err, size_t errsz, size_t got, int lim) {
    long by_prompt = (long)qwen_tts_batch_max_prompt() * 7 / 2;
    long by_time   = (g_srv.max_request_ms > 0)
                   ? (long)(g_srv.max_request_ms / 1000) * QWEN_CHARS_PER_CAP_SECOND : -1;
    if (lim >= MAX_TTS_TEXT)
        snprintf(err, errsz, "text too long: %zu characters, maximum %d", got, lim);
    else if (by_time >= 0 && by_time < by_prompt)
        snprintf(err, errsz, "text too long: %zu characters, maximum %d - that is what this "
                             "server can finish within its %.0f s generation limit "
                             "(--max-request-seconds)", got, lim, g_srv.max_request_ms / 1000.0);
    else
        snprintf(err, errsz, "text too long: %zu characters, maximum %d - a longer prompt does "
                             "not fit a batch slot's %d-token budget (QWEN_BATCH_MAX_PROMPT)",
                 got, lim, qwen_tts_batch_max_prompt());
}

/* ── H4: reject a request that names a field this server does not implement ──────
 * A silently ignored field is a caller bug that never surfaces: the client believes it
 * asked for something, the server never did it, and both sides think they agree. The
 * list is derived from the fields the parsers actually read, so it cannot drift away
 * from the implementation without this check failing first.
 *
 * Only the TOP LEVEL is walked, and only in strict mode. Nested objects belong to a
 * schema this server does not define. */
/* The OpenAI speech API's own fields come first, because a client written against it
 * must work here unchanged: input, model, voice, response_format, speed, instructions,
 * stream_format. The rest are this engine's extensions, and the two spellings of the
 * same idea are both accepted rather than one of them silently ignored --
 * speed/rate and instructions/instruct. */
static const char *const g_known_fields[] = {
    /* OpenAI speech API */
    "input", "model", "voice", "response_format", "speed", "instructions",
    "stream_format", "stream",
    /* engine extensions */
    "chunk_frames", "emotion", "instruct", "language", "max_new_tokens", "rate",
    "rep_penalty", "seed", "speaker", "temperature", "text", "top_k", "top_p",
    "voice_design", "volume", NULL
};

/* response_format: this engine emits PCM only. Accepting "mp3" and quietly returning a
 * WAV would be the same silent substitution as the speaker fallback -- the caller would
 * hand our bytes to an mp3 decoder and get noise. Refused by name, with the list. */
static int check_response_format(const char *body, char *err, size_t errsz) {
    char *f = json_extract_string(body, "response_format");
    if (!f) return 0;
    int ok = !strcasecmp(f, "wav") || !strcasecmp(f, "pcm");
    if (!ok) snprintf(err, errsz, "response_format '%.16s' is not supported - this server "
                                  "emits 'wav' (default) or 'pcm'", f);
    free(f);
    return ok ? 0 : -1;
}

static int reject_unknown_fields(const char *body, char *err, size_t errsz) {
    if (!qwen_server_strict() || !body) return 0;
    int depth = 0, in_str = 0, esc = 0;
    const char *p = body;
    for (; *p; p++) {
        if (esc) { esc = 0; continue; }
        if (in_str) {
            if (*p == '\\') { esc = 1; continue; }
            if (*p == '"') { in_str = 0; }
            continue;
        }
        if (*p == '"') {
            const char *k = p + 1;
            in_str = 1;
            if (depth != 1) continue;
            /* a key is a string at depth 1 followed by ':' */
            const char *q = k; int e2 = 0;
            while (*q && (e2 || *q != '"')) { e2 = (!e2 && *q == '\\'); q++; }
            if (*q != '"') continue;
            const char *c = q + 1;
            while (*c == ' ' || *c == '\t' || *c == '\n' || *c == '\r') c++;
            if (*c != ':') continue;
            size_t klen = (size_t)(q - k);
            int known = 0;
            for (int i = 0; g_known_fields[i]; i++)
                if (strlen(g_known_fields[i]) == klen && !strncmp(g_known_fields[i], k, klen)) { known = 1; break; }
            if (!known) {
                snprintf(err, errsz, "unknown field '%.*s' - this server implements: "
                                     "text, speaker, language, seed, temperature, top_k, "
                                     "top_p, rep_penalty, instruct, emotion, volume, rate",
                         (int)(klen > 48 ? 48 : klen), k);
                return -1;
            }
            p = q; in_str = 0;
            continue;
        }
        if (*p == '{' || *p == '[') depth++;
        else if (*p == '}' || *p == ']') depth--;
    }
    return 0;
}

/* Resolve a requested speaker. Returns 0 on success. On failure in strict mode it
 * fills `err` with a message that tells the caller what to fix -- the valid names are
 * already public on /v1/speakers, so naming them costs nothing and saves a round trip. */
static int resolve_speaker_checked(qwen_tts_ctx_t *ctx, const char *name, int *out_id,
                                   char *err, size_t errsz) {
    int sid = qwen_tts_resolve_speaker(ctx, name);
    if (sid >= 0) { *out_id = sid; return 0; }
    if (qwen_server_strict()) {
        snprintf(err, errsz, "unknown speaker '%.64s' for this model - see /v1/speakers "
                             "for the names this checkpoint declares", name);
        return -1;
    }
    fprintf(stderr, "[server] unknown speaker '%s' - falling back to the default voice "
                    "(strict mode would refuse this)\n", name);
    return 0;
}

void qwen_tts_server_set_limits(int max_queue, int queue_timeout_ms) {
    g_cfg_max_queue = max_queue;
    g_cfg_queue_timeout_ms = queue_timeout_ms;
}

/* ── H1: a wall-clock ceiling on time IN SERVICE, per request ────────────────────
 * The only bound today is the token cap, and at 12.5 Hz that is ~11 minutes of audio.
 * With 6-8 slots, one caller sending an enormous text holds a channel that far, which
 * is a denial of service that needs no malice. This bounds it.
 *
 * It is deliberately measured from ADMISSION, not from arrival: queue time is already
 * bounded separately by queue_timeout_ms, and charging a request for the queue it did
 * not cause would make the two limits interact in a way nobody could reason about.
 *
 * 0 disables. Env QWEN_MAX_REQUEST_S overrides the CLI so a running deployment can be
 * bounded without a rebuild. */
void qwen_tts_server_set_max_request_ms(int ms) { g_cfg_max_request_ms = ms; }
void qwen_tts_server_set_max_text_chars(int chars) { g_cfg_max_text_chars = chars; }

/* Resolve the cap ONCE, for every serving mode. It used to be resolved inside the batched
 * scheduler's setup, which meant `--serve` without a batch size ran with no cap at all and
 * said nothing about it -- a safety limit that silently does not apply in one mode is the
 * failure this project keeps meeting, so it is initialised where every mode passes. */
/* Effective text ceiling: the smaller of the hard buffer limit and what the generation
 * cap can actually finish. With the cap disabled only the hard limit applies. */
static int srv_max_text_chars(void) {
    if (g_srv.max_text_chars > 0) return g_srv.max_text_chars;
    /* Structural first: a prompt longer than a batch slot's budget CANNOT be served, and
     * that limit is a token count, not a rate. Measured on this tokenizer, 4000 characters
     * of ordinary prose became 846 text tokens (~4.7 chars/token); 3.5 is used instead so
     * the edge refuses slightly early rather than letting a request through to be rejected
     * at admission. This replaces the earlier derivation from the generation cap, whose
     * rate was never measured cleanly. */
    long by_prompt = (long)qwen_tts_batch_max_prompt() * 7 / 2;
    long lim = by_prompt;
    /* Then the time budget, if it binds sooner. Kept deliberately generous because its
     * rate is still unmeasured: it must not be the limit that fires in normal use. */
    if (g_srv.max_request_ms > 0) {
        long by_time = (long)(g_srv.max_request_ms / 1000) * QWEN_CHARS_PER_CAP_SECOND;
        if (by_time < lim) lim = by_time;
    }
    if (lim < 200)          lim = 200;          /* never refuse a normal sentence */
    if (lim > MAX_TTS_TEXT) lim = MAX_TTS_TEXT;
    return (int)lim;
}

static void srv_init_request_cap(void) {
    g_srv.max_request_ms = g_cfg_max_request_ms;
    const char *e = getenv("QWEN_MAX_REQUEST_S");
    if (e && *e) { double v = atof(e); if (v >= 0) g_srv.max_request_ms = (int)(v * 1000.0); }
    g_srv.max_text_chars = g_cfg_max_text_chars;
    { const char *e = getenv("QWEN_MAX_TEXT_CHARS");
      if (e && *e) { int v = atoi(e); if (v > 0) g_srv.max_text_chars = v; } }
    if (g_srv.max_request_ms > 0)
        fprintf(stderr, "[serve] per-request generation cap: %.0f s -> text limit %d characters "
                        "(--max-request-seconds N / --max-text-chars N; 0 disables the cap)\n",
                g_srv.max_request_ms / 1000.0, srv_max_text_chars());
    else
        fprintf(stderr, "[serve] per-request generation cap: DISABLED - one caller can hold a "
                        "slot for as long as the token ceiling allows; text limit %d characters\n",
                srv_max_text_chars());
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
             "\"queue_max\":%d,\"queue_timeout_ms\":%d,\"max_request_ms\":%d,"
             "\"max_text_chars\":%d,"
             "\"admitted\":%d,\"done\":%d,"
             "\"rejected_queue_full\":%d,\"rejected_queue_timeout\":%d,"
             "\"timed_out\":%d}",
             alive ? "ok" : "unavailable", alive ? "running" : "down",
             atomic_load(&g_srv.running), waiting,
             g_srv.queue_max, g_srv.queue_timeout_ms, g_srv.max_request_ms,
             srv_max_text_chars(),
             atomic_load(&g_srv.admitted), atomic_load(&g_srv.done),
             atomic_load(&g_srv.rejected_full), atomic_load(&g_srv.rejected_stale),
             atomic_load(&g_srv.timed_out));
    /* 503 when the scheduler is gone: that is the signal a load balancer uses to take
     * this backend out of rotation instead of continuing to send it calls. */
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
    if (json_validate_object(body, g_req_err, sizeof(g_req_err))) { free(text); return NULL; }
    if (reject_unknown_fields(body, g_req_err, sizeof(g_req_err))) { free(text); return NULL; }
    if (check_response_format(body, g_req_err, sizeof(g_req_err))) { free(text); return NULL; }
    { double sp = json_extract_number(body, "speed", 1.0);
      if (sp < 0.25 || sp > 4.0) {
          snprintf(g_req_err, sizeof(g_req_err),
                   "speed %.3g out of range - allowed 0.25 to 4.0", sp);
          free(text); return NULL;
      } }
    if ((int)strlen(text) > srv_max_text_chars()) {
        int lim = srv_max_text_chars();
        srv_text_limit_reason(g_req_err, sizeof(g_req_err), strlen(text), lim);
        free(text);
        return NULL;
    }

    char *speaker = json_extract_string(body, "speaker");
    if (!speaker) speaker = json_extract_string(body, "voice");
    if (speaker) {
        /* qwen_tts_resolve_speaker, NOT qwen_tts_speaker_id: the latter knows only the 9
         * hardcoded CustomVoice presets and returns -1 for every voice of a finetuned
         * pool — and -1 was then silently dropped, so a request for a pool voice was
         * served by the DEFAULT slot. Measured: 98% language identification accuracy from the CLI vs
         * 14.5% from the server, same model/text/seed, because the server was rendering
         * a different voice. Same class of silent failure as PLAN fact F9, on the
         * serving path, where nobody had looked. */
        int sid = ctx->speaker_id;
        int bad = resolve_speaker_checked(ctx, speaker, &sid, g_req_err, sizeof(g_req_err));
        free(speaker);
        if (bad) { free(text); return NULL; }
        ctx->speaker_id = sid;
    }

    char *language = json_extract_string(body, "language");  /* kept for the emotion resolver below */
    if (language) {
        int lid = qwen_tts_language_id(language);
        if (lid >= 0) ctx->language_id = lid;
    }

    /* Instruct (1.7B only) */
    free(ctx->instruct);
    ctx->instruct = json_extract_string(body, "instruct");
    /* OpenAI calls it "instructions". Same field, both spellings honoured. */
    if (!ctx->instruct) ctx->instruct = json_extract_string(body, "instructions");

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
    /* OpenAI calls it "speed", range 0.25-4.0. Same meaning as our "rate": whichever
     * the caller sent is used, and an out-of-range value is refused rather than clamped
     * in silence, because a clamp makes the caller believe it got what it asked for. */
    float req_rate = (float)json_extract_number(body, "rate",
                          json_extract_number(body, "speed", 1.0));
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
    g_req_err[0] = '\0';
    char *text = parse_tts_request(ctx, body, &volume, &rate);
    if (!text) {
        send_error(fd, 400, g_req_err[0] ? g_req_err
                                         : "missing, empty, or oversized 'text' (max "
                                           QWEN_STR(MAX_TTS_TEXT) " characters)");
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
        send_error(fd, 400, g_req_err[0] ? g_req_err
                                         : "missing, empty, or oversized 'text' (max "
                                           QWEN_STR(MAX_TTS_TEXT) " characters)");
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
/* ── HTTP hygiene, shared by both routers ───────────────────────────────────────────
 * This server does one thing: JSON in, audio out. Everything else is refused with the
 * status that says WHY, because a caller that gets 404 for a wrong method, or "missing
 * text" for a form upload, has been told nothing it can act on.
 *
 * Refused here: any method other than the one a path implements (405, with Allow), any
 * body that is not JSON -- form encodings, multipart uploads, raw text -- (415), a body
 * too large for the read buffer (413), and anything that is not a JSON object (400).
 *
 * Returns 1 when it has already answered and the caller must stop. */
static int http_precheck(int fd, const char *method, const char *path,
                         const char *headers, const char *body) {
    struct { const char *path; const char *allow; } ROUTES[] = {
        { "/v1/health",       "GET"  }, { "/v1/speakers",    "GET"  },
        { "/v1/tts",          "POST" }, { "/v1/tts/stream",  "POST" },
        { "/v1/audio/speech", "POST" },
    };
    const char *allow = NULL;
    for (size_t i = 0; i < sizeof(ROUTES)/sizeof(ROUTES[0]); i++)
        if (!strcmp(path, ROUTES[i].path)) { allow = ROUTES[i].allow; break; }
    if (!allow) return 0;                       /* unknown path: the router answers 404 */
    if (strcmp(method, allow) != 0) {
        /* One response path, the one already proven to work. The Allow header is what
         * makes 405 actionable, so it is stated in the message too rather than lost. */
        char m[96];
        snprintf(m, sizeof(m), "method not allowed - %s takes %s", path, allow);
        send_error(fd, 405, m);
        return 1;
    }
    if (strcmp(allow, "POST") != 0) return 0;   /* GET endpoints take no body */

    if (g_req_too_large) {
        send_error(fd, 413, "request body too large");
        return 1;
    }
    /* Content-Type: only JSON. Naming the offending type is what turns a rejection into
     * something the caller can fix in one edit. */
    const char *ct = headers ? strcasestr(headers, "Content-Type:") : NULL;
    if (ct) {
        ct += strlen("Content-Type:");
        while (*ct == ' ' || *ct == '\t') ct++;
        if (!strcasestr(ct, "application/json") ||
            (size_t)(strcspn(ct, "\r\n")) == 0) {
            char m[200]; int n = (int)strcspn(ct, ";\r\n");
            if (n > 80) n = 80;
            snprintf(m, sizeof(m), "unsupported Content-Type '%.*s' - this endpoint takes "
                                   "application/json only (no form data, no multipart, "
                                   "no file upload)", n, ct);
            send_error(fd, 415, m);
            return 1;
        }
    }
    const char *b = body ? body : "";
    while (*b == ' ' || *b == '\t' || *b == '\r' || *b == '\n') b++;
    if (*b != '{') {
        send_error(fd, 400, *b ? "body is not a JSON object"
                               : "empty body - expected a JSON object");
        return 1;
    }
    return 0;
}

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

    if (strcmp(method, "OPTIONS") != 0 && http_precheck(client_fd, method, path, buf, body)) {
        free(buf); srv_conn_close(client_fd); return;
    }

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
    /* Provenance at the top of the server log: when a harness collects stderr, the first
     * line of the artifact says WHICH binary and WHICH flags the numbers came from. Without
     * it two runs are not comparable, and there is no way to find that out afterwards. */
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
    int timed_out;                /* H1: exceeded the wall-clock cap while in service */
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
/* Returns 1 if queued, 0 if the queue is FULL (the caller must refuse).
 *
 * WHY THERE IS A CAP. This queue used to be unbounded: the fourth request was queued and
 * the client waited forever — no 503, no deadline. That is not a performance problem, it is
 * a CONTRACT problem: overload showed up as silence, and a caller can handle a refusal but
 * cannot handle a service that never answers. It is the same hole vLLM still has open
 * (issue #18826: "the queue can grow indefinitely... until OOM").
 *
 * The cap is NOT meant to refuse under normal conditions: it is 2x the slots, that is, at
 * most two generation times of waiting. A longer queue does not "absorb a spike", it
 * accumulates a backlog nobody wants any more by the time you serve it. */
static int jq_push(job_queue_t *q, batch_job_t *j) {
    j->next = NULL;
    j->enq_ms = srv_now_ms();
    if (getenv("QWEN_TTFA_TRACE"))
        qwen_admit_probe_read(&j->enq_adm_seq, &j->enq_adm_ts, &j->enq_last_iter_ms);
    pthread_mutex_lock(&q->mtx);
    /* THE CAP IS ON THE TOTAL IN THE SYSTEM, NOT ON THE QUEUE LENGTH.
     * `cap` is how many may WAIT beyond those running, so the condition is
     *      running + waiting < slots + cap
     * The first version compared only `count >= cap` and was broken in the case that
     * mattered most: the queue is the ONLY way into the scheduler — even on an idle machine
     * a request passes through it and the scheduler picks it up — so with cap=0 the
     * condition was always true and NOTHING ever got in. Measured: `--max-queue 0` refused
     * 56 requests out of 56 and served none. The test caught it on the first run, which is
     * exactly why it exists. */
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
                             qwen_batch_req_t *req, int *needs_single,
                             char *err, size_t errsz) {
    if (err && errsz) err[0] = '\0';
    *needs_single = 0;
    char *text = json_extract_string(body, "text");
    if (!text) text = json_extract_string(body, "input");
    if (json_validate_object(body, err, errsz)) { free(text); return NULL; }
    if (reject_unknown_fields(body, err, errsz)) { free(text); return NULL; }
    if (check_response_format(body, err, errsz)) { free(text); return NULL; }
    { double sp = json_extract_number(body, "speed", 1.0);
      if (sp < 0.25 || sp > 4.0) {
          snprintf(err, errsz, "speed %.3g out of range - allowed 0.25 to 4.0", sp);
          free(text); return NULL;
      } }
    if (!text || text[0] == '\0') {
        snprintf(err, errsz, "missing or empty 'text'");
        free(text); return NULL;
    }
    if ((int)strlen(text) > srv_max_text_chars()) {
        int lim = srv_max_text_chars();
        srv_text_limit_reason(err, errsz, strlen(text), lim);
        free(text); return NULL;
    }

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
        int sid = req->speaker_id;
        int bad = resolve_speaker_checked(ctx, speaker, &sid, err, errsz);
        free(speaker);
        if (bad) { free(text); return NULL; }
        req->speaker_id = sid;
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

        if (strcmp(method, "OPTIONS") != 0 && http_precheck(fd, method, path, buf, body)) {
            srv_conn_close(fd); free(buf); continue;
        }
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
            char rerr[256] = {0};
            char *text = parse_batch_req(ra->ctx, ra->def_speaker_id, ra->def_language_id, body, &j->req, &needs_single, rerr, sizeof(rerr));
            if (!text) {
                send_error(fd, 400, rerr[0] ? rerr
                                            : "missing, empty, or oversized 'text' (max "
                                              QWEN_STR(MAX_TTS_TEXT) " characters)");
                srv_conn_close(fd); free(j); free(buf); continue;
            }
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
    /* WAIT DEADLINE. A request that has waited beyond the budget will never be served in
     * time: delivering it late audio is worse than telling it no, because meanwhile it has
     * occupied a slot that would have served a request still worth serving. It is the same
     * binary-viability idea streaming uses, applied to ADMISSION: below the threshold the
     * service is good, above it is of no use to anyone. */
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
    if (!j) return 0;
    /* H1. Checked BEFORE the disconnect gate and independently of it: a client that
     * waits patiently is exactly the case this bounds, and cancel-on-disconnect being
     * off must not disable it. Stopping one slot is the same mechanism cancellation
     * uses, whose K4 gate proves the other rows of the batch stay byte-identical. */
    if (!j->timed_out && g_srv.max_request_ms > 0 && j->t_admit > 0.0 &&
        srv_now_ms() - j->t_admit > (double)g_srv.max_request_ms) {
        j->timed_out = 1;
        if (j->t_cancel_stop == 0.0) j->t_cancel_stop = srv_now_ms();
        atomic_fetch_add(&g_srv.timed_out, 1);
        fprintf(stderr, "[server] request seed=%u exceeded the %d ms service cap - stopping it\n",
                j->life_seed, g_srv.max_request_ms);
    }
    if (j->timed_out) return 1;
    if (!qwen_cancel_on_disconnect()) return 0;
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
            j->timed_out ? " state=TIMEOUT" :
            j->client_gone ? " state=CANCELLED" : " state=COMPLETED");
}

/* on_reject: the driver could not admit this request. Previously this arrived as
 * on_done(NULL,0) and the HTTP layer answered 500 "generation failed" -- a refusal
 * reported as an internal error, with nothing the caller could act on. */
static void sink_on_reject(void *ud, void *tag, const char *reason) {
    sink_ctx_t *sc = (sink_ctx_t *)ud;
    batch_job_t *j = (batch_job_t *)tag;
    char m[220];
    /* The character figure must come from the budget that REFUSED this request, not from
     * whatever text limit happens to be configured -- quoting the latter told a caller
     * "about 8192 characters" while refusing 4000. */
    snprintf(m, sizeof(m),
             "%s - this server accepts at most %d prompt tokens per request "
             "(roughly %ld characters); split the text or raise QWEN_BATCH_MAX_PROMPT",
             reason ? reason : "request rejected",
             qwen_tts_batch_max_prompt(), (long)qwen_tts_batch_max_prompt() * 7 / 2);
    if (j->is_stream && j->header_sent) send_chunked_end(j->fd);
    else send_api_error(j->fd, 400, m, "text");
    fprintf(stderr, "[server] rejected seed=%u: %s\n", j->life_seed, reason ? reason : "?");
    srv_conn_close(j->fd);
    job_free(j);
    sc->done++;
    atomic_fetch_add(&g_srv.done, 1);
    atomic_fetch_sub(&g_srv.running, 1);
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
    } else if (j->timed_out && (!samples || n_samples <= 0)) {
        /* A timeout that produced nothing is not "generation failed": the caller can act
         * on the difference. 503 rather than 5xx-generic, and the cap is named so the
         * fix -- shorter text, or a longer cap agreed with the operator -- is obvious. */
        char m[160];
        snprintf(m, sizeof(m),
                 "request exceeded the server's %d ms generation limit and was stopped",
                 g_srv.max_request_ms);
        send_error(j->fd, 503, m);
        free(samples);
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
        .on_reject = sink_on_reject,
    };
    atomic_store(&g_srv.sched_alive, 1);
    int rc = qwen_tts_serve_continuous(sa->ctx, sa->max_batch, &sink);
    /* From here on /v1/health has to tell the truth: either this is an orderly shutdown or
     * the scheduler is dead — in both cases this backend serves no more requests, and a load
     * balancer needs to know BEFORE sending it another one. */
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
 * ear on 2026-08-18 (judged clean) and it applies where the memory is money: a
 * rented box, where RSS decides how many instances fit. The CLI keeps the old
 * behaviour so `make test-golden` stays a stable reference rather than becoming a
 * moving one.
 *
 * setenv with overwrite=0: an explicit QWEN_PREFILL_QUANT=0 from the operator still
 * wins. And it runs BEFORE the pre-warm, whose first prefill is what triggers the
 * release. */
/* The cross-slot batched decoder: on by default on the server, and only there.
 *
 * Batching the decoder improved both columns at once — at c=4, throughput +8.1 % and
 * first-audio p50 -8.9 %, p95 -6.1 % — which made it the only change that day with no
 * trade-off to declare. And the audio does not move: 81 samples out of 161,280 differ by
 * ONE LSB in 32768, correlation 1.00000000, and a listening pass judged both arms
 * indistinguishable.
 *
 * On the server only, because with a single slot there is nothing to batch: the CLI would
 * gain nothing and the golden reference stays still.
 *
 * The decoder THREAD (QWEN_DECODER_THREAD) is NOT enabled here: that one takes the decode
 * off the critical path but contends for cores, and on our bench first-audio latency did
 * not improve. They are two different levers and each needs its own measurement. */
static void server_default_decoder_batch(qwen_tts_ctx_t *ctx) {
    (void)ctx;
    if (getenv("QWEN_SERVER_NO_DECODER_BATCH")) return;
    setenv("QWEN_DECODER_BATCH", "1", 0);      /* un QWEN_DECODER_BATCH=0 esplicito vince */
    fprintf(stderr, "[serve] batched speech decoder ON by default (one pass over the decoder "
                    "weights for all active slots) — QWEN_DECODER_BATCH=0 to opt out\n");
}

/* ⛔ QUANTIZED PREFILL IS **NOT** A DEFAULT — measured on a finetuned checkpoint, and it
 * is the reason this function no longer turns anything on.
 *
 * WHAT HAPPENED. The lever was validated by ear on base models in English, where it is
 * harmless: the audio is clean, the memory saving is real (4.0 GB on the 1.7B int8) and
 * the prefill gets faster. On a finetuned pool checkpoint with a mixed 4-bit map, one
 * pool voice, same texts and same seeds, six clips per arm:
 *
 *     prefill quant OFF   language identification accuracy 96.3% mean, worst clip 86.1%
 *     prefill quant ON    language identification accuracy 38.0% mean, three clips at 0.0 / 1.4 / 11.7%
 *
 * The failure mode is the one this project keeps meeting: the audio stays clean, the
 * duration stays normal, nothing rasps — and the model DRIFTS TOWARDS THE BASE LANGUAGE,
 * losing the accent the finetune exists for. No signal-level metric sees it; only a
 * language-identification accuracy check does. The isolation was clean: an arm with the batched decoder off instead scored
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
                        "language identification accuracy 96%% -> 38%% when measured. Base models only.\n");
    } else {
        fprintf(stderr, "[serve] quantized prefill OFF (default: it loses the accent on "
                        "finetunes) — QWEN_PREFILL_QUANT=1 to opt in on a base model\n");
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
    /* Third serving entry, and the one --batch-size >= 2 reaches. All three must declare:
     * a driver cannot know which door main.c picked from the command line alone, and a
     * path that stays silent looks exactly like a flag that never arrived. */
    qwen_provenance_report(stderr);
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
    /* ── THE QUEUE IS SIZED BY THE LATENCY BUDGET, NOT BY THE SLOT COUNT ──────────
     *
     * The cap used to be 2x the slots. Measured with realistic traffic and bursts of 10:
     * with 3 slots and ~39 s of work per slot under a burst, a queue 6 deep is worth up to
     * ~78 s of waiting — and indeed the first-audio p95 of the NORMAL arrivals shot up to
     * **28 seconds**, because an ordinary call ended up behind five requests from the
     * burst. The p50 stayed at 418 ms: the server was not slow, the queue was deeper than
     * the budget.
     *
     * A QUEUE DEEPER THAN THE LATENCY BUDGET DOES NOT ABSORB A SPIKE: it produces answers
     * nobody wants any more. It is streaming's binary-viability idea applied to admission:
     * below the threshold the service is good, above it is of no use to anyone, and serving
     * late is worse than saying no — because meanwhile that slot would have served a
     * request still worth serving.
     *
     * And there is an ARCHITECTURAL reason stronger than the latency one: if there is a
     * load balancer above, the queue belongs THERE, not here. Harchol-Balter (SIGMETRICS
     * 2009) shows that a shared central queue IS an M/GI/n, whereas per-server queues
     * commit a request to a worker before knowing which one will free up first. HAProxy is
     * built that way: `maxconn` on the server says how many it may RUN, the balancer holds
     * the queue, and past `maxqueue` it redistributes to other servers. A worker that
     * queues steals from the balancer the one piece of information it could do better with.
     *
     * So:  0 = NO QUEUE, an immediate 503 when the slots are full (behind a load balancer
     *          this is the right configuration)
     *      N = at most N waiting
     *     <0 = automatic: 1, a single grace position for the request that arrives a few
     *          hundred ms before a slot frees up
     * The old unbounded behaviour — the one where the fourth request waited forever without
     * an answer — is reachable ONLY with QWEN_QUEUE_UNBOUNDED=1, because it is the defect,
     * not a configuration. */
    jq.cap = (g_cfg_max_queue >= 0) ? g_cfg_max_queue : 1;
    if (getenv("QWEN_QUEUE_UNBOUNDED")) {
        jq.cap = -1;
        fprintf(stderr, "[serve] WARNING QWEN_QUEUE_UNBOUNDED: the queue is UNBOUNDED — an "
                        "excess request will wait without limit and without an error. This is "
                        "the old behaviour, kept only for A/B comparison.\n");
    }
    g_srv.queue_max = jq.cap;
    g_srv.slots = max_batch;
    g_srv.queue_timeout_ms = g_cfg_queue_timeout_ms;
    srv_init_request_cap();
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
    /* Declare build and flags at the TOP of the threaded path too. print_banner runs just
     * before the accept loop -- after prewarm, after the socket exists -- so a driver that
     * waits for the port could assert against a log the engine had not written yet. And
     * --prefork 1 reaches this function, not the prefork one (main.c dispatches on > 1),
     * which is how a one-worker topology looked like a missing flag. */
    qwen_provenance_report(stderr);
    srv_init_request_cap();
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
    srv_init_request_cap();
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

    /* Provenance first, in the PREFORK path too, and BEFORE the listening socket
     * exists: a driver that waits for the port can then trust that the declaration
     * is already in the log. It used to be printed only by the two threaded accept
     * loops, so every benchmark on the production topology carried a log with no
     * build line and no record of the flags the numbers came from. */
    qwen_provenance_report(stderr);
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
