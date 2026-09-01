/* qwen_tts_server.c - Minimal HTTP server for Qwen3-TTS */
#ifdef __linux__
#ifndef _GNU_SOURCE
#define _GNU_SOURCE 1
#endif
#endif
#include "qwen_tts_server.h"
#include "qwen_tts_kernels.h"
#include <dlfcn.h>
#include "qwen_tts.h"
#include "qwen_tts_thread.h"
#include "qwen_tts_emotion.h"
#include "qwen_tts_compose.h"
#include "qwen_tts_audio.h"

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
#ifndef POLLRDHUP
#define QWEN_POLL_GONE (POLLHUP | POLLERR | POLLNVAL)
#define QWEN_HAVE_RDHUP 0
#else
#define QWEN_POLL_GONE (POLLRDHUP | POLLHUP | POLLERR | POLLNVAL)
#define QWEN_HAVE_RDHUP 1
#endif
#include <sys/time.h>
#include <stdatomic.h>

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

#define MAX_TTS_TEXT 8192
#define QWEN_STR2(x) #x
#define QWEN_STR(x) QWEN_STR2(x)

#define QWEN_CHARS_PER_CAP_SECOND 30

static void srv_conn_close(int fd);

static pthread_mutex_t g_synth_lock = PTHREAD_MUTEX_INITIALIZER;

static int g_serialize_synth = 0;

static inline float clampf(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

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

static double json_extract_number(const char *json, const char *key, double def) {
    char pattern[256];
    snprintf(pattern, sizeof(pattern), "\"%s\"", key);
    const char *p = strstr(json, pattern);
    if (!p) return def;
    p += strlen(pattern);
    while (*p == ' ' || *p == '\t' || *p == '\n' || *p == ':') p++;
    if (*p == '"') return def;
    return atof(p);
}

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

        if (header_end < 0) {
            char *hend = strstr(buf, "\r\n\r\n");
            if (hend) {
                header_end = (int)(hend - buf) + 4;
                char *cl = strcasestr(buf, "Content-Length:");
                if (cl) content_length = atoi(cl + 15);
                else content_length = 0;
                if (content_length < 0) content_length = 0;
                if (content_length > buf_size - 1) {
                    g_req_too_large = 1;
                    content_length = buf_size - 1;
                }
            }
        }

        if (header_end >= 0) {
            int body_received = total - header_end;
            if (body_received >= content_length) break;
        }
    }
    return total;
}

static void send_response(int fd, int status, const char *content_type,
                          const void *body, int body_len) {
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

typedef struct {
    int fd;
    int total_samples;
    float volume;
} stream_http_state_t;

static int qwen_cancel_on_disconnect(void) {
    static int v = -1;
    if (v < 0) { const char *e = getenv("QWEN_CANCEL_ON_DISCONNECT"); v = (e && e[0] == '1'); }
    return v;
}

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
        return -1;
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
    int16_t *pcm = (int16_t *)malloc(n_samples * sizeof(int16_t));
    for (int i = 0; i < n_samples; i++) {
        float s = samples[i] * g;
        if (s < -1.0f) s = -1.0f;
        if (s > 1.0f) s = 1.0f;
        pcm[i] = (int16_t)(s * 32767);
    }
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

static void compose_stream_emit(const float *pcm, int n, void *user) {
    stream_http_callback(pcm, n, user);
}

static void *build_wav(const float *samples, int n_samples, int *out_size) {
    int sample_rate = QWEN_TTS_SAMPLE_RATE;
    int bits = 16, channels = 1;
    int data_size = n_samples * channels * (bits / 8);
    int file_size = 36 + data_size;
    int total = 44 + data_size;
    char *wav = (char *)malloc(total);
    char *p = wav;

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

typedef struct {
    atomic_int sched_alive;
    atomic_int running;
    atomic_int waiting;
    atomic_int admitted, done;
    atomic_int rejected_full;
    atomic_int rejected_stale;
    atomic_int timed_out;
    int queue_max;
    int slots;
    int queue_timeout_ms;
    int max_request_ms;
    int max_text_chars;
} server_state_t;

static server_state_t g_srv;

static int g_cfg_max_queue = -1;
static int g_cfg_queue_timeout_ms = 0;
static int g_cfg_max_request_ms = 60000;
static int g_cfg_max_text_chars = 0;

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

static const char *const g_known_fields[] = {
    "input", "model", "voice", "response_format", "speed", "instructions",
    "stream_format", "stream",
    "chunk_frames", "emotion", "instruct", "language", "max_new_tokens", "rate",
    "rep_penalty", "seed", "speaker", "temperature", "text", "top_k", "top_p",
    "voice_design", "volume", NULL
};

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

void qwen_tts_server_set_max_request_ms(int ms) { g_cfg_max_request_ms = ms; }
void qwen_tts_server_set_max_text_chars(int chars) { g_cfg_max_text_chars = chars; }

static int srv_max_text_chars(void) {
    if (g_srv.max_text_chars > 0) return g_srv.max_text_chars;
    long by_prompt = (long)qwen_tts_batch_max_prompt() * 7 / 2;
    long lim = by_prompt;
    if (g_srv.max_request_ms > 0) {
        long by_time = (long)(g_srv.max_request_ms / 1000) * QWEN_CHARS_PER_CAP_SECOND;
        if (by_time < lim) lim = by_time;
    }
    if (lim < 200)          lim = 200;
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

static void handle_health(int fd) {
    int alive = atomic_load(&g_srv.sched_alive);
    int waiting = atomic_load(&g_srv.waiting);
    char json[512];
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

static void reset_request_state(qwen_tts_ctx_t *ctx) {
    if (!ctx->voice_clone) {
        ctx->speaker_id = 3061;
        ctx->language_id = 2050;
    }

    ctx->temperature = 0.5f;
    ctx->top_k = 50;
    ctx->top_p = 1.0f;
    ctx->rep_penalty = 1.05f;

    ctx->voice_design = 0;
    free(ctx->instruct);
    ctx->instruct = NULL;

    ctx->cp_roughness = 0.0f;
    if (ctx->ml_steer) { free(ctx->ml_steer); ctx->ml_steer = NULL; ctx->ml_steer_layers = 0; }

    struct timeval tv;
    gettimeofday(&tv, NULL);
    ctx->seed = (uint32_t)(tv.tv_sec ^ tv.tv_usec);
}

static char *parse_tts_request(qwen_tts_ctx_t *ctx, const char *body,
                               float *out_volume, float *out_rate) {
    reset_request_state(ctx);

    char *text = json_extract_string(body, "text");
    if (!text) {
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
        int sid = ctx->speaker_id;
        int bad = resolve_speaker_checked(ctx, speaker, &sid, g_req_err, sizeof(g_req_err));
        free(speaker);
        if (bad) { free(text); return NULL; }
        ctx->speaker_id = sid;
    }

    char *language = json_extract_string(body, "language");
    if (language) {
        int lid = qwen_tts_language_id(language);
        if (lid >= 0) ctx->language_id = lid;
    }

    free(ctx->instruct);
    ctx->instruct = json_extract_string(body, "instruct");
    if (!ctx->instruct) ctx->instruct = json_extract_string(body, "instructions");

    char *vd = json_extract_string(body, "voice_design");
    if (vd) {
        if (strcmp(vd, "true") == 0 || strcmp(vd, "1") == 0) ctx->voice_design = 1;
        free(vd);
    }

    ctx->temperature = clampf((float)json_extract_number(body, "temperature", ctx->temperature), 0.0f, 2.0f);
    ctx->top_k       = (int)json_extract_number(body, "top_k", ctx->top_k);
    if (ctx->top_k < 0) ctx->top_k = 0;
    if (ctx->top_k > ctx->config.codec_vocab_size) ctx->top_k = ctx->config.codec_vocab_size;
    ctx->top_p       = clampf((float)json_extract_number(body, "top_p", ctx->top_p), 0.0f, 1.0f);
    ctx->rep_penalty = clampf((float)json_extract_number(body, "rep_penalty", ctx->rep_penalty), 0.5f, 2.0f);

    int seed = (int)json_extract_number(body, "seed", -1);
    if (seed >= 0) ctx->seed = (uint32_t)seed;

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

    float eff_vol = 1.0f, eff_rate = 1.0f;
    int vol_present  = strstr(body, "\"volume\"") != NULL;
    int rate_present = strstr(body, "\"rate\"") != NULL;
    float req_vol  = (float)json_extract_number(body, "volume", 1.0);
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

    ctx->stream = 0;
    ctx->audio_cb = NULL;

    float *audio = NULL;
    int n_samples = 0;

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

    if (volume != 1.0f) qwen_audio_apply_gain(audio, n_samples, volume);
    if (rate != 1.0f) {
        float *stretched = NULL; int stretched_n = 0;
        if (qwen_audio_time_stretch(audio, n_samples, rate, QWEN_TTS_SAMPLE_RATE, &stretched, &stretched_n) == 0) {
            free(audio); audio = stretched; n_samples = stretched_n;
        }
    }

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
    (void)rate;
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

    stream_http_state_t state = { .fd = fd, .total_samples = 0, .volume = volume };
    ctx->stream = 1;
    int chunk_frames = (int)json_extract_number(body, "chunk_frames", 10);
    if (chunk_frames < 2)   chunk_frames = 2;
    if (chunk_frames > 250) chunk_frames = 250;
    ctx->stream_chunk_frames = chunk_frames;
    qwen_tts_set_audio_callback(ctx, stream_http_callback, &state);

    send_chunked_header(fd);

    if (qwen_compose_has_markup(text)) {
        char *language = json_extract_string(body, "language");
        qwen_cspan_t *spans = NULL; int nspans = 0;
        if (qwen_compose_parse(text, &spans, &nspans) == 0 && nspans > 0) {
            fprintf(stderr, "[HTTP] inline markup -> per-sentence compose stream (%d spans)\n", nspans);
            ctx->stream = 0;
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

    send_chunked_end(fd);

    ctx->stream = 0;
    ctx->audio_cb = NULL;

    double elapsed = server_time_ms() - t0;
    float audio_secs = (float)state.total_samples / QWEN_TTS_SAMPLE_RATE;
    fprintf(stderr, "[HTTP] Streamed %d samples (%.2fs audio) in %.1fs (RTF %.2f)\n",
            state.total_samples, audio_secs, elapsed / 1000.0, (elapsed / 1000.0) / audio_secs);
}

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
    if (!allow) return 0;
    if (strcmp(method, allow) != 0) {
        char m[96];
        snprintf(m, sizeof(m), "method not allowed - %s takes %s", path, allow);
        send_error(fd, 405, m);
        return 1;
    }
    if (strcmp(allow, "POST") != 0) return 0;

    if (g_req_too_large) {
        send_error(fd, 413, "request body too large");
        return 1;
    }
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
    char *buf = (char *)malloc(1024 * 1024);
    if (!buf) { srv_conn_close(client_fd); return; }
    int total = read_request(client_fd, buf, 1024 * 1024);
    if (total <= 0) { free(buf); srv_conn_close(client_fd); return; }

    char method[16] = {0}, path[256] = {0};
    sscanf(buf, "%15s %255s", method, path);

    const char *body = strstr(buf, "\r\n\r\n");
    if (body) body += 4;
    else body = "";

    char client_ip[INET_ADDRSTRLEN] = {0};
    inet_ntop(AF_INET, &client_addr.sin_addr, client_ip, sizeof(client_ip));
    fprintf(stderr, "[HTTP] %s %s %s from %s\n", method, path,
            (strcmp(method, "POST") == 0 && body[0]) ? "(has body)" : "", client_ip);

    if (strcmp(method, "OPTIONS") != 0 && http_precheck(client_fd, method, path, buf, body)) {
        free(buf); srv_conn_close(client_fd); return;
    }

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
        handle_tts(ctx, client_fd, body);
        if (g_serialize_synth) pthread_mutex_unlock(&g_synth_lock);
    }
    else {
        send_error(client_fd, 404, "not found");
    }

    free(buf);
    srv_conn_close(client_fd);
}

#define CONN_QUEUE_CAP 256

typedef struct {
    int fds[CONN_QUEUE_CAP];
    int head, tail, count;
    pthread_mutex_t mtx;
    pthread_cond_t not_empty;
    pthread_cond_t not_full;
    int shutdown;
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
        pthread_cond_wait(&q->not_full, &q->mtx);
    if (q->shutdown) { pthread_mutex_unlock(&q->mtx); srv_conn_close(fd); return; }
    q->fds[q->tail] = fd;
    q->tail = (q->tail + 1) % CONN_QUEUE_CAP;
    q->count++;
    pthread_cond_signal(&q->not_empty);
    pthread_mutex_unlock(&q->mtx);
}

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
        if (fd < 0) break;
        handle_connection(wa->ctx, fd, (struct sockaddr_in){0});
    }
    return NULL;
}

static volatile sig_atomic_t server_running = 1;

static void sigint_handler(int sig) {
    (void)sig;
    server_running = 0;
}

static void set_client_timeout(int fd) {
    struct timeval tv = { .tv_sec = 30, .tv_usec = 0 };
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
}

static int g_conn_chan_fd = -1;
static int g_conn_done_fd = -1;

static void srv_conn_close(int fd) {
    if (fd >= 0) close(fd);
    if (g_conn_done_fd >= 0) {
        char b = 1;
        ssize_t r = write(g_conn_done_fd, &b, 1);
        (void)r;
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
    ssize_t n = recvmsg(chan, &msg, 0);
    if (n < 0 && errno == EINTR) return -3;
    if (n <= 0) return -1;
    struct cmsghdr *cm = CMSG_FIRSTHDR(&msg);
    if (!cm || cm->cmsg_type != SCM_RIGHTS) return -2;
    int fd; memcpy(&fd, CMSG_DATA(cm), sizeof(int));
    return fd;
}
#endif

static int setup_listen_socket(int port) {
    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0) { perror("socket"); return -1; }
    int opt = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
#ifdef SO_REUSEPORT
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

static volatile sig_atomic_t g_srv_dump = 0;
static void srv_dump_sig(int sig) { (void)sig; g_srv_dump = 1; }

/* SIGUSR1 asks for counters. Without a handler its default action is to KILL the process,
   so a stats signal to a non-prefork server would end it. */
static void srv_dump_counters_if_asked(void) {
    if (!g_srv_dump) return;
    g_srv_dump = 0;
    qwen_pool_stats_report();
    if (qwen_census_enabled()) qwen_census_report(NULL);
    if (qwen_matmat_stats_enabled()) qwen_matmat_stats_report(NULL);
    fflush(stderr);
}

static void install_signal_handlers(void) {
    struct sigaction sa = { .sa_handler = sigint_handler };
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    sigaction(SIGINT, &sa, NULL);
    sigaction(SIGTERM, &sa, NULL);
    struct sigaction su = { .sa_handler = srv_dump_sig };
    sigemptyset(&su.sa_mask); su.sa_flags = 0;
    sigaction(SIGUSR1, &su, NULL);
    signal(SIGPIPE, SIG_IGN);
}

static void print_banner(int port, int n_workers) {
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

enum { JOB_BATCH = 0, JOB_SINGLE = 1 };

typedef struct batch_job {
    int fd;
    int kind;
    int is_stream;
    int header_sent;
    char *text;
    char *body;
    qwen_batch_req_t req;
    double enq_ms;
    double t_recv, t_parsed, t_admit, t_first;
    double t_write_attempt;
    double t_write_complete;
    unsigned long long enq_adm_seq;
    double enq_adm_ts;
    double enq_last_iter_ms;
    unsigned int life_seed;
    int client_gone;
    int cancelled;
    int timed_out;
    double t_abort_detected;
    double t_cancel_stop;
    struct batch_job *next;
} batch_job_t;

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
    int cap;
    pthread_mutex_t mtx;
    pthread_cond_t not_empty;
    int shutdown;
} job_queue_t;

static void jq_init(job_queue_t *q) {
    q->head = q->tail = NULL; q->count = 0; q->cap = 0; q->shutdown = 0;
    pthread_mutex_init(&q->mtx, NULL);
    pthread_cond_init(&q->not_empty, NULL);
}
static int jq_push(job_queue_t *q, batch_job_t *j) {
    j->next = NULL;
    j->enq_ms = srv_now_ms();
    if (getenv("QWEN_TTFA_TRACE"))
        qwen_admit_probe_read(&j->enq_adm_seq, &j->enq_adm_ts, &j->enq_last_iter_ms);
    pthread_mutex_lock(&q->mtx);
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

    if (ctx->voice_clone) { req->speaker_id = def_speaker_id; req->language_id = def_language_id; }
    else { req->speaker_id = 3061  ; req->language_id = 2050  ; }
    req->temperature = 0.5f; req->top_k = 50; req->top_p = 1.0f; req->rep_penalty = 1.05f;
    req->greedy_warmup = ctx->greedy_warmup;
    struct timeval tv; gettimeofday(&tv, NULL);
    req->seed = (uint32_t)(tv.tv_sec ^ tv.tv_usec);

    char *speaker = json_extract_string(body, "speaker");
    if (!speaker) speaker = json_extract_string(body, "voice");
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

    char *instruct = json_extract_string(body, "instruct");
    if (instruct && instruct[0]) *needs_single = 1;
    free(instruct);
    char *vd = json_extract_string(body, "voice_design");
    if (vd) { if (strcmp(vd, "true") == 0 || strcmp(vd, "1") == 0) *needs_single = 1; free(vd); }

    req->text = NULL;
    return text;
}

static void respond_wav(int fd, const float *audio, int n_samples) {
    if (!audio || n_samples <= 0) { send_error(fd, 500, "generation failed"); return; }
    int wav_size = 0;
    void *wav = build_wav(audio, n_samples, &wav_size);
    send_response(fd, 200, "audio/wav", wav, wav_size);
    free(wav);
}

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
                j->kind = JOB_SINGLE; j->is_stream = is_stream;
                j->body = strdup(body); j->text = text;
                j->t_recv = _t_recv; j->t_parsed = srv_now_ms();
                if (!jq_push(ra->jq_single, j)) {
                    atomic_fetch_add(&g_srv.rejected_full, 1);
                    send_error(fd, 503, "server at capacity: queue full");
                    srv_conn_close(fd); job_free(j); free(buf); continue;
                }
            } else {
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

typedef struct {
    job_queue_t *jq;
    volatile sig_atomic_t *running;
    int admitted, done;
} sink_ctx_t;

static int sink_next_job(void *ud, qwen_batch_req_t *req, void **tag, int block) {
    sink_ctx_t *sc = (sink_ctx_t *)ud;
    batch_job_t *j;
    for (;;) {
        j = block ? jq_pop(sc->jq) : jq_trypop(sc->jq);
        if (!j) return 0;
        if (g_srv.queue_timeout_ms > 0 &&
            srv_now_ms() - j->enq_ms > (double)g_srv.queue_timeout_ms) {
            atomic_fetch_add(&g_srv.rejected_stale, 1);
            send_error(j->fd, 503, "server at capacity: queued too long");
            srv_conn_close(j->fd); job_free(j);
            continue;
        }
        break;
    }
    atomic_fetch_add(&g_srv.running, 1);
    atomic_fetch_add(&g_srv.admitted, 1);
    j->t_admit = srv_now_ms();
    j->life_seed = j->req.seed;
    *req = j->req;
    *tag = j;
    sc->admitted++;
    fprintf(stderr, "[BATCH] admit #%d (in-flight admitted=%d, done=%d)\n",
            sc->admitted, sc->admitted - sc->done, sc->done);
    return 1;
}

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

static int sink_cancelled(void *ud, void *tag) {
    (void)ud;
    batch_job_t *j = (batch_job_t *)tag;
    if (!j) return 0;
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
            j->t_parsed - j->t_recv,
            j->t_admit  - j->enq_ms,
            j->t_admit  - j->t_recv,
            j->t_first  - j->t_admit,
            d - j->t_admit,
            d - j->t_recv,
            j->timed_out ? " state=TIMEOUT" :
            j->client_gone ? " state=CANCELLED" : " state=COMPLETED");
}

static void sink_on_reject(void *ud, void *tag, const char *reason) {
    sink_ctx_t *sc = (sink_ctx_t *)ud;
    batch_job_t *j = (batch_job_t *)tag;
    char m[220];
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

static void sink_on_done(void *ud, void *tag, float *samples, int n_samples) {
    sink_ctx_t *sc = (sink_ctx_t *)ud;
    batch_job_t *j = (batch_job_t *)tag;
    qwen_life_emit(j);
    if (j->is_stream) {
        if (!j->header_sent) { send_chunked_header(j->fd); j->header_sent = 1; }
        send_chunked_end(j->fd);
    } else if (j->timed_out && (!samples || n_samples <= 0)) {
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
    atomic_store(&g_srv.sched_alive, 0);
    if (rc != 0 && server_running) {
        fprintf(stderr, "[BATCH] FATAL: continuous scheduler failed (rc=%d) — "
                        "draining batch jobs with 503 until shutdown\n", rc);
        for (;;) {
            batch_job_t *j = jq_pop(sa->jq);
            if (!j) break;
            send_error(j->fd, 503, "batch scheduler unavailable (startup failure)");
            srv_conn_close(j->fd); job_free(j);
        }
    }
    return NULL;
}

typedef struct { qwen_tts_ctx_t *ctx; job_queue_t *jq; int reject; } single_arg_t;
static void *single_worker_main(void *arg) {
    single_arg_t *sw = (single_arg_t *)arg;
    for (;;) {
        batch_job_t *j = jq_pop(sw->jq);
        if (!j) break;
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

static void server_default_decoder_batch(qwen_tts_ctx_t *ctx) {
    (void)ctx;
    if (getenv("QWEN_SERVER_NO_DECODER_BATCH")) return;
    setenv("QWEN_DECODER_BATCH", "1", 0);
    fprintf(stderr, "[serve] batched speech decoder ON by default (one pass over the decoder "
                    "weights for all active slots) — QWEN_DECODER_BATCH=0 to opt out\n");
}

static void server_default_memory_levers(qwen_tts_ctx_t *ctx) {
    int quantized = ctx->layers && (ctx->layers[0].wq_int8 || ctx->layers[0].wq_q4 || ctx->layers[0].wq_q6);
    if (!quantized) return;
    const char *e = getenv("QWEN_PREFILL_QUANT");
    int on = (e && e[0] && e[0] != '0');
    if (on) {
        setenv("QWEN_FREE_BF16", "1", 0);
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

int qwen_tts_serve_batched(qwen_tts_ctx_t *ctx, int port, int max_batch) {
    qwen_provenance_report(stderr);
    if (max_batch < 2) max_batch = 2;
    int server_fd = (g_conn_chan_fd >= 0) ? -1 : setup_listen_socket(port);
    if (server_fd < 0 && g_conn_chan_fd < 0) return -1;
    install_signal_handlers();
    ctx->silent = 1;
    server_default_memory_levers(ctx);
    server_default_decoder_batch(ctx);
    server_prewarm(ctx);

    int n_readers = max_batch; if (n_readers < 2) n_readers = 2; if (n_readers > 16) n_readers = 16;

    conn_queue_t cq; cq_init(&cq);
    job_queue_t jq; jq_init(&jq);
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
    fprintf(stderr, "[serve] %d slots · %d may wait (%d in the system) · queue deadline %s\n",
            max_batch, jq.cap, max_batch + jq.cap,
            g_srv.queue_timeout_ms > 0 ? "on" : "none");
    job_queue_t jq_single; jq_init(&jq_single);

    pthread_t *readers = (pthread_t *)calloc(n_readers, sizeof(pthread_t));
    reader_arg_t *rargs = (reader_arg_t *)calloc(n_readers, sizeof(reader_arg_t));
    int def_spk = ctx->speaker_id, def_lang = ctx->language_id;
    for (int i = 0; i < n_readers; i++) {
        rargs[i].ctx = ctx; rargs[i].cq = &cq; rargs[i].jq = &jq; rargs[i].jq_single = &jq_single;
        rargs[i].def_speaker_id = def_spk; rargs[i].def_language_id = def_lang;
        pthread_create(&readers[i], NULL, reader_main, &rargs[i]);
    }
    pthread_t sched;
    sched_arg_t sarg = { .ctx = ctx, .jq = &jq, .max_batch = max_batch };
    pthread_create(&sched, NULL, scheduler_main, &sarg);
    qwen_tts_ctx_t *single_ctx = qwen_tts_clone_for_worker(ctx);
    pthread_t single_thr;
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
            client_fd = srv_recv_fd(g_conn_chan_fd);
            if (client_fd == -1) break;
            if (client_fd < 0) continue;
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
            if (client_fd < 0) {
                if (errno == EINTR) { srv_dump_counters_if_asked(); continue; }
                perror("accept"); continue;
            }
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
    qwen_provenance_report(stderr);
    srv_init_request_cap();
    if (n_workers < 1) n_workers = 1;
    int server_fd = setup_listen_socket(port);
    if (server_fd < 0) return -1;
    install_signal_handlers();

    ctx->silent = 1;
    server_default_memory_levers(ctx);
    server_default_decoder_batch(ctx);
    server_prewarm(ctx);

    if (n_workers == 1) {
        print_banner(port, 1);
        while (server_running) {
            struct sockaddr_in client_addr;
            socklen_t client_len = sizeof(client_addr);
            int client_fd = accept(server_fd, (struct sockaddr *)&client_addr, &client_len);
            if (client_fd < 0) {
                if (errno == EINTR) { srv_dump_counters_if_asked(); continue; }
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

    g_serialize_synth = !qwen_parallel_is_reentrant();

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
            if (errno == EINTR) { srv_dump_counters_if_asked(); continue; }
            perror("accept");
            continue;
        }
        set_client_timeout(client_fd);
        cq_push(&q, client_fd);
    }

    close(server_fd);
    cq_shutdown(&q);
    for (int i = 0; i < spawned; i++)
        pthread_join(threads[i], NULL);

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

#if defined(__linux__)
#include <sched.h>
#include <sys/wait.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/un.h>

static volatile sig_atomic_t g_prefork_stop = 0;
static volatile sig_atomic_t g_prefork_dump = 0;
static void qwen_worker_dump_counters(void) {
    qwen_pool_stats_report();
    if (qwen_census_enabled()) qwen_census_report(NULL);
    if (qwen_matmat_stats_enabled()) qwen_matmat_stats_report(NULL);
    void (*yt)(void) = (void (*)(void))dlsym(RTLD_DEFAULT, "yieldtrace_report");
    if (yt) yt();
    fflush(stderr);
}

static void prefork_parent_sig(int sig) { (void)sig; g_prefork_stop = 1; }
static void prefork_dump_sig(int sig) { (void)sig; g_prefork_dump = 1; }

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

static int elastic_apply(int workers, int ncpu, const int *slice, const pid_t *kids,
                         int *base_out) {
    int changed = 0, next = 0;
    for (int w = 0; w < workers; w++) {
        if (kids[w] <= 0) continue;
        int lo, hi;
        if (slice[w] > 0) { lo = next; hi = next + slice[w] - 1; next += slice[w]; }
        else              { lo = hi = ncpu - 1; }
        if (hi >= ncpu) hi = ncpu - 1;
        if (base_out[2 * w] == lo && base_out[2 * w + 1] == hi) continue;
        cpu_set_t set; CPU_ZERO(&set);
        for (int c = lo; c <= hi; c++) CPU_SET(c, &set);
        if (sched_setaffinity(kids[w], sizeof set, &set) != 0) {
            perror("sched_setaffinity(child)");
            continue;
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
    const int cap = max_batch >= 1 ? max_batch : 1;

    qwen_provenance_report(stderr);
    int listen_fd = setup_listen_socket(port);
    if (listen_fd < 0) return -1;

    int (*sp)[2] = (int (*)[2])calloc((size_t)workers, sizeof(int[2]));
    pid_t *kids = (pid_t *)calloc((size_t)workers, sizeof(pid_t));
    long long *assigned = (long long *)calloc((size_t)workers, sizeof(long long));
    long long *completed = (long long *)calloc((size_t)workers, sizeof(long long));
    int *active = (int *)calloc((size_t)workers, sizeof(int));
    int *slice = (int *)calloc((size_t)workers, sizeof(int));
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
            close(listen_fd);
            for (int p2 = 0; p2 <= w; p2++) close(sp[p2][0]);
            g_conn_chan_fd = sp[w][1];
            g_conn_done_fd = sp[w][1];
            cpu_set_t set; CPU_ZERO(&set);
            for (int c = w * per; c < (w + 1) * per && c < ncpu; c++) CPU_SET(c, &set);
            if (sched_setaffinity(0, sizeof(set), &set) != 0) perror("sched_setaffinity");
            qwen_threadpool_after_fork();
            qwen_set_threads(threads_per);
            fprintf(stderr, "prefork: worker %d pid %d cpus %d-%d threads %d\n",
                    w, (int)getpid(), w * per, w * per + per - 1, threads_per);
            int rc = (max_batch >= 2) ? qwen_tts_serve_batched(ctx, port, max_batch)
                                      : qwen_tts_serve_ex(ctx, port, 1);
            qwen_worker_dump_counters();
#ifdef QWEN_ASAN
            __lsan_do_recoverable_leak_check();
#endif
            _exit(rc == 0 ? 0 : 1);
        }
        kids[w] = pid;
        close(sp[w][1]);
    }

    struct sigaction sa = { .sa_handler = prefork_parent_sig };
    sigemptyset(&sa.sa_mask); sa.sa_flags = 0;
    sigaction(SIGINT, &sa, NULL);
    sigaction(SIGTERM, &sa, NULL);
    signal(SIGPIPE, SIG_IGN);

    struct sigaction su = { .sa_handler = prefork_dump_sig };
    sigemptyset(&su.sa_mask); su.sa_flags = 0;
    sigaction(SIGUSR1, &su, NULL);

    struct pollfd *pfd = (struct pollfd *)calloc((size_t)workers + 1, sizeof(struct pollfd));
    if (!pfd) return -1;
    long long dispatched = 0;
    double act_area = 0.0, act_time = 0.0;
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

        int best = -1;
        for (int w = 0; w < workers; w++) {
            if (kids[w] <= 0 || active[w] >= cap) continue;
            if (best < 0 || active[w] < active[best]) best = w;
        }
        if (best < 0) {
            rejected++;
            send_error(cfd, 503, "all workers at capacity");
            close(cfd);
            continue;
        }
        set_client_timeout(cfd);
        if (elastic) {
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
        close(cfd);
        active[best]++; assigned[best]++; dispatched++;
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
