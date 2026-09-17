/* qwen_tts_server.c - Minimal HTTP server for Qwen3-TTS */
#ifdef __linux__
#ifndef _GNU_SOURCE
#define _GNU_SOURCE 1
#endif
#endif
#include "qwen_tts_server.h"
#if defined(__linux__)
#include <sys/prctl.h>
#include <sys/mman.h>
#endif
#include "qwen_tts_costmap.h"
#include "qwen_tts_kernels.h"
#include <dlfcn.h>
#include "qwen_tts.h"
#include "qwen_tts_thread.h"
#include "qwen_tts_emotion.h"
#include "qwen_tts_compose.h"
#include "qwen_tts_audio.h"
#include "qwen_json.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
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
#include <time.h>
#include <fcntl.h>
#include <stdatomic.h>
#include "qwen_build_id.h"

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
static void qwen_thread_name(const char *prefix);

typedef struct stream_output stream_output_t;
static int stream_output_enqueue(stream_output_t *out, const float *samples,
                                 int n_samples, float gain);
static stream_output_t *stream_output_start(int fd, unsigned int seed);
static void stream_output_finish(stream_output_t *out);
static void stream_output_release(stream_output_t *out);
static int stream_output_failed(stream_output_t *out);

static pthread_mutex_t g_synth_lock = PTHREAD_MUTEX_INITIALIZER;

static int g_serialize_synth = 0;

static inline float clampf(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

static qwen_json_string_status_t json_extract_string(const char *json, const char *key,
                                                     char **out, const char **why) {
    return qwen_json_extract_string(json, key, out, why);
}

static int json_string_error(char *err, size_t errsz, const char *field, const char *why) {
    snprintf(err, errsz, "invalid JSON string for '%s': %s", field,
             why ? why : "malformed string");
    return -1;
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
    qwen_json_escape(dst, dstsz, src);
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
    stream_output_t *out;
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
        if (w < 0 && errno == EINTR) continue;
        /* A socket SO_SNDTIMEO expiry is reported as EAGAIN/EWOULDBLOCK.  Do not
         * spin forever: the output owner turns it into a stream cancellation. */
        if (w < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) return -1;
        if (w < 0 && (errno == EPIPE || errno == ECONNRESET ||
                      errno == ENOTCONN || errno == EBADF)) return -1;
        return -1;
    }
    return 0;
}

static int send_chunked_header(int fd) {
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
    return write_all_or_gone(fd, header, strlen(header));
}

static int stream_http_callback(const float *samples, int n_samples, void *userdata) {
    stream_http_state_t *st = (stream_http_state_t *)userdata;
    if (st->out) {
        int rc = stream_output_enqueue(st->out, samples, n_samples, st->volume);
        if (rc == 0) st->total_samples += n_samples;
        return rc;
    }
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

static int send_chunked_end(int fd) {
    return write_all_or_gone(fd, "0\r\n\r\n", 5);
}

typedef struct stream_output_chunk {
    int16_t *pcm;
    int n_samples;
    size_t bytes;
    struct stream_output_chunk *next;
} stream_output_chunk_t;

struct stream_output {
    int fd;
    unsigned int seed;
    size_t max_bytes;
    int send_timeout_ms;
    pthread_mutex_t mtx;
    pthread_cond_t cv;
    stream_output_chunk_t *head;
    stream_output_chunk_t *tail;
    size_t queued_bytes;
    size_t peak_bytes;
    unsigned long enqueued_chunks;
    unsigned long failed_enqueues;
    int producer_done;
    int failed;
    int total_samples;
    double enqueue_first_ms;
    double write_attempt_ms;
    double write_complete_ms;
    _Atomic int refs;       /* producer + detached writer */
};

static double stream_output_now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

static int stream_output_enabled(void) {
    const char *e = getenv("QWEN_SERVER_ASYNC_OUTPUT");
    return e && e[0] && e[0] != '0';
}

static size_t stream_output_max_bytes(void) {
    const char *e = getenv("QWEN_STREAM_OUTPUT_MAX_BYTES");
    if (!e || !e[0]) return (size_t)1 << 20;
    char *end = NULL;
    unsigned long long v = strtoull(e, &end, 10);
    if (end == e || *end != '\0' || v < 4096 || v > ((unsigned long long)1 << 30))
        return (size_t)1 << 20;
    return (size_t)v;
}

static int stream_output_timeout_ms(void) {
    const char *e = getenv("QWEN_STREAM_OUTPUT_SEND_TIMEOUT_MS");
    if (!e || !e[0]) return 5000;
    char *end = NULL;
    long v = strtol(e, &end, 10);
    if (end == e || *end != '\0' || v < 1 || v > 120000) return 5000;
    return (int)v;
}

static void stream_output_free_chunks_locked(stream_output_t *out) {
    stream_output_chunk_t *p = out->head;
    while (p) {
        stream_output_chunk_t *next = p->next;
        free(p->pcm);
        free(p);
        p = next;
    }
    out->head = out->tail = NULL;
    out->queued_bytes = 0;
}

static void stream_output_release(stream_output_t *out) {
    if (!out || atomic_fetch_sub(&out->refs, 1) != 1) return;
    pthread_cond_destroy(&out->cv);
    pthread_mutex_destroy(&out->mtx);
    free(out);
}

static int stream_output_send_pcm(stream_output_t *out,
                                  const int16_t *pcm, int n_samples) {
    int data_len = n_samples * (int)sizeof(int16_t);
    char ch[32];
    int chlen = snprintf(ch, sizeof ch, "%x\r\n", data_len);
    if (write_all_or_gone(out->fd, ch, (size_t)chlen) < 0) return -1;
    if (write_all_or_gone(out->fd, pcm, (size_t)data_len) < 0) return -1;
    return write_all_or_gone(out->fd, "\r\n", 2);
}

static void stream_output_mark_failed(stream_output_t *out) {
    pthread_mutex_lock(&out->mtx);
    out->failed = 1;
    out->producer_done = 1;
    stream_output_free_chunks_locked(out);
    pthread_cond_broadcast(&out->cv);
    pthread_mutex_unlock(&out->mtx);
}

static void *stream_output_writer_main(void *arg) {
    stream_output_t *out = (stream_output_t *)arg;
    qwen_thread_name("srv-output");

    pthread_mutex_lock(&out->mtx);
    out->write_attempt_ms = stream_output_now_ms();
    pthread_mutex_unlock(&out->mtx);
    if (send_chunked_header(out->fd) < 0) {
        stream_output_mark_failed(out);
    } else {
        for (;;) {
            pthread_mutex_lock(&out->mtx);
            while (!out->head && !out->producer_done)
                pthread_cond_wait(&out->cv, &out->mtx);
            stream_output_chunk_t *chunk = out->head;
            if (chunk) {
                out->head = chunk->next;
                if (!out->head) out->tail = NULL;
                out->queued_bytes -= chunk->bytes;
            }
            int done = out->producer_done && !chunk;
            int failed = out->failed;
            pthread_mutex_unlock(&out->mtx);

            if (!chunk) {
                if (done || failed) break;
                continue;
            }
            int rc = failed ? -1 : stream_output_send_pcm(out, chunk->pcm, chunk->n_samples);
            if (rc < 0) {
                free(chunk->pcm); free(chunk);
                stream_output_mark_failed(out);
                break;
            }
            pthread_mutex_lock(&out->mtx);
            out->total_samples += chunk->n_samples;
            if (out->write_complete_ms == 0.0)
                out->write_complete_ms = stream_output_now_ms();
            pthread_mutex_unlock(&out->mtx);
            free(chunk->pcm); free(chunk);
        }
    }

    pthread_mutex_lock(&out->mtx);
    int failed = out->failed;
    if (failed) stream_output_free_chunks_locked(out);
    pthread_mutex_unlock(&out->mtx);
    int end_rc = 0;
    if (!failed) end_rc = send_chunked_end(out->fd);
    if (end_rc < 0) {
        pthread_mutex_lock(&out->mtx);
        out->failed = 1;
        pthread_mutex_unlock(&out->mtx);
    }
    srv_conn_close(out->fd);

    pthread_mutex_lock(&out->mtx);
    fprintf(stderr, "[OUT] v=1 pid=%d seed=%u async=1 queued_cap_bytes=%zu "
                    "peak_bytes=%zu enqueued_chunks=%lu failed_enqueues=%lu "
                    "samples=%d failed=%d enqueue_first_ms=%.3f "
                    "write_attempt_ms=%.3f write_complete_ms=%.3f "
                    "send_timeout_ms=%d\n",
            (int)getpid(), out->seed, out->max_bytes, out->peak_bytes,
            out->enqueued_chunks, out->failed_enqueues, out->total_samples,
            out->failed, out->enqueue_first_ms, out->write_attempt_ms,
            out->write_complete_ms, out->send_timeout_ms);
    pthread_mutex_unlock(&out->mtx);
    stream_output_release(out); /* detached writer reference */
    return NULL;
}

static stream_output_t *stream_output_start(int fd, unsigned int seed) {
    stream_output_t *out = (stream_output_t *)calloc(1, sizeof(*out));
    if (!out) return NULL;
    out->fd = fd;
    out->seed = seed;
    out->max_bytes = stream_output_max_bytes();
    out->send_timeout_ms = stream_output_timeout_ms();
    atomic_init(&out->refs, 2);
    pthread_mutex_init(&out->mtx, NULL);
    pthread_cond_init(&out->cv, NULL);
    struct timeval tv = {
        .tv_sec = out->send_timeout_ms / 1000,
        .tv_usec = (out->send_timeout_ms % 1000) * 1000
    };
    (void)setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof tv);
    pthread_t thr;
    if (pthread_create(&thr, NULL, stream_output_writer_main, out) != 0) {
        pthread_cond_destroy(&out->cv);
        pthread_mutex_destroy(&out->mtx);
        free(out);
        return NULL;
    }
    pthread_detach(thr);
    return out;
}

static int stream_output_enqueue(stream_output_t *out, const float *samples,
                                  int n_samples, float gain) {
    if (!out || !samples || n_samples <= 0) return -1;
    stream_output_chunk_t *chunk = (stream_output_chunk_t *)calloc(1, sizeof(*chunk));
    if (!chunk) {
        stream_output_mark_failed(out);
        pthread_mutex_lock(&out->mtx);
        out->failed_enqueues++;
        pthread_mutex_unlock(&out->mtx);
        return -1;
    }
    chunk->n_samples = n_samples;
    chunk->bytes = (size_t)n_samples * sizeof(int16_t);
    chunk->pcm = (int16_t *)malloc(chunk->bytes);
    if (!chunk->pcm) {
        free(chunk);
        stream_output_mark_failed(out);
        pthread_mutex_lock(&out->mtx);
        out->failed_enqueues++;
        pthread_mutex_unlock(&out->mtx);
        return -1;
    }
    for (int i = 0; i < n_samples; i++) {
        float s = samples[i] * gain;
        if (s < -1.0f) s = -1.0f;
        if (s > 1.0f) s = 1.0f;
        chunk->pcm[i] = (int16_t)(s * 32767);
    }

    pthread_mutex_lock(&out->mtx);
    if (out->failed || out->producer_done ||
        chunk->bytes > out->max_bytes || out->queued_bytes > out->max_bytes - chunk->bytes) {
        out->failed_enqueues++;
        out->failed = 1;
        out->producer_done = 1;
        pthread_cond_broadcast(&out->cv);
        pthread_mutex_unlock(&out->mtx);
        free(chunk->pcm); free(chunk);
        return -1;
    }
    if (out->tail) out->tail->next = chunk; else out->head = chunk;
    out->tail = chunk;
    out->queued_bytes += chunk->bytes;
    if (out->queued_bytes > out->peak_bytes) out->peak_bytes = out->queued_bytes;
    out->enqueued_chunks++;
    if (out->enqueue_first_ms == 0.0) out->enqueue_first_ms = stream_output_now_ms();
    pthread_cond_signal(&out->cv);
    pthread_mutex_unlock(&out->mtx);
    return 0;
}

static void stream_output_finish(stream_output_t *out) {
    if (!out) return;
    pthread_mutex_lock(&out->mtx);
    out->producer_done = 1;
    pthread_cond_signal(&out->cv);
    pthread_mutex_unlock(&out->mtx);
}

static int stream_output_failed(stream_output_t *out) {
    if (!out) return 0;
    pthread_mutex_lock(&out->mtx);
    int failed = out->failed;
    pthread_mutex_unlock(&out->mtx);
    return failed;
}

static int compose_stream_emit(const float *pcm, int n, void *user) {
    return stream_http_callback(pcm, n, user);
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
    int batched;                /* 1 only when the continuous scheduler thread exists */
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

static int qwen_admit_util_requested(void) {
    const char *e = getenv("QWEN_ADMIT_UTIL");
    return e && *e && atoi(e) != 0;
}

static double qwen_admit_util_limit_ms(void) {
    const double fallback = 60.0;
    const char *e = getenv("QWEN_ADMIT_UTIL_LIMIT_MS");
    if (!e || !*e) return fallback;
    char *end = NULL;
    double v = strtod(e, &end);
    if (end == e || *end != '\0' || !isfinite(v) || v <= 0.0) return fallback;
    return v;
}

static int qwen_admit_util_trace(void) {
    const char *e = getenv("QWEN_ADMIT_UTIL_TRACE");
    return e && *e && atoi(e) != 0;
}

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
    return qwen_json_string_end(p, why);
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
    char *f = NULL;
    const char *why = NULL;
    qwen_json_string_status_t st = json_extract_string(body, "response_format", &f, &why);
    if (st == QWEN_JSON_STRING_INVALID)
        return json_string_error(err, errsz, "response_format", why);
    if (st == QWEN_JSON_STRING_ABSENT) return 0;
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
    /* The same seconds also bound the GENERATION: without this the batched engine stopped at
     * its compiled 600-frame ceiling (48 s) while the text limit admitted 60 s of speech, and
     * the stream ended as if the model had finished.  An explicit QWEN_BATCH_MAX_FRAMES wins. */
    if (g_srv.max_request_ms > 0)
        qwen_tts_set_batch_max_frames((int)((g_srv.max_request_ms / 1000.0) * 12.5 + 0.5));
    {
        int fcap = qwen_tts_batch_max_frames();
        int src = qwen_tts_batch_max_frames_source();
        const char *why = src == 2 ? "QWEN_BATCH_MAX_FRAMES" : src == 1 ? "from --max-request-seconds" : "compiled default";
        if (g_srv.max_request_ms > 0)
            fprintf(stderr, "[serve] per-request generation cap: %.0f s -> text limit %d characters, "
                            "frame cap %d = %.1f s of audio (%s); a request that reaches the frame cap "
                            "is TRUNCATED and logged (--max-request-seconds N / --max-text-chars N; 0 disables the text cap)\n",
                    g_srv.max_request_ms / 1000.0, srv_max_text_chars(), fcap, fcap / 12.5, why);
        else
            fprintf(stderr, "[serve] per-request generation cap: DISABLED - text limit %d characters; "
                            "generation still stops at the frame cap %d = %.1f s of audio (%s) and is "
                            "TRUNCATED and logged there\n",
                    srv_max_text_chars(), fcap, fcap / 12.5, why);
    }
}

static void handle_health(int fd) {
    int alive = atomic_load(&g_srv.sched_alive);
    int waiting = atomic_load(&g_srv.waiting);
    /* The plain server has no scheduler thread, so there is nothing for sched_alive to
       report and 0 is its resting value. Reading that as "down" made a perfectly healthy
       single-worker server answer 503 to every health probe -- which is precisely what a
       load balancer reads as "take this host out of rotation". Only a server that HAS a
       scheduler can be unavailable for want of one. */
    int ready = g_srv.batched ? alive : 1;
    const char *sched = g_srv.batched ? (alive ? "running" : "down") : "none";
    const char *mode  = g_srv.batched ? "batched" : "single";
    char json[640];
    snprintf(json, sizeof(json),
             "{\"status\":\"%s\",\"mode\":\"%s\",\"scheduler\":\"%s\","
             "\"num_requests_running\":%d,\"num_requests_waiting\":%d,"
             "\"queue_max\":%d,\"queue_timeout_ms\":%d,\"max_request_ms\":%d,"
             "\"max_text_chars\":%d,"
             "\"admitted\":%d,\"done\":%d,"
             "\"rejected_queue_full\":%d,\"rejected_queue_timeout\":%d,"
             "\"timed_out\":%d}",
             ready ? "ok" : "unavailable", mode, sched,
             atomic_load(&g_srv.running), waiting,
             g_srv.queue_max, g_srv.queue_timeout_ms, g_srv.max_request_ms,
             srv_max_text_chars(),
             atomic_load(&g_srv.admitted), atomic_load(&g_srv.done),
             atomic_load(&g_srv.rejected_full), atomic_load(&g_srv.rejected_stale),
             atomic_load(&g_srv.timed_out));
    send_json(fd, ready ? 200 : 503, json);
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

    /* Forget the previous request's prefill.  qwen_tts_generate() keeps prev_input_embeds and
     * re-prefills only from the first position whose embedding differs, reusing the KV rows
     * of the common prefix.  That reuse is causally sound but NOT bit-identical: the tail
     * positions are then computed in a shorter prefill, with different GEMM tiling and
     * accumulation order, and over ~96 autoregressive frames the difference forks the
     * trajectory.  The effect is that output depended on WHICH request came before
     * (identical consecutive texts matched fully and fell back to a full prefill, so they
     * were right; anything else diverged from the CLI reference by mel_corr ~0.93).
     * qwen_tts_generate_batch() already clears this per item, so the batched server was
     * never affected — this makes the single-process server agree with it and with the CLI. */
    ctx->prev_prefill_len = 0;

    struct timeval tv;
    gettimeofday(&tv, NULL);
    ctx->seed = (uint32_t)(tv.tv_sec ^ tv.tv_usec);
}

static char *parse_tts_request(qwen_tts_ctx_t *ctx, const char *body,
                               float *out_volume, float *out_rate) {
    g_req_err[0] = '\0';
    reset_request_state(ctx);

    char *text = NULL;
    const char *why = NULL;
    qwen_json_string_status_t text_status = json_extract_string(body, "text", &text, &why);
    if (text_status == QWEN_JSON_STRING_INVALID) {
        json_string_error(g_req_err, sizeof(g_req_err), "text", why);
        return NULL;
    }
    if (text_status == QWEN_JSON_STRING_ABSENT) {
        text_status = json_extract_string(body, "input", &text, &why);
        if (text_status == QWEN_JSON_STRING_INVALID) {
            json_string_error(g_req_err, sizeof(g_req_err), "input", why);
            return NULL;
        }
    }
    if (text_status != QWEN_JSON_STRING_VALID || !text || text[0] == '\0') {
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

    char *speaker = NULL;
    qwen_json_string_status_t speaker_status = json_extract_string(body, "speaker", &speaker, &why);
    if (speaker_status == QWEN_JSON_STRING_INVALID) {
        json_string_error(g_req_err, sizeof(g_req_err), "speaker", why);
        free(text); return NULL;
    }
    if (speaker_status == QWEN_JSON_STRING_ABSENT) {
        speaker_status = json_extract_string(body, "voice", &speaker, &why);
        if (speaker_status == QWEN_JSON_STRING_INVALID) {
            json_string_error(g_req_err, sizeof(g_req_err), "voice", why);
            free(text); return NULL;
        }
    }
    if (speaker) {
        int sid = ctx->speaker_id;
        int bad = resolve_speaker_checked(ctx, speaker, &sid, g_req_err, sizeof(g_req_err));
        free(speaker);
        if (bad) { free(text); return NULL; }
        ctx->speaker_id = sid;
    }

    char *language = NULL;
    qwen_json_string_status_t language_status = json_extract_string(body, "language", &language, &why);
    if (language_status == QWEN_JSON_STRING_INVALID) {
        json_string_error(g_req_err, sizeof(g_req_err), "language", why);
        free(text); return NULL;
    }
    if (language) {
        int lid = qwen_tts_language_id(language);
        if (lid >= 0) ctx->language_id = lid;
    }

    free(ctx->instruct);
    ctx->instruct = NULL;
    qwen_json_string_status_t instruct_status = json_extract_string(body, "instruct",
                                                                     &ctx->instruct, &why);
    if (instruct_status == QWEN_JSON_STRING_INVALID) {
        json_string_error(g_req_err, sizeof(g_req_err), "instruct", why);
        free(text); free(language); return NULL;
    }
    if (instruct_status == QWEN_JSON_STRING_ABSENT) {
        instruct_status = json_extract_string(body, "instructions", &ctx->instruct, &why);
        if (instruct_status == QWEN_JSON_STRING_INVALID) {
            json_string_error(g_req_err, sizeof(g_req_err), "instructions", why);
            free(text); free(language); return NULL;
        }
    }

    char *vd = NULL;
    qwen_json_string_status_t vd_status = json_extract_string(body, "voice_design", &vd, &why);
    if (vd_status == QWEN_JSON_STRING_INVALID) {
        json_string_error(g_req_err, sizeof(g_req_err), "voice_design", why);
        free(text); free(language); return NULL;
    }
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
    char *emotion = NULL;
    qwen_json_string_status_t emotion_status = json_extract_string(body, "emotion", &emotion, &why);
    if (emotion_status == QWEN_JSON_STRING_INVALID) {
        json_string_error(g_req_err, sizeof(g_req_err), "emotion", why);
        free(emotion); free(language); free(text); return NULL;
    }
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
        char *language = NULL;
        (void)json_extract_string(body, "language", &language, NULL);
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

static int handle_tts_stream(qwen_tts_ctx_t *ctx, int fd, const char *body) {
    float volume = 1.0f, rate = 1.0f;
    char *text = parse_tts_request(ctx, body, &volume, &rate);
    (void)rate;
    if (!text) {
        send_error(fd, 400, g_req_err[0] ? g_req_err
                                         : "missing, empty, or oversized 'text' (max "
                                           QWEN_STR(MAX_TTS_TEXT) " characters)");
        return 0;
    }
    if (ctx->voice_design && ctx->config.hidden_size < 2048) {
        send_error(fd, 400, "voice_design requires the 1.7B VoiceDesign model");
        free(text);
        return 0;
    }

    fprintf(stderr, "[HTTP] TTS stream: \"%s\" (speaker=%d, lang=%d, seed=%u)\n",
            text, ctx->speaker_id, ctx->language_id, ctx->seed);
    double t0 = server_time_ms();

    stream_http_state_t state = { .fd = fd, .total_samples = 0, .volume = volume, .out = NULL };
    state.out = stream_output_enabled() ? stream_output_start(fd, ctx->seed) : NULL;
    ctx->stream = 1;
    int chunk_frames = (int)json_extract_number(body, "chunk_frames", 10);
    if (chunk_frames < 2)   chunk_frames = 2;
    if (chunk_frames > 250) chunk_frames = 250;
    ctx->stream_chunk_frames = chunk_frames;
    qwen_tts_set_audio_callback(ctx, stream_http_callback, &state);

    if (!state.out) (void)send_chunked_header(fd);

    if (qwen_compose_has_markup(text)) {
        char *language = NULL;
        (void)json_extract_string(body, "language", &language, NULL);
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

    int stream_fd_owned = state.out != NULL;
    if (state.out) {
        stream_output_finish(state.out);
        stream_output_release(state.out); /* release the producer reference */
    } else {
        (void)send_chunked_end(fd);
    }

    ctx->stream = 0;
    ctx->audio_cb = NULL;

    double elapsed = server_time_ms() - t0;
    float audio_secs = (float)state.total_samples / QWEN_TTS_SAMPLE_RATE;
    fprintf(stderr, "[HTTP] Streamed %d samples (%.2fs audio) in %.1fs (RTF %.2f)\n",
            state.total_samples, audio_secs, elapsed / 1000.0, (elapsed / 1000.0) / audio_secs);
    return stream_fd_owned;
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
    int stream_fd_owned = 0;
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
        stream_fd_owned = handle_tts_stream(ctx, client_fd, body);
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
    if (!stream_fd_owned) srv_conn_close(client_fd);
}

#define CONN_QUEUE_CAP 256

typedef struct {
    double parent_accept_ms;
    double parent_slot_ms;
    double parent_dispatch_ms;
    double child_receive_ms;
    unsigned long long parent_seq;
    int parent_worker;
    int free_slots_before;
    int free_slots_at_accept;
    int cap;
} server_handoff_t;

typedef struct {
    int fds[CONN_QUEUE_CAP];
    server_handoff_t handoff[CONN_QUEUE_CAP];
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

static void cq_push(conn_queue_t *q, int fd, const server_handoff_t *handoff) {
    pthread_mutex_lock(&q->mtx);
    while (q->count == CONN_QUEUE_CAP && !q->shutdown)
        pthread_cond_wait(&q->not_full, &q->mtx);
    if (q->shutdown) { pthread_mutex_unlock(&q->mtx); srv_conn_close(fd); return; }
    q->fds[q->tail] = fd;
    if (handoff) q->handoff[q->tail] = *handoff;
    else memset(&q->handoff[q->tail], 0, sizeof(q->handoff[q->tail]));
    q->tail = (q->tail + 1) % CONN_QUEUE_CAP;
    q->count++;
    pthread_cond_signal(&q->not_empty);
    pthread_mutex_unlock(&q->mtx);
}

static int cq_pop(conn_queue_t *q, server_handoff_t *handoff) {
    pthread_mutex_lock(&q->mtx);
    while (q->count == 0 && !q->shutdown)
        pthread_cond_wait(&q->not_empty, &q->mtx);
    if (q->count == 0 && q->shutdown) { pthread_mutex_unlock(&q->mtx); return -1; }
    int head = q->head;
    int fd = q->fds[head];
    if (handoff) *handoff = q->handoff[head];
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

static void qwen_thread_name(const char *prefix);
static void *worker_main(void *arg) {
    qwen_thread_name("srv-slot");
    worker_arg_t *wa = (worker_arg_t *)arg;
    for (;;) {
        int fd = cq_pop(wa->q, NULL);
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
#ifdef TCP_NODELAY
    int one = 1;
    (void)setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
#endif
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
static int srv_send_fd(int chan, int fd, const server_handoff_t *handoff) {
    char dummy = 'F';
    struct iovec iov = { .iov_base = handoff ? (void *)handoff : (void *)&dummy,
                         .iov_len = handoff ? sizeof(*handoff) : 1 };
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
    return n == (ssize_t)iov.iov_len ? 0 : -1;
}

static int srv_recv_fd(int chan, server_handoff_t *handoff) {
    char dummy;
    struct iovec iov = { .iov_base = handoff ? (void *)handoff : (void *)&dummy,
                         .iov_len = handoff ? sizeof(*handoff) : 1 };
    char cbuf[CMSG_SPACE(sizeof(int))];
    memset(cbuf, 0, sizeof cbuf);
    struct msghdr msg = { .msg_iov = &iov, .msg_iovlen = 1,
                          .msg_control = cbuf, .msg_controllen = sizeof cbuf };
    ssize_t n = recvmsg(chan, &msg, 0);
    if (n < 0 && errno == EINTR) return -3;
    if (n <= 0) return -1;
    if (handoff && n != (ssize_t)sizeof(*handoff)) return -2;
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

/* ---------------------------------------------------------------------------
 * Metrics endpoint — publishes what the process already knows, when asked.
 *
 * Deliberately NOT an instrument.  Nothing here counts anything new, nothing
 * here runs between scrapes, and not one line is added to any path a request
 * travels: every number below is state the server was already maintaining for
 * its own scheduling, rendered as Prometheus/OpenMetrics text when a scraper
 * connects and forgotten again when it disconnects.  Collection, retention,
 * rates and alerting belong to whatever is polling — Prometheus, Grafana
 * Alloy, VictoriaMetrics, or an OpenTelemetry Collector through its
 * `prometheus` receiver, which is how this reaches OTLP without a protobuf
 * client entering a server that has no dependencies.
 *
 * Per-worker series are emitted and never summed.  A reader can always add
 * them up; it cannot take an average apart again.  "One worker is wedged while
 * the others absorb the load" — inflight pinned at the cap with a flat
 * completion count — is precisely the failure a summed view hides, and it is
 * the one the soak campaigns kept finding.
 * ------------------------------------------------------------------------- */

enum { QWEN_METRICS_OFF = 0, QWEN_METRICS_SINGLE = 1, QWEN_METRICS_PREFORK = 2 };

static struct {
    int mode;
    int workers;
    int cap;                        /* per-worker slot cap, prefork only */
    int single_batched;             /* single mode: is the batched scheduler running? */
    /* Prefork: pointers to the state the parent's accept loop already owns.  Read from
       that same thread, so there is no snapshot to take and no atomic to add. */
    const int       *active;
    const pid_t     *kids;
    const int       *cur;           /* elastic slice bounds, 2 per worker, or NULL */
    const long long *dispatched_tot;
    const long long *completed_tot;
    const long long *rejected_tot;   /* [0] all workers full, [1] fd dispatch failed */
    const long long *replans_tot;
    char  *page;
    size_t page_cap;
} g_metrics;

static int g_metrics_port = 0;
static const char *g_metrics_bind = "127.0.0.1";

/* Quality of service for the scrape port.
 *
 * Rendering a page is cheap, but in --prefork it happens INSIDE the parent's dispatch loop,
 * which is the same reason it costs nothing at a sane scrape interval and the reason a runaway
 * client must not be allowed to set the pace.  A `watch -n 0.01 curl` or a scraper misconfigured
 * to 10 ms would otherwise buy itself a hundred renders a second out of the budget that hands
 * requests to workers.
 *
 * A token bucket rather than a minimum interval, for the same reason DynamoDB uses one: a fixed
 * floor punishes the legitimate case where two scrapers -- Prometheus and somebody's curl --
 * happen to land together, while doing nothing about a sustained flood.  The bucket absorbs the
 * burst and caps the average.
 *
 * Refusal is 429 with Retry-After, which is what the scrapers already understand, and it does
 * NOT render the page: a refusal must be cheaper than an answer or the limit funds the attack.
 * Refused requests do not consume tokens, so a client hammering at 100/s gets the configured
 * rate served and the rest refused, instead of locking everyone out including itself.
 *
 * Honest scope: this is QoS against accident -- a runaway loop, a bad scrape_interval -- not
 * DDoS protection.  A hostile flood is a firewall's problem, and the port is loopback-bound by
 * default precisely so that it is not reachable to flood. */
static double srv_now_ms(void);             /* defined with the request-timing helpers below */
static double g_metrics_rate = 5.0;         /* served scrapes per second; 0 disables the limit */
static double g_metrics_tokens = 0.0;
static double g_metrics_bucket_ms = 0.0;
static unsigned long long g_metrics_throttled = 0;

void qwen_tts_server_set_metrics(int port, const char *bind_addr) {
    g_metrics_port = port;
    if (bind_addr && *bind_addr) g_metrics_bind = bind_addr;
}

void qwen_tts_server_set_metrics_rate(double per_second) {
    g_metrics_rate = per_second > 0.0 ? per_second : 0.0;
}

/* Single-threaded by construction in both modes -- the prefork parent's dispatch loop, or the
   one metrics thread of a single process -- so no atomics are needed here. */
static int qwen_metrics_admit(void) {
    if (g_metrics_rate <= 0.0) return 1;
    const double burst = g_metrics_rate * 2.0 < 2.0 ? 2.0 : g_metrics_rate * 2.0;
    const double now = srv_now_ms();
    if (g_metrics_bucket_ms == 0.0) { g_metrics_bucket_ms = now; g_metrics_tokens = burst; }
    g_metrics_tokens += (now - g_metrics_bucket_ms) * 0.001 * g_metrics_rate;
    if (g_metrics_tokens > burst) g_metrics_tokens = burst;
    g_metrics_bucket_ms = now;
    if (g_metrics_tokens < 1.0) { g_metrics_throttled++; return 0; }
    g_metrics_tokens -= 1.0;
    return 1;
}

/* Deliberately not setup_listen_socket(): that one sets SO_REUSEPORT, which is right for the
   service port and wrong here.  With it, a second process could bind the same metrics port and
   answer a share of the scrapes with its own partial view — the exact defect this endpoint
   exists to avoid.  A double bind must fail, loudly, at startup.
   Non-blocking, because the parent variant accepts from inside the dispatch loop and a
   listening socket can still block on accept() after poll() says POLLIN (a client that sends
   RST first); the single-process variant polls the descriptor, so it does not spin. */
static int qwen_metrics_listen(void) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) { perror("metrics: socket"); return -1; }
    int opt = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof opt);
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof addr);
    addr.sin_family = AF_INET;
    addr.sin_port = htons((uint16_t)g_metrics_port);
    if (inet_pton(AF_INET, g_metrics_bind, &addr.sin_addr) != 1) {
        fprintf(stderr, "metrics: --metrics-bind %s is not an IPv4 address\n", g_metrics_bind);
        close(fd); return -1;
    }
    if (bind(fd, (struct sockaddr *)&addr, sizeof addr) < 0) {
        fprintf(stderr, "metrics: cannot bind %s:%d: %s\n",
                g_metrics_bind, g_metrics_port, strerror(errno));
        close(fd); return -1;
    }
    if (listen(fd, 8) < 0) { perror("metrics: listen"); close(fd); return -1; }
    int fl = fcntl(fd, F_GETFL, 0);
    if (fl >= 0) fcntl(fd, F_SETFL, fl | O_NONBLOCK);
    return fd;
}

static size_t qwen_metrics_render(char *b, size_t cap) {
    size_t n = 0;
#define QM_P(...) do { \
        if (n < cap) { \
            int _w = snprintf(b + n, cap - n, __VA_ARGS__); \
            if (_w > 0) n += ((size_t)_w < cap - n) ? (size_t)_w : (cap - n); \
        } \
    } while (0)

    const int prefork = (g_metrics.mode == QWEN_METRICS_PREFORK);

    QM_P("# HELP qwen_tts_build_info Identity of the running binary; the value is always 1.\n"
         "# TYPE qwen_tts_build_info gauge\n"
         "qwen_tts_build_info{git_rev=\"%s\",source_fp=\"%s\",simd=\"%s\",mode=\"%s\"} 1\n",
         QWEN_BUILD_GIT_REV, QWEN_BUILD_SOURCE_FP, QWEN_BUILD_SIMD,
         prefork ? "prefork" : "single");

    QM_P("# HELP qwen_tts_metrics_throttled_total Scrapes this endpoint refused with 429 "
         "because they exceeded --metrics-max-rate. Nonzero means something is polling too "
         "fast, not that the server is unhealthy.\n"
         "# TYPE qwen_tts_metrics_throttled_total counter\n"
         "qwen_tts_metrics_throttled_total %llu\n", g_metrics_throttled);

    QM_P("# HELP qwen_tts_workers Processes serving requests.\n"
         "# TYPE qwen_tts_workers gauge\n"
         "qwen_tts_workers %d\n", prefork ? g_metrics.workers : 1);

    /* A plain single server (--batch-size 1, no --prefork) has no scheduler, and the
       admitted/done/running counters it would report are never incremented by that path.
       Publishing them would be publishing zeros that look like measurements, so the series
       are ABSENT instead — which is also what a scraper is built to handle. */
    if (!prefork && !g_metrics.single_batched) {
        QM_P("# A plain single server maintains no request counters: run with --batch-size 2\n"
             "# or more, or with --prefork, for anything beyond this identity block.\n");
        return n;
    }

    QM_P("# HELP qwen_tts_worker_up Whether this worker process is alive.\n"
         "# TYPE qwen_tts_worker_up gauge\n");
    if (prefork)
        for (int w = 0; w < g_metrics.workers; w++)
            QM_P("qwen_tts_worker_up{worker=\"%d\"} %d\n", w, g_metrics.kids[w] > 0 ? 1 : 0);
    else
        QM_P("qwen_tts_worker_up{worker=\"0\"} 1\n");

    QM_P("# HELP qwen_tts_worker_inflight Requests currently in flight on this worker.\n"
         "# TYPE qwen_tts_worker_inflight gauge\n");
    if (prefork)
        for (int w = 0; w < g_metrics.workers; w++)
            QM_P("qwen_tts_worker_inflight{worker=\"%d\"} %d\n", w, g_metrics.active[w]);
    else
        QM_P("qwen_tts_worker_inflight{worker=\"0\"} %d\n", atomic_load(&g_srv.running));

    QM_P("# HELP qwen_tts_worker_slots Concurrent requests this worker will accept.\n"
         "# TYPE qwen_tts_worker_slots gauge\n");
    if (prefork)
        for (int w = 0; w < g_metrics.workers; w++)
            QM_P("qwen_tts_worker_slots{worker=\"%d\"} %d\n", w, g_metrics.cap);
    else
        QM_P("qwen_tts_worker_slots{worker=\"0\"} %d\n", g_srv.slots);

    QM_P("# HELP qwen_tts_worker_dispatched_total Requests handed to this worker.\n"
         "# TYPE qwen_tts_worker_dispatched_total counter\n");
    if (prefork)
        for (int w = 0; w < g_metrics.workers; w++)
            QM_P("qwen_tts_worker_dispatched_total{worker=\"%d\"} %lld\n",
                 w, g_metrics.dispatched_tot[w]);
    else
        QM_P("qwen_tts_worker_dispatched_total{worker=\"0\"} %d\n", atomic_load(&g_srv.admitted));

    QM_P("# HELP qwen_tts_worker_completed_total Requests this worker finished.\n"
         "# TYPE qwen_tts_worker_completed_total counter\n");
    if (prefork)
        for (int w = 0; w < g_metrics.workers; w++)
            QM_P("qwen_tts_worker_completed_total{worker=\"%d\"} %lld\n",
                 w, g_metrics.completed_tot[w]);
    else
        QM_P("qwen_tts_worker_completed_total{worker=\"0\"} %d\n", atomic_load(&g_srv.done));

    if (prefork) {
        /* What the PARENT refused, which is not the whole story and must not be read as
           such: a worker's own queue rejections and timeouts happen inside that worker and
           are invisible from here.  Naming the reason is what keeps the number honest. */
        QM_P("# HELP qwen_tts_rejected_total Requests refused, by reason. In prefork this is "
             "the parent's view only: rejections inside a worker's own queue are not visible here.\n"
             "# TYPE qwen_tts_rejected_total counter\n"
             "qwen_tts_rejected_total{reason=\"all_workers_full\"} %lld\n"
             "qwen_tts_rejected_total{reason=\"fd_dispatch_failed\"} %lld\n",
             g_metrics.rejected_tot[0], g_metrics.rejected_tot[1]);
        if (g_metrics.cur) {
            QM_P("# HELP qwen_tts_worker_cpus CPUs currently assigned to this worker.\n"
                 "# TYPE qwen_tts_worker_cpus gauge\n");
            for (int w = 0; w < g_metrics.workers; w++) {
                int lo = g_metrics.cur[2 * w], hi = g_metrics.cur[2 * w + 1];
                if (lo >= 0 && hi >= lo)
                    QM_P("qwen_tts_worker_cpus{worker=\"%d\"} %d\n", w, hi - lo + 1);
            }
            QM_P("# HELP qwen_tts_elastic_replans_total Elastic re-slicings of the CPU set.\n"
                 "# TYPE qwen_tts_elastic_replans_total counter\n"
                 "qwen_tts_elastic_replans_total %lld\n", *g_metrics.replans_tot);
        }
    } else {
        QM_P("# HELP qwen_tts_worker_waiting Requests queued and not yet started.\n"
             "# TYPE qwen_tts_worker_waiting gauge\n"
             "qwen_tts_worker_waiting{worker=\"0\"} %d\n", atomic_load(&g_srv.waiting));
        QM_P("# HELP qwen_tts_rejected_total Requests refused, by reason.\n"
             "# TYPE qwen_tts_rejected_total counter\n"
             "qwen_tts_rejected_total{reason=\"queue_full\"} %d\n"
             "qwen_tts_rejected_total{reason=\"queue_timeout\"} %d\n",
             atomic_load(&g_srv.rejected_full), atomic_load(&g_srv.rejected_stale));
        QM_P("# HELP qwen_tts_timed_out_total Requests that exceeded the per-request budget.\n"
             "# TYPE qwen_tts_timed_out_total counter\n"
             "qwen_tts_timed_out_total{worker=\"0\"} %d\n", atomic_load(&g_srv.timed_out));
    }
#undef QM_P
    return n;
}

/* One scrape: accept, discard the request, write the page, close.  Every step is
   non-blocking and bounded, because the prefork variant runs this inside the parent's
   dispatch loop and nothing a scraper does may be allowed to delay a request.  A scrape
   that cannot be written without blocking is DROPPED, not retried: losing one sample is
   cheaper than delaying the server that produced it.  The path is not examined — every
   request gets the same page — so a scraper pointed at /metrics or at / both work. */
static void qwen_metrics_answer(int cfd) {
    int fl = fcntl(cfd, F_GETFL, 0);
    if (fl >= 0) fcntl(cfd, F_SETFL, fl | O_NONBLOCK);
    char junk[1024];
    /* accept() can return before the request bytes land.  Answering and closing right then
       leaves the request arriving at a closed socket, the kernel replies RST, and the scraper
       discards the response we had already written -- which showed up as roughly one empty
       scrape in three hundred under load.  So wait for the request, but only briefly and only
       once: this runs inside the prefork dispatch loop, where an unbounded wait would be a
       scraper holding up request dispatch. 2 ms is generous on loopback and invisible next to
       the loop's own 1000 ms poll. */
    {
        struct pollfd pw = { .fd = cfd, .events = POLLIN, .revents = 0 };
        (void)poll(&pw, 1, 2);
    }
    for (int i = 0; i < 4; i++) { ssize_t r = recv(cfd, junk, sizeof junk, 0); if (r <= 0) break; }

    if (!qwen_metrics_admit()) {
        /* Retry-After is integer seconds by the HTTP grammar, so a sub-second budget cannot be
           expressed there; the body carries the exact figure for a human reading it by hand. */
        char body[192];
        int bn = snprintf(body, sizeof body,
                          "rate limited: this endpoint serves at most %.3g scrape(s) per second "
                          "(--metrics-max-rate; 0 disables). Slow your scrape_interval down.\n",
                          g_metrics_rate);
        char head[224];
        int retry = (int)(1.0 / g_metrics_rate);
        if (retry < 1) retry = 1;
        int hn = snprintf(head, sizeof head,
                          "HTTP/1.1 429 Too Many Requests\r\n"
                          "Content-Type: text/plain; charset=utf-8\r\n"
                          "Retry-After: %d\r\n"
                          "Content-Length: %d\r\n"
                          "Connection: close\r\n\r\n", retry, bn > 0 ? bn : 0);
        if (hn > 0 && write(cfd, head, (size_t)hn) == (ssize_t)hn && bn > 0) {
            ssize_t w = write(cfd, body, (size_t)bn); (void)w;
        }
        for (int i = 0; i < 4; i++) { ssize_t r = recv(cfd, junk, sizeof junk, 0); if (r <= 0) break; }
        close(cfd);
        return;
    }

    size_t n = qwen_metrics_render(g_metrics.page, g_metrics.page_cap);
    char head[192];
    int hn = snprintf(head, sizeof head,
                      "HTTP/1.1 200 OK\r\n"
                      "Content-Type: text/plain; version=0.0.4; charset=utf-8\r\n"
                      "Content-Length: %zu\r\n"
                      "Connection: close\r\n\r\n", n);
    if (hn > 0 && write(cfd, head, (size_t)hn) == (ssize_t)hn) {
        ssize_t w = write(cfd, g_metrics.page, n);
        (void)w;
    }
    /* Drain whatever else arrived before closing: closing a socket with unread data queued
       sends RST, and a scraper that gets RST discards the response we just wrote. */
    for (int i = 0; i < 4; i++) { ssize_t r = recv(cfd, junk, sizeof junk, 0); if (r <= 0) break; }
    close(cfd);
}

static const char *qwen_metrics_rate_note(void) {
    static char note[64];
    if (g_metrics_rate <= 0.0) return "rate limit OFF";
    snprintf(note, sizeof note, "max %.3g scrape/s", g_metrics_rate);
    return note;
}

static int qwen_metrics_alloc_page(int workers) {
    size_t cap = 4096 + (size_t)(workers > 0 ? workers : 1) * 512;
    g_metrics.page = (char *)malloc(cap);
    if (!g_metrics.page) return -1;
    g_metrics.page_cap = cap;
    return 0;
}

static void *qwen_metrics_thread(void *arg) {
    int lfd = (int)(intptr_t)arg;
    qwen_thread_name("metrics");
    for (;;) {
        struct pollfd p = { .fd = lfd, .events = POLLIN, .revents = 0 };
        int r = poll(&p, 1, -1);
        if (r < 0) { if (errno == EINTR) continue; break; }
        if (!(p.revents & POLLIN)) continue;
        int c = accept(lfd, NULL, NULL);
        if (c < 0) { if (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK) continue; break; }
        qwen_metrics_answer(c);
    }
    close(lfd);
    return NULL;
}

/* Single-process serving: the CPU server without --prefork, and EVERY GPU server — a CUDA or
   Metal context does not survive fork(), so main.c refuses --backend with --prefork and the
   GPU path is always this one.  Gated on g_conn_chan_fd because a prefork WORKER also enters
   qwen_tts_serve_batched(): letting each worker bind the metrics port would publish one
   worker's view as the server's, which is the defect this endpoint exists to avoid. */
static void qwen_metrics_start_single(int batched) {
    if (g_metrics_port <= 0 || g_conn_chan_fd >= 0 || g_metrics.mode != QWEN_METRICS_OFF) return;
    if (qwen_metrics_alloc_page(1) != 0) return;
    int fd = qwen_metrics_listen();
    if (fd < 0) {
        free(g_metrics.page); g_metrics.page = NULL; g_metrics.page_cap = 0;
        fprintf(stderr, "[serve] metrics: DISABLED (listener could not start)\n");
        return;
    }
    g_metrics.mode = QWEN_METRICS_SINGLE;
    g_metrics.single_batched = batched;
    pthread_t t;
    if (pthread_create(&t, NULL, qwen_metrics_thread, (void *)(intptr_t)fd) != 0) {
        g_metrics.mode = QWEN_METRICS_OFF;
        close(fd); free(g_metrics.page); g_metrics.page = NULL; g_metrics.page_cap = 0;
        fprintf(stderr, "[serve] metrics: DISABLED (thread could not start)\n");
        return;
    }
    pthread_detach(t);
    fprintf(stderr, "[serve] metrics: http://%s:%d/metrics (single process, %s)\n",
            g_metrics_bind, g_metrics_port, qwen_metrics_rate_note());
    if (!batched)
        fprintf(stderr, "[serve] metrics: this server keeps no request counters "
                        "(--batch-size 1, no --prefork), so the page carries build identity "
                        "only — use --batch-size 2 or more, or --prefork\n");
}

/* Prefork parent.  The page is rendered from inside the dispatch loop, on the one thread that
   owns active[]/kids[]/the totals — so there is no shared memory, no snapshot and no atomic
   here, and nothing in a worker changes at all.  Returns the listening descriptor for the
   caller to put in its poll set, or -1 when metrics are off or could not start. */
static int qwen_metrics_start_prefork(int workers, int cap, const int *active,
                                      const pid_t *kids, const int *cur,
                                      const long long *dispatched_tot,
                                      const long long *completed_tot,
                                      const long long *rejected_tot,
                                      const long long *replans_tot) {
    if (g_metrics_port <= 0) return -1;
    if (qwen_metrics_alloc_page(workers) != 0) return -1;
    int fd = qwen_metrics_listen();
    if (fd < 0) {
        free(g_metrics.page); g_metrics.page = NULL; g_metrics.page_cap = 0;
        fprintf(stderr, "[serve] metrics: DISABLED (listener could not start)\n");
        return -1;
    }
    g_metrics.mode           = QWEN_METRICS_PREFORK;
    g_metrics.workers        = workers;
    g_metrics.cap            = cap;
    g_metrics.active         = active;
    g_metrics.kids           = kids;
    g_metrics.cur            = cur;
    g_metrics.dispatched_tot = dispatched_tot;
    g_metrics.completed_tot  = completed_tot;
    g_metrics.rejected_tot   = rejected_tot;
    g_metrics.replans_tot    = replans_tot;
    fprintf(stderr, "[serve] metrics: http://%s:%d/metrics (prefork parent, %d workers, "
                    "one series set per worker, never summed, %s)\n",
            g_metrics_bind, g_metrics_port, workers, qwen_metrics_rate_note());
    return fd;
}

/* Only the prefork parent calls this: it is about to free the arrays g_metrics points into,
   and a renderer left holding them would read freed memory on a late scrape. */
static void qwen_metrics_stop(void) {
    g_metrics.mode = QWEN_METRICS_OFF;
    g_metrics.active = NULL; g_metrics.kids = NULL; g_metrics.cur = NULL;
    g_metrics.dispatched_tot = NULL; g_metrics.completed_tot = NULL;
    g_metrics.rejected_tot = NULL; g_metrics.replans_tot = NULL;
    free(g_metrics.page); g_metrics.page = NULL; g_metrics.page_cap = 0;
}

static volatile sig_atomic_t g_srv_dump = 0;
static void srv_dump_sig(int sig) { (void)sig; g_srv_dump = 1; }

/* SIGUSR1 asks for counters. Without a handler its default action is to KILL the process,
   so a stats signal to a non-prefork server would end it. */
static void srv_dump_counters_if_asked(void) {
    if (!g_srv_dump) return;
    g_srv_dump = 0;
    /* One delimiter per dump: a harness that signals before and after a cell needs to
       tell the two apart, and until now they were undifferentiated stderr. */
    static long long dump_seq = 0;
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    double now = (double)ts.tv_sec + ts.tv_nsec * 1e-9;
    fprintf(stderr, "[DUMP] v=1 pid=%d seq=%lld ts=%.3f clock=CLOCK_MONOTONIC begin\n",
            (int)getpid(), ++dump_seq, now);
    qwen_pool_stats_report();
    if (qwen_census_enabled()) qwen_census_report(NULL);
    if (qwen_matmat_stats_enabled()) qwen_matmat_stats_report(NULL);
    qwen_kernel_timing_report(NULL);
    qwen_costmap_dump_env();
    fprintf(stderr, "[DUMP] v=1 pid=%d seq=%lld end\n", (int)getpid(), dump_seq);
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
    /* The resolved dispatch table INSIDE the run's own log: engagement proof belongs in
       the timed run, not in a separate invocation with "the same" env. */
    if (getenv("QWEN_DISPATCH_MAP") || getenv("QWEN_SERVE_PROFILE") ||
        getenv("QWEN_SHAPE_CENSUS") || getenv("QWEN_COST_MAP"))
        qwen_dispatch_map_report(stderr, NULL);
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
    stream_output_t *out;       /* detached writer owns the socket when non-NULL */
    char *text;
    char *body;
    qwen_batch_req_t req;
    double enq_ms;
    double t_recv, t_parsed, t_admit, t_first;
    double t_client_start;
    double t_parent_accept, t_parent_slot, t_parent_dispatch, t_child_receive;
    unsigned long long parent_seq;
    int parent_worker;
    int free_slots_before, free_slots_at_accept, parent_cap;
    double t_write_attempt;
    double t_write_complete;
    unsigned long long enq_adm_seq;
    double enq_adm_ts;
    double enq_last_iter_ms;
    unsigned int life_seed;
    int client_gone;
    int cancelled;
    int timed_out;
    _Atomic long long audio_ready_samples;
    _Atomic long long first_audio_ready_us;
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

/* LS-4 deliberately uses a recent service-loop interval, not aggregate CPU
 * utilization.  The child publishes this through the shared health page; a
 * stale or incomplete sample is never a reason to admit the temporary slot. */
static int qwen_admit_util_sample_ok(const qwen_admission_health_t *health, int worker,
                                     double now_ms, double limit_ms,
                                     double *last_iter_ms, double *age_ms,
                                     const char **reason) {
    const qwen_admission_health_t *h = &health[worker];
    const unsigned long long seq = atomic_load_explicit(&h->seq, memory_order_acquire);
    const double ts = atomic_load_explicit(&h->ts_ms, memory_order_relaxed);
    const double iter = atomic_load_explicit(&h->last_iter_ms, memory_order_relaxed);
    const double age = ts > 0.0 ? now_ms - ts : 1.0e300;
    const double stale_limit = limit_ms * 2.0 > 100.0 ? limit_ms * 2.0 : 100.0;
    if (last_iter_ms) *last_iter_ms = iter;
    if (age_ms) *age_ms = age;
    if (!seq || ts <= 0.0) { if (reason) *reason = "no_sample"; return 0; }
    if (!isfinite(age) || age < 0.0 || age > stale_limit) {
        if (reason) *reason = "stale_sample";
        return 0;
    }
    if (!isfinite(iter) || iter <= 0.0) { if (reason) *reason = "warmup"; return 0; }
    if (iter >= limit_ms) { if (reason) *reason = "iteration_over_limit"; return 0; }
    if (reason) *reason = "headroom";
    return 1;
}

/* F2 reuses the existing TTFA diagnostic switch.  When it is disabled, no handoff
 * metadata is sent over the prefork channel and no extra request timestamps are read. */
static int qwen_f2_trace(void) {
    static int enabled = -1;
    if (enabled < 0) {
        const char *e = getenv("QWEN_TTFA_TRACE");
        enabled = e && e[0] && atoi(e) != 0;
    }
    return enabled;
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
    char *text = NULL;
    const char *why = NULL;
    qwen_json_string_status_t text_status = json_extract_string(body, "text", &text, &why);
    if (text_status == QWEN_JSON_STRING_INVALID) {
        json_string_error(err, errsz, "text", why);
        return NULL;
    }
    if (text_status == QWEN_JSON_STRING_ABSENT) {
        text_status = json_extract_string(body, "input", &text, &why);
        if (text_status == QWEN_JSON_STRING_INVALID) {
            json_string_error(err, errsz, "input", why);
            return NULL;
        }
    }
    if (json_validate_object(body, err, errsz)) { free(text); return NULL; }
    if (reject_unknown_fields(body, err, errsz)) { free(text); return NULL; }
    if (check_response_format(body, err, errsz)) { free(text); return NULL; }
    { double sp = json_extract_number(body, "speed", 1.0);
      if (sp < 0.25 || sp > 4.0) {
          snprintf(err, errsz, "speed %.3g out of range - allowed 0.25 to 4.0", sp);
          free(text); return NULL;
      } }
    if (text_status != QWEN_JSON_STRING_VALID || !text || text[0] == '\0') {
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

    char *speaker = NULL;
    qwen_json_string_status_t speaker_status = json_extract_string(body, "speaker", &speaker, &why);
    if (speaker_status == QWEN_JSON_STRING_INVALID) {
        json_string_error(err, errsz, "speaker", why);
        free(text); return NULL;
    }
    if (speaker_status == QWEN_JSON_STRING_ABSENT) {
        speaker_status = json_extract_string(body, "voice", &speaker, &why);
        if (speaker_status == QWEN_JSON_STRING_INVALID) {
            json_string_error(err, errsz, "voice", why);
            free(text); return NULL;
        }
    }
    if (speaker) {
        int sid = req->speaker_id;
        int bad = resolve_speaker_checked(ctx, speaker, &sid, err, errsz);
        free(speaker);
        if (bad) { free(text); return NULL; }
        req->speaker_id = sid;
    }
    char *language = NULL;
    qwen_json_string_status_t language_status = json_extract_string(body, "language", &language, &why);
    if (language_status == QWEN_JSON_STRING_INVALID) {
        json_string_error(err, errsz, "language", why);
        free(text); return NULL;
    }
    if (language) { int lid = qwen_tts_language_id(language); if (lid >= 0) req->language_id = lid; free(language); }

    req->temperature = clampf((float)json_extract_number(body, "temperature", req->temperature), 0.0f, 2.0f);
    req->top_k = (int)json_extract_number(body, "top_k", req->top_k);
    if (req->top_k < 0) req->top_k = 0;
    if (req->top_k > ctx->config.codec_vocab_size) req->top_k = ctx->config.codec_vocab_size;
    req->top_p = clampf((float)json_extract_number(body, "top_p", req->top_p), 0.0f, 1.0f);
    req->rep_penalty = clampf((float)json_extract_number(body, "rep_penalty", req->rep_penalty), 0.5f, 2.0f);
    int seed = (int)json_extract_number(body, "seed", -1);
    if (seed >= 0) req->seed = (uint32_t)seed;

    char *instruct = NULL;
    qwen_json_string_status_t instruct_status = json_extract_string(body, "instruct", &instruct, &why);
    if (instruct_status == QWEN_JSON_STRING_INVALID) {
        json_string_error(err, errsz, "instruct", why);
        free(text); free(instruct); return NULL;
    }
    if (instruct && instruct[0]) *needs_single = 1;
    free(instruct);
    char *vd = NULL;
    qwen_json_string_status_t vd_status = json_extract_string(body, "voice_design", &vd, &why);
    if (vd_status == QWEN_JSON_STRING_INVALID) {
        json_string_error(err, errsz, "voice_design", why);
        free(text); free(vd); return NULL;
    }
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

static double f2_client_start_ms(const char *request) {
    static const char key[] = "X-Qwen-F2-Client-Start-Monotonic-Ms:";
    const char *p = strstr(request, key);
    if (!p) return 0.0;
    p += sizeof(key) - 1;
    while (*p == ' ' || *p == '\t') p++;
    char *end = NULL;
    double v = strtod(p, &end);
    return (end != p && v > 0.0) ? v : 0.0;
}

static void *reader_main(void *arg) {
    qwen_thread_name("srv-read");
    reader_arg_t *ra = (reader_arg_t *)arg;
    for (;;) {
        server_handoff_t handoff;
        int fd = cq_pop(ra->cq, &handoff);
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
            if (!j) {
                send_error(fd, 503, "server allocation failure");
                srv_conn_close(fd); free(buf); continue;
            }
            atomic_init(&j->audio_ready_samples, 0);
            atomic_init(&j->first_audio_ready_us, 0);
            j->fd = fd;
            j->t_client_start = f2_client_start_ms(buf);
            j->t_parent_accept = handoff.parent_accept_ms;
            j->t_parent_slot = handoff.parent_slot_ms;
            j->t_parent_dispatch = handoff.parent_dispatch_ms;
            j->t_child_receive = handoff.child_receive_ms;
            j->parent_seq = handoff.parent_seq;
            j->parent_worker = handoff.parent_worker;
            j->free_slots_before = handoff.free_slots_before;
            j->free_slots_at_accept = handoff.free_slots_at_accept;
            j->parent_cap = handoff.cap;
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
    int lead_target_ms;
    unsigned long long lead_checks;
    unsigned long long lead_suppressed;
    unsigned long long lead_cancelled;
} sink_ctx_t;

static void sink_mark_audio_ready(batch_job_t *j, int n_samples) {
    if (!j || n_samples <= 0) return;
    long long now_us = (long long)(srv_now_ms() * 1000.0);
    long long zero = 0;
    atomic_compare_exchange_strong_explicit(&j->first_audio_ready_us, &zero, now_us,
                                            memory_order_relaxed, memory_order_relaxed);
    atomic_fetch_add_explicit(&j->audio_ready_samples, n_samples, memory_order_relaxed);
}

static int stream_lead_gate_enabled(void) {
    const char *e = getenv("QWEN_STREAM_LEAD_GATE");
    return e && e[0] && e[0] != '0';
}

static int stream_lead_target_ms(void) {
    const int def = 250;
    const char *e = getenv("QWEN_STREAM_LEAD_TARGET_MS");
    if (!e || !e[0]) return def;
    char *end = NULL;
    long v = strtol(e, &end, 10);
    if (end == e || *end != '\0' || v < 50 || v > 2000) {
        fprintf(stderr, "[serve] invalid QWEN_STREAM_LEAD_TARGET_MS=%s; using %d ms\n", e, def);
        return def;
    }
    return (int)v;
}

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
    if (j->is_stream && stream_output_enabled()) {
        j->out = stream_output_start(j->fd, j->life_seed);
        if (j->out) j->header_sent = 1; /* header is owned by the writer */
    } else if (j->is_stream) {
        /* The batched synchronous path used to wait for the first generated
         * audio before sending the response headers.  That made TTFB equal
         * TTFA and hid admission/prefill latency from the client metric.  A
         * valid stream request is already admitted here, so publish the
         * chunked response before entering the engine loop. */
        j->t_write_attempt = srv_now_ms();
        if (send_chunked_header(j->fd) == 0) {
            j->header_sent = 1;
            j->t_write_complete = srv_now_ms();
        } else {
            j->client_gone = 1;
            j->cancelled = 1;
            j->t_abort_detected = srv_now_ms();
        }
    }
    *req = j->req;
    *tag = j;
    sc->admitted++;
    fprintf(stderr, "[BATCH] admit #%d (in-flight admitted=%d, done=%d)\n",
            sc->admitted, sc->admitted - sc->done, sc->done);
    return 1;
}

static int send_pcm_chunk(int fd, const float *samples, int n) {
    /* per-thread grow-once conversion buffer: no allocation per chunk on the streaming path */
    static __thread int16_t *pcm = NULL; static __thread size_t pcm_cap = 0;
    if ((size_t)n > pcm_cap) { free(pcm); pcm = (int16_t *)malloc((size_t)n * sizeof(int16_t)); pcm_cap = pcm ? (size_t)n : 0; }
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
    return gone;
}

static void sink_on_chunk(void *ud, void *tag, float *samples, int n_samples) {
    (void)ud;
    batch_job_t *j = (batch_job_t *)tag;
    if (n_samples <= 0 || !samples) return;
    if (j->t_first == 0.0) j->t_first = srv_now_ms();
    if (j->out) {
        if (stream_output_enqueue(j->out, samples, n_samples, 1.0f) < 0) {
            j->client_gone = 1;
            j->cancelled = 1;
            if (j->t_abort_detected == 0.0) j->t_abort_detected = srv_now_ms();
        } else sink_mark_audio_ready(j, n_samples);
        return;
    }
    if (j->t_write_attempt == 0.0) j->t_write_attempt = srv_now_ms();
    if (!j->header_sent) { send_chunked_header(j->fd); j->header_sent = 1; }
    if (!j->client_gone && peer_hung_up(j->fd)) {
        j->client_gone = 1; j->t_abort_detected = srv_now_ms();
    }
    int _gone = send_pcm_chunk(j->fd, samples, n_samples);
    if (j->t_write_complete == 0.0 && !_gone) j->t_write_complete = srv_now_ms();
    if (!_gone) sink_mark_audio_ready(j, n_samples);
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
    if (j->cancelled || (j->out && stream_output_failed(j->out))) {
        j->cancelled = 1;
        if (!j->client_gone) {
            j->client_gone = 1;
            if (j->t_abort_detected == 0.0) j->t_abort_detected = srv_now_ms();
        }
        if (j->t_cancel_stop == 0.0) j->t_cancel_stop = srv_now_ms();
        return 1;
    }
    if (!qwen_cancel_on_disconnect()) return 0;
    if (!j->client_gone && j->fd >= 0 && peer_hung_up(j->fd)) {
        j->client_gone = 1; j->t_abort_detected = srv_now_ms();
    }
    if (j->client_gone && j->t_cancel_stop == 0.0) j->t_cancel_stop = srv_now_ms();
    return j->client_gone;
}

static int sink_step_allowed(void *ud, void *tag, int first_step) {
    sink_ctx_t *sc = (sink_ctx_t *)ud;
    batch_job_t *j = (batch_job_t *)tag;
    if (!sc || !j || !j->is_stream || first_step) return 1;
    sc->lead_checks++;
    /* Let cancellation through even while a stream is parked above its lead
     * target.  The normal cancellation callback owns disconnect/timeout
     * semantics and the next frame boundary remains the safe stop point. */
    if (sink_cancelled(ud, tag)) {
        sc->lead_cancelled++;
        return 1;
    }
    long long first_us = atomic_load_explicit(&j->first_audio_ready_us, memory_order_relaxed);
    long long samples = atomic_load_explicit(&j->audio_ready_samples, memory_order_relaxed);
    if (first_us <= 0 || samples <= 0) return 1;
    double elapsed_ms = srv_now_ms() - (double)first_us / 1000.0;
    double audio_ms = (double)samples * 1000.0 / (double)QWEN_TTS_SAMPLE_RATE;
    double lead_ms = audio_ms - elapsed_ms;
    if (lead_ms <= (double)sc->lead_target_ms) return 1;
    sc->lead_suppressed++;
    return 0;
}

void qwen_topology_emit(int worker, int threads, const char *configured_mask,
                        const char *mode) {
    char actual[512];
    snprintf(actual, sizeof actual, "%s", "unknown");
#if defined(__linux__)
    cpu_set_t set; CPU_ZERO(&set);
    if (sched_getaffinity(0, sizeof set, &set) == 0) {
        size_t o = 0; actual[0] = '\0';
        int ncpu = (int)sysconf(_SC_NPROCESSORS_CONF);
        if (ncpu <= 0 || ncpu > CPU_SETSIZE) ncpu = CPU_SETSIZE;
        for (int i = 0; i < ncpu; ) {
            if (!CPU_ISSET(i, &set)) { i++; continue; }
            int j = i;
            while (j + 1 < ncpu && CPU_ISSET(j + 1, &set)) j++;
            int k = (i == j) ? snprintf(actual + o, sizeof actual - o, "%s%d", o ? "," : "", i)
                             : snprintf(actual + o, sizeof actual - o, "%s%d-%d", o ? "," : "", i, j);
            if (k < 0 || (size_t)k >= sizeof actual - o) break;
            o += (size_t)k; i = j + 1;
        }
        if (!o) snprintf(actual, sizeof actual, "?");
    }
#endif
    fprintf(stderr, "[TOPOLOGY] v=1 worker=%d pid=%d configured_mask=%s actual_mask=%s "
                    "threads=%d mode=%s\n",
            worker, (int)getpid(), configured_mask ? configured_mask : "inherited",
            actual, threads, mode ? mode : "?");
    fflush(stderr);
}

static int qwen_life_trace(void) {
    static int v = -1;
    if (v < 0) v = getenv("QWEN_LIFE_TRACE") ? 1 : 0;
    return v;
}
static void qwen_life_emit(batch_job_t *j) {
    if (qwen_costmap_level()) {
        const double d = srv_now_ms();
        if (j->t_recv > 0.0 && d > j->t_recv)
            qwen_region_add_ns(QWEN_RGN_RT_REQUEST,
                               (unsigned long long)((d - j->t_recv) * 1e6));
        if (j->t_admit > 0.0 && j->enq_ms > 0.0 && j->t_admit > j->enq_ms)
            qwen_region_add_ns(QWEN_RGN_RT_ADMISSION,
                               (unsigned long long)((j->t_admit - j->enq_ms) * 1e6));
        qwen_costmap_request_done();
    }
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
                        "boot_id=%s client_start=%.3f parent_accept=%.3f parent_slot=%.3f "
                        "parent_dispatch=%.3f child_receive=%.3f recv=%.3f parsed=%.3f "
                        "enqueued=%.3f admitted=%.3f first_pcm=%.3f "
                        "write_attempt=%.3f write_complete=%.3f enq_adm_seq=%llu "
                        "enq_adm_ts=%.3f enq_last_iter_ms=%.3f parent_seq=%llu "
                        "parent_worker=%d free_slots_before=%d free_slots_at_accept=%d "
                        "parent_cap=%d\n",
                j->life_seed, (int)getpid(), qwen_boot_id(),
                j->t_client_start, j->t_parent_accept, j->t_parent_slot,
                j->t_parent_dispatch, j->t_child_receive, j->t_recv, j->t_parsed,
                j->enq_ms, j->t_admit, j->t_first,
                j->t_write_attempt, j->t_write_complete,
                j->enq_adm_seq, j->enq_adm_ts, j->enq_last_iter_ms,
                j->parent_seq, j->parent_worker, j->free_slots_before,
                j->free_slots_at_accept, j->parent_cap);
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
    int async_output = j->out != NULL;
    char m[220];
    snprintf(m, sizeof(m),
             "%s - this server accepts at most %d prompt tokens per request "
             "(roughly %ld characters); split the text or raise QWEN_BATCH_MAX_PROMPT",
             reason ? reason : "request rejected",
             qwen_tts_batch_max_prompt(), (long)qwen_tts_batch_max_prompt() * 7 / 2);
    if (j->out) {
        stream_output_finish(j->out);
        stream_output_release(j->out);
        j->out = NULL;
    } else if (j->is_stream && j->header_sent) (void)send_chunked_end(j->fd);
    else send_api_error(j->fd, 400, m, "text");
    fprintf(stderr, "[server] rejected seed=%u: %s\n", j->life_seed, reason ? reason : "?");
    if (!async_output) srv_conn_close(j->fd);
    job_free(j);
    sc->done++;
    atomic_fetch_add(&g_srv.done, 1);
    atomic_fetch_sub(&g_srv.running, 1);
}

static void sink_on_done(void *ud, void *tag, float *samples, int n_samples) {
    sink_ctx_t *sc = (sink_ctx_t *)ud;
    batch_job_t *j = (batch_job_t *)tag;
    int async_output = j->is_stream && j->out != NULL;
    if (async_output) {
        stream_output_finish(j->out);
        stream_output_release(j->out); /* writer drains and closes the socket */
        j->out = NULL;
        qwen_life_emit(j);
        free(samples);
    } else {
        qwen_life_emit(j);
        if (j->is_stream) {
        if (!j->header_sent) { send_chunked_header(j->fd); j->header_sent = 1; }
        (void)send_chunked_end(j->fd);
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
    }
    if (!async_output) srv_conn_close(j->fd);
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
    qwen_thread_name("srv-sched");
    sched_arg_t *sa = (sched_arg_t *)arg;
    int lead_gate = stream_lead_gate_enabled();
    int lead_target = lead_gate ? stream_lead_target_ms() : 0;
    sink_ctx_t sc = { .jq = sa->jq, .running = &server_running, .admitted = 0, .done = 0,
                      .lead_target_ms = lead_target };
    qwen_batch_sink_t sink = {
        .ud = &sc, .next_job = sink_next_job, .on_done = sink_on_done,
        .on_chunk = sink_on_chunk, .running = sink_running,
        .cancelled = sink_cancelled,
        .on_reject = sink_on_reject,
        .step_allowed = lead_gate ? sink_step_allowed : NULL,
    };
    if (lead_gate)
        fprintf(stderr, "[serve] playback lead gate ENABLED target=%d ms (first frame always eligible)\n",
                lead_target);
    atomic_store(&g_srv.sched_alive, 1);
    int rc = qwen_tts_serve_continuous(sa->ctx, sa->max_batch, &sink);
    if (lead_gate)
        fprintf(stderr, "[lead] checks=%llu suppressed=%llu cancellation_passthrough=%llu target_ms=%d\n",
                sc.lead_checks, sc.lead_suppressed, sc.lead_cancelled, lead_target);
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
/* Name the thread for /proc and top: the thread ownership table of a worker is then
 * readable without a debugger.  Zero cost after creation. */
static void qwen_thread_name(const char *prefix) {
    static _Atomic int counter = 0;
    char name[16];
    int n = atomic_fetch_add(&counter, 1);
    snprintf(name, sizeof name, "%.9s-%d", prefix, n);
#if defined(__APPLE__)
    pthread_setname_np(name);
#elif defined(__linux__)
    prctl(PR_SET_NAME, name, 0, 0, 0);
#endif
}
static void *single_worker_main(void *arg) {
    qwen_thread_name("srv-single");
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
        int stream_fd_owned = 0;
        if (j->is_stream) stream_fd_owned = handle_tts_stream(sw->ctx, j->fd, j->body);
        else handle_tts(sw->ctx, j->fd, j->body);
        if (!stream_fd_owned) srv_conn_close(j->fd);
        job_free(j);
    }
    return NULL;
}

static void server_default_decoder_batch(qwen_tts_ctx_t *ctx) {
    (void)ctx;
    if (getenv("QWEN_SERVER_NO_DECODER_BATCH")) {
        fprintf(stderr, "[serve] batched speech decoder OFF (QWEN_SERVER_NO_DECODER_BATCH)\n");
        return;
    }
    /* setenv() with overwrite=0 leaves an explicit QWEN_DECODER_BATCH=0 alone, so read the
       value back before announcing: saying "ON by default" to someone who just asked for it to
       be off is how an A/B ends up comparing a configuration against itself. */
    setenv("QWEN_DECODER_BATCH", "1", 0);
    const char *v = getenv("QWEN_DECODER_BATCH");
    if (v && atoi(v) != 0)
        fprintf(stderr, "[serve] batched speech decoder ON (one pass over the decoder "
                        "weights for all active slots) — QWEN_DECODER_BATCH=0 to opt out\n");
    else
        fprintf(stderr, "[serve] batched speech decoder OFF (QWEN_DECODER_BATCH=%s)\n", v);
}

static void server_default_memory_levers(qwen_tts_ctx_t *ctx) {
    int quantized = ctx->layers && (ctx->layers[0].wq_int8 || ctx->layers[0].wq_q4 || ctx->layers[0].wq_q6);
    if (!quantized) return;
    const char *e = getenv("QWEN_PREFILL_QUANT");
    int on = (e && e[0] && e[0] != '0');
    if (on) {
        setenv("QWEN_FREE_BF16", "1", 0);
        fprintf(stderr, "[serve] quantized prefill ON (explicitly requested): frees the bf16 "
                        "(~4 GB on the 1.7B) but MEASURABLY DEGRADES OUTPUT QUALITY on some "
                        "models. Base models only.\n");
    } else {
        fprintf(stderr, "[serve] quantized prefill OFF (default: it can degrade output "
                        "quality) — QWEN_PREFILL_QUANT=1 to opt in on a base model\n");
    }
}

static double prewarm_now_ms(void) {
    struct timeval tv; gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}
static void server_prewarm(qwen_tts_ctx_t *ctx) {
    if (getenv("QWEN_NO_PREWARM")) return;
    qwen_costmap_count_requests(0);   /* a pre-warm is not a request */
    /* Warm the path a REQUEST will take.  Without this the warm-up ran on whatever the CLI
     * left in the context -- language_id -1 among other things -- so it primed per-request
     * state for a configuration no request uses, and the first real request came out
     * different from every one after it (test-serve-repro's "trajectory fork"). */
    reset_request_state(ctx);
    int sv_silent = ctx->silent;
    ctx->silent = 1;
    float *aud = NULL; int n = 0;
    double t0 = prewarm_now_ms();
    int rc = qwen_tts_generate(ctx, "Warm up.", &aud, &n);
    free(aud);
    ctx->silent = sv_silent;
    qwen_costmap_count_requests(1);
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
    qwen_metrics_start_single(1);
    ctx->silent = 1;
    server_default_memory_levers(ctx);
    server_default_decoder_batch(ctx);
    qwen_exec_budget_engine_owned("serve");
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
    g_srv.batched = 1;          /* from here on, health may report the scheduler's state */
    g_srv.queue_max = jq.cap;
    g_srv.slots = max_batch;
    g_srv.queue_timeout_ms = g_cfg_queue_timeout_ms;
    srv_init_request_cap();
    fprintf(stderr, "[serve] %d slots · %d may wait (%d in the system) · queue deadline %s\n",
            max_batch, jq.cap, max_batch + jq.cap,
            g_srv.queue_timeout_ms > 0 ? "on" : "none");
    {
        /* Say it once at start instead of letting the throughput quietly not happen: above
         * the batched int8 ceiling every gate declines and a step runs one GEMV per slot. */
        int ceil_b = qwen_matmat_int8_max_b();
        if (ceil_b > 0 && max_batch > ceil_b)
            fprintf(stderr, "[serve] WARNING --batch-size %d is above the batched int8 ceiling "
                            "(B<=%d on this build): a fuller batch runs one GEMV per slot "
                            "instead of the batched kernel. See matmat.int8.batch_ceiling in "
                            "--dispatch-map.\n", max_batch, ceil_b);
    }
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

    if (getenv("QWEN_DISPATCH_MAP") || getenv("QWEN_SERVE_PROFILE") ||
        getenv("QWEN_SHAPE_CENSUS") || getenv("QWEN_COST_MAP"))
        qwen_dispatch_map_report(stderr, NULL);   /* engagement proof inside this run's log */
    /* Always: an artifact must open with what the engine is ACTUALLY doing, not with what was
     * in the environment.  Silent when nothing is set, one line per flag that is. */
    (void)qwen_effective_config_report(stderr);
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
        server_handoff_t handoff = {0};
        if (g_conn_chan_fd >= 0) {
#if defined(__linux__)
            client_fd = srv_recv_fd(g_conn_chan_fd, qwen_f2_trace() ? &handoff : NULL);
            if (client_fd == -1) break;
            if (client_fd < 0) continue;
            if (qwen_f2_trace()) handoff.child_receive_ms = srv_now_ms();
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
        cq_push(&cq, client_fd,
                (g_conn_chan_fd >= 0 && qwen_f2_trace()) ? &handoff : NULL);
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
    qwen_metrics_start_single(0);

    ctx->silent = 1;
    server_default_memory_levers(ctx);
    server_default_decoder_batch(ctx);
    qwen_exec_budget_engine_owned("serve");
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

    /* This asked the wrong question twice.  It used to read the prefill-helper opt-in, which
     * is a feature flag; 47ede94 moved it to qwen_pool_concurrent_submit_ok(), which is a real
     * capability but not THIS one.  Whether two syntheses may overlap in one process is an
     * ENGINE question, and nothing here answers it: the worker contexts are separate but the
     * engine's process-wide state is not, and that has never been audited.  47ede94 therefore
     * flipped Linux from serialised to concurrent on the strength of a predicate that does not
     * cover the risk; this restores the long-standing default until the engine state is
     * actually shown to be per-worker.
     *
     * It is NOT what makes tests/test_parallel.sh fail.  That failure is deterministic --
     * identical mel-correlations across runs, unchanged by serialising -- and is the same
     * sequential-history dependency test-serve-repro reports on this non-batched path: an
     * identical request returns a different trajectory depending on the length of the request
     * before it.  Neither reproduces on the batched/prefork server, which isolates workers by
     * process and is the production path. */
    g_serialize_synth = 1;

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
        cq_push(&q, client_fd, NULL);
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
    qwen_kernel_timing_report(NULL);
    qwen_costmap_dump_env();
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
                         int *base_out, const int *order) {
    int changed = 0, next = 0;
    for (int w = 0; w < workers; w++) {
        if (kids[w] <= 0) continue;
        int lo, hi;
        if (slice[w] > 0) { lo = next; hi = next + slice[w] - 1; next += slice[w]; }
        else              { lo = hi = ncpu - 1; }
        if (hi >= ncpu) hi = ncpu - 1;
        if (base_out[2 * w] == lo && base_out[2 * w + 1] == hi) continue;
        cpu_set_t set; CPU_ZERO(&set);
        for (int c = lo; c <= hi; c++) CPU_SET(order[c], &set);
        if (sched_setaffinity(kids[w], sizeof set, &set) != 0) {
            perror("sched_setaffinity(child)");
            continue;
        }
        base_out[2 * w] = lo; base_out[2 * w + 1] = hi;
        changed = 1;
    }
    return changed;
}

/* Order the logical CPUs CORE-MAJOR: core 0's threads, then core 1's, ...
 *
 * Prefork slices this order contiguously, so the slice model stays a range -- but a range over
 * PHYSICAL cores instead of over Linux's logical numbering.  It matters because Linux numbers
 * every core's first thread before any sibling: on the 12-core SMT-2 host we measure on
 * (siblings cpu N and cpu N+12), `--prefork 2` used to give worker 0 cpus 0-11 and worker 1
 * cpus 12-23 -- the SAME twelve physical cores, one worker per hyperthread.  The workers were
 * not isolated at all, and the AMX tile unit is per physical core, so they serialised on it too.
 * Core-major ordering gives worker 0 cores 0-5 (both threads) and worker 1 cores 6-11.
 *
 * Falls back to the identity order when sysfs is unavailable or inconsistent; on a host without
 * SMT the two orders are the same anyway. */
static int qwen_cpu_core_major_order(int *order, int ncpu, const int *cand) {
    /* ARM-3: `cand` is the list of cpu ids to order (length ncpu).  NULL means 0..ncpu-1,
     * which is the historical behaviour, bit for bit. */
#define QCPU(i) (cand ? cand[(i)] : (i))
    for (int i = 0; i < ncpu; i++) order[i] = QCPU(i);
#if defined(__linux__)
    int *pkg = (int *)malloc((size_t)ncpu * sizeof(int));
    int *core = (int *)malloc((size_t)ncpu * sizeof(int));
    if (!pkg || !core) { free(pkg); free(core); return 0; }
    for (int c = 0; c < ncpu; c++) {
        char path[128]; FILE *f; long v;
        pkg[c] = core[c] = -1;
        snprintf(path, sizeof path,
                 "/sys/devices/system/cpu/cpu%d/topology/physical_package_id", QCPU(c));
        if ((f = fopen(path, "r"))) { if (fscanf(f, "%ld", &v) == 1) pkg[c] = (int)v; fclose(f); }
        snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu%d/topology/core_id", QCPU(c));
        if ((f = fopen(path, "r"))) { if (fscanf(f, "%ld", &v) == 1) core[c] = (int)v; fclose(f); }
        if (pkg[c] < 0 || core[c] < 0) { free(pkg); free(core); return 0; }
    }
    int n = 0;
    for (int c = 0; c < ncpu; c++) {          /* first thread of each core, in first-seen order */
        int seen = 0;
        for (int d = 0; d < c; d++) if (pkg[d] == pkg[c] && core[d] == core[c]) { seen = 1; break; }
        if (seen) continue;
        if (n >= ncpu) { free(pkg); free(core); for (int i = 0; i < ncpu; i++) order[i] = QCPU(i); return 0; }
        order[n++] = QCPU(c);
        for (int d = c + 1; d < ncpu; d++)     /* then that core's siblings, next to it */
            if (pkg[d] == pkg[c] && core[d] == core[c]) {
                if (n >= ncpu) { free(pkg); free(core); for (int i = 0; i < ncpu; i++) order[i] = QCPU(i); return 0; }
                order[n++] = QCPU(d);
            }
    }
    free(pkg); free(core);
    if (n != ncpu) { for (int i = 0; i < ncpu; i++) order[i] = QCPU(i); return 0; }
    return 1;
#else
    (void)ncpu; return 0;
#endif
#undef QCPU
}

int qwen_tts_serve_prefork(qwen_tts_ctx_t *ctx, int port, int workers,
                           int threads_per, int max_batch) {
    const int elastic = getenv("QWEN_PREFORK_ELASTIC") &&
                        atoi(getenv("QWEN_PREFORK_ELASTIC")) != 0;
    /* With --max-queue 0, a full prefork parent must still accept the connection so it can
     * return the documented immediate 503.  Previously the parent stopped polling the
     * listening socket while every child slot was occupied; the client then waited in the
     * kernel backlog and the child's queue deadline could never see that wait.  Keep the
     * historical backlog behaviour for the default grace queue and for the explicit old
     * unbounded A/B override. */
    const int reject_full_at_parent = (g_cfg_max_queue == 0 && !getenv("QWEN_QUEUE_UNBOUNDED"));
    if (workers < 1) workers = 1;
    /* ARM-3: plan on the cpus this process is ALLOWED to use, not on the machine's.
     * sysconf(_SC_NPROCESSORS_ONLN) sees neither an inherited taskset mask nor a cpuset
     * cgroup, so a pinned run silently escaped its mask and a container planned the whole
     * host.  qwen_lane_split_prepare() next door already reads the mask; this makes the
     * prefork planner agree with it.  Full mask => byte-identical behaviour to before. */
    int allow_n = 0; int *allow = NULL;
#if defined(__linux__)
    {   cpu_set_t aff; CPU_ZERO(&aff);
        if (sched_getaffinity(0, sizeof aff, &aff) == 0) {
            const int an = CPU_COUNT(&aff);
            if (an > 0 && (allow = (int *)malloc((size_t)an * sizeof(int)))) {
                int k = 0;
                for (int c = 0; c < CPU_SETSIZE && k < an; c++)
                    if (CPU_ISSET(c, &aff)) allow[k++] = c;
                allow_n = k;
            }
        }
    }
#endif
    const int online = (int)sysconf(_SC_NPROCESSORS_ONLN);
    const int ncpu = allow_n > 0 ? allow_n : online;
    if (allow_n > 0 && allow_n != online)
        fprintf(stderr, "prefork: inherited cpu mask has %d of %d online cpus; planning on the mask\n",
                allow_n, online);
    if (allow_n == online) { free(allow); allow = NULL; }   /* identity: keep the old path */
    const int per = ncpu / workers > 0 ? ncpu / workers : 1;
    int *cpu_order = (int *)malloc((size_t)(ncpu > 0 ? ncpu : 1) * sizeof(int));
    if (!cpu_order) { free(allow); return -1; }
    const int core_major = qwen_cpu_core_major_order(cpu_order, ncpu, allow);
    if (threads_per < 1) threads_per = per;
    const int cap = max_batch >= 1 ? max_batch : 1;
    int admit_util = qwen_admit_util_requested() && cap == 2;
    const double admit_util_limit = qwen_admit_util_limit_ms();
    const int admit_util_trace = qwen_admit_util_trace();
    qwen_admission_health_t *admit_health = NULL;
    if (qwen_admit_util_requested() && cap != 2) {
        fprintf(stderr, "[serve] QWEN_ADMIT_UTIL requires --batch-size 2; disabled\n");
        admit_util = 0;
    }
#if defined(__linux__)
    if (admit_util) {
        size_t bytes = (size_t)workers * sizeof(*admit_health);
        admit_health = (qwen_admission_health_t *)mmap(NULL, bytes,
                            PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
        if (admit_health == MAP_FAILED) {
            fprintf(stderr, "[serve] QWEN_ADMIT_UTIL shared health allocation failed; disabled\n");
            admit_health = NULL;
            admit_util = 0;
        } else {
            for (int w = 0; w < workers; w++) {
                atomic_init(&admit_health[w].seq, 0);
                atomic_init(&admit_health[w].ts_ms, 0.0);
                atomic_init(&admit_health[w].last_iter_ms, 0.0);
            }
        }
    }
#else
    if (admit_util) {
        fprintf(stderr, "[serve] QWEN_ADMIT_UTIL is Linux-prefork-only; disabled\n");
        admit_util = 0;
    }
#endif

    qwen_provenance_report(stderr);
    int listen_fd = setup_listen_socket(port);
    if (listen_fd < 0) return -1;

    int (*sp)[2] = (int (*)[2])calloc((size_t)workers, sizeof(int[2]));
    pid_t *kids = (pid_t *)calloc((size_t)workers, sizeof(pid_t));
    long long *assigned = (long long *)calloc((size_t)workers, sizeof(long long));
    long long *completed = (long long *)calloc((size_t)workers, sizeof(long long));
    /* The [prefork-stats] dump on SIGUSR1 zeroes assigned[]/completed[] by design, so a
       reader can ask "what happened since I last asked".  A Prometheus counter may never go
       backwards, so the metrics page reads these never-reset twins instead. */
    long long *dispatched_tot = (long long *)calloc((size_t)workers, sizeof(long long));
    long long *completed_tot  = (long long *)calloc((size_t)workers, sizeof(long long));
    long long rejected_tot[2] = { 0, 0 };
    int *active = (int *)calloc((size_t)workers, sizeof(int));
    int *slice = (int *)calloc((size_t)workers, sizeof(int));
    int *cur = (int *)malloc((size_t)workers * 2 * sizeof(int));
    long long rejected = 0, replans = 0;
    if (cur) for (int w = 0; w < 2 * workers; w++) cur[w] = -1;
    if (!sp || !kids || !assigned || !completed || !active || !slice || !cur) return -1;
    if (!dispatched_tot || !completed_tot) return -1;

    {
        /* Say the host topology out loud before any measurement starts.  A benchmark that does
         * not know whether SMT is on is not a benchmark: with siblings enumerated N and N+12,
         * a contiguous logical slice hands two workers the same physical cores, and the AMX
         * tile unit is per CORE, so they serialise on it while the numbers look like isolation. */
        int cores = 0, smt = 1;
#if defined(__linux__)
        for (int c = 0; c < ncpu; c++) {
            char path[128]; FILE *f; long v = -1;
            snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu%d/topology/core_id", c);
            if ((f = fopen(path, "r"))) { if (fscanf(f, "%ld", &v) != 1) v = -1; fclose(f); }
            if (v < 0) { cores = 0; break; }
            int dup = 0;
            for (int d = 0; d < c; d++) {
                char p2[128]; FILE *g; long v2 = -1;
                snprintf(p2, sizeof p2, "/sys/devices/system/cpu/cpu%d/topology/core_id", d);
                if ((g = fopen(p2, "r"))) { if (fscanf(g, "%ld", &v2) != 1) v2 = -1; fclose(g); }
                if (v2 == v) { dup = 1; break; }
            }
            if (!dup) cores++;
        }
#endif
        if (cores > 0) smt = ncpu / cores;
        fprintf(stderr, "prefork topology: %d logical cpus", ncpu);
        if (cores > 0) fprintf(stderr, " = %d physical cores x %d SMT", cores, smt);
        fprintf(stderr, " · %d workers x %d threads · %d cpus per worker (%s)\n",
                workers, threads_per, per,
                core_major ? "core-major masks" : "logical masks, sysfs topology unavailable");
        if (cores > 0 && smt > 1)
            fprintf(stderr, "prefork: ⚠️  SMT is ON (%d threads per core). Each worker's mask "
                            "covers %d physical cores, both siblings each; the AMX tile unit is "
                            "PER CORE, so two busy threads on one core share it. For a clean "
                            "measurement disable SMT on the host or give each worker one thread "
                            "per core.\n", smt, per / smt > 0 ? per / smt : 1);
        if (cores > 0 && workers * per > ncpu)
            fprintf(stderr, "prefork: ⚠️  worker masks OVERLAP (%d workers x %d cpus > %d)\n",
                    workers, per, ncpu);
    }
    fprintf(stderr, "prefork: %d workers x %d threads, %d cpus (%d per worker), "
                    "cap %d in flight each, port %d%s\n",
            workers, threads_per, ncpu, per, cap, port,
            elastic ? " · ELASTIC core allocation" : "");
    if (admit_util)
        fprintf(stderr, "prefork: QWEN_ADMIT_UTIL ON · parent cap=%d · one transient extra slot "
                        "· iteration limit %.1f ms · child batch=%d%s\n",
                cap, admit_util_limit, cap + 1, admit_util_trace ? " · trace ON" : "");
    if (reject_full_at_parent)
        fprintf(stderr, "prefork: --max-queue 0 -> accept and return 503 immediately when all "
                        "worker slots are occupied\n");
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
            char cpulist[128]; int cl = 0; cpulist[0] = 0;
            for (int c = w * per; c < (w + 1) * per && c < ncpu; c++) {
                int lc = cpu_order[c];
                CPU_SET(lc, &set);
                if (cl < (int)sizeof cpulist - 8)
                    cl += snprintf(cpulist + cl, sizeof cpulist - (size_t)cl,
                                   "%s%d", cl ? "," : "", lc);
            }
            if (sched_setaffinity(0, sizeof(set), &set) != 0) perror("sched_setaffinity");
            qwen_threadpool_after_fork();
            qwen_costmap_after_fork();
            { int lane_threads = threads_per;
              if (qwen_lane_split_prepare(&lane_threads)) threads_per = lane_threads; }
            qwen_set_threads(threads_per);
            /* Print the mask that was actually SET, not the slice indices: on an SMT host
             * those are no longer the same thing, and the mask is what a run manifest needs. */
            fprintf(stderr, "prefork: worker %d pid %d cpus %s threads %d (%s)\n",
                    w, (int)getpid(), cpulist, threads_per,
                    core_major ? "core-major slice, siblings kept together"
                               : "logical slice, no sysfs topology");
            qwen_topology_emit(w, threads_per, cpulist, "prefork");
            if (admit_util)
                qwen_admission_health_bind(admit_health, w);
            const int child_batch = admit_util ? cap + 1 : max_batch;
            int rc = (child_batch >= 2) ? qwen_tts_serve_batched(ctx, port, child_batch)
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

    /* workers + listen_fd + metrics_fd */
    struct pollfd *pfd = (struct pollfd *)calloc((size_t)workers + 2, sizeof(struct pollfd));
    if (!pfd) return -1;
    /* Started here, AFTER the fork loop, so no worker inherits the listening descriptor:
       a worker answering scrapes would publish its own slice as if it were the server. */
    const int metrics_fd = qwen_metrics_start_prefork(workers, cap, active, kids,
                                                      elastic ? cur : NULL,
                                                      dispatched_tot, completed_tot,
                                                      rejected_tot, &replans);
    long long dispatched = 0;
    unsigned long long admit_parent_seq = 0;
    unsigned long long f2_parent_seq = 0;
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
        if (free_slots > 0 || reject_full_at_parent || admit_util) {
            li = nf;
            pfd[nf].fd = listen_fd; pfd[nf].events = POLLIN; pfd[nf].revents = 0;
            nf++;
        }
        /* Polled unconditionally, unlike listen_fd: a saturated server is precisely when
           somebody needs to read the numbers, so metrics must not vanish with capacity. */
        int mi = -1;
        if (metrics_fd >= 0) {
            mi = nf;
            pfd[nf].fd = metrics_fd; pfd[nf].events = POLLIN; pfd[nf].revents = 0;
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
            /* the counters that matter (shape census, batch audit, kernel timing) live in
               the workers: forward the request so every process dumps under [DUMP] */
            for (int w = 0; w < workers; w++) if (kids[w] > 0) kill(kids[w], SIGUSR1);
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
                completed_tot[w] += n;
                active[w] -= (int)n;
                if (active[w] < 0) active[w] = 0;
                if (elastic) {
                    elastic_plan(workers, ncpu, active, slice);
                    replans += elastic_apply(workers, ncpu, slice, kids, cur, cpu_order);
                }
            } else if (n == 0 || (n < 0 && errno != EINTR && errno != EAGAIN)) {
                fprintf(stderr, "prefork: worker %d (pid %d) channel closed\n", w, (int)kids[w]);
                close(sp[w][0]); kids[w] = -1; active[w] = 0;
            }
        }
        if (mi >= 0 && (pfd[mi].revents & POLLIN)) {
            int mfd = accept(metrics_fd, NULL, NULL);
            if (mfd >= 0) qwen_metrics_answer(mfd);
        }
        if (li < 0 || !(pfd[li].revents & POLLIN)) continue;

        struct sockaddr_in ca; socklen_t cl = sizeof(ca);
        int cfd = accept(listen_fd, (struct sockaddr *)&ca, &cl);
        if (cfd < 0) { if (errno == EINTR || errno == EAGAIN) continue; perror("accept"); continue; }

        const double f2_accept_ms = qwen_f2_trace() ? srv_now_ms() : 0.0;
        const double f2_slot_ms = qwen_f2_trace() ? srv_now_ms() : 0.0;
        int f2_free_at_accept = 0;
        if (qwen_f2_trace())
            for (int w = 0; w < workers; w++)
                if (kids[w] > 0 && active[w] < cap) f2_free_at_accept++;
        const unsigned long long f2_seq = qwen_f2_trace() ? ++f2_parent_seq : 0;
        const unsigned long long admit_seq = admit_util ? ++admit_parent_seq : 0;

        int best = -1;
        int temporary_extra = 0;
        for (int w = 0; w < workers; w++) {
            if (kids[w] <= 0 || active[w] >= cap) continue;
            if (best < 0 || active[w] < active[best]) best = w;
        }
        if (best < 0 && admit_util) {
            int extra_in_use = 0;
            for (int w = 0; w < workers; w++)
                if (kids[w] > 0 && active[w] > cap) extra_in_use = 1;
            if (!extra_in_use) {
                const double now_ms = srv_now_ms();
                double best_iter = 1.0e300;
                int saw_full = 0;
                for (int w = 0; w < workers; w++) {
                    if (kids[w] <= 0 || active[w] != cap) continue;
                    saw_full = 1;
                    double iter = 0.0, age = 1.0e300;
                    const char *reason = "unavailable";
                    int ok = qwen_admit_util_sample_ok(admit_health, w, now_ms,
                                                       admit_util_limit, &iter, &age, &reason);
                    if (admit_util_trace)
                        fprintf(stderr,
                                "[ADMITUTIL] v=1 clock=CLOCK_MONOTONIC worker=%d active=%d "
                                "cap=%d last_iter_ms=%.3f age_ms=%.3f limit_ms=%.3f "
                                "decision=%s reason=%s request_seq=%llu\n",
                                w, active[w], cap, iter, age, admit_util_limit,
                                ok ? "candidate" : "reject", reason, admit_seq);
                    if (ok && (best < 0 || iter < best_iter)) {
                        best = w; best_iter = iter;
                    }
                }
                if (best >= 0) {
                    temporary_extra = 1;
                    if (admit_util_trace)
                        fprintf(stderr,
                                "[ADMITUTIL] v=1 clock=CLOCK_MONOTONIC worker=%d active=%d "
                                "cap=%d last_iter_ms=%.3f age_ms=%.3f limit_ms=%.3f "
                                "decision=admit3 reason=headroom request_seq=%llu\n",
                                best, active[best], cap, best_iter,
                                now_ms - atomic_load_explicit(&admit_health[best].ts_ms,
                                                               memory_order_relaxed),
                                admit_util_limit, admit_seq);
                } else if (admit_util_trace && saw_full) {
                    fprintf(stderr,
                            "[ADMITUTIL] v=1 clock=CLOCK_MONOTONIC worker=-1 active=%d "
                            "cap=%d last_iter_ms=-1 age_ms=-1 limit_ms=%.3f "
                            "decision=reject reason=no_healthy_worker request_seq=%llu\n",
                            cap, cap, admit_util_limit, admit_seq);
                }
            } else if (admit_util_trace) {
                fprintf(stderr,
                        "[ADMITUTIL] v=1 clock=CLOCK_MONOTONIC worker=-1 active=%d "
                        "cap=%d last_iter_ms=-1 age_ms=-1 limit_ms=%.3f "
                        "decision=reject reason=extra_slot_in_use request_seq=%llu\n",
                        cap, cap, admit_util_limit, admit_seq);
            }
        }
        if (best < 0) {
            rejected++; rejected_tot[0]++;
            if (qwen_f2_trace())
                fprintf(stderr, "[F2REJECT] v=1 seq=%llu reason=all_workers_full "
                                "clock=CLOCK_MONOTONIC accept=%.3f slot=%.3f "
                                "free_slots_before=%d free_slots_at_accept=%d cap=%d\n",
                        f2_seq, f2_accept_ms, f2_slot_ms, free_slots,
                        f2_free_at_accept, cap);
            send_error(cfd, 503, "all workers at capacity");
            close(cfd);
            continue;
        }
        set_client_timeout(cfd);
        if (elastic) {
            active[best]++;
            elastic_plan(workers, ncpu, active, slice);
            replans += elastic_apply(workers, ncpu, slice, kids, cur, cpu_order);
            active[best]--;
        }
        server_handoff_t handoff = {0};
        if (qwen_f2_trace()) {
            handoff.parent_accept_ms = f2_accept_ms;
            handoff.parent_slot_ms = f2_slot_ms;
            handoff.parent_dispatch_ms = srv_now_ms();
            handoff.parent_seq = f2_seq;
            handoff.parent_worker = best;
            handoff.free_slots_before = free_slots;
            handoff.free_slots_at_accept = f2_free_at_accept;
            handoff.cap = cap;
        }
        if (srv_send_fd(sp[best][0], cfd, qwen_f2_trace() ? &handoff : NULL) != 0) {
            rejected++; rejected_tot[1]++;
            if (qwen_f2_trace())
                fprintf(stderr, "[F2REJECT] v=1 seq=%llu reason=fd_dispatch_failed "
                                "clock=CLOCK_MONOTONIC accept=%.3f slot=%.3f "
                                "free_slots_before=%d free_slots_at_accept=%d cap=%d\n",
                        f2_seq, f2_accept_ms, f2_slot_ms, free_slots,
                        f2_free_at_accept, cap);
            close(cfd);
            continue;
        }
        close(cfd);
        active[best]++; assigned[best]++; dispatched++; dispatched_tot[best]++;
        if (admit_util_trace && temporary_extra)
            fprintf(stderr, "[ADMITUTIL] v=1 clock=CLOCK_MONOTONIC worker=%d active=%d "
                            "cap=%d last_iter_ms=-1 age_ms=-1 limit_ms=%.3f "
                            "decision=dispatched reason=temporary_slot request_seq=%llu\n",
                    best, active[best], cap, admit_util_limit, admit_seq);
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
    if (metrics_fd >= 0) close(metrics_fd);
    qwen_metrics_stop();
    free(pfd); free(sp); free(kids); free(assigned); free(completed); free(active);
    free(dispatched_tot); free(completed_tot);
    free(act_area_w); free(slice); free(cur); free(cpu_order); free(allow);
#if defined(__linux__)
    if (admit_health)
        munmap(admit_health, (size_t)workers * sizeof(*admit_health));
#endif
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
