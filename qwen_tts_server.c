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
        if (*end == '\\') end++;
        end++;
    }
    int len = (int)(end - p);
    char *result = (char *)malloc(len + 1);
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
    /* Convert float to s16le */
    int16_t *pcm = (int16_t *)malloc(n_samples * sizeof(int16_t));
    for (int i = 0; i < n_samples; i++) {
        float s = samples[i];
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
    /* Reset to default speaker (Ryan) and language (English) */
    ctx->speaker_id = 3061;   /* ryan */
    ctx->language_id = 2050;  /* English */

    /* Reset sampling params to defaults */
    ctx->temperature = 0.9f;
    ctx->top_k = 50;
    ctx->top_p = 1.0f;
    ctx->rep_penalty = 1.05f;

    /* Reset transient flags */
    ctx->voice_design = 0;
    free(ctx->instruct);
    ctx->instruct = NULL;

    /* Fresh seed per request (time-based) */
    struct timeval tv;
    gettimeofday(&tv, NULL);
    ctx->seed = (uint32_t)(tv.tv_sec ^ tv.tv_usec);
}

/* Apply TTS params from JSON body to context. Returns text (malloc'd) or NULL on error. */
static char *parse_tts_request(qwen_tts_ctx_t *ctx, const char *body) {
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

    char *speaker = json_extract_string(body, "speaker");
    if (!speaker) speaker = json_extract_string(body, "voice");
    if (speaker) {
        int sid = qwen_tts_speaker_id(speaker);
        if (sid >= 0) ctx->speaker_id = sid;
        free(speaker);
    }

    char *language = json_extract_string(body, "language");
    if (language) {
        int lid = qwen_tts_language_id(language);
        if (lid >= 0) ctx->language_id = lid;
        free(language);
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

    /* Sampling params (override defaults only if provided) */
    ctx->temperature = (float)json_extract_number(body, "temperature", ctx->temperature);
    ctx->top_k = (int)json_extract_number(body, "top_k", ctx->top_k);
    ctx->top_p = (float)json_extract_number(body, "top_p", ctx->top_p);
    ctx->rep_penalty = (float)json_extract_number(body, "rep_penalty", ctx->rep_penalty);

    /* Seed (optional: 0 or negative = keep time-based from reset) */
    int seed = (int)json_extract_number(body, "seed", -1);
    if (seed >= 0) ctx->seed = (uint32_t)seed;

    return text;
}

static double server_time_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

static void handle_tts(qwen_tts_ctx_t *ctx, int fd, const char *body) {
    char *text = parse_tts_request(ctx, body);
    if (!text) {
        send_error(fd, 400, "missing 'text' field");
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
    if (qwen_tts_generate(ctx, text, &audio, &n_samples) != 0 || !audio || n_samples == 0) {
        send_error(fd, 500, "generation failed");
        free(text);
        free(audio);
        return;
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
    fprintf(stderr, "[HTTP] Sent %d bytes WAV (%.2fs audio) in %.1fs (%.1fx realtime)\n",
            wav_size, audio_secs, elapsed / 1000.0, audio_secs / (elapsed / 1000.0));
}

static void handle_tts_stream(qwen_tts_ctx_t *ctx, int fd, const char *body) {
    char *text = parse_tts_request(ctx, body);
    if (!text) {
        send_error(fd, 400, "missing 'text' field");
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

    /* Set up streaming */
    stream_http_state_t state = { .fd = fd, .total_samples = 0 };
    ctx->stream = 1;
    ctx->stream_chunk_frames = 10;
    qwen_tts_set_audio_callback(ctx, stream_http_callback, &state);

    /* Send chunked response header */
    send_chunked_header(fd);

    float *audio = NULL;
    int n_samples = 0;
    qwen_tts_generate(ctx, text, &audio, &n_samples);
    free(audio);
    free(text);

    /* Terminate chunked encoding */
    send_chunked_end(fd);

    /* Clean up streaming state */
    ctx->stream = 0;
    ctx->audio_cb = NULL;

    double elapsed = server_time_ms() - t0;
    float audio_secs = (float)state.total_samples / QWEN_TTS_SAMPLE_RATE;
    fprintf(stderr, "[HTTP] Streamed %d samples (%.2fs audio) in %.1fs (%.1fx realtime)\n",
            state.total_samples, audio_secs, elapsed / 1000.0, audio_secs / (elapsed / 1000.0));
}

/* ── Main server loop ────────────────────────────────────────────────── */

static volatile int server_running = 1;

static void sigint_handler(int sig) {
    (void)sig;
    server_running = 0;
}

int qwen_tts_serve(qwen_tts_ctx_t *ctx, int port) {
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
        perror("bind");
        close(server_fd);
        return -1;
    }

    if (listen(server_fd, 5) < 0) {
        perror("listen");
        close(server_fd);
        return -1;
    }

    struct sigaction sa = { .sa_handler = sigint_handler };
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0; /* no SA_RESTART — let accept() return EINTR */
    sigaction(SIGINT, &sa, NULL);
    sigaction(SIGTERM, &sa, NULL);
    signal(SIGPIPE, SIG_IGN);

    fprintf(stderr, "Server listening on http://0.0.0.0:%d\n", port);
    fprintf(stderr, "Endpoints:\n");
    fprintf(stderr, "  POST /v1/tts          — generate speech (returns WAV)\n");
    fprintf(stderr, "  POST /v1/tts/stream   — generate speech (chunked PCM stream)\n");
    fprintf(stderr, "  POST /v1/audio/speech — OpenAI-compatible TTS\n");
    fprintf(stderr, "  GET  /v1/speakers     — list speakers\n");
    fprintf(stderr, "  GET  /v1/health       — health check\n\n");
    fprintf(stderr, "Examples:\n");
    fprintf(stderr, "  # Full WAV:\n");
    fprintf(stderr, "  curl -s http://localhost:%d/v1/tts -d '{\"text\":\"Hello world\"}' -o out.wav\n\n", port);
    fprintf(stderr, "  # Streaming playback (macOS):\n");
    fprintf(stderr, "  curl -sN http://localhost:%d/v1/tts/stream -d '{\"text\":\"Hello world\"}' | "
                    "ffplay -f s16le -ar 24000 -ac 1 -nodisp -autoexit -\n\n", port);
    fprintf(stderr, "  # With options:\n");
    fprintf(stderr, "  curl -s http://localhost:%d/v1/tts -d '{\"text\":\"Hello\",\"speaker\":\"ryan\","
                    "\"language\":\"English\"}' -o out.wav\n\n", port);
    fprintf(stderr, "Press Ctrl+C to stop.\n\n");

    /* Suppress model output during request handling */
    ctx->silent = 1;

    while (server_running) {
        struct sockaddr_in client_addr;
        socklen_t client_len = sizeof(client_addr);
        int client_fd = accept(server_fd, (struct sockaddr *)&client_addr, &client_len);
        if (client_fd < 0) {
            if (errno == EINTR) continue;
            perror("accept");
            continue;
        }

        /* Read request */
        char *buf = (char *)malloc(1024 * 1024); /* 1MB max request */
        int total = read_request(client_fd, buf, 1024 * 1024);
        if (total <= 0) { free(buf); close(client_fd); continue; }

        /* Parse method and path */
        char method[16] = {0}, path[256] = {0};
        sscanf(buf, "%15s %255s", method, path);

        /* Find body (after \r\n\r\n) */
        const char *body = strstr(buf, "\r\n\r\n");
        if (body) body += 4;
        else body = "";

        char *client_ip = inet_ntoa(client_addr.sin_addr);
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
        /* Route requests */
        else if (strcmp(path, "/v1/health") == 0 && strcmp(method, "GET") == 0) {
            handle_health(client_fd);
        }
        else if (strcmp(path, "/v1/speakers") == 0 && strcmp(method, "GET") == 0) {
            handle_speakers(client_fd);
        }
        else if (strcmp(path, "/v1/tts") == 0 && strcmp(method, "POST") == 0) {
            handle_tts(ctx, client_fd, body);
        }
        else if (strcmp(path, "/v1/tts/stream") == 0 && strcmp(method, "POST") == 0) {
            handle_tts_stream(ctx, client_fd, body);
        }
        else if (strcmp(path, "/v1/audio/speech") == 0 && strcmp(method, "POST") == 0) {
            /* OpenAI-compatible: same as /v1/tts */
            handle_tts(ctx, client_fd, body);
        }
        else {
            send_error(client_fd, 404, "not found");
        }

        free(buf);
        close(client_fd);
    }

    close(server_fd);
    fprintf(stderr, "\nServer stopped.\n");
    return 0;
}
