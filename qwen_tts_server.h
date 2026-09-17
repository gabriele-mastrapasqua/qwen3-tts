/* qwen_tts_server.h - Minimal HTTP server for Qwen3-TTS */
#ifndef QWEN_TTS_SERVER_H
#define QWEN_TTS_SERVER_H

#include "qwen_tts.h"

int qwen_tts_serve(qwen_tts_ctx_t *ctx, int port);

int qwen_tts_serve_ex(qwen_tts_ctx_t *ctx, int port, int n_workers);

int qwen_tts_serve_batched(qwen_tts_ctx_t *ctx, int port, int max_batch);

void qwen_tts_server_set_limits(int max_queue, int queue_timeout_ms);

/* Emit one machine-readable line describing this process's execution domain:
 *   [TOPOLOGY] v=1 worker=N pid=P configured_mask=M actual_mask=A threads=K mode=X
 * Both masks, because they are not the same claim: `configured` is what the invocation
 * asked for, `actual` is what the OS reports for this process right now.  A topology
 * label such as "1x8" says neither, and on a multi-CCX host the difference between
 * cpus 0-7 and cpus 0-15 is a different bandwidth domain, not a smaller one. */
void qwen_topology_emit(int worker, int threads, const char *configured_mask,
                        const char *mode);
/* Enable the read-only metrics page on its own port.  Server mode only, and main.c refuses
 * the flag without --serve: there is nothing to publish about a one-shot CLI run. */
void qwen_tts_server_set_metrics(int port, const char *bind_addr);
/* Served scrapes per second, token bucket with a burst of twice the rate. 0 disables the
 * limit. Refusals are 429 + Retry-After and do not render the page. */
void qwen_tts_server_set_metrics_rate(double per_second);
void qwen_tts_server_set_max_request_ms(int ms);
void qwen_tts_server_set_max_text_chars(int chars);

#endif
