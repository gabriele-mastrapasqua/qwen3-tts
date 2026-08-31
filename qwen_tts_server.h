/* qwen_tts_server.h - Minimal HTTP server for Qwen3-TTS */
#ifndef QWEN_TTS_SERVER_H
#define QWEN_TTS_SERVER_H

#include "qwen_tts.h"

int qwen_tts_serve(qwen_tts_ctx_t *ctx, int port);

int qwen_tts_serve_ex(qwen_tts_ctx_t *ctx, int port, int n_workers);

int qwen_tts_serve_batched(qwen_tts_ctx_t *ctx, int port, int max_batch);

void qwen_tts_server_set_limits(int max_queue, int queue_timeout_ms);
void qwen_tts_server_set_max_request_ms(int ms);
void qwen_tts_server_set_max_text_chars(int chars);

#endif
