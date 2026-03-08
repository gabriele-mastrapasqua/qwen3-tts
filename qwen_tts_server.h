/* qwen_tts_server.h - Minimal HTTP server for Qwen3-TTS */
#ifndef QWEN_TTS_SERVER_H
#define QWEN_TTS_SERVER_H

#include "qwen_tts.h"

/* Start HTTP server. Blocks until killed. Returns 0 on clean shutdown, -1 on error. */
int qwen_tts_serve(qwen_tts_ctx_t *ctx, int port);

#endif /* QWEN_TTS_SERVER_H */
