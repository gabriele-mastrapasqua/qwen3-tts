/* qwen_tts_server.h - Minimal HTTP server for Qwen3-TTS */
#ifndef QWEN_TTS_SERVER_H
#define QWEN_TTS_SERVER_H

#include "qwen_tts.h"

/* Start HTTP server. Blocks until killed. Returns 0 on clean shutdown, -1 on error. */
int qwen_tts_serve(qwen_tts_ctx_t *ctx, int port);

/* Like qwen_tts_serve, but with n_workers concurrent synthesis workers.
 *   n_workers <= 1 : single-threaded inline accept loop (original behavior).
 *   n_workers >= 2 : acceptor thread + worker pool; worker 0 uses `ctx`, the
 *                    rest are independent clones (qwen_tts_clone_for_worker).
 * On thread-pool backends that are NOT reentrant (pthread/Win32) synthesis is
 * serialized with an internal lock — correct but no intra-op overlap; full
 * parallelism only on the GCD backend. */
int qwen_tts_serve_ex(qwen_tts_ctx_t *ctx, int port, int n_workers);

/* vLLM-style request-batching server (opt-in --batch-size N, N>=2). A single
 * scheduler thread owns ctx and steps up to N concurrent users' requests together
 * through Talker+CP weight-stationary (qwen_tts_generate_batch_multi); a reader
 * pool parses HTTP into jobs. Throughput lever for many concurrent users —
 * distinct from --workers (N independent single-stream synths). Preset voices +
 * sampling params per request; instruct/voice_design/stream fall back to single
 * jobs on the scheduler. */
int qwen_tts_serve_batched(qwen_tts_ctx_t *ctx, int port, int max_batch);

/* Admission limits for the batched server. Call this BEFORE qwen_tts_serve_batched.
 *   max_queue         queue cap; -1 = automatic (2x the slots), 0 = unbounded (the old
 *                     behaviour, in which the fourth request waited forever without ever
 *                     receiving an error)
 *   queue_timeout_ms  wait deadline in the queue; 0 = none. Past the deadline the request
 *                     gets a 503: delivering late audio is worse than saying no, because
 *                     meanwhile it occupies a slot.
 * The refusal is 503, not 429: 503 = THE SERVER has no capacity (RFC 9110), 429 = THIS
 * CLIENT exceeded a quota (RFC 6585) — and there are no per-client quotas here. */
void qwen_tts_server_set_limits(int max_queue, int queue_timeout_ms);
/* Wall-clock ceiling on time IN SERVICE, per request, in ms. 0 disables.
 * The token cap is not a safety limit: at 12.5 Hz it corresponds to roughly eleven
 * minutes of audio, so one caller can hold a slot that long. Measured from admission,
 * because queue time is bounded separately. QWEN_MAX_REQUEST_S (seconds) overrides. */
void qwen_tts_server_set_max_request_ms(int ms);
/* Maximum accepted text length. 0 = derive it from the generation cap, which is the
 * relationship that matters: a caller must not be able to submit work the server
 * cannot finish. QWEN_MAX_TEXT_CHARS overrides. */
void qwen_tts_server_set_max_text_chars(int chars);

#endif /* QWEN_TTS_SERVER_H */
