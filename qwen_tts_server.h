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

/* Limiti di ammissione del server batchato. Da chiamare PRIMA di qwen_tts_serve_batched.
 *   max_queue         tetto della coda; -1 = automatico (2x gli slot), 0 = illimitata
 *                     (il comportamento fino al 2026-08-20, in cui la quarta richiesta
 *                     aspettava all'infinito senza mai ricevere un errore)
 *   queue_timeout_ms  scadenza di attesa in coda; 0 = nessuna. Oltre la scadenza la
 *                     richiesta riceve 503: consegnare audio in ritardo e' peggio che
 *                     dire di no, perche' nel frattempo occupa uno slot.
 * Il rifiuto e' 503, non 429: 503 = IL SERVER e' senza capacita' (RFC 9110), 429 = QUESTO
 * CLIENT ha superato una quota (RFC 6585) — e quote per cliente non ne abbiamo. */
void qwen_tts_server_set_limits(int max_queue, int queue_timeout_ms);

#endif /* QWEN_TTS_SERVER_H */
