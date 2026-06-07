/* qwen_tts_batch.h - OPT-IN batched Talker inference (feat/batching).
 *
 * Strictly ADDITIVE: this is a NEW path used only when batching is requested. The
 * single-stream qwen_talker_step() is untouched. Reuses the existing per-vector
 * kernels (rmsnorm/rope/attention/swiglu) looped over B, and batches ONLY the
 * matvecs via qwen_matmat_bf16 (the bandwidth-amortizing primitive).
 *
 * v1 scope: bf16 weights only, B sequences in LOCKSTEP (same position/length, no
 * ragged EOS yet). int8/int4 twins + ragged batch + CP batching come next.
 */
#ifndef QWEN_TTS_BATCH_H
#define QWEN_TTS_BATCH_H
#include "qwen_tts.h"

typedef struct {
    int B;                                  /* batch width (chunks in flight) */
    int h, q_dim, kv_dim, inter, num_layers, kv_max;
    int kv_len;                             /* shared lockstep position */
    /* B-wide activation buffers (each sequence contiguous: [B][dim]) */
    float *x, *x_norm, *q, *k, *v, *attn_out, *proj_out, *gate, *swiglu_tmp;
    /* transpose scratch for the matmat ([dim][B]) */
    float *Xt, *Yt;
    /* per-sequence KV caches: [B][num_layers][kv_max][kv_dim] bf16 */
    uint16_t *kv_k, *kv_v;
    int force_matvec;   /* diagnostic: do projections as B matvecs (bit-matches single-stream)
                           instead of one batched matmat. Default 0 (use the batched matmat). */
} qwen_batch_t;

/* Allocate batched buffers + B KV caches from ctx config. kv_max = max frames per
 * chunk. Returns NULL on OOM or if the model isn't bf16 (v1 limitation). */
qwen_batch_t *qwen_batch_alloc(qwen_tts_ctx_t *ctx, int B, int kv_max);
void qwen_batch_free(qwen_batch_t *bb);

/* One batched Talker step: embeds[B*h] -> hidden_out[B*h], advancing all B KV
 * caches by one position. Returns 0 ok, -1 error, -2 unsupported (non-bf16). */
int qwen_batch_talker_step(qwen_tts_ctx_t *ctx, qwen_batch_t *bb,
                           const float *embeds, float *hidden_out);

/* Correctness self-test: runs K steps of B identical sequences through the batched
 * step and asserts each column matches the single-stream qwen_talker_step (within
 * fp tolerance). Prints a report; returns 0 on pass. (`./qwen_tts --batch-test`) */
int qwen_batch_self_test(qwen_tts_ctx_t *ctx);

#endif
