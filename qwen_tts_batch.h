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

    /* ---- Code Predictor batched buffers (B frames in lockstep) ---- */
    int cp_h, cp_q_dim, cp_kv_dim, cp_inter, cp_num_layers, cp_kv_max;
    float *cp_x, *cp_x_norm, *cp_q, *cp_k, *cp_v, *cp_attn, *cp_proj, *cp_gate, *cp_swiglu_tmp;
    float *cp_Xt, *cp_Yt;
    uint16_t *cp_kv_k, *cp_kv_v;   /* [B][cp_num_layers][cp_kv_max][cp_kv_dim] */
} qwen_batch_t;

/* Batched projection dst[B][rows] = W @ src[B][cols] (src row b at b*srcstride).
 * Shared by the batched Talker and Code Predictor. force_matvec=1 -> B matvecs
 * (bit-matches single-stream); else one batched matmat. Xt/Yt = scratch. */
void qwen_batch_proj(float *dst, const uint16_t *W, const float *src,
                     int rows, int cols, int srcstride, int B, int force_matvec,
                     float *Xt, float *Yt);

/* Single-stream Code Predictor (defined in qwen_tts_code_predictor.c) — declared here
 * so the batched self-test can use it as the reference. */
int qwen_cp_predict(qwen_tts_ctx_t *ctx, float *talker_hidden, int code0, int *out_codes);

/* One batched Code Predictor: for B frames, talker_hidden[B*hidden] + code0[B] ->
 * out_codes[B*15]. Reuses the CP layer math, batching the matvecs. Returns 0 ok,
 * -2 if non-bf16. (kv reset internally; B sequences in lockstep.) */
int qwen_batch_cp_predict(qwen_tts_ctx_t *ctx, qwen_batch_t *bb,
                          const float *talker_hidden, const int *code0, int *out_codes);

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

/* End-to-end batched-compute throughput bench (Talker+CP, batched vs single). */
int qwen_batch_bench(qwen_tts_ctx_t *ctx);

#endif
