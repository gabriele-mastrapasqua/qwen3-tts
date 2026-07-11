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

/* Precision-aware batched projection (B2): dispatches q4 (Wq) > int8 (Wi+Wscale) >
 * bf16 (Wb) by which weight set is non-NULL, using the batched matmat twins (weights
 * read once across B). Uses bb's own scratch/width. Shared by batched Talker + CP. */
void qwen_batch_proj_q(float *dst,
                       const uint16_t *Wb, const int8_t *Wi, const float *Wscale,
                       const q4_0_block_t *Wq,
                       const float *src, int rows, int cols, int srcstride,
                       int B, int force_matvec, float *Xt, float *Yt);

/* Single-stream Code Predictor (defined in qwen_tts_code_predictor.c) — declared here
 * so the batched self-test can use it as the reference. */
int qwen_cp_predict(qwen_tts_ctx_t *ctx, float *talker_hidden, int code0, int *out_codes);

/* One batched Code Predictor: for B frames, talker_hidden[B*hidden] + code0[B] ->
 * out_codes[B*15]. Reuses the CP layer math, batching the matvecs. Returns 0 ok,
 * -2 if non-bf16. (kv reset internally; B sequences in lockstep.)
 * `active` (may be NULL = all active): slot compaction (rental-prep, audit MED-5) —
 * inactive slots skip ALL per-slot vector work (mtp projection, lm_head argmax,
 * rope/attn/norm/swiglu); their out_codes are zeroed. The batched matmats still run
 * full-B width (near-free on bandwidth-bound ARM; the B_eff-gather deep cut is the
 * x86 follow-up, see plan_v4). */
int qwen_batch_cp_predict(qwen_tts_ctx_t *ctx, qwen_batch_t *bb,
                          const float *talker_hidden, const int *code0, int *out_codes,
                          const int *active);

/* Allocate batched buffers + B KV caches from ctx config. kv_max = max frames per
 * chunk. Returns NULL on OOM or if the model isn't bf16 (v1 limitation). */
qwen_batch_t *qwen_batch_alloc(qwen_tts_ctx_t *ctx, int B, int kv_max);
void qwen_batch_free(qwen_batch_t *bb);

/* One batched Talker step: embeds[B*h] -> hidden_out[B*h], advancing all B KV
 * caches by one position. Returns 0 ok, -1 error, -2 unsupported (non-bf16). */
int qwen_batch_talker_step(qwen_tts_ctx_t *ctx, qwen_batch_t *bb,
                           const float *embeds, float *hidden_out);

/* Ragged variant: each sequence at its OWN position pos_arr[b] (so chunks whose
 * prompts prefilled to different lengths can generate together). active[b]=0 skips
 * a finished sequence (ragged EOS). The caller advances pos_arr[b] for active
 * sequences after each step. NULL pos_arr == the lockstep call above. */
int qwen_batch_talker_step_ragged(qwen_tts_ctx_t *ctx, qwen_batch_t *bb,
                                  const float *embeds, const int *pos_arr,
                                  const uint8_t *active, float *hidden_out);

/* Correctness self-test: runs K steps of B identical sequences through the batched
 * step and asserts each column matches the single-stream qwen_talker_step (within
 * fp tolerance). Prints a report; returns 0 on pass. (`./qwen_tts --batch-test`) */
int qwen_batch_self_test(qwen_tts_ctx_t *ctx);

/* End-to-end batched-compute throughput bench (Talker+CP, batched vs single). */
int qwen_batch_bench(qwen_tts_ctx_t *ctx);

#endif
