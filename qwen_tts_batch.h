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
    int B;                                  /* batch width ALLOCATED (max chunks in flight) */
    /* Width the projections actually run at this step: (highest active slot + 1).
     * Columns above it belong to inactive slots and their output is discarded, so
     * computing them is work thrown away — on EVERY ISA. Only the SIZE of the loss is
     * machine-dependent (bandwidth/compute ratio), not its existence: the "near-free on
     * bandwidth-bound ARM" claim below holds only where the kernel is fully
     * bandwidth-bound, which is a property of the box and not of the instruction set.
     * Set per step by the batched Talker/CP; always <= B, == B when the last slot is
     * busy. Idle slots BELOW it still cost — closing those holes needs slot compaction
     * (PLAN T5.piano #2b). Measured on M1; to be re-measured on x86-64 (AVX2-only and
     * AVX-512/VNNI) and arm64 Linux. */
    int B_eff;
    /* act_idx[0..B_eff) = the slot index of each column the GEMMs actually compute.
     * With this map the batched path follows the ACTIVE slots instead of the allocated
     * width, so idle slots cost nothing even when they sit in the MIDDLE of the batch —
     * which the high-water mark alone could not fix, since both orchestrators seed every
     * slot up front and drain them in arbitrary order. No KV is moved and the
     * orchestrator is untouched: the packing lives entirely in gather/scatter. */
    int act_idx[64];
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
/* `idx` (may be NULL = identity) maps each computed column to its slot: pass the
 * active-slot map to make the GEMM width follow the load instead of the allocation. */
void qwen_batch_proj(float *dst, const uint16_t *W, const float *src,
                     int rows, int cols, int srcstride, int B, const int *idx,
                     int force_matvec, float *Xt, float *Yt);

/* Precision-aware batched projection (B2): dispatches q4 (Wq) > int8 (Wi+Wscale) >
 * bf16 (Wb) by which weight set is non-NULL, using the batched matmat twins (weights
 * read once across B). Uses bb's own scratch/width. Shared by batched Talker + CP. */
/* One QKV: one LHS pack and one barrier when the fused int8 path applies, otherwise
 * exactly the three qwen_batch_proj_q calls it replaces. QWEN_KAI_QKV_FUSED=0 forces
 * the latter. */
void qwen_batch_proj_qkv(float *dq, float *dk, float *dv,
                         const uint16_t *Wqb, const int8_t *Wqi, const float *Wqs,
                         const q4_0_block_t *Wqq,
                         const uint16_t *Wkb, const int8_t *Wki, const float *Wks,
                         const q4_0_block_t *Wkq,
                         const uint16_t *Wvb, const int8_t *Wvi, const float *Wvs,
                         const q4_0_block_t *Wvq,
                         const float *src, int q_rows, int kv_rows, int cols,
                         int srcstride, int B, const int *idx, int force_matvec,
                         float *Xt, float *Yt);

void qwen_batch_proj_q(float *dst,
                       const uint16_t *Wb, const int8_t *Wi, const float *Wscale,
                       const q4_0_block_t *Wq,
                       const float *src, int rows, int cols, int srcstride,
                       int B, const int *idx, int force_matvec, float *Xt, float *Yt);

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
                          const uint8_t *active);

/* 1 if QWEN_BATCH_NO_SOLO=1: the B_eff==1 shortcut is off and every step stays on the
 * batched path. Shared by the Talker and CP solo fallbacks, so one switch keeps them
 * consistent — one on and one off would be a configuration nobody has tested. */
int qwen_batch_solo_disabled(void);

/* 1 if QWEN_BATCH_NO_BEFF=1: projections stay pinned to the allocated B (the "before"
 * of T5.piano #2), so the win can be A/B'd on any box without a rebuild. */
int qwen_batch_beff_disabled(void);

/* Write bb->act_idx / bb->B_eff so they describe THIS step's active slots.
 *
 * INVARIANT, and it exists because breaking it cost a day (2026-08-20): after any step —
 * batched or the B_eff==1 shortcut — act_idx/B_eff must describe that step. The frame
 * loop reads them straight afterwards for the codec head (qwen_tts.c), and both solo
 * shortcuts used to return BEFORE the packing block, leaving whatever the last batched
 * step had left. When the surviving slot was not in that stale set, its logits column was
 * never computed and the sampler read the PREVIOUS frame's values — for another request.
 * Generation derailed from there: noise, speech and beeps mixed, and double the duration.
 *
 * It only fired on the transition from several active slots to one, which a server
 * reaches constantly (open-loop arrivals put three requests in flight even at a nominal
 * concurrency of 1), and never in a sequential test — without overlap the batched path
 * never runs and act_idx stays the identity. Hence one function, called on every exit
 * path, instead of the same nine lines copied into four places.
 *
 * `active` NULL (bench, self-test: no mask) means full width. */
void qwen_batch_pack_active(qwen_batch_t *bb, const uint8_t *active);

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
