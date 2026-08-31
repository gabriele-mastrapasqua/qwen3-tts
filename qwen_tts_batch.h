/* qwen_tts_batch.h - OPT-IN batched Talker inference (feat/batching). */
#ifndef QWEN_TTS_BATCH_H
#define QWEN_TTS_BATCH_H
#include "qwen_tts.h"

typedef struct {
    int B;
    int B_eff;
    int act_idx[64];
    int h, q_dim, kv_dim, inter, num_layers, kv_max;
    int kv_len;
    float *x, *x_norm, *q, *k, *v, *attn_out, *proj_out, *gate, *swiglu_tmp;
    float *Xt, *Yt;
    uint16_t *kv_k, *kv_v;
    int force_matvec;

    int cp_h, cp_q_dim, cp_kv_dim, cp_inter, cp_num_layers, cp_kv_max;
    float *cp_x, *cp_x_norm, *cp_q, *cp_k, *cp_v, *cp_attn, *cp_proj, *cp_gate, *cp_swiglu_tmp;
    float *cp_Xt, *cp_Yt;
    uint16_t *cp_kv_k, *cp_kv_v;
} qwen_batch_t;

void qwen_batch_proj(float *dst, const uint16_t *W, const float *src,
                     int rows, int cols, int srcstride, int B, const int *idx,
                     int force_matvec, float *Xt, float *Yt);

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

int qwen_cp_predict(qwen_tts_ctx_t *ctx, float *talker_hidden, int code0, int *out_codes);

int qwen_batch_cp_predict(qwen_tts_ctx_t *ctx, qwen_batch_t *bb,
                          const float *talker_hidden, const int *code0, int *out_codes,
                          const uint8_t *active);

int qwen_batch_solo_disabled(void);

int qwen_batch_beff_disabled(void);

void qwen_batch_pack_active(qwen_batch_t *bb, const uint8_t *active);

qwen_batch_t *qwen_batch_alloc(qwen_tts_ctx_t *ctx, int B, int kv_max);
void qwen_batch_free(qwen_batch_t *bb);

int qwen_batch_talker_step(qwen_tts_ctx_t *ctx, qwen_batch_t *bb,
                           const float *embeds, float *hidden_out);

int qwen_batch_talker_step_ragged(qwen_tts_ctx_t *ctx, qwen_batch_t *bb,
                                  const float *embeds, const int *pos_arr,
                                  const uint8_t *active, float *hidden_out);

int qwen_batch_self_test(qwen_tts_ctx_t *ctx);

int qwen_batch_bench(qwen_tts_ctx_t *ctx);

#endif
