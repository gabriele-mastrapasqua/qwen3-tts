/* qwen_tts_metal.h — Apple Metal backend (G2), C-callable surface. */
#ifndef QWEN_TTS_METAL_H
#define QWEN_TTS_METAL_H

#include <stdint.h>
#include "qwen_tts_kernels.h"

#ifdef __cplusplus
extern "C" {
#endif

int   qwen_metal_available(void);
void *qwen_metal_init(void);
void  qwen_metal_free(void *ctx);

void  qwen_metal_matvec_bf16(void *ctx, float *y, const uint16_t *W,
                             const float *x, int rows, int cols);
void  qwen_metal_matmat_bf16(void *ctx, float *Y, const uint16_t *W,
                             const float *X, int rows, int cols, int B);
void  qwen_metal_matvec_int8(void *ctx, float *y, const int8_t *W,
                             const float *scale, const float *x, int rows, int cols);
void  qwen_metal_matvec_q4_0(void *ctx, float *y, const q4_0_block_t *W,
                             const float *x, int rows, int cols);

void  qwen_metal_rms_norm(void *ctx, float *out, const float *x,
                          const float *weight, int dim, float eps);
void  qwen_metal_swiglu(void *ctx, float *out, const float *gate_up, int n);
void  qwen_metal_silu(void *ctx, float *out, const float *x, int n);
void  qwen_metal_add(void *ctx, float *out, const float *a, const float *b, int n);
void  qwen_metal_mul(void *ctx, float *out, const float *a, const float *b, int n);
void  qwen_metal_scale(void *ctx, float *out, const float *a, float s, int n);
void  qwen_metal_rope(void *ctx, float *x, const float *cosv, const float *sinv,
                      int n_heads, int head_dim);

void  qwen_metal_snake(void *ctx, float *data, const float *log_alpha,
                       const float *log_beta, int channels, int length);
void  qwen_metal_attention(void *ctx, float *O, const float *Q, const float *K, const float *V,
                           int seq_q, int seq_k, int n_heads, int n_kv, int head_dim,
                           float scale, int q_offset);
void  qwen_metal_matmat_f32(void *ctx, float *Y, const float *W, const float *X,
                            int rows, int cols, int B);
void  qwen_metal_conv1d(void *ctx, float *out, const float *in, const float *weight,
                        const float *bias, int in_ch, int out_ch, int length, int ksz, int dilation);
void  qwen_metal_conv_transpose1d(void *ctx, float *out, const float *in, const float *weight,
                                  const float *bias, int in_ch, int out_ch, int in_len, int out_len,
                                  int ksz, int stride);

void  qwen_metal_ffn_swiglu(void *ctx, float *out, const float *x, const float *norm_w,
                            const uint16_t *Wgu, const uint16_t *Wd,
                            int H, int inter, float eps);
void  qwen_metal_ffn_swiglu_batched(void *ctx, float *out, const float *x, const float *norm_w,
                                    const uint16_t *Wgu, const uint16_t *Wd,
                                    int H, int inter, int B, float eps);

double qwen_metal_matvec_bench_fused(void *ctx, const uint16_t *W, const float *x,
                                     int rows, int cols, int reps);

int   qwen_metal_selftest(void *out);

struct qwen_tts_ctx;
void *qwen_metal_talker_init(void *metal_ctx, struct qwen_tts_ctx *ctx);
void  qwen_metal_talker_step(void *state, const float *embed, float *hidden_out, int pos);
void  qwen_metal_talker_upload_kv(void *state, struct qwen_tts_ctx *ctx, int prefill_len);
void  qwen_metal_talker_free(void *state);
void *qwen_metal_cp_init(void *metal_ctx, struct qwen_tts_ctx *ctx);
void  qwen_metal_cp_step(void *state, float *x, int pos);
void  qwen_metal_cp_free(void *state);
void *qwen_metal_cp_frame_init(void *metal_ctx, struct qwen_tts_ctx *ctx);
void  qwen_metal_cp_frame(void *state, const float *talker_hidden, int code0, int *out_codes);
void  qwen_metal_cp_frame_free(void *state);

void *qwen_metal_talker_batch_init(void *single_state, int B);
void  qwen_metal_talker_batch_step(void *state, const float *embeds, const int *pos_arr, float *hidden_out);
void  qwen_metal_talker_batch_upload_slot(void *state, int b, const uint16_t *kv_k, const uint16_t *kv_v,
                                          int src_kv_max, int prefill_len);
void  qwen_metal_talker_batch_free(void *state);
void *qwen_metal_cp_batch_init(void *single_talker_state, int B);
void  qwen_metal_cp_batch_step(void *state, float *x, const int *pos_arr);
void  qwen_metal_cp_batch_free(void *state);

#ifdef __cplusplus
}
#endif

#endif
