/* qwen_tts_cuda.h — NVIDIA CUDA backend (G3), C-callable surface. */
#ifndef QWEN_TTS_CUDA_H
#define QWEN_TTS_CUDA_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

int   qwen_cuda_available(void);
void *qwen_cuda_init(void);
void  qwen_cuda_free(void *ctx);

void  qwen_cuda_matvec_bf16(void *ctx, float *y,
                            const uint16_t *W, const float *x,
                            int rows, int cols);

void  qwen_cuda_matmat_bf16(void *ctx, float *Y,
                            const uint16_t *W, const float *X,
                            int rows, int cols, int B);

void  qwen_cuda_rms_norm(float *out, const float *x, const float *w, int dim, float eps);
void  qwen_cuda_swiglu(float *out, const float *gate_up, int n);
void  qwen_cuda_silu(float *out, const float *x, int n);
void  qwen_cuda_add(float *out, const float *a, const float *b, int n);
void  qwen_cuda_mul(float *out, const float *a, const float *b, int n);
void  qwen_cuda_scale(float *out, const float *a, float s, int n);
void  qwen_cuda_rope(float *x, const float *cosv, const float *sinv, int n_heads, int head_dim);
void  qwen_cuda_snake(float *data, const float *la, const float *lb, int channels, int length);
void  qwen_cuda_attention(float *O, const float *Q, const float *K, const float *V,
                          int seq_q, int seq_k, int n_heads, int n_kv, int head_dim, float scale, int q_offset);
void  qwen_cuda_conv1d(float *out, const float *in, const float *weight, const float *bias,
                       int in_ch, int out_ch, int length, int ksz, int dilation);
void  qwen_cuda_conv_transpose1d(float *out, const float *in, const float *weight, const float *bias,
                                 int in_ch, int out_ch, int in_len, int out_len, int ksz, int stride);

extern int g_cuda_decoder_on;
int qwen_cuda_sd_sgemm(int transA, int transB, int M, int N, int K,
                        float alpha, const float *A, int lda, const float *B, int ldb,
                        float beta, float *C, int ldc);

struct qwen_tts_ctx;
void *qwen_cuda_talker_init(struct qwen_tts_ctx *ctx);
void  qwen_cuda_talker_step(void *state, const float *embed, float *hidden_out, int pos);
void  qwen_cuda_talker_free(void *state);

#ifdef __cplusplus
}
#endif

#endif
