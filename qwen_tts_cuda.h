/*
 * qwen_tts_cuda.h — NVIDIA CUDA backend (G3), C-callable surface.
 *
 * Implemented in qwen_tts_cuda.c, built only by `make cuda` (defines
 * QWEN_HAVE_CUDA, links -lcublas -lcudart). When the define is absent the TU
 * compiles to no-op stubs (available()→0), so the file is safe to build on M1.
 *
 * v1 = cuBLAS-first (no nvcc): matmat/matvec via cublasSgemm on device f32.
 * Weights are bf16→f32-converted on upload for now; the resident-bf16 +
 * cublasGemmEx (sm_80+) and custom decode matvec + CUDA Graphs are G3b
 * (plan_v4 §E4.ter) — validated on the DGX/5090, not here.
 */

#ifndef QWEN_TTS_CUDA_H
#define QWEN_TTS_CUDA_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

int   qwen_cuda_available(void);
void *qwen_cuda_init(void);
void  qwen_cuda_free(void *ctx);

/* y[rows] = W[rows,cols] @ x[cols]   (W bf16, x/y f32). */
void  qwen_cuda_matvec_bf16(void *ctx, float *y,
                            const uint16_t *W, const float *x,
                            int rows, int cols);

/* Y[rows,B] = W[rows,cols] @ X[cols,B]  (row-major f32; B<=64). */
void  qwen_cuda_matmat_bf16(void *ctx, float *Y,
                            const uint16_t *W, const float *X,
                            int rows, int cols, int B);

/* Custom CUDA compute kernels (qwen_tts_cuda_kernels.cu, nvcc). Host-array API
 * mirroring the Metal ops; built only by `make cuda`. cuBLAS covers the GEMM;
 * these cover norm/rope/act/elementwise/snake/attention/conv. */
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

/* Speech-decoder cuBLAS sgemm (M3): drop-in for the decoder's RowMajor cblas_sgemm — its
 * convs are big matmuls (compute-bound, the 40x cuBLAS regime). g_cuda_decoder_on gates it. */
extern int g_cuda_decoder_on;
void qwen_cuda_sd_sgemm(int transA, int transB, int M, int N, int K,
                        float alpha, const float *A, int lda, const float *B, int ldb,
                        float beta, float *C, int ldc);

/* GPU-RESIDENT fused Talker step (qwen_tts_cuda_talker.cu, nvcc). Weights+KV resident;
 * one sync/step. init reads the loaded model; step runs the whole Talker step on-device. */
struct qwen_tts_ctx;
void *qwen_cuda_talker_init(struct qwen_tts_ctx *ctx);
void  qwen_cuda_talker_step(void *state, const float *embed, float *hidden_out, int pos);
void  qwen_cuda_talker_free(void *state);

#ifdef __cplusplus
}
#endif

#endif /* QWEN_TTS_CUDA_H */
