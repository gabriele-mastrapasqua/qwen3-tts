/*
 * qwen_tts_metal.h - Metal GPU acceleration for Qwen3-TTS
 *
 * Optional GPU backend for Apple Silicon Macs.
 * Accelerates bf16 matvec operations in Talker and Code Predictor.
 * CPU remains the default; Metal is opt-in via --gpu flag.
 */

#ifndef QWEN_TTS_METAL_H
#define QWEN_TTS_METAL_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque Metal context (implementation in qwen_tts_metal.m) */
typedef struct qwen_metal_ctx qwen_metal_ctx_t;

/* Initialize Metal: create device, compile shaders, allocate workspace.
 * Returns NULL if Metal is unavailable (e.g., Intel Mac, Linux). */
qwen_metal_ctx_t *qwen_metal_init(void);

/* Upload bf16 weight matrix to GPU buffer. Returns a handle (index).
 * The weight data is copied once at load time.
 * Returns -1 on failure. */
int qwen_metal_upload_weight(qwen_metal_ctx_t *ctx,
                             const uint16_t *data, int rows, int cols);

/* bf16 matvec on GPU: y[rows] = W[rows,cols] @ x[cols]
 * W is identified by the handle from qwen_metal_upload_weight().
 * x and y are CPU-side f32 buffers (copied to/from GPU). */
void qwen_metal_matvec_bf16(qwen_metal_ctx_t *ctx, int weight_handle,
                            float *y, const float *x, int rows, int cols);

/* Unified QKV matvec: single command buffer for Q, K, V projections */
void qwen_metal_matvec_bf16_qkv(qwen_metal_ctx_t *ctx,
                                 int wq_handle, int wk_handle, int wv_handle,
                                 float *q, float *k, float *v,
                                 const float *x, int in_dim,
                                 int q_dim, int kv_dim);

/* Cleanup: free GPU buffers and Metal objects */
void qwen_metal_free(qwen_metal_ctx_t *ctx);

/* Check if Metal context is active */
int qwen_metal_is_active(qwen_metal_ctx_t *ctx);

/* Global Metal context for kernel dispatch.
 * Set by qwen_tts_init_metal(), read by kernel functions. */
extern qwen_metal_ctx_t *g_metal_ctx;

#ifdef __cplusplus
}
#endif

#endif /* QWEN_TTS_METAL_H */
