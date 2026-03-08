/*
 * qwen_tts_metal_stub.c - No-op stubs when Metal GPU is not enabled
 *
 * Compiled instead of qwen_tts_metal.m when ENABLE_METAL is not defined.
 * All functions return failure/no-op so CPU path is always used.
 */

#include "qwen_tts_metal.h"
#include <stddef.h>

/* Global Metal context (always NULL when Metal disabled) */
qwen_metal_ctx_t *g_metal_ctx = NULL;

qwen_metal_ctx_t *qwen_metal_init(void) { return NULL; }
void qwen_metal_free(qwen_metal_ctx_t *ctx) { (void)ctx; }
int qwen_metal_is_active(qwen_metal_ctx_t *ctx) { (void)ctx; return 0; }

int qwen_metal_upload_weight(qwen_metal_ctx_t *ctx,
                             const uint16_t *data, int rows, int cols) {
    (void)ctx; (void)data; (void)rows; (void)cols; return -1;
}

void qwen_metal_matvec_bf16(qwen_metal_ctx_t *ctx, int weight_handle,
                            float *y, const float *x, int rows, int cols) {
    (void)ctx; (void)weight_handle; (void)y; (void)x; (void)rows; (void)cols;
}

void qwen_metal_matvec_bf16_qkv(qwen_metal_ctx_t *ctx,
                                 int wq_handle, int wk_handle, int wv_handle,
                                 float *q, float *k, float *v,
                                 const float *x, int in_dim,
                                 int q_dim, int kv_dim) {
    (void)ctx; (void)wq_handle; (void)wk_handle; (void)wv_handle;
    (void)q; (void)k; (void)v; (void)x; (void)in_dim; (void)q_dim; (void)kv_dim;
}
