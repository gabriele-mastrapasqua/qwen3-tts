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

int qwen_metal_upload_f32(qwen_metal_ctx_t *ctx, const float *data, int count) {
    (void)ctx; (void)data; (void)count; return -1;
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

void qwen_metal_begin(qwen_metal_ctx_t *ctx) { (void)ctx; }
void qwen_metal_encode_matvec(qwen_metal_ctx_t *ctx, int weight_handle,
                               int y_offset, int x_offset,
                               int rows, int cols) {
    (void)ctx; (void)weight_handle; (void)y_offset; (void)x_offset; (void)rows; (void)cols;
}
void qwen_metal_sync(qwen_metal_ctx_t *ctx) { (void)ctx; }
float *qwen_metal_get_x(qwen_metal_ctx_t *ctx) { (void)ctx; return NULL; }
float *qwen_metal_get_y(qwen_metal_ctx_t *ctx) { (void)ctx; return NULL; }
void qwen_metal_ensure_workspace(qwen_metal_ctx_t *ctx, int x_bytes, int y_bytes) {
    (void)ctx; (void)x_bytes; (void)y_bytes;
}

/* Full GPU step stubs */
int qwen_metal_alloc_kv_cache(qwen_metal_ctx_t *ctx, int kv_id,
                               int n_layers, int kv_max, int kv_dim) {
    (void)ctx; (void)kv_id; (void)n_layers; (void)kv_max; (void)kv_dim; return -1;
}

void qwen_metal_sync_kv_cache(qwen_metal_ctx_t *ctx, int kv_id,
                               const uint16_t *k, const uint16_t *v,
                               int n_layers, int kv_max, int kv_dim, int kv_len) {
    (void)ctx; (void)kv_id; (void)k; (void)v; (void)n_layers; (void)kv_max; (void)kv_dim; (void)kv_len;
}

int qwen_metal_alloc_work_buffers(qwen_metal_ctx_t *ctx,
                                   int max_hidden, int max_q_dim,
                                   int max_kv_dim, int max_inter) {
    (void)ctx; (void)max_hidden; (void)max_q_dim; (void)max_kv_dim; (void)max_inter; return -1;
}

int qwen_metal_transformer_step(qwen_metal_ctx_t *ctx,
                                 const qwen_metal_step_config_t *cfg,
                                 const qwen_metal_layer_config_t *layers,
                                 float *x_inout,
                                 float *normed_out,
                                 int pos,
                                 int kv_id) {
    (void)ctx; (void)cfg; (void)layers; (void)x_inout; (void)normed_out; (void)pos; (void)kv_id;
    return -1;
}

int qwen_metal_has_full_step(qwen_metal_ctx_t *ctx) { (void)ctx; return 0; }
