/*
 * qwen_tts_metal.h - Metal GPU acceleration for Qwen3-TTS
 *
 * Optional GPU backend for Apple Silicon Macs.
 * Supports both per-dispatch matvec and full GPU transformer step.
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

/* ── Init / Free ──────────────────────────────────────────────────────── */

qwen_metal_ctx_t *qwen_metal_init(void);
void qwen_metal_free(qwen_metal_ctx_t *ctx);
int qwen_metal_is_active(qwen_metal_ctx_t *ctx);

/* ── Weight Upload ────────────────────────────────────────────────────── */

/* Upload bf16 weight matrix to GPU. Returns handle, or -1 on failure. */
int qwen_metal_upload_weight(qwen_metal_ctx_t *ctx,
                             const uint16_t *data, int rows, int cols);

/* Upload f32 data to GPU. Returns handle, or -1 on failure. */
int qwen_metal_upload_f32(qwen_metal_ctx_t *ctx, const float *data, int count);

/* ── Per-Dispatch Matvec (legacy) ─────────────────────────────────────── */

void qwen_metal_matvec_bf16(qwen_metal_ctx_t *ctx, int weight_handle,
                            float *y, const float *x, int rows, int cols);

void qwen_metal_matvec_bf16_qkv(qwen_metal_ctx_t *ctx,
                                 int wq_handle, int wk_handle, int wv_handle,
                                 float *q, float *k, float *v,
                                 const float *x, int in_dim,
                                 int q_dim, int kv_dim);

/* ── Batched Dispatch API ─────────────────────────────────────────────── */

void qwen_metal_begin(qwen_metal_ctx_t *ctx);
void qwen_metal_encode_matvec(qwen_metal_ctx_t *ctx, int weight_handle,
                               int y_offset, int x_offset,
                               int rows, int cols);
void qwen_metal_sync(qwen_metal_ctx_t *ctx);
float *qwen_metal_get_x(qwen_metal_ctx_t *ctx);
float *qwen_metal_get_y(qwen_metal_ctx_t *ctx);
void qwen_metal_ensure_workspace(qwen_metal_ctx_t *ctx, int x_bytes, int y_bytes);

/* ── Full GPU Transformer Step ────────────────────────────────────────── */

/* Per-layer GPU handle configuration */
typedef struct {
    int gpu_wq, gpu_wk, gpu_wv, gpu_wo;
    int gpu_gate_up_fused, gpu_down;
    int gpu_input_norm, gpu_post_attn_norm;
    int gpu_q_norm, gpu_k_norm;
} qwen_metal_layer_config_t;

/* Transformer step configuration */
typedef struct {
    int n_layers;
    int hidden_size;
    int num_heads;
    int num_kv_heads;
    int head_dim;
    int intermediate_size;
    float rms_norm_eps;
    int gpu_rope_cos;     /* handle from qwen_metal_upload_f32 */
    int gpu_rope_sin;
    int gpu_final_norm;   /* handle for final RMSNorm, or -1 to skip */
} qwen_metal_step_config_t;

/* Allocate GPU KV cache. kv_id: 0=talker, 1=CP. */
int qwen_metal_alloc_kv_cache(qwen_metal_ctx_t *ctx, int kv_id,
                               int n_layers, int kv_max, int kv_dim);

/* Copy CPU KV cache to GPU after prefill. */
void qwen_metal_sync_kv_cache(qwen_metal_ctx_t *ctx, int kv_id,
                               const uint16_t *k, const uint16_t *v,
                               int n_layers, int kv_max, int kv_dim, int kv_len);

/* Allocate working buffers (call once with max dims across transformers). */
int qwen_metal_alloc_work_buffers(qwen_metal_ctx_t *ctx,
                                   int max_hidden, int max_q_dim,
                                   int max_kv_dim, int max_inter);

/* Full GPU transformer step: all layers in one command buffer.
 * x_inout: [hidden] input/output (updated in-place with final residual).
 * normed_out: [hidden] final RMSNorm output, or NULL to skip.
 * Returns 0 on success. */
int qwen_metal_transformer_step(qwen_metal_ctx_t *ctx,
                                 const qwen_metal_step_config_t *cfg,
                                 const qwen_metal_layer_config_t *layers,
                                 float *x_inout,
                                 float *normed_out,
                                 int pos,
                                 int kv_id);

/* Check if full GPU step is available */
int qwen_metal_has_full_step(qwen_metal_ctx_t *ctx);

/* Global Metal context */
extern qwen_metal_ctx_t *g_metal_ctx;

#ifdef __cplusplus
}
#endif

#endif /* QWEN_TTS_METAL_H */
