/*
 * qwen_tts_kernels.h - Kernel function declarations
 */

#ifndef QWEN_TTS_KERNELS_H
#define QWEN_TTS_KERNELS_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ========================================================================
 * Threading
 * ======================================================================== */

void qwen_set_threads(int n);
int qwen_get_threads(void);
int qwen_get_num_cpus(void);
void qwen_init_threads(void);

/* ========================================================================
 * Norm functions
 * ======================================================================== */

/* RMSNorm: out = x / sqrt(mean(x^2) + eps) * weight */
void qwen_rms_norm(float *out, const float *x, const float *weight,
                   int seq, int dim, float eps);

/* RMSNorm per-head */
void qwen_rms_norm_per_head(float *x, const float *weight,
                            int seq, int n_heads, int head_dim, float eps);

/* ========================================================================
 * Linear / MatVec
 * ======================================================================== */

/* bf16 matvec: y[rows] = W[rows,cols] @ x[cols]  (W is bf16, x/y are f32)
 * NEON-optimized + multi-threaded via dispatch_apply on macOS. */
void qwen_matvec_bf16(float *y, const uint16_t *W, const float *x, int rows, int cols);

/* Unified QKV matvec: single dispatch for Q, K, V (avoids 3 barriers) */
void qwen_matvec_bf16_qkv(float *q, float *k, float *v,
                           const uint16_t *Wq, const uint16_t *Wk, const uint16_t *Wv,
                           const float *x, int in_dim, int q_dim, int kv_dim);

/* Matrix-vector: y = W @ x (W is bf16) - batched over seq */
void qwen_linear_nobias_bf16(float *y, const float *x,
                             const uint16_t *W, int seq, int in_dim, int out_dim);

/* Generic linear */
void qwen_linear(float *y, const float *x, const float *W, const float *bias,
                 int seq, int in_dim, int out_dim);

/* ========================================================================
 * Attention
 * ======================================================================== */

/* Causal GQA attention (f32 KV cache) */
void qwen_causal_attention(float *out, const float *Q, const float *K, const float *V,
                           int seq_q, int seq_k, int n_heads, int n_kv_heads,
                           int head_dim, float scale, int q_offset);

/* Causal GQA attention with bf16 KV cache (K/V stored as uint16_t bf16) */
void qwen_causal_attention_bf16kv(float *out, const float *Q,
                                  const uint16_t *K_bf16, const uint16_t *V_bf16,
                                  int seq_q, int seq_k, int n_heads, int n_kv_heads,
                                  int head_dim, float scale, int q_offset);

/* ========================================================================
 * RoPE - INTERLEAVED STYLE
 * ======================================================================== */

/* Compute RoPE cos/sin cache for interleaved RoPE */
void qwen_compute_rope_interleaved(float *cos_out, float *sin_out, const int *positions,
                                   int seq, int head_dim, float theta);

/* Apply interleaved RoPE to x[seq, n_heads * head_dim] */
void qwen_apply_rope_interleaved(float *x, const float *cos_vals, const float *sin_vals,
                                 int seq, int n_heads, int head_dim);

/* ========================================================================
 * Element-wise ops
 * ======================================================================== */

/* SiLU: x = x / (1 + exp(-x)) */
void qwen_silu(float *x, int n);

/* Add: y += x */
void qwen_add_inplace(float *y, const float *x, int n);

/* Mul: y *= x */
void qwen_mul_inplace(float *y, const float *x, int n);

/* Scale: y *= s */
void qwen_vec_scale_inplace(float *y, float s, int n);

/* bf16 rounding */
void qwen_round_bf16(float *x, int n);

/* ========================================================================
 * Argmax / Sampling
 * ======================================================================== */

int qwen_argmax_matvec_bf16(const float *x, const uint16_t *W_bf16, int in_dim, int out_dim);

#ifdef __cplusplus
}
#endif

#endif /* QWEN_TTS_KERNELS_H */
