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

/* INT8 matvec: y[rows] = (W_int8[rows,cols] * scale[rows]) @ x[cols]
 * Per-row absmax dequantization. NEON-optimized + multi-threaded. */
void qwen_matvec_int8(float *y, const int8_t *W, const float *scale,
                      const float *x, int rows, int cols);

/* Unified QKV matvec (INT8 variant) */
void qwen_matvec_int8_qkv(float *q, float *k, float *v,
                           const int8_t *Wq, const float *sq,
                           const int8_t *Wk, const float *sk,
                           const int8_t *Wv, const float *sv,
                           const float *x, int in_dim, int q_dim, int kv_dim);

/* INT8 fused argmax+matvec (returns argmax of W @ x without materializing logits) */
int qwen_argmax_matvec_int8(const float *x, const int8_t *W, const float *scale,
                            int in_dim, int out_dim);

/* Quantize bf16 weight matrix to int8 with per-row absmax scaling */
void qwen_quantize_bf16_to_int8(const uint16_t *src_bf16, int rows, int cols,
                                 int8_t *dst_int8, float *dst_scale);

/* Q4_0 block: 32 weights packed into 18 bytes (16 nibble-pairs + fp32 scale) */
#define Q4_0_BLOCK_SIZE 32
typedef struct {
    float scale;           /* per-block scale factor */
    uint8_t qs[16];        /* 32 nibbles: low 4 bits = even idx, high 4 bits = odd idx */
} q4_0_block_t;            /* 20 bytes per 32 weights */

/* Quantize bf16 weight matrix to Q4_0 blocks.
 * cols must be a multiple of 32. Returns number of blocks per row = cols/32.
 * dst must have rows * (cols/32) blocks pre-allocated. */
void qwen_quantize_bf16_to_q4_0(const uint16_t *src_bf16, int rows, int cols,
                                 q4_0_block_t *dst);

/* Q4_0 matvec: y[rows] = dequant(W_q4[rows, cols/32 blocks]) @ x[cols]
 * NEON-optimized + multi-threaded. */
void qwen_matvec_q4_0(float *y, const q4_0_block_t *W, const float *x,
                       int rows, int cols);

/* Unified QKV matvec (Q4_0 variant) */
void qwen_matvec_q4_0_qkv(float *q, float *k, float *v,
                            const q4_0_block_t *Wq, const q4_0_block_t *Wk,
                            const q4_0_block_t *Wv,
                            const float *x, int in_dim, int q_dim, int kv_dim);

/* ========================================================================
 * Attention
 * ======================================================================== */

/* Causal GQA attention (f32 KV cache) */
void qwen_causal_attention(float *out, const float *Q, const float *K, const float *V,
                           int seq_q, int seq_k, int n_heads, int n_kv_heads,
                           int head_dim, float scale, int q_offset);

/* Causal GQA attention with sliding window (f32 KV, window=0 means no window) */
void qwen_causal_attention_windowed(float *out, const float *Q, const float *K, const float *V,
                                     int seq_q, int seq_k, int n_heads, int n_kv_heads,
                                     int head_dim, float scale, int q_offset, int window);

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

/* Snake activation: x += (1/exp(beta)) * sin²(exp(alpha) * x)
 * Applied per-channel to channel-first data [channels, length].
 * log_alpha/log_beta are per-channel params in LOG SPACE. */
void qwen_snake_activation(float *data, int channels, int length,
                            const float *log_alpha, const float *log_beta);

/* ========================================================================
 * Argmax / Sampling
 * ======================================================================== */

int qwen_argmax_matvec_bf16(const float *x, const uint16_t *W_bf16, int in_dim, int out_dim);

#ifdef __cplusplus
}
#endif

#endif /* QWEN_TTS_KERNELS_H */
