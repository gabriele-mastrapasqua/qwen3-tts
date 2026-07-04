/*
 * qwen_tts_metal.h — Apple Metal backend (G2), C-callable surface.
 *
 * Implemented in qwen_tts_metal.m (Objective-C, clang -fobjc-arc). Only built
 * by `make metal` (defines QWEN_HAVE_METAL). Plain C linkage so the gcc-compiled
 * engine can call it.
 *
 * ARCHITECTURE (the important part): weights are RESIDENT. A weight pointer is
 * uploaded to a device MTLBuffer ONCE and cached by pointer; every later call
 * reuses it — NO per-call allocation or re-upload (the naive-loop trap). IO
 * (activations/scales) reuse a small pool of persistent shared buffers. This is
 * additive: the CPU path stays the default; Metal only runs when selected.
 *
 * bf16/int8/q4_0 weights are read raw and dequantized IN-SHADER, matching the
 * CPU kernels bit-for-bit (int8 scale=amax/127; q4_0 scale=amax/7, even idx→low
 * nibble/odd→high, val=(nib-8)*scale).
 */

#ifndef QWEN_TTS_METAL_H
#define QWEN_TTS_METAL_H

#include <stdint.h>
#include "qwen_tts_kernels.h"   /* q4_0_block_t layout shared with the CPU path */

#ifdef __cplusplus
extern "C" {
#endif

int   qwen_metal_available(void);
void *qwen_metal_init(void);
void  qwen_metal_free(void *ctx);

/* ---- linear / matmul ---- */
/* y[rows] = W[rows,cols] @ x[cols]   (W bf16, x/y f32) */
void  qwen_metal_matvec_bf16(void *ctx, float *y, const uint16_t *W,
                             const float *x, int rows, int cols);
/* Y[rows,B] = W[rows,cols] @ X[cols,B]  (row-major f32; B<=64) */
void  qwen_metal_matmat_bf16(void *ctx, float *Y, const uint16_t *W,
                             const float *X, int rows, int cols, int B);
/* y[rows] = scale[rows] * (W_int8[rows,cols] @ x[cols]) */
void  qwen_metal_matvec_int8(void *ctx, float *y, const int8_t *W,
                             const float *scale, const float *x, int rows, int cols);
/* y[rows] = dequant(W_q4[rows, cols/32]) @ x[cols] */
void  qwen_metal_matvec_q4_0(void *ctx, float *y, const q4_0_block_t *W,
                             const float *x, int rows, int cols);

/* ---- norm / activation / elementwise / rope ---- */
/* out[dim] = x/sqrt(mean(x^2)+eps) * weight   (single row) */
void  qwen_metal_rms_norm(void *ctx, float *out, const float *x,
                          const float *weight, int dim, float eps);
/* out[n] = silu(gate_up[2i]) * gate_up[2i+1]  (interleaved SwiGLU) */
void  qwen_metal_swiglu(void *ctx, float *out, const float *gate_up, int n);
/* out[n] = x/(1+exp(-x)) */
void  qwen_metal_silu(void *ctx, float *out, const float *x, int n);
void  qwen_metal_add(void *ctx, float *out, const float *a, const float *b, int n);
void  qwen_metal_mul(void *ctx, float *out, const float *a, const float *b, int n);
void  qwen_metal_scale(void *ctx, float *out, const float *a, float s, int n);
/* interleaved RoPE on x[n_heads*head_dim] for a single position (cos/sin over pairs) */
void  qwen_metal_rope(void *ctx, float *x, const float *cosv, const float *sinv,
                      int n_heads, int head_dim);

/* Snake activation (decoder): data[c*L+t] += exp(-log_beta[c])*sin(exp(log_alpha[c])*x)^2 */
void  qwen_metal_snake(void *ctx, float *data, const float *log_alpha,
                       const float *log_beta, int channels, int length);
/* Direct causal GQA attention (matches qwen_causal_attention). */
void  qwen_metal_attention(void *ctx, float *O, const float *Q, const float *K, const float *V,
                           int seq_q, int seq_k, int n_heads, int n_kv, int head_dim,
                           float scale, int q_offset);
/* matmat with f32 weights (prefill GEMM), simdgroup_matrix MMA. */
void  qwen_metal_matmat_f32(void *ctx, float *Y, const float *W, const float *X,
                            int rows, int cols, int B);
/* Causal conv1d / conv-transpose1d, channel-first [ch,length] (decoder ConvNet). */
void  qwen_metal_conv1d(void *ctx, float *out, const float *in, const float *weight,
                        const float *bias, int in_ch, int out_ch, int length, int ksz, int dilation);
void  qwen_metal_conv_transpose1d(void *ctx, float *out, const float *in, const float *weight,
                                  const float *bias, int in_ch, int out_ch, int in_len, int out_len,
                                  int ksz, int stride);

/* FUSED RESIDENT FFN (the heavy block): rms_norm → gate_up matvec → SwiGLU →
 * down matvec → residual, encoded as ONE command buffer with all intermediates
 * kept in device buffers (no per-op CPU sync). gate_up [2*inter,H] interleaved,
 * down [H,inter]; out += x. The resident-decode pattern. */
void  qwen_metal_ffn_swiglu(void *ctx, float *out, const float *x, const float *norm_w,
                            const uint16_t *Wgu, const uint16_t *Wd,
                            int H, int inter, float eps);
/* BATCHED fused FFN (B tokens, [dim,B]-native): gate_up + down are MMA matmats
 * (the compute-bound win), one command buffer, activations resident. */
void  qwen_metal_ffn_swiglu_batched(void *ctx, float *out, const float *x, const float *norm_w,
                                    const uint16_t *Wgu, const uint16_t *Wd,
                                    int H, int inter, int B, float eps);

/* Amortized matvec cost with K dispatches in ONE command buffer (ms/op) —
 * the fused regime a real decode step runs in (isolates kernel throughput from
 * the per-op CPU<->GPU sync). */
double qwen_metal_matvec_bench_fused(void *ctx, const uint16_t *W, const float *x,
                                     int rows, int cols, int reps);

/* Full per-op correctness + resident-timing suite vs the CPU kernels.
 * out (FILE*) may be NULL → stdout. Returns number of failing ops (0 = PASS). */
int   qwen_metal_selftest(void *out);

/* GPU-resident fused Talker step (G2, mirrors the CUDA path). metal_ctx = the backend
 * impl (qwen_backend_t.impl from qwen_backend_init). ctx supplies config + weight pointers. */
struct qwen_tts_ctx;
void *qwen_metal_talker_init(void *metal_ctx, struct qwen_tts_ctx *ctx);
void  qwen_metal_talker_step(void *state, const float *embed, float *hidden_out, int pos);
void  qwen_metal_talker_upload_kv(void *state, struct qwen_tts_ctx *ctx, int prefill_len);
void  qwen_metal_talker_free(void *state);

#ifdef __cplusplus
}
#endif

#endif /* QWEN_TTS_METAL_H */
