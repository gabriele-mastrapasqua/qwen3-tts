/*
 * qwen_tts_kernels.h - Kernel function declarations
 */

#ifndef QWEN_TTS_KERNELS_H
#define QWEN_TTS_KERNELS_H

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ========================================================================
 * Cache-line aligned allocation (64B for Apple M1/M2/x86-64)
 * Cross-platform: uses POSIX posix_memalign on all targets.
 * All BLAS/SIMD buffers MUST use these to avoid cache-line splits.
 * ======================================================================== */

static inline void *aligned_malloc(size_t size) {
    void *ptr = NULL;
    if (posix_memalign(&ptr, 64, size) != 0) return NULL;
    return ptr;
}
static inline void *aligned_calloc(size_t count, size_t size) {
    size_t total = count * size;
    void *ptr = aligned_malloc(total);
    if (ptr) memset(ptr, 0, total);
    return ptr;
}

/* ========================================================================
 * Threading
 * ======================================================================== */

void qwen_set_threads(int n);
int qwen_get_threads(void);
int qwen_get_num_cpus(void);
void qwen_init_threads(void);

/* Enable flush-to-zero / denormals-are-zero on the CURRENT thread (FPCR on ARM,
 * MXCSR on x86). Per-thread state, so every compute thread — including pool
 * workers — must call it. Cheap (~1-2 cycles); inaudible quality impact. */
void qwen_ftz_on(void);

/* Abort with a clear message if this binary was compiled for an ISA the running
 * CPU does not support (x86: -mavx2 build on a CPU without AVX2). No-op on ARM
 * and on portable builds. Call once at startup before any SIMD kernel runs. */
void qwen_check_runtime_isa(void);

/* Print the ACTUAL compiled SIMD/threading capabilities of this binary to `out`
 * (derived from the same #ifdef guards the kernels use). Makes the real state
 * visible + testable so a "we thought AVX existed" gap can't hide behind docs.
 * `out` may be NULL -> stderr. */
void qwen_caps_report(void *out);

/* Kernel numeric self-test: runs the dispatched matvecs (bf16/int8/argmax-int8/
 * q4_0/argmax-q4_0) against an f32 reference on deterministic random data. Cross-ISA correctness
 * proof for the SIMD kernels (esp. the AVX-512/VNNI paths) that does NOT depend
 * on a full-pipeline golden, so it's immune to the greedy trajectory fork.
 * `out` may be NULL -> stdout. Returns 0 on PASS, >0 = number of failed cases. */
int qwen_kernel_selftest(void *out);

/* ========================================================================
 * Norm functions
 * ======================================================================== */

/* RMSNorm: out = x / sqrt(mean(x^2) + eps) * weight */
void qwen_rms_norm(float *out, const float *x, const float *weight,
                   int seq, int dim, float eps);

/* Fused residual-add + RMSNorm: x[i] += residual[i], then out = RMSNorm(x, weight).
 * Saves one full pass over x compared to separate add + norm.
 * x is modified in-place (residual added), then normalized into out. */
void qwen_rms_norm_residual(float *out, float *x, const float *residual,
                            const float *weight, int dim, float eps);

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

/* Q4_0 block: 32 weights packed into 16 nibble-pair bytes + fp32 scale.
 * DEINTERLEAVED packing: qs[i] low 4 bits = weight i, high 4 bits = weight
 * i+16 — so SIMD unpack (and 0x0F / shr 4) yields the two 16-weight halves
 * already in natural order (no zip), which is what the SDOT path needs. */
#define Q4_0_BLOCK_SIZE 32
typedef struct {
    float scale;           /* per-block scale factor */
    uint8_t qs[16];        /* 32 nibbles: lo = weights 0-15, hi = weights 16-31 */
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

/* Q2_0 block: 32 weights at 2 bits each (8 bytes) + fp32 scale = 12 bytes.
 * 4 symmetric levels: dequant(code) = (code - 1.5) * scale, code in {0,1,2,3}
 * -> {-1.5,-0.5,0.5,1.5}*scale, scale = absmax/1.5. EXPERIMENTAL hybrid lever:
 * used on the quant-tolerant FFN matrices to shrink the CP working set below int4. */
#define Q2_0_BLOCK_SIZE 32
typedef struct {
    float scale;           /* per-block scale factor */
    uint8_t qs[8];         /* 32 codes × 2 bits, 4 codes per byte (idx i -> byte i/4, bits (i%4)*2) */
} q2_0_block_t;            /* 12 bytes per 32 weights */

void qwen_quantize_bf16_to_q2_0(const uint16_t *src_bf16, int rows, int cols,
                                 q2_0_block_t *dst);
void qwen_matvec_q2_0(float *y, const q2_0_block_t *W, const float *x,
                       int rows, int cols);

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

/* Fused SwiGLU: interleaved [g0,u0,g1,u1,...] → [silu(g0)*u0, silu(g1)*u1, ...]
 * Uses vvexpf (Accelerate) on macOS for batch exp, scalar loop elsewhere.
 * tmp must have space for n floats (used for batch exp). */
void qwen_swiglu_inplace(float *gate_up, float *tmp, int n);

/* Add: y += x */
void qwen_add_inplace(float *y, const float *x, int n);

/* Mul: y *= x */
void qwen_mul_inplace(float *y, const float *x, int n);

/* Scale: y *= s */
void qwen_vec_scale_inplace(float *y, float s, int n);

/* bf16 rounding */
void qwen_round_bf16(float *x, int n);

/* Accumulate bf16 vector into f32: dst[i] += bf16_to_f32(src[i])
 * NEON/AVX optimized for batch BF16→F32 conversion + addition. */
void qwen_bf16_accum_f32(float *dst, const uint16_t *src_bf16, int n);

/* Convert bf16 vector to f32: dst[i] = bf16_to_f32(src[i])
 * NEON/AVX2 vectorized. */
void qwen_bf16_to_f32_vec(float *dst, const uint16_t *src_bf16, int n);

/* Snake activation: x += (1/exp(beta)) * sin²(exp(alpha) * x)
 * Applied per-channel to channel-first data [channels, length].
 * log_alpha/log_beta are per-channel params in LOG SPACE. */
void qwen_snake_activation(float *data, int channels, int length,
                            const float *log_alpha, const float *log_beta);

/* ========================================================================
 * Argmax / Sampling
 * ======================================================================== */

int qwen_argmax_matvec_bf16(const float *x, const uint16_t *W_bf16, int in_dim, int out_dim);
int qwen_argmax_matvec_q4_0(const float *x, const q4_0_block_t *W, int in_dim, int out_dim);

#ifdef __cplusplus
}
#endif

#endif /* QWEN_TTS_KERNELS_H */
