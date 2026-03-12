/*
 * qwen_tts_kernels.c - Kernel implementations
 */

#include "qwen_tts_kernels.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/types.h>
#ifdef __APPLE__
#include <sys/sysctl.h>
#endif
#ifdef __linux__
#include <unistd.h>
#endif

#ifdef USE_BLAS
#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif
#endif

/* Threading */
static int g_n_threads = 1;

void qwen_set_threads(int n) { g_n_threads = n > 0 ? n : 1; }
int qwen_get_threads(void) { return g_n_threads; }

int qwen_get_num_cpus(void) {
    int ncpus = 1;
#if defined(__APPLE__)
    size_t len = sizeof(ncpus);
    sysctlbyname("hw.ncpu", &ncpus, &len, NULL, 0);
#elif defined(__linux__)
    ncpus = sysconf(_SC_NPROCESSORS_ONLN);
#endif
    return ncpus > 1 ? ncpus : 1;
}

void qwen_init_threads(void) {
    int ncpus = qwen_get_num_cpus();
    /* 4 threads is the sweet spot for bf16 matvec (memory-bandwidth-bound).
     * More threads add GCD dispatch overhead without bandwidth gain. */
    g_n_threads = ncpus < 4 ? ncpus : 4;
}

#if defined(__APPLE__)
#include <dispatch/dispatch.h>
#endif

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

/* ========================================================================
 * Norm functions
 * ======================================================================== */

void qwen_rms_norm(float *out, const float *x, const float *weight,
                   int seq, int dim, float eps) {
    for (int s = 0; s < seq; s++) {
        const float *xs = x + s * dim;
        float *os = out + s * dim;

#ifdef __ARM_NEON
        /* NEON: compute sum of squares */
        float32x4_t vsum0 = vdupq_n_f32(0), vsum1 = vdupq_n_f32(0);
        int i = 0;
        for (; i + 7 < dim; i += 8) {
            float32x4_t v0 = vld1q_f32(xs + i);
            float32x4_t v1 = vld1q_f32(xs + i + 4);
            vsum0 = vfmaq_f32(vsum0, v0, v0);
            vsum1 = vfmaq_f32(vsum1, v1, v1);
        }
        float sum = vaddvq_f32(vaddq_f32(vsum0, vsum1));
        for (; i < dim; i++) sum += xs[i] * xs[i];

        float inv_rms = 1.0f / sqrtf(sum / dim + eps);
        float32x4_t vinv = vdupq_n_f32(inv_rms);

        /* NEON: normalize and scale */
        i = 0;
        for (; i + 7 < dim; i += 8) {
            float32x4_t v0 = vld1q_f32(xs + i);
            float32x4_t v1 = vld1q_f32(xs + i + 4);
            float32x4_t w0 = vld1q_f32(weight + i);
            float32x4_t w1 = vld1q_f32(weight + i + 4);
            vst1q_f32(os + i,     vmulq_f32(vmulq_f32(v0, vinv), w0));
            vst1q_f32(os + i + 4, vmulq_f32(vmulq_f32(v1, vinv), w1));
        }
        for (; i < dim; i++) os[i] = xs[i] * inv_rms * weight[i];
#else
        float sum = 0.0f;
        for (int i = 0; i < dim; i++) sum += xs[i] * xs[i];
        float inv_rms = 1.0f / sqrtf(sum / dim + eps);
        for (int i = 0; i < dim; i++) os[i] = xs[i] * inv_rms * weight[i];
#endif
    }
}

void qwen_rms_norm_per_head(float *x, const float *weight,
                            int seq, int n_heads, int head_dim, float eps) {
    int dim = n_heads * head_dim;
    for (int s = 0; s < seq; s++) {
        float *xs = x + s * dim;
        for (int h = 0; h < n_heads; h++) {
            float *hs = xs + h * head_dim;

#ifdef __ARM_NEON
            float32x4_t vsum0 = vdupq_n_f32(0), vsum1 = vdupq_n_f32(0);
            int i = 0;
            for (; i + 7 < head_dim; i += 8) {
                float32x4_t v0 = vld1q_f32(hs + i);
                float32x4_t v1 = vld1q_f32(hs + i + 4);
                vsum0 = vfmaq_f32(vsum0, v0, v0);
                vsum1 = vfmaq_f32(vsum1, v1, v1);
            }
            float sum = vaddvq_f32(vaddq_f32(vsum0, vsum1));
            for (; i < head_dim; i++) sum += hs[i] * hs[i];

            float inv_rms = 1.0f / sqrtf(sum / head_dim + eps);
            float32x4_t vinv = vdupq_n_f32(inv_rms);

            i = 0;
            for (; i + 7 < head_dim; i += 8) {
                float32x4_t v0 = vld1q_f32(hs + i);
                float32x4_t v1 = vld1q_f32(hs + i + 4);
                float32x4_t w0 = vld1q_f32(weight + i);
                float32x4_t w1 = vld1q_f32(weight + i + 4);
                vst1q_f32(hs + i,     vmulq_f32(vmulq_f32(v0, vinv), w0));
                vst1q_f32(hs + i + 4, vmulq_f32(vmulq_f32(v1, vinv), w1));
            }
            for (; i < head_dim; i++) hs[i] *= inv_rms * weight[i];
#else
            float sum = 0.0f;
            for (int i = 0; i < head_dim; i++) sum += hs[i] * hs[i];
            float inv_rms = 1.0f / sqrtf(sum / head_dim + eps);
            for (int i = 0; i < head_dim; i++) hs[i] *= inv_rms * weight[i];
#endif
        }
    }
}

/* ========================================================================
 * Linear / MatVec
 * ======================================================================== */

static inline float bf16_to_f32(uint16_t bf) {
    uint32_t bits = (uint32_t)bf << 16;
    float val;
    memcpy(&val, &bits, sizeof(float));
    return val;
}


/* Fused bf16 matvec: processes 2 output rows at a time to amortize x vector loads.
 * On NEON: 32 elements/iter, 8 accumulators per row pair (from qwen-asr). */
static void bf16_matvec_fused(float *y, const float *x, const uint16_t *W,
                               int in_dim, int out_dim) {
    int o = 0;
#ifdef __ARM_NEON
    /* Process 2 output rows at a time — x loaded once, reused for both rows */
    for (; o + 1 < out_dim; o += 2) {
        const uint16_t *w0 = W + (size_t)o * in_dim;
        const uint16_t *w1 = W + (size_t)(o + 1) * in_dim;
        float32x4_t a0 = vdupq_n_f32(0), a1 = vdupq_n_f32(0),
                    a2 = vdupq_n_f32(0), a3 = vdupq_n_f32(0);
        float32x4_t b0 = vdupq_n_f32(0), b1 = vdupq_n_f32(0),
                    b2 = vdupq_n_f32(0), b3 = vdupq_n_f32(0);
        int k = 0;

        for (; k + 32 <= in_dim; k += 32) {
            float32x4_t x0 = vld1q_f32(x + k);
            float32x4_t x1 = vld1q_f32(x + k + 4);
            float32x4_t x2 = vld1q_f32(x + k + 8);
            float32x4_t x3 = vld1q_f32(x + k + 12);
            float32x4_t x4 = vld1q_f32(x + k + 16);
            float32x4_t x5 = vld1q_f32(x + k + 20);
            float32x4_t x6 = vld1q_f32(x + k + 24);
            float32x4_t x7 = vld1q_f32(x + k + 28);

            uint16x8_t r0a = vld1q_u16(w0 + k);
            uint16x8_t r0b = vld1q_u16(w0 + k + 8);
            uint16x8_t r0c = vld1q_u16(w0 + k + 16);
            uint16x8_t r0d = vld1q_u16(w0 + k + 24);
            a0 = vfmaq_f32(a0, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(r0a), 16)), x0);
            a1 = vfmaq_f32(a1, vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(r0a), 16)), x1);
            a2 = vfmaq_f32(a2, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(r0b), 16)), x2);
            a3 = vfmaq_f32(a3, vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(r0b), 16)), x3);
            a0 = vfmaq_f32(a0, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(r0c), 16)), x4);
            a1 = vfmaq_f32(a1, vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(r0c), 16)), x5);
            a2 = vfmaq_f32(a2, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(r0d), 16)), x6);
            a3 = vfmaq_f32(a3, vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(r0d), 16)), x7);

            uint16x8_t r1a = vld1q_u16(w1 + k);
            uint16x8_t r1b = vld1q_u16(w1 + k + 8);
            uint16x8_t r1c = vld1q_u16(w1 + k + 16);
            uint16x8_t r1d = vld1q_u16(w1 + k + 24);
            b0 = vfmaq_f32(b0, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(r1a), 16)), x0);
            b1 = vfmaq_f32(b1, vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(r1a), 16)), x1);
            b2 = vfmaq_f32(b2, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(r1b), 16)), x2);
            b3 = vfmaq_f32(b3, vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(r1b), 16)), x3);
            b0 = vfmaq_f32(b0, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(r1c), 16)), x4);
            b1 = vfmaq_f32(b1, vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(r1c), 16)), x5);
            b2 = vfmaq_f32(b2, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(r1d), 16)), x6);
            b3 = vfmaq_f32(b3, vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(r1d), 16)), x7);
        }
        for (; k + 8 <= in_dim; k += 8) {
            float32x4_t xv0 = vld1q_f32(x + k);
            float32x4_t xv1 = vld1q_f32(x + k + 4);
            uint16x8_t r0 = vld1q_u16(w0 + k);
            uint16x8_t r1 = vld1q_u16(w1 + k);
            a0 = vfmaq_f32(a0, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(r0), 16)), xv0);
            a1 = vfmaq_f32(a1, vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(r0), 16)), xv1);
            b0 = vfmaq_f32(b0, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(r1), 16)), xv0);
            b1 = vfmaq_f32(b1, vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(r1), 16)), xv1);
        }
        float s0 = vaddvq_f32(vaddq_f32(vaddq_f32(a0, a2), vaddq_f32(a1, a3)));
        float s1 = vaddvq_f32(vaddq_f32(vaddq_f32(b0, b2), vaddq_f32(b1, b3)));

        for (; k < in_dim; k++) {
            float wv0 = bf16_to_f32(w0[k]);
            float wv1 = bf16_to_f32(w1[k]);
            s0 += wv0 * x[k];
            s1 += wv1 * x[k];
        }
        y[o] = s0;
        y[o + 1] = s1;
    }
    /* Handle remaining odd row */
    if (o < out_dim) {
        const uint16_t *w_row = W + (size_t)o * in_dim;
        float32x4_t acc0 = vdupq_n_f32(0), acc1 = vdupq_n_f32(0);
        int k = 0;
        for (; k + 8 <= in_dim; k += 8) {
            uint16x8_t bf = vld1q_u16(w_row + k);
            acc0 = vfmaq_f32(acc0, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(bf), 16)),
                             vld1q_f32(x + k));
            acc1 = vfmaq_f32(acc1, vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(bf), 16)),
                             vld1q_f32(x + k + 4));
        }
        float sum = vaddvq_f32(vaddq_f32(acc0, acc1));
        for (; k < in_dim; k++) sum += bf16_to_f32(w_row[k]) * x[k];
        y[o] = sum;
    }
#else
    /* Generic fallback: single-row */
    for (; o < out_dim; o++) {
        const uint16_t *row = W + (size_t)o * in_dim;
        float sum = 0.0f;
        for (int k = 0; k < in_dim; k++) sum += bf16_to_f32(row[k]) * x[k];
        y[o] = sum;
    }
#endif
}

/* bf16 matvec: y[rows] = W[rows,cols] @ x[cols]
 * Multi-threaded via dispatch_apply on macOS. */
void qwen_matvec_bf16(float *y, const uint16_t *W, const float *x, int rows, int cols) {
#if defined(__APPLE__) && defined(__BLOCKS__)
    int nt = g_n_threads;
    if (nt > 1 && rows >= 256) {
        dispatch_apply((size_t)nt,
                       dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0),
                       ^(size_t tid) {
            int r0 = (int)tid * rows / nt;
            int r1 = (int)(tid + 1) * rows / nt;
            bf16_matvec_fused(y + r0, x, W + (size_t)r0 * cols, cols, r1 - r0);
        });
        return;
    }
#endif
    bf16_matvec_fused(y, x, W, cols, rows);
}

/* Unified QKV matvec: single dispatch for Q, K, V projections.
 * Avoids 3 separate dispatch_apply barriers per layer. */
void qwen_matvec_bf16_qkv(float *q, float *k, float *v,
                           const uint16_t *Wq, const uint16_t *Wk, const uint16_t *Wv,
                           const float *x, int in_dim, int q_dim, int kv_dim) {
#if defined(__APPLE__) && defined(__BLOCKS__)
    int nt = g_n_threads;
    int total_dim = q_dim + 2 * kv_dim;
    if (nt > 1 && total_dim >= 256) {
        dispatch_apply((size_t)nt,
                       dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0),
                       ^(size_t tid) {
            int r0 = (int)tid * total_dim / nt;
            int r1 = (int)(tid + 1) * total_dim / nt;
            for (int r = r0; r < r1; ) {
                if (r < q_dim) {
                    int chunk_end = r1 < q_dim ? r1 : q_dim;
                    bf16_matvec_fused(q + r, x, Wq + (size_t)r * in_dim,
                                       in_dim, chunk_end - r);
                    r = chunk_end;
                } else if (r < q_dim + kv_dim) {
                    int local = r - q_dim;
                    int chunk_end = r1 < q_dim + kv_dim ? r1 : q_dim + kv_dim;
                    int local_end = chunk_end - q_dim;
                    bf16_matvec_fused(k + local, x, Wk + (size_t)local * in_dim,
                                       in_dim, local_end - local);
                    r = chunk_end;
                } else {
                    int local = r - q_dim - kv_dim;
                    int local_end = r1 - q_dim - kv_dim;
                    bf16_matvec_fused(v + local, x, Wv + (size_t)local * in_dim,
                                       in_dim, local_end - local);
                    r = r1;
                }
            }
        });
        return;
    }
#endif
    bf16_matvec_fused(q, x, Wq, in_dim, q_dim);
    bf16_matvec_fused(k, x, Wk, in_dim, kv_dim);
    bf16_matvec_fused(v, x, Wv, in_dim, kv_dim);
}

void qwen_linear_nobias_bf16(float *y, const float *x,
                             const uint16_t *W, int seq, int in_dim, int out_dim) {
    for (int s = 0; s < seq; s++)
        qwen_matvec_bf16(y + s * out_dim, W, x + s * in_dim, out_dim, in_dim);
}

void qwen_linear(float *y, const float *x, const float *W, const float *bias,
                 int seq, int in_dim, int out_dim) {
    for (int s = 0; s < seq; s++) {
        const float *xs = x + s * in_dim;
        float *ys = y + s * out_dim;
        
        for (int o = 0; o < out_dim; o++) {
            float sum = bias ? bias[o] : 0.0f;
            const float *row = W + (int64_t)o * in_dim;
            for (int i = 0; i < in_dim; i++)
                sum += row[i] * xs[i];
            ys[o] = sum;
        }
    }
}

/* ========================================================================
 * INT8 MatVec (per-row absmax quantization)
 * ======================================================================== */

/* Quantize bf16 weight matrix to int8 with per-row absmax scaling.
 * scale[row] = max(|W_row|) / 127, W_int8[row][k] = round(W_bf16[row][k] / scale[row]) */
void qwen_quantize_bf16_to_int8(const uint16_t *src_bf16, int rows, int cols,
                                 int8_t *dst_int8, float *dst_scale) {
    for (int r = 0; r < rows; r++) {
        const uint16_t *row = src_bf16 + (size_t)r * cols;
        /* Find absmax */
        float amax = 0.0f;
#ifdef __ARM_NEON
        float32x4_t vmax = vdupq_n_f32(0);
        int k = 0;
        for (; k + 7 < cols; k += 8) {
            uint16x8_t bf = vld1q_u16(row + k);
            float32x4_t f0 = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(bf), 16));
            float32x4_t f1 = vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(bf), 16));
            vmax = vmaxq_f32(vmax, vabsq_f32(f0));
            vmax = vmaxq_f32(vmax, vabsq_f32(f1));
        }
        amax = vmaxvq_f32(vmax);
        for (; k < cols; k++) {
            uint32_t bits = (uint32_t)row[k] << 16;
            float val; memcpy(&val, &bits, sizeof(float));
            float a = fabsf(val);
            if (a > amax) amax = a;
        }
#else
        for (int k = 0; k < cols; k++) {
            uint32_t bits = (uint32_t)row[k] << 16;
            float val; memcpy(&val, &bits, sizeof(float));
            float a = fabsf(val);
            if (a > amax) amax = a;
        }
#endif
        float s = amax / 127.0f;
        dst_scale[r] = s;
        float inv_s = (s > 0) ? 127.0f / amax : 0.0f;

        /* Quantize */
        int8_t *dst_row = dst_int8 + (size_t)r * cols;
#ifdef __ARM_NEON
        float32x4_t vinv = vdupq_n_f32(inv_s);
        k = 0;
        for (; k + 7 < cols; k += 8) {
            uint16x8_t bf = vld1q_u16(row + k);
            float32x4_t f0 = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(bf), 16));
            float32x4_t f1 = vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(bf), 16));
            int32x4_t i0 = vcvtnq_s32_f32(vmulq_f32(f0, vinv));
            int32x4_t i1 = vcvtnq_s32_f32(vmulq_f32(f1, vinv));
            int16x4_t s0 = vqmovn_s32(i0);
            int16x4_t s1 = vqmovn_s32(i1);
            int8x8_t q = vqmovn_s16(vcombine_s16(s0, s1));
            vst1_s8(dst_row + k, q);
        }
        for (; k < cols; k++) {
            uint32_t bits = (uint32_t)row[k] << 16;
            float val; memcpy(&val, &bits, sizeof(float));
            int v = (int)roundf(val * inv_s);
            dst_row[k] = (int8_t)(v < -128 ? -128 : (v > 127 ? 127 : v));
        }
#else
        for (int k = 0; k < cols; k++) {
            uint32_t bits = (uint32_t)row[k] << 16;
            float val; memcpy(&val, &bits, sizeof(float));
            int v = (int)roundf(val * inv_s);
            dst_row[k] = (int8_t)(v < -128 ? -128 : (v > 127 ? 127 : v));
        }
#endif
    }
}

/* INT8 matvec inner kernel: process 2 rows at a time (NEON). */
static void int8_matvec_fused(float *y, const float *x, const int8_t *W,
                               const float *scale, int in_dim, int out_dim) {
    int o = 0;
#ifdef __ARM_NEON
    for (; o + 1 < out_dim; o += 2) {
        const int8_t *w0 = W + (size_t)o * in_dim;
        const int8_t *w1 = W + (size_t)(o + 1) * in_dim;
        float32x4_t a0 = vdupq_n_f32(0), a1 = vdupq_n_f32(0),
                    a2 = vdupq_n_f32(0), a3 = vdupq_n_f32(0);
        float32x4_t b0 = vdupq_n_f32(0), b1 = vdupq_n_f32(0),
                    b2 = vdupq_n_f32(0), b3 = vdupq_n_f32(0);
        int k = 0;

        for (; k + 15 < in_dim; k += 16) {
            /* Load 4 x vectors (f32) */
            float32x4_t x0 = vld1q_f32(x + k);
            float32x4_t x1 = vld1q_f32(x + k + 4);
            float32x4_t x2 = vld1q_f32(x + k + 8);
            float32x4_t x3 = vld1q_f32(x + k + 12);

            /* Load 16 int8 weights, convert to f32 */
            int8x16_t r0 = vld1q_s8(w0 + k);
            int16x8_t r0lo = vmovl_s8(vget_low_s8(r0));
            int16x8_t r0hi = vmovl_s8(vget_high_s8(r0));
            float32x4_t f00 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(r0lo)));
            float32x4_t f01 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(r0lo)));
            float32x4_t f02 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(r0hi)));
            float32x4_t f03 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(r0hi)));
            a0 = vfmaq_f32(a0, f00, x0);
            a1 = vfmaq_f32(a1, f01, x1);
            a2 = vfmaq_f32(a2, f02, x2);
            a3 = vfmaq_f32(a3, f03, x3);

            int8x16_t r1 = vld1q_s8(w1 + k);
            int16x8_t r1lo = vmovl_s8(vget_low_s8(r1));
            int16x8_t r1hi = vmovl_s8(vget_high_s8(r1));
            float32x4_t f10 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(r1lo)));
            float32x4_t f11 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(r1lo)));
            float32x4_t f12 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(r1hi)));
            float32x4_t f13 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(r1hi)));
            b0 = vfmaq_f32(b0, f10, x0);
            b1 = vfmaq_f32(b1, f11, x1);
            b2 = vfmaq_f32(b2, f12, x2);
            b3 = vfmaq_f32(b3, f13, x3);
        }
        float s0 = vaddvq_f32(vaddq_f32(vaddq_f32(a0, a2), vaddq_f32(a1, a3)));
        float s1 = vaddvq_f32(vaddq_f32(vaddq_f32(b0, b2), vaddq_f32(b1, b3)));
        for (; k < in_dim; k++) {
            s0 += (float)w0[k] * x[k];
            s1 += (float)w1[k] * x[k];
        }
        y[o] = s0 * scale[o];
        y[o + 1] = s1 * scale[o + 1];
    }
    if (o < out_dim) {
        const int8_t *w_row = W + (size_t)o * in_dim;
        float32x4_t acc0 = vdupq_n_f32(0), acc1 = vdupq_n_f32(0);
        int k = 0;
        for (; k + 7 < in_dim; k += 8) {
            int8x8_t r = vld1_s8(w_row + k);
            int16x8_t r16 = vmovl_s8(r);
            float32x4_t f0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(r16)));
            float32x4_t f1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(r16)));
            acc0 = vfmaq_f32(acc0, f0, vld1q_f32(x + k));
            acc1 = vfmaq_f32(acc1, f1, vld1q_f32(x + k + 4));
        }
        float sum = vaddvq_f32(vaddq_f32(acc0, acc1));
        for (; k < in_dim; k++) sum += (float)w_row[k] * x[k];
        y[o] = sum * scale[o];
    }
#else
    for (; o < out_dim; o++) {
        const int8_t *row = W + (size_t)o * in_dim;
        float sum = 0.0f;
        for (int k = 0; k < in_dim; k++) sum += (float)row[k] * x[k];
        y[o] = sum * scale[o];
    }
#endif
}

void qwen_matvec_int8(float *y, const int8_t *W, const float *scale,
                      const float *x, int rows, int cols) {
#if defined(__APPLE__) && defined(__BLOCKS__)
    int nt = g_n_threads;
    if (nt > 1 && rows >= 256) {
        dispatch_apply((size_t)nt,
                       dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0),
                       ^(size_t tid) {
            int r0 = (int)tid * rows / nt;
            int r1 = (int)(tid + 1) * rows / nt;
            int8_matvec_fused(y + r0, x, W + (size_t)r0 * cols,
                               scale + r0, cols, r1 - r0);
        });
        return;
    }
#endif
    int8_matvec_fused(y, x, W, scale, cols, rows);
}

void qwen_matvec_int8_qkv(float *q, float *k, float *v,
                           const int8_t *Wq, const float *sq,
                           const int8_t *Wk, const float *sk,
                           const int8_t *Wv, const float *sv,
                           const float *x, int in_dim, int q_dim, int kv_dim) {
#if defined(__APPLE__) && defined(__BLOCKS__)
    int nt = g_n_threads;
    if (nt > 1) {
        int total = q_dim + 2 * kv_dim;
        dispatch_apply((size_t)nt,
                       dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0),
                       ^(size_t tid) {
            int r0 = (int)tid * total / nt;
            int r1 = (int)(tid + 1) * total / nt;
            for (int r = r0; r < r1; r++) {
                float *dst;
                const int8_t *W;
                const float *sc;
                int dim;
                if (r < q_dim) {
                    dst = q; W = Wq; sc = sq; dim = q_dim;
                } else if (r < q_dim + kv_dim) {
                    dst = k; W = Wk; sc = sk; dim = kv_dim;
                    r -= q_dim;
                } else {
                    dst = v; W = Wv; sc = sv; dim = kv_dim;
                    r -= q_dim + kv_dim;
                }
                (void)dim;
                const int8_t *row = W + (size_t)r * in_dim;
                float sum = 0.0f;
#ifdef __ARM_NEON
                float32x4_t a0 = vdupq_n_f32(0), a1 = vdupq_n_f32(0);
                int kk = 0;
                for (; kk + 7 < in_dim; kk += 8) {
                    int8x8_t rr = vld1_s8(row + kk);
                    int16x8_t r16 = vmovl_s8(rr);
                    a0 = vfmaq_f32(a0, vcvtq_f32_s32(vmovl_s16(vget_low_s16(r16))),
                                   vld1q_f32(x + kk));
                    a1 = vfmaq_f32(a1, vcvtq_f32_s32(vmovl_s16(vget_high_s16(r16))),
                                   vld1q_f32(x + kk + 4));
                }
                sum = vaddvq_f32(vaddq_f32(a0, a1));
                for (; kk < in_dim; kk++) sum += (float)row[kk] * x[kk];
#else
                for (int kk = 0; kk < in_dim; kk++) sum += (float)row[kk] * x[kk];
#endif
                dst[r] = sum * sc[r];
            }
        });
        return;
    }
#endif
    int8_matvec_fused(q, x, Wq, sq, in_dim, q_dim);
    int8_matvec_fused(k, x, Wk, sk, in_dim, kv_dim);
    int8_matvec_fused(v, x, Wv, sv, in_dim, kv_dim);
}

int qwen_argmax_matvec_int8(const float *x, const int8_t *W, const float *scale,
                            int in_dim, int out_dim) {
    int best = 0;
    float best_val = -1e30f;
    for (int o = 0; o < out_dim; o++) {
        const int8_t *row = W + (size_t)o * in_dim;
        float sum = 0.0f;
#ifdef __ARM_NEON
        float32x4_t a0 = vdupq_n_f32(0), a1 = vdupq_n_f32(0);
        int k = 0;
        for (; k + 7 < in_dim; k += 8) {
            int8x8_t r = vld1_s8(row + k);
            int16x8_t r16 = vmovl_s8(r);
            a0 = vfmaq_f32(a0, vcvtq_f32_s32(vmovl_s16(vget_low_s16(r16))),
                           vld1q_f32(x + k));
            a1 = vfmaq_f32(a1, vcvtq_f32_s32(vmovl_s16(vget_high_s16(r16))),
                           vld1q_f32(x + k + 4));
        }
        sum = vaddvq_f32(vaddq_f32(a0, a1));
        for (; k < in_dim; k++) sum += (float)row[k] * x[k];
#else
        for (int k = 0; k < in_dim; k++) sum += (float)row[k] * x[k];
#endif
        sum *= scale[o];
        if (sum > best_val) { best_val = sum; best = o; }
    }
    return best;
}

/* ========================================================================
 * Attention
 * ======================================================================== */

void qwen_causal_attention(float *out, const float *Q, const float *K, const float *V,
                           int seq_q, int seq_k, int n_heads, int n_kv_heads,
                           int head_dim, float scale, int q_offset) {
    int heads_per_kv = n_heads / n_kv_heads;
    int q_hidden = n_heads * head_dim;
    int kv_hidden = n_kv_heads * head_dim;

    for (int h = 0; h < n_heads; h++) {
        int kv_h = h / heads_per_kv;
        
        for (int i = 0; i < seq_q; i++) {
            const float *q_row = Q + i * q_hidden + h * head_dim;
            float *o_row = out + i * q_hidden + h * head_dim;
            int k_end = q_offset + i + 1;  /* Causal: only attend to past */
            if (k_end > seq_k) k_end = seq_k;

            float max_score = -1e30f;
            float sum_exp = 0.0f;
            memset(o_row, 0, head_dim * sizeof(float));

            for (int j = 0; j < k_end; j++) {
                const float *k_row = K + j * kv_hidden + kv_h * head_dim;
                const float *v_row = V + j * kv_hidden + kv_h * head_dim;

                /* Dot product */
                float score;
#ifdef __ARM_NEON
                {
                    float32x4_t a0 = vdupq_n_f32(0), a1 = vdupq_n_f32(0);
                    float32x4_t a2 = vdupq_n_f32(0), a3 = vdupq_n_f32(0);
                    int d = 0;
                    for (; d + 15 < head_dim; d += 16) {
                        a0 = vfmaq_f32(a0, vld1q_f32(q_row + d),     vld1q_f32(k_row + d));
                        a1 = vfmaq_f32(a1, vld1q_f32(q_row + d + 4), vld1q_f32(k_row + d + 4));
                        a2 = vfmaq_f32(a2, vld1q_f32(q_row + d + 8), vld1q_f32(k_row + d + 8));
                        a3 = vfmaq_f32(a3, vld1q_f32(q_row + d + 12),vld1q_f32(k_row + d + 12));
                    }
                    score = vaddvq_f32(vaddq_f32(vaddq_f32(a0, a2), vaddq_f32(a1, a3)));
                    for (; d < head_dim; d++) score += q_row[d] * k_row[d];
                }
#else
                score = 0.0f;
                for (int d = 0; d < head_dim; d++)
                    score += q_row[d] * k_row[d];
#endif
                score *= scale;

                /* Softmax with numerical stability */
                if (score > max_score) {
                    float correction = expf(max_score - score);
                    sum_exp = sum_exp * correction + 1.0f;
#ifdef __ARM_NEON
                    {
                        float32x4_t vc = vdupq_n_f32(correction);
                        int d = 0;
                        for (; d + 15 < head_dim; d += 16) {
                            vst1q_f32(o_row + d,      vaddq_f32(vmulq_f32(vld1q_f32(o_row + d),      vc), vld1q_f32(v_row + d)));
                            vst1q_f32(o_row + d + 4,  vaddq_f32(vmulq_f32(vld1q_f32(o_row + d + 4),  vc), vld1q_f32(v_row + d + 4)));
                            vst1q_f32(o_row + d + 8,  vaddq_f32(vmulq_f32(vld1q_f32(o_row + d + 8),  vc), vld1q_f32(v_row + d + 8)));
                            vst1q_f32(o_row + d + 12, vaddq_f32(vmulq_f32(vld1q_f32(o_row + d + 12), vc), vld1q_f32(v_row + d + 12)));
                        }
                        for (; d < head_dim; d++)
                            o_row[d] = o_row[d] * correction + v_row[d];
                    }
#else
                    for (int d = 0; d < head_dim; d++)
                        o_row[d] = o_row[d] * correction + v_row[d];
#endif
                    max_score = score;
                } else {
                    float wt = expf(score - max_score);
                    sum_exp += wt;
#ifdef __ARM_NEON
                    {
                        float32x4_t vw = vdupq_n_f32(wt);
                        int d = 0;
                        for (; d + 15 < head_dim; d += 16) {
                            vst1q_f32(o_row + d,      vfmaq_f32(vld1q_f32(o_row + d),      vld1q_f32(v_row + d),      vw));
                            vst1q_f32(o_row + d + 4,  vfmaq_f32(vld1q_f32(o_row + d + 4),  vld1q_f32(v_row + d + 4),  vw));
                            vst1q_f32(o_row + d + 8,  vfmaq_f32(vld1q_f32(o_row + d + 8),  vld1q_f32(v_row + d + 8),  vw));
                            vst1q_f32(o_row + d + 12, vfmaq_f32(vld1q_f32(o_row + d + 12), vld1q_f32(v_row + d + 12), vw));
                        }
                        for (; d < head_dim; d++)
                            o_row[d] += v_row[d] * wt;
                    }
#else
                    for (int d = 0; d < head_dim; d++)
                        o_row[d] += v_row[d] * wt;
#endif
                }
            }

            if (sum_exp > 0.0f) {
                float inv_sum = 1.0f / sum_exp;
#ifdef __ARM_NEON
                {
                    float32x4_t vi = vdupq_n_f32(inv_sum);
                    int d = 0;
                    for (; d + 15 < head_dim; d += 16) {
                        vst1q_f32(o_row + d,      vmulq_f32(vld1q_f32(o_row + d),      vi));
                        vst1q_f32(o_row + d + 4,  vmulq_f32(vld1q_f32(o_row + d + 4),  vi));
                        vst1q_f32(o_row + d + 8,  vmulq_f32(vld1q_f32(o_row + d + 8),  vi));
                        vst1q_f32(o_row + d + 12, vmulq_f32(vld1q_f32(o_row + d + 12), vi));
                    }
                    for (; d < head_dim; d++) o_row[d] *= inv_sum;
                }
#else
                for (int d = 0; d < head_dim; d++)
                    o_row[d] *= inv_sum;
#endif
            }
        }
    }
}

/* Causal GQA attention with bf16 KV cache.
 * K_bf16/V_bf16 are stored as uint16_t (bf16), converted to f32 inline. */
void qwen_causal_attention_bf16kv(float *out, const float *Q,
                                  const uint16_t *K_bf16, const uint16_t *V_bf16,
                                  int seq_q, int seq_k, int n_heads, int n_kv_heads,
                                  int head_dim, float scale, int q_offset) {
    int heads_per_kv = n_heads / n_kv_heads;
    int q_hidden = n_heads * head_dim;
    int kv_hidden = n_kv_heads * head_dim;

    for (int h = 0; h < n_heads; h++) {
        int kv_h = h / heads_per_kv;

        for (int i = 0; i < seq_q; i++) {
            const float *q_row = Q + i * q_hidden + h * head_dim;
            float *o_row = out + i * q_hidden + h * head_dim;
            int k_end = q_offset + i + 1;
            if (k_end > seq_k) k_end = seq_k;

            float max_score = -1e30f;
            float sum_exp = 0.0f;
            memset(o_row, 0, head_dim * sizeof(float));

            for (int j = 0; j < k_end; j++) {
                const uint16_t *k_row_bf16 = K_bf16 + j * kv_hidden + kv_h * head_dim;
                const uint16_t *v_row_bf16 = V_bf16 + j * kv_hidden + kv_h * head_dim;

                /* Dot product: Q (f32) . K (bf16→f32) */
                float score;
#ifdef __ARM_NEON
                {
                    float32x4_t a0 = vdupq_n_f32(0), a1 = vdupq_n_f32(0);
                    float32x4_t a2 = vdupq_n_f32(0), a3 = vdupq_n_f32(0);
                    int d = 0;
                    for (; d + 15 < head_dim; d += 16) {
                        /* Convert bf16 K to f32 inline */
                        uint16x8_t bk0 = vld1q_u16(k_row_bf16 + d);
                        uint16x8_t bk1 = vld1q_u16(k_row_bf16 + d + 8);
                        float32x4_t k0 = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(bk0), 16));
                        float32x4_t k1 = vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(bk0), 16));
                        float32x4_t k2 = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(bk1), 16));
                        float32x4_t k3 = vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(bk1), 16));
                        a0 = vfmaq_f32(a0, vld1q_f32(q_row + d),      k0);
                        a1 = vfmaq_f32(a1, vld1q_f32(q_row + d + 4),  k1);
                        a2 = vfmaq_f32(a2, vld1q_f32(q_row + d + 8),  k2);
                        a3 = vfmaq_f32(a3, vld1q_f32(q_row + d + 12), k3);
                    }
                    score = vaddvq_f32(vaddq_f32(vaddq_f32(a0, a2), vaddq_f32(a1, a3)));
                    for (; d < head_dim; d++)
                        score += q_row[d] * bf16_to_f32(k_row_bf16[d]);
                }
#else
                score = 0.0f;
                for (int d = 0; d < head_dim; d++)
                    score += q_row[d] * bf16_to_f32(k_row_bf16[d]);
#endif
                score *= scale;

                /* Softmax with numerical stability + V accumulation (bf16→f32) */
                if (score > max_score) {
                    float correction = expf(max_score - score);
                    sum_exp = sum_exp * correction + 1.0f;
#ifdef __ARM_NEON
                    {
                        float32x4_t vc = vdupq_n_f32(correction);
                        int d = 0;
                        for (; d + 15 < head_dim; d += 16) {
                            uint16x8_t bv0 = vld1q_u16(v_row_bf16 + d);
                            uint16x8_t bv1 = vld1q_u16(v_row_bf16 + d + 8);
                            float32x4_t v0 = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(bv0), 16));
                            float32x4_t v1 = vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(bv0), 16));
                            float32x4_t v2 = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(bv1), 16));
                            float32x4_t v3 = vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(bv1), 16));
                            vst1q_f32(o_row + d,      vaddq_f32(vmulq_f32(vld1q_f32(o_row + d),      vc), v0));
                            vst1q_f32(o_row + d + 4,  vaddq_f32(vmulq_f32(vld1q_f32(o_row + d + 4),  vc), v1));
                            vst1q_f32(o_row + d + 8,  vaddq_f32(vmulq_f32(vld1q_f32(o_row + d + 8),  vc), v2));
                            vst1q_f32(o_row + d + 12, vaddq_f32(vmulq_f32(vld1q_f32(o_row + d + 12), vc), v3));
                        }
                        for (; d < head_dim; d++)
                            o_row[d] = o_row[d] * correction + bf16_to_f32(v_row_bf16[d]);
                    }
#else
                    for (int d = 0; d < head_dim; d++)
                        o_row[d] = o_row[d] * correction + bf16_to_f32(v_row_bf16[d]);
#endif
                    max_score = score;
                } else {
                    float wt = expf(score - max_score);
                    sum_exp += wt;
#ifdef __ARM_NEON
                    {
                        float32x4_t vw = vdupq_n_f32(wt);
                        int d = 0;
                        for (; d + 15 < head_dim; d += 16) {
                            uint16x8_t bv0 = vld1q_u16(v_row_bf16 + d);
                            uint16x8_t bv1 = vld1q_u16(v_row_bf16 + d + 8);
                            float32x4_t v0 = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(bv0), 16));
                            float32x4_t v1 = vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(bv0), 16));
                            float32x4_t v2 = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(bv1), 16));
                            float32x4_t v3 = vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(bv1), 16));
                            vst1q_f32(o_row + d,      vfmaq_f32(vld1q_f32(o_row + d),      v0, vw));
                            vst1q_f32(o_row + d + 4,  vfmaq_f32(vld1q_f32(o_row + d + 4),  v1, vw));
                            vst1q_f32(o_row + d + 8,  vfmaq_f32(vld1q_f32(o_row + d + 8),  v2, vw));
                            vst1q_f32(o_row + d + 12, vfmaq_f32(vld1q_f32(o_row + d + 12), v3, vw));
                        }
                        for (; d < head_dim; d++)
                            o_row[d] += bf16_to_f32(v_row_bf16[d]) * wt;
                    }
#else
                    for (int d = 0; d < head_dim; d++)
                        o_row[d] += bf16_to_f32(v_row_bf16[d]) * wt;
#endif
                }
            }

            if (sum_exp > 0.0f) {
                float inv_sum = 1.0f / sum_exp;
#ifdef __ARM_NEON
                {
                    float32x4_t vi = vdupq_n_f32(inv_sum);
                    int d = 0;
                    for (; d + 15 < head_dim; d += 16) {
                        vst1q_f32(o_row + d,      vmulq_f32(vld1q_f32(o_row + d),      vi));
                        vst1q_f32(o_row + d + 4,  vmulq_f32(vld1q_f32(o_row + d + 4),  vi));
                        vst1q_f32(o_row + d + 8,  vmulq_f32(vld1q_f32(o_row + d + 8),  vi));
                        vst1q_f32(o_row + d + 12, vmulq_f32(vld1q_f32(o_row + d + 12), vi));
                    }
                    for (; d < head_dim; d++) o_row[d] *= inv_sum;
                }
#else
                for (int d = 0; d < head_dim; d++)
                    o_row[d] *= inv_sum;
#endif
            }
        }
    }
}

/* ========================================================================
 * Element-wise ops
 * ======================================================================== */

void qwen_silu(float *x, int n) {
    for (int i = 0; i < n; i++)
        x[i] = x[i] / (1.0f + expf(-x[i]));
}


void qwen_add_inplace(float *y, const float *x, int n) {
    for (int i = 0; i < n; i++) y[i] += x[i];
}

void qwen_mul_inplace(float *y, const float *x, int n) {
    for (int i = 0; i < n; i++) y[i] *= x[i];
}

void qwen_vec_scale_inplace(float *y, float s, int n) {
    for (int i = 0; i < n; i++) y[i] *= s;
}

void qwen_round_bf16(float *x, int n) {
    for (int i = 0; i < n; i++) {
        uint16_t bf = (uint16_t)(((uint32_t)*(uint32_t*)&x[i]) >> 16);
        uint32_t bits = (uint32_t)bf << 16;
        memcpy(&x[i], &bits, sizeof(float));
    }
}

/* ========================================================================
 * Snake activation: x += (1/exp(beta)) * sin²(exp(alpha) * x)
 * ======================================================================== */

void qwen_snake_activation(float *data, int channels, int length,
                            const float *log_alpha, const float *log_beta) {
    for (int c = 0; c < channels; c++) {
        float a = expf(log_alpha[c]);
        float inv_b = expf(-log_beta[c]);
        float *row = data + (int64_t)c * length;

#if defined(__APPLE__) && defined(USE_BLAS)
        /* Use Accelerate vForce for vectorized sin — fast on Apple Silicon */
        {
            int n = length;
            float *temp = (float *)malloc(n * sizeof(float));

            /* temp = a * row */
            vDSP_vsmul(row, 1, &a, temp, 1, n);

            /* temp = sin(temp) */
            vvsinf(temp, temp, &n);

            /* temp = temp * temp (sin²) */
            vDSP_vsq(temp, 1, temp, 1, n);

            /* row += inv_b * temp */
            vDSP_vsma(temp, 1, &inv_b, row, 1, row, 1, n);

            free(temp);
        }
#elif defined(__ARM_NEON)
        {
            float32x4_t va = vdupq_n_f32(a);
            float32x4_t vinv_b = vdupq_n_f32(inv_b);
            int t = 0;
            for (; t + 3 < length; t += 4) {
                float32x4_t x = vld1q_f32(row + t);
                float32x4_t ax = vmulq_f32(va, x);
                /* Scalar sinf for each lane (no NEON intrinsic for sin) */
                float ax_s[4];
                vst1q_f32(ax_s, ax);
                float s_arr[4] = { sinf(ax_s[0]), sinf(ax_s[1]),
                                   sinf(ax_s[2]), sinf(ax_s[3]) };
                float32x4_t s = vld1q_f32(s_arr);
                float32x4_t s2 = vmulq_f32(s, s);
                x = vfmaq_f32(x, vinv_b, s2);
                vst1q_f32(row + t, x);
            }
            for (; t < length; t++) {
                float s = sinf(a * row[t]);
                row[t] += inv_b * s * s;
            }
        }
#else
        for (int t = 0; t < length; t++) {
            float s = sinf(a * row[t]);
            row[t] += inv_b * s * s;
        }
#endif
    }
}

/* ========================================================================
 * RoPE - Interleaved (already defined in talker.c, stub here)
 * ======================================================================== */

void qwen_compute_rope_interleaved(float *cos_out, float *sin_out, const int *positions,
                                   int seq, int head_dim, float theta) {
    int num_pairs = head_dim / 2;
    for (int s = 0; s < seq; s++) {
        float pos = (float)positions[s];
        for (int d = 0; d < num_pairs; d++) {
            float freq = 1.0f / powf(theta, (float)(2 * d) / head_dim);
            float angle = pos * freq;
            cos_out[s * num_pairs + d] = cosf(angle);
            sin_out[s * num_pairs + d] = sinf(angle);
        }
    }
}

void qwen_apply_rope_interleaved(float *x, const float *cos_vals, const float *sin_vals,
                                 int seq, int n_heads, int head_dim) {
    int num_pairs = head_dim / 2;
    int hidden = n_heads * head_dim;
    
    for (int s = 0; s < seq; s++) {
        const float *c = cos_vals + s * num_pairs;
        const float *sn = sin_vals + s * num_pairs;
        
        for (int h = 0; h < n_heads; h++) {
            float *vec = x + s * hidden + h * head_dim;
            for (int d = 0; d < num_pairs; d++) {
                float x_even = vec[2 * d];
                float x_odd  = vec[2 * d + 1];
                vec[2 * d]     = x_even * c[d] - x_odd * sn[d];
                vec[2 * d + 1] = x_odd  * c[d] + x_even * sn[d];
            }
        }
    }
}

/* ========================================================================
 * Argmax
 * ======================================================================== */

int qwen_argmax_matvec_bf16(const float *x, const uint16_t *W_bf16, int in_dim, int out_dim) {
    int best_idx = 0;
    float best_val = -1e30f;
    int o = 0;

#ifdef __ARM_NEON
    /* Process 2 rows at a time, reusing x vector loads */
    for (; o + 1 < out_dim; o += 2) {
        const uint16_t *w0 = W_bf16 + (size_t)o * in_dim;
        const uint16_t *w1 = W_bf16 + (size_t)(o + 1) * in_dim;
        float32x4_t a0 = vdupq_n_f32(0), a1 = vdupq_n_f32(0),
                    a2 = vdupq_n_f32(0), a3 = vdupq_n_f32(0);
        float32x4_t b0 = vdupq_n_f32(0), b1 = vdupq_n_f32(0),
                    b2 = vdupq_n_f32(0), b3 = vdupq_n_f32(0);
        int k = 0;
        for (; k + 32 <= in_dim; k += 32) {
            float32x4_t x0 = vld1q_f32(x + k);
            float32x4_t x1 = vld1q_f32(x + k + 4);
            float32x4_t x2 = vld1q_f32(x + k + 8);
            float32x4_t x3 = vld1q_f32(x + k + 12);
            float32x4_t x4 = vld1q_f32(x + k + 16);
            float32x4_t x5 = vld1q_f32(x + k + 20);
            float32x4_t x6 = vld1q_f32(x + k + 24);
            float32x4_t x7 = vld1q_f32(x + k + 28);

            uint16x8_t r0a = vld1q_u16(w0 + k), r0b = vld1q_u16(w0 + k + 8);
            uint16x8_t r0c = vld1q_u16(w0 + k + 16), r0d = vld1q_u16(w0 + k + 24);
            a0 = vfmaq_f32(a0, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(r0a), 16)), x0);
            a1 = vfmaq_f32(a1, vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(r0a), 16)), x1);
            a2 = vfmaq_f32(a2, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(r0b), 16)), x2);
            a3 = vfmaq_f32(a3, vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(r0b), 16)), x3);
            a0 = vfmaq_f32(a0, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(r0c), 16)), x4);
            a1 = vfmaq_f32(a1, vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(r0c), 16)), x5);
            a2 = vfmaq_f32(a2, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(r0d), 16)), x6);
            a3 = vfmaq_f32(a3, vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(r0d), 16)), x7);

            uint16x8_t r1a = vld1q_u16(w1 + k), r1b = vld1q_u16(w1 + k + 8);
            uint16x8_t r1c = vld1q_u16(w1 + k + 16), r1d = vld1q_u16(w1 + k + 24);
            b0 = vfmaq_f32(b0, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(r1a), 16)), x0);
            b1 = vfmaq_f32(b1, vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(r1a), 16)), x1);
            b2 = vfmaq_f32(b2, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(r1b), 16)), x2);
            b3 = vfmaq_f32(b3, vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(r1b), 16)), x3);
            b0 = vfmaq_f32(b0, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(r1c), 16)), x4);
            b1 = vfmaq_f32(b1, vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(r1c), 16)), x5);
            b2 = vfmaq_f32(b2, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(r1d), 16)), x6);
            b3 = vfmaq_f32(b3, vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(r1d), 16)), x7);
        }
        float s0 = vaddvq_f32(vaddq_f32(vaddq_f32(a0, a2), vaddq_f32(a1, a3)));
        float s1 = vaddvq_f32(vaddq_f32(vaddq_f32(b0, b2), vaddq_f32(b1, b3)));
        for (; k < in_dim; k++) {
            float wv0 = bf16_to_f32(w0[k]), wv1 = bf16_to_f32(w1[k]);
            s0 += wv0 * x[k];
            s1 += wv1 * x[k];
        }
        if (s0 > best_val) { best_val = s0; best_idx = o; }
        if (s1 > best_val) { best_val = s1; best_idx = o + 1; }
    }
#endif

    /* Handle remaining rows (odd count or generic fallback) */
    for (; o < out_dim; o++) {
        const uint16_t *row = W_bf16 + (size_t)o * in_dim;
        float sum = 0.0f;
        for (int k = 0; k < in_dim; k++) sum += bf16_to_f32(row[k]) * x[k];
        if (sum > best_val) { best_val = sum; best_idx = o; }
    }
    return best_idx;
}
