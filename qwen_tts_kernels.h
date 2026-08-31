/* qwen_tts_kernels.h - Kernel function declarations */

#ifndef QWEN_TTS_KERNELS_H
#define QWEN_TTS_KERNELS_H

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

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

void qwen_set_threads(int n);

void qwen_blas_set_threads(int n);
int qwen_get_threads(void);
int qwen_get_num_cpus(void);
void qwen_init_threads(void);

void qwen_set_threads_soft(int n);
int  qwen_get_threads_hard(void);

void qwen_ftz_on(void);

void qwen_check_runtime_isa(void);

void qwen_caps_report(void *out);
void qwen_provenance_report(void *out);

enum {
    QWEN_MMK_NONE = 0,
    QWEN_MMK_BF16_BFMMLA, QWEN_MMK_BF16_FIXEDB, QWEN_MMK_BF16_GENERIC,
    QWEN_MMK_INT8_AMX, QWEN_MMK_INT8_VNNI, QWEN_MMK_INT8_AVX2,
    QWEN_MMK_INT8_SMMLA, QWEN_MMK_INT8_SDOT, QWEN_MMK_INT8_F32TWIN,
    QWEN_MMK_Q4_VNNI, QWEN_MMK_Q4_AVX2, QWEN_MMK_Q4_SMMLA,
    QWEN_MMK_Q4_BMATVEC, QWEN_MMK_Q4_GENERIC,
    QWEN_MMK_FORCED_MATVEC,
    QWEN_MMK_SOLO,
    QWEN_MMK_BF16_AMX,
    QWEN_MMK_Q4_AMX,
    QWEN_MMK_KLEIDI_Q4,
    QWEN_MMK_BF16_GEMV, QWEN_MMK_INT8_GEMV, QWEN_MMK_Q4_GEMV,
    QWEN_MMK_KLEIDI_I8,
    QWEN_MMK_KLEIDI_I8_GEMV,
    QWEN_MMK_KLEIDI_BF16,
    QWEN_MMK_KLEIDI_BF16_GEMV,
    QWEN_MMK_Q8_REPACK_I8MM,
    QWEN_MMK_Q8_REPACK_GEMV,
    QWEN_MMK_COUNT
};
enum { QWEN_COMP_OTHER = 0, QWEN_COMP_TALKER, QWEN_COMP_CP, QWEN_COMP_DECODER, QWEN_COMP_COUNT };
void qwen_mm_component(int comp);
int  qwen_mm_component_get(void);

int  qwen_matmat_stats_enabled(void);
void qwen_matmat_stats_note(int kernel_id, long long macs);
void qwen_matmat_stats_note_bytes(long long weight_bytes);
int  qwen_census_enabled(void);
void qwen_census_op(const char *entry, int rows, int cols, int B);
void qwen_census_frame(void);
void qwen_census_frame_at(int site);
void qwen_census_report(void *out);

void qwen_matmat_stats_reset(void);
void qwen_matmat_stats_report(void *out);
void qwen_kernel_selection_report(void *out, int rows, int cols);

int qwen_kernel_selftest(void *out);

int qwen_matmat_bench(void *out);

int qwen_matmat_tune(void *out, const char *model_dir);

void qwen_rms_norm(float *out, const float *x, const float *weight,
                   int seq, int dim, float eps);

void qwen_rms_norm_residual(float *out, float *x, const float *residual,
                            const float *weight, int dim, float eps);

void qwen_rms_norm_per_head(float *x, const float *weight,
                            int seq, int n_heads, int head_dim, float eps);

void qwen_matvec_bf16(float *y, const uint16_t *W, const float *x, int rows, int cols);

extern void (*g_qwen_matvec_bf16_hook)(float *, const uint16_t *, const float *, int, int);
extern void (*g_qwen_matmat_bf16_hook)(float *, const uint16_t *, const float *, int, int, int);

void qwen_matmat_bf16(float *Y, const uint16_t *W, const float *X, int rows, int cols, int B);

void qwen_matmat_int8(float *Y, const int8_t *W, const float *scale,
                      const float *X, int rows, int cols, int B);

void qwen_matvec_bf16_qkv(float *q, float *k, float *v,
                           const uint16_t *Wq, const uint16_t *Wk, const uint16_t *Wv,
                           const float *x, int in_dim, int q_dim, int kv_dim);

void qwen_linear_nobias_bf16(float *y, const float *x,
                             const uint16_t *W, int seq, int in_dim, int out_dim);

void qwen_linear(float *y, const float *x, const float *W, const float *bias,
                 int seq, int in_dim, int out_dim);

void qwen_matvec_int8(float *y, const int8_t *W, const float *scale,
                      const float *x, int rows, int cols);

void qwen_matvec_int8_qkv(float *q, float *k, float *v,
                           const int8_t *Wq, const float *sq,
                           const int8_t *Wk, const float *sk,
                           const int8_t *Wv, const float *sv,
                           const float *x, int in_dim, int q_dim, int kv_dim);

int qwen_argmax_matvec_int8(const float *x, const int8_t *W, const float *scale,
                            int in_dim, int out_dim);

void qwen_quantize_bf16_to_int8(const uint16_t *src_bf16, int rows, int cols,
                                 int8_t *dst_int8, float *dst_scale);

static inline float qwen_f16_to_f32(uint16_t h) {
#if defined(__aarch64__)
    __fp16 v; memcpy(&v, &h, sizeof(v)); return (float)v;
#else
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t em   = h & 0x7FFF;
    uint32_t bits;
    if (em >= 0x7C00)      bits = sign | 0x7F800000u | ((em & 0x03FF) << 13);
    else if (em >= 0x0400) bits = sign | ((em + ((127u - 15u) << 10)) << 13);
    else if (em == 0)      bits = sign;
    else {
        int shift = 0; uint32_t m = em;
        while (!(m & 0x0400)) { m <<= 1; shift++; }
        bits = sign | ((uint32_t)(127 - 15 - shift) << 23) | ((m & 0x03FF) << 13);
    }
    float f; memcpy(&f, &bits, sizeof(f)); return f;
#endif
}
static inline uint16_t qwen_f32_to_f16(float f) {
#if defined(__aarch64__)
    __fp16 v = (__fp16)f; uint16_t h; memcpy(&h, &v, sizeof(h)); return h;
#else
    uint32_t bits; memcpy(&bits, &f, sizeof(bits));
    uint32_t sign = (bits >> 16) & 0x8000;
    int32_t  e    = (int32_t)((bits >> 23) & 0xFF) - 127 + 15;
    uint32_t m    = bits & 0x007FFFFF;
    if (e >= 0x1F) return (uint16_t)(sign | 0x7C00);
    if (e <= 0) {
        if (e < -10) return (uint16_t)sign;
        m |= 0x00800000;
        uint32_t shift = (uint32_t)(14 - e);
        uint16_t sub = (uint16_t)(m >> shift);
        if ((m >> (shift - 1)) & 1) sub++;
        return (uint16_t)(sign | sub);
    }
    uint16_t out = (uint16_t)(sign | ((uint32_t)e << 10) | (m >> 13));
    if (m & 0x1000) out++;
    return out;
#endif
}

#define Q4_0_BLOCK_SIZE 32
typedef struct {
    uint16_t scale_f16;
    uint8_t qs[16];
} q4_0_block_t;

void qwen_quantize_bf16_to_q4_0(const uint16_t *src_bf16, int rows, int cols,
                                 q4_0_block_t *dst);

void qwen_matvec_q4_0(float *y, const q4_0_block_t *W, const float *x,
                       int rows, int cols);

void qwen_matmat_q4_0(float *Y, const q4_0_block_t *W, const float *X,
                      int rows, int cols, int B);

void qwen_matvec_q4_0_qkv(float *q, float *k, float *v,
                            const q4_0_block_t *Wq, const q4_0_block_t *Wk,
                            const q4_0_block_t *Wv,
                            const float *x, int in_dim, int q_dim, int kv_dim);

#define Q2_0_BLOCK_SIZE 32
typedef struct {
    float scale;
    uint8_t qs[8];
} q2_0_block_t;

void qwen_quantize_bf16_to_q2_0(const uint16_t *src_bf16, int rows, int cols,
                                 q2_0_block_t *dst);
void qwen_matvec_q2_0(float *y, const q2_0_block_t *W, const float *x,
                       int rows, int cols);

#define Q6_0_BLOCK_SIZE 32
typedef struct {
    uint16_t scale_f16;
    uint8_t  ql[16];
    uint8_t  qh[8];
} q6_0_block_t;

void qwen_quantize_bf16_to_q6_0(const uint16_t *src_bf16, int rows, int cols,
                                 q6_0_block_t *dst);

void qwen_matvec_q6_0(float *y, const q6_0_block_t *W, const float *x,
                       int rows, int cols);

void qwen_matvec_q6_0_qkv(float *q, float *k, float *v,
                          const q6_0_block_t *Wq, const q6_0_block_t *Wk,
                          const q6_0_block_t *Wv,
                          const float *x, int in_dim, int q_dim, int kv_dim);

void qwen_dequant_row_q6_0(float *dst, const q6_0_block_t *row, int cols);

void qwen_causal_attention_heads(float *out, const float *Q, const float *K, const float *V,
                                 int seq_q, int seq_k, int n_heads, int n_kv_heads,
                                 int head_dim, float scale, int q_offset, int h_lo, int h_hi);
void qwen_causal_attention_prefill(float *out, const float *Q, const float *K, const float *V,
                                   int seq_q, int seq_k, int n_heads, int n_kv_heads,
                                   int head_dim, float scale, int q_offset);
void qwen_causal_attention(float *out, const float *Q, const float *K, const float *V,
                           int seq_q, int seq_k, int n_heads, int n_kv_heads,
                           int head_dim, float scale, int q_offset);

void qwen_causal_attention_windowed(float *out, const float *Q, const float *K, const float *V,
                                     int seq_q, int seq_k, int n_heads, int n_kv_heads,
                                     int head_dim, float scale, int q_offset, int window);

void qwen_causal_attention_bf16kv(float *out, const float *Q,
                                  const uint16_t *K_bf16, const uint16_t *V_bf16,
                                  int seq_q, int seq_k, int n_heads, int n_kv_heads,
                                  int head_dim, float scale, int q_offset);

void qwen_compute_rope_interleaved(float *cos_out, float *sin_out, const int *positions,
                                   int seq, int head_dim, float theta);

void qwen_apply_rope_interleaved(float *x, const float *cos_vals, const float *sin_vals,
                                 int seq, int n_heads, int head_dim);

void qwen_silu(float *x, int n);

void qwen_swiglu_inplace(float *gate_up, float *tmp, int n);
void qwen_swiglu_prefill(float *gate_up, float *tmp, int n);

void qwen_add_inplace(float *y, const float *x, int n);

void qwen_mul_inplace(float *y, const float *x, int n);

void qwen_vec_scale_inplace(float *y, float s, int n);

void qwen_round_bf16(float *x, int n);

void qwen_bf16_accum_f32(float *dst, const uint16_t *src_bf16, int n);

void qwen_bf16_to_f32_vec(float *dst, const uint16_t *src_bf16, int n);

void qwen_snake_activation(float *data, int channels, int length,
                            const float *log_alpha, const float *log_beta);

int qwen_argmax_matvec_bf16(const float *x, const uint16_t *W_bf16, int in_dim, int out_dim);
int qwen_argmax_matvec_q4_0(const float *x, const q4_0_block_t *W, int in_dim, int out_dim);

#ifdef __cplusplus
}
#endif

int qwen_sd_int8_available(void);

int qwen_int8_kp(int K, int blk);

void qwen_int8_quant_rows(int8_t *dst, float *scales, const float *src,
                          int rows, int K, int Kp, int blk);

int qwen_amx_bf16_available(void);
int qwen_arm_bf16_matmat_available(void);

void qwen_conv1d_int8(float *out, const float *in,
                      const int8_t *Wq, const float *sw, const int32_t *wsum,
                      const float *bias,
                      int in_ch, int out_ch, int length, int kernel, int dilation,
                      int Kp, int blk);

void qwen_gemm_int8(float *out, int out_ld,
                    const int8_t *Wq, const float *sw, const int32_t *wsum,
                    const int8_t *Xq, const float *sa,
                    int M, int N, int Kp, int blk);

#endif
