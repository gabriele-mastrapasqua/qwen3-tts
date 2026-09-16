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
/* Execution budget: when "own" is on, OpenBLAS is held at ONE thread (it may compute a
 * serial SGEMM, it may not run its own worker team) and the decoder's SGEMMs are
 * partitioned across the engine pool by qwen_sd_sgemm instead.  QWEN_BLAS_OWN overrides. */
void qwen_blas_own(int on);
int  qwen_blas_own_get(void);
int  qwen_blas_own_effective(void);   /* own AND the platform really has BLAS thread control */
int  qwen_blas_threads_now(void);       /* openblas_get_num_threads(), -1 if unavailable */
/* Decoder parallel tiles: 1 = run on the engine pool (inline when already inside a region),
 * 0 = the decoder's private worker team.  QWEN_SD_POOL=engine|private overrides the default;
 * legacy qwen/q/1 and 0 aliases remain accepted. */
void qwen_sd_pool_default(int mode);
void qwen_sd_pool_validate(void);
int  qwen_sd_pool_mode(void);
/* Run one decoder job on the existing decoder/engine pool.  The callback must partition
 * its work with its own atomic cursor; this wrapper preserves the no-nested-pool rule used
 * by the established decoder workers. */
void qwen_sd_pool_run(void (*fn)(void *), void *ctx);
/* Claim the engine pool as the single owner of this process's compute budget:
 * decoder tiles on the engine pool, BLAS serial and its GEMMs partitioned there.
 * No-op at one thread; explicit QWEN_SD_POOL / QWEN_BLAS_OWN still win. */
void qwen_exec_budget_engine_owned(const char *who);
/* In-region int8 matmat (x86 AVX-512 VNNI): the dispatched drivers split into a per-column
 * activation quantisation and a per-thread row block, so a persistent parallel region can
 * run the same kernels between its own barriers.  Same partition and same kernels as the
 * dispatched path: outputs are bit-identical.  Xt is k-major [cols][B]; Y is [rows][B]. */
/* Prefill-only wide bf16 matmat (16 < B <= 64): separate entry, qwen_matmat_bf16 unchanged. */
int qwen_matmat_bf16_wide_available(int rows, int cols);
int qwen_matmat_bf16_wide(float *Y, const uint16_t *W, const float *X, int rows, int cols, int B);
/* Named for the capability, not for an ISA feature: "can SOME in-region row-block runner
 * execute this shape inside a held team".  It used to be called qwen_i8mm_usable, which
 * stopped being true the moment the AMX tiles became a valid in-region runner too. */
int   qwen_region_i8_usable(int rows, int cols, int B);
int   qwen_region_i8_qkv_usable(int q_rows, int kv_rows, int cols, int B);
float qwen_region_i8_quant_col(int8_t *qb, const float *Xt, int cols, int B, int b);
void  qwen_region_i8_run(float *Y, const int8_t *W, const float *scale, const int8_t *qXt,
                    const float *sx, int rows, int cols, int B, size_t tid, size_t nt);
void  qwen_region_i8_run_qkv(float *q, float *k, float *v,
                        const int8_t *Wq, const float *sq, const int8_t *Wk, const float *sk,
                        const int8_t *Wv, const float *sv, const int8_t *qXt, const float *sx,
                        int q_rows, int kv_rows, int cols, int B, size_t tid, size_t nt);
void qwen_sd_sgemm(int order, int ta, int tb, int M, int N, int K, float alpha,
                   const float *A, int lda, const float *B, int ldb, float beta,
                   float *C, int ldc);
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
    QWEN_MMK_BF16_BFMMLA, QWEN_MMK_BF16_AVX512, QWEN_MMK_BF16_FIXEDB, QWEN_MMK_BF16_GENERIC,
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

/* Stable runtime PATH ids for the shape census.  Append only, never renumber: the ids are
 * the join key between census.json, tools/census_report.py and dispatch_expect.json.
 * A path is an ENTRY the engine calls (what work was asked); which instructions did it
 * is the kernel mask (QWEN_MMK_*) plus the LEAF noted at the branch that ran. */
enum {
    QWEN_PATH_NONE = 0,
    QWEN_PATH_MATVEC_BF16 = 1, QWEN_PATH_MATVEC_BF16_QKV, QWEN_PATH_MATVEC_INT8,
    QWEN_PATH_MATVEC_INT8_QKV, QWEN_PATH_MATVEC_Q4_0, QWEN_PATH_MATVEC_Q4_0_QKV,
    QWEN_PATH_MATVEC_Q2_0, QWEN_PATH_MATVEC_Q6_0, QWEN_PATH_MATVEC_Q6_0_QKV,
    QWEN_PATH_ARGMAX_MATVEC_BF16 = 10, QWEN_PATH_ARGMAX_MATVEC_INT8, QWEN_PATH_ARGMAX_MATVEC_Q4_0,
    QWEN_PATH_MATMAT_BF16 = 20, QWEN_PATH_MATMAT_BF16_ROWS, QWEN_PATH_MATMAT_BF16_QKV,
    QWEN_PATH_MATMAT_INT8, QWEN_PATH_MATMAT_INT8_QKV, QWEN_PATH_MATMAT_Q4_0,
    QWEN_PATH_MATMAT_INT8_VNNI_PACKED_SLICE = 30, QWEN_PATH_MATMAT_INT8_VNNI_M4N2_SLICE,
    QWEN_PATH_PREFILL_BF16_NATIVE = 40, QWEN_PATH_PREFILL_F32_SGEMM, QWEN_PATH_BF16_ROWPACK_SHARED,
    QWEN_PATH_MATMAT_INT8_NATIVE, QWEN_PATH_MATMAT_BF16_NATIVE, QWEN_PATH_MATMAT_INT8_QKV_NATIVE,
    QWEN_PATH_DECODER_SGEMM = 50, QWEN_PATH_DECODER_CONV_INT8, QWEN_PATH_DECODER_CONV_NAIVE,
    QWEN_PATH_DECODER_CONV_AMX_INT8 = 53,
    QWEN_PATH_DECODER_CONV_AMX_INT8_D = 54,
    QWEN_PATH_DECODER_CONV_AMX_BF16 = 55,
    QWEN_PATH_COUNT
};
/* kind: what a row means for coverage accounting */
enum { QWEN_PATHK_CALL = 0, QWEN_PATHK_SLICE, QWEN_PATHK_WRAPPER, QWEN_PATHK_TRANSFORM };
const char *qwen_path_name(int path);
int         qwen_path_kind(int path);

/* LEAF: the instruction class that actually executed inside a path, noted at the branch. */
enum {
    QWEN_LEAF_NONE = 0, QWEN_LEAF_VNNI, QWEN_LEAF_DPBF16, QWEN_LEAF_SDOT, QWEN_LEAF_AVX512F,
    QWEN_LEAF_AVX2, QWEN_LEAF_NEON, QWEN_LEAF_SCALAR, QWEN_LEAF_BLAS, QWEN_LEAF_F32_FUSED,
    QWEN_LEAF_KLEIDI, QWEN_LEAF_AMX,
    QWEN_LEAF_DELEGATED,   /* the entry delegated to per-matrix calls: their rows carry the work */
    QWEN_LEAF_AVX2_INT8_GEMV, /* experimental exact signed widening dot product */
    QWEN_LEAF_COUNT
};
const char *qwen_leaf_name(int leaf);
void qwen_census_leaf(int leaf);

/* Experimental legacy-x86 GEMV candidate. It is opt-in until a real AVX2
 * hardware run proves that it beats the FMA reference. */
int qwen_avx2_int8_gemv_compiled(void);
int qwen_avx2_int8_gemv_supported(void);
int qwen_avx2_int8_gemv_enabled(void);
/* Same as qwen_census_op, but with an explicit MAC count and a bucketed B for the key:
 * the speech decoder's conv/GEMM "B" is the time length and differs on nearly every call,
 * which would fill the 256-row table with one row per length. */
void qwen_census_op_len(int path, int rows, int cols, int len);
void qwen_mm_component(int comp);
int  qwen_mm_component_get(void);

int  qwen_matmat_stats_enabled(void);
void qwen_matmat_stats_note(int kernel_id, long long macs);
void qwen_matmat_stats_note_bytes(long long weight_bytes);
int  qwen_census_enabled(void);
void qwen_census_op(int path, int rows, int cols, int B);
void qwen_census_frame(void);
void qwen_census_frame_at(int site);
void qwen_census_report(void *out);

void qwen_matmat_stats_reset(void);
void qwen_matmat_stats_report(void *out);
void qwen_kernel_timing_report(void *out);
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

/* Run the opt-in x86 AMX B=32 kernel when QWEN_AMX_B32=1. */
int qwen_matmat_int8_amx_b32(float *Y, const int8_t *W, const float *scale,
                             const float *X, int rows, int cols);

/* Return non-zero when the native x86 batched QKV path ran. */
int qwen_matmat_bf16_qkv(float *q, float *k, float *v,
                         const uint16_t *Wq, const uint16_t *Wk, const uint16_t *Wv,
                         const float *X, int in_dim, int q_dim, int kv_dim, int B);

int qwen_matmat_int8_qkv(float *q, float *k, float *v,
                         const int8_t *Wq, const float *sq,
                         const int8_t *Wk, const float *sk,
                         const int8_t *Wv, const float *sv,
                         const float *X, int in_dim, int q_dim, int kv_dim, int B);

void qwen_matvec_bf16_qkv(float *q, float *k, float *v,
                           const uint16_t *Wq, const uint16_t *Wk, const uint16_t *Wv,
                           const float *x, int in_dim, int q_dim, int kv_dim);

void qwen_linear_nobias_bf16(float *y, const float *x,
                             const uint16_t *W, int seq, int in_dim, int out_dim);

void qwen_linear(float *y, const float *x, const float *W, const float *bias,
                 int seq, int in_dim, int out_dim);

void qwen_matvec_int8(float *y, const int8_t *W, const float *scale,
                      const float *x, int rows, int cols);

/* Drop cached x86 VNNI weight row sums before unloading a model. */
void qwen_vnni_row_sums_reset(void);
/* Optional parent-side x86 VNNI weight packing for B>1 matmat. */
int qwen_vnni_prepack_weight(const int8_t *source, int rows, int cols);
void qwen_vnni_prepack_stats(int *n_packed, size_t *bytes);
void qwen_vnni_weight_cache_reset(void);
void qwen_amx_weight_cache_reset(void);

enum {
    QWEN_AMX_WEIGHT_BF16 = 1,
    QWEN_AMX_WEIGHT_INT8 = 2,
};
int qwen_amx_prepack_weight(const void *source, int rows, int cols, int kind);
void qwen_amx_prepack_stats(int *n_packed, size_t *bytes);

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
void qwen_causal_attention_bf16kv_heads(float *out, const float *Q,
                                        const uint16_t *K_bf16, const uint16_t *V_bf16,
                                        int seq_q, int seq_k, int n_heads, int n_kv_heads,
                                        int head_dim, float scale, int q_offset,
                                        int h_lo, int h_hi);
void qwen_causal_attention_bf16kv_prefill(float *out, const float *Q,
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
int qwen_sd_int8_usable(int in_ch, int out_ch);   /* available AND a shape the kernels cover */
void qwen_sd_int8_cache_reset(void);
/* Resolved speech-decoder policy used by the operational serving-profile gate.  These
 * answers are capability/policy answers, not benchmark claims: the exact convolution
 * shape still gates individual calls. */
int qwen_sd_amx_d_active(void);
int qwen_sd_amx_bf16_active(void);
int qwen_sd_stream_strip_active(void);
int qwen_sd_fused_residual_active(void);
const char *qwen_sd_decoder_mode(void);
int  qwen_sd_res1_v2_active(void);
int  qwen_sd_multislot_active(void);
int  qwen_sd_glue_active(void);       /* QWEN_SD_GLUE=1 on top of RES1_V2 (C12-WIN-12) */   /* QWEN_SD_RES1_V2=1 and the kernel is available on this ISA */
int8_t *qwen_sd_amx_int8_pack_weights(const int8_t *Wq, int rows, int Kp,
                                      size_t *bytes_out);
void qwen_sd_amx_int8_free_weights(int8_t *packed);
uint16_t *qwen_sd_amx_bf16_pack_weights(const float *W, int rows, int K,
                                        int *Kp_out, size_t *bytes_out);
void qwen_sd_amx_bf16_free_weights(uint16_t *packed);
/* Decoder ragged-batch panel entry points.  X is panel-local [N][K] and the output column
 * offset is global in out[rows][out_ld].  Return 1 only when the real AMX tile path ran. */
int qwen_sd_amx_int8_panel(float *out, int out_ld, int M,
                           const int8_t *Wpack, const float *sw,
                           const int32_t *wsum, const float *bias,
                           const int8_t *Xq, const float *sa,
                           int tcol0, int nc, int Kp, int blk);
int qwen_sd_amx_bf16_panel(float *out, int out_ld, int M,
                           const uint16_t *Wpack, const float *bias,
                           const float *Xf, int tcol0, int nc, int K, int Kp);
/* B=1 capability: 1 = a native integer GEMV runs, 0 = B=1 dequantises to the f32 twin. */
int qwen_int8_gemv_native(void);
int qwen_q4_gemv_native(void);
/* B>1 capability: the family that actually serves each dtype on this build, dispatcher
 * order, evaluated at a representative large shape.  Never NULL. */
const char *qwen_matmat_family_int8(void);
const char *qwen_matmat_family_q4(void);
const char *qwen_matmat_family_bf16(void);
int qwen_matmat_int8_max_b(void);
int qwen_amx_int8_pack_worth(int rows, int cols, int gate_rows, int threads);
void qwen_mm_force(int mmk);   /* bench hook: pin the batched dispatcher; 0 = normal */
const char *qwen_region_i8_backend(void);

int qwen_int8_kp(int K, int blk);

void qwen_int8_quant_rows(int8_t *dst, float *scales, const float *src,
                          int rows, int K, int Kp, int blk);

int qwen_amx_bf16_available(void);
int qwen_amx_int8_available(void);
int qwen_arm_bf16_matmat_available(void);
int qwen_avx512_bf16_matmat_available(void);

/* --dispatch-map support: resolved state of one g_mm_gate[] row, computed by the
 * dispatcher's own predicate (qwen_mm_use), plus the env that explains it. */
typedef struct {
    int mmk;
    const char *name;
    const char *off_env, *on_env, *minb_env, *minrows_env, *mincols_env;
    int compiled, supported, on;
    int min_b, compiled_min_b, max_b;
    int min_rows, compiled_min_rows, min_cols, compiled_min_cols;
    int amx, apple_off;
    const char *reason;
} qwen_mm_gate_desc_t;
int qwen_mm_gate_describe(int mmk, qwen_mm_gate_desc_t *d);   /* 0 = not a gated row */
int qwen_amx_prepack_requested(void);
int qwen_vnni_prepack_requested(void);
int qwen_q4_vnni_variant(void);        /* 2, 3 or 4; meaningful only with AVX-512 VNNI */
int qwen_bf16dot_enabled(void);        /* AVX-512 BF16 matvec (VDPBF16PS), QWEN_NO_BF16DOT */
int qwen_arm_bfdot_on(void);           /* Arm BFDOT bf16 matvec, opt-in QWEN_ARM_BFDOT=1 (Linux bf16 hosts) */
int qwen_prefill_matmat_resolved(const char **why);   /* the Talker prefill predicate */
int qwen_cp_prefill2_requested(void);  /* env/arch part of cp_prefill2 (weights decide the rest) */
int qwen_sd_int8_enabled(void);
int qwen_sd_bf16_preup_active(void);
int qwen_pool_spin_value(void);
int qwen_pool_narrow_value(void);
int qwen_dispatch_map_report(void *out, const char *json_path);
int qwen_effective_config_report(void *out);
int qwen_flag_gate_status(const char *flag, int *compiled, const char **kernel);
int qwen_blas_env_overridden(void);   /* requested vs effective, per declared flag */
int qwen_matmat_bf16_rows(float *Y, const uint16_t *W, const float *Xr,
                          int ldx, int rows, int cols, int B);
int qwen_matmat_bf16_rows_usable(int rows, int cols, int B);
void qwen_bf16_pack_rows(uint16_t *Xb, const float *Xr, int ldx, int cols, int B);
void qwen_matmat_bf16_packed(float *Y, const uint16_t *W, const uint16_t *Xb,
                             int rows, int cols, int B);

void qwen_conv1d_int8(float *out, const float *in,
                      const int8_t *Wq, const float *sw, const int32_t *wsum,
                      const float *bias,
                      int in_ch, int out_ch, int length, int kernel, int dilation,
                      int Kp, int blk);

/* DL-4 (QWEN_SD_RES1_V2): direct dilated causal conv1d on int8 VNNI for the decoder's
 * residual convs (in_ch == out_ch, k taps).  Activations are quantised once per time
 * position (one scale per position), weights once per (channel, tap) — no im2col panel.
 * wq [ch][kernel][Cp] s8, sw/wsum [ch][kernel]; Cp = ch rounded up to 64. */
int  qwen_conv1d_int8_v2_available(void);
void qwen_conv1d_int8_v2_ctx(float *out, const float *in, const float *tail, int tail_cols,
                             const float *residual,
                             const int8_t *wq, const float *sw, const int32_t *wsum,
                             const float *bias, int in_ch, int out_ch, int length, int kernel, int dilation, int Cp);
void qwen_convt_stack_epilogue(float *out, const float *R, float *carry, const float *bias,
                               int out_ch, int len, int stride);      /* C12-WIN-11 step A */
void qwen_convt_pack_stack(float *stack, const float *packed, int in_ch, int out_ch, int kernel);
int  qwen_sd_convt_stack_active(void);
int  qwen_conv1d_int8_v2_cp(int ch);                      /* padded channel count of the DL-4 layout */
void qwen_conv1d_int8_v2_pack(int8_t *q2, float *sw2, int32_t *ws2,
                              const float *w, int in_ch, int out_ch, int kernel, int Cp);
void qwen_conv1d_int8_v2(float *out, const float *in,
                         const int8_t *wq, const float *sw, const int32_t *wsum,
                         const float *bias, int in_ch, int out_ch, int length, int kernel, int dilation, int Cp);
/* DL-4 multi-slot cohort.  All slots share the packed weights and geometry; the
 * compact API uses [channel][length] buffers per slot.  The strided form is for
 * the ragged decoder workset, where each slot is a window into one global
 * [channel][total] allocation.  Both preserve the single-slot numerical order. */
int  qwen_conv1d_int8_v2_multi_available(void);
void qwen_conv1d_int8_v2_multi_ctx(float *const *out, const float *const *in,
                                   const float *const *tail, const int *tail_cols,
                                   const float *const *residual,
                                   const int8_t *wq, const float *sw, const int32_t *wsum,
                                   const float *bias, int in_ch, int out_ch, int nslots,
                                   int length, int kernel, int dilation, int Cp);
void qwen_conv1d_int8_v2_multi_ctx_strided(
                                   float *const *out, const float *const *in,
                                   const float *const *tail, const int *tail_cols,
                                   const float *const *residual,
                                   const size_t *in_stride, const size_t *out_stride,
                                   const int8_t *wq, const float *sw, const int32_t *wsum,
                                   const float *bias, int in_ch, int out_ch, int nslots,
                                   int length, int kernel, int dilation, int Cp);
void qwen_conv1d_int8_design_d(float *out, const float *in,
                               const int8_t *Wq, const float *sw, const int32_t *wsum,
                               const float *bias, const int8_t *Wpack,
                               int in_ch, int out_ch, int length, int kernel, int dilation,
                               int Kp, int blk);

/* Decoder-only experimental epilogue: compute the same Design-D convolution and add
 * `residual` before storing each output element.  The caller keeps separate input and
 * output buffers, so an unsupported/failing backend can fall back without corrupting
 * the convolution input. */
void qwen_conv1d_int8_design_d_residual(float *out, const float *in,
                                        const float *residual,
                                        const int8_t *Wq, const float *sw,
                                        const int32_t *wsum, const float *bias,
                                        const int8_t *Wpack,
                                        int in_ch, int out_ch, int length, int kernel,
                                        int dilation, int Kp, int blk);

/* Streaming causal slice: `in` has input_length columns, while the output buffer has
 * output_length columns corresponding to logical output columns [input_offset,
 * input_offset + output_length).  The AMX B pack and per-column quantisation contract
 * are identical to qwen_conv1d_int8_design_d(); return 1 when the backend accepted the
 * range, 0 when the caller must use its complete-path fallback. */
int qwen_conv1d_int8_design_d_range(float *out, const float *in,
                                    const int8_t *Wq, const float *sw,
                                    const int32_t *wsum, const float *bias,
                                    const int8_t *Wpack,
                                    int in_ch, int out_ch, int input_length,
                                    int input_offset, int output_length,
                                    int kernel, int dilation, int Kp, int blk);

/* Streaming causal slice with a split [prefix tail | suffix new-input] source.  The
 * output is exactly the suffix range, so the caller can avoid materialising the
 * concatenated fp32 activation buffer.  Return 1 when the backend accepted it. */
int qwen_conv1d_int8_design_d_range_split(float *out,
                                          const float *prefix, int prefix_length,
                                          const float *suffix, int suffix_length,
                                          const int8_t *Wq, const float *sw,
                                          const int32_t *wsum, const float *bias,
                                          const int8_t *Wpack,
                                          int in_ch, int out_ch, int output_length,
                                          int kernel, int dilation, int Kp, int blk);

void qwen_conv1d_bf16_amx(float *out, const float *in,
                          const float *bias, const uint16_t *Wpack,
                          int in_ch, int out_ch, int length, int kernel, int dilation,
                          int Kp);

void qwen_gemm_int8(float *out, int out_ld,
                    const int8_t *Wq, const float *sw, const int32_t *wsum,
                    const int8_t *Xq, const float *sa,
                    int M, int N, int Kp, int blk);

#endif
