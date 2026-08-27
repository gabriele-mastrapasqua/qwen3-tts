/*
 * qwen_tts_speech_decoder.c - Speech Decoder (ConvNet) forward pass
 * Converts 16 codebook codes per frame → 24kHz audio waveform
 *
 * Architecture:
 * 1. VQ dequant (16 codebooks × 2048 × 256) → sum → project to 512
 * 2. Pre-conv (512→1024, k=3, causal)
 * 3. Pre-transformer (8 layers, hidden=512, sliding window=72, layer_scale)
 * 4. Output proj (512→1024)
 * 5. ConvNeXt upsample (2 blocks, 2x each)
 * 6. Initial conv (1024→1536, k=7)
 * 7. 4 Decoder upsample blocks (rates: 8,5,4,3) with 3 residual blocks each
 * 8. Final snake + conv (96→1, k=7) → audio
 *
 * Tensor naming from safetensors:
 *   decoder.upsample.{0,1}.0.conv.{weight,bias}        - ConvNeXt ConvTranspose
 *   decoder.upsample.{0,1}.1.{dwconv.conv,norm,...}     - ConvNeXt block
 *   decoder.decoder.0.conv.{weight,bias}                - initial conv
 *   decoder.decoder.{1-4}.block.0.{alpha,beta}          - snake before upsample
 *   decoder.decoder.{1-4}.block.1.conv.{weight,bias}    - ConvTranspose upsample
 *   decoder.decoder.{1-4}.block.{2-4}.{act1,conv1,act2,conv2} - ResBlocks
 *   decoder.decoder.5.{alpha,beta}                      - final snake
 *   decoder.decoder.6.conv.{weight,bias}                - final conv
 */

#include <pthread.h>
#include <unistd.h>   /* getpid() for the QWEN_SD_PHASE line: glibc does not pull it in transitively the way macOS does */
#include "qwen_tts.h"
#include "qwen_tts_kernels.h"
#include "ingot/safetensors.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif
#ifdef __AVX2__
#include <immintrin.h>
#endif

/* aligned_malloc/aligned_calloc now in qwen_tts_kernels.h */

#ifdef USE_BLAS
#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
/* Modern Accelerate (ACCELERATE_NEW_LAPACK, cblas_new.h) declares `enum CBLAS_TRANSPOSE`
 * (C-style tag) rather than the old `typedef ... CBLAS_TRANSPOSE`, so the cast needs the
 * `enum` keyword. OpenBLAS keeps the typedef. Portable alias: */
#define QWEN_CBLAS_TRANSPOSE enum CBLAS_TRANSPOSE
#else
#include <cblas.h>
#define QWEN_CBLAS_TRANSPOSE CBLAS_TRANSPOSE
#endif
#define CONV_TILE_MAX_BYTES (256 * 1024 * 1024)

/* SD_GEMM: RowMajor sgemm that routes the decoder's big conv matmuls to cuBLAS (GPU, M3)
 * when g_cuda_decoder_on, else OpenBLAS/Accelerate. Signature drops the CblasRowMajor arg. */
#ifdef QWEN_HAVE_CUDA
#include "qwen_tts_cuda.h"
static inline void SD_GEMM(int ta,int tb,int M,int N,int K,float al,const float *A,int lda,
                           const float *B,int ldb,float be,float *C,int ldc){
    if (g_cuda_decoder_on &&
        qwen_cuda_sd_sgemm(ta==CblasTrans, tb==CblasTrans, M,N,K, al,A,lda,B,ldb,be,C,ldc) == 0)
        return;   /* did it on the GPU; else fall through to CPU (too big / unsupported) */
    cblas_sgemm(CblasRowMajor,(QWEN_CBLAS_TRANSPOSE)ta,(QWEN_CBLAS_TRANSPOSE)tb,M,N,K,al,A,lda,B,ldb,be,C,ldc);
}
#else
static inline void SD_GEMM(int ta,int tb,int M,int N,int K,float al,const float *A,int lda,
                           const float *B,int ldb,float be,float *C,int ldc){
    cblas_sgemm(CblasRowMajor,(QWEN_CBLAS_TRANSPOSE)ta,(QWEN_CBLAS_TRANSPOSE)tb,M,N,K,al,A,lda,B,ldb,be,C,ldc);
}
#endif
#endif

static const float *get_f32(void *ms, const char *name) {
    const ingot_st_tensor *t = ingot_st_find((ingot_st *)ms, name);
    return t ? (const float *)ingot_st_data((ingot_st *)ms, t) : NULL;
}

/* Causal Conv1d: out_len = (in_len + pad_left - kernel) / stride + 1 */
static int conv1d_out_len(int in_len, int kernel, int stride, int pad_left) {
    return (in_len + pad_left - kernel) / stride + 1;
}

/* Causal ConvTranspose1d: out_len = (in_len-1)*stride + kernel - (kernel-stride) = in_len*stride */
static int conv_transpose1d_out_len(int in_len, int kernel, int stride) {
    return (in_len - 1) * stride + kernel - (kernel - stride);
}

/* Snake activation dispatched through qwen_snake_activation() kernel
 * (NEON/Accelerate-optimized in qwen_tts_kernels.c) */
#define snake_activation qwen_snake_activation

#ifndef USE_BLAS
/* Naive causal Conv1d: [out_ch, in_ch, kernel], pad_left=(kernel-1)*dilation */
static void causal_conv1d_naive(float *out, const float *in,
                                const float *weight, const float *bias,
                                int in_ch, int out_ch, int length,
                                int kernel, int dilation) {
    int pad_left = (kernel - 1) * dilation;
    for (int oc = 0; oc < out_ch; oc++) {
        float b = bias ? bias[oc] : 0;
        for (int t = 0; t < length; t++) {
            float sum = b;
            for (int ic = 0; ic < in_ch; ic++) {
                for (int k = 0; k < kernel; k++) {
                    int in_pos = t - pad_left + k * dilation;
                    if (in_pos >= 0 && in_pos < length) {
                        sum += weight[((int64_t)oc * in_ch + ic) * kernel + k]
                             * in[(int64_t)ic * length + in_pos];
                    }
                }
            }
            out[(int64_t)oc * length + t] = sum;
        }
    }
}

/* Naive causal ConvTranspose1d: [in_ch, out_ch, kernel], stride, trim right by (kernel-stride) */
static void causal_conv_transpose1d_naive(float *out, const float *in,
                                          const float *weight, const float *bias,
                                          int in_ch, int out_ch, int in_len, int out_len,
                                          int kernel, int stride) {
    memset(out, 0, (int64_t)out_ch * out_len * sizeof(float));
    /* The right trim is expressed by out_len alone: one-shot callers pass
     * out_len = in_len*stride = full_len - (kernel-stride), which drops the tail
     * columns exactly as before. The streaming path passes out_len = full_len to
     * keep those columns as the overlap-add carry. */

    for (int ic = 0; ic < in_ch; ic++) {
        for (int t = 0; t < in_len; t++) {
            float x = in[(int64_t)ic * in_len + t];
            for (int k = 0; k < kernel; k++) {
                int out_pos = t * stride + k;
                if (out_pos < out_len) {
                    for (int oc = 0; oc < out_ch; oc++) {
                        out[(int64_t)oc * out_len + out_pos] +=
                            x * weight[((int64_t)ic * out_ch + oc) * kernel + k];
                    }
                }
            }
        }
    }
    if (bias) {
        for (int oc = 0; oc < out_ch; oc++)
            for (int t = 0; t < out_len; t++)
                out[(int64_t)oc * out_len + t] += bias[oc];
    }
}
#endif /* !USE_BLAS */


/* ========================================================================
 * INT8 conv path for the decoder (PR #17 sub-change B, ported)
 *
 * OPT-IN ONLY (QWEN_SD_INT8=1) and only on ARM dotprod: it trades audio
 * quality for speed. Measured on Neoverse-N1 (0.6B --int4 -j4): stream RTF
 * 1.41 -> 1.15, decoder 7735 -> 5112 ms. Added noise floor ~-65 dBFS RMS.
 * The ConvTranspose int8 variant is NOT ported: the author measured it slower
 * than fp32 sgemm (small K pads poorly into blocks) and ships it disabled.
 * ======================================================================== */
static int sd_int8_enabled(void) {
    static int en = -1;
    if (en < 0) {
        const char *e = getenv("QWEN_SD_INT8");
#if defined(__AVX512VNNI__)
        /* x86: DEFAULT ON since 2026-08-19, after the ear passed it.
         * The listening pack (samples/tests/2026-08-19_sd-int8-x86/) compared fp32,
         * int8 blk=64 and int8 blk=256 on three cells; the verdict on the worst cell
         * by SNR - vivian, 36.5 dB - was "no audible noise". It buys RTF 1.19-1.23 ->
         * 1.10-1.14 on the 0.6B. QWEN_SD_INT8=0 goes back to fp32 sgemm.
         * ARM stays opt-in: there the block that is fast (64) is a different point on
         * the same curve, and it has its own ear history. */
        en = qwen_sd_int8_available() && !(e && *e == '0');
#else
        en = (e && *e && *e != '0') && qwen_sd_int8_available();
#endif
    }
    return en;
}


/* ── SD PHASE ATTRIBUTION (QWEN_SD_PHASE=1) ───────────────────────────────────
 * Measurement only, and OFF by default: when the env is unset every macro below
 * compiles to a branch on one cached int and nothing else runs, so the production
 * path is untouched.
 *
 * The point of the design is that it can FAIL. Each phase is timed with its own
 * start/stop, and the WHOLE call is timed separately, so `total - sum(phases)` is
 * printed as `unacc` rather than being absorbed into the last phase. A phase
 * scheme that does not reconcile says so in its own output.
 *
 * The two decode entry points have the same six steps and are instrumented the
 * same way; the line records which one ran, because they are different code:
 * `qwen_speech_decoder_decode_streaming_batch` hands any single-item call to the
 * per-slot function, so group=1 and group>=2 are not the same implementation.
 *
 * Single-threaded by construction: the driver calls decode inline from the loop,
 * one thread per worker process, so the file-scope accumulators for the step-6
 * sub-split are not shared. */
static int sd_phase_on(void) {
    static int v = -1;
    if (v < 0) { const char *e = getenv("QWEN_SD_PHASE"); v = (e && *e && *e != '0'); }
    return v;
}
static double sd_ph_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}
/* step-6 sub-split, filled by conv_decoder_forward_streaming[_batch] */
static double sd_p6a, sd_p6b, sd_p6c;


/* One line per decode call. `unacc` is total minus the six phases: it is the part
 * the phase scheme does NOT explain, and it is printed rather than hidden. */
#define SD_PHASE_EMIT(path_, group_, frames_) do {                                   \
    if (_ph) {                                                                       \
        _ph_p[5] += sd_ph_now() - _ph_mark;                                          \
        double _tot = sd_ph_now() - _ph_call0;                                       \
        double _sum = _ph_p[0]+_ph_p[1]+_ph_p[2]+_ph_p[3]+_ph_p[4]+_ph_p[5];         \
        fprintf(stderr, "[SDPHASE] v=1 pid=%d path=%s group=%d frames=%d "           \
                "clock=CLOCK_MONOTONIC domain=S vq=%.3f preconv=%.3f inproj=%.3f "   \
                "pretf=%.3f outproj=%.3f conv=%.3f conv_cnext=%.3f conv_init=%.3f "  \
                "conv_up=%.3f sum=%.3f total=%.3f unacc=%.3f\n",                     \
                (int)getpid(), (path_), (group_), (frames_),                         \
                _ph_p[0], _ph_p[1], _ph_p[2], _ph_p[3], _ph_p[4], _ph_p[5],          \
                sd_p6a, sd_p6b, sd_p6c, _sum, _tot, _tot - _sum);                    \
    }                                                                                \
} while (0)

/* Scale-block size along K (multiple of 16). Smaller = more accurate, slightly
 * slower; the author measured 64 as the sweet spot on N1 (same RTF as 128,
 * ~4 dB better SNR). QWEN_SD_INT8_BLK overrides. */
static int sd_int8_blk(void) {
    static int blk = -1;
    if (blk < 0) {
        const char *e = getenv("QWEN_SD_INT8_BLK");
#if defined(__AVX512VNNI__)
        /* x86 wants a BIGGER block than ARM, and the reason is the instruction mix, not
         * the numerics. Per scale block each tile pays a fixed f32 epilogue (convert +
         * FMA per accumulator). On NEON a 64-element block is 32 SDOTs against 16 f32
         * ops — 2:1. On a 512-bit VNNI lane the same block is EIGHT dpbusd against the
         * same 16 f32 ops — 1:2, and the epilogue dominates. Measured on the c3 (0.6B,
         * -j4, same utterance), decoder ms / RTF:
         *     blk=64  6923 / 1.22   blk=128 6305 / 1.15
         *     blk=256 5797 / 1.10   blk=512 5826 / 1.10      (fp32 sgemm: 5897 / 1.19)
         * 256 is the knee. ⚠️ Coarser blocks mean coarser scales: this trades SNR for
         * speed on top of the trade int8 already makes, which is why the whole path
         * stays opt-in and must pass the ear before it is ever a default. */
        blk = e ? atoi(e) : 256;
#else
        blk = e ? atoi(e) : 64;
#endif
        if (blk < 16) blk = 16;
        blk = (blk + 15) & ~15;
    }
    return blk;
}

/* Quantized-weight registry keyed by source pointer. Quantize-once; the entries
 * live until process exit (bounded at SD_WQ_MAX, and only populated when the
 * env flag is on). Weights are read-only and shared across streaming slots. */
typedef struct {
    const float *src;
    int8_t *q;
    float *scales;
    int32_t *wsum;      /* [out_ch][Kp/blk] sums of the quantized bytes — the x86 VNNI
                         * correction, precomputed here because it is a property of the
                         * weights and must never enter the inner loop. */
    int Kp;
} sd_wq_entry_t;

#define SD_WQ_MAX 128
static sd_wq_entry_t sd_wq[SD_WQ_MAX];
static int sd_wq_n = 0;
static pthread_mutex_t sd_wq_mu = PTHREAD_MUTEX_INITIALIZER;

/* Conv1d weight [out_ch, in_ch*kernel]: rows are already in im2col order. */
static sd_wq_entry_t *sd_wq_get_conv(const float *w, int out_ch, int K) {
    pthread_mutex_lock(&sd_wq_mu);
    for (int i = 0; i < sd_wq_n; i++)
        if (sd_wq[i].src == w) { pthread_mutex_unlock(&sd_wq_mu); return &sd_wq[i]; }
    if (sd_wq_n >= SD_WQ_MAX) { pthread_mutex_unlock(&sd_wq_mu); return NULL; }
    int blk = sd_int8_blk();
    int Kp = qwen_int8_kp(K, blk);
    sd_wq_entry_t *e = &sd_wq[sd_wq_n];
    int nblk = Kp / blk;
    e->q = (int8_t *)aligned_malloc((size_t)out_ch * Kp);
    e->scales = (float *)aligned_malloc((size_t)out_ch * nblk * sizeof(float));
    e->wsum = (int32_t *)aligned_malloc((size_t)out_ch * nblk * sizeof(int32_t));
    if (!e->q || !e->scales || !e->wsum) {
        free(e->q); free(e->scales); free(e->wsum);
        pthread_mutex_unlock(&sd_wq_mu); return NULL;
    }
    e->src = w; e->Kp = Kp;
    qwen_int8_quant_rows(e->q, e->scales, w, out_ch, K, Kp, blk);
    for (int r = 0; r < out_ch; r++) {
        const int8_t *row = e->q + (size_t)r * Kp;
        int32_t *ws = e->wsum + (size_t)r * nblk;
        for (int b = 0; b < nblk; b++) {
            int32_t acc = 0;
            for (int k = b * blk; k < (b + 1) * blk; k++) acc += (int32_t)row[k];
            ws[b] = acc;
        }
    }
    sd_wq_n++;
    pthread_mutex_unlock(&sd_wq_mu);
    return e;
}

#ifdef USE_BLAS
/* Add bias to channel-first output [channels, length] */
static void conv_add_bias(float *out, const float *bias, int channels, int length) {
    if (!bias) return;
    for (int c = 0; c < channels; c++) {
        float b = bias[c];
        float *row = out + (int64_t)c * length;
        for (int t = 0; t < length; t++)
            row[t] += b;
    }
}

/* BLAS causal Conv1d: im2col + sgemm (k>1), direct sgemm (k=1) */
static void causal_conv1d_blas(float *out, const float *in,
                               const float *weight, const float *bias,
                               int in_ch, int out_ch, int length,
                               int kernel, int dilation) {
    /* int8 only for the upsample-block res convs: the latent-domain convs
     * (pre-conv 512->1024, ConvNeXt) are quality-sensitive and cheap, so they
     * stay fp32. Same scoping as the author's. */
    if (sd_int8_enabled() && in_ch == out_ch && in_ch <= 768) {
        sd_wq_entry_t *e = sd_wq_get_conv(weight, out_ch, in_ch * kernel);
        if (e) {
            qwen_conv1d_int8(out, in, e->q, e->scales, e->wsum, bias,
                             in_ch, out_ch, length, kernel, dilation,
                             e->Kp, sd_int8_blk());
            return;
        }
    }

    if (kernel == 1) {
        /* k=1: weight is [out_ch, in_ch], direct matmul */
        SD_GEMM(CblasNoTrans, CblasNoTrans,
                    out_ch, length, in_ch,
                    1.0f, weight, in_ch,
                    in, length,
                    0.0f, out, length);
        conv_add_bias(out, bias, out_ch, length);
        return;
    }

    /* im2col + sgemm for k>1 */
    int pad_left = (kernel - 1) * dilation;
    int64_t col_rows = (int64_t)in_ch * kernel;

    /* Tile along time if im2col buffer would exceed limit */
    int64_t max_tile = CONV_TILE_MAX_BYTES / (col_rows * (int64_t)sizeof(float));
    if (max_tile < 1) max_tile = 1;
    if (max_tile > length) max_tile = length;

    float *col = (float *)aligned_malloc(col_rows * max_tile * sizeof(float));

    for (int ts = 0; ts < length; ts += (int)max_tile) {
        int tile = ((int64_t)ts + max_tile > length) ? length - ts : (int)max_tile;

        /* Build im2col: col[in_ch*kernel, tile] */
        memset(col, 0, col_rows * tile * sizeof(float));
        for (int ic = 0; ic < in_ch; ic++) {
            for (int k = 0; k < kernel; k++) {
                float *col_row = col + ((int64_t)ic * kernel + k) * tile;
                for (int t = 0; t < tile; t++) {
                    int in_pos = (t + ts) - pad_left + k * dilation;
                    if (in_pos >= 0 && in_pos < length)
                        col_row[t] = in[(int64_t)ic * length + in_pos];
                }
            }
        }

        /* sgemm: out_tile = weight[out_ch, col_rows] × col[col_rows, tile] */
        SD_GEMM(CblasNoTrans, CblasNoTrans,
                    out_ch, tile, (int)col_rows,
                    1.0f, weight, (int)col_rows,
                    col, tile,
                    0.0f, out + ts, length);
    }

    free(col);
    conv_add_bias(out, bias, out_ch, length);
}

/* BLAS causal ConvTranspose1d: per-kernel sgemm + scatter */
static void causal_conv_transpose1d_blas(float *out, const float *in,
                                         const float *weight, const float *bias,
                                         int in_ch, int out_ch, int in_len, int out_len,
                                         int kernel, int stride) {
    memset(out, 0, (int64_t)out_ch * out_len * sizeof(float));
    /* Right trim is carried by out_len alone — see causal_conv_transpose1d_naive. */

    /* Per-kernel-position: extract weight slice, sgemm, scatter */
    float *wk = (float *)aligned_malloc((int64_t)in_ch * out_ch * sizeof(float));
    float *rk = (float *)aligned_malloc((int64_t)out_ch * in_len * sizeof(float));

    for (int k = 0; k < kernel; k++) {
        /* Extract W_k[in_ch, out_ch] from weight[in_ch, out_ch, kernel] */
        for (int ic = 0; ic < in_ch; ic++)
            for (int oc = 0; oc < out_ch; oc++)
                wk[(int64_t)ic * out_ch + oc] =
                    weight[((int64_t)ic * out_ch + oc) * kernel + k];

        /* rk[out_ch, in_len] = W_k^T[out_ch, in_ch] × in[in_ch, in_len] */
        SD_GEMM(CblasTrans, CblasNoTrans,
                    out_ch, in_len, in_ch,
                    1.0f, wk, out_ch,
                    in, in_len,
                    0.0f, rk, in_len);

        /* Scatter to strided output positions */
        for (int oc = 0; oc < out_ch; oc++) {
            const float *src = rk + (int64_t)oc * in_len;
            float *dst = out + (int64_t)oc * out_len;
            for (int t = 0; t < in_len; t++) {
                int out_pos = t * stride + k;
                if (out_pos < out_len)
                    dst[out_pos] += src[t];
            }
        }
    }

    free(wk);
    free(rk);
    conv_add_bias(out, bias, out_ch, out_len);
}
#endif /* USE_BLAS */

/* Dispatch wrappers */
static void causal_conv1d(float *out, const float *in,
                           const float *weight, const float *bias,
                           int in_ch, int out_ch, int length,
                           int kernel, int dilation) {
#ifdef USE_BLAS
    causal_conv1d_blas(out, in, weight, bias, in_ch, out_ch, length, kernel, dilation);
#else
    causal_conv1d_naive(out, in, weight, bias, in_ch, out_ch, length, kernel, dilation);
#endif
}

static void causal_conv_transpose1d(float *out, const float *in,
                                     const float *weight, const float *bias,
                                     int in_ch, int out_ch, int in_len, int out_len,
                                     int kernel, int stride) {
#ifdef USE_BLAS
    causal_conv_transpose1d_blas(out, in, weight, bias, in_ch, out_ch, in_len, out_len, kernel, stride);
#else
    causal_conv_transpose1d_naive(out, in, weight, bias, in_ch, out_ch, in_len, out_len, kernel, stride);
#endif
}

/* ========================================================================
 * Weight Loading
 * ======================================================================== */

int qwen_speech_decoder_load(qwen_tts_ctx_t *ctx) {
    qwen_tts_config_t *c = &ctx->config;
    void *ms = ctx->speech_safetensors;
    qwen_speech_decoder_t *sd = &ctx->speech_dec;

    if (!ctx->silent) fprintf(stderr, "Loading Speech Decoder weights...\n");

    int cb_dim = QWEN_TTS_CODEBOOK_DIM;
    int cb_size = c->codebook_size;

    /* Codebook 0 (rvq_first) - dequantize from EMA */
    const float *emb_sum = get_f32(ms, "decoder.quantizer.rvq_first.vq.layers.0._codebook.embedding_sum");
    const float *usage = get_f32(ms, "decoder.quantizer.rvq_first.vq.layers.0._codebook.cluster_usage");
    if (emb_sum && usage) {
        sd->codebook[0] = (float *)aligned_malloc((int64_t)cb_size * cb_dim * sizeof(float));
        for (int i = 0; i < cb_size; i++)
            for (int d = 0; d < cb_dim; d++)
                sd->codebook[0][(int64_t)i * cb_dim + d] = emb_sum[(int64_t)i * cb_dim + d] / fmaxf(usage[i], 1e-5f);
    }

    /* Codebooks 1-15 (rvq_rest) */
    for (int k = 0; k < 15; k++) {
        char es_name[128], cu_name[128];
        snprintf(es_name, sizeof(es_name), "decoder.quantizer.rvq_rest.vq.layers.%d._codebook.embedding_sum", k);
        snprintf(cu_name, sizeof(cu_name), "decoder.quantizer.rvq_rest.vq.layers.%d._codebook.cluster_usage", k);
        emb_sum = get_f32(ms, es_name);
        usage = get_f32(ms, cu_name);
        if (emb_sum && usage) {
            sd->codebook[k + 1] = (float *)aligned_malloc((int64_t)cb_size * cb_dim * sizeof(float));
            for (int i = 0; i < cb_size; i++)
                for (int d = 0; d < cb_dim; d++)
                    sd->codebook[k + 1][(int64_t)i * cb_dim + d] = emb_sum[(int64_t)i * cb_dim + d] / fmaxf(usage[i], 1e-5f);
        }
    }

    /* VQ projections */
    sd->rvq_first_output_proj = get_f32(ms, "decoder.quantizer.rvq_first.output_proj.weight");
    sd->rvq_rest_output_proj = get_f32(ms, "decoder.quantizer.rvq_rest.output_proj.weight");

    /* Pre-conv */
    sd->pre_conv_weight = get_f32(ms, "decoder.pre_conv.conv.weight");
    sd->pre_conv_bias = get_f32(ms, "decoder.pre_conv.conv.bias");

    /* Pre-transformer */
    sd->input_proj_weight = get_f32(ms, "decoder.pre_transformer.input_proj.weight");
    sd->input_proj_bias = get_f32(ms, "decoder.pre_transformer.input_proj.bias");
    sd->final_norm_weight = get_f32(ms, "decoder.pre_transformer.norm.weight");
    sd->output_proj_weight = get_f32(ms, "decoder.pre_transformer.output_proj.weight");
    sd->output_proj_bias = get_f32(ms, "decoder.pre_transformer.output_proj.bias");

    sd->pre_layers = (qwen_sd_pre_layer_t *)calloc(c->dec_num_layers, sizeof(qwen_sd_pre_layer_t));
    for (int i = 0; i < c->dec_num_layers; i++) {
        qwen_sd_pre_layer_t *l = &sd->pre_layers[i];
        char name[128];
        snprintf(name, sizeof(name), "decoder.pre_transformer.layers.%d.input_layernorm.weight", i);
        l->attn_norm = get_f32(ms, name);
        snprintf(name, sizeof(name), "decoder.pre_transformer.layers.%d.self_attn.q_proj.weight", i);
        l->attn_q = get_f32(ms, name);
        snprintf(name, sizeof(name), "decoder.pre_transformer.layers.%d.self_attn.k_proj.weight", i);
        l->attn_k = get_f32(ms, name);
        snprintf(name, sizeof(name), "decoder.pre_transformer.layers.%d.self_attn.v_proj.weight", i);
        l->attn_v = get_f32(ms, name);
        snprintf(name, sizeof(name), "decoder.pre_transformer.layers.%d.self_attn.o_proj.weight", i);
        l->attn_o = get_f32(ms, name);
        snprintf(name, sizeof(name), "decoder.pre_transformer.layers.%d.self_attn_layer_scale.scale", i);
        l->attn_layer_scale = get_f32(ms, name);
        snprintf(name, sizeof(name), "decoder.pre_transformer.layers.%d.post_attention_layernorm.weight", i);
        l->ffn_norm = get_f32(ms, name);
        snprintf(name, sizeof(name), "decoder.pre_transformer.layers.%d.mlp.gate_proj.weight", i);
        l->ffn_gate = get_f32(ms, name);
        snprintf(name, sizeof(name), "decoder.pre_transformer.layers.%d.mlp.up_proj.weight", i);
        l->ffn_up = get_f32(ms, name);
        snprintf(name, sizeof(name), "decoder.pre_transformer.layers.%d.mlp.down_proj.weight", i);
        l->ffn_down = get_f32(ms, name);
        snprintf(name, sizeof(name), "decoder.pre_transformer.layers.%d.mlp_layer_scale.scale", i);
        l->ffn_layer_scale = get_f32(ms, name);
    }

    /* RoPE cache for pre-transformer (NeoX split-half) */
    int half_dim = c->dec_head_dim / 2;
    sd->rope_cos = (float *)aligned_malloc(8000 * half_dim * sizeof(float));
    sd->rope_sin = (float *)aligned_malloc(8000 * half_dim * sizeof(float));
    for (int pos = 0; pos < 8000; pos++) {
        for (int i = 0; i < half_dim; i++) {
            float angle = pos / powf(c->dec_rope_theta, (float)(2*i) / c->dec_head_dim);
            sd->rope_cos[pos * half_dim + i] = cosf(angle);
            sd->rope_sin[pos * half_dim + i] = sinf(angle);
        }
    }

    /* ConvNeXt upsample blocks (2 blocks) */
    for (int b = 0; b < 2; b++) {
        qwen_sd_convnext_t *cn = &sd->convnext[b];
        char name[128];
        /* ConvTranspose1d is sub-layer 0 */
        snprintf(name, sizeof(name), "decoder.upsample.%d.0.conv.weight", b);
        cn->conv_weight = get_f32(ms, name);
        snprintf(name, sizeof(name), "decoder.upsample.%d.0.conv.bias", b);
        cn->conv_bias = get_f32(ms, name);
        /* ConvNeXt block is sub-layer 1 */
        snprintf(name, sizeof(name), "decoder.upsample.%d.1.dwconv.conv.weight", b);
        cn->dwconv_weight = get_f32(ms, name);
        snprintf(name, sizeof(name), "decoder.upsample.%d.1.dwconv.conv.bias", b);
        cn->dwconv_bias = get_f32(ms, name);
        snprintf(name, sizeof(name), "decoder.upsample.%d.1.pwconv1.weight", b);
        cn->pwconv1_weight = get_f32(ms, name);
        snprintf(name, sizeof(name), "decoder.upsample.%d.1.pwconv1.bias", b);
        cn->pwconv1_bias = get_f32(ms, name);
        snprintf(name, sizeof(name), "decoder.upsample.%d.1.pwconv2.weight", b);
        cn->pwconv2_weight = get_f32(ms, name);
        snprintf(name, sizeof(name), "decoder.upsample.%d.1.pwconv2.bias", b);
        cn->pwconv2_bias = get_f32(ms, name);
        snprintf(name, sizeof(name), "decoder.upsample.%d.1.norm.weight", b);
        cn->norm_weight = get_f32(ms, name);
        snprintf(name, sizeof(name), "decoder.upsample.%d.1.norm.bias", b);
        cn->norm_bias = get_f32(ms, name);
        snprintf(name, sizeof(name), "decoder.upsample.%d.1.gamma", b);
        cn->gamma = get_f32(ms, name);
    }

    /* Initial conv: decoder.decoder.0 */
    sd->initial_conv_weight = get_f32(ms, "decoder.decoder.0.conv.weight");
    sd->initial_conv_bias = get_f32(ms, "decoder.decoder.0.conv.bias");

    /* Decoder upsample blocks: decoder.decoder.{1-4} */
    for (int b = 0; b < 4; b++) {
        qwen_sd_upsample_block_t *ub = &sd->upsample_blocks[b];
        int bi = b + 1; /* tensor index: 1-4 */
        char name[128];

        /* Snake before upsample: block.0.{alpha,beta} */
        snprintf(name, sizeof(name), "decoder.decoder.%d.block.0.alpha", bi);
        ub->upsample.snake_alpha = get_f32(ms, name);
        snprintf(name, sizeof(name), "decoder.decoder.%d.block.0.beta", bi);
        ub->upsample.snake_beta = get_f32(ms, name);

        /* ConvTranspose upsample: block.1.conv.{weight,bias} */
        snprintf(name, sizeof(name), "decoder.decoder.%d.block.1.conv.weight", bi);
        ub->upsample.conv_weight = get_f32(ms, name);
        snprintf(name, sizeof(name), "decoder.decoder.%d.block.1.conv.bias", bi);
        ub->upsample.conv_bias = get_f32(ms, name);

        /* 3 residual blocks: block.{2,3,4} */
        for (int r = 0; r < 3; r++) {
            int ri = r + 2; /* tensor index: 2,3,4 */
            snprintf(name, sizeof(name), "decoder.decoder.%d.block.%d.act1.alpha", bi, ri);
            ub->res_blocks[r].snake1_alpha = get_f32(ms, name);
            snprintf(name, sizeof(name), "decoder.decoder.%d.block.%d.act1.beta", bi, ri);
            ub->res_blocks[r].snake1_beta = get_f32(ms, name);
            snprintf(name, sizeof(name), "decoder.decoder.%d.block.%d.conv1.conv.weight", bi, ri);
            ub->res_blocks[r].conv1_weight = get_f32(ms, name);
            snprintf(name, sizeof(name), "decoder.decoder.%d.block.%d.conv1.conv.bias", bi, ri);
            ub->res_blocks[r].conv1_bias = get_f32(ms, name);
            snprintf(name, sizeof(name), "decoder.decoder.%d.block.%d.act2.alpha", bi, ri);
            ub->res_blocks[r].snake2_alpha = get_f32(ms, name);
            snprintf(name, sizeof(name), "decoder.decoder.%d.block.%d.act2.beta", bi, ri);
            ub->res_blocks[r].snake2_beta = get_f32(ms, name);
            snprintf(name, sizeof(name), "decoder.decoder.%d.block.%d.conv2.conv.weight", bi, ri);
            ub->res_blocks[r].conv2_weight = get_f32(ms, name);
            snprintf(name, sizeof(name), "decoder.decoder.%d.block.%d.conv2.conv.bias", bi, ri);
            ub->res_blocks[r].conv2_bias = get_f32(ms, name);
        }
    }

    /* Final snake: decoder.decoder.5 */
    sd->final_snake.alpha = get_f32(ms, "decoder.decoder.5.alpha");
    sd->final_snake.beta = get_f32(ms, "decoder.decoder.5.beta");

    /* Final conv: decoder.decoder.6 */
    sd->final_conv_weight = get_f32(ms, "decoder.decoder.6.conv.weight");
    sd->final_conv_bias = get_f32(ms, "decoder.decoder.6.conv.bias");

    /* Debug: verify pre_conv weights right after loading */
    if (sd->pre_conv_weight) {
        fprintf(stderr, "  [LOAD] pre_conv_w[:5]: [%.6f, %.6f, %.6f, %.6f, %.6f] bias[:3]: [%.6f, %.6f, %.6f]\n",
                sd->pre_conv_weight[0], sd->pre_conv_weight[1], sd->pre_conv_weight[2],
                sd->pre_conv_weight[3], sd->pre_conv_weight[4],
                sd->pre_conv_bias[0], sd->pre_conv_bias[1], sd->pre_conv_bias[2]);
    } else {
        fprintf(stderr, "  [LOAD] pre_conv_weight is NULL!\n");
    }

    if (!ctx->silent) {
        fprintf(stderr, "  Codebooks: 16/16 (dequantized from EMA)\n");
        fprintf(stderr, "  Pre-transformer: %d layers, input_proj=%s\n",
                c->dec_num_layers, sd->input_proj_weight ? "ok" : "MISSING");
        fprintf(stderr, "  ConvNeXt upsample: %s\n",
                (sd->convnext[0].conv_weight && sd->convnext[1].conv_weight) ? "ok" : "MISSING");
        fprintf(stderr, "  Conv decoder: initial=%s, final=%s\n",
                sd->initial_conv_weight ? "ok" : "MISSING",
                sd->final_conv_weight ? "ok" : "MISSING");
        fprintf(stderr, "  Upsample blocks: [%s %s %s %s]\n",
                sd->upsample_blocks[0].upsample.conv_weight ? "ok" : "MISSING",
                sd->upsample_blocks[1].upsample.conv_weight ? "ok" : "MISSING",
                sd->upsample_blocks[2].upsample.conv_weight ? "ok" : "MISSING",
                sd->upsample_blocks[3].upsample.conv_weight ? "ok" : "MISSING");
        fprintf(stderr, "  Final snake: %s\n",
                sd->final_snake.alpha ? "ok" : "MISSING");
    }

    return 0;
}

/* ========================================================================
 * Decode: codes → audio
 * ======================================================================== */

int qwen_speech_decoder_decode(qwen_tts_ctx_t *ctx, const int *codes, int n_frames,
                                float **audio_out, int *n_samples) {
    qwen_mm_component(QWEN_COMP_DECODER);   /* MAC audit: attribute this step's kernels */
    qwen_speech_decoder_t *sd = &ctx->speech_dec;
    qwen_tts_config_t *c = &ctx->config;

    int cb_dim = QWEN_TTS_CODEBOOK_DIM;
    int vq_hidden = 512;
    int latent_dim = 1024;

    if (!ctx->silent)
        fprintf(stderr, "  Speech decoder: %d frames -> audio...\n", n_frames);

    /* Debug: check if weights are still intact */
    if (ctx->debug && sd->pre_conv_weight) {
        fprintf(stderr, "[DECODER] ENTRY pre_conv_w[:5]: [%.6f, %.6f, %.6f, %.6f, %.6f]\n",
                sd->pre_conv_weight[0], sd->pre_conv_weight[1], sd->pre_conv_weight[2],
                sd->pre_conv_weight[3], sd->pre_conv_weight[4]);
    }

    /* Step 1: VQ dequant + output projection (batched with BLAS) */
    float *vq_out = (float *)aligned_calloc((int64_t)n_frames * vq_hidden, sizeof(float));

    /* Gather codebook embeddings into matrices for batched projection */
    float *emb_first = (float *)aligned_malloc((int64_t)n_frames * cb_dim * sizeof(float));
    float *emb_rest = (float *)aligned_calloc((int64_t)n_frames * cb_dim, sizeof(float));

    for (int f = 0; f < n_frames; f++) {
        /* Codebook 0 (rvq_first) */
        int code0 = codes[f * 16];
        if (code0 >= 0 && code0 < c->codebook_size && sd->codebook[0]) {
            memcpy(emb_first + (int64_t)f * cb_dim,
                   sd->codebook[0] + (int64_t)code0 * cb_dim, cb_dim * sizeof(float));
        } else {
            memset(emb_first + (int64_t)f * cb_dim, 0, cb_dim * sizeof(float));
        }

        /* Codebooks 1-15 (rvq_rest): sum embeddings */
        float *rest_row = emb_rest + (int64_t)f * cb_dim;
        for (int k = 1; k < 16; k++) {
            int code = codes[f * 16 + k];
            if (code >= 0 && code < c->codebook_size && sd->codebook[k]) {
                const float *emb = sd->codebook[k] + (int64_t)code * cb_dim;
                for (int d = 0; d < cb_dim; d++) rest_row[d] += emb[d];
            }
        }
    }

    /* Batched projection: vq_out[n_frames, vq_hidden] = emb[n_frames, cb_dim] × W^T[cb_dim, vq_hidden] */
#ifdef USE_BLAS
    if (sd->rvq_first_output_proj) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    n_frames, vq_hidden, cb_dim, 1.0f,
                    emb_first, cb_dim, sd->rvq_first_output_proj, cb_dim,
                    0.0f, vq_out, vq_hidden);
    }
    if (sd->rvq_rest_output_proj) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    n_frames, vq_hidden, cb_dim, 1.0f,
                    emb_rest, cb_dim, sd->rvq_rest_output_proj, cb_dim,
                    1.0f, vq_out, vq_hidden);  /* accumulate */
    }
#else
    for (int f = 0; f < n_frames; f++) {
        if (sd->rvq_first_output_proj) {
            const float *emb = emb_first + (int64_t)f * cb_dim;
            for (int o = 0; o < vq_hidden; o++) {
                float sum = 0;
                for (int i = 0; i < cb_dim; i++)
                    sum += sd->rvq_first_output_proj[(int64_t)o * cb_dim + i] * emb[i];
                vq_out[(int64_t)f * vq_hidden + o] += sum;
            }
        }
        if (sd->rvq_rest_output_proj) {
            const float *rest = emb_rest + (int64_t)f * cb_dim;
            for (int o = 0; o < vq_hidden; o++) {
                float sum = 0;
                for (int i = 0; i < cb_dim; i++)
                    sum += sd->rvq_rest_output_proj[(int64_t)o * cb_dim + i] * rest[i];
                vq_out[(int64_t)f * vq_hidden + o] += sum;
            }
        }
    }
#endif
    free(emb_first); free(emb_rest);

    /* Debug: dump first frame's RVQ output */
    if (ctx->debug) {
        float rms0 = 0;
        for (int i = 0; i < vq_hidden; i++) rms0 += vq_out[i] * vq_out[i];
        rms0 = sqrtf(rms0 / vq_hidden);
        fprintf(stderr, "[DECODER] RVQ out frame 0 [:5]: [%.6f, %.6f, %.6f, %.6f, %.6f] RMS=%.6f\n",
                vq_out[0], vq_out[1], vq_out[2], vq_out[3], vq_out[4], rms0);
        /* Also check the transposed buffer */
    }

    /* Step 2: Pre-conv (512→1024, k=3, causal, pad_left=2) */
    /* Need channel-first format for conv: [vq_hidden, n_frames] */
    float *vq_cf = (float *)aligned_malloc((int64_t)vq_hidden * n_frames * sizeof(float));
    for (int f = 0; f < n_frames; f++)
        for (int d = 0; d < vq_hidden; d++)
            vq_cf[(int64_t)d * n_frames + f] = vq_out[(int64_t)f * vq_hidden + d];
    free(vq_out);

    /* Debug: check transposed buffer */
    if (ctx->debug) {
        float rms_cf = 0;
        for (int d = 0; d < vq_hidden; d++) {
            float v = vq_cf[(int64_t)d * n_frames + 0]; /* frame 0, channel d */
            rms_cf += v * v;
        }
        rms_cf = sqrtf(rms_cf / vq_hidden);
        fprintf(stderr, "[DECODER] vq_cf frame 0 ch[:5]: [%.6f, %.6f, %.6f, %.6f, %.6f] RMS=%.6f\n",
                vq_cf[0 * n_frames + 0], vq_cf[1 * n_frames + 0],
                vq_cf[2 * n_frames + 0], vq_cf[3 * n_frames + 0],
                vq_cf[4 * n_frames + 0], rms_cf);
        /* Check pre_conv weight */
        fprintf(stderr, "[DECODER] pre_conv_w[:5]: [%.6f, %.6f, %.6f, %.6f, %.6f] bias[:3]: [%.6f, %.6f, %.6f]\n",
                sd->pre_conv_weight[0], sd->pre_conv_weight[1], sd->pre_conv_weight[2],
                sd->pre_conv_weight[3], sd->pre_conv_weight[4],
                sd->pre_conv_bias[0], sd->pre_conv_bias[1], sd->pre_conv_bias[2]);
    }

    float *pre_conv_out = (float *)aligned_calloc((int64_t)latent_dim * n_frames, sizeof(float));
    causal_conv1d(pre_conv_out, vq_cf, sd->pre_conv_weight, sd->pre_conv_bias,
                  vq_hidden, latent_dim, n_frames, 3, 1);
    free(vq_cf);

    /* Debug: dump pre_conv output */
    if (ctx->debug) {
        fprintf(stderr, "[DECODER] pre_conv out frame 0 [:5]: [%.6f, %.6f, %.6f, %.6f, %.6f]\n",
                pre_conv_out[0 * n_frames + 0], pre_conv_out[1 * n_frames + 0],
                pre_conv_out[2 * n_frames + 0], pre_conv_out[3 * n_frames + 0],
                pre_conv_out[4 * n_frames + 0]);
    }

    /* Step 3: Transpose to row-major + input_proj (1024→512) */
    int dec_hidden = 512;
    float *hidden = (float *)aligned_malloc((int64_t)n_frames * dec_hidden * sizeof(float));
#ifdef USE_BLAS
    /* Transpose pre_conv_out from channel-first [1024, n_frames] to row-major [n_frames, 1024] */
    float *pre_conv_rm = (float *)aligned_malloc((int64_t)n_frames * latent_dim * sizeof(float));
    for (int f = 0; f < n_frames; f++)
        for (int d = 0; d < latent_dim; d++)
            pre_conv_rm[(int64_t)f * latent_dim + d] = pre_conv_out[(int64_t)d * n_frames + f];
    free(pre_conv_out);
    /* hidden[n_frames, 512] = pre_conv_rm[n_frames, 1024] × W^T[1024, 512] */
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                n_frames, dec_hidden, latent_dim, 1.0f,
                pre_conv_rm, latent_dim,
                sd->input_proj_weight, latent_dim,
                0.0f, hidden, dec_hidden);
    free(pre_conv_rm);
    if (sd->input_proj_bias) {
        for (int f = 0; f < n_frames; f++)
            for (int o = 0; o < dec_hidden; o++)
                hidden[(int64_t)f * dec_hidden + o] += sd->input_proj_bias[o];
    }
#else
    for (int f = 0; f < n_frames; f++) {
        for (int o = 0; o < dec_hidden; o++) {
            float sum = sd->input_proj_bias ? sd->input_proj_bias[o] : 0;
            for (int i = 0; i < latent_dim; i++)
                sum += sd->input_proj_weight[(int64_t)o * latent_dim + i] * pre_conv_out[(int64_t)i * n_frames + f];
            hidden[(int64_t)f * dec_hidden + o] = sum;
        }
    }
    free(pre_conv_out);
#endif

    /* Debug: dump input_proj output */
    if (ctx->debug) {
        fprintf(stderr, "[DECODER] input_proj out frame 0 [:5]: [%.6f, %.6f, %.6f, %.6f, %.6f]\n",
                hidden[0], hidden[1], hidden[2], hidden[3], hidden[4]);
    }

    /* Step 4: Pre-transformer (8 layers with sliding window causal attention) */
    int dec_inter = 1024;
    int n_heads = 16;
    int head_dim = c->dec_head_dim; /* 64 */
    int qkv_dim = n_heads * head_dim;
    int window = 72;
    float eps = c->dec_rms_norm_eps;
    int half_hd = head_dim / 2;

    float *q = (float *)aligned_malloc((int64_t)n_frames * qkv_dim * sizeof(float));
    float *kk = (float *)aligned_malloc((int64_t)n_frames * qkv_dim * sizeof(float));
    float *vv = (float *)aligned_malloc((int64_t)n_frames * qkv_dim * sizeof(float));
    float *x_norm = (float *)aligned_malloc((int64_t)n_frames * dec_hidden * sizeof(float));
    float *attn_out = (float *)aligned_malloc((int64_t)n_frames * qkv_dim * sizeof(float));

    for (int layer = 0; layer < c->dec_num_layers; layer++) {
        qwen_sd_pre_layer_t *l = &sd->pre_layers[layer];

        /* Input RMSNorm (NEON-optimized) */
        qwen_rms_norm(x_norm, hidden, l->attn_norm, n_frames, dec_hidden, eps);

        /* Debug after input RMSNorm for layer 0 */
        if (ctx->debug && layer == 0) {
            fprintf(stderr, "[DECODER] Layer 0 input_norm frame 0 [:5]: [%.6f, %.6f, %.6f, %.6f, %.6f]\n",
                    x_norm[0], x_norm[1], x_norm[2], x_norm[3], x_norm[4]);
        }

        /* QKV projections */
#ifdef USE_BLAS
        /* x_norm[n_frames, dec_hidden] × W^T = out[n_frames, qkv_dim] */
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    n_frames, qkv_dim, dec_hidden, 1.0f,
                    x_norm, dec_hidden, l->attn_q, dec_hidden, 0.0f, q, qkv_dim);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    n_frames, qkv_dim, dec_hidden, 1.0f,
                    x_norm, dec_hidden, l->attn_k, dec_hidden, 0.0f, kk, qkv_dim);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    n_frames, qkv_dim, dec_hidden, 1.0f,
                    x_norm, dec_hidden, l->attn_v, dec_hidden, 0.0f, vv, qkv_dim);
#else
        for (int s = 0; s < n_frames; s++) {
            const float *xs = x_norm + s * dec_hidden;
            float *qs = q + s * qkv_dim;
            float *ks = kk + s * qkv_dim;
            float *vs = vv + s * qkv_dim;
            for (int o = 0; o < qkv_dim; o++) {
                float sum_q = 0, sum_k = 0, sum_v = 0;
                for (int i = 0; i < dec_hidden; i++) {
                    sum_q += l->attn_q[(int64_t)o * dec_hidden + i] * xs[i];
                    sum_k += l->attn_k[(int64_t)o * dec_hidden + i] * xs[i];
                    sum_v += l->attn_v[(int64_t)o * dec_hidden + i] * xs[i];
                }
                qs[o] = sum_q; ks[o] = sum_k; vs[o] = sum_v;
            }
        }
#endif

        /* Debug QKV for layer 0 */
        if (ctx->debug && layer == 0) {
            fprintf(stderr, "[DECODER] Layer 0 Q frame 0 [:5]: [%.6f, %.6f, %.6f, %.6f, %.6f]\n",
                    q[0], q[1], q[2], q[3], q[4]);
            fprintf(stderr, "[DECODER] Layer 0 K frame 0 [:5]: [%.6f, %.6f, %.6f, %.6f, %.6f]\n",
                    kk[0], kk[1], kk[2], kk[3], kk[4]);
            fprintf(stderr, "[DECODER] Layer 0 V frame 0 [:5]: [%.6f, %.6f, %.6f, %.6f, %.6f]\n",
                    vv[0], vv[1], vv[2], vv[3], vv[4]);
        }

        /* NeoX split-half RoPE (NEON-optimized, NO QK-norm for pre-transformer) */
        for (int s = 0; s < n_frames; s++) {
            const float *cos_ptr = sd->rope_cos + s * half_hd;
            const float *sin_ptr = sd->rope_sin + s * half_hd;
            for (int hh = 0; hh < n_heads; hh++) {
                float *qh = q + s * qkv_dim + hh * head_dim;
                float *kh = kk + s * qkv_dim + hh * head_dim;
#ifdef __ARM_NEON
                int i = 0;
                for (; i + 3 < half_hd; i += 4) {
                    float32x4_t c = vld1q_f32(cos_ptr + i);
                    float32x4_t si = vld1q_f32(sin_ptr + i);
                    float32x4_t q1 = vld1q_f32(qh + i), q2 = vld1q_f32(qh + i + half_hd);
                    float32x4_t k1 = vld1q_f32(kh + i), k2 = vld1q_f32(kh + i + half_hd);
                    vst1q_f32(qh + i,           vmlsq_f32(vmulq_f32(q1, c), q2, si));
                    vst1q_f32(qh + i + half_hd, vmlaq_f32(vmulq_f32(q2, c), q1, si));
                    vst1q_f32(kh + i,           vmlsq_f32(vmulq_f32(k1, c), k2, si));
                    vst1q_f32(kh + i + half_hd, vmlaq_f32(vmulq_f32(k2, c), k1, si));
                }
                for (; i < half_hd; i++) {
                    float qv1 = qh[i], qv2 = qh[i + half_hd];
                    float kv1 = kh[i], kv2 = kh[i + half_hd];
                    float co = cos_ptr[i], sn = sin_ptr[i];
                    qh[i] = qv1 * co - qv2 * sn; qh[i + half_hd] = qv2 * co + qv1 * sn;
                    kh[i] = kv1 * co - kv2 * sn; kh[i + half_hd] = kv2 * co + kv1 * sn;
                }
#elif defined(__AVX2__)
                int i = 0;
                for (; i + 8 <= half_hd; i += 8) {
                    __m256 c = _mm256_loadu_ps(cos_ptr + i);
                    __m256 si = _mm256_loadu_ps(sin_ptr + i);
                    __m256 q1 = _mm256_loadu_ps(qh + i), q2 = _mm256_loadu_ps(qh + i + half_hd);
                    __m256 k1 = _mm256_loadu_ps(kh + i), k2 = _mm256_loadu_ps(kh + i + half_hd);
                    _mm256_storeu_ps(qh + i,           _mm256_fmsub_ps(q1, c, _mm256_mul_ps(q2, si)));
                    _mm256_storeu_ps(qh + i + half_hd, _mm256_fmadd_ps(q2, c, _mm256_mul_ps(q1, si)));
                    _mm256_storeu_ps(kh + i,           _mm256_fmsub_ps(k1, c, _mm256_mul_ps(k2, si)));
                    _mm256_storeu_ps(kh + i + half_hd, _mm256_fmadd_ps(k2, c, _mm256_mul_ps(k1, si)));
                }
                for (; i < half_hd; i++) {
                    float qv1 = qh[i], qv2 = qh[i + half_hd];
                    float kv1 = kh[i], kv2 = kh[i + half_hd];
                    float co = cos_ptr[i], sn = sin_ptr[i];
                    qh[i] = qv1 * co - qv2 * sn; qh[i + half_hd] = qv2 * co + qv1 * sn;
                    kh[i] = kv1 * co - kv2 * sn; kh[i + half_hd] = kv2 * co + kv1 * sn;
                }
#else
                for (int i = 0; i < half_hd; i++) {
                    float q1 = qh[i], q2 = qh[i + half_hd];
                    float k1 = kh[i], k2 = kh[i + half_hd];
                    float co = cos_ptr[i], si = sin_ptr[i];
                    qh[i] = q1 * co - q2 * si; qh[i + half_hd] = q2 * co + q1 * si;
                    kh[i] = k1 * co - k2 * si; kh[i + half_hd] = k2 * co + k1 * si;
                }
#endif
            }
        }

        /* Sliding window causal attention (NEON-optimized) */
        float scale = 1.0f / sqrtf((float)head_dim);
        qwen_causal_attention_windowed(attn_out, q, kk, vv,
                                        n_frames, n_frames, n_heads, n_heads,
                                        head_dim, scale, 0, window);

        /* Output proj + layer_scale + residual */
#ifdef USE_BLAS
        {
            /* proj[n_frames, dec_hidden] = attn_out[n_frames, qkv_dim] × attn_o^T */
            float *oproj = x_norm; /* reuse x_norm as temp (same size: n_frames * dec_hidden) */
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        n_frames, dec_hidden, qkv_dim, 1.0f,
                        attn_out, qkv_dim, l->attn_o, qkv_dim,
                        0.0f, oproj, dec_hidden);
            for (int s = 0; s < n_frames; s++) {
                float *xs = hidden + s * dec_hidden;
                float *ps = oproj + s * dec_hidden;
                if (l->attn_layer_scale) {
                    for (int o = 0; o < dec_hidden; o++)
                        xs[o] += ps[o] * l->attn_layer_scale[o];
                } else {
                    for (int o = 0; o < dec_hidden; o++)
                        xs[o] += ps[o];
                }
            }
        }
#else
        for (int s = 0; s < n_frames; s++) {
            float *xs = hidden + s * dec_hidden;
            const float *attn = attn_out + s * qkv_dim;
            for (int o = 0; o < dec_hidden; o++) {
                float sum = 0;
                for (int i = 0; i < qkv_dim; i++)
                    sum += l->attn_o[(int64_t)o * qkv_dim + i] * attn[i];
                if (l->attn_layer_scale)
                    sum *= l->attn_layer_scale[o];
                xs[o] += sum;
            }
        }
#endif

        /* Debug after attention + residual */
        if (ctx->debug && layer == 0) {
            fprintf(stderr, "[DECODER] Layer 0 after attn+res frame 0 [:5]: [%.6f, %.6f, %.6f, %.6f, %.6f]\n",
                    hidden[0], hidden[1], hidden[2], hidden[3], hidden[4]);
        }

        /* Post-attn RMSNorm (NEON-optimized) */
        qwen_rms_norm(x_norm, hidden, l->ffn_norm, n_frames, dec_hidden, eps);

        /* SwiGLU FFN: down_proj(SiLU(gate_proj(x)) * up_proj(x)) + layer_scale + residual */
#ifdef USE_BLAS
        {
            float *ffn_gate = (float *)aligned_malloc((int64_t)n_frames * dec_inter * sizeof(float));
            float *ffn_up = (float *)aligned_malloc((int64_t)n_frames * dec_inter * sizeof(float));
            /* gate[n_frames, dec_inter] = x_norm[n_frames, dec_hidden] × W_gate^T */
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        n_frames, dec_inter, dec_hidden, 1.0f,
                        x_norm, dec_hidden, l->ffn_gate, dec_hidden,
                        0.0f, ffn_gate, dec_inter);
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        n_frames, dec_inter, dec_hidden, 1.0f,
                        x_norm, dec_hidden, l->ffn_up, dec_hidden,
                        0.0f, ffn_up, dec_inter);
            /* SiLU(gate) * up */
            for (int64_t i = 0; i < (int64_t)n_frames * dec_inter; i++)
                ffn_gate[i] = (ffn_gate[i] / (1.0f + expf(-ffn_gate[i]))) * ffn_up[i];
            free(ffn_up);
            /* down[n_frames, dec_hidden] = ffn_gate[n_frames, dec_inter] × W_down^T */
            float *ffn_down_out = ffn_up = (float *)aligned_malloc((int64_t)n_frames * dec_hidden * sizeof(float));
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        n_frames, dec_hidden, dec_inter, 1.0f,
                        ffn_gate, dec_inter, l->ffn_down, dec_inter,
                        0.0f, ffn_down_out, dec_hidden);
            free(ffn_gate);
            /* layer_scale + residual */
            for (int s = 0; s < n_frames; s++) {
                float *hs = hidden + s * dec_hidden;
                float *ds = ffn_down_out + s * dec_hidden;
                if (l->ffn_layer_scale) {
                    for (int o = 0; o < dec_hidden; o++)
                        hs[o] += ds[o] * l->ffn_layer_scale[o];
                } else {
                    for (int o = 0; o < dec_hidden; o++)
                        hs[o] += ds[o];
                }
            }
            free(ffn_down_out);
        }
#else
        for (int s = 0; s < n_frames; s++) {
            const float *xs = x_norm + s * dec_hidden;
            float *hs = hidden + s * dec_hidden;

            /* gate and up projections */
            float gate_up[dec_inter * 2]; /* VLA */
            for (int o = 0; o < dec_inter; o++) {
                float sum_g = 0, sum_u = 0;
                for (int i = 0; i < dec_hidden; i++) {
                    sum_g += l->ffn_gate[(int64_t)o * dec_hidden + i] * xs[i];
                    sum_u += l->ffn_up[(int64_t)o * dec_hidden + i] * xs[i];
                }
                /* SiLU on gate, multiply by up */
                gate_up[o] = (sum_g / (1.0f + expf(-sum_g))) * sum_u;
            }

            /* down projection + layer_scale + residual */
            for (int o = 0; o < dec_hidden; o++) {
                float sum = 0;
                for (int i = 0; i < dec_inter; i++)
                    sum += l->ffn_down[(int64_t)o * dec_inter + i] * gate_up[i];
                if (l->ffn_layer_scale)
                    sum *= l->ffn_layer_scale[o];
                hs[o] += sum;
            }
        }
#endif

        /* Per-layer debug */
        if (ctx->debug) {
            fprintf(stderr, "[DECODER] Layer %d out frame 0 [:5]: [%.6f, %.6f, %.6f, %.6f, %.6f]\n",
                    layer, hidden[0], hidden[1], hidden[2], hidden[3], hidden[4]);
        }
    }

    free(q); free(kk); free(vv); free(x_norm); free(attn_out);

    /* Debug: after pre-transformer */
    if (ctx->debug) {
        fprintf(stderr, "[DECODER] pre-trans out frame 0 [:5]: [%.6f, %.6f, %.6f, %.6f, %.6f]\n",
                hidden[0], hidden[1], hidden[2], hidden[3], hidden[4]);
    }

    /* Step 5: Final RMSNorm + Output proj (512→1024) */
    if (sd->final_norm_weight) {
        qwen_rms_norm(hidden, hidden, sd->final_norm_weight, n_frames, dec_hidden, eps);
    }

    if (ctx->debug) {
        fprintf(stderr, "[DECODER] final_norm frame 0 [:5]: [%.6f, %.6f, %.6f, %.6f, %.6f]\n",
                hidden[0], hidden[1], hidden[2], hidden[3], hidden[4]);
    }

    float *latent_out = (float *)aligned_malloc((int64_t)n_frames * latent_dim * sizeof(float));
#ifdef USE_BLAS
    /* latent_out[n_frames, 1024] = hidden[n_frames, 512] × W^T[512, 1024] */
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                n_frames, latent_dim, dec_hidden, 1.0f,
                hidden, dec_hidden,
                sd->output_proj_weight, dec_hidden,
                0.0f, latent_out, latent_dim);
    if (sd->output_proj_bias) {
        for (int f = 0; f < n_frames; f++)
            for (int o = 0; o < latent_dim; o++)
                latent_out[(int64_t)f * latent_dim + o] += sd->output_proj_bias[o];
    }
#else
    for (int f = 0; f < n_frames; f++) {
        for (int o = 0; o < latent_dim; o++) {
            float sum = sd->output_proj_bias ? sd->output_proj_bias[o] : 0;
            for (int i = 0; i < dec_hidden; i++)
                sum += sd->output_proj_weight[(int64_t)o * dec_hidden + i] * hidden[(int64_t)f * dec_hidden + i];
            latent_out[(int64_t)f * latent_dim + o] = sum;
        }
    }
#endif
    free(hidden);

    /* Step 6: Transpose to channel-first [1024, n_frames] */
    float *signal = (float *)aligned_malloc((int64_t)latent_dim * n_frames * sizeof(float));
    for (int f = 0; f < n_frames; f++)
        for (int d = 0; d < latent_dim; d++)
            signal[(int64_t)d * n_frames + f] = latent_out[(int64_t)f * latent_dim + d];
    free(latent_out);

    /* Debug: after output_proj */
    if (ctx->debug) {
        fprintf(stderr, "[DECODER] output_proj out frame 0 [:5]: [%.6f, %.6f, %.6f, %.6f, %.6f]\n",
                signal[0 * n_frames + 0], signal[1 * n_frames + 0],
                signal[2 * n_frames + 0], signal[3 * n_frames + 0],
                signal[4 * n_frames + 0]);
    }
    if (ctx->debug) fprintf(stderr, "[DECODER] ConvNeXt upsample...\n");
    int cur_ch = latent_dim;
    int cur_len = n_frames;

    /* Step 7: ConvNeXt upsample (2 blocks, 2x each) */
    for (int b = 0; b < 2; b++) {
        qwen_sd_convnext_t *cn = &sd->convnext[b];
        if (!cn->conv_weight) { fprintf(stderr, "ERROR: ConvNeXt block %d weights missing!\n", b); free(signal); return -1; }

        int new_len = conv_transpose1d_out_len(cur_len, 2, 2);

        /* Full ConvTranspose1d 2x upsample: [1024, 1024, 2] */
        float *up_out = (float *)aligned_calloc((int64_t)cur_ch * new_len, sizeof(float));
        causal_conv_transpose1d(up_out, signal, cn->conv_weight, cn->conv_bias,
                                 cur_ch, cur_ch, cur_len, new_len, 2, 2);
        free(signal); signal = up_out; cur_len = new_len;

        /* ConvNeXt block: DW conv → LayerNorm → PW1 → GELU → PW2 → gamma → residual */
        float *residual = (float *)aligned_malloc((int64_t)cur_ch * cur_len * sizeof(float));
        memcpy(residual, signal, (int64_t)cur_ch * cur_len * sizeof(float));

        /* Depthwise conv (k=7, groups=cur_ch, pad_left=6) */
        float *dw_out = (float *)aligned_calloc((int64_t)cur_ch * cur_len, sizeof(float));
        for (int ci = 0; ci < cur_ch; ci++) {
            for (int t = 0; t < cur_len; t++) {
                float sum = cn->dwconv_bias ? cn->dwconv_bias[ci] : 0;
                for (int k = 0; k < 7; k++) {
                    int in_pos = t - (6 - k);
                    if (in_pos >= 0 && in_pos < cur_len)
                        sum += cn->dwconv_weight[(int64_t)ci * 7 + k] * signal[(int64_t)ci * cur_len + in_pos];
                }
                dw_out[(int64_t)ci * cur_len + t] = sum;
            }
        }
        memcpy(signal, dw_out, (int64_t)cur_ch * cur_len * sizeof(float));
        free(dw_out);

        /* LayerNorm per timestep (over channels) */
        for (int t = 0; t < cur_len; t++) {
            float sum = 0, sum_sq = 0;
            for (int ci = 0; ci < cur_ch; ci++) {
                float val = signal[(int64_t)ci * cur_len + t];
                sum += val; sum_sq += val * val;
            }
            float mean = sum / cur_ch;
            float var = sum_sq / cur_ch - mean * mean;
            float inv_std = 1.0f / sqrtf(var + 1e-5f);
            for (int ci = 0; ci < cur_ch; ci++) {
                float *p = &signal[(int64_t)ci * cur_len + t];
                *p = (*p - mean) * inv_std * cn->norm_weight[ci] + cn->norm_bias[ci];
            }
        }

        /* PW1: 1024→4096 (pointwise = 1x1 conv = matmul per timestep) */
        int pw_dim = cur_ch * 4;
        float *pw1_out = (float *)aligned_malloc((int64_t)pw_dim * cur_len * sizeof(float));
#ifdef USE_BLAS
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    pw_dim, cur_len, cur_ch,
                    1.0f, cn->pwconv1_weight, cur_ch,
                    signal, cur_len,
                    0.0f, pw1_out, cur_len);
        conv_add_bias(pw1_out, cn->pwconv1_bias, pw_dim, cur_len);
#else
        for (int t = 0; t < cur_len; t++) {
            for (int o = 0; o < pw_dim; o++) {
                float sum = cn->pwconv1_bias ? cn->pwconv1_bias[o] : 0;
                for (int i = 0; i < cur_ch; i++)
                    sum += cn->pwconv1_weight[(int64_t)o * cur_ch + i] * signal[(int64_t)i * cur_len + t];
                pw1_out[(int64_t)o * cur_len + t] = sum;
            }
        }
#endif

        /* Exact GELU: x * 0.5 * (1 + erf(x / sqrt(2))) */
        for (int64_t i = 0; i < (int64_t)pw_dim * cur_len; i++) {
            float x = pw1_out[i];
            pw1_out[i] = 0.5f * x * (1.0f + erff(x * 0.7071067811865476f));
        }

        /* PW2: 4096→1024 */
#ifdef USE_BLAS
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    cur_ch, cur_len, pw_dim,
                    1.0f, cn->pwconv2_weight, pw_dim,
                    pw1_out, cur_len,
                    0.0f, signal, cur_len);
        conv_add_bias(signal, cn->pwconv2_bias, cur_ch, cur_len);
#else
        for (int t = 0; t < cur_len; t++) {
            for (int o = 0; o < cur_ch; o++) {
                float sum = cn->pwconv2_bias ? cn->pwconv2_bias[o] : 0;
                for (int i = 0; i < pw_dim; i++)
                    sum += cn->pwconv2_weight[(int64_t)o * pw_dim + i] * pw1_out[(int64_t)i * cur_len + t];
                signal[(int64_t)o * cur_len + t] = sum;
            }
        }
#endif
        free(pw1_out);

        /* Gamma + residual */
        for (int ci = 0; ci < cur_ch; ci++) {
            float g = cn->gamma[ci];
            for (int t = 0; t < cur_len; t++)
                signal[(int64_t)ci * cur_len + t] = residual[(int64_t)ci * cur_len + t]
                    + signal[(int64_t)ci * cur_len + t] * g;
        }
        free(residual);
    }

    if (ctx->debug) fprintf(stderr, "[DECODER] Initial conv...\n");

    /* Step 8: Initial conv (1024→1536, k=7, pad_left=6) */
    if (!sd->initial_conv_weight) { free(signal); return -1; }
    int new_ch = 1536;
    int new_len = conv1d_out_len(cur_len, 7, 1, 6);
    float *conv_out = (float *)aligned_calloc((int64_t)new_ch * new_len, sizeof(float));
    causal_conv1d(conv_out, signal, sd->initial_conv_weight, sd->initial_conv_bias,
                  cur_ch, new_ch, cur_len, 7, 1);
    free(signal); signal = conv_out; cur_ch = new_ch; cur_len = new_len;

    /* Step 9: 4 Decoder upsample blocks */
    int up_rates[4] = {8, 5, 4, 3};
    int out_channels[4] = {768, 384, 192, 96};

    if (ctx->debug) fprintf(stderr, "[DECODER] Upsample blocks...\n");
    for (int b = 0; b < 4; b++) {
        qwen_sd_upsample_block_t *ub = &sd->upsample_blocks[b];
        int rate = up_rates[b];
        int kernel = rate * 2;
        int out_ch = out_channels[b];

        if (!ub->upsample.conv_weight) {
            fprintf(stderr, "ERROR: Upsample block %d weights missing!\n", b);
            free(signal); return -1;
        }

        /* Snake activation before upsample */
        if (ub->upsample.snake_alpha && ub->upsample.snake_beta)
            snake_activation(signal, cur_ch, cur_len, ub->upsample.snake_alpha, ub->upsample.snake_beta);

        /* ConvTranspose1d upsample: [in_ch, out_ch, kernel] */
        int up_len = conv_transpose1d_out_len(cur_len, kernel, rate);
        float *up_out = (float *)aligned_calloc((int64_t)out_ch * up_len, sizeof(float));
        causal_conv_transpose1d(up_out, signal, ub->upsample.conv_weight, ub->upsample.conv_bias,
                                 cur_ch, out_ch, cur_len, up_len, kernel, rate);
        free(signal); signal = up_out; cur_ch = out_ch; cur_len = up_len;

        /* 3 residual blocks with dilations [1, 3, 9] */
        int dilations[3] = {1, 3, 9};
        for (int r = 0; r < 3; r++) {
            int dil = dilations[r];

            float *res = (float *)aligned_malloc((int64_t)cur_ch * cur_len * sizeof(float));
            memcpy(res, signal, (int64_t)cur_ch * cur_len * sizeof(float));

            /* Snake 1 */
            if (ub->res_blocks[r].snake1_alpha && ub->res_blocks[r].snake1_beta)
                snake_activation(signal, cur_ch, cur_len,
                                  ub->res_blocks[r].snake1_alpha, ub->res_blocks[r].snake1_beta);

            /* Conv1 (k=7, dilation, causal): [ch, ch, 7] */
            float *c1_out = (float *)aligned_calloc((int64_t)cur_ch * cur_len, sizeof(float));
            causal_conv1d(c1_out, signal, ub->res_blocks[r].conv1_weight, ub->res_blocks[r].conv1_bias,
                          cur_ch, cur_ch, cur_len, 7, dil);
            memcpy(signal, c1_out, (int64_t)cur_ch * cur_len * sizeof(float));
            free(c1_out);

            /* Snake 2 */
            if (ub->res_blocks[r].snake2_alpha && ub->res_blocks[r].snake2_beta)
                snake_activation(signal, cur_ch, cur_len,
                                  ub->res_blocks[r].snake2_alpha, ub->res_blocks[r].snake2_beta);

            /* Conv2 (k=1): [ch, ch, 1] */
            float *c2_out = (float *)aligned_calloc((int64_t)cur_ch * cur_len, sizeof(float));
            causal_conv1d(c2_out, signal, ub->res_blocks[r].conv2_weight, ub->res_blocks[r].conv2_bias,
                          cur_ch, cur_ch, cur_len, 1, 1);

            /* Residual add */
            for (int64_t i = 0; i < (int64_t)cur_ch * cur_len; i++)
                signal[i] = res[i] + c2_out[i];
            free(c2_out);
            free(res);
        }

        if (ctx->debug) fprintf(stderr, "[DECODER]   Block %d done: ch=%d, len=%d\n", b+1, cur_ch, cur_len);
    }

    if (ctx->debug) fprintf(stderr, "[DECODER] Final conv...\n");

    /* Step 10: Final Snake + Conv (96→1, k=7) */
    if (!sd->final_snake.alpha || !sd->final_conv_weight) {
        fprintf(stderr, "ERROR: Final snake/conv weights missing!\n");
        free(signal); return -1;
    }

    /* Final snake activation */
    snake_activation(signal, cur_ch, cur_len, sd->final_snake.alpha, sd->final_snake.beta);

    /* Final conv: [1, 96, 7] */
    int audio_len = conv1d_out_len(cur_len, 7, 1, 6);
    float *audio = (float *)aligned_calloc(audio_len, sizeof(float));
    for (int t = 0; t < audio_len; t++) {
        float sum = sd->final_conv_bias ? sd->final_conv_bias[0] : 0;
        for (int ic = 0; ic < cur_ch; ic++) {
            for (int k = 0; k < 7; k++) {
                int in_pos = t - (6 - k);
                if (in_pos >= 0 && in_pos < cur_len)
                    sum += sd->final_conv_weight[(int64_t)ic * 7 + k] * signal[(int64_t)ic * cur_len + in_pos];
            }
        }
        audio[t] = sum;
    }
    free(signal);

    /* Clamp to [-1, 1] */
    for (int i = 0; i < audio_len; i++) {
        if (audio[i] < -1.0f) audio[i] = -1.0f;
        if (audio[i] > 1.0f) audio[i] = 1.0f;
    }

    *audio_out = audio;
    *n_samples = audio_len;

    if (!ctx->silent)
        fprintf(stderr, "  Speech decoder output: %d samples (%.2fs @ 24kHz)\n",
                audio_len, (float)audio_len / 24000.0f);

    return 0;
}

/* ========================================================================
 * Streaming Incremental Decode
 *
 * Instead of re-decoding ALL accumulated frames each streaming chunk (O(n²)),
 * this processes only NEW frames through VQ → pre-transformer (with KV cache),
 * caches the latent output, and runs the conv decoder on a small window
 * (context + new frames) for O(1) per chunk.
 *
 * Audio output is exactly 1920 samples per codec frame (by design: 4×480× upsample).
 * ======================================================================== */

/* Initialize/reset streaming state */
void qwen_sd_stream_init(qwen_sd_stream_state_t *st) {
    memset(st, 0, sizeof(*st));
}

/* Free streaming state buffers */
void qwen_sd_stream_free(qwen_sd_stream_state_t *st) {
    for (int i = 0; i < QWEN_SD_STREAM_MAX_LAYERS; i++) {
        free(st->k_cache[i]); st->k_cache[i] = NULL;
        free(st->v_cache[i]); st->v_cache[i] = NULL;
    }
    free(st->latent_cache); st->latent_cache = NULL;
    free(st->vq_pad); st->vq_pad = NULL;
    /* Exact-streaming conv state (may be NULL if the stream never decoded). */
    for (int b = 0; b < 2; b++) { free(st->cs_cn_dw_tail[b]); st->cs_cn_dw_tail[b] = NULL; }
    free(st->cs_init_tail);  st->cs_init_tail = NULL;
    free(st->cs_final_tail); st->cs_final_tail = NULL;
    for (int b = 0; b < 4; b++) {
        free(st->cs_up_carry[b]); st->cs_up_carry[b] = NULL;
        for (int r = 0; r < 3; r++) { free(st->cs_res_tail[b][r]); st->cs_res_tail[b][r] = NULL; }
    }
    st->cs_alloc = 0;
    memset(st, 0, sizeof(*st));
}

/* Run conv decoder (ConvNeXt + initial conv + upsample blocks + final conv)
 * on a signal in channel-first format [latent_dim, n_frames].
 * Returns audio samples. This is the same pipeline as steps 7-10 in the
 * full decode, extracted as a helper to avoid duplication. */
#ifdef QWEN_HAVE_CUDA
extern int g_cuda_decoder_conv_on;
extern int qwen_cuda_conv_decoder_run(void *ctx, float *signal, int cur_ch, int cur_len, float **audio_out, int *n_out);
#endif

/* ConvNeXt block tail: LayerNorm -> pw1(+GELU) -> pw2 -> gamma*x + residual,
 * in place on `signal` [cur_ch, cur_len]. Every step is per-timestep, so this is
 * identical for the one-shot and the streaming decode (no cross-chunk state) —
 * hence shared by both. */
static void convnext_mlp(qwen_sd_convnext_t *cn, float *signal, const float *residual,
                         int cur_ch, int cur_len) {
    /* LayerNorm (per-timestep) */
    for (int t = 0; t < cur_len; t++) {
        float mean = 0, var = 0;
        for (int c = 0; c < cur_ch; c++) mean += signal[(int64_t)c * cur_len + t];
        mean /= cur_ch;
        for (int c = 0; c < cur_ch; c++) {
            float d = signal[(int64_t)c * cur_len + t] - mean;
            var += d * d;
        }
        var = 1.0f / sqrtf(var / cur_ch + 1e-5f);
        for (int c = 0; c < cur_ch; c++) {
            float x = (signal[(int64_t)c * cur_len + t] - mean) * var;
            signal[(int64_t)c * cur_len + t] = x * cn->norm_weight[c] + cn->norm_bias[c];
        }
    }

    /* Pointwise convs: pw1 (1024→4096, GELU), pw2 (4096→1024) */
    int pw_dim = 4096;
    float *pw1_out = (float *)aligned_malloc((int64_t)pw_dim * cur_len * sizeof(float));
#ifdef USE_BLAS
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                pw_dim, cur_len, cur_ch, 1.0f,
                cn->pwconv1_weight, cur_ch, signal, cur_len,
                0.0f, pw1_out, cur_len);
    if (cn->pwconv1_bias)
        for (int i = 0; i < pw_dim; i++)
            for (int t = 0; t < cur_len; t++)
                pw1_out[(int64_t)i * cur_len + t] += cn->pwconv1_bias[i];
#else
    for (int o = 0; o < pw_dim; o++)
        for (int t = 0; t < cur_len; t++) {
            float sum = cn->pwconv1_bias ? cn->pwconv1_bias[o] : 0;
            for (int i = 0; i < cur_ch; i++)
                sum += cn->pwconv1_weight[(int64_t)o * cur_ch + i] * signal[(int64_t)i * cur_len + t];
            pw1_out[(int64_t)o * cur_len + t] = sum;
        }
#endif
    /* Exact GELU */
    for (int64_t i = 0; i < (int64_t)pw_dim * cur_len; i++)
        pw1_out[i] = 0.5f * pw1_out[i] * (1.0f + erff(pw1_out[i] * 0.7071067811865476f));

    /* pw2 (sgemm writes with beta=0, so no memset needed) */
#ifdef USE_BLAS
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                cur_ch, cur_len, pw_dim, 1.0f,
                cn->pwconv2_weight, pw_dim, pw1_out, cur_len,
                0.0f, signal, cur_len);
    if (cn->pwconv2_bias)
        for (int o = 0; o < cur_ch; o++)
            for (int t = 0; t < cur_len; t++)
                signal[(int64_t)o * cur_len + t] += cn->pwconv2_bias[o];
#else
    for (int o = 0; o < cur_ch; o++)
        for (int t = 0; t < cur_len; t++) {
            float sum = cn->pwconv2_bias ? cn->pwconv2_bias[o] : 0;
            for (int i = 0; i < pw_dim; i++)
                sum += cn->pwconv2_weight[(int64_t)o * pw_dim + i] * pw1_out[(int64_t)i * cur_len + t];
            signal[(int64_t)o * cur_len + t] = sum;
        }
#endif
    free(pw1_out);

    /* Gamma + residual */
    for (int ci = 0; ci < cur_ch; ci++) {
        float g = cn->gamma[ci];
        for (int t = 0; t < cur_len; t++)
            signal[(int64_t)ci * cur_len + t] = residual[(int64_t)ci * cur_len + t]
                + signal[(int64_t)ci * cur_len + t] * g;
    }
}
static int conv_decoder_forward(qwen_tts_ctx_t *ctx,
                                 float *signal, int cur_ch, int cur_len,
                                 float **audio_out, int *n_samples_out) {
    qwen_speech_decoder_t *sd = &ctx->speech_dec;
#ifdef QWEN_HAVE_CUDA
    if (g_cuda_decoder_conv_on) {   /* GPU-resident ConvNet decoder (M3): activations stay on device */
        int rc = qwen_cuda_conv_decoder_run(ctx, signal, cur_ch, cur_len, audio_out, n_samples_out);
        free(signal);   /* the GPU path copies signal in; free the host copy here */
        return rc;
    }
#endif

    /* ConvNeXt upsample (2 blocks, 2x each → 4x total) */
    for (int b = 0; b < 2; b++) {
        qwen_sd_convnext_t *cn = &sd->convnext[b];
        if (!cn->conv_weight) { free(signal); return -1; }

        int new_len = conv_transpose1d_out_len(cur_len, 2, 2);
        float *up_out = (float *)aligned_calloc((int64_t)cur_ch * new_len, sizeof(float));
        causal_conv_transpose1d(up_out, signal, cn->conv_weight, cn->conv_bias,
                                 cur_ch, cur_ch, cur_len, new_len, 2, 2);
        free(signal); signal = up_out; cur_len = new_len;

        /* Depthwise conv (k=7, pad=6) */
        float *dw_out = (float *)aligned_calloc((int64_t)cur_ch * cur_len, sizeof(float));
        for (int c = 0; c < cur_ch; c++) {
            for (int t = 0; t < cur_len; t++) {
                float sum = cn->dwconv_bias ? cn->dwconv_bias[c] : 0;
                for (int k = 0; k < 7; k++) {
                    int in_pos = t - 6 + k;
                    if (in_pos >= 0 && in_pos < cur_len)
                        sum += cn->dwconv_weight[c * 7 + k] * signal[(int64_t)c * cur_len + in_pos];
                }
                dw_out[(int64_t)c * cur_len + t] = sum;
            }
        }

        float *residual = signal; signal = dw_out;

        convnext_mlp(cn, signal, residual, cur_ch, cur_len);
        free(residual);
    }

    /* Initial conv (1024→1536, k=7, pad_left=6) */
    if (!sd->initial_conv_weight) { free(signal); return -1; }
    int new_ch = 1536;
    int new_len = conv1d_out_len(cur_len, 7, 1, 6);
    float *conv_out = (float *)aligned_calloc((int64_t)new_ch * new_len, sizeof(float));
    causal_conv1d(conv_out, signal, sd->initial_conv_weight, sd->initial_conv_bias,
                  cur_ch, new_ch, cur_len, 7, 1);
    free(signal); signal = conv_out; cur_ch = new_ch; cur_len = new_len;

    /* 4 Decoder upsample blocks */
    int up_rates[4] = {8, 5, 4, 3};
    int out_channels[4] = {768, 384, 192, 96};

    for (int b = 0; b < 4; b++) {
        qwen_sd_upsample_block_t *ub = &sd->upsample_blocks[b];
        int rate = up_rates[b];
        int kernel = rate * 2;
        int out_ch = out_channels[b];

        if (!ub->upsample.conv_weight) { free(signal); return -1; }

        if (ub->upsample.snake_alpha && ub->upsample.snake_beta)
            snake_activation(signal, cur_ch, cur_len, ub->upsample.snake_alpha, ub->upsample.snake_beta);

        int up_len = conv_transpose1d_out_len(cur_len, kernel, rate);
        float *up_out = (float *)aligned_calloc((int64_t)out_ch * up_len, sizeof(float));
        causal_conv_transpose1d(up_out, signal, ub->upsample.conv_weight, ub->upsample.conv_bias,
                                 cur_ch, out_ch, cur_len, up_len, kernel, rate);
        free(signal); signal = up_out; cur_ch = out_ch; cur_len = up_len;

        int dilations[3] = {1, 3, 9};
        for (int r = 0; r < 3; r++) {
            int dil = dilations[r];
            float *res = (float *)aligned_malloc((int64_t)cur_ch * cur_len * sizeof(float));
            memcpy(res, signal, (int64_t)cur_ch * cur_len * sizeof(float));

            if (ub->res_blocks[r].snake1_alpha && ub->res_blocks[r].snake1_beta)
                snake_activation(signal, cur_ch, cur_len,
                                  ub->res_blocks[r].snake1_alpha, ub->res_blocks[r].snake1_beta);

            float *c1_out = (float *)aligned_calloc((int64_t)cur_ch * cur_len, sizeof(float));
            causal_conv1d(c1_out, signal, ub->res_blocks[r].conv1_weight, ub->res_blocks[r].conv1_bias,
                          cur_ch, cur_ch, cur_len, 7, dil);
            memcpy(signal, c1_out, (int64_t)cur_ch * cur_len * sizeof(float));
            free(c1_out);

            if (ub->res_blocks[r].snake2_alpha && ub->res_blocks[r].snake2_beta)
                snake_activation(signal, cur_ch, cur_len,
                                  ub->res_blocks[r].snake2_alpha, ub->res_blocks[r].snake2_beta);

            float *c2_out = (float *)aligned_calloc((int64_t)cur_ch * cur_len, sizeof(float));
            causal_conv1d(c2_out, signal, ub->res_blocks[r].conv2_weight, ub->res_blocks[r].conv2_bias,
                          cur_ch, cur_ch, cur_len, 1, 1);

            for (int64_t i = 0; i < (int64_t)cur_ch * cur_len; i++)
                signal[i] = res[i] + c2_out[i];
            free(c2_out);
            free(res);
        }
    }

    /* Final Snake + Conv (96→1, k=7) */
    if (!sd->final_snake.alpha || !sd->final_conv_weight) { free(signal); return -1; }
    snake_activation(signal, cur_ch, cur_len, sd->final_snake.alpha, sd->final_snake.beta);

    int audio_len = conv1d_out_len(cur_len, 7, 1, 6);
    float *audio = (float *)aligned_calloc(audio_len, sizeof(float));
    for (int t = 0; t < audio_len; t++) {
        float sum = sd->final_conv_bias ? sd->final_conv_bias[0] : 0;
        for (int ic = 0; ic < cur_ch; ic++) {
            for (int k = 0; k < 7; k++) {
                int in_pos = t - (6 - k);
                if (in_pos >= 0 && in_pos < cur_len)
                    sum += sd->final_conv_weight[(int64_t)ic * 7 + k] * signal[(int64_t)ic * cur_len + in_pos];
            }
        }
        audio[t] = sum;
    }
    free(signal);

    for (int i = 0; i < audio_len; i++) {
        if (audio[i] < -1.0f) audio[i] = -1.0f;
        if (audio[i] > 1.0f) audio[i] = 1.0f;
    }

    *audio_out = audio;
    *n_samples_out = audio_len;
    return 0;
}

/* ========================================================================
 * Exact streaming conv decoder (ported from external PR #17, TrinityTF)
 *
 * Same math as conv_decoder_forward, but consumes ONLY the new latent frames
 * and carries the causal state across calls:
 *   - conv1d (k>1): keep the last pad_left = (k-1)*dilation input columns. A
 *     zero-initialized tail is exactly the causal zero padding of chunk 0.
 *   - ConvTranspose: keep the (kernel-stride) untrimmed partial output columns
 *     and overlap-add them into the next chunk (bias is added on emit only, so
 *     the carry stays a pure partial sum).
 *   - snake / LayerNorm / pointwise / GELU / k=1 conv: per-timestep, stateless.
 * Chunked output therefore equals the one-shot decode. The old windowed path
 * re-decoded conv_rf=20 context frames per chunk (3x the work at chunk=10) and
 * only approximated the boundary; it is kept behind QWEN_SD_WINDOWED=1.
 * ======================================================================== */

/* Save the last tail_cols columns of [tail | chunk] back into tail. When the
 * chunk is shorter than the tail the window straddles both, so shift. */
static void cs_save_tail(float *tail, const float *in, int in_ch, int len, int tail_cols) {
    if (len >= tail_cols) {
        for (int ic = 0; ic < in_ch; ic++)
            memcpy(tail + (int64_t)ic * tail_cols, in + (int64_t)ic * len + (len - tail_cols),
                   (size_t)tail_cols * sizeof(float));
    } else {
        int keep = tail_cols - len;   /* columns of the old tail that survive */
        for (int ic = 0; ic < in_ch; ic++) {
            float *t = tail + (int64_t)ic * tail_cols;
            memmove(t, t + len, (size_t)keep * sizeof(float));
            memcpy(t + keep, in + (int64_t)ic * len, (size_t)len * sizeof(float));
        }
    }
}

/* Stateful causal conv1d on one chunk: prepend the saved tail, convolve, emit
 * [out_ch x len], update the tail. Returns a freshly allocated output.
 *
 * `warm` = 0 on the very first chunk of a stream, when the tail is still all
 * zeros. An all-zero prepended tail is *exactly* the causal zero padding that
 * causal_conv1d already applies on its own, so we skip building `ext` and
 * convolve the chunk directly: same output, none of the work. That work was not
 * free -- at chunk=2 the res-block conv1 (dilation 9) has tail_cols=54 against
 * len=64, i.e. ~1.8x the convolution, all of it thrown away. It is what made
 * TTFA regress ~8% on Neoverse-N1 when the exact path landed. */
static float *cs_conv1d(const float *in, int in_ch, int out_ch, int len,
                        int kernel, int dilation,
                        const float *w, const float *b, float *tail, int warm) {
    int tail_cols = (kernel - 1) * dilation;

    if (!warm) {
        float *out = (float *)aligned_calloc((int64_t)out_ch * len, sizeof(float));
        if (!out) return NULL;
        causal_conv1d(out, in, w, b, in_ch, out_ch, len, kernel, dilation);
        cs_save_tail(tail, in, in_ch, len, tail_cols);
        return out;
    }

    int ext_len = tail_cols + len;
    float *ext = (float *)aligned_malloc((int64_t)in_ch * ext_len * sizeof(float));
    if (!ext) return NULL;
    for (int ic = 0; ic < in_ch; ic++) {
        memcpy(ext + (int64_t)ic * ext_len, tail + (int64_t)ic * tail_cols,
               (size_t)tail_cols * sizeof(float));
        memcpy(ext + (int64_t)ic * ext_len + tail_cols, in + (int64_t)ic * len,
               (size_t)len * sizeof(float));
    }
    /* New tail = last tail_cols columns of [tail | chunk]. When len < tail_cols
     * the window still straddles the old tail, so copy from ext (not from in). */
    for (int ic = 0; ic < in_ch; ic++)
        memcpy(tail + (int64_t)ic * tail_cols, ext + (int64_t)ic * ext_len + len,
               (size_t)tail_cols * sizeof(float));

    float *full = (float *)aligned_calloc((int64_t)out_ch * ext_len, sizeof(float));
    if (!full) { free(ext); return NULL; }
    causal_conv1d(full, ext, w, b, in_ch, out_ch, ext_len, kernel, dilation);
    free(ext);

    /* Drop the first tail_cols outputs: those re-pad with zeros (wrong) and were
     * already emitted by earlier chunks. */
    float *out = (float *)aligned_malloc((int64_t)out_ch * len * sizeof(float));
    if (!out) { free(full); return NULL; }
    for (int oc = 0; oc < out_ch; oc++)
        memcpy(out + (int64_t)oc * len, full + (int64_t)oc * ext_len + tail_cols,
               (size_t)len * sizeof(float));
    free(full);
    return out;
}

/* Stateful causal ConvTranspose1d on one chunk: run untrimmed, overlap-add the
 * carry into the head, emit len*stride columns (+bias), save the new
 * (kernel-stride)-column carry (bias-free partial sums).
 * carry == NULL is allowed when kernel == stride (no cross-chunk overlap). */
static float *cs_convt(const float *in, int in_ch, int out_ch, int len,
                       int kernel, int stride,
                       const float *w, const float *b, float *carry) {
    int cs = kernel - stride;
    int full_len = (len - 1) * stride + kernel;   /* = len*stride + cs */
    int out_len = len * stride;
    float *full = (float *)aligned_calloc((int64_t)out_ch * full_len, sizeof(float));
    if (!full) return NULL;
    /* out_len = full_len here, so causal_conv_transpose1d writes the untrimmed
     * output including the cs tail columns we need as the carry. Bias is NULL:
     * it must not be baked into the carry (it is added once, on emit). */
    causal_conv_transpose1d(full, in, w, NULL, in_ch, out_ch, len, full_len,
                            kernel, stride);
    float *out = (float *)aligned_malloc((int64_t)out_ch * out_len * sizeof(float));
    if (!out) { free(full); return NULL; }
    for (int oc = 0; oc < out_ch; oc++) {
        float *f = full + (int64_t)oc * full_len;
        float *o = out + (int64_t)oc * out_len;
        if (carry) {
            const float *cr = carry + (int64_t)oc * cs;
            for (int i = 0; i < cs; i++) f[i] += cr[i];
        }
        float bb = b ? b[oc] : 0.0f;
        for (int t = 0; t < out_len; t++) o[t] = f[t] + bb;
        if (carry) {
            float *cr = carry + (int64_t)oc * cs;
            for (int i = 0; i < cs; i++) cr[i] = f[out_len + i];
        }
    }
    free(full);
    return out;
}

/* Stateful ConvNeXt depthwise conv (k=7, pad_left=6) with a 6-column tail. */
static float *cs_dwconv(const float *in, int ch, int len,
                        const float *w, const float *b, float *tail) {
    float *out = (float *)aligned_malloc((int64_t)ch * len * sizeof(float));
    if (!out) return NULL;
    for (int c = 0; c < ch; c++) {
        const float *src = in + (int64_t)c * len;
        float *tl = tail + (int64_t)c * 6;
        float *dst = out + (int64_t)c * len;
        float bb = b ? b[c] : 0.0f;
        const float *wc = w + (int64_t)c * 7;
        for (int t = 0; t < len; t++) {
            float sum = bb;
            for (int k = 0; k < 7; k++) {
                int p = t - 6 + k;                  /* chunk-relative input pos */
                sum += wc[k] * (p >= 0 ? src[p] : tl[6 + p]);
            }
            dst[t] = sum;
        }
        /* New tail = last 6 inputs of [tail | chunk] */
        if (len >= 6) memcpy(tl, src + len - 6, 6 * sizeof(float));
        else
            for (int i = 0; i < 6; i++)
                tl[i] = (i + len < 6) ? tl[i + len] : src[i + len - 6];
    }
    return out;
}

/* Final conv (96 -> 1, k=7) with the per-slot 6-column input tail, written as a
 * STRIDED view so that the per-slot and the batched decoder call the very same
 * function on the very same instruction stream.
 *
 * WHY IT IS A SHARED FUNCTION RATHER THAN COPY-PASTE. Duplicating this loop in the
 * batched path made it disagree with the per-slot one in the last bits — identical
 * source, different codegen (an out-parameter the compiler cannot prove non-aliasing
 * for is contracted into FMAs in one context and not in the other). That is ~1 ULP
 * here, but it is the last stage before the samples, so it lands straight in the
 * audio. One function, one answer. `stride`/`off` select this item's columns of the
 * flattened batched buffer (per-slot passes stride = len, off = 0). */
static void cs_final_conv(float *audio, const float *signal, int64_t stride, int64_t off,
                          int cur_ch, int len, const float *w, const float *bias,
                          float *tail) {
    if (bias)
        for (int t = 0; t < len; t++) audio[t] = bias[0];
    for (int ic = 0; ic < cur_ch; ic++) {
        const float *src = signal + (int64_t)ic * stride + off;
        const float *wc  = w + (int64_t)ic * 7;
        const float *tc  = tail + (int64_t)ic * 6;
        for (int t = 0; t < len; t++) {
            float sum = 0;
            for (int k = 0; k < 7; k++) {
                int p = t - 6 + k;
                sum += wc[k] * (p >= 0 ? src[p] : tc[6 + p]);
            }
            audio[t] += sum;
        }
    }
    /* New tail = the last 6 (snake-activated) inputs of this item. */
    for (int ic = 0; ic < cur_ch; ic++) {
        float *tc = tail + (int64_t)ic * 6;
        const float *src = signal + (int64_t)ic * stride + off;
        if (len >= 6) memcpy(tc, src + len - 6, 6 * sizeof(float));
        else
            for (int i = 0; i < 6; i++)
                tc[i] = (i + len < 6) ? tc[i + len] : src[i + len - 6];
    }
}

/* Lazily allocate (zeroed) per-slot streaming conv state. */
static int cs_ensure_alloc(qwen_sd_stream_state_t *st) {
    if (st->cs_alloc) return 0;
    static const int up_out_ch[4] = {768, 384, 192, 96};
    static const int up_rate[4]   = {8, 5, 4, 3};
    static const int dils[3]      = {1, 3, 9};
    st->cs_cn_dw_tail[0] = (float *)aligned_calloc(1024 * 6, sizeof(float));
    st->cs_cn_dw_tail[1] = (float *)aligned_calloc(1024 * 6, sizeof(float));
    st->cs_init_tail     = (float *)aligned_calloc(1024 * 6, sizeof(float));
    st->cs_final_tail    = (float *)aligned_calloc(96 * 6, sizeof(float));
    if (!st->cs_cn_dw_tail[0] || !st->cs_cn_dw_tail[1] || !st->cs_init_tail || !st->cs_final_tail)
        return -1;
    for (int b = 0; b < 4; b++) {
        st->cs_up_carry[b] = (float *)aligned_calloc((int64_t)up_out_ch[b] * up_rate[b],
                                                     sizeof(float));
        if (!st->cs_up_carry[b]) return -1;
        for (int r = 0; r < 3; r++) {
            st->cs_res_tail[b][r] = (float *)aligned_calloc((int64_t)up_out_ch[b] * 6 * dils[r],
                                                            sizeof(float));
            if (!st->cs_res_tail[b][r]) return -1;
        }
    }
    st->cs_alloc = 1;
    return 0;
}


/* Streaming conv decoder: consumes `signal` [1024 x m] (takes ownership),
 * emits exactly m*1920 samples. Mirrors conv_decoder_forward stage by stage. */
static int conv_decoder_forward_streaming(qwen_tts_ctx_t *ctx, qwen_sd_stream_state_t *st,
                                          float *signal, int m,
                                          float **audio_out, int *n_samples_out) {
    qwen_speech_decoder_t *sd = &ctx->speech_dec;
    if (cs_ensure_alloc(st) != 0) { free(signal); return -1; }
    int cur_ch = 1024, cur_len = m;
    const int _ph = sd_phase_on();
    double _m = _ph ? sd_ph_now() : 0.0;

    /* ConvNeXt upsample (2 blocks, 2x each). k=2,s=2 → no overlap, carry-free. */
    for (int b = 0; b < 2; b++) {
        qwen_sd_convnext_t *cn = &sd->convnext[b];
        if (!cn->conv_weight) { free(signal); return -1; }
        float *up = cs_convt(signal, cur_ch, cur_ch, cur_len, 2, 2,
                             cn->conv_weight, cn->conv_bias, NULL);
        free(signal);
        if (!up) return -1;
        cur_len *= 2;
        float *dw = cs_dwconv(up, cur_ch, cur_len, cn->dwconv_weight, cn->dwconv_bias,
                              st->cs_cn_dw_tail[b]);
        if (!dw) { free(up); return -1; }
        convnext_mlp(cn, dw, up, cur_ch, cur_len);
        free(up);
        signal = dw;
    }

    if (_ph) { double _t = sd_ph_now(); sd_p6a += _t - _m; _m = _t; }

    /* Initial conv (1024→1536, k=7) */
    if (!sd->initial_conv_weight) { free(signal); return -1; }
    float *ic_out = cs_conv1d(signal, cur_ch, 1536, cur_len, 7, 1,
                              sd->initial_conv_weight, sd->initial_conv_bias,
                              st->cs_init_tail, st->cs_warm);
    free(signal);
    if (!ic_out) return -1;
    signal = ic_out; cur_ch = 1536;
    if (_ph) { double _t = sd_ph_now(); sd_p6b += _t - _m; _m = _t; }

    /* 4 Decoder upsample blocks */
    int up_rates[4] = {8, 5, 4, 3};
    int out_channels[4] = {768, 384, 192, 96};

    for (int b = 0; b < 4; b++) {
        qwen_sd_upsample_block_t *ub = &sd->upsample_blocks[b];
        int rate = up_rates[b];
        int kernel = rate * 2;
        int out_ch = out_channels[b];
        if (!ub->upsample.conv_weight) { free(signal); return -1; }

        if (ub->upsample.snake_alpha && ub->upsample.snake_beta)
            snake_activation(signal, cur_ch, cur_len, ub->upsample.snake_alpha, ub->upsample.snake_beta);

        float *up_out = cs_convt(signal, cur_ch, out_ch, cur_len, kernel, rate,
                                 ub->upsample.conv_weight, ub->upsample.conv_bias,
                                 st->cs_up_carry[b]);
        free(signal);
        if (!up_out) return -1;
        signal = up_out; cur_ch = out_ch; cur_len *= rate;

        int dilations[3] = {1, 3, 9};
        for (int r = 0; r < 3; r++) {
            int dil = dilations[r];
            float *res = (float *)aligned_malloc((int64_t)cur_ch * cur_len * sizeof(float));
            if (!res) { free(signal); return -1; }
            memcpy(res, signal, (int64_t)cur_ch * cur_len * sizeof(float));

            if (ub->res_blocks[r].snake1_alpha && ub->res_blocks[r].snake1_beta)
                snake_activation(signal, cur_ch, cur_len,
                                 ub->res_blocks[r].snake1_alpha, ub->res_blocks[r].snake1_beta);

            float *c1_out = cs_conv1d(signal, cur_ch, cur_ch, cur_len, 7, dil,
                                      ub->res_blocks[r].conv1_weight, ub->res_blocks[r].conv1_bias,
                                      st->cs_res_tail[b][r], st->cs_warm);
            free(signal);
            if (!c1_out) { free(res); return -1; }
            signal = c1_out;

            if (ub->res_blocks[r].snake2_alpha && ub->res_blocks[r].snake2_beta)
                snake_activation(signal, cur_ch, cur_len,
                                 ub->res_blocks[r].snake2_alpha, ub->res_blocks[r].snake2_beta);

            /* conv2 is k=1: stateless */
            float *c2_out = (float *)aligned_calloc((int64_t)cur_ch * cur_len, sizeof(float));
            if (!c2_out) { free(res); free(signal); return -1; }
            causal_conv1d(c2_out, signal, ub->res_blocks[r].conv2_weight, ub->res_blocks[r].conv2_bias,
                          cur_ch, cur_ch, cur_len, 1, 1);

            for (int64_t i = 0; i < (int64_t)cur_ch * cur_len; i++)
                signal[i] = res[i] + c2_out[i];
            free(c2_out);
            free(res);
        }
    }

    /* Final Snake + Conv (96→1, k=7) with a 6-column input tail */
    if (!sd->final_snake.alpha || !sd->final_conv_weight) { free(signal); return -1; }
    snake_activation(signal, cur_ch, cur_len, sd->final_snake.alpha, sd->final_snake.beta);

    float *audio = (float *)aligned_calloc(cur_len, sizeof(float));
    if (!audio) { free(signal); return -1; }
    cs_final_conv(audio, signal, cur_len, 0, cur_ch, cur_len,
                  sd->final_conv_weight, sd->final_conv_bias, st->cs_final_tail);
    free(signal);

    for (int i = 0; i < cur_len; i++) {
        if (audio[i] < -1.0f) audio[i] = -1.0f;
        if (audio[i] > 1.0f) audio[i] = 1.0f;
    }

    st->cs_warm = 1;   /* tails now hold real context; later chunks must prepend them */

    if (_ph) sd_p6c += sd_ph_now() - _m;
    *audio_out = audio;
    *n_samples_out = cur_len;
    return 0;
}

/* Use the exact stateful conv decoder unless the caller opts back into the old
 * windowed re-decode, or the CUDA-resident conv decoder owns the conv stack
 * (which the stateful path does not implement). */
static int sd_exact_stream_enabled(void) {
#ifdef QWEN_HAVE_CUDA
    if (g_cuda_decoder_conv_on) return 0;
#endif
    const char *e = getenv("QWEN_SD_WINDOWED");
    return !(e && *e && *e != '0');
}

/* Incremental streaming decode: process only new_frames through VQ→pre-transformer
 * (using KV cache), cache latent output, run conv decoder on windowed latent.
 * Returns only NEW audio samples (not previously emitted ones). */
/* Explicit-state variant: streams using the caller-owned `st` instead of the
 * single ctx->sd_stream. Lets the continuous-batching driver keep B independent
 * per-slot streaming decoder states (weights in ctx->speech_dec + ctx->config are
 * read-only, shared safely). */
int qwen_speech_decoder_decode_streaming_st(qwen_tts_ctx_t *ctx, qwen_sd_stream_state_t *st,
                                          const int *new_codes, int new_frames,
                                          float **audio_out, int *n_samples) {
    qwen_mm_component(QWEN_COMP_DECODER);   /* MAC audit: the streaming paths too, or their work lands in "other" */
    const int  _ph    = sd_phase_on();
    const double _ph_call0 = _ph ? sd_ph_now() : 0.0;
    double _ph_p[6] = {0,0,0,0,0,0}, _ph_mark = 0.0;
    if (_ph) { sd_p6a = sd_p6b = sd_p6c = 0.0; }
    qwen_speech_decoder_t *sd = &ctx->speech_dec;
    qwen_tts_config_t *c = &ctx->config;

    int cb_dim = QWEN_TTS_CODEBOOK_DIM;
    int vq_hidden = 512;
    int latent_dim = 1024;
    int dec_hidden = 512;
    int dec_inter = 1024;
    int n_heads = 16;
    int head_dim = c->dec_head_dim;
    int qkv_dim = n_heads * head_dim;
    int window = 72;
    float eps = c->dec_rms_norm_eps;
    int half_hd = head_dim / 2;

    if (_ph) _ph_mark = sd_ph_now();
    /* === Step 1: VQ dequant for new frames only === */
    /* Output: vq_out row-major [new_frames, 512] */
    float *vq_out = (float *)aligned_calloc((int64_t)new_frames * vq_hidden, sizeof(float));
    float *cb_sum = (float *)aligned_malloc(cb_dim * sizeof(float));

    for (int f = 0; f < new_frames; f++) {
        int code0 = new_codes[f * 16];
        if (code0 >= 0 && code0 < c->codebook_size && sd->codebook[0]) {
            const float *emb = sd->codebook[0] + (int64_t)code0 * cb_dim;
            if (sd->rvq_first_output_proj) {
                for (int o = 0; o < vq_hidden; o++) {
                    float sum = 0;
                    for (int i = 0; i < cb_dim; i++)
                        sum += sd->rvq_first_output_proj[(int64_t)o * cb_dim + i] * emb[i];
                    vq_out[(int64_t)f * vq_hidden + o] += sum;
                }
            }
        }
        memset(cb_sum, 0, cb_dim * sizeof(float));
        for (int k = 1; k < 16; k++) {
            int code = new_codes[f * 16 + k];
            if (code >= 0 && code < c->codebook_size && sd->codebook[k]) {
                const float *emb = sd->codebook[k] + (int64_t)code * cb_dim;
                for (int d = 0; d < cb_dim; d++) cb_sum[d] += emb[d];
            }
        }
        if (sd->rvq_rest_output_proj) {
            for (int o = 0; o < vq_hidden; o++) {
                float sum = 0;
                for (int i = 0; i < cb_dim; i++)
                    sum += sd->rvq_rest_output_proj[(int64_t)o * cb_dim + i] * cb_sum[i];
                vq_out[(int64_t)f * vq_hidden + o] += sum;
            }
        }
    }
    free(cb_sum);

    if (_ph) { double _t = sd_ph_now(); _ph_p[0] += _t - _ph_mark; _ph_mark = _t; }
    /* === Step 2: Pre-conv on new frames with padding from previous chunk === */
    /* VQ output is row-major [new_frames, 512]. Transpose to channel-first [512, new_frames]
     * for conv1d, prepending 2 frames of padding. */
    int pad_frames = st->vq_pad_valid ? 2 : 0;
    int conv_in_len = pad_frames + new_frames;
    float *vq_cf = (float *)aligned_calloc((int64_t)vq_hidden * conv_in_len, sizeof(float));

    /* Copy padding */
    if (st->vq_pad_valid && st->vq_pad) {
        for (int ch = 0; ch < vq_hidden; ch++)
            for (int t = 0; t < 2; t++)
                vq_cf[(int64_t)ch * conv_in_len + t] = st->vq_pad[(int64_t)ch * 2 + t];
    }
    /* Copy new VQ output (transpose row→channel-first) */
    for (int f = 0; f < new_frames; f++)
        for (int ch = 0; ch < vq_hidden; ch++)
            vq_cf[(int64_t)ch * conv_in_len + pad_frames + f] = vq_out[(int64_t)f * vq_hidden + ch];

    /* Save last 2 frames of VQ output (channel-first) as padding for next chunk */
    if (!st->vq_pad) st->vq_pad = (float *)aligned_malloc(vq_hidden * 2 * sizeof(float));
    int save_start = (conv_in_len >= 2) ? conv_in_len - 2 : 0;
    int save_count = (conv_in_len >= 2) ? 2 : conv_in_len;
    for (int ch = 0; ch < vq_hidden; ch++)
        for (int t = 0; t < save_count; t++)
            st->vq_pad[(int64_t)ch * 2 + (2 - save_count + t)] =
                vq_cf[(int64_t)ch * conv_in_len + save_start + t];
    if (save_count < 2) {
        /* Zero-fill the earlier positions if we had fewer than 2 frames */
        for (int ch = 0; ch < vq_hidden; ch++)
            for (int t = 0; t < 2 - save_count; t++)
                st->vq_pad[(int64_t)ch * 2 + t] = 0;
    }
    st->vq_pad_valid = 1;
    free(vq_out);

    /* Pre-conv (512→1024, k=3, causal, pad_left=2) */
    float *pre_conv_out = (float *)aligned_calloc((int64_t)latent_dim * conv_in_len, sizeof(float));
    causal_conv1d(pre_conv_out, vq_cf, sd->pre_conv_weight, sd->pre_conv_bias,
                  vq_hidden, latent_dim, conv_in_len, 3, 1);
    free(vq_cf);

    /* Take only the last new_frames from pre_conv output */
    /* The first pad_frames outputs may have been computed with actual previous context */

    if (_ph) { double _t = sd_ph_now(); _ph_p[1] += _t - _ph_mark; _ph_mark = _t; }
    /* === Step 3: Input proj on new frames (1024→512, row-major) === */
    float *hidden = (float *)aligned_malloc((int64_t)new_frames * dec_hidden * sizeof(float));
#ifdef USE_BLAS
    /* Transpose new portion [1024, new_frames] → [new_frames, 1024] */
    float *pre_conv_rm = (float *)aligned_malloc((int64_t)new_frames * latent_dim * sizeof(float));
    for (int f = 0; f < new_frames; f++)
        for (int d = 0; d < latent_dim; d++)
            pre_conv_rm[(int64_t)f * latent_dim + d] = pre_conv_out[(int64_t)d * conv_in_len + pad_frames + f];
    free(pre_conv_out);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                new_frames, dec_hidden, latent_dim, 1.0f,
                pre_conv_rm, latent_dim,
                sd->input_proj_weight, latent_dim,
                0.0f, hidden, dec_hidden);
    free(pre_conv_rm);
    if (sd->input_proj_bias)
        for (int f = 0; f < new_frames; f++)
            for (int o = 0; o < dec_hidden; o++)
                hidden[(int64_t)f * dec_hidden + o] += sd->input_proj_bias[o];
#else
    for (int f = 0; f < new_frames; f++) {
        for (int o = 0; o < dec_hidden; o++) {
            float sum = sd->input_proj_bias ? sd->input_proj_bias[o] : 0;
            for (int i = 0; i < latent_dim; i++)
                sum += sd->input_proj_weight[(int64_t)o * latent_dim + i]
                     * pre_conv_out[(int64_t)i * conv_in_len + pad_frames + f];
            hidden[(int64_t)f * dec_hidden + o] = sum;
        }
    }
    free(pre_conv_out);
#endif

    if (_ph) { double _t = sd_ph_now(); _ph_p[2] += _t - _ph_mark; _ph_mark = _t; }
    /* === Step 4: Pre-transformer with KV cache === */
    /* plan_v4 D2: cap the KV cache at O(window+chunk) instead of the full stream.
     * Physical frames needed = (kv_len + new_frames) - kv_base. When that exceeds
     * the allocation, first COMPACT: the next queries only read back to
     * kv_len-window+1, so drop everything older and memmove the live tail (<window
     * frames) to physical slot 0, rebasing kv_base. Only if a single chunk is
     * larger than the cap do we then grow (rare). physical index = abs - kv_base. */
    int need = (st->kv_len + new_frames) - st->kv_base;
    if (need > st->kv_alloc) {
        int keep_from = st->kv_len - (window - 1);
        if (keep_from < 0) keep_from = 0;
        if (keep_from > st->kv_base && st->kv_len > st->kv_base) {
            int keep = st->kv_len - keep_from;          /* live tail, < window */
            int shift = keep_from - st->kv_base;        /* physical frames dropped */
            for (int l = 0; l < c->dec_num_layers; l++) {
                if (keep > 0) {
                    memmove(st->k_cache[l], st->k_cache[l] + (int64_t)shift * qkv_dim,
                            (int64_t)keep * qkv_dim * sizeof(float));
                    memmove(st->v_cache[l], st->v_cache[l] + (int64_t)shift * qkv_dim,
                            (int64_t)keep * qkv_dim * sizeof(float));
                }
            }
            st->kv_base = keep_from;
            need = (st->kv_len + new_frames) - st->kv_base;
        }
        if (need > st->kv_alloc) {                        /* chunk bigger than cap */
            int new_alloc = need + 256;
            for (int l = 0; l < c->dec_num_layers; l++) {
                st->k_cache[l] = (float *)realloc(st->k_cache[l], (int64_t)new_alloc * qkv_dim * sizeof(float));
                st->v_cache[l] = (float *)realloc(st->v_cache[l], (int64_t)new_alloc * qkv_dim * sizeof(float));
            }
            st->kv_alloc = new_alloc;
        }
    }

    float *q = (float *)aligned_malloc((int64_t)new_frames * qkv_dim * sizeof(float));
    float *new_k = (float *)aligned_malloc((int64_t)new_frames * qkv_dim * sizeof(float));
    float *new_v = (float *)aligned_malloc((int64_t)new_frames * qkv_dim * sizeof(float));
    float *x_norm = (float *)aligned_malloc((int64_t)new_frames * dec_hidden * sizeof(float));
    float *attn_out = (float *)aligned_malloc((int64_t)new_frames * qkv_dim * sizeof(float));

    for (int layer = 0; layer < c->dec_num_layers; layer++) {
        qwen_sd_pre_layer_t *l = &sd->pre_layers[layer];

        /* Input RMSNorm (NEON-optimized) */
        qwen_rms_norm(x_norm, hidden, l->attn_norm, new_frames, dec_hidden, eps);

        /* QKV projections for new frames */
#ifdef USE_BLAS
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    new_frames, qkv_dim, dec_hidden, 1.0f,
                    x_norm, dec_hidden, l->attn_q, dec_hidden, 0.0f, q, qkv_dim);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    new_frames, qkv_dim, dec_hidden, 1.0f,
                    x_norm, dec_hidden, l->attn_k, dec_hidden, 0.0f, new_k, qkv_dim);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    new_frames, qkv_dim, dec_hidden, 1.0f,
                    x_norm, dec_hidden, l->attn_v, dec_hidden, 0.0f, new_v, qkv_dim);
#else
        for (int s = 0; s < new_frames; s++) {
            const float *xs = x_norm + s * dec_hidden;
            float *qs = q + s * qkv_dim;
            float *ks = new_k + s * qkv_dim;
            float *vs = new_v + s * qkv_dim;
            for (int o = 0; o < qkv_dim; o++) {
                float sum_q = 0, sum_k = 0, sum_v = 0;
                for (int i = 0; i < dec_hidden; i++) {
                    sum_q += l->attn_q[(int64_t)o * dec_hidden + i] * xs[i];
                    sum_k += l->attn_k[(int64_t)o * dec_hidden + i] * xs[i];
                    sum_v += l->attn_v[(int64_t)o * dec_hidden + i] * xs[i];
                }
                qs[o] = sum_q; ks[o] = sum_k; vs[o] = sum_v;
            }
        }
#endif

        /* NeoX split-half RoPE using absolute positions */
        for (int s = 0; s < new_frames; s++) {
            int abs_pos = st->kv_len + s;
            const float *cos_ptr = sd->rope_cos + abs_pos * half_hd;
            const float *sin_ptr = sd->rope_sin + abs_pos * half_hd;
            for (int h = 0; h < n_heads; h++) {
                float *qh = q + s * qkv_dim + h * head_dim;
                float *kh = new_k + s * qkv_dim + h * head_dim;
                for (int i = 0; i < half_hd; i++) {
                    float q1 = qh[i], q2 = qh[i + half_hd];
                    float k1 = kh[i], k2 = kh[i + half_hd];
                    float co = cos_ptr[i], si = sin_ptr[i];
                    qh[i]           = q1 * co - q2 * si;
                    qh[i + half_hd] = q2 * co + q1 * si;
                    kh[i]           = k1 * co - k2 * si;
                    kh[i + half_hd] = k2 * co + k1 * si;
                }
            }
        }

        /* Append new K, V to cache for this layer (physical = abs - kv_base) */
        memcpy(st->k_cache[layer] + (int64_t)(st->kv_len - st->kv_base) * qkv_dim,
               new_k, (int64_t)new_frames * qkv_dim * sizeof(float));
        memcpy(st->v_cache[layer] + (int64_t)(st->kv_len - st->kv_base) * qkv_dim,
               new_v, (int64_t)new_frames * qkv_dim * sizeof(float));

        /* Sliding window causal attention: Q from new frames, K/V from cache */
        float scale = 1.0f / sqrtf((float)head_dim);
        for (int sq = 0; sq < new_frames; sq++) {
            int abs_sq = st->kv_len + sq;
            float *out = attn_out + sq * qkv_dim;
            memset(out, 0, qkv_dim * sizeof(float));
            int sk_start = (abs_sq - window + 1 > 0) ? abs_sq - window + 1 : 0;
            int sk_end = abs_sq; /* inclusive */

            for (int h = 0; h < n_heads; h++) {
                const float *qh = q + sq * qkv_dim + h * head_dim;
                float *oh = out + h * head_dim;

                int n_keys = sk_end - sk_start + 1;
                float scores_buf[512];
                float *scores = n_keys <= 512 ? scores_buf : (float *)malloc(n_keys * sizeof(float));
                float max_score = -1e30f;
                for (int j = 0; j < n_keys; j++) {
                    int sk = sk_start + j;
                    const float *kh = st->k_cache[layer] + (int64_t)(sk - st->kv_base) * qkv_dim + h * head_dim;
                    float dot = 0;
                    for (int d = 0; d < head_dim; d++) dot += qh[d] * kh[d];
                    scores[j] = dot * scale;
                    if (scores[j] > max_score) max_score = scores[j];
                }

                float sum_exp = 0;
                for (int j = 0; j < n_keys; j++) {
                    scores[j] = expf(scores[j] - max_score);
                    sum_exp += scores[j];
                }
                float inv_sum = 1.0f / sum_exp;

                for (int j = 0; j < n_keys; j++) {
                    int sk = sk_start + j;
                    const float *vh = st->v_cache[layer] + (int64_t)(sk - st->kv_base) * qkv_dim + h * head_dim;
                    float w = scores[j] * inv_sum;
                    for (int d = 0; d < head_dim; d++) oh[d] += vh[d] * w;
                }
                if (scores != scores_buf) free(scores);
            }
        }

        /* Output proj + layer_scale + residual */
#ifdef USE_BLAS
        {
            float *oproj = x_norm;
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        new_frames, dec_hidden, qkv_dim, 1.0f,
                        attn_out, qkv_dim, l->attn_o, qkv_dim,
                        0.0f, oproj, dec_hidden);
            for (int s = 0; s < new_frames; s++) {
                float *xs = hidden + s * dec_hidden;
                float *ps = oproj + s * dec_hidden;
                if (l->attn_layer_scale) {
                    for (int o = 0; o < dec_hidden; o++) xs[o] += ps[o] * l->attn_layer_scale[o];
                } else {
                    for (int o = 0; o < dec_hidden; o++) xs[o] += ps[o];
                }
            }
        }
#else
        for (int s = 0; s < new_frames; s++) {
            float *xs = hidden + s * dec_hidden;
            const float *attn = attn_out + s * qkv_dim;
            for (int o = 0; o < dec_hidden; o++) {
                float sum = 0;
                for (int i = 0; i < qkv_dim; i++)
                    sum += l->attn_o[(int64_t)o * qkv_dim + i] * attn[i];
                if (l->attn_layer_scale) sum *= l->attn_layer_scale[o];
                xs[o] += sum;
            }
        }
#endif

        /* Post-attn RMSNorm (NEON-optimized) */
        qwen_rms_norm(x_norm, hidden, l->ffn_norm, new_frames, dec_hidden, eps);

        /* SwiGLU FFN */
#ifdef USE_BLAS
        {
            float *ffn_gate = (float *)aligned_malloc((int64_t)new_frames * dec_inter * sizeof(float));
            float *ffn_up = (float *)aligned_malloc((int64_t)new_frames * dec_inter * sizeof(float));
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        new_frames, dec_inter, dec_hidden, 1.0f,
                        x_norm, dec_hidden, l->ffn_gate, dec_hidden,
                        0.0f, ffn_gate, dec_inter);
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        new_frames, dec_inter, dec_hidden, 1.0f,
                        x_norm, dec_hidden, l->ffn_up, dec_hidden,
                        0.0f, ffn_up, dec_inter);
            for (int64_t i = 0; i < (int64_t)new_frames * dec_inter; i++)
                ffn_gate[i] = (ffn_gate[i] / (1.0f + expf(-ffn_gate[i]))) * ffn_up[i];
            free(ffn_up);
            float *ffn_down_out = (float *)aligned_malloc((int64_t)new_frames * dec_hidden * sizeof(float));
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        new_frames, dec_hidden, dec_inter, 1.0f,
                        ffn_gate, dec_inter, l->ffn_down, dec_inter,
                        0.0f, ffn_down_out, dec_hidden);
            free(ffn_gate);
            for (int s = 0; s < new_frames; s++) {
                float *hs = hidden + s * dec_hidden;
                float *ds = ffn_down_out + s * dec_hidden;
                if (l->ffn_layer_scale) {
                    for (int o = 0; o < dec_hidden; o++) hs[o] += ds[o] * l->ffn_layer_scale[o];
                } else {
                    for (int o = 0; o < dec_hidden; o++) hs[o] += ds[o];
                }
            }
            free(ffn_down_out);
        }
#else
        for (int s = 0; s < new_frames; s++) {
            const float *xs = x_norm + s * dec_hidden;
            float *hs = hidden + s * dec_hidden;
            float gate_up[dec_inter * 2];
            for (int o = 0; o < dec_inter; o++) {
                float sum_g = 0, sum_u = 0;
                for (int i = 0; i < dec_hidden; i++) {
                    sum_g += l->ffn_gate[(int64_t)o * dec_hidden + i] * xs[i];
                    sum_u += l->ffn_up[(int64_t)o * dec_hidden + i] * xs[i];
                }
                gate_up[o] = (sum_g / (1.0f + expf(-sum_g))) * sum_u;
            }
            for (int o = 0; o < dec_hidden; o++) {
                float sum = 0;
                for (int i = 0; i < dec_inter; i++)
                    sum += l->ffn_down[(int64_t)o * dec_inter + i] * gate_up[i];
                if (l->ffn_layer_scale) sum *= l->ffn_layer_scale[o];
                hs[o] += sum;
            }
        }
#endif
    }

    /* Update KV cache length (after all layers processed) */
    st->kv_len += new_frames;

    free(q); free(new_k); free(new_v); free(x_norm); free(attn_out);

    if (_ph) { double _t = sd_ph_now(); _ph_p[3] += _t - _ph_mark; _ph_mark = _t; }
    /* === Step 5: Final RMSNorm + Output proj (512→1024) on new frames === */
    if (sd->final_norm_weight) {
        qwen_rms_norm(hidden, hidden, sd->final_norm_weight, new_frames, dec_hidden, eps);
    }

    /* Grow/compact latent cache (plan_v4 D2): the conv decoder (Step 6) only reads
     * the last conv_rf+chunk frames, so cap the cache instead of keeping the whole
     * stream — same base-offset trim as the KV cache. */
    int lat_need = (st->latent_frames + new_frames) - st->latent_base;
    if (lat_need > st->latent_alloc) {
        int lkeep_from = st->latent_frames - QWEN_SD_STREAM_CONV_RF;  /* context the next chunk needs */
        if (lkeep_from < 0) lkeep_from = 0;
        if (lkeep_from > st->latent_base && st->latent_frames > st->latent_base) {
            int lkeep = st->latent_frames - lkeep_from;
            int lshift = lkeep_from - st->latent_base;
            if (lkeep > 0)
                memmove(st->latent_cache, st->latent_cache + (int64_t)lshift * latent_dim,
                        (int64_t)lkeep * latent_dim * sizeof(float));
            st->latent_base = lkeep_from;
            lat_need = (st->latent_frames + new_frames) - st->latent_base;
        }
        if (lat_need > st->latent_alloc) {
            int new_alloc = lat_need + 256;
            st->latent_cache = (float *)realloc(st->latent_cache,
                (int64_t)new_alloc * latent_dim * sizeof(float));
            st->latent_alloc = new_alloc;
        }
    }

    /* Output proj new frames → append to latent cache [row-major: frames × 1024] */
    float *lat_dst = st->latent_cache + (int64_t)(st->latent_frames - st->latent_base) * latent_dim;
#ifdef USE_BLAS
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                new_frames, latent_dim, dec_hidden, 1.0f,
                hidden, dec_hidden,
                sd->output_proj_weight, dec_hidden,
                0.0f, lat_dst, latent_dim);
    if (sd->output_proj_bias)
        for (int f = 0; f < new_frames; f++)
            for (int o = 0; o < latent_dim; o++)
                lat_dst[(int64_t)f * latent_dim + o] += sd->output_proj_bias[o];
#else
    for (int f = 0; f < new_frames; f++) {
        for (int o = 0; o < latent_dim; o++) {
            float sum = sd->output_proj_bias ? sd->output_proj_bias[o] : 0;
            for (int i = 0; i < dec_hidden; i++)
                sum += sd->output_proj_weight[(int64_t)o * dec_hidden + i] * hidden[(int64_t)f * dec_hidden + i];
            lat_dst[(int64_t)f * latent_dim + o] = sum;
        }
    }
#endif
    st->latent_frames += new_frames;
    free(hidden);

    if (_ph) { double _t = sd_ph_now(); _ph_p[4] += _t - _ph_mark; _ph_mark = _t; }
    /* === Step 6: conv decoder === */
    /* Exact path: feed ONLY the new frames; the per-conv tails/carries in `st`
     * hold everything the causal stack needs from the past. Emits exactly
     * new_frames*1920 samples, equal to the corresponding slice of a one-shot
     * decode. NOTE the physical index must be rebased by latent_base (D2 cache
     * compaction) — the upstream PR reads without it and lands out of bounds
     * once the cache has been trimmed. */
    if (sd_exact_stream_enabled()) {
        float *signal = (float *)aligned_malloc((int64_t)latent_dim * new_frames * sizeof(float));
        if (!signal) return -1;
        const float *lat_new = st->latent_cache
            + (int64_t)(st->latent_frames - new_frames - st->latent_base) * latent_dim;
        for (int f = 0; f < new_frames; f++)
            for (int d = 0; d < latent_dim; d++)
                signal[(int64_t)d * new_frames + f] = lat_new[(int64_t)f * latent_dim + d];

        float *new_audio = NULL;
        int new_samples = 0;
        int ret = conv_decoder_forward_streaming(ctx, st, signal, new_frames,
                                                 &new_audio, &new_samples);
        if (ret != 0) return ret;

        st->frames_decoded += new_frames;
        st->samples_produced += new_samples;
        *audio_out = new_audio;
        *n_samples = new_samples;
        SD_PHASE_EMIT("per-slot", 1, new_frames);
        return 0;
    }

    /* Legacy windowed path (QWEN_SD_WINDOWED=1): re-decode conv_rf context
     * frames per chunk and throw them away. Approximate at the seams. */
    /* Take last (RF + new_frames) from latent cache, or all if fewer */
    int conv_rf = QWEN_SD_STREAM_CONV_RF;
    int window_frames = st->latent_frames;
    int context_frames = 0;
    if (window_frames > conv_rf + new_frames) {
        context_frames = conv_rf;
        window_frames = conv_rf + new_frames;
    } else {
        context_frames = window_frames - new_frames;
    }
    int window_start = st->latent_frames - window_frames;

    /* Transpose window to channel-first [1024, window_frames] for conv decoder */
    float *signal = (float *)aligned_malloc((int64_t)latent_dim * window_frames * sizeof(float));
    const float *lat_src = st->latent_cache + (int64_t)(window_start - st->latent_base) * latent_dim;
    for (int f = 0; f < window_frames; f++)
        for (int d = 0; d < latent_dim; d++)
            signal[(int64_t)d * window_frames + f] = lat_src[(int64_t)f * latent_dim + d];

    /* Run conv decoder (ConvNeXt + initial conv + upsample blocks + final conv) */
    float *full_audio = NULL;
    int full_samples = 0;
    int ret = conv_decoder_forward(ctx, signal, latent_dim, window_frames,
                                    &full_audio, &full_samples);
    if (ret != 0) return ret;

    /* Extract only the new audio (skip context portion) */
    /* Audio is exactly 1920 samples per latent frame */
    int context_samples = context_frames * 1920;
    int new_samples = full_samples - context_samples;
    if (new_samples <= 0) {
        free(full_audio);
        *audio_out = NULL;
        *n_samples = 0;
        return 0;
    }

    float *new_audio = (float *)aligned_malloc(new_samples * sizeof(float));
    memcpy(new_audio, full_audio + context_samples, new_samples * sizeof(float));
    free(full_audio);

    st->frames_decoded += new_frames;
    st->samples_produced += new_samples;

    *audio_out = new_audio;
    *n_samples = new_samples;
    SD_PHASE_EMIT("per-slot-windowed", 1, new_frames);
    return 0;
}

/* Back-compat wrapper: stream using the single ctx->sd_stream state. */
int qwen_speech_decoder_decode_streaming(qwen_tts_ctx_t *ctx,
                                          const int *new_codes, int new_frames,
                                          float **audio_out, int *n_samples) {
    qwen_mm_component(QWEN_COMP_DECODER);   /* MAC audit: the streaming paths too, or their work lands in "other" */
    return qwen_speech_decoder_decode_streaming_st(ctx, &ctx->sd_stream,
                                                   new_codes, new_frames, audio_out, n_samples);
}

/* ==========================================================================
 * CROSS-SLOT BATCHED STREAMING DECODE   (server flag: QWEN_DECODER_BATCH=1)
 *
 * THE PROBLEM THIS SOLVES. The batched Talker and Code Predictor read each weight
 * ONCE for every slot the scheduler steps. The speech decoder did not: it was called
 * once per slot, in sequence, so it re-read the entire decoder weight set (pre-
 * transformer + ConvNeXt + 4 upsample blocks, hundreds of MB in fp32) N times for N
 * slots. That is why QWEN_SERVE_PROFILE=1 puts it at 58.5% of the streaming scheduler
 * loop and 29.8% of the WAV one at c=4 - more than the Code Predictor, more than the
 * Talker. It is the same defect vllm-omni#3163 found in Code2Wav: a per-request
 * for-loop de-batching what the scheduler had already batched.
 *
 * WHAT IS BATCHED. The math. Every GEMM and every convolution runs ONCE over the
 * concatenation of all the items' frames, so each weight is read once per batch
 * instead of once per slot. The batch is the leading (N) dimension of the flattened
 * time axis of the channel-first buffers: buf[c * total + off[b] + t].
 *
 * WHAT IS NOT BATCHED, AND MUST NOT BE. The state. Every item keeps its own
 * qwen_sd_stream_state_t - conv input tails, ConvTranspose overlap-add carries,
 * vq_pad, the pre-transformer KV cache, the absolute RoPE positions. Those are read
 * and written per item, inside the batched stages. Sharing any of them would splice
 * two speakers together, which is exactly the failure dec_busy[] exists to prevent
 * one level up.
 *
 * ---------------------------------------------------------------------------
 * RAGGED STRATEGY: CONCATENATE, NEVER PAD.  Worst-case wasted work: ZERO frames.
 *
 * With the ramped chunking (QWEN_STREAM_DECODE_CHUNK, ramp 1 -> 2 -> 4 -> 8) slots
 * reach their decode trigger at different times and with different frame counts, so
 * a batch in which every slot has the same nframes is the exception, not the rule.
 * Two strategies were on the table:
 *
 *   (a) group only items with IDENTICAL nframes. Rejected, and not on taste: in
 *       steady state every slot uses the same chunk size (8) but fires on a
 *       different iteration, because its phase is set by when the request arrived.
 *       Slots at independent phase co-fire in the same iteration with probability
 *       ~(1/8) per extra slot, so the batch would almost always be B=1 - the change
 *       would measure as "no effect" and we would conclude the wrong thing.
 *
 *   (b) pad every item up to max(nframes) and throw the tail away. Not merely
 *       wasteful - IMPOSSIBLE here: the padding frames do not exist yet (their codes
 *       have not been generated), and feeding invented codes would advance the conv
 *       tails, carries and KV cache of a *stateful* decoder with garbage. There is no
 *       cheap rollback.
 *
 * So: ragged concatenation. Item b owns columns [off[b], off[b]+len[b]) of a single
 * flattened buffer; the im2col of every conv is built per item, from that item's own
 * columns and its own tail, and then ONE GEMM covers all of them. No padded frame is
 * ever computed, so the wasted work is exactly zero frames in the worst case. What
 * the strategy costs instead is bookkeeping (per-item offsets at every stage) and the
 * per-item tail/carry handling - which is the same work the per-slot path already did,
 * not extra work.
 *
 * The phase problem of (a) is then solved in the CALLER, not here (qwen_tts.c: once
 * any slot fires, the others with frames pending join the same batch), which is why
 * this function has to accept ragged input in the first place.
 * ---------------------------------------------------------------------------
 *
 * WHAT WOULD FALSIFY THIS, AND WHAT WAS MEASURED. The audio must not move:
 * tests/decoder_batch_parity.c decodes the same codes both ways and reports the max
 * abs sample difference, and `make test-serve-stream-batch` must stay at corr=1.00000
 * with an exact sample count. Both hold. Most streams come out at exactly 0.
 *
 * The residue that is left is NOT the batching, and it is worth writing down because
 * the obvious reading ("fp noise, whatever") is wrong twice over:
 *   - Accelerate routes an sgemm with M==1 through a GEMV kernel whose K-reduction
 *     order differs from the M>1 kernel (M=1 vs M=5, N=512, K=1024 -> 2.9e-05;
 *     M=2..16 -> exactly 0). That fires on the ramp's ONE-frame first chunk.
 *   - and N is not neutral either at every shape: M=768, K=5376 gives exactly 0 for
 *     N=256/262/274/320/608, but 2.4e-04 for N=310 - which is exactly the width the
 *     per-slot conv1d asks for (256 columns + a 54-column dilation-9 tail).
 * Both are properties of the BLAS that the PER-SLOT path already has: change its chunk
 * size and its own last bits move too. End to end that is 8.1e-06 with 1-frame chunks
 * and 3.1e-07 without, against a 16-bit LSB of 3.05e-05.
 *
 * TWO DEFECTS THIS FOUND THAT WERE NOT ARITHMETIC AT ALL, and the rule that came out
 * of them: the depthwise conv and the final conv were first written as strided copies
 * of the per-slot loops. Identical source, different codegen - through an
 * out-parameter the compiler cannot prove non-aliasing and does not contract the
 * k-loop into FMAs, while in the per-slot version the output buffer is malloc'd
 * locally and it does. One ULP, amplified 480x by the upsampling stack into ~1e-05 on
 * the audio. Hence cs_dwconv() and cs_final_conv() are now called by BOTH paths.
 * **When two paths must agree to the last bit, share the function - do not copy the
 * loop.** (Cost of that decision, stated: cs_dwconv is now called from two sites, so
 * the compiler no longer inlines it into the per-slot decoder, and the per-slot path's
 * own last bits shifted by ~1 ULP relative to the previous build. The gate that
 * matters - corr and sample count against a single-stream reference - is unchanged.)
 *
 * NOT BATCHED (falls back to the per-slot loop, by design):
 *   - n_items == 1                  -> bit-identical by construction
 *   - QWEN_SD_WINDOWED=1            -> the legacy approximate path
 *   - QWEN_SD_INT8=1                -> the int8 conv kernel is per-call, not ragged
 *   - a CUDA-resident conv decoder  -> owns the conv stack
 *   - builds without USE_BLAS       -> the ragged helpers are sgemm-based
 * ========================================================================== */

#ifdef USE_BLAS

/* Ragged time axis over `n` items: item b owns [off[b], off[b]+len[b]). */
typedef struct {
    int      n;
    int     *len;
    int64_t *off;
    int64_t  total;
} sd_rag_t;

static int rag_alloc(sd_rag_t *r, int n) {
    r->n = n; r->total = 0;
    r->len = (int *)calloc((size_t)n, sizeof(int));
    r->off = (int64_t *)calloc((size_t)n, sizeof(int64_t));
    return (r->len && r->off) ? 0 : -1;
}
static void rag_recompute(sd_rag_t *r) {
    int64_t o = 0;
    for (int i = 0; i < r->n; i++) { r->off[i] = o; o += r->len[i]; }
    r->total = o;
}
static void rag_free(sd_rag_t *r) { free(r->len); free(r->off); r->len = NULL; r->off = NULL; }

/* cs_save_tail for one item inside a flattened [in_ch x total] buffer (row stride is
 * `total`, not `len`). Same semantics otherwise: new tail = last tail_cols columns of
 * [old tail | this item's chunk]. */
static void rag_save_tail(float *tail, const float *in, int in_ch,
                          int64_t total, int64_t off, int len, int tail_cols) {
    if (len >= tail_cols) {
        for (int ic = 0; ic < in_ch; ic++)
            memcpy(tail + (int64_t)ic * tail_cols,
                   in + (int64_t)ic * total + off + (len - tail_cols),
                   (size_t)tail_cols * sizeof(float));
    } else {
        int keep = tail_cols - len;   /* columns of the old tail that survive */
        for (int ic = 0; ic < in_ch; ic++) {
            float *t = tail + (int64_t)ic * tail_cols;
            memmove(t, t + len, (size_t)keep * sizeof(float));
            memcpy(t + keep, in + (int64_t)ic * total + off, (size_t)len * sizeof(float));
        }
    }
}

/* Ragged batched causal conv1d. Semantically identical to running cs_conv1d() once
 * per item: for output column j of item b the input window is
 *     X[p] = (p >= 0) ? in[ic][off_b + p] : tail_b[ic][tail_cols + p],
 * i.e. the item's own frames backed by its own saved tail (all zeros before the first
 * chunk - the same values causal zero padding produces, which is why the `warm`
 * shortcut of cs_conv1d is unnecessary here and no work is wasted by dropping it).
 * ONE im2col matrix and ONE sgemm cover every item: that is where the per-slot weight
 * re-reads go away. Tails are updated AFTER the GEMM (they are read during it). */
static int rag_conv1d(float *out, const float *in, int in_ch, int out_ch,
                      const sd_rag_t *r, int kernel, int dilation,
                      const float *w, const float *bias, float * const *tails) {
    int64_t total = r->total;
    if (kernel == 1) {
        /* k=1 has no time coupling at all, so the flattened axis IS the batch. */
        SD_GEMM(CblasNoTrans, CblasNoTrans, out_ch, (int)total, in_ch,
                1.0f, w, in_ch, in, (int)total, 0.0f, out, (int)total);
        conv_add_bias(out, bias, out_ch, (int)total);
        return 0;
    }

    int tail_cols = (kernel - 1) * dilation;
    int64_t col_rows = (int64_t)in_ch * kernel;

    /* Same im2col memory cap as the per-slot path; tiles are cut on the FLATTENED
     * axis, and a tile still spans several items, so it costs one weight read for
     * all of them. */
    int64_t max_tile = CONV_TILE_MAX_BYTES / (col_rows * (int64_t)sizeof(float));
    if (max_tile < 1) max_tile = 1;
    if (max_tile > total) max_tile = total;
    float *col = (float *)aligned_malloc(col_rows * max_tile * sizeof(float));
    if (!col) return -1;

    for (int64_t ts = 0; ts < total; ts += max_tile) {
        int64_t tile = (ts + max_tile > total) ? (total - ts) : max_tile;
        memset(col, 0, col_rows * tile * sizeof(float));
        for (int b = 0; b < r->n; b++) {
            int64_t lo = r->off[b] > ts ? r->off[b] : ts;
            int64_t hi = (r->off[b] + r->len[b]) < (ts + tile) ? (r->off[b] + r->len[b]) : (ts + tile);
            if (lo >= hi) continue;
            const float *tl_base = tails ? tails[b] : NULL;
            for (int ic = 0; ic < in_ch; ic++) {
                const float *src = in + (int64_t)ic * total + r->off[b];
                const float *tl  = tl_base ? tl_base + (int64_t)ic * tail_cols : NULL;
                for (int k = 0; k < kernel; k++) {
                    float *cr = col + ((int64_t)ic * kernel + k) * tile;
                    int64_t shift = (int64_t)tail_cols - (int64_t)k * dilation;   /* p = j - shift */
                    for (int64_t gc = lo; gc < hi; gc++) {
                        int64_t p = (gc - r->off[b]) - shift;
                        cr[gc - ts] = (p >= 0) ? src[p] : (tl ? tl[tail_cols + p] : 0.0f);
                    }
                }
            }
        }
        SD_GEMM(CblasNoTrans, CblasNoTrans, out_ch, (int)tile, (int)col_rows,
                1.0f, w, (int)col_rows, col, (int)tile, 0.0f, out + ts, (int)total);
    }
    free(col);
    conv_add_bias(out, bias, out_ch, (int)total);

    if (tails)
        for (int b = 0; b < r->n; b++)
            rag_save_tail(tails[b], in, in_ch, total, r->off[b], r->len[b], tail_cols);
    return 0;
}

/* Ragged batched causal ConvTranspose1d. Mirrors cs_convt() exactly, item by item:
 * run untrimmed (len*stride + (kernel-stride) columns), overlap-add the item's carry
 * into the head, emit len*stride columns with bias, save the new carry. The only
 * thing that changes is that the per-kernel-position sgemm covers ALL items at once.
 * Writes `rout` (len[b] = in len[b]*stride) and returns a fresh [out_ch x rout.total]. */
static float *rag_convt(const float *in, int in_ch, int out_ch,
                        const sd_rag_t *rin, int kernel, int stride,
                        const float *w, const float *bias, float * const *carries,
                        sd_rag_t *rout) {
    int cs = kernel - stride;
    sd_rag_t rfull;
    if (rag_alloc(&rfull, rin->n) != 0) { rag_free(&rfull); return NULL; }
    for (int b = 0; b < rin->n; b++) {
        rout->len[b]  = rin->len[b] * stride;
        rfull.len[b]  = rin->len[b] * stride + cs;
    }
    rag_recompute(rout);
    rag_recompute(&rfull);

    float *full = (float *)aligned_calloc((int64_t)out_ch * rfull.total, sizeof(float));
    float *wk   = (float *)aligned_malloc((int64_t)in_ch * out_ch * sizeof(float));
    float *rk   = (float *)aligned_malloc((int64_t)out_ch * rin->total * sizeof(float));
    float *out  = (float *)aligned_malloc((int64_t)out_ch * rout->total * sizeof(float));
    if (!full || !wk || !rk || !out) {
        free(full); free(wk); free(rk); free(out); rag_free(&rfull); return NULL;
    }

    for (int k = 0; k < kernel; k++) {
        for (int ic = 0; ic < in_ch; ic++)
            for (int oc = 0; oc < out_ch; oc++)
                wk[(int64_t)ic * out_ch + oc] = w[((int64_t)ic * out_ch + oc) * kernel + k];

        SD_GEMM(CblasTrans, CblasNoTrans, out_ch, (int)rin->total, in_ch,
                1.0f, wk, out_ch, in, (int)rin->total, 0.0f, rk, (int)rin->total);

        for (int b = 0; b < rin->n; b++) {
            int ilen = rin->len[b], flen = rfull.len[b];
            for (int oc = 0; oc < out_ch; oc++) {
                const float *src = rk + (int64_t)oc * rin->total + rin->off[b];
                float *dst = full + (int64_t)oc * rfull.total + rfull.off[b];
                for (int t = 0; t < ilen; t++) {
                    int pos = t * stride + k;
                    if (pos < flen) dst[pos] += src[t];
                }
            }
        }
    }

    for (int b = 0; b < rin->n; b++) {
        int olen = rout->len[b];
        for (int oc = 0; oc < out_ch; oc++) {
            float *f = full + (int64_t)oc * rfull.total + rfull.off[b];
            float *o = out  + (int64_t)oc * rout->total + rout->off[b];
            if (carries && cs > 0) {
                float *cr = carries[b] + (int64_t)oc * cs;
                for (int i = 0; i < cs; i++) f[i] += cr[i];
            }
            float bb = bias ? bias[oc] : 0.0f;
            for (int t = 0; t < olen; t++) o[t] = f[t] + bb;
            if (carries && cs > 0) {
                float *cr = carries[b] + (int64_t)oc * cs;
                for (int i = 0; i < cs; i++) cr[i] = f[olen + i];
            }
        }
    }
    free(full); free(wk); free(rk);
    rag_free(&rfull);
    return out;
}

/* Ragged depthwise conv (ConvNeXt, k=7, pad_left=6): 7 weights per channel, so there
 * is nothing to amortise across items - kept per item, on the shared flat buffer. */
static int rag_dwconv(float *out, const float *in, int ch, const sd_rag_t *r,
                      const float *w, const float *b, float * const *tails) {
    /* NOT a batched kernel, and deliberately so: this calls cs_dwconv() — the SAME
     * function the per-slot path calls — on a contiguous copy of each item's columns.
     *
     * WHY THE COPY, since the loop body is trivially portable to a strided view: it
     * was measured. A hand-inlined strided twin of the loop diverged from cs_dwconv in
     * the last bits (the very first ConvNeXt block already showed sum -100.3078361 vs
     * -100.3078340). Same source, different codegen: in cs_dwconv the output buffer is
     * malloc'd inside the function, so the compiler knows it cannot alias the input and
     * contracts the k-loop into FMAs; through an out-parameter it cannot prove that and
     * emits mul+add. One ULP there is amplified by the 480x upsampling stack into ~1e-5
     * on the audio. Calling the identical function removes the whole question.
     *
     * The copy is affordable precisely because there is nothing to batch here: 7 weights
     * per channel, so the win would have been zero anyway, and at this stage the buffer
     * is 1024 x (2..4 * nframes) floats. */
    int64_t total = r->total;
    for (int i = 0; i < r->n; i++) {
        int len = r->len[i];
        if (len <= 0) continue;
        int64_t off = r->off[i];
        float *tmp = (float *)aligned_malloc((int64_t)ch * len * sizeof(float));
        if (!tmp) return -1;
        for (int c = 0; c < ch; c++)
            memcpy(tmp + (int64_t)c * len, in + (int64_t)c * total + off,
                   (size_t)len * sizeof(float));
        float *o = cs_dwconv(tmp, ch, len, w, b, tails[i]);
        free(tmp);
        if (!o) return -1;
        for (int c = 0; c < ch; c++)
            memcpy(out + (int64_t)c * total + off, o + (int64_t)c * len,
                   (size_t)len * sizeof(float));
        free(o);
    }
    return 0;
}

/* Batched conv decoder: consumes `signal` [1024 x rg->total] (takes ownership), emits
 * item b's 1920*nframes[b] samples into audio_out[b]. Stage for stage the same graph
 * as conv_decoder_forward_streaming(), with the ragged batch as the flattened time
 * axis and the per-item tails/carries applied inside each stage. */
static int conv_decoder_forward_streaming_batch(qwen_tts_ctx_t *ctx,
                                                qwen_sd_stream_state_t **sts, int nb,
                                                float *signal, sd_rag_t *rg,
                                                float **audio_out, int *nsamp_out) {
    qwen_speech_decoder_t *sd = &ctx->speech_dec;
    int rc = -1;
    int cur_ch = 1024;
    sd_rag_t rnext;
    float **tails = (float **)calloc((size_t)nb, sizeof(float *));
    if (rag_alloc(&rnext, nb) != 0 || !tails) { rag_free(&rnext); free(tails); free(signal); return -1; }
    for (int b = 0; b < nb; b++) if (cs_ensure_alloc(sts[b]) != 0) goto done;

    const int _ph = sd_phase_on();
    double _m = _ph ? sd_ph_now() : 0.0;

    /* ConvNeXt upsample (2 blocks, 2x each). k=2,s=2 -> no overlap, carry-free. */
    for (int blk = 0; blk < 2; blk++) {
        qwen_sd_convnext_t *cn = &sd->convnext[blk];
        if (!cn->conv_weight) goto done;
        float *up = rag_convt(signal, cur_ch, cur_ch, rg, 2, 2,
                              cn->conv_weight, cn->conv_bias, NULL, &rnext);
        free(signal); signal = NULL;
        if (!up) goto done;
        for (int b = 0; b < nb; b++) rg->len[b] = rnext.len[b];
        rag_recompute(rg);

        float *dw = (float *)aligned_malloc((int64_t)cur_ch * rg->total * sizeof(float));
        if (!dw) { free(up); goto done; }
        for (int b = 0; b < nb; b++) tails[b] = sts[b]->cs_cn_dw_tail[blk];
        if (rag_dwconv(dw, up, cur_ch, rg, cn->dwconv_weight, cn->dwconv_bias, tails) != 0) {
            free(up); free(dw); signal = NULL; goto done;
        }
        /* convnext_mlp is strictly per-timestep, so the flattened axis is safe. */
        convnext_mlp(cn, dw, up, cur_ch, (int)rg->total);
        free(up);
        signal = dw;
    }

    if (_ph) { double _t = sd_ph_now(); sd_p6a += _t - _m; _m = _t; }

    /* Initial conv (1024->1536, k=7) */
    if (!sd->initial_conv_weight) goto done;
    {
        float *ic_out = (float *)aligned_malloc((int64_t)1536 * rg->total * sizeof(float));
        if (!ic_out) goto done;
        for (int b = 0; b < nb; b++) tails[b] = sts[b]->cs_init_tail;
        int crc = rag_conv1d(ic_out, signal, cur_ch, 1536, rg, 7, 1,
                             sd->initial_conv_weight, sd->initial_conv_bias, tails);
        free(signal); signal = ic_out; cur_ch = 1536;
        if (crc != 0) goto done;
    }

    if (_ph) { double _t = sd_ph_now(); sd_p6b += _t - _m; _m = _t; }

    /* 4 decoder upsample blocks */
    {
        static const int up_rates[4]    = {8, 5, 4, 3};
        static const int out_channels[4] = {768, 384, 192, 96};
        static const int dilations[3]   = {1, 3, 9};
        for (int blk = 0; blk < 4; blk++) {
            qwen_sd_upsample_block_t *ub = &sd->upsample_blocks[blk];
            int rate = up_rates[blk], kernel = rate * 2, out_ch = out_channels[blk];
            if (!ub->upsample.conv_weight) goto done;

            if (ub->upsample.snake_alpha && ub->upsample.snake_beta)
                snake_activation(signal, cur_ch, (int)rg->total,
                                 ub->upsample.snake_alpha, ub->upsample.snake_beta);

            for (int b = 0; b < nb; b++) tails[b] = sts[b]->cs_up_carry[blk];
            float *up_out = rag_convt(signal, cur_ch, out_ch, rg, kernel, rate,
                                      ub->upsample.conv_weight, ub->upsample.conv_bias,
                                      tails, &rnext);
            free(signal); signal = NULL;
            if (!up_out) goto done;
            for (int b = 0; b < nb; b++) rg->len[b] = rnext.len[b];
            rag_recompute(rg);
            signal = up_out; cur_ch = out_ch;

            for (int r = 0; r < 3; r++) {
                int dil = dilations[r];
                int64_t nel = (int64_t)cur_ch * rg->total;
                float *res = (float *)aligned_malloc(nel * sizeof(float));
                if (!res) goto done;
                memcpy(res, signal, nel * sizeof(float));

                if (ub->res_blocks[r].snake1_alpha && ub->res_blocks[r].snake1_beta)
                    snake_activation(signal, cur_ch, (int)rg->total,
                                     ub->res_blocks[r].snake1_alpha, ub->res_blocks[r].snake1_beta);

                float *c1_out = (float *)aligned_malloc(nel * sizeof(float));
                if (!c1_out) { free(res); goto done; }
                for (int b = 0; b < nb; b++) tails[b] = sts[b]->cs_res_tail[blk][r];
                int crc = rag_conv1d(c1_out, signal, cur_ch, cur_ch, rg, 7, dil,
                                     ub->res_blocks[r].conv1_weight, ub->res_blocks[r].conv1_bias, tails);
                free(signal); signal = c1_out;
                if (crc != 0) { free(res); goto done; }

                if (ub->res_blocks[r].snake2_alpha && ub->res_blocks[r].snake2_beta)
                    snake_activation(signal, cur_ch, (int)rg->total,
                                     ub->res_blocks[r].snake2_alpha, ub->res_blocks[r].snake2_beta);

                float *c2_out = (float *)aligned_calloc(nel, sizeof(float));
                if (!c2_out) { free(res); goto done; }
                rag_conv1d(c2_out, signal, cur_ch, cur_ch, rg, 1, 1,
                           ub->res_blocks[r].conv2_weight, ub->res_blocks[r].conv2_bias, NULL);

                for (int64_t i = 0; i < nel; i++) signal[i] = res[i] + c2_out[i];
                free(c2_out); free(res);
            }
        }
    }

    /* Final snake + conv (96->1, k=7), per item into its own audio buffer. */
    if (!sd->final_snake.alpha || !sd->final_conv_weight) goto done;
    snake_activation(signal, cur_ch, (int)rg->total, sd->final_snake.alpha, sd->final_snake.beta);
    for (int b = 0; b < nb; b++) {
        int len = rg->len[b];
        int64_t off = rg->off[b];
        float *audio = (float *)aligned_calloc(len > 0 ? len : 1, sizeof(float));
        if (!audio) goto done;
        cs_final_conv(audio, signal, rg->total, off, cur_ch, len,
                      sd->final_conv_weight, sd->final_conv_bias, sts[b]->cs_final_tail);
        for (int i = 0; i < len; i++) {
            if (audio[i] < -1.0f) audio[i] = -1.0f;
            if (audio[i] >  1.0f) audio[i] =  1.0f;
        }
        audio_out[b] = audio;
        nsamp_out[b] = len;
        sts[b]->cs_warm = 1;   /* tails now hold real context */
    }
    if (_ph) sd_p6c += sd_ph_now() - _m;
    rc = 0;

done:
    free(signal);
    rag_free(&rnext);
    free(tails);
    return rc;
}

#endif /* USE_BLAS (ragged batched helpers) */

/* Per-slot fallback used by every "not batched" case listed at the top. */
static int sd_batch_fallback(qwen_tts_ctx_t *ctx, qwen_sd_batch_item_t *it, int n) {
    int rc = 0;
    for (int i = 0; i < n; i++) {
        if (it[i].nframes <= 0) continue;
        it[i].rc = qwen_speech_decoder_decode_streaming_st(ctx, it[i].st, it[i].codes,
                                                          it[i].nframes,
                                                          &it[i].audio, &it[i].n_samples);
        if (it[i].rc != 0) rc = it[i].rc;
    }
    return rc;
}

#ifdef USE_BLAS
/* The batched twin of qwen_speech_decoder_decode_streaming_st(). Stage numbering and
 * comments follow that function one-to-one; only the shapes change (a ragged batch of
 * frames instead of one item's frames) and the state accesses become per item. */
int qwen_speech_decoder_decode_streaming_batch(qwen_tts_ctx_t *ctx,
                                               qwen_sd_batch_item_t *it, int n_items) {
    qwen_mm_component(QWEN_COMP_DECODER);   /* MAC audit: the streaming paths too, or their work lands in "other" */
    const int  _ph    = sd_phase_on();
    const double _ph_call0 = _ph ? sd_ph_now() : 0.0;
    double _ph_p[6] = {0,0,0,0,0,0}, _ph_mark = 0.0;
    for (int i = 0; i < n_items; i++) { it[i].audio = NULL; it[i].n_samples = 0; it[i].rc = 0; }

    /* Compact away empty items; batching one item buys nothing and the per-slot path
     * is the reference, so hand it straight over (this also keeps c=1 bit-identical). */
    int *idx = (int *)calloc((size_t)(n_items > 0 ? n_items : 1), sizeof(int));
    if (!idx) return -1;
    int nb = 0;
    for (int i = 0; i < n_items; i++) if (it[i].nframes > 0) idx[nb++] = i;
    if (nb == 0) { free(idx); return 0; }
    if (nb == 1 || !sd_exact_stream_enabled() || sd_int8_enabled()) {
        /* Handed to the per-slot function, which is a DIFFERENT implementation and emits
         * its own [SDPHASE] line. Emitting here too would count the same call twice. */
        free(idx);
        return sd_batch_fallback(ctx, it, n_items);
    }

    qwen_speech_decoder_t *sd = &ctx->speech_dec;
    qwen_tts_config_t *c = &ctx->config;

    int cb_dim = QWEN_TTS_CODEBOOK_DIM;
    int vq_hidden = 512;
    int latent_dim = 1024;
    int dec_hidden = 512;
    int dec_inter = 1024;
    int n_heads = 16;
    int head_dim = c->dec_head_dim;
    int qkv_dim = n_heads * head_dim;
    int window = 72;
    float eps = c->dec_rms_norm_eps;
    int half_hd = head_dim / 2;

    int rc = -1;
    sd_rag_t fr;                       /* frame-level ragged descriptor */
    float **tails = NULL, **auds = NULL;
    int *ans = NULL;
    qwen_sd_stream_state_t **sts = NULL;
    float *vq_out = NULL, *cb_sum = NULL, *vq_cf = NULL, *pre_conv_out = NULL;
    float *pre_conv_rm = NULL, *hidden = NULL, *q = NULL, *new_k = NULL, *new_v = NULL;
    float *x_norm = NULL, *attn_out = NULL, *lat_tmp = NULL, *signal = NULL;

    int64_t TF = 0;
    if (rag_alloc(&fr, nb) != 0) { rag_free(&fr); free(idx); return -1; }
    sts   = (qwen_sd_stream_state_t **)calloc((size_t)nb, sizeof(*sts));
    tails = (float **)calloc((size_t)nb, sizeof(float *));
    auds  = (float **)calloc((size_t)nb, sizeof(float *));
    ans   = (int *)calloc((size_t)nb, sizeof(int));
    if (!sts || !tails || !auds || !ans) goto done;
    for (int b = 0; b < nb; b++) { fr.len[b] = it[idx[b]].nframes; sts[b] = it[idx[b]].st; }
    rag_recompute(&fr);
    TF = fr.total;

    if (_ph) { sd_p6a = sd_p6b = sd_p6c = 0.0; _ph_mark = sd_ph_now(); }
    /* === Step 1: VQ dequant, per frame (kept scalar and per item: the RVQ projections
     * are 2 x 512x256 MACs per frame, nothing to amortise, and reordering them would
     * move the sums). === */
    vq_out = (float *)aligned_calloc(TF * vq_hidden, sizeof(float));
    cb_sum = (float *)aligned_malloc(cb_dim * sizeof(float));
    if (!vq_out || !cb_sum) goto done;
    for (int b = 0; b < nb; b++) {
        const int *new_codes = it[idx[b]].codes;
        int64_t base = fr.off[b];
        for (int f = 0; f < fr.len[b]; f++) {
            int code0 = new_codes[f * 16];
            if (code0 >= 0 && code0 < c->codebook_size && sd->codebook[0]) {
                const float *emb = sd->codebook[0] + (int64_t)code0 * cb_dim;
                if (sd->rvq_first_output_proj) {
                    for (int o = 0; o < vq_hidden; o++) {
                        float sum = 0;
                        for (int i = 0; i < cb_dim; i++)
                            sum += sd->rvq_first_output_proj[(int64_t)o * cb_dim + i] * emb[i];
                        vq_out[(base + f) * vq_hidden + o] += sum;
                    }
                }
            }
            memset(cb_sum, 0, cb_dim * sizeof(float));
            for (int k = 1; k < 16; k++) {
                int code = new_codes[f * 16 + k];
                if (code >= 0 && code < c->codebook_size && sd->codebook[k]) {
                    const float *emb = sd->codebook[k] + (int64_t)code * cb_dim;
                    for (int d = 0; d < cb_dim; d++) cb_sum[d] += emb[d];
                }
            }
            if (sd->rvq_rest_output_proj) {
                for (int o = 0; o < vq_hidden; o++) {
                    float sum = 0;
                    for (int i = 0; i < cb_dim; i++)
                        sum += sd->rvq_rest_output_proj[(int64_t)o * cb_dim + i] * cb_sum[i];
                    vq_out[(base + f) * vq_hidden + o] += sum;
                }
            }
        }
    }

    if (_ph) { double _t = sd_ph_now(); _ph_p[0] += _t - _ph_mark; _ph_mark = _t; }
    /* === Step 2: pre-conv (512->1024, k=3, causal, pad_left=2) ===
     * vq_pad IS the 2-column input tail of this conv, so it is just rag_conv1d with
     * tails = vq_pad. An unwarmed slot must see zeros there: the per-slot path gets
     * that from pad_frames=0 (pure causal zero padding), here from a zeroed tail. */
    vq_cf = (float *)aligned_malloc((int64_t)vq_hidden * TF * sizeof(float));
    if (!vq_cf) goto done;
    for (int b = 0; b < nb; b++)
        for (int f = 0; f < fr.len[b]; f++)
            for (int ch = 0; ch < vq_hidden; ch++)
                vq_cf[(int64_t)ch * TF + fr.off[b] + f] = vq_out[(fr.off[b] + f) * vq_hidden + ch];
    free(vq_out); vq_out = NULL;

    for (int b = 0; b < nb; b++) {
        if (!sts[b]->vq_pad) {
            sts[b]->vq_pad = (float *)aligned_calloc((int64_t)vq_hidden * 2, sizeof(float));
            if (!sts[b]->vq_pad) goto done;
        }
        if (!sts[b]->vq_pad_valid) memset(sts[b]->vq_pad, 0, (size_t)vq_hidden * 2 * sizeof(float));
        tails[b] = sts[b]->vq_pad;
    }
    pre_conv_out = (float *)aligned_malloc((int64_t)latent_dim * TF * sizeof(float));
    if (!pre_conv_out) goto done;
    if (rag_conv1d(pre_conv_out, vq_cf, vq_hidden, latent_dim, &fr, 3, 1,
                   sd->pre_conv_weight, sd->pre_conv_bias, tails) != 0) goto done;
    free(vq_cf); vq_cf = NULL;
    for (int b = 0; b < nb; b++) sts[b]->vq_pad_valid = 1;

    if (_ph) { double _t = sd_ph_now(); _ph_p[1] += _t - _ph_mark; _ph_mark = _t; }
    /* === Step 3: input proj (1024->512) — one sgemm over all items' frames === */
    hidden = (float *)aligned_malloc(TF * dec_hidden * sizeof(float));
    pre_conv_rm = (float *)aligned_malloc(TF * latent_dim * sizeof(float));
    if (!hidden || !pre_conv_rm) goto done;
    for (int64_t f = 0; f < TF; f++)
        for (int d = 0; d < latent_dim; d++)
            pre_conv_rm[f * latent_dim + d] = pre_conv_out[(int64_t)d * TF + f];
    free(pre_conv_out); pre_conv_out = NULL;
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                (int)TF, dec_hidden, latent_dim, 1.0f,
                pre_conv_rm, latent_dim, sd->input_proj_weight, latent_dim,
                0.0f, hidden, dec_hidden);
    free(pre_conv_rm); pre_conv_rm = NULL;
    if (sd->input_proj_bias)
        for (int64_t f = 0; f < TF; f++)
            for (int o = 0; o < dec_hidden; o++)
                hidden[f * dec_hidden + o] += sd->input_proj_bias[o];

    if (_ph) { double _t = sd_ph_now(); _ph_p[2] += _t - _ph_mark; _ph_mark = _t; }
    /* === Step 4: pre-transformer with a PER-ITEM KV cache ===
     * The projections and the FFN are batched (one sgemm each, weights read once for
     * the whole batch); RoPE, the cache append and the attention are per item because
     * they are the state. Cache growth/compaction is done for every item BEFORE the
     * layer loop: it rewrites all layers at once. */
    for (int b = 0; b < nb; b++) {
        qwen_sd_stream_state_t *st = sts[b];
        int nf = fr.len[b];
        int need = (st->kv_len + nf) - st->kv_base;
        if (need > st->kv_alloc) {
            int keep_from = st->kv_len - (window - 1);
            if (keep_from < 0) keep_from = 0;
            if (keep_from > st->kv_base && st->kv_len > st->kv_base) {
                int keep = st->kv_len - keep_from;
                int shift = keep_from - st->kv_base;
                for (int l = 0; l < c->dec_num_layers; l++) {
                    if (keep > 0) {
                        memmove(st->k_cache[l], st->k_cache[l] + (int64_t)shift * qkv_dim,
                                (int64_t)keep * qkv_dim * sizeof(float));
                        memmove(st->v_cache[l], st->v_cache[l] + (int64_t)shift * qkv_dim,
                                (int64_t)keep * qkv_dim * sizeof(float));
                    }
                }
                st->kv_base = keep_from;
                need = (st->kv_len + nf) - st->kv_base;
            }
            if (need > st->kv_alloc) {
                int new_alloc = need + 256;
                for (int l = 0; l < c->dec_num_layers; l++) {
                    st->k_cache[l] = (float *)realloc(st->k_cache[l], (int64_t)new_alloc * qkv_dim * sizeof(float));
                    st->v_cache[l] = (float *)realloc(st->v_cache[l], (int64_t)new_alloc * qkv_dim * sizeof(float));
                    if (!st->k_cache[l] || !st->v_cache[l]) goto done;
                }
                st->kv_alloc = new_alloc;
            }
        }
    }

    q        = (float *)aligned_malloc(TF * qkv_dim * sizeof(float));
    new_k    = (float *)aligned_malloc(TF * qkv_dim * sizeof(float));
    new_v    = (float *)aligned_malloc(TF * qkv_dim * sizeof(float));
    x_norm   = (float *)aligned_malloc(TF * dec_hidden * sizeof(float));
    attn_out = (float *)aligned_malloc(TF * qkv_dim * sizeof(float));
    if (!q || !new_k || !new_v || !x_norm || !attn_out) goto done;

    for (int layer = 0; layer < c->dec_num_layers; layer++) {
        qwen_sd_pre_layer_t *l = &sd->pre_layers[layer];

        qwen_rms_norm(x_norm, hidden, l->attn_norm, (int)TF, dec_hidden, eps);

        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, (int)TF, qkv_dim, dec_hidden,
                    1.0f, x_norm, dec_hidden, l->attn_q, dec_hidden, 0.0f, q, qkv_dim);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, (int)TF, qkv_dim, dec_hidden,
                    1.0f, x_norm, dec_hidden, l->attn_k, dec_hidden, 0.0f, new_k, qkv_dim);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, (int)TF, qkv_dim, dec_hidden,
                    1.0f, x_norm, dec_hidden, l->attn_v, dec_hidden, 0.0f, new_v, qkv_dim);

        /* NeoX split-half RoPE at the item's own absolute positions */
        for (int b = 0; b < nb; b++) {
            for (int s = 0; s < fr.len[b]; s++) {
                int64_t row = fr.off[b] + s;
                int abs_pos = sts[b]->kv_len + s;
                const float *cos_ptr = sd->rope_cos + (int64_t)abs_pos * half_hd;
                const float *sin_ptr = sd->rope_sin + (int64_t)abs_pos * half_hd;
                for (int h = 0; h < n_heads; h++) {
                    float *qh = q + row * qkv_dim + h * head_dim;
                    float *kh = new_k + row * qkv_dim + h * head_dim;
                    for (int i = 0; i < half_hd; i++) {
                        float q1 = qh[i], q2 = qh[i + half_hd];
                        float k1 = kh[i], k2 = kh[i + half_hd];
                        float co = cos_ptr[i], si = sin_ptr[i];
                        qh[i]           = q1 * co - q2 * si;
                        qh[i + half_hd] = q2 * co + q1 * si;
                        kh[i]           = k1 * co - k2 * si;
                        kh[i + half_hd] = k2 * co + k1 * si;
                    }
                }
            }
        }

        /* Append K/V and run sliding-window causal attention, per item */
        float scale = 1.0f / sqrtf((float)head_dim);
        for (int b = 0; b < nb; b++) {
            qwen_sd_stream_state_t *st = sts[b];
            int nf = fr.len[b];
            int64_t base = fr.off[b];
            memcpy(st->k_cache[layer] + (int64_t)(st->kv_len - st->kv_base) * qkv_dim,
                   new_k + base * qkv_dim, (int64_t)nf * qkv_dim * sizeof(float));
            memcpy(st->v_cache[layer] + (int64_t)(st->kv_len - st->kv_base) * qkv_dim,
                   new_v + base * qkv_dim, (int64_t)nf * qkv_dim * sizeof(float));

            for (int sq = 0; sq < nf; sq++) {
                int abs_sq = st->kv_len + sq;
                float *out = attn_out + (base + sq) * qkv_dim;
                memset(out, 0, qkv_dim * sizeof(float));
                int sk_start = (abs_sq - window + 1 > 0) ? abs_sq - window + 1 : 0;
                int sk_end = abs_sq;
                for (int h = 0; h < n_heads; h++) {
                    const float *qh = q + (base + sq) * qkv_dim + h * head_dim;
                    float *oh = out + h * head_dim;
                    int n_keys = sk_end - sk_start + 1;
                    float scores_buf[512];
                    float *scores = n_keys <= 512 ? scores_buf : (float *)malloc(n_keys * sizeof(float));
                    float max_score = -1e30f;
                    for (int j = 0; j < n_keys; j++) {
                        int sk = sk_start + j;
                        const float *kh = st->k_cache[layer] + (int64_t)(sk - st->kv_base) * qkv_dim + h * head_dim;
                        float dot = 0;
                        for (int d = 0; d < head_dim; d++) dot += qh[d] * kh[d];
                        scores[j] = dot * scale;
                        if (scores[j] > max_score) max_score = scores[j];
                    }
                    float sum_exp = 0;
                    for (int j = 0; j < n_keys; j++) {
                        scores[j] = expf(scores[j] - max_score);
                        sum_exp += scores[j];
                    }
                    float inv_sum = 1.0f / sum_exp;
                    for (int j = 0; j < n_keys; j++) {
                        int sk = sk_start + j;
                        const float *vh = st->v_cache[layer] + (int64_t)(sk - st->kv_base) * qkv_dim + h * head_dim;
                        float w = scores[j] * inv_sum;
                        for (int d = 0; d < head_dim; d++) oh[d] += vh[d] * w;
                    }
                    if (scores != scores_buf) free(scores);
                }
            }
        }

        /* Output proj + layer_scale + residual */
        {
            float *oproj = x_norm;
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, (int)TF, dec_hidden, qkv_dim,
                        1.0f, attn_out, qkv_dim, l->attn_o, qkv_dim, 0.0f, oproj, dec_hidden);
            for (int64_t s = 0; s < TF; s++) {
                float *xs = hidden + s * dec_hidden;
                float *ps = oproj + s * dec_hidden;
                if (l->attn_layer_scale) {
                    for (int o = 0; o < dec_hidden; o++) xs[o] += ps[o] * l->attn_layer_scale[o];
                } else {
                    for (int o = 0; o < dec_hidden; o++) xs[o] += ps[o];
                }
            }
        }

        qwen_rms_norm(x_norm, hidden, l->ffn_norm, (int)TF, dec_hidden, eps);

        {
            float *ffn_gate = (float *)aligned_malloc(TF * dec_inter * sizeof(float));
            float *ffn_up   = (float *)aligned_malloc(TF * dec_inter * sizeof(float));
            float *ffn_down_out = NULL;
            if (!ffn_gate || !ffn_up) { free(ffn_gate); free(ffn_up); goto done; }
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, (int)TF, dec_inter, dec_hidden,
                        1.0f, x_norm, dec_hidden, l->ffn_gate, dec_hidden, 0.0f, ffn_gate, dec_inter);
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, (int)TF, dec_inter, dec_hidden,
                        1.0f, x_norm, dec_hidden, l->ffn_up, dec_hidden, 0.0f, ffn_up, dec_inter);
            for (int64_t i = 0; i < TF * dec_inter; i++)
                ffn_gate[i] = (ffn_gate[i] / (1.0f + expf(-ffn_gate[i]))) * ffn_up[i];
            free(ffn_up);
            ffn_down_out = (float *)aligned_malloc(TF * dec_hidden * sizeof(float));
            if (!ffn_down_out) { free(ffn_gate); goto done; }
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, (int)TF, dec_hidden, dec_inter,
                        1.0f, ffn_gate, dec_inter, l->ffn_down, dec_inter, 0.0f, ffn_down_out, dec_hidden);
            free(ffn_gate);
            for (int64_t s = 0; s < TF; s++) {
                float *hs = hidden + s * dec_hidden;
                float *ds = ffn_down_out + s * dec_hidden;
                if (l->ffn_layer_scale) {
                    for (int o = 0; o < dec_hidden; o++) hs[o] += ds[o] * l->ffn_layer_scale[o];
                } else {
                    for (int o = 0; o < dec_hidden; o++) hs[o] += ds[o];
                }
            }
            free(ffn_down_out);
        }
    }
    for (int b = 0; b < nb; b++) sts[b]->kv_len += fr.len[b];
    free(q); q = NULL; free(new_k); new_k = NULL; free(new_v); new_v = NULL;
    free(x_norm); x_norm = NULL; free(attn_out); attn_out = NULL;

    if (_ph) { double _t = sd_ph_now(); _ph_p[3] += _t - _ph_mark; _ph_mark = _t; }
    /* === Step 5: final RMSNorm + output proj (512->1024), then per-item latent cache
     * (grown/compacted per item, then filled by memcpy — the sgemm itself is batched). */
    if (sd->final_norm_weight)
        qwen_rms_norm(hidden, hidden, sd->final_norm_weight, (int)TF, dec_hidden, eps);

    lat_tmp = (float *)aligned_malloc(TF * latent_dim * sizeof(float));
    if (!lat_tmp) goto done;
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, (int)TF, latent_dim, dec_hidden,
                1.0f, hidden, dec_hidden, sd->output_proj_weight, dec_hidden,
                0.0f, lat_tmp, latent_dim);
    if (sd->output_proj_bias)
        for (int64_t f = 0; f < TF; f++)
            for (int o = 0; o < latent_dim; o++)
                lat_tmp[f * latent_dim + o] += sd->output_proj_bias[o];
    free(hidden); hidden = NULL;

    for (int b = 0; b < nb; b++) {
        qwen_sd_stream_state_t *st = sts[b];
        int nf = fr.len[b];
        int lat_need = (st->latent_frames + nf) - st->latent_base;
        if (lat_need > st->latent_alloc) {
            int lkeep_from = st->latent_frames - QWEN_SD_STREAM_CONV_RF;
            if (lkeep_from < 0) lkeep_from = 0;
            if (lkeep_from > st->latent_base && st->latent_frames > st->latent_base) {
                int lkeep = st->latent_frames - lkeep_from;
                int lshift = lkeep_from - st->latent_base;
                if (lkeep > 0)
                    memmove(st->latent_cache, st->latent_cache + (int64_t)lshift * latent_dim,
                            (int64_t)lkeep * latent_dim * sizeof(float));
                st->latent_base = lkeep_from;
                lat_need = (st->latent_frames + nf) - st->latent_base;
            }
            if (lat_need > st->latent_alloc) {
                int new_alloc = lat_need + 256;
                float *nc = (float *)realloc(st->latent_cache,
                                             (int64_t)new_alloc * latent_dim * sizeof(float));
                if (!nc) goto done;
                st->latent_cache = nc;
                st->latent_alloc = new_alloc;
            }
        }
        memcpy(st->latent_cache + (int64_t)(st->latent_frames - st->latent_base) * latent_dim,
               lat_tmp + fr.off[b] * latent_dim, (int64_t)nf * latent_dim * sizeof(float));
        st->latent_frames += nf;
    }

    if (_ph) { double _t = sd_ph_now(); _ph_p[4] += _t - _ph_mark; _ph_mark = _t; }
    /* === Step 6: batched conv decoder on the new frames only === */
    signal = (float *)aligned_malloc((int64_t)latent_dim * TF * sizeof(float));
    if (!signal) goto done;
    for (int64_t f = 0; f < TF; f++)
        for (int d = 0; d < latent_dim; d++)
            signal[(int64_t)d * TF + f] = lat_tmp[f * latent_dim + d];
    free(lat_tmp); lat_tmp = NULL;

    {
        /* conv_decoder_forward_streaming_batch takes ownership of `signal` and mutates
         * fr.len in place as the stack upsamples (x2, x2, x8, x5, x4, x3 = x1920). */
        int frc = conv_decoder_forward_streaming_batch(ctx, sts, nb, signal, &fr, auds, ans);
        signal = NULL;
        if (frc != 0) { for (int b = 0; b < nb; b++) free(auds[b]); goto done; }
    }
    for (int b = 0; b < nb; b++) {
        sts[b]->frames_decoded += it[idx[b]].nframes;
        sts[b]->samples_produced += ans[b];
        it[idx[b]].audio = auds[b];
        it[idx[b]].n_samples = ans[b];
        auds[b] = NULL;
    }
    rc = 0;
    SD_PHASE_EMIT("ragged", nb, (int)TF);

done:
    free(idx); rag_free(&fr);
    free(sts); free(tails); free(auds); free(ans);
    free(vq_out); free(cb_sum); free(vq_cf); free(pre_conv_out); free(pre_conv_rm);
    free(hidden); free(q); free(new_k); free(new_v); free(x_norm); free(attn_out);
    free(lat_tmp); free(signal);
    if (rc != 0) for (int i = 0; i < n_items; i++) it[i].rc = rc;
    return rc;
}
#else  /* !USE_BLAS */
int qwen_speech_decoder_decode_streaming_batch(qwen_tts_ctx_t *ctx,
                                               qwen_sd_batch_item_t *it, int n_items) {
    /* The ragged helpers are sgemm-based; without BLAS there is no batched twin to
     * call, and a hand-rolled one would be slower than the scalar per-slot path. */
    for (int i = 0; i < n_items; i++) { it[i].audio = NULL; it[i].n_samples = 0; it[i].rc = 0; }
    return sd_batch_fallback(ctx, it, n_items);
}
#endif /* USE_BLAS */
