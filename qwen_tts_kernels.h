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

/* Retarget the BLAS thread pool at a phase boundary. Prefill is BLAS-heavy and
 * runs with no decoder thread beside it, so it wants every thread; generation
 * runs concurrently with the decoder (which is the BLAS user there), so BLAS
 * must step back or the two pools fight for the same cores. No-op unless linked
 * against OpenBLAS, and always a no-op if the user set OPENBLAS_NUM_THREADS. */
void qwen_blas_set_threads(int n);
int qwen_get_threads(void);
int qwen_get_num_cpus(void);
void qwen_init_threads(void);

/* SOFT thread budget, for a per-stage policy (PLAN S14): change how many runners the
 * NEXT qwen_parallel uses WITHOUT resizing the pool. This matters off macOS: there
 * qwen_set_threads() joins and respawns the pthread pool, and a server that alternates
 * a Talker budget and a decoder budget would pay a thread create/join per frame — at
 * 12.5 frames/s, per stage, that is the kind of overhead that swallows the effect it
 * was meant to measure. The pool stays sized at the HARD budget (the largest any stage
 * asks for) and the extra workers simply claim no chunk: qwen_parallel's work claim is
 * an atomic counter over nt, so nt below the pool size is correct by construction.
 * Clamped to [1, hard]. qwen_get_threads_hard() is the pool size actually spawned. */
void qwen_set_threads_soft(int n);
int  qwen_get_threads_hard(void);

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
/* Revisione git incisa nel binario + profilo SIMD + le flag QWEN_* attive adesso.
 * Va in cima a ogni report: due tabelle di numeri senza questa riga non sono
 * confrontabili, e non c'e' modo di accorgersene dopo. */
void qwen_provenance_report(void *out);

/* ── Batched-path audit (PLAN 0.nonies S10) ─────────────────────────────────
 *
 * WHY. The batching pays off only when the projection is a REAL matrix-matrix
 * kernel: one weight read serving all B requests. Several dispatch paths quietly
 * fall back to B sequential matvecs (no int8 GEMM on M1-class ARM or on AVX2
 * before 2026-08-18; the q4 floor fallback; QWEN_BATCH_FORCE_MATVEC). A B=1..8
 * scaling curve measured on top of a fallback measures the fallback, not the box —
 * so before trusting any concurrency number, ask the engine which path it took.
 *
 * Enable with QWEN_BATCH_STATS=1; the report prints at exit to stderr. Counting is
 * off (a relaxed atomic load) otherwise. */
enum {
    QWEN_MMK_NONE = 0,
    QWEN_MMK_BF16_BFMMLA, QWEN_MMK_BF16_FIXEDB, QWEN_MMK_BF16_GENERIC,
    QWEN_MMK_INT8_AMX, QWEN_MMK_INT8_VNNI, QWEN_MMK_INT8_AVX2,
    QWEN_MMK_INT8_SMMLA, QWEN_MMK_INT8_SDOT, QWEN_MMK_INT8_F32TWIN,
    QWEN_MMK_Q4_VNNI, QWEN_MMK_Q4_AVX2, QWEN_MMK_Q4_SMMLA,
    QWEN_MMK_Q4_BMATVEC, QWEN_MMK_Q4_GENERIC,
    QWEN_MMK_FORCED_MATVEC,   /* QWEN_BATCH_FORCE_MATVEC / QWEN_BATCH_NOMATMUL */
    QWEN_MMK_SOLO,            /* B_eff==1 -> single-stream step (by design, not a fallback) */
    /* Appended 2026-08-18 (AMX round 2). New ids go HERE, at the end, before COUNT:
     * g_mmk_info[] in qwen_tts_kernels.c is a positional array, so inserting in the
     * middle silently relabels every row after it. */
    QWEN_MMK_BF16_AMX,        /* _tile_dpbf16ps  — bf16 GEMM on Sapphire/Emerald Rapids */
    QWEN_MMK_Q4_AMX,          /* _tile_dpbssd on nibbles decoded to int8, per-block scale */
    QWEN_MMK_KLEIDI_Q4,       /* Arm KleidiAI: i8mm GEMM / dotprod GEMV, weights PRE-PACKED at load */
    /* Single-vector GEMV, one id per precision. These exist so the audit is SYMMETRIC:
     * before them only the batched paths and KleidiAI were counted, so an --int8 run
     * showed almost nothing and the percentages compared unlike things. They are not
     * dispatchable (no gate row): they are counters on the matvec entry points. */
    QWEN_MMK_BF16_GEMV, QWEN_MMK_INT8_GEMV, QWEN_MMK_Q4_GEMV,
    /* GGUF Q8_0 kept per-block-32 and repacked 4-row at load (ggml's q8_0x4 layout).
     * Two ids, because GEMM and GEMV are different instructions on different operands
     * and "fired" has to be answerable for each. */
    QWEN_MMK_KLEIDI_I8,       /* Arm KleidiAI qsi8cxp GEMM (i8mm, B>=2) */
    QWEN_MMK_KLEIDI_I8_GEMV,  /* same family at B==1: dotprod, and NOT a matrix instruction */
    QWEN_MMK_KLEIDI_BF16,     /* Arm KleidiAI bf16p GEMM (mmla, B>=2) */
    QWEN_MMK_KLEIDI_BF16_GEMV,/* same family at B==1 */
    QWEN_MMK_Q8_REPACK_I8MM,  /* vmmlaq_s32, B >= 2 */
    QWEN_MMK_Q8_REPACK_GEMV,  /* vdotq_s32,  B == 1 */
    QWEN_MMK_COUNT
};
/* Which part of the model is running right now. Set by the step functions, read by the
 * MAC counters, so the audit can say "the Code Predictor used KleidiAI 7392 times"
 * instead of only "something did". Thread-local: the pool workers run INSIDE a kernel,
 * past the point where this is read, so nothing has to be propagated into them. */
enum { QWEN_COMP_OTHER = 0, QWEN_COMP_TALKER, QWEN_COMP_CP, QWEN_COMP_DECODER, QWEN_COMP_COUNT };
void qwen_mm_component(int comp);
int  qwen_mm_component_get(void);

int  qwen_matmat_stats_enabled(void);
void qwen_matmat_stats_note(int kernel_id, long long macs);
/* WEIGHT TRAFFIC (PLAN S16.4): bytes of WEIGHTS pulled from memory by the projections.
 * Not derivable from the MAC counter, and that is the whole point: B matvecs and a
 * batched twin do the same MACs and read the weights B times vs once. Divided by the
 * measured STREAM bandwidth of the box it gives BW_utilization — the number that
 * separates memory-bound from compute-bound from scheduler-bound.
 * ⚠️ Scope, stated so nobody reads it as total DRAM traffic: Talker + Code Predictor
 * projections only. KV cache, activations, embeddings and the speech decoder are NOT
 * counted, so it is a LOWER bound on the engine's real traffic. */
void qwen_matmat_stats_note_bytes(long long weight_bytes);
/* ── SHAPE CENSUS (QWEN_SHAPE_CENSUS=1) ──────────────────────────────────────────
 *
 * The MAC audit above answers "which kernel ran". It cannot answer "on WHAT shape",
 * for two reasons: the batched kernels book their counters from inside a pool worker,
 * so `rows` there is the THREAD'S SLICE and not the matrix; and nothing records how
 * many times per audio frame a given projection is entered.
 *
 * A shape-aware kernel policy needs exactly those two numbers. So this records at the
 * DISPATCHER ENTRY, on the calling thread, before any slicing: the logical (N,K,B) of
 * every projection, its call count, its MACs, and a bitmask of the kernels that
 * actually fired underneath it. Divided by the frame counter it yields calls/frame and
 * GMAC/frame per shape - the census the policy is built from.
 *
 * Off by default and gated at every site, so a normal run pays one predictable
 * not-taken branch per projection. Meant for a SINGLE-STREAM diagnostic run: the
 * kernel bitmask is filled by workers through a global "current op" pointer, which
 * under concurrent requests can attribute a kernel to a neighbouring shape. It is a
 * bitmask, so the error is over-inclusion, never a wrong exclusion. */
int  qwen_census_enabled(void);
void qwen_census_op(const char *entry, int rows, int cols, int B);
void qwen_census_frame(void);
void qwen_census_frame_at(int site);  /* 0 single, 1 batched, 2 batched per-slot */
void qwen_census_report(void *out);   /* NULL -> stderr; also runs at exit */

void qwen_matmat_stats_reset(void);
void qwen_matmat_stats_report(void *out);   /* NULL -> stderr */
/* Print which kernel the dispatcher actually selects for a given shape. Compiled-in
 * capability and selected kernel are different questions; this answers the second. */
void qwen_kernel_selection_report(void *out, int rows, int cols);

/* Kernel numeric self-test: runs the dispatched matvecs (bf16/int8/argmax-int8)
 * against an f32 reference on deterministic random data. Cross-ISA correctness
 * proof for the SIMD kernels (esp. the AVX-512/VNNI paths) that does NOT depend
 * on a full-pipeline golden, so it's immune to the greedy trajectory fork.
 * `out` may be NULL -> stdout. Returns 0 on PASS, >0 = number of failed cases. */
int qwen_kernel_selftest(void *out);

/* Batched matmat throughput microbench: times the real qwen_matmat_{bf16,int8,q4_0}
 * vs B*qwen_matvec_* per precision/shape at the current thread count (no model).
 * B via QWEN_BATCH_B (default 8). `out` may be NULL -> stdout. Returns 0. */
int qwen_matmat_bench(void *out);

/* `--matmat-tune`: MEASURE the g_mm_gate[] dispatcher thresholds instead of guessing
 * them. Sweeps B = 1,2,4,8,16 x the model's REAL projection shapes x every batched
 * kernel this binary/ISA can dispatch (each pinned in turn), at ONE thread AND at the
 * current -j, always against B x qwen_matvec_* of the same format. Prints the per-
 * kernel crossover B, the QWEN_*_MIN_B / QWEN_NO_* lines to export on this box, and
 * flags any kernel that wins only with the thread pool (= hiding dispatch overhead,
 * not sharing weight reads). `model_dir` may be NULL -> declared 0.6B+1.7B shapes.
 * QWEN_TUNE_JSON=<path> also writes the full grid as JSON; QWEN_TUNE_QUICK=1 drops
 * the shapes above 4 M elements. `out` may be NULL -> stdout. Returns 0. */
int qwen_matmat_tune(void *out, const char *model_dir);

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

/* Optional GPU offload hook for qwen_matvec_bf16 (and the bf16 QKV fused path).
 * NULL = CPU default. Installed by the Metal/CUDA backend when --backend is set. */
extern void (*g_qwen_matvec_bf16_hook)(float *, const uint16_t *, const float *, int, int);
/* Optional GPU offload hook for the batched matmat (where the MMA win lands). */
extern void (*g_qwen_matmat_bf16_hook)(float *, const uint16_t *, const float *, int, int, int);

/* bf16 BATCHED matmat (the batching/spec-decode-verify primitive):
 *   Y[rows,B] = W[rows,cols] @ X[cols,B]     (W bf16; X,Y f32, row-major)
 * Each weight element is read from DRAM ONCE and reused across all B columns
 * (amortizes the per-token weight re-read that bounds single-stream). B<=64.
 * Threaded by row-slice, matching qwen_matvec_bf16. With B==1 it equals matvec. */
void qwen_matmat_bf16(float *Y, const uint16_t *W, const float *X, int rows, int cols, int B);

/* INT8 batched matmat twin (Y[rows,B] = (W_int8*scale) @ X[cols,B]). Low precision
 * is where batching pays MOST: int8 halves the weight read. Same compile-time-B
 * register-blocking as bf16. X is f32 [cols,B]; weights are the existing int8
 * per-row-scale format. B<=64. (q4_0 twin declared after the q4_0_block_t typedef.) */
void qwen_matmat_int8(float *Y, const int8_t *W, const float *scale,
                      const float *X, int rows, int cols, int B);

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

/* fp16 (IEEE binary16) <-> f32 for the q4_0 block scale. Storage-only: all math
 * stays f32 — one convert per 32-weight block on kernels that are bandwidth-bound,
 * so the cost is noise while the block shrinks 20 -> 18 bytes (-10% q4 traffic;
 * perf item 2, 2026-07-11 — same layout as llama.cpp q4_0). aarch64 uses the
 * native __fp16; elsewhere a portable bit-exact fallback (handles subnormals). */
static inline float qwen_f16_to_f32(uint16_t h) {
#if defined(__aarch64__)
    __fp16 v; memcpy(&v, &h, sizeof(v)); return (float)v;
#else
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t em   = h & 0x7FFF;
    uint32_t bits;
    if (em >= 0x7C00)      bits = sign | 0x7F800000u | ((em & 0x03FF) << 13); /* inf/NaN */
    else if (em >= 0x0400) bits = sign | ((em + ((127u - 15u) << 10)) << 13); /* normal */
    else if (em == 0)      bits = sign;                                        /* +-0 */
    else {                                                                     /* subnormal */
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
    if (e >= 0x1F) return (uint16_t)(sign | 0x7C00);       /* overflow -> inf */
    if (e <= 0) {                                          /* subnormal / zero */
        if (e < -10) return (uint16_t)sign;
        m |= 0x00800000;
        uint32_t shift = (uint32_t)(14 - e);
        uint16_t sub = (uint16_t)(m >> shift);
        if ((m >> (shift - 1)) & 1) sub++;                 /* round-to-nearest */
        return (uint16_t)(sign | sub);
    }
    uint16_t out = (uint16_t)(sign | ((uint32_t)e << 10) | (m >> 13));
    if (m & 0x1000) out++;                                 /* round-to-nearest (carry into exp is fine) */
    return out;
#endif
}

/* Q4_0 block: 32 weights packed into 18 bytes (16 nibble-pairs + fp16 scale).
 * The fp16 scale (was f32, 20 B/block) cuts q4 weight traffic 10% — the block is
 * pure bandwidth on the 16x-reread CP. Read with qwen_f16_to_f32(). */
#define Q4_0_BLOCK_SIZE 32
typedef struct {
    uint16_t scale_f16;    /* per-block scale factor, IEEE fp16 bits */
    uint8_t qs[16];        /* 32 nibbles: low 4 bits = even idx, high 4 bits = odd idx */
} q4_0_block_t;            /* 18 bytes per 32 weights */

/* Quantize bf16 weight matrix to Q4_0 blocks.
 * cols must be a multiple of 32. Returns number of blocks per row = cols/32.
 * dst must have rows * (cols/32) blocks pre-allocated. */
void qwen_quantize_bf16_to_q4_0(const uint16_t *src_bf16, int rows, int cols,
                                 q4_0_block_t *dst);

/* Q4_0 matvec: y[rows] = dequant(W_q4[rows, cols/32 blocks]) @ x[cols]
 * NEON-optimized + multi-threaded. */
void qwen_matvec_q4_0(float *y, const q4_0_block_t *W, const float *x,
                       int rows, int cols);

/* Q4_0 batched matmat twin: Y[rows,B] = dequant(W_q4) @ X[cols,B]. The nibble
 * unpack is done once and reused across the B columns (amortized) — where int4
 * batching pays most. X is f32 [cols,B]. B<=64. */
void qwen_matmat_q4_0(float *Y, const q4_0_block_t *W, const float *X,
                      int rows, int cols, int B);

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

/* ── Q6_0: 6 bits, one fp16 scale every 32 weights (PLAN T2 / T2.next) ──
 *
 * WHY THIS FORMAT EXISTS. The Talker cannot go to int4 — it drops the language
 * outright (a low-resource-language clip classified FRENCH at 89%, ~1 seed in 5; PLAN 0.septies).
 * But that wall was measured on FLAT per-row scales. With a scale every 32 weights
 * the same bit budget carries far more information, and the fakequant sweep (error
 * baked into a bf16 copy, no kernel) scored it on LANGUAGE IDENTITY, not on perplexity:
 *
 *     int8r (= what ships today)   language identity 98.0% mean / 96.3% min   control
 *     int7b            -6% band    language identity 97.6% / 93.0%            holds
 *     int6b           -19% band    language identity 78.3% / 26.3%            collapses ALONE
 *     int8+int6 mixed, 7 of 28 layers at int8:  -14% band, 96.2% / 91.0%
 *
 * So int6 is NOT usable uniformly; it is usable as the CHEAP HALF of a per-layer
 * mixed map. This type is that cheap half. `qwen_quantize_bf16_to_q6_0` reproduces
 * tools/quant/fakequant_cp.py `int6b` EXACTLY (fp16 absmax/31 scale, C roundf,
 * clamp +-31) — that bit-exactness is what carries the measured language-identity numbers over
 * from the fakequant to the real kernel. tests/quant_q6_kernel_bench.c gates it.
 *
 * LAYOUT — 26 bytes per 32 weights = 0.8125 B/weight (vs int8's 1.0 -> -18.75%).
 * Split 4+2 (the Q6_K trick) because 6 bits is not byte-aligned:
 *   ql = the low 4 bits in the SAME interleaved order as q4_0 (low nibble = even
 *        index, high nibble = odd), so the already-tested vzip / unpacklo
 *        value-order trick is reused verbatim;
 *   qh = the high 2 bits, laid out so ONE 4-byte broadcast + ONE variable shift +
 *        ONE mask yields 16 lanes ALREADY in value order:
 *        qh[4g + j], bits (2k..2k+1) = weight (16g + j + 4k), g = half, j,k = 0..3.
 * Stored code is u = q + 32 (q in [-31,31] -> u in [1,63], 6 bits). Dequant:
 *   w = scale * (int8)((ql | (qh << 4)) - 32).
 *
 * WARNING — THE OPEN QUESTION IS SPEED, NOT ERROR. -18.75% of weight bytes is a win
 * only if the unpack does not eat it: on EPYC 9555P the q4 VNNI path came out ~37%
 * SLOWER than int8 despite half the bytes (see q4_0_matvec_vnni_v3). Fewer bytes is
 * a hypothesis about time until a kernel is timed. */
#define Q6_0_BLOCK_SIZE 32
typedef struct {
    uint16_t scale_f16;    /* fp16 absmax/31, the same value the fakequant int6b uses */
    uint8_t  ql[16];       /* low 4 bits, q4_0-interleaved (even idx = low nibble) */
    uint8_t  qh[8];        /* high 2 bits; qh[4g+j] bits 2k..2k+1 = weight 16g+j+4k */
} q6_0_block_t;            /* 26 bytes per 32 weights */

/* Quantize a bf16 weight matrix to Q6_0. cols must be a multiple of 32.
 * dst must hold rows * (cols/32) blocks. Bit-exact with fakequant int6b. */
void qwen_quantize_bf16_to_q6_0(const uint16_t *src_bf16, int rows, int cols,
                                 q6_0_block_t *dst);

/* Q6_0 matvec: y[rows] = dequant(W_q6[rows, cols/32 blocks]) @ x[cols].
 * SDOT (ARM) / VNNI (x86) native + multi-threaded, scalar fallback elsewhere. */
void qwen_matvec_q6_0(float *y, const q6_0_block_t *W, const float *x,
                       int rows, int cols);

/* Unified QKV matvec (Q6_0 variant) — quantizes the shared activation once for
 * Q, K and V, like the int8 and q4_0 twins. Not an optimization detail: without
 * it a q6-vs-int8 comparison is rigged, because int8 has a fused QKV. */
void qwen_matvec_q6_0_qkv(float *q, float *k, float *v,
                          const q6_0_block_t *Wq, const q6_0_block_t *Wk,
                          const q6_0_block_t *Wv,
                          const float *x, int in_dim, int q_dim, int kv_dim);

/* Dequantize one Q6_0 row back to f32 — the parity gate's reference reader. */
void qwen_dequant_row_q6_0(float *dst, const q6_0_block_t *row, int cols);

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


/* ========================================================================
 * INT8 SDOT conv engine (speech decoder, opt-in via QWEN_SD_INT8=1)
 * Ported from external PR #17 (TrinityTF). ARM dotprod only; fp32 elsewhere.
 * ======================================================================== */

/* 1 if the SDOT int8 conv path is compiled in (ARM dotprod). */
int qwen_sd_int8_available(void);

/* K padded up to a multiple of blk (blk must be a multiple of 16). */
int qwen_int8_kp(int K, int blk);

/* Per-row, per-blk-block absmax int8 quantization: scales is [rows][Kp/blk],
 * rows padded to Kp with zeros. */
void qwen_int8_quant_rows(int8_t *dst, float *scales, const float *src,
                          int rows, int K, int Kp, int blk);

/* 1 when this box has a usable AMX bf16 tile unit (compiled, present, XTILEDATA
 * granted). Used to pick the prefill GEMM — see the definition for the measurement. */
int qwen_amx_bf16_available(void);
int qwen_arm_bf16_matmat_available(void);

/* Threaded int8 causal conv1d: im2col + SDOT GEMM per column panel.
 * Wq: [out_ch, Kp] with block scales sw [out_ch][Kp/blk] (K = in_ch*kernel,
 * im2col order ic*kernel+kk). out: channel-first [out_ch, length], bias applied. */
/* wsum: [out_ch][Kp/blk] sums of the QUANTIZED weight bytes per block. Needed only by
 * the x86 VNNI kernel, whose dpbusd is unsigned x signed: it dots (x+128) against w and
 * subtracts 128*sum(w). Precomputed once with the weights (it is a property of them),
 * so the correction never enters the inner loop. NULL is allowed on ARM, where SDOT is
 * signed x signed and no correction exists. */
void qwen_conv1d_int8(float *out, const float *in,
                      const int8_t *Wq, const float *sw, const int32_t *wsum,
                      const float *bias,
                      int in_ch, int out_ch, int length, int kernel, int dilation,
                      int Kp, int blk);

/* Threaded int8 GEMM on pre-quantized activations: out[M,N] (ld out_ld) =
 * sum_b sw[m][b]*sa[t][b]*dot_b(Wq[m], Xq[t]); no bias. Rows Kp-strided. */
void qwen_gemm_int8(float *out, int out_ld,
                    const int8_t *Wq, const float *sw, const int32_t *wsum,
                    const int8_t *Xq, const float *sa,
                    int M, int N, int Kp, int blk);

#endif /* QWEN_TTS_KERNELS_H */
