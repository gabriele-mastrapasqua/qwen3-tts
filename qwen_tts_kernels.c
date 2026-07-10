/*
 * qwen_tts_kernels.c - Kernel implementations
 */

#include <pthread.h>
#include "qwen_tts_kernels.h"
#include "qwen_tts_thread.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>       /* clock_gettime / CLOCK_MONOTONIC / struct timespec (matmat-bench).
                         * macOS pulls this in transitively; Linux (esp. aarch64) needs it explicit. */
#include <stdatomic.h>
#include <sys/types.h>
#ifdef __APPLE__
#include <sys/sysctl.h>
#endif
#ifdef __linux__
#include <unistd.h>
#if defined(__aarch64__)
#include <sys/auxv.h>
#include <asm/hwcap.h>
#endif
#endif
#ifdef __AVX2__
#include <immintrin.h>
#endif

#ifdef USE_BLAS
#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif
#endif

/* Flush-to-zero for denormals. INT8 dequant can drive activations into the
 * subnormal range; denormal FP arithmetic is ~100x slower (looks like a hang).
 * FTZ is per-thread on ARM (FPCR), so it must be set on every compute thread —
 * including each GCD worker — not just the main thread. Cheap (~1-2 cycles),
 * called once per matvec, negligible. Inaudible quality impact. */
void qwen_ftz_on(void) {
#if defined(__aarch64__)
    uint64_t fpcr;
    __asm__ volatile("mrs %0, fpcr" : "=r"(fpcr));
    if (!(fpcr & (1ULL << 24))) {
        fpcr |= (1ULL << 24); /* FZ: flush-to-zero */
        __asm__ volatile("msr fpcr, %0" : : "r"(fpcr));
    }
#elif defined(__x86_64__)
    unsigned int mxcsr = __builtin_ia32_stmxcsr();
    __builtin_ia32_ldmxcsr(mxcsr | 0x8040); /* FTZ (bit15) | DAZ (bit6) */
#endif
}

/* Threading */
static int g_n_threads = 1;
/* OpenBLAS spawns one thread per core by default and knows nothing about our
 * pool, so `-j4` on a 64-core box meant 4 threads of ours + 64 of theirs, on 64
 * cores. `perf` on a 4-core Neoverse-N1 put ~21% of wall time in __schedule /
 * el0_svc / sched_yield: the two pools fighting. Bind BLAS to the budget `-j`
 * actually asks for.
 *
 * Weak symbol: resolved when linked against OpenBLAS, NULL with Accelerate
 * (which manages its own threads) or a reference BLAS, where this is a no-op.
 * OPENBLAS_NUM_THREADS in the environment still wins -- OpenBLAS reads it at
 * init, and a user tuning by hand should not be second-guessed. */
#if defined(__GNUC__) && !defined(__APPLE__)
extern void openblas_set_num_threads(int) __attribute__((weak));
#endif

void qwen_blas_set_threads(int n) {
#if defined(__GNUC__) && !defined(__APPLE__)
    if (getenv("OPENBLAS_NUM_THREADS")) return;   /* explicit user choice wins */
    if (openblas_set_num_threads) openblas_set_num_threads(n > 0 ? n : 1);
#else
    (void)n;
#endif
}

void qwen_set_threads(int n) {
    g_n_threads = n > 0 ? n : 1;
    qwen_ftz_on();
    qwen_threadpool_start(g_n_threads);  /* (re)size the off-Mac worker pool */
    qwen_blas_set_threads(g_n_threads);
}
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
    qwen_ftz_on();  /* main thread: flush denormals (int8 activations) */
    qwen_threadpool_start(g_n_threads);  /* spawn the off-Mac persistent pool */
    qwen_blas_set_threads(g_n_threads);  /* else BLAS grabs all ncpus (see below) */
}

/* Report ACTUAL compiled capabilities (mirrors the kernels' own #ifdef guards).
 * Run `./qwen_tts --caps`. Makes the real SIMD/threading state visible + testable so
 * a false "we have AVX2/threading" claim can't survive — the binary tells the truth. */
void qwen_caps_report(void *out) {
    FILE *f = out ? (FILE *)out : stderr;
    fprintf(f, "qwen-tts compiled capabilities:\n");
#if defined(__aarch64__)
    fprintf(f, "  arch:             arm64\n");
#elif defined(__x86_64__)
    fprintf(f, "  arch:             x86-64\n");
#else
    fprintf(f, "  arch:             (other)\n");
#endif
    /* Hot path: bf16/int8/q4 matvecs + attention (~90%% of decode). Both NEON and
     * AVX2 are full 2-row, multi-accumulator, prefetching kernels (PLAN 21.3). */
#ifdef __ARM_NEON
    fprintf(f, "  matvec + attn:    NEON (2-row fused)\n");
#elif defined(__AVX2__)
    fprintf(f, "  matvec + attn:    AVX2 (2-row fused, FMA)\n");
#else
    fprintf(f, "  matvec + attn:    scalar\n");
#endif
#if defined(__ARM_FEATURE_DOTPROD)
    fprintf(f, "  int8 dot:         SDOT vdotq_s32 (native)\n");
#elif defined(__AVX512VNNI__)
    fprintf(f, "  int8 dot:         VNNI _mm512_dpbusd_epi32 (native)\n");
#elif defined(__AVX2__)
    fprintf(f, "  int8 dot:         widen->FMA (AVX2; no VNNI)\n");
#else
    fprintf(f, "  int8 dot:         dequant->FMA (no SDOT/VNNI)\n");
#endif
#if defined(__AVX2__)
    fprintf(f, "  rms/bf16-conv:    AVX2\n");
#elif defined(__ARM_NEON)
    fprintf(f, "  rms/bf16-conv:    NEON\n");
#else
    fprintf(f, "  rms/bf16-conv:    scalar\n");
#endif
#if defined(__ARM_FEATURE_BF16)
    fprintf(f, "  arm bf16 matmul:  bfdot/bfmmla AVAILABLE but UNUSED (PLAN 21.3b)\n");
#endif
#if defined(__ARM_FEATURE_MATMUL_INT8)
    fprintf(f, "  arm i8mm:         smmla AVAILABLE but UNUSED (PLAN 21.3b)\n");
#endif
#if defined(__APPLE__) && defined(__BLOCKS__) && !defined(QWEN_FORCE_PTHREAD)
    fprintf(f, "  matvec threads:   GCD dispatch_apply (%d threads)\n", qwen_get_threads());
#elif defined(_WIN32) && !defined(QWEN_USE_PTHREADS)
    fprintf(f, "  matvec threads:   Win32 pool (%d threads)\n", qwen_get_threads());
#else
    fprintf(f, "  matvec threads:   pthread pool (%d threads)\n", qwen_get_threads());
#endif
#if defined(USE_BLAS) && defined(__APPLE__)
    fprintf(f, "  BLAS (prefill):   Accelerate\n");
#elif defined(USE_BLAS)
    fprintf(f, "  BLAS (prefill):   OpenBLAS\n");
#else
    fprintf(f, "  BLAS (prefill):   none\n");
#endif
    /* ---- Runtime ISA actually present on THIS CPU (independent of how the binary
     * was compiled above). This is the "does the extension fire?" check — run it on
     * a freshly-rented box to see what the CPU offers before deciding the build/kernel
     * path. A gap vs the compiled features is the "compiled past the CPU -> SIGILL" trap.
     * See docs/hardware-testing.md for the per-platform plan. */
#if defined(__x86_64__)
    __builtin_cpu_init();
    /* clang's __builtin_cpu_supports rejects the "amx-int8" feature string (gcc-only) →
     * "invalid cpu feature string for builtin". Compute it guarded so clang-tidy/clang builds
     * still parse; gcc keeps the AMX runtime probe. */
    const char *amx_str = "";
#if defined(__GNUC__) && !defined(__clang__)
    if (__builtin_cpu_supports("amx-int8")) amx_str = " amx-int8";
#endif
    fprintf(f, "  runtime cpu:      sse2%s%s%s%s%s%s%s%s\n",
            __builtin_cpu_supports("avx")        ? " avx"          : "",
            __builtin_cpu_supports("avx2")       ? " avx2"         : "",
            __builtin_cpu_supports("fma")        ? " fma"          : "",
            __builtin_cpu_supports("avx512f")    ? " avx512f"      : "",
            __builtin_cpu_supports("avx512bw")   ? " avx512bw"     : "",
            __builtin_cpu_supports("avx512vnni") ? " avx512vnni"   : "",
            __builtin_cpu_supports("avx512bf16") ? " avx512bf16"   : "",
            amx_str);
    fprintf(f, "  lever (x86):      %s\n",
            __builtin_cpu_supports("avx512vnni") ? "VNNI int8 dot (native) — int8/int4 + batching is the throughput play"
          : __builtin_cpu_supports("avx2")       ? "AVX2 only (no VNNI) — int8 via widen+FMA; bandwidth-bound, batching helps"
          :                                        "no AVX2 — scalar; rebuild SIMD=scalar");
#if defined(__AVX2__)
    if (!__builtin_cpu_supports("avx2"))
        fprintf(f, "  WARNING: built with AVX2 but this CPU lacks it -> will SIGILL. "
                   "Rebuild with `make blas SIMD=scalar`.\n");
#endif
#elif defined(__aarch64__)
    /* ARM runtime features. macOS: per-feature sysctls (works on every M-series).
     * Linux: getauxval HWCAP bits (Graviton / Ampere / Grace). */
    int has_dotprod = 0, has_bf16 = 0, has_i8mm = 0, has_sve = 0, has_sve2 = 0, has_sme = 0;
#if defined(__APPLE__)
    { int v; size_t s;
      #define QFEAT(name) (s = sizeof(v), v = 0, sysctlbyname(name, &v, &s, NULL, 0) == 0 && v)
      has_dotprod = QFEAT("hw.optional.arm.FEAT_DotProd");
      has_bf16    = QFEAT("hw.optional.arm.FEAT_BF16");
      has_i8mm    = QFEAT("hw.optional.arm.FEAT_I8MM");
      has_sme     = QFEAT("hw.optional.arm.FEAT_SME");
      #undef QFEAT
    }
#elif defined(__linux__)
    { unsigned long h1 = getauxval(AT_HWCAP), h2 = getauxval(AT_HWCAP2);
      #ifdef HWCAP_ASIMDDP
      has_dotprod = (h1 & HWCAP_ASIMDDP) != 0;
      #endif
      #ifdef HWCAP_SVE
      has_sve = (h1 & HWCAP_SVE) != 0;
      #endif
      #ifdef HWCAP2_BF16
      has_bf16 = (h2 & HWCAP2_BF16) != 0;
      #endif
      #ifdef HWCAP2_I8MM
      has_i8mm = (h2 & HWCAP2_I8MM) != 0;
      #endif
      #ifdef HWCAP2_SVE2
      has_sve2 = (h2 & HWCAP2_SVE2) != 0;
      #endif
      #ifdef HWCAP2_SME
      has_sme = (h2 & HWCAP2_SME) != 0;
      #endif
      (void)h1; (void)h2;
    }
#endif
    fprintf(f, "  runtime cpu:      NEON%s%s%s%s%s%s\n",
            has_dotprod ? " dotprod/SDOT" : "",
            has_bf16    ? " bf16/BFDOT"   : "",
            has_i8mm    ? " i8mm/SMMLA"   : "",
            has_sve     ? " SVE"          : "",
            has_sve2    ? " SVE2"         : "",
            has_sme     ? " SME"          : "");
    fprintf(f, "  lever (arm):      %s%s\n",
            has_i8mm ? "i8mm SMMLA + " : (has_dotprod ? "SDOT + " : ""),
            has_bf16 ? "bf16 BFDOT/BFMMLA available -> native batched matmat twin (PLAN 21.3b, currently scalar-decode)"
                     : "no bf16 matmul (M1-class) -> batched matmat uses scalar bf16 decode");
    if (!has_bf16 && !has_i8mm)
        fprintf(f, "  note:             M1-class (Armv8.5, dotprod only). M2/M3/M4/M5 add bf16+i8mm -> the native-matmul lever.\n");
#endif
}

void qwen_check_runtime_isa(void) {
#if defined(__x86_64__) && defined(__AVX2__)
    __builtin_cpu_init();
    if (!__builtin_cpu_supports("avx2")) {
        fprintf(stderr,
            "qwen-tts: FATAL — this binary was built with AVX2 but the CPU does not "
            "support it.\n  Rebuild a portable binary with: make blas SIMD=scalar\n");
        exit(1);
    }
#endif
}

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
#elif defined(__AVX2__)
        __m256 vsum0 = _mm256_setzero_ps(), vsum1 = _mm256_setzero_ps();
        int i = 0;
        for (; i + 15 < dim; i += 16) {
            __m256 v0 = _mm256_loadu_ps(xs + i);
            __m256 v1 = _mm256_loadu_ps(xs + i + 8);
            vsum0 = _mm256_fmadd_ps(v0, v0, vsum0);
            vsum1 = _mm256_fmadd_ps(v1, v1, vsum1);
        }
        __m256 vs = _mm256_add_ps(vsum0, vsum1);
        float tmp[8]; _mm256_storeu_ps(tmp, vs);
        float sum = tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7];
        for (; i < dim; i++) sum += xs[i] * xs[i];

        float inv_rms = 1.0f / sqrtf(sum / dim + eps);
        __m256 vinv = _mm256_set1_ps(inv_rms);
        i = 0;
        for (; i + 15 < dim; i += 16) {
            __m256 v0 = _mm256_loadu_ps(xs + i);
            __m256 v1 = _mm256_loadu_ps(xs + i + 8);
            __m256 w0 = _mm256_loadu_ps(weight + i);
            __m256 w1 = _mm256_loadu_ps(weight + i + 8);
            _mm256_storeu_ps(os + i,     _mm256_mul_ps(_mm256_mul_ps(v0, vinv), w0));
            _mm256_storeu_ps(os + i + 8, _mm256_mul_ps(_mm256_mul_ps(v1, vinv), w1));
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

void qwen_rms_norm_residual(float *out, float *x, const float *residual,
                            const float *weight, int dim, float eps) {
    /* Fuse: x[i] += residual[i], then out = x * inv_rms * weight */
#ifdef __ARM_NEON
    float32x4_t vsum0 = vdupq_n_f32(0), vsum1 = vdupq_n_f32(0);
    int i = 0;
    /* Pass 1: add residual to x AND compute sum of squares in one pass */
    for (; i + 7 < dim; i += 8) {
        float32x4_t x0 = vld1q_f32(x + i);
        float32x4_t x1 = vld1q_f32(x + i + 4);
        float32x4_t r0 = vld1q_f32(residual + i);
        float32x4_t r1 = vld1q_f32(residual + i + 4);
        x0 = vaddq_f32(x0, r0);
        x1 = vaddq_f32(x1, r1);
        vst1q_f32(x + i, x0);
        vst1q_f32(x + i + 4, x1);
        vsum0 = vfmaq_f32(vsum0, x0, x0);
        vsum1 = vfmaq_f32(vsum1, x1, x1);
    }
    float sum = vaddvq_f32(vaddq_f32(vsum0, vsum1));
    for (; i < dim; i++) { x[i] += residual[i]; sum += x[i] * x[i]; }

    float inv_rms = 1.0f / sqrtf(sum / dim + eps);
    float32x4_t vinv = vdupq_n_f32(inv_rms);

    /* Pass 2: normalize and scale */
    i = 0;
    for (; i + 7 < dim; i += 8) {
        float32x4_t v0 = vld1q_f32(x + i);
        float32x4_t v1 = vld1q_f32(x + i + 4);
        float32x4_t w0 = vld1q_f32(weight + i);
        float32x4_t w1 = vld1q_f32(weight + i + 4);
        vst1q_f32(out + i,     vmulq_f32(vmulq_f32(v0, vinv), w0));
        vst1q_f32(out + i + 4, vmulq_f32(vmulq_f32(v1, vinv), w1));
    }
    for (; i < dim; i++) out[i] = x[i] * inv_rms * weight[i];
#elif defined(__AVX2__)
    __m256 vsum0 = _mm256_setzero_ps(), vsum1 = _mm256_setzero_ps();
    int i = 0;
    for (; i + 15 < dim; i += 16) {
        __m256 x0 = _mm256_loadu_ps(x + i);
        __m256 x1 = _mm256_loadu_ps(x + i + 8);
        __m256 r0 = _mm256_loadu_ps(residual + i);
        __m256 r1 = _mm256_loadu_ps(residual + i + 8);
        x0 = _mm256_add_ps(x0, r0);
        x1 = _mm256_add_ps(x1, r1);
        _mm256_storeu_ps(x + i, x0);
        _mm256_storeu_ps(x + i + 8, x1);
        vsum0 = _mm256_fmadd_ps(x0, x0, vsum0);
        vsum1 = _mm256_fmadd_ps(x1, x1, vsum1);
    }
    __m256 vs = _mm256_add_ps(vsum0, vsum1);
    float tmp[8]; _mm256_storeu_ps(tmp, vs);
    float sum = tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7];
    for (; i < dim; i++) { x[i] += residual[i]; sum += x[i] * x[i]; }

    float inv_rms = 1.0f / sqrtf(sum / dim + eps);
    __m256 vinv = _mm256_set1_ps(inv_rms);
    i = 0;
    for (; i + 15 < dim; i += 16) {
        __m256 v0 = _mm256_loadu_ps(x + i);
        __m256 v1 = _mm256_loadu_ps(x + i + 8);
        __m256 w0 = _mm256_loadu_ps(weight + i);
        __m256 w1 = _mm256_loadu_ps(weight + i + 8);
        _mm256_storeu_ps(out + i,     _mm256_mul_ps(_mm256_mul_ps(v0, vinv), w0));
        _mm256_storeu_ps(out + i + 8, _mm256_mul_ps(_mm256_mul_ps(v1, vinv), w1));
    }
    for (; i < dim; i++) out[i] = x[i] * inv_rms * weight[i];
#else
    float sum = 0.0f;
    for (int i = 0; i < dim; i++) { x[i] += residual[i]; sum += x[i] * x[i]; }
    float inv_rms = 1.0f / sqrtf(sum / dim + eps);
    for (int i = 0; i < dim; i++) out[i] = x[i] * inv_rms * weight[i];
#endif
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
#elif defined(__AVX2__)
            __m256 vsum0 = _mm256_setzero_ps(), vsum1 = _mm256_setzero_ps();
            int i = 0;
            for (; i + 15 < head_dim; i += 16) {
                __m256 v0 = _mm256_loadu_ps(hs + i);
                __m256 v1 = _mm256_loadu_ps(hs + i + 8);
                vsum0 = _mm256_fmadd_ps(v0, v0, vsum0);
                vsum1 = _mm256_fmadd_ps(v1, v1, vsum1);
            }
            __m256 vs = _mm256_add_ps(vsum0, vsum1);
            float tmp[8]; _mm256_storeu_ps(tmp, vs);
            float sum = tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7];
            for (; i < head_dim; i++) sum += hs[i] * hs[i];

            float inv_rms = 1.0f / sqrtf(sum / head_dim + eps);
            __m256 vinv = _mm256_set1_ps(inv_rms);
            i = 0;
            for (; i + 15 < head_dim; i += 16) {
                __m256 v0 = _mm256_loadu_ps(hs + i);
                __m256 v1 = _mm256_loadu_ps(hs + i + 8);
                __m256 w0 = _mm256_loadu_ps(weight + i);
                __m256 w1 = _mm256_loadu_ps(weight + i + 8);
                _mm256_storeu_ps(hs + i,     _mm256_mul_ps(_mm256_mul_ps(v0, vinv), w0));
                _mm256_storeu_ps(hs + i + 8, _mm256_mul_ps(_mm256_mul_ps(v1, vinv), w1));
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

#if defined(__AVX2__)
/* Horizontal sum of an 8-wide f32 accumulator. */
static inline float qwen_hsum256_ps(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    __m128 sh = _mm_movehl_ps(lo, lo);
    lo = _mm_add_ps(lo, sh);
    sh = _mm_shuffle_ps(lo, lo, 0x1);
    lo = _mm_add_ss(lo, sh);
    return _mm_cvtss_f32(lo);
}
/* Load 8 bf16 (uint16) and widen to f32 by shifting into the high half. */
static inline __m256 qwen_loadu_bf16_8(const uint16_t *p) {
    __m128i b = _mm_loadu_si128((const __m128i *)p);
    return _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(b), 16));
}
/* Load 8 int8 and widen to f32 (sign-extended). */
static inline __m256 qwen_loadu_s8_8(const int8_t *p) {
    return _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *)p)));
}
#if defined(__AVX512F__)
/* Load 16 bf16 (uint16) and widen to 16×f32 (shift into the high half). */
static inline __m512 qwen_loadu_bf16_16(const uint16_t *p) {
    __m256i b = _mm256_loadu_si256((const __m256i *)p);
    return _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(b), 16));
}
#endif
/* f32 dot product, AVX2/FMA with scalar tail (attention score).
 * 4 accumulators (32 elem/iter) so the FMA reduction isn't latency-bound. */
static inline float qwen_dot_f32_avx2(const float *a, const float *b, int n) {
    __m256 c0 = _mm256_setzero_ps(), c1 = _mm256_setzero_ps(),
           c2 = _mm256_setzero_ps(), c3 = _mm256_setzero_ps();
    int d = 0;
    for (; d + 32 <= n; d += 32) {
        c0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + d),      _mm256_loadu_ps(b + d),      c0);
        c1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + d + 8),  _mm256_loadu_ps(b + d + 8),  c1);
        c2 = _mm256_fmadd_ps(_mm256_loadu_ps(a + d + 16), _mm256_loadu_ps(b + d + 16), c2);
        c3 = _mm256_fmadd_ps(_mm256_loadu_ps(a + d + 24), _mm256_loadu_ps(b + d + 24), c3);
    }
    for (; d + 8 <= n; d += 8)
        c0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + d), _mm256_loadu_ps(b + d), c0);
    float s = qwen_hsum256_ps(_mm256_add_ps(_mm256_add_ps(c0, c2), _mm256_add_ps(c1, c3)));
    for (; d < n; d++) s += a[d] * b[d];
    return s;
}
/* q·(bf16 k) dot product, AVX2/FMA with scalar tail (bf16-KV attention score). */
static inline float qwen_dot_f32_bf16_avx2(const float *q, const uint16_t *k, int n) {
    __m256 c0 = _mm256_setzero_ps(), c1 = _mm256_setzero_ps(),
           c2 = _mm256_setzero_ps(), c3 = _mm256_setzero_ps();
    int d = 0;
    for (; d + 32 <= n; d += 32) {
        c0 = _mm256_fmadd_ps(_mm256_loadu_ps(q + d),      qwen_loadu_bf16_8(k + d),      c0);
        c1 = _mm256_fmadd_ps(_mm256_loadu_ps(q + d + 8),  qwen_loadu_bf16_8(k + d + 8),  c1);
        c2 = _mm256_fmadd_ps(_mm256_loadu_ps(q + d + 16), qwen_loadu_bf16_8(k + d + 16), c2);
        c3 = _mm256_fmadd_ps(_mm256_loadu_ps(q + d + 24), qwen_loadu_bf16_8(k + d + 24), c3);
    }
    for (; d + 8 <= n; d += 8)
        c0 = _mm256_fmadd_ps(_mm256_loadu_ps(q + d), qwen_loadu_bf16_8(k + d), c0);
    float s = qwen_hsum256_ps(_mm256_add_ps(_mm256_add_ps(c0, c2), _mm256_add_ps(c1, c3)));
    for (; d < n; d++) s += q[d] * bf16_to_f32(k[d]);
    return s;
}
/* Attention online-softmax accumulators (AVX2). */
static inline void qwen_acc_corr_avx2(float *o, const float *v, float c, int n) {
    __m256 vc = _mm256_set1_ps(c); int d = 0;
    for (; d + 8 <= n; d += 8)
        _mm256_storeu_ps(o + d, _mm256_fmadd_ps(_mm256_loadu_ps(o + d), vc, _mm256_loadu_ps(v + d)));
    for (; d < n; d++) o[d] = o[d] * c + v[d];
}
static inline void qwen_acc_wt_avx2(float *o, const float *v, float w, int n) {
    __m256 vw = _mm256_set1_ps(w); int d = 0;
    for (; d + 8 <= n; d += 8)
        _mm256_storeu_ps(o + d, _mm256_fmadd_ps(_mm256_loadu_ps(v + d), vw, _mm256_loadu_ps(o + d)));
    for (; d < n; d++) o[d] += v[d] * w;
}
static inline void qwen_scale_avx2(float *o, float s, int n) {
    __m256 vs = _mm256_set1_ps(s); int d = 0;
    for (; d + 8 <= n; d += 8)
        _mm256_storeu_ps(o + d, _mm256_mul_ps(_mm256_loadu_ps(o + d), vs));
    for (; d < n; d++) o[d] *= s;
}
static inline void qwen_acc_corr_bf16_avx2(float *o, const uint16_t *v, float c, int n) {
    __m256 vc = _mm256_set1_ps(c); int d = 0;
    for (; d + 8 <= n; d += 8)
        _mm256_storeu_ps(o + d, _mm256_fmadd_ps(_mm256_loadu_ps(o + d), vc, qwen_loadu_bf16_8(v + d)));
    for (; d < n; d++) o[d] = o[d] * c + bf16_to_f32(v[d]);
}
static inline void qwen_acc_wt_bf16_avx2(float *o, const uint16_t *v, float w, int n) {
    __m256 vw = _mm256_set1_ps(w); int d = 0;
    for (; d + 8 <= n; d += 8)
        _mm256_storeu_ps(o + d, _mm256_fmadd_ps(qwen_loadu_bf16_8(v + d), vw, _mm256_loadu_ps(o + d)));
    for (; d < n; d++) o[d] += bf16_to_f32(v[d]) * w;
}
#endif


/* Fused bf16 matvec: processes 2 output rows at a time to amortize x vector loads.
 * On NEON: 32 elements/iter, 8 accumulators per row pair (from qwen-asr). */
static void bf16_matvec_fused(float *y, const float *x, const uint16_t *W,
                               int in_dim, int out_dim) {
    int o = 0;
#if defined(__AVX512F__)
    /* AVX-512: 2 rows, 4 __m512 accumulators/row (8 chains), 64 f32/iter, + prefetch.
     * Genuinely 16-wide on the hot path; helps where the working set fits in cache
     * (e.g. 3D V-Cache chips) and the matvec turns compute-bound. */
    for (; o + 1 < out_dim; o += 2) {
        const uint16_t *w0 = W + (size_t)o * in_dim;
        const uint16_t *w1 = W + (size_t)(o + 1) * in_dim;
        if (o + 5 < out_dim) {
            __builtin_prefetch(W + (size_t)(o + 4) * in_dim, 0, 0);
            __builtin_prefetch(W + (size_t)(o + 5) * in_dim, 0, 0);
        }
        __m512 a0 = _mm512_setzero_ps(), a1 = _mm512_setzero_ps(),
               a2 = _mm512_setzero_ps(), a3 = _mm512_setzero_ps();
        __m512 b0 = _mm512_setzero_ps(), b1 = _mm512_setzero_ps(),
               b2 = _mm512_setzero_ps(), b3 = _mm512_setzero_ps();
        int k = 0;
        for (; k + 64 <= in_dim; k += 64) {
            __m512 x0 = _mm512_loadu_ps(x + k);
            __m512 x1 = _mm512_loadu_ps(x + k + 16);
            __m512 x2 = _mm512_loadu_ps(x + k + 32);
            __m512 x3 = _mm512_loadu_ps(x + k + 48);
            a0 = _mm512_fmadd_ps(qwen_loadu_bf16_16(w0 + k),      x0, a0);
            a1 = _mm512_fmadd_ps(qwen_loadu_bf16_16(w0 + k + 16), x1, a1);
            a2 = _mm512_fmadd_ps(qwen_loadu_bf16_16(w0 + k + 32), x2, a2);
            a3 = _mm512_fmadd_ps(qwen_loadu_bf16_16(w0 + k + 48), x3, a3);
            b0 = _mm512_fmadd_ps(qwen_loadu_bf16_16(w1 + k),      x0, b0);
            b1 = _mm512_fmadd_ps(qwen_loadu_bf16_16(w1 + k + 16), x1, b1);
            b2 = _mm512_fmadd_ps(qwen_loadu_bf16_16(w1 + k + 32), x2, b2);
            b3 = _mm512_fmadd_ps(qwen_loadu_bf16_16(w1 + k + 48), x3, b3);
        }
        for (; k + 16 <= in_dim; k += 16) {
            __m512 xv = _mm512_loadu_ps(x + k);
            a0 = _mm512_fmadd_ps(qwen_loadu_bf16_16(w0 + k), xv, a0);
            b0 = _mm512_fmadd_ps(qwen_loadu_bf16_16(w1 + k), xv, b0);
        }
        a0 = _mm512_add_ps(_mm512_add_ps(a0, a2), _mm512_add_ps(a1, a3));
        b0 = _mm512_add_ps(_mm512_add_ps(b0, b2), _mm512_add_ps(b1, b3));
        float s0 = _mm512_reduce_add_ps(a0), s1 = _mm512_reduce_add_ps(b0);
        for (; k < in_dim; k++) { s0 += bf16_to_f32(w0[k]) * x[k]; s1 += bf16_to_f32(w1[k]) * x[k]; }
        y[o] = s0;
        y[o + 1] = s1;
    }
    if (o < out_dim) {
        const uint16_t *w_row = W + (size_t)o * in_dim;
        __m512 acc = _mm512_setzero_ps();
        int k = 0;
        for (; k + 16 <= in_dim; k += 16)
            acc = _mm512_fmadd_ps(qwen_loadu_bf16_16(w_row + k), _mm512_loadu_ps(x + k), acc);
        float sum = _mm512_reduce_add_ps(acc);
        for (; k < in_dim; k++) sum += bf16_to_f32(w_row[k]) * x[k];
        y[o] = sum;
    }
#elif defined(__ARM_NEON)
    /* Process 2 output rows at a time — x loaded once, reused for both rows */
    for (; o + 1 < out_dim; o += 2) {
        const uint16_t *w0 = W + (size_t)o * in_dim;
        const uint16_t *w1 = W + (size_t)(o + 1) * in_dim;
        /* Prefetch next 2 rows well ahead for the memory controller */
        if (o + 5 < out_dim) {
            const uint16_t *pf0 = W + (size_t)(o + 4) * in_dim;
            const uint16_t *pf1 = W + (size_t)(o + 5) * in_dim;
            __builtin_prefetch(pf0, 0, 0);
            __builtin_prefetch(pf0 + 64, 0, 0);
            __builtin_prefetch(pf1, 0, 0);
            __builtin_prefetch(pf1 + 64, 0, 0);
        }
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
#elif defined(__AVX2__)
    /* AVX2: 2 output rows at a time, 32 f32 elem/iter, 4 __m256 accumulators per
     * row (8 independent FMA chains) to hide the ~4-cycle FMA latency, + prefetch.
     * Mirrors the NEON path above — a single accumulator chain is latency-bound. */
    for (; o + 1 < out_dim; o += 2) {
        const uint16_t *w0 = W + (size_t)o * in_dim;
        const uint16_t *w1 = W + (size_t)(o + 1) * in_dim;
        /* Prefetch next 2 rows well ahead for the memory controller */
        if (o + 5 < out_dim) {
            const uint16_t *pf0 = W + (size_t)(o + 4) * in_dim;
            const uint16_t *pf1 = W + (size_t)(o + 5) * in_dim;
            __builtin_prefetch(pf0, 0, 0);
            __builtin_prefetch(pf0 + 64, 0, 0);
            __builtin_prefetch(pf1, 0, 0);
            __builtin_prefetch(pf1 + 64, 0, 0);
        }
        __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps(),
               a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
        __m256 b0 = _mm256_setzero_ps(), b1 = _mm256_setzero_ps(),
               b2 = _mm256_setzero_ps(), b3 = _mm256_setzero_ps();
        int k = 0;
        for (; k + 32 <= in_dim; k += 32) {
            __m256 x0 = _mm256_loadu_ps(x + k);
            __m256 x1 = _mm256_loadu_ps(x + k + 8);
            __m256 x2 = _mm256_loadu_ps(x + k + 16);
            __m256 x3 = _mm256_loadu_ps(x + k + 24);
            a0 = _mm256_fmadd_ps(qwen_loadu_bf16_8(w0 + k),      x0, a0);
            a1 = _mm256_fmadd_ps(qwen_loadu_bf16_8(w0 + k + 8),  x1, a1);
            a2 = _mm256_fmadd_ps(qwen_loadu_bf16_8(w0 + k + 16), x2, a2);
            a3 = _mm256_fmadd_ps(qwen_loadu_bf16_8(w0 + k + 24), x3, a3);
            b0 = _mm256_fmadd_ps(qwen_loadu_bf16_8(w1 + k),      x0, b0);
            b1 = _mm256_fmadd_ps(qwen_loadu_bf16_8(w1 + k + 8),  x1, b1);
            b2 = _mm256_fmadd_ps(qwen_loadu_bf16_8(w1 + k + 16), x2, b2);
            b3 = _mm256_fmadd_ps(qwen_loadu_bf16_8(w1 + k + 24), x3, b3);
        }
        for (; k + 8 <= in_dim; k += 8) {
            __m256 xv = _mm256_loadu_ps(x + k);
            a0 = _mm256_fmadd_ps(qwen_loadu_bf16_8(w0 + k), xv, a0);
            b0 = _mm256_fmadd_ps(qwen_loadu_bf16_8(w1 + k), xv, b0);
        }
        a0 = _mm256_add_ps(_mm256_add_ps(a0, a2), _mm256_add_ps(a1, a3));
        b0 = _mm256_add_ps(_mm256_add_ps(b0, b2), _mm256_add_ps(b1, b3));
        float s0 = qwen_hsum256_ps(a0), s1 = qwen_hsum256_ps(b0);
        for (; k < in_dim; k++) { s0 += bf16_to_f32(w0[k]) * x[k]; s1 += bf16_to_f32(w1[k]) * x[k]; }
        y[o] = s0;
        y[o + 1] = s1;
    }
    if (o < out_dim) {
        const uint16_t *w_row = W + (size_t)o * in_dim;
        __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
        int k = 0;
        for (; k + 16 <= in_dim; k += 16) {
            acc0 = _mm256_fmadd_ps(qwen_loadu_bf16_8(w_row + k),     _mm256_loadu_ps(x + k),     acc0);
            acc1 = _mm256_fmadd_ps(qwen_loadu_bf16_8(w_row + k + 8), _mm256_loadu_ps(x + k + 8), acc1);
        }
        for (; k + 8 <= in_dim; k += 8)
            acc0 = _mm256_fmadd_ps(qwen_loadu_bf16_8(w_row + k), _mm256_loadu_ps(x + k), acc0);
        float sum = qwen_hsum256_ps(_mm256_add_ps(acc0, acc1));
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
 * Multi-threaded via qwen_parallel (GCD on macOS, pthread pool elsewhere). */
typedef struct {
    float *y; const uint16_t *W; const float *x; int rows, cols;
} bf16_mv_ctx;
static void bf16_mv_task(size_t tid, size_t nt, void *vc) {
    bf16_mv_ctx *c = (bf16_mv_ctx *)vc;
    int r0 = (int)(tid * (size_t)c->rows / nt);
    int r1 = (int)((tid + 1) * (size_t)c->rows / nt);
    bf16_matvec_fused(c->y + r0, c->x, c->W + (size_t)r0 * c->cols, c->cols, r1 - r0);
}
/* Optional GPU offload hook: installed by the Metal/CUDA backend when --backend
 * is set. NULL = CPU default (the always-on path). Additive + opt-in — one
 * predictable, not-taken branch when GPU is off. */
void (*g_qwen_matvec_bf16_hook)(float *, const uint16_t *, const float *, int, int) = NULL;

void qwen_matvec_bf16(float *y, const uint16_t *W, const float *x, int rows, int cols) {
    if (g_qwen_matvec_bf16_hook) { g_qwen_matvec_bf16_hook(y, W, x, rows, cols); return; }
    int nt = g_n_threads;
    if (nt > 1 && rows >= 256) {
        bf16_mv_ctx c = { y, W, x, rows, cols };
        qwen_parallel((size_t)nt, bf16_mv_task, &c);
        return;
    }
    bf16_matvec_fused(y, x, W, cols, rows);
}

/* ---- Batched matmat: Y[rows,B] = W[rows,cols] @ X[cols,B] (the batching /
 * spec-decode-verify primitive). Each weight element is loaded ONCE and FMA'd
 * into all B accumulators, so W streams from DRAM exactly once regardless of B
 * (X[k][0..B] is contiguous).
 *
 * Two implementations, dispatched on B by bf16_matmat_slice():
 *  - FIXED-B specializations (B in {1..8,16}) = the production-quality path:
 *    accumulators are register-resident (named scalars, the compile-time inner
 *    loop unrolls), rows blocked 2 at a time, broadcast-FMA auto-vectorized to
 *    the target ISA by -march=native. This is where the weight-stationary win is.
 *  - GENERIC fallback (other B) = explicit NEON/AVX2/AVX-512 over the B lanes but
 *    with an L1-resident acc[] (slower; correctness safety net only).
 * The bf16 weight decode is scalar (a single shift, amortized over the B FMAs).
 *
 * TODO (newer ISAs, annotate-now / exploit-later): the bf16 decode + FMA could use
 *   - ARM bf16 BFDOT/BFMMLA (Apple M2+, Neoverse V1/V2, NVIDIA Grace, NVIDIA GPU) and
 *     i8mm SMMLA for the int8 batched twin — NEON here does a scalar bf16->f32 shift.
 *   - ARM SVE/SVE2 (Grace/Spark): vector-length-agnostic B loop.
 *   - x86 AVX-512-BF16 (VDPBF16PS) to fuse decode+FMA; AVX-512-VNNI for int8 batched.
 * Add an int8/int4 batched twin (qwen_matmat_int8/_int4) — that's where batching
 * pays most (it amortizes the unpack). See docs/batching.md. */
/* Generic batched matmat (any B up to 64). Vectorizes over the B (accumulator)
 * dimension. Used as the FALLBACK for B values without a compile-time
 * specialization below. NOTE: `acc[64]` is indexed by a runtime b, so it lives
 * in L1 (load/store every k) rather than registers — correct but slow. The
 * weight-stationary win is realized by the fixed-B specializations, where the
 * accumulators are register-resident; this generic path is just a safety net. */
static void bf16_matmat_generic(float *Y, const uint16_t *W, const float *X,
                                int r0, int r1, int cols, int B) {
    for (int r = r0; r < r1; r++) {
        const uint16_t *w = W + (size_t)r * cols;
        float *y = Y + (size_t)r * B;
        float acc[64];
        for (int b = 0; b < B; b++) acc[b] = 0.0f;
        for (int k = 0; k < cols; k++) {
            float wv = bf16_to_f32(w[k]);
            const float *xk = X + (size_t)k * B;
            int b = 0;
#if defined(__AVX512F__)
            __m512 wq16 = _mm512_set1_ps(wv);
            for (; b + 16 <= B; b += 16)
                _mm512_storeu_ps(acc + b, _mm512_fmadd_ps(wq16, _mm512_loadu_ps(xk + b), _mm512_loadu_ps(acc + b)));
#endif
#if defined(__AVX2__)
            __m256 wq8 = _mm256_set1_ps(wv);
            for (; b + 8 <= B; b += 8)
                _mm256_storeu_ps(acc + b, _mm256_fmadd_ps(wq8, _mm256_loadu_ps(xk + b), _mm256_loadu_ps(acc + b)));
#endif
#if defined(__ARM_NEON)
            float32x4_t wq4 = vdupq_n_f32(wv);
            for (; b + 4 <= B; b += 4)
                vst1q_f32(acc + b, vfmaq_f32(vld1q_f32(acc + b), wq4, vld1q_f32(xk + b)));
#endif
            for (; b < B; b++) acc[b] += wv * xk[b];
        }
        for (int b = 0; b < B; b++) y[b] = acc[b];
    }
}

/* Compile-time-B specializations — the production-quality batched matmat.
 *
 * With B a compile-time constant the inner `for (j<BV)` fully unrolls, so the
 * BV accumulators become NAMED scalars the compiler register-allocates (not an
 * L1 array), and the broadcast-FMA over the B lanes auto-vectorizes to whatever
 * the target ISA has (-march=native picks NEON / AVX2 / AVX-512 / scalar). This
 * is the multi-ISA story for the fixed-B path: one portable body, the compiler
 * emits the right SIMD — and the weight `w*v` streams from DRAM exactly once and
 * is FMA'd into all BV register-resident accumulators (weight-stationary).
 *
 * Rows are blocked 2 at a time so each X[k][0..B] load feeds two weight rows
 * (mirrors the production `bf16_matvec_fused` 2-rows-at-a-time discipline) and
 * gives two independent FMA chains to hide latency.
 *
 * Specialized for B in {1..8, 16} (the natural chunk counts for splitting one
 * long text into 2/3/4/6/8 parallel streams; the bench uses B=8). Other B fall
 * back to bf16_matmat_generic. The bf16 decode stays scalar (a single shift,
 * amortized over BV FMAs); a future ARM BFDOT / x86 AVX-512-BF16 twin could fuse
 * decode+FMA (see TODO above). */
#define DEFINE_MATMAT_FIXED_B(BV)                                              \
static void bf16_matmat_b##BV(float *Y, const uint16_t *W, const float *X,     \
                              int r0, int r1, int cols) {                      \
    int r = r0;                                                               \
    for (; r + 1 < r1; r += 2) {                                              \
        const uint16_t *w0 = W + (size_t)r * cols;                            \
        const uint16_t *w1 = W + (size_t)(r + 1) * cols;                      \
        float *y0 = Y + (size_t)r * (BV);                                     \
        float *y1 = Y + (size_t)(r + 1) * (BV);                               \
        float a[BV], b[BV];                                                   \
        for (int j = 0; j < (BV); j++) { a[j] = 0.0f; b[j] = 0.0f; }          \
        for (int k = 0; k < cols; k++) {                                      \
            float w0v = bf16_to_f32(w0[k]);                                   \
            float w1v = bf16_to_f32(w1[k]);                                   \
            const float *xk = X + (size_t)k * (BV);                           \
            for (int j = 0; j < (BV); j++) {                                  \
                float xv = xk[j];                                            \
                a[j] += w0v * xv;                                            \
                b[j] += w1v * xv;                                            \
            }                                                                 \
        }                                                                     \
        for (int j = 0; j < (BV); j++) { y0[j] = a[j]; y1[j] = b[j]; }        \
    }                                                                         \
    for (; r < r1; r++) {                                                     \
        const uint16_t *w = W + (size_t)r * cols;                             \
        float *y = Y + (size_t)r * (BV);                                      \
        float acc[BV];                                                        \
        for (int j = 0; j < (BV); j++) acc[j] = 0.0f;                         \
        for (int k = 0; k < cols; k++) {                                      \
            float wv = bf16_to_f32(w[k]);                                     \
            const float *xk = X + (size_t)k * (BV);                           \
            for (int j = 0; j < (BV); j++) acc[j] += wv * xk[j];              \
        }                                                                     \
        for (int j = 0; j < (BV); j++) y[j] = acc[j];                         \
    }                                                                         \
}
DEFINE_MATMAT_FIXED_B(1)
DEFINE_MATMAT_FIXED_B(2)
DEFINE_MATMAT_FIXED_B(3)
DEFINE_MATMAT_FIXED_B(4)
DEFINE_MATMAT_FIXED_B(5)
DEFINE_MATMAT_FIXED_B(6)
DEFINE_MATMAT_FIXED_B(7)
DEFINE_MATMAT_FIXED_B(8)
DEFINE_MATMAT_FIXED_B(16)
#undef DEFINE_MATMAT_FIXED_B

static void bf16_matmat_slice(float *Y, const uint16_t *W, const float *X,
                              int r0, int r1, int cols, int B) {
    switch (B) {
        case 1:  bf16_matmat_b1 (Y, W, X, r0, r1, cols); return;
        case 2:  bf16_matmat_b2 (Y, W, X, r0, r1, cols); return;
        case 3:  bf16_matmat_b3 (Y, W, X, r0, r1, cols); return;
        case 4:  bf16_matmat_b4 (Y, W, X, r0, r1, cols); return;
        case 5:  bf16_matmat_b5 (Y, W, X, r0, r1, cols); return;
        case 6:  bf16_matmat_b6 (Y, W, X, r0, r1, cols); return;
        case 7:  bf16_matmat_b7 (Y, W, X, r0, r1, cols); return;
        case 8:  bf16_matmat_b8 (Y, W, X, r0, r1, cols); return;
        case 16: bf16_matmat_b16(Y, W, X, r0, r1, cols); return;
        default: bf16_matmat_generic(Y, W, X, r0, r1, cols, B); return;
    }
}
typedef struct { float *Y; const uint16_t *W; const float *X; int rows, cols, B; } bf16_mm_ctx;
static void bf16_mm_task(size_t tid, size_t nt, void *vc) {
    bf16_mm_ctx *c = (bf16_mm_ctx *)vc;
    int r0 = (int)(tid * (size_t)c->rows / nt);
    int r1 = (int)((tid + 1) * (size_t)c->rows / nt);
    bf16_matmat_slice(c->Y, c->W, c->X, r0, r1, c->cols, c->B);
}
/* Optional GPU offload hook for the batched matmat (server --batch-size path).
 * NULL = CPU default. This is where the GPU's matrix-unit (MMA) win lands. */
void (*g_qwen_matmat_bf16_hook)(float *, const uint16_t *, const float *, int, int, int) = NULL;

void qwen_matmat_bf16(float *Y, const uint16_t *W, const float *X, int rows, int cols, int B) {
    if (g_qwen_matmat_bf16_hook) { g_qwen_matmat_bf16_hook(Y, W, X, rows, cols, B); return; }
    if (B <= 0) return;
    if (B > 64) B = 64;  /* contract: B<=64 */
    int nt = g_n_threads;
    if (nt > 1 && rows >= 256) {
        bf16_mm_ctx c = { Y, W, X, rows, cols, B };
        qwen_parallel((size_t)nt, bf16_mm_task, &c);
        return;
    }
    bf16_matmat_slice(Y, W, X, 0, rows, cols, B);
}

/* ---- INT8 batched matmat twin: Y[rows,B] = (W_int8[rows,cols]*scale[rows]) @ X[cols,B]
 * Mirrors qwen_matvec_int8's ARM semantics (int8 weight -> f32, f32 activation,
 * accumulate in f32, * per-row scale). Weight-stationary: each int8 weight (half
 * the bytes of bf16) streams from DRAM once and is FMA'd into all B accumulators.
 * Same compile-time-B register-blocking discipline as bf16; generic fallback for
 * other B. The activation is kept f32 (no per-column requant) so this is bit-
 * comparable to B independent qwen_matvec_int8 calls (fp-order aside). */
static void int8_matmat_generic(float *Y, const int8_t *W, const float *scale,
                                const float *X, int r0, int r1, int cols, int B) {
    for (int r = r0; r < r1; r++) {
        const int8_t *w = W + (size_t)r * cols;
        float *y = Y + (size_t)r * B;
        float acc[64];
        for (int b = 0; b < B; b++) acc[b] = 0.0f;
        for (int k = 0; k < cols; k++) {
            float wv = (float)w[k];
            const float *xk = X + (size_t)k * B;
            for (int b = 0; b < B; b++) acc[b] += wv * xk[b];
        }
        float s = scale[r];
        for (int b = 0; b < B; b++) y[b] = acc[b] * s;
    }
}
#define DEFINE_MATMAT_INT8_FIXED_B(BV)                                         \
static void int8_matmat_b##BV(float *Y, const int8_t *W, const float *scale,    \
                              const float *X, int r0, int r1, int cols) {      \
    int r = r0;                                                               \
    for (; r + 1 < r1; r += 2) {                                              \
        const int8_t *w0 = W + (size_t)r * cols;                              \
        const int8_t *w1 = W + (size_t)(r + 1) * cols;                        \
        float *y0 = Y + (size_t)r * (BV);                                     \
        float *y1 = Y + (size_t)(r + 1) * (BV);                               \
        float a[BV], b[BV];                                                   \
        for (int j = 0; j < (BV); j++) { a[j] = 0.0f; b[j] = 0.0f; }          \
        for (int k = 0; k < cols; k++) {                                      \
            float w0v = (float)w0[k], w1v = (float)w1[k];                     \
            const float *xk = X + (size_t)k * (BV);                           \
            for (int j = 0; j < (BV); j++) {                                  \
                float xv = xk[j]; a[j] += w0v * xv; b[j] += w1v * xv;         \
            }                                                                 \
        }                                                                     \
        float s0 = scale[r], s1 = scale[r + 1];                              \
        for (int j = 0; j < (BV); j++) { y0[j] = a[j] * s0; y1[j] = b[j] * s1; } \
    }                                                                         \
    for (; r < r1; r++) {                                                     \
        const int8_t *w = W + (size_t)r * cols;                              \
        float *y = Y + (size_t)r * (BV);                                     \
        float acc[BV];                                                        \
        for (int j = 0; j < (BV); j++) acc[j] = 0.0f;                         \
        for (int k = 0; k < cols; k++) {                                      \
            float wv = (float)w[k];                                          \
            const float *xk = X + (size_t)k * (BV);                           \
            for (int j = 0; j < (BV); j++) acc[j] += wv * xk[j];              \
        }                                                                     \
        float s = scale[r];                                                  \
        for (int j = 0; j < (BV); j++) y[j] = acc[j] * s;                     \
    }                                                                         \
}
DEFINE_MATMAT_INT8_FIXED_B(2)
DEFINE_MATMAT_INT8_FIXED_B(3)
DEFINE_MATMAT_INT8_FIXED_B(4)
DEFINE_MATMAT_INT8_FIXED_B(6)
DEFINE_MATMAT_INT8_FIXED_B(8)
DEFINE_MATMAT_INT8_FIXED_B(16)
#undef DEFINE_MATMAT_INT8_FIXED_B
static void int8_matmat_slice(float *Y, const int8_t *W, const float *scale,
                              const float *X, int r0, int r1, int cols, int B) {
    qwen_ftz_on();
    switch (B) {
        case 2:  int8_matmat_b2 (Y, W, scale, X, r0, r1, cols); return;
        case 3:  int8_matmat_b3 (Y, W, scale, X, r0, r1, cols); return;
        case 4:  int8_matmat_b4 (Y, W, scale, X, r0, r1, cols); return;
        case 6:  int8_matmat_b6 (Y, W, scale, X, r0, r1, cols); return;
        case 8:  int8_matmat_b8 (Y, W, scale, X, r0, r1, cols); return;
        case 16: int8_matmat_b16(Y, W, scale, X, r0, r1, cols); return;
        default: int8_matmat_generic(Y, W, scale, X, r0, r1, cols, B); return;
    }
}
typedef struct { float *Y; const int8_t *W; const float *scale; const float *X; int rows, cols, B; } int8_mm_ctx;
static void int8_mm_task(size_t tid, size_t nt, void *vc) {
    int8_mm_ctx *c = (int8_mm_ctx *)vc;
    int r0 = (int)(tid * (size_t)c->rows / nt);
    int r1 = (int)((tid + 1) * (size_t)c->rows / nt);
    int8_matmat_slice(c->Y, c->W, c->scale, c->X, r0, r1, c->cols, c->B);
}

#if defined(__ARM_FEATURE_DOTPROD)
/* ── int8 SDOT batched twin (#3) — weight-stationary native int8 dot ──────────
 * Y[rows,B] = (W_int8 @ qXt^T) · scales. Activations are pre-quantized per column
 * to int8 (qXt[b][cols], scale sx[b]); each weight 16-block is loaded ONCE and
 * SDOT-dotted against all B activation blocks. Amortizes the weight read (bandwidth)
 * AND keeps SDOT (compute) — unlike int8_matmat_slice, which dequants to f32 and
 * loses SDOT, so int8+batch was SLOWER than int8-single on M1 (long-form A/B 0.81×). */
static void int8_matmat_sdot_slice(float *Y, const int8_t *W, const float *scale,
                                   const int8_t *qXt, const float *sx,
                                   int r0, int r1, int cols, int B) {
    qwen_ftz_on();
    for (int r = r0; r < r1; r++) {
        const int8_t *w = W + (size_t)r * cols;
        int32x4_t acc[16];
        for (int b = 0; b < B; b++) acc[b] = vdupq_n_s32(0);
        int k = 0;
        for (; k + 15 < cols; k += 16) {
            int8x16_t wv = vld1q_s8(w + k);              /* weight block: loaded once */
            for (int b = 0; b < B; b++)
                acc[b] = vdotq_s32(acc[b], wv, vld1q_s8(qXt + (size_t)b * cols + k));
        }
        float s = scale[r];
        for (int b = 0; b < B; b++) {
            int32_t sum = vaddvq_s32(acc[b]);
            const int8_t *qb = qXt + (size_t)b * cols;
            for (int kk = k; kk < cols; kk++) sum += (int32_t)w[kk] * qb[kk];
            Y[(size_t)r * B + b] = (float)sum * s * sx[b];
        }
    }
}
typedef struct { float *Y; const int8_t *W; const float *scale; const int8_t *qXt; const float *sx; int rows, cols, B; } int8_smm_ctx;
static void int8_smm_task(size_t tid, size_t nt, void *vc) {
    int8_smm_ctx *c = (int8_smm_ctx *)vc;
    int r0 = (int)(tid * (size_t)c->rows / nt);
    int r1 = (int)((tid + 1) * (size_t)c->rows / nt);
    int8_matmat_sdot_slice(c->Y, c->W, c->scale, c->qXt, c->sx, r0, r1, c->cols, c->B);
}
#endif /* __ARM_FEATURE_DOTPROD */

#if defined(__AVX512VNNI__)
/* Quantize column b of the [cols][B] activation matrix X to int8 (per-column absmax). */
static float quantize_act_int8_col(int8_t *qb, const float *X, int cols, int B, int b) {
    float amax = 0.0f;
    for (int k = 0; k < cols; k++) { float a = fabsf(X[(size_t)k * B + b]); if (a > amax) amax = a; }
    if (amax == 0.0f) { memset(qb, 0, (size_t)cols); return 0.0f; }
    float inv = 127.0f / amax;
    for (int k = 0; k < cols; k++) {
        int v = (int)lrintf(X[(size_t)k * B + b] * inv);
        qb[k] = (int8_t)(v > 127 ? 127 : (v < -128 ? -128 : v));
    }
    return amax / 127.0f;
}
/* ── int8 VNNI batched matmat (the x86 int8 GEMM the SDOT comment above asks for) ──
 * Y[rows,B] = (W_int8 @ qXt^T)·scales. Weight-stationary: each 64-int8 W block is
 * loaded ONCE and dpbusd'd against all B pre-quantized activation columns. VNNI is
 * unsigned×signed → activations passed pre-offset as ua = qXt+128 (unsigned), corrected
 * −128·Σw per row (ws = dpbusd(ones,w)), exactly like the single-stream int8_matvec_vnni. */
static void int8_matmat_vnni_slice(float *Y, const int8_t *W, const float *scale,
                                   const int8_t *qXt, const float *sx,
                                   int r0, int r1, int cols, int B) {
    const __m512i ones = _mm512_set1_epi8(1);
    const __m512i v128 = _mm512_set1_epi8((char)128);
    for (int r = r0; r < r1; r++) {
        const int8_t *w = W + (size_t)r * cols;
        __m512i acc[16], ws = _mm512_setzero_si512();
        for (int b = 0; b < B; b++) acc[b] = _mm512_setzero_si512();
        int k = 0;
        for (; k + 64 <= cols; k += 64) {
            __m512i wv = _mm512_loadu_si512((const void *)(w + k));   /* weight block: once */
            ws = _mm512_dpbusd_epi32(ws, ones, wv);
            for (int b = 0; b < B; b++) {
                __m512i ua = _mm512_add_epi8(_mm512_loadu_si512((const void *)(qXt + (size_t)b * cols + k)), v128);
                acc[b] = _mm512_dpbusd_epi32(acc[b], ua, wv);
            }
        }
        int sw = _mm512_reduce_add_epi32(ws);
        float s = scale[r];
        for (int b = 0; b < B; b++) {
            int sum = _mm512_reduce_add_epi32(acc[b]) - 128 * sw;
            const int8_t *qb = qXt + (size_t)b * cols;
            for (int kk = k; kk < cols; kk++) sum += (int)w[kk] * (int)qb[kk];  /* signed tail */
            Y[(size_t)r * B + b] = (float)sum * s * sx[b];
        }
    }
}
typedef struct { float *Y; const int8_t *W; const float *scale; const int8_t *qXt; const float *sx; int rows, cols, B; } int8_vmm_ctx;
static void int8_vmm_task(size_t tid, size_t nt, void *vc) {
    int8_vmm_ctx *c = (int8_vmm_ctx *)vc;
    int r0 = (int)(tid * (size_t)c->rows / nt);
    int r1 = (int)((tid + 1) * (size_t)c->rows / nt);
    int8_matmat_vnni_slice(c->Y, c->W, c->scale, c->qXt, c->sx, r0, r1, c->cols, c->B);
}
#endif /* __AVX512VNNI__ */

void qwen_matmat_int8(float *Y, const int8_t *W, const float *scale,
                      const float *X, int rows, int cols, int B) {
    if (B <= 0) return;
    if (B > 64) B = 64;
#if defined(__AVX512VNNI__)
    /* x86 VNNI batched int8 GEMM — default ON (QWEN_NO_VNNI=1 disables), the right
     * int8 matmat primitive on x86 (the ARM SDOT path below stays M1 opt-in). */
    {
        static atomic_int no_vnni = -1;
        int nv = atomic_load_explicit(&no_vnni, memory_order_relaxed);
        if (nv < 0) { const char *e = getenv("QWEN_NO_VNNI"); nv = (e && e[0] == '1'); atomic_store_explicit(&no_vnni, nv, memory_order_relaxed); }
        if (!nv && B >= 2 && B <= 16) {
            int8_t *qXt = (int8_t *)malloc((size_t)B * cols);
            if (qXt) {
                float sx[16];
                for (int b = 0; b < B; b++)
                    sx[b] = quantize_act_int8_col(qXt + (size_t)b * cols, X, cols, B, b);
                int nt = g_n_threads;
                if (nt > 1 && rows >= 256) {
                    int8_vmm_ctx c = { Y, W, scale, qXt, sx, rows, cols, B };
                    qwen_parallel((size_t)nt, int8_vmm_task, &c);
                } else {
                    int8_matmat_vnni_slice(Y, W, scale, qXt, sx, 0, rows, cols, B);
                }
                free(qXt);
                return;
            }
        }
    }
#endif
#if defined(__ARM_FEATURE_DOTPROD)
    /* SDOT batched path (#3) — OPT-IN (QWEN_INT8_SDOT_MM=1), default OFF.
     * MEASURED slower on M1 than the f32-accum batched twin below: SDOT contracts
     * over the reduction dim k, but batching wants to parallelize over B, so this
     * does B sequential vdotq per weight block (B not vectorized) and loses to the
     * f32-accum path that vectorizes over B. Kept as a bit-exact A/B reference (it
     * equals B×int8-matvec-SDOT, self-test L2=0) for M2+/x86, where the RIGHT int8
     * matrix-matrix primitive is i8mm SMMLA / AVX-512 VNNI (true int8 GEMM), not a
     * looped SDOT. On M1, int8+batch doesn't win (SDOT-seq is already near-optimal);
     * batching pays on bf16. See PLAN batching #3 finding. */
    {
        static int sdot_mm = -1;
        if (sdot_mm < 0) { const char *e = getenv("QWEN_INT8_SDOT_MM"); sdot_mm = (e && e[0] == '1'); }
        if (sdot_mm && B >= 2 && B <= 16) {
            int8_t *qXt = (int8_t *)malloc((size_t)B * cols);
            if (qXt) {
                float sx[16];
                for (int b = 0; b < B; b++) {
                    float amax = 0.0f;
                    for (int k = 0; k < cols; k++) { float a = fabsf(X[(size_t)k * B + b]); if (a > amax) amax = a; }
                    int8_t *qb = qXt + (size_t)b * cols;
                    if (amax == 0.0f) { memset(qb, 0, (size_t)cols); sx[b] = 0.0f; continue; }
                    float inv = 127.0f / amax;
                    for (int k = 0; k < cols; k++) {
                        int v = (int)lrintf(X[(size_t)k * B + b] * inv);
                        qb[k] = (int8_t)(v > 127 ? 127 : (v < -128 ? -128 : v));
                    }
                    sx[b] = amax / 127.0f;
                }
                int nt = g_n_threads;
                if (nt > 1 && rows >= 256) {
                    int8_smm_ctx c = { Y, W, scale, qXt, sx, rows, cols, B };
                    qwen_parallel((size_t)nt, int8_smm_task, &c);
                } else {
                    int8_matmat_sdot_slice(Y, W, scale, qXt, sx, 0, rows, cols, B);
                }
                free(qXt);
                return;
            }
        }
    }
#endif
    int nt = g_n_threads;
    if (nt > 1 && rows >= 256) {
        int8_mm_ctx c = { Y, W, scale, X, rows, cols, B };
        qwen_parallel((size_t)nt, int8_mm_task, &c);
        return;
    }
    int8_matmat_slice(Y, W, scale, X, 0, rows, cols, B);
}

/* ---- INT4 (Q4_0) batched matmat twin: Y[rows,B] = dequant(W_q4) @ X[cols,B]
 * THE big batching synergy: the per-nibble UNPACK ((qs>>shift & 0xF) - 8) * scale)
 * — which dominates the single-stream q4_0 GEMV and is REDONE per token there — is
 * done ONCE here and reused across all B columns (weight + unpack amortized over B).
 * Per the PLAN this is where batching pays most, and where int4 could become viable
 * on M1 (nibble-unpack is exactly what makes int4 slow single-stream). 1-row blocked
 * (the unpack is per-row; B accumulators stay register-resident at fixed B). */
static void q4_matmat_generic(float *Y, const q4_0_block_t *W, const float *X,
                              int r0, int r1, int cols, int B) {
    int nb = cols / Q4_0_BLOCK_SIZE;
    for (int r = r0; r < r1; r++) {
        const q4_0_block_t *wr = W + (size_t)r * nb;
        float *y = Y + (size_t)r * B;
        float acc[64];
        for (int b = 0; b < B; b++) acc[b] = 0.0f;
        for (int bl = 0; bl < nb; bl++) {
            float sc = wr[bl].scale;
            const uint8_t *qs = wr[bl].qs;
            int k0 = bl * Q4_0_BLOCK_SIZE;
            for (int i = 0; i < 16; i++) {
                float wlo = (float)((qs[i] & 0x0F) - 8) * sc;
                float whi = (float)((qs[i] >> 4)   - 8) * sc;
                const float *xl = X + (size_t)(k0 + 2 * i) * B;
                const float *xh = X + (size_t)(k0 + 2 * i + 1) * B;
                for (int b = 0; b < B; b++) acc[b] += wlo * xl[b] + whi * xh[b];
            }
        }
        for (int b = 0; b < B; b++) y[b] = acc[b];
    }
}
#define DEFINE_MATMAT_Q4_FIXED_B(BV)                                           \
static void q4_matmat_b##BV(float *Y, const q4_0_block_t *W, const float *X,    \
                            int r0, int r1, int cols) {                        \
    int nb = cols / Q4_0_BLOCK_SIZE;                                          \
    for (int r = r0; r < r1; r++) {                                           \
        const q4_0_block_t *wr = W + (size_t)r * nb;                          \
        float *y = Y + (size_t)r * (BV);                                      \
        float acc[BV];                                                        \
        for (int j = 0; j < (BV); j++) acc[j] = 0.0f;                         \
        for (int bl = 0; bl < nb; bl++) {                                     \
            float sc = wr[bl].scale;                                          \
            const uint8_t *qs = wr[bl].qs;                                    \
            int k0 = bl * Q4_0_BLOCK_SIZE;                                    \
            for (int i = 0; i < 16; i++) {                                    \
                float wlo = (float)((qs[i] & 0x0F) - 8) * sc;                 \
                float whi = (float)((qs[i] >> 4)   - 8) * sc;                 \
                const float *xl = X + (size_t)(k0 + 2 * i) * (BV);            \
                const float *xh = X + (size_t)(k0 + 2 * i + 1) * (BV);        \
                for (int j = 0; j < (BV); j++) acc[j] += wlo * xl[j] + whi * xh[j]; \
            }                                                                 \
        }                                                                     \
        for (int j = 0; j < (BV); j++) y[j] = acc[j];                         \
    }                                                                         \
}
DEFINE_MATMAT_Q4_FIXED_B(2)
DEFINE_MATMAT_Q4_FIXED_B(3)
DEFINE_MATMAT_Q4_FIXED_B(4)
DEFINE_MATMAT_Q4_FIXED_B(6)
DEFINE_MATMAT_Q4_FIXED_B(8)
DEFINE_MATMAT_Q4_FIXED_B(16)
#undef DEFINE_MATMAT_Q4_FIXED_B
static void q4_matmat_slice(float *Y, const q4_0_block_t *W, const float *X,
                            int r0, int r1, int cols, int B) {
    qwen_ftz_on();
    switch (B) {
        case 2:  q4_matmat_b2 (Y, W, X, r0, r1, cols); return;
        case 3:  q4_matmat_b3 (Y, W, X, r0, r1, cols); return;
        case 4:  q4_matmat_b4 (Y, W, X, r0, r1, cols); return;
        case 6:  q4_matmat_b6 (Y, W, X, r0, r1, cols); return;
        case 8:  q4_matmat_b8 (Y, W, X, r0, r1, cols); return;
        case 16: q4_matmat_b16(Y, W, X, r0, r1, cols); return;
        default: q4_matmat_generic(Y, W, X, r0, r1, cols, B); return;
    }
}
typedef struct { float *Y; const q4_0_block_t *W; const float *X; int rows, cols, B; } q4_mm_ctx;
static void q4_mm_task(size_t tid, size_t nt, void *vc) {
    q4_mm_ctx *c = (q4_mm_ctx *)vc;
    int r0 = (int)(tid * (size_t)c->rows / nt);
    int r1 = (int)((tid + 1) * (size_t)c->rows / nt);
    q4_matmat_slice(c->Y, c->W, c->X, r0, r1, c->cols, c->B);
}
#if defined(__AVX512VNNI__)
/* q4 VNNI batched matmat (mirrors int8_matmat_vnni_slice + the C7-v2 unsigned-nibble
 * trick): weight-stationary — unpack each 32-nibble W block ONCE to u8 value order,
 * dpbusd against all B pre-quantized activation columns. q4 per-block scale + per-column
 * act scale; corr[b][bl] = −8·ΣqXt[b] over block bl (precomputed once, shared across rows). */
static void q4_matmat_vnni_slice(float *Y, const q4_0_block_t *W, const int8_t *qXt,
                                 const float *sx, const int *corr,
                                 int r0, int r1, int cols, int B) {
    int nb = cols / Q4_0_BLOCK_SIZE;
    const __m128i lomask = _mm_set1_epi8(0x0F);
    for (int r = r0; r < r1; r++) {
        const q4_0_block_t *row = W + (size_t)r * nb;
        float sum[16];
        for (int b = 0; b < B; b++) sum[b] = 0.0f;
        for (int bl = 0; bl < nb; bl++) {
            __m128i raw = _mm_loadu_si128((const __m128i *)row[bl].qs);
            __m128i lo = _mm_and_si128(raw, lomask);
            __m128i hi = _mm_and_si128(_mm_srli_epi16(raw, 4), lomask);
            __m512i wv = _mm512_zextsi256_si512(_mm256_set_m128i(_mm_unpackhi_epi8(lo, hi),
                                                                 _mm_unpacklo_epi8(lo, hi)));
            float scl = row[bl].scale;
            for (int b = 0; b < B; b++) {
                __m512i xv = _mm512_zextsi256_si512(_mm256_loadu_si256(
                    (const __m256i *)(qXt + (size_t)b * cols + (size_t)bl * Q4_0_BLOCK_SIZE)));
                int dot = _mm512_reduce_add_epi32(_mm512_dpbusd_epi32(_mm512_setzero_si512(), wv, xv))
                        + corr[(size_t)b * nb + bl];
                sum[b] += scl * (float)dot;
            }
        }
        for (int b = 0; b < B; b++) Y[(size_t)r * B + b] = sum[b] * sx[b];
    }
}
typedef struct { float *Y; const q4_0_block_t *W; const int8_t *qXt; const float *sx; const int *corr; int rows, cols, B; } q4_vmm_ctx;
static void q4_vmm_task(size_t tid, size_t nt, void *vc) {
    q4_vmm_ctx *c = (q4_vmm_ctx *)vc;
    int r0 = (int)(tid * (size_t)c->rows / nt);
    int r1 = (int)((tid + 1) * (size_t)c->rows / nt);
    q4_matmat_vnni_slice(c->Y, c->W, c->qXt, c->sx, c->corr, r0, r1, c->cols, c->B);
}
#endif /* __AVX512VNNI__ */
void qwen_matmat_q4_0(float *Y, const q4_0_block_t *W, const float *X,
                      int rows, int cols, int B) {
    if (B <= 0) return;
    if (B > 64) B = 64;
#if defined(__AVX512VNNI__)
    {
        static atomic_int no_vnni_q4 = -1;
        int nv = atomic_load_explicit(&no_vnni_q4, memory_order_relaxed);
        if (nv < 0) { const char *e = getenv("QWEN_NO_VNNI"); nv = (e && e[0] == '1'); atomic_store_explicit(&no_vnni_q4, nv, memory_order_relaxed); }
        if (!nv && B >= 2 && B <= 16 && cols % Q4_0_BLOCK_SIZE == 0) {
            int nb = cols / Q4_0_BLOCK_SIZE;
            int8_t *qXt = (int8_t *)malloc((size_t)B * cols);
            int *corr = (int *)malloc((size_t)B * nb * sizeof(int));
            if (qXt && corr) {
                float sx[16];
                for (int b = 0; b < B; b++) {
                    sx[b] = quantize_act_int8_col(qXt + (size_t)b * cols, X, cols, B, b);
                    const int8_t *qb = qXt + (size_t)b * cols;
                    for (int bl = 0; bl < nb; bl++) {
                        int s = 0;
                        for (int k = 0; k < Q4_0_BLOCK_SIZE; k++) s += qb[bl * Q4_0_BLOCK_SIZE + k];
                        corr[(size_t)b * nb + bl] = -8 * s;
                    }
                }
                int nt2 = g_n_threads;
                if (nt2 > 1 && rows >= 256) {
                    q4_vmm_ctx c = { Y, W, qXt, sx, corr, rows, cols, B };
                    qwen_parallel((size_t)nt2, q4_vmm_task, &c);
                } else {
                    q4_matmat_vnni_slice(Y, W, qXt, sx, corr, 0, rows, cols, B);
                }
                free(qXt); free(corr);
                return;
            }
            free(qXt); free(corr);
        }
    }
#endif
    int nt = g_n_threads;
    if (nt > 1 && rows >= 256) {
        q4_mm_ctx c = { Y, W, X, rows, cols, B };
        qwen_parallel((size_t)nt, q4_mm_task, &c);
        return;
    }
    q4_matmat_slice(Y, W, X, 0, rows, cols, B);
}

/* Unified QKV matvec: single parallel-for for Q, K, V projections.
 * The concatenated [Q|K|V] row space is partitioned for balance, avoiding 3
 * separate barriers per layer. */
typedef struct {
    float *q, *k, *v;
    const uint16_t *Wq, *Wk, *Wv;
    const float *x;
    int in_dim, q_dim, kv_dim;
} bf16_qkv_ctx;
static void bf16_qkv_task(size_t tid, size_t nt, void *vc) {
    bf16_qkv_ctx *c = (bf16_qkv_ctx *)vc;
    int total_dim = c->q_dim + 2 * c->kv_dim;
    int r0 = (int)(tid * (size_t)total_dim / nt);
    int r1 = (int)((tid + 1) * (size_t)total_dim / nt);
    for (int r = r0; r < r1; ) {
        if (r < c->q_dim) {
            int chunk_end = r1 < c->q_dim ? r1 : c->q_dim;
            bf16_matvec_fused(c->q + r, c->x, c->Wq + (size_t)r * c->in_dim,
                               c->in_dim, chunk_end - r);
            r = chunk_end;
        } else if (r < c->q_dim + c->kv_dim) {
            int local = r - c->q_dim;
            int chunk_end = r1 < c->q_dim + c->kv_dim ? r1 : c->q_dim + c->kv_dim;
            int local_end = chunk_end - c->q_dim;
            bf16_matvec_fused(c->k + local, c->x, c->Wk + (size_t)local * c->in_dim,
                               c->in_dim, local_end - local);
            r = chunk_end;
        } else {
            int local = r - c->q_dim - c->kv_dim;
            int local_end = r1 - c->q_dim - c->kv_dim;
            bf16_matvec_fused(c->v + local, c->x, c->Wv + (size_t)local * c->in_dim,
                               c->in_dim, local_end - local);
            r = r1;
        }
    }
}
void qwen_matvec_bf16_qkv(float *q, float *k, float *v,
                           const uint16_t *Wq, const uint16_t *Wk, const uint16_t *Wv,
                           const float *x, int in_dim, int q_dim, int kv_dim) {
    if (g_qwen_matvec_bf16_hook) {
        g_qwen_matvec_bf16_hook(q, Wq, x, q_dim, in_dim);
        g_qwen_matvec_bf16_hook(k, Wk, x, kv_dim, in_dim);
        g_qwen_matvec_bf16_hook(v, Wv, x, kv_dim, in_dim);
        return;
    }
    int nt = g_n_threads;
    int total_dim = q_dim + 2 * kv_dim;
    if (nt > 1 && total_dim >= 256) {
        bf16_qkv_ctx c = { q, k, v, Wq, Wk, Wv, x, in_dim, q_dim, kv_dim };
        qwen_parallel((size_t)nt, bf16_qkv_task, &c);
        return;
    }
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
#elif defined(__AVX2__)
        __m256 vmax = _mm256_setzero_ps();
        const __m256 signmask = _mm256_set1_ps(-0.0f);
        int k = 0;
        for (; k + 7 < cols; k += 8)
            vmax = _mm256_max_ps(vmax, _mm256_andnot_ps(signmask, qwen_loadu_bf16_8(row + k)));
        float mtmp[8]; _mm256_storeu_ps(mtmp, vmax);
        for (int j = 0; j < 8; j++) if (mtmp[j] > amax) amax = mtmp[j];
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
#elif defined(__AVX2__)
        __m256 vinv = _mm256_set1_ps(inv_s);
        k = 0;
        for (; k + 7 < cols; k += 8) {
            __m256i q = _mm256_cvtps_epi32(_mm256_mul_ps(qwen_loadu_bf16_8(row + k), vinv));
            __m128i q16 = _mm_packs_epi32(_mm256_castsi256_si128(q),
                                          _mm256_extracti128_si256(q, 1));
            _mm_storel_epi64((__m128i *)(dst_row + k), _mm_packs_epi16(q16, q16));
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
    qwen_ftz_on();  /* runs on each GCD worker — flush int8-induced denormals */
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
#elif defined(__AVX2__)
    /* AVX2: 2 rows at a time, 32 int8/iter, 4 __m256 accumulators per row (8 chains)
     * + prefetch — mirrors the NEON path so x86 hides FMA latency too. */
    for (; o + 1 < out_dim; o += 2) {
        const int8_t *w0 = W + (size_t)o * in_dim;
        const int8_t *w1 = W + (size_t)(o + 1) * in_dim;
        if (o + 5 < out_dim) {
            __builtin_prefetch(W + (size_t)(o + 4) * in_dim, 0, 0);
            __builtin_prefetch(W + (size_t)(o + 5) * in_dim, 0, 0);
        }
        __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps(),
               a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
        __m256 b0 = _mm256_setzero_ps(), b1 = _mm256_setzero_ps(),
               b2 = _mm256_setzero_ps(), b3 = _mm256_setzero_ps();
        int k = 0;
        for (; k + 32 <= in_dim; k += 32) {
            __m256 x0 = _mm256_loadu_ps(x + k);
            __m256 x1 = _mm256_loadu_ps(x + k + 8);
            __m256 x2 = _mm256_loadu_ps(x + k + 16);
            __m256 x3 = _mm256_loadu_ps(x + k + 24);
            a0 = _mm256_fmadd_ps(qwen_loadu_s8_8(w0 + k),      x0, a0);
            a1 = _mm256_fmadd_ps(qwen_loadu_s8_8(w0 + k + 8),  x1, a1);
            a2 = _mm256_fmadd_ps(qwen_loadu_s8_8(w0 + k + 16), x2, a2);
            a3 = _mm256_fmadd_ps(qwen_loadu_s8_8(w0 + k + 24), x3, a3);
            b0 = _mm256_fmadd_ps(qwen_loadu_s8_8(w1 + k),      x0, b0);
            b1 = _mm256_fmadd_ps(qwen_loadu_s8_8(w1 + k + 8),  x1, b1);
            b2 = _mm256_fmadd_ps(qwen_loadu_s8_8(w1 + k + 16), x2, b2);
            b3 = _mm256_fmadd_ps(qwen_loadu_s8_8(w1 + k + 24), x3, b3);
        }
        for (; k + 8 <= in_dim; k += 8) {
            __m256 xv = _mm256_loadu_ps(x + k);
            a0 = _mm256_fmadd_ps(qwen_loadu_s8_8(w0 + k), xv, a0);
            b0 = _mm256_fmadd_ps(qwen_loadu_s8_8(w1 + k), xv, b0);
        }
        a0 = _mm256_add_ps(_mm256_add_ps(a0, a2), _mm256_add_ps(a1, a3));
        b0 = _mm256_add_ps(_mm256_add_ps(b0, b2), _mm256_add_ps(b1, b3));
        float s0 = qwen_hsum256_ps(a0), s1 = qwen_hsum256_ps(b0);
        for (; k < in_dim; k++) { s0 += (float)w0[k] * x[k]; s1 += (float)w1[k] * x[k]; }
        y[o] = s0 * scale[o];
        y[o + 1] = s1 * scale[o + 1];
    }
    if (o < out_dim) {
        const int8_t *w_row = W + (size_t)o * in_dim;
        __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
        int k = 0;
        for (; k + 16 <= in_dim; k += 16) {
            acc0 = _mm256_fmadd_ps(qwen_loadu_s8_8(w_row + k),     _mm256_loadu_ps(x + k),     acc0);
            acc1 = _mm256_fmadd_ps(qwen_loadu_s8_8(w_row + k + 8), _mm256_loadu_ps(x + k + 8), acc1);
        }
        for (; k + 8 <= in_dim; k += 8)
            acc0 = _mm256_fmadd_ps(qwen_loadu_s8_8(w_row + k), _mm256_loadu_ps(x + k), acc0);
        float sum = qwen_hsum256_ps(_mm256_add_ps(acc0, acc1));
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

#if defined(__ARM_FEATURE_DOTPROD)
/* Dynamically quantize an f32 activation vector to int8 (per-vector absmax).
 * Returns the scale (amax/127); writes int8 codes into qx[n]. This is the
 * activation half that native int8 dot (SDOT) needs: SDOT multiplies int8×int8,
 * so x must be int8 too (the current dequant→f32→FMA path kept x in f32). */
static float quantize_act_int8(int8_t *qx, const float *x, int n) {
    float amax = 0.0f;
    int i = 0;
    float32x4_t vmax = vdupq_n_f32(0);
    for (; i + 3 < n; i += 4)
        vmax = vmaxq_f32(vmax, vabsq_f32(vld1q_f32(x + i)));
    amax = vmaxvq_f32(vmax);
    for (; i < n; i++) { float a = fabsf(x[i]); if (a > amax) amax = a; }
    if (amax == 0.0f) { memset(qx, 0, (size_t)n); return 0.0f; }
    float inv = 127.0f / amax;
    float32x4_t vinv = vdupq_n_f32(inv);
    i = 0;
    for (; i + 15 < n; i += 16) {
        int32x4_t q0 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(x + i),      vinv));
        int32x4_t q1 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(x + i + 4),  vinv));
        int32x4_t q2 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(x + i + 8),  vinv));
        int32x4_t q3 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(x + i + 12), vinv));
        int16x8_t s01 = vcombine_s16(vqmovn_s32(q0), vqmovn_s32(q1));
        int16x8_t s23 = vcombine_s16(vqmovn_s32(q2), vqmovn_s32(q3));
        vst1q_s8(qx + i, vcombine_s8(vqmovn_s16(s01), vqmovn_s16(s23)));
    }
    for (; i < n; i++) {
        int v = (int)lrintf(x[i] * inv);
        qx[i] = (int8_t)(v > 127 ? 127 : (v < -128 ? -128 : v));
    }
    return amax / 127.0f;
}

/* Native int8 dot matvec via SDOT: y[o] = scale[o] * sx * Σ_k W[o][k]·qx[k].
 * 4 int8×int8 MACs per vdotq_s32 instruction — no per-weight dequant. 2-row
 * fused to amortize the qx loads (matches the bf16/int8 2-row pattern). */
static void int8_matvec_sdot(float *y, const int8_t *qx, float sx,
                             const int8_t *W, const float *scale,
                             int in_dim, int out_dim) {
    int o = 0;
    for (; o + 1 < out_dim; o += 2) {
        const int8_t *w0 = W + (size_t)o * in_dim;
        const int8_t *w1 = W + (size_t)(o + 1) * in_dim;
        int32x4_t a0 = vdupq_n_s32(0), a1 = vdupq_n_s32(0);
        int k = 0;
        for (; k + 15 < in_dim; k += 16) {
            int8x16_t xv = vld1q_s8(qx + k);
            a0 = vdotq_s32(a0, vld1q_s8(w0 + k), xv);
            a1 = vdotq_s32(a1, vld1q_s8(w1 + k), xv);
        }
        int32_t s0 = vaddvq_s32(a0), s1 = vaddvq_s32(a1);
        for (; k < in_dim; k++) { s0 += (int32_t)w0[k] * qx[k]; s1 += (int32_t)w1[k] * qx[k]; }
        y[o]     = (float)s0 * scale[o]     * sx;
        y[o + 1] = (float)s1 * scale[o + 1] * sx;
    }
    if (o < out_dim) {
        const int8_t *w0 = W + (size_t)o * in_dim;
        int32x4_t a0 = vdupq_n_s32(0);
        int k = 0;
        for (; k + 15 < in_dim; k += 16)
            a0 = vdotq_s32(a0, vld1q_s8(w0 + k), vld1q_s8(qx + k));
        int32_t s0 = vaddvq_s32(a0);
        for (; k < in_dim; k++) s0 += (int32_t)w0[k] * qx[k];
        y[o] = (float)s0 * scale[o] * sx;
    }
}
#endif /* __ARM_FEATURE_DOTPROD */

#if defined(__AVX512VNNI__)
/* ── x86 native int8 dot via AVX-512 VNNI (the SDOT analog for x86) ──
 * UNVALIDATED ON HARDWARE — written for a rented AVX-512-VNNI VPS (e.g. Zen4/Zen5
 * 9950X3D, Cascade Lake+). Cross-compiles clean; validate with `make test-golden`
 * on the VPS before trusting. Opt out at runtime with QWEN_NO_VNNI=1. */

/* f32 activation -> signed int8 (per-vector absmax). Scalar inner (n is small,
 * e.g. hidden=1024); correctness-first — vectorize later if it shows up. */
static float quantize_act_int8_x86(int8_t *qx, const float *x, int n) {
    float amax = 0.0f;
    for (int i = 0; i < n; i++) { float a = fabsf(x[i]); if (a > amax) amax = a; }
    if (amax == 0.0f) { memset(qx, 0, (size_t)n); return 0.0f; }
    float inv = 127.0f / amax;
    for (int i = 0; i < n; i++) {
        int v = (int)lrintf(x[i] * inv);
        qx[i] = (int8_t)(v > 127 ? 127 : (v < -128 ? -128 : v));
    }
    return amax / 127.0f;
}

/* y[o] = scale[o]*sx * Σ_k W[o][k]·qx[k], via _mm512_dpbusd_epi32.
 * VNNI multiplies UNSIGNED u8 × SIGNED s8, but qx is signed. Use ua = qx+128
 * (unsigned) and correct: Σ w·qx = Σ w·ua − 128·Σ w. Both Σw·ua and Σw are
 * accumulated with VNNI in the same loop (the latter via dpbusd(ones_u8, w)).
 * 2-row fused; 64 int8/iter per 512-bit lane. */
static void int8_matvec_vnni(float *y, const int8_t *qx, float sx,
                             const int8_t *W, const float *scale,
                             int in_dim, int out_dim) {
    const __m512i v128 = _mm512_set1_epi8((char)128);
    const __m512i ones = _mm512_set1_epi8(1);
    int o = 0;
    for (; o + 1 < out_dim; o += 2) {
        const int8_t *w0 = W + (size_t)o * in_dim;
        const int8_t *w1 = W + (size_t)(o + 1) * in_dim;
        __m512i acc0 = _mm512_setzero_si512(), acc1 = _mm512_setzero_si512();
        __m512i ws0  = _mm512_setzero_si512(), ws1  = _mm512_setzero_si512();
        int k = 0;
        for (; k + 64 <= in_dim; k += 64) {
            __m512i ua  = _mm512_add_epi8(_mm512_loadu_si512((const void *)(qx + k)), v128);
            __m512i wv0 = _mm512_loadu_si512((const void *)(w0 + k));
            __m512i wv1 = _mm512_loadu_si512((const void *)(w1 + k));
            acc0 = _mm512_dpbusd_epi32(acc0, ua, wv0);
            acc1 = _mm512_dpbusd_epi32(acc1, ua, wv1);
            ws0  = _mm512_dpbusd_epi32(ws0, ones, wv0);
            ws1  = _mm512_dpbusd_epi32(ws1, ones, wv1);
        }
        int s0 = _mm512_reduce_add_epi32(acc0) - 128 * _mm512_reduce_add_epi32(ws0);
        int s1 = _mm512_reduce_add_epi32(acc1) - 128 * _mm512_reduce_add_epi32(ws1);
        for (; k < in_dim; k++) { s0 += (int)w0[k] * qx[k]; s1 += (int)w1[k] * qx[k]; }
        y[o]     = (float)s0 * scale[o]     * sx;
        y[o + 1] = (float)s1 * scale[o + 1] * sx;
    }
    if (o < out_dim) {
        const int8_t *w0 = W + (size_t)o * in_dim;
        __m512i acc0 = _mm512_setzero_si512(), ws0 = _mm512_setzero_si512();
        int k = 0;
        for (; k + 64 <= in_dim; k += 64) {
            __m512i ua  = _mm512_add_epi8(_mm512_loadu_si512((const void *)(qx + k)), v128);
            __m512i wv0 = _mm512_loadu_si512((const void *)(w0 + k));
            acc0 = _mm512_dpbusd_epi32(acc0, ua, wv0);
            ws0  = _mm512_dpbusd_epi32(ws0, ones, wv0);
        }
        int s0 = _mm512_reduce_add_epi32(acc0) - 128 * _mm512_reduce_add_epi32(ws0);
        for (; k < in_dim; k++) s0 += (int)w0[k] * qx[k];
        y[o] = (float)s0 * scale[o] * sx;
    }
}

typedef struct {
    float *y; const int8_t *qx; float sx; const int8_t *W; const float *scale; int rows, cols;
} int8_vnni_ctx;
static void int8_vnni_task(size_t tid, size_t nt, void *vc) {
    int8_vnni_ctx *c = (int8_vnni_ctx *)vc;
    int r0 = (int)(tid * (size_t)c->rows / nt);
    int r1 = (int)((tid + 1) * (size_t)c->rows / nt);
    int8_matvec_vnni(c->y + r0, c->qx, c->sx, c->W + (size_t)r0 * c->cols,
                     c->scale + r0, c->cols, r1 - r0);
}
#endif /* __AVX512VNNI__ */

typedef struct {
    float *y; const float *x; const int8_t *W; const float *scale; int rows, cols;
} int8_mv_ctx;
static void int8_mv_task(size_t tid, size_t nt, void *vc) {
    int8_mv_ctx *c = (int8_mv_ctx *)vc;
    int r0 = (int)(tid * (size_t)c->rows / nt);
    int r1 = (int)((tid + 1) * (size_t)c->rows / nt);
    int8_matvec_fused(c->y + r0, c->x, c->W + (size_t)r0 * c->cols,
                      c->scale + r0, c->cols, r1 - r0);
}
#if defined(__ARM_FEATURE_DOTPROD)
typedef struct {
    float *y; const int8_t *qx; float sx; const int8_t *W; const float *scale; int rows, cols;
} int8_sdot_ctx;
static void int8_sdot_task(size_t tid, size_t nt, void *vc) {
    int8_sdot_ctx *c = (int8_sdot_ctx *)vc;
    int r0 = (int)(tid * (size_t)c->rows / nt);
    int r1 = (int)((tid + 1) * (size_t)c->rows / nt);
    int8_matvec_sdot(c->y + r0, c->qx, c->sx, c->W + (size_t)r0 * c->cols,
                     c->scale + r0, c->cols, r1 - r0);
}
#endif

void qwen_matvec_int8(float *y, const int8_t *W, const float *scale,
                      const float *x, int rows, int cols) {
#if defined(__AVX512VNNI__)
    /* x86 native int8 dot (VNNI). Same shape as the ARM SDOT path: quantize the
     * shared activation once, then int8×int8 via dpbusd. QWEN_NO_VNNI=1 opts out. */
    enum { QXV_MAX = 8192 };
    static atomic_int vnni_off = -1;  /* audit #10: race-free one-time env cache (relaxed = plain load) */
    int vnni_o = atomic_load_explicit(&vnni_off, memory_order_relaxed);
    if (vnni_o < 0) { const char *e = getenv("QWEN_NO_VNNI"); vnni_o = (e && e[0] == '1'); atomic_store_explicit(&vnni_off, vnni_o, memory_order_relaxed); }
    if (!vnni_o && cols <= QXV_MAX) {
        int8_t qx_buf[QXV_MAX];
        float sx = quantize_act_int8_x86(qx_buf, x, cols);
        int nt = g_n_threads;
        if (nt > 1 && rows >= 256) {
            int8_vnni_ctx c = { y, qx_buf, sx, W, scale, rows, cols };
            qwen_parallel((size_t)nt, int8_vnni_task, &c);
            return;
        }
        int8_matvec_vnni(y, qx_buf, sx, W, scale, cols, rows);
        return;
    }
#endif
#if defined(__ARM_FEATURE_DOTPROD)
    /* SDOT path: quantize the shared activation x once, then int8×int8 dot.
     * qx is a fixed-size stack buffer; qwen_parallel is synchronous so the pool
     * workers safely read it for the call's duration. cols beyond the cap (rare;
     * only very large matrices) falls through to the f32 path. */
    enum { QX_MAX = 8192 };
    static atomic_int sdot_off = -1;  /* QWEN_NO_SDOT=1 forces the legacy f32 path (A/B bench); audit #10 */
    int sdot_o = atomic_load_explicit(&sdot_off, memory_order_relaxed);
    if (sdot_o < 0) { const char *e = getenv("QWEN_NO_SDOT"); sdot_o = (e && e[0] == '1'); atomic_store_explicit(&sdot_off, sdot_o, memory_order_relaxed); }
    if (!sdot_o && cols <= QX_MAX) {
        int8_t qx_buf[QX_MAX];
        float sx = quantize_act_int8(qx_buf, x, cols);
        int nt = g_n_threads;
        if (nt > 1 && rows >= 256) {
            int8_sdot_ctx c = { y, qx_buf, sx, W, scale, rows, cols };
            qwen_parallel((size_t)nt, int8_sdot_task, &c);
            return;
        }
        int8_matvec_sdot(y, qx_buf, sx, W, scale, cols, rows);
        return;
    }
#endif
    int nt = g_n_threads;
    if (nt > 1 && rows >= 256) {
        int8_mv_ctx c = { y, x, W, scale, rows, cols };
        qwen_parallel((size_t)nt, int8_mv_task, &c);
        return;
    }
    int8_matvec_fused(y, x, W, scale, cols, rows);
}

void qwen_matvec_int8_qkv(float *q, float *k, float *v,
                           const int8_t *Wq, const float *sq,
                           const int8_t *Wk, const float *sk,
                           const int8_t *Wv, const float *sv,
                           const float *x, int in_dim, int q_dim, int kv_dim) {
    /* The inline threaded qkv block hung at 4 threads on int8-induced denormals
     * even with FTZ set in-block; the per-projection fused matvec is robust.
     * qwen_matvec_int8 dispatches across threads and sets FTZ in each worker. */
    qwen_matvec_int8(q, Wq, sq, x, q_dim, in_dim);
    qwen_matvec_int8(k, Wk, sk, x, kv_dim, in_dim);
    qwen_matvec_int8(v, Wv, sv, x, kv_dim, in_dim);
}

int qwen_argmax_matvec_int8(const float *x, const int8_t *W, const float *scale,
                            int in_dim, int out_dim) {
    qwen_ftz_on();
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
#elif defined(__AVX2__)
        __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps(),
               a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
        int k = 0;
        for (; k + 32 <= in_dim; k += 32) {
            a0 = _mm256_fmadd_ps(qwen_loadu_s8_8(row + k),      _mm256_loadu_ps(x + k),      a0);
            a1 = _mm256_fmadd_ps(qwen_loadu_s8_8(row + k + 8),  _mm256_loadu_ps(x + k + 8),  a1);
            a2 = _mm256_fmadd_ps(qwen_loadu_s8_8(row + k + 16), _mm256_loadu_ps(x + k + 16), a2);
            a3 = _mm256_fmadd_ps(qwen_loadu_s8_8(row + k + 24), _mm256_loadu_ps(x + k + 24), a3);
        }
        for (; k + 8 <= in_dim; k += 8)
            a0 = _mm256_fmadd_ps(qwen_loadu_s8_8(row + k), _mm256_loadu_ps(x + k), a0);
        sum = qwen_hsum256_ps(_mm256_add_ps(_mm256_add_ps(a0, a2), _mm256_add_ps(a1, a3)));
        for (; k < in_dim; k++) sum += (float)row[k] * x[k];
#else
        for (int k = 0; k < in_dim; k++) sum += (float)row[k] * x[k];
#endif
        sum *= scale[o];
        if (sum > best_val) { best_val = sum; best = o; }
    }
    return best;
}

/* Argmax over a Q4_0 matvec (CP lm_head with --int4). Reuses the optimized
 * (multi-threaded, SIMD) q4_0 matvec into a small scratch buffer, then argmaxes.
 * The scratch is a per-thread grow-on-demand buffer (plan_v4 #8): the CP calls
 * this 15×/frame, so a fresh malloc/free each time is pure churn. __thread keeps
 * it race-free under concurrent server synthesis (mirrors sampling.c's buffers). */
int qwen_argmax_matvec_q4_0(const float *x, const q4_0_block_t *W, int in_dim, int out_dim) {
    static __thread float *y = NULL;
    static __thread int y_cap = 0;
    if (out_dim > y_cap) {
        float *ny = (float *)realloc(y, (size_t)out_dim * sizeof(float));
        if (!ny) return 0;
        y = ny; y_cap = out_dim;
    }
    qwen_matvec_q4_0(y, W, x, out_dim, in_dim);
    int best = 0;
    float best_val = y[0];
    for (int o = 1; o < out_dim; o++)
        if (y[o] > best_val) { best_val = y[o]; best = o; }
    return best;
}

/* ========================================================================
 * Q4_0 Quantization + Matvec
 * ======================================================================== */

void qwen_quantize_bf16_to_q4_0(const uint16_t *src_bf16, int rows, int cols,
                                 q4_0_block_t *dst) {
    int blocks_per_row = cols / Q4_0_BLOCK_SIZE;
    for (int r = 0; r < rows; r++) {
        const uint16_t *row = src_bf16 + (size_t)r * cols;
        q4_0_block_t *dst_row = dst + (size_t)r * blocks_per_row;

        for (int b = 0; b < blocks_per_row; b++) {
            const uint16_t *blk = row + b * Q4_0_BLOCK_SIZE;

            /* Convert bf16 block to f32 and find absmax */
            float vals[Q4_0_BLOCK_SIZE];
            float amax = 0.0f;
            for (int i = 0; i < Q4_0_BLOCK_SIZE; i++) {
                uint32_t bits = (uint32_t)blk[i] << 16;
                memcpy(&vals[i], &bits, sizeof(float));
                float a = fabsf(vals[i]);
                if (a > amax) amax = a;
            }

            float s = amax / 7.0f;  /* map to [-7, 7] → unsigned [0, 15] */
            dst_row[b].scale = s;
            float inv_s = (s > 0) ? 1.0f / s : 0.0f;

            /* Quantize: round to [-8, 7], store as unsigned [0, 15] */
            for (int i = 0; i < 16; i++) {
                int lo = (int)roundf(vals[2*i] * inv_s);
                int hi = (int)roundf(vals[2*i+1] * inv_s);
                lo = lo < -8 ? -8 : (lo > 7 ? 7 : lo);
                hi = hi < -8 ? -8 : (hi > 7 ? 7 : hi);
                dst_row[b].qs[i] = (uint8_t)((lo + 8) | ((hi + 8) << 4));
            }
        }
    }
}

/* Q4_0 matvec inner kernel: one row at a time */
static void q4_0_matvec_inner(float *y, const float *x, const q4_0_block_t *W,
                               int cols, int out_dim) {
    int blocks_per_row = cols / Q4_0_BLOCK_SIZE;
    for (int o = 0; o < out_dim; o++) {
        const q4_0_block_t *row = W + (size_t)o * blocks_per_row;
        float sum = 0.0f;
#ifdef __ARM_NEON
        for (int b = 0; b < blocks_per_row; b++) {
            float scale = row[b].scale;
            const uint8_t *qs = row[b].qs;
            const float *xb = x + b * Q4_0_BLOCK_SIZE;

            /* Load 16 bytes = 32 nibbles */
            uint8x16_t raw = vld1q_u8(qs);
            uint8x16_t lo_nibble = vandq_u8(raw, vdupq_n_u8(0x0F));
            uint8x16_t hi_nibble = vshrq_n_u8(raw, 4);

            /* Interleave: [lo0,hi0,lo1,hi1,...] to get 32 values in order */
            /* Convert to signed: subtract 8 */
            int16x8_t s0 = vreinterpretq_s16_u16(vsubl_u8(vget_low_u8(lo_nibble), vdup_n_u8(8)));
            int16x8_t s1 = vreinterpretq_s16_u16(vsubl_u8(vget_low_u8(hi_nibble), vdup_n_u8(8)));
            int16x8_t s2 = vreinterpretq_s16_u16(vsubl_u8(vget_high_u8(lo_nibble), vdup_n_u8(8)));
            int16x8_t s3 = vreinterpretq_s16_u16(vsubl_u8(vget_high_u8(hi_nibble), vdup_n_u8(8)));

            /* s0 has lo[0..7], s1 has hi[0..7] — need to zip them:
             * [lo0,hi0,lo1,hi1,lo2,hi2,lo3,hi3] and [lo4,hi4,...,lo7,hi7] */
            int16x8x2_t z0 = vzipq_s16(s0, s1);  /* z0.val[0]=[lo0,hi0,lo1,hi1,lo2,hi2,lo3,hi3] */
            int16x8x2_t z1 = vzipq_s16(s2, s3);  /* z1.val[0]=[lo8,hi8,lo9,hi9,...] */

            /* Convert to f32 and FMA with x — 8 groups of 4 */
            float32x4_t vscale = vdupq_n_f32(scale);
            float32x4_t acc0 = vdupq_n_f32(0), acc1 = vdupq_n_f32(0);
            float32x4_t acc2 = vdupq_n_f32(0), acc3 = vdupq_n_f32(0);

            float32x4_t f0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(z0.val[0])));
            float32x4_t f1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(z0.val[0])));
            float32x4_t f2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(z0.val[1])));
            float32x4_t f3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(z0.val[1])));
            acc0 = vfmaq_f32(acc0, vmulq_f32(f0, vscale), vld1q_f32(xb));
            acc1 = vfmaq_f32(acc1, vmulq_f32(f1, vscale), vld1q_f32(xb + 4));
            acc2 = vfmaq_f32(acc2, vmulq_f32(f2, vscale), vld1q_f32(xb + 8));
            acc3 = vfmaq_f32(acc3, vmulq_f32(f3, vscale), vld1q_f32(xb + 12));

            float32x4_t f4 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(z1.val[0])));
            float32x4_t f5 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(z1.val[0])));
            float32x4_t f6 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(z1.val[1])));
            float32x4_t f7 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(z1.val[1])));
            acc0 = vfmaq_f32(acc0, vmulq_f32(f4, vscale), vld1q_f32(xb + 16));
            acc1 = vfmaq_f32(acc1, vmulq_f32(f5, vscale), vld1q_f32(xb + 20));
            acc2 = vfmaq_f32(acc2, vmulq_f32(f6, vscale), vld1q_f32(xb + 24));
            acc3 = vfmaq_f32(acc3, vmulq_f32(f7, vscale), vld1q_f32(xb + 28));

            sum += vaddvq_f32(vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3)));
        }
#elif defined(__AVX2__)
        /* 4 independent accumulators across the whole row (one per quarter-block)
         * so the FMAs aren't serialized into one latency-bound chain. */
        __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps(),
               acc2 = _mm256_setzero_ps(), acc3 = _mm256_setzero_ps();
        for (int b = 0; b < blocks_per_row; b++) {
            float scale = row[b].scale;
            const uint8_t *qs = row[b].qs;
            const float *xb = x + b * Q4_0_BLOCK_SIZE;
            __m128i raw = _mm_loadu_si128((const __m128i *)qs);   /* 16 bytes = 32 nibbles */
            __m128i lo = _mm_and_si128(raw, _mm_set1_epi8(0x0F));
            __m128i hi = _mm_and_si128(_mm_srli_epi16(raw, 4), _mm_set1_epi8(0x0F));
            /* Interleave to value order [lo0,hi0,lo1,hi1,...] and bias by -8 */
            __m128i il0 = _mm_sub_epi8(_mm_unpacklo_epi8(lo, hi), _mm_set1_epi8(8));
            __m128i il1 = _mm_sub_epi8(_mm_unpackhi_epi8(lo, hi), _mm_set1_epi8(8));
            __m256 vs = _mm256_set1_ps(scale);
            __m256 f0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(il0));
            __m256 f1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(il0, 8)));
            __m256 f2 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(il1));
            __m256 f3 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(il1, 8)));
            acc0 = _mm256_fmadd_ps(_mm256_mul_ps(f0, vs), _mm256_loadu_ps(xb),      acc0);
            acc1 = _mm256_fmadd_ps(_mm256_mul_ps(f1, vs), _mm256_loadu_ps(xb + 8),  acc1);
            acc2 = _mm256_fmadd_ps(_mm256_mul_ps(f2, vs), _mm256_loadu_ps(xb + 16), acc2);
            acc3 = _mm256_fmadd_ps(_mm256_mul_ps(f3, vs), _mm256_loadu_ps(xb + 24), acc3);
        }
        sum += qwen_hsum256_ps(_mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3)));
#else
        for (int b = 0; b < blocks_per_row; b++) {
            float scale = row[b].scale;
            const uint8_t *qs = row[b].qs;
            const float *xb = x + b * Q4_0_BLOCK_SIZE;
            for (int i = 0; i < 16; i++) {
                int lo = (int)(qs[i] & 0x0F) - 8;
                int hi = (int)(qs[i] >> 4) - 8;
                sum += scale * (float)lo * xb[2*i];
                sum += scale * (float)hi * xb[2*i+1];
            }
        }
#endif
        y[o] = sum;
    }
}

#if defined(__ARM_FEATURE_DOTPROD)
/* SDOT-native q4_0 matvec (the "int4 viable on ARM" kernel, plan_v4 B1 / perf #3).
 * The legacy q4_0_matvec_inner dequants every nibble to f32 and FMAs against an
 * f32 activation — on M1 that nibble-unpack dominates (int4 loses to int8 despite
 * half the bytes). Here we do the llama.cpp q4_0×q8_0 trick instead: quantize the
 * SHARED activation x to int8 once (per-vector scale sx, done by the caller), then
 * per block unpack the 32 nibbles into int8 *value order* [w0,w1,...,w31] and dot
 * them against the int8 activation with vdotq_s32 (4 int8×int8 MACs/instr, no
 * per-weight f32 convert). q4_0 has a PER-BLOCK scale, so each block's int32 dot is
 * scaled and summed in f32 (unlike int8's single per-row scale). 2-row fused to
 * amortize the qx loads, mirroring int8_matvec_sdot. */
static void q4_0_matvec_sdot(float *y, const int8_t *qx, float sx,
                             const q4_0_block_t *W, int cols, int out_dim) {
    int nb = cols / Q4_0_BLOCK_SIZE;   /* blocks per row */
    const uint8x16_t mask = vdupq_n_u8(0x0F);
    const int8x16_t bias = vdupq_n_s8(8);
    int o = 0;
    /* Per-block accumulation stays in a float32x4 lane vector: the cross-lane
     * vaddvq_s32 + scalar FMA that used to sit in the inner loop serialized on a
     * high-latency reduce every 32 weights. Deferring it (cvt + vfmaq_n_f32, one
     * vaddvq_f32 per row) is algebraically the same sum of scale*lane. Idea from
     * PR #17 (TrinityTF); kept on our interleaved q4_0 layout. */
    for (; o + 1 < out_dim; o += 2) {
        const q4_0_block_t *r0 = W + (size_t)o * nb;
        const q4_0_block_t *r1 = W + (size_t)(o + 1) * nb;
        float32x4_t fa0 = vdupq_n_f32(0.0f), fa1 = vdupq_n_f32(0.0f);
        for (int b = 0; b < nb; b++) {
            const int8_t *xb = qx + b * Q4_0_BLOCK_SIZE;
            int8x16_t x0 = vld1q_s8(xb);
            int8x16_t x1 = vld1q_s8(xb + 16);
            /* row 0 */
            uint8x16_t raw0 = vld1q_u8(r0[b].qs);
            uint8x16x2_t z0 = vzipq_u8(vandq_u8(raw0, mask), vshrq_n_u8(raw0, 4));
            int8x16_t w0a = vsubq_s8(vreinterpretq_s8_u8(z0.val[0]), bias);
            int8x16_t w0b = vsubq_s8(vreinterpretq_s8_u8(z0.val[1]), bias);
            int32x4_t acc0 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), w0a, x0), w0b, x1);
            fa0 = vfmaq_n_f32(fa0, vcvtq_f32_s32(acc0), r0[b].scale);
            /* row 1 */
            uint8x16_t raw1 = vld1q_u8(r1[b].qs);
            uint8x16x2_t z1 = vzipq_u8(vandq_u8(raw1, mask), vshrq_n_u8(raw1, 4));
            int8x16_t w1a = vsubq_s8(vreinterpretq_s8_u8(z1.val[0]), bias);
            int8x16_t w1b = vsubq_s8(vreinterpretq_s8_u8(z1.val[1]), bias);
            int32x4_t acc1 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), w1a, x0), w1b, x1);
            fa1 = vfmaq_n_f32(fa1, vcvtq_f32_s32(acc1), r1[b].scale);
        }
        y[o]     = vaddvq_f32(fa0) * sx;
        y[o + 1] = vaddvq_f32(fa1) * sx;
    }
    if (o < out_dim) {
        const q4_0_block_t *r0 = W + (size_t)o * nb;
        float32x4_t fa0 = vdupq_n_f32(0.0f);
        for (int b = 0; b < nb; b++) {
            const int8_t *xb = qx + b * Q4_0_BLOCK_SIZE;
            int8x16_t x0 = vld1q_s8(xb);
            int8x16_t x1 = vld1q_s8(xb + 16);
            uint8x16_t raw0 = vld1q_u8(r0[b].qs);
            uint8x16x2_t z0 = vzipq_u8(vandq_u8(raw0, mask), vshrq_n_u8(raw0, 4));
            int8x16_t w0a = vsubq_s8(vreinterpretq_s8_u8(z0.val[0]), bias);
            int8x16_t w0b = vsubq_s8(vreinterpretq_s8_u8(z0.val[1]), bias);
            int32x4_t acc0 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), w0a, x0), w0b, x1);
            fa0 = vfmaq_n_f32(fa0, vcvtq_f32_s32(acc0), r0[b].scale);
        }
        y[o] = vaddvq_f32(fa0) * sx;
    }
}

typedef struct {
    float *y; const int8_t *qx; float sx; const q4_0_block_t *W; int rows, cols;
} q4_0_sdot_ctx;
static void q4_0_sdot_task(size_t tid, size_t nt, void *vc) {
    q4_0_sdot_ctx *c = (q4_0_sdot_ctx *)vc;
    int r0 = (int)(tid * (size_t)c->rows / nt);
    int r1 = (int)((tid + 1) * (size_t)c->rows / nt);
    q4_0_matvec_sdot(c->y + r0, c->qx, c->sx,
                     c->W + (size_t)r0 * (c->cols / Q4_0_BLOCK_SIZE), c->cols, r1 - r0);
}
#endif /* __ARM_FEATURE_DOTPROD */

typedef struct {
    float *y; const q4_0_block_t *W; const float *x; int rows, cols, blocks_per_row;
} q4_0_mv_ctx;
static void q4_0_mv_task(size_t tid, size_t nt, void *vc) {
    q4_0_mv_ctx *c = (q4_0_mv_ctx *)vc;
    int r0 = (int)(tid * (size_t)c->rows / nt);
    int r1 = (int)((tid + 1) * (size_t)c->rows / nt);
    q4_0_matvec_inner(c->y + r0, c->x, c->W + (size_t)r0 * c->blocks_per_row,
                      c->cols, r1 - r0);
}
#if defined(__ARM_FEATURE_DOTPROD) || defined(__AVX512VNNI__)
enum { Q4_QX_MAX = 8192 };
/* Shared one-time QWEN_NO_SDOT cache (audit #10 race-free). 1 = force the legacy
 * f32-dequant q4 path (the A/B bench + quant-ladder gate); shared by the ARM SDOT and
 * x86 VNNI q4 paths and the fused QKV so single-stream stays self-consistent. */
static int q4_sdot_disabled(void) {
    static atomic_int off = -1;
    int v = atomic_load_explicit(&off, memory_order_relaxed);
    if (v < 0) { const char *e = getenv("QWEN_NO_SDOT"); v = (e && e[0] == '1'); atomic_store_explicit(&off, v, memory_order_relaxed); }
    return v;
}
#endif

#if defined(__AVX512VNNI__)
/* x86 AVX-512-VNNI q4_0 matvec (plan_v4 C7) — the x86 twin of the ARM q4_0_matvec_sdot.
 * Quantize the shared activation to int8 once (caller), then per 32-weight block:
 * unpack the 16 nibble-bytes to 32 signed int8 in value order (nibble−8, the SAME
 * layout as the tested AVX2 q4 path — value order [lo0,hi0,lo1,hi1,…]), and dot against
 * the int8 activation with _mm512_dpbusd_epi32. VNNI is unsigned×signed, so (mirroring
 * the validated int8_matvec_vnni) make the activation unsigned via ua = qx+128 and
 * correct: Σ w·qx = Σ w·ua − 128·Σw (Σw via dpbusd(ones, w)). q4_0 has a PER-BLOCK
 * scale, so each block's int32 dot is scaled and summed in f32 (like the ARM twin).
 * The 32-wide block is zero-extended into the 512-bit dpbusd (upper half → 0); packing
 * 2 blocks per 512-bit op + 2-row fusion are the obvious throughput follow-ups.
 * ⚠️ COMPILE-CHECKED ONLY (`make check-isa`) — NOT yet validated on real AVX-512-VNNI
 * silicon (Zen4/SPR). See the plan_v4 C7 rental TODO before trusting it. */
static void q4_0_matvec_vnni(float *y, const int8_t *qx, float sx,
                             const q4_0_block_t *W, int cols, int out_dim) {
    int nb = cols / Q4_0_BLOCK_SIZE;
    const __m128i lomask = _mm_set1_epi8(0x0F);
    const __m512i ones   = _mm512_set1_epi8(1);
    /* Key opt (plan_v4 C7 v2): the q4 nibbles are ALREADY unsigned (0..15), so
     * dpbusd(nibble_u8, qx_s8) = Σ nibble·qx DIRECTLY — no `−8` on the weight, no `+128`
     * offset trick. Since w = scale·(nibble−8), Σ w·qx = scale·(Σ nibble·qx − 8·Σqx),
     * and −8·Σqx depends only on the (shared) activation → precompute it ONCE per block.
     * So the row loop is ONE dpbusd + ONE reduce per block (v1 did two of each + a
     * broadcast +128 add). qx is shared across all out_dim rows → the precompute amortizes. */
    int corr[Q4_QX_MAX / Q4_0_BLOCK_SIZE];
    for (int b = 0; b < nb; b++) {
        __m512i xv = _mm512_zextsi256_si512(_mm256_loadu_si256((const __m256i *)(qx + (size_t)b * Q4_0_BLOCK_SIZE)));
        corr[b] = -8 * _mm512_reduce_add_epi32(_mm512_dpbusd_epi32(_mm512_setzero_si512(), ones, xv));
    }
    for (int o = 0; o < out_dim; o++) {
        const q4_0_block_t *row = W + (size_t)o * nb;
        float sum = 0.0f;
        for (int b = 0; b < nb; b++) {
            __m128i raw = _mm_loadu_si128((const __m128i *)row[b].qs);
            __m128i lo = _mm_and_si128(raw, lomask);
            __m128i hi = _mm_and_si128(_mm_srli_epi16(raw, 4), lomask);
            /* value order [lo0,hi0,lo1,hi1,...] as UNSIGNED nibbles 0..15 (no −8 bias) */
            __m512i wv = _mm512_zextsi256_si512(_mm256_set_m128i(_mm_unpackhi_epi8(lo, hi),
                                                                 _mm_unpacklo_epi8(lo, hi)));
            __m512i xv = _mm512_zextsi256_si512(_mm256_loadu_si256((const __m256i *)(qx + (size_t)b * Q4_0_BLOCK_SIZE)));
            int dot = _mm512_reduce_add_epi32(_mm512_dpbusd_epi32(_mm512_setzero_si512(), wv, xv)) + corr[b];
            sum += row[b].scale * (float)dot;
        }
        y[o] = sum * sx;
    }
}
/* v3 (throughput-packing): the two follow-ups the v2 comment names.
 *
 * v2 wasted half the datapath (a 32-int8 block zero-extended into a 512-bit
 * dpbusd) and put a cross-lane _mm512_reduce_add_epi32 on the critical path
 * every block. On EPYC 9555P that made int4-VNNI ~37% SLOWER than int8
 * (project_x86_epyc_vnni_validation). v3:
 *   - packs 2 blocks per 512-bit dpbusd (64 int8 = full width): the low 256-bit
 *     half is block b, the high half is block b+1, and the activation load is a
 *     single _mm512_loadu of qx[b*32 .. b*32+63];
 *   - unrolls 4 output rows with independent dpbusd accumulator chains, so the
 *     per-block reduces from different rows overlap and hide dpbusd's ~4-5c
 *     latency instead of serializing.
 * Per-block q4 scale still forces a scalar dot per block (like the ARM SDOT
 * twin), but the two block-dots come out of ONE dpbusd as the two 256-bit-half
 * sums, and the FMA into the float accumulator carries the scale.
 *
 * QWEN_Q4_VNNI_V3=0 falls back to v2, so the box can A/B without a rebuild.
 * ⚠️ COMPILE-CHECKED ONLY here (cross-compile + Rosetta numeric spot-check);
 * the SPEED claim is a hypothesis until measured on real Zen4/SPR silicon. */
static inline int q4_hsum256(__m256i v) {           /* Σ of 8 int32 lanes */
    __m128i lo = _mm256_castsi256_si128(v);
    __m128i hi = _mm256_extracti128_si256(v, 1);
    __m128i s  = _mm_add_epi32(lo, hi);
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(1, 0, 3, 2)));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(2, 3, 0, 1)));
    return _mm_cvtsi128_si32(s);
}
/* Unpack one q4_0 block (16 nibble bytes) to 32 unsigned int8 in value order,
 * placed in the given 256-bit half. Nibbles stay 0..15; the −8 bias is folded
 * into `corr` by the caller, exactly as in v2. */
static inline __m256i q4_unpack_block_u8(const uint8_t *qs) {
    const __m128i lomask = _mm_set1_epi8(0x0F);
    __m128i raw = _mm_loadu_si128((const __m128i *)qs);
    __m128i lo = _mm_and_si128(raw, lomask);
    __m128i hi = _mm_and_si128(_mm_srli_epi16(raw, 4), lomask);
    return _mm256_set_m128i(_mm_unpackhi_epi8(lo, hi), _mm_unpacklo_epi8(lo, hi));
}
static void q4_0_matvec_vnni_v3(float *y, const int8_t *qx, float sx,
                                const q4_0_block_t *W, int cols, int out_dim) {
    int nb = cols / Q4_0_BLOCK_SIZE;
    const __m512i ones = _mm512_set1_epi8(1);
    /* Per-block −8·Σqx correction, shared across all rows (like v2). */
    int corr[Q4_QX_MAX / Q4_0_BLOCK_SIZE];
    for (int b = 0; b < nb; b++) {
        __m512i xv = _mm512_zextsi256_si512(
            _mm256_loadu_si256((const __m256i *)(qx + (size_t)b * Q4_0_BLOCK_SIZE)));
        corr[b] = -8 * _mm512_reduce_add_epi32(_mm512_dpbusd_epi32(_mm512_setzero_si512(), ones, xv));
    }

    int o = 0;
    for (; o + 3 < out_dim; o += 4) {          /* 4 independent rows */
        const q4_0_block_t *r0 = W + (size_t)o * nb, *r1 = r0 + nb, *r2 = r1 + nb, *r3 = r2 + nb;
        float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
        int b = 0;
        for (; b + 1 < nb; b += 2) {           /* 2 blocks / 512-bit op */
            __m512i xv = _mm512_loadu_si512((const void *)(qx + (size_t)b * Q4_0_BLOCK_SIZE));
            __m512i w0 = _mm512_inserti64x4(
                _mm512_castsi256_si512(q4_unpack_block_u8(r0[b].qs)), q4_unpack_block_u8(r0[b + 1].qs), 1);
            __m512i w1 = _mm512_inserti64x4(
                _mm512_castsi256_si512(q4_unpack_block_u8(r1[b].qs)), q4_unpack_block_u8(r1[b + 1].qs), 1);
            __m512i w2 = _mm512_inserti64x4(
                _mm512_castsi256_si512(q4_unpack_block_u8(r2[b].qs)), q4_unpack_block_u8(r2[b + 1].qs), 1);
            __m512i w3 = _mm512_inserti64x4(
                _mm512_castsi256_si512(q4_unpack_block_u8(r3[b].qs)), q4_unpack_block_u8(r3[b + 1].qs), 1);
            __m512i d0 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), w0, xv);
            __m512i d1 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), w1, xv);
            __m512i d2 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), w2, xv);
            __m512i d3 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), w3, xv);
            /* low 256 = block b, high 256 = block b+1; reduce each half. */
            s0 += r0[b].scale * (q4_hsum256(_mm512_castsi512_si256(d0)) + corr[b])
                + r0[b + 1].scale * (q4_hsum256(_mm512_extracti64x4_epi64(d0, 1)) + corr[b + 1]);
            s1 += r1[b].scale * (q4_hsum256(_mm512_castsi512_si256(d1)) + corr[b])
                + r1[b + 1].scale * (q4_hsum256(_mm512_extracti64x4_epi64(d1, 1)) + corr[b + 1]);
            s2 += r2[b].scale * (q4_hsum256(_mm512_castsi512_si256(d2)) + corr[b])
                + r2[b + 1].scale * (q4_hsum256(_mm512_extracti64x4_epi64(d2, 1)) + corr[b + 1]);
            s3 += r3[b].scale * (q4_hsum256(_mm512_castsi512_si256(d3)) + corr[b])
                + r3[b + 1].scale * (q4_hsum256(_mm512_extracti64x4_epi64(d3, 1)) + corr[b + 1]);
        }
        for (; b < nb; b++) {                  /* odd tail block */
            __m512i xv = _mm512_zextsi256_si512(
                _mm256_loadu_si256((const __m256i *)(qx + (size_t)b * Q4_0_BLOCK_SIZE)));
            __m512i xw0 = _mm512_zextsi256_si512(q4_unpack_block_u8(r0[b].qs));
            __m512i xw1 = _mm512_zextsi256_si512(q4_unpack_block_u8(r1[b].qs));
            __m512i xw2 = _mm512_zextsi256_si512(q4_unpack_block_u8(r2[b].qs));
            __m512i xw3 = _mm512_zextsi256_si512(q4_unpack_block_u8(r3[b].qs));
            s0 += r0[b].scale * (_mm512_reduce_add_epi32(_mm512_dpbusd_epi32(_mm512_setzero_si512(), xw0, xv)) + corr[b]);
            s1 += r1[b].scale * (_mm512_reduce_add_epi32(_mm512_dpbusd_epi32(_mm512_setzero_si512(), xw1, xv)) + corr[b]);
            s2 += r2[b].scale * (_mm512_reduce_add_epi32(_mm512_dpbusd_epi32(_mm512_setzero_si512(), xw2, xv)) + corr[b]);
            s3 += r3[b].scale * (_mm512_reduce_add_epi32(_mm512_dpbusd_epi32(_mm512_setzero_si512(), xw3, xv)) + corr[b]);
        }
        y[o] = s0 * sx; y[o + 1] = s1 * sx; y[o + 2] = s2 * sx; y[o + 3] = s3 * sx;
    }
    /* remaining rows via v2 (correct, just not 4-unrolled) */
    if (o < out_dim)
        q4_0_matvec_vnni(y + o, qx, sx, W + (size_t)o * nb, cols, out_dim - o);
}

static int q4_vnni_v3_on(void) {
    static atomic_int v = -1;
    int r = atomic_load_explicit(&v, memory_order_relaxed);
    if (r < 0) { const char *e = getenv("QWEN_Q4_VNNI_V3"); r = !(e && e[0] == '0'); /* default ON */
                 atomic_store_explicit(&v, r, memory_order_relaxed); }
    return r;
}

typedef struct { float *y; const int8_t *qx; float sx; const q4_0_block_t *W; int rows, cols; } q4_0_vnni_ctx;
static void q4_0_vnni_task(size_t tid, size_t nt, void *vc) {
    q4_0_vnni_ctx *c = (q4_0_vnni_ctx *)vc;
    int r0 = (int)(tid * (size_t)c->rows / nt);
    int r1 = (int)((tid + 1) * (size_t)c->rows / nt);
    const q4_0_block_t *W = c->W + (size_t)r0 * (c->cols / Q4_0_BLOCK_SIZE);
    if (q4_vnni_v3_on())
        q4_0_matvec_vnni_v3(c->y + r0, c->qx, c->sx, W, c->cols, r1 - r0);
    else
        q4_0_matvec_vnni(c->y + r0, c->qx, c->sx, W, c->cols, r1 - r0);
}
#endif

void qwen_matvec_q4_0(float *y, const q4_0_block_t *W, const float *x,
                       int rows, int cols) {
#if defined(__AVX512VNNI__)
    /* VNNI-native q4 path (plan_v4 C7), the x86 twin of the ARM SDOT-q4: quantize the
     * shared activation to int8 once, then unpack nibbles→int8 + dpbusd per block. */
    if (!q4_sdot_disabled() && cols <= Q4_QX_MAX && cols % Q4_0_BLOCK_SIZE == 0) {
        int8_t qx_buf[Q4_QX_MAX];
        float sx = quantize_act_int8_x86(qx_buf, x, cols);
        int nt = g_n_threads;
        if (nt > 1 && rows >= 256) {
            q4_0_vnni_ctx c = { y, qx_buf, sx, W, rows, cols };
            qwen_parallel((size_t)nt, q4_0_vnni_task, &c);
            return;
        }
        if (q4_vnni_v3_on()) q4_0_matvec_vnni_v3(y, qx_buf, sx, W, cols, rows);
        else                 q4_0_matvec_vnni(y, qx_buf, sx, W, cols, rows);
        return;
    }
#endif
#if defined(__ARM_FEATURE_DOTPROD)
    /* SDOT-native path (plan_v4 B1): quantize the shared activation to int8 once,
     * then int8×int8 dot per nibble-block. cols beyond the cap (rare; only very
     * large matrices) falls through to the f32 path. */
    if (!q4_sdot_disabled() && cols <= Q4_QX_MAX && cols % Q4_0_BLOCK_SIZE == 0) {
        int8_t qx_buf[Q4_QX_MAX];
        float sx = quantize_act_int8(qx_buf, x, cols);
        int nt = g_n_threads;
        if (nt > 1 && rows >= 256) {
            q4_0_sdot_ctx c = { y, qx_buf, sx, W, rows, cols };
            qwen_parallel((size_t)nt, q4_0_sdot_task, &c);
            return;
        }
        q4_0_matvec_sdot(y, qx_buf, sx, W, cols, rows);
        return;
    }
#endif
    int nt = g_n_threads;
    if (nt > 1 && rows >= 256) {
        q4_0_mv_ctx c = { y, W, x, rows, cols, cols / Q4_0_BLOCK_SIZE };
        qwen_parallel((size_t)nt, q4_0_mv_task, &c);
        return;
    }
    q4_0_matvec_inner(y, x, W, cols, rows);
}

/* QKV q4_0: partition the concatenated [Q|K|V] row space, reusing the inner
 * kernel on each contiguous sub-segment (same result as the old inlined block,
 * and now picks up any AVX2/NEON improvement to q4_0_matvec_inner for free). */
typedef struct {
    float *q, *k, *v;
    const q4_0_block_t *Wq, *Wk, *Wv;
    const float *x;
    int in_dim, q_dim, kv_dim, blocks_per_row;
} q4_0_qkv_ctx;
static void q4_0_qkv_task(size_t tid, size_t nt, void *vc) {
    q4_0_qkv_ctx *c = (q4_0_qkv_ctx *)vc;
    int total = c->q_dim + 2 * c->kv_dim;
    int r0 = (int)(tid * (size_t)total / nt);
    int r1 = (int)((tid + 1) * (size_t)total / nt);
    for (int r = r0; r < r1; ) {
        if (r < c->q_dim) {
            int end = r1 < c->q_dim ? r1 : c->q_dim;
            q4_0_matvec_inner(c->q + r, c->x, c->Wq + (size_t)r * c->blocks_per_row,
                              c->in_dim, end - r);
            r = end;
        } else if (r < c->q_dim + c->kv_dim) {
            int local = r - c->q_dim;
            int end = r1 < c->q_dim + c->kv_dim ? r1 : c->q_dim + c->kv_dim;
            int local_end = end - c->q_dim;
            q4_0_matvec_inner(c->k + local, c->x, c->Wk + (size_t)local * c->blocks_per_row,
                              c->in_dim, local_end - local);
            r = end;
        } else {
            int local = r - c->q_dim - c->kv_dim;
            int local_end = r1 - c->q_dim - c->kv_dim;
            q4_0_matvec_inner(c->v + local, c->x, c->Wv + (size_t)local * c->blocks_per_row,
                              c->in_dim, local_end - local);
            r = r1;
        }
    }
}
#if defined(__ARM_FEATURE_DOTPROD)
/* SDOT fused-QKV (plan_v4 B1 + #7): quantize the shared activation to int8 ONCE,
 * then SDOT for Q/K/V. Keeps single-stream int4 QKV consistent with the standalone
 * q4 matvec (and with the batched path). Partitions the [Q|K|V] rows like the
 * f32 twin so it picks up the same threading. */
typedef struct {
    float *q, *k, *v;
    const q4_0_block_t *Wq, *Wk, *Wv;
    const int8_t *qx; float sx;
    int in_dim, q_dim, kv_dim;
} q4_0_qkv_sdot_ctx;
static void q4_0_qkv_sdot_task(size_t tid, size_t nt, void *vc) {
    q4_0_qkv_sdot_ctx *c = (q4_0_qkv_sdot_ctx *)vc;
    int total = c->q_dim + 2 * c->kv_dim;
    int nb = c->in_dim / Q4_0_BLOCK_SIZE;
    int r0 = (int)(tid * (size_t)total / nt);
    int r1 = (int)((tid + 1) * (size_t)total / nt);
    for (int r = r0; r < r1; ) {
        if (r < c->q_dim) {
            int end = r1 < c->q_dim ? r1 : c->q_dim;
            q4_0_matvec_sdot(c->q + r, c->qx, c->sx, c->Wq + (size_t)r * nb, c->in_dim, end - r);
            r = end;
        } else if (r < c->q_dim + c->kv_dim) {
            int local = r - c->q_dim;
            int end = r1 < c->q_dim + c->kv_dim ? r1 : c->q_dim + c->kv_dim;
            q4_0_matvec_sdot(c->k + local, c->qx, c->sx, c->Wk + (size_t)local * nb, c->in_dim, (end - c->q_dim) - local);
            r = end;
        } else {
            int local = r - c->q_dim - c->kv_dim;
            int local_end = r1 - c->q_dim - c->kv_dim;
            q4_0_matvec_sdot(c->v + local, c->qx, c->sx, c->Wv + (size_t)local * nb, c->in_dim, local_end - local);
            r = r1;
        }
    }
}
#endif
void qwen_matvec_q4_0_qkv(float *q, float *k, float *v,
                            const q4_0_block_t *Wq, const q4_0_block_t *Wk,
                            const q4_0_block_t *Wv,
                            const float *x, int in_dim, int q_dim, int kv_dim) {
#if defined(__ARM_FEATURE_DOTPROD)
    if (!q4_sdot_disabled() && in_dim <= Q4_QX_MAX && in_dim % Q4_0_BLOCK_SIZE == 0) {
        int8_t qx_buf[Q4_QX_MAX];
        float sx = quantize_act_int8(qx_buf, x, in_dim);
        int nt = g_n_threads;
        if (nt > 1) {
            q4_0_qkv_sdot_ctx c = { q, k, v, Wq, Wk, Wv, qx_buf, sx, in_dim, q_dim, kv_dim };
            qwen_parallel((size_t)nt, q4_0_qkv_sdot_task, &c);
            return;
        }
        q4_0_matvec_sdot(q, qx_buf, sx, Wq, in_dim, q_dim);
        q4_0_matvec_sdot(k, qx_buf, sx, Wk, in_dim, kv_dim);
        q4_0_matvec_sdot(v, qx_buf, sx, Wv, in_dim, kv_dim);
        return;
    }
#endif
    int nt = g_n_threads;
    if (nt > 1) {
        q4_0_qkv_ctx c = { q, k, v, Wq, Wk, Wv, x, in_dim, q_dim, kv_dim,
                           in_dim / Q4_0_BLOCK_SIZE };
        qwen_parallel((size_t)nt, q4_0_qkv_task, &c);
        return;
    }
    q4_0_matvec_inner(q, x, Wq, in_dim, q_dim);
    q4_0_matvec_inner(k, x, Wk, in_dim, kv_dim);
    q4_0_matvec_inner(v, x, Wv, in_dim, kv_dim);
}

/* ========================================================================
 * Q2_0 (2-bit) — EXPERIMENTAL hybrid lever for the quant-tolerant FFN matrices.
 * Scalar only for now (quality-first; SIMD added if it sounds OK). 4 symmetric
 * levels {-1.5,-0.5,0.5,1.5}×scale, scale = absmax/1.5.
 * ======================================================================== */
void qwen_quantize_bf16_to_q2_0(const uint16_t *src_bf16, int rows, int cols,
                                 q2_0_block_t *dst) {
    int bpr = cols / Q2_0_BLOCK_SIZE;
    for (int r = 0; r < rows; r++) {
        const uint16_t *row = src_bf16 + (size_t)r * cols;
        q2_0_block_t *drow = dst + (size_t)r * bpr;
        for (int b = 0; b < bpr; b++) {
            const uint16_t *blk = row + b * Q2_0_BLOCK_SIZE;
            float vals[Q2_0_BLOCK_SIZE], amax = 0.0f;
            for (int i = 0; i < Q2_0_BLOCK_SIZE; i++) {
                vals[i] = bf16_to_f32(blk[i]);
                float a = fabsf(vals[i]); if (a > amax) amax = a;
            }
            float scale = amax / 1.5f;
            drow[b].scale = scale;
            float inv = (scale > 0.0f) ? 1.0f / scale : 0.0f;
            for (int i = 0; i < 8; i++) drow[b].qs[i] = 0;
            for (int i = 0; i < Q2_0_BLOCK_SIZE; i++) {
                int code = (int)lrintf(vals[i] * inv + 1.5f);  /* {-1.5..1.5}/scale -> {0..3} */
                code = code < 0 ? 0 : (code > 3 ? 3 : code);
                drow[b].qs[i >> 2] |= (uint8_t)(code << ((i & 3) * 2));
            }
        }
    }
}

void qwen_matvec_q2_0(float *y, const q2_0_block_t *W, const float *x,
                      int rows, int cols) {
    int bpr = cols / Q2_0_BLOCK_SIZE;
    for (int o = 0; o < rows; o++) {
        const q2_0_block_t *wr = W + (size_t)o * bpr;
        float sum = 0.0f;
        for (int b = 0; b < bpr; b++) {
            float scale = wr[b].scale;
            const uint8_t *qs = wr[b].qs;
            const float *xb = x + b * Q2_0_BLOCK_SIZE;
            for (int i = 0; i < Q2_0_BLOCK_SIZE; i++) {
                int code = (qs[i >> 2] >> ((i & 3) * 2)) & 0x3;
                sum += ((float)code - 1.5f) * scale * xb[i];
            }
        }
        y[o] = sum;
    }
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
#elif defined(__AVX2__)
                score = qwen_dot_f32_avx2(q_row, k_row, head_dim);
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
#elif defined(__AVX2__)
                    qwen_acc_corr_avx2(o_row, v_row, correction, head_dim);
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
#elif defined(__AVX2__)
                    qwen_acc_wt_avx2(o_row, v_row, wt, head_dim);
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
#elif defined(__AVX2__)
                qwen_scale_avx2(o_row, inv_sum, head_dim);
#else
                for (int d = 0; d < head_dim; d++)
                    o_row[d] *= inv_sum;
#endif
            }
        }
    }
}

/* Causal GQA attention with sliding window support.
 * window <= 0 means no window (full causal). */
void qwen_causal_attention_windowed(float *out, const float *Q, const float *K, const float *V,
                                     int seq_q, int seq_k, int n_heads, int n_kv_heads,
                                     int head_dim, float scale, int q_offset, int window) {
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
            int k_start = 0;
            if (window > 0 && k_end - window > 0) k_start = k_end - window;

            float max_score = -1e30f;
            float sum_exp = 0.0f;
            memset(o_row, 0, head_dim * sizeof(float));

            for (int j = k_start; j < k_end; j++) {
                const float *k_row = K + j * kv_hidden + kv_h * head_dim;
                const float *v_row = V + j * kv_hidden + kv_h * head_dim;

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
#elif defined(__AVX2__)
                score = qwen_dot_f32_avx2(q_row, k_row, head_dim);
#else
                score = 0.0f;
                for (int d = 0; d < head_dim; d++)
                    score += q_row[d] * k_row[d];
#endif
                score *= scale;

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
#elif defined(__AVX2__)
                    qwen_acc_corr_avx2(o_row, v_row, correction, head_dim);
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
#elif defined(__AVX2__)
                    qwen_acc_wt_avx2(o_row, v_row, wt, head_dim);
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
#elif defined(__AVX2__)
                qwen_scale_avx2(o_row, inv_sum, head_dim);
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
#elif defined(__AVX2__)
                score = qwen_dot_f32_bf16_avx2(q_row, k_row_bf16, head_dim);
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
#elif defined(__AVX2__)
                    qwen_acc_corr_bf16_avx2(o_row, v_row_bf16, correction, head_dim);
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
#elif defined(__AVX2__)
                    qwen_acc_wt_bf16_avx2(o_row, v_row_bf16, wt, head_dim);
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
#elif defined(__AVX2__)
                qwen_scale_avx2(o_row, inv_sum, head_dim);
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

void qwen_swiglu_inplace(float *gate_up, float *tmp, int n) {
    /* 1. Extract gate values and negate: tmp[i] = -gate_up[2*i] */
    for (int i = 0; i < n; i++)
        tmp[i] = -gate_up[2 * i];

    /* 2. Batch exp: tmp[i] = exp(-g[i])
     * On macOS, vvexpf computes vectorized exp via Accelerate/vForce.
     * On other platforms, scalar loop (compiler auto-vectorizes with -ffast-math). */
#if defined(__APPLE__) && defined(USE_BLAS)
    vvexpf(tmp, tmp, &n);
#else
    for (int i = 0; i < n; i++)
        tmp[i] = expf(tmp[i]);
#endif

    /* 3. Apply sigmoid(g) * up = g / (1 + exp(-g)) * up */
    for (int i = 0; i < n; i++) {
        float g = gate_up[2 * i];
        float u = gate_up[2 * i + 1];
        gate_up[i] = g / (1.0f + tmp[i]) * u;
    }
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

void qwen_bf16_accum_f32(float *dst, const uint16_t *src_bf16, int n) {
    int i = 0;
#ifdef __ARM_NEON
    for (; i + 7 < n; i += 8) {
        uint16x8_t bf = vld1q_u16(src_bf16 + i);
        float32x4_t f0 = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(bf), 16));
        float32x4_t f1 = vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(bf), 16));
        vst1q_f32(dst + i,     vaddq_f32(vld1q_f32(dst + i), f0));
        vst1q_f32(dst + i + 4, vaddq_f32(vld1q_f32(dst + i + 4), f1));
    }
#elif defined(__AVX2__)
    for (; i + 7 < n; i += 8) {
        /* Load 8 bf16 values, zero-extend to 32-bit, shift left 16 to get f32 */
        __m128i bf = _mm_loadu_si128((const __m128i *)(src_bf16 + i));
        __m256i wide = _mm256_cvtepu16_epi32(bf);
        __m256 f = _mm256_castsi256_ps(_mm256_slli_epi32(wide, 16));
        __m256 d = _mm256_loadu_ps(dst + i);
        _mm256_storeu_ps(dst + i, _mm256_add_ps(d, f));
    }
#endif
    for (; i < n; i++) {
        uint32_t bits = (uint32_t)src_bf16[i] << 16;
        float val; memcpy(&val, &bits, sizeof(float));
        dst[i] += val;
    }
}

/* Convert bf16 vector to f32 (no accumulation — pure conversion).
 * NEON/AVX2 vectorized. */
void qwen_bf16_to_f32_vec(float *dst, const uint16_t *src_bf16, int n) {
    int i = 0;
#ifdef __ARM_NEON
    for (; i + 7 < n; i += 8) {
        uint16x8_t bf = vld1q_u16(src_bf16 + i);
        vst1q_f32(dst + i,     vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(bf), 16)));
        vst1q_f32(dst + i + 4, vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(bf), 16)));
    }
#elif defined(__AVX2__)
    for (; i + 7 < n; i += 8) {
        __m128i bf = _mm_loadu_si128((const __m128i *)(src_bf16 + i));
        __m256i wide = _mm256_cvtepu16_epi32(bf);
        _mm256_storeu_ps(dst + i, _mm256_castsi256_ps(_mm256_slli_epi32(wide, 16)));
    }
#endif
    for (; i < n; i++) {
        uint32_t bits = (uint32_t)src_bf16[i] << 16;
        memcpy(&dst[i], &bits, sizeof(float));
    }
}

/* ========================================================================
 * Snake activation: x += (1/exp(beta)) * sin²(exp(alpha) * x)
 * ======================================================================== */


/* ========================================================================
 * INT8 SDOT conv engine for the speech decoder (PR #17 sub-change B, ported)
 *
 * Opt-in: QWEN_SD_INT8=1, and only on ARM dotprod. Quantizes BOTH operands
 * with per-64-element block scales (Q8_0-style) -- a single per-column scale
 * measures only ~17 dB SNR because large channels crush small ones.
 * Measured (Neoverse-N1, 0.6B --int4 -j4): stream RTF 1.41 -> 1.15 (-18%),
 * decoder 7735 -> 5112 ms. Added noise floor ~-65 dBFS RMS, ear-validated
 * on preset / breathy-instruct / voice-clone before landing.
 * NEVER default-on: it trades audio quality for speed.
 * ======================================================================== */
int qwen_sd_int8_available(void) {
#if defined(__ARM_FEATURE_DOTPROD)
    return 1;
#else
    return 0;
#endif
}

int qwen_int8_kp(int K, int blk) { return (K + blk - 1) / blk * blk; }

/* Per-row, per-BLK-block absmax int8 quantization. dst rows are Kp-strided
 * and zero-padded; scales is [rows][Kp/blk]. blk must be a multiple of 16. */
void qwen_int8_quant_rows(int8_t *dst, float *scales, const float *src,
                          int rows, int K, int Kp, int blk) {
    int nblk = Kp / blk;
    for (int r = 0; r < rows; r++) {
        const float *s = src + (int64_t)r * K;
        int8_t *d = dst + (int64_t)r * Kp;
        float *sc = scales + (int64_t)r * nblk;
        for (int b = 0; b < nblk; b++) {
            int k0 = b * blk;
            int kn = K - k0 < blk ? K - k0 : blk;  /* valid elems in block */
            if (kn <= 0) { sc[b] = 1.0f; memset(d + k0, 0, blk); continue; }
            float amax = 0.0f;
            int i = 0;
#ifdef __ARM_NEON
            float32x4_t vmax = vdupq_n_f32(0.0f);
            for (; i + 3 < kn; i += 4)
                vmax = vmaxq_f32(vmax, vabsq_f32(vld1q_f32(s + k0 + i)));
            amax = vmaxvq_f32(vmax);
#endif
            for (; i < kn; i++) { float a = fabsf(s[k0 + i]); if (a > amax) amax = a; }
            float scale = amax > 0.0f ? amax / 127.0f : 1.0f;
            float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
            sc[b] = scale;
            i = 0;
#ifdef __ARM_NEON
            float32x4_t vinv = vdupq_n_f32(inv);
            for (; i + 15 < kn; i += 16) {
                int32x4_t q0 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(s + k0 + i),      vinv));
                int32x4_t q1 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(s + k0 + i + 4),  vinv));
                int32x4_t q2 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(s + k0 + i + 8),  vinv));
                int32x4_t q3 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(s + k0 + i + 12), vinv));
                int16x8_t p0 = vcombine_s16(vqmovn_s32(q0), vqmovn_s32(q1));
                int16x8_t p1 = vcombine_s16(vqmovn_s32(q2), vqmovn_s32(q3));
                vst1q_s8(d + k0 + i, vcombine_s8(vqmovn_s16(p0), vqmovn_s16(p1)));
            }
#endif
            for (; i < kn; i++) {
                float q = s[k0 + i] * inv;
                int v = (int)(q >= 0 ? q + 0.5f : q - 0.5f);
                if (v > 127) v = 127;
                if (v < -127) v = -127;
                d[k0 + i] = (int8_t)v;
            }
            for (; i < blk; i++) d[k0 + i] = 0;
        }
    }
}

/* ---------------- decoder-side parallel dispatcher ----------------------
 * Used by the int8 conv AND by the snake activation -- hence it must live
 * outside the dotprod guard below.
 * The decoder runs on its own thread while the generation thread is inside
 * qwen_parallel. On macOS qwen_parallel is GCD (reentrant) so we just use it.
 * On the POSIX pool it is NOT reentrant and, since d2b5df2, a second submitter
 * blocks on submit_mtx -- calling it from the decoder would stall generation.
 * So there we run a small pool of our own (PR #17's design), one job at a time.
 * Workers park on a condvar for process life; bounded, not a leak.
 * QWEN_SD_THREADS overrides the total thread count. */
#define SD_POOL_MAX_WORKERS 8

static pthread_mutex_t sdp_mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t  sdp_cv = PTHREAD_COND_INITIALIZER;
static pthread_cond_t  sdp_done_cv = PTHREAD_COND_INITIALIZER;
static pthread_t sdp_threads[SD_POOL_MAX_WORKERS];
static int sdp_nworkers = 0;
static int sdp_started = 0;
static unsigned sdp_gen = 0;
static int sdp_pending = 0;
static void (*sdp_fn)(void *) = NULL;
static void *sdp_ctx = NULL;

static void *sdp_worker_main(void *arg) {
    (void)arg;
    qwen_ftz_on();
    unsigned seen = 0;
    for (;;) {
        pthread_mutex_lock(&sdp_mu);
        while (sdp_gen == seen)
            pthread_cond_wait(&sdp_cv, &sdp_mu);
        seen = sdp_gen;
        void (*fn)(void *) = sdp_fn;
        void *ctx = sdp_ctx;
        pthread_mutex_unlock(&sdp_mu);
        fn(ctx);
        pthread_mutex_lock(&sdp_mu);
        if (--sdp_pending == 0) pthread_cond_signal(&sdp_done_cv);
        pthread_mutex_unlock(&sdp_mu);
    }
    return NULL;
}

static int sd_pool_threads(void) {
    static int cfg = -1;
    if (cfg < 0) { const char *e = getenv("QWEN_SD_THREADS"); cfg = e ? atoi(e) : 0; }
    return cfg > 0 ? cfg : qwen_get_threads();
}

/* Adapter: the job fn claims work from an atomic counter inside ctx, so every
 * thread runs the same fn and tid/nt are irrelevant. The pair travels in the
 * ctx so the GCD path touches no global (it can run on any thread). */
typedef struct { void (*fn)(void *); void *ctx; } sd_gcd_job_t;
static void sd_gcd_task(size_t tid, size_t nt, void *vj) {
    (void)tid; (void)nt;
    sd_gcd_job_t *j = (sd_gcd_job_t *)vj;
    j->fn(j->ctx);
}

/* Run fn(ctx) on the caller plus helpers. One job at a time: the decoder is
 * single-threaded per synthesis, and the server decodes slots sequentially. */
static void sd_pool_run(void (*fn)(void *), void *ctx) {
    int nt = sd_pool_threads();
    if (nt < 1) nt = 1;
    if (nt == 1) { fn(ctx); return; }

    if (qwen_parallel_is_reentrant()) {
        sd_gcd_job_t j = { fn, ctx };   /* macOS/GCD: no private pool needed */
        qwen_parallel((size_t)nt, sd_gcd_task, &j);
        return;
    }

    int want = nt - 1;
    if (want > SD_POOL_MAX_WORKERS) want = SD_POOL_MAX_WORKERS;
    if (want < 0) want = 0;
    pthread_mutex_lock(&sdp_mu);
    if (!sdp_started) {
        for (int i = 0; i < want; i++)
            if (pthread_create(&sdp_threads[sdp_nworkers], NULL, sdp_worker_main, NULL) == 0)
                sdp_nworkers++;
        sdp_started = 1;
    }
    sdp_fn = fn; sdp_ctx = ctx;
    sdp_pending = sdp_nworkers;
    sdp_gen++;
    pthread_cond_broadcast(&sdp_cv);
    pthread_mutex_unlock(&sdp_mu);

    fn(ctx);   /* caller participates */

    pthread_mutex_lock(&sdp_mu);
    while (sdp_pending > 0) pthread_cond_wait(&sdp_done_cv, &sdp_mu);
    pthread_mutex_unlock(&sdp_mu);
}


#if defined(__ARM_FEATURE_DOTPROD)

/* 2 rows x 4 cols register tile with per-block fp32 accumulation.
 * 8 int32 dot accs + 8 fp32 accs + 6 live loads = 22 registers. */
static inline void sd_tile_2x4(float *out, int out_ld, int m, int tcol,
                               const int8_t *Wq, const float *swb, const float *bias,
                               const int8_t *Xq, const float *sab, int xrow,
                               int Kp, int blk, int nblk) {
    const int8_t *w0 = Wq + (size_t)(m + 0) * Kp, *w1 = Wq + (size_t)(m + 1) * Kp;
    const int8_t *x0 = Xq + (size_t)(xrow + 0) * Kp, *x1 = Xq + (size_t)(xrow + 1) * Kp;
    const int8_t *x2 = Xq + (size_t)(xrow + 2) * Kp, *x3 = Xq + (size_t)(xrow + 3) * Kp;
    const float *sw0 = swb + (size_t)(m + 0) * nblk, *sw1 = swb + (size_t)(m + 1) * nblk;
    const float *sa0 = sab + (size_t)(xrow + 0) * nblk, *sa1 = sab + (size_t)(xrow + 1) * nblk;
    const float *sa2 = sab + (size_t)(xrow + 2) * nblk, *sa3 = sab + (size_t)(xrow + 3) * nblk;
    float32x4_t f00 = vdupq_n_f32(0), f01 = f00, f02 = f00, f03 = f00;
    float32x4_t f10 = f00, f11 = f00, f12 = f00, f13 = f00;
    for (int b = 0; b < nblk; b++) {
        int32x4_t a00 = vdupq_n_s32(0), a01 = a00, a02 = a00, a03 = a00;
        int32x4_t a10 = a00, a11 = a00, a12 = a00, a13 = a00;
        int kend = (b + 1) * blk;
        for (int k = b * blk; k < kend; k += 16) {
            int8x16_t xv0 = vld1q_s8(x0 + k), xv1 = vld1q_s8(x1 + k);
            int8x16_t xv2 = vld1q_s8(x2 + k), xv3 = vld1q_s8(x3 + k);
            int8x16_t wv = vld1q_s8(w0 + k);
            a00 = vdotq_s32(a00, wv, xv0); a01 = vdotq_s32(a01, wv, xv1);
            a02 = vdotq_s32(a02, wv, xv2); a03 = vdotq_s32(a03, wv, xv3);
            wv = vld1q_s8(w1 + k);
            a10 = vdotq_s32(a10, wv, xv0); a11 = vdotq_s32(a11, wv, xv1);
            a12 = vdotq_s32(a12, wv, xv2); a13 = vdotq_s32(a13, wv, xv3);
        }
        float s0 = sw0[b], s1 = sw1[b];
        f00 = vfmaq_n_f32(f00, vcvtq_f32_s32(a00), s0 * sa0[b]);
        f01 = vfmaq_n_f32(f01, vcvtq_f32_s32(a01), s0 * sa1[b]);
        f02 = vfmaq_n_f32(f02, vcvtq_f32_s32(a02), s0 * sa2[b]);
        f03 = vfmaq_n_f32(f03, vcvtq_f32_s32(a03), s0 * sa3[b]);
        f10 = vfmaq_n_f32(f10, vcvtq_f32_s32(a10), s1 * sa0[b]);
        f11 = vfmaq_n_f32(f11, vcvtq_f32_s32(a11), s1 * sa1[b]);
        f12 = vfmaq_n_f32(f12, vcvtq_f32_s32(a12), s1 * sa2[b]);
        f13 = vfmaq_n_f32(f13, vcvtq_f32_s32(a13), s1 * sa3[b]);
    }
    float b0 = bias ? bias[m + 0] : 0.0f, b1 = bias ? bias[m + 1] : 0.0f;
    float *o0 = out + (size_t)(m + 0) * out_ld + tcol;
    float *o1 = out + (size_t)(m + 1) * out_ld + tcol;
    o0[0] = vaddvq_f32(f00) + b0; o0[1] = vaddvq_f32(f01) + b0;
    o0[2] = vaddvq_f32(f02) + b0; o0[3] = vaddvq_f32(f03) + b0;
    o1[0] = vaddvq_f32(f10) + b1; o1[1] = vaddvq_f32(f11) + b1;
    o1[2] = vaddvq_f32(f12) + b1; o1[3] = vaddvq_f32(f13) + b1;
}

/* 1 row x up to 4 cols tail tile */
static inline void sd_tile_1xN(float *out, int out_ld, int m, int tcol,
                               const int8_t *Wq, const float *swb, const float *bias,
                               const int8_t *Xq, const float *sab, int xrow, int ncols,
                               int Kp, int blk, int nblk) {
    const int8_t *w0 = Wq + (size_t)m * Kp;
    const float *sw0 = swb + (size_t)m * nblk;
    float32x4_t fc[4] = { vdupq_n_f32(0), vdupq_n_f32(0), vdupq_n_f32(0), vdupq_n_f32(0) };
    for (int b = 0; b < nblk; b++) {
        int32x4_t ac[4] = { vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0) };
        int kend = (b + 1) * blk;
        for (int k = b * blk; k < kend; k += 16) {
            int8x16_t wv = vld1q_s8(w0 + k);
            for (int c = 0; c < ncols; c++)
                ac[c] = vdotq_s32(ac[c], wv, vld1q_s8(Xq + (size_t)(xrow + c) * Kp + k));
        }
        float s0 = sw0[b];
        for (int c = 0; c < ncols; c++)
            fc[c] = vfmaq_n_f32(fc[c], vcvtq_f32_s32(ac[c]),
                                s0 * sab[(size_t)(xrow + c) * nblk + b]);
    }
    float bb = bias ? bias[m] : 0.0f;
    for (int c = 0; c < ncols; c++)
        out[(size_t)m * out_ld + tcol + c] = vaddvq_f32(fc[c]) + bb;
}

/* GEMM over one activation panel: out[m, tcol0 + c] for c in [0, nc).
 * Row-blocked (32) so the 4-column activation quad stays cache-resident
 * across row pairs. */
static void sd_gemm_panel(float *out, int out_ld, int M,
                          const int8_t *Wq, const float *swb, const float *bias,
                          const int8_t *Xq, const float *sab,
                          int tcol0, int nc, int Kp, int blk) {
    int nblk = Kp / blk;
    for (int rb = 0; rb < M; rb += 32) {
        int rbe = rb + 32 < M ? rb + 32 : M;
        int c = 0;
        for (; c + 3 < nc; c += 4) {
            int m = rb;
            for (; m + 1 < rbe; m += 2)
                sd_tile_2x4(out, out_ld, m, tcol0 + c, Wq, swb, bias, Xq, sab, c, Kp, blk, nblk);
            for (; m < rbe; m++)
                sd_tile_1xN(out, out_ld, m, tcol0 + c, Wq, swb, bias, Xq, sab, c, 4, Kp, blk, nblk);
        }
        if (c < nc)
            for (int m = rb; m < rbe; m++)
                sd_tile_1xN(out, out_ld, m, tcol0 + c, Wq, swb, bias, Xq, sab, c, nc - c, Kp, blk, nblk);
    }
}

/* ---------------- threaded int8 conv1d (im2col per panel) -------------- */

#define SD_INT8_NC 128  /* activation columns per panel */

typedef struct {
    float *out;
    const float *in;
    const int8_t *Wq; const float *sw; const float *bias;
    int in_ch, out_ch, length, kernel, dilation, Kp, blk;
    _Atomic int next_panel;
    int n_panels;
} sd_conv_job_t;

static void sd_conv1d_worker(void *vj) {
    sd_conv_job_t *j = (sd_conv_job_t *)vj;
    int K = j->in_ch * j->kernel;
    int nblk = j->Kp / j->blk;
    int pad_left = (j->kernel - 1) * j->dilation;
    float *colf = (float *)aligned_malloc((size_t)SD_INT8_NC * K * sizeof(float));
    int8_t *colq = (int8_t *)aligned_malloc((size_t)SD_INT8_NC * j->Kp);
    float *sa = (float *)aligned_malloc((size_t)SD_INT8_NC * nblk * sizeof(float));
    for (;;) {
        int p = atomic_fetch_add(&j->next_panel, 1);
        if (p >= j->n_panels) break;
        int t0 = p * SD_INT8_NC;
        int nc = j->length - t0 < SD_INT8_NC ? j->length - t0 : SD_INT8_NC;
        /* transposed im2col: colf[c][ic*kernel+kk] = in[ic][t0+c-pad+kk*dil] */
        for (int c = 0; c < nc; c++) {
            float *dst = colf + (size_t)c * K;
            int tt = t0 + c - pad_left;
            for (int ic = 0; ic < j->in_ch; ic++) {
                const float *src = j->in + (size_t)ic * j->length;
                float *dk = dst + (size_t)ic * j->kernel;
                for (int kk = 0; kk < j->kernel; kk++) {
                    int pos = tt + kk * j->dilation;
                    dk[kk] = (pos >= 0 && pos < j->length) ? src[pos] : 0.0f;
                }
            }
        }
        qwen_int8_quant_rows(colq, sa, colf, nc, K, j->Kp, j->blk);
        sd_gemm_panel(j->out, j->length, j->out_ch, j->Wq, j->sw, j->bias,
                      colq, sa, t0, nc, j->Kp, j->blk);
    }
    free(colf); free(colq); free(sa);
}

void qwen_conv1d_int8(float *out, const float *in,
                      const int8_t *Wq, const float *sw, const float *bias,
                      int in_ch, int out_ch, int length, int kernel, int dilation,
                      int Kp, int blk) {
    sd_conv_job_t job = {
        .out = out, .in = in, .Wq = Wq, .sw = sw, .bias = bias,
        .in_ch = in_ch, .out_ch = out_ch, .length = length,
        .kernel = kernel, .dilation = dilation, .Kp = Kp, .blk = blk,
        .n_panels = (length + SD_INT8_NC - 1) / SD_INT8_NC,
    };
    atomic_store(&job.next_panel, 0);
    sd_pool_run(sd_conv1d_worker, &job);
}

/* ---------------- threaded int8 GEMM on pre-quantized activations ------ */
/* Used by ConvTranspose: activations (transposed input) are quantized once
 * and reused across all kernel positions. Chunks are row blocks. */

typedef struct {
    float *out; int out_ld;
    const int8_t *Wq; const float *sw;
    const int8_t *Xq; const float *sa;
    int M, N, Kp, blk;
    _Atomic int next_block;
    int n_blocks, rows_per_block;
} sd_gemm_job_t;

static void sd_gemm_worker(void *vj) {
    sd_gemm_job_t *j = (sd_gemm_job_t *)vj;
    int nblk = j->Kp / j->blk;
    for (;;) {
        int b = atomic_fetch_add(&j->next_block, 1);
        if (b >= j->n_blocks) break;
        int m0 = b * j->rows_per_block;
        int m1 = m0 + j->rows_per_block < j->M ? m0 + j->rows_per_block : j->M;
        for (int t0 = 0; t0 < j->N; t0 += SD_INT8_NC) {
            int nc = j->N - t0 < SD_INT8_NC ? j->N - t0 : SD_INT8_NC;
            sd_gemm_panel(j->out + (size_t)m0 * j->out_ld, j->out_ld, m1 - m0,
                          j->Wq + (size_t)m0 * j->Kp, j->sw + (size_t)m0 * nblk, NULL,
                          j->Xq + (size_t)t0 * j->Kp, j->sa + (size_t)t0 * nblk,
                          t0, nc, j->Kp, j->blk);
        }
    }
}

void qwen_gemm_int8(float *out, int out_ld,
                    const int8_t *Wq, const float *sw,
                    const int8_t *Xq, const float *sa,
                    int M, int N, int Kp, int blk) {
    int nt = qwen_get_threads();
    int rpb = (M + nt * 2 - 1) / (nt * 2);
    rpb = (rpb + 1) & ~1;               /* multiple of 2 rows */
    if (rpb < 2) rpb = 2;
    sd_gemm_job_t job = {
        .out = out, .out_ld = out_ld, .Wq = Wq, .sw = sw, .Xq = Xq, .sa = sa,
        .M = M, .N = N, .Kp = Kp, .blk = blk,
        .rows_per_block = rpb, .n_blocks = (M + rpb - 1) / rpb,
    };
    atomic_store(&job.next_block, 0);
    sd_pool_run(sd_gemm_worker, &job);
}

#else /* !__ARM_FEATURE_DOTPROD: scalar reference (unused in practice) */

static float sd_scalar_dot(const int8_t *w, const float *swb,
                           const int8_t *x, const float *sab, int Kp, int blk) {
    int nblk = Kp / blk;
    float acc = 0.0f;
    for (int b = 0; b < nblk; b++) {
        int32_t ai = 0;
        for (int k = b * blk; k < (b + 1) * blk; k++)
            ai += (int32_t)w[k] * x[k];
        acc += (float)ai * swb[b] * sab[b];
    }
    return acc;
}

void qwen_conv1d_int8(float *out, const float *in,
                      const int8_t *Wq, const float *sw, const float *bias,
                      int in_ch, int out_ch, int length, int kernel, int dilation,
                      int Kp, int blk) {
    int K = in_ch * kernel;
    int nblk = Kp / blk;
    int pad_left = (kernel - 1) * dilation;
    float *colf = (float *)aligned_malloc((size_t)K * sizeof(float));
    int8_t *colq = (int8_t *)aligned_malloc((size_t)Kp);
    float *sa = (float *)aligned_malloc((size_t)nblk * sizeof(float));
    for (int t = 0; t < length; t++) {
        for (int ic = 0; ic < in_ch; ic++)
            for (int kk = 0; kk < kernel; kk++) {
                int pos = t - pad_left + kk * dilation;
                colf[ic * kernel + kk] =
                    (pos >= 0 && pos < length) ? in[(size_t)ic * length + pos] : 0.0f;
            }
        qwen_int8_quant_rows(colq, sa, colf, 1, K, Kp, blk);
        for (int m = 0; m < out_ch; m++)
            out[(size_t)m * length + t] =
                sd_scalar_dot(Wq + (size_t)m * Kp, sw + (size_t)m * nblk, colq, sa, Kp, blk)
                + (bias ? bias[m] : 0.0f);
    }
    free(colf); free(colq); free(sa);
}

void qwen_gemm_int8(float *out, int out_ld,
                    const int8_t *Wq, const float *sw,
                    const int8_t *Xq, const float *sa,
                    int M, int N, int Kp, int blk) {
    int nblk = Kp / blk;
    for (int m = 0; m < M; m++)
        for (int t = 0; t < N; t++)
            out[(size_t)m * out_ld + t] =
                sd_scalar_dot(Wq + (size_t)m * Kp, sw + (size_t)m * nblk,
                              Xq + (size_t)t * Kp, sa + (size_t)t * nblk, Kp, blk);
}

#endif /* __ARM_FEATURE_DOTPROD */

/* ------------------------------------------------------------------------
 * Vectorized sin²(y) for the snake activation. The snake needs only sin², so:
 * reduce y by π (not 2π): u = y − n·π ∈ [−π/2, π/2], n = round(y/π). That leaves
 * sin(u) = ±sin(y), and SQUARING discards the sign — no quadrant bookkeeping.
 * π is split Cody-Waite (hi+lo) because n·π_float alone loses the low bits.
 * Odd Taylor series to u¹¹ (dropped term u¹³/13! ≈ 1.2e-8 at |u|=π/2, under a
 * float ULP). Idea from PR #17 (TrinityTF); the |y| guard is ours.
 *
 * Above QWEN_SIN_POLY_MAX the 2-term reduction loses too many bits (n grows, the
 * π_lo residual is amplified), so the caller falls back to libm there. Snake's
 * α·x stays far below it — the guard just refuses to be silently wrong if not.
 *
 * NEON and AVX2 twins: the __AVX2__ branch of the snake had the exact same
 * scalar-sinf-per-lane problem the NEON branch did, and PR #17 only fixed NEON.
 * ------------------------------------------------------------------------ */
#if (defined(__ARM_NEON) || defined(__AVX2__)) && !(defined(__APPLE__) && defined(USE_BLAS))
#define QWEN_SIN_POLY_MAX 8192.0f
#define QWEN_SIN_C1  (-1.0f / 6.0f)
#define QWEN_SIN_C2  ( 1.0f / 120.0f)
#define QWEN_SIN_C3  (-1.0f / 5040.0f)
#define QWEN_SIN_C4  ( 1.0f / 362880.0f)
#define QWEN_SIN_C5  (-1.0f / 39916800.0f)
#define QWEN_PI_HI   3.14159274101257324f    /* float(π)     */
#define QWEN_PI_LO  (-8.74227800708368e-8f)  /* π − float(π) */
#define QWEN_INV_PI  0.31830988618379067f

/* QWEN_NO_SIN_POLY=1 restores the per-lane libm sinf — the A/B switch (like
 * QWEN_NO_SDOT). Cached once; read outside the hot loop. */
static int qwen_sin_poly_off(void) {
    static atomic_int off = -1;
    int v = atomic_load_explicit(&off, memory_order_relaxed);
    if (v < 0) {
        const char *e = getenv("QWEN_NO_SIN_POLY");
        v = (e && e[0] == '1');
        atomic_store_explicit(&off, v, memory_order_relaxed);
    }
    return v;
}
#endif

#if defined(__ARM_NEON) && !(defined(__APPLE__) && defined(USE_BLAS))
static inline float32x4_t qwen_vsin2q_f32(float32x4_t y) {
    const float32x4_t inv_pi = vdupq_n_f32(QWEN_INV_PI);
    const float32x4_t pi_hi  = vdupq_n_f32(QWEN_PI_HI);
    const float32x4_t pi_lo  = vdupq_n_f32(QWEN_PI_LO);
    float32x4_t n = vrndaq_f32(vmulq_f32(y, inv_pi));               /* round-to-nearest, ties away */
    float32x4_t u = vfmsq_f32(y, n, pi_hi);
    u = vfmaq_f32(u, n, pi_lo);
    float32x4_t u2 = vmulq_f32(u, u);
    /* Horner on u²: s = u·(1 − u²/6 + u⁴/120 − u⁶/5040 + u⁸/362880 − u¹⁰/39916800) */
    float32x4_t p = vdupq_n_f32(QWEN_SIN_C5);
    p = vfmaq_f32(vdupq_n_f32(QWEN_SIN_C4), p, u2);
    p = vfmaq_f32(vdupq_n_f32(QWEN_SIN_C3), p, u2);
    p = vfmaq_f32(vdupq_n_f32(QWEN_SIN_C2), p, u2);
    p = vfmaq_f32(vdupq_n_f32(QWEN_SIN_C1), p, u2);
    p = vfmaq_f32(vdupq_n_f32(1.0f),        p, u2);
    float32x4_t s = vmulq_f32(u, p);
    return vmulq_f32(s, s);
}
#endif /* __ARM_NEON */

#if defined(__AVX2__) && !(defined(__APPLE__) && defined(USE_BLAS))
/* AVX2 twin of qwen_vsin2q_f32 (8 lanes). Same reduction and coefficients.
 * _mm256_round_ps with NEAREST|NO_EXC == round-to-nearest-even; the ½-ULP
 * difference from NEON's ties-away is immaterial after the u¹¹ series + square. */
static inline __m256 qwen_vsin2_avx2(__m256 y) {
    const __m256 inv_pi = _mm256_set1_ps(QWEN_INV_PI);
    const __m256 pi_hi  = _mm256_set1_ps(QWEN_PI_HI);
    const __m256 pi_lo  = _mm256_set1_ps(QWEN_PI_LO);
    __m256 n = _mm256_round_ps(_mm256_mul_ps(y, inv_pi),
                              _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m256 u = _mm256_fnmadd_ps(n, pi_hi, y);   /* y − n·pi_hi */
    u = _mm256_fmadd_ps(n, pi_lo, u);           /* + n·pi_lo   */
    __m256 u2 = _mm256_mul_ps(u, u);
    __m256 p = _mm256_set1_ps(QWEN_SIN_C5);
    p = _mm256_fmadd_ps(p, u2, _mm256_set1_ps(QWEN_SIN_C4));
    p = _mm256_fmadd_ps(p, u2, _mm256_set1_ps(QWEN_SIN_C3));
    p = _mm256_fmadd_ps(p, u2, _mm256_set1_ps(QWEN_SIN_C2));
    p = _mm256_fmadd_ps(p, u2, _mm256_set1_ps(QWEN_SIN_C1));
    p = _mm256_fmadd_ps(p, u2, _mm256_set1_ps(1.0f));
    __m256 s = _mm256_mul_ps(u, p);
    return _mm256_mul_ps(s, s);
}
#endif /* __AVX2__ */

/* One channel row of the snake: y = x + (1/beta)*sin^2(alpha*x). Rows are fully
 * independent; the only thing that stood between us and threading them was a
 * dispatcher callable from the decoder thread -- which sd_pool_run is. */
static void snake_row(float *data, int c, int length,
                      const float *log_alpha, const float *log_beta) {
#if (defined(__ARM_NEON) || defined(__AVX2__)) && !(defined(__APPLE__) && defined(USE_BLAS))
    const int sin_poly = !qwen_sin_poly_off();
#endif
    {
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
                float32x4_t s2;
                if (sin_poly && vmaxvq_f32(vabsq_f32(ax)) <= QWEN_SIN_POLY_MAX) {
                    s2 = qwen_vsin2q_f32(ax);
                } else {
                    /* Range reduction would lose too many bits here: fall back to libm. */
                    float ax_s[4];
                    vst1q_f32(ax_s, ax);
                    float s_arr[4] = { sinf(ax_s[0]), sinf(ax_s[1]),
                                       sinf(ax_s[2]), sinf(ax_s[3]) };
                    float32x4_t s = vld1q_f32(s_arr);
                    s2 = vmulq_f32(s, s);
                }
                x = vfmaq_f32(x, vinv_b, s2);
                vst1q_f32(row + t, x);
            }
            for (; t < length; t++) {
                float s = sinf(a * row[t]);
                row[t] += inv_b * s * s;
            }
        }
#elif defined(__AVX2__)
        {
            __m256 va = _mm256_set1_ps(a);
            __m256 vinv_b = _mm256_set1_ps(inv_b);
            int t = 0;
            const __m256 sign_mask = _mm256_set1_ps(-0.0f);
            const __m256 poly_max  = _mm256_set1_ps(QWEN_SIN_POLY_MAX);
            for (; t + 8 <= length; t += 8) {
                __m256 x = _mm256_loadu_ps(row + t);
                __m256 ax = _mm256_mul_ps(va, x);
                __m256 s2;
                /* Poly path unless disabled or any lane is beyond the reduction's
                 * safe range (then libm, like the NEON twin). movemask != 0 means
                 * at least one lane exceeds the guard. */
                __m256 over = _mm256_cmp_ps(_mm256_andnot_ps(sign_mask, ax), poly_max, _CMP_GT_OQ);
                if (sin_poly && _mm256_movemask_ps(over) == 0) {
                    s2 = qwen_vsin2_avx2(ax);
                } else {
                    float ax_s[8]; _mm256_storeu_ps(ax_s, ax);
                    float s_arr[8] = { sinf(ax_s[0]), sinf(ax_s[1]), sinf(ax_s[2]), sinf(ax_s[3]),
                                       sinf(ax_s[4]), sinf(ax_s[5]), sinf(ax_s[6]), sinf(ax_s[7]) };
                    __m256 s = _mm256_loadu_ps(s_arr);
                    s2 = _mm256_mul_ps(s, s);
                }
                x = _mm256_fmadd_ps(vinv_b, s2, x);
                _mm256_storeu_ps(row + t, x);
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

/* Threaded snake. Rows are claimed from an atomic counter, so a slow row cannot
 * stall a whole thread's stripe. PR #17 measured the snake at 1209 ms on a 7.4 s
 * clip: the polynomial sine took it to 341 ms, threading the rows took it to
 * ~90 ms. Threading was the larger half, and this is that half.
 *
 * Below a work threshold we stay serial: the early ConvNeXt-stage snakes are
 * small and the dispatch costs more than it saves. */
typedef struct {
    float *data; int channels, length;
    const float *log_alpha, *log_beta;
    _Atomic int next;
} snake_job_t;

static void snake_worker(void *vj) {
    snake_job_t *j = (snake_job_t *)vj;
    for (;;) {
        int c = atomic_fetch_add(&j->next, 1);
        if (c >= j->channels) break;
        snake_row(j->data, c, j->length, j->log_alpha, j->log_beta);
    }
}

#define QWEN_SNAKE_MIN_WORK 65536   /* elements; below this, dispatch dominates */

void qwen_snake_activation(float *data, int channels, int length,
                            const float *log_alpha, const float *log_beta) {
    if ((int64_t)channels * length < QWEN_SNAKE_MIN_WORK || qwen_get_threads() <= 1) {
        for (int c = 0; c < channels; c++)
            snake_row(data, c, length, log_alpha, log_beta);
        return;
    }
    snake_job_t job;
    job.data = data; job.channels = channels; job.length = length;
    job.log_alpha = log_alpha; job.log_beta = log_beta;
    atomic_init(&job.next, 0);
    sd_pool_run(snake_worker, &job);
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
#elif defined(__AVX2__)
    /* 2 rows × 4 __m256 accumulators (8 chains), 32 elem/iter, + prefetch — NEON parity. */
    for (; o + 1 < out_dim; o += 2) {
        const uint16_t *w0 = W_bf16 + (size_t)o * in_dim;
        const uint16_t *w1 = W_bf16 + (size_t)(o + 1) * in_dim;
        if (o + 5 < out_dim) {
            __builtin_prefetch(W_bf16 + (size_t)(o + 4) * in_dim, 0, 0);
            __builtin_prefetch(W_bf16 + (size_t)(o + 5) * in_dim, 0, 0);
        }
        __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps(),
               a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
        __m256 b0 = _mm256_setzero_ps(), b1 = _mm256_setzero_ps(),
               b2 = _mm256_setzero_ps(), b3 = _mm256_setzero_ps();
        int k = 0;
        for (; k + 32 <= in_dim; k += 32) {
            __m256 x0 = _mm256_loadu_ps(x + k);
            __m256 x1 = _mm256_loadu_ps(x + k + 8);
            __m256 x2 = _mm256_loadu_ps(x + k + 16);
            __m256 x3 = _mm256_loadu_ps(x + k + 24);
            a0 = _mm256_fmadd_ps(qwen_loadu_bf16_8(w0 + k),      x0, a0);
            a1 = _mm256_fmadd_ps(qwen_loadu_bf16_8(w0 + k + 8),  x1, a1);
            a2 = _mm256_fmadd_ps(qwen_loadu_bf16_8(w0 + k + 16), x2, a2);
            a3 = _mm256_fmadd_ps(qwen_loadu_bf16_8(w0 + k + 24), x3, a3);
            b0 = _mm256_fmadd_ps(qwen_loadu_bf16_8(w1 + k),      x0, b0);
            b1 = _mm256_fmadd_ps(qwen_loadu_bf16_8(w1 + k + 8),  x1, b1);
            b2 = _mm256_fmadd_ps(qwen_loadu_bf16_8(w1 + k + 16), x2, b2);
            b3 = _mm256_fmadd_ps(qwen_loadu_bf16_8(w1 + k + 24), x3, b3);
        }
        for (; k + 8 <= in_dim; k += 8) {
            __m256 xv = _mm256_loadu_ps(x + k);
            a0 = _mm256_fmadd_ps(qwen_loadu_bf16_8(w0 + k), xv, a0);
            b0 = _mm256_fmadd_ps(qwen_loadu_bf16_8(w1 + k), xv, b0);
        }
        a0 = _mm256_add_ps(_mm256_add_ps(a0, a2), _mm256_add_ps(a1, a3));
        b0 = _mm256_add_ps(_mm256_add_ps(b0, b2), _mm256_add_ps(b1, b3));
        float s0 = qwen_hsum256_ps(a0), s1 = qwen_hsum256_ps(b0);
        for (; k < in_dim; k++) { s0 += bf16_to_f32(w0[k]) * x[k]; s1 += bf16_to_f32(w1[k]) * x[k]; }
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

/* ========================================================================
 * Kernel numeric self-test (`./qwen_tts --self-test`)
 *
 * Cross-ISA correctness proof for the matvec kernels that does NOT depend on a
 * full-pipeline golden — immune to the greedy-decode trajectory fork that makes
 * cross-ISA / cross-precision audio mel-corr drop benignly. Runs each dispatched
 * matvec (bf16 / int8 / argmax-int8) against an independent f32 reference on
 * deterministic pseudo-random data and checks the error is within tolerance.
 *
 * On x86 with SIMD=avx512vnni this exercises the VNNI int8 dot (`_mm512_dpbusd`)
 * and the __m512 bf16 matvec (the two UNVALIDATED AVX-512 paths). On ARM it
 * exercises SDOT/NEON — useful as a methodology check before trusting the VPS run.
 * Run twice to A/B the dispatch:  ./qwen_tts --self-test   (VNNI/SDOT on)
 *                          QWEN_NO_VNNI=1 ./qwen_tts --self-test   (scalar/AVX2 fallback)
 * Returns 0 on PASS, non-zero on FAIL (so it can gate CI / `make test-selftest`).
 * ======================================================================== */
int qwen_kernel_selftest(void *out) {
    FILE *f = out ? (FILE *)out : stdout;
    /* Deterministic LCG — no rand()/Date dependence (those break reproducibility). */
    uint64_t rng = 0x9E3779B97F4A7C15ull;
    #define NEXT_F (( (rng = rng * 6364136223846793005ull + 1442695040888963407ull) >> 40) \
                    / (float)(1u << 24) * 2.0f - 1.0f)   /* uniform in [-1,1) */

    /* Two CP-shaped matrices: gate_up [3072×1024] and a rows-not-multiple case. */
    const int cases[][2] = { {3072, 1024}, {2048, 1024}, {257, 320} };
    const int ncases = (int)(sizeof(cases) / sizeof(cases[0]));
    int failures = 0;

    fprintf(f, "qwen-tts kernel self-test (matvec correctness vs f32 reference)\n");
    qwen_caps_report(f);
    fprintf(f, "  (run with QWEN_NO_VNNI=1 / QWEN_NO_SDOT=1 to test the fallback path)\n\n");

    for (int ci = 0; ci < ncases; ci++) {
        int rows = cases[ci][0], cols = cases[ci][1];
        float    *x   = malloc((size_t)cols * sizeof(float));
        float    *wf  = malloc((size_t)rows * cols * sizeof(float));
        uint16_t *wb  = malloc((size_t)rows * cols * sizeof(uint16_t));
        int8_t   *wi  = malloc((size_t)rows * cols * sizeof(int8_t));
        float    *sc  = malloc((size_t)rows * sizeof(float));
        float    *ref = malloc((size_t)rows * sizeof(float));
        float    *y   = malloc((size_t)rows * sizeof(float));
        if (!x || !wf || !wb || !wi || !sc || !ref || !y) {
            fprintf(f, "  [case %dx%d] OOM, skipped\n", rows, cols);
            free(x); free(wf); free(wb); free(wi); free(sc); free(ref); free(y);
            continue;
        }
        for (int k = 0; k < cols; k++) x[k] = NEXT_F;
        for (size_t i = 0; i < (size_t)rows * cols; i++) {
            float v = NEXT_F;
            wf[i] = v;
            /* round-to-nearest f32->bf16 so the reference dequant matches exactly */
            uint32_t bits; memcpy(&bits, &v, 4);
            wb[i] = (uint16_t)((bits + 0x8000u) >> 16);
        }

        /* ---- bf16 matvec ---- */
        for (int r = 0; r < rows; r++) {
            float s = 0.0f;
            const uint16_t *row = wb + (size_t)r * cols;
            for (int k = 0; k < cols; k++) s += bf16_to_f32(row[k]) * x[k];
            ref[r] = s;
        }
        qwen_matvec_bf16(y, wb, x, rows, cols);
        double max_rel_bf16 = 0.0;
        for (int r = 0; r < rows; r++) {
            double denom = fabs(ref[r]) + 1e-3;
            double rel = fabs((double)y[r] - ref[r]) / denom;
            if (rel > max_rel_bf16) max_rel_bf16 = rel;
        }

        /* ---- batched matmat: Y[rows,B] must equal B independent matvecs ----
         * (the batching / spec-decode-verify primitive). Each column b is x scaled,
         * so qwen_matvec_bf16 gives the per-column reference; only fp accumulation
         * ORDER differs, so compare with a global L2 relative error. */
        {
            const int B = 8;
            float *Xb  = malloc((size_t)cols * B * sizeof(float));
            float *Yb  = malloc((size_t)rows * B * sizeof(float));
            float *xb  = malloc((size_t)cols * sizeof(float));
            float *yc  = malloc((size_t)rows * sizeof(float));
            if (Xb && Yb && xb && yc) {
                for (int k = 0; k < cols; k++)
                    for (int b = 0; b < B; b++) Xb[(size_t)k * B + b] = x[k] * (1.0f + 0.05f * b);
                qwen_matmat_bf16(Yb, wb, Xb, rows, cols, B);
                double l2n = 0.0, l2d = 0.0;
                for (int b = 0; b < B; b++) {
                    for (int k = 0; k < cols; k++) xb[k] = x[k] * (1.0f + 0.05f * b);
                    qwen_matvec_bf16(yc, wb, xb, rows, cols);
                    for (int r = 0; r < rows; r++) {
                        double d = (double)Yb[(size_t)r * B + b] - yc[r];
                        l2n += d * d; l2d += (double)yc[r] * yc[r];
                    }
                }
                double l2rel = l2d > 0 ? sqrt(l2n / l2d) : 0.0;
                fprintf(f, "  [%4dx%4d] matmat(B=%d) vs B*matvec: L2_rel=%.2e  %s\n",
                        rows, cols, B, l2rel, l2rel < 1e-4 ? "PASS" : "FAIL");
                if (!(l2rel < 1e-4)) failures++;
            }
            free(Xb); free(Yb); free(xb); free(yc);
        }

        /* ---- int8 matvec ---- (reference = exact int8 dot with the SAME scales) */
        qwen_quantize_bf16_to_int8(wb, rows, cols, wi, sc);
        for (int r = 0; r < rows; r++) {
            const int8_t *row = wi + (size_t)r * cols;
            float s = 0.0f;
            for (int k = 0; k < cols; k++) s += (float)row[k] * x[k];
            ref[r] = sc[r] * s;   /* dequant-W · x  (activation kept f32 in the reference) */
        }
        qwen_matvec_int8(y, wi, sc, x, rows, cols);
        /* int8 dispatch (SDOT/VNNI) quantizes the ACTIVATION -> a roughly-CONSTANT
         * absolute error per row, so a per-row *relative* error explodes on the rows
         * where ref[r]≈0 (random dots cluster near zero) and means nothing. The right,
         * near-zero-robust metric is the GLOBAL L2 relative error ||y-ref|| / ||ref||:
         * activation-quant noise lands it ~0.7% for a correct kernel; a broken VNNI
         * offset would blow it up. (bf16 has only fp accumulation-order drift -> tiny,
         * so max-rel is fine there.) */
        double l2_num = 0.0, l2_den = 0.0;
        for (int r = 0; r < rows; r++) {
            double d = (double)y[r] - ref[r];
            l2_num += d * d;
            l2_den += (double)ref[r] * ref[r];
        }
        double rel_l2_i8 = sqrt(l2_num / (l2_den + 1e-12));

        /* ---- argmax int8 ---- (must agree with the reference argmax, or tie within eps) */
        int amax_ref = 0; float amax_val = ref[0];
        for (int r = 1; r < rows; r++) if (ref[r] > amax_val) { amax_val = ref[r]; amax_ref = r; }
        int amax_got = qwen_argmax_matvec_int8(x, wi, sc, cols, rows);
        int argmax_ok = (amax_got == amax_ref) ||
                        (amax_got >= 0 && amax_got < rows &&
                         (amax_val - ref[amax_got]) < 0.02 * (fabs(amax_val) + 1e-3));

        int bf16_ok = max_rel_bf16 < 1e-2;   /* bf16: only fp accumulation-order drift */
        int i8_ok   = rel_l2_i8    < 3e-2;   /* int8: activation-quant noise (~0.7% expected) */
        if (!bf16_ok || !i8_ok || !argmax_ok) failures++;
        fprintf(f, "  [%4dx%-4d] bf16 max_rel=%.2e %s | int8 rel_L2=%.2e %s | argmax %s (ref=%d got=%d)\n",
                rows, cols, max_rel_bf16, bf16_ok ? "OK" : "FAIL",
                rel_l2_i8, i8_ok ? "OK" : "FAIL",
                argmax_ok ? "OK" : "FAIL", amax_ref, amax_got);

        /* ---- int8 batched matmat: Y[rows,B] vs B independent int8 matvecs ----
         * matmat_int8 keeps the activation f32 (like the ARM matvec) and reuses the
         * same per-row scales, so it should track B× qwen_matvec_int8 closely; allow
         * the same activation-quant tolerance the dispatched int8 matvec needs. */
        {
            const int B = 8;
            float *Xb = malloc((size_t)cols * B * sizeof(float));
            float *Yb = malloc((size_t)rows * B * sizeof(float));
            float *xb = malloc((size_t)cols * sizeof(float));
            float *yc = malloc((size_t)rows * sizeof(float));
            if (Xb && Yb && xb && yc) {
                for (int k = 0; k < cols; k++)
                    for (int b = 0; b < B; b++) Xb[(size_t)k * B + b] = x[k] * (1.0f + 0.05f * b);
                qwen_matmat_int8(Yb, wi, sc, Xb, rows, cols, B);
                double l2n = 0.0, l2d = 0.0;
                for (int b = 0; b < B; b++) {
                    for (int k = 0; k < cols; k++) xb[k] = x[k] * (1.0f + 0.05f * b);
                    qwen_matvec_int8(yc, wi, sc, xb, rows, cols);
                    for (int r = 0; r < rows; r++) {
                        double d = (double)Yb[(size_t)r * B + b] - yc[r];
                        l2n += d * d; l2d += (double)yc[r] * yc[r];
                    }
                }
                double l2rel = l2d > 0 ? sqrt(l2n / l2d) : 0.0;
                int ok = l2rel < 3e-2;
                fprintf(f, "  [%4dx%4d] matmat_int8(B=%d) vs B*matvec_int8: L2_rel=%.2e  %s\n",
                        rows, cols, B, l2rel, ok ? "PASS" : "FAIL");
                if (!ok) failures++;
            }
            free(Xb); free(Yb); free(xb); free(yc);
        }

        /* ---- q4_0 batched matmat: Y[rows,B] vs B independent q4_0 matvecs ----
         * matmat_q4_0 keeps the activation f32 (nibble-dequant -> f32 FMA), while
         * the dispatched matvec_q4_0 now uses the SDOT path (activation quantized
         * to int8, plan_v4 B1). So they differ by the same activation-quant tolerance
         * the int8 twin needs (was 1e-3 fp-order back when both kept f32 act). */
        if (cols % Q4_0_BLOCK_SIZE == 0) {
            const int B = 8;
            int nb = cols / Q4_0_BLOCK_SIZE;
            q4_0_block_t *wq = malloc((size_t)rows * nb * sizeof(q4_0_block_t));
            float *Xb = malloc((size_t)cols * B * sizeof(float));
            float *Yb = malloc((size_t)rows * B * sizeof(float));
            float *xb = malloc((size_t)cols * sizeof(float));
            float *yc = malloc((size_t)rows * sizeof(float));
            if (wq && Xb && Yb && xb && yc) {
                qwen_quantize_bf16_to_q4_0(wb, rows, cols, wq);
                for (int k = 0; k < cols; k++)
                    for (int b = 0; b < B; b++) Xb[(size_t)k * B + b] = x[k] * (1.0f + 0.05f * b);
                qwen_matmat_q4_0(Yb, wq, Xb, rows, cols, B);
                double l2n = 0.0, l2d = 0.0;
                for (int b = 0; b < B; b++) {
                    for (int k = 0; k < cols; k++) xb[k] = x[k] * (1.0f + 0.05f * b);
                    qwen_matvec_q4_0(yc, wq, xb, rows, cols);
                    for (int r = 0; r < rows; r++) {
                        double d = (double)Yb[(size_t)r * B + b] - yc[r];
                        l2n += d * d; l2d += (double)yc[r] * yc[r];
                    }
                }
                double l2rel = l2d > 0 ? sqrt(l2n / l2d) : 0.0;
                int ok = l2rel < 3e-2;
                fprintf(f, "  [%4dx%4d] matmat_q4_0(B=%d) vs B*matvec_q4_0: L2_rel=%.2e  %s\n",
                        rows, cols, B, l2rel, ok ? "PASS" : "FAIL");
                if (!ok) failures++;
            }
            free(wq); free(Xb); free(Yb); free(xb); free(yc);
        }

        free(x); free(wf); free(wb); free(wi); free(sc); free(ref); free(y);
    }
    #undef NEXT_F
    fprintf(f, "\n%s (%d case%s failed)\n", failures ? "SELF-TEST FAILED" : "SELF-TEST PASSED",
            failures, failures == 1 ? "" : "s");
    return failures;
}

/* ========================================================================
 * Batched matmat throughput microbench (`./qwen_tts --matmat-bench`)
 *
 * Times the REAL library kernels (NOT the naive premise bench): for each
 * precision and shape, B independent qwen_matvec_* calls (= today's single-
 * stream, weights re-read B×) vs one qwen_matmat_* call (= batched, weights
 * read once). speedup = t_seq / t_batch. Answers "does batching beat sequential
 * per precision, and by how much" using the production kernels, at the current
 * -j thread count. No model needed. Tune B with QWEN_BATCH_B (default 8).
 * ======================================================================== */
int qwen_matmat_bench(void *out) {
    FILE *f = out ? (FILE *)out : stdout;
    const char *be = getenv("QWEN_BATCH_B"); int B = be ? atoi(be) : 8;
    if (B < 1 || B > 64) B = 8;
    const int shapes[][2] = { {3072, 1024}, {1024, 3072}, {2048, 1024} };
    const int nshapes = (int)(sizeof(shapes) / sizeof(shapes[0]));
    uint64_t rng = 0x1234567ull;
    #define RF (((rng = rng * 6364136223846793005ull + 1442695040888963407ull) >> 40) \
                / (float)(1u << 24) * 2.0f - 1.0f)
    #define NOW_S(t) clock_gettime(CLOCK_MONOTONIC, &(t))
    #define MS(a,b) (((b).tv_sec-(a).tv_sec)*1e3 + ((b).tv_nsec-(a).tv_nsec)*1e-6)
    struct timespec t0, t1;

    fprintf(f, "matmat-bench: B=%d, threads=%d  (B*matvec [seq] vs matmat [batched])\n", B, qwen_get_threads());
    fprintf(f, "  speedup>1 => batching (weight read+unpack once) beats re-reading per stream\n\n");

    for (int si = 0; si < nshapes; si++) {
        int rows = shapes[si][0], cols = shapes[si][1];
        int nb = cols / Q4_0_BLOCK_SIZE;
        uint16_t *wb = malloc((size_t)rows * cols * sizeof(uint16_t));
        int8_t   *wi = malloc((size_t)rows * cols * sizeof(int8_t));
        float    *sc = malloc((size_t)rows * sizeof(float));
        q4_0_block_t *wq = malloc((size_t)rows * nb * sizeof(q4_0_block_t));
        float *X  = malloc((size_t)cols * B * sizeof(float));
        float *xb = malloc((size_t)cols * sizeof(float));
        float *Y  = malloc((size_t)rows * B * sizeof(float));
        float *yc = malloc((size_t)rows * sizeof(float));
        if (!wb || !wi || !sc || !wq || !X || !xb || !Y || !yc) {
            fprintf(f, "  [%dx%d] OOM, skipped\n", rows, cols);
            free(wb); free(wi); free(sc); free(wq); free(X); free(xb); free(Y); free(yc); continue;
        }
        for (size_t i = 0; i < (size_t)rows * cols; i++) {
            float v = RF; uint32_t bits; memcpy(&bits, &v, 4);
            wb[i] = (uint16_t)((bits + 0x8000u) >> 16);
        }
        qwen_quantize_bf16_to_int8(wb, rows, cols, wi, sc);
        qwen_quantize_bf16_to_q4_0(wb, rows, cols, wq);
        for (int k = 0; k < cols; k++) for (int b = 0; b < B; b++) X[(size_t)k * B + b] = RF;
        for (int k = 0; k < cols; k++) xb[k] = X[(size_t)k * B];

        /* reps scaled so each timed region is ~hundreds of ms */
        double mb = (double)rows * cols * 2 / (1024 * 1024);
        int reps = mb > 8 ? 8 : 24;

        fprintf(f, "  [%4dx%4d]  (%.1f MB bf16)\n", rows, cols, mb);
        for (int p = 0; p < 3; p++) {
            const char *pn = p == 0 ? "bf16" : p == 1 ? "int8" : "int4";
            /* warm */
            if (p == 0) { qwen_matvec_bf16(yc, wb, xb, rows, cols); qwen_matmat_bf16(Y, wb, X, rows, cols, B); }
            else if (p == 1) { qwen_matvec_int8(yc, wi, sc, xb, rows, cols); qwen_matmat_int8(Y, wi, sc, X, rows, cols, B); }
            else { qwen_matvec_q4_0(yc, wq, xb, rows, cols); qwen_matmat_q4_0(Y, wq, X, rows, cols, B); }

            NOW_S(t0);
            for (int it = 0; it < reps; it++)
                for (int b = 0; b < B; b++) {
                    for (int k = 0; k < cols; k++) xb[k] = X[(size_t)k * B + b];
                    if (p == 0) qwen_matvec_bf16(yc, wb, xb, rows, cols);
                    else if (p == 1) qwen_matvec_int8(yc, wi, sc, xb, rows, cols);
                    else qwen_matvec_q4_0(yc, wq, xb, rows, cols);
                }
            NOW_S(t1); double t_seq = MS(t0, t1) / reps;

            NOW_S(t0);
            for (int it = 0; it < reps; it++) {
                if (p == 0) qwen_matmat_bf16(Y, wb, X, rows, cols, B);
                else if (p == 1) qwen_matmat_int8(Y, wi, sc, X, rows, cols, B);
                else qwen_matmat_q4_0(Y, wq, X, rows, cols, B);
            }
            NOW_S(t1); double t_batch = MS(t0, t1) / reps;

            fprintf(f, "     %-5s  seq %7.2f ms   batch %7.2f ms   SPEEDUP %.2fx\n",
                    pn, t_seq, t_batch, t_seq / t_batch);
        }
        free(wb); free(wi); free(sc); free(wq); free(X); free(xb); free(Y); free(yc);
    }
    #undef RF
    #undef NOW_S
    #undef MS
    return 0;
}
