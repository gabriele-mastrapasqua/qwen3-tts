/* qwen_tts_kernels.c - Kernel implementations */

#include <pthread.h>
#if defined(__linux__)
#include <sys/prctl.h>
#endif
#include "qwen_tts_kernels.h"
#include "qwen_tts_kleidi.h"
#include "qwen_tts_q8repack.h"

#define MMSTAT(k, r, c, b) do {                                                            \
        if (qwen_matmat_stats_enabled() || qwen_census_enabled())                          \
            qwen_matmat_stats_note((k), (long long)(r) * (long long)(c) * (long long)(b)); \
    } while (0)

#include "qwen_tts_thread.h"
#include "qwen_tts_costmap.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
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
#if defined(__x86_64__)
#include <cpuid.h>
#endif
#if (defined(__AMX_INT8__) || defined(__AMX_BF16__)) && defined(__AMX_TILE__) && defined(__linux__)
#include <sys/syscall.h>
#endif

#if defined(__x86_64__) || defined(_M_X64)
static int qwen_x86_nchunk(int mmk, int B);
#endif

#ifdef USE_BLAS
#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif
#endif

void qwen_ftz_on(void) {
#if defined(__aarch64__)
    uint64_t fpcr;
    __asm__ volatile("mrs %0, fpcr" : "=r"(fpcr));
    if (!(fpcr & (1ULL << 24))) {
        fpcr |= (1ULL << 24);
        __asm__ volatile("msr fpcr, %0" : : "r"(fpcr));
    }
#elif defined(__x86_64__)
    unsigned int mxcsr = __builtin_ia32_stmxcsr();
    __builtin_ia32_ldmxcsr(mxcsr | 0x8040);
#endif
}

static int g_n_threads = 1;
#if defined(__GNUC__) && !defined(__APPLE__)
extern void openblas_set_num_threads(int) __attribute__((weak));
extern int  openblas_get_num_threads(void) __attribute__((weak));
#endif

static int g_blas_own = -1;   /* -1 = not decided: env, else the default set by the server */
int qwen_blas_own_get(void) {
    if (g_blas_own < 0) { const char *e = getenv("QWEN_BLAS_OWN"); g_blas_own = (e && e[0] == '1') ? 1 : 0; }
    return g_blas_own;
}
void qwen_blas_own(int on) {
    /* A default from the server; an explicit QWEN_BLAS_OWN in the environment wins. */
    if (!getenv("QWEN_BLAS_OWN")) g_blas_own = on ? 1 : 0;
    qwen_blas_set_threads(g_n_threads);
}
/* Owning the BLAS is a CLAIM; this is the fact.  qwen_blas_set_threads() below is compiled
 * out where the vendor BLAS has no thread control (Accelerate, or a build without the
 * OpenBLAS symbol), so there "own" never caps anything and slicing the SGEMM across the
 * engine pool would run on top of the BLAS's own team instead of replacing it. */
int qwen_blas_own_effective(void) {
#if defined(__GNUC__) && !defined(__APPLE__)
    /* An OPENBLAS_NUM_THREADS in the environment used to mean "we are not really in control",
     * which was true while qwen_blas_set_threads() returned early on seeing it.  It no longer
     * does: ownership overrides the variable and holds BLAS at one thread, so counting the env
     * as a loss of control made this report the opposite of what the engine was doing. */
    return qwen_blas_own_get() && openblas_set_num_threads != NULL;
#else
    return 0;
#endif
}
int qwen_blas_threads_now(void) {
#if defined(__GNUC__) && !defined(__APPLE__)
    return openblas_get_num_threads ? openblas_get_num_threads() : -1;
#else
    return -1;
#endif
}

/* Set when an OPENBLAS_NUM_THREADS in the environment was overridden because the engine owns
 * the budget, so --effective-config can say so instead of leaving it to be discovered. */
static atomic_int g_blas_env_overridden;
int qwen_blas_env_overridden(void) {
    return atomic_load_explicit(&g_blas_env_overridden, memory_order_relaxed);
}

void qwen_blas_set_threads(int n) {
#if defined(__GNUC__) && !defined(__APPLE__)
    const char *env = getenv("OPENBLAS_NUM_THREADS");
    if (qwen_blas_own_get()) {
        /* Engine ownership is structural, not advisory.  This used to RETURN when the variable
         * was present, so an exported OPENBLAS_NUM_THREADS silently left OpenBLAS with its own
         * compute team while the engine believed it owned the budget -- two schedulers in one
         * process, decided by whether someone remembered a variable.  The engine now wins and
         * the override is reported. */
        if (env && *env && atoi(env) != 1)
            atomic_store_explicit(&g_blas_env_overridden, 1, memory_order_relaxed);
        if (openblas_set_num_threads) openblas_set_num_threads(1);
        return;
    }
    if (env) return;                  /* not owned: an explicit control experiment may set it */
    if (openblas_set_num_threads) openblas_set_num_threads(n > 0 ? n : 1);
#else
    (void)n;
#endif
}

static int g_n_threads_hard = 0;

void qwen_set_threads(int n) {
    g_n_threads = n > 0 ? n : 1;
    g_n_threads_hard = g_n_threads;
    qwen_ftz_on();
    qwen_threadpool_start(g_n_threads);
    qwen_blas_set_threads(g_n_threads);
}
int qwen_get_threads(void) { return g_n_threads; }

void qwen_set_threads_soft(int n) {
    if (g_n_threads_hard <= 0) g_n_threads_hard = g_n_threads > 0 ? g_n_threads : 1;
    if (n <= 0) n = g_n_threads_hard;
    if (n > g_n_threads_hard) n = g_n_threads_hard;
    if (n == g_n_threads) return;
    g_n_threads = n;
    qwen_blas_set_threads(n);
}
int qwen_get_threads_hard(void) {
    return g_n_threads_hard > 0 ? g_n_threads_hard : g_n_threads;
}

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
    g_n_threads = ncpus < 4 ? ncpus : 4;
    g_n_threads_hard = g_n_threads;
    qwen_ftz_on();
    qwen_threadpool_start(g_n_threads);
    qwen_blas_set_threads(g_n_threads);
}

#if defined(__x86_64__)
static int qwen_x86_has_amx_int8(void) {
    unsigned eax, ebx, ecx, edx;
    if (!__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) return 0;
    return ((edx >> 24) & 1u) && ((edx >> 25) & 1u);
}
static int qwen_x86_has_amx_bf16(void) {
    unsigned eax, ebx, ecx, edx;
    if (!__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) return 0;
    return ((edx >> 24) & 1u) && ((edx >> 22) & 1u);
}
#endif

#if (defined(__AMX_INT8__) || defined(__AMX_BF16__)) && defined(__AMX_TILE__)
#define QWEN_ARCH_GET_XCOMP_PERM 0x1022
#define QWEN_ARCH_REQ_XCOMP_PERM 0x1023
#define QWEN_XFEATURE_XTILEDATA  18
#if defined(__linux__) && !defined(SYS_arch_prctl)
#define SYS_arch_prctl 158
#endif
static int qwen_amx_perm_ok(void) {
    static atomic_int perm_state = -1;
    int st = atomic_load_explicit(&perm_state, memory_order_relaxed);
    if (st >= 0) return st;
    st = 0;
#if defined(__linux__)
    if (syscall(SYS_arch_prctl, QWEN_ARCH_REQ_XCOMP_PERM, QWEN_XFEATURE_XTILEDATA) == 0) {
        unsigned long bits = 0;
        if (syscall(SYS_arch_prctl, QWEN_ARCH_GET_XCOMP_PERM, &bits) == 0 &&
            (bits & (1UL << QWEN_XFEATURE_XTILEDATA)) != 0)
            st = 1;
    }
#else
    st = 0;
#endif
    atomic_store_explicit(&perm_state, st, memory_order_relaxed);
    return st;
}
#endif

#if defined(__AMX_INT8__) && defined(__AMX_TILE__)
static int qwen_amx_int8_ready(void) {
    static atomic_int amx_state = -1;
    int st = atomic_load_explicit(&amx_state, memory_order_relaxed);
    if (st >= 0) return st;
    st = qwen_x86_has_amx_int8() && qwen_amx_perm_ok();
    atomic_store_explicit(&amx_state, st, memory_order_relaxed);
    return st;
}
#endif

#if defined(__AMX_BF16__) && defined(__AMX_TILE__)
static int qwen_amx_bf16_ready(void) {
    static atomic_int amx_state = -1;
    int st = atomic_load_explicit(&amx_state, memory_order_relaxed);
    if (st >= 0) return st;
    st = qwen_x86_has_amx_bf16() && qwen_amx_perm_ok();
    atomic_store_explicit(&amx_state, st, memory_order_relaxed);
    return st;
}
#endif

int qwen_amx_bf16_available(void) {
#if defined(__AMX_BF16__) && defined(__AMX_TILE__)
    return qwen_amx_bf16_ready();
#else
    return 0;
#endif
}

int qwen_amx_int8_available(void) {
#if defined(__AMX_INT8__) && defined(__AMX_TILE__)
    return qwen_amx_int8_ready();
#else
    return 0;
#endif
}

#if !defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC)
int qwen_arm_bfdot_on(void) { return 0; }
#endif

int qwen_arm_bf16_matmat_available(void) {
#if defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC) && !defined(__APPLE__)
    const char *e = getenv("QWEN_NO_BFMMLA");
    return !(e && e[0] == '1');
#else
    return 0;
#endif
}

/* AVX-512 BF16 without AMX still has a real batched bf16 matmat
 * (QWEN_MMK_BF16_AVX512, VDPBF16PS).  Without this the prefill default fell back
 * to converting every weight matrix to f32 and calling SGEMM, which is ~3x the
 * memory traffic on the TTFA critical path. */
int qwen_avx512_bf16_matmat_available(void) {
#if defined(__AVX512BF16__)
    if (!__builtin_cpu_supports("avx512bf16")) return 0;
    const char *e = getenv("QWEN_NO_BF16_MATMUL");
    return !(e && e[0] == '1');
#else
    return 0;
#endif
}

#include "qwen_build_id.h"
#ifndef QWEN_GIT_REV
#define QWEN_GIT_REV QWEN_BUILD_GIT_REV
#endif
#ifndef QWEN_SOURCE_FP
#define QWEN_SOURCE_FP QWEN_BUILD_SOURCE_FP
#endif
#ifndef QWEN_SIMD_PROFILE
#define QWEN_SIMD_PROFILE "unknown"
#endif

static const char *const g_qwen_reported_flags[] = {
    /* kernel dispatch — which GEMM, dot or conv actually runs */
    "QWEN_NO_AMX", "QWEN_NO_AMX_BF16", "QWEN_NO_AMX_INT8", "QWEN_NO_AMX_Q4", "QWEN_NO_VNNI",
    "QWEN_NO_VNNI_TILE", "QWEN_NO_VNNI_QKV", "QWEN_NO_X86_QKV", "QWEN_NO_VNNI_ROWSUM",
    "QWEN_NO_VNNI_ACT_QUANT", "QWEN_VNNI_PREPACK", "QWEN_VNNI_TILE_N8",
    "QWEN_VNNI_TILE_M4N2", "QWEN_VNNI_GEMV_MR", "QWEN_VNNI_UACT",
    "QWEN_PREFILL_ROWPACK", "QWEN_PREFILL_QKV_SHARE",
    "QWEN_AMX_PERSIST_CFG", "QWEN_AMX_PREPACK", "QWEN_AMX_PREPACK_KINDS", "QWEN_AMX_B32", "QWEN_NO_AVX2MM", "QWEN_NO_BF16DOT",
    "QWEN_NO_BF16_MATMUL", "QWEN_NO_SDOT", "QWEN_NO_SMMLA", "QWEN_NO_BFMMLA", "QWEN_ARM_BFDOT",
    "QWEN_APPLE_MMLA", "QWEN_INT8_SDOT_MM", "QWEN_Q4_NAIVE", "QWEN_Q4_VNNI_V3", "QWEN_Q4_VNNI_V4",
    "QWEN_Q6_SCALAR", "QWEN_Q8_SCALAR_ACT", "QWEN_NO_SIMD_QUANT", "QWEN_NO_Q8REPACK", "QWEN_NO_SIN_POLY",
    /* kernel gates and tiling — when a kernel may run, and how it tiles */
    "QWEN_AMX_MIN_B", "QWEN_AMX_BF16_MIN_B", "QWEN_AMX_INT8_MIN_B", "QWEN_AMX_MIN_ROWS",
    "QWEN_AMX_BF16_MIN_COLS", "QWEN_AMX_INT8_MIN_COLS", "QWEN_AMX_Q4_MIN_COLS",
    "QWEN_AMX_INT8_QKV_MIN_B", "QWEN_AMX_INT8_MIN_ROWS_PER_THREAD", "QWEN_VNNI_MIN_B",
    "QWEN_AVX2MM_MIN_B", "QWEN_BF16_MATMUL_MIN_B", "QWEN_BFMMLA_MIN_B", "QWEN_SMMLA_MIN_B",
    "QWEN_INT8_SDOT_MIN_B", "QWEN_KLEIDI_MIN_B", "QWEN_X86_NCHUNK", "QWEN_AMX_NCHUNK",
    "QWEN_VNNI_NCHUNK", "QWEN_AVX512_NCHUNK", "QWEN_KAI_NCHUNK",
    /* ARM KleidiAI */
    "QWEN_NO_KLEIDI", "QWEN_NO_KAI_BF16", "QWEN_NO_KAI_I8", "QWEN_KAI_OPS", "QWEN_KAI_QKV_FUSED",
    "QWEN_KAI_LHS", "QWEN_KAI_REPEAT",
    /* prefill */
    "QWEN_PREFILL_MATMAT", "QWEN_PREFILL_QUANT", "QWEN_PREFILL_HELPER", "QWEN_GGUF_QUANT_PREFILL",
    "QWEN_PREFIX_CACHE",
    /* code predictor */
    "QWEN_CP_PREC", "QWEN_CP_LAYER_PREC", "QWEN_CP_LMHEAD_PREC", "QWEN_CP_PREFILL2",
    "QWEN_CP_Q2_FFN",
    /* speech decoder and streaming */
    "QWEN_SD_INT8", "QWEN_SD_AMX", "QWEN_SD_INT8_BLK", "QWEN_SD_CONV_NC", "QWEN_SD_THREADS", "QWEN_SD_WINDOWED", "QWEN_SD_PHASE",
    "QWEN_SD_POOL", "QWEN_BLAS_OWN", "QWEN_SD_SGEMM_CENSUS", "QWEN_PREFILL_LOW_MS", "QWEN_POOL_HI_WINDOW_US", "QWEN_CP_REGION", "QWEN_CP_BATCH_HEAD", "QWEN_CP_FRAME_REGION", "QWEN_TK_REGION", "QWEN_PREFILL_INT8MM", "QWEN_PREFILL_CHUNK", "QWEN_SD_SCRATCH_STATS",
    "QWEN_STREAM_DECODE_CHUNK", "QWEN_STREAM_DECODE_CHUNK_BUSY", "QWEN_DECODER_BATCH",
    "QWEN_DECODER_THREAD", "QWEN_DECODER_GANG_LEAD", "QWEN_DECODER_GANG_MIN",
    "QWEN_DEC_FIRSTCHUNK_GROUP", "QWEN_SERVER_NO_DECODER_BATCH",
    /* server, admission and request batching */
    "QWEN_ADMIT_M1", "QWEN_SERVE_BLAS", "QWEN_SERVE_BLAS_BUSY", "QWEN_SERVER_STRICT",
    "QWEN_QUEUE_PREFILL", "QWEN_QUEUE_UNBOUNDED", "QWEN_MAX_REQUEST_S", "QWEN_MAX_TEXT_CHARS",
    "QWEN_CANCEL_ON_DISCONNECT", "QWEN_PREFORK_ELASTIC", "QWEN_TTFA_PRIORITY",
    "QWEN_TTFA_PRIO_STRICT", "QWEN_TTFA_FREEZE_CAP", "QWEN_BATCH_B", "QWEN_BATCH_SEQ",
    "QWEN_BATCH_TALKER", "QWEN_BATCH_DECODER", "QWEN_BATCH_FORCE_MATVEC", "QWEN_BATCH_MAX_FRAMES",
    "QWEN_BATCH_MAX_PROMPT", "QWEN_BATCH_NO_SOLO", "QWEN_BATCH_NO_BEFF", "QWEN_BATCH_NOMATMUL",
    /* threading, pool and memory */
    "QWEN_POOL_SPIN", "QWEN_POOL_NARROW", "QWEN_THREADS_TALKER", "QWEN_THREADS_DECODER",
    "QWEN_BLAS_GEN_THREADS", "QWEN_THP", "QWEN_NO_OVERLAP", "QWEN_NO_AMORT", "QWEN_AMORT_CPU",
    "QWEN_NO_PREWARM", "QWEN_FREE_BF16",
    /* precision, voice and conditioning */
    "QWEN_TALKER_PREC", "QWEN_TALKER_MIXED_INT6", "QWEN_ICL_FRAMES", "QWEN_ICL_TRIM_FRAMES",
    "QWEN_NO_REF_TRIM", "QWEN_SPK_SCALE", "QWEN_GRAFT_NO_WOVR", "QWEN_ACT_MAP",
    "QWEN_FFN_SPARSITY", "QWEN_TF_CODES", "QWEN_TF_PREFIX", "QWEN_TF_CB_KEEP",
    /* GPU backends */
    "QWEN_CUDA_BATCH", "QWEN_CUDA_CONVDEC", "QWEN_CUDA_DECODER", "QWEN_CUDA_FUSED_TALKER",
    "QWEN_CUDA_DP4A", "QWEN_DEC_NAIVE7", "QWEN_DEC_NAIVET",
    "QWEN_METAL_BATCH", "QWEN_METAL_BATCH_NOCP", "QWEN_METAL_BATCH_MMA", "QWEN_METAL_CP_NOSYNC",
    "QWEN_METAL_CP_PERPASS", "QWEN_METAL_FUSED_TALKER", "QWEN_METAL_PROFILE", "QWEN_METAL_Q4_VEC",
    /* diagnostics — never in a run that produces a number */
    "QWEN_BATCH_STATS", "QWEN_SHAPE_CENSUS", "QWEN_SERVE_PROFILE", "QWEN_TTFA_TRACE",
    "QWEN_LIFE_TRACE", "QWEN_REQ_TRACE", "QWEN_KERNEL_TIMING", "QWEN_VNNI_PHASE_TIMING", "QWEN_DUMP_CODE0", "QWEN_DUMP_CODES", "QWEN_EXPR_DEBUG",
    "QWEN_SD_DEBUG", "QWEN_SPK_DEBUG", "QWEN_TUNE_JSON", "QWEN_TUNE_QUICK",
    "QWEN_DISPATCH_MAP", "QWEN_DISPATCH_JSON", "QWEN_CENSUS_JSON",
    "QWEN_COST_MAP", "QWEN_COSTMAP_JSON",
    NULL
};

void qwen_provenance_report(void *out) {
    FILE *f = out ? (FILE *)out : stderr;
    fprintf(f, "  build:            %s · SIMD=%s · src=%s · %s %s\n",
            QWEN_GIT_REV, QWEN_SIMD_PROFILE, QWEN_SOURCE_FP, __DATE__, __TIME__);
    int n = 0;
    for (int i = 0; g_qwen_reported_flags[i]; i++) {
        const char *v = getenv(g_qwen_reported_flags[i]);
        if (!v) continue;
        if (n == 0) fprintf(f, "  active flags:     ");
        else if (n % 3 == 0) fprintf(f, "\n                    ");
        fprintf(f, "%s=%s  ", g_qwen_reported_flags[i], v);
        n++;
    }
    if (n == 0) fprintf(f, "  active flags:     none (every default of this build)\n");
    else fprintf(f, "\n");

    fprintf(f, "[FLAGS] v=1 pid=%d", (int)getpid());
    for (int i = 0; g_qwen_reported_flags[i]; i++) {
        const char *v = getenv(g_qwen_reported_flags[i]);
        if (v) fprintf(f, " %s=%s", g_qwen_reported_flags[i], v);
    }
    fprintf(f, "\n");
    fflush(f);
}

void qwen_caps_report(void *out) {
    FILE *f = out ? (FILE *)out : stderr;
    fprintf(f, "qwen-tts compiled capabilities:\n");
    qwen_provenance_report(f);
#if defined(__aarch64__)
    fprintf(f, "  arch:             arm64\n");
#elif defined(__x86_64__)
    fprintf(f, "  arch:             x86-64\n");
#else
    fprintf(f, "  arch:             (other)\n");
#endif
#ifdef __ARM_NEON
    fprintf(f, "  matvec + attn:    NEON (2-row fused)\n");
#elif defined(__AVX512F__)
    fprintf(f, "  matvec + attn:    AVX-512 (2-row fused, FMA, 16-wide attention)\n");
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
#if defined(__AVX512VNNI__)
    fprintf(f, "  VNNI GEMM tile:   MR=4/NR=4 for B=2..4; MR=2/NR=4 for compact B=5..8\n");
#endif
#if defined(__AVX512BF16__)
    fprintf(f, "  bf16 dot:         VDPBF16PS _mm512_dpbf16_ps (native; QWEN_NO_BF16DOT=1 disables)\n");
    fprintf(f, "  bf16 matmat:      AVX-512 BF16 tiles (QWEN_NO_BF16_MATMUL=1 disables)\n");
#elif defined(__x86_64__)
    fprintf(f, "  bf16 dot:         widen->FMA (no AVX-512-BF16)\n");
#endif
#if defined(__AVX512F__)
    fprintf(f, "  rms/bf16-conv:    AVX-512\n");
#elif defined(__AVX2__)
    fprintf(f, "  rms/bf16-conv:    AVX2\n");
#elif defined(__ARM_NEON)
    fprintf(f, "  rms/bf16-conv:    NEON\n");
#else
    fprintf(f, "  rms/bf16-conv:    scalar\n");
#endif
#if defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC)
# if defined(__APPLE__)
    fprintf(f, "  arm bf16 matmul:  BFMMLA compiled, default OFF on Apple (M4-measured loss on bandwidth-rich cores; QWEN_APPLE_MMLA=1 re-enables)\n");
# else
    fprintf(f, "  arm bf16 matmul:  BFMMLA ACTIVE (native bf16 GEMM batched matmat; QWEN_NO_BFMMLA=1 disables)\n");
# endif
#elif defined(__ARM_FEATURE_BF16)
    fprintf(f, "  arm bf16 matmul:  bfdot available (BFMMLA twin needs +bf16 vector arithmetic)\n");
#endif
#if defined(__ARM_FEATURE_MATMUL_INT8)
# if defined(__APPLE__)
    fprintf(f, "  arm i8mm:         q4-SMMLA ACTIVE; int8-SMMLA default OFF on Apple (M4-measured loss; QWEN_APPLE_MMLA=1 re-enables)\n");
# else
    fprintf(f, "  arm i8mm:         SMMLA ACTIVE (native int8 GEMM batched matmat; QWEN_NO_SMMLA=1 disables)\n");
# endif
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
#if defined(__x86_64__)
    __builtin_cpu_init();
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
    /* The lever is a property of THIS BINARY, not of the CPU: a SIMD=avx512 or SIMD=portable
     * build running on a VNNI host has no VNNI dot to recommend, and used to advertise one. */
    {
#if defined(__AVX512VNNI__)
        const int vnni_built = 1;
#else
        const int vnni_built = 0;
#endif
        const int vnni_cpu = __builtin_cpu_supports("avx512vnni") ? 1 : 0;
        fprintf(f, "  lever (x86):      %s\n",
                (vnni_built && vnni_cpu)
                  ? "VNNI int8 dot (native) — int8/int4 + batching is the throughput play"
              : (vnni_cpu && !vnni_built)
                  ? "this CPU has VNNI but this build does NOT — rebuild SIMD=avx512vnni "
                    "(or avx512bf16/amx) to get the native int8 dot"
              : __builtin_cpu_supports("avx2")
                  ? "AVX2 only (no VNNI) — int8 via widen+FMA; bandwidth-bound, batching helps"
                  : "no AVX2 — scalar; rebuild SIMD=scalar");
    }
#if defined(__AMX_INT8__) && defined(__AMX_TILE__)
    fprintf(f, "  x86 amx int8:     %s\n",
            qwen_amx_int8_ready()
              ? "AMX ACTIVE (tile 16x64 int8 GEMM for batched matmat; QWEN_NO_AMX=1 disables)"
              : (qwen_x86_has_amx_int8()
                   ? "compiled, XTILEDATA permission DENIED (needs Linux >= 5.16) -> VNNI fallback"
                   : "compiled, but this CPU has no AMX-INT8 -> VNNI fallback"));
#else
    if (qwen_x86_has_amx_int8())
        fprintf(f, "  x86 amx int8:     DETECTED on this CPU but not compiled in "
                   "(rebuild with -march=sapphirerapids)\n");
#endif
#if defined(__AMX_BF16__) && defined(__AMX_TILE__)
    fprintf(f, "  x86 amx bf16:     %s\n",
            qwen_amx_bf16_ready()
              ? "AMX ACTIVE (tile 16x32 bf16 GEMM for batched matmat; QWEN_NO_AMX=1 disables)"
              : (qwen_x86_has_amx_bf16()
                   ? "compiled, XTILEDATA permission DENIED (needs Linux >= 5.16) -> fixed-B twin"
                   : "compiled, but this CPU has no AMX-BF16 -> fixed-B twin"));
#else
    if (qwen_x86_has_amx_bf16())
        fprintf(f, "  x86 amx bf16:     DETECTED on this CPU but not compiled in "
                   "(rebuild with -march=sapphirerapids)\n");
#endif
#if defined(__AVX2__)
    if (!__builtin_cpu_supports("avx2"))
        fprintf(f, "  WARNING: built with AVX2 but this CPU lacks it -> will SIGILL. "
                   "Rebuild with `make blas SIMD=scalar`.\n");
#endif
#elif defined(__aarch64__)
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
            has_bf16 ? "bf16 BFMMLA -> native GEMM batched matmat twins (Graviton3-measured: int8 batch 2.1x, bf16 1.5x)"
                     : "no bf16 matmul (M1-class) -> batched matmat uses scalar bf16 decode");
    if (!has_bf16 && !has_i8mm)
        fprintf(f, "  note:             M1-class (Armv8.5, dotprod only). M2/M3/M4/M5 add bf16+i8mm -> the native-matmul lever.\n");
#if !defined(__ARM_FEATURE_MATMUL_INT8)
    if (has_i8mm)
        fprintf(f, "  arm i8mm:         DETECTED on this CPU but not compiled in "
                   "(rebuild with -march=native, or -march=armv8.6-a+i8mm)\n");
#endif
#if !defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC)
    if (has_bf16)
        fprintf(f, "  arm bf16 matmul:  DETECTED on this CPU but not compiled in "
                   "(rebuild with -march=native, or -march=armv8.6-a+bf16)\n");
#endif
#endif
    qwen_kernel_selection_report(f, 0, 0);
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

void qwen_rms_norm(float *out, const float *x, const float *weight,
                   int seq, int dim, float eps) {
    for (int s = 0; s < seq; s++) {
        const float *xs = x + s * dim;
        float *os = out + s * dim;

#ifdef __ARM_NEON
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
#elif defined(__AVX512F__)
        __m512 vsum0 = _mm512_setzero_ps(), vsum1 = _mm512_setzero_ps();
        int i = 0;
        for (; i + 31 < dim; i += 32) {
            __m512 v0 = _mm512_loadu_ps(xs + i);
            __m512 v1 = _mm512_loadu_ps(xs + i + 16);
            vsum0 = _mm512_fmadd_ps(v0, v0, vsum0);
            vsum1 = _mm512_fmadd_ps(v1, v1, vsum1);
        }
        float sum = _mm512_reduce_add_ps(_mm512_add_ps(vsum0, vsum1));
        for (; i < dim; i++) sum += xs[i] * xs[i];

        float inv_rms = 1.0f / sqrtf(sum / dim + eps);
        __m512 vinv = _mm512_set1_ps(inv_rms);
        i = 0;
        for (; i + 31 < dim; i += 32) {
            __m512 v0 = _mm512_loadu_ps(xs + i);
            __m512 v1 = _mm512_loadu_ps(xs + i + 16);
            __m512 w0 = _mm512_loadu_ps(weight + i);
            __m512 w1 = _mm512_loadu_ps(weight + i + 16);
            _mm512_storeu_ps(os + i,      _mm512_mul_ps(_mm512_mul_ps(v0, vinv), w0));
            _mm512_storeu_ps(os + i + 16, _mm512_mul_ps(_mm512_mul_ps(v1, vinv), w1));
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
#ifdef __ARM_NEON
    float32x4_t vsum0 = vdupq_n_f32(0), vsum1 = vdupq_n_f32(0);
    int i = 0;
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
#elif defined(__AVX512F__)
    __m512 vsum0 = _mm512_setzero_ps(), vsum1 = _mm512_setzero_ps();
    int i = 0;
    for (; i + 31 < dim; i += 32) {
        __m512 x0 = _mm512_add_ps(_mm512_loadu_ps(x + i),      _mm512_loadu_ps(residual + i));
        __m512 x1 = _mm512_add_ps(_mm512_loadu_ps(x + i + 16), _mm512_loadu_ps(residual + i + 16));
        _mm512_storeu_ps(x + i, x0);
        _mm512_storeu_ps(x + i + 16, x1);
        vsum0 = _mm512_fmadd_ps(x0, x0, vsum0);
        vsum1 = _mm512_fmadd_ps(x1, x1, vsum1);
    }
    float sum = _mm512_reduce_add_ps(_mm512_add_ps(vsum0, vsum1));
    for (; i < dim; i++) { x[i] += residual[i]; sum += x[i] * x[i]; }

    float inv_rms = 1.0f / sqrtf(sum / dim + eps);
    __m512 vinv = _mm512_set1_ps(inv_rms);
    i = 0;
    for (; i + 31 < dim; i += 32) {
        __m512 v0 = _mm512_loadu_ps(x + i);
        __m512 v1 = _mm512_loadu_ps(x + i + 16);
        __m512 w0 = _mm512_loadu_ps(weight + i);
        __m512 w1 = _mm512_loadu_ps(weight + i + 16);
        _mm512_storeu_ps(out + i,      _mm512_mul_ps(_mm512_mul_ps(v0, vinv), w0));
        _mm512_storeu_ps(out + i + 16, _mm512_mul_ps(_mm512_mul_ps(v1, vinv), w1));
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
#elif defined(__AVX512F__)
            __m512 vsum0 = _mm512_setzero_ps(), vsum1 = _mm512_setzero_ps();
            int i = 0;
            for (; i + 31 < head_dim; i += 32) {
                __m512 v0 = _mm512_loadu_ps(hs + i);
                __m512 v1 = _mm512_loadu_ps(hs + i + 16);
                vsum0 = _mm512_fmadd_ps(v0, v0, vsum0);
                vsum1 = _mm512_fmadd_ps(v1, v1, vsum1);
            }
            float sum = _mm512_reduce_add_ps(_mm512_add_ps(vsum0, vsum1));
            for (; i < head_dim; i++) sum += hs[i] * hs[i];

            float inv_rms = 1.0f / sqrtf(sum / head_dim + eps);
            __m512 vinv = _mm512_set1_ps(inv_rms);
            i = 0;
            for (; i + 31 < head_dim; i += 32) {
                __m512 v0 = _mm512_loadu_ps(hs + i);
                __m512 v1 = _mm512_loadu_ps(hs + i + 16);
                __m512 w0 = _mm512_loadu_ps(weight + i);
                __m512 w1 = _mm512_loadu_ps(weight + i + 16);
                _mm512_storeu_ps(hs + i,      _mm512_mul_ps(_mm512_mul_ps(v0, vinv), w0));
                _mm512_storeu_ps(hs + i + 16, _mm512_mul_ps(_mm512_mul_ps(v1, vinv), w1));
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

static inline float bf16_to_f32(uint16_t bf) {
    uint32_t bits = (uint32_t)bf << 16;
    float val;
    memcpy(&val, &bits, sizeof(float));
    return val;
}

#if defined(__AVX2__)
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
static inline __m256 qwen_loadu_bf16_8(const uint16_t *p) {
    __m128i b = _mm_loadu_si128((const __m128i *)p);
    return _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(b), 16));
}
static inline __m256 qwen_loadu_s8_8(const int8_t *p) {
    return _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *)p)));
}
#if defined(__AVX512F__)
static inline __m512 qwen_loadu_bf16_16(const uint16_t *p) {
    __m256i b = _mm256_loadu_si256((const __m256i *)p);
    return _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(b), 16));
}
#endif
static inline float qwen_dot_f32_avx2(const float *a, const float *b, int n) {
#if defined(__AVX512F__)
    __m512 c0 = _mm512_setzero_ps(), c1 = _mm512_setzero_ps(),
           c2 = _mm512_setzero_ps(), c3 = _mm512_setzero_ps();
    int d = 0;
    for (; d + 64 <= n; d += 64) {
        c0 = _mm512_fmadd_ps(_mm512_loadu_ps(a + d),      _mm512_loadu_ps(b + d),      c0);
        c1 = _mm512_fmadd_ps(_mm512_loadu_ps(a + d + 16), _mm512_loadu_ps(b + d + 16), c1);
        c2 = _mm512_fmadd_ps(_mm512_loadu_ps(a + d + 32), _mm512_loadu_ps(b + d + 32), c2);
        c3 = _mm512_fmadd_ps(_mm512_loadu_ps(a + d + 48), _mm512_loadu_ps(b + d + 48), c3);
    }
    for (; d + 16 <= n; d += 16)
        c0 = _mm512_fmadd_ps(_mm512_loadu_ps(a + d), _mm512_loadu_ps(b + d), c0);
    float s = _mm512_reduce_add_ps(_mm512_add_ps(_mm512_add_ps(c0, c2), _mm512_add_ps(c1, c3)));
    for (; d < n; d++) s += a[d] * b[d];
    return s;
#else
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
#endif
}
static inline float qwen_dot_f32_bf16_avx2(const float *q, const uint16_t *k, int n) {
#if defined(__AVX512F__)
    __m512 c0 = _mm512_setzero_ps(), c1 = _mm512_setzero_ps(),
           c2 = _mm512_setzero_ps(), c3 = _mm512_setzero_ps();
    int d = 0;
    for (; d + 64 <= n; d += 64) {
        c0 = _mm512_fmadd_ps(_mm512_loadu_ps(q + d),      qwen_loadu_bf16_16(k + d),      c0);
        c1 = _mm512_fmadd_ps(_mm512_loadu_ps(q + d + 16), qwen_loadu_bf16_16(k + d + 16), c1);
        c2 = _mm512_fmadd_ps(_mm512_loadu_ps(q + d + 32), qwen_loadu_bf16_16(k + d + 32), c2);
        c3 = _mm512_fmadd_ps(_mm512_loadu_ps(q + d + 48), qwen_loadu_bf16_16(k + d + 48), c3);
    }
    for (; d + 16 <= n; d += 16)
        c0 = _mm512_fmadd_ps(_mm512_loadu_ps(q + d), qwen_loadu_bf16_16(k + d), c0);
    float s = _mm512_reduce_add_ps(_mm512_add_ps(_mm512_add_ps(c0, c2), _mm512_add_ps(c1, c3)));
    for (; d < n; d++) s += q[d] * bf16_to_f32(k[d]);
    return s;
#else
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
#endif
}
static inline void qwen_acc_corr_avx2(float *o, const float *v, float c, int n) {
    int d = 0;
#if defined(__AVX512F__)
    __m512 zc = _mm512_set1_ps(c);
    for (; d + 16 <= n; d += 16)
        _mm512_storeu_ps(o + d, _mm512_fmadd_ps(_mm512_loadu_ps(o + d), zc, _mm512_loadu_ps(v + d)));
#else
    __m256 vc = _mm256_set1_ps(c);
    for (; d + 8 <= n; d += 8)
        _mm256_storeu_ps(o + d, _mm256_fmadd_ps(_mm256_loadu_ps(o + d), vc, _mm256_loadu_ps(v + d)));
#endif
    for (; d < n; d++) o[d] = o[d] * c + v[d];
}
static inline void qwen_acc_wt_avx2(float *o, const float *v, float w, int n) {
    int d = 0;
#if defined(__AVX512F__)
    __m512 zw = _mm512_set1_ps(w);
    for (; d + 16 <= n; d += 16)
        _mm512_storeu_ps(o + d, _mm512_fmadd_ps(_mm512_loadu_ps(v + d), zw, _mm512_loadu_ps(o + d)));
#else
    __m256 vw = _mm256_set1_ps(w);
    for (; d + 8 <= n; d += 8)
        _mm256_storeu_ps(o + d, _mm256_fmadd_ps(_mm256_loadu_ps(v + d), vw, _mm256_loadu_ps(o + d)));
#endif
    for (; d < n; d++) o[d] += v[d] * w;
}
static inline void qwen_scale_avx2(float *o, float s, int n) {
    int d = 0;
#if defined(__AVX512F__)
    __m512 zs = _mm512_set1_ps(s);
    for (; d + 16 <= n; d += 16)
        _mm512_storeu_ps(o + d, _mm512_mul_ps(_mm512_loadu_ps(o + d), zs));
#else
    __m256 vs = _mm256_set1_ps(s);
    for (; d + 8 <= n; d += 8)
        _mm256_storeu_ps(o + d, _mm256_mul_ps(_mm256_loadu_ps(o + d), vs));
#endif
    for (; d < n; d++) o[d] *= s;
}
static inline void qwen_acc_corr_bf16_avx2(float *o, const uint16_t *v, float c, int n) {
    int d = 0;
#if defined(__AVX512F__)
    __m512 zc = _mm512_set1_ps(c);
    for (; d + 16 <= n; d += 16)
        _mm512_storeu_ps(o + d, _mm512_fmadd_ps(_mm512_loadu_ps(o + d), zc, qwen_loadu_bf16_16(v + d)));
#else
    __m256 vc = _mm256_set1_ps(c);
    for (; d + 8 <= n; d += 8)
        _mm256_storeu_ps(o + d, _mm256_fmadd_ps(_mm256_loadu_ps(o + d), vc, qwen_loadu_bf16_8(v + d)));
#endif
    for (; d < n; d++) o[d] = o[d] * c + bf16_to_f32(v[d]);
}
static inline void qwen_acc_wt_bf16_avx2(float *o, const uint16_t *v, float w, int n) {
    int d = 0;
#if defined(__AVX512F__)
    __m512 zw = _mm512_set1_ps(w);
    for (; d + 16 <= n; d += 16)
        _mm512_storeu_ps(o + d, _mm512_fmadd_ps(qwen_loadu_bf16_16(v + d), zw, _mm512_loadu_ps(o + d)));
#else
    __m256 vw = _mm256_set1_ps(w);
    for (; d + 8 <= n; d += 8)
        _mm256_storeu_ps(o + d, _mm256_fmadd_ps(qwen_loadu_bf16_8(v + d), vw, _mm256_loadu_ps(o + d)));
#endif
    for (; d < n; d++) o[d] += bf16_to_f32(v[d]) * w;
}
#endif

#if defined(__AVX512BF16__)
enum { QWEN_BF16DOT_XMAX = 8192 };
/* A1: opt-in row-major activation entry for the AVX-512 bf16 matmat.
 * The generic qwen_matmat_bf16() takes X as [cols][B] because AMX and the ARM
 * kernels want that layout; the AVX-512 branch then immediately builds
 * Xb[b][k] = bf16(X[k][b]), i.e. it undoes the transpose.  A caller that
 * already holds row-major activations therefore pays a transpose and an
 * inverse transpose to produce a plain contiguous convert.  Default off. */
static int qwen_prefill_rowpack_enabled(void) {
    static atomic_int on = -1;
    int v = atomic_load_explicit(&on, memory_order_relaxed);
    if (v < 0) {
        const char *e = getenv("QWEN_PREFILL_ROWPACK");
        v = (e && e[0] == '1');
        atomic_store_explicit(&on, v, memory_order_relaxed);
    }
    return v;
}

static int qwen_bf16dot_disabled(void) {
    static atomic_int off = -1;
    int v = atomic_load_explicit(&off, memory_order_relaxed);
    if (v < 0) { const char *e = getenv("QWEN_NO_BF16DOT"); v = (e && e[0] == '1'); atomic_store_explicit(&off, v, memory_order_relaxed); }
    return v;
}
int qwen_bf16dot_enabled(void) { return !qwen_bf16dot_disabled(); }
static inline __m512bh qwen_loadu_pbh(const uint16_t *p) {
    union { __m512i i; __m512bh bh; } u;
    u.i = _mm512_loadu_si512((const void *)p);
    return u.bh;
}
static void qwen_f32_to_bf16_row(uint16_t *dst, const float *src, int n) {
    int k = 0;
    for (; k + 16 <= n; k += 16) {
        union { __m256bh bh; __m256i i; } u;
        u.bh = _mm512_cvtneps_pbh(_mm512_loadu_ps(src + k));
        _mm256_storeu_si256((__m256i *)(dst + k), u.i);
    }
    for (; k < n; k++) {
        uint32_t bits; memcpy(&bits, &src[k], 4);
        uint32_t lsb = (bits >> 16) & 1;
        dst[k] = (uint16_t)((bits + 0x7FFFu + lsb) >> 16);
    }
}
static void bf16_matvec_dpbf16(float *y, const uint16_t *xb, const float *x,
                               const uint16_t *W, int in_dim, int out_dim) {
    int o = 0;
    for (; o + 1 < out_dim; o += 2) {
        const uint16_t *w0 = W + (size_t)o * in_dim;
        const uint16_t *w1 = W + (size_t)(o + 1) * in_dim;
        if (o + 5 < out_dim) {
            __builtin_prefetch(W + (size_t)(o + 4) * in_dim, 0, 0);
            __builtin_prefetch(W + (size_t)(o + 5) * in_dim, 0, 0);
        }
        __m512 a0 = _mm512_setzero_ps(), a1 = _mm512_setzero_ps();
        __m512 b0 = _mm512_setzero_ps(), b1 = _mm512_setzero_ps();
        int k = 0;
        for (; k + 64 <= in_dim; k += 64) {
            __m512bh x0 = qwen_loadu_pbh(xb + k);
            __m512bh x1 = qwen_loadu_pbh(xb + k + 32);
            a0 = _mm512_dpbf16_ps(a0, qwen_loadu_pbh(w0 + k),      x0);
            a1 = _mm512_dpbf16_ps(a1, qwen_loadu_pbh(w0 + k + 32), x1);
            b0 = _mm512_dpbf16_ps(b0, qwen_loadu_pbh(w1 + k),      x0);
            b1 = _mm512_dpbf16_ps(b1, qwen_loadu_pbh(w1 + k + 32), x1);
        }
        for (; k + 32 <= in_dim; k += 32) {
            __m512bh xv = qwen_loadu_pbh(xb + k);
            a0 = _mm512_dpbf16_ps(a0, qwen_loadu_pbh(w0 + k), xv);
            b0 = _mm512_dpbf16_ps(b0, qwen_loadu_pbh(w1 + k), xv);
        }
        float s0 = _mm512_reduce_add_ps(_mm512_add_ps(a0, a1));
        float s1 = _mm512_reduce_add_ps(_mm512_add_ps(b0, b1));
        for (; k < in_dim; k++) { s0 += bf16_to_f32(w0[k]) * x[k]; s1 += bf16_to_f32(w1[k]) * x[k]; }
        y[o] = s0;
        y[o + 1] = s1;
    }
    if (o < out_dim) {
        const uint16_t *w_row = W + (size_t)o * in_dim;
        __m512 acc = _mm512_setzero_ps();
        int k = 0;
        for (; k + 32 <= in_dim; k += 32)
            acc = _mm512_dpbf16_ps(acc, qwen_loadu_pbh(w_row + k), qwen_loadu_pbh(xb + k));
        float sum = _mm512_reduce_add_ps(acc);
        for (; k < in_dim; k++) sum += bf16_to_f32(w_row[k]) * x[k];
        y[o] = sum;
    }
}
#endif

#if defined(__AVX512BF16__)
static inline uint16_t qwen_f32_to_bf16_scalar(float x) {
    uint32_t bits;
    memcpy(&bits, &x, sizeof bits);
    return (uint16_t)((bits + 0x7FFFu + ((bits >> 16) & 1u)) >> 16);
}

static void bf16_matmat_avx512_m4(float *Y, const uint16_t *W, const uint16_t *Xb,
                                  int r, int cols, int B, int ldy) {
    __m512 acc[4][4];
    for (int m = 0; m < 4; m++)
        for (int b = 0; b < B; b++) acc[m][b] = _mm512_setzero_ps();
    int k = 0, kfull = cols & ~31;
    for (; k < kfull; k += 32) {
        __m512bh w0 = qwen_loadu_pbh(W + (size_t)(r + 0) * cols + k);
        __m512bh w1 = qwen_loadu_pbh(W + (size_t)(r + 1) * cols + k);
        __m512bh w2 = qwen_loadu_pbh(W + (size_t)(r + 2) * cols + k);
        __m512bh w3 = qwen_loadu_pbh(W + (size_t)(r + 3) * cols + k);
        for (int b = 0; b < B; b++) {
            __m512bh x = qwen_loadu_pbh(Xb + (size_t)b * cols + k);
            acc[0][b] = _mm512_dpbf16_ps(acc[0][b], w0, x);
            acc[1][b] = _mm512_dpbf16_ps(acc[1][b], w1, x);
            acc[2][b] = _mm512_dpbf16_ps(acc[2][b], w2, x);
            acc[3][b] = _mm512_dpbf16_ps(acc[3][b], w3, x);
        }
    }
    for (int m = 0; m < 4; m++) {
        const uint16_t *w = W + (size_t)(r + m) * cols;
        float *y = Y + (size_t)(r + m) * ldy;
        for (int b = 0; b < B; b++) {
            float s = _mm512_reduce_add_ps(acc[m][b]);
            const uint16_t *x = Xb + (size_t)b * cols;
            for (int i = kfull; i < cols; i++) s += bf16_to_f32(w[i]) * bf16_to_f32(x[i]);
            y[b] = s;
        }
    }
}

static void bf16_matmat_avx512_m2(float *Y, const uint16_t *W, const uint16_t *Xb,
                                  int r, int cols, int B, int ldy) {
    __m512 acc[2][8];
    for (int m = 0; m < 2; m++)
        for (int b = 0; b < B; b++) acc[m][b] = _mm512_setzero_ps();
    int k = 0, kfull = cols & ~31;
    for (; k < kfull; k += 32) {
        __m512bh w0 = qwen_loadu_pbh(W + (size_t)(r + 0) * cols + k);
        __m512bh w1 = qwen_loadu_pbh(W + (size_t)(r + 1) * cols + k);
        for (int b = 0; b < B; b++) {
            __m512bh x = qwen_loadu_pbh(Xb + (size_t)b * cols + k);
            acc[0][b] = _mm512_dpbf16_ps(acc[0][b], w0, x);
            acc[1][b] = _mm512_dpbf16_ps(acc[1][b], w1, x);
        }
    }
    for (int m = 0; m < 2; m++) {
        const uint16_t *w = W + (size_t)(r + m) * cols;
        float *y = Y + (size_t)(r + m) * ldy;
        for (int b = 0; b < B; b++) {
            float s = _mm512_reduce_add_ps(acc[m][b]);
            const uint16_t *x = Xb + (size_t)b * cols;
            for (int i = kfull; i < cols; i++) s += bf16_to_f32(w[i]) * bf16_to_f32(x[i]);
            y[b] = s;
        }
    }
}

static void bf16_matmat_avx512_m1(float *Y, const uint16_t *W, const uint16_t *Xb,
                                  int r, int cols, int B, int ldy) {
    __m512 acc[16];
    for (int b = 0; b < B; b++) acc[b] = _mm512_setzero_ps();
    const uint16_t *w = W + (size_t)r * cols;
    int k = 0, kfull = cols & ~31;
    for (; k < kfull; k += 32) {
        __m512bh wv = qwen_loadu_pbh(w + k);
        for (int b = 0; b < B; b++)
            acc[b] = _mm512_dpbf16_ps(acc[b], wv,
                                      qwen_loadu_pbh(Xb + (size_t)b * cols + k));
    }
    float *y = Y + (size_t)r * ldy;
    for (int b = 0; b < B; b++) {
        float s = _mm512_reduce_add_ps(acc[b]);
        const uint16_t *x = Xb + (size_t)b * cols;
        for (int i = kfull; i < cols; i++) s += bf16_to_f32(w[i]) * bf16_to_f32(x[i]);
        y[b] = s;
    }
}

static void bf16_matmat_avx512_slice(float *Y, const uint16_t *W, const uint16_t *Xb,
                                     int r0, int r1, int cols, int B) {
    MMSTAT(QWEN_MMK_BF16_AVX512, r1 - r0, cols, B);
    int r = r0;
    if (B > 16) {
        /* Wider than the register tile: sweep each weight row once and run the 16-column
         * kernel per column block, so the row leaves DRAM once for all B columns.  Every
         * (row, column) accumulates in the same k order as a 16-wide call, so the result
         * is bit-identical to running the blocks as separate matmats. */
        for (; r < r1; r++)
            for (int b0 = 0; b0 < B; b0 += 16) {
                int nb = B - b0 < 16 ? B - b0 : 16;
                bf16_matmat_avx512_m1(Y + b0, W, Xb + (size_t)b0 * cols, r, cols, nb, B);
            }
        return;
    }
    if (B <= 4) {
        for (; r + 3 < r1; r += 4) bf16_matmat_avx512_m4(Y, W, Xb, r, cols, B, B);
    } else if (B <= 8) {
        for (; r + 1 < r1; r += 2) bf16_matmat_avx512_m2(Y, W, Xb, r, cols, B, B);
    }
    for (; r < r1; r++) bf16_matmat_avx512_m1(Y, W, Xb, r, cols, B, B);
}

typedef struct {
    float *Y; const uint16_t *W; const uint16_t *Xb; int rows, cols, B;
} bf16_avx512_ctx;
static void bf16_avx512_task(size_t tid, size_t nt, void *vc) {
    bf16_avx512_ctx *c = (bf16_avx512_ctx *)vc;
    int r0 = (int)(tid * (size_t)c->rows / nt);
    int r1 = (int)((tid + 1) * (size_t)c->rows / nt);
    int chunk = qwen_x86_nchunk(QWEN_MMK_BF16_AVX512, c->B);
    if (chunk > 0) {
        for (int r = r0; r < r1; r += chunk) {
            int e = r + chunk < r1 ? r + chunk : r1;
            bf16_matmat_avx512_slice(c->Y, c->W, c->Xb, r, e, c->cols, c->B);
        }
    } else {
        bf16_matmat_avx512_slice(c->Y, c->W, c->Xb, r0, r1, c->cols, c->B);
    }
}
#endif

#if defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC)
enum { QWEN_ARM_BFDOT_XMAX = 8192 };
static void qwen_arm_f32_to_bf16_row(uint16_t *dst, const float *src, int n) {
    for (int k = 0; k < n; k++) {
        uint32_t u; memcpy(&u, &src[k], sizeof u);
        dst[k] = (uint16_t)(u >> 16);
    }
}
static int qwen_arm_bfdot_enabled(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *e = getenv("QWEN_ARM_BFDOT");
        cached = (e && e[0] == '1') ? 1 : 0;
    }
    return cached;
}
int qwen_arm_bfdot_on(void) { return qwen_arm_bfdot_enabled(); }
static void bf16_matvec_bfdot(float *y, const uint16_t *xb, const uint16_t *W,
                              int in_dim, int out_dim) {
    const bfloat16_t *xv = (const bfloat16_t *)xb;
    int o = 0;
    for (; o + 3 < out_dim; o += 4) {
        const bfloat16_t *w0 = (const bfloat16_t *)(W + (size_t)o * in_dim);
        const bfloat16_t *w1 = (const bfloat16_t *)(W + (size_t)(o + 1) * in_dim);
        const bfloat16_t *w2 = (const bfloat16_t *)(W + (size_t)(o + 2) * in_dim);
        const bfloat16_t *w3 = (const bfloat16_t *)(W + (size_t)(o + 3) * in_dim);
        float32x4_t a0 = vdupq_n_f32(0), a1 = vdupq_n_f32(0);
        float32x4_t a2 = vdupq_n_f32(0), a3 = vdupq_n_f32(0);
        int k = 0;
        for (; k + 7 < in_dim; k += 8) {
            bfloat16x8_t xk = vld1q_bf16(xv + k);
            a0 = vbfdotq_f32(a0, vld1q_bf16(w0 + k), xk);
            a1 = vbfdotq_f32(a1, vld1q_bf16(w1 + k), xk);
            a2 = vbfdotq_f32(a2, vld1q_bf16(w2 + k), xk);
            a3 = vbfdotq_f32(a3, vld1q_bf16(w3 + k), xk);
        }
        float s0 = vaddvq_f32(a0), s1 = vaddvq_f32(a1);
        float s2 = vaddvq_f32(a2), s3 = vaddvq_f32(a3);
        for (; k < in_dim; k++) {
            float xf = bf16_to_f32(xb[k]);
            s0 += bf16_to_f32(W[(size_t)o * in_dim + k])       * xf;
            s1 += bf16_to_f32(W[(size_t)(o + 1) * in_dim + k]) * xf;
            s2 += bf16_to_f32(W[(size_t)(o + 2) * in_dim + k]) * xf;
            s3 += bf16_to_f32(W[(size_t)(o + 3) * in_dim + k]) * xf;
        }
        y[o] = s0; y[o + 1] = s1; y[o + 2] = s2; y[o + 3] = s3;
    }
    for (; o < out_dim; o++) {
        const bfloat16_t *w0 = (const bfloat16_t *)(W + (size_t)o * in_dim);
        float32x4_t a0 = vdupq_n_f32(0);
        int k = 0;
        for (; k + 7 < in_dim; k += 8)
            a0 = vbfdotq_f32(a0, vld1q_bf16(w0 + k), vld1q_bf16(xv + k));
        float s0 = vaddvq_f32(a0);
        for (; k < in_dim; k++)
            s0 += bf16_to_f32(W[(size_t)o * in_dim + k]) * bf16_to_f32(xb[k]);
        y[o] = s0;
    }
}
#endif

static void bf16_matvec_fused(float *y, const float *x, const uint16_t *W,
                               int in_dim, int out_dim) {
    int o = 0;
#if defined(__AVX512BF16__)
    if (!qwen_bf16dot_disabled() && in_dim <= QWEN_BF16DOT_XMAX) {
        qwen_census_leaf(QWEN_LEAF_DPBF16);
        uint16_t xb[QWEN_BF16DOT_XMAX];
        qwen_f32_to_bf16_row(xb, x, in_dim);
        bf16_matvec_dpbf16(y, xb, x, W, in_dim, out_dim);
        return;
    }
#endif
#if defined(__AVX512F__)
    qwen_census_leaf(QWEN_LEAF_AVX512F);
#elif defined(__ARM_NEON)
    qwen_census_leaf(QWEN_LEAF_NEON);
#elif defined(__AVX2__)
    qwen_census_leaf(QWEN_LEAF_AVX2);
#else
    qwen_census_leaf(QWEN_LEAF_SCALAR);
#endif
#if defined(__AVX512F__)
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
#  if defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC)
    if (qwen_arm_bfdot_enabled() && in_dim <= QWEN_ARM_BFDOT_XMAX) {
        uint16_t xb[QWEN_ARM_BFDOT_XMAX];
        qwen_arm_f32_to_bf16_row(xb, x, in_dim);
        bf16_matvec_bfdot(y, xb, W, in_dim, out_dim);
        return;
    }
#  endif
    for (; o + 1 < out_dim; o += 2) {
        const uint16_t *w0 = W + (size_t)o * in_dim;
        const uint16_t *w1 = W + (size_t)(o + 1) * in_dim;
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
    for (; o + 1 < out_dim; o += 2) {
        const uint16_t *w0 = W + (size_t)o * in_dim;
        const uint16_t *w1 = W + (size_t)(o + 1) * in_dim;
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
    for (; o < out_dim; o++) {
        const uint16_t *row = W + (size_t)o * in_dim;
        float sum = 0.0f;
        for (int k = 0; k < in_dim; k++) sum += bf16_to_f32(row[k]) * x[k];
        y[o] = sum;
    }
#endif
}

static int kai_i8_try(float *Y, const int8_t *W, const float *scale, const float *X,
                      int rows, int cols, int B) {
    if (!qwen_kleidi_i8_enabled()) return 0;
    if (qwen_kleidi_matmul_i8(Y, W, X, rows, cols, B)) return 1;
    if (!qwen_kleidi_register_i8(W, W, scale, rows, cols)) return 0;
    return qwen_kleidi_matmul_i8(Y, W, X, rows, cols, B);
}
static int kai_bf16_try(float *Y, const uint16_t *W, const float *X,
                        int rows, int cols, int B) {
    if (!qwen_kleidi_bf16_enabled()) return 0;
    if (qwen_kleidi_matmul_bf16(Y, W, X, rows, cols, B)) return 1;
    if (!qwen_kleidi_register_bf16(W, W, rows, cols)) return 0;
    return qwen_kleidi_matmul_bf16(Y, W, X, rows, cols, B);
}

typedef struct {
    float *y; const uint16_t *W; const float *x; int rows, cols;
} bf16_mv_ctx;
static void bf16_mv_task(size_t tid, size_t nt, void *vc) {
    bf16_mv_ctx *c = (bf16_mv_ctx *)vc;
    int r0 = (int)(tid * (size_t)c->rows / nt);
    int r1 = (int)((tid + 1) * (size_t)c->rows / nt);
    bf16_matvec_fused(c->y + r0, c->x, c->W + (size_t)r0 * c->cols, c->cols, r1 - r0);
}
void (*g_qwen_matvec_bf16_hook)(float *, const uint16_t *, const float *, int, int) = NULL;

/* Forward declarations for the optional complete-call timing census.  The
 * implementation lives with the other matmul counters below, while the BF16
 * matvec entry point is intentionally kept near its dispatch helpers. */
enum { QWEN_KT_INT8 = 0, QWEN_KT_BF16, QWEN_KT_KIND_COUNT };
enum { QWEN_KT_B1 = 0, QWEN_KT_B2, QWEN_KT_B34, QWEN_KT_B58, QWEN_KT_B9P, QWEN_KT_BUCKET_COUNT };
static double qwen_mm_now_s(void);
static int qwen_kernel_timing_enabled(void);
static void qwen_kernel_timing_note(int kind, int B, int rows, int cols,
                                    double start_s);
static void qwen_vnni_phase_report(FILE *out);

void qwen_matvec_bf16(float *y, const uint16_t *W, const float *x, int rows, int cols) {
    qwen_census_op(QWEN_PATH_MATVEC_BF16, rows, cols, 1);
    const int kt_on = qwen_kernel_timing_enabled();
    const double kt_t0 = kt_on ? qwen_mm_now_s() : 0.0;
    if (qwen_q8r_matmul(y, (const void *)W, x, rows, cols, 1)) {
        MMSTAT(QWEN_MMK_Q8_REPACK_GEMV, rows, cols, 1);
        goto qwen_matvec_bf16_timed_done;
    }
    if (kai_bf16_try(y, W, x, rows, cols, 1)) {
        MMSTAT(QWEN_MMK_KLEIDI_BF16_GEMV, rows, cols, 1);
        goto qwen_matvec_bf16_timed_done;
    }
    MMSTAT(QWEN_MMK_BF16_GEMV, rows, cols, 1);

    if (g_qwen_matvec_bf16_hook) {
        g_qwen_matvec_bf16_hook(y, W, x, rows, cols);
        goto qwen_matvec_bf16_timed_done;
    }
    int nt = g_n_threads;
    if (nt > 1 && rows >= 256) {
        bf16_mv_ctx c = { y, W, x, rows, cols };
        qwen_parallel((size_t)nt, bf16_mv_task, &c);
        goto qwen_matvec_bf16_timed_done;
    }
    bf16_matvec_fused(y, x, W, cols, rows);

qwen_matvec_bf16_timed_done:
    qwen_kernel_timing_note(QWEN_KT_BF16, 1, rows, cols, kt_t0);
}

static atomic_int g_mm_stats = -1;
static atomic_llong g_mm_macs[QWEN_MMK_COUNT];
/* Which arm of the fixed-width switch a twin actually took. A dispatcher that never
   reaches the twin at all leaves both at zero, which is the answer to "does this kernel
   change anything on this build" - and it is not visible from the per-kernel MAC table,
   because that is recorded before the width switch. */
static atomic_llong g_mm_fixedw[2];   /* [0] bf16, [1] int8 */
static atomic_llong g_mm_generic[2];
static atomic_llong g_mm_calls[QWEN_MMK_COUNT];
static atomic_llong g_mm_wbytes;
static _Atomic double g_mm_wb_t0, g_mm_wb_t1;
static double qwen_mm_now_s(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

/*
 * Diagnostic wall-time census for the complete public matvec/matmat calls.
 * This deliberately measures the caller-visible interval, including activation
 * preparation, dispatch, worker-pool wait, dot product, scaling and stores. It
 * is not a kernel benchmark and must never be enabled for a timing claim.
 * Weight bytes are the one-read lower bound (rows * cols * element_size), not
 * a hardware counter or a claim that the implementation made one DRAM pass.
 */
typedef struct {
    atomic_llong calls;
    atomic_llong wall_ns;
    atomic_llong macs;
    atomic_llong weight_bytes;
} qwen_kernel_timing_cell_t;
static qwen_kernel_timing_cell_t
    g_kernel_timing[QWEN_COMP_COUNT][QWEN_KT_KIND_COUNT][QWEN_KT_BUCKET_COUNT];
static atomic_int g_kernel_timing_on = -1;

static void qwen_kernel_timing_atexit(void);

static int qwen_kernel_timing_enabled(void) {
    int v = atomic_load_explicit(&g_kernel_timing_on, memory_order_relaxed);
    if (v < 0) {
        const char *e = getenv("QWEN_KERNEL_TIMING");
        v = e && e[0] && e[0] != '0';
        static atomic_int registered = 0;
        if (v && !atomic_exchange_explicit(&registered, 1, memory_order_relaxed))
            atexit(qwen_kernel_timing_atexit);
        atomic_store_explicit(&g_kernel_timing_on, v, memory_order_relaxed);
    }
    return v;
}

static int qwen_kernel_timing_bucket(int B) {
    if (B <= 1) return QWEN_KT_B1;
    if (B == 2) return QWEN_KT_B2;
    if (B <= 4) return QWEN_KT_B34;
    if (B <= 8) return QWEN_KT_B58;
    return QWEN_KT_B9P;
}

static void qwen_kernel_timing_note(int kind, int B, int rows, int cols,
                                    double start_s) {
    if (!start_s || rows <= 0 || cols <= 0 || kind < 0 || kind >= QWEN_KT_KIND_COUNT)
        return;
    const double elapsed = qwen_mm_now_s() - start_s;
    const long long ns = elapsed > 0.0 ? (long long)(elapsed * 1e9) : 0;
    int comp = qwen_tls_tag_get();
    if (comp < 0 || comp >= QWEN_COMP_COUNT) comp = QWEN_COMP_OTHER;
    qwen_kernel_timing_cell_t *cell =
        &g_kernel_timing[comp][kind][qwen_kernel_timing_bucket(B)];
    atomic_fetch_add_explicit(&cell->calls, 1, memory_order_relaxed);
    atomic_fetch_add_explicit(&cell->wall_ns, ns, memory_order_relaxed);
    atomic_fetch_add_explicit(&cell->macs,
                              (long long)rows * (long long)cols * (long long)(B > 0 ? B : 1),
                              memory_order_relaxed);
    atomic_fetch_add_explicit(&cell->weight_bytes,
                              (long long)rows * (long long)cols *
                              (long long)(kind == QWEN_KT_BF16 ? sizeof(uint16_t) : sizeof(int8_t)),
                              memory_order_relaxed);
}

void qwen_kernel_timing_report(void *out) {
    static const char *const cname[QWEN_COMP_COUNT] = { "other", "talker", "cp", "decoder" };
    static const char *const kname[QWEN_KT_KIND_COUNT] = { "int8", "bf16" };
    static const char *const bname[QWEN_KT_BUCKET_COUNT] = { "B1", "B2", "B3-4", "B5-8", "B9+" };
    FILE *f = out ? (FILE *)out : stderr;
    int any = 0;
    for (int c = 0; c < QWEN_COMP_COUNT; c++)
        for (int k = 0; k < QWEN_KT_KIND_COUNT; k++)
            for (int b = 0; b < QWEN_KT_BUCKET_COUNT; b++)
                if (atomic_load_explicit(&g_kernel_timing[c][k][b].calls,
                                         memory_order_relaxed) > 0) any = 1;
    if (any) {
        fprintf(f, "\n[kernel-timing] v=1 pid=%d complete public matvec/matmat wall time\n",
                (int)getpid());
        fprintf(f, "# csv: component,kind,bucket,calls,wall_ms,wall_ms_per_call,gmac,"
                        "nominal_weight_gb,nominal_weight_gbps\n");
        for (int c = 0; c < QWEN_COMP_COUNT; c++)
            for (int k = 0; k < QWEN_KT_KIND_COUNT; k++)
                for (int b = 0; b < QWEN_KT_BUCKET_COUNT; b++) {
                    qwen_kernel_timing_cell_t *cell = &g_kernel_timing[c][k][b];
                    long long calls = atomic_load_explicit(&cell->calls, memory_order_relaxed);
                    if (!calls) continue;
                    long long ns = atomic_load_explicit(&cell->wall_ns, memory_order_relaxed);
                    long long macs = atomic_load_explicit(&cell->macs, memory_order_relaxed);
                    long long bytes = atomic_load_explicit(&cell->weight_bytes, memory_order_relaxed);
                    double sec = (double)ns / 1e9;
                    double gb = (double)bytes / 1e9;
                    fprintf(f, "timing,%s,%s,%s,%lld,%.3f,%.3f,%.6f,%.6f,%.3f\n",
                            cname[c], kname[k], bname[b], calls, (double)ns / 1e6,
                            calls ? (double)ns / 1e6 / (double)calls : 0.0,
                            (double)macs / 1e9, gb, sec > 0.0 ? gb / sec : 0.0);
                }
        fprintf(f, "[kernel-timing] nominal_weight_gbps is a one-read lower-bound rate; "
                        "it excludes non-weight traffic and does not measure DRAM bytes.\n");
    }
    qwen_vnni_phase_report(f);
    fflush(f);
}
static void qwen_kernel_timing_atexit(void) { qwen_kernel_timing_report(NULL); }

/*
 * Optional phase census for the AVX-512 VNNI B=1 path.  Unlike the public
 * kernel-timing census, this is deliberately a diagnostic: phase times are
 * aggregated across worker tasks and therefore are not a critical-path
 * attribution.  It is disabled unless QWEN_VNNI_PHASE_TIMING=1.
 */
enum {
    QWEN_VP_QUANT = 0,
    QWEN_VP_ROWSUM,
    QWEN_VP_DOT,
    QWEN_VP_EPILOGUE,
    QWEN_VP_PARALLEL,
    QWEN_VP_COUNT
};
typedef struct {
    atomic_llong calls;
    atomic_llong wall_ns;
    atomic_llong work_bytes;
} qwen_vnni_phase_cell_t;
static qwen_vnni_phase_cell_t
    g_vnni_phase[QWEN_COMP_COUNT][QWEN_VP_COUNT];
static atomic_int g_vnni_phase_on = -1;

#if defined(__GNUC__) || defined(__clang__)
#define QWEN_VNNI_PHASE_UNUSED __attribute__((unused))
#else
#define QWEN_VNNI_PHASE_UNUSED
#endif

static int QWEN_VNNI_PHASE_UNUSED qwen_vnni_phase_timing_enabled(void) {
    int v = atomic_load_explicit(&g_vnni_phase_on, memory_order_relaxed);
    if (v < 0) {
        const char *e = getenv("QWEN_VNNI_PHASE_TIMING");
        v = e && e[0] && e[0] != '0';
        atomic_store_explicit(&g_vnni_phase_on, v, memory_order_relaxed);
    }
    return v;
}

static void QWEN_VNNI_PHASE_UNUSED qwen_vnni_phase_note(
    int phase, long long work_bytes, double start_s) {
    if (!start_s || phase < 0 || phase >= QWEN_VP_COUNT) return;
    const double elapsed = qwen_mm_now_s() - start_s;
    const long long ns = elapsed > 0.0 ? (long long)(elapsed * 1e9) : 0;
    int comp = qwen_tls_tag_get();
    if (comp < 0 || comp >= QWEN_COMP_COUNT) comp = QWEN_COMP_OTHER;
    qwen_vnni_phase_cell_t *cell = &g_vnni_phase[comp][phase];
    atomic_fetch_add_explicit(&cell->calls, 1, memory_order_relaxed);
    atomic_fetch_add_explicit(&cell->wall_ns, ns, memory_order_relaxed);
    if (work_bytes > 0)
        atomic_fetch_add_explicit(&cell->work_bytes, work_bytes, memory_order_relaxed);
}

static void qwen_vnni_phase_report(FILE *f) {
    static const char *const cname[QWEN_COMP_COUNT] = {
        "other", "talker", "cp", "decoder"
    };
    static const char *const pname[QWEN_VP_COUNT] = {
        "activation_quant", "row_sum_lookup_or_build", "vnni_dot",
        "scale_store_epilogue", "parallel_span"
    };
    int any = 0;
    for (int c = 0; c < QWEN_COMP_COUNT; c++)
        for (int p = 0; p < QWEN_VP_COUNT; p++)
            if (atomic_load_explicit(&g_vnni_phase[c][p].calls,
                                     memory_order_relaxed) > 0) any = 1;
    if (!any) return;
    fprintf(f, "\n[vnni-phase] v=1 pid=%d AVX-512 VNNI B=1 diagnostic\n",
            (int)getpid());
    fprintf(f, "# csv: component,phase,calls,wall_ms,wall_ms_per_call,work_gb,effective_gbps\n");
    for (int c = 0; c < QWEN_COMP_COUNT; c++)
        for (int p = 0; p < QWEN_VP_COUNT; p++) {
            qwen_vnni_phase_cell_t *cell = &g_vnni_phase[c][p];
            long long calls = atomic_load_explicit(&cell->calls, memory_order_relaxed);
            if (!calls) continue;
            long long ns = atomic_load_explicit(&cell->wall_ns, memory_order_relaxed);
            long long bytes = atomic_load_explicit(&cell->work_bytes, memory_order_relaxed);
            double sec = (double)ns / 1e9;
            double gb = (double)bytes / 1e9;
            fprintf(f, "phase,%s,%s,%lld,%.3f,%.6f,%.6f,%.3f\n",
                    cname[c], pname[p], calls, (double)ns / 1e6,
                    (double)ns / 1e6 / (double)calls, gb,
                    sec > 0.0 ? gb / sec : 0.0);
        }
    fprintf(f, "[vnni-phase] calls are timed output-group regions; aggregate wall is across calls/tasks, not a critical-path sum.\n");
    fprintf(f, "[vnni-phase] vnni_dot work_gb is the one-read INT8 weight byte count, not measured DRAM traffic.\n");
    fflush(f);
}
#undef QWEN_VNNI_PHASE_UNUSED

enum { MMC_GEMM = 0, MMC_TWIN, MMC_MATVEC, MMC_SOLO, MMC_GEMV, MMC_NCLS };
static const struct { const char *name; int cls; } g_mmk_info[QWEN_MMK_COUNT] = {
    { "(none)",                 MMC_TWIN   },
    { "bf16 BFMMLA (arm)",      MMC_GEMM   },
    { "bf16 AVX-512 dpbf16",    MMC_GEMM   },
    { "bf16 fixed-B twin",      MMC_TWIN   },
    { "bf16 generic twin",      MMC_TWIN   },
    { "int8 AMX tiles",         MMC_GEMM   },
    { "int8 VNNI vpdpbusd",     MMC_GEMM   },
    { "int8 AVX2 maddubs",      MMC_GEMM   },
    { "int8 SMMLA (i8mm)",      MMC_GEMM   },
    { "int8 SDOT loop over B",  MMC_MATVEC },
    { "int8 f32-accum twin",    MMC_TWIN   },
    { "q4   VNNI vpdpbusd",     MMC_GEMM   },
    { "q4   AVX2 maddubs",      MMC_GEMM   },
    { "q4   SMMLA (i8mm)",      MMC_GEMM   },
    { "q4   B x matvec",        MMC_MATVEC },
    { "q4   generic twin",      MMC_TWIN   },
    { "FORCED B x matvec",      MMC_MATVEC },
    { "solo (B_eff==1)",        MMC_SOLO   },
    { "bf16 AMX tiles",         MMC_GEMM   },
    { "q4   AMX tiles",         MMC_GEMM   },
    { "q4   KleidiAI (arm)",    MMC_GEMM   },
    { "bf16 GEMV",              MMC_GEMV   },
    { "int8 GEMV",              MMC_GEMV   },
    { "q4   GEMV",              MMC_GEMV   },
    { "int8 KleidiAI GEMM",     MMC_GEMM   },
    { "int8 KleidiAI GEMV",     MMC_GEMV   },
    { "bf16 KleidiAI GEMM",     MMC_GEMM   },
    { "bf16 KleidiAI GEMV",     MMC_GEMV   },
    { "q8_0 repack SMMLA",      MMC_GEMM   },
    { "q8_0 repack GEMV",       MMC_GEMV   },
};
_Static_assert(sizeof(g_mmk_info) / sizeof(g_mmk_info[0]) == QWEN_MMK_COUNT,
               "g_mmk_info[] is out of sync with the QWEN_MMK_* enum");
static const char *g_mmc_note[MMC_NCLS] = {
    "real matrix-matrix instruction",
    "batched, but no matrix instruction (shares the weight read only)",
    "<- B sequential matvecs: weights read B times, no sharing",
    "by design (one active slot)",
    "single-vector GEMV (B=1: one weight read, no sharing to be had)",
};

static void qwen_mm_stats_atexit(void);
int qwen_matmat_stats_enabled(void) {
    int v = atomic_load_explicit(&g_mm_stats, memory_order_relaxed);
    if (v < 0) {
        const char *e = getenv("QWEN_BATCH_STATS");
        v = (e && e[0] && e[0] != '0');
        static atomic_int reg = 0;
        if (v && !atomic_exchange_explicit(&reg, 1, memory_order_relaxed))
            atexit(qwen_mm_stats_atexit);
        atomic_store_explicit(&g_mm_stats, v, memory_order_relaxed);
    }
    return v;
}
void qwen_mm_component(int comp) {
    qwen_tls_tag_set((comp >= 0 && comp < QWEN_COMP_COUNT) ? comp : QWEN_COMP_OTHER);
}
int qwen_mm_component_get(void) { return qwen_tls_tag_get(); }

static atomic_llong g_mm_macs_c[QWEN_COMP_COUNT][QWEN_MMK_COUNT];
static atomic_llong g_mm_calls_c[QWEN_COMP_COUNT][QWEN_MMK_COUNT];

#define QWEN_CENSUS_MAX 256
typedef struct {
    int path;
    int comp, rows, cols, B;
    atomic_llong calls, macs;
    atomic_uint  kmask;     /* QWEN_MMK_* that ran (matmat dispatch) */
    /* kmask is an OR of every kernel this shape ever used, so it cannot say HOW MUCH ran on
     * each one: a row that took AMX once and VNNI a thousand times looks identical to a row
     * that is pure AMX.  A coverage share read off the mask is an upper bound, not a
     * measurement, so keep MACs and calls PER KERNEL as well. */
    atomic_llong kmacs[QWEN_MMK_COUNT];
    atomic_llong kcalls[QWEN_MMK_COUNT];
    atomic_uint  lmask;     /* QWEN_LEAF_* that ran (branch inside the entry) */
} qwen_census_row_t;
static qwen_census_row_t g_census[QWEN_CENSUS_MAX];
static atomic_int   g_census_n;
static atomic_int   g_census_on = -1;
static atomic_llong g_census_frames;
/* The row the CURRENT THREAD is attributing to.  Was one global: under the pool a
 * worker's kernel mask landed on whatever row another thread had just opened. */
static __thread qwen_census_row_t *t_census_cur;

static const struct { int id; const char *name; int kind; } g_path_info[] = {
    { QWEN_PATH_MATVEC_BF16, "matvec_bf16", QWEN_PATHK_CALL },
    { QWEN_PATH_MATVEC_BF16_QKV, "matvec_bf16_qkv", QWEN_PATHK_CALL },
    { QWEN_PATH_MATVEC_INT8, "matvec_int8", QWEN_PATHK_CALL },
    { QWEN_PATH_MATVEC_INT8_QKV, "matvec_int8_qkv", QWEN_PATHK_CALL },
    { QWEN_PATH_MATVEC_Q4_0, "matvec_q4_0", QWEN_PATHK_CALL },
    { QWEN_PATH_MATVEC_Q4_0_QKV, "matvec_q4_0_qkv", QWEN_PATHK_CALL },
    { QWEN_PATH_MATVEC_Q2_0, "matvec_q2_0", QWEN_PATHK_CALL },
    { QWEN_PATH_MATVEC_Q6_0, "matvec_q6_0", QWEN_PATHK_CALL },
    { QWEN_PATH_MATVEC_Q6_0_QKV, "matvec_q6_0_qkv", QWEN_PATHK_CALL },
    { QWEN_PATH_ARGMAX_MATVEC_BF16, "argmax_matvec_bf16", QWEN_PATHK_CALL },
    { QWEN_PATH_ARGMAX_MATVEC_INT8, "argmax_matvec_int8", QWEN_PATHK_WRAPPER },
    { QWEN_PATH_ARGMAX_MATVEC_Q4_0, "argmax_matvec_q4_0", QWEN_PATHK_WRAPPER },
    { QWEN_PATH_MATMAT_BF16, "matmat_bf16", QWEN_PATHK_CALL },
    { QWEN_PATH_MATMAT_BF16_ROWS, "matmat_bf16_rows", QWEN_PATHK_CALL },
    { QWEN_PATH_MATMAT_BF16_QKV, "matmat_bf16_qkv", QWEN_PATHK_CALL },
    { QWEN_PATH_MATMAT_INT8, "matmat_int8", QWEN_PATHK_CALL },
    { QWEN_PATH_MATMAT_INT8_QKV, "matmat_int8_qkv", QWEN_PATHK_CALL },
    { QWEN_PATH_MATMAT_Q4_0, "matmat_q4_0", QWEN_PATHK_CALL },
    { QWEN_PATH_MATMAT_INT8_VNNI_PACKED_SLICE, "matmat_int8_vnni_packed.slice", QWEN_PATHK_SLICE },
    { QWEN_PATH_MATMAT_INT8_VNNI_M4N2_SLICE, "matmat_int8_vnni_m4n2.slice", QWEN_PATHK_SLICE },
    { QWEN_PATH_PREFILL_BF16_NATIVE, "prefill_bf16_native", QWEN_PATHK_WRAPPER },
    { QWEN_PATH_PREFILL_F32_SGEMM, "prefill_f32_sgemm", QWEN_PATHK_CALL },
    { QWEN_PATH_BF16_ROWPACK_SHARED, "bf16_rowpack_shared", QWEN_PATHK_TRANSFORM },
    { QWEN_PATH_MATMAT_INT8_NATIVE, "matmat_int8_native", QWEN_PATHK_WRAPPER },
    { QWEN_PATH_MATMAT_BF16_NATIVE, "matmat_bf16_native", QWEN_PATHK_WRAPPER },
    { QWEN_PATH_MATMAT_INT8_QKV_NATIVE, "matmat_int8_qkv_native", QWEN_PATHK_WRAPPER },
    { QWEN_PATH_DECODER_SGEMM, "decoder_sgemm", QWEN_PATHK_CALL },
    { QWEN_PATH_DECODER_CONV_INT8, "decoder_conv_int8", QWEN_PATHK_CALL },
    { QWEN_PATH_DECODER_CONV_AMX_INT8, "decoder_conv_amx_int8", QWEN_PATHK_CALL },
    { QWEN_PATH_DECODER_CONV_NAIVE, "decoder_conv_naive", QWEN_PATHK_CALL },
};
const char *qwen_path_name(int path) {
    for (size_t i = 0; i < sizeof g_path_info / sizeof g_path_info[0]; i++)
        if (g_path_info[i].id == path) return g_path_info[i].name;
    return "unknown_path";
}
int qwen_path_kind(int path) {
    for (size_t i = 0; i < sizeof g_path_info / sizeof g_path_info[0]; i++)
        if (g_path_info[i].id == path) return g_path_info[i].kind;
    return QWEN_PATHK_CALL;
}
static const char *const g_leaf_name[QWEN_LEAF_COUNT] = {
    "none", "vnni", "dpbf16", "sdot", "avx512f", "avx2", "neon", "scalar", "blas",
    "f32_fused", "kleidi", "amx", "delegated"
};
const char *qwen_leaf_name(int leaf) {
    return (leaf > 0 && leaf < QWEN_LEAF_COUNT) ? g_leaf_name[leaf] : "none";
}
void qwen_census_leaf(int leaf) {
    if (leaf <= 0 || leaf >= QWEN_LEAF_COUNT) return;
    if (atomic_load_explicit(&g_census_on, memory_order_relaxed) <= 0) return;
    if (t_census_cur) atomic_fetch_or_explicit(&t_census_cur->lmask, 1u << leaf, memory_order_relaxed);
}
static pthread_mutex_t g_census_mu = PTHREAD_MUTEX_INITIALIZER;
static atomic_int   g_census_overflow;

static void qwen_census_atexit(void) { qwen_census_report(NULL); }

int qwen_census_enabled(void) {
    int v = atomic_load_explicit(&g_census_on, memory_order_relaxed);
    if (v < 0) {
        const char *e = getenv("QWEN_SHAPE_CENSUS");
        v = (e && e[0] && e[0] != '0');
        static atomic_int reg = 0;
        if (v && !atomic_exchange_explicit(&reg, 1, memory_order_relaxed))
            atexit(qwen_census_atexit);
        atomic_store_explicit(&g_census_on, v, memory_order_relaxed);
    }
    return v;
}

static atomic_llong g_census_frames_at[3];
void qwen_census_frame_at(int site) {
    if (!qwen_census_enabled()) return;
    if (site >= 0 && site < 3)
        atomic_fetch_add_explicit(&g_census_frames_at[site], 1, memory_order_relaxed);
    atomic_fetch_add_explicit(&g_census_frames, 1, memory_order_relaxed);
}
void qwen_census_frame(void) { qwen_census_frame_at(0); }

static void qwen_census_op_impl(int path, int rows, int cols, int B, long long macs);
void qwen_census_op(int path, int rows, int cols, int B) {
    qwen_census_op_impl(path, rows, cols, B, (long long)rows * (long long)cols * (long long)B);
}
void qwen_census_op_len(int path, int rows, int cols, int len) {
    if (len <= 0) return;
    int b = 1; while (b < len) b <<= 1;          /* key = next power of two of the length */
    qwen_census_op_impl(path, rows, cols, b, (long long)rows * (long long)cols * (long long)len);
}
static void qwen_census_op_impl(int path, int rows, int cols, int B, long long macs) {
    if (!qwen_census_enabled() || B <= 0) return;
    const int comp = qwen_tls_tag_get();
    const int n = atomic_load_explicit(&g_census_n, memory_order_acquire);
    qwen_census_row_t *hit = NULL;
    for (int i = 0; i < n; i++) {
        qwen_census_row_t *r = &g_census[i];
        if (r->rows == rows && r->cols == cols && r->B == B &&
            r->comp == comp && r->path == path) { hit = r; break; }
    }
    if (!hit) {
        pthread_mutex_lock(&g_census_mu);
        int m = atomic_load_explicit(&g_census_n, memory_order_relaxed);
        for (int i = n; i < m && !hit; i++) {
            qwen_census_row_t *r = &g_census[i];
            if (r->rows == rows && r->cols == cols && r->B == B &&
                r->comp == comp && r->path == path) hit = r;
        }
        if (!hit) {
            if (m >= QWEN_CENSUS_MAX) {
                atomic_fetch_add_explicit(&g_census_overflow, 1, memory_order_relaxed);
                pthread_mutex_unlock(&g_census_mu);
                return;
            }
            hit = &g_census[m];
            hit->path = path; hit->comp = comp;
            hit->rows = rows; hit->cols = cols; hit->B = B;
            atomic_store_explicit(&g_census_n, m + 1, memory_order_release);
        }
        pthread_mutex_unlock(&g_census_mu);
    }
    atomic_fetch_add_explicit(&hit->calls, 1, memory_order_relaxed);
    atomic_fetch_add_explicit(&hit->macs, macs, memory_order_relaxed);
    t_census_cur = hit;
}

void qwen_census_report(void *out) {
    FILE *f = out ? (FILE *)out : stderr;
    const int n = atomic_load_explicit(&g_census_n, memory_order_acquire);
    if (n <= 0) { fprintf(f, "\n[shape-census] nothing recorded\n"); return; }
    const long long frames = atomic_load_explicit(&g_census_frames, memory_order_relaxed);
    static const char *cname[QWEN_COMP_COUNT] = { "other", "talker", "cp", "decoder" };
    fprintf(f, "\n[shape-census] frames=%lld (single=%lld batched=%lld batched_slot=%lld) "
               "threads=%d  (N=out features, K=in features, B=batch columns)\n", frames,
            atomic_load_explicit(&g_census_frames_at[0], memory_order_relaxed),
            atomic_load_explicit(&g_census_frames_at[1], memory_order_relaxed),
            atomic_load_explicit(&g_census_frames_at[2], memory_order_relaxed),
            g_n_threads);
    fprintf(f, "# csv: comp,path,N,K,B,calls,calls_per_frame,gmac,gmac_per_frame,kernels,leaves,kind\n");
    /* Per-kernel split, emitted as its own record type: the "kernels" column above is an OR
     * mask and cannot carry a share. */
    fprintf(f, "# csv2: kcensus,comp,path,N,K,B,kernel,kernel_calls,kernel_gmac\n");
    for (int i = 0; i < n; i++) {
        qwen_census_row_t *r = &g_census[i];
        long long c = atomic_load_explicit(&r->calls, memory_order_relaxed);
        long long m = atomic_load_explicit(&r->macs,  memory_order_relaxed);
        unsigned km = atomic_load_explicit(&r->kmask, memory_order_relaxed);
        unsigned lm = atomic_load_explicit(&r->lmask, memory_order_relaxed);
        char kbuf[256]; kbuf[0] = 0;
        for (int k = 1; k < QWEN_MMK_COUNT; k++) {
            if (!(km & (1u << k))) continue;
            if (kbuf[0]) strncat(kbuf, "+", sizeof kbuf - strlen(kbuf) - 1);
            strncat(kbuf, g_mmk_info[k].name, sizeof kbuf - strlen(kbuf) - 1);
        }
        char lbuf[128]; lbuf[0] = 0;
        for (int k = 1; k < QWEN_LEAF_COUNT; k++) {
            if (!(lm & (1u << k))) continue;
            if (lbuf[0]) strncat(lbuf, "+", sizeof lbuf - strlen(lbuf) - 1);
            strncat(lbuf, g_leaf_name[k], sizeof lbuf - strlen(lbuf) - 1);
        }
        for (int k = 1; k < QWEN_MMK_COUNT; k++) {
            long long kc = atomic_load_explicit(&r->kcalls[k], memory_order_relaxed);
            if (!kc) continue;
            fprintf(f, "kcensus,%s,%s,%d,%d,%d,%s,%lld,%.4f\n",
                    cname[r->comp < 0 || r->comp >= QWEN_COMP_COUNT ? 0 : r->comp], qwen_path_name(r->path), r->rows, r->cols, r->B,
                    g_mmk_info[k].name, kc,
                    (double)atomic_load_explicit(&r->kmacs[k], memory_order_relaxed) / 1e9);
        }
        fprintf(f, "census,%s,%s,%d,%d,%d,%lld,%.3f,%.4f,%.6f,%s,%s,%s\n",
                cname[r->comp < 0 || r->comp >= QWEN_COMP_COUNT ? 0 : r->comp], qwen_path_name(r->path),
                r->rows, r->cols, r->B, c,
                frames ? (double)c / (double)frames : 0.0,
                (double)m / 1e9,
                frames ? (double)m / 1e9 / (double)frames : 0.0,
                kbuf[0] ? kbuf : "(none)", lbuf[0] ? lbuf : "(none)",
                qwen_path_kind(r->path) == QWEN_PATHK_CALL      ? "call"
              : qwen_path_kind(r->path) == QWEN_PATHK_SLICE     ? "slice"
              : qwen_path_kind(r->path) == QWEN_PATHK_WRAPPER   ? "wrapper" : "transform");
    }
#if defined(__AMX_INT8__) && defined(__AMX_TILE__)
    { void qwen_sd_amx_rej_report(void); qwen_sd_amx_rej_report(); }
#endif
    int ov = atomic_load_explicit(&g_census_overflow, memory_order_relaxed);
    if (ov) fprintf(f, "[shape-census] WARNING: %d ops dropped, table full (%d rows)\n",
                    ov, QWEN_CENSUS_MAX);
    fflush(f);

    /* Machine-readable twin: QWEN_CENSUS_JSON=path, "%d" -> pid (prefork workers each
       write their own).  Counters are cumulative, so the last dump is the whole run. */
    const char *jp = getenv("QWEN_CENSUS_JSON");
    if (jp && jp[0]) {
        char path[1024];
        const char *pct = strstr(jp, "%d");
        if (pct) snprintf(path, sizeof path, "%.*s%d%s", (int)(pct - jp), jp, (int)getpid(), pct + 2);
        else     snprintf(path, sizeof path, "%s", jp);
        FILE *j = fopen(path, "w");
        if (j) {
            fprintf(j, "{\n  \"v\": 1,\n  \"pid\": %d,\n  \"threads\": %d,\n  \"frames\": %lld,\n"
                       "  \"frames_single\": %lld,\n  \"frames_batched\": %lld,\n  \"frames_batched_slot\": %lld,\n"
                       "  \"dropped_ops\": %d,\n  \"rows\": [\n",
                    (int)getpid(), g_n_threads, frames,
                    atomic_load_explicit(&g_census_frames_at[0], memory_order_relaxed),
                    atomic_load_explicit(&g_census_frames_at[1], memory_order_relaxed),
                    atomic_load_explicit(&g_census_frames_at[2], memory_order_relaxed), ov);
            for (int i = 0; i < n; i++) {
                qwen_census_row_t *r = &g_census[i];
                long long c = atomic_load_explicit(&r->calls, memory_order_relaxed);
                long long m = atomic_load_explicit(&r->macs,  memory_order_relaxed);
                unsigned km = atomic_load_explicit(&r->kmask, memory_order_relaxed);
                unsigned lm = atomic_load_explicit(&r->lmask, memory_order_relaxed);
                fprintf(j, "    {\"path_id\": %d, \"path\": \"%s\", \"kind\": %d, \"comp\": \"%s\", "
                           "\"N\": %d, \"K\": %d, \"B\": %d, \"calls\": %lld, \"macs\": %lld, \"kernels\": [",
                        r->path, qwen_path_name(r->path), qwen_path_kind(r->path),
                        cname[r->comp < 0 || r->comp >= QWEN_COMP_COUNT ? 0 : r->comp],
                        r->rows, r->cols, r->B, c, m);
                int first = 1;
                for (int k = 1; k < QWEN_MMK_COUNT; k++)
                    if (km & (1u << k)) { fprintf(j, "%s\"%s\"", first ? "" : ", ", g_mmk_info[k].name); first = 0; }
                fprintf(j, "], \"leaves\": [");
                first = 1;
                for (int k = 1; k < QWEN_LEAF_COUNT; k++)
                    if (lm & (1u << k)) { fprintf(j, "%s\"%s\"", first ? "" : ", ", g_leaf_name[k]); first = 0; }
                fprintf(j, "]}%s\n", i + 1 < n ? "," : "");
            }
            fprintf(j, "  ]\n}\n");
            fclose(j);
        }
    }
}

void qwen_matmat_stats_note(int k, long long macs) {
    if (k <= 0 || k >= QWEN_MMK_COUNT) return;
    if (atomic_load_explicit(&g_census_on, memory_order_relaxed) > 0) {
        if (t_census_cur) {
            atomic_fetch_or_explicit(&t_census_cur->kmask, 1u << k, memory_order_relaxed);
            atomic_fetch_add_explicit(&t_census_cur->kmacs[k], macs, memory_order_relaxed);
            atomic_fetch_add_explicit(&t_census_cur->kcalls[k], 1, memory_order_relaxed);
        }
    }
    atomic_fetch_add_explicit(&g_mm_macs[k], macs, memory_order_relaxed);
    atomic_fetch_add_explicit(&g_mm_calls[k], 1, memory_order_relaxed);
    int c = qwen_tls_tag_get();
    if (c < 0 || c >= QWEN_COMP_COUNT) c = QWEN_COMP_OTHER;
    atomic_fetch_add_explicit(&g_mm_macs_c[c][k], macs, memory_order_relaxed);
    atomic_fetch_add_explicit(&g_mm_calls_c[c][k], 1, memory_order_relaxed);
}
void qwen_matmat_stats_note_bytes(long long weight_bytes) {
    if (weight_bytes <= 0) return;
    atomic_fetch_add_explicit(&g_mm_wbytes, weight_bytes, memory_order_relaxed);
    if (atomic_load_explicit(&g_mm_wb_t0, memory_order_relaxed) == 0.0) {
        double now = qwen_mm_now_s();
        double expect = 0.0;
        atomic_compare_exchange_strong_explicit(&g_mm_wb_t0, &expect, now,
                                                memory_order_relaxed, memory_order_relaxed);
    }
    atomic_store_explicit(&g_mm_wb_t1, qwen_mm_now_s(), memory_order_relaxed);
}
void qwen_matmat_stats_reset_components(void) {
    for (int c = 0; c < QWEN_COMP_COUNT; c++)
        for (int i = 0; i < QWEN_MMK_COUNT; i++) {
            atomic_store_explicit(&g_mm_macs_c[c][i], 0, memory_order_relaxed);
            atomic_store_explicit(&g_mm_calls_c[c][i], 0, memory_order_relaxed);
        }
}
void qwen_matmat_stats_reset(void) {
    for (int i = 0; i < QWEN_MMK_COUNT; i++) {
        atomic_store_explicit(&g_mm_macs[i], 0, memory_order_relaxed);
        atomic_store_explicit(&g_mm_calls[i], 0, memory_order_relaxed);
    }
    atomic_store_explicit(&g_mm_wbytes, 0, memory_order_relaxed);
    atomic_store_explicit(&g_mm_wb_t0, 0.0, memory_order_relaxed);
    atomic_store_explicit(&g_mm_wb_t1, 0.0, memory_order_relaxed);
}
void qwen_matmat_stats_report(void *out) {
    FILE *f = out ? (FILE *)out : stderr;
    long long tot = 0, by_cls[MMC_NCLS] = { 0 };
    for (int i = 1; i < QWEN_MMK_COUNT; i++) {
        long long m = atomic_load_explicit(&g_mm_macs[i], memory_order_relaxed);
        tot += m; by_cls[g_mmk_info[i].cls] += m;
    }
    if (tot == 0) { fprintf(f, "\n[batch-audit] no batched projection ran\n"); return; }
    fprintf(f, "\n[batch-audit] which kernel did the batched projections actually use\n");
    fprintf(f, "  %-24s %12s %10s %8s  %s\n", "kernel", "GMAC", "calls", "share", "");
    for (int i = 1; i < QWEN_MMK_COUNT; i++) {
        long long m = atomic_load_explicit(&g_mm_macs[i], memory_order_relaxed);
        if (!m) continue;
        fprintf(f, "  %-24s %12.2f %10lld %7.1f%%  %s\n", g_mmk_info[i].name,
                (double)m / 1e9, atomic_load_explicit(&g_mm_calls[i], memory_order_relaxed),
                100.0 * (double)m / (double)tot, g_mmc_note[g_mmk_info[i].cls]);
    }
    static const char *cname[QWEN_COMP_COUNT] = { "other", "Talker", "Code Predictor", "speech decoder" };
    for (int c = 0; c < QWEN_COMP_COUNT; c++) {
        long long ct = 0;
        for (int i = 1; i < QWEN_MMK_COUNT; i++)
            ct += atomic_load_explicit(&g_mm_macs_c[c][i], memory_order_relaxed);
        if (!ct) continue;
        fprintf(f, "  --- %s\n", cname[c]);
        for (int i = 1; i < QWEN_MMK_COUNT; i++) {
            long long m = atomic_load_explicit(&g_mm_macs_c[c][i], memory_order_relaxed);
            if (!m) continue;
            fprintf(f, "      %-24s %10.2f GMAC %9lld calls %6.1f%%  %s\n", g_mmk_info[i].name,
                    (double)m / 1e9, atomic_load_explicit(&g_mm_calls_c[c][i], memory_order_relaxed),
                    100.0 * (double)m / (double)ct, g_mmc_note[g_mmk_info[i].cls]);
        }
    }
    fprintf(f, "  ---\n");
    {
        long long bf = atomic_load_explicit(&g_mm_fixedw[0], memory_order_relaxed);
        long long bg = atomic_load_explicit(&g_mm_generic[0], memory_order_relaxed);
        long long qf = atomic_load_explicit(&g_mm_fixedw[1], memory_order_relaxed);
        long long qg = atomic_load_explicit(&g_mm_generic[1], memory_order_relaxed);
        if (bf || bg || qf || qg)
            fprintf(f, "  fallback twin dispatch: bf16 %lld fixed-width / %lld generic  ·  "
                       "int8 %lld fixed-width / %lld generic\n", bf, bg, qf, qg);
        else
            fprintf(f, "  fallback twin dispatch: never reached (a wider matmat took every "
                       "batched call on this build)\n");
    }
    fprintf(f, "  matrix-matrix %5.1f%%  ·  batched twin %5.1f%%  ·  B x matvec %5.1f%%  ·  single-slot %5.1f%%  ·  GEMV %5.1f%%\n",
            100.0 * (double)by_cls[MMC_GEMM]   / (double)tot,
            100.0 * (double)by_cls[MMC_TWIN]   / (double)tot,
            100.0 * (double)by_cls[MMC_MATVEC] / (double)tot,
            100.0 * (double)by_cls[MMC_SOLO]   / (double)tot,
            100.0 * (double)by_cls[MMC_GEMV]   / (double)tot);
    if (by_cls[MMC_GEMM] * 2 < tot && by_cls[MMC_GEMV] * 2 < tot)
        fprintf(f, "  ⚠️  less than half the work went through a real matrix-matrix instruction:\n"
                   "      a B=1..8 curve measured here describes the FALLBACK, not this silicon.\n");
    long long wb = atomic_load_explicit(&g_mm_wbytes, memory_order_relaxed);
    if (wb > 0) {
        double t0 = atomic_load_explicit(&g_mm_wb_t0, memory_order_relaxed);
        double t1 = atomic_load_explicit(&g_mm_wb_t1, memory_order_relaxed);
        double el = (t1 > t0) ? t1 - t0 : 0.0;
        fprintf(f, "  weight traffic  %.2f GB in %.1f s = %.2f GB/s"
                   "  (Talker+CP projections only: no KV, no activations, no decoder -> LOWER bound)\n",
                (double)wb / 1e9, el, el > 0 ? (double)wb / 1e9 / el : 0.0);
        fprintf(f, "                  divide by the box's measured STREAM bandwidth"
                   " (make server-hw-check) -> BW_utilization\n");
    }
    fflush(f);
}
static void qwen_mm_stats_atexit(void) { qwen_matmat_stats_report(NULL); }

#if defined(__GNUC__)
#define QWEN_MAYBE_UNUSED __attribute__((unused))
#else
#define QWEN_MAYBE_UNUSED
#endif

typedef struct {
    const char *off_env;
    const char *on_env;
    const char *minb_env;
    const char *minrows_env;
    const char *mincols_env;
    short min_b, max_b;
    int   min_rows, min_cols;
    unsigned char amx;
    unsigned char apple_off;
} qwen_mm_gate_t;

static int qwen_mm_env_int(const char *name, int dflt, int lo, int hi) QWEN_MAYBE_UNUSED;
static int qwen_mm_env_int(const char *name, int dflt, int lo, int hi) {
    if (!name) return dflt;
    const char *e = getenv(name);
    if (!e || !e[0]) return dflt;
    int v = atoi(e);
    return (v >= lo && v <= hi) ? v : dflt;
}

static const char *qwen_mm_specific_minb_env(int mmk) {
#if defined(__x86_64__) || defined(_M_X64)
    if (mmk == QWEN_MMK_BF16_AMX) return "QWEN_AMX_BF16_MIN_B";
    if (mmk == QWEN_MMK_INT8_AMX) return "QWEN_AMX_INT8_MIN_B";
#else
    (void)mmk;
#endif
    return NULL;
}

static int qwen_mm_minb_value(int mmk, const qwen_mm_gate_t *g) {
    int v = qwen_mm_env_int(g->minb_env, g->min_b, 1, 64);
    const char *specific = qwen_mm_specific_minb_env(mmk);
    if (specific) v = qwen_mm_env_int(specific, v, 1, 64);
    return v;
}

static const qwen_mm_gate_t g_mm_gate[QWEN_MMK_COUNT] QWEN_MAYBE_UNUSED = {
    [QWEN_MMK_BF16_BFMMLA] = { "QWEN_NO_BFMMLA",   NULL,                "QWEN_BFMMLA_MIN_B",  NULL,                NULL,                     2, 64,  0,  0, 0, 1 },
    [QWEN_MMK_BF16_AVX512] = { "QWEN_NO_BF16_MATMUL", NULL,              "QWEN_BF16_MATMUL_MIN_B", NULL,             NULL,                     1, 16,  0,  0, 0, 0 },
    [QWEN_MMK_BF16_AMX]    = { "QWEN_NO_AMX_BF16", NULL,                "QWEN_AMX_MIN_B",     "QWEN_AMX_MIN_ROWS", "QWEN_AMX_BF16_MIN_COLS", 4, 16, 32, 32, 1, 0 },
    /* INT8 AMX starts at B=3, not 4: the batched server measured B 0.9-3.8 per prefork worker
     * across C=1..8, so a B>=4 gate left the tile path essentially unused in production.  At
     * B=3 with the rows-per-thread rule below, the paired measurement gives CP Gate/Up -14.5%,
     * TK Gate/Up -12.4%, TK Down -4.4%, TK WO -3.0% and QKV neutral, while the two projections
     * that lose there (CP WO +19.9%, CP Down +7.0%) are the ones the rule already excludes.
     * B=2 stays VNNI: the wins shrink to -3..-5% and more shapes turn negative.  BF16 AMX
     * keeps its own default. */
    [QWEN_MMK_INT8_AMX]    = { "QWEN_NO_AMX_INT8", NULL,                "QWEN_AMX_MIN_B",     "QWEN_AMX_MIN_ROWS", "QWEN_AMX_INT8_MIN_COLS", 3, 16, 32, 64, 1, 0 },
    [QWEN_MMK_INT8_VNNI]   = { "QWEN_NO_VNNI",     NULL,                "QWEN_VNNI_MIN_B",    NULL,                NULL,                     2, 16,  0,  0, 0, 0 },
    [QWEN_MMK_INT8_AVX2]   = { "QWEN_NO_AVX2MM",   NULL,                "QWEN_AVX2MM_MIN_B",  NULL,                NULL,                     2, 16,  0,  0, 0, 0 },
    [QWEN_MMK_INT8_SMMLA]  = { "QWEN_NO_SMMLA",    NULL,                "QWEN_SMMLA_MIN_B",   NULL,                NULL,                     2, 16,  0,  0, 0, 1 },
    [QWEN_MMK_INT8_SDOT]   = { NULL,               "QWEN_INT8_SDOT_MM", "QWEN_INT8_SDOT_MIN_B", NULL,              NULL,                     2, 16,  0,  0, 0, 0 },
    [QWEN_MMK_Q4_AMX]      = { "QWEN_NO_AMX_Q4",   NULL,                "QWEN_AMX_MIN_B",     "QWEN_AMX_MIN_ROWS", "QWEN_AMX_Q4_MIN_COLS",   4, 16, 32, 32, 1, 0 },
    [QWEN_MMK_Q4_VNNI]     = { "QWEN_NO_VNNI",     NULL,                "QWEN_VNNI_MIN_B",    NULL,                NULL,                     2, 16,  0,  0, 0, 0 },
    [QWEN_MMK_Q4_AVX2]     = { "QWEN_NO_AVX2MM",   NULL,                "QWEN_AVX2MM_MIN_B",  NULL,                NULL,                     2, 16,  0,  0, 0, 0 },
    [QWEN_MMK_Q4_SMMLA]    = { "QWEN_NO_SMMLA",    NULL,                "QWEN_SMMLA_MIN_B",   NULL,                NULL,                     2, 16,  0,  0, 0, 0 },
    [QWEN_MMK_KLEIDI_Q4]   = { "QWEN_NO_KLEIDI",   NULL,                "QWEN_KLEIDI_MIN_B",  NULL,                NULL,                     1, 64,  0,  0, 0, 0 },
};
static atomic_int g_mm_gate_on[QWEN_MMK_COUNT];
static atomic_int g_mm_gate_minb[QWEN_MMK_COUNT];
static atomic_int g_mm_gate_minrows[QWEN_MMK_COUNT];
static atomic_int g_mm_gate_mincols[QWEN_MMK_COUNT];

static atomic_int g_mm_force;
static void qwen_mm_force_kernel(int mmk) QWEN_MAYBE_UNUSED;
/* Bench hook: pin the batched dispatcher to ONE kernel so two arms can be interleaved inside
 * a single process.  Comparing arms across processes on a shared box measured 14% swings on an
 * unchanged configuration -- larger than the effect under test -- so a paired design is the
 * only honest way to time these. 0 restores normal dispatch. */
void qwen_mm_force(int mmk) { qwen_mm_force_kernel(mmk); }
static void qwen_mm_force_kernel(int mmk) {
    atomic_store_explicit(&g_mm_force, mmk, memory_order_relaxed);
}

/* AMX INT8 needs a WORK-PER-THREAD condition, not only a batch one.  Measured on a Xeon
 * Platinum 8581C (Emerald Rapids, 12 physical cores, SMT off) with the complete in-region contract (activation pack + packed RHS + matmul + scale)
 * against the VNNI row blocks, on the real 1.7B projections, paired and interleaved inside one
 * process, median of 7 rounds, B=4.  The AMX arm is the winner marked (+ means AMX is slower):
 *
 *   rows/thread   projection (threads)              AMX vs VNNI
 *          85     CP WO (12), CP Down (12)          +38.3%, +13.4%   VNNI
 *         128     CP WO  (8), CP Down  (8)           +9.6%,  +1.3%   VNNI
 *         170     TK WO (12), TK Down (12)           +1.1%,  +1.2%   VNNI
 *         170     CP QKV(12), TK QKV  (12)           -7.9%,  -6.0%   AMX (fused, see below)
 *         256     CP QKV (8), TK QKV (8), TK WO (8), TK Down (8), CP Down (4)
 *                                                    -16.4% .. -5.2% AMX
 *         512+    QKV/WO/Down (4), Gate/Up (all)     -28.2% .. -3.8% AMX
 *
 * The tile path needs enough output rows PER WORKER to amortise its tile setup and its
 * activation pack; below roughly 256 it cannot, and the same projection flips sign purely by
 * changing the thread count -- CP Down is -17.3% at 4 threads and +1.3% at 8.  A rows-vs-cols
 * rule looked convincing on unpaired numbers and was wrong: TK Down (2048x6144) is AMX -5.2%
 * at 8 threads despite being three times deeper than tall.  rows/thread >= 256 agrees with 23
 * of the 24 measured cells (the exception is CP WO at 4 threads, +4.4%).
 *
 * 0 disables the rule.  Thread count is the engine's, which in a prefork worker is that
 * worker's slice -- the same workers that will split these rows. */
static int qwen_amx_int8_rows_ok_nt(long long rows, int nt) QWEN_MAYBE_UNUSED;
static int qwen_amx_int8_rows_ok_nt(long long rows, int nt) {
    static atomic_int rpt = 0;
    int v = atomic_load_explicit(&rpt, memory_order_relaxed);
    if (v == 0) {
        v = qwen_mm_env_int("QWEN_AMX_INT8_MIN_ROWS_PER_THREAD", 256, 0, 1 << 20) + 1;
        atomic_store_explicit(&rpt, v, memory_order_relaxed);
    }
    v -= 1;
    if (v <= 0) return 1;
    if (nt < 1) nt = 1;
    return rows >= (long long)v * nt;
}
static int qwen_amx_int8_rows_ok(long long rows) QWEN_MAYBE_UNUSED;
static int qwen_amx_int8_rows_ok(long long rows) {
    return qwen_amx_int8_rows_ok_nt(rows, qwen_get_threads());
}

static int qwen_mm_use_(int mmk, int B, int rows, int cols, int amx_shape) QWEN_MAYBE_UNUSED;
static int qwen_mm_use(int mmk, int B, int rows, int cols) QWEN_MAYBE_UNUSED;
static int qwen_mm_use(int mmk, int B, int rows, int cols) {
    return qwen_mm_use_(mmk, B, rows, cols, 1);
}
static int qwen_mm_use_(int mmk, int B, int rows, int cols, int amx_shape) {
    if (mmk <= 0 || mmk >= QWEN_MMK_COUNT) return 0;
    const qwen_mm_gate_t *g = &g_mm_gate[mmk];
    if (g->max_b == 0) return 0;
    int force = atomic_load_explicit(&g_mm_force, memory_order_relaxed);
    if (force != 0) return force == mmk && B <= g->max_b;
    int st = atomic_load_explicit(&g_mm_gate_on[mmk], memory_order_relaxed);
    if (st == 0) {
        int on = 1;
        const char *e;
        if (g->on_env)          { e = getenv(g->on_env);  on = (e && e[0] == '1'); }
        if (on && g->off_env)   { e = getenv(g->off_env); if (e && e[0] == '1') on = 0; }
        if (on && g->amx)       { e = getenv("QWEN_NO_AMX"); if (e && e[0] == '1') on = 0; }
#if defined(__APPLE__)
        if (on && g->apple_off) { e = getenv("QWEN_APPLE_MMLA"); on = (e && e[0] == '1'); }
#endif
        st = on ? 1 : 2;
        atomic_store_explicit(&g_mm_gate_on[mmk], st, memory_order_relaxed);
    }
    if (st != 1) return 0;
    int minb = atomic_load_explicit(&g_mm_gate_minb[mmk], memory_order_relaxed);
    if (minb == 0) {
        minb = qwen_mm_minb_value(mmk, g);
        atomic_store_explicit(&g_mm_gate_minb[mmk], minb, memory_order_relaxed);
    }
    int minr = atomic_load_explicit(&g_mm_gate_minrows[mmk], memory_order_relaxed);
    if (minr == 0) {
        minr = qwen_mm_env_int(g->minrows_env, g->min_rows, 0, 1 << 20) + 1;
        atomic_store_explicit(&g_mm_gate_minrows[mmk], minr, memory_order_relaxed);
    }
    int minc = atomic_load_explicit(&g_mm_gate_mincols[mmk], memory_order_relaxed);
    if (minc == 0) {
        minc = qwen_mm_env_int(g->mincols_env, g->min_cols, 0, 1 << 20) + 1;
        atomic_store_explicit(&g_mm_gate_mincols[mmk], minc, memory_order_relaxed);
    }
    if (!(B >= minb && B <= g->max_b && rows >= minr - 1 && cols >= minc - 1)) return 0;
#if defined(__x86_64__) || defined(_M_X64)
    if (amx_shape && mmk == QWEN_MMK_INT8_AMX && !qwen_amx_int8_rows_ok(rows)) return 0;
#else
    (void)amx_shape;
#endif
    return 1;
}

/* --dispatch-map: which gate rows are compiled into THIS binary.  Mirrors the
 * candidate lists of qwen_kernel_selection_report(); a row that is not compiled
 * can never be selected whatever the env says. */
static int qwen_mmk_compiled(int mmk) {
    switch (mmk) {
#if defined(__AMX_BF16__) && defined(__AMX_TILE__)
    case QWEN_MMK_BF16_AMX: return 1;
#endif
#if defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC)
    case QWEN_MMK_BF16_BFMMLA: return 1;
#endif
#if defined(__AVX512BF16__)
    case QWEN_MMK_BF16_AVX512: return 1;
#endif
#if defined(__AMX_INT8__) && defined(__AMX_TILE__)
    case QWEN_MMK_INT8_AMX: case QWEN_MMK_Q4_AMX: return 1;
#endif
#if defined(__AVX512VNNI__)
    case QWEN_MMK_INT8_VNNI: case QWEN_MMK_Q4_VNNI: return 1;
#endif
#if defined(__ARM_FEATURE_MATMUL_INT8)
    case QWEN_MMK_INT8_SMMLA: case QWEN_MMK_Q4_SMMLA: case QWEN_MMK_KLEIDI_Q4: return 1;
#endif
#if defined(__AVX2__)
    case QWEN_MMK_INT8_AVX2: case QWEN_MMK_Q4_AVX2: return 1;
#endif
#if defined(__ARM_FEATURE_DOTPROD)
    case QWEN_MMK_INT8_SDOT: return 1;
#endif
    default: return 0;
    }
}

/* Does the CPU we are running on have the instructions this row needs?  Independent
 * of the env: the env decides "on", this decides "could it ever be on". */
static int qwen_mmk_supported(int mmk) {
    switch (mmk) {
    case QWEN_MMK_BF16_AMX:  return qwen_amx_bf16_available();
    case QWEN_MMK_INT8_AMX: case QWEN_MMK_Q4_AMX: return qwen_amx_int8_available();
#if defined(__x86_64__) || defined(_M_X64)
    case QWEN_MMK_BF16_AVX512: return __builtin_cpu_supports("avx512bf16") ? 1 : 0;
    case QWEN_MMK_INT8_VNNI: case QWEN_MMK_Q4_VNNI: return __builtin_cpu_supports("avx512vnni") ? 1 : 0;
    case QWEN_MMK_INT8_AVX2: case QWEN_MMK_Q4_AVX2: return __builtin_cpu_supports("avx2") ? 1 : 0;
#endif
    case QWEN_MMK_KLEIDI_Q4: return qwen_kleidi_supported();
    default: return qwen_mmk_compiled(mmk);   /* -march=native: compiled == the host has it */
    }
}

/* The env-side explanation of a gate row.  This is a DESCRIPTION for the report;
 * the truth (`on`) comes from calling qwen_mm_use() itself, never from here. */
static const char *qwen_mm_gate_reason(const qwen_mm_gate_t *g, int on) {
    const char *e;
    if (g->on_env) {
        e = getenv(g->on_env);
        if (!(e && e[0] == '1')) return "opt-in, env unset";
    }
    if (g->off_env && (e = getenv(g->off_env)) && e[0] == '1') return "off_env=1";
    if (g->amx && (e = getenv("QWEN_NO_AMX")) && e[0] == '1') return "QWEN_NO_AMX=1";
#if defined(__APPLE__)
    if (g->apple_off) {
        e = getenv("QWEN_APPLE_MMLA");
        return (e && e[0] == '1') ? "QWEN_APPLE_MMLA=1" : "default OFF on Apple (QWEN_APPLE_MMLA=1)";
    }
#endif
    if (g->on_env) return "opt-in env set";
    return on ? "default ON" : "gate refused at probe shape";
}

int qwen_mm_gate_describe(int mmk, qwen_mm_gate_desc_t *d) {
    if (!d || mmk <= 0 || mmk >= QWEN_MMK_COUNT) return 0;
    const qwen_mm_gate_t *g = &g_mm_gate[mmk];
    if (g->max_b == 0) return 0;
    memset(d, 0, sizeof *d);
    d->mmk = mmk;
    d->name = g_mmk_info[mmk].name;
    d->off_env = g->off_env; d->on_env = g->on_env; d->minb_env = g->minb_env;
    d->minrows_env = g->minrows_env; d->mincols_env = g->mincols_env;
    d->compiled_min_b = g->min_b; d->max_b = g->max_b;
    d->compiled_min_rows = g->min_rows; d->compiled_min_cols = g->min_cols;
    d->amx = g->amx; d->apple_off = g->apple_off;
    d->compiled = qwen_mmk_compiled(mmk);
    d->supported = d->compiled ? qwen_mmk_supported(mmk) : 0;
    d->min_b = qwen_mm_minb_value(mmk, g);
    d->min_rows = qwen_mm_env_int(g->minrows_env, g->min_rows, 0, 1 << 20);
    d->min_cols = qwen_mm_env_int(g->mincols_env, g->min_cols, 0, 1 << 20);
    /* the real predicate, at the smallest B and a shape past every row/col floor */
    d->on = qwen_mm_use(mmk, d->min_b, 1 << 16, 1 << 16);
    d->reason = qwen_mm_gate_reason(g, d->on);
    return 1;
}

int qwen_amx_prepack_requested(void) {
    const char *e = getenv("QWEN_AMX_PREPACK");
    return e && e[0] == '1';
}
int qwen_vnni_prepack_requested(void) {
    const char *e = getenv("QWEN_VNNI_PREPACK");
    return e && (e[0] == '1' || !strcmp(e, "all") || !strcmp(e, "cp") ||
                 !strcmp(e, "talker"));
}

#if defined(__x86_64__) || defined(_M_X64)
static atomic_int g_x86_nchunk = -1;
static atomic_int g_amx_nchunk = -1;
static atomic_int g_vnni_nchunk = -1;
static atomic_int g_avx512_nchunk = -1;

static int qwen_x86_nchunk_value(const char *specific, atomic_int *cache) {
    int v = atomic_load_explicit(cache, memory_order_relaxed);
    if (v >= 0) return v;
    v = qwen_mm_env_int(specific, -1, 0, 1 << 20);
    if (v < 0) v = qwen_mm_env_int("QWEN_X86_NCHUNK", 0, 0, 1 << 20);
    atomic_store_explicit(cache, v, memory_order_relaxed);
    return v;
}

static int qwen_x86_nchunk(int mmk, int B) {
    const char *specific = NULL;
    atomic_int *cache = &g_x86_nchunk;
    int align = 16;
    if (mmk == QWEN_MMK_INT8_AMX || mmk == QWEN_MMK_BF16_AMX) {
        specific = "QWEN_AMX_NCHUNK";
        cache = &g_amx_nchunk;
    } else if (mmk == QWEN_MMK_INT8_VNNI) {
        specific = "QWEN_VNNI_NCHUNK";
        cache = &g_vnni_nchunk;
        align = B <= 4 ? 4 : 2;
    } else if (mmk == QWEN_MMK_BF16_AVX512) {
        specific = "QWEN_AVX512_NCHUNK";
        cache = &g_avx512_nchunk;
        align = B <= 4 ? 4 : 2;
    } else {
        return 0;
    }
    int v = qwen_x86_nchunk_value(specific, cache);
    v = (v / align) * align;
    return v >= align ? v : 0;
}
#endif

void qwen_kernel_selection_report(void *out, int rows, int cols) {
    FILE *f = out ? (FILE *)out : stderr;
    if (rows <= 0) rows = 2048;
    if (cols <= 0) cols = 2048;

    int bf16_c[5], int8_c[6], q4_c[6];
    int nbf = 0, nint8 = 0, nq4 = 0;
#if defined(__AMX_BF16__) && defined(__AMX_TILE__)
    if (qwen_amx_bf16_ready()) bf16_c[nbf++] = QWEN_MMK_BF16_AMX;
#endif
#if defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC)
    bf16_c[nbf++] = QWEN_MMK_BF16_BFMMLA;
#endif
#if defined(__AVX512BF16__)
    bf16_c[nbf++] = QWEN_MMK_BF16_AVX512;
#endif
#if defined(__AMX_INT8__) && defined(__AMX_TILE__)
    if (qwen_amx_int8_ready()) { int8_c[nint8++] = QWEN_MMK_INT8_AMX; q4_c[nq4++] = QWEN_MMK_Q4_AMX; }
#endif
#if defined(__AVX512VNNI__)
    int8_c[nint8++] = QWEN_MMK_INT8_VNNI;  q4_c[nq4++] = QWEN_MMK_Q4_VNNI;
#endif
#if defined(__ARM_FEATURE_MATMUL_INT8)
    int8_c[nint8++] = QWEN_MMK_INT8_SMMLA; q4_c[nq4++] = QWEN_MMK_Q4_SMMLA;
#endif
#if defined(__AVX2__)
    int8_c[nint8++] = QWEN_MMK_INT8_AVX2;  q4_c[nq4++] = QWEN_MMK_Q4_AVX2;
#endif
#if defined(__ARM_FEATURE_DOTPROD)
    int8_c[nint8++] = QWEN_MMK_INT8_SDOT;
#endif

    fprintf(f, "  kernel selection (shape %dx%d, asked to the dispatcher):\n", rows, cols);

    fprintf(f, "    B=1  (CLI, server c=1) matvec: bf16 -> %s | int8 -> %s | q4_0 -> %s\n",
#if defined(__ARM_FEATURE_BF16) && !defined(__APPLE__)
            getenv("QWEN_ARM_BFDOT") ? "BFDOT" : "NEON 2-row fused",
#else
            "NEON/scalar 2-row fused",
#endif
#if defined(__ARM_FEATURE_DOTPROD)
            getenv("QWEN_NO_SDOT") ? "f32-accum (SDOT off)" : "SDOT vdotq_s32",
            getenv("QWEN_NO_SDOT") ? "f32 dequant"          : "SDOT vdotq_s32"
#elif defined(__AVX512VNNI__)
            getenv("QWEN_NO_VNNI") ? "f32-accum (VNNI off)" : "VNNI vpdpbusd",
            getenv("QWEN_NO_VNNI") ? "f32 dequant"          : "VNNI vpdpbusd"
#else
            "f32-accum fused", "f32 dequant"
#endif
            );

    const struct { const char *what; const int *c; int n; const char *fallback; } rows_[] = {
        { "bf16", bf16_c, nbf,   "fixed-B twin"  },
        { "int8", int8_c, nint8, "f32-accum twin" },
        { "q4_0", q4_c,   nq4,
#if defined(__ARM_FEATURE_DOTPROD) && !defined(__ARM_FEATURE_MATMUL_INT8)
          "B x matvec (no matrix unit)"
#else
          "generic twin"
#endif
        },
    };
    for (int b = 2; b <= 16; b *= 2) {
        fprintf(f, "    B=%-2d %-16s matmat:", b, b == 2 ? "(server c>=2)" : "");
        for (size_t r = 0; r < sizeof rows_ / sizeof rows_[0]; r++) {
            const char *pick = rows_[r].fallback;
            for (int i = 0; i < rows_[r].n; i++) {
                if (qwen_mm_use(rows_[r].c[i], b, rows, cols)) { pick = g_mmk_info[rows_[r].c[i]].name; break; }
            }
            fprintf(f, "%s %s -> %s", r ? " |" : "", rows_[r].what, pick);
        }
        fprintf(f, "\n");
    }
    (void)0;
#if 0
    fprintf(f, "    %-12s B=1  -> %s\n", "matvec int8",
#if defined(__ARM_FEATURE_DOTPROD)
            getenv("QWEN_NO_SDOT") ? "f32-accum fused (SDOT disabled by env)" : "SDOT vdotq_s32");
#elif defined(__AVX512VNNI__)
            getenv("QWEN_NO_VNNI") ? "f32-accum fused (VNNI disabled by env)" : "VNNI vpdpbusd");
#else
            "f32-accum fused");
#endif
#endif
}

#if (defined(__AMX_INT8__) || defined(__AMX_BF16__)) && defined(__AMX_TILE__)
typedef struct {
    uint8_t  palette_id;
    uint8_t  start_row;
    uint8_t  reserved0[14];
    uint16_t colsb[8];
    uint16_t reserved1[8];
    uint8_t  rows[8];
    uint8_t  reserved2[8];
} qwen_amx_tilecfg;

static int qwen_amx_persistent_config(void) {
    static atomic_int enabled = -1;
    int v = atomic_load_explicit(&enabled, memory_order_relaxed);
    if (v < 0) {
        const char *e = getenv("QWEN_AMX_PERSIST_CFG");
        v = !(e && e[0] == '0');
        atomic_store_explicit(&enabled, v, memory_order_relaxed);
    }
    return v;
}

static _Thread_local unsigned qwen_amx_cfg_key;
static _Thread_local int qwen_amx_cfg_valid;

static void qwen_amx_prepare_config(const qwen_amx_tilecfg *cfg, unsigned key) {
    if (!qwen_amx_persistent_config() || !qwen_amx_cfg_valid ||
        qwen_amx_cfg_key != key) {
        _tile_loadconfig(cfg);
        if (qwen_amx_persistent_config()) {
            qwen_amx_cfg_key = key;
            qwen_amx_cfg_valid = 1;
        }
    }
}

static void qwen_amx_finish_config(void) {
    if (!qwen_amx_persistent_config()) {
        _tile_release();
        qwen_amx_cfg_valid = 0;
    }
}

enum {
    QWEN_AMX_WEIGHT_CACHE_MAX = 512,
};

typedef struct {
    const void *source;
    int rows;
    int cols;
    int kind;
    size_t bytes;
    void *packed;
} qwen_amx_weight_entry;

static qwen_amx_weight_entry g_amx_weights[QWEN_AMX_WEIGHT_CACHE_MAX];
static atomic_int g_amx_weight_count;
static pthread_mutex_t g_amx_weight_mu = PTHREAD_MUTEX_INITIALIZER;

static int qwen_amx_prepack_enabled(void) {
    static atomic_int enabled = -1;
    int v = atomic_load_explicit(&enabled, memory_order_relaxed);
    if (v < 0) {
        const char *e = getenv("QWEN_AMX_PREPACK");
        v = e && e[0] == '1';
        atomic_store_explicit(&enabled, v, memory_order_relaxed);
    }
    return v;
}

static int qwen_amx_prepack_kind_enabled(int kind) {
    const char *e = getenv("QWEN_AMX_PREPACK_KINDS");
    if (!e || !*e || !strcmp(e, "both") || !strcmp(e, "all")) return 1;
    const char *want = kind == QWEN_AMX_WEIGHT_BF16 ? "bf16" : "int8";
    size_t want_len = strlen(want);
    for (const char *p = e; *p;) {
        while (*p == ',' || *p == ' ' || *p == '\t') p++;
        const char *start = p;
        while (*p && *p != ',') p++;
        const char *end = p;
        while (end > start && (end[-1] == ' ' || end[-1] == '\t')) end--;
        if ((size_t)(end - start) == want_len && !strncmp(start, want, want_len)) return 1;
        if (*p == ',') p++;
    }
    return 0;
}

static int qwen_amx_b32_enabled(void) {
    const char *e = getenv("QWEN_AMX_B32");
    return e && e[0] == '1';
}

static void *qwen_amx_pack_weights(const void *source, int rows, int cols, int kind) {
    if (!qwen_amx_prepack_enabled() || !qwen_amx_prepack_kind_enabled(kind) ||
        !source || rows < 16 || cols <= 0) return NULL;
    const int kstep = kind == QWEN_AMX_WEIGHT_BF16 ? 32 : 64;
    const size_t elem_size = kind == QWEN_AMX_WEIGHT_BF16 ? sizeof(uint16_t) : sizeof(int8_t);
    const int row_blocks = rows / 16;
    const int col_blocks = cols / kstep;
    if (row_blocks <= 0 || col_blocks <= 0) return NULL;
    const size_t tile_bytes = (size_t)16 * (size_t)kstep * elem_size;
    if ((size_t)row_blocks > SIZE_MAX / (size_t)col_blocks ||
        (size_t)row_blocks * (size_t)col_blocks > SIZE_MAX / tile_bytes)
        return NULL;
    const size_t bytes = (size_t)row_blocks * (size_t)col_blocks * tile_bytes;

    int n = atomic_load_explicit(&g_amx_weight_count, memory_order_acquire);
    for (int i = 0; i < n; i++) {
        const qwen_amx_weight_entry *e = &g_amx_weights[i];
        if (e->source == source && e->rows == rows && e->cols == cols && e->kind == kind)
            return e->packed;
    }

    pthread_mutex_lock(&g_amx_weight_mu);
    n = atomic_load_explicit(&g_amx_weight_count, memory_order_relaxed);
    for (int i = 0; i < n; i++) {
        qwen_amx_weight_entry *e = &g_amx_weights[i];
        if (e->source == source && e->rows == rows && e->cols == cols && e->kind == kind) {
            void *packed = e->packed;
            pthread_mutex_unlock(&g_amx_weight_mu);
            return packed;
        }
    }
    if (n >= QWEN_AMX_WEIGHT_CACHE_MAX) {
        pthread_mutex_unlock(&g_amx_weight_mu);
        return NULL;
    }

    void *packed = NULL;
    if (posix_memalign(&packed, 64, bytes) != 0) {
        pthread_mutex_unlock(&g_amx_weight_mu);
        return NULL;
    }
    const size_t source_stride = (size_t)cols * elem_size;
    for (int rb = 0; rb < row_blocks; rb++) {
        for (int cb = 0; cb < col_blocks; cb++) {
            uint8_t *dst = (uint8_t *)packed +
                           ((size_t)rb * (size_t)col_blocks + (size_t)cb) * tile_bytes;
            for (int m = 0; m < 16; m++) {
                const uint8_t *src = (const uint8_t *)source +
                                     (size_t)(rb * 16 + m) * source_stride +
                                     (size_t)cb * (size_t)kstep * elem_size;
                memcpy(dst + (size_t)m * (size_t)kstep * elem_size,
                       src, (size_t)kstep * elem_size);
            }
        }
    }
    g_amx_weights[n].source = source;
    g_amx_weights[n].rows = rows;
    g_amx_weights[n].cols = cols;
    g_amx_weights[n].kind = kind;
    g_amx_weights[n].bytes = bytes;
    g_amx_weights[n].packed = packed;
    atomic_store_explicit(&g_amx_weight_count, n + 1, memory_order_release);
    pthread_mutex_unlock(&g_amx_weight_mu);
    return packed;
}

int qwen_amx_prepack_weight(const void *source, int rows, int cols, int kind) {
    if (!source || rows <= 0 || cols <= 0 ||
        (kind != QWEN_AMX_WEIGHT_BF16 && kind != QWEN_AMX_WEIGHT_INT8)) return 0;
#if defined(__AMX_BF16__) && defined(__AMX_TILE__)
    if (kind == QWEN_AMX_WEIGHT_BF16 && !qwen_amx_bf16_ready()) return 0;
#elif !defined(__AMX_BF16__)
    if (kind == QWEN_AMX_WEIGHT_BF16) return 0;
#endif
#if defined(__AMX_INT8__) && defined(__AMX_TILE__)
    if (kind == QWEN_AMX_WEIGHT_INT8 && !qwen_amx_int8_ready()) return 0;
#elif !defined(__AMX_INT8__)
    if (kind == QWEN_AMX_WEIGHT_INT8) return 0;
#endif
    return qwen_amx_pack_weights(source, rows, cols, kind) != NULL;
}

void qwen_amx_prepack_stats(int *n_packed, size_t *bytes) {
    int n = 0;
    size_t total = 0;
    pthread_mutex_lock(&g_amx_weight_mu);
    n = atomic_load_explicit(&g_amx_weight_count, memory_order_relaxed);
    for (int i = 0; i < n; i++) total += g_amx_weights[i].bytes;
    pthread_mutex_unlock(&g_amx_weight_mu);
    if (n_packed) *n_packed = n;
    if (bytes) *bytes = total;
}

void qwen_amx_weight_cache_reset(void) {
    pthread_mutex_lock(&g_amx_weight_mu);
    int n = atomic_load_explicit(&g_amx_weight_count, memory_order_relaxed);
    for (int i = 0; i < n; i++) {
        free(g_amx_weights[i].packed);
        g_amx_weights[i].source = NULL;
        g_amx_weights[i].rows = 0;
        g_amx_weights[i].cols = 0;
        g_amx_weights[i].kind = 0;
        g_amx_weights[i].bytes = 0;
        g_amx_weights[i].packed = NULL;
    }
    atomic_store_explicit(&g_amx_weight_count, 0, memory_order_release);
    pthread_mutex_unlock(&g_amx_weight_mu);
}
#endif

#if !((defined(__AMX_INT8__) || defined(__AMX_BF16__)) && defined(__AMX_TILE__))
int qwen_amx_prepack_weight(const void *source, int rows, int cols, int kind) {
    (void)source; (void)rows; (void)cols; (void)kind;
    return 0;
}
void qwen_amx_prepack_stats(int *n_packed, size_t *bytes) {
    if (n_packed) *n_packed = 0;
    if (bytes) *bytes = 0;
}
void qwen_amx_weight_cache_reset(void) {}
#endif

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
/* 9..15 existed only as the generic fallback, and it costs about 10x the fixed-width kernel:
   measured on an M1, one 2048x2048 call is 2.6 ms at B=8 and B=16 but 21-26 ms at B=9..15.
   A prefill's last chunk is B = positions mod 16, so seven input lengths in every sixteen were
   paying that. */
DEFINE_MATMAT_FIXED_B(9)
DEFINE_MATMAT_FIXED_B(10)
DEFINE_MATMAT_FIXED_B(11)
DEFINE_MATMAT_FIXED_B(12)
DEFINE_MATMAT_FIXED_B(13)
DEFINE_MATMAT_FIXED_B(14)
DEFINE_MATMAT_FIXED_B(15)
DEFINE_MATMAT_FIXED_B(16)
#undef DEFINE_MATMAT_FIXED_B

static void bf16_matmat_slice(float *Y, const uint16_t *W, const float *X,
                              int r0, int r1, int cols, int B) {
    MMSTAT(QWEN_MMK_BF16_FIXEDB, r1 - r0, cols, B);
    if (qwen_matmat_stats_enabled() || qwen_census_enabled()) {
        int fixed = (B >= 1 && B <= 16);
        atomic_fetch_add_explicit(&g_mm_fixedw[0], fixed, memory_order_relaxed);
        atomic_fetch_add_explicit(&g_mm_generic[0], !fixed, memory_order_relaxed);
    }
    switch (B) {
        case 1:  bf16_matmat_b1 (Y, W, X, r0, r1, cols); return;
        case 2:  bf16_matmat_b2 (Y, W, X, r0, r1, cols); return;
        case 3:  bf16_matmat_b3 (Y, W, X, r0, r1, cols); return;
        case 4:  bf16_matmat_b4 (Y, W, X, r0, r1, cols); return;
        case 5:  bf16_matmat_b5 (Y, W, X, r0, r1, cols); return;
        case 6:  bf16_matmat_b6 (Y, W, X, r0, r1, cols); return;
        case 7:  bf16_matmat_b7 (Y, W, X, r0, r1, cols); return;
        case 8:  bf16_matmat_b8 (Y, W, X, r0, r1, cols); return;
        case 9:  bf16_matmat_b9 (Y, W, X, r0, r1, cols); return;
        case 10: bf16_matmat_b10(Y, W, X, r0, r1, cols); return;
        case 11: bf16_matmat_b11(Y, W, X, r0, r1, cols); return;
        case 12: bf16_matmat_b12(Y, W, X, r0, r1, cols); return;
        case 13: bf16_matmat_b13(Y, W, X, r0, r1, cols); return;
        case 14: bf16_matmat_b14(Y, W, X, r0, r1, cols); return;
        case 15: bf16_matmat_b15(Y, W, X, r0, r1, cols); return;
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
void (*g_qwen_matmat_bf16_hook)(float *, const uint16_t *, const float *, int, int, int) = NULL;

#if defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC)
static inline float qbf16_to_f32(uint16_t b) {
    uint32_t u = (uint32_t)b << 16; float f; memcpy(&f, &u, 4); return f;
}
static void bf16_matmat_bfmmla_slice(float *Y, const uint16_t *W, const uint16_t *Xb,
                                     int r0, int r1, int cols, int B) {
    MMSTAT(QWEN_MMK_BF16_BFMMLA, r1 - r0, cols, B);
    int r = r0;
    for (; r + 3 < r1 && B >= 4; r += 4) {
        const bfloat16_t *w0 = (const bfloat16_t *)(W + (size_t)r * cols);
        const bfloat16_t *w1 = (const bfloat16_t *)(W + (size_t)(r + 1) * cols);
        const bfloat16_t *w2 = (const bfloat16_t *)(W + (size_t)(r + 2) * cols);
        const bfloat16_t *w3 = (const bfloat16_t *)(W + (size_t)(r + 3) * cols);
        int j = 0;
        for (; j + 3 < B; j += 4) {
            const bfloat16_t *x0 = (const bfloat16_t *)(Xb + (size_t)j * cols);
            const bfloat16_t *x1 = (const bfloat16_t *)(Xb + (size_t)(j + 1) * cols);
            const bfloat16_t *x2 = (const bfloat16_t *)(Xb + (size_t)(j + 2) * cols);
            const bfloat16_t *x3 = (const bfloat16_t *)(Xb + (size_t)(j + 3) * cols);
            float32x4_t a00 = vdupq_n_f32(0), a01 = vdupq_n_f32(0);
            float32x4_t a10 = vdupq_n_f32(0), a11 = vdupq_n_f32(0);
            int k = 0;
            for (; k + 3 < cols; k += 4) {
                bfloat16x8_t A01 = vcombine_bf16(vld1_bf16(w0 + k), vld1_bf16(w1 + k));
                bfloat16x8_t A23 = vcombine_bf16(vld1_bf16(w2 + k), vld1_bf16(w3 + k));
                bfloat16x8_t B01 = vcombine_bf16(vld1_bf16(x0 + k), vld1_bf16(x1 + k));
                bfloat16x8_t B23 = vcombine_bf16(vld1_bf16(x2 + k), vld1_bf16(x3 + k));
                a00 = vbfmmlaq_f32(a00, A01, B01);
                a01 = vbfmmlaq_f32(a01, A01, B23);
                a10 = vbfmmlaq_f32(a10, A23, B01);
                a11 = vbfmmlaq_f32(a11, A23, B23);
            }
            float t00[4], t01[4], t10[4], t11[4];
            vst1q_f32(t00, a00); vst1q_f32(t01, a01);
            vst1q_f32(t10, a10); vst1q_f32(t11, a11);
            for (; k < cols; k++) {
                float wv0 = qbf16_to_f32(W[(size_t)r * cols + k]),       wv1 = qbf16_to_f32(W[(size_t)(r+1) * cols + k]);
                float wv2 = qbf16_to_f32(W[(size_t)(r+2) * cols + k]),   wv3 = qbf16_to_f32(W[(size_t)(r+3) * cols + k]);
                float xv0 = qbf16_to_f32(Xb[(size_t)j * cols + k]),      xv1 = qbf16_to_f32(Xb[(size_t)(j+1) * cols + k]);
                float xv2 = qbf16_to_f32(Xb[(size_t)(j+2) * cols + k]),  xv3 = qbf16_to_f32(Xb[(size_t)(j+3) * cols + k]);
                t00[0] += wv0*xv0; t00[1] += wv0*xv1; t00[2] += wv1*xv0; t00[3] += wv1*xv1;
                t01[0] += wv0*xv2; t01[1] += wv0*xv3; t01[2] += wv1*xv2; t01[3] += wv1*xv3;
                t10[0] += wv2*xv0; t10[1] += wv2*xv1; t10[2] += wv3*xv0; t10[3] += wv3*xv1;
                t11[0] += wv2*xv2; t11[1] += wv2*xv3; t11[2] += wv3*xv2; t11[3] += wv3*xv3;
            }
            float *Y0 = Y + (size_t)r * B,       *Y1 = Y + (size_t)(r + 1) * B;
            float *Y2 = Y + (size_t)(r + 2) * B, *Y3 = Y + (size_t)(r + 3) * B;
            Y0[j] = t00[0]; Y0[j+1] = t00[1]; Y1[j] = t00[2]; Y1[j+1] = t00[3];
            Y0[j+2] = t01[0]; Y0[j+3] = t01[1]; Y1[j+2] = t01[2]; Y1[j+3] = t01[3];
            Y2[j] = t10[0]; Y2[j+1] = t10[1]; Y3[j] = t10[2]; Y3[j+1] = t10[3];
            Y2[j+2] = t11[0]; Y2[j+3] = t11[1]; Y3[j+2] = t11[2]; Y3[j+3] = t11[3];
        }
        for (; j < B; j++) {
            float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
            for (int k = 0; k < cols; k++) {
                float xv = qbf16_to_f32(Xb[(size_t)j * cols + k]);
                s0 += qbf16_to_f32(W[(size_t)r * cols + k])       * xv;
                s1 += qbf16_to_f32(W[(size_t)(r + 1) * cols + k]) * xv;
                s2 += qbf16_to_f32(W[(size_t)(r + 2) * cols + k]) * xv;
                s3 += qbf16_to_f32(W[(size_t)(r + 3) * cols + k]) * xv;
            }
            Y[(size_t)r * B + j] = s0;       Y[(size_t)(r + 1) * B + j] = s1;
            Y[(size_t)(r + 2) * B + j] = s2; Y[(size_t)(r + 3) * B + j] = s3;
        }
    }
    for (; r + 1 < r1; r += 2) {
        const bfloat16_t *w0 = (const bfloat16_t *)(W + (size_t)r * cols);
        const bfloat16_t *w1 = (const bfloat16_t *)(W + (size_t)(r + 1) * cols);
        int j = 0;
        for (; j + 1 < B; j += 2) {
            const bfloat16_t *x0 = (const bfloat16_t *)(Xb + (size_t)j * cols);
            const bfloat16_t *x1 = (const bfloat16_t *)(Xb + (size_t)(j + 1) * cols);
            float32x4_t acc = vdupq_n_f32(0.0f);
            int k = 0;
            for (; k + 3 < cols; k += 4) {
                bfloat16x8_t a = vcombine_bf16(vld1_bf16(w0 + k), vld1_bf16(w1 + k));
                bfloat16x8_t b = vcombine_bf16(vld1_bf16(x0 + k), vld1_bf16(x1 + k));
                acc = vbfmmlaq_f32(acc, a, b);
            }
            float t[4]; vst1q_f32(t, acc);
            for (; k < cols; k++) {
                float wv0 = qbf16_to_f32(W[(size_t)r * cols + k]);
                float wv1 = qbf16_to_f32(W[(size_t)(r + 1) * cols + k]);
                float xv0 = qbf16_to_f32(Xb[(size_t)j * cols + k]);
                float xv1 = qbf16_to_f32(Xb[(size_t)(j + 1) * cols + k]);
                t[0] += wv0 * xv0; t[1] += wv0 * xv1; t[2] += wv1 * xv0; t[3] += wv1 * xv1;
            }
            Y[(size_t)r * B + j]           = t[0];
            Y[(size_t)r * B + j + 1]       = t[1];
            Y[(size_t)(r + 1) * B + j]     = t[2];
            Y[(size_t)(r + 1) * B + j + 1] = t[3];
        }
        for (; j < B; j++) {
            float s0 = 0.0f, s1 = 0.0f;
            for (int k = 0; k < cols; k++) {
                float xv = qbf16_to_f32(Xb[(size_t)j * cols + k]);
                s0 += qbf16_to_f32(W[(size_t)r * cols + k]) * xv;
                s1 += qbf16_to_f32(W[(size_t)(r + 1) * cols + k]) * xv;
            }
            Y[(size_t)r * B + j] = s0; Y[(size_t)(r + 1) * B + j] = s1;
        }
    }
    for (; r < r1; r++) {
        for (int j = 0; j < B; j++) {
            float s = 0.0f;
            for (int k = 0; k < cols; k++)
                s += qbf16_to_f32(W[(size_t)r * cols + k]) * qbf16_to_f32(Xb[(size_t)j * cols + k]);
            Y[(size_t)r * B + j] = s;
        }
    }
}
typedef struct { float *Y; const uint16_t *W; const uint16_t *Xb; int rows, cols, B; } bfmmla_ctx;
static void bfmmla_task(size_t tid, size_t nt, void *vc) {
    bfmmla_ctx *c = (bfmmla_ctx *)vc;
    int r0 = (int)(tid * (size_t)c->rows / nt), r1 = (int)((tid + 1) * (size_t)c->rows / nt);
    r0 &= ~3; if (tid + 1 < nt) r1 &= ~3;
    bf16_matmat_bfmmla_slice(c->Y, c->W, c->Xb, r0, r1, c->cols, c->B);
}
#endif

#if defined(__AMX_BF16__) && defined(__AMX_TILE__)

static void amx_pack_act_bf16(uint16_t *pXb, const uint16_t *Xb, int cols, int kpack, int B) {
    const int nchunk = kpack >> 5;
    const size_t cstride = (size_t)B * 2;
    for (int kc = 0; kc < nchunk; kc++) {
        uint16_t *dst = pXb + (size_t)kc * 32 * (size_t)B;
        for (int n = 0; n < B; n++) {
            const uint16_t *src = Xb + (size_t)n * cols + (size_t)kc * 32;
            for (int j = 0; j < 16; j++)
                memcpy(dst + (size_t)j * cstride + (size_t)n * 2, src + 2 * j, 2 * sizeof(uint16_t));
        }
    }
}

static void bf16_matmat_amx_slice(float *Y, const uint16_t *W, const uint8_t *pW,
                                  const uint16_t *pXb, const uint16_t *Xb,
                                  int r0, int r1, int cols, int B) {
    MMSTAT(QWEN_MMK_BF16_AMX, r1 - r0, cols, B);
    const int kfull   = cols & ~31;
    const int nchunk  = kfull >> 5;
    const int cstride = B * 4;
    const size_t wstride = (size_t)cols * sizeof(uint16_t);
    const int wblocks = cols >> 5;

    qwen_amx_tilecfg cfg;
    memset(&cfg, 0, sizeof cfg);
    cfg.palette_id = 1;
    cfg.rows[0] = 16; cfg.colsb[0] = (uint16_t)cstride;
    cfg.rows[1] = 16; cfg.colsb[1] = (uint16_t)cstride;
    cfg.rows[2] = 16; cfg.colsb[2] = 64;
    cfg.rows[3] = 16; cfg.colsb[3] = 64;
    cfg.rows[4] = 16; cfg.colsb[4] = (uint16_t)cstride;
    qwen_amx_prepare_config(&cfg, 0x10000u | (unsigned)cstride);

    float cbuf[2][16 * 16] __attribute__((aligned(64)));

    int r = r0;
    for (; r + 31 < r1; r += 32) {
        _tile_zero(0); _tile_zero(1);
        for (int kc = 0; kc < nchunk; kc++) {
            _tile_loadd(4, pXb + (size_t)kc * 32 * (size_t)B, cstride);
            const void *w0 = pW
                ? (const void *)(pW + ((size_t)(r >> 4) * wblocks + (size_t)kc) * 1024)
                : (const void *)(W + (size_t)r * cols + (size_t)kc * 32);
            const void *w1 = pW
                ? (const void *)(pW + ((size_t)((r >> 4) + 1) * wblocks + (size_t)kc) * 1024)
                : (const void *)(W + (size_t)(r + 16) * cols + (size_t)kc * 32);
            _tile_loadd(2, w0, pW ? 64 : wstride);
            _tile_loadd(3, w1, pW ? 64 : wstride);
            _tile_dpbf16ps(0, 2, 4);
            _tile_dpbf16ps(1, 3, 4);
        }
        _tile_stored(0, cbuf[0], cstride);
        _tile_stored(1, cbuf[1], cstride);
        for (int h = 0; h < 2; h++)
            for (int m = 0; m < 16; m++) {
                const int rr = r + h * 16 + m;
                const uint16_t *w = W + (size_t)rr * cols;
                for (int b = 0; b < B; b++) {
                    float acc = cbuf[h][m * B + b];
                    const uint16_t *xb = Xb + (size_t)b * cols;
                    for (int kk = kfull; kk < cols; kk++)
                        acc += bf16_to_f32(w[kk]) * bf16_to_f32(xb[kk]);
                    Y[(size_t)rr * B + b] = acc;
                }
            }
    }
    for (; r + 15 < r1; r += 16) {
        _tile_zero(0);
        for (int kc = 0; kc < nchunk; kc++) {
            _tile_loadd(4, pXb + (size_t)kc * 32 * (size_t)B, cstride);
            const void *w0 = pW
                ? (const void *)(pW + ((size_t)(r >> 4) * wblocks + (size_t)kc) * 1024)
                : (const void *)(W + (size_t)r * cols + (size_t)kc * 32);
            _tile_loadd(2, w0, pW ? 64 : wstride);
            _tile_dpbf16ps(0, 2, 4);
        }
        _tile_stored(0, cbuf[0], cstride);
        for (int m = 0; m < 16; m++) {
            const int rr = r + m;
            const uint16_t *w = W + (size_t)rr * cols;
            for (int b = 0; b < B; b++) {
                float acc = cbuf[0][m * B + b];
                const uint16_t *xb = Xb + (size_t)b * cols;
                for (int kk = kfull; kk < cols; kk++)
                    acc += bf16_to_f32(w[kk]) * bf16_to_f32(xb[kk]);
                Y[(size_t)rr * B + b] = acc;
            }
        }
    }
    qwen_amx_finish_config();

    for (; r < r1; r++) {
        const uint16_t *w = W + (size_t)r * cols;
        for (int b = 0; b < B; b++) {
            const uint16_t *xb = Xb + (size_t)b * cols;
            float acc = 0.0f;
            for (int k = 0; k < cols; k++) acc += bf16_to_f32(w[k]) * bf16_to_f32(xb[k]);
            Y[(size_t)r * B + b] = acc;
        }
    }
}

typedef struct {
    float *Y; const uint16_t *W; const uint8_t *pW;
    const uint16_t *pXb; const uint16_t *Xb;
    int rows, cols, B;
} bf16_amx_ctx;
static void bf16_amx_task(size_t tid, size_t nt, void *vc) {
    bf16_amx_ctx *c = (bf16_amx_ctx *)vc;
    int r0 = (int)(tid * (size_t)c->rows / nt);
    int r1 = (int)((tid + 1) * (size_t)c->rows / nt);
    r0 &= ~15; if (tid + 1 < nt) r1 &= ~15;
    int chunk = qwen_x86_nchunk(QWEN_MMK_BF16_AMX, c->B);
    if (chunk > 0) {
        for (int r = r0; r < r1; r += chunk) {
            int e = r + chunk < r1 ? r + chunk : r1;
            bf16_matmat_amx_slice(c->Y, c->W, c->pW, c->pXb, c->Xb,
                                  r, e, c->cols, c->B);
        }
    } else {
        bf16_matmat_amx_slice(c->Y, c->W, c->pW, c->pXb, c->Xb,
                              r0, r1, c->cols, c->B);
    }
}
#endif

#define QWEN_MM_SCRATCH(name, type)                                                      \
    static __thread type *g_mms_##name = NULL;                                           \
    static __thread size_t g_mms_cap_##name = 0;                                         \
    static type *mm_scratch_##name(size_t nelem) QWEN_MAYBE_UNUSED;                      \
    static type *mm_scratch_##name(size_t nelem) {                                       \
        size_t need = nelem * sizeof(type);                                              \
        if (need > g_mms_cap_##name) {                                                   \
            void *np = NULL;                                                             \
            if (posix_memalign(&np, 64, need) != 0) return NULL;                         \
            free(g_mms_##name);                                                          \
            g_mms_##name = (type *)np; g_mms_cap_##name = need;                          \
        }                                                                                \
        return g_mms_##name;                                                             \
    }
QWEN_MM_SCRATCH(qx,   int8_t)
QWEN_MM_SCRATCH(pack, int8_t)
QWEN_MM_SCRATCH(packb, uint16_t)
QWEN_MM_SCRATCH(sdcolf, float)
QWEN_MM_SCRATCH(sdcolq, int8_t)
QWEN_MM_SCRATCH(sdsa, float)
QWEN_MM_SCRATCH(corr, int)
QWEN_MM_SCRATCH(xb,   uint16_t)
QWEN_MM_SCRATCH(xcol, float)
QWEN_MM_SCRATCH(ycol, float)
QWEN_MM_SCRATCH(snk,  float)

void qwen_matmat_bf16(float *Y, const uint16_t *W, const float *X, int rows, int cols, int B) {
    qwen_census_op(QWEN_PATH_MATMAT_BF16, rows, cols, B);
    const int kt_on = qwen_kernel_timing_enabled();
    const double kt_t0 = kt_on ? qwen_mm_now_s() : 0.0;
    const int kt_B = B;
    if (g_qwen_matmat_bf16_hook) {
        g_qwen_matmat_bf16_hook(Y, W, X, rows, cols, B);
        goto qwen_matmat_bf16_timed_done;
    }
    if (B <= 0) return;
    if (kai_bf16_try(Y, W, X, rows, cols, B)) {
        MMSTAT(B > 1 ? QWEN_MMK_KLEIDI_BF16 : QWEN_MMK_KLEIDI_BF16_GEMV, rows, cols, B);
        goto qwen_matmat_bf16_timed_done;
    }
    if (qwen_q8r_matmul(Y, (const void *)W, X, rows, cols, B)) {
        MMSTAT(QWEN_MMK_Q8_REPACK_I8MM, rows, cols, B);
        goto qwen_matmat_bf16_timed_done;
    }
    if (B > 64) B = 64;
    int nt = g_n_threads;
#if defined(__AMX_BF16__) && defined(__AMX_TILE__)
    if (qwen_mm_use(QWEN_MMK_BF16_AMX, B, rows, cols) && qwen_amx_bf16_ready()) {
        const size_t kfull = (size_t)(cols & ~31);
        uint16_t *Xb  = mm_scratch_xb((size_t)B * cols);
        uint16_t *pXb = NULL;
        if (Xb) pXb = mm_scratch_packb(kfull * (size_t)B);
        if (Xb && pXb) {
            const uint8_t *pW = (const uint8_t *)qwen_amx_pack_weights(
                W, rows, cols, QWEN_AMX_WEIGHT_BF16);
            for (int b = 0; b < B; b++)
                for (int k = 0; k < cols; k++) {
                    uint32_t u; memcpy(&u, &X[(size_t)k * B + b], 4);
                    Xb[(size_t)b * cols + k] = (uint16_t)(u >> 16);
                }
            amx_pack_act_bf16(pXb, Xb, cols, (int)kfull, B);
            bf16_amx_ctx c = { Y, W, pW, pXb, Xb, rows, cols, B };
            if (nt > 1 && rows >= 256) {
                qwen_parallel((size_t)nt, bf16_amx_task, &c);
            } else {
                bf16_amx_task(0, 1, &c);
            }
            goto qwen_matmat_bf16_timed_done;
        }
    }
#endif
#if defined(__AVX512BF16__)
    if (B <= 16 && cols >= 32 && !qwen_bf16dot_disabled() &&
        qwen_mm_use(QWEN_MMK_BF16_AVX512, B, rows, cols)) {
        uint16_t *Xb = mm_scratch_packb((size_t)B * cols);
        if (Xb) {
            for (int b = 0; b < B; b++)
                for (int k = 0; k < cols; k++)
                    Xb[(size_t)b * cols + k] =
                        qwen_f32_to_bf16_scalar(X[(size_t)k * B + b]);
            bf16_avx512_ctx c = { Y, W, Xb, rows, cols, B };
            if (nt > 1 && rows >= 256) {
                qwen_parallel((size_t)nt, bf16_avx512_task, &c);
            } else {
                bf16_avx512_task(0, 1, &c);
            }
            goto qwen_matmat_bf16_timed_done;
        }
    }
#endif
#if defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC)
    {
        if (qwen_mm_use(QWEN_MMK_BF16_BFMMLA, B, rows, cols)) {
            uint16_t *Xb = mm_scratch_xb((size_t)B * cols);
            if (Xb) {
                for (int b = 0; b < B; b++)
                    for (int k = 0; k < cols; k++) {
                        uint32_t u; memcpy(&u, &X[(size_t)k * B + b], 4);
                        Xb[(size_t)b * cols + k] = (uint16_t)(u >> 16);
                    }
                if (nt > 1 && rows >= 256) {
                    bfmmla_ctx c = { Y, W, Xb, rows, cols, B };
                    qwen_parallel((size_t)nt, bfmmla_task, &c);
                } else {
                    bf16_matmat_bfmmla_slice(Y, W, Xb, 0, rows, cols, B);
                }
                goto qwen_matmat_bf16_timed_done;
            }
        }
    }
#endif
    if (nt > 1 && rows >= 256) {
        bf16_mm_ctx c = { Y, W, X, rows, cols, B };
        qwen_parallel((size_t)nt, bf16_mm_task, &c);
        goto qwen_matmat_bf16_timed_done;
    }
    bf16_matmat_slice(Y, W, X, 0, rows, cols, B);

qwen_matmat_bf16_timed_done:
    qwen_kernel_timing_note(QWEN_KT_BF16, kt_B, rows, cols, kt_t0);
}

/* Returns 1 when it handled the call, 0 when the caller must fall back to
 * qwen_matmat_bf16() with a [cols][B] buffer.  The guard mirrors the AVX-512
 * branch of qwen_matmat_bf16() AND the four dispatch steps that precede it;
 * the two must be kept in sync. */
int qwen_matmat_bf16_rows_usable(int rows, int cols, int B) {
#if defined(__AVX512BF16__)
    if (!qwen_prefill_rowpack_enabled()) return 0;
    if (g_qwen_matmat_bf16_hook) return 0;           /* GPU hook owns the call */
    if (qwen_kleidi_bf16_enabled()) return 0;        /* KleidiAI runs first    */
    if (qwen_q8r_enabled()) return 0;                /* q8 repack runs first   */
#if defined(__AMX_BF16__) && defined(__AMX_TILE__)
    if (qwen_mm_use(QWEN_MMK_BF16_AMX, B, rows, cols) && qwen_amx_bf16_ready())
        return 0;                                    /* AMX branch runs first  */
#endif
    return B >= 1 && B <= 16 && cols >= 32 && !qwen_bf16dot_disabled() &&
           qwen_mm_use(QWEN_MMK_BF16_AVX512, B, rows, cols);
#else
    (void)rows; (void)cols; (void)B; return 0;
#endif
}

/* [B][cols] f32 rows with stride ldx -> [B][cols] bf16, the layout the AVX-512
 * kernel consumes.  Depends only on (Xr, ldx, cols, B): callers that run several
 * projections over the same activation can build this once. */
void qwen_bf16_pack_rows(uint16_t *Xb, const float *Xr, int ldx, int cols, int B) {
    for (int b = 0; b < B; b++) {
        const float *xr = Xr + (size_t)b * (size_t)ldx;
        uint16_t *dst = Xb + (size_t)b * (size_t)cols;
#if defined(__AVX512BF16__)
        for (int k = 0; k < cols; k++) dst[k] = qwen_f32_to_bf16_scalar(xr[k]);
#else
        /* qwen_f32_to_bf16_scalar lives in the AVX-512-BF16 section; this entry is never
           selected elsewhere (qwen_matmat_bf16_rows_usable() returns 0) but it must link. */
        for (int k = 0; k < cols; k++) {
            uint32_t u; memcpy(&u, &xr[k], sizeof u);
            u += 0x7FFFu + ((u >> 16) & 1u);
            dst[k] = (uint16_t)(u >> 16);
        }
#endif
    }
}

void qwen_matmat_bf16_packed(float *Y, const uint16_t *W, const uint16_t *Xb,
                             int rows, int cols, int B) {
#if defined(__AVX512BF16__)
    qwen_census_op(QWEN_PATH_MATMAT_BF16_ROWS, rows, cols, B);
    MMSTAT(QWEN_MMK_BF16_AVX512, rows, cols, B);
    bf16_avx512_ctx c = { Y, W, (uint16_t *)Xb, rows, cols, B };
    const int nt = g_n_threads;
    if (nt > 1 && rows >= 256) qwen_parallel((size_t)nt, bf16_avx512_task, &c);
    else                       bf16_avx512_task(0, 1, &c);
#else
    (void)Y; (void)W; (void)Xb; (void)rows; (void)cols; (void)B;
#endif
}

/* Returns 1 when it handled the call, 0 when the caller must fall back to
 * qwen_matmat_bf16() with a [cols][B] buffer.  The guard mirrors the AVX-512
 * branch of qwen_matmat_bf16() AND the four dispatch steps that precede it;
 * the two must be kept in sync. */
int qwen_matmat_bf16_rows(float *Y, const uint16_t *W, const float *Xr,
                          int ldx, int rows, int cols, int B) {
#if defined(__AVX512BF16__)
    if (!qwen_matmat_bf16_rows_usable(rows, cols, B)) return 0;
    uint16_t *Xb = mm_scratch_packb((size_t)B * cols);
    if (!Xb) return 0;
    qwen_bf16_pack_rows(Xb, Xr, ldx, cols, B);
    qwen_matmat_bf16_packed(Y, W, Xb, rows, cols, B);
    return 1;
#else
    (void)Y; (void)W; (void)Xr; (void)ldx; (void)rows; (void)cols; (void)B;
    return 0;
#endif
}

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
/* 1, 5, 7 and 9..15 were reaching int8_matmat_generic, which measures 5-10x the fixed-width
   kernel on the same shape: 2.6 ms at B=8 against 14.5 at B=5, 25.4 at B=7 and 20-26 across
   9..15. This is the fallback twin, so it is what runs wherever no VNNI, AMX, SDOT or SMMLA
   matmat takes the call - and the widths it was missing are ordinary ones. */
DEFINE_MATMAT_INT8_FIXED_B(1)
DEFINE_MATMAT_INT8_FIXED_B(5)
DEFINE_MATMAT_INT8_FIXED_B(7)
DEFINE_MATMAT_INT8_FIXED_B(9)
DEFINE_MATMAT_INT8_FIXED_B(10)
DEFINE_MATMAT_INT8_FIXED_B(11)
DEFINE_MATMAT_INT8_FIXED_B(12)
DEFINE_MATMAT_INT8_FIXED_B(13)
DEFINE_MATMAT_INT8_FIXED_B(14)
DEFINE_MATMAT_INT8_FIXED_B(15)
DEFINE_MATMAT_INT8_FIXED_B(16)
#undef DEFINE_MATMAT_INT8_FIXED_B
static void int8_matmat_slice(float *Y, const int8_t *W, const float *scale,
                              const float *X, int r0, int r1, int cols, int B) {
    MMSTAT(QWEN_MMK_INT8_F32TWIN, r1 - r0, cols, B);
    if (qwen_matmat_stats_enabled() || qwen_census_enabled()) {
        int fixed = (B >= 1 && B <= 16);
        atomic_fetch_add_explicit(&g_mm_fixedw[1], fixed, memory_order_relaxed);
        atomic_fetch_add_explicit(&g_mm_generic[1], !fixed, memory_order_relaxed);
    }
    qwen_ftz_on();
    switch (B) {
        case 2:  int8_matmat_b2 (Y, W, scale, X, r0, r1, cols); return;
        case 3:  int8_matmat_b3 (Y, W, scale, X, r0, r1, cols); return;
        case 4:  int8_matmat_b4 (Y, W, scale, X, r0, r1, cols); return;
        case 6:  int8_matmat_b6 (Y, W, scale, X, r0, r1, cols); return;
        case 1:  int8_matmat_b1 (Y, W, scale, X, r0, r1, cols); return;
        case 5:  int8_matmat_b5 (Y, W, scale, X, r0, r1, cols); return;
        case 7:  int8_matmat_b7 (Y, W, scale, X, r0, r1, cols); return;
        case 8:  int8_matmat_b8 (Y, W, scale, X, r0, r1, cols); return;
        case 9:  int8_matmat_b9 (Y, W, scale, X, r0, r1, cols); return;
        case 10: int8_matmat_b10(Y, W, scale, X, r0, r1, cols); return;
        case 11: int8_matmat_b11(Y, W, scale, X, r0, r1, cols); return;
        case 12: int8_matmat_b12(Y, W, scale, X, r0, r1, cols); return;
        case 13: int8_matmat_b13(Y, W, scale, X, r0, r1, cols); return;
        case 14: int8_matmat_b14(Y, W, scale, X, r0, r1, cols); return;
        case 15: int8_matmat_b15(Y, W, scale, X, r0, r1, cols); return;
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
static void int8_matmat_sdot_slice(float *Y, const int8_t *W, const float *scale,
                                   const int8_t *qXt, const float *sx,
                                   int r0, int r1, int cols, int B) {
    MMSTAT(QWEN_MMK_INT8_SDOT, r1 - r0, cols, B);
    qwen_ftz_on();
    for (int r = r0; r < r1; r++) {
        const int8_t *w = W + (size_t)r * cols;
        int32x4_t acc[16];
        for (int b = 0; b < B; b++) acc[b] = vdupq_n_s32(0);
        int k = 0;
        for (; k + 15 < cols; k += 16) {
            int8x16_t wv = vld1q_s8(w + k);
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
#endif

/* The matmat input is [cols][B], so each activation column is strided by B. */
#if defined(__AVX512F__) && defined(__AVX512BW__)
static int qwen_vnni_col_quant_enabled(void) {
    static atomic_int enabled = -1;
    int v = atomic_load_explicit(&enabled, memory_order_relaxed);
    if (v < 0) {
        const char *e = getenv("QWEN_NO_VNNI_ACT_QUANT");
        v = !(e && e[0] == '1');
        atomic_store_explicit(&enabled, v, memory_order_relaxed);
    }
    return v;
}

/* |x| via an integer AND, not _mm512_andnot_ps: the float-domain logic ops are AVX512DQ,
 * and the plain AVX-512F profile (SIMD=avx512: F/BW/VL, no DQ, no VNNI) must build too.
 * Same bits, same kernel -- 0x7FFFFFFF clears the sign exactly as andnot(-0.0f, v) does. */
static float quantize_act_int8_col_avx512(int8_t *qb, const float *X,
                                         int cols, int B, int b) {
    const __m512i absmask = _mm512_set1_epi32(0x7FFFFFFF);
    const __m512i offsets = _mm512_setr_epi32(
        0 * B, 1 * B, 2 * B, 3 * B, 4 * B, 5 * B, 6 * B, 7 * B,
        8 * B, 9 * B, 10 * B, 11 * B, 12 * B, 13 * B, 14 * B, 15 * B);
    __m512 vmax = _mm512_setzero_ps();
    int k = 0;
    for (; k + 16 <= cols; k += 16) {
        const __m512i idx = _mm512_add_epi32(offsets, _mm512_set1_epi32(k * B));
        const __m512 v = B == 1
            ? _mm512_loadu_ps(X + b + k)
            : _mm512_i32gather_ps(idx, X + b, 4);
        vmax = _mm512_max_ps(vmax, _mm512_castsi512_ps(
            _mm512_and_si512(_mm512_castps_si512(v), absmask)));
    }
    float amax = _mm512_reduce_max_ps(vmax);
    for (; k < cols; k++) {
        const float a = fabsf(X[(size_t)k * B + b]);
        if (a > amax) amax = a;
    }
    if (amax == 0.0f) { memset(qb, 0, (size_t)cols); return 0.0f; }

    const __m512 inv = _mm512_set1_ps(127.0f / amax);
    k = 0;
    for (; k + 16 <= cols; k += 16) {
        const __m512i idx = _mm512_add_epi32(offsets, _mm512_set1_epi32(k * B));
        const __m512 v = B == 1
            ? _mm512_loadu_ps(X + b + k)
            : _mm512_i32gather_ps(idx, X + b, 4);
        __m512i q = _mm512_cvtps_epi32(_mm512_mul_ps(v, inv));
        q = _mm512_max_epi32(q, _mm512_set1_epi32(-128));
        q = _mm512_min_epi32(q, _mm512_set1_epi32(127));
        _mm_storeu_si128((__m128i *)(void *)(qb + k), _mm512_cvtsepi32_epi8(q));
    }
    for (; k < cols; k++) {
        const int v = (int)lrintf(X[(size_t)k * B + b] * (127.0f / amax));
        qb[k] = (int8_t)(v > 127 ? 127 : (v < -128 ? -128 : v));
    }
    return amax / 127.0f;
}
#endif

static float quantize_act_int8_col(int8_t *qb, const float *X, int cols, int B, int b) QWEN_MAYBE_UNUSED;
static float quantize_act_int8_col(int8_t *qb, const float *X, int cols, int B, int b) {
#if defined(__AVX512F__) && defined(__AVX512BW__)
    if (qwen_vnni_col_quant_enabled() && cols >= 16)
        return quantize_act_int8_col_avx512(qb, X, cols, B, b);
#endif
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
#if defined(__AMX_INT8__) && defined(__AMX_TILE__)
static void amx_pack_act_int8(int8_t *pXt, const int8_t *qXt, int cols, int kpack, int B) {
    const int nchunk = kpack >> 5;
    const size_t cstride = (size_t)B * 4;
    for (int kc = 0; kc < nchunk; kc++) {
        int8_t *dst = pXt + (size_t)kc * 32 * (size_t)B;
        for (int n = 0; n < B; n++) {
            const int8_t *src = qXt + (size_t)n * cols + (size_t)kc * 32;
            for (int j = 0; j < 8; j++)
                memcpy(dst + (size_t)j * cstride + (size_t)n * 4, src + j * 4, 4);
        }
    }
}

static void int8_matmat_amx_slice(float *Y, const int8_t *W, const uint8_t *pW,
                                  const float *scale, const int8_t *pXt,
                                  const int8_t *qXt, const float *sx,
                                  int r0, int r1, int cols, int B) {
    MMSTAT(QWEN_MMK_INT8_AMX, r1 - r0, cols, B);
    const int kfull   = cols & ~63;
    const int nchunk  = kfull >> 6;
    const int cstride = B * 4;
    const int wblocks = cols >> 6;

    qwen_amx_tilecfg cfg;
    memset(&cfg, 0, sizeof cfg);
    cfg.palette_id = 1;
    cfg.rows[0] = 16; cfg.colsb[0] = (uint16_t)cstride;
    cfg.rows[1] = 16; cfg.colsb[1] = (uint16_t)cstride;
    cfg.rows[2] = 16; cfg.colsb[2] = 64;
    cfg.rows[3] = 16; cfg.colsb[3] = 64;
    cfg.rows[4] = 16; cfg.colsb[4] = (uint16_t)cstride;
    qwen_amx_prepare_config(&cfg, 0x20000u | (unsigned)cstride);

    int32_t cbuf[2][16 * 16] __attribute__((aligned(64)));

    int r = r0;
    for (; r + 31 < r1; r += 32) {
        _tile_zero(0); _tile_zero(1);
        for (int kc = 0; kc < nchunk; kc++) {
            _tile_loadd(4, pXt + (size_t)kc * 64 * (size_t)B, cstride);
            const void *w0 = pW
                ? (const void *)(pW + ((size_t)(r >> 4) * wblocks + (size_t)kc) * 1024)
                : (const void *)(W + (size_t)r * cols + (size_t)kc * 64);
            const void *w1 = pW
                ? (const void *)(pW + ((size_t)((r >> 4) + 1) * wblocks + (size_t)kc) * 1024)
                : (const void *)(W + (size_t)(r + 16) * cols + (size_t)kc * 64);
            _tile_loadd(2, w0, pW ? 64 : cols);
            _tile_loadd(3, w1, pW ? 64 : cols);
            _tile_dpbssd(0, 2, 4);
            _tile_dpbssd(1, 3, 4);
        }
        _tile_stored(0, cbuf[0], cstride);
        _tile_stored(1, cbuf[1], cstride);
        for (int h = 0; h < 2; h++)
            for (int m = 0; m < 16; m++) {
                const int rr = r + h * 16 + m;
                const int8_t *w = W + (size_t)rr * cols;
                const float s = scale[rr];
                for (int b = 0; b < B; b++) {
                    int sum = cbuf[h][m * B + b];
                    const int8_t *qb = qXt + (size_t)b * cols;
                    for (int kk = kfull; kk < cols; kk++) sum += (int)w[kk] * (int)qb[kk];
                    Y[(size_t)rr * B + b] = (float)sum * s * sx[b];
                }
            }
    }
    for (; r + 15 < r1; r += 16) {
        _tile_zero(0);
        for (int kc = 0; kc < nchunk; kc++) {
            _tile_loadd(4, pXt + (size_t)kc * 64 * (size_t)B, cstride);
            const void *w0 = pW
                ? (const void *)(pW + ((size_t)(r >> 4) * wblocks + (size_t)kc) * 1024)
                : (const void *)(W + (size_t)r * cols + (size_t)kc * 64);
            _tile_loadd(2, w0, pW ? 64 : cols);
            _tile_dpbssd(0, 2, 4);
        }
        _tile_stored(0, cbuf[0], cstride);
        for (int m = 0; m < 16; m++) {
            const int rr = r + m;
            const int8_t *w = W + (size_t)rr * cols;
            const float s = scale[rr];
            for (int b = 0; b < B; b++) {
                int sum = cbuf[0][m * B + b];
                const int8_t *qb = qXt + (size_t)b * cols;
                for (int kk = kfull; kk < cols; kk++) sum += (int)w[kk] * (int)qb[kk];
                Y[(size_t)rr * B + b] = (float)sum * s * sx[b];
            }
        }
    }
    qwen_amx_finish_config();

    for (; r < r1; r++) {
        const int8_t *w = W + (size_t)r * cols;
        const float s = scale[r];
        for (int b = 0; b < B; b++) {
            const int8_t *qb = qXt + (size_t)b * cols;
            int32_t sum = 0;
            for (int k = 0; k < cols; k++) sum += (int)w[k] * (int)qb[k];
            Y[(size_t)r * B + b] = (float)sum * s * sx[b];
        }
    }
}

static int int8_matmat_amx_b32_run(float *Y, const int8_t *W, const float *scale,
                                   const float *X, int rows, int cols) {
    if (!Y || !W || !scale || !X || rows < 16 || (rows & 15) != 0 || cols < 64)
        return 0;

    const int B = 32;
    const int kfull = cols & ~63;
    const int nchunk = kfull >> 6;
    if (nchunk <= 0) return 0;

    int8_t *qXt = mm_scratch_qx((size_t)B * cols);
    int8_t *pXt = mm_scratch_pack((size_t)kfull * B);
    if (!qXt || !pXt) return 0;
    float sx[32];
    for (int b = 0; b < B; b++)
        sx[b] = quantize_act_int8_col(qXt + (size_t)b * cols, X, cols, B, b);
    amx_pack_act_int8(pXt, qXt, cols, kfull, 16);
    amx_pack_act_int8(pXt + (size_t)kfull * 16,
                      qXt + (size_t)16 * cols, cols, kfull, 16);

    const uint8_t *pW = (const uint8_t *)qwen_amx_pack_weights(
        W, rows, cols, QWEN_AMX_WEIGHT_INT8);
    const int wblocks = cols >> 6;
    qwen_amx_tilecfg cfg;
    memset(&cfg, 0, sizeof cfg);
    cfg.palette_id = 1;
    cfg.rows[0] = 16; cfg.colsb[0] = 64;
    cfg.rows[1] = 16; cfg.colsb[1] = 64;
    cfg.rows[2] = 16; cfg.colsb[2] = 64;
    cfg.rows[4] = 16; cfg.colsb[4] = 64;
    cfg.rows[5] = 16; cfg.colsb[5] = 64;
    qwen_amx_prepare_config(&cfg, 0x40000u);

    int32_t cbuf0[16 * 16] __attribute__((aligned(64)));
    int32_t cbuf1[16 * 16] __attribute__((aligned(64)));
    for (int r = 0; r < rows; r += 16) {
        _tile_zero(0);
        _tile_zero(1);
        for (int kc = 0; kc < nchunk; kc++) {
            const size_t act_offset = (size_t)kc * 64 * 16;
            _tile_loadd(4, pXt + act_offset, 64);
            _tile_loadd(5, pXt + (size_t)kfull * 16 + act_offset, 64);
            const void *w = pW
                ? (const void *)((const uint8_t *)pW +
                                 ((size_t)(r >> 4) * wblocks + (size_t)kc) * 1024)
                : (const void *)(W + (size_t)r * cols + (size_t)kc * 64);
            _tile_loadd(2, w, pW ? 64 : cols);
            _tile_dpbssd(0, 2, 4);
            _tile_dpbssd(1, 2, 5);
        }
        _tile_stored(0, cbuf0, 64);
        _tile_stored(1, cbuf1, 64);
        for (int m = 0; m < 16; m++) {
            const int rr = r + m;
            const int8_t *w = W + (size_t)rr * cols;
            const float s = scale[rr];
            for (int b = 0; b < B; b++) {
                int32_t sum = b < 16 ? cbuf0[m * 16 + b] : cbuf1[m * 16 + b - 16];
                const int8_t *qb = qXt + (size_t)b * cols;
                for (int kk = kfull; kk < cols; kk++) sum += (int)w[kk] * (int)qb[kk];
                Y[(size_t)rr * B + b] = (float)sum * s * sx[b];
            }
        }
    }
    qwen_amx_finish_config();
    return 1;
}

int qwen_matmat_int8_amx_b32(float *Y, const int8_t *W, const float *scale,
                             const float *X, int rows, int cols) {
    if (qwen_amx_b32_enabled() && qwen_amx_int8_ready() &&
        int8_matmat_amx_b32_run(Y, W, scale, X, rows, cols)) return 1;
    qwen_matmat_int8(Y, W, scale, X, rows, cols, 32);
    return 0;
}

typedef struct {
    float *Y; const int8_t *W; const uint8_t *pW; const float *scale;
    const int8_t *pXt; const int8_t *qXt; const float *sx;
    int rows, cols, B;
} int8_amx_ctx;
static void int8_amx_task(size_t tid, size_t nt, void *vc) {
    int8_amx_ctx *c = (int8_amx_ctx *)vc;
    int r0 = (int)(tid * (size_t)c->rows / nt);
    int r1 = (int)((tid + 1) * (size_t)c->rows / nt);
    r0 &= ~15; if (tid + 1 < nt) r1 &= ~15;
    int chunk = qwen_x86_nchunk(QWEN_MMK_INT8_AMX, c->B);
    if (chunk > 0) {
        for (int r = r0; r < r1; r += chunk) {
            int e = r + chunk < r1 ? r + chunk : r1;
            int8_matmat_amx_slice(c->Y, c->W, c->pW, c->scale, c->pXt, c->qXt, c->sx,
                                  r, e, c->cols, c->B);
        }
    } else {
        int8_matmat_amx_slice(c->Y, c->W, c->pW, c->scale, c->pXt, c->qXt, c->sx,
                              r0, r1, c->cols, c->B);
    }
}

static inline void amx_q4_unpack16(int8_t *stage, const q4_0_block_t *W,
                                   int nb, int r, int bl) {
    const __m128i lomask = _mm_set1_epi8(0x0F);
    for (int m = 0; m < 16; m++) {
        __m128i raw = _mm_loadu_si128((const __m128i *)W[(size_t)(r + m) * nb + bl].qs);
        __m128i lo  = _mm_and_si128(raw, lomask);
        __m128i hi  = _mm_and_si128(_mm_srli_epi16(raw, 4), lomask);
        _mm256_store_si256((__m256i *)(stage + m * 32),
                           _mm256_set_m128i(_mm_unpackhi_epi8(lo, hi),
                                            _mm_unpacklo_epi8(lo, hi)));
    }
}

static void q4_matmat_amx_slice(float *Y, const q4_0_block_t *W, const int8_t *pXt,
                                const int8_t *qXt, const float *sx, const int *corr,
                                int r0, int r1, int cols, int B) {
    MMSTAT(QWEN_MMK_Q4_AMX, r1 - r0, cols, B);
    const int nb      = cols / Q4_0_BLOCK_SIZE;
    const int cstride = B * 4;

    qwen_amx_tilecfg cfg;
    memset(&cfg, 0, sizeof cfg);
    cfg.palette_id = 1;
    cfg.rows[0] = 16; cfg.colsb[0] = (uint16_t)cstride;
    cfg.rows[1] = 16; cfg.colsb[1] = (uint16_t)cstride;
    cfg.rows[2] = 16; cfg.colsb[2] = 32;
    cfg.rows[3] = 16; cfg.colsb[3] = 32;
    cfg.rows[4] =  8; cfg.colsb[4] = (uint16_t)cstride;
    cfg.rows[5] =  8; cfg.colsb[5] = (uint16_t)cstride;
    qwen_amx_prepare_config(&cfg, 0x30000u | (unsigned)cstride);

    int8_t  stage[2][16 * 32] __attribute__((aligned(64)));
    int32_t cbuf[2][16 * 16]  __attribute__((aligned(64)));
    float   acc[16 * 16];

    int r = r0;
    for (; r + 15 < r1; r += 16) {
        for (int i = 0; i < 16 * B; i++) acc[i] = 0.0f;
        int bl = 0;
        for (; bl + 1 < nb; bl += 2) {
            amx_q4_unpack16(stage[0], W, nb, r, bl);
            _tile_zero(0);
            _tile_loadd(2, stage[0], 32);
            _tile_loadd(4, pXt + (size_t)bl * 32 * (size_t)B, cstride);
            _tile_dpbssd(0, 2, 4);
            amx_q4_unpack16(stage[1], W, nb, r, bl + 1);
            _tile_zero(1);
            _tile_loadd(3, stage[1], 32);
            _tile_loadd(5, pXt + (size_t)(bl + 1) * 32 * (size_t)B, cstride);
            _tile_dpbssd(1, 3, 5);
            _tile_stored(0, cbuf[0], cstride);
            _tile_stored(1, cbuf[1], cstride);
            for (int h = 0; h < 2; h++) {
                const int blh = bl + h;
                for (int m = 0; m < 16; m++) {
                    const float sc = qwen_f16_to_f32(W[(size_t)(r + m) * nb + blh].scale_f16);
                    for (int b = 0; b < B; b++)
                        acc[m * B + b] += sc * (float)(cbuf[h][m * B + b] + corr[(size_t)b * nb + blh]);
                }
            }
        }
        for (; bl < nb; bl++) {
            amx_q4_unpack16(stage[0], W, nb, r, bl);
            _tile_zero(0);
            _tile_loadd(2, stage[0], 32);
            _tile_loadd(4, pXt + (size_t)bl * 32 * (size_t)B, cstride);
            _tile_dpbssd(0, 2, 4);
            _tile_stored(0, cbuf[0], cstride);
            for (int m = 0; m < 16; m++) {
                const float sc = qwen_f16_to_f32(W[(size_t)(r + m) * nb + bl].scale_f16);
                for (int b = 0; b < B; b++)
                    acc[m * B + b] += sc * (float)(cbuf[0][m * B + b] + corr[(size_t)b * nb + bl]);
            }
        }
        for (int m = 0; m < 16; m++)
            for (int b = 0; b < B; b++)
                Y[(size_t)(r + m) * B + b] = acc[m * B + b] * sx[b];
    }
    qwen_amx_finish_config();

    for (; r < r1; r++) {
        const q4_0_block_t *wr = W + (size_t)r * nb;
        for (int b = 0; b < B; b++) {
            const int8_t *xb = qXt + (size_t)b * cols;
            float f = 0.0f;
            for (int bl = 0; bl < nb; bl++) {
                const uint8_t *q = wr[bl].qs;
                const int8_t *x = xb + (size_t)bl * Q4_0_BLOCK_SIZE;
                int t = 0;
                for (int i = 0; i < 16; i++)
                    t += (q[i] & 0x0F) * x[2 * i] + (q[i] >> 4) * x[2 * i + 1];
                f += qwen_f16_to_f32(wr[bl].scale_f16) * (float)(t + corr[(size_t)b * nb + bl]);
            }
            Y[(size_t)r * B + b] = f * sx[b];
        }
    }
}
typedef struct {
    float *Y; const q4_0_block_t *W; const int8_t *pXt; const int8_t *qXt;
    const float *sx; const int *corr; int rows, cols, B;
} q4_amx_ctx;
static void q4_amx_task(size_t tid, size_t nt, void *vc) {
    q4_amx_ctx *c = (q4_amx_ctx *)vc;
    int r0 = (int)(tid * (size_t)c->rows / nt);
    int r1 = (int)((tid + 1) * (size_t)c->rows / nt);
    r0 &= ~15; if (tid + 1 < nt) r1 &= ~15;
    q4_matmat_amx_slice(c->Y, c->W, c->pXt, c->qXt, c->sx, c->corr, r0, r1, c->cols, c->B);
}

#endif

#if !defined(__AMX_INT8__) || !defined(__AMX_TILE__)
int qwen_matmat_int8_amx_b32(float *Y, const int8_t *W, const float *scale,
                             const float *X, int rows, int cols) {
    qwen_matmat_int8(Y, W, scale, X, rows, cols, 32);
    return 0;
}
#endif

#if defined(__AVX512VNNI__)
#if defined(__GNUC__) || defined(__clang__)
#define QWEN_VNNI_NOINLINE __attribute__((noinline))
#else
#define QWEN_VNNI_NOINLINE
#endif

static const int32_t *qwen_vnni_row_sums(const int8_t *W, int rows, int cols);

enum { QWEN_VNNI_WEIGHT_CACHE_MAX = 512 };
typedef struct {
    const int8_t *weights;
    int rows;
    int cols;
    size_t bytes;
    int8_t *packed;
} qwen_vnni_weight_entry_t;

static qwen_vnni_weight_entry_t g_vnni_weights[QWEN_VNNI_WEIGHT_CACHE_MAX];
static atomic_int g_vnni_weight_count;
static pthread_mutex_t g_vnni_weight_mu = PTHREAD_MUTEX_INITIALIZER;

static int qwen_vnni_prepack_enabled(void) {
    const char *e = getenv("QWEN_VNNI_PREPACK");
    return e && (e[0] == '1' || !strcmp(e, "all") || !strcmp(e, "cp") ||
                 !strcmp(e, "talker"));
}

static int8_t *qwen_vnni_pack_n16_k4(const int8_t *source, int rows, int cols) {
    if (!source || rows < 16 || (rows & 15) || cols <= 0 || (cols & 63)) return NULL;
    const size_t bytes = (size_t)rows * (size_t)cols;
    int8_t *packed = (int8_t *)aligned_malloc(bytes);
    if (!packed) return NULL;

    size_t at = 0;
    for (int r0 = 0; r0 < rows; r0 += 16) {
        for (int k = 0; k < cols; k += 4) {
            for (int r = 0; r < 16; r++) {
                memcpy(packed + at, source + (size_t)(r0 + r) * cols + k, 4);
                at += 4;
            }
        }
    }
    return packed;
}

int qwen_vnni_prepack_weight(const int8_t *source, int rows, int cols) {
    if (!qwen_vnni_prepack_enabled() || !source || rows < 16 || (rows & 15) ||
        cols <= 0 || (cols & 63)) return 0;

    pthread_mutex_lock(&g_vnni_weight_mu);
    int n = atomic_load_explicit(&g_vnni_weight_count, memory_order_relaxed);
    for (int i = 0; i < n; i++) {
        qwen_vnni_weight_entry_t *e = &g_vnni_weights[i];
        if (e->weights == source && e->rows == rows && e->cols == cols) {
            pthread_mutex_unlock(&g_vnni_weight_mu);
            return 1;
        }
    }
    if (n >= QWEN_VNNI_WEIGHT_CACHE_MAX) {
        pthread_mutex_unlock(&g_vnni_weight_mu);
        return 0;
    }

    int8_t *packed = qwen_vnni_pack_n16_k4(source, rows, cols);
    if (!packed) {
        pthread_mutex_unlock(&g_vnni_weight_mu);
        return 0;
    }
    g_vnni_weights[n].weights = source;
    g_vnni_weights[n].rows = rows;
    g_vnni_weights[n].cols = cols;
    g_vnni_weights[n].bytes = (size_t)rows * (size_t)cols;
    g_vnni_weights[n].packed = packed;
    atomic_store_explicit(&g_vnni_weight_count, n + 1, memory_order_release);
    pthread_mutex_unlock(&g_vnni_weight_mu);
    (void)qwen_vnni_row_sums(source, rows, cols);
    return 1;
}

static const int8_t *qwen_vnni_packed_lookup(const int8_t *source, int rows, int cols) {
    if (!qwen_vnni_prepack_enabled() || !source || rows < 16 || (rows & 15) ||
        cols <= 0 || (cols & 63)) return NULL;
    int n = atomic_load_explicit(&g_vnni_weight_count, memory_order_acquire);
    for (int i = 0; i < n; i++) {
        const qwen_vnni_weight_entry_t *e = &g_vnni_weights[i];
        if (e->weights == source && e->rows == rows && e->cols == cols)
            return e->packed;
    }
    return NULL;
}

void qwen_vnni_prepack_stats(int *n_packed, size_t *bytes) {
    int n = 0;
    size_t total = 0;
    pthread_mutex_lock(&g_vnni_weight_mu);
    n = atomic_load_explicit(&g_vnni_weight_count, memory_order_relaxed);
    for (int i = 0; i < n; i++) total += g_vnni_weights[i].bytes;
    pthread_mutex_unlock(&g_vnni_weight_mu);
    if (n_packed) *n_packed = n;
    if (bytes) *bytes = total;
}

void qwen_vnni_weight_cache_reset(void) {
    pthread_mutex_lock(&g_vnni_weight_mu);
    int n = atomic_load_explicit(&g_vnni_weight_count, memory_order_relaxed);
    for (int i = 0; i < n; i++) {
        free(g_vnni_weights[i].packed);
        g_vnni_weights[i].weights = NULL;
        g_vnni_weights[i].rows = 0;
        g_vnni_weights[i].cols = 0;
        g_vnni_weights[i].bytes = 0;
        g_vnni_weights[i].packed = NULL;
    }
    atomic_store_explicit(&g_vnni_weight_count, 0, memory_order_release);
    pthread_mutex_unlock(&g_vnni_weight_mu);
}

static void int8_matmat_vnni_row(float *Y, const int8_t *W, const float *scale,
                                 const int8_t *qXt, const float *sx,
                                 const int32_t *row_sums,
                                 int r, int cols, int B) {
    const __m512i ones = _mm512_set1_epi8(1);
    const __m512i v128 = _mm512_set1_epi8((char)128);
    const int8_t *w = W + (size_t)r * cols;
    __m512i acc[16], ws = _mm512_setzero_si512();
    for (int b = 0; b < B; b++) acc[b] = _mm512_setzero_si512();
    int k = 0;
    for (; k + 64 <= cols; k += 64) {
        __m512i wv = _mm512_loadu_si512((const void *)(w + k));
        if (!row_sums) ws = _mm512_dpbusd_epi32(ws, ones, wv);
        for (int b = 0; b < B; b++) {
            __m512i ua = _mm512_add_epi8(
                _mm512_loadu_si512((const void *)(qXt + (size_t)b * cols + k)), v128);
            acc[b] = _mm512_dpbusd_epi32(acc[b], ua, wv);
        }
    }
    int sw = row_sums ? row_sums[r] : _mm512_reduce_add_epi32(ws);
    for (int b = 0; b < B; b++) {
        int sum = _mm512_reduce_add_epi32(acc[b]) - 128 * sw;
        const int8_t *qb = qXt + (size_t)b * cols;
        for (int kk = k; kk < cols; kk++) sum += (int)w[kk] * (int)qb[kk];
        Y[(size_t)r * B + b] = (float)sum * scale[r] * sx[b];
    }
}

static inline void int8_matmat_vnni_store(float *Y, const int8_t *W, const float *scale,
                                          const int8_t *qXt, const float *sx,
                                          int r, int b, int cols, int k,
                                          __m512i acc, int sw, int B) {
    const int8_t *w = W + (size_t)r * cols;
    int sum = _mm512_reduce_add_epi32(acc) - 128 * sw;
    const int8_t *qb = qXt + (size_t)b * cols;
    for (int kk = k; kk < cols; kk++) sum += (int)w[kk] * (int)qb[kk];
    Y[(size_t)r * B + b] = (float)sum * scale[r] * sx[b];
}

static inline __m512i qwen_vnni_broadcast_q4(const int8_t *x) {
    uint32_t q;
    memcpy(&q, x, sizeof(q));
    return _mm512_set1_epi32((int)q);
}

/* The packed layout is 16 output rows x 4 K values per 64-byte block. */
static void int8_matmat_vnni_packed_slice(float *Y, const int8_t *W,
                                          const int8_t *pW, const float *scale,
                                          const int8_t *qXt, const float *sx,
                                          const int32_t *row_sums,
                                          int r0, int r1, int rows, int cols, int B) {
    qwen_census_op(QWEN_PATH_MATMAT_INT8_VNNI_PACKED_SLICE, r1 - r0, cols, B);
    const __m512i v128 = _mm512_set1_epi8((char)128);
    int start = (r0 + 15) & ~15;
    int end = r1 & ~15;
    if (start > r1) start = r1;
    if (end < start) end = start;

    for (int r = r0; r < start; r++)
        int8_matmat_vnni_row(Y, W, scale, qXt, sx, row_sums, r, cols, B);

    for (int r0b = start; r0b < end; r0b += 16) {
        const int8_t *p = pW + (size_t)(r0b / 16) * (size_t)cols * 16;
        __m512i acc[8];
        for (int b = 0; b < B; b++) acc[b] = _mm512_setzero_si512();

        for (int k = 0; k < cols; k += 4) {
            const __m512i wv = _mm512_loadu_si512((const void *)p);
            p += 64;
            for (int b = 0; b < B; b++) {
                const __m512i ua = _mm512_add_epi8(
                    qwen_vnni_broadcast_q4(qXt + (size_t)b * cols + k), v128);
                acc[b] = _mm512_dpbusd_epi32(acc[b], ua, wv);
            }
        }

        int32_t sums[8][16];
        for (int b = 0; b < B; b++) _mm512_storeu_si512(sums[b], acc[b]);
        for (int r = 0; r < 16; r++) {
            const int rr = r0b + r;
            const int correction = 128 * row_sums[rr];
            const float sr = scale[rr];
            float *dst = Y + (size_t)rr * B;
            for (int b = 0; b < B; b++)
                dst[b] = (float)(sums[b][r] - correction) * sr * sx[b];
        }
    }

    for (int r = end; r < r1; r++)
        int8_matmat_vnni_row(Y, W, scale, qXt, sx, row_sums, r, cols, B);
    (void)rows;
}

static void int8_matmat_vnni_tile_m4n4(float *Y, const int8_t *W, const float *scale,
                                       const int8_t *qXt, const float *sx,
                                       const int32_t *row_sums,
                                       int row0, int b0, int nb, int cols,
                                       int B) QWEN_VNNI_NOINLINE;
static void int8_matmat_vnni_tile_m4n4(float *Y, const int8_t *W, const float *scale,
                                       const int8_t *qXt, const float *sx,
                                       const int32_t *row_sums,
                                       int row0, int b0, int nb, int cols,
                                       int B) {
    const __m512i ones = _mm512_set1_epi8(1);
    const __m512i v128 = _mm512_set1_epi8((char)128);
    __m512i a00 = _mm512_setzero_si512(), a01 = a00, a02 = a00, a03 = a00;
    __m512i a10 = a00, a11 = a00, a12 = a00, a13 = a00;
    __m512i a20 = a00, a21 = a00, a22 = a00, a23 = a00;
    __m512i a30 = a00, a31 = a00, a32 = a00, a33 = a00;
    __m512i ws0 = a00, ws1 = a00, ws2 = a00, ws3 = a00;
    int k = 0;
    for (; k + 64 <= cols; k += 64) {
        __m512i ua0 = _mm512_add_epi8(
            _mm512_loadu_si512((const void *)(qXt + (size_t)(b0 + 0) * cols + k)), v128);
        __m512i ua1 = _mm512_setzero_si512(), ua2 = ua1, ua3 = ua1;
        if (nb > 1) ua1 = _mm512_add_epi8(
            _mm512_loadu_si512((const void *)(qXt + (size_t)(b0 + 1) * cols + k)), v128);
        if (nb > 2) ua2 = _mm512_add_epi8(
            _mm512_loadu_si512((const void *)(qXt + (size_t)(b0 + 2) * cols + k)), v128);
        if (nb > 3) ua3 = _mm512_add_epi8(
            _mm512_loadu_si512((const void *)(qXt + (size_t)(b0 + 3) * cols + k)), v128);

        __m512i wv = _mm512_loadu_si512((const void *)(W + (size_t)(row0 + 0) * cols + k));
        if (!row_sums) ws0 = _mm512_dpbusd_epi32(ws0, ones, wv);
        a00 = _mm512_dpbusd_epi32(a00, ua0, wv);
        if (nb > 1) a01 = _mm512_dpbusd_epi32(a01, ua1, wv);
        if (nb > 2) a02 = _mm512_dpbusd_epi32(a02, ua2, wv);
        if (nb > 3) a03 = _mm512_dpbusd_epi32(a03, ua3, wv);

        wv = _mm512_loadu_si512((const void *)(W + (size_t)(row0 + 1) * cols + k));
        if (!row_sums) ws1 = _mm512_dpbusd_epi32(ws1, ones, wv);
        a10 = _mm512_dpbusd_epi32(a10, ua0, wv);
        if (nb > 1) a11 = _mm512_dpbusd_epi32(a11, ua1, wv);
        if (nb > 2) a12 = _mm512_dpbusd_epi32(a12, ua2, wv);
        if (nb > 3) a13 = _mm512_dpbusd_epi32(a13, ua3, wv);

        wv = _mm512_loadu_si512((const void *)(W + (size_t)(row0 + 2) * cols + k));
        if (!row_sums) ws2 = _mm512_dpbusd_epi32(ws2, ones, wv);
        a20 = _mm512_dpbusd_epi32(a20, ua0, wv);
        if (nb > 1) a21 = _mm512_dpbusd_epi32(a21, ua1, wv);
        if (nb > 2) a22 = _mm512_dpbusd_epi32(a22, ua2, wv);
        if (nb > 3) a23 = _mm512_dpbusd_epi32(a23, ua3, wv);

        wv = _mm512_loadu_si512((const void *)(W + (size_t)(row0 + 3) * cols + k));
        if (!row_sums) ws3 = _mm512_dpbusd_epi32(ws3, ones, wv);
        a30 = _mm512_dpbusd_epi32(a30, ua0, wv);
        if (nb > 1) a31 = _mm512_dpbusd_epi32(a31, ua1, wv);
        if (nb > 2) a32 = _mm512_dpbusd_epi32(a32, ua2, wv);
        if (nb > 3) a33 = _mm512_dpbusd_epi32(a33, ua3, wv);
    }
    const int sw0 = row_sums ? row_sums[row0 + 0] : _mm512_reduce_add_epi32(ws0);
    const int sw1 = row_sums ? row_sums[row0 + 1] : _mm512_reduce_add_epi32(ws1);
    const int sw2 = row_sums ? row_sums[row0 + 2] : _mm512_reduce_add_epi32(ws2);
    const int sw3 = row_sums ? row_sums[row0 + 3] : _mm512_reduce_add_epi32(ws3);
    int8_matmat_vnni_store(Y, W, scale, qXt, sx, row0 + 0, b0 + 0, cols, k, a00, sw0, B);
    if (nb > 1) int8_matmat_vnni_store(Y, W, scale, qXt, sx, row0 + 0, b0 + 1, cols, k, a01, sw0, B);
    if (nb > 2) int8_matmat_vnni_store(Y, W, scale, qXt, sx, row0 + 0, b0 + 2, cols, k, a02, sw0, B);
    if (nb > 3) int8_matmat_vnni_store(Y, W, scale, qXt, sx, row0 + 0, b0 + 3, cols, k, a03, sw0, B);
    int8_matmat_vnni_store(Y, W, scale, qXt, sx, row0 + 1, b0 + 0, cols, k, a10, sw1, B);
    if (nb > 1) int8_matmat_vnni_store(Y, W, scale, qXt, sx, row0 + 1, b0 + 1, cols, k, a11, sw1, B);
    if (nb > 2) int8_matmat_vnni_store(Y, W, scale, qXt, sx, row0 + 1, b0 + 2, cols, k, a12, sw1, B);
    if (nb > 3) int8_matmat_vnni_store(Y, W, scale, qXt, sx, row0 + 1, b0 + 3, cols, k, a13, sw1, B);
    int8_matmat_vnni_store(Y, W, scale, qXt, sx, row0 + 2, b0 + 0, cols, k, a20, sw2, B);
    if (nb > 1) int8_matmat_vnni_store(Y, W, scale, qXt, sx, row0 + 2, b0 + 1, cols, k, a21, sw2, B);
    if (nb > 2) int8_matmat_vnni_store(Y, W, scale, qXt, sx, row0 + 2, b0 + 2, cols, k, a22, sw2, B);
    if (nb > 3) int8_matmat_vnni_store(Y, W, scale, qXt, sx, row0 + 2, b0 + 3, cols, k, a23, sw2, B);
    int8_matmat_vnni_store(Y, W, scale, qXt, sx, row0 + 3, b0 + 0, cols, k, a30, sw3, B);
    if (nb > 1) int8_matmat_vnni_store(Y, W, scale, qXt, sx, row0 + 3, b0 + 1, cols, k, a31, sw3, B);
    if (nb > 2) int8_matmat_vnni_store(Y, W, scale, qXt, sx, row0 + 3, b0 + 2, cols, k, a32, sw3, B);
    if (nb > 3) int8_matmat_vnni_store(Y, W, scale, qXt, sx, row0 + 3, b0 + 3, cols, k, a33, sw3, B);
}

static void int8_matmat_vnni_tile_m2n4(float *Y, const int8_t *W, const float *scale,
                                       const int8_t *qXt, const float *sx,
                                       const int32_t *row_sums,
                                       int row0, int b0, int nb, int cols,
                                       int B) QWEN_VNNI_NOINLINE;
static void int8_matmat_vnni_tile_m2n4(float *Y, const int8_t *W, const float *scale,
                                       const int8_t *qXt, const float *sx,
                                       const int32_t *row_sums,
                                       int row0, int b0, int nb, int cols,
                                       int B) {
    const __m512i ones = _mm512_set1_epi8(1);
    const __m512i v128 = _mm512_set1_epi8((char)128);
    __m512i a00 = _mm512_setzero_si512(), a01 = a00, a02 = a00, a03 = a00;
    __m512i a10 = a00, a11 = a00, a12 = a00, a13 = a00;
    __m512i ws0 = a00, ws1 = a00;
    int k = 0;
    for (; k + 64 <= cols; k += 64) {
        __m512i ua0 = _mm512_add_epi8(
            _mm512_loadu_si512((const void *)(qXt + (size_t)(b0 + 0) * cols + k)), v128);
        __m512i ua1 = _mm512_setzero_si512(), ua2 = ua1, ua3 = ua1;
        if (nb > 1) ua1 = _mm512_add_epi8(
            _mm512_loadu_si512((const void *)(qXt + (size_t)(b0 + 1) * cols + k)), v128);
        if (nb > 2) ua2 = _mm512_add_epi8(
            _mm512_loadu_si512((const void *)(qXt + (size_t)(b0 + 2) * cols + k)), v128);
        if (nb > 3) ua3 = _mm512_add_epi8(
            _mm512_loadu_si512((const void *)(qXt + (size_t)(b0 + 3) * cols + k)), v128);

        __m512i wv = _mm512_loadu_si512((const void *)(W + (size_t)(row0 + 0) * cols + k));
        if (!row_sums) ws0 = _mm512_dpbusd_epi32(ws0, ones, wv);
        a00 = _mm512_dpbusd_epi32(a00, ua0, wv);
        if (nb > 1) a01 = _mm512_dpbusd_epi32(a01, ua1, wv);
        if (nb > 2) a02 = _mm512_dpbusd_epi32(a02, ua2, wv);
        if (nb > 3) a03 = _mm512_dpbusd_epi32(a03, ua3, wv);

        wv = _mm512_loadu_si512((const void *)(W + (size_t)(row0 + 1) * cols + k));
        if (!row_sums) ws1 = _mm512_dpbusd_epi32(ws1, ones, wv);
        a10 = _mm512_dpbusd_epi32(a10, ua0, wv);
        if (nb > 1) a11 = _mm512_dpbusd_epi32(a11, ua1, wv);
        if (nb > 2) a12 = _mm512_dpbusd_epi32(a12, ua2, wv);
        if (nb > 3) a13 = _mm512_dpbusd_epi32(a13, ua3, wv);
    }
    const int sw0 = row_sums ? row_sums[row0 + 0] : _mm512_reduce_add_epi32(ws0);
    const int sw1 = row_sums ? row_sums[row0 + 1] : _mm512_reduce_add_epi32(ws1);
    int8_matmat_vnni_store(Y, W, scale, qXt, sx, row0 + 0, b0 + 0, cols, k, a00, sw0, B);
    if (nb > 1) int8_matmat_vnni_store(Y, W, scale, qXt, sx, row0 + 0, b0 + 1, cols, k, a01, sw0, B);
    if (nb > 2) int8_matmat_vnni_store(Y, W, scale, qXt, sx, row0 + 0, b0 + 2, cols, k, a02, sw0, B);
    if (nb > 3) int8_matmat_vnni_store(Y, W, scale, qXt, sx, row0 + 0, b0 + 3, cols, k, a03, sw0, B);
    int8_matmat_vnni_store(Y, W, scale, qXt, sx, row0 + 1, b0 + 0, cols, k, a10, sw1, B);
    if (nb > 1) int8_matmat_vnni_store(Y, W, scale, qXt, sx, row0 + 1, b0 + 1, cols, k, a11, sw1, B);
    if (nb > 2) int8_matmat_vnni_store(Y, W, scale, qXt, sx, row0 + 1, b0 + 2, cols, k, a12, sw1, B);
    if (nb > 3) int8_matmat_vnni_store(Y, W, scale, qXt, sx, row0 + 1, b0 + 3, cols, k, a13, sw1, B);
}

static void int8_matmat_vnni_tile_m2n8(float *Y, const int8_t *W, const float *scale,
                                       const int8_t *qXt, const float *sx,
                                       const int32_t *row_sums,
                                       int row0, int cols, int B) QWEN_VNNI_NOINLINE;
static void int8_matmat_vnni_tile_m2n8(float *Y, const int8_t *W, const float *scale,
                                       const int8_t *qXt, const float *sx,
                                       const int32_t *row_sums,
                                       int row0, int cols, int B) {
    const __m512i ones = _mm512_set1_epi8(1);
    const __m512i v128 = _mm512_set1_epi8((char)128);
    __m512i a00 = _mm512_setzero_si512(), a01 = a00, a02 = a00, a03 = a00;
    __m512i a04 = a00, a05 = a00, a06 = a00, a07 = a00;
    __m512i a10 = a00, a11 = a00, a12 = a00, a13 = a00;
    __m512i a14 = a00, a15 = a00, a16 = a00, a17 = a00;
    __m512i ws0 = a00, ws1 = a00;
    int k = 0;
    for (; k + 64 <= cols; k += 64) {
        __m512i ua0 = _mm512_add_epi8(
            _mm512_loadu_si512((const void *)(qXt + (size_t)0 * cols + k)), v128);
        __m512i ua1 = _mm512_add_epi8(
            _mm512_loadu_si512((const void *)(qXt + (size_t)1 * cols + k)), v128);
        __m512i ua2 = _mm512_add_epi8(
            _mm512_loadu_si512((const void *)(qXt + (size_t)2 * cols + k)), v128);
        __m512i ua3 = _mm512_add_epi8(
            _mm512_loadu_si512((const void *)(qXt + (size_t)3 * cols + k)), v128);
        __m512i ua4 = _mm512_add_epi8(
            _mm512_loadu_si512((const void *)(qXt + (size_t)4 * cols + k)), v128);
        __m512i ua5 = _mm512_add_epi8(
            _mm512_loadu_si512((const void *)(qXt + (size_t)5 * cols + k)), v128);
        __m512i ua6 = _mm512_add_epi8(
            _mm512_loadu_si512((const void *)(qXt + (size_t)6 * cols + k)), v128);
        __m512i ua7 = _mm512_add_epi8(
            _mm512_loadu_si512((const void *)(qXt + (size_t)7 * cols + k)), v128);

        __m512i wv = _mm512_loadu_si512((const void *)(W + (size_t)(row0 + 0) * cols + k));
        if (!row_sums) ws0 = _mm512_dpbusd_epi32(ws0, ones, wv);
        a00 = _mm512_dpbusd_epi32(a00, ua0, wv);
        a01 = _mm512_dpbusd_epi32(a01, ua1, wv);
        a02 = _mm512_dpbusd_epi32(a02, ua2, wv);
        a03 = _mm512_dpbusd_epi32(a03, ua3, wv);
        a04 = _mm512_dpbusd_epi32(a04, ua4, wv);
        a05 = _mm512_dpbusd_epi32(a05, ua5, wv);
        a06 = _mm512_dpbusd_epi32(a06, ua6, wv);
        a07 = _mm512_dpbusd_epi32(a07, ua7, wv);

        wv = _mm512_loadu_si512((const void *)(W + (size_t)(row0 + 1) * cols + k));
        if (!row_sums) ws1 = _mm512_dpbusd_epi32(ws1, ones, wv);
        a10 = _mm512_dpbusd_epi32(a10, ua0, wv);
        a11 = _mm512_dpbusd_epi32(a11, ua1, wv);
        a12 = _mm512_dpbusd_epi32(a12, ua2, wv);
        a13 = _mm512_dpbusd_epi32(a13, ua3, wv);
        a14 = _mm512_dpbusd_epi32(a14, ua4, wv);
        a15 = _mm512_dpbusd_epi32(a15, ua5, wv);
        a16 = _mm512_dpbusd_epi32(a16, ua6, wv);
        a17 = _mm512_dpbusd_epi32(a17, ua7, wv);
    }
    const int sw0 = row_sums ? row_sums[row0 + 0] : _mm512_reduce_add_epi32(ws0);
    const int sw1 = row_sums ? row_sums[row0 + 1] : _mm512_reduce_add_epi32(ws1);
    int8_matmat_vnni_store(Y, W, scale, qXt, sx, row0 + 0, 0, cols, k, a00, sw0, B);
    int8_matmat_vnni_store(Y, W, scale, qXt, sx, row0 + 0, 1, cols, k, a01, sw0, B);
    int8_matmat_vnni_store(Y, W, scale, qXt, sx, row0 + 0, 2, cols, k, a02, sw0, B);
    int8_matmat_vnni_store(Y, W, scale, qXt, sx, row0 + 0, 3, cols, k, a03, sw0, B);
    int8_matmat_vnni_store(Y, W, scale, qXt, sx, row0 + 0, 4, cols, k, a04, sw0, B);
    int8_matmat_vnni_store(Y, W, scale, qXt, sx, row0 + 0, 5, cols, k, a05, sw0, B);
    int8_matmat_vnni_store(Y, W, scale, qXt, sx, row0 + 0, 6, cols, k, a06, sw0, B);
    int8_matmat_vnni_store(Y, W, scale, qXt, sx, row0 + 0, 7, cols, k, a07, sw0, B);
    int8_matmat_vnni_store(Y, W, scale, qXt, sx, row0 + 1, 0, cols, k, a10, sw1, B);
    int8_matmat_vnni_store(Y, W, scale, qXt, sx, row0 + 1, 1, cols, k, a11, sw1, B);
    int8_matmat_vnni_store(Y, W, scale, qXt, sx, row0 + 1, 2, cols, k, a12, sw1, B);
    int8_matmat_vnni_store(Y, W, scale, qXt, sx, row0 + 1, 3, cols, k, a13, sw1, B);
    int8_matmat_vnni_store(Y, W, scale, qXt, sx, row0 + 1, 4, cols, k, a14, sw1, B);
    int8_matmat_vnni_store(Y, W, scale, qXt, sx, row0 + 1, 5, cols, k, a15, sw1, B);
    int8_matmat_vnni_store(Y, W, scale, qXt, sx, row0 + 1, 6, cols, k, a16, sw1, B);
    int8_matmat_vnni_store(Y, W, scale, qXt, sx, row0 + 1, 7, cols, k, a17, sw1, B);
}

/* A fixed B=2 tile, analogous to the ARM SMMLA 2x2 cross-product path.
 * Unlike the general M4xN4 helper, this has no inactive batch accumulators or
 * nb branches.  It remains opt-in because the best M dimension is ISA- and
 * shape-dependent; the default dispatcher is unchanged. */
static void int8_matmat_vnni_tile_m4n2(float *Y, const int8_t *W, const float *scale,
                                       const int8_t *qXt, const float *sx,
                                       const int32_t *row_sums,
                                       int row0, int cols, int B) QWEN_VNNI_NOINLINE;
static void int8_matmat_vnni_tile_m4n2(float *Y, const int8_t *W, const float *scale,
                                       const int8_t *qXt, const float *sx,
                                       const int32_t *row_sums,
                                       int row0, int cols, int B) {
    const __m512i ones = _mm512_set1_epi8(1);
    const __m512i v128 = _mm512_set1_epi8((char)128);
    __m512i a00 = _mm512_setzero_si512(), a01 = a00;
    __m512i a10 = a00, a11 = a00;
    __m512i a20 = a00, a21 = a00;
    __m512i a30 = a00, a31 = a00;
    __m512i ws0 = a00, ws1 = a00, ws2 = a00, ws3 = a00;
    int k = 0;
    for (; k + 64 <= cols; k += 64) {
        const __m512i ua0 = _mm512_add_epi8(
            _mm512_loadu_si512((const void *)(qXt + k)), v128);
        const __m512i ua1 = _mm512_add_epi8(
            _mm512_loadu_si512((const void *)(qXt + (size_t)cols + k)), v128);

        __m512i wv = _mm512_loadu_si512(
            (const void *)(W + (size_t)(row0 + 0) * cols + k));
        if (!row_sums) ws0 = _mm512_dpbusd_epi32(ws0, ones, wv);
        a00 = _mm512_dpbusd_epi32(a00, ua0, wv);
        a01 = _mm512_dpbusd_epi32(a01, ua1, wv);

        wv = _mm512_loadu_si512(
            (const void *)(W + (size_t)(row0 + 1) * cols + k));
        if (!row_sums) ws1 = _mm512_dpbusd_epi32(ws1, ones, wv);
        a10 = _mm512_dpbusd_epi32(a10, ua0, wv);
        a11 = _mm512_dpbusd_epi32(a11, ua1, wv);

        wv = _mm512_loadu_si512(
            (const void *)(W + (size_t)(row0 + 2) * cols + k));
        if (!row_sums) ws2 = _mm512_dpbusd_epi32(ws2, ones, wv);
        a20 = _mm512_dpbusd_epi32(a20, ua0, wv);
        a21 = _mm512_dpbusd_epi32(a21, ua1, wv);

        wv = _mm512_loadu_si512(
            (const void *)(W + (size_t)(row0 + 3) * cols + k));
        if (!row_sums) ws3 = _mm512_dpbusd_epi32(ws3, ones, wv);
        a30 = _mm512_dpbusd_epi32(a30, ua0, wv);
        a31 = _mm512_dpbusd_epi32(a31, ua1, wv);
    }
    const int sw0 = row_sums ? row_sums[row0 + 0] : _mm512_reduce_add_epi32(ws0);
    const int sw1 = row_sums ? row_sums[row0 + 1] : _mm512_reduce_add_epi32(ws1);
    const int sw2 = row_sums ? row_sums[row0 + 2] : _mm512_reduce_add_epi32(ws2);
    const int sw3 = row_sums ? row_sums[row0 + 3] : _mm512_reduce_add_epi32(ws3);
    int8_matmat_vnni_store(Y, W, scale, qXt, sx, row0 + 0, 0, cols, k, a00, sw0, B);
    int8_matmat_vnni_store(Y, W, scale, qXt, sx, row0 + 0, 1, cols, k, a01, sw0, B);
    int8_matmat_vnni_store(Y, W, scale, qXt, sx, row0 + 1, 0, cols, k, a10, sw1, B);
    int8_matmat_vnni_store(Y, W, scale, qXt, sx, row0 + 1, 1, cols, k, a11, sw1, B);
    int8_matmat_vnni_store(Y, W, scale, qXt, sx, row0 + 2, 0, cols, k, a20, sw2, B);
    int8_matmat_vnni_store(Y, W, scale, qXt, sx, row0 + 2, 1, cols, k, a21, sw2, B);
    int8_matmat_vnni_store(Y, W, scale, qXt, sx, row0 + 3, 0, cols, k, a30, sw3, B);
    int8_matmat_vnni_store(Y, W, scale, qXt, sx, row0 + 3, 1, cols, k, a31, sw3, B);
}

static int qwen_vnni_tile_n8_enabled(void) {
    static atomic_int enabled = -1;
    int v = atomic_load_explicit(&enabled, memory_order_relaxed);
    if (v < 0) {
        const char *e = getenv("QWEN_VNNI_TILE_N8");
        v = e && e[0] == '1';
        atomic_store_explicit(&enabled, v, memory_order_relaxed);
    }
    return v;
}

static int qwen_vnni_tile_m4n2_enabled(void) {
    static atomic_int enabled = -1;
    int v = atomic_load_explicit(&enabled, memory_order_relaxed);
    if (v < 0) {
        const char *e = getenv("QWEN_VNNI_TILE_M4N2");
        v = e && e[0] == '1';
        atomic_store_explicit(&enabled, v, memory_order_relaxed);
    }
    return v;
}

static int qwen_vnni_tile_enabled(void) {
    static atomic_int enabled = -1;
    int v = atomic_load_explicit(&enabled, memory_order_relaxed);
    if (v < 0) {
        const char *off = getenv("QWEN_NO_VNNI_TILE");
        v = !(off && off[0] == '1');
        atomic_store_explicit(&enabled, v, memory_order_relaxed);
    }
    return v;
}

static void int8_matmat_vnni_slice(float *Y, const int8_t *W, const float *scale,
                                   const int8_t *qXt, const float *sx,
                                   int r0, int r1, int rows, int cols, int B) {
    MMSTAT(QWEN_MMK_INT8_VNNI, r1 - r0, cols, B);
    const int32_t *row_sums = qwen_vnni_row_sums(W, rows, cols);
    const int8_t *pW = qwen_vnni_packed_lookup(W, rows, cols);
    if (pW && row_sums && B >= 2 && B <= 8) {
        int8_matmat_vnni_packed_slice(Y, W, pW, scale, qXt, sx, row_sums,
                                      r0, r1, rows, cols, B);
        return;
    }
    if (B == 2 && qwen_vnni_tile_m4n2_enabled() &&
        qwen_vnni_tile_enabled() && cols >= 64) {
        qwen_census_op(QWEN_PATH_MATMAT_INT8_VNNI_M4N2_SLICE, r1 - r0, cols, B);
        int r = r0;
        for (; r + 4 <= r1; r += 4)
            int8_matmat_vnni_tile_m4n2(Y, W, scale, qXt, sx, row_sums,
                                       r, cols, B);
        for (; r < r1; r++)
            int8_matmat_vnni_row(Y, W, scale, qXt, sx, row_sums, r, cols, B);
        return;
    }
    const int compact = B <= 4 || (B <= 8 && rows <= 2048 && cols <= 3072);
    const int tile_n8 = B == 8 && qwen_vnni_tile_n8_enabled();
    if (!qwen_vnni_tile_enabled() || cols < 64 || B < 2 || B > 8 ||
        (!compact && !tile_n8)) {
        for (int r = r0; r < r1; r++)
            int8_matmat_vnni_row(Y, W, scale, qXt, sx, row_sums, r, cols, B);
        return;
    }

    int r = r0;
    if (tile_n8) {
        for (; r + 2 <= r1; r += 2)
            int8_matmat_vnni_tile_m2n8(Y, W, scale, qXt, sx, row_sums,
                                        r, cols, B);
        for (; r < r1; r++)
            int8_matmat_vnni_row(Y, W, scale, qXt, sx, row_sums, r, cols, B);
        return;
    }
    if (B <= 4) {
        for (; r + 4 <= r1; r += 4)
            for (int b0 = 0; b0 < B; b0 += 4) {
                const int nb = (B - b0 < 4) ? (B - b0) : 4;
                int8_matmat_vnni_tile_m4n4(Y, W, scale, qXt, sx, row_sums,
                                            r, b0, nb, cols, B);
            }
    } else {
        for (; r + 2 <= r1; r += 2)
            for (int b0 = 0; b0 < B; b0 += 4) {
                const int nb = (B - b0 < 4) ? (B - b0) : 4;
                int8_matmat_vnni_tile_m2n4(Y, W, scale, qXt, sx, row_sums,
                                            r, b0, nb, cols, B);
            }
    }
    for (; r < r1; r++)
        int8_matmat_vnni_row(Y, W, scale, qXt, sx, row_sums, r, cols, B);
}

typedef struct { float *Y; const int8_t *W; const float *scale; const int8_t *qXt; const float *sx; int rows, cols, B; } int8_vmm_ctx;
static void int8_vmm_task(size_t tid, size_t nt, void *vc) {
    int8_vmm_ctx *c = (int8_vmm_ctx *)vc;
    int r0 = (int)(tid * (size_t)c->rows / nt);
    int r1 = (int)((tid + 1) * (size_t)c->rows / nt);
    int chunk = qwen_x86_nchunk(QWEN_MMK_INT8_VNNI, c->B);
    if (chunk > 0) {
        for (int r = r0; r < r1; r += chunk) {
            int e = r + chunk < r1 ? r + chunk : r1;
            int8_matmat_vnni_slice(c->Y, c->W, c->scale, c->qXt, c->sx,
                                   r, e, c->rows, c->cols, c->B);
        }
    } else {
        int8_matmat_vnni_slice(c->Y, c->W, c->scale, c->qXt, c->sx,
                               r0, r1, c->rows, c->cols, c->B);
    }
}
#undef QWEN_VNNI_NOINLINE
#endif

#if defined(__AVX2__)
static inline int avx2_hsum_epi32(__m256i v) {
    __m128i lo = _mm256_castsi256_si128(v);
    __m128i hi = _mm256_extracti128_si256(v, 1);
    __m128i s  = _mm_add_epi32(lo, hi);
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(1, 0, 3, 2)));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(2, 3, 0, 1)));
    return _mm_cvtsi128_si32(s);
}
static void int8_matmat_avx2_slice(float *Y, const int8_t *W, const float *scale,
                                   const int8_t *qXt, const float *sx,
                                   int r0, int r1, int cols, int B) {
    MMSTAT(QWEN_MMK_INT8_AVX2, r1 - r0, cols, B);
    const __m256i ones16 = _mm256_set1_epi16(1);
    for (int r = r0; r < r1; r++) {
        const int8_t *w = W + (size_t)r * cols;
        const float s = scale[r];
        for (int b0 = 0; b0 < B; b0 += 4) {
            const int bn = (B - b0 < 4) ? (B - b0) : 4;
            const int8_t *p0 = qXt + (size_t)b0 * cols;
            const int8_t *p1 = qXt + (size_t)(bn > 1 ? b0 + 1 : b0) * cols;
            const int8_t *p2 = qXt + (size_t)(bn > 2 ? b0 + 2 : b0) * cols;
            const int8_t *p3 = qXt + (size_t)(bn > 3 ? b0 + 3 : b0) * cols;
            __m256i a0 = _mm256_setzero_si256(), a1 = _mm256_setzero_si256();
            __m256i a2 = _mm256_setzero_si256(), a3 = _mm256_setzero_si256();
            int k = 0;
            for (; k + 32 <= cols; k += 32) {
                __m256i wv = _mm256_loadu_si256((const __m256i *)(w + k));
                __m256i wa = _mm256_abs_epi8(wv);
                __m256i x0 = _mm256_sign_epi8(_mm256_loadu_si256((const __m256i *)(p0 + k)), wv);
                __m256i x1 = _mm256_sign_epi8(_mm256_loadu_si256((const __m256i *)(p1 + k)), wv);
                __m256i x2 = _mm256_sign_epi8(_mm256_loadu_si256((const __m256i *)(p2 + k)), wv);
                __m256i x3 = _mm256_sign_epi8(_mm256_loadu_si256((const __m256i *)(p3 + k)), wv);
                a0 = _mm256_add_epi32(a0, _mm256_madd_epi16(_mm256_maddubs_epi16(wa, x0), ones16));
                a1 = _mm256_add_epi32(a1, _mm256_madd_epi16(_mm256_maddubs_epi16(wa, x1), ones16));
                a2 = _mm256_add_epi32(a2, _mm256_madd_epi16(_mm256_maddubs_epi16(wa, x2), ones16));
                a3 = _mm256_add_epi32(a3, _mm256_madd_epi16(_mm256_maddubs_epi16(wa, x3), ones16));
            }
            int acc[4] = { avx2_hsum_epi32(a0), avx2_hsum_epi32(a1),
                           avx2_hsum_epi32(a2), avx2_hsum_epi32(a3) };
            for (int j = 0; j < bn; j++) {
                const int b = b0 + j;
                const int8_t *qb = qXt + (size_t)b * cols;
                int sum = acc[j];
                for (int kk = k; kk < cols; kk++) sum += (int)w[kk] * (int)qb[kk];
                Y[(size_t)r * B + b] = (float)sum * s * sx[b];
            }
        }
    }
}
typedef struct { float *Y; const int8_t *W; const float *scale; const int8_t *qXt; const float *sx; int rows, cols, B; } int8_amm_ctx;
static void int8_amm_task(size_t tid, size_t nt, void *vc) {
    int8_amm_ctx *c = (int8_amm_ctx *)vc;
    int r0 = (int)(tid * (size_t)c->rows / nt);
    int r1 = (int)((tid + 1) * (size_t)c->rows / nt);
    int8_matmat_avx2_slice(c->Y, c->W, c->scale, c->qXt, c->sx, r0, r1, c->cols, c->B);
}
#endif

#if defined(__ARM_FEATURE_MATMUL_INT8)
static void int8_matmat_smmla_slice(float *Y, const int8_t *W, const float *scale,
                                    const int8_t *qXt, const float *sx,
                                    int r0, int r1, int cols, int B) {
    MMSTAT(QWEN_MMK_INT8_SMMLA, r1 - r0, cols, B);
    int r = r0;
    for (; r + 3 < r1 && B >= 4; r += 4) {
        const int8_t *w0 = W + (size_t)r * cols,       *w1 = W + (size_t)(r + 1) * cols;
        const int8_t *w2 = W + (size_t)(r + 2) * cols, *w3 = W + (size_t)(r + 3) * cols;
        int j = 0;
        for (; j + 3 < B; j += 4) {
            const int8_t *x0 = qXt + (size_t)j * cols,       *x1 = qXt + (size_t)(j + 1) * cols;
            const int8_t *x2 = qXt + (size_t)(j + 2) * cols, *x3 = qXt + (size_t)(j + 3) * cols;
            int32x4_t a00 = vdupq_n_s32(0), a01 = vdupq_n_s32(0);
            int32x4_t a10 = vdupq_n_s32(0), a11 = vdupq_n_s32(0);
            int k = 0;
            for (; k + 7 < cols; k += 8) {
                int8x16_t A01 = vcombine_s8(vld1_s8(w0 + k), vld1_s8(w1 + k));
                int8x16_t A23 = vcombine_s8(vld1_s8(w2 + k), vld1_s8(w3 + k));
                int8x16_t B01 = vcombine_s8(vld1_s8(x0 + k), vld1_s8(x1 + k));
                int8x16_t B23 = vcombine_s8(vld1_s8(x2 + k), vld1_s8(x3 + k));
                a00 = vmmlaq_s32(a00, A01, B01);
                a01 = vmmlaq_s32(a01, A01, B23);
                a10 = vmmlaq_s32(a10, A23, B01);
                a11 = vmmlaq_s32(a11, A23, B23);
            }
            int32_t t00[4], t01[4], t10[4], t11[4];
            vst1q_s32(t00, a00); vst1q_s32(t01, a01);
            vst1q_s32(t10, a10); vst1q_s32(t11, a11);
            for (; k < cols; k++) {
                int32_t wv[4] = { w0[k], w1[k], w2[k], w3[k] };
                int32_t xv[4] = { x0[k], x1[k], x2[k], x3[k] };
                t00[0] += wv[0]*xv[0]; t00[1] += wv[0]*xv[1]; t00[2] += wv[1]*xv[0]; t00[3] += wv[1]*xv[1];
                t01[0] += wv[0]*xv[2]; t01[1] += wv[0]*xv[3]; t01[2] += wv[1]*xv[2]; t01[3] += wv[1]*xv[3];
                t10[0] += wv[2]*xv[0]; t10[1] += wv[2]*xv[1]; t10[2] += wv[3]*xv[0]; t10[3] += wv[3]*xv[1];
                t11[0] += wv[2]*xv[2]; t11[1] += wv[2]*xv[3]; t11[2] += wv[3]*xv[2]; t11[3] += wv[3]*xv[3];
            }
            const float s_r0 = scale[r], s_r1 = scale[r+1], s_r2 = scale[r+2], s_r3 = scale[r+3];
            const float x_j0 = sx[j], x_j1 = sx[j+1], x_j2 = sx[j+2], x_j3 = sx[j+3];
            float *Y0 = Y + (size_t)r * B,       *Y1 = Y + (size_t)(r + 1) * B;
            float *Y2 = Y + (size_t)(r + 2) * B, *Y3 = Y + (size_t)(r + 3) * B;
            Y0[j]   = (float)t00[0] * s_r0 * x_j0;  Y0[j+1] = (float)t00[1] * s_r0 * x_j1;
            Y1[j]   = (float)t00[2] * s_r1 * x_j0;  Y1[j+1] = (float)t00[3] * s_r1 * x_j1;
            Y0[j+2] = (float)t01[0] * s_r0 * x_j2;  Y0[j+3] = (float)t01[1] * s_r0 * x_j3;
            Y1[j+2] = (float)t01[2] * s_r1 * x_j2;  Y1[j+3] = (float)t01[3] * s_r1 * x_j3;
            Y2[j]   = (float)t10[0] * s_r2 * x_j0;  Y2[j+1] = (float)t10[1] * s_r2 * x_j1;
            Y3[j]   = (float)t10[2] * s_r3 * x_j0;  Y3[j+1] = (float)t10[3] * s_r3 * x_j1;
            Y2[j+2] = (float)t11[0] * s_r2 * x_j2;  Y2[j+3] = (float)t11[1] * s_r2 * x_j3;
            Y3[j+2] = (float)t11[2] * s_r3 * x_j2;  Y3[j+3] = (float)t11[3] * s_r3 * x_j3;
        }
        for (; j < B; j++) {
            const int8_t *xj = qXt + (size_t)j * cols;
            int64_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;
            for (int k = 0; k < cols; k++) {
                int32_t xv = xj[k];
                s0 += w0[k] * xv; s1 += w1[k] * xv; s2 += w2[k] * xv; s3 += w3[k] * xv;
            }
            Y[(size_t)r * B + j]       = (float)s0 * scale[r]     * sx[j];
            Y[(size_t)(r + 1) * B + j] = (float)s1 * scale[r + 1] * sx[j];
            Y[(size_t)(r + 2) * B + j] = (float)s2 * scale[r + 2] * sx[j];
            Y[(size_t)(r + 3) * B + j] = (float)s3 * scale[r + 3] * sx[j];
        }
    }
    for (; r + 1 < r1; r += 2) {
        const int8_t *w0 = W + (size_t)r * cols, *w1 = W + (size_t)(r + 1) * cols;
        int j = 0;
        for (; j + 1 < B; j += 2) {
            const int8_t *x0 = qXt + (size_t)j * cols, *x1 = qXt + (size_t)(j + 1) * cols;
            int32x4_t acc = vdupq_n_s32(0);
            int k = 0;
            for (; k + 7 < cols; k += 8) {
                int8x16_t a = vcombine_s8(vld1_s8(w0 + k), vld1_s8(w1 + k));
                int8x16_t b = vcombine_s8(vld1_s8(x0 + k), vld1_s8(x1 + k));
                acc = vmmlaq_s32(acc, a, b);
            }
            int32_t t[4]; vst1q_s32(t, acc);
            for (; k < cols; k++) {
                t[0] += w0[k] * x0[k]; t[1] += w0[k] * x1[k];
                t[2] += w1[k] * x0[k]; t[3] += w1[k] * x1[k];
            }
            Y[(size_t)r * B + j]           = (float)t[0] * scale[r]     * sx[j];
            Y[(size_t)r * B + j + 1]       = (float)t[1] * scale[r]     * sx[j + 1];
            Y[(size_t)(r + 1) * B + j]     = (float)t[2] * scale[r + 1] * sx[j];
            Y[(size_t)(r + 1) * B + j + 1] = (float)t[3] * scale[r + 1] * sx[j + 1];
        }
        for (; j < B; j++) {
            const int8_t *xj = qXt + (size_t)j * cols;
            int64_t s0 = 0, s1 = 0;
            for (int k = 0; k < cols; k++) { s0 += w0[k] * xj[k]; s1 += w1[k] * xj[k]; }
            Y[(size_t)r * B + j]       = (float)s0 * scale[r]     * sx[j];
            Y[(size_t)(r + 1) * B + j] = (float)s1 * scale[r + 1] * sx[j];
        }
    }
    for (; r < r1; r++) {
        const int8_t *w = W + (size_t)r * cols;
        for (int j = 0; j < B; j++) {
            const int8_t *xj = qXt + (size_t)j * cols;
            int64_t s = 0;
            for (int k = 0; k < cols; k++) s += w[k] * xj[k];
            Y[(size_t)r * B + j] = (float)s * scale[r] * sx[j];
        }
    }
}
typedef struct {
    float *Y; const int8_t *W; const float *scale; const int8_t *qXt; const float *sx;
    int rows, cols, B;
} int8_smmla_ctx;
static void int8_smmla_task(size_t tid, size_t nt, void *vc) {
    int8_smmla_ctx *c = (int8_smmla_ctx *)vc;
    int r0 = (int)(tid * (size_t)c->rows / nt), r1 = (int)((tid + 1) * (size_t)c->rows / nt);
    r0 &= ~3; if (tid + 1 < nt) r1 &= ~3;
    int8_matmat_smmla_slice(c->Y, c->W, c->scale, c->qXt, c->sx, r0, r1, c->cols, c->B);
}
#endif

void qwen_matmat_int8(float *Y, const int8_t *W, const float *scale,
                      const float *X, int rows, int cols, int B) {
    qwen_census_op(QWEN_PATH_MATMAT_INT8, rows, cols, B);
    const int kt_on = qwen_kernel_timing_enabled();
    const double kt_t0 = kt_on ? qwen_mm_now_s() : 0.0;
    const int kt_B = B;
    if (B <= 0) return;
    if (kai_i8_try(Y, W, scale, X, rows, cols, B)) {
        MMSTAT(B > 1 ? QWEN_MMK_KLEIDI_I8 : QWEN_MMK_KLEIDI_I8_GEMV, rows, cols, B);
        goto qwen_matmat_int8_timed_done;
    }
    if (B > 64) B = 64;
#if defined(__AMX_INT8__) && defined(__AMX_TILE__)
    {
        if (qwen_mm_use(QWEN_MMK_INT8_AMX, B, rows, cols) && qwen_amx_int8_ready()) {
            const size_t kfull = (size_t)(cols & ~63);
            int8_t *qXt = mm_scratch_qx((size_t)B * cols);
            int8_t *pXt = NULL;
            if (qXt) pXt = mm_scratch_pack(kfull * (size_t)B);
            if (qXt && pXt) {
                const uint8_t *pW = (const uint8_t *)qwen_amx_pack_weights(
                    W, rows, cols, QWEN_AMX_WEIGHT_INT8);
                float sx[16];
                for (int b = 0; b < B; b++)
                    sx[b] = quantize_act_int8_col(qXt + (size_t)b * cols, X, cols, B, b);
                amx_pack_act_int8(pXt, qXt, cols, (int)kfull, B);
                int nt = g_n_threads;
                int8_amx_ctx c = { Y, W, pW, scale, pXt, qXt, sx, rows, cols, B };
                if (nt > 1 && rows >= 256) {
                    qwen_parallel((size_t)nt, int8_amx_task, &c);
                } else {
                    int8_amx_task(0, 1, &c);
                }
                goto qwen_matmat_int8_timed_done;
            }
        }
    }
#endif
#if defined(__AVX512VNNI__)
    {
        if (qwen_mm_use(QWEN_MMK_INT8_VNNI, B, rows, cols)) {
            int8_t *qXt = mm_scratch_qx((size_t)B * cols);
            if (qXt) {
                float sx[16];
                for (int b = 0; b < B; b++)
                    sx[b] = quantize_act_int8_col(qXt + (size_t)b * cols, X, cols, B, b);
                int nt = g_n_threads;
                int8_vmm_ctx c = { Y, W, scale, qXt, sx, rows, cols, B };
                if (nt > 1 && rows >= 256) {
                    qwen_parallel((size_t)nt, int8_vmm_task, &c);
                } else {
                    int8_vmm_task(0, 1, &c);
                }
                goto qwen_matmat_int8_timed_done;
            }
        }
    }
#endif

#if defined(__AVX2__)
    /* AVX2 without VNNI: without this, B requests fall through to the f32 twin and
     * read the weights B times. QWEN_NO_AVX2MM=1 disables it. */
    {
        if (qwen_mm_use(QWEN_MMK_INT8_AVX2, B, rows, cols)) {
            int8_t *qXt = mm_scratch_qx((size_t)B * cols);
            if (qXt) {
                float sx[16];
                for (int b = 0; b < B; b++)
                    sx[b] = quantize_act_int8_col(qXt + (size_t)b * cols, X, cols, B, b);
                int nt = g_n_threads;
                if (nt > 1 && rows >= 256) {
                    int8_amm_ctx c = { Y, W, scale, qXt, sx, rows, cols, B };
                    qwen_parallel((size_t)nt, int8_amm_task, &c);
                } else {
                    int8_matmat_avx2_slice(Y, W, scale, qXt, sx, 0, rows, cols, B);
                }
                goto qwen_matmat_int8_timed_done;
            }
        }
    }
#endif
#if defined(__ARM_FEATURE_MATMUL_INT8)
    {
        if (qwen_mm_use(QWEN_MMK_INT8_SMMLA, B, rows, cols)) {
            int8_t *qXt = mm_scratch_qx((size_t)B * cols);
            if (qXt) {
                float sx[16];
                for (int b = 0; b < B; b++)
                    sx[b] = quantize_act_int8_col(qXt + (size_t)b * cols, X, cols, B, b);
                int nt2 = g_n_threads;
                if (nt2 > 1 && rows >= 256) {
                    int8_smmla_ctx c = { Y, W, scale, qXt, sx, rows, cols, B };
                    qwen_parallel((size_t)nt2, int8_smmla_task, &c);
                } else {
                    int8_matmat_smmla_slice(Y, W, scale, qXt, sx, 0, rows, cols, B);
                }
                goto qwen_matmat_int8_timed_done;
            }
        }
    }
#endif
#if defined(__ARM_FEATURE_DOTPROD)
    {
        if (qwen_mm_use(QWEN_MMK_INT8_SDOT, B, rows, cols)) {
            int8_t *qXt = mm_scratch_qx((size_t)B * cols);
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
                goto qwen_matmat_int8_timed_done;
            }
        }
    }
#endif
    int nt = g_n_threads;
    if (nt > 1 && rows >= 256) {
        int8_mm_ctx c = { Y, W, scale, X, rows, cols, B };
        qwen_parallel((size_t)nt, int8_mm_task, &c);
        goto qwen_matmat_int8_timed_done;
    }
    int8_matmat_slice(Y, W, scale, X, 0, rows, cols, B);

qwen_matmat_int8_timed_done:
    qwen_kernel_timing_note(QWEN_KT_INT8, kt_B, rows, cols, kt_t0);
}

static int qwen_x86_qkv_disabled(void) QWEN_MAYBE_UNUSED;
static int qwen_x86_qkv_disabled(void) {
    static atomic_int disabled = -1;
    int v = atomic_load_explicit(&disabled, memory_order_relaxed);
    if (v < 0) {
        const char *e = getenv("QWEN_NO_X86_QKV");
        v = e && e[0] == '1';
        atomic_store_explicit(&disabled, v, memory_order_relaxed);
    }
    return v;
}

static int qwen_amx_int8_qkv_allowed(int B, int q_dim, int kv_dim, int in_dim)
    QWEN_MAYBE_UNUSED;
static int qwen_amx_int8_qkv_allowed(int B, int q_dim, int kv_dim, int in_dim) {
    if (!qwen_mm_use_(QWEN_MMK_INT8_AMX, B, q_dim, in_dim, 0) ||
        !qwen_mm_use_(QWEN_MMK_INT8_AMX, B, kv_dim, in_dim, 0)) return 0;
    /* Q, K and V share ONE activation pack and one tile configuration here, so the work-per-
     * thread rule belongs to their combined height, not to k and v on their own.  That is also
     * what the measurement shows: at 12 threads both QKV projections are AMX wins (-7.9% and
     * -6.0%) while the plain 2048-row projections at the same rows/thread are VNNI. */
    if (!qwen_amx_int8_rows_ok((long long)q_dim + 2LL * (long long)kv_dim)) return 0;

    /* QKV has a different working set from the other projections.  This is an
     * additional lower bound for that fused path; the general AMX INT8 gate
     * remains authoritative and still applies to every other projection. */
    int base = qwen_mm_minb_value(QWEN_MMK_INT8_AMX,
                                  &g_mm_gate[QWEN_MMK_INT8_AMX]);
    int qkv = qwen_mm_env_int("QWEN_AMX_INT8_QKV_MIN_B", base, 1, 64);
    if (qkv < base) qkv = base;
    return B >= qkv;
}

typedef struct {
    float *Y[3];
    const int8_t *W[3];
    const uint8_t *pW[3];
    const float *scale[3];
    const int8_t *pXt, *qXt;
    const float *sx;
    int rows[3], cols, B;
} int8_qkv_mm_ctx;

#if defined(__AMX_INT8__) && defined(__AMX_TILE__)
static void int8_qkv_amx_run(float *Y, const int8_t *W, const uint8_t *pW,
                             const float *scale,
                             const int8_t *pXt, const int8_t *qXt, const float *sx,
                             int r0, int r1, int cols, int B) {
    int chunk = qwen_x86_nchunk(QWEN_MMK_INT8_AMX, B);
    if (chunk > 0) {
        for (int r = r0; r < r1; r += chunk) {
            int e = r + chunk < r1 ? r + chunk : r1;
            int8_matmat_amx_slice(Y, W, pW, scale, pXt, qXt, sx, r, e, cols, B);
        }
    } else {
        int8_matmat_amx_slice(Y, W, pW, scale, pXt, qXt, sx, r0, r1, cols, B);
    }
}

static void int8_qkv_amx_task(size_t tid, size_t nt, void *vc) {
    int8_qkv_mm_ctx *c = (int8_qkv_mm_ctx *)vc;
    const int total = c->rows[0] + c->rows[1] + c->rows[2];
    const int tiles = total / 16;
    const int t0 = (int)(tid * (size_t)tiles / nt);
    const int t1 = (int)((tid + 1) * (size_t)tiles / nt);
    const int g0 = t0 * 16, g1 = t1 * 16;
    int base = 0;
    for (int i = 0; i < 3; i++) {
        int lo = g0 > base ? g0 : base;
        int hi = g1 < base + c->rows[i] ? g1 : base + c->rows[i];
        if (hi > lo)
            int8_qkv_amx_run(c->Y[i], c->W[i], c->pW[i], c->scale[i], c->pXt, c->qXt,
                             c->sx, lo - base, hi - base, c->cols, c->B);
        base += c->rows[i];
    }
}
#endif

#if defined(__AVX512VNNI__)
static void int8_qkv_vnni_run(float *Y, const int8_t *W, const float *scale,
                               const int8_t *qXt, const float *sx,
                               int r0, int r1, int rows, int cols, int B) {
    int chunk = qwen_x86_nchunk(QWEN_MMK_INT8_VNNI, B);
    if (chunk > 0) {
        for (int r = r0; r < r1; r += chunk) {
            int e = r + chunk < r1 ? r + chunk : r1;
            int8_matmat_vnni_slice(Y, W, scale, qXt, sx, r, e, rows, cols, B);
        }
    } else {
        int8_matmat_vnni_slice(Y, W, scale, qXt, sx, r0, r1, rows, cols, B);
    }
}

static void int8_qkv_vnni_mm_task(size_t tid, size_t nt, void *vc) {
    int8_qkv_mm_ctx *c = (int8_qkv_mm_ctx *)vc;
    const int total = c->rows[0] + c->rows[1] + c->rows[2];
    const int g0 = (int)(tid * (size_t)total / nt);
    const int g1 = (int)((tid + 1) * (size_t)total / nt);
    int base = 0;
    for (int i = 0; i < 3; i++) {
        int lo = g0 > base ? g0 : base;
        int hi = g1 < base + c->rows[i] ? g1 : base + c->rows[i];
        if (hi > lo)
            int8_qkv_vnni_run(c->Y[i], c->W[i], c->scale[i], c->qXt, c->sx,
                              lo - base, hi - base, c->rows[i], c->cols, c->B);
        base += c->rows[i];
    }
}
#endif

int qwen_matmat_int8_qkv(float *q, float *k, float *v,
                         const int8_t *Wq, const float *sq,
                         const int8_t *Wk, const float *sk,
                         const int8_t *Wv, const float *sv,
                         const float *X, int in_dim, int q_dim, int kv_dim, int B) {
    const int total = q_dim + 2 * kv_dim;
    qwen_census_op(QWEN_PATH_MATMAT_INT8_QKV, total, in_dim, B);
    const int kt_on = qwen_kernel_timing_enabled();
    const double kt_t0 = kt_on ? qwen_mm_now_s() : 0.0;
    const int kt_B = B;
#if !defined(__x86_64__) && !defined(_M_X64)
    (void)q; (void)k; (void)v; (void)Wq; (void)sq; (void)Wk; (void)sk;
    (void)Wv; (void)sv; (void)X; (void)in_dim; (void)q_dim; (void)kv_dim; (void)B;
    return 0;
#else
    if (qwen_x86_qkv_disabled() || B <= 1 || B > 16 || in_dim <= 0 ||
        q_dim <= 0 || kv_dim <= 0 || !Wq || !Wk || !Wv) return 0;

    int8_t *qXt = mm_scratch_qx((size_t)B * in_dim);
    if (!qXt) return 0;
    float sx[16];
    for (int b = 0; b < B; b++)
        sx[b] = quantize_act_int8_col(qXt + (size_t)b * in_dim, X, in_dim, B, b);

    int8_qkv_mm_ctx c = {
        { q, k, v }, { Wq, Wk, Wv }, { NULL, NULL, NULL }, { sq, sk, sv },
        NULL, qXt, sx, { q_dim, kv_dim, kv_dim }, in_dim, B
    };
    const int nt = g_n_threads;
#if defined(__AMX_INT8__) && defined(__AMX_TILE__)
    if ((q_dim & 15) == 0 && (kv_dim & 15) == 0 &&
        qwen_amx_int8_qkv_allowed(B, q_dim, kv_dim, in_dim) &&
        qwen_amx_int8_ready()) {
        const size_t kfull = (size_t)(in_dim & ~63);
        c.pXt = mm_scratch_pack(kfull * (size_t)B);
        if (c.pXt) {
            c.pW[0] = (const uint8_t *)qwen_amx_pack_weights(
                Wq, q_dim, in_dim, QWEN_AMX_WEIGHT_INT8);
            c.pW[1] = (const uint8_t *)qwen_amx_pack_weights(
                Wk, kv_dim, in_dim, QWEN_AMX_WEIGHT_INT8);
            c.pW[2] = (const uint8_t *)qwen_amx_pack_weights(
                Wv, kv_dim, in_dim, QWEN_AMX_WEIGHT_INT8);
            amx_pack_act_int8((int8_t *)c.pXt, qXt, in_dim, (int)kfull, B);
            if (nt > 1 && total >= 256)
                qwen_parallel((size_t)nt, int8_qkv_amx_task, &c);
            else
                int8_qkv_amx_task(0, 1, &c);
            qwen_kernel_timing_note(QWEN_KT_INT8, kt_B, total, in_dim, kt_t0);
            return 1;
        }
    }
#endif
#if defined(__AVX512VNNI__)
    if (qwen_mm_use(QWEN_MMK_INT8_VNNI, B, q_dim, in_dim) &&
        qwen_mm_use(QWEN_MMK_INT8_VNNI, B, kv_dim, in_dim)) {
        if (nt > 1 && total >= 256)
            qwen_parallel((size_t)nt, int8_qkv_vnni_mm_task, &c);
        else
            int8_qkv_vnni_mm_task(0, 1, &c);
        qwen_kernel_timing_note(QWEN_KT_INT8, kt_B, total, in_dim, kt_t0);
        return 1;
    }
#endif
    return 0;
#endif
}

typedef struct {
    float *Y[3];
    const uint16_t *W[3];
    const uint8_t *pW[3];
    const uint16_t *pXb, *Xb;
    int rows[3], cols, B;
} bf16_qkv_mm_ctx;

#if defined(__AMX_BF16__) && defined(__AMX_TILE__)
static void bf16_qkv_amx_run(float *Y, const uint16_t *W, const uint8_t *pW,
                             const uint16_t *pXb, const uint16_t *Xb,
                             int r0, int r1, int cols, int B) {
    int chunk = qwen_x86_nchunk(QWEN_MMK_BF16_AMX, B);
    if (chunk > 0) {
        for (int r = r0; r < r1; r += chunk) {
            int e = r + chunk < r1 ? r + chunk : r1;
            bf16_matmat_amx_slice(Y, W, pW, pXb, Xb, r, e, cols, B);
        }
    } else {
        bf16_matmat_amx_slice(Y, W, pW, pXb, Xb, r0, r1, cols, B);
    }
}

static void bf16_qkv_amx_task(size_t tid, size_t nt, void *vc) {
    bf16_qkv_mm_ctx *c = (bf16_qkv_mm_ctx *)vc;
    const int total = c->rows[0] + c->rows[1] + c->rows[2];
    const int tiles = total / 16;
    const int t0 = (int)(tid * (size_t)tiles / nt);
    const int t1 = (int)((tid + 1) * (size_t)tiles / nt);
    const int g0 = t0 * 16, g1 = t1 * 16;
    int base = 0;
    for (int i = 0; i < 3; i++) {
        int lo = g0 > base ? g0 : base;
        int hi = g1 < base + c->rows[i] ? g1 : base + c->rows[i];
        if (hi > lo)
            bf16_qkv_amx_run(c->Y[i], c->W[i], c->pW[i], c->pXb, c->Xb,
                             lo - base, hi - base, c->cols, c->B);
        base += c->rows[i];
    }
}
#endif

#if defined(__AVX512BF16__)
static void bf16_qkv_avx512_run(float *Y, const uint16_t *W, const uint16_t *Xb,
                                int r0, int r1, int rows, int cols, int B) {
    int chunk = qwen_x86_nchunk(QWEN_MMK_BF16_AVX512, B);
    if (chunk > 0) {
        for (int r = r0; r < r1; r += chunk) {
            int e = r + chunk < r1 ? r + chunk : r1;
            bf16_matmat_avx512_slice(Y, W, Xb, r, e, cols, B);
        }
    } else {
        bf16_matmat_avx512_slice(Y, W, Xb, r0, r1, cols, B);
    }
    (void)rows;
}

static void bf16_qkv_avx512_task(size_t tid, size_t nt, void *vc) {
    bf16_qkv_mm_ctx *c = (bf16_qkv_mm_ctx *)vc;
    const int total = c->rows[0] + c->rows[1] + c->rows[2];
    const int g0 = (int)(tid * (size_t)total / nt);
    const int g1 = (int)((tid + 1) * (size_t)total / nt);
    int base = 0;
    for (int i = 0; i < 3; i++) {
        int lo = g0 > base ? g0 : base;
        int hi = g1 < base + c->rows[i] ? g1 : base + c->rows[i];
        if (hi > lo)
            bf16_qkv_avx512_run(c->Y[i], c->W[i], c->Xb,
                                lo - base, hi - base, c->rows[i], c->cols, c->B);
        base += c->rows[i];
    }
}
#endif

int qwen_matmat_bf16_qkv(float *q, float *k, float *v,
                         const uint16_t *Wq, const uint16_t *Wk, const uint16_t *Wv,
                         const float *X, int in_dim, int q_dim, int kv_dim, int B) {
    const int total = q_dim + 2 * kv_dim;
    qwen_census_op(QWEN_PATH_MATMAT_BF16_QKV, total, in_dim, B);
    const int kt_on = qwen_kernel_timing_enabled();
    const double kt_t0 = kt_on ? qwen_mm_now_s() : 0.0;
    const int kt_B = B;
#if !defined(__x86_64__) && !defined(_M_X64)
    (void)q; (void)k; (void)v; (void)Wq; (void)Wk; (void)Wv; (void)X;
    (void)in_dim; (void)q_dim; (void)kv_dim; (void)B;
    return 0;
#else
    if (qwen_x86_qkv_disabled() || B <= 1 || B > 16 || in_dim <= 0 ||
        q_dim <= 0 || kv_dim <= 0 || !Wq || !Wk || !Wv) return 0;

    uint16_t *Xb = mm_scratch_xb((size_t)B * in_dim);
    if (!Xb) return 0;
    for (int b = 0; b < B; b++)
        for (int i = 0; i < in_dim; i++) {
            uint32_t u;
            memcpy(&u, &X[(size_t)i * B + b], sizeof u);
#if defined(__AVX512BF16__)
            Xb[(size_t)b * in_dim + i] =
                (uint16_t)((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
#else
            Xb[(size_t)b * in_dim + i] = (uint16_t)(u >> 16);
#endif
        }

    bf16_qkv_mm_ctx c = {
        { q, k, v }, { Wq, Wk, Wv }, { NULL, NULL, NULL }, NULL, Xb,
        { q_dim, kv_dim, kv_dim }, in_dim, B
    };
    const int nt = g_n_threads;
#if defined(__AMX_BF16__) && defined(__AMX_TILE__)
    if ((q_dim & 15) == 0 && (kv_dim & 15) == 0 &&
        qwen_mm_use(QWEN_MMK_BF16_AMX, B, q_dim, in_dim) &&
        qwen_mm_use(QWEN_MMK_BF16_AMX, B, kv_dim, in_dim) &&
        qwen_amx_bf16_ready()) {
        const size_t kfull = (size_t)(in_dim & ~31);
        c.pXb = mm_scratch_packb(kfull * (size_t)B);
        if (c.pXb) {
            c.pW[0] = (const uint8_t *)qwen_amx_pack_weights(
                Wq, q_dim, in_dim, QWEN_AMX_WEIGHT_BF16);
            c.pW[1] = (const uint8_t *)qwen_amx_pack_weights(
                Wk, kv_dim, in_dim, QWEN_AMX_WEIGHT_BF16);
            c.pW[2] = (const uint8_t *)qwen_amx_pack_weights(
                Wv, kv_dim, in_dim, QWEN_AMX_WEIGHT_BF16);
            amx_pack_act_bf16((uint16_t *)c.pXb, Xb, in_dim, (int)kfull, B);
            if (nt > 1 && total >= 256)
                qwen_parallel((size_t)nt, bf16_qkv_amx_task, &c);
            else
                bf16_qkv_amx_task(0, 1, &c);
            qwen_kernel_timing_note(QWEN_KT_BF16, kt_B, total, in_dim, kt_t0);
            return 1;
        }
    }
#endif
#if defined(__AVX512BF16__)
    if (!qwen_bf16dot_disabled() &&
        qwen_mm_use(QWEN_MMK_BF16_AVX512, B, q_dim, in_dim) &&
        qwen_mm_use(QWEN_MMK_BF16_AVX512, B, kv_dim, in_dim)) {
        if (nt > 1 && total >= 256)
            qwen_parallel((size_t)nt, bf16_qkv_avx512_task, &c);
        else
            bf16_qkv_avx512_task(0, 1, &c);
        qwen_kernel_timing_note(QWEN_KT_BF16, kt_B, total, in_dim, kt_t0);
        return 1;
    }
#endif
    return 0;
#endif
}

static void q4_matmat_generic(float *Y, const q4_0_block_t *W, const float *X,
                              int r0, int r1, int cols, int B) {
    int nb = cols / Q4_0_BLOCK_SIZE;
    for (int r = r0; r < r1; r++) {
        const q4_0_block_t *wr = W + (size_t)r * nb;
        float *y = Y + (size_t)r * B;
        float acc[64];
        for (int b = 0; b < B; b++) acc[b] = 0.0f;
        for (int bl = 0; bl < nb; bl++) {
            float sc = qwen_f16_to_f32(wr[bl].scale_f16);
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
            float sc = qwen_f16_to_f32(wr[bl].scale_f16);                     \
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
    MMSTAT(QWEN_MMK_Q4_GENERIC, r1 - r0, cols, B);
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
static void q4_matmat_vnni_slice(float *Y, const q4_0_block_t *W, const int8_t *qXt,
                                 const float *sx, const int *corr,
                                 int r0, int r1, int cols, int B) {
    MMSTAT(QWEN_MMK_Q4_VNNI, r1 - r0, cols, B);
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
            float scl = qwen_f16_to_f32(row[bl].scale_f16);
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
#endif

#if defined(__AVX2__)
static void q4_matmat_avx2_slice(float *Y, const q4_0_block_t *W, const int8_t *qXt,
                                 const float *sx, const int *corr,
                                 int r0, int r1, int cols, int B) {
    MMSTAT(QWEN_MMK_Q4_AVX2, r1 - r0, cols, B);
    int nb = cols / Q4_0_BLOCK_SIZE;
    const __m128i lomask = _mm_set1_epi8(0x0F);
    const __m256i ones16 = _mm256_set1_epi16(1);
    for (int r = r0; r < r1; r++) {
        const q4_0_block_t *row = W + (size_t)r * nb;
        float sum[16];
        for (int b = 0; b < B; b++) sum[b] = 0.0f;
        for (int bl = 0; bl < nb; bl++) {
            __m128i raw = _mm_loadu_si128((const __m128i *)row[bl].qs);
            __m128i lo = _mm_and_si128(raw, lomask);
            __m128i hi = _mm_and_si128(_mm_srli_epi16(raw, 4), lomask);
            __m256i wv = _mm256_set_m128i(_mm_unpackhi_epi8(lo, hi),
                                          _mm_unpacklo_epi8(lo, hi));
            float scl = qwen_f16_to_f32(row[bl].scale_f16);
            for (int b = 0; b < B; b++) {
                __m256i xv = _mm256_loadu_si256((const __m256i *)
                    (qXt + (size_t)b * cols + (size_t)bl * Q4_0_BLOCK_SIZE));
                int dot = avx2_hsum_epi32(_mm256_madd_epi16(_mm256_maddubs_epi16(wv, xv), ones16))
                        + corr[(size_t)b * nb + bl];
                sum[b] += scl * (float)dot;
            }
        }
        for (int b = 0; b < B; b++) Y[(size_t)r * B + b] = sum[b] * sx[b];
    }
}
typedef struct { float *Y; const q4_0_block_t *W; const int8_t *qXt; const float *sx; const int *corr; int rows, cols, B; } q4_amm_ctx;
static void q4_amm_task(size_t tid, size_t nt, void *vc) {
    q4_amm_ctx *c = (q4_amm_ctx *)vc;
    int r0 = (int)(tid * (size_t)c->rows / nt);
    int r1 = (int)((tid + 1) * (size_t)c->rows / nt);
    q4_matmat_avx2_slice(c->Y, c->W, c->qXt, c->sx, c->corr, r0, r1, c->cols, c->B);
}
#endif
#if defined(__ARM_FEATURE_MATMUL_INT8)
static void q4_matmat_smmla_slice(float *Y, const q4_0_block_t *W,
                                  const int8_t *qXt, const float *sx, const int *corr,
                                  int r0, int r1, int cols, int B) {
    MMSTAT(QWEN_MMK_Q4_SMMLA, r1 - r0, cols, B);
    int nb = cols / Q4_0_BLOCK_SIZE;
    const uint8x16_t mask = vdupq_n_u8(0x0F);
    int r = r0;
    for (; r + 1 < r1; r += 2) {
        const q4_0_block_t *w0 = W + (size_t)r * nb, *w1 = W + (size_t)(r + 1) * nb;
        if (B <= 16) {
            float f0[16], f1[16];
            for (int c = 0; c < B; c++) { f0[c] = 0.0f; f1[c] = 0.0f; }
            for (int bl = 0; bl < nb; bl++) {
                uint8x16_t raw0 = vld1q_u8(w0[bl].qs), raw1 = vld1q_u8(w1[bl].qs);
                uint8x16x2_t z0 = vzipq_u8(vandq_u8(raw0, mask), vshrq_n_u8(raw0, 4));
                uint8x16x2_t z1 = vzipq_u8(vandq_u8(raw1, mask), vshrq_n_u8(raw1, 4));
                int8x16_t a0lo = vreinterpretq_s8_u8(z0.val[0]), a0hi = vreinterpretq_s8_u8(z0.val[1]);
                int8x16_t a1lo = vreinterpretq_s8_u8(z1.val[0]), a1hi = vreinterpretq_s8_u8(z1.val[1]);
                int8x16_t A_lo_lo = vcombine_s8(vget_low_s8(a0lo),  vget_low_s8(a1lo));
                int8x16_t A_lo_hi = vcombine_s8(vget_high_s8(a0lo), vget_high_s8(a1lo));
                int8x16_t A_hi_lo = vcombine_s8(vget_low_s8(a0hi),  vget_low_s8(a1hi));
                int8x16_t A_hi_hi = vcombine_s8(vget_high_s8(a0hi), vget_high_s8(a1hi));
                const float s0 = qwen_f16_to_f32(w0[bl].scale_f16);
                const float s1 = qwen_f16_to_f32(w1[bl].scale_f16);
                int c = 0;
                for (; c + 1 < B; c += 2) {
                    const int8_t *xb0 = qXt + (size_t)c * cols + (size_t)bl * 32;
                    const int8_t *xb1 = qXt + (size_t)(c + 1) * cols + (size_t)bl * 32;
                    int32x4_t acc = vdupq_n_s32(0);
                    acc = vmmlaq_s32(acc, A_lo_lo, vcombine_s8(vld1_s8(xb0),      vld1_s8(xb1)));
                    acc = vmmlaq_s32(acc, A_lo_hi, vcombine_s8(vld1_s8(xb0 + 8),  vld1_s8(xb1 + 8)));
                    acc = vmmlaq_s32(acc, A_hi_lo, vcombine_s8(vld1_s8(xb0 + 16), vld1_s8(xb1 + 16)));
                    acc = vmmlaq_s32(acc, A_hi_hi, vcombine_s8(vld1_s8(xb0 + 24), vld1_s8(xb1 + 24)));
                    int32_t t[4]; vst1q_s32(t, acc);
                    const int cc0 = corr[(size_t)c * nb + bl], cc1 = corr[(size_t)(c + 1) * nb + bl];
                    f0[c]     += s0 * (float)(t[0] - 8 * cc0);
                    f0[c + 1] += s0 * (float)(t[1] - 8 * cc1);
                    f1[c]     += s1 * (float)(t[2] - 8 * cc0);
                    f1[c + 1] += s1 * (float)(t[3] - 8 * cc1);
                }
                for (; c < B; c++) {
                    const int8_t *xb = qXt + (size_t)c * cols + (size_t)bl * 32;
                    const uint8_t *qa = w0[bl].qs, *qb = w1[bl].qs;
                    int64_t ta = 0, tb = 0;
                    for (int i = 0; i < 16; i++) {
                        ta += (qa[i] & 0x0F) * xb[2*i] + (qa[i] >> 4) * xb[2*i + 1];
                        tb += (qb[i] & 0x0F) * xb[2*i] + (qb[i] >> 4) * xb[2*i + 1];
                    }
                    const int cc = corr[(size_t)c * nb + bl];
                    f0[c] += s0 * (float)(ta - 8 * cc);
                    f1[c] += s1 * (float)(tb - 8 * cc);
                }
            }
            for (int c = 0; c < B; c++) {
                Y[(size_t)r * B + c]       = f0[c] * sx[c];
                Y[(size_t)(r + 1) * B + c] = f1[c] * sx[c];
            }
            continue;
        }
        int j = 0;
        for (; j + 1 < B; j += 2) {
            const int8_t *x0 = qXt + (size_t)j * cols, *x1 = qXt + (size_t)(j + 1) * cols;
            const int *c0 = corr + (size_t)j * nb,     *c1 = corr + (size_t)(j + 1) * nb;
            float f00 = 0, f01 = 0, f10 = 0, f11 = 0;
            for (int bl = 0; bl < nb; bl++) {
                uint8x16_t raw0 = vld1q_u8(w0[bl].qs), raw1 = vld1q_u8(w1[bl].qs);
                uint8x16x2_t z0 = vzipq_u8(vandq_u8(raw0, mask), vshrq_n_u8(raw0, 4));
                uint8x16x2_t z1 = vzipq_u8(vandq_u8(raw1, mask), vshrq_n_u8(raw1, 4));
                int8x16_t a0lo = vreinterpretq_s8_u8(z0.val[0]);
                int8x16_t a0hi = vreinterpretq_s8_u8(z0.val[1]);
                int8x16_t a1lo = vreinterpretq_s8_u8(z1.val[0]);
                int8x16_t a1hi = vreinterpretq_s8_u8(z1.val[1]);
                const int8_t *xb0 = x0 + (size_t)bl * 32, *xb1 = x1 + (size_t)bl * 32;
                int32x4_t acc = vdupq_n_s32(0);
                acc = vmmlaq_s32(acc, vcombine_s8(vget_low_s8(a0lo),  vget_low_s8(a1lo)),
                                       vcombine_s8(vld1_s8(xb0),      vld1_s8(xb1)));
                acc = vmmlaq_s32(acc, vcombine_s8(vget_high_s8(a0lo), vget_high_s8(a1lo)),
                                       vcombine_s8(vld1_s8(xb0 + 8),  vld1_s8(xb1 + 8)));
                acc = vmmlaq_s32(acc, vcombine_s8(vget_low_s8(a0hi),  vget_low_s8(a1hi)),
                                       vcombine_s8(vld1_s8(xb0 + 16), vld1_s8(xb1 + 16)));
                acc = vmmlaq_s32(acc, vcombine_s8(vget_high_s8(a0hi), vget_high_s8(a1hi)),
                                       vcombine_s8(vld1_s8(xb0 + 24), vld1_s8(xb1 + 24)));
                int32_t t[4]; vst1q_s32(t, acc);
                float s0 = qwen_f16_to_f32(w0[bl].scale_f16);
                float s1 = qwen_f16_to_f32(w1[bl].scale_f16);
                f00 += s0 * (float)(t[0] - 8 * c0[bl]);
                f01 += s0 * (float)(t[1] - 8 * c1[bl]);
                f10 += s1 * (float)(t[2] - 8 * c0[bl]);
                f11 += s1 * (float)(t[3] - 8 * c1[bl]);
            }
            Y[(size_t)r * B + j]           = f00 * sx[j];
            Y[(size_t)r * B + j + 1]       = f01 * sx[j + 1];
            Y[(size_t)(r + 1) * B + j]     = f10 * sx[j];
            Y[(size_t)(r + 1) * B + j + 1] = f11 * sx[j + 1];
        }
        for (; j < B; j++) {
            const int8_t *xj = qXt + (size_t)j * cols;
            const int *cj = corr + (size_t)j * nb;
            float fa = 0, fb = 0;
            for (int bl = 0; bl < nb; bl++) {
                int64_t ta = 0, tb = 0;
                const uint8_t *qa = w0[bl].qs, *qb = w1[bl].qs;
                const int8_t *xb = xj + (size_t)bl * 32;
                for (int i = 0; i < 16; i++) {
                    ta += (qa[i] & 0x0F) * xb[2*i] + (qa[i] >> 4) * xb[2*i + 1];
                    tb += (qb[i] & 0x0F) * xb[2*i] + (qb[i] >> 4) * xb[2*i + 1];
                }
                fa += qwen_f16_to_f32(w0[bl].scale_f16) * (float)(ta - 8 * cj[bl]);
                fb += qwen_f16_to_f32(w1[bl].scale_f16) * (float)(tb - 8 * cj[bl]);
            }
            Y[(size_t)r * B + j]       = fa * sx[j];
            Y[(size_t)(r + 1) * B + j] = fb * sx[j];
        }
    }
    for (; r < r1; r++) {
        const q4_0_block_t *wr = W + (size_t)r * nb;
        for (int j = 0; j < B; j++) {
            const int8_t *xj = qXt + (size_t)j * cols;
            const int *cj = corr + (size_t)j * nb;
            float f = 0;
            for (int bl = 0; bl < nb; bl++) {
                int64_t t = 0;
                const uint8_t *q = wr[bl].qs;
                const int8_t *xb = xj + (size_t)bl * 32;
                for (int i = 0; i < 16; i++)
                    t += (q[i] & 0x0F) * xb[2*i] + (q[i] >> 4) * xb[2*i + 1];
                f += qwen_f16_to_f32(wr[bl].scale_f16) * (float)(t - 8 * cj[bl]);
            }
            Y[(size_t)r * B + j] = f * sx[j];
        }
    }
}
typedef struct {
    float *Y; const q4_0_block_t *W; const int8_t *qXt; const float *sx; const int *corr;
    int rows, cols, B;
} q4_smmla_ctx;
static void q4_smmla_task(size_t tid, size_t nt, void *vc) {
    q4_smmla_ctx *c = (q4_smmla_ctx *)vc;
    int r0 = (int)(tid * (size_t)c->rows / nt), r1 = (int)((tid + 1) * (size_t)c->rows / nt);
    r0 &= ~1; if (tid + 1 < nt) r1 &= ~1;
    q4_matmat_smmla_slice(c->Y, c->W, c->qXt, c->sx, c->corr, r0, r1, c->cols, c->B);
}
#endif

void qwen_matmat_q4_0(float *Y, const q4_0_block_t *W, const float *X,
                      int rows, int cols, int B) {
    qwen_census_op(QWEN_PATH_MATMAT_Q4_0, rows, cols, B);
    if (B <= 0) return;
    if (B > 64) B = 64;
    if (qwen_mm_use(QWEN_MMK_KLEIDI_Q4, B, rows, cols) &&
        qwen_kleidi_matmul_q4(Y, (const void *)W, X, rows, cols, B)) {
        MMSTAT(QWEN_MMK_KLEIDI_Q4, rows, cols, B);
        return;
    }
#if defined(__AMX_INT8__) && defined(__AMX_TILE__)
    {
        if (qwen_mm_use(QWEN_MMK_Q4_AMX, B, rows, cols) && cols % Q4_0_BLOCK_SIZE == 0 &&
            qwen_amx_int8_ready()) {
            int nb = cols / Q4_0_BLOCK_SIZE;
            int8_t *qXt = mm_scratch_qx((size_t)B * cols);
            int *corr = mm_scratch_corr((size_t)B * nb);
            int8_t *pXt = NULL;
            if (qXt && corr) pXt = mm_scratch_pack((size_t)cols * (size_t)B);
            if (qXt && corr && pXt) {
                float sx[16];
                for (int b = 0; b < B; b++) {
                    sx[b] = quantize_act_int8_col(qXt + (size_t)b * cols, X, cols, B, b);
                    const int8_t *qb = qXt + (size_t)b * cols;
                    for (int bl = 0; bl < nb; bl++) {
                        int t = 0;
                        for (int k = 0; k < Q4_0_BLOCK_SIZE; k++) t += qb[bl * Q4_0_BLOCK_SIZE + k];
                        corr[(size_t)b * nb + bl] = -8 * t;
                    }
                }
                amx_pack_act_int8(pXt, qXt, cols, cols, B);
                int nt2 = g_n_threads;
                if (nt2 > 1 && rows >= 256) {
                    q4_amx_ctx c = { Y, W, pXt, qXt, sx, corr, rows, cols, B };
                    qwen_parallel((size_t)nt2, q4_amx_task, &c);
                } else {
                    q4_matmat_amx_slice(Y, W, pXt, qXt, sx, corr, 0, rows, cols, B);
                }
                return;
            }
        }
    }
#endif
#if defined(__AVX512VNNI__)
    {
        if (qwen_mm_use(QWEN_MMK_Q4_VNNI, B, rows, cols) && cols % Q4_0_BLOCK_SIZE == 0) {
            int nb = cols / Q4_0_BLOCK_SIZE;
            int8_t *qXt = mm_scratch_qx((size_t)B * cols);
            int *corr = mm_scratch_corr((size_t)B * nb);
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
                return;
            }
        }
    }
#endif

#if defined(__AVX2__)
    {
        if (qwen_mm_use(QWEN_MMK_Q4_AVX2, B, rows, cols) && cols % Q4_0_BLOCK_SIZE == 0) {
            int nb = cols / Q4_0_BLOCK_SIZE;
            int8_t *qXt = mm_scratch_qx((size_t)B * cols);
            int *corr = mm_scratch_corr((size_t)B * nb);
            if (qXt && corr) {
                float sx[16];
                for (int b = 0; b < B; b++) {
                    sx[b] = quantize_act_int8_col(qXt + (size_t)b * cols, X, cols, B, b);
                    const int8_t *qb = qXt + (size_t)b * cols;
                    for (int bl = 0; bl < nb; bl++) {
                        int t = 0;
                        for (int k = 0; k < Q4_0_BLOCK_SIZE; k++) t += qb[bl * Q4_0_BLOCK_SIZE + k];
                        corr[(size_t)b * nb + bl] = -8 * t;
                    }
                }
                int nt2 = g_n_threads;
                if (nt2 > 1 && rows >= 256) {
                    q4_amm_ctx c = { Y, W, qXt, sx, corr, rows, cols, B };
                    qwen_parallel((size_t)nt2, q4_amm_task, &c);
                } else {
                    q4_matmat_avx2_slice(Y, W, qXt, sx, corr, 0, rows, cols, B);
                }
                return;
            }
        }
    }
#endif
#if defined(__ARM_FEATURE_MATMUL_INT8)
    {
        if (qwen_mm_use(QWEN_MMK_Q4_SMMLA, B, rows, cols) && cols % Q4_0_BLOCK_SIZE == 0) {
            int nb = cols / Q4_0_BLOCK_SIZE;
            int8_t *qXt = mm_scratch_qx((size_t)B * cols);
            int *corr = mm_scratch_corr((size_t)B * nb);
            if (qXt && corr) {
                float sx[16];
                for (int b = 0; b < B; b++) {
                    sx[b] = quantize_act_int8_col(qXt + (size_t)b * cols, X, cols, B, b);
                    const int8_t *qb = qXt + (size_t)b * cols;
                    for (int bl = 0; bl < nb; bl++) {
                        int s = 0;
                        for (int k = 0; k < Q4_0_BLOCK_SIZE; k++) s += qb[bl * Q4_0_BLOCK_SIZE + k];
                        corr[(size_t)b * nb + bl] = s;
                    }
                }
                int nt2 = g_n_threads;
                if (nt2 > 1 && rows >= 256) {
                    q4_smmla_ctx c = { Y, W, qXt, sx, corr, rows, cols, B };
                    qwen_parallel((size_t)nt2, q4_smmla_task, &c);
                } else {
                    q4_matmat_smmla_slice(Y, W, qXt, sx, corr, 0, rows, cols, B);
                }
                return;
            }
        }
    }
#elif defined(__ARM_FEATURE_DOTPROD)
    if (cols % Q4_0_BLOCK_SIZE == 0) {
        float *xcol = mm_scratch_xcol((size_t)cols);
        float *ycol = mm_scratch_ycol((size_t)rows);
        if (xcol && ycol) {
            MMSTAT(QWEN_MMK_Q4_BMATVEC, rows, cols, B);
            for (int b = 0; b < B; b++) {
                for (int k = 0; k < cols; k++) xcol[k] = X[(size_t)k * B + b];
                qwen_matvec_q4_0(ycol, W, xcol, rows, cols);
                for (int r = 0; r < rows; r++) Y[(size_t)r * B + b] = ycol[r];
            }
            return;
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
    qwen_census_op(QWEN_PATH_MATVEC_BF16_QKV, q_dim + 2 * kv_dim, in_dim, 1);
    if (qwen_q8r_matmul(q, (const void *)Wq, x, q_dim,  in_dim, 1) &&
        qwen_q8r_matmul(k, (const void *)Wk, x, kv_dim, in_dim, 1) &&
        qwen_q8r_matmul(v, (const void *)Wv, x, kv_dim, in_dim, 1)) {
        MMSTAT(QWEN_MMK_Q8_REPACK_GEMV, q_dim + 2 * kv_dim, in_dim, 1);
        return;
    }
    MMSTAT(QWEN_MMK_BF16_GEMV, q_dim + 2 * kv_dim, in_dim, 1);

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

void qwen_quantize_bf16_to_int8(const uint16_t *src_bf16, int rows, int cols,
                                 int8_t *dst_int8, float *dst_scale) {
    for (int r = 0; r < rows; r++) {
        const uint16_t *row = src_bf16 + (size_t)r * cols;
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

static void int8_matvec_fused(float *y, const float *x, const int8_t *W,
                               const float *scale, int in_dim, int out_dim) {
    qwen_ftz_on();
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
            float32x4_t x0 = vld1q_f32(x + k);
            float32x4_t x1 = vld1q_f32(x + k + 4);
            float32x4_t x2 = vld1q_f32(x + k + 8);
            float32x4_t x3 = vld1q_f32(x + k + 12);

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

static void int8_matvec_sdot(float *y, const int8_t *qx, float sx,
                             const int8_t *W, const float *scale,
                             int in_dim, int out_dim) {
    int o = 0;
    for (; o + 3 < out_dim; o += 4) {
        const int8_t *w0 = W + (size_t)o * in_dim,       *w1 = W + (size_t)(o + 1) * in_dim;
        const int8_t *w2 = W + (size_t)(o + 2) * in_dim, *w3 = W + (size_t)(o + 3) * in_dim;
        int32x4_t a0 = vdupq_n_s32(0), a1 = vdupq_n_s32(0);
        int32x4_t a2 = vdupq_n_s32(0), a3 = vdupq_n_s32(0);
        int k = 0;
        for (; k + 15 < in_dim; k += 16) {
            int8x16_t xv = vld1q_s8(qx + k);
            a0 = vdotq_s32(a0, vld1q_s8(w0 + k), xv);
            a1 = vdotq_s32(a1, vld1q_s8(w1 + k), xv);
            a2 = vdotq_s32(a2, vld1q_s8(w2 + k), xv);
            a3 = vdotq_s32(a3, vld1q_s8(w3 + k), xv);
        }
        int32_t s0 = vaddvq_s32(a0), s1 = vaddvq_s32(a1);
        int32_t s2 = vaddvq_s32(a2), s3 = vaddvq_s32(a3);
        for (; k < in_dim; k++) {
            int32_t xv = qx[k];
            s0 += (int32_t)w0[k] * xv; s1 += (int32_t)w1[k] * xv;
            s2 += (int32_t)w2[k] * xv; s3 += (int32_t)w3[k] * xv;
        }
        y[o]     = (float)s0 * scale[o]     * sx;
        y[o + 1] = (float)s1 * scale[o + 1] * sx;
        y[o + 2] = (float)s2 * scale[o + 2] * sx;
        y[o + 3] = (float)s3 * scale[o + 3] * sx;
    }
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
#endif

#if defined(__AVX512VNNI__)

enum { QWEN_VNNI_ROWSUM_MAX = 512 };
typedef struct {
    const int8_t *weights;
    int rows;
    int cols;
    int32_t *sums;
} qwen_vnni_rowsum_entry_t;

static qwen_vnni_rowsum_entry_t g_vnni_rowsums[QWEN_VNNI_ROWSUM_MAX];
static atomic_int g_vnni_rowsum_count;
static pthread_mutex_t g_vnni_rowsum_mu = PTHREAD_MUTEX_INITIALIZER;

static int qwen_vnni_rowsum_enabled(void) {
    static atomic_int enabled = -1;
    int v = atomic_load_explicit(&enabled, memory_order_relaxed);
    if (v < 0) {
        const char *e = getenv("QWEN_NO_VNNI_ROWSUM");
        v = !(e && e[0] == '1');
        atomic_store_explicit(&enabled, v, memory_order_relaxed);
    }
    return v;
}

static int qwen_vnni_gemv_mr(void) {
    static atomic_int mr = -1;
    int v = atomic_load_explicit(&mr, memory_order_relaxed);
    if (v < 0) {
        v = qwen_mm_env_int("QWEN_VNNI_GEMV_MR", 2, 2, 4);
        if (v != 4) v = 2;
        atomic_store_explicit(&mr, v, memory_order_relaxed);
    }
    return v;
}

static const int32_t *qwen_vnni_row_sums(const int8_t *W, int rows, int cols) {
    if (!qwen_vnni_rowsum_enabled() || !W || rows <= 0 || cols <= 0) return NULL;
    const int full_cols = cols & ~63;
    if (full_cols == 0) return NULL;
    int n = atomic_load_explicit(&g_vnni_rowsum_count, memory_order_acquire);
    for (int i = 0; i < n; i++) {
        const qwen_vnni_rowsum_entry_t *e = &g_vnni_rowsums[i];
        if (e->weights == W && e->rows == rows && e->cols == cols) return e->sums;
    }

    pthread_mutex_lock(&g_vnni_rowsum_mu);
    n = atomic_load_explicit(&g_vnni_rowsum_count, memory_order_relaxed);
    for (int i = 0; i < n; i++) {
        qwen_vnni_rowsum_entry_t *e = &g_vnni_rowsums[i];
        if (e->weights == W && e->rows == rows && e->cols == cols) {
            const int32_t *sums = e->sums;
            pthread_mutex_unlock(&g_vnni_rowsum_mu);
            return sums;
        }
    }
    if (n >= QWEN_VNNI_ROWSUM_MAX) {
        pthread_mutex_unlock(&g_vnni_rowsum_mu);
        return NULL;
    }
    int32_t *sums = (int32_t *)malloc((size_t)rows * sizeof(*sums));
    if (!sums) {
        pthread_mutex_unlock(&g_vnni_rowsum_mu);
        return NULL;
    }
    for (int r = 0; r < rows; r++) {
        const int8_t *row = W + (size_t)r * cols;
        int32_t sum = 0;
        for (int k = 0; k < full_cols; k++) sum += (int32_t)row[k];
        sums[r] = sum;
    }
    g_vnni_rowsums[n].weights = W;
    g_vnni_rowsums[n].rows = rows;
    g_vnni_rowsums[n].cols = cols;
    g_vnni_rowsums[n].sums = sums;
    atomic_store_explicit(&g_vnni_rowsum_count, n + 1, memory_order_release);
    pthread_mutex_unlock(&g_vnni_rowsum_mu);
    return sums;
}

void qwen_vnni_row_sums_reset(void) {
    pthread_mutex_lock(&g_vnni_rowsum_mu);
    int n = atomic_load_explicit(&g_vnni_rowsum_count, memory_order_relaxed);
    for (int i = 0; i < n; i++) {
        free(g_vnni_rowsums[i].sums);
        g_vnni_rowsums[i].weights = NULL;
        g_vnni_rowsums[i].rows = 0;
        g_vnni_rowsums[i].cols = 0;
        g_vnni_rowsums[i].sums = NULL;
    }
    atomic_store_explicit(&g_vnni_rowsum_count, 0, memory_order_release);
    pthread_mutex_unlock(&g_vnni_rowsum_mu);
}

static int qwen_vnni_act_quant_enabled(void) {
    static atomic_int enabled = -1;
    int v = atomic_load_explicit(&enabled, memory_order_relaxed);
    if (v < 0) {
        const char *e = getenv("QWEN_NO_VNNI_ACT_QUANT");
        v = !(e && e[0] == '1');
        atomic_store_explicit(&enabled, v, memory_order_relaxed);
    }
    return v;
}

static float quantize_act_int8_x86_scalar(int8_t *qx, const float *x, int n) {
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

static float quantize_act_int8_x86(int8_t *qx, const float *x, int n) {
#if defined(__AVX512F__) && defined(__AVX512BW__)
    if (qwen_vnni_act_quant_enabled() && n >= 16) {
        const __m512i absmask = _mm512_set1_epi32(0x7FFFFFFF);
        __m512 vmax = _mm512_setzero_ps();
        int i = 0;
        for (; i + 16 <= n; i += 16) {
            __m512 v = _mm512_loadu_ps(x + i);
            vmax = _mm512_max_ps(vmax, _mm512_castsi512_ps(
            _mm512_and_si512(_mm512_castps_si512(v), absmask)));
        }
        float amax = _mm512_reduce_max_ps(vmax);
        for (; i < n; i++) {
            float a = fabsf(x[i]);
            if (a > amax) amax = a;
        }
        if (amax == 0.0f) { memset(qx, 0, (size_t)n); return 0.0f; }

        const __m512 inv = _mm512_set1_ps(127.0f / amax);
        i = 0;
        for (; i + 16 <= n; i += 16) {
            __m512 v = _mm512_loadu_ps(x + i);
            __m512i q = _mm512_cvtps_epi32(_mm512_mul_ps(v, inv));
            q = _mm512_max_epi32(q, _mm512_set1_epi32(-128));
            q = _mm512_min_epi32(q, _mm512_set1_epi32(127));
            _mm_storeu_si128((__m128i *)(void *)(qx + i), _mm512_cvtsepi32_epi8(q));
        }
        for (; i < n; i++) {
            int v = (int)lrintf(x[i] * (127.0f / amax));
            qx[i] = (int8_t)(v > 127 ? 127 : (v < -128 ? -128 : v));
        }
        return amax / 127.0f;
    }
#endif
    return quantize_act_int8_x86_scalar(qx, x, n);
}

/* A1: opt-in pre-biased (unsigned) activation for the VNNI B=1 GEMV path.
 *
 * vpdpbusd takes an UNSIGNED first operand, so the signed int8 activation has
 * to be shifted by +128 before it can be fed to the instruction.  The default
 * kernels redo that _mm512_add_epi8 for every output-row group and every K
 * chunk, i.e. out_dim/MR times more often than necessary.  With QWEN_VNNI_UACT=1
 * the shift is applied once, while the activation is quantized, and the kernels
 * consume the biased bytes directly.  The integer math is unchanged: the
 * existing "- 128 * row_sum" correction already accounts for the bias, so the
 * results are bit-identical to the signed path.
 */
static int qwen_vnni_uact_enabled(void) {
    static atomic_int enabled = -1;
    int v = atomic_load_explicit(&enabled, memory_order_relaxed);
    if (v < 0) {
        v = qwen_mm_env_int("QWEN_VNNI_UACT", 0, 0, 1);
        atomic_store_explicit(&enabled, v, memory_order_relaxed);
    }
    return v;
}

/* Same quantization as quantize_act_int8_x86(), storing q + 128 as u8. */
static float quantize_act_u8_x86(uint8_t *ux, const float *x, int n) {
#if defined(__AVX512F__) && defined(__AVX512BW__)
    if (qwen_vnni_act_quant_enabled() && n >= 16) {
        const __m512i absmask = _mm512_set1_epi32(0x7FFFFFFF);
        __m512 vmax = _mm512_setzero_ps();
        int i = 0;
        for (; i + 16 <= n; i += 16) {
            __m512 v = _mm512_loadu_ps(x + i);
            vmax = _mm512_max_ps(vmax, _mm512_castsi512_ps(
            _mm512_and_si512(_mm512_castps_si512(v), absmask)));
        }
        float amax = _mm512_reduce_max_ps(vmax);
        for (; i < n; i++) {
            float a = fabsf(x[i]);
            if (a > amax) amax = a;
        }
        if (amax == 0.0f) { memset(ux, 128, (size_t)n); return 0.0f; }

        const __m512 inv = _mm512_set1_ps(127.0f / amax);
        const __m128i bias = _mm_set1_epi8((char)128);
        i = 0;
        for (; i + 16 <= n; i += 16) {
            __m512 v = _mm512_loadu_ps(x + i);
            __m512i q = _mm512_cvtps_epi32(_mm512_mul_ps(v, inv));
            q = _mm512_max_epi32(q, _mm512_set1_epi32(-128));
            q = _mm512_min_epi32(q, _mm512_set1_epi32(127));
            _mm_storeu_si128((__m128i *)(void *)(ux + i),
                             _mm_add_epi8(_mm512_cvtsepi32_epi8(q), bias));
        }
        for (; i < n; i++) {
            int v = (int)lrintf(x[i] * (127.0f / amax));
            v = v > 127 ? 127 : (v < -128 ? -128 : v);
            ux[i] = (uint8_t)(v + 128);
        }
        return amax / 127.0f;
    }
#endif
    {
        float amax = 0.0f;
        for (int i = 0; i < n; i++) { float a = fabsf(x[i]); if (a > amax) amax = a; }
        if (amax == 0.0f) { memset(ux, 128, (size_t)n); return 0.0f; }
        float inv = 127.0f / amax;
        for (int i = 0; i < n; i++) {
            int v = (int)lrintf(x[i] * inv);
            v = v > 127 ? 127 : (v < -128 ? -128 : v);
            ux[i] = (uint8_t)(v + 128);
        }
        return amax / 127.0f;
    }
}

static void int8_matvec_vnni_legacy(float *y, const int8_t *qx, float sx,
                                    const int8_t *W, const float *scale,
                                    int in_dim, int out_dim) {
    const __m512i v128 = _mm512_set1_epi8((char)128);
    const __m512i ones = _mm512_set1_epi8(1);
    const int phase_on = qwen_vnni_phase_timing_enabled();
    int o = 0;
    for (; o + 1 < out_dim; o += 2) {
        const int8_t *w0 = W + (size_t)o * in_dim;
        const int8_t *w1 = W + (size_t)(o + 1) * in_dim;
        __m512i acc0 = _mm512_setzero_si512(), acc1 = _mm512_setzero_si512();
        __m512i ws0  = _mm512_setzero_si512(), ws1  = _mm512_setzero_si512();
        double dot_t0 = phase_on ? qwen_mm_now_s() : 0.0;
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
        if (phase_on) qwen_vnni_phase_note(QWEN_VP_DOT, (long long)2 * in_dim, dot_t0);
        double ep_t0 = phase_on ? qwen_mm_now_s() : 0.0;
        y[o]     = (float)s0 * scale[o]     * sx;
        y[o + 1] = (float)s1 * scale[o + 1] * sx;
        if (phase_on) qwen_vnni_phase_note(QWEN_VP_EPILOGUE, 0, ep_t0);
    }
    if (o < out_dim) {
        const int8_t *w0 = W + (size_t)o * in_dim;
        __m512i acc0 = _mm512_setzero_si512(), ws0 = _mm512_setzero_si512();
        double dot_t0 = phase_on ? qwen_mm_now_s() : 0.0;
        int k = 0;
        for (; k + 64 <= in_dim; k += 64) {
            __m512i ua  = _mm512_add_epi8(_mm512_loadu_si512((const void *)(qx + k)), v128);
            __m512i wv0 = _mm512_loadu_si512((const void *)(w0 + k));
            acc0 = _mm512_dpbusd_epi32(acc0, ua, wv0);
            ws0  = _mm512_dpbusd_epi32(ws0, ones, wv0);
        }
        int s0 = _mm512_reduce_add_epi32(acc0) - 128 * _mm512_reduce_add_epi32(ws0);
        for (; k < in_dim; k++) s0 += (int)w0[k] * qx[k];
        if (phase_on) qwen_vnni_phase_note(QWEN_VP_DOT, in_dim, dot_t0);
        double ep_t0 = phase_on ? qwen_mm_now_s() : 0.0;
        y[o] = (float)s0 * scale[o] * sx;
        if (phase_on) qwen_vnni_phase_note(QWEN_VP_EPILOGUE, 0, ep_t0);
    }
}

static void int8_matvec_vnni_rowsum(float *y, const int8_t *qx, float sx,
                                    const int8_t *W, const float *scale,
                                    const int32_t *row_sums,
                                    int in_dim, int out_dim) {
    const __m512i v128 = _mm512_set1_epi8((char)128);
    const int phase_on = qwen_vnni_phase_timing_enabled();
    int o = 0;
    for (; o + 1 < out_dim; o += 2) {
        const int8_t *w0 = W + (size_t)o * in_dim;
        const int8_t *w1 = W + (size_t)(o + 1) * in_dim;
        __m512i acc0 = _mm512_setzero_si512(), acc1 = _mm512_setzero_si512();
        double dot_t0 = phase_on ? qwen_mm_now_s() : 0.0;
        int k = 0;
        for (; k + 64 <= in_dim; k += 64) {
            __m512i ua  = _mm512_add_epi8(_mm512_loadu_si512((const void *)(qx + k)), v128);
            __m512i wv0 = _mm512_loadu_si512((const void *)(w0 + k));
            __m512i wv1 = _mm512_loadu_si512((const void *)(w1 + k));
            acc0 = _mm512_dpbusd_epi32(acc0, ua, wv0);
            acc1 = _mm512_dpbusd_epi32(acc1, ua, wv1);
        }
        int s0 = _mm512_reduce_add_epi32(acc0) - 128 * row_sums[o];
        int s1 = _mm512_reduce_add_epi32(acc1) - 128 * row_sums[o + 1];
        for (; k < in_dim; k++) { s0 += (int)w0[k] * qx[k]; s1 += (int)w1[k] * qx[k]; }
        if (phase_on) qwen_vnni_phase_note(QWEN_VP_DOT, (long long)2 * in_dim, dot_t0);
        double ep_t0 = phase_on ? qwen_mm_now_s() : 0.0;
        y[o]     = (float)s0 * scale[o]     * sx;
        y[o + 1] = (float)s1 * scale[o + 1] * sx;
        if (phase_on) qwen_vnni_phase_note(QWEN_VP_EPILOGUE, 0, ep_t0);
    }
    if (o < out_dim) {
        const int8_t *w0 = W + (size_t)o * in_dim;
        __m512i acc0 = _mm512_setzero_si512();
        double dot_t0 = phase_on ? qwen_mm_now_s() : 0.0;
        int k = 0;
        for (; k + 64 <= in_dim; k += 64) {
            __m512i ua  = _mm512_add_epi8(_mm512_loadu_si512((const void *)(qx + k)), v128);
            __m512i wv0 = _mm512_loadu_si512((const void *)(w0 + k));
            acc0 = _mm512_dpbusd_epi32(acc0, ua, wv0);
        }
        int s0 = _mm512_reduce_add_epi32(acc0) - 128 * row_sums[o];
        for (; k < in_dim; k++) s0 += (int)w0[k] * qx[k];
        if (phase_on) qwen_vnni_phase_note(QWEN_VP_DOT, in_dim, dot_t0);
        double ep_t0 = phase_on ? qwen_mm_now_s() : 0.0;
        y[o] = (float)s0 * scale[o] * sx;
        if (phase_on) qwen_vnni_phase_note(QWEN_VP_EPILOGUE, 0, ep_t0);
    }
}

static void int8_matvec_vnni_rowsum_mr4(float *y, const int8_t *qx, float sx,
                                        const int8_t *W, const float *scale,
                                        const int32_t *row_sums,
                                        int in_dim, int out_dim) {
    const __m512i v128 = _mm512_set1_epi8((char)128);
    const int phase_on = qwen_vnni_phase_timing_enabled();
    int o = 0;
    for (; o + 3 < out_dim; o += 4) {
        const int8_t *w0 = W + (size_t)o * in_dim;
        const int8_t *w1 = w0 + in_dim;
        const int8_t *w2 = w1 + in_dim;
        const int8_t *w3 = w2 + in_dim;
        __m512i acc0 = _mm512_setzero_si512(), acc1 = _mm512_setzero_si512();
        __m512i acc2 = _mm512_setzero_si512(), acc3 = _mm512_setzero_si512();
        double dot_t0 = phase_on ? qwen_mm_now_s() : 0.0;
        int k = 0;
        for (; k + 64 <= in_dim; k += 64) {
            __m512i ua = _mm512_add_epi8(
                _mm512_loadu_si512((const void *)(qx + k)), v128);
            acc0 = _mm512_dpbusd_epi32(acc0, ua,
                                       _mm512_loadu_si512((const void *)(w0 + k)));
            acc1 = _mm512_dpbusd_epi32(acc1, ua,
                                       _mm512_loadu_si512((const void *)(w1 + k)));
            acc2 = _mm512_dpbusd_epi32(acc2, ua,
                                       _mm512_loadu_si512((const void *)(w2 + k)));
            acc3 = _mm512_dpbusd_epi32(acc3, ua,
                                       _mm512_loadu_si512((const void *)(w3 + k)));
        }
        int s0 = _mm512_reduce_add_epi32(acc0) - 128 * row_sums[o];
        int s1 = _mm512_reduce_add_epi32(acc1) - 128 * row_sums[o + 1];
        int s2 = _mm512_reduce_add_epi32(acc2) - 128 * row_sums[o + 2];
        int s3 = _mm512_reduce_add_epi32(acc3) - 128 * row_sums[o + 3];
        for (; k < in_dim; k++) {
            const int q = qx[k];
            s0 += (int)w0[k] * q;
            s1 += (int)w1[k] * q;
            s2 += (int)w2[k] * q;
            s3 += (int)w3[k] * q;
        }
        if (phase_on) qwen_vnni_phase_note(QWEN_VP_DOT, (long long)4 * in_dim, dot_t0);
        double ep_t0 = phase_on ? qwen_mm_now_s() : 0.0;
        y[o]     = (float)s0 * scale[o]     * sx;
        y[o + 1] = (float)s1 * scale[o + 1] * sx;
        y[o + 2] = (float)s2 * scale[o + 2] * sx;
        y[o + 3] = (float)s3 * scale[o + 3] * sx;
        if (phase_on) qwen_vnni_phase_note(QWEN_VP_EPILOGUE, 0, ep_t0);
    }
    if (o < out_dim)
        int8_matvec_vnni_rowsum(y + o, qx, sx, W + (size_t)o * in_dim,
                                scale + o, row_sums + o, in_dim, out_dim - o);
}

static void int8_matvec_vnni_legacy_mr4(float *y, const int8_t *qx, float sx,
                                        const int8_t *W, const float *scale,
                                        int in_dim, int out_dim) {
    const __m512i v128 = _mm512_set1_epi8((char)128);
    const __m512i ones = _mm512_set1_epi8(1);
    const int phase_on = qwen_vnni_phase_timing_enabled();
    int o = 0;
    for (; o + 3 < out_dim; o += 4) {
        const int8_t *w0 = W + (size_t)o * in_dim;
        const int8_t *w1 = w0 + in_dim;
        const int8_t *w2 = w1 + in_dim;
        const int8_t *w3 = w2 + in_dim;
        __m512i acc0 = _mm512_setzero_si512(), acc1 = _mm512_setzero_si512();
        __m512i acc2 = _mm512_setzero_si512(), acc3 = _mm512_setzero_si512();
        __m512i ws0 = _mm512_setzero_si512(), ws1 = _mm512_setzero_si512();
        __m512i ws2 = _mm512_setzero_si512(), ws3 = _mm512_setzero_si512();
        double dot_t0 = phase_on ? qwen_mm_now_s() : 0.0;
        int k = 0;
        for (; k + 64 <= in_dim; k += 64) {
            __m512i ua = _mm512_add_epi8(
                _mm512_loadu_si512((const void *)(qx + k)), v128);
            __m512i wv0 = _mm512_loadu_si512((const void *)(w0 + k));
            __m512i wv1 = _mm512_loadu_si512((const void *)(w1 + k));
            __m512i wv2 = _mm512_loadu_si512((const void *)(w2 + k));
            __m512i wv3 = _mm512_loadu_si512((const void *)(w3 + k));
            acc0 = _mm512_dpbusd_epi32(acc0, ua, wv0);
            acc1 = _mm512_dpbusd_epi32(acc1, ua, wv1);
            acc2 = _mm512_dpbusd_epi32(acc2, ua, wv2);
            acc3 = _mm512_dpbusd_epi32(acc3, ua, wv3);
            ws0 = _mm512_dpbusd_epi32(ws0, ones, wv0);
            ws1 = _mm512_dpbusd_epi32(ws1, ones, wv1);
            ws2 = _mm512_dpbusd_epi32(ws2, ones, wv2);
            ws3 = _mm512_dpbusd_epi32(ws3, ones, wv3);
        }
        int s0 = _mm512_reduce_add_epi32(acc0) - 128 * _mm512_reduce_add_epi32(ws0);
        int s1 = _mm512_reduce_add_epi32(acc1) - 128 * _mm512_reduce_add_epi32(ws1);
        int s2 = _mm512_reduce_add_epi32(acc2) - 128 * _mm512_reduce_add_epi32(ws2);
        int s3 = _mm512_reduce_add_epi32(acc3) - 128 * _mm512_reduce_add_epi32(ws3);
        for (; k < in_dim; k++) {
            const int q = qx[k];
            s0 += (int)w0[k] * q;
            s1 += (int)w1[k] * q;
            s2 += (int)w2[k] * q;
            s3 += (int)w3[k] * q;
        }
        if (phase_on) qwen_vnni_phase_note(QWEN_VP_DOT, (long long)4 * in_dim, dot_t0);
        double ep_t0 = phase_on ? qwen_mm_now_s() : 0.0;
        y[o]     = (float)s0 * scale[o]     * sx;
        y[o + 1] = (float)s1 * scale[o + 1] * sx;
        y[o + 2] = (float)s2 * scale[o + 2] * sx;
        y[o + 3] = (float)s3 * scale[o + 3] * sx;
        if (phase_on) qwen_vnni_phase_note(QWEN_VP_EPILOGUE, 0, ep_t0);
    }
    if (o < out_dim)
        int8_matvec_vnni_legacy(y + o, qx, sx, W + (size_t)o * in_dim,
                                scale + o, in_dim, out_dim - o);
}

/* A1 kernels: identical to int8_matvec_vnni_rowsum{,_mr4} except that the
 * activation arrives pre-biased, so the per-row-group _mm512_add_epi8 and the
 * extra activation load are gone from the inner loop. */
static void int8_matvec_vnni_urowsum(float *y, const uint8_t *ux, float sx,
                                     const int8_t *W, const float *scale,
                                     const int32_t *row_sums,
                                     int in_dim, int out_dim) {
    int o = 0;
    for (; o + 1 < out_dim; o += 2) {
        const int8_t *w0 = W + (size_t)o * in_dim;
        const int8_t *w1 = w0 + in_dim;
        __m512i acc0 = _mm512_setzero_si512(), acc1 = _mm512_setzero_si512();
        int k = 0;
        for (; k + 64 <= in_dim; k += 64) {
            __m512i ua = _mm512_loadu_si512((const void *)(ux + k));
            acc0 = _mm512_dpbusd_epi32(acc0, ua,
                                       _mm512_loadu_si512((const void *)(w0 + k)));
            acc1 = _mm512_dpbusd_epi32(acc1, ua,
                                       _mm512_loadu_si512((const void *)(w1 + k)));
        }
        int s0 = _mm512_reduce_add_epi32(acc0) - 128 * row_sums[o];
        int s1 = _mm512_reduce_add_epi32(acc1) - 128 * row_sums[o + 1];
        for (; k < in_dim; k++) {
            const int q = (int)ux[k] - 128;
            s0 += (int)w0[k] * q;
            s1 += (int)w1[k] * q;
        }
        y[o]     = (float)s0 * scale[o]     * sx;
        y[o + 1] = (float)s1 * scale[o + 1] * sx;
    }
    if (o < out_dim) {
        const int8_t *w0 = W + (size_t)o * in_dim;
        __m512i acc0 = _mm512_setzero_si512();
        int k = 0;
        for (; k + 64 <= in_dim; k += 64)
            acc0 = _mm512_dpbusd_epi32(
                acc0, _mm512_loadu_si512((const void *)(ux + k)),
                _mm512_loadu_si512((const void *)(w0 + k)));
        int s0 = _mm512_reduce_add_epi32(acc0) - 128 * row_sums[o];
        for (; k < in_dim; k++) s0 += (int)w0[k] * ((int)ux[k] - 128);
        y[o] = (float)s0 * scale[o] * sx;
    }
}

static void int8_matvec_vnni_urowsum_mr4(float *y, const uint8_t *ux, float sx,
                                         const int8_t *W, const float *scale,
                                         const int32_t *row_sums,
                                         int in_dim, int out_dim) {
    int o = 0;
    for (; o + 3 < out_dim; o += 4) {
        const int8_t *w0 = W + (size_t)o * in_dim;
        const int8_t *w1 = w0 + in_dim;
        const int8_t *w2 = w1 + in_dim;
        const int8_t *w3 = w2 + in_dim;
        __m512i acc0 = _mm512_setzero_si512(), acc1 = _mm512_setzero_si512();
        __m512i acc2 = _mm512_setzero_si512(), acc3 = _mm512_setzero_si512();
        int k = 0;
        for (; k + 64 <= in_dim; k += 64) {
            __m512i ua = _mm512_loadu_si512((const void *)(ux + k));
            acc0 = _mm512_dpbusd_epi32(acc0, ua,
                                       _mm512_loadu_si512((const void *)(w0 + k)));
            acc1 = _mm512_dpbusd_epi32(acc1, ua,
                                       _mm512_loadu_si512((const void *)(w1 + k)));
            acc2 = _mm512_dpbusd_epi32(acc2, ua,
                                       _mm512_loadu_si512((const void *)(w2 + k)));
            acc3 = _mm512_dpbusd_epi32(acc3, ua,
                                       _mm512_loadu_si512((const void *)(w3 + k)));
        }
        int s0 = _mm512_reduce_add_epi32(acc0) - 128 * row_sums[o];
        int s1 = _mm512_reduce_add_epi32(acc1) - 128 * row_sums[o + 1];
        int s2 = _mm512_reduce_add_epi32(acc2) - 128 * row_sums[o + 2];
        int s3 = _mm512_reduce_add_epi32(acc3) - 128 * row_sums[o + 3];
        for (; k < in_dim; k++) {
            const int q = (int)ux[k] - 128;
            s0 += (int)w0[k] * q;
            s1 += (int)w1[k] * q;
            s2 += (int)w2[k] * q;
            s3 += (int)w3[k] * q;
        }
        y[o]     = (float)s0 * scale[o]     * sx;
        y[o + 1] = (float)s1 * scale[o + 1] * sx;
        y[o + 2] = (float)s2 * scale[o + 2] * sx;
        y[o + 3] = (float)s3 * scale[o + 3] * sx;
    }
    if (o < out_dim)
        int8_matvec_vnni_urowsum(y + o, ux, sx, W + (size_t)o * in_dim,
                                 scale + o, row_sums + o, in_dim, out_dim - o);
}

static void int8_matvec_vnni_u(float *y, const uint8_t *ux, float sx,
                               const int8_t *W, const float *scale,
                               const int32_t *row_sums,
                               int in_dim, int out_dim) {
    if (qwen_vnni_gemv_mr() == 4)
        int8_matvec_vnni_urowsum_mr4(y, ux, sx, W, scale, row_sums,
                                     in_dim, out_dim);
    else
        int8_matvec_vnni_urowsum(y, ux, sx, W, scale, row_sums, in_dim, out_dim);
}

static void int8_matvec_vnni(float *y, const int8_t *qx, float sx,
                             const int8_t *W, const float *scale,
                             const int32_t *row_sums,
                             int in_dim, int out_dim) {
    if (row_sums) {
        if (qwen_vnni_gemv_mr() == 4)
            int8_matvec_vnni_rowsum_mr4(y, qx, sx, W, scale, row_sums,
                                        in_dim, out_dim);
        else
            int8_matvec_vnni_rowsum(y, qx, sx, W, scale, row_sums, in_dim, out_dim);
        return;
    }
    if (qwen_vnni_gemv_mr() == 4)
        int8_matvec_vnni_legacy_mr4(y, qx, sx, W, scale, in_dim, out_dim);
    else
        int8_matvec_vnni_legacy(y, qx, sx, W, scale, in_dim, out_dim);
}

typedef struct {
    float *y; const int8_t *qx; float sx; const int8_t *W; const float *scale;
    const int32_t *row_sums; int rows, cols; const uint8_t *ux;
} int8_vnni_ctx;
static void int8_vnni_task(size_t tid, size_t nt, void *vc) {
    int8_vnni_ctx *c = (int8_vnni_ctx *)vc;
    int r0 = (int)(tid * (size_t)c->rows / nt);
    int r1 = (int)((tid + 1) * (size_t)c->rows / nt);
    if (c->ux) {
        int8_matvec_vnni_u(c->y + r0, c->ux, c->sx, c->W + (size_t)r0 * c->cols,
                           c->scale + r0, c->row_sums + r0, c->cols, r1 - r0);
        return;
    }
    int8_matvec_vnni(c->y + r0, c->qx, c->sx, c->W + (size_t)r0 * c->cols,
                     c->scale + r0, c->row_sums ? c->row_sums + r0 : NULL,
                     c->cols, r1 - r0);
}

typedef struct {
    float *q, *k, *v;
    const int8_t *qx; float sx;
    const int8_t *Wq, *Wk, *Wv;
    const float *sq, *sk, *sv;
    const int32_t *row_sums_q, *row_sums_k, *row_sums_v;
    int in_dim, q_dim, kv_dim;
    const uint8_t *ux;
} int8_qkv_vnni_ctx;

static void int8_qkv_vnni_task(size_t tid, size_t nt, void *vc) {
    int8_qkv_vnni_ctx *c = (int8_qkv_vnni_ctx *)vc;
    const int total = c->q_dim + 2 * c->kv_dim;
    const int g0 = (int)(tid * (size_t)total / nt);
    const int g1 = (int)((tid + 1) * (size_t)total / nt);
    const struct { float *y; const int8_t *W; const float *scale; int base, rows; } seg[3] = {
        { c->q, c->Wq, c->sq, 0,                        c->q_dim  },
        { c->k, c->Wk, c->sk, c->q_dim,                 c->kv_dim },
        { c->v, c->Wv, c->sv, c->q_dim + c->kv_dim,     c->kv_dim },
    };
    for (int i = 0; i < 3; i++) {
        const int lo = seg[i].base > g0 ? seg[i].base : g0;
        const int hi0 = seg[i].base + seg[i].rows;
        const int hi = hi0 < g1 ? hi0 : g1;
        if (hi <= lo) continue;
        const int r0 = lo - seg[i].base;
        const int32_t *row_sums = i == 0 ? c->row_sums_q :
                                   (i == 1 ? c->row_sums_k : c->row_sums_v);
        if (c->ux) {
            int8_matvec_vnni_u(seg[i].y + r0, c->ux, c->sx,
                               seg[i].W + (size_t)r0 * c->in_dim,
                               seg[i].scale + r0, row_sums + r0,
                               c->in_dim, hi - lo);
            continue;
        }
        int8_matvec_vnni(seg[i].y + r0, c->qx, c->sx,
                         seg[i].W + (size_t)r0 * c->in_dim, seg[i].scale + r0,
                         row_sums ? row_sums + r0 : NULL,
                         c->in_dim, hi - lo);
    }
}

static int qwen_vnni_qkv_disabled(void) {
    static atomic_int disabled = -1;
    int v = atomic_load_explicit(&disabled, memory_order_relaxed);
    if (v < 0) {
        const char *no_vnni = getenv("QWEN_NO_VNNI");
        const char *no_qkv = getenv("QWEN_NO_VNNI_QKV");
        v = (no_vnni && no_vnni[0] == '1') || (no_qkv && no_qkv[0] == '1');
        atomic_store_explicit(&disabled, v, memory_order_relaxed);
    }
    return v;
}
#endif

#if !defined(__AVX512VNNI__)
void qwen_vnni_row_sums_reset(void) {}
int qwen_vnni_prepack_weight(const int8_t *source, int rows, int cols) {
    (void)source; (void)rows; (void)cols;
    return 0;
}
void qwen_vnni_prepack_stats(int *n_packed, size_t *bytes) {
    if (n_packed) *n_packed = 0;
    if (bytes) *bytes = 0;
}
void qwen_vnni_weight_cache_reset(void) {}
#endif

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

typedef struct {
    float *q, *k, *v;
    const int8_t *qx; float sx;
    const int8_t *Wq, *Wk, *Wv;
    const float *sq, *sk, *sv;
    int in_dim, q_dim, kv_dim;
} int8_qkv_sdot_ctx;
static void int8_qkv_sdot_task(size_t tid, size_t nt, void *vc) {
    int8_qkv_sdot_ctx *c = (int8_qkv_sdot_ctx *)vc;
    const int total = c->q_dim + 2 * c->kv_dim;
    int g0 = (int)(tid * (size_t)total / nt);
    int g1 = (int)((tid + 1) * (size_t)total / nt);
    const struct { float *y; const int8_t *W; const float *sc; int base, rows; } seg[3] = {
        { c->q, c->Wq, c->sq, 0,                        c->q_dim  },
        { c->k, c->Wk, c->sk, c->q_dim,                 c->kv_dim },
        { c->v, c->Wv, c->sv, c->q_dim + c->kv_dim,     c->kv_dim },
    };
    for (int i = 0; i < 3; i++) {
        int lo = seg[i].base > g0 ? seg[i].base : g0;
        int hi = (seg[i].base + seg[i].rows) < g1 ? (seg[i].base + seg[i].rows) : g1;
        if (hi <= lo) continue;
        int r0 = lo - seg[i].base, nrows = hi - lo;
        int8_matvec_sdot(seg[i].y + r0, c->qx, c->sx,
                         seg[i].W + (size_t)r0 * c->in_dim, seg[i].sc + r0,
                         c->in_dim, nrows);
    }
}
#endif

void qwen_matvec_int8(float *y, const int8_t *W, const float *scale,
                      const float *x, int rows, int cols) {
    qwen_census_op(QWEN_PATH_MATVEC_INT8, rows, cols, 1);
    const int kt_on = qwen_kernel_timing_enabled();
    const double kt_t0 = kt_on ? qwen_mm_now_s() : 0.0;
    if (kai_i8_try(y, W, scale, x, rows, cols, 1)) {
        MMSTAT(QWEN_MMK_KLEIDI_I8_GEMV, rows, cols, 1);
        goto qwen_matvec_int8_timed_done;
    }
    MMSTAT(QWEN_MMK_INT8_GEMV, rows, cols, 1);

#if defined(__AVX512VNNI__)
    enum { QXV_MAX = 8192 };
    const int vp_on = qwen_vnni_phase_timing_enabled();
    static atomic_int vnni_off = -1;
    int vnni_o = atomic_load_explicit(&vnni_off, memory_order_relaxed);
    if (vnni_o < 0) { const char *e = getenv("QWEN_NO_VNNI"); vnni_o = (e && e[0] == '1'); atomic_store_explicit(&vnni_off, vnni_o, memory_order_relaxed); }
    if (!vnni_o && cols <= QXV_MAX) {
        qwen_census_leaf(QWEN_LEAF_VNNI);
        int8_t qx_buf[QXV_MAX];
        double vp_rs = vp_on ? qwen_mm_now_s() : 0.0;
        const int32_t *row_sums = qwen_vnni_row_sums(W, rows, cols);
        if (vp_on) qwen_vnni_phase_note(QWEN_VP_ROWSUM, 0, vp_rs);
        const int uact = row_sums && qwen_vnni_uact_enabled();
        const uint8_t *ux = uact ? (const uint8_t *)qx_buf : NULL;
        double vp_t0 = vp_on ? qwen_mm_now_s() : 0.0;
        float sx = uact ? quantize_act_u8_x86((uint8_t *)qx_buf, x, cols)
                        : quantize_act_int8_x86(qx_buf, x, cols);
        if (vp_on) qwen_vnni_phase_note(
            QWEN_VP_QUANT, (long long)cols * (long long)sizeof(float), vp_t0);
        int nt = g_n_threads;
        if (nt > 1 && rows >= 256) {
            int8_vnni_ctx c = { y, qx_buf, sx, W, scale, row_sums, rows, cols, ux };
            vp_t0 = vp_on ? qwen_mm_now_s() : 0.0;
            qwen_parallel((size_t)nt, int8_vnni_task, &c);
            if (vp_on) qwen_vnni_phase_note(QWEN_VP_PARALLEL, 0, vp_t0);
            goto qwen_matvec_int8_timed_done;
        }
        if (uact) int8_matvec_vnni_u(y, ux, sx, W, scale, row_sums, cols, rows);
        else      int8_matvec_vnni(y, qx_buf, sx, W, scale, row_sums, cols, rows);
        goto qwen_matvec_int8_timed_done;
    }
#endif
#if defined(__ARM_FEATURE_DOTPROD)
    enum { QX_MAX = 8192 };
    static atomic_int sdot_off = -1;
    int sdot_o = atomic_load_explicit(&sdot_off, memory_order_relaxed);
    if (sdot_o < 0) { const char *e = getenv("QWEN_NO_SDOT"); sdot_o = (e && e[0] == '1'); atomic_store_explicit(&sdot_off, sdot_o, memory_order_relaxed); }
    if (!sdot_o && cols <= QX_MAX) {
        int8_t qx_buf[QX_MAX];
        float sx = quantize_act_int8(qx_buf, x, cols);
        qwen_census_leaf(QWEN_LEAF_SDOT);
        int nt = g_n_threads;
        if (nt > 1 && rows >= 256) {
            int8_sdot_ctx c = { y, qx_buf, sx, W, scale, rows, cols };
            qwen_parallel((size_t)nt, int8_sdot_task, &c);
            goto qwen_matvec_int8_timed_done;
        }
        int8_matvec_sdot(y, qx_buf, sx, W, scale, cols, rows);
        goto qwen_matvec_int8_timed_done;
    }
#endif
    qwen_census_leaf(QWEN_LEAF_F32_FUSED);
    int nt = g_n_threads;
    if (nt > 1 && rows >= 256) {
        int8_mv_ctx c = { y, x, W, scale, rows, cols };
        qwen_parallel((size_t)nt, int8_mv_task, &c);
        goto qwen_matvec_int8_timed_done;
    }
    int8_matvec_fused(y, x, W, scale, cols, rows);

qwen_matvec_int8_timed_done:
    qwen_kernel_timing_note(QWEN_KT_INT8, 1, rows, cols, kt_t0);
}

void qwen_matvec_int8_qkv(float *q, float *k, float *v,
                           const int8_t *Wq, const float *sq,
                           const int8_t *Wk, const float *sk,
                           const int8_t *Wv, const float *sv,
                           const float *x, int in_dim, int q_dim, int kv_dim) {
    qwen_census_op(QWEN_PATH_MATVEC_INT8_QKV, q_dim + 2 * kv_dim, in_dim, 1);
    const int kt_on = qwen_kernel_timing_enabled();
    const double kt_t0 = kt_on ? qwen_mm_now_s() : 0.0;
    const int total = q_dim + 2 * kv_dim;
    if (qwen_kleidi_i8_enabled() &&
        qwen_kleidi_matmul_i8_qkv(q, k, v, Wq, Wk, Wv, x, in_dim, q_dim, kv_dim)) {
        MMSTAT(QWEN_MMK_KLEIDI_I8_GEMV, total, in_dim, 1);
        qwen_kernel_timing_note(QWEN_KT_INT8, 1, total, in_dim, kt_t0);
        return;
    }
    MMSTAT(QWEN_MMK_INT8_GEMV, q_dim + 2 * kv_dim, in_dim, 1);

#if defined(__ARM_FEATURE_DOTPROD)
    {
        enum { QX_MAX_QKV = 8192 };
        static atomic_int qkv_off = -1;
        int qo = atomic_load_explicit(&qkv_off, memory_order_relaxed);
        if (qo < 0) { const char *e = getenv("QWEN_NO_SDOT"); qo = (e && e[0] == '1'); atomic_store_explicit(&qkv_off, qo, memory_order_relaxed); }
        int nt = g_n_threads;
        if (!qo && in_dim <= QX_MAX_QKV && nt > 1 && (q_dim + 2 * kv_dim) >= 256) {
            int8_t qx_buf[QX_MAX_QKV];
            float sx = quantize_act_int8(qx_buf, x, in_dim);
            qwen_census_leaf(QWEN_LEAF_SDOT);
            int8_qkv_sdot_ctx c = { q, k, v, qx_buf, sx, Wq, Wk, Wv, sq, sk, sv,
                                    in_dim, q_dim, kv_dim };
            qwen_parallel((size_t)nt, int8_qkv_sdot_task, &c);
            return;
        }
    }
#endif
#if defined(__AVX512VNNI__)
    {
        enum { QX_MAX_QKV = 8192 };
        const int total = q_dim + 2 * kv_dim;
        if (!qwen_vnni_qkv_disabled() && in_dim <= QX_MAX_QKV) {
            qwen_census_leaf(QWEN_LEAF_VNNI);
            int8_t qx_buf[QX_MAX_QKV];
            const int vp_on = qwen_vnni_phase_timing_enabled();
            const int nt = g_n_threads;
            double vp_t0 = vp_on ? qwen_mm_now_s() : 0.0;
            const int32_t *row_sums_q = qwen_vnni_row_sums(Wq, q_dim, in_dim);
            const int32_t *row_sums_k = qwen_vnni_row_sums(Wk, kv_dim, in_dim);
            const int32_t *row_sums_v = qwen_vnni_row_sums(Wv, kv_dim, in_dim);
            if (vp_on) qwen_vnni_phase_note(QWEN_VP_ROWSUM, 0, vp_t0);
            const int uact = row_sums_q && row_sums_k && row_sums_v &&
                             qwen_vnni_uact_enabled();
            const uint8_t *ux = uact ? (const uint8_t *)qx_buf : NULL;
            vp_t0 = vp_on ? qwen_mm_now_s() : 0.0;
            const float sx = uact ? quantize_act_u8_x86((uint8_t *)qx_buf, x, in_dim)
                                  : quantize_act_int8_x86(qx_buf, x, in_dim);
            if (vp_on) qwen_vnni_phase_note(
                QWEN_VP_QUANT, (long long)in_dim * (long long)sizeof(float), vp_t0);
            int8_qkv_vnni_ctx c = { q, k, v, qx_buf, sx, Wq, Wk, Wv,
                                    sq, sk, sv, row_sums_q, row_sums_k, row_sums_v,
                                    in_dim, q_dim, kv_dim, ux };
            if (nt > 1 && total >= 256) {
                vp_t0 = vp_on ? qwen_mm_now_s() : 0.0;
                qwen_parallel((size_t)nt, int8_qkv_vnni_task, &c);
                if (vp_on) qwen_vnni_phase_note(QWEN_VP_PARALLEL, 0, vp_t0);
            } else {
                int8_qkv_vnni_task(0, 1, &c);
            }
            qwen_kernel_timing_note(QWEN_KT_INT8, 1, total, in_dim, kt_t0);
            return;
        }
    }
#endif
    qwen_census_leaf(QWEN_LEAF_DELEGATED);   /* the three rows below carry the work */
    qwen_matvec_int8(q, Wq, sq, x, q_dim, in_dim);
    qwen_matvec_int8(k, Wk, sk, x, kv_dim, in_dim);
    qwen_matvec_int8(v, Wv, sv, x, kv_dim, in_dim);
}

int qwen_argmax_matvec_int8(const float *x, const int8_t *W, const float *scale,
                            int in_dim, int out_dim) {
    qwen_census_op(QWEN_PATH_ARGMAX_MATVEC_INT8, out_dim, in_dim, 1);
    static __thread float *y = NULL;
    static __thread int y_cap = 0;
    if (out_dim > y_cap) {
        float *ny = (float *)realloc(y, (size_t)out_dim * sizeof(float));
        if (!ny) return 0;
        y = ny; y_cap = out_dim;
    }
    qwen_matvec_int8(y, W, scale, x, out_dim, in_dim);
    int best = 0;
    float best_val = y[0];
    for (int o = 1; o < out_dim; o++)
        if (y[o] > best_val) { best_val = y[o]; best = o; }
    return best;
}

int qwen_argmax_matvec_q4_0(const float *x, const q4_0_block_t *W, int in_dim, int out_dim) {
    qwen_census_op(QWEN_PATH_ARGMAX_MATVEC_Q4_0, out_dim, in_dim, 1);
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

void qwen_quantize_bf16_to_q4_0(const uint16_t *src_bf16, int rows, int cols,
                                 q4_0_block_t *dst) {
    static int naive = -1;
    if (naive < 0) { const char *e = getenv("QWEN_Q4_NAIVE"); naive = (e && *e) ? 1 : 0; }
    int blocks_per_row = cols / Q4_0_BLOCK_SIZE;
    for (int r = 0; r < rows; r++) {
        const uint16_t *row = src_bf16 + (size_t)r * cols;
        q4_0_block_t *dst_row = dst + (size_t)r * blocks_per_row;

        for (int b = 0; b < blocks_per_row; b++) {
            const uint16_t *blk = row + b * Q4_0_BLOCK_SIZE;

            float vals[Q4_0_BLOCK_SIZE];
            float amax = 0.0f, vmax = 0.0f;
            for (int i = 0; i < Q4_0_BLOCK_SIZE; i++) {
                uint32_t bits = (uint32_t)blk[i] << 16;
                memcpy(&vals[i], &bits, sizeof(float));
                float a = fabsf(vals[i]);
                if (a > amax) { amax = a; vmax = vals[i]; }
            }

            float s;
            int q[Q4_0_BLOCK_SIZE];
            if (naive) {
                s = amax / 7.0f;
                uint16_t s16 = qwen_f32_to_f16(s);
                s = qwen_f16_to_f32(s16);
                float inv_s = (s > 0) ? 1.0f / s : 0.0f;
                for (int i = 0; i < Q4_0_BLOCK_SIZE; i++) {
                    int v = (int)roundf(vals[i] * inv_s);
                    q[i] = v < -8 ? -8 : (v > 7 ? 7 : v);
                }
                dst_row[b].scale_f16 = s16;
            } else {
                float isc = (vmax != 0.0f) ? -8.0f / vmax : 0.0f;
                double num = 0.0, den = 0.0;
                for (int i = 0; i < Q4_0_BLOCK_SIZE; i++) {
                    int v = (int)roundf(vals[i] * isc);
                    v = v < -8 ? -8 : (v > 7 ? 7 : v);
                    q[i] = v;
                    double w = (double)vals[i] * vals[i];
                    num += w * vals[i] * v;
                    den += w * (double)v * v;
                }
                s = (den > 0.0) ? (float)(num / den) : 0.0f;
                dst_row[b].scale_f16 = qwen_f32_to_f16(s);
            }

            for (int i = 0; i < 16; i++)
                dst_row[b].qs[i] = (uint8_t)((q[2*i] + 8) | ((q[2*i+1] + 8) << 4));
        }
    }
}

static void q4_0_matvec_inner(float *y, const float *x, const q4_0_block_t *W,
                               int cols, int out_dim) {
    int blocks_per_row = cols / Q4_0_BLOCK_SIZE;
    for (int o = 0; o < out_dim; o++) {
        const q4_0_block_t *row = W + (size_t)o * blocks_per_row;
        float sum = 0.0f;
#ifdef __ARM_NEON
        for (int b = 0; b < blocks_per_row; b++) {
            float scale = qwen_f16_to_f32(row[b].scale_f16);
            const uint8_t *qs = row[b].qs;
            const float *xb = x + b * Q4_0_BLOCK_SIZE;

            uint8x16_t raw = vld1q_u8(qs);
            uint8x16_t lo_nibble = vandq_u8(raw, vdupq_n_u8(0x0F));
            uint8x16_t hi_nibble = vshrq_n_u8(raw, 4);

            int16x8_t s0 = vreinterpretq_s16_u16(vsubl_u8(vget_low_u8(lo_nibble), vdup_n_u8(8)));
            int16x8_t s1 = vreinterpretq_s16_u16(vsubl_u8(vget_low_u8(hi_nibble), vdup_n_u8(8)));
            int16x8_t s2 = vreinterpretq_s16_u16(vsubl_u8(vget_high_u8(lo_nibble), vdup_n_u8(8)));
            int16x8_t s3 = vreinterpretq_s16_u16(vsubl_u8(vget_high_u8(hi_nibble), vdup_n_u8(8)));

            int16x8x2_t z0 = vzipq_s16(s0, s1);
            int16x8x2_t z1 = vzipq_s16(s2, s3);

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
        __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps(),
               acc2 = _mm256_setzero_ps(), acc3 = _mm256_setzero_ps();
        for (int b = 0; b < blocks_per_row; b++) {
            float scale = qwen_f16_to_f32(row[b].scale_f16);
            const uint8_t *qs = row[b].qs;
            const float *xb = x + b * Q4_0_BLOCK_SIZE;
            __m128i raw = _mm_loadu_si128((const __m128i *)qs);
            __m128i lo = _mm_and_si128(raw, _mm_set1_epi8(0x0F));
            __m128i hi = _mm_and_si128(_mm_srli_epi16(raw, 4), _mm_set1_epi8(0x0F));
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
            float scale = qwen_f16_to_f32(row[b].scale_f16);
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
static void q4_0_matvec_sdot(float *y, const int8_t *qx, float sx,
                             const q4_0_block_t *W, int cols, int out_dim) {
    int nb = cols / Q4_0_BLOCK_SIZE;
    const uint8x16_t mask = vdupq_n_u8(0x0F);
    const int8x16_t bias = vdupq_n_s8(8);
    int o = 0;
    for (; o + 1 < out_dim; o += 2) {
        const q4_0_block_t *r0 = W + (size_t)o * nb;
        const q4_0_block_t *r1 = W + (size_t)(o + 1) * nb;
        float32x4_t fa0 = vdupq_n_f32(0.0f), fa1 = vdupq_n_f32(0.0f);
        for (int b = 0; b < nb; b++) {
            const int8_t *xb = qx + b * Q4_0_BLOCK_SIZE;
            int8x16_t x0 = vld1q_s8(xb);
            int8x16_t x1 = vld1q_s8(xb + 16);
            uint8x16_t raw0 = vld1q_u8(r0[b].qs);
            uint8x16x2_t z0 = vzipq_u8(vandq_u8(raw0, mask), vshrq_n_u8(raw0, 4));
            int8x16_t w0a = vsubq_s8(vreinterpretq_s8_u8(z0.val[0]), bias);
            int8x16_t w0b = vsubq_s8(vreinterpretq_s8_u8(z0.val[1]), bias);
            int32x4_t acc0 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), w0a, x0), w0b, x1);
            fa0 = vfmaq_n_f32(fa0, vcvtq_f32_s32(acc0), qwen_f16_to_f32(r0[b].scale_f16));
            uint8x16_t raw1 = vld1q_u8(r1[b].qs);
            uint8x16x2_t z1 = vzipq_u8(vandq_u8(raw1, mask), vshrq_n_u8(raw1, 4));
            int8x16_t w1a = vsubq_s8(vreinterpretq_s8_u8(z1.val[0]), bias);
            int8x16_t w1b = vsubq_s8(vreinterpretq_s8_u8(z1.val[1]), bias);
            int32x4_t acc1 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), w1a, x0), w1b, x1);
            fa1 = vfmaq_n_f32(fa1, vcvtq_f32_s32(acc1), qwen_f16_to_f32(r1[b].scale_f16));
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
            fa0 = vfmaq_n_f32(fa0, vcvtq_f32_s32(acc0), qwen_f16_to_f32(r0[b].scale_f16));
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
#endif

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
static int q4_sdot_disabled(void) {
    static atomic_int off = -1;
    int v = atomic_load_explicit(&off, memory_order_relaxed);
    if (v < 0) { const char *e = getenv("QWEN_NO_SDOT"); v = (e && e[0] == '1'); atomic_store_explicit(&off, v, memory_order_relaxed); }
    return v;
}
#endif

#if defined(__AVX512VNNI__)
static void q4_0_matvec_vnni(float *y, const int8_t *qx, float sx,
                             const q4_0_block_t *W, int cols, int out_dim) {
    int nb = cols / Q4_0_BLOCK_SIZE;
    const __m128i lomask = _mm_set1_epi8(0x0F);
    const __m512i ones   = _mm512_set1_epi8(1);
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
            __m512i wv = _mm512_zextsi256_si512(_mm256_set_m128i(_mm_unpackhi_epi8(lo, hi),
                                                                 _mm_unpacklo_epi8(lo, hi)));
            __m512i xv = _mm512_zextsi256_si512(_mm256_loadu_si256((const __m256i *)(qx + (size_t)b * Q4_0_BLOCK_SIZE)));
            int dot = _mm512_reduce_add_epi32(_mm512_dpbusd_epi32(_mm512_setzero_si512(), wv, xv)) + corr[b];
            sum += qwen_f16_to_f32(row[b].scale_f16) * (float)dot;
        }
        y[o] = sum * sx;
    }
}
static inline int q4_hsum256(__m256i v) {
    __m128i lo = _mm256_castsi256_si128(v);
    __m128i hi = _mm256_extracti128_si256(v, 1);
    __m128i s  = _mm_add_epi32(lo, hi);
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(1, 0, 3, 2)));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(2, 3, 0, 1)));
    return _mm_cvtsi128_si32(s);
}
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
    int corr[Q4_QX_MAX / Q4_0_BLOCK_SIZE];
    for (int b = 0; b < nb; b++) {
        __m512i xv = _mm512_zextsi256_si512(
            _mm256_loadu_si256((const __m256i *)(qx + (size_t)b * Q4_0_BLOCK_SIZE)));
        corr[b] = -8 * _mm512_reduce_add_epi32(_mm512_dpbusd_epi32(_mm512_setzero_si512(), ones, xv));
    }

    int o = 0;
    for (; o + 3 < out_dim; o += 4) {
        const q4_0_block_t *r0 = W + (size_t)o * nb, *r1 = r0 + nb, *r2 = r1 + nb, *r3 = r2 + nb;
        float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
        int b = 0;
        for (; b + 1 < nb; b += 2) {
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
            s0 += qwen_f16_to_f32(r0[b].scale_f16) * (q4_hsum256(_mm512_castsi512_si256(d0)) + corr[b])
                + qwen_f16_to_f32(r0[b + 1].scale_f16) * (q4_hsum256(_mm512_extracti64x4_epi64(d0, 1)) + corr[b + 1]);
            s1 += qwen_f16_to_f32(r1[b].scale_f16) * (q4_hsum256(_mm512_castsi512_si256(d1)) + corr[b])
                + qwen_f16_to_f32(r1[b + 1].scale_f16) * (q4_hsum256(_mm512_extracti64x4_epi64(d1, 1)) + corr[b + 1]);
            s2 += qwen_f16_to_f32(r2[b].scale_f16) * (q4_hsum256(_mm512_castsi512_si256(d2)) + corr[b])
                + qwen_f16_to_f32(r2[b + 1].scale_f16) * (q4_hsum256(_mm512_extracti64x4_epi64(d2, 1)) + corr[b + 1]);
            s3 += qwen_f16_to_f32(r3[b].scale_f16) * (q4_hsum256(_mm512_castsi512_si256(d3)) + corr[b])
                + qwen_f16_to_f32(r3[b + 1].scale_f16) * (q4_hsum256(_mm512_extracti64x4_epi64(d3, 1)) + corr[b + 1]);
        }
        for (; b < nb; b++) {
            __m512i xv = _mm512_zextsi256_si512(
                _mm256_loadu_si256((const __m256i *)(qx + (size_t)b * Q4_0_BLOCK_SIZE)));
            __m512i xw0 = _mm512_zextsi256_si512(q4_unpack_block_u8(r0[b].qs));
            __m512i xw1 = _mm512_zextsi256_si512(q4_unpack_block_u8(r1[b].qs));
            __m512i xw2 = _mm512_zextsi256_si512(q4_unpack_block_u8(r2[b].qs));
            __m512i xw3 = _mm512_zextsi256_si512(q4_unpack_block_u8(r3[b].qs));
            s0 += qwen_f16_to_f32(r0[b].scale_f16) * (_mm512_reduce_add_epi32(_mm512_dpbusd_epi32(_mm512_setzero_si512(), xw0, xv)) + corr[b]);
            s1 += qwen_f16_to_f32(r1[b].scale_f16) * (_mm512_reduce_add_epi32(_mm512_dpbusd_epi32(_mm512_setzero_si512(), xw1, xv)) + corr[b]);
            s2 += qwen_f16_to_f32(r2[b].scale_f16) * (_mm512_reduce_add_epi32(_mm512_dpbusd_epi32(_mm512_setzero_si512(), xw2, xv)) + corr[b]);
            s3 += qwen_f16_to_f32(r3[b].scale_f16) * (_mm512_reduce_add_epi32(_mm512_dpbusd_epi32(_mm512_setzero_si512(), xw3, xv)) + corr[b]);
        }
        y[o] = s0 * sx; y[o + 1] = s1 * sx; y[o + 2] = s2 * sx; y[o + 3] = s3 * sx;
    }
    if (o < out_dim)
        q4_0_matvec_vnni(y + o, qx, sx, W + (size_t)o * nb, cols, out_dim - o);
}

static int q4_vnni_v3_on(void) {
    static atomic_int v = -1;
    int r = atomic_load_explicit(&v, memory_order_relaxed);
    if (r < 0) { const char *e = getenv("QWEN_Q4_VNNI_V3"); r = !(e && e[0] == '0');
                 atomic_store_explicit(&v, r, memory_order_relaxed); }
    return r;
}

static void q4_0_matvec_vnni_v4(float *y, const int8_t *qx, float sx,
                                const q4_0_block_t *W, int cols, int out_dim) {
    int nb = cols / Q4_0_BLOCK_SIZE;
    const __m512i ones = _mm512_set1_epi8(1);
    int corr[Q4_QX_MAX / Q4_0_BLOCK_SIZE];
    for (int b = 0; b < nb; b++) {
        __m512i xv = _mm512_zextsi256_si512(
            _mm256_loadu_si256((const __m256i *)(qx + (size_t)b * Q4_0_BLOCK_SIZE)));
        corr[b] = -8 * _mm512_reduce_add_epi32(_mm512_dpbusd_epi32(_mm512_setzero_si512(), ones, xv));
    }

    int o = 0;
    for (; o + 3 < out_dim; o += 4) {
        const q4_0_block_t *r0 = W + (size_t)o * nb, *r1 = r0 + nb, *r2 = r1 + nb, *r3 = r2 + nb;
        __m512 f0 = _mm512_setzero_ps(), f1 = _mm512_setzero_ps(),
               f2 = _mm512_setzero_ps(), f3 = _mm512_setzero_ps();
        float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f;
        int b = 0;
        for (; b + 1 < nb; b += 2) {
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
            float s0a = qwen_f16_to_f32(r0[b].scale_f16), s0b = qwen_f16_to_f32(r0[b + 1].scale_f16);
            float s1a = qwen_f16_to_f32(r1[b].scale_f16), s1b = qwen_f16_to_f32(r1[b + 1].scale_f16);
            float s2a = qwen_f16_to_f32(r2[b].scale_f16), s2b = qwen_f16_to_f32(r2[b + 1].scale_f16);
            float s3a = qwen_f16_to_f32(r3[b].scale_f16), s3b = qwen_f16_to_f32(r3[b + 1].scale_f16);
            __m512 sv0 = _mm512_insertf32x8(_mm512_castps256_ps512(_mm256_set1_ps(s0a)), _mm256_set1_ps(s0b), 1);
            __m512 sv1 = _mm512_insertf32x8(_mm512_castps256_ps512(_mm256_set1_ps(s1a)), _mm256_set1_ps(s1b), 1);
            __m512 sv2 = _mm512_insertf32x8(_mm512_castps256_ps512(_mm256_set1_ps(s2a)), _mm256_set1_ps(s2b), 1);
            __m512 sv3 = _mm512_insertf32x8(_mm512_castps256_ps512(_mm256_set1_ps(s3a)), _mm256_set1_ps(s3b), 1);
            f0 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(d0), sv0, f0);
            f1 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(d1), sv1, f1);
            f2 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(d2), sv2, f2);
            f3 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(d3), sv3, f3);
            c0 += s0a * corr[b] + s0b * corr[b + 1];
            c1 += s1a * corr[b] + s1b * corr[b + 1];
            c2 += s2a * corr[b] + s2b * corr[b + 1];
            c3 += s3a * corr[b] + s3b * corr[b + 1];
        }
        for (; b < nb; b++) {
            __m512i xv = _mm512_zextsi256_si512(
                _mm256_loadu_si256((const __m256i *)(qx + (size_t)b * Q4_0_BLOCK_SIZE)));
            __m512i xw0 = _mm512_zextsi256_si512(q4_unpack_block_u8(r0[b].qs));
            __m512i xw1 = _mm512_zextsi256_si512(q4_unpack_block_u8(r1[b].qs));
            __m512i xw2 = _mm512_zextsi256_si512(q4_unpack_block_u8(r2[b].qs));
            __m512i xw3 = _mm512_zextsi256_si512(q4_unpack_block_u8(r3[b].qs));
            c0 += qwen_f16_to_f32(r0[b].scale_f16) * (_mm512_reduce_add_epi32(_mm512_dpbusd_epi32(_mm512_setzero_si512(), xw0, xv)) + corr[b]);
            c1 += qwen_f16_to_f32(r1[b].scale_f16) * (_mm512_reduce_add_epi32(_mm512_dpbusd_epi32(_mm512_setzero_si512(), xw1, xv)) + corr[b]);
            c2 += qwen_f16_to_f32(r2[b].scale_f16) * (_mm512_reduce_add_epi32(_mm512_dpbusd_epi32(_mm512_setzero_si512(), xw2, xv)) + corr[b]);
            c3 += qwen_f16_to_f32(r3[b].scale_f16) * (_mm512_reduce_add_epi32(_mm512_dpbusd_epi32(_mm512_setzero_si512(), xw3, xv)) + corr[b]);
        }
        y[o]     = (_mm512_reduce_add_ps(f0) + c0) * sx;
        y[o + 1] = (_mm512_reduce_add_ps(f1) + c1) * sx;
        y[o + 2] = (_mm512_reduce_add_ps(f2) + c2) * sx;
        y[o + 3] = (_mm512_reduce_add_ps(f3) + c3) * sx;
    }
    if (o < out_dim)
        q4_0_matvec_vnni_v3(y + o, qx, sx, W + (size_t)o * nb, cols, out_dim - o);
}

static int q4_vnni_v4_on(void) {
    static atomic_int v = -1;
    int r = atomic_load_explicit(&v, memory_order_relaxed);
    if (r < 0) { const char *e = getenv("QWEN_Q4_VNNI_V4"); r = (e && e[0] == '1');
                 atomic_store_explicit(&v, r, memory_order_relaxed); }
    return r;
}
int qwen_q4_vnni_variant(void) { return q4_vnni_v4_on() ? 4 : (q4_vnni_v3_on() ? 3 : 2); }

static inline void q4_vnni_rows(float *y, const int8_t *qx, float sx,
                                const q4_0_block_t *W, int cols, int rows) {
    if (q4_vnni_v4_on())      q4_0_matvec_vnni_v4(y, qx, sx, W, cols, rows);
    else if (q4_vnni_v3_on()) q4_0_matvec_vnni_v3(y, qx, sx, W, cols, rows);
    else                      q4_0_matvec_vnni(y, qx, sx, W, cols, rows);
}

typedef struct { float *y; const int8_t *qx; float sx; const q4_0_block_t *W; int rows, cols; } q4_0_vnni_ctx;
static void q4_0_vnni_task(size_t tid, size_t nt, void *vc) {
    q4_0_vnni_ctx *c = (q4_0_vnni_ctx *)vc;
    int r0 = (int)(tid * (size_t)c->rows / nt);
    int r1 = (int)((tid + 1) * (size_t)c->rows / nt);
    const q4_0_block_t *W = c->W + (size_t)r0 * (c->cols / Q4_0_BLOCK_SIZE);
    q4_vnni_rows(c->y + r0, c->qx, c->sx, W, c->cols, r1 - r0);
}
#endif

void qwen_matvec_q4_0(float *y, const q4_0_block_t *W, const float *x,
                       int rows, int cols) {
    qwen_census_op(QWEN_PATH_MATVEC_Q4_0, rows, cols, 1);
    if (qwen_mm_use(QWEN_MMK_KLEIDI_Q4, 1, rows, cols) &&
        qwen_kleidi_matmul_q4(y, (const void *)W, x, rows, cols, 1)) {
        MMSTAT(QWEN_MMK_KLEIDI_Q4, rows, cols, 1);
        return;
    }
    MMSTAT(QWEN_MMK_Q4_GEMV, rows, cols, 1);
#if defined(__AVX512VNNI__)
    if (!q4_sdot_disabled() && cols <= Q4_QX_MAX && cols % Q4_0_BLOCK_SIZE == 0) {
        int8_t qx_buf[Q4_QX_MAX];
        float sx = quantize_act_int8_x86(qx_buf, x, cols);
        qwen_census_leaf(QWEN_LEAF_VNNI);
        int nt = g_n_threads;
        if (nt > 1 && rows >= 256) {
            q4_0_vnni_ctx c = { y, qx_buf, sx, W, rows, cols };
            qwen_parallel((size_t)nt, q4_0_vnni_task, &c);
            return;
        }
        q4_vnni_rows(y, qx_buf, sx, W, cols, rows);
        return;
    }
#endif
#if defined(__ARM_FEATURE_DOTPROD)
    if (!q4_sdot_disabled() && cols <= Q4_QX_MAX && cols % Q4_0_BLOCK_SIZE == 0) {
        int8_t qx_buf[Q4_QX_MAX];
        float sx = quantize_act_int8(qx_buf, x, cols);
        int nt = g_n_threads;
        qwen_census_leaf(QWEN_LEAF_SDOT);
        if (nt > 1 && rows >= 256) {
            q4_0_sdot_ctx c = { y, qx_buf, sx, W, rows, cols };
            qwen_parallel((size_t)nt, q4_0_sdot_task, &c);
            return;
        }
        q4_0_matvec_sdot(y, qx_buf, sx, W, cols, rows);
        return;
    }
#endif
    qwen_census_leaf(QWEN_LEAF_F32_FUSED);
    int nt = g_n_threads;
    if (nt > 1 && rows >= 256) {
        q4_0_mv_ctx c = { y, W, x, rows, cols, cols / Q4_0_BLOCK_SIZE };
        qwen_parallel((size_t)nt, q4_0_mv_task, &c);
        return;
    }
    q4_0_matvec_inner(y, x, W, cols, rows);
}

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
#if defined(__AVX512VNNI__)
typedef struct {
    float *q, *k, *v;
    const q4_0_block_t *Wq, *Wk, *Wv;
    const int8_t *qx; float sx;
    int in_dim, q_dim, kv_dim;
} q4_0_qkv_vnni_ctx;
static void q4_0_qkv_vnni_task(size_t tid, size_t nt, void *vc) {
    q4_0_qkv_vnni_ctx *c = (q4_0_qkv_vnni_ctx *)vc;
    int total = c->q_dim + 2 * c->kv_dim;
    int nb = c->in_dim / Q4_0_BLOCK_SIZE;
    int r0 = (int)(tid * (size_t)total / nt);
    int r1 = (int)((tid + 1) * (size_t)total / nt);
    for (int r = r0; r < r1; ) {
        if (r < c->q_dim) {
            int end = r1 < c->q_dim ? r1 : c->q_dim;
            q4_vnni_rows(c->q + r, c->qx, c->sx, c->Wq + (size_t)r * nb, c->in_dim, end - r);
            r = end;
        } else if (r < c->q_dim + c->kv_dim) {
            int local = r - c->q_dim;
            int end = r1 < c->q_dim + c->kv_dim ? r1 : c->q_dim + c->kv_dim;
            q4_vnni_rows(c->k + local, c->qx, c->sx, c->Wk + (size_t)local * nb, c->in_dim, (end - c->q_dim) - local);
            r = end;
        } else {
            int local = r - c->q_dim - c->kv_dim;
            int local_end = r1 - c->q_dim - c->kv_dim;
            q4_vnni_rows(c->v + local, c->qx, c->sx, c->Wv + (size_t)local * nb, c->in_dim, local_end - local);
            r = r1;
        }
    }
}
#endif
void qwen_matvec_q4_0_qkv(float *q, float *k, float *v,
                            const q4_0_block_t *Wq, const q4_0_block_t *Wk,
                            const q4_0_block_t *Wv,
                            const float *x, int in_dim, int q_dim, int kv_dim) {
    qwen_census_op(QWEN_PATH_MATVEC_Q4_0_QKV, q_dim + 2 * kv_dim, in_dim, 1);
    if (qwen_mm_use(QWEN_MMK_KLEIDI_Q4, 1, q_dim, in_dim) &&
        qwen_kleidi_matmul_q4(q, (const void *)Wq, x, q_dim,  in_dim, 1) &&
        qwen_kleidi_matmul_q4(k, (const void *)Wk, x, kv_dim, in_dim, 1) &&
        qwen_kleidi_matmul_q4(v, (const void *)Wv, x, kv_dim, in_dim, 1)) {
        MMSTAT(QWEN_MMK_KLEIDI_Q4, q_dim + 2 * kv_dim, in_dim, 1);
        return;
    }
    MMSTAT(QWEN_MMK_Q4_GEMV, q_dim + 2 * kv_dim, in_dim, 1);

#if defined(__AVX512VNNI__)
    static atomic_int qkv_off = -1;
    int qkv_o = atomic_load_explicit(&qkv_off, memory_order_relaxed);
    if (qkv_o < 0) { const char *e = getenv("QWEN_NO_VNNI_QKV"); qkv_o = (e && e[0] == '1'); atomic_store_explicit(&qkv_off, qkv_o, memory_order_relaxed); }
    if (!qkv_o && !q4_sdot_disabled() && in_dim <= Q4_QX_MAX && in_dim % Q4_0_BLOCK_SIZE == 0) {
        qwen_census_leaf(QWEN_LEAF_VNNI);
        int8_t qx_buf[Q4_QX_MAX];
        float sx = quantize_act_int8_x86(qx_buf, x, in_dim);
        int nt = g_n_threads;
        if (nt > 1) {
            q4_0_qkv_vnni_ctx c = { q, k, v, Wq, Wk, Wv, qx_buf, sx, in_dim, q_dim, kv_dim };
            qwen_parallel((size_t)nt, q4_0_qkv_vnni_task, &c);
            return;
        }
        q4_vnni_rows(q, qx_buf, sx, Wq, in_dim, q_dim);
        q4_vnni_rows(k, qx_buf, sx, Wk, in_dim, kv_dim);
        q4_vnni_rows(v, qx_buf, sx, Wv, in_dim, kv_dim);
        return;
    }
#endif
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
                int code = (int)lrintf(vals[i] * inv + 1.5f);
                code = code < 0 ? 0 : (code > 3 ? 3 : code);
                drow[b].qs[i >> 2] |= (uint8_t)(code << ((i & 3) * 2));
            }
        }
    }
}

void qwen_matvec_q2_0(float *y, const q2_0_block_t *W, const float *x,
                      int rows, int cols) {
    qwen_census_op(QWEN_PATH_MATVEC_Q2_0, rows, cols, 1);
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

void qwen_quantize_bf16_to_q6_0(const uint16_t *src_bf16, int rows, int cols,
                                 q6_0_block_t *dst) {
    int bpr = cols / Q6_0_BLOCK_SIZE;
    for (int r = 0; r < rows; r++) {
        const uint16_t *row = src_bf16 + (size_t)r * cols;
        q6_0_block_t *dst_row = dst + (size_t)r * bpr;
        for (int b = 0; b < bpr; b++) {
            const uint16_t *blk = row + b * Q6_0_BLOCK_SIZE;
            float vals[Q6_0_BLOCK_SIZE];
            float amax = 0.0f;
            for (int i = 0; i < Q6_0_BLOCK_SIZE; i++) {
                uint32_t bits = (uint32_t)blk[i] << 16;
                memcpy(&vals[i], &bits, sizeof(float));
                float a = fabsf(vals[i]);
                if (a > amax) amax = a;
            }
            uint16_t s16 = qwen_f32_to_f16(amax / 31.0f);
            float s = qwen_f16_to_f32(s16);
            dst_row[b].scale_f16 = s16;
            memset(dst_row[b].ql, 0, sizeof(dst_row[b].ql));
            memset(dst_row[b].qh, 0, sizeof(dst_row[b].qh));
            for (int i = 0; i < Q6_0_BLOCK_SIZE; i++) {
                int q = 0;
                if (s > 0.0f) {
                    float a = fabsf(vals[i] / s);
                    float f = floorf(a + 0.5f);
                    if (f > 31.0f) f = 31.0f;
                    q = (vals[i] / s) < 0.0f ? -(int)f : (int)f;
                }
                unsigned u = (unsigned)(q + 32);
                dst_row[b].ql[i >> 1] |= (uint8_t)((u & 0xF) << ((i & 1) * 4));
                int g = i >> 4, rem = i & 15, j = rem & 3, k = rem >> 2;
                dst_row[b].qh[4 * g + j] |= (uint8_t)((u >> 4) << (2 * k));
            }
        }
    }
}

static inline void q6_unpack_codes(const q6_0_block_t *blk, uint8_t *u) {
    for (int i = 0; i < Q6_0_BLOCK_SIZE; i++) {
        int lo = (blk->ql[i >> 1] >> ((i & 1) * 4)) & 0xF;
        int g = i >> 4, rem = i & 15, j = rem & 3, k = rem >> 2;
        int hi = (blk->qh[4 * g + j] >> (2 * k)) & 0x3;
        u[i] = (uint8_t)(lo | (hi << 4));
    }
}

void qwen_dequant_row_q6_0(float *dst, const q6_0_block_t *row, int cols) {
    int bpr = cols / Q6_0_BLOCK_SIZE;
    for (int b = 0; b < bpr; b++) {
        uint8_t u[Q6_0_BLOCK_SIZE];
        q6_unpack_codes(row + b, u);
        float s = qwen_f16_to_f32(row[b].scale_f16);
        for (int i = 0; i < Q6_0_BLOCK_SIZE; i++)
            dst[b * Q6_0_BLOCK_SIZE + i] = s * (float)((int)u[i] - 32);
    }
}

enum { Q6_QX_MAX = 8192 };
static float q6_quant_act(int8_t *qx, int32_t *sumx, const float *x, int n) {
    float amax = 0.0f;
    for (int i = 0; i < n; i++) { float a = fabsf(x[i]); if (a > amax) amax = a; }
    if (amax == 0.0f) {
        memset(qx, 0, (size_t)n);
        memset(sumx, 0, (size_t)(n / Q6_0_BLOCK_SIZE) * sizeof(int32_t));
        return 0.0f;
    }
    float inv = 127.0f / amax;
    for (int i = 0; i < n; i++) {
        int v = (int)lrintf(x[i] * inv);
        qx[i] = (int8_t)(v > 127 ? 127 : (v < -128 ? -128 : v));
    }
    for (int b = 0; b < n / Q6_0_BLOCK_SIZE; b++) {
        int32_t s = 0;
        for (int i = 0; i < Q6_0_BLOCK_SIZE; i++) s += qx[b * Q6_0_BLOCK_SIZE + i];
        sumx[b] = s;
    }
    return amax / 127.0f;
}

static void q6_0_matvec_scalar(float *y, const int8_t *qx, const int32_t *sumx, float sx,
                               const q6_0_block_t *W, int cols, int out_dim) {
    int nb = cols / Q6_0_BLOCK_SIZE;
    for (int o = 0; o < out_dim; o++) {
        const q6_0_block_t *row = W + (size_t)o * nb;
        float sum = 0.0f;
        for (int b = 0; b < nb; b++) {
            uint8_t u[Q6_0_BLOCK_SIZE];
            q6_unpack_codes(row + b, u);
            const int8_t *xb = qx + b * Q6_0_BLOCK_SIZE;
            int32_t dot = 0;
            for (int i = 0; i < Q6_0_BLOCK_SIZE; i++) dot += (int32_t)u[i] * xb[i];
            sum += qwen_f16_to_f32(row[b].scale_f16) * (float)(dot - 32 * sumx[b]);
        }
        y[o] = sum * sx;
    }
}

#if defined(__ARM_FEATURE_DOTPROD)
static const int8_t q6_shift_tab[16] = { 0,0,0,0, -2,-2,-2,-2, -4,-4,-4,-4, -6,-6,-6,-6 };

static inline void q6_unpack_neon(const q6_0_block_t *blk, int8x16_t *wa, int8x16_t *wb) {
    const uint8x16_t mask4 = vdupq_n_u8(0x0F);
    const uint8x16_t mask2 = vdupq_n_u8(0x03);
    const int8x16_t  sh    = vld1q_s8(q6_shift_tab);
    uint8x16_t raw = vld1q_u8(blk->ql);
    uint8x16x2_t z = vzipq_u8(vandq_u8(raw, mask4), vshrq_n_u8(raw, 4));
    uint8x16_t ha = vandq_u8(vshlq_u8(vreinterpretq_u8_u32(
                        vld1q_dup_u32((const uint32_t *)(const void *)blk->qh)), sh), mask2);
    uint8x16_t hb = vandq_u8(vshlq_u8(vreinterpretq_u8_u32(
                        vld1q_dup_u32((const uint32_t *)(const void *)(blk->qh + 4))), sh), mask2);
    *wa = vreinterpretq_s8_u8(vorrq_u8(z.val[0], vshlq_n_u8(ha, 4)));
    *wb = vreinterpretq_s8_u8(vorrq_u8(z.val[1], vshlq_n_u8(hb, 4)));
}

static void q6_0_matvec_sdot(float *y, const int8_t *qx, const int32_t *sumx, float sx,
                             const q6_0_block_t *W, int cols, int out_dim) {
    int nb = cols / Q6_0_BLOCK_SIZE;
    int o = 0;
    for (; o + 1 < out_dim; o += 2) {
        const q6_0_block_t *r0 = W + (size_t)o * nb;
        const q6_0_block_t *r1 = W + (size_t)(o + 1) * nb;
        float32x4_t fa0 = vdupq_n_f32(0.0f), fa1 = vdupq_n_f32(0.0f);
        for (int b = 0; b < nb; b++) {
            const int8_t *xb = qx + b * Q6_0_BLOCK_SIZE;
            int8x16_t x0 = vld1q_s8(xb), x1 = vld1q_s8(xb + 16);
            int32x4_t corr = vsetq_lane_s32(-32 * sumx[b], vdupq_n_s32(0), 0);
            int8x16_t w0a, w0b, w1a, w1b;
            q6_unpack_neon(&r0[b], &w0a, &w0b);
            q6_unpack_neon(&r1[b], &w1a, &w1b);
            int32x4_t a0 = vdotq_s32(vdotq_s32(corr, w0a, x0), w0b, x1);
            int32x4_t a1 = vdotq_s32(vdotq_s32(corr, w1a, x0), w1b, x1);
            fa0 = vfmaq_n_f32(fa0, vcvtq_f32_s32(a0), qwen_f16_to_f32(r0[b].scale_f16));
            fa1 = vfmaq_n_f32(fa1, vcvtq_f32_s32(a1), qwen_f16_to_f32(r1[b].scale_f16));
        }
        y[o]     = vaddvq_f32(fa0) * sx;
        y[o + 1] = vaddvq_f32(fa1) * sx;
    }
    if (o < out_dim) {
        const q6_0_block_t *r0 = W + (size_t)o * nb;
        float32x4_t fa0 = vdupq_n_f32(0.0f);
        for (int b = 0; b < nb; b++) {
            const int8_t *xb = qx + b * Q6_0_BLOCK_SIZE;
            int8x16_t x0 = vld1q_s8(xb), x1 = vld1q_s8(xb + 16);
            int32x4_t corr = vsetq_lane_s32(-32 * sumx[b], vdupq_n_s32(0), 0);
            int8x16_t w0a, w0b;
            q6_unpack_neon(&r0[b], &w0a, &w0b);
            int32x4_t a0 = vdotq_s32(vdotq_s32(corr, w0a, x0), w0b, x1);
            fa0 = vfmaq_n_f32(fa0, vcvtq_f32_s32(a0), qwen_f16_to_f32(r0[b].scale_f16));
        }
        y[o] = vaddvq_f32(fa0) * sx;
    }
}
#endif

#if defined(__AVX2__)
static inline __m128i q6_high_128(uint32_t h) {
    const __m128i m2 = _mm_set1_epi8(0x03);
    __m128i d  = _mm_set1_epi32((int)h);
    __m128i s0 = _mm_and_si128(d, m2);
    __m128i s1 = _mm_and_si128(_mm_srli_epi32(d, 2), m2);
    __m128i s2 = _mm_and_si128(_mm_srli_epi32(d, 4), m2);
    __m128i s3 = _mm_and_si128(_mm_srli_epi32(d, 6), m2);
    __m128i a  = _mm_blend_epi32(s0, s1, 0x2);
    __m128i b  = _mm_blend_epi32(s2, s3, 0x8);
    return _mm_blend_epi32(a, b, 0xC);
}
static inline __m256i q6_unpack_block_u8(const q6_0_block_t *blk) {
    const __m128i m4 = _mm_set1_epi8(0x0F);
    __m128i raw = _mm_loadu_si128((const __m128i *)blk->ql);
    __m128i lo  = _mm_and_si128(raw, m4);
    __m128i hi  = _mm_and_si128(_mm_srli_epi16(raw, 4), m4);
    __m128i l_a = _mm_unpacklo_epi8(lo, hi);
    __m128i l_b = _mm_unpackhi_epi8(lo, hi);
    uint32_t h0, h1;
    memcpy(&h0, blk->qh, 4);
    memcpy(&h1, blk->qh + 4, 4);
    __m128i h_a = _mm_slli_epi16(q6_high_128(h0), 4);
    __m128i h_b = _mm_slli_epi16(q6_high_128(h1), 4);
    return _mm256_set_m128i(_mm_or_si128(l_b, h_b), _mm_or_si128(l_a, h_a));
}
static inline int q6_hsum256(__m256i v) {
    __m128i s = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(1, 0, 3, 2)));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(2, 3, 0, 1)));
    return _mm_cvtsi128_si32(s);
}
static void q6_0_matvec_avx2(float *y, const int8_t *qx, const int32_t *sumx, float sx,
                             const q6_0_block_t *W, int cols, int out_dim) {
    int nb = cols / Q6_0_BLOCK_SIZE;
    const __m256i ones16 = _mm256_set1_epi16(1);
    for (int o = 0; o < out_dim; o++) {
        const q6_0_block_t *row = W + (size_t)o * nb;
        float sum = 0.0f;
        for (int b = 0; b < nb; b++) {
            __m256i wv = q6_unpack_block_u8(&row[b]);
            __m256i xv = _mm256_loadu_si256((const __m256i *)(qx + (size_t)b * Q6_0_BLOCK_SIZE));
            __m256i p  = _mm256_madd_epi16(_mm256_maddubs_epi16(wv, xv), ones16);
            int dot = q6_hsum256(p) - 32 * sumx[b];
            sum += qwen_f16_to_f32(row[b].scale_f16) * (float)dot;
        }
        y[o] = sum * sx;
    }
}
#endif

#if defined(__AVX512VNNI__)
static void q6_0_matvec_vnni(float *y, const int8_t *qx, const int32_t *sumx, float sx,
                             const q6_0_block_t *W, int cols, int out_dim) {
    int nb = cols / Q6_0_BLOCK_SIZE;
    for (int o = 0; o < out_dim; o++) {
        const q6_0_block_t *row = W + (size_t)o * nb;
        float sum = 0.0f;
        int b = 0;
        for (; b + 1 < nb; b += 2) {
            __m512i wv = _mm512_inserti64x4(
                _mm512_castsi256_si512(q6_unpack_block_u8(&row[b])),
                q6_unpack_block_u8(&row[b + 1]), 1);
            __m512i xv = _mm512_loadu_si512((const void *)(qx + (size_t)b * Q6_0_BLOCK_SIZE));
            __m512i acc = _mm512_dpbusd_epi32(_mm512_setzero_si512(), wv, xv);
            int d0 = q6_hsum256(_mm512_castsi512_si256(acc)) - 32 * sumx[b];
            int d1 = q6_hsum256(_mm512_extracti64x4_epi64(acc, 1)) - 32 * sumx[b + 1];
            sum += qwen_f16_to_f32(row[b].scale_f16)     * (float)d0
                 + qwen_f16_to_f32(row[b + 1].scale_f16) * (float)d1;
        }
        for (; b < nb; b++) {
            __m512i wv = _mm512_zextsi256_si512(q6_unpack_block_u8(&row[b]));
            __m512i xv = _mm512_zextsi256_si512(
                _mm256_loadu_si256((const __m256i *)(qx + (size_t)b * Q6_0_BLOCK_SIZE)));
            __m512i acc = _mm512_dpbusd_epi32(_mm512_setzero_si512(), wv, xv);
            int dot = _mm512_reduce_add_epi32(acc) - 32 * sumx[b];
            sum += qwen_f16_to_f32(row[b].scale_f16) * (float)dot;
        }
        y[o] = sum * sx;
    }
}
#endif

static inline void q6_rows(float *y, const int8_t *qx, const int32_t *sumx, float sx,
                           const q6_0_block_t *W, int cols, int rows) {
#if defined(__ARM_FEATURE_DOTPROD)
    q6_0_matvec_sdot(y, qx, sumx, sx, W, cols, rows);
#elif defined(__AVX512VNNI__)
    q6_0_matvec_vnni(y, qx, sumx, sx, W, cols, rows);
#elif defined(__AVX2__)
    q6_0_matvec_avx2(y, qx, sumx, sx, W, cols, rows);
#else
    q6_0_matvec_scalar(y, qx, sumx, sx, W, cols, rows);
#endif
}

typedef struct {
    float *y; const int8_t *qx; const int32_t *sumx; float sx;
    const q6_0_block_t *W; int rows, cols;
} q6_0_ctx;
static void q6_0_task(size_t tid, size_t nt, void *vc) {
    q6_0_ctx *c = (q6_0_ctx *)vc;
    int r0 = (int)(tid * (size_t)c->rows / nt);
    int r1 = (int)((tid + 1) * (size_t)c->rows / nt);
    q6_rows(c->y + r0, c->qx, c->sumx, c->sx,
            c->W + (size_t)r0 * (c->cols / Q6_0_BLOCK_SIZE), c->cols, r1 - r0);
}

static int q6_scalar_forced(void) {
    static atomic_int on = -1;
    int v = atomic_load_explicit(&on, memory_order_relaxed);
    if (v < 0) {
        const char *e = getenv("QWEN_Q6_SCALAR");
        v = (e && e[0] == '1');
        atomic_store_explicit(&on, v, memory_order_relaxed);
    }
    return v;
}

void qwen_matvec_q6_0(float *y, const q6_0_block_t *W, const float *x,
                       int rows, int cols) {
    qwen_census_op(QWEN_PATH_MATVEC_Q6_0, rows, cols, 1);
    int8_t qx_buf[Q6_QX_MAX];
    int32_t sumx_buf[Q6_QX_MAX / Q6_0_BLOCK_SIZE];
    if (cols > Q6_QX_MAX || cols % Q6_0_BLOCK_SIZE != 0) {
        float *tmp = (float *)malloc((size_t)cols * sizeof(float));
        if (!tmp) { memset(y, 0, (size_t)rows * sizeof(float)); return; }
        int bpr = cols / Q6_0_BLOCK_SIZE;
        for (int o = 0; o < rows; o++) {
            qwen_dequant_row_q6_0(tmp, W + (size_t)o * bpr, cols - (cols % Q6_0_BLOCK_SIZE));
            double s = 0.0;
            for (int i = 0; i < cols - (cols % Q6_0_BLOCK_SIZE); i++) s += (double)tmp[i] * x[i];
            y[o] = (float)s;
        }
        free(tmp);
        return;
    }
    float sx = q6_quant_act(qx_buf, sumx_buf, x, cols);
    int nt = g_n_threads;
    if (!q6_scalar_forced() && nt > 1 && rows >= 256) {
        q6_0_ctx c = { y, qx_buf, sumx_buf, sx, W, rows, cols };
        qwen_parallel((size_t)nt, q6_0_task, &c);
        return;
    }
    if (q6_scalar_forced()) {
        q6_0_matvec_scalar(y, qx_buf, sumx_buf, sx, W, cols, rows);
        return;
    }
    q6_rows(y, qx_buf, sumx_buf, sx, W, cols, rows);
}

typedef struct {
    float *q, *k, *v;
    const q6_0_block_t *Wq, *Wk, *Wv;
    const int8_t *qx; const int32_t *sumx; float sx;
    int in_dim, q_dim, kv_dim;
} q6_0_qkv_ctx;
static void q6_0_qkv_task(size_t tid, size_t nt, void *vc) {
    q6_0_qkv_ctx *c = (q6_0_qkv_ctx *)vc;
    int total = c->q_dim + 2 * c->kv_dim;
    int nb = c->in_dim / Q6_0_BLOCK_SIZE;
    int r0 = (int)(tid * (size_t)total / nt);
    int r1 = (int)((tid + 1) * (size_t)total / nt);
    for (int r = r0; r < r1; ) {
        if (r < c->q_dim) {
            int end = r1 < c->q_dim ? r1 : c->q_dim;
            q6_rows(c->q + r, c->qx, c->sumx, c->sx,
                    c->Wq + (size_t)r * nb, c->in_dim, end - r);
            r = end;
        } else if (r < c->q_dim + c->kv_dim) {
            int local = r - c->q_dim;
            int end = r1 < c->q_dim + c->kv_dim ? r1 : c->q_dim + c->kv_dim;
            q6_rows(c->k + local, c->qx, c->sumx, c->sx,
                    c->Wk + (size_t)local * nb, c->in_dim, (end - c->q_dim) - local);
            r = end;
        } else {
            int local = r - c->q_dim - c->kv_dim;
            int local_end = r1 - c->q_dim - c->kv_dim;
            q6_rows(c->v + local, c->qx, c->sumx, c->sx,
                    c->Wv + (size_t)local * nb, c->in_dim, local_end - local);
            r = r1;
        }
    }
}

void qwen_matvec_q6_0_qkv(float *q, float *k, float *v,
                          const q6_0_block_t *Wq, const q6_0_block_t *Wk,
                          const q6_0_block_t *Wv,
                          const float *x, int in_dim, int q_dim, int kv_dim) {
    qwen_census_op(QWEN_PATH_MATVEC_Q6_0_QKV, q_dim + 2 * kv_dim, in_dim, 1);
    if (in_dim > Q6_QX_MAX || in_dim % Q6_0_BLOCK_SIZE != 0) {
        qwen_matvec_q6_0(q, Wq, x, q_dim, in_dim);
        qwen_matvec_q6_0(k, Wk, x, kv_dim, in_dim);
        qwen_matvec_q6_0(v, Wv, x, kv_dim, in_dim);
        return;
    }
    int8_t qx_buf[Q6_QX_MAX];
    int32_t sumx_buf[Q6_QX_MAX / Q6_0_BLOCK_SIZE];
    float sx = q6_quant_act(qx_buf, sumx_buf, x, in_dim);
    int nb = in_dim / Q6_0_BLOCK_SIZE;
    int nt = g_n_threads;
    int total = q_dim + 2 * kv_dim;
    if (!q6_scalar_forced() && nt > 1 && total >= 256) {
        q6_0_qkv_ctx c = { q, k, v, Wq, Wk, Wv, qx_buf, sumx_buf, sx, in_dim, q_dim, kv_dim };
        qwen_parallel((size_t)nt, q6_0_qkv_task, &c);
        return;
    }
    if (q6_scalar_forced()) {
        q6_0_matvec_scalar(q, qx_buf, sumx_buf, sx, Wq, in_dim, q_dim);
        q6_0_matvec_scalar(k, qx_buf, sumx_buf, sx, Wk, in_dim, kv_dim);
        q6_0_matvec_scalar(v, qx_buf, sumx_buf, sx, Wv, in_dim, kv_dim);
        return;
    }
    (void)nb;
    q6_rows(q, qx_buf, sumx_buf, sx, Wq, in_dim, q_dim);
    q6_rows(k, qx_buf, sumx_buf, sx, Wk, in_dim, kv_dim);
    q6_rows(v, qx_buf, sumx_buf, sx, Wv, in_dim, kv_dim);
}

typedef struct {
    float *out; const float *Q, *K, *V;
    int seq_q, seq_k, n_heads, n_kv_heads, head_dim, q_offset;
    float scale;
} qwen_attn_job_t;

static void qwen_attn_task(size_t tid, size_t nt, void *ctx) {
    const qwen_attn_job_t *j = (const qwen_attn_job_t *)ctx;
    int per = (j->n_heads + (int)nt - 1) / (int)nt;
    int h0 = (int)tid * per, h1 = h0 + per;
    if (h0 >= j->n_heads) return;
    if (h1 > j->n_heads) h1 = j->n_heads;
    qwen_causal_attention_heads(j->out, j->Q, j->K, j->V, j->seq_q, j->seq_k,
                                j->n_heads, j->n_kv_heads, j->head_dim, j->scale,
                                j->q_offset, h0, h1);
}

void qwen_causal_attention_prefill(float *out, const float *Q, const float *K, const float *V,
                                   int seq_q, int seq_k, int n_heads, int n_kv_heads,
                                   int head_dim, float scale, int q_offset) {
    int nt = qwen_get_threads();
    if (nt > n_heads) nt = n_heads;
    if (nt > 1 && seq_q > 1) {
        qwen_attn_job_t job = { out, Q, K, V, seq_q, seq_k, n_heads, n_kv_heads,
                                head_dim, q_offset, scale };
        qwen_parallel((size_t)nt, qwen_attn_task, &job);
        return;
    }
    qwen_causal_attention(out, Q, K, V, seq_q, seq_k, n_heads, n_kv_heads,
                          head_dim, scale, q_offset);
}

void qwen_causal_attention_heads(float *out, const float *Q, const float *K, const float *V,
                                 int seq_q, int seq_k, int n_heads, int n_kv_heads,
                                 int head_dim, float scale, int q_offset, int h_lo, int h_hi) {
    int heads_per_kv = n_heads / n_kv_heads;
    int q_hidden = n_heads * head_dim;
    int kv_hidden = n_kv_heads * head_dim;

    for (int h = h_lo; h < h_hi; h++) {
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

void qwen_causal_attention(float *out, const float *Q, const float *K, const float *V,
                           int seq_q, int seq_k, int n_heads, int n_kv_heads,
                           int head_dim, float scale, int q_offset) {
    qwen_causal_attention_heads(out, Q, K, V, seq_q, seq_k, n_heads, n_kv_heads,
                                head_dim, scale, q_offset, 0, n_heads);
}

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

                float score;
#ifdef __ARM_NEON
                {
                    float32x4_t a0 = vdupq_n_f32(0), a1 = vdupq_n_f32(0);
                    float32x4_t a2 = vdupq_n_f32(0), a3 = vdupq_n_f32(0);
                    int d = 0;
                    for (; d + 15 < head_dim; d += 16) {
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

void qwen_silu(float *x, int n) {
    for (int i = 0; i < n; i++)
        x[i] = x[i] / (1.0f + expf(-x[i]));
}

typedef struct { float *gate_up; float *tmp; int n; } qwen_swiglu_job_t;

static void qwen_swiglu_task(size_t tid, size_t nt, void *ctx) {
    qwen_swiglu_job_t *j = (qwen_swiglu_job_t *)ctx;
    int per = (j->n + (int)nt - 1) / (int)nt;
    per = (per + 3) & ~3;
    int lo = (int)tid * per, hi = lo + per;
    if (lo >= j->n) return;
    if (hi > j->n) hi = j->n;
    float *tmp = j->tmp; const float *gu = j->gate_up;

    for (int i = lo; i < hi; i++) tmp[i] = -gu[2 * i];
    for (int i = lo; i < hi; i++) tmp[i] = expf(tmp[i]);
    for (int i = lo; i < hi; i++) tmp[i] = gu[2 * i] / (1.0f + tmp[i]) * gu[2 * i + 1];
}

void qwen_swiglu_prefill(float *gate_up, float *tmp, int n) {
    int nt = qwen_get_threads();
    if (nt > 1 && n >= 2048) {
        qwen_swiglu_job_t job = { gate_up, tmp, n };
        qwen_parallel((size_t)nt, qwen_swiglu_task, &job);
        memcpy(gate_up, tmp, (size_t)n * sizeof(float));
        return;
    }
    qwen_swiglu_inplace(gate_up, tmp, n);
}

void qwen_swiglu_inplace(float *gate_up, float *tmp, int n) {
    for (int i = 0; i < n; i++)
        tmp[i] = -gate_up[2 * i];

#if defined(__APPLE__) && defined(USE_BLAS)
    vvexpf(tmp, tmp, &n);
#else
    for (int i = 0; i < n; i++)
        tmp[i] = expf(tmp[i]);
#endif

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

/* Capability, split from policy.  _available says the int8 decoder-convolution kernels are
 * compiled for this build; _usable adds the shapes those kernels actually cover, which the
 * decoder used to spell out at its call site (in_ch == out_ch && in_ch <= 768) even though
 * the constraint belongs to the kernel.  Neither answers whether the path is ENABLED --
 * that is a per-backend policy decision and lives with the decoder. */
int qwen_sd_int8_available(void) {
#if defined(__ARM_FEATURE_DOTPROD)
    return 1;
#elif defined(__AVX512VNNI__)
    return 1;
#else
    return 0;
#endif
}
/* Which family actually serves a B>1 matmat on THIS build, in the order the dispatcher
 * tries them.  The gate table already prints compiled/supported per family, but it cannot
 * say which one wins, and it has no row at all for the unconditional B-twin fallback -- so
 * on a build where every gate is off (AVX2 and AVX-512F for bf16, any non-VNNI x86 for the
 * integer paths) the map used to say nothing about what runs.  Evaluated at a representative
 * large shape so the thresholds in g_mm_gate[] are applied rather than duplicated here. */
/* One probe shape, so say which: a family row answers "who serves this dtype at a batched
 * shape", not "who serves every shape".  A gate with a higher min_b or a rows/cols floor can
 * still decline the real projection, and the per-gate table below carries those numbers. */
#define QWEN_MMK_PROBE_B    4
#define QWEN_MMK_PROBE_ROWS 4096
#define QWEN_MMK_PROBE_COLS 4096
static const char *mmk_first_available(const int *cand, int n, const char *none,
                                       char *buf, size_t bsz) {
    enum { RB = QWEN_MMK_PROBE_B, RR = QWEN_MMK_PROBE_ROWS, RC = QWEN_MMK_PROBE_COLS };
    for (int i = 0; i < n; i++) {
        int k = cand[i];
        if (qwen_mmk_compiled(k) && qwen_mmk_supported(k) && qwen_mm_use(k, RB, RR, RC)) {
            snprintf(buf, bsz, "%s (probe B=%d %dx%d)", g_mmk_info[k].name, RB, RR, RC);
            return buf;
        }
    }
    snprintf(buf, bsz, "%s (probe B=%d %dx%d)", none, RB, RR, RC);
    return buf;
}
const char *qwen_matmat_family_int8(void) {
    static const int cand[] = { QWEN_MMK_KLEIDI_I8, QWEN_MMK_INT8_AMX, QWEN_MMK_INT8_VNNI,
                                QWEN_MMK_INT8_AVX2, QWEN_MMK_INT8_SMMLA, QWEN_MMK_INT8_SDOT };
    static char buf[96];
    return mmk_first_available(cand, (int)(sizeof cand / sizeof cand[0]),
                               "int8 f32-accum twin (no int8 GEMM gate on this build)",
                               buf, sizeof buf);
}
const char *qwen_matmat_family_q4(void) {
    static const int cand[] = { QWEN_MMK_KLEIDI_Q4, QWEN_MMK_Q4_AMX, QWEN_MMK_Q4_VNNI,
                                QWEN_MMK_Q4_AVX2, QWEN_MMK_Q4_SMMLA, QWEN_MMK_Q4_BMATVEC };
    static char buf[96];
    return mmk_first_available(cand, (int)(sizeof cand / sizeof cand[0]),
                               "q4 generic twin (no q4 GEMM gate on this build)",
                               buf, sizeof buf);
}
/* Largest B the int8 matmat family still accepts, probed at the same shape as the family
 * rows.  It matters because --batch-size is NOT clamped to it: every batched int8 gate has
 * max_b = 16, and above that they all decline, so a server started with --batch-size 24
 * steps its slots through one GEMV each -- exactly the work batching was asked to avoid --
 * without a word.  Prefill already chunks itself to 16 (prefill_proj_matmat); the batched
 * decode path passes the live slot count straight through, and chunking it here would move
 * a remainder column onto the B=1 dequant twin, i.e. change its arithmetic.  So this is
 * reported, not silently corrected. */
/* Which in-region int8 runner a build/host can hold, named.  The persistent CP and Talker
 * regions are the largest backend difference we have -- present on VNNI and AMX, absent on
 * AVX2/AVX-512F and (until the gather shape lands) on Arm -- and until now the only place
 * that said so was a one-shot stderr line printed by the engine at the first batched step,
 * i.e. after traffic.  The model's own shapes still decide; this answers the prior question
 * of whether the runner exists at all. */
static const char *region_i8_backend_at(int B) {
    enum { RR = QWEN_MMK_PROBE_ROWS, RC = QWEN_MMK_PROBE_COLS };
#if defined(__AVX512VNNI__)
    if (!qwen_region_i8_usable(RR, RC, B)) return "none";
#if defined(__AMX_INT8__) && defined(__AMX_TILE__)
    if (qwen_mm_use(QWEN_MMK_INT8_AMX, B, RR, RC) && qwen_amx_int8_ready())
        return "AMX int8 tiles";
#endif
    return "VNNI row blocks";
#else
    (void)B; return "none";
#endif
}

const char *qwen_region_i8_backend(void) {
    /* Report BOTH batch widths: the AMX gate starts at B=4, so an AMX host answers "VNNI row
     * blocks" at B=2 and "AMX int8 tiles" at B=4, and a single-B row would hide half of that. */
    static char buf[112];
    const char *b2 = region_i8_backend_at(2), *b4 = region_i8_backend_at(4);
    if (!strcmp(b2, "none") && !strcmp(b4, "none")) {
#if defined(__AVX512VNNI__)
        return "none: the int8 gate declines at the probe shape for B=2 and B=4";
#else
        return "none: no in-region int8 runner in this build (needs VNNI or AMX; Arm wiring open)";
#endif
    }
    if (!strcmp(b2, b4)) snprintf(buf, sizeof buf, "%s (B=2 and B=4)", b2);
    else                 snprintf(buf, sizeof buf, "%s at B=2, %s at B=4", b2, b4);
    return buf;
}

/* Would the INT8 AMX gate ever select this projection on this host?  Prepacking a weight the
 * gate can never choose costs its whole size in RAM and buys nothing: `gate_rows` is the height
 * the gate actually judges, which for a fused QKV member is q+2kv, not that member's own rows. */
int qwen_amx_int8_pack_worth(int rows, int cols, int gate_rows, int threads) {
#if defined(__AMX_INT8__) && defined(__AMX_TILE__)
    if (!qwen_amx_int8_ready()) return 0;
    if (gate_rows <= 0) gate_rows = rows;
    /* threads is the SERVING worker's count, not the packing process's: the parent prepacks
     * before it forks, and a prefork worker runs with --prefork-threads, so asking
     * qwen_get_threads() here would judge the gate with the wrong denominator. */
    if (!qwen_amx_int8_rows_ok_nt(gate_rows, threads)) return 0;
    /* B is irrelevant to the shape half of the gate; probe at the kernel's own minimum. */
    return qwen_mm_use_(QWEN_MMK_INT8_AMX, qwen_mm_minb_value(QWEN_MMK_INT8_AMX,
                                                              &g_mm_gate[QWEN_MMK_INT8_AMX]),
                        rows, cols, 0);
#else
    (void)rows; (void)cols; (void)gate_rows; (void)threads; return 0;
#endif
}

/* Ground truth for --effective-config, better than any static inference: the gate table
 * already maps a flag name to the kernel it controls, and qwen_mmk_compiled() knows whether
 * that kernel exists in this build.  A QWEN_NO_VNNI on an Arm binary is not "honoured", it
 * controls a kernel that was never compiled.  Returns 1 when the flag belongs to a gate,
 * writing 1/0 into *compiled; 0 when the flag is not a gate flag at all. */
int qwen_flag_gate_status(const char *flag, int *compiled, const char **kernel) {
    if (!flag || !*flag) return 0;
    for (int k = 1; k < QWEN_MMK_COUNT; k++) {
        const qwen_mm_gate_t *g = &g_mm_gate[k];
        const char *names[5] = { g->off_env, g->on_env, g->minb_env, g->minrows_env, g->mincols_env };
        for (int i = 0; i < 5; i++) {
            if (names[i] && !strcmp(names[i], flag)) {
                if (compiled) *compiled = qwen_mmk_compiled(k);
                if (kernel) *kernel = g_mmk_info[k].name;
                return 1;
            }
        }
    }
    return 0;
}

int qwen_matmat_int8_max_b(void) {
    static const int cand[] = { QWEN_MMK_KLEIDI_I8, QWEN_MMK_INT8_AMX, QWEN_MMK_INT8_VNNI,
                                QWEN_MMK_INT8_AVX2, QWEN_MMK_INT8_SMMLA, QWEN_MMK_INT8_SDOT };
    const int n = (int)(sizeof cand / sizeof cand[0]);
    int best = 0;
    for (int B = 1; B <= 64; B++)
        for (int i = 0; i < n; i++) {
            int k = cand[i];
            if (qwen_mmk_compiled(k) && qwen_mmk_supported(k) &&
                qwen_mm_use(k, B, QWEN_MMK_PROBE_ROWS, QWEN_MMK_PROBE_COLS)) { best = B; break; }
        }
    return best;
}

const char *qwen_matmat_family_bf16(void) {
    static const int cand[] = { QWEN_MMK_KLEIDI_BF16, QWEN_MMK_BF16_AMX,
                                QWEN_MMK_BF16_AVX512, QWEN_MMK_BF16_BFMMLA };
    static char buf[96];
    return mmk_first_available(cand, (int)(sizeof cand / sizeof cand[0]),
                               "bf16 fixed-B twin (no bf16 GEMM gate on this build)",
                               buf, sizeof buf);
}

/* Does B=1 reach a native integer kernel on this build, or the f32 fused twin?
 * The AVX2 and AVX-512F(-no-VNNI) builds have int8/q4 GEMM but no integer GEMV, so every
 * B=1 call dequantises into the f32 path.  That is a missing kernel, not accidental
 * overhead -- there is no wasted conversion to remove, and reusing the int8 GEMM at B=1
 * would change the arithmetic -- so the honest fix here is to stop being silent about it.
 * These mirror the ladders in qwen_matvec_int8 / qwen_matvec_q4_0, disable envs included. */
int qwen_int8_gemv_native(void) {
    if (qwen_mmk_compiled(QWEN_MMK_KLEIDI_I8_GEMV) && qwen_kleidi_i8_enabled()) return 1;
#if defined(__AVX512VNNI__)
    { const char *e = getenv("QWEN_NO_VNNI"); if (!(e && e[0] == '1')) return 1; }
#endif
#if defined(__ARM_FEATURE_DOTPROD)
    { const char *e = getenv("QWEN_NO_SDOT"); if (!(e && e[0] == '1')) return 1; }
#endif
    return 0;
}
int qwen_q4_gemv_native(void) {
    /* the gate row exists on every build; only qwen_mmk_compiled() knows if the kernel does */
    if (qwen_mmk_compiled(QWEN_MMK_KLEIDI_Q4) && qwen_mmk_supported(QWEN_MMK_KLEIDI_Q4) &&
        qwen_mm_use(QWEN_MMK_KLEIDI_Q4, 1, 4096, 4096)) return 1;
#if defined(__AVX512VNNI__)
    if (!q4_sdot_disabled()) return 1;
#endif
#if defined(__ARM_FEATURE_DOTPROD)
    if (!q4_sdot_disabled()) return 1;
#endif
    return 0;
}
int qwen_sd_int8_usable(int in_ch, int out_ch) {
    return qwen_sd_int8_available() && in_ch == out_ch && in_ch > 0 && in_ch <= 768;
}

int qwen_int8_kp(int K, int blk) { return (K + blk - 1) / blk * blk; }

/* The x86 half of the activation-panel quantiser.
 *
 * Note what the reference actually is: the scalar tail rounds HALF AWAY FROM ZERO
 * ((int)(q >= 0 ? q + 0.5f : q - 0.5f), i.e. add the signed half then truncate toward zero)
 * and clamps to [-127, 127].  It is not lrintf, which rounds half to even.  x86 had no SIMD
 * path at all, so on x86 every byte this function has ever produced came from that scalar
 * expression -- it is the contract to preserve here, and the code below reproduces it
 * exactly rather than using _mm512_cvtps_epi32, whose round-half-to-even would differ on
 * every value that lands on a .5 boundary.  amax is a max reduction, which is
 * order-independent, so vectorising it is exact by construction.
 *
 * The NEON body rounds half to EVEN (vcvtnq_s32_f32) and saturates to [-128, 127].  So the
 * two platforms have DIFFERENT rounding contracts, and normalising them is a separate
 * decision -- ARM audio was qualified with half-to-even, x86 with half-away.  What is not
 * defensible is the body and the TAIL of the same kernel disagreeing, which is what
 * quant_round_i32() below fixes: each platform now rounds one way everywhere, so a value
 * quantises the same whether its index lands in the vector body or the remainder.  (The
 * -128 saturation is unreachable either way: inv = 127/amax bounds |q| by 127.)
 *
 * QWEN_NO_SIMD_QUANT=1 forces the scalar path on both x86 and ARM; the self-test uses it to
 * compare the two, which is only a real comparison because the tail follows the platform. */
static inline int quant_round_i32(float q) {
#ifdef __ARM_NEON
    return (int)vcvtns_s32_f32(q);              /* nearest, ties to even: what the body does */
#else
    return (int)(q >= 0 ? q + 0.5f : q - 0.5f); /* nearest, ties away: what x86 always did */
#endif
}
static int g_quant_simd_off = -1;
static int quant_simd_off(void) {
    if (g_quant_simd_off < 0) {
        const char *e = getenv("QWEN_NO_SIMD_QUANT");
        g_quant_simd_off = (e && e[0] == '1') ? 1 : 0;
    }
    return g_quant_simd_off;
}

void qwen_int8_quant_rows(int8_t *dst, float *scales, const float *src,
                          int rows, int K, int Kp, int blk) {
    int nblk = Kp / blk;
    for (int r = 0; r < rows; r++) {
        const float *s = src + (int64_t)r * K;
        int8_t *d = dst + (int64_t)r * Kp;
        float *sc = scales + (int64_t)r * nblk;
        for (int b = 0; b < nblk; b++) {
            int k0 = b * blk;
            int kn = K - k0 < blk ? K - k0 : blk;
            if (kn <= 0) { sc[b] = 1.0f; memset(d + k0, 0, blk); continue; }
            float amax = 0.0f;
            int i = 0;
#if defined(__AVX512F__)
            if (!quant_simd_off()) {
                const __m512i absmask = _mm512_set1_epi32(0x7FFFFFFF);
                __m512 vmax = _mm512_setzero_ps();
                for (; i + 15 < kn; i += 16) {
                    __m512i v = _mm512_castps_si512(_mm512_loadu_ps(s + k0 + i));
                    vmax = _mm512_max_ps(vmax, _mm512_castsi512_ps(_mm512_and_si512(v, absmask)));
                }
                amax = _mm512_reduce_max_ps(vmax);
            }
#endif
#ifdef __ARM_NEON
            if (!quant_simd_off()) {
                float32x4_t vmax = vdupq_n_f32(0.0f);
                for (; i + 3 < kn; i += 4)
                    vmax = vmaxq_f32(vmax, vabsq_f32(vld1q_f32(s + k0 + i)));
                amax = vmaxvq_f32(vmax);
            }
#endif
            for (; i < kn; i++) { float a = fabsf(s[k0 + i]); if (a > amax) amax = a; }
            float scale = amax > 0.0f ? amax / 127.0f : 1.0f;
            float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
            sc[b] = scale;
            i = 0;
#if defined(__AVX512F__)
            if (!quant_simd_off()) {
                const __m512i signbit = _mm512_set1_epi32((int)0x80000000);
                const __m512i halfbits = _mm512_castps_si512(_mm512_set1_ps(0.5f));
                const __m512 vinv = _mm512_set1_ps(inv);
                const __m512i clo = _mm512_set1_epi32(-127), chi = _mm512_set1_epi32(127);
                for (; i + 15 < kn; i += 16) {
                    __m512 q = _mm512_mul_ps(_mm512_loadu_ps(s + k0 + i), vinv);
                    /* the signed half the scalar adds: copysign(0.5f, q) */
                    __m512 bias = _mm512_castsi512_ps(_mm512_or_si512(
                        _mm512_and_si512(_mm512_castps_si512(q), signbit), halfbits));
                    __m512i v = _mm512_cvttps_epi32(_mm512_add_ps(q, bias));  /* toward zero */
                    v = _mm512_max_epi32(_mm512_min_epi32(v, chi), clo);
                    _mm_storeu_si128((__m128i *)(d + k0 + i), _mm512_cvtsepi32_epi8(v));
                }
            }
#endif
#ifdef __ARM_NEON
            if (!quant_simd_off()) {
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
            }
#endif
            for (; i < kn; i++) {
                float q = s[k0 + i] * inv;
                int v = quant_round_i32(q);
                if (v > 127) v = 127;
                if (v < -127) v = -127;
                d[k0 + i] = (int8_t)v;
            }
            for (; i < blk; i++) d[k0 + i] = 0;
        }
    }
}

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

/* Name the thread for /proc and top: the thread ownership table of a worker is then
 * readable without a debugger.  Zero cost after creation. */
static void qwen_thread_name_k(const char *prefix) {
    static _Atomic int counter = 0;
    char name[16];
    int n = atomic_fetch_add(&counter, 1);
    snprintf(name, sizeof name, "%.9s-%d", prefix, n);
#if defined(__APPLE__)
    pthread_setname_np(name);
#elif defined(__linux__)
    prctl(PR_SET_NAME, name, 0, 0, 0);
#endif
}
static void *sdp_worker_main(void *arg) {
    qwen_thread_name_k("sd-pool");
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

typedef struct { void (*fn)(void *); void *ctx; } sd_gcd_job_t;
static void sd_gcd_task(size_t tid, size_t nt, void *vj) {
    (void)tid; (void)nt;
    sd_gcd_job_t *j = (sd_gcd_job_t *)vj;
    j->fn(j->ctx);
}

static int g_sd_pool_mode = -1;
void qwen_sd_pool_default(int mode) { if (!getenv("QWEN_SD_POOL")) g_sd_pool_mode = mode ? 1 : 0; }
int qwen_sd_pool_mode(void) {
    if (g_sd_pool_mode < 0) {
        const char *e = getenv("QWEN_SD_POOL");
        g_sd_pool_mode = (e && (e[0] == '1' || e[0] == 'q')) ? 1 : 0;
    }
    return g_sd_pool_mode;
}

/* One execution budget, decided in one place.
 *
 * The CLI, the plain server and the batched server all drive the same engine, but only the
 * batched server used to claim the budget.  The other two could therefore run three compute
 * teams over the same cores at once: the engine pool, the decoder's private team and the
 * BLAS's own team.  The rule is not "copy the batched server's defaults everywhere"; it is:
 * when the engine already owns a usable multi-thread pool, the decoder tiles and the decoder
 * SGEMM run on THAT pool instead of raising teams beside it.
 *
 * Both setters yield to an explicit QWEN_SD_POOL / QWEN_BLAS_OWN, so a user override
 * survives.  With one thread there is no budget to own and this does nothing.  Idempotent:
 * a prefork child calls it again after its own qwen_set_threads(). */
void qwen_exec_budget_engine_owned(const char *who) {
    if (qwen_get_threads() <= 1) return;
    qwen_sd_pool_default(1);
    qwen_blas_own(1);
    /* Report once per process, not once per request: a prefork child is a new pid and says
     * it again, an entry point that claims the budget after another one stays quiet. */
    static pid_t reported = 0;
    if (reported == getpid()) return;
    reported = getpid();
    fprintf(stderr, "[%s] execution budget: decoder pool=%s · blas=%s · openblas threads now %d\n",
            who ? who : "engine",
            qwen_sd_pool_mode() ? "engine" : "private",
            qwen_blas_own_effective() ? "serial+partitioned"
                                      : (qwen_blas_own_get() ? "claimed but no thread control"
                                                             : "own team"),
            qwen_blas_threads_now());
}

static void sd_pool_run(void (*fn)(void *), void *ctx) {
    int nt = sd_pool_threads();
    if (nt < 1) nt = 1;
    if (nt == 1) { fn(ctx); return; }
    if (qwen_sd_pool_mode()) {
        /* The engine pool owns the CPU budget.  Every worker body pulls its tiles from an
         * atomic counter, so running it once per chunk on the pool is the same schedule the
         * private team used; inside an existing region one worker drains all tiles. */
        if (qwen_parallel_active()) { fn(ctx); return; }
        sd_gcd_job_t j = { fn, ctx };
        qwen_parallel((size_t)nt, sd_gcd_task, &j);
        return;
    }

    /* Private mode, but the pool can still absorb this without a second team: only where a
     * nested dispatch is actually safe, and never from inside a held team. */
    if (qwen_parallel_active()) { fn(ctx); return; }
    if (qwen_pool_nested_dispatch_ok()) {
        sd_gcd_job_t j = { fn, ctx };
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

    fn(ctx);

    pthread_mutex_lock(&sdp_mu);
    while (sdp_pending > 0) pthread_cond_wait(&sdp_done_cv, &sdp_mu);
    pthread_mutex_unlock(&sdp_mu);
}

#if defined(__ARM_FEATURE_DOTPROD) || defined(__AVX512VNNI__)

#if defined(__ARM_FEATURE_DOTPROD)

static inline void sd_tile_2x4(float *out, int out_ld, int m, int tcol,
                               const int8_t *Wq, const float *swb, const int32_t *wsum,
                               const float *bias,
                               const int8_t *Xq, const float *sab, int xrow,
                               int Kp, int blk, int nblk) {
    (void)wsum;
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

static inline void sd_tile_1xN(float *out, int out_ld, int m, int tcol,
                               const int8_t *Wq, const float *swb, const int32_t *wsum,
                               const float *bias,
                               const int8_t *Xq, const float *sab, int xrow, int ncols,
                               int Kp, int blk, int nblk) {
    (void)wsum;
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

#else

static inline float sd_hsum512(__m512 v) { return _mm512_reduce_add_ps(v); }

static inline void sd_tile_2x4(float *out, int out_ld, int m, int tcol,
                               const int8_t *Wq, const float *swb, const int32_t *wsum,
                               const float *bias,
                               const int8_t *Xq, const float *sab, int xrow,
                               int Kp, int blk, int nblk) {
    const __m512i flip = _mm512_set1_epi8((char)0x80);
    const int8_t *w0 = Wq + (size_t)(m + 0) * Kp, *w1 = Wq + (size_t)(m + 1) * Kp;
    const int8_t *xp[4] = { Xq + (size_t)(xrow + 0) * Kp, Xq + (size_t)(xrow + 1) * Kp,
                            Xq + (size_t)(xrow + 2) * Kp, Xq + (size_t)(xrow + 3) * Kp };
    const float *sw0 = swb + (size_t)(m + 0) * nblk, *sw1 = swb + (size_t)(m + 1) * nblk;
    const float *sap[4] = { sab + (size_t)(xrow + 0) * nblk, sab + (size_t)(xrow + 1) * nblk,
                            sab + (size_t)(xrow + 2) * nblk, sab + (size_t)(xrow + 3) * nblk };
    const int32_t *ws0 = wsum + (size_t)(m + 0) * nblk, *ws1 = wsum + (size_t)(m + 1) * nblk;

    __m512 f0[4] = { _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps() };
    __m512 f1[4] = { _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps() };

    for (int b = 0; b < nblk; b++) {
        __m512i a0[4] = { _mm512_setzero_si512(), _mm512_setzero_si512(),
                          _mm512_setzero_si512(), _mm512_setzero_si512() };
        __m512i a1[4] = { _mm512_setzero_si512(), _mm512_setzero_si512(),
                          _mm512_setzero_si512(), _mm512_setzero_si512() };
        int kend = (b + 1) * blk;
        for (int k = b * blk; k < kend; k += 64) {
            int rem = kend - k;
            __mmask64 msk = rem >= 64 ? ~(__mmask64)0 : (((__mmask64)1 << rem) - 1);
            __m512i wv0 = _mm512_maskz_loadu_epi8(msk, w0 + k);
            __m512i wv1 = _mm512_maskz_loadu_epi8(msk, w1 + k);
            for (int c = 0; c < 4; c++) {
                __m512i xu = _mm512_xor_si512(_mm512_maskz_loadu_epi8(msk, xp[c] + k), flip);
                a0[c] = _mm512_dpbusd_epi32(a0[c], xu, wv0);
                a1[c] = _mm512_dpbusd_epi32(a1[c], xu, wv1);
            }
        }
        float s0 = sw0[b], s1 = sw1[b];
        for (int c = 0; c < 4; c++) {
            float g0 = s0 * sap[c][b], g1 = s1 * sap[c][b];
            f0[c] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(a0[c]), _mm512_set1_ps(g0), f0[c]);
            f1[c] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(a1[c]), _mm512_set1_ps(g1), f1[c]);
        }
    }
    float b0 = bias ? bias[m + 0] : 0.0f, b1 = bias ? bias[m + 1] : 0.0f;
    float *o0 = out + (size_t)(m + 0) * out_ld + tcol;
    float *o1 = out + (size_t)(m + 1) * out_ld + tcol;
    for (int c = 0; c < 4; c++) {
        float k0 = 0.0f, k1 = 0.0f;
        for (int b = 0; b < nblk; b++) {
            k0 += 128.0f * (float)ws0[b] * sw0[b] * sap[c][b];
            k1 += 128.0f * (float)ws1[b] * sw1[b] * sap[c][b];
        }
        o0[c] = sd_hsum512(f0[c]) - k0 + b0;
        o1[c] = sd_hsum512(f1[c]) - k1 + b1;
    }
}

static inline void sd_tile_1xN(float *out, int out_ld, int m, int tcol,
                               const int8_t *Wq, const float *swb, const int32_t *wsum,
                               const float *bias,
                               const int8_t *Xq, const float *sab, int xrow, int ncols,
                               int Kp, int blk, int nblk) {
    const __m512i flip = _mm512_set1_epi8((char)0x80);
    const int8_t *w0 = Wq + (size_t)m * Kp;
    const float *sw0 = swb + (size_t)m * nblk;
    const int32_t *ws0 = wsum + (size_t)m * nblk;
    __m512 fc[4] = { _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps() };
    for (int b = 0; b < nblk; b++) {
        __m512i ac[4] = { _mm512_setzero_si512(), _mm512_setzero_si512(),
                          _mm512_setzero_si512(), _mm512_setzero_si512() };
        int kend = (b + 1) * blk;
        for (int k = b * blk; k < kend; k += 64) {
            int rem = kend - k;
            __mmask64 msk = rem >= 64 ? ~(__mmask64)0 : (((__mmask64)1 << rem) - 1);
            __m512i wv = _mm512_maskz_loadu_epi8(msk, w0 + k);
            for (int c = 0; c < ncols; c++) {
                __m512i xu = _mm512_xor_si512(
                    _mm512_maskz_loadu_epi8(msk, Xq + (size_t)(xrow + c) * Kp + k), flip);
                ac[c] = _mm512_dpbusd_epi32(ac[c], xu, wv);
            }
        }
        float s0 = sw0[b];
        for (int c = 0; c < ncols; c++)
            fc[c] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(ac[c]),
                                    _mm512_set1_ps(s0 * sab[(size_t)(xrow + c) * nblk + b]), fc[c]);
    }
    float bb = bias ? bias[m] : 0.0f;
    for (int c = 0; c < ncols; c++) {
        float kk = 0.0f;
        for (int b = 0; b < nblk; b++)
            kk += 128.0f * (float)ws0[b] * sw0[b] * sab[(size_t)(xrow + c) * nblk + b];
        out[(size_t)m * out_ld + tcol + c] = sd_hsum512(fc[c]) - kk + bb;
    }
}

#endif

/* ---- decoder AMX INT8 -------------------------------------------------------------
 *
 * The decoder conv is the largest matrix block in a request (76.6% of MACs at C=4) and it
 * never reached the matmat dispatcher, so it has run at 0% AMX.  Its geometry is already
 * tile-friendly: M = out_ch = 96 (six 16-row blocks), N = panel columns, and the quantised
 * buffers are padded to Kp = 768, which is a whole number of both 64-byte INT8 K-steps (12)
 * and 32-byte BF16 K-steps (24), so NO K-tail path is needed.  qwen_int8_quant_rows zero-fills
 * the padding on weights and activations alike, so [K, Kp) contributes exactly zero.
 *
 * The arithmetic contract of sd_gemm_panel is preserved: per 256-wide quant block accumulate
 * in int32, convert, apply that block's swb[m][b] * sab[c][b], and accumulate into fp32 in the
 * same block order.  The one deliberate difference is the zero-point term: VNNI only has
 * dpbusd (u8 x s8), so the scalar path biases the activation by +128 and subtracts
 * 128 * wsum[m][b] afterwards.  AMX has tdpbssd (s8 x s8), so both the bias and its correction
 * disappear.  The integer products are identical; removing the correction removes a
 * AMX signed x signed avoids the VNNI u8 offset/correction formulation entirely.  Numerical
 * equivalence against the existing decoder path is established by the parity harness, not
 * asserted here.
 *
 * Tiles: tmm0..5 hold the six 16x16 int32 accumulators (one per row block), tmm6 the 16x64
 * weight tile, tmm7 the packed activation tile.  Exactly the eight the ISA has.
 *
 * The Talker/CP `rows/thread >= 256` rule is deliberately NOT consulted: M=96 would fail it
 * outright.  This kernel is chosen on decoder geometry, and the parallelism is over column
 * panels, which the caller already owns.
 */
#if defined(__AMX_INT8__) && defined(__AMX_TILE__)

static atomic_llong g_sd_amx_rej[5];
void qwen_sd_amx_rej_report(void) {
    fprintf(stderr, "[SDAMX] reject M%%16=%lld Kp%%64=%lld blk%%64=%lld nc=%lld noamx=%lld\n",
            (long long)atomic_load(&g_sd_amx_rej[0]), (long long)atomic_load(&g_sd_amx_rej[1]),
            (long long)atomic_load(&g_sd_amx_rej[2]), (long long)atomic_load(&g_sd_amx_rej[3]),
            (long long)atomic_load(&g_sd_amx_rej[4]));
}
#define SD_AMX_MB 6                      /* 16-row accumulator tiles = 96 rows per pass */

/* Pack one 64-K x ncol slab of the activation panel into the AMX B layout:
 * row i of the tile carries k = 4i..4i+3 for every column, i.e. [i][n*4 + j]. */
static inline void sd_amx_pack_act(int8_t *dst, const int8_t *Xq, int Kp,
                                   int c0, int ncol, int k0) {
    const int cstride = ncol * 4;
    for (int i = 0; i < 16; i++) {
        int8_t *d = dst + (size_t)i * cstride;
        for (int n = 0; n < ncol; n++)
            memcpy(d + n * 4, Xq + (size_t)(c0 + n) * Kp + k0 + i * 4, 4);
    }
}

/* Same inputs as sd_gemm_panel.  Returns 0 when the geometry is not supported, so the caller
 * falls straight back to the existing path.
 *
 * M is covered in groups of 96 rows (six 16-row accumulator tiles), so the same eight-tile
 * design handles every decoder conv width: 96 is one group, 192 two, 384 four, 768 eight.
 * The activation slab is packed per K-step inside the quant-block loop: packing every K-step
 * up front would need 86 KB of stack at the widest real conv (Kp=5376), so the pack is
 * repeated per row group instead.  That is a memcpy, and v1 buys correctness across every
 * shape with it; hoisting it is a tuning step once the exact-shape numbers exist. */
static int sd_gemm_panel_amx(float *out, int out_ld, int M,
                             const int8_t *Wq, const float *swb, const int32_t *wsum,
                             const float *bias,
                             const int8_t *Xq, const float *sab,
                             int tcol0, int nc, int Kp, int blk) {
    (void)wsum;                          /* tdpbssd needs no zero-point correction */
    if (M % 16)            { atomic_fetch_add(&g_sd_amx_rej[0], 1); return 0; }
    if (Kp % 64)           { atomic_fetch_add(&g_sd_amx_rej[1], 1); return 0; }
    if (blk % 64)          { atomic_fetch_add(&g_sd_amx_rej[2], 1); return 0; }
    if (nc <= 0 || M <= 0) { atomic_fetch_add(&g_sd_amx_rej[3], 1); return 0; }
    if (!qwen_amx_int8_available()) { atomic_fetch_add(&g_sd_amx_rej[4], 1); return 0; }

    const int nblk   = Kp / blk;
    const int ksteps = blk / 64;         /* 64-byte INT8 K-steps inside one quant block */

    float   acc[SD_AMX_MB * 16][16];
    int32_t cbuf[16 * 16] __attribute__((aligned(64)));
    int8_t  bpack[16 * 64] __attribute__((aligned(64)));

    for (int c0 = 0; c0 < nc; c0 += 16) {
        const int ncol    = nc - c0 < 16 ? nc - c0 : 16;
        const int cstride = ncol * 4;

        qwen_amx_tilecfg cfg;
        memset(&cfg, 0, sizeof cfg);
        cfg.palette_id = 1;
        for (int t = 0; t < SD_AMX_MB; t++) { cfg.rows[t] = 16; cfg.colsb[t] = (uint16_t)cstride; }
        cfg.rows[6] = 16; cfg.colsb[6] = 64;                 /* weights, 64 int8 of K */
        cfg.rows[7] = 16; cfg.colsb[7] = (uint16_t)cstride;  /* packed activation */
        qwen_amx_prepare_config(&cfg, 0x53440000u | (unsigned)cstride);

        for (int m0 = 0; m0 < M; m0 += SD_AMX_MB * 16) {
            const int rows_left = M - m0;
            const int mb = rows_left / 16 < SD_AMX_MB ? rows_left / 16 : SD_AMX_MB;

            for (int i = 0; i < mb * 16; i++)
                for (int n = 0; n < ncol; n++) acc[i][n] = 0.0f;

            for (int b = 0; b < nblk; b++) {
                for (int t = 0; t < mb; t++) _tile_zero(t);

                for (int ks = 0; ks < ksteps; ks++) {
                    const int k = (b * ksteps + ks) * 64;
                    sd_amx_pack_act(bpack, Xq, Kp, c0, ncol, k);
                    _tile_loadd(7, bpack, cstride);
                    for (int t = 0; t < mb; t++) {
                        _tile_loadd(6, Wq + (size_t)(m0 + t * 16) * Kp + k, Kp);
                        switch (t) {
                            case 0: _tile_dpbssd(0, 6, 7); break;
                            case 1: _tile_dpbssd(1, 6, 7); break;
                            case 2: _tile_dpbssd(2, 6, 7); break;
                            case 3: _tile_dpbssd(3, 6, 7); break;
                            case 4: _tile_dpbssd(4, 6, 7); break;
                            default: _tile_dpbssd(5, 6, 7); break;
                        }
                    }
                }

                /* same epilogue order as the scalar path: convert this block, scale by
                 * swb[m][b] * sab[c][b], accumulate */
                for (int t = 0; t < mb; t++) {
                    switch (t) {
                        case 0: _tile_stored(0, cbuf, cstride); break;
                        case 1: _tile_stored(1, cbuf, cstride); break;
                        case 2: _tile_stored(2, cbuf, cstride); break;
                        case 3: _tile_stored(3, cbuf, cstride); break;
                        case 4: _tile_stored(4, cbuf, cstride); break;
                        default: _tile_stored(5, cbuf, cstride); break;
                    }
                    for (int r = 0; r < 16; r++) {
                        const int m = m0 + t * 16 + r;
                        const float sw = swb[(size_t)m * nblk + b];
                        for (int n = 0; n < ncol; n++)
                            acc[t * 16 + r][n] += (float)cbuf[r * ncol + n] *
                                                  (sw * sab[(size_t)(c0 + n) * nblk + b]);
                    }
                }
            }

            for (int i = 0; i < mb * 16; i++) {
                const int m = m0 + i;
                float *o = out + (size_t)m * out_ld + tcol0 + c0;
                const float bb = bias ? bias[m] : 0.0f;
                for (int n = 0; n < ncol; n++) o[n] = acc[i][n] + bb;
            }
        }
    }

    qwen_amx_finish_config();
    qwen_census_op(QWEN_PATH_DECODER_CONV_AMX_INT8, M, Kp, nc);
    MMSTAT(QWEN_MMK_INT8_AMX, M, Kp, nc);
    return 1;
}

static int sd_amx_enabled(void) {
    static atomic_int v = -1;
    int c = atomic_load_explicit(&v, memory_order_relaxed);
    if (c < 0) {
        const char *e = getenv("QWEN_SD_AMX");
        c = (e && e[0] && e[0] != '0');
        atomic_store_explicit(&v, c, memory_order_relaxed);
    }
    return c;
}
#else
static int sd_amx_enabled(void) { return 0; }
#endif  /* __AMX_INT8__ */

static void sd_gemm_panel(float *out, int out_ld, int M,
                          const int8_t *Wq, const float *swb, const int32_t *wsum,
                          const float *bias,
                          const int8_t *Xq, const float *sab,
                          int tcol0, int nc, int Kp, int blk) {
    int nblk = Kp / blk;
    for (int rb = 0; rb < M; rb += 32) {
        int rbe = rb + 32 < M ? rb + 32 : M;
        int c = 0;
        for (; c + 3 < nc; c += 4) {
            int m = rb;
            for (; m + 1 < rbe; m += 2)
                sd_tile_2x4(out, out_ld, m, tcol0 + c, Wq, swb, wsum, bias, Xq, sab, c, Kp, blk, nblk);
            for (; m < rbe; m++)
                sd_tile_1xN(out, out_ld, m, tcol0 + c, Wq, swb, wsum, bias, Xq, sab, c, 4, Kp, blk, nblk);
        }
        if (c < nc)
            for (int m = rb; m < rbe; m++)
                sd_tile_1xN(out, out_ld, m, tcol0 + c, Wq, swb, wsum, bias, Xq, sab, c, nc - c, Kp, blk, nblk);
    }
}

#define SD_INT8_NC 128

typedef struct {
    float *out;
    const float *in;
    const int8_t *Wq; const float *sw; const int32_t *wsum; const float *bias;
    int in_ch, out_ch, length, kernel, dilation, Kp, blk;
    _Atomic int next_panel;
    _Atomic int entered;      /* workers that reached the body: the honest denominator */
    int n_panels;
    int nc;                 /* output columns per panel: the parallel unit, see sd_conv_nc() */
} sd_conv_job_t;

/* The decoder conv parallelises over OUTPUT COLUMNS only, one panel per work item, so a short
 * layer cannot fill the pool: measured on the real 1.7B decoder, the first upsample block runs
 * 256 columns (M=768, K=5376, 1057 MMAC -- a quarter of all conv1 work) which is TWO panels of
 * 128, so four of six workers sat idle on the most expensive layer, and at one frame per chunk
 * it was a single panel running single-threaded. Size the panel from the work instead: each
 * column is im2col'd and quantised exactly once whatever the panel size, and its scale is
 * per column, so this changes only who computes what, never a single output byte -- proven at
 * one thread, where forcing four different panel widths gives one md5. Keep a floor so the
 * panel GEMM stays wide enough to be worth its setup. */
#define SD_INT8_NC_MIN 24
static int sd_conv_nc(int length, int nt) {
    static atomic_int forced = 0;                 /* QWEN_SD_CONV_NC=128 restores the old fixed
                                                   * panel, which is how this is A/B'd */
    int f = atomic_load_explicit(&forced, memory_order_relaxed);
    if (f == 0) {
        const char *e = getenv("QWEN_SD_CONV_NC");
        int v = e && *e ? atoi(e) : 0;
        f = (v >= SD_INT8_NC_MIN && v <= SD_INT8_NC) ? v + 1 : 1;
        atomic_store_explicit(&forced, f, memory_order_relaxed);
    }
    if (f > 1) return f - 1;
    if (nt < 2 || length <= SD_INT8_NC) return SD_INT8_NC;
    int want = (length + nt - 1) / nt;
    if (want >= SD_INT8_NC) return SD_INT8_NC;
    if (want < SD_INT8_NC_MIN) want = SD_INT8_NC_MIN;
    /* Round up to a multiple of 4 for SPEED, not for correctness: sd_gemm_panel walks a panel
     * in groups of four columns (sd_tile_2x4) and sends the remainder to the narrower
     * sd_tile_1xN, so an unaligned width pays that tail once per panel instead of once per
     * layer.  Correctness does not depend on it -- forcing 128, 124, 44 and 24 at one thread
     * gives the same WAV md5, and the differences seen at -j6 were the multi-thread float
     * ordering the engine already has, not the panel width. */
    want = (want + 3) & ~3;
    if (want > SD_INT8_NC) want = SD_INT8_NC;
    return want;
}

static void sd_conv1d_worker(void *vj) {
    sd_conv_job_t *j = (sd_conv_job_t *)vj;
    if (qwen_costmap_level()) atomic_fetch_add(&j->entered, 1);
    int K = j->in_ch * j->kernel;
    int nblk = j->Kp / j->blk;
    int pad_left = (j->kernel - 1) * j->dilation;
    /* per-thread, grow-once: the worker runs one panel at a time, so the column scratch is
     * reused across panels, conv layers and chunks instead of being re-allocated per call */
    float *colf = mm_scratch_sdcolf((size_t)SD_INT8_NC * K);
    int8_t *colq = mm_scratch_sdcolq((size_t)SD_INT8_NC * j->Kp);
    float *sa = mm_scratch_sdsa((size_t)SD_INT8_NC * nblk);
    long long claimed = 0;      /* accumulate locally: ONE profiler hook per worker, not
                                 * one per panel -- the per-unit call was 20% of all
                                 * instrumentation events and buys nothing a sum cannot give */
    for (;;) {
        int p = atomic_fetch_add(&j->next_panel, 1);
        if (p >= j->n_panels) break;
        claimed++;
        int t0 = p * j->nc;
        int nc = j->length - t0 < j->nc ? j->length - t0 : j->nc;
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
#if defined(__AMX_INT8__) && defined(__AMX_TILE__)
        if (!sd_amx_enabled() ||
            !sd_gemm_panel_amx(j->out, j->length, j->out_ch, j->Wq, j->sw, j->wsum, j->bias,
                               colq, sa, t0, nc, j->Kp, j->blk))
#endif
        sd_gemm_panel(j->out, j->length, j->out_ch, j->Wq, j->sw, j->wsum, j->bias,
                      colq, sa, t0, nc, j->Kp, j->blk);
    }
    qwen_region_units_at(QWEN_RGN_SD_CONV_INT8, claimed);
}

void qwen_conv1d_int8(float *out, const float *in,
                      const int8_t *Wq, const float *sw, const int32_t *wsum,
                      const float *bias,
                      int in_ch, int out_ch, int length, int kernel, int dilation,
                      int Kp, int blk) {
    qwen_census_op_len(QWEN_PATH_DECODER_CONV_INT8, out_ch, in_ch * kernel, length);
#if defined(__ARM_FEATURE_DOTPROD)
    qwen_census_leaf(QWEN_LEAF_SDOT);
#elif defined(__AVX512VNNI__)
    qwen_census_leaf(QWEN_LEAF_VNNI);
#elif defined(__AVX512F__)
    qwen_census_leaf(QWEN_LEAF_AVX512F);
#else
    qwen_census_leaf(QWEN_LEAF_SCALAR);
#endif
    sd_conv_job_t job = {
        .out = out, .in = in, .Wq = Wq, .sw = sw, .wsum = wsum, .bias = bias,
        .in_ch = in_ch, .out_ch = out_ch, .length = length,
        .kernel = kernel, .dilation = dilation, .Kp = Kp, .blk = blk,
    };
    job.nc = sd_conv_nc(length, sd_pool_threads());
    job.n_panels = (length + job.nc - 1) / job.nc;
    atomic_store(&job.next_panel, 0);
    /* The decomposition itself, so a report can say 2 of 6 workers instead of only a
     * wall time: this layer's panels are the parallel unit, and a short layer cannot
     * fill the pool no matter how fast the kernel is. */
    qwen_region_pool_at(QWEN_RGN_SD_CONV_INT8, sd_pool_threads(), job.n_panels);
    atomic_store(&job.entered, 0);
    sd_pool_run(sd_conv1d_worker, &job);
    qwen_region_workers_at(QWEN_RGN_SD_CONV_INT8, atomic_load(&job.entered));
}

typedef struct {
    float *out; int out_ld;
    const int8_t *Wq; const float *sw; const int32_t *wsum;
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
#if defined(__AMX_INT8__) && defined(__AMX_TILE__)
            if (!sd_amx_enabled() ||
                !sd_gemm_panel_amx(j->out + (size_t)m0 * j->out_ld, j->out_ld, m1 - m0,
                          j->Wq + (size_t)m0 * j->Kp, j->sw + (size_t)m0 * nblk,
                          j->wsum ? j->wsum + (size_t)m0 * nblk : NULL, NULL,
                          j->Xq + (size_t)t0 * j->Kp, j->sa + (size_t)t0 * nblk,
                          t0, nc, j->Kp, j->blk))
#endif
            sd_gemm_panel(j->out + (size_t)m0 * j->out_ld, j->out_ld, m1 - m0,
                          j->Wq + (size_t)m0 * j->Kp, j->sw + (size_t)m0 * nblk,
                          j->wsum ? j->wsum + (size_t)m0 * nblk : NULL, NULL,
                          j->Xq + (size_t)t0 * j->Kp, j->sa + (size_t)t0 * nblk,
                          t0, nc, j->Kp, j->blk);
        }
    }
}

void qwen_gemm_int8(float *out, int out_ld,
                    const int8_t *Wq, const float *sw, const int32_t *wsum,
                    const int8_t *Xq, const float *sa,
                    int M, int N, int Kp, int blk) {
    int nt = qwen_get_threads();
    int rpb = (M + nt * 2 - 1) / (nt * 2);
    rpb = (rpb + 1) & ~1;
    if (rpb < 2) rpb = 2;
    sd_gemm_job_t job = {
        .out = out, .out_ld = out_ld, .Wq = Wq, .sw = sw, .wsum = wsum, .Xq = Xq, .sa = sa,
        .M = M, .N = N, .Kp = Kp, .blk = blk,
        .rows_per_block = rpb, .n_blocks = (M + rpb - 1) / rpb,
    };
    atomic_store(&job.next_block, 0);
    sd_pool_run(sd_gemm_worker, &job);
}

#else

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
                      const int8_t *Wq, const float *sw, const int32_t *wsum,
                      const float *bias,
                      int in_ch, int out_ch, int length, int kernel, int dilation,
                      int Kp, int blk) {
    qwen_census_op_len(QWEN_PATH_DECODER_CONV_INT8, out_ch, in_ch * kernel, length);
    qwen_census_leaf(QWEN_LEAF_SCALAR);   /* the portable fallback: sd_scalar_dot */
    (void)wsum;
    int K = in_ch * kernel;
    int nblk = Kp / blk;
    int pad_left = (kernel - 1) * dilation;
    float *colf = mm_scratch_sdcolf((size_t)K);
    int8_t *colq = mm_scratch_sdcolq((size_t)Kp);
    float *sa = mm_scratch_sdsa((size_t)nblk);
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
}

void qwen_gemm_int8(float *out, int out_ld,
                    const int8_t *Wq, const float *sw, const int32_t *wsum,
                    const int8_t *Xq, const float *sa,
                    int M, int N, int Kp, int blk) {
    (void)wsum;
    int nblk = Kp / blk;
    for (int m = 0; m < M; m++)
        for (int t = 0; t < N; t++)
            out[(size_t)m * out_ld + t] =
                sd_scalar_dot(Wq + (size_t)m * Kp, sw + (size_t)m * nblk,
                              Xq + (size_t)t * Kp, sa + (size_t)t * nblk, Kp, blk);
}

#endif

#if (defined(__ARM_NEON) || defined(__AVX2__)) && !(defined(__APPLE__) && defined(USE_BLAS))
#define QWEN_SIN_POLY_MAX 8192.0f
#define QWEN_SIN_C1  (-1.0f / 6.0f)
#define QWEN_SIN_C2  ( 1.0f / 120.0f)
#define QWEN_SIN_C3  (-1.0f / 5040.0f)
#define QWEN_SIN_C4  ( 1.0f / 362880.0f)
#define QWEN_SIN_C5  (-1.0f / 39916800.0f)
#define QWEN_PI_HI   3.14159274101257324f
#define QWEN_PI_LO  (-8.74227800708368e-8f)
#define QWEN_INV_PI  0.31830988618379067f

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
    float32x4_t n = vrndaq_f32(vmulq_f32(y, inv_pi));
    float32x4_t u = vfmsq_f32(y, n, pi_hi);
    u = vfmaq_f32(u, n, pi_lo);
    float32x4_t u2 = vmulq_f32(u, u);
    float32x4_t p = vdupq_n_f32(QWEN_SIN_C5);
    p = vfmaq_f32(vdupq_n_f32(QWEN_SIN_C4), p, u2);
    p = vfmaq_f32(vdupq_n_f32(QWEN_SIN_C3), p, u2);
    p = vfmaq_f32(vdupq_n_f32(QWEN_SIN_C2), p, u2);
    p = vfmaq_f32(vdupq_n_f32(QWEN_SIN_C1), p, u2);
    p = vfmaq_f32(vdupq_n_f32(1.0f),        p, u2);
    float32x4_t s = vmulq_f32(u, p);
    return vmulq_f32(s, s);
}
#endif

#if defined(__AVX2__) && !(defined(__APPLE__) && defined(USE_BLAS))
static inline __m256 qwen_vsin2_avx2(__m256 y) {
    const __m256 inv_pi = _mm256_set1_ps(QWEN_INV_PI);
    const __m256 pi_hi  = _mm256_set1_ps(QWEN_PI_HI);
    const __m256 pi_lo  = _mm256_set1_ps(QWEN_PI_LO);
    __m256 n = _mm256_round_ps(_mm256_mul_ps(y, inv_pi),
                              _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m256 u = _mm256_fnmadd_ps(n, pi_hi, y);
    u = _mm256_fmadd_ps(n, pi_lo, u);
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
#endif

static int sd_phase_on_kern(void) {
    static int v = -1;
    if (v < 0) { const char *e = getenv("QWEN_SD_PHASE"); v = (e && *e && *e != '0'); }
    return v;
}
long long qwen_snake_expf_calls = 0, qwen_snake_vec_poly = 0, qwen_snake_vec_libm = 0,
          qwen_snake_scalar_tail = 0;
static void snake_row(float *data, int c, int length,
                      const float *log_alpha, const float *log_beta) {
    const int _snk = sd_phase_on_kern();
#if (defined(__ARM_NEON) || defined(__AVX2__)) && !(defined(__APPLE__) && defined(USE_BLAS))
    const int sin_poly = !qwen_sin_poly_off();
#endif
    {
        float a = expf(log_alpha[c]);
        float inv_b = expf(-log_beta[c]);
        if (_snk) qwen_snake_expf_calls += 2;
        float *row = data + (int64_t)c * length;

#if defined(__APPLE__) && defined(USE_BLAS)
        {
            int n = length;
            /* grow-once per thread: this ran once per channel row */
            float *temp = mm_scratch_snk((size_t)n);
            if (temp) {
            vDSP_vsmul(row, 1, &a, temp, 1, n);

            vvsinf(temp, temp, &n);

            vDSP_vsq(temp, 1, temp, 1, n);

            vDSP_vsma(temp, 1, &inv_b, row, 1, row, 1, n);
            }
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
                    if (_snk) qwen_snake_vec_poly++;
                    s2 = qwen_vsin2q_f32(ax);
                } else {
                    if (_snk) qwen_snake_vec_libm++;
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
                if (_snk) qwen_snake_scalar_tail++;
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

#define QWEN_SNAKE_MIN_WORK 65536

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

int qwen_argmax_matvec_bf16(const float *x, const uint16_t *W_bf16, int in_dim, int out_dim) {
    qwen_census_op(QWEN_PATH_ARGMAX_MATVEC_BF16, out_dim, in_dim, 1);
    int best_idx = 0;
    float best_val = -1e30f;
    int o = 0;

#ifdef __ARM_NEON
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

    for (; o < out_dim; o++) {
        const uint16_t *row = W_bf16 + (size_t)o * in_dim;
        float sum = 0.0f;
        for (int k = 0; k < in_dim; k++) sum += bf16_to_f32(row[k]) * x[k];
        if (sum > best_val) { best_val = sum; best_idx = o; }
    }
    return best_idx;
}

int qwen_kernel_selftest(void *out) {
    FILE *f = out ? (FILE *)out : stdout;
    uint64_t rng = 0x9E3779B97F4A7C15ull;
    #define NEXT_F (( (rng = rng * 6364136223846793005ull + 1442695040888963407ull) >> 40) \
                    / (float)(1u << 24) * 2.0f - 1.0f)

    const int cases[][2] = { {3072, 1024}, {2048, 1024}, {257, 320} };
    const int ncases = (int)(sizeof(cases) / sizeof(cases[0]));
    int failures = 0;

    fprintf(f, "qwen-tts kernel self-test (matvec correctness vs f32 reference)\n");
    qwen_caps_report(f);
    fprintf(f, "  (run with QWEN_NO_VNNI=1 / QWEN_NO_SDOT=1 / QWEN_NO_AMX=1 to test the fallback path)\n\n");

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
            uint32_t bits; memcpy(&bits, &v, 4);
            wb[i] = (uint16_t)((bits + 0x8000u) >> 16);
        }

        for (int r = 0; r < rows; r++) {
            float s = 0.0f;
            const uint16_t *row = wb + (size_t)r * cols;
            for (int k = 0; k < cols; k++) s += bf16_to_f32(row[k]) * x[k];
            ref[r] = s;
        }
        qwen_matvec_bf16(y, wb, x, rows, cols);
        double max_rel_bf16 = 0.0;
        {
            double l2n_bf = 0.0, l2d_bf = 0.0;
            for (int r = 0; r < rows; r++) {
                double denom = fabs(ref[r]) + 1e-3;
                double rel = fabs((double)y[r] - ref[r]) / denom;
                if (rel > max_rel_bf16) max_rel_bf16 = rel;
                double d = (double)y[r] - ref[r];
                l2n_bf += d * d; l2d_bf += (double)ref[r] * ref[r];
            }
#if defined(__AVX512BF16__) || defined(__ARM_FEATURE_BF16)
            max_rel_bf16 = sqrt(l2n_bf / (l2d_bf + 1e-12));
#else
            (void)l2n_bf; (void)l2d_bf;
#endif
        }

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
#if defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC) || defined(__AVX512BF16__) || \
    (defined(__AMX_BF16__) && defined(__AMX_TILE__))
                const double mmthr = 1e-2;
#else
                const double mmthr = 1e-4;
#endif
                fprintf(f, "  [%4dx%4d] matmat(B=%d) vs B*matvec: L2_rel=%.2e  %s\n",
                        rows, cols, B, l2rel, l2rel < mmthr ? "PASS" : "FAIL");
                if (!(l2rel < mmthr)) failures++;
            }
            free(Xb); free(Yb); free(xb); free(yc);
        }

        qwen_quantize_bf16_to_int8(wb, rows, cols, wi, sc);
        for (int r = 0; r < rows; r++) {
            const int8_t *row = wi + (size_t)r * cols;
            float s = 0.0f;
            for (int k = 0; k < cols; k++) s += (float)row[k] * x[k];
            ref[r] = sc[r] * s;
        }
        qwen_matvec_int8(y, wi, sc, x, rows, cols);
        double l2_num = 0.0, l2_den = 0.0;
        for (int r = 0; r < rows; r++) {
            double d = (double)y[r] - ref[r];
            l2_num += d * d;
            l2_den += (double)ref[r] * ref[r];
        }
        double rel_l2_i8 = sqrt(l2_num / (l2_den + 1e-12));

        int amax_ref = 0; float amax_val = ref[0];
        for (int r = 1; r < rows; r++) if (ref[r] > amax_val) { amax_val = ref[r]; amax_ref = r; }
        int amax_got = qwen_argmax_matvec_int8(x, wi, sc, cols, rows);
        int argmax_ok = (amax_got == amax_ref) ||
                        (amax_got >= 0 && amax_got < rows &&
                         (amax_val - ref[amax_got]) < 0.02 * (fabs(amax_val) + 1e-3));

        int bf16_ok = max_rel_bf16 < 1e-2;
        int i8_ok   = rel_l2_i8    < 3e-2;
        if (!bf16_ok || !i8_ok || !argmax_ok) failures++;
#if defined(__AVX512BF16__)
        const char *bf16_metric = "rel_L2";
#else
        const char *bf16_metric = "max_rel";
#endif
        fprintf(f, "  [%4dx%-4d] bf16 %s=%.2e %s | int8 rel_L2=%.2e %s | argmax %s (ref=%d got=%d)\n",
                rows, cols, bf16_metric, max_rel_bf16, bf16_ok ? "OK" : "FAIL",
                rel_l2_i8, i8_ok ? "OK" : "FAIL",
                argmax_ok ? "OK" : "FAIL", amax_ref, amax_got);

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

                {
                    for (int r = 0; r < rows; r++) {
                        const q4_0_block_t *row = wq + (size_t)r * nb;
                        float s = 0.0f;
                        for (int b = 0; b < nb; b++) {
                            float bs = qwen_f16_to_f32(row[b].scale_f16);
                            const float *xk = x + b * Q4_0_BLOCK_SIZE;
                            for (int i = 0; i < 16; i++) {
                                s += bs * (float)((int)(row[b].qs[i] & 0x0F) - 8) * xk[2 * i];
                                s += bs * (float)((int)(row[b].qs[i] >> 4)   - 8) * xk[2 * i + 1];
                            }
                        }
                        ref[r] = s;
                    }
                    qwen_matvec_q4_0(y, wq, x, rows, cols);
                    double l2n = 0.0, l2d = 0.0;
                    for (int r = 0; r < rows; r++) {
                        double d = (double)y[r] - ref[r];
                        l2n += d * d; l2d += (double)ref[r] * ref[r];
                    }
                    double l2rel = sqrt(l2n / (l2d + 1e-12));
                    int ok = l2rel < 3e-2;
                    fprintf(f, "  [%4dx%4d] matvec_q4_0 vs dequant ref: rel_L2=%.2e  %s\n",
                            rows, cols, l2rel, ok ? "PASS" : "FAIL");
                    if (!ok) failures++;
                }

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

    {
        const int shapes[][3] = { {96, 7, 1}, {192, 7, 3}, {384, 7, 9}, {768, 1, 1} };
        const int nsh = (int)(sizeof(shapes) / sizeof(shapes[0]));
        const int length = 67;
        const int blk = 64;
        for (int si = 0; si < nsh; si++) {
            int ch = shapes[si][0], kern = shapes[si][1], dil = shapes[si][2];
            int K = ch * kern, Kp = qwen_int8_kp(K, blk), nblk = Kp / blk;
            float   *in   = malloc((size_t)ch * length * sizeof(float));
            float   *wf2  = malloc((size_t)ch * K * sizeof(float));
            int8_t  *wq2  = aligned_malloc((size_t)ch * Kp);
            float   *sw2  = aligned_malloc((size_t)ch * nblk * sizeof(float));
            int32_t *ws2  = aligned_malloc((size_t)ch * nblk * sizeof(int32_t));
            float   *outk = malloc((size_t)ch * length * sizeof(float));
            float   *colf = malloc((size_t)K * sizeof(float));
            int8_t  *colq = aligned_malloc((size_t)Kp);
            float   *sa2  = aligned_malloc((size_t)nblk * sizeof(float));
            if (!in || !wf2 || !wq2 || !sw2 || !ws2 || !outk || !colf || !colq || !sa2) {
                fprintf(f, "  [conv1d_int8 ch=%d k=%d] OOM, skipped\n", ch, kern);
            } else {
                for (size_t i = 0; i < (size_t)ch * length; i++) in[i] = NEXT_F;
                for (size_t i = 0; i < (size_t)ch * K; i++) wf2[i] = NEXT_F;
                qwen_int8_quant_rows(wq2, sw2, wf2, ch, K, Kp, blk);
                for (int r = 0; r < ch; r++)
                    for (int b = 0; b < nblk; b++) {
                        int32_t acc = 0;
                        for (int k = b * blk; k < (b + 1) * blk; k++)
                            acc += (int32_t)wq2[(size_t)r * Kp + k];
                        ws2[(size_t)r * nblk + b] = acc;
                    }
                qwen_conv1d_int8(outk, in, wq2, sw2, ws2, NULL, ch, ch, length, kern, dil, Kp, blk);

                int pad_left = (kern - 1) * dil;
                double l2n = 0.0, l2d = 0.0; float worst = 0.0f;
                for (int t = 0; t < length; t++) {
                    for (int ic = 0; ic < ch; ic++)
                        for (int kk = 0; kk < kern; kk++) {
                            int pos = t - pad_left + kk * dil;
                            colf[ic * kern + kk] =
                                (pos >= 0 && pos < length) ? in[(size_t)ic * length + pos] : 0.0f;
                        }
                    qwen_int8_quant_rows(colq, sa2, colf, 1, K, Kp, blk);
                    for (int m = 0; m < ch; m++) {
                        float acc = 0.0f;
                        for (int b = 0; b < nblk; b++) {
                            int32_t ai = 0;
                            for (int k = b * blk; k < (b + 1) * blk; k++)
                                ai += (int32_t)wq2[(size_t)m * Kp + k] * (int32_t)colq[k];
                            acc += (float)ai * sw2[(size_t)m * nblk + b] * sa2[b];
                        }
                        float got = outk[(size_t)m * length + t];
                        float d = got - acc;
                        if (fabsf(d) > worst) worst = fabsf(d);
                        l2n += (double)d * d; l2d += (double)acc * acc;
                    }
                }
                double l2rel = l2d > 0 ? sqrt(l2n / l2d) : 0.0;
                int ok = l2rel < 1e-5;
                fprintf(f, "  [conv1d_int8 ch=%3d k=%d dil=%d] vs integer ref: rel_L2=%.2e max_abs=%.2e  %s\n",
                        ch, kern, dil, l2rel, worst, ok ? "PASS" : "FAIL");
                if (!ok) failures++;
            }
            free(in); free(wf2); free(wq2); free(sw2); free(ws2);
            free(outk); free(colf); free(colq); free(sa2);
        }
    }

    /* ---- activation-panel quantiser: SIMD must equal the scalar reference byte for byte --
     * The scalar expression is the contract on x86 (it is what this function has always
     * produced there), so the gate is exact equality of every output byte and every scale
     * bit, not a tolerance.  Runs both paths over the input classes that can separate them:
     * zeros, sign extremes, exact .5 rounding boundaries, tails that are not a multiple of
     * the vector width, and real decoder panel shapes. */
    {
        static const struct { int rows, K, blk; const char *what; } qcase[] = {
            {  1,   64, 32, "zeros"                 },
            {  1,  128, 32, "exact .5 boundaries"   },
            {  1,  128, 32, "sign extremes"         },
            {  3,   77, 32, "tail, K % 16 != 0"      },
            {  1,  100, 64, "tail, K % blk != 0"     },
            { 96,  672, 32, "decoder panel 96x7"    },
            {384, 2688, 32, "decoder panel 384x7"   },
            {768,  768, 64, "decoder panel 768x1"   },
        };
        const int ncase = (int)(sizeof qcase / sizeof qcase[0]);
#if !defined(__AVX512F__) && !defined(__ARM_NEON)
        /* Honest instead of a green tautology: with no SIMD path at all both runs execute
         * the same code here, so the comparison would prove nothing. */
        fprintf(f, "  [quant_rows] SIMD/scalar parity: n/a, this build has no SIMD "
                   "quantiser (%d cases skipped)\n", ncase);
#else
        /* Real on both sides now: QWEN_NO_SIMD_QUANT gates the NEON body as well as the
         * AVX-512 one, and the scalar tail follows the platform's rounding contract, so
         * this compares the vector body against the scalar reference instead of comparing
         * a half-to-even body against a half-away tail (PLAN P3.11). */
        for (int c = 0; c < ncase; c++) {
            const int rows = qcase[c].rows, K = qcase[c].K, blk = qcase[c].blk;
            const int Kp = qwen_int8_kp(K, blk), nblk = Kp / blk;
            float *src = (float *)malloc((size_t)rows * K * sizeof(float));
            int8_t *d1 = (int8_t *)malloc((size_t)rows * Kp);
            int8_t *d2 = (int8_t *)malloc((size_t)rows * Kp);
            float *s1 = (float *)malloc((size_t)rows * nblk * sizeof(float));
            float *s2 = (float *)malloc((size_t)rows * nblk * sizeof(float));
            if (!src || !d1 || !d2 || !s1 || !s2) {
                fprintf(f, "  [quant_rows %s] OOM, skipped\n", qcase[c].what);
                free(src); free(d1); free(d2); free(s1); free(s2); continue;
            }
            for (int r = 0; r < rows; r++)
                for (int k = 0; k < K; k++) {
                    float v;
                    switch (c) {
                        case 0: v = 0.0f; break;
                        /* amax becomes 127 so inv == 1.0f exactly and q lands on .5 */
                        case 1: v = (k == 0) ? 127.0f
                                             : ((k & 1) ? 1.0f : -1.0f) * ((float)((k % 254) / 2) + 0.5f);
                                break;
                        case 2: v = (k % 4 == 0) ? 1e30f : (k % 4 == 1) ? -1e30f
                                  : (k % 4 == 2) ? 1e-30f : -0.0f; break;
                        default: v = ((float)((k * 37 + r * 11) % 2001) - 1000.0f) / 250.0f; break;
                    }
                    src[(size_t)r * K + k] = v;
                }
            memset(d1, 0x5A, (size_t)rows * Kp); memset(d2, 0xA5, (size_t)rows * Kp);
            g_quant_simd_off = 0; qwen_int8_quant_rows(d1, s1, src, rows, K, Kp, blk);
            g_quant_simd_off = 1; qwen_int8_quant_rows(d2, s2, src, rows, K, Kp, blk);
            g_quant_simd_off = -1;
            int bad_b = memcmp(d1, d2, (size_t)rows * Kp) != 0;
            int bad_s = memcmp(s1, s2, (size_t)rows * nblk * sizeof(float)) != 0;
            int ok = !bad_b && !bad_s;
            fprintf(f, "  [quant_rows %-22s rows=%3d K=%4d blk=%2d] SIMD vs scalar: "
                       "bytes %s scales %s  %s\n",
                    qcase[c].what, rows, K, blk, bad_b ? "DIFFER" : "equal",
                    bad_s ? "DIFFER" : "equal", ok ? "PASS" : "FAIL");
            if (!ok) failures++;
            free(src); free(d1); free(d2); free(s1); free(s2);
        }
#endif
    }

    #undef NEXT_F
    fprintf(f, "\n%s (%d case%s failed)\n", failures ? "SELF-TEST FAILED" : "SELF-TEST PASSED",
            failures, failures == 1 ? "" : "s");
    return failures;
}

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

        double mb = (double)rows * cols * 2 / (1024 * 1024);
        int reps = mb > 8 ? 8 : 24;

        fprintf(f, "  [%4dx%4d]  (%.1f MB bf16)\n", rows, cols, mb);
        for (int p = 0; p < 3; p++) {
            const char *pn = p == 0 ? "bf16" : p == 1 ? "int8" : "int4";
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

#define QTUNE_MAXK   8
#define QTUNE_NB     5
static const int g_qtune_B[QTUNE_NB] = { 1, 2, 4, 8, 16 };
#define QTUNE_WIN    1.05

typedef struct { char label[96]; int rows, cols; } qtune_shape_t;
typedef struct { int hidden, heads, kvheads, head_dim, inter, vocab; } qtune_dims_t;

static const qtune_dims_t g_qtune_dims_06b_talker = { 1024, 16, 8, 128, 3072, 3072 };
static const qtune_dims_t g_qtune_dims_17b_talker = { 2048, 16, 8, 128, 6144, 3072 };
static const qtune_dims_t g_qtune_dims_cp         = { 1024, 16, 8, 128, 3072, 2048 };

static void qtune_add_shape(qtune_shape_t *v, int *n, int cap, int rows, int cols,
                            const char *label) {
    if (rows <= 0 || cols <= 0) return;
    for (int i = 0; i < *n; i++) {
        if (v[i].rows == rows && v[i].cols == cols) {
            size_t l = strlen(v[i].label);
            if (l + strlen(label) + 3 < sizeof(v[i].label))
                snprintf(v[i].label + l, sizeof(v[i].label) - l, " + %s", label);
            return;
        }
    }
    if (*n >= cap) return;
    snprintf(v[*n].label, sizeof(v[*n].label), "%s", label);
    v[*n].rows = rows; v[*n].cols = cols; (*n)++;
}

static void qtune_shapes_from_dims(const qtune_dims_t *d, const char *tag,
                                   qtune_shape_t *v, int *n, int cap) {
    int qd = d->heads * d->head_dim, kd = d->kvheads * d->head_dim;
    char lb[64];
    snprintf(lb, sizeof(lb), "%s q_proj",   tag); qtune_add_shape(v, n, cap, qd,        d->hidden, lb);
    snprintf(lb, sizeof(lb), "%s k/v_proj", tag); qtune_add_shape(v, n, cap, kd,        d->hidden, lb);
    snprintf(lb, sizeof(lb), "%s o_proj",   tag); qtune_add_shape(v, n, cap, d->hidden, qd,        lb);
    snprintf(lb, sizeof(lb), "%s gate/up",  tag); qtune_add_shape(v, n, cap, d->inter,  d->hidden, lb);
    snprintf(lb, sizeof(lb), "%s down",     tag); qtune_add_shape(v, n, cap, d->hidden, d->inter,  lb);
    snprintf(lb, sizeof(lb), "%s head",     tag); qtune_add_shape(v, n, cap, d->vocab,  d->hidden, lb);
}

static long qtune_json_int(const char *p) {
    while (*p && *p != ':') p++;
    if (*p) p++;
    while (*p == ' ' || *p == '\n' || *p == '\t' || *p == '\r') p++;
    return strtol(p, NULL, 10);
}

static int qtune_scan_config(const char *json, qtune_dims_t *tk, qtune_dims_t *cp) {
    const char *p = strstr(json, "\"talker_config\"");
    if (!p) return 0;
    p = strchr(p, '{');
    if (!p) return 0;
    int depth = 0, cpd = -1;
    for (; *p; p++) {
        if (*p == '{') { depth++; continue; }
        if (*p == '}') {
            depth--;
            if (cpd > 0 && depth < cpd) cpd = -1;
            if (depth == 0) break;
            continue;
        }
        if (*p != '"') continue;
        const char *k = p + 1, *e = strchr(k, '"');
        if (!e) break;
        size_t n = (size_t)(e - k);
        const char *after = e + 1;
        p = e;
        while (*after == ' ' || *after == '\n' || *after == '\t' || *after == '\r') after++;
        if (*after != ':') continue;
        #define QKEY(S) (n == sizeof(S) - 1 && strncmp(k, S, n) == 0)
        if (depth == 1 && QKEY("code_predictor_config")) { cpd = 2; continue; }
        qtune_dims_t *d = (depth == 1) ? tk : ((cpd > 0 && depth == cpd) ? cp : NULL);
        if (!d) continue;
        if      (QKEY("hidden_size"))         d->hidden   = (int)qtune_json_int(after);
        else if (QKEY("num_attention_heads")) d->heads    = (int)qtune_json_int(after);
        else if (QKEY("num_key_value_heads")) d->kvheads  = (int)qtune_json_int(after);
        else if (QKEY("head_dim"))            d->head_dim = (int)qtune_json_int(after);
        else if (QKEY("intermediate_size"))   d->inter    = (int)qtune_json_int(after);
        else if (QKEY("vocab_size"))          d->vocab    = (int)qtune_json_int(after);
        #undef QKEY
    }
    return tk->hidden > 0 && tk->heads > 0 && tk->head_dim > 0 && tk->inter > 0;
}

typedef struct {
    int fmt;
    int rows, cols, B;
    const uint16_t *wb; const int8_t *wi; const float *sc; const q4_0_block_t *wq;
    const float *X; float *Y; float *xcol; float *ycol;
} qtune_ctx;

static void qtune_run_matmat(void *v) {
    qtune_ctx *c = (qtune_ctx *)v;
    if (c->fmt == 0)      qwen_matmat_bf16(c->Y, c->wb, c->X, c->rows, c->cols, c->B);
    else if (c->fmt == 1) qwen_matmat_int8(c->Y, c->wi, c->sc, c->X, c->rows, c->cols, c->B);
    else                  qwen_matmat_q4_0(c->Y, c->wq, c->X, c->rows, c->cols, c->B);
}
static void qtune_run_bmatvec(void *v) {
    qtune_ctx *c = (qtune_ctx *)v;
    for (int b = 0; b < c->B; b++) {
        for (int k = 0; k < c->cols; k++) c->xcol[k] = c->X[(size_t)k * c->B + b];
        if (c->fmt == 0)      qwen_matvec_bf16(c->ycol, c->wb, c->xcol, c->rows, c->cols);
        else if (c->fmt == 1) qwen_matvec_int8(c->ycol, c->wi, c->sc, c->xcol, c->rows, c->cols);
        else                  qwen_matvec_q4_0(c->ycol, c->wq, c->xcol, c->rows, c->cols);
        for (int r = 0; r < c->rows; r++) c->Y[(size_t)r * c->B + b] = c->ycol[r];
    }
}

static double qtune_now_ms(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec * 1e3 + (double)t.tv_nsec * 1e-6;
}
static double qtune_bench(void (*fn)(void *), void *ctx, double target_ms) {
    fn(ctx);
    double t0 = qtune_now_ms(); fn(ctx);
    double one = qtune_now_ms() - t0;
    int reps = 1;
    if (one > 1e-6 && one < target_ms) {
        reps = (int)(target_ms / one);
        if (reps < 1) reps = 1;
        if (reps > 200) reps = 200;
    }
    double best = 1e30;
    for (int round = 0; round < 3; round++) {
        t0 = qtune_now_ms();
        for (int i = 0; i < reps; i++) fn(ctx);
        double t = (qtune_now_ms() - t0) / reps;
        if (t < best) best = t;
    }
    return best;
}

static int qtune_which_fired(qtune_ctx *c) {
    atomic_store_explicit(&g_mm_stats, 1, memory_order_relaxed);
    qwen_matmat_stats_reset();
    qtune_run_matmat(c);
    atomic_store_explicit(&g_mm_stats, 0, memory_order_relaxed);
    int best = 0; long long bm = 0;
    for (int i = 1; i < QWEN_MMK_COUNT; i++) {
        long long m = atomic_load_explicit(&g_mm_macs[i], memory_order_relaxed);
        if (m > bm) { bm = m; best = i; }
    }
    return best;
}

typedef struct {
    int  mmk;
    int  fired;
    char name[40];
    double ms[QTUNE_NB];
    double sp[QTUNE_NB];
    int    ok;
} qtune_kres_t;

static int qtune_kernels(int fmt, qtune_kres_t *out) {
    int n = 0;
    #define PUSHK(ID) do { if (n < QTUNE_MAXK) { out[n].mmk = (ID); out[n].fired = 0; \
        snprintf(out[n].name, sizeof(out[n].name), "%s", g_mmk_info[ID].name); n++; } } while (0)
    if (fmt == 0) {
#if defined(__AMX_BF16__) && defined(__AMX_TILE__)
        if (qwen_amx_bf16_ready()) PUSHK(QWEN_MMK_BF16_AMX);
#endif
#if defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC)
        PUSHK(QWEN_MMK_BF16_BFMMLA);
#endif
#if defined(__AVX512BF16__)
        PUSHK(QWEN_MMK_BF16_AVX512);
#endif
    } else if (fmt == 1) {
#if defined(__AMX_INT8__) && defined(__AMX_TILE__)
        if (qwen_amx_int8_ready()) PUSHK(QWEN_MMK_INT8_AMX);
#endif
#if defined(__AVX512VNNI__)
        PUSHK(QWEN_MMK_INT8_VNNI);
#endif
#if defined(__AVX2__)
        PUSHK(QWEN_MMK_INT8_AVX2);
#endif
#if defined(__ARM_FEATURE_MATMUL_INT8)
        PUSHK(QWEN_MMK_INT8_SMMLA);
#endif
#if defined(__ARM_FEATURE_DOTPROD)
        PUSHK(QWEN_MMK_INT8_SDOT);
#endif
    } else {
#if defined(__AMX_INT8__) && defined(__AMX_TILE__)
        if (qwen_amx_int8_ready()) PUSHK(QWEN_MMK_Q4_AMX);
#endif
#if defined(__AVX512VNNI__)
        PUSHK(QWEN_MMK_Q4_VNNI);
#endif
#if defined(__AVX2__)
        PUSHK(QWEN_MMK_Q4_AVX2);
#endif
#if defined(__ARM_FEATURE_MATMUL_INT8)
        PUSHK(QWEN_MMK_Q4_SMMLA);
#endif
    }
    #undef PUSHK
    if (n < QTUNE_MAXK) {
        out[n].mmk = -1; out[n].fired = 0;
        snprintf(out[n].name, sizeof(out[n].name), "(dispatcher tail)");
        n++;
    }
    return n;
}

static int qtune_crossover(const int *win) {
    for (int i = 0; i < QTUNE_NB; i++) {
        if (!win[i]) continue;
        int all = 1;
        for (int j = i; j < QTUNE_NB; j++) if (!win[j]) { all = 0; break; }
        if (all) return g_qtune_B[i];
    }
    return 0;
}

int qwen_matmat_tune(void *out, const char *model_dir) {
    FILE *f = out ? (FILE *)out : stdout;
    const int full = qwen_get_threads() > 0 ? qwen_get_threads() : 1;
    const char *jpath = getenv("QWEN_TUNE_JSON");
    FILE *js = NULL;

    qtune_shape_t shp[24]; int nshp = 0;
    char src[256];
    qtune_dims_t tk = { 0, 0, 0, 0, 0, 0 }, cp = { 0, 0, 0, 0, 0, 0 };
    int from_cfg = 0;
    if (model_dir && model_dir[0]) {
        char path[1024];
        snprintf(path, sizeof(path), "%s/config.json", model_dir);
        FILE *cf = fopen(path, "rb");
        if (cf) {
            fseek(cf, 0, SEEK_END); long sz = ftell(cf); fseek(cf, 0, SEEK_SET);
            if (sz > 0 && sz < (1 << 22)) {
                char *buf = (char *)malloc((size_t)sz + 1);
                if (buf && fread(buf, 1, (size_t)sz, cf) == (size_t)sz) {
                    buf[sz] = 0;
                    from_cfg = qtune_scan_config(buf, &tk, &cp);
                }
                free(buf);
            }
            fclose(cf);
        }
    }
    if (from_cfg) {
        qtune_shapes_from_dims(&tk, "talker", shp, &nshp, 24);
        if (cp.hidden > 0) qtune_shapes_from_dims(&cp, "cp", shp, &nshp, 24);
        snprintf(src, sizeof(src),
                 "%s/config.json  talker{hidden=%d heads=%d kv=%d hd=%d ffn=%d vocab=%d}"
                 "  cp{hidden=%d ffn=%d vocab=%d}",
                 model_dir, tk.hidden, tk.heads, tk.kvheads, tk.head_dim, tk.inter, tk.vocab,
                 cp.hidden, cp.inter, cp.vocab);
    } else {
        qtune_shapes_from_dims(&g_qtune_dims_06b_talker, "0.6B", shp, &nshp, 24);
        qtune_shapes_from_dims(&g_qtune_dims_cp,         "cp",   shp, &nshp, 24);
        qtune_shapes_from_dims(&g_qtune_dims_17b_talker, "1.7B", shp, &nshp, 24);
        snprintf(src, sizeof(src),
                 "DECLARED DEFAULTS (no -d): 0.6B talker{h=1024 ffn=3072 vocab=3072}, "
                 "1.7B talker{h=2048 ffn=6144 vocab=3072}, cp{h=1024 ffn=3072 vocab=2048}, "
                 "heads=16 kv=8 head_dim=128");
    }
    const char *qk = getenv("QWEN_TUNE_QUICK");
    if (qk && qk[0] && qk[0] != '0') {
        int m = 0;
        for (int i = 0; i < nshp; i++)
            if ((long long)shp[i].rows * shp[i].cols <= 4L * 1024 * 1024) shp[m++] = shp[i];
        nshp = m;
    }

    fprintf(f, "matmat-tune: measure the g_mm_gate[] thresholds on THIS box\n");
    fprintf(f, "  shapes   : %s\n", src);
    fprintf(f, "  B grid   : 1 2 4 8 16   (16 = max_b, a buffer bound, not a knob)\n");
    fprintf(f, "             B=1 is INFORMATIONAL: the engine takes the single-slot shortcut\n");
    fprintf(f, "             (QWEN_MMK_SOLO) and never calls a matmat at B=1, so a bad number\n");
    fprintf(f, "             there costs nothing — it only says the twin has no B=1 kernel.\n");
    fprintf(f, "  threads  : j=1 (compute-bound) AND j=%d (what the server runs at)\n", full);
    fprintf(f, "  reference: B x qwen_matvec_* of the SAME format, gather/scatter included\n");
    fprintf(f, "  a win    : speedup >= %.2fx AND fastest of the available kernels, and it\n", QTUNE_WIN);
    fprintf(f, "             must hold for every larger B in the grid\n\n");

    if (jpath && jpath[0]) {
        js = fopen(jpath, "w");
        if (js) {
            time_t now = time(NULL);
            struct tm tmv;
            char ts[32] = "";
            if (gmtime_r(&now, &tmv)) strftime(ts, sizeof(ts), "%Y-%m-%dT%H:%M:%SZ", &tmv);
            fprintf(js, "{\n  \"tool\": \"qwen_tts --matmat-tune\",\n");
            fprintf(js, "  \"generated_utc\": \"%s\",\n", ts);
            fprintf(js, "  \"shapes_source\": \"%s\",\n", src);
            fprintf(js, "  \"threads_full\": %d,\n  \"win_margin\": %.2f,\n", full, QTUNE_WIN);
            fprintf(js, "  \"B_grid\": [1, 2, 4, 8, 16],\n  \"cells\": [\n");
        }
    }

    int  agg_worst[3][QTUNE_MAXK + 1][2], agg_best[3][QTUNE_MAXK + 1][2];
    int  agg_fired[3][QTUNE_MAXK + 1];
    char agg_name[3][QTUNE_MAXK + 1][40];
    int  agg_mmk[3][QTUNE_MAXK + 1], agg_n[3] = { 0, 0, 0 };
    memset(agg_worst, 0, sizeof(agg_worst));
    memset(agg_best, 0, sizeof(agg_best));
    memset(agg_fired, 0, sizeof(agg_fired));
    memset(agg_name, 0, sizeof(agg_name));
    memset(agg_mmk, 0, sizeof(agg_mmk));

    double noise_max = 0.0, noise_sum = 0.0; int noise_n = 0;
    char noise_where[128] = "";

    int first_cell = 1;
    for (int si = 0; si < nshp; si++) {
        int rows = shp[si].rows, cols = shp[si].cols;
        int nb = cols / Q4_0_BLOCK_SIZE;
        const int Bmax = g_qtune_B[QTUNE_NB - 1];
        uint16_t *wb = (uint16_t *)malloc((size_t)rows * cols * sizeof(uint16_t));
        int8_t   *wi = (int8_t *)malloc((size_t)rows * cols);
        float    *sc = (float *)malloc((size_t)rows * sizeof(float));
        q4_0_block_t *wq = (q4_0_block_t *)malloc((size_t)rows * nb * sizeof(q4_0_block_t));
        float *X  = (float *)malloc((size_t)cols * Bmax * sizeof(float));
        float *Y  = (float *)malloc((size_t)rows * Bmax * sizeof(float));
        float *xc = (float *)malloc((size_t)cols * sizeof(float));
        float *yc = (float *)malloc((size_t)rows * sizeof(float));
        if (!wb || !wi || !sc || !wq || !X || !Y || !xc || !yc) {
            fprintf(f, "── %s [%dx%d] : OOM, skipped\n", shp[si].label, rows, cols);
            free(wb); free(wi); free(sc); free(wq); free(X); free(Y); free(xc); free(yc);
            continue;
        }
        uint64_t rng = 0x9E3779B97F4A7C15ull ^ ((uint64_t)rows << 20) ^ (uint64_t)cols;
        #define QRF (((rng = rng * 6364136223846793005ull + 1442695040888963407ull) >> 40) \
                     / (float)(1u << 24) * 2.0f - 1.0f)
        for (size_t i = 0; i < (size_t)rows * cols; i++) {
            float v = QRF; uint32_t bits; memcpy(&bits, &v, 4);
            wb[i] = (uint16_t)((bits + 0x8000u) >> 16);
        }
        qwen_quantize_bf16_to_int8(wb, rows, cols, wi, sc);
        qwen_quantize_bf16_to_q4_0(wb, rows, cols, wq);
        for (size_t i = 0; i < (size_t)cols * Bmax; i++) X[i] = QRF;
        #undef QRF

        fprintf(f, "── %s  [%d x %d]  (%.1f MB bf16 / %.1f MB int8)\n",
                shp[si].label, rows, cols,
                (double)rows * cols * 2 / (1024 * 1024), (double)rows * cols / (1024 * 1024));

        for (int fmt = 0; fmt < 3; fmt++) {
            const char *fn = fmt == 0 ? "bf16" : fmt == 1 ? "int8" : "q4_0";
            if (fmt == 2 && cols % Q4_0_BLOCK_SIZE) {
                fprintf(f, "   %-5s  cols not a multiple of %d — no q4 path\n", fn, Q4_0_BLOCK_SIZE);
                continue;
            }
            qtune_kres_t kr[QTUNE_MAXK + 1];
            memset(kr, 0, sizeof(kr));
            int nk = qtune_kernels(fmt, kr);
            if (agg_n[fmt] == 0) {
                agg_n[fmt] = nk;
                for (int k = 0; k < nk; k++) {
                    agg_mmk[fmt][k] = kr[k].mmk;
                    snprintf(agg_name[fmt][k], sizeof(agg_name[fmt][k]), "%s", kr[k].name);
                    for (int t = 0; t < 2; t++) { agg_worst[fmt][k][t] = -1; agg_best[fmt][k][t] = -1; }
                }
            }

            for (int tm = 0; tm < 2; tm++) {
                int nt = tm == 0 ? 1 : full;
                if (tm == 1 && full == 1) break;
                qwen_set_threads(nt);
                double ref[QTUNE_NB];
                qtune_ctx c;
                memset(&c, 0, sizeof(c));
                c.fmt = fmt; c.rows = rows; c.cols = cols;
                c.wb = wb; c.wi = wi; c.sc = sc; c.wq = wq;
                c.X = X; c.Y = Y; c.xcol = xc; c.ycol = yc;

                qwen_mm_force_kernel(0);
                for (int bi = 0; bi < QTUNE_NB; bi++) {
                    c.B = g_qtune_B[bi];
                    ref[bi] = qtune_bench(qtune_run_bmatvec, &c, 30.0);
                }
                c.B = g_qtune_B[QTUNE_NB - 1];
                double refchk = qtune_bench(qtune_run_bmatvec, &c, 30.0);
                double nz = ref[QTUNE_NB - 1] > 0 ? fabs(refchk / ref[QTUNE_NB - 1] - 1.0) : 0.0;
                noise_sum += nz; noise_n++;
                if (nz > noise_max) {
                    noise_max = nz;
                    snprintf(noise_where, sizeof(noise_where), "%s %dx%d j=%d",
                             fn, rows, cols, nt);
                }
                for (int k = 0; k < nk; k++) {
                    qwen_mm_force_kernel(kr[k].mmk);
                    kr[k].ok = 1;
                    for (int bi = 0; bi < QTUNE_NB; bi++) {
                        c.B = g_qtune_B[bi];
                        if (bi == 0) kr[k].fired = qtune_which_fired(&c);
                        kr[k].ms[bi] = qtune_bench(qtune_run_matmat, &c, 30.0);
                        kr[k].sp[bi] = kr[k].ms[bi] > 0 ? ref[bi] / kr[k].ms[bi] : 0.0;
                    }
                    if (kr[k].mmk > 0 && kr[k].fired != kr[k].mmk) kr[k].ok = 0;
                }
                qwen_mm_force_kernel(0);

                fprintf(f, "   %-5s j=%-3d %-26s", fn, nt, "B x matvec (reference)");
                for (int bi = 0; bi < QTUNE_NB; bi++) fprintf(f, " %8.3fms", ref[bi]);
                fprintf(f, "\n");
                for (int k = 0; k < nk; k++) {
                    const char *nm = kr[k].name;
                    char tail[48];
                    if (kr[k].mmk < 0) {
                        snprintf(tail, sizeof(tail), "tail: %s",
                                 kr[k].fired > 0 ? g_mmk_info[kr[k].fired].name : "(unattributed)");
                        nm = tail;
                        snprintf(agg_name[fmt][k], sizeof(agg_name[fmt][k]), "%s", tail);
                        snprintf(kr[k].name, sizeof(kr[k].name), "%s", tail);
                    }
                    if (kr[k].fired > 0) agg_fired[fmt][k] = kr[k].fired;
                    fprintf(f, "   %-5s j=%-3d %-26s", fn, nt, nm);
                    if (!kr[k].ok) {
                        fprintf(f, "   DID NOT FIRE (blocked by a capability check, not by a threshold)\n");
                        continue;
                    }
                    for (int bi = 0; bi < QTUNE_NB; bi++) fprintf(f, "   %6.2fx ", kr[k].sp[bi]);
                    fprintf(f, "\n");
                }

                for (int k = 0; k < nk; k++) {
                    int win[QTUNE_NB];
                    for (int bi = 0; bi < QTUNE_NB; bi++) {
                        int w = kr[k].ok && kr[k].sp[bi] >= QTUNE_WIN;
                        for (int o = 0; o < nk && w; o++)
                            if (o != k && kr[o].ok && kr[o].ms[bi] < kr[k].ms[bi]) w = 0;
                        win[bi] = w;
                    }
                    int x = qtune_crossover(win);
                    int *aw = &agg_worst[fmt][k][tm], *ab = &agg_best[fmt][k][tm];
                    if (*aw < 0)                  *aw = x;
                    else if (*aw == 0 || x == 0)  *aw = 0;
                    else if (x > *aw)             *aw = x;
                    if (x > 0 && (*ab <= 0 || x < *ab)) *ab = x;

                    if (js) {
                        fprintf(js, "%s    {\"format\": \"%s\", \"rows\": %d, \"cols\": %d, "
                                    "\"label\": \"%s\", \"threads\": %d, \"kernel\": \"%s\", "
                                    "\"mmk\": %d, \"ran\": %d, \"fired_mmk\": %d, \"crossover_B\": %d,\n"
                                    "     \"ref_ms\": [", first_cell ? "" : ",\n", fn, rows, cols,
                                shp[si].label, nt, kr[k].name, kr[k].mmk, kr[k].ok, kr[k].fired, x);
                        for (int bi = 0; bi < QTUNE_NB; bi++)
                            fprintf(js, "%s%.5f", bi ? ", " : "", ref[bi]);
                        fprintf(js, "], \"ms\": [");
                        for (int bi = 0; bi < QTUNE_NB; bi++)
                            fprintf(js, "%s%.5f", bi ? ", " : "", kr[k].ms[bi]);
                        fprintf(js, "], \"speedup\": [");
                        for (int bi = 0; bi < QTUNE_NB; bi++)
                            fprintf(js, "%s%.4f", bi ? ", " : "", kr[k].sp[bi]);
                        fprintf(js, "]}");
                        first_cell = 0;
                    }
                }
            }
            fprintf(f, "\n");
        }
        free(wb); free(wi); free(sc); free(wq); free(X); free(Y); free(xc); free(yc);
    }
    qwen_set_threads(full);
    if (js) fprintf(js, "\n  ],\n");

    fprintf(f, "══ CROSSOVER SUMMARY  (min B from which the kernel wins on EVERY measured shape;\n");
    fprintf(f, "   \"easiest\" = the same on the single most favourable shape — if they differ a\n");
    fprintf(f, "   lot, one global min_b is the wrong shape of answer and the gate wants rows/cols)\n");
    fprintf(f, "   %-26s %-9s %-9s %-9s  %s\n", "kernel", "j=1", "j=full", "easiest", "verdict");
    for (int fmt = 0; fmt < 3; fmt++) {
        for (int k = 0; k < agg_n[fmt]; k++) {
            int w1 = agg_worst[fmt][k][0], wf = full > 1 ? agg_worst[fmt][k][1] : w1;
            int eb = full > 1 ? agg_best[fmt][k][1] : agg_best[fmt][k][0];
            char c1[16], cf[16], ce[16];
            if (w1 <= 0) snprintf(c1, sizeof(c1), "never"); else snprintf(c1, sizeof(c1), "B>=%d", w1);
            if (wf <= 0) snprintf(cf, sizeof(cf), "never"); else snprintf(cf, sizeof(cf), "B>=%d", wf);
            if (eb <= 0) snprintf(ce, sizeof(ce), "never"); else snprintf(ce, sizeof(ce), "B>=%d", eb);
            const char *verdict;
            const qwen_mm_gate_t *gk = agg_mmk[fmt][k] > 0 ? &g_mm_gate[agg_mmk[fmt][k]] : NULL;
            if (w1 <= 0 && wf <= 0)
                verdict = (gk && gk->on_env) ? "loses everywhere — opt-in, keep it OFF"
                                             : "loses everywhere — disable it";
            else if (w1 <= 0 && wf > 0)  verdict = "⚠️  WINS ONLY WITH THE THREAD POOL";
            else if (wf <= 0)            verdict = "wins single-threaded only (pool hurts it)";
            else                         verdict = "real win";
            if (agg_fired[fmt][k] == QWEN_MMK_Q4_BMATVEC ||
                agg_fired[fmt][k] == QWEN_MMK_FORCED_MATVEC)
                verdict = "== the reference by construction (no batched q4 kernel on this ISA)";
            char vbuf[128];
            if (agg_mmk[fmt][k] < 0) {
                snprintf(vbuf, sizeof(vbuf), "%s  [tail: no env gate; only QWEN_BATCH_FORCE_MATVEC=1]",
                         verdict);
                verdict = vbuf;
            }
            fprintf(f, "   %-26s %-9s %-9s %-9s  %s\n", agg_name[fmt][k], c1,
                    full > 1 ? cf : "(=j=1)", ce, verdict);
        }
    }
    fprintf(f,
        "\n   ⚠️  \"WINS ONLY WITH THE THREAD POOL\" means: at one real thread this kernel is\n"
        "       NOT faster than B separate matvecs, and its full-thread advantage comes from\n"
        "       amortizing %d pool launches over one call instead of B. That is not a kernel\n"
        "       win, it is an overhead being hidden — the fix is cheaper dispatch, not a\n"
        "       lower min_b. (This is exactly what the bf16 twin's \"1.70x\" turned out to be\n"
        "       on M1 once --matmat-bench stopped ignoring -j: 0.79x at one real thread.)\n", full);

    if (noise_n) {
        double navg = noise_sum / noise_n;
        fprintf(f, "\n   noise floor (the reference re-measured against itself): mean %.1f%%, "
                   "worst %.1f%% (%s)\n", 100.0 * navg, 100.0 * noise_max, noise_where);
        if (noise_max > (QTUNE_WIN - 1.0))
            fprintf(f, "   ⚠️  the worst cell's noise exceeds the %.0f%% win margin: on THIS box a\n"
                       "       single %.0f%% \"win\" is not resolvable. Trust the rows that win by a\n"
                       "       lot and at every B, re-run on a quiet machine, or raise the margin.\n",
                    100.0 * (QTUNE_WIN - 1.0), 100.0 * (QTUNE_WIN - 1.0));
    }

    fprintf(f, "\n══ CONFIGURATION FOR THIS BOX  (paste, or use tests/kernel_tune.sh's .env)\n");
    if (js) fprintf(js, "  \"recommend\": [\n");
    int jfirst = 1, printed = 0;
    for (int mmk = 1; mmk < QWEN_MMK_COUNT; mmk++) {
        const qwen_mm_gate_t *g = &g_mm_gate[mmk];
        if (g->max_b == 0) continue;
        const char *minb_env = qwen_mm_specific_minb_env(mmk);
        if (!minb_env) minb_env = g->minb_env;
        int fmt = -1, slot = -1;
        for (int fq = 0; fq < 3 && fmt < 0; fq++)
            for (int k = 0; k < agg_n[fq]; k++)
                if (agg_mmk[fq][k] == mmk) { fmt = fq; slot = k; break; }
        if (fmt < 0) continue;
        int wf = full > 1 ? agg_worst[fmt][slot][1] : agg_worst[fmt][slot][0];
        int w1 = agg_worst[fmt][slot][0];
        char line[160], why[200];
        if (wf <= 0) {
            if (g->on_env) {
                snprintf(line, sizeof(line), "# leave %s unset", g->on_env);
                snprintf(why, sizeof(why), "%s is opt-in and loses on every measured shape "
                         "— the default OFF is CONFIRMED, not assumed", agg_name[fmt][slot]);
            } else if (!g->off_env) {
                snprintf(line, sizeof(line), "# %s: loses, and has NO kill switch",
                         agg_name[fmt][slot]);
                snprintf(why, sizeof(why), "no off_env in g_mm_gate[] — add one if this matters");
            } else {
                snprintf(line, sizeof(line), "export %s=1", g->off_env);
                snprintf(why, sizeof(why), "%s never beats B x matvec on any measured shape",
                         agg_name[fmt][slot]);
            }
        } else if (g->on_env && minb_env) {
            snprintf(line, sizeof(line), "export %s=1 %s=%d", g->on_env, minb_env, wf);
            snprintf(why, sizeof(why), "%s is opt-in and DOES win from B>=%d here%s",
                     agg_name[fmt][slot], wf,
                     (w1 <= 0) ? "; POOL-ONLY WIN, see the warning above" : "");
        } else if (!minb_env) {
            snprintf(line, sizeof(line), "# %s: wins from B>=%d but has NO min_b env",
                     agg_name[fmt][slot], wf);
            snprintf(why, sizeof(why), "missing minb_env in g_mm_gate[]");
        } else {
            snprintf(line, sizeof(line), "export %s=%d", minb_env, wf);
            snprintf(why, sizeof(why), "%s wins from B>=%d (compiled default %d)%s",
                     agg_name[fmt][slot], wf, g->min_b,
                     (w1 <= 0) ? "; POOL-ONLY WIN, see the warning above" : "");
        }
        fprintf(f, "   %-34s # %s\n", line, why);
        printed++;
        if (js) {
            fprintf(js, "%s    {\"kernel\": \"%s\", \"mmk\": %d, \"line\": \"%s\", "
                        "\"crossover_j1\": %d, \"crossover_jfull\": %d, \"compiled_min_b\": %d}",
                    jfirst ? "" : ",\n", agg_name[fmt][slot], mmk, line, w1, wf, g->min_b);
            jfirst = 0;
        }
    }
    if (!printed) fprintf(f, "   (this binary dispatches no gated batched kernel at all)\n");
    if (js) fprintf(js, "\n  ],\n");

    fprintf(f, "\n══ THRESHOLD INVENTORY  (every guessed number in g_mm_gate[], and its override)\n");
    fprintf(f, "   %-24s %-24s %-22s %-8s %s\n", "kernel", "min_b env", "min_rows/min_cols env",
            "defaults", "off switch");
    for (int mmk = 1; mmk < QWEN_MMK_COUNT; mmk++) {
        const qwen_mm_gate_t *g = &g_mm_gate[mmk];
        if (g->max_b == 0) continue;
        const char *minb_env = qwen_mm_specific_minb_env(mmk);
        if (!minb_env) minb_env = g->minb_env;
        int avail = 0;
        for (int fq = 0; fq < 3 && !avail; fq++)
            for (int k = 0; k < agg_n[fq]; k++) if (agg_mmk[fq][k] == mmk) { avail = 1; break; }
        if (!avail) {
            char df0[24];
            snprintf(df0, sizeof(df0), "%d/%d/%d", g->min_b, g->min_rows, g->min_cols);
            fprintf(f, "   %-24s %-24s %-22s %-8s %s\n", g_mmk_info[mmk].name,
                    minb_env ? minb_env : "MISSING",
                    g->min_rows || g->min_cols ? "(see AMX rows below)" : "-", df0,
                    "NOT DISPATCHABLE HERE (ISA / kernel permission) — the box will answer");
            continue;
        }
        char rc[64];
        snprintf(rc, sizeof(rc), "%s / %s",
                 g->min_rows ? (g->minrows_env ? g->minrows_env : "MISSING") : "-",
                 g->min_cols ? (g->mincols_env ? g->mincols_env : "MISSING") : "-");
        char df[24];
        snprintf(df, sizeof(df), "%d/%d/%d", g->min_b, g->min_rows, g->min_cols);
        fprintf(f, "   %-24s %-24s %-22s %-8s %s\n", g_mmk_info[mmk].name,
                minb_env ? minb_env : "MISSING", rc, df,
                g->off_env ? g->off_env : (g->on_env ? g->on_env : "-"));
    }
    fprintf(f,
        "   max_b is deliberately NOT overridable: it bounds the float sx[16] activation-\n"
        "   scale array on the dispatcher's stack. Neither are qwen_amx_*_ready() and\n"
        "   cols %% %d — those are capabilities (SIGILL / wrong result), not thresholds.\n",
        Q4_0_BLOCK_SIZE);

    if (js) {
        fprintf(js, "  \"note\": \"crossover 0 = never wins; a kernel that wins only at "
                    "threads>1 is amortizing pool launches, not sharing weight reads\"\n}\n");
        fclose(js);
        fprintf(f, "\nJSON: %s\n", jpath);
    }
    return 0;
}

/* ---- in-region int8 matmat: the dispatched drivers, split into prep + per-thread run ----
 * Rule 6 of ENGINEERING.md: which implementation ran must be visible.  The in-region
 * runners bypass the dispatcher (and therefore the census), so each distinct backend
 * announces itself once per process the first time a region actually uses it. */
QWEN_MAYBE_UNUSED static void qwen_region_i8_note_backend(const char *what, int rows, int cols, int B) {
    static const char *seen[8]; static int nseen = 0;
    for (int i = 0; i < nseen; i++) if (seen[i] == what) return;
    if (nseen < 8) seen[nseen++] = what;
    fprintf(stderr, "[region] int8 in-region runner: %s (first use %dx%d B=%d)\n",
            what, rows, cols, B);
}


/* A shape is region-usable when SOME in-region row-block runner can execute it, not only
 * the VNNI one.  The AMX tiles are a valid runner too: qwen_region_i8_run below packs the
 * activations per thread and calls the same int8_amx_task the dispatched path calls, so the
 * row results are the ones the dispatcher would have produced.  Returning 0 for AMX shapes
 * (as this did) silently switched the CP/Talker regions and the batched heads off exactly
 * when batching got wide enough for AMX. */
int qwen_region_i8_usable(int rows, int cols, int B) {
#if defined(__AVX512VNNI__)
    if (B < 2 || B > 16 || rows < 256) return 0;
#if defined(__AMX_INT8__) && defined(__AMX_TILE__)
    if (qwen_mm_use(QWEN_MMK_INT8_AMX, B, rows, cols) && qwen_amx_int8_ready()) return 1;
#endif
    return qwen_mm_use(QWEN_MMK_INT8_VNNI, B, rows, cols);
#else
    (void)rows; (void)cols; (void)B; return 0;
#endif
}
int qwen_region_i8_qkv_usable(int q_rows, int kv_rows, int cols, int B) {
#if defined(__AVX512VNNI__) && defined(__x86_64__)
    if (qwen_x86_qkv_disabled() || B <= 1 || B > 16) return 0;
    if ((q_rows + 2 * kv_rows) < 256) return 0;
#if defined(__AMX_INT8__) && defined(__AMX_TILE__)
    if ((q_rows & 15) == 0 && (kv_rows & 15) == 0 &&
        qwen_amx_int8_qkv_allowed(B, q_rows, kv_rows, cols) && qwen_amx_int8_ready()) return 1;
#endif
    return qwen_mm_use(QWEN_MMK_INT8_VNNI, B, q_rows, cols) &&
           qwen_mm_use(QWEN_MMK_INT8_VNNI, B, kv_rows, cols);
#else
    (void)q_rows; (void)kv_rows; (void)cols; (void)B; return 0;
#endif
}
/* ISA-neutral on purpose: this is the same per-column quantiser the dispatched int8
 * matmat uses on every backend that has one, so a future ARM/other in-region runner needs
 * no second copy of it. */
float qwen_region_i8_quant_col(int8_t *qb, const float *Xt, int cols, int B, int b) {
    return quantize_act_int8_col(qb, Xt, cols, B, b);
}
void qwen_region_i8_run(float *Y, const int8_t *W, const float *scale, const int8_t *qXt,
                   const float *sx, int rows, int cols, int B, size_t tid, size_t nt) {
#if defined(__AVX512VNNI__)
#if defined(__AMX_INT8__) && defined(__AMX_TILE__)
    if (qwen_mm_use(QWEN_MMK_INT8_AMX, B, rows, cols) && qwen_amx_int8_ready()) {
        /* Every thread packs the same B x cols activation into its own scratch: O(B*cols)
         * against O(rows*cols/nt) of real work, and it keeps the runner allocation-free and
         * lock-free inside the region.  The weight pack cache is already mutex-guarded. */
        const size_t kfull = (size_t)(cols & ~63);
        int8_t *pXt = mm_scratch_pack(kfull * (size_t)B);
        if (pXt) {
            const uint8_t *pW = (const uint8_t *)qwen_amx_pack_weights(
                W, rows, cols, QWEN_AMX_WEIGHT_INT8);
            amx_pack_act_int8(pXt, qXt, cols, (int)kfull, B);
            int8_amx_ctx ac = { Y, W, pW, scale, pXt, qXt, sx, rows, cols, B };
            if (tid == 0) qwen_region_i8_note_backend("AMX int8 tiles", rows, cols, B);
            int8_amx_task(tid, nt, &ac);
            return;
        }
    }
#endif
    if (tid == 0) qwen_region_i8_note_backend("VNNI row blocks", rows, cols, B);
    if (tid == 0) qwen_region_pool_at2(QWEN_RGN_MM_REGION_I8, (int)nt, rows);
    {   /* Every thread that reaches the runner counts as an entry, and reports the rows it
         * owns: the split is by tid/nt, so a thread with r1 == r0 entered but had nothing to
         * do -- which is the difference between an idle worker and an unrecorded one. */
        int r0 = (int)(tid * (size_t)rows / nt), r1 = (int)((tid + 1) * (size_t)rows / nt);
        qwen_region_workers_at2(QWEN_RGN_MM_REGION_I8, 1);
        if (r1 > r0) qwen_region_units_at2(QWEN_RGN_MM_REGION_I8, r1 - r0); }
    int8_vmm_ctx c = { Y, W, scale, qXt, sx, rows, cols, B };
    int8_vmm_task(tid, nt, &c);
#else
    (void)Y; (void)W; (void)scale; (void)qXt; (void)sx; (void)rows; (void)cols; (void)B; (void)tid; (void)nt;
#endif
}
void qwen_region_i8_run_qkv(float *q, float *k, float *v,
                       const int8_t *Wq, const float *sq, const int8_t *Wk, const float *sk,
                       const int8_t *Wv, const float *sv, const int8_t *qXt, const float *sx,
                       int q_rows, int kv_rows, int cols, int B, size_t tid, size_t nt) {
#if defined(__AVX512VNNI__) && defined(__x86_64__)
    int8_qkv_mm_ctx c = {
        { q, k, v }, { Wq, Wk, Wv }, { NULL, NULL, NULL }, { sq, sk, sv },
        NULL, qXt, sx, { q_rows, kv_rows, kv_rows }, cols, B
    };
#if defined(__AMX_INT8__) && defined(__AMX_TILE__)
    if ((q_rows & 15) == 0 && (kv_rows & 15) == 0 &&
        qwen_amx_int8_qkv_allowed(B, q_rows, kv_rows, cols) && qwen_amx_int8_ready()) {
        const size_t kfull = (size_t)(cols & ~63);
        int8_t *pXt = mm_scratch_pack(kfull * (size_t)B);
        if (pXt) {
            c.pXt = pXt;
            c.pW[0] = (const uint8_t *)qwen_amx_pack_weights(Wq, q_rows,  cols, QWEN_AMX_WEIGHT_INT8);
            c.pW[1] = (const uint8_t *)qwen_amx_pack_weights(Wk, kv_rows, cols, QWEN_AMX_WEIGHT_INT8);
            c.pW[2] = (const uint8_t *)qwen_amx_pack_weights(Wv, kv_rows, cols, QWEN_AMX_WEIGHT_INT8);
            amx_pack_act_int8(pXt, qXt, cols, (int)kfull, B);
            if (tid == 0) qwen_region_i8_note_backend("AMX int8 tiles (fused QKV)", q_rows, cols, B);
            int8_qkv_amx_task(tid, nt, &c);
            return;
        }
    }
#endif
    if (tid == 0) qwen_region_i8_note_backend("VNNI row blocks (fused QKV)", q_rows, cols, B);
    int8_qkv_vnni_mm_task(tid, nt, &c);
#else
    (void)q; (void)k; (void)v; (void)Wq; (void)sq; (void)Wk; (void)sk; (void)Wv; (void)sv;
    (void)qXt; (void)sx; (void)q_rows; (void)kv_rows; (void)cols; (void)B; (void)tid; (void)nt;
#endif
}

/* ---- wide bf16 matmat for the prefill only (16 < B <= 64) ------------------------------
 * A separate entry point, so qwen_matmat_bf16 and every existing caller keep their exact
 * behaviour.  Same AVX-512 kernels, one weight-row sweep per call for all B columns. */
int qwen_matmat_bf16_wide_available(int rows, int cols) {
#if defined(__AVX512BF16__)
    return cols >= 32 && !qwen_bf16dot_disabled() && qwen_mm_use(QWEN_MMK_BF16_AVX512, 16, rows, cols);
#else
    (void)rows; (void)cols; return 0;
#endif
}
int qwen_matmat_bf16_wide(float *Y, const uint16_t *W, const float *X, int rows, int cols, int B) {
#if defined(__AVX512BF16__)
    if (B <= 16 || B > 64 || !qwen_matmat_bf16_wide_available(rows, cols)) return 0;
    uint16_t *Xb = mm_scratch_packb((size_t)B * cols);
    if (!Xb) return 0;
    for (int b = 0; b < B; b++)
        for (int k = 0; k < cols; k++)
            Xb[(size_t)b * cols + k] = qwen_f32_to_bf16_scalar(X[(size_t)k * B + b]);
    bf16_avx512_ctx c = { Y, W, Xb, rows, cols, B };
    int nt = g_n_threads;
    if (nt > 1 && rows >= 256) qwen_parallel((size_t)nt, bf16_avx512_task, &c);
    else bf16_avx512_task(0, 1, &c);
    return 1;
#else
    (void)Y; (void)W; (void)X; (void)rows; (void)cols; (void)B; return 0;
#endif
}
