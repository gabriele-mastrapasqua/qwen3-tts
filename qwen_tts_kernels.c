/*
 * qwen_tts_kernels.c - Kernel implementations
 */

#include <pthread.h>
#include "qwen_tts_kernels.h"
#include "qwen_tts_kleidi.h"
#include "qwen_tts_q8repack.h"

/* Defined here, above every kernel, because the GEMV counters live in matvec
 * entry points that appear earlier in this file than the batched ones. */
#define MMSTAT(k, r, c, b) do {                                                            \
        if (qwen_matmat_stats_enabled() || qwen_census_enabled())                          \
            qwen_matmat_stats_note((k), (long long)(r) * (long long)(c) * (long long)(b)); \
    } while (0)

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
#if defined(__x86_64__)
#include <cpuid.h>      /* __get_cpuid_count: runtime AMX probe. clang rejects
                         * __builtin_cpu_supports("amx-int8") (gcc-only string), so the
                         * AMX presence test is raw CPUID and works on both compilers. */
#endif
#if (defined(__AMX_INT8__) || defined(__AMX_BF16__)) && defined(__AMX_TILE__) && defined(__linux__)
#include <sys/syscall.h>   /* SYS_arch_prctl: XTILEDATA permission request (see below) */
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

static int g_n_threads_hard = 0;   /* pool size actually spawned (see qwen_set_threads_soft) */

void qwen_set_threads(int n) {
    g_n_threads = n > 0 ? n : 1;
    g_n_threads_hard = g_n_threads;
    qwen_ftz_on();
    qwen_threadpool_start(g_n_threads);  /* (re)size the off-Mac worker pool */
    qwen_blas_set_threads(g_n_threads);
}
int qwen_get_threads(void) { return g_n_threads; }

/* Per-stage thread budget WITHOUT touching the pool — see the header for why the
 * obvious qwen_set_threads() is the wrong tool here (join + respawn per frame off
 * macOS). BLAS is retargeted too: it is the other consumer of the same cores, and
 * leaving it at the hard budget while the stage runs narrow is exactly the "two pools
 * fighting" that qwen_blas_set_threads exists to avoid. */
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
    /* 4 threads is the sweet spot for bf16 matvec (memory-bandwidth-bound).
     * More threads add GCD dispatch overhead without bandwidth gain. */
    g_n_threads = ncpus < 4 ? ncpus : 4;
    g_n_threads_hard = g_n_threads;
    qwen_ftz_on();  /* main thread: flush denormals (int8 activations) */
    qwen_threadpool_start(g_n_threads);  /* spawn the off-Mac persistent pool */
    qwen_blas_set_threads(g_n_threads);  /* else BLAS grabs all ncpus (see below) */
}

/* ── AMX (Intel Sapphire/Emerald Rapids) runtime availability ────────────────────
 *
 * WHY A RUNTIME GATE AT ALL, when every other kernel here is a plain #ifdef: AMX is
 * the one x86 extension whose instructions fault even on a CPU that HAS it. The tile
 * registers are 8 KB of extended XSAVE state that Linux does NOT hand out by default —
 * a process must ask once with arch_prctl(ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA)
 * (kernel >= 5.16). Skip that and the very first _tile_loadd is SIGILL, i.e. a crash
 * on the user's box rather than a slow path. So: compiled-in is necessary and NOT
 * sufficient, and the dispatcher must be able to fall back to VNNI at runtime.
 *
 * The request is per PROCESS (the kernel grants it for every thread, present and
 * future), so asking once from whichever thread gets here first is enough — and the
 * dispatcher calls this BEFORE handing slices to the pool, so no worker can issue a
 * tile op before the permission exists. Two threads racing here both perform the same
 * idempotent syscall and store the same answer; that is benign, not a data race.
 *
 * Silent by design: no printing on failure, because this is called per matmat. What
 * happened is visible once, on demand, in `--caps`.
 *
 * ⚠️ WRITTEN ON M1, NEVER EXECUTED. The Linux branch is not even syntax-checked by
 * `make check-isa` (the cross-compile targets x86_64-apple-macos, where __linux__ is
 * undefined) — it is the first thing to verify on the rented c3/c4 box. */
#if defined(__x86_64__)
static int qwen_x86_has_amx_int8(void) {
    unsigned eax, ebx, ecx, edx;
    if (!__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) return 0;
    /* CPUID.(EAX=07H,ECX=0):EDX bit 24 = AMX-TILE, bit 25 = AMX-INT8. Both are needed:
     * TILE for the config/load/store ops, INT8 for _tile_dpbssd itself. */
    return ((edx >> 24) & 1u) && ((edx >> 25) & 1u);
}
/* Same register, bit 22 = AMX-BF16 (_tile_dpbf16ps). Separate from AMX-INT8 because
 * they are separate CPUID bits: a part can in principle expose one and not the other,
 * and the bf16 GEMM must not ride on the int8 probe. */
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
#define SYS_arch_prctl 158        /* x86-64 syscall number; older headers omit the macro */
#endif
/* The XTILEDATA grant is per PROCESS and covers every tile instruction, int8 and bf16
 * alike — so it is asked ONCE here and the per-format probes below only add their own
 * CPUID bit on top. Splitting it per format would ask the kernel the same question
 * twice and, worse, would let one format's failure look like the other's. */
static int qwen_amx_perm_ok(void) {
    static atomic_int perm_state = -1;
    int st = atomic_load_explicit(&perm_state, memory_order_relaxed);
    if (st >= 0) return st;
    st = 0;
#if defined(__linux__)
    if (syscall(SYS_arch_prctl, QWEN_ARCH_REQ_XCOMP_PERM, QWEN_XFEATURE_XTILEDATA) == 0) {
        /* Ask back what we actually got: REQ can succeed on a kernel that then hands
         * out nothing, and believing it would put the SIGILL right back. */
        unsigned long bits = 0;
        if (syscall(SYS_arch_prctl, QWEN_ARCH_GET_XCOMP_PERM, &bits) == 0 &&
            (bits & (1UL << QWEN_XFEATURE_XTILEDATA)) != 0)
            st = 1;
    }
#else
    /* No XTILEDATA protocol outside Linux (macOS/x86 has no AMX silicon at all,
     * and Windows uses a different opt-in): report unusable rather than fault. */
    st = 0;
#endif
    atomic_store_explicit(&perm_state, st, memory_order_relaxed);
    return st;
}
#endif /* (__AMX_INT8__ || __AMX_BF16__) && __AMX_TILE__ */

#if defined(__AMX_INT8__) && defined(__AMX_TILE__)
/* 1 once the CPU has AMX-INT8 *and* the kernel has granted XTILEDATA to this process. */
static int qwen_amx_int8_ready(void) {
    static atomic_int amx_state = -1;
    int st = atomic_load_explicit(&amx_state, memory_order_relaxed);
    if (st >= 0) return st;
    st = qwen_x86_has_amx_int8() && qwen_amx_perm_ok();
    atomic_store_explicit(&amx_state, st, memory_order_relaxed);
    return st;
}
#endif /* __AMX_INT8__ && __AMX_TILE__ */

#if defined(__AMX_BF16__) && defined(__AMX_TILE__)
/* Same, for _tile_dpbf16ps. Kept separate so `--caps` can report the two states
 * independently: on the box "AMX active" is not one fact but two. */
static int qwen_amx_bf16_ready(void) {
    static atomic_int amx_state = -1;
    int st = atomic_load_explicit(&amx_state, memory_order_relaxed);
    if (st >= 0) return st;
    st = qwen_x86_has_amx_bf16() && qwen_amx_perm_ok();
    atomic_store_explicit(&amx_state, st, memory_order_relaxed);
    return st;
}
#endif /* __AMX_BF16__ && __AMX_TILE__ */

/* Public: does THIS box have a usable AMX bf16 tile unit (compiled in, present at
 * runtime, and XTILEDATA granted)? The prefill uses it to pick its GEMM: measured on
 * the c3 (0.6B, server, TTFA trace), routing the prefill projections through
 * qwen_matmat_bf16 instead of the bf16->f32 convert + sgemm takes the prefill from
 * 330 ms to 113 ms and the whole TTFA from 623 ms to 300 ms. The same switch was
 * measured NEUTRAL on an EPYC (avx512bf16 dot, no tile unit), which is why the default
 * is tied to the tile unit rather than to x86. */
int qwen_amx_bf16_available(void) {
#if defined(__AMX_BF16__) && defined(__AMX_TILE__)
    return qwen_amx_bf16_ready();
#else
    return 0;
#endif
}

/* Il gemello ARM del predicato qui sopra: questa build ha un vero GEMM bf16
 * matrice-matrice (BFMMLA, i8mm/bf16 di Armv8.6 — Neoverse-V1/V2, Graviton3+, M2+)?
 *
 * Serve al prefill esattamente come l'AMX serve su x86, e la ragione per cui il default
 * puo' dipendere da questo e non da "siamo su ARM" e' che senza l'istruzione il gemello
 * batchato PERDE contro B x matvec (misurato su M1 e su M4: 0,72-0,91x).
 *
 * ⚠️ Fino al 2026-08-21 questo predicato non sarebbe bastato lo stesso: il kernel BFMMLA
 * era un tile 2x2 con UN accumulatore e sull'Axion misurava 0,73-1,02x, cioe' perdeva pur
 * avendo l'istruzione. Con il blocco 4x4 misura 2,75-3,92x, e il prefill matmat batte BLAS
 * a OGNI budget di thread (j1..j16). Il predicato descrive il silicio; il kernel deve
 * essere all'altezza, e le due cose vanno verificate separatamente. */
int qwen_arm_bf16_matmat_available(void) {
#if defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC) && !defined(__APPLE__)
    const char *e = getenv("QWEN_NO_BFMMLA");
    return !(e && e[0] == '1');
#else
    return 0;   /* Apple: i gemelli MMLA sono in perdita e restano opt-in (QWEN_APPLE_MMLA) */
#endif
}

/* Report ACTUAL compiled capabilities (mirrors the kernels' own #ifdef guards).
 * Run `./qwen_tts --caps`. Makes the real SIMD/threading state visible + testable so
 * a false "we have AVX2/threading" claim can't survive — the binary tells the truth. */
#ifndef QWEN_GIT_REV
#define QWEN_GIT_REV "unknown"
#endif
#ifndef QWEN_SIMD_PROFILE
#define QWEN_SIMD_PROFILE "unknown"
#endif

/* Le variabili che cambiano i NUMERI o l'AUDIO, non ogni QWEN_* esistente: un elenco
 * che include il rumore non viene letto. Registro completo the design notes */
static const char *const g_qwen_reported_flags[] = {
    "QWEN_SD_INT8", "QWEN_PREFILL_MATMAT", "QWEN_PREFILL_QUANT", "QWEN_DECODER_BATCH",
    "QWEN_DECODER_THREAD", "QWEN_BATCH_NO_SOLO", "QWEN_BATCH_NO_BEFF", "QWEN_BATCH_NOMATMUL",
    "QWEN_NO_AMX", "QWEN_NO_VNNI", "QWEN_NO_SDOT", "QWEN_NO_BF16DOT",
    "QWEN_STREAM_DECODE_CHUNK", "QWEN_STREAM_DECODE_CHUNK_BUSY", "QWEN_TTFA_PRIORITY",
    "QWEN_SERVE_BLAS", "QWEN_SERVE_BLAS_BUSY", "QWEN_DECODER_GANG_LEAD", "QWEN_DECODER_GANG_MIN",
    "QWEN_AMX_MIN_B", "QWEN_VNNI_MIN_B", "QWEN_BATCH_STATS", "QWEN_THP", "QWEN_POOL_SPIN",
    "QWEN_ARM_BFDOT", NULL
};

/* Stampa provenienza + flag attive. Sta in cima al report perche' e' la prima cosa da
 * incollare in un confronto: senza di essa due tabelle di numeri non sono confrontabili
 * e nessuno se ne accorge finche' non e' troppo tardi. */
void qwen_provenance_report(void *out) {
    FILE *f = out ? (FILE *)out : stderr;
    fprintf(f, "  build:            %s · SIMD=%s · %s %s\n",
            QWEN_GIT_REV, QWEN_SIMD_PROFILE, __DATE__, __TIME__);
    int n = 0;
    for (int i = 0; g_qwen_reported_flags[i]; i++) {
        const char *v = getenv(g_qwen_reported_flags[i]);
        if (!v) continue;
        if (n == 0) fprintf(f, "  flag attive:      ");
        else if (n % 3 == 0) fprintf(f, "\n                    ");
        fprintf(f, "%s=%s  ", g_qwen_reported_flags[i], v);
        n++;
    }
    if (n == 0) fprintf(f, "  flag attive:      nessuna (tutti i default di questa build)\n");
    else fprintf(f, "\n");
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
    /* Hot path: bf16/int8/q4 matvecs + attention (~90%% of decode). Both NEON and
     * AVX2 are full 2-row, multi-accumulator, prefetching kernels (PLAN 21.3). */
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
#if defined(__AVX512BF16__)
    fprintf(f, "  bf16 dot:         VDPBF16PS _mm512_dpbf16_ps (native; QWEN_NO_BF16DOT=1 disables)\n");
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
    /* AMX is the only x86 matrix-MATRIX integer unit (VNNI is still a dot product), so
     * it gets its own line: "compiled" and "usable" are different states here, and only
     * the second one changes a number. */
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
    /* bf16 gets its own line, and it is NOT redundant: bf16 is the only format where a
     * batched twin has ever beaten B x matvec on any box we own, and the codec head
     * (vocab x hidden) is bf16 and unquantized, so it crosses this kernel once per frame
     * per slot. If this line does not say ACTIVE on the c3, the bf16 batching numbers
     * that follow are AVX-512 numbers. */
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
            has_bf16 ? "bf16 BFMMLA -> native GEMM batched matmat twins (Graviton3-measured: int8 batch 2.1x, bf16 1.5x)"
                     : "no bf16 matmul (M1-class) -> batched matmat uses scalar bf16 decode");
    if (!has_bf16 && !has_i8mm)
        fprintf(f, "  note:             M1-class (Armv8.5, dotprod only). M2/M3/M4/M5 add bf16+i8mm -> the native-matmul lever.\n");
    /* Il gemello ARM dell'avviso AMX qui sopra. Senza, una build priva di +i8mm/+bf16 su
     * una CPU che li ha semplicemente OMETTE le righe "ACTIVE": tace, non avverte, e il
     * report sembra completo. E' la stessa trappola del "binario mutilato" x86 (successo
     * il 2026-08-20 su Emerald), che su ARM non aveva ancora un canarino. */
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
 * 4 accumulators (32 elem/iter) so the FMA reduction isn't latency-bound.
 * The `_avx2` names are historical: under __AVX512F__ each helper carries a
 * 16-wide body (head_dim=128 → whole rows per iter), call sites unchanged. */
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
/* q·(bf16 k) dot product, AVX2/FMA with scalar tail (bf16-KV attention score). */
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
/* Attention online-softmax accumulators (AVX2 names, 16-wide under __AVX512F__). */
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
/* ── C4 (plan_v4): bf16 matvec via VDPBF16PS (_mm512_dpbf16_ps) ──
 * Fuses the bf16→f32 widen INTO the FMA: 32 bf16 pairs per instruction (2× the
 * width of the shift+FMA __AVX512F__ path below) and no shift/convert uops on
 * the weight stream. The ACTIVATION is rounded to bf16 once per call (same
 * numeric class as the ARM BFMMLA batched path: ~1e-3 L2 vs the f32-activation
 * reference — activation-rounding, not a defect; weights are bf16 anyway).
 * Runtime kill-switch: QWEN_NO_BF16DOT=1 falls back to the widen+FMA path so
 * the box can A/B speed AND mel-corr without a rebuild.
 * ⚠️ COMPILE-CHECKED ONLY on M1 (make check-isa) — validate on real
 * avx512_bf16 silicon (Zen4/5, Cooper Lake+) before trusting. */
enum { QWEN_BF16DOT_XMAX = 8192 };
static int qwen_bf16dot_disabled(void) {
    static atomic_int off = -1;
    int v = atomic_load_explicit(&off, memory_order_relaxed);
    if (v < 0) { const char *e = getenv("QWEN_NO_BF16DOT"); v = (e && e[0] == '1'); atomic_store_explicit(&off, v, memory_order_relaxed); }
    return v;
}
/* gcc/clang lack _mm512_castsi512_pbh-style casts across versions → union punning. */
static inline __m512bh qwen_loadu_pbh(const uint16_t *p) {
    union { __m512i i; __m512bh bh; } u;
    u.i = _mm512_loadu_si512((const void *)p);
    return u.bh;
}
/* Round a f32 row to bf16 (RNE, matches _mm512_cvtneps_pbh) with scalar tail. */
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
        for (; k + 64 <= in_dim; k += 64) {          /* 64 bf16 = 2 dpbf16/row */
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
#endif /* __AVX512BF16__ */

/* Fused bf16 matvec: processes 2 output rows at a time to amortize x vector loads.
 * On NEON: 32 elements/iter, 8 accumulators per row pair (from qwen-asr). */
#if defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC)
/* ── BFDOT: il dot bf16 nativo ARM per il matvec (2026-08-21, ARM epic A4) ──
 *
 * Il gemello x86 di questo kernel esiste da mesi (bf16_matvec_dpbf16, VDPBF16PS). Su ARM
 * non c'era: `vbfdotq_f32` non compariva NEMMENO UNA VOLTA nel repo, e il matvec bf16
 * allargava bf16->f32 e poi faceva FMA su una CPU che il dot bf16 ce l'ha.
 *
 * Dove si vede: la TESTA CODEC (vocab x hidden, bf16 e NON quantizzata) attraversa questo
 * kernel una volta per frame per stream, piu' ogni matvec bf16 quando non si quantizza.
 *
 * ⚠️ NON e' identico numericamente, ed e' per questo che sta dietro una flag e sta OFF di
 * default: BFDOT tronca l'ATTIVAZIONE a bf16 (i pesi lo sono gia'). E' la stessa troncatura
 * che subiscono la KV cache e il gemello BFMMLA, quindi e' una perdita nota e gia' accettata
 * altrove — ma su un finetune l'unica prova che vale e' la language identity, non il self-test.
 * QWEN_ARM_BFDOT=1 turns it on. Before making it a default: run a language-identity
 * gate on finetuned checkpoints, where the numerics matter most. */
enum { QWEN_ARM_BFDOT_XMAX = 8192 };
/* Troncatura, non arrotondamento: e' esattamente cio' che fa il gemello BFMMLA
 * ("truncate, like the KV") piu' sotto. Due percorsi bf16 che arrotondano in modo diverso
 * darebbero due audio diversi a seconda di B, che e' il tipo di differenza che poi si
 * insegue per giorni. */
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
static void bf16_matvec_bfdot(float *y, const uint16_t *xb, const uint16_t *W,
                              int in_dim, int out_dim) {
    const bfloat16_t *xv = (const bfloat16_t *)xb;
    int o = 0;
    for (; o + 3 < out_dim; o += 4) {          /* quattro catene, una lettura di attivazione */
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
        for (; k < in_dim; k++) {              /* coda k (in_dim%8; le dim del modello non la toccano) */
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
#endif /* __ARM_FEATURE_BF16_VECTOR_ARITHMETIC */

static void bf16_matvec_fused(float *y, const float *x, const uint16_t *W,
                               int in_dim, int out_dim) {
    int o = 0;
#if defined(__AVX512BF16__)
    /* C4: native bf16 dot when available (QWEN_NO_BF16DOT=1 opts out). The
     * activation is rounded to bf16 once per call; threads each convert their
     * own copy (in_dim ≤ 3072 in practice → negligible vs the row work). */
    if (!qwen_bf16dot_disabled() && in_dim <= QWEN_BF16DOT_XMAX) {
        uint16_t xb[QWEN_BF16DOT_XMAX];
        qwen_f32_to_bf16_row(xb, x, in_dim);
        bf16_matvec_dpbf16(y, xb, x, W, in_dim, out_dim);
        return;
    }
#endif
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
#  if defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC)
    /* BFDOT nativo, opt-in (vedi il commento sul kernel: tronca l'attivazione a bf16). */
    if (qwen_arm_bfdot_enabled() && in_dim <= QWEN_ARM_BFDOT_XMAX) {
        uint16_t xb[QWEN_ARM_BFDOT_XMAX];
        qwen_arm_f32_to_bf16_row(xb, x, in_dim);
        bf16_matvec_bfdot(y, xb, W, in_dim, out_dim);
        return;
    }
#  endif
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


/* ── KleidiAI drop-in helpers ────────────────────────────────────────────────────
 * Order matters: try the matmul FIRST so the steady state costs one registry lookup,
 * and only register on a miss. Registration is normally done eagerly at load
 * (qwen_kleidi_prepack); this covers the matrices that appear later - --expr,
 * --wdelta and any override that rebuilds a weight after the model is up. */
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
    qwen_census_op("matvec_bf16", rows, cols, 1);
    /* If this exact matrix came from a GGUF Q8_0, its per-block-32 form is registered
     * under this very pointer and runs on the repacked ARM path instead. Hooking the
     * bf16 entry rather than adding a `_q8` slot keeps every dispatcher untouched: the
     * loader already replaced the bf16 pointer, so decode arrives here by itself. */
    if (qwen_q8r_matmul(y, (const void *)W, x, rows, cols, 1)) {
        MMSTAT(QWEN_MMK_Q8_REPACK_GEMV, rows, cols, 1);
        return;
    }
    if (kai_bf16_try(y, W, x, rows, cols, 1)) {
        MMSTAT(QWEN_MMK_KLEIDI_BF16_GEMV, rows, cols, 1);
        return;
    }
    MMSTAT(QWEN_MMK_BF16_GEMV, rows, cols, 1);   /* audit symmetry: GEMV counts too */

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
/* ══ Batched-path audit (PLAN 0.nonies S10) ═══════════════════════════════════
 *
 * The question this answers is not "how fast", it is "WHICH KERNEL RAN". A B=1..8
 * scaling curve is only meaningful if the projections actually went through a
 * matrix-matrix kernel; where the dispatcher falls back to B sequential matvecs the
 * curve measures the fallback. That fallback is not hypothetical — it is what M1-class
 * ARM does for int8 and q4, and what every AVX2 box did until 2026-08-18.
 *
 * Counted in MACs (rows*cols*B), not calls: one lm_head call dwarfs a hundred small
 * projections, so counting calls would make a fallback on the biggest matrix look like
 * a rounding error. Relaxed atomics — this is a counter, not a synchronisation point. */
static atomic_int g_mm_stats = -1;
static atomic_llong g_mm_macs[QWEN_MMK_COUNT];
static atomic_llong g_mm_calls[QWEN_MMK_COUNT];
/* Weight traffic (PLAN S16.4) + the window over which it was accumulated. Separate
 * from the MAC counters on purpose: MACs say how much ARITHMETIC was asked for, these
 * say how much MEMORY it cost — and the whole batching thesis is that the second can
 * stay flat while the first grows. */
static atomic_llong g_mm_wbytes;
static _Atomic double g_mm_wb_t0, g_mm_wb_t1;
static double qwen_mm_now_s(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

/* Three classes, because "batched" and "a real GEMM" are NOT the same thing and
 * conflating them hides the interesting case:
 *   GEMM  a matrix-matrix INSTRUCTION does the work (AMX tiles, VNNI vpdpbusd,
 *         AVX2 maddubs, ARM SMMLA/BFMMLA). This is what the batching thesis rests on.
 *   TWIN  a batched kernel with no matrix instruction: the weight block is read once
 *         and reused across B columns with ordinary vector ops. It shares the weight
 *         read — but MEASURED on M1 at one thread it is 0.34x vs B matvecs for int8,
 *         so sharing the read is not sufficient. Worth its own class precisely because
 *         it looks like batching and does not behave like it.
 *   MATVEC literally B sequential matvecs. Weights read B times. No sharing at all.
 * SOLO is the single-slot shortcut: by design, and neither a win nor a defect. */
enum { MMC_GEMM = 0, MMC_TWIN, MMC_MATVEC, MMC_SOLO, MMC_GEMV, MMC_NCLS };
static const struct { const char *name; int cls; } g_mmk_info[QWEN_MMK_COUNT] = {
    { "(none)",                 MMC_TWIN   },
    { "bf16 BFMMLA (arm)",      MMC_GEMM   },
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
/* Positional array vs a header enum is exactly the pairing that rots silently: add an id
 * and forget the row and every kernel after it gets mislabelled in the audit — which is
 * worse than no audit, because the report still looks right. Catch it at compile time. */
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
        /* register EXACTLY once: this is read from every worker thread, and a plain
         * "if (v) atexit(...)" printed the report three times in the first live test. */
        static atomic_int reg = 0;
        if (v && !atomic_exchange_explicit(&reg, 1, memory_order_relaxed))
            atexit(qwen_mm_stats_atexit);
        atomic_store_explicit(&g_mm_stats, v, memory_order_relaxed);
    }
    return v;
}
/* Component attribution. A plain __thread int: the kernels are always entered from the
 * thread that set it (the pool workers run INSIDE a kernel, past this point), so no
 * propagation into workers is needed and none is faked. */
/* Stored in the THREAD LAYER's tag, not in a TLS of our own, because the kernels book
 * their counters from inside qwen_parallel workers: a private TLS made an 8-thread run
 * file 7/8 of its work under "other" (measured: 336 calls attributed, 2352 lost). */
void qwen_mm_component(int comp) {
    qwen_tls_tag_set((comp >= 0 && comp < QWEN_COMP_COUNT) ? comp : QWEN_COMP_OTHER);
}
int qwen_mm_component_get(void) { return qwen_tls_tag_get(); }

static atomic_llong g_mm_macs_c[QWEN_COMP_COUNT][QWEN_MMK_COUNT];
static atomic_llong g_mm_calls_c[QWEN_COMP_COUNT][QWEN_MMK_COUNT];

/* ── SHAPE CENSUS ────────────────────────────────────────────────────────────────
 * See the contract in qwen_tts_kernels.h. Recorded at the dispatcher entry, on the
 * calling thread, so (rows, cols, B) is the LOGICAL shape and not a thread's slice. */
#define QWEN_CENSUS_MAX 256
typedef struct {
    const char *entry;          /* dispatcher name - a string literal, never freed */
    int comp, rows, cols, B;
    atomic_llong calls, macs;
    atomic_uint  kmask;         /* which QWEN_MMK_* fired under this shape */
} qwen_census_row_t;
static qwen_census_row_t g_census[QWEN_CENSUS_MAX];
static atomic_int   g_census_n;
static atomic_int   g_census_on = -1;
static atomic_llong g_census_frames;
static _Atomic(qwen_census_row_t *) g_census_cur;
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

/* Per-CALL-SITE frame counts. One number could not say WHICH generation loop ran, and
 * on the batched server that is the whole question: a census divided by the frames of a
 * loop that did not run is not a rate. Sites: 0 single-request, 1 batched, 2 batched
 * (per-slot variant). */
static atomic_llong g_census_frames_at[3];
void qwen_census_frame_at(int site) {
    if (!qwen_census_enabled()) return;
    if (site >= 0 && site < 3)
        atomic_fetch_add_explicit(&g_census_frames_at[site], 1, memory_order_relaxed);
    atomic_fetch_add_explicit(&g_census_frames, 1, memory_order_relaxed);
}
void qwen_census_frame(void) { qwen_census_frame_at(0); }

void qwen_census_op(const char *entry, int rows, int cols, int B) {
    if (!qwen_census_enabled() || B <= 0) return;
    const int comp = qwen_tls_tag_get();
    const int n = atomic_load_explicit(&g_census_n, memory_order_acquire);
    qwen_census_row_t *hit = NULL;
    for (int i = 0; i < n; i++) {
        qwen_census_row_t *r = &g_census[i];
        if (r->rows == rows && r->cols == cols && r->B == B &&
            r->comp == comp && r->entry == entry) { hit = r; break; }
    }
    if (!hit) {
        /* Insert is rare (once per distinct shape) - a mutex is the right cost here,
         * and it also makes the row's immutable fields safe to read lock-free above. */
        pthread_mutex_lock(&g_census_mu);
        int m = atomic_load_explicit(&g_census_n, memory_order_relaxed);
        for (int i = n; i < m && !hit; i++) {
            qwen_census_row_t *r = &g_census[i];
            if (r->rows == rows && r->cols == cols && r->B == B &&
                r->comp == comp && r->entry == entry) hit = r;
        }
        if (!hit) {
            if (m >= QWEN_CENSUS_MAX) {
                atomic_fetch_add_explicit(&g_census_overflow, 1, memory_order_relaxed);
                pthread_mutex_unlock(&g_census_mu);
                return;
            }
            hit = &g_census[m];
            hit->entry = entry; hit->comp = comp;
            hit->rows = rows; hit->cols = cols; hit->B = B;
            atomic_store_explicit(&g_census_n, m + 1, memory_order_release);
        }
        pthread_mutex_unlock(&g_census_mu);
    }
    atomic_fetch_add_explicit(&hit->calls, 1, memory_order_relaxed);
    atomic_fetch_add_explicit(&hit->macs,
                              (long long)rows * (long long)cols * (long long)B,
                              memory_order_relaxed);
    atomic_store_explicit(&g_census_cur, hit, memory_order_relaxed);
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
    fprintf(f, "# csv: comp,entry,N,K,B,calls,calls_per_frame,gmac,gmac_per_frame,kernels\n");
    for (int i = 0; i < n; i++) {
        qwen_census_row_t *r = &g_census[i];
        long long c = atomic_load_explicit(&r->calls, memory_order_relaxed);
        long long m = atomic_load_explicit(&r->macs,  memory_order_relaxed);
        unsigned km = atomic_load_explicit(&r->kmask, memory_order_relaxed);
        char kbuf[256]; kbuf[0] = 0;
        for (int k = 1; k < QWEN_MMK_COUNT; k++) {
            if (!(km & (1u << k))) continue;
            if (kbuf[0]) strncat(kbuf, "+", sizeof kbuf - strlen(kbuf) - 1);
            strncat(kbuf, g_mmk_info[k].name, sizeof kbuf - strlen(kbuf) - 1);
        }
        fprintf(f, "census,%s,%s,%d,%d,%d,%lld,%.3f,%.4f,%.6f,%s\n",
                cname[r->comp < 0 || r->comp >= QWEN_COMP_COUNT ? 0 : r->comp], r->entry,
                r->rows, r->cols, r->B, c,
                frames ? (double)c / (double)frames : 0.0,
                (double)m / 1e9,
                frames ? (double)m / 1e9 / (double)frames : 0.0,
                kbuf[0] ? kbuf : "(none)");
    }
    int ov = atomic_load_explicit(&g_census_overflow, memory_order_relaxed);
    if (ov) fprintf(f, "[shape-census] WARNING: %d ops dropped, table full (%d rows)\n",
                    ov, QWEN_CENSUS_MAX);
    fflush(f);
}

void qwen_matmat_stats_note(int k, long long macs) {
    if (k <= 0 || k >= QWEN_MMK_COUNT) return;
    /* Which kernel actually served the shape recorded at the dispatcher entry. OR, not
     * store, so an interleaving under concurrency over-includes rather than lies. */
    if (atomic_load_explicit(&g_census_on, memory_order_relaxed) > 0) {
        qwen_census_row_t *cur = atomic_load_explicit(&g_census_cur, memory_order_relaxed);
        if (cur) atomic_fetch_or_explicit(&cur->kmask, 1u << k, memory_order_relaxed);
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
    /* First byte counted starts the clock: dividing the total by the WHOLE process
     * lifetime would fold model loading and idle waiting into a bandwidth figure. */
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
    /* Per component. This is the table that answers "is the Code Predictor actually
     * on the new path, or still on the old one" - a global percentage cannot. */
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
    fprintf(f, "  matrix-matrix %5.1f%%  ·  batched twin %5.1f%%  ·  B x matvec %5.1f%%  ·  single-slot %5.1f%%  ·  GEMV %5.1f%%\n",
            100.0 * (double)by_cls[MMC_GEMM]   / (double)tot,
            100.0 * (double)by_cls[MMC_TWIN]   / (double)tot,
            100.0 * (double)by_cls[MMC_MATVEC] / (double)tot,
            100.0 * (double)by_cls[MMC_SOLO]   / (double)tot,
            100.0 * (double)by_cls[MMC_GEMV]   / (double)tot);
    if (by_cls[MMC_GEMM] * 2 < tot && by_cls[MMC_GEMV] * 2 < tot)
        fprintf(f, "  ⚠️  less than half the work went through a real matrix-matrix instruction:\n"
                   "      a B=1..8 curve measured here describes the FALLBACK, not this silicon.\n");
    /* Weight traffic — the input to BW_utilization. Printed as GB and GB/s with the
     * window it was measured over, so the bench does not have to guess the denominator. */
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

/* ══ Shape-aware kernel gate (PLAN 0.nonies, "quant miste con scelta del kernel in
 *    funzione di ISA *e* shape") ══════════════════════════════════════════════════
 *
 * THE PROBLEM. Until now every batched dispatcher below picked its kernel from
 * COMPILE-TIME guards alone: if the binary has VNNI, VNNI runs — at B=1 and at B=8,
 * on a 3072x2048 codec head and on a 320-column CP projection. But the right kernel
 * is a function of (ISA, B, rows, cols), not of the ISA alone:
 *
 *  - AMX is priced by M, not by N. One _tile_dpbssd costs what it costs for 16 weight
 *    rows x K=64 whatever the column count, and with the ACTIVATIONS as the packed
 *    operand the column count IS B. At B=2 the 16x16 C tile carries 32 of 256 useful
 *    int32 (12.5%) and the same instruction does 1/8 of the MACs it could — VNNI,
 *    which is priced per 64 MACs, does not care about B at all. So there is a crossover
 *    in B below which the smaller primitive wins, and it is a MEASUREMENT.
 *  - rows/cols too small never repay the fixed cost: the TILECFG, the activation
 *    repack, and (for anything under one tile edge) a scalar tail that does all the
 *    work anyway.
 *  - and the thresholds are hardware-specific, so they must be overridable WITHOUT a
 *    rebuild: on the box we want the sweep, not a recompile per cell.
 *
 * THE MEASUREMENT THAT MAKES THIS URGENT (M1, 2026-08-18, after `--matmat-bench` was
 * fixed to honour -j): at ONE REAL THREAD the batched twins are
 *      bf16 0.79x   int8 0.34x   int4 0.98x   vs B x matvec.
 * The 1.70x that bf16 used to show was a 4-thread number, i.e. it was amortizing pool
 * launches, not sharing weight reads. So "batched" is not automatically better even
 * when the kernel is correct, and a dispatcher that cannot be told "not here" is a
 * dispatcher that cannot be tuned. Every threshold below is therefore a DECLARED
 * GUESS to be re-cut on the box with `--matmat-bench -j1` AND at full threads — the
 * two answers differ, and the server runs at full threads with several slots.
 *
 * WHAT THIS IS NOT: it is not a cost model. It is one readable place where the numbers
 * live, so that changing them is a one-line diff and an env var, instead of a hunt
 * through six #if blocks.
 *
 * Env overrides (all read once, cached in relaxed atomics — this is on the per-call
 * path of every projection):
 *   QWEN_NO_AMX=1        master off for every AMX kernel (kept: it is the documented one)
 *   QWEN_NO_AMX_INT8/_BF16/_Q4=1   off for one AMX format only (A/B one format at a time)
 *   QWEN_NO_VNNI / QWEN_NO_AVX2MM / QWEN_NO_SMMLA / QWEN_NO_BFMMLA=1   as before
 *   QWEN_INT8_SDOT_MM=1  opt-in for the looped-SDOT twin (unchanged: it is a reference)
 *   QWEN_APPLE_MMLA=1    re-enables the ARM MMLA paths that measured a LOSS on Apple
 *   QWEN_AMX_MIN_B / QWEN_VNNI_MIN_B / QWEN_AVX2MM_MIN_B / QWEN_SMMLA_MIN_B /
 *   QWEN_BFMMLA_MIN_B / QWEN_INT8_SDOT_MIN_B=<n>   move the B crossover at runtime
 *                        (this is the sweep knob, and `--matmat-tune` prints the values)
 *   QWEN_AMX_MIN_ROWS / QWEN_AMX_{BF16,INT8,Q4}_MIN_COLS=<n>  the other two guessed
 *                        AMX thresholds, overridable for the same reason. Lowering a
 *                        min_cols below one k-chunk is legal but pointless: the kernel
 *                        then sends everything through its scalar tail. */
#if defined(__GNUC__)
#define QWEN_MAYBE_UNUSED __attribute__((unused))
#else
#define QWEN_MAYBE_UNUSED
#endif

typedef struct {
    const char *off_env;      /* "=1" disables this kernel                       */
    const char *on_env;       /* non-NULL => OPT-IN: runs only if this is "1"    */
    const char *minb_env;     /* "=<n>" overrides min_b (the on-box sweep knob)  */
    const char *minrows_env;  /* "=<n>" overrides min_rows (NULL = no guess here) */
    const char *mincols_env;  /* "=<n>" overrides min_cols (NULL = no guess here) */
    short min_b, max_b;       /* B window this kernel is allowed to serve        */
    int   min_rows, min_cols; /* below this the setup cost is not repaid         */
    unsigned char amx;        /* also honours the QWEN_NO_AMX master switch      */
    unsigned char apple_off;  /* measured LOSS on Apple cores; QWEN_APPLE_MMLA=1 */
} qwen_mm_gate_t;

/* WHY max_b HAS NO ENV OVERRIDE, AND MUST NOT GET ONE. It is not a tuning number: the
 * int8/q4 dispatchers hold the per-column activation scales in a `float sx[16]` on the
 * stack and fill it with `for (b = 0; b < B; b++)`. max_b = 16 is what keeps that loop
 * inside the array. Raising it from the environment would be a stack smash, not a
 * slower run — same category as qwen_amx_*_ready() and cols % Q4_0_BLOCK_SIZE, i.e. a
 * precondition, not a threshold. (BFMMLA carries 64 because bf16 has no activation
 * scales at all.) `--matmat-tune` therefore sweeps B only up to 16. */

/* One place to read an integer threshold from the environment, so every override
 * behaves the same: unset/empty/out-of-range = keep the compiled default. Out-of-range
 * silently falling back is deliberate — a typo in a sweep script must not turn into a
 * kernel that runs at B=0 or at B=100000. */
static int qwen_mm_env_int(const char *name, int dflt, int lo, int hi) QWEN_MAYBE_UNUSED;
static int qwen_mm_env_int(const char *name, int dflt, int lo, int hi) {
    if (!name) return dflt;
    const char *e = getenv(name);
    if (!e || !e[0]) return dflt;
    int v = atoi(e);
    return (v >= lo && v <= hi) ? v : dflt;
}

/* min_b rationale, kernel by kernel — these are the numbers to argue with, not the code:
 *  AMX (all three formats) = 4. The C tile is 16 wide; at B=4 a quarter of it is live,
 *    which is where the 16x16x64 (int8/q4) or 16x16x32 (bf16) MAC block still beats a
 *    64-MAC VPDPBUSD per cycle on raw throughput. At B=2 it does not, and below that
 *    the engine does not even call a matmat (see the B=1 note in qwen_matmat_int8).
 *  AMX min_cols: 64 for int8 (one whole k-chunk; below it everything is scalar tail),
 *    32 for bf16 and q4 (their k-chunk is 32 — bf16 because a tile row is 32 bf16,
 *    q4 because a Q4_0 block is 32 weights and the per-block scale forbids joining two).
 *  AMX min_rows = 32 = two 16-row blocks: with one block the two accumulators that hide
 *    the ~52-cycle TDP latency have nothing to interleave.
 *  VNNI / AVX2 / SMMLA / BFMMLA = 2, i.e. unchanged from before this gate existed. They
 *    are ordinary vector kernels: no tile state, no repack, nothing to amortize, so the
 *    old "any B >= 2" is still the honest default until the box says otherwise. */
static const qwen_mm_gate_t g_mm_gate[QWEN_MMK_COUNT] QWEN_MAYBE_UNUSED = {
    [QWEN_MMK_BF16_BFMMLA] = { "QWEN_NO_BFMMLA",   NULL,                "QWEN_BFMMLA_MIN_B",  NULL,                NULL,                     2, 64,  0,  0, 0, 1 },
    [QWEN_MMK_BF16_AMX]    = { "QWEN_NO_AMX_BF16", NULL,                "QWEN_AMX_MIN_B",     "QWEN_AMX_MIN_ROWS", "QWEN_AMX_BF16_MIN_COLS", 4, 16, 32, 32, 1, 0 },
    [QWEN_MMK_INT8_AMX]    = { "QWEN_NO_AMX_INT8", NULL,                "QWEN_AMX_MIN_B",     "QWEN_AMX_MIN_ROWS", "QWEN_AMX_INT8_MIN_COLS", 4, 16, 32, 64, 1, 0 },
    [QWEN_MMK_INT8_VNNI]   = { "QWEN_NO_VNNI",     NULL,                "QWEN_VNNI_MIN_B",    NULL,                NULL,                     2, 16,  0,  0, 0, 0 },
    [QWEN_MMK_INT8_AVX2]   = { "QWEN_NO_AVX2MM",   NULL,                "QWEN_AVX2MM_MIN_B",  NULL,                NULL,                     2, 16,  0,  0, 0, 0 },
    [QWEN_MMK_INT8_SMMLA]  = { "QWEN_NO_SMMLA",    NULL,                "QWEN_SMMLA_MIN_B",   NULL,                NULL,                     2, 16,  0,  0, 0, 1 },
    [QWEN_MMK_INT8_SDOT]   = { NULL,               "QWEN_INT8_SDOT_MM", "QWEN_INT8_SDOT_MIN_B", NULL,              NULL,                     2, 16,  0,  0, 0, 0 },
    [QWEN_MMK_Q4_AMX]      = { "QWEN_NO_AMX_Q4",   NULL,                "QWEN_AMX_MIN_B",     "QWEN_AMX_MIN_ROWS", "QWEN_AMX_Q4_MIN_COLS",   4, 16, 32, 32, 1, 0 },
    [QWEN_MMK_Q4_VNNI]     = { "QWEN_NO_VNNI",     NULL,                "QWEN_VNNI_MIN_B",    NULL,                NULL,                     2, 16,  0,  0, 0, 0 },
    [QWEN_MMK_Q4_AVX2]     = { "QWEN_NO_AVX2MM",   NULL,                "QWEN_AVX2MM_MIN_B",  NULL,                NULL,                     2, 16,  0,  0, 0, 0 },
    [QWEN_MMK_Q4_SMMLA]    = { "QWEN_NO_SMMLA",    NULL,                "QWEN_SMMLA_MIN_B",   NULL,                NULL,                     2, 16,  0,  0, 0, 0 },
    /* min_b = 1, and it is the only row with that: KleidiAI ships a real GEMV
     * (qsi8d32p1x8, dotprod) for m=1, so B=1 is a first-class case rather than a
     * degenerate GEMM. That is exactly the asymmetry llama.cpp encodes too - it
     * pairs an i8mm GEMM with a dotprod GEMV and switches on the row count. */
    [QWEN_MMK_KLEIDI_Q4]   = { "QWEN_NO_KLEIDI",   NULL,                "QWEN_KLEIDI_MIN_B",  NULL,                NULL,                     1, 64,  0,  0, 0, 0 },
};
/* 0 = not resolved yet, 1 = enabled, 2 = disabled (0 is the zero-init sentinel). */
static atomic_int g_mm_gate_on[QWEN_MMK_COUNT];
static atomic_int g_mm_gate_minb[QWEN_MMK_COUNT];   /* 0 = not resolved yet */
/* min_rows / min_cols caches, stored +1-BIASED so that a resolved value of 0 (which is
 * the default for every non-AMX kernel) is still distinguishable from "not resolved
 * yet". A separate "resolved" flag would need release/acquire ordering against the
 * value; one self-describing cell does not, and this sits on the per-call path. */
static atomic_int g_mm_gate_minrows[QWEN_MMK_COUNT];
static atomic_int g_mm_gate_mincols[QWEN_MMK_COUNT];

/* ── `--matmat-tune` only: pin the dispatcher to ONE kernel ───────────────────────
 * 0 = normal dispatch. >0 = only that QWEN_MMK_* may run, with min_b/min_rows/min_cols
 * BYPASSED (the whole point is to measure the kernel below its guessed threshold, so
 * the threshold can be re-derived instead of assumed). <0 = every gated kernel off, so
 * the dispatcher falls through to its untimed tail (the twin / B x matvec floor).
 *
 * What this does NOT bypass, deliberately: qwen_amx_*_ready() and cols %
 * Q4_0_BLOCK_SIZE. Those live at the call sites, are preconditions rather than
 * thresholds, and forcing them is a SIGILL or a wrong answer. A forced kernel that is
 * blocked by one of them therefore silently falls through to the tail — which is why
 * the tuner VERIFIES via the MMSTAT counters that the kernel it forced is the kernel
 * that ran, and prints "did not fire" instead of attributing the tail's time to it. */
static atomic_int g_mm_force;
static void qwen_mm_force_kernel(int mmk) QWEN_MAYBE_UNUSED;
static void qwen_mm_force_kernel(int mmk) {
    atomic_store_explicit(&g_mm_force, mmk, memory_order_relaxed);
}

/* Answers ONE question: at this (ISA-selected kernel, B, rows, cols), should this
 * kernel run? Callers still own the correctness preconditions that are not tuning —
 * cols % Q4_0_BLOCK_SIZE for the q4 twins, qwen_amx_*_ready() for the tile paths —
 * because those are not thresholds and must never be overridable by an env var. */
static int qwen_mm_use(int mmk, int B, int rows, int cols) QWEN_MAYBE_UNUSED;
static int qwen_mm_use(int mmk, int B, int rows, int cols) {
    if (mmk <= 0 || mmk >= QWEN_MMK_COUNT) return 0;
    /* Tuning override first: it must win over the kill switches and the B window, or
     * the sweep could only ever confirm the guesses it exists to replace. */
    int force = atomic_load_explicit(&g_mm_force, memory_order_relaxed);
    if (force != 0) return force == mmk;
    const qwen_mm_gate_t *g = &g_mm_gate[mmk];
    if (g->max_b == 0) return 0;                     /* no gate declared = not dispatchable */
    int st = atomic_load_explicit(&g_mm_gate_on[mmk], memory_order_relaxed);
    if (st == 0) {
        int on = 1;
        const char *e;
        if (g->on_env)          { e = getenv(g->on_env);  on = (e && e[0] == '1'); }
        if (on && g->off_env)   { e = getenv(g->off_env); if (e && e[0] == '1') on = 0; }
        if (on && g->amx)       { e = getenv("QWEN_NO_AMX"); if (e && e[0] == '1') on = 0; }
#if defined(__APPLE__)
        /* M4-measured 2026-07-11: the ARM MMLA batched twins LOSE on Apple's
         * bandwidth-rich cores (int8-SMMLA 0.61-0.91x, BFMMLA 0.72-0.91x) while
         * winning ~1.5-2.1x on Graviton3 -> default OFF here, opt-in to re-evaluate. */
        if (on && g->apple_off) { e = getenv("QWEN_APPLE_MMLA"); on = (e && e[0] == '1'); }
#endif
        st = on ? 1 : 2;
        atomic_store_explicit(&g_mm_gate_on[mmk], st, memory_order_relaxed);
    }
    if (st != 1) return 0;
    int minb = atomic_load_explicit(&g_mm_gate_minb[mmk], memory_order_relaxed);
    if (minb == 0) {
        minb = qwen_mm_env_int(g->minb_env, g->min_b, 1, 64);
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
    return B >= minb && B <= g->max_b && rows >= minr - 1 && cols >= minc - 1;
}

/* Which kernel will ACTUALLY fire, asked to the real dispatcher.
 *
 * llama.cpp prints "primary q4 kernel feature I8MM" at load, and that one line is
 * what makes its acceleration claims checkable: a build flag says what was compiled
 * in, this says what got selected. We had the ingredients already - qwen_mm_use() is
 * the live predicate and g_mmk_info[] holds the names - but nothing ever printed the
 * answer, so "is SMMLA firing?" could only be answered by reading a profile.
 *
 * It probes the gate with the caller's real shapes rather than reporting capability
 * bits, so a kernel that is compiled in but gated off by min_b / min_rows / an env
 * kill-switch shows up as not selected, which is the honest answer. */
void qwen_kernel_selection_report(void *out, int rows, int cols) {
    FILE *f = out ? (FILE *)out : stderr;
    if (rows <= 0) rows = 2048;
    if (cols <= 0) cols = 2048;

    /* Candidates in dispatch priority order, per precision. The first one whose
     * gate accepts the shape is the one that runs. */
    /* Only kernels this binary actually CONTAINS are candidates. The gate table
     * alone is not enough: the AMX entries pass qwen_mm_use() on any build, and
     * the real dispatchers gate them a second time on qwen_amx_*_ready(), which
     * lives outside the table. Reporting a kernel that cannot run is worse than
     * reporting nothing - it is a plausible false answer. */
    int bf16_c[4], int8_c[6], q4_c[6];
    int nbf = 0, nint8 = 0, nq4 = 0;
#if defined(__AMX_BF16__) && defined(__AMX_TILE__)
    if (qwen_amx_bf16_ready()) bf16_c[nbf++] = QWEN_MMK_BF16_AMX;
#endif
#if defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC)
    bf16_c[nbf++] = QWEN_MMK_BF16_BFMMLA;
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

    /* B=1 never reaches the matmat gate at all: single-stream runs the matvec
     * path, chosen by one-shot env flags rather than the table. It is reported
     * first and separately because that is the regime the CLI and a server at
     * c=1 actually run in - and the matrix units do NOT serve it. */
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


/* ══ WHY THERE IS NO AMX *MATVEC* (B=1), with the arithmetic ═══════════════════════
 *
 * Asked explicitly on 2026-08-18, so the answer is written down rather than
 * re-derived: an AMX kernel for the single-stream path (B_eff==1, which is what a
 * lightly-loaded server spends most of its life in) is NOT worth writing. Three
 * independent counts, any one of which is enough.
 *
 * 1. THERE IS NO COMPUTE HEADROOM TO WIN. An int8 matvec does rows*cols MACs while
 *    reading rows*cols weight bytes: arithmetic intensity is exactly 1 MAC/byte. One
 *    SPR core sustains on the order of 10-20 GB/s on its own (a single core is limited
 *    by its outstanding-miss window, not by the socket's 200+ GB/s), i.e. ~4-8
 *    bytes/cycle at ~2.5-3 GHz, i.e. ~4-8 MAC/cycle of useful work. A single VPDPBUSD
 *    already delivers 64 MACs and issues about once per cycle. The existing VNNI matvec
 *    is therefore ~10x over-provisioned on compute BEFORE AMX is considered; a bigger
 *    multiplier multiplies a number that is not the bottleneck.
 *
 * 2. AT N=1 THE TILE IS 6.25% LIVE, AND THE INSTRUCTION IS PRICED BY M. _tile_dpbssd
 *    streams the M rows of operand A through the array; its cost tracks M (and K), not
 *    N. With one activation column, C is 16x16 int32 of which 16 entries (16/256 =
 *    6.25%) hold a result: 16*1*64 = 1024 useful MACs for the price of the 16384 the
 *    instruction can do — about 64 MAC/cycle at the ~16-cycle full-tile throughput.
 *    That is at best a tie with one VPDPBUSD per cycle, and it still pays a TILECFG per
 *    call and a TILESTORED per 16 rows on top. Slower, not faster.
 *
 * 3. "USE THE 16 OUTPUT ROWS AS N INSTEAD OF THE BATCH" — the tempting fix, and it
 *    inverts the one finding this whole AMX effort rests on. To put output rows in the
 *    N dimension the WEIGHTS have to become operand B, and operand B is precisely the
 *    one that must be quadruple-interleaved. That is the layout already costed here:
 *    16x16 = 256 dword moves to feed ONE _tile_dpbssd, an order of magnitude more
 *    shuffling than math, forever. It only stops being absurd with a LOAD-TIME repack
 *    (PLAN T5.5 A5) — at which point it is a storage-format decision, not a kernel, and
 *    point 1 still says there is nothing to win. Note the asymmetry: in the matmat the
 *    packed operand is the activations, O(B*cols) per call; here it would be the
 *    weights, O(rows*cols) per call.
 *
 * AND THE ENGINE AGREES: qwen_tts_talker.c takes the B_eff==1 shortcut to the ordinary
 * single-stream step (audited as QWEN_MMK_SOLO), so B=1 does not reach these matmat
 * dispatchers at all. Writing an AMX matvec would mean first routing B=1 INTO a matmat.
 *
 * WHAT WOULD FALSIFY THIS, in one measurement: if `--matmat-bench -j1` on the c3 shows
 * the int8 MATVEC running far above ~8 MAC/cycle/thread, the weights are being served
 * from L2/L3 rather than DRAM — plausible for the small Code-Predictor projections,
 * which are re-read 16x per frame and may fit — and point 1 collapses for exactly those
 * shapes. Then, and only then, the argument to re-open is A5 + an AMX matvec for the
 * L2-resident shapes only. Measure per-core bandwidth first; do not reason from the
 * socket number. */

#if (defined(__AMX_INT8__) || defined(__AMX_BF16__)) && defined(__AMX_TILE__)
/* AMX tile-shape descriptor (the operand of LDTILECFG). Hoisted here because all three
 * formats configure tiles and the bf16 GEMM sits above the int8 one in this file.
 * Layout is architectural — do not reorder the fields. */
typedef struct {
    uint8_t  palette_id;
    uint8_t  start_row;
    uint8_t  reserved0[14];
    uint16_t colsb[8];
    uint16_t reserved1[8];
    uint8_t  rows[8];
    uint8_t  reserved2[8];
} qwen_amx_tilecfg;
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
    MMSTAT(QWEN_MMK_BF16_FIXEDB, r1 - r0, cols, B);   /* batched-path audit: MACs by kernel */
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

/* ── ARM MMLA batched matmat twins (rental-prep, 2026-07-11) ──────────────────
 * BFMMLA (bf16) + SMMLA (i8mm): the native 2x2-tile matrix-multiply units on
 * M2+/M3/M4, Neoverse V1/V2, Graviton3+. Compile-time guarded — on the M1 these
 * guards are false and the scalar/NEON twins below stay the path; `make
 * check-isa` (-march=armv8.6-a+bf16+i8mm) keeps them compiling from the M1.
 * These are exactly the "AVAILABLE but UNUSED (PLAN 21.3b)" paths in --caps.
 * Numerics: BFMMLA computes bf16×bf16→f32 with the ACTIVATION truncated to bf16
 * (same truncation as the KV cache) → vs the f32-activation twin expect ~1e-3
 * L2 (self-test threshold relaxed to 1e-2 under this guard). SMMLA reuses the
 * per-column int8 activation quant → integer-exact per column vs SDOT.
 * A/B kill-switches on the box: QWEN_NO_BFMMLA=1 / QWEN_NO_SMMLA=1. */
#if defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC)
static inline float qbf16_to_f32(uint16_t b) {
    uint32_t u = (uint32_t)b << 16; float f; memcpy(&f, &u, 4); return f;
}
static void bf16_matmat_bfmmla_slice(float *Y, const uint16_t *W, const uint16_t *Xb,
                                     int r0, int r1, int cols, int B) {
    MMSTAT(QWEN_MMK_BF16_BFMMLA, r1 - r0, cols, B);   /* batched-path audit: MACs by kernel */
    /* Xb: [B][cols] bf16 bits (pre-transposed + truncated).
     *
     * ── 4x4, per la stessa ragione dell'int8 (2026-08-21) ────────────────────────
     * Una BFMMLA e' un tile 2x2x4: con UN accumulatore il ciclo e' `acc -> acc` a ogni
     * k += 4, cioe' legato alla latenza dell'istruzione. Sull'Axion questo gemello
     * misurava 0,73-1,02x contro B x matvec — PERDEVA — e non perche' il bf16 non si
     * presti: perche' il kernel teneva ferma l'unita'. Quattro catene indipendenti dagli
     * stessi due carichi di pesi e due di attivazione.
     *
     * Perche' conta oltre al banco: questo e' il kernel su cui poggia il PREFILL quando
     * QWEN_PREFILL_MATMAT=1 (prefill_proj_matmat -> qwen_matmat_bf16), cioe' la leva che
     * su AMX ha dimezzato il TTFA. Finche' il gemello perdeva, quella strada su ARM era
     * chiusa in partenza.
     * Numerica invariata: stessa aritmetica, stesso ordine di accumulo per (riga, colonna).
     * Il ciclo 2x2 resta sotto per i resti di riga e di colonna. */
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
            for (; k < cols; k++) {                  /* coda k (cols%4; le dim del modello non la toccano) */
                float wv0 = qbf16_to_f32(W[(size_t)r * cols + k]),       wv1 = qbf16_to_f32(W[(size_t)(r+1) * cols + k]);
                float wv2 = qbf16_to_f32(W[(size_t)(r+2) * cols + k]),   wv3 = qbf16_to_f32(W[(size_t)(r+3) * cols + k]);
                float xv0 = qbf16_to_f32(Xb[(size_t)j * cols + k]),      xv1 = qbf16_to_f32(Xb[(size_t)(j+1) * cols + k]);
                float xv2 = qbf16_to_f32(Xb[(size_t)(j+2) * cols + k]),  xv3 = qbf16_to_f32(Xb[(size_t)(j+3) * cols + k]);
                t00[0] += wv0*xv0; t00[1] += wv0*xv1; t00[2] += wv1*xv0; t00[3] += wv1*xv1;
                t01[0] += wv0*xv2; t01[1] += wv0*xv3; t01[2] += wv1*xv2; t01[3] += wv1*xv3;
                t10[0] += wv2*xv0; t10[1] += wv2*xv1; t10[2] += wv3*xv0; t10[3] += wv3*xv1;
                t11[0] += wv2*xv2; t11[1] += wv2*xv3; t11[2] += wv3*xv2; t11[3] += wv3*xv3;
            }
            /* lane: [rA·cP, rA·cQ, rB·cP, rB·cQ] */
            float *Y0 = Y + (size_t)r * B,       *Y1 = Y + (size_t)(r + 1) * B;
            float *Y2 = Y + (size_t)(r + 2) * B, *Y3 = Y + (size_t)(r + 3) * B;
            Y0[j] = t00[0]; Y0[j+1] = t00[1]; Y1[j] = t00[2]; Y1[j+1] = t00[3];
            Y0[j+2] = t01[0]; Y0[j+3] = t01[1]; Y1[j+2] = t01[2]; Y1[j+3] = t01[3];
            Y2[j] = t10[0]; Y2[j+1] = t10[1]; Y3[j] = t10[2]; Y3[j+1] = t10[3];
            Y2[j+2] = t11[0]; Y2[j+3] = t11[1]; Y3[j+2] = t11[2]; Y3[j+3] = t11[3];
        }
        for (; j < B; j++) {                          /* resto di colonna, ancora 4 righe */
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
                acc = vbfmmlaq_f32(acc, a, b);   /* lanes: [r·c, r·c1, r1·c, r1·c1] */
            }
            float t[4]; vst1q_f32(t, acc);
            for (; k < cols; k++) {              /* k tail (cols%4!=0 — not hit by model dims) */
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
        for (; j < B; j++) {                     /* odd-B col tail */
            float s0 = 0.0f, s1 = 0.0f;
            for (int k = 0; k < cols; k++) {
                float xv = qbf16_to_f32(Xb[(size_t)j * cols + k]);
                s0 += qbf16_to_f32(W[(size_t)r * cols + k]) * xv;
                s1 += qbf16_to_f32(W[(size_t)(r + 1) * cols + k]) * xv;
            }
            Y[(size_t)r * B + j] = s0; Y[(size_t)(r + 1) * B + j] = s1;
        }
    }
    for (; r < r1; r++) {                        /* odd-rows tail */
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
    r0 &= ~3; if (tid + 1 < nt) r1 &= ~3;        /* keep 4-row blocks within one slice */
    bf16_matmat_bfmmla_slice(c->Y, c->W, c->Xb, r0, r1, c->cols, c->B);
}
#endif /* __ARM_FEATURE_BF16_VECTOR_ARITHMETIC */


#if defined(__AMX_BF16__) && defined(__AMX_TILE__)
/* ── bf16 batched matmat on Intel AMX (_tile_dpbf16ps; Sapphire c3 / Emerald c4) ────
 *
 * WHY BF16 FIRST, AHEAD OF ANY OTHER AMX FORMAT. Two reasons, and neither is "it was
 * next on the list".
 *  (a) bf16 is the ONLY format where batching has ever paid on hardware we own: the
 *      batched twin measured 1.70x against B x matvec at 4 threads, against 1.12x for
 *      int8 and 0.97x for int4 — because bf16 is the one with a real matrix primitive
 *      behind it (BFMMLA on ARM). It is the best starting base, so it is the format
 *      where an AMX win is most likely to be a REAL win rather than a wash.
 *      (Caveat that must travel with that number: at ONE thread the same twin is 0.79x.
 *      The 1.70x was partly pool-launch amortization. See the gate note above.)
 *  (b) the codec head is bf16 and is NOT quantized: `vocab x hidden` = 3072x2048 on the
 *      1.7B, ~12.6 MB, crossed once per frame FOR EVERY SLOT. On the server path it is
 *      already batched, so it lands exactly here. If AMX moves anything at B=4-8, this
 *      is the matrix where it shows up first.
 *
 * THE INSTRUCTION. _tile_dpbf16ps(C,A,B) computes
 *     C[m][n] += fp32(A[m][2k])*fp32(B[k][2n]) + fp32(A[m][2k+1])*fp32(B[k][2n+1])
 * so it is the int8 shape with the dword split into 2 bf16 instead of 4 int8:
 *     A = 16 rows x 64 bytes = 16 rows x 32 bf16   (plain row-major)
 *     B = 16 rows x 4*N bytes                       (pairs of k interleaved per column)
 *     C = 16 x N f32
 * K per instruction is therefore 32, not 64 — half the int8 depth, which is why the
 * bf16 gate asks only cols >= 32 while int8 asks 64.
 *
 * SAME ORIENTATION AS THE INT8 KERNEL, FOR THE SAME REASON: weights in A (loaded
 * straight from W with stride 2*cols, no repack ever), activations packed into B once
 * per call (O(B*cols), amortized over every row block). Inverting it would put a
 * 16x16 = 256-move shuffle in front of every single tile op. See the int8 note below.
 *
 * NUMERICS, AND THIS IS A REAL DIFFERENCE FROM THE FIXED-B TWIN. _tile_dpbf16ps takes
 * bf16 inputs, so the ACTIVATION is truncated to bf16 exactly as the ARM BFMMLA path
 * does (and as the KV cache does). Against the f32-activation twin expect ~1e-3 L2, not
 * zero — that is the correct value, not a defect. `--self-test` already relaxes the
 * matmat threshold to 1e-2 under __AVX512BF16__ / __ARM_FEATURE_BF16_VECTOR_ARITHMETIC,
 * and a Sapphire Rapids build defines __AVX512BF16__; the guard there now names
 * __AMX_BF16__ too so the relaxation cannot depend on an unrelated feature bit.
 * Truncation (not round-to-nearest) is deliberate: it matches bf16_matmat_bfmmla_slice,
 * so the two matrix paths stay comparable to each other and to the KV cache.
 *
 * ⚠️ WRITTEN ON M1, NEVER EXECUTED. No Apple silicon has AMX and Rosetta 2 emulates
 * neither AVX-512 nor AMX, so this has never run: the only gate it has had is a forced
 * cross-compile (-march=sapphirerapids -fsyntax-only) plus review against the
 * TDPBF16PS pseudocode. NO performance number is claimed here.
 * What would falsify it, in order, on the first c3/c4:
 *   1. `--caps` must say `x86 amx bf16: AMX ACTIVE`. If it says permission DENIED the
 *      kernel is < 5.16 and every bf16 batching number below it is an AVX-512 number.
 *   2. `--self-test` matmat(B=8) L2_rel must land ~1e-3, i.e. activation-truncation
 *      noise. A wrong pack does NOT give a small error — it gives garbage (~1e0), so
 *      this test discriminates pack bugs from rounding.
 *   3. `--matmat-bench` bf16, at -j1 AND at full threads, with and without
 *      QWEN_NO_AMX_BF16=1. AMX must beat the fixed-B twin on BOTH. If it wins only at
 *      full threads it is amortizing pool launches, which is the trap the M1 1.70x
 *      turned out to be — say so instead of shipping it.
 *   4. QWEN_AMX_MIN_B sweep 2/4/6/8 at the codec-head shape (3072x2048): that is what
 *      sets the real crossover against the twin, and 4 is a guess. */

/* Xb[B][cols] bf16 -> pXb, operand-B layout for dpbf16ps:
 *   pXb[kc][j][2n+q] = Xb[n][kc*32 + 2j + q]      (kc = 32-value k chunk, j = 0..15)
 * 16-row tiles, row stride 2*B bf16 = 4*B bytes; the whole buffer is kpack*B bf16. */
static void amx_pack_act_bf16(uint16_t *pXb, const uint16_t *Xb, int cols, int kpack, int B) {
    const int nchunk = kpack >> 5;
    const size_t cstride = (size_t)B * 2;            /* bf16 elements per packed row */
    for (int kc = 0; kc < nchunk; kc++) {
        uint16_t *dst = pXb + (size_t)kc * 32 * (size_t)B;
        for (int n = 0; n < B; n++) {
            const uint16_t *src = Xb + (size_t)n * cols + (size_t)kc * 32;
            for (int j = 0; j < 16; j++)
                memcpy(dst + (size_t)j * cstride + (size_t)n * 2, src + 2 * j, 2 * sizeof(uint16_t));
        }
    }
}

static void bf16_matmat_amx_slice(float *Y, const uint16_t *W, const uint16_t *pXb,
                                  const uint16_t *Xb, int r0, int r1, int cols, int B) {
    MMSTAT(QWEN_MMK_BF16_AMX, r1 - r0, cols, B);   /* batched-path audit: MACs by kernel */
    const int kfull   = cols & ~31;          /* one tile row = 32 bf16 */
    const int nchunk  = kfull >> 5;
    const int cstride = B * 4;               /* bytes per C row (B f32) = per packed row */
    const size_t wstride = (size_t)cols * sizeof(uint16_t);

    /* Tile shapes set ONCE per slice: LDTILECFG invalidates all tiles, so doing it per
     * row block would throw away the accumulators it just configured. Released at the
     * end so a worker does not carry 8 KB of live XSAVE state into the next kernel. */
    qwen_amx_tilecfg cfg;
    memset(&cfg, 0, sizeof cfg);
    cfg.palette_id = 1;
    cfg.rows[0] = 16; cfg.colsb[0] = (uint16_t)cstride;   /* tmm0: C, rows r..r+15    */
    cfg.rows[1] = 16; cfg.colsb[1] = (uint16_t)cstride;   /* tmm1: C, rows r+16..+31  */
    cfg.rows[2] = 16; cfg.colsb[2] = 64;                  /* tmm2: W rows r..r+15     */
    cfg.rows[3] = 16; cfg.colsb[3] = 64;                  /* tmm3: W rows r+16..+31   */
    cfg.rows[4] = 16; cfg.colsb[4] = (uint16_t)cstride;   /* tmm4: packed activations */
    _tile_loadconfig(&cfg);

    float cbuf[2][16 * 16] __attribute__((aligned(64)));

    int r = r0;
    /* Two row blocks per iteration: TDPBF16PS has a long latency against a short
     * throughput, and accumulating over k into ONE C tile is a dependent chain. Two
     * blocks sharing the same activation tile give two independent chains. */
    for (; r + 31 < r1; r += 32) {
        _tile_zero(0); _tile_zero(1);
        for (int kc = 0; kc < nchunk; kc++) {
            _tile_loadd(4, pXb + (size_t)kc * 32 * (size_t)B, cstride);
            _tile_loadd(2, W + (size_t)r * cols + (size_t)kc * 32, wstride);
            _tile_loadd(3, W + (size_t)(r + 16) * cols + (size_t)kc * 32, wstride);
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
                    for (int kk = kfull; kk < cols; kk++)   /* k tail (cols % 32) */
                        acc += bf16_to_f32(w[kk]) * bf16_to_f32(xb[kk]);
                    Y[(size_t)rr * B + b] = acc;
                }
            }
    }
    for (; r + 15 < r1; r += 16) {
        _tile_zero(0);
        for (int kc = 0; kc < nchunk; kc++) {
            _tile_loadd(4, pXb + (size_t)kc * 32 * (size_t)B, cstride);
            _tile_loadd(2, W + (size_t)r * cols + (size_t)kc * 32, wstride);
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
    _tile_release();

    /* Row tail (< 16 rows left in this slice): scalar, on the SAME truncated
     * activations, so a row's numerics do not depend on where the slice boundary fell. */
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
    float *Y; const uint16_t *W; const uint16_t *pXb; const uint16_t *Xb;
    int rows, cols, B;
} bf16_amx_ctx;
static void bf16_amx_task(size_t tid, size_t nt, void *vc) {
    bf16_amx_ctx *c = (bf16_amx_ctx *)vc;
    int r0 = (int)(tid * (size_t)c->rows / nt);
    int r1 = (int)((tid + 1) * (size_t)c->rows / nt);
    r0 &= ~15; if (tid + 1 < nt) r1 &= ~15;   /* keep 16-row tiles inside one slice */
    bf16_matmat_amx_slice(c->Y, c->W, c->pXb, c->Xb, r0, r1, c->cols, c->B);
}
#endif /* __AMX_BF16__ && __AMX_TILE__ */

/* ── Scratch per-thread al posto di malloc/free nel ciclo autoregressivo (2026-08-21) ──
 *
 * Ogni via batchata (AMX / VNNI / SMMLA / AVX2 / SDOT / q4) allocava e liberava il suo
 * buffer DENTRO la chiamata: una coppia malloc+free per matmat, per layer, per frame,
 * per thread. Nel Code Predictor sono 15 passi per frame, quindi e' traffico d'allocatore
 * puro su un percorso che gira migliaia di volte al secondo — ed e' esattamente cio' che
 * aveva reso interessante la questione allocatore il 20/08 (the design notes.
 *
 * I buffer sono piccoli e limitati per costruzione (B <= max_b = 16, cols <= dim del
 * modello), quindi un buffer per thread che cresce fino al massimo visto e poi non si
 * muove piu' e' la forma giusta: dopo i primi frame non c'e' piu' nessuna allocazione.
 * Stesso schema gia' usato dal prefill (pp_xT/pp_yT in qwen_tts_talker.c).
 *
 * Proprieta' e vita: il buffer appartiene al thread CHIAMANTE, che e' quello che
 * quantizza; i worker di qwen_parallel lo leggono per la durata della chiamata, che e'
 * sincrona — la stessa proprieta' che rendeva corretto il malloc di prima.
 * NON cambia un bit del risultato: il gate e' che il WAV resti byte-identico. */
#define QWEN_MM_SCRATCH(name, type)                                                      \
    static __thread type *g_mms_##name = NULL;                                           \
    static __thread size_t g_mms_cap_##name = 0;                                         \
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
QWEN_MM_SCRATCH(qx,   int8_t)     /* attivazione quantizzata [B][cols] */
QWEN_MM_SCRATCH(pack, int8_t)     /* attivazione impacchettata per i tile (AMX/VNNI) */
QWEN_MM_SCRATCH(packb, uint16_t)  /* idem, bf16 */
QWEN_MM_SCRATCH(corr, int)        /* correzioni per blocco del q4 */

void qwen_matmat_bf16(float *Y, const uint16_t *W, const float *X, int rows, int cols, int B) {
    qwen_census_op("matmat_bf16", rows, cols, B);
    if (g_qwen_matmat_bf16_hook) { g_qwen_matmat_bf16_hook(Y, W, X, rows, cols, B); return; }
    if (B <= 0) return;
    /* Same interception as the matvec above: a GGUF Q8_0 matrix is registered under
     * its bf16 pointer and runs on the repacked SMMLA path. This also puts PREFILL on
     * Q8_0 - without it prefill would keep reading the bf16 copy and the run would be
     * a mix of two numerics, which is the exact trap the Q4 path had to be told about. */
    if (kai_bf16_try(Y, W, X, rows, cols, B)) {
        MMSTAT(B > 1 ? QWEN_MMK_KLEIDI_BF16 : QWEN_MMK_KLEIDI_BF16_GEMV, rows, cols, B);
        return;
    }
    if (qwen_q8r_matmul(Y, (const void *)W, X, rows, cols, B)) {
        MMSTAT(QWEN_MMK_Q8_REPACK_I8MM, rows, cols, B);
        return;
    }
    if (B > 64) B = 64;  /* contract: B<=64 */
    int nt = g_n_threads;
#if defined(__AMX_BF16__) && defined(__AMX_TILE__)
    /* AMX bf16 FIRST on x86: it is the only matrix-MATRIX bf16 unit here (AVX-512-BF16's
     * VDPBF16PS is still a dot product). Shape/ISA decision lives in qwen_mm_use();
     * qwen_amx_bf16_ready() is NOT a threshold — it is CPU + kernel permission, and it
     * must be called HERE, on the calling thread, before qwen_parallel hands out slices,
     * or a worker issues the first tile op without the XTILEDATA grant and takes SIGILL. */
    if (qwen_mm_use(QWEN_MMK_BF16_AMX, B, rows, cols) && qwen_amx_bf16_ready()) {
        const size_t kfull = (size_t)(cols & ~31);
        uint16_t *Xb  = (uint16_t *)malloc((size_t)B * cols * sizeof(uint16_t));
        uint16_t *pXb = NULL;
        /* 64-byte aligned: TILELOADD does not require it, but each tile row is read once
         * per row block and straddling a cache line costs on every one of them. */
        if (Xb) pXb = mm_scratch_packb(kfull * (size_t)B);
        if (Xb && pXb) {
            for (int b = 0; b < B; b++)
                for (int k = 0; k < cols; k++) {
                    uint32_t u; memcpy(&u, &X[(size_t)k * B + b], 4);
                    Xb[(size_t)b * cols + k] = (uint16_t)(u >> 16);   /* truncate, like the KV */
                }
            amx_pack_act_bf16(pXb, Xb, cols, (int)kfull, B);   /* once per call, not per slice */
            if (nt > 1 && rows >= 256) {
                bf16_amx_ctx c = { Y, W, pXb, Xb, rows, cols, B };
                qwen_parallel((size_t)nt, bf16_amx_task, &c);
            } else {
                bf16_matmat_amx_slice(Y, W, pXb, Xb, 0, rows, cols, B);
            }
            /* scratch per-thread: niente da liberare */ free(Xb);
            return;
        }
        /* scratch per-thread: niente da liberare */ free(Xb);
    }
#endif
#if defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC)
    /* Native bf16 GEMM (BFMMLA). Transpose+truncate X once ([cols][B] f32 →
     * [B][cols] bf16 bits) then 2x2 MMLA tiles. QWEN_NO_BFMMLA=1 opts out. */
    {
        /* Gate (kill switch, Apple default-off, B window) is now in qwen_mm_use(). */
        if (qwen_mm_use(QWEN_MMK_BF16_BFMMLA, B, rows, cols)) {
            uint16_t *Xb = (uint16_t *)malloc((size_t)B * cols * sizeof(uint16_t));
            if (Xb) {
                for (int b = 0; b < B; b++)
                    for (int k = 0; k < cols; k++) {
                        uint32_t u; memcpy(&u, &X[(size_t)k * B + b], 4);
                        Xb[(size_t)b * cols + k] = (uint16_t)(u >> 16);   /* truncate, like the KV */
                    }
                if (nt > 1 && rows >= 256) {
                    bfmmla_ctx c = { Y, W, Xb, rows, cols, B };
                    qwen_parallel((size_t)nt, bfmmla_task, &c);
                } else {
                    bf16_matmat_bfmmla_slice(Y, W, Xb, 0, rows, cols, B);
                }
                free(Xb);
                return;
            }
        }
    }
#endif
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
    MMSTAT(QWEN_MMK_INT8_F32TWIN, r1 - r0, cols, B);   /* batched-path audit: MACs by kernel */
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
    MMSTAT(QWEN_MMK_INT8_SDOT, r1 - r0, cols, B);   /* batched-path audit: MACs by kernel */
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

/* Quantize column b of the [cols][B] activation matrix X to int8 (per-column absmax).
 * Plain C — shared by the x86 VNNI matmat AND the ARM i8mm SMMLA matmat (rental-prep),
 * so it lives OUTSIDE the ISA guards (it was VNNI-guarded; check-isa caught the ARM use). */
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
#if defined(__AMX_INT8__) && defined(__AMX_TILE__)
/* ── int8 batched matmat on Intel AMX (Sapphire Rapids c3 / Emerald Rapids c4) ────
 *
 * WHY THIS AND NOT "MORE SIMD". On our load, SIMD width buys almost nothing (AVX2 over
 * scalar +6%, VNNI over widen+FMA +5%) because the bottleneck is reading the weights,
 * not multiplying them (PLAN T5.6). Batching is supposed to fix exactly that — read the
 * weight block ONCE for B concurrent requests — but it only pays where an integer
 * matrix-MATRIX primitive exists: measured on M1, int8 batched is 1.12x and int4 0.97x
 * against B x matvec, versus 1.70x for bf16, which has one. AMX is the only real one on
 * x86, and it is a whole category up from VNNI: one _tile_dpbssd does 16x16x64 MACs.
 * Our B tops out at 16 (--batch-size), which is exactly one tile edge. If the x86
 * concurrency ceiling moves at all, it moves here.
 *
 * THE ORIENTATION, AND WHY IT IS THE OPPOSITE OF THE OBVIOUS ONE. _tile_dpbssd(C,A,B)
 * computes C[m][n] += sum_k A[m][4k+q] * B[k][4n+q]: operand A is plain row-major
 * (M rows x K bytes, loaded straight from memory with a row stride), operand B must be
 * VNNI-packed (K/4 rows, with 4 consecutive k values interleaved per output column).
 * The tempting mapping puts the ACTIVATIONS in A (M = B = 16, "B=16 fills a tile") and
 * the WEIGHTS in B — and then every weight byte has to be shuffled into the packed
 * layout on every call. That is rows*cols bytes of repack per call to feed rows*cols
 * bytes of math: the repack, not the tile unit, sets the pace (see the cost note below).
 * So we map it the other way round:
 *
 *     A = weights  : M = 16 weight rows x K = 64 bytes, _tile_loadd straight from
 *                    W + r*cols + k with stride `cols`. NO REPACK AT ALL.
 *     B = qXt      : packed once per call into pXt, K/4 = 16 rows x 4*Bn bytes.
 *     C = [16][Bn] int32, which is already Y's [rows][B] row-major layout.
 *
 * The repack now costs O(B*cols) per call instead of O(rows*cols), and it is amortized
 * over every row block of the whole matmat. That is why the packing happens in the
 * dispatcher (once, before threading) and not in the slice.
 *
 * WHAT THIS COSTS TODAY, AND WHAT A5 WOULD STILL BUY (PLAN T5.5 A5, "repack weights into
 * the kernel's layout at load"). Concretely, for rows=cols=1024, B=16: the pack moves
 * B*cols/4 = 4096 dwords (~2k cycles), while the tile math is (rows/16)*(cols/64) = 1024
 * _tile_dpbssd at ~16 cycles throughput (~16k cycles) — roughly 12% overhead, and it
 * SHRINKS as rows grows. Had we packed the weights instead it would have been 16 rows x
 * 16 dwords = 256 dword moves per (16-row, 64-k) block against ONE _tile_dpbssd, i.e.
 * an order of magnitude more shuffling than math — that path is not worth writing.
 * A5 therefore is NOT needed to make AMX correct or even competitive here; what it would
 * still buy is (i) killing the 16-row strided tile load (each tile load touches 16
 * separate cache lines `cols` bytes apart — fine while a row block stays in L2, worse as
 * cols grows) by storing weights pre-blocked as contiguous 16x64 tiles, and (ii) padding
 * cols to a multiple of 64 so the scalar k-tail disappears. And it confirms the reason
 * A5 is a format decision, not a kernel one: the optimal on-disk layout differs between
 * ARM SMMLA (2x2 int8 tiles), VNNI (flat rows) and AMX (16x64 blocks), so the
 * pre-quantized artifact has to declare which one it holds.
 *
 * TWO ACCUMULATORS. _tile_dpbssd has ~52-cycle latency against ~16-cycle throughput on
 * SPR, and accumulating over k into ONE C tile is a dependent chain — it would run at
 * a third of peak. Two independent row blocks (r..r+15 and r+16..r+31) sharing the same
 * activation tile interleave those chains. If the box says we are still latency-bound,
 * 4 accumulators is the next knob (tiles are free: we use 5 of 8).
 *
 * ⚠️ WRITTEN ON M1, NEVER EXECUTED. M1 has no AMX and Rosetta 2 emulates neither AVX-512
 * nor AMX, so nothing here has ever run: correctness is a forced cross-compile
 * (`clang -target x86_64-apple-macos13 -march=sapphirerapids -fsyntax-only`) plus review
 * of the arithmetic against the TDPBSSD pseudocode. No performance number is claimed.
 * What would falsify the whole idea, in order, on the first c3/c4 box:
 *   1. `--caps` says AMX ACTIVE (if it says permission DENIED, the kernel is < 5.16 and
 *      every number below is a VNNI number);
 *   2. `--self-test` matches the scalar oracle (a wrong pack shows up as garbage, not
 *      as a small error);
 *   3. `--matmat-bench` int8: AMX must beat both B x matvec AND the VNNI twin
 *      (QWEN_NO_AMX=1 gives the VNNI arm without recompiling). If it does not beat VNNI,
 *      say so and keep VNNI — AMX needs a large M to repay the tile loads, and that is
 *      a prediction, not a measurement. */
/* qXt[B][cols] -> pXt, the VNNI-style quadruple-interleaved operand-B layout:
 *   pXt[kc][j][4n+q] = qXt[n][kc*32 + 4j + q]     (kc = 32-value k chunk, j = 0..7)
 * Each chunk is an 8-row tile with row stride 4*B bytes, so the whole buffer is exactly
 * kpack*B bytes. Called ONCE per matmat, before threading.
 *
 * WHY THE CHUNK IS 32 AND NOT 64, when the int8 kernel consumes 64 k per tile: the two
 * layouts COINCIDE. A 64-k chunk is a 16-row tile whose rows 0..7 hold k 0..31 and rows
 * 8..15 hold k 32..63 — which is byte-for-byte two consecutive 32-k chunks. Packing in
 * 32s therefore serves the int8 kernel (which reads 16-row tiles at stride 64*B) AND the
 * q4 kernel (which needs an 8-row tile per 32-weight Q4_0 block, because the per-block
 * scale forbids accumulating two blocks into one C tile) from ONE buffer and one loop.
 * `kpack` is the number of k values to pack and MUST be a multiple of 32 and <= cols;
 * the caller passes what its kernel will actually read (int8: cols & ~63, q4: cols),
 * and it is also the size in bytes of pXt divided by B — get it wrong and it overruns. */
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

static void int8_matmat_amx_slice(float *Y, const int8_t *W, const float *scale,
                                  const int8_t *pXt, const int8_t *qXt, const float *sx,
                                  int r0, int r1, int cols, int B) {
    MMSTAT(QWEN_MMK_INT8_AMX, r1 - r0, cols, B);   /* batched-path audit: MACs by kernel */
    const int kfull   = cols & ~63;          /* AMX covers whole 64-byte k chunks */
    const int nchunk  = kfull >> 6;
    const int cstride = B * 4;               /* bytes per row of a packed / C tile */

    /* Tile shapes are set ONCE per slice, never per row: TILECFG is architectural state
     * and _tile_loadconfig is expensive (it invalidates all tiles). Released at the end
     * so the thread stops carrying 8 KB of live XSAVE state into the next kernel. */
    qwen_amx_tilecfg cfg;
    memset(&cfg, 0, sizeof cfg);
    cfg.palette_id = 1;
    cfg.rows[0] = 16; cfg.colsb[0] = (uint16_t)cstride;   /* tmm0: C, rows r..r+15   */
    cfg.rows[1] = 16; cfg.colsb[1] = (uint16_t)cstride;   /* tmm1: C, rows r+16..+31 */
    cfg.rows[2] = 16; cfg.colsb[2] = 64;                  /* tmm2: W rows r..r+15    */
    cfg.rows[3] = 16; cfg.colsb[3] = 64;                  /* tmm3: W rows r+16..+31  */
    cfg.rows[4] = 16; cfg.colsb[4] = (uint16_t)cstride;   /* tmm4: packed activations */
    _tile_loadconfig(&cfg);

    /* 16 rows x at most 16 int32 columns per accumulator; stride is cstride bytes. */
    int32_t cbuf[2][16 * 16] __attribute__((aligned(64)));

    int r = r0;
    for (; r + 31 < r1; r += 32) {
        _tile_zero(0); _tile_zero(1);
        for (int kc = 0; kc < nchunk; kc++) {
            _tile_loadd(4, pXt + (size_t)kc * 64 * (size_t)B, cstride);
            _tile_loadd(2, W + (size_t)r * cols + (size_t)kc * 64, cols);
            _tile_loadd(3, W + (size_t)(r + 16) * cols + (size_t)kc * 64, cols);
            _tile_dpbssd(0, 2, 4);     /* two independent chains -> latency hidden */
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
            _tile_loadd(2, W + (size_t)r * cols + (size_t)kc * 64, cols);
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
    _tile_release();

    /* Row tail (< 16 rows left in this slice): scalar. Slices are cut on 16-row
     * boundaries, so this only fires when `rows` itself is not a multiple of 16. */
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
typedef struct {
    float *Y; const int8_t *W; const float *scale;
    const int8_t *pXt; const int8_t *qXt; const float *sx;
    int rows, cols, B;
} int8_amx_ctx;
static void int8_amx_task(size_t tid, size_t nt, void *vc) {
    int8_amx_ctx *c = (int8_amx_ctx *)vc;
    int r0 = (int)(tid * (size_t)c->rows / nt);
    int r1 = (int)((tid + 1) * (size_t)c->rows / nt);
    r0 &= ~15; if (tid + 1 < nt) r1 &= ~15;   /* keep 16-row tiles inside one slice */
    int8_matmat_amx_slice(c->Y, c->W, c->scale, c->pXt, c->qXt, c->sx, r0, r1, c->cols, c->B);
}

/* ── q4_0 batched matmat on AMX (nibbles decoded to int8, then the int8 tile machine) ──
 *
 * THE COST ACCOUNTING FIRST, because this kernel is the one that could reasonably not
 * have been written, and "we did the arithmetic" has to be checkable.
 *
 * The structural problem is the Q4_0 format, not AMX: the fp16 scale is per (row,
 * 32-weight block), so 32 k values is the LONGEST run that may be accumulated inside one
 * C tile. The int8 kernel configures A with colsb=64 and reduces over the entire `cols`
 * before it ever stores C — one TILESTORED per 16-row block for the whole matmat. Here
 * the tile has to be drained and rescaled every 32 k. That is exactly the cost that made
 * q4-VNNI ~37% SLOWER than int8-VNNI on EPYC while reading half the bytes.
 *
 * So, per (16 weight rows x one 32-weight block), on a 3 GHz SPR core, order of magnitude:
 *   - nibble unpack, 16 rows x (load + and + shift + and + 2 unpack + store) ~ 6 uops/row
 *     ~ 96 uops   ~ 20-25 cycles   [independent of B — this is the part batching amortizes]
 *   - 1 TDPBSSD at K=32 (half the int8 depth)              ~ 8 cycles
 *   - 1 TILESTORED (16 x B int32) + rescale: B/16 zmm x (load, cvtdq2ps, fma) per row
 *     block, i.e. ~48 uops at B=16                          ~ 16-25 cycles
 *   => ~50 cycles for 16*B*32 = 512*B MACs. At B=16 that is ~180 MAC/cycle.
 * Against the two things it has to beat:
 *   - q4-VNNI, the incumbent on this silicon: its inner loop does ONE _mm512_dpbusd
 *     (32 useful MACs, the operand is only half-full) followed by a FULL
 *     _mm512_reduce_add_epi32 — a horizontal reduce per (row, block, column), ~10 uops.
 *     16 rows x B columns x ~10 uops => ~40*B cycles for the same 512*B MACs, i.e.
 *     ~13 MAC/cycle. AMX is an order of magnitude ahead here, and the reason is not tile
 *     width: it is that the tile amortizes the reduce over 16 rows AND B columns at once.
 *   - int8 on AMX, which pays no unpack and drains C once per matmat: ~1000 MAC/cycle,
 *     ~5x more compute — but it reads 2x the weight bytes. So the q4-vs-int8 verdict is a
 *     ROOFLINE question, not a kernel question: q4 wins only where the matmat is
 *     bandwidth-bound, which for the talker projections at these sizes it should be.
 * VERDICT: worth writing, and NOT worth assuming. The honest form of the claim is "the
 * per-block drain costs ~5x of AMX peak and it still beats VNNI on paper" — the box says
 * whether the paper is right.
 *
 * HOW THE PACKED ACTIVATION IS SHARED WITH THE INT8 PATH. amx_pack_act_int8() packs in
 * 32-k chunks of 8 rows; two of them are byte-identical to one 64-k chunk of 16 rows
 * (see the note there). So block `bl` reads its operand-B tile at pXt + bl*32*B with
 * rows=8, and no second packing pass exists.
 *
 * THE -8 OFFSET is folded exactly as VNNI and SMMLA do it: the nibbles stay unsigned
 * 0..15 (which is a valid signed int8, so TDPBSSD is exact), and corr[b][bl] = -8 * sum
 * of the block's activations is added to the integer dot before the per-block scale.
 * The lo/hi unpack is the same one q4_matmat_vnni_slice and q4_matmat_avx2_slice use, so
 * all four q4 paths agree bit-for-bit on which weight multiplies which activation.
 *
 * TWO ACCUMULATORS, PAIRED OVER BLOCKS, not over row blocks. TDPBSSD latency (~52c) far
 * exceeds its throughput (~16c) and here EVERY block ends in a store, so the dependent
 * chain is TDPBSSD -> TILESTORED -> rescale. Processing blocks in pairs on tmm0/tmm1
 * lets block bl+1's multiply run under block bl's drain. (The int8 kernel pairs ROW
 * blocks instead, because there the drain happens once and the chain is over k.)
 *
 * ⚠️ WRITTEN ON M1, NEVER EXECUTED — same status as the int8 and bf16 tile kernels.
 * What would falsify it, in order, on the first c3/c4:
 *   1. `--caps` says `x86 amx int8: AMX ACTIVE` (this kernel rides the int8 tile unit);
 *   2. `--self-test` / `make check-matmat-parity` on q4: the integer path has no
 *      summation order to argue about, so the expected error against the integer
 *      reference is EXACTLY zero. Anything else is a pack or offset bug, not noise;
 *   3. `--matmat-bench` q4 at -j1 and at full threads, QWEN_NO_AMX_Q4=1 as the A/B arm:
 *      AMX-q4 must beat q4-VNNI. If it does not, the cost model above is wrong in the
 *      drain term and the next move is NOT more SIMD — it is accumulating in int32
 *      across blocks under a shared scale, i.e. a format change (q4 with a per-row
 *      scale, or q8 blocks), which is a PLAN A5 decision;
 *   4. q4-AMX vs int8-AMX on the same shape: if int8 wins, the matmat is compute-bound
 *      at that size and q4's halved byte count buys nothing — report the shape, because
 *      that is the roofline crossover we actually need for the server. */
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
    MMSTAT(QWEN_MMK_Q4_AMX, r1 - r0, cols, B);   /* batched-path audit: MACs by kernel */
    const int nb      = cols / Q4_0_BLOCK_SIZE;
    const int cstride = B * 4;               /* bytes per C row and per packed-B row */

    qwen_amx_tilecfg cfg;
    memset(&cfg, 0, sizeof cfg);
    cfg.palette_id = 1;
    cfg.rows[0] = 16; cfg.colsb[0] = (uint16_t)cstride;   /* tmm0: C, even block */
    cfg.rows[1] = 16; cfg.colsb[1] = (uint16_t)cstride;   /* tmm1: C, odd  block */
    cfg.rows[2] = 16; cfg.colsb[2] = 32;                  /* tmm2: A, 16 rows x 32 int8 */
    cfg.rows[3] = 16; cfg.colsb[3] = 32;                  /* tmm3: A, odd block         */
    cfg.rows[4] =  8; cfg.colsb[4] = (uint16_t)cstride;   /* tmm4: B, K/4 = 8 rows      */
    cfg.rows[5] =  8; cfg.colsb[5] = (uint16_t)cstride;   /* tmm5: B, odd block         */
    _tile_loadconfig(&cfg);

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
        for (; bl < nb; bl++) {                  /* odd block count: one last block */
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
    _tile_release();

    /* Row tail (< 16 rows left in this slice): scalar, same value-order decode. Slices
     * are cut on 16-row boundaries, so this only fires when `rows` is not a multiple of 16. */
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
    r0 &= ~15; if (tid + 1 < nt) r1 &= ~15;   /* keep 16-row tiles inside one slice */
    q4_matmat_amx_slice(c->Y, c->W, c->pXt, c->qXt, c->sx, c->corr, r0, r1, c->cols, c->B);
}

#endif /* __AMX_INT8__ && __AMX_TILE__ */

#if defined(__AVX512VNNI__)
/* ── int8 VNNI batched matmat (the x86 int8 GEMM the SDOT comment above asks for) ──
 * Y[rows,B] = (W_int8 @ qXt^T)·scales. Weight-stationary: each 64-int8 W block is
 * loaded ONCE and dpbusd'd against all B pre-quantized activation columns. VNNI is
 * unsigned×signed → activations passed pre-offset as ua = qXt+128 (unsigned), corrected
 * −128·Σw per row (ws = dpbusd(ones,w)), exactly like the single-stream int8_matvec_vnni. */
static void int8_matmat_vnni_slice(float *Y, const int8_t *W, const float *scale,
                                   const int8_t *qXt, const float *sx,
                                   int r0, int r1, int cols, int B) {
    MMSTAT(QWEN_MMK_INT8_VNNI, r1 - r0, cols, B);   /* batched-path audit: MACs by kernel */
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

#if defined(__AVX2__)
/* ── int8 batched matmat for AVX2 WITHOUT VNNI (PLAN 0.nonies S1) ────────────────
 *
 * WHY THIS EXISTS. The dispatcher had a true int8 GEMM for AVX-512/VNNI (vpdpbusd)
 * and for ARM i8mm (smmla), and NOTHING for plain AVX2 — which is the configuration
 * of most rentable VPS (VNNI needs Zen4/Ice Lake+; GCP n2d/c2d are Zen2/3, e2 and n1
 * older still). There the batched path fell through to the generic f32 twin, so B
 * concurrent requests re-read the weights B times and concurrency did not scale —
 * the same non-scaling measured on M1, and for the same reason: no matrix-matrix
 * integer primitive. See PLAN T5.6.
 *
 * THE ARITHMETIC. AVX2's integer dot is maddubs(u8,s8)->s16 + madd(s16,1)->s32, and
 * maddubs SATURATES at int16. With the activation offset to unsigned (the vpdpbusd
 * trick) the products reach 255*127*2 = 64770 and saturate. So we use the other
 * decomposition: |w| as the UNSIGNED operand and a*sign(w) as the signed one
 *   |w|*(a*sign w) == w*a,  and  128*127*2 = 32512 < 32767  -> exact, never saturates.
 * (a = -128 cannot occur: quantize_act_int8_col scales by 127/absmax.) This is the
 * same "unsigned code x signed activation" shape as the q6_0 AVX2 kernel; there the
 * codes are naturally unsigned, here abs/sign makes them so.
 *
 * WEIGHT-STATIONARY, IN BLOCKS OF 4 COLUMNS. The whole point of batching is to read
 * each weight block once and reuse it across B activations, so the b-loop is INSIDE
 * the k-loop. But AVX2 has 16 ymm registers: with B accumulators live the compiler
 * spills, and a spilled accumulator gives back exactly what the batching bought. So
 * columns are processed 4 at a time with 4 named accumulators (fully unrolled, no
 * indexed array), and the weights are re-read ceil(B/4) times instead of B times —
 * a 4x cut in weight traffic at B>=4, which is the regime a server runs in.
 * A last block with fewer than 4 columns computes 4 anyway (spare lanes point at
 * column b0) and discards the extra: branch-free inner loop, at most 3 wasted
 * columns once per row.
 *
 * ⚠️ WRITTEN ON M1, NEVER EXECUTED. Correctness is checked by `make check-isa`
 * (x86-64-v3 cross-compile) and by --self-test on the box; the SPEED is a hypothesis
 * until `./qwen_tts --matmat-bench` runs on an AVX2-only machine. Declared criterion
 * (PLAN 0.nonies S1): int8 must go from ~1.0x to >=1.5x vs B x matvec. Below 1.2x the
 * kernel did not fix it and the diagnosis has to be redone (bandwidth already
 * saturated -> nothing to share). QWEN_NO_AVX2MM=1 opts out for the A/B. */
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
    MMSTAT(QWEN_MMK_INT8_AVX2, r1 - r0, cols, B);   /* batched-path audit: MACs by kernel */
    const __m256i ones16 = _mm256_set1_epi16(1);
    for (int r = r0; r < r1; r++) {
        const int8_t *w = W + (size_t)r * cols;
        const float s = scale[r];
        for (int b0 = 0; b0 < B; b0 += 4) {
            const int bn = (B - b0 < 4) ? (B - b0) : 4;
            /* spare lanes alias column b0: computed, then discarded */
            const int8_t *p0 = qXt + (size_t)b0 * cols;
            const int8_t *p1 = qXt + (size_t)(bn > 1 ? b0 + 1 : b0) * cols;
            const int8_t *p2 = qXt + (size_t)(bn > 2 ? b0 + 2 : b0) * cols;
            const int8_t *p3 = qXt + (size_t)(bn > 3 ? b0 + 3 : b0) * cols;
            __m256i a0 = _mm256_setzero_si256(), a1 = _mm256_setzero_si256();
            __m256i a2 = _mm256_setzero_si256(), a3 = _mm256_setzero_si256();
            int k = 0;
            for (; k + 32 <= cols; k += 32) {
                __m256i wv = _mm256_loadu_si256((const __m256i *)(w + k));   /* read once... */
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
                for (int kk = k; kk < cols; kk++) sum += (int)w[kk] * (int)qb[kk];   /* tail */
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
#endif /* __AVX2__ */

#if defined(__ARM_FEATURE_MATMUL_INT8)
/* i8mm SMMLA int8 GEMM slice (see the dispatch note in qwen_matmat_int8).
 *
 * ── Why this kernel is 4 rows x 4 columns and not the obvious 2x2 (2026-08-21) ──
 * One SMMLA is a 2x2x8 tile, so the natural loop is "one accumulator, k += 8". That
 * shape is what this kernel used until today, and on a Neoverse-V2 (Axion, Graviton4)
 * it is bound by the INSTRUCTION LATENCY, not by memory: every iteration feeds acc back
 * into the next SMMLA, and the unit sits idle waiting for it. AMX hides this in its tile
 * registers; ARM does not, which is the whole reason the bf16 twin measured 0.73-1.02x
 * on the box while int8 measured 1.6-2.3x (PLAN, ARM epic S1).
 *
 * So the block is 4x4: FOUR independent accumulator chains fed from the same two W and
 * two X loads. Same arithmetic, same integer-exact result (check-matmat-parity must stay
 * at max_abs = 0), ~2x fewer weight loads per MAC, and the dependency stall is amortised
 * across four instructions instead of hitting every one.
 * Register budget: 4 accumulators + 2 A operands + 2 B operands = 8 of 32 NEON registers,
 * so there is headroom left for an 8x4 variant if this one pays.
 * The 2x2 loop is kept below, unchanged, for the row and column remainders. */
static void int8_matmat_smmla_slice(float *Y, const int8_t *W, const float *scale,
                                    const int8_t *qXt, const float *sx,
                                    int r0, int r1, int cols, int B) {
    MMSTAT(QWEN_MMK_INT8_SMMLA, r1 - r0, cols, B);   /* batched-path audit: MACs by kernel */
    int r = r0;
    /* ── main block: 4 rows x 4 columns, four independent SMMLA chains ── */
    for (; r + 3 < r1 && B >= 4; r += 4) {
        const int8_t *w0 = W + (size_t)r * cols,       *w1 = W + (size_t)(r + 1) * cols;
        const int8_t *w2 = W + (size_t)(r + 2) * cols, *w3 = W + (size_t)(r + 3) * cols;
        int j = 0;
        for (; j + 3 < B; j += 4) {
            const int8_t *x0 = qXt + (size_t)j * cols,       *x1 = qXt + (size_t)(j + 1) * cols;
            const int8_t *x2 = qXt + (size_t)(j + 2) * cols, *x3 = qXt + (size_t)(j + 3) * cols;
            int32x4_t a00 = vdupq_n_s32(0), a01 = vdupq_n_s32(0);   /* rows r,r+1 x cols j,j+1 / j+2,j+3 */
            int32x4_t a10 = vdupq_n_s32(0), a11 = vdupq_n_s32(0);   /* rows r+2,r+3 x the same two pairs */
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
            for (; k < cols; k++) {                  /* k tail (cols%8; not hit by model dims) */
                int32_t wv[4] = { w0[k], w1[k], w2[k], w3[k] };
                int32_t xv[4] = { x0[k], x1[k], x2[k], x3[k] };
                t00[0] += wv[0]*xv[0]; t00[1] += wv[0]*xv[1]; t00[2] += wv[1]*xv[0]; t00[3] += wv[1]*xv[1];
                t01[0] += wv[0]*xv[2]; t01[1] += wv[0]*xv[3]; t01[2] += wv[1]*xv[2]; t01[3] += wv[1]*xv[3];
                t10[0] += wv[2]*xv[0]; t10[1] += wv[2]*xv[1]; t10[2] += wv[3]*xv[0]; t10[3] += wv[3]*xv[1];
                t11[0] += wv[2]*xv[2]; t11[1] += wv[2]*xv[3]; t11[2] += wv[3]*xv[2]; t11[3] += wv[3]*xv[3];
            }
            /* lanes of vmmlaq_s32(acc, [rA|rB], [cP|cQ]) = [rA.cP, rA.cQ, rB.cP, rB.cQ] */
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
        for (; j < B; j++) {                          /* column remainder, still 4 rows at a time */
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
    /* ── remainder: the original 2x2 tile, unchanged ── */
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
                acc = vmmlaq_s32(acc, a, b);   /* lanes: [r·c, r·c1, r1·c, r1·c1] */
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
    r0 &= ~3; if (tid + 1 < nt) r1 &= ~3;   /* keep 4-row blocks within one slice (both ends round
                                             * the same way, so adjacent slices still meet exactly) */
    int8_matmat_smmla_slice(c->Y, c->W, c->scale, c->qXt, c->sx, r0, r1, c->cols, c->B);
}
#endif /* __ARM_FEATURE_MATMUL_INT8 */

void qwen_matmat_int8(float *Y, const int8_t *W, const float *scale,
                      const float *X, int rows, int cols, int B) {
    qwen_census_op("matmat_int8", rows, cols, B);
    if (B <= 0) return;
    /* Engine layout, so this transposes at B>1. qwen_batch_proj_q calls the _native
     * entry instead and pays neither transpose; this hook exists for the call sites
     * that still hand over [cols,B]. */
    if (kai_i8_try(Y, W, scale, X, rows, cols, B)) {
        MMSTAT(B > 1 ? QWEN_MMK_KLEIDI_I8 : QWEN_MMK_KLEIDI_I8_GEMV, rows, cols, B);
        return;
    }
    if (B > 64) B = 64;
#if defined(__AMX_INT8__) && defined(__AMX_TILE__)
    /* AMX FIRST: it is the biggest primitive on x86 (16x16x64 MACs per instruction vs
     * VNNI's 16x4), so where it exists it should get the work; VNNI below stays the
     * fallback and the A/B arm (QWEN_NO_AMX=1, no recompile).
     *
     * Gates, all of them deliberate and all of them to be re-measured on the box:
     *  - qwen_amx_int8_ready(): CPU has AMX-INT8 *and* Linux granted XTILEDATA. Without
     *    this the first _tile_loadd is SIGILL, so a "compiled with AMX" binary must be
     *    able to run on a 5.15 kernel and simply be slower. Called HERE, on the calling
     *    thread, before qwen_parallel hands slices to the pool. This is a CAPABILITY,
     *    not a threshold, which is why it stays out of qwen_mm_use() and out of reach
     *    of any env override.
     *  - the shape window (B >= 4, rows >= 32, cols >= 64) and the kill switches moved
     *    into qwen_mm_use() — see the reasoning next to the gate table above; those are
     *    guesses to be re-cut on the box, and QWEN_AMX_MIN_B moves the B crossover
     *    without a rebuild. */
    {
        if (qwen_mm_use(QWEN_MMK_INT8_AMX, B, rows, cols) && qwen_amx_int8_ready()) {
            const size_t kfull = (size_t)(cols & ~63);
            int8_t *qXt = mm_scratch_qx((size_t)B * cols);
            int8_t *pXt = NULL;
            /* 64-byte aligned: _tile_loadd does not require it but crossing a cache line
             * per tile row costs, and the packed buffer is read once per row block. */
            if (qXt) pXt = mm_scratch_pack(kfull * (size_t)B);
            if (qXt && pXt) {
                float sx[16];
                for (int b = 0; b < B; b++)
                    sx[b] = quantize_act_int8_col(qXt + (size_t)b * cols, X, cols, B, b);
                amx_pack_act_int8(pXt, qXt, cols, (int)kfull, B);   /* once per call, not per slice */
                int nt = g_n_threads;
                if (nt > 1 && rows >= 256) {
                    int8_amx_ctx c = { Y, W, scale, pXt, qXt, sx, rows, cols, B };
                    qwen_parallel((size_t)nt, int8_amx_task, &c);
                } else {
                    int8_matmat_amx_slice(Y, W, scale, pXt, qXt, sx, 0, rows, cols, B);
                }
                /* scratch per-thread: niente da liberare */
                return;
            }
            /* scratch per-thread: niente da liberare */
        }
    }
#endif
#if defined(__AVX512VNNI__)
    /* x86 VNNI batched int8 GEMM — default ON (QWEN_NO_VNNI=1 disables), the right
     * int8 matmat primitive on x86 (the ARM SDOT path below stays M1 opt-in). */
    {
        if (qwen_mm_use(QWEN_MMK_INT8_VNNI, B, rows, cols)) {
            int8_t *qXt = mm_scratch_qx((size_t)B * cols);
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
                /* scratch per-thread: niente da liberare */
                return;
            }
        }
    }
#endif

#if defined(__AVX2__)
    /* AVX2 without VNNI (GCP n2d/c2d, e2, n1 — the most rented class): the real int8
     * GEMM lives above only on VNNI silicon, so without this block B concurrent
     * requests fall through to the generic f32 twin and read the weights B times.
     * Reached on a VNNI build only with QWEN_NO_VNNI=1, which makes it the A/B arm.
     * QWEN_NO_AVX2MM=1 disables it (falls through to the old behaviour). */
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
                /* scratch per-thread: niente da liberare */
                return;
            }
        }
    }
#endif
#if defined(__ARM_FEATURE_MATMUL_INT8)
    /* i8mm SMMLA batched path — the TRUE int8 GEMM primitive the SDOT note below
     * asks for (M2+/Neoverse-V1+/Graviton3+; compile-guarded, absent on M1).
     * 2x2 tiles: vmmlaq_s32 does 2 rows × 2 cols × 8-deep per instruction.
     * Reuses the per-column activation quant. QWEN_NO_SMMLA=1 opts out. */
    {
        /* Kill switch + the Apple default-off (int8-SMMLA LOSES on Apple cores, wins
         * 2.1x on Graviton3; the q4-SMMLA twin stays ON, it wins on both) now live in
         * the gate table above, so all of it is visible in one place. */
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
                /* scratch per-thread: niente da liberare */
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
                /* scratch per-thread: niente da liberare */
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
    MMSTAT(QWEN_MMK_Q4_GENERIC, r1 - r0, cols, B);   /* batched-path audit: MACs by kernel */
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
    MMSTAT(QWEN_MMK_Q4_VNNI, r1 - r0, cols, B);   /* batched-path audit: MACs by kernel */
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
#endif /* __AVX512VNNI__ */

#if defined(__AVX2__)
/* ── q4_0 batched matmat for AVX2 WITHOUT VNNI (PLAN 0.nonies S1, q4 twin) ───────
 * Same gap as the int8 twin above, same machines. Here the nibbles are ALREADY
 * unsigned (0..15), which is what maddubs wants, so no abs/sign trick is needed and
 * saturation is impossible (15*127*2 = 3810 « 32767): the −8 offset is folded into
 * the per-block activation sums (corr), exactly as the VNNI and SMMLA twins do.
 * Nibbles are decoded to VALUE order with the same lo/hi unpack as q4_matmat_vnni_slice
 * so all three paths agree bit-for-bit on which weight multiplies which activation.
 *
 * ⚠️ One structural cost is inherited from the VNNI twin and is worth stating: the
 * per-block fp16 scale forces a horizontal reduce per (row, block, column) instead of
 * one per row. That is precisely what made q4-VNNI ~37% SLOWER than int8 on EPYC
 * despite half the bytes. So on an AVX2 box q4 may well lose to int8 here too — the
 * measurement decides, and if it does lose, the fix is accumulating in int32 across
 * blocks with a shared scale, not more SIMD width. WRITTEN ON M1, NEVER EXECUTED. */
static void q4_matmat_avx2_slice(float *Y, const q4_0_block_t *W, const int8_t *qXt,
                                 const float *sx, const int *corr,
                                 int r0, int r1, int cols, int B) {
    MMSTAT(QWEN_MMK_Q4_AVX2, r1 - r0, cols, B);   /* batched-path audit: MACs by kernel */
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
                                          _mm_unpacklo_epi8(lo, hi));   /* read once... */
            float scl = qwen_f16_to_f32(row[bl].scale_f16);
            for (int b = 0; b < B; b++) {                                /* ...reused across B */
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
#endif /* __AVX2__ */
#if defined(__ARM_FEATURE_MATMUL_INT8)
/* q4×q8 SMMLA GEMM slice: nibbles decoded to VALUE order via the same vzip the
 * SDOT matvec uses (kept as raw 0..15, the −8 offset folded in via the per-block
 * activation sums corr), then 2×2 vmmlaq_s32 tiles, per-block fp16 weight scale
 * applied at block granularity, per-column activation scale at the end. */
static void q4_matmat_smmla_slice(float *Y, const q4_0_block_t *W,
                                  const int8_t *qXt, const float *sx, const int *corr,
                                  int r0, int r1, int cols, int B) {
    MMSTAT(QWEN_MMK_Q4_SMMLA, r1 - r0, cols, B);   /* batched-path audit: MACs by kernel */
    int nb = cols / Q4_0_BLOCK_SIZE;
    const uint8x16_t mask = vdupq_n_u8(0x0F);
    int r = r0;
    for (; r + 1 < r1; r += 2) {
        const q4_0_block_t *w0 = W + (size_t)r * nb, *w1 = W + (size_t)(r + 1) * nb;
        /* ── blocks OUTER, columns INNER: unpack each weight block ONCE (2026-08-21) ──
         * With the column loop outside, the same 32 nibbles were widened to int8 once per
         * PAIR OF COLUMNS: B/2 times, four times over at B=8. That is the whole reason the
         * q4 twin measured 1.45-1.57x on the box while int8 - reading TWICE the bytes -
         * measured 2.19x. Same shape of defect as the q4-VNNI path on EPYC, which came out
         * 37% slower than int8 while reading half the memory.
         * The float accumulation order per (row, column) is unchanged - still block 0..nb-1
         * in sequence - so the result is bit-identical, not merely close.
         * B is capped at 16 by the gate table (max_b, which exists to bound exactly this
         * kind of stack array); anything wider falls through to the original loop below. */
        if (B <= 16) {
            float f0[16], f1[16];
            for (int c = 0; c < B; c++) { f0[c] = 0.0f; f1[c] = 0.0f; }
            for (int bl = 0; bl < nb; bl++) {
                uint8x16_t raw0 = vld1q_u8(w0[bl].qs), raw1 = vld1q_u8(w1[bl].qs);
                uint8x16x2_t z0 = vzipq_u8(vandq_u8(raw0, mask), vshrq_n_u8(raw0, 4));
                uint8x16x2_t z1 = vzipq_u8(vandq_u8(raw1, mask), vshrq_n_u8(raw1, 4));
                int8x16_t a0lo = vreinterpretq_s8_u8(z0.val[0]), a0hi = vreinterpretq_s8_u8(z0.val[1]);
                int8x16_t a1lo = vreinterpretq_s8_u8(z1.val[0]), a1hi = vreinterpretq_s8_u8(z1.val[1]);
                /* the four A operands the tiles need, built once for every column below */
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
                for (; c < B; c++) {                 /* odd column, same per-block order */
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
                int8x16_t a0lo = vreinterpretq_s8_u8(z0.val[0]);  /* r   w0..15  */
                int8x16_t a0hi = vreinterpretq_s8_u8(z0.val[1]);  /* r   w16..31 */
                int8x16_t a1lo = vreinterpretq_s8_u8(z1.val[0]);  /* r+1 w0..15  */
                int8x16_t a1hi = vreinterpretq_s8_u8(z1.val[1]);  /* r+1 w16..31 */
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
        for (; j < B; j++) {                       /* odd-B col tail (scalar int dots) */
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
    for (; r < r1; r++) {                          /* odd-rows tail */
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
#endif /* __ARM_FEATURE_MATMUL_INT8 */

void qwen_matmat_q4_0(float *Y, const q4_0_block_t *W, const float *X,
                      int rows, int cols, int B) {
    qwen_census_op("matmat_q4_0", rows, cols, B);
    if (B <= 0) return;
    if (B > 64) B = 64;
    /* Same rule as the matvec above: only for weights pre-packed at load. Here the
     * i8mm GEMM (qsi8d32p4x8 x qsi4c32p4x8) runs, over the SAME packed RHS buffer -
     * the two kernels share the qsi4c32p4x8 geometry, so packing once serves both. */
    if (qwen_mm_use(QWEN_MMK_KLEIDI_Q4, B, rows, cols) &&
        qwen_kleidi_matmul_q4(Y, (const void *)W, X, rows, cols, B)) {
        MMSTAT(QWEN_MMK_KLEIDI_Q4, rows, cols, B);
        return;
    }
#if defined(__AMX_INT8__) && defined(__AMX_TILE__)
    /* AMX before VNNI, same ordering rule as int8: the biggest primitive gets the work
     * and VNNI stays the fallback AND the A/B arm (QWEN_NO_AMX_Q4=1, no rebuild).
     * qwen_amx_int8_ready() is the capability check (this kernel uses the INT8 tile unit
     * on decoded nibbles) and must run on the calling thread, before qwen_parallel. */
    {
        if (qwen_mm_use(QWEN_MMK_Q4_AMX, B, rows, cols) && cols % Q4_0_BLOCK_SIZE == 0 &&
            qwen_amx_int8_ready()) {
            int nb = cols / Q4_0_BLOCK_SIZE;
            int8_t *qXt = mm_scratch_qx((size_t)B * cols);
            int *corr = mm_scratch_corr((size_t)B * nb);
            int8_t *pXt = NULL;
            /* cols is a multiple of 32 here, so the whole activation is packed (the q4
             * kernel consumes every block; there is no 64-byte k tail to skip). */
            if (qXt && corr) pXt = mm_scratch_pack((size_t)cols * (size_t)B);
            if (qXt && corr && pXt) {
                float sx[16];
                for (int b = 0; b < B; b++) {
                    sx[b] = quantize_act_int8_col(qXt + (size_t)b * cols, X, cols, B, b);
                    const int8_t *qb = qXt + (size_t)b * cols;
                    for (int bl = 0; bl < nb; bl++) {
                        int t = 0;
                        for (int k = 0; k < Q4_0_BLOCK_SIZE; k++) t += qb[bl * Q4_0_BLOCK_SIZE + k];
                        corr[(size_t)b * nb + bl] = -8 * t;   /* same convention as the VNNI/AVX2 twins */
                    }
                }
                amx_pack_act_int8(pXt, qXt, cols, cols, B);   /* once per call, not per slice */
                int nt2 = g_n_threads;
                if (nt2 > 1 && rows >= 256) {
                    q4_amx_ctx c = { Y, W, pXt, qXt, sx, corr, rows, cols, B };
                    qwen_parallel((size_t)nt2, q4_amx_task, &c);
                } else {
                    q4_matmat_amx_slice(Y, W, pXt, qXt, sx, corr, 0, rows, cols, B);
                }
                /* scratch per-thread: niente da liberare */
                return;
            }
            /* scratch per-thread: niente da liberare */
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
                /* scratch per-thread: niente da liberare */ /* scratch per-thread: niente da liberare */
                return;
            }
            /* scratch per-thread: niente da liberare */ /* scratch per-thread: niente da liberare */
        }
    }
#endif

#if defined(__AVX2__)
    /* AVX2-without-VNNI q4 twin (see the int8 dispatch above for the why).
     * QWEN_NO_AVX2MM=1 opts out; on a VNNI build this is reached only with
     * QWEN_NO_VNNI=1, which makes it the A/B arm against vpdpbusd. */
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
                /* scratch per-thread: niente da liberare */ /* scratch per-thread: niente da liberare */
                return;
            }
            /* scratch per-thread: niente da liberare */ /* scratch per-thread: niente da liberare */
        }
    }
#endif
#if defined(__ARM_FEATURE_MATMUL_INT8)
    /* q4 SMMLA twin (rental-prep follow-up, Graviton3 2026-07-11): the scalar batch
     * below was a big LOSS on ARM (0.29× vs B×matvec on Neoverse-V1, 0.43-0.65× on
     * M1) — decode nibbles in value order (same vzip as the SDOT matvec) and feed
     * 2×2 vmmlaq_s32 tiles against the per-column int8-quantized activations, with
     * the −8 offset corrected via per-block activation sums. QWEN_NO_SMMLA=1 opts out. */
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
                /* scratch per-thread: niente da liberare */ /* scratch per-thread: niente da liberare */
                return;
            }
            /* scratch per-thread: niente da liberare */ /* scratch per-thread: niente da liberare */
        }
    }
#elif defined(__ARM_FEATURE_DOTPROD)
    /* Floor fallback (M1-class ARM, no i8mm): the scalar fixed-B batch LOSES to B
     * sequential SDOT matvecs (measured 0.43-0.65× on M1) — so just do the matvecs.
     * Column gather/scatter is noise next to the weight sweep. */
    if (cols % Q4_0_BLOCK_SIZE == 0) {
        float *xcol = (float *)malloc((size_t)cols * sizeof(float));
        float *ycol = (float *)malloc((size_t)rows * sizeof(float));
        if (xcol && ycol) {
            MMSTAT(QWEN_MMK_Q4_BMATVEC, rows, cols, B);   /* audit: this is NOT a GEMM */
            for (int b = 0; b < B; b++) {
                for (int k = 0; k < cols; k++) xcol[k] = X[(size_t)k * B + b];
                qwen_matvec_q4_0(ycol, W, xcol, rows, cols);
                for (int r = 0; r < rows; r++) Y[(size_t)r * B + b] = ycol[r];
            }
            free(xcol); free(ycol);
            return;
        }
        free(xcol); free(ycol);
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
    qwen_census_op("matvec_bf16_qkv", q_dim + 2 * kv_dim, in_dim, 1);
    /* The fused QKV reads the activation once for three matrices. If they came from a
     * GGUF Q8_0 they are registered individually, so the fused shortcut is given up in
     * exchange for staying on the Q8_0 numerics - and, more importantly, for not
     * leaving an undeclared bf16 fallback here (it was 10.6% of the Talker's and 26.0%
     * of the CP's work before this). */
    if (qwen_q8r_matmul(q, (const void *)Wq, x, q_dim,  in_dim, 1) &&
        qwen_q8r_matmul(k, (const void *)Wk, x, kv_dim, in_dim, 1) &&
        qwen_q8r_matmul(v, (const void *)Wv, x, kv_dim, in_dim, 1)) {
        MMSTAT(QWEN_MMK_Q8_REPACK_GEMV, q_dim + 2 * kv_dim, in_dim, 1);
        return;
    }
    MMSTAT(QWEN_MMK_BF16_GEMV, q_dim + 2 * kv_dim, in_dim, 1);   /* three products, one call */

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
    /* ── four output rows per pass, sharing one activation load (2026-08-21) ──
     * This is the B=1 kernel: the one the TTFA and the single-request RTF are made of.
     * It used to run two SDOT chains, and two is not enough to cover the ~3-cycle
     * latency of SDOT on the two (of four) Neoverse-V2 pipes that issue it. And the
     * limit is NOT memory: on the c3 box this path moved 6.78 GB/s against a 47.9 GB/s
     * Triad roof - 14% - so it is issue-bound, not bandwidth-bound, and more chains are
     * the lever. Four rows also means one activation load feeds four dot products
     * instead of two. Integer arithmetic, so the result is unchanged bit for bit; the
     * 2-row and 1-row loops below stay for the remainder. */
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

/* ── Q, K e V in UN colpo solo: una quantizzazione e un dispatch (2026-08-21) ──
 * qwen_matvec_int8_qkv chiamava tre volte qwen_matvec_int8, e ognuna delle tre
 * ri-quantizzava LA STESSA attivazione e apriva il suo giro di thread. Nel Code
 * Predictor questo succede 5 layer x 15 passi per frame: a quelle taglie (in_dim
 * 1024-2048) il costo fisso per chiamata pesa quanto il ciclo interno.
 * Qui la x si quantizza una volta e le tre matrici diventano UN solo spazio di
 * righe [q_dim | kv_dim | kv_dim] diviso fra i thread. Stessa aritmetica e stessa
 * quantizzazione, quindi risultato identico bit a bit. */
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
    /* tre segmenti contigui nello spazio globale delle righe: q, poi k, poi v */
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
    qwen_census_op("matvec_int8", rows, cols, 1);
    /* B=1 IS KleidiAI's layout: x is [1][cols] and y is [1][rows]. No transpose,
     * no gather - which is why the decode path was the cheapest one to move. */
    if (kai_i8_try(y, W, scale, x, rows, cols, 1)) {
        MMSTAT(QWEN_MMK_KLEIDI_I8_GEMV, rows, cols, 1);
        return;
    }
    MMSTAT(QWEN_MMK_INT8_GEMV, rows, cols, 1);   /* audit symmetry: GEMV counts too */

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
    qwen_census_op("matvec_int8_qkv", q_dim + 2 * kv_dim, in_dim, 1);
    /* Fused QKV on KleidiAI keeps the property this entry point exists for: quantize
     * the activation ONCE and serve all three matrices from it. */
    if (qwen_kleidi_i8_enabled() &&
        qwen_kleidi_matmul_i8_qkv(q, k, v, Wq, Wk, Wv, x, in_dim, q_dim, kv_dim)) {
        MMSTAT(QWEN_MMK_KLEIDI_I8_GEMV, q_dim + 2 * kv_dim, in_dim, 1);
        return;
    }
    MMSTAT(QWEN_MMK_INT8_GEMV, q_dim + 2 * kv_dim, in_dim, 1);   /* three products, one call */

    /* The inline threaded qkv block hung at 4 threads on int8-induced denormals
     * even with FTZ set in-block; the per-projection fused matvec is robust.
     * qwen_matvec_int8 dispatches across threads and sets FTZ in each worker. */
#if defined(__ARM_FEATURE_DOTPROD)
    /* SDOT: quantizza x UNA volta e serve le tre proiezioni con un solo dispatch.
     * Stesso qx, stessa sx e stesse righe di prima -> identico bit a bit; quello che
     * sparisce e' due terzi della quantizzazione e due giri di thread su tre.
     * QWEN_NO_SDOT=1 riporta tutto sul percorso precedente, senza ricompilare. */
    {
        enum { QX_MAX_QKV = 8192 };
        static atomic_int qkv_off = -1;
        int qo = atomic_load_explicit(&qkv_off, memory_order_relaxed);
        if (qo < 0) { const char *e = getenv("QWEN_NO_SDOT"); qo = (e && e[0] == '1'); atomic_store_explicit(&qkv_off, qo, memory_order_relaxed); }
        int nt = g_n_threads;
        if (!qo && in_dim <= QX_MAX_QKV && nt > 1 && (q_dim + 2 * kv_dim) >= 256) {
            int8_t qx_buf[QX_MAX_QKV];
            float sx = quantize_act_int8(qx_buf, x, in_dim);
            int8_qkv_sdot_ctx c = { q, k, v, qx_buf, sx, Wq, Wk, Wv, sq, sk, sv,
                                    in_dim, q_dim, kv_dim };
            qwen_parallel((size_t)nt, int8_qkv_sdot_task, &c);
            return;
        }
    }
#endif
    qwen_matvec_int8(q, Wq, sq, x, q_dim, in_dim);
    qwen_matvec_int8(k, Wk, sk, x, kv_dim, in_dim);
    qwen_matvec_int8(v, Wv, sv, x, kv_dim, in_dim);
}

/* Argmax over an INT8 matvec (CP lm_head with --int8 / quant-mixed; runs 15×/frame).
 * Reuses the optimized (multi-threaded, SDOT/VNNI) int8 matvec into a per-thread
 * scratch, then argmaxes — mirrors qwen_argmax_matvec_q4_0 below. The old body was
 * the only hot int8 kernel still doing single-threaded widen-to-f32 FMA (audit
 * 2026-07-11, perf item 4). __thread keeps the scratch race-free under concurrent
 * server synthesis (mirrors sampling.c's buffers). */
int qwen_argmax_matvec_int8(const float *x, const int8_t *W, const float *scale,
                            int in_dim, int out_dim) {
    qwen_census_op("argmax_matvec_int8", out_dim, in_dim, 1);
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

/* Argmax over a Q4_0 matvec (CP lm_head with --int4). Reuses the optimized
 * (multi-threaded, SIMD) q4_0 matvec into a small scratch buffer, then argmaxes.
 * The scratch is a per-thread grow-on-demand buffer (plan_v4 #8): the CP calls
 * this 15×/frame, so a fresh malloc/free each time is pure churn. __thread keeps
 * it race-free under concurrent server synthesis (mirrors sampling.c's buffers). */
int qwen_argmax_matvec_q4_0(const float *x, const q4_0_block_t *W, int in_dim, int out_dim) {
    qwen_census_op("argmax_matvec_q4_0", out_dim, in_dim, 1);
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
    /* Weighted-LSQ q4_0 quantizer (2026-07-14, feat/quant-sub4 "q4_0s1"):
     * same block layout, same kernels, same bytes — only the SCALE is smarter.
     * Per block: map the signed max-|.| value to -8 (full [-8,7] range instead of
     * the old absmax/7 = ±7), round, then closed-form weighted least-squares
     * rescale s* = Σw·v·q / Σw·q² with w = v². Measured on the TF harness (0.6B,
     * 143 frames): Talker code0 (words) 83.9% → 92.3%, CP c1-15 46.3% → 48.9%,
     * at identical load cost (one pass + one MAC pair per weight).
     * QWEN_Q4_NAIVE=1 restores the old absmax RTN (A/B + regression hunting). */
    static int naive = -1;
    if (naive < 0) { const char *e = getenv("QWEN_Q4_NAIVE"); naive = (e && *e) ? 1 : 0; }
    int blocks_per_row = cols / Q4_0_BLOCK_SIZE;
    for (int r = 0; r < rows; r++) {
        const uint16_t *row = src_bf16 + (size_t)r * cols;
        q4_0_block_t *dst_row = dst + (size_t)r * blocks_per_row;

        for (int b = 0; b < blocks_per_row; b++) {
            const uint16_t *blk = row + b * Q4_0_BLOCK_SIZE;

            /* Convert bf16 block to f32 and find the signed max-|.| value */
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
                /* legacy: absmax → ±7, plain RTN against the roundtripped scale */
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
                /* q4_0s1: signed max → -8, then weighted-LSQ rescale (w = v²) */
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

            /* Store as unsigned [0, 15] nibbles */
            for (int i = 0; i < 16; i++)
                dst_row[b].qs[i] = (uint8_t)((q[2*i] + 8) | ((q[2*i+1] + 8) << 4));
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
            float scale = qwen_f16_to_f32(row[b].scale_f16);
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
            float scale = qwen_f16_to_f32(row[b].scale_f16);
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
            fa0 = vfmaq_n_f32(fa0, vcvtq_f32_s32(acc0), qwen_f16_to_f32(r0[b].scale_f16));
            /* row 1 */
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
            sum += qwen_f16_to_f32(row[b].scale_f16) * (float)dot;
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
            s0 += qwen_f16_to_f32(r0[b].scale_f16) * (q4_hsum256(_mm512_castsi512_si256(d0)) + corr[b])
                + qwen_f16_to_f32(r0[b + 1].scale_f16) * (q4_hsum256(_mm512_extracti64x4_epi64(d0, 1)) + corr[b + 1]);
            s1 += qwen_f16_to_f32(r1[b].scale_f16) * (q4_hsum256(_mm512_castsi512_si256(d1)) + corr[b])
                + qwen_f16_to_f32(r1[b + 1].scale_f16) * (q4_hsum256(_mm512_extracti64x4_epi64(d1, 1)) + corr[b + 1]);
            s2 += qwen_f16_to_f32(r2[b].scale_f16) * (q4_hsum256(_mm512_castsi512_si256(d2)) + corr[b])
                + qwen_f16_to_f32(r2[b + 1].scale_f16) * (q4_hsum256(_mm512_extracti64x4_epi64(d2, 1)) + corr[b + 1]);
            s3 += qwen_f16_to_f32(r3[b].scale_f16) * (q4_hsum256(_mm512_castsi512_si256(d3)) + corr[b])
                + qwen_f16_to_f32(r3[b + 1].scale_f16) * (q4_hsum256(_mm512_extracti64x4_epi64(d3, 1)) + corr[b + 1]);
        }
        for (; b < nb; b++) {                  /* odd tail block */
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

/* v4 (deferred reduce, plan_v4 C7 round 3): v3 killed the half-width waste but
 * kept TWO cross-lane q4_hsum256 per dpbusd on the critical path — on the EPYC
 * that left int4-VNNI ~21% behind int8 (whose inner loop has NO cross-lane op).
 * v4 removes ALL cross-lane reduces from the block loop:
 *   - the dpbusd result (16 int32 partial sums; low 256-bit half = block b,
 *     high half = block b+1) is cvtepi32_ps'd and FMA'd into a per-row f32
 *     ACCUMULATOR VECTOR against a per-half scale vector [s_b ×8 | s_b+1 ×8];
 *   - the −8·Σqx correction (already scale-weighted) accumulates in a scalar;
 *   - ONE _mm512_reduce_add_ps per ROW at the end (v3: 2 per block per row).
 * Inner loop per row = dpbusd + cvt + fmadd (+ scale-vec build), all in-lane.
 * QWEN_Q4_VNNI_V4=0 falls back to v3 (and QWEN_Q4_VNNI_V3=0 to v2) so the box
 * can ladder-A/B without a rebuild. ⚠️ COMPILE-CHECKED ONLY on M1 (check-isa +
 * Rosetta has no AVX-512): numerics + the speed claim need the VNNI box. */
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
    for (; o + 3 < out_dim; o += 4) {          /* 4 independent rows */
        const q4_0_block_t *r0 = W + (size_t)o * nb, *r1 = r0 + nb, *r2 = r1 + nb, *r3 = r2 + nb;
        __m512 f0 = _mm512_setzero_ps(), f1 = _mm512_setzero_ps(),
               f2 = _mm512_setzero_ps(), f3 = _mm512_setzero_ps();
        float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f;
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
        for (; b < nb; b++) {                  /* odd tail block (rare: nb odd) */
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
    /* remaining rows via v3 (correct, just not 4-unrolled) */
    if (o < out_dim)
        q4_0_matvec_vnni_v3(y + o, qx, sx, W + (size_t)o * nb, cols, out_dim - o);
}

static int q4_vnni_v4_on(void) {
    /* Default OFF (= v3): measured on EPYC 9555P Zen5 2026-08-04, 3×-repeated
     * -j1 temp0 A/B — v3 1.05-1.06 vs v4 1.07-1.08 RTF. Zen5's cross-lane
     * reduce is cheap enough that v4's cvt+scale-vec build costs more than the
     * hsums it removes. QWEN_Q4_VNNI_V4=1 re-enables (re-test on Sapphire
     * Rapids / Ice Lake, where reduce latency may tip the other way). */
    static atomic_int v = -1;
    int r = atomic_load_explicit(&v, memory_order_relaxed);
    if (r < 0) { const char *e = getenv("QWEN_Q4_VNNI_V4"); r = (e && e[0] == '1'); /* default OFF */
                 atomic_store_explicit(&v, r, memory_order_relaxed); }
    return r;
}

/* Version ladder for the q4-VNNI matvec: v4 (default) → v3 (QWEN_Q4_VNNI_V4=0)
 * → v2 (also QWEN_Q4_VNNI_V3=0). One entry point so the matvec, the threaded
 * task and the fused QKV all pick the same kernel. */
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
    qwen_census_op("matvec_q4_0", rows, cols, 1);
    /* KleidiAI first, and ONLY if this exact weight matrix was pre-packed at load
     * (qwen_kleidi_matmul_q4 returns 0 when the pointer is not in the registry, which
     * is every weight that did not come from a GGUF Q4_0). Placed at the very top
     * because it replaces the whole kernel, threading included - it is not a variant
     * of the loop below, it is a different implementation of the same product.
     * B=1 here uses KleidiAI's dotprod GEMV, not a one-row GEMM. */
    if (qwen_mm_use(QWEN_MMK_KLEIDI_Q4, 1, rows, cols) &&
        qwen_kleidi_matmul_q4(y, (const void *)W, x, rows, cols, 1)) {
        MMSTAT(QWEN_MMK_KLEIDI_Q4, rows, cols, 1);
        return;
    }
    MMSTAT(QWEN_MMK_Q4_GEMV, rows, cols, 1);   /* everything that did NOT go to KleidiAI */
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
        q4_vnni_rows(y, qx_buf, sx, W, cols, rows);
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
#if defined(__AVX512VNNI__)
/* VNNI fused-QKV (the x86 twin of the SDOT one above — closes the last f32-dequant
 * hole in the single-stream int4 Talker path, plan_v4 C7 "fused-QKV VNNI twin"):
 * quantize the shared activation to int8 ONCE, then VNNI for Q/K/V, partitioning
 * the concatenated [Q|K|V] row space like the f32/SDOT twins. */
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
    qwen_census_op("matvec_q4_0_qkv", q_dim + 2 * kv_dim, in_dim, 1);
    /* KleidiAI first, per matrix. The fused QKV exists to read the activation once;
     * KleidiAI's GEMV re-quantizes it three times, which is the price of using the
     * pre-packed weights. Measured on Axion this is still the faster of the two, and
     * more importantly it removes an UNDECLARED fallback: before this, 15.7% of the
     * Talker's and 25.4% of the CP's q4 work quietly bypassed KleidiAI here. */
    if (qwen_mm_use(QWEN_MMK_KLEIDI_Q4, 1, q_dim, in_dim) &&
        qwen_kleidi_matmul_q4(q, (const void *)Wq, x, q_dim,  in_dim, 1) &&
        qwen_kleidi_matmul_q4(k, (const void *)Wk, x, kv_dim, in_dim, 1) &&
        qwen_kleidi_matmul_q4(v, (const void *)Wv, x, kv_dim, in_dim, 1)) {
        MMSTAT(QWEN_MMK_KLEIDI_Q4, q_dim + 2 * kv_dim, in_dim, 1);
        return;
    }
    MMSTAT(QWEN_MMK_Q4_GEMV, q_dim + 2 * kv_dim, in_dim, 1);     /* three products, one call */

#if defined(__AVX512VNNI__)
    /* QWEN_NO_VNNI_QKV=1 isolates THIS lever in the box A/B (falls back to the
     * f32-dequant QKV below while the standalone q4 matvec stays VNNI). */
    static atomic_int qkv_off = -1;
    int qkv_o = atomic_load_explicit(&qkv_off, memory_order_relaxed);
    if (qkv_o < 0) { const char *e = getenv("QWEN_NO_VNNI_QKV"); qkv_o = (e && e[0] == '1'); atomic_store_explicit(&qkv_off, qkv_o, memory_order_relaxed); }
    if (!qkv_o && !q4_sdot_disabled() && in_dim <= Q4_QX_MAX && in_dim % Q4_0_BLOCK_SIZE == 0) {
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
    qwen_census_op("matvec_q2_0", rows, cols, 1);
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
 * Q6_0 — 6 bits with an fp16 scale every 32 weights (PLAN T2 / T2.next)
 *
 * The cheap half of the per-layer mixed map (int8 on the sensitive layers, int6
 * on the rest). See the type comment in qwen_tts_kernels.h for WHY 6 bits and why
 * bit-exactness with tools/quant/fakequant_cp.py `int6b` is load-bearing: it is the
 * only thing that carries the measured language-identity numbers over to the real kernel.
 *
 * ONE ARITHMETIC IDENTITY SHARED BY ALL THREE ISAs. The stored code is unsigned,
 * u = q + 32 in [1,63]. Every dot product then uses u directly and corrects once
 * per block against the SHARED activation:
 *
 *     Σ w·x = scale · Σ (u−32)·x = scale · ( Σ u·x − 32·Σ x )
 *
 * `Σ x` per block depends only on the activation, so it is precomputed ONCE per
 * matvec and amortized over every output row (the trick q4_0_matvec_vnni already
 * uses for its −8 bias). Three consequences, all wanted:
 *   - no per-weight bias subtraction in the inner loop on any ISA;
 *   - u in [0,63] is a valid *positive* int8, so ARM's signed SDOT eats it as-is,
 *     while x86's unsigned×signed VNNI/maddubs want exactly an unsigned operand —
 *     one layout serves both instead of one trick per ISA;
 *   - the accumulation is integer and therefore BIT-IDENTICAL across NEON, AVX2,
 *     AVX-512 and the scalar fallback. Cross-ISA drift here would be a real bug,
 *     not tolerance noise, so `make check-isa` + the bench's parity gate can demand
 *     exact equality.
 *
 * ⚠️ Linux/x86_64 AND Linux/arm64 are first-class here, not a follow-up: AVX2 is
 * written alongside AVX-512-VNNI because most rentable VPS have AVX2 and NOT VNNI
 * (that needs Zen4 / Ice Lake+). A format that exists only on the newest x86 — or
 * only on NEON — collapses to the scalar fallback precisely where the product runs.
 * ======================================================================== */

void qwen_quantize_bf16_to_q6_0(const uint16_t *src_bf16, int rows, int cols,
                                 q6_0_block_t *dst) {
    /* Bit-exact replica of fakequant_cp.py `_q_blk(v, 6)`:
     *     s = fp16(absmax / 31);  q = clamp(roundf(v / s), -31, 31)
     * Note it DIVIDES by the fp16-rounded scale — multiplying by a reciprocal
     * differs in the last ulp on some blocks, and "close enough" is exactly what
     * would silently break the carry-over from the measured fakequant runs. */
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
                    /* floorf(|x|+0.5) with the sign put back — NOT roundf(). This
                     * replicates fakequant_cp.py's own `round_c` including its
                     * quirk: the two disagree on ~2 weights per million (values
                     * where |x|+0.5 rounds up to the next integer in f32). The
                     * audio cannot tell, but the parity gate can — and the gate is
                     * only worth having if it is exact, so the kernel matches the
                     * reference that the language-identity numbers were actually measured on.
                     * Changing this to true round-half-away-from-zero is a real
                     * (if tiny) format change and must be re-gated, not assumed. */
                    float a = fabsf(vals[i] / s);
                    float f = floorf(a + 0.5f);
                    if (f > 31.0f) f = 31.0f;
                    q = (vals[i] / s) < 0.0f ? -(int)f : (int)f;
                }
                unsigned u = (unsigned)(q + 32);              /* 1..63 */
                /* low 4 bits, q4_0-interleaved: even idx low nibble, odd idx high */
                dst_row[b].ql[i >> 1] |= (uint8_t)((u & 0xF) << ((i & 1) * 4));
                /* high 2 bits: qh[4g + j] bits 2k..2k+1 = weight 16g + j + 4k */
                int g = i >> 4, rem = i & 15, j = rem & 3, k = rem >> 2;
                dst_row[b].qh[4 * g + j] |= (uint8_t)((u >> 4) << (2 * k));
            }
        }
    }
}

/* Scalar unpack of one block into the 32 unsigned codes, in value order. The
 * reference the SIMD unpacks are checked against (and the scalar path's own). */
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

/* f32 activation -> int8 (per-vector absmax) + per-block Σx, both shared by every
 * output row. Scalar: cols is small (2048-ish) and this runs ONCE per matvec, so it
 * is ~0.05% of the work — vectorizing it would optimize noise. */
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

/* Portable reference kernel — also the real fallback on any ISA without a dot
 * product instruction. Same integer identity, so it must agree EXACTLY with the
 * SIMD paths. */
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
/* ── arm64: SDOT ──────────────────────────────────────────────────────────
 * Per block: ONE 16-byte load for the low nibbles (reusing q4_0's proven
 * vzip value-order trick) + TWO 4-byte broadcasts for the high bits, each
 * expanded with a single variable shift (vshlq_u8 with negative amounts) and a
 * mask — that layout was chosen precisely so the high-bit expansion costs 3
 * instructions per 16 weights instead of a shuffle table.
 * Codes stay unsigned 0..63, which is a positive int8, so signed SDOT is exact;
 * the −32 bias rides on the shared per-block Σx. 2-row fused like every other
 * matvec here, to amortize the activation loads. */
static const int8_t q6_shift_tab[16] = { 0,0,0,0, -2,-2,-2,-2, -4,-4,-4,-4, -6,-6,-6,-6 };

static inline void q6_unpack_neon(const q6_0_block_t *blk, int8x16_t *wa, int8x16_t *wb) {
    const uint8x16_t mask4 = vdupq_n_u8(0x0F);
    const uint8x16_t mask2 = vdupq_n_u8(0x03);
    const int8x16_t  sh    = vld1q_s8(q6_shift_tab);
    uint8x16_t raw = vld1q_u8(blk->ql);
    uint8x16x2_t z = vzipq_u8(vandq_u8(raw, mask4), vshrq_n_u8(raw, 4));
    /* LD1R straight from the block: loading the 4 high-bit bytes into a GPR and
     * moving them across to the vector unit costs a register-file crossing on
     * every block of every row, and it is on the critical path of the unpack.
     * vld1q_dup_u32 is the same broadcast done by the load unit. (AArch64 SIMD
     * loads have no alignment requirement, which matters here: the 26-byte block
     * stride leaves qh at a different alignment every other block.) */
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
        /* Deferred reduce, as in q4_0_matvec_sdot: keep the per-block scaling in a
         * float32x4 lane accumulator and do ONE vaddvq_f32 per row, instead of a
         * cross-lane reduce every 32 weights. */
        float32x4_t fa0 = vdupq_n_f32(0.0f), fa1 = vdupq_n_f32(0.0f);
        for (int b = 0; b < nb; b++) {
            const int8_t *xb = qx + b * Q6_0_BLOCK_SIZE;
            int8x16_t x0 = vld1q_s8(xb), x1 = vld1q_s8(xb + 16);
            /* ⚠️ Lane 0 ONLY. The four lanes are summed at the very end (deferred
             * reduce), so a vdupq of the correction would count it FOUR times —
             * the first version did exactly that and the parity gate caught it as
             * a relative error of ~10 instead of ~1e-2. */
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
            /* ⚠️ Lane 0 ONLY. The four lanes are summed at the very end (deferred
             * reduce), so a vdupq of the correction would count it FOUR times —
             * the first version did exactly that and the parity gate caught it as
             * a relative error of ~10 instead of ~1e-2. */
            int32x4_t corr = vsetq_lane_s32(-32 * sumx[b], vdupq_n_s32(0), 0);
            int8x16_t w0a, w0b;
            q6_unpack_neon(&r0[b], &w0a, &w0b);
            int32x4_t a0 = vdotq_s32(vdotq_s32(corr, w0a, x0), w0b, x1);
            fa0 = vfmaq_n_f32(fa0, vcvtq_f32_s32(a0), qwen_f16_to_f32(r0[b].scale_f16));
        }
        y[o] = vaddvq_f32(fa0) * sx;
    }
}
#endif /* __ARM_FEATURE_DOTPROD */

#if defined(__AVX2__)
/* Unpack one Q6_0 block to 32 UNSIGNED codes (0..63) in value order.
 * Low nibbles: the same unpacklo/unpackhi value-order trick as the tested AVX2 q4
 * path. High bits: AVX2 has no per-byte variable shift, so the four 2-bit fields
 * are produced as four whole-vector shifts of the SAME 4-byte broadcast and then
 * selected by 32-bit lane with _mm_blend_epi32 — 4 shifts + 4 ands + 3 blends per
 * 16 weights, no shuffle table, no memory traffic. */
static inline __m128i q6_high_128(uint32_t h) {
    const __m128i m2 = _mm_set1_epi8(0x03);
    __m128i d  = _mm_set1_epi32((int)h);
    __m128i s0 = _mm_and_si128(d, m2);                       /* lane0 -> weights 0..3   */
    __m128i s1 = _mm_and_si128(_mm_srli_epi32(d, 2), m2);    /* lane1 -> weights 4..7   */
    __m128i s2 = _mm_and_si128(_mm_srli_epi32(d, 4), m2);    /* lane2 -> weights 8..11  */
    __m128i s3 = _mm_and_si128(_mm_srli_epi32(d, 6), m2);    /* lane3 -> weights 12..15 */
    __m128i a  = _mm_blend_epi32(s0, s1, 0x2);
    __m128i b  = _mm_blend_epi32(s2, s3, 0x8);
    return _mm_blend_epi32(a, b, 0xC);
}
static inline __m256i q6_unpack_block_u8(const q6_0_block_t *blk) {
    const __m128i m4 = _mm_set1_epi8(0x0F);
    __m128i raw = _mm_loadu_si128((const __m128i *)blk->ql);
    __m128i lo  = _mm_and_si128(raw, m4);
    __m128i hi  = _mm_and_si128(_mm_srli_epi16(raw, 4), m4);
    __m128i l_a = _mm_unpacklo_epi8(lo, hi);   /* weights 0..15  low nibbles */
    __m128i l_b = _mm_unpackhi_epi8(lo, hi);   /* weights 16..31 low nibbles */
    uint32_t h0, h1;
    memcpy(&h0, blk->qh, 4);
    memcpy(&h1, blk->qh + 4, 4);
    /* values are 0..3, so <<4 cannot bleed past bit 7 despite the 16-bit shift */
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
/* ── x86-64 without VNNI (the common rented VPS) ──
 * maddubs is unsigned×signed, which is exactly the operand pair this layout
 * produces: codes 0..63 unsigned, activation int8. |u·x| <= 63·127 = 8001 and the
 * pairwise sum <= 16002, so the 16-bit intermediate cannot saturate.
 * ⚠️ COMPILE-CHECKED ONLY on this Mac (`make check-isa`); validate on the rented
 * box with `make test-golden` before trusting either numbers or speed. */
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
#endif /* __AVX2__ */

#if defined(__AVX512VNNI__)
/* ── x86-64 with VNNI (Zen4/5, Ice Lake+) ──
 * dpbusd wants unsigned×signed and gets it with no offset trick at all — the
 * unsigned-code layout was chosen partly for this. Two 32-weight blocks are packed
 * into ONE 512-bit dpbusd (the lesson from q4_0_matvec_vnni_v3, where a half-empty
 * 512-bit datapath plus a per-block cross-lane reduce made int4 ~37% SLOWER than
 * int8 on EPYC 9555P despite half the bytes).
 * ⚠️ COMPILE-CHECKED ONLY here; speed is a hypothesis until measured on silicon. */
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
            /* the two blocks' partial sums are the two 256-bit halves */
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
#endif /* __AVX512VNNI__ */

/* ONE dispatch point for "run these rows on whatever kernel this build has".
 * The q4 code repeats its row-partition logic once per ISA; doing it once here
 * means the fused QKV below cannot drift from the plain matvec on one ISA only —
 * which is the kind of divergence that shows up as "x86 sounds different". */
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

/* QWEN_Q6_SCALAR=1 forces the portable path — the A/B that proves a SIMD unpack
 * agrees with the reference on the machine actually running, not on the machine
 * the kernel was written on. */
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
    qwen_census_op("matvec_q6_0", rows, cols, 1);
    int8_t qx_buf[Q6_QX_MAX];
    int32_t sumx_buf[Q6_QX_MAX / Q6_0_BLOCK_SIZE];
    if (cols > Q6_QX_MAX || cols % Q6_0_BLOCK_SIZE != 0) {
        /* No silent wrong answer on an unsupported shape: dequantize the row and
         * fall back to plain f32. Slow and correct beats fast and unnoticed. */
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

/* Fused QKV — the SAME shape of win the int8 and q4 paths already have: the
 * activation is shared by Q, K and V, so quantizing it (and its per-block sums)
 * ONCE instead of three times removes two redundant passes AND two thread
 * barriers. Without this, a q6-vs-int8 comparison would be rigged against q6,
 * since int8 has a fused QKV and q6 would not. */
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
    qwen_census_op("matvec_q6_0_qkv", q_dim + 2 * kv_dim, in_dim, 1);
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
#elif defined(__AVX512VNNI__)
    /* S18: the x86 twin. Same panel/im2col scaffolding, VNNI micro-kernel.
     * Measured need (c3, 2026-08-19): the speech decoder is 55% of the wall on the
     * 0.6B and 41% on the 1.7B, and it was the ONE stage still running entirely in
     * fp32 on a box whose Talker/CP projections are already 90% AMX. */
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


#if defined(__ARM_FEATURE_DOTPROD) || defined(__AVX512VNNI__)

#if defined(__ARM_FEATURE_DOTPROD)

/* 2 rows x 4 cols register tile with per-block fp32 accumulation.
 * 8 int32 dot accs + 8 fp32 accs + 6 live loads = 22 registers. */
static inline void sd_tile_2x4(float *out, int out_ld, int m, int tcol,
                               const int8_t *Wq, const float *swb, const int32_t *wsum,
                               const float *bias,
                               const int8_t *Xq, const float *sab, int xrow,
                               int Kp, int blk, int nblk) {
    (void)wsum;   /* SDOT is signed x signed: no unsigned correction to apply */
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

#else  /* __AVX512VNNI__ : the x86 twin of the two tiles above */

/* WHY A CORRECTION EXISTS HERE AND NOT ON ARM. NEON's SDOT is signed x signed, so it
 * dots the quantized weight against the quantized activation directly. x86's VNNI
 * primitive (vpdpbusd) is UNSIGNED x signed. The standard fix is to make one side
 * unsigned by adding 128 and to subtract what that added:
 *
 *     sum(w_k * x_k) = dpbusd(x_k + 128, w_k) - 128 * sum(w_k)
 *
 * x + 128 on a signed byte is exactly XOR 0x80, one vector op per load. sum(w) is a
 * property of the WEIGHTS, which are quantized once and reused for the life of the
 * process, so it is precomputed in the registry and never touches this loop.
 *
 * ACCUMULATION SHAPE, copied from the ARM tile on purpose: int32 lanes per scale block,
 * converted to f32 and FMA'd with (weight scale x activation scale) once per block, and
 * ONE horizontal reduction at the very end. Reducing per block would put a cross-lane
 * add on the critical path every 64 bytes - measured on this codebase before, in the q4
 * VNNI matvec, where exactly that made int4 SLOWER than int8 on an EPYC. */

static inline float sd_hsum512(__m512 v) { return _mm512_reduce_add_ps(v); }

/* 2 rows x 4 cols, mirroring sd_tile_2x4: 8 int32 accumulators, 8 f32 accumulators. */
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
            /* blk is a multiple of 16, so the last step may be a partial vector: mask it
             * rather than reading past the row (rows are Kp-strided and zero-padded, but
             * the LAST row must not be read beyond its end). */
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
        /* the correction depends on the column too (through its block scale), so it is
         * summed here rather than hoisted: nblk multiply-adds against blk/64 dpbusd. */
        float k0 = 0.0f, k1 = 0.0f;
        for (int b = 0; b < nblk; b++) {
            k0 += 128.0f * (float)ws0[b] * sw0[b] * sap[c][b];
            k1 += 128.0f * (float)ws1[b] * sw1[b] * sap[c][b];
        }
        o0[c] = sd_hsum512(f0[c]) - k0 + b0;
        o1[c] = sd_hsum512(f1[c]) - k1 + b1;
    }
}

/* 1 row x up to 4 cols tail tile */
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

#endif /* tile implementations */

/* GEMM over one activation panel: out[m, tcol0 + c] for c in [0, nc).
 * Row-blocked (32) so the 4-column activation quad stays cache-resident
 * across row pairs. */
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

/* ---------------- threaded int8 conv1d (im2col per panel) -------------- */

#define SD_INT8_NC 128  /* activation columns per panel */

typedef struct {
    float *out;
    const float *in;
    const int8_t *Wq; const float *sw; const int32_t *wsum; const float *bias;
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
        sd_gemm_panel(j->out, j->length, j->out_ch, j->Wq, j->sw, j->wsum, j->bias,
                      colq, sa, t0, nc, j->Kp, j->blk);
    }
    free(colf); free(colq); free(sa);
}

void qwen_conv1d_int8(float *out, const float *in,
                      const int8_t *Wq, const float *sw, const int32_t *wsum,
                      const float *bias,
                      int in_ch, int out_ch, int length, int kernel, int dilation,
                      int Kp, int blk) {
    sd_conv_job_t job = {
        .out = out, .in = in, .Wq = Wq, .sw = sw, .wsum = wsum, .bias = bias,
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
    rpb = (rpb + 1) & ~1;               /* multiple of 2 rows */
    if (rpb < 2) rpb = 2;
    sd_gemm_job_t job = {
        .out = out, .out_ld = out_ld, .Wq = Wq, .sw = sw, .wsum = wsum, .Xq = Xq, .sa = sa,
        .M = M, .N = N, .Kp = Kp, .blk = blk,
        .rows_per_block = rpb, .n_blocks = (M + rpb - 1) / rpb,
    };
    atomic_store(&job.next_block, 0);
    sd_pool_run(sd_gemm_worker, &job);
}

#else /* neither SDOT nor VNNI: scalar reference (correctness oracle, not a fast path) */

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
    (void)wsum;   /* signed x signed here too: the correction is a VNNI-only artefact */
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

#endif /* SDOT or VNNI tiled path vs scalar reference */

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
    qwen_census_op("argmax_matvec_bf16", out_dim, in_dim, 1);
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
            /* VDPBF16PS (x86) and BFDOT/BFMMLA (ARM) both round the ACTIVATION to
             * bf16 → constant-ish absolute error per row; per-row max-rel explodes
             * on near-zero rows and means nothing (same reasoning as int8 below).
             * Gate on global L2.
             *
             * ⚠️ The ARM half was missing, and it made --self-test report
             * "SELF-TEST FAILED (3 cases)" on every Neoverse-V2 box: bf16 max_rel
             * 1.5-6.2 while int8 rel_L2 was 3.9e-03 and every argmax matched. A
             * self-test that cries wolf on correct hardware is worse than none -
             * it is the gate people learn to ignore. */
            max_rel_bf16 = sqrt(l2n_bf / (l2d_bf + 1e-12));
#else
            (void)l2n_bf; (void)l2d_bf;
#endif
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
                /* BFMMLA builds truncate the ACTIVATION to bf16 (native bf16 GEMM) →
                 * ~1e-3 L2 vs the f32-activation matvec is the expected correct value,
                 * not a defect (same class as the int8 act-quant tolerance). Same story
                 * under __AVX512BF16__: the dispatched MATVEC is now bf16-activation
                 * (VDPBF16PS) while the matmat keeps f32 activation. */
#if defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC) || defined(__AVX512BF16__) || \
    (defined(__AMX_BF16__) && defined(__AMX_TILE__))
                /* __AMX_BF16__ named explicitly: the AMX bf16 GEMM truncates the
                 * ACTIVATION exactly like BFMMLA, so it needs this relaxation for its
                 * OWN reason. It happens that Sapphire Rapids also has AVX512-BF16, but
                 * a tolerance must not depend on an unrelated feature bit being present. */
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

        int bf16_ok = max_rel_bf16 < 1e-2;   /* bf16: fp-order drift (or act-rounding L2 under VDPBF16PS) */
        int i8_ok   = rel_l2_i8    < 3e-2;   /* int8: activation-quant noise (~0.7% expected) */
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

                /* ---- q4_0 matvec vs exact dequant reference (plan_v4 C7 gate) ----
                 * Reference: dequantized-W · x with the activation kept f32. The
                 * dispatched matvec (SDOT/VNNI v2/v3/v4) quantizes the activation
                 * to int8 → same global-L2 tolerance class as int8 above. This is
                 * the case that catches a broken VNNI offset/scale on real silicon
                 * (per-row max-rel is meaningless near ref≈0, see int8 comment). */
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

    /* ---- S18: the decoder's int8 conv, against an INTEGER reference ----------------
     * Not a duplicate of the matvec cases above: this is the only kernel in the engine
     * whose output is the WAVEFORM, and the only one where ARM (SDOT, signed x signed)
     * and x86 (VNNI, unsigned x signed with a +128 correction) compute the same number
     * by different routes. A sign error in that correction does not crash and does not
     * look wrong in a profile — it comes out as noise in the audio, which is exactly the
     * failure this project has learned not to catch by ear alone.
     *
     * The reference re-does the im2col and calls the SAME qwen_int8_quant_rows, so both
     * sides see identical bytes and the only thing under test is the dot product. */
    {
        const int shapes[][3] = { {96, 7, 1}, {192, 7, 3}, {384, 7, 9}, {768, 1, 1} };
        const int nsh = (int)(sizeof(shapes) / sizeof(shapes[0]));
        const int length = 67;          /* not a multiple of the 128-column panel */
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

/* ========================================================================
 * `./qwen_tts --matmat-tune` — MEASURE the dispatcher thresholds instead of
 * guessing them (PLAN 0.nonies S9/S10, "scelta del kernel in funzione di ISA
 * *e* shape")
 *
 * WHAT PROBLEM THIS SOLVES. Every number in g_mm_gate[] above is a DECLARED
 * GUESS: AMX min_b = 4 because a quarter of the C tile is live, min_rows = 32
 * because two 16-row blocks hide the TDP latency, min_cols 32/64 because that is
 * one k-chunk, and 2 for VNNI/AVX2/SMMLA/BFMMLA because that is what the code did
 * before the gate existed. A GCP c3 carries AMX *and* AVX-512/VNNI/BF16 *and*
 * AVX2 on the same die — three generations of SIMD competing for the same
 * projection — and picking between them with hand-written constants would waste
 * the box. This mode sweeps the grid and prints the constants back.
 *
 * WHY IT IS NOT `--matmat-bench` WITH MORE LOOPS. The bench answers "does the
 * batched twin beat B matvecs at one B, on synthetic shapes, with whatever kernel
 * the ISA picked". Three things are missing to make a THRESHOLD out of that:
 *   1. per KERNEL, not per format. On a c3 the int8 dispatcher has four possible
 *      answers; a single aggregate number cannot say which of them earned the win.
 *      Here each kernel is pinned in turn (qwen_mm_force_kernel) and timed alone.
 *   2. per B, over a grid, because the whole question is "from which B on".
 *   3. at ONE THREAD *and* at full threads, side by side. This is the lesson of
 *      2026-08-18: `--matmat-bench` was ignoring -j, and once fixed, the bf16 twin
 *      on M1 went from a reported 1.70x to 0.79x at one real thread. The 1.70x was
 *      amortized pool-launch overhead, not shared weight reads. A KERNEL THAT WINS
 *      ONLY AT FULL THREADS IS NOT WINNING — IT IS HIDING AN OVERHEAD, and the
 *      report says so in words, per kernel, rather than leaving it to be inferred.
 *
 * THE REFERENCE IS ALWAYS B x matvec OF THE SAME FORMAT, including the column
 * gather/scatter, because that is literally what the engine does when the batched
 * path declines. speedup = t(B x matvec) / t(matmat); below 1.0 the batched twin
 * loses and the honest gate value is "never".
 *
 * SHAPES ARE THE MODEL'S, NOT ROUND NUMBERS. A 3072x1024 codec head and a
 * 1024x1024 KV projection sit on opposite sides of every cache threshold, and a
 * bench on {1024x1024, 2048x2048} would answer a question nobody asked. With -d
 * the dimensions come from that model's config.json; without it, from the two
 * declared OSS configs below.
 *
 * WHAT WOULD FALSIFY THE OUTPUT: run the recommended exports and re-run
 * `--matmat-bench`, or the server microbench, and see no change. The thresholds
 * are per (kernel, shape, thread count); this collapses them to one number per
 * env var by requiring the win on EVERY measured shape, which is deliberately
 * conservative — the full grid is in the JSON for anyone who wants a shape-aware
 * gate instead.
 * ======================================================================== */

#define QTUNE_MAXK   8              /* kernels timed per format (+1 fallback row)   */
#define QTUNE_NB     5
static const int g_qtune_B[QTUNE_NB] = { 1, 2, 4, 8, 16 };
/* 16 is a hard ceiling, not a taste: see the max_b comment next to g_mm_gate[] —
 * the int8/q4 dispatchers hold per-column activation scales in a float sx[16]. */
#define QTUNE_WIN    1.05           /* below +5% it is noise, not a win             */

typedef struct { char label[96]; int rows, cols; } qtune_shape_t;
typedef struct { int hidden, heads, kvheads, head_dim, inter, vocab; } qtune_dims_t;

/* The two OSS configs, transcribed from qwen3-tts-{0.6b,1.7b}/config.json. Declared
 * here rather than hidden in a shape list so the derivation stays auditable. */
static const qtune_dims_t g_qtune_dims_06b_talker = { 1024, 16, 8, 128, 3072, 3072 };
static const qtune_dims_t g_qtune_dims_17b_talker = { 2048, 16, 8, 128, 6144, 3072 };
static const qtune_dims_t g_qtune_dims_cp         = { 1024, 16, 8, 128, 3072, 2048 };

static void qtune_add_shape(qtune_shape_t *v, int *n, int cap, int rows, int cols,
                            const char *label) {
    if (rows <= 0 || cols <= 0) return;
    for (int i = 0; i < *n; i++) {
        if (v[i].rows == rows && v[i].cols == cols) {      /* dedup, keep both names */
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

/* The six projection shapes a Qwen3-TTS decoder block actually issues. Written as a
 * derivation from config fields so a finetune with different dims needs no new table. */
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

static long qtune_json_int(const char *p) {   /* p at the ':' or before the number */
    while (*p && *p != ':') p++;
    if (*p) p++;
    while (*p == ' ' || *p == '\n' || *p == '\t' || *p == '\r') p++;
    return strtol(p, NULL, 10);
}

/* Brace-aware scan of config.json for the dims of talker AND code predictor.
 * NOT a JSON parser, but not a strstr() either, and the difference matters: the keys
 * are alphabetical, so "code_predictor_config" nests BEFORE the talker's own
 * "hidden_size", and a naive search for the first "hidden_size" after
 * "talker_config" returns the CODE PREDICTOR's 1024 for every model — silently right
 * on the 0.6B and silently wrong on the 1.7B. Returns 1 if the talker dims are
 * complete. */
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
        p = e;                                  /* never rescan inside a string value */
        while (*after == ' ' || *after == '\n' || *after == '\t' || *after == '\r') after++;
        if (*after != ':') continue;            /* it was a value, not a key */
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

/* ── the timed work ─────────────────────────────────────────────────────────── */
typedef struct {
    int fmt;                       /* 0 bf16, 1 int8, 2 q4_0 */
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
/* The reference INCLUDES the gather/scatter, because the engine pays it too when the
 * batched path declines: X is stored [cols][B] interleaved either way. */
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
/* Best-of-3 over a rep loop sized to ~target_ms. MIN, not mean: the distribution is
 * one-sided (an interrupt can only make a run slower), so the minimum is the closest
 * estimate of the machine and the least sensitive to a noisy laptop. */
static double qtune_bench(void (*fn)(void *), void *ctx, double target_ms) {
    fn(ctx);                                     /* warm: first touch of W faults pages */
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

/* Which kernel actually ran? Uses the MMSTAT audit counters (same instrumentation as
 * QWEN_BATCH_STATS) for ONE untimed call, then switches them off again so the timed
 * runs carry no counter traffic. This is the check that keeps the table honest: a
 * forced kernel blocked by a CAPABILITY (qwen_amx_*_ready, cols % Q4_0_BLOCK_SIZE —
 * correctly NOT overridable) falls through to the tail, and without this we would
 * print the tail's time under the forced kernel's name. */
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
    int  mmk;                      /* >0 = forced kernel id, -1 = "no matmat kernel" */
    int  fired;                    /* id the audit counters actually saw             */
    char name[40];
    double ms[QTUNE_NB];           /* per B, per thread mode (filled by the caller)  */
    double sp[QTUNE_NB];
    int    ok;
} qtune_kres_t;

/* Kernels this BINARY could dispatch for a format, in dispatcher order. The #if set
 * mirrors qwen_matmat_{bf16,int8,q4_0} exactly; the qwen_amx_*_ready() calls mirror
 * their runtime capability check, so an AMX build on a pre-5.16 kernel lists no AMX
 * row instead of listing one that cannot fire. */
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
    /* The tail every dispatcher falls through to when no gated kernel accepts. Named
     * from the audit counter rather than guessed, because it differs per ISA: on
     * M1-class ARM the q4 tail is literally B x matvec (so a 1.00x there is a
     * tautology, not a result), while elsewhere it is a scalar batched twin. */
    if (n < QTUNE_MAXK) {
        out[n].mmk = -1; out[n].fired = 0;
        snprintf(out[n].name, sizeof(out[n].name), "(dispatcher tail)");
        n++;
    }
    return n;
}

/* Smallest B in the grid from which `sp` wins AND keeps winning (monotone): one lucky
 * cell at B=2 that reverses at B=4 is not a crossover, it is noise, and a min_b cut
 * there would send the engine to a kernel that loses at the B it will actually see. */
static int qtune_crossover(const int *win) {
    for (int i = 0; i < QTUNE_NB; i++) {
        if (!win[i]) continue;
        int all = 1;
        for (int j = i; j < QTUNE_NB; j++) if (!win[j]) { all = 0; break; }
        if (all) return g_qtune_B[i];
    }
    return 0;                                    /* 0 = never */
}

int qwen_matmat_tune(void *out, const char *model_dir) {
    FILE *f = out ? (FILE *)out : stdout;
    const int full = qwen_get_threads() > 0 ? qwen_get_threads() : 1;
    const char *jpath = getenv("QWEN_TUNE_JSON");
    FILE *js = NULL;

    /* ---- shapes ---- */
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
    if (qk && qk[0] && qk[0] != '0') {           /* drop the >4 M-element shapes */
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

    /* crossover[fmt][kernel_slot][thread_mode] aggregated over shapes:
     * worst = the largest per-shape crossover (0 if it never wins somewhere) — the
     * conservative value, because below it SOME real projection regresses;
     * best  = the smallest, kept only to show how much the shapes disagree. */
    int  agg_worst[3][QTUNE_MAXK + 1][2], agg_best[3][QTUNE_MAXK + 1][2];
    int  agg_fired[3][QTUNE_MAXK + 1];
    char agg_name[3][QTUNE_MAXK + 1][40];
    int  agg_mmk[3][QTUNE_MAXK + 1], agg_n[3] = { 0, 0, 0 };
    memset(agg_worst, 0, sizeof(agg_worst));
    memset(agg_best, 0, sizeof(agg_best));
    memset(agg_fired, 0, sizeof(agg_fired));
    memset(agg_name, 0, sizeof(agg_name));
    memset(agg_mmk, 0, sizeof(agg_mmk));

    /* Noise floor, measured rather than assumed (CLAUDE.md: state a metric's noise
     * floor before reporting a number). The reference is re-measured once per cell and
     * compared with itself; whatever spread that shows is the smallest difference this
     * box can resolve, and any "win" inside it is not a win. */
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
                if (tm == 1 && full == 1) break;      /* identical row, do not print twice */
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
                    /* a pinned kernel that did not reach the counter ran nothing of its
                     * own: report it, never attribute the tail's time to it */
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
                        /* the summary must name what actually ran, not "(dispatcher tail)" */
                        snprintf(agg_name[fmt][k], sizeof(agg_name[fmt][k]), "%s", tail);
                        snprintf(kr[k].name, sizeof(kr[k].name), "%s", tail);   /* and the JSON */
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

                /* per-shape crossover, needs every kernel's times to compare them */
                for (int k = 0; k < nk; k++) {
                    int win[QTUNE_NB];
                    for (int bi = 0; bi < QTUNE_NB; bi++) {
                        int w = kr[k].ok && kr[k].sp[bi] >= QTUNE_WIN;
                        for (int o = 0; o < nk && w; o++)
                            if (o != k && kr[o].ok && kr[o].ms[bi] < kr[k].ms[bi]) w = 0;
                        win[bi] = w;
                    }
                    int x = qtune_crossover(win);
                    /* worst-shape aggregate: "never" is ABSORBING. One projection that
                     * regresses is enough to make a global min_b wrong, and the engine
                     * issues all of these shapes in the same forward pass. */
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

    /* ── summary ───────────────────────────────────────────────────────────── */
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
            /* A tail that IS B x matvec (M1-class ARM for q4, or FORCE_MATVEC) cannot
             * lose to B x matvec: it is the same code. Calling that "loses everywhere"
             * would invite someone to disable the thing they are measuring against. */
            if (agg_fired[fmt][k] == QWEN_MMK_Q4_BMATVEC ||
                agg_fired[fmt][k] == QWEN_MMK_FORCED_MATVEC)
                verdict = "== the reference by construction (no batched q4 kernel on this ISA)";
            /* The tail has no gate of its own: it is what runs when everything else
             * declines, so the only lever over it is QWEN_BATCH_FORCE_MATVEC=1 (which
             * turns the whole batched path off). Say that instead of printing an env
             * var that does not exist. */
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

    /* ── the lines to export ───────────────────────────────────────────────── */
    fprintf(f, "\n══ CONFIGURATION FOR THIS BOX  (paste, or use tests/kernel_tune.sh's .env)\n");
    if (js) fprintf(js, "  \"recommend\": [\n");
    int jfirst = 1, printed = 0;
    for (int mmk = 1; mmk < QWEN_MMK_COUNT; mmk++) {
        const qwen_mm_gate_t *g = &g_mm_gate[mmk];
        if (g->max_b == 0) continue;
        int fmt = -1, slot = -1;
        for (int fq = 0; fq < 3 && fmt < 0; fq++)
            for (int k = 0; k < agg_n[fq]; k++)
                if (agg_mmk[fq][k] == mmk) { fmt = fq; slot = k; break; }
        if (fmt < 0) continue;                       /* not available in this binary */
        int wf = full > 1 ? agg_worst[fmt][slot][1] : agg_worst[fmt][slot][0];
        int w1 = agg_worst[fmt][slot][0];
        char line[160], why[200];
        if (wf <= 0) {
            if (g->on_env) {                     /* opt-in kernel: default OFF is already right */
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
        } else if (g->on_env && g->minb_env) {
            snprintf(line, sizeof(line), "export %s=1 %s=%d", g->on_env, g->minb_env, wf);
            snprintf(why, sizeof(why), "%s is opt-in and DOES win from B>=%d here%s",
                     agg_name[fmt][slot], wf,
                     (w1 <= 0) ? "; POOL-ONLY WIN, see the warning above" : "");
        } else if (!g->minb_env) {
            snprintf(line, sizeof(line), "# %s: wins from B>=%d but has NO min_b env",
                     agg_name[fmt][slot], wf);
            snprintf(why, sizeof(why), "missing minb_env in g_mm_gate[]");
        } else {
            snprintf(line, sizeof(line), "export %s=%d", g->minb_env, wf);
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

    /* ── which thresholds exist, and are they all reachable without a rebuild? ── */
    fprintf(f, "\n══ THRESHOLD INVENTORY  (every guessed number in g_mm_gate[], and its override)\n");
    fprintf(f, "   %-24s %-24s %-22s %-8s %s\n", "kernel", "min_b env", "min_rows/min_cols env",
            "defaults", "off switch");
    for (int mmk = 1; mmk < QWEN_MMK_COUNT; mmk++) {
        const qwen_mm_gate_t *g = &g_mm_gate[mmk];
        if (g->max_b == 0) continue;
        int avail = 0;
        for (int fq = 0; fq < 3 && !avail; fq++)
            for (int k = 0; k < agg_n[fq]; k++) if (agg_mmk[fq][k] == mmk) { avail = 1; break; }
        if (!avail) {
            /* Not a defect: it means this ISA/build/kernel-permission cannot reach it.
             * Printed so the C3 run can be diffed against the M1 run and the difference
             * read as "these are the rows only the box can answer". */
            char df0[24];
            snprintf(df0, sizeof(df0), "%d/%d/%d", g->min_b, g->min_rows, g->min_cols);
            fprintf(f, "   %-24s %-24s %-22s %-8s %s\n", g_mmk_info[mmk].name,
                    g->minb_env ? g->minb_env : "MISSING",
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
                g->minb_env ? g->minb_env : "MISSING", rc, df,
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
