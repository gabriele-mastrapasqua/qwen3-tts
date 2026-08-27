/* CPU feature detection and the thread-injection point.
 *
 * The usual two ways of getting this wrong are picking the SIMD level at BUILD
 * time, through -D flags a Makefile has to remember to pass, and doing proper
 * runtime detection on one architecture only. Here the "compiled" half comes
 * from the compiler's own predefined macros — so there is no build flag to
 * forget — and the "runtime" half is checked on every platform.
 *
 * SPDX-License-Identifier: MIT */
/* posix_memalign, below. glibc hides it under -std=c11 without this, and an
 * implicitly declared function is an error on current compilers and undefined
 * behaviour on the ones that let it through. macOS declares it either way,
 * which is why only Linux ever complained. */
#define _POSIX_C_SOURCE 200809L
#if defined(__APPLE__)
/* And <sys/sysctl.h> is outside the strict POSIX namespace on Darwin, so the
 * line above alone breaks the SDK headers. Same pair as in safetensors.c. */
#define _DARWIN_C_SOURCE 1
#endif

#include "ingot/quant.h"
#include "internal.h"

#include <pthread.h>
#include <stdlib.h>

#if defined(__APPLE__)
#include <sys/sysctl.h>
#elif defined(__linux__) && defined(__aarch64__)
#include <sys/auxv.h>
#include <asm/hwcap.h>
#endif
#if defined(__x86_64__) || defined(__i386__)
#if defined(__GNUC__) || defined(__clang__)
#include <cpuid.h>
#endif
#endif

/* What this translation unit was actually compiled with. */
#if defined(__ARM_NEON) || defined(__aarch64__)
#define INGOT_COMPILED_NEON 1
#endif
#if defined(__ARM_FEATURE_DOTPROD)
#define INGOT_COMPILED_DOTPROD 1
#endif
#if defined(__ARM_FEATURE_MATMUL_INT8)
#define INGOT_COMPILED_I8MM 1
#endif
#if defined(__AVX2__)
#define INGOT_COMPILED_AVX2 1
#endif
#if defined(__AVX512F__)
#define INGOT_COMPILED_AVX512 1
#endif
#if defined(__AVX512VNNI__)
#define INGOT_COMPILED_AVX512_VNNI 1
#endif
#if defined(__F16C__)
#define INGOT_COMPILED_F16C 1
#endif
#if defined(__AVX512BF16__)
#define INGOT_COMPILED_AVX512_BF16 1
#endif
#if defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC) || defined(__ARM_FEATURE_BF16)
#define INGOT_COMPILED_ARM_BF16 1
#endif

static ingot_cpu_caps cached;
static int cap_level_cap = -1;          /* -1 = auto */
static pthread_once_t caps_once = PTHREAD_ONCE_INIT;

/* Name to cap level, with no locking of any kind: init_caps runs INSIDE
 * pthread_once and must never re-enter it. -2 means the name is not one we
 * know. */
#define INGOT_LEVEL_UNKNOWN (-2)
static int level_from_name(const char *name) {
    if (name == NULL || strcmp(name, "auto") == 0) return -1;
    if (strcmp(name, "scalar") == 0) return 0;
    if (strcmp(name, "avx2") == 0 || strcmp(name, "neon") == 0) return 1;
    if (strcmp(name, "vnni") == 0 || strcmp(name, "dotprod") == 0) return 2;
    return INGOT_LEVEL_UNKNOWN;
}

/* The effective level of `caps` once `cap_level_cap` is applied. */
static int level_of(ingot_cpu_caps caps) {
    if (caps.avx512_vnni || caps.dotprod) return 2;
    if (caps.avx2 || caps.neon) return 1;
    return 0;
}

static ingot_cpu_caps capped(ingot_cpu_caps caps) {
    if (cap_level_cap >= 0) {
        if (cap_level_cap < 2) {
            caps.avx512_vnni = 0; caps.i8mm = 0; caps.dotprod = 0;
            caps.bf16 = 0; caps.avx512_bf16 = 0;
        }
        if (cap_level_cap < 1) {
            caps.avx2 = 0; caps.avx512 = 0; caps.neon = 0;
            caps.f16c = 0;
        }
    }
    return caps;
}

#if defined(__APPLE__) && defined(__aarch64__)
static int sysctl_flag(const char *name) {
    int value = 0;
    size_t size = sizeof(value);
    return sysctlbyname(name, &value, &size, NULL, 0) == 0 && value != 0;
}
#endif

#if defined(__x86_64__) || defined(__i386__)
static unsigned long long xgetbv0(void) {
    unsigned int eax, edx;
    __asm__ __volatile__(".byte 0x0f, 0x01, 0xd0" : "=a"(eax), "=d"(edx) : "c"(0));
    return ((unsigned long long)edx << 32) | eax;
}
#endif

static void init_caps(void) {
    memset(&cached, 0, sizeof(cached));

#if defined(INGOT_COMPILED_NEON)
    cached.neon = 1;                     /* NEON is baseline on aarch64 */
#endif
#if defined(INGOT_COMPILED_DOTPROD)
#if defined(__APPLE__)
    cached.dotprod = sysctl_flag("hw.optional.arm.FEAT_DotProd");
#elif defined(__linux__) && defined(__aarch64__) && defined(HWCAP_ASIMDDP)
    cached.dotprod = (getauxval(AT_HWCAP) & HWCAP_ASIMDDP) != 0;
#else
    cached.dotprod = 1;                  /* compiled for it, cannot ask: trust it */
#endif
#endif
#if defined(INGOT_COMPILED_I8MM)
#if defined(__APPLE__)
    cached.i8mm = sysctl_flag("hw.optional.arm.FEAT_I8MM");
#elif defined(__linux__) && defined(__aarch64__) && defined(HWCAP2_I8MM)
    cached.i8mm = (getauxval(AT_HWCAP2) & HWCAP2_I8MM) != 0;
#else
    cached.i8mm = 1;
#endif
#endif
#if defined(INGOT_COMPILED_ARM_BF16)
#if defined(__APPLE__)
    cached.bf16 = sysctl_flag("hw.optional.arm.FEAT_BF16");
#elif defined(__linux__) && defined(__aarch64__) && defined(HWCAP2_BF16)
    cached.bf16 = (getauxval(AT_HWCAP2) & HWCAP2_BF16) != 0;
#else
    cached.bf16 = 1;
#endif
#endif

#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
    {
        unsigned int a = 0, b = 0, c = 0, d = 0;
        int os_ymm = 0, os_zmm = 0;
        if (__get_cpuid(1, &a, &b, &c, &d) && ((c >> 27) & 1)) {   /* OSXSAVE */
            const unsigned long long xcr0 = xgetbv0();
            os_ymm = (xcr0 & 0x6) == 0x6;                          /* XMM|YMM  */
            os_zmm = os_ymm && (xcr0 & 0xe0) == 0xe0;              /* + ZMM    */
#if defined(INGOT_COMPILED_F16C)
            cached.f16c = os_ymm && ((c >> 29) & 1);               /* leaf 1 ECX */
#endif
        }
#if !defined(INGOT_COMPILED_AVX512) && !defined(INGOT_COMPILED_AVX512_VNNI) && \
    !defined(INGOT_COMPILED_AVX512_BF16)
        (void)os_zmm;   /* only the AVX-512 checks below read it */
#endif
        if (__get_cpuid_count(7, 0, &a, &b, &c, &d)) {
#if defined(INGOT_COMPILED_AVX2)
            cached.avx2 = os_ymm && ((b >> 5) & 1);
#endif
#if defined(INGOT_COMPILED_AVX512)
            cached.avx512 = os_zmm && ((b >> 16) & 1);
#endif
#if defined(INGOT_COMPILED_AVX512_VNNI)
            cached.avx512_vnni = os_zmm && ((c >> 11) & 1);
#endif
        }
#if defined(INGOT_COMPILED_AVX512_BF16)
        /* AVX512-BF16 lives in leaf 7 SUBLEAF 1 (EAX bit 5), not subleaf 0. */
        if (__get_cpuid_count(7, 1, &a, &b, &c, &d))
            cached.avx512_bf16 = os_zmm && ((a >> 5) & 1);
#endif
    }
#endif

    /* INGOT_CAPS caps the level without a rebuild, which is how you bisect a
     * bug that only shows up on one SIMD path. Parsed inline, NOT through
     * ingot_cpu_set_level(): we are inside pthread_once here, and that function
     * enters it again — recursing on the same pthread_once_t is undefined and
     * killed the process outright. An unknown name is ignored rather than
     * fatal: an environment variable must not be able to stop a library. */
    const char *env = getenv("INGOT_CAPS");
    if (env != NULL) {
        const int level = level_from_name(env);
        if (level != INGOT_LEVEL_UNKNOWN) cap_level_cap = level;
    }

    /* INGOT_CAPS_ASSUME goes the other way: trust the BUILD instead of CPUID.
     *
     * It exists because CPUID is not always telling the truth. Rosetta 2
     * executes AVX2 but does not advertise it in leaf 7, so on an Apple
     * Silicon machine — which for many of us is the only place x86 code can be
     * run at all — every AVX2 kernel in this library silently falls back to
     * scalar. They are then neither tested nor measured, and an untested
     * kernel is the one that is wrong.
     *
     * This can only switch on what the compiler was already told to emit, so
     * it cannot conjure an instruction the binary does not contain. It CAN
     * fault on a machine that genuinely lacks the feature, which is why it is
     * opt-in, off by default, and a testing tool rather than a tuning knob.
     *
     * AVX-512 is deliberately not forceable: it additionally needs the OS to
     * have enabled ZMM state, and that is not something a build flag knows. */
    const char *assume = getenv("INGOT_CAPS_ASSUME");
    if (assume != NULL) {
        const int level = level_from_name(assume);
        if (level >= 1) {
#if defined(INGOT_COMPILED_NEON)
            cached.neon = 1;
#endif
#if defined(INGOT_COMPILED_AVX2)
            cached.avx2 = 1;
#endif
#if defined(INGOT_COMPILED_F16C)
            cached.f16c = 1;
#endif
        }
        if (level >= 2) {
#if defined(INGOT_COMPILED_DOTPROD)
            cached.dotprod = 1;
#endif
#if defined(INGOT_COMPILED_I8MM)
            cached.i8mm = 1;
#endif
        }
    }
}

ingot_cpu_caps ingot_cpu(void) {
    pthread_once(&caps_once, init_caps);
    return capped(cached);
}

int ingot_cpu_set_level(const char *name) {
    pthread_once(&caps_once, init_caps);
    const int level = level_from_name(name);
    if (level == INGOT_LEVEL_UNKNOWN) return -1;
    cap_level_cap = level;
    return level_of(capped(cached));
}

/* ── thread injection ───────────────────────────────────────────────────── */
static void parallel_for_serial(size_t count, ingot_range_fn fn, void *user) {
    if (count != 0) fn(0, count, user);
}

static ingot_parallel_for_fn parallel_for_impl = parallel_for_serial;

void ingot_set_parallel_for(ingot_parallel_for_fn fn) {
    parallel_for_impl = (fn != NULL) ? fn : parallel_for_serial;
}

void ingot_parallel_for(size_t count, ingot_range_fn fn, void *user);
void ingot_parallel_for(size_t count, ingot_range_fn fn, void *user) {
    parallel_for_impl(count, fn, user);
}

/* ── aligned allocation ─────────────────────────────────────────────────── */
void *ingot_aligned_alloc(size_t alignment, size_t size);
void  ingot_aligned_free(void *ptr);

void *ingot_aligned_alloc(size_t alignment, size_t size) {
    if (alignment == 0 || (alignment & (alignment - 1)) != 0) return NULL;
    if (alignment < sizeof(void *)) alignment = sizeof(void *);
    void *ptr = NULL;
    const size_t padded = (size + alignment - 1) & ~(alignment - 1);
    if (posix_memalign(&ptr, alignment, padded == 0 ? alignment : padded) != 0) return NULL;
    return ptr;
}

void ingot_aligned_free(void *ptr) { free(ptr); }
