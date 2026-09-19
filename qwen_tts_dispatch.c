/* qwen_tts_dispatch.c — `--dispatch-map`: every dispatch decision RESOLVED for this host
 * and this environment, as one table.
 *
 * Why this exists.  `[FLAGS]` prints the env exactly as the operator typed it: a flag
 * that was never set does not appear, a compiled default is invisible, and a predicate
 * that decides a whole path (the Talker prefill's use_matmat) lives outside the gate
 * table.  On 2026-09-03 that combination hid a fallback worth ~400 ms of TTFA on an
 * AVX-512-BF16 host.  This report prints, per logical feature:
 *
 *     compiled   is the implementation in this binary at all
 *     supported  does this CPU have the instructions / units it needs
 *     env        the raw variable (or "unset")
 *     resolved   what the runtime predicate returns NOW
 *     reason     which branch of the predicate decided it
 *
 * Rule: `resolved` is obtained by CALLING the runtime predicate, never by re-deriving
 * it here.  If a feature's predicate is static in another translation unit, that unit
 * exports a thin wrapper (see qwen_tts_kernels.h) rather than this file guessing.
 *
 * Output: a human table on `out` and, when json_path is given (or QWEN_DISPATCH_JSON),
 * a JSON document with the same rows for tools/cpu_check.sh and tools/dispatch_gate.py.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "qwen_tts_kernels.h"
#include "qwen_tts_kleidi.h"
#include "qwen_tts_thread.h"
#include "qwen_tts_q8repack.h"

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

int qwen_prefix_cache_enabled(void);          /* qwen_tts_talker.c */
int qwen_get_threads(void);

typedef struct {
    const char *id;         /* stable, dotted: component.op.impl */
    const char *compiled;   /* "yes" / "no" / "-" (not an implementation) */
    const char *supported;  /* "yes" / "no" / "-" */
    const char *env_name;   /* the variable that steers it, or NULL */
    char env_val[48];       /* raw value or "unset" */
    char resolved[24];      /* ON / OFF / a value */
    char reason[160];
} feat_t;

static const char *env_or_unset(const char *name, char *buf, size_t n) {
    const char *e = name ? getenv(name) : NULL;
    if (!e) { snprintf(buf, n, "unset"); return buf; }
    if (!e[0]) { snprintf(buf, n, "\"\""); return buf; }
    snprintf(buf, n, "%s", e);
    return buf;
}

static feat_t *row(feat_t *f, const char *id, const char *compiled, const char *supported,
                   const char *env_name, const char *resolved, const char *reason) {
    memset(f, 0, sizeof *f);
    f->id = id; f->compiled = compiled; f->supported = supported; f->env_name = env_name;
    env_or_unset(env_name, f->env_val, sizeof f->env_val);
    snprintf(f->resolved, sizeof f->resolved, "%s", resolved);
    snprintf(f->reason, sizeof f->reason, "%s", reason);
    return f;
}

static const char *yn(int v) { return v ? "yes" : "no"; }
static const char *onoff(int v) { return v ? "ON" : "OFF"; }

#if defined(__x86_64__) || defined(_M_X64)
static int x86_avx512_profile_available(void) {
    return __builtin_cpu_supports("avx2") &&
           __builtin_cpu_supports("fma") &&
           __builtin_cpu_supports("avx512f") &&
           __builtin_cpu_supports("avx512bw") &&
           __builtin_cpu_supports("avx512vl");
}
#endif

/* The class a tools/dispatch_expect.json entry is keyed by.  Coarse on purpose: it
 * names the best matrix lever the host+binary pair can use, which is what the
 * expected-vs-observed rules are about. */
static const char *isa_class(void) {
#if defined(__x86_64__) || defined(_M_X64)
    if (qwen_amx_int8_available() || qwen_amx_bf16_available()) return "x86_amx";
#if defined(__AVX512BF16__)
    if (x86_avx512_profile_available() && __builtin_cpu_supports("avx512bf16"))
        return "x86_avx512bf16";
#endif
#if defined(__AVX512VNNI__)
    if (x86_avx512_profile_available() && __builtin_cpu_supports("avx512vnni"))
        return "x86_avx512vnni";
#endif
#if defined(__AVX512F__)
    /* SIMD=avx512 deliberately has AVX-512F/BW/VL but no VNNI. Keep it
     * distinct from the AVX2 build: integer GEMV/matmat still resolve through
     * AVX2, while this binary also has wider conversion helpers. */
    if (x86_avx512_profile_available()) return "x86_avx512f_no_vnni";
#endif
#if defined(__AVX2__)
    if (__builtin_cpu_supports("avx2")) return "x86_avx2";
#endif
    return "x86_portable";
#elif defined(__aarch64__)
#if defined(__APPLE__)
#if defined(__ARM_FEATURE_MATMUL_INT8) && defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC)
    return "apple_i8mm_bf16";
#else
    return "apple_m1";
#endif
#else
#if defined(__ARM_FEATURE_MATMUL_INT8) && defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC)
    return "arm_i8mm_bf16";
#elif defined(__ARM_FEATURE_DOTPROD)
    return "arm_dotprod";
#else
    return "arm_portable";
#endif
#endif
#else
    return "other";
#endif
}

static const char *gate_id(int mmk) {
    switch (mmk) {
    case QWEN_MMK_BF16_BFMMLA: return "gate.bf16.bfmmla";
    case QWEN_MMK_BF16_AVX512: return "gate.bf16.avx512";
    case QWEN_MMK_BF16_AMX:    return "gate.bf16.amx";
    case QWEN_MMK_INT8_AMX:    return "gate.int8.amx";
    case QWEN_MMK_INT8_VNNI:   return "gate.int8.vnni";
    case QWEN_MMK_INT8_AVX2:   return "gate.int8.avx2";
    case QWEN_MMK_INT8_SMMLA:  return "gate.int8.smmla";
    case QWEN_MMK_INT8_SDOT:   return "gate.int8.sdot_mm";
    case QWEN_MMK_Q4_AMX:      return "gate.q4.amx";
    case QWEN_MMK_Q4_VNNI:     return "gate.q4.vnni";
    case QWEN_MMK_Q4_AVX2:     return "gate.q4.avx2";
    case QWEN_MMK_Q4_SMMLA:    return "gate.q4.smmla";
    case QWEN_MMK_Q4_SDOT:     return "gate.q4.sdot_mm";
    case QWEN_MMK_KLEIDI_Q4:   return "gate.q4.kleidi";
    case QWEN_MMK_KLEIDI_I8:   return "gate.int8.kleidi";
    case QWEN_MMK_KLEIDI_BF16: return "gate.bf16.kleidi";
    default:                   return "gate.other";
    }
}

static void json_str(FILE *j, const char *s) {
    fputc('"', j);
    for (; s && *s; s++) {
        if (*s == '"' || *s == '\\') { fputc('\\', j); fputc(*s, j); }
        else if (*s == '\n') fputs("\\n", j);
        else fputc(*s, j);
    }
    fputc('"', j);
}

#define NFEAT 64

#include "qwen_flag_scope.h"

/* --effective-config: what the engine is ACTUALLY doing, per flag.
 *
 * The dispatch map answers "which kernel runs"; this answers the question that kept biting us
 * one level below it -- an operator sets a variable, the engine parses it, and nothing happens
 * because this build or this CPU cannot reach the code it controls.  QWEN_NO_SIMD_QUANT was
 * inert on ARM for exactly that reason and nothing said so.  For every declared flag this
 * prints requested / default / effective and, when they differ, WHY.
 *
 * The scope table is generated from the sources by tools/flag_parity.py, so it cannot drift
 * from the guards the code actually has. */
static unsigned qwen_build_scope(void) {
    unsigned s = 0;
#if defined(__ARM_NEON) || defined(__aarch64__)
    s |= QWEN_FSCOPE_ARM;
#endif
#if defined(__APPLE__)
    s |= QWEN_FSCOPE_APPLE;
#endif
#if defined(__AVX2__) || defined(__x86_64__)
    s |= QWEN_FSCOPE_AVX2;
#endif
#if defined(__AVX512F__)
    s |= QWEN_FSCOPE_AVX512F;
#endif
#if defined(__AVX512VNNI__)
    s |= QWEN_FSCOPE_VNNI;
#endif
#if defined(__AMX_INT8__) && defined(__AMX_TILE__)
    s |= QWEN_FSCOPE_AMX;
#endif
#if defined(QWEN_HAVE_CUDA) || defined(QWEN_HAVE_METAL)
    s |= QWEN_FSCOPE_GPU;
#endif
    return s ? s : QWEN_FSCOPE_ALL;
}

static const char *qwen_scope_names(unsigned sc, char *buf, size_t n) {
    struct { unsigned bit; const char *name; } m[] = {
        { QWEN_FSCOPE_ARM, "arm" }, { QWEN_FSCOPE_AVX2, "avx2" },
        { QWEN_FSCOPE_AVX512F, "avx512f" }, { QWEN_FSCOPE_VNNI, "vnni" },
        { QWEN_FSCOPE_AMX, "amx" }, { QWEN_FSCOPE_APPLE, "apple" },
        { QWEN_FSCOPE_GPU, "gpu" },
    };
    size_t k = 0; buf[0] = 0;
    for (size_t i = 0; i < sizeof m / sizeof m[0]; i++)
        if (sc & m[i].bit)
            k += (size_t)snprintf(buf + k, k < n ? n - k : 0, "%s%s", k ? "," : "", m[i].name);
    if (!buf[0]) snprintf(buf, n, "none");
    return buf;
}

int qwen_effective_config_report(void *out) {
    FILE *f = out ? (FILE *)out : stdout;
    const unsigned build = qwen_build_scope();
    char sb[96], bb[96];
    const int n = (int)(sizeof g_qwen_flag_scope / sizeof g_qwen_flag_scope[0]);
    int set_n = 0, inert_n = 0;

    fprintf(f, "[EFFECTIVE-CONFIG] v=1 build_scope=%s flags=%d\n",
            qwen_scope_names(build, bb, sizeof bb), n);
    fprintf(f, "  %-34s %-10s %-9s %s\n", "flag", "requested", "effective", "scope / reason");
    for (int i = 0; i < n; i++) {
        const char *name = g_qwen_flag_scope[i].name;
        const unsigned sc = g_qwen_flag_scope[i].scope;
        const char *req = getenv(name);
        const int reachable = (sc & build) != 0;
        if (req && *req) set_n++;
        if (!req || !*req) {
            /* Only the flags an operator actually set are worth a line here; the rest are
             * their documented defaults and live in docs/feature-flags.md. */
            continue;
        }
        const char *inert = qwen_pool_flag_inert(name);
        if (!inert) inert = qwen_kleidi_flag_inert(name);
        if (inert) {
            inert_n++;
            fprintf(f, "  %-34s %-10s %-9s IGNORED: %s\n", name, req, "ignored", inert);
            continue;
        }
        if (!strcmp(name, "QWEN_SD_POOL")) {
            const char *resolved = qwen_sd_pool_mode() ? "engine" : "private";
            fprintf(f, "  %-34s %-10s %-9s requested=%s resolved=%s; explicit pool policy\n",
                    name, req, "honoured", req, resolved);
            continue;
        }
        /* A gate flag has a better answer than any static scope: is its kernel compiled? */
        int gate_compiled = 0; const char *kernel = NULL;
        if (qwen_flag_gate_status(name, &gate_compiled, &kernel)) {
            if (!gate_compiled) {
                inert_n++;
                fprintf(f, "  %-34s %-10s %-9s IGNORED: \"%s\" is not compiled into this build\n",
                        name, req, "ignored", kernel ? kernel : "?");
            } else {
                fprintf(f, "  %-34s %-10s %-9s gate: %s\n", name, req, "honoured",
                        kernel ? kernel : "?");
            }
            continue;
        }
        if (reachable) {
            fprintf(f, "  %-34s %-10s %-9s %s\n", name, req, "honoured",
                    qwen_scope_names(sc, sb, sizeof sb));
        } else {
            inert_n++;
            fprintf(f, "  %-34s %-10s %-9s %s -- IGNORED: this build reaches %s\n",
                    name, req, "ignored", qwen_scope_names(sc, sb, sizeof sb),
                    qwen_scope_names(build, bb, sizeof bb));
        }
    }
    {   /* Not a QWEN_ flag, but the one env that can put a second compute scheduler in this
         * process.  Engine ownership now overrides it; say so rather than leaving it silent. */
        const char *ob = getenv("OPENBLAS_NUM_THREADS");
        if (ob && *ob) {
            set_n++;
            if (qwen_blas_own_effective() || qwen_blas_env_overridden()) {
                inert_n++;
                fprintf(f, "  %-34s %-10s %-9s OVERRIDDEN: the engine owns the compute budget; "
                           "BLAS is forced to 1 thread\n", "OPENBLAS_NUM_THREADS", ob, "1");
            } else {
                fprintf(f, "  %-34s %-10s %-9s BLAS runs its own team by request\n",
                        "OPENBLAS_NUM_THREADS", ob, ob);
            }
        }
        fprintf(f, "  %-34s %-10s %-9s %s\n", "blas.ownership", "-",
                qwen_blas_own_effective() ? "engine" : "blas",
                qwen_blas_own_effective()
                  ? "engine owns the budget: BLAS is held at one thread and cannot escape it"
                  : "BLAS keeps its own team (no thread control here, or ownership not claimed)");
    }
    fprintf(f, "  %d flag%s set in the environment, %d of them IGNORED by this build\n",
            set_n, set_n == 1 ? "" : "s", inert_n);
    if (inert_n)
        fprintf(f, "  WARNING: an ignored flag is a configuration that is not being applied.\n");
    return inert_n;
}

int qwen_dispatch_map_report(void *out, const char *json_path) {
    FILE *f = out ? (FILE *)out : stderr;
    feat_t feats[NFEAT];
    int n = 0;
    const char *why = NULL;
    char tmp[64];

    /* AVX-512BW without VNNI gets its own B=1 signed-dot candidates; the
     * existing B>1 integer matmat family remains AVX2-width. */
    {
        const int compiled = qwen_avx512bw_int8_gemv_compiled();
        const int supported = qwen_avx512bw_int8_gemv_supported();
        const int enabled = qwen_avx512bw_int8_gemv_enabled();
        const char *reason = !compiled ? "candidate requires AVX-512F/BW without compiled VNNI"
                           : !supported ? "compiled, but this CPU lacks AVX-512F/BW/VL or AVX2/FMA"
                           : enabled ? "experimental AVX-512BW signed-widening GEMV selected before AVX2"
                                     : "default OFF; set QWEN_AVX512_INT8_GEMV=1 for A/B";
        row(&feats[n++], "matvec.int8.avx512bw-emulated-dot-gemv", yn(compiled), yn(supported),
            "QWEN_AVX512_INT8_GEMV", (compiled && supported && enabled) ? "ON" : "OFF", reason);
    }
    {
        const int compiled = qwen_avx512bw_q4_gemv_compiled();
        const int supported = qwen_avx512bw_q4_gemv_supported();
        const int enabled = qwen_avx512bw_q4_gemv_enabled();
        const char *reason = !compiled ? "candidate requires AVX-512F/BW without compiled VNNI"
                           : !supported ? "compiled, but this CPU lacks AVX-512F/BW/VL or AVX2/FMA"
                           : enabled ? "experimental AVX-512BW Q4 GEMV selected before AVX2"
                                     : "default OFF; set QWEN_AVX512_Q4_GEMV=1 for A/B";
        row(&feats[n++], "matvec.q4.avx512bw-emulated-dot-gemv", yn(compiled), yn(supported),
            "QWEN_AVX512_Q4_GEMV", (compiled && supported && enabled) ? "ON" : "OFF", reason);
    }

    /* Legacy AVX2 GEMV is a separate policy-controlled candidate, not the
     * existing AVX2 B>1 matmat gate. Keep compiled, runtime support and the
     * opt-in policy visible independently. */
    {
        const int compiled = qwen_avx2_int8_gemv_compiled();
        const int supported = qwen_avx2_int8_gemv_supported();
        const int enabled = qwen_avx2_int8_gemv_enabled();
        const char *reason = !compiled ? "candidate not compiled into this build"
                           : !supported ? "compiled, but this CPU lacks AVX2"
                           : enabled ? "experimental signed-widening GEMV selected before other x86 GEMV paths"
                                     : "default OFF; set QWEN_AVX2_INT8_GEMV=1 for A/B";
        row(&feats[n++], "matvec.int8.avx2-emulated-dot-gemv", yn(compiled), yn(supported),
            "QWEN_AVX2_INT8_GEMV", (compiled && supported && enabled) ? "ON" : "OFF", reason);
    }

    /* Q4 B=1 is a separate complete-call candidate.  It intentionally does not
     * reuse the B>1 QWEN_AVX2MM gate: that gate's batch threshold is irrelevant
     * to a single streaming projection. */
    {
        const int compiled = qwen_avx2_q4_gemv_compiled();
        const int supported = qwen_avx2_q4_gemv_supported();
        const int enabled = qwen_avx2_q4_gemv_enabled();
        const char *reason = !compiled ? "candidate not compiled into this build"
                           : !supported ? "compiled, but this CPU lacks AVX2"
                           : enabled ? "experimental Q4 integer GEMV selected before native/f32 Q4 paths"
                                     : "default OFF; set QWEN_AVX2_Q4_GEMV=1 for A/B";
        row(&feats[n++], "matvec.q4.avx2-emulated-dot-gemv", yn(compiled), yn(supported),
            "QWEN_AVX2_Q4_GEMV", (compiled && supported && enabled) ? "ON" : "OFF", reason);
    }

    /* ---- Talker prefill: the predicate that was invisible ------------------------ */
    {
        int on = qwen_prefill_matmat_resolved(&why);
        int compiled =
#if (defined(__AMX_BF16__) && defined(__AMX_TILE__)) || defined(__AVX512BF16__) || \
    (defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC) && !defined(__APPLE__))
            1;
#else
            0;
#endif
        /* `supported` is a hardware/permission fact, not the result of an operator
         * kill switch.  The effective env-controlled choice is reported separately by
         * qwen_prefill_matmat_resolved().  In particular, QWEN_NO_BF16_MATMUL=1 must
         * not make an AVX-512-BF16 CPU look as if it lacked the instruction. */
        int supported = 0;
#if defined(__AMX_BF16__) && defined(__AMX_TILE__)
        supported |= qwen_amx_bf16_available();
#endif
#if defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC) && !defined(__APPLE__)
        supported = 1;
#endif
#if defined(__AVX512BF16__)
        supported |= __builtin_cpu_supports("avx512bf16") ? 1 : 0;
#endif
        row(&feats[n++], "talker.prefill.matmat_bf16", yn(compiled), yn(supported),
            "QWEN_PREFILL_MATMAT", onoff(on), why);
#ifdef USE_BLAS
        row(&feats[n++], "talker.prefill.f32_blas_fallback", "yes", "yes", NULL, onoff(!on),
            on ? "not taken: bf16 matmat selected"
               : "bf16->f32 weight convert (single-threaded) + SGEMM on the TTFA path");
#else
        row(&feats[n++], "talker.prefill.f32_blas_fallback", "no", "-", NULL, "OFF",
            "no BLAS in this build");
#endif
        {
            const char *e = getenv("QWEN_PREFILL_SLICE");
            int slice = (e && e[0]) ? atoi(e) : 0;
            static char why_slice[128];
            if (slice < 0)
                snprintf(why_slice, sizeof(why_slice),
                         "slices of %d tokens, idle worker included - PARITY TEST setting, "
                         "not a product arm", -slice);
            else
                snprintf(why_slice, sizeof(why_slice),
                         slice > 0 ? "admission prefilled in slices of %d tokens, one per frame iteration"
                                   : "monolithic inline prefill (product default)", slice);
            row(&feats[n++], "talker.prefill.slice", "yes", "-", "QWEN_PREFILL_SLICE",
                onoff(slice != 0), why_slice);
        }
        row(&feats[n++], "talker.prefix_cache", "yes", "-", "QWEN_PREFIX_CACHE",
            onoff(qwen_prefix_cache_enabled()), "default ON; QWEN_PREFIX_CACHE=0 disables");
    }

    /* ---- Code predictor ----------------------------------------------------------- */
    {
        int req = qwen_cp_prefill2_requested();
        row(&feats[n++], "cp.prefill2",
#if defined(__AVX512VNNI__)
            "yes", yn(__builtin_cpu_supports("avx512vnni")),
#else
            "yes", "-",
#endif
            "QWEN_CP_PREFILL2", req ? "ON*" : "OFF",
            req ?
#if defined(__AVX512VNNI__)
                "default ON with AVX-512 VNNI; * needs every CP layer int8/int4 (weights decide at load)"
#else
                "explicit env; * needs every CP layer int8/int4 (weights decide at load)"
#endif
                : (getenv("QWEN_CP_PREFILL2") ? "explicit env off" : "opt-in on this ISA, env unset"));
        const char *e = getenv("QWEN_CP_PREC");
        row(&feats[n++], "cp.precision", "-", "-", "QWEN_CP_PREC",
            (e && e[0]) ? e : "follows",
            (e && e[0]) ? "CP precision decoupled from the Talker" : "follows --int8/--int4 of the Talker");
    }

    /* ---- Speech decoder / server ------------------------------------------------- */
    {
        const char *e = getenv("QWEN_DECODER_BATCH");
        int on = (e && atoi(e) != 0);
        row(&feats[n++], "decoder.batch", "yes", "-", "QWEN_DECODER_BATCH", onoff(on),
            e ? "explicit env" : "unset here; the SERVER sets it to 1 at start unless the env says 0");
        row(&feats[n++], "decoder.int8", yn(qwen_sd_int8_available()), yn(qwen_sd_int8_available()),
            "QWEN_SD_INT8", onoff(qwen_sd_int8_enabled()),
#if defined(__AVX512VNNI__)
            "default ON with AVX-512 VNNI when the int8 decoder kernels are available"
#elif defined(__AVX2__)
            "opt-in AVX2 signed-widening decoder path; default OFF pending AVX2 host qualification"
#else
            "opt-in on this ISA (measured slower on the first frame elsewhere)"
#endif
            );
        row(&feats[n++], "decoder.res1_v2", yn(qwen_conv1d_int8_v2_available()),
            yn(qwen_conv1d_int8_v2_available()), "QWEN_SD_RES1_V2",
            onoff(qwen_sd_res1_v2_active()),
            qwen_sd_res1_v2_active()
                ? "direct dilated int8 DL-4 conv (AVX2 signed widening / VNNI / Arm dotprod; per-position activation scale, per-(channel,tap) weight scale). Any shape: it serves res1, res2, the rectangular initial/pre convs and wide channels; the in_ch<=768 square-only bound applies to the v1/Design-D paths, not here"
                : "opt-in (QWEN_SD_RES1_V2=1); off: the residual convs run on the im2col panel kernel (v1 int8 where the square/768 shape allows, f32 otherwise)");
        row(&feats[n++], "decoder.pre_up_bf16", "yes",
            yn(qwen_avx512_bf16_matmat_available() || qwen_kleidi_bf16_enabled()),
            "QWEN_SD_BF16_PREUP", onoff(qwen_sd_bf16_preup_active()),
            qwen_sd_bf16_preup_active()
                ? "persistent BF16 pre-transformer weights; native AVX-512 BF16 or Arm KleidiAI prepared rows"
                : "default OFF; f32 decoder pre-transformer path remains the control");
        row(&feats[n++], "decoder.multislot", yn(qwen_conv1d_int8_v2_multi_available()),
            yn(qwen_conv1d_int8_v2_multi_available()), "QWEN_SD_MULTISLOT",
            onoff(qwen_sd_multislot_active()),
            qwen_sd_multislot_active()
                ? "lane cohort DL-4 shared-weight sweep for 2..3 equal-length slots; rectangular in/out shapes supported"
                : "default OFF; requires QWEN_SD_INT8=1, QWEN_SD_RES1_V2=1 and a shape-compatible lane cohort");
        row(&feats[n++], "decoder.convt_stack", "yes", "yes", "QWEN_SD_CONVT_STACK",
            onoff(qwen_sd_convt_stack_active()),
            qwen_sd_convt_stack_active()
                ? "ConvT as one un-expanded GEMM per layer with the two-tap/carry/bias epilogue (exact; C12-WIN-11 A)"
                : "opt-in (QWEN_SD_CONVT_STACK=1); off: per-tap GEMMs + scatter + full-length intermediate");
        row(&feats[n++], "decoder.glue_fused", yn(qwen_conv1d_int8_v2_available()),
            yn(qwen_conv1d_int8_v2_available()), "QWEN_SD_GLUE", onoff(qwen_sd_glue_active()),
            qwen_sd_glue_active()
                ? "residual unit fused on the DL-4 path (AVX2 signed widening / VNNI / Arm dotprod): context-aware conv, residual in the epilogue, no ext/full/cut/add passes"
                : "opt-in (QWEN_SD_GLUE=1, needs QWEN_SD_RES1_V2=1); off: the control residual unit");
        row(&feats[n++], "decoder.mode", "yes", "yes", "QWEN_DECODER_BATCH",
            qwen_sd_decoder_mode(),
            "resolved decoder leaf: ragged AMX only when batch+exact-stream+AMX decoder support all hold; otherwise per-item");
        row(&feats[n++], "decoder.design_d",
#if defined(__AMX_INT8__) && defined(__AMX_TILE__)
            "yes",
#else
            "no",
#endif
            yn(qwen_amx_int8_available()), "QWEN_SD_AMX_D", onoff(qwen_sd_amx_d_active()),
            qwen_sd_amx_d_active() ? "persistent AMX INT8 Design-D decoder representation is active"
                                   : "Design-D unavailable or not requested; decoder uses its resolved fallback");
        row(&feats[n++], "decoder.fused_residual",
#if defined(__AMX_INT8__) && defined(__AMX_TILE__)
            "yes",
#else
            "no",
#endif
            yn(qwen_amx_int8_available()), "QWEN_SD_FUSED_RESIDUAL",
            onoff(qwen_sd_fused_residual_active()),
            qwen_sd_fused_residual_active() ? "AMX Design-D fused residual is shape-gated at 1x1 same-width projections"
                                            : "not active; ordinary projection plus residual/fallback remains available");
        row(&feats[n++], "decoder.stream_strip", "yes", yn(qwen_amx_int8_available()),
            "QWEN_SD_STREAM_STRIP", onoff(qwen_sd_stream_strip_active()),
            qwen_sd_stream_strip_active() ? "warm Design-D range slice is active"
                                          : "warm strip unavailable or not requested");
    }

    /* ---- Prepacks (persistent weight layouts) ----------------------------------- */
    {
        row(&feats[n++], "prepack.amx",
#if defined(__AMX_INT8__) && defined(__AMX_TILE__)
            "yes",
#else
            "no",
#endif
            yn(qwen_amx_int8_available() || qwen_amx_bf16_available()), "QWEN_AMX_PREPACK",
            onoff(qwen_amx_prepack_requested()), "opt-in; parent packs, prefork workers inherit");
        row(&feats[n++], "prepack.vnni",
#if defined(__AVX512VNNI__)
            "yes", yn(__builtin_cpu_supports("avx512vnni")),
#else
            "no", "-",
#endif
            "QWEN_VNNI_PREPACK", onoff(qwen_vnni_prepack_requested()),
            "opt-in; int8 only; REJECTED 2026-09-03 on Zen5 (+1.1% Talker, +3.5% CP, +1.4 GB)");
    }

    /* ---- B=1 matvec dot products (not gate rows) --------------------------------- */
    {
        /* Unconditional, because the interesting case is the build that has NO row below:
         * AVX2 and AVX-512F without VNNI have int8/q4 GEMM but no integer GEMV, and used to
         * say nothing at all while every B=1 call ran the f32 fused twin. */
        row(&feats[n++], "matvec.int8.native", yn(qwen_int8_gemv_native()),
            yn(qwen_int8_gemv_native()), NULL, onoff(qwen_int8_gemv_native()),
            qwen_int8_gemv_native() ? "native integer GEMV"
                                    : "NO integer GEMV in this build: B=1 runs the f32 fused twin");
        row(&feats[n++], "matmat.int8.family", "-", "-", NULL, "see reason",
            qwen_matmat_family_int8());
        row(&feats[n++], "matmat.q4.family", "-", "-", NULL, "see reason",
            qwen_matmat_family_q4());
        row(&feats[n++], "matmat.bf16.family", "-", "-", NULL, "see reason",
            qwen_matmat_family_bf16());
        {
            /* This is the largest B accepted by any optimized int8 matmat gate for the
             * standard probe shape. It is not a server batch clamp: the normal B>1 matmat
             * route falls back to the fixed-B/generic f32 twin, unless matvec is forced. */
            char mb[24]; int ceil_b = qwen_matmat_int8_max_b();
            snprintf(mb, sizeof mb, "%d", ceil_b);
            row(&feats[n++], "matmat.int8.batch_ceiling", "-", "-", NULL, mb,
                ceil_b ? "max optimized int8 B on 4096x4096 probe; above it B>1 uses fixed-B/"
                         "generic f32-accum matmat (unless matvec is forced)"
                       : "no optimized int8 gate on 4096x4096 probe; B>1 uses fixed-B/generic "
                         "f32-accum matmat (unless matvec is forced)");
        }
        row(&feats[n++], "matvec.q4.native", yn(qwen_q4_gemv_native()),
            yn(qwen_q4_gemv_native()), NULL, onoff(qwen_q4_gemv_native()),
            qwen_q4_gemv_native() ? "native q4 GEMV"
                                  : "NO q4 GEMV in this build: B=1 runs the f32 fused twin");
#if defined(__AVX512BF16__)
        row(&feats[n++], "matvec.bf16.dpbf16", "yes", yn(__builtin_cpu_supports("avx512bf16")),
            "QWEN_NO_BF16DOT", onoff(qwen_bf16dot_enabled()), "default ON; QWEN_NO_BF16DOT=1 disables");
#endif
#if defined(__AVX512VNNI__)
        {
            const char *e = getenv("QWEN_NO_VNNI");
            int off = (e && e[0] == '1');
            row(&feats[n++], "matvec.int8.vnni", "yes", yn(__builtin_cpu_supports("avx512vnni")),
                "QWEN_NO_VNNI", onoff(!off), off ? "QWEN_NO_VNNI=1" : "default ON (vpdpbusd)");
            snprintf(tmp, sizeof tmp, "v%d", qwen_q4_vnni_variant());
            row(&feats[n++], "matvec.q4.vnni_variant", "yes", yn(__builtin_cpu_supports("avx512vnni")),
                "QWEN_Q4_VNNI_V4", tmp, "v3 default (wins on Zen5); QWEN_Q4_VNNI_V4=1 -> v4; QWEN_Q4_VNNI_V3=0 -> v2");
        }
#endif
#if defined(__ARM_FEATURE_DOTPROD)
        {
            const char *e = getenv("QWEN_NO_SDOT");
            int off = (e && e[0] == '1');
            row(&feats[n++], "matvec.int8.sdot", "yes", "yes", "QWEN_NO_SDOT", onoff(!off),
                off ? "QWEN_NO_SDOT=1" : "default ON (vdotq_s32)");
        }
#endif
#if defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC)
        row(&feats[n++], "matvec.bf16.bfdot", "yes", "yes", "QWEN_ARM_BFDOT", onoff(qwen_arm_bfdot_on()),
            qwen_arm_bfdot_on() ? "explicit QWEN_ARM_BFDOT=1" : "opt-in; default is the NEON 2-row fused bf16 matvec");
#endif
#if defined(__aarch64__)
        row(&feats[n++], "q8repack.neon", "yes", yn(qwen_q8r_supported()),
            "QWEN_NO_Q8REPACK", onoff(qwen_q8r_enabled()),
            !qwen_q8r_supported() ? "CPU has neither dotprod nor i8mm"
                                  : (qwen_q8r_enabled() ? "q8_0 4-row repack, SDOT (SMMLA where i8mm exists); default ON"
                                                        : "QWEN_NO_Q8REPACK=1"));
#endif
    }

    /* ---- KleidiAI (Arm) ------------------------------------------------------------ */
    {
        int build =
#if defined(__ARM_FEATURE_MATMUL_INT8)
            1;
#else
            0;
#endif
        row(&feats[n++], "kleidi.enabled", yn(build), yn(qwen_kleidi_supported()), "QWEN_NO_KLEIDI",
            onoff(qwen_kleidi_enabled()),
            !build ? "not compiled (needs an i8mm target)" :
            !qwen_kleidi_supported() ? "compiled, CPU lacks i8mm/bf16" : "default ON when supported");
        if (build) {
            row(&feats[n++], "kleidi.int8", "yes", yn(qwen_kleidi_supported()), "QWEN_NO_KAI_I8",
                onoff(qwen_kleidi_i8_enabled()), "SMMLA int8 GEMM/GEMV via KleidiAI");
            row(&feats[n++], "kleidi.bf16", "yes", yn(qwen_kleidi_supported()), "QWEN_NO_KAI_BF16",
                onoff(qwen_kleidi_bf16_enabled()), "BFMMLA bf16 GEMM/GEMV via KleidiAI");
            row(&feats[n++], "kleidi.prefill", "yes", yn(qwen_kleidi_supported()), "QWEN_KAI_OPS",
                onoff(qwen_kleidi_prefill_enabled()), "bf16 prefill through KleidiAI (needs kleidi.bf16)");
            row(&feats[n++], "kleidi.qkv_fused", "yes", yn(qwen_kleidi_supported()), "QWEN_KAI_QKV_FUSED",
                onoff(qwen_kleidi_qkv_fused_on()), "one packed LHS, Q/K/V tiles scheduled from it (default ON)");
            row(&feats[n++], "kleidi.lhs_sym", "yes", yn(qwen_kleidi_supported()), "QWEN_KAI_LHS",
                onoff(qwen_kleidi_lhs_sym()), "symmetric LHS quantisation (opt-in, QWEN_KAI_LHS=sym)");
            snprintf(tmp, sizeof tmp, "%d", qwen_kleidi_nchunk_value());
            row(&feats[n++], "kleidi.nchunk", "-", "-", "QWEN_KAI_NCHUNK", tmp,
                getenv("QWEN_KAI_NCHUNK") ? "explicit env (rows per bf16 GEMM chunk)" : "compiled default 384 rows per bf16 GEMM chunk");
        }
    }

    /* ---- Thread pool ---------------------------------------------------------------- */
    {
        int spin = qwen_pool_spin_value(), narrow = qwen_pool_narrow_value();
        if (spin < 0) {
            row(&feats[n++], "pool.spin", "-", "-", "QWEN_POOL_SPIN", "n/a",
                "GCD dispatch on macOS: no spin knob");
        } else {
            snprintf(tmp, sizeof tmp, "%d", spin);
            row(&feats[n++], "pool.spin", "-", "-", "QWEN_POOL_SPIN", tmp,
                getenv("QWEN_POOL_SPIN") ? "explicit env"
#if defined(__linux__) && defined(__aarch64__)
                : "compiled default 65536 (Linux/aarch64)");
#else
                : "compiled default 4096");
#endif
            row(&feats[n++], "pool.narrow", "-", "-", "QWEN_POOL_NARROW", onoff(narrow),
                "clamp workers to nt-1 (main thread also works)");
        }
        row(&feats[n++], "blas.owned_effective", yn(qwen_blas_own_get()),
            yn(qwen_blas_own_effective()), "QWEN_BLAS_OWN",
            onoff(qwen_blas_own_effective()),
            qwen_blas_own_effective()
                ? "BLAS held at one thread; decoder SGEMMs partitioned on the engine pool"
                : (qwen_blas_own_get()
                       ? "requested, but this build has no BLAS thread control: NOT partitioned, "
                         "so the vendor BLAS keeps its own team and no nesting is created"
                       : "BLAS runs its own team by request"));
        char pool_reason[160];
        const char *pool_requested = getenv("QWEN_SD_POOL");
        const char *pool_resolved = qwen_sd_pool_mode() ? "engine" : "private";
        snprintf(pool_reason, sizeof pool_reason,
                 "requested=%s resolved=%s; %s",
                 pool_requested ? pool_requested : "unset", pool_resolved,
                 pool_resolved[0] == 'e'
                     ? "decoder tiles run on the engine pool (inline inside a region)"
                     : "decoder raises its own worker team");
        row(&feats[n++], "decoder.pool", "-", "-", "QWEN_SD_POOL",
            pool_resolved, pool_reason);
        {
            const char *lm_step = "", *lm_dec = "";
            qwen_lane_masks(&lm_step, &lm_dec);
            static char lane_reason[256];
            if (lm_dec && lm_dec[0])
                snprintf(lane_reason, sizeof lane_reason,
                         "step cpus %s (engine pool) · decoder cpus %s (private team); one decoder unit "
                         "in flight per slot, the frame loop never waits for another slot's decode",
                         lm_step, lm_dec);
            else
                snprintf(lane_reason, sizeof lane_reason,
                         "off: the decoder runs inline on the frame loop (QWEN_SD_LANE_SPLIT=N reserves the "
                         "last N cpus of the worker mask for a private decoder team)");
            row(&feats[n++], "decoder.lane", "yes", "-", "QWEN_SD_LANE_SPLIT",
                (lm_dec && lm_dec[0]) ? "ON" : "OFF", lane_reason);
        }
        row(&feats[n++], "pool.nested_dispatch", "-", "-", NULL,
            onoff(qwen_pool_nested_dispatch_ok()),
            qwen_pool_nested_dispatch_ok() ? "a task on the pool may dispatch again"
                                           : "a task on the pool must run nested work inline");
        row(&feats[n++], "pool.concurrent_submit", "-", "-", NULL,
            onoff(qwen_pool_concurrent_submit_ok()),
            qwen_pool_concurrent_submit_ok() ? "two threads may submit at once"
                                             : "submission must be serialised by the caller");
        snprintf(tmp, sizeof tmp, "%d", qwen_get_threads());
        row(&feats[n++], "pool.submit_priority", "-", "-", "QWEN_PREFILL_LOW_MS",
            onoff(qwen_pool_priority_ok()),
            qwen_pool_priority_ok() ? "a submitter can step aside for the frame loop (LOW)"
                                    : "no submit priority here: QWEN_PREFILL_LOW_MS is ignored");
        row(&feats[n++], "pool.threads", "-", "-", NULL, tmp, "matvec threads in this process (-j)");
    }

    /* ---- Persistent execution regions ------------------------------------------------
     * The engine prints the resolved answer, with the model's own shapes, at the first
     * batched step ("[cp] transformer step as one parallel region: ..."). These rows are
     * what can be known BEFORE traffic: whether an in-region runner exists at all, whether
     * the pool can hold a team, and which knobs are set. */
    {
        const char *rb = qwen_region_i8_backend();
        int held = qwen_parallel_team();
        char tb[24]; snprintf(tb, sizeof tb, "%d", held);
        row(&feats[n++], "region.int8_runner", "-", "-", NULL,
            (rb[0] == 'n' && rb[1] == 'o') ? "OFF" : "ON", rb);
        row(&feats[n++], "region.team", "-", "-", NULL, tb,
            held >= 2 ? "holdable team: a region can keep its workers between phases"
                      : "no holdable team (needs >= 2): every region predicate is off");
        row(&feats[n++], "region.cp", "-", "-", "QWEN_CP_REGION", "see reason",
            "CP transformer step as one parallel region; needs int8 weights (no q4), B in 2..16 "
            "and the runner above. QWEN_CP_REGION=0 restores the dispatched path");
        row(&feats[n++], "region.cp_frame", "-", "-", "QWEN_CP_FRAME_REGION", "see reason",
            "all 16 CP steps in one pool entry; also needs the batched heads");
        row(&feats[n++], "region.cp_batch_head", "-", "-", "QWEN_CP_BATCH_HEAD", "see reason",
            "MTP projection and lm_heads once for all active slots");
        row(&feats[n++], "region.talker", "-", "-", "QWEN_TK_REGION", "see reason",
            "batched Talker step as one parallel region, same conditions as region.cp. "
            "QWEN_TK_REGION=0 restores the dispatched path");
    }

    /* ---- Print ---------------------------------------------------------------------- */
    const char *cls = isa_class();
    fprintf(f, "[DISPATCH] v=1 pid=%d isa_class=%s build=%s simd=%s src=%s\n",
            (int)getpid(), cls, QWEN_GIT_REV, QWEN_SIMD_PROFILE, QWEN_SOURCE_FP);
    fprintf(f, "  %-34s %-8s %-9s %-30s %-10s %s\n",
            "feature", "compiled", "supported", "env", "resolved", "reason");
    for (int i = 0; i < n; i++) {
        char envc[80];
        if (feats[i].env_name) snprintf(envc, sizeof envc, "%s=%s", feats[i].env_name, feats[i].env_val);
        else snprintf(envc, sizeof envc, "-");
        fprintf(f, "  %-34s %-8s %-9s %-30s %-10s %s\n", feats[i].id, feats[i].compiled,
                feats[i].supported, envc, feats[i].resolved, feats[i].reason);
    }

    qwen_mm_gate_desc_t gates[QWEN_MMK_COUNT];
    int ng = 0;
    for (int m = 1; m < QWEN_MMK_COUNT; m++)
        if (qwen_mm_gate_describe(m, &gates[ng])) ng++;
    fprintf(f, "[DISPATCH-GATE] v=1 rows=%d (resolved by qwen_mm_use at B=min_b; min_b/rows/cols = resolved(compiled))\n", ng);
    fprintf(f, "  %-18s %-24s %-8s %-9s %-4s %-9s %-13s %-13s %-22s %s\n",
            "gate", "kernel", "compiled", "supported", "on", "min_b", "min_rows", "min_cols", "switch", "reason");
    for (int i = 0; i < ng; i++) {
        qwen_mm_gate_desc_t *g = &gates[i];
        char mb[16], mr[16], mc[16], sw[40];
        snprintf(mb, sizeof mb, "%d(%d)", g->min_b, g->compiled_min_b);
        snprintf(mr, sizeof mr, "%d(%d)", g->min_rows, g->compiled_min_rows);
        snprintf(mc, sizeof mc, "%d(%d)", g->min_cols, g->compiled_min_cols);
        snprintf(sw, sizeof sw, "%s%s", g->on_env ? g->on_env : g->off_env, g->on_env ? " (opt-in)" : "");
        fprintf(f, "  %-18s %-24s %-8s %-9s %-4s %-9s %-13s %-13s %-22s %s\n",
                gate_id(g->mmk), g->name, yn(g->compiled), yn(g->supported),
                g->compiled ? onoff(g->on) : "-", mb, mr, mc, sw,
                g->compiled ? g->reason : "not compiled");
    }
    fflush(f);

    /* ---- JSON ----------------------------------------------------------------------- */
    if (!json_path || !json_path[0]) json_path = getenv("QWEN_DISPATCH_JSON");
    if (json_path && json_path[0]) {
        FILE *j = fopen(json_path, "w");
        if (!j) { fprintf(stderr, "dispatch-map: cannot write %s\n", json_path); return 1; }
        fprintf(j, "{\n  \"v\": 1,\n  \"pid\": %d,\n  \"isa_class\": ", (int)getpid());
        json_str(j, cls);
        fprintf(j, ",\n  \"build\": "); json_str(j, QWEN_GIT_REV);
        fprintf(j, ",\n  \"simd\": ");  json_str(j, QWEN_SIMD_PROFILE);
        fprintf(j, ",\n  \"source_fp\": ");  json_str(j, QWEN_SOURCE_FP);
        fprintf(j, ",\n  \"threads\": %d,\n  \"features\": [\n", qwen_get_threads());
        for (int i = 0; i < n; i++) {
            fprintf(j, "    {\"id\": "); json_str(j, feats[i].id);
            fprintf(j, ", \"compiled\": "); json_str(j, feats[i].compiled);
            fprintf(j, ", \"supported\": "); json_str(j, feats[i].supported);
            fprintf(j, ", \"env\": "); json_str(j, feats[i].env_name ? feats[i].env_name : "");
            fprintf(j, ", \"env_value\": "); json_str(j, feats[i].env_name ? feats[i].env_val : "");
            fprintf(j, ", \"resolved\": "); json_str(j, feats[i].resolved);
            fprintf(j, ", \"reason\": "); json_str(j, feats[i].reason);
            fprintf(j, "}%s\n", i + 1 < n ? "," : "");
        }
        {
            const char *prefill_why = NULL;
            const int prefill_on = qwen_prefill_matmat_resolved(&prefill_why);
            const char *batch_env = getenv("QWEN_DECODER_BATCH");
            const int batch_requested = batch_env && atoi(batch_env) != 0;
            fprintf(j, "  ],\n  \"serving\": {\n");
            fprintf(j, "    \"decoder_batch_requested\": %s,\n", batch_requested ? "true" : "false");
            fprintf(j, "    \"decoder_mode\": "); json_str(j, qwen_sd_decoder_mode());
            fprintf(j, ",\n    \"design_d_active\": %s,\n", qwen_sd_amx_d_active() ? "true" : "false");
            fprintf(j, "    \"fused_residual_active\": %s,\n",
                    qwen_sd_fused_residual_active() ? "true" : "false");
            fprintf(j, "    \"stream_strip_active\": %s,\n",
                    qwen_sd_stream_strip_active() ? "true" : "false");
            fprintf(j, "    \"res1_v2_active\": %s,\n", qwen_sd_res1_v2_active() ? "true" : "false");
            fprintf(j, "    \"pre_up_bf16_active\": %s,\n", qwen_sd_bf16_preup_active() ? "true" : "false");
            fprintf(j, "    \"decoder_multislot_active\": %s,\n",
                    qwen_sd_multislot_active() ? "true" : "false");
            { const char *lm_step = "", *lm_dec = ""; qwen_lane_masks(&lm_step, &lm_dec);
              fprintf(j, "    \"decoder_lane_active\": %s,\n", (lm_dec && lm_dec[0]) ? "true" : "false");
              fprintf(j, "    \"decoder_lane_elastic\": %s,\n", qwen_lane_elastic() ? "true" : "false"); }
            fprintf(j, "    \"decoder_pool\": ");
            json_str(j, qwen_sd_pool_mode() ? "engine" : "private");
            fprintf(j, ",\n    \"talker_cp_int8_backend\": ");
            json_str(j, qwen_matmat_family_int8());
            fprintf(j, ",\n    \"q4_backend\": ");
            json_str(j, qwen_matmat_family_q4());
            fprintf(j, ",\n    \"bf16_backend\": ");
            json_str(j, qwen_matmat_family_bf16());
            fprintf(j, ",\n    \"prefill_matmat_active\": %s,\n", prefill_on ? "true" : "false");
            fprintf(j, "    \"prefill_reason\": "); json_str(j, prefill_why ? prefill_why : "unknown");
            fprintf(j, ",\n    \"kleidi_active\": %s\n", qwen_kleidi_enabled() ? "true" : "false");
            fprintf(j, "  },\n  \"gates\": [\n");
        }
        for (int i = 0; i < ng; i++) {
            qwen_mm_gate_desc_t *g = &gates[i];
            fprintf(j, "    {\"id\": "); json_str(j, gate_id(g->mmk));
            fprintf(j, ", \"kernel\": "); json_str(j, g->name);
            fprintf(j, ", \"mmk\": %d, \"compiled\": %s, \"supported\": %s, \"on\": %s",
                    g->mmk, g->compiled ? "true" : "false", g->supported ? "true" : "false",
                    (g->compiled && g->on) ? "true" : "false");
            fprintf(j, ", \"min_b\": %d, \"compiled_min_b\": %d, \"max_b\": %d", g->min_b, g->compiled_min_b, g->max_b);
            fprintf(j, ", \"min_rows\": %d, \"min_cols\": %d", g->min_rows, g->min_cols);
            fprintf(j, ", \"off_env\": "); json_str(j, g->off_env ? g->off_env : "");
            fprintf(j, ", \"on_env\": ");  json_str(j, g->on_env ? g->on_env : "");
            fprintf(j, ", \"minb_env\": "); json_str(j, g->minb_env ? g->minb_env : "");
            fprintf(j, ", \"reason\": "); json_str(j, g->compiled ? g->reason : "not compiled");
            fprintf(j, "}%s\n", i + 1 < ng ? "," : "");
        }
        fprintf(j, "  ]\n}\n");
        fclose(j);
    }
    return 0;
}
