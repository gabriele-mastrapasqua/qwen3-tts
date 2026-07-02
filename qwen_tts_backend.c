/*
 * qwen_tts_backend.c — backend resolver + CPU default (G1 seam).
 *
 * Compiled into the GPU build targets (`make metal` / `make cuda`), NOT into the
 * default `make blas` (which stays byte-identical). The CPU thunks forward to
 * the existing kernels, so a GPU backend that only implements SOME ops can be
 * mixed with CPU for the rest by copying the CPU thunk into the empty slots.
 */

#include "qwen_tts_backend.h"
#include "qwen_tts_kernels.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>

#ifdef QWEN_HAVE_METAL
#include "qwen_tts_metal.h"
#endif
#ifdef QWEN_HAVE_CUDA
#include "qwen_tts_cuda.h"
#endif

/* ---- CPU thunks (the default; always available) ------------------------- */
static void cpu_matvec_bf16(qwen_backend_t *b, float *y,
                            const uint16_t *W, const float *x, int rows, int cols) {
    (void)b;
    qwen_matvec_bf16(y, W, x, rows, cols);
}
static void cpu_matmat_bf16(qwen_backend_t *b, float *Y,
                            const uint16_t *W, const float *X, int rows, int cols, int B) {
    (void)b;
    qwen_matmat_bf16(Y, W, X, rows, cols, B);
}
static void cpu_free(qwen_backend_t *b) { free(b); }

static qwen_backend_t *make_cpu(void) {
    qwen_backend_t *b = calloc(1, sizeof(*b));
    if (!b) return NULL;
    b->kind = QWEN_BACKEND_CPU;
    b->name = "cpu";
    b->impl = NULL;
    b->matvec_bf16 = cpu_matvec_bf16;
    b->matmat_bf16 = cpu_matmat_bf16;
    b->free = cpu_free;
    return b;
}

#ifdef QWEN_HAVE_METAL
static void metal_matvec_bf16(qwen_backend_t *b, float *y,
                              const uint16_t *W, const float *x, int rows, int cols) {
    qwen_metal_matvec_bf16(b->impl, y, W, x, rows, cols);
}
static void metal_matmat_bf16(qwen_backend_t *b, float *Y,
                              const uint16_t *W, const float *X, int rows, int cols, int B) {
    qwen_metal_matmat_bf16(b->impl, Y, W, X, rows, cols, B);
}
static void metal_free(qwen_backend_t *b) {
    if (b->impl) qwen_metal_free(b->impl);
    free(b);
}
#endif

#ifdef QWEN_HAVE_CUDA
static void cuda_matvec_bf16(qwen_backend_t *b, float *y,
                             const uint16_t *W, const float *x, int rows, int cols) {
    qwen_cuda_matvec_bf16(b->impl, y, W, x, rows, cols);
}
static void cuda_matmat_bf16(qwen_backend_t *b, float *Y,
                             const uint16_t *W, const float *X, int rows, int cols, int B) {
    qwen_cuda_matmat_bf16(b->impl, Y, W, X, rows, cols, B);
}
static void cuda_free(qwen_backend_t *b) {
    if (b->impl) qwen_cuda_free(b->impl);
    free(b);
}
#endif

int qwen_backend_available(qwen_backend_kind_t kind) {
    switch (kind) {
        case QWEN_BACKEND_CPU:  return 1;
        case QWEN_BACKEND_METAL:
#ifdef QWEN_HAVE_METAL
            return qwen_metal_available();
#else
            return 0;
#endif
        case QWEN_BACKEND_CUDA:
#ifdef QWEN_HAVE_CUDA
            return qwen_cuda_available();
#else
            return 0;
#endif
    }
    return 0;
}

qwen_backend_kind_t qwen_backend_kind_from_str(const char *s) {
    if (!s) return QWEN_BACKEND_CPU;
    if (strcasecmp(s, "metal") == 0) return QWEN_BACKEND_METAL;
    if (strcasecmp(s, "cuda")  == 0) return QWEN_BACKEND_CUDA;
    return QWEN_BACKEND_CPU;
}

qwen_backend_t *qwen_backend_init(qwen_backend_kind_t want) {
    if (want == QWEN_BACKEND_METAL) {
#ifdef QWEN_HAVE_METAL
        if (qwen_metal_available()) {
            void *impl = qwen_metal_init();
            if (impl) {
                qwen_backend_t *b = calloc(1, sizeof(*b));
                if (!b) { qwen_metal_free(impl); return make_cpu(); }
                b->kind = QWEN_BACKEND_METAL; b->name = "metal"; b->impl = impl;
                b->matvec_bf16 = metal_matvec_bf16;
                b->matmat_bf16 = metal_matmat_bf16;
                b->free = metal_free;
                return b;
            }
        }
        fprintf(stderr, "backend: Metal requested but unavailable — using CPU\n");
#else
        fprintf(stderr, "backend: Metal not compiled in (build with `make metal`) — using CPU\n");
#endif
        return make_cpu();
    }

    if (want == QWEN_BACKEND_CUDA) {
#ifdef QWEN_HAVE_CUDA
        if (qwen_cuda_available()) {
            void *impl = qwen_cuda_init();
            if (impl) {
                qwen_backend_t *b = calloc(1, sizeof(*b));
                if (!b) { qwen_cuda_free(impl); return make_cpu(); }
                b->kind = QWEN_BACKEND_CUDA; b->name = "cuda"; b->impl = impl;
                b->matvec_bf16 = cuda_matvec_bf16;
                b->matmat_bf16 = cuda_matmat_bf16;
                b->free = cuda_free;
                return b;
            }
        }
        fprintf(stderr, "backend: CUDA requested but unavailable — using CPU\n");
#else
        fprintf(stderr, "backend: CUDA not compiled in (build with `make cuda`) — using CPU\n");
#endif
        return make_cpu();
    }

    return make_cpu();
}

void qwen_backend_free(qwen_backend_t *b) {
    if (b && b->free) b->free(b);
}

/* ---- selftest ----------------------------------------------------------- */
#include <time.h>
#include <math.h>

static uint16_t f32_to_bf16_trunc(float f) {
    union { float f; uint32_t u; } v; v.f = f;
    return (uint16_t)(v.u >> 16);
}
static double now_ms(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}
/* xorshift32 — deterministic, no dependence on rand() impl */
static uint32_t rng_state = 0x1234567u;
static float rnd_unit(void) {
    rng_state ^= rng_state << 13; rng_state ^= rng_state >> 17; rng_state ^= rng_state << 5;
    return ((float)(rng_state & 0xFFFFFF) / (float)0xFFFFFF) * 2.0f - 1.0f; /* [-1,1] */
}

int qwen_gpu_selftest(qwen_backend_kind_t kind, void *out) {
    FILE *f = out ? (FILE *)out : stdout;
#ifdef QWEN_HAVE_METAL
    /* Metal has a full per-op suite (all tensors, resident weights). */
    if (kind == QWEN_BACKEND_METAL && qwen_metal_available())
        return qwen_metal_selftest(out);
#endif
    const int rows = 2048, cols = 2048, B = 8;
    int fails = 0;

    uint16_t *W = malloc((size_t)rows * cols * sizeof(uint16_t));
    float *x   = malloc((size_t)cols * sizeof(float));
    float *X   = malloc((size_t)cols * B * sizeof(float));
    float *y_cpu = malloc((size_t)rows * sizeof(float));
    float *y_gpu = malloc((size_t)rows * sizeof(float));
    float *Y_cpu = malloc((size_t)rows * B * sizeof(float));
    float *Y_gpu = malloc((size_t)rows * B * sizeof(float));
    if (!W || !x || !X || !y_cpu || !y_gpu || !Y_cpu || !Y_gpu) {
        fprintf(f, "gpu-selftest: OOM\n");
        free(W); free(x); free(X); free(y_cpu); free(y_gpu); free(Y_cpu); free(Y_gpu);
        return 1;
    }
    rng_state = 0x1234567u;
    for (size_t i = 0; i < (size_t)rows * cols; ++i) W[i] = f32_to_bf16_trunc(rnd_unit() * 0.1f);
    for (int i = 0; i < cols; ++i) x[i] = rnd_unit();
    for (int i = 0; i < cols * B; ++i) X[i] = rnd_unit();

    qwen_backend_t *cpu = make_cpu();
    qwen_backend_t *gpu = qwen_backend_init(kind);
    fprintf(f, "gpu-selftest: backend='%s' (requested %d)  shape rows=%d cols=%d B=%d\n",
            gpu->name, (int)kind, rows, cols, B);
    if (gpu->kind == QWEN_BACKEND_CPU && kind != QWEN_BACKEND_CPU) {
        fprintf(f, "  NOTE: GPU backend unavailable — nothing to validate against CPU.\n");
    }

    /* ---- matvec correctness ---- */
    cpu->matvec_bf16(cpu, y_cpu, W, x, rows, cols);
    gpu->matvec_bf16(gpu, y_gpu, W, x, rows, cols);
    double mv_abs = 0, mv_ref = 0;
    for (int i = 0; i < rows; ++i) {
        double d = fabs((double)y_gpu[i] - y_cpu[i]); if (d > mv_abs) mv_abs = d;
        double r = fabs((double)y_cpu[i]);            if (r > mv_ref) mv_ref = r;
    }
    double mv_rel = mv_ref > 0 ? mv_abs / mv_ref : 0;
    int mv_ok = mv_rel < 1e-2;
    fprintf(f, "  matvec_bf16: max|abs|=%.3e  rel=%.3e  %s\n", mv_abs, mv_rel, mv_ok ? "PASS" : "FAIL");
    if (!mv_ok) fails++;

    /* ---- matmat correctness ---- */
    cpu->matmat_bf16(cpu, Y_cpu, W, X, rows, cols, B);
    gpu->matmat_bf16(gpu, Y_gpu, W, X, rows, cols, B);
    double mm_abs = 0, mm_ref = 0;
    for (int i = 0; i < rows * B; ++i) {
        double d = fabs((double)Y_gpu[i] - Y_cpu[i]); if (d > mm_abs) mm_abs = d;
        double r = fabs((double)Y_cpu[i]);            if (r > mm_ref) mm_ref = r;
    }
    double mm_rel = mm_ref > 0 ? mm_abs / mm_ref : 0;
    int mm_ok = mm_rel < 1e-2;
    fprintf(f, "  matmat_bf16: max|abs|=%.3e  rel=%.3e  %s\n", mm_abs, mm_rel, mm_ok ? "PASS" : "FAIL");
    if (!mm_ok) fails++;

    /* ---- rough throughput (matmat is the compute-bound win on M1) ---- */
    const int iters = 20;
    double t0 = now_ms();
    for (int it = 0; it < iters; ++it) cpu->matmat_bf16(cpu, Y_cpu, W, X, rows, cols, B);
    double t_cpu = (now_ms() - t0) / iters;
    t0 = now_ms();
    for (int it = 0; it < iters; ++it) gpu->matmat_bf16(gpu, Y_gpu, W, X, rows, cols, B);
    double t_gpu = (now_ms() - t0) / iters;
    fprintf(f, "  matmat timing: cpu=%.3f ms  %s=%.3f ms  (%.2fx)\n",
            t_cpu, gpu->name, t_gpu, t_gpu > 0 ? t_cpu / t_gpu : 0);
    fprintf(f, "gpu-selftest: %s\n", fails ? "FAIL" : "PASS");

    qwen_backend_free(cpu);
    qwen_backend_free(gpu);
    free(W); free(x); free(X); free(y_cpu); free(y_gpu); free(Y_cpu); free(Y_gpu);
    return fails;
}
