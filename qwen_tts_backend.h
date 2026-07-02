/*
 * qwen_tts_backend.h — GPU backend seam (G1)
 *
 * A tiny vtable one level ABOVE the kernels. The default is the CPU path
 * (qwen_matvec_* / qwen_matmat_* in qwen_tts_kernels.c) — this header only
 * matters when an experimental GPU backend (Metal / CUDA) is compiled in via
 * `make metal` / `make cuda`. In a plain `make blas` build no GPU TU is linked
 * and qwen_backend_init(GPU) cleanly reports "unavailable" and falls back.
 *
 * v1 exposes only the two primitives the offload path needs first:
 *   - matvec_bf16  (decode hot path; correctness vehicle for the selftest)
 *   - matmat_bf16  (batched, COMPUTE-bound → the honest win on M1)
 * More ops (int8/q4 matvec, rms_norm, rope, im2col/conv) get added as the
 * backends grow — see plan_v4 §E4.ter.
 *
 * Hard rule: this never changes CPU behavior. It is additive and opt-in.
 */

#ifndef QWEN_TTS_BACKEND_H
#define QWEN_TTS_BACKEND_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    QWEN_BACKEND_CPU   = 0,
    QWEN_BACKEND_METAL = 1,
    QWEN_BACKEND_CUDA  = 2,
} qwen_backend_kind_t;

typedef struct qwen_backend {
    qwen_backend_kind_t kind;
    const char *name;              /* human label, e.g. "metal" / "cpu" */
    void *impl;                    /* backend-private ctx (device/queue/psos) */

    /* y[rows] = W[rows,cols] @ x[cols]   (W bf16 uint16, x/y f32) */
    void (*matvec_bf16)(struct qwen_backend *b, float *y,
                        const uint16_t *W, const float *x, int rows, int cols);

    /* Y[rows,B] = W[rows,cols] @ X[cols,B]  (row-major f32 X/Y; B<=64) */
    void (*matmat_bf16)(struct qwen_backend *b, float *Y,
                        const uint16_t *W, const float *X,
                        int rows, int cols, int B);

    void (*free)(struct qwen_backend *b);
} qwen_backend_t;

/* Resolve a backend. Requesting METAL/CUDA that was not compiled in (or whose
 * device is absent) prints a one-line note and returns the CPU backend, so
 * callers never have to special-case availability. Never returns NULL. */
qwen_backend_t *qwen_backend_init(qwen_backend_kind_t want);

/* Parse "cpu"/"metal"/"cuda" (case-insensitive) → kind; defaults to CPU. */
qwen_backend_kind_t qwen_backend_kind_from_str(const char *s);

void qwen_backend_free(qwen_backend_t *b);

/* 1 if a real GPU backend of this kind is compiled in AND its device exists. */
int qwen_backend_available(qwen_backend_kind_t kind);

/* Install `b` as the global offload target: routes qwen_matvec_bf16 (+ the bf16
 * QKV fused path) through b->matvec_bf16. Pass NULL to restore the CPU default.
 * This is the opt-in wiring for end-to-end synthesis on the GPU. */
void qwen_backend_install_global(qwen_backend_t *b);

/* Correctness + rough-throughput selftest: runs matvec_bf16 and matmat_bf16 on
 * the requested backend vs the CPU reference on deterministic random data,
 * prints max abs/rel diff and CPU-vs-GPU timing to `out` (NULL → stdout).
 * Returns 0 on PASS (within fp-order tolerance), >0 = number of failed checks. */
int qwen_gpu_selftest(qwen_backend_kind_t kind, void *out);

#ifdef __cplusplus
}
#endif

#endif /* QWEN_TTS_BACKEND_H */
