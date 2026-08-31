/* qwen_tts_backend.h — GPU backend seam (G1) */
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
    const char *name;
    void *impl;

    void (*matvec_bf16)(struct qwen_backend *b, float *y,
                        const uint16_t *W, const float *x, int rows, int cols);

    void (*matmat_bf16)(struct qwen_backend *b, float *Y,
                        const uint16_t *W, const float *X,
                        int rows, int cols, int B);

    void (*free)(struct qwen_backend *b);
} qwen_backend_t;

qwen_backend_t *qwen_backend_init(qwen_backend_kind_t want);

qwen_backend_kind_t qwen_backend_kind_from_str(const char *s);

void qwen_backend_free(qwen_backend_t *b);

int qwen_backend_available(qwen_backend_kind_t kind);

void qwen_backend_install_global(qwen_backend_t *b);

int qwen_gpu_selftest(qwen_backend_kind_t kind, void *out);

#ifdef __cplusplus
}
#endif

#endif
