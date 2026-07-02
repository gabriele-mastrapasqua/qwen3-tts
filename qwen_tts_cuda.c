/*
 * qwen_tts_cuda.c — NVIDIA CUDA backend (G3), cuBLAS-first.
 *
 * Built only by `make cuda` (-DQWEN_HAVE_CUDA -lcublas -lcudart). Without the
 * define, compiles to no-op stubs so `make blas` / M1 builds are unaffected.
 *
 * ARCHITECTURE (mirrors the Metal backend): weights are RESIDENT — each weight
 * pointer is converted bf16→f32 and uploaded to the device ONCE, cached by
 * pointer, reused every call. IO buffers (dX/dY) are reused and grown, never
 * malloc'd per call. This fixes the naive skeleton that converted+cudaMalloc'd
 * every call (the same per-call-alloc trap we killed on Metal).
 *
 * Row-major → cuBLAS column-major mapping: for Y[rows,B]=W[rows,cols]@X[cols,B]
 * (all row-major) we compute Y_cm[B,rows] = X_cm[B,cols] * W_cm[cols,rows], i.e.
 * cublasSgemm(N,N, m=B, n=rows, k=cols, dX lda=B, dW ldb=cols, dY ldc=B). With
 * B=1 this is the matvec.
 *
 * v1 = cuBLAS Sgemm on RESIDENT fp32 weights (no nvcc). The tensor-core path
 * (resident bf16 + cublasGemmEx CUDA_R_16BF on sm_80+, ~another 4-8× on the
 * matmul) needs a tiny f32→bf16 activation-convert kernel (nvcc) → that is G3b,
 * built + RTF-measured on the DGX. A fully-fused resident decode (activations
 * kept on device across a whole step, CUDA Graphs over the 16 CP passes/frame)
 * is the CUDA twin of the Metal resident-decode work.
 */

#include "qwen_tts_cuda.h"

#ifndef QWEN_HAVE_CUDA

int   qwen_cuda_available(void) { return 0; }
void *qwen_cuda_init(void) { return 0; }
void  qwen_cuda_free(void *ctx) { (void)ctx; }
void  qwen_cuda_matvec_bf16(void *ctx, float *y, const uint16_t *W,
                            const float *x, int rows, int cols) {
    (void)ctx; (void)y; (void)W; (void)x; (void)rows; (void)cols;
}
void  qwen_cuda_matmat_bf16(void *ctx, float *Y, const uint16_t *W,
                            const float *X, int rows, int cols, int B) {
    (void)ctx; (void)Y; (void)W; (void)X; (void)rows; (void)cols; (void)B;
}

#else /* QWEN_HAVE_CUDA */

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct { const void *key; float *dbuf; } wc_ent;   /* resident weight (device fp32) */

typedef struct {
    cublasHandle_t handle;
    wc_ent *wc; int wc_n, wc_cap;          /* weight cache by host pointer */
    float *dX, *dY; size_t dX_cap, dY_cap; /* reusable IO device buffers */
} qwen_cuda_ctx;

static inline float bf16_to_f32_host(uint16_t b) {
    union { uint32_t u; float f; } v; v.u = (uint32_t)b << 16; return v.f;
}

int qwen_cuda_available(void) {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess) return 0;
    return n > 0;
}

void *qwen_cuda_init(void) {
    qwen_cuda_ctx *c = calloc(1, sizeof(*c));
    if (!c) return NULL;
    if (cublasCreate(&c->handle) != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "CUDA: cublasCreate failed\n"); free(c); return NULL;
    }
    /* opportunistically enable tensor-core math for the fp32 API where legal */
    cublasSetMathMode(c->handle, CUBLAS_TF32_TENSOR_OP_MATH);
    return c;
}

void qwen_cuda_free(void *ctx) {
    if (!ctx) return;
    qwen_cuda_ctx *c = ctx;
    for (int i = 0; i < c->wc_n; ++i) cudaFree(c->wc[i].dbuf);
    free(c->wc);
    if (c->dX) cudaFree(c->dX);
    if (c->dY) cudaFree(c->dY);
    if (c->handle) cublasDestroy(c->handle);
    free(c);
}

/* Resident weight: convert bf16→f32 once (host), upload once, cache by pointer. */
static float *cuda_weight(qwen_cuda_ctx *c, const uint16_t *W, size_t n) {
    for (int i = 0; i < c->wc_n; ++i)
        if (c->wc[i].key == W) return c->wc[i].dbuf;
    float *h = (float *)malloc(n * sizeof(float));
    if (!h) return NULL;
    for (size_t i = 0; i < n; ++i) h[i] = bf16_to_f32_host(W[i]);
    float *d = NULL;
    if (cudaMalloc((void **)&d, n * sizeof(float)) != cudaSuccess) { free(h); return NULL; }
    cudaMemcpy(d, h, n * sizeof(float), cudaMemcpyHostToDevice);
    free(h);
    if (c->wc_n == c->wc_cap) {
        c->wc_cap = c->wc_cap ? c->wc_cap * 2 : 64;
        c->wc = realloc(c->wc, (size_t)c->wc_cap * sizeof(wc_ent));
    }
    c->wc[c->wc_n].key = W; c->wc[c->wc_n].dbuf = d; c->wc_n++;
    return d;
}

static float *cuda_io(float **buf, size_t *cap, size_t need) {
    if (*cap < need) { if (*buf) cudaFree(*buf); if (cudaMalloc((void **)buf, need) != cudaSuccess) { *buf = NULL; *cap = 0; return NULL; } *cap = need; }
    return *buf;
}

void qwen_cuda_matmat_bf16(void *ctx, float *Y, const uint16_t *W,
                           const float *X, int rows, int cols, int B) {
    qwen_cuda_ctx *c = ctx;
    float *dW = cuda_weight(c, W, (size_t)rows * cols);
    float *dX = cuda_io(&c->dX, &c->dX_cap, (size_t)cols * B * sizeof(float));
    float *dY = cuda_io(&c->dY, &c->dY_cap, (size_t)rows * B * sizeof(float));
    if (!dW || !dX || !dY) { fprintf(stderr, "CUDA: alloc failed\n"); return; }
    cudaMemcpy(dX, X, (size_t)cols * B * sizeof(float), cudaMemcpyHostToDevice);

    const float alpha = 1.0f, beta = 0.0f;
    /* Y_cm[B,rows] = X_cm[B,cols] * W_cm[cols,rows] */
    cublasSgemm(c->handle, CUBLAS_OP_N, CUBLAS_OP_N,
                /*m=*/B, /*n=*/rows, /*k=*/cols,
                &alpha, dX, /*lda=*/B, dW, /*ldb=*/cols,
                &beta,  dY, /*ldc=*/B);

    cudaMemcpy(Y, dY, (size_t)rows * B * sizeof(float), cudaMemcpyDeviceToHost);
}

void qwen_cuda_matvec_bf16(void *ctx, float *y, const uint16_t *W,
                           const float *x, int rows, int cols) {
    qwen_cuda_matmat_bf16(ctx, y, W, x, rows, cols, 1);
}

#endif /* QWEN_HAVE_CUDA */
