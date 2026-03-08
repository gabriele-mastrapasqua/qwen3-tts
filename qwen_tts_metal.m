/*
 * qwen_tts_metal.m - Metal GPU backend for Qwen3-TTS
 *
 * Objective-C implementation using Metal framework.
 * Compiles only on macOS with ENABLE_METAL defined.
 */

#ifdef ENABLE_METAL

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include "qwen_tts_metal.h"

/* Global Metal context (declared extern in qwen_tts_metal.h) */
qwen_metal_ctx_t *g_metal_ctx = NULL;
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ========================================================================
 * Metal context
 * ======================================================================== */

#define MAX_WEIGHT_BUFFERS 512  /* enough for 28 talker + 5 CP layers × ~6 weights each */

struct qwen_metal_ctx {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLComputePipelineState> matvec_pipeline;

    /* Weight buffers (uploaded once at load time) */
    id<MTLBuffer> weight_bufs[MAX_WEIGHT_BUFFERS];
    int weight_rows[MAX_WEIGHT_BUFFERS];
    int weight_cols[MAX_WEIGHT_BUFFERS];
    int n_weights;

    /* Workspace buffers (reused across calls) */
    id<MTLBuffer> x_buf;       /* input vector */
    id<MTLBuffer> y_buf;       /* output vector */
    int x_buf_size;            /* current capacity in bytes */
    int y_buf_size;
};

/* ========================================================================
 * Init / Free
 * ======================================================================== */

qwen_metal_ctx_t *qwen_metal_init(void) {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            fprintf(stderr, "Metal: no GPU device found\n");
            return NULL;
        }

        /* Check for Apple Silicon (discrete/integrated GPU with unified memory) */
        if (![device supportsFamily:MTLGPUFamilyApple7]) {
            fprintf(stderr, "Metal: device '%s' does not support Apple7 family (need Apple Silicon)\n",
                    [[device name] UTF8String]);
            return NULL;
        }

        qwen_metal_ctx_t *ctx = (qwen_metal_ctx_t *)calloc(1, sizeof(qwen_metal_ctx_t));
        if (!ctx) return NULL;

        ctx->device = device;
        ctx->queue = [device newCommandQueue];
        if (!ctx->queue) {
            fprintf(stderr, "Metal: failed to create command queue\n");
            free(ctx);
            return NULL;
        }

        /* Compile shader from embedded source */
        NSError *error = nil;

        /* Load shader from .metallib file compiled at build time */
        NSString *libPath = [[NSBundle mainBundle] pathForResource:@"qwen_tts_metal" ofType:@"metallib"];
        id<MTLLibrary> lib = nil;

        if (libPath) {
            NSURL *libURL = [NSURL fileURLWithPath:libPath];
            lib = [device newLibraryWithURL:libURL error:&error];
        }

        /* Fallback: compile from source at runtime */
        if (!lib) {
            /* Find the .metal source file next to the binary */
            NSString *srcPath = nil;
            NSString *execPath = [[NSProcessInfo processInfo] arguments][0];
            NSString *execDir = [execPath stringByDeletingLastPathComponent];
            srcPath = [execDir stringByAppendingPathComponent:@"qwen_tts_metal.metal"];

            if (![[NSFileManager defaultManager] fileExistsAtPath:srcPath]) {
                /* Try current directory */
                srcPath = @"qwen_tts_metal.metal";
            }

            NSString *source = [NSString stringWithContentsOfFile:srcPath
                                                         encoding:NSUTF8StringEncoding
                                                            error:&error];
            if (!source) {
                fprintf(stderr, "Metal: failed to load shader source: %s\n",
                        [[error localizedDescription] UTF8String]);
                free(ctx);
                return NULL;
            }

            MTLCompileOptions *opts = [[MTLCompileOptions alloc] init];
            opts.mathMode = MTLMathModeFast;
            lib = [device newLibraryWithSource:source options:opts error:&error];
            if (!lib) {
                fprintf(stderr, "Metal: failed to compile shaders: %s\n",
                        [[error localizedDescription] UTF8String]);
                free(ctx);
                return NULL;
            }
        }

        id<MTLFunction> matvec_fn = [lib newFunctionWithName:@"matvec_bf16"];
        if (!matvec_fn) {
            fprintf(stderr, "Metal: matvec_bf16 kernel not found\n");
            free(ctx);
            return NULL;
        }

        ctx->matvec_pipeline = [device newComputePipelineStateWithFunction:matvec_fn error:&error];
        if (!ctx->matvec_pipeline) {
            fprintf(stderr, "Metal: failed to create pipeline: %s\n",
                    [[error localizedDescription] UTF8String]);
            free(ctx);
            return NULL;
        }

        fprintf(stderr, "Metal: initialized on %s (max threadgroup: %lu)\n",
                [[device name] UTF8String],
                (unsigned long)[ctx->matvec_pipeline maxTotalThreadsPerThreadgroup]);

        return ctx;
    }
}

void qwen_metal_free(qwen_metal_ctx_t *ctx) {
    if (!ctx) return;
    /* ARC handles Metal object cleanup */
    free(ctx);
}

int qwen_metal_is_active(qwen_metal_ctx_t *ctx) {
    return ctx && ctx->device != nil;
}

/* ========================================================================
 * Weight upload
 * ======================================================================== */

int qwen_metal_upload_weight(qwen_metal_ctx_t *ctx,
                             const uint16_t *data, int rows, int cols) {
    if (!ctx || ctx->n_weights >= MAX_WEIGHT_BUFFERS) return -1;

    @autoreleasepool {
        size_t size = (size_t)rows * cols * sizeof(uint16_t);
        id<MTLBuffer> buf = [ctx->device newBufferWithBytes:data
                                                     length:size
                                                    options:MTLResourceStorageModeShared];
        if (!buf) {
            fprintf(stderr, "Metal: failed to upload weight (%d x %d, %.1f MB)\n",
                    rows, cols, size / (1024.0 * 1024.0));
            return -1;
        }

        int handle = ctx->n_weights;
        ctx->weight_bufs[handle] = buf;
        ctx->weight_rows[handle] = rows;
        ctx->weight_cols[handle] = cols;
        ctx->n_weights++;
        return handle;
    }
}

/* ========================================================================
 * Ensure workspace buffers are large enough
 * ======================================================================== */

static void ensure_workspace(qwen_metal_ctx_t *ctx, int x_bytes, int y_bytes) {
    @autoreleasepool {
        if (x_bytes > ctx->x_buf_size) {
            ctx->x_buf = [ctx->device newBufferWithLength:x_bytes
                                                  options:MTLResourceStorageModeShared];
            ctx->x_buf_size = x_bytes;
        }
        if (y_bytes > ctx->y_buf_size) {
            ctx->y_buf = [ctx->device newBufferWithLength:y_bytes
                                                  options:MTLResourceStorageModeShared];
            ctx->y_buf_size = y_bytes;
        }
    }
}

/* ========================================================================
 * Matvec dispatch
 * ======================================================================== */

/* Params struct must match Metal shader */
typedef struct {
    int rows;
    int cols;
} matvec_params_t;

#define THREADS_PER_GROUP 256

static void dispatch_matvec(qwen_metal_ctx_t *ctx,
                            id<MTLComputeCommandEncoder> encoder,
                            id<MTLBuffer> W_buf,
                            id<MTLBuffer> x_buf,
                            id<MTLBuffer> y_buf, int y_offset,
                            int rows, int cols) {
    matvec_params_t params = { rows, cols };

    [encoder setComputePipelineState:ctx->matvec_pipeline];
    [encoder setBuffer:W_buf offset:0 atIndex:0];
    [encoder setBuffer:x_buf offset:0 atIndex:1];
    [encoder setBuffer:y_buf offset:y_offset * (int)sizeof(float) atIndex:2];
    [encoder setBytes:&params length:sizeof(params) atIndex:3];

    /* One thread per output row, grouped in blocks of 256 */
    MTLSize grid = MTLSizeMake(rows, 1, 1);
    MTLSize group = MTLSizeMake(THREADS_PER_GROUP < rows ? THREADS_PER_GROUP : rows, 1, 1);

    [encoder dispatchThreads:grid threadsPerThreadgroup:group];
}

void qwen_metal_matvec_bf16(qwen_metal_ctx_t *ctx, int weight_handle,
                            float *y, const float *x, int rows, int cols) {
    if (!ctx || weight_handle < 0 || weight_handle >= ctx->n_weights) return;

    @autoreleasepool {
        ensure_workspace(ctx, cols * sizeof(float), rows * sizeof(float));

        /* Copy input to GPU */
        memcpy([ctx->x_buf contents], x, cols * sizeof(float));

        /* Encode and submit */
        id<MTLCommandBuffer> cmd = [ctx->queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];

        dispatch_matvec(ctx, encoder, ctx->weight_bufs[weight_handle],
                        ctx->x_buf, ctx->y_buf, 0, rows, cols);

        [encoder endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        /* Copy output back */
        memcpy(y, [ctx->y_buf contents], rows * sizeof(float));
    }
}

void qwen_metal_matvec_bf16_qkv(qwen_metal_ctx_t *ctx,
                                 int wq_handle, int wk_handle, int wv_handle,
                                 float *q, float *k, float *v,
                                 const float *x, int in_dim,
                                 int q_dim, int kv_dim) {
    if (!ctx) return;

    @autoreleasepool {
        int total_out = q_dim + 2 * kv_dim;
        ensure_workspace(ctx, in_dim * sizeof(float), total_out * sizeof(float));

        /* Copy input once */
        memcpy([ctx->x_buf contents], x, in_dim * sizeof(float));

        /* Encode all 3 matvecs in a single command buffer */
        id<MTLCommandBuffer> cmd = [ctx->queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];

        dispatch_matvec(ctx, encoder, ctx->weight_bufs[wq_handle],
                        ctx->x_buf, ctx->y_buf, 0, q_dim, in_dim);
        dispatch_matvec(ctx, encoder, ctx->weight_bufs[wk_handle],
                        ctx->x_buf, ctx->y_buf, q_dim, kv_dim, in_dim);
        dispatch_matvec(ctx, encoder, ctx->weight_bufs[wv_handle],
                        ctx->x_buf, ctx->y_buf, q_dim + kv_dim, kv_dim, in_dim);

        [encoder endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        /* Copy outputs */
        const float *out = (const float *)[ctx->y_buf contents];
        memcpy(q, out, q_dim * sizeof(float));
        memcpy(k, out + q_dim, kv_dim * sizeof(float));
        memcpy(v, out + q_dim + kv_dim, kv_dim * sizeof(float));
    }
}

#endif /* ENABLE_METAL */
