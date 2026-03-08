/*
 * qwen_tts_metal.m - Metal GPU backend for Qwen3-TTS
 *
 * Full GPU transformer inference: all layers in one command buffer.
 * Compiles only on macOS with ENABLE_METAL defined.
 */

#ifdef ENABLE_METAL

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include "qwen_tts_metal.h"

qwen_metal_ctx_t *g_metal_ctx = NULL;
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ========================================================================
 * Metal context
 * ======================================================================== */

#define MAX_WEIGHT_BUFFERS 1024
#define MAX_KV_CACHES 2

struct qwen_metal_ctx {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;

    /* Pipeline states */
    id<MTLComputePipelineState> matvec_pipeline;
    id<MTLComputePipelineState> rmsnorm_pipeline;
    id<MTLComputePipelineState> rmsnorm_perhead_pipeline;
    id<MTLComputePipelineState> rope_neox_pipeline;
    id<MTLComputePipelineState> kv_store_pipeline;
    id<MTLComputePipelineState> attention_pipeline;
    id<MTLComputePipelineState> swiglu_pipeline;
    id<MTLComputePipelineState> residual_add_pipeline;

    /* Weight/data buffers (uploaded at load time) */
    id<MTLBuffer> weight_bufs[MAX_WEIGHT_BUFFERS];
    int weight_rows[MAX_WEIGHT_BUFFERS];
    int weight_cols[MAX_WEIGHT_BUFFERS];
    int n_weights;

    /* Workspace buffers for per-dispatch matvec (legacy) */
    id<MTLBuffer> x_buf;
    id<MTLBuffer> y_buf;
    int x_buf_size;
    int y_buf_size;

    /* Batched dispatch state (legacy) */
    id<MTLCommandBuffer> batch_cmd;
    id<MTLComputeCommandEncoder> batch_encoder;

    /* ── Full GPU step resources ─────────────────────────────────────── */

    /* Working buffers (shared memory, f32) */
    id<MTLBuffer> work_x;        /* [max_hidden] */
    id<MTLBuffer> work_x_norm;   /* [max_hidden] */
    id<MTLBuffer> work_q;        /* [max_q_dim] */
    id<MTLBuffer> work_k;        /* [max_kv_dim] */
    id<MTLBuffer> work_v;        /* [max_kv_dim] */
    id<MTLBuffer> work_attn;     /* [max_q_dim] */
    id<MTLBuffer> work_proj;     /* [max_hidden] */
    id<MTLBuffer> work_gate;     /* [2 * max_inter] */
    id<MTLBuffer> work_ffn;      /* [max_inter] */
    int has_work_buffers;

    /* GPU KV caches (bf16) */
    struct {
        id<MTLBuffer> k;
        id<MTLBuffer> v;
        int n_layers;
        int kv_max;
        int kv_dim;
    } kv_caches[MAX_KV_CACHES];
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

        if (![device supportsFamily:MTLGPUFamilyApple7]) {
            fprintf(stderr, "Metal: device '%s' does not support Apple7 family\n",
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

        /* Compile shaders */
        NSError *error = nil;

        NSString *libPath = [[NSBundle mainBundle] pathForResource:@"qwen_tts_metal" ofType:@"metallib"];
        id<MTLLibrary> lib = nil;

        if (libPath) {
            NSURL *libURL = [NSURL fileURLWithPath:libPath];
            lib = [device newLibraryWithURL:libURL error:&error];
        }

        if (!lib) {
            NSString *srcPath = nil;
            NSString *execPath = [[NSProcessInfo processInfo] arguments][0];
            NSString *execDir = [execPath stringByDeletingLastPathComponent];
            srcPath = [execDir stringByAppendingPathComponent:@"qwen_tts_metal.metal"];

            if (![[NSFileManager defaultManager] fileExistsAtPath:srcPath]) {
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

        /* Create pipeline states for all kernels */
        typedef struct { const char *name; int idx; } kernel_entry_t;
        kernel_entry_t kernels[] = {
            {"matvec_bf16", 0}, {"rmsnorm", 1}, {"rmsnorm_perhead", 2},
            {"rope_neox", 3}, {"kv_store_bf16", 4}, {"attention_bf16kv", 5},
            {"swiglu", 6}, {"residual_add", 7}
        };
        int n_kernels = sizeof(kernels) / sizeof(kernels[0]);

        for (int i = 0; i < n_kernels; i++) {
            NSString *name = [NSString stringWithUTF8String:kernels[i].name];
            id<MTLFunction> fn = [lib newFunctionWithName:name];
            if (!fn) {
                fprintf(stderr, "Metal: kernel '%s' not found\n", kernels[i].name);
                free(ctx);
                return NULL;
            }
            id<MTLComputePipelineState> pso = [device newComputePipelineStateWithFunction:fn error:&error];
            if (!pso) {
                fprintf(stderr, "Metal: failed to create pipeline for '%s': %s\n",
                        kernels[i].name, [[error localizedDescription] UTF8String]);
                free(ctx);
                return NULL;
            }
            switch (kernels[i].idx) {
                case 0: ctx->matvec_pipeline = pso; break;
                case 1: ctx->rmsnorm_pipeline = pso; break;
                case 2: ctx->rmsnorm_perhead_pipeline = pso; break;
                case 3: ctx->rope_neox_pipeline = pso; break;
                case 4: ctx->kv_store_pipeline = pso; break;
                case 5: ctx->attention_pipeline = pso; break;
                case 6: ctx->swiglu_pipeline = pso; break;
                case 7: ctx->residual_add_pipeline = pso; break;
            }
        }

        fprintf(stderr, "Metal: initialized on %s (%d kernels)\n",
                [[device name] UTF8String], n_kernels);

        return ctx;
    }
}

void qwen_metal_free(qwen_metal_ctx_t *ctx) {
    if (!ctx) return;
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
        if (!buf) return -1;

        int handle = ctx->n_weights;
        ctx->weight_bufs[handle] = buf;
        ctx->weight_rows[handle] = rows;
        ctx->weight_cols[handle] = cols;
        ctx->n_weights++;
        return handle;
    }
}

int qwen_metal_upload_f32(qwen_metal_ctx_t *ctx, const float *data, int count) {
    if (!ctx || ctx->n_weights >= MAX_WEIGHT_BUFFERS) return -1;

    @autoreleasepool {
        size_t size = (size_t)count * sizeof(float);
        id<MTLBuffer> buf = [ctx->device newBufferWithBytes:data
                                                     length:size
                                                    options:MTLResourceStorageModeShared];
        if (!buf) return -1;

        int handle = ctx->n_weights;
        ctx->weight_bufs[handle] = buf;
        ctx->weight_rows[handle] = count;
        ctx->weight_cols[handle] = 1;
        ctx->n_weights++;
        return handle;
    }
}

/* ========================================================================
 * Workspace helpers (legacy per-dispatch)
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
 * Per-dispatch matvec (legacy)
 * ======================================================================== */

typedef struct { int rows; int cols; } matvec_params_t;
#define SIMD_SIZE 32

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

    MTLSize grid = MTLSizeMake(rows * SIMD_SIZE, 1, 1);
    MTLSize group = MTLSizeMake(SIMD_SIZE, 1, 1);
    [encoder dispatchThreads:grid threadsPerThreadgroup:group];
}

void qwen_metal_matvec_bf16(qwen_metal_ctx_t *ctx, int weight_handle,
                            float *y, const float *x, int rows, int cols) {
    if (!ctx || weight_handle < 0 || weight_handle >= ctx->n_weights) return;

    ensure_workspace(ctx, cols * (int)sizeof(float), rows * (int)sizeof(float));
    memcpy([ctx->x_buf contents], x, cols * sizeof(float));

    id<MTLCommandBuffer> cmd = [ctx->queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];

    dispatch_matvec(ctx, encoder, ctx->weight_bufs[weight_handle],
                    ctx->x_buf, ctx->y_buf, 0, rows, cols);

    [encoder endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];

    memcpy(y, [ctx->y_buf contents], rows * sizeof(float));
}

void qwen_metal_matvec_bf16_qkv(qwen_metal_ctx_t *ctx,
                                 int wq_handle, int wk_handle, int wv_handle,
                                 float *q, float *k, float *v,
                                 const float *x, int in_dim,
                                 int q_dim, int kv_dim) {
    if (!ctx) return;

    int total_out = q_dim + 2 * kv_dim;
    ensure_workspace(ctx, in_dim * (int)sizeof(float), total_out * (int)sizeof(float));
    memcpy([ctx->x_buf contents], x, in_dim * sizeof(float));

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

    const float *out = (const float *)[ctx->y_buf contents];
    memcpy(q, out, q_dim * sizeof(float));
    memcpy(k, out + q_dim, kv_dim * sizeof(float));
    memcpy(v, out + q_dim + kv_dim, kv_dim * sizeof(float));
}

/* ========================================================================
 * Batched dispatch API (legacy)
 * ======================================================================== */

void qwen_metal_begin(qwen_metal_ctx_t *ctx) {
    if (!ctx) return;
    ctx->batch_cmd = [ctx->queue commandBuffer];
    ctx->batch_encoder = [ctx->batch_cmd computeCommandEncoder];
}

void qwen_metal_encode_matvec(qwen_metal_ctx_t *ctx, int weight_handle,
                               int y_offset, int x_offset,
                               int rows, int cols) {
    if (!ctx || !ctx->batch_encoder) return;
    if (weight_handle < 0 || weight_handle >= ctx->n_weights) return;

    matvec_params_t params = { rows, cols };

    [ctx->batch_encoder setComputePipelineState:ctx->matvec_pipeline];
    [ctx->batch_encoder setBuffer:ctx->weight_bufs[weight_handle] offset:0 atIndex:0];
    [ctx->batch_encoder setBuffer:ctx->x_buf offset:x_offset * sizeof(float) atIndex:1];
    [ctx->batch_encoder setBuffer:ctx->y_buf offset:y_offset * sizeof(float) atIndex:2];
    [ctx->batch_encoder setBytes:&params length:sizeof(params) atIndex:3];

    MTLSize grid = MTLSizeMake(rows * SIMD_SIZE, 1, 1);
    MTLSize group = MTLSizeMake(SIMD_SIZE, 1, 1);
    [ctx->batch_encoder dispatchThreads:grid threadsPerThreadgroup:group];
}

void qwen_metal_sync(qwen_metal_ctx_t *ctx) {
    if (!ctx || !ctx->batch_encoder) return;
    [ctx->batch_encoder endEncoding];
    [ctx->batch_cmd commit];
    [ctx->batch_cmd waitUntilCompleted];
    ctx->batch_encoder = nil;
    ctx->batch_cmd = nil;
}

float *qwen_metal_get_x(qwen_metal_ctx_t *ctx) {
    return ctx ? (float *)[ctx->x_buf contents] : NULL;
}

float *qwen_metal_get_y(qwen_metal_ctx_t *ctx) {
    return ctx ? (float *)[ctx->y_buf contents] : NULL;
}

void qwen_metal_ensure_workspace(qwen_metal_ctx_t *ctx, int x_bytes, int y_bytes) {
    if (ctx) ensure_workspace(ctx, x_bytes, y_bytes);
}

/* ========================================================================
 * Full GPU transformer step: resource allocation
 * ======================================================================== */

int qwen_metal_alloc_kv_cache(qwen_metal_ctx_t *ctx, int kv_id,
                               int n_layers, int kv_max, int kv_dim) {
    if (!ctx || kv_id < 0 || kv_id >= MAX_KV_CACHES) return -1;

    @autoreleasepool {
        size_t size = (size_t)n_layers * kv_max * kv_dim * sizeof(uint16_t);
        ctx->kv_caches[kv_id].k = [ctx->device newBufferWithLength:size
                                                           options:MTLResourceStorageModeShared];
        ctx->kv_caches[kv_id].v = [ctx->device newBufferWithLength:size
                                                           options:MTLResourceStorageModeShared];
        if (!ctx->kv_caches[kv_id].k || !ctx->kv_caches[kv_id].v) {
            fprintf(stderr, "Metal: failed to allocate KV cache (%d layers x %d max x %d dim, %.1f MB)\n",
                    n_layers, kv_max, kv_dim, 2.0 * size / (1024.0 * 1024.0));
            return -1;
        }

        ctx->kv_caches[kv_id].n_layers = n_layers;
        ctx->kv_caches[kv_id].kv_max = kv_max;
        ctx->kv_caches[kv_id].kv_dim = kv_dim;

        fprintf(stderr, "Metal: KV cache %d allocated (%d layers x %d slots, %.1f MB)\n",
                kv_id, n_layers, kv_max, 2.0 * size / (1024.0 * 1024.0));
        return 0;
    }
}

void qwen_metal_sync_kv_cache(qwen_metal_ctx_t *ctx, int kv_id,
                               const uint16_t *k, const uint16_t *v,
                               int n_layers, int kv_max, int kv_dim, int kv_len) {
    if (!ctx || kv_id < 0 || kv_id >= MAX_KV_CACHES) return;
    if (!ctx->kv_caches[kv_id].k) return;

    uint16_t *gpu_k = (uint16_t *)[ctx->kv_caches[kv_id].k contents];
    uint16_t *gpu_v = (uint16_t *)[ctx->kv_caches[kv_id].v contents];
    int gpu_kv_max = ctx->kv_caches[kv_id].kv_max;

    for (int layer = 0; layer < n_layers; layer++) {
        size_t src_offset = (size_t)layer * kv_max * kv_dim;
        size_t dst_offset = (size_t)layer * gpu_kv_max * kv_dim;
        size_t copy_size = (size_t)kv_len * kv_dim * sizeof(uint16_t);
        memcpy(gpu_k + dst_offset, k + src_offset, copy_size);
        memcpy(gpu_v + dst_offset, v + src_offset, copy_size);
    }
}

int qwen_metal_alloc_work_buffers(qwen_metal_ctx_t *ctx,
                                   int max_hidden, int max_q_dim,
                                   int max_kv_dim, int max_inter) {
    if (!ctx) return -1;

    @autoreleasepool {
        ctx->work_x      = [ctx->device newBufferWithLength:max_hidden * sizeof(float) options:MTLResourceStorageModeShared];
        ctx->work_x_norm = [ctx->device newBufferWithLength:max_hidden * sizeof(float) options:MTLResourceStorageModeShared];
        ctx->work_q      = [ctx->device newBufferWithLength:max_q_dim * sizeof(float) options:MTLResourceStorageModeShared];
        ctx->work_k      = [ctx->device newBufferWithLength:max_kv_dim * sizeof(float) options:MTLResourceStorageModeShared];
        ctx->work_v      = [ctx->device newBufferWithLength:max_kv_dim * sizeof(float) options:MTLResourceStorageModeShared];
        ctx->work_attn   = [ctx->device newBufferWithLength:max_q_dim * sizeof(float) options:MTLResourceStorageModeShared];
        ctx->work_proj   = [ctx->device newBufferWithLength:max_hidden * sizeof(float) options:MTLResourceStorageModeShared];
        ctx->work_gate   = [ctx->device newBufferWithLength:2 * max_inter * sizeof(float) options:MTLResourceStorageModeShared];
        ctx->work_ffn    = [ctx->device newBufferWithLength:max_inter * sizeof(float) options:MTLResourceStorageModeShared];

        if (!ctx->work_x || !ctx->work_x_norm || !ctx->work_q || !ctx->work_k ||
            !ctx->work_v || !ctx->work_attn || !ctx->work_proj || !ctx->work_gate || !ctx->work_ffn) {
            fprintf(stderr, "Metal: failed to allocate working buffers\n");
            return -1;
        }

        ctx->has_work_buffers = 1;
        fprintf(stderr, "Metal: working buffers allocated (hidden=%d q=%d kv=%d inter=%d)\n",
                max_hidden, max_q_dim, max_kv_dim, max_inter);
        return 0;
    }
}

int qwen_metal_has_full_step(qwen_metal_ctx_t *ctx) {
    return ctx && ctx->has_work_buffers && ctx->rmsnorm_pipeline;
}

/* ========================================================================
 * Full GPU transformer step
 *
 * ALL operations for ALL layers in ONE command buffer.
 * Eliminates per-dispatch overhead (~50μs × N dispatches → 1 × ~50μs).
 * ======================================================================== */

typedef struct { int dim; float eps; } rmsnorm_params_t;
typedef struct { int n_heads; int head_dim; float eps; } rmsnorm_perhead_params_t;
typedef struct { int n_heads; int head_dim; int pos; } rope_params_t;
typedef struct { int kv_dim; int offset; } kv_store_params_t;
typedef struct { int num_heads; int num_kv_heads; int head_dim; int kv_dim; int seq_len; float scale; } attention_params_t;

int qwen_metal_transformer_step(qwen_metal_ctx_t *ctx,
                                 const qwen_metal_step_config_t *cfg,
                                 const qwen_metal_layer_config_t *layers,
                                 float *x_inout,
                                 float *normed_out,
                                 int pos,
                                 int kv_id) {
    if (!ctx || !cfg || !layers) return -1;
    if (!ctx->has_work_buffers) return -1;
    if (kv_id < 0 || kv_id >= MAX_KV_CACHES || !ctx->kv_caches[kv_id].k) return -1;
    if (pos >= ctx->kv_caches[kv_id].kv_max) return -1;

    int h = cfg->hidden_size;
    int q_dim = cfg->num_heads * cfg->head_dim;
    int kv_dim = cfg->num_kv_heads * cfg->head_dim;
    int inter = cfg->intermediate_size;
    int hd = cfg->head_dim;
    int half_dim = hd / 2;
    float eps = cfg->rms_norm_eps;
    float scale = 1.0f / sqrtf((float)hd);
    int gpu_kv_max = ctx->kv_caches[kv_id].kv_max;

    /* Copy input to GPU */
    memcpy([ctx->work_x contents], x_inout, h * sizeof(float));

    @autoreleasepool {
        id<MTLCommandBuffer> cmd = [ctx->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

        for (int layer = 0; layer < cfg->n_layers; layer++) {
            const qwen_metal_layer_config_t *l = &layers[layer];
            size_t kv_layer_offset = (size_t)layer * gpu_kv_max * kv_dim * sizeof(uint16_t);

            /* ── 1. Input RMSNorm: work_x → work_x_norm ── */
            {
                rmsnorm_params_t p = { h, eps };
                [enc setComputePipelineState:ctx->rmsnorm_pipeline];
                [enc setBuffer:ctx->work_x offset:0 atIndex:0];
                [enc setBuffer:ctx->work_x_norm offset:0 atIndex:1];
                [enc setBuffer:ctx->weight_bufs[l->gpu_input_norm] offset:0 atIndex:2];
                [enc setBytes:&p length:sizeof(p) atIndex:3];
                [enc dispatchThreads:MTLSizeMake(SIMD_SIZE, 1, 1)
                     threadsPerThreadgroup:MTLSizeMake(SIMD_SIZE, 1, 1)];
            }

            /* ── 2. QKV matvec: work_x_norm → work_q, work_k, work_v ── */
            {
                matvec_params_t p;

                p = (matvec_params_t){ q_dim, h };
                [enc setComputePipelineState:ctx->matvec_pipeline];
                [enc setBuffer:ctx->weight_bufs[l->gpu_wq] offset:0 atIndex:0];
                [enc setBuffer:ctx->work_x_norm offset:0 atIndex:1];
                [enc setBuffer:ctx->work_q offset:0 atIndex:2];
                [enc setBytes:&p length:sizeof(p) atIndex:3];
                [enc dispatchThreads:MTLSizeMake(q_dim * SIMD_SIZE, 1, 1)
                     threadsPerThreadgroup:MTLSizeMake(SIMD_SIZE, 1, 1)];

                p = (matvec_params_t){ kv_dim, h };
                [enc setComputePipelineState:ctx->matvec_pipeline];
                [enc setBuffer:ctx->weight_bufs[l->gpu_wk] offset:0 atIndex:0];
                [enc setBuffer:ctx->work_x_norm offset:0 atIndex:1];
                [enc setBuffer:ctx->work_k offset:0 atIndex:2];
                [enc setBytes:&p length:sizeof(p) atIndex:3];
                [enc dispatchThreads:MTLSizeMake(kv_dim * SIMD_SIZE, 1, 1)
                     threadsPerThreadgroup:MTLSizeMake(SIMD_SIZE, 1, 1)];

                [enc setComputePipelineState:ctx->matvec_pipeline];
                [enc setBuffer:ctx->weight_bufs[l->gpu_wv] offset:0 atIndex:0];
                [enc setBuffer:ctx->work_x_norm offset:0 atIndex:1];
                [enc setBuffer:ctx->work_v offset:0 atIndex:2];
                [enc setBytes:&p length:sizeof(p) atIndex:3];
                [enc dispatchThreads:MTLSizeMake(kv_dim * SIMD_SIZE, 1, 1)
                     threadsPerThreadgroup:MTLSizeMake(SIMD_SIZE, 1, 1)];
            }

            /* ── 3. Per-head Q/K RMSNorm ── */
            {
                rmsnorm_perhead_params_t p;

                p = (rmsnorm_perhead_params_t){ cfg->num_heads, hd, eps };
                [enc setComputePipelineState:ctx->rmsnorm_perhead_pipeline];
                [enc setBuffer:ctx->work_q offset:0 atIndex:0];
                [enc setBuffer:ctx->weight_bufs[l->gpu_q_norm] offset:0 atIndex:1];
                [enc setBytes:&p length:sizeof(p) atIndex:2];
                [enc dispatchThreads:MTLSizeMake(cfg->num_heads * SIMD_SIZE, 1, 1)
                     threadsPerThreadgroup:MTLSizeMake(SIMD_SIZE, 1, 1)];

                p = (rmsnorm_perhead_params_t){ cfg->num_kv_heads, hd, eps };
                [enc setComputePipelineState:ctx->rmsnorm_perhead_pipeline];
                [enc setBuffer:ctx->work_k offset:0 atIndex:0];
                [enc setBuffer:ctx->weight_bufs[l->gpu_k_norm] offset:0 atIndex:1];
                [enc setBytes:&p length:sizeof(p) atIndex:2];
                [enc dispatchThreads:MTLSizeMake(cfg->num_kv_heads * SIMD_SIZE, 1, 1)
                     threadsPerThreadgroup:MTLSizeMake(SIMD_SIZE, 1, 1)];
            }

            /* ── 4. NeoX RoPE ── */
            {
                rope_params_t p = { 0, hd, pos };

                p.n_heads = cfg->num_heads;
                [enc setComputePipelineState:ctx->rope_neox_pipeline];
                [enc setBuffer:ctx->work_q offset:0 atIndex:0];
                [enc setBuffer:ctx->weight_bufs[cfg->gpu_rope_cos] offset:0 atIndex:1];
                [enc setBuffer:ctx->weight_bufs[cfg->gpu_rope_sin] offset:0 atIndex:2];
                [enc setBytes:&p length:sizeof(p) atIndex:3];
                [enc dispatchThreads:MTLSizeMake(cfg->num_heads * half_dim, 1, 1)
                     threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];

                p.n_heads = cfg->num_kv_heads;
                [enc setComputePipelineState:ctx->rope_neox_pipeline];
                [enc setBuffer:ctx->work_k offset:0 atIndex:0];
                [enc setBuffer:ctx->weight_bufs[cfg->gpu_rope_cos] offset:0 atIndex:1];
                [enc setBuffer:ctx->weight_bufs[cfg->gpu_rope_sin] offset:0 atIndex:2];
                [enc setBytes:&p length:sizeof(p) atIndex:3];
                [enc dispatchThreads:MTLSizeMake(cfg->num_kv_heads * half_dim, 1, 1)
                     threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
            }

            /* ── 5. KV store (f32 → bf16 into cache) ── */
            {
                kv_store_params_t p = { kv_dim, pos * kv_dim };
                [enc setComputePipelineState:ctx->kv_store_pipeline];
                [enc setBuffer:ctx->work_k offset:0 atIndex:0];
                [enc setBuffer:ctx->work_v offset:0 atIndex:1];
                [enc setBuffer:ctx->kv_caches[kv_id].k offset:kv_layer_offset atIndex:2];
                [enc setBuffer:ctx->kv_caches[kv_id].v offset:kv_layer_offset atIndex:3];
                [enc setBytes:&p length:sizeof(p) atIndex:4];
                [enc dispatchThreads:MTLSizeMake(kv_dim, 1, 1)
                     threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            }

            /* ── 6. GQA Attention ── */
            {
                attention_params_t p = {
                    cfg->num_heads, cfg->num_kv_heads, hd,
                    kv_dim, pos + 1, scale
                };
                [enc setComputePipelineState:ctx->attention_pipeline];
                [enc setBuffer:ctx->work_q offset:0 atIndex:0];
                [enc setBuffer:ctx->kv_caches[kv_id].k offset:kv_layer_offset atIndex:1];
                [enc setBuffer:ctx->kv_caches[kv_id].v offset:kv_layer_offset atIndex:2];
                [enc setBuffer:ctx->work_attn offset:0 atIndex:3];
                [enc setBytes:&p length:sizeof(p) atIndex:4];
                [enc dispatchThreadgroups:MTLSizeMake(cfg->num_heads, 1, 1)
                      threadsPerThreadgroup:MTLSizeMake(SIMD_SIZE, 1, 1)];
            }

            /* ── 7. Output projection + residual ── */
            {
                matvec_params_t p = { h, q_dim };
                [enc setComputePipelineState:ctx->matvec_pipeline];
                [enc setBuffer:ctx->weight_bufs[l->gpu_wo] offset:0 atIndex:0];
                [enc setBuffer:ctx->work_attn offset:0 atIndex:1];
                [enc setBuffer:ctx->work_proj offset:0 atIndex:2];
                [enc setBytes:&p length:sizeof(p) atIndex:3];
                [enc dispatchThreads:MTLSizeMake(h * SIMD_SIZE, 1, 1)
                     threadsPerThreadgroup:MTLSizeMake(SIMD_SIZE, 1, 1)];
            }
            {
                int dim = h;
                [enc setComputePipelineState:ctx->residual_add_pipeline];
                [enc setBuffer:ctx->work_x offset:0 atIndex:0];
                [enc setBuffer:ctx->work_proj offset:0 atIndex:1];
                [enc setBytes:&dim length:sizeof(dim) atIndex:2];
                [enc dispatchThreads:MTLSizeMake(h, 1, 1)
                     threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            }

            /* ── 8. Post-attention RMSNorm ── */
            {
                rmsnorm_params_t p = { h, eps };
                [enc setComputePipelineState:ctx->rmsnorm_pipeline];
                [enc setBuffer:ctx->work_x offset:0 atIndex:0];
                [enc setBuffer:ctx->work_x_norm offset:0 atIndex:1];
                [enc setBuffer:ctx->weight_bufs[l->gpu_post_attn_norm] offset:0 atIndex:2];
                [enc setBytes:&p length:sizeof(p) atIndex:3];
                [enc dispatchThreads:MTLSizeMake(SIMD_SIZE, 1, 1)
                     threadsPerThreadgroup:MTLSizeMake(SIMD_SIZE, 1, 1)];
            }

            /* ── 9. Fused gate+up matvec ── */
            {
                matvec_params_t p = { 2 * inter, h };
                [enc setComputePipelineState:ctx->matvec_pipeline];
                [enc setBuffer:ctx->weight_bufs[l->gpu_gate_up_fused] offset:0 atIndex:0];
                [enc setBuffer:ctx->work_x_norm offset:0 atIndex:1];
                [enc setBuffer:ctx->work_gate offset:0 atIndex:2];
                [enc setBytes:&p length:sizeof(p) atIndex:3];
                [enc dispatchThreads:MTLSizeMake(2 * inter * SIMD_SIZE, 1, 1)
                     threadsPerThreadgroup:MTLSizeMake(SIMD_SIZE, 1, 1)];
            }

            /* ── 10. SwiGLU ── */
            {
                int inter_val = inter;
                [enc setComputePipelineState:ctx->swiglu_pipeline];
                [enc setBuffer:ctx->work_gate offset:0 atIndex:0];
                [enc setBuffer:ctx->work_ffn offset:0 atIndex:1];
                [enc setBytes:&inter_val length:sizeof(inter_val) atIndex:2];
                [enc dispatchThreads:MTLSizeMake(inter, 1, 1)
                     threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            }

            /* ── 11. Down projection + residual ── */
            {
                matvec_params_t p = { h, inter };
                [enc setComputePipelineState:ctx->matvec_pipeline];
                [enc setBuffer:ctx->weight_bufs[l->gpu_down] offset:0 atIndex:0];
                [enc setBuffer:ctx->work_ffn offset:0 atIndex:1];
                [enc setBuffer:ctx->work_proj offset:0 atIndex:2];
                [enc setBytes:&p length:sizeof(p) atIndex:3];
                [enc dispatchThreads:MTLSizeMake(h * SIMD_SIZE, 1, 1)
                     threadsPerThreadgroup:MTLSizeMake(SIMD_SIZE, 1, 1)];
            }
            {
                int dim = h;
                [enc setComputePipelineState:ctx->residual_add_pipeline];
                [enc setBuffer:ctx->work_x offset:0 atIndex:0];
                [enc setBuffer:ctx->work_proj offset:0 atIndex:1];
                [enc setBytes:&dim length:sizeof(dim) atIndex:2];
                [enc dispatchThreads:MTLSizeMake(h, 1, 1)
                     threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            }
        }

        /* ── Final RMSNorm (if requested) ── */
        if (normed_out && cfg->gpu_final_norm >= 0) {
            rmsnorm_params_t p = { h, eps };
            [enc setComputePipelineState:ctx->rmsnorm_pipeline];
            [enc setBuffer:ctx->work_x offset:0 atIndex:0];
            [enc setBuffer:ctx->work_x_norm offset:0 atIndex:1];
            [enc setBuffer:ctx->weight_bufs[cfg->gpu_final_norm] offset:0 atIndex:2];
            [enc setBytes:&p length:sizeof(p) atIndex:3];
            [enc dispatchThreads:MTLSizeMake(SIMD_SIZE, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(SIMD_SIZE, 1, 1)];
        }

        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
    }

    /* Copy results back to CPU */
    memcpy(x_inout, [ctx->work_x contents], h * sizeof(float));
    if (normed_out && cfg->gpu_final_norm >= 0) {
        memcpy(normed_out, [ctx->work_x_norm contents], h * sizeof(float));
    }

    return 0;
}

#endif /* ENABLE_METAL */
