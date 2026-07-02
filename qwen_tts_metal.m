/*
 * qwen_tts_metal.m — Apple Metal backend (G2). Objective-C, clang -fobjc-arc.
 *
 * Design (see qwen_tts_metal.h): weights RESIDENT (uploaded once, cached by
 * pointer), IO buffers pooled+reused — zero per-call allocation in steady state.
 * All quant dequant happens in-shader, matching the CPU kernels bit-for-bit.
 *
 * Honest M1 framing (plan_v4 §E4.ter): single-stream matvec on the shared-memory
 * M1 GPU is bandwidth-bound → ~parity-or-worse vs the tuned NEON CPU path; the
 * real wins are batched matmat (compute-bound) and decoder offload + CPU/GPU
 * overlap. This file provides the correct primitives + a per-op selftest so we
 * can MEASURE where the GPU actually helps and optimize from real numbers.
 */

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include "qwen_tts_metal.h"
#include "qwen_tts_kernels.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* ---- embedded Metal Shading Language ------------------------------------ */
static const char *QWEN_METAL_SRC =
"#include <metal_stdlib>\n"
"using namespace metal;\n"
"inline float bf16_to_f32(ushort b) { return as_type<float>(uint(b) << 16); }\n"
"struct q4blk { float scale; uchar qs[16]; };\n"
"\n"
"kernel void matvec_bf16(device const ushort *W [[buffer(0)]],\n"
"    device const float *x [[buffer(1)]], device float *y [[buffer(2)]],\n"
"    constant uint &cols [[buffer(3)]], uint row [[thread_position_in_grid]]) {\n"
"    float acc = 0.0f; device const ushort *w = W + (ulong)row * cols;\n"
"    for (uint c = 0; c < cols; ++c) acc += bf16_to_f32(w[c]) * x[c];\n"
"    y[row] = acc;\n"
"}\n"
"\n"
"kernel void matmat_bf16(device const ushort *W [[buffer(0)]],\n"
"    device const float *X [[buffer(1)]], device float *Y [[buffer(2)]],\n"
"    constant uint &cols [[buffer(3)]], constant uint &B [[buffer(4)]],\n"
"    uint row [[thread_position_in_grid]]) {\n"
"    float acc[64]; for (uint b = 0; b < B; ++b) acc[b] = 0.0f;\n"
"    device const ushort *w = W + (ulong)row * cols;\n"
"    for (uint c = 0; c < cols; ++c) { float wv = bf16_to_f32(w[c]);\n"
"        device const float *xc = X + (ulong)c * B;\n"
"        for (uint b = 0; b < B; ++b) acc[b] += wv * xc[b]; }\n"
"    device float *yr = Y + (ulong)row * B;\n"
"    for (uint b = 0; b < B; ++b) yr[b] = acc[b];\n"
"}\n"
"\n"
"kernel void matvec_int8(device const char *W [[buffer(0)]],\n"
"    device const float *scale [[buffer(1)]], device const float *x [[buffer(2)]],\n"
"    device float *y [[buffer(3)]], constant uint &cols [[buffer(4)]],\n"
"    uint row [[thread_position_in_grid]]) {\n"
"    float acc = 0.0f; device const char *w = W + (ulong)row * cols;\n"
"    for (uint c = 0; c < cols; ++c) acc += float(w[c]) * x[c];\n"
"    y[row] = acc * scale[row];\n"
"}\n"
"\n"
"kernel void matvec_q4_0(device const q4blk *W [[buffer(0)]],\n"
"    device const float *x [[buffer(1)]], device float *y [[buffer(2)]],\n"
"    constant uint &cols [[buffer(3)]], uint row [[thread_position_in_grid]]) {\n"
"    uint bpr = cols / 32; device const q4blk *wr = W + (ulong)row * bpr;\n"
"    float acc = 0.0f;\n"
"    for (uint b = 0; b < bpr; ++b) { float s = wr[b].scale;\n"
"        for (uint i = 0; i < 16; ++i) { uchar q = wr[b].qs[i];\n"
"            int lo = int(q & 0x0f) - 8; int hi = int(q >> 4) - 8;\n"
"            acc += float(lo) * s * x[b*32 + 2*i];\n"
"            acc += float(hi) * s * x[b*32 + 2*i + 1]; } }\n"
"    y[row] = acc;\n"
"}\n"
"\n"
"kernel void rms_norm(device const float *x [[buffer(0)]],\n"
"    device const float *w [[buffer(1)]], device float *y [[buffer(2)]],\n"
"    constant uint &dim [[buffer(3)]], constant float &eps [[buffer(4)]],\n"
"    uint tid [[thread_position_in_threadgroup]], uint tc [[threads_per_threadgroup]]) {\n"
"    threadgroup float part[256]; float s = 0.0f;\n"
"    for (uint i = tid; i < dim; i += tc) s += x[i] * x[i];\n"
"    part[tid] = s; threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    for (uint stride = tc / 2; stride > 0; stride >>= 1) {\n"
"        if (tid < stride) part[tid] += part[tid + stride];\n"
"        threadgroup_barrier(mem_flags::mem_threadgroup); }\n"
"    float inv = 1.0f / sqrt(part[0] / float(dim) + eps);\n"
"    for (uint i = tid; i < dim; i += tc) y[i] = x[i] * inv * w[i];\n"
"}\n"
"\n"
"kernel void swiglu(device const float *in [[buffer(0)]], device float *out [[buffer(1)]],\n"
"    uint i [[thread_position_in_grid]]) {\n"
"    float g = in[2*i], u = in[2*i + 1]; out[i] = g / (1.0f + exp(-g)) * u;\n"
"}\n"
"kernel void silu(device const float *x [[buffer(0)]], device float *out [[buffer(1)]],\n"
"    uint i [[thread_position_in_grid]]) { float v = x[i]; out[i] = v / (1.0f + exp(-v)); }\n"
"kernel void eadd(device const float *a [[buffer(0)]], device const float *b [[buffer(1)]],\n"
"    device float *out [[buffer(2)]], uint i [[thread_position_in_grid]]) { out[i] = a[i] + b[i]; }\n"
"kernel void emul(device const float *a [[buffer(0)]], device const float *b [[buffer(1)]],\n"
"    device float *out [[buffer(2)]], uint i [[thread_position_in_grid]]) { out[i] = a[i] * b[i]; }\n"
"kernel void escale(device const float *a [[buffer(0)]], device float *out [[buffer(1)]],\n"
"    constant float &s [[buffer(2)]], uint i [[thread_position_in_grid]]) { out[i] = a[i] * s; }\n"
"\n"
"kernel void rope(device float *x [[buffer(0)]], device const float *cosv [[buffer(1)]],\n"
"    device const float *sinv [[buffer(2)]], constant uint &head_dim [[buffer(3)]],\n"
"    uint gid [[thread_position_in_grid]]) {\n"
"    uint pairs = head_dim / 2; uint h = gid / pairs; uint d = gid % pairs;\n"
"    device float *vec = x + h * head_dim; float c = cosv[d], sn = sinv[d];\n"
"    float xe = vec[2*d], xo = vec[2*d + 1];\n"
"    vec[2*d] = xe * c - xo * sn; vec[2*d + 1] = xo * c + xe * sn;\n"
"}\n";

/* ---- context: device, queue, pipelines, resident weight cache, IO pool --- */
typedef struct { const void *key; void *buf; } wcache_ent;   /* buf = bridge-retained id<MTLBuffer> */

#define QWEN_MTL_IO_SLOTS 6
typedef struct {
    void *device, *queue;
    /* pipelines */
    void *pso_matvec_bf16, *pso_matmat_bf16, *pso_matvec_int8, *pso_matvec_q4_0;
    void *pso_rms, *pso_swiglu, *pso_silu, *pso_add, *pso_mul, *pso_scale, *pso_rope;
    /* resident weight cache (by pointer) */
    wcache_ent *wc; int wc_n, wc_cap;
    /* reusable IO buffers (grow on demand) */
    void *io[QWEN_MTL_IO_SLOTS]; size_t io_len[QWEN_MTL_IO_SLOTS];
} qwen_metal_ctx;

int qwen_metal_available(void) {
    @autoreleasepool { return MTLCreateSystemDefaultDevice() != nil; }
}

static void *make_pso(id<MTLDevice> dev, id<MTLLibrary> lib, const char *name) {
    NSError *err = nil;
    id<MTLFunction> fn = [lib newFunctionWithName:[NSString stringWithUTF8String:name]];
    if (!fn) { fprintf(stderr, "Metal: missing kernel '%s'\n", name); return NULL; }
    id<MTLComputePipelineState> p = [dev newComputePipelineStateWithFunction:fn error:&err];
    if (!p) { fprintf(stderr, "Metal: pso '%s' failed: %s\n", name,
                      err ? err.localizedDescription.UTF8String : "(unknown)"); return NULL; }
    return (__bridge_retained void *)p;
}

void *qwen_metal_init(void) {
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev) { fprintf(stderr, "Metal: no device\n"); return NULL; }
        NSError *err = nil;
        id<MTLLibrary> lib = [dev newLibraryWithSource:[NSString stringWithUTF8String:QWEN_METAL_SRC]
                                               options:nil error:&err];
        if (!lib) { fprintf(stderr, "Metal: shader compile failed: %s\n",
                            err ? err.localizedDescription.UTF8String : "(unknown)"); return NULL; }
        id<MTLCommandQueue> q = [dev newCommandQueue];
        if (!q) { fprintf(stderr, "Metal: queue failed\n"); return NULL; }

        qwen_metal_ctx *c = calloc(1, sizeof(*c));
        if (!c) return NULL;
        c->device = (__bridge_retained void *)dev;
        c->queue  = (__bridge_retained void *)q;
        c->pso_matvec_bf16 = make_pso(dev, lib, "matvec_bf16");
        c->pso_matmat_bf16 = make_pso(dev, lib, "matmat_bf16");
        c->pso_matvec_int8 = make_pso(dev, lib, "matvec_int8");
        c->pso_matvec_q4_0 = make_pso(dev, lib, "matvec_q4_0");
        c->pso_rms    = make_pso(dev, lib, "rms_norm");
        c->pso_swiglu = make_pso(dev, lib, "swiglu");
        c->pso_silu   = make_pso(dev, lib, "silu");
        c->pso_add    = make_pso(dev, lib, "eadd");
        c->pso_mul    = make_pso(dev, lib, "emul");
        c->pso_scale  = make_pso(dev, lib, "escale");
        c->pso_rope   = make_pso(dev, lib, "rope");
        if (!c->pso_matvec_bf16 || !c->pso_matmat_bf16 || !c->pso_matvec_int8 ||
            !c->pso_matvec_q4_0 || !c->pso_rms || !c->pso_swiglu || !c->pso_silu ||
            !c->pso_add || !c->pso_mul || !c->pso_scale || !c->pso_rope) {
            qwen_metal_free(c); return NULL;
        }
        return c;
    }
}

static void release_ptr(void **p) { if (*p) { id o = (__bridge_transfer id)*p; (void)o; *p = NULL; } }

void qwen_metal_free(void *ctx) {
    if (!ctx) return;
    qwen_metal_ctx *c = ctx;
    for (int i = 0; i < c->wc_n; ++i) { void *b = c->wc[i].buf; release_ptr(&b); }
    free(c->wc);
    for (int i = 0; i < QWEN_MTL_IO_SLOTS; ++i) release_ptr(&c->io[i]);
    release_ptr(&c->pso_matvec_bf16); release_ptr(&c->pso_matmat_bf16);
    release_ptr(&c->pso_matvec_int8); release_ptr(&c->pso_matvec_q4_0);
    release_ptr(&c->pso_rms); release_ptr(&c->pso_swiglu); release_ptr(&c->pso_silu);
    release_ptr(&c->pso_add); release_ptr(&c->pso_mul); release_ptr(&c->pso_scale);
    release_ptr(&c->pso_rope);
    release_ptr(&c->queue); release_ptr(&c->device);
    free(c);
}

/* Resident weight buffer, cached by pointer (uploaded once). */
static id<MTLBuffer> weight_buf(qwen_metal_ctx *c, const void *ptr, size_t bytes) {
    for (int i = 0; i < c->wc_n; ++i)
        if (c->wc[i].key == ptr) return (__bridge id<MTLBuffer>)c->wc[i].buf;
    id<MTLDevice> dev = (__bridge id<MTLDevice>)c->device;
    id<MTLBuffer> b = [dev newBufferWithBytes:ptr length:bytes options:MTLResourceStorageModeShared];
    if (c->wc_n == c->wc_cap) {
        c->wc_cap = c->wc_cap ? c->wc_cap * 2 : 64;
        c->wc = realloc(c->wc, (size_t)c->wc_cap * sizeof(wcache_ent));
    }
    c->wc[c->wc_n].key = ptr;
    c->wc[c->wc_n].buf = (__bridge_retained void *)b;
    c->wc_n++;
    return b;
}

/* Reusable IO buffer at a slot, grown as needed (shared storage → CPU can r/w). */
static id<MTLBuffer> io_buf(qwen_metal_ctx *c, int slot, size_t bytes) {
    if (c->io_len[slot] < bytes) {
        release_ptr(&c->io[slot]);
        id<MTLDevice> dev = (__bridge id<MTLDevice>)c->device;
        id<MTLBuffer> b = [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        c->io[slot] = (__bridge_retained void *)b;
        c->io_len[slot] = bytes;
    }
    return (__bridge id<MTLBuffer>)c->io[slot];
}

static NSUInteger cap256(id<MTLComputePipelineState> p) {
    NSUInteger t = p.maxTotalThreadsPerThreadgroup; return t > 256 ? 256 : t;
}

/* ---- op wrappers -------------------------------------------------------- */
void qwen_metal_matvec_bf16(void *ctx, float *y, const uint16_t *W,
                            const float *x, int rows, int cols) {
    @autoreleasepool {
        qwen_metal_ctx *c = ctx;
        id<MTLCommandQueue> q = (__bridge id<MTLCommandQueue>)c->queue;
        id<MTLComputePipelineState> pso = (__bridge id<MTLComputePipelineState>)c->pso_matvec_bf16;
        id<MTLBuffer> bW = weight_buf(c, W, (size_t)rows * cols * sizeof(uint16_t));
        id<MTLBuffer> bx = io_buf(c, 0, (size_t)cols * sizeof(float));
        id<MTLBuffer> by = io_buf(c, 1, (size_t)rows * sizeof(float));
        memcpy(bx.contents, x, (size_t)cols * sizeof(float));
        id<MTLCommandBuffer> cb = [q commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        [e setComputePipelineState:pso];
        [e setBuffer:bW offset:0 atIndex:0]; [e setBuffer:bx offset:0 atIndex:1];
        [e setBuffer:by offset:0 atIndex:2];
        uint32_t cc = (uint32_t)cols; [e setBytes:&cc length:4 atIndex:3];
        [e dispatchThreads:MTLSizeMake((NSUInteger)rows,1,1)
        threadsPerThreadgroup:MTLSizeMake(cap256(pso),1,1)];
        [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
        memcpy(y, by.contents, (size_t)rows * sizeof(float));
    }
}

void qwen_metal_matmat_bf16(void *ctx, float *Y, const uint16_t *W,
                            const float *X, int rows, int cols, int B) {
    @autoreleasepool {
        qwen_metal_ctx *c = ctx;
        id<MTLCommandQueue> q = (__bridge id<MTLCommandQueue>)c->queue;
        id<MTLComputePipelineState> pso = (__bridge id<MTLComputePipelineState>)c->pso_matmat_bf16;
        id<MTLBuffer> bW = weight_buf(c, W, (size_t)rows * cols * sizeof(uint16_t));
        id<MTLBuffer> bX = io_buf(c, 0, (size_t)cols * B * sizeof(float));
        id<MTLBuffer> bY = io_buf(c, 1, (size_t)rows * B * sizeof(float));
        memcpy(bX.contents, X, (size_t)cols * B * sizeof(float));
        id<MTLCommandBuffer> cb = [q commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        [e setComputePipelineState:pso];
        [e setBuffer:bW offset:0 atIndex:0]; [e setBuffer:bX offset:0 atIndex:1];
        [e setBuffer:bY offset:0 atIndex:2];
        uint32_t cc = (uint32_t)cols, cB = (uint32_t)B;
        [e setBytes:&cc length:4 atIndex:3]; [e setBytes:&cB length:4 atIndex:4];
        [e dispatchThreads:MTLSizeMake((NSUInteger)rows,1,1)
        threadsPerThreadgroup:MTLSizeMake(cap256(pso),1,1)];
        [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
        memcpy(Y, bY.contents, (size_t)rows * B * sizeof(float));
    }
}

void qwen_metal_matvec_int8(void *ctx, float *y, const int8_t *W,
                            const float *scale, const float *x, int rows, int cols) {
    @autoreleasepool {
        qwen_metal_ctx *c = ctx;
        id<MTLCommandQueue> q = (__bridge id<MTLCommandQueue>)c->queue;
        id<MTLComputePipelineState> pso = (__bridge id<MTLComputePipelineState>)c->pso_matvec_int8;
        id<MTLBuffer> bW = weight_buf(c, W, (size_t)rows * cols * sizeof(int8_t));
        id<MTLBuffer> bS = weight_buf(c, scale, (size_t)rows * sizeof(float));
        id<MTLBuffer> bx = io_buf(c, 0, (size_t)cols * sizeof(float));
        id<MTLBuffer> by = io_buf(c, 1, (size_t)rows * sizeof(float));
        memcpy(bx.contents, x, (size_t)cols * sizeof(float));
        id<MTLCommandBuffer> cb = [q commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        [e setComputePipelineState:pso];
        [e setBuffer:bW offset:0 atIndex:0]; [e setBuffer:bS offset:0 atIndex:1];
        [e setBuffer:bx offset:0 atIndex:2]; [e setBuffer:by offset:0 atIndex:3];
        uint32_t cc = (uint32_t)cols; [e setBytes:&cc length:4 atIndex:4];
        [e dispatchThreads:MTLSizeMake((NSUInteger)rows,1,1)
        threadsPerThreadgroup:MTLSizeMake(cap256(pso),1,1)];
        [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
        memcpy(y, by.contents, (size_t)rows * sizeof(float));
    }
}

void qwen_metal_matvec_q4_0(void *ctx, float *y, const q4_0_block_t *W,
                            const float *x, int rows, int cols) {
    @autoreleasepool {
        qwen_metal_ctx *c = ctx;
        id<MTLCommandQueue> q = (__bridge id<MTLCommandQueue>)c->queue;
        id<MTLComputePipelineState> pso = (__bridge id<MTLComputePipelineState>)c->pso_matvec_q4_0;
        size_t blocks = (size_t)rows * (cols / 32);
        id<MTLBuffer> bW = weight_buf(c, W, blocks * sizeof(q4_0_block_t));
        id<MTLBuffer> bx = io_buf(c, 0, (size_t)cols * sizeof(float));
        id<MTLBuffer> by = io_buf(c, 1, (size_t)rows * sizeof(float));
        memcpy(bx.contents, x, (size_t)cols * sizeof(float));
        id<MTLCommandBuffer> cb = [q commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        [e setComputePipelineState:pso];
        [e setBuffer:bW offset:0 atIndex:0]; [e setBuffer:bx offset:0 atIndex:1];
        [e setBuffer:by offset:0 atIndex:2];
        uint32_t cc = (uint32_t)cols; [e setBytes:&cc length:4 atIndex:3];
        [e dispatchThreads:MTLSizeMake((NSUInteger)rows,1,1)
        threadsPerThreadgroup:MTLSizeMake(cap256(pso),1,1)];
        [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
        memcpy(y, by.contents, (size_t)rows * sizeof(float));
    }
}

void qwen_metal_rms_norm(void *ctx, float *out, const float *x,
                         const float *weight, int dim, float eps) {
    @autoreleasepool {
        qwen_metal_ctx *c = ctx;
        id<MTLCommandQueue> q = (__bridge id<MTLCommandQueue>)c->queue;
        id<MTLComputePipelineState> pso = (__bridge id<MTLComputePipelineState>)c->pso_rms;
        id<MTLBuffer> bx = io_buf(c, 0, (size_t)dim * sizeof(float));
        id<MTLBuffer> bw = io_buf(c, 1, (size_t)dim * sizeof(float));
        id<MTLBuffer> by = io_buf(c, 2, (size_t)dim * sizeof(float));
        memcpy(bx.contents, x, (size_t)dim * sizeof(float));
        memcpy(bw.contents, weight, (size_t)dim * sizeof(float));
        id<MTLCommandBuffer> cb = [q commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        [e setComputePipelineState:pso];
        [e setBuffer:bx offset:0 atIndex:0]; [e setBuffer:bw offset:0 atIndex:1];
        [e setBuffer:by offset:0 atIndex:2];
        uint32_t d = (uint32_t)dim; float ep = eps;
        [e setBytes:&d length:4 atIndex:3]; [e setBytes:&ep length:4 atIndex:4];
        [e dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
        memcpy(out, by.contents, (size_t)dim * sizeof(float));
    }
}

/* one-in one-out elementwise dispatch helper */
static void run_1in_1out(qwen_metal_ctx *c, id<MTLComputePipelineState> pso,
                         float *out, const float *in, int n) {
    @autoreleasepool {
        id<MTLCommandQueue> q = (__bridge id<MTLCommandQueue>)c->queue;
        id<MTLBuffer> bi = io_buf(c, 0, (size_t)n * sizeof(float));
        id<MTLBuffer> bo = io_buf(c, 1, (size_t)n * sizeof(float));
        memcpy(bi.contents, in, (size_t)n * sizeof(float));
        id<MTLCommandBuffer> cb = [q commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        [e setComputePipelineState:pso];
        [e setBuffer:bi offset:0 atIndex:0]; [e setBuffer:bo offset:0 atIndex:1];
        [e dispatchThreads:MTLSizeMake((NSUInteger)n,1,1)
        threadsPerThreadgroup:MTLSizeMake(cap256(pso),1,1)];
        [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
        memcpy(out, bo.contents, (size_t)n * sizeof(float));
    }
}

void qwen_metal_swiglu(void *ctx, float *out, const float *gate_up, int n) {
    qwen_metal_ctx *c = ctx;
    /* input has 2n elements (interleaved g,u); output n */
    @autoreleasepool {
        id<MTLCommandQueue> q = (__bridge id<MTLCommandQueue>)c->queue;
        id<MTLComputePipelineState> pso = (__bridge id<MTLComputePipelineState>)c->pso_swiglu;
        id<MTLBuffer> bi = io_buf(c, 0, (size_t)(2*n) * sizeof(float));
        id<MTLBuffer> bo = io_buf(c, 1, (size_t)n * sizeof(float));
        memcpy(bi.contents, gate_up, (size_t)(2*n) * sizeof(float));
        id<MTLCommandBuffer> cb = [q commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        [e setComputePipelineState:pso];
        [e setBuffer:bi offset:0 atIndex:0]; [e setBuffer:bo offset:0 atIndex:1];
        [e dispatchThreads:MTLSizeMake((NSUInteger)n,1,1)
        threadsPerThreadgroup:MTLSizeMake(cap256(pso),1,1)];
        [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
        memcpy(out, bo.contents, (size_t)n * sizeof(float));
    }
}

void qwen_metal_silu(void *ctx, float *out, const float *x, int n) {
    qwen_metal_ctx *c = ctx;
    run_1in_1out(c, (__bridge id<MTLComputePipelineState>)c->pso_silu, out, x, n);
}

static void run_2in_1out(qwen_metal_ctx *c, id<MTLComputePipelineState> pso,
                         float *out, const float *a, const float *b, int n) {
    @autoreleasepool {
        id<MTLCommandQueue> q = (__bridge id<MTLCommandQueue>)c->queue;
        id<MTLBuffer> ba = io_buf(c, 0, (size_t)n * sizeof(float));
        id<MTLBuffer> bb = io_buf(c, 1, (size_t)n * sizeof(float));
        id<MTLBuffer> bo = io_buf(c, 2, (size_t)n * sizeof(float));
        memcpy(ba.contents, a, (size_t)n * sizeof(float));
        memcpy(bb.contents, b, (size_t)n * sizeof(float));
        id<MTLCommandBuffer> cb = [q commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        [e setComputePipelineState:pso];
        [e setBuffer:ba offset:0 atIndex:0]; [e setBuffer:bb offset:0 atIndex:1];
        [e setBuffer:bo offset:0 atIndex:2];
        [e dispatchThreads:MTLSizeMake((NSUInteger)n,1,1)
        threadsPerThreadgroup:MTLSizeMake(cap256(pso),1,1)];
        [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
        memcpy(out, bo.contents, (size_t)n * sizeof(float));
    }
}

void qwen_metal_add(void *ctx, float *out, const float *a, const float *b, int n) {
    qwen_metal_ctx *c = ctx;
    run_2in_1out(c, (__bridge id<MTLComputePipelineState>)c->pso_add, out, a, b, n);
}
void qwen_metal_mul(void *ctx, float *out, const float *a, const float *b, int n) {
    qwen_metal_ctx *c = ctx;
    run_2in_1out(c, (__bridge id<MTLComputePipelineState>)c->pso_mul, out, a, b, n);
}
void qwen_metal_scale(void *ctx, float *out, const float *a, float s, int n) {
    @autoreleasepool {
        qwen_metal_ctx *c = ctx;
        id<MTLCommandQueue> q = (__bridge id<MTLCommandQueue>)c->queue;
        id<MTLComputePipelineState> pso = (__bridge id<MTLComputePipelineState>)c->pso_scale;
        id<MTLBuffer> ba = io_buf(c, 0, (size_t)n * sizeof(float));
        id<MTLBuffer> bo = io_buf(c, 1, (size_t)n * sizeof(float));
        memcpy(ba.contents, a, (size_t)n * sizeof(float));
        id<MTLCommandBuffer> cb = [q commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        [e setComputePipelineState:pso];
        [e setBuffer:ba offset:0 atIndex:0]; [e setBuffer:bo offset:0 atIndex:1];
        [e setBytes:&s length:4 atIndex:2];
        [e dispatchThreads:MTLSizeMake((NSUInteger)n,1,1)
        threadsPerThreadgroup:MTLSizeMake(cap256(pso),1,1)];
        [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
        memcpy(out, bo.contents, (size_t)n * sizeof(float));
    }
}

void qwen_metal_rope(void *ctx, float *x, const float *cosv, const float *sinv,
                     int n_heads, int head_dim) {
    @autoreleasepool {
        qwen_metal_ctx *c = ctx;
        int pairs = head_dim / 2; int total = n_heads * pairs;
        id<MTLCommandQueue> q = (__bridge id<MTLCommandQueue>)c->queue;
        id<MTLComputePipelineState> pso = (__bridge id<MTLComputePipelineState>)c->pso_rope;
        id<MTLBuffer> bx = io_buf(c, 0, (size_t)n_heads * head_dim * sizeof(float));
        id<MTLBuffer> bc = io_buf(c, 1, (size_t)pairs * sizeof(float));
        id<MTLBuffer> bs = io_buf(c, 2, (size_t)pairs * sizeof(float));
        memcpy(bx.contents, x, (size_t)n_heads * head_dim * sizeof(float));
        memcpy(bc.contents, cosv, (size_t)pairs * sizeof(float));
        memcpy(bs.contents, sinv, (size_t)pairs * sizeof(float));
        id<MTLCommandBuffer> cb = [q commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        [e setComputePipelineState:pso];
        [e setBuffer:bx offset:0 atIndex:0]; [e setBuffer:bc offset:0 atIndex:1];
        [e setBuffer:bs offset:0 atIndex:2];
        uint32_t hd = (uint32_t)head_dim; [e setBytes:&hd length:4 atIndex:3];
        [e dispatchThreads:MTLSizeMake((NSUInteger)total,1,1)
        threadsPerThreadgroup:MTLSizeMake(cap256(pso),1,1)];
        [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
        memcpy(x, bx.contents, (size_t)n_heads * head_dim * sizeof(float));
    }
}

/* ---- per-op selftest vs CPU kernels ------------------------------------- */
static uint32_t g_rng = 0x1234567u;
static float rnd(void) {
    g_rng ^= g_rng << 13; g_rng ^= g_rng >> 17; g_rng ^= g_rng << 5;
    return ((float)(g_rng & 0xFFFFFF) / (float)0xFFFFFF) * 2.0f - 1.0f;
}
static double nowms(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1000.0 + t.tv_nsec / 1e6; }
static uint16_t to_bf16(float f) { union { float f; uint32_t u; } v; v.f = f; return (uint16_t)(v.u >> 16); }

static int check(FILE *f, const char *name, const float *a, const float *b, int n, double tol) {
    double mabs = 0, mref = 0;
    for (int i = 0; i < n; ++i) { double d = fabs((double)a[i]-b[i]); if (d>mabs) mabs=d;
        double r = fabs((double)b[i]); if (r>mref) mref=r; }
    double rel = mref > 0 ? mabs/mref : mabs;
    int ok = rel < tol;
    fprintf(f, "  %-14s rel=%.2e  %s\n", name, rel, ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

int qwen_metal_selftest(void *out) {
    FILE *f = out ? (FILE *)out : stdout;
    void *ctx = qwen_metal_init();
    if (!ctx) { fprintf(f, "metal-selftest: init failed\n"); return 1; }
    int fails = 0;
    const int rows = 2048, cols = 2048, dim = 2048, B = 8;

    uint16_t *Wb = malloc((size_t)rows*cols*2);
    int8_t   *Wi = malloc((size_t)rows*cols);
    float    *Ws = malloc((size_t)rows*4);
    q4_0_block_t *Wq = malloc((size_t)rows*(cols/32)*sizeof(q4_0_block_t));
    float *x = malloc((size_t)cols*4), *X = malloc((size_t)cols*B*4);
    float *w = malloc((size_t)dim*4), *gu = malloc((size_t)2*dim*4);
    float *ca = malloc((size_t)dim*4), *cb2 = malloc((size_t)dim*4);
    float *rc = malloc((size_t)dim*4), *rs = malloc((size_t)dim*4);
    float *cpu = malloc((size_t)rows*B*4), *gpu = malloc((size_t)rows*B*4);
    float *cpu2 = malloc((size_t)dim*4), *gpu2 = malloc((size_t)dim*4);

    g_rng = 0x1234567u;
    for (int i = 0; i < cols; ++i) x[i] = rnd();
    for (int i = 0; i < cols*B; ++i) X[i] = rnd();
    for (int i = 0; i < dim; ++i) { w[i] = rnd()*0.5f + 1.0f; ca[i] = rnd(); cb2[i] = rnd(); }
    for (int i = 0; i < 2*dim; ++i) gu[i] = rnd();
    /* bf16 weights + quantize to int8/q4 with the CPU quantizers (bit-exact refs) */
    for (size_t i = 0; i < (size_t)rows*cols; ++i) Wb[i] = to_bf16(rnd()*0.1f);
    qwen_quantize_bf16_to_int8(Wb, rows, cols, Wi, Ws);
    qwen_quantize_bf16_to_q4_0(Wb, rows, cols, Wq);
    int pairs = 64; int nh = dim / 128; int hd = 128;   /* rope: nh heads × 128 */
    for (int i = 0; i < pairs; ++i) { float a = rnd(); rc[i] = cosf(a); rs[i] = sinf(a); }

    fprintf(f, "metal-selftest: rows=%d cols=%d dim=%d B=%d (resident weights)\n", rows, cols, dim, B);

    /* matvec bf16 */
    qwen_matvec_bf16(cpu, Wb, x, rows, cols);
    qwen_metal_matvec_bf16(ctx, gpu, Wb, x, rows, cols);
    fails += check(f, "matvec_bf16", gpu, cpu, rows, 1e-2);
    /* matmat bf16 */
    qwen_matmat_bf16(cpu, Wb, X, rows, cols, B);
    qwen_metal_matmat_bf16(ctx, gpu, Wb, X, rows, cols, B);
    fails += check(f, "matmat_bf16", gpu, cpu, rows*B, 1e-2);
    /* matvec int8 */
    qwen_matvec_int8(cpu, Wi, Ws, x, rows, cols);
    qwen_metal_matvec_int8(ctx, gpu, Wi, Ws, x, rows, cols);
    fails += check(f, "matvec_int8", gpu, cpu, rows, 1e-2);
    /* matvec q4_0 */
    qwen_matvec_q4_0(cpu, Wq, x, rows, cols);
    qwen_metal_matvec_q4_0(ctx, gpu, Wq, x, rows, cols);
    fails += check(f, "matvec_q4_0", gpu, cpu, rows, 1e-2);
    /* rms_norm */
    qwen_rms_norm(cpu2, ca, w, 1, dim, 1e-6f);
    qwen_metal_rms_norm(ctx, gpu2, ca, w, dim, 1e-6f);
    fails += check(f, "rms_norm", gpu2, cpu2, dim, 1e-3);
    /* swiglu (n=dim; input 2*dim) */
    { float *t = malloc((size_t)dim*4); float *g2 = malloc((size_t)2*dim*4);
      memcpy(g2, gu, (size_t)2*dim*4);
      qwen_swiglu_inplace(g2, t, dim);          /* CPU writes result to g2[0..dim) */
      qwen_metal_swiglu(ctx, gpu2, gu, dim);
      fails += check(f, "swiglu", gpu2, g2, dim, 1e-3);
      free(t); free(g2); }
    /* silu */
    memcpy(cpu2, ca, (size_t)dim*4); qwen_silu(cpu2, dim);
    qwen_metal_silu(ctx, gpu2, ca, dim);
    fails += check(f, "silu", gpu2, cpu2, dim, 1e-3);
    /* add / mul / scale */
    memcpy(cpu2, ca, (size_t)dim*4); qwen_add_inplace(cpu2, cb2, dim);
    qwen_metal_add(ctx, gpu2, ca, cb2, dim);
    fails += check(f, "add", gpu2, cpu2, dim, 1e-4);
    memcpy(cpu2, ca, (size_t)dim*4); qwen_mul_inplace(cpu2, cb2, dim);
    qwen_metal_mul(ctx, gpu2, ca, cb2, dim);
    fails += check(f, "mul", gpu2, cpu2, dim, 1e-4);
    memcpy(cpu2, ca, (size_t)dim*4); qwen_vec_scale_inplace(cpu2, 1.7f, dim);
    qwen_metal_scale(ctx, gpu2, ca, 1.7f, dim);
    fails += check(f, "scale", gpu2, cpu2, dim, 1e-4);
    /* rope (seq=1, nh heads × 128) */
    { float *xr = malloc((size_t)nh*hd*4); float *xr2 = malloc((size_t)nh*hd*4);
      for (int i = 0; i < nh*hd; ++i) { float v = rnd(); xr[i] = v; xr2[i] = v; }
      qwen_apply_rope_interleaved(xr, rc, rs, 1, nh, hd);
      qwen_metal_rope(ctx, xr2, rc, rs, nh, hd);
      fails += check(f, "rope", xr2, xr, nh*hd, 1e-3);
      free(xr); free(xr2); }

    /* ---- resident timing: matvec bf16 + matmat bf16 (weights already cached) ---- */
    const int it = 30;
    double t0 = nowms(); for (int k = 0; k < it; ++k) qwen_matvec_bf16(cpu, Wb, x, rows, cols);
    double tc_mv = (nowms()-t0)/it;
    t0 = nowms(); for (int k = 0; k < it; ++k) qwen_metal_matvec_bf16(ctx, gpu, Wb, x, rows, cols);
    double tg_mv = (nowms()-t0)/it;
    t0 = nowms(); for (int k = 0; k < it; ++k) qwen_matmat_bf16(cpu, Wb, X, rows, cols, B);
    double tc_mm = (nowms()-t0)/it;
    t0 = nowms(); for (int k = 0; k < it; ++k) qwen_metal_matmat_bf16(ctx, gpu, Wb, X, rows, cols, B);
    double tg_mm = (nowms()-t0)/it;
    fprintf(f, "  timing matvec_bf16: cpu=%.3f ms  metal=%.3f ms  (%.2fx)\n", tc_mv, tg_mv, tg_mv>0?tc_mv/tg_mv:0);
    fprintf(f, "  timing matmat_bf16: cpu=%.3f ms  metal=%.3f ms  (%.2fx)\n", tc_mm, tg_mm, tg_mm>0?tc_mm/tg_mm:0);
    fprintf(f, "metal-selftest: %s (%d op%s failing)\n", fails ? "FAIL" : "PASS", fails, fails==1?"":"s");

    free(Wb); free(Wi); free(Ws); free(Wq); free(x); free(X); free(w); free(gu);
    free(ca); free(cb2); free(rc); free(rs); free(cpu); free(gpu); free(cpu2); free(gpu2);
    qwen_metal_free(ctx);
    return fails;
}
