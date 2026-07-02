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
/* simdgroup matvec: one output row per simdgroup; 32 lanes stride the cols with
 * coalesced weight reads, then simd_sum reduces (the ggml-metal mmv pattern). */
"kernel void matvec_bf16(device const ushort *W [[buffer(0)]],\n"
"    device const float *x [[buffer(1)]], device float *y [[buffer(2)]],\n"
"    constant uint &cols [[buffer(3)]], constant uint &rows [[buffer(4)]],\n"
"    uint3 tgpig [[threadgroup_position_in_grid]],\n"
"    uint tiisg [[thread_index_in_simdgroup]],\n"
"    uint sgitg [[simdgroup_index_in_threadgroup]],\n"
"    uint nsg [[simdgroups_per_threadgroup]]) {\n"
"    uint row = tgpig.x * nsg + sgitg; if (row >= rows) return;\n"
"    device const ushort4 *w4 = (device const ushort4 *)(W + (ulong)row * cols);\n"
"    device const float4 *x4 = (device const float4 *)x;\n"
"    uint n4 = cols >> 2; float acc = 0.0f;\n"
"    for (uint c = tiisg; c < n4; c += 32) { ushort4 b = w4[c];\n"
"        float4 wf = float4(as_type<float>(uint(b.x)<<16), as_type<float>(uint(b.y)<<16),\n"
"                           as_type<float>(uint(b.z)<<16), as_type<float>(uint(b.w)<<16));\n"
"        acc += dot(wf, x4[c]); }\n"
"    acc = simd_sum(acc); if (tiisg == 0) y[row] = acc;\n"
"}\n"
"\n"
/* MMA matmat: simdgroup_matrix 8x8 tiles (the GPU matrix units, 512 MAC/instr).
 * One simdgroup computes an 8-row x 8-B output tile; K looped in 8-tiles staged
 * to threadgroup memory (bf16->float on load). Y[rows,B]=W[rows,cols]@X[cols,B].
 * Grid = (ceil(rows/8), ceil(B/8)); 32 threads/threadgroup. Handles ragged
 * rows/B via zero-pad. cols assumed %8==0 (all model dims are). */
"kernel void matmat_bf16(device const ushort *W [[buffer(0)]],\n"
"    device const float *X [[buffer(1)]], device float *Y [[buffer(2)]],\n"
"    constant uint &cols [[buffer(3)]], constant uint &B [[buffer(4)]],\n"
"    constant uint &rows [[buffer(5)]],\n"
"    uint2 tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_simdgroup]]) {\n"
"    threadgroup float Ws[64]; threadgroup float Xs[64]; threadgroup float Ys[64];\n"
"    uint row0 = tgid.x * 8, col0 = tgid.y * 8;\n"
"    simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float,8,8>(0.0f);\n"
"    for (uint k0 = 0; k0 < cols; k0 += 8) {\n"
"        for (uint i = tid; i < 64; i += 32) { uint r = i >> 3, k = i & 7; uint gr = row0 + r;\n"
"            Ws[i] = (gr < rows) ? bf16_to_f32(W[(ulong)gr * cols + k0 + k]) : 0.0f; }\n"
"        for (uint i = tid; i < 64; i += 32) { uint k = i >> 3, b = i & 7; uint gb = col0 + b;\n"
"            Xs[i] = (gb < B) ? X[(ulong)(k0 + k) * B + gb] : 0.0f; }\n"
"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"        simdgroup_float8x8 wm, xm; simdgroup_load(wm, Ws, 8); simdgroup_load(xm, Xs, 8);\n"
"        simdgroup_multiply_accumulate(acc, wm, xm, acc);\n"
"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    }\n"
"    simdgroup_store(acc, Ys, 8);\n"
"    for (uint i = tid; i < 64; i += 32) { uint r = i >> 3, b = i & 7; uint gr = row0 + r, gb = col0 + b;\n"
"        if (gr < rows && gb < B) Y[(ulong)gr * B + gb] = Ys[i]; }\n"
"}\n"
"\n"
"kernel void matvec_int8(device const char *W [[buffer(0)]],\n"
"    device const float *scale [[buffer(1)]], device const float *x [[buffer(2)]],\n"
"    device float *y [[buffer(3)]], constant uint &cols [[buffer(4)]], constant uint &rows [[buffer(5)]],\n"
"    uint3 tgpig [[threadgroup_position_in_grid]], uint tiisg [[thread_index_in_simdgroup]],\n"
"    uint sgitg [[simdgroup_index_in_threadgroup]], uint nsg [[simdgroups_per_threadgroup]]) {\n"
"    uint row = tgpig.x*nsg + sgitg; if (row >= rows) return;\n"
"    device const char4 *w4 = (device const char4 *)(W + (ulong)row * cols);\n"
"    device const float4 *x4 = (device const float4 *)x; uint n4 = cols >> 2; float acc = 0.0f;\n"
"    for (uint c = tiisg; c < n4; c += 32) { char4 wv = w4[c]; float4 xv = x4[c];\n"
"        acc += float(wv.x)*xv.x + float(wv.y)*xv.y + float(wv.z)*xv.z + float(wv.w)*xv.w; }\n"
"    acc = simd_sum(acc); if (tiisg == 0) y[row] = acc * scale[row];\n"
"}\n"
"\n"
"kernel void matvec_q4_0(device const q4blk *W [[buffer(0)]],\n"
"    device const float *x [[buffer(1)]], device float *y [[buffer(2)]],\n"
"    constant uint &cols [[buffer(3)]], constant uint &rows [[buffer(4)]],\n"
"    uint3 tgpig [[threadgroup_position_in_grid]], uint tiisg [[thread_index_in_simdgroup]],\n"
"    uint sgitg [[simdgroup_index_in_threadgroup]], uint nsg [[simdgroups_per_threadgroup]]) {\n"
"    uint row = tgpig.x*nsg + sgitg; if (row >= rows) return;\n"
"    uint bpr = cols / 32; device const q4blk *wr = W + (ulong)row * bpr; float acc = 0.0f;\n"
"    for (uint b = tiisg; b < bpr; b += 32) { float s = wr[b].scale;\n"
"        for (uint i = 0; i < 16; ++i) { uchar q = wr[b].qs[i];\n"
"            int lo = int(q & 0x0f) - 8; int hi = int(q >> 4) - 8;\n"
"            acc += float(lo) * s * x[b*32 + 2*i];\n"
"            acc += float(hi) * s * x[b*32 + 2*i + 1]; } }\n"
"    acc = simd_sum(acc); if (tiisg == 0) y[row] = acc;\n"
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
"kernel void rms_norm_batched(device const float *x [[buffer(0)]], device const float *w [[buffer(1)]],\n"
"    device float *y [[buffer(2)]], constant uint &dim [[buffer(3)]], constant uint &B [[buffer(4)]],\n"
"    constant float &eps [[buffer(5)]], uint b [[threadgroup_position_in_grid]],\n"
"    uint tid [[thread_position_in_threadgroup]], uint tc [[threads_per_threadgroup]]) {\n"
"    threadgroup float part[256]; float s = 0.0f;\n"
"    for (uint i = tid; i < dim; i += tc) { float v = x[(ulong)i*B + b]; s += v*v; }\n"
"    part[tid] = s; threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    for (uint st = tc/2; st > 0; st >>= 1) { if (tid < st) part[tid] += part[tid+st]; threadgroup_barrier(mem_flags::mem_threadgroup); }\n"
"    float inv = 1.0f / sqrt(part[0]/float(dim) + eps);\n"
"    for (uint i = tid; i < dim; i += tc) y[(ulong)i*B + b] = x[(ulong)i*B + b]*inv*w[i];\n"
"}\n"
"kernel void swiglu_batched(device const float *gu [[buffer(0)]], device float *h [[buffer(1)]],\n"
"    constant uint &inter [[buffer(2)]], constant uint &B [[buffer(3)]], uint gid [[thread_position_in_grid]]) {\n"
"    uint i = gid / B, b = gid % B; if (i >= inter) return;\n"
"    float g = gu[(ulong)(2*i)*B + b], u = gu[(ulong)(2*i+1)*B + b];\n"
"    h[(ulong)i*B + b] = g / (1.0f + exp(-g)) * u;\n"
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
"}\n"
"\n"
"kernel void snake(device float *data [[buffer(0)]], device const float *la [[buffer(1)]],\n"
"    device const float *lb [[buffer(2)]], constant uint &length [[buffer(3)]],\n"
"    uint2 gid [[thread_position_in_grid]]) {\n"
"    uint c = gid.y, t = gid.x; if (t >= length) return;\n"
"    float a = exp(la[c]), inv_b = exp(-lb[c]);\n"
"    ulong idx = (ulong)c * length + t; float x = data[idx]; float s = sin(a * x);\n"
"    data[idx] = x + inv_b * s * s;\n"
"}\n"
"\n"
/* Direct causal GQA attention: 1 thread per (query pos, head), online-softmax
 * over the causal window. Matches qwen_causal_attention. */
"kernel void attention(device const float *Q [[buffer(0)]], device const float *K [[buffer(1)]],\n"
"    device const float *V [[buffer(2)]], device float *O [[buffer(3)]],\n"
"    constant uint &seq_q [[buffer(4)]], constant uint &seq_k [[buffer(5)]],\n"
"    constant uint &n_heads [[buffer(6)]], constant uint &n_kv [[buffer(7)]],\n"
"    constant uint &head_dim [[buffer(8)]], constant float &scale [[buffer(9)]],\n"
"    constant uint &q_offset [[buffer(10)]], uint2 gid [[thread_position_in_grid]]) {\n"
"    uint sq = gid.x, h = gid.y; if (sq >= seq_q || h >= n_heads) return;\n"
"    uint kvh = h / (n_heads / n_kv); uint qpos = q_offset + sq;\n"
"    uint valid = qpos + 1; if (valid > seq_k) valid = seq_k;\n"
"    device const float *q = Q + ((ulong)sq * n_heads + h) * head_dim;\n"
"    device float *o = O + ((ulong)sq * n_heads + h) * head_dim;\n"
"    float m = -1e30f;\n"
"    for (uint j = 0; j < valid; ++j) { device const float *k = K + ((ulong)j * n_kv + kvh) * head_dim;\n"
"        float dot = 0.0f; for (uint d = 0; d < head_dim; ++d) dot += q[d]*k[d]; dot *= scale; if (dot > m) m = dot; }\n"
"    for (uint d = 0; d < head_dim; ++d) o[d] = 0.0f; float denom = 0.0f;\n"
"    for (uint j = 0; j < valid; ++j) { device const float *k = K + ((ulong)j * n_kv + kvh) * head_dim;\n"
"        float dot = 0.0f; for (uint d = 0; d < head_dim; ++d) dot += q[d]*k[d]; dot = exp(dot*scale - m); denom += dot;\n"
"        device const float *v = V + ((ulong)j * n_kv + kvh) * head_dim;\n"
"        for (uint d = 0; d < head_dim; ++d) o[d] += dot * v[d]; }\n"
"    float inv = 1.0f / denom; for (uint d = 0; d < head_dim; ++d) o[d] *= inv;\n"
"}\n"
"\n"
/* matmat with f32 weights (prefill GEMM), simdgroup_matrix MMA. */
"kernel void matmat_f32(device const float *W [[buffer(0)]], device const float *X [[buffer(1)]],\n"
"    device float *Y [[buffer(2)]], constant uint &cols [[buffer(3)]], constant uint &B [[buffer(4)]],\n"
"    constant uint &rows [[buffer(5)]],\n"
"    uint2 tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_simdgroup]]) {\n"
"    threadgroup float Ws[64]; threadgroup float Xs[64]; threadgroup float Ys[64];\n"
"    uint row0 = tgid.x * 8, col0 = tgid.y * 8;\n"
"    simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float,8,8>(0.0f);\n"
"    for (uint k0 = 0; k0 < cols; k0 += 8) {\n"
"        for (uint i = tid; i < 64; i += 32) { uint r = i>>3, k = i&7; uint gr = row0+r;\n"
"            Ws[i] = (gr < rows) ? W[(ulong)gr*cols + k0+k] : 0.0f; }\n"
"        for (uint i = tid; i < 64; i += 32) { uint k = i>>3, b = i&7; uint gb = col0+b;\n"
"            Xs[i] = (gb < B) ? X[(ulong)(k0+k)*B + gb] : 0.0f; }\n"
"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"        simdgroup_float8x8 wm, xm; simdgroup_load(wm, Ws, 8); simdgroup_load(xm, Xs, 8);\n"
"        simdgroup_multiply_accumulate(acc, wm, xm, acc);\n"
"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    }\n"
"    simdgroup_store(acc, Ys, 8);\n"
"    for (uint i = tid; i < 64; i += 32) { uint r = i>>3, b = i&7; uint gr = row0+r, gb = col0+b;\n"
"        if (gr < rows && gb < B) Y[(ulong)gr*B + gb] = Ys[i]; }\n"
"}\n"
"\n"
/* Causal conv1d, channel-first [ch,length]; bias always provided. */
"kernel void conv1d(device const float *in [[buffer(0)]], device const float *w [[buffer(1)]],\n"
"    device const float *bias [[buffer(2)]], device float *out [[buffer(3)]],\n"
"    constant uint &in_ch [[buffer(4)]], constant uint &length [[buffer(5)]],\n"
"    constant uint &ksz [[buffer(6)]], constant uint &dil [[buffer(7)]],\n"
"    uint2 gid [[thread_position_in_grid]]) {\n"
"    uint oc = gid.y, t = gid.x; if (t >= length) return; int pad = (int)(ksz-1) * (int)dil;\n"
"    float sum = bias[oc];\n"
"    for (uint ic = 0; ic < in_ch; ++ic) for (uint k = 0; k < ksz; ++k) {\n"
"        int ip = (int)t - pad + (int)k * (int)dil;\n"
"        if (ip >= 0 && ip < (int)length)\n"
"            sum += w[((ulong)oc*in_ch + ic)*ksz + k] * in[(ulong)ic*length + ip]; }\n"
"    out[(ulong)oc*length + t] = sum;\n"
"}\n"
"\n"
/* Causal conv-transpose1d, tap-solve gather (GPU-friendly upsampling). */
"kernel void conv_transpose1d(device const float *in [[buffer(0)]], device const float *w [[buffer(1)]],\n"
"    device const float *bias [[buffer(2)]], device float *out [[buffer(3)]],\n"
"    constant uint &in_ch [[buffer(4)]], constant uint &out_ch [[buffer(5)]],\n"
"    constant uint &in_len [[buffer(6)]], constant uint &out_len [[buffer(7)]],\n"
"    constant uint &ksz [[buffer(8)]], constant uint &stride [[buffer(9)]],\n"
"    uint2 gid [[thread_position_in_grid]]) {\n"
"    uint oc = gid.y, p = gid.x; if (p >= out_len) return;\n"
"    int full_len = (int)(in_len-1)*(int)stride + (int)ksz; int trim = (int)ksz - (int)stride;\n"
"    float sum = bias[oc];\n"
"    if ((int)p < full_len - trim) {\n"
"        for (uint k = 0; k < ksz; ++k) { int sh = (int)p - (int)k; if (sh < 0) continue;\n"
"            if ((uint)sh % stride != 0) continue; int tt = sh / (int)stride;\n"
"            if (tt < 0 || tt >= (int)in_len) continue;\n"
"            for (uint ic = 0; ic < in_ch; ++ic)\n"
"                sum += in[(ulong)ic*in_len + tt] * w[((ulong)ic*out_ch + oc)*ksz + k]; } }\n"
"    out[(ulong)oc*out_len + p] = sum;\n"
"}\n";

/* ---- context: device, queue, pipelines, resident weight cache, IO pool --- */
typedef struct { const void *key; void *buf; } wcache_ent;   /* buf = bridge-retained id<MTLBuffer> */

#define QWEN_MTL_IO_SLOTS 6
typedef struct {
    void *device, *queue;
    /* pipelines */
    void *pso_matvec_bf16, *pso_matmat_bf16, *pso_matvec_int8, *pso_matvec_q4_0;
    void *pso_rms, *pso_swiglu, *pso_silu, *pso_add, *pso_mul, *pso_scale, *pso_rope;
    void *pso_snake, *pso_attn, *pso_matmat_f32, *pso_conv1d, *pso_convt1d;
    void *pso_rms_b, *pso_swiglu_b;
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
        c->pso_snake  = make_pso(dev, lib, "snake");
        c->pso_attn   = make_pso(dev, lib, "attention");
        c->pso_matmat_f32 = make_pso(dev, lib, "matmat_f32");
        c->pso_conv1d = make_pso(dev, lib, "conv1d");
        c->pso_convt1d = make_pso(dev, lib, "conv_transpose1d");
        c->pso_rms_b = make_pso(dev, lib, "rms_norm_batched");
        c->pso_swiglu_b = make_pso(dev, lib, "swiglu_batched");
        if (!c->pso_matvec_bf16 || !c->pso_matmat_bf16 || !c->pso_matvec_int8 ||
            !c->pso_matvec_q4_0 || !c->pso_rms || !c->pso_swiglu || !c->pso_silu ||
            !c->pso_add || !c->pso_mul || !c->pso_scale || !c->pso_rope ||
            !c->pso_snake || !c->pso_attn || !c->pso_matmat_f32 || !c->pso_conv1d || !c->pso_convt1d ||
            !c->pso_rms_b || !c->pso_swiglu_b) {
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
    release_ptr(&c->pso_snake); release_ptr(&c->pso_attn); release_ptr(&c->pso_matmat_f32);
    release_ptr(&c->pso_conv1d); release_ptr(&c->pso_convt1d);
    release_ptr(&c->pso_rms_b); release_ptr(&c->pso_swiglu_b);
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

/* Drop all cached resident weights. Real engine: never needed (mmap pointers are
 * stable). Selftest only: it frees+reallocs weight arrays, so a reused malloc
 * address must not return a stale cached device buffer. */
static void flush_weights(qwen_metal_ctx *c) {
    for (int i = 0; i < c->wc_n; ++i) { void *b = c->wc[i].buf; release_ptr(&b); }
    c->wc_n = 0;
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
        uint32_t cc = (uint32_t)cols, rr = (uint32_t)rows;
        [e setBytes:&cc length:4 atIndex:3]; [e setBytes:&rr length:4 atIndex:4];
        const NSUInteger NSG = 8;   /* simdgroups per threadgroup → 256 threads */
        NSUInteger ntg = ((NSUInteger)rows + NSG - 1) / NSG;
        [e dispatchThreadgroups:MTLSizeMake(ntg,1,1)
        threadsPerThreadgroup:MTLSizeMake(NSG * 32,1,1)];
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
        uint32_t cc = (uint32_t)cols, cB = (uint32_t)B, rr = (uint32_t)rows;
        [e setBytes:&cc length:4 atIndex:3]; [e setBytes:&cB length:4 atIndex:4];
        [e setBytes:&rr length:4 atIndex:5];
        /* one simdgroup (32 threads) per 8x8 output tile */
        [e dispatchThreadgroups:MTLSizeMake(((NSUInteger)rows+7)/8, ((NSUInteger)B+7)/8, 1)
        threadsPerThreadgroup:MTLSizeMake(32,1,1)];
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
        uint32_t cc = (uint32_t)cols, rr = (uint32_t)rows;
        [e setBytes:&cc length:4 atIndex:4]; [e setBytes:&rr length:4 atIndex:5];
        [e dispatchThreadgroups:MTLSizeMake(((NSUInteger)rows+7)/8,1,1)
        threadsPerThreadgroup:MTLSizeMake(8*32,1,1)];
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
        uint32_t cc = (uint32_t)cols, rr = (uint32_t)rows;
        [e setBytes:&cc length:4 atIndex:3]; [e setBytes:&rr length:4 atIndex:4];
        [e dispatchThreadgroups:MTLSizeMake(((NSUInteger)rows+7)/8,1,1)
        threadsPerThreadgroup:MTLSizeMake(8*32,1,1)];
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

void qwen_metal_snake(void *ctx, float *data, const float *log_alpha,
                      const float *log_beta, int channels, int length) {
    @autoreleasepool {
        qwen_metal_ctx *c = ctx;
        id<MTLCommandQueue> q = (__bridge id<MTLCommandQueue>)c->queue;
        id<MTLComputePipelineState> pso = (__bridge id<MTLComputePipelineState>)c->pso_snake;
        id<MTLBuffer> bd = io_buf(c, 0, (size_t)channels*length*sizeof(float));
        id<MTLBuffer> ba = io_buf(c, 1, (size_t)channels*sizeof(float));
        id<MTLBuffer> bb = io_buf(c, 2, (size_t)channels*sizeof(float));
        memcpy(bd.contents, data, (size_t)channels*length*sizeof(float));
        memcpy(ba.contents, log_alpha, (size_t)channels*sizeof(float));
        memcpy(bb.contents, log_beta, (size_t)channels*sizeof(float));
        id<MTLCommandBuffer> cb = [q commandBuffer]; id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        [e setComputePipelineState:pso];
        [e setBuffer:bd offset:0 atIndex:0]; [e setBuffer:ba offset:0 atIndex:1]; [e setBuffer:bb offset:0 atIndex:2];
        uint32_t L = (uint32_t)length; [e setBytes:&L length:4 atIndex:3];
        [e dispatchThreads:MTLSizeMake((NSUInteger)length,(NSUInteger)channels,1)
        threadsPerThreadgroup:MTLSizeMake(cap256(pso),1,1)];
        [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
        memcpy(data, bd.contents, (size_t)channels*length*sizeof(float));
    }
}

void qwen_metal_attention(void *ctx, float *O, const float *Q, const float *K, const float *V,
                          int seq_q, int seq_k, int n_heads, int n_kv, int head_dim,
                          float scale, int q_offset) {
    @autoreleasepool {
        qwen_metal_ctx *c = ctx;
        id<MTLCommandQueue> q = (__bridge id<MTLCommandQueue>)c->queue;
        id<MTLComputePipelineState> pso = (__bridge id<MTLComputePipelineState>)c->pso_attn;
        size_t qn = (size_t)seq_q*n_heads*head_dim, kn = (size_t)seq_k*n_kv*head_dim;
        id<MTLBuffer> bQ = io_buf(c, 0, qn*sizeof(float));
        id<MTLBuffer> bK = io_buf(c, 1, kn*sizeof(float));
        id<MTLBuffer> bV = io_buf(c, 2, kn*sizeof(float));
        id<MTLBuffer> bO = io_buf(c, 3, qn*sizeof(float));
        memcpy(bQ.contents, Q, qn*sizeof(float)); memcpy(bK.contents, K, kn*sizeof(float));
        memcpy(bV.contents, V, kn*sizeof(float));
        id<MTLCommandBuffer> cb = [q commandBuffer]; id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        [e setComputePipelineState:pso];
        [e setBuffer:bQ offset:0 atIndex:0]; [e setBuffer:bK offset:0 atIndex:1];
        [e setBuffer:bV offset:0 atIndex:2]; [e setBuffer:bO offset:0 atIndex:3];
        uint32_t sq=(uint32_t)seq_q, sk=(uint32_t)seq_k, nh=(uint32_t)n_heads, nkv=(uint32_t)n_kv,
                 hd=(uint32_t)head_dim, qo=(uint32_t)q_offset; float sc=scale;
        [e setBytes:&sq length:4 atIndex:4]; [e setBytes:&sk length:4 atIndex:5];
        [e setBytes:&nh length:4 atIndex:6]; [e setBytes:&nkv length:4 atIndex:7];
        [e setBytes:&hd length:4 atIndex:8]; [e setBytes:&sc length:4 atIndex:9];
        [e setBytes:&qo length:4 atIndex:10];
        [e dispatchThreads:MTLSizeMake((NSUInteger)seq_q,(NSUInteger)n_heads,1)
        threadsPerThreadgroup:MTLSizeMake(1,1,1)];
        [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
        memcpy(O, bO.contents, qn*sizeof(float));
    }
}

void qwen_metal_matmat_f32(void *ctx, float *Y, const float *W, const float *X,
                           int rows, int cols, int B) {
    @autoreleasepool {
        qwen_metal_ctx *c = ctx;
        id<MTLCommandQueue> q = (__bridge id<MTLCommandQueue>)c->queue;
        id<MTLComputePipelineState> pso = (__bridge id<MTLComputePipelineState>)c->pso_matmat_f32;
        id<MTLBuffer> bW = weight_buf(c, W, (size_t)rows*cols*sizeof(float));
        id<MTLBuffer> bX = io_buf(c, 0, (size_t)cols*B*sizeof(float));
        id<MTLBuffer> bY = io_buf(c, 1, (size_t)rows*B*sizeof(float));
        memcpy(bX.contents, X, (size_t)cols*B*sizeof(float));
        id<MTLCommandBuffer> cb = [q commandBuffer]; id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        [e setComputePipelineState:pso];
        [e setBuffer:bW offset:0 atIndex:0]; [e setBuffer:bX offset:0 atIndex:1]; [e setBuffer:bY offset:0 atIndex:2];
        uint32_t cc=(uint32_t)cols, cB=(uint32_t)B, rr=(uint32_t)rows;
        [e setBytes:&cc length:4 atIndex:3]; [e setBytes:&cB length:4 atIndex:4]; [e setBytes:&rr length:4 atIndex:5];
        [e dispatchThreadgroups:MTLSizeMake(((NSUInteger)rows+7)/8,((NSUInteger)B+7)/8,1)
        threadsPerThreadgroup:MTLSizeMake(32,1,1)];
        [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
        memcpy(Y, bY.contents, (size_t)rows*B*sizeof(float));
    }
}

void qwen_metal_conv1d(void *ctx, float *out, const float *in, const float *weight,
                       const float *bias, int in_ch, int out_ch, int length, int ksz, int dilation) {
    @autoreleasepool {
        qwen_metal_ctx *c = ctx;
        id<MTLCommandQueue> q = (__bridge id<MTLCommandQueue>)c->queue;
        id<MTLComputePipelineState> pso = (__bridge id<MTLComputePipelineState>)c->pso_conv1d;
        id<MTLBuffer> bin = io_buf(c, 0, (size_t)in_ch*length*sizeof(float));
        id<MTLBuffer> bout= io_buf(c, 1, (size_t)out_ch*length*sizeof(float));
        id<MTLBuffer> bw = weight_buf(c, weight, (size_t)out_ch*in_ch*ksz*sizeof(float));
        id<MTLBuffer> bb = weight_buf(c, bias, (size_t)out_ch*sizeof(float));
        memcpy(bin.contents, in, (size_t)in_ch*length*sizeof(float));
        id<MTLCommandBuffer> cb = [q commandBuffer]; id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        [e setComputePipelineState:pso];
        [e setBuffer:bin offset:0 atIndex:0]; [e setBuffer:bw offset:0 atIndex:1];
        [e setBuffer:bb offset:0 atIndex:2]; [e setBuffer:bout offset:0 atIndex:3];
        uint32_t ic=(uint32_t)in_ch, L=(uint32_t)length, ks=(uint32_t)ksz, di=(uint32_t)dilation;
        [e setBytes:&ic length:4 atIndex:4]; [e setBytes:&L length:4 atIndex:5];
        [e setBytes:&ks length:4 atIndex:6]; [e setBytes:&di length:4 atIndex:7];
        [e dispatchThreads:MTLSizeMake((NSUInteger)length,(NSUInteger)out_ch,1)
        threadsPerThreadgroup:MTLSizeMake(cap256(pso),1,1)];
        [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
        memcpy(out, bout.contents, (size_t)out_ch*length*sizeof(float));
    }
}

void qwen_metal_conv_transpose1d(void *ctx, float *out, const float *in, const float *weight,
                                 const float *bias, int in_ch, int out_ch, int in_len, int out_len,
                                 int ksz, int stride) {
    @autoreleasepool {
        qwen_metal_ctx *c = ctx;
        id<MTLCommandQueue> q = (__bridge id<MTLCommandQueue>)c->queue;
        id<MTLComputePipelineState> pso = (__bridge id<MTLComputePipelineState>)c->pso_convt1d;
        id<MTLBuffer> bin = io_buf(c, 0, (size_t)in_ch*in_len*sizeof(float));
        id<MTLBuffer> bout= io_buf(c, 1, (size_t)out_ch*out_len*sizeof(float));
        id<MTLBuffer> bw = weight_buf(c, weight, (size_t)in_ch*out_ch*ksz*sizeof(float));
        id<MTLBuffer> bb = weight_buf(c, bias, (size_t)out_ch*sizeof(float));
        memcpy(bin.contents, in, (size_t)in_ch*in_len*sizeof(float));
        id<MTLCommandBuffer> cb = [q commandBuffer]; id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        [e setComputePipelineState:pso];
        [e setBuffer:bin offset:0 atIndex:0]; [e setBuffer:bw offset:0 atIndex:1];
        [e setBuffer:bb offset:0 atIndex:2]; [e setBuffer:bout offset:0 atIndex:3];
        uint32_t ic=(uint32_t)in_ch, oc=(uint32_t)out_ch, il=(uint32_t)in_len, ol=(uint32_t)out_len,
                 ks=(uint32_t)ksz, st=(uint32_t)stride;
        [e setBytes:&ic length:4 atIndex:4]; [e setBytes:&oc length:4 atIndex:5];
        [e setBytes:&il length:4 atIndex:6]; [e setBytes:&ol length:4 atIndex:7];
        [e setBytes:&ks length:4 atIndex:8]; [e setBytes:&st length:4 atIndex:9];
        [e dispatchThreads:MTLSizeMake((NSUInteger)out_len,(NSUInteger)out_ch,1)
        threadsPerThreadgroup:MTLSizeMake(cap256(pso),1,1)];
        [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
        memcpy(out, bout.contents, (size_t)out_ch*out_len*sizeof(float));
    }
}

/* ---- FUSED RESIDENT FFN: the heavy block, entirely on GPU -----------------
 * rms_norm → gate_up matvec → SwiGLU → down matvec → residual, encoded as ONE
 * command buffer. All intermediates (xn, gate_up, h) stay in DEVICE buffers —
 * never copied to the CPU. On-GPU memoryBarriers order the dependent dispatches;
 * a single commit+wait at the end. Only `out` comes back. This is the resident-
 * decode pattern (llama.cpp/mlx): the win comes from the heavy matmuls running
 * back-to-back on the GPU with zero per-op CPU<->GPU round-trips.
 *
 * Layouts (match the CPU path): gate_up [2*inter, H] interleaved rows
 * (row 2i=gate_i, 2i+1=up_i); down [H, inter]; residual out += x. */
void qwen_metal_ffn_swiglu(void *ctx, float *out, const float *x, const float *norm_w,
                           const uint16_t *Wgu, const uint16_t *Wd,
                           int H, int inter, float eps) {
    @autoreleasepool {
        qwen_metal_ctx *c = ctx;
        id<MTLCommandQueue> q = (__bridge id<MTLCommandQueue>)c->queue;
        id<MTLComputePipelineState> rms = (__bridge id<MTLComputePipelineState>)c->pso_rms;
        id<MTLComputePipelineState> mv  = (__bridge id<MTLComputePipelineState>)c->pso_matvec_bf16;
        id<MTLComputePipelineState> sg  = (__bridge id<MTLComputePipelineState>)c->pso_swiglu;
        id<MTLComputePipelineState> add = (__bridge id<MTLComputePipelineState>)c->pso_add;

        id<MTLBuffer> bWgu  = weight_buf(c, Wgu,   (size_t)(2*inter) * H * sizeof(uint16_t));
        id<MTLBuffer> bWd   = weight_buf(c, Wd,    (size_t)H * inter * sizeof(uint16_t));
        id<MTLBuffer> bnorm = weight_buf(c, norm_w,(size_t)H * sizeof(float));
        id<MTLBuffer> bx  = io_buf(c, 0, (size_t)H * sizeof(float));
        id<MTLBuffer> bxn = io_buf(c, 1, (size_t)H * sizeof(float));
        id<MTLBuffer> bgu = io_buf(c, 2, (size_t)(2*inter) * sizeof(float));
        id<MTLBuffer> bh  = io_buf(c, 3, (size_t)inter * sizeof(float));
        id<MTLBuffer> bout= io_buf(c, 4, (size_t)H * sizeof(float));
        memcpy(bx.contents, x, (size_t)H * sizeof(float));

        const NSUInteger NSG = 8;
        uint32_t uH = (uint32_t)H, uInter = (uint32_t)inter, uGU = (uint32_t)(2*inter);
        float ep = eps;

        id<MTLCommandBuffer> cb = [q commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];

        /* 1) rms_norm(x) -> xn */
        [e setComputePipelineState:rms];
        [e setBuffer:bx offset:0 atIndex:0]; [e setBuffer:bnorm offset:0 atIndex:1];
        [e setBuffer:bxn offset:0 atIndex:2];
        [e setBytes:&uH length:4 atIndex:3]; [e setBytes:&ep length:4 atIndex:4];
        [e dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        [e memoryBarrierWithScope:MTLBarrierScopeBuffers];

        /* 2) gate_up = Wgu @ xn  (rows=2*inter, cols=H) */
        [e setComputePipelineState:mv];
        [e setBuffer:bWgu offset:0 atIndex:0]; [e setBuffer:bxn offset:0 atIndex:1];
        [e setBuffer:bgu offset:0 atIndex:2];
        [e setBytes:&uH length:4 atIndex:3]; [e setBytes:&uGU length:4 atIndex:4];
        [e dispatchThreadgroups:MTLSizeMake(((NSUInteger)(2*inter)+NSG-1)/NSG,1,1)
        threadsPerThreadgroup:MTLSizeMake(NSG*32,1,1)];
        [e memoryBarrierWithScope:MTLBarrierScopeBuffers];

        /* 3) SwiGLU: gate_up[2*inter] -> h[inter] */
        [e setComputePipelineState:sg];
        [e setBuffer:bgu offset:0 atIndex:0]; [e setBuffer:bh offset:0 atIndex:1];
        [e dispatchThreads:MTLSizeMake((NSUInteger)inter,1,1)
        threadsPerThreadgroup:MTLSizeMake(cap256(sg),1,1)];
        [e memoryBarrierWithScope:MTLBarrierScopeBuffers];

        /* 4) down = Wd @ h  (rows=H, cols=inter) */
        [e setComputePipelineState:mv];
        [e setBuffer:bWd offset:0 atIndex:0]; [e setBuffer:bh offset:0 atIndex:1];
        [e setBuffer:bout offset:0 atIndex:2];
        [e setBytes:&uInter length:4 atIndex:3]; [e setBytes:&uH length:4 atIndex:4];
        [e dispatchThreadgroups:MTLSizeMake(((NSUInteger)H+NSG-1)/NSG,1,1)
        threadsPerThreadgroup:MTLSizeMake(NSG*32,1,1)];
        [e memoryBarrierWithScope:MTLBarrierScopeBuffers];

        /* 5) residual: out += x */
        [e setComputePipelineState:add];
        [e setBuffer:bout offset:0 atIndex:0]; [e setBuffer:bx offset:0 atIndex:1];
        [e setBuffer:bout offset:0 atIndex:2];
        [e dispatchThreads:MTLSizeMake((NSUInteger)H,1,1)
        threadsPerThreadgroup:MTLSizeMake(cap256(add),1,1)];

        [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
        memcpy(out, bout.contents, (size_t)H * sizeof(float));
    }
}

/* BATCHED fused FFN: B tokens, [dim,B]-native; gate_up + down are MMA matmats
 * (the 6.96x compute-bound win), rms/swiglu batched, residual — ONE command
 * buffer, activations resident. This is #3: the FFN heavy matmuls at MMA speed. */
void qwen_metal_ffn_swiglu_batched(void *ctx, float *out, const float *x, const float *norm_w,
                                   const uint16_t *Wgu, const uint16_t *Wd,
                                   int H, int inter, int B, float eps) {
    @autoreleasepool {
        qwen_metal_ctx *c = ctx;
        id<MTLCommandQueue> q = (__bridge id<MTLCommandQueue>)c->queue;
        id<MTLComputePipelineState> rmsb = (__bridge id<MTLComputePipelineState>)c->pso_rms_b;
        id<MTLComputePipelineState> mm  = (__bridge id<MTLComputePipelineState>)c->pso_matmat_bf16;
        id<MTLComputePipelineState> sgb = (__bridge id<MTLComputePipelineState>)c->pso_swiglu_b;
        id<MTLComputePipelineState> add = (__bridge id<MTLComputePipelineState>)c->pso_add;
        id<MTLBuffer> bWgu  = weight_buf(c, Wgu,   (size_t)(2*inter)*H*sizeof(uint16_t));
        id<MTLBuffer> bWd   = weight_buf(c, Wd,    (size_t)H*inter*sizeof(uint16_t));
        id<MTLBuffer> bnorm = weight_buf(c, norm_w,(size_t)H*sizeof(float));
        id<MTLBuffer> bx  = io_buf(c, 0, (size_t)H*B*sizeof(float));
        id<MTLBuffer> bxn = io_buf(c, 1, (size_t)H*B*sizeof(float));
        id<MTLBuffer> bgu = io_buf(c, 2, (size_t)(2*inter)*B*sizeof(float));
        id<MTLBuffer> bh  = io_buf(c, 3, (size_t)inter*B*sizeof(float));
        id<MTLBuffer> bout= io_buf(c, 4, (size_t)H*B*sizeof(float));
        memcpy(bx.contents, x, (size_t)H*B*sizeof(float));
        uint32_t uH=(uint32_t)H, uI=(uint32_t)inter, uB=(uint32_t)B, uGU=(uint32_t)(2*inter); float ep=eps;

        id<MTLCommandBuffer> cb = [q commandBuffer]; id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        /* 1) rms_norm per token */
        [e setComputePipelineState:rmsb];
        [e setBuffer:bx offset:0 atIndex:0]; [e setBuffer:bnorm offset:0 atIndex:1]; [e setBuffer:bxn offset:0 atIndex:2];
        [e setBytes:&uH length:4 atIndex:3]; [e setBytes:&uB length:4 atIndex:4]; [e setBytes:&ep length:4 atIndex:5];
        [e dispatchThreadgroups:MTLSizeMake((NSUInteger)B,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        [e memoryBarrierWithScope:MTLBarrierScopeBuffers];
        /* 2) gate_up = Wgu @ xn  (MMA) */
        [e setComputePipelineState:mm];
        [e setBuffer:bWgu offset:0 atIndex:0]; [e setBuffer:bxn offset:0 atIndex:1]; [e setBuffer:bgu offset:0 atIndex:2];
        [e setBytes:&uH length:4 atIndex:3]; [e setBytes:&uB length:4 atIndex:4]; [e setBytes:&uGU length:4 atIndex:5];
        [e dispatchThreadgroups:MTLSizeMake(((NSUInteger)(2*inter)+7)/8,((NSUInteger)B+7)/8,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
        [e memoryBarrierWithScope:MTLBarrierScopeBuffers];
        /* 3) swiglu batched */
        [e setComputePipelineState:sgb];
        [e setBuffer:bgu offset:0 atIndex:0]; [e setBuffer:bh offset:0 atIndex:1];
        [e setBytes:&uI length:4 atIndex:2]; [e setBytes:&uB length:4 atIndex:3];
        [e dispatchThreads:MTLSizeMake((NSUInteger)inter*B,1,1) threadsPerThreadgroup:MTLSizeMake(cap256(sgb),1,1)];
        [e memoryBarrierWithScope:MTLBarrierScopeBuffers];
        /* 4) down = Wd @ h  (MMA) */
        [e setComputePipelineState:mm];
        [e setBuffer:bWd offset:0 atIndex:0]; [e setBuffer:bh offset:0 atIndex:1]; [e setBuffer:bout offset:0 atIndex:2];
        [e setBytes:&uI length:4 atIndex:3]; [e setBytes:&uB length:4 atIndex:4]; [e setBytes:&uH length:4 atIndex:5];
        [e dispatchThreadgroups:MTLSizeMake(((NSUInteger)H+7)/8,((NSUInteger)B+7)/8,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
        [e memoryBarrierWithScope:MTLBarrierScopeBuffers];
        /* 5) residual */
        [e setComputePipelineState:add];
        [e setBuffer:bout offset:0 atIndex:0]; [e setBuffer:bx offset:0 atIndex:1]; [e setBuffer:bout offset:0 atIndex:2];
        [e dispatchThreads:MTLSizeMake((NSUInteger)H*B,1,1) threadsPerThreadgroup:MTLSizeMake(cap256(add),1,1)];
        [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
        memcpy(out, bout.contents, (size_t)H*B*sizeof(float));
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

/* Amortized matvec cost when K dispatches share ONE command buffer + one
 * commit/wait (resident weight + IO) — isolates kernel throughput from the
 * per-op CPU<->GPU round-trip that dominates the naive one-op-per-commit path.
 * Returns ms/op. This is the regime a real fused decode step runs in. */
double qwen_metal_matvec_bench_fused(void *ctx, const uint16_t *W, const float *x,
                                     int rows, int cols, int reps) {
    @autoreleasepool {
        qwen_metal_ctx *c = ctx;
        id<MTLCommandQueue> q = (__bridge id<MTLCommandQueue>)c->queue;
        id<MTLComputePipelineState> pso = (__bridge id<MTLComputePipelineState>)c->pso_matvec_bf16;
        id<MTLBuffer> bW = weight_buf(c, W, (size_t)rows * cols * sizeof(uint16_t));
        id<MTLBuffer> bx = io_buf(c, 0, (size_t)cols * sizeof(float));
        id<MTLBuffer> by = io_buf(c, 1, (size_t)rows * sizeof(float));
        memcpy(bx.contents, x, (size_t)cols * sizeof(float));
        uint32_t cc = (uint32_t)cols, rr = (uint32_t)rows;
        const NSUInteger NSG = 8; NSUInteger ntg = ((NSUInteger)rows + NSG - 1) / NSG;
        double t0 = nowms();
        id<MTLCommandBuffer> cb = [q commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        [e setComputePipelineState:pso];
        [e setBuffer:bW offset:0 atIndex:0]; [e setBuffer:bx offset:0 atIndex:1];
        [e setBuffer:by offset:0 atIndex:2];
        [e setBytes:&cc length:4 atIndex:3]; [e setBytes:&rr length:4 atIndex:4];
        for (int r = 0; r < reps; ++r)
            [e dispatchThreadgroups:MTLSizeMake(ntg,1,1)
            threadsPerThreadgroup:MTLSizeMake(NSG * 32,1,1)];
        [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
        return (nowms() - t0) / reps;
    }
}

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

    /* ---- snake ---- */
    { int ch = 64, L = 256; float *d1 = malloc((size_t)ch*L*4), *d2 = malloc((size_t)ch*L*4);
      float *la = malloc((size_t)ch*4), *lb = malloc((size_t)ch*4);
      for (int i=0;i<ch*L;++i){ float v=rnd(); d1[i]=v; d2[i]=v; }
      for (int i=0;i<ch;++i){ la[i]=rnd()*0.3f; lb[i]=rnd()*0.3f; }
      qwen_snake_activation(d1, ch, L, la, lb);
      qwen_metal_snake(ctx, d2, la, lb, ch, L);
      fails += check(f, "snake", d2, d1, ch*L, 1e-3); free(d1);free(d2);free(la);free(lb); }

    /* ---- attention (causal GQA) ---- */
    { int sq=4, sk=16, nh=8, nkv=2, hd=64, qoff=12; float sc=1.0f/sqrtf((float)hd);
      float *Q=malloc((size_t)sq*nh*hd*4), *Kk=malloc((size_t)sk*nkv*hd*4), *Vv=malloc((size_t)sk*nkv*hd*4);
      float *oc=malloc((size_t)sq*nh*hd*4), *og=malloc((size_t)sq*nh*hd*4);
      for (int i=0;i<sq*nh*hd;++i) Q[i]=rnd(); for (int i=0;i<sk*nkv*hd;++i){ Kk[i]=rnd(); Vv[i]=rnd(); }
      qwen_causal_attention(oc, Q, Kk, Vv, sq, sk, nh, nkv, hd, sc, qoff);
      qwen_metal_attention(ctx, og, Q, Kk, Vv, sq, sk, nh, nkv, hd, sc, qoff);
      fails += check(f, "attention", og, oc, sq*nh*hd, 1e-3); free(Q);free(Kk);free(Vv);free(oc);free(og); }

    /* ---- matmat_f32 (prefill GEMM) ---- */
    flush_weights((qwen_metal_ctx *)ctx);
    { int r=512, cc=512, b2=16; float *Wf=malloc((size_t)r*cc*4), *Xf=malloc((size_t)cc*b2*4);
      float *yc=malloc((size_t)r*b2*4), *yg=malloc((size_t)r*b2*4);
      for (int i=0;i<r*cc;++i) Wf[i]=rnd()*0.1f; for (int i=0;i<cc*b2;++i) Xf[i]=rnd();
      for (int i=0;i<r;++i) for (int j=0;j<b2;++j){ float s=0; for(int k=0;k<cc;++k) s+=Wf[(size_t)i*cc+k]*Xf[(size_t)k*b2+j]; yc[(size_t)i*b2+j]=s; }
      qwen_metal_matmat_f32(ctx, yg, Wf, Xf, r, cc, b2);
      fails += check(f, "matmat_f32", yg, yc, r*b2, 1e-3); free(Wf);free(Xf);free(yc);free(yg); }

    /* ---- conv1d + conv_transpose1d (local CPU ref matches the decoder naive) ---- */
    flush_weights((qwen_metal_ctx *)ctx);
    { int ic=8, oc=8, L=64, ks=7, dil=1; int pad=(ks-1)*dil;
      float *in=malloc((size_t)ic*L*4), *w=malloc((size_t)oc*ic*ks*4), *bi=malloc((size_t)oc*4);
      float *cc=malloc((size_t)oc*L*4), *gg=malloc((size_t)oc*L*4);
      for (int i=0;i<ic*L;++i) in[i]=rnd(); for (int i=0;i<oc*ic*ks;++i) w[i]=rnd()*0.2f; for (int i=0;i<oc;++i) bi[i]=rnd()*0.1f;
      for (int o=0;o<oc;++o) for (int t=0;t<L;++t){ float s=bi[o];
          for (int i=0;i<ic;++i) for (int k=0;k<ks;++k){ int ip=t-pad+k*dil; if(ip>=0&&ip<L) s+=w[((size_t)o*ic+i)*ks+k]*in[(size_t)i*L+ip]; }
          cc[(size_t)o*L+t]=s; }
      qwen_metal_conv1d(ctx, gg, in, w, bi, ic, oc, L, ks, dil);
      fails += check(f, "conv1d", gg, cc, oc*L, 1e-3); free(in);free(w);free(bi);free(cc);free(gg); }
    { int ic=8, oc=8, il=32, ks=4, st=2, ol=il*st; int full=(il-1)*st+ks, trim=ks-st;
      float *in=malloc((size_t)ic*il*4), *w=malloc((size_t)ic*oc*ks*4), *bi=malloc((size_t)oc*4);
      float *cc=malloc((size_t)oc*ol*4), *gg=malloc((size_t)oc*ol*4);
      for (int i=0;i<ic*il;++i) in[i]=rnd(); for (int i=0;i<ic*oc*ks;++i) w[i]=rnd()*0.2f; for (int i=0;i<oc;++i) bi[i]=rnd()*0.1f;
      for (int o=0;o<oc;++o) for (int p=0;p<ol;++p){ float s=bi[o];
          if (p<full-trim) for (int k=0;k<ks;++k){ int sh=p-k; if(sh<0||sh%st) continue; int tt=sh/st; if(tt<0||tt>=il) continue;
              for (int i=0;i<ic;++i) s+=in[(size_t)i*il+tt]*w[((size_t)i*oc+o)*ks+k]; }
          cc[(size_t)o*ol+p]=s; }
      flush_weights((qwen_metal_ctx *)ctx);
      qwen_metal_conv_transpose1d(ctx, gg, in, w, bi, ic, oc, il, ol, ks, st);
      fails += check(f, "conv_transpose1d", gg, cc, oc*ol, 1e-3); free(in);free(w);free(bi);free(cc);free(gg); }

    /* ---- FUSED RESIDENT FFN (the heavy block, 1.7B Talker sizes) ---- */
    flush_weights((qwen_metal_ctx *)ctx);
    {
        int H = 2048, inter = 6144;
        uint16_t *Wgu = malloc((size_t)(2*inter)*H*2);
        uint16_t *Wd  = malloc((size_t)H*inter*2);
        float *nw = malloc((size_t)H*4), *xf = malloc((size_t)H*4);
        float *ocpu = malloc((size_t)H*4), *ogpu = malloc((size_t)H*4);
        float *xn = malloc((size_t)H*4), *guf = malloc((size_t)(2*inter)*4), *tmp = malloc((size_t)inter*4);
        for (size_t i=0;i<(size_t)(2*inter)*H;++i) Wgu[i]=to_bf16(rnd()*0.05f);
        for (size_t i=0;i<(size_t)H*inter;++i) Wd[i]=to_bf16(rnd()*0.05f);
        for (int i=0;i<H;++i){ nw[i]=rnd()*0.3f+1.0f; xf[i]=rnd(); }
        qwen_rms_norm(xn, xf, nw, 1, H, 1e-6f);
        qwen_matvec_bf16(guf, Wgu, xn, 2*inter, H);
        qwen_swiglu_inplace(guf, tmp, inter);
        qwen_matvec_bf16(ocpu, Wd, guf, H, inter);
        qwen_add_inplace(ocpu, xf, H);
        qwen_metal_ffn_swiglu(ctx, ogpu, xf, nw, Wgu, Wd, H, inter, 1e-6f);
        fails += check(f, "ffn_fused", ogpu, ocpu, H, 5e-3);
        int itf = 30; double tf0 = nowms();
        for (int k=0;k<itf;++k){ qwen_rms_norm(xn,xf,nw,1,H,1e-6f); qwen_matvec_bf16(guf,Wgu,xn,2*inter,H);
            qwen_swiglu_inplace(guf,tmp,inter); qwen_matvec_bf16(ocpu,Wd,guf,H,inter); qwen_add_inplace(ocpu,xf,H); }
        double tcf = (nowms()-tf0)/itf;
        tf0 = nowms(); for (int k=0;k<itf;++k) qwen_metal_ffn_swiglu(ctx,ogpu,xf,nw,Wgu,Wd,H,inter,1e-6f);
        double tgf = (nowms()-tf0)/itf;
        fprintf(f, "  FFN FUSED (H=2048 inter=6144, 1 cmdbuf): cpu=%.3f ms  metal=%.3f ms  (%.2fx)\n",
                tcf, tgf, tgf>0?tcf/tgf:0);
        free(Wgu); free(Wd); free(nw); free(xf); free(ocpu); free(ogpu); free(xn); free(guf); free(tmp);
    }

    /* ---- BATCHED fused FFN (B=16, MMA matmuls) ---- */
    flush_weights((qwen_metal_ctx *)ctx);
    {
        int H = 2048, inter = 6144, B = 16;
        uint16_t *Wgu = malloc((size_t)(2*inter)*H*2), *Wd = malloc((size_t)H*inter*2);
        float *nw = malloc((size_t)H*4), *xb = malloc((size_t)H*B*4);
        float *ocpu = malloc((size_t)H*B*4), *ogpu = malloc((size_t)H*B*4);
        float *xg = malloc((size_t)H*4), *xn = malloc((size_t)H*4), *guf = malloc((size_t)(2*inter)*4), *tmp = malloc((size_t)inter*4), *ob = malloc((size_t)H*4);
        for (size_t i=0;i<(size_t)(2*inter)*H;++i) Wgu[i]=to_bf16(rnd()*0.05f);
        for (size_t i=0;i<(size_t)H*inter;++i) Wd[i]=to_bf16(rnd()*0.05f);
        for (int i=0;i<H;++i) nw[i]=rnd()*0.3f+1.0f; for (int i=0;i<H*B;++i) xb[i]=rnd();
        /* CPU ref: per token (gather [dim,B] column → contiguous) */
        for (int b=0;b<B;++b){ for (int i=0;i<H;++i) xg[i]=xb[(size_t)i*B+b];
            qwen_rms_norm(xn, xg, nw, 1, H, 1e-6f); qwen_matvec_bf16(guf, Wgu, xn, 2*inter, H);
            qwen_swiglu_inplace(guf, tmp, inter); qwen_matvec_bf16(ob, Wd, guf, H, inter);
            for (int i=0;i<H;++i) ocpu[(size_t)i*B+b]=ob[i]+xg[i]; }
        qwen_metal_ffn_swiglu_batched(ctx, ogpu, xb, nw, Wgu, Wd, H, inter, B, 1e-6f);
        fails += check(f, "ffn_batched", ogpu, ocpu, H*B, 5e-3);
        int itb=20; double b0=nowms();
        for (int k=0;k<itb;++k) for (int b=0;b<B;++b){ for (int i=0;i<H;++i) xg[i]=xb[(size_t)i*B+b];
            qwen_rms_norm(xn,xg,nw,1,H,1e-6f); qwen_matvec_bf16(guf,Wgu,xn,2*inter,H);
            qwen_swiglu_inplace(guf,tmp,inter); qwen_matvec_bf16(ob,Wd,guf,H,inter);
            for (int i=0;i<H;++i) ocpu[(size_t)i*B+b]=ob[i]+xg[i]; }
        double bc=(nowms()-b0)/itb;
        b0=nowms(); for (int k=0;k<itb;++k) qwen_metal_ffn_swiglu_batched(ctx,ogpu,xb,nw,Wgu,Wd,H,inter,B,1e-6f);
        double bg=(nowms()-b0)/itb;
        fprintf(f, "  FFN BATCHED B=16 (MMA, 1 cmdbuf): cpu=%.3f ms  metal=%.3f ms  (%.2fx)\n", bc, bg, bg>0?bc/bg:0);
        free(Wgu);free(Wd);free(nw);free(xb);free(ocpu);free(ogpu);free(xg);free(xn);free(guf);free(tmp);free(ob);
    }

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
    fprintf(f, "  timing matvec_bf16 (per-op commit): cpu=%.3f ms  metal=%.3f ms  (%.2fx)\n", tc_mv, tg_mv, tg_mv>0?tc_mv/tg_mv:0);
    fprintf(f, "  timing matmat_bf16 (per-op commit): cpu=%.3f ms  metal=%.3f ms  (%.2fx)\n", tc_mm, tg_mm, tg_mm>0?tc_mm/tg_mm:0);
    /* fused: 200 matvecs in ONE command buffer (the real decode regime) */
    double tg_fused = qwen_metal_matvec_bench_fused(ctx, Wb, x, rows, cols, 200);
    t0 = nowms(); for (int k = 0; k < 200; ++k) qwen_matvec_bf16(cpu, Wb, x, rows, cols);
    double tc_op = (nowms()-t0)/200;
    fprintf(f, "  matvec FUSED (200 ops/1 cmdbuf): cpu=%.4f ms/op  metal=%.4f ms/op  (%.2fx)\n",
            tc_op, tg_fused, tg_fused>0?tc_op/tg_fused:0);
    /* compute-bound matmat at B=32 (batch/prefill regime — weight read amortized) */
    {
        int B2 = 32;
        float *X2 = malloc((size_t)cols*B2*4), *Yc = malloc((size_t)rows*B2*4), *Yg = malloc((size_t)rows*B2*4);
        for (int i = 0; i < cols*B2; ++i) X2[i] = rnd();
        qwen_matmat_bf16(Yc, Wb, X2, rows, cols, B2);
        qwen_metal_matmat_bf16(ctx, Yg, Wb, X2, rows, cols, B2);
        fails += check(f, "matmat B=32", Yg, Yc, rows*B2, 1e-2);
        int itm = 20; double m0 = nowms();
        for (int k = 0; k < itm; ++k) qwen_matmat_bf16(Yc, Wb, X2, rows, cols, B2);
        double mc = (nowms()-m0)/itm;
        m0 = nowms(); for (int k = 0; k < itm; ++k) qwen_metal_matmat_bf16(ctx, Yg, Wb, X2, rows, cols, B2);
        double mg = (nowms()-m0)/itm;
        fprintf(f, "  matmat B=32 (compute-bound): cpu=%.3f ms  metal=%.3f ms  (%.2fx)\n",
                mc, mg, mg>0?mc/mg:0);
        free(X2); free(Yc); free(Yg);
    }
    fprintf(f, "metal-selftest: %s (%d op%s failing)\n", fails ? "FAIL" : "PASS", fails, fails==1?"":"s");

    free(Wb); free(Wi); free(Ws); free(Wq); free(x); free(X); free(w); free(gu);
    free(ca); free(cb2); free(rc); free(rs); free(cpu); free(gpu); free(cpu2); free(gpu2);
    qwen_metal_free(ctx);
    return fails;
}
