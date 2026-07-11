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
#include "qwen_tts.h"
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
/* MUST match qwen_tts_kernels.h q4_0_block_t exactly (fp16 scale, 18 B/block since
 * perf item 2). MSL `half` is IEEE binary16; reads auto-convert to float in exprs. */
"struct q4blk { half scale; uchar qs[16]; };\n"
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
"}\n"
"\n"
/* ---- fused-step kernels (resident Talker/CP) ---------------------------- */
/* NeoX split-half RoPE at a given position (cos/sin base tables indexed by pos). */
"kernel void rope_neox(device float *x [[buffer(0)]], device const float *cos_base [[buffer(1)]],\n"
"    device const float *sin_base [[buffer(2)]], constant uint &n_heads [[buffer(3)]],\n"
"    constant uint &head_dim [[buffer(4)]], constant uint &pos [[buffer(5)]],\n"
"    uint gid [[thread_position_in_grid]]) {\n"
"    uint hd2 = head_dim/2; if (gid >= n_heads*hd2) return;\n"
"    device const float *cosp = cos_base + (ulong)pos*hd2; device const float *sinp = sin_base + (ulong)pos*hd2;\n"
"    uint h = gid/hd2, i = gid%hd2; device float *xh = x + (ulong)h*head_dim;\n"
"    float c = cosp[i], sn = sinp[i], x1 = xh[i], x2 = xh[i+hd2];\n"
"    xh[i] = x1*c - x2*sn; xh[i+hd2] = x2*c + x1*sn;\n"
"}\n"
/* per-head RMSNorm: one threadgroup per head, weight w[head_dim] shared. */
"kernel void rms_norm_ph(device float *x [[buffer(0)]], device const float *w [[buffer(1)]],\n"
"    constant uint &head_dim [[buffer(2)]], constant float &eps [[buffer(3)]],\n"
"    uint hh [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],\n"
"    uint tc [[threads_per_threadgroup]]) {\n"
"    threadgroup float part[256]; device float *xh = x + (ulong)hh*head_dim; float s = 0.0f;\n"
"    for (uint i = tid; i < head_dim; i += tc) s += xh[i]*xh[i];\n"
"    part[tid]=s; threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    for (uint st=tc/2; st>0; st>>=1){ if(tid<st) part[tid]+=part[tid+st]; threadgroup_barrier(mem_flags::mem_threadgroup); }\n"
"    float inv = 1.0f/sqrt(part[0]/float(head_dim)+eps);\n"
"    for (uint i = tid; i < head_dim; i += tc) xh[i]=xh[i]*inv*w[i];\n"
"}\n"
/* store k,v into KV cache at pos, bf16-TRUNCATED (match CPU bf16 KV: bits & 0xFFFF0000). */
"kernel void kv_store(device float *kc [[buffer(0)]], device float *vc [[buffer(1)]],\n"
"    device const float *k [[buffer(2)]], device const float *v [[buffer(3)]],\n"
"    constant uint &kvd [[buffer(4)]], constant uint &pos [[buffer(5)]], constant uint &base [[buffer(6)]],\n"
"    uint i [[thread_position_in_grid]]) {\n"
"    if (i >= kvd) return; ulong off = (ulong)base + (ulong)pos*kvd + i;\n"
"    kc[off] = as_type<float>(as_type<uint>(k[i]) & 0xFFFF0000u);\n"
"    vc[off] = as_type<float>(as_type<uint>(v[i]) & 0xFFFF0000u);\n"
"}\n"
/* single-token causal GQA attention over resident KV[0..pos]. One threadgroup per head,\n"
 * head_dim threads, online softmax (flash-style). head_dim must be a power of 2 (<=256). */
"kernel void attn_resident(device const float *Q [[buffer(0)]], device const float *K [[buffer(1)]],\n"
"    device const float *V [[buffer(2)]], device float *O [[buffer(3)]],\n"
"    constant uint &n_heads [[buffer(4)]], constant uint &n_kv [[buffer(5)]],\n"
"    constant uint &head_dim [[buffer(6)]], constant float &scale [[buffer(7)]], constant uint &pos [[buffer(8)]],\n"
"    constant uint &kvbase [[buffer(9)]],\n"
"    uint h [[threadgroup_position_in_grid]], uint t [[thread_position_in_threadgroup]]) {\n"
"    if (t >= head_dim) return; uint kvh = h/(n_heads/n_kv); uint valid = pos+1;\n"
"    threadgroup float sh[256]; float qt = Q[(ulong)h*head_dim + t];\n"
"    float m = -1e30f, denom = 0.0f, acc = 0.0f;\n"
"    for (uint j = 0; j < valid; ++j) {\n"
"        device const float *k = K + (ulong)kvbase + ((ulong)j*n_kv + kvh)*head_dim;\n"
"        sh[t] = qt * k[t]; threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"        for (uint s = head_dim/2; s > 0; s >>= 1) { if (t < s) sh[t] += sh[t+s]; threadgroup_barrier(mem_flags::mem_threadgroup); }\n"
"        float score = sh[0]*scale; threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"        float mn = max(m, score), corr = exp(m-mn), p = exp(score-mn);\n"
"        denom = denom*corr + p; acc = acc*corr + p*V[(ulong)kvbase + ((ulong)j*n_kv + kvh)*head_dim + t]; m = mn;\n"
"    }\n"
"    O[(ulong)h*head_dim + t] = acc/denom;\n"
"}\n"
/* in-place a += b */
"kernel void eadd_ip(device float *a [[buffer(0)]], device const float *b [[buffer(1)]],\n"
"    uint i [[thread_position_in_grid]]) { a[i] += b[i]; }\n"
/* FUSED residual-add + RMSNorm: x += proj (in place), xn = rmsnorm(x, w). (2 dispatches → 1) */
"kernel void add_rms(device float *x [[buffer(0)]], device const float *proj [[buffer(1)]],\n"
"    device const float *w [[buffer(2)]], device float *xn [[buffer(3)]], constant uint &dim [[buffer(4)]],\n"
"    constant float &eps [[buffer(5)]], uint tid [[thread_position_in_threadgroup]], uint tc [[threads_per_threadgroup]]) {\n"
"    threadgroup float part[256]; float s=0.0f;\n"
"    for (uint i=tid;i<dim;i+=tc){ float v=x[i]+proj[i]; x[i]=v; s+=v*v; }\n"
"    part[tid]=s; threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    for (uint st=tc/2;st>0;st>>=1){ if(tid<st) part[tid]+=part[tid+st]; threadgroup_barrier(mem_flags::mem_threadgroup); }\n"
"    float inv=1.0f/sqrt(part[0]/float(dim)+eps);\n"
"    for (uint i=tid;i<dim;i+=tc) xn[i]=x[i]*inv*w[i];\n"
"}\n"
/* FUSED per-head norm+rope for Q (rms_ph + rope_neox in one dispatch, one threadgroup/head). */
"kernel void qnorm_rope(device float *x [[buffer(0)]], device const float *w [[buffer(1)]],\n"
"    device const float *cos_base [[buffer(2)]], device const float *sin_base [[buffer(3)]],\n"
"    constant uint &head_dim [[buffer(4)]], constant uint &pos [[buffer(5)]], constant float &eps [[buffer(6)]],\n"
"    uint h [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]], uint tc [[threads_per_threadgroup]]) {\n"
"    threadgroup float part[256]; device float *xh = x + (ulong)h*head_dim; float s=0.0f;\n"
"    for (uint i=tid;i<head_dim;i+=tc) s+=xh[i]*xh[i]; part[tid]=s; threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    for (uint st=tc/2;st>0;st>>=1){ if(tid<st) part[tid]+=part[tid+st]; threadgroup_barrier(mem_flags::mem_threadgroup); }\n"
"    float inv=1.0f/sqrt(part[0]/float(head_dim)+eps);\n"
"    for (uint i=tid;i<head_dim;i+=tc) xh[i]=xh[i]*inv*w[i]; threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    uint hd2=head_dim/2; device const float *cp=cos_base+(ulong)pos*hd2, *sp=sin_base+(ulong)pos*hd2;\n"
"    for (uint i=tid;i<hd2;i+=tc){ float c=cp[i],sn=sp[i],x1=xh[i],x2=xh[i+hd2]; xh[i]=x1*c-x2*sn; xh[i+hd2]=x2*c+x1*sn; }\n"
"}\n"
/* FUSED per-head norm+rope for K + store K,V (bf16-trunc) to KV. One threadgroup per kv-head. */
"kernel void knorm_rope_store(device float *k [[buffer(0)]], device const float *v [[buffer(1)]],\n"
"    device const float *w [[buffer(2)]], device const float *cos_base [[buffer(3)]], device const float *sin_base [[buffer(4)]],\n"
"    device float *kc [[buffer(5)]], device float *vc [[buffer(6)]], constant uint &head_dim [[buffer(7)]],\n"
"    constant uint &pos [[buffer(8)]], constant float &eps [[buffer(9)]], constant uint &kvd [[buffer(10)]],\n"
"    constant uint &kvbase [[buffer(11)]], uint h [[threadgroup_position_in_grid]],\n"
"    uint tid [[thread_position_in_threadgroup]], uint tc [[threads_per_threadgroup]]) {\n"
"    threadgroup float part[256]; device float *kh = k + (ulong)h*head_dim; float s=0.0f;\n"
"    for (uint i=tid;i<head_dim;i+=tc) s+=kh[i]*kh[i]; part[tid]=s; threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    for (uint st=tc/2;st>0;st>>=1){ if(tid<st) part[tid]+=part[tid+st]; threadgroup_barrier(mem_flags::mem_threadgroup); }\n"
"    float inv=1.0f/sqrt(part[0]/float(head_dim)+eps);\n"
"    for (uint i=tid;i<head_dim;i+=tc) kh[i]=kh[i]*inv*w[i]; threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    uint hd2=head_dim/2; device const float *cp=cos_base+(ulong)pos*hd2, *sp=sin_base+(ulong)pos*hd2;\n"
"    for (uint i=tid;i<hd2;i+=tc){ float c=cp[i],sn=sp[i],x1=kh[i],x2=kh[i+hd2]; kh[i]=x1*c-x2*sn; kh[i+hd2]=x2*c+x1*sn; }\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    device const float *vh = v + (ulong)h*head_dim; ulong base = (ulong)kvbase + (ulong)pos*kvd + (ulong)h*head_dim;\n"
"    for (uint i=tid;i<head_dim;i+=tc){ kc[base+i]=as_type<float>(as_type<uint>(kh[i])&0xFFFF0000u); vc[base+i]=as_type<float>(as_type<uint>(vh[i])&0xFFFF0000u); }\n"
"}\n"
/* ---- device-frame CP: embed gather + argmax (16 passes on GPU, 1 sync/frame) ---------- */
/* out[i] = bf16_to_f32(table[idx*dim + i]); idx = code[cslot] (device int). code<0 → zeros. */
"kernel void embed_gather_bf16(device const ushort *table [[buffer(0)]], device const int *code [[buffer(1)]],\n"
"    device float *out [[buffer(2)]], constant uint &dim [[buffer(3)]], constant uint &cslot [[buffer(4)]],\n"
"    constant uint &vocab [[buffer(5)]], uint i [[thread_position_in_grid]]) {\n"
"    if (i >= dim) return; int idx = code[cslot];\n"
"    out[i] = (idx >= 0 && (uint)idx < vocab) ? bf16_to_f32(table[(ulong)idx*dim + i]) : 0.0f;\n"
"}\n"
"kernel void copy_vec(device const float *src [[buffer(0)]], device float *dst [[buffer(1)]],\n"
"    uint i [[thread_position_in_grid]]) { dst[i] = src[i]; }\n"
/* argmax over logits[rows]: one threadgroup, reduce (max,idx). Writes code[cslot]. */
"kernel void argmax_vec(device const float *logits [[buffer(0)]], device int *code [[buffer(1)]],\n"
"    constant uint &rows [[buffer(2)]], constant uint &cslot [[buffer(3)]],\n"
"    uint tid [[thread_position_in_threadgroup]], uint tc [[threads_per_threadgroup]]) {\n"
"    threadgroup float mv[256]; threadgroup int mi[256]; float best=-1e30f; int bi=0;\n"
"    for (uint r=tid;r<rows;r+=tc){ float v=logits[r]; if(v>best){best=v;bi=(int)r;} }\n"
"    mv[tid]=best; mi[tid]=bi; threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    for (uint st=tc/2;st>0;st>>=1){ if(tid<st){ if(mv[tid+st]>mv[tid]){mv[tid]=mv[tid+st];mi[tid]=mi[tid+st];} } threadgroup_barrier(mem_flags::mem_threadgroup); }\n"
"    if (tid==0) code[cslot] = mi[0];\n"
"}\n"
/* ===== BATCHED fused-step kernels (throughput epic) — direct port of the CUDA k_*_b path.
 * Layout [B][dim] (B-major, like CUDA); KV [B][kv_max][kvd]; d_pos[B] per-slot. mv_b_* = one
 * simdgroup per output row, reads W[row,:] ONCE, accumulates s[B] (weight DRAM amortized over B). */
"kernel void mv_b_bf16(device const ushort *W [[buffer(0)]],\n"
"    device const float *X [[buffer(1)]], device float *Y [[buffer(2)]],\n"
"    constant uint &cols [[buffer(3)]], constant uint &rows [[buffer(4)]], constant uint &B [[buffer(5)]],\n"
"    uint3 tgpig [[threadgroup_position_in_grid]], uint tiisg [[thread_index_in_simdgroup]],\n"
"    uint sgitg [[simdgroup_index_in_threadgroup]], uint nsg [[simdgroups_per_threadgroup]]) {\n"
"    uint row = tgpig.x*nsg + sgitg; if (row >= rows) return;\n"
"    device const ushort4 *w4 = (device const ushort4 *)(W + (ulong)row*cols);\n"
"    uint n4 = cols >> 2; float s[8]; for (uint b=0;b<8;++b) s[b]=0.0f;\n"
"    for (uint c=tiisg; c<n4; c+=32) { ushort4 bw = w4[c];\n"
"        float4 wf = float4(as_type<float>(uint(bw.x)<<16), as_type<float>(uint(bw.y)<<16),\n"
"                           as_type<float>(uint(bw.z)<<16), as_type<float>(uint(bw.w)<<16));\n"
"        for (uint b=0;b<B;++b) { device const float4 *x4=(device const float4*)(X+(ulong)b*cols); s[b] += dot(wf, x4[c]); } }\n"
"    for (uint b=0;b<B;++b) { float v = simd_sum(s[b]); if (tiisg==0) Y[(ulong)b*rows+row]=v; }\n"
"}\n"
"kernel void mv_b_int8(device const char *W [[buffer(0)]], device const float *scale [[buffer(1)]],\n"
"    device const float *X [[buffer(2)]], device float *Y [[buffer(3)]],\n"
"    constant uint &cols [[buffer(4)]], constant uint &rows [[buffer(5)]], constant uint &B [[buffer(6)]],\n"
"    uint3 tgpig [[threadgroup_position_in_grid]], uint tiisg [[thread_index_in_simdgroup]],\n"
"    uint sgitg [[simdgroup_index_in_threadgroup]], uint nsg [[simdgroups_per_threadgroup]]) {\n"
"    uint row = tgpig.x*nsg + sgitg; if (row >= rows) return;\n"
"    device const char4 *w4 = (device const char4 *)(W + (ulong)row*cols); float sc = scale[row];\n"
"    uint n4 = cols >> 2; float s[8]; for (uint b=0;b<8;++b) s[b]=0.0f;\n"
"    for (uint c=tiisg; c<n4; c+=32) { char4 wv = w4[c];\n"
"        float4 wf = float4(float(wv.x), float(wv.y), float(wv.z), float(wv.w));\n"
"        for (uint b=0;b<B;++b) { device const float4 *x4=(device const float4*)(X+(ulong)b*cols); s[b] += dot(wf, x4[c]); } }\n"
"    for (uint b=0;b<B;++b) { float v = simd_sum(s[b]); if (tiisg==0) Y[(ulong)b*rows+row]=sc*v; }\n"
"}\n"
"kernel void mv_b_q4(device const q4blk *W [[buffer(0)]],\n"
"    device const float *X [[buffer(1)]], device float *Y [[buffer(2)]],\n"
"    constant uint &cols [[buffer(3)]], constant uint &rows [[buffer(4)]], constant uint &B [[buffer(5)]],\n"
"    uint3 tgpig [[threadgroup_position_in_grid]], uint tiisg [[thread_index_in_simdgroup]],\n"
"    uint sgitg [[simdgroup_index_in_threadgroup]], uint nsg [[simdgroups_per_threadgroup]]) {\n"
"    uint row = tgpig.x*nsg + sgitg; if (row >= rows) return;\n"
"    uint nb = cols >> 5; device const q4blk *wr = W + (ulong)row*nb; float s[8]; for (uint b=0;b<8;++b) s[b]=0.0f;\n"
"    for (uint c=tiisg; c<cols; c+=32) { device const q4blk *bk = wr + (c>>5); uint ic = c & 31;\n"
"        uchar byte = bk->qs[ic>>1]; int nib = (ic&1) ? (byte>>4) : (byte&0x0F); float w = float(nib-8) * bk->scale;\n"
"        for (uint b=0;b<B;++b) s[b] += w * X[(ulong)b*cols + c]; }\n"
"    for (uint b=0;b<B;++b) { float v = simd_sum(s[b]); if (tiisg==0) Y[(ulong)b*rows+row]=v; }\n"
"}\n"
"kernel void rmsnorm_full_b(device const float *X [[buffer(0)]], device const float *w [[buffer(1)]],\n"
"    device float *Y [[buffer(2)]], constant uint &dim [[buffer(3)]], constant float &eps [[buffer(4)]],\n"
"    uint b [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]], uint tc [[threads_per_threadgroup]]) {\n"
"    threadgroup float part[256]; device const float *x = X + (ulong)b*dim; device float *y = Y + (ulong)b*dim;\n"
"    float s=0.0f; for (uint i=tid;i<dim;i+=tc){ float v=x[i]; s+=v*v; }\n"
"    part[tid]=s; threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    for (uint st=tc/2;st>0;st>>=1){ if(tid<st) part[tid]+=part[tid+st]; threadgroup_barrier(mem_flags::mem_threadgroup); }\n"
"    float inv = 1.0f/sqrt(part[0]/float(dim)+eps); for (uint i=tid;i<dim;i+=tc) y[i]=x[i]*inv*w[i];\n"
"}\n"
"kernel void rmsnorm_ph_b(device float *X [[buffer(0)]], device const float *w [[buffer(1)]],\n"
"    constant uint &head_dim [[buffer(2)]], constant uint &nh [[buffer(3)]], constant uint &stride [[buffer(4)]],\n"
"    constant float &eps [[buffer(5)]], uint blk [[threadgroup_position_in_grid]],\n"
"    uint tid [[thread_position_in_threadgroup]], uint tc [[threads_per_threadgroup]]) {\n"
"    uint b = blk/nh, h = blk%nh; device float *xh = X + (ulong)b*stride + (ulong)h*head_dim;\n"
"    threadgroup float part[256]; float s=0.0f; for (uint i=tid;i<head_dim;i+=tc){ float v=xh[i]; s+=v*v; }\n"
"    part[tid]=s; threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    for (uint st=tc/2;st>0;st>>=1){ if(tid<st) part[tid]+=part[tid+st]; threadgroup_barrier(mem_flags::mem_threadgroup); }\n"
"    float inv = 1.0f/sqrt(part[0]/float(head_dim)+eps); for (uint i=tid;i<head_dim;i+=tc) xh[i]=xh[i]*inv*w[i];\n"
"}\n"
"kernel void rope_neox_b(device float *X [[buffer(0)]], device const float *cos_base [[buffer(1)]],\n"
"    device const float *sin_base [[buffer(2)]], constant uint &n_heads [[buffer(3)]], constant uint &head_dim [[buffer(4)]],\n"
"    device const int *d_pos [[buffer(5)]], constant uint &stride [[buffer(6)]], constant uint &B [[buffer(7)]],\n"
"    uint gid [[thread_position_in_grid]]) {\n"
"    uint hf = head_dim/2, per = n_heads*hf; if (gid >= B*per) return; uint b = gid/per, rem = gid%per;\n"
"    device const float *cosp = cos_base + (ulong)d_pos[b]*hf; device const float *sinp = sin_base + (ulong)d_pos[b]*hf;\n"
"    uint h = rem/hf, i = rem%hf; device float *xh = X + (ulong)b*stride + (ulong)h*head_dim;\n"
"    float c=cosp[i], sn=sinp[i], x1=xh[i], x2=xh[i+hf]; xh[i]=x1*c - x2*sn; xh[i+hf]=x2*c + x1*sn;\n"
"}\n"
"kernel void trunc_bf16_b(device float *x [[buffer(0)]], constant uint &n [[buffer(1)]], uint i [[thread_position_in_grid]]) {\n"
"    if (i>=n) return; uint u = as_type<uint>(x[i]) & 0xFFFF0000u; x[i] = as_type<float>(u);\n"
"}\n"
"kernel void kv_store_b(device float *kc [[buffer(0)]], device float *vc [[buffer(1)]],\n"
"    device const float *K [[buffer(2)]], device const float *V [[buffer(3)]], constant uint &kvd [[buffer(4)]],\n"
"    device const int *d_pos [[buffer(5)]], constant uint &kv_max [[buffer(6)]], constant uint &B [[buffer(7)]],\n"
"    uint gid [[thread_position_in_grid]]) { if (gid >= B*kvd) return; uint b=gid/kvd, i=gid%kvd;\n"
"    ulong off = (ulong)b*kv_max*kvd + (ulong)d_pos[b]*kvd + i; kc[off]=K[(ulong)b*kvd+i]; vc[off]=V[(ulong)b*kvd+i];\n"
"}\n"
"kernel void attn_b(device const float *Q [[buffer(0)]], device const float *K [[buffer(1)]],\n"
"    device const float *V [[buffer(2)]], device float *O [[buffer(3)]], constant uint &n_heads [[buffer(4)]],\n"
"    constant uint &n_kv [[buffer(5)]], constant uint &hd [[buffer(6)]], constant float &scale [[buffer(7)]],\n"
"    device const int *d_pos [[buffer(8)]], constant uint &kv_max [[buffer(9)]], constant uint &qd [[buffer(10)]],\n"
"    constant uint &kvd [[buffer(11)]], uint blk [[threadgroup_position_in_grid]], uint t [[thread_position_in_threadgroup]]) {\n"
"    uint b = blk/n_heads, h = blk%n_heads; if (t >= hd) return; uint kvh = h/(n_heads/n_kv); uint valid = uint(d_pos[b])+1;\n"
"    device const float *Kb = K + (ulong)b*kv_max*kvd; device const float *Vb = V + (ulong)b*kv_max*kvd;\n"
"    float qt = Q[(ulong)b*qd + (ulong)h*hd + t]; threadgroup float sh[256]; float m=-1e30f, denom=0.0f, acc=0.0f;\n"
"    for (uint j=0;j<valid;++j) { device const float *k = Kb + ((ulong)j*n_kv+kvh)*hd;\n"
"        sh[t] = qt*k[t]; threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"        for (uint sr=hd/2;sr>0;sr>>=1){ if(t<sr) sh[t]+=sh[t+sr]; threadgroup_barrier(mem_flags::mem_threadgroup); }\n"
"        float score = sh[0]*scale; threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"        float mn = max(m,score), corr=exp(m-mn), p=exp(score-mn);\n"
"        denom=denom*corr+p; acc=acc*corr + p*Vb[((ulong)j*n_kv+kvh)*hd + t]; m=mn; }\n"
"    O[(ulong)b*qd + (ulong)h*hd + t] = acc/denom;\n"
"}\n"
"kernel void swiglu_il_b(device const float *in [[buffer(0)]], device float *out [[buffer(1)]],\n"
"    constant uint &inter [[buffer(2)]], constant uint &B [[buffer(3)]], uint e [[thread_position_in_grid]]) {\n"
"    if (e >= B*inter) return; uint b=e/inter, j=e%inter;\n"
"    device const float *ib = in + (ulong)b*2*inter; float g=ib[2*j], u=ib[2*j+1]; out[(ulong)b*inter+j] = g/(1.0f+exp(-g))*u;\n"
"}\n"
/* MMA batched matvec (opt-in QWEN_METAL_BATCH_MMA): simdgroup_float8x8, [B][cols] X -> [B][rows] Y.
 * Compute-bound (512 MAC/instr) → higher throughput at larger B; NOT bit-identical to the float4 mv_b
 * (different fp order) so the batch diverges from single-stream (valid but different audio). */
"kernel void mma_b_bf16(device const ushort *W [[buffer(0)]],\n"
"    device const float *X [[buffer(1)]], device float *Y [[buffer(2)]],\n"
"    constant uint &cols [[buffer(3)]], constant uint &rows [[buffer(4)]], constant uint &B [[buffer(5)]],\n"
"    uint2 tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_simdgroup]]) {\n"
"    threadgroup float Ws[64]; threadgroup float Xs[64]; threadgroup float Ys[64];\n"
"    uint row0 = tgid.x*8, col0 = tgid.y*8;\n"
"    simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float,8,8>(0.0f);\n"
"    for (uint k0=0;k0<cols;k0+=8) {\n"
"        for (uint i=tid;i<64;i+=32){ uint r=i>>3,k=i&7; uint gr=row0+r; Ws[i]=(gr<rows)?as_type<float>(uint(W[(ulong)gr*cols+k0+k])<<16):0.0f; }\n"
"        for (uint i=tid;i<64;i+=32){ uint k=i>>3,b=i&7; uint gb=col0+b; Xs[i]=(gb<B)?X[(ulong)gb*cols+k0+k]:0.0f; }\n"
"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"        simdgroup_float8x8 wm,xm; simdgroup_load(wm,Ws,8); simdgroup_load(xm,Xs,8);\n"
"        simdgroup_multiply_accumulate(acc,wm,xm,acc);\n"
"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    }\n"
"    simdgroup_store(acc,Ys,8);\n"
"    for (uint i=tid;i<64;i+=32){ uint r=i>>3,b=i&7; uint gr=row0+r,gb=col0+b; if(gr<rows&&gb<B) Y[(ulong)gb*rows+gr]=Ys[i]; }\n"
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
    /* fused-step pipelines (resident Talker/CP) */
    void *pso_rope_neox, *pso_rms_ph, *pso_kv_store, *pso_attn_res, *pso_eadd_ip;
    void *pso_embed_gather, *pso_copy_vec, *pso_argmax;   /* device-frame CP */
    void *pso_qnorm_rope, *pso_knorm_rope_store, *pso_add_rms;   /* fused attention preamble + add+norm */
    /* batched-step pipelines (throughput epic) */
    void *pso_mvb_bf16, *pso_mvb_int8, *pso_mvb_q4, *pso_rmsf_b, *pso_rmsph_b;
    void *pso_ropeneox_b, *pso_trunc_b, *pso_kvstore_b, *pso_attnb, *pso_swiglu_ilb;
    void *pso_mmab_bf16;   /* opt-in MMA batched matvec (QWEN_METAL_BATCH_MMA) */
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
        c->pso_rope_neox = make_pso(dev, lib, "rope_neox");
        c->pso_rms_ph = make_pso(dev, lib, "rms_norm_ph");
        c->pso_kv_store = make_pso(dev, lib, "kv_store");
        c->pso_attn_res = make_pso(dev, lib, "attn_resident");
        c->pso_eadd_ip = make_pso(dev, lib, "eadd_ip");
        c->pso_embed_gather = make_pso(dev, lib, "embed_gather_bf16");
        c->pso_copy_vec = make_pso(dev, lib, "copy_vec");
        c->pso_argmax = make_pso(dev, lib, "argmax_vec");
        c->pso_qnorm_rope = make_pso(dev, lib, "qnorm_rope");
        c->pso_knorm_rope_store = make_pso(dev, lib, "knorm_rope_store");
        c->pso_add_rms = make_pso(dev, lib, "add_rms");
        c->pso_mvb_bf16 = make_pso(dev, lib, "mv_b_bf16");
        c->pso_mvb_int8 = make_pso(dev, lib, "mv_b_int8");
        c->pso_mvb_q4 = make_pso(dev, lib, "mv_b_q4");
        c->pso_rmsf_b = make_pso(dev, lib, "rmsnorm_full_b");
        c->pso_rmsph_b = make_pso(dev, lib, "rmsnorm_ph_b");
        c->pso_ropeneox_b = make_pso(dev, lib, "rope_neox_b");
        c->pso_trunc_b = make_pso(dev, lib, "trunc_bf16_b");
        c->pso_kvstore_b = make_pso(dev, lib, "kv_store_b");
        c->pso_attnb = make_pso(dev, lib, "attn_b");
        c->pso_swiglu_ilb = make_pso(dev, lib, "swiglu_il_b");
        c->pso_mmab_bf16 = make_pso(dev, lib, "mma_b_bf16");
        if (!c->pso_mvb_bf16 || !c->pso_mvb_int8 || !c->pso_mvb_q4 || !c->pso_rmsf_b || !c->pso_rmsph_b ||
            !c->pso_ropeneox_b || !c->pso_trunc_b || !c->pso_kvstore_b || !c->pso_attnb || !c->pso_swiglu_ilb ||
            !c->pso_mmab_bf16) {
            qwen_metal_free(c); return NULL;
        }
        if (!c->pso_matvec_bf16 || !c->pso_matmat_bf16 || !c->pso_matvec_int8 ||
            !c->pso_matvec_q4_0 || !c->pso_rms || !c->pso_swiglu || !c->pso_silu ||
            !c->pso_add || !c->pso_mul || !c->pso_scale || !c->pso_rope ||
            !c->pso_snake || !c->pso_attn || !c->pso_matmat_f32 || !c->pso_conv1d || !c->pso_convt1d ||
            !c->pso_rms_b || !c->pso_swiglu_b ||
            !c->pso_rope_neox || !c->pso_rms_ph || !c->pso_kv_store || !c->pso_attn_res || !c->pso_eadd_ip ||
            !c->pso_embed_gather || !c->pso_copy_vec || !c->pso_argmax ||
            !c->pso_qnorm_rope || !c->pso_knorm_rope_store || !c->pso_add_rms) {
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
    release_ptr(&c->pso_rope_neox); release_ptr(&c->pso_rms_ph); release_ptr(&c->pso_kv_store);
    release_ptr(&c->pso_attn_res); release_ptr(&c->pso_eadd_ip);
    release_ptr(&c->pso_embed_gather); release_ptr(&c->pso_copy_vec); release_ptr(&c->pso_argmax);
    release_ptr(&c->pso_qnorm_rope); release_ptr(&c->pso_knorm_rope_store); release_ptr(&c->pso_add_rms);
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

/* ======================================================================== *
 *  GPU-RESIDENT FUSED TALKER STEP (Metal, G2) — mirrors qwen_cuda_talker_step.
 *  Weights + KV + activations stay in MTLBuffers; the whole 28-layer step is
 *  encoded into ONE command buffer (dispatches ordered by buffer barriers),
 *  one commit/wait per step. Base = M1 and up (simdgroup matvec, threadgroup
 *  reductions, unified shared buffers). Precision picked per weight (bf16/int8/q4).
 * ======================================================================== */
#define MTB(x) ((__bridge id<MTLBuffer>)(x))
typedef struct {
    qwen_metal_ctx *mc; qwen_tts_ctx_t *ctx;
    int H, qd, kvd, inter, nh, nkv, hd, L, kv_max; float eps;
    void *kcache, *vcache, *xb, *xnb, *qb, *kb, *vb, *attnb, *projb, *gateb, *gub;
    void *rope_cos, *rope_sin;
} qwen_metal_talker_t;

static id<MTLBuffer> mk_buf(qwen_metal_ctx *c, size_t bytes) {
    id<MTLDevice> dev = (__bridge id<MTLDevice>)c->device;
    return [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
}
static void enc_bar(id<MTLComputeCommandEncoder> e){ [e memoryBarrierWithScope:MTLBarrierScopeBuffers]; }

/* precision-dispatched matvec: y[rows] = W[rows,cols] @ x, resident buffers. */
static void enc_mv(id<MTLComputeCommandEncoder> e, qwen_metal_talker_t *s,
                   const uint16_t *wbf, const int8_t *wi8, const float *wsc, const q4_0_block_t *wq4,
                   id<MTLBuffer> xb, id<MTLBuffer> yb, int rows, int cols, int bar) {
    qwen_metal_ctx *c = s->mc; uint32_t cc=(uint32_t)cols, rr=(uint32_t)rows;
    MTLSize tg = MTLSizeMake(((NSUInteger)rows+7)/8,1,1), tp = MTLSizeMake(8*32,1,1);
    if (wq4) {
        [e setComputePipelineState:MTB(c->pso_matvec_q4_0)];
        [e setBuffer:weight_buf(c,wq4,(size_t)rows*(cols/32)*sizeof(q4_0_block_t)) offset:0 atIndex:0];
        [e setBuffer:xb offset:0 atIndex:1]; [e setBuffer:yb offset:0 atIndex:2];
        [e setBytes:&cc length:4 atIndex:3]; [e setBytes:&rr length:4 atIndex:4];
    } else if (wi8) {
        [e setComputePipelineState:MTB(c->pso_matvec_int8)];
        [e setBuffer:weight_buf(c,wi8,(size_t)rows*cols) offset:0 atIndex:0];
        [e setBuffer:weight_buf(c,wsc,(size_t)rows*sizeof(float)) offset:0 atIndex:1];
        [e setBuffer:xb offset:0 atIndex:2]; [e setBuffer:yb offset:0 atIndex:3];
        [e setBytes:&cc length:4 atIndex:4]; [e setBytes:&rr length:4 atIndex:5];
    } else {
        [e setComputePipelineState:MTB(c->pso_matvec_bf16)];
        [e setBuffer:weight_buf(c,wbf,(size_t)rows*cols*sizeof(uint16_t)) offset:0 atIndex:0];
        [e setBuffer:xb offset:0 atIndex:1]; [e setBuffer:yb offset:0 atIndex:2];
        [e setBytes:&cc length:4 atIndex:3]; [e setBytes:&rr length:4 atIndex:4];
    }
    [e dispatchThreadgroups:tg threadsPerThreadgroup:tp]; if(bar) enc_bar(e);
}
/* full RMSNorm: y[dim] = rmsnorm(x, w). */
static void enc_rms(id<MTLComputeCommandEncoder> e, qwen_metal_talker_t *s,
                    id<MTLBuffer> xb, id<MTLBuffer> yb, const float *w, int dim, int bar) {
    qwen_metal_ctx *c = s->mc; uint32_t d=(uint32_t)dim; float ep=s->eps;
    [e setComputePipelineState:MTB(c->pso_rms)];
    [e setBuffer:xb offset:0 atIndex:0];
    [e setBuffer:weight_buf(c,w,(size_t)dim*sizeof(float)) offset:0 atIndex:1];
    [e setBuffer:yb offset:0 atIndex:2];
    [e setBytes:&d length:4 atIndex:3]; [e setBytes:&ep length:4 atIndex:4];
    [e dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)]; if(bar) enc_bar(e);
}
static void enc_rms_ph(id<MTLComputeCommandEncoder> e, qwen_metal_talker_t *s,
                       id<MTLBuffer> xb, const float *w, int n_heads, int bar) {
    qwen_metal_ctx *c = s->mc; uint32_t hd=(uint32_t)s->hd; float ep=s->eps;
    [e setComputePipelineState:MTB(c->pso_rms_ph)];
    [e setBuffer:xb offset:0 atIndex:0];
    [e setBuffer:weight_buf(c,w,(size_t)s->hd*sizeof(float)) offset:0 atIndex:1];
    [e setBytes:&hd length:4 atIndex:2]; [e setBytes:&ep length:4 atIndex:3];
    NSUInteger tp = (NSUInteger)s->hd; if (tp > 256) tp = 256;
    [e dispatchThreadgroups:MTLSizeMake((NSUInteger)n_heads,1,1) threadsPerThreadgroup:MTLSizeMake(tp,1,1)]; if(bar) enc_bar(e);
}
static void enc_rope(id<MTLComputeCommandEncoder> e, qwen_metal_talker_t *s, id<MTLBuffer> xb, int n_heads, uint32_t pos, int bar) {
    qwen_metal_ctx *c = s->mc; uint32_t nh=(uint32_t)n_heads, hd=(uint32_t)s->hd;
    [e setComputePipelineState:MTB(c->pso_rope_neox)];
    [e setBuffer:xb offset:0 atIndex:0]; [e setBuffer:MTB(s->rope_cos) offset:0 atIndex:1];
    [e setBuffer:MTB(s->rope_sin) offset:0 atIndex:2];
    [e setBytes:&nh length:4 atIndex:3]; [e setBytes:&hd length:4 atIndex:4]; [e setBytes:&pos length:4 atIndex:5];
    NSUInteger n = (NSUInteger)n_heads*(s->hd/2), tp = n<256?n:256;
    [e dispatchThreadgroups:MTLSizeMake((n+tp-1)/tp,1,1) threadsPerThreadgroup:MTLSizeMake(tp,1,1)]; if(bar) enc_bar(e);
}
/* fused per-head RMSNorm+RoPE for Q (2 dispatches → 1). */
static void enc_qnorm_rope(id<MTLComputeCommandEncoder> e, qwen_metal_talker_t *s,
                           id<MTLBuffer> xb, const float *w, int n_heads, uint32_t pos, int bar) {
    qwen_metal_ctx *c=s->mc; uint32_t uhd=(uint32_t)s->hd; float ep=s->eps;
    [e setComputePipelineState:MTB(c->pso_qnorm_rope)];
    [e setBuffer:xb offset:0 atIndex:0]; [e setBuffer:weight_buf(c,w,(size_t)s->hd*sizeof(float)) offset:0 atIndex:1];
    [e setBuffer:MTB(s->rope_cos) offset:0 atIndex:2]; [e setBuffer:MTB(s->rope_sin) offset:0 atIndex:3];
    [e setBytes:&uhd length:4 atIndex:4]; [e setBytes:&pos length:4 atIndex:5]; [e setBytes:&ep length:4 atIndex:6];
    NSUInteger tp=(NSUInteger)s->hd<256?(NSUInteger)s->hd:256;
    [e dispatchThreadgroups:MTLSizeMake((NSUInteger)n_heads,1,1) threadsPerThreadgroup:MTLSizeMake(tp,1,1)]; if(bar) enc_bar(e);
}
/* fused per-head RMSNorm+RoPE for K + store K,V (bf16-trunc) to KV (3 dispatches → 1). */
static void enc_knorm_rope_store(id<MTLComputeCommandEncoder> e, qwen_metal_talker_t *s,
                                 id<MTLBuffer> kb, id<MTLBuffer> vb, const float *w, int n_kv, uint32_t pos,
                                 id<MTLBuffer> kc, id<MTLBuffer> vc, uint32_t kvbase) {
    qwen_metal_ctx *c=s->mc; uint32_t uhd=(uint32_t)s->hd, ukvd=(uint32_t)s->kvd; float ep=s->eps;
    [e setComputePipelineState:MTB(c->pso_knorm_rope_store)];
    [e setBuffer:kb offset:0 atIndex:0]; [e setBuffer:vb offset:0 atIndex:1];
    [e setBuffer:weight_buf(c,w,(size_t)s->hd*sizeof(float)) offset:0 atIndex:2];
    [e setBuffer:MTB(s->rope_cos) offset:0 atIndex:3]; [e setBuffer:MTB(s->rope_sin) offset:0 atIndex:4];
    [e setBuffer:kc offset:0 atIndex:5]; [e setBuffer:vc offset:0 atIndex:6];
    [e setBytes:&uhd length:4 atIndex:7]; [e setBytes:&pos length:4 atIndex:8]; [e setBytes:&ep length:4 atIndex:9];
    [e setBytes:&ukvd length:4 atIndex:10]; [e setBytes:&kvbase length:4 atIndex:11];
    NSUInteger tp=(NSUInteger)s->hd<256?(NSUInteger)s->hd:256;
    [e dispatchThreadgroups:MTLSizeMake((NSUInteger)n_kv,1,1) threadsPerThreadgroup:MTLSizeMake(tp,1,1)]; enc_bar(e);
}
/* fused residual-add + RMSNorm: xb += projb, xnb = rmsnorm(xb, w). (2 dispatches → 1) */
static void enc_add_rms(id<MTLComputeCommandEncoder> e, qwen_metal_talker_t *s,
                        id<MTLBuffer> xb, id<MTLBuffer> projb, const float *w, id<MTLBuffer> xnb, int dim) {
    qwen_metal_ctx *c=s->mc; uint32_t d=(uint32_t)dim; float ep=s->eps;
    [e setComputePipelineState:MTB(c->pso_add_rms)];
    [e setBuffer:xb offset:0 atIndex:0]; [e setBuffer:projb offset:0 atIndex:1];
    [e setBuffer:weight_buf(c,w,(size_t)dim*sizeof(float)) offset:0 atIndex:2]; [e setBuffer:xnb offset:0 atIndex:3];
    [e setBytes:&d length:4 atIndex:4]; [e setBytes:&ep length:4 atIndex:5];
    [e dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)]; enc_bar(e);
}

void *qwen_metal_talker_init(void *metal_ctx, qwen_tts_ctx_t *ctx) {
    if (!ctx || !metal_ctx) return NULL;
    @autoreleasepool {
        qwen_metal_ctx *c = (qwen_metal_ctx *)metal_ctx;
        qwen_tts_config_t *cf = &ctx->config;
        qwen_metal_talker_t *s = calloc(1, sizeof(*s));
        s->mc = c; s->ctx = ctx;
        s->H = cf->hidden_size; s->nh = cf->num_heads; s->nkv = cf->num_kv_heads; s->hd = cf->head_dim;
        s->inter = cf->intermediate_size; s->L = cf->num_layers; s->kv_max = ctx->kv_max;
        s->qd = s->nh*s->hd; s->kvd = s->nkv*s->hd; s->eps = cf->rms_norm_eps;
        int half = s->hd/2;
        id<MTLDevice> dev = (__bridge id<MTLDevice>)c->device;
        s->rope_cos = (__bridge_retained void *)[dev newBufferWithBytes:ctx->rope_cos length:(size_t)s->kv_max*half*sizeof(float) options:MTLResourceStorageModeShared];
        s->rope_sin = (__bridge_retained void *)[dev newBufferWithBytes:ctx->rope_sin length:(size_t)s->kv_max*half*sizeof(float) options:MTLResourceStorageModeShared];
        s->kcache = (__bridge_retained void *)mk_buf(c,(size_t)s->L*s->kv_max*s->kvd*sizeof(float));
        s->vcache = (__bridge_retained void *)mk_buf(c,(size_t)s->L*s->kv_max*s->kvd*sizeof(float));
        s->xb    = (__bridge_retained void *)mk_buf(c,(size_t)s->H*sizeof(float));
        s->xnb   = (__bridge_retained void *)mk_buf(c,(size_t)s->H*sizeof(float));
        s->qb    = (__bridge_retained void *)mk_buf(c,(size_t)s->qd*sizeof(float));
        s->kb    = (__bridge_retained void *)mk_buf(c,(size_t)s->kvd*sizeof(float));
        s->vb    = (__bridge_retained void *)mk_buf(c,(size_t)s->kvd*sizeof(float));
        s->attnb = (__bridge_retained void *)mk_buf(c,(size_t)s->qd*sizeof(float));
        s->projb = (__bridge_retained void *)mk_buf(c,(size_t)s->H*sizeof(float));
        s->gateb = (__bridge_retained void *)mk_buf(c,(size_t)s->inter*sizeof(float));
        s->gub   = (__bridge_retained void *)mk_buf(c,(size_t)2*s->inter*sizeof(float));
        fprintf(stderr,"Metal talker: resident fused step ready (%d layers, hidden=%d, kv_max=%d)\n", s->L, s->H, s->kv_max);
        return s;
    }
}

void qwen_metal_talker_step(void *st, const float *embed, float *hidden_out, int pos) {
    qwen_metal_talker_t *s = st; if (!s) return;
    @autoreleasepool {
        qwen_metal_ctx *c = s->mc;
        id<MTLCommandQueue> q = (__bridge id<MTLCommandQueue>)c->queue;
        id<MTLBuffer> xb=MTB(s->xb), xnb=MTB(s->xnb), qb=MTB(s->qb), kb=MTB(s->kb), vb=MTB(s->vb);
        id<MTLBuffer> attnb=MTB(s->attnb), projb=MTB(s->projb), gateb=MTB(s->gateb), gub=MTB(s->gub);
        id<MTLBuffer> kc=MTB(s->kcache), vc=MTB(s->vcache);
        memcpy(xb.contents, embed, (size_t)s->H*sizeof(float));
        int H=s->H, qd=s->qd, kvd=s->kvd, hd=s->hd, nh=s->nh, nkv=s->nkv, inter=s->inter;
        float scale = 1.0f/sqrtf((float)hd); uint32_t upos=(uint32_t)pos;
        static int mprof=-1; if(mprof<0) mprof=getenv("QWEN_METAL_PROFILE")?1:0;
        static double t_enc=0,t_gpu=0; static long np=0; double te0 = mprof?nowms():0;
        id<MTLCommandBuffer> cb = [q commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        for (int l=0; l<s->L; ++l) {
            qwen_talker_layer_t *ly = &s->ctx->layers[l];
            uint32_t kvbase = (uint32_t)((size_t)l*s->kv_max*kvd);
            enc_rms(e, s, xb, xnb, ly->input_norm, H, 1);
            /* q/k/v read the same xnb, write different buffers → NO barrier between (overlap) */
            enc_mv(e, s, ly->wq_bf16, ly->wq_int8, ly->wq_scale, ly->wq_q4, xnb, qb, qd, H, 0);
            enc_mv(e, s, ly->wk_bf16, ly->wk_int8, ly->wk_scale, ly->wk_q4, xnb, kb, kvd, H, 0);
            enc_mv(e, s, ly->wv_bf16, ly->wv_int8, ly->wv_scale, ly->wv_q4, xnb, vb, kvd, H, 1);
            enc_qnorm_rope(e, s, qb, ly->q_norm, nh, upos, 0);            /* fused: norm+rope Q */
            enc_knorm_rope_store(e, s, kb, vb, ly->k_norm, nkv, upos, kc, vc, kvbase); /* fused: norm+rope K + store K,V */
            /* attention over resident KV */
            { uint32_t unh=(uint32_t)nh, unkv=(uint32_t)nkv, uhd=(uint32_t)hd;
              [e setComputePipelineState:MTB(c->pso_attn_res)];
              [e setBuffer:qb offset:0 atIndex:0]; [e setBuffer:kc offset:0 atIndex:1]; [e setBuffer:vc offset:0 atIndex:2];
              [e setBuffer:attnb offset:0 atIndex:3];
              [e setBytes:&unh length:4 atIndex:4]; [e setBytes:&unkv length:4 atIndex:5]; [e setBytes:&uhd length:4 atIndex:6];
              [e setBytes:&scale length:4 atIndex:7]; [e setBytes:&upos length:4 atIndex:8]; [e setBytes:&kvbase length:4 atIndex:9];
              [e dispatchThreadgroups:MTLSizeMake((NSUInteger)nh,1,1) threadsPerThreadgroup:MTLSizeMake((NSUInteger)hd,1,1)]; enc_bar(e); }
            enc_mv(e, s, ly->wo_bf16, ly->wo_int8, ly->wo_scale, ly->wo_q4, attnb, projb, H, qd, 1);
            enc_add_rms(e, s, xb, projb, ly->post_attn_norm, xnb, H);   /* fused: x+=proj, xn=rmsnorm(x) */
            enc_mv(e, s, ly->gate_up_fused_bf16, ly->gate_up_fused_int8, ly->gate_up_fused_scale, ly->gate_up_fused_q4, xnb, gub, 2*inter, H, 1);
            /* swiglu gub -> gateb */
            [e setComputePipelineState:MTB(c->pso_swiglu)];
            [e setBuffer:gub offset:0 atIndex:0]; [e setBuffer:gateb offset:0 atIndex:1];
            { NSUInteger tp=(NSUInteger)inter<256?(NSUInteger)inter:256; [e dispatchThreadgroups:MTLSizeMake(((NSUInteger)inter+tp-1)/tp,1,1) threadsPerThreadgroup:MTLSizeMake(tp,1,1)]; } enc_bar(e);
            enc_mv(e, s, ly->down_bf16, ly->down_int8, ly->down_scale, ly->down_q4, gateb, projb, H, inter, 1);
            [e setComputePipelineState:MTB(c->pso_eadd_ip)];
            [e setBuffer:xb offset:0 atIndex:0]; [e setBuffer:projb offset:0 atIndex:1];
            { NSUInteger tp=(NSUInteger)H<256?(NSUInteger)H:256; [e dispatchThreadgroups:MTLSizeMake(((NSUInteger)H+tp-1)/tp,1,1) threadsPerThreadgroup:MTLSizeMake(tp,1,1)]; } enc_bar(e);
        }
        enc_rms(e, s, xb, xnb, s->ctx->talker_norm, H, 1);
        [e endEncoding];
        if(mprof) t_enc += nowms()-te0;
        double tg0 = mprof?nowms():0;
        [cb commit]; [cb waitUntilCompleted];
        if(mprof){ t_gpu += nowms()-tg0; if(++np%50==0)
            fprintf(stderr,"[MTL-PROF] avg encode=%.2f ms | gpu(commit+wait)=%.2f ms | total=%.2f ms/step (%ld steps)\n",
                    t_enc/np, t_gpu/np, (t_enc+t_gpu)/np, np); }
        memcpy(hidden_out, xnb.contents, (size_t)H*sizeof(float));
    }
}

/* Seed the device KV from the CPU batched prefill (ctx->kv_cache_{k,v} bf16), so the fused
 * decode steps attend to the prompt. Mirrors qwen_cuda_talker_upload_kv. bf16->f32 = bits<<16
 * (= the bf16-truncated f32 kv_store also writes). Shared buffers → write .contents directly. */
void qwen_metal_talker_upload_kv(void *st, qwen_tts_ctx_t *ctx, int prefill_len) {
    qwen_metal_talker_t *s = st; if (!s || prefill_len <= 0) return;
    @autoreleasepool {
        int L=s->L, kvd=s->kvd, kvm=s->kv_max;
        float *kcp = (float *)((__bridge id<MTLBuffer>)s->kcache).contents;
        float *vcp = (float *)((__bridge id<MTLBuffer>)s->vcache).contents;
        for (int l=0; l<L; ++l) {
            const uint16_t *ck = ctx->kv_cache_k + (size_t)l*kvm*kvd;
            const uint16_t *cv = ctx->kv_cache_v + (size_t)l*kvm*kvd;
            float *dk = kcp + (size_t)l*kvm*kvd, *dv = vcp + (size_t)l*kvm*kvd;
            size_t n = (size_t)prefill_len*kvd;
            for (size_t i=0;i<n;++i){ union{uint32_t u;float f;}a,b;
                a.u=(uint32_t)ck[i]<<16; dk[i]=a.f; b.u=(uint32_t)cv[i]<<16; dv[i]=b.f; }
        }
    }
}

void qwen_metal_talker_free(void *st) {
    qwen_metal_talker_t *s = st; if (!s) return;
    void *ps[] = {s->kcache,s->vcache,s->xb,s->xnb,s->qb,s->kb,s->vb,s->attnb,s->projb,s->gateb,s->gub,s->rope_cos,s->rope_sin};
    for (unsigned i=0;i<sizeof(ps)/sizeof(ps[0]);++i){ void *p=ps[i]; if(p){ id o=(__bridge_transfer id)p; (void)o; } }
    free(s);
}

/* ======================================================================== *
 *  BATCHED fused Talker step (Metal, throughput epic) — mirrors qwen_cuda_talker_batch_*.
 *  Shares the single state's resident weights (via ctx->layers + weight_buf cache); activations
 *  [B][dim], KV [L][B][kv_max][kvd], d_pos[B] per-slot. Every dispatch is barrier-serialized
 *  (correctness-first; overlap tuning later). B<=QMB_MAX. bf16/int8/q4 via mv_b_* (weight read
 *  once, s[B] accumulator = the amortization win). Validate B=1==single before trusting B>1.
 * ======================================================================== */
#define QMB_MAX 8
typedef struct {
    qwen_metal_ctx *mc; qwen_tts_ctx_t *ctx;
    int B, H, qd, kvd, inter, nh, nkv, hd, L, kv_max; float eps;
    void *kcache, *vcache;                              /* [L][B][kv_max][kvd] */
    void *x, *xn, *q, *k, *v, *attn, *proj, *gate, *gu; /* [B][dim] */
    void *d_pos;                                        /* int[B] */
    void *rope_cos, *rope_sin;                          /* owned (talker rope or cp rope) */
} qwen_metal_talker_batch_t;

/* precision-dispatched BATCHED matmat: Y[B][rows] = W[rows,cols] @ X[B][cols]. */
static void enc_mvb(id<MTLComputeCommandEncoder> e, qwen_metal_talker_batch_t *s,
                    const uint16_t *wbf, const int8_t *wi8, const float *wsc, const q4_0_block_t *wq4,
                    id<MTLBuffer> xb, id<MTLBuffer> yb, int rows, int cols, int bar) {
    qwen_metal_ctx *c=s->mc; uint32_t cc=(uint32_t)cols, rr=(uint32_t)rows, BB=(uint32_t)s->B;
    MTLSize tg=MTLSizeMake(((NSUInteger)rows+7)/8,1,1), tp=MTLSizeMake(8*32,1,1);
    static int mma=-1; if(mma<0) mma=getenv("QWEN_METAL_BATCH_MMA")?1:0;
    if (mma && wbf) {   /* opt-in MMA: bf16 only, simdgroup_float8x8 (compute-bound, not bit-identical) */
        [e setComputePipelineState:MTB(c->pso_mmab_bf16)];
        [e setBuffer:weight_buf(c,wbf,(size_t)rows*cols*sizeof(uint16_t)) offset:0 atIndex:0];
        [e setBuffer:xb offset:0 atIndex:1]; [e setBuffer:yb offset:0 atIndex:2];
        [e setBytes:&cc length:4 atIndex:3]; [e setBytes:&rr length:4 atIndex:4]; [e setBytes:&BB length:4 atIndex:5];
        [e dispatchThreadgroups:MTLSizeMake(((NSUInteger)rows+7)/8, ((NSUInteger)s->B+7)/8, 1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
        if(bar) enc_bar(e); return;
    }
    if (wq4) {
        [e setComputePipelineState:MTB(c->pso_mvb_q4)];
        [e setBuffer:weight_buf(c,wq4,(size_t)rows*(cols/32)*sizeof(q4_0_block_t)) offset:0 atIndex:0];
        [e setBuffer:xb offset:0 atIndex:1]; [e setBuffer:yb offset:0 atIndex:2];
        [e setBytes:&cc length:4 atIndex:3]; [e setBytes:&rr length:4 atIndex:4]; [e setBytes:&BB length:4 atIndex:5];
    } else if (wi8) {
        [e setComputePipelineState:MTB(c->pso_mvb_int8)];
        [e setBuffer:weight_buf(c,wi8,(size_t)rows*cols) offset:0 atIndex:0];
        [e setBuffer:weight_buf(c,wsc,(size_t)rows*sizeof(float)) offset:0 atIndex:1];
        [e setBuffer:xb offset:0 atIndex:2]; [e setBuffer:yb offset:0 atIndex:3];
        [e setBytes:&cc length:4 atIndex:4]; [e setBytes:&rr length:4 atIndex:5]; [e setBytes:&BB length:4 atIndex:6];
    } else {
        [e setComputePipelineState:MTB(c->pso_mvb_bf16)];
        [e setBuffer:weight_buf(c,wbf,(size_t)rows*cols*sizeof(uint16_t)) offset:0 atIndex:0];
        [e setBuffer:xb offset:0 atIndex:1]; [e setBuffer:yb offset:0 atIndex:2];
        [e setBytes:&cc length:4 atIndex:3]; [e setBytes:&rr length:4 atIndex:4]; [e setBytes:&BB length:4 atIndex:5];
    }
    [e dispatchThreadgroups:tg threadsPerThreadgroup:tp]; if(bar) enc_bar(e);
}
static void enc_rmsf_b(id<MTLComputeCommandEncoder> e, qwen_metal_talker_batch_t *s,
                       id<MTLBuffer> xb, const float *w, id<MTLBuffer> yb, int dim) {
    qwen_metal_ctx *c=s->mc; uint32_t d=(uint32_t)dim; float ep=s->eps;
    [e setComputePipelineState:MTB(c->pso_rmsf_b)];
    [e setBuffer:xb offset:0 atIndex:0]; [e setBuffer:weight_buf(c,w,(size_t)dim*sizeof(float)) offset:0 atIndex:1];
    [e setBuffer:yb offset:0 atIndex:2]; [e setBytes:&d length:4 atIndex:3]; [e setBytes:&ep length:4 atIndex:4];
    [e dispatchThreadgroups:MTLSizeMake((NSUInteger)s->B,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)]; enc_bar(e);
}
static void enc_rmsph_b(id<MTLComputeCommandEncoder> e, qwen_metal_talker_batch_t *s,
                        id<MTLBuffer> xb, const float *w, int n_heads, int stride) {
    qwen_metal_ctx *c=s->mc; uint32_t uhd=(uint32_t)s->hd, unh=(uint32_t)n_heads, ust=(uint32_t)stride; float ep=s->eps;
    [e setComputePipelineState:MTB(c->pso_rmsph_b)];
    [e setBuffer:xb offset:0 atIndex:0]; [e setBuffer:weight_buf(c,w,(size_t)s->hd*sizeof(float)) offset:0 atIndex:1];
    [e setBytes:&uhd length:4 atIndex:2]; [e setBytes:&unh length:4 atIndex:3]; [e setBytes:&ust length:4 atIndex:4]; [e setBytes:&ep length:4 atIndex:5];
    NSUInteger tp=(NSUInteger)s->hd<256?(NSUInteger)s->hd:256;
    [e dispatchThreadgroups:MTLSizeMake((NSUInteger)(s->B*n_heads),1,1) threadsPerThreadgroup:MTLSizeMake(tp,1,1)]; enc_bar(e);
}
static void enc_ropeneox_b(id<MTLComputeCommandEncoder> e, qwen_metal_talker_batch_t *s,
                           id<MTLBuffer> xb, int n_heads, int stride) {
    qwen_metal_ctx *c=s->mc; uint32_t unh=(uint32_t)n_heads, uhd=(uint32_t)s->hd, ust=(uint32_t)stride, uB=(uint32_t)s->B;
    [e setComputePipelineState:MTB(c->pso_ropeneox_b)];
    [e setBuffer:xb offset:0 atIndex:0]; [e setBuffer:MTB(s->rope_cos) offset:0 atIndex:1]; [e setBuffer:MTB(s->rope_sin) offset:0 atIndex:2];
    [e setBytes:&unh length:4 atIndex:3]; [e setBytes:&uhd length:4 atIndex:4];
    [e setBuffer:MTB(s->d_pos) offset:0 atIndex:5]; [e setBytes:&ust length:4 atIndex:6]; [e setBytes:&uB length:4 atIndex:7];
    NSUInteger n=(NSUInteger)s->B*n_heads*(s->hd/2), tp=n<256?n:256;
    [e dispatchThreadgroups:MTLSizeMake((n+tp-1)/tp,1,1) threadsPerThreadgroup:MTLSizeMake(tp,1,1)]; enc_bar(e);
}
static void enc_trunc_b(id<MTLComputeCommandEncoder> e, qwen_metal_talker_batch_t *s, id<MTLBuffer> xb, int n) {
    qwen_metal_ctx *c=s->mc; uint32_t un=(uint32_t)n;
    [e setComputePipelineState:MTB(c->pso_trunc_b)]; [e setBuffer:xb offset:0 atIndex:0]; [e setBytes:&un length:4 atIndex:1];
    NSUInteger tp=(NSUInteger)n<256?(NSUInteger)n:256;
    [e dispatchThreadgroups:MTLSizeMake(((NSUInteger)n+tp-1)/tp,1,1) threadsPerThreadgroup:MTLSizeMake(tp,1,1)]; enc_bar(e);
}
static void enc_eadd_flat(id<MTLComputeCommandEncoder> e, qwen_metal_talker_batch_t *s, id<MTLBuffer> a, id<MTLBuffer> b, int n) {
    qwen_metal_ctx *c=s->mc;
    [e setComputePipelineState:MTB(c->pso_eadd_ip)]; [e setBuffer:a offset:0 atIndex:0]; [e setBuffer:b offset:0 atIndex:1];
    NSUInteger tp=(NSUInteger)n<256?(NSUInteger)n:256;
    [e dispatchThreadgroups:MTLSizeMake(((NSUInteger)n+tp-1)/tp,1,1) threadsPerThreadgroup:MTLSizeMake(tp,1,1)]; enc_bar(e);
}
static void enc_kvstore_b(id<MTLComputeCommandEncoder> e, qwen_metal_talker_batch_t *s,
                          id<MTLBuffer> kc, id<MTLBuffer> vc, size_t kvoff) {
    qwen_metal_ctx *c=s->mc; uint32_t ukvd=(uint32_t)s->kvd, ukvm=(uint32_t)s->kv_max, uB=(uint32_t)s->B;
    [e setComputePipelineState:MTB(c->pso_kvstore_b)];
    [e setBuffer:kc offset:kvoff atIndex:0]; [e setBuffer:vc offset:kvoff atIndex:1];
    [e setBuffer:MTB(s->k) offset:0 atIndex:2]; [e setBuffer:MTB(s->v) offset:0 atIndex:3];
    [e setBytes:&ukvd length:4 atIndex:4]; [e setBuffer:MTB(s->d_pos) offset:0 atIndex:5];
    [e setBytes:&ukvm length:4 atIndex:6]; [e setBytes:&uB length:4 atIndex:7];
    NSUInteger n=(NSUInteger)s->B*s->kvd, tp=n<256?n:256;
    [e dispatchThreadgroups:MTLSizeMake((n+tp-1)/tp,1,1) threadsPerThreadgroup:MTLSizeMake(tp,1,1)]; enc_bar(e);
}
static void enc_attnb(id<MTLComputeCommandEncoder> e, qwen_metal_talker_batch_t *s,
                      id<MTLBuffer> kc, id<MTLBuffer> vc, size_t kvoff, float scale) {
    qwen_metal_ctx *c=s->mc; uint32_t unh=(uint32_t)s->nh, unkv=(uint32_t)s->nkv, uhd=(uint32_t)s->hd, uqd=(uint32_t)s->qd, ukvd=(uint32_t)s->kvd, ukvm=(uint32_t)s->kv_max;
    [e setComputePipelineState:MTB(c->pso_attnb)];
    [e setBuffer:MTB(s->q) offset:0 atIndex:0]; [e setBuffer:kc offset:kvoff atIndex:1]; [e setBuffer:vc offset:kvoff atIndex:2]; [e setBuffer:MTB(s->attn) offset:0 atIndex:3];
    [e setBytes:&unh length:4 atIndex:4]; [e setBytes:&unkv length:4 atIndex:5]; [e setBytes:&uhd length:4 atIndex:6]; [e setBytes:&scale length:4 atIndex:7];
    [e setBuffer:MTB(s->d_pos) offset:0 atIndex:8]; [e setBytes:&ukvm length:4 atIndex:9]; [e setBytes:&uqd length:4 atIndex:10]; [e setBytes:&ukvd length:4 atIndex:11];
    [e dispatchThreadgroups:MTLSizeMake((NSUInteger)(s->B*s->nh),1,1) threadsPerThreadgroup:MTLSizeMake((NSUInteger)s->hd,1,1)]; enc_bar(e);
}
static void enc_swiglu_ilb(id<MTLComputeCommandEncoder> e, qwen_metal_talker_batch_t *s, id<MTLBuffer> gub, id<MTLBuffer> gateb) {
    qwen_metal_ctx *c=s->mc; uint32_t uinter=(uint32_t)s->inter, uB=(uint32_t)s->B;
    [e setComputePipelineState:MTB(c->pso_swiglu_ilb)];
    [e setBuffer:gub offset:0 atIndex:0]; [e setBuffer:gateb offset:0 atIndex:1]; [e setBytes:&uinter length:4 atIndex:2]; [e setBytes:&uB length:4 atIndex:3];
    NSUInteger n=(NSUInteger)s->B*s->inter, tp=n<256?n:256;
    [e dispatchThreadgroups:MTLSizeMake((n+tp-1)/tp,1,1) threadsPerThreadgroup:MTLSizeMake(tp,1,1)]; enc_bar(e);
}
static void enc_talker_batch_body(id<MTLComputeCommandEncoder> e, qwen_metal_talker_batch_t *s) {
    int B=s->B, H=s->H, qd=s->qd, kvd=s->kvd, nh=s->nh, nkv=s->nkv, inter=s->inter;
    float scale=1.0f/sqrtf((float)s->hd);
    id<MTLBuffer> x=MTB(s->x), xn=MTB(s->xn), qb=MTB(s->q), kb=MTB(s->k), vb=MTB(s->v);
    id<MTLBuffer> attnb=MTB(s->attn), projb=MTB(s->proj), gateb=MTB(s->gate), gub=MTB(s->gu);
    id<MTLBuffer> kc=MTB(s->kcache), vc=MTB(s->vcache);
    for (int l=0; l<s->L; ++l) {
        qwen_talker_layer_t *ly = &s->ctx->layers[l];
        size_t kvoff = (size_t)l*B*s->kv_max*kvd*sizeof(float);
        enc_rmsf_b(e, s, x, ly->input_norm, xn, H);
        enc_mvb(e, s, ly->wq_bf16, ly->wq_int8, ly->wq_scale, ly->wq_q4, xn, qb, qd, H, 1);
        enc_mvb(e, s, ly->wk_bf16, ly->wk_int8, ly->wk_scale, ly->wk_q4, xn, kb, kvd, H, 1);
        enc_mvb(e, s, ly->wv_bf16, ly->wv_int8, ly->wv_scale, ly->wv_q4, xn, vb, kvd, H, 1);
        enc_rmsph_b(e, s, qb, ly->q_norm, nh, qd);
        enc_rmsph_b(e, s, kb, ly->k_norm, nkv, kvd);
        enc_ropeneox_b(e, s, qb, nh, qd);
        enc_ropeneox_b(e, s, kb, nkv, kvd);
        enc_trunc_b(e, s, kb, B*kvd);
        enc_trunc_b(e, s, vb, B*kvd);
        enc_kvstore_b(e, s, kc, vc, kvoff);
        enc_attnb(e, s, kc, vc, kvoff, scale);
        enc_mvb(e, s, ly->wo_bf16, ly->wo_int8, ly->wo_scale, ly->wo_q4, attnb, projb, H, qd, 1);
        enc_eadd_flat(e, s, x, projb, B*H);
        enc_rmsf_b(e, s, x, ly->post_attn_norm, xn, H);
        enc_mvb(e, s, ly->gate_up_fused_bf16, ly->gate_up_fused_int8, ly->gate_up_fused_scale, ly->gate_up_fused_q4, xn, gub, 2*inter, H, 1);
        enc_swiglu_ilb(e, s, gub, gateb);
        enc_mvb(e, s, ly->down_bf16, ly->down_int8, ly->down_scale, ly->down_q4, gateb, projb, H, inter, 1);
        enc_eadd_flat(e, s, x, projb, B*H);
    }
    enc_rmsf_b(e, s, x, s->ctx->talker_norm, xn, H);
}

void *qwen_metal_talker_batch_init(void *single_state, int B) {
    qwen_metal_talker_t *ss = single_state; if (!ss || B<1 || B>QMB_MAX) return NULL;
    @autoreleasepool {
        qwen_metal_ctx *c = ss->mc;
        qwen_metal_talker_batch_t *s = calloc(1, sizeof(*s));
        s->mc=c; s->ctx=ss->ctx; s->B=B;
        s->H=ss->H; s->qd=ss->qd; s->kvd=ss->kvd; s->inter=ss->inter;
        s->nh=ss->nh; s->nkv=ss->nkv; s->hd=ss->hd; s->L=ss->L; s->kv_max=ss->kv_max; s->eps=ss->eps;
        int H=s->H,qd=s->qd,kvd=s->kvd,inter=s->inter,L=s->L,kvm=s->kv_max, half=s->hd/2;
        id<MTLDevice> dev=(__bridge id<MTLDevice>)c->device;
        s->rope_cos=(__bridge_retained void*)[dev newBufferWithBytes:ss->ctx->rope_cos length:(size_t)kvm*half*sizeof(float) options:MTLResourceStorageModeShared];
        s->rope_sin=(__bridge_retained void*)[dev newBufferWithBytes:ss->ctx->rope_sin length:(size_t)kvm*half*sizeof(float) options:MTLResourceStorageModeShared];
        s->kcache=(__bridge_retained void*)mk_buf(c,(size_t)L*B*kvm*kvd*sizeof(float));
        s->vcache=(__bridge_retained void*)mk_buf(c,(size_t)L*B*kvm*kvd*sizeof(float));
        s->x =(__bridge_retained void*)mk_buf(c,(size_t)B*H*sizeof(float));
        s->xn=(__bridge_retained void*)mk_buf(c,(size_t)B*H*sizeof(float));
        s->q =(__bridge_retained void*)mk_buf(c,(size_t)B*qd*sizeof(float));
        s->k =(__bridge_retained void*)mk_buf(c,(size_t)B*kvd*sizeof(float));
        s->v =(__bridge_retained void*)mk_buf(c,(size_t)B*kvd*sizeof(float));
        s->attn=(__bridge_retained void*)mk_buf(c,(size_t)B*qd*sizeof(float));
        s->proj=(__bridge_retained void*)mk_buf(c,(size_t)B*H*sizeof(float));
        s->gate=(__bridge_retained void*)mk_buf(c,(size_t)B*inter*sizeof(float));
        s->gu =(__bridge_retained void*)mk_buf(c,(size_t)B*2*inter*sizeof(float));
        s->d_pos=(__bridge_retained void*)mk_buf(c,(size_t)B*sizeof(int));
        fprintf(stderr,"Metal talker BATCH ready (B=%d, %d layers, hidden=%d)\n", B, L, H);
        return s;
    }
}
/* embeds=[B][H] host, pos_arr=[B] host; hidden_out=[B][H] host (final-normed). */
void qwen_metal_talker_batch_step(void *st, const float *embeds, const int *pos_arr, float *hidden_out) {
    qwen_metal_talker_batch_t *s = st; if (!s) return;
    @autoreleasepool {
        qwen_metal_ctx *c=s->mc; id<MTLCommandQueue> q=(__bridge id<MTLCommandQueue>)c->queue;
        int B=s->B, H=s->H;
        id<MTLBuffer> xb=MTB(s->x), dpos=MTB(s->d_pos), xnb=MTB(s->xn);
        memcpy(xb.contents, embeds, (size_t)B*H*sizeof(float));
        memcpy(dpos.contents, pos_arr, (size_t)B*sizeof(int));
        id<MTLCommandBuffer> cb=[q commandBuffer]; id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
        enc_talker_batch_body(e, s);
        [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
        if (hidden_out) memcpy(hidden_out, xnb.contents, (size_t)B*H*sizeof(float));
    }
}
/* Seed slot b's device KV from the CPU batch engine bf16 KV (layout ((b*L+l)*src_kv_max+pos)*kvd). */
void qwen_metal_talker_batch_upload_slot(void *st, int b, const uint16_t *kv_k, const uint16_t *kv_v,
                                         int src_kv_max, int prefill_len) {
    qwen_metal_talker_batch_t *s = st; if (!s || prefill_len<=0) return;
    @autoreleasepool {
        int L=s->L, B=s->B, kvd=s->kvd, dkvm=s->kv_max;
        float *kcp=(float*)((__bridge id<MTLBuffer>)s->kcache).contents;
        float *vcp=(float*)((__bridge id<MTLBuffer>)s->vcache).contents;
        size_t nper=(size_t)prefill_len*kvd;
        for (int l=0; l<L; ++l) {
            const uint16_t *ck=kv_k+(((size_t)b*L+l)*src_kv_max)*kvd, *cv=kv_v+(((size_t)b*L+l)*src_kv_max)*kvd;
            float *dk=kcp+(((size_t)l*B+b)*dkvm)*kvd, *dv=vcp+(((size_t)l*B+b)*dkvm)*kvd;
            for (size_t i=0;i<nper;++i){ union{uint32_t u;float f;}a,cc; a.u=(uint32_t)ck[i]<<16; dk[i]=a.f; cc.u=(uint32_t)cv[i]<<16; dv[i]=cc.f; }
        }
    }
}
void qwen_metal_talker_batch_free(void *st) {
    qwen_metal_talker_batch_t *s=st; if(!s) return;
    void *ps[]={s->kcache,s->vcache,s->x,s->xn,s->q,s->k,s->v,s->attn,s->proj,s->gate,s->gu,s->d_pos,s->rope_cos,s->rope_sin};
    for(unsigned i=0;i<sizeof(ps)/sizeof(ps[0]);++i){ void*p=ps[i]; if(p){ id o=(__bridge_transfer id)p; (void)o; } }
    free(s);
}

/* BATCHED CP body: same machinery as the Talker batch body but reads ctx->cp_layers and has NO
 * final norm — returns the residual x (caller applies cp_norm + lm-head + argmax per slot). Mirrors
 * cp_body_batch. Reuses qwen_metal_talker_batch_t/init/free (init from a single CP state = CP dims). */
static void enc_cp_batch_body(id<MTLComputeCommandEncoder> e, qwen_metal_talker_batch_t *s) {
    int B=s->B, H=s->H, qd=s->qd, kvd=s->kvd, nh=s->nh, nkv=s->nkv, inter=s->inter;
    float scale=1.0f/sqrtf((float)s->hd);
    id<MTLBuffer> x=MTB(s->x), xn=MTB(s->xn), qb=MTB(s->q), kb=MTB(s->k), vb=MTB(s->v);
    id<MTLBuffer> attnb=MTB(s->attn), projb=MTB(s->proj), gateb=MTB(s->gate), gub=MTB(s->gu);
    id<MTLBuffer> kc=MTB(s->kcache), vc=MTB(s->vcache);
    for (int l=0; l<s->L; ++l) {
        qwen_cp_layer_t *ly = &s->ctx->cp_layers[l];
        size_t kvoff = (size_t)l*B*s->kv_max*kvd*sizeof(float);
        enc_rmsf_b(e, s, x, ly->input_norm, xn, H);
        enc_mvb(e, s, ly->wq_bf16, ly->wq_int8, ly->wq_scale, ly->wq_q4, xn, qb, qd, H, 1);
        enc_mvb(e, s, ly->wk_bf16, ly->wk_int8, ly->wk_scale, ly->wk_q4, xn, kb, kvd, H, 1);
        enc_mvb(e, s, ly->wv_bf16, ly->wv_int8, ly->wv_scale, ly->wv_q4, xn, vb, kvd, H, 1);
        enc_rmsph_b(e, s, qb, ly->q_norm, nh, qd);
        enc_rmsph_b(e, s, kb, ly->k_norm, nkv, kvd);
        enc_ropeneox_b(e, s, qb, nh, qd);
        enc_ropeneox_b(e, s, kb, nkv, kvd);
        enc_trunc_b(e, s, kb, B*kvd);
        enc_trunc_b(e, s, vb, B*kvd);
        enc_kvstore_b(e, s, kc, vc, kvoff);
        enc_attnb(e, s, kc, vc, kvoff, scale);
        enc_mvb(e, s, ly->wo_bf16, ly->wo_int8, ly->wo_scale, ly->wo_q4, attnb, projb, H, qd, 1);
        enc_eadd_flat(e, s, x, projb, B*H);
        enc_rmsf_b(e, s, x, ly->post_attn_norm, xn, H);
        enc_mvb(e, s, ly->gate_up_fused_bf16, ly->gate_up_fused_int8, ly->gate_up_fused_scale, ly->gate_up_fused_q4, xn, gub, 2*inter, H, 1);
        enc_swiglu_ilb(e, s, gub, gateb);
        enc_mvb(e, s, ly->down_bf16, ly->down_int8, ly->down_scale, ly->down_q4, gateb, projb, H, inter, 1);
        enc_eadd_flat(e, s, x, projb, B*H);
    }
    /* NO final norm — the CP caller norms + lm-heads + argmaxes each slot. */
}
void *qwen_metal_cp_batch_init(void *single_talker_state, int B) {
    qwen_metal_talker_t *ss = single_talker_state; if (!ss || B<1 || B>QMB_MAX) return NULL;
    @autoreleasepool {
        qwen_metal_ctx *c = ss->mc; qwen_tts_ctx_t *ctx = ss->ctx; qwen_tts_config_t *cf = &ctx->config;
        qwen_metal_talker_batch_t *s = calloc(1, sizeof(*s));
        s->mc=c; s->ctx=ctx; s->B=B;
        s->H=cf->cp_hidden_size; s->nh=cf->cp_num_heads; s->nkv=cf->cp_num_kv_heads; s->hd=cf->cp_head_dim;
        s->inter=cf->cp_intermediate_size; s->L=cf->cp_num_layers; s->kv_max=ctx->cp_kv_max; s->eps=cf->rms_norm_eps;
        s->qd=s->nh*s->hd; s->kvd=s->nkv*s->hd;
        int H=s->H,qd=s->qd,kvd=s->kvd,inter=s->inter,L=s->L,kvm=s->kv_max, half=s->hd/2;
        id<MTLDevice> dev=(__bridge id<MTLDevice>)c->device;
        s->rope_cos=(__bridge_retained void*)[dev newBufferWithBytes:ctx->cp_rope_cos length:(size_t)kvm*half*sizeof(float) options:MTLResourceStorageModeShared];
        s->rope_sin=(__bridge_retained void*)[dev newBufferWithBytes:ctx->cp_rope_sin length:(size_t)kvm*half*sizeof(float) options:MTLResourceStorageModeShared];
        s->kcache=(__bridge_retained void*)mk_buf(c,(size_t)L*B*kvm*kvd*sizeof(float));
        s->vcache=(__bridge_retained void*)mk_buf(c,(size_t)L*B*kvm*kvd*sizeof(float));
        s->x=(__bridge_retained void*)mk_buf(c,(size_t)B*H*sizeof(float));
        s->xn=(__bridge_retained void*)mk_buf(c,(size_t)B*H*sizeof(float));
        s->q=(__bridge_retained void*)mk_buf(c,(size_t)B*qd*sizeof(float));
        s->k=(__bridge_retained void*)mk_buf(c,(size_t)B*kvd*sizeof(float));
        s->v=(__bridge_retained void*)mk_buf(c,(size_t)B*kvd*sizeof(float));
        s->attn=(__bridge_retained void*)mk_buf(c,(size_t)B*qd*sizeof(float));
        s->proj=(__bridge_retained void*)mk_buf(c,(size_t)B*H*sizeof(float));
        s->gate=(__bridge_retained void*)mk_buf(c,(size_t)B*inter*sizeof(float));
        s->gu=(__bridge_retained void*)mk_buf(c,(size_t)B*2*inter*sizeof(float));
        s->d_pos=(__bridge_retained void*)mk_buf(c,(size_t)B*sizeof(int));
        fprintf(stderr,"Metal CP BATCH ready (B=%d, %d layers, cp_hidden=%d)\n", B, L, H);
        return s;
    }
}
/* x=[B][cp_h] host residual in/out; pos_arr=[B] host (the CP pass position per slot). */
void qwen_metal_cp_batch_step(void *st, float *x, const int *pos_arr) {
    qwen_metal_talker_batch_t *s = st; if (!s) return;
    @autoreleasepool {
        qwen_metal_ctx *c=s->mc; id<MTLCommandQueue> q=(__bridge id<MTLCommandQueue>)c->queue;
        int B=s->B, H=s->H;
        id<MTLBuffer> xb=MTB(s->x), dpos=MTB(s->d_pos);
        memcpy(xb.contents, x, (size_t)B*H*sizeof(float));
        memcpy(dpos.contents, pos_arr, (size_t)B*sizeof(int));
        id<MTLCommandBuffer> cb=[q commandBuffer]; id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
        enc_cp_batch_body(e, s);
        [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
        memcpy(x, xb.contents, (size_t)B*H*sizeof(float));   /* residual out (NOT normed) */
    }
}
void qwen_metal_cp_batch_free(void *st) { qwen_metal_talker_batch_free(st); }

/* ======================================================================== *
 *  GPU-RESIDENT FUSED CODE PREDICTOR STEP (Metal, G2). Same machinery as the
 *  Talker (reuses enc_mv/enc_rms/enc_rms_ph/enc_rope + the resident kernels).
 *  CP: hidden=1024, 5 layers, per-frame KV (pos 0..15, overwritten each frame),
 *  NO final norm (caller applies cp_norm before the lm-head). Reuses qwen_metal_talker_t. */
void *qwen_metal_cp_init(void *metal_ctx, qwen_tts_ctx_t *ctx) {
    if (!ctx || !metal_ctx) return NULL;
    @autoreleasepool {
        qwen_metal_ctx *c = (qwen_metal_ctx *)metal_ctx;
        qwen_tts_config_t *cf = &ctx->config;
        qwen_metal_talker_t *s = calloc(1, sizeof(*s));
        s->mc = c; s->ctx = ctx;
        s->H = cf->cp_hidden_size; s->nh = cf->cp_num_heads; s->nkv = cf->cp_num_kv_heads; s->hd = cf->cp_head_dim;
        s->inter = cf->cp_intermediate_size; s->L = cf->cp_num_layers; s->kv_max = ctx->cp_kv_max;
        s->qd = s->nh*s->hd; s->kvd = s->nkv*s->hd; s->eps = cf->rms_norm_eps;
        int half = s->hd/2;
        id<MTLDevice> dev = (__bridge id<MTLDevice>)c->device;
        s->rope_cos = (__bridge_retained void *)[dev newBufferWithBytes:ctx->cp_rope_cos length:(size_t)s->kv_max*half*sizeof(float) options:MTLResourceStorageModeShared];
        s->rope_sin = (__bridge_retained void *)[dev newBufferWithBytes:ctx->cp_rope_sin length:(size_t)s->kv_max*half*sizeof(float) options:MTLResourceStorageModeShared];
        s->kcache = (__bridge_retained void *)mk_buf(c,(size_t)s->L*s->kv_max*s->kvd*sizeof(float));
        s->vcache = (__bridge_retained void *)mk_buf(c,(size_t)s->L*s->kv_max*s->kvd*sizeof(float));
        s->xb=(__bridge_retained void*)mk_buf(c,(size_t)s->H*sizeof(float));   s->xnb=(__bridge_retained void*)mk_buf(c,(size_t)s->H*sizeof(float));
        s->qb=(__bridge_retained void*)mk_buf(c,(size_t)s->qd*sizeof(float));  s->kb=(__bridge_retained void*)mk_buf(c,(size_t)s->kvd*sizeof(float));
        s->vb=(__bridge_retained void*)mk_buf(c,(size_t)s->kvd*sizeof(float)); s->attnb=(__bridge_retained void*)mk_buf(c,(size_t)s->qd*sizeof(float));
        s->projb=(__bridge_retained void*)mk_buf(c,(size_t)s->H*sizeof(float));s->gateb=(__bridge_retained void*)mk_buf(c,(size_t)s->inter*sizeof(float));
        s->gub=(__bridge_retained void*)mk_buf(c,(size_t)2*s->inter*sizeof(float));
        fprintf(stderr,"Metal CP: resident fused step ready (%d layers, hidden=%d)\n", s->L, s->H);
        return s;
    }
}

/* Encode the 5-layer CP transformer into an existing encoder (no commit). xb = residual in/out. */
static void enc_cp_layers(id<MTLComputeCommandEncoder> e, qwen_metal_talker_t *s, uint32_t upos) {
    qwen_metal_ctx *c = s->mc;
    id<MTLBuffer> xb=MTB(s->xb), xnb=MTB(s->xnb), qb=MTB(s->qb), kb=MTB(s->kb), vb=MTB(s->vb);
    id<MTLBuffer> attnb=MTB(s->attnb), projb=MTB(s->projb), gateb=MTB(s->gateb), gub=MTB(s->gub);
    id<MTLBuffer> kc=MTB(s->kcache), vc=MTB(s->vcache);
    int H=s->H, qd=s->qd, kvd=s->kvd, hd=s->hd, nh=s->nh, nkv=s->nkv, inter=s->inter;
    float scale = 1.0f/sqrtf((float)hd);
    for (int l=0; l<s->L; ++l) {
        qwen_cp_layer_t *ly = &s->ctx->cp_layers[l];
        uint32_t kvbase = (uint32_t)((size_t)l*s->kv_max*kvd);
        enc_rms(e, s, xb, xnb, ly->input_norm, H, 1);
        enc_mv(e, s, ly->wq_bf16, ly->wq_int8, ly->wq_scale, ly->wq_q4, xnb, qb, qd, H, 0);
        enc_mv(e, s, ly->wk_bf16, ly->wk_int8, ly->wk_scale, ly->wk_q4, xnb, kb, kvd, H, 0);
        enc_mv(e, s, ly->wv_bf16, ly->wv_int8, ly->wv_scale, ly->wv_q4, xnb, vb, kvd, H, 1);
        enc_qnorm_rope(e, s, qb, ly->q_norm, nh, upos, 0);            /* fused: norm+rope Q */
        enc_knorm_rope_store(e, s, kb, vb, ly->k_norm, nkv, upos, kc, vc, kvbase); /* fused: norm+rope K + store K,V */
        { uint32_t unh=(uint32_t)nh, unkv=(uint32_t)nkv, uhd=(uint32_t)hd; [e setComputePipelineState:MTB(c->pso_attn_res)];
          [e setBuffer:qb offset:0 atIndex:0]; [e setBuffer:kc offset:0 atIndex:1]; [e setBuffer:vc offset:0 atIndex:2]; [e setBuffer:attnb offset:0 atIndex:3];
          [e setBytes:&unh length:4 atIndex:4]; [e setBytes:&unkv length:4 atIndex:5]; [e setBytes:&uhd length:4 atIndex:6];
          [e setBytes:&scale length:4 atIndex:7]; [e setBytes:&upos length:4 atIndex:8]; [e setBytes:&kvbase length:4 atIndex:9];
          [e dispatchThreadgroups:MTLSizeMake((NSUInteger)nh,1,1) threadsPerThreadgroup:MTLSizeMake((NSUInteger)hd,1,1)]; enc_bar(e); }
        enc_mv(e, s, ly->wo_bf16, ly->wo_int8, ly->wo_scale, ly->wo_q4, attnb, projb, H, qd, 1);
        enc_add_rms(e, s, xb, projb, ly->post_attn_norm, xnb, H);   /* fused: x+=proj, xn=rmsnorm(x) */
        enc_mv(e, s, ly->gate_up_fused_bf16, ly->gate_up_fused_int8, ly->gate_up_fused_scale, ly->gate_up_fused_q4, xnb, gub, 2*inter, H, 1);
        [e setComputePipelineState:MTB(c->pso_swiglu)]; [e setBuffer:gub offset:0 atIndex:0]; [e setBuffer:gateb offset:0 atIndex:1];
        { NSUInteger tp=(NSUInteger)inter<256?(NSUInteger)inter:256; [e dispatchThreadgroups:MTLSizeMake(((NSUInteger)inter+tp-1)/tp,1,1) threadsPerThreadgroup:MTLSizeMake(tp,1,1)]; } enc_bar(e);
        enc_mv(e, s, ly->down_bf16, ly->down_int8, ly->down_scale, ly->down_q4, gateb, projb, H, inter, 1);
        [e setComputePipelineState:MTB(c->pso_eadd_ip)]; [e setBuffer:xb offset:0 atIndex:0]; [e setBuffer:projb offset:0 atIndex:1];
        { NSUInteger tp=(NSUInteger)H<256?(NSUInteger)H:256; [e dispatchThreadgroups:MTLSizeMake(((NSUInteger)H+tp-1)/tp,1,1) threadsPerThreadgroup:MTLSizeMake(tp,1,1)]; } enc_bar(e);
    }
}

/* x[cp_h] in/out (residual stream; caller norms). pos = the CP pass position (0..15). */
void qwen_metal_cp_step(void *st, float *x, int pos) {
    qwen_metal_talker_t *s = st; if (!s) return;
    @autoreleasepool {
        qwen_metal_ctx *c = s->mc;
        id<MTLCommandQueue> q = (__bridge id<MTLCommandQueue>)c->queue;
        id<MTLBuffer> xb=MTB(s->xb), xnb=MTB(s->xnb), qb=MTB(s->qb), kb=MTB(s->kb), vb=MTB(s->vb);
        id<MTLBuffer> attnb=MTB(s->attnb), projb=MTB(s->projb), gateb=MTB(s->gateb), gub=MTB(s->gub);
        id<MTLBuffer> kc=MTB(s->kcache), vc=MTB(s->vcache);
        memcpy(xb.contents, x, (size_t)s->H*sizeof(float));
        int H=s->H, qd=s->qd, kvd=s->kvd, hd=s->hd, nh=s->nh, nkv=s->nkv, inter=s->inter;
        float scale = 1.0f/sqrtf((float)hd); uint32_t upos=(uint32_t)pos;
        static int cprof=-1; if(cprof<0) cprof=getenv("QWEN_METAL_PROFILE")?1:0;
        static double c_enc=0,c_gpu=0; static long cnp=0; double ce0 = cprof?nowms():0;
        id<MTLCommandBuffer> cb = [q commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        for (int l=0; l<s->L; ++l) {
            qwen_cp_layer_t *ly = &s->ctx->cp_layers[l];
            uint32_t kvbase = (uint32_t)((size_t)l*s->kv_max*kvd);
            enc_rms(e, s, xb, xnb, ly->input_norm, H, 1);
            /* q/k/v read the same xnb, write different buffers → NO barrier between (overlap) */
            enc_mv(e, s, ly->wq_bf16, ly->wq_int8, ly->wq_scale, ly->wq_q4, xnb, qb, qd, H, 0);
            enc_mv(e, s, ly->wk_bf16, ly->wk_int8, ly->wk_scale, ly->wk_q4, xnb, kb, kvd, H, 0);
            enc_mv(e, s, ly->wv_bf16, ly->wv_int8, ly->wv_scale, ly->wv_q4, xnb, vb, kvd, H, 1);
            enc_rms_ph(e, s, qb, ly->q_norm, nh, 0);
            enc_rms_ph(e, s, kb, ly->k_norm, nkv, 1);
            enc_rope(e, s, qb, nh, upos, 0);
            enc_rope(e, s, kb, nkv, upos, 1);
            { uint32_t ukvd=(uint32_t)kvd;
              [e setComputePipelineState:MTB(c->pso_kv_store)];
              [e setBuffer:kc offset:0 atIndex:0]; [e setBuffer:vc offset:0 atIndex:1];
              [e setBuffer:kb offset:0 atIndex:2]; [e setBuffer:vb offset:0 atIndex:3];
              [e setBytes:&ukvd length:4 atIndex:4]; [e setBytes:&upos length:4 atIndex:5]; [e setBytes:&kvbase length:4 atIndex:6];
              NSUInteger tp=(NSUInteger)kvd<256?(NSUInteger)kvd:256;
              [e dispatchThreadgroups:MTLSizeMake(((NSUInteger)kvd+tp-1)/tp,1,1) threadsPerThreadgroup:MTLSizeMake(tp,1,1)]; enc_bar(e); }
            { uint32_t unh=(uint32_t)nh, unkv=(uint32_t)nkv, uhd=(uint32_t)hd;
              [e setComputePipelineState:MTB(c->pso_attn_res)];
              [e setBuffer:qb offset:0 atIndex:0]; [e setBuffer:kc offset:0 atIndex:1]; [e setBuffer:vc offset:0 atIndex:2];
              [e setBuffer:attnb offset:0 atIndex:3];
              [e setBytes:&unh length:4 atIndex:4]; [e setBytes:&unkv length:4 atIndex:5]; [e setBytes:&uhd length:4 atIndex:6];
              [e setBytes:&scale length:4 atIndex:7]; [e setBytes:&upos length:4 atIndex:8]; [e setBytes:&kvbase length:4 atIndex:9];
              [e dispatchThreadgroups:MTLSizeMake((NSUInteger)nh,1,1) threadsPerThreadgroup:MTLSizeMake((NSUInteger)hd,1,1)]; enc_bar(e); }
            enc_mv(e, s, ly->wo_bf16, ly->wo_int8, ly->wo_scale, ly->wo_q4, attnb, projb, H, qd, 1);
            [e setComputePipelineState:MTB(c->pso_eadd_ip)];
            [e setBuffer:xb offset:0 atIndex:0]; [e setBuffer:projb offset:0 atIndex:1];
            { NSUInteger tp=(NSUInteger)H<256?(NSUInteger)H:256; [e dispatchThreadgroups:MTLSizeMake(((NSUInteger)H+tp-1)/tp,1,1) threadsPerThreadgroup:MTLSizeMake(tp,1,1)]; } enc_bar(e);
            enc_rms(e, s, xb, xnb, ly->post_attn_norm, H, 1);
            enc_mv(e, s, ly->gate_up_fused_bf16, ly->gate_up_fused_int8, ly->gate_up_fused_scale, ly->gate_up_fused_q4, xnb, gub, 2*inter, H, 1);
            [e setComputePipelineState:MTB(c->pso_swiglu)];
            [e setBuffer:gub offset:0 atIndex:0]; [e setBuffer:gateb offset:0 atIndex:1];
            { NSUInteger tp=(NSUInteger)inter<256?(NSUInteger)inter:256; [e dispatchThreadgroups:MTLSizeMake(((NSUInteger)inter+tp-1)/tp,1,1) threadsPerThreadgroup:MTLSizeMake(tp,1,1)]; } enc_bar(e);
            enc_mv(e, s, ly->down_bf16, ly->down_int8, ly->down_scale, ly->down_q4, gateb, projb, H, inter, 1);
            [e setComputePipelineState:MTB(c->pso_eadd_ip)];
            [e setBuffer:xb offset:0 atIndex:0]; [e setBuffer:projb offset:0 atIndex:1];
            { NSUInteger tp=(NSUInteger)H<256?(NSUInteger)H:256; [e dispatchThreadgroups:MTLSizeMake(((NSUInteger)H+tp-1)/tp,1,1) threadsPerThreadgroup:MTLSizeMake(tp,1,1)]; } enc_bar(e);
        }
        [e endEncoding];
        if(cprof) c_enc += nowms()-ce0;
        double cg0 = cprof?nowms():0;
        static int cnosync=-1; if(cnosync<0) cnosync=getenv("QWEN_METAL_CP_NOSYNC")?1:0;
        [cb commit]; if(!cnosync) [cb waitUntilCompleted];   /* PROBE: skip wait to measure the sync ceiling */
        if(cprof){ c_gpu += nowms()-cg0; if(++cnp%800==0)
            fprintf(stderr,"[MTL-CP-PROF] per-pass: encode=%.3f ms | gpu(commit+wait)=%.3f ms | total=%.3f ms (x16/frame=%.1f ms/f)\n",
                    c_enc/cnp, c_gpu/cnp, (c_enc+c_gpu)/cnp, 16*(c_enc+c_gpu)/cnp); }
        memcpy(x, xb.contents, (size_t)H*sizeof(float));   /* residual, NOT normed */
    }
}

void qwen_metal_cp_free(void *st) { qwen_metal_talker_free(st); }

/* ======================================================================== *
 *  DEVICE-FRAME CP (Metal): the whole 16-pass RVQ loop + argmax + embed on GPU,
 *  ONE command buffer / ONE wait per frame (vs 16 commit+wait). The M1 win — the
 *  CP was sync-round-trip-bound (measured: 16 waits ≈ 30 ms/f). Mirrors qwen_cp_predict. */
typedef struct {
    qwen_metal_talker_t *cp; qwen_tts_ctx_t *ctx; qwen_metal_ctx *mc;
    int cp_h, emb_dim, h, codebook, cvocab, has_proj, lm_prec;
    void *emb_buf, *normed, *logits, *codes_dev;
} qwen_metal_cpframe_t;

void *qwen_metal_cp_frame_init(void *metal_ctx, qwen_tts_ctx_t *ctx) {
    if (!ctx || !metal_ctx) return NULL;
    if (!ctx->cp_lm_head_int8[0] && !ctx->cp_lm_head_q4[0] && !ctx->cp_lm_head_bf16[0]) return NULL;
    @autoreleasepool {
        qwen_metal_ctx *c = (qwen_metal_ctx *)metal_ctx; qwen_tts_config_t *cf = &ctx->config;
        qwen_metal_cpframe_t *f = calloc(1, sizeof(*f));
        f->mc = c; f->ctx = ctx; f->cp_h = cf->cp_hidden_size; f->emb_dim = ctx->cp_emb_dim;
        f->h = cf->hidden_size; f->codebook = cf->codebook_size; f->cvocab = cf->codec_vocab_size;
        f->has_proj = (ctx->cp_mtp_proj_bf16 != NULL);
        f->lm_prec = ctx->cp_lm_head_q4[0] ? 2 : (ctx->cp_lm_head_int8[0] ? 1 : 0);
        f->cp = (qwen_metal_talker_t *)qwen_metal_cp_init(metal_ctx, ctx);
        if (!f->cp) { free(f); return NULL; }
        int emax = f->emb_dim > f->h ? f->emb_dim : f->h;
        f->emb_buf = (__bridge_retained void *)mk_buf(c, (size_t)emax*sizeof(float));
        f->normed  = (__bridge_retained void *)mk_buf(c, (size_t)f->cp_h*sizeof(float));
        f->logits  = (__bridge_retained void *)mk_buf(c, (size_t)f->codebook*sizeof(float));
        f->codes_dev = (__bridge_retained void *)mk_buf(c, 16*sizeof(int));
        fprintf(stderr, "Metal CP device-frame ready (proj=%s, lm=%s, 1 sync/frame)\n",
                f->has_proj?"yes":"identity", f->lm_prec==2?"q4":f->lm_prec==1?"int8":"bf16");
        return f;
    }
}

/* GPU embed→(proj)→cp_x seed for a pass; table bf16, code slot in codes_dev, vocab guard. */
static void enc_seed(id<MTLComputeCommandEncoder> e, qwen_metal_cpframe_t *f,
                     const uint16_t *table, uint32_t cslot, uint32_t vocab, int dim) {
    qwen_metal_ctx *c = f->mc; qwen_metal_talker_t *s = f->cp;
    id<MTLBuffer> emb = MTB(f->emb_buf), cpx = MTB(s->xb), codes = MTB(f->codes_dev);
    uint32_t ud=(uint32_t)dim;
    [e setComputePipelineState:MTB(c->pso_embed_gather)];
    [e setBuffer:weight_buf(c,table,(size_t)vocab*dim*sizeof(uint16_t)) offset:0 atIndex:0];
    [e setBuffer:codes offset:0 atIndex:1]; [e setBuffer:emb offset:0 atIndex:2];
    [e setBytes:&ud length:4 atIndex:3]; [e setBytes:&cslot length:4 atIndex:4]; [e setBytes:&vocab length:4 atIndex:5];
    { NSUInteger tp=(NSUInteger)dim<256?(NSUInteger)dim:256; [e dispatchThreadgroups:MTLSizeMake(((NSUInteger)dim+tp-1)/tp,1,1) threadsPerThreadgroup:MTLSizeMake(tp,1,1)]; } enc_bar(e);
    if (f->has_proj) {
        enc_mv(e, s, f->ctx->cp_mtp_proj_bf16, NULL, NULL, NULL, emb, cpx, f->cp_h, f->emb_dim, 0);
        if (f->ctx->cp_mtp_proj_bias) {
            [e setComputePipelineState:MTB(c->pso_eadd_ip)]; [e setBuffer:cpx offset:0 atIndex:0];
            [e setBuffer:weight_buf(c,f->ctx->cp_mtp_proj_bias,(size_t)f->cp_h*sizeof(float)) offset:0 atIndex:1];
            NSUInteger tp=(NSUInteger)f->cp_h<256?(NSUInteger)f->cp_h:256; [e dispatchThreadgroups:MTLSizeMake(((NSUInteger)f->cp_h+tp-1)/tp,1,1) threadsPerThreadgroup:MTLSizeMake(tp,1,1)];
        }
        enc_bar(e);
    } else {
        [e setComputePipelineState:MTB(c->pso_copy_vec)]; [e setBuffer:emb offset:0 atIndex:0]; [e setBuffer:cpx offset:0 atIndex:1];
        NSUInteger tp=(NSUInteger)f->cp_h<256?(NSUInteger)f->cp_h:256; [e dispatchThreadgroups:MTLSizeMake(((NSUInteger)f->cp_h+tp-1)/tp,1,1) threadsPerThreadgroup:MTLSizeMake(tp,1,1)]; enc_bar(e);
    }
}
/* rmsnorm(cp_x → normed, cp_norm) + lm_head[g] matvec → logits + argmax → codes_dev[oslot]. */
static void enc_predict(id<MTLComputeCommandEncoder> e, qwen_metal_cpframe_t *f, int g, uint32_t oslot) {
    qwen_metal_ctx *c = f->mc; qwen_metal_talker_t *s = f->cp;
    id<MTLBuffer> normed=MTB(f->normed), logits=MTB(f->logits), codes=MTB(f->codes_dev);
    enc_rms(e, s, MTB(s->xb), normed, f->ctx->cp_norm, f->cp_h, 1);
    enc_mv(e, s, f->ctx->cp_lm_head_bf16[g], f->ctx->cp_lm_head_int8[g], f->ctx->cp_lm_head_scale[g],
           f->ctx->cp_lm_head_q4[g], normed, logits, f->codebook, f->cp_h, 1);
    uint32_t rows=(uint32_t)f->codebook, os=oslot;
    [e setComputePipelineState:MTB(c->pso_argmax)];
    [e setBuffer:logits offset:0 atIndex:0]; [e setBuffer:codes offset:0 atIndex:1];
    [e setBytes:&rows length:4 atIndex:2]; [e setBytes:&os length:4 atIndex:3];
    [e dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)]; enc_bar(e);
}

/* Whole frame: talker_hidden + code0 → out_codes[15], 1 command buffer, 1 wait. */
void qwen_metal_cp_frame(void *st, const float *talker_hidden, int code0, int *out_codes) {
    qwen_metal_cpframe_t *f = st; if (!f) return;
    @autoreleasepool {
        qwen_metal_ctx *c = f->mc; qwen_metal_talker_t *s = f->cp;
        id<MTLCommandQueue> q = (__bridge id<MTLCommandQueue>)c->queue;
        id<MTLBuffer> emb=MTB(f->emb_buf), cpx=MTB(s->xb), codes=MTB(f->codes_dev);
        ((int *)codes.contents)[0] = code0;   /* codes_dev[0] = code0 (from the Talker) */
        static int cprof=-1; if(cprof<0) cprof=getenv("QWEN_METAL_PROFILE")?1:0;
        static double t=0; static long n=0; double t0=cprof?nowms():0;
        id<MTLCommandBuffer> cb = [q commandBuffer]; id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        /* pass 0: project talker_hidden → cp_x, transformer(pos 0) */
        memcpy(emb.contents, talker_hidden, (size_t)f->h*sizeof(float));
        if (f->has_proj) {
            enc_mv(e, s, f->ctx->cp_mtp_proj_bf16, NULL, NULL, NULL, emb, cpx, f->cp_h, f->emb_dim, 0);
            if (f->ctx->cp_mtp_proj_bias) { [e setComputePipelineState:MTB(c->pso_eadd_ip)]; [e setBuffer:cpx offset:0 atIndex:0];
                [e setBuffer:weight_buf(c,f->ctx->cp_mtp_proj_bias,(size_t)f->cp_h*sizeof(float)) offset:0 atIndex:1];
                NSUInteger tp=(NSUInteger)f->cp_h<256?(NSUInteger)f->cp_h:256; [e dispatchThreadgroups:MTLSizeMake(((NSUInteger)f->cp_h+tp-1)/tp,1,1) threadsPerThreadgroup:MTLSizeMake(tp,1,1)]; }
            enc_bar(e);
        } else { [e setComputePipelineState:MTB(c->pso_copy_vec)]; [e setBuffer:emb offset:0 atIndex:0]; [e setBuffer:cpx offset:0 atIndex:1];
            NSUInteger tp=(NSUInteger)f->cp_h<256?(NSUInteger)f->cp_h:256; [e dispatchThreadgroups:MTLSizeMake(((NSUInteger)f->cp_h+tp-1)/tp,1,1) threadsPerThreadgroup:MTLSizeMake(tp,1,1)]; enc_bar(e); }
        enc_cp_layers(e, s, 0);
        /* 15 codebooks: g=0 embeds code0 via TALKER codec emb; g>=1 via cp_codec_emb[g-1] */
        for (int g = 0; g < 15; g++) {
            if (g == 0) enc_seed(e, f, f->ctx->codec_embedding_bf16, 0, (uint32_t)f->cvocab, f->h);
            else        enc_seed(e, f, f->ctx->cp_codec_emb_bf16[g-1], (uint32_t)g, (uint32_t)f->codebook, f->emb_dim);
            enc_cp_layers(e, s, (uint32_t)(g+1));
            enc_predict(e, f, g, (uint32_t)(g+1));   /* argmax → codes_dev[g+1] = out[g] */
        }
        [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
        if(cprof){ t+=nowms()-t0; if(++n%50==0) fprintf(stderr,"[MTL-CPFRAME] %.2f ms/frame (1 sync)\n", t/n); }
        int *cd = (int *)codes.contents; for (int g=0; g<15; g++) out_codes[g] = cd[g+1];
    }
}
void qwen_metal_cp_frame_free(void *st) {
    qwen_metal_cpframe_t *f = st; if (!f) return;
    qwen_metal_cp_free(f->cp);
    void *ps[] = {f->emb_buf, f->normed, f->logits, f->codes_dev};
    for (unsigned i=0;i<4;++i){ void *p=ps[i]; if(p){ id o=(__bridge_transfer id)p; (void)o; } }
    free(f);
}
