/*
 * qwen_tts_metal.metal - Metal compute shaders for Qwen3-TTS
 *
 * Full transformer inference on Apple Silicon GPU.
 * Kernels: bf16 matvec, RMSNorm, RoPE, attention, SwiGLU, residual add.
 * Uses simdgroup reduction for efficient per-row dot products.
 */

#include <metal_stdlib>
using namespace metal;

/* ========================================================================
 * Utility functions
 * ======================================================================== */

static inline float bf16_to_f32(ushort bf) {
    return as_type<float>((uint(bf)) << 16);
}

static inline ushort f32_to_bf16(float v) {
    return ushort(as_type<uint>(v) >> 16);
}

/* ========================================================================
 * Parameter structs
 * ======================================================================== */

struct matvec_params {
    int rows;
    int cols;
};

struct rmsnorm_params {
    int dim;
    float eps;
};

struct rmsnorm_perhead_params {
    int n_heads;
    int head_dim;
    float eps;
};

struct rope_params {
    int n_heads;
    int head_dim;
    int pos;        /* current position for cos/sin table lookup */
};

struct kv_store_params {
    int kv_dim;
    int offset;     /* pos * kv_dim (in ushort units) */
};

struct attention_params {
    int num_heads;
    int num_kv_heads;
    int head_dim;
    int kv_dim;     /* num_kv_heads * head_dim */
    int seq_len;    /* pos + 1 */
    float scale;    /* 1/sqrt(head_dim) */
};

/* ========================================================================
 * bf16 matvec: y[rows] = W_bf16[rows, cols] @ x[cols]
 *
 * Each simdgroup (32 threads) computes one output row.
 * Grid:  [rows * 32, 1, 1]
 * Group: [32, 1, 1]
 * ======================================================================== */

kernel void matvec_bf16(
    device const ushort *W     [[buffer(0)]],
    device const float  *x     [[buffer(1)]],
    device float        *y     [[buffer(2)]],
    constant matvec_params &p  [[buffer(3)]],
    uint tid                   [[thread_position_in_grid]],
    uint lane                  [[thread_index_in_simdgroup]])
{
    int row = (int)(tid / 32);
    if (row >= p.rows) return;

    int cols = p.cols;
    device const ushort *w_row = W + (long)row * cols;

    float acc = 0.0f;
    for (int c = (int)lane; c < cols; c += 32) {
        acc += bf16_to_f32(w_row[c]) * x[c];
    }

    acc = simd_sum(acc);

    if (lane == 0) {
        y[row] = acc;
    }
}

/* ========================================================================
 * RMSNorm: out[dim] = x[dim] * rsqrt(mean(x^2) + eps) * weight[dim]
 *
 * Single simdgroup (32 threads), strided access.
 * Grid: [32, 1, 1], Group: [32, 1, 1]
 * ======================================================================== */

kernel void rmsnorm(
    device const float *x      [[buffer(0)]],
    device float       *out    [[buffer(1)]],
    device const float *weight [[buffer(2)]],
    constant rmsnorm_params &p [[buffer(3)]],
    uint lane                  [[thread_index_in_simdgroup]])
{
    float sum_sq = 0.0f;
    for (int i = (int)lane; i < p.dim; i += 32) {
        float v = x[i];
        sum_sq += v * v;
    }
    sum_sq = simd_sum(sum_sq);

    float inv_rms = rsqrt(sum_sq / float(p.dim) + p.eps);

    for (int i = (int)lane; i < p.dim; i += 32) {
        out[i] = x[i] * inv_rms * weight[i];
    }
}

/* ========================================================================
 * Per-head RMSNorm (in-place):
 * For each head h: x[h*hd..(h+1)*hd] = normalize(x[h*hd..]) * weight[0..hd]
 *
 * One simdgroup per head, strided access within head.
 * Grid: [n_heads * 32, 1, 1], Group: [32, 1, 1]
 * ======================================================================== */

kernel void rmsnorm_perhead(
    device float       *x      [[buffer(0)]],
    device const float *weight [[buffer(1)]],
    constant rmsnorm_perhead_params &p [[buffer(2)]],
    uint tid                   [[thread_position_in_grid]],
    uint lane                  [[thread_index_in_simdgroup]])
{
    int head = (int)(tid / 32);
    if (head >= p.n_heads) return;

    device float *xh = x + head * p.head_dim;

    float sum_sq = 0.0f;
    for (int i = (int)lane; i < p.head_dim; i += 32) {
        float v = xh[i];
        sum_sq += v * v;
    }
    sum_sq = simd_sum(sum_sq);

    float inv_rms = rsqrt(sum_sq / float(p.head_dim) + p.eps);

    for (int i = (int)lane; i < p.head_dim; i += 32) {
        xh[i] *= inv_rms * weight[i];
    }
}

/* ========================================================================
 * NeoX split-half RoPE (in-place):
 * For each head: x[i] = x1*cos - x2*sin, x[i+half] = x2*cos + x1*sin
 *
 * One thread per (head, dim_pair). Thread handles one rotation pair.
 * Grid: [n_heads * half_dim, 1, 1], Group: [64, 1, 1]
 * ======================================================================== */

kernel void rope_neox(
    device float       *x         [[buffer(0)]],
    device const float *cos_table [[buffer(1)]],   /* [max_pos * half_dim] */
    device const float *sin_table [[buffer(2)]],
    constant rope_params &p       [[buffer(3)]],
    uint tid                      [[thread_position_in_grid]])
{
    int half_dim = p.head_dim / 2;
    int head = (int)tid / half_dim;
    int i = (int)tid % half_dim;
    if (head >= p.n_heads) return;

    device float *xh = x + head * p.head_dim;
    int table_idx = p.pos * half_dim + i;

    float x1 = xh[i];
    float x2 = xh[i + half_dim];
    float c = cos_table[table_idx];
    float s = sin_table[table_idx];

    xh[i]            = x1 * c - x2 * s;
    xh[i + half_dim] = x2 * c + x1 * s;
}

/* ========================================================================
 * KV store: f32 → bf16, write K and V to cache at given offset
 *
 * Grid: [kv_dim, 1, 1], Group: [256, 1, 1]
 * ======================================================================== */

kernel void kv_store_bf16(
    device const float *k      [[buffer(0)]],
    device const float *v      [[buffer(1)]],
    device ushort      *cache_k [[buffer(2)]],
    device ushort      *cache_v [[buffer(3)]],
    constant kv_store_params &p [[buffer(4)]],
    uint tid                   [[thread_position_in_grid]])
{
    if ((int)tid >= p.kv_dim) return;
    cache_k[p.offset + tid] = f32_to_bf16(k[tid]);
    cache_v[p.offset + tid] = f32_to_bf16(v[tid]);
}

/* ========================================================================
 * GQA Attention with bf16 KV cache (single-token decode)
 *
 * Online softmax (no extra storage for scores).
 * One threadgroup (1 simdgroup = 32 threads) per Q head.
 * Each thread handles head_dim/32 output dimensions with strided access.
 *
 * Dispatch: threadgroups=[num_heads,1,1], threadsPerThreadgroup=[32,1,1]
 * ======================================================================== */

kernel void attention_bf16kv(
    device const float  *q       [[buffer(0)]],   /* [num_heads * head_dim] */
    device const ushort *cache_k [[buffer(1)]],   /* [kv_max * kv_dim] bf16, this layer */
    device const ushort *cache_v [[buffer(2)]],   /* [kv_max * kv_dim] bf16, this layer */
    device float        *out     [[buffer(3)]],   /* [num_heads * head_dim] */
    constant attention_params &p [[buffer(4)]],
    uint head_id                 [[threadgroup_position_in_grid]],
    uint lane                    [[thread_index_in_simdgroup]])
{
    int kv_head = (int)head_id * p.num_kv_heads / p.num_heads;
    int hd = p.head_dim;

    device const float *q_head = q + (int)head_id * hd;
    int dims_per_lane = hd / 32;

    /* Online softmax with V accumulation */
    float m = -INFINITY;
    float d_sum = 0.0f;

    /* Per-lane V accumulators (supports head_dim up to 256) */
    float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};

    for (int pos = 0; pos < p.seq_len; pos++) {
        /* Dot product: q . k_pos */
        float dot = 0.0f;
        device const ushort *k_pos = cache_k + (long)pos * p.kv_dim + kv_head * hd;
        for (int d = (int)lane; d < hd; d += 32) {
            dot += q_head[d] * bf16_to_f32(k_pos[d]);
        }
        float s = simd_sum(dot) * p.scale;

        /* Online softmax update */
        float m_new = max(m, s);
        float correction = exp(m - m_new);
        float weight = exp(s - m_new);
        d_sum = d_sum * correction + weight;

        /* V accumulation with correction */
        device const ushort *v_pos = cache_v + (long)pos * p.kv_dim + kv_head * hd;
        for (int i = 0; i < dims_per_lane; i++) {
            int d = (int)lane + i * 32;
            acc[i] = acc[i] * correction + weight * bf16_to_f32(v_pos[d]);
        }

        m = m_new;
    }

    /* Write output: acc / d_sum */
    float inv_d = (d_sum > 0.0f) ? (1.0f / d_sum) : 0.0f;
    for (int i = 0; i < dims_per_lane; i++) {
        int d_off = (int)lane + i * 32;
        out[(int)head_id * hd + d_off] = acc[i] * inv_d;
    }
}

/* ========================================================================
 * SwiGLU: out[i] = silu(gate_up[2i]) * gate_up[2i+1]
 *
 * Grid: [inter, 1, 1], Group: [256, 1, 1]
 * ======================================================================== */

kernel void swiglu(
    device const float *gate_up [[buffer(0)]],   /* [2 * inter] interleaved */
    device float       *out     [[buffer(1)]],   /* [inter] */
    constant int       &inter   [[buffer(2)]],
    uint tid                    [[thread_position_in_grid]])
{
    if ((int)tid >= inter) return;
    float g = gate_up[2 * tid];
    float u = gate_up[2 * tid + 1];
    out[tid] = g / (1.0f + exp(-g)) * u;
}

/* ========================================================================
 * Residual add: x[i] += proj[i]
 *
 * Grid: [dim, 1, 1], Group: [256, 1, 1]
 * ======================================================================== */

kernel void residual_add(
    device float       *x    [[buffer(0)]],
    device const float *proj [[buffer(1)]],
    constant int       &dim  [[buffer(2)]],
    uint tid                 [[thread_position_in_grid]])
{
    if ((int)tid >= dim) return;
    x[tid] += proj[tid];
}
