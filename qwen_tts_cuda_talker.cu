/*
 * qwen_tts_cuda_talker.cu — GPU-RESIDENT fused Talker step (CUDA fused-forward epic, M1).
 *
 * The per-op matvec hook (qwen_tts_cuda.c) uploads/computes/downloads PER matvec → a device
 * sync every op. This TU keeps WEIGHTS + KV + activations RESIDENT on the device and runs the
 * WHOLE Talker step as a chain of kernels with a SINGLE sync at the end. Only the input embed
 * goes in and the hidden comes out.
 *
 * KEY perf insight (measured on GB10): single-token decode is BANDWIDTH-BOUND on the weight
 * reads (1.7B ≈ 3.4 GB bf16 / step). fp32-resident weights read 2× the bytes → no speedup vs the
 * per-op path. So weights stay **bf16** on the device and a custom bf16 matvec reads them at 2
 * bytes/elem while keeping the ACTIVATION in fp32 — exactly the CPU's bf16-weight × f32-act
 * semantics (no extra precision loss, unlike cublasGemmEx which needs bf16 activations too).
 *
 * Kernels match the CPU semantics EXACTLY (qwen_tts_talker.c / qwen_tts_kernels.c):
 *   - matvec: bf16 weight (row-major [rows,cols]) × f32 activation, f32 accumulate  [k_matvec_bf16]
 *   - SwiGLU: interleaved gate/up (gate_up[2i], gate_up[2i+1])                        [k_swiglu_il]
 *   - RoPE: NeoX SPLIT-HALF (xh[i],xh[i+half]), NOT interleaved                       [k_rope_neox]
 *   - per-head RMSNorm on Q/K with a shared [head_dim] weight                         [k_rmsnorm_ph]
 *   - causal GQA attention, online softmax (flash-style)                             [k_attn]
 *   - bf16-TRUNCATED KV cache (CPU f32_to_bf16 = bits>>16 = truncation)              [k_trunc_bf16]
 */

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>   /* __half q4blk scale (matches q4_0_block_t fp16 layout) */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

extern "C" {
#include "qwen_tts.h"
}

#define CK(x) do { cudaError_t e_=(x); if(e_!=cudaSuccess){fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e_));} } while(0)
#define TPB 256
#define CEIL(a,b) (((a)+(b)-1)/(b))

/* ---- kernels (device pointers, NO per-op copy) -------------------------- */

/* y[rows] = W[rows,cols] @ x[cols].  W row-major bf16, x/y f32.  ONE WARP per output row:
 * 32 lanes stride over cols (coalesced bf16 reads), reduce via __shfl (no __syncthreads, no
 * shared mem). Bandwidth-efficient (each lane does cols/32 MACs) — approaches the DRAM limit,
 * unlike a 256-thread tree reduce where the reduction dominates the tiny per-thread work. */
__global__ void k_matvec_bf16(const __nv_bfloat16 *W, const float *x, float *y, int rows, int cols) {
    int row = (blockIdx.x*blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (row >= rows) return;
    const __nv_bfloat16 *wr = W + (size_t)row * cols;
    float s = 0.f;
    for (int i = lane; i < cols; i += 32) s += __bfloat162float(wr[i]) * x[i];
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) s += __shfl_down_sync(0xffffffffu, s, o);
    if (lane == 0) y[row] = s;
}

/* q4_0 block: 32 weights = fp16 scale + 16 bytes (low nibble=even idx, high=odd), val=(nib-8)*scale.
 * MUST match qwen_tts_kernels.h q4_0_block_t exactly (fp16 scale, 18 B/block since perf item 2). */
typedef struct { __half scale; unsigned char qs[16]; } q4blk;
/* warp-per-row q4_0 matvec: int4 weight (0.5 byte) × f32 activation. Half the bytes of int8. */
__global__ void k_matvec_q4_0(const q4blk *W, const float *x, float *y, int rows, int cols) {
    int row = (blockIdx.x*blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (row >= rows) return;
    int nb = cols >> 5;
    const q4blk *wr = W + (size_t)row * nb;
    float s = 0.f;
    for (int c = lane; c < cols; c += 32) {
        const q4blk *b = wr + (c>>5); int ic = c & 31;
        unsigned char byte = b->qs[ic>>1];
        int nib = (ic&1) ? (byte>>4) : (byte&0x0F);
        s += (float)(nib-8) * __half2float(b->scale) * x[c];
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) s += __shfl_down_sync(0xffffffffu, s, o);
    if (lane == 0) y[row] = s;
}

/* ── dp4a q4×q8 twin (rental-prep 2026-07-11, OPT-IN QWEN_CUDA_DP4A=1) ─────────
 * The gpu-accel-status "⏳ dp4a q4→q8_1" item: quantize the activation once per
 * matvec into per-32-block int8 (DEINTERLEAVED even/odd to match q4_0's within-
 * byte packing) + per-block scale/sum, then integer __dp4a dots with the -8
 * offset corrected via the block sum. NOTE qs sits at byte offset 2 of the
 * 18-byte block → words are built from BYTE loads (no misaligned uint32 reads).
 * ⚠ COMPILE-UNTESTED here (no nvcc on the M1 dev box) — default-off so the
 * proven f32-act kernel above stays the path; first build/validate on the
 * rented NVIDIA box (--gpu-selftest), then A/B with the env flag. */
__global__ void k_quant_act_q4dp(const float *x, int nb, signed char *qe, signed char *qo,
                                 float *sxb, int *sumb) {
    int b = blockIdx.x*blockDim.x + threadIdx.x; if (b >= nb) return;
    const float *xb = x + (size_t)b*32;
    float amax = 0.f;
    for (int i = 0; i < 32; i++) { float a = fabsf(xb[i]); if (a > amax) amax = a; }
    float sc  = amax > 0.f ? amax/127.f : 0.f;
    float inv = amax > 0.f ? 127.f/amax : 0.f;
    int sum = 0;
    for (int i = 0; i < 16; i++) {
        int e = (int)lrintf(xb[2*i]   * inv);
        int o = (int)lrintf(xb[2*i+1] * inv);
        e = max(-128, min(127, e)); o = max(-128, min(127, o));
        qe[(size_t)b*16 + i] = (signed char)e;
        qo[(size_t)b*16 + i] = (signed char)o;
        sum += e + o;
    }
    sxb[b] = sc; sumb[b] = sum;
}
__global__ void k_matvec_q4_0_dp4a(const q4blk *W, const signed char *qe, const signed char *qo,
                                   const float *sxb, const int *sumb,
                                   float *y, int rows, int cols) {
    int row = (blockIdx.x*blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (row >= rows) return;
    int nb = cols >> 5;
    const q4blk *wr = W + (size_t)row * nb;
    float s = 0.f;
    for (int b = lane; b < nb; b += 32) {
        const unsigned char *qs = wr[b].qs;
        const int *ie = (const int *)(qe + (size_t)b*16);
        const int *io = (const int *)(qo + (size_t)b*16);
        int t = 0;
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            unsigned int w = (unsigned int)qs[4*j] | ((unsigned int)qs[4*j+1] << 8)
                           | ((unsigned int)qs[4*j+2] << 16) | ((unsigned int)qs[4*j+3] << 24);
            t = __dp4a((int)(w & 0x0F0F0F0Fu),        ie[j], t);   /* even weights (lo nibbles) */
            t = __dp4a((int)((w >> 4) & 0x0F0F0F0Fu), io[j], t);   /* odd weights (hi nibbles) */
        }
        s += __half2float(wr[b].scale) * sxb[b] * (float)(t - 8*sumb[b]);
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) s += __shfl_down_sync(0xffffffffu, s, o);
    if (lane == 0) y[row] = s;
}

/* Same, int8 weight (1 byte) × f32 activation, per-row scale. Warp-per-row. */
__global__ void k_matvec_int8(const int8_t *W, const float *scale, const float *x, float *y, int rows, int cols) {
    int row = (blockIdx.x*blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (row >= rows) return;
    const int8_t *wr = W + (size_t)row * cols;
    float s = 0.f;
    for (int i = lane; i < cols; i += 32) s += (float)wr[i] * x[i];
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) s += __shfl_down_sync(0xffffffffu, s, o);
    if (lane == 0) y[row] = scale[row] * s;
}

__global__ void k_rmsnorm_full(const float *x, const float *w, float *y, int dim, float eps) {
    extern __shared__ float part[];
    int tid = threadIdx.x, tc = blockDim.x;
    float s = 0.f; for (int i=tid;i<dim;i+=tc) s += x[i]*x[i];
    part[tid]=s; __syncthreads();
    for (int st=tc/2; st>0; st>>=1){ if(tid<st) part[tid]+=part[tid+st]; __syncthreads(); }
    float inv = rsqrtf(part[0]/(float)dim + eps);
    for (int i=tid;i<dim;i+=tc) y[i] = x[i]*inv*w[i];
}

/* per-head RMSNorm: one block per head, weight w[head_dim] shared across heads. */
__global__ void k_rmsnorm_ph(float *x, const float *w, int head_dim, float eps) {
    extern __shared__ float part[];
    int h = blockIdx.x, tid = threadIdx.x, tc = blockDim.x;
    float *xh = x + (size_t)h*head_dim;
    float s = 0.f; for (int i=tid;i<head_dim;i+=tc) s += xh[i]*xh[i];
    part[tid]=s; __syncthreads();
    for (int st=tc/2; st>0; st>>=1){ if(tid<st) part[tid]+=part[tid+st]; __syncthreads(); }
    float inv = rsqrtf(part[0]/(float)head_dim + eps);
    for (int i=tid;i<head_dim;i+=tc) xh[i] = xh[i]*inv*w[i];
}

/* NeoX split-half RoPE. cos/sin BASE + device pos (so the launch is CUDA-graph-invariant). */
__global__ void k_rope_neox(float *x, const float *cos_base, const float *sin_base,
                            int n_heads, int head_dim, const int *d_pos) {
    int half = head_dim/2;
    int gid = blockIdx.x*blockDim.x + threadIdx.x;
    if (gid >= n_heads*half) return;
    const float *cosp = cos_base + (size_t)(*d_pos)*half, *sinp = sin_base + (size_t)(*d_pos)*half;
    int h = gid/half, i = gid%half;
    float *xh = x + (size_t)h*head_dim;
    float c = cosp[i], sn = sinp[i];
    float x1 = xh[i], x2 = xh[i+half];
    xh[i] = x1*c - x2*sn; xh[i+half] = x2*c + x1*sn;
}

/* store k,v (already bf16-truncated) into the layer's KV cache at device pos (graph-invariant). */
__global__ void k_kv_store(float *kc_layer, float *vc_layer, const float *k, const float *v,
                           int kvd, const int *d_pos) {
    int i = blockIdx.x*blockDim.x + threadIdx.x; if (i>=kvd) return;
    size_t off = (size_t)(*d_pos)*kvd + i;
    kc_layer[off] = k[i]; vc_layer[off] = v[i];
}

__global__ void k_swiglu_il(const float *in, float *out, int n) {
    int i = blockIdx.x*blockDim.x + threadIdx.x; if (i>=n) return;
    float g = in[2*i], u = in[2*i+1];
    out[i] = g/(1.f+expf(-g))*u;
}

__global__ void k_add_ip(float *a, const float *b, int n) {  /* a += b */
    int i = blockIdx.x*blockDim.x + threadIdx.x; if (i<n) a[i]+=b[i];
}

/* round-trip f32→bf16→f32 (truncate mantissa) to MATCH the CPU's bf16 KV cache (bits>>16). */
__global__ void k_trunc_bf16(float *x, int n) {
    int i = blockIdx.x*blockDim.x + threadIdx.x; if (i>=n) return;
    uint32_t u = __float_as_uint(x[i]);
    x[i] = __uint_as_float(u & 0xFFFF0000u);
}

/* causal GQA attention, online softmax (flash-style, single-pass). Q[1,n_heads,hd],
 * K/V[seq_k,n_kv,hd]. ONE block per head, blockDim = head_dim: thread t owns output dim t and
 * accumulates online (no materialized scores). Score per key = block-reduce of q[t]*k[t]. This
 * uses head_dim×n_heads threads vs the old 16-thread-total version. */
__global__ void k_attn(const float *Q, const float *K, const float *V, float *O,
                       int n_heads, int n_kv, int hd, float scale, const int *d_pos) {
    int h = blockIdx.x, t = threadIdx.x; if (h>=n_heads || t>=hd) return;
    int kvh = h/(n_heads/n_kv);
    int valid = (*d_pos)+1;
    float qt = Q[(size_t)h*hd + t];
    extern __shared__ float sh[];
    float m=-1e30f, denom=0.f, acc=0.f;
    for (int j=0;j<valid;++j){
        const float *k = K + ((size_t)j*n_kv+kvh)*hd;
        sh[t] = qt * k[t]; __syncthreads();
        for (int s=hd/2; s>0; s>>=1){ if(t<s) sh[t]+=sh[t+s]; __syncthreads(); }
        float score = sh[0]*scale; __syncthreads();
        float m_new = fmaxf(m, score);
        float corr = expf(m - m_new), p = expf(score - m_new);
        denom = denom*corr + p;
        acc = acc*corr + p * V[((size_t)j*n_kv+kvh)*hd + t];
        m = m_new;
    }
    O[(size_t)h*hd + t] = acc/denom;
}

/* ---- resident state ----------------------------------------------------- */

typedef struct {
    int hidden, q_dim, kv_dim, inter, n_heads, n_kv, head_dim, n_layers, kv_max;
    float eps;
    void **wq,**wk,**wv,**wo,**wgu,**wdn;             /* resident weights (bf16 or int8), per layer */
    float **wqs,**wks,**wvs,**wos,**wgus,**wdns;      /* per-row scales (NULL = bf16) */
    float **inorm,**pnorm,**qn,**kn;                  /* f32 norms, per layer */
    float *tnorm, *rope_cos, *rope_sin;
    float *kcache,*vcache;                           /* [n_layers*kv_max*kv_dim] f32 */
    float *x,*xn,*q,*k,*v,*attn,*proj,*gate,*gu;     /* work buffers */
    int prec;   /* 0=bf16 1=int8 2=q4_0 */
    int *d_pos;                                      /* device position (graph-invariant) */
    cudaGraphExec_t exec; int cap_ready;             /* CUDA graph of the 28-layer step */
} cuda_talker_t;

static __nv_bfloat16 *up_bf16(const uint16_t *w, size_t n) {
    __nv_bfloat16 *d=NULL; CK(cudaMalloc(&d,n*sizeof(__nv_bfloat16)));
    CK(cudaMemcpy(d,w,n*sizeof(uint16_t),cudaMemcpyHostToDevice));  /* bf16 bits == uint16 bits */
    return d;
}
static int8_t *up_int8(const int8_t *w, size_t n) {
    int8_t *d=NULL; CK(cudaMalloc(&d,n*sizeof(int8_t)));
    CK(cudaMemcpy(d,w,n*sizeof(int8_t),cudaMemcpyHostToDevice)); return d;
}
static float *up_f32(const float *w, size_t n) {
    float *d=NULL; CK(cudaMalloc(&d,n*sizeof(float)));
    CK(cudaMemcpy(d,w,n*sizeof(float),cudaMemcpyHostToDevice)); return d;
}
static void *up_q4(const void *w, size_t nblocks) {   /* q4_0 blocks (18 bytes each: fp16 scale), raw */
    void *d=NULL; size_t bytes=nblocks*sizeof(q4blk);
    CK(cudaMalloc(&d,bytes)); CK(cudaMemcpy(d,w,bytes,cudaMemcpyHostToDevice)); return d;
}
/* dp4a activation scratch (opt-in path; lazily allocated once, sized to the max col dim). */
enum { QDP_MAX_COLS = 16384 };
static signed char *g_qdp_qe = NULL, *g_qdp_qo = NULL;
static float *g_qdp_sx = NULL; static int *g_qdp_sum = NULL;
static int qdp_enabled(void) {
    static int on = -1;
    if (on < 0) {
        const char *e = getenv("QWEN_CUDA_DP4A");
        on = (e && e[0]=='1');
        if (on) {
            if (cudaMalloc(&g_qdp_qe,  QDP_MAX_COLS/2) != cudaSuccess ||
                cudaMalloc(&g_qdp_qo,  QDP_MAX_COLS/2) != cudaSuccess ||
                cudaMalloc(&g_qdp_sx,  (QDP_MAX_COLS/32)*sizeof(float)) != cudaSuccess ||
                cudaMalloc(&g_qdp_sum, (QDP_MAX_COLS/32)*sizeof(int))   != cudaSuccess) {
                fprintf(stderr, "[cuda] dp4a scratch alloc failed — falling back to f32-act q4 kernel\n");
                on = 0;
            } else {
                fprintf(stderr, "[cuda] q4_0 matvec: dp4a q4xq8 path ENABLED (QWEN_CUDA_DP4A=1, experimental)\n");
            }
        }
    }
    return on;
}
/* dispatch by precision (0=bf16, 1=int8, 2=q4_0). Warp-per-row: one warp per output row. */
static inline void mv(int prec, const void *W, const float *scale, const float *dX, float *dY, int rows, int cols) {
    int grid = CEIL(rows*32, TPB);
    if (prec==2 && cols <= QDP_MAX_COLS && qdp_enabled()) {
        int nb = cols >> 5;
        k_quant_act_q4dp<<<CEIL(nb,128),128>>>(dX, nb, g_qdp_qe, g_qdp_qo, g_qdp_sx, g_qdp_sum);
        k_matvec_q4_0_dp4a<<<grid,TPB>>>((const q4blk*)W, g_qdp_qe, g_qdp_qo, g_qdp_sx, g_qdp_sum, dY, rows, cols);
    }
    else if (prec==2) k_matvec_q4_0<<<grid,TPB>>>((const q4blk*)W, dX, dY, rows, cols);
    else if (prec==1) k_matvec_int8<<<grid,TPB>>>((const int8_t*)W, scale, dX, dY, rows, cols);
    else              k_matvec_bf16<<<grid,TPB>>>((const __nv_bfloat16*)W, dX, dY, rows, cols);
}
/* per-weight upload: q4_0 (0.5 byte) → int8 (1 byte) → bf16 (2 byte), whichever the engine quantized.
 * rows×cols weight; q4 has rows*(cols/32) blocks. Sets *pprec to the chosen precision. */
#define UPW(dst, dsts, w4, w8, w8s, wbf, rows, cols, pprec) do { \
    if (w4)      { (dst)=up_q4((const void*)(w4),(size_t)(rows)*((cols)/32)); (dsts)=NULL; *(pprec)=2; } \
    else if (w8) { (dst)=up_int8((w8),(size_t)(rows)*(cols)); (dsts)=up_f32((w8s),(rows)); *(pprec)=1; } \
    else         { (dst)=up_bf16((wbf),(size_t)(rows)*(cols)); (dsts)=NULL; *(pprec)=0; } } while(0)

extern "C" void *qwen_cuda_talker_init(qwen_tts_ctx_t *ctx) {
    qwen_tts_config_t *c=&ctx->config;
    cuda_talker_t *s=(cuda_talker_t*)calloc(1,sizeof(*s));
    s->hidden=c->hidden_size; s->n_heads=c->num_heads; s->n_kv=c->num_kv_heads;
    s->head_dim=c->head_dim; s->inter=c->intermediate_size; s->n_layers=c->num_layers;
    s->q_dim=c->num_heads*c->head_dim; s->kv_dim=c->num_kv_heads*c->head_dim;
    s->eps=c->rms_norm_eps; s->kv_max=ctx->kv_max;
    int L=s->n_layers, H=s->hidden, hd=s->head_dim, half=hd/2;
    s->wq=(void**)calloc(L,sizeof(void*)); s->wk=(void**)calloc(L,sizeof(void*));
    s->wv=(void**)calloc(L,sizeof(void*)); s->wo=(void**)calloc(L,sizeof(void*));
    s->wgu=(void**)calloc(L,sizeof(void*)); s->wdn=(void**)calloc(L,sizeof(void*));
    s->wqs=(float**)calloc(L,sizeof(float*)); s->wks=(float**)calloc(L,sizeof(float*));
    s->wvs=(float**)calloc(L,sizeof(float*)); s->wos=(float**)calloc(L,sizeof(float*));
    s->wgus=(float**)calloc(L,sizeof(float*)); s->wdns=(float**)calloc(L,sizeof(float*));
    s->inorm=(float**)calloc(L,sizeof(float*)); s->pnorm=(float**)calloc(L,sizeof(float*));
    s->qn=(float**)calloc(L,sizeof(float*)); s->kn=(float**)calloc(L,sizeof(float*));
    int used_int8=0;
    for (int l=0;l<L;++l){
        qwen_talker_layer_t *ly=&ctx->layers[l];
        if (!ly->wq_bf16 && !ly->wq_int8 && !ly->wq_q4){ fprintf(stderr,"CUDA talker: layer %d has no weights\n",l); return NULL; }
        UPW(s->wq[l], s->wqs[l], ly->wq_q4, ly->wq_int8, ly->wq_scale, ly->wq_bf16, s->q_dim, H, &s->prec);
        UPW(s->wk[l], s->wks[l], ly->wk_q4, ly->wk_int8, ly->wk_scale, ly->wk_bf16, s->kv_dim, H, &s->prec);
        UPW(s->wv[l], s->wvs[l], ly->wv_q4, ly->wv_int8, ly->wv_scale, ly->wv_bf16, s->kv_dim, H, &s->prec);
        UPW(s->wo[l], s->wos[l], ly->wo_q4, ly->wo_int8, ly->wo_scale, ly->wo_bf16, H, s->q_dim, &s->prec);
        UPW(s->wgu[l],s->wgus[l],ly->gate_up_fused_q4, ly->gate_up_fused_int8, ly->gate_up_fused_scale, ly->gate_up_fused_bf16, 2*s->inter, H, &s->prec);
        UPW(s->wdn[l],s->wdns[l],ly->down_q4, ly->down_int8, ly->down_scale, ly->down_bf16, H, s->inter, &s->prec);
        used_int8 = (s->prec==1);
        s->inorm[l]=up_f32(ly->input_norm,H); s->pnorm[l]=up_f32(ly->post_attn_norm,H);
        s->qn[l]=up_f32(ly->q_norm,hd); s->kn[l]=up_f32(ly->k_norm,hd);
    }
    s->tnorm=up_f32(ctx->talker_norm,H);
    s->rope_cos=up_f32(ctx->rope_cos,(size_t)s->kv_max*half);
    s->rope_sin=up_f32(ctx->rope_sin,(size_t)s->kv_max*half);
    CK(cudaMalloc(&s->kcache,(size_t)L*s->kv_max*s->kv_dim*sizeof(float)));
    CK(cudaMalloc(&s->vcache,(size_t)L*s->kv_max*s->kv_dim*sizeof(float)));
    CK(cudaMalloc(&s->x,H*sizeof(float)));  CK(cudaMalloc(&s->xn,H*sizeof(float)));
    CK(cudaMalloc(&s->q,s->q_dim*sizeof(float))); CK(cudaMalloc(&s->k,s->kv_dim*sizeof(float)));
    CK(cudaMalloc(&s->v,s->kv_dim*sizeof(float))); CK(cudaMalloc(&s->attn,s->q_dim*sizeof(float)));
    CK(cudaMalloc(&s->proj,H*sizeof(float))); CK(cudaMalloc(&s->gate,s->inter*sizeof(float)));
    CK(cudaMalloc(&s->gu,(size_t)2*s->inter*sizeof(float)));
    CK(cudaMalloc(&s->d_pos,sizeof(int)));
    fprintf(stderr,"CUDA talker: resident fused step ready (%d layers, hidden=%d, %s weights, CUDA graph)\n",L,H,s->prec==2?"q4_0":s->prec==1?"int8":"bf16");
    return s;
}


/* The 28-layer step body — pure kernel launches (read d_x/d_pos, write d_xn). Captured ONCE
 * into a CUDA graph; every op is pos-independent (rope/kv_store/attn read *d_pos device-side),
 * so the same graph replays for any position → ~420 launches/frame become one graph launch. */
static void talker_body(cuda_talker_t *s) {
    int H=s->hidden, qd=s->q_dim, kvd=s->kv_dim, hd=s->head_dim, half=hd/2;
    int nh=s->n_heads, nkv=s->n_kv, inter=s->inter;
    float scale=1.f/sqrtf((float)hd);
    for (int l=0;l<s->n_layers;++l){
        k_rmsnorm_full<<<1,TPB,TPB*sizeof(float)>>>(s->x,s->inorm[l],s->xn,H,s->eps);
        mv(s->prec,s->wq[l],s->wqs[l],s->xn,s->q,qd,H);
        mv(s->prec,s->wk[l],s->wks[l],s->xn,s->k,kvd,H);
        mv(s->prec,s->wv[l],s->wvs[l],s->xn,s->v,kvd,H);
        k_rmsnorm_ph<<<nh, TPB, TPB*sizeof(float)>>>(s->q,s->qn[l],hd,s->eps);
        k_rmsnorm_ph<<<nkv,TPB, TPB*sizeof(float)>>>(s->k,s->kn[l],hd,s->eps);
        k_rope_neox<<<CEIL(nh*half,TPB),TPB>>>(s->q,s->rope_cos,s->rope_sin,nh,hd,s->d_pos);
        k_rope_neox<<<CEIL(nkv*half,TPB),TPB>>>(s->k,s->rope_cos,s->rope_sin,nkv,hd,s->d_pos);
        k_trunc_bf16<<<CEIL(kvd,TPB),TPB>>>(s->k,kvd);
        k_trunc_bf16<<<CEIL(kvd,TPB),TPB>>>(s->v,kvd);
        float *Kl=s->kcache+(size_t)l*s->kv_max*kvd, *Vl=s->vcache+(size_t)l*s->kv_max*kvd;
        k_kv_store<<<CEIL(kvd,TPB),TPB>>>(Kl,Vl,s->k,s->v,kvd,s->d_pos);
        k_attn<<<nh,hd,hd*sizeof(float)>>>(s->q,Kl,Vl,s->attn,nh,nkv,hd,scale,s->d_pos);
        mv(s->prec,s->wo[l],s->wos[l],s->attn,s->proj,H,qd);
        k_add_ip<<<CEIL(H,TPB),TPB>>>(s->x,s->proj,H);
        k_rmsnorm_full<<<1,TPB,TPB*sizeof(float)>>>(s->x,s->pnorm[l],s->xn,H,s->eps);
        mv(s->prec,s->wgu[l],s->wgus[l],s->xn,s->gu,2*inter,H);
        k_swiglu_il<<<CEIL(inter,TPB),TPB>>>(s->gu,s->gate,inter);
        mv(s->prec,s->wdn[l],s->wdns[l],s->gate,s->proj,H,inter);
        k_add_ip<<<CEIL(H,TPB),TPB>>>(s->x,s->proj,H);
    }
    k_rmsnorm_full<<<1,TPB,TPB*sizeof(float)>>>(s->x,s->tnorm,s->xn,H,s->eps);
}

extern "C" void qwen_cuda_talker_step(void *st, const float *embed, float *hidden_out, int pos) {
    cuda_talker_t *s=(cuda_talker_t*)st;
    int H=s->hidden;
    CK(cudaMemcpy(s->x,embed,H*sizeof(float),cudaMemcpyHostToDevice));
    CK(cudaMemcpy(s->d_pos,&pos,sizeof(int),cudaMemcpyHostToDevice));
    if (!s->cap_ready) {
        cudaGraph_t g;
        cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal);
        talker_body(s);
        cudaStreamEndCapture(cudaStreamPerThread, &g);
        if (cudaGraphInstantiate(&s->exec, g, 0) != cudaSuccess) { fprintf(stderr,"talker graph instantiate failed\n"); }
        cudaGraphDestroy(g);
        s->cap_ready=1;
    }
    CK(cudaGraphLaunch(s->exec, cudaStreamPerThread));
    CK(cudaStreamSynchronize(cudaStreamPerThread));
    CK(cudaMemcpy(hidden_out,s->xn,H*sizeof(float),cudaMemcpyDeviceToHost));
}

extern "C" void qwen_cuda_talker_upload_kv(void *state, qwen_tts_ctx_t *ctx, int prefill_len) {
    cuda_talker_t *s=(cuda_talker_t*)state; if(!s||prefill_len<=0) return;
    int kvd=s->kv_dim, L=s->n_layers, kvm=s->kv_max;
    size_t nper=(size_t)prefill_len*kvd;
    float *hk=(float*)malloc(nper*sizeof(float)), *hv=(float*)malloc(nper*sizeof(float));
    for (int l=0;l<L;++l){
        const uint16_t *ck=ctx->kv_cache_k+(size_t)l*kvm*kvd;
        const uint16_t *cv=ctx->kv_cache_v+(size_t)l*kvm*kvd;
        for (size_t i=0;i<nper;++i){ union{uint32_t u;float f;}a,b;
            a.u=(uint32_t)ck[i]<<16; hk[i]=a.f; b.u=(uint32_t)cv[i]<<16; hv[i]=b.f; }
        CK(cudaMemcpy(s->kcache+(size_t)l*kvm*kvd, hk, nper*sizeof(float), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(s->vcache+(size_t)l*kvm*kvd, hv, nper*sizeof(float), cudaMemcpyHostToDevice));
    }
    free(hk); free(hv);
}

/* ======================================================================== *
 *  BATCHED fused Talker/CP steps (throughput epic). B sequences share the
 *  resident weights (uploaded ONCE); activations are [B][dim], KV is [B][kv_max][kvd].
 *  The matvec→MATMAT kernels read each weight row ONCE and apply it to all B activation
 *  columns (register accumulator s[B]) → weight DRAM traffic amortized over B = the win.
 *  Single-stream gen is idle/sync-bound; B rides that idle capacity for near-free until
 *  compute-bound. Lockstep (all B at same pos) first cut; d_pos[B] is per-sequence so it
 *  generalizes to ragged/continuous batching. B capped at QB_MAX.
 * ======================================================================== */
#define QB_MAX 8

/* batched matmat: X[B][cols], Y[B][rows]; warp per output row reads W[row,:] once → B dots. */
__global__ void k_matmat_bf16(const __nv_bfloat16 *W,const float *X,float *Y,int rows,int cols,int B){
    int row=(blockIdx.x*blockDim.x+threadIdx.x)>>5, lane=threadIdx.x&31; if(row>=rows) return;
    const __nv_bfloat16 *wr=W+(size_t)row*cols; float s[QB_MAX];
    #pragma unroll
    for(int b=0;b<QB_MAX;++b) s[b]=0.f;
    for(int i=lane;i<cols;i+=32){ float w=__bfloat162float(wr[i]);
        for(int b=0;b<B;++b) s[b]+=w*X[(size_t)b*cols+i]; }
    for(int b=0;b<B;++b){ float v=s[b];
        #pragma unroll
        for(int o=16;o>0;o>>=1) v+=__shfl_down_sync(0xffffffffu,v,o);
        if(lane==0) Y[(size_t)b*rows+row]=v; }
}
__global__ void k_matmat_int8(const int8_t *W,const float *scale,const float *X,float *Y,int rows,int cols,int B){
    int row=(blockIdx.x*blockDim.x+threadIdx.x)>>5, lane=threadIdx.x&31; if(row>=rows) return;
    const int8_t *wr=W+(size_t)row*cols; float s[QB_MAX];
    #pragma unroll
    for(int b=0;b<QB_MAX;++b) s[b]=0.f;
    for(int i=lane;i<cols;i+=32){ float w=(float)wr[i];
        for(int b=0;b<B;++b) s[b]+=w*X[(size_t)b*cols+i]; }
    float sc=scale[row];
    for(int b=0;b<B;++b){ float v=s[b];
        #pragma unroll
        for(int o=16;o>0;o>>=1) v+=__shfl_down_sync(0xffffffffu,v,o);
        if(lane==0) Y[(size_t)b*rows+row]=sc*v; }
}
__global__ void k_matmat_q4_0(const q4blk *W,const float *X,float *Y,int rows,int cols,int B){
    int row=(blockIdx.x*blockDim.x+threadIdx.x)>>5, lane=threadIdx.x&31; if(row>=rows) return;
    int nb=cols>>5; const q4blk *wr=W+(size_t)row*nb; float s[QB_MAX];
    #pragma unroll
    for(int b=0;b<QB_MAX;++b) s[b]=0.f;
    for(int c=lane;c<cols;c+=32){ const q4blk *bk=wr+(c>>5); int ic=c&31;
        unsigned char byte=bk->qs[ic>>1]; int nib=(ic&1)?(byte>>4):(byte&0x0F);
        float w=(float)(nib-8)*__half2float(bk->scale);
        for(int b=0;b<B;++b) s[b]+=w*X[(size_t)b*cols+c]; }
    for(int b=0;b<B;++b){ float v=s[b];
        #pragma unroll
        for(int o=16;o>0;o>>=1) v+=__shfl_down_sync(0xffffffffu,v,o);
        if(lane==0) Y[(size_t)b*rows+row]=v; }
}
static inline void mvB(int prec,const void*W,const float*scale,const float*X,float*Y,int rows,int cols,int B){
    int grid=CEIL(rows*32,TPB);
    if(prec==2) k_matmat_q4_0<<<grid,TPB>>>((const q4blk*)W,X,Y,rows,cols,B);
    else if(prec==1) k_matmat_int8<<<grid,TPB>>>((const int8_t*)W,scale,X,Y,rows,cols,B);
    else k_matmat_bf16<<<grid,TPB>>>((const __nv_bfloat16*)W,X,Y,rows,cols,B);
}
/* one block per sequence */
__global__ void k_rmsnorm_full_b(const float *X,const float *w,float *Y,int dim,float eps){
    int b=blockIdx.x; const float *x=X+(size_t)b*dim; float *y=Y+(size_t)b*dim;
    extern __shared__ float part[]; int tid=threadIdx.x,tc=blockDim.x;
    float s=0.f; for(int i=tid;i<dim;i+=tc) s+=x[i]*x[i];
    part[tid]=s; __syncthreads();
    for(int st=tc/2;st>0;st>>=1){ if(tid<st) part[tid]+=part[tid+st]; __syncthreads(); }
    float inv=rsqrtf(part[0]/(float)dim+eps);
    for(int i=tid;i<dim;i+=tc) y[i]=x[i]*inv*w[i];
}
/* one block per (b,head): blk = b*nh + h; stride = per-seq q_dim or kv_dim */
__global__ void k_rmsnorm_ph_b(float *X,const float *w,int head_dim,int nh,int stride,float eps){
    int blk=blockIdx.x, b=blk/nh, h=blk%nh; float *xh=X+(size_t)b*stride+(size_t)h*head_dim;
    extern __shared__ float part[]; int tid=threadIdx.x,tc=blockDim.x;
    float s=0.f; for(int i=tid;i<head_dim;i+=tc) s+=xh[i]*xh[i];
    part[tid]=s; __syncthreads();
    for(int st=tc/2;st>0;st>>=1){ if(tid<st) part[tid]+=part[tid+st]; __syncthreads(); }
    float inv=rsqrtf(part[0]/(float)head_dim+eps);
    for(int i=tid;i<head_dim;i+=tc) xh[i]=xh[i]*inv*w[i];
}
__global__ void k_rope_neox_b(float *X,const float *cos_base,const float *sin_base,
                              int n_heads,int head_dim,const int *d_pos,int stride,int B){
    int half=head_dim/2, per=n_heads*half, gid=blockIdx.x*blockDim.x+threadIdx.x;
    if(gid>=B*per) return; int b=gid/per, rem=gid%per;
    const float *cosp=cos_base+(size_t)d_pos[b]*half, *sinp=sin_base+(size_t)d_pos[b]*half;
    int h=rem/half, i=rem%half; float *xh=X+(size_t)b*stride+(size_t)h*head_dim;
    float c=cosp[i], sn=sinp[i], x1=xh[i], x2=xh[i+half];
    xh[i]=x1*c-x2*sn; xh[i+half]=x2*c+x1*sn;
}
/* KV per sequence: kc/vc layer base = [B][kv_max][kvd]; K/V = [B][kvd] */
__global__ void k_kv_store_b(float *kc,float *vc,const float *K,const float *V,int kvd,const int *d_pos,int kv_max,int B){
    int gid=blockIdx.x*blockDim.x+threadIdx.x; if(gid>=B*kvd) return; int b=gid/kvd, i=gid%kvd;
    size_t off=(size_t)b*kv_max*kvd+(size_t)d_pos[b]*kvd+i;
    kc[off]=K[(size_t)b*kvd+i]; vc[off]=V[(size_t)b*kvd+i];
}
/* one block per (b,head): blk=b*n_heads+h, blockDim=hd. KV base per seq = [kv_max][kvd] */
__global__ void k_attn_b(const float *Q,const float *K,const float *V,float *O,
                         int n_heads,int n_kv,int hd,float scale,const int *d_pos,int kv_max,int qd,int kvd){
    int blk=blockIdx.x, b=blk/n_heads, h=blk%n_heads, t=threadIdx.x; if(t>=hd) return;
    int kvh=h/(n_heads/n_kv), valid=d_pos[b]+1;
    const float *Kb=K+(size_t)b*kv_max*kvd, *Vb=V+(size_t)b*kv_max*kvd;
    float qt=Q[(size_t)b*qd+(size_t)h*hd+t];
    extern __shared__ float sh[];
    float m=-1e30f, denom=0.f, acc=0.f;
    for(int j=0;j<valid;++j){ const float *k=Kb+((size_t)j*n_kv+kvh)*hd;
        sh[t]=qt*k[t]; __syncthreads();
        for(int s=hd/2;s>0;s>>=1){ if(t<s) sh[t]+=sh[t+s]; __syncthreads(); }
        float score=sh[0]*scale; __syncthreads();
        float mn=fmaxf(m,score), corr=expf(m-mn), p=expf(score-mn);
        denom=denom*corr+p; acc=acc*corr+p*Vb[((size_t)j*n_kv+kvh)*hd+t]; m=mn; }
    O[(size_t)b*qd+(size_t)h*hd+t]=acc/denom;
}
/* in=[B][2inter], out=[B][inter] */
__global__ void k_swiglu_il_b(const float *in,float *out,int inter,int B){
    int e=blockIdx.x*blockDim.x+threadIdx.x; if(e>=B*inter) return; int b=e/inter, j=e%inter;
    const float *ib=in+(size_t)b*2*inter; float g=ib[2*j], u=ib[2*j+1];
    out[(size_t)b*inter+j]=g/(1.f+expf(-g))*u;
}

typedef struct {
    int B, hidden, q_dim, kv_dim, inter, n_heads, n_kv, head_dim, n_layers, kv_max; float eps;
    void **wq,**wk,**wv,**wo,**wgu,**wdn; float **wqs,**wks,**wvs,**wos,**wgus,**wdns;
    float **inorm,**pnorm,**qn,**kn; float *tnorm,*rope_cos,*rope_sin;
    float *kcache,*vcache;                            /* [L][B][kv_max][kvd] */
    float *x,*xn,*q,*k,*v,*attn,*proj,*gate,*gu;      /* [B][dim] */
    int prec; int *d_pos;                             /* device int[B] */
} cuda_talker_batch_t;

static void talker_body_batch(cuda_talker_batch_t *s){
    int B=s->B,H=s->hidden,qd=s->q_dim,kvd=s->kv_dim,hd=s->head_dim,half=hd/2;
    int nh=s->n_heads,nkv=s->n_kv,inter=s->inter; float scale=1.f/sqrtf((float)hd);
    for(int l=0;l<s->n_layers;++l){
        k_rmsnorm_full_b<<<B,TPB,TPB*sizeof(float)>>>(s->x,s->inorm[l],s->xn,H,s->eps);
        mvB(s->prec,s->wq[l],s->wqs[l],s->xn,s->q,qd,H,B);
        mvB(s->prec,s->wk[l],s->wks[l],s->xn,s->k,kvd,H,B);
        mvB(s->prec,s->wv[l],s->wvs[l],s->xn,s->v,kvd,H,B);
        k_rmsnorm_ph_b<<<B*nh,TPB,TPB*sizeof(float)>>>(s->q,s->qn[l],hd,nh,qd,s->eps);
        k_rmsnorm_ph_b<<<B*nkv,TPB,TPB*sizeof(float)>>>(s->k,s->kn[l],hd,nkv,kvd,s->eps);
        k_rope_neox_b<<<CEIL(B*nh*half,TPB),TPB>>>(s->q,s->rope_cos,s->rope_sin,nh,hd,s->d_pos,qd,B);
        k_rope_neox_b<<<CEIL(B*nkv*half,TPB),TPB>>>(s->k,s->rope_cos,s->rope_sin,nkv,hd,s->d_pos,kvd,B);
        k_trunc_bf16<<<CEIL(B*kvd,TPB),TPB>>>(s->k,B*kvd);
        k_trunc_bf16<<<CEIL(B*kvd,TPB),TPB>>>(s->v,B*kvd);
        float *Kl=s->kcache+(size_t)l*B*s->kv_max*kvd, *Vl=s->vcache+(size_t)l*B*s->kv_max*kvd;
        k_kv_store_b<<<CEIL(B*kvd,TPB),TPB>>>(Kl,Vl,s->k,s->v,kvd,s->d_pos,s->kv_max,B);
        k_attn_b<<<B*nh,hd,hd*sizeof(float)>>>(s->q,Kl,Vl,s->attn,nh,nkv,hd,scale,s->d_pos,s->kv_max,qd,kvd);
        mvB(s->prec,s->wo[l],s->wos[l],s->attn,s->proj,H,qd,B);
        k_add_ip<<<CEIL(B*H,TPB),TPB>>>(s->x,s->proj,B*H);
        k_rmsnorm_full_b<<<B,TPB,TPB*sizeof(float)>>>(s->x,s->pnorm[l],s->xn,H,s->eps);
        mvB(s->prec,s->wgu[l],s->wgus[l],s->xn,s->gu,2*inter,H,B);
        k_swiglu_il_b<<<CEIL(B*inter,TPB),TPB>>>(s->gu,s->gate,inter,B);
        mvB(s->prec,s->wdn[l],s->wdns[l],s->gate,s->proj,H,inter,B);
        k_add_ip<<<CEIL(B*H,TPB),TPB>>>(s->x,s->proj,B*H);
    }
    k_rmsnorm_full_b<<<B,TPB,TPB*sizeof(float)>>>(s->x,s->tnorm,s->xn,H,s->eps);
}

/* Build a batched Talker state, SHARING the already-resident weights of a single state `ss`
 * (weights uploaded once — B multiplies only activations + KV). */
extern "C" void *qwen_cuda_talker_batch_init(void *single, int B){
    cuda_talker_t *ss=(cuda_talker_t*)single; if(!ss||B<1||B>QB_MAX) return NULL;
    cuda_talker_batch_t *s=(cuda_talker_batch_t*)calloc(1,sizeof(*s));
    s->B=B; s->hidden=ss->hidden; s->q_dim=ss->q_dim; s->kv_dim=ss->kv_dim; s->inter=ss->inter;
    s->n_heads=ss->n_heads; s->n_kv=ss->n_kv; s->head_dim=ss->head_dim; s->n_layers=ss->n_layers;
    s->kv_max=ss->kv_max; s->eps=ss->eps; s->prec=ss->prec;
    /* share weight pointers (do NOT free them in batch_free) */
    s->wq=ss->wq; s->wk=ss->wk; s->wv=ss->wv; s->wo=ss->wo; s->wgu=ss->wgu; s->wdn=ss->wdn;
    s->wqs=ss->wqs; s->wks=ss->wks; s->wvs=ss->wvs; s->wos=ss->wos; s->wgus=ss->wgus; s->wdns=ss->wdns;
    s->inorm=ss->inorm; s->pnorm=ss->pnorm; s->qn=ss->qn; s->kn=ss->kn;
    s->tnorm=ss->tnorm; s->rope_cos=ss->rope_cos; s->rope_sin=ss->rope_sin;
    int L=s->n_layers,H=s->hidden,qd=s->q_dim,kvd=s->kv_dim,inter=s->inter;
    CK(cudaMalloc(&s->kcache,(size_t)L*B*s->kv_max*kvd*sizeof(float)));
    CK(cudaMalloc(&s->vcache,(size_t)L*B*s->kv_max*kvd*sizeof(float)));
    CK(cudaMalloc(&s->x,(size_t)B*H*sizeof(float)));   CK(cudaMalloc(&s->xn,(size_t)B*H*sizeof(float)));
    CK(cudaMalloc(&s->q,(size_t)B*qd*sizeof(float)));   CK(cudaMalloc(&s->k,(size_t)B*kvd*sizeof(float)));
    CK(cudaMalloc(&s->v,(size_t)B*kvd*sizeof(float)));  CK(cudaMalloc(&s->attn,(size_t)B*qd*sizeof(float)));
    CK(cudaMalloc(&s->proj,(size_t)B*H*sizeof(float))); CK(cudaMalloc(&s->gate,(size_t)B*inter*sizeof(float)));
    CK(cudaMalloc(&s->gu,(size_t)B*2*inter*sizeof(float)));
    CK(cudaMalloc(&s->d_pos,B*sizeof(int)));
    return s;
}
/* embeds=[B][H] host, pos_arr=[B] host; hidden_out=[B][H] host (final-normed). */
extern "C" void qwen_cuda_talker_batch_step(void *st,const float *embeds,const int *pos_arr,float *hidden_out){
    cuda_talker_batch_t *s=(cuda_talker_batch_t*)st; int B=s->B,H=s->hidden;
    CK(cudaMemcpy(s->x,embeds,(size_t)B*H*sizeof(float),cudaMemcpyHostToDevice));
    CK(cudaMemcpy(s->d_pos,pos_arr,B*sizeof(int),cudaMemcpyHostToDevice));
    talker_body_batch(s);
    CK(cudaStreamSynchronize(cudaStreamPerThread));
    if(hidden_out) CK(cudaMemcpy(hidden_out,s->xn,(size_t)B*H*sizeof(float),cudaMemcpyDeviceToHost));
}
/* Seed slot b's device KV from the batch engine's bf16 KV (bb->kv_k/kv_v, layout
 * ((b*num_layers+l)*kv_max+pos)*kv_dim). Called on admit before the first decode step so the
 * GPU-resident batched Talker attends to the prompt. prefill_len = prompt length. */
extern "C" void qwen_cuda_talker_batch_upload_slot(void *st,int b,const uint16_t *kv_k,const uint16_t *kv_v,
                                                   int src_kv_max,int prefill_len){
    cuda_talker_batch_t *s=(cuda_talker_batch_t*)st; if(!s||prefill_len<=0) return;
    int L=s->n_layers,B=s->B,kvd=s->kv_dim,dkvm=s->kv_max; size_t nper=(size_t)prefill_len*kvd;
    float *hk=(float*)malloc(nper*sizeof(float)), *hv=(float*)malloc(nper*sizeof(float));
    for(int l=0;l<L;++l){
        const uint16_t *ck=kv_k+(((size_t)b*L+l)*src_kv_max)*kvd, *cv=kv_v+(((size_t)b*L+l)*src_kv_max)*kvd;
        for(size_t i=0;i<nper;++i){ union{uint32_t u;float f;}a,c; a.u=(uint32_t)ck[i]<<16; hk[i]=a.f; c.u=(uint32_t)cv[i]<<16; hv[i]=c.f; }
        float *dk=s->kcache+(((size_t)l*B+b)*dkvm)*kvd, *dv=s->vcache+(((size_t)l*B+b)*dkvm)*kvd;
        CK(cudaMemcpy(dk,hk,nper*sizeof(float),cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dv,hv,nper*sizeof(float),cudaMemcpyHostToDevice));
    }
    free(hk); free(hv);
}
extern "C" void qwen_cuda_talker_batch_free(void *st){
    cuda_talker_batch_t *s=(cuda_talker_batch_t*)st; if(!s) return;
    cudaFree(s->kcache);cudaFree(s->vcache);cudaFree(s->x);cudaFree(s->xn);cudaFree(s->q);
    cudaFree(s->k);cudaFree(s->v);cudaFree(s->attn);cudaFree(s->proj);cudaFree(s->gate);
    cudaFree(s->gu);cudaFree(s->d_pos); free(s);   /* weights are shared — not freed here */
}

/* Correctness (batched row b MUST equal a single-stream run with the same embeds/pos) +
 * throughput scaling (batched ms/frame for B seqs vs single ms/frame for 1 seq). */
static double now_ms(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e3+t.tv_nsec/1e6; }
extern "C" void qwen_cuda_talker_free(void *);   /* defined below (after the CP section) */
extern "C" void *qwen_cuda_cp_batch_init(void *, int);   /* defined below (CP section) */
extern "C" void  qwen_cuda_cp_batch_step(void *, float *, const int *);
extern "C" void  qwen_cuda_cp_batch_free(void *);
extern "C" int qwen_cuda_batch_selftest(qwen_tts_ctx_t *ctx, int B, int frames){
    int H=ctx->config.hidden_size, kvm=ctx->kv_max;
    if(frames>kvm-2) frames=kvm-2; if(frames<8) frames=8;
    void *S=qwen_cuda_talker_init(ctx); if(!S){ fprintf(stderr,"batch selftest: single init failed\n"); return -1; }
    void *Bs=qwen_cuda_talker_batch_init(S,B); if(!Bs){ fprintf(stderr,"batch selftest: batch init failed (B=%d>max %d?)\n",B,QB_MAX); return -1; }
    float *e1=(float*)malloc(H*sizeof(float)), *eB=(float*)malloc((size_t)B*H*sizeof(float));
    float *h1=(float*)malloc(H*sizeof(float)), *hB=(float*)malloc((size_t)B*H*sizeof(float));
    int *posB=(int*)malloc(B*sizeof(int));
    /* --- correctness: first min(frames,32) lockstep steps, same embed to single + all B rows --- */
    double maxdiff=0; int chk=frames<32?frames:32;
    for(int p=0;p<chk;++p){
        for(int i=0;i<H;++i) e1[i]=0.02f*sinf(0.11f*(i+1)+0.3f*p);
        for(int b=0;b<B;++b){ memcpy(eB+(size_t)b*H,e1,H*sizeof(float)); posB[b]=p; }
        qwen_cuda_talker_step(S,e1,h1,p);
        qwen_cuda_talker_batch_step(Bs,eB,posB,hB);
        for(int b=0;b<B;++b) for(int i=0;i<H;++i){ double d=fabs(hB[(size_t)b*H+i]-h1[i]); if(d>maxdiff)maxdiff=d; }
    }
    fprintf(stderr,"batch selftest: correctness max|batched-single|=%.2e over %d steps (%s)\n",
            maxdiff,chk, maxdiff<1e-3?"PASS":"FAIL");
    qwen_cuda_talker_batch_free(Bs); qwen_cuda_talker_free(S);
    /* --- throughput: fresh states, time single (1 seq) vs batched (B seq) --- */
    S=qwen_cuda_talker_init(ctx); Bs=qwen_cuda_talker_batch_init(S,B);
    for(int i=0;i<H;++i) e1[i]=0.02f*sinf(0.11f*(i+1));
    for(int b=0;b<B;++b){ memcpy(eB+(size_t)b*H,e1,H*sizeof(float)); posB[b]=0; }
    qwen_cuda_talker_step(S,e1,h1,0); qwen_cuda_talker_batch_step(Bs,eB,posB,hB);  /* warmup */
    double t0=now_ms(); for(int p=1;p<=frames;++p) qwen_cuda_talker_step(S,e1,h1,p);
    double ts=(now_ms()-t0)/frames;
    t0=now_ms(); for(int p=1;p<=frames;++p){ for(int b=0;b<B;++b) posB[b]=p; qwen_cuda_talker_batch_step(Bs,eB,posB,hB); }
    double tb=(now_ms()-t0)/frames;
    fprintf(stderr,"batch Talker throughput (B=%d, %d frames): single %.2f ms/f (1 seq) | batched %.2f ms/f (%d seq) | per-seq %.2f ms | GAIN %.2fx\n",
            B,frames,ts,tb,B,tb/B, B*ts/tb);
    free(e1);free(eB);free(h1);free(hB);free(posB);
    qwen_cuda_talker_batch_free(Bs); qwen_cuda_talker_free(S);

    /* --- Code Predictor: correctness + throughput (16 passes/frame, the CP loop) --- */
    extern void *qwen_cuda_cp_init(qwen_tts_ctx_t *);
    extern void  qwen_cuda_cp_step(void *, float *, int);
    extern void  qwen_cuda_cp_free(void *);
    int cph=ctx->config.cp_hidden_size, cpkv=ctx->cp_kv_max;
    void *CS=qwen_cuda_cp_init(ctx); void *CB=qwen_cuda_cp_batch_init(CS,B);
    if(CS&&CB){
        float *cx1=(float*)malloc(cph*sizeof(float)), *cxB=(float*)malloc((size_t)B*cph*sizeof(float));
        int *cpos=(int*)malloc(B*sizeof(int));
        double cdiff=0; int cchk=cpkv<16?cpkv:16;
        for(int p=0;p<cchk;++p){
            float v; for(int i=0;i<cph;++i){ v=0.02f*sinf(0.13f*(i+1)+0.2f*p); cx1[i]=v; for(int b=0;b<B;++b) cxB[(size_t)b*cph+i]=v; }
            for(int b=0;b<B;++b) cpos[b]=p;
            qwen_cuda_cp_step(CS,cx1,p);
            qwen_cuda_cp_batch_step(CB,cxB,cpos);
            for(int b=0;b<B;++b) for(int i=0;i<cph;++i){ double d=fabs(cxB[(size_t)b*cph+i]-cx1[i]); if(d>cdiff)cdiff=d; }
        }
        fprintf(stderr,"batch selftest: CP correctness max|batched-single|=%.2e over %d steps (%s)\n",
                cdiff,cchk, cdiff<1e-3?"PASS":"FAIL");
        /* throughput: 16 passes/frame (CP KV holds 16), pos 0..15 lockstep */
        int passes=cpkv<16?cpkv:16;
        for(int i=0;i<cph;++i){ float v=0.02f*sinf(0.13f*(i+1)); cx1[i]=v; for(int b=0;b<B;++b) cxB[(size_t)b*cph+i]=v; }
        for(int p=0;p<passes;++p) qwen_cuda_cp_step(CS,cx1,p);                 /* warmup */
        for(int p=0;p<passes;++p){ for(int b=0;b<B;++b) cpos[b]=p; qwen_cuda_cp_batch_step(CB,cxB,cpos); }
        int NF=frames/4; if(NF<10) NF=10;
        double c0=now_ms();
        for(int f=0;f<NF;++f) for(int p=0;p<passes;++p) qwen_cuda_cp_step(CS,cx1,p);
        double cs=(now_ms()-c0)/NF;
        c0=now_ms();
        for(int f=0;f<NF;++f) for(int p=0;p<passes;++p){ for(int b=0;b<B;++b) cpos[b]=p; qwen_cuda_cp_batch_step(CB,cxB,cpos); }
        double cb=(now_ms()-c0)/NF;
        fprintf(stderr,"batch CP throughput (B=%d, %d passes/frame): single %.2f ms/f (1 seq) | batched %.2f ms/f (%d seq) | GAIN %.2fx\n",
                B,passes,cs,cb,B, B*cs/cb);
        fprintf(stderr,"batch GEN (Talker+CP) aggregate throughput GAIN (B=%d): %.2fx\n", B, B*(ts+cs)/(tb+cb));
        free(cx1);free(cxB);free(cpos);
        if(cdiff>=1e-3) maxdiff=cdiff;
    }
    if(CB) qwen_cuda_cp_batch_free(CB); if(CS) qwen_cuda_cp_free(CS);
    return maxdiff<1e-3?0:1;
}

/* ======================================================================== *
 *  GPU-RESIDENT fused Code Predictor step (M2). Same layer structure as the
 *  Talker (reuses ALL the kernels above); differences: CP dims (hidden 1024, 5
 *  layers), the CP KV is per-frame (built fresh each frame, no prefill upload),
 *  and there is NO final norm (the caller applies cp_norm before the lm-head).
 *  Emotion steer is a pre-step input add on the CPU → the fused step is emo-safe.
 * ======================================================================== */
typedef struct {
    int hidden, q_dim, kv_dim, inter, n_heads, n_kv, head_dim, n_layers, kv_max;
    float eps;
    void **wq,**wk,**wv,**wo,**wgu,**wdn;            /* bf16 or int8 */
    float **wqs,**wks,**wvs,**wos,**wgus,**wdns;     /* per-row scales (NULL = bf16) */
    float **inorm,**pnorm,**qn,**kn;
    float *rope_cos,*rope_sin;
    float *kcache,*vcache;
    float *x,*xn,*q,*k,*v,*attn,*proj,*gate,*gu;
    int prec; int *d_pos; cudaGraphExec_t exec; int cap_ready;   /* CUDA graph of the 5-layer CP step */
} cuda_cp_t;

extern "C" void *qwen_cuda_cp_init(qwen_tts_ctx_t *ctx) {
    qwen_tts_config_t *c=&ctx->config;
    cuda_cp_t *s=(cuda_cp_t*)calloc(1,sizeof(*s));
    s->hidden=c->cp_hidden_size; s->n_heads=c->cp_num_heads; s->n_kv=c->cp_num_kv_heads;
    s->head_dim=c->cp_head_dim; s->inter=c->cp_intermediate_size; s->n_layers=c->cp_num_layers;
    s->q_dim=c->cp_num_heads*c->cp_head_dim; s->kv_dim=c->cp_num_kv_heads*c->cp_head_dim;
    s->eps=c->rms_norm_eps; s->kv_max=ctx->cp_kv_max;
    int L=s->n_layers, H=s->hidden, hd=s->head_dim, half=hd/2;
    s->wq=(void**)calloc(L,sizeof(void*)); s->wk=(void**)calloc(L,sizeof(void*));
    s->wv=(void**)calloc(L,sizeof(void*)); s->wo=(void**)calloc(L,sizeof(void*));
    s->wgu=(void**)calloc(L,sizeof(void*)); s->wdn=(void**)calloc(L,sizeof(void*));
    s->wqs=(float**)calloc(L,sizeof(float*)); s->wks=(float**)calloc(L,sizeof(float*));
    s->wvs=(float**)calloc(L,sizeof(float*)); s->wos=(float**)calloc(L,sizeof(float*));
    s->wgus=(float**)calloc(L,sizeof(float*)); s->wdns=(float**)calloc(L,sizeof(float*));
    s->inorm=(float**)calloc(L,sizeof(float*)); s->pnorm=(float**)calloc(L,sizeof(float*));
    s->qn=(float**)calloc(L,sizeof(float*)); s->kn=(float**)calloc(L,sizeof(float*));
    int used_int8=0;
    for (int l=0;l<L;++l){
        qwen_cp_layer_t *ly=&ctx->cp_layers[l];
        if (!ly->wq_bf16 && !ly->wq_int8 && !ly->wq_q4){ fprintf(stderr,"CUDA CP: layer %d has no weights\n",l); return NULL; }
        UPW(s->wq[l], s->wqs[l], ly->wq_q4, ly->wq_int8, ly->wq_scale, ly->wq_bf16, s->q_dim, H, &s->prec);
        UPW(s->wk[l], s->wks[l], ly->wk_q4, ly->wk_int8, ly->wk_scale, ly->wk_bf16, s->kv_dim, H, &s->prec);
        UPW(s->wv[l], s->wvs[l], ly->wv_q4, ly->wv_int8, ly->wv_scale, ly->wv_bf16, s->kv_dim, H, &s->prec);
        UPW(s->wo[l], s->wos[l], ly->wo_q4, ly->wo_int8, ly->wo_scale, ly->wo_bf16, H, s->q_dim, &s->prec);
        UPW(s->wgu[l],s->wgus[l],ly->gate_up_fused_q4, ly->gate_up_fused_int8, ly->gate_up_fused_scale, ly->gate_up_fused_bf16, 2*s->inter, H, &s->prec);
        UPW(s->wdn[l],s->wdns[l],ly->down_q4, ly->down_int8, ly->down_scale, ly->down_bf16, H, s->inter, &s->prec);
        used_int8 = (s->prec==1);
        s->inorm[l]=up_f32(ly->input_norm,H); s->pnorm[l]=up_f32(ly->post_attn_norm,H);
        s->qn[l]=up_f32(ly->q_norm,hd); s->kn[l]=up_f32(ly->k_norm,hd);
    }
    s->rope_cos=up_f32(ctx->cp_rope_cos,(size_t)s->kv_max*half);
    s->rope_sin=up_f32(ctx->cp_rope_sin,(size_t)s->kv_max*half);
    CK(cudaMalloc(&s->kcache,(size_t)L*s->kv_max*s->kv_dim*sizeof(float)));
    CK(cudaMalloc(&s->vcache,(size_t)L*s->kv_max*s->kv_dim*sizeof(float)));
    CK(cudaMalloc(&s->x,H*sizeof(float)));  CK(cudaMalloc(&s->xn,H*sizeof(float)));
    CK(cudaMalloc(&s->q,s->q_dim*sizeof(float))); CK(cudaMalloc(&s->k,s->kv_dim*sizeof(float)));
    CK(cudaMalloc(&s->v,s->kv_dim*sizeof(float))); CK(cudaMalloc(&s->attn,s->q_dim*sizeof(float)));
    CK(cudaMalloc(&s->proj,H*sizeof(float))); CK(cudaMalloc(&s->gate,s->inter*sizeof(float)));
    CK(cudaMalloc(&s->gu,(size_t)2*s->inter*sizeof(float)));
    CK(cudaMalloc(&s->d_pos,sizeof(int)));
    fprintf(stderr,"CUDA CP: resident fused step ready (%d layers, hidden=%d, %s, CUDA graph)\n",L,H,s->prec==2?"q4_0":s->prec==1?"int8":"bf16");
    return s;
}

/* One resident CP transformer step. x[cp_h] in/out (residual stream; caller norms). */
static void cp_body(cuda_cp_t *s) {
    int H=s->hidden, qd=s->q_dim, kvd=s->kv_dim, hd=s->head_dim, half=hd/2;
    int nh=s->n_heads, nkv=s->n_kv, inter=s->inter;
    float scale=1.f/sqrtf((float)hd);
    for (int l=0;l<s->n_layers;++l){
        k_rmsnorm_full<<<1,TPB,TPB*sizeof(float)>>>(s->x,s->inorm[l],s->xn,H,s->eps);
        mv(s->prec,s->wq[l],s->wqs[l],s->xn,s->q,qd,H);
        mv(s->prec,s->wk[l],s->wks[l],s->xn,s->k,kvd,H);
        mv(s->prec,s->wv[l],s->wvs[l],s->xn,s->v,kvd,H);
        k_rmsnorm_ph<<<nh, TPB, TPB*sizeof(float)>>>(s->q,s->qn[l],hd,s->eps);
        k_rmsnorm_ph<<<nkv,TPB, TPB*sizeof(float)>>>(s->k,s->kn[l],hd,s->eps);
        k_rope_neox<<<CEIL(nh*half,TPB),TPB>>>(s->q,s->rope_cos,s->rope_sin,nh,hd,s->d_pos);
        k_rope_neox<<<CEIL(nkv*half,TPB),TPB>>>(s->k,s->rope_cos,s->rope_sin,nkv,hd,s->d_pos);
        k_trunc_bf16<<<CEIL(kvd,TPB),TPB>>>(s->k,kvd);
        k_trunc_bf16<<<CEIL(kvd,TPB),TPB>>>(s->v,kvd);
        float *Kl=s->kcache+(size_t)l*s->kv_max*kvd, *Vl=s->vcache+(size_t)l*s->kv_max*kvd;
        k_kv_store<<<CEIL(kvd,TPB),TPB>>>(Kl,Vl,s->k,s->v,kvd,s->d_pos);
        k_attn<<<nh,hd,hd*sizeof(float)>>>(s->q,Kl,Vl,s->attn,nh,nkv,hd,scale,s->d_pos);
        mv(s->prec,s->wo[l],s->wos[l],s->attn,s->proj,H,qd);
        k_add_ip<<<CEIL(H,TPB),TPB>>>(s->x,s->proj,H);
        k_rmsnorm_full<<<1,TPB,TPB*sizeof(float)>>>(s->x,s->pnorm[l],s->xn,H,s->eps);
        mv(s->prec,s->wgu[l],s->wgus[l],s->xn,s->gu,2*inter,H);
        k_swiglu_il<<<CEIL(inter,TPB),TPB>>>(s->gu,s->gate,inter);
        mv(s->prec,s->wdn[l],s->wdns[l],s->gate,s->proj,H,inter);
        k_add_ip<<<CEIL(H,TPB),TPB>>>(s->x,s->proj,H);
    }
}

extern "C" void qwen_cuda_cp_step(void *st, float *x, int pos) {
    cuda_cp_t *s=(cuda_cp_t*)st;
    int H=s->hidden;
    CK(cudaMemcpy(s->x,x,H*sizeof(float),cudaMemcpyHostToDevice));
    CK(cudaMemcpy(s->d_pos,&pos,sizeof(int),cudaMemcpyHostToDevice));
    if (!s->cap_ready) {
        cudaGraph_t g;
        cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal);
        cp_body(s);
        cudaStreamEndCapture(cudaStreamPerThread, &g);
        if (cudaGraphInstantiate(&s->exec, g, 0) != cudaSuccess) fprintf(stderr,"CP graph instantiate failed\n");
        cudaGraphDestroy(g);
        s->cap_ready=1;
    }
    CK(cudaGraphLaunch(s->exec, cudaStreamPerThread));
    CK(cudaStreamSynchronize(cudaStreamPerThread));
    CK(cudaMemcpy(x,s->x,H*sizeof(float),cudaMemcpyDeviceToHost));   /* residual, NOT normed */
}

/* ---- BATCHED CP step (B sequences in lockstep through one pass; caller does the per-seq
 * argmax + embed between passes). Reuses the _b kernels + mvB. No final norm (caller norms). --- */
typedef struct {
    int B, hidden, q_dim, kv_dim, inter, n_heads, n_kv, head_dim, n_layers, kv_max; float eps;
    void **wq,**wk,**wv,**wo,**wgu,**wdn; float **wqs,**wks,**wvs,**wos,**wgus,**wdns;
    float **inorm,**pnorm,**qn,**kn; float *rope_cos,*rope_sin;
    float *kcache,*vcache; float *x,*xn,*q,*k,*v,*attn,*proj,*gate,*gu;
    int prec; int *d_pos;
} cuda_cp_batch_t;

static void cp_body_batch(cuda_cp_batch_t *s){
    int B=s->B,H=s->hidden,qd=s->q_dim,kvd=s->kv_dim,hd=s->head_dim,half=hd/2;
    int nh=s->n_heads,nkv=s->n_kv,inter=s->inter; float scale=1.f/sqrtf((float)hd);
    for(int l=0;l<s->n_layers;++l){
        k_rmsnorm_full_b<<<B,TPB,TPB*sizeof(float)>>>(s->x,s->inorm[l],s->xn,H,s->eps);
        mvB(s->prec,s->wq[l],s->wqs[l],s->xn,s->q,qd,H,B);
        mvB(s->prec,s->wk[l],s->wks[l],s->xn,s->k,kvd,H,B);
        mvB(s->prec,s->wv[l],s->wvs[l],s->xn,s->v,kvd,H,B);
        k_rmsnorm_ph_b<<<B*nh,TPB,TPB*sizeof(float)>>>(s->q,s->qn[l],hd,nh,qd,s->eps);
        k_rmsnorm_ph_b<<<B*nkv,TPB,TPB*sizeof(float)>>>(s->k,s->kn[l],hd,nkv,kvd,s->eps);
        k_rope_neox_b<<<CEIL(B*nh*half,TPB),TPB>>>(s->q,s->rope_cos,s->rope_sin,nh,hd,s->d_pos,qd,B);
        k_rope_neox_b<<<CEIL(B*nkv*half,TPB),TPB>>>(s->k,s->rope_cos,s->rope_sin,nkv,hd,s->d_pos,kvd,B);
        k_trunc_bf16<<<CEIL(B*kvd,TPB),TPB>>>(s->k,B*kvd);
        k_trunc_bf16<<<CEIL(B*kvd,TPB),TPB>>>(s->v,B*kvd);
        float *Kl=s->kcache+(size_t)l*B*s->kv_max*kvd, *Vl=s->vcache+(size_t)l*B*s->kv_max*kvd;
        k_kv_store_b<<<CEIL(B*kvd,TPB),TPB>>>(Kl,Vl,s->k,s->v,kvd,s->d_pos,s->kv_max,B);
        k_attn_b<<<B*nh,hd,hd*sizeof(float)>>>(s->q,Kl,Vl,s->attn,nh,nkv,hd,scale,s->d_pos,s->kv_max,qd,kvd);
        mvB(s->prec,s->wo[l],s->wos[l],s->attn,s->proj,H,qd,B);
        k_add_ip<<<CEIL(B*H,TPB),TPB>>>(s->x,s->proj,B*H);
        k_rmsnorm_full_b<<<B,TPB,TPB*sizeof(float)>>>(s->x,s->pnorm[l],s->xn,H,s->eps);
        mvB(s->prec,s->wgu[l],s->wgus[l],s->xn,s->gu,2*inter,H,B);
        k_swiglu_il_b<<<CEIL(B*inter,TPB),TPB>>>(s->gu,s->gate,inter,B);
        mvB(s->prec,s->wdn[l],s->wdns[l],s->gate,s->proj,H,inter,B);
        k_add_ip<<<CEIL(B*H,TPB),TPB>>>(s->x,s->proj,B*H);
    }
}
extern "C" void *qwen_cuda_cp_batch_init(void *single, int B){
    cuda_cp_t *ss=(cuda_cp_t*)single; if(!ss||B<1||B>QB_MAX) return NULL;
    cuda_cp_batch_t *s=(cuda_cp_batch_t*)calloc(1,sizeof(*s));
    s->B=B; s->hidden=ss->hidden; s->q_dim=ss->q_dim; s->kv_dim=ss->kv_dim; s->inter=ss->inter;
    s->n_heads=ss->n_heads; s->n_kv=ss->n_kv; s->head_dim=ss->head_dim; s->n_layers=ss->n_layers;
    s->kv_max=ss->kv_max; s->eps=ss->eps; s->prec=ss->prec;
    s->wq=ss->wq; s->wk=ss->wk; s->wv=ss->wv; s->wo=ss->wo; s->wgu=ss->wgu; s->wdn=ss->wdn;
    s->wqs=ss->wqs; s->wks=ss->wks; s->wvs=ss->wvs; s->wos=ss->wos; s->wgus=ss->wgus; s->wdns=ss->wdns;
    s->inorm=ss->inorm; s->pnorm=ss->pnorm; s->qn=ss->qn; s->kn=ss->kn;
    s->rope_cos=ss->rope_cos; s->rope_sin=ss->rope_sin;
    int L=s->n_layers,H=s->hidden,qd=s->q_dim,kvd=s->kv_dim,inter=s->inter;
    CK(cudaMalloc(&s->kcache,(size_t)L*B*s->kv_max*kvd*sizeof(float)));
    CK(cudaMalloc(&s->vcache,(size_t)L*B*s->kv_max*kvd*sizeof(float)));
    CK(cudaMalloc(&s->x,(size_t)B*H*sizeof(float)));   CK(cudaMalloc(&s->xn,(size_t)B*H*sizeof(float)));
    CK(cudaMalloc(&s->q,(size_t)B*qd*sizeof(float)));   CK(cudaMalloc(&s->k,(size_t)B*kvd*sizeof(float)));
    CK(cudaMalloc(&s->v,(size_t)B*kvd*sizeof(float)));  CK(cudaMalloc(&s->attn,(size_t)B*qd*sizeof(float)));
    CK(cudaMalloc(&s->proj,(size_t)B*H*sizeof(float))); CK(cudaMalloc(&s->gate,(size_t)B*inter*sizeof(float)));
    CK(cudaMalloc(&s->gu,(size_t)B*2*inter*sizeof(float)));
    CK(cudaMalloc(&s->d_pos,B*sizeof(int)));
    return s;
}
/* x=[B][cp_h] host (each row = the caller's per-seq embed/residual seed), pos_arr=[B] host;
 * x updated in place with the B residual streams (caller norms + argmaxes each). */
extern "C" void qwen_cuda_cp_batch_step(void *st,float *x,const int *pos_arr){
    cuda_cp_batch_t *s=(cuda_cp_batch_t*)st; int B=s->B,H=s->hidden;
    CK(cudaMemcpy(s->x,x,(size_t)B*H*sizeof(float),cudaMemcpyHostToDevice));
    CK(cudaMemcpy(s->d_pos,pos_arr,B*sizeof(int),cudaMemcpyHostToDevice));
    cp_body_batch(s);
    CK(cudaStreamSynchronize(cudaStreamPerThread));
    CK(cudaMemcpy(x,s->x,(size_t)B*H*sizeof(float),cudaMemcpyDeviceToHost));
}
extern "C" void qwen_cuda_cp_batch_free(void *st){
    cuda_cp_batch_t *s=(cuda_cp_batch_t*)st; if(!s) return;
    cudaFree(s->kcache);cudaFree(s->vcache);cudaFree(s->x);cudaFree(s->xn);cudaFree(s->q);
    cudaFree(s->k);cudaFree(s->v);cudaFree(s->attn);cudaFree(s->proj);cudaFree(s->gate);
    cudaFree(s->gu);cudaFree(s->d_pos); free(s);
}

extern "C" void qwen_cuda_cp_free(void *st) {
    cuda_cp_t *s=(cuda_cp_t*)st; if(!s) return;
    for(int l=0;l<s->n_layers;++l){ cudaFree(s->wq[l]);cudaFree(s->wk[l]);cudaFree(s->wv[l]);
        cudaFree(s->wo[l]);cudaFree(s->wgu[l]);cudaFree(s->wdn[l]);cudaFree(s->inorm[l]);
        cudaFree(s->pnorm[l]);cudaFree(s->qn[l]);cudaFree(s->kn[l]);
        cudaFree(s->wqs[l]);cudaFree(s->wks[l]);cudaFree(s->wvs[l]);cudaFree(s->wos[l]);cudaFree(s->wgus[l]);cudaFree(s->wdns[l]); }
    free(s->wq);free(s->wk);free(s->wv);free(s->wo);free(s->wgu);free(s->wdn);
    free(s->wqs);free(s->wks);free(s->wvs);free(s->wos);free(s->wgus);free(s->wdns);
    free(s->inorm);free(s->pnorm);free(s->qn);free(s->kn);
    cudaFree(s->rope_cos);cudaFree(s->rope_sin);cudaFree(s->kcache);cudaFree(s->vcache);
    cudaFree(s->x);cudaFree(s->xn);cudaFree(s->q);cudaFree(s->k);cudaFree(s->v);
    cudaFree(s->attn);cudaFree(s->proj);cudaFree(s->gate);cudaFree(s->gu);cudaFree(s->d_pos);
    if(s->cap_ready) cudaGraphExecDestroy(s->exec);
    free(s);
}

extern "C" void qwen_cuda_talker_free(void *st) {
    cuda_talker_t *s=(cuda_talker_t*)st; if(!s) return;
    for(int l=0;l<s->n_layers;++l){ cudaFree(s->wq[l]);cudaFree(s->wk[l]);cudaFree(s->wv[l]);
        cudaFree(s->wo[l]);cudaFree(s->wgu[l]);cudaFree(s->wdn[l]);cudaFree(s->inorm[l]);
        cudaFree(s->pnorm[l]);cudaFree(s->qn[l]);cudaFree(s->kn[l]);
        cudaFree(s->wqs[l]);cudaFree(s->wks[l]);cudaFree(s->wvs[l]);cudaFree(s->wos[l]);cudaFree(s->wgus[l]);cudaFree(s->wdns[l]); }
    free(s->wq);free(s->wk);free(s->wv);free(s->wo);free(s->wgu);free(s->wdn);
    free(s->wqs);free(s->wks);free(s->wvs);free(s->wos);free(s->wgus);free(s->wdns);
    free(s->inorm);free(s->pnorm);free(s->qn);free(s->kn);
    cudaFree(s->tnorm);cudaFree(s->rope_cos);cudaFree(s->rope_sin);cudaFree(s->kcache);cudaFree(s->vcache);
    cudaFree(s->x);cudaFree(s->xn);cudaFree(s->q);cudaFree(s->k);cudaFree(s->v);
    cudaFree(s->attn);cudaFree(s->proj);cudaFree(s->gate);cudaFree(s->gu);cudaFree(s->d_pos);
    if(s->cap_ready) cudaGraphExecDestroy(s->exec);
    free(s);
}
