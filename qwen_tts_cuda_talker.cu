/*
 * qwen_tts_cuda_talker.cu - GPU-resident fused Talker step.
 * Weights, KV and activations stay resident; the whole step runs as a chain of kernels with a
 * single sync. Single-token decode is bandwidth-bound on the weight reads, so weights stay
 * bf16 on the device and a custom bf16 matvec reads them directly.
 */

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
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

/* q4_0 block: 32 weights = fp16 scale + 16 bytes (low nibble = even index, high = odd),
 * value = (nibble - 8) * scale. MUST match q4_0_block_t in qwen_tts_kernels.h exactly. */
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

/* dp4a q4xq8 twin (opt-in QWEN_CUDA_DP4A=1): quantize the activation once per matvec into
 * per-32-block int8 (deinterleaved even/odd to match q4_0's within-byte packing) plus a
 * per-block scale and sum, then integer __dp4a dots with the -8 offset corrected via the
 * block sum. `qs` sits at byte offset 2 of the 18-byte block, so words are built from BYTE
 * loads - a uint32 read there would be misaligned. */
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
        /* Default on. QWEN_CUDA_DP4A=0 opts out. */
        const char *e = getenv("QWEN_CUDA_DP4A");
        on = !(e && e[0]=='0');
        if (on) {
            if (cudaMalloc(&g_qdp_qe,  QDP_MAX_COLS/2) != cudaSuccess ||
                cudaMalloc(&g_qdp_qo,  QDP_MAX_COLS/2) != cudaSuccess ||
                cudaMalloc(&g_qdp_sx,  (QDP_MAX_COLS/32)*sizeof(float)) != cudaSuccess ||
                cudaMalloc(&g_qdp_sum, (QDP_MAX_COLS/32)*sizeof(int))   != cudaSuccess) {
                fprintf(stderr, "[cuda] dp4a scratch alloc failed — falling back to f32-act q4 kernel\n");
                on = 0;
            } else {
                fprintf(stderr, "[cuda] q4_0 matvec: dp4a q4xq8 path ENABLED (default; QWEN_CUDA_DP4A=0 disables)\n");
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

static int cublas_mm_enabled(void);   /* defined with the batched matmat, below */
extern "C" void *qwen_cuda_talker_init(qwen_tts_ctx_t *ctx) {
    qwen_tts_config_t *c=&ctx->config;
    /* The dp4a scratch MUST be allocated here, before any CUDA-graph capture: cudaMalloc
     * inside a capturing stream fails, and a lazy allocation inside mv() both fails and
     * perturbs the capture. qdp_enabled() is idempotent. */
    (void)qdp_enabled();
    (void)cublas_mm_enabled();   /* same hazard: creating the handle inside a capture fails */
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

/* Last stepped token's pre-final-norm residual (s->x, stable after the step's sync).
 * qwen_talker_step uses it to refresh ctx->dec_x, which the fused step otherwise never
 * touches: the generation loop seeds last_hidden = rms_norm(dec_x, talker_norm) after
 * prefill, and a stale dec_x from a previous request would corrupt the first frame. */
extern "C" void qwen_cuda_talker_get_dec_x(void *state, float *out) {
    cuda_talker_t *s=(cuda_talker_t*)state;
    if (s && out) CK(cudaMemcpy(out, s->x, (size_t)s->hidden*sizeof(float), cudaMemcpyDeviceToHost));
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

/* BATCHED fused Talker/CP steps. B sequences share the resident weights; activations are
 * [B][dim], KV is [B][kv_max][kvd]. The matmat kernels read each weight row once and apply it
 * to all B columns (accumulator s[B]), amortizing weight DRAM traffic over B. Lockstep, but
 * d_pos[B] is per-sequence so it generalizes to ragged batching. B capped at QB_MAX. */
#define QB_MAX 16
/* The batched CUDA path's lane cap, stated once. The C side used to hardcode 8 next to the
 * Metal cap, which meant raising one silently required remembering the other. */
extern "C" int qwen_cuda_batch_max(void){ return QB_MAX; }


/* batched matmat: X[B][cols], Y[B][rows]; warp per output row reads W[row,:] once → B dots.
 *
 * The per-sequence loops are unrolled over the compile-time QB_MAX with a b<B guard rather
 * than written as `for(b=0;b<B;++b)`.  With a runtime bound the compiler cannot prove the
 * index range of the s[] accumulator, so it places the array in LOCAL memory -- which is
 * backed by global memory.  That cost correctness and speed at once: the batch self-test went
 * from exact at B<=2 to max|batched-single|=2.93e+01 at B>=4, compute-sanitizer reported
 * "Invalid __global__ write of size 4 bytes" inside this kernel, and throughput collapsed from
 * 0.89x to 0.06x of single-stream. Unrolled, s[] stays in registers and every index is a
 * compile-time constant. */
/* Four weight loads in flight per lane.
 *
 * At B=2, 4 and 8 this kernel took 0.77, 0.72 and 0.82 ms per code-predictor pass: nearly
 * flat, although eight lanes do four times the arithmetic and four times the activation
 * traffic of two.  A kernel whose cost does not move with the work it does is not limited by
 * bandwidth or by FLOPs, and 139 MB in 0.82 ms is 180 GB/s on a card that does 768.  What it
 * was limited by is latency: one warp, one row, one 2-byte load per iteration, and the very
 * next instruction needs the value it just loaded.
 *
 * The fix is four independent loads before any of them is consumed, which is four misses in
 * flight per lane instead of one.  The stride stays 32, so each of the four loads is still a
 * fully coalesced warp-wide read of consecutive elements, on the weights and on the
 * activations alike -- this is the reason not to switch to a vector type, which would have a
 * lane read four ADJACENT elements and scatter the activation reads across four sectors.
 *
 * The accumulator stays unrolled over QB_MAX for the reason recorded above: indexed by a
 * runtime lane it spills to local memory and costs 33x. */
/* NB, the lane count, is a template parameter and not an argument.
 *
 * s[] is an array of registers sized at compile time. Sized at QB_MAX with a runtime `b<B`
 * guard it burns QB_MAX registers whatever B actually is, so raising the cap from 8 to 16 made
 * EVERY batch size slower -- measured 4.81 -> 7.38 ms/frame on the Talker at B=8, and 10.62 ->
 * 18.06 on the CP -- because occupancy fell. That is the quiet sibling of the spill that cost
 * 33x in the batched matmats, and it is why the cap could not simply be raised. */
template<int U,int NB>
__global__ void k_matmat_bf16_u(const __nv_bfloat16 *__restrict__ W,const float *__restrict__ X,
                                float *__restrict__ Y,int rows,int cols,int B){
    int row=(blockIdx.x*blockDim.x+threadIdx.x)>>5, lane=threadIdx.x&31; if(row>=rows) return;
    const __nv_bfloat16 *wr=W+(size_t)row*cols; float s[NB];
    #pragma unroll
    for(int b=0;b<NB;++b) s[b]=0.f;
    int i=lane; const int span=32*U;
    for(;i+span-32<cols;i+=span){
        float w[U];
        #pragma unroll
        for(int u=0;u<U;++u) w[u]=__bfloat162float(wr[i+32*u]);   /* U misses in flight */
        #pragma unroll
        for(int b=0;b<NB;++b) if(b<B){ const float *xb=X+(size_t)b*cols;
            /* One at a time, in the scalar loop's order, so the result is bit-identical to it.
             * Summed as a single expression the compiler builds a different reduction tree and
             * the 28-layer Talker drifts 1.6e-01 from the single-stream reference -- reordering
             * rather than error, but indistinguishable from a bad index in a maximum. */
            #pragma unroll
            for(int u=0;u<U;++u) s[b]+=w[u]*xb[i+32*u]; }
    }
    for(;i<cols;i+=32){ float wv=__bfloat162float(wr[i]);
        #pragma unroll
        for(int b=0;b<NB;++b) if(b<B) s[b]+=wv*X[(size_t)b*cols+i]; }
    #pragma unroll
    for(int b=0;b<NB;++b){ if(b>=B) break; float v=s[b];
        #pragma unroll
        for(int o=16;o>0;o>>=1) v+=__shfl_down_sync(0xffffffffu,v,o);
        if(lane==0) Y[(size_t)b*rows+row]=v; }
}
/* Unroll depth, i.e. how many weight loads a lane keeps in flight. The kernel is latency-bound,
 * not bandwidth-bound -- its cost barely moves between two and eight lanes -- so this is the
 * knob that matters. Swept on the hardware rather than picked. */
static int mm_unroll(void){
    static int u=-1;
    if(u<0){ const char *e=getenv("QWEN_CUDA_MM_UNROLL"); u=e&&*e?atoi(e):4;
             if(u!=4&&u!=8&&u!=16) u=4; }
    return u;
}
/* Same two treatments the bf16 kernel got, for the same two measured reasons: NB as a template
 * parameter so s[] is not sized at QB_MAX, and four weight loads in flight because one load
 * feeding the very next instruction leaves the kernel latency-bound.
 *
 * Without them int8 was a trap. Halving the weight bytes made the SINGLE-stream path 22% faster
 * exactly as expected -- Talker 5.23 -> 4.64 ms/frame, CP 7.70 -> 5.97 -- while the BATCHED path
 * collapsed, Talker 4.82 -> 15.24 and CP 10.60 -> 40.48, purely because this kernel had been
 * left behind. A quantisation that helps one lane and cripples eight would have read as "int8 is
 * no good on the GPU". */
template<int U,int NB>
__global__ void k_matmat_int8_u(const int8_t *__restrict__ W,const float *__restrict__ scale,
                                const float *__restrict__ X,float *__restrict__ Y,int rows,int cols,int B){
    int row=(blockIdx.x*blockDim.x+threadIdx.x)>>5, lane=threadIdx.x&31; if(row>=rows) return;
    const int8_t *wr=W+(size_t)row*cols; float s[NB];
    #pragma unroll
    for(int b=0;b<NB;++b) s[b]=0.f;
    int i=lane; const int span=32*U;
    for(;i+span-32<cols;i+=span){
        float w[U];
        #pragma unroll
        for(int u=0;u<U;++u) w[u]=(float)wr[i+32*u];
        #pragma unroll
        for(int b=0;b<NB;++b) if(b<B){ const float *xb=X+(size_t)b*cols;
            /* one at a time, in the scalar loop's order -- see k_matmat_bf16_u */
            #pragma unroll
            for(int u=0;u<U;++u) s[b]+=w[u]*xb[i+32*u]; }
    }
    for(;i<cols;i+=32){ float wv=(float)wr[i];
        #pragma unroll
        for(int b=0;b<NB;++b) if(b<B) s[b]+=wv*X[(size_t)b*cols+i]; }
    float sc=scale[row];
    #pragma unroll
    for(int b=0;b<NB;++b){ if(b>=B) break; float v=s[b];
        #pragma unroll
        for(int o=16;o>0;o>>=1) v+=__shfl_down_sync(0xffffffffu,v,o);
        if(lane==0) Y[(size_t)b*rows+row]=sc*v; }
}
/* NB templated for the same reason; the four-way unroll is left out because each lane's loads
 * here are nibbles inside a shared block header, not independent addresses, so the pattern does
 * not transfer unchanged and no measurement justifies guessing at it yet. */
template<int NB>
__global__ void k_matmat_q4_0_u(const q4blk *__restrict__ W,const float *__restrict__ X,
                                float *__restrict__ Y,int rows,int cols,int B){
    int row=(blockIdx.x*blockDim.x+threadIdx.x)>>5, lane=threadIdx.x&31; if(row>=rows) return;
    int nb=cols>>5; const q4blk *wr=W+(size_t)row*nb; float s[NB];
    #pragma unroll
    for(int b=0;b<NB;++b) s[b]=0.f;
    for(int c=lane;c<cols;c+=32){ const q4blk *bk=wr+(c>>5); int ic=c&31;
        unsigned char byte=bk->qs[ic>>1]; int nib=(ic&1)?(byte>>4):(byte&0x0F);
        float w=(float)(nib-8)*__half2float(bk->scale);
        #pragma unroll
        for(int b=0;b<NB;++b) if(b<B) s[b]+=w*X[(size_t)b*cols+c]; }
    #pragma unroll
    for(int b=0;b<NB;++b){ if(b>=B) break; float v=s[b];
        #pragma unroll
        for(int o=16;o>0;o>>=1) v+=__shfl_down_sync(0xffffffffu,v,o);
        if(lane==0) Y[(size_t)b*rows+row]=v; }
}
/* cuBLAS for the shapes where it actually wins.
 *
 * The hand kernel gives one warp to each output row, so the block count is set by `rows` alone.
 * At rows=1024 that is 128 blocks on 84 SMs -- one and a half blocks per SM, far too few warps
 * to hide memory latency -- while at rows=6144 it is 768 blocks and the kernel runs 2x faster
 * per byte. Measured on an A6000 at B=8, GB/s on the weights, hand vs cublasGemmEx:
 *
 *   q/o     1024x1024   225  vs  198   0.88x   hand wins, keep it
 *   cp q    2048x1024   299  vs  508   1.70x
 *   gate+up 6144x1024   365  vs  501   1.37x
 *   down    1024x3072   191  vs  573   3.01x
 *
 * So the rule is shape-based and comes from that table: anything wider than 1024x1024 goes to
 * cuBLAS, the square case stays on the hand kernel, and nothing is chosen by intuition.
 *
 * The cost is precision, not correctness: cublasGemmEx requires A and B to share a type, so the
 * activations are narrowed to bf16 while the hand kernel keeps them in f32 against bf16 weights.
 * The accumulator stays f32 either way. This is a different generation, not a wrong one, and it
 * is why the flag ships off until the audio has been listened to. */
#define CUBLAS_WS_BYTES (32u<<20)
#define CUBLAS_XB_MAX ((size_t)QB_MAX*8192)   /* B x cols staging, allocated once at init */
static cublasHandle_t g_cublas = NULL;
static __nv_bfloat16 *g_xb = NULL; static size_t g_xb_cap = 0;
static int cublas_mm_enabled(void){
    static int on=-1;
    if(on<0){ const char *e=getenv("QWEN_CUDA_CUBLAS"); on=(e&&*e&&*e!='0')?1:0;
        if(on){
            if(cublasCreate(&g_cublas)!=CUBLAS_STATUS_SUCCESS){
                fprintf(stderr,"[cuda] cublasCreate failed — keeping the hand matmat\n"); on=0;
            }else{
                cublasSetStream(g_cublas,cudaStreamPerThread);
                /* A user-provided workspace is what makes cuBLAS safe to capture in a CUDA
                 * graph: without it the library may allocate on first use of a shape, and an
                 * allocation inside a capturing stream fails the capture.  The body must NOT be
                 * pre-executed to dodge that -- capture records without executing, so running
                 * the body first would apply the step twice to the KV cache and the residual. */
                void *ws=NULL;
                if(cudaMalloc(&ws,CUBLAS_WS_BYTES)==cudaSuccess)
                    cublasSetWorkspace(g_cublas,ws,CUBLAS_WS_BYTES);
                else
                    fprintf(stderr,"[cuda] cuBLAS workspace alloc failed — graph capture may fall back\n");
                if(cudaMalloc(&g_xb,CUBLAS_XB_MAX*sizeof(__nv_bfloat16))==cudaSuccess) g_xb_cap=CUBLAS_XB_MAX;
                else { g_xb=NULL; g_xb_cap=0; on=0;
                       fprintf(stderr,"[cuda] cuBLAS activation staging alloc failed — keeping the hand matmat\n"); }
                fprintf(stderr,"[cuda] batched matmat: cuBLAS for shapes wider than 1024x1024 "
                               "(activations narrowed to bf16); QWEN_CUDA_CUBLAS=0 disables\n");
            }
        }
    }
    return on;
}
__global__ void k_f32_to_bf16(const float *__restrict__ src,__nv_bfloat16 *__restrict__ dst,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) dst[i]=__float2bfloat16(src[i]);
}
/* True when cuBLAS measured faster for this shape. The square 1024x1024 case is the exception
 * the table above records, not an oversight. */
static inline int cublas_wins(int rows,int cols){ return !(rows<=1024 && cols<=1024); }
/* Block size for the batched matmats, swept on hardware.
 *
 * ncu on the rows=1024 shapes reported Waves Per SM 0.25 and achieved occupancy 25% against a
 * theoretical 100%: the kernel launches 128 blocks where the device holds 504, so three
 * quarters of the GPU is idle, and DRAM throughput sits at 22%.
 *
 * One warp owns one output row, so the block count is rows*32/blockDim. Shrinking the block
 * multiplies the blocks WITHOUT changing what any warp does -- same elements, same order, same
 * reduction -- so unlike split-K, which chops the reduction dimension and reassociates the sum,
 * this is bit-identical. Split-K was tried first and was slower at every setting: quartering
 * each block's work left roughly two unrolled iterations per block, and the per-block fixed cost
 * ate the gain. */
static int mm_tpb(void){
    static int t=-1;
    if(t<0){ const char *e=getenv("QWEN_CUDA_MM_TPB"); t=e&&*e?atoi(e):64;
             if(t!=32&&t!=64&&t!=128&&t!=256) t=64; }
    return t;
}
static inline void mvB(int prec,const void*W,const float*scale,const float*X,float*Y,int rows,int cols,int B){
    if(prec==0 && cublas_mm_enabled() && cublas_wins(rows,cols)){
        size_t need=(size_t)B*cols;
        /* No allocation here on purpose: mvB runs inside the graph capture, and cudaMalloc in a
         * capturing stream fails.  The staging buffer is sized once at init; an unexpectedly
         * wide shape falls through to the hand kernel instead of allocating. */
        if(g_xb && need<=g_xb_cap){
            k_f32_to_bf16<<<CEIL((int)need,TPB),TPB>>>(X,g_xb,(int)need);
            float alpha=1.f,beta=0.f;
            /* W is row-major [rows][cols], i.e. column-major [cols][rows]; the gemm is
             * Y_cm[rows x B] = W_cm^T * X_cm, with X row-major [B][cols] = column-major [cols][B]. */
            if(cublasGemmEx(g_cublas,CUBLAS_OP_T,CUBLAS_OP_N,rows,B,cols,&alpha,
                            W,CUDA_R_16BF,cols, g_xb,CUDA_R_16BF,cols, &beta,
                            Y,CUDA_R_32F,rows, CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT)==CUBLAS_STATUS_SUCCESS)
                return;
        }
        /* anything unsupported falls through to the hand kernel rather than failing the step */
    }

    const int tpb=mm_tpb();
    int grid=CEIL(rows*32,tpb);
    if(prec!=0){
#define MMQ(NB) do{ if(prec==2) k_matmat_q4_0_u<NB><<<grid,tpb>>>((const q4blk*)W,X,Y,rows,cols,B); \
                    else { int u=mm_unroll(); \
                           if(u==16)     k_matmat_int8_u<16,NB><<<grid,tpb>>>((const int8_t*)W,scale,X,Y,rows,cols,B); \
                           else if(u==8) k_matmat_int8_u< 8,NB><<<grid,tpb>>>((const int8_t*)W,scale,X,Y,rows,cols,B); \
                           else          k_matmat_int8_u< 4,NB><<<grid,tpb>>>((const int8_t*)W,scale,X,Y,rows,cols,B); } }while(0)
      switch(B){
        case  1: MMQ( 1); break;  case  2: MMQ( 2); break;  case  3: MMQ( 3); break;
        case  4: MMQ( 4); break;  case  5: MMQ( 5); break;  case  6: MMQ( 6); break;
        case  7: MMQ( 7); break;  case  8: MMQ( 8); break;  case  9: MMQ( 9); break;
        case 10: MMQ(10); break;  case 11: MMQ(11); break;  case 12: MMQ(12); break;
        case 13: MMQ(13); break;  case 14: MMQ(14); break;  case 15: MMQ(15); break;
        default: MMQ(QB_MAX); break;
      }
#undef MMQ
    }
    else { int u=mm_unroll();
#define MM_U(NB) do{ if(u==16)     k_matmat_bf16_u<16,NB><<<grid,tpb>>>((const __nv_bfloat16*)W,X,Y,rows,cols,B); \
                     else if(u==8) k_matmat_bf16_u< 8,NB><<<grid,tpb>>>((const __nv_bfloat16*)W,X,Y,rows,cols,B); \
                     else          k_matmat_bf16_u< 4,NB><<<grid,tpb>>>((const __nv_bfloat16*)W,X,Y,rows,cols,B); }while(0)
           switch(B){
             case  1: MM_U( 1); break;  case  2: MM_U( 2); break;  case  3: MM_U( 3); break;
             case  4: MM_U( 4); break;  case  5: MM_U( 5); break;  case  6: MM_U( 6); break;
             case  7: MM_U( 7); break;  case  8: MM_U( 8); break;  case  9: MM_U( 9); break;
             case 10: MM_U(10); break;  case 11: MM_U(11); break;  case 12: MM_U(12); break;
             case 13: MM_U(13); break;  case 14: MM_U(14); break;  case 15: MM_U(15); break;
             default: MM_U(QB_MAX); break;
           }
#undef MM_U
         }
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
                              int n_heads,int head_dim,const int *d_pos,int stride,int B,
                              const unsigned char *d_act){
    int half=head_dim/2, per=n_heads*half, gid=blockIdx.x*blockDim.x+threadIdx.x;
    if(gid>=B*per) return; int b=gid/per, rem=gid%per;
    if(d_act && !d_act[b]) return;   /* lane is not stepping: d_pos[b] is stale, do not index with it */
    const float *cosp=cos_base+(size_t)d_pos[b]*half, *sinp=sin_base+(size_t)d_pos[b]*half;
    int h=rem/half, i=rem%half; float *xh=X+(size_t)b*stride+(size_t)h*head_dim;
    float c=cosp[i], sn=sinp[i], x1=xh[i], x2=xh[i+half];
    xh[i]=x1*c-x2*sn; xh[i+half]=x2*c+x1*sn;
}
/* KV per sequence: kc/vc layer base = [B][kv_max][kvd]; K/V = [B][kvd] */
__global__ void k_kv_store_b(float *kc,float *vc,const float *K,const float *V,int kvd,const int *d_pos,int kv_max,int B,
                             const unsigned char *d_act,const int *d_slot){
    int gid=blockIdx.x*blockDim.x+threadIdx.x; if(gid>=B*kvd) return; int b=gid/kvd, i=gid%kvd;
    if(d_act && !d_act[b]) return;   /* never write a paused slot's KV: it owns that position */
    /* activations are dense in b; the KV belongs to the slot this lane stands for */
    int sb = d_slot ? d_slot[b] : b;
    size_t off=(size_t)sb*kv_max*kvd+(size_t)d_pos[b]*kvd+i;
    kc[off]=K[(size_t)b*kvd+i]; vc[off]=V[(size_t)b*kvd+i];
}
/* one block per (b,head): blk=b*n_heads+h, blockDim=hd. KV base per seq = [kv_max][kvd] */
__global__ void k_attn_b(const float *Q,const float *K,const float *V,float *O,
                         int n_heads,int n_kv,int hd,float scale,const int *d_pos,int kv_max,int qd,int kvd,
                         const unsigned char *d_act,const int *d_slot){
    int blk=blockIdx.x, b=blk/n_heads, h=blk%n_heads, t=threadIdx.x; if(t>=hd) return;
    /* whole block shares b (blk = b*n_heads + h), so this return is uniform and the
     * __syncthreads() below stay collective */
    if(d_act && !d_act[b]) return;
    int sb = d_slot ? d_slot[b] : b;   /* dense lane -> owning KV slot */
    int kvh=h/(n_heads/n_kv), valid=d_pos[b]+1;
    const float *Kb=K+(size_t)sb*kv_max*kvd, *Vb=V+(size_t)sb*kv_max*kvd;
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
/* Same attention, one warp per (b,head), no barriers.
 *
 * nsys put this kernel at 24.9% of all GPU time, second only to the matmats, running at about
 * 78 GB/s.  The reason is in the loop above: for EVERY key position it does a shared-memory
 * tree reduction with a __syncthreads() at each level, so roughly eight block-wide barriers per
 * position -- ten thousand of them per (lane, head) per layer at a realistic kv_len -- across a
 * block of only two warps that spends most of its life waiting at them.
 *
 * This is bit-identical to it, not merely equivalent.  Give lane t the elements t, t+32, t+64...
 * and reduce those privately first: for hd=64 that single local add IS the tree's s=32 step, and
 * the five __shfl_down steps that follow ARE its s=16..1 steps, in that order.  The online
 * softmax recurrence is untouched and still walks positions in sequence, so every rounding in
 * the kernel happens exactly where it happened before.
 *
 * E = hd/32 is a template parameter so the private reduction is fully unrolled; anything that is
 * not 32, 64 or 128 wide keeps the original kernel. */
template<int E>
__global__ void k_attn_bw(const float *Q,const float *K,const float *V,float *O,
                          int n_heads,int n_kv,int hd,float scale,const int *d_pos,int kv_max,int qd,int kvd,
                          const unsigned char *d_act,const int *d_slot){
    int blk=blockIdx.x, b=blk/n_heads, h=blk%n_heads, t=threadIdx.x;
    if(d_act && !d_act[b]) return;
    int sb = d_slot ? d_slot[b] : b;
    int kvh=h/(n_heads/n_kv), valid=d_pos[b]+1;
    const float *Kb=K+(size_t)sb*kv_max*kvd, *Vb=V+(size_t)sb*kv_max*kvd;
    const float *Qh=Q+(size_t)b*qd+(size_t)h*hd;
    float qv[E];
    #pragma unroll
    for(int e=0;e<E;++e) qv[e]=Qh[t+32*e];
    float m=-1e30f, denom=0.f, acc[E];
    #pragma unroll
    for(int e=0;e<E;++e) acc[e]=0.f;
    for(int j=0;j<valid;++j){
        const float *k=Kb+((size_t)j*n_kv+kvh)*hd;
        float p[E];
        /* __fmul_rn, not `*`: the original wrote each product to shared memory before adding,
         * which the compiler cannot fuse.  Written as a plain multiply here it contracts
         * `p[0]+=p[1]` into an FMA -- one rounding instead of two -- and the batched Talker then
         * drifts 3.98e-01 from the single-stream reference over 28 layers.  Arguably more
         * accurate, but not the same generation, and the point of this rewrite is speed at
         * identical output. */
        #pragma unroll
        for(int e=0;e<E;++e) p[e]=__fmul_rn(qv[e],k[t+32*e]);
        /* the tree's wide strides, in its order */
        #pragma unroll
        for(int st=E/2;st>0;st>>=1)
            #pragma unroll
            for(int e=0;e<st;++e) p[e]+=p[e+st];
        float v=p[0];
        #pragma unroll
        for(int sft=16;sft>0;sft>>=1) v+=__shfl_down_sync(0xffffffffu,v,sft);
        float score=__shfl_sync(0xffffffffu,v,0)*scale;
        float mn=fmaxf(m,score), corr=expf(m-mn), pr=expf(score-mn);
        denom=denom*corr+pr;
        const float *vrow=Vb+((size_t)j*n_kv+kvh)*hd;
        #pragma unroll
        for(int e=0;e<E;++e) acc[e]=acc[e]*corr+pr*vrow[t+32*e];
        m=mn;
    }
    float *Oh=O+(size_t)b*qd+(size_t)h*hd;
    #pragma unroll
    for(int e=0;e<E;++e) Oh[t+32*e]=acc[e]/denom;
}
static inline void attn_b_launch(int blocks,int hd,const float *Q,const float *K,const float *V,float *O,
                                 int n_heads,int n_kv,float scale,const int *d_pos,int kv_max,int qd,int kvd,
                                 const unsigned char *d_act,const int *d_slot){
    if(hd==32)       k_attn_bw<1><<<blocks,32>>>(Q,K,V,O,n_heads,n_kv,hd,scale,d_pos,kv_max,qd,kvd,d_act,d_slot);
    else if(hd==64)  k_attn_bw<2><<<blocks,32>>>(Q,K,V,O,n_heads,n_kv,hd,scale,d_pos,kv_max,qd,kvd,d_act,d_slot);
    else if(hd==128) k_attn_bw<4><<<blocks,32>>>(Q,K,V,O,n_heads,n_kv,hd,scale,d_pos,kv_max,qd,kvd,d_act,d_slot);
    else             k_attn_b<<<blocks,hd,hd*sizeof(float)>>>(Q,K,V,O,n_heads,n_kv,hd,scale,d_pos,kv_max,qd,kvd,d_act,d_slot);
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
    unsigned char *d_act;                             /* device uint8[B]: which lanes step */
    int *d_slot;                                      /* device int[B]: dense lane -> owning KV slot */
    int B_eff;                                        /* lanes actually computed this step (<= B) */
    int slot_identity;                                /* 1 when d_slot is the identity map */
    float *h_emb; int *h_pos; int *h_slot; float *h_hid;   /* host staging for compaction */
    /* One CUDA graph per effective lane count: the grid dimensions and the by-value B
     * argument of every kernel are baked in at capture, so a graph taken at four lanes
     * cannot be replayed at three. */
    cudaGraphExec_t gexec[QB_MAX+1]; unsigned char gmode[QB_MAX+1];   /* 0 untried, 1 graph, 2 disabled */
} cuda_talker_batch_t;

/* Replay a batched body from a CUDA graph instead of relaunching its kernels.
 *
 * The batched bodies are the ones the server actually runs, and neither had a graph: only
 * the single-stream talker and CP did.  One batched Talker step is 28 layers x 19 launches
 * and one batched CP frame is 15 passes x 5 layers x 19, so a frame issues on the order of
 * two thousand launches whose arguments never change from frame to frame.
 *
 * Nothing about the computation changes.  The same kernels run in the same order on the same
 * pointers; positions, the active mask and the lane->slot map all live in device buffers the
 * graph reads at replay, so they stay free to vary.  What is baked in at capture is the grid
 * geometry and the by-value lane count, which is why the cache is keyed on the effective lane
 * count rather than holding a single graph.
 *
 * Capture is skipped entirely for n==0 and falls back to plain launches for good if it fails,
 * so a driver that refuses capture costs one wasted attempt per lane count and nothing else. */
static int batch_graph_enabled(void){
    static int t=-1;
    if(t<0){ const char *e=getenv("QWEN_CUDA_BATCH_GRAPH"); t=(!e||!e[0])?1:(e[0]!='0'); }
    return t;
}
template<typename T, void (*BODY)(T*)>
static void batch_graph_run(T *s,int n){
    if(n<1||n>QB_MAX||!batch_graph_enabled()){ BODY(s); return; }
    if(s->gmode[n]==2){ BODY(s); return; }
    if(s->gmode[n]==0){
        cudaGraph_t g=NULL;
        if(cudaStreamBeginCapture(cudaStreamPerThread,cudaStreamCaptureModeThreadLocal)!=cudaSuccess){
            s->gmode[n]=2; BODY(s); return;
        }
        BODY(s);
        if(cudaStreamEndCapture(cudaStreamPerThread,&g)!=cudaSuccess||!g){
            s->gmode[n]=2; cudaGetLastError(); BODY(s); return;
        }
        cudaError_t rc=cudaGraphInstantiate(&s->gexec[n],g,0);
        cudaGraphDestroy(g);
        if(rc!=cudaSuccess){
            fprintf(stderr,"batched graph instantiate failed at %d lanes (%s); using plain launches\n",
                    n,cudaGetErrorString(rc));
            s->gmode[n]=2; BODY(s); return;   /* capture consumed the work: re-issue it */
        }
        s->gmode[n]=1;
    }
    CK(cudaGraphLaunch(s->gexec[n],cudaStreamPerThread));
}

static void talker_body_batch(cuda_talker_batch_t *s){
    /* Bmax is the ALLOCATED lane count and fixes the KV layer stride; B is how many
     * lanes we actually compute this step.  They differ when the caller compacts idle
     * lanes away, and confusing them silently reindexes every layer's cache. */
    const int Bmax=s->B;
    int B=(s->B_eff>0&&s->B_eff<=s->B)?s->B_eff:s->B;
    int H=s->hidden,qd=s->q_dim,kvd=s->kv_dim,hd=s->head_dim,half=hd/2;
    int nh=s->n_heads,nkv=s->n_kv,inter=s->inter; float scale=1.f/sqrtf((float)hd);
    for(int l=0;l<s->n_layers;++l){
        k_rmsnorm_full_b<<<B,TPB,TPB*sizeof(float)>>>(s->x,s->inorm[l],s->xn,H,s->eps);
        mvB(s->prec,s->wq[l],s->wqs[l],s->xn,s->q,qd,H,B);
        mvB(s->prec,s->wk[l],s->wks[l],s->xn,s->k,kvd,H,B);
        mvB(s->prec,s->wv[l],s->wvs[l],s->xn,s->v,kvd,H,B);
        k_rmsnorm_ph_b<<<B*nh,TPB,TPB*sizeof(float)>>>(s->q,s->qn[l],hd,nh,qd,s->eps);
        k_rmsnorm_ph_b<<<B*nkv,TPB,TPB*sizeof(float)>>>(s->k,s->kn[l],hd,nkv,kvd,s->eps);
        k_rope_neox_b<<<CEIL(B*nh*half,TPB),TPB>>>(s->q,s->rope_cos,s->rope_sin,nh,hd,s->d_pos,qd,B,s->d_act);
        k_rope_neox_b<<<CEIL(B*nkv*half,TPB),TPB>>>(s->k,s->rope_cos,s->rope_sin,nkv,hd,s->d_pos,kvd,B,s->d_act);
        k_trunc_bf16<<<CEIL(B*kvd,TPB),TPB>>>(s->k,B*kvd);
        k_trunc_bf16<<<CEIL(B*kvd,TPB),TPB>>>(s->v,B*kvd);
        float *Kl=s->kcache+(size_t)l*Bmax*s->kv_max*kvd, *Vl=s->vcache+(size_t)l*Bmax*s->kv_max*kvd;
        k_kv_store_b<<<CEIL(B*kvd,TPB),TPB>>>(Kl,Vl,s->k,s->v,kvd,s->d_pos,s->kv_max,B,s->d_act,s->d_slot);
        attn_b_launch(B*nh,hd,s->q,Kl,Vl,s->attn,nh,nkv,scale,s->d_pos,s->kv_max,qd,kvd,s->d_act,s->d_slot);
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
    CK(cudaMalloc(&s->d_act,(size_t)B));
    CK(cudaMalloc(&s->d_slot,B*sizeof(int)));
    { int *ids=(int*)malloc(B*sizeof(int)); for(int i=0;i<B;++i) ids[i]=i;
      CK(cudaMemcpy(s->d_slot,ids,B*sizeof(int),cudaMemcpyHostToDevice)); free(ids); }
    s->B_eff=0; s->slot_identity=1;
    s->h_emb=(float*)malloc((size_t)B*s->hidden*sizeof(float));
    s->h_hid=(float*)malloc((size_t)B*s->hidden*sizeof(float));
    s->h_pos=(int*)malloc(B*sizeof(int)); s->h_slot=(int*)malloc(B*sizeof(int));
    if(getenv("QWEN_CUDA_VERBOSE")){
        fprintf(stderr,"[cuda batch] B=%d L=%d H=%d qd=%d kvd=%d inter=%d kv_max=%d prec=%d\n",
                B,L,H,qd,kvd,inter,s->kv_max,s->prec);
        fprintf(stderr,"[cuda batch] x=%p xn=%p q=%p k=%p v=%p attn=%p proj=%p gate=%p gu=%p kc=%p vc=%p\n",
                (void*)s->x,(void*)s->xn,(void*)s->q,(void*)s->k,(void*)s->v,(void*)s->attn,
                (void*)s->proj,(void*)s->gate,(void*)s->gu,(void*)s->kcache,(void*)s->vcache);
        fprintf(stderr,"[cuda batch] wq[0]=%p wk[0]=%p wv[0]=%p wo[0]=%p wgu[0]=%p wdn[0]=%p\n",
                (void*)s->wq[0],(void*)s->wk[0],(void*)s->wv[0],(void*)s->wo[0],(void*)s->wgu[0],(void*)s->wdn[0]);
    }
    return s;
}
/* embeds=[B][H] host, pos_arr=[B] host; hidden_out=[B][H] host (final-normed). */
/* QWEN_CUDA_BATCH_COMPACT=1: compute only the lanes that are actually stepping.
 * The kernels otherwise run every configured lane, so a server with --batch-size 8 and four
 * requests in flight spends half its GPU work on empty lanes -- and a partly loaded server is
 * the normal case, not the exception.  Default OFF until measured on hardware. */
static int batch_compact_enabled(void){
    static int v=-1;
    if(v<0){ const char *e=getenv("QWEN_CUDA_BATCH_COMPACT"); v=(e&&*e&&*e!='0')?1:0; }
    return v;
}
extern "C" void qwen_cuda_talker_batch_step(void *st,const float *embeds,const int *pos_arr,float *hidden_out,
                                           const unsigned char *active){
    cuda_talker_batch_t *s=(cuda_talker_batch_t*)st; int B=s->B,H=s->hidden;

    if(batch_compact_enabled() && active){
        int n=0;
        for(int b=0;b<B;++b) if(active[b]){
            s->h_slot[n]=b; s->h_pos[n]=pos_arr[b];
            memcpy(s->h_emb+(size_t)n*H, embeds+(size_t)b*H, (size_t)H*sizeof(float));
            ++n;
        }
        if(n==0) return;                      /* no lane is stepping: nothing to launch */
        if(n<B){
            /* dense lanes 0..n-1 stand for slots h_slot[0..n-1]; the KV stays addressed by slot */
            CK(cudaMemcpy(s->x,s->h_emb,(size_t)n*H*sizeof(float),cudaMemcpyHostToDevice));
            CK(cudaMemcpy(s->d_pos,s->h_pos,(size_t)n*sizeof(int),cudaMemcpyHostToDevice));
            CK(cudaMemcpy(s->d_slot,s->h_slot,(size_t)n*sizeof(int),cudaMemcpyHostToDevice));
            { unsigned char ones[QB_MAX]; for(int i=0;i<n;++i) ones[i]=1;
              CK(cudaMemcpy(s->d_act,ones,(size_t)n,cudaMemcpyHostToDevice)); }
            s->B_eff=n;
            batch_graph_run<cuda_talker_batch_t,talker_body_batch>(s,n);
            CK(cudaStreamSynchronize(cudaStreamPerThread));
            if(hidden_out){
                CK(cudaMemcpy(s->h_hid,s->xn,(size_t)n*H*sizeof(float),cudaMemcpyDeviceToHost));
                for(int i=0;i<n;++i)
                    memcpy(hidden_out+(size_t)s->h_slot[i]*H, s->h_hid+(size_t)i*H, (size_t)H*sizeof(float));
            }
            s->B_eff=0; s->slot_identity=0;
            return;
        }
        /* n==B: every lane steps, so the dense form is the ordinary full-width path */
    }

    if(!s->slot_identity){   /* a previous compacted step left a permuted map behind */
        int ids[QB_MAX]; for(int i=0;i<B;++i) ids[i]=i;
        CK(cudaMemcpy(s->d_slot,ids,(size_t)B*sizeof(int),cudaMemcpyHostToDevice));
        s->slot_identity=1;
    }
    s->B_eff=0;
    CK(cudaMemcpy(s->x,embeds,(size_t)B*H*sizeof(float),cudaMemcpyHostToDevice));
    CK(cudaMemcpy(s->d_pos,pos_arr,B*sizeof(int),cudaMemcpyHostToDevice));
    /* Lanes the caller is not stepping keep a stale pos_arr[b] from whatever request last
     * used the slot.  Every position-indexed kernel derives an address from it, so without
     * this mask an idle lane reads and writes out of bounds -- observed as
     * "an illegal memory access was encountered".  NULL means every lane steps. */
    { unsigned char all[QB_MAX]; if(!active){ for(int i=0;i<B;++i) all[i]=1; }
      CK(cudaMemcpy(s->d_act, active?active:all, (size_t)B, cudaMemcpyHostToDevice)); }
    batch_graph_run<cuda_talker_batch_t,talker_body_batch>(s,B);
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
    cudaFree(s->gu);cudaFree(s->d_pos);cudaFree(s->d_act);cudaFree(s->d_slot);
    for(int i=0;i<=QB_MAX;++i) if(s->gmode[i]==1) cudaGraphExecDestroy(s->gexec[i]);
    free(s->h_emb);free(s->h_hid);free(s->h_pos);free(s->h_slot); free(s);   /* weights are shared — not freed here */
}

/* Correctness (batched row b MUST equal a single-stream run with the same embeds/pos) +
 * throughput scaling (batched ms/frame for B seqs vs single ms/frame for 1 seq). */
static double now_ms(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e3+t.tv_nsec/1e6; }
extern "C" void qwen_cuda_talker_free(void *);   /* defined below (after the CP section) */
extern "C" void *qwen_cuda_cp_batch_init(void *, int);   /* defined below (CP section) */
extern "C" void  qwen_cuda_cp_batch_step(void *, float *, const int *, const unsigned char *);
extern "C" void  qwen_cuda_cp_batch_free(void *);
/* Partial-occupancy correctness: a lane that is NOT stepping this step must keep its KV
 * intact, and a lane that IS stepping must read and write the cache of its own slot.
 *
 * The lockstep test above passes active=NULL, so every lane steps and QWEN_CUDA_BATCH_COMPACT
 * never reorders anything -- it proves the flag is inert, not that it is correct.  The
 * dangerous failure mode of compaction is silent: a lane compacted to a different dense index
 * reads another request's history and produces plausible audio, which no crash and no
 * throughput number would reveal.  So compare against the only reference that cannot be wrong,
 * B independent one-wide states, each stepped exactly on the steps where its lane was active.
 *
 * Idle lanes are handed a wrong position and a wrong embedding on purpose: if the mask is
 * honoured they are never read, and if it is not the reference disagrees immediately. */
static int batch_mask_selftest(void *S, int B, int H, int steps){
    if(B<2) return 0;
    void *tst=qwen_cuda_talker_batch_init(S,B);
    if(!tst){ fprintf(stderr,"batch mask selftest: batch init failed (B=%d)\n",B); return 1; }
    void **ref=(void**)calloc(B,sizeof(void*));
    for(int b=0;b<B;++b){
        ref[b]=qwen_cuda_talker_batch_init(S,1);
        if(!ref[b]){ fprintf(stderr,"batch mask selftest: reference init failed\n");
                     for(int j=0;j<b;++j) qwen_cuda_talker_batch_free(ref[j]);
                     free(ref); qwen_cuda_talker_batch_free(tst); return 1; }
    }
    float *eB=(float*)malloc((size_t)B*H*sizeof(float));
    float *hB=(float*)malloc((size_t)B*H*sizeof(float));
    float *e1=(float*)malloc((size_t)H*sizeof(float));
    float *h1=(float*)malloc((size_t)H*sizeof(float));
    int *posB=(int*)malloc((size_t)B*sizeof(int));
    int *posc=(int*)calloc(B,sizeof(int));      /* each lane's own position counter */
    unsigned char act[QB_MAX];
    double maxdiff=0; long compared=0, idled=0;
    for(int p=0;p<steps;++p){
        int any=0;
        for(int b=0;b<B;++b){
            act[b]=(unsigned char)(((p+b)%3)!=0);   /* each lane idles on a different step */
            if(act[b]){
                any=1;
                for(int i=0;i<H;++i)
                    eB[(size_t)b*H+i]=0.02f*sinf(0.11f*(i+1)+0.3f*posc[b]+1.7f*b);
                posB[b]=posc[b];
            }else{
                ++idled;
                for(int i=0;i<H;++i) eB[(size_t)b*H+i]=7.0f;   /* must never be read */
                posB[b]=0;                                     /* wrong on purpose */
            }
        }
        if(!any) continue;
        qwen_cuda_talker_batch_step(tst,eB,posB,hB,act);
        for(int b=0;b<B;++b){
            if(!act[b]) continue;
            memcpy(e1,eB+(size_t)b*H,(size_t)H*sizeof(float));
            int one=posc[b];
            unsigned char on=1;
            qwen_cuda_talker_batch_step(ref[b],e1,&one,h1,&on);
            for(int i=0;i<H;++i){ double d=fabs((double)hB[(size_t)b*H+i]-(double)h1[i]);
                                  if(d>maxdiff) maxdiff=d; }
            ++compared; ++posc[b];
        }
    }
    int bad=!(maxdiff<1e-3);
    fprintf(stderr,"batch mask selftest (B=%d, %d steps, %ld lane-steps compared, %ld idled): "
                   "max|masked-reference|=%.2e (%s)\n",
            B,steps,compared,idled,maxdiff, bad?"FAIL":"PASS");
    free(eB);free(hB);free(e1);free(h1);free(posB);free(posc);
    for(int b=0;b<B;++b) qwen_cuda_talker_batch_free(ref[b]);
    free(ref); qwen_cuda_talker_batch_free(tst);
    return bad;
}

/* Is the hand-written batched matvec leaving anything on the table against cuBLAS?
 *
 * The kernel is latency-bound at roughly 180 GB/s on a 768 GB/s card, and a PyTorch/vLLM
 * implementation of this model would route every linear through cuBLAS and its tensor cores.
 * So measure it rather than argue about it, on the shapes this engine actually issues.
 *
 * The comparison is not like for like and the printout says so: cublasGemmEx requires A and B
 * to carry the SAME type, so the activations have to be narrowed to bf16, while the hand kernel
 * keeps them in f32 against bf16 weights. If cuBLAS wins by a wide margin the narrowing is
 * worth pricing; if it does not, the question is closed and the activations stay f32. */
extern "C" void qwen_cuda_gemm_probe(int B){
    cublasHandle_t h; if(cublasCreate(&h)!=CUBLAS_STATUS_SUCCESS){ fprintf(stderr,"gemm probe: cublasCreate failed\n"); return; }
    cublasSetStream(h,cudaStreamPerThread);
    struct { int rows, cols; const char *what; } shp[] = {
        {1024,1024,"talker q/o      "},{6144,1024,"talker gate+up  "},
        {1024,3072,"talker down     "},{2048,1024,"cp q            "},
        {6144,1024,"cp gate+up      "},{1024,3072,"cp down         "},
    };
    fprintf(stderr,"\n  GEMM probe at B=%d (weights bf16; hand kernel keeps activations f32, cuBLAS narrows them to bf16)\n",B);
    fprintf(stderr,"  %-16s %10s %10s %10s %10s %8s\n","shape","hand ms","hand GB/s","cublas ms","cublas GB/s","ratio");
    const int IT=200;
    for(size_t k=0;k<sizeof(shp)/sizeof(shp[0]);++k){
        int rows=shp[k].rows, cols=shp[k].cols;
        size_t nw=(size_t)rows*cols;
        __nv_bfloat16 *W; float *X,*Y; __nv_bfloat16 *Xb;
        if(cudaMalloc(&W,nw*sizeof(__nv_bfloat16))!=cudaSuccess) break;
        CK(cudaMalloc(&X,(size_t)B*cols*sizeof(float)));
        CK(cudaMalloc(&Xb,(size_t)B*cols*sizeof(__nv_bfloat16)));
        CK(cudaMalloc(&Y,(size_t)B*rows*sizeof(float)));
        CK(cudaMemset(W,0,nw*sizeof(__nv_bfloat16)));
        CK(cudaMemset(X,0,(size_t)B*cols*sizeof(float)));
        CK(cudaMemset(Xb,0,(size_t)B*cols*sizeof(__nv_bfloat16)));
        double bytes=(double)nw*2.0;            /* the weights: what both kernels must read */
        int grid=CEIL(rows*32,TPB);
        k_matmat_bf16_u<4,QB_MAX><<<grid,TPB>>>(W,X,Y,rows,cols,B);
        CK(cudaStreamSynchronize(cudaStreamPerThread));
        double t0=now_ms();
        for(int i=0;i<IT;++i) k_matmat_bf16_u<4,QB_MAX><<<grid,TPB>>>(W,X,Y,rows,cols,B);
        CK(cudaStreamSynchronize(cudaStreamPerThread));
        double th=(now_ms()-t0)/IT;
        /* Y_cm[rows x B] = Wc^T * Xc, with W row-major [rows][cols] = col-major [cols][rows] */
        float alpha=1.f,beta=0.f;
        cublasStatus_t st=cublasGemmEx(h,CUBLAS_OP_T,CUBLAS_OP_N,rows,B,cols,&alpha,
                                       W,CUDA_R_16BF,cols, Xb,CUDA_R_16BF,cols, &beta,
                                       Y,CUDA_R_32F,rows, CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT);
        if(st!=CUBLAS_STATUS_SUCCESS){
            fprintf(stderr,"  %-16s %10.3f %10.0f   cublasGemmEx unsupported (status %d)\n",
                    shp[k].what,th,bytes/th/1e6,(int)st);
        }else{
            CK(cudaStreamSynchronize(cudaStreamPerThread));
            t0=now_ms();
            for(int i=0;i<IT;++i) cublasGemmEx(h,CUBLAS_OP_T,CUBLAS_OP_N,rows,B,cols,&alpha,
                                               W,CUDA_R_16BF,cols, Xb,CUDA_R_16BF,cols, &beta,
                                               Y,CUDA_R_32F,rows, CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT);
            CK(cudaStreamSynchronize(cudaStreamPerThread));
            double tc=(now_ms()-t0)/IT;
            fprintf(stderr,"  %-16s %10.3f %10.0f %10.3f %10.0f %7.2fx\n",
                    shp[k].what,th,bytes/th/1e6,tc,bytes/tc/1e6,th/tc);
        }
        cudaFree(W);cudaFree(X);cudaFree(Xb);cudaFree(Y);
    }
    cublasDestroy(h);
}

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
        qwen_cuda_talker_batch_step(Bs,eB,posB,hB,NULL);
        double sd=0;
        for(int b=0;b<B;++b) for(int i=0;i<H;++i){ double d=fabs(hB[(size_t)b*H+i]-h1[i]);
            if(d>maxdiff)maxdiff=d; if(d>sd)sd=d; }
        /* Print the first steps: a reordering drift starts near zero and compounds through the
         * residual stream, a wrong index is large on step 0.  The two look identical in a max. */
        if(p<6) fprintf(stderr,"  step %d: max|batched-single|=%.3e\n",p,sd);
    }
    fprintf(stderr,"batch selftest: correctness max|batched-single|=%.2e over %d steps (%s)\n",
            maxdiff,chk, maxdiff<1e-3?"PASS":"FAIL");
    qwen_cuda_talker_batch_free(Bs); qwen_cuda_talker_free(S);
    /* --- throughput: fresh states, time single (1 seq) vs batched (B seq) --- */
    S=qwen_cuda_talker_init(ctx); Bs=qwen_cuda_talker_batch_init(S,B);
    for(int i=0;i<H;++i) e1[i]=0.02f*sinf(0.11f*(i+1));
    for(int b=0;b<B;++b){ memcpy(eB+(size_t)b*H,e1,H*sizeof(float)); posB[b]=0; }
    qwen_cuda_talker_step(S,e1,h1,0); qwen_cuda_talker_batch_step(Bs,eB,posB,hB,NULL);  /* warmup */
    double t0=now_ms(); for(int p=1;p<=frames;++p) qwen_cuda_talker_step(S,e1,h1,p);
    double ts=(now_ms()-t0)/frames;
    t0=now_ms(); for(int p=1;p<=frames;++p){ for(int b=0;b<B;++b) posB[b]=p; qwen_cuda_talker_batch_step(Bs,eB,posB,hB,NULL); }
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
            qwen_cuda_cp_batch_step(CB,cxB,cpos,NULL);
            for(int b=0;b<B;++b) for(int i=0;i<cph;++i){ double d=fabs(cxB[(size_t)b*cph+i]-cx1[i]); if(d>cdiff)cdiff=d; }
        }
        fprintf(stderr,"batch selftest: CP correctness max|batched-single|=%.2e over %d steps (%s)\n",
                cdiff,cchk, cdiff<1e-3?"PASS":"FAIL");
        /* throughput: 16 passes/frame (CP KV holds 16), pos 0..15 lockstep */
        int passes=cpkv<16?cpkv:16;
        for(int i=0;i<cph;++i){ float v=0.02f*sinf(0.13f*(i+1)); cx1[i]=v; for(int b=0;b<B;++b) cxB[(size_t)b*cph+i]=v; }
        for(int p=0;p<passes;++p) qwen_cuda_cp_step(CS,cx1,p);                 /* warmup */
        for(int p=0;p<passes;++p){ for(int b=0;b<B;++b) cpos[b]=p; qwen_cuda_cp_batch_step(CB,cxB,cpos,NULL); }
        int NF=frames/4; if(NF<10) NF=10;
        double c0=now_ms();
        for(int f=0;f<NF;++f) for(int p=0;p<passes;++p) qwen_cuda_cp_step(CS,cx1,p);
        double cs=(now_ms()-c0)/NF;
        c0=now_ms();
        for(int f=0;f<NF;++f) for(int p=0;p<passes;++p){ for(int b=0;b<B;++b) cpos[b]=p; qwen_cuda_cp_batch_step(CB,cxB,cpos,NULL); }
        double cb=(now_ms()-c0)/NF;
        fprintf(stderr,"batch CP throughput (B=%d, %d passes/frame): single %.2f ms/f (1 seq) | batched %.2f ms/f (%d seq) | GAIN %.2fx\n",
                B,passes,cs,cb,B, B*cs/cb);
        fprintf(stderr,"batch GEN (Talker+CP) aggregate throughput GAIN (B=%d): %.2fx\n", B, B*(ts+cs)/(tb+cb));
        { extern double qwen_cuda_cp_batch_bench_fused(void *,int,int);
          double cf=qwen_cuda_cp_batch_bench_fused(CB,passes,NF);
          if(cf>0) fprintf(stderr,"batch CP FUSED CEILING (B=%d, %d passes, 1 sync/frame): %.2f ms/f "
                                  "vs %.2f ms/f served — %.0f%% of the loop is sync+copies\n",
                           B,passes,cf,cb,100.0*(cb-cf)/cb); }
        free(cx1);free(cxB);free(cpos);
        if(cdiff>=1e-3) maxdiff=cdiff;
    }
    if(CB) qwen_cuda_cp_batch_free(CB); if(CS) qwen_cuda_cp_free(CS);
    { extern void qwen_cuda_gemm_probe(int); qwen_cuda_gemm_probe(B); }
    /* Partial occupancy last: it needs B+1 live states, so run it after the others free theirs. */
    S=qwen_cuda_talker_init(ctx);
    int mbad = S ? batch_mask_selftest(S,B,H,24) : 1;
    if(S) qwen_cuda_talker_free(S);
    return (maxdiff<1e-3 && !mbad)?0:1;
}

/* ======================================================================== *
 *  GPU-resident fused Code Predictor step. Same layer structure as the Talker
 *  (reuses all the kernels above); differences: CP dims (hidden 1024, 5 layers),
 *  the CP KV is per-frame (built fresh each frame, no prefill upload), and there
 *  is no final norm (the caller applies cp_norm before the lm-head).
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
    qwen_tts_ctx_t *ctx;         /* kept so the batched state can reach cp_norm and the lm_heads */
} cuda_cp_t;

extern "C" void *qwen_cuda_cp_init(qwen_tts_ctx_t *ctx) {
    qwen_tts_config_t *c=&ctx->config;
    cuda_cp_t *s=(cuda_cp_t*)calloc(1,sizeof(*s));
    s->ctx=ctx;
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
    unsigned char *d_act;                             /* device uint8[B]: which lanes step */
    int *d_slot;                                      /* device int[B]: dense lane -> owning KV slot */
    int B_eff;                                        /* lanes actually computed this step (<= B) */
    int slot_identity;                                /* 1 when d_slot is the identity map */
    float *h_emb; int *h_pos; int *h_slot; float *h_hid;   /* host staging for compaction */
    cudaGraphExec_t gexec[QB_MAX+1]; unsigned char gmode[QB_MAX+1];   /* see cuda_talker_batch_t */
    /* GPU code-predictor head (final norm + lm_head + argmax) */
    qwen_tts_ctx_t *ctx; int head_state;      /* 0 untried, 1 resident, -1 unavailable */
    __nv_bfloat16 *d_lm[15]; float *d_cpnorm; int vocab;
    float *d_partv; int *d_parti; int *d_code; int *h_code;
} cuda_cp_batch_t;

static void cp_body_batch(cuda_cp_batch_t *s){
    /* Bmax is the ALLOCATED lane count and fixes the KV layer stride; B is how many
     * lanes we actually compute this step.  They differ when the caller compacts idle
     * lanes away, and confusing them silently reindexes every layer's cache. */
    const int Bmax=s->B;
    int B=(s->B_eff>0&&s->B_eff<=s->B)?s->B_eff:s->B;
    int H=s->hidden,qd=s->q_dim,kvd=s->kv_dim,hd=s->head_dim,half=hd/2;
    int nh=s->n_heads,nkv=s->n_kv,inter=s->inter; float scale=1.f/sqrtf((float)hd);
    for(int l=0;l<s->n_layers;++l){
        k_rmsnorm_full_b<<<B,TPB,TPB*sizeof(float)>>>(s->x,s->inorm[l],s->xn,H,s->eps);
        mvB(s->prec,s->wq[l],s->wqs[l],s->xn,s->q,qd,H,B);
        mvB(s->prec,s->wk[l],s->wks[l],s->xn,s->k,kvd,H,B);
        mvB(s->prec,s->wv[l],s->wvs[l],s->xn,s->v,kvd,H,B);
        k_rmsnorm_ph_b<<<B*nh,TPB,TPB*sizeof(float)>>>(s->q,s->qn[l],hd,nh,qd,s->eps);
        k_rmsnorm_ph_b<<<B*nkv,TPB,TPB*sizeof(float)>>>(s->k,s->kn[l],hd,nkv,kvd,s->eps);
        k_rope_neox_b<<<CEIL(B*nh*half,TPB),TPB>>>(s->q,s->rope_cos,s->rope_sin,nh,hd,s->d_pos,qd,B,s->d_act);
        k_rope_neox_b<<<CEIL(B*nkv*half,TPB),TPB>>>(s->k,s->rope_cos,s->rope_sin,nkv,hd,s->d_pos,kvd,B,s->d_act);
        k_trunc_bf16<<<CEIL(B*kvd,TPB),TPB>>>(s->k,B*kvd);
        k_trunc_bf16<<<CEIL(B*kvd,TPB),TPB>>>(s->v,B*kvd);
        float *Kl=s->kcache+(size_t)l*Bmax*s->kv_max*kvd, *Vl=s->vcache+(size_t)l*Bmax*s->kv_max*kvd;
        k_kv_store_b<<<CEIL(B*kvd,TPB),TPB>>>(Kl,Vl,s->k,s->v,kvd,s->d_pos,s->kv_max,B,s->d_act,s->d_slot);
        attn_b_launch(B*nh,hd,s->q,Kl,Vl,s->attn,nh,nkv,scale,s->d_pos,s->kv_max,qd,kvd,s->d_act,s->d_slot);
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
    s->ctx=ss->ctx;
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
    CK(cudaMalloc(&s->d_act,(size_t)B));
    CK(cudaMalloc(&s->d_slot,B*sizeof(int)));
    { int *ids=(int*)malloc(B*sizeof(int)); for(int i=0;i<B;++i) ids[i]=i;
      CK(cudaMemcpy(s->d_slot,ids,B*sizeof(int),cudaMemcpyHostToDevice)); free(ids); }
    s->B_eff=0; s->slot_identity=1;
    s->h_emb=(float*)malloc((size_t)B*s->hidden*sizeof(float));
    s->h_hid=(float*)malloc((size_t)B*s->hidden*sizeof(float));
    s->h_pos=(int*)malloc(B*sizeof(int)); s->h_slot=(int*)malloc(B*sizeof(int));
    return s;
}
/* x=[B][cp_h] host (each row = the caller's per-seq embed/residual seed), pos_arr=[B] host;
 * x updated in place with the B residual streams (caller norms + argmaxes each). */
extern "C" void qwen_cuda_cp_batch_step(void *st,float *x,const int *pos_arr,const unsigned char *active){
    cuda_cp_batch_t *s=(cuda_cp_batch_t*)st; int B=s->B,H=s->hidden;

    /* Compacting idle lanes matters more here than in the Talker.  The code predictor runs
     * fifteen passes per frame against the Talker's one, so a server sized for eight lanes and
     * serving four spent half of the dominant component on empty lanes: the C4 ladder measured
     * 12.31 ms/frame of step at --batch-size 8 against 8.86 at --batch-size 4, for the same
     * four requests.  The kernels already take a lane->slot map, so the KV stays addressed by
     * slot and a compacted lane still reads the history of the request it stands for.
     *
     * B_eff and h_slot are deliberately left set on return: qwen_cuda_cp_batch_head reads the
     * residual this leaves on the device, so it has to know the rows are dense and which slot
     * each one belongs to. */
    if(batch_compact_enabled() && active){
        int n=0;
        for(int b=0;b<B;++b) if(active[b]){
            s->h_slot[n]=b; s->h_pos[n]=pos_arr[b];
            memcpy(s->h_emb+(size_t)n*H, x+(size_t)b*H, (size_t)H*sizeof(float));
            ++n;
        }
        if(n==0){ s->B_eff=0; return; }
        if(n<B){
            CK(cudaMemcpy(s->x,s->h_emb,(size_t)n*H*sizeof(float),cudaMemcpyHostToDevice));
            CK(cudaMemcpy(s->d_pos,s->h_pos,(size_t)n*sizeof(int),cudaMemcpyHostToDevice));
            CK(cudaMemcpy(s->d_slot,s->h_slot,(size_t)n*sizeof(int),cudaMemcpyHostToDevice));
            { unsigned char ones[QB_MAX]; for(int i=0;i<n;++i) ones[i]=1;
              CK(cudaMemcpy(s->d_act,ones,(size_t)n,cudaMemcpyHostToDevice)); }
            s->B_eff=n; s->slot_identity=0;
            batch_graph_run<cuda_cp_batch_t,cp_body_batch>(s,n);
            CK(cudaStreamSynchronize(cudaStreamPerThread));
            CK(cudaMemcpy(s->h_hid,s->x,(size_t)n*H*sizeof(float),cudaMemcpyDeviceToHost));
            for(int i=0;i<n;++i)
                memcpy(x+(size_t)s->h_slot[i]*H, s->h_hid+(size_t)i*H, (size_t)H*sizeof(float));
            return;
        }
        /* n==B: every lane steps, so the dense form is the ordinary full-width path */
    }
    if(!s->slot_identity){
        int ids[QB_MAX]; for(int i=0;i<B;++i) ids[i]=i;
        CK(cudaMemcpy(s->d_slot,ids,(size_t)B*sizeof(int),cudaMemcpyHostToDevice));
        s->slot_identity=1;
    }
    s->B_eff=0;
    CK(cudaMemcpy(s->x,x,(size_t)B*H*sizeof(float),cudaMemcpyHostToDevice));
    CK(cudaMemcpy(s->d_pos,pos_arr,B*sizeof(int),cudaMemcpyHostToDevice));
    /* Same stale-position hazard as the talker, and worse here: cp_kv_max is 64, so any
     * leftover position >= 64 indexes outside the cache immediately. */
    { unsigned char all[QB_MAX]; if(!active){ for(int i=0;i<B;++i) all[i]=1; }
      CK(cudaMemcpy(s->d_act, active?active:all, (size_t)B, cudaMemcpyHostToDevice)); }
    batch_graph_run<cuda_cp_batch_t,cp_body_batch>(s,B);
    CK(cudaStreamSynchronize(cudaStreamPerThread));
    CK(cudaMemcpy(x,s->x,(size_t)B*H*sizeof(float),cudaMemcpyDeviceToHost));
}

/* ---- GPU code-predictor head: final norm + lm_head + argmax -----------------------------
 *
 * QWEN_CP_PROFILE measured this at 11.0 ms of every 24.4 ms frame on an A6000 (45% of the
 * code predictor, which is itself half of all serving work).  The cost is not arithmetic, it
 * is re-reading: one lm_head is [2048 x 1024] bf16 = 4 MB, the frame walks fifteen of them,
 * and the CPU fallback walks each one once PER LANE because qwen_argmax_matvec_bf16 takes a
 * single activation vector.  At four lanes that is 240 MB pulled from DRAM per frame.
 *
 * Batching over lanes is what removes it.  One block owns a slice of the vocabulary and holds
 * every lane's activation in shared memory, so a weight row is read once and dotted against
 * all B lanes while it is still in registers -- the whole frame then reads 60 MB once instead
 * of 240 MB, on a bus that is fifteen times wider.
 *
 * The accumulator is unrolled over QB_MAX rather than indexed by a runtime b.  This is not
 * style: a runtime-indexed local array cannot live in registers, spills to local memory, and
 * the same mistake in the batched matmats cost a measured 33x before it was found.
 *
 * Ties follow the CPU rule exactly -- strictly greater wins, so the lowest index survives a
 * tie -- but the dot products themselves are summed in a different order than the AVX2 path,
 * so a near-tie can still resolve the other way.  That is a different generation, not a wrong
 * one, and it is why this is measured against audio and not asserted to be bit-identical. */
#define CPH_NCHUNK 32
template<int NB>
__global__ void k_cp_head_part(const __nv_bfloat16 *W,const float *X,int ch,int vocab,int B,
                               const unsigned char *act,float *partv,int *parti){
    const int chunk=blockIdx.x, lane=threadIdx.x&31, warp=threadIdx.x>>5, nwarp=blockDim.x>>5;
    const int rows=CEIL(vocab,CPH_NCHUNK), r0=chunk*rows, r1=min(r0+rows,vocab);
    extern __shared__ float xs[];                       /* B x ch activations, loaded once */
    for(int i=threadIdx.x;i<B*ch;i+=blockDim.x) xs[i]=X[i];
    __syncthreads();
    float bestv[NB]; int besti[NB];
    #pragma unroll
    for(int b=0;b<NB;++b){ bestv[b]=-1e30f; besti[b]=0; }
    for(int row=r0+warp;row<r1;row+=nwarp){
        const __nv_bfloat16 *wr=W+(size_t)row*ch;
        float acc[NB];
        #pragma unroll
        for(int b=0;b<NB;++b) acc[b]=0.f;
        for(int i=lane;i<ch;i+=32){ float w=__bfloat162float(wr[i]);
            #pragma unroll
            for(int b=0;b<NB;++b) if(b<B) acc[b]+=w*xs[(size_t)b*ch+i]; }
        #pragma unroll
        for(int b=0;b<NB;++b){
            if(b>=B) break;
            float v=acc[b];
            for(int o=16;o>0;o>>=1) v+=__shfl_down_sync(0xffffffffu,v,o);
            if(lane==0 && v>bestv[b]){ bestv[b]=v; besti[b]=row; }
        }
    }
    __shared__ float sv[32*NB]; __shared__ int si[32*NB];
    if(lane==0){
        #pragma unroll
        for(int b=0;b<NB;++b){ if(b>=B) break; sv[b*32+warp]=bestv[b]; si[b*32+warp]=besti[b]; }
    }
    __syncthreads();
    if(threadIdx.x<B){
        int b=threadIdx.x;
        if(act && !act[b]){ partv[(size_t)b*CPH_NCHUNK+chunk]=-1e30f; parti[(size_t)b*CPH_NCHUNK+chunk]=0; return; }
        float bv=sv[b*32]; int bi=si[b*32];
        for(int w=1;w<nwarp;++w){ float v=sv[b*32+w]; int i=si[b*32+w];
            if(v>bv||(v==bv&&i<bi)){ bv=v; bi=i; } }
        partv[(size_t)b*CPH_NCHUNK+chunk]=bv; parti[(size_t)b*CPH_NCHUNK+chunk]=bi;
    }
}
__global__ void k_cp_head_final(const float *partv,const int *parti,int B,
                                const unsigned char *act,int *out){
    int b=blockIdx.x; if(b>=B) return;
    if(act && !act[b]){ if(threadIdx.x==0) out[b]=-1; return; }
    if(threadIdx.x!=0) return;
    float bv=partv[(size_t)b*CPH_NCHUNK]; int bi=parti[(size_t)b*CPH_NCHUNK];
    for(int c=1;c<CPH_NCHUNK;++c){ float v=partv[(size_t)b*CPH_NCHUNK+c]; int i=parti[(size_t)b*CPH_NCHUNK+c];
        if(v>bv||(v==bv&&i<bi)){ bv=v; bi=i; } }
    out[b]=bi;
}
static int cp_head_enabled(void){
    static int t=-1;
    if(t<0){ const char *e=getenv("QWEN_CUDA_CP_HEAD"); t=(!e||!e[0])?1:(e[0]!='0'); }
    return t;
}
/* Upload cp_norm and the fifteen bf16 lm_heads once (about 60 MB). Returns 0 if the model
 * does not carry bf16 heads, in which case the caller keeps the CPU path. */
static int cp_head_ready(cuda_cp_batch_t *s){
    if(s->head_state) return s->head_state>0;
    qwen_tts_ctx_t *ctx=s->ctx;
    if(!ctx||!ctx->cp_norm){ s->head_state=-1; return 0; }
    for(int g=0;g<15;++g) if(!ctx->cp_lm_head_bf16[g]){ s->head_state=-1; return 0; }
    int ch=s->hidden, vocab=ctx->config.codebook_size;
    if(vocab<=0||ch<=0){ s->head_state=-1; return 0; }
    s->vocab=vocab;
    if(cudaMalloc(&s->d_cpnorm,(size_t)ch*sizeof(float))!=cudaSuccess){ s->head_state=-1; return 0; }
    CK(cudaMemcpy(s->d_cpnorm,ctx->cp_norm,(size_t)ch*sizeof(float),cudaMemcpyHostToDevice));
    for(int g=0;g<15;++g){
        if(cudaMalloc(&s->d_lm[g],(size_t)vocab*ch*sizeof(uint16_t))!=cudaSuccess){ s->head_state=-1; return 0; }
        CK(cudaMemcpy(s->d_lm[g],ctx->cp_lm_head_bf16[g],(size_t)vocab*ch*sizeof(uint16_t),cudaMemcpyHostToDevice));
    }
    CK(cudaMalloc(&s->d_partv,(size_t)s->B*CPH_NCHUNK*sizeof(float)));
    CK(cudaMalloc(&s->d_parti,(size_t)s->B*CPH_NCHUNK*sizeof(int)));
    CK(cudaMalloc(&s->d_code,(size_t)s->B*sizeof(int)));
    s->h_code=(int*)malloc((size_t)s->B*sizeof(int));
    fprintf(stderr,"CUDA CP head: 15 lm_heads resident (%d x %d bf16, %.0f MB) — QWEN_CUDA_CP_HEAD=0 disables\n",
            vocab,ch,15.0*vocab*ch*2/1e6);
    s->head_state=1;
    return 1;
}
/* Norm + head + argmax for codebook g, for every active lane, entirely on the device.
 * Reads the residual the step left in s->x, so it must be called straight after the step.
 * Returns 0 when it declines, and then the caller's CPU path must run. */
extern "C" int qwen_cuda_cp_batch_head(void *st,int g,const unsigned char *active,
                                       int *out_codes,int stride){
    cuda_cp_batch_t *s=(cuda_cp_batch_t*)st;
    if(!s||g<0||g>=15||!cp_head_enabled()||!cp_head_ready(s)) return 0;
    int ch=s->hidden;
    /* The step may have left the residual compacted into dense rows 0..B_eff-1. Follow it:
     * the head then also skips the idle lanes, and the codes are scattered back by slot. */
    const int dense=(s->B_eff>0&&s->B_eff<=s->B);
    const int n=dense?s->B_eff:s->B;
    /* One block holds every lane's activation in shared memory. Past twelve lanes at ch=1024
     * that exceeds the 48 KB a block gets without an opt-in, and the launch would fail silently
     * and leave the codes unwritten; hand those widths back to the CPU head instead. */
    if((size_t)n*ch*sizeof(float) > 48000u) return 0;
    if(!dense){
        unsigned char all[QB_MAX];
        if(!active){ for(int i=0;i<n;++i) all[i]=1; }
        CK(cudaMemcpy(s->d_act,active?active:all,(size_t)n,cudaMemcpyHostToDevice));
    }   /* the compacted path already uploaded an all-ones mask of length n */
    k_rmsnorm_full_b<<<n,TPB,TPB*sizeof(float)>>>(s->x,s->d_cpnorm,s->xn,ch,s->eps);
    size_t shm=(size_t)n*ch*sizeof(float);
#define CPH_N(NB) k_cp_head_part<NB><<<CPH_NCHUNK,TPB,shm>>>(s->d_lm[g],s->xn,ch,s->vocab,n,s->d_act,s->d_partv,s->d_parti)
    switch(n){
      case  1: CPH_N( 1); break;  case  2: CPH_N( 2); break;  case  3: CPH_N( 3); break;
      case  4: CPH_N( 4); break;  case  5: CPH_N( 5); break;  case  6: CPH_N( 6); break;
      case  7: CPH_N( 7); break;  case  8: CPH_N( 8); break;  case  9: CPH_N( 9); break;
      case 10: CPH_N(10); break;  case 11: CPH_N(11); break;  case 12: CPH_N(12); break;
      case 13: CPH_N(13); break;  case 14: CPH_N(14); break;  case 15: CPH_N(15); break;
      default: CPH_N(QB_MAX); break;
    }
#undef CPH_N
    k_cp_head_final<<<n,32>>>(s->d_partv,s->d_parti,n,s->d_act,s->d_code);
    CK(cudaStreamSynchronize(cudaStreamPerThread));
    CK(cudaMemcpy(s->h_code,s->d_code,(size_t)n*sizeof(int),cudaMemcpyDeviceToHost));
    if(dense) for(int i=0;i<n;++i) out_codes[(size_t)s->h_slot[i]*stride+g]=s->h_code[i];
    else      for(int b=0;b<n;++b) if(!active||active[b]) out_codes[(size_t)b*stride+g]=s->h_code[b];
    return 1;
}

/* Upper bound on what fusing the code-predictor loop could buy.
 *
 * The served loop alternates: upload x, replay the body, synchronise, download x, and then
 * the host does the embedding gather and the MTP projection before the next pass. Fifteen
 * times a frame. The host work between passes measures 40 microseconds for the WHOLE frame,
 * so if the alternation is expensive it is the synchronising and the copies, not the work
 * they are there to serialise.
 *
 * This times the same fifteen body replays back to back with a single synchronise at the
 * end and no copies. The result is nonsense as audio -- each pass reads whatever the last
 * one left -- but it is the exact ceiling of a fused loop, and the gap against the real CP
 * throughput is the budget available for fusing. Measuring it costs ten seconds; building
 * the fused loop to find out costs a day. */
extern "C" double qwen_cuda_cp_batch_bench_fused(void *st,int passes,int frames){
    cuda_cp_batch_t *s=(cuda_cp_batch_t*)st; if(!s||passes<1||frames<1) return -1.0;
    int B=s->B;
    { unsigned char all[QB_MAX]; for(int i=0;i<B;++i) all[i]=1;
      CK(cudaMemcpy(s->d_act,all,(size_t)B,cudaMemcpyHostToDevice)); }
    int *ph=(int*)malloc((size_t)B*sizeof(int));
    for(int p=0;p<passes;++p){ for(int b=0;b<B;++b) ph[b]=p;
        CK(cudaMemcpy(s->d_pos,ph,(size_t)B*sizeof(int),cudaMemcpyHostToDevice));
        batch_graph_run<cuda_cp_batch_t,cp_body_batch>(s,B); }      /* warm + capture */
    CK(cudaStreamSynchronize(cudaStreamPerThread));
    double t0=now_ms();
    for(int f=0;f<frames;++f){
        for(int p=0;p<passes;++p){
            CK(cudaMemcpyAsync(s->d_pos,ph,(size_t)B*sizeof(int),cudaMemcpyHostToDevice,cudaStreamPerThread));
            batch_graph_run<cuda_cp_batch_t,cp_body_batch>(s,B);
        }
        CK(cudaStreamSynchronize(cudaStreamPerThread));             /* one sync per frame */
    }
    double ms=(now_ms()-t0)/frames;
    free(ph);
    return ms;
}

extern "C" void qwen_cuda_cp_batch_free(void *st){
    cuda_cp_batch_t *s=(cuda_cp_batch_t*)st; if(!s) return;
    cudaFree(s->kcache);cudaFree(s->vcache);cudaFree(s->x);cudaFree(s->xn);cudaFree(s->q);
    cudaFree(s->k);cudaFree(s->v);cudaFree(s->attn);cudaFree(s->proj);cudaFree(s->gate);
    cudaFree(s->gu);cudaFree(s->d_pos);cudaFree(s->d_act);cudaFree(s->d_slot);
    for(int i=0;i<=QB_MAX;++i) if(s->gmode[i]==1) cudaGraphExecDestroy(s->gexec[i]);
    if(s->head_state>0){ for(int g=0;g<15;++g) cudaFree(s->d_lm[g]);
        cudaFree(s->d_cpnorm);cudaFree(s->d_partv);cudaFree(s->d_parti);cudaFree(s->d_code);free(s->h_code); }
    free(s->h_emb);free(s->h_hid);free(s->h_pos);free(s->h_slot); free(s);
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
