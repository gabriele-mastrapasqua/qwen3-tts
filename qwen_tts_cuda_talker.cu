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
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

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

/* q4_0 block: 32 weights = fp32 scale + 16 bytes (low nibble=even idx, high=odd), val=(nib-8)*scale.
 * MUST match qwen_tts_kernels.h q4_0_block_t exactly. */
typedef struct { float scale; unsigned char qs[16]; } q4blk;
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
        s += (float)(nib-8) * b->scale * x[c];
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
static void *up_q4(const void *w, size_t nblocks) {   /* q4_0 blocks (20 bytes each), raw */
    void *d=NULL; size_t bytes=nblocks*sizeof(q4blk);
    CK(cudaMalloc(&d,bytes)); CK(cudaMemcpy(d,w,bytes,cudaMemcpyHostToDevice)); return d;
}
/* dispatch by precision (0=bf16, 1=int8, 2=q4_0). Warp-per-row: one warp per output row. */
static inline void mv(int prec, const void *W, const float *scale, const float *dX, float *dY, int rows, int cols) {
    int grid = CEIL(rows*32, TPB);
    if (prec==2)      k_matvec_q4_0<<<grid,TPB>>>((const q4blk*)W, dX, dY, rows, cols);
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
