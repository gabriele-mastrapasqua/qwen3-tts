/*
 * qwen_tts_cuda_decoder.cu — GPU-RESIDENT ConvNet speech decoder (M3, the real one).
 *
 * The per-op sgemm redirect (qwen_tts_cuda.c sd_sgemm) was TRANSFER-BOUND (uploads the im2col
 * col buffer per call). This keeps the ACTIVATION resident: upload the short latent `signal`
 * ([ch×len]) once, run the WHOLE conv stack (ConvNeXt ×2 + initial conv + 4 upsample blocks +
 * final) as device kernels on resident buffers, download the audio once. Weights cached resident
 * by pointer. Mirrors conv_decoder_forward() in qwen_tts_speech_decoder.c EXACTLY.
 */
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

extern "C" {
#include "qwen_tts.h"
int g_cuda_decoder_conv_on = 0;
}

#define CK(x) do { cudaError_t e_=(x); if(e_!=cudaSuccess){fprintf(stderr,"CUDA-dec %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e_));} } while(0)
#define TPB 256
#define CEIL(a,b) (((a)+(b)-1)/(b))

/* ---- kernels (channel-first [ch,len], causal) --------------------------- */
/* out[oc,t] = bias[oc] + Σ_ic Σ_k w[oc,ic,k] · in[ic, t-(ksz-1)*dil + k*dil]  (causal left pad) */
__global__ void kd_conv1d(const float *in,const float *w,const float *bias,float *out,
                          int in_ch,int out_ch,int length,int ksz,int dil){
    int t=blockIdx.x*blockDim.x+threadIdx.x, oc=blockIdx.y; if(t>=length||oc>=out_ch) return;
    int pad=(ksz-1)*dil; float s=bias?bias[oc]:0.f;
    for(int ic=0;ic<in_ch;++ic) for(int k=0;k<ksz;++k){ int ip=t-pad+k*dil;
        if(ip>=0&&ip<length) s+=w[((size_t)oc*in_ch+ic)*ksz+k]*in[(size_t)ic*length+ip]; }
    out[(size_t)oc*length+t]=s;
}
/* depthwise k=7 pad=6 (per-channel): out[c,t]=bias[c]+Σ_k w[c,k]·in[c,t-6+k] */
__global__ void kd_depthwise7(const float *in,const float *w,const float *bias,float *out,int ch,int length){
    int t=blockIdx.x*blockDim.x+threadIdx.x, c=blockIdx.y; if(t>=length||c>=ch) return;
    float s=bias?bias[c]:0.f;
    for(int k=0;k<7;++k){ int ip=t-6+k; if(ip>=0&&ip<length) s+=w[(size_t)c*7+k]*in[(size_t)c*length+ip]; }
    out[(size_t)c*length+t]=s;
}
/* causal conv_transpose1d (matches causal_conv_transpose1d_naive). w[in_ch,out_ch,ksz]. */
__global__ void kd_convT(const float *in,const float *w,const float *bias,float *out,
                         int in_ch,int out_ch,int in_len,int out_len,int ksz,int stride){
    int p=blockIdx.x*blockDim.x+threadIdx.x, oc=blockIdx.y; if(p>=out_len||oc>=out_ch) return;
    int full=(in_len-1)*stride+ksz, trim=ksz-stride; float s=bias?bias[oc]:0.f;
    if(p<full-trim) for(int k=0;k<ksz;++k){ int sh=p-k; if(sh<0||sh%stride) continue; int tt=sh/stride;
        if(tt<0||tt>=in_len) continue; for(int ic=0;ic<in_ch;++ic) s+=in[(size_t)ic*in_len+tt]*w[((size_t)ic*out_ch+oc)*ksz+k]; }
    out[(size_t)oc*out_len+p]=s;
}
/* per-timestep LayerNorm over channels: for each t, normalize over c, then *w[c]+b[c]. one block per t. */
__global__ void kd_layernorm_ct(float *x,const float *w,const float *b,int ch,int length,float eps){
    int t=blockIdx.x; if(t>=length) return;
    extern __shared__ float sh[]; int tid=threadIdx.x,tc=blockDim.x;
    float s=0.f; for(int c=tid;c<ch;c+=tc) s+=x[(size_t)c*length+t];
    sh[tid]=s; __syncthreads(); for(int st=tc/2;st>0;st>>=1){ if(tid<st) sh[tid]+=sh[tid+st]; __syncthreads(); }
    float mean=sh[0]/ch; __syncthreads();
    float v=0.f; for(int c=tid;c<ch;c+=tc){ float d=x[(size_t)c*length+t]-mean; v+=d*d; }
    sh[tid]=v; __syncthreads(); for(int st=tc/2;st>0;st>>=1){ if(tid<st) sh[tid]+=sh[tid+st]; __syncthreads(); }
    float inv=rsqrtf(sh[0]/ch+eps);
    for(int c=tid;c<ch;c+=tc) x[(size_t)c*length+t]=(x[(size_t)c*length+t]-mean)*inv*w[c]+b[c];
}
__global__ void kd_gelu(float *x,size_t n){ size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x; if(i<n){ float v=x[i]; x[i]=0.5f*v*(1.f+erff(v*0.7071067811865476f)); } }
/* snake: x += (1/b)·sin²(a·x), a=exp(la), b=exp(lb), per channel (log-space params). */
__global__ void kd_snake(float *x,const float *la,const float *lb,int ch,int length){
    int t=blockIdx.x*blockDim.x+threadIdx.x, c=blockIdx.y; if(t>=length||c>=ch) return;
    float a=expf(la[c]), invb=1.f/expf(lb[c]); size_t idx=(size_t)c*length+t; float v=x[idx]; float s=sinf(a*v);
    x[idx]=v+invb*s*s;
}
/* signal[c,t] = residual[c,t] + signal[c,t]*gamma[c] */
__global__ void kd_gamma_res(float *sig,const float *res,const float *gamma,int ch,int length){
    int t=blockIdx.x*blockDim.x+threadIdx.x, c=blockIdx.y; if(t>=length||c>=ch) return;
    size_t idx=(size_t)c*length+t; sig[idx]=res[idx]+sig[idx]*gamma[c];
}
__global__ void kd_add(float *a,const float *b,size_t n){ size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x; if(i<n) a[i]+=b[i]; }
/* out[oc,t] += bias[oc] (broadcast over t) */
__global__ void kd_add_bias(float *out,const float *bias,int oc_n,int length){
    int t=blockIdx.x*blockDim.x+threadIdx.x, oc=blockIdx.y; if(t>=length||oc>=oc_n||!bias) return;
    out[(size_t)oc*length+t]+=bias[oc];
}

/* pointwise conv (ksz=1) = matmul out[M,N] = W[M,K] @ in[K,N], ALL resident on device → cuBLAS
 * (the compute-bound 40× regime, no transfer). Row-major identity: cublas(N,N,N,M,K,in,W,out). */
static cublasHandle_t g_dh=NULL;
static void dmatmul(float *out,const float *W,const float *in,int M,int K,int N){
    if(!g_dh){ if(cublasCreate(&g_dh)!=CUBLAS_STATUS_SUCCESS){g_dh=NULL;return;} }
    cublasSetStream(g_dh,cudaStreamPerThread);   /* same per-thread stream as the kernels → ordered */
    const float a=1.f,b=0.f;
    cublasSgemm(g_dh,CUBLAS_OP_N,CUBLAS_OP_N,N,M,K,&a,in,N,W,K,&b,out,N);
}

/* im2col for a causal DILATED conv1d: col[(ic*ksz+k), t] = in[ic, t-(ksz-1)*dil + k*dil] (0 if OOB).
 * Then out[oc,t] = W[oc, ic*ksz+k] @ col[ic*ksz+k, t] via cuBLAS — same math as kd_conv1d but the
 * ksz=7 convs (initial + resblock conv1) become the compute-bound 40× gemm regime, all resident. */
__global__ void kd_im2col(const float *in,float *col,int in_ch,int length,int ksz,int dil){
    int t=blockIdx.x*blockDim.x+threadIdx.x, row=blockIdx.y;   /* row = ic*ksz + k */
    if(t>=length||row>=in_ch*ksz) return;
    int ic=row/ksz, k=row%ksz, pad=(ksz-1)*dil, ip=t-pad+k*dil;
    col[(size_t)row*length+t] = (ip>=0&&ip<length) ? in[(size_t)ic*length+ip] : 0.f;
}
/* --- conv_transpose1d via gemm+gather (kd_convT was 74% of decoder GPU time, naive) --------
 * Repack weight w[in_ch,out_ch,ksz] -> WM[out_ch*ksz, in_ch]; P2[out_ch*ksz, in_len]=WM@in (cuBLAS);
 * then gather out[oc,p]=bias[oc]+Σ_{k: (p-k)%stride==0, tt=(p-k)/stride<in_len} P2[oc*ksz+k, tt].
 * Same terms as kd_convT (out_len==in_len*stride so the causal trim is implicit), gemm-reassociated. */
__global__ void kd_wpack_convT(const float *w,float *wm,int in_ch,int out_ch,int ksz){
    int ic=blockIdx.x*blockDim.x+threadIdx.x, row=blockIdx.y;   /* row = oc*ksz + k */
    if(ic>=in_ch||row>=out_ch*ksz) return;
    int oc=row/ksz, k=row%ksz;
    wm[(size_t)row*in_ch+ic] = w[((size_t)ic*out_ch+oc)*ksz+k];
}
__global__ void kd_convT_gather(const float *P2,const float *bias,float *out,
                                int out_ch,int out_len,int in_len,int ksz,int stride){
    int p=blockIdx.x*blockDim.x+threadIdx.x, oc=blockIdx.y; if(p>=out_len||oc>=out_ch) return;
    float s=bias?bias[oc]:0.f;
    for(int k=0;k<ksz;++k){ int sh=p-k; if(sh<0||sh%stride) continue; int tt=sh/stride; if(tt>=in_len) continue;
        s+=P2[((size_t)oc*ksz+k)*in_len+tt]; }
    out[(size_t)oc*out_len+p]=s;
}

/* ---- resident weight cache (by host pointer) ---------------------------- */
typedef struct { const void *key; float *dbuf; } dwc;
static dwc *g_dwc=NULL; static int g_dwc_n=0,g_dwc_cap=0;
static float *dev_w(const float *W,size_t n){
    if(!W) return NULL;
    for(int i=0;i<g_dwc_n;++i) if(g_dwc[i].key==W) return g_dwc[i].dbuf;
    float *d=NULL; if(cudaMalloc(&d,n*sizeof(float))!=cudaSuccess) return NULL;
    cudaMemcpy(d,W,n*sizeof(float),cudaMemcpyHostToDevice);
    if(g_dwc_n==g_dwc_cap){ g_dwc_cap=g_dwc_cap?g_dwc_cap*2:128; g_dwc=(dwc*)realloc(g_dwc,g_dwc_cap*sizeof(dwc)); }
    g_dwc[g_dwc_n].key=W; g_dwc[g_dwc_n].dbuf=d; g_dwc_n++; return d;
}
/* repacked conv_transpose weights WM[out_ch*ksz, in_ch] (built once per host weight, cached) */
static dwc *g_wmc=NULL; static int g_wmc_n=0,g_wmc_cap=0;
static float *dev_wT_convT(const float *w_host,int in_ch,int out_ch,int ksz){
    if(!w_host) return NULL;
    for(int i=0;i<g_wmc_n;++i) if(g_wmc[i].key==w_host) return g_wmc[i].dbuf;
    const float *dw=dev_w(w_host,(size_t)in_ch*out_ch*ksz); if(!dw) return NULL;
    float *wm=NULL; if(cudaMalloc(&wm,(size_t)out_ch*ksz*in_ch*sizeof(float))!=cudaSuccess) return NULL;
    kd_wpack_convT<<<dim3(CEIL(in_ch,TPB),out_ch*ksz),TPB>>>(dw,wm,in_ch,out_ch,ksz);
    if(g_wmc_n==g_wmc_cap){ g_wmc_cap=g_wmc_cap?g_wmc_cap*2:32; g_wmc=(dwc*)realloc(g_wmc,g_wmc_cap*sizeof(dwc)); }
    g_wmc[g_wmc_n].key=w_host; g_wmc[g_wmc_n].dbuf=wm; g_wmc_n++; return wm;
}
/* device scratch buffers (grown, reused) */
static float *g_a=NULL,*g_b=NULL,*g_c=NULL; static size_t g_ac=0,g_bc=0,g_cc=0;
static float *grow(float **buf,size_t *cap,size_t need){ if(*cap<need){ if(*buf)cudaFree(*buf); if(cudaMalloc((void**)buf,need*sizeof(float))!=cudaSuccess){*buf=NULL;*cap=0;return NULL;} *cap=need; } return *buf; }

/* conv_transpose out length = (in_len-1)*stride + ksz ... trimmed to (in_len-1)*stride? The CPU uses
 * conv_transpose1d_out_len = (in_len-1)*stride + (ksz - stride) ... we replicate via the caller-passed len. */

/* ============================================================================
 * TODO (next session): qwen_cuda_conv_decoder_run — orchestration.
 * All kernels above are DONE + reusable. Mirror conv_decoder_forward() exactly:
 *   upload signal→dsig (grow g_a). Then, ping-pong device buffers (cur/nxt/res + pw):
 *   ConvNeXt ×2 (b=0,1):
 *     new_len=cur_len*2; kd_convT(cur→up, in=out=cur_ch, ksz=2, stride=2)          [conv_transpose1d_out_len = in*stride]
 *     res = copy(up); kd_depthwise7(up→dw, cur_ch)                                 [dwconv_weight cur_ch*7]
 *     kd_layernorm_ct(dw, norm_w, norm_b, cur_ch, len, 1e-5) <<<len, TPB, TPB*4>>>
 *     kd_conv1d(dw→pw, pwconv1_w, pwconv1_b, in=cur_ch, out=4096, ksz=1) ; kd_gelu(pw, 4096*len)
 *     kd_conv1d(pw→cur, pwconv2_w, pwconv2_b, in=4096, out=cur_ch, ksz=1)
 *     kd_gamma_res(cur, res, gamma, cur_ch, len)
 *   initial conv: kd_conv1d(cur→nxt, initial_conv_w/b, in=cur_ch, out=1536, ksz=7, dil=1); cur_ch=1536
 *   4 upsample blocks (rates 8,5,4,3; out_ch 768,384,192,96; kernel=rate*2):
 *     kd_snake(cur, up.snake_a, up.snake_b, cur_ch, len)
 *     up_len=len*rate; kd_convT(cur→nxt, in=cur_ch,out=out_ch,ksz=rate*2,stride=rate); cur_ch=out_ch; len=up_len
 *     3 resblocks (dil 1,3,9): res=copy; kd_snake(snake1); kd_conv1d(k=7,dil); kd_snake(snake2);
 *                             kd_conv1d(k=1); kd_add(cur, res)   [wait: signal=res+c2, so kd_add(c2_out,res)->cur]
 *   final: kd_snake(final_snake); kd_conv1d(cur→audio, final_conv_w/b, in=cur_ch,out=1,ksz=7,dil=1)
 *   clamp audio to [-1,1] (small kernel or on host); cudaMemcpy audio→host; return audio_len=len.
 * Weight sizes for dev_w(): convT in*out*ksz; conv1d out*in*ksz; depthwise ch*7; norm/gamma/bias = ch.
 * Validate: mel-corr vs CPU conv_decoder_forward MUST be 1.0 (bit-identical, same fp ops). Then wire into
 * conv_decoder_forward (if g_cuda_decoder_conv_on: upload signal, call this, return) + add to Makefile cuda.
 * ============================================================================ */
/* ping-pong device buffers */
typedef struct { float *p; size_t cap; } DB;
static float *dbg(DB *d,size_t need){ if(d->cap<need){ if(d->p)cudaFree(d->p); if(cudaMalloc(&d->p,need*sizeof(float))!=cudaSuccess){d->p=NULL;d->cap=0;return NULL;} d->cap=need; } return d->p; }
static DB S={0,0},O={0,0},R={0,0},P={0,0},Col={0,0},T={0,0};   /* signal / op-out / residual / pointwise / im2col / convT-P2 */
#define SWAP() do{ DB _t=S; S=O; O=_t; cur=S.p; }while(0)
#define GRID2(len,ch) dim3(CEIL((len),TPB),(ch))

/* causal dilated conv1d (ksz=7) via im2col + cuBLAS, all resident (no transfer). Bit-equivalent to
 * kd_conv1d up to gemm fp-accumulation order (mel-corr 1.0, like the pointwise convs). */
static int g_naive7=-1;   /* A/B toggle: QWEN_DEC_NAIVE7=1 → old naive kd_conv1d for the ksz=7 convs */
static void conv1d_gemm(const float *cur,const float *W,const float *bias,float *out,
                        int in_ch,int out_ch,int length,int ksz,int dil){
    if(g_naive7<0) g_naive7 = getenv("QWEN_DEC_NAIVE7") ? 1 : 0;
    if(g_naive7){ kd_conv1d<<<GRID2(length,out_ch),TPB>>>(cur,W,bias,out,in_ch,out_ch,length,ksz,dil); return; }
    int K=in_ch*ksz; float *col=dbg(&Col,(size_t)K*length); if(!col) return;
    kd_im2col<<<GRID2(length,K),TPB>>>(cur,col,in_ch,length,ksz,dil);
    dmatmul(out,W,col,out_ch,K,length);                                   /* out[out_ch,len]=W[out_ch,K]@col[K,len] */
    if(bias) kd_add_bias<<<GRID2(length,out_ch),TPB>>>(out,bias,out_ch,length);
}

/* causal conv_transpose1d via gemm+gather, all resident. Bit-equivalent to kd_convT up to gemm
 * fp-accumulation order. QWEN_DEC_NAIVET=1 keeps the old kernel for A/B. */
static int g_naivet=-1;
static void convT_gemm(const float *in,const float *w_host,const float *bias,float *out,
                       int in_ch,int out_ch,int in_len,int out_len,int ksz,int stride){
    if(g_naivet<0) g_naivet = getenv("QWEN_DEC_NAIVET") ? 1 : 0;
    float *WM = g_naivet ? NULL : dev_wT_convT(w_host,in_ch,out_ch,ksz);
    if(!WM){ kd_convT<<<GRID2(out_len,out_ch),TPB>>>(in,dev_w(w_host,(size_t)in_ch*out_ch*ksz),bias,out,in_ch,out_ch,in_len,out_len,ksz,stride); return; }
    float *P2=dbg(&T,(size_t)out_ch*ksz*in_len); if(!P2){ kd_convT<<<GRID2(out_len,out_ch),TPB>>>(in,dev_w(w_host,(size_t)in_ch*out_ch*ksz),bias,out,in_ch,out_ch,in_len,out_len,ksz,stride); return; }
    dmatmul(P2,WM,in,out_ch*ksz,in_ch,in_len);            /* P2[out_ch*ksz,in_len]=WM[out_ch*ksz,in_ch]@in[in_ch,in_len] */
    kd_convT_gather<<<GRID2(out_len,out_ch),TPB>>>(P2,bias,out,out_ch,out_len,in_len,ksz,stride);
}

extern "C" int qwen_cuda_conv_decoder_run(void *ctxv, float *signal_host, int cur_ch, int cur_len,
                                          float **audio_out, int *n_out) {
    qwen_tts_ctx_t *ctx=(qwen_tts_ctx_t*)ctxv;
    qwen_speech_decoder_t *sd=&ctx->speech_dec;
    float *cur=dbg(&S,(size_t)cur_ch*cur_len); if(!cur) return -1;
    cudaMemcpy(cur,signal_host,(size_t)cur_ch*cur_len*sizeof(float),cudaMemcpyHostToDevice);

    /* ConvNeXt upsample ×2 (2× each) */
    for(int b=0;b<2;++b){
        qwen_sd_convnext_t *cn=&sd->convnext[b];
        int nl=cur_len*2;
        float *up=dbg(&O,(size_t)cur_ch*nl);
        convT_gemm(cur,cn->conv_weight,dev_w(cn->conv_bias,cur_ch),up,cur_ch,cur_ch,cur_len,nl,2,2);
        SWAP(); cur_len=nl;
        float *res=dbg(&R,(size_t)cur_ch*cur_len); cudaMemcpy(res,cur,(size_t)cur_ch*cur_len*sizeof(float),cudaMemcpyDeviceToDevice);
        float *dw=dbg(&O,(size_t)cur_ch*cur_len);
        kd_depthwise7<<<GRID2(cur_len,cur_ch),TPB>>>(cur,dev_w(cn->dwconv_weight,(size_t)cur_ch*7),dev_w(cn->dwconv_bias,cur_ch),dw,cur_ch,cur_len);
        SWAP();
        kd_layernorm_ct<<<cur_len,TPB,TPB*sizeof(float)>>>(cur,dev_w(cn->norm_weight,cur_ch),dev_w(cn->norm_bias,cur_ch),cur_ch,cur_len,1e-5f);
        float *pw=dbg(&P,(size_t)4096*cur_len);
        dmatmul(pw,dev_w(cn->pwconv1_weight,(size_t)4096*cur_ch),cur,4096,cur_ch,cur_len);   /* pw1 = matmul (cuBLAS, resident) */
        kd_add_bias<<<GRID2(cur_len,4096),TPB>>>(pw,dev_w(cn->pwconv1_bias,4096),4096,cur_len);
        kd_gelu<<<CEIL((size_t)4096*cur_len,TPB),TPB>>>(pw,(size_t)4096*cur_len);
        float *o2=dbg(&O,(size_t)cur_ch*cur_len);
        dmatmul(o2,dev_w(cn->pwconv2_weight,(size_t)cur_ch*4096),pw,cur_ch,4096,cur_len);     /* pw2 = matmul */
        kd_add_bias<<<GRID2(cur_len,cur_ch),TPB>>>(o2,dev_w(cn->pwconv2_bias,cur_ch),cur_ch,cur_len);
        SWAP();
        kd_gamma_res<<<GRID2(cur_len,cur_ch),TPB>>>(cur,res,dev_w(cn->gamma,cur_ch),cur_ch,cur_len);
    }

    /* Initial conv (cur_ch→1536, k=7) — im2col + cuBLAS (resident) */
    { int oc=1536; float *o=dbg(&O,(size_t)oc*cur_len);
      conv1d_gemm(cur,dev_w(sd->initial_conv_weight,(size_t)oc*cur_ch*7),dev_w(sd->initial_conv_bias,oc),o,cur_ch,oc,cur_len,7,1);
      SWAP(); cur_ch=oc; }

    /* 4 decoder upsample blocks */
    int up_rates[4]={8,5,4,3}, out_channels[4]={768,384,192,96}, dils[3]={1,3,9};
    for(int b=0;b<4;++b){
        qwen_sd_upsample_block_t *ub=&sd->upsample_blocks[b];
        int rate=up_rates[b], ksz=rate*2, oc=out_channels[b];
        if(ub->upsample.snake_alpha)
            kd_snake<<<GRID2(cur_len,cur_ch),TPB>>>(cur,ub->upsample.snake_alpha?dev_w(ub->upsample.snake_alpha,cur_ch):NULL,dev_w(ub->upsample.snake_beta,cur_ch),cur_ch,cur_len);
        int ul=cur_len*rate; float *o=dbg(&O,(size_t)oc*ul);
        convT_gemm(cur,ub->upsample.conv_weight,dev_w(ub->upsample.conv_bias,oc),o,cur_ch,oc,cur_len,ul,ksz,rate);
        SWAP(); cur_ch=oc; cur_len=ul;
        for(int r=0;r<3;++r){
            int dil=dils[r];
            float *res=dbg(&R,(size_t)cur_ch*cur_len); cudaMemcpy(res,cur,(size_t)cur_ch*cur_len*sizeof(float),cudaMemcpyDeviceToDevice);
            if(ub->res_blocks[r].snake1_alpha)
                kd_snake<<<GRID2(cur_len,cur_ch),TPB>>>(cur,dev_w(ub->res_blocks[r].snake1_alpha,cur_ch),dev_w(ub->res_blocks[r].snake1_beta,cur_ch),cur_ch,cur_len);
            float *c1=dbg(&O,(size_t)cur_ch*cur_len);
            conv1d_gemm(cur,dev_w(ub->res_blocks[r].conv1_weight,(size_t)cur_ch*cur_ch*7),dev_w(ub->res_blocks[r].conv1_bias,cur_ch),c1,cur_ch,cur_ch,cur_len,7,dil);
            SWAP();
            if(ub->res_blocks[r].snake2_alpha)
                kd_snake<<<GRID2(cur_len,cur_ch),TPB>>>(cur,dev_w(ub->res_blocks[r].snake2_alpha,cur_ch),dev_w(ub->res_blocks[r].snake2_beta,cur_ch),cur_ch,cur_len);
            float *c2=dbg(&O,(size_t)cur_ch*cur_len);
            dmatmul(c2,dev_w(ub->res_blocks[r].conv2_weight,(size_t)cur_ch*cur_ch),cur,cur_ch,cur_ch,cur_len);   /* conv2 k=1 = matmul */
            kd_add_bias<<<GRID2(cur_len,cur_ch),TPB>>>(c2,dev_w(ub->res_blocks[r].conv2_bias,cur_ch),cur_ch,cur_len);
            SWAP();
            kd_add<<<CEIL((size_t)cur_ch*cur_len,TPB),TPB>>>(cur,res,(size_t)cur_ch*cur_len);   /* cur = c2 + res */
        }
    }

    /* Final snake + conv (cur_ch→1, k=7) */
    kd_snake<<<GRID2(cur_len,cur_ch),TPB>>>(cur,dev_w(sd->final_snake.alpha,cur_ch),dev_w(sd->final_snake.beta,cur_ch),cur_ch,cur_len);
    int audio_len=cur_len; float *daudio=dbg(&O,(size_t)audio_len);
    kd_conv1d<<<GRID2(audio_len,1),TPB>>>(cur,dev_w(sd->final_conv_weight,(size_t)cur_ch*7),dev_w(sd->final_conv_bias,1),daudio,cur_ch,1,cur_len,7,1);
    CK(cudaStreamSynchronize(cudaStreamPerThread));
    float *audio=(float*)malloc((size_t)audio_len*sizeof(float)); if(!audio) return -1;
    cudaMemcpy(audio,daudio,(size_t)audio_len*sizeof(float),cudaMemcpyDeviceToHost);
    for(int i=0;i<audio_len;++i){ if(audio[i]<-1.f)audio[i]=-1.f; if(audio[i]>1.f)audio[i]=1.f; }
    *audio_out=audio; *n_out=audio_len;
    return 0;
}
