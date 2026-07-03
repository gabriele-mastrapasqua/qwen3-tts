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
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

extern "C" { int g_cuda_decoder_conv_on = 0; }

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
extern "C" int qwen_cuda_conv_decoder_run(void *ctxv, float *signal_host, int cur_ch, int cur_len,
                                          float **audio_out, int *n_out);
