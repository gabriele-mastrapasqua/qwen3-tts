/*
 * qwen_tts_cuda_kernels.cu — CUDA compute kernels (G3), nvcc.
 *
 * Compiled only by `make cuda` (nvcc). Mirrors the M1-VALIDATED Metal kernels in
 * qwen_tts_metal.m 1:1 (same math, validated there via --gpu-selftest rel<5e-3),
 * so numerical correctness is high-confidence; only nvcc compilation is verified
 * on the DGX (no CUDA on the M1 dev box). cuBLAS handles the GEMM (matmat/matvec)
 * in qwen_tts_cuda.c; these kernels cover the ops cuBLAS can't: norm/rope/swiglu/
 * silu/elementwise/snake/attention/conv1d/transpose. Host-array launchers
 * (extern "C") mirror the Metal host API so a DGX selftest can reuse the CPU refs.
 *
 * PERF NOTE: these launchers upload/run/download per call (correctness harness).
 * The resident/fused versions (weights on device, one CUDA-graph per step) are
 * the perf follow-up — same as the Metal per-op → fused progression.
 */

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

__device__ __forceinline__ float bf16_to_f32_dev(uint16_t b) {
    return __uint_as_float((uint32_t)b << 16);
}

/* ---- kernels (mirror qwen_tts_metal.m) ---------------------------------- */

__global__ void k_rms_norm(const float *x, const float *w, float *y, uint dim, float eps) {
    extern __shared__ float part[];
    uint tid = threadIdx.x, tc = blockDim.x;
    float s = 0.0f;
    for (uint i = tid; i < dim; i += tc) s += x[i] * x[i];
    part[tid] = s; __syncthreads();
    for (uint st = tc / 2; st > 0; st >>= 1) { if (tid < st) part[tid] += part[tid + st]; __syncthreads(); }
    float inv = 1.0f / sqrtf(part[0] / (float)dim + eps);
    for (uint i = tid; i < dim; i += tc) y[i] = x[i] * inv * w[i];
}

__global__ void k_swiglu(const float *in, float *out, uint n) {
    uint i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
    float g = in[2*i], u = in[2*i+1]; out[i] = g / (1.0f + expf(-g)) * u;
}
__global__ void k_silu(const float *x, float *out, uint n) {
    uint i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
    float v = x[i]; out[i] = v / (1.0f + expf(-v));
}
__global__ void k_add(const float *a, const float *b, float *o, uint n) {
    uint i = blockIdx.x*blockDim.x+threadIdx.x; if (i<n) o[i]=a[i]+b[i]; }
__global__ void k_mul(const float *a, const float *b, float *o, uint n) {
    uint i = blockIdx.x*blockDim.x+threadIdx.x; if (i<n) o[i]=a[i]*b[i]; }
__global__ void k_scale(const float *a, float *o, float s, uint n) {
    uint i = blockIdx.x*blockDim.x+threadIdx.x; if (i<n) o[i]=a[i]*s; }

__global__ void k_rope(float *x, const float *cosv, const float *sinv, uint head_dim) {
    uint pairs = head_dim/2; uint gid = blockIdx.x*blockDim.x+threadIdx.x;
    uint h = gid / pairs, d = gid % pairs;
    float *vec = x + (size_t)h*head_dim; float c = cosv[d], sn = sinv[d];
    float xe = vec[2*d], xo = vec[2*d+1];
    vec[2*d] = xe*c - xo*sn; vec[2*d+1] = xo*c + xe*sn;
}

__global__ void k_snake(float *data, const float *la, const float *lb, uint length) {
    uint t = blockIdx.x*blockDim.x+threadIdx.x, c = blockIdx.y; if (t>=length) return;
    float a = expf(la[c]), inv_b = expf(-lb[c]);
    size_t idx = (size_t)c*length + t; float x = data[idx]; float s = sinf(a*x);
    data[idx] = x + inv_b*s*s;
}

__global__ void k_attention(const float *Q, const float *K, const float *V, float *O,
                            uint seq_q, uint seq_k, uint n_heads, uint n_kv, uint head_dim,
                            float scale, uint q_offset) {
    uint sq = blockIdx.x, h = blockIdx.y; if (sq>=seq_q || h>=n_heads) return;
    uint kvh = h / (n_heads/n_kv); uint qpos = q_offset+sq;
    uint valid = qpos+1; if (valid > seq_k) valid = seq_k;
    const float *q = Q + ((size_t)sq*n_heads+h)*head_dim;
    float *o = O + ((size_t)sq*n_heads+h)*head_dim;
    float m = -1e30f;
    for (uint j=0;j<valid;++j){ const float *k=K+((size_t)j*n_kv+kvh)*head_dim; float dot=0;
        for (uint d=0;d<head_dim;++d) dot+=q[d]*k[d]; dot*=scale; if (dot>m) m=dot; }
    for (uint d=0;d<head_dim;++d) o[d]=0; float denom=0;
    for (uint j=0;j<valid;++j){ const float *k=K+((size_t)j*n_kv+kvh)*head_dim; float dot=0;
        for (uint d=0;d<head_dim;++d) dot+=q[d]*k[d]; dot=expf(dot*scale-m); denom+=dot;
        const float *v=V+((size_t)j*n_kv+kvh)*head_dim; for (uint d=0;d<head_dim;++d) o[d]+=dot*v[d]; }
    float inv=1.0f/denom; for (uint d=0;d<head_dim;++d) o[d]*=inv;
}

__global__ void k_conv1d(const float *in, const float *w, const float *bias, float *out,
                         uint in_ch, uint length, uint ksz, uint dil) {
    uint t = blockIdx.x*blockDim.x+threadIdx.x, oc = blockIdx.y; if (t>=length) return;
    int pad = (int)(ksz-1)*(int)dil; float sum = bias[oc];
    for (uint ic=0;ic<in_ch;++ic) for (uint k=0;k<ksz;++k){ int ip=(int)t-pad+(int)k*(int)dil;
        if (ip>=0 && ip<(int)length) sum += w[((size_t)oc*in_ch+ic)*ksz+k]*in[(size_t)ic*length+ip]; }
    out[(size_t)oc*length+t] = sum;
}

__global__ void k_conv_transpose1d(const float *in, const float *w, const float *bias, float *out,
                                   uint in_ch, uint out_ch, uint in_len, uint out_len, uint ksz, uint stride) {
    uint p = blockIdx.x*blockDim.x+threadIdx.x, oc = blockIdx.y; if (p>=out_len) return;
    int full = (int)(in_len-1)*(int)stride+(int)ksz; int trim = (int)ksz-(int)stride;
    float sum = bias[oc];
    if ((int)p < full-trim) for (uint k=0;k<ksz;++k){ int sh=(int)p-(int)k; if (sh<0) continue;
        if ((uint)sh % stride) continue; int tt=sh/(int)stride; if (tt<0||tt>=(int)in_len) continue;
        for (uint ic=0;ic<in_ch;++ic) sum += in[(size_t)ic*in_len+tt]*w[((size_t)ic*out_ch+oc)*ksz+k]; }
    out[(size_t)oc*out_len+p] = sum;
}

/* ---- host-array launchers (extern "C"): upload → launch → download ------- */
#define CDIV(a,b) (((a)+(b)-1)/(b))
static float *dup_up(const float *h, size_t n) { float *d; cudaMalloc(&d, n*sizeof(float)); if (h) cudaMemcpy(d,h,n*sizeof(float),cudaMemcpyHostToDevice); return d; }

extern "C" {

void qwen_cuda_rms_norm(float *out, const float *x, const float *w, int dim, float eps) {
    float *dx=dup_up(x,dim), *dw=dup_up(w,dim), *dy=dup_up(NULL,dim);
    k_rms_norm<<<1,256,256*sizeof(float)>>>(dx,dw,dy,dim,eps);
    cudaMemcpy(out,dy,dim*sizeof(float),cudaMemcpyDeviceToHost);
    cudaFree(dx);cudaFree(dw);cudaFree(dy);
}
void qwen_cuda_swiglu(float *out, const float *gate_up, int n) {
    float *di=dup_up(gate_up,2*n), *dq=dup_up(NULL,n);
    k_swiglu<<<CDIV(n,256),256>>>(di,dq,n); cudaMemcpy(out,dq,n*sizeof(float),cudaMemcpyDeviceToHost);
    cudaFree(di);cudaFree(dq);
}
void qwen_cuda_silu(float *out, const float *x, int n) {
    float *di=dup_up(x,n), *dq=dup_up(NULL,n); k_silu<<<CDIV(n,256),256>>>(di,dq,n);
    cudaMemcpy(out,dq,n*sizeof(float),cudaMemcpyDeviceToHost); cudaFree(di);cudaFree(dq);
}
void qwen_cuda_add(float *out, const float *a, const float *b, int n) {
    float *da=dup_up(a,n),*db=dup_up(b,n),*dq=dup_up(NULL,n); k_add<<<CDIV(n,256),256>>>(da,db,dq,n);
    cudaMemcpy(out,dq,n*sizeof(float),cudaMemcpyDeviceToHost); cudaFree(da);cudaFree(db);cudaFree(dq);
}
void qwen_cuda_mul(float *out, const float *a, const float *b, int n) {
    float *da=dup_up(a,n),*db=dup_up(b,n),*dq=dup_up(NULL,n); k_mul<<<CDIV(n,256),256>>>(da,db,dq,n);
    cudaMemcpy(out,dq,n*sizeof(float),cudaMemcpyDeviceToHost); cudaFree(da);cudaFree(db);cudaFree(dq);
}
void qwen_cuda_scale(float *out, const float *a, float s, int n) {
    float *da=dup_up(a,n),*dq=dup_up(NULL,n); k_scale<<<CDIV(n,256),256>>>(da,dq,s,n);
    cudaMemcpy(out,dq,n*sizeof(float),cudaMemcpyDeviceToHost); cudaFree(da);cudaFree(dq);
}
void qwen_cuda_rope(float *x, const float *cosv, const float *sinv, int n_heads, int head_dim) {
    int pairs=head_dim/2, total=n_heads*pairs;
    float *dx=dup_up(x,(size_t)n_heads*head_dim), *dc=dup_up(cosv,pairs), *ds=dup_up(sinv,pairs);
    k_rope<<<CDIV(total,256),256>>>(dx,dc,ds,head_dim);
    cudaMemcpy(x,dx,(size_t)n_heads*head_dim*sizeof(float),cudaMemcpyDeviceToHost);
    cudaFree(dx);cudaFree(dc);cudaFree(ds);
}
void qwen_cuda_snake(float *data, const float *la, const float *lb, int channels, int length) {
    float *dd=dup_up(data,(size_t)channels*length),*dla=dup_up(la,channels),*dlb=dup_up(lb,channels);
    dim3 grid(CDIV(length,256),channels); k_snake<<<grid,256>>>(dd,dla,dlb,length);
    cudaMemcpy(data,dd,(size_t)channels*length*sizeof(float),cudaMemcpyDeviceToHost);
    cudaFree(dd);cudaFree(dla);cudaFree(dlb);
}
void qwen_cuda_attention(float *O, const float *Q, const float *K, const float *V,
                         int seq_q, int seq_k, int n_heads, int n_kv, int head_dim, float scale, int q_offset) {
    size_t qn=(size_t)seq_q*n_heads*head_dim, kn=(size_t)seq_k*n_kv*head_dim;
    float *dQ=dup_up(Q,qn),*dK=dup_up(K,kn),*dV=dup_up(V,kn),*dO=dup_up(NULL,qn);
    dim3 grid(seq_q,n_heads); k_attention<<<grid,1>>>(dQ,dK,dV,dO,seq_q,seq_k,n_heads,n_kv,head_dim,scale,q_offset);
    cudaMemcpy(O,dO,qn*sizeof(float),cudaMemcpyDeviceToHost);
    cudaFree(dQ);cudaFree(dK);cudaFree(dV);cudaFree(dO);
}
void qwen_cuda_conv1d(float *out, const float *in, const float *weight, const float *bias,
                      int in_ch, int out_ch, int length, int ksz, int dilation) {
    float *di=dup_up(in,(size_t)in_ch*length),*dw=dup_up(weight,(size_t)out_ch*in_ch*ksz),
          *db=dup_up(bias,out_ch),*dq=dup_up(NULL,(size_t)out_ch*length);
    dim3 grid(CDIV(length,256),out_ch); k_conv1d<<<grid,256>>>(di,dw,db,dq,in_ch,length,ksz,dilation);
    cudaMemcpy(out,dq,(size_t)out_ch*length*sizeof(float),cudaMemcpyDeviceToHost);
    cudaFree(di);cudaFree(dw);cudaFree(db);cudaFree(dq);
}
void qwen_cuda_conv_transpose1d(float *out, const float *in, const float *weight, const float *bias,
                                int in_ch, int out_ch, int in_len, int out_len, int ksz, int stride) {
    float *di=dup_up(in,(size_t)in_ch*in_len),*dw=dup_up(weight,(size_t)in_ch*out_ch*ksz),
          *db=dup_up(bias,out_ch),*dq=dup_up(NULL,(size_t)out_ch*out_len);
    dim3 grid(CDIV(out_len,256),out_ch); k_conv_transpose1d<<<grid,256>>>(di,dw,db,dq,in_ch,out_ch,in_len,out_len,ksz,stride);
    cudaMemcpy(out,dq,(size_t)out_ch*out_len*sizeof(float),cudaMemcpyDeviceToHost);
    cudaFree(di);cudaFree(dw);cudaFree(db);cudaFree(dq);
}

} /* extern "C" */
