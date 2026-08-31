/* batching_bench.c — premise test for text-chunk BATCHING, across precisions. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <pthread.h>
#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

static double now_s(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }

enum { P_BF16, P_INT8, P_INT4, P_INT2, NPREC };
static const char *PNAME[NPREC] = { "bf16", "int8", "int4", "int2" };
static const double PBYTES[NPREC] = { 2.0, 1.0, 0.5, 0.25 };

static inline float decode(const uint8_t *W, size_t i, int prec) {
    switch (prec) {
        case P_BF16: { uint32_t u=((uint32_t)((const uint16_t*)W)[i])<<16; float f; memcpy(&f,&u,4); return f; }
        case P_INT8: return (float)((int8_t)W[i]) * 0.01f;
        case P_INT4: { uint8_t b=W[i>>1]; int n=(i&1)?(b>>4):(b&0xF); return (float)(n-8)*0.02f; }
        default:     { uint8_t b=W[i>>2]; int q=(b>>((i&3)*2))&0x3; return (float)(q-2)*0.05f; }
    }
}

static float gemv(const uint8_t *W, const float *x, int r0, int r1, int C, int prec) {
    float sink=0;
    for (int r=r0; r<r1; r++) { const uint8_t *w=W+(size_t)r*C*0; size_t base=(size_t)r*C; (void)w;
        float s=0; for (int k=0;k<C;k++) s += decode(W, base+k, prec) * x[k];
        sink += s;
    }
    return sink;
}

static float gemm16(const uint8_t *W, const float *X, int r0, int r1, int C, int prec) {
    float sink=0;
    for (int r=r0; r<r1; r++) {
        size_t base=(size_t)r*C;
#ifdef __ARM_NEON
        float32x4_t a0=vdupq_n_f32(0),a1=vdupq_n_f32(0),a2=vdupq_n_f32(0),a3=vdupq_n_f32(0);
        for (int k=0;k<C;k++) {
            float32x4_t wq=vdupq_n_f32(decode(W,base+k,prec));
            const float *xk=X+(size_t)k*16;
            a0=vfmaq_f32(a0,wq,vld1q_f32(xk));   a1=vfmaq_f32(a1,wq,vld1q_f32(xk+4));
            a2=vfmaq_f32(a2,wq,vld1q_f32(xk+8)); a3=vfmaq_f32(a3,wq,vld1q_f32(xk+12));
        }
        sink += vaddvq_f32(a0)+vaddvq_f32(a1)+vaddvq_f32(a2)+vaddvq_f32(a3);
#else
        float acc[16]={0};
        for (int k=0;k<C;k++){ float wv=decode(W,base+k,prec); const float*xk=X+(size_t)k*16;
            for(int b=0;b<16;b++) acc[b]+=wv*xk[b]; }
        for(int b=0;b<16;b++) sink+=acc[b];
#endif
    }
    return sink;
}

typedef struct { const uint8_t *W; const float *X; int r0,r1,C,B,prec,mode; float sink; } job_t;
static void *worker(void *a){ job_t *j=a;
    if (j->mode==0) for(int b=0;b<j->B;b++) j->sink+=gemv(j->W,j->X+(size_t)b,j->r0,j->r1,j->C,j->prec);
    else j->sink+=gemm16(j->W,j->X,j->r0,j->r1,j->C,j->prec);
    return NULL; }
static double timed_T(const uint8_t *W,const float *X,int R,int C,int B,int T,int prec,int mode,int reps,volatile float *sink){
    double t0=now_s();
    for(int it=0;it<reps;it++){ pthread_t th[64]; job_t jb[64];
        for(int t=0;t<T;t++){ jb[t]=(job_t){W,X,(int)((long)t*R/T),(int)((long)(t+1)*R/T),C,B,prec,mode,0};
            pthread_create(&th[t],NULL,worker,&jb[t]); }
        for(int t=0;t<T;t++){ pthread_join(th[t],NULL); *sink+=jb[t].sink; } }
    return (now_s()-t0)/reps*1e3;
}

static void run_shape(const char *name, int R, int C, int T) {
    const int B=16;
    uint8_t *W = malloc((size_t)R*C*2);
    float *X = malloc((size_t)C*B*sizeof(float));
    for (size_t i=0;i<(size_t)R*C*2;i++) W[i]=(uint8_t)((i*1103515245u+12345u)>>16);
    for (int k=0;k<C;k++) for(int b=0;b<B;b++) X[(size_t)k*B+b]=((float)(k%13)-6.0f)*0.02f+b*1e-4f;
    int reps = ((double)R*C*2/(1024*1024))>8 ? 30 : 200;
    volatile float sink=0;
    printf("  %-13s [%d x %d]\n", name, R, C);
    for (int p=0;p<NPREC;p++) {
        double wMB=(double)R*C*PBYTES[p]/(1024*1024);
        sink+=gemv(W,X,0,R,C,p); sink+=gemm16(W,X,0,R,C,p);
        double tv1=now_s(); for(int it=0;it<reps;it++) for(int b=0;b<B;b++) sink+=gemv(W,X+(size_t)b,0,R,C,p);
        tv1=(now_s()-tv1)/reps*1e3;
        double tg1=now_s(); for(int it=0;it<reps;it++) sink+=gemm16(W,X,0,R,C,p); tg1=(now_s()-tg1)/reps*1e3;
        double tvT=timed_T(W,X,R,C,B,T,p,0,reps,&sink);
        double tgT=timed_T(W,X,R,C,B,T,p,1,reps,&sink);
        printf("     %-5s %5.1fMB | 1T %2.2fx (GEMV %6.2f GEMM %6.2f) | %dT %2.2fx (GEMV %6.2f GEMM %6.2f)\n",
               PNAME[p], wMB, tv1/tg1, tv1, tg1, T, tvT/tgT, tvT, tgT);
    }
    free(W); free(X);
}

int main(void){
    int T=4;
    printf("=== BATCHING x PRECISION: 16x GEMV vs GEMM(16) speedup, per weight precision ===\n");
    printf("speedup>1 => batching amortizes weight read+unpack. Trend across precisions = the answer.\n\n");
    run_shape("1.7B gate_up", 5632, 2048, T);
    run_shape("1.7B down",    2048, 5632, T);
    run_shape("0.6B gate_up", 2816, 1024, T);
    return 0;
}
