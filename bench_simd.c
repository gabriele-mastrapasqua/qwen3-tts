/*
 * Micro-benchmark: SIMD vs scalar for hot loops
 * Tests NEON versions of the top scalar consumers
 * Build: clang -O3 -march=native -ffast-math -o bench_simd bench_simd.c -lm
 * Usage: ./bench_simd [0.6b|1.7b]
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

#ifdef __ARM_NEON
/* Fast sigmoid via Padé approximant of tanh:
 * sigmoid(x) = 0.5 * (1 + tanh(0.5*x))
 * tanh(a) ≈ a*(27+a²)/(27+9a²) — good for |a|<4, max error ~0.004 */
static inline float32x4_t fast_sigmoid_neon(float32x4_t x) {
    float32x4_t half = vdupq_n_f32(0.5f);
    float32x4_t hx = vmulq_f32(half, x);
    float32x4_t hx2 = vmulq_f32(hx, hx);
    float32x4_t c27 = vdupq_n_f32(27.0f);
    float32x4_t c9 = vdupq_n_f32(9.0f);
    float32x4_t one = vdupq_n_f32(1.0f);
    float32x4_t num = vmulq_f32(hx, vaddq_f32(c27, hx2));
    float32x4_t den = vaddq_f32(c27, vmulq_f32(c9, hx2));
    float32x4_t inv = vrecpeq_f32(den);
    inv = vmulq_f32(inv, vrecpsq_f32(den, inv));
    inv = vmulq_f32(inv, vrecpsq_f32(den, inv));
    float32x4_t tanh_approx = vmulq_f32(num, inv);
    return vmulq_f32(half, vaddq_f32(one, tanh_approx));
}

static inline void swiglu_neon_fast(float *gate, int inter) {
    int o = 0;
    for (; o + 3 < inter; o += 4) {
        float32x4x2_t gu = vld2q_f32(gate + 2 * o);
        float32x4_t g = gu.val[0];
        float32x4_t u = gu.val[1];
        float32x4_t sig = fast_sigmoid_neon(g);
        float32x4_t result = vmulq_f32(vmulq_f32(g, sig), u);
        vst1q_f32(gate + o, result);
    }
    for (; o < inter; o++) {
        float g = gate[2*o], u = gate[2*o+1];
        gate[o] = g / (1.0f + expf(-g)) * u;
    }
}
#endif

int main(int argc, char **argv) {
    int large = (argc > 1 && strstr(argv[1], "1.7"));

    int hidden     = large ? 2048 : 1024;
    int inter      = large ? 6144 : 3072;
    int cp_inter   = 1024;
    int talker_layers = 28;
    int cp_layers  = 5;
    int cp_passes  = 15;
    int n_frames   = 60;
    int convnext_ch  = 1024;
    int convnext_len = 120;
    int pw_dim     = 4096;

    printf("=== SIMD vs SCALAR COMPARISON (%s model, %d frames) ===\n",
           large ? "1.7B" : "0.6B", n_frames);
    printf("Talker: hidden=%d, inter=%d | CP: hidden=1024, inter=%d\n\n", hidden, inter, cp_inter);

    float *gate = (float *)malloc(2 * inter * sizeof(float));
    float *gate_backup = (float *)malloc(2 * inter * sizeof(float));
    float *cp_gate = (float *)malloc(2 * cp_inter * sizeof(float));
    float *cp_gate_backup = (float *)malloc(2 * cp_inter * sizeof(float));
    float *pw1 = (float *)malloc((size_t)pw_dim * convnext_len * sizeof(float));
    float *pw1_backup = (float *)malloc((size_t)pw_dim * convnext_len * sizeof(float));
    float *signal = (float *)malloc((size_t)convnext_ch * convnext_len * sizeof(float));
    float *norm_w = (float *)malloc(convnext_ch * sizeof(float));
    float *norm_b = (float *)malloc(convnext_ch * sizeof(float));
    float *dw_out = (float *)calloc((size_t)convnext_ch * convnext_len, sizeof(float));
    float *dw_weight = (float *)malloc((size_t)convnext_ch * 7 * sizeof(float));

    srand(42);
    for (int i = 0; i < 2 * inter; i++) gate[i] = gate_backup[i] = (float)rand()/RAND_MAX * 2 - 1;
    for (int i = 0; i < 2 * cp_inter; i++) cp_gate[i] = cp_gate_backup[i] = (float)rand()/RAND_MAX * 2 - 1;
    for (size_t i = 0; i < (size_t)pw_dim * convnext_len; i++) pw1[i] = pw1_backup[i] = (float)rand()/RAND_MAX * 2 - 1;
    for (size_t i = 0; i < (size_t)convnext_ch * convnext_len; i++) signal[i] = (float)rand()/RAND_MAX;
    for (int i = 0; i < convnext_ch; i++) { norm_w[i] = 1.0f; norm_b[i] = 0.0f; }
    for (int i = 0; i < convnext_ch * 7; i++) dw_weight[i] = (float)rand()/RAND_MAX * 0.1f;

    double t0, t1;
    volatile float sink = 0;

#ifdef __ARM_NEON
    /* === 1. Talker SwiGLU === */
    int talker_calls = talker_layers * n_frames;

    memcpy(gate, gate_backup, 2 * inter * sizeof(float));
    t0 = now_ms();
    for (int c = 0; c < talker_calls; c++) {
        for (int o = 0; o < inter; o++) {
            float g = gate[2*o], u = gate[2*o+1];
            gate[o] = g / (1.0f + expf(-g)) * u;
        }
    }
    t1 = now_ms();
    sink += gate[0];
    double t_swiglu_scalar = t1 - t0;
    printf("1a. Talker SwiGLU SCALAR:        %7.2f ms  (%.1fM expf)\n",
           t_swiglu_scalar, (double)talker_calls * inter / 1e6);

    memcpy(gate, gate_backup, 2 * inter * sizeof(float));
    t0 = now_ms();
    for (int c = 0; c < talker_calls; c++)
        swiglu_neon_fast(gate, inter);
    t1 = now_ms();
    sink += gate[0];
    double t_swiglu_fast = t1 - t0;
    printf("1b. Talker SwiGLU NEON fast_sig: %7.2f ms  (%.1fx speedup)\n",
           t_swiglu_fast, t_swiglu_scalar/t_swiglu_fast);

    /* Accuracy check */
    memcpy(gate, gate_backup, 2 * inter * sizeof(float));
    float ref_vals[4], fast_vals[4];
    for (int o = 0; o < 4; o++) {
        float g = gate[2*o], u = gate[2*o+1];
        ref_vals[o] = g / (1.0f + expf(-g)) * u;
    }
    swiglu_neon_fast(gate, inter);
    for (int o = 0; o < 4; o++) fast_vals[o] = gate[o];
    float max_err = 0;
    for (int o = 0; o < 4; o++) {
        float err = fabsf(ref_vals[o] - fast_vals[o]) / (fabsf(ref_vals[o]) + 1e-8f);
        if (err > max_err) max_err = err;
    }
    printf("   Accuracy: max relative error = %.6f (first 4 elements)\n", max_err);

    /* === 2. CP SwiGLU === */
    int cp_calls = cp_layers * cp_passes * n_frames;

    memcpy(cp_gate, cp_gate_backup, 2 * cp_inter * sizeof(float));
    t0 = now_ms();
    for (int c = 0; c < cp_calls; c++) {
        for (int o = 0; o < cp_inter; o++) {
            float g = cp_gate[2*o], u = cp_gate[2*o+1];
            cp_gate[o] = g / (1.0f + expf(-g)) * u;
        }
    }
    t1 = now_ms();
    sink += cp_gate[0];
    double t_cp_scalar = t1 - t0;
    printf("\n2a. CP SwiGLU SCALAR:            %7.2f ms  (%.1fM expf)\n",
           t_cp_scalar, (double)cp_calls * cp_inter / 1e6);

    memcpy(cp_gate, cp_gate_backup, 2 * cp_inter * sizeof(float));
    t0 = now_ms();
    for (int c = 0; c < cp_calls; c++)
        swiglu_neon_fast(cp_gate, cp_inter);
    t1 = now_ms();
    sink += cp_gate[0];
    double t_cp_fast = t1 - t0;
    printf("2b. CP SwiGLU NEON fast_sig:     %7.2f ms  (%.1fx speedup)\n",
           t_cp_fast, t_cp_scalar/t_cp_fast);

    /* === 3. GELU === */
    memcpy(pw1, pw1_backup, (size_t)pw_dim * convnext_len * sizeof(float));
    t0 = now_ms();
    for (int block = 0; block < 2; block++) {
        for (size_t i = 0; i < (size_t)pw_dim * convnext_len; i++) {
            float x = pw1[i];
            pw1[i] = 0.5f * x * (1.0f + erff(x * 0.7071067811865476f));
        }
    }
    t1 = now_ms();
    sink += pw1[0];
    double t_gelu_scalar = t1 - t0;
    printf("\n3a. GELU erff SCALAR:            %7.2f ms  (%.0fk erff)\n",
           t_gelu_scalar, 2.0 * pw_dim * convnext_len / 1e3);

    memcpy(pw1, pw1_backup, (size_t)pw_dim * convnext_len * sizeof(float));
    t0 = now_ms();
    for (int block = 0; block < 2; block++) {
        size_t total = (size_t)pw_dim * convnext_len;
        size_t i = 0;
        float32x4_t half = vdupq_n_f32(0.5f);
        float32x4_t one = vdupq_n_f32(1.0f);
        float32x4_t c_sqrt2pi = vdupq_n_f32(0.7978845608f);
        float32x4_t c_coeff = vdupq_n_f32(0.044715f);
        float32x4_t c27 = vdupq_n_f32(27.0f);
        float32x4_t c9 = vdupq_n_f32(9.0f);
        for (; i + 3 < total; i += 4) {
            float32x4_t x = vld1q_f32(pw1 + i);
            float32x4_t x3 = vmulq_f32(vmulq_f32(x, x), x);
            float32x4_t inner = vmulq_f32(c_sqrt2pi, vaddq_f32(x, vmulq_f32(c_coeff, x3)));
            float32x4_t a2 = vmulq_f32(inner, inner);
            float32x4_t num = vmulq_f32(inner, vaddq_f32(c27, a2));
            float32x4_t den = vaddq_f32(c27, vmulq_f32(c9, a2));
            float32x4_t inv_den = vrecpeq_f32(den);
            inv_den = vmulq_f32(inv_den, vrecpsq_f32(den, inv_den));
            inv_den = vmulq_f32(inv_den, vrecpsq_f32(den, inv_den));
            float32x4_t tanh_val = vmulq_f32(num, inv_den);
            float32x4_t result = vmulq_f32(vmulq_f32(half, x), vaddq_f32(one, tanh_val));
            vst1q_f32(pw1 + i, result);
        }
        for (; i < total; i++) {
            float x = pw1[i];
            pw1[i] = 0.5f * x * (1.0f + erff(x * 0.7071067811865476f));
        }
    }
    t1 = now_ms();
    sink += pw1[0];
    double t_gelu_neon = t1 - t0;
    printf("3b. GELU NEON tanh-approx:       %7.2f ms  (%.1fx speedup)\n",
           t_gelu_neon, t_gelu_scalar/t_gelu_neon);

    /* === 4. Depthwise conv NEON === */
    t0 = now_ms();
    for (int block = 0; block < 2; block++) {
        memset(dw_out, 0, (size_t)convnext_ch * convnext_len * sizeof(float));
        int ci = 0;
        for (; ci + 3 < convnext_ch; ci += 4) {
            for (int t = 0; t < convnext_len; t++) {
                float32x4_t sum = vdupq_n_f32(0);
                for (int k = 0; k < 7; k++) {
                    int in_pos = t - (6 - k);
                    if (in_pos >= 0 && in_pos < convnext_len) {
                        float32x4_t w = {dw_weight[ci*7+k], dw_weight[(ci+1)*7+k],
                                         dw_weight[(ci+2)*7+k], dw_weight[(ci+3)*7+k]};
                        float32x4_t s = {signal[(size_t)ci*convnext_len+in_pos],
                                         signal[(size_t)(ci+1)*convnext_len+in_pos],
                                         signal[(size_t)(ci+2)*convnext_len+in_pos],
                                         signal[(size_t)(ci+3)*convnext_len+in_pos]};
                        sum = vmlaq_f32(sum, w, s);
                    }
                }
                dw_out[(size_t)ci*convnext_len+t] = vgetq_lane_f32(sum, 0);
                dw_out[(size_t)(ci+1)*convnext_len+t] = vgetq_lane_f32(sum, 1);
                dw_out[(size_t)(ci+2)*convnext_len+t] = vgetq_lane_f32(sum, 2);
                dw_out[(size_t)(ci+3)*convnext_len+t] = vgetq_lane_f32(sum, 3);
            }
        }
    }
    t1 = now_ms();
    sink += dw_out[0];
    printf("\n4. Depthwise conv NEON:          %7.2f ms  (scalar ~0.88ms)\n", t1-t0);

    /* === 5. LayerNorm NEON === */
    t0 = now_ms();
    for (int block = 0; block < 2; block++) {
        for (int t = 0; t < convnext_len; t++) {
            float32x4_t vsum = vdupq_n_f32(0);
            float32x4_t vsumsq = vdupq_n_f32(0);
            int ci = 0;
            for (; ci + 3 < convnext_ch; ci += 4) {
                float32x4_t v = {
                    signal[(size_t)ci * convnext_len + t],
                    signal[(size_t)(ci+1) * convnext_len + t],
                    signal[(size_t)(ci+2) * convnext_len + t],
                    signal[(size_t)(ci+3) * convnext_len + t]
                };
                vsum = vaddq_f32(vsum, v);
                vsumsq = vmlaq_f32(vsumsq, v, v);
            }
            float sum = vaddvq_f32(vsum);
            float sum_sq = vaddvq_f32(vsumsq);
            for (; ci < convnext_ch; ci++) {
                float val = signal[(size_t)ci * convnext_len + t];
                sum += val; sum_sq += val * val;
            }
            float mean = sum / convnext_ch;
            float var = sum_sq / convnext_ch - mean * mean;
            float inv_std = 1.0f / sqrtf(var + 1e-5f);
            for (ci = 0; ci < convnext_ch; ci++) {
                float *p = &signal[(size_t)ci * convnext_len + t];
                *p = (*p - mean) * inv_std * norm_w[ci] + norm_b[ci];
            }
        }
    }
    t1 = now_ms();
    sink += signal[0];
    printf("5. LayerNorm NEON accum:         %7.2f ms  (scalar ~0.29ms)\n", t1-t0);

    /* === SUMMARY === */
    printf("\n=== POTENTIAL SAVINGS ===\n");
    double main_saved = (t_swiglu_scalar - t_swiglu_fast) + (t_cp_scalar - t_cp_fast);
    double decoder_saved = (t_gelu_scalar - t_gelu_neon);
    printf("MAIN THREAD:    %.1f ms saved (SwiGLU T+CP: %.1f → %.1f ms)\n",
           main_saved, t_swiglu_scalar + t_cp_scalar, t_swiglu_fast + t_cp_fast);
    printf("DECODER THREAD: %.1f ms saved (GELU: %.1f → %.1f ms)\n",
           decoder_saved, t_gelu_scalar, t_gelu_neon);

#else
    printf("No ARM NEON available, skipping SIMD benchmarks\n");
#endif

    printf("\nvolatile sink = %f\n", sink);
    free(gate); free(gate_backup); free(cp_gate); free(cp_gate_backup);
    free(pw1); free(pw1_backup); free(signal); free(norm_w); free(norm_b);
    free(dw_out); free(dw_weight);
    return 0;
}
