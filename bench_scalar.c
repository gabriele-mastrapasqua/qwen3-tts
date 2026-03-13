/*
 * Micro-benchmark: measure time spent in scalar hot loops
 * Supports both 0.6B and 1.7B model dimensions
 * Build: clang -O3 -march=native -ffast-math -o bench_scalar bench_scalar.c -lm
 * Usage: ./bench_scalar [0.6b|1.7b]
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

int main(int argc, char **argv) {
    int large = (argc > 1 && strstr(argv[1], "1.7"));

    /* Model dimensions */
    int hidden     = large ? 2048 : 1024;
    int inter      = large ? 6144 : 3072;  /* Talker intermediate */
    int cp_inter   = 1024;                  /* CP intermediate (same for both!) */
    int talker_layers = 28;
    int cp_layers  = 5;
    int cp_passes  = 15;
    int n_frames   = 60;
    int vocab      = 3072;      /* codec vocab for sampling */
    int convnext_ch  = 1024;
    int convnext_len = 120;     /* ~60 frames * 2 upsample */
    int pw_dim     = 4096;

    printf("=== SCALAR HOT LOOP MICRO-BENCHMARKS (%s model, %d frames) ===\n",
           large ? "1.7B" : "0.6B", n_frames);
    printf("Talker: hidden=%d, inter=%d, layers=%d\n", hidden, inter, talker_layers);
    printf("CP: hidden=1024, inter=%d, layers=%d, passes=%d\n\n", cp_inter, cp_layers, cp_passes);

    /* Allocate buffers */
    float *gate = (float *)malloc(2 * inter * sizeof(float));
    float *cp_gate = (float *)malloc(2 * cp_inter * sizeof(float));
    float *logits = (float *)malloc(vocab * sizeof(float));
    float *residual = (float *)malloc(hidden * sizeof(float));
    float *proj = (float *)malloc(hidden * sizeof(float));
    float *signal = (float *)malloc((size_t)convnext_ch * convnext_len * sizeof(float));
    float *dw_out = (float *)calloc((size_t)convnext_ch * convnext_len, sizeof(float));
    float *pw1 = (float *)malloc((size_t)pw_dim * convnext_len * sizeof(float));
    float *dw_weight = (float *)malloc((size_t)convnext_ch * 7 * sizeof(float));
    float *norm_w = (float *)malloc(convnext_ch * sizeof(float));
    float *norm_b = (float *)malloc(convnext_ch * sizeof(float));

    /* Fill with random data */
    srand(42);
    for (int i = 0; i < 2 * inter; i++) gate[i] = (float)rand()/RAND_MAX * 2 - 1;
    for (int i = 0; i < 2 * cp_inter; i++) cp_gate[i] = (float)rand()/RAND_MAX * 2 - 1;
    for (int i = 0; i < vocab; i++) logits[i] = (float)rand()/RAND_MAX * 10 - 5;
    for (int i = 0; i < hidden; i++) { residual[i] = (float)rand()/RAND_MAX; proj[i] = (float)rand()/RAND_MAX; }
    for (size_t i = 0; i < (size_t)convnext_ch * convnext_len; i++) signal[i] = (float)rand()/RAND_MAX;
    for (int i = 0; i < convnext_ch * 7; i++) dw_weight[i] = (float)rand()/RAND_MAX * 0.1f;
    for (int i = 0; i < convnext_ch; i++) { norm_w[i] = 1.0f; norm_b[i] = 0.0f; }
    for (size_t i = 0; i < (size_t)pw_dim * convnext_len; i++) pw1[i] = (float)rand()/RAND_MAX * 2 - 1;

    double t0, t1;
    volatile float sink = 0;

    /* 1. Talker SwiGLU: 28 layers × n_frames steps */
    int talker_swiglu_calls = talker_layers * n_frames;
    t0 = now_ms();
    for (int c = 0; c < talker_swiglu_calls; c++) {
        for (int o = 0; o < inter; o++) {
            float g = gate[2*o];
            float u = gate[2*o + 1];
            gate[o] = g / (1.0f + expf(-g)) * u;
        }
    }
    t1 = now_ms();
    sink += gate[0];
    printf("1. Talker SwiGLU expf:    %7.2f ms  (%d calls × %d elements = %.1fM expf)\n",
           t1-t0, talker_swiglu_calls, inter, (double)talker_swiglu_calls * inter / 1e6);

    /* 2. CP SwiGLU: 5 layers × 15 passes × n_frames */
    int cp_swiglu_calls = cp_layers * cp_passes * n_frames;
    t0 = now_ms();
    for (int c = 0; c < cp_swiglu_calls; c++) {
        for (int o = 0; o < cp_inter; o++) {
            float g = cp_gate[2*o];
            float u = cp_gate[2*o + 1];
            cp_gate[o] = g / (1.0f + expf(-g)) * u;
        }
    }
    t1 = now_ms();
    sink += cp_gate[0];
    printf("2. CP SwiGLU expf:        %7.2f ms  (%d calls × %d elements = %.1fM expf)\n",
           t1-t0, cp_swiglu_calls, cp_inter, (double)cp_swiglu_calls * cp_inter / 1e6);

    /* 3. Residual add: talker + CP */
    int residual_calls = (talker_layers * 2 + cp_layers * cp_passes * 2) * n_frames;
    t0 = now_ms();
    for (int c = 0; c < residual_calls; c++) {
        for (int i = 0; i < hidden; i++)
            residual[i] += proj[i];
    }
    t1 = now_ms();
    sink += residual[0];
    printf("3. Residual add (T+CP):   %7.2f ms  (%d calls × %d elements)\n",
           t1-t0, residual_calls, hidden);

    /* 4. Softmax (codec vocab, once per frame) */
    t0 = now_ms();
    for (int f = 0; f < n_frames; f++) {
        float max_val = logits[0];
        for (int i = 1; i < vocab; i++) if (logits[i] > max_val) max_val = logits[i];
        float sum = 0;
        float inv_temp = 1.0f / 0.9f;
        for (int i = 0; i < vocab; i++) {
            logits[i] = expf((logits[i] - max_val) * inv_temp);
            sum += logits[i];
        }
        for (int i = 0; i < vocab; i++) logits[i] /= sum;
    }
    t1 = now_ms();
    sink += logits[0];
    printf("4. Softmax (codec):       %7.2f ms  (%d frames × %d vocab = %.0fk expf)\n",
           t1-t0, n_frames, vocab, (double)n_frames * vocab / 1e3);

    /* 5. ConvNeXt depthwise conv k=7 (2 blocks, speech decoder) */
    t0 = now_ms();
    for (int block = 0; block < 2; block++) {
        memset(dw_out, 0, (size_t)convnext_ch * convnext_len * sizeof(float));
        for (int ci = 0; ci < convnext_ch; ci++) {
            for (int t = 0; t < convnext_len; t++) {
                float sum = 0;
                for (int k = 0; k < 7; k++) {
                    int in_pos = t - (6 - k);
                    if (in_pos >= 0 && in_pos < convnext_len)
                        sum += dw_weight[ci * 7 + k] * signal[(size_t)ci * convnext_len + in_pos];
                }
                dw_out[(size_t)ci * convnext_len + t] = sum;
            }
        }
    }
    t1 = now_ms();
    sink += dw_out[0];
    printf("5. Depthwise conv k=7:    %7.2f ms  (2 blocks × %d ch × %d len)\n",
           t1-t0, convnext_ch, convnext_len);

    /* 6. ConvNeXt LayerNorm per timestep */
    t0 = now_ms();
    for (int block = 0; block < 2; block++) {
        for (int t = 0; t < convnext_len; t++) {
            float sum = 0, sum_sq = 0;
            for (int ci = 0; ci < convnext_ch; ci++) {
                float val = signal[(size_t)ci * convnext_len + t];
                sum += val; sum_sq += val * val;
            }
            float mean = sum / convnext_ch;
            float var = sum_sq / convnext_ch - mean * mean;
            float inv_std = 1.0f / sqrtf(var + 1e-5f);
            for (int ci = 0; ci < convnext_ch; ci++) {
                float *p = &signal[(size_t)ci * convnext_len + t];
                *p = (*p - mean) * inv_std * norm_w[ci] + norm_b[ci];
            }
        }
    }
    t1 = now_ms();
    sink += signal[0];
    printf("6. LayerNorm per-t:       %7.2f ms  (2 blocks × %d timesteps × %d ch)\n",
           t1-t0, convnext_len, convnext_ch);

    /* 7. GELU (erff) */
    t0 = now_ms();
    for (int block = 0; block < 2; block++) {
        for (size_t i = 0; i < (size_t)pw_dim * convnext_len; i++) {
            float x = pw1[i];
            pw1[i] = 0.5f * x * (1.0f + erff(x * 0.7071067811865476f));
        }
    }
    t1 = now_ms();
    sink += pw1[0];
    printf("7. GELU erff:             %7.2f ms  (2 blocks × %d × %d = %.0fk erff)\n",
           t1-t0, pw_dim, convnext_len, 2.0 * pw_dim * convnext_len / 1e3);

    /* 8. ConvNeXt gamma+residual */
    float *gamma_vec = (float *)malloc(convnext_ch * sizeof(float));
    float *resid = (float *)malloc((size_t)convnext_ch * convnext_len * sizeof(float));
    for (int i = 0; i < convnext_ch; i++) gamma_vec[i] = 0.99f;
    memcpy(resid, signal, (size_t)convnext_ch * convnext_len * sizeof(float));
    t0 = now_ms();
    for (int block = 0; block < 2; block++) {
        for (int ci = 0; ci < convnext_ch; ci++) {
            float g = gamma_vec[ci];
            for (int t = 0; t < convnext_len; t++)
                signal[(size_t)ci * convnext_len + t] = resid[(size_t)ci * convnext_len + t] + signal[(size_t)ci * convnext_len + t] * g;
        }
    }
    t1 = now_ms();
    sink += signal[0];
    printf("8. Gamma+residual:        %7.2f ms  (2 blocks × %d ch × %d len)\n",
           t1-t0, convnext_ch, convnext_len);

    printf("\n--- TOTALS ---\n");
    printf("Items 1-4 are on MAIN THREAD (block generation)\n");
    printf("Items 5-8 are on DECODER THREAD (overlapped)\n");
    printf("\nvolatile sink = %f (prevents DCE)\n", sink);

    free(gate); free(cp_gate); free(logits); free(residual); free(proj);
    free(signal); free(dw_out); free(pw1); free(dw_weight);
    free(norm_w); free(norm_b); free(gamma_vec); free(resid);
    return 0;
}
