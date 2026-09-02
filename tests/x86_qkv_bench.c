/* Measure the x86 INT8 QKV path with one shared activation quantization. */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "../qwen_tts_kernels.h"
#include "../qwen_tts_thread.h"

#if defined(__AVX512VNNI__)

static uint32_t rng_state = 0x4d595df4u;

static float random_value(void) {
    rng_state = rng_state * 1664525u + 1013904223u;
    return ((float)(rng_state >> 8) / 16777216.0f) * 2.0f - 1.0f;
}

static void *aligned_zero(size_t bytes) {
    void *p = NULL;
    size_t size = (bytes + 63u) & ~63u;
    if (size == 0) size = 64;
    if (posix_memalign(&p, 64, size) != 0) return NULL;
    memset(p, 0, size);
    return p;
}

static double now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e9 + (double)ts.tv_nsec;
}

static double max_abs_diff(const float *a, const float *b, size_t n) {
    double max_err = 0.0;
    for (size_t i = 0; i < n; i++) {
        double e = fabs((double)a[i] - (double)b[i]);
        if (e > max_err) max_err = e;
    }
    return max_err;
}

static double rel_l2_diff(const float *a, const float *b, size_t n) {
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < n; i++) {
        double x = (double)a[i], y = (double)b[i];
        double d = x - y;
        num += d * d;
        den += y * y;
    }
    return sqrt(num / (den > 0.0 ? den : 1.0));
}

static double median(double *values, int count) {
    for (int i = 1; i < count; i++) {
        double value = values[i];
        int j = i;
        while (j > 0 && values[j - 1] > value) {
            values[j] = values[j - 1];
            j--;
        }
        values[j] = value;
    }
    return values[count / 2];
}

static int run_case(const char *name, int in_dim, int q_dim, int kv_dim,
                    int reps) {
    qwen_vnni_row_sums_reset();
    const size_t q_count = (size_t)q_dim * in_dim;
    const size_t kv_count = (size_t)kv_dim * in_dim;
    int8_t *wq = (int8_t *)aligned_zero(q_count);
    int8_t *wk = (int8_t *)aligned_zero(kv_count);
    int8_t *wv = (int8_t *)aligned_zero(kv_count);
    float *sq = (float *)aligned_zero((size_t)q_dim * sizeof(float));
    float *sk = (float *)aligned_zero((size_t)kv_dim * sizeof(float));
    float *sv = (float *)aligned_zero((size_t)kv_dim * sizeof(float));
    float *x = (float *)aligned_zero((size_t)in_dim * sizeof(float));
    float *q = (float *)aligned_zero((size_t)q_dim * sizeof(float));
    float *k = (float *)aligned_zero((size_t)kv_dim * sizeof(float));
    float *v = (float *)aligned_zero((size_t)kv_dim * sizeof(float));
    float *q_ref = (float *)aligned_zero((size_t)q_dim * sizeof(float));
    float *k_ref = (float *)aligned_zero((size_t)kv_dim * sizeof(float));
    float *v_ref = (float *)aligned_zero((size_t)kv_dim * sizeof(float));
    if (!wq || !wk || !wv || !sq || !sk || !sv || !x || !q || !k || !v ||
        !q_ref || !k_ref || !v_ref) {
        fprintf(stderr, "allocation failed for %s\n", name);
        free(wq); free(wk); free(wv); free(sq); free(sk); free(sv); free(x);
        free(q); free(k); free(v); free(q_ref); free(k_ref); free(v_ref);
        return 2;
    }
    for (size_t i = 0; i < q_count; i++) wq[i] = (int8_t)(random_value() * 127.0f);
    for (size_t i = 0; i < kv_count; i++) {
        wk[i] = (int8_t)(random_value() * 127.0f);
        wv[i] = (int8_t)(random_value() * 127.0f);
    }
    for (int i = 0; i < q_dim; i++) sq[i] = 0.002f + 0.001f * fabsf(random_value());
    for (int i = 0; i < kv_dim; i++) {
        sk[i] = 0.002f + 0.001f * fabsf(random_value());
        sv[i] = 0.002f + 0.001f * fabsf(random_value());
    }
    for (int i = 0; i < in_dim; i++) x[i] = random_value();

    qwen_matvec_int8(q_ref, wq, sq, x, q_dim, in_dim);
    qwen_matvec_int8(k_ref, wk, sk, x, kv_dim, in_dim);
    qwen_matvec_int8(v_ref, wv, sv, x, kv_dim, in_dim);
    qwen_matvec_int8_qkv(q, k, v, wq, sq, wk, sk, wv, sv,
                         x, in_dim, q_dim, kv_dim);
    double err = max_abs_diff(q, q_ref, q_dim);
    double err_k = max_abs_diff(k, k_ref, kv_dim);
    double err_v = max_abs_diff(v, v_ref, kv_dim);
    double rel = rel_l2_diff(q, q_ref, q_dim);
    double rel_k = rel_l2_diff(k, k_ref, kv_dim);
    double rel_v = rel_l2_diff(v, v_ref, kv_dim);
    if (err > 1e-5 || err_k > 1e-5 || err_v > 1e-5) {
        fprintf(stderr, "FAIL %s: max_abs q/k/v %.3e %.3e %.3e\n",
                name, err, err_k, err_v);
        free(wq); free(wk); free(wv); free(sq); free(sk); free(sv); free(x);
        free(q); free(k); free(v); free(q_ref); free(k_ref); free(v_ref);
        return 1;
    }

    /* Warm both entry points before collecting samples. */
    qwen_matvec_int8(q_ref, wq, sq, x, q_dim, in_dim);
    qwen_matvec_int8(k_ref, wk, sk, x, kv_dim, in_dim);
    qwen_matvec_int8(v_ref, wv, sv, x, kv_dim, in_dim);
    qwen_matvec_int8_qkv(q, k, v, wq, sq, wk, sk, wv, sv,
                         x, in_dim, q_dim, kv_dim);

    double direct_samples[64], combined_samples[64];
    if (reps > (int)(sizeof direct_samples / sizeof direct_samples[0])) reps = 64;
    for (int r = 0; r < reps; r++) {
        double t0 = now_ns();
        qwen_matvec_int8(q_ref, wq, sq, x, q_dim, in_dim);
        qwen_matvec_int8(k_ref, wk, sk, x, kv_dim, in_dim);
        qwen_matvec_int8(v_ref, wv, sv, x, kv_dim, in_dim);
        direct_samples[r] = now_ns() - t0;
        t0 = now_ns();
        qwen_matvec_int8_qkv(q, k, v, wq, sq, wk, sk, wv, sv,
                             x, in_dim, q_dim, kv_dim);
        combined_samples[r] = now_ns() - t0;
    }
    double direct_ns = median(direct_samples, reps);
    double combined_ns = median(combined_samples, reps);
    printf("%s in=%d q=%d kv=%d direct_ns=%.0f qkv_ns=%.0f speedup=%.3fx "
           "max_abs=%.3e/%.3e/%.3e rel_l2=%.3e/%.3e/%.3e\n",
           name, in_dim, q_dim, kv_dim, direct_ns, combined_ns,
           direct_ns / combined_ns, err, err_k, err_v, rel, rel_k, rel_v);

    free(wq); free(wk); free(wv); free(sq); free(sk); free(sv); free(x);
    free(q); free(k); free(v); free(q_ref); free(k_ref); free(v_ref);
    return 0;
}

static int run_matmat_case(const char *name, int in_dim, int q_dim, int kv_dim,
                           int B, int reps) {
    qwen_vnni_row_sums_reset();
    const size_t q_count = (size_t)q_dim * in_dim;
    const size_t kv_count = (size_t)kv_dim * in_dim;
    const size_t q_out = (size_t)q_dim * B;
    const size_t kv_out = (size_t)kv_dim * B;
    int8_t *wq = (int8_t *)aligned_zero(q_count);
    int8_t *wk = (int8_t *)aligned_zero(kv_count);
    int8_t *wv = (int8_t *)aligned_zero(kv_count);
    float *sq = (float *)aligned_zero((size_t)q_dim * sizeof(float));
    float *sk = (float *)aligned_zero((size_t)kv_dim * sizeof(float));
    float *sv = (float *)aligned_zero((size_t)kv_dim * sizeof(float));
    float *x = (float *)aligned_zero((size_t)in_dim * B * sizeof(float));
    float *q = (float *)aligned_zero(q_out * sizeof(float));
    float *k = (float *)aligned_zero(kv_out * sizeof(float));
    float *v = (float *)aligned_zero(kv_out * sizeof(float));
    float *q_ref = (float *)aligned_zero(q_out * sizeof(float));
    float *k_ref = (float *)aligned_zero(kv_out * sizeof(float));
    float *v_ref = (float *)aligned_zero(kv_out * sizeof(float));
    if (!wq || !wk || !wv || !sq || !sk || !sv || !x || !q || !k || !v ||
        !q_ref || !k_ref || !v_ref) {
        fprintf(stderr, "allocation failed for %s B=%d\n", name, B);
        free(wq); free(wk); free(wv); free(sq); free(sk); free(sv); free(x);
        free(q); free(k); free(v); free(q_ref); free(k_ref); free(v_ref);
        return 2;
    }
    for (size_t i = 0; i < q_count; i++) wq[i] = (int8_t)(random_value() * 127.0f);
    for (size_t i = 0; i < kv_count; i++) {
        wk[i] = (int8_t)(random_value() * 127.0f);
        wv[i] = (int8_t)(random_value() * 127.0f);
    }
    for (int i = 0; i < q_dim; i++) sq[i] = 0.002f + 0.001f * fabsf(random_value());
    for (int i = 0; i < kv_dim; i++) {
        sk[i] = 0.002f + 0.001f * fabsf(random_value());
        sv[i] = 0.002f + 0.001f * fabsf(random_value());
    }
    for (int i = 0; i < in_dim * B; i++) x[i] = random_value();

    qwen_matmat_int8(q_ref, wq, sq, x, q_dim, in_dim, B);
    qwen_matmat_int8(k_ref, wk, sk, x, kv_dim, in_dim, B);
    qwen_matmat_int8(v_ref, wv, sv, x, kv_dim, in_dim, B);
    int native = qwen_matmat_int8_qkv(q, k, v, wq, sq, wk, sk, wv, sv,
                                      x, in_dim, q_dim, kv_dim, B);
    if (!native) {
        printf("%s B=%d path=not-selected\n", name, B);
        free(wq); free(wk); free(wv); free(sq); free(sk); free(sv); free(x);
        free(q); free(k); free(v); free(q_ref); free(k_ref); free(v_ref);
        return 0;
    }
    double err = max_abs_diff(q, q_ref, q_out);
    double err_k = max_abs_diff(k, k_ref, kv_out);
    double err_v = max_abs_diff(v, v_ref, kv_out);
    if (err > 1e-5 || err_k > 1e-5 || err_v > 1e-5) {
        fprintf(stderr, "FAIL %s B=%d: max_abs q/k/v %.3e %.3e %.3e\n",
                name, B, err, err_k, err_v);
        free(wq); free(wk); free(wv); free(sq); free(sk); free(sv); free(x);
        free(q); free(k); free(v); free(q_ref); free(k_ref); free(v_ref);
        return 1;
    }

    qwen_matmat_int8(q_ref, wq, sq, x, q_dim, in_dim, B);
    qwen_matmat_int8(k_ref, wk, sk, x, kv_dim, in_dim, B);
    qwen_matmat_int8(v_ref, wv, sv, x, kv_dim, in_dim, B);
    qwen_matmat_int8_qkv(q, k, v, wq, sq, wk, sk, wv, sv,
                         x, in_dim, q_dim, kv_dim, B);
    double direct_samples[64], combined_samples[64];
    if (reps > (int)(sizeof direct_samples / sizeof direct_samples[0])) reps = 64;
    for (int r = 0; r < reps; r++) {
        double t0 = now_ns();
        qwen_matmat_int8(q_ref, wq, sq, x, q_dim, in_dim, B);
        qwen_matmat_int8(k_ref, wk, sk, x, kv_dim, in_dim, B);
        qwen_matmat_int8(v_ref, wv, sv, x, kv_dim, in_dim, B);
        direct_samples[r] = now_ns() - t0;
        t0 = now_ns();
        qwen_matmat_int8_qkv(q, k, v, wq, sq, wk, sk, wv, sv,
                             x, in_dim, q_dim, kv_dim, B);
        combined_samples[r] = now_ns() - t0;
    }
    double direct_ns = median(direct_samples, reps);
    double combined_ns = median(combined_samples, reps);
    printf("%s B=%d direct_ns=%.0f qkv_ns=%.0f speedup=%.3fx max_abs=%.3e/%.3e/%.3e\n",
           name, B, direct_ns, combined_ns, direct_ns / combined_ns,
           err, err_k, err_v);

    free(wq); free(wk); free(wv); free(sq); free(sk); free(sv); free(x);
    free(q); free(k); free(v); free(q_ref); free(k_ref); free(v_ref);
    return 0;
}

#endif

int main(int argc, char **argv) {
    int threads = argc > 1 ? atoi(argv[1]) : 4;
    int reps = argc > 2 ? atoi(argv[2]) : 15;
    if (threads < 1) threads = 1;
    if (reps < 3) reps = 3;
    qwen_set_threads(threads);
    printf("x86-qkv-bench threads=%d reps=%d mode=%s\n", threads, reps,
           getenv("QWEN_NO_VNNI_QKV") && getenv("QWEN_NO_VNNI_QKV")[0] == '1'
               ? "fallback-3calls" : "shared-activation");
#if !defined(__AVX512VNNI__)
    printf("SKIP: binary was not built with AVX-512 VNNI\n");
    return 0;
#else
    int rc = 0;
    rc |= run_case("talker", 2048, 2048, 1024, reps);
    rc |= run_case("code_predictor", 1024, 2048, 1024, reps);
    for (int B = 2; B <= 8; B *= 2) {
        rc |= run_matmat_case("talker_matmat", 2048, 2048, 1024, B, reps);
        rc |= run_matmat_case("code_predictor_matmat", 1024, 2048, 1024, B, reps);
    }
    return rc;
#endif
}
