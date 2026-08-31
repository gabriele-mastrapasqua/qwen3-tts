/* qwen_tts_sampling.c - Sampling utilities */

#include "qwen_tts.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static __thread float *g_topk_tmp = NULL;
static __thread int   *g_topp_idx = NULL;
static __thread int    g_work_cap = 0;

static void ensure_work_buffers(int n) {
    if (n <= g_work_cap) return;
    free(g_topk_tmp); free(g_topp_idx);
    g_topk_tmp = (float *)malloc(n * sizeof(float));
    g_topp_idx = (int *)malloc(n * sizeof(int));
    g_work_cap = n;
}

static __thread uint32_t g_seed = 12345;
static float rand_uniform(void) {
    g_seed = g_seed * 1103515245 + 12345;
    return (float)((g_seed >> 16) & 0x7FFF) / 32768.0f;
}

void qwen_set_seed(uint32_t seed) { g_seed = seed; }

uint32_t qwen_get_seed(void) { return g_seed; }

static void softmax(float *logits, int n, float temp) {
    float max_val = logits[0];
    for (int i = 1; i < n; i++) if (logits[i] > max_val) max_val = logits[i];
    float sum = 0;
    float inv_temp = 1.0f / temp;
    for (int i = 0; i < n; i++) {
        logits[i] = expf((logits[i] - max_val) * inv_temp);
        sum += logits[i];
    }
    for (int i = 0; i < n; i++) logits[i] /= sum;
}

static float quickselect_kth_largest(float *arr, int n, int k) {
    int lo = 0, hi = n - 1;
    while (lo < hi) {
        float pivot = arr[lo + (hi - lo) / 2];
        int i = lo, j = hi;
        int p = lo;
        while (i <= j) {
            if (arr[i] > pivot) {
                float t = arr[p]; arr[p] = arr[i]; arr[i] = t;
                p++; i++;
            } else if (arr[i] < pivot) {
                float t = arr[i]; arr[i] = arr[j]; arr[j] = t;
                j--;
            } else {
                i++;
            }
        }
        if (k - 1 < p) {
            hi = p - 1;
        } else if (k - 1 > j) {
            lo = j + 1;
        } else {
            return pivot;
        }
    }
    return arr[lo];
}

static int topk_filter(float *logits, int n, int k) {
    if (k <= 0 || k >= n) return n;

    float *tmp = g_topk_tmp;
    memcpy(tmp, logits, n * sizeof(float));
    float threshold = quickselect_kth_largest(tmp, n, k);

    int count = 0;
    for (int i = 0; i < n; i++) {
        if (logits[i] < threshold) logits[i] = 0;
        else count++;
    }
    return count;
}

static int topp_filter(float *logits, int n, float p) {
    if (p >= 1.0f) return n;

    int *idx = g_topp_idx;
    for (int i = 0; i < n; i++) idx[i] = i;

    float cumsum = 0;
    int cutoff = n;
    for (int i = 0; i < n; i++) {
        int max_idx = i;
        for (int j = i + 1; j < n; j++)
            if (logits[idx[j]] > logits[idx[max_idx]]) max_idx = j;
        int t = idx[i]; idx[i] = idx[max_idx]; idx[max_idx] = t;
        cumsum += logits[idx[i]];
        if (cumsum > p) { cutoff = i + 1; break; }
    }

    for (int i = cutoff; i < n; i++)
        logits[idx[i]] = 0.0f;
    return cutoff;
}

static int sample_from_probs(float *probs, int n) {
    float r = rand_uniform();
    float cumsum = 0;
    for (int i = 0; i < n; i++) {
        cumsum += probs[i];
        if (r < cumsum) return i;
    }
    return n - 1;
}

int qwen_tts_sample(float *logits, int vocab_size, float temp, int top_k, float top_p,
                    float rep_penalty, int *prev_tokens, int n_prev) {
    ensure_work_buffers(vocab_size);

    if (rep_penalty != 1.0f && n_prev > 0) {
        for (int i = 0; i < n_prev; i++) {
            int tok = prev_tokens[i];
            if (tok >= 0 && tok < vocab_size) {
                if (logits[tok] > 0) logits[tok] /= rep_penalty;
                else logits[tok] *= rep_penalty;
            }
        }
    }

    if (temp < 1e-6f) {
        int best = 0; float best_v = logits[0];
        for (int i = 1; i < vocab_size; i++)
            if (logits[i] > best_v) { best_v = logits[i]; best = i; }
        return best;
    }

    softmax(logits, vocab_size, temp);

    if (top_k > 0 && top_k < vocab_size)
        topk_filter(logits, vocab_size, top_k);

    if (top_p < 1.0f && top_p > 0.0f)
        topp_filter(logits, vocab_size, top_p);

    float sum = 0;
    for (int i = 0; i < vocab_size; i++) sum += logits[i];
    if (sum > 0) for (int i = 0; i < vocab_size; i++) logits[i] /= sum;

    return sample_from_probs(logits, vocab_size);
}
