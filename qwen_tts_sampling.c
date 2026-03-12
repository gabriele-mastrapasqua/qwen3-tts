/*
 * qwen_tts_sampling.c - Sampling utilities
 */

#include "qwen_tts.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Pre-allocated work buffers (avoids malloc per sample call) */
static float *g_topk_tmp = NULL;
static int   *g_topp_idx = NULL;
static int    g_work_cap = 0;

static void ensure_work_buffers(int n) {
    if (n <= g_work_cap) return;
    free(g_topk_tmp); free(g_topp_idx);
    g_topk_tmp = (float *)malloc(n * sizeof(float));
    g_topp_idx = (int *)malloc(n * sizeof(int));
    g_work_cap = n;
}

/* Simple LCG random number generator */
static uint32_t g_seed = 12345;
static float rand_uniform(void) {
    g_seed = g_seed * 1103515245 + 12345;
    return (float)((g_seed >> 16) & 0x7FFF) / 32768.0f;
}

void qwen_set_seed(uint32_t seed) { g_seed = seed; }

/* Softmax with temperature */
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

/* Top-k filtering */
static int topk_filter(float *logits, int n, int k) {
    if (k <= 0 || k >= n) return n;
    
    /* Find k-th largest */
    float *tmp = g_topk_tmp;
    memcpy(tmp, logits, n * sizeof(float));

    /* Simple selection */
    for (int i = 0; i < k; i++) {
        int max_idx = i;
        for (int j = i + 1; j < n; j++)
            if (tmp[j] > tmp[max_idx]) max_idx = j;
        float t = tmp[i]; tmp[i] = tmp[max_idx]; tmp[max_idx] = t;
    }
    float threshold = tmp[k - 1];
    
    /* Zero out below threshold */
    int count = 0;
    for (int i = 0; i < n; i++) {
        if (logits[i] < threshold) logits[i] = 0;
        else count++;
    }
    return count;
}

/* Top-p (nucleus) filtering */
static int topp_filter(float *logits, int n, float p) {
    if (p >= 1.0f) return n;
    
    /* Sort indices by probability */
    int *idx = g_topp_idx;
    for (int i = 0; i < n; i++) idx[i] = i;

    for (int i = 0; i < n - 1; i++) {
        int max_idx = i;
        for (int j = i + 1; j < n; j++)
            if (logits[idx[j]] > logits[idx[max_idx]]) max_idx = j;
        int t = idx[i]; idx[i] = idx[max_idx]; idx[max_idx] = t;
    }

    /* Find cutoff */
    float cumsum = 0;
    int cutoff = n;
    for (int i = 0; i < n; i++) {
        cumsum += logits[idx[i]];
        if (cumsum > p) { cutoff = i + 1; break; }
    }
    
    /* Zero out beyond cutoff */
    int count = 0;
    for (int i = 0; i < n; i++) {
        if (i < cutoff) count++;
        else logits[i] = 0;
    }
    return count;
}

/* Sample from probability distribution */
static int sample_from_probs(float *probs, int n) {
    float r = rand_uniform();
    float cumsum = 0;
    for (int i = 0; i < n; i++) {
        cumsum += probs[i];
        if (r < cumsum) return i;
    }
    return n - 1;
}

/* Main sampling function */
int qwen_tts_sample(float *logits, int vocab_size, float temp, int top_k, float top_p,
                    float rep_penalty, int *prev_tokens, int n_prev) {
    ensure_work_buffers(vocab_size);

    /* Apply repetition penalty */
    if (rep_penalty != 1.0f && n_prev > 0) {
        for (int i = 0; i < n_prev; i++) {
            int tok = prev_tokens[i];
            if (tok >= 0 && tok < vocab_size) {
                if (logits[tok] > 0) logits[tok] /= rep_penalty;
                else logits[tok] *= rep_penalty;
            }
        }
    }
    
    /* Temperature scaling */
    if (temp < 1e-6f) {
        /* Greedy */
        int best = 0; float best_v = logits[0];
        for (int i = 1; i < vocab_size; i++)
            if (logits[i] > best_v) { best_v = logits[i]; best = i; }
        return best;
    }
    
    /* Softmax */
    softmax(logits, vocab_size, temp);
    
    /* Top-k */
    if (top_k > 0 && top_k < vocab_size)
        topk_filter(logits, vocab_size, top_k);
    
    /* Top-p */
    if (top_p < 1.0f && top_p > 0.0f)
        topp_filter(logits, vocab_size, top_p);
    
    /* Renormalize */
    float sum = 0;
    for (int i = 0; i < vocab_size; i++) sum += logits[i];
    if (sum > 0) for (int i = 0; i < vocab_size; i++) logits[i] /= sum;
    
    /* Sample */
    return sample_from_probs(logits, vocab_size);
}
