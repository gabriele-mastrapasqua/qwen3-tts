/* qwen_tts_audio.c - WAV writer + post-processing (gain, time-stretch) */
#include "qwen_tts.h"
#include "qwen_tts_audio.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

void qwen_audio_apply_gain(float *samples, int n_samples, float gain) {
    if (gain == 1.0f || !samples) return;
    for (int i = 0; i < n_samples; i++) {
        float v = samples[i] * gain;
        if (v < -1.0f) v = -1.0f;
        if (v >  1.0f) v =  1.0f;
        samples[i] = v;
    }
}

int qwen_audio_time_stretch(const float *in, int n_in, float rate, int sample_rate,
                            float **out, int *out_n) {
    (void)sample_rate;
    if (rate <= 0.0f) rate = 1.0f;
    if (rate == 1.0f || n_in <= 0) {
        float *o = (float *)malloc((size_t)(n_in > 0 ? n_in : 1) * sizeof(float));
        if (!o) return -1;
        if (n_in > 0) memcpy(o, in, (size_t)n_in * sizeof(float));
        *out = o; *out_n = n_in > 0 ? n_in : 0;
        return 0;
    }

    int N = 1024;
    if (N > n_in) N = n_in;
    if (N < 16) {
        float *o = (float *)malloc((size_t)n_in * sizeof(float));
        if (!o) return -1;
        memcpy(o, in, (size_t)n_in * sizeof(float));
        *out = o; *out_n = n_in;
        return 0;
    }
    int Hs  = N / 2;
    int tol = N / 4;

    float *win = (float *)malloc((size_t)N * sizeof(float));
    if (!win) return -1;
    for (int i = 0; i < N; i++)
        win[i] = 0.5f * (1.0f - cosf(2.0f * (float)M_PI * i / (float)(N - 1)));

    int cap = (int)((double)n_in / rate) + 2 * N + 16;
    float *o    = (float *)calloc((size_t)cap, sizeof(float));
    float *wsum = (float *)calloc((size_t)cap, sizeof(float));
    if (!o || !wsum) { free(win); free(o); free(wsum); return -1; }

    double Ha = (double)Hs * rate;
    int    xs = 0;
    int    oy = 0;
    double nominal = 0.0;

    while (xs + N <= n_in && oy + N <= cap) {
        for (int i = 0; i < N; i++) { o[oy + i] += in[xs + i] * win[i]; wsum[oy + i] += win[i]; }

        int tstart = xs + Hs;
        nominal += Ha;
        int center = (int)(nominal + 0.5);

        int next;
        if (tstart + N > n_in) {
            next = center;
        } else {
            int best_d = 0; double best = -1e300;
            for (int d = -tol; d <= tol; d++) {
                int s = center + d;
                if (s < 0 || s + N > n_in) continue;
                double num = 0.0, en = 0.0;
                for (int i = 0; i < N; i++) {
                    double a = in[s + i];
                    num += a * in[tstart + i];
                    en  += a * a;
                }
                double score = num / (sqrt(en) + 1e-9);
                if (score > best) { best = score; best_d = d; }
            }
            next = center + best_d;
        }
        if (next < 0) next = 0;
        if (next + N > n_in) break;
        xs  = next;
        oy += Hs;
    }

    int total = oy + N;
    if (total > cap) total = cap;
    for (int i = 0; i < total; i++) if (wsum[i] > 1e-6f) o[i] /= wsum[i];

    free(win); free(wsum);
    *out = o; *out_n = total;
    return 0;
}

int qwen_tts_write_wav(const char *path, const float *samples, int n_samples, int sample_rate) {
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    int bits = 16, channels = 1;
    int data_size = n_samples * channels * (bits/8);
    int file_size = 36 + data_size;
    int byte_rate = sample_rate * channels * (bits/8);
    short block_align = channels * (bits/8);
    fwrite("RIFF", 1, 4, f);
    fwrite(&file_size, 4, 1, f);
    fwrite("WAVEfmt ", 1, 8, f);
    int fmt_size = 16; short audio_fmt = 1;
    fwrite(&fmt_size, 4, 1, f);
    fwrite(&audio_fmt, 2, 1, f);
    fwrite(&channels, 2, 1, f);
    fwrite(&sample_rate, 4, 1, f);
    fwrite(&byte_rate, 4, 1, f);
    fwrite(&block_align, 2, 1, f);
    fwrite(&bits, 2, 1, f);
    fwrite("data", 1, 4, f);
    fwrite(&data_size, 4, 1, f);
    int fade_in  = sample_rate / 200;
    int fade_out = sample_rate * 40 / 1000;
    if (fade_in  > n_samples / 2) fade_in  = n_samples / 2;
    if (fade_out > n_samples / 2) fade_out = n_samples / 2;
    for (int i = 0; i < n_samples; i++) {
        float s = samples[i];
        if (fade_in  > 0 && i < fade_in)                 s *= (float)i / (float)fade_in;
        if (fade_out > 0 && i >= n_samples - fade_out)   s *= (float)(n_samples - 1 - i) / (float)fade_out;
        if (s != s) s = 0;
        if (s < -1) s = -1; if (s > 1) s = 1;
        int16_t sample = (int16_t)(s * 32767);
        fwrite(&sample, 2, 1, f);
    }
    int werr = ferror(f);
    if (fclose(f) != 0 || werr) return -1;
    return 0;
}

int qwen_audio_first_onset(const float *s, int n, int sample_rate) {
    int win = sample_rate / 200; if (win < 1) win = 1;
    const float floor_rms = 3.16e-3f;
    for (int i = 0; i + win <= n; i += win) {
        double e = 0.0;
        for (int k = 0; k < win; k++) { float v = s[i+k]; e += (double)v * v; }
        if (e / win > (double)floor_rms * floor_rms) return i;
    }
    return 0;
}

void qwen_audio_onset_fade(float *s, int n, int sample_rate, int fade_ms) {
    if (fade_ms <= 0 || n <= 0) return;
    int onset = qwen_audio_first_onset(s, n, sample_rate);
    int f = sample_rate * fade_ms / 1000;
    if (f > (n - onset)) f = n - onset;
    for (int k = 0; k < f; k++) s[onset + k] *= (float)k / (float)f;
}

float qwen_audio_tail_glitch_score(const float *s, int n, int sample_rate, int *out_trim_at) {
    if (out_trim_at) *out_trim_at = n;
    int fr = sample_rate / 100; if (fr < 1) fr = 1;
    int nf = n / fr;
    if (nf < 8) return 0.0f;
    const float floor_rms = 5.62e-3f;
    float *zcr = (float *)malloc((size_t)nf * sizeof(float));
    float *rms = (float *)malloc((size_t)nf * sizeof(float));
    char  *loud = (char *)malloc((size_t)nf);
    if (!zcr || !rms || !loud) { free(zcr); free(rms); free(loud); return 0.0f; }
    for (int i = 0; i < nf; i++) {
        const float *b = s + (size_t)i * fr;
        double e = 0.0; int zc = 0;
        for (int k = 0; k < fr; k++) { e += (double)b[k]*b[k]; if (k && ((b[k]>=0)!=(b[k-1]>=0))) zc++; }
        rms[i]  = (float)sqrt(e / fr);
        loud[i] = (rms[i] > floor_rms) ? 1 : 0;
        zcr[i]  = (float)zc / (float)fr;
    }
    int last = -1; for (int i = nf - 1; i >= 0; i--) if (loud[i]) { last = i; break; }
    int first = -1; for (int i = 0; i < nf; i++) if (loud[i]) { first = i; break; }
    if (last < 0 || first < 0 || last - first < 6) { free(zcr); free(rms); free(loud); return 0.0f; }
    int body_end = first + (int)((last - first) * 0.7f);
    float zb[4096], rb[4096]; int m = 0;
    for (int i = first; i <= body_end && m < 4096; i++) if (loud[i]) { zb[m] = zcr[i]; rb[m] = rms[i]; m++; }
    if (m < 3) { free(zcr); free(rms); free(loud); return 0.0f; }
    for (int a = 1; a < m; a++) { float v=zb[a]; int b=a-1; while(b>=0&&zb[b]>v){zb[b+1]=zb[b];b--;} zb[b+1]=v; }
    for (int a = 1; a < m; a++) { float v=rb[a]; int b=a-1; while(b>=0&&rb[b]>v){rb[b+1]=rb[b];b--;} rb[b+1]=v; }
    float zthr = zb[m/2] * 2.5f;  if (zthr < 0.20f) zthr = 0.20f;
    float ethr = rb[m/2] * 0.6f;  if (ethr < 0.10f) ethr = 0.10f;
    int run = 0, cur = -1, best_len = 0, best_start = -1;
    for (int i = body_end + 1; i <= last; i++) {
        if (loud[i] && zcr[i] > zthr && rms[i] > ethr) {
            if (run == 0) cur = i;
            run++;
            if (run > best_len) { best_len = run; best_start = cur; }
        } else run = 0;
    }
    free(zcr); free(rms); free(loud);
    if (best_len < 8) return 0.0f;
    int flagged = (last + 1) - best_start;
    if (out_trim_at) *out_trim_at = best_start * fr;
    float ms = (float)flagged * (float)fr * 1000.0f / (float)sample_rate;
    float score = ms / 300.0f; if (score > 1.0f) score = 1.0f;
    return score;
}

int qwen_audio_tail_trim(float *s, int *n, int sample_rate, float min_score) {
    int trim_at = *n;
    float sc = qwen_audio_tail_glitch_score(s, *n, sample_rate, &trim_at);
    if (sc < min_score || trim_at >= *n) return 0;
    int guard = sample_rate * 15 / 1000;
    int new_n = trim_at + guard; if (new_n > *n) new_n = *n;
    int trimmed = *n - new_n;
    (void)s;
    *n = new_n;
    return trimmed;
}
