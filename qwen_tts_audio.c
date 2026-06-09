/* qwen_tts_audio.c - WAV writer + post-processing (gain, time-stretch) */
#include "qwen_tts.h"
#include "qwen_tts_audio.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* --volume: linear PCM gain, in place. Soft-clamp to [-1,1] (the WAV writer
 * clamps too, but doing it here keeps any downstream consumer in range). */
void qwen_audio_apply_gain(float *samples, int n_samples, float gain) {
    if (gain == 1.0f || !samples) return;
    for (int i = 0; i < n_samples; i++) {
        float v = samples[i] * gain;
        if (v < -1.0f) v = -1.0f;
        if (v >  1.0f) v =  1.0f;
        samples[i] = v;
    }
}

/* --rate: pitch-preserving time-stretch via WSOLA (Waveform Similarity
 * Overlap-Add). rate>1 = faster/shorter, rate<1 = slower/longer.
 *
 * Each synthesis frame is Hann-windowed and overlap-added at a fixed synthesis
 * hop Hs; the next analysis segment is chosen (within ±tol of the nominal
 * position rate*Hs ahead) to best continue the previous frame, so periodicity —
 * and thus pitch — is preserved while duration changes. */
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

    int N = 1024;                 /* analysis/synthesis frame (~43ms @ 24kHz) */
    if (N > n_in) N = n_in;       /* tiny-input guard */
    if (N < 16) {                 /* too small to stretch meaningfully -> copy */
        float *o = (float *)malloc((size_t)n_in * sizeof(float));
        if (!o) return -1;
        memcpy(o, in, (size_t)n_in * sizeof(float));
        *out = o; *out_n = n_in;
        return 0;
    }
    int Hs  = N / 2;              /* synthesis hop (50% overlap) */
    int tol = N / 4;              /* similarity search radius */

    float *win = (float *)malloc((size_t)N * sizeof(float));
    if (!win) return -1;
    for (int i = 0; i < N; i++)
        win[i] = 0.5f * (1.0f - cosf(2.0f * (float)M_PI * i / (float)(N - 1)));

    int cap = (int)((double)n_in / rate) + 2 * N + 16;
    float *o    = (float *)calloc((size_t)cap, sizeof(float));
    float *wsum = (float *)calloc((size_t)cap, sizeof(float));
    if (!o || !wsum) { free(win); free(o); free(wsum); return -1; }

    double Ha = (double)Hs * rate;   /* nominal analysis hop */
    int    xs = 0;                   /* current analysis read position */
    int    oy = 0;                   /* output write position */
    double nominal = 0.0;            /* float nominal analysis position */

    while (xs + N <= n_in && oy + N <= cap) {
        /* overlap-add the current windowed frame */
        for (int i = 0; i < N; i++) { o[oy + i] += in[xs + i] * win[i]; wsum[oy + i] += win[i]; }

        /* template = the segment that naturally continues this frame */
        int tstart = xs + Hs;
        nominal += Ha;
        int center = (int)(nominal + 0.5);

        int next;
        if (tstart + N > n_in) {
            next = center;                       /* near end: no template, take nominal */
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
                double score = num / (sqrt(en) + 1e-9);   /* normalized similarity */
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
    /* Linear fade in/out to kill boundary transients (asymmetric). Fade-in 5ms removes
     * micro edge-clicks; fade-out 40ms smooths the ICL closing "tic" — the causal ConvNet
     * decoder's last frame ends mid-energy at the reference->target boundary, whereas the
     * non-ICL/qvoice path trails into silence. Both are harmless on normal trailing silence. */
    int fade_in  = sample_rate / 200;        /* 5 ms */
    int fade_out = sample_rate * 40 / 1000;  /* 40 ms (ear-tuned on ICL closing transient) */
    if (fade_in  > n_samples / 2) fade_in  = n_samples / 2;
    if (fade_out > n_samples / 2) fade_out = n_samples / 2;
    for (int i = 0; i < n_samples; i++) {
        float s = samples[i];
        if (fade_in  > 0 && i < fade_in)                 s *= (float)i / (float)fade_in;
        if (fade_out > 0 && i >= n_samples - fade_out)   s *= (float)(n_samples - 1 - i) / (float)fade_out;
        if (s < -1) s = -1; if (s > 1) s = 1;
        int16_t sample = (int16_t)(s * 32767);
        fwrite(&sample, 2, 1, f);
    }
    fclose(f);
    return 0;
}
