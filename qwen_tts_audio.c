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
        if (s != s) s = 0;  /* leaks-audit #7: NaN guard (NaN passes both clamps → UB int16 cast) */
        if (s < -1) s = -1; if (s > 1) s = 1;
        int16_t sample = (int16_t)(s * 32767);
        fwrite(&sample, 2, 1, f);
    }
    fclose(f);
    return 0;
}

/* ---- Edge cleanup + glitch scoring (feat: onset-leveling / tail-trim / seed-audition) ----
 * All operate on the decoded float PCM in place / read-only; default-OFF in main.c so the
 * shipped golden output is bit-identical unless a flag opts in. */

/* Index of the first sample where a 5ms window's RMS rises above a silence floor (~-50 dB).
 * Skips the ~80ms of leading digital silence every render has, so an onset fade lands on the
 * REAL attack, not on silence. Returns 0 if nothing crosses the floor. */
int qwen_audio_first_onset(const float *s, int n, int sample_rate) {
    int win = sample_rate / 200; if (win < 1) win = 1;          /* 5 ms */
    const float floor_rms = 3.16e-3f;                            /* ~-50 dBFS */
    for (int i = 0; i + win <= n; i += win) {
        double e = 0.0;
        for (int k = 0; k < win; k++) { float v = s[i+k]; e += (double)v * v; }
        if (e / win > (double)floor_rms * floor_rms) return i;
    }
    return 0;
}

/* Apply a linear fade-in of fade_ms over the REAL onset (post leading-silence) to kill the
 * attack "CRR" transient that strong emotion steering produces. In place. fade_ms<=0 = no-op. */
void qwen_audio_onset_fade(float *s, int n, int sample_rate, int fade_ms) {
    if (fade_ms <= 0 || n <= 0) return;
    int onset = qwen_audio_first_onset(s, n, sample_rate);
    int f = sample_rate * fade_ms / 1000;
    if (f > (n - onset)) f = n - onset;
    for (int k = 0; k < f; k++) s[onset + k] *= (float)k / (float)f;
}

/* Heuristic 0..1 score of a DEGENERATE TAIL (the "metallic/electric" noise some seeds emit
 * after stopping the sentence early). Higher = worse. Read-only. Method: 10ms frames; find the
 * loud region (RMS > -45 dB); take the speech-body median zero-crossing rate over its first 70%;
 * flag a contiguous TRAILING run of loud frames whose ZCR exceeds 2.5x the body median (noise is
 * high-ZCR vs voiced speech). score = min(1, flagged_ms/300). out_trim_at (optional) = sample
 * index where the flagged tail begins (for trimming). */
float qwen_audio_tail_glitch_score(const float *s, int n, int sample_rate, int *out_trim_at) {
    if (out_trim_at) *out_trim_at = n;
    int fr = sample_rate / 100; if (fr < 1) fr = 1;             /* 10 ms */
    int nf = n / fr;
    if (nf < 8) return 0.0f;
    const float floor_rms = 5.62e-3f;                           /* ~-45 dBFS = "loud" gate */
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
    /* speech-body medians (ZCR + RMS) over the loud frames in the first 70% of the span */
    int body_end = first + (int)((last - first) * 0.7f);
    float zb[4096], rb[4096]; int m = 0;
    for (int i = first; i <= body_end && m < 4096; i++) if (loud[i]) { zb[m] = zcr[i]; rb[m] = rms[i]; m++; }
    if (m < 3) { free(zcr); free(rms); free(loud); return 0.0f; }
    for (int a = 1; a < m; a++) { float v=zb[a]; int b=a-1; while(b>=0&&zb[b]>v){zb[b+1]=zb[b];b--;} zb[b+1]=v; }
    for (int a = 1; a < m; a++) { float v=rb[a]; int b=a-1; while(b>=0&&rb[b]>v){rb[b+1]=rb[b];b--;} rb[b+1]=v; }
    /* A degenerate metallic tail is high-ZCR AND loud AND sustained — a clean fricative/breath
     * ending is high-ZCR but QUIET, a loud release is short. Gate on all three. */
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
    if (best_len < 8) return 0.0f;                              /* require >=80ms sustained noise */
    int flagged = (last + 1) - best_start;
    if (out_trim_at) *out_trim_at = best_start * fr;
    float ms = (float)flagged * (float)fr * 1000.0f / (float)sample_rate;
    float score = ms / 300.0f; if (score > 1.0f) score = 1.0f;
    return score;
}

/* Conservative tail-trim: if the degenerate-tail score is high enough, cut the flagged trailing
 * run (keeping a 15ms guard so a clean release isn't clipped). In place via *n. Returns samples
 * trimmed. Gated default-off in main.c. */
int qwen_audio_tail_trim(float *s, int *n, int sample_rate, float min_score) {
    int trim_at = *n;
    float sc = qwen_audio_tail_glitch_score(s, *n, sample_rate, &trim_at);
    if (sc < min_score || trim_at >= *n) return 0;
    int guard = sample_rate * 15 / 1000;                        /* keep 15ms of the release */
    int new_n = trim_at + guard; if (new_n > *n) new_n = *n;
    int trimmed = *n - new_n;
    (void)s;
    *n = new_n;
    return trimmed;
}
