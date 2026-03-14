/*
 * qwen_tts_voice_clone.c - Voice cloning support for Qwen3-TTS Base models
 *
 * Implements:
 * - WAV file reader
 * - Mel spectrogram (STFT + mel filterbank + log compression)
 * - ECAPA-TDNN speaker encoder
 */

#include "qwen_tts.h"
#include "qwen_tts_voice_clone.h"
#include "qwen_tts_safetensors.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ════════════════════════════════════════════════════════════════════════
 * WAV Reader
 * ════════════════════════════════════════════════════════════════════════ */

int qwen_read_wav(const char *path, float **out_samples, int *out_n_samples, int *out_sample_rate) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "Error: cannot open WAV file: %s\n", path);
        return -1;
    }

    /* Read RIFF header */
    char riff[4];
    if (fread(riff, 1, 4, f) != 4 || memcmp(riff, "RIFF", 4) != 0) {
        fprintf(stderr, "Error: not a WAV file (no RIFF header): %s\n", path);
        fclose(f); return -1;
    }

    uint32_t file_size;
    fread(&file_size, 4, 1, f);

    char wave[4];
    if (fread(wave, 1, 4, f) != 4 || memcmp(wave, "WAVE", 4) != 0) {
        fprintf(stderr, "Error: not a WAV file (no WAVE marker): %s\n", path);
        fclose(f); return -1;
    }

    /* Parse chunks */
    int sample_rate = 0, bits_per_sample = 0, num_channels = 0;
    int16_t audio_format = 0;
    uint8_t *raw_data = NULL;
    uint32_t data_size = 0;

    while (!feof(f)) {
        char chunk_id[4];
        uint32_t chunk_size;
        if (fread(chunk_id, 1, 4, f) != 4) break;
        if (fread(&chunk_size, 4, 1, f) != 1) break;

        if (memcmp(chunk_id, "fmt ", 4) == 0) {
            fread(&audio_format, 2, 1, f);
            int16_t ch;
            fread(&ch, 2, 1, f);
            num_channels = ch;
            uint32_t sr;
            fread(&sr, 4, 1, f);
            sample_rate = (int)sr;
            uint32_t byte_rate;
            fread(&byte_rate, 4, 1, f);
            int16_t block_align;
            fread(&block_align, 2, 1, f);
            int16_t bps;
            fread(&bps, 2, 1, f);
            bits_per_sample = bps;
            /* Skip extra fmt bytes */
            if (chunk_size > 16) {
                fseek(f, chunk_size - 16, SEEK_CUR);
            }
        } else if (memcmp(chunk_id, "data", 4) == 0) {
            data_size = chunk_size;
            raw_data = (uint8_t *)malloc(data_size);
            if (fread(raw_data, 1, data_size, f) != data_size) {
                fprintf(stderr, "Warning: incomplete data chunk in %s\n", path);
            }
            break;  /* got data, done */
        } else {
            /* Skip unknown chunks */
            fseek(f, chunk_size, SEEK_CUR);
        }
    }
    fclose(f);

    if (!raw_data || data_size == 0) {
        fprintf(stderr, "Error: no data chunk in WAV file: %s\n", path);
        free(raw_data);
        return -1;
    }

    if (audio_format != 1) {
        fprintf(stderr, "Error: unsupported WAV format %d (only PCM=1 supported): %s\n", audio_format, path);
        free(raw_data);
        return -1;
    }

    if (bits_per_sample != 16 && bits_per_sample != 32) {
        fprintf(stderr, "Error: unsupported bits per sample %d (only 16/32 supported): %s\n", bits_per_sample, path);
        free(raw_data);
        return -1;
    }

    /* Convert to float32 mono */
    int bytes_per_sample = bits_per_sample / 8;
    int total_samples = (int)(data_size / (bytes_per_sample * num_channels));
    float *samples = (float *)malloc(total_samples * sizeof(float));

    for (int i = 0; i < total_samples; i++) {
        float val = 0;
        if (num_channels == 1) {
            if (bits_per_sample == 16) {
                int16_t s;
                memcpy(&s, raw_data + i * 2, 2);
                val = s / 32768.0f;
            } else {
                int32_t s;
                memcpy(&s, raw_data + i * 4, 4);
                val = s / 2147483648.0f;
            }
        } else {
            /* Stereo → mono: average channels */
            float sum = 0;
            for (int c = 0; c < num_channels; c++) {
                int offset = (i * num_channels + c) * bytes_per_sample;
                if (bits_per_sample == 16) {
                    int16_t s;
                    memcpy(&s, raw_data + offset, 2);
                    sum += s / 32768.0f;
                } else {
                    int32_t s;
                    memcpy(&s, raw_data + offset, 4);
                    sum += s / 2147483648.0f;
                }
            }
            val = sum / num_channels;
        }
        samples[i] = val;
    }

    free(raw_data);
    *out_samples = samples;
    *out_n_samples = total_samples;
    *out_sample_rate = sample_rate;
    return 0;
}


/* ════════════════════════════════════════════════════════════════════════
 * FFT (radix-2 Cooley-Tukey)
 * ════════════════════════════════════════════════════════════════════════ */

static void fft_radix2(float *re, float *im, int n) {
    /* Bit-reversal permutation */
    int log2n = 0;
    for (int tmp = n; tmp > 1; tmp >>= 1) log2n++;

    for (int i = 0; i < n; i++) {
        int j = 0;
        for (int b = 0; b < log2n; b++)
            if (i & (1 << b)) j |= (1 << (log2n - 1 - b));
        if (j > i) {
            float tr = re[i]; re[i] = re[j]; re[j] = tr;
            float ti = im[i]; im[i] = im[j]; im[j] = ti;
        }
    }

    /* Butterfly stages */
    for (int s = 1; s <= log2n; s++) {
        int m = 1 << s;
        int half = m >> 1;
        float angle = -2.0f * (float)M_PI / m;
        float wm_re = cosf(angle), wm_im = sinf(angle);

        for (int k = 0; k < n; k += m) {
            float w_re = 1.0f, w_im = 0.0f;
            for (int j = 0; j < half; j++) {
                int idx1 = k + j;
                int idx2 = k + j + half;
                float t_re = w_re * re[idx2] - w_im * im[idx2];
                float t_im = w_re * im[idx2] + w_im * re[idx2];
                re[idx2] = re[idx1] - t_re;
                im[idx2] = im[idx1] - t_im;
                re[idx1] += t_re;
                im[idx1] += t_im;
                float new_w_re = w_re * wm_re - w_im * wm_im;
                w_im = w_re * wm_im + w_im * wm_re;
                w_re = new_w_re;
            }
        }
    }
}


/* ════════════════════════════════════════════════════════════════════════
 * Mel Spectrogram
 * ════════════════════════════════════════════════════════════════════════ */

/* Convert frequency to mel scale (HTK formula) */
static float hz_to_mel_slaney(float hz) {
    /* Slaney's Auditory Toolbox formula (linear below 1000 Hz, log above) */
    float f_sp = 200.0f / 3.0f;  /* 66.667 Hz */
    if (hz < 1000.0f) return hz / f_sp;
    float min_log_mel = 1000.0f / f_sp;  /* 15.0 */
    float logstep = logf(6.4f) / 27.0f;
    return min_log_mel + logf(hz / 1000.0f) / logstep;
}

static float mel_to_hz_slaney(float mel) {
    float f_sp = 200.0f / 3.0f;
    float min_log_mel = 1000.0f / f_sp;
    float logstep = logf(6.4f) / 27.0f;
    if (mel < min_log_mel) return mel * f_sp;
    return 1000.0f * expf((mel - min_log_mel) * logstep);
}

/* Build mel filterbank: [n_mels, n_fft/2+1] */
static float *build_mel_filterbank(int sr, int n_fft, int n_mels, float fmin, float fmax) {
    int n_freqs = n_fft / 2 + 1;
    float *fb = (float *)calloc((size_t)n_mels * n_freqs, sizeof(float));

    /* Mel scale points */
    float mel_min = hz_to_mel_slaney(fmin);
    float mel_max = hz_to_mel_slaney(fmax);
    float *mels = (float *)malloc((n_mels + 2) * sizeof(float));
    for (int i = 0; i < n_mels + 2; i++) {
        float m = mel_min + (mel_max - mel_min) * i / (n_mels + 1);
        mels[i] = mel_to_hz_slaney(m);
    }

    float freq_step = (float)sr / n_fft;

    /* Slaney normalization: each filter has area = 1 */
    for (int i = 0; i < n_mels; i++) {
        float f_low = mels[i];
        float f_center = mels[i + 1];
        float f_high = mels[i + 2];
        float enorm = 2.0f / (f_high - f_low);

        for (int j = 0; j < n_freqs; j++) {
            float f = j * freq_step;
            float w = 0.0f;
            if (f >= f_low && f <= f_center && f_center > f_low)
                w = (f - f_low) / (f_center - f_low);
            else if (f > f_center && f <= f_high && f_high > f_center)
                w = (f_high - f) / (f_high - f_center);
            fb[i * n_freqs + j] = w * enorm;
        }
    }

    free(mels);
    return fb;
}

int qwen_mel_spectrogram(const float *audio, int n_samples, int sample_rate,
                         float **out_mel, int *out_n_frames) {
    const int n_fft = 1024;
    const int n_mels = 128;
    const int hop_size = 256;
    const int win_size = 1024;
    const float fmin = 0.0f;
    const float fmax = 12000.0f;
    (void)sample_rate;  /* assumed 24000 */

    /* Reflect-pad the audio: padding = (n_fft - hop_size) / 2 = 384 */
    int padding = (n_fft - hop_size) / 2;
    int padded_len = n_samples + 2 * padding;
    float *padded = (float *)malloc(padded_len * sizeof(float));

    /* Reflect padding (not including boundary sample) */
    for (int i = 0; i < padding; i++)
        padded[i] = audio[padding - i];  /* reflect: audio[pad], audio[pad-1], ..., audio[1] */
    memcpy(padded + padding, audio, n_samples * sizeof(float));
    for (int i = 0; i < padding; i++) {
        int src = n_samples - 2 - i;
        if (src < 0) src = 0;
        padded[padding + n_samples + i] = audio[src];
    }

    /* Number of frames */
    int n_frames = (padded_len - n_fft) / hop_size + 1;
    if (n_frames <= 0) {
        free(padded);
        fprintf(stderr, "Error: audio too short for mel spectrogram\n");
        return -1;
    }

    /* Hann window */
    float *window = (float *)malloc(win_size * sizeof(float));
    for (int i = 0; i < win_size; i++)
        window[i] = 0.5f * (1.0f - cosf(2.0f * (float)M_PI * i / win_size));

    /* Mel filterbank */
    float *mel_fb = build_mel_filterbank(24000, n_fft, n_mels, fmin, fmax);
    int n_freqs = n_fft / 2 + 1;

    /* Output mel spectrogram [n_frames, n_mels] */
    float *mel = (float *)malloc((size_t)n_frames * n_mels * sizeof(float));

    /* FFT buffers */
    float *fft_re = (float *)malloc(n_fft * sizeof(float));
    float *fft_im = (float *)malloc(n_fft * sizeof(float));

    for (int f = 0; f < n_frames; f++) {
        int start = f * hop_size;

        /* Window the frame */
        for (int i = 0; i < n_fft; i++) {
            fft_re[i] = padded[start + i] * window[i];
            fft_im[i] = 0.0f;
        }

        /* FFT */
        fft_radix2(fft_re, fft_im, n_fft);

        /* Magnitude spectrum (only positive frequencies) */
        float spec[513];  /* n_fft/2+1 = 513 */
        for (int i = 0; i < n_freqs; i++) {
            spec[i] = sqrtf(fft_re[i] * fft_re[i] + fft_im[i] * fft_im[i] + 1e-9f);
        }

        /* Apply mel filterbank + log compression */
        for (int m = 0; m < n_mels; m++) {
            float val = 0.0f;
            for (int i = 0; i < n_freqs; i++)
                val += mel_fb[m * n_freqs + i] * spec[i];
            /* Dynamic range compression: log(clamp(x, 1e-5)) */
            if (val < 1e-5f) val = 1e-5f;
            mel[f * n_mels + m] = logf(val);
        }
    }

    free(fft_re);
    free(fft_im);
    free(window);
    free(mel_fb);
    free(padded);

    *out_mel = mel;
    *out_n_frames = n_frames;
    return 0;
}


/* ════════════════════════════════════════════════════════════════════════
 * ECAPA-TDNN Speaker Encoder
 * ════════════════════════════════════════════════════════════════════════ */

/* Helper: 1D convolution with "same" padding (reflect mode)
 * Input:  [in_ch, in_len]  (channel-first)
 * Weight: [out_ch, in_ch, kernel]
 * Bias:   [out_ch]
 * Output: [out_ch, in_len]
 * Uses dilation, reflect padding to achieve "same" output size. */
static void conv1d_same_reflect(
    const float *input, int in_ch, int in_len,
    const float *weight, const float *bias,
    int out_ch, int kernel, int dilation,
    float *output)
{
    int effective_k = 1 + (kernel - 1) * dilation;
    int pad = (effective_k - 1) / 2;
    int pad_left = pad;
    int pad_right = effective_k - 1 - pad_left;

    /* Build padded input with reflect padding */
    int padded_len = in_len + pad_left + pad_right;
    float *padded = (float *)malloc((size_t)in_ch * padded_len * sizeof(float));

    for (int c = 0; c < in_ch; c++) {
        float *dst = padded + (size_t)c * padded_len;
        const float *src = input + (size_t)c * in_len;

        /* Left reflect pad */
        for (int i = 0; i < pad_left; i++) {
            int idx = pad_left - i;
            if (idx >= in_len) idx = in_len - 1;
            dst[i] = src[idx];
        }
        /* Center */
        memcpy(dst + pad_left, src, in_len * sizeof(float));
        /* Right reflect pad */
        for (int i = 0; i < pad_right; i++) {
            int idx = in_len - 2 - i;
            if (idx < 0) idx = 0;
            dst[pad_left + in_len + i] = src[idx];
        }
    }

    /* Convolution */
    for (int oc = 0; oc < out_ch; oc++) {
        for (int t = 0; t < in_len; t++) {
            float val = bias ? bias[oc] : 0.0f;
            for (int ic = 0; ic < in_ch; ic++) {
                const float *w = weight + ((size_t)oc * in_ch + ic) * kernel;
                const float *p = padded + (size_t)ic * padded_len + t;
                for (int k = 0; k < kernel; k++) {
                    val += w[k] * p[k * dilation];
                }
            }
            output[(size_t)oc * in_len + t] = val;
        }
    }

    free(padded);
}

/* ReLU in-place */
static void relu_inplace(float *x, int n) {
    for (int i = 0; i < n; i++)
        if (x[i] < 0.0f) x[i] = 0.0f;
}

/* Sigmoid in-place (used by SE blocks inline, kept for future use) */
static inline float sigmoidf(float x) { return 1.0f / (1.0f + expf(-x)); }

/* TimeDelayNetBlock: Conv1d + ReLU */
static void tdnn_forward(
    const float *input, int in_ch, int in_len,
    const float *conv_w, const float *conv_b,
    int out_ch, int kernel, int dilation,
    float *output)
{
    conv1d_same_reflect(input, in_ch, in_len, conv_w, conv_b,
                        out_ch, kernel, dilation, output);
    relu_inplace(output, out_ch * in_len);
}

/* Res2NetBlock forward */
static void res2net_forward(
    const float *input, int channels, int in_len, int scale,
    float *res2net_w[7], float *res2net_b[7],
    int kernel, int dilation,
    float *output)
{
    int chunk_ch = channels / scale;  /* 512/8 = 64 */
    float *prev_output = (float *)malloc((size_t)chunk_ch * in_len * sizeof(float));
    float *block_input = (float *)malloc((size_t)chunk_ch * in_len * sizeof(float));
    float *block_output = (float *)malloc((size_t)chunk_ch * in_len * sizeof(float));

    for (int i = 0; i < scale; i++) {
        const float *chunk = input + (size_t)i * chunk_ch * in_len;

        if (i == 0) {
            /* Passthrough */
            memcpy(output + (size_t)i * chunk_ch * in_len, chunk,
                   (size_t)chunk_ch * in_len * sizeof(float));
            memcpy(prev_output, chunk, (size_t)chunk_ch * in_len * sizeof(float));
        } else if (i == 1) {
            tdnn_forward(chunk, chunk_ch, in_len,
                        res2net_w[i - 1], res2net_b[i - 1],
                        chunk_ch, kernel, dilation, block_output);
            memcpy(output + (size_t)i * chunk_ch * in_len, block_output,
                   (size_t)chunk_ch * in_len * sizeof(float));
            memcpy(prev_output, block_output, (size_t)chunk_ch * in_len * sizeof(float));
        } else {
            /* Add previous output to current chunk */
            for (int j = 0; j < chunk_ch * in_len; j++)
                block_input[j] = chunk[j] + prev_output[j];
            tdnn_forward(block_input, chunk_ch, in_len,
                        res2net_w[i - 1], res2net_b[i - 1],
                        chunk_ch, kernel, dilation, block_output);
            memcpy(output + (size_t)i * chunk_ch * in_len, block_output,
                   (size_t)chunk_ch * in_len * sizeof(float));
            memcpy(prev_output, block_output, (size_t)chunk_ch * in_len * sizeof(float));
        }
    }

    free(prev_output);
    free(block_input);
    free(block_output);
}

/* SE-Res2Net block forward */
static void se_res2net_block_forward(
    const float *input, int channels, int in_len,
    float *tdnn1_w, float *tdnn1_b,
    float *res2net_w[7], float *res2net_b[7],
    float *tdnn2_w, float *tdnn2_b,
    float *se_conv1_w, float *se_conv1_b,
    float *se_conv2_w, float *se_conv2_b,
    int kernel, int dilation, int scale,
    float *output)
{
    int n = channels * in_len;

    /* TDNN1 */
    float *h1 = (float *)malloc(n * sizeof(float));
    tdnn_forward(input, channels, in_len, tdnn1_w, tdnn1_b, channels, 1, 1, h1);

    /* Res2Net */
    float *h2 = (float *)malloc(n * sizeof(float));
    res2net_forward(h1, channels, in_len, scale, res2net_w, res2net_b, kernel, dilation, h2);
    free(h1);

    /* TDNN2 */
    float *h3 = (float *)malloc(n * sizeof(float));
    tdnn_forward(h2, channels, in_len, tdnn2_w, tdnn2_b, channels, 1, 1, h3);
    free(h2);

    /* Squeeze-Excitation */
    /* Mean across time */
    int se_ch = 128;  /* hardcoded from config */
    float *mean = (float *)calloc(channels, sizeof(float));
    for (int c = 0; c < channels; c++) {
        float sum = 0;
        for (int t = 0; t < in_len; t++)
            sum += h3[(size_t)c * in_len + t];
        mean[c] = sum / in_len;
    }

    /* SE conv1 (channels→se_ch) + ReLU */
    float *se1 = (float *)malloc(se_ch * sizeof(float));
    for (int i = 0; i < se_ch; i++) {
        float val = se_conv1_b[i];
        for (int c = 0; c < channels; c++)
            val += se_conv1_w[(size_t)i * channels] * mean[c];
        /* Note: kernel=1, so just dot product with the mean (which is length=1 per channel) */
        se1[i] = val;
    }
    /* Wait, the SE conv uses Conv1d on the mean which has length=1.
     * So it's just a matrix multiply: [se_ch, channels, 1] x [channels, 1] → [se_ch, 1] */
    for (int i = 0; i < se_ch; i++) {
        float val = se_conv1_b[i];
        for (int c = 0; c < channels; c++)
            val += se_conv1_w[i * channels + c] * mean[c];
        se1[i] = val > 0 ? val : 0;  /* ReLU */
    }

    /* SE conv2 (se_ch→channels) + Sigmoid */
    float *se2 = (float *)malloc(channels * sizeof(float));
    for (int i = 0; i < channels; i++) {
        float val = se_conv2_b[i];
        for (int c = 0; c < se_ch; c++)
            val += se_conv2_w[i * se_ch + c] * se1[c];
        se2[i] = sigmoidf(val);
    }

    free(mean);
    free(se1);

    /* Apply SE: h3 * se2 (broadcast over time) + residual */
    for (int c = 0; c < channels; c++) {
        float scale_val = se2[c];
        for (int t = 0; t < in_len; t++) {
            size_t idx = (size_t)c * in_len + t;
            output[idx] = h3[idx] * scale_val + input[idx];
        }
    }

    free(se2);
    free(h3);
}


/* Helper: load f32 tensor from safetensors by name */
static float *load_f32_tensor(multi_safetensors_t *ms, const char *name) {
    safetensors_file_t *sf = NULL;
    const safetensor_t *t = multi_safetensors_find(ms, name, &sf);
    if (!t || !sf) return NULL;
    return safetensors_get_f32(sf, t);
}

int qwen_speaker_encoder_load(qwen_speaker_encoder_t *enc, void *safetensors) {
    multi_safetensors_t *st = (multi_safetensors_t *)safetensors;
    int saved_enc_dim = enc->enc_dim;  /* may be pre-set from config */
    memset(enc, 0, sizeof(*enc));
    enc->enc_dim = saved_enc_dim > 0 ? saved_enc_dim : 1024;
    enc->mel_dim = 128;

    /* Helper macro */
    #define LOAD_F32(name, ptr) do { \
        (ptr) = load_f32_tensor(st, (name)); \
        if (!(ptr)) { fprintf(stderr, "Error: missing speaker_encoder weight: %s\n", (name)); return -1; } \
    } while(0)

    /* blocks.0 (initial TDNN) */
    LOAD_F32("speaker_encoder.blocks.0.conv.weight", enc->block0_conv_w);
    LOAD_F32("speaker_encoder.blocks.0.conv.bias", enc->block0_conv_b);

    /* blocks.1-3 (SE-Res2Net) */
    int dilations[] = {2, 3, 4};
    for (int b = 0; b < 3; b++) {
        enc->se_blocks[b].dilation = dilations[b];
        char name[128];

        snprintf(name, sizeof(name), "speaker_encoder.blocks.%d.tdnn1.conv.weight", b + 1);
        LOAD_F32(name, enc->se_blocks[b].tdnn1_conv_w);
        snprintf(name, sizeof(name), "speaker_encoder.blocks.%d.tdnn1.conv.bias", b + 1);
        LOAD_F32(name, enc->se_blocks[b].tdnn1_conv_b);

        for (int r = 0; r < 7; r++) {
            snprintf(name, sizeof(name), "speaker_encoder.blocks.%d.res2net_block.blocks.%d.conv.weight", b + 1, r);
            LOAD_F32(name, enc->se_blocks[b].res2net_conv_w[r]);
            snprintf(name, sizeof(name), "speaker_encoder.blocks.%d.res2net_block.blocks.%d.conv.bias", b + 1, r);
            LOAD_F32(name, enc->se_blocks[b].res2net_conv_b[r]);
        }

        snprintf(name, sizeof(name), "speaker_encoder.blocks.%d.tdnn2.conv.weight", b + 1);
        LOAD_F32(name, enc->se_blocks[b].tdnn2_conv_w);
        snprintf(name, sizeof(name), "speaker_encoder.blocks.%d.tdnn2.conv.bias", b + 1);
        LOAD_F32(name, enc->se_blocks[b].tdnn2_conv_b);

        snprintf(name, sizeof(name), "speaker_encoder.blocks.%d.se_block.conv1.weight", b + 1);
        LOAD_F32(name, enc->se_blocks[b].se_conv1_w);
        snprintf(name, sizeof(name), "speaker_encoder.blocks.%d.se_block.conv1.bias", b + 1);
        LOAD_F32(name, enc->se_blocks[b].se_conv1_b);
        snprintf(name, sizeof(name), "speaker_encoder.blocks.%d.se_block.conv2.weight", b + 1);
        LOAD_F32(name, enc->se_blocks[b].se_conv2_w);
        snprintf(name, sizeof(name), "speaker_encoder.blocks.%d.se_block.conv2.bias", b + 1);
        LOAD_F32(name, enc->se_blocks[b].se_conv2_b);
    }

    /* MFA */
    LOAD_F32("speaker_encoder.mfa.conv.weight", enc->mfa_conv_w);
    LOAD_F32("speaker_encoder.mfa.conv.bias", enc->mfa_conv_b);

    /* ASP */
    LOAD_F32("speaker_encoder.asp.tdnn.conv.weight", enc->asp_tdnn_conv_w);
    LOAD_F32("speaker_encoder.asp.tdnn.conv.bias", enc->asp_tdnn_conv_b);
    LOAD_F32("speaker_encoder.asp.conv.weight", enc->asp_conv_w);
    LOAD_F32("speaker_encoder.asp.conv.bias", enc->asp_conv_b);

    /* FC */
    LOAD_F32("speaker_encoder.fc.weight", enc->fc_w);
    LOAD_F32("speaker_encoder.fc.bias", enc->fc_b);

    #undef LOAD_F32
    enc->loaded = 1;
    return 0;
}


int qwen_speaker_encoder_forward(qwen_speaker_encoder_t *enc,
                                 const float *mel, int n_frames,
                                 float *out_embedding) {
    int T = n_frames;

    /* Input: mel is [n_frames, 128] row-major
     * Python does hidden_states = hidden_states.transpose(1, 2) to get [batch, 128, T]
     * We need channel-first: [128, T] */
    float *x = (float *)malloc((size_t)enc->mel_dim * T * sizeof(float));
    for (int f = 0; f < T; f++)
        for (int m = 0; m < enc->mel_dim; m++)
            x[(size_t)m * T + f] = mel[(size_t)f * enc->mel_dim + m];

    /* blocks.0: TDNN(128→512, k=5, d=1) */
    int ch0 = 512;
    float *h0 = (float *)malloc((size_t)ch0 * T * sizeof(float));
    tdnn_forward(x, enc->mel_dim, T, enc->block0_conv_w, enc->block0_conv_b,
                 ch0, 5, 1, h0);
    free(x);

    /* blocks.1-3: SE-Res2Net */
    float *block_outputs[3];
    float *prev = h0;
    for (int b = 0; b < 3; b++) {
        float *out = (float *)malloc((size_t)ch0 * T * sizeof(float));
        se_res2net_block_forward(
            prev, ch0, T,
            enc->se_blocks[b].tdnn1_conv_w, enc->se_blocks[b].tdnn1_conv_b,
            enc->se_blocks[b].res2net_conv_w, enc->se_blocks[b].res2net_conv_b,
            enc->se_blocks[b].tdnn2_conv_w, enc->se_blocks[b].tdnn2_conv_b,
            enc->se_blocks[b].se_conv1_w, enc->se_blocks[b].se_conv1_b,
            enc->se_blocks[b].se_conv2_w, enc->se_blocks[b].se_conv2_b,
            3, enc->se_blocks[b].dilation, 8,
            out);
        block_outputs[b] = out;
        prev = out;
    }
    free(h0);

    /* MFA: concatenate block_outputs[0,1,2] → [1536, T], then TDNN(1536→1536, k=1) */
    int mfa_ch = 1536;
    float *cat = (float *)malloc((size_t)mfa_ch * T * sizeof(float));
    for (int b = 0; b < 3; b++) {
        memcpy(cat + (size_t)b * ch0 * T, block_outputs[b],
               (size_t)ch0 * T * sizeof(float));
        free(block_outputs[b]);
    }

    float *mfa_out = (float *)malloc((size_t)mfa_ch * T * sizeof(float));
    tdnn_forward(cat, mfa_ch, T, enc->mfa_conv_w, enc->mfa_conv_b,
                 mfa_ch, 1, 1, mfa_out);
    free(cat);

    /* ASP (Attentive Statistics Pooling)
     * Input: [1536, T]
     * 1. Compute mean and std across time
     * 2. Concatenate [hidden, mean_expanded, std_expanded] → [4608, T]
     * 3. TDNN(4608→128, k=1) → tanh → Conv1d(128→1536, k=1) → softmax over time
     * 4. Weighted mean and std → concatenate → [3072, 1] */

    /* Compute mean and std */
    float *asp_mean = (float *)calloc(mfa_ch, sizeof(float));
    float *asp_std = (float *)calloc(mfa_ch, sizeof(float));
    for (int c = 0; c < mfa_ch; c++) {
        float sum = 0;
        for (int t = 0; t < T; t++)
            sum += mfa_out[(size_t)c * T + t];
        asp_mean[c] = sum / T;
    }
    for (int c = 0; c < mfa_ch; c++) {
        float sum_sq = 0;
        for (int t = 0; t < T; t++) {
            float d = mfa_out[(size_t)c * T + t] - asp_mean[c];
            sum_sq += d * d;
        }
        asp_std[c] = sqrtf(sum_sq / T + 1e-12f);
    }

    /* Build attention input: [hidden, mean_expanded, std_expanded] → [4608, T] */
    int att_ch = mfa_ch * 3;  /* 4608 */
    float *att_input = (float *)malloc((size_t)att_ch * T * sizeof(float));
    memcpy(att_input, mfa_out, (size_t)mfa_ch * T * sizeof(float));
    for (int c = 0; c < mfa_ch; c++)
        for (int t = 0; t < T; t++)
            att_input[(size_t)(mfa_ch + c) * T + t] = asp_mean[c];
    for (int c = 0; c < mfa_ch; c++)
        for (int t = 0; t < T; t++)
            att_input[(size_t)(2 * mfa_ch + c) * T + t] = asp_std[c];

    /* TDNN(4608→128, k=1) + tanh */
    int att_hidden = 128;
    float *att_h = (float *)malloc((size_t)att_hidden * T * sizeof(float));
    tdnn_forward(att_input, att_ch, T, enc->asp_tdnn_conv_w, enc->asp_tdnn_conv_b,
                 att_hidden, 1, 1, att_h);
    free(att_input);
    /* Replace ReLU with tanh (tdnn_forward applied ReLU, but ASP uses tanh) */
    /* Actually, tdnn_forward does ReLU. But for ASP, the TDNN output goes through tanh.
     * The Python code: attention = self.conv(self.tanh(self.tdnn(attention)))
     * The tdnn itself has ReLU activation. Then the outer code applies tanh.
     * Wait, looking at the Python: self.tdnn is a TimeDelayNetBlock which has ReLU.
     * Then self.tanh is applied. So: x → ReLU(conv(x)) → tanh(x).
     * That's a bit unusual but let's follow it exactly. */
    for (int i = 0; i < att_hidden * T; i++)
        att_h[i] = tanhf(att_h[i]);

    /* Conv1d(128→1536, k=1) */
    float *att_w = (float *)malloc((size_t)mfa_ch * T * sizeof(float));
    conv1d_same_reflect(att_h, att_hidden, T, enc->asp_conv_w, enc->asp_conv_b,
                        mfa_ch, 1, 1, att_w);
    free(att_h);

    /* Softmax over time dimension for each channel */
    for (int c = 0; c < mfa_ch; c++) {
        float max_val = -1e30f;
        for (int t = 0; t < T; t++) {
            float v = att_w[(size_t)c * T + t];
            if (v > max_val) max_val = v;
        }
        float sum = 0;
        for (int t = 0; t < T; t++) {
            att_w[(size_t)c * T + t] = expf(att_w[(size_t)c * T + t] - max_val);
            sum += att_w[(size_t)c * T + t];
        }
        for (int t = 0; t < T; t++)
            att_w[(size_t)c * T + t] /= sum;
    }

    /* Weighted mean and std */
    float *w_mean = (float *)calloc(mfa_ch, sizeof(float));
    float *w_std = (float *)calloc(mfa_ch, sizeof(float));
    for (int c = 0; c < mfa_ch; c++) {
        float m = 0;
        for (int t = 0; t < T; t++)
            m += att_w[(size_t)c * T + t] * mfa_out[(size_t)c * T + t];
        w_mean[c] = m;
    }
    for (int c = 0; c < mfa_ch; c++) {
        float var = 0;
        for (int t = 0; t < T; t++) {
            float d = mfa_out[(size_t)c * T + t] - w_mean[c];
            var += att_w[(size_t)c * T + t] * d * d;
        }
        w_std[c] = sqrtf(var + 1e-12f);
    }

    free(asp_mean);
    free(asp_std);
    free(att_w);
    free(mfa_out);

    /* Pooled stats: concatenate [w_mean, w_std] → [3072, 1] */
    float *pooled = (float *)malloc(3072 * sizeof(float));
    memcpy(pooled, w_mean, mfa_ch * sizeof(float));
    memcpy(pooled + mfa_ch, w_std, mfa_ch * sizeof(float));
    free(w_mean);
    free(w_std);

    /* FC: Conv1d(3072→1024, k=1) — just a matrix multiply on length-1 input
     * fc weight is [1024, 3072, 1]: out[i] = bias[i] + sum_j(w[i,j,0] * in[j]) */
    for (int i = 0; i < enc->enc_dim; i++) {
        float val = enc->fc_b[i];
        for (int j = 0; j < 3072; j++)
            val += enc->fc_w[(size_t)i * 3072 + j] * pooled[j];
        out_embedding[i] = val;
    }

    free(pooled);
    return 0;
}


/* ════════════════════════════════════════════════════════════════════════
 * High-level API
 * ════════════════════════════════════════════════════════════════════════ */

int qwen_extract_speaker_embedding(qwen_tts_ctx_t *ctx, const char *ref_audio_path,
                                   float *out_embedding) {
    /* Read WAV file */
    float *audio = NULL;
    int n_samples = 0, sample_rate = 0;
    if (qwen_read_wav(ref_audio_path, &audio, &n_samples, &sample_rate) != 0)
        return -1;

    if (!ctx->silent)
        fprintf(stderr, "Reference audio: %s (%d samples, %d Hz, %.2fs)\n",
                ref_audio_path, n_samples, sample_rate, (float)n_samples / sample_rate);

    /* Truncate to max_ref_seconds if set (default 15s, 0=use all) */
    if (ctx->max_ref_seconds > 0) {
        int max_samples = (int)(ctx->max_ref_seconds * sample_rate);
        if (n_samples > max_samples) {
            if (!ctx->silent)
                fprintf(stderr, "  Truncating to %.0fs (was %.1fs)\n",
                        ctx->max_ref_seconds, (float)n_samples / sample_rate);
            n_samples = max_samples;
        }
    }

    /* TODO: resample to 24kHz if needed. For now, require 24kHz input. */
    if (sample_rate != 24000) {
        fprintf(stderr, "Error: reference audio must be 24 kHz (got %d Hz)\n", sample_rate);
        fprintf(stderr, "Convert with: ffmpeg -i input.wav -ar 24000 output.wav\n");
        free(audio);
        return -1;
    }

    /* Compute mel spectrogram */
    float *mel = NULL;
    int n_frames = 0;
    if (qwen_mel_spectrogram(audio, n_samples, sample_rate, &mel, &n_frames) != 0) {
        free(audio);
        return -1;
    }
    free(audio);

    if (!ctx->silent)
        fprintf(stderr, "Mel spectrogram: %d frames x 128 mels\n", n_frames);

    /* Run speaker encoder */
    if (qwen_speaker_encoder_forward(&ctx->speaker_enc, mel, n_frames, out_embedding) != 0) {
        free(mel);
        return -1;
    }
    free(mel);

    return 0;
}


/* Speech tokenizer encoder moved to qwen_tts_speech_encoder.c */
