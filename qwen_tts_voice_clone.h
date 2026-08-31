/* qwen_tts_voice_clone.h - Voice cloning support for Qwen3-TTS Base models */
#ifndef QWEN_TTS_VOICE_CLONE_H
#define QWEN_TTS_VOICE_CLONE_H

#include <stdint.h>

typedef struct qwen_tts_ctx qwen_tts_ctx_t;

typedef struct {
    int enc_dim;
    int mel_dim;
    int loaded;

    float *block0_conv_w;
    float *block0_conv_b;

    struct {
        float *tdnn1_conv_w;
        float *tdnn1_conv_b;
        float *res2net_conv_w[7];
        float *res2net_conv_b[7];
        float *tdnn2_conv_w;
        float *tdnn2_conv_b;
        float *se_conv1_w;
        float *se_conv1_b;
        float *se_conv2_w;
        float *se_conv2_b;
        int dilation;
    } se_blocks[3];

    float *mfa_conv_w;
    float *mfa_conv_b;

    float *asp_tdnn_conv_w;
    float *asp_tdnn_conv_b;
    float *asp_conv_w;
    float *asp_conv_b;

    float *fc_w;
    float *fc_b;
} qwen_speaker_encoder_t;

int qwen_read_wav(const char *path, float **out_samples, int *out_n_samples, int *out_sample_rate);

void qwen_trim_trailing_silence(float *audio, int *n_samples, int sample_rate, int silent);

int qwen_mel_spectrogram(const float *audio, int n_samples, int sample_rate,
                         float **out_mel, int *out_n_frames);

int qwen_speaker_encoder_load(qwen_speaker_encoder_t *enc, void *safetensors);

int qwen_speaker_encoder_forward(qwen_speaker_encoder_t *enc,
                                 const float *mel, int n_frames,
                                 float *out_embedding);

int qwen_extract_speaker_embedding(qwen_tts_ctx_t *ctx, const char *ref_audio_path,
                                   float *out_embedding);

#endif
