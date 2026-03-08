/*
 * qwen_tts_voice_clone.h - Voice cloning support for Qwen3-TTS Base models
 *
 * The speaker_encoder_t struct is defined in qwen_tts.h (part of ctx).
 * This header declares the operational functions.
 */

#ifndef QWEN_TTS_VOICE_CLONE_H
#define QWEN_TTS_VOICE_CLONE_H

#include <stdint.h>

/* Forward declarations */
typedef struct qwen_tts_ctx qwen_tts_ctx_t;

/* ── ECAPA-TDNN Speaker Encoder state ───────────────────────────────── */

typedef struct {
    int enc_dim;       /* output embedding dimension (1024) */
    int mel_dim;       /* input mel dimension (128) */
    int loaded;        /* 1 if weights loaded, 0 otherwise */

    /* blocks.0 (initial TDNN): Conv1d(128→512, k=5, d=1) */
    float *block0_conv_w;   /* [512, 128, 5] */
    float *block0_conv_b;   /* [512] */

    /* blocks.1-3 (SE-Res2Net) */
    struct {
        float *tdnn1_conv_w;   /* [512, 512, 1] */
        float *tdnn1_conv_b;   /* [512] */
        float *res2net_conv_w[7];  /* each [64, 64, 3] */
        float *res2net_conv_b[7];  /* each [64] */
        float *tdnn2_conv_w;   /* [512, 512, 1] */
        float *tdnn2_conv_b;   /* [512] */
        float *se_conv1_w;     /* [128, 512, 1] */
        float *se_conv1_b;     /* [128] */
        float *se_conv2_w;     /* [512, 128, 1] */
        float *se_conv2_b;     /* [512] */
        int dilation;
    } se_blocks[3];

    /* mfa: Conv1d(1536→1536, k=1) */
    float *mfa_conv_w;
    float *mfa_conv_b;

    /* asp (attentive statistics pooling) */
    float *asp_tdnn_conv_w;  /* [128, 4608, 1] */
    float *asp_tdnn_conv_b;  /* [128] */
    float *asp_conv_w;       /* [1536, 128, 1] */
    float *asp_conv_b;       /* [1536] */

    /* fc: Conv1d(3072→1024, k=1) */
    float *fc_w;    /* [1024, 3072, 1] */
    float *fc_b;    /* [1024] */
} qwen_speaker_encoder_t;

/* ── WAV reader ─────────────────────────────────────────────────────── */

int qwen_read_wav(const char *path, float **out_samples, int *out_n_samples, int *out_sample_rate);

/* ── Mel spectrogram ────────────────────────────────────────────────── */

int qwen_mel_spectrogram(const float *audio, int n_samples, int sample_rate,
                         float **out_mel, int *out_n_frames);

/* ── Speaker encoder ────────────────────────────────────────────────── */

int qwen_speaker_encoder_load(qwen_speaker_encoder_t *enc, void *safetensors);

int qwen_speaker_encoder_forward(qwen_speaker_encoder_t *enc,
                                 const float *mel, int n_frames,
                                 float *out_embedding);

/* ── High-level API (requires full ctx — include qwen_tts.h first) ── */

int qwen_extract_speaker_embedding(qwen_tts_ctx_t *ctx, const char *ref_audio_path,
                                   float *out_embedding);

#endif /* QWEN_TTS_VOICE_CLONE_H */
