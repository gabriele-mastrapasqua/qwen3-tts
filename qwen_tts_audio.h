/* qwen_tts_audio.h */
#ifndef QWEN_TTS_AUDIO_H
#define QWEN_TTS_AUDIO_H
int qwen_tts_write_wav(const char *path, const float *samples, int n_samples, int sample_rate);

void qwen_audio_apply_gain(float *samples, int n_samples, float gain);

int qwen_audio_time_stretch(const float *in, int n_in, float rate, int sample_rate,
                            float **out, int *out_n);

int   qwen_audio_first_onset(const float *s, int n, int sample_rate);

void  qwen_audio_onset_fade(float *s, int n, int sample_rate, int fade_ms);

float qwen_audio_tail_glitch_score(const float *s, int n, int sample_rate, int *out_trim_at);

int   qwen_audio_tail_trim(float *s, int *n, int sample_rate, float min_score);

#endif
