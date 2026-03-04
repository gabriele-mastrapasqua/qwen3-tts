/* qwen_tts_audio.h */
#ifndef QWEN_TTS_AUDIO_H
#define QWEN_TTS_AUDIO_H
int qwen_tts_write_wav(const char *path, const float *samples, int n_samples, int sample_rate);
#endif
