/* qwen_tts_audio.h */
#ifndef QWEN_TTS_AUDIO_H
#define QWEN_TTS_AUDIO_H
int qwen_tts_write_wav(const char *path, const float *samples, int n_samples, int sample_rate);

/* Apply a linear gain to a PCM float buffer in place (soft-clamps to [-1,1]).
 * gain==1.0 is a no-op. Used by --volume. */
void qwen_audio_apply_gain(float *samples, int n_samples, float gain);

/* Pitch-preserving time-stretch (WSOLA). rate>1 = faster/shorter, rate<1 = slower/longer.
 * Allocates *out (caller frees) of length *out_n ≈ n_in / rate. rate==1.0 copies through.
 * Returns 0 on success, -1 on allocation failure. Used by --rate. */
int qwen_audio_time_stretch(const float *in, int n_in, float rate, int sample_rate,
                            float **out, int *out_n);

#endif
