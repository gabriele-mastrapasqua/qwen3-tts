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

/* --- Edge cleanup + glitch scoring (onset-leveling / tail-trim / seed-audition); default-off --- */

/* First sample index past the leading digital silence (5ms window over a -50dB floor). */
int   qwen_audio_first_onset(const float *s, int n, int sample_rate);

/* Linear fade-in of fade_ms over the REAL onset (kills the strong-emotion attack transient). */
void  qwen_audio_onset_fade(float *s, int n, int sample_rate, int fade_ms);

/* 0..1 degenerate-tail score (high = metallic/noise tail). out_trim_at (optional) = sample index
 * where the flagged tail begins. Read-only; used to rank seed-audition takes AND to drive trim. */
float qwen_audio_tail_glitch_score(const float *s, int n, int sample_rate, int *out_trim_at);

/* Conservative tail-trim: cut the flagged degenerate tail (15ms guard) if score>=min_score.
 * Returns samples trimmed; updates *n in place. */
int   qwen_audio_tail_trim(float *s, int *n, int sample_rate, float min_score);

#endif
