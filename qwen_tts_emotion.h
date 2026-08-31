/* qwen_tts_emotion.h - emotion steering (the qlsteer STEER shelf) */
#ifndef QWEN_TTS_EMOTION_H
#define QWEN_TTS_EMOTION_H

typedef struct qwen_tts_ctx qwen_tts_ctx_t;
int qwen_tts_apply_emotion(qwen_tts_ctx_t *ctx,
        const char *emotion_spec, const char *language,
        float ro, int ro_set,
        float vo, int vo_set, float ra, int ra_set,
        float *out_volume, float *out_rate, int silent);

const char *qwen_emotion_name_to_tok(const char *name);

const char *const *qwen_emotion_steer_names(int *count);

int qwen_emotion_steer_install(qwen_tts_ctx_t *ctx, const char *tok, float weight, int l0, int l1, int silent);

#endif
