#ifndef QWEN_TTS_COMPOSE_H
#define QWEN_TTS_COMPOSE_H

typedef struct qwen_tts_ctx qwen_tts_ctx_t;

typedef struct {
    int   is_pause;
    float pause_s;
    char  mood[48];
    char *text;
    float steer_weight;
    float rate;
    float volume;
    int   is_filler;
} qwen_cspan_t;

int qwen_compose_has_markup(const char *text);
int qwen_compose_has_para_event(const char *text);
int qwen_compose_is_para_event_tag(const char *tag);

char *qwen_compose_para_substitute(const char *text, int voice_class, int small_model,
                                   int *did, int *seed, float *temp);

int  qwen_compose_parse(const char *input, qwen_cspan_t **out, int *out_n);
void qwen_compose_free_spans(qwen_cspan_t *spans, int n);

int  qwen_compose_render_buffer(qwen_tts_ctx_t *ctx, qwen_cspan_t *spans, int nspans,
                                const char *language, float default_pause,
                                float **out_audio, int *out_n, int silent);

typedef void (*qwen_compose_chunk_cb)(const float *pcm, int n, void *user);
int  qwen_compose_render_stream(qwen_tts_ctx_t *ctx, qwen_cspan_t *spans, int nspans,
                                const char *language, float default_pause,
                                qwen_compose_chunk_cb cb, void *user, int silent);

#endif
