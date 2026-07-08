#ifndef QWEN_TTS_COMPOSE_H
#define QWEN_TTS_COMPOSE_H

/* Inline expressive-markup composer (shared by the CLI and the HTTP server).
 *
 * A single text body carries ENGLISH square-bracket tags, switchable mid-text:
 *   [happy] [sad] [excited] [annoyed] [proud] [calm] ...  emotion for following text
 *   [neutral]                                  back to plain delivery
 *   [laugh] [sigh]                             paralinguistic events (inline onomatopoeia)
 *   [sigh] [huff] [ugh] [hmm] ...              paralinguistic soft fillers (DSP macros)
 *   [pause:400ms] [pause:1s] [pause:0.5]       a pause (also [break:...] or bare [0.5])
 * The text is split into spans; each span is synthesized with its own emotion recipe and
 * the results are concatenated (model-generated -> seamless, same voice/codec). This is the
 * mechanism behind per-sentence dynamic emotion on both the CLI (--compose / auto-detected
 * --text) and the server (/v1/tts and /v1/tts/stream). */

typedef struct qwen_tts_ctx qwen_tts_ctx_t;

/* steer_weight: <0 = use the mood recipe's weight; >=0 = override (0 = NO steering).
 * rate/volume: >0 = override the recipe value, else inherit it. */
typedef struct {
    int   is_pause;      /* 1 = silence gap of pause_s seconds, no text */
    float pause_s;
    char  mood[48];      /* emotion name ("" = neutral) */
    char *text;          /* spoken text (owned by caller; NULL for a pause span) */
    float steer_weight;
    float rate;
    float volume;
    int   is_filler;     /* paralinguistic soft-filler macro (crossfade seam) */
} qwen_cspan_t;

/* ── Detection (auto-routing) ─────────────────────────────────────────── */
/* Does `text` contain at least one RECOGNIZED inline tag (emotion/pause/para)? */
int qwen_compose_has_markup(const char *text);
/* Does `text` contain at least one PARALINGUISTIC event tag ([laugh]/[sigh]/[huff]/…)? */
int qwen_compose_has_para_event(const char *text);
/* Is `tag` one of the inline paralinguistic event tags ([laugh]/[sigh]/…)? */
int qwen_compose_is_para_event_tag(const char *tag);

/* ── Paralinguistic inline substitution ───────────────────────────────── */
/* Replace every [laugh]/[sigh] in `text` with its validated onomatopoeia, comma-delimited,
 * so the event renders in the active voice's own timbre within ONE generation. Returns a new
 * string (caller frees); sets *did=1 if any tag matched and *seed to the FIRST tag's validated
 * seed. voice_class: 0 = ryan/clone/default, 1 = vivian. */
char *qwen_compose_para_substitute(const char *text, int voice_class, int *did, int *seed, float *temp);

/* ── Parse + render ───────────────────────────────────────────────────── */
/* Parse inline markup into a span list. Free with qwen_compose_free_spans. */
int  qwen_compose_parse(const char *input, qwen_cspan_t **out, int *out_n);
void qwen_compose_free_spans(qwen_cspan_t *spans, int n);

/* Synthesize a parsed span list into ONE in-memory mono/24kHz buffer (caller frees
 * *out_audio). Applies per-span emotion + rate/volume DSP + filler crossfades, exactly
 * like the CLI --compose path. Returns 0 on success. */
int  qwen_compose_render_buffer(qwen_tts_ctx_t *ctx, qwen_cspan_t *spans, int nspans,
                                const char *language, float default_pause,
                                float **out_audio, int *out_n, int silent);

/* Streaming variant: synthesize span-by-span and hand each emitted PCM run (span audio and
 * inter-span pauses) to `cb` as soon as it is ready — low time-to-first-audio. Per-span emotion
 * and rate/volume DSP are applied; filler crossfades across the network boundary are skipped
 * (each span is emitted independently). Returns 0 on success. */
typedef void (*qwen_compose_chunk_cb)(const float *pcm, int n, void *user);
int  qwen_compose_render_stream(qwen_tts_ctx_t *ctx, qwen_cspan_t *spans, int nspans,
                                const char *language, float default_pause,
                                qwen_compose_chunk_cb cb, void *user, int silent);

#endif /* QWEN_TTS_COMPOSE_H */
