/* qwen_tts_emotion.h - compound-emotion manifest
 *
 * Maps a single human mood name (joy, sad, stern, annoyed, ...) to a full,
 * ear-validated recipe across ALL expressivity knobs at once:
 *   { control vector, steering weight, roughness, volume gain, tempo rate }.
 *
 * This is the v2 of --emotion: instead of the user having to remember
 * "excited at weight 2.6 plus atempo 1.10 plus volume 1.10 = joy", they type
 * `--emotion joy`. Any explicitly-passed flag (--steer-weight/--roughness/
 * --volume/--rate) overrides the baked recipe value.
 *
 * Recipes encode docs/expressivity-recipes.md (validated 2026-06-06).
 */
#ifndef QWEN_TTS_EMOTION_H
#define QWEN_TTS_EMOTION_H

typedef struct {
    const char *name;      /* mood name the user types (case-insensitive) */
    const char *vec_spec;  /* preset .vec name to steer with (resolver-looked-up),
                              blend syntax "a:0.5,b:0.5" allowed, or NULL = no steering */
    float steer_weight;    /* baked global injection scale (--steer-weight) */
    float roughness;       /* baked --roughness [0..1] */
    float volume;          /* baked --volume gain (1.0 = unchanged) */
    float rate;            /* baked --rate tempo (1.0 = unchanged, >1 faster) */
    const char *desc;      /* one-line human description */
} qwen_emotion_recipe_t;

/* Case-insensitive lookup. Returns NULL if `name` is not a manifest mood
 * (caller then falls back to treating it as a raw .vec preset name/blend). */
const qwen_emotion_recipe_t *qwen_emotion_lookup(const char *name);

/* Full manifest table (NULL-name-terminated) + count, for help/listing. */
const qwen_emotion_recipe_t *qwen_emotion_table(int *count);

/* Apply an emotion recipe to a context (shared by the CLI and the HTTP server).
 * Resolves a manifest mood (or raw .vec spec/blend) for `language`, loads the
 * steering vector into ctx->cp_steer_vec/dim/weight, sets ctx->cp_roughness, and
 * returns the effective volume/rate via out params (recipe value unless the
 * matching *_set override is passed). Returns 0 on success, -1 on hard error. */
typedef struct qwen_tts_ctx qwen_tts_ctx_t;
int qwen_tts_apply_emotion(qwen_tts_ctx_t *ctx,
        const char *emotion_spec, const char *steer_vector_path, const char *language,
        float sw, int sw_set, float ro, int ro_set,
        float vo, int vo_set, float ra, int ra_set,
        float *out_volume, float *out_rate, int silent);

#endif
