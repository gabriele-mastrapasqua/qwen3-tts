/* qwen_tts_emotion.h - emotion steering (the qlsteer STEER shelf)
 *
 * `--emotion <name>` on 1.7B applies a per-layer Talker steering vector
 * (presets/steer/emotion/ryan_<tok>.qlsteer @ L21-25 w12): the 6 primaries
 * (sad/joy/anger/fear/disgust/surprise) + 7 Plutchik dyads built as blends
 * (contempt/awe/nostalgia/disapproval/remorse/outrage/despair). The same name
 * table drives CLI --emotion, per-span inline [emotion] in --text, and the server.
 *
 * The legacy .vec CP-steer mood palette (joy=excited, proud, calm, …) was RETIRED
 * 2026-07-08 — it was weak; named emotions now always route to the qlsteer STEER.
 */
#ifndef QWEN_TTS_EMOTION_H
#define QWEN_TTS_EMOTION_H

/* Apply emotion to a context (shared by the CLI and the HTTP server). A named emotion
 * (primary or dyad) routes to the qlsteer STEER on 1.7B; --roughness is the only other
 * (generic, non-emotion) knob. Returns the effective volume/rate via out params. 0 on
 * success, -1 on error. The legacy .vec mood palette + --steer-vector were retired
 * (2026-07-08 palette, 2026-07-09 the raw control-vector path). */
typedef struct qwen_tts_ctx qwen_tts_ctx_t;
int qwen_tts_apply_emotion(qwen_tts_ctx_t *ctx,
        const char *emotion_spec, const char *language,
        float ro, int ro_set,
        float vo, int vo_set, float ra, int ra_set,
        float *out_volume, float *out_rate, int silent);

/* ── The (only) emotion system on 1.7B: qlsteer STEER @ L21-25 ──
 * The 6 primaries + the Plutchik dyads (contempt/awe/nostalgia/disapproval/remorse/outrage/despair),
 * each a `presets/steer/emotion/ryan_<tok>.qlsteer` [num_layers+1 x hidden] vector. Used by the CLI
 * --emotion router, the per-span inline [emotion] compose path, and the server. */

/* Case-insensitive name/alias -> steer token (e.g. "angry"->"ang", "contempt"->"contempt").
 * Returns NULL if `name` is not a known emotion. */
const char *qwen_emotion_name_to_tok(const char *name);

/* Full name list (NULL-terminated) for help/listing: distinct canonical tokens. */
const char *const *qwen_emotion_steer_names(int *count);

/* Load presets/steer/emotion/ryan_<tok>.qlsteer into ctx->ml_steer @ `weight`, layers [l0,l1].
 * Installs a FRESH buffer WITHOUT freeing a previous ctx->ml_steer (caller saves/restores for
 * per-span use). Fails gracefully (-1, no state change) on a shape mismatch (e.g. 0.6B). */
int qwen_emotion_steer_install(qwen_tts_ctx_t *ctx, const char *tok, float weight, int l0, int l1, int silent);

#endif
