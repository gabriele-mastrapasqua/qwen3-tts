/* qwen_tts_emotion.c - compound-emotion manifest (see qwen_tts_emotion.h)
 *
 * Each row is an ear-validated recipe from docs/expressivity-recipes.md.
 * KEY findings baked in here:
 *   - joy = `excited` (NOT `happy`) pushed hard + faster + louder. `happy`
 *     loses energy when pushed because neutral is already upbeat.
 *   - down-moods (sad/gloomy/calm) = rate&volume DOWN, not more steering weight
 *     (a single-point CP injection can't impose tempo/energy; DSP does).
 *   - annoyed/stern = `angry` direction + roughness grit + slightly faster/louder.
 *     Full furious rage is out of reach (the model forces it to proud/forceful).
 *   - news/announcer = `proud` (reads as authoritative anchor).
 *
 * `vec_spec` is resolved through the normal emotion resolver (language-aware:
 * Italian prefers the centered palette). If a recipe's vector is missing for the
 * active language, the caller degrades gracefully — the prosodic knobs
 * (roughness/volume/rate) still apply.
 */
#include "qwen_tts_emotion.h"
#include "qwen_tts.h"
#include <strings.h>  /* strcasecmp */
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/* ─────────────────────────────────────────────────────────────────────────────
 * The qlsteer STEER emotion system (the ONLY emotion path on 1.7B) — shared by the
 * CLI --emotion router, the per-span inline [emotion] compose path, and the server.
 * ──────────────────────────────────────────────────────────────────────────── */
typedef struct { const char *name, *tok; } emo_name_t;
static const emo_name_t EMOTION_NAMES[] = {
    { "sad", "sad" }, { "sadness", "sad" },
    { "joy", "joy" }, { "happy", "joy" }, { "joyful", "joy" },
    { "anger", "ang" }, { "angry", "ang" }, { "rage", "ang" },
    { "fear", "fear" }, { "afraid", "fear" },
    { "disgust", "disgust" }, { "disgusted", "disgust" },
    { "surprise", "surprise" }, { "surprised", "surprise" },
    /* Plutchik dyads — 2-primary blends, ear-validated 2026-07-08 (ryan EN+IT). */
    { "contempt", "contempt" }, { "scorn", "contempt" },
    { "awe", "awe" }, { "wonder", "awe" },
    { "nostalgia", "nostalgia" }, { "wistful", "nostalgia" },
    { "disapproval", "disapproval" },
    { "remorse", "remorse" }, { "regret", "remorse" },
    { "outrage", "outrage" },
    { "despair", "despair" },
    { NULL, NULL }
};
/* Distinct canonical tokens, in a sensible display order (primaries then dyads). */
static const char *const EMOTION_STEER_TOKS[] = {
    "sad", "joy", "ang", "fear", "disgust", "surprise",
    "contempt", "awe", "nostalgia", "disapproval", "remorse", "outrage", "despair", NULL
};

const char *qwen_emotion_name_to_tok(const char *name) {
    if (!name) return NULL;
    for (int i = 0; EMOTION_NAMES[i].name; i++)
        if (strcasecmp(name, EMOTION_NAMES[i].name) == 0) return EMOTION_NAMES[i].tok;
    return NULL;
}

const char *const *qwen_emotion_steer_names(int *count) {
    if (count) { int n = 0; while (EMOTION_STEER_TOKS[n]) n++; *count = n; }
    return EMOTION_STEER_TOKS;
}

int qwen_emotion_steer_install(qwen_tts_ctx_t *ctx, const char *tok, float weight, int l0, int l1, int silent) {
    if (!ctx || !tok) return -1;
    char path[256];
    snprintf(path, sizeof(path), "presets/steer/emotion/ryan_%s.qlsteer", tok);
    FILE *f = fopen(path, "rb");
    if (!f) { if (!silent) fprintf(stderr, "Emotion: steer vector '%s' not found\n", path); return -1; }
    uint32_t magic = 0; int32_t L = 0, D = 0;
    int want_L = ctx->config.num_layers + 1, want_D = ctx->config.hidden_size;
    if (fread(&magic, 4, 1, f) != 1 || fread(&L, 4, 1, f) != 1 || fread(&D, 4, 1, f) != 1 ||
        magic != 0x54534C51u /* 'QLST' */ || L != want_L || D != want_D) {
        if (!silent) fprintf(stderr, "Emotion: '%s' shape %dx%d != %dx%d (skipped)\n", path, L, D, want_L, want_D);
        fclose(f); return -1;
    }
    size_t n = (size_t)L * D;
    float *buf = (float *)malloc(n * sizeof(float));
    if (!buf || fread(buf, sizeof(float), n, f) != n) { free(buf); fclose(f); return -1; }
    fclose(f);
    /* Install WITHOUT freeing a previous ml_steer — caller owns save/restore. */
    ctx->ml_steer = buf; ctx->ml_steer_layers = L; ctx->ml_steer_dim = D;
    ctx->ml_steer_weight = weight;
    ctx->ml_steer_l0 = l0 < 0 ? 0 : l0;
    ctx->ml_steer_l1 = l1 >= L ? L - 1 : l1;
    if (!silent) fprintf(stderr, "Emotion steer: %s (L%d-%d, weight %.1f)\n",
                         path, ctx->ml_steer_l0, ctx->ml_steer_l1, weight);
    return 0;
}

/* The legacy .vec emotion MOOD palette (joy=excited, proud, calm, annoyed, …) is RETIRED
 * (2026-07-08, user verdict: the old CP-steer .vec emotions were weak). Named emotions now
 * ALWAYS route to the new qlsteer STEER above (primaries + dyads). The generic --roughness
 * texture knob and the raw --steer-vector power-user vector are kept below (not "emotions"). */

/* ── Emotion application (shared by CLI + server) — new steer on 1.7B, else knobs ── */

/* --emotion routes to the qlsteer STEER above; --roughness is the only other knob. The
 * legacy --steer-vector / .vec (QSTV) control-vector path was retired 2026-07-09. */
int qwen_tts_apply_emotion(qwen_tts_ctx_t *ctx,
        const char *emotion_spec, const char *language,
        float ro, int ro_set,
        float vo, int vo_set, float ra, int ra_set,
        float *out_volume, float *out_rate, int silent) {
    (void)language;
    float eff_roughness = ro; (void)ro_set;

    ctx->cp_roughness = 0.0f;

    /* ── The ONLY emotion path: the new qlsteer STEER (primaries + Plutchik dyads), 1.7B. ──
     * A named emotion ALWAYS routes here (the legacy .vec mood palette is retired). On 0.6B the
     * install fails on shape and emotion is a no-op (emotion is a 1.7B feature). */
    if (emotion_spec && emotion_spec[0] && ctx->config.hidden_size >= 2048) {
        if (ctx->ml_steer) { free(ctx->ml_steer); ctx->ml_steer = NULL; ctx->ml_steer_layers = 0; }
        const char *tok = qwen_emotion_name_to_tok(emotion_spec);
        if (tok) qwen_emotion_steer_install(ctx, tok, 12.0f, 21, 25, silent);      /* best-effort */
        else if (!silent) fprintf(stderr, "Note: '%s' is not a known emotion (ignored)\n", emotion_spec);
        if (out_volume) *out_volume = vo_set ? vo : 1.0f;
        if (out_rate)   *out_rate   = ra_set ? ra : 1.0f;
        return 0;
    }

    /* ── Generic knob (NOT an emotion): --roughness texture. ── */
    if (eff_roughness > 0.0f) {
        if (eff_roughness > 1.0f) eff_roughness = 1.0f;
        ctx->cp_roughness = eff_roughness;
        if (!silent) fprintf(stderr, "Roughness: %.2f (q2-down blend on Code Predictor)\n", eff_roughness);
    }
    if (out_volume) *out_volume = vo_set ? vo : 1.0f;
    if (out_rate)   *out_rate   = ra_set ? ra : 1.0f;
    return 0;
}
