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

static const qwen_emotion_recipe_t MANIFEST[] = {
    /* name        vec_spec    weight rough  vol   rate   description */
    { "joy",       "excited",  2.6f,  0.00f, 1.10f, 1.10f, "bright, energetic (excited pushed + faster + louder)" },
    { "happy",     "happy",    1.6f,  0.00f, 1.00f, 1.00f, "mild upbeat (gentle; use 'joy' for strong brightness)" },
    { "excited",   "excited",  2.2f,  0.00f, 1.00f, 1.00f, "lively, animated" },
    { "eager",     "eager",    2.0f,  0.00f, 1.00f, 1.00f, "keen, forward-leaning" },
    { "proud",     "proud",    2.0f,  0.00f, 1.00f, 1.00f, "confident, self-assured" },
    { "news",      "proud",    2.0f,  0.00f, 1.00f, 1.00f, "authoritative news-anchor delivery" },
    { "dramatic",  "dramatic", 2.0f,  0.00f, 1.00f, 1.00f, "theatrical, weighty" },
    { "calm",      "calm",     1.6f,  0.00f, 0.95f, 0.96f, "relaxed, unhurried (slightly slower/softer)" },
    /* down-moods: the sadness comes from PROSODY (slow + soft), steering is LOW —
     * high weight pushes the IT direction off-manifold into a 'Chinese tone' basin
     * (ear-validated 2026-06-07: sad sweet spot = weight ~1.1 + rate 0.80 + vol 0.86;
     * the gloomy direction goes off-manifold even at 1.0, so it stays lower). */
    { "sad",       "sad",      1.1f,  0.00f, 0.86f, 1.08f, "downcast (steering + quiet, lightly compressed so words don't drag)" },
    { "gloomy",    "gloomy",   0.5f,  0.00f, 0.88f, 0.82f, "somber, low (mostly prosody; gloomy dir is off-manifold-prone on IT)" },
    { "annoyed",   "angry",    2.6f,  0.32f, 1.05f, 1.05f, "irritated/short-tempered (angry + grit + brisk)" },
    { "stern",     "angry",    2.6f,  0.28f, 1.05f, 1.00f, "firm, authoritative reprimand" },
    { "angry",     "angry",    2.6f,  0.40f, 1.05f, 1.05f, "forceful/heated (full furious rage is out of model reach)" },
    { NULL, NULL, 0.0f, 0.0f, 0.0f, 0.0f, NULL }
};

const qwen_emotion_recipe_t *qwen_emotion_lookup(const char *name) {
    if (!name) return NULL;
    for (int i = 0; MANIFEST[i].name; i++)
        if (strcasecmp(name, MANIFEST[i].name) == 0) return &MANIFEST[i];
    return NULL;
}

const qwen_emotion_recipe_t *qwen_emotion_table(int *count) {
    if (count) {
        int n = 0;
        while (MANIFEST[n].name) n++;
        *count = n;
    }
    return MANIFEST;
}

/* ── Emotion application (shared by CLI + server) ──────────────────────────
 * Moved out of main.c so both the CLI and the HTTP server apply the SAME
 * ear-validated recipe. Loads the steering vector for (emotion, language) into
 * ctx->cp_steer_vec/dim/weight + sets ctx->cp_roughness, and returns the
 * effective volume/rate (the recipe value unless a *_set override is passed). */

/* Load a .vec steer file (QSTV magic + int32 dim + dim*float32), accumulate scaled. */
static int load_steer_vec_accum(const char *path, float scale, float **acc, int expect_dim) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Error: cannot open steer vector '%s'\n", path); return -1; }
    uint32_t magic = 0; int32_t dim = 0;
    if (fread(&magic, 4, 1, f) != 1 || fread(&dim, 4, 1, f) != 1 ||
        magic != 0x56545351u /* 'QSTV' */ || dim != expect_dim) {
        fprintf(stderr, "Error: '%s' is not a valid steer vector for this model "
                        "(dim=%d, expected %d)\n", path, dim, expect_dim);
        fclose(f); return -1;
    }
    float *tmp = (float *)malloc((size_t)dim * sizeof(float));
    if (!tmp || fread(tmp, sizeof(float), dim, f) != (size_t)dim) {
        fprintf(stderr, "Error: failed to read steer vector '%s'\n", path);
        free(tmp); fclose(f); return -1;
    }
    fclose(f);
    if (!*acc) *acc = (float *)calloc(dim, sizeof(float));
    if (*acc) for (int i = 0; i < dim; i++) (*acc)[i] += scale * tmp[i];
    free(tmp);
    return *acc ? 0 : -1;
}

/* Resolve an emotion preset name to its .vec path (first hit wins), language-aware. */
static int resolve_emotion_path(const char *name, const char *language, char *out, size_t outsz) {
    const char *bases[] = { getenv("QWEN_EMOTION_DIR"), "presets/emotions", "voices/emotions" };
    const char *it_subs[] = { "it_centered", "it", NULL };
    const char *none[]    = { NULL };
    const char **subs = none;
    if (language && strcasecmp(language, "Italian") == 0) subs = it_subs;
    for (size_t i = 0; i < sizeof(bases) / sizeof(bases[0]); i++) {
        if (!bases[i] || !*bases[i]) continue;
        for (int s = 0; subs[s]; s++) {
            snprintf(out, outsz, "%s/%s/%s.vec", bases[i], subs[s], name);
            FILE *f = fopen(out, "rb");
            if (f) { fclose(f); return 0; }
        }
        snprintf(out, outsz, "%s/%s.vec", bases[i], name);
        FILE *f = fopen(out, "rb");
        if (f) { fclose(f); return 0; }
    }
    return -1;
}

int qwen_tts_apply_emotion(qwen_tts_ctx_t *ctx,
        const char *emotion_spec, const char *steer_vector_path, const char *language,
        float sw, int sw_set, float ro, int ro_set,
        float vo, int vo_set, float ra, int ra_set,
        float *out_volume, float *out_rate, int silent) {
    int cp_h = ctx->config.cp_hidden_size;
    float eff_steer_weight = sw, eff_roughness = ro, eff_volume = vo, eff_rate = ra;
    const char *vec_spec = emotion_spec;
    const qwen_emotion_recipe_t *recipe = NULL;

    if (ctx->cp_steer_vec) { free(ctx->cp_steer_vec); ctx->cp_steer_vec = NULL; ctx->cp_steer_dim = 0; }
    ctx->cp_roughness = 0.0f;

    if (emotion_spec && !strchr(emotion_spec, ',') && !strchr(emotion_spec, ':'))
        recipe = qwen_emotion_lookup(emotion_spec);
    if (recipe) {
        vec_spec = recipe->vec_spec;
        if (!sw_set) eff_steer_weight = recipe->steer_weight;
        if (!ro_set) eff_roughness    = recipe->roughness;
        if (!vo_set) eff_volume       = recipe->volume;
        if (!ra_set) eff_rate         = recipe->rate;
        if (!silent)
            fprintf(stderr, "Emotion '%s' -> %s\n  (vec=%s weight=%.2f roughness=%.2f volume=%.2f rate=%.2f)\n",
                    emotion_spec, recipe->desc, vec_spec ? vec_spec : "(none)",
                    eff_steer_weight, eff_roughness, eff_volume, eff_rate);
    }

    if (eff_roughness > 0.0f) {
        if (eff_roughness > 1.0f) eff_roughness = 1.0f;
        ctx->cp_roughness = eff_roughness;
        if (!silent && !recipe) fprintf(stderr, "Roughness: %.2f (q2-down blend on Code Predictor)\n", eff_roughness);
    }

    if (vec_spec || steer_vector_path) {
        float *steer_acc = NULL;
        if (vec_spec) {
            char *spec = strdup(vec_spec);
            for (char *tok = strtok(spec, ","); tok; tok = strtok(NULL, ",")) {
                float scale = 1.0f;
                char *colon = strchr(tok, ':');
                if (colon) { *colon = '\0'; scale = (float)atof(colon + 1); }
                char path[1024];
                if (resolve_emotion_path(tok, language, path, sizeof(path)) != 0) {
                    if (recipe) {
                        if (!silent) fprintf(stderr, "Note: mood '%s' has no '%s' vector for this language; "
                                                     "applying roughness/volume/rate only.\n", emotion_spec, tok);
                    } else {
                        fprintf(stderr, "Error: unknown emotion preset '%s' "
                            "(looked in $QWEN_EMOTION_DIR, presets/emotions/[<lang>/], voices/emotions/)\n", tok);
                        free(spec); free(steer_acc); return -1;
                    }
                } else if (load_steer_vec_accum(path, scale, &steer_acc, cp_h) != 0) {
                    free(spec); free(steer_acc); return -1;
                } else if (!silent && !recipe) {
                    fprintf(stderr, "Emotion: %s x%.2f (%s)\n", tok, scale, path);
                }
            }
            free(spec);
        }
        if (steer_vector_path && load_steer_vec_accum(steer_vector_path, 1.0f, &steer_acc, cp_h) != 0) {
            free(steer_acc); return -1;
        }
        if (steer_acc) {
            ctx->cp_steer_vec = steer_acc;
            ctx->cp_steer_dim = cp_h;
            ctx->cp_steer_weight = eff_steer_weight;
            if (!silent) fprintf(stderr, "Steering: active (dim=%d, weight=%.2f)\n", cp_h, eff_steer_weight);
        }
    }
    if (out_volume) *out_volume = eff_volume;
    if (out_rate)   *out_rate   = eff_rate;
    return 0;
}
