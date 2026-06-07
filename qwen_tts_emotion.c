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
#include <strings.h>  /* strcasecmp */
#include <stddef.h>

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
    { "sad",       "sad",      2.0f,  0.00f, 0.90f, 0.84f, "downcast (slower + quieter; add ... pauses in text)" },
    { "gloomy",    "gloomy",   2.0f,  0.00f, 0.90f, 0.85f, "somber, low (slower + quieter)" },
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
