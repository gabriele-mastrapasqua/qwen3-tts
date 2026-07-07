/* qwen_tts_compose.c — inline expressive-markup composer (see qwen_tts_compose.h).
 *
 * Extracted from main.c so the CLI AND the HTTP server share one implementation of
 * per-span (per-sentence) emotion. The parsing/substitution logic is byte-for-byte the
 * CLI behavior; render_buffer/render_stream are the file-free entry points the server needs.
 */
#include "qwen_tts_compose.h"
#include "qwen_tts.h"
#include "qwen_tts_emotion.h"   /* qwen_tts_apply_emotion, qwen_emotion_lookup */
#include "qwen_tts_audio.h"     /* qwen_audio_time_stretch, qwen_audio_apply_gain */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>            /* strcasecmp / strncasecmp */

/* ── Paralinguistic soft-filler macros (DSP prosody, NO steering) ─────────
 * Ear-validated 2026-06-07 (ryan IT/EN): steering a short stretched vowel goes metallic, so
 * fillers use only a breathy onomatopoeia + gentle slowdown + low volume. */
typedef struct { const char *tag; const char *text; float steer_weight; float rate; float volume; } cmacro_t;
static const cmacro_t COMPOSE_MACROS[] = {
    /* tag        onomatopoeia  steer  rate   volume   (ear-validated 2026-06-07, ryan IT/EN)
     * NOTE: keep rate >= ~0.90 — slower stretches the short vowel into a metallic WSOLA
     * artifact. A trailing "..." can make the model emit a 2nd spurious vocalization. */
    /* sighs / relief */
    { "sigh",    "Hah...",    0.0f, 1.12f, 0.67f },  /* "aaaahhhww" sigh — the "..." IS the sigh (no dots -> a laugh!); rate shaves length */
    { "sighs",   "Hah...",    0.0f, 1.12f, 0.67f },
    { "ahh",     "Haaa...",   0.0f, 0.90f, 0.70f },  /* "ahhhww" pleasure/relief — TOP */
    { "relief",  "Haaa...",   0.0f, 0.90f, 0.70f },
    { "phew",    "Uao...",    0.0f, 1.00f, 0.82f },  /* big tired relief "ooowww" — TOP */
    /* thinking / hesitation / dismissive */
    { "hmm",     "Hmmm...",   0.0f, 0.88f, 0.65f },  /* pensive "hmm" — TOP */
    { "mmm",     "\xe5\x97\xaf", 0.0f, 1.00f, 0.85f }, /* CN 嗯 = soft assent "mmm", less smug than [hmm] — TOP */
    { "hmpf",    "Hmpf...",   0.0f, 1.00f, 0.75f },  /* closed "mmmm" — TOP */
    { "mah",     "Mah...",    0.0f, 0.95f, 0.78f },  /* dismissive "mah" (very IT) — TOP */
    { "uhm",     "Uhm...",    0.0f, 0.95f, 0.72f },  /* tired/bored drawl — TOP */
    /* laughs (language-dependent) */
    { "laugh",   "Eheh...",   0.0f, 0.95f, 0.78f },  /* real chuckle "eheheh" — TOP (IT) */
    { "laughs",  "Eheh...",   0.0f, 0.95f, 0.78f },
    { "haha",    "Haha!",     0.0f, 1.00f, 0.80f },  /* short laugh — TOP (EN) */
    { "heh",     "Hehhh...",  0.0f, 0.95f, 0.70f },  /* smug "eh eh" (EN) */
    /* pain */
    { "ouch",    "Ouch!",     0.0f, 1.00f, 0.85f },  /* sharp pain (EN) — TOP */
    { "ahi",     "Ahi!",      0.0f, 1.00f, 0.85f },  /* sharp pain (IT) */
    /* irritation */
    { "huff",    "Uff...",    0.0f, 1.00f, 0.78f },  /* irritated huff */
    { "ugh",     "Ugh...",    0.0f, 1.00f, 0.78f },
    { NULL, NULL, 0.0f, 0.0f, 0.0f }
};

/* ── Inline paralinguistic EVENTS via a validated onomatopoeia (the shipped [laugh]/[sigh]) ──
 * The tag becomes an onomatopoeia INSIDE the sentence, so the event is produced in the active
 * voice's own timbre within ONE generation — never a separate "splice" span. Mapping is
 * universal across voices/languages; sigh differs per voice (唉 s42 ryan/clone, ahh s7 vivian).
 * [yawn] 哈啊 is vocal + cross-voice (preset s7 / clone s42). [moan]/[throat] stay ryan-only (unshipped,
 * see docs/para-experiments.md 2026-07-07); cry is decoder-ceiling-blocked (needs FT). */
static void para_pick(const char *tag, int voice_class, const char **onom, int *seed, float *temp) {
    /* voice_class: 0 = ryan / other preset · 1 = vivian · 2 = clone (--load-voice).
     * *temp = the per-tag validated temperature (default 1.1; a tag may soften it). */
    *onom = NULL; *seed = 7; *temp = 1.1f;
    if (!strcasecmp(tag, "laugh") || !strcasecmp(tag, "laughs")) {
        *onom = "\xe5\x93\x88\xe5\x93\x88\xe5\x93\x88"; *seed = 7;              /* 哈哈哈 — all voices */
    } else if (!strcasecmp(tag, "sigh") || !strcasecmp(tag, "sighs")) {
        if (voice_class == 1) { *onom = "ahh"; *seed = 7; }                    /* vivian */
        else                  { *onom = "\xe5\x94\x89"; *seed = 42; }          /* 唉 — ryan/clone */
    } else if (!strcasecmp(tag, "yawn") || !strcasecmp(tag, "yawns")) {
        *onom = "\xe5\x93\x88\xe5\x95\x8a";                                    /* 哈啊 — vocal, cross-voice (ear T3/2026-07-07) */
        *seed = (voice_class == 2) ? 42 : 7;                                   /* clone s42 · preset s7 */
    } else if (!strcasecmp(tag, "wow")) {
        *onom = "\xe5\x93\x87"; *seed = 7;                                     /* 哇 — "wow!" interjection, universal (ear 2026-07-07) */
    } else if (!strcasecmp(tag, "giggle") || !strcasecmp(tag, "giggles")) {
        *onom = "\xe5\x98\xbf\xe5\x98\xbf"; *seed = 42;                        /* 嘿嘿 — sly giggle, universal (ear 2026-07-07) */
    } else if (!strcasecmp(tag, "scoff")) {
        *onom = "\xe5\x88\x87"; *seed = 7; *temp = 1.0f;                       /* 切 — disdain/scoff; T1.0 (1.1 over-drives pitch); pair w/ --emotion disgust */
    } else if (!strcasecmp(tag, "phew")) {
        if (voice_class == 2) { *onom = "\xe5\x94\x89"; *seed = 42; }          /* clone READS 呼 → graceful fallback to sad-sigh 唉 */
        else { *onom = "\xe5\x91\xbc"; *seed = (voice_class == 1) ? 42 : 7; }  /* 呼 — relief exhale; preset (vivian s42 / ryan s7) */
    } else if (!strcasecmp(tag, "huff")) {
        *onom = "\xe5\x97\xa4"; *seed = 7;                                     /* 嗤 — "uff uff" tired/exertion huff, universal (ear 2026-07-07) */
    }
}

int qwen_compose_is_para_event_tag(const char *t) {
    const char *o; int s; float tf; para_pick(t, 0, &o, &s, &tf); return o != NULL;
}

char *qwen_compose_para_substitute(const char *text, int voice_class, int *did, int *seed, float *temp) {
    *did = 0; *temp = 1.1f;
    if (!text) return NULL;
    size_t cap = strlen(text) + 48, n = 0;
    char *out = (char *)malloc(cap);
    if (!out) return NULL;
    #define PIS_ENS(extra) do { while (n + (extra) + 1 > cap) { cap *= 2; char *nb = (char *)realloc(out, cap); if (!nb) { free(out); return NULL; } out = nb; } } while (0)
    const char *p = text;
    while (*p) {
        if (*p == '[') {
            const char *c = strchr(p, ']');
            if (c) {
                size_t tl = (size_t)(c - p - 1);
                char tag[32];
                if (tl < sizeof(tag)) {
                    memcpy(tag, p + 1, tl); tag[tl] = 0;
                    char *t = tag; while (*t == ' ') t++;
                    char *te = t + strlen(t); while (te > t && te[-1] == ' ') *--te = 0;
                    const char *onom; int sd; float td;
                    para_pick(t, voice_class, &onom, &sd, &td);
                    if (onom) {
                        /* comma BEFORE: strip trailing spaces + one comma in out, then emit ", " */
                        while (n > 0 && out[n - 1] == ' ') n--;
                        if (n > 0 && out[n - 1] == ',') n--;
                        size_t ol = strlen(onom);
                        PIS_ENS(ol + 4);
                        if (n > 0) { out[n++] = ','; out[n++] = ' '; }         /* not at sentence start */
                        memcpy(out + n, onom, ol); n += ol;
                        out[n++] = ','; out[n++] = ' ';                        /* comma AFTER (the pause) */
                        if (!*did) { *seed = sd; *temp = td; }                 /* pin the first tag's seed + temp */
                        *did = 1;
                        p = c + 1;
                        while (*p == ' ') p++;                                 /* absorb following spaces */
                        if (*p == ',') p++;                                    /* and a redundant comma */
                        while (*p == ' ') p++;
                        continue;
                    }
                }
            }
        }
        PIS_ENS(1);
        out[n++] = *p++;
    }
    #undef PIS_ENS
    out[n] = 0;
    return out;
}

/* ── Span list helpers ────────────────────────────────────────────────── */
static int cspan_push(qwen_cspan_t **arr, int *n, int *cap, qwen_cspan_t s) {
    if (*n >= *cap) {
        int nc = *cap * 2 + 8;
        qwen_cspan_t *t = (qwen_cspan_t *)realloc(*arr, (size_t)nc * sizeof(qwen_cspan_t));
        if (!t) return -1;
        *arr = t; *cap = nc;
    }
    (*arr)[(*n)++] = s;
    return 0;
}

/* "400ms"/"1s"/"0.5s"/"0.5" -> seconds */
static float parse_duration_s(const char *s) {
    while (*s == ' ') s++;
    float v = (float)atof(s);
    const char *u = s;
    while (*u && ((*u >= '0' && *u <= '9') || *u == '.' || *u == '+' || *u == '-')) u++;
    while (*u == ' ') u++;
    if (strncasecmp(u, "ms", 2) == 0) return v / 1000.0f;
    return v;  /* "s" or bare = seconds */
}

int qwen_compose_parse(const char *input, qwen_cspan_t **out, int *out_n) {
    qwen_cspan_t *arr = NULL; int n = 0, cap = 0;
    char cur_mood[48] = "";
    char *seg = (char *)malloc(strlen(input) + 1);
    if (!seg) return -1;
    int seglen = 0;
    #define MK_FLUSH() do {                                                      \
        int _a = 0, _b = seglen;                                                 \
        while (_a < _b && (seg[_a]==' '||seg[_a]=='\t'||seg[_a]=='\n')) _a++;     \
        while (_b > _a && (seg[_b-1]==' '||seg[_b-1]=='\t'||seg[_b-1]=='\n')) _b--; \
        if (_b > _a) {                                                           \
            qwen_cspan_t _s; _s.is_pause = 0; _s.pause_s = 0; _s.is_filler = 0;   \
            _s.steer_weight = -1.0f; _s.rate = 0; _s.volume = 0;                  \
            snprintf(_s.mood, sizeof(_s.mood), "%s", cur_mood);                   \
            _s.text = (char *)malloc((size_t)(_b - _a) + 1);                      \
            if (!_s.text) { free(seg); free(arr); return -1; }                    \
            memcpy(_s.text, seg + _a, (size_t)(_b - _a)); _s.text[_b - _a] = 0;   \
            if (cspan_push(&arr, &n, &cap, _s) != 0) { free(_s.text); free(seg); free(arr); return -1; } \
        }                                                                        \
        seglen = 0;                                                              \
    } while (0)

    for (const char *p = input; *p; ) {
        if (*p == '|') { MK_FLUSH(); p++; continue; }
        if (*p == '[') {
            const char *close = strchr(p, ']');
            if (close) {
                size_t tl = (size_t)(close - p - 1);
                char tag[64];
                if (tl < sizeof(tag)) {
                    memcpy(tag, p + 1, tl); tag[tl] = 0;
                    char *t = tag; while (*t == ' ') t++;
                    char *te = t + strlen(t); while (te > t && te[-1] == ' ') *--te = 0;
                    int handled = 0;

                    if (strncasecmp(t, "pause", 5) == 0 || strncasecmp(t, "break", 5) == 0) {
                        const char *col = strchr(t, ':'); const char *eq = strchr(t, '=');
                        const char *num = col ? col + 1 : (eq ? eq + 1 : t + 5);
                        MK_FLUSH();
                        qwen_cspan_t s; memset(&s, 0, sizeof(s)); s.is_pause = 1; s.pause_s = parse_duration_s(num);
                        if (cspan_push(&arr, &n, &cap, s) != 0) { free(seg); free(arr); return -1; }
                        handled = 1;
                    } else if ((t[0] >= '0' && t[0] <= '9') || t[0] == '.') {
                        MK_FLUSH();
                        qwen_cspan_t s; memset(&s, 0, sizeof(s)); s.is_pause = 1; s.pause_s = parse_duration_s(t);
                        if (cspan_push(&arr, &n, &cap, s) != 0) { free(seg); free(arr); return -1; }
                        handled = 1;
                    } else {
                        /* [laugh]/[sigh] are handled INLINE upstream (qwen_compose_para_substitute) before
                         * compose, so they don't reach here from --text. COMPOSE_MACROS = the old DSP soft
                         * fillers ([huff]/[ugh]/[hmm]/…). */
                        for (int m = 0; COMPOSE_MACROS[m].tag && !handled; m++) {
                            if (strcasecmp(t, COMPOSE_MACROS[m].tag) == 0) {
                                MK_FLUSH();
                                qwen_cspan_t s; s.is_pause = 0; s.pause_s = 0; s.is_filler = 1;
                                s.steer_weight = COMPOSE_MACROS[m].steer_weight;
                                s.rate = COMPOSE_MACROS[m].rate;
                                s.volume = COMPOSE_MACROS[m].volume;
                                s.mood[0] = 0;  /* macros are no-steer prosody; mood unused */
                                s.text = strdup(COMPOSE_MACROS[m].text);
                                if (!s.text || cspan_push(&arr, &n, &cap, s) != 0) { free(s.text); free(seg); free(arr); return -1; }
                                handled = 1;
                            }
                        }
                        if (!handled) {
                            if (strcasecmp(t, "neutral") == 0 || strcasecmp(t, "none") == 0 || strcasecmp(t, "normal") == 0) {
                                MK_FLUSH(); cur_mood[0] = 0; handled = 1;
                            } else if (qwen_emotion_lookup(t)) {
                                MK_FLUSH(); snprintf(cur_mood, sizeof(cur_mood), "%s", t); handled = 1;
                            }
                        }
                    }
                    if (handled) { p = close + 1; continue; }
                }
            }
            seg[seglen++] = *p++;   /* unrecognized -> literal '[' */
            continue;
        }
        seg[seglen++] = *p++;
    }
    MK_FLUSH();
    free(seg);
    #undef MK_FLUSH
    *out = arr; *out_n = n;
    return 0;
}

void qwen_compose_free_spans(qwen_cspan_t *spans, int n) {
    if (!spans) return;
    for (int i = 0; i < n; i++) if (!spans[i].is_pause) free(spans[i].text);
    free(spans);
}

/* Quick scan: does `text` contain at least one RECOGNIZED inline tag? (auto-route) */
int qwen_compose_has_markup(const char *text) {
    for (const char *p = strchr(text, '['); p; p = strchr(p + 1, '[')) {
        const char *c = strchr(p, ']');
        if (!c) continue;
        size_t tl = (size_t)(c - p - 1);
        char tag[64];
        if (tl >= sizeof(tag)) continue;
        memcpy(tag, p + 1, tl); tag[tl] = 0;
        char *t = tag; while (*t == ' ') t++;
        char *te = t + strlen(t); while (te > t && te[-1] == ' ') *--te = 0;
        if (strncasecmp(t, "pause", 5) == 0 || strncasecmp(t, "break", 5) == 0) return 1;
        if ((t[0] >= '0' && t[0] <= '9') || t[0] == '.') return 1;
        if (strcasecmp(t, "neutral") == 0 || strcasecmp(t, "none") == 0 || strcasecmp(t, "normal") == 0) return 1;
        if (qwen_compose_is_para_event_tag(t)) return 1;
        for (int m = 0; COMPOSE_MACROS[m].tag; m++) if (strcasecmp(t, COMPOSE_MACROS[m].tag) == 0) return 1;
        if (qwen_emotion_lookup(t)) return 1;
    }
    return 0;
}

int qwen_compose_has_para_event(const char *text) {
    for (const char *p = strchr(text, '['); p; p = strchr(p + 1, '[')) {
        const char *c = strchr(p, ']');
        if (!c) continue;
        size_t tl = (size_t)(c - p - 1);
        char tag[64];
        if (tl >= sizeof(tag)) continue;
        memcpy(tag, p + 1, tl); tag[tl] = 0;
        char *t = tag; while (*t == ' ') t++;
        char *te = t + strlen(t); while (te > t && te[-1] == ' ') *--te = 0;
        if (qwen_compose_is_para_event_tag(t)) return 1;
        for (int m = 0; COMPOSE_MACROS[m].tag; m++) if (strcasecmp(t, COMPOSE_MACROS[m].tag) == 0) return 1;
    }
    return 0;
}

/* ── Render one span into a fresh audio buffer (per-span emotion + rate/volume DSP) ──
 * *out=malloc'd mono/24k, *out_n=samples. Returns 0 on success. Sets *is_filler for the
 * seam logic in the caller. Forces a cold prefill so the previous span can't leak in. */
static int synth_one_span(qwen_tts_ctx_t *ctx, const qwen_cspan_t *sp, const char *language,
                          int idx, int silent, float **out, int *out_n) {
    const char *mood = sp->mood[0] ? sp->mood : NULL;
    float vol = 1.0f, rate = 1.0f;
    int   sw_set = sp->steer_weight >= 0.0f;                 /* macro -> explicit (0 = no steer) */
    float sw     = sw_set ? sp->steer_weight : 1.0f;
    if (qwen_tts_apply_emotion(ctx, mood, NULL, language, sw, sw_set, 0.0f, 0, 1.0f, 0, 1.0f, 0,
                               &vol, &rate, silent) != 0) return -1;
    if (sp->rate   > 0.0f) rate = sp->rate;                  /* macro filler overrides recipe rate */
    if (sp->volume > 0.0f) vol  = sp->volume;                /* and volume */
    if (!silent) fprintf(stderr, "Span %d: [%s] \"%s\"\n", idx, mood ? mood : "neutral", sp->text);

    /* Each span is an INDEPENDENT synthesis: force a cold prefill so the previous span's
     * KV/trajectory can't leak in via delta-prefill (ear-caught 2026-06-07). */
    ctx->prev_prefill_len = 0;

    float *audio = NULL; int n = 0;
    int grc = qwen_tts_generate(ctx, sp->text, &audio, &n);
    if (grc != 0 || !audio || n <= 0) {
        fprintf(stderr, "Compose: synthesis failed for span %d\n", idx);
        free(audio); return -1;
    }
    float *seg = audio; int seg_n = n; float *stretched = NULL;
    if (rate != 1.0f) {
        int sn = 0;
        if (qwen_audio_time_stretch(audio, n, rate, QWEN_TTS_SAMPLE_RATE, &stretched, &sn) == 0) { seg = stretched; seg_n = sn; }
    }
    if (vol != 1.0f) qwen_audio_apply_gain(seg, seg_n, vol);
    if (stretched) {
        /* keep the stretched buffer, drop the original */
        free(audio);
        *out = stretched; *out_n = seg_n;
    } else {
        *out = audio; *out_n = seg_n;
    }
    return 0;
}

int qwen_compose_render_buffer(qwen_tts_ctx_t *ctx, qwen_cspan_t *spans, int nspans,
                               const char *language, float default_pause,
                               float **out_audio, int *out_n, int silent) {
    const int SR = QWEN_TTS_SAMPLE_RATE;
    float *out = NULL; size_t out_len = 0, out_cap = 0;
    int spoken = 0, idx = 0, last_spoken = 0, prev_filler = 0;
    #define RS_APPEND(src, cnt) do {                                       \
        size_t _c = (cnt);                                                 \
        if (out_len + _c > out_cap) {                                      \
            out_cap = (out_len + _c) * 2 + 1024;                           \
            float *_t = (float *)realloc(out, out_cap * sizeof(float));    \
            if (!_t) { free(out); return -1; }                            \
            out = _t;                                                      \
        }                                                                  \
        if (src) memcpy(out + out_len, (src), _c * sizeof(float));         \
        else memset(out + out_len, 0, _c * sizeof(float));                 \
        out_len += _c;                                                     \
    } while (0)

    for (int i = 0; i < nspans; i++) {
        if (spans[i].is_pause) {
            if (spans[i].pause_s > 0) RS_APPEND(NULL, (size_t)(spans[i].pause_s * SR));
            if (!silent) fprintf(stderr, "  [pause %.2fs]\n", spans[i].pause_s);
            last_spoken = 0; prev_filler = 0;
            continue;
        }
        /* No silence gap around a paralinguistic filler — it should flow OUT of / INTO the speech
         * (crossfade instead, below). */
        int xfade_seam = (spans[i].is_filler || prev_filler);
        if (last_spoken && default_pause > 0 && !xfade_seam) RS_APPEND(NULL, (size_t)(default_pause * SR));

        float *seg = NULL; int seg_n = 0;
        if (synth_one_span(ctx, &spans[i], language, idx, silent, &seg, &seg_n) != 0) { free(out); return -1; }

        /* Crossfade a paralinguistic-filler seam so the laugh/sigh blends into the speech instead of
         * sounding pasted-on. ~45ms equal-gain overlap-add on the existing tail + the new head. */
        if (xfade_seam && out_len > 0 && seg_n > 0) {
            size_t xf = (size_t)(0.045f * SR);
            if (xf > out_len) xf = out_len;
            if (xf > (size_t)seg_n) xf = (size_t)seg_n;
            for (size_t k = 0; k < xf; k++) {
                float a = (float)(xf - k) / (float)xf;          /* fade out previous tail */
                out[out_len - xf + k] = out[out_len - xf + k] * a + seg[k] * (1.0f - a);
            }
            RS_APPEND(seg + xf, (size_t)seg_n - xf);
        } else {
            RS_APPEND(seg, (size_t)seg_n);
        }
        free(seg);
        spoken++; idx++; last_spoken = 1; prev_filler = spans[i].is_filler;
    }
    #undef RS_APPEND

    if (out_len == 0) { fprintf(stderr, "Compose: nothing to synthesize\n"); free(out); return -1; }
    if (!silent) fprintf(stderr, "[composed %d spans, %.2fs]\n", spoken, (double)out_len / SR);
    *out_audio = out; *out_n = (int)out_len;
    return 0;
}

int qwen_compose_render_stream(qwen_tts_ctx_t *ctx, qwen_cspan_t *spans, int nspans,
                               const char *language, float default_pause,
                               qwen_compose_chunk_cb cb, void *user, int silent) {
    const int SR = QWEN_TTS_SAMPLE_RATE;
    int spoken = 0, idx = 0, last_spoken = 0;
    for (int i = 0; i < nspans; i++) {
        if (spans[i].is_pause) {
            if (spans[i].pause_s > 0) {
                int pn = (int)(spans[i].pause_s * SR);
                if (pn > 0) {
                    float *sil = (float *)calloc((size_t)pn, sizeof(float));
                    if (sil) { cb(sil, pn, user); free(sil); }
                }
            }
            if (!silent) fprintf(stderr, "  [pause %.2fs]\n", spans[i].pause_s);
            last_spoken = 0;
            continue;
        }
        /* Emit an inter-span pause as silence (no crossfade across the stream boundary). */
        if (last_spoken && default_pause > 0) {
            int pn = (int)(default_pause * SR);
            if (pn > 0) {
                float *sil = (float *)calloc((size_t)pn, sizeof(float));
                if (sil) { cb(sil, pn, user); free(sil); }
            }
        }
        float *seg = NULL; int seg_n = 0;
        if (synth_one_span(ctx, &spans[i], language, idx, silent, &seg, &seg_n) != 0) return -1;
        if (seg_n > 0) cb(seg, seg_n, user);
        free(seg);
        spoken++; idx++; last_spoken = 1;
    }
    if (spoken == 0) { fprintf(stderr, "Compose: nothing to synthesize\n"); return -1; }
    return 0;
}
