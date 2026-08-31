/* qwen_tts_compose.c — inline expressive-markup composer (see qwen_tts_compose.h). */
#include "qwen_tts_compose.h"
#include "qwen_tts.h"
#include "qwen_tts_emotion.h"
#include "qwen_tts_audio.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>

typedef struct { const char *tag; const char *text; float steer_weight; float rate; float volume; } cmacro_t;
static const cmacro_t COMPOSE_MACROS[] = {
    { "sigh",    "Hah...",    0.0f, 1.12f, 0.67f },
    { "sighs",   "Hah...",    0.0f, 1.12f, 0.67f },
    { "ahh",     "Haaa...",   0.0f, 0.90f, 0.70f },
    { "relief",  "Haaa...",   0.0f, 0.90f, 0.70f },
    { "phew",    "Uao...",    0.0f, 1.00f, 0.82f },
    { "hmm",     "Hmmm...",   0.0f, 0.88f, 0.65f },
    { "mmm",     "\xe5\x97\xaf", 0.0f, 1.00f, 0.85f },
    { "hmpf",    "Hmpf...",   0.0f, 1.00f, 0.75f },
    { "mah",     "Mah...",    0.0f, 0.95f, 0.78f },
    { "uhm",     "Uhm...",    0.0f, 0.95f, 0.72f },
    { "laugh",   "Eheh...",   0.0f, 0.95f, 0.78f },
    { "laughs",  "Eheh...",   0.0f, 0.95f, 0.78f },
    { "haha",    "Haha!",     0.0f, 1.00f, 0.80f },
    { "heh",     "Hehhh...",  0.0f, 0.95f, 0.70f },
    { "ouch",    "Ouch!",     0.0f, 1.00f, 0.85f },
    { "ahi",     "Ahi!",      0.0f, 1.00f, 0.85f },
    { "huff",    "Uff...",    0.0f, 1.00f, 0.78f },
    { "ugh",     "Ugh...",    0.0f, 1.00f, 0.78f },
    { NULL, NULL, 0.0f, 0.0f, 0.0f }
};

static void para_pick(const char *tag, int voice_class, int small_model,
                      const char **onom, int *seed, float *temp) {
    *onom = NULL; *seed = 7; *temp = 1.1f;

    if (small_model) {
        if (!strcasecmp(tag, "laugh") || !strcasecmp(tag, "laughs")) {
            *onom = "\xe5\x93\x88\xe5\x93\x88\xe5\x93\x88"; *seed = 2024;
        } else if (!strcasecmp(tag, "sigh") || !strcasecmp(tag, "sighs")) {
            if (voice_class == 1) { *onom = "ahh"; *seed = 7; }
            else                  { *onom = "\xe5\x94\x89"; *seed = 42; }
        } else if (!strcasecmp(tag, "yawn") || !strcasecmp(tag, "yawns")) {
            *onom = "\xe5\x93\x88\xe5\x95\x8a"; *seed = 42;
        } else if (!strcasecmp(tag, "wow")) {
            *onom = "\xe5\x93\x87"; *seed = 2024;
        } else if (!strcasecmp(tag, "giggle") || !strcasecmp(tag, "giggles")) {
            *onom = "\xe5\x98\xbf\xe5\x98\xbf"; *seed = 42;
        } else if (!strcasecmp(tag, "scoff")) {
            *onom = "\xe5\x88\x87"; *seed = 42; *temp = 1.0f;
        }
        return;
    }

    if (!strcasecmp(tag, "laugh") || !strcasecmp(tag, "laughs")) {
        *onom = "\xe5\x93\x88\xe5\x93\x88\xe5\x93\x88"; *seed = 7;
    } else if (!strcasecmp(tag, "sigh") || !strcasecmp(tag, "sighs")) {
        if (voice_class == 1) { *onom = "ahh"; *seed = 7; }
        else                  { *onom = "\xe5\x94\x89"; *seed = 42; }
    } else if (!strcasecmp(tag, "yawn") || !strcasecmp(tag, "yawns")) {
        *onom = "\xe5\x93\x88\xe5\x95\x8a";
        *seed = (voice_class == 2) ? 42 : 7;
    } else if (!strcasecmp(tag, "wow")) {
        *onom = "\xe5\x93\x87"; *seed = 7;
    } else if (!strcasecmp(tag, "giggle") || !strcasecmp(tag, "giggles")) {
        *onom = "\xe5\x98\xbf\xe5\x98\xbf"; *seed = 42;
    } else if (!strcasecmp(tag, "scoff")) {
        *onom = "\xe5\x88\x87"; *seed = 42; *temp = 1.0f;
    }
}

int qwen_compose_is_para_event_tag(const char *t) {
    const char *o; int s; float tf; para_pick(t, 0, 0, &o, &s, &tf); return o != NULL;
}

char *qwen_compose_para_substitute(const char *text, int voice_class, int small_model,
                                   int *did, int *seed, float *temp) {
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
                    para_pick(t, voice_class, small_model, &onom, &sd, &td);
                    if (onom) {
                        while (n > 0 && out[n - 1] == ' ') n--;
                        if (n > 0 && out[n - 1] == ',') n--;
                        size_t ol = strlen(onom);
                        PIS_ENS(ol + 4);
                        if (n > 0) { out[n++] = ','; out[n++] = ' '; }
                        memcpy(out + n, onom, ol); n += ol;
                        out[n++] = ','; out[n++] = ' ';
                        if (!*did) { *seed = sd; *temp = td; }
                        *did = 1;
                        p = c + 1;
                        while (*p == ' ') p++;
                        if (*p == ',') p++;
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

static float parse_duration_s(const char *s) {
    while (*s == ' ') s++;
    float v = (float)atof(s);
    const char *u = s;
    while (*u && ((*u >= '0' && *u <= '9') || *u == '.' || *u == '+' || *u == '-')) u++;
    while (*u == ' ') u++;
    if (strncasecmp(u, "ms", 2) == 0) return v / 1000.0f;
    return v;
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
            if (!_s.text) { free(seg); qwen_compose_free_spans(arr, n); return -1; } \
            memcpy(_s.text, seg + _a, (size_t)(_b - _a)); _s.text[_b - _a] = 0;   \
            if (cspan_push(&arr, &n, &cap, _s) != 0) { free(_s.text); free(seg); qwen_compose_free_spans(arr, n); return -1; } \
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
                        if (cspan_push(&arr, &n, &cap, s) != 0) { free(seg); qwen_compose_free_spans(arr, n); return -1; }
                        handled = 1;
                    } else if ((t[0] >= '0' && t[0] <= '9') || t[0] == '.') {
                        MK_FLUSH();
                        qwen_cspan_t s; memset(&s, 0, sizeof(s)); s.is_pause = 1; s.pause_s = parse_duration_s(t);
                        if (cspan_push(&arr, &n, &cap, s) != 0) { free(seg); qwen_compose_free_spans(arr, n); return -1; }
                        handled = 1;
                    } else {
                        for (int m = 0; COMPOSE_MACROS[m].tag && !handled; m++) {
                            if (strcasecmp(t, COMPOSE_MACROS[m].tag) == 0) {
                                MK_FLUSH();
                                qwen_cspan_t s; s.is_pause = 0; s.pause_s = 0; s.is_filler = 1;
                                s.steer_weight = COMPOSE_MACROS[m].steer_weight;
                                s.rate = COMPOSE_MACROS[m].rate;
                                s.volume = COMPOSE_MACROS[m].volume;
                                s.mood[0] = 0;
                                s.text = strdup(COMPOSE_MACROS[m].text);
                                if (!s.text || cspan_push(&arr, &n, &cap, s) != 0) { free(s.text); free(seg); qwen_compose_free_spans(arr, n); return -1; }
                                handled = 1;
                            }
                        }
                        if (!handled) {
                            if (strcasecmp(t, "neutral") == 0 || strcasecmp(t, "none") == 0 || strcasecmp(t, "normal") == 0) {
                                MK_FLUSH(); cur_mood[0] = 0; handled = 1;
                            } else if (qwen_emotion_name_to_tok(t)) {
                                MK_FLUSH(); snprintf(cur_mood, sizeof(cur_mood), "%s", t); handled = 1;
                            }
                        }
                    }
                    if (handled) { p = close + 1; continue; }
                }
            }
            seg[seglen++] = *p++;
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
        if (qwen_emotion_name_to_tok(t)) return 1;
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

static int synth_one_span(qwen_tts_ctx_t *ctx, const qwen_cspan_t *sp, const char *language,
                          int idx, int silent, float **out, int *out_n) {
    (void)language;
    const char *mood = sp->mood[0] ? sp->mood : NULL;
    float vol = 1.0f, rate = 1.0f;

    float *saved_ml = ctx->ml_steer;
    int  s_L = ctx->ml_steer_layers, s_D = ctx->ml_steer_dim, s_l0 = ctx->ml_steer_l0, s_l1 = ctx->ml_steer_l1;
    float s_w = ctx->ml_steer_weight;
    int installed = 0;
    if (mood && ctx->config.hidden_size >= 2048) {
        const char *tok = qwen_emotion_name_to_tok(mood);
        if (tok && qwen_emotion_steer_install(ctx, tok, 12.0f, 21, 25, silent) == 0) installed = 1;
    }
    if (sp->rate   > 0.0f) rate = sp->rate;
    if (sp->volume > 0.0f) vol  = sp->volume;
    if (!silent) fprintf(stderr, "Span %d: [%s] \"%s\"\n", idx, mood ? mood : "neutral", sp->text);

    ctx->prev_prefill_len = 0;

    float *audio = NULL; int n = 0;
    int grc = qwen_tts_generate(ctx, sp->text, &audio, &n);

    if (installed) {
        free(ctx->ml_steer);
        ctx->ml_steer = saved_ml; ctx->ml_steer_layers = s_L; ctx->ml_steer_dim = s_D;
        ctx->ml_steer_l0 = s_l0; ctx->ml_steer_l1 = s_l1; ctx->ml_steer_weight = s_w;
    }
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
        int xfade_seam = (spans[i].is_filler || prev_filler);
        if (last_spoken && default_pause > 0 && !xfade_seam) RS_APPEND(NULL, (size_t)(default_pause * SR));

        float *seg = NULL; int seg_n = 0;
        if (synth_one_span(ctx, &spans[i], language, idx, silent, &seg, &seg_n) != 0) { free(out); return -1; }

        if (xfade_seam && out_len > 0 && seg_n > 0) {
            size_t xf = (size_t)(0.045f * SR);
            if (xf > out_len) xf = out_len;
            if (xf > (size_t)seg_n) xf = (size_t)seg_n;
            for (size_t k = 0; k < xf; k++) {
                float a = (float)(xf - k) / (float)xf;
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
