/*
 * main.c - Qwen3-TTS CLI
 */

#include "qwen_tts.h"
#include "qwen_tts_audio.h"
#include "qwen_tts_emotion.h"
#include "qwen_tts_compose.h"   /* inline expressive-markup composer (shared with the server) */
#include "qwen_tts_batch.h"
#include "qwen_tts_kernels.h"
#include "qwen_tts_server.h"
#if defined(QWEN_HAVE_METAL) || defined(QWEN_HAVE_CUDA)
#include "qwen_tts_backend.h"   /* experimental GPU backends (make metal / make cuda) */
#endif
#if defined(QWEN_HAVE_METAL)
#include "qwen_tts_metal.h"     /* fused Talker step + selftest */
#endif

#include <stdio.h>
#include <lz4.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <getopt.h>
#include <math.h>
#include <dirent.h>
#include <sys/stat.h>

/* bf16<->f32 helpers (local; for partial-strength WDELTA interpolation on .qvoice load) */
static inline float main_bf16_to_f32(uint16_t bf) {
    uint32_t bits = (uint32_t)bf << 16; float v; memcpy(&v, &bits, sizeof(v)); return v;
}
static inline uint16_t main_f32_to_bf16(float v) {
    uint32_t bits; memcpy(&bits, &v, sizeof(bits)); return (uint16_t)(bits >> 16);
}

/* TARGETED WDELTA scale: apply base_alpha (<1) ONLY to the deep Talker attn/mlp projection
 * weights (layers [l0,l1]) where the instruct/emotion response is gated. Keep FULL voice
 * (alpha=1) on embeddings, text/codec projections, norms, early identity layers, and ALL
 * code_predictor.* tensors — those carry timbre/gender, and uniform dilution flips them
 * (observed: voice turning female / 'Chinese accent'). */
static float qvoice_tensor_alpha(const char *tname, float base_alpha, int l0, int l1) {
    if (base_alpha == 1.0f) return 1.0f;
    const char *p = strstr(tname, "talker.model.layers.");   /* NOT talker.code_predictor.* */
    if (!p) return 1.0f;
    int layer = -1;
    if (sscanf(p + 20, "%d", &layer) != 1) return 1.0f;       /* 20 = strlen("talker.model.layers.") */
    if (layer < l0 || layer > l1) return 1.0f;                /* identity/early layers -> full voice */
    if ((strstr(tname, ".self_attn.") || strstr(tname, ".mlp.")) && strstr(tname, "_proj.weight"))
        return base_alpha;                                    /* attn/mlp projections only */
    return 1.0f;                                              /* layernorms / q_norm / k_norm */
}

/* Print info about a .qvoice file. Returns 0 on success. */
static int print_qvoice_info(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;

    char magic[4];
    uint32_t version;
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "QVCE", 4) != 0) {
        fclose(f); return -1;
    }
    if (fread(&version, sizeof(uint32_t), 1, f) != 1) {
        fclose(f); return -1;
    }

    /* Read/skip speaker embedding (v2+ has enc_dim header, v1 assumes 1024) */
    uint32_t file_enc_dim = 1024;
    if (version >= 2) {
        if (fread(&file_enc_dim, sizeof(uint32_t), 1, f) != 1) { fclose(f); return -1; }
    }
    fseek(f, file_enc_dim * sizeof(float), SEEK_CUR);

    /* Read ref_text */
    uint32_t ref_text_len = 0;
    if (fread(&ref_text_len, sizeof(uint32_t), 1, f) != 1) { fclose(f); return -1; }
    char ref_text[256] = {0};
    if (ref_text_len > 0) {
        int read_len = ref_text_len < 255 ? (int)ref_text_len : 255;
        if (fread(ref_text, 1, read_len, f) != (size_t)read_len) { fclose(f); return -1; }
        ref_text[read_len] = '\0';
        if (ref_text_len > 255) fseek(f, ref_text_len - 255, SEEK_CUR);
    }

    /* Read n_ref_frames */
    uint32_t n_ref_frames = 0;
    if (fread(&n_ref_frames, sizeof(uint32_t), 1, f) != 1) { fclose(f); return -1; }
    /* Skip ref_codes data */
    if (n_ref_frames > 0)
        fseek(f, (long)n_ref_frames * 16 * sizeof(int), SEEK_CUR);

    /* v3 metadata */
    char meta_lang_name[16] = {0};
    char meta_voice_name[64] = {0};
    float meta_ref_dur = 0;
    uint32_t meta_model_size = 0;
    int has_meta = 0;
    if (version >= 3) {
        char meta_magic[4];
        if (fread(meta_magic, 1, 4, f) == 4 && memcmp(meta_magic, "META", 4) == 0) {
            has_meta = 1;
            uint32_t lang_id;
            fread(&lang_id, sizeof(uint32_t), 1, f);
            fread(meta_lang_name, 1, 16, f);
            meta_lang_name[15] = '\0';
            fread(&meta_model_size, sizeof(uint32_t), 1, f);
            uint32_t enc_dim_meta;
            fread(&enc_dim_meta, sizeof(uint32_t), 1, f);
            fread(&meta_ref_dur, sizeof(float), 1, f);
            fread(meta_voice_name, 1, 64, f);
            meta_voice_name[63] = '\0';
        }
    }

    fclose(f);

    /* File size */
    struct stat st;
    stat(path, &st);

    /* Extract just filename */
    const char *basename = strrchr(path, '/');
    basename = basename ? basename + 1 : path;

    printf("  %-30s  v%u  %3u frames (%.1fs ref)  %5.1f KB",
           basename, version, n_ref_frames, n_ref_frames / 12.5f,
           (float)st.st_size / 1024.0f);
    if (has_meta) {
        if (meta_voice_name[0])
            printf("  [%s]", meta_voice_name);
        if (meta_lang_name[0])
            printf("  lang=%s", meta_lang_name);
        printf("  model=%s", meta_model_size >= 2048 ? "1.7B" : (meta_model_size > 0 ? "0.6B" : "?"));
    }
    if (ref_text_len > 0)
        printf("  \"%s\"", ref_text);
    printf("\n");
    return 0;
}

/* List all .qvoice files in a directory */
static int list_voices(const char *dir_path) {
    DIR *dir = opendir(dir_path);
    if (!dir) {
        /* Maybe it's a single file */
        if (strstr(dir_path, ".qvoice")) {
            printf("Voice profiles:\n");
            if (print_qvoice_info(dir_path) != 0) {
                fprintf(stderr, "Error: %s is not a valid .qvoice file\n", dir_path);
                return 1;
            }
            return 0;
        }
        fprintf(stderr, "Error: cannot open directory %s\n", dir_path);
        return 1;
    }

    printf("Voice profiles in %s:\n\n", dir_path);
    int count = 0;
    struct dirent *entry;
    while ((entry = readdir(dir)) != NULL) {
        size_t len = strlen(entry->d_name);
        if (len > 7 && strcmp(entry->d_name + len - 7, ".qvoice") == 0) {
            char fullpath[4096];
            snprintf(fullpath, sizeof(fullpath), "%s/%s", dir_path, entry->d_name);
            if (print_qvoice_info(fullpath) == 0)
                count++;
        }
    }
    closedir(dir);

    if (count == 0)
        printf("  (no .qvoice files found)\n");
    else
        printf("\n  %d voice profile(s)\n", count);
    return 0;
}

/* Streaming callback state */
typedef struct {
    FILE *file;            /* WAV file or stdout */
    int is_stdout;         /* 1 = raw PCM to stdout, 0 = WAV file */
    int total_samples;     /* running count of samples written */
    float volume;          /* --volume gain applied per chunk (1.0 = none) */
} stream_state_t;

static int stream_audio_callback(const float *samples, int n_samples, void *userdata) {
    stream_state_t *st = (stream_state_t *)userdata;
    if (!st->file) return -1;
    for (int i = 0; i < n_samples; i++) {
        float s = samples[i] * st->volume;
        if (s != s) s = 0.0f;               /* leaks-audit #7: NaN passes both clamps → (int16_t)(NaN*32767) UB */
        if (s < -1.0f) s = -1.0f;
        if (s > 1.0f) s = 1.0f;
        int16_t sample = (int16_t)(s * 32767);
        fwrite(&sample, 2, 1, st->file);
    }
    fflush(st->file);
    st->total_samples += n_samples;
    return 0;
}

/* Write a WAV header with placeholder data size (will be updated at end) */
static void write_wav_header(FILE *f, int sample_rate) {
    /* Leaks-audit fix (#4): WAV size fields are uint32 LE. The old code used signed int with a
     * 0x7FFFFFFF placeholder, so `file_size = 36 + 0x7FFFFFFF` overflowed to a negative int (UB).
     * Use uint32 and the conventional unknown-length streaming placeholder; finalize_wav_header
     * overwrites both with the real (clamped) sizes when the stream ends. */
    uint16_t bits = 16, channels = 1;
    uint16_t block_align = channels * (bits/8);
    uint16_t audio_fmt = 1;
    uint32_t data_size = 0xFFFFFFFFu;            /* placeholder for unknown stream length */
    uint32_t file_size = 0xFFFFFFFFu;
    uint32_t byte_rate = (uint32_t)sample_rate * channels * (bits/8);
    uint32_t fmt_size = 16;
    uint32_t sr = (uint32_t)sample_rate;
    fwrite("RIFF", 1, 4, f);
    fwrite(&file_size, 4, 1, f);
    fwrite("WAVEfmt ", 1, 8, f);
    fwrite(&fmt_size, 4, 1, f);
    fwrite(&audio_fmt, 2, 1, f);
    fwrite(&channels, 2, 1, f);
    fwrite(&sr, 4, 1, f);
    fwrite(&byte_rate, 4, 1, f);
    fwrite(&block_align, 2, 1, f);
    fwrite(&bits, 2, 1, f);
    fwrite("data", 1, 4, f);
    fwrite(&data_size, 4, 1, f);
}

/* Update WAV header with actual data size. Clamps to 0xFFFFFFFF so a >4GB / >~12h stream can't
 * wrap the uint32 size fields (or overflow signed int as the old `total_samples*2` did). */
static void finalize_wav_header(FILE *f, int total_samples) {
    uint64_t data_size = (uint64_t)(unsigned)total_samples * 2u;  /* 16-bit mono */
    if (data_size > 0xFFFFFFFFu) data_size = 0xFFFFFFFFu;
    uint64_t file_size = 36u + data_size;
    if (file_size > 0xFFFFFFFFu) file_size = 0xFFFFFFFFu;
    uint32_t ds = (uint32_t)data_size, fs = (uint32_t)file_size;
    fseek(f, 4, SEEK_SET);
    fwrite(&fs, 4, 1, f);
    fseek(f, 40, SEEK_SET);
    fwrite(&ds, 4, 1, f);
}

/* Write a tensor to .qvoice file, optionally as WDELTA (int8 compressed delta vs CV) */
static void write_tensor_impl(FILE *vf, FILE *cv_sf, const char *cv_hdr_json, size_t cv_data_off,
                               const char *tname, const void *ptr, size_t nbytes,
                               int use_wdelta, int is_bf16,
                               int64_t *total_bytes, int *count) {
    uint16_t nl = (uint16_t)strlen(tname);
    fwrite(&nl, sizeof(uint16_t), 1, vf);
    fwrite(tname, 1, nl, vf);
    uint32_t raw = (uint32_t)nbytes;
    fwrite(&raw, sizeof(uint32_t), 1, vf);

    if (use_wdelta && is_bf16 && cv_sf && cv_hdr_json) {
        /* Find this tensor in CV safetensors */
        char key[300];
        snprintf(key, sizeof(key), "\"%s\"", tname);
        const char *p = strstr(cv_hdr_json, key);
        long cvs = -1, cve = -1;
        if (p) {
            const char *doff = strstr(p, "data_offsets");
            if (doff) {
                const char *br = strchr(doff, '[');
                if (br) sscanf(br, "[%ld,%ld]", &cvs, &cve);
            }
        }
        if (cvs >= 0 && (uint32_t)(cve - cvs) == raw) {
            /* Read CV tensor */
            uint8_t *cv_buf = (uint8_t *)malloc(raw);
            fseek(cv_sf, cv_data_off + cvs, SEEK_SET);
            fread(cv_buf, 1, raw, cv_sf);
            /* Compute int16 delta (lossless — no clamping) */
            size_t n16 = raw / 2;
            int16_t *delta = (int16_t *)malloc(n16 * sizeof(int16_t));
            const uint16_t *base16 = (const uint16_t *)ptr;
            const uint16_t *cv16 = (const uint16_t *)cv_buf;
            for (size_t i = 0; i < n16; i++)
                delta[i] = (int16_t)((int)base16[i] - (int)cv16[i]);
            unsigned long delta_bytes = n16 * sizeof(int16_t);
            /* Compress delta with LZ4 (~7x faster decompress than zlib) */
            int lz4_bound = LZ4_compressBound((int)delta_bytes);
            uint8_t *compressed = (uint8_t *)malloc(lz4_bound);
            int comp_size = LZ4_compress_default((const char *)delta, (char *)compressed,
                                                  (int)delta_bytes, lz4_bound);
            /* dtype=4: int16 delta + LZ4 */
            uint8_t dtype = 4;
            fwrite(&dtype, 1, 1, vf);
            uint32_t csz = (uint32_t)comp_size;
            fwrite(&csz, sizeof(uint32_t), 1, vf);
            fwrite(compressed, 1, comp_size, vf);
            *total_bytes += comp_size;
            free(cv_buf); free(delta); free(compressed);
        } else {
            /* Fallback: raw */
            uint8_t dtype = 0;
            fwrite(&dtype, 1, 1, vf);
            uint32_t csz = raw;
            fwrite(&csz, sizeof(uint32_t), 1, vf);
            fwrite(ptr, 1, raw, vf);
            *total_bytes += raw;
        }
    } else {
        /* WFULL mode or F32 tensor: write raw */
        uint8_t dtype = 0;
        fwrite(&dtype, 1, 1, vf);
        uint32_t csz = raw;
        fwrite(&csz, sizeof(uint32_t), 1, vf);
        fwrite(ptr, 1, raw, vf);
        *total_bytes += raw;
    }
    /* ferror is sticky → one check catches a short write from any fwrite above
     * (e.g. disk full) instead of silently truncating the voice file. */
    if (ferror(vf))
        fprintf(stderr, "Error: short write while saving tensor '%s' (disk full?)\n", tname);
    (*count)++;
}

/* qwen_tts_apply_emotion (the --emotion qlsteer STEER router) lives in
 * qwen_tts_emotion.c so the CLI and the HTTP server apply the SAME recipe. */

/* Load a multi-layer Talker steer (.qlsteer: 'QLST' magic + int32 L + int32 D + L*D float32)
 * into a FRESH buffer (does NOT touch ctx state). L must be num_layers+1, D = hidden.
 * Caller owns *out_buf. Used both for the global --ml-steer and per-span paralinguistic vectors. */
static int load_qlsteer_buf(qwen_tts_ctx_t *ctx, const char *path, float **out_buf, int *out_L, int *out_D) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Error: cannot open ml-steer '%s'\n", path); return -1; }
    uint32_t magic = 0; int32_t L = 0, D = 0;
    int want_L = ctx->config.num_layers + 1, want_D = ctx->config.hidden_size;
    if (fread(&magic, 4, 1, f) != 1 || fread(&L, 4, 1, f) != 1 || fread(&D, 4, 1, f) != 1 ||
        magic != 0x54534C51u /* 'QLST' */ || L != want_L || D != want_D) {
        fprintf(stderr, "Error: '%s' invalid ml-steer (L=%d D=%d, expected %d x %d)\n",
                path, L, D, want_L, want_D);
        fclose(f); return -1;
    }
    size_t n = (size_t)L * D;
    float *buf = (float *)malloc(n * sizeof(float));
    if (!buf || fread(buf, sizeof(float), n, f) != n) {
        fprintf(stderr, "Error: failed to read ml-steer '%s'\n", path); free(buf); fclose(f); return -1;
    }
    fclose(f);
    *out_buf = buf; *out_L = L; *out_D = D;
    return 0;
}

/* Load a .qlsteer into ctx->ml_steer (the global --ml-steer). */
static int load_ml_steer(qwen_tts_ctx_t *ctx, const char *path, float weight, int l0, int l1) {
    float *buf = NULL; int L = 0, D = 0;
    if (load_qlsteer_buf(ctx, path, &buf, &L, &D) != 0) return -1;
    free(ctx->ml_steer);
    ctx->ml_steer = buf; ctx->ml_steer_layers = L; ctx->ml_steer_dim = D;
    ctx->ml_steer_weight = weight;
    ctx->ml_steer_l0 = l0 < 0 ? 0 : l0;
    ctx->ml_steer_l1 = l1 >= L ? L - 1 : l1;
    fprintf(stderr, "ML-steer: %s (L%d-%d, weight %.1f)\n", path, ctx->ml_steer_l0, ctx->ml_steer_l1, weight);
    return 0;
}

/* resolve_emotion_path + qwen_apply_emotion moved to qwen_tts_emotion.c
 * (shared by CLI + server). CLI calls qwen_tts_apply_emotion(). */

/* ---- Inline expressive markup (ElevenLabs/Bark-style square-bracket tags) ----
 * One text body; tags are ENGLISH and inline, switchable mid-text:
 *   [happy] [sad] [excited] [annoyed] [proud] [calm] ...  emotion for following text
 *   [neutral]                                  back to plain delivery
 *   [sigh] [sighs] [huff] [ugh] [groan] [hmm]  paralinguistic fillers (generated)
 *   [pause:400ms] [pause:1s] [pause:0.5]       a pause (also [break:...] or bare [0.5])
 * Unrecognized [..] tags are kept as literal text. A legacy '|' is a hard span break.
 * Used by --compose AND auto-detected inside --text. Each span is synthesized with
 * its own recipe and concatenated (model-generated -> seamless, same voice/codec). */

/* The span type + the paralinguistic soft-filler macro table now live in qwen_tts_compose.c
 * (shared with the server). `cspan_t` is the local alias kept for the batch/compose call sites. */
typedef qwen_cspan_t cspan_t;

/* Paralinguistic events rendered via a STEERING VECTOR (the validated win, plan §8.9-DONE/§9.13):
 * the inline [tag] becomes a native-trigger onomatopoeia ANCHOR (which the model performs) and the
 * vector SHAPES it into the real event. VOCAL family only (laugh/sigh — articulatory events hit the
 * decoder ceiling). RAW vectors, inject L21-25. Default weight ~8 (ryan over-steers a bit at 8, wants
 * ~6). Checked BEFORE COMPOSE_MACROS so laugh/sigh use the vector path, not the old DSP filler. */
/* NOTE (2026-07-01): the paralinguistic STEERING-VECTOR split-span ("splice") for [laugh]/[sigh] was
 * REMOVED — it rendered the event as a separate cold-prefill span with its own vector, which mixed
 * voices (sounded like a different speaker on clones). Replaced by INLINE substitution (para_pick /
 * para_inline_substitute above): the tag becomes a validated onomatopoeia IN the sentence, one
 * generation, event in the active voice's own timbre. Do NOT reintroduce the split-span. */

/* INLINE paralinguistics — the shipped method (2026-07-01, docs/para-experiments.md). A `[laugh]`/`[sigh]`
 * is replaced by a validated ONOMATOPOEIA *inside* the sentence, so the event is produced in the active
 * voice's own timbre within ONE generation — NEVER a separate "splice" span (which mixed voices). The
 * mapping is universal across voices AND languages (ear-validated on ryan EN/IT, vivian IT, galatea clone):
 *   [laugh]→哈哈哈  [sigh]→唉/ahh  [yawn]→哈啊(preset s7/clone s42)  [wow]→哇 s7  [giggle]→嘿嘿 s42  [scoff]→切 s42(T1.0)
 * and seed 7 makes laugh fire (哈哈哈 s7 laughs / s42 hyperventilates). SHORT form only (哈哈哈 not longer;
 * long over-laughs into a pant); no event-instruct (goes metallic). [yawn] added 2026-07-07 via the E1
 * discovery harness. Robustness gate 2026-07-08: [wow]/[yawn]/[scoff] SHIP (scoff re-pinned s7→s42);
 * [giggle] SHIPS standalone-only (do NOT stack with --emotion joy — over-laughs); [phew] PARKED (only IT
 * clean, metallic/literal on EN). See the doc for the full WIN/KO trail + the ryan-only/parked events. */
/* para_pick / para_inline_substitute / is_para_event_tag moved to qwen_tts_compose.c
 * (shared with the server). Use qwen_compose_para_substitute / qwen_compose_is_para_event_tag. */

/* ── --emotion auto-router (1.7B): the ear-validated per-(voice×emotion) recipe ────────────────
 * `--emotion <name>` reproduces the weeks-long shippable recipe table (plan_emo_v3 §8.3 recipe_final,
 * CLEAN+decay default, user-validated 2026-06-24): ONE flag instead of hand-wiring --expr/--ml-steer.
 * CRUCIAL: it is NOT "always combine". Each (voice,emotion) cell has a specific MODE —
 *   EXPR-only  (e.g. ryan/vivian anger — steer goes metallic, the FT carries it),
 *   STEER-only (most cells — the clean steer @ L21-25 IS the emotion; adding expr SOFTENS it → e.g.
 *               galatea anger went "sad" when expr was wrongly added),
 *   COMBINE    (ryan/vivian joy, galatea disgust/fear — expr renders + steer pushes).
 * Plus the PRIMARY levers (memory): a default English instruct + temp 1.1(preset)/1.3(clone), set early.
 * Cross-language (§8.6): for FAR languages (ZH/JA/KO/RU) the IT expr is added as a stabilizer even on
 * STEER cells. Manual --expr/--ml-steer override. 0.6B keeps the legacy .vec path. */
/* Canonical filename token for an --emotion spec, or NULL if it isn't a routed emotion.
 * Single source of truth = qwen_tts_emotion.c (the name/dyad table), shared with the compose
 * per-span path and the server. */
static const char *emotion_tok(const char *spec) {
    return qwen_emotion_name_to_tok(spec);
}
/* The shippable per-(voice×emotion) recipe (plan §8.3). use_expr/use_steer pick the cell's MODE. */
typedef struct { const char *voice; const char *tok; int use_expr; float expr_w; int use_steer; float steer_w; } emo_cell_t;
/* THE emotion recipe (docs/emotion-THE-recipe.md, ear-validated 2026-06-29 across ALL languages):
 * pure STEER WINS everywhere, clean, timbre intact — `ryan_<emo>` @ **w12** (w10 also good) — so the recipe is
 * DEAD SIMPLE. The earlier per-(voice×emotion) cells and per-language EXPR/COMBINE policy are SUPERSEDED.
 *   PRESET voice → STEER ryan_<emo> @ w12 (no expr, no instruct). Use the NATIVE preset per language.
 *   CLONE  voice → COMBINE: the language .expr @1.0 renders/stabilizes + STEER @ w12 + EN instruct
 *                  (the one easy clone recipe — expr keeps a cross-language clone clean).
 * anger/fear confirmed best at w12; the rest win at w10 or w12 → w12 everywhere is the single safe default. */
static void resolve_emotion_recipe(const char *language, const char *voice_key, int is_clone,
                                   const char *tok, emo_cell_t *cell, float *temp) {
    (void)language; (void)voice_key;
    *temp = 1.1f;
    if (is_clone)
        *cell = (emo_cell_t){ "*", tok, 1, 1.0f, 1, 12.0f };   /* clone: COMBINE (expr renders the language) + steer w12 */
    else
        *cell = (emo_cell_t){ "*", tok, 0, 0.0f, 1, 12.0f };   /* preset: STEER w12 — clean, every language */
}
/* Recommended NATIVE preset per language (hint when the user picked a weaker voice). Qwen presets:
 * ryan (EN → Romance), vivian/uncle_fu (ZH), ono_anna (JA), sohee (KO). NULL = no strong preference. */
static const char *recommended_voice_for_language(const char *language) {
    if (!language) return NULL;
    if (!strcasecmp(language, "Japanese")) return "ono_anna";
    if (!strcasecmp(language, "Korean"))   return "sohee";
    if (!strcasecmp(language, "Chinese"))  return "vivian";
    if (!strcasecmp(language, "German") || !strcasecmp(language, "French") || !strcasecmp(language, "Spanish"))
        return "vivian";
    if (!strcasecmp(language, "Italian") || !strcasecmp(language, "English") || !strcasecmp(language, "Portuguese"))
        return "ryan";
    return NULL;
}
/* Pick the .qlsteer for an emotion. The validated recipe (recipe_final.sh) uses the **ryan_<tok>**
 * CLEAN palette for ALL voices incl. clones — the `*_ft` voice-native dirs were an idea-2 experiment
 * that did NOT win (galatea_ang_ft is weaker/sad-ish, captured WITH expr). So: always ryan_<tok>. */
static int resolve_emotion_qlsteer(const char *voice_key, const char *tok, char *out, size_t outsz) {
    (void)voice_key;
    snprintf(out, outsz, "presets/steer/emotion/ryan_%s.qlsteer", tok);
    FILE *f = fopen(out, "rb"); if (f) { fclose(f); return 0; }
    return -1;
}
/* Per-language expr (FT): native DE/FR where trained, else the IT pack as the cross-language renderer. */
static const char *resolve_emotion_expr(const char *language) {
    if (language && strcasecmp(language, "German") == 0) return "presets/expr/german_csp_k6.expr";
    if (language && strcasecmp(language, "French") == 0) return "presets/expr/french_csp_k6.expr";
    return "presets/expr/italian_csp_topk6.expr";
}
/* Derive a voice key for the steer palette: clone basename ("voices/galatea_graft.qvoice" -> "galatea")
 * or the preset speaker name. Returns a pointer into `buf` (clone) or `speaker_name` (preset), or NULL. */
static const char *emotion_voice_key(int is_clone, const char *load_voice, const char *speaker_name,
                                     char *buf, size_t bufsz) {
    if (is_clone && load_voice) {
        const char *b = strrchr(load_voice, '/'); b = b ? b + 1 : load_voice;
        size_t n = 0; while (b[n] && b[n] != '.' && n < bufsz - 1) { buf[n] = b[n]; n++; }
        buf[n] = '\0';
        char *us;
        if ((us = strstr(buf, "_graft"))) *us = '\0';   /* galatea_graft -> galatea */
        if ((us = strstr(buf, "_06b")))   *us = '\0';
        return buf[0] ? buf : NULL;
    }
    return speaker_name;   /* preset name (NULL = default → ryan palette) */
}
/* Default vivid ENGLISH instruct per routed emotion. The instruct (in English/Chinese, the model's
 * training languages) is the PRIMARY emotion lever (memory: temp + EN-instruct >> steer alone) — without
 * it `--emotion` renders flat. The model follows it verbatim while speaking the target `-l` language.
 * Returns NULL for non-routed moods. A user `--instruct` always overrides this. */
static const char *default_emotion_instruct(const char *spec) {
    const char *tok = emotion_tok(spec);
    if (!tok) return NULL;
    /* The exact validated instructs from recipe_final.sh (ear-validated 2026-06-24). */
    if (!strcmp(tok, "sad"))      return "Speak in a sad, sorrowful, gloomy and downcast tone, voice low and heavy, on the verge of tears.";
    if (!strcmp(tok, "joy"))      return "Speak with bright, radiant joy, light and warm, smiling through every word.";
    if (!strcmp(tok, "ang"))      return "Speak in a furious, seething, enraged tone, voice sharp and hard, barely holding back the rage.";
    if (!strcmp(tok, "fear"))     return "Speak in a frightened, trembling, anxious tone, voice shaky and breathless with dread.";
    if (!strcmp(tok, "disgust"))  return "Speak with deep disgust and revulsion, lip-curling contempt, as if something repels you.";
    if (!strcmp(tok, "surprise")) return "Speak with sudden astonishment and surprise, gasping and caught off guard.";
    return NULL;
}

static int render_spans(qwen_tts_ctx_t *ctx, cspan_t *spans, int nspans,
                        const char *language, float default_pause,
                        const char *output, int silent);   /* fwd (defined below) */

/* ============================================================================
 * Long-text BATCHING: sentence-aware chunk splitter (--batch).
 *
 * Top-player practice for long-form TTS: segment on sentence boundaries, then
 * greedily PACK sentences into chunks up to a word budget (so chunks are balanced
 * and not micro-fragments). Seams fall on sentence-ending pauses -> inaudible.
 * v1: greedy packing only; an over-long single sentence becomes its own chunk
 * (comma sub-split is a TODO). Used by Milestone A (split + sequential synth +
 * concat, reusing render_spans) and later by the batched-compute path.
 * ============================================================================ */
static int qwen_word_count(const char *s, int len) {
    int n = 0, in_word = 0;
    for (int i = 0; i < len; i++) {
        int sp = (s[i] == ' ' || s[i] == '\t' || s[i] == '\n' || s[i] == '\r');
        if (!sp && !in_word) { n++; in_word = 1; } else if (sp) in_word = 0;
    }
    return n;
}
/* Is the '.' at text[dot] part of an abbreviation/initial (not a sentence end)? */
static int qwen_is_abbrev_dot(const char *text, int dot) {
    int s = dot;
    while (s > 0 && !(text[s-1] == ' ' || text[s-1] == '\n' || text[s-1] == '\t')) s--;
    int tl = dot - s;
    if (tl <= 0 || tl > 4) return 0;
    char buf[8]; memcpy(buf, text + s, (size_t)tl); buf[tl] = 0;
    static const char *ab[] = { "Sig","Sigg","Dott","Dr","Prof","Egr","On","Rev",
                                "St","vs","pag","art","ecc","etc","es","cfr",NULL };
    for (int i = 0; ab[i]; i++) if (strcasecmp(buf, ab[i]) == 0) return 1;
    if (tl == 1 && text[s] >= 'A' && text[s] <= 'Z') return 1;   /* "A." initial */
    return 0;
}
static int qwen_push_str(char ***arr, int *n, int *cap, const char *s, int len) {
    /* trim leading/trailing whitespace */
    while (len > 0 && (*s == ' ' || *s == '\t' || *s == '\n' || *s == '\r')) { s++; len--; }
    while (len > 0 && (s[len-1] == ' ' || s[len-1] == '\t' || s[len-1] == '\n' || s[len-1] == '\r')) len--;
    if (len <= 0) return 0;
    if (*n >= *cap) { int nc = *cap ? *cap * 2 : 16;
        char **t = (char **)realloc(*arr, (size_t)nc * sizeof(char *)); if (!t) return -1; *arr = t; *cap = nc; }
    char *c = (char *)malloc((size_t)len + 1); if (!c) return -1;
    memcpy(c, s, (size_t)len); c[len] = 0; (*arr)[(*n)++] = c;
    return 0;
}
/* Segment `text` into sentences (caller frees each + array). */
static int qwen_split_sentences(const char *text, char ***out, int *out_n) {
    int len = (int)strlen(text);
    char **arr = NULL; int n = 0, cap = 0, start = 0;
    for (int i = 0; i < len; i++) {
        char ch = text[i];
        int is_end = (ch == '.' || ch == '!' || ch == '?' || ch == ';' || ch == '\n');
        if (ch == '.') {
            if (i > 0 && i + 1 < len && isdigit((unsigned char)text[i-1]) && isdigit((unsigned char)text[i+1])) is_end = 0;
            if (qwen_is_abbrev_dot(text, i)) is_end = 0;
        }
        if (is_end) {
            int e = i + 1;
            while (e < len && (text[e] == '"' || text[e] == '\'' || text[e] == ')' || text[e] == ']')) e++;
            if (qwen_push_str(&arr, &n, &cap, text + start, e - start) < 0) { goto oom; }
            start = e; i = e - 1;
        }
    }
    if (start < len) if (qwen_push_str(&arr, &n, &cap, text + start, len - start) < 0) goto oom;
    *out = arr; *out_n = n; return 0;
oom:
    for (int i = 0; i < n; i++) free(arr[i]); free(arr); return -1;
}
/* Split long text into chunks: pack sentences up to ~target_words, merge a
 * trailing sub-min_words fragment into the previous chunk. */
static int qwen_split_text_for_batch(const char *text, int target_words, int min_words,
                                     char ***out, int *out_n) {
    char **sent = NULL; int ns = 0;
    if (qwen_split_sentences(text, &sent, &ns) != 0) return -1;
    char **chunks = NULL; int nc = 0, cap = 0;
    char *acc = NULL; int acc_cap = 0, acc_len = 0, acc_words = 0;
    #define ACC_APPEND(str, sl) do {                                            \
        int _sl = (sl); int need = acc_len + (acc_len ? 1 : 0) + _sl + 1;       \
        if (need > acc_cap) { acc_cap = need * 2; acc = (char *)realloc(acc, (size_t)acc_cap); } \
        if (acc_len) acc[acc_len++] = ' ';                                      \
        memcpy(acc + acc_len, (str), (size_t)_sl); acc_len += _sl; acc[acc_len] = 0; \
    } while (0)
    for (int i = 0; i < ns; i++) {
        int sw = qwen_word_count(sent[i], (int)strlen(sent[i]));
        ACC_APPEND(sent[i], (int)strlen(sent[i]));
        acc_words += sw;
        if (acc_words >= target_words) {
            if (qwen_push_str(&chunks, &nc, &cap, acc, acc_len) < 0) goto oom;
            acc_len = 0; acc_words = 0; if (acc) acc[0] = 0;
        }
    }
    if (acc_len > 0) {
        if (nc > 0 && acc_words < min_words) {
            /* merge trailing fragment into the previous chunk */
            char *prev = chunks[nc-1]; int pl = (int)strlen(prev);
            char *m = (char *)malloc((size_t)pl + 1 + (size_t)acc_len + 1);
            if (!m) goto oom;
            memcpy(m, prev, (size_t)pl); m[pl] = ' '; memcpy(m + pl + 1, acc, (size_t)acc_len); m[pl + 1 + acc_len] = 0;
            free(prev); chunks[nc-1] = m;
        } else {
            if (qwen_push_str(&chunks, &nc, &cap, acc, acc_len) < 0) goto oom;
        }
    }
    #undef ACC_APPEND
    free(acc);
    for (int i = 0; i < ns; i++) free(sent[i]); free(sent);
    *out = chunks; *out_n = nc; return 0;
oom:
    free(acc);
    for (int i = 0; i < ns; i++) free(sent[i]); free(sent);
    for (int i = 0; i < nc; i++) free(chunks[i]); free(chunks);
    return -1;
}

/* --batch Milestone A: split long text -> synthesize each chunk via the existing
 * single-stream path -> concatenate (reuses render_spans). Correct audio + the
 * sequential baseline; Milestone B swaps the inner loop for batched compute. */
static int run_batch(qwen_tts_ctx_t *ctx, const char *text, int target_words, int dry,
                     const char *language, float chunk_pause, const char *output, int silent) {
    char **chunks = NULL; int nc = 0;
    int min_words = target_words / 3 < 6 ? 6 : target_words / 3;
    if (qwen_split_text_for_batch(text, target_words, min_words, &chunks, &nc) != 0 || nc == 0) {
        fprintf(stderr, "--batch: text split failed\n"); return -1;
    }
    if (!silent || dry) {
        fprintf(stderr, "--batch: %d chunk(s) (target ~%d words, min %d):\n", nc, target_words, min_words);
        for (int i = 0; i < nc; i++)
            fprintf(stderr, "  [chunk %d, %d words] %s\n", i, qwen_word_count(chunks[i], (int)strlen(chunks[i])), chunks[i]);
    }
    if (dry) { for (int i = 0; i < nc; i++) free(chunks[i]); free(chunks); return 0; }

    /* Milestone B: batched compute (chunks stepped together, weights reused). Falls
     * back to Milestone A (sequential render_spans) if the model can't use the bf16
     * batched path or only one chunk. QWEN_BATCH_SEQ=1 forces the sequential path
     * (diagnostic: the per-chunk bit-exact reference for the batched output). */
    int rc;
    if (nc >= 2 && !getenv("QWEN_BATCH_SEQ")) {
        float *audio = NULL; int n = 0;
        rc = qwen_tts_generate_batch(ctx, chunks, nc, chunk_pause, &audio, &n);
        if (rc == 0 && audio && n > 0) {
            rc = qwen_tts_write_wav(output, audio, n, QWEN_TTS_SAMPLE_RATE);
            if (rc == 0 && !silent)
                fprintf(stderr, "Wrote %s (%d samples, %.2fs) [batched %d chunks]\n", output, n, (double)n / QWEN_TTS_SAMPLE_RATE, nc);
            free(audio);
            for (int i = 0; i < nc; i++) free(chunks[i]); free(chunks);
            return rc;
        }
        free(audio);
        if (!silent) fprintf(stderr, "--batch: batched path unavailable (rc=%d), falling back to sequential\n", rc);
    }

    /* Milestone A fallback: sequential synth + concat via render_spans. */
    cspan_t *spans = (cspan_t *)calloc((size_t)nc, sizeof(cspan_t));
    if (!spans) { for (int i = 0; i < nc; i++) free(chunks[i]); free(chunks); return -1; }
    for (int i = 0; i < nc; i++) {
        spans[i].is_pause = 0; spans[i].mood[0] = 0; spans[i].text = chunks[i];
        spans[i].steer_weight = -1.0f; spans[i].rate = 0.0f; spans[i].volume = 0.0f; spans[i].is_filler = 0;
    }
    rc = render_spans(ctx, spans, nc, language, chunk_pause, output, silent);
    free(spans);
    for (int i = 0; i < nc; i++) free(chunks[i]); free(chunks);
    return rc;
}

/* Synthesize a parsed span list into one WAV file (thin wrapper over the shared composer). */
static int render_spans(qwen_tts_ctx_t *ctx, cspan_t *spans, int nspans,
                        const char *language, float default_pause,
                        const char *output, int silent) {
    float *audio = NULL; int n = 0;
    int rc = qwen_compose_render_buffer(ctx, spans, nspans, language, default_pause, &audio, &n, silent);
    if (rc != 0) return rc;
    rc = qwen_tts_write_wav(output, audio, n, QWEN_TTS_SAMPLE_RATE);
    if (rc == 0 && !silent)
        fprintf(stderr, "Wrote %s (%d samples, %.2fs)\n", output, n, (double)n / QWEN_TTS_SAMPLE_RATE);
    free(audio);
    return rc;
}

/* Parse a markup spec (from --compose or --text) and render it to one WAV. */
static int run_compose(qwen_tts_ctx_t *ctx, const char *spec, const char *language,
                       float default_pause, const char *output, int silent) {
    cspan_t *spans = NULL; int n = 0;
    if (qwen_compose_parse(spec, &spans, &n) != 0) { fprintf(stderr, "Compose: parse error\n"); return -1; }
    int rc = render_spans(ctx, spans, n, language, default_pause, output, silent);
    qwen_compose_free_spans(spans, n);
    return rc;
}

/* Apply a `<lang>.expr` micro-file: an additive emotion/expressivity weight delta
 * (the fine-tune's L16-26 attn+gate change) on TOP of the currently-loaded talker
 * weights. The file body is a standard "WDLT" stream (same encoding as the .qvoice
 * WDELTA section): int16 bit-pattern deltas + LZ4. Reconstruct result = current + delta
 * (mod 2^16 on the bf16 bits) → on a CV preset or an --icl-only graft (CV weights intact)
 * this restores the fine-tuned weights EXACTLY on the changed tensors. Composable with a
 * voice (the voice's ICL/x-vector loads separately; this only touches the backbone weights).
 * Returns 0 on success (>=1 tensor applied), -1 on error. */
static int apply_expr_file(qwen_tts_ctx_t *ctx, const char *path, float expr_weight, int silent) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Error: cannot open --expr file %s\n", path); return -1; }
    char magic[4];
    char lang[16] = {0};
    uint32_t version = 0, reserved = 0;
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "QEXP", 4) != 0) {
        fprintf(stderr, "Error: %s is not a valid .expr file (bad magic)\n", path);
        fclose(f); return -1;
    }
    /* audit #8: check the header freads (a truncated file must not proceed with garbage).
     * The tensor DATA itself is integrity-guarded by download_assets.sh's sha256 vs
     * presets/expr/MANIFEST.md, so per-tensor size validation is covered upstream. */
    if (fread(&version, sizeof(uint32_t), 1, f) != 1 ||
        fread(lang, 1, 16, f) != 16 ||
        fread(&reserved, sizeof(uint32_t), 1, f) != 1) {
        fprintf(stderr, "Error: %s truncated header\n", path);
        fclose(f); return -1;
    }
    char wmagic[4];
    if (fread(wmagic, 1, 4, f) != 4 || memcmp(wmagic, "WDLT", 4) != 0) {
        fprintf(stderr, "Error: %s missing WDLT stream\n", path);
        fclose(f); return -1;
    }
    uint32_t target_h = 0, n_tensors = 0;
    if (fread(&target_h, sizeof(uint32_t), 1, f) != 1 ||
        fread(&n_tensors, sizeof(uint32_t), 1, f) != 1) { fclose(f); return -1; }  /* audit #8 */
    if ((int)target_h != ctx->config.hidden_size) {
        fprintf(stderr, "Error: .expr is for hidden=%u but model has hidden=%d (model/.expr mismatch)\n",
                target_h, ctx->config.hidden_size);
        fclose(f); return -1;
    }

    int nl = ctx->config.num_layers;
    int inter = ctx->config.intermediate_size;
    int loaded = 0;

    for (uint32_t t = 0; t < n_tensors; t++) {
        uint16_t name_len;
        if (fread(&name_len, sizeof(uint16_t), 1, f) != 1) break;
        char tname[256] = {0};
        if (name_len >= 256) break;
        uint32_t data_bytes = 0, comp_size = 0;
        uint8_t dtype_flag = 0;
        /* audit #8: check the per-record metadata freads; a truncated record must not feed
         * garbage sizes into the mallocs below. Cap sizes to a sane bound (256 MB) so a
         * corrupt length can't trigger an absurd allocation. */
        if (fread(tname, 1, name_len, f) != name_len ||
            fread(&data_bytes, sizeof(uint32_t), 1, f) != 1 ||
            fread(&dtype_flag, 1, 1, f) != 1 ||
            fread(&comp_size, sizeof(uint32_t), 1, f) != 1) break;
        if (data_bytes > (256u << 20) || comp_size > (256u << 20)) break;

        /* Resolve tname -> the weight pointer to override. bf16 weight matrices use the
         * int16-delta path (dtype 4); RMSNorm weights are f32 in the engine and use raw
         * f32 replacement (dtype 0). */
        void **target_ptr = NULL;
        int is_f32 = 0;
        if (strcmp(tname, "talker.model.text_embedding.weight") == 0)
            target_ptr = (void **)&ctx->tok_embeddings_bf16;
        else if (strcmp(tname, "talker.text_projection.linear_fc1.weight") == 0)
            target_ptr = (void **)&ctx->text_proj_fc1_bf16;
        else if (strcmp(tname, "talker.text_projection.linear_fc2.weight") == 0)
            target_ptr = (void **)&ctx->text_proj_fc2_bf16;
        else if (strcmp(tname, "talker.model.codec_embedding.weight") == 0)
            target_ptr = (void **)&ctx->codec_embedding_bf16;
        else if (strcmp(tname, "talker.codec_head.weight") == 0)
            target_ptr = (void **)&ctx->codec_head_bf16;
        else {
            int layer = -1;
            sscanf(tname, "talker.model.layers.%d.", &layer);
            if (layer >= 0 && layer < nl) {
                qwen_talker_layer_t *l = &ctx->layers[layer];
                char *suffix = strstr(tname, "self_attn.");
                if (!suffix) suffix = strstr(tname, "mlp.");
                if (suffix) {
                    if (strstr(suffix, "q_proj.weight"))      target_ptr = (void **)&l->wq_bf16;
                    else if (strstr(suffix, "k_proj.weight")) target_ptr = (void **)&l->wk_bf16;
                    else if (strstr(suffix, "v_proj.weight")) target_ptr = (void **)&l->wv_bf16;
                    else if (strstr(suffix, "o_proj.weight")) target_ptr = (void **)&l->wo_bf16;
                    else if (strstr(suffix, "gate_proj.weight")) target_ptr = (void **)&l->gate_bf16;
                    else if (strstr(suffix, "up_proj.weight"))   target_ptr = (void **)&l->up_bf16;
                    else if (strstr(suffix, "down_proj.weight")) target_ptr = (void **)&l->down_bf16;
                    else if (strstr(suffix, "q_norm.weight")) { target_ptr = (void **)&l->q_norm; is_f32 = 1; }
                    else if (strstr(suffix, "k_norm.weight")) { target_ptr = (void **)&l->k_norm; is_f32 = 1; }
                }
                /* BUG FIX: the layer-level RMSNorms have NO `self_attn.`/`mlp.` infix, so they were
                 * gated behind `suffix` above and SILENTLY NEVER APPLIED — every .expr's input/post
                 * layernorm deltas were dropped, under-applying the FT (e.g. CSP topk4 = 35/40). Match
                 * them on `tname` directly so the full fine-tune is applied. */
                else if (strstr(tname, "input_layernorm.weight"))          { target_ptr = (void **)&l->input_norm;     is_f32 = 1; }
                else if (strstr(tname, "post_attention_layernorm.weight")) { target_ptr = (void **)&l->post_attn_norm; is_f32 = 1; }
            }
        }

        if (target_ptr && is_f32 && dtype_flag == 0) {
            /* RMSNorm f32 replacement: read the full f32 array and swap the pointer. */
            void *buf = malloc(comp_size);
            if (buf && fread(buf, 1, comp_size, f) == comp_size) {
                *target_ptr = buf; qwen_track_override(ctx, buf);
                loaded++;
            } else {
                free(buf);
                fseek(f, comp_size, SEEK_CUR);
            }
        } else if (target_ptr && !is_f32 && dtype_flag == 4) {
            size_t n16 = data_bytes / 2;
            uint8_t *lz4_data = (uint8_t *)malloc(comp_size);
            int16_t *delta16 = (int16_t *)malloc(n16 * sizeof(int16_t));
            uint16_t *result = (uint16_t *)malloc(data_bytes);
            const uint16_t *cur = (const uint16_t *)*target_ptr;
            if (lz4_data && delta16 && result && cur &&
                fread(lz4_data, 1, comp_size, f) == comp_size &&
                LZ4_decompress_safe((const char *)lz4_data, (char *)delta16,
                                    (int)comp_size, (int)(n16 * sizeof(int16_t))) == (int)(n16 * sizeof(int16_t))) {
                if (expr_weight == 1.0f) {
                    /* exact reconstruction: ft_bits = cur_bits + delta_bits (mod 2^16) */
                    for (size_t i = 0; i < n16; i++)
                        result[i] = (uint16_t)((int)cur[i] + (int)delta16[i]);
                } else {
                    /* --expr-weight dosing on a DENSE delta: scale in FLOAT space,
                     * base + w*(ft - base). cur=base bf16; ft=base+delta (bit add, modular,
                     * correct even if the extract wrapped). w>1 amplifies, w<1 dampens. */
                    for (size_t i = 0; i < n16; i++) {
                        uint16_t ft_bits = (uint16_t)((int)cur[i] + (int)delta16[i]);
                        float base_f = main_bf16_to_f32(cur[i]);
                        float ft_f   = main_bf16_to_f32(ft_bits);
                        result[i] = main_f32_to_bf16(base_f + expr_weight * (ft_f - base_f));
                    }
                }
                *target_ptr = result; qwen_track_override(ctx, result);
                loaded++;
            } else {
                if (getenv("QWEN_EXPR_DEBUG"))
                    fprintf(stderr, "  [expr] SKIP delta-fail %s (cur=%p lz4=%p d16=%p res=%p comp=%u)\n",
                            tname, (void*)cur, (void*)lz4_data, (void*)delta16, (void*)result, comp_size);
                free(result);
            }
            free(lz4_data); free(delta16);
        } else if (target_ptr && !is_f32 && dtype_flag == 5) {
            /* LoRA factors: payload = u32 r, u32 in, u32 out, f32 scale, A[r*in] f32, B[out*r] f32.
             * Reconstruct delta = scale*(B@A) [out,in] and add to the bf16 weight, row by row. */
            uint8_t *pl = (uint8_t *)malloc(comp_size);
            if (pl && fread(pl, 1, comp_size, f) == comp_size && comp_size >= 16) {
                uint32_t r, n_in, n_out; float scale;
                memcpy(&r, pl, 4); memcpy(&n_in, pl + 4, 4);
                memcpy(&n_out, pl + 8, 4); memcpy(&scale, pl + 12, 4);
                scale *= expr_weight;   /* --expr-weight: dose the LoRA delta (factored only) */
                const float *A  = (const float *)(pl + 16);                       /* [r, in]  */
                const float *Bm = (const float *)(pl + 16 + (size_t)r * n_in * 4);/* [out, r] */
                const uint16_t *cur = (const uint16_t *)*target_ptr;
                uint16_t *result = (uint16_t *)malloc((size_t)n_out * n_in * sizeof(uint16_t));
                float *drow = (float *)malloc((size_t)n_in * sizeof(float));
                int ok = (size_t)16 + (size_t)r*(n_in+n_out)*4 == (size_t)comp_size;
                if (result && drow && cur && ok) {
                    for (uint32_t o = 0; o < n_out; o++) {
                        for (uint32_t i = 0; i < n_in; i++) drow[i] = 0.0f;
                        for (uint32_t k = 0; k < r; k++) {
                            float bok = scale * Bm[(size_t)o * r + k];
                            const float *Ar = A + (size_t)k * n_in;
                            for (uint32_t i = 0; i < n_in; i++) drow[i] += bok * Ar[i];
                        }
                        const uint16_t *cr = cur + (size_t)o * n_in;
                        uint16_t *rr = result + (size_t)o * n_in;
                        for (uint32_t i = 0; i < n_in; i++)
                            rr[i] = main_f32_to_bf16(main_bf16_to_f32(cr[i]) + drow[i]);
                    }
                    *target_ptr = result; qwen_track_override(ctx, result); loaded++;
                } else { free(result); }
                free(drow);
            }
            free(pl);
        } else {
            /* Unknown tensor or unsupported dtype — skip its payload. */
            if (getenv("QWEN_EXPR_DEBUG"))
                fprintf(stderr, "  [expr] SKIP unmatched %s (tgt=%p is_f32=%d dtype=%d)\n",
                        tname, (void*)target_ptr, is_f32, dtype_flag);
            fseek(f, comp_size, SEEK_CUR);
        }
    }
    fclose(f);

    /* Rebuild gate_up_fused for talker layers (gate_proj changed). */
    int h = ctx->config.hidden_size;
    for (int li = 0; li < nl; li++) {
        qwen_talker_layer_t *l = &ctx->layers[li];
        if (l->gate_bf16 && l->up_bf16 && l->gate_up_fused_bf16) {
            size_t row_bytes = (size_t)h * sizeof(uint16_t);
            for (int r = 0; r < inter; r++) {
                memcpy(l->gate_up_fused_bf16 + (size_t)(2*r)*h,   l->gate_bf16 + (size_t)r*h, row_bytes);
                memcpy(l->gate_up_fused_bf16 + (size_t)(2*r+1)*h, l->up_bf16   + (size_t)r*h, row_bytes);
            }
        }
    }

    /* In INT8 mode the quantized weights were built from the pre-delta bf16 —
     * re-quantize from the overridden bf16 so the delta is honored (mirrors the
     * .qvoice WDELTA path). NOTE: int4 (Q4_0) re-quant after weight override is not
     * yet wired here — .expr in --int4 mode would ignore the delta (TODO). */
    if (loaded > 0 && ctx->use_int8) {
        extern void qwen_talker_quantize_int8(qwen_tts_ctx_t *ctx);
        extern void qwen_cp_quantize_int8(qwen_tts_ctx_t *ctx);
        if (!silent) fprintf(stderr, "  Re-quantizing INT8 from .expr-overridden weights...\n");
        qwen_talker_quantize_int8(ctx);
        qwen_cp_quantize_int8(ctx);
    } else if (loaded > 0 && ctx->use_int4 && !silent) {
        fprintf(stderr, "  Warning: --expr in --int4 mode does not yet re-quantize; delta ignored.\n");
    }

    if (!silent)
        fprintf(stderr, "Expressivity: applied %d/%u tensors from %s (lang=%s)\n",
                loaded, n_tensors, path, lang[0] ? lang : "?");
    return loaded > 0 ? 0 : -1;
}

int main(int argc, char **argv) {
    const char *model_dir = NULL;
    const char *text = NULL;
    const char *output = "output.wav";
    int speaker_id = -1;
    const char *language = NULL;
    const char *instruct = NULL;
    float temperature = 0.9f;
    int   temp_set = 0;              /* did the user pass -T explicitly? (for the emo-bump) */
    int top_k = 50;
    float top_p = 1.0f;
    float rep_penalty = 1.05f;
    int max_tokens = 8192;
    int silent = 0;
    int debug = 0;
    int threads = 0;  /* 0 = auto-detect */
    int do_stream = 0;
    int do_stdout = 0;
    int stream_chunk = 10;
    int serve_port = 0;  /* 0 = not serving */
    int serve_workers = 1;  /* --workers: concurrent synthesis workers (server mode) */
    int serve_batch = 1;    /* --batch-size: vLLM-style request-batching (server; N>=2 enables) */
    const char *ml_steer_path = NULL;  /* --ml-steer: multi-layer Talker emotion steer (.qlsteer) */
    float ml_steer_weight = 8.0f;      /* --ml-weight */
    int ml_l0 = 21, ml_l1 = 25;        /* --ml-range "l0-l1" (identity layers) */
    float ml_decay = 0.985f;           /* --ml-decay: per-frame weight multiplier. DEFAULT 0.985 = derail-fix
                                          always-on (tames the EOS "so so so" tail on steering). 1.0 = no decay */
    int ml_frames = 0;                 /* --ml-frames: apply only first N gen frames (0=all) */
    int show_caps = 0;   /* --caps: print compiled SIMD/threading capabilities and exit */
    int run_self_test = 0; /* --self-test: kernel numeric self-test (matvec vs f32 ref) and exit */
    int run_matmat_bench = 0; /* --matmat-bench: batched matmat vs B*matvec throughput, per precision, exit */
    int run_gpu_selftest = 0; /* --gpu-selftest: GPU-vs-CPU matvec/matmat correctness + timing, exit (GPU builds) */
    int run_gpu_selftest_talker = 0; /* --gpu-selftest-talker: fused resident Talker step vs CPU (needs model) */
    int run_gpu_batch_bench = 0; int gpu_batch_B = 4; /* --gpu-batch-bench N: batched Talker correctness + throughput */
    const char *gpu_backend_str = NULL; /* --backend cpu|metal|cuda (v1: selects the --gpu-selftest target) */
    float cp_roughness = 0.0f;        /* --roughness: q2-down blend on the CP (texture knob) */
    const char *emotion_spec = NULL;  /* --emotion: mood name or preset(s), e.g. "joy", "happy:0.5,proud:0.5" */
    const char *speaker_name = NULL;  /* -s preset name kept verbatim (id discards it) for the emotion router */
    float audio_volume = 1.0f;        /* --volume: linear PCM gain on the output */
    float audio_rate = 1.0f;          /* --rate: pitch-preserving tempo (>1 faster) */
    /* "_set" = flag was explicitly passed -> overrides any --emotion manifest recipe value */
    int roughness_set = 0, volume_set = 0, rate_set = 0;
    const char *compose_spec = NULL;  /* --compose: multi-span "[mood] text | [mood] text | [pause=0.5]" */
    float compose_pause = 0.12f;      /* --compose-pause: default gap (s) between spoken spans */
    int no_compose = 0;               /* --no-compose: pass [tags] LITERALLY to the model (don't auto-route
                                       * through the macro composer) — used to test a paralinguistic LoRA
                                       * that emits [laugh]/[sigh]/... itself instead of the synth macro. */
    int batch_mode = 0;               /* --batch: split long text into chunks + synth (long-form) */
    int batch_words = 16;             /* --batch-words: target words/chunk (sentence-packed) */
    int batch_dry = 0;                /* --batch-dry: print the chunking and exit (no synth) */
    int run_batch_test = 0;           /* --batch-test: verify batched Talker step vs single-stream, exit */
    int run_batch_bench = 0;          /* --batch-bench: batched-compute throughput vs single-stream, exit */
    int batch_multi_test = 0;         /* --batch-multi-test N: run N copies of the request through the
                                         server batch-multi engine, write bm_<i>.wav each, exit */
    int seed = -1;       /* -1 = use time-based seed */
    float max_duration = 0;  /* 0 = no limit */
    int voice_design = 0;
    const char *ref_audio = NULL;
    const char *ref_text_str = NULL;
    int xvector_only = 0;
    const char *save_voice = NULL;
    const char *load_voice = NULL;
    const char *expr_path = NULL;    /* --expr: additive expressivity weight delta (<lang>.expr) */
    float expr_weight = 1.0f;        /* --expr-weight: dose a factored-LoRA .expr (1=as trained) */
    int   onset_fade_ms = 0;         /* --onset-fade <ms>: fade-in over the REAL attack (0=off) */
    int   tail_trim = 0;             /* --tail-trim: cut a degenerate metallic tail (default off) */
    int   seed_audition = 0;         /* --seed-audition <N>: render N seeds, keep cleanest (0=off) */
    int   audition_keep = 0;         /* --audition-keep: also write every audition take as <out>.seedNN.wav */
    const char *list_voices_dir = NULL;
    const char *delete_voice = NULL;
    const char *voice_name = NULL;
    const char *target_cv_dir = NULL;
    int ctx_greedy_warmup = 0;
    float max_ref_duration = 30.0f;  /* default: use first 30s of ref audio */
    float voice_strength = 1.0f;     /* --voice-strength: scale .qvoice WDELTA (1=full voice, <1=blend
                                        toward CV model = less fidelity but more instruct/emotion response) */
    int vs_l0 = 11, vs_l1 = 27;      /* --vs-layers: Talker layer range diluted by --voice-strength
                                        (deep attn/mlp only; identity tensors stay at full voice) */
    int icl_only = 0;                /* --icl-only: load a .qvoice's ICL prefix (speaker-emb + ref-codes)
                                        but SKIP the WDELTA weight-swap → keep the instruct-capable CV weights
                                        intact = clone-via-ICL + full instruct (the "graft" experiment) */
    int icl_frames = 0;              /* --icl-frames N: cap ICL ref frames (dilute the prosody anchor for
                                        more emotion room). 0 = use all (default). */
    int graft = 0;                   /* --graft: ignore a lite .qvoice's ref_codes, clone via x-vector
                                        (CV weights + instruct emote). Same file, emotive mode. */
    int use_int8 = 0;
    int use_int4 = 0;
    static struct option long_options[] = {
        {"model-dir",     required_argument, 0, 'd'},
        {"text",          required_argument, 0, 't'},
        {"output",        required_argument, 0, 'o'},
        {"speaker",       required_argument, 0, 's'},
        {"language",      required_argument, 0, 'l'},
        {"temperature",   required_argument, 0, 'T'},
        {"top-k",         required_argument, 0, 'k'},
        {"top-p",         required_argument, 0, 'p'},
        {"rep-penalty",   required_argument, 0, 'r'},
        {"max-tokens",    required_argument, 0, 'm'},
        {"threads",       required_argument, 0, 'j'},
        {"instruct",      required_argument, 0, 'I'},
        {"stream",        no_argument,       0, 1001},
        {"stdout",        no_argument,       0, 1002},
        {"stream-chunk",  required_argument, 0, 1003},
        {"serve",         required_argument, 0, 1004},
        {"seed",          required_argument, 0, 1005},
        {"max-duration",  required_argument, 0, 1006},
        {"voice-design",  no_argument,       0, 1007},
        {"ref-audio",     required_argument, 0, 1008},
        {"ref-text",      required_argument, 0, 1009},
        {"xvector-only",  no_argument,       0, 1010},
        {"save-voice",    required_argument, 0, 1011},
        {"load-voice",    required_argument, 0, 1012},
        {"max-ref-duration", required_argument, 0, 1013},
        {"voice-strength", required_argument, 0, 1049},
        {"vs-layers",     required_argument, 0, 1050},
        {"icl-only",      no_argument,       0, 1051},
        {"icl-frames",    required_argument, 0, 1052},
        {"graft",         no_argument,       0, 1053},
        {"expr",          required_argument, 0, 1054},
        {"expr-weight",   required_argument, 0, 1055},
        {"silent",        no_argument,       0, 'S'},
        {"debug",         no_argument,       0, 'D'},
        {"list-voices",   required_argument, 0, 1016},
        {"delete-voice",  required_argument, 0, 1017},
        {"int8",          no_argument,       0, 1014},
        {"int4",          no_argument,       0, 1015},
        {"quant-mixed",   no_argument,       0, 1073},
        {"voice-name",    required_argument, 0, 1022},
        {"greedy-warmup", required_argument, 0, 1023},
        {"target-cv",     required_argument, 0, 1024},
        {"caps",          no_argument,       0, 1025},
        {"workers",       required_argument, 0, 1026},
        {"batch-size",    required_argument, 0, 1043},
        {"ml-steer",      required_argument, 0, 1044},
        {"ml-weight",     required_argument, 0, 1045},
        {"ml-range",      required_argument, 0, 1046},
        {"ml-decay",      required_argument, 0, 1047},
        {"ml-frames",     required_argument, 0, 1048},
        {"self-test",     no_argument,       0, 1027},
        {"matmat-bench",  no_argument,       0, 1038},
        {"gpu-selftest",  no_argument,       0, 1070},
        {"backend",       required_argument, 0, 1071},
        {"gpu-selftest-talker", no_argument, 0, 1072},
        {"gpu-batch-bench", required_argument, 0, 1074},
        {"roughness",     required_argument, 0, 1028},
        {"emotion",       required_argument, 0, 1031},
        {"volume",        required_argument, 0, 1032},
        {"rate",          required_argument, 0, 1033},
        {"compose",       required_argument, 0, 1034},
        {"no-compose",    no_argument,       0, 1056},
        {"batch",         no_argument,       0, 1039},
        {"batch-words",   required_argument, 0, 1040},
        {"batch-dry",     no_argument,       0, 1041},
        {"compose-pause", required_argument, 0, 1035},
        {"batch-test",    no_argument,       0, 1036},
        {"batch-bench",   no_argument,       0, 1037},
        {"batch-multi-test", required_argument, 0, 1042},
        {"onset-fade",    required_argument, 0, 1057},
        {"tail-trim",     no_argument,       0, 1058},
        {"seed-audition", required_argument, 0, 1059},
        {"audition-keep", no_argument,       0, 1060},
        {"help",          no_argument,       0, 'h'},
        {0, 0, 0, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "d:t:o:s:l:T:k:p:r:m:j:I:SDh", long_options, NULL)) != -1) {
        switch (opt) {
            case 'd': model_dir = optarg; break;
            case 't': text = optarg; break;
            case 'o': output = optarg; break;
            case 's': speaker_id = qwen_tts_speaker_id(optarg); speaker_name = optarg; break;
            case 'l': language = optarg; break;
            case 'T': temperature = (float)atof(optarg); temp_set = 1; break;
            case 'k': top_k = atoi(optarg); break;
            case 'p': top_p = (float)atof(optarg); break;
            case 'r': rep_penalty = (float)atof(optarg); break;
            case 'm': max_tokens = atoi(optarg); break;
            case 'j': threads = atoi(optarg); break;
            case 'I': instruct = optarg; break;
            case 1001: do_stream = 1; break;
            case 1002: do_stdout = 1; do_stream = 1; break;  /* --stdout implies --stream */
            case 1003: stream_chunk = atoi(optarg); break;
            case 1004: serve_port = atoi(optarg); break;
            case 1005: seed = atoi(optarg); break;
            case 1006: max_duration = (float)atof(optarg); break;
            case 1007: voice_design = 1; break;
            case 1008: ref_audio = optarg; break;
            case 1009: ref_text_str = optarg; break;
            case 1010: xvector_only = 1; break;
            case 1011: save_voice = optarg; break;
            case 1012: load_voice = optarg; break;
            case 1013: max_ref_duration = (float)atof(optarg); break;
            case 1049: voice_strength = (float)atof(optarg); break;
            case 1050: { int a,b; if (sscanf(optarg,"%d-%d",&a,&b)==2){vs_l0=a;vs_l1=b;} break; }
            case 1051: icl_only = 1; break;
            case 1052: icl_frames = atoi(optarg); break;
            case 1053: graft = 1; icl_only = 1; break;  /* --graft implies --icl-only (skip WDELTA too) */
            case 1054: expr_path = optarg; break;        /* --expr <lang>.expr expressivity delta */
            case 1055: expr_weight = atof(optarg); break;/* --expr-weight: scale factored-LoRA delta */
            case 1014: use_int8 = 1; break;
            case 1015: use_int4 = 1; break;
            case 1073: use_int4 = 1; setenv("QWEN_CP_PREC", "int8", 1); break;  /* --quant-mixed: int4 Talker + int8 CP (best CUDA quant) */
            case 1022: voice_name = optarg; break;
            case 1023: { int gw = atoi(optarg); ctx_greedy_warmup = gw; } break;
            case 1024: target_cv_dir = optarg; break;
            case 1025: show_caps = 1; break;
            case 1026: serve_workers = atoi(optarg); break;
            case 1043: serve_batch = atoi(optarg); if (serve_batch < 1) serve_batch = 1; break;
            case 1044: ml_steer_path = optarg; break;
            case 1045: ml_steer_weight = atof(optarg); break;
            case 1046: { int a, b; if (sscanf(optarg, "%d-%d", &a, &b) == 2) { ml_l0 = a; ml_l1 = b; } break; }
            case 1047: ml_decay = (float)atof(optarg); break;
            case 1048: ml_frames = atoi(optarg); break;
            case 1027: run_self_test = 1; break;
            case 1038: run_matmat_bench = 1; break;
            case 1070: run_gpu_selftest = 1; break;
            case 1071: gpu_backend_str = optarg; break;
            case 1072: run_gpu_selftest_talker = 1; break;
            case 1074: run_gpu_batch_bench = 1; gpu_batch_B = atoi(optarg); break;
            case 1028: cp_roughness = (float)atof(optarg); roughness_set = 1; break;
            case 1031: emotion_spec = optarg; break;
            case 1032: audio_volume = (float)atof(optarg); volume_set = 1; break;
            case 1033: audio_rate = (float)atof(optarg); rate_set = 1; break;
            case 1034: compose_spec = optarg; break;
            case 1056: no_compose = 1; break;
            case 1035: compose_pause = (float)atof(optarg); break;
            case 1039: batch_mode = 1; break;
            case 1040: batch_words = atoi(optarg); if (batch_words < 4) batch_words = 4; break;
            case 1041: batch_mode = 1; batch_dry = 1; break;
            case 1036: run_batch_test = 1; break;
            case 1037: run_batch_bench = 1; break;
            case 1042: batch_multi_test = atoi(optarg); if (batch_multi_test < 1) batch_multi_test = 1; break;
            case 1057: onset_fade_ms = atoi(optarg); if (onset_fade_ms < 0) onset_fade_ms = 0; break;
            case 1058: tail_trim = 1; break;
            case 1059: seed_audition = atoi(optarg); if (seed_audition < 1) seed_audition = 1; break;
            case 1060: audition_keep = 1; break;
            case 1016: list_voices_dir = optarg; break;
            case 1017: delete_voice = optarg; break;
            case 'S': silent = 1; break;
            case 'D': debug = 1; break;
            case 'h':
            default:
                fprintf(stderr, "Usage: %s -d <model_dir> -t <text> [options]\n", argv[0]);
                fprintf(stderr, "Options:\n");
                fprintf(stderr, "  -d, --model-dir <path>     Model directory\n");
                fprintf(stderr, "  -t, --text <string>        Text to synthesize\n");
                fprintf(stderr, "  -o, --output <path>        Output WAV file\n");
                fprintf(stderr, "  -s, --speaker <name>       Speaker name\n");
                fprintf(stderr, "  -l, --language <name>      Language\n");
                fprintf(stderr, "  -T, --temperature <float>  Sampling temperature\n");
                fprintf(stderr, "  -k, --top-k <int>          Top-k sampling\n");
                fprintf(stderr, "  -p, --top-p <float>        Top-p sampling\n");
                fprintf(stderr, "  -r, --rep-penalty <float>  Repetition penalty\n");
                fprintf(stderr, "  -m, --max-tokens <int>     Max tokens\n");
                fprintf(stderr, "  -j, --threads <int>        Number of threads (0=auto)\n");
                fprintf(stderr, "  -I, --instruct <text>      Style instruction (1.7B only)\n");
                fprintf(stderr, "                             e.g. \"Speak in an angry tone\"\n");
                fprintf(stderr, "  --stream                   Stream audio (decode during generation)\n");
                fprintf(stderr, "  --stdout                   Output raw s16le PCM to stdout (implies --stream)\n");
                fprintf(stderr, "  --stream-chunk <n>         Frames per stream chunk (default: 10)\n");
                fprintf(stderr, "  --serve <port>             Start HTTP server on port\n");
                fprintf(stderr, "  --workers <n>              Concurrent synthesis workers (server; default 1)\n");
                fprintf(stderr, "  --batch-size <n>           Request-batching: step up to n concurrent users together (server; n>=2)\n");
                fprintf(stderr, "  --seed <n>                 Random seed (default: time-based)\n");
                fprintf(stderr, "  --max-duration <secs>      Max audio duration in seconds\n");
                fprintf(stderr, "  --voice-design             VoiceDesign mode (create voice from --instruct)\n");
                fprintf(stderr, "  --ref-audio <path>         Reference audio for voice cloning (Base model; must be 24 kHz mono WAV: ffmpeg -i in -ar 24000 -ac 1 out.wav)\n");
                fprintf(stderr, "  --xvector-only             Use speaker embedding only (no ref text/codes)\n");
                fprintf(stderr, "  --save-voice <path>        Save voice. DEFAULT .qvoice = ~16-25MB GRAFT (identity+prosody,\n");
                fprintf(stderr, "                             instruct/expr/steer all work). .bin = 8KB x-vector only. Heavy WDELTA\n");
                fprintf(stderr, "                             qvoice (~0.8-3GB, exact-identity) ONLY with --target-cv. Without --text: create+exit\n");
                fprintf(stderr, "  --load-voice <path>        Load voice (.qvoice graft/heavy or .bin x-vector; use --icl-only for the graft)\n");
                fprintf(stderr, "  --voice-name <name>        Name for the voice (stored in .qvoice metadata)\n");
                fprintf(stderr, "  --list-voices <dir>        List .qvoice files in directory\n");
                fprintf(stderr, "  --delete-voice <path>      Delete a .qvoice file\n");
                fprintf(stderr, "  --max-ref-duration <secs>  Max ref audio for embedding (default: 30, 0=all)\n");
                fprintf(stderr, "  --voice-strength <a>       .qvoice WDELTA scale (1=full voice; <1=more emotion, less fidelity)\n");
                fprintf(stderr, "  --icl-frames <N>           Cap ICL ref frames (dilute prosody anchor; more emotion; 0=all)\n");
                fprintf(stderr, "  --graft                    Ignore lite .qvoice ref_codes; clone via x-vector (emotive; use with --instruct)\n");
                fprintf(stderr, "  --expr <file>              Apply a <lang>.expr expressivity weight delta on top (composable with any voice; 1.7B)\n");
                fprintf(stderr, "  --expr-weight <m>          Dose a .expr (factored LoRA AND dense): 1=as trained, 0.6=subtler, 1.5=stronger\n");
                fprintf(stderr, "  --int8                     INT8 quantized Talker + Code Predictor\n");
                fprintf(stderr, "  --int4                     Q4_0 quantized Talker (1.7B only, smallest memory)\n");
                fprintf(stderr, "  --quant-mixed              int4 Talker + int8 CP (best CUDA quant: q4 Talker win, no CP degradation)\n");
                fprintf(stderr, "  --roughness <0..1>         Texture/roughness knob (q2-down blend on Code Predictor)\n");
                fprintf(stderr, "  --emotion <spec>           Emotion in ONE flag (1.7B). Primaries: sad/joy/anger/fear/disgust/surprise.\n");
                fprintf(stderr, "                             Dyads: contempt/awe/nostalgia/disapproval/remorse/outrage/despair (blended steer).\n");
                fprintf(stderr, "                             PRESET = STEER ryan_<emo> @ w12 (clean, every language); CLONE = COMBINE (lang .expr + steer).\n");
                fprintf(stderr, "                             Also inline: write [emotion] tags in --text to switch emotion per sentence (one gen).\n");
                fprintf(stderr, "                             NATIVE preset per language (JA ono_anna, KO sohee, ZH vivian, EN/Romance ryan). --instruct/-T override.\n");
                fprintf(stderr, "  --volume <f>               Output gain (1.0=unchanged, e.g. 1.1 louder, 0.9 softer)\n");
                fprintf(stderr, "  --rate <f>                 Speaking rate, pitch-preserving (1.0=unchanged, >1 faster, <1 slower)\n");
                fprintf(stderr, "  --compose <spec>           Inline markup synthesis (also works inside --text):\n");
                fprintf(stderr, "                             'Hi! [sad] I have to go... [sigh] [pause:400ms] [neutral] Bye.'\n");
                fprintf(stderr, "                             Tags: [happy|sad|excited|annoyed|...] emotion, [sigh|huff|ugh|groan|hmm]\n");
                fprintf(stderr, "                             paralinguistic, [pause:400ms]/[pause:1s] pause. One WAV, spans joined.\n");
                fprintf(stderr, "  --compose-pause <s>        Default gap between adjacent spoken spans (default 0.12s)\n");
                fprintf(stderr, "  -S, --silent               Silent mode\n");
                fprintf(stderr, "  -D, --debug                Debug mode\n");
                fprintf(stderr, "  --caps                     Print compiled SIMD/threading capabilities and exit\n");
                fprintf(stderr, "  --self-test                Run kernel numeric self-test (matvec vs f32 ref) and exit\n");
                return opt == 'h' ? 0 : 1;
        }
    }

    /* --caps: report the binary's ACTUAL compiled SIMD/threading capabilities and exit
     * (no model needed). Honest, testable source of truth — catches "we thought AVX
     * existed" regressions that docs/comments can hide. */
    if (show_caps) {
        qwen_init_threads();
        qwen_caps_report(stdout);
        return 0;
    }

    /* --self-test: prove the dispatched matvec kernels are numerically correct vs
     * an f32 reference (no model needed). This is the cross-ISA correctness gate for
     * the AVX-512/VNNI paths — immune to the greedy trajectory fork that makes
     * end-to-end audio mel-corr a false alarm cross-ISA. Exits non-zero on failure. */
    if (run_self_test) {
        qwen_init_threads();
        return qwen_kernel_selftest(stdout);
    }

    /* --matmat-bench: time the batched matmat twins (bf16/int8/int4) vs B sequential
     * matvecs, per precision, at the current -j thread count. No model needed. */
    if (run_matmat_bench) {
        qwen_init_threads();
        return qwen_matmat_bench(stdout);
    }

    /* --gpu-selftest: compare the experimental GPU backend's matvec/matmat against
     * the CPU reference (correctness + rough timing). GPU builds only. */
    if (run_gpu_selftest) {
#if defined(QWEN_HAVE_METAL) || defined(QWEN_HAVE_CUDA)
        qwen_init_threads();
        const char *want = gpu_backend_str;
        if (!want) {
#if defined(QWEN_HAVE_METAL)
            want = "metal";
#else
            want = "cuda";
#endif
        }
        return qwen_gpu_selftest(qwen_backend_kind_from_str(want), stdout);
#else
        (void)gpu_backend_str;
        fprintf(stderr, "--gpu-selftest requires a GPU build: `make metal` or `make cuda`\n");
        return 1;
#endif
    }

    /* Voice library management (no model loading needed) */
    if (list_voices_dir) {
        return list_voices(list_voices_dir);
    }
    if (delete_voice) {
        /* Validate it's a .qvoice file */
        size_t dlen = strlen(delete_voice);
        if (dlen <= 7 || strcmp(delete_voice + dlen - 7, ".qvoice") != 0) {
            fprintf(stderr, "Error: --delete-voice only works with .qvoice files\n");
            return 1;
        }
        /* Check it exists and is valid */
        FILE *vf = fopen(delete_voice, "rb");
        if (!vf) {
            fprintf(stderr, "Error: file not found: %s\n", delete_voice);
            return 1;
        }
        char magic[4];
        int valid = (fread(magic, 1, 4, vf) == 4 && memcmp(magic, "QVCE", 4) == 0);
        fclose(vf);
        if (!valid) {
            fprintf(stderr, "Error: %s is not a valid .qvoice file\n", delete_voice);
            return 1;
        }
        if (remove(delete_voice) != 0) {
            fprintf(stderr, "Error: failed to delete %s\n", delete_voice);
            return 1;
        }
        printf("Deleted %s\n", delete_voice);
        return 0;
    }

    if (!model_dir) {
        fprintf(stderr, "Error: --model-dir is required\n");
        return 1;
    }
    /* --save-voice without --text = create voice only (no generation) */
    int create_voice_only = (save_voice && !text && serve_port <= 0);
    if (!text && !compose_spec && serve_port <= 0 && !create_voice_only && !run_batch_test && !run_batch_bench
        && !run_gpu_selftest_talker && !run_gpu_batch_bench) {
        fprintf(stderr, "Error: --text, --compose or --serve is required\n");
        return 1;
    }

    if (!silent) {
        fprintf(stderr, "Model dir: %s\n", model_dir);
        if (text) fprintf(stderr, "Text: \"%s\"\n", text);
        fprintf(stderr, "Output: %s\n", output);
    }

    /* Early validation: check ref-audio format BEFORE loading the model.
     * This saves the user from waiting for model load (~2s) only to discover
     * their input file is MP4/MP3/wrong format. */
    if (ref_audio) {
        /* Extension check */
        const char *ext = strrchr(ref_audio, '.');
        if (ext && (strcasecmp(ext, ".mp4") == 0 || strcasecmp(ext, ".m4a") == 0 ||
                    strcasecmp(ext, ".mp3") == 0 || strcasecmp(ext, ".ogg") == 0 ||
                    strcasecmp(ext, ".opus") == 0 || strcasecmp(ext, ".flac") == 0 ||
                    strcasecmp(ext, ".aac") == 0 || strcasecmp(ext, ".wma") == 0 ||
                    strcasecmp(ext, ".webm") == 0 || strcasecmp(ext, ".mkv") == 0 ||
                    strcasecmp(ext, ".avi") == 0 || strcasecmp(ext, ".mov") == 0)) {
            fprintf(stderr, "Error: %s is not a WAV file (detected %s format)\n", ref_audio, ext);
            fprintf(stderr, "Voice cloning requires 24 kHz WAV (PCM, 16-bit, mono).\n");
            fprintf(stderr, "Convert first:\n");
            fprintf(stderr, "  ffmpeg -i \"%s\" -ar 24000 -ac 1 output.wav\n", ref_audio);
            return 1;
        }
        /* Quick header check — read first 12 bytes */
        FILE *check_f = fopen(ref_audio, "rb");
        if (check_f) {
            unsigned char hdr[12];
            size_t n = fread(hdr, 1, 12, check_f);
            fclose(check_f);
            int bad = 0;
            if (n >= 8 && memcmp(hdr + 4, "ftyp", 4) == 0) {
                fprintf(stderr, "Error: %s is an MP4/M4A file, not a WAV file\n", ref_audio);
                bad = 1;
            } else if (n >= 3 && hdr[0] == 0xFF && (hdr[1] & 0xE0) == 0xE0) {
                fprintf(stderr, "Error: %s is an MP3 file, not a WAV file\n", ref_audio);
                bad = 1;
            } else if (n >= 4 && memcmp(hdr, "OggS", 4) == 0) {
                fprintf(stderr, "Error: %s is an OGG/Opus file, not a WAV file\n", ref_audio);
                bad = 1;
            } else if (n >= 4 && memcmp(hdr, "fLaC", 4) == 0) {
                fprintf(stderr, "Error: %s is a FLAC file, not a WAV file\n", ref_audio);
                bad = 1;
            } else if (n >= 3 && memcmp(hdr, "ID3", 3) == 0) {
                fprintf(stderr, "Error: %s is an MP3 file (ID3 tagged), not a WAV file\n", ref_audio);
                bad = 1;
            } else if (n >= 4 && memcmp(hdr, "RIFF", 4) != 0) {
                fprintf(stderr, "Error: %s is not a WAV file (unrecognized format)\n", ref_audio);
                bad = 1;
            }
            if (bad) {
                fprintf(stderr, "Voice cloning requires 24 kHz WAV (PCM, 16-bit, mono).\n");
                fprintf(stderr, "Convert first:\n");
                fprintf(stderr, "  ffmpeg -i \"%s\" -ar 24000 -ac 1 output.wav\n", ref_audio);
                return 1;
            }
        }
    }

    /* Fail fast if this binary was built for an ISA the CPU lacks (avoid SIGILL
     * mid-inference on x86 — e.g. an -mavx2 build on a pre-Haswell core). */
    qwen_check_runtime_isa();

    /* Initialize threading: auto-detect or user override */
    if (threads > 0) qwen_set_threads(threads);
    else qwen_init_threads();

    /* Load model — pass int8/int4/silent via env so load can quantize inline */
    qwen_tts_ctx_t *ctx = qwen_tts_load_ex(model_dir, silent, use_int8, use_int4);
    if (!ctx) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }

#if defined(QWEN_HAVE_CUDA)
    /* --gpu-selftest-talker: validate the GPU-resident fused Talker step against the CPU
     * qwen_talker_step (same weights, same deterministic embeds, each builds its own KV).
     * Correctness gate for the fused-forward epic (plan_v4). GPU builds only; needs the model. */
    if (run_gpu_selftest_talker) {
        extern int qwen_talker_step(qwen_tts_ctx_t *, float *, float *);
        extern void *qwen_cuda_talker_init(qwen_tts_ctx_t *);
        extern void  qwen_cuda_talker_step(void *, const float *, float *, int);
        extern void  qwen_cuda_talker_free(void *);
        int h = ctx->config.hidden_size, N = 6, fail = 0;
        float *emb = (float *)malloc((size_t)N * h * sizeof(float));
        float *hc  = (float *)malloc((size_t)N * h * sizeof(float));
        float *hg  = (float *)malloc((size_t)N * h * sizeof(float));
        uint64_t rng = 0x1234567ull;
        for (int i = 0; i < N * h; i++) { rng = rng * 6364136223846793005ull + 1442695040888963407ull;
            emb[i] = (float)((rng >> 40) / (double)(1u << 24)) * 2.0f - 1.0f; }
        ctx->ml_steer = NULL; ctx->kv_len = 0;
        for (int s = 0; s < N; s++) qwen_talker_step(ctx, emb + (size_t)s * h, hc + (size_t)s * h);
        void *st = qwen_cuda_talker_init(ctx);
        if (!st) { fprintf(stderr, "gpu-selftest-talker: init failed\n"); return 1; }
        for (int s = 0; s < N; s++) qwen_cuda_talker_step(st, emb + (size_t)s * h, hg + (size_t)s * h, s);
        qwen_cuda_talker_free(st);
        for (int s = 0; s < N; s++) {
            double mx = 0, ref = 0, se = 0, sr = 0; int argmax_c = 0, argmax_g = 0;
            double bc = -1e30, bg = -1e30;
            for (int i = 0; i < h; i++) {
                double c = hc[s*h+i], g = hg[s*h+i], d = fabs(c - g);
                if (d > mx) mx = d; if (fabs(c) > ref) ref = fabs(c);
                se += d*d; sr += c*c;
                if (c > bc) { bc = c; argmax_c = i; }
                if (g > bg) { bg = g; argmax_g = i; }
            }
            double rel = mx / (ref + 1e-9), rmsrel = sqrt(se) / (sqrt(sr) + 1e-9);
            printf("  step %d: max-rel=%.3e  RMS-rel=%.3e  argmax(cpu=%d gpu=%d %s)  %s\n",
                   s, rel, rmsrel, argmax_c, argmax_g, argmax_c==argmax_g?"match":"FORK",
                   rmsrel < 2e-3 ? "PASS" : "FAIL");
            if (!(rmsrel < 2e-3)) fail = 1;
        }
        printf("gpu-selftest-talker: %s\n", fail ? "FAIL" : "PASS");
        free(emb); free(hc); free(hg);
        qwen_tts_unload(ctx);
        return fail;
    }
    /* --gpu-batch-bench N: batched fused Talker — correctness (batched row == single) +
     * throughput scaling (B seqs/frame vs 1). Needs the model + --int8/--quant-mixed. */
    if (run_gpu_batch_bench) {
        extern int qwen_cuda_batch_selftest(qwen_tts_ctx_t *, int, int);
        int rc = qwen_cuda_batch_selftest(ctx, gpu_batch_B, 200);
        qwen_tts_unload(ctx);
        return rc;
    }
#endif

#if defined(QWEN_HAVE_METAL)
    /* --gpu-selftest-talker (Metal): GPU-resident fused Talker step vs the CPU qwen_talker_step. */
    if (run_gpu_selftest_talker) {
        extern int qwen_talker_step(qwen_tts_ctx_t *, float *, float *);
        int h = ctx->config.hidden_size, N = 48, fail = 0;
        float *emb = (float *)malloc((size_t)N * h * sizeof(float));
        float *hc  = (float *)malloc((size_t)N * h * sizeof(float));
        float *hg  = (float *)malloc((size_t)N * h * sizeof(float));
        uint64_t rng = 0x1234567ull;
        for (int i = 0; i < N * h; i++) { rng = rng * 6364136223846793005ull + 1442695040888963407ull;
            emb[i] = (float)((rng >> 40) / (double)(1u << 24)) * 2.0f - 1.0f; }
        ctx->ml_steer = NULL; ctx->kv_len = 0;
        for (int s = 0; s < N; s++) qwen_talker_step(ctx, emb + (size_t)s * h, hc + (size_t)s * h);
        void *mc = qwen_metal_init();
        if (!mc) { fprintf(stderr, "gpu-selftest-talker(metal): metal init failed\n"); return 1; }
        void *st = qwen_metal_talker_init(mc, ctx);
        if (!st) { fprintf(stderr, "gpu-selftest-talker(metal): talker init failed\n"); return 1; }
        for (int s = 0; s < N; s++) qwen_metal_talker_step(st, emb + (size_t)s * h, hg + (size_t)s * h, s);
        /* ---- BATCHED step validation (the check that was missing last time): B replicas of the
         * SAME stream must match the single Metal step (argmax + RMS-rel). Fresh KV, pos 0..N-1. ---- */
        { int B = 4, bfail = 0;
          void *bs = qwen_metal_talker_batch_init(st, B);
          if (!bs) { fprintf(stderr, "gpu-selftest-batch(metal): batch init failed (B=%d)\n", B); fail = 1; }
          else {
            float *embB = (float*)malloc((size_t)B*h*sizeof(float));
            float *hB   = (float*)malloc((size_t)B*h*sizeof(float));
            int   *posB = (int*)malloc((size_t)B*sizeof(int));
            double worst = 0;
            for (int s = 0; s < N; s++) {
                for (int b=0;b<B;b++){ posB[b]=s; memcpy(embB+(size_t)b*h, emb+(size_t)s*h, (size_t)h*sizeof(float)); }
                qwen_metal_talker_batch_step(bs, embB, posB, hB);
                for (int b=0;b<B;b++){
                    double se=0, sr=0; int am_s=0, am_b=0; double bsg=-1e30, bbg=-1e30;
                    for (int i=0;i<h;i++){ double g=hg[(size_t)s*h+i], gb=hB[(size_t)b*h+i], d=g-gb; se+=d*d; sr+=g*g;
                        if (g>bsg){bsg=g;am_s=i;} if (gb>bbg){bbg=gb;am_b=i;} }
                    double rmsrel = sqrt(se)/(sqrt(sr)+1e-9); if (rmsrel>worst) worst=rmsrel;
                    /* Gate = argmax match (drives sampling). RMS-rel up to ~2e-2 is the benign fp fork
                     * over 28 fused layers (mv_b scalar accum vs single float4) — same as single-vs-CPU. */
                    if (am_s != am_b || rmsrel > 2.5e-2) { bfail=1;
                        printf("  BATCH step %d slot %d: RMS-rel=%.3e argmax(single=%d batch=%d) FAIL\n", s, b, rmsrel, am_s, am_b); }
                }
            }
            printf("gpu-selftest-batch (metal, B=%d): worst RMS-rel=%.3e  %s\n", B, worst, bfail?"FAIL":"PASS");
            if (bfail) fail = 1;
            free(embB); free(hB); free(posB);
            qwen_metal_talker_batch_free(bs);
          }
        }
        /* ---- PROMPT-KV test: reproduce the SERVER flow — seed a prompt KV, then ONE decode step.
         * single(upload_kv) vs batch(upload_slot, slot 0) must match. This exercises upload_slot +
         * the batch step reading pre-seeded prompt KV (the path --gpu-selftest-talker's fresh-KV run
         * does NOT cover). If this FORKs, the bug is upload_slot/batch-step; if it matches, it's the
         * server integration around them. ---- */
        {
            int pl = 20, Ln = ctx->config.num_layers, kvd = ctx->config.num_kv_heads*ctx->config.head_dim, kvm = ctx->kv_max;
            uint64_t r2 = 0xBEEF1234ull;
            for (int l=0;l<Ln;l++) for (int p=0;p<pl;p++) for (int i=0;i<kvd;i++) {
                r2 = r2*6364136223846793005ull+1442695040888963407ull; float fv=(float)((r2>>40)/(double)(1u<<24))*2.0f-1.0f;
                uint32_t u; memcpy(&u,&fv,4); ctx->kv_cache_k[(size_t)l*kvm*kvd+(size_t)p*kvd+i]=(uint16_t)(u>>16);
                r2 = r2*6364136223846793005ull+1442695040888963407ull; float fw=(float)((r2>>40)/(double)(1u<<24))*2.0f-1.0f;
                memcpy(&u,&fw,4); ctx->kv_cache_v[(size_t)l*kvm*kvd+(size_t)p*kvd+i]=(uint16_t)(u>>16);
            }
            void *ss = qwen_metal_talker_init(mc, ctx);
            qwen_metal_talker_upload_kv(ss, ctx, pl);
            float *he = (float*)malloc((size_t)h*sizeof(float));
            qwen_metal_talker_step(ss, emb, he, pl);
            void *bs2 = qwen_metal_talker_batch_init(ss, 2);
            qwen_metal_talker_batch_upload_slot(bs2, 0, ctx->kv_cache_k, ctx->kv_cache_v, kvm, pl);
            float *embB2=(float*)calloc((size_t)2*h,sizeof(float)); memcpy(embB2, emb, (size_t)h*sizeof(float));
            int posB2[2]={pl,0}; float *hB2=(float*)malloc((size_t)2*h*sizeof(float));
            qwen_metal_talker_batch_step(bs2, embB2, posB2, hB2);
            double se2=0, sr2=0; int am_s=0, am_b=0; double bsg=-1e30, bbg=-1e30;
            for (int i=0;i<h;i++){ double a=he[i], bc2=hB2[i], d=a-bc2; se2+=d*d; sr2+=a*a;
                if(a>bsg){bsg=a;am_s=i;} if(bc2>bbg){bbg=bc2;am_b=i;} }
            double rr = sqrt(se2)/(sqrt(sr2)+1e-9);
            printf("gpu-selftest-batch-promptKV (pl=%d): RMS-rel=%.3e argmax(single=%d batch=%d) %s\n",
                   pl, rr, am_s, am_b, (am_s==am_b && rr<2.5e-2)?"PASS":"FAIL");
            if (!(am_s==am_b && rr<2.5e-2)) fail=1;
            free(he); free(embB2); free(hB2);
            qwen_metal_talker_batch_free(bs2); qwen_metal_talker_free(ss);
        }
        qwen_metal_talker_free(st); qwen_metal_free(mc);
        for (int s = 0; s < N; s++) {
            double mx = 0, ref = 0, se = 0, sr = 0; int argmax_c = 0, argmax_g = 0; double bc = -1e30, bg = -1e30;
            for (int i = 0; i < h; i++) {
                double c = hc[s*h+i], g = hg[s*h+i], d = fabs(c - g);
                if (d > mx) mx = d; if (fabs(c) > ref) ref = fabs(c); se += d*d; sr += c*c;
                if (c > bc) { bc = c; argmax_c = i; } if (g > bg) { bg = g; argmax_g = i; }
            }
            double rel = mx / (ref + 1e-9), rmsrel = sqrt(se) / (sqrt(sr) + 1e-9);
            /* Gate = ARGMAX match (drives sampling); RMS-rel ~6-8e-3 is benign fp-accumulation
             * fork over 28 deep fused layers (same magnitude as the CUDA path). */
            int ok = (argmax_c == argmax_g) && (rmsrel < 1.5e-2);
            printf("  step %d: max-rel=%.3e  RMS-rel=%.3e  argmax(cpu=%d gpu=%d %s)  %s\n",
                   s, rel, rmsrel, argmax_c, argmax_g, argmax_c==argmax_g?"match":"FORK",
                   ok ? "PASS" : "FAIL");
            if (!ok) fail = 1;
        }
        printf("gpu-selftest-talker (metal): %s\n", fail ? "FAIL" : "PASS");
        free(emb); free(hc); free(hg);
        qwen_tts_unload(ctx);
        return fail;
    }
#endif

#if defined(QWEN_HAVE_METAL) || defined(QWEN_HAVE_CUDA)
    /* Optional GPU offload (opt-in): route the bf16 matvec hot path through the
     * selected backend. CPU stays the default everywhere else; passing no
     * --backend (or --backend cpu) leaves the engine 100% on the CPU path. */
    if (gpu_backend_str) {
        qwen_backend_kind_t bk = qwen_backend_kind_from_str(gpu_backend_str);
        if (bk != QWEN_BACKEND_CPU) {
            qwen_backend_t *gpu_backend = qwen_backend_init(bk);
            /* When a fused resident step is active it owns the GPU work; the per-op matvec hook
             * would only slow the OTHER components (e.g. CP) with sync round-trips → skip it. */
            int metal_fused = 0;
#if defined(QWEN_HAVE_METAL)
            metal_fused = (bk == QWEN_BACKEND_METAL && getenv("QWEN_METAL_FUSED_TALKER") != NULL);
#endif
            if (!metal_fused) qwen_backend_install_global(gpu_backend);
            fprintf(stderr, "GPU offload: bf16 matvec via '%s' backend "
                            "(EXPERIMENTAL; CPU stays default elsewhere)\n", gpu_backend->name);
#if defined(QWEN_HAVE_CUDA)
            /* Fused-forward epic (M1b): QWEN_CUDA_FUSED_TALKER=1 → run the WHOLE Talker step
             * GPU-resident (weights+KV on device, one sync/step) instead of the per-op matvec
             * hook. Decode-only; prefill stays CPU-batched + KV uploaded. Off = per-op path. */
            if (bk == QWEN_BACKEND_CUDA && getenv("QWEN_CUDA_FUSED_TALKER")) {
                extern void *g_cuda_talker_state, *g_cuda_cp_state;
                extern void *qwen_cuda_talker_init(qwen_tts_ctx_t *);
                extern void *qwen_cuda_cp_init(qwen_tts_ctx_t *);
                g_cuda_talker_state = qwen_cuda_talker_init(ctx);
                g_cuda_cp_state = qwen_cuda_cp_init(ctx);
                if (g_cuda_talker_state && g_cuda_cp_state)
                    fprintf(stderr, "GPU fused Talker+CP steps ENABLED (resident, 1 sync/step each)\n");
            }
            /* M3: route the Speech-decoder convs to cuBLAS (big matmuls) — QWEN_CUDA_DECODER=1. */
            if (bk == QWEN_BACKEND_CUDA && getenv("QWEN_CUDA_DECODER")) {
                extern int g_cuda_decoder_on;
                g_cuda_decoder_on = 1;
                fprintf(stderr, "GPU Speech-decoder convs via cuBLAS ENABLED\n");
            }
            /* M3 (real): GPU-RESIDENT ConvNet decoder — QWEN_CUDA_CONVDEC=1. */
            if (bk == QWEN_BACKEND_CUDA && getenv("QWEN_CUDA_CONVDEC")) {
                extern int g_cuda_decoder_conv_on;
                g_cuda_decoder_conv_on = 1;
                fprintf(stderr, "GPU-RESIDENT ConvNet decoder ENABLED\n");
            }
#endif
#if defined(QWEN_HAVE_METAL)
            /* G2: QWEN_METAL_FUSED_TALKER=1 → run the whole Talker step GPU-resident on Metal
             * (weights+KV in MTLBuffers, one command buffer/step) instead of the per-op hook. */
            if (bk == QWEN_BACKEND_METAL && getenv("QWEN_METAL_FUSED_TALKER")) {
                extern void *g_metal_talker_state, *g_metal_cp_state, *g_metal_cp_frame_state;
                g_metal_talker_state = qwen_metal_talker_init(gpu_backend->impl, ctx);
                /* CP: device-frame (1 sync/frame — the M1 win) by default; QWEN_METAL_CP_PERPASS = old path. */
                if (getenv("QWEN_METAL_CP_PERPASS"))
                    g_metal_cp_state = qwen_metal_cp_init(gpu_backend->impl, ctx);
                else
                    g_metal_cp_frame_state = qwen_metal_cp_frame_init(gpu_backend->impl, ctx);
                if (g_metal_talker_state && (g_metal_cp_state || g_metal_cp_frame_state))
                    fprintf(stderr, "GPU fused Talker+CP ENABLED (Metal, resident%s)\n",
                            g_metal_cp_frame_state ? ", CP device-frame" : "");
            }
#endif
        }
    }
#endif

    /* Multi-layer emotion steer (.qlsteer), if requested */
    if (ml_steer_path) {
        load_ml_steer(ctx, ml_steer_path, ml_steer_weight, ml_l0, ml_l1);
        ctx->ml_steer_decay = ml_decay; ctx->ml_steer_frames = ml_frames;
        if (ml_decay != 1.0f || ml_frames > 0)
            fprintf(stderr, "ML-steer schedule: decay=%.3f/frame, first-frames=%d (0=all)\n",
                    ml_decay, ml_frames);
    }

    /* --emotion DEFAULTS — match recipe_final.sh exactly: temperature 1.1 for ALL (preset AND clone), and
     * a vivid English instruct ONLY on EXPR/COMBINE cells (validated STEER cells use NO instruct — the steer
     * vector carries the emotion). instruct + temperature are CONSUMED before the late router, so set them
     * here. The user's -I / -T always override. 1.7B routed emotions only. */
    /* INLINE paralinguistics (shipped 2026-07-01): rewrite [laugh]/[sigh] → onomatopoeia IN the text so the
     * event renders in the voice's own timbre in ONE generation (no "splice" span). Done BEFORE the emotion
     * router / compose / seed so the tags become plain text and the whole sentence is one --emotion take.
     * Pin the para-validated seed 7 (both 哈哈哈 and ahh fire at 7) and T1.1 when the user gave no --seed/-T. */
    char *para_sub_text = NULL;
    if (text && !no_compose) {
        int did = 0, para_seed = 7; float para_temp = 1.1f;
        /* voice_class for para_pick: 2 = clone (--load-voice), 1 = vivian preset, 0 = ryan/other preset.
         * (clone-vs-preset matters for [yawn]: 哈啊 clone s42 / preset s7.) */
        int para_voice = load_voice ? 2 : ((speaker_name && !strcasecmp(speaker_name, "vivian")) ? 1 : 0);
        para_sub_text = qwen_compose_para_substitute(text, para_voice, &did, &para_seed, &para_temp);
        if (para_sub_text && did) {
            text = para_sub_text;
            if (seed < 0) seed = para_seed;      /* pin the validated per-tag seed (laugh 7 / sigh 42) */
            if (!temp_set) { temperature = para_temp; temp_set = 1; }   /* per-tag T (1.1 default, [scoff] 1.0) */
            if (!silent) fprintf(stderr, "Paralinguistics: inline [tag]->onomatopoeia (seed %d, T%.1f): \"%s\"\n",
                                 seed, (double)temperature, text);
        } else { free(para_sub_text); para_sub_text = NULL; }
    }

    /* Para-active (legacy): kept only for the explicit --compose per-span path; the --text auto path above
     * has already inlined [laugh]/[sigh], so text_has_para_event(text) is normally 0 here. */
    int para_active = text && !no_compose && qwen_compose_has_para_event(text);
    if (emotion_spec && !compose_spec && ctx->config.hidden_size >= 2048 && emotion_tok(emotion_spec)) {
        const char *etok = emotion_tok(emotion_spec);
        char evk[64];
        const char *evoice = emotion_voice_key(load_voice != NULL, load_voice, speaker_name, evk, sizeof(evk));
        emo_cell_t ecell; float etemp;
        resolve_emotion_recipe(language, evoice, load_voice != NULL, etok, &ecell, &etemp);
        if (!instruct && (ecell.use_expr || para_active)) {   /* instruct on EXPR/COMBINE cells, AND on para+emo */
            instruct = default_emotion_instruct(emotion_spec);
            if (instruct && !silent)
                fprintf(stderr, "Emotion '%s': default English instruct \"%s\" (override with -I)\n", emotion_spec, instruct);
        }
        if (!temp_set) {
            temperature = etemp; temp_set = 1;
            if (!silent) fprintf(stderr, "Emotion '%s': temperature %.1f (override with -T)\n", emotion_spec, temperature);
        }
    }

    /* Emo-bump: when emotion is requested (--expr or --instruct) and the user did NOT set -T,
     * raise the temperature so the emotion isn't flat — the neutral 0.9 default reads weak for
     * emotional delivery (the recipe wants ~1.1–1.3). Neutral speech (no expr/instruct) keeps the
     * stable 0.9 default. An explicit -T always wins. */
    if (!temp_set && (expr_path || instruct)) {
        temperature = 1.1f;
        if (!silent) fprintf(stderr, "Emotion requested without -T -> temperature %.1f (emo-bump)\n", temperature);
    }

    /* Set parameters */
    ctx->temperature = temperature;
    ctx->top_k = top_k;
    ctx->top_p = top_p;
    ctx->rep_penalty = rep_penalty;
    ctx->max_tokens = max_tokens;
    ctx->debug = debug;

    if (speaker_id >= 0) ctx->speaker_id = speaker_id;
    if (language) ctx->language_id = qwen_tts_language_id(language);
    if (seed >= 0) ctx->seed = (uint32_t)seed;
    /* Echo the resolved seed so a good (esp. emotional) take is reproducible: without --seed the
     * engine uses a time-based seed (qwen_tts.c: ctx->seed=time(NULL)) that would otherwise be
     * invisible. Pass the printed value back via --seed to reproduce/curate "good seeds". */
    if (!silent) fprintf(stderr, "seed: %u%s\n", ctx->seed, seed >= 0 ? "" : " (auto/time-based)");
    if (max_duration > 0) ctx->max_tokens = (int)(max_duration * 12.5f);
    if (ctx_greedy_warmup > 0) ctx->greedy_warmup = ctx_greedy_warmup;
    if (icl_frames > 0) ctx->icl_frames_cap = icl_frames;
    ctx->graft_mode = graft;

    /* ---- Expressivity controls (feat/expressivity) ----
     * --emotion accepts a compound MOOD name (joy/sad/stern/...) resolved through
     * the manifest to a full recipe {vec, steer_weight, roughness, volume, rate};
     * any explicitly-passed knob overrides its baked value. A blend/scale spec
     * (e.g. "happy:0.5,proud:0.5") bypasses the manifest and steers raw presets.
     * (--compose calls the same helper per span; see below.) */
    /* Auto-route a tagged --text (e.g. "Ciao [sad] ... [sigh]") through the
     * inline-markup composer, so users get expressive markup without a new flag. */
    int compose_from_text = 0;
    if (!compose_spec && !no_compose && text && qwen_compose_has_markup(text)) {
        compose_spec = text;
        compose_from_text = 1;   /* inline [tags] in --text: the WHOLE text is one --emotion, so keep the
                                  * routed emotion's global expr/steer (applied below) — the compose loop
                                  * preserves ctx->ml_steer across spoken spans while each [tag] span swaps
                                  * in its own paralinguistic vector. (An explicit --compose spec is per-span
                                  * and does NOT get the blanket emotion router.) */
        if (!silent) fprintf(stderr, "Inline markup detected in --text -> compose mode\n");
    }
    if (no_compose && !silent && text && qwen_compose_has_markup(text))
        fprintf(stderr, "--no-compose: passing inline [tags] literally to the model\n");
    /* On the 1.7B model, a routed --emotion (anger/sad/joy/fear/disgust/surprise) is handled by the
     * COMBINE auto-router AFTER the voice block (expr+qlsteer), NOT the legacy .vec CP-steer here —
     * skip it so we don't double-apply. The inline-tag/compose path keeps using qwen_apply_emotion. */
    int emotion_routed = emotion_spec && !compose_spec &&
                         ctx->config.hidden_size >= 2048 && emotion_tok(emotion_spec) != NULL;
    if (!compose_spec && !emotion_routed &&
        qwen_tts_apply_emotion(ctx, emotion_spec, language,
                           cp_roughness, roughness_set,
                           audio_volume, volume_set, audio_rate, rate_set,
                           &audio_volume, &audio_rate, silent) != 0) {
        qwen_tts_unload(ctx);
        return 1;
    }

    if (voice_design) {
        if (ctx->config.hidden_size < 2048) {
            fprintf(stderr, "Error: --voice-design requires the 1.7B VoiceDesign model\n");
            fprintf(stderr, "Download it with: ./download_model.sh --model voice-design\n");
            qwen_tts_unload(ctx);
            return 1;
        }
        ctx->voice_design = 1;
    }
    /* Voice cloning setup */
    if (ref_audio || load_voice) {
        if (!ctx->is_base_model && ref_audio) {
            /* --ref-audio requires speaker encoder (Base model only) */
            fprintf(stderr, "Error: --ref-audio requires a Base model (not CustomVoice)\n");
            fprintf(stderr, "Extract a speaker embedding first with the Base model:\n");
            fprintf(stderr, "  ./qwen_tts -d qwen3-tts-1.7b-base --ref-audio %s --save-voice voice.bin\n", ref_audio);
            fprintf(stderr, "Then use it here with --load-voice voice.bin\n");
            qwen_tts_unload(ctx);
            return 1;
        }
        if (!ctx->is_base_model && load_voice) {
            /* Cross-model voice injection: use ECAPA-TDNN embedding from Base model
             * in CustomVoice/VoiceDesign model. This works because the embedding spaces
             * are compatible (cosine similarity ~0.94 between ECAPA and discrete speakers). */
            if (!silent)
                fprintf(stderr, "Cross-model voice: loading speaker embedding into %s model\n",
                        ctx->voice_design ? "VoiceDesign" : "CustomVoice");
        }
        ctx->voice_clone = 1;
        ctx->xvector_only = xvector_only ? 1 : (ref_text_str ? 0 : 1);
        ctx->max_ref_seconds = max_ref_duration;
        if (ref_audio) ctx->ref_audio_path = strdup(ref_audio);
        if (ref_text_str) ctx->ref_text = strdup(ref_text_str);

        /* Use speaker encoder dim if available, otherwise model hidden_size */
        int enc_dim = ctx->speaker_enc.enc_dim > 0 ? ctx->speaker_enc.enc_dim : ctx->config.hidden_size;
        ctx->speaker_embedding = (float *)malloc(enc_dim * sizeof(float));
        if (!ctx->speaker_embedding) {
            fprintf(stderr, "Error: failed to allocate speaker embedding\n");
            qwen_tts_unload(ctx);
            return 1;
        }

        /* Check if file has .qvoice extension */
        int load_is_qvoice = load_voice && strlen(load_voice) > 7 &&
            strcmp(load_voice + strlen(load_voice) - 7, ".qvoice") == 0;
        int save_is_qvoice = save_voice && strlen(save_voice) > 7 &&
            strcmp(save_voice + strlen(save_voice) - 7, ".qvoice") == 0;

        if (load_voice) {
            if (load_is_qvoice) {
                /* Load .qvoice file: speaker embedding + ref_codes + ref_text */
                FILE *vf = fopen(load_voice, "rb");
                if (!vf) {
                    fprintf(stderr, "Error: cannot open voice file %s\n", load_voice);
                    qwen_tts_unload(ctx);
                    return 1;
                }
                /* Read and validate magic */
                char magic[4];
                uint32_t version;
                if (fread(magic, 1, 4, vf) != 4 || memcmp(magic, "QVCE", 4) != 0) {
                    fprintf(stderr, "Error: %s is not a valid .qvoice file\n", load_voice);
                    fclose(vf); qwen_tts_unload(ctx); return 1;
                }
                if (fread(&version, sizeof(uint32_t), 1, vf) != 1 || (version < 1 || version > 3)) {
                    fprintf(stderr, "Error: unsupported .qvoice version %u\n", version);
                    fclose(vf); qwen_tts_unload(ctx); return 1;
                }
                /* Read speaker embedding (v2 has enc_dim header, v1 assumes model's enc_dim) */
                int file_enc_dim = enc_dim;
                if (version >= 2) {
                    uint32_t d;
                    if (fread(&d, sizeof(uint32_t), 1, vf) != 1) {
                        fprintf(stderr, "Error: failed to read enc_dim from %s\n", load_voice);
                        fclose(vf); qwen_tts_unload(ctx); return 1;
                    }
                    file_enc_dim = (int)d;
                }
                if (file_enc_dim != enc_dim) {
                    fprintf(stderr, "Error: .qvoice has enc_dim=%d but model expects %d\n", file_enc_dim, enc_dim);
                    fprintf(stderr, "Re-create the .qvoice with the matching Base model.\n");
                    fclose(vf); qwen_tts_unload(ctx); return 1;
                }
                if (fread(ctx->speaker_embedding, sizeof(float), enc_dim, vf) != (size_t)enc_dim) {
                    fprintf(stderr, "Error: failed to read speaker embedding from %s\n", load_voice);
                    fclose(vf); qwen_tts_unload(ctx); return 1;
                }
                /* Read ref_text */
                uint32_t ref_text_len;
                if (fread(&ref_text_len, sizeof(uint32_t), 1, vf) != 1) {
                    fclose(vf); qwen_tts_unload(ctx); return 1;
                }
                if (ref_text_len > 0) {
                    char *rt = (char *)malloc(ref_text_len + 1);
                    if (fread(rt, 1, ref_text_len, vf) != ref_text_len) {
                        free(rt); fclose(vf); qwen_tts_unload(ctx); return 1;
                    }
                    rt[ref_text_len] = '\0';
                    free(ctx->ref_text);
                    ctx->ref_text = rt;
                }
                /* Read ref_codes */
                uint32_t n_ref_frames;
                if (fread(&n_ref_frames, sizeof(uint32_t), 1, vf) != 1) {
                    fclose(vf); qwen_tts_unload(ctx); return 1;
                }
                if (n_ref_frames > 0) {
                    /* Leaks-audit fix (#2 CRIT): n_ref_frames is untrusted (read from file).
                     * The old `int n_codes = (int)n_ref_frames*16` overflowed (signed UB) for
                     * n_ref_frames > ~134M, yielding a negative/undersized malloc and then an
                     * out-of-bounds fread = heap overflow (no NULL-check either). Cap to a sane
                     * max (ref is 12.5 Hz; 1e6 frames ≈ 22 h, 64 MB), use size_t, NULL-check.
                     * A truncated/lying-but-in-range count is still caught by the fread mismatch. */
                    const uint32_t MAX_REF_FRAMES = 1000000u;
                    if (n_ref_frames > MAX_REF_FRAMES) {
                        fprintf(stderr, "Error: .qvoice ref frame count %u exceeds max %u (corrupt file?)\n",
                                n_ref_frames, MAX_REF_FRAMES);
                        fclose(vf); qwen_tts_unload(ctx); return 1;
                    }
                    size_t n_codes = (size_t)n_ref_frames * 16;
                    ctx->cached_ref_codes = (int *)malloc(n_codes * sizeof(int));
                    if (!ctx->cached_ref_codes) {
                        fprintf(stderr, "Error: out of memory allocating %zu ref codes\n", n_codes);
                        fclose(vf); qwen_tts_unload(ctx); return 1;
                    }
                    if (fread(ctx->cached_ref_codes, sizeof(int), n_codes, vf) != n_codes) {
                        free(ctx->cached_ref_codes);
                        ctx->cached_ref_codes = NULL;
                        fclose(vf); qwen_tts_unload(ctx); return 1;
                    }
                    ctx->cached_ref_n_frames = (int)n_ref_frames;
                    ctx->xvector_only = 0;  /* ICL mode */
                }
                /* Save original mmap'd pointers for WDELTA (before WOVR modifies them) */
                uint16_t *orig_tok_emb = ctx->tok_embeddings_bf16;
                uint16_t *orig_fc1 = ctx->text_proj_fc1_bf16;
                uint16_t *orig_fc2 = ctx->text_proj_fc2_bf16;
                uint16_t *orig_codec = ctx->codec_embedding_bf16;
                uint16_t *orig_codec_head = ctx->codec_head_bf16;
                (void)orig_tok_emb; (void)orig_fc1; (void)orig_fc2;
                (void)orig_codec; (void)orig_codec_head;

                /* v3 metadata */
                char meta_lang_name[16] = {0};
                char meta_voice_name[64] = {0};
                uint32_t meta_lang_id = 0;
                uint32_t meta_model_size = 0;
                float meta_ref_dur = 0;
                int has_meta = 0;
                int has_tpad = 0;
                if (version >= 3) {
                    char meta_magic[4];
                    if (fread(meta_magic, 1, 4, vf) == 4 && memcmp(meta_magic, "META", 4) == 0) {
                        has_meta = 1;
                        fread(&meta_lang_id, sizeof(uint32_t), 1, vf);
                        fread(meta_lang_name, 1, 16, vf);
                        meta_lang_name[15] = '\0';
                        fread(&meta_model_size, sizeof(uint32_t), 1, vf);
                        uint32_t meta_enc_dim;
                        fread(&meta_enc_dim, sizeof(uint32_t), 1, vf);
                        fread(&meta_ref_dur, sizeof(float), 1, vf);
                        fread(meta_voice_name, 1, 64, vf);
                        meta_voice_name[63] = '\0';
                        uint32_t meta_flags;
                        fread(&meta_flags, sizeof(uint32_t), 1, vf);
                        /* Auto-set language from metadata if not specified on CLI */
                        if (!language && meta_lang_id > 0) {
                            ctx->language_id = (int)meta_lang_id;
                            if (!silent)
                                fprintf(stderr, "  Auto-set language from voice: %s\n", meta_lang_name);
                        }
                        /* Warn if CLI language doesn't match voice metadata */
                        if (language && meta_lang_id > 0 && ctx->language_id != (int)meta_lang_id) {
                            fprintf(stderr, "WARNING: voice was created with language '%s' but you specified '%s'\n",
                                    meta_lang_name, language);
                            fprintf(stderr, "  Voice fidelity may be reduced. Consider using -l %s\n", meta_lang_name);
                        }
                        /* Warn if model size doesn't match */
                        if (meta_model_size > 0 && meta_model_size != (uint32_t)ctx->config.hidden_size) {
                            fprintf(stderr, "WARNING: voice was created with model hidden=%u but current model has hidden=%d\n",
                                    meta_model_size, ctx->config.hidden_size);
                            fprintf(stderr, "  Cross-size injection may reduce quality.\n");
                        }
                    }
                    /* TPAD section: source model's tts_pad/bos/eos embeddings. */
                    char tpad_magic[4];
                    if (fread(tpad_magic, 1, 4, vf) == 4 && memcmp(tpad_magic, "TPAD", 4) == 0) {
                        uint32_t tpad_hidden;
                        if (fread(&tpad_hidden, sizeof(uint32_t), 1, vf) == 1 &&
                            (int)tpad_hidden == ctx->config.hidden_size &&
                            ctx->cached_tts_pad_embed) {
                            int h = ctx->config.hidden_size;
                            fread(ctx->cached_tts_pad_embed, sizeof(float), h, vf);
                            fread(ctx->cached_tts_bos_embed, sizeof(float), h, vf);
                            fread(ctx->cached_tts_eos_embed, sizeof(float), h, vf);
                            has_tpad = 1;
                            if (!silent)
                                fprintf(stderr, "  Loaded source tts_pad/bos/eos embeddings\n");
                        }
                    }
                    /* WOVR section: source model weights override.
                     * Replaces text_projection and codec_embedding with source model's
                     * weights to eliminate ALL per-frame divergence from weight diffs. */
                    char wovr_magic[4];
                    if (fread(wovr_magic, 1, 4, vf) == 4 && memcmp(wovr_magic, "WOVR", 4) == 0) {
                        uint32_t wh, wth, wcv;
                        if (fread(&wh, sizeof(uint32_t), 1, vf) == 1 &&
                            fread(&wth, sizeof(uint32_t), 1, vf) == 1 &&
                            fread(&wcv, sizeof(uint32_t), 1, vf) == 1 &&
                            (int)wh == ctx->config.hidden_size &&
                            (int)wth == ctx->config.text_hidden_size) {
                            int h = (int)wh, th = (int)wth, cv = (int)wcv;
                            /* graft experiment (QWEN_GRAFT_NO_WOVR, --icl-only only): skip the base-model
                             * text_projection + codec_embedding override so the graft keeps ALL CV weights
                             * (pure-CV graft, x-vector only). Tests whether the ~25MB WOVR is needed for the
                             * graft, or whether the 8KB x-vector alone suffices → minimal "qvoice-lite". */
                            int graft_no_wovr = (getenv("QWEN_GRAFT_NO_WOVR") != NULL);
                            if (graft_no_wovr) {
                                long body = (long)th*th*2 + (long)th*4 + (long)h*th*2 + (long)h*4 + (long)cv*h*2;
                                fseek(vf, body, SEEK_CUR);
                                if (!silent) fprintf(stderr, "  QWEN_GRAFT_NO_WOVR: skipped WOVR override (pure-CV graft)\n");
                            }
                            /* Allocate owned copies (can't write to mmap'd weights) */
                            uint16_t *fc1 = graft_no_wovr ? NULL : (uint16_t *)malloc((size_t)th * th * sizeof(uint16_t));
                            float *fc1_b = (float *)malloc((size_t)th * sizeof(float));
                            uint16_t *fc2 = (uint16_t *)malloc((size_t)h * th * sizeof(uint16_t));
                            float *fc2_b = (float *)malloc((size_t)h * sizeof(float));
                            uint16_t *ce = (uint16_t *)malloc((size_t)cv * h * sizeof(uint16_t));
                            if (fc1 && fc1_b && fc2 && fc2_b && ce) {
                                fread(fc1, sizeof(uint16_t), (size_t)th * th, vf);
                                fread(fc1_b, sizeof(float), th, vf);
                                fread(fc2, sizeof(uint16_t), (size_t)h * th, vf);
                                fread(fc2_b, sizeof(float), h, vf);
                                fread(ce, sizeof(uint16_t), (size_t)cv * h, vf);
                                /* Override model weights */
                                ctx->text_proj_fc1_bf16 = fc1;
                                ctx->text_proj_fc1_bias = fc1_b;
                                ctx->text_proj_fc2_bf16 = fc2;
                                ctx->text_proj_fc2_bias = fc2_b;
                                /* leaks-audit #3: these malloc'd overrides replace mmap ptrs → unload must free them
                                 * (the fail-path `else` below frees them only when NOT assigned → no double-free). */
                                qwen_track_override(ctx, fc1);  qwen_track_override(ctx, fc1_b);
                                qwen_track_override(ctx, fc2);  qwen_track_override(ctx, fc2_b);
                                /* Override codec_embedding: need owned copy since original is mmap'd */
                                int full_codec_vocab = ctx->config.codec_vocab_size;
                                uint16_t *ce_full = (uint16_t *)malloc((size_t)full_codec_vocab * h * sizeof(uint16_t));
                                if (ce_full) {
                                    /* Copy entire original table (includes speaker presets) */
                                    memcpy(ce_full, ctx->codec_embedding_bf16, (size_t)full_codec_vocab * h * sizeof(uint16_t));
                                    /* Override codebook entries 0-2047 with source model's */
                                    int copy_entries = cv < full_codec_vocab ? cv : full_codec_vocab;
                                    memcpy(ce_full, ce, (size_t)copy_entries * h * sizeof(uint16_t));
                                    ctx->codec_embedding_bf16 = ce_full; qwen_track_override(ctx, ce_full);  /* leaks-audit #3 */
                                }
                                free(ce);
                                /* Recompute tts_pad/bos/eos with new text_projection
                                 * (only if TPAD section wasn't loaded — TPAD has exact Base values) */
                                int tts_pad_id = 151671, tts_bos_id = 151672, tts_eos_id = 151673;
                                float *tmp1 = (float *)malloc(th * sizeof(float));
                                float *tmp2 = (float *)malloc(th * sizeof(float));
                                if (tmp1 && tmp2 && !has_tpad) {
                                    for (int tid_i = 0; tid_i < 3; tid_i++) {
                                        int tid = (tid_i == 0) ? tts_pad_id : (tid_i == 1) ? tts_bos_id : tts_eos_id;
                                        float *out = (tid_i == 0) ? ctx->cached_tts_pad_embed :
                                                     (tid_i == 1) ? ctx->cached_tts_bos_embed :
                                                                    ctx->cached_tts_eos_embed;
                                        /* text_embedding lookup */
                                        const uint16_t *emb = ctx->tok_embeddings_bf16 + (int64_t)tid * th;
                                        for (int j = 0; j < th; j++) {
                                            uint32_t bits = (uint32_t)emb[j] << 16;
                                            memcpy(&tmp1[j], &bits, 4);
                                        }
                                        /* fc1 + bias + SiLU */
                                        for (int i = 0; i < th; i++) {
                                            float sum = fc1_b[i];
                                            const uint16_t *row = fc1 + (size_t)i * th;
                                            for (int j = 0; j < th; j++) {
                                                uint32_t bits = (uint32_t)row[j] << 16;
                                                float w; memcpy(&w, &bits, 4);
                                                sum += w * tmp1[j];
                                            }
                                            /* SiLU = x * sigmoid(x) */
                                            tmp2[i] = sum / (1.0f + expf(-sum));
                                        }
                                        /* fc2 + bias */
                                        for (int i = 0; i < h; i++) {
                                            float sum = fc2_b[i];
                                            const uint16_t *row = fc2 + (size_t)i * th;
                                            for (int j = 0; j < th; j++) {
                                                uint32_t bits = (uint32_t)row[j] << 16;
                                                float w; memcpy(&w, &bits, 4);
                                                sum += w * tmp2[j];
                                            }
                                            out[i] = sum;
                                        }
                                    }
                                }
                                free(tmp1); free(tmp2);
                                if (!silent) {
                                    int64_t wovr_bytes = (int64_t)th*th*2 + th*4 + (int64_t)h*th*2 + h*4 + (int64_t)cv*h*2;
                                    fprintf(stderr, "  Loaded source model weights (%.1f MB) — full cross-model fidelity\n",
                                            wovr_bytes / 1024.0f / 1024.0f);
                                    fprintf(stderr, "  tts_pad_embed[:3]=[%.6f,%.6f,%.6f] (recomputed from source weights)\n",
                                            ctx->cached_tts_pad_embed[0], ctx->cached_tts_pad_embed[1], ctx->cached_tts_pad_embed[2]);
                                }
                            } else {
                                free(fc1); free(fc1_b); free(fc2); free(fc2_b); free(ce);
                            }
                        }
                    }
                    /* WFULL section: full talker weight override from source model.
                     * Replaces ALL talker weights in the target model with source weights
                     * stored in the .qvoice file. This achieves bit-identical output to
                     * the source model without requiring it to be present. */
                    char wfull_magic[5] = {0};
                    int is_wdelta = 0;
                    size_t magic_read = fread(wfull_magic, 1, 5, vf);
                    if (magic_read == 5 && memcmp(wfull_magic, "WDLT", 4) == 0) {
                        /* WDELTA format (4-byte magic) — push back the 5th byte.
                         * Require a full 5-byte read: a real WDLT stream always has the
                         * next field after the magic, so magic_read==4 means truncated —
                         * don't fseek(-1) past the magic in that edge case. */
                        fseek(vf, -1, SEEK_CUR);
                        is_wdelta = 1;
                    }
                    if (icl_only && ((magic_read == 5 && memcmp(wfull_magic, "WFULL", 5) == 0) || is_wdelta)) {
                        /* --icl-only: SKIP the entire WDELTA/WFULL weight-swap. The speaker-embedding +
                         * ref-codes ICL prefix were already loaded above; leaving the model's own (CV)
                         * weights intact lets us clone via ICL WITHOUT losing CV's instruct capability. */
                        if (!silent)
                            fprintf(stderr, "  --icl-only: skipping WDELTA weight-swap (ICL prefix only, CV weights intact)\n");
                    } else if ((magic_read == 5 && memcmp(wfull_magic, "WFULL", 5) == 0) || is_wdelta) {
                        /* WDELTA: validate target model.
                         * Format v2: "WDLT" + target_hidden_size(u32) + n_tensors(u32) + ...
                         * Format v1 (legacy): "WDLT" + n_tensors(u32) + ... (no target_h)
                         * Detect: valid hidden_size is 1024 or 2048; n_tensors is ~402/404 */
                        int wdelta_has_target_h = 0;
                        if (is_wdelta) {
                            uint32_t target_h;
                            if (fread(&target_h, sizeof(uint32_t), 1, vf) == 1) {
                                if (target_h == 1024 || target_h == 2048) {
                                    /* New format with target_hidden_size field */
                                    wdelta_has_target_h = 1;
                                    if (ctx->is_base_model) {
                                        fprintf(stderr, "ERROR: this .qvoice contains weight deltas for CustomVoice,\n");
                                        fprintf(stderr, "  but you're loading it on a Base model. This would corrupt weights.\n");
                                        fprintf(stderr, "  Use --load-voice on the CustomVoice model instead:\n");
                                        fprintf(stderr, "    ./qwen_tts -d qwen3-tts-%s --load-voice %s ...\n",
                                                target_h >= 2048 ? "1.7b" : "0.6b", load_voice);
                                        fclose(vf); qwen_tts_unload(ctx); return 1;
                                    }
                                    if ((int)target_h != ctx->config.hidden_size) {
                                        fprintf(stderr, "ERROR: .qvoice was created for %s model (hidden=%u)\n",
                                                target_h >= 2048 ? "1.7B" : "0.6B", target_h);
                                        fprintf(stderr, "  but current model has hidden=%d. Recreate with matching --target-cv.\n",
                                                ctx->config.hidden_size);
                                        fclose(vf); qwen_tts_unload(ctx); return 1;
                                    }
                                } else {
                                    /* Legacy format: what we read as target_h is actually n_tensors.
                                     * Seek back so the next read picks it up as n_tensors. */
                                    fseek(vf, -4, SEEK_CUR);
                                }
                            }
                        }
                        uint32_t n_tensors;
                        fread(&n_tensors, sizeof(uint32_t), 1, vf);
                        int loaded = 0;
                        int zlib_warned = 0;
                        int64_t wfull_bytes = 0;
                        int h = ctx->config.hidden_size;
                        int th = ctx->config.text_hidden_size;
                        int nl = ctx->config.num_layers;
                        int q_dim = ctx->config.num_heads * ctx->config.head_dim;
                        int kv_dim = ctx->config.num_kv_heads * ctx->config.head_dim;
                        int inter = ctx->config.intermediate_size;

                        for (uint32_t t = 0; t < n_tensors; t++) {
                            uint16_t name_len;
                            if (fread(&name_len, sizeof(uint16_t), 1, vf) != 1) break;
                            char tname[256] = {0};
                            if (name_len >= 256) break;
                            fread(tname, 1, name_len, vf);
                            uint32_t data_bytes;
                            fread(&data_bytes, sizeof(uint32_t), 1, vf);

                            /* Match tensor name to ctx field and replace */
                            void **target_ptr = NULL;
                            int is_f32 = 0;

                            /* Global tensors */
                            if (strcmp(tname, "talker.model.text_embedding.weight") == 0)
                                target_ptr = (void **)&ctx->tok_embeddings_bf16;
                            else if (strcmp(tname, "talker.text_projection.linear_fc1.weight") == 0)
                                target_ptr = (void **)&ctx->text_proj_fc1_bf16;
                            else if (strcmp(tname, "talker.text_projection.linear_fc1.bias") == 0)
                                { target_ptr = (void **)&ctx->text_proj_fc1_bias; is_f32 = 1; }
                            else if (strcmp(tname, "talker.text_projection.linear_fc2.weight") == 0)
                                target_ptr = (void **)&ctx->text_proj_fc2_bf16;
                            else if (strcmp(tname, "talker.text_projection.linear_fc2.bias") == 0)
                                { target_ptr = (void **)&ctx->text_proj_fc2_bias; is_f32 = 1; }
                            else if (strcmp(tname, "talker.model.codec_embedding.weight") == 0)
                                target_ptr = (void **)&ctx->codec_embedding_bf16;
                            else if (strcmp(tname, "talker.codec_head.weight") == 0)
                                target_ptr = (void **)&ctx->codec_head_bf16;
                            else if (strcmp(tname, "talker.model.norm.weight") == 0)
                                { target_ptr = (void **)&ctx->talker_norm; is_f32 = 1; }
                            /* Code Predictor tensors */
                            else if (strstr(tname, "code_predictor.model.norm.weight"))
                                { target_ptr = (void **)&ctx->cp_norm; is_f32 = 1; }
                            else if (strstr(tname, "code_predictor.small_to_mtp_projection.weight"))
                                target_ptr = (void **)&ctx->cp_mtp_proj_bf16;
                            else if (strstr(tname, "code_predictor.small_to_mtp_projection.bias"))
                                { target_ptr = (void **)&ctx->cp_mtp_proj_bias; is_f32 = 1; }
                            else if (strstr(tname, "code_predictor.model.codec_embedding.")) {
                                int ci = -1;
                                sscanf(tname, "talker.code_predictor.model.codec_embedding.%d.", &ci);
                                if (ci >= 0 && ci < 15)
                                    target_ptr = (void **)&ctx->cp_codec_emb_bf16[ci];
                            }
                            else if (strstr(tname, "code_predictor.lm_head.")) {
                                int ci = -1;
                                sscanf(tname, "talker.code_predictor.lm_head.%d.", &ci);
                                if (ci >= 0 && ci < 15)
                                    target_ptr = (void **)&ctx->cp_lm_head_bf16[ci];
                            }
                            else if (strstr(tname, "code_predictor.model.layers.")) {
                                int layer = -1;
                                sscanf(tname, "talker.code_predictor.model.layers.%d.", &layer);
                                if (layer >= 0 && layer < ctx->config.cp_num_layers) {
                                    qwen_cp_layer_t *cl = &ctx->cp_layers[layer];
                                    /* Find last occurrence of the weight type suffix */
                                    char *suffix = strstr(tname, "self_attn.");
                                    if (!suffix) suffix = strstr(tname, "mlp.");
                                    if (!suffix) suffix = strstr(tname, "input_layernorm");
                                    if (!suffix) suffix = strstr(tname, "post_attention");
                                    if (suffix) {
                                        if (strstr(suffix, "q_proj.weight"))
                                            target_ptr = (void **)&cl->wq_bf16;
                                        else if (strstr(suffix, "k_proj.weight"))
                                            target_ptr = (void **)&cl->wk_bf16;
                                        else if (strstr(suffix, "v_proj.weight"))
                                            target_ptr = (void **)&cl->wv_bf16;
                                        else if (strstr(suffix, "o_proj.weight"))
                                            target_ptr = (void **)&cl->wo_bf16;
                                        else if (strstr(suffix, "q_norm.weight"))
                                            { target_ptr = (void **)&cl->q_norm; is_f32 = 1; }
                                        else if (strstr(suffix, "k_norm.weight"))
                                            { target_ptr = (void **)&cl->k_norm; is_f32 = 1; }
                                        else if (strstr(suffix, "input_layernorm.weight"))
                                            { target_ptr = (void **)&cl->input_norm; is_f32 = 1; }
                                        else if (strstr(suffix, "post_attention_layernorm.weight"))
                                            { target_ptr = (void **)&cl->post_attn_norm; is_f32 = 1; }
                                        else if (strstr(suffix, "gate_proj.weight"))
                                            target_ptr = (void **)&cl->gate_bf16;
                                        else if (strstr(suffix, "up_proj.weight"))
                                            target_ptr = (void **)&cl->up_bf16;
                                        else if (strstr(suffix, "down_proj.weight"))
                                            target_ptr = (void **)&cl->down_bf16;
                                    }
                                }
                            }
                            else {
                                /* Per-layer tensors: talker.model.layers.N.xxx */
                                int layer = -1;
                                sscanf(tname, "talker.model.layers.%d.", &layer);
                                if (layer >= 0 && layer < nl) {
                                    qwen_talker_layer_t *l = &ctx->layers[layer];
                                    char *suffix = strstr(tname, "self_attn.");
                                    if (!suffix) suffix = strstr(tname, "mlp.");
                                    if (!suffix) suffix = strstr(tname, "input_layernorm");
                                    if (!suffix) suffix = strstr(tname, "post_attention");
                                    if (suffix) {
                                        if (strstr(suffix, "q_proj.weight"))
                                            target_ptr = (void **)&l->wq_bf16;
                                        else if (strstr(suffix, "k_proj.weight"))
                                            target_ptr = (void **)&l->wk_bf16;
                                        else if (strstr(suffix, "v_proj.weight"))
                                            target_ptr = (void **)&l->wv_bf16;
                                        else if (strstr(suffix, "o_proj.weight"))
                                            target_ptr = (void **)&l->wo_bf16;
                                        else if (strstr(suffix, "q_norm.weight"))
                                            { target_ptr = (void **)&l->q_norm; is_f32 = 1; }
                                        else if (strstr(suffix, "k_norm.weight"))
                                            { target_ptr = (void **)&l->k_norm; is_f32 = 1; }
                                        else if (strstr(suffix, "input_layernorm.weight"))
                                            { target_ptr = (void **)&l->input_norm; is_f32 = 1; }
                                        else if (strstr(suffix, "post_attention_layernorm.weight"))
                                            { target_ptr = (void **)&l->post_attn_norm; is_f32 = 1; }
                                        else if (strstr(suffix, "gate_proj.weight"))
                                            target_ptr = (void **)&l->gate_bf16;
                                        else if (strstr(suffix, "up_proj.weight"))
                                            target_ptr = (void **)&l->up_bf16;
                                        else if (strstr(suffix, "down_proj.weight"))
                                            target_ptr = (void **)&l->down_bf16;
                                    }
                                }
                            }

                            /* Read dtype flag and compressed size */
                            uint8_t dtype_flag = 0;
                            uint32_t compressed_size = data_bytes;
                            fread(&dtype_flag, 1, 1, vf);
                            fread(&compressed_size, sizeof(uint32_t), 1, vf);

                            if (target_ptr) {
                                if (dtype_flag == 4) {
                                    /* WDELTA: LZ4-compressed int16 deltas vs CV weights */
                                    uint8_t *lz4_data = (uint8_t *)malloc(compressed_size);
                                    size_t n16 = data_bytes / 2;
                                    uint16_t *result = (uint16_t *)malloc(data_bytes);
                                    int16_t *delta16 = (int16_t *)malloc(n16 * sizeof(int16_t));
                                    /* Use ORIGINAL mmap'd CV weight (before WOVR modified it). */
                                    const uint16_t *cv_orig = (const uint16_t *)*target_ptr;
                                    if (strcmp(tname, "talker.model.text_embedding.weight") == 0)
                                        cv_orig = orig_tok_emb;
                                    else if (strcmp(tname, "talker.text_projection.linear_fc1.weight") == 0)
                                        cv_orig = orig_fc1;
                                    else if (strcmp(tname, "talker.text_projection.linear_fc2.weight") == 0)
                                        cv_orig = orig_fc2;
                                    else if (strcmp(tname, "talker.model.codec_embedding.weight") == 0)
                                        cv_orig = orig_codec;
                                    else if (strcmp(tname, "talker.codec_head.weight") == 0)
                                        cv_orig = orig_codec_head;
                                    if (lz4_data && result && delta16 &&
                                        fread(lz4_data, 1, compressed_size, vf) == compressed_size) {
                                        LZ4_decompress_safe((const char *)lz4_data, (char *)delta16,
                                                             (int)compressed_size, (int)(n16 * sizeof(int16_t)));
                                        float a = qvoice_tensor_alpha(tname, voice_strength, vs_l0, vs_l1);
                                        if (a == 1.0f) {
                                            for (size_t i = 0; i < n16; i++)
                                                result[i] = (uint16_t)((int)cv_orig[i] + (int)delta16[i]);
                                        } else {
                                            /* Partial voice on this tensor: real-value lerp CV -> cloned
                                             * by alpha (more instruct/emotion response, less fidelity). */
                                            for (size_t i = 0; i < n16; i++) {
                                                float vc  = main_bf16_to_f32(cv_orig[i]);
                                                float vcl = main_bf16_to_f32((uint16_t)((int)cv_orig[i] + (int)delta16[i]));
                                                result[i] = main_f32_to_bf16(vc + a * (vcl - vc));
                                            }
                                        }
                                        *target_ptr = result; qwen_track_override(ctx, result);  /* leaks-audit #3 */
                                        loaded++;
                                        wfull_bytes += compressed_size;
                                    } else {
                                        free(result);
                                        fseek(vf, compressed_size, SEEK_CUR);
                                    }
                                    free(lz4_data); free(delta16);
                                } else if (dtype_flag == 2 || dtype_flag == 3) {
                                    /* Legacy zlib-compressed deltas — no longer supported.
                                     * Print error once, then silently skip remaining tensors. */
                                    if (!zlib_warned) {
                                        zlib_warned = 1;
                                        fprintf(stderr, "Error: .qvoice uses legacy zlib compression (dtype=%d).\n", dtype_flag);
                                        fprintf(stderr, "  Recreate with --target-cv to use LZ4. Skipping %u delta tensors.\n",
                                                n_tensors);
                                    }
                                    fseek(vf, compressed_size, SEEK_CUR);
                                } else {
                                    /* WFULL: raw data, just read and replace */
                                    void *buf = malloc(compressed_size);
                                    if (buf && fread(buf, 1, compressed_size, vf) == compressed_size) {
                                        *target_ptr = buf; qwen_track_override(ctx, buf);  /* leaks-audit #3 */
                                        loaded++;
                                        wfull_bytes += compressed_size;
                                    } else {
                                        free(buf);
                                        fseek(vf, compressed_size, SEEK_CUR);
                                    }
                                }
                            } else {
                                /* Unknown tensor — skip */
                                fseek(vf, compressed_size, SEEK_CUR);
                            }
                        }

                        /* Rebuild gate_up_fused for all talker layers */
                        for (int li = 0; li < nl; li++) {
                            qwen_talker_layer_t *l = &ctx->layers[li];
                            if (l->gate_bf16 && l->up_bf16 && l->gate_up_fused_bf16) {
                                size_t row_bytes = (size_t)h * sizeof(uint16_t);
                                for (int r = 0; r < inter; r++) {
                                    memcpy(l->gate_up_fused_bf16 + (size_t)(2*r)*h,
                                           l->gate_bf16 + (size_t)r*h, row_bytes);
                                    memcpy(l->gate_up_fused_bf16 + (size_t)(2*r+1)*h,
                                           l->up_bf16 + (size_t)r*h, row_bytes);
                                }
                            }
                        }
                        /* Rebuild gate_up_fused for all CP layers */
                        {
                            int cp_h2 = ctx->config.cp_hidden_size;
                            int cp_inter2 = ctx->config.cp_intermediate_size;
                            for (int li = 0; li < ctx->config.cp_num_layers; li++) {
                                qwen_cp_layer_t *cl = &ctx->cp_layers[li];
                                if (cl->gate_bf16 && cl->up_bf16 && cl->gate_up_fused_bf16) {
                                    size_t row_bytes = (size_t)cp_h2 * sizeof(uint16_t);
                                    for (int r = 0; r < cp_inter2; r++) {
                                        memcpy(cl->gate_up_fused_bf16 + (size_t)(2*r)*cp_h2,
                                               cl->gate_bf16 + (size_t)r*cp_h2, row_bytes);
                                        memcpy(cl->gate_up_fused_bf16 + (size_t)(2*r+1)*cp_h2,
                                               cl->up_bf16 + (size_t)r*cp_h2, row_bytes);
                                    }
                                }
                            }
                        }

                        /* Recompute tts_pad/bos/eos from new text_embedding + text_projection */
                        if (loaded > 0 && ctx->cached_tts_pad_embed) {
                            extern void embed_one_text_token_compute(qwen_tts_ctx_t *ctx, int tid, float *out);
                            embed_one_text_token_compute(ctx, 151671, ctx->cached_tts_pad_embed);
                            embed_one_text_token_compute(ctx, 151672, ctx->cached_tts_bos_embed);
                            embed_one_text_token_compute(ctx, 151673, ctx->cached_tts_eos_embed);
                        }

                        /* WDELTA override swapped the bf16 weights (CV -> Base).
                         * In INT8 mode the int8 weights were quantized at load
                         * from the CV weights, so re-quantize from the new Base
                         * bf16 — otherwise int8 ignores the override (voice loses
                         * fidelity / ~half volume). */
                        if (loaded > 0 && ctx->use_int8) {
                            extern void qwen_talker_quantize_int8(qwen_tts_ctx_t *ctx);
                            extern void qwen_cp_quantize_int8(qwen_tts_ctx_t *ctx);
                            if (!silent)
                                fprintf(stderr, "  Re-quantizing INT8 from WDELTA-overridden weights...\n");
                            qwen_talker_quantize_int8(ctx);
                            qwen_cp_quantize_int8(ctx);
                        }

                        if (!silent && loaded > 0)
                            fprintf(stderr, "  Loaded %d/%u source talker tensors (%.1f MB) — full weight override\n",
                                    loaded, n_tensors, wfull_bytes / 1024.0f / 1024.0f);
                    }
                }
                fclose(vf);
                if (!silent) {
                    fprintf(stderr, "Voice clone: loaded .qvoice v%u from %s", version, load_voice);
                    if (ctx->cached_ref_n_frames > 0)
                        fprintf(stderr, " (%d ICL frames)", ctx->cached_ref_n_frames);
                    fprintf(stderr, "\n");
                    if (has_meta) {
                        fprintf(stderr, "  Voice: %s | Language: %s | Model: %s | Ref: %.0fs\n",
                                meta_voice_name[0] ? meta_voice_name : "(unnamed)",
                                meta_lang_name[0] ? meta_lang_name : "auto",
                                meta_model_size >= 2048 ? "1.7B" : (meta_model_size > 0 ? "0.6B" : "unknown"),
                                meta_ref_dur);
                    }
                }
            } else {
                /* Load legacy voice file: raw speaker embedding only */
                FILE *vf = fopen(load_voice, "rb");
                if (!vf) {
                    fprintf(stderr, "Error: cannot open voice file %s\n", load_voice);
                    qwen_tts_unload(ctx);
                    return 1;
                }
                size_t n = fread(ctx->speaker_embedding, sizeof(float), enc_dim, vf);
                fclose(vf);
                if ((int)n != enc_dim) {
                    fprintf(stderr, "Error: voice file has %zu floats, expected %d\n", n, enc_dim);
                    qwen_tts_unload(ctx);
                    return 1;
                }
                if (!silent) {
                    fprintf(stderr, "Voice clone: loaded speaker embedding from %s (%d floats)\n", load_voice, enc_dim);
                    /* Debug: print embedding stats */
                    float norm = 0;
                    for (int i = 0; i < enc_dim; i++) norm += ctx->speaker_embedding[i] * ctx->speaker_embedding[i];
                    norm = sqrtf(norm);
                    fprintf(stderr, "  embedding norm=%.4f, first5=[%.4f,%.4f,%.4f,%.4f,%.4f]\n",
                            norm, ctx->speaker_embedding[0], ctx->speaker_embedding[1],
                            ctx->speaker_embedding[2], ctx->speaker_embedding[3], ctx->speaker_embedding[4]);
                }
            }
        } else {
            /* Extract speaker embedding from reference audio */
            if (qwen_extract_speaker_embedding(ctx, ref_audio, ctx->speaker_embedding) != 0) {
                fprintf(stderr, "Error: failed to extract speaker embedding from %s\n", ref_audio);
                qwen_tts_unload(ctx);
                return 1;
            }
            if (!silent)
                fprintf(stderr, "Voice clone: extracted speaker embedding from %s\n", ref_audio);
        }

        /* If ICL mode (not xvector_only), load the speech encoder for ref audio encoding */
        if (!ctx->xvector_only && ref_audio) {
            if (qwen_speech_encoder_load(ctx) != 0) {
                fprintf(stderr, "Warning: failed to load speech encoder, falling back to x-vector only\n");
                ctx->xvector_only = 1;
            }
        }

        /* For .qvoice save: need to encode ref audio now to get ref_codes */
        if (save_is_qvoice && ref_audio && ref_text_str && !ctx->cached_ref_codes) {
            /* Ensure speech encoder is loaded */
            if (!ctx->xvector_only) {
                float *ref_audio_samples = NULL;
                int ref_n_samples = 0, ref_sr = 0;
                if (qwen_read_wav(ref_audio, &ref_audio_samples, &ref_n_samples, &ref_sr) == 0) {
                    int *codes = NULL;
                    int n_frames = 0;
                    if (qwen_speech_encoder_encode(ctx, ref_audio_samples, ref_n_samples,
                                                    &codes, &n_frames) == 0) {
                        ctx->cached_ref_codes = codes;
                        ctx->cached_ref_n_frames = n_frames;
                        if (!silent)
                            fprintf(stderr, "Encoded ref audio: %d ICL frames\n", n_frames);
                    }
                    free(ref_audio_samples);
                }
            }
        }

        /* Save voice file */
        if (save_voice) {
            if (save_is_qvoice) {
                /* Save .qvoice v3: enc_dim + embedding + ref_text + ref_codes + metadata */
                FILE *vf = fopen(save_voice, "wb");
                if (!vf) {
                    fprintf(stderr, "Error: cannot write voice file %s\n", save_voice);
                } else {
                    fwrite("QVCE", 1, 4, vf);
                    uint32_t version = 3;
                    fwrite(&version, sizeof(uint32_t), 1, vf);
                    uint32_t saved_dim = (uint32_t)enc_dim;
                    fwrite(&saved_dim, sizeof(uint32_t), 1, vf);
                    fwrite(ctx->speaker_embedding, sizeof(float), enc_dim, vf);
                    /* ref_text */
                    uint32_t ref_text_len = ref_text_str ? (uint32_t)strlen(ref_text_str) : 0;
                    fwrite(&ref_text_len, sizeof(uint32_t), 1, vf);
                    if (ref_text_len > 0)
                        fwrite(ref_text_str, 1, ref_text_len, vf);
                    /* ref_codes */
                    uint32_t n_ref_frames = ctx->cached_ref_codes ? (uint32_t)ctx->cached_ref_n_frames : 0;
                    fwrite(&n_ref_frames, sizeof(uint32_t), 1, vf);
                    if (n_ref_frames > 0)
                        fwrite(ctx->cached_ref_codes, sizeof(int), (int)n_ref_frames * 16, vf);
                    /* v3 metadata section */
                    fwrite("META", 1, 4, vf);
                    uint32_t lang_id = (uint32_t)(ctx->language_id >= 0 ? ctx->language_id : 0);
                    fwrite(&lang_id, sizeof(uint32_t), 1, vf);
                    /* language name (16 bytes, null-padded) */
                    char lang_name[16] = {0};
                    if (language) strncpy(lang_name, language, 15);
                    fwrite(lang_name, 1, 16, vf);
                    /* source model size: 0=unknown, 600=0.6B, 1700=1.7B */
                    uint32_t model_size = (uint32_t)ctx->config.hidden_size;
                    fwrite(&model_size, sizeof(uint32_t), 1, vf);
                    /* source enc_dim */
                    fwrite(&saved_dim, sizeof(uint32_t), 1, vf);
                    /* ref audio duration in seconds */
                    float ref_dur = max_ref_duration;
                    fwrite(&ref_dur, sizeof(float), 1, vf);
                    /* voice name (64 bytes, null-padded) */
                    char vname[64] = {0};
                    if (voice_name) strncpy(vname, voice_name, 63);
                    fwrite(vname, 1, 64, vf);
                    /* flags: bit 0=xvector_only, bit 1=has_icl, bit 2=is_base_model */
                    uint32_t flags = 0;
                    if (ctx->xvector_only) flags |= 1;
                    if (n_ref_frames > 0) flags |= 2;
                    if (ctx->is_base_model) flags |= 4;
                    fwrite(&flags, sizeof(uint32_t), 1, vf);
                    /* TPAD section: save tts_pad/bos/eos embeddings for cross-model fidelity.
                     * When a Base voice is loaded into CustomVoice, these embeddings override
                     * the target model's own, eliminating per-frame drift from micro-differences
                     * in text_projection weights. */
                    if (ctx->cached_tts_pad_embed) {
                        int h = ctx->config.hidden_size;
                        fwrite("TPAD", 1, 4, vf);
                        uint32_t hidden = (uint32_t)h;
                        fwrite(&hidden, sizeof(uint32_t), 1, vf);
                        fwrite(ctx->cached_tts_pad_embed, sizeof(float), h, vf);
                        fwrite(ctx->cached_tts_bos_embed, sizeof(float), h, vf);
                        fwrite(ctx->cached_tts_eos_embed, sizeof(float), h, vf);
                    }
                    /* WOVR section: source model weights for cross-model fidelity.
                     * Stores text_projection + codec_embedding as BF16 so any target
                     * model can override its own weights with the source model's.
                     * This eliminates per-frame drift from weight micro-differences
                     * without requiring the source Base model to be present. */
                    if (ctx->text_proj_fc1_bf16 && ctx->codec_embedding_bf16) {
                        int h = ctx->config.hidden_size;
                        int th = ctx->config.text_hidden_size;
                        int cv = 2048;  /* codebook entries only, not speaker presets */
                        fwrite("WOVR", 1, 4, vf);
                        uint32_t wh = (uint32_t)h, wth = (uint32_t)th, wcv = (uint32_t)cv;
                        fwrite(&wh, sizeof(uint32_t), 1, vf);
                        fwrite(&wth, sizeof(uint32_t), 1, vf);
                        fwrite(&wcv, sizeof(uint32_t), 1, vf);
                        /* text_proj fc1: [th × th] BF16 + [th] F32 bias */
                        fwrite(ctx->text_proj_fc1_bf16, sizeof(uint16_t), (size_t)th * th, vf);
                        if (ctx->text_proj_fc1_bias)
                            fwrite(ctx->text_proj_fc1_bias, sizeof(float), th, vf);
                        /* text_proj fc2: [h × th] BF16 + [h] F32 bias */
                        fwrite(ctx->text_proj_fc2_bf16, sizeof(uint16_t), (size_t)h * th, vf);
                        if (ctx->text_proj_fc2_bias)
                            fwrite(ctx->text_proj_fc2_bias, sizeof(float), h, vf);
                        /* codec_embedding codebook entries: [cv × h] BF16 */
                        fwrite(ctx->codec_embedding_bf16, sizeof(uint16_t), (size_t)cv * h, vf);
                        if (!silent) {
                            int64_t wovr_bytes = (int64_t)th*th*2 + th*4 + (int64_t)h*th*2 + h*4 + (int64_t)cv*h*2;
                            fprintf(stderr, "  Saved source model weights (%.1f MB) for cross-model fidelity\n",
                                    wovr_bytes / 1024.0f / 1024.0f);
                        }
                    }
                    /* WFULL section: dump ALL talker weights from source model.
                     * This enables perfect cross-model voice fidelity by replacing
                     * the target model's entire talker with the source model's weights.
                     * Size: ~840 MB for 0.6B, ~3.3 GB for 1.7B. */
                    /* WFULL/WDELTA: write talker weights for cross-model fidelity.
                     * Only written when --target-cv is specified (WDELTA mode).
                     * Without --target-cv, the .qvoice contains only TPAD+WOVR (~16MB). */
                    if (target_cv_dir) {
                        int h = ctx->config.hidden_size;
                        int th = ctx->config.text_hidden_size;
                        int nl_layers = ctx->config.num_layers;
                        int q_dim = ctx->config.num_heads * ctx->config.head_dim;
                        int kv_dim = ctx->config.num_kv_heads * ctx->config.head_dim;
                        int inter = ctx->config.intermediate_size;
                        int vocab = 151936; /* text vocab size */
                        int codec_vocab = ctx->config.codec_vocab_size;
                        int head_dim = ctx->config.head_dim;

                        /* Open target CV safetensors for WDELTA if specified */
                        FILE *cv_sf = NULL;
                        char *cv_hdr_json = NULL;
                        size_t cv_data_off = 0;
                        int use_wdelta = 0;
                        if (target_cv_dir) {
                            char cv_sf_path[512];
                            snprintf(cv_sf_path, sizeof(cv_sf_path), "%s/model.safetensors", target_cv_dir);
                            cv_sf = fopen(cv_sf_path, "rb");
                            if (cv_sf) {
                                uint64_t cv_hs;
                                fread(&cv_hs, 8, 1, cv_sf);
                                cv_hdr_json = (char *)malloc(cv_hs + 1);
                                fread(cv_hdr_json, 1, cv_hs, cv_sf);
                                cv_hdr_json[cv_hs] = '\0';
                                cv_data_off = 8 + cv_hs;
                                use_wdelta = 1;
                                if (!silent)
                                    fprintf(stderr, "  Computing deltas vs %s for WDELTA encoding\n", target_cv_dir);
                            }
                        }

                        /* Helper: find tensor offset in CV safetensors JSON header */
                        /* Returns data offset+start in cv_sf, sets *out_size. Returns -1 if not found. */
                        /* Helper macros for WFULL/WDELTA tensor writing */
                        #define WRITE_TENSOR_BF16(tname_str, ptr, nbytes) \
                            write_tensor_impl(vf, cv_sf, cv_hdr_json, cv_data_off, \
                                              tname_str, ptr, nbytes, use_wdelta, 1, \
                                              &wfull_bytes, &wfull_written)
                        #define WRITE_TENSOR_F32(tname_str, ptr, nbytes) \
                            write_tensor_impl(vf, cv_sf, cv_hdr_json, cv_data_off, \
                                              tname_str, ptr, nbytes, use_wdelta, 0, \
                                              &wfull_bytes, &wfull_written)
                        /* Legacy macro for compatibility — BF16 by default */
                        #define WRITE_TENSOR(tname_str, ptr, nbytes) \
                            WRITE_TENSOR_BF16(tname_str, ptr, nbytes)

                        fwrite(use_wdelta ? "WDLT" : "WFULL", 1, use_wdelta ? 4 : 5, vf);
                        /* For WDELTA: store target model hidden_size for validation */
                        if (use_wdelta) {
                            uint32_t target_h = (uint32_t)ctx->config.hidden_size;
                            fwrite(&target_h, sizeof(uint32_t), 1, vf);
                        }
                        int cp_nl = ctx->config.cp_num_layers;
                        int cp_h = ctx->config.cp_hidden_size;
                        int cp_inter = ctx->config.cp_intermediate_size;
                        int cp_q_dim = ctx->config.cp_num_heads * ctx->config.head_dim;
                        int cp_kv_dim = ctx->config.cp_num_kv_heads * ctx->config.head_dim;
                        /* Count: 8 global + 11 per talker layer + CP tensors + mtp_proj */
                        uint32_t n_tensors = 8 + nl_layers * 11 + 1 + 15 + 15 + cp_nl * 11;
                        if (ctx->cp_mtp_proj_bf16) n_tensors += 2;
                        fwrite(&n_tensors, sizeof(uint32_t), 1, vf);
                        int64_t wfull_bytes = 0;
                        int wfull_written = 0;

                        /* Global tensors */
                        WRITE_TENSOR("talker.model.text_embedding.weight",
                                     ctx->tok_embeddings_bf16, (size_t)vocab * th * 2);
                        WRITE_TENSOR("talker.text_projection.linear_fc1.weight",
                                     ctx->text_proj_fc1_bf16, (size_t)th * th * 2);
                        WRITE_TENSOR_F32("talker.text_projection.linear_fc1.bias",
                                     ctx->text_proj_fc1_bias, (size_t)th * 4);
                        WRITE_TENSOR("talker.text_projection.linear_fc2.weight",
                                     ctx->text_proj_fc2_bf16, (size_t)h * th * 2);
                        WRITE_TENSOR_F32("talker.text_projection.linear_fc2.bias",
                                     ctx->text_proj_fc2_bias, (size_t)h * 4);
                        WRITE_TENSOR("talker.model.codec_embedding.weight",
                                     ctx->codec_embedding_bf16, (size_t)codec_vocab * h * 2);
                        WRITE_TENSOR("talker.codec_head.weight",
                                     ctx->codec_head_bf16, (size_t)codec_vocab * h * 2);
                        WRITE_TENSOR_F32("talker.model.norm.weight",
                                     ctx->talker_norm, (size_t)h * 4);

                        /* Per-layer tensors */
                        for (int li = 0; li < nl_layers; li++) {
                            qwen_talker_layer_t *l = &ctx->layers[li];
                            char tn[256];
                            #define LT(field, suffix, sz) do { \
                                snprintf(tn, sizeof(tn), "talker.model.layers.%d.%s", li, suffix); \
                                WRITE_TENSOR(tn, l->field, sz); \
                            } while(0)
                            LT(wq_bf16, "self_attn.q_proj.weight", (size_t)q_dim * h * 2);
                            LT(wk_bf16, "self_attn.k_proj.weight", (size_t)kv_dim * h * 2);
                            LT(wv_bf16, "self_attn.v_proj.weight", (size_t)kv_dim * h * 2);
                            LT(wo_bf16, "self_attn.o_proj.weight", (size_t)h * q_dim * 2);
                            #define LTF(field, suffix, sz) do { \
                                snprintf(tn, sizeof(tn), "talker.model.layers.%d.%s", li, suffix); \
                                WRITE_TENSOR_F32(tn, l->field, sz); \
                            } while(0)
                            LTF(q_norm, "self_attn.q_norm.weight", (size_t)head_dim * 4);
                            LTF(k_norm, "self_attn.k_norm.weight", (size_t)head_dim * 4);
                            LTF(input_norm, "input_layernorm.weight", (size_t)h * 4);
                            LTF(post_attn_norm, "post_attention_layernorm.weight", (size_t)h * 4);
                            #undef LTF
                            LT(gate_bf16, "mlp.gate_proj.weight", (size_t)inter * h * 2);
                            LT(up_bf16, "mlp.up_proj.weight", (size_t)inter * h * 2);
                            LT(down_bf16, "mlp.down_proj.weight", (size_t)h * inter * 2);
                            #undef LT
                        }
                        /* Code Predictor tensors */
                        WRITE_TENSOR_F32("talker.code_predictor.model.norm.weight",
                                     ctx->cp_norm, (size_t)cp_h * 4);
                        for (int ci = 0; ci < 15; ci++) {
                            char tn2[256];
                            snprintf(tn2, sizeof(tn2), "talker.code_predictor.model.codec_embedding.%d.weight", ci);
                            if (ctx->cp_codec_emb_bf16[ci])
                                WRITE_TENSOR(tn2, ctx->cp_codec_emb_bf16[ci],
                                             (size_t)ctx->config.codebook_size * ctx->cp_emb_dim * 2);
                            snprintf(tn2, sizeof(tn2), "talker.code_predictor.lm_head.%d.weight", ci);
                            if (ctx->cp_lm_head_bf16[ci])
                                WRITE_TENSOR(tn2, ctx->cp_lm_head_bf16[ci],
                                             (size_t)ctx->config.codebook_size * cp_h * 2);
                        }
                        for (int li = 0; li < cp_nl; li++) {
                            qwen_cp_layer_t *cl = &ctx->cp_layers[li];
                            char tn2[256];
                            #define CLT(field, suffix, sz) do { \
                                snprintf(tn2, sizeof(tn2), "talker.code_predictor.model.layers.%d.%s", li, suffix); \
                                WRITE_TENSOR(tn2, cl->field, sz); \
                            } while(0)
                            CLT(wq_bf16, "self_attn.q_proj.weight", (size_t)cp_q_dim * cp_h * 2);
                            CLT(wk_bf16, "self_attn.k_proj.weight", (size_t)cp_kv_dim * cp_h * 2);
                            CLT(wv_bf16, "self_attn.v_proj.weight", (size_t)cp_kv_dim * cp_h * 2);
                            CLT(wo_bf16, "self_attn.o_proj.weight", (size_t)cp_h * cp_q_dim * 2);
                            #define CLTF(field, suffix, sz) do { \
                                snprintf(tn2, sizeof(tn2), "talker.code_predictor.model.layers.%d.%s", li, suffix); \
                                WRITE_TENSOR_F32(tn2, cl->field, sz); \
                            } while(0)
                            CLTF(q_norm, "self_attn.q_norm.weight", (size_t)head_dim * 4);
                            CLTF(k_norm, "self_attn.k_norm.weight", (size_t)head_dim * 4);
                            CLTF(input_norm, "input_layernorm.weight", (size_t)cp_h * 4);
                            CLTF(post_attn_norm, "post_attention_layernorm.weight", (size_t)cp_h * 4);
                            #undef CLTF
                            CLT(gate_bf16, "mlp.gate_proj.weight", (size_t)cp_inter * cp_h * 2);
                            CLT(up_bf16, "mlp.up_proj.weight", (size_t)cp_inter * cp_h * 2);
                            CLT(down_bf16, "mlp.down_proj.weight", (size_t)cp_h * cp_inter * 2);
                            #undef CLT
                        }
                        /* MTP projection (1.7B only: 2048→1024) */
                        if (ctx->cp_mtp_proj_bf16) {
                            WRITE_TENSOR("talker.code_predictor.small_to_mtp_projection.weight",
                                         ctx->cp_mtp_proj_bf16, (size_t)cp_h * h * 2);
                            if (ctx->cp_mtp_proj_bias)
                                WRITE_TENSOR_F32("talker.code_predictor.small_to_mtp_projection.bias",
                                             ctx->cp_mtp_proj_bias, (size_t)cp_h * 4);
                        }

                        #undef WRITE_TENSOR

                        if (cv_sf) fclose(cv_sf);
                        free(cv_hdr_json);
                        #undef WRITE_TENSOR

                        if (!silent)
                            fprintf(stderr, "  Saved %d tensors (%.1f MB%s) for full cross-model fidelity\n",
                                    wfull_written, wfull_bytes / 1024.0f / 1024.0f,
                                    use_wdelta ? " WDELTA compressed" : " WFULL raw");
                    }
                    fclose(vf);
                    if (!silent)
                        fprintf(stderr, "Saved .qvoice v3 to %s (embedding + %u ICL frames + metadata + weights)\n",
                                save_voice, n_ref_frames);
                    if (!silent && language)
                        fprintf(stderr, "  Language: %s, Voice: %s, Model: %s\n",
                                language, voice_name ? voice_name : "(unnamed)",
                                ctx->config.hidden_size >= 2048 ? "1.7B" : "0.6B");
                }
            } else {
                /* Save legacy format: raw speaker embedding only */
                FILE *vf = fopen(save_voice, "wb");
                if (!vf) {
                    fprintf(stderr, "Error: cannot write voice file %s\n", save_voice);
                } else {
                    fwrite(ctx->speaker_embedding, sizeof(float), enc_dim, vf);
                    fclose(vf);
                    if (!silent)
                        fprintf(stderr, "Saved speaker embedding to %s (%d floats)\n", save_voice, enc_dim);
                }
            }
        }

        if (!silent) {
            if (ctx->xvector_only && !ctx->cached_ref_codes)
                fprintf(stderr, "Mode: x-vector only (no reference transcription)\n");
            else if (ctx->cached_ref_codes)
                fprintf(stderr, "Mode: ICL with %d cached ref frames\n", ctx->cached_ref_n_frames);
            else
                fprintf(stderr, "Mode: ICL with ref text: \"%s\"\n", ref_text_str);
        }
    }

    if (instruct) {
        if (ctx->config.hidden_size < 2048) {
            fprintf(stderr, "Warning: --instruct is only supported on 1.7B model (ignored)\n");
        } else if (ctx->voice_clone && ctx->is_base_model) {
            fprintf(stderr, "Warning: --instruct with voice cloning on a Base model is not officially supported.\n");
            fprintf(stderr, "  For best results, extract the voice with the Base model and use it with CustomVoice:\n");
            fprintf(stderr, "    ./qwen_tts -d qwen3-tts-1.7b-base --ref-audio ref.wav --save-voice voice.bin\n");
            fprintf(stderr, "    ./qwen_tts -d qwen3-tts-1.7b --load-voice voice.bin --instruct \"...\" --text \"...\"\n");
            ctx->instruct = strdup(instruct);
        } else {
            ctx->instruct = strdup(instruct);
        }
    }

    /* Create voice only: save and exit without generating */
    if (create_voice_only) {
        if (!save_voice) {
            fprintf(stderr, "Error: --save-voice is required when no --text is provided\n");
            qwen_tts_unload(ctx);
            return 1;
        }
        if (!ref_audio) {
            fprintf(stderr, "Error: --ref-audio is required to create a voice profile\n");
            qwen_tts_unload(ctx);
            return 1;
        }
        /* Voice was already saved above in the voice clone setup block */
        if (!silent)
            fprintf(stderr, "Voice profile created. Use --load-voice to generate speech.\n");
        qwen_tts_unload(ctx);
        return 0;
    }

    /* --emotion AUTO-ROUTER: on 1.7B, --emotion <sad|joy|anger|fear|disgust|surprise> applies the
     * ear-validated per-(voice×emotion) recipe CELL (plan §8.3) — NOT a blanket combine. Each cell's
     * mode (EXPR-only / STEER-only / COMBINE) is chosen by use_expr/use_steer; adding expr to a STEER
     * cell softens the emotion (galatea anger → "sad"). Runs here (voice+lang+ctx known). Manual
     * --expr/--ml-steer override; instruct+temp were defaulted earlier. Also runs when compose mode was
     * auto-entered from inline [tags] in --text (compose_from_text) so `--emotion` + e.g. [laugh]/[sigh]
     * still emotes the spoken spans — the per-span compose loop preserves this global steer/expr. */
    if (emotion_spec && (!compose_spec || compose_from_text) && ctx->config.hidden_size >= 2048) {
        const char *tok = emotion_tok(emotion_spec);
        if (tok) {
            char vkbuf[64];
            const char *voice_key = emotion_voice_key(ctx->voice_clone, load_voice, speaker_name,
                                                      vkbuf, sizeof(vkbuf));
            /* THE recipe — per-language policy (DE/FR/ZH/JA/KO/RU/ES) or per-(voice×emotion) for IT/EN.
             * docs/emotion-THE-recipe.md is the aligned single source of truth. */
            emo_cell_t cell; float rtemp;
            resolve_emotion_recipe(language, voice_key, ctx->voice_clone, tok, &cell, &rtemp);
            /* PARA+EMO: when a paralinguistic [tag] is active, force COMBINE even on a preset STEER cell —
             * the .expr is the language-correction that keeps the EN-captured para anchor ([laugh]/[sigh]/…)
             * from drifting the accent (validated para+emo recipe). The emotion steer stays at the cell's
             * weight; the para vector rides per-span at its per-voice weight. NO-para path is unchanged. */
            int want_expr = cell.use_expr || para_active;
            float ew = cell.use_expr ? cell.expr_w : 1.0f;
            if (para_active && !cell.use_expr && !silent)
                fprintf(stderr, "Emotion '%s': para active -> COMBINE (expr language-correction) for this clip\n", emotion_spec);

            /* (1) per-language .expr (FT). Skip if the user passed --expr (their override applies below). */
            if (want_expr && !expr_path) {
                const char *ep = resolve_emotion_expr(language);
                FILE *ef = fopen(ep, "rb");
                if (ef) {
                    fclose(ef);
                    if (apply_expr_file(ctx, ep, ew, silent) != 0)
                        fprintf(stderr, "Warning: --emotion: failed to apply %s\n", ep);
                    else if (!silent)
                        fprintf(stderr, "Emotion '%s': +expr %s (weight %.1f)\n", emotion_spec, ep, ew);
                } else if (!silent) {
                    fprintf(stderr, "Note: --emotion: %s missing — run `bash download_assets.sh`\n", ep);
                }
            }
            /* (2) clean .qlsteer steer @ L21-25 (ryan_<emo> palette, all voices) */
            if (cell.use_steer && cell.steer_w > 0.0f) {
                char qs[256];
                if (resolve_emotion_qlsteer(voice_key, tok, qs, sizeof(qs)) == 0) {
                    load_ml_steer(ctx, qs, cell.steer_w, 21, 25);
                    ctx->ml_steer_decay = ml_decay; ctx->ml_steer_frames = ml_frames;
                } else if (!silent) {
                    fprintf(stderr, "Note: --emotion: no .qlsteer for '%s' — expr only\n", tok);
                }
            } else if (!silent) {
                fprintf(stderr, "Emotion '%s': expr-carried (steer disabled for this voice/emotion)\n", emotion_spec);
            }
            if (!silent) {
                const char *mode = cell.use_steer ? (want_expr ? "COMBINE" : "STEER") : "EXPR";
                fprintf(stderr, "Emotion '%s': mode=%s (voice=%s)\n", emotion_spec, mode, voice_key ? voice_key : "default");
                const char *rec = recommended_voice_for_language(language);
                if (rec && (!voice_key || !strstr(rec, voice_key)))
                    fprintf(stderr, "  tip: for %s the strongest voice is %s.\n", language, rec);
            }
        }
    }

    /* --expr: apply the expressivity weight delta on top of the loaded (preset/clone)
     * weights. After the voice block + before any generation/serve dispatch, so single,
     * batch and server paths all use the expressivity-enhanced backbone. 1.7B only. */
    if (expr_path) {
        if (ctx->config.hidden_size < 2048) {
            fprintf(stderr, "Warning: --expr is only supported on the 1.7B model (ignored)\n");
        } else if (apply_expr_file(ctx, expr_path, expr_weight, silent) != 0) {
            fprintf(stderr, "Error: failed to apply --expr file %s\n", expr_path);
            qwen_tts_unload(ctx);
            return 1;
        }
    }

    /* --batch-test: verify the OPT-IN batched Talker step matches single-stream, then exit. */
    if (run_batch_test) {
        int rc = qwen_batch_self_test(ctx);
        qwen_tts_unload(ctx);
        return rc;
    }
    /* --batch-multi-test N: run N independent requests (same text, seed+i per slot)
     * through the server batch-multi engine, write bm_<i>.wav each. Compare each to a
     * single-stream run with --seed (seed+i) to validate per-slot RNG independence +
     * no cross-talk. Exits. */
    if (batch_multi_test > 0) {
        int B = batch_multi_test;
        qwen_batch_req_t *reqs = (qwen_batch_req_t *)calloc(B, sizeof(qwen_batch_req_t));
        for (int b = 0; b < B; b++) {
            reqs[b].text = text;
            reqs[b].speaker_id = ctx->speaker_id;
            reqs[b].language_id = ctx->language_id;
            reqs[b].temperature = ctx->temperature;
            reqs[b].top_k = ctx->top_k;
            reqs[b].top_p = ctx->top_p;
            reqs[b].rep_penalty = ctx->rep_penalty;
            reqs[b].seed = ctx->seed + (uint32_t)b;
            reqs[b].greedy_warmup = ctx->greedy_warmup;
        }
        float **outs = (float **)calloc(B, sizeof(float *));
        int *outn = (int *)calloc(B, sizeof(int));
        double t0 = 0;
        int rc = qwen_tts_generate_batch_multi(ctx, reqs, B, outs, outn);
        if (rc != 0) {
            fprintf(stderr, "batch-multi-test: engine returned %d\n", rc);
        } else {
            for (int b = 0; b < B; b++) {
                char fn[64]; snprintf(fn, sizeof(fn), "bm_%d.wav", b);
                if (outs[b] && outn[b] > 0) {
                    qwen_tts_write_wav(fn, outs[b], outn[b], QWEN_TTS_SAMPLE_RATE);
                    fprintf(stderr, "  slot %d (seed %u): %d samples -> %s\n",
                            b, reqs[b].seed, outn[b], fn);
                } else {
                    fprintf(stderr, "  slot %d (seed %u): EMPTY\n", b, reqs[b].seed);
                }
                free(outs[b]);
            }
        }
        (void)t0;
        free(outs); free(outn); free(reqs);
        qwen_tts_unload(ctx);
        return rc;
    }

    /* --batch-bench: measure batched-compute throughput vs single-stream, then exit. */
    if (run_batch_bench) {
        int rc = qwen_batch_bench(ctx);
        qwen_tts_unload(ctx);
        return rc;
    }

    /* Server mode: start HTTP server and block */
    if (serve_port > 0) {
        int ret;
        if (serve_batch >= 2)
            ret = qwen_tts_serve_batched(ctx, serve_port, serve_batch);  /* vLLM-style request batching */
        else
            ret = qwen_tts_serve_ex(ctx, serve_port, serve_workers);
        qwen_tts_unload(ctx);
        return ret;
    }

    /* Batch mode (long-form): split text into sentence-packed chunks, synth each,
     * concatenate. Milestone A = sequential synth (correct audio + baseline);
     * Milestone B swaps the inner loop for batched compute. */
    if (batch_mode && text) {
        if (!silent) fprintf(stderr, "Batch mode: long-form chunked synthesis...\n");
        int rc = run_batch(ctx, text, batch_words, batch_dry, language, compose_pause, output, silent);
        qwen_tts_unload(ctx);
        return rc == 0 ? 0 : 1;
    }

    /* Compose mode: multi-span synthesis into a single WAV (no streaming). */
    if (compose_spec) {
        if (!silent) fprintf(stderr, "Compose mode: rendering spans...\n");
        int rc = run_compose(ctx, compose_spec, language, compose_pause, output, silent);
        qwen_tts_unload(ctx);
        return rc == 0 ? 0 : 1;
    }

    /* Streaming setup */
    stream_state_t stream_state = {0};
    stream_state.volume = audio_volume;   /* --volume applies per chunk while streaming */
    ctx->stream = do_stream;
    ctx->stream_chunk_frames = stream_chunk;
    if (do_stream && audio_rate != 1.0f && !silent)
        fprintf(stderr, "Warning: --rate has no effect in --stream mode (time-stretch needs the full buffer)\n");

    if (do_stream) {
        if (do_stdout) {
            /* Raw s16le 24kHz mono PCM to stdout */
            stream_state.file = stdout;
            stream_state.is_stdout = 1;
            /* Force silent mode — all status goes to stderr, audio to stdout */
            silent = 1;
            ctx->silent = 1;
        } else {
            /* Streaming WAV: write header now, update at end */
            stream_state.file = fopen(output, "wb");
            if (!stream_state.file) {
                fprintf(stderr, "Error: cannot open %s for writing\n", output);
                qwen_tts_unload(ctx);
                return 1;
            }
            write_wav_header(stream_state.file, QWEN_TTS_SAMPLE_RATE);
        }
        qwen_tts_set_audio_callback(ctx, stream_audio_callback, &stream_state);
        if (!silent)
            fprintf(stderr, "Streaming: chunk=%d frames (%.1fs), %s\n",
                    stream_chunk, stream_chunk / 12.5f,
                    do_stdout ? "raw PCM to stdout" : output);
    }

    /* Generate */
    float *audio = NULL;
    int n_samples = 0;

    if (!silent) fprintf(stderr, "Starting generation...\n");
    if (audition_keep && (seed_audition <= 1 || do_stream) && !silent)
        fprintf(stderr, "Warning: --audition-keep has no effect without --seed-audition N (N>1, non-stream)\n");
    if (seed_audition > 1 && !do_stream) {
        /* Best-of-N seed audition: render N seeds SEQUENTIALLY (one process), keep the cleanest
         * take. Rejects degenerate metallic tails via the glitch score and truncation/runaway via
         * duration deviation from the median. The seed is the only entropy source, so different
         * seeds realize different valid renderings — this picks a clean+complete one. */
        uint32_t base_seed = ctx->seed;
        float *cand_a[64]; int cand_n[64]; uint32_t cand_s[64]; float cand_g[64];
        int N = seed_audition > 64 ? 64 : seed_audition, got = 0;
        for (int i = 0; i < N; i++) {
            ctx->seed = base_seed + (uint32_t)i;
            float *a = NULL; int an = 0;
            if (qwen_tts_generate(ctx, text, &a, &an) == 0 && a && an > 0) {
                cand_a[got] = a; cand_n[got] = an; cand_s[got] = ctx->seed;
                cand_g[got] = qwen_audio_tail_glitch_score(a, an, QWEN_TTS_SAMPLE_RATE, NULL);
                if (!silent) fprintf(stderr, "  audition seed %u: %.2fs glitch=%.2f\n",
                                     ctx->seed, (float)an / QWEN_TTS_SAMPLE_RATE, cand_g[got]);
                if (audition_keep) {
                    /* --audition-keep: save EVERY take as <out>.seed<seed>.wav so the user can
                     * browse the palette and pick by ear (the glitch+duration pick is only a guess). */
                    char keep_path[1100];
                    const char *dot = strrchr(output, '.');
                    int stem = dot ? (int)(dot - output) : (int)strlen(output);
                    snprintf(keep_path, sizeof(keep_path), "%.*s.seed%u.wav", stem, output, ctx->seed);
                    if (qwen_tts_write_wav(keep_path, a, an, QWEN_TTS_SAMPLE_RATE) == 0) {
                        if (!silent) fprintf(stderr, "    kept -> %s\n", keep_path);
                    } else {
                        fprintf(stderr, "    Warning: could not write audition take %s\n", keep_path);
                    }
                }
                got++;
            } else if (a) free(a);
        }
        if (got == 0) { fprintf(stderr, "Generation failed\n"); qwen_tts_unload(ctx); return 1; }
        int tmp[64]; for (int i = 0; i < got; i++) tmp[i] = cand_n[i];
        for (int a = 1; a < got; a++) { int v = tmp[a], b = a-1; while (b>=0 && tmp[b]>v){tmp[b+1]=tmp[b];b--;} tmp[b+1]=v; }
        int med_n = tmp[got/2];
        /* "Realistically OK" filter for high-weight/high-temp breakage: a take much SHORTER than the
         * median is a truncated noise-then-stop (the "1s of noise and done" case); an EXTREMELY longer
         * one is a runaway. Hard-penalize those (a survivor always wins; if all break we still return
         * the least-bad). We do NOT reject merely-long takes — those are the expressive variants
         * (e.g. a longer "menacing" anger) — the glitch score already catches the metallic runaway tail. */
        int n_rej = 0;
        int best = 0; float best_cost = 1e30f;
        for (int i = 0; i < got; i++) {
            float r  = med_n > 0 ? (float)cand_n[i] / (float)med_n : 1.0f;
            float dd = r - 1.0f; if (dd < 0) dd = -dd;
            int   bad = (r < 0.55f || r > 2.50f         /* truncated-short or extreme-runaway */
                         || cand_g[i] >= 0.5f);          /* sustained metallic/noise tail */
            if (bad) n_rej++;
            float cost = cand_g[i] * 10.0f + dd + (bad ? 100.0f : 0.0f);
            if (cost < best_cost) { best_cost = cost; best = i; }
        }
        if (!silent) fprintf(stderr, "  audition -> picked seed %u (glitch=%.2f, %.2fs of %d takes; %d rejected as broken)\n",
                             cand_s[best], cand_g[best], (float)cand_n[best] / QWEN_TTS_SAMPLE_RATE, got, n_rej);
        audio = cand_a[best]; n_samples = cand_n[best]; ctx->seed = cand_s[best];
        for (int i = 0; i < got; i++) if (i != best) free(cand_a[i]);
    } else if (qwen_tts_generate(ctx, text, &audio, &n_samples) != 0) {
        fprintf(stderr, "Generation failed\n");
        if (do_stream && !do_stdout && stream_state.file) fclose(stream_state.file);
        qwen_tts_unload(ctx);
        return 1;
    }

    if (do_stream) {
        /* Finalize streaming output */
        if (!do_stdout && stream_state.file) {
            finalize_wav_header(stream_state.file, stream_state.total_samples);
            fclose(stream_state.file);
            if (!silent)
                fprintf(stderr, "Wrote %s (%d samples, %.2fs) [streamed]\n",
                        output, stream_state.total_samples,
                        (float)stream_state.total_samples / QWEN_TTS_SAMPLE_RATE);
        }
        /* Free the full decode output (streaming already wrote everything) */
        free(audio);
    } else {
        /* Non-streaming: write WAV from full decode, with optional rate/volume post. */
        if (audio && n_samples > 0) {
            float *final_audio = audio;
            int    final_n     = n_samples;
            float *stretched   = NULL;
            if (audio_rate != 1.0f) {
                int sn = 0;
                if (qwen_audio_time_stretch(audio, n_samples, audio_rate,
                                            QWEN_TTS_SAMPLE_RATE, &stretched, &sn) == 0) {
                    final_audio = stretched; final_n = sn;
                    if (!silent) fprintf(stderr, "Rate: %.2fx (%d -> %d samples)\n",
                                         audio_rate, n_samples, sn);
                } else if (!silent) {
                    fprintf(stderr, "Warning: time-stretch failed, writing at original rate\n");
                }
            }
            if (audio_volume != 1.0f) {
                qwen_audio_apply_gain(final_audio, final_n, audio_volume);
                if (!silent) fprintf(stderr, "Volume: %.2fx\n", audio_volume);
            }
            /* Edge cleanup (default-off; golden bit-identical unless opted in). */
            if (tail_trim) {
                int cut = qwen_audio_tail_trim(final_audio, &final_n, QWEN_TTS_SAMPLE_RATE, 0.30f);
                if (cut > 0 && !silent)
                    fprintf(stderr, "Tail-trim: cut %.2fs degenerate tail (%d samples)\n",
                            (float)cut / QWEN_TTS_SAMPLE_RATE, cut);
            }
            if (onset_fade_ms > 0) {
                qwen_audio_onset_fade(final_audio, final_n, QWEN_TTS_SAMPLE_RATE, onset_fade_ms);
                if (!silent) fprintf(stderr, "Onset-fade: %d ms over the real attack\n", onset_fade_ms);
            }
            if (qwen_tts_write_wav(output, final_audio, final_n, QWEN_TTS_SAMPLE_RATE) == 0) {
                if (!silent)
                    fprintf(stderr, "Wrote %s (%d samples, %.2fs)\n", output, final_n,
                            (float)final_n / QWEN_TTS_SAMPLE_RATE);
            } else {
                fprintf(stderr, "Failed to write WAV\n");
            }
            free(stretched);
            free(audio);
        }
    }

    qwen_tts_unload(ctx);
    return 0;
}
