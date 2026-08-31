/* decoder_batch_parity.c - does the cross-slot batched streaming decoder match the reference */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

#include "qwen_tts.h"

extern int qwen_speech_decoder_decode_streaming_st(qwen_tts_ctx_t *ctx,
                                                   qwen_sd_stream_state_t *st,
                                                   const int *new_codes, int new_frames,
                                                   float **audio_out, int *n_samples);
extern int qwen_speech_decoder_decode_streaming_batch(qwen_tts_ctx_t *ctx,
                                                      qwen_sd_batch_item_t *items,
                                                      int n_items);
extern void qwen_sd_stream_init(qwen_sd_stream_state_t *st);
extern void qwen_sd_stream_free(qwen_sd_stream_state_t *st);

#define NCB 16
#define MAX_STREAMS 8

static uint32_t rng_state = 0x9E3779B9u;
static uint32_t rng_next(void) {
    rng_state ^= rng_state << 13; rng_state ^= rng_state >> 17; rng_state ^= rng_state << 5;
    return rng_state;
}

static int chunk_for(int s, int r) {
    static const int pat_a[6] = {1, 2, 0, 4, 3, 8};
    static const int pat_b[6] = {2, 3, 0, 4, 5, 8};
    const char *e = getenv("QWEN_PARITY_PAT");
    const int *pat = (e && *e == '2') ? pat_b : pat_a;
    return pat[(s * 2 + r * 3) % 6];
}

static void audio_append(float **buf, int *n, int *cap, const float *src, int cnt) {
    if (cnt <= 0) return;
    if (*n + cnt > *cap) {
        *cap = (*n + cnt) * 2;
        *buf = (float *)realloc(*buf, (size_t)(*cap) * sizeof(float));
    }
    memcpy(*buf + *n, src, (size_t)cnt * sizeof(float));
    *n += cnt;
}

int main(int argc, char **argv) {
    const char *model_dir = (argc > 1) ? argv[1] : "qwen3-tts-0.6b";
    int n_streams = (argc > 2) ? atoi(argv[2]) : 4;
    int n_rounds  = (argc > 3) ? atoi(argv[3]) : 6;
    const char *codes_file = (argc > 4) ? argv[4] : NULL;
    if (n_streams < 2) n_streams = 2;
    if (n_streams > MAX_STREAMS) n_streams = MAX_STREAMS;
    if (n_rounds < 1) n_rounds = 1;

    int total_frames[MAX_STREAMS];
    int max_total = 0;
    for (int s = 0; s < n_streams; s++) {
        total_frames[s] = 0;
        for (int r = 0; r < n_rounds; r++) total_frames[s] += chunk_for(s, r);
        if (total_frames[s] > max_total) max_total = total_frames[s];
    }

    fprintf(stderr, "[parity] loading %s ...\n", model_dir);
    qwen_tts_ctx_t *ctx = qwen_tts_load(model_dir);
    if (!ctx) { fprintf(stderr, "[parity] model load failed (%s)\n", model_dir); return 1; }
    int cbsz = ctx->config.codebook_size;
    if (cbsz <= 1) cbsz = 1024;

    int *codes[MAX_STREAMS];
    for (int s = 0; s < n_streams; s++) {
        codes[s] = (int *)malloc((size_t)max_total * NCB * sizeof(int));
        if (!codes[s]) { fprintf(stderr, "[parity] OOM\n"); return 1; }
    }

    if (codes_file) {
        FILE *f = fopen(codes_file, "r");
        if (!f) { fprintf(stderr, "[parity] cannot open %s\n", codes_file); return 1; }
        int nread = 0;
        int *flat = (int *)malloc((size_t)max_total * n_streams * NCB * sizeof(int));
        while (nread < max_total * n_streams) {
            int got = 0, *row = flat + (size_t)nread * NCB;
            while (got < NCB && fscanf(f, "%d", &row[got]) == 1) got++;
            if (got < NCB) break;
            nread++;
        }
        fclose(f);
        if (nread < max_total) {
            fprintf(stderr, "[parity] %s has %d frames, need %d — falling back to synthetic\n",
                    codes_file, nread, max_total);
            codes_file = NULL;
            free(flat);
        } else {
            for (int s = 0; s < n_streams; s++)
                for (int f2 = 0; f2 < max_total; f2++)
                    memcpy(codes[s] + (size_t)f2 * NCB,
                           flat + (size_t)((f2 + s * 7) % nread) * NCB,
                           NCB * sizeof(int));
            free(flat);
        }
    }
    if (!codes_file) {
        for (int s = 0; s < n_streams; s++)
            for (int f2 = 0; f2 < max_total * NCB; f2++)
                codes[s][f2] = (int)(rng_next() % (uint32_t)cbsz);
    }

    qwen_sd_stream_state_t st_a[MAX_STREAMS], st_b[MAX_STREAMS];
    float *aud_a[MAX_STREAMS], *aud_b[MAX_STREAMS];
    int na[MAX_STREAMS], ca[MAX_STREAMS], nb_[MAX_STREAMS], cb_[MAX_STREAMS];
    int pos_a[MAX_STREAMS], pos_b[MAX_STREAMS];
    for (int s = 0; s < n_streams; s++) {
        qwen_sd_stream_init(&st_a[s]); qwen_sd_stream_init(&st_b[s]);
        aud_a[s] = NULL; aud_b[s] = NULL;
        na[s] = ca[s] = nb_[s] = cb_[s] = 0;
        pos_a[s] = pos_b[s] = 0;
    }

    fprintf(stderr, "=== ARM A (per-slot) ===\n");
    for (int r = 0; r < n_rounds; r++) {
        for (int s = 0; s < n_streams; s++) {
            int m = chunk_for(s, r);
            if (m <= 0) continue;
            float *a = NULL; int an = 0;
            if (qwen_speech_decoder_decode_streaming_st(ctx, &st_a[s],
                    codes[s] + (size_t)pos_a[s] * NCB, m, &a, &an) != 0) {
                fprintf(stderr, "[parity] per-slot decode failed (stream %d round %d)\n", s, r);
                return 1;
            }
            pos_a[s] += m;
            audio_append(&aud_a[s], &na[s], &ca[s], a, an);
            free(a);
        }
    }

    fprintf(stderr, "=== ARM B (batched) ===\n");
    int batched_rounds = 0, max_bsize = 0;
    for (int r = 0; r < n_rounds; r++) {
        qwen_sd_batch_item_t items[MAX_STREAMS];
        int live = 0;
        for (int s = 0; s < n_streams; s++) {
            int m = chunk_for(s, r);
            items[s].st = &st_b[s];
            items[s].codes = codes[s] + (size_t)pos_b[s] * NCB;
            items[s].nframes = m;
            items[s].audio = NULL; items[s].n_samples = 0; items[s].rc = 0;
            if (m > 0) live++;
        }
        if (live > 1) { batched_rounds++; if (live > max_bsize) max_bsize = live; }
        if (qwen_speech_decoder_decode_streaming_batch(ctx, items, n_streams) != 0) {
            fprintf(stderr, "[parity] batched decode failed (round %d)\n", r);
            return 1;
        }
        for (int s = 0; s < n_streams; s++) {
            pos_b[s] += items[s].nframes;
            audio_append(&aud_b[s], &nb_[s], &cb_[s], items[s].audio, items[s].n_samples);
            free(items[s].audio);
        }
    }

    int fail = 0;
    double worst = 0.0;

    {
        double lworst = 0.0;
        for (int s = 0; s < n_streams; s++) {
            int live = st_a[s].latent_frames - st_a[s].latent_base;
            int liveb = st_b[s].latent_frames - st_b[s].latent_base;
            int nl = live < liveb ? live : liveb;
            for (int64_t i = 0; i < (int64_t)nl * 1024; i++) {
                double d = fabs((double)st_a[s].latent_cache[i] - (double)st_b[s].latent_cache[i]);
                if (d > lworst) lworst = d;
            }
        }
        printf("\n[parity] pre-transformer latent (steps 1-5) worst abs diff: %.6e\n", lworst);
    }
    printf("\n[parity] model=%s streams=%d rounds=%d (batched rounds: %d, max batch %d)\n",
           model_dir, n_streams, n_rounds, batched_rounds, max_bsize);
    printf("  %-8s %10s %10s %14s %14s\n", "stream", "frames", "samples", "max_abs_diff", "rms_diff");
    for (int s = 0; s < n_streams; s++) {
        if (na[s] != nb_[s]) {
            printf("  %-8d %10d %10d %14s   SAMPLE COUNT MISMATCH (per-slot %d, batched %d)\n",
                   s, total_frames[s], na[s], "-", na[s], nb_[s]);
            fail = 1;
            continue;
        }
        double mx = 0.0, sq = 0.0;
        for (int i = 0; i < na[s]; i++) {
            double d = fabs((double)aud_a[s][i] - (double)aud_b[s][i]);
            if (d > mx) mx = d;
            sq += d * d;
        }
        double rms = na[s] > 0 ? sqrt(sq / na[s]) : 0.0;
        if (mx > worst) worst = mx;
        printf("  %-8d %10d %10d %14.3e %14.3e\n", s, total_frames[s], na[s], mx, rms);
        if (mx > 1e-4) fail = 1;
    }
    printf("\n[parity] worst max_abs_diff over all streams: %.6e  %s\n", worst,
           worst == 0.0 ? "(EXACT - bit-identical)"
                        : "(nonzero - if the schedule contains a 1-frame chunk this is the\n"
                          "            BLAS GEMV-vs-GEMM reduction order; re-run with"
                          " QWEN_PARITY_PAT=2 to check)");
    printf("[parity] %s\n", fail ? "FAIL" : "PASS");

    for (int s = 0; s < n_streams; s++) {
        qwen_sd_stream_free(&st_a[s]); qwen_sd_stream_free(&st_b[s]);
        free(aud_a[s]); free(aud_b[s]); free(codes[s]);
    }
    qwen_tts_unload(ctx);
    return fail ? 1 : 0;
}
