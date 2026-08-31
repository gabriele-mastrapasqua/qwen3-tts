/* decode_quantum_bench.c — how long does ONE batched speech-decode call hold the */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

#include "qwen_tts.h"
#include "qwen_tts_kernels.h"

extern int qwen_speech_decoder_decode_streaming_batch(qwen_tts_ctx_t *ctx,
                                                      qwen_sd_batch_item_t *items,
                                                      int n_items);
extern void qwen_sd_stream_init(qwen_sd_stream_state_t *st);
extern void qwen_sd_stream_free(qwen_sd_stream_state_t *st);

#define NCB          16
#define MAX_STREAMS  8
#define MAX_FRAMES   768
#define MAX_REPS     64

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

static int cmp_d(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}
static double pctl(double *v, int n, double q) {
    if (n <= 0) return 0.0;
    double *s = (double *)malloc((size_t)n * sizeof(double));
    memcpy(s, v, (size_t)n * sizeof(double));
    qsort(s, (size_t)n, sizeof(double), cmp_d);
    double k = (n - 1) * q / 100.0;
    int lo = (int)k, hi = lo + 1 < n ? lo + 1 : n - 1;
    double r = s[lo] + (s[hi] - s[lo]) * (k - lo);
    free(s);
    return r;
}

static uint32_t rng_state = 0x9E3779B9u;
static uint32_t rng_next(void) {
    rng_state ^= rng_state << 13; rng_state ^= rng_state >> 17; rng_state ^= rng_state << 5;
    return rng_state;
}

static double run_cell(qwen_tts_ctx_t *ctx, int group, int chunk, int reps,
                       int **codes, double *p95_out, double *max_out, double *cold_out)
{
    qwen_sd_stream_state_t st[MAX_STREAMS];
    qwen_sd_batch_item_t   items[MAX_STREAMS];
    double t[MAX_REPS];
    int pos[MAX_STREAMS];

    for (int s = 0; s < group; s++) { qwen_sd_stream_init(&st[s]); pos[s] = 0; }

    const int warm = 4;
    for (int r = 0; r < warm + reps; r++) {
        for (int s = 0; s < group; s++) {
            items[s].st       = &st[s];
            items[s].codes    = codes[s] + (size_t)pos[s] * NCB;
            items[s].nframes  = chunk;
            items[s].audio    = NULL;
            items[s].n_samples = 0;
            items[s].rc       = 0;
        }
        double t0 = now_ms();
        qwen_speech_decoder_decode_streaming_batch(ctx, items, group);
        double dt = now_ms() - t0;
        for (int s = 0; s < group; s++) {
            free(items[s].audio);
            pos[s] += chunk;
            if (pos[s] + chunk > MAX_FRAMES) pos[s] = 0;
        }
        if (r == 0) *cold_out = dt;
        if (r >= warm) t[r - warm] = dt;
    }
    for (int s = 0; s < group; s++) qwen_sd_stream_free(&st[s]);

    *p95_out = pctl(t, reps, 95);
    *max_out = pctl(t, reps, 100);
    return pctl(t, reps, 50);
}

int main(int argc, char **argv) {
    const char *model_dir = (argc > 1) ? argv[1] : "qwen3-tts-0.6b";
    int threads = (argc > 2) ? atoi(argv[2]) : 8;
    int reps    = (argc > 3) ? atoi(argv[3]) : 15;
    if (reps < 3) reps = 3;
    if (reps > MAX_REPS) reps = MAX_REPS;

    fprintf(stderr, "[quantum] loading %s ...\n", model_dir);
    qwen_tts_ctx_t *ctx = qwen_tts_load(model_dir);
    if (!ctx) { fprintf(stderr, "[quantum] model load failed (%s)\n", model_dir); return 1; }
    qwen_set_threads(threads);

    int cbsz = ctx->config.codebook_size;
    if (cbsz <= 1) cbsz = 1024;
    int *codes[MAX_STREAMS];
    for (int s = 0; s < MAX_STREAMS; s++) {
        codes[s] = (int *)malloc((size_t)MAX_FRAMES * NCB * sizeof(int));
        if (!codes[s]) { fprintf(stderr, "[quantum] OOM\n"); return 1; }
        for (int i = 0; i < MAX_FRAMES * NCB; i++)
            codes[s][i] = (int)(rng_next() % (uint32_t)cbsz);
    }

    const int groups[] = {1, 2, 3, 4};
    const int chunks[] = {1, 2, 3, 4, 6, 8, 12, 16};
    const int NG = (int)(sizeof groups / sizeof groups[0]);
    const int NC = (int)(sizeof chunks / sizeof chunks[0]);
    double p50[8][16], p95g[8][16], mx[8][16], cold[8][16];

    printf("\n===============================================================================\n");
    printf("DECODE CALL COST — one call of the SAME entry point the driver uses\n");
    printf("model= %s  threads= %d  warm reps= %d  state= WARM (4 warm-up rounds dropped)\n",
           model_dir, threads, reps);
    printf("frame = 80 ms of audio at 12.5 Hz\n");
    printf("===============================================================================\n");

    for (int gi = 0; gi < NG; gi++) {
        for (int ci = 0; ci < NC; ci++) {
            p50[gi][ci] = run_cell(ctx, groups[gi], chunks[ci], reps, codes,
                                   &p95g[gi][ci], &mx[gi][ci], &cold[gi][ci]);
            fprintf(stderr, "  cell g=%d chunk=%2d -> %7.2f ms\n",
                    groups[gi], chunks[ci], p50[gi][ci]);
        }
    }

    printf("\n-- A. CALL WALL TIME, ms (p50 of %d warm calls)\n", reps);
    printf("   %-14s", "group \\ chunk");
    for (int ci = 0; ci < NC; ci++) printf("%8d", chunks[ci]);
    printf("\n");
    for (int gi = 0; gi < NG; gi++) {
        printf("   %-14d", groups[gi]);
        for (int ci = 0; ci < NC; ci++) printf("%8.1f", p50[gi][ci]);
        printf("\n");
    }

    printf("\n-- B. COST PER FRAME, ms  (call / (group x chunk))  <- the efficiency curve\n");
    printf("   %-14s", "group \\ chunk");
    for (int ci = 0; ci < NC; ci++) printf("%8d", chunks[ci]);
    printf("\n");
    for (int gi = 0; gi < NG; gi++) {
        printf("   %-14d", groups[gi]);
        for (int ci = 0; ci < NC; ci++)
            printf("%8.2f", p50[gi][ci] / (double)(groups[gi] * chunks[ci]));
        printf("\n");
    }

    printf("\n-- C. THE SPLIT QUESTION: same work, fewer frames per call\n");
    printf("   For each group, decoding 8 frames per slot as ONE call vs as N smaller\n");
    printf("   calls. 'compute' is the total decoder time for the same 8 frames; 'block'\n");
    printf("   is the longest single stretch the driver cannot be interrupted for.\n\n");
    printf("   %-6s %-10s %10s %10s %10s %10s\n",
           "group", "split", "calls", "block ms", "compute ms", "vs 1x8");
    for (int gi = 0; gi < NG; gi++) {
        int base_ci = -1;
        for (int ci = 0; ci < NC; ci++) if (chunks[ci] == 8) base_ci = ci;
        double base = p50[gi][base_ci];
        const int splits[][2] = {{1,8},{2,4},{4,2},{8,1}};
        for (int si = 0; si < 4; si++) {
            int n = splits[si][0], f = splits[si][1];
            int ci = -1;
            for (int c = 0; c < NC; c++) if (chunks[c] == f) ci = c;
            if (ci < 0) continue;
            double compute = n * p50[gi][ci];
            printf("   %-6d %-10s %10d %10.1f %10.1f %+9.1f%%\n",
                   groups[gi], si == 0 ? "1 x 8" : (si == 1 ? "2 x 4" : (si == 2 ? "4 x 2" : "8 x 1")),
                   n, p50[gi][ci], compute, (compute / base - 1.0) * 100.0);
        }
        printf("\n");
    }

    printf("-- D. COLD (first) call vs warm, ms — the ramp's first chunk lives here\n");
    printf("   %-14s", "group \\ chunk");
    for (int ci = 0; ci < NC; ci++) printf("%8d", chunks[ci]);
    printf("\n");
    for (int gi = 0; gi < NG; gi++) {
        printf("   %-14d", groups[gi]);
        for (int ci = 0; ci < NC; ci++) printf("%8.1f", cold[gi][ci]);
        printf("\n");
    }

    printf("\n-- E. SPREAD of the warm calls (p95 / max), group=1\n");
    for (int ci = 0; ci < NC; ci++)
        printf("   chunk %2d   p50 %7.2f   p95 %7.2f   max %7.2f\n",
               chunks[ci], p50[0][ci], p95g[0][ci], mx[0][ci]);

    for (int s = 0; s < MAX_STREAMS; s++) free(codes[s]);
    qwen_tts_unload(ctx);
    return 0;
}
